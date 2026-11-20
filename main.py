"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every PS5 trophy of Grand Theft Auto VI, five quantiles of its
earn rate on Exophase at the case's target time. Run once per case:
reads the case file at $MAKEFUN_INPUT_DIR/<id>.in (else /input) and writes
$MAKEFUN_OUTPUT_DIR/<id>.out (else /output). Python 3.13, standard library
only, no network, fully deterministic (two runs write identical bytes).

Model: damped recent-trend forecaster.
---------------------------------------
An Exophase earn rate is earners / players-tracked, and both sides move:

* dilution: as the tracked population grows (launch week brings a flood of
  new players who have earned almost nothing yet), a trophy's rate is
  pulled down, even while its earner count grows;
* progression: as the tracked players play on, they earn trophies, which
  pulls rates back up, fastest for early, easy content.

So per trophy we work on the logit scale and fit two one-regressor trends,
then keep the one that fits better:

  A: logit(rate) ~ a + b * ln(1 + hours since release)
  B: logit(rate) ~ a + b * ln(players tracked)

Model B is the dilution view: if the rate is really driven by who has joined
the tracked population, it tracks ln(N). Model A is the clock view: steady
progress with time. The final case's history (launch night through the
deadline, reads every 10-30 minutes) has real variation in both, so the fit
can tell them apart; on a flat history (the practice cases, where every read
is the same pre-refresh page) both fits are degenerate and we fall back to
persistence.

Three choices keep the extrapolation honest, and all three were checked on
synthetic launch-week cases (a saturating rise, a dilution-dominated fall,
and a late-start rise) against the house baseline's persistence:

* recency: the fit uses only the last 24 hours of reads. Launch night's
  explosive first hours do not describe the regime a week out; the recent
  slope does. (If the last 24 hours hold too few reads to fit, the full
  history is used instead.)
* damping: only half of the log-distance from the last read to the target
  is extrapolated, and the median's move from the last rate is capped at
  0.75 logit units. A straight line in logit space overshoots every
  saturating curve; the trend is a nudge, not a forecast of its own.
* shrinkage: the median blends the trend with the last read's rate, with
  weight w = R^2 / (1 + leverage), capped at 0.85, where leverage measures
  how far the target sits outside the history's range.

The spread is the house baseline's random-walk term (0.5 * sqrt(horizon in
days)) combined with the fit's residual spread, floored at 0.45 and capped
at 1.3 logit units. The one trophy that starts at ~100% ("Complete Gone
Fishing", earned by essentially every tracked player from the first mission
on) is forecast to stay there with a narrow spread.

What this does NOT do: it never fetches anything (the grader has no
network), never hard-codes a rate for a trophy, and never uses a number from
outside the case file. Every parameter above is a modelling choice, fixed
before seeing any case; the rates, the population and the trends all come
from the case's own history.

Sources used while designing the model (see README.md): the kit's RULES.txt
and input/NOTICE.txt; Exophase's public trophy page for the game; Sensor
Tower launch-day sales estimates reported by Reuters / TweakTown
(~19-24.5m copies day one, ~75% PlayStation), which say the player flood
behind the dilution effect is real but say nothing about Exophase's own
tracked population, which the program estimates from the case history.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Random-walk spread of the house baseline, logit units per sqrt day.
SIGMA_RW = 0.5
# Game release, 2026-11-19 00:00 UTC (case game.release_date).
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Fallback median (percent) when a case has no usable history at all.
TIER_PRIOR = {"platinum": 0.03, "gold": 0.05, "silver": 0.1, "bronze": 0.5}


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    c = clamp(pct)
    return math.log(c / (100.0 - c))


def sigmoid(l: float) -> float:
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l)) if l < 700 else 100.0
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as the starter rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ols(xs: list[float], ys: list[float]) -> dict | None:
    """Least squares y ~ a + b*x. None when x has no variation."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {
        "a": a,
        "b": b,
        "s_res": math.sqrt(ss_res / max(n - 2, 1)),
        "r2": max(0.0, min(1.0, r2)),
        "mx": mx,
        "sxx": sxx,
        "xmin": min(xs),
        "xmax": max(xs),
        "n": n,
    }


def _trend(points: list[dict], t_target: float) -> tuple | None:
    """Fit the log-linear trends on the given points; return
    (median_logit, residual_std) or None when no fit is possible."""
    if len(points) < 3:
        return None
    xs_time = [math.log1p(p["t"]) for p in points]
    xs_n = [math.log(p["n"]) for p in points]
    ys = [logit(p["r"]) for p in points]
    fit_a = ols(xs_time, ys)
    fit_b = ols(xs_n, ys)
    fits = [f for f in (fit_a, fit_b) if f is not None]
    if not fits:
        return None
    # Best-fitting trend; on a tie prefer the clock view.
    fits.sort(key=lambda f: (f["s_res"], 0 if f is fit_a else 1))
    fit = fits[0]
    if fit is fit_b:
        ln_n_fit = ols(xs_time, xs_n)
        if ln_n_fit is None:
            return None
        n_last = points[-1]["n"]
        n_target = math.exp(ln_n_fit["a"] + ln_n_fit["b"] * math.log1p(t_target))
        n_target = min(max(n_target, n_last), n_last * 30.0)
        x_target = math.log(n_target)
        x_last = xs_n[-1]
    else:
        x_target = math.log1p(t_target)
        x_last = xs_time[-1]
    # Damped extrapolation: only half the log-distance past the last read.
    x_eff = x_last + (x_target - x_last) * 0.5
    l_trend = fit["a"] + fit["b"] * x_eff
    leverage = (x_eff - fit["mx"]) ** 2 / fit["sxx"]
    w = max(0.0, min(0.85, fit["r2"] / (1.0 + leverage)))
    l_last = logit(points[-1]["r"])
    l_med = w * l_trend + (1.0 - w) * l_last
    l_med = min(max(l_med, l_last - 0.75), l_last + 0.75)
    return l_med, fit["s_res"]


def forecast_trophy(trophy: dict, points: list[dict], horizon_days: float,
                    t_target: float) -> list:
    """points: [{t (hours since release), n (players tracked), r (rate %)}]
    oldest first. Returns [q05, q25, q50, q75, q95] in percent."""
    tier = trophy.get("tier", "bronze")
    if not points:
        med = TIER_PRIOR.get(tier, 0.5)
        return quantiles(logit(med), 1.5)
    r_last = points[-1]["r"]
    # The ~100% trophy (first mission) stays ~100%: nearly every tracked
    # player earns it within minutes of starting, dilution cannot move it.
    if r_last >= 99.5:
        return quantiles(logit(min(r_last, 99.99)), 0.25)
    base_spread = SIGMA_RW * math.sqrt(horizon_days)
    if len(points) < 3:
        return quantiles(logit(r_last), max(base_spread * 1.2, 0.45))
    # Recent window first: launch night's explosive phase does not describe
    # the regime a week out. Fall back to the full history if it won't fit.
    t_max = points[-1]["t"]
    recent = [p for p in points if p["t"] >= t_max - 24.0]
    trend = _trend(recent, t_target) if len(recent) >= 3 else None
    if trend is None:
        trend = _trend(points, t_target)
    if trend is None:
        # Flat history (practice cases): persistence, slightly widened,
        # because a refresh the history never saw can move every rate.
        return quantiles(logit(r_last), max(base_spread * 1.5, 0.45))
    l_med, s_res = trend
    spread = math.sqrt(base_spread ** 2 + s_res ** 2)
    spread = min(max(spread, 0.45), 1.3)
    return quantiles(l_med, spread)


def quantiles(l_med: float, spread: float) -> list:
    qs = []
    prev = 0.01
    for z in Z:
        q = clamp(round4(sigmoid(l_med + z * spread)))
        q = max(q, prev)  # keep the row non-decreasing after rounding
        qs.append(q)
        prev = q
    return qs


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = sorted(case.get("history", []), key=lambda r: r.get("at", ""))
    release = RELEASE
    try:
        release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    except Exception:
        pass
    t_target = (parse_time(case["target"]) - release).total_seconds() / 3600.0
    horizon_days = case["horizon_minutes"] / 1440.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        points = []
        for read in history:
            rates = read.get("rates") or []
            n = read.get("players_tracked")
            if i >= len(rates) or rates[i] is None or not n:
                continue
            try:
                t = (parse_time(read["at"]) - release).total_seconds() / 3600.0
            except Exception:
                continue
            points.append({"t": max(t, 0.0), "n": float(n), "r": float(rates[i])})
        rows.append({"id": trophy["id"], "q": forecast_trophy(trophy, points, horizon_days, t_target)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
