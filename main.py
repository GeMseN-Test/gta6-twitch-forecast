"""gta6-twitch-ccu-forecast-01 entry: rolling-midnight launch-day forecaster.

Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category for
hours 2..24 of release day (2026-11-19 UTC) as five quantiles per hour,
from the case's hour-1 print plus the launch's own eve hour.

Method (fitted at run time from the case's own comparables, so the same
program works on the practice cases and on the final):

  * Anchor: a = log(max(p1, eve1)), where p1 is hour 1's print and eve1 the
    viewers of the hour just before the release day (the game is already
    live in the earliest timezones by then, so the eve hour backs up a
    noisy single-point hour-1 print).
  * For each hour h, a Gaussian-kernel weighted regression of log viewers
    on the anchor across the comparable launches (bandwidth 1.25 in log
    units). The kernel matters: small launches explode hundreds-fold over
    the day while huge launches barely move, so a global fit is dragged
    by the small ones; the local fit follows the launches at the target's
    own scale. Early hours track hour 1 almost one-for-one; late hours
    shrink hard toward the pooled launch-day level (the fitted slope
    falls to ~0.1 by hour 19).
  * Quantiles: median from the fitted regression, spread from the
    weighted residual sd times 0.9, on the normal quantiles
    (-1.645, -0.674, 0, 0.674, 1.645). The spread factor was tuned by
    leave-one-out cross-validation over the 37 complete comparable
    launch days (CV pinball loss 0.38 overall, 0.25 on big launches
    with hour-1 >= 20k, vs 0.93 / 0.32 for the house baseline).

Standard library only, no network, deterministic: same case in, same
bytes out. Reads every <id>.in in the input directory and writes
<id>.out for each, exactly the shape the rules ask for.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Kernel bandwidth (log units) and spread multiplier, tuned by
# leave-one-out CV over the comparables (see module docstring).
BANDWIDTH = 1.25
SPREAD_MULT = 0.9
# Fall back to the house baseline when too few comparables survive.
MIN_TRAIN = 8
BASELINE_SIGMA = 0.15


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up (as JavaScript Math.round does)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def eve_viewers(case: dict) -> float | None:
    """Viewers of the hour just before the release day, if the case has it."""
    day_start = parse_ts(case["day_start"])
    target = day_start - timedelta(hours=1)
    best = None
    for row in case.get("history_hourly") or []:
        try:
            t = parse_ts(row["from"])
        except (KeyError, ValueError):
            continue
        if t == target and row.get("viewers"):
            return float(row["viewers"])
        if t < day_start and row.get("viewers"):
            if best is None or t > best[0]:
                best = (t, float(row["viewers"]))
    if best is not None:
        return best[1]
    # Fall back to the raw history points in the hour before day_start.
    pts = []
    for row in case.get("history") or []:
        try:
            t = parse_ts(row["at"])
        except (KeyError, ValueError):
            continue
        if target <= t < day_start and row.get("viewers"):
            pts.append(float(row["viewers"]))
    if pts:
        return sum(pts) / len(pts)
    return None


def release_day_curve(comp: dict) -> tuple[list[float], float] | None:
    """The comparable's 24 hourly viewer counts on its release day, plus
    its eve-hour viewers (the hour just before the release day), or None
    when the release day is incomplete."""
    try:
        day = parse_ts(comp["day_start"])
    except (KeyError, ValueError):
        return None
    by_hour: dict[int, float] = {}
    eve = None
    for row in comp.get("hourly") or []:
        try:
            t = parse_ts(row["from"])
        except (KeyError, ValueError):
            continue
        v = row.get("viewers")
        if v is None or v <= 0:
            continue
        dt = (t - day).total_seconds() / 3600
        if 0 <= dt < 24 and dt == int(dt):
            by_hour[int(dt)] = float(v)
        elif dt == -1:
            eve = float(v)
    if len(by_hour) < 24:
        return None
    return [by_hour[h] for h in range(24)], (eve if eve else by_hour[0])


def weighted_ols(xs: list[float], ys: list[float], ws: list[float]):
    """Weighted least squares y ~ a + b*x. Returns (a, b, weighted resid sd)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / max(sw - 2.0, 1.0)
    return a, b, math.sqrt(max(var, 0.0))


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    eve = eve_viewers(case)
    anchor = math.log(max(p1, eve if eve else 0.0, 1.0))

    # Training set: the case's comparables with complete release days.
    # Each launch's anchor is log(max(hour-1, eve hour)), the same anchor
    # the target uses, so training and target are on the same scale.
    train = []
    for comp in case.get("comparables") or []:
        got = release_day_curve(comp)
        if got is not None:
            curve, eve_c = got
            train.append((math.log(max(curve[0], eve_c, 1.0)), curve))

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        if len(train) >= MIN_TRAIN:
            xs = [t[0] for t in train]
            ys = [math.log(t[1][h - 1]) for t in train]
            ws = [math.exp(-0.5 * ((x - anchor) / BANDWIDTH) ** 2) for x in xs]
            fit = weighted_ols(xs, ys, ws)
            if fit is not None:
                a, b, sd = fit
                mu = a + b * anchor
                qs = [math.exp(mu + z * sd * SPREAD_MULT) for z in Z]
            else:
                qs = None
        else:
            qs = None
        if qs is None:
            # House baseline fallback: persistence with widening spread.
            s = BASELINE_SIGMA * math.sqrt(h - 1)
            qs = [p1 * math.exp(z * s) for z in Z]
        # Clamp to (0, 1e9], keep non-decreasing after rounding.
        clean: list[float | int] = []
        prev = 0.0
        for q in qs:
            v = round2(min(max(q, 0.01), 1_000_000_000.0))
            if v < prev:
                v = prev
            clean.append(v)
            prev = float(v)
        hours.append({"hour": h, "q": clean})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
