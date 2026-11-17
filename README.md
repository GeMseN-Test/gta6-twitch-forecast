# GTA VI Launch-Day Twitch Forecaster — make.fun bounty `gta6-twitch-ccu-forecast-01`

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/GeMseN-Test/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/GeMseN-Test/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for the [make.fun "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) ($250 in USDC).

A forecaster for Twitch concurrent viewers in the Grand Theft Auto VI
category on release day (19 November 2026, UTC), written for the make.fun
"Forecast GTA VI's launch-day Twitch viewers hour by hour" competition.
Python 3.13, standard library only, no network at runtime.

## Run

The entrypoint is `main.py` at the repository root. The grader runs it as:

```sh
python3 -P -s /bot/main.py
```

with `PYTHONHASHSEED=0`. It reads every `<id>.in` case in `/input`
(`$MAKEFUN_INPUT_DIR` outside the grader) and writes `<id>.out` to
`/output` (`$MAKEFUN_OUTPUT_DIR`): 23 rows, hours 2–24, five quantiles
(5%, 25%, 50%, 75%, 95%) each.

## Method (see STRATEGY.md for the full write-up)

Rolling-midnight launch-day forecaster, fitted at run time from the
case's own comparables:

1. **Anchor** — `log(max(hour-1 print, eve-hour viewers))`. The game is
   already live in the earliest timezones before UTC midnight, so the
   eve hour backs up a noisy single-point hour-1 print.
2. **Local fit** — for each hour, a Gaussian-kernel weighted regression
   of log viewers on the anchor across comparable launches (bandwidth
   1.25). Small launches explode hundreds-fold over the day while huge
   launches barely move, so a global fit is dragged by the small ones;
   the local fit follows the launches at the target's own scale.
3. **Quantiles** — median from the fit, spread from the weighted
   residual sd × 0.9 on the normal quantiles.

Leave-one-out cross-validation over the 37 complete comparable launch
days: pinball loss **0.381** overall (**0.252** on big launches with
hour-1 ≥ 20k, **0.204** on huge launches with hour-1 ≥ 100k), against
0.930 / 0.321 / 0.261 for the house baseline.
