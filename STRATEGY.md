# Strategy — GTA VI launch-day Twitch forecaster

## The problem

Forecast, one hour into GTA VI's release day (19 Nov 2026, UTC), how many
people will be watching the game's Twitch category in each of hours 2–24,
as five quantiles (5/25/50/75/95%). Score is mean pinball loss on the log
scale; lower is better. The house baseline (every hour stays at hour 1's
count, spread widening like a random walk) scores ~0.387 on the practice.

## What the data says

The kit's practice cases carry 38 comparable past launches with full
17-day hourly curves. Pooling them shows the launch-day shape is NOT a
random walk around hour 1:

- Hour 1 (00:00–01:00 UTC) only captures regions already unlocked
  (New Zealand/Australia/Asia) plus the UK just unlocking. The US east
  coast unlocks at 05:00 UTC, the west coast at 08:00 UTC.
- Across launches, the median ratio to hour 1 dips to ~0.75 in hours 3–5
  (the dead of night in most of the world), then climbs to ~2.4x by
  hours 19–20 (US daytime + European evening), settling ~2.0x at hour 24.
- The spread is enormous in ratio terms: small launches (hour 1 = a few
  hundred viewers) grow 100–2000x over the day; huge launches (hour 1 =
  500k–875k) barely move, peaking ~1.0–1.3x. The correlation between
  log(hour-1) and the log-ratio turns strongly negative by midday.
- Regressing log viewers of hour h on log(hour-1): slope ~1.0 at hours
  2–6 (persistence works), collapsing to ~0.06–0.1 by hours 15–24
  (hour 1 says almost nothing about the evening). Meanwhile the
  unconditional spread of log viewers tightens over the day (sd 2.2 at
  hour 2 → 0.8 at hours 19–24): launch days converge to a similar
  absolute level by evening, whatever hour 1 said.

So the right model is a local (scale-dependent) regression, not
persistence: early hours anchor on hour 1, late hours shrink hard
toward the pooled launch-day level.

## The forecaster (main.py)

Fitted at run time from the case's own comparables, so the same program
runs on the practice cases and on the final:

1. **Anchor.** `a = log(max(p1, eve1))`, p1 = hour-1 print, eve1 =
   viewers of the hour just before the release day (from
   `history_hourly`). The game is already live in the earliest
   timezones by then, so the eve hour backs up a noisy single-point
   hour-1 print. Using the max (not the mean) won in CV; longer eve
   windows and a second eve regressor both lost.
2. **Local fit.** For each hour h, Gaussian-kernel weighted OLS of
   log viewers on the anchor across the comparable launches with
   complete release days (bandwidth 1.25 in log units). The kernel is
   the whole trick: it follows the launches at the target's own scale
   instead of being dragged by the small ones.
3. **Quantiles.** Median from the fitted regression; spread from the
   weighted residual sd × 0.9 on the normal quantiles (−1.645, −0.674,
   0, 0.674, 1.645). Empirical residual quantiles were tested and lost
   (too noisy with ~36 launches); the 0.9 spread factor was tuned on CV.

Features tested and rejected in CV: pre-release 7-day / 24-hour median
viewers, hour-1 channel count, two-feature fits, unconditional pooled
quantiles (good late, terrible early), longer eve windows.

## Validation

Leave-one-out CV over the 37 complete comparable launch days, scoring
the shipped `main.py` end-to-end (pseudo-case per launch, comparables
minus the target, exactly as the grader runs it):

| | ALL (37) | big, p1 ≥ 20k (27) | huge, p1 ≥ 100k (16) |
|---|---|---|---|
| this entry | **0.381** | **0.252** | **0.204** |
| house baseline | 0.930 | 0.321 | 0.261 |

Format checks: 23 rows, 5 non-decreasing quantiles each, all in
(0, 1e9], outputs ≤ 65,536 bytes; two runs byte-identical
(deterministic); ~0.11 s per case, far under the 8 s cap.

## Limits

- The practice is 4 launches; the final is 1 launch (GTA VI). A
  practice score is a proxy, not a promise — the final's hour-1 print
  may sit outside the comparable range (GTA VI could open above the
  largest comparable, 875k). The local fit extrapolates gracefully
  there (late-hour slopes ~0.1 pull the forecast toward the pooled
  evening level), but it is still extrapolation.
- `streams` / `categories_hourly` (only present in the final case) are
  deliberately unused: they are null in every practice case, so no
  use of them could be validated.
