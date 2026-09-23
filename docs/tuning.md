# Parameter tuning log

## Conclusion

Both parameters are insensitive across the ranges tested (xi 0.001–0.007,
prior_sd 0.25–0.50); the apparent xi gain on the short window vanished on the
full history, so the defaults are retained and further gains must come from
model structure or better data rather than tuning.

## Method

Walk-forward backtest (`python -m src.backtest`), refitting every 7 days using
only matches played before each fixture. Scored by log loss against Pinnacle /
market-average closing odds with the overround stripped, and against a
base-rate baseline.

Sweeps were run on 2425 onward (810 matches) for speed, then the best candidate
was confirmed on the full history from 1920 (2,710 matches), which includes
seasons never used for tuning.

## Sweep: time decay (xi), prior_sd = 0.35

Short window, 2425–2627, 810 matches.

| xi     | half-life | 1X2        | O/U 2.5    |
| ------ | --------- | ---------- | ---------- |
| 0.0010 | 693 d     | 1.0205     | **0.6835** |
| 0.0018 | 385 d     | 1.0113     | 0.6854     |
| 0.0030 | 231 d     | **1.0078** | 0.6894     |
| 0.0050 | 139 d     | 1.0103     | 0.6950     |
| 0.0070 | 99 d      | 1.0149     | 0.6988     |

Market 0.9941, baseline 1.0815 (1X2); market 0.6802, baseline 0.6870 (O/U).

1X2 forms a clean U-shape with a minimum at 0.0030. Over/under degrades
monotonically as xi rises.

The two markets want opposite things: 1X2 depends on the *difference* between
team strengths, which moves with form and rewards recency; totals depend on the
*sum*, which is more stable and is better estimated over a longer window.

## Sweep: prior_sd (shrinkage), xi = 0.0018

Short window, 2425–2627, 810 matches.

| prior_sd | 1X2        | O/U 2.5    |
| -------- | ---------- | ---------- |
| 0.25     | 1.0133     | **0.6845** |
| 0.35     | 1.0113     | 0.6854     |
| 0.50     | **1.0105** | 0.6863     |

A 0.003 spread across a twofold range in shrinkage strength. No meaningful
effect — this knob is not worth further tuning.

## Confirmation on held-out seasons

Full history, 1920–2627, 2,710 matches. Seasons 1920–2324 were never used for
tuning.

| xi     | 1X2        | O/U 2.5    |
| ------ | ---------- | ---------- |
| 0.0018 | 0.9850     | **0.6809** |
| 0.0030 | **0.9847** | 0.6822     |

Market 0.9657, baseline 1.0699 (1X2); market 0.6734, baseline 0.6882 (O/U).

The 1X2 gain shrank from 0.0035 to 0.0003 — a tenth of its size on the tuning
window — while over/under got 0.0013 worse. The short-window improvement was
noise specific to 2425–2526, not a real property of the model.

**Defaults retained: xi = 0.0018, prior_sd = 0.35.**

## Baseline performance to beat

Full history, 1920–2627, 2,710 matches, at the retained defaults:

| Market  | Model  | Market | Baseline | Gap captured |
| ------- | ------ | ------ | -------- | ------------ |
| 1X2     | 0.9850 | 0.9657 | 1.0699   | 81%          |
| O/U 2.5 | 0.6809 | 0.6734 | 0.6882   | 49%          |

"Gap captured" is how far the model closes the distance from the naive baseline
to the closing line. Any future change must beat these numbers on the same
full-history backtest to count as an improvement.

## Next candidates

Tuning is exhausted; the remaining gains are structural.

- xG-based strength estimates instead of goals — likely the largest single
  upgrade, and it should help totals most, which is the weaker market
- Negative binomial marginals to handle overdispersion in match totals
- Separate decay rates per market, at the cost of the single-scoreline-matrix
  consistency that currently makes 1X2 and O/U impossible to contradict