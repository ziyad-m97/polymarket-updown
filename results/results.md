### Data

24,383 resolved 15-minute BTC markets from 2026-02-10 to 2026-10-21, with 339,885 minute snapshots. A snapshot's price is Polymarket's last price-history point stamped at or before the minute (median 47 s earlier).

| regime | from | to | cycles | up share |
|---|---|---|---:|---:|
| development | 2026-02-10 | 2026-03-13 | 3072 | 0.507 |
| holdout, fee v1 | 2026-03-14 | 2026-03-29 | 1536 | 0.508 |
| holdout, fee v2 | 2026-03-30 | 2026-08-06 | 12479 | 0.496 |
| holdout, TWAP resolution | 2026-08-07 | 2026-10-21 | 7296 | 0.501 |

Those stamps are late. The original bot logged the live order-book mid on the minute; a history point equals that mid most often when it is stamped 10 to 20 s afterwards, and less and less as the gap grows:

| stamped after | n | equal share | mean gap |
|---|---:|---:|---:|
| 11-20 s | 1609 | 0.772 | 0.0038 |
| 21-30 s | 5082 | 0.698 | 0.0057 |
| 31-40 s | 1154 | 0.533 | 0.0162 |
| 41-50 s | 190 | 0.321 | 0.0421 |
| 51-60 s | 38 | 0.421 | 0.0284 |

So a history point shows the mid from about 20 s before its stamp, and the BTC reference is read at that observation time. Read at the stamp instead, it would hand the model 20 s of the future; the last table under the rules below shows what that does. Binance's open-to-close direction matches the official Chainlink result in 97.1% of cycles, so Binance feeds the model and never the labels.

### Are the prices calibrated?

Brier score (lower is better) of the market price and of the Brownian model at each minute, and the market's calibration slope from a logistic fit of the outcome on logit(price). A slope above 1 means prices are too timid: favourites win more often than they are priced to.

| t | n | brier_market | brier_model | slope | intercept |
|---:|---:|---:|---:|---:|---:|
| 60 | 23725 | 0.2474 | 0.2493 | 0.971 | -0.002 |
| 180 | 24314 | 0.2216 | 0.2251 | 1.064 | +0.000 |
| 300 | 24324 | 0.2004 | 0.2034 | 1.060 | -0.007 |
| 420 | 24322 | 0.1778 | 0.1813 | 1.059 | -0.006 |
| 540 | 24314 | 0.1512 | 0.1557 | 1.067 | -0.003 |
| 660 | 24319 | 0.1229 | 0.1285 | 1.055 | -0.003 |
| 780 | 24325 | 0.0927 | 0.0992 | 1.021 | -0.018 |
| 840 | 24327 | 0.0725 | 0.0807 | 1.026 | -0.025 |

Realised frequency of Up by price bucket at minute 14, with 95% Wilson intervals:

| bin | n | price | freq | lo | hi |
|---|---:|---:|---:|---:|---:|
| [0.00, 0.05) | 6833 | 0.013 | 0.011 | 0.009 | 0.014 |
| [0.05, 0.10) | 1324 | 0.072 | 0.066 | 0.054 | 0.081 |
| [0.10, 0.20) | 1409 | 0.147 | 0.135 | 0.118 | 0.154 |
| [0.20, 0.30) | 1008 | 0.248 | 0.248 | 0.222 | 0.276 |
| [0.30, 0.40) | 794 | 0.348 | 0.338 | 0.305 | 0.371 |
| [0.40, 0.50) | 774 | 0.451 | 0.439 | 0.405 | 0.474 |
| [0.50, 0.60) | 793 | 0.546 | 0.547 | 0.512 | 0.582 |
| [0.60, 0.70) | 794 | 0.650 | 0.661 | 0.628 | 0.693 |
| [0.70, 0.80) | 935 | 0.753 | 0.754 | 0.725 | 0.781 |
| [0.80, 0.90) | 1377 | 0.855 | 0.843 | 0.823 | 0.861 |
| [0.90, 0.95) | 1336 | 0.928 | 0.929 | 0.914 | 0.941 |
| [0.95, 1.00) | 6950 | 0.987 | 0.989 | 0.986 | 0.991 |

![Calibration at minute 14](results/reliability_m14.png)

Does the model know something the price does not? A logistic blend of logit(price) and logit(model), fitted on the development period and scored on the holdout:

| t | w_market | w_model | logloss_market | logloss_blend |
|---:|---:|---:|---:|---:|
| 300 | 0.814 | +0.220 | 0.5822 | 0.5813 |
| 600 | 0.581 | +0.445 | 0.4233 | 0.4247 |
| 840 | 0.943 | +0.096 | 0.2337 | 0.2330 |

### Do the rules survive out of sample?

Each rule is replayed through the same strategy object that trades live, buying at mid + 0.01 and paying the taker fee in force at the time. ROI is P&L over stake; the interval resamples whole UTC days; p-values test ROI <= 0 and are Holm-adjusted across the rules. The legacy rules were written during the development period; nothing after 13 March 2026 was looked at while choosing them.

| rule | dev n | dev ROI | holdout n | win rate | avg price | holdout ROI | 95% CI | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| legacy A | 169 | -0.126 | 1031 | 0.341 | 0.375 | -0.137 | [-0.211, -0.060] | 1.000 |
| legacy C1 | 162 | -0.039 | 866 | 0.300 | 0.348 | -0.183 | [-0.271, -0.089] | 1.000 |
| legacy C2 | 108 | -0.073 | 693 | 0.405 | 0.383 | +0.010 | [-0.082, +0.102] | 1.000 |
| legacy E | 455 | -0.096 | 2991 | 0.130 | 0.143 | -0.149 | [-0.226, -0.068] | 1.000 |
| fair value | 1978 | +0.035 | 15313 | 0.364 | 0.361 | -0.070 | [-0.104, -0.036] | 1.000 |

![Holdout P&L](results/holdout_pnl.png)

Holdout ROI (trades) in each regime: Polymarket changed its fee curve on 30 March and moved to a 60-second TWAP for resolution in August.

| rule | holdout, fee v1 | holdout, fee v2 | holdout, TWAP resolution |
|---|---|---|---|
| legacy A | -0.287 (66) | -0.089 (647) | -0.205 (318) |
| legacy C1 | -0.190 (67) | -0.189 (525) | -0.169 (274) |
| legacy C2 | -0.125 (59) | +0.016 (417) | +0.035 (217) |
| legacy E | -0.259 (231) | -0.237 (1797) | +0.043 (963) |
| fair value | +0.077 (1031) | -0.093 (9129) | -0.059 (5153) |

Holdout ROI by the half-spread paid on entry:

| rule | 0.000 | 0.005 | 0.010 | 0.020 |
|---|---:|---:|---:|---:|
| legacy A | -0.113 | -0.125 | -0.137 | -0.159 |
| legacy C1 | -0.158 | -0.170 | -0.183 | -0.206 |
| legacy C2 | +0.037 | +0.023 | +0.010 | -0.016 |
| legacy E | -0.081 | -0.116 | -0.149 | -0.206 |
| fair value | -0.025 | -0.054 | -0.070 | -0.095 |

The fair-value rule is the one that depends on timing. With the BTC reference read at the price point's stamp, as a naive join would do, it shows a significant edge; read when the price was actually observed, the edge is gone:

| BTC reference read | dev ROI | holdout n | holdout ROI | 95% CI | p |
|---|---:|---:|---:|---|---:|
| at the stamp (leaks ~20 s of future) | +0.209 | 14231 | +0.051 | [+0.012, +0.089] | 0.007 |
| at the observation time | +0.035 | 15313 | -0.070 | [-0.104, -0.036] | 1.000 |

### What the original bot actually did

The bot's state file books 79 live orders. They were resting limit orders at the mid, and the wallet's public trade history shows that only 22 of them (28%) ever filled. A resting buy fills when sellers come to it, which tends to be when the price is moving against the side being bought.

| orders | n | win rate | 95% CI | staked | P&L | ROI |
|---|---:|---:|---|---:|---:|---:|
| as booked by the bot (all filled at the limit) | 79 | 0.278 | [0.19, 0.39] | 373.18 | +69.01 | +0.185 |
| actually filled, at the fill price | 22 | 0.182 | [0.07, 0.39] | 103.76 | -35.59 | -0.343 |
| never filled, priced at the limit | 57 | 0.316 | [0.21, 0.44] | 261.88 | +110.71 | +0.423 |

The orders that filled won less often than the ones that did not, as adverse selection predicts, but with 22 fills the gap is not significant on its own (one-sided Fisher exact p = 0.18). What is certain is that the bot's own ledger, which counted every order as filled at its limit, did not describe its account.
