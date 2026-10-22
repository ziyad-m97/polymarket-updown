# Data

All times are unix seconds, UTC. `make data` rebuilds the first two files from public APIs; the
last two were imported once from the original bot's files and cannot be rebuilt.

## cycles.parquet

One row per resolved `btc-updown-15m-{epoch}` market.

| column | type | meaning |
|---|---|---|
| epoch | int64 | cycle start; the market resolves at `epoch + 900` |
| winner | enum(Up, Down) | official result from Gamma (Chainlink BTC/USD) |
| fee_rate, fee_exponent | float64 | taker fee curve: `fee = shares * rate * (p (1 - p)) ** exponent` |
| twap_s | int32 | 60 when resolution uses a 60-second TWAP, 0 for the closing print |
| ref_open, ref_close | float64 | Binance BTCUSDT last price before `epoch` and before `epoch + 900` |

## snapshots.parquet

The Up price at each minute `t = 60, 120, ..., 840` of each cycle. Down is `1 - mid_up` (the two
books mirror each other).

| column | type | meaning |
|---|---|---|
| epoch, t | int64, int16 | cycle and seconds since it opened |
| age_s | int16 | how long before `epoch + t` the price point was printed (at most 120) |
| mid_up | float64 | last Polymarket price-history point at or before `epoch + t` |
| ref | float64 | Binance last price before that point's own timestamp |
| vol_1m | float64 | std of the 60 one-minute log returns closed before that timestamp |

Sources: Gamma `/events` (series 10192), CLOB `/prices-history` (fidelity 1), Binance's daily
1-second kline archive (checksums verified).

## collector.parquet

Up/Down mids logged by the original bot's own collector (Feb to Mar 2026), at the minutes it
sampled. Rows whose columns were shifted by a mid-file schema change fail the `|up + down - 1| <=
0.03` check and are dropped. Used only to cross-check `snapshots.parquet`.

## live_trades.parquet

Every order the original bot logged, with the features it logged. `mode` is `live` (sent to the
exchange), `dry` (logged only) or `unlabelled` (an early revision without paths).

| column | meaning |
|---|---|
| decision_price | the mid the bot used as its limit price |
| stake | USDC the bot meant to spend |
| filled | whether the wallet's public trade history shows a buy of that side in that market; null when two logged orders share a market and side |
| fill_price | volume-weighted price of those buys |
| fill_ratio | USDC filled over stake |

The wallet address was used only to query its public history and is not stored.
