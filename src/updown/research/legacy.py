"""One-off import of the original bot's own records, which no public API can rebuild.

    collector.parquet    Up/Down mids the bot's collector logged, to cross-check the price history
    live_trades.parquet  every order the bot logged, joined with the wallet's actual fills

The wallet address is only used to query its public trade history; it is never written out.
"""

import asyncio
import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import msgspec
import polars as pl

from updown.clock import Interval
from updown.net import Fetcher
from updown.venue.gamma import Gamma

DATA_API = "https://data-api.polymarket.com"
MID_SUM_TOLERANCE = 0.03
SOURCES = {
    "data/zscore_data.csv": {
        13: ("price_up_13m", "price_down_13m"),
        14: ("price_up_14m", "price_down_14m"),
    },
    "data/zscore_data_v2.csv": {
        m: (f"price_up_m{m}", f"price_down_m{m}") for m in (3, 5, 7, 9, 11, 13, 14)
    },
}
COLLECTOR = pl.Schema(
    {
        "epoch": pl.Int64(),
        "t": pl.Int16(),
        "mid_up": pl.Float64(),
        "mid_down": pl.Float64(),
        "source": pl.String(),
    }
)


def _float(record: Sequence[str], i: int) -> float | None:
    try:
        return float(record[i])
    except (IndexError, ValueError):
        return None


def _collector_rows(path: Path, columns: dict[int, tuple[str, str]]) -> Iterator[dict[str, object]]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        index = {name: i for i, name in enumerate(next(reader))}
        for record in reader:
            if not record or not record[0].isdigit() or int(record[0]) % 900:
                continue
            for minute, (up_col, down_col) in columns.items():
                up, down = _float(record, index[up_col]), _float(record, index[down_col])
                if up is None or down is None or not (0.0 < up < 1.0 and 0.0 < down < 1.0):
                    continue
                if abs(up + down - 1.0) > MID_SUM_TOLERANCE:
                    continue
                yield {
                    "epoch": int(record[0]),
                    "t": minute * 60,
                    "mid_up": up,
                    "mid_down": down,
                    "source": path.name,
                }


def collector(root: Path) -> pl.DataFrame:
    """Rows whose columns are misaligned (the logger's schema changed mid-file) fail the
    mid-sum check and are dropped; later files win on duplicates."""
    rows = [row for name, cols in SOURCES.items() for row in _collector_rows(root / name, cols)]
    frame = pl.DataFrame(rows, schema=COLLECTOR)
    return frame.unique(["epoch", "t"], keep="last", maintain_order=True).sort("epoch", "t")


class _Trade(msgspec.Struct):
    side: str
    outcome: str
    price: float
    size: float


_trades = msgspec.json.Decoder(list[_Trade], strict=False)


@dataclass(frozen=True, slots=True)
class WalletFill:
    price: float
    usdc: float


async def wallet_fills(
    wallet: str, legs: Sequence[tuple[int, str]], concurrency: int = 4
) -> dict[tuple[int, str], WalletFill]:
    """Volume-weighted buys of `side` by `wallet` in each 15-minute market `epoch`."""
    async with httpx.AsyncClient(timeout=30.0) as http:
        fetcher = Fetcher(http, concurrency=concurrency)
        gamma = Gamma(fetcher)

        async def one(epoch: int, side: str) -> tuple[tuple[int, str], WalletFill | None]:
            market = await gamma.market("btc", Interval.M15, epoch)
            if market is None:
                return (epoch, side), None
            params: dict[str, str | int] = {
                "user": wallet,
                "market": market.condition_id,
                "limit": 500,
            }
            trades = _trades.decode(await fetcher.get(f"{DATA_API}/trades", params))
            buys = [t for t in trades if t.side == "BUY" and t.outcome == side]
            size = sum(t.size for t in buys)
            if size <= 0:
                return (epoch, side), None
            usdc = sum(t.price * t.size for t in buys)
            return (epoch, side), WalletFill(usdc / size, usdc)

        results = await asyncio.gather(*(one(e, s) for e, s in sorted(set(legs))))
    return {key: fill for key, fill in results if fill is not None}


LIVE = pl.Schema(
    {
        "epoch": pl.Int64(),
        "path": pl.String(),
        "mode": pl.String(),
        "side": pl.String(),
        "decision_price": pl.Float64(),
        "stake": pl.Float64(),
        "signals": pl.List(pl.String()),
        "z": pl.Float64(),
        "lead_changes": pl.Int16(),
        "fav_m7": pl.Float64(),
        "ud_delta": pl.Float64(),
        "conf_delta": pl.Float64(),
        "btc_mismatch": pl.Boolean(),
        "fav_ground_delta": pl.Float64(),
        "filled": pl.Boolean(),
        "fill_price": pl.Float64(),
        "fill_ratio": pl.Float64(),
    }
)


def _mode(trade: dict[str, object]) -> str:
    if trade.get("path") is None:
        return "unlabelled"
    return "dry" if trade.get("dry_run") else "live"


def live_trades(state: Path, wallet: str | None) -> pl.DataFrame:
    """Two logged orders on the same market and side cannot be told apart in the wallet's
    history; their fill columns are left null."""
    trades: list[dict[str, object]] = json.loads(state.read_text())["trades"]
    legs = [(int(str(t["epoch"])), str(t["ud_side"])) for t in trades]
    shared = {leg for leg in legs if legs.count(leg) > 1}
    fills = asyncio.run(wallet_fills(wallet, legs)) if wallet else {}
    rows = []
    for trade, leg in zip(trades, legs, strict=True):
        fill = None if leg in shared else fills.get(leg)
        known = wallet is not None and leg not in shared
        stake = float(str(trade["bet_size"]))
        rows.append(
            {
                "epoch": leg[0],
                "path": trade.get("path"),
                "mode": _mode(trade),
                "side": leg[1],
                "decision_price": trade["entry_price"],
                "stake": stake,
                "signals": trade.get("signals", []),
                "z": trade.get("z_score"),
                "lead_changes": trade.get("lead_changes"),
                "fav_m7": trade.get("fav_m7") or None,
                "ud_delta": trade.get("ud_delta"),
                "conf_delta": trade.get("conf_delta"),
                "btc_mismatch": trade.get("btc_mismatch"),
                "fav_ground_delta": trade.get("fav_ground_delta"),
                "filled": fill is not None if known else None,
                "fill_price": None if fill is None else round(fill.price, 4),
                "fill_ratio": None if fill is None else round(fill.usdc / stake, 2),
            }
        )
    return pl.DataFrame(rows, schema=LIVE).sort("epoch")
