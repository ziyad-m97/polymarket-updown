"""Build the research dataset from public data.

    cycles.parquet     one row per resolved 15-minute BTC market
    snapshots.parquet  the Up price at every minute t = 60..840, with the BTC reference

Polymarket's price history has about one point a minute. Each grid time takes the last point at
or before it, never after, and records its age. The reference BTC price is read at that point's
own second from Binance's one-second archive, so the market price and the model's input describe
the same instant. Responses and archive days are cached, which makes rebuilds cheap and exact.
"""

import asyncio
import bisect
import hashlib
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

from updown.clock import Interval, epoch_of
from updown.feeds.archive import SecondSeries, seconds
from updown.feeds.binance import Candle, CandleSeries
from updown.net import Fetcher
from updown.venue.clob import Clob
from updown.venue.gamma import BTC_15M_SERIES, Gamma, GammaMarket, to_market, winner

log = logging.getLogger(__name__)

GRID = tuple(range(60, 900, 60))
MAX_AGE_S = 120
PREFIX = "btc-updown-15m-"
USER_AGENT = "updown-research/0.1"
_DAYS_IN_FLIGHT = 8

SIDE = pl.Enum(["Up", "Down"])
CYCLES = pl.Schema(
    {
        "epoch": pl.Int64(),
        "winner": SIDE,
        "fee_rate": pl.Float64(),
        "fee_exponent": pl.Float64(),
        "twap_s": pl.Int32(),
        "ref_open": pl.Float64(),
        "ref_close": pl.Float64(),
    }
)
SNAPSHOTS = pl.Schema(
    {
        "epoch": pl.Int64(),
        "t": pl.Int16(),
        "age_s": pl.Int16(),
        "mid_up": pl.Float64(),
        "ref": pl.Float64(),
        "vol_1m": pl.Float64(),
    }
)


@dataclass(frozen=True, slots=True)
class Resolved:
    raw: GammaMarket
    epoch: int
    winner: str


@dataclass(frozen=True, slots=True)
class Point:
    epoch: int
    t: int
    ts: int
    """When the sampled price was printed."""
    mid_up: float


def as_of(points: Sequence[tuple[int, float]], ts: int) -> tuple[int, float] | None:
    """The last point at or before `ts`."""
    i = bisect.bisect_right(points, ts, key=lambda pt: pt[0])
    return points[i - 1] if i else None


def sample(epoch: int, history: Sequence[tuple[int, float]]) -> list[Point]:
    points = sorted(history)
    out = []
    for t in GRID:
        hit = as_of(points, epoch + t)
        if hit is not None and epoch + t - hit[0] <= MAX_AGE_S:
            out.append(Point(epoch, t, hit[0], hit[1]))
    return out


def _day(ts: int) -> date:
    return datetime.fromtimestamp(ts, UTC).date()


def _ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


async def _resolved(gamma: Gamma, start: int, end: int) -> list[Resolved]:
    margin = timedelta(days=2)
    since = datetime.fromtimestamp(start, UTC) - margin
    until = datetime.fromtimestamp(end, UTC) + margin
    found: dict[int, Resolved] = {}
    unresolved = 0
    async for raw in gamma.series(BTC_15M_SERIES, since, until):
        if not raw.slug.startswith(PREFIX) or raw.clob_token_ids is None:
            continue
        epoch = epoch_of(raw.slug)
        if not start <= epoch < end:
            continue
        side = winner(raw)
        if side is None:
            unresolved += 1
            continue
        found[epoch] = Resolved(raw, epoch, str(side))
    log.info("%d resolved markets, %d skipped as unresolved", len(found), unresolved)
    return [found[e] for e in sorted(found)]


async def _points(clob: Clob, markets: Sequence[Resolved]) -> list[Point]:
    done = 0

    async def one(item: Resolved) -> list[Point]:
        nonlocal done
        token = to_market(item.raw, Interval.M15).token_up
        history = await clob.history(token, item.epoch, item.epoch + 900, cache=True)
        done += 1
        if done % 2000 == 0:
            log.info("price history %d/%d", done, len(markets))
        return sample(item.epoch, history)

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(one(m)) for m in markets]
    return [p for task in tasks for p in task.result()]


async def _reference(
    fetcher: Fetcher, needed: set[int], cache: Path
) -> tuple[dict[int, float | None], CandleSeries]:
    """BTC price at each needed second, and one-minute candles spanning them."""
    by_day: dict[date, list[int]] = defaultdict(list)
    for ts in needed:
        by_day[_day(ts - 1)].append(ts)
    first, last = min(by_day), max(by_day)
    days = [first - timedelta(days=1) + timedelta(days=i) for i in range((last - first).days + 2)]
    prices: dict[int, float | None] = {}
    minutes: list[Candle] = []
    for chunk in (days[i : i + _DAYS_IN_FLIGHT] for i in range(0, len(days), _DAYS_IN_FLIGHT)):
        frames = await asyncio.gather(*(seconds(fetcher, d, cache) for d in chunk))
        for d, frame in zip(chunk, frames, strict=True):
            series = SecondSeries(frame)
            minutes.extend(series.minutes())
            for ts in by_day.get(d, ()):
                prices[ts] = series.price_at(ts)
        log.info("reference through %s", chunk[-1])
    return prices, CandleSeries(minutes)


def assemble(
    markets: Sequence[Resolved],
    points: Sequence[Point],
    prices: dict[int, float | None],
    minutes: CandleSeries,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    cycles = []
    for item in markets:
        market = to_market(item.raw, Interval.M15)
        cycles.append(
            {
                "epoch": item.epoch,
                "winner": item.winner,
                "fee_rate": market.fees.rate,
                "fee_exponent": market.fees.exponent,
                "twap_s": market.twap_s,
                "ref_open": prices.get(item.epoch),
                "ref_close": prices.get(item.epoch + 900),
            }
        )
    snaps = [
        {
            "epoch": p.epoch,
            "t": p.t,
            "age_s": p.epoch + p.t - p.ts,
            "mid_up": p.mid_up,
            "ref": prices.get(p.ts),
            "vol_1m": minutes.vol_1m(p.ts),
        }
        for p in points
    ]
    return (
        pl.DataFrame(cycles, schema=CYCLES).sort("epoch"),
        pl.DataFrame(snaps, schema=SNAPSHOTS).sort("epoch", "t"),
    )


async def fetch(
    start: date, end: date, cache: Path, concurrency: int = 8
) -> tuple[pl.DataFrame, pl.DataFrame]:
    lo, hi = _ts(start), _ts(end)
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), headers=headers) as http:
        fetcher = Fetcher(http, concurrency=concurrency, cache_dir=cache / "http")
        markets = await _resolved(Gamma(fetcher), lo, hi)
        points = await _points(Clob(fetcher), markets)
        needed = {p.ts for p in points} | {m.epoch for m in markets}
        needed |= {m.epoch + 900 for m in markets}
        prices, minutes = await _reference(fetcher, needed, cache / "binance")
    return assemble(markets, points, prices, minutes)


def write(frame: pl.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd", statistics=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build(out: Path, start: date, end: date, cache: Path, concurrency: int = 8) -> None:
    cycles, snapshots = asyncio.run(fetch(start, end, cache, concurrency))
    for name, frame in (("cycles", cycles), ("snapshots", snapshots)):
        digest = write(frame, out / f"{name}.parquet")
        log.info("%s.parquet: %d rows, sha256 %s", name, frame.height, digest)
