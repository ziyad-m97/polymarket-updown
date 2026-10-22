"""Build the research dataset from public data.

    cycles.parquet     one row per resolved 15-minute BTC market
    snapshots.parquet  the Up price at every minute t = 60..840, with the BTC reference

Polymarket's price history has about one point a minute, and a point reports the mid from about
`HISTORY_LAG_S` seconds before its timestamp (measured against the original bot's live midpoint
samples; the report reproduces the measurement). Each grid time takes the last point stamped at
or before it, never after. The BTC reference is read from Binance's one-second archive at the
point's observation time, the stamp minus the lag. `ref_stamped` keeps the value at the stamp
itself: it is ~20 s in the future of the price, and exists to show what that leak does.

Markets and price points come from Polymarket's APIs; the reference only needs Binance's
archive. Responses and archive days are cached, which makes rebuilds cheap and exact.
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
from updown.research.dataset import HISTORY_LAG_S
from updown.venue.clob import Clob
from updown.venue.gamma import BTC_15M_SERIES, Gamma, to_market, winner

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
        "stamp_age_s": pl.Int16(),
        "mid_up": pl.Float64(),
        "ref": pl.Float64(),
        "ref_stamped": pl.Float64(),
        "vol_1m": pl.Float64(),
    }
)


@dataclass(frozen=True, slots=True)
class MarketMeta:
    epoch: int
    winner: str
    fee_rate: float
    fee_exponent: float
    twap_s: int


@dataclass(frozen=True, slots=True)
class Point:
    epoch: int
    t: int
    ts: int
    """The price point's timestamp."""
    mid_up: float

    @property
    def observed(self) -> int:
        return self.ts - HISTORY_LAG_S


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


async def markets_and_points(
    fetcher: Fetcher, start: int, end: int
) -> tuple[list[MarketMeta], list[Point]]:
    """Resolved markets with epoch in [start, end) and their sampled price points."""
    margin = timedelta(days=2)
    since = datetime.fromtimestamp(start, UTC) - margin
    until = datetime.fromtimestamp(end, UTC) + margin
    found: dict[int, tuple[MarketMeta, str]] = {}
    unresolved = 0
    async for raw in Gamma(fetcher).series(BTC_15M_SERIES, since, until):
        if not raw.slug.startswith(PREFIX) or raw.clob_token_ids is None:
            continue
        epoch = epoch_of(raw.slug)
        if not start <= epoch < end:
            continue
        side = winner(raw)
        if side is None:
            unresolved += 1
            continue
        market = to_market(raw, Interval.M15)
        meta = MarketMeta(epoch, str(side), market.fees.rate, market.fees.exponent, market.twap_s)
        found[epoch] = (meta, market.token_up)
    log.info("%d resolved markets, %d skipped as unresolved", len(found), unresolved)

    clob, done = Clob(fetcher), 0

    async def one(epoch: int, token: str) -> list[Point]:
        nonlocal done
        history = await clob.history(token, epoch, epoch + 900, cache=True)
        done += 1
        if done % 2000 == 0:
            log.info("price history %d/%d", done, len(found))
        return sample(epoch, history)

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(one(e, found[e][1])) for e in sorted(found)]
    points = [p for task in tasks for p in task.result()]
    return [found[e][0] for e in sorted(found)], points


async def reference(
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


def needed_seconds(markets: Sequence[MarketMeta], points: Sequence[Point]) -> set[int]:
    needed = {p.ts for p in points} | {p.observed for p in points}
    return needed | {m.epoch for m in markets} | {m.epoch + 900 for m in markets}


def assemble(
    markets: Sequence[MarketMeta],
    points: Sequence[Point],
    prices: dict[int, float | None],
    minutes: CandleSeries,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    cycles = [
        {
            "epoch": m.epoch,
            "winner": m.winner,
            "fee_rate": m.fee_rate,
            "fee_exponent": m.fee_exponent,
            "twap_s": m.twap_s,
            "ref_open": prices.get(m.epoch),
            "ref_close": prices.get(m.epoch + 900),
        }
        for m in markets
    ]
    snaps = [
        {
            "epoch": p.epoch,
            "t": p.t,
            "stamp_age_s": p.epoch + p.t - p.ts,
            "mid_up": p.mid_up,
            "ref": prices.get(p.observed),
            "ref_stamped": prices.get(p.ts),
            "vol_1m": minutes.vol_1m(p.observed),
        }
        for p in points
    ]
    return (
        pl.DataFrame(cycles, schema=CYCLES).sort("epoch"),
        pl.DataFrame(snaps, schema=SNAPSHOTS).sort("epoch", "t"),
    )


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(60.0), headers={"User-Agent": USER_AGENT})


async def fetch(
    start: date, end: date, cache: Path, concurrency: int = 8
) -> tuple[pl.DataFrame, pl.DataFrame]:
    async with _client() as http:
        fetcher = Fetcher(http, concurrency=concurrency, cache_dir=cache / "http")
        markets, points = await markets_and_points(fetcher, _ts(start), _ts(end))
        prices, minutes = await reference(
            fetcher, needed_seconds(markets, points), cache / "binance"
        )
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
