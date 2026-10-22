import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from tests.helpers import EPOCH, book
from updown.clock import Interval
from updown.domain import Book, CycleState, Fill, Intent, Market, Side, Snapshot
from updown.engine import Engine, Venue
from updown.execution import PaperExecutor
from updown.feeds.binance import Candle, CandleSeries
from updown.risk import Limits, RiskGate
from updown.sinks import Collector, FillRecord, Ledger, SettleRecord, Sink
from updown.venue.gamma import GammaMarket

TOKENS = json.dumps(["token-up", "token-down"])


class FakeClock:
    def __init__(self, now: float) -> None:
        self._now = now

    def now(self) -> float:
        return self._now

    async def sleep_until(self, ts: float) -> None:
        self._now = max(self._now, ts)
        await asyncio.sleep(0)


class FakeMarkets:
    """An open market until the cycle ends, then resolved Down."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock

    async def market(self, asset: str, interval: Interval, epoch: int) -> GammaMarket:
        closed = self._clock.now() >= epoch + interval.seconds
        return GammaMarket(
            slug=f"{asset}-updown-{interval.label}-{epoch}",
            condition_id="0xcondition",
            outcomes='["Up", "Down"]',
            outcome_prices='["0", "1"]' if closed else '["0.5", "0.5"]',
            clob_token_ids=TOKENS,
            closed=closed,
        )


class FakeBooks:
    async def book(self, token_id: str) -> Book:
        if token_id == "token-up":
            return book(bids=[(0.69, 100)], asks=[(0.71, 100)])
        return book(bids=[(0.29, 100)], asks=[(0.31, 100)])


class FakeReference:
    async def minutes(self, now: int, lookback: int = 62) -> CandleSeries:
        start = now - lookback * 60
        return CandleSeries([Candle((start + 60 * i) * 1000, 1.0, 1.0) for i in range(lookback)])

    async def price_at(self, ts: int) -> float | None:
        return 100_000.0


class BuyUnderdogAt840:
    name = "test"

    def decide(self, state: CycleState) -> Intent | None:
        if state.t != 840:
            return None
        return Intent(state.last.favorite.other, limit=0.35, stake=5.0, rule="test")


class Broken(Sink):
    async def on_snapshot(
        self, market: Market, snapshot: Snapshot, books: Mapping[Side, Book]
    ) -> None:
        raise RuntimeError("sink down")

    async def on_fill(self, market: Market, fill: Fill) -> None:
        raise RuntimeError("sink down")

    async def on_resolution(
        self, market: Market, winner: Side | None, fills: Sequence[Fill]
    ) -> None:
        raise RuntimeError("sink down")


def run_one_cycle(tmp_path: Path, limits: Limits, *extra: Sink) -> Ledger:
    clock = FakeClock(EPOCH - 30)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    engine = Engine(
        venue=Venue(markets=FakeMarkets(clock), books=FakeBooks(), reference=FakeReference()),
        strategy=BuyUnderdogAt840(),
        risk=RiskGate(limits),
        executor=PaperExecutor(),
        sinks=[*extra, ledger, Collector(tmp_path / "live")],
        clock=clock,
    )
    asyncio.run(engine.run(cycles=1))
    return ledger


def test_one_cycle_trades_settles_and_records(tmp_path: Path) -> None:
    fill, settle = run_one_cycle(tmp_path, Limits()).read()
    assert isinstance(fill, FillRecord)
    assert isinstance(settle, SettleRecord)
    assert fill.side == "Down"
    assert settle.slug == f"btc-updown-15m-{EPOCH}"
    assert settle.winner == "Down"
    assert settle.pnl is not None
    assert settle.pnl > 0
    assert list((tmp_path / "live" / "snapshots").glob("*.parquet"))
    assert list((tmp_path / "live" / "cycles").glob("*.parquet"))


def test_risk_rejection_sends_nothing(tmp_path: Path) -> None:
    assert run_one_cycle(tmp_path, Limits(max_stake=1.0)).read() == []


def test_a_failing_sink_does_not_stop_trading(tmp_path: Path) -> None:
    records = run_one_cycle(tmp_path, Limits(), Broken()).read()
    assert [type(r) for r in records] == [FillRecord, SettleRecord]
