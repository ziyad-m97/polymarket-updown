"""The event loop: one pass per cycle, a snapshot every minute, at most one order per cycle.

Paper and live trading differ only in the executor. Each cycle settles in the background once
Gamma reports the result, which also feeds realised P&L back into the risk gate.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from updown.clock import Clock, Interval, SystemClock, cycle_start
from updown.domain import Book, CycleState, Fill, Market, Side, Snapshot
from updown.execution import Executor
from updown.feeds.binance import CandleSeries
from updown.risk import RiskGate
from updown.sinks import Fanout, Sink
from updown.strategy.base import Strategy
from updown.venue.gamma import GammaMarket, to_market, winner

log = logging.getLogger(__name__)

SCHEDULE = {
    Interval.M15: tuple(range(60, 900, 60)),
    Interval.M5: tuple(range(60, 300, 60)),
}
DISCOVERY_RETRY_S = 5.0
RESOLUTION_POLL_S = 30.0
RESOLUTION_TIMEOUT_S = 2 * 3600.0


class Markets(Protocol):
    async def market(self, asset: str, interval: Interval, epoch: int) -> GammaMarket | None: ...


class Books(Protocol):
    async def book(self, token_id: str) -> Book: ...


class Reference(Protocol):
    async def minutes(self, now: int, lookback: int = ...) -> CandleSeries: ...

    async def price_at(self, ts: int) -> float | None: ...


@dataclass(frozen=True, slots=True)
class Venue:
    markets: Markets
    """Gamma in production."""
    books: Books
    """The CLOB in production."""
    reference: Reference
    """Binance in production."""


class Engine:
    def __init__(
        self,
        *,
        venue: Venue,
        strategy: Strategy,
        risk: RiskGate,
        executor: Executor,
        sinks: Sequence[Sink] = (),
        clock: Clock | None = None,
        interval: Interval = Interval.M15,
        asset: str = "btc",
    ) -> None:
        self._venue = venue
        self._strategy = strategy
        self._risk = risk
        self._executor = executor
        self._sinks = Fanout(sinks)
        self._clock = clock or SystemClock()
        self._interval = interval
        self._asset = asset
        self._schedule = SCHEDULE[interval]

    async def run(self, cycles: int | None = None) -> None:
        settling: set[asyncio.Task[None]] = set()
        epoch = self._first_epoch()
        done = 0
        try:
            while cycles is None or done < cycles:
                if (reason := self._risk.halted) is not None:
                    log.warning("stopping: %s", reason)
                    break
                market = await self._discover(epoch)
                if market is None:
                    log.warning("no market for epoch %d", epoch)
                else:
                    fills = await self._trade(market)
                    task = asyncio.create_task(self._settle(market, fills))
                    settling.add(task)
                    task.add_done_callback(settling.discard)
                epoch += self._interval.seconds
                done += 1
        finally:
            await asyncio.gather(*settling)

    def _first_epoch(self) -> int:
        now = self._clock.now()
        epoch = cycle_start(now, self._interval)
        return epoch if now < epoch + self._schedule[0] else epoch + self._interval.seconds

    async def _discover(self, epoch: int) -> Market | None:
        deadline = epoch + self._schedule[0]
        while True:
            try:
                raw = await self._venue.markets.market(self._asset, self._interval, epoch)
                if raw is not None and raw.clob_token_ids is not None:
                    return to_market(raw, self._interval)
            except httpx.HTTPError as exc:
                log.warning("discovery of %d failed: %s", epoch, exc)
            if self._clock.now() + DISCOVERY_RETRY_S > deadline:
                return None
            await self._clock.sleep_until(self._clock.now() + DISCOVERY_RETRY_S)

    async def _trade(self, market: Market) -> list[Fill]:
        snapshots: list[Snapshot] = []
        ref_open: float | None = None
        routed = False
        fills: list[Fill] = []
        for t in self._schedule:
            await self._clock.sleep_until(market.epoch + t)
            try:
                snap, books, ref_open = await self._observe(market, t, ref_open)
            except httpx.HTTPError as exc:
                log.warning("%s t=%d: observation failed: %s", market.slug, t, exc)
                continue
            if snap is None:
                continue
            snapshots.append(snap)
            await self._sinks.on_snapshot(market, snap, books)
            if routed:
                continue
            intent = self._strategy.decide(CycleState(market, ref_open, tuple(snapshots)))
            if intent is None:
                continue
            routed = True
            if (reason := self._risk.check(intent, market, self._clock.now())) is not None:
                log.info("%s %s rejected: %s", market.slug, intent.rule, reason)
                await self._sinks.on_skip(market, intent, reason)
                continue
            fill = await self._executor.execute(
                market, intent, books[intent.side], self._clock.now()
            )
            if fill is None:
                await self._sinks.on_skip(market, intent, "not filled")
                continue
            self._risk.opened(fill)
            fills.append(fill)
            log.info("%s %s filled %.2f @ %.3f", market.slug, fill.rule, fill.shares, fill.price)
            await self._sinks.on_fill(market, fill)
        return fills

    async def _observe(
        self, market: Market, t: int, ref_open: float | None
    ) -> tuple[Snapshot | None, dict[Side, Book], float | None]:
        venue = self._venue
        now = market.epoch + t
        up, down, minutes, ref = await asyncio.gather(
            venue.books.book(market.token_up),
            venue.books.book(market.token_down),
            venue.reference.minutes(now),
            venue.reference.price_at(now),
        )
        books = {Side.UP: up, Side.DOWN: down}
        if ref_open is None:
            ref_open = await venue.reference.price_at(market.epoch)
        if up.mid is None:
            return None, books, ref_open
        return Snapshot(t, up.mid, ref, minutes.vol_1m(now)), books, ref_open

    async def _settle(self, market: Market, fills: list[Fill]) -> None:
        result = await self._await_result(market)
        if result is not None:
            now = self._clock.now()
            for fill in fills:
                self._risk.settled(fill, fill.pnl(result), now)
        await self._sinks.on_resolution(market, result, fills)

    async def _await_result(self, market: Market) -> Side | None:
        deadline = market.end + RESOLUTION_TIMEOUT_S
        await self._clock.sleep_until(market.end + RESOLUTION_POLL_S)
        while self._clock.now() < deadline:
            try:
                raw = await self._venue.markets.market(self._asset, self._interval, market.epoch)
            except httpx.HTTPError as exc:
                log.warning("resolution poll for %s failed: %s", market.slug, exc)
            else:
                if raw is not None and (side := winner(raw)) is not None:
                    return side
            await self._clock.sleep_until(self._clock.now() + RESOLUTION_POLL_S)
        log.warning("%s unresolved after %.0f s", market.slug, RESOLUTION_TIMEOUT_S)
        return None
