"""Executors turn an approved intent into a fill. Paper and live share one interface."""

import math
from typing import Protocol

from updown.domain import Book, Fill, Intent, Market
from updown.venue.orders import OrderClient


class Executor(Protocol):
    async def execute(self, market: Market, intent: Intent, book: Book, now: float) -> Fill | None:
        """Fill-or-kill against `book` (the intent side's book); None when nothing filled."""
        ...


def floor_to_tick(price: float, tick: float) -> float:
    decimals = max(0, -math.floor(math.log10(tick)))
    return round(math.floor(price / tick + 1e-9) * tick, decimals)


class PaperExecutor:
    """Sweeps the ask side up to the limit and charges the market's taker fee per level."""

    async def execute(self, market: Market, intent: Intent, book: Book, now: float) -> Fill | None:
        taken = book.sweep(intent.stake, floor_to_tick(intent.limit, market.tick))
        shares = sum(q for _, q in taken)
        if shares < market.min_shares:
            return None
        notional = sum(p * q for p, q in taken)
        fee = sum(market.fees.fee(q, p) for p, q in taken)
        return Fill(market.slug, intent.side, shares, notional + fee, fee, intent.rule, now)


class LiveExecutor:
    """Fill-and-kill orders through an `OrderClient`.

    The exchange charges the taker fee on top of the matched notional, so the fill's cost adds
    the fee implied by the market's schedule to the USDC the order spent.
    """

    def __init__(self, orders: OrderClient) -> None:
        self._orders = orders

    async def execute(self, market: Market, intent: Intent, book: Book, now: float) -> Fill | None:  # noqa: ARG002
        limit = floor_to_tick(intent.limit, market.tick)
        if limit <= 0.0 or intent.stake / limit < market.min_shares:
            return None
        done = await self._orders.buy(
            market.token(intent.side), budget=intent.stake, max_price=limit
        )
        if done is None or done.shares <= 0.0:
            return None
        fee = market.fees.fee(done.shares, done.spent / done.shares)
        return Fill(market.slug, intent.side, done.shares, done.spent + fee, fee, intent.rule, now)
