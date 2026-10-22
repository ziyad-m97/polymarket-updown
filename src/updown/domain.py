"""Core value types shared by the live engine and the research replay."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from updown.clock import Interval


class Side(StrEnum):
    UP = "Up"
    DOWN = "Down"

    @property
    def other(self) -> "Side":
        return Side.DOWN if self is Side.UP else Side.UP


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """Taker fee curve: fee = shares * rate * (p * (1 - p)) ** exponent, in USDC."""

    rate: float
    exponent: float = 1.0

    def fee(self, shares: float, price: float) -> float:
        return round(shares * self.rate * math.pow(price * (1.0 - price), self.exponent), 5)


NO_FEES = FeeSchedule(rate=0.0)


@dataclass(frozen=True, slots=True)
class Market:
    slug: str
    interval: Interval
    epoch: int
    condition_id: str
    token_up: str
    token_down: str
    tick: float
    min_shares: float
    fees: FeeSchedule
    twap_s: int
    """Resolution compares a trailing average of this many seconds; 0 means the closing print."""

    @property
    def end(self) -> int:
        return self.epoch + self.interval.seconds

    def token(self, side: Side) -> str:
        return self.token_up if side is Side.UP else self.token_down


@dataclass(frozen=True, slots=True)
class Level:
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class Book:
    bids: tuple[Level, ...]
    """Best (highest) first."""
    asks: tuple[Level, ...]
    """Best (lowest) first."""

    @classmethod
    def of(cls, bids: Sequence[Level], asks: Sequence[Level]) -> "Book":
        return cls(
            bids=tuple(sorted(bids, key=lambda lv: -lv.price)),
            asks=tuple(sorted(asks, key=lambda lv: lv.price)),
        )

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    def sweep(self, budget: float, limit: float) -> list[tuple[float, float]]:
        """(price, shares) taken by spending `budget` USDC on asks priced at or below `limit`."""
        taken: list[tuple[float, float]] = []
        left = budget
        for level in self.asks:
            if level.price > limit or left <= 0.0:
                break
            shares = min(level.size, left / level.price)
            taken.append((level.price, shares))
            left -= shares * level.price
        return taken


@dataclass(frozen=True, slots=True)
class Snapshot:
    t: int
    """Seconds since the cycle opened."""
    mid_up: float
    ref: float | None = None
    """Reference BTC price at `t`."""
    vol_1m: float | None = None
    """Std of trailing one-minute log returns of the reference at `t`."""
    age: int = 0
    """Seconds by which the observation predates `t`: 0 live, ~35 in the price history."""

    @property
    def observed(self) -> int:
        return self.t - self.age

    @property
    def mid_down(self) -> float:
        return 1.0 - self.mid_up

    def mid(self, side: Side) -> float:
        return self.mid_up if side is Side.UP else self.mid_down

    @property
    def favorite(self) -> Side:
        return Side.UP if self.mid_up >= 0.5 else Side.DOWN


@dataclass(frozen=True, slots=True)
class CycleState:
    """Everything a strategy may look at: only data observed up to the latest snapshot."""

    market: Market
    ref_open: float | None
    snapshots: tuple[Snapshot, ...]

    @property
    def t(self) -> int:
        return self.snapshots[-1].t

    @property
    def last(self) -> Snapshot:
        return self.snapshots[-1]

    def at(self, t: int) -> Snapshot | None:
        return next((s for s in self.snapshots if s.t == t), None)


@dataclass(frozen=True, slots=True)
class Intent:
    side: Side
    limit: float
    """Worst acceptable price per share."""
    stake: float
    """USDC to spend before fees."""
    rule: str


@dataclass(frozen=True, slots=True)
class Fill:
    slug: str
    side: Side
    shares: float
    cost: float
    """USDC paid, fees included."""
    fee: float
    rule: str
    ts: float

    @property
    def price(self) -> float:
        """Average execution price, excluding fees."""
        return (self.cost - self.fee) / self.shares

    def pnl(self, winner: Side) -> float:
        return (self.shares if self.side is winner else 0.0) - self.cost
