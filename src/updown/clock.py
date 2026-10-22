"""Cycle arithmetic and a clock abstraction the engine can be driven by in tests."""

import asyncio
import time
from enum import Enum
from typing import Protocol


class Interval(Enum):
    M5 = 300
    M15 = 900

    @property
    def seconds(self) -> int:
        return self.value

    @property
    def label(self) -> str:
        return f"{self.value // 60}m"


def cycle_start(ts: float, interval: Interval) -> int:
    """Start (unix seconds) of the cycle containing `ts`."""
    return int(ts) // interval.seconds * interval.seconds


def slug(asset: str, interval: Interval, epoch: int) -> str:
    if epoch % interval.seconds:
        raise ValueError(f"{epoch} is not aligned to {interval.label}")
    return f"{asset}-updown-{interval.label}-{epoch}"


def epoch_of(market_slug: str) -> int:
    return int(market_slug.rsplit("-", 1)[1])


class Clock(Protocol):
    def now(self) -> float: ...

    async def sleep_until(self, ts: float) -> None: ...


class SystemClock:
    def now(self) -> float:
        return time.time()

    async def sleep_until(self, ts: float) -> None:
        delay = ts - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
