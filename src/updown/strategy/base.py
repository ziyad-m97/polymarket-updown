"""A strategy maps what has been observed so far in a cycle to at most one order intent.

The engine calls `decide` after every snapshot, in time order, with only past data, and stops
asking once an intent has been routed for the cycle. The research replay drives the same objects
the same way, so a backtest exercises exactly the code that trades.
"""

from typing import Protocol

from updown.domain import CycleState, Intent


class Strategy(Protocol):
    @property
    def name(self) -> str: ...

    def decide(self, state: CycleState) -> Intent | None: ...


class Idle:
    """Never trades; used by the collector."""

    name = "idle"

    def decide(self, state: CycleState) -> Intent | None:  # noqa: ARG002
        return None
