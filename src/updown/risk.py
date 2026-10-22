"""Pre-trade risk checks. Every intent passes through `RiskGate.check` before any executor."""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from updown.domain import Fill, Intent, Market


@dataclass(frozen=True, slots=True)
class Limits:
    max_stake: float = 10.0
    max_daily_loss: float = 50.0
    """Realised losses today plus open exposure may not exceed this."""
    max_open: int = 3
    price_floor: float = 0.02
    price_cap: float = 0.98


def _day(ts: float) -> date:
    return datetime.fromtimestamp(ts, UTC).date()


class RiskGate:
    def __init__(self, limits: Limits, kill_switch: Path | None = None) -> None:
        self.limits = limits
        self._kill_switch = kill_switch
        self._open: dict[str, float] = {}
        self._day: date | None = None
        self._realised = 0.0
        self._done_for_day = False
        self._halted: str | None = None

    @property
    def halted(self) -> str | None:
        """Why trading is stopped for good, if it is. Touching the kill-switch file stops it."""
        if self._halted is None and self._kill_switch is not None and self._kill_switch.exists():
            self._halted = f"kill switch {self._kill_switch}"
        return self._halted

    def halt(self, reason: str) -> None:
        self._halted = reason

    def check(self, intent: Intent, market: Market, now: float) -> str | None:
        """None if the intent may be sent, otherwise the reason it may not."""
        self._roll(now)
        lim = self.limits
        if (reason := self.halted) is not None:
            return f"halted: {reason}"
        if self._done_for_day:
            return f"daily loss limit {lim.max_daily_loss:.2f} reached"
        if market.slug in self._open:
            return "already positioned in this market"
        if not 0.0 < intent.stake <= lim.max_stake:
            return f"stake {intent.stake:.2f} outside (0, {lim.max_stake:.2f}]"
        if not lim.price_floor <= intent.limit <= lim.price_cap:
            return f"limit {intent.limit:.3f} outside [{lim.price_floor}, {lim.price_cap}]"
        if len(self._open) >= lim.max_open:
            return f"{len(self._open)} positions already open"
        worst = self._realised - sum(self._open.values()) - intent.stake
        if worst < -lim.max_daily_loss:
            return f"worst case for the day {worst:.2f} breaches -{lim.max_daily_loss:.2f}"
        return None

    def opened(self, fill: Fill) -> None:
        self._open[fill.slug] = fill.cost

    def settled(self, fill: Fill, pnl: float, now: float) -> None:
        self._roll(now)
        self._open.pop(fill.slug, None)
        self._realised += pnl
        if self._realised <= -self.limits.max_daily_loss:
            self._done_for_day = True

    def _roll(self, now: float) -> None:
        today = _day(now)
        if today != self._day:
            self._day, self._realised, self._done_for_day = today, 0.0, False
