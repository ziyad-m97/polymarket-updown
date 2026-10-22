from collections.abc import Sequence

from updown.clock import Interval
from updown.domain import Book, CycleState, FeeSchedule, Level, Market, Snapshot

EPOCH = 1_773_000_000 - 1_773_000_000 % 900
CRYPTO_V2 = FeeSchedule(0.07)


def market(
    epoch: int = EPOCH,
    fees: FeeSchedule = CRYPTO_V2,
    tick: float = 0.01,
    twap_s: int = 0,
) -> Market:
    return Market(
        slug=f"btc-updown-15m-{epoch}",
        interval=Interval.M15,
        epoch=epoch,
        condition_id="0xcondition",
        token_up="token-up",
        token_down="token-down",
        tick=tick,
        min_shares=5.0,
        fees=fees,
        twap_s=twap_s,
    )


def path(
    mids: dict[int, float], ref: float | None = None, vol: float | None = None
) -> tuple[Snapshot, ...]:
    return tuple(Snapshot(t, mid, ref, vol) for t, mid in sorted(mids.items()))


def state(
    snapshots: Sequence[Snapshot], ref_open: float | None = None, m: Market | None = None
) -> CycleState:
    return CycleState(m or market(), ref_open, tuple(snapshots))


def book(bids: Sequence[tuple[float, float]], asks: Sequence[tuple[float, float]]) -> Book:
    return Book.of([Level(p, s) for p, s in bids], [Level(p, s) for p, s in asks])
