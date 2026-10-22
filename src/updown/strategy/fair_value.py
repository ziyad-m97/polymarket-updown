"""Buy a side when the Brownian fair value beats its all-in cost by a margin."""

from dataclasses import dataclass

from updown.domain import CycleState, FeeSchedule, Intent, Side
from updown.model.brownian import prob_up


@dataclass(frozen=True, slots=True)
class Params:
    decision_times: tuple[int, ...] = (600, 660, 720, 780, 840)
    min_edge: float = 0.03
    """Required probability edge per share, after fees."""
    half_spread: float = 0.005
    """Expected distance from mid to ask, used to decide whether to try."""
    stake: float = 5.0


DEFAULT = Params()


def max_price(fair: float, fees: FeeSchedule, min_edge: float) -> float:
    """Highest price x with fair - x - fee(x) >= min_edge, or 0 if there is none."""
    if fair - min_edge <= 0.0:
        return 0.0
    lo, hi = 0.0, fair - min_edge
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if fair - mid - fees.fee(1.0, mid) >= min_edge:
            lo = mid
        else:
            hi = mid
    return lo


class FairValue:
    name = "fair_value"

    def __init__(self, params: Params = DEFAULT) -> None:
        self._p = params

    def decide(self, state: CycleState) -> Intent | None:
        p = self._p
        snap = state.last
        if snap.t not in p.decision_times or state.ref_open is None:
            return None
        if snap.ref is None or snap.vol_1m is None:
            return None
        market = state.market
        remaining = market.interval.seconds - snap.observed
        up = prob_up(state.ref_open, snap.ref, snap.vol_1m, remaining, market.twap_s)
        for side, fair in ((Side.UP, up), (Side.DOWN, 1.0 - up)):
            limit = max_price(fair, market.fees, p.min_edge)
            if 0.0 < snap.mid(side) + p.half_spread <= limit:
                return Intent(side=side, limit=limit, stake=p.stake, rule=self.name)
        return None
