"""Replay recorded cycles through a live `Strategy`.

Each cycle is fed to the strategy one snapshot at a time, exactly as the engine does, so a
decision can only depend on the past, and only the first intent of a cycle is routed. Decisions
do not depend on costs, so they are made once and then priced: the dataset has mids rather than
books, so a buy fills at mid + half_spread when that is within the intent's limit, and pays the
market's taker fee.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import polars as pl

from updown.domain import CycleState, Intent
from updown.research.dataset import Cycle
from updown.strategy.base import Strategy

TRADES = pl.Schema(
    {
        "epoch": pl.Int64(),
        "t": pl.Int16(),
        "rule": pl.String(),
        "side": pl.String(),
        "mid": pl.Float64(),
        "price": pl.Float64(),
        "shares": pl.Float64(),
        "cost": pl.Float64(),
        "pnl": pl.Float64(),
        "won": pl.Boolean(),
    }
)


@dataclass(frozen=True, slots=True)
class Costs:
    half_spread: float = 0.01

    def price(self, intent: Intent, mid: float) -> float | None:
        price = mid + self.half_spread
        return price if 0.0 < price <= intent.limit and price < 1.0 else None


DEFAULT_COSTS = Costs()


@dataclass(frozen=True, slots=True)
class Decision:
    cycle: Cycle
    t: int
    intent: Intent
    mid: float
    """Mid of the intent's side when it was decided."""


def decide(cycles: Iterable[Cycle], strategy: Strategy) -> list[Decision]:
    out = []
    for cycle in cycles:
        for i, snap in enumerate(cycle.snapshots, start=1):
            intent = strategy.decide(CycleState(cycle.market, cycle.ref_open, cycle.snapshots[:i]))
            if intent is not None:
                out.append(Decision(cycle, snap.t, intent, snap.mid(intent.side)))
                break
    return out


def price(decisions: Sequence[Decision], costs: Costs = DEFAULT_COSTS) -> pl.DataFrame:
    rows = []
    for d in decisions:
        fill = costs.price(d.intent, d.mid)
        if fill is None:
            continue
        shares = d.intent.stake / fill
        cost = d.intent.stake + d.cycle.market.fees.fee(shares, fill)
        won = d.intent.side is d.cycle.winner
        rows.append(
            {
                "epoch": d.cycle.market.epoch,
                "t": d.t,
                "rule": d.intent.rule,
                "side": str(d.intent.side),
                "mid": d.mid,
                "price": fill,
                "shares": shares,
                "cost": cost,
                "pnl": (shares if won else 0.0) - cost,
                "won": won,
            }
        )
    return pl.DataFrame(rows, schema=TRADES)


def replay(
    cycles: Iterable[Cycle], strategy: Strategy, costs: Costs = DEFAULT_COSTS
) -> pl.DataFrame:
    return price(decide(cycles, strategy), costs)
