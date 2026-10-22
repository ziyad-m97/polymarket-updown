import dataclasses

import numpy as np
import polars as pl
import pytest

from tests.helpers import market
from updown.domain import CycleState, Intent, Side, Snapshot
from updown.research.backtest import Costs, replay
from updown.research.build import MAX_AGE_S, as_of, sample
from updown.research.dataset import Cycle, DataError, Dataset, validate
from updown.research.stats import holm, logistic_fit, return_on_stake, wilson


class BuyFavouriteAt600:
    name = "test"

    def __init__(self) -> None:
        self.seen: list[tuple[int, ...]] = []

    def decide(self, state: CycleState) -> Intent | None:
        self.seen.append(tuple(s.t for s in state.snapshots))
        if state.t != 600:
            return None
        return Intent(state.last.favorite, limit=0.99, stake=10.0, rule="test")


def cycle(mids: list[float], winner: Side) -> Cycle:
    snaps = tuple(Snapshot(60 * (i + 1), m) for i, m in enumerate(mids))
    return Cycle(market(), winner, 100_000.0, snaps)


def test_replay_shows_only_the_past_and_routes_one_order_per_cycle() -> None:
    mids = [0.5 + 0.02 * i for i in range(14)]
    strategy = BuyFavouriteAt600()
    trades = replay([cycle(mids, Side.UP)], strategy, Costs(0.01))
    assert trades.height == 1
    assert strategy.seen[-1] == tuple(range(60, 660, 60))
    row = trades.row(0, named=True)
    assert row["t"] == 600
    assert row["price"] == pytest.approx(mids[9] + 0.01)
    assert row["pnl"] == pytest.approx(10.0 / row["price"] - row["cost"])


def test_replay_decisions_do_not_depend_on_the_future() -> None:
    mids = [0.5 + 0.02 * i for i in range(14)]
    future_changed = [*mids[:10], 0.01, 0.99, 0.01, 0.99]
    a = replay([cycle(mids, Side.UP)], BuyFavouriteAt600())
    b = replay([cycle(future_changed, Side.UP)], BuyFavouriteAt600())
    assert a.select("t", "side", "price").equals(b.select("t", "side", "price"))


def test_costs_respect_the_intent_limit() -> None:
    order = Intent(Side.UP, limit=0.50, stake=1.0, rule="r")
    assert Costs(0.01).price(order, 0.49) == pytest.approx(0.50)
    assert Costs(0.01).price(order, 0.495) is None


def test_as_of_never_looks_ahead() -> None:
    points = [(100, 0.4), (160, 0.5), (220, 0.6)]
    assert as_of(points, 99) is None
    assert as_of(points, 160) == (160, 0.5)
    assert as_of(points, 219) == (160, 0.5)


def test_sample_drops_stale_points() -> None:
    epoch = 1_773_000_000 - 1_773_000_000 % 900
    history = [(epoch + 30, 0.5), (epoch + 30 + MAX_AGE_S + 200, 0.6)]
    snaps = sample(epoch, history)
    assert [p.t for p in snaps] == [60, 120, 360, 420]


def test_validate_rejects_broken_data() -> None:
    cycles = pl.DataFrame({"epoch": [900, 1800], "winner": ["Up", None]})
    snaps = pl.DataFrame({"epoch": [900], "t": [60], "mid_up": [1.2]})
    with pytest.raises(DataError, match="missing winner") as err:
        validate(Dataset(cycles, snaps))
    assert "price outside" in str(err.value)


def test_wilson_interval_known_value() -> None:
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)


def test_holm_adjustment() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_return_on_stake_detects_a_real_edge_and_not_noise() -> None:
    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(200), 20)
    stake = np.ones(days.size)
    edge = np.where(rng.random(days.size) < 0.6, 1.0, -1.0)
    noise = np.where(rng.random(days.size) < 0.5, 1.0, -1.0)
    good = return_on_stake(edge, stake, days)
    assert good.lo > 0
    assert good.p_value < 0.01
    assert return_on_stake(noise, stake, days).lo < 0


def test_logistic_fit_recovers_coefficients() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=20_000)
    y = (rng.random(x.size) < 1 / (1 + np.exp(-(0.3 + 1.5 * x)))).astype(float)
    assert logistic_fit(x, y) == pytest.approx([0.3, 1.5], abs=0.06)


def test_cycle_is_immutable() -> None:
    c = cycle([0.5] * 14, Side.UP)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.winner = Side.DOWN  # type: ignore[misc]
