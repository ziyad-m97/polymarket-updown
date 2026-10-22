import pytest

from tests.helpers import book
from updown.clock import Interval, cycle_start, epoch_of, slug
from updown.domain import FeeSchedule, Fill, Side, Snapshot


def test_cycle_start_floors_to_the_interval() -> None:
    assert cycle_start(1_773_000_899.9, Interval.M15) == 1_772_999_100 + 900
    assert cycle_start(1_773_000_000, Interval.M5) == 1_773_000_000


def test_slug_round_trips_and_rejects_misaligned_epochs() -> None:
    assert slug("btc", Interval.M15, 1_771_813_800) == "btc-updown-15m-1771813800"
    assert epoch_of("btc-updown-15m-1771813800") == 1_771_813_800
    with pytest.raises(ValueError, match="not aligned"):
        slug("btc", Interval.M15, 1_771_813_801)


@pytest.mark.parametrize(
    ("price", "expected"),
    [(0.50, 1.75), (0.10, 0.63), (0.30, 1.47), (0.95, 0.3325), (0.99, 0.0693)],
)
def test_fee_matches_the_published_crypto_table(price: float, expected: float) -> None:
    assert FeeSchedule(rate=0.07).fee(100, price) == pytest.approx(expected, abs=1e-4)


def test_fee_schedule_v1_peaks_at_half() -> None:
    v1 = FeeSchedule(rate=0.25, exponent=2)
    assert v1.fee(100, 0.5) == pytest.approx(1.5625)
    assert v1.fee(100, 0.2) == v1.fee(100, 0.8)


def test_book_sorts_levels_best_first() -> None:
    b = book(bids=[(0.40, 10), (0.45, 5)], asks=[(0.55, 5), (0.50, 10)])
    assert b.best_bid == 0.45
    assert b.best_ask == 0.50
    assert b.mid == pytest.approx(0.475)


def test_sweep_walks_asks_up_to_the_limit_and_budget() -> None:
    b = book(bids=[], asks=[(0.50, 4), (0.52, 100), (0.60, 100)])
    taken = b.sweep(budget=5.0, limit=0.55)
    assert taken[0] == (0.50, 4)
    assert taken[1][0] == 0.52
    assert sum(p * q for p, q in taken) == pytest.approx(5.0)
    assert b.sweep(budget=5.0, limit=0.49) == []


def test_fill_pnl_pays_one_per_share_to_the_winner() -> None:
    fill = Fill("m", Side.UP, shares=10, cost=4.1, fee=0.1, rule="r", ts=0)
    assert fill.price == pytest.approx(0.4)
    assert fill.pnl(Side.UP) == pytest.approx(5.9)
    assert fill.pnl(Side.DOWN) == pytest.approx(-4.1)


def test_snapshot_favourite_and_mirrored_down_price() -> None:
    s = Snapshot(t=60, mid_up=0.3)
    assert s.favorite is Side.DOWN
    assert s.mid(Side.DOWN) == pytest.approx(0.7)
    assert Side.UP.other is Side.DOWN
