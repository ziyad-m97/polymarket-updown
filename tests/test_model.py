import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from updown.domain import FeeSchedule, Snapshot
from updown.model.brownian import horizon_std, legacy_z, prob_up
from updown.model.features import lead_changes
from updown.strategy.fair_value import max_price

prices = st.floats(min_value=10_000, max_value=200_000)
vols = st.floats(min_value=1e-5, max_value=1e-2)
remaining = st.floats(min_value=1.0, max_value=900.0)


@given(open_=prices, move=st.floats(-0.01, 0.01), vol=vols, tau=remaining)
def test_prob_up_is_symmetric_in_the_move(
    open_: float, move: float, vol: float, tau: float
) -> None:
    up = prob_up(open_, open_ * math.exp(move), vol, tau)
    down = prob_up(open_, open_ * math.exp(-move), vol, tau)
    assert up + down == pytest.approx(1.0, abs=1e-9) or move == 0.0


@given(open_=prices, vol=vols, tau=remaining, a=st.floats(-0.01, 0.01), b=st.floats(-0.01, 0.01))
def test_prob_up_is_monotone_in_the_move(
    open_: float, vol: float, tau: float, a: float, b: float
) -> None:
    lo, hi = sorted((a, b))
    assert prob_up(open_, open_ * math.exp(lo), vol, tau) <= prob_up(
        open_, open_ * math.exp(hi), vol, tau
    )


def test_prob_up_tends_to_a_coin_flip_as_uncertainty_grows() -> None:
    assert prob_up(100.0, 100.1, vol_1m=10.0, remaining_s=900) == pytest.approx(0.5, abs=1e-3)


def test_prob_up_at_expiry_is_the_sign_of_the_move() -> None:
    assert prob_up(100.0, 100.0001, 1e-3, remaining_s=0) == 1.0
    assert prob_up(100.0, 99.9999, 1e-3, remaining_s=0) == 0.0


def test_twap_horizon_is_continuous_and_shrinks_variance() -> None:
    vol, window = 1e-3, 60.0
    edge_outside = horizon_std(vol, window, window)
    edge_inside = horizon_std(vol, window - 1e-9, window)
    assert edge_outside == pytest.approx(edge_inside)
    assert edge_outside == pytest.approx(vol * math.sqrt(window / 3 / 60))
    for tau in (30.0, 60.0, 300.0, 900.0):
        assert horizon_std(vol, tau, window) < horizon_std(vol, tau)


def test_legacy_z_matches_the_original_formula() -> None:
    z = legacy_z(100_000, 100_100, vol_1m=0.001, remaining_min=2.0)
    assert z == pytest.approx(0.001 / (0.001 * math.sqrt(2)))


def test_lead_changes_counts_favourite_switches() -> None:
    mids = [0.6, 0.4, 0.45, 0.55, 0.7]
    assert lead_changes([Snapshot(60 * i, m) for i, m in enumerate(mids, start=1)]) == 2
    assert lead_changes([]) == 0


def test_fair_value_limit_leaves_exactly_the_required_edge() -> None:
    fees = FeeSchedule(rate=0.07)
    limit = max_price(0.60, fees, min_edge=0.03)
    assert 0.60 - limit - fees.fee(1.0, limit) == pytest.approx(0.03, abs=1e-6)
    assert max_price(0.02, fees, min_edge=0.03) == 0.0
