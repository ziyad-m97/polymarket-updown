import asyncio
from pathlib import Path

import pytest

from tests.helpers import EPOCH, book, market
from updown.config import ConfigError, Credentials, Secret
from updown.domain import Fill, Intent, Side
from updown.execution import LiveExecutor, PaperExecutor, floor_to_tick
from updown.risk import Limits, RiskGate
from updown.venue.orders import Execution

NOON = EPOCH + 12 * 3600


def intent(stake: float = 5.0, limit: float = 0.5) -> Intent:
    return Intent(Side.UP, limit=limit, stake=stake, rule="test")


def fill(slug: str, cost: float = 5.0) -> Fill:
    return Fill(slug, Side.UP, shares=10, cost=cost, fee=0.0, rule="test", ts=NOON)


def test_risk_accepts_a_plain_intent() -> None:
    assert RiskGate(Limits()).check(intent(), market(), NOON) is None


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [(intent(stake=11.0), "stake"), (intent(stake=0.0), "stake"), (intent(limit=0.99), "limit")],
)
def test_risk_rejects_out_of_bounds_orders(candidate: Intent, reason: str) -> None:
    rejection = RiskGate(Limits()).check(candidate, market(), NOON)
    assert rejection is not None
    assert reason in rejection


def test_risk_allows_one_position_per_market_and_caps_open_positions() -> None:
    gate = RiskGate(Limits(max_open=2))
    gate.opened(fill(market().slug))
    assert "already positioned" in (gate.check(intent(), market(), NOON) or "")
    gate.opened(fill("other"))
    assert "positions already open" in (gate.check(intent(), market(EPOCH + 900), NOON) or "")


def test_daily_loss_stops_trading_until_the_next_utc_day() -> None:
    gate = RiskGate(Limits(max_daily_loss=10.0))
    gate.opened(fill("a", cost=10.0))
    gate.settled(fill("a", cost=10.0), pnl=-10.0, now=NOON)
    assert "daily loss" in (gate.check(intent(), market(), NOON) or "")
    assert gate.check(intent(), market(), NOON + 86_400) is None


def test_worst_case_counts_open_exposure() -> None:
    gate = RiskGate(Limits(max_daily_loss=12.0))
    gate.opened(fill("a", cost=8.0))
    assert "worst case" in (gate.check(intent(stake=5.0), market(), NOON) or "")


def test_kill_switch_file_halts_everything(tmp_path: Path) -> None:
    switch = tmp_path / "KILL"
    gate = RiskGate(Limits(), kill_switch=switch)
    assert gate.check(intent(), market(), NOON) is None
    switch.touch()
    assert "kill switch" in (gate.check(intent(), market(), NOON) or "")
    switch.unlink()
    assert gate.halted is not None


def test_floor_to_tick_never_rounds_a_buy_limit_up() -> None:
    assert floor_to_tick(0.5199, 0.01) == 0.51
    assert floor_to_tick(0.52, 0.01) == 0.52
    assert floor_to_tick(0.1234, 0.001) == 0.123


def test_paper_fill_sweeps_the_book_and_pays_the_fee() -> None:
    b = book(bids=[(0.48, 100)], asks=[(0.50, 4), (0.52, 100)])
    result = asyncio.run(PaperExecutor().execute(market(), intent(stake=5.0, limit=0.55), b, NOON))
    assert result is not None
    expected_fee = market().fees.fee(4, 0.50) + market().fees.fee(3 / 0.52, 0.52)
    assert result.fee == pytest.approx(expected_fee, abs=1e-5)
    assert result.cost == pytest.approx(5.0 + result.fee)
    assert result.shares == pytest.approx(4 + 3 / 0.52)


def test_paper_fill_respects_the_limit_and_minimum_size() -> None:
    b = book(bids=[], asks=[(0.60, 100)])
    assert asyncio.run(PaperExecutor().execute(market(), intent(limit=0.55), b, NOON)) is None
    tiny = intent(stake=1.0, limit=0.61)
    assert asyncio.run(PaperExecutor().execute(market(), tiny, b, NOON)) is None


class FakeOrders:
    def __init__(self, result: Execution | None) -> None:
        self.result = result
        self.calls: list[tuple[str, float, float]] = []

    async def buy(self, token_id: str, *, budget: float, max_price: float) -> Execution | None:
        self.calls.append((token_id, budget, max_price))
        return self.result

    async def close(self) -> None:
        pass


def test_live_executor_floors_the_limit_and_books_the_fee() -> None:
    orders = FakeOrders(Execution("order-1", shares=10.0, spent=4.9))
    result = asyncio.run(
        LiveExecutor(orders).execute(market(), intent(limit=0.5299), book([], []), NOON)
    )
    assert orders.calls == [("token-up", 5.0, 0.52)]
    assert result is not None
    assert result.fee == market().fees.fee(10.0, 0.49)
    assert result.cost == pytest.approx(4.9 + result.fee)


def test_live_executor_skips_orders_below_the_minimum_size() -> None:
    orders = FakeOrders(None)
    small = intent(stake=2.0, limit=0.5)
    assert asyncio.run(LiveExecutor(orders).execute(market(), small, book([], []), NOON)) is None
    assert orders.calls == []


def test_secrets_never_print() -> None:
    secret = Secret("0xdeadbeef")
    assert "deadbeef" not in repr(secret)
    assert "deadbeef" not in str(secret)
    assert "deadbeef" not in repr(Credentials(secret, "0xwallet"))


def test_credentials_require_the_environment() -> None:
    with pytest.raises(ConfigError, match="POLYMARKET_PRIVATE_KEY"):
        Credentials.from_env({})
