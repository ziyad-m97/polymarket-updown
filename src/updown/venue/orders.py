"""Order placement. The engine depends on `OrderClient`; only the SDK adapter signs anything."""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from updown.config import Credentials

if TYPE_CHECKING:
    from polymarket import AsyncSecureClient
    from polymarket.models.clob.account import OpenOrder

_NOTHING_MATCHED = frozenset({"fak_not_filled", "fok_not_filled", "unmatched"})
_DELAY_POLLS = 20
_DELAY_POLL_S = 0.25


class OrderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Execution:
    order_id: str
    shares: float
    spent: float
    """USDC debited for the matched shares."""


class OrderClient(Protocol):
    async def buy(self, token_id: str, *, budget: float, max_price: float) -> Execution | None:
        """Fill-and-kill buy spending at most `budget`; None when nothing matched."""
        ...

    async def close(self) -> None: ...


class PolymarketOrders:
    """`OrderClient` backed by the official SDK (`pip install updown[live]`)."""

    def __init__(self, client: "AsyncSecureClient") -> None:
        self._client = client

    @classmethod
    async def connect(cls, creds: Credentials) -> "PolymarketOrders":
        from polymarket import AsyncSecureClient  # noqa: PLC0415 - optional dependency

        client = await AsyncSecureClient.create(
            private_key=creds.private_key.reveal(), wallet=creds.wallet
        )
        return cls(client)

    async def buy(self, token_id: str, *, budget: float, max_price: float) -> Execution | None:
        response = await self._client.place_market_order(
            token_id=token_id,
            side="BUY",
            amount=f"{budget:.2f}",
            max_price=f"{max_price:.4f}",
            order_type="FAK",
        )
        if not response.ok:
            if response.code in _NOTHING_MATCHED:
                return None
            raise OrderError(f"{response.code}: {response.message}")
        if response.status == "delayed":
            return await self._await_delayed(response.order_id)
        if response.taking_amount <= 0:
            return None
        return Execution(
            response.order_id, float(response.taking_amount), float(response.making_amount)
        )

    async def _await_delayed(self, order_id: str) -> Execution | None:
        order = await self._poll(order_id)
        matched = float(order.size_matched)
        if matched <= 0:
            return None
        spent = 0.0
        for trade_id in order.associate_trades:
            page = await self._client.list_account_trades(id=trade_id).first_page()
            spent += sum(float(t.price * t.size) for t in page.items)
        return Execution(order_id, matched, spent or matched * float(order.price))

    async def _poll(self, order_id: str) -> "OpenOrder":
        order = await self._client.get_order(order_id=order_id)
        for _ in range(_DELAY_POLLS):
            if order.status.lower() != "delayed":
                break
            await asyncio.sleep(_DELAY_POLL_S)
            order = await self._client.get_order(order_id=order_id)
        return order

    async def close(self) -> None:
        await self._client.close()
