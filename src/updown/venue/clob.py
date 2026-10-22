"""CLOB public endpoints: order books and price history."""

import msgspec

from updown.domain import Book, Level
from updown.net import Fetcher

CLOB = "https://clob.polymarket.com"


class _Level(msgspec.Struct):
    price: float
    size: float


class _Book(msgspec.Struct):
    bids: list[_Level] = msgspec.field(default_factory=list)
    asks: list[_Level] = msgspec.field(default_factory=list)


class _Point(msgspec.Struct):
    t: int
    p: float


class _History(msgspec.Struct):
    history: list[_Point] = msgspec.field(default_factory=list)


_book = msgspec.json.Decoder(_Book, strict=False)
_history = msgspec.json.Decoder(_History, strict=False)


def parse_book(body: bytes) -> Book:
    raw = _book.decode(body)
    return Book.of(
        [Level(lv.price, lv.size) for lv in raw.bids],
        [Level(lv.price, lv.size) for lv in raw.asks],
    )


class Clob:
    def __init__(self, fetcher: Fetcher, base: str = CLOB) -> None:
        self._fetch = fetcher
        self._base = base

    async def book(self, token_id: str) -> Book:
        return parse_book(await self._fetch.get(f"{self._base}/book", {"token_id": token_id}))

    async def history(
        self, token_id: str, start: int, end: int, *, cache: bool = False
    ) -> list[tuple[int, float]]:
        """One-minute price points for a token in [start, end]."""
        params: dict[str, str | int] = {
            "market": token_id,
            "startTs": start,
            "endTs": end,
            "fidelity": 1,
        }
        body = await self._fetch.get(f"{self._base}/prices-history", params, cache=cache)
        return [(pt.t, pt.p) for pt in _history.decode(body).history]
