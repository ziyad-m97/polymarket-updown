"""Gamma API: market discovery and resolution."""

from collections.abc import AsyncIterator
from datetime import datetime

import msgspec

from updown.clock import Interval, epoch_of, slug
from updown.domain import NO_FEES, FeeSchedule, Market, Side
from updown.net import Fetcher

GAMMA = "https://gamma-api.polymarket.com"
BTC_15M_SERIES = "10192"
PAGE = 100


class _Fees(msgspec.Struct):
    rate: float
    exponent: float = 1.0


class _CryptoConfig(msgspec.Struct, rename="camel"):
    twap_enabled: bool = False
    twap_lookback_seconds: int = 0


class GammaMarket(msgspec.Struct, rename="camel", kw_only=True):
    slug: str
    condition_id: str
    outcomes: str
    outcome_prices: str | None = None
    clob_token_ids: str | None = None
    closed: bool = False
    order_min_size: float = 5.0
    order_price_min_tick_size: float = 0.01
    fees_enabled: bool | None = None
    fee_schedule: _Fees | None = None
    crypto_market_config: _CryptoConfig | None = None


class GammaEvent(msgspec.Struct, rename="camel", kw_only=True):
    slug: str
    start_date: datetime | None = None
    markets: list[GammaMarket] = msgspec.field(default_factory=list)


_events = msgspec.json.Decoder(list[GammaEvent], strict=False)
_strings = msgspec.json.Decoder(list[str])


def _by_outcome(m: GammaMarket, field: str) -> dict[str, str]:
    return dict(zip(_strings.decode(m.outcomes), _strings.decode(field), strict=True))


def to_market(m: GammaMarket, interval: Interval) -> Market:
    if m.clob_token_ids is None:
        raise ValueError(f"{m.slug} has no CLOB tokens")
    tokens = _by_outcome(m, m.clob_token_ids)
    cfg = m.crypto_market_config
    fees = m.fee_schedule if m.fees_enabled else None
    return Market(
        slug=m.slug,
        interval=interval,
        epoch=epoch_of(m.slug),
        condition_id=m.condition_id,
        token_up=tokens[Side.UP],
        token_down=tokens[Side.DOWN],
        tick=m.order_price_min_tick_size,
        min_shares=m.order_min_size,
        fees=FeeSchedule(fees.rate, fees.exponent) if fees else NO_FEES,
        twap_s=cfg.twap_lookback_seconds if cfg and cfg.twap_enabled else 0,
    )


def winner(m: GammaMarket) -> Side | None:
    """The resolved side, or None while the market is open or the result is not final."""
    if not m.closed or m.outcome_prices is None:
        return None
    prices = {k: float(v) for k, v in _by_outcome(m, m.outcome_prices).items()}
    for side in Side:
        if prices[side] == 1.0 and prices[side.other] == 0.0:
            return side
    return None


class Gamma:
    def __init__(self, fetcher: Fetcher, base: str = GAMMA) -> None:
        self._fetch = fetcher
        self._base = base

    async def market(self, asset: str, interval: Interval, epoch: int) -> GammaMarket | None:
        body = await self._fetch.get(f"{self._base}/events", {"slug": slug(asset, interval, epoch)})
        events = _events.decode(body)
        return events[0].markets[0] if events and events[0].markets else None

    async def series(
        self, series_id: str, created_after: datetime, created_before: datetime
    ) -> AsyncIterator[GammaMarket]:
        """Closed markets of a series created in the window, oldest first.

        Pages on the creation time rather than an offset, which Gamma caps.
        """
        cursor, seen = created_after, set[str]()
        while cursor < created_before:
            params: dict[str, str | int] = {
                "series_id": series_id,
                "closed": "true",
                "start_date_min": cursor.isoformat(),
                "order": "startDate",
                "ascending": "true",
                "limit": PAGE,
            }
            events = _events.decode(await self._fetch.get(f"{self._base}/events", params))
            fresh = [e for e in events if e.slug not in seen]
            for event in fresh:
                seen.add(event.slug)
                for m in event.markets:
                    yield m
            stamps = [e.start_date for e in events if e.start_date is not None]
            if len(events) < PAGE or not fresh or not stamps:
                return
            cursor = max(stamps)
