"""Binance klines: the reference BTC price (one-second) and its volatility (one-minute).

Markets resolve on Chainlink's BTC/USD stream, not on Binance; the study measures how often the
two disagree on direction.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import msgspec
import numpy as np

from updown.net import Fetcher

BINANCE = "https://data-api.binance.vision"
MINUTE_MS = 60_000
_MAX_ROWS = 1000
_STEP_MS = {"1s": 1_000, "1m": MINUTE_MS}

_Row = tuple[int, str, str, str, str, str, int, str, int, str, str, str]
_klines = msgspec.json.Decoder(list[_Row])


@dataclass(frozen=True, slots=True)
class Candle:
    open_ms: int
    open: float
    close: float


class CandleSeries:
    """Contiguous one-minute candles with O(1) point-in-time lookups."""

    def __init__(self, candles: Sequence[Candle]) -> None:
        self._open_ms = np.fromiter(
            (c.open_ms for c in candles), dtype=np.int64, count=len(candles)
        )
        self._open = np.fromiter((c.open for c in candles), dtype=np.float64, count=len(candles))
        self._close = np.fromiter((c.close for c in candles), dtype=np.float64, count=len(candles))
        returns = np.diff(np.log(self._close), prepend=np.nan)
        returns[0] = 0.0
        self._s1 = np.concatenate(([0.0], np.cumsum(returns)))
        self._s2 = np.concatenate(([0.0], np.cumsum(returns * returns)))

    def __len__(self) -> int:
        return len(self._open_ms)

    def open_at(self, ts: int) -> float | None:
        """Open of the candle starting exactly at `ts` (unix seconds)."""
        i = int(np.searchsorted(self._open_ms, ts * 1000))
        if i < len(self) and self._open_ms[i] == ts * 1000:
            return float(self._open[i])
        return None

    def _last_closed(self, ts: int) -> int:
        """Index of the last candle closed at or before `ts`, or -1."""
        return int(np.searchsorted(self._open_ms, ts * 1000 - MINUTE_MS, side="right")) - 1

    def close_at(self, ts: int) -> float | None:
        i = self._last_closed(ts)
        return float(self._close[i]) if i >= 0 else None

    def vol_1m(self, ts: int, lookback: int = 60) -> float | None:
        """Sample std of the `lookback` one-minute log returns closed at or before `ts`."""
        end = self._last_closed(ts)
        if end < lookback:
            return None
        n = float(lookback)
        total = self._s1[end + 1] - self._s1[end + 1 - lookback]
        squares = self._s2[end + 1] - self._s2[end + 1 - lookback]
        var = max((squares - total * total / n) / (n - 1.0), 0.0)
        return float(np.sqrt(var))


class Binance:
    def __init__(self, fetcher: Fetcher, base: str = BINANCE, symbol: str = "BTCUSDT") -> None:
        self._fetch = fetcher
        self._base = base
        self._symbol = symbol

    async def candles(
        self, start: int, end: int, *, interval: str = "1m", cache: bool = False
    ) -> list[Candle]:
        """Candles opening in [start, end), unix seconds."""
        step = _STEP_MS[interval]
        out: list[Candle] = []
        cursor, stop = start * 1000, end * 1000
        while cursor < stop:
            params: dict[str, str | int] = {
                "symbol": self._symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": stop - 1,
                "limit": _MAX_ROWS,
            }
            rows = _klines.decode(
                await self._fetch.get(f"{self._base}/api/v3/klines", params, cache=cache)
            )
            if not rows:
                break
            out.extend(Candle(row[0], float(row[1]), float(row[4])) for row in rows)
            cursor = rows[-1][0] + step
        return out

    async def minutes(self, now: int, lookback: int = 62) -> CandleSeries:
        return CandleSeries(await self.candles(now - lookback * 60, now + 60))

    async def price_at(self, ts: int) -> float | None:
        """Close of the one-second candle that opened at `ts - 1`: the last price before `ts`."""
        rows = await self.candles(ts - 1, ts, interval="1s")
        return rows[-1].close if rows else None
