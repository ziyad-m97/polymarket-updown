"""Binance's public daily archive of one-second klines, checksum-verified and cached as Parquet."""

import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from updown.feeds.binance import Candle
from updown.net import Fetcher

ARCHIVE = "https://data.binance.vision/data/spot/daily/klines"
_SECOND_US = 1_000_000
_SECOND_MS = 1_000
_KLINE = pl.Schema(
    {
        "open_time": pl.Int64(),
        "open": pl.Float64(),
        "high": pl.Float64(),
        "low": pl.Float64(),
        "close": pl.Float64(),
        "volume": pl.Float64(),
        "close_time": pl.Int64(),
        "quote_volume": pl.Float64(),
        "trades": pl.Int64(),
        "taker_base": pl.Float64(),
        "taker_quote": pl.Float64(),
        "ignore": pl.String(),
    }
)


class ArchiveError(RuntimeError):
    pass


async def seconds(
    fetcher: Fetcher, day: date, cache: Path, symbol: str = "BTCUSDT"
) -> pl.DataFrame:
    """One day of one-second candles: `ts` (open, unix seconds), `open`, `close`."""
    path = cache / f"{symbol}-1s-{day.isoformat()}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    url = f"{ARCHIVE}/{symbol}/1s/{symbol}-1s-{day.isoformat()}.zip"
    blob = await fetcher.get(url)
    expected = (await fetcher.get(f"{url}.CHECKSUM")).split()[0].decode()
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ArchiveError(f"checksum mismatch for {url}")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        raw = archive.read(archive.namelist()[0])
    frame = pl.read_csv(raw, has_header=False, schema=_KLINE)
    unit = _SECOND_US if frame["open_time"][0] > 10**14 else _SECOND_MS
    frame = frame.select((pl.col("open_time") // unit).alias("ts"), "open", "close").sort("ts")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return frame


class SecondSeries:
    """Point-in-time BTC price from one-second candles."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self._ts: npt.NDArray[np.int64] = frame["ts"].to_numpy()
        self._open: npt.NDArray[np.float64] = frame["open"].to_numpy()
        self._close: npt.NDArray[np.float64] = frame["close"].to_numpy()

    def price_at(self, ts: int) -> float | None:
        """Close of the one-second candle that opened at `ts - 1`: the last price before `ts`."""
        i = int(np.searchsorted(self._ts, ts - 1, side="right")) - 1
        return float(self._close[i]) if i >= 0 and self._ts[i] == ts - 1 else None

    def minutes(self) -> list[Candle]:
        """Aggregate to one-minute candles (complete minutes only)."""
        minute = self._ts // 60
        starts = np.flatnonzero(np.diff(minute, prepend=minute[0] - 1))
        ends = np.append(starts[1:], len(minute)) - 1
        full = (ends - starts) == 59
        return [
            Candle(int(minute[s]) * 60_000, float(self._open[s]), float(self._close[e]))
            for s, e in zip(starts[full], ends[full], strict=True)
        ]
