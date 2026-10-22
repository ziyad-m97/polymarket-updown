"""Engine event consumers: the ledger, the collector and notifications."""

import logging
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import override

import httpx
import msgspec
import polars as pl

from updown.config import TelegramTarget
from updown.domain import Book, Fill, Intent, Market, Side, Snapshot

log = logging.getLogger(__name__)


class Sink:
    """No-op hooks; subclasses override the events they care about."""

    async def on_snapshot(
        self, market: Market, snapshot: Snapshot, books: Mapping[Side, Book]
    ) -> None:
        pass

    async def on_skip(self, market: Market, intent: Intent, reason: str) -> None:
        pass

    async def on_fill(self, market: Market, fill: Fill) -> None:
        pass

    async def on_resolution(
        self, market: Market, winner: Side | None, fills: Sequence[Fill]
    ) -> None:
        pass


async def _guarded(sink: Sink, call: Awaitable[None]) -> None:
    try:
        await call
    except Exception:
        log.exception("sink %s failed", type(sink).__name__)


class Fanout(Sink):
    """Delivers each event to every sink; a failing sink is logged and never stops trading."""

    def __init__(self, sinks: Iterable[Sink]) -> None:
        self._sinks = tuple(sinks)

    @override
    async def on_snapshot(
        self, market: Market, snapshot: Snapshot, books: Mapping[Side, Book]
    ) -> None:
        for sink in self._sinks:
            await _guarded(sink, sink.on_snapshot(market, snapshot, books))

    @override
    async def on_skip(self, market: Market, intent: Intent, reason: str) -> None:
        for sink in self._sinks:
            await _guarded(sink, sink.on_skip(market, intent, reason))

    @override
    async def on_fill(self, market: Market, fill: Fill) -> None:
        for sink in self._sinks:
            await _guarded(sink, sink.on_fill(market, fill))

    @override
    async def on_resolution(
        self, market: Market, winner: Side | None, fills: Sequence[Fill]
    ) -> None:
        for sink in self._sinks:
            await _guarded(sink, sink.on_resolution(market, winner, fills))


class FillRecord(msgspec.Struct, tag="fill"):
    slug: str
    side: str
    shares: float
    cost: float
    fee: float
    rule: str
    ts: float


class SettleRecord(msgspec.Struct, tag="settle"):
    slug: str
    side: str
    winner: str | None
    pnl: float | None


Record = FillRecord | SettleRecord


class Ledger(Sink):
    """Append-only JSONL of fills and settlements."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._encode = msgspec.json.Encoder()

    def _append(self, record: Record) -> None:
        with self._path.open("ab") as f:
            f.write(self._encode.encode(record) + b"\n")

    @override
    async def on_fill(self, market: Market, fill: Fill) -> None:
        self._append(
            FillRecord(fill.slug, fill.side, fill.shares, fill.cost, fill.fee, fill.rule, fill.ts)
        )

    @override
    async def on_resolution(
        self, market: Market, winner: Side | None, fills: Sequence[Fill]
    ) -> None:
        for fill in fills:
            pnl = None if winner is None else fill.pnl(winner)
            self._append(SettleRecord(fill.slug, fill.side, winner, pnl))

    def read(self) -> list[Record]:
        if not self._path.exists():
            return []
        decode = msgspec.json.Decoder(Record).decode
        return [decode(line) for line in self._path.read_bytes().splitlines() if line]


class Collector(Sink):
    """Writes every snapshot (with top of book) and every resolution to daily Parquet files."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._pending: dict[str, list[dict[str, object]]] = {}

    @override
    async def on_snapshot(
        self, market: Market, snapshot: Snapshot, books: Mapping[Side, Book]
    ) -> None:
        up, down = books[Side.UP], books[Side.DOWN]
        self._pending.setdefault(market.slug, []).append(
            {
                "epoch": market.epoch,
                "t": snapshot.t,
                "mid_up": snapshot.mid_up,
                "ref": snapshot.ref,
                "vol_1m": snapshot.vol_1m,
                "up_bid": up.best_bid,
                "up_ask": up.best_ask,
                "down_bid": down.best_bid,
                "down_ask": down.best_ask,
            }
        )

    @override
    async def on_resolution(
        self, market: Market, winner: Side | None, fills: Sequence[Fill]
    ) -> None:
        rows = self._pending.pop(market.slug, [])
        day = datetime.fromtimestamp(market.epoch, UTC).date().isoformat()
        if rows:
            _append_parquet(self._root / "snapshots" / f"{day}.parquet", pl.DataFrame(rows))
        cycle = {
            "epoch": market.epoch,
            "winner": None if winner is None else str(winner),
            "twap_s": market.twap_s,
            "fee_rate": market.fees.rate,
            "fee_exponent": market.fees.exponent,
        }
        _append_parquet(self._root / "cycles" / f"{day}.parquet", pl.DataFrame([cycle]))


def _append_parquet(path: Path, rows: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = (
        pl.concat([pl.read_parquet(path), rows], how="diagonal_relaxed") if path.exists() else rows
    )
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


class Telegram(Sink):
    def __init__(self, target: TelegramTarget, http: httpx.AsyncClient) -> None:
        self._target = target
        self._http = http

    async def _send(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self._target.token.reveal()}/sendMessage"
        try:
            response = await self._http.post(
                url, json={"chat_id": self._target.chat_id, "text": text}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("telegram send failed: %s", type(exc).__name__)

    @override
    async def on_fill(self, market: Market, fill: Fill) -> None:
        await self._send(
            f"{market.slug} {fill.rule}: bought {fill.shares:.2f} {fill.side} "
            f"at {fill.price:.3f} (cost {fill.cost:.2f})"
        )

    @override
    async def on_resolution(
        self, market: Market, winner: Side | None, fills: Sequence[Fill]
    ) -> None:
        for fill in fills:
            result = "unresolved" if winner is None else f"{fill.pnl(winner):+.2f}"
            await self._send(f"{market.slug} {fill.rule}: winner {winner}, pnl {result}")
