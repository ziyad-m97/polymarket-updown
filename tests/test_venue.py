import json
from pathlib import Path

import msgspec
import polars as pl
import pytest

from updown.clock import Interval
from updown.domain import Side
from updown.feeds.archive import SecondSeries
from updown.feeds.binance import Candle, CandleSeries
from updown.venue.clob import parse_book
from updown.venue.gamma import GammaEvent, to_market, winner

FIXTURE = Path(__file__).parent / "fixtures" / "gamma_events.json"


def events() -> list[GammaEvent]:
    return msgspec.json.decode(FIXTURE.read_bytes(), type=list[GammaEvent], strict=False)


def test_market_parses_tokens_fees_and_twap() -> None:
    point, twap = (to_market(e.markets[0], Interval.M15) for e in events())
    assert point.epoch == 1_771_813_800
    assert point.twap_s == 0
    assert (point.fees.rate, point.fees.exponent) == (0.25, 2.0)
    assert point.token_up.startswith("3017")
    assert point.token_down.startswith("4805")
    assert twap.twap_s == 60
    assert (twap.fees.rate, twap.fees.exponent) == (0.07, 1.0)


def test_tokens_follow_outcome_labels_not_positions() -> None:
    raw = events()[0].markets[0]
    labels = json.loads(raw.outcomes)
    tokens = json.loads(raw.clob_token_ids or "[]")
    flipped = msgspec.structs.replace(
        raw,
        outcomes=json.dumps(labels[::-1]),
        clob_token_ids=json.dumps(tokens[::-1]),
        outcome_prices=json.dumps(json.loads(raw.outcome_prices or "[]")[::-1]),
    )
    original, swapped = to_market(raw, Interval.M15), to_market(flipped, Interval.M15)
    assert (original.token_up, original.token_down) == (swapped.token_up, swapped.token_down)
    assert winner(raw) == winner(flipped) == Side.UP


def test_open_or_unsettled_markets_have_no_winner() -> None:
    raw = events()[0].markets[0]
    assert winner(msgspec.structs.replace(raw, closed=False)) is None
    assert winner(msgspec.structs.replace(raw, outcome_prices='["0.5", "0.5"]')) is None


def test_book_parsing_sorts_the_api_order() -> None:
    body = (
        b'{"bids":[{"price":"0.01","size":"100"},{"price":"0.48","size":"5"}],'
        b'"asks":[{"price":"0.99","size":"100"},{"price":"0.52","size":"5"}]}'
    )
    book = parse_book(body)
    assert (book.best_bid, book.best_ask) == (0.48, 0.52)


def test_candle_series_lookups_are_point_in_time() -> None:
    start = 1_773_000_000 - 1_773_000_000 % 60
    closes = [100.0 * (1.001**i) for i in range(120)]
    series = CandleSeries([Candle((start + 60 * i) * 1000, c, c) for i, c in enumerate(closes)])
    assert series.open_at(start + 60) == closes[1]
    assert series.close_at(start + 60) == closes[0]
    assert series.close_at(start + 59) is None
    assert series.vol_1m(start + 61 * 60) == pytest.approx(0.0, abs=1e-12)
    assert series.vol_1m(start + 30 * 60) is None


def test_second_series_reads_the_last_price_before_a_time() -> None:
    frame = pl.DataFrame({"ts": [100, 101, 102], "open": [1.0, 2.0, 3.0], "close": [1.5, 2.5, 3.5]})
    series = SecondSeries(frame)
    assert series.price_at(102) == 2.5
    assert series.price_at(100) is None


def test_second_series_aggregates_complete_minutes_only() -> None:
    ts = list(range(60, 60 + 90))
    frame = pl.DataFrame({"ts": ts, "open": [float(t) for t in ts], "close": [t + 0.5 for t in ts]})
    minutes = SecondSeries(frame).minutes()
    assert [(c.open_ms, c.open, c.close) for c in minutes] == [(60_000, 60.0, 119.5)]
