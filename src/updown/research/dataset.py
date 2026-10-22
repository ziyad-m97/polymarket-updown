"""Load and validate the dataset, label regimes, and iterate cycles for replay."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from updown.clock import Interval, slug
from updown.domain import FeeSchedule, Market, Side, Snapshot

DATA = Path("data")
GRID = frozenset(range(60, 900, 60))
HOLDOUT_START = int(datetime(2026, 3, 14, tzinfo=UTC).timestamp())
"""The legacy rules last changed on 13 March 2026; everything from the next UTC day is unseen."""
HISTORY_LAG_S = 20
"""A price-history point reports the mid from about this many seconds before its timestamp."""

Reference = Literal["observed", "stamped"]


class DataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Dataset:
    cycles: pl.DataFrame
    snapshots: pl.DataFrame


@dataclass(frozen=True, slots=True)
class Cycle:
    market: Market
    winner: Side
    ref_open: float | None
    snapshots: tuple[Snapshot, ...]


def load(root: Path = DATA) -> Dataset:
    ds = Dataset(
        pl.read_parquet(root / "cycles.parquet"), pl.read_parquet(root / "snapshots.parquet")
    )
    validate(ds)
    return ds


def validate(ds: Dataset) -> None:
    cycles, snaps = ds.cycles, ds.snapshots
    checks = {
        "duplicate cycles": cycles["epoch"].is_duplicated().any(),
        "cycle epoch not aligned to 15 minutes": (cycles["epoch"] % 900 != 0).any(),
        "missing winner": cycles["winner"].null_count() > 0,
        "duplicate (epoch, t)": snaps.select(pl.struct("epoch", "t").is_duplicated().any()).item(),
        "snapshot time off the minute grid": (~snaps["t"].is_in(sorted(GRID))).any(),
        "price outside [0, 1]": (~snaps["mid_up"].is_between(0.0, 1.0)).any(),
        "snapshot without a cycle": (~snaps["epoch"].is_in(cycles["epoch"].implode())).any(),
    }
    failed = [name for name, bad in checks.items() if bad]
    if failed:
        raise DataError(", ".join(failed))


def with_regime(cycles: pl.DataFrame) -> pl.DataFrame:
    """Development period, then the holdout split by fee schedule and resolution rule."""
    return cycles.with_columns(
        regime=pl.when(pl.col("epoch") < HOLDOUT_START)
        .then(pl.lit("development"))
        .when(pl.col("twap_s") > 0)
        .then(pl.lit("holdout, TWAP resolution"))
        .when(pl.col("fee_exponent") == 2.0)
        .then(pl.lit("holdout, fee v1"))
        .otherwise(pl.lit("holdout, fee v2"))
    )


def _opt(x: float) -> float | None:
    return None if np.isnan(x) else float(x)


def iter_cycles(ds: Dataset, reference: Reference = "observed") -> Iterator[Cycle]:
    """Cycles in time order, each with its snapshots, as the engine would have seen them.

    With `reference="stamped"` the BTC reference is read at the price point's timestamp instead
    of its observation time, which hands anything compared with the price ~20 s of the future.
    """
    snaps = ds.snapshots.sort("epoch", "t")
    meta = {row["epoch"]: row for row in ds.cycles.iter_rows(named=True)}
    epoch = snaps["epoch"].to_numpy()
    t = snaps["t"].to_numpy()
    mid = snaps["mid_up"].to_numpy()
    column, lag = ("ref", HISTORY_LAG_S) if reference == "observed" else ("ref_stamped", 0)
    ref = snaps[column].fill_null(np.nan).to_numpy()
    vol = snaps["vol_1m"].fill_null(np.nan).to_numpy()
    age = snaps["stamp_age_s"].to_numpy() + lag
    starts = np.flatnonzero(np.diff(epoch, prepend=-1))
    ends = np.append(starts[1:], len(epoch))
    for lo, hi in zip(starts, ends, strict=True):
        row = meta[int(epoch[lo])]
        market = Market(
            slug=slug("btc", Interval.M15, row["epoch"]),
            interval=Interval.M15,
            epoch=row["epoch"],
            condition_id="",
            token_up="",
            token_down="",
            tick=0.01,
            min_shares=5.0,
            fees=FeeSchedule(row["fee_rate"], row["fee_exponent"]),
            twap_s=row["twap_s"],
        )
        snapshots = tuple(
            Snapshot(int(t[i]), float(mid[i]), _opt(ref[i]), _opt(vol[i]), int(age[i]))
            for i in range(lo, hi)
        )
        yield Cycle(market, Side(row["winner"]), row["ref_open"], snapshots)
