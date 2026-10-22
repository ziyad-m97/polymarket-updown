"""How well market prices and the Brownian model predict the winner."""

from itertools import pairwise

import numpy as np
import polars as pl

from updown.model.brownian import prob_up
from updown.research.dataset import Dataset, with_regime
from updown.research.stats import Floats, brier, log_loss, logistic_fit, logit, wilson

BINS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


def observations(ds: Dataset) -> pl.DataFrame:
    """One row per (cycle, minute): market P(Up), model P(Up) and the outcome.

    Both probabilities use information as of the market price's own timestamp, so the model's
    remaining time is counted from there.
    """
    frame = ds.snapshots.join(with_regime(ds.cycles), on="epoch", how="inner").filter(
        pl.col("ref").is_not_null()
        & pl.col("vol_1m").is_not_null()
        & pl.col("ref_open").is_not_null()
    )
    inputs = frame.select("ref_open", "ref", "vol_1m", "t", "age_s", "twap_s").iter_rows()
    model = [prob_up(o, r, v, 900 - t + age, tw) for o, r, v, t, age, tw in inputs]
    return frame.with_columns(
        model=pl.Series(model, dtype=pl.Float64),
        up=(pl.col("winner") == "Up").cast(pl.Float64),
    ).select("epoch", "t", "age_s", "regime", "mid_up", "model", "up")


def _arrays(frame: pl.DataFrame, *columns: str) -> list[Floats]:
    return [frame[c].to_numpy().astype(np.float64) for c in columns]


def by_minute(obs: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (t,), part in obs.group_by("t", maintain_order=True):
        p, m, y = _arrays(part, "mid_up", "model", "up")
        intercept, slope = logistic_fit(logit(p), y)
        rows.append(
            {
                "t": t,
                "n": part.height,
                "brier_market": brier(p, y),
                "brier_model": brier(m, y),
                "logloss_market": log_loss(p, y),
                "logloss_model": log_loss(m, y),
                "slope": float(slope),
                "intercept": float(intercept),
            }
        )
    return pl.DataFrame(rows).sort("t")


def reliability(obs: pl.DataFrame, t: int) -> pl.DataFrame:
    """Realised frequency of Up against the market price, in price bins, at minute `t`."""
    part = obs.filter(pl.col("t") == t)
    rows = []
    for lo, hi in pairwise(BINS):
        cell = part.filter(pl.col("mid_up").is_between(lo, hi, closed="left"))
        if cell.height == 0:
            continue
        price, up = _arrays(cell, "mid_up", "up")
        wins = int(up.sum())
        low, high = wilson(wins, cell.height)
        rows.append(
            {
                "bin": f"[{lo:.2f}, {hi:.2f})",
                "n": cell.height,
                "price": float(price.mean()),
                "freq": wins / cell.height,
                "lo": low,
                "hi": high,
            }
        )
    return pl.DataFrame(rows)


def blend(obs: pl.DataFrame, t: int) -> dict[str, float]:
    """Does the model add information to the price? Fit on development, score on holdout."""
    part = obs.filter(pl.col("t") == t)
    dev = part.filter(pl.col("regime") == "development")
    hold = part.filter(pl.col("regime") != "development")

    def features(frame: pl.DataFrame) -> Floats:
        p, m = _arrays(frame, "mid_up", "model")
        return np.column_stack([logit(p), logit(m)])

    beta = logistic_fit(features(dev), _arrays(dev, "up")[0])
    p_hold, y_hold = _arrays(hold, "mid_up", "up")
    p_blend = 1.0 / (1.0 + np.exp(-(beta[0] + features(hold) @ beta[1:])))
    return {
        "t": float(t),
        "w_market": float(beta[1]),
        "w_model": float(beta[2]),
        "logloss_market": log_loss(p_hold, y_hold),
        "logloss_blend": log_loss(p_blend, y_hold),
    }
