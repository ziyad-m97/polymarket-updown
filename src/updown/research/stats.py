"""Inference for sparse, regime-dependent trade P&L."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Floats = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Estimate:
    value: float
    lo: float
    hi: float
    p_value: float
    """One-sided, for H0: value <= 0."""


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def return_on_stake(
    pnl: Floats,
    stake: Floats,
    day: npt.NDArray[np.int64],
    *,
    draws: int = 10_000,
    seed: int = 7,
) -> Estimate:
    """Total P&L over total stake, with a bootstrap that resamples whole UTC days.

    Trades on the same day share a volatility regime, so days rather than trades are treated
    as the independent units. The p-value uses the bootstrap distribution shifted to the null.
    """
    days, index = np.unique(day, return_inverse=True)
    pnl_by_day = np.bincount(index, weights=pnl, minlength=len(days))
    stake_by_day = np.bincount(index, weights=stake, minlength=len(days))
    value = float(pnl_by_day.sum() / stake_by_day.sum())
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(days), size=(draws, len(days)))
    boot = pnl_by_day[pick].sum(axis=1) / stake_by_day[pick].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p_value = float(np.mean(boot - value >= value)) if value > 0 else 1.0
    return Estimate(value, float(lo), float(hi), p_value)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def logistic_fit(x: Floats, y: Floats, iterations: int = 50) -> Floats:
    """Maximum-likelihood logistic regression of y on [1, x] by Newton's method."""
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-design @ beta))
        gradient = design.T @ (y - p)
        hessian = design.T @ (design * (p * (1.0 - p))[:, None])
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def logit(p: Floats, eps: float = 1e-4) -> Floats:
    q = np.clip(p, eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def brier(p: Floats, y: Floats) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: Floats, y: Floats, eps: float = 1e-4) -> float:
    q = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))
