"""Fair probability of "Up" when log price is a driftless Brownian motion.

With x = log(S_t / S_0) and remaining time tau, the market resolves Up when the resolution value
is >= S_0, so P(Up) = Phi(x / s), where s is the std of the log change still to come.

When the resolution value is the average of the last L seconds (TWAP markets), the remaining
variance shrinks: for tau >= L it is sigma^2 * (tau - 2L/3); inside the window it is
sigma^2 * tau^3 / (3 L^2). Inside the window the already-realised part of the average is
ignored, which is exact only if the price has not moved since the window opened.
"""

import math

from scipy.special import ndtr


def horizon_std(vol_1m: float, remaining_s: float, twap_s: float = 0.0) -> float:
    if remaining_s <= 0.0:
        return 0.0
    if twap_s <= 0.0:
        effective = remaining_s
    elif remaining_s >= twap_s:
        effective = remaining_s - 2.0 * twap_s / 3.0
    else:
        effective = remaining_s**3 / (3.0 * twap_s * twap_s)
    return vol_1m * math.sqrt(effective / 60.0)


def prob_up(
    ref_open: float, ref_now: float, vol_1m: float, remaining_s: float, twap_s: float = 0.0
) -> float:
    x = math.log(ref_now / ref_open)
    s = horizon_std(vol_1m, remaining_s, twap_s)
    if s == 0.0:
        return 1.0 if x >= 0.0 else 0.0
    return float(ndtr(x / s))


def legacy_z(ref_open: float, ref_now: float, vol_1m: float, remaining_min: float = 2.0) -> float:
    """The March 2026 bot's filter: |move| in units of the move expected in the time left."""
    expected = vol_1m * math.sqrt(remaining_min)
    if expected <= 0.0:
        return 0.0
    return abs(ref_now / ref_open - 1.0) / expected
