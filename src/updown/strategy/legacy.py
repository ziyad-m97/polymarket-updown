"""The March 2026 bot's minute-14 underdog rules, ported as they executed.

The bot bought the minute-14 underdog when the cycle looked "contested". Paths and stakes are
those of its last revision (13 March 2026). Two quirks of the original are kept on purpose, so
that a backtest describes what actually traded:

* a favourite flip between minutes 13 and 14 was meant to route to a dedicated path (F), but
  that branch fell through into the chain below and never traded as F;
* a "BTC reversed mid-cycle" skip was computed and logged but never applied.

The original posted resting limit orders at the mid; here the limit is the mid plus `slippage`
and the executor fills or kills.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from updown.domain import CycleState, Intent, Side
from updown.model.brownian import legacy_z
from updown.model.features import lead_changes

EARLY = (180, 300, 420, 540, 660)
M7, M13, M14 = 420, 780, 840
PATHS = frozenset({"A", "C1", "C2", "E"})
TRADED_LIVE = frozenset({"C1", "E"})


@dataclass(frozen=True, slots=True)
class Params:
    z_max: float = 2.0
    ud_min: float = 0.20
    c2_ud_min: float = 0.25
    e_min: float = 0.08
    contested_conf: float = 0.80
    signal_ud_min: float = 0.25
    fav_losing: float = -0.05
    base_bet: float = 2.0
    signal_bet: float = 2.0
    c1_bet: float = 10.0
    c2_bet: float = 2.0
    e_bet: float = 4.0
    slippage: float = 0.02


DEFAULT = Params()


@dataclass(frozen=True, slots=True)
class Features:
    z: float
    fav13: Side
    conf13: float
    lead_changes: int
    fav_m7: float | None
    """Minute-7 price of the minute-13 favourite."""
    fav14: Side
    ud_price: float
    conf14: float
    ud_delta: float
    conf_delta: float
    btc_mismatch: bool

    @property
    def underdog(self) -> Side:
        return self.fav14.other

    @property
    def fav_behind_m7(self) -> bool:
        return self.fav_m7 is not None and self.fav_m7 < 0.5

    @property
    def fav_ground_delta(self) -> float | None:
        return None if self.fav_m7 is None else self.conf13 - self.fav_m7


@dataclass(frozen=True, slots=True)
class Decision:
    path: str
    stake: float
    signals: tuple[str, ...]


def features(state: CycleState) -> Features | None:
    m13, m14 = state.at(M13), state.at(M14)
    if m13 is None or m14 is None or state.ref_open is None:
        return None
    if m13.ref is None or m13.vol_1m is None:
        return None
    m7 = state.at(M7)
    early = [s for t in EARLY if (s := state.at(t)) is not None]
    fav13, fav14 = m13.favorite, m14.favorite
    underdog = fav14.other
    move = m13.ref - state.ref_open
    return Features(
        z=legacy_z(state.ref_open, m13.ref, m13.vol_1m),
        fav13=fav13,
        conf13=m13.mid(fav13),
        lead_changes=lead_changes([*early, m13]),
        fav_m7=m7.mid(fav13) if m7 is not None else None,
        fav14=fav14,
        ud_price=m14.mid(underdog),
        conf14=m14.mid(fav14),
        ud_delta=m14.mid(underdog) - m13.mid(underdog),
        conf_delta=m14.mid(fav14) - m13.mid(fav13),
        btc_mismatch=(move > 0 and fav14 is Side.DOWN) or (move < 0 and fav14 is Side.UP),
    )


def _contested(f: Features) -> list[str]:
    signals = []
    if f.ud_delta > 0:
        signals.append("ud_delta>0")
    if f.conf_delta <= 0:
        signals.append("conf_fall")
    if f.btc_mismatch:
        signals.append("btc_mismatch")
    return signals


def _path_a_signals(f: Features, p: Params) -> list[str]:
    signals = []
    if f.ud_delta > 0 and f.ud_price >= p.signal_ud_min:
        signals.append("ud_delta>0")
    if f.conf_delta <= 0 and f.ud_price >= p.signal_ud_min:
        signals.append("conf_fall")
    if f.btc_mismatch:
        signals.append("btc_mismatch")
    fgd = f.fav_ground_delta
    if fgd is not None and fgd < p.fav_losing:
        signals.append("fav_losing")
    return signals


def _path_e_signals(f: Features) -> list[str]:
    signals = ["mid_range"]
    if f.ud_delta > 0:
        signals.append("ud_rising")
    if f.conf_delta < 0:
        signals.append("conf_fall")
    if f.btc_mismatch:
        signals.append("btc_mm")
    if f.lead_changes >= 2:
        signals.append(f"lc={f.lead_changes}")
    if f.fav_behind_m7:
        signals.append("bm7")
    return signals


def classify(f: Features, p: Params = DEFAULT) -> Decision | None:
    """The original if/elif chain, in its original order."""
    ud, contested = f.ud_price, f.conf14 < p.contested_conf
    if ud >= p.ud_min and contested and f.fav_behind_m7 and f.lead_changes >= 2:
        return Decision("C1", p.c1_bet, ("contested_triple", *_contested(f)))
    if ud >= p.c2_ud_min and contested and f.lead_changes >= 2 and not f.fav_behind_m7:
        return Decision("C2", p.c2_bet, ("contested_leads", *_contested(f)))
    if ud >= p.ud_min and f.lead_changes >= 1:
        signals = _path_a_signals(f, p)
        if not signals:
            return None
        return Decision("A", p.base_bet + p.signal_bet * len(signals), tuple(signals))
    if ud >= p.ud_min:
        return None
    if ud >= p.e_min:
        return Decision("E", p.e_bet, tuple(_path_e_signals(f)))
    return None


class LegacyRules:
    def __init__(self, paths: Iterable[str] = TRADED_LIVE, params: Params = DEFAULT) -> None:
        self._paths = frozenset(paths)
        if not self._paths <= PATHS:
            raise ValueError(f"unknown paths {sorted(self._paths - PATHS)}")
        self._params = params

    @property
    def name(self) -> str:
        return "legacy:" + "+".join(sorted(self._paths))

    def decide(self, state: CycleState) -> Intent | None:
        if state.t != M14:
            return None
        f = features(state)
        if f is None or f.z >= self._params.z_max:
            return None
        decision = classify(f, self._params)
        if decision is None or decision.path not in self._paths:
            return None
        return Intent(
            side=f.underdog,
            limit=min(f.ud_price + self._params.slippage, 0.99),
            stake=decision.stake,
            rule=f"legacy.{decision.path}",
        )
