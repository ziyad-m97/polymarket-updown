from pathlib import Path

import msgspec
import pytest

from tests.helpers import path, state
from updown.domain import Side, Snapshot
from updown.strategy.legacy import Features, LegacyRules, classify, features

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_decisions.json"


def contested_cycle(ud14: float = 0.30) -> tuple[Snapshot, ...]:
    """Up leads, loses the lead twice, and is behind at minute 7; Down is the underdog at m14."""
    mids = {180: 0.55, 300: 0.45, 420: 0.45, 540: 0.55, 660: 0.60, 780: 0.65, 840: 1 - ud14}
    snaps = list(path(mids))
    snaps[-2] = Snapshot(780, 0.65, ref=100_050.0, vol_1m=0.001)
    return tuple(snaps)


def test_features_follow_the_original_definitions() -> None:
    f = features(state(contested_cycle(), ref_open=100_000.0))
    assert f is not None
    assert f.fav13 is Side.UP
    assert f.fav14 is Side.UP
    assert f.underdog is Side.DOWN
    assert f.lead_changes == 2
    assert f.fav_m7 == pytest.approx(0.45)
    assert f.fav_behind_m7
    assert f.ud_price == pytest.approx(0.30)
    assert f.ud_delta == pytest.approx(0.30 - 0.35)
    assert f.conf_delta == pytest.approx(0.70 - 0.65)
    assert not f.btc_mismatch


def test_contested_cycle_routes_to_c1() -> None:
    intent = LegacyRules({"C1"}).decide(state(contested_cycle(), ref_open=100_000.0))
    assert intent is not None
    assert (intent.side, intent.stake, intent.rule) == (Side.DOWN, 10.0, "legacy.C1")
    assert intent.limit == pytest.approx(0.32)


def test_rules_only_act_at_minute_14_and_respect_enabled_paths() -> None:
    snaps = contested_cycle()
    assert LegacyRules({"C1"}).decide(state(snaps[:-1], ref_open=100_000.0)) is None
    assert LegacyRules({"E"}).decide(state(snaps, ref_open=100_000.0)) is None


def test_z_filter_skips_decisive_moves() -> None:
    snaps = list(contested_cycle())
    snaps[-2] = Snapshot(780, 0.65, ref=101_000.0, vol_1m=0.001)
    assert LegacyRules({"C1"}).decide(state(snaps, ref_open=100_000.0)) is None


def test_cheap_underdog_routes_to_e() -> None:
    intent = LegacyRules({"E"}).decide(state(contested_cycle(ud14=0.12), ref_open=100_000.0))
    assert intent is not None
    assert intent.rule == "legacy.E"
    assert intent.stake == 4.0


def test_unknown_paths_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown paths"):
        LegacyRules({"F"})


class Logged(msgspec.Struct):
    ts: str
    path: str
    entry_price: float
    z_score: float
    fav_14: str
    lead_changes: int
    fav_m7: float | None
    fav_ground_delta: float | None
    ud_delta: float
    conf_delta: float
    btc_mismatch: bool
    signals: list[str]


LOGGED = msgspec.json.decode(FIXTURE.read_bytes(), type=list[Logged])


@pytest.mark.parametrize("row", LOGGED, ids=lambda r: f"{r.ts[:16]}-{r.path}")
def test_port_reproduces_every_logged_decision(row: Logged) -> None:
    """The bot logged its inputs with each order; the port must reach the same path and signals.

    Stakes changed between revisions, so only the decision itself is compared. The minute-14
    favourite price was not logged; mirrored books make it 1 - underdog price. The minute-13
    favourite price is recovered from the logged fav_ground_delta = conf13 - fav_m7.
    """
    fav14 = Side(row.fav_14)
    f = Features(
        z=row.z_score,
        fav13=fav14,
        conf13=0.5 if row.fav_m7 is None else row.fav_m7 + (row.fav_ground_delta or 0.0),
        lead_changes=row.lead_changes,
        fav_m7=row.fav_m7,
        fav14=fav14,
        ud_price=row.entry_price,
        conf14=1.0 - row.entry_price,
        ud_delta=row.ud_delta,
        conf_delta=row.conf_delta,
        btc_mismatch=row.btc_mismatch,
    )
    decision = classify(f)
    assert decision is not None
    assert decision.path == row.path
    assert list(decision.signals) == row.signals
