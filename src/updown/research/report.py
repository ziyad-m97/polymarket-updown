"""Regenerate every number, table and figure the README shows.

`updown report` rewrites `results/` and the README block between the result markers, so the
prose can never drift from the data.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import fisher_exact

from updown.domain import FeeSchedule
from updown.research import calibration
from updown.research.backtest import Costs, decide, price
from updown.research.dataset import HOLDOUT_START, Cycle, Dataset, iter_cycles, load, with_regime
from updown.research.stats import holm, return_on_stake, wilson
from updown.strategy.base import Strategy
from updown.strategy.fair_value import FairValue
from updown.strategy.legacy import LegacyRules

log = logging.getLogger(__name__)

BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"
MINUTES = (60, 180, 300, 420, 540, 660, 780, 840)
HALF_SPREADS = (0.0, 0.005, 0.01, 0.02)
BASE = Costs(half_spread=0.01)
DEV = "development"


@dataclass(frozen=True, slots=True)
class Section:
    title: str
    body: str
    numbers: Mapping[str, object]


def _num(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def _rounded(value: object) -> object:
    """Six decimals keep summary.json identical across platforms' floating-point libraries."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in value]
    return value


def _date(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, UTC).date().isoformat()


def table(frame: pl.DataFrame, formats: dict[str, str] | None = None) -> str:
    fmt = formats or {}

    def cell(name: str, value: object) -> str:
        if value is None:
            return ""
        return format(value, fmt[name]) if name in fmt else str(value)

    head = "| " + " | ".join(frame.columns) + " |"
    align = "|" + "|".join("---:" if frame[c].dtype.is_numeric() else "---" for c in frame.columns)
    body = [
        "| " + " | ".join(cell(c, v) for c, v in zip(frame.columns, row, strict=True)) + " |"
        for row in frame.iter_rows()
    ]
    return "\n".join([head, align + "|", *body])


def strategies() -> dict[str, Strategy]:
    rules: dict[str, Strategy] = {f"legacy {p}": LegacyRules({p}) for p in ("A", "C1", "C2", "E")}
    rules["fair value"] = FairValue()
    return rules


def data_section(ds: Dataset, collector: pl.DataFrame) -> Section:
    cycles = with_regime(ds.cycles)
    regimes = (
        cycles.group_by("regime")
        .agg(
            pl.col("epoch").min().alias("from"),
            pl.col("epoch").max().alias("to"),
            pl.len().alias("cycles"),
            (pl.col("winner") == "Up").mean().alias("up share"),
        )
        .sort("from")
        .with_columns(pl.col("from", "to").map_elements(_date, return_dtype=pl.String))
    )
    flat = cycles.filter(pl.col("ref_open").is_not_null() & pl.col("ref_close").is_not_null())
    agree = flat.select(
        ((pl.col("ref_close") >= pl.col("ref_open")) == (pl.col("winner") == "Up")).mean()
    ).item()
    pairs = collector.join(ds.snapshots, on=["epoch", "t"], how="inner", suffix="_api")
    gap = (pairs["mid_up"] - pairs["mid_up_api"]).abs()
    same = ((pairs["mid_up"] >= 0.5) == (pairs["mid_up_api"] >= 0.5)).mean()
    n = {
        "cycles": ds.cycles.height,
        "snapshots": ds.snapshots.height,
        "first": _date(int(_num(ds.cycles["epoch"].min()))),
        "last": _date(int(_num(ds.cycles["epoch"].max()))),
        "median_age_s": _num(ds.snapshots["age_s"].median()),
        "binance_direction_agreement": _num(agree),
        "collector_pairs": pairs.height,
        "collector_median_gap": _num(gap.median()),
        "collector_p90_gap": _num(gap.quantile(0.9)),
        "collector_same_favourite": _num(same),
    }
    body = [
        f"{n['cycles']:,} resolved 15-minute BTC markets from {n['first']} to {n['last']}, with "
        f"{n['snapshots']:,} minute snapshots. A snapshot's price is Polymarket's last "
        f"price-history point at or before the minute (median age {n['median_age_s']:.0f} s); "
        "the BTC reference is read from Binance at that same second.",
        table(regimes, {"up share": ".3f"}),
        f"Binance's open-to-close direction matches the official Chainlink result in "
        f"{n['binance_direction_agreement']:.1%} of cycles, so Binance feeds the model and never "
        f"the labels. The original bot's own collector agrees with the price history on the "
        f"favourite in {n['collector_same_favourite']:.1%} of {n['collector_pairs']:,} shared "
        f"snapshots (median gap {n['collector_median_gap']:.3f}, 90th percentile "
        f"{n['collector_p90_gap']:.3f}); the bot sampled on the minute, a median "
        f"{n['median_age_s']:.0f} s after the history point it is compared with.",
    ]
    return Section("Data", "\n\n".join(body), n)


def calibration_section(ds: Dataset, results: Path) -> Section:
    obs = calibration.observations(ds)
    minutes = calibration.by_minute(obs).filter(pl.col("t").is_in(MINUTES))
    rel = calibration.reliability(obs, 840)
    blends = pl.DataFrame([calibration.blend(obs, t) for t in (300, 600, 840)])
    _reliability_figure(rel, results / "reliability_m14.png")
    body = [
        "Brier score (lower is better) of the market price and of the Brownian model at each "
        "minute, and the market's calibration slope from a logistic fit of the outcome on "
        "logit(price). A slope above 1 means prices are too timid: favourites win more often "
        "than they are priced to.",
        table(
            minutes.select("t", "n", "brier_market", "brier_model", "slope", "intercept"),
            {"brier_market": ".4f", "brier_model": ".4f", "slope": ".3f", "intercept": "+.3f"},
        ),
        "Realised frequency of Up by price bucket at minute 14, with 95% Wilson intervals:",
        table(rel, {"price": ".3f", "freq": ".3f", "lo": ".3f", "hi": ".3f"}),
        "![Calibration at minute 14](results/reliability_m14.png)",
        "Does the model know something the price does not? A logistic blend of logit(price) and "
        "logit(model), fitted on the development period and scored on the holdout:",
        table(
            blends.with_columns(pl.col("t").cast(pl.Int64)),
            {"w_market": ".3f", "w_model": "+.3f", "logloss_market": ".4f", "logloss_blend": ".4f"},
        ),
    ]
    numbers = {
        "by_minute": minutes.to_dicts(),
        "reliability_m14": rel.to_dicts(),
        "blend": blends.to_dicts(),
    }
    return Section("Are the prices calibrated?", "\n\n".join(body), numbers)


def _reliability_figure(rel: pl.DataFrame, path: Path) -> None:
    price, freq = rel["price"].to_numpy(), rel["freq"].to_numpy()
    err = np.vstack([freq - rel["lo"].to_numpy(), rel["hi"].to_numpy() - freq])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="0.7", linewidth=1)
    ax.errorbar(price, freq, yerr=err, fmt="o", color="C0", capsize=3)
    ax.set(xlabel="market price of Up", ylabel="realised frequency of Up", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Calibration at minute 14")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(trades: pl.DataFrame) -> dict[str, float]:
    if trades.height == 0:
        return {"n": 0.0}
    day = (trades["epoch"] // 86_400).to_numpy()
    est = return_on_stake(trades["pnl"].to_numpy(), trades["cost"].to_numpy(), day)
    return {
        "n": float(trades.height),
        "win_rate": _num(trades["won"].mean()),
        "avg_price": _num(trades["price"].mean()),
        "roi": est.value,
        "lo": est.lo,
        "hi": est.hi,
        "p": est.p_value,
    }


def rules_section(ds: Dataset, results: Path) -> Section:
    regimes = with_regime(ds.cycles).select("epoch", "regime")
    cycles: list[Cycle] = list(iter_cycles(ds))
    rows: list[dict[str, object]] = []
    curves: dict[str, pl.DataFrame] = {}
    sensitivity: list[dict[str, object]] = []
    by_regime: list[dict[str, object]] = []
    for name, strategy in strategies().items():
        decisions = decide(cycles, strategy)
        trades = price(decisions, BASE).join(regimes, on="epoch")
        held = trades.filter(pl.col("regime") != DEV)
        dev, hold = evaluate(trades.filter(pl.col("regime") == DEV)), evaluate(held)
        rows.append({"rule": name} | _prefix("dev_", dev) | _prefix("hold_", hold))
        curves[name] = held.sort("epoch")
        for (regime,), part in held.group_by("regime", maintain_order=True):
            stats = evaluate(part)
            cell = f"{stats['roi']:+.3f} ({int(stats['n'])})" if stats["n"] else ""
            by_regime.append({"rule": name, "regime": regime, "cell": cell})
        for half in HALF_SPREADS:
            alt = price(decisions, Costs(half)).filter(pl.col("epoch") >= HOLDOUT_START)
            sensitivity.append(
                {"rule": name, "half_spread": f"{half:.3f}", "roi": evaluate(alt).get("roi")}
            )
    adjusted = holm([_num(r.get("hold_p", 1.0)) for r in rows])
    summary = pl.DataFrame(
        [r | {"hold_p_holm": p} for r, p in zip(rows, adjusted, strict=True)],
        infer_schema_length=None,
    )
    _curves_figure(curves, results / "holdout_pnl.png")
    view = summary.select(
        "rule",
        pl.col("dev_n").cast(pl.Int64).alias("dev n"),
        pl.col("dev_roi").alias("dev ROI"),
        pl.col("hold_n").cast(pl.Int64).alias("holdout n"),
        pl.col("hold_win_rate").alias("win rate"),
        pl.col("hold_avg_price").alias("avg price"),
        pl.col("hold_roi").alias("holdout ROI"),
        pl.format("[{}, {}]", pl.col("hold_lo").round(3), pl.col("hold_hi").round(3)).alias(
            "95% CI"
        ),
        pl.col("hold_p_holm").alias("p (Holm)"),
    )
    sens = pl.DataFrame(sensitivity).pivot(on="half_spread", index="rule", values="roi")
    regime_view = pl.DataFrame(by_regime).pivot(on="regime", index="rule", values="cell")
    body = [
        "Each rule is replayed through the same strategy object that trades live, buying at "
        f"mid + {BASE.half_spread:.2f} and paying the taker fee in force at the time. ROI is P&L "
        "over stake; the interval resamples whole UTC days; p-values test ROI <= 0 and are "
        "Holm-adjusted across the rules. The legacy rules were written during the development "
        "period; nothing after 13 March 2026 was looked at while choosing them.",
        table(
            view,
            {
                "dev ROI": "+.3f",
                "win rate": ".3f",
                "avg price": ".3f",
                "holdout ROI": "+.3f",
                "p (Holm)": ".3f",
            },
        ),
        "![Holdout P&L](results/holdout_pnl.png)",
        "Holdout ROI (trades) in each regime: Polymarket changed its fee curve on 30 March and "
        "moved to a 60-second TWAP for resolution in August.",
        table(regime_view),
        "Holdout ROI by the half-spread paid on entry:",
        table(sens, dict.fromkeys(sens.columns[1:], "+.3f")),
    ]
    numbers = {"rules": summary.to_dicts(), "sensitivity": sensitivity, "by_regime": by_regime}
    return Section("Do the old rules survive out of sample?", "\n\n".join(body), numbers)


def _prefix(prefix: str, values: dict[str, float]) -> dict[str, object]:
    return {f"{prefix}{k}": v for k, v in values.items()}


def _curves_figure(curves: dict[str, pl.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, trades in curves.items():
        if trades.height:
            when = (trades["epoch"].to_numpy() * 1000).astype("datetime64[ms]")
            per_dollar = trades["pnl"].to_numpy() / trades["cost"].to_numpy()
            ax.plot(when, np.cumsum(per_dollar), label=name, linewidth=1.2)
    ax.axhline(0, color="0.6", linewidth=1)
    ax.set(ylabel="cumulative P&L per $1 staked", title="Holdout, from 14 March 2026")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _settle(
    rows: Sequence[dict[str, object]], price_key: str, ratio_key: str | None
) -> pl.DataFrame:
    out = []
    for r in rows:
        price = _num(r[price_key])
        usdc = _num(r["stake"]) * (_num(r[ratio_key]) if ratio_key else 1.0)
        shares = usdc / price
        cost = usdc + FeeSchedule(_num(r["fee_rate"]), _num(r["fee_exponent"])).fee(shares, price)
        won = r["side"] == r["winner"]
        out.append({"cost": cost, "pnl": (shares if won else 0.0) - cost, "won": won})
    return pl.DataFrame(out, schema={"cost": pl.Float64, "pnl": pl.Float64, "won": pl.Boolean})


def _ledger_row(label: str, frame: pl.DataFrame) -> dict[str, object]:
    staked, total, wins = (
        _num(frame["cost"].sum()),
        _num(frame["pnl"].sum()),
        int(frame["won"].sum()),
    )
    lo, hi = wilson(wins, frame.height)
    return {
        "orders": label,
        "n": frame.height,
        "win rate": wins / frame.height if frame.height else 0.0,
        "95% CI": f"[{lo:.2f}, {hi:.2f}]",
        "staked": staked,
        "P&L": total,
        "ROI": total / staked if staked else 0.0,
    }


def live_section(ds: Dataset, live: pl.DataFrame) -> Section:
    outcomes = ds.cycles.select(
        "epoch", pl.col("winner").cast(pl.String), "fee_rate", "fee_exponent"
    )
    orders = live.filter(pl.col("mode") == "live", pl.col("filled").is_not_null()).join(
        outcomes, on="epoch"
    )
    booked = orders.to_dicts()
    filled = [r for r in booked if r["filled"]]
    unfilled = [r for r in booked if not r["filled"]]
    rows = [
        _ledger_row(
            "as booked by the bot (all filled at the limit)",
            _settle(booked, "decision_price", None),
        ),
        _ledger_row(
            "actually filled, at the fill price", _settle(filled, "fill_price", "fill_ratio")
        ),
        _ledger_row("never filled, priced at the limit", _settle(unfilled, "decision_price", None)),
    ]
    rate = len(filled) / max(len(booked), 1)
    won_filled = sum(r["side"] == r["winner"] for r in filled)
    won_unfilled = sum(r["side"] == r["winner"] for r in unfilled)
    contingency = [
        [won_filled, len(filled) - won_filled],
        [won_unfilled, len(unfilled) - won_unfilled],
    ]
    p_value = float(fisher_exact(contingency, alternative="less").pvalue)
    body = [
        f"The bot's state file books {len(booked)} live orders. They were resting limit orders "
        f"at the mid, and the wallet's public trade history shows that only {len(filled)} of "
        f"them ({rate:.0%}) ever filled. A resting buy fills when sellers come to it, which "
        "tends to be when the price is moving against the side being bought.",
        table(
            pl.DataFrame(rows), {"win rate": ".3f", "staked": ".2f", "P&L": "+.2f", "ROI": "+.3f"}
        ),
        "The orders that filled won less often than the ones that did not, as adverse selection "
        f"predicts, but with {len(filled)} fills the gap is not significant on its own (one-sided "
        f"Fisher exact p = {p_value:.2f}). What is certain is that the bot's own ledger, which "
        "counted every order as filled at its limit, did not describe its account.",
    ]
    numbers = {
        "orders": len(booked),
        "filled": len(filled),
        "fill_rate": rate,
        "rows": rows,
        "fisher_p": p_value,
    }
    return Section("What the original bot actually did", "\n\n".join(body), numbers)


def update_readme(readme: Path, block: str) -> None:
    text = readme.read_text()
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{BEGIN}\n\n{block}\n\n{END}{tail}")


def run(data: Path, results: Path, readme: Path) -> None:
    results.mkdir(parents=True, exist_ok=True)
    ds = load(data)
    sections = [
        data_section(ds, pl.read_parquet(data / "collector.parquet")),
        calibration_section(ds, results),
        rules_section(ds, results),
        live_section(ds, pl.read_parquet(data / "live_trades.parquet")),
    ]
    numbers = {s.title: s.numbers for s in sections}
    summary = json.dumps(_rounded(numbers), indent=2, default=str)
    (results / "summary.json").write_text(summary + "\n")
    block = "\n\n".join(f"### {s.title}\n\n{s.body}" for s in sections)
    (results / "results.md").write_text(block + "\n")
    update_readme(readme, block)
    log.info("wrote %s and updated %s", results, readme)
