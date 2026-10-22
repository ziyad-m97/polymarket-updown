"""`updown` command line: collect, paper, live, data, import-legacy, report."""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import httpx

from updown.config import ConfigError, Credentials, TelegramTarget, live_armed
from updown.engine import Engine, Venue
from updown.execution import Executor, LiveExecutor, PaperExecutor
from updown.feeds.binance import Binance
from updown.net import Fetcher
from updown.risk import Limits, RiskGate
from updown.sinks import Collector, Ledger, Sink, Telegram
from updown.strategy.base import Idle, Strategy
from updown.strategy.fair_value import FairValue
from updown.strategy.legacy import LegacyRules
from updown.venue.clob import Clob
from updown.venue.gamma import Gamma
from updown.venue.orders import PolymarketOrders

log = logging.getLogger("updown")
USER_AGENT = "updown/0.1"


def _strategy(args: argparse.Namespace) -> Strategy:
    if args.strategy == "fair-value":
        return FairValue()
    return LegacyRules(args.paths.split(","))


def _limits(args: argparse.Namespace) -> Limits:
    return Limits(max_stake=args.max_stake, max_daily_loss=args.max_daily_loss)


async def _run(
    args: argparse.Namespace, strategy: Strategy, executor: Executor, sinks: list[Sink]
) -> None:
    async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": USER_AGENT}) as http:
        fetcher = Fetcher(http, attempts=3)
        venue = Venue(markets=Gamma(fetcher), books=Clob(fetcher), reference=Binance(fetcher))
        if (target := TelegramTarget.from_env()) is not None:
            sinks.append(Telegram(target, http))
        risk = RiskGate(_limits(args), kill_switch=args.kill_switch)
        log.info(
            "strategy %s, limits %s, kill switch %s", strategy.name, risk.limits, args.kill_switch
        )
        engine = Engine(venue=venue, strategy=strategy, risk=risk, executor=executor, sinks=sinks)
        await engine.run(args.cycles)


def collect(args: argparse.Namespace) -> None:
    asyncio.run(_run(args, Idle(), PaperExecutor(), [Collector(args.out)]))


def paper(args: argparse.Namespace) -> None:
    asyncio.run(_run(args, _strategy(args), PaperExecutor(), [Ledger(args.ledger)]))


def live(args: argparse.Namespace) -> None:
    if not (args.live and live_armed()):
        raise ConfigError("live trading needs both --live and UPDOWN_LIVE=1")
    credentials = Credentials.from_env()

    async def go() -> None:
        orders = await PolymarketOrders.connect(credentials)
        try:
            await _run(args, _strategy(args), LiveExecutor(orders), [Ledger(args.ledger)])
        finally:
            await orders.close()

    asyncio.run(go())


def data(args: argparse.Namespace) -> None:
    from updown.research.build import build  # noqa: PLC0415 - research-only imports

    build(args.out, args.start, args.end, args.cache, args.concurrency)


def import_legacy(args: argparse.Namespace) -> None:
    from updown.research.legacy import collector, live_trades  # noqa: PLC0415

    args.out.mkdir(parents=True, exist_ok=True)
    collector(args.root).write_parquet(args.out / "collector.parquet", statistics=False)
    trades = live_trades(args.root / "sniper_state.json", args.wallet)
    trades.write_parquet(args.out / "live_trades.parquet", statistics=False)


def report(args: argparse.Namespace) -> None:
    from updown.research.report import run  # noqa: PLC0415

    run(args.data, args.results, args.readme)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="updown", description=__doc__)
    sub = parser.add_subparsers(required=True)

    def trading(name: str, handler: object, helptext: str) -> argparse.ArgumentParser:
        cmd = sub.add_parser(name, help=helptext)
        cmd.set_defaults(handler=handler)
        cmd.add_argument("--cycles", type=int, default=None, help="stop after N cycles")
        cmd.add_argument("--max-stake", type=float, default=Limits().max_stake)
        cmd.add_argument("--max-daily-loss", type=float, default=Limits().max_daily_loss)
        cmd.add_argument("--kill-switch", type=Path, default=Path("runs/KILL"))
        return cmd

    cmd = trading("collect", collect, "record snapshots, books and results")
    cmd.add_argument("--out", type=Path, default=Path("data/live"))
    for name, handler in (("paper", paper), ("live", live)):
        cmd = trading(name, handler, f"{name} trading")
        cmd.add_argument("--strategy", choices=("legacy", "fair-value"), default="legacy")
        cmd.add_argument("--paths", default="C1,E", help="legacy paths to trade")
        cmd.add_argument("--ledger", type=Path, default=Path(f"runs/{name}.jsonl"))
    cmd.add_argument("--live", action="store_true", help="really send orders")

    cmd = sub.add_parser("data", help="rebuild the dataset from public APIs")
    cmd.set_defaults(handler=data)
    cmd.add_argument("--start", type=date.fromisoformat, default=date(2026, 2, 10))
    cmd.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 22))
    cmd.add_argument("--out", type=Path, default=Path("data"))
    cmd.add_argument("--cache", type=Path, default=Path(".cache"))
    cmd.add_argument("--concurrency", type=int, default=8)

    cmd = sub.add_parser("import-legacy", help="one-off import of the original bot's records")
    cmd.set_defaults(handler=import_legacy)
    cmd.add_argument("root", type=Path)
    cmd.add_argument("--wallet", default=None, help="public address whose fills to look up")
    cmd.add_argument("--out", type=Path, default=Path("data"))

    cmd = sub.add_parser("report", help="regenerate results/ and the README results")
    cmd.set_defaults(handler=report)
    cmd.add_argument("--data", type=Path, default=Path("data"))
    cmd.add_argument("--results", type=Path, default=Path("results"))
    cmd.add_argument("--readme", type=Path, default=Path("README.md"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = _parser().parse_args(argv)
    try:
        args.handler(args)
    except ConfigError as exc:
        sys.exit(f"updown: {exc}")
    except KeyboardInterrupt:
        log.info("interrupted")
