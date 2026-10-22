"""Runtime settings. Secrets come from the environment only and never print."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

LIVE_FLAG = "UPDOWN_LIVE"


class ConfigError(RuntimeError):
    pass


class Secret:
    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not value:
            raise ConfigError("empty secret")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(***)"

    __str__ = __repr__


def _require(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "")
    if not value:
        raise ConfigError(f"set {name} in the environment")
    return value


@dataclass(frozen=True, slots=True)
class Credentials:
    private_key: Secret
    wallet: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Credentials":
        return cls(
            private_key=Secret(_require(env, "POLYMARKET_PRIVATE_KEY")),
            wallet=_require(env, "POLYMARKET_WALLET_ADDRESS"),
        )


@dataclass(frozen=True, slots=True)
class TelegramTarget:
    token: Secret
    chat_id: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "TelegramTarget | None":
        if not env.get("TELEGRAM_BOT_TOKEN"):
            return None
        return cls(Secret(env["TELEGRAM_BOT_TOKEN"]), _require(env, "TELEGRAM_CHAT_ID"))


def live_armed(env: Mapping[str, str] = os.environ) -> bool:
    return env.get(LIVE_FLAG) == "1"
