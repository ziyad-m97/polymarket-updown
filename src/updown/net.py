"""HTTP GET with bounded retries and an optional on-disk cache for immutable responses."""

import asyncio
import hashlib
from collections.abc import Mapping
from pathlib import Path

import httpx

RETRYABLE = frozenset({429, 500, 502, 503, 504})
Params = Mapping[str, str | int]


class Fetcher:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        attempts: int = 6,
        concurrency: int = 8,
        cache_dir: Path | None = None,
    ) -> None:
        self._http = http
        self._attempts = attempts
        self._gate = asyncio.Semaphore(concurrency)
        self._cache_dir = cache_dir

    async def get(self, url: str, params: Params | None = None, *, cache: bool = False) -> bytes:
        path = _cache_path(self._cache_dir, url, params) if cache and self._cache_dir else None
        if path is not None and path.exists():
            return path.read_bytes()
        body = await self._get(url, params)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(body)
            tmp.replace(path)
        return body

    async def _get(self, url: str, params: Params | None) -> bytes:
        delay = 0.5
        async with self._gate:
            for _ in range(self._attempts - 1):
                try:
                    response = await self._http.get(url, params=params)
                except httpx.TransportError:
                    pass
                else:
                    if response.status_code not in RETRYABLE:
                        response.raise_for_status()
                        return response.content
                    delay = max(delay, float(response.headers.get("Retry-After", 0)))
                await asyncio.sleep(delay)
                delay = min(2.0 * delay, 30.0)
            response = await self._http.get(url, params=params)
            response.raise_for_status()
            return response.content


def _cache_path(root: Path, url: str, params: Params | None) -> Path:
    query = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
    digest = hashlib.sha256(f"{url}?{query}".encode()).hexdigest()
    return root / digest[:2] / f"{digest}.json"
