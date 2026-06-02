from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass

from openai import AsyncOpenAI


@dataclass
class APIClientConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 3
    timeout_seconds: float = 60.0


class APIClient:
    def __init__(self, config: APIClientConfig | None = None) -> None:
        self._config = config or APIClientConfig()
        api_key = self._config.api_key or os.environ.get("OPENAI_API_KEY", "sk-")
        self._client = AsyncOpenAI(
            base_url=self._config.base_url,
            api_key=api_key,
            timeout=self._config.timeout_seconds,
        )

    async def generate(self, messages: list[dict[str, str]]) -> str:
        last_err: Exception | None = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,
                    temperature=self._config.temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: PERF203
                last_err = exc
                if attempt < self._config.max_retries:
                    await asyncio.sleep(min(2**attempt, 10))

        raise RuntimeError(
            f"Failed after {self._config.max_retries} retries"
        ) from last_err

    async def generate_batch(
        self,
        prompt_list: list[list[dict[str, str]]],
        concurrency: int = 5,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[str]:
        semaphore = asyncio.Semaphore(concurrency)
        completed = 0

        async def _with_limit(prompt: list[dict[str, str]]) -> str:
            nonlocal completed
            async with semaphore:
                result = await self.generate(prompt)
                completed += 1
                if progress_callback:
                    progress_callback(completed)
                return result

        tasks = [_with_limit(p) for p in prompt_list]
        return await asyncio.gather(*tasks, return_exceptions=False)
