from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger
from openai import APITimeoutError, APIConnectionError, AsyncOpenAI, RateLimitError


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
        api_key = self._config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Use --api-key or set $OPENAI_API_KEY."
            )
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
            except (APIConnectionError, RateLimitError, APITimeoutError) as exc:
                last_err = exc
                if attempt < self._config.max_retries:
                    logger.warning(f"Request failed (attempt {attempt}): {exc}")
                    await asyncio.sleep(min(2**attempt, 10))

        raise RuntimeError(
            f"Failed after {self._config.max_retries} retries"
        ) from last_err

    @property
    def model(self) -> str:
        return self._config.model

    async def generate_batch(
        self,
        prompt_list: list[list[dict[str, str]]],
        concurrency: int = 5,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[str]:
        semaphore = asyncio.Semaphore(concurrency)
        completed = 0
        results: list[str] = ["" for _ in prompt_list]

        async def _with_index(idx: int, prompt: list[dict[str, str]]) -> None:
            nonlocal completed
            async with semaphore:
                results[idx] = await self.generate(prompt)
                completed += 1
                if progress_callback:
                    progress_callback(completed)

        async with asyncio.TaskGroup() as tg:
            for i, prompt in enumerate(prompt_list):
                tg.create_task(_with_index(i, prompt), name=f"prompt-{i}")

        return results
