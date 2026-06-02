from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIConnectionError

from german_llm_eval.client import APIClient, APIClientConfig

_REQ = httpx.Request("GET", "http://test")


@pytest.mark.asyncio
async def test_generate_batch_basic() -> None:
    config = APIClientConfig(model="test-model", api_key="sk-test-key")
    client = APIClient(config)

    prompts = [
        [{"role": "user", "content": "hello"}],
        [{"role": "user", "content": "world"}],
    ]

    mock_resp = AsyncMock()
    mock_resp.choices[0].message.content = "response"

    with patch.object(
        client._client.chat.completions, "create", new=AsyncMock(return_value=mock_resp)
    ):
        results = await client.generate_batch(prompts, concurrency=2)

    assert len(results.responses) == 2
    assert all(r == "response" for r in results.responses)
    assert all(results.successes)


@pytest.mark.asyncio
async def test_generate_batch_progress_callback() -> None:
    config = APIClientConfig(model="test-model", api_key="sk-test-key")
    client = APIClient(config)

    prompts = [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
        [{"role": "user", "content": "three"}],
    ]

    mock_resp = AsyncMock()
    mock_resp.choices[0].message.content = "ok"

    progress_calls: list[int] = []

    with patch.object(
        client._client.chat.completions, "create", new=AsyncMock(return_value=mock_resp)
    ):
        await client.generate_batch(
            prompts, concurrency=2, progress_callback=lambda n: progress_calls.append(n)
        )

    assert len(progress_calls) == 3
    assert progress_calls[-1] == 3


@pytest.mark.asyncio
async def test_generate_retry_success() -> None:
    config = APIClientConfig(model="test-model", api_key="sk-test-key", max_retries=3)
    client = APIClient(config)

    call_count = 0

    async def flaky_create(**_: object) -> object:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise APIConnectionError(request=_REQ)
        resp = AsyncMock()
        resp.choices[0].message.content = "recovered"
        return resp

    with patch.object(client._client.chat.completions, "create", new=flaky_create):
        with patch.object(asyncio, "sleep", new=AsyncMock()):
            result = await client.generate([{"role": "user", "content": "test"}])

    assert result == "recovered"
    assert call_count == 2


@pytest.mark.asyncio
async def test_generate_retry_exhausted() -> None:
    config = APIClientConfig(model="test-model", api_key="sk-test-key", max_retries=2)
    client = APIClient(config)

    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(side_effect=APIConnectionError(request=_REQ)),
    ):
        with patch.object(asyncio, "sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="Failed after 2 retries"):
                await client.generate([{"role": "user", "content": "test"}])


@pytest.mark.asyncio
async def test_generate_batch_partial_failure() -> None:
    """Failed requests in a batch don't abort the entire batch."""
    config = APIClientConfig(model="test-model", api_key="sk-test-key", max_retries=1)
    client = APIClient(config)

    prompts = [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
        [{"role": "user", "content": "three"}],
    ]

    call_count = 0

    async def flaky_create(**_: object) -> object:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise APIConnectionError(request=_REQ)
        resp = AsyncMock()
        resp.choices[0].message.content = "ok"
        return resp

    with patch.object(client._client.chat.completions, "create", new=flaky_create):
        result = await client.generate_batch(prompts, concurrency=3)

    assert len(result.responses) == 3
    assert result.successes == [True, False, True]
    assert result.responses[0] == "ok"
    assert result.responses[1] == ""
    assert result.responses[2] == "ok"


def test_api_key_validation() -> None:
    """Missing API key raises ValueError."""
    config = APIClientConfig(model="test-model", api_key=None)
    with pytest.raises(ValueError, match="API key"):
        APIClient(config)
