from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIConnectionError

from german_llm_eval.client import APIClient, APIClientConfig

_REQ = httpx.Request("GET", "http://test")


@pytest.mark.asyncio
async def test_generate_batch_basic(tmp_path) -> None:
    config = APIClientConfig(
        model="test-model", api_key="sk-test-key", cache_dir=tmp_path / "cache"
    )
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
async def test_generate_batch_progress_callback(tmp_path) -> None:
    config = APIClientConfig(
        model="test-model", api_key="sk-test-key", cache_dir=tmp_path / "cache"
    )
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
async def test_generate_retry_success(tmp_path) -> None:
    config = APIClientConfig(
        model="test-model",
        api_key="sk-test-key",
        max_retries=3,
        cache_dir=tmp_path / "cache",
    )
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
async def test_generate_retry_exhausted(tmp_path) -> None:
    config = APIClientConfig(
        model="test-model",
        api_key="sk-test-key",
        max_retries=2,
        cache_dir=tmp_path / "cache",
    )
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
async def test_generate_batch_partial_failure(tmp_path) -> None:
    """Failed requests in a batch don't abort the entire batch."""
    config = APIClientConfig(
        model="test-model",
        api_key="sk-test-key",
        max_retries=1,
        cache_dir=tmp_path / "cache",
    )
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


@pytest.mark.asyncio
async def test_cache_miss_calls_api_and_saves(tmp_path) -> None:
    """Cache miss calls API and writes cache file."""
    config = APIClientConfig(
        model="test-model", api_key="sk-test-key", cache_dir=tmp_path / "cache"
    )
    client = APIClient(config)

    prompts = [[{"role": "user", "content": "hello"}]]

    mock_resp = AsyncMock()
    mock_resp.choices[0].message.content = "response"

    with patch.object(
        client._client.chat.completions, "create", new=AsyncMock(return_value=mock_resp)
    ):
        result = await client.generate_batch(prompts, concurrency=1)

    assert result.responses == ["response"]
    assert all(result.successes)

    cache_files = list((tmp_path / "cache").glob("*.json"))
    assert len(cache_files) == 1


@pytest.mark.asyncio
async def test_cache_hit_returns_from_disk(tmp_path) -> None:
    """Cache hit returns from disk without calling API."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = APIClientConfig(
        model="test-model", api_key="sk-test-key", cache_dir=cache_dir
    )
    client = APIClient(config)

    prompts = [[{"role": "user", "content": "hello"}]]
    cache_key = client._cache_key(prompts, "test-model", 0.0)
    (cache_dir / f"{cache_key}.json").write_text(
        json.dumps(
            {
                "responses": ["cached"],
                "successes": [True],
            }
        )
    )

    result = await client.generate_batch(prompts, concurrency=1)

    assert result.responses == ["cached"]
    assert all(result.successes)


@pytest.mark.asyncio
async def test_cache_disabled_no_file_written(tmp_path) -> None:
    """When cache_dir is None, no cache file is written."""
    config = APIClientConfig(model="test-model", api_key="sk-test-key", cache_dir=None)
    client = APIClient(config)

    prompts = [[{"role": "user", "content": "hello"}]]

    mock_resp = AsyncMock()
    mock_resp.choices[0].message.content = "response"

    with patch.object(
        client._client.chat.completions, "create", new=AsyncMock(return_value=mock_resp)
    ):
        result = await client.generate_batch(prompts, concurrency=1)

    assert result.responses == ["response"]
    assert len(list(tmp_path.glob("**/*.json"))) == 0


def test_cache_key_deterministic() -> None:
    """Same inputs produce the same cache key."""
    prompts = [[{"role": "user", "content": "hello"}]]
    key1 = APIClient._cache_key(prompts, "gpt-4o", 0.0)
    key2 = APIClient._cache_key(prompts, "gpt-4o", 0.0)
    assert key1 == key2


def test_cache_key_differs_with_temperature() -> None:
    """Different temperature produces different cache key."""
    prompts = [[{"role": "user", "content": "hello"}]]
    key1 = APIClient._cache_key(prompts, "gpt-4o", 0.0)
    key2 = APIClient._cache_key(prompts, "gpt-4o", 0.7)
    assert key1 != key2
