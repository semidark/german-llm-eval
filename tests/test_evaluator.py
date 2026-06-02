from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from german_llm_eval.client import APIClientConfig, BatchResult
from german_llm_eval.evaluator import Evaluator
from german_llm_eval.metrics import TaskResult


@pytest.mark.asyncio
async def test_evaluator_run_basic(tmp_path: Path) -> None:
    """Evaluator.run() loads samples, calls API, returns TaskResult."""
    (tmp_path / "Germeval" / "2017").mkdir(parents=True)
    (tmp_path / "Germeval" / "2017" / "test.tsv").write_text(
        "id\tGut\t-src\tpositive\nid\tSchlecht\t-src\tnegative\n"
    )

    evaluator = Evaluator(
        api_config=APIClientConfig(model="fake", api_key="sk-test-key"),
        data_root=tmp_path,
    )

    with patch.object(evaluator._api, "generate_batch", new_callable=AsyncMock) as mock:
        mock.return_value = BatchResult(
            responses=["positive", "negative"],
            successes=[True, True],
        )
        results = await evaluator.run(task_names=["polarity"], split="test")

    assert len(results) == 1
    assert results[0].name == "polarity"
    assert results[0].total == 2
    assert results[0].correct == 2
    assert results[0].accuracy == 1.0
    mock.assert_called_once()


@pytest.mark.asyncio
async def test_evaluator_run_unknown_task(tmp_path: Path) -> None:
    """Unknown task name is skipped gracefully."""
    evaluator = Evaluator(
        api_config=APIClientConfig(model="fake", api_key="sk-test-key"),
        data_root=tmp_path,
    )

    with patch.object(evaluator._api, "generate_batch", new_callable=AsyncMock) as mock:
        results = await evaluator.run(task_names=["nonexistent"], split="test")

    assert len(results) == 0
    mock.assert_not_called()


@pytest.mark.asyncio
async def test_evaluator_run_missing_data(tmp_path: Path) -> None:
    """Missing data directory produces skip, no crash."""
    evaluator = Evaluator(
        api_config=APIClientConfig(model="fake", api_key="sk-test-key"),
        data_root=tmp_path,
    )

    results = await evaluator.run(task_names=["polarity"], split="test")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_evaluator_max_samples(tmp_path: Path) -> None:
    """max_samples limits the number of samples processed."""
    (tmp_path / "Germeval" / "2017").mkdir(parents=True)
    rows = "\n".join(f"id\trow{i}\t-src\tpositive" for i in range(100))
    (tmp_path / "Germeval" / "2017" / "test.tsv").write_text(rows)

    evaluator = Evaluator(
        api_config=APIClientConfig(model="fake", api_key="sk-test-key"),
        data_root=tmp_path,
        max_samples=10,
    )

    with patch.object(evaluator._api, "generate_batch", new_callable=AsyncMock) as mock:
        mock.return_value = BatchResult(
            responses=["positive"] * 10,
            successes=[True] * 10,
        )
        results = await evaluator.run(task_names=["polarity"], split="test")

    assert results[0].total == 10
    calls = mock.call_args[0][0]
    assert len(calls) == 10


def test_evaluator_print_scorecard(capsys) -> None:
    """print_scorecard renders without error."""
    evaluator = Evaluator(
        api_config=APIClientConfig(model="fake", api_key="sk-test-key")
    )
    results = [
        TaskResult(
            name="polarity",
            correct=80,
            total=100,
            accuracy=0.8,
            details={"metric": "accuracy"},
        ),
        TaskResult(
            name="ner_news",
            correct=50,
            total=100,
            accuracy=0.5,
            details={"metric": "f1", "precision": 0.6, "recall": 0.5},
            metric_label="F1=0.500",
        ),
    ]
    evaluator.print_scorecard(results)
    captured = capsys.readouterr()
    assert "Scorecard" in captured.out or "scorecard" in captured.out.lower()
    assert "polarity" in captured.out
    assert "ner_news" in captured.out


def test_evaluator_to_json() -> None:
    """to_json produces valid JSON-serializable dict."""
    evaluator = Evaluator(
        api_config=APIClientConfig(model="gpt-4o", api_key="sk-test-key")
    )
    results = [
        TaskResult(
            name="polarity",
            correct=80,
            total=100,
            accuracy=0.8,
            details={"metric": "accuracy"},
        )
    ]
    data = evaluator.to_json(results)
    assert data["model"] == "gpt-4o"
    assert len(data["tasks"]) == 1
    assert data["mean_score"] == 0.8
    json.dumps(data)  # Should not raise


def test_evaluator_save_results(tmp_path: Path) -> None:
    """save_results writes valid JSON file."""
    evaluator = Evaluator(
        api_config=APIClientConfig(model="fake", api_key="sk-test-key")
    )
    results = [
        TaskResult(
            name="polarity",
            correct=50,
            total=100,
            accuracy=0.5,
            details={"metric": "accuracy"},
        )
    ]
    out = tmp_path / "results.json"
    evaluator.save_results(results, out)
    data = json.loads(out.read_text())
    assert data["tasks"][0]["name"] == "polarity"


@pytest.mark.asyncio
async def test_evaluator_ner_task(tmp_path: Path) -> None:
    """Evaluator correctly handles NER task with entity F1 metric."""
    (tmp_path / "NER" / "Wiki_News").mkdir(parents=True)
    (tmp_path / "NER" / "Wiki_News" / "test.txt").write_text(
        "0\tNico\tB-MISC\n1\tlives\tO\n"
    )

    evaluator = Evaluator(
        api_config=APIClientConfig(model="fake", api_key="sk-test-key"),
        data_root=tmp_path,
    )

    with patch.object(evaluator._api, "generate_batch", new_callable=AsyncMock) as mock:
        mock.return_value = BatchResult(
            responses=["MISC: Nico"],
            successes=[True],
        )
        results = await evaluator.run(task_names=["ner_wiki_news"], split="test")

    assert len(results) == 1
    assert results[0].name == "ner_wiki_news"
    assert results[0].accuracy == 1.0


@pytest.mark.asyncio
async def test_evaluator_partial_batch_success(tmp_path: Path) -> None:
    """Partial batch failures still produce results from successful requests."""
    (tmp_path / "Germeval" / "2017").mkdir(parents=True)
    (tmp_path / "Germeval" / "2017" / "test.tsv").write_text(
        "id\tGut\t-src\tpositive\nid\tSchlecht\t-src\tnegative\nid\tOkay\t-src\tpositive\n"
    )

    evaluator = Evaluator(
        api_config=APIClientConfig(model="fake", api_key="sk-test-key"),
        data_root=tmp_path,
    )

    with patch.object(evaluator._api, "generate_batch", new_callable=AsyncMock) as mock:
        mock.return_value = BatchResult(
            responses=["positive", "", "positive"],
            successes=[True, False, True],
        )
        results = await evaluator.run(task_names=["polarity"], split="test")

    assert len(results) == 1
    assert results[0].total == 2
    assert results[0].correct == 2
    assert results[0].details["skipped"] == 1


@pytest.mark.asyncio
async def test_evaluator_all_requests_fail(tmp_path: Path) -> None:
    """When all requests fail, task is skipped."""
    (tmp_path / "Germeval" / "2017").mkdir(parents=True)
    (tmp_path / "Germeval" / "2017" / "test.tsv").write_text(
        "id\tGut\t-src\tpositive\nid\tSchlecht\t-src\tnegative\n"
    )

    evaluator = Evaluator(
        api_config=APIClientConfig(model="fake", api_key="sk-test-key"),
        data_root=tmp_path,
    )

    with patch.object(evaluator._api, "generate_batch", new_callable=AsyncMock) as mock:
        mock.return_value = BatchResult(
            responses=["", ""],
            successes=[False, False],
        )
        results = await evaluator.run(task_names=["polarity"], split="test")

    assert len(results) == 0


@pytest.mark.asyncio
async def test_evaluator_metric_in_details(tmp_path: Path) -> None:
    """TaskResult.details contains correct metric name."""
    (tmp_path / "Germeval" / "2017").mkdir(parents=True)
    (tmp_path / "Germeval" / "2017" / "test.tsv").write_text(
        "id\tGut\t-src\tpositive\n"
    )

    evaluator = Evaluator(
        api_config=APIClientConfig(model="fake", api_key="sk-test-key"),
        data_root=tmp_path,
    )

    with patch.object(evaluator._api, "generate_batch", new_callable=AsyncMock) as mock:
        mock.return_value = BatchResult(
            responses=["positive"],
            successes=[True],
        )
        results = await evaluator.run(task_names=["polarity"], split="test")

    assert results[0].details["metric"] == "accuracy"
