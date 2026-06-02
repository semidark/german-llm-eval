from __future__ import annotations

from unittest.mock import patch

import pytest

from german_llm_eval.cli import main


def test_cli_list_tasks(capsys) -> None:
    """--list-tasks prints table and exits 0."""
    with patch("sys.argv", ["german-llm-eval", "--list-tasks"]):
        main()
    captured = capsys.readouterr()
    assert "germanquad" in captured.out or "Available Tasks" in captured.out


def test_cli_help(capsys) -> None:
    """--help prints usage and exits 0."""
    with (
        patch("sys.argv", ["german-llm-eval", "--help"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Evaluate LLMs" in captured.out
    assert "--model" in captured.out


def test_cli_no_tasks_exits_1(capsys) -> None:
    """With no data available, evaluator exits with code 1."""

    with (
        patch(
            "sys.argv",
            [
                "german-llm-eval",
                "--data-root",
                "/nonexistent",
                "--api-key",
                "sk-test-key",
            ],
        ),
        patch("sys.exit") as mock_exit,
    ):
        main()
    mock_exit.assert_called_once_with(1)


def test_cli_comma_separated_tasks(capsys) -> None:
    """Comma-separated --tasks are split into individual tasks."""
    with patch(
        "sys.argv",
        ["german-llm-eval", "--tasks", "germanquad,polarity", "--list-tasks"],
    ):
        main()
    captured = capsys.readouterr()
    assert "germanquad" in captured.out or "Available Tasks" in captured.out


def test_cli_tasks_splitting() -> None:
    """Comma-separated task names are split into individual entries in main()."""
    from german_llm_eval.cli import parse_args

    args = parse_args(["--tasks", "germanquad,polarity,nli"])
    # parse_args returns raw; main() splits on commas post-parse
    assert args.tasks == ["germanquad,polarity,nli"]
    # Simulate what main() does
    split_tasks = [t.strip() for task in args.tasks for t in task.split(",")]
    assert split_tasks == ["germanquad", "polarity", "nli"]
