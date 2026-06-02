"""CLI entry point for german-llm-eval."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console

from german_llm_eval.client import APIClientConfig
from german_llm_eval.evaluator import Evaluator
from german_llm_eval.tasks.registry import ALL_TASKS

_cli_console = Console(stderr=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="german-llm-eval",
        description="Evaluate LLMs on German benchmark datasets via OpenAI-compatible API",
    )

    # API settings
    api = parser.add_argument_group("API")
    api.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API base URL (default: OpenAI)",
    )
    api.add_argument(
        "--api-key",
        default="",
        help="API key (falls back to $OPENAI_API_KEY)",
    )
    api.add_argument("--model", default="gpt-4o-mini", help="Model name")
    api.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    api.add_argument(
        "--max-retries", type=int, default=3, help="Max retries per request"
    )

    # Data settings
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--data-root",
        default="~/src/SuperGLEBer/data",
        help="Root directory containing benchmark datasets",
    )
    data.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=f"Task names to evaluate (default: all). Available: {', '.join(ALL_TASKS)}",
    )
    data.add_argument(
        "--split", default="test", help="Dataset split to use (default: test)"
    )
    data.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per task (useful for quick testing)",
    )

    # Runtime settings
    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument(
        "--concurrency", type=int, default=5, help="Max concurrent API requests"
    )
    runtime.add_argument(
        "--output",
        default=None,
        help="Path to save JSON scorecard (default: stdout only)",
    )
    runtime.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available tasks and exit",
    )

    args = parser.parse_args()

    if args.list_tasks:
        console_list_tasks()
        return

    api_config = APIClientConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )

    evaluator = Evaluator(
        api_config=api_config,
        data_root=args.data_root,
        max_samples=args.max_samples,
        concurrency=args.concurrency,
    )

    results = asyncio.run(evaluator.run(task_names=args.tasks, split=args.split))

    if not results:
        _cli_console.print(
            "No tasks completed. Check data paths and task availability.",
            style="red",
        )
        sys.exit(1)

    evaluator.print_scorecard(results)

    if args.output:
        evaluator.save_results(results, Path(args.output))
        _cli_console.print(f"\nResults saved to {args.output}")


def console_list_tasks() -> None:
    """Print available tasks in a readable format."""
    from rich.table import Table

    con = Console()
    table = Table(title="Available Tasks")
    table.add_column("Task Name", style="cyan")
    table.add_column("Category")
    table.add_column("Description")

    categories: dict[str, list[tuple[str, str]]] = {
        "QA": [],
        "Text-Pair Classification": [],
        "Triple Classification": [],
        "Single-Text Classification": [],
        "NER (Prompt-based)": [],
    }

    from german_llm_eval.tasks.registry import TASK_DEFINITIONS

    for name, (task_def, _, _) in TASK_DEFINITIONS.items():
        desc = getattr(task_def, "description", "")
        if "QA" in desc or "QATask" in type(task_def).__name__:
            categories["QA"].append((name, desc))
        elif "Triple" in type(task_def).__name__:
            categories["Triple Classification"].append((name, desc))
        elif "Pair" in type(task_def).__name__:
            categories["Text-Pair Classification"].append((name, desc))
        elif "NER" in type(task_def).__name__:
            categories["NER (Prompt-based)"].append((name, desc))
        else:
            categories["Single-Text Classification"].append((name, desc))

    for cat, items in categories.items():
        for name, desc in sorted(items):
            table.add_row(name, cat, desc)

    con.print(table)


if __name__ == "__main__":
    main()
