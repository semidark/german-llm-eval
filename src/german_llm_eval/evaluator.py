from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from german_llm_eval.client import APIClient, APIClientConfig
from german_llm_eval.metrics import TaskResult
from german_llm_eval.tasks.registry import ALL_TASKS, TASK_DEFINITIONS

console = Console()


class Evaluator:
    def __init__(
        self,
        api_config: APIClientConfig | None = None,
        data_root: str | Path = "~/src/SuperGLEBer/data",
        max_samples: int | None = None,
        concurrency: int = 5,
    ) -> None:
        self._api = APIClient(api_config)
        self._data_root = Path(data_root).expanduser()
        self._max_samples = max_samples
        self._concurrency = concurrency

    async def run(
        self, task_names: list[str] | None = None, split: str = "test"
    ) -> list[TaskResult]:
        tasks = task_names or ALL_TASKS
        results: list[TaskResult] = []

        for task_name in tasks:
            if task_name not in TASK_DEFINITIONS:
                console.print(f"[red]Unknown task: {task_name}[/red]")
                continue

            task_def, loader, data_subdir = TASK_DEFINITIONS[task_name]
            data_dir = self._data_root / data_subdir

            try:
                samples = loader.load(data_dir, split=split)
            except FileNotFoundError as exc:
                console.print(f"[yellow]Skipping {task_name}: {exc}[/yellow]")
                continue

            if self._max_samples and len(samples) > self._max_samples:
                samples = samples[: self._max_samples]

            if not samples:
                console.print(
                    f"[yellow]Skipping {task_name}: no samples loaded[/yellow]"
                )
                continue

            console.print(
                f"[bold cyan]{task_def.name}[/bold cyan] "
                f"({len(samples)} samples, metric={task_def.metric})"
            )

            start = time.time()
            prompt_list = [task_def.build_prompt(s) for s in samples]

            with tqdm(total=len(prompt_list), desc=task_name) as pbar:

                def _progress(n: int) -> None:
                    pbar.update(n - pbar.n)

                responses = await self._api.generate_batch(
                    prompt_list, self._concurrency, _progress
                )

            elapsed = time.time() - start
            result: TaskResult = task_def.evaluate(responses, samples)  # type: ignore[assignment]
            results.append(result)

            if result.metric_label:
                console.print(
                    f"  [green]✓[/green] {result.metric_label} in {elapsed:.1f}s"
                )
            else:
                console.print(
                    f"  [green]✓[/green] "
                    f"Accuracy: {result.accuracy:.2%} "
                    f"({result.correct}/{result.total}) "
                    f"in {elapsed:.1f}s"
                )

        return results

    def print_scorecard(self, results: list[TaskResult]) -> None:
        if not results:
            console.print("[yellow]No results to display.[/yellow]")
            return

        table = Table(title="German LLM Evaluation Scorecard")
        table.add_column("Task", style="cyan")
        table.add_column("Metric")
        table.add_column("Correct / Total")
        table.add_column("Score")

        for r in sorted(results, key=lambda x: x.accuracy):
            score_str = f"{r.accuracy:.2%}"
            detail_str = ""
            if "precision" in r.details:
                detail_str = (
                    f"(P={r.details['precision']:.3f} R={r.details['recall']:.3f})"
                )
            ct_str = r.metric_label or f"{r.correct}/{r.total}"
            table.add_row(
                r.name,
                r.details.get("metric", "accuracy"),
                ct_str,
                f"[green]{score_str}[/green]{detail_str}",
            )

        console.print(table)

        avg = sum(r.accuracy for r in results) / len(results)
        console.print(f"\n[bold]Mean Score: {avg:.2%}[/bold]")

    def to_json(self, results: list[TaskResult]) -> dict[str, Any]:
        return {
            "model": self._api._config.model,
            "tasks": [r.to_dict() for r in results],
            "mean_score": (
                sum(r.accuracy for r in results) / len(results) if results else 0.0
            ),
        }

    def save_results(
        self, results: list[TaskResult], path: str | Path = "results.json"
    ) -> None:
        Path(path).write_text(json.dumps(self.to_json(results), indent=2))
