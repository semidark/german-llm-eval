from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TaskResult:
    name: str
    correct: int
    total: int
    accuracy: float
    details: dict[str, Any]
    metric_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.accuracy,
            "metric_label": self.metric_label,
            "details": self.details,
        }


def compute_f1_entities(
    predictions: list[list[tuple[str, str]]],
    labels: list[list[tuple[str, str]]],
) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for pred_set, label_set in zip(predictions, labels):
        pred_norm = {(t.lower(), e.lower()) for t, e in pred_set}
        label_norm = {(t.lower(), e.lower()) for t, e in label_set}
        tp += len(pred_norm & label_norm)
        fp += len(pred_norm - label_norm)
        fn += len(label_norm - pred_norm)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1
