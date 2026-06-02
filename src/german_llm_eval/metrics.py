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


def compute_accuracy(
    predictions: list[str], labels: list[list[str]]
) -> tuple[int, int]:
    correct = 0
    for pred, lbl in zip(predictions, labels):
        if pred.strip().lower() == lbl[0].strip().lower():
            correct += 1
    return correct, len(labels)


def compute_f1_tokens(pred_tokens: list[str], label_tokens: list[str]) -> float:
    tp = sum(1 for t in pred_tokens if t in label_tokens)
    precision = tp / len(pred_tokens) if pred_tokens else 0.0
    recall = tp / len(label_tokens) if label_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_f1_entities(
    predictions: list[list[tuple[str, str]]],
    labels: list[list[tuple[str, str]]],
) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for pred_set, label_set in zip(predictions, labels):
        pred_norm = {(t, e.lower()) for t, e in pred_set}
        label_norm = {(t, e.lower()) for t, e in label_set}
        tp += len(pred_norm & label_norm)
        fp += len(pred_norm - label_norm)
        fn += len(label_norm - pred_norm)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1
