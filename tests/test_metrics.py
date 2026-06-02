from __future__ import annotations

from german_llm_eval.metrics import (
    TaskResult,
    compute_f1_entities,
)


# -- TaskResult --


def test_task_result_to_dict() -> None:
    r = TaskResult(
        name="polarity",
        correct=80,
        total=100,
        accuracy=0.8,
        details={"classes": ["pos", "neg"]},
        metric_label="F1=0.8000",
    )
    d = r.to_dict()
    assert d["name"] == "polarity"
    assert d["correct"] == 80
    assert d["total"] == 100
    assert d["accuracy"] == 0.8
    assert d["metric_label"] == "F1=0.8000"
    assert d["details"] == {"classes": ["pos", "neg"]}


def test_task_result_default_metric_label() -> None:
    r = TaskResult(
        name="polarity",
        correct=50,
        total=100,
        accuracy=0.5,
        details={},
    )
    assert r.metric_label is None


# -- compute_f1_entities --


def test_compute_f1_entities_perfect() -> None:
    preds = [[("PER", "nico"), ("LOC", "berlin")]]
    labels = [[("PER", "nico"), ("LOC", "berlin")]]
    prec, rec, f1 = compute_f1_entities(preds, labels)
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0


def test_compute_f1_entities_partial() -> None:
    preds = [[("PER", "nico"), ("LOC", "münchen")]]
    labels = [[("PER", "nico"), ("LOC", "berlin")]]
    prec, rec, f1 = compute_f1_entities(preds, labels)
    assert 0.0 < prec <= 1.0
    assert 0.0 < rec <= 1.0
    assert 0.0 < f1 < 1.0


def test_compute_f1_entities_case_insensitive() -> None:
    preds = [[("PER", "Nico")]]
    labels = [[("PER", "nico")]]
    prec, rec, f1 = compute_f1_entities(preds, labels)
    assert f1 == 1.0


def test_compute_f1_entities_no_prediction() -> None:
    preds = [[]]
    labels = [[("PER", "nico")]]
    prec, rec, f1 = compute_f1_entities(preds, labels)
    assert prec == 0.0
    assert rec == 0.0
    assert f1 == 0.0


def test_compute_f1_entities_no_labels() -> None:
    preds = [[("PER", "nico")]]
    labels = [[]]
    prec, rec, f1 = compute_f1_entities(preds, labels)
    assert prec == 0.0
    assert rec == 0.0
    assert f1 == 0.0


def test_compute_f1_entities_multiple_samples() -> None:
    preds = [[("PER", "nico")], [("LOC", "berlin")]]
    labels = [[("PER", "nico")], [("LOC", "münchen")]]
    prec, rec, f1 = compute_f1_entities(preds, labels)
    assert prec == 0.5
    assert rec == 0.5
    assert 0.0 < f1 < 1.0
