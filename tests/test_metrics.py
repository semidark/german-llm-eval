from __future__ import annotations

from german_llm_eval.metrics import (
    TaskResult,
    compute_accuracy,
    compute_f1_entities,
    compute_f1_tokens,
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


# -- compute_accuracy --

def test_compute_accuracy_all_correct() -> None:
    preds = ["pos", "neg", "neu"]
    labels = [["pos"], ["neg"], ["neu"]]
    correct, total = compute_accuracy(preds, labels)
    assert correct == 3
    assert total == 3


def test_compute_accuracy_none_correct() -> None:
    preds = ["neg", "pos", "pos"]
    labels = [["pos"], ["neg"], ["neu"]]
    correct, total = compute_accuracy(preds, labels)
    assert correct == 0
    assert total == 3


def test_compute_accuracy_case_insensitive() -> None:
    preds = ["POS", "Neg"]
    labels = [["pos"], ["neg"]]
    correct, total = compute_accuracy(preds, labels)
    assert correct == 2
    assert total == 2


def test_compute_accuracy_whitespace() -> None:
    preds = [" pos ", " neg"]
    labels = [["pos"], ["neg"]]
    correct, total = compute_accuracy(preds, labels)
    assert correct == 2


# -- compute_f1_tokens --

def test_compute_f1_tokens_perfect() -> None:
    pred = ["hello", "world"]
    labels = ["hello", "world"]
    assert compute_f1_tokens(pred, labels) == 1.0


def test_compute_f1_tokens_partial() -> None:
    pred = ["hello", "world", "foo"]
    labels = ["hello", "world"]
    f1 = compute_f1_tokens(pred, labels)
    assert 0.0 < f1 < 1.0


def test_compute_f1_tokens_no_overlap() -> None:
    pred = ["foo", "bar"]
    labels = ["hello", "world"]
    assert compute_f1_tokens(pred, labels) == 0.0


def test_compute_f1_tokens_empty_pred() -> None:
    assert compute_f1_tokens([], ["hello"]) == 0.0


def test_compute_f1_tokens_empty_label() -> None:
    assert compute_f1_tokens(["hello"], []) == 0.0


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
