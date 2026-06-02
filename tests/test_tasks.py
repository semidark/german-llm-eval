from __future__ import annotations

from german_llm_eval.loaders.base import Sample
from german_llm_eval.tasks.base import (
    _normalize_text,
    BinaryClassificationTask,
    MultiClassificationTask,
    NERPromptTask,
    QATask,
    TextPairClassificationTask,
    TextTripleClassificationTask,
)
from german_llm_eval.tasks.registry import ALL_TASKS, TASK_DEFINITIONS


# -- _normalize_text --


def test_normalize_text_basic() -> None:
    assert _normalize_text("Hello World") == "hello world"


def test_normalize_text_punctuation() -> None:
    assert _normalize_text("Hello, World!") == "hello world"


def test_normalize_text_whitespace() -> None:
    assert _normalize_text("  Hello   World  ") == "hello world"


def test_normalize_text_unicode() -> None:
    assert _normalize_text("Straße") == "strasse"


# -- QATask --


def test_qa_task_prompt() -> None:
    task = QATask(name="germanquad")
    sample = Sample(
        inputs={"context": "Berlin ist die Hauptstadt.", "question": "Wo ist Berlin?"},
        labels=["Hauptstadt"],
    )
    prompt = task.build_prompt(sample)
    assert len(prompt) == 1
    assert prompt[0]["role"] == "user"
    assert "Berlin" in prompt[0]["content"]


def test_qa_task_evaluate_exact_match() -> None:
    task = QATask(name="germanquad")
    samples = [
        Sample(inputs={}, labels=["Hauptstadt"]),
        Sample(inputs={}, labels=["Berlin"]),
    ]
    result = task.evaluate(["Hauptstadt", "Wrong"], samples)
    assert result.correct == 1
    assert result.total == 2
    assert result.accuracy == 0.5


def test_qa_task_evaluate_case_insensitive() -> None:
    task = QATask(name="germanquad")
    samples = [Sample(inputs={}, labels=["Hauptstadt"])]
    result = task.evaluate(["HAUPTSTADT"], samples)
    assert result.accuracy == 1.0


def test_qa_task_evaluate_punctuation_stripped() -> None:
    task = QATask(name="germanquad")
    samples = [Sample(inputs={}, labels=["Hauptstadt"])]
    result = task.evaluate(["Hauptstadt!"], samples)
    assert result.accuracy == 1.0


def test_qa_task_evaluate_whitespace_normalized() -> None:
    task = QATask(name="germanquad")
    samples = [Sample(inputs={}, labels=["Berlin Deutschland"])]
    result = task.evaluate(["  Berlin   Deutschland  "], samples)
    assert result.accuracy == 1.0


def test_qa_task_evaluate_unicode_normalized() -> None:
    task = QATask(name="germanquad")
    samples = [Sample(inputs={}, labels=["Stra\u00dfe"])]
    result = task.evaluate(["strasse"], samples)
    assert result.accuracy == 1.0


def test_qa_task_details_has_metric() -> None:
    task = QATask(name="germanquad")
    samples = [Sample(inputs={}, labels=["Berlin"])]
    result = task.evaluate(["Berlin"], samples)
    assert result.details["metric"] == "exact_match"


# -- BinaryClassificationTask --


def test_binary_task_prompt() -> None:
    task = BinaryClassificationTask(
        name="test",
        description="Test",
        label_map={"0": "neg", "1": "pos"},
    )
    sample = Sample(inputs={"text": "Hallo Welt"}, labels=["pos"])
    prompt = task.build_prompt(sample)
    assert "neg" in prompt[0]["content"]
    assert "pos" in prompt[0]["content"]


def test_binary_task_evaluate() -> None:
    """Model returns mapped labels ('pos'/'neg'), raw data has '0'/'1'."""
    task = BinaryClassificationTask(
        name="test",
        description="Test",
        label_map={"0": "neg", "1": "pos"},
    )
    samples = [
        Sample(inputs={}, labels=["1"]),
        Sample(inputs={}, labels=["0"]),
    ]
    result = task.evaluate(["pos", "neg"], samples)
    assert result.correct == 2
    assert result.accuracy == 1.0


def test_binary_task_evaluate_inverse_mapping() -> None:
    """Model returns mapped label values that map back to raw labels."""
    task = BinaryClassificationTask(
        name="test",
        description="Test",
        label_map={"0": "neg", "1": "pos"},
    )
    samples = [
        Sample(inputs={}, labels=["1"]),
        Sample(inputs={}, labels=["0"]),
    ]
    result = task.evaluate(["POS", "NEG"], samples)
    assert result.correct == 2
    assert result.accuracy == 1.0


def test_binary_task_evaluate_partial() -> None:
    task = BinaryClassificationTask(
        name="test",
        description="Test",
        label_map={"0": "neg", "1": "pos"},
    )
    samples = [
        Sample(inputs={}, labels=["1"]),
        Sample(inputs={}, labels=["0"]),
    ]
    result = task.evaluate(["pos", "wrong"], samples)
    assert result.correct == 1
    assert result.accuracy == 0.5


def test_binary_task_details_has_metric() -> None:
    task = BinaryClassificationTask(
        name="test",
        description="Test",
        label_map={"0": "neg", "1": "pos"},
    )
    samples = [Sample(inputs={}, labels=["1"])]
    result = task.evaluate(["pos"], samples)
    assert result.details["metric"] == "accuracy"


# -- MultiClassificationTask --


def test_multi_task_prompt() -> None:
    task = MultiClassificationTask(
        name="polarity",
        description="Polarity",
        classes=["positive", "negative", "neutral"],
    )
    sample = Sample(inputs={"text": "Es ist okay"}, labels=["neutral"])
    prompt = task.build_prompt(sample)
    assert "positive" in prompt[0]["content"]
    assert "negative" in prompt[0]["content"]
    assert "neutral" in prompt[0]["content"]


def test_multi_task_evaluate() -> None:
    task = MultiClassificationTask(
        name="polarity",
        description="Polarity",
        classes=["positive", "negative", "neutral"],
    )
    samples = [
        Sample(inputs={}, labels=["positive"]),
        Sample(inputs={}, labels=["negative"]),
        Sample(inputs={}, labels=["neutral"]),
    ]
    result = task.evaluate(["positive", "wrong", "neutral"], samples)
    assert result.correct == 2
    assert result.accuracy == 2 / 3


def test_multi_task_details_has_metric() -> None:
    task = MultiClassificationTask(
        name="polarity",
        description="Polarity",
        classes=["positive", "negative", "neutral"],
    )
    samples = [Sample(inputs={}, labels=["positive"])]
    result = task.evaluate(["positive"], samples)
    assert result.details["metric"] == "accuracy"


# -- NERPromptTask --


def test_ner_task_entity_types() -> None:
    task = NERPromptTask(name="ner_news")
    assert task._entity_types == ["PERSON", "LOC", "ORG"]


def test_ner_task_legal_types() -> None:
    task = NERPromptTask(name="ner_legal")
    assert "PER" in task._entity_types
    assert "LOC" in task._entity_types
    assert len(task._entity_types) > 0


def test_ner_task_wiki_news_no_o() -> None:
    task = NERPromptTask(name="ner_wiki_news")
    assert "O" not in task._entity_types


def test_ner_task_prompt() -> None:
    task = NERPromptTask(name="ner_news")
    sample = Sample(inputs={"text": "Nico wohnt in Berlin."}, labels=[])
    prompt = task.build_prompt(sample)
    assert "PERSON" in prompt[0]["content"]
    assert "LOC" in prompt[0]["content"]
    assert "Nico" in prompt[0]["content"]


def test_ner_task_parse_response() -> None:
    task = NERPromptTask(name="ner_news")
    entities = task._parse_response("PERSON: Nico\nLOC: Berlin")
    assert ("PERSON", "Nico") in entities
    assert ("LOC", "Berlin") in entities


def test_ner_task_parse_no_entities() -> None:
    task = NERPromptTask(name="ner_news")
    entities = task._parse_response("<no-entities>")
    assert entities == []


def test_ner_task_parse_parentheses_format() -> None:
    task = NERPromptTask(name="ner_news")
    entities = task._parse_response("(PERSON) Nico\n(LOC) Berlin")
    assert ("PERSON", "Nico") in entities
    assert ("LOC", "Berlin") in entities


def test_ner_task_parse_entity_parentheses_format() -> None:
    task = NERPromptTask(name="ner_news")
    entities = task._parse_response("Nico (PERSON)\nBerlin (LOC)")
    assert ("PERSON", "Nico") in entities
    assert ("LOC", "Berlin") in entities


def test_ner_task_parse_mixed_format() -> None:
    task = NERPromptTask(name="ner_news")
    entities = task._parse_response("PERSON: Nico\n(LOC) Berlin\nMünchen (ORG)")
    assert ("PERSON", "Nico") in entities
    assert ("LOC", "Berlin") in entities
    assert ("ORG", "München") in entities


def test_ner_task_evaluate_f1() -> None:
    task = NERPromptTask(name="ner_news")
    samples = [
        Sample(inputs={}, labels=["PERSON: Nico", "LOC: Berlin"]),
    ]
    result = task.evaluate(["PERSON: Nico\nLOC: Berlin"], samples)
    assert result.accuracy == 1.0
    assert result.metric_label == "F1=1.0000"
    assert result.details["precision"] == 1.0
    assert result.details["recall"] == 1.0
    assert result.details["metric"] == "entity_f1"


def test_ner_task_evaluate_partial() -> None:
    task = NERPromptTask(name="ner_news")
    samples = [
        Sample(inputs={}, labels=["PERSON: Nico", "LOC: Berlin"]),
    ]
    result = task.evaluate(["PERSON: Nico"], samples)
    assert 0.0 < result.accuracy < 1.0
    assert result.correct == 0
    assert result.total == 0


# -- TextPairClassificationTask --


def test_text_pair_task_prompt() -> None:
    task = TextPairClassificationTask(
        name="nli",
        description="NLI",
        classes=["entailment", "contradiction", "neutral"],
    )
    sample = Sample(
        inputs={"text_a": "Hallo", "text_b": "Welt"},
        labels=["neutral"],
    )
    prompt = task.build_prompt(sample)
    assert "Hallo" in prompt[0]["content"]
    assert "Welt" in prompt[0]["content"]


def test_text_pair_task_evaluate() -> None:
    task = TextPairClassificationTask(
        name="nli",
        description="NLI",
        classes=["entailment", "contradiction", "neutral"],
    )
    samples = [
        Sample(inputs={}, labels=["entailment"]),
        Sample(inputs={}, labels=["contradiction"]),
    ]
    result = task.evaluate(["entailment", "wrong"], samples)
    assert result.correct == 1
    assert result.accuracy == 0.5


def test_text_pair_task_details_has_metric() -> None:
    task = TextPairClassificationTask(
        name="nli",
        description="NLI",
        classes=["entailment", "contradiction", "neutral"],
    )
    samples = [Sample(inputs={}, labels=["entailment"])]
    result = task.evaluate(["entailment"], samples)
    assert result.details["metric"] == "accuracy"


# -- TextTripleClassificationTask --


def test_text_triple_task_prompt() -> None:
    task = TextTripleClassificationTask(
        name="query_ad",
        description="Query Ad",
        classes=["relevant", "irrelevant"],
    )
    sample = Sample(
        inputs={"text_a": "query", "text_b": "title", "text_c": "body"},
        labels=["relevant"],
    )
    prompt = task.build_prompt(sample)
    assert "query" in prompt[0]["content"]
    assert "title" in prompt[0]["content"]
    assert "body" in prompt[0]["content"]


def test_text_triple_task_prompt_no_typo() -> None:
    """Prompt uses correct spelling 'Beurteile' not 'Beurtele'."""
    task = TextTripleClassificationTask(
        name="query_ad",
        description="Query Ad",
        classes=["relevant", "irrelevant"],
    )
    sample = Sample(
        inputs={"text_a": "query", "text_b": "title", "text_c": "body"},
        labels=["relevant"],
    )
    prompt = task.build_prompt(sample)
    assert "Beurteile" in prompt[0]["content"]
    assert "Beurtele" not in prompt[0]["content"]


def test_text_triple_task_evaluate() -> None:
    task = TextTripleClassificationTask(
        name="query_ad",
        description="Query Ad",
        classes=["relevant", "irrelevant"],
    )
    samples = [
        Sample(inputs={}, labels=["relevant"]),
        Sample(inputs={}, labels=["irrelevant"]),
    ]
    result = task.evaluate(["relevant", "wrong"], samples)
    assert result.correct == 1
    assert result.accuracy == 0.5
    assert result.details["metric"] == "accuracy"


# -- Registry --


def test_registry_has_all_tasks() -> None:
    assert len(ALL_TASKS) >= 19


def test_registry_task_definitions() -> None:
    for name in ALL_TASKS:
        assert name in TASK_DEFINITIONS
        task_def, loader, data_subdir = TASK_DEFINITIONS[name]
        assert task_def is not None
        assert loader is not None
        assert data_subdir is not None
