from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from german_llm_eval.loaders.base import Sample


@dataclass
class TaskDefinition(ABC):
    name: str
    description: str
    metric: str

    @abstractmethod
    def build_prompt(self, sample: Sample) -> list[dict[str, str]]: ...

    @abstractmethod
    def evaluate(self, responses: list[str], samples: list[Sample]) -> "TaskResult": ...


from german_llm_eval.metrics import TaskResult, compute_f1_entities  # noqa: E402


class QATask(TaskDefinition):
    """Open-domain QA. Model gets context + question, returns answer string."""

    metric = "exact_match"

    def __init__(
        self, name: str, description: str | None = None, metric: str | None = None
    ) -> None:
        super().__init__(
            name=name,
            description=description or f"Question Answering ({name})",
            metric=metric or self.metric,
        )

    def build_prompt(self, sample: Sample) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": (
                    "Beantworte die folgende Frage basierend auf dem gegebenen Kontext. "
                    "Antworte nur mit der Antwort, nichts anderes.\n\n"
                    f"Kontext: {sample.inputs['context']}\n\n"
                    f"Frage: {sample.inputs['question']}"
                ),
            }
        ]

    def evaluate(self, responses: list[str], samples: list[Sample]) -> TaskResult:
        exact_match = 0
        for resp, sample in zip(responses, samples):
            normalized_resp = resp.strip().lower()
            normalized_labels = {lb.strip().lower() for lb in sample.labels}
            if normalized_resp in normalized_labels:
                exact_match += 1

        accuracy = exact_match / len(samples) if samples else 0.0
        return TaskResult(
            name=self.name,
            correct=exact_match,
            total=len(samples),
            accuracy=accuracy,
            details={"exact_match_rate": accuracy},
        )


class BinaryClassificationTask(TaskDefinition):
    """Binary classification (two classes)."""

    metric = "accuracy"

    def __init__(
        self,
        name: str,
        description: str,
        label_map: dict[str, str],
        instructions: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description, metric=self.metric)
        self._label_map = label_map
        self._instructions = instructions

    def build_prompt(self, sample: Sample) -> list[dict[str, str]]:
        text = sample.inputs.get("text", "")
        class_options = " / ".join(self._label_map.values())
        prompt_text = (
            self._instructions
            or f"Klassifiziere den folgenden Text als eine der beiden Kategorien.\n\n"
            f"Text: {text}\n\n"
            f"Mögliche Antworten: {class_options}. Gib nur die Klasse aus."
        )
        return [{"role": "user", "content": prompt_text}]

    def evaluate(self, responses: list[str], samples: list[Sample]) -> TaskResult:
        correct = 0
        for resp, sample in zip(responses, samples):
            normalized_resp = resp.strip().lower()
            label_set = {lb.strip().lower() for lb in sample.labels}
            mapped = {v.lower() for v in self._label_map.values()}
            if normalized_resp in label_set or normalized_resp in mapped:
                correct += 1

        total = len(samples)
        return TaskResult(
            name=self.name,
            correct=correct,
            total=total,
            accuracy=correct / total if total else 0.0,
            details={},
        )


class MultiClassificationTask(TaskDefinition):
    """Multi-class classification (text -> one label from set)."""

    metric = "accuracy"

    def __init__(
        self,
        name: str,
        description: str,
        classes: list[str],
        instructions: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description, metric=self.metric)
        self._classes = classes
        self._instructions = instructions

    def build_prompt(self, sample: Sample) -> list[dict[str, str]]:
        text = sample.inputs.get("text", "")
        class_options = " / ".join(self._classes)
        prompt_text = (
            self._instructions
            or f"Klassifiziere den folgenden Text in eine der folgenden Kategorien.\n\n"
            f"Text: {text}\n\n"
            f"Mögliche Kategorien: {class_options}. Gib nur die Kategorie aus."
        )
        return [{"role": "user", "content": prompt_text}]

    def evaluate(self, responses: list[str], samples: list[Sample]) -> TaskResult:
        correct = 0
        for resp, sample in zip(responses, samples):
            normalized_resp = resp.strip().lower()
            label_set = {lb.strip().lower() for lb in sample.labels}
            if normalized_resp in label_set:
                correct += 1

        total = len(samples)
        return TaskResult(
            name=self.name,
            correct=correct,
            total=total,
            accuracy=correct / total if total else 0.0,
            details={"classes": self._classes},
        )


class NERPromptTask(TaskDefinition):
    """NER via prompt-based entity extraction.

    The model is asked to list all entities of a given text with their types.
    Evaluated via entity-level F1 (type + normalized text match).
    """

    metric = "entity_f1"

    ENTITY_TYPES: dict[str, list[str]] = {
        "ner_news": ["PERSON", "LOC", "ORG"],
        "ner_europarl": ["NC", "PC"],
        "ner_legal": ["PER", "LOC", "ORG", "GS", "RV", "UR", "LL"],
        "ner_biofid": ["PER", "LOC", "ORG", "OTHER"],
        "ner_wiki_news": ["MISC"],
    }

    def __init__(self, name: str, entity_types: list[str] | None = None) -> None:
        etypes = entity_types or self.ENTITY_TYPES.get(name, [])
        super().__init__(
            name=name,
            description=f"Named Entity Recognition ({name})",
            metric="entity_f1",
        )
        self._entity_types = etypes

    def build_prompt(self, sample: Sample) -> list[dict[str, str]]:
        text = sample.inputs["text"]
        return [
            {
                "role": "user",
                "content": (
                    "Erkenne alle benannten Entitäten im folgenden Text. "
                    "Gib jede Entität in der Form 'TYP: ENTITÄT' pro Zeile aus. "
                    f"Mögliche Typen: {', '.join(self._entity_types)}. "
                    "Wenn keine Entitäten vorhanden sind, gib '<no-entities>' aus.\n\n"
                    f"Text: {text}"
                ),
            }
        ]

    def _parse_response(self, response: str) -> list[tuple[str, str]]:
        entities: list[tuple[str, str]] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or line == "<no-entities>":
                continue
            if ":" in line:
                typ, ent = line.split(":", 1)
                entities.append((typ.strip(), ent.strip()))
        return entities

    def _parse_labels(self, labels: list[str]) -> list[tuple[str, str]]:
        entities: list[tuple[str, str]] = []
        for lbl in labels:
            lbl = lbl.strip()
            if lbl == "<no-entities>":
                continue
            if ":" in lbl:
                typ, ent = lbl.split(":", 1)
                entities.append((typ.strip(), ent.strip()))
        return entities

    def evaluate(self, responses: list[str], samples: list[Sample]) -> TaskResult:
        pred_entities_list = [self._parse_response(r) for r in responses]
        label_entities_list = [self._parse_labels(s.labels) for s in samples]
        prec, rec, f1 = compute_f1_entities(pred_entities_list, label_entities_list)
        return TaskResult(
            name=self.name,
            correct=0,
            total=0,
            accuracy=f1,
            metric_label=f"F1={f1:.4f}",
            details={
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "entity_types": self._entity_types,
            },
        )


class TextPairClassificationTask(TaskDefinition):
    """Two-text pair classification (NLI, similarity, QA relevance)."""

    metric = "accuracy"

    def __init__(
        self,
        name: str,
        description: str,
        classes: list[str],
        instructions: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description, metric=self.metric)
        self._classes = classes
        self._instructions = instructions

    def build_prompt(self, sample: Sample) -> list[dict[str, str]]:
        text_a = sample.inputs.get("text_a", "")
        text_b = sample.inputs.get("text_b", "")

        class_options = " / ".join(self._classes)
        prompt_text = (
            self._instructions
            or f"Klassifiziere die Beziehung zwischen den folgenden beiden Texten. "
            f"Mögliche Antworten: {class_options}. Gib nur die Klasse aus.\n\n"
            f"Text 1: {text_a}\nText 2: {text_b}"
        )
        return [{"role": "user", "content": prompt_text}]

    def evaluate(self, responses: list[str], samples: list[Sample]) -> TaskResult:
        correct = 0
        for resp, sample in zip(responses, samples):
            normalized_resp = resp.strip().lower()
            normalized_labels = {lb.strip().lower() for lb in sample.labels}
            if normalized_resp in normalized_labels:
                correct += 1
        return TaskResult(
            name=self.name,
            correct=correct,
            total=len(samples),
            accuracy=correct / len(samples) if samples else 0.0,
            details={"classes": self._classes},
        )


class TextTripleClassificationTask(TextPairClassificationTask):
    """Three-text classification (Query_Ad)."""

    def build_prompt(self, sample: Sample) -> list[dict[str, str]]:
        text_a = sample.inputs.get("text_a", "")
        text_b = sample.inputs.get("text_b", "")
        text_c = sample.inputs.get("text_c", "")
        class_options = " / ".join(self._classes)

        prompt_text = (
            f"Beurtele ob die folgende Suchanfrage zu einer guten Anzeige passt.\n\n"
            f"Suchanfrage: {text_a}\n"
            f"Anzeigentitel: {text_b}\n"
            f"Anzeigentext: {text_c}\n\n"
            f"Mögliche Antworten: {class_options}. Gib nur die Klasse aus."
        )
        return [{"role": "user", "content": prompt_text}]
