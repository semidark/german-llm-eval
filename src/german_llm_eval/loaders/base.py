from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

from loguru import logger


class Sample:
    __slots__ = ("inputs", "labels")

    def __init__(self, inputs: dict[str, Any], labels: list[str] | str) -> None:
        self.inputs = inputs
        self.labels = [labels] if isinstance(labels, str) else labels


class TaskLoader(abc.ABC):
    @abc.abstractmethod
    def load(self, data_dir: Path, split: str = "test") -> list[Sample]: ...


class QALoader(TaskLoader):
    """HuggingFace arrow format QA (GermanQuAD, MLQA)."""

    def __init__(self, is_arrow: bool = True) -> None:
        self._is_arrow = is_arrow

    def load(self, data_dir: Path, split: str = "test") -> list[Sample]:
        if self._is_arrow:
            import pyarrow.ipc as ipc  # type: ignore[import-unavailable]

            arrow_files = sorted(data_dir.glob("*.arrow"))
            if not arrow_files:
                arrow_files = list(data_dir.glob("**/*.arrow"))
            records: list[dict[str, Any]] = []
            for fpath in arrow_files:
                with open(fpath, "rb") as f:
                    reader = ipc.open_stream(f)
                    table = reader.read_all()
                records.extend(table.to_pylist())
        else:
            import pyarrow.parquet as pq  # type: ignore[import-unavailable]

            files = sorted(data_dir.glob("*.parquet"))
            if not files:
                files = list(data_dir.glob("**/*.parquet"))
            table = pq.read_table(
                files[0] if files else data_dir / "data-00000-of-00001.parquet"
            )
            records = table.to_pylist()

        rows: list[Sample] = []
        for record in records:
            answers = record.get("answers") or {}
            answer_texts = (
                answers.get("text", []) if isinstance(answers, dict) else str(answers)
            )
            ctx = record.get("context", "")
            qst = record.get("question", "")
            if not ctx and not qst:
                continue
            rows.append(
                Sample(
                    inputs={
                        "context": ctx,
                        "question": qst,
                    },
                    labels=(
                        answer_texts
                        if isinstance(answer_texts, list)
                        else [str(answer_texts)]
                    ),
                )
            )
        return rows


class TSVClassificationLoader(TaskLoader):
    """Tab-separated files with text + label columns.

    Args:
        text_col: 0-based column index for text.
        label_col: 0-based column index for label.
        has_header: Whether the first line is a header.
        multi_label: If True, split labels on whitespace.
    """

    def __init__(
        self,
        text_col: int,
        label_col: int,
        has_header: bool = False,
        multi_label: bool = False,
        delimiter: str = "\t",
    ) -> None:
        self._text_col = text_col
        self._label_col = label_col
        self._has_header = has_header
        self._multi_label = multi_label
        self._delimiter = delimiter

    def load(self, data_dir: Path, split: str = "test") -> list[Sample]:
        split_file = self._find_split_file(data_dir, split)
        rows: list[Sample] = []
        with open(split_file, encoding="utf-8") as fh:
            lines = fh.readlines()
        if self._has_header:
            lines = lines[1:]
        for line in lines:
            parts = line.strip().split(self._delimiter)
            if len(parts) <= max(self._text_col, self._label_col):
                continue
            text = parts[self._text_col]
            label_str = parts[self._label_col]
            labels = (
                [lb.strip() for lb in label_str.split()]
                if self._multi_label
                else [label_str]
            )
            rows.append(Sample(inputs={"text": text}, labels=labels))
        return rows

    def _find_split_file(self, data_dir: Path, split: str) -> Path:
        for ext in (".tsv", ".csv", ".txt"):
            candidate = data_dir / f"{split}{ext}"
            if candidate.exists():
                return candidate
        logger.warning(f"No {split} file found in {data_dir} (tried .tsv, .csv, .txt)")
        raise FileNotFoundError(
            f"No {split} file found in {data_dir} (tried .tsv, .csv, .txt)"
        )


class CSVClassificationLoader(TSVClassificationLoader):
    """CSV/semicolon-separated files, parsed with the csv module for safety."""

    def __init__(
        self,
        text_col: int,
        label_col: int,
        has_header: bool = True,
        delimiter: str = ";",
    ) -> None:
        super().__init__(
            text_col=text_col,
            label_col=label_col,
            has_header=has_header,
            delimiter=delimiter,
        )

    def load(self, data_dir: Path, split: str = "test") -> list[Sample]:
        import csv

        split_file = self._find_split_file(data_dir, split)
        rows: list[Sample] = []
        with open(split_file, encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=self._delimiter)
            for i, parts in enumerate(reader):
                if i == 0 and self._has_header:
                    continue
                if len(parts) <= max(self._text_col, self._label_col):
                    continue
                rows.append(
                    Sample(
                        inputs={"text": parts[self._text_col]},
                        labels=[parts[self._label_col]],
                    )
                )
        return rows


class TXTTabClassificationLoader(TaskLoader):
    """Plain TXT files with tab-separated text(s) + label."""

    def __init__(
        self,
        text_cols: list[int],
        label_col: int,
    ) -> None:
        self._text_cols = text_cols
        self._label_col = label_col

    def load(self, data_dir: Path, split: str = "test") -> list[Sample]:
        split_file = self._find_split_file(data_dir, split)
        rows: list[Sample] = []
        with open(split_file, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(self._text_cols + [self._label_col]):
                    continue
                inputs: dict[str, str] = {}
                col_names = ["text_a", "text_b", "text_c"]
                for idx, col in enumerate(self._text_cols):
                    inputs[col_names[idx]] = parts[col]
                rows.append(
                    Sample(
                        inputs=inputs,
                        labels=[parts[self._label_col]],
                    )
                )
        return rows

    def _find_split_file(self, data_dir: Path, split: str) -> Path:
        for ext in (".txt", ".tsv", ".csv"):
            candidate = data_dir / f"{split}{ext}"
            if candidate.exists():
                return candidate
        logger.warning(f"No {split} file found in {data_dir} (tried .txt, .tsv, .csv)")
        raise FileNotFoundError(
            f"No {split} file found in {data_dir} (tried .txt, .tsv, .csv)"
        )


class TextPairLoader(TXTTabClassificationLoader):
    """Two-text pair classification (NLI, News_Class, Quest_Ans, etc.)."""

    def __init__(self, label_col: int = 2) -> None:
        super().__init__(text_cols=[0, 1], label_col=label_col)


class TSVTextPairLoader(TaskLoader):
    """TSV text-pair classification with header support (PAWS-X).

    Args:
        text_a_col: 0-based column index for first text.
        text_b_col: 0-based column index for second text.
        label_col: 0-based column index for label.
        has_header: Whether the first line is a header.
    """

    def __init__(
        self,
        text_a_col: int = 0,
        text_b_col: int = 1,
        label_col: int = 2,
        has_header: bool = False,
    ) -> None:
        self._text_a_col = text_a_col
        self._text_b_col = text_b_col
        self._label_col = label_col
        self._has_header = has_header

    def load(self, data_dir: Path, split: str = "test") -> list[Sample]:
        split_file = self._find_split_file(data_dir, split)
        rows: list[Sample] = []
        with open(split_file, encoding="utf-8") as fh:
            lines = fh.readlines()
        if self._has_header:
            lines = lines[1:]
        for line in lines:
            parts = line.rstrip("\n").split("\t")
            max_col = max(self._text_a_col, self._text_b_col, self._label_col)
            if len(parts) <= max_col:
                continue
            rows.append(
                Sample(
                    inputs={
                        "text_a": parts[self._text_a_col],
                        "text_b": parts[self._text_b_col],
                    },
                    labels=[parts[self._label_col]],
                )
            )
        return rows

    def _find_split_file(self, data_dir: Path, split: str) -> Path:
        for ext in (".tsv", ".txt", ".csv"):
            candidate = data_dir / f"{split}{ext}"
            if candidate.exists():
                return candidate
        logger.warning(f"No {split} file found in {data_dir} (tried .tsv, .txt, .csv)")
        raise FileNotFoundError(
            f"No {split} file found in {data_dir} (tried .tsv, .txt, .csv)"
        )


class TextTripleLoader(TXTTabClassificationLoader):
    """Three-text triple classification (Query_Ad)."""

    def __init__(self, label_col: int = 3) -> None:
        super().__init__(text_cols=[0, 1, 2], label_col=label_col)


class NERBIOLOnlyLoader(TaskLoader):
    """Loads BIO/BIOES NER files and aggregates tokens into entity spans.

    Supports multiple file formats:
    - .txt with whitespace-separated columns (conllu-like or plain token\\ttag)
    - .tsv with tab-separated columns
    - .conll format
    """

    def __init__(
        self, text_col: int = 0, tag_col: int = 1, tag_prefix_sep: str = "-"
    ) -> None:
        self._text_col = text_col
        self._tag_col = tag_col
        self._sep = tag_prefix_sep

    def _extract_entities(self, sentences: list[list[tuple[str, str]]]) -> list[Sample]:
        rows: list[Sample] = []
        for sent in sentences:
            entities: list[tuple[str, str]] = []
            current_type: str | None = None
            current_tokens: list[str] = []

            for token, tag in sent:
                if tag == "O":
                    if current_type is not None:
                        entities.append((current_type, " ".join(current_tokens)))
                        current_type = None
                        current_tokens = []
                    continue

                if tag.startswith("B-") or tag.startswith("B2"):
                    if current_type is not None:
                        entities.append((current_type, " ".join(current_tokens)))
                    current_type = tag.split(self._sep, 1)[1]
                    current_tokens = [token]
                elif tag.startswith("I-") or tag.startswith("I2"):
                    ent_type = tag.split(self._sep, 1)[1]
                    if current_type == ent_type:
                        current_tokens.append(token)
                    else:
                        if current_type is not None:
                            entities.append((current_type, " ".join(current_tokens)))
                        current_type = ent_type
                        current_tokens = [token]
                else:
                    if current_type is not None:
                        entities.append((current_type, " ".join(current_tokens)))
                    current_type = None
                    current_tokens = []

            if current_type is not None:
                entities.append((current_type, " ".join(current_tokens)))

            text = " ".join(t for t, _ in sent)
            label_list = [f"{etype}: {etxt}" for etype, etxt in entities]
            rows.append(
                Sample(
                    inputs={"text": text},
                    labels=label_list if label_list else ["<no-entities>"],
                )
            )
        return rows

    def load(self, data_dir: Path, split: str = "test") -> list[Sample]:
        candidates: list[Path] = []
        for ext in (
            f"{split}.txt",
            f"{split}.tsv",
            f"{split}.conll",
            f"{split}.conllu",
        ):
            p = data_dir / ext
            if p.exists():
                candidates.append(p)

        if not candidates:
            matches = list(data_dir.glob(f"*{split}*"))
            if matches:
                candidates.append(matches[0])

        if not candidates:
            logger.warning(f"No {split} file found in {data_dir}")
            raise FileNotFoundError(f"No {split} file found in {data_dir}")

        sentences: list[list[tuple[str, str]]] = []
        current_sent: list[tuple[str, str]] = []

        with open(candidates[0], encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    if current_sent:
                        sentences.append(current_sent)
                        current_sent = []
                    continue
                parts = stripped.split()
                text = parts[self._text_col]
                tag = parts[self._tag_col]
                current_sent.append((text, tag))

        if current_sent:
            sentences.append(current_sent)

        return self._extract_entities(sentences)
