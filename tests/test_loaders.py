from __future__ import annotations

from pathlib import Path

from german_llm_eval.loaders.base import (
    CSVClassificationLoader,
    NERBIOLOnlyLoader,
    QALoader,
    Sample,
    TextPairLoader,
    TextTripleLoader,
    TSVClassificationLoader,
    TSVTextPairLoader,
)


def test_conll_parser_removed() -> None:
    """CONLLParser class has been removed (dead code)."""
    from german_llm_eval.loaders import base

    assert not hasattr(base, "CONLLParser")


# -- Sample --


def test_sample_string_label() -> None:
    s = Sample(inputs={"text": "hallo"}, labels="pos")
    assert s.labels == ["pos"]


def test_sample_list_label() -> None:
    s = Sample(inputs={"text": "hallo"}, labels=["pos", "neg"])
    assert s.labels == ["pos", "neg"]


# -- TSVClassificationLoader --


def test_tsv_loader_basic(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text("Hallo Welt\tpos\nGuten Tag\tneg\n")
    loader = TSVClassificationLoader(text_col=0, label_col=1)
    samples = loader.load(tmp_path, split="test")
    assert len(samples) == 2
    assert samples[0].inputs["text"] == "Hallo Welt"
    assert samples[0].labels == ["pos"]
    assert samples[1].labels == ["neg"]


def test_tsv_loader_header(tmp_path: Path) -> None:
    (tmp_path / "test.tsv").write_text("text\tlabel\nHallo\tpos\nWelt\tneg\n")
    loader = TSVClassificationLoader(text_col=0, label_col=1, has_header=True)
    samples = loader.load(tmp_path, split="test")
    assert len(samples) == 2
    assert samples[0].inputs["text"] == "Hallo"


def test_tsv_loader_short_rows_skipped(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text("Hallo\tpos\nshort\nWelt\tneg\n")
    loader = TSVClassificationLoader(text_col=0, label_col=1)
    samples = loader.load(tmp_path, split="test")
    assert len(samples) == 2


def test_tsv_loader_not_found(tmp_path: Path) -> None:
    loader = TSVClassificationLoader(text_col=0, label_col=1)
    try:
        loader.load(tmp_path, split="test")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# -- CSVClassificationLoader --


def test_csv_loader_comma(tmp_path: Path) -> None:
    (tmp_path / "test.csv").write_text(
        "text;label\nHallo;pos\nWelt;neg\n",
    )
    loader = CSVClassificationLoader(text_col=0, label_col=1, delimiter=";")
    samples = loader.load(tmp_path, split="test")
    assert len(samples) == 2
    assert samples[0].inputs["text"] == "Hallo"


def test_csv_loader_comma_delimiter(tmp_path: Path) -> None:
    (tmp_path / "test.csv").write_text(
        'text,label\n"Hello World",pos\n"Goodbye",neg\n',
    )
    loader = CSVClassificationLoader(text_col=0, label_col=1, delimiter=",")
    samples = loader.load(tmp_path, split="test")
    assert len(samples) == 2


def test_csv_loader_quoted_comma(tmp_path: Path) -> None:
    """CSV with commas inside quoted fields are parsed correctly."""
    (tmp_path / "test.csv").write_text(
        'text,label\n"Hello, World",pos\n"Good-bye, friend",neg\n',
    )
    loader = CSVClassificationLoader(text_col=0, label_col=1, delimiter=",")
    samples = loader.load(tmp_path, split="test")
    assert len(samples) == 2
    assert samples[0].inputs["text"] == "Hello, World"
    assert samples[0].labels == ["pos"]
    assert samples[1].inputs["text"] == "Good-bye, friend"


# -- TextPairLoader --


def test_text_pair_loader(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text(
        "Text A\tText B\tentailment\nHello\tHi\tcontradiction\n"
    )
    loader = TextPairLoader(label_col=2)
    samples = loader.load(tmp_path, split="test")
    assert len(samples) == 2
    assert samples[0].inputs["text_a"] == "Text A"
    assert samples[0].inputs["text_b"] == "Text B"
    assert samples[0].labels == ["entailment"]


# -- TextTripleLoader --


def test_text_triple_loader(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text("Query\tTitle\tBody\tlabel\nFoo\tBar\tBaz\t1\n")
    loader = TextTripleLoader(label_col=3)
    samples = loader.load(tmp_path, split="test")
    assert len(samples) == 2
    assert samples[1].inputs["text_a"] == "Foo"
    assert samples[1].inputs["text_b"] == "Bar"
    assert samples[1].inputs["text_c"] == "Baz"
    assert samples[1].labels == ["1"]


# -- NERBIOLOnlyLoader --


def test_ner_bio_loader_basic(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text(
        "Nico\tB-PER\nist\tO\nbei\tO\nBerlin\tB-LOC\nBrandenburg\tI-LOC\n\n"
    )
    loader = NERBIOLOnlyLoader(text_col=0, tag_col=1)
    samples = loader.load(tmp_path, split="test")
    assert len(samples) == 1
    assert samples[0].inputs["text"] == "Nico ist bei Berlin Brandenburg"
    assert "PER: Nico" in samples[0].labels
    assert "LOC: Berlin Brandenburg" in samples[0].labels


def test_ner_bio_loader_no_entities(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text("Hallo\tO\nWelt\tO\n\n")
    loader = NERBIOLOnlyLoader(text_col=0, tag_col=1)
    samples = loader.load(tmp_path, split="test")
    assert len(samples) == 1
    assert samples[0].labels == ["<no-entities>"]


def test_ner_bio_loader_multiple_sentences(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text("Nico\tB-PER\n\nBerlin\tB-LOC\n")
    loader = NERBIOLOnlyLoader(text_col=0, tag_col=1)
    samples = loader.load(tmp_path, split="test")
    assert len(samples) == 2
    assert "PER: Nico" in samples[0].labels
    assert "LOC: Berlin" in samples[1].labels


def test_ner_bio_loader_short_row_skipped(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text("Nico\tB-PER\norphan\nBerlin\tB-LOC\n\n")
    loader = NERBIOLOnlyLoader(text_col=0, tag_col=1)
    samples = loader.load(tmp_path, split="test")
    assert len(samples) == 1
    assert "PER: Nico" in samples[0].labels
    assert "LOC: Berlin" in samples[0].labels


# -- TSVTextPairLoader --


def test_tsv_text_pair_loader(tmp_path: Path) -> None:
    (tmp_path / "test.tsv").write_text(
        "id\tsentence1\tsentence2\tlabel\n1\tHallo\tHallo\t1\n2\tHallo\tWelt\t0\n"
    )
    loader = TSVTextPairLoader(text_a_col=1, text_b_col=2, label_col=3, has_header=True)
    samples = loader.load(tmp_path, split="test")
    assert len(samples) == 2
    assert samples[0].inputs["text_a"] == "Hallo"
    assert samples[0].inputs["text_b"] == "Hallo"
    assert samples[0].labels == ["1"]
    assert samples[1].inputs["text_a"] == "Hallo"
    assert samples[1].inputs["text_b"] == "Welt"
    assert samples[1].labels == ["0"]


# -- QALoader --


def test_qa_loader_none_answers(tmp_path: Path) -> None:
    """QALoader handles None answers gracefully."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    table = pa.table(
        {
            "context": ["Some context here"],
            "question": ["What is the answer?"],
            "answers": pa.array([None], type=pa.large_string()),
        }
    )
    with open(tmp_path / "data-00000-of-00001.arrow", "wb") as f:
        with ipc.new_stream(f, table.schema) as writer:
            writer.write_table(table)

    loader = QALoader(is_arrow=True)
    samples = loader.load(tmp_path)
    assert len(samples) == 1
    assert samples[0].inputs["context"] == "Some context here"
    assert samples[0].inputs["question"] == "What is the answer?"
    assert samples[0].labels == []


def test_qa_loader_valid_answers(tmp_path: Path) -> None:
    """QALoader loads valid arrow QA data."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    table = pa.table(
        {
            "context": ["Berlin ist die Hauptstadt."],
            "question": ["Was ist die Hauptstadt?"],
            "answers": pa.array(
                [{"text": ["Berlin"], "answer_start": [0]}],
                type=pa.struct(
                    [
                        pa.field("text", pa.list_(pa.large_string())),
                        pa.field("answer_start", pa.list_(pa.int64())),
                    ]
                ),
            ),
        }
    )
    with open(tmp_path / "data-00000-of-00001.arrow", "wb") as f:
        with ipc.new_stream(f, table.schema) as writer:
            writer.write_table(table)

    loader = QALoader(is_arrow=True)
    samples = loader.load(tmp_path)
    assert len(samples) == 1
    assert samples[0].labels == ["Berlin"]
