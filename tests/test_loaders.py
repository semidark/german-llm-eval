from __future__ import annotations

from pathlib import Path

from german_llm_eval.loaders.base import (
    CSVClassificationLoader,
    NERBIOLOnlyLoader,
    Sample,
    TextPairLoader,
    TextTripleLoader,
    TSVClassificationLoader,
)


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
