# AGENTS.md -- german-llm-eval

## Overview
Custom API-based evaluator for German LLMs using SuperGLEBer benchmark datasets. Targets OpenAI-compatible `chat/completions` endpoints, focused on RAG and GraphRAG evaluation.

## Quick Start
```sh
uv sync                              # install deps
german-llm-eval --list-tasks         # see all 19 tasks
german-llm-eval --model gpt-4o --base-url https://api.openai.com/v1 \
  --api-key $OPENAI_API_KEY --tasks germanquad,polarity --max-samples 50
```

## Build / Lint / Test
```sh
uv run ruff check src/ tests/        # lint
uv run ruff format src/ tests/       # format
uv run pytest tests/                  # all tests
```

## Data Source
All data lives at `~/src/SuperGLEBer/data`. The `--data-root` CLI flag overrides this default. Each task's registry entry specifies a subdirectory under this root.

## Architecture
```
src/german_llm_eval/
├── cli.py              # argparse entry point, wired as console script
├── client.py           # async APIClient (semaphore batching, retry/backoff)
├── evaluator.py        # Evaluator class: task runner, rich scorecard table, JSON export
├── loaders/base.py     # format-specific data loaders
├── metrics.py          # accuracy, token F1, entity F1
└── tasks/
    ├── base.py         # TaskDefinition subclasses with German prompt templates
    └── registry.py     # 19-task registry (task_def, loader, data_subdir tuples)
```

## Loaders (`loaders/base.py`)
| Loader | Format | Used By |
|---|---|---|
| `QALoader(is_arrow=True)` | HuggingFace Arrow IPC | germanquad, mlqa |
| `CSVClassificationLoader` | CSV with configurable delimiter | Germeval 2021 tasks, flausch |
| `TSVClassificationLoader` | TSV/TXT (auto-discovers extension) | polarity, massive_intents, offensive_lang |
| `TextPairLoader` | TXT tab-separated pairs + label | nli, quest_ans, news_class |
| `TSVTextPairLoader` | TSV tab-separated pairs, optional header | pawsx |
| `TextTripleLoader` | TXT tab-separated triples + label | query_ad |

## Key Gotchas
- **QALoader**: uses `pyarrow.ipc.open_stream`, NOT `datasets.load_from_disk` (avoids HF timeout)
- **Germeval 2021** CSVs use comma delimiter -- loader must be `CSVClassificationLoader(delimiter=",")`
- **polarity**: 3 classes (positive/negative/neutral), registered as `MultiClassificationTask`
- **offensive_lang**: labels are uppercase (`OTHER`, `ABUSE`, `INSULT`, `PROFANITY`)
- **NER tasks**: prompt-based entity extraction, NOT token-level BIO tagging
- **pawsx**: uses `TSVTextPairLoader(has_header=True)`, not `TextPairLoader`
- All loaders auto-discover file extension (`.tsv`, `.csv`, `.txt`) and skip short rows
- **Evaluator**: catches `BaseExceptionGroup` from `TaskGroup` to skip failed tasks gracefully
- **Scorecard Metric column**: reads from `TaskResult.details["metric"]`; each task's `evaluate()` must write it

## Code Style
- Python 3.12+, modern type hints (`list[str]`, `str | None`)
- ruff format + check; line length 88
- Google-style docstrings, snake_case functions, PascalCase classes
- `loguru` for logging, never `print()` for operational output
- pytest with Arrange-Act-Assert pattern

## Remaining Work
- [ ] YAML config file support for reusable eval setups
- [x] NER evaluation: entity-level F1 implemented in `NERPromptTask`.
  **Note:** Response parser supports `TYPE: ENTITY`, `(TYPE) entity`, and `entity (TYPE)` formats.
- [x] QA evaluation: normalized exact match with NFKC, `ß`→`ss`, punctuation stripping, whitespace collapsing.
- [ ] Mock API backend for offline testing
- [ ] End-to-end integration test with real API

## Changes Log
- **2025-06-02**: Removed `SyncAPIClient` (dead code), added NER regex fallback parser,
  QA normalized exact match, `QALoader` `.get()` safety, removed unused `datasets`/`pyyaml` deps.
- **2025-06-02**: Scorecard metric fix (Issue 1), ExceptionGroup graceful skip (Issue 2),
  BinaryClassificationTask inverse label map, QuestAnsTask sample.labels fix, Polarity classes= fix,
  entity F1 case normalization, MultiClassificationTask _label_map fix, pawsx TSVTextPairLoader,
  quest_ans TextPairClassificationTask, CLI comma-separated tasks, API key ValueError,
  CSVClassificationLoader csv.reader, dead code removal (CONLLParser), ner_legal entity types,
  loguru integration, asyncio.TaskGroup concurrency, QALoader None guard, parse_args extraction.
