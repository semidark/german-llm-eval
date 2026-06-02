# Session Notes

## Design Decisions
- Dropped LightEval for custom Python evaluator using SuperGLEBer datasets directly
- Include all 5 NER tasks due to GraphRAG relevance; prompt-based entity extraction instead of token-level BIO tagging
- Unified `Sample` dataclass: `inputs: dict[str, Any]`, `labels: list[str]`
- Prompt templates live in `TaskDefinition` subclasses (no per-task template files)
- Default German-language prompts built into classification tasks
- Async `APIClient` with `asyncio.TaskGroup` concurrency for all API calls
- `pyarrow.ipc.open_stream` for arrow datasets (faster, fewer deps than `datasets` lib)

## Bugs Fixed During Build

### Loader Extension Discovery
All loaders originally hardcoded extensions (`.tsv`, `.txt`). Added `_find_split_file()` to auto-discover `.tsv`, `.csv`, `.txt` in that order. Also added row-length guards to skip malformed rows.

### Germeval 2021 CSV Format
`toxic_comments`, `engaging_comments`, `factclaiming_comments` tasks have comma-separated CSV data (not TSV). Switched loaders from `TSVClassificationLoader` to `CSVClassificationLoader(delimiter=",")`.

### MLQA Arrow Format
MLQA data is `.arrow` format. Changed `QALoader(is_arrow=False)` -> `is_arrow=True`.

### Polarity Task Type
Data contains 3 classes (neutral=1681, negative=780, positive=105). Changed from `BinaryClassificationTask` to `MultiClassificationTask`.

### Offensive Lang Labels
Original classes were lowercase English (`offensive`, `non-offensive`). Actual labels are uppercase: `OTHER`, `ABUSE`, `INSULT`, `PROFANITY`.

### QALoader Timeout Issue
`datasets.load_from_disk()` on HF arrow datasets caused timeouts (tries to fetch from hub). Fixed by using `pyarrow.ipc.open_stream()` directly on the `.arrow` file.

## Data Format Reference

| Task | File Format | Delimiter | Key Columns |
|---|---|---|---|
| germanquad | Arrow IPC | N/A | `context`, `question`, `answers.text` |
| mlqa | Arrow IPC | N/A | `context`, `question`, `answers.text` |
| polarity | TSV | tab | col 1=text, col 3=label |
| massive_intents | TXT | tab | col 0=text, col 1=intent_id (numeric) |
| offensive_lang | TXT | tab | col 0=text, col 2=label |
| Germeval 2021 tasks | CSV | comma | col 0=id, col 1=text, col 2-4=labels |
| flausch | CSV | comma | col 2=comment, col 3=flausch (yes/no) |
| nli / pawsx / quest_ans / news_class | TXT | tab | text_a, text_b, label |
| query_ad | TXT | tab | query, title, ad_text, label |

## Remaining Work Items
- YAML config file support for reusable evaluation setups
- ~~NER: parse LLM free-text output -> compare against gold entities with `compute_f1_entities`~~ ✅ Done (2025-06-02)
- ~~QA: exact match + normalized EM scoring against gold answers~~ ✅ Done (2025-06-02)
- Mock API backend for offline testing
- End-to-end integration test with real API
