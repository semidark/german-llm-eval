# German LLM Evaluator

Custom API-based evaluation tool for German-language LLMs using SuperGLEBer benchmark datasets. Designed for evaluating models behind OpenAI-compatible `chat/completions` endpoints, with focus on RAG and GraphRAG use cases.

## Quick Start

```sh
uv sync                              # install deps
uv run german-llm-eval --list-tasks  # see all 19 tasks
```

### Sanity Check (Recommended)

Before running a full evaluation, verify your setup with a small subset:

```sh
uv run german-llm-eval \
  --base-url http://localhost:11434/v1 \
  --api-key ollama \
  --model qwen3.5:9b \
  --tasks polarity,germanquad,nli \
  --max-samples 20 \
  --timeout 300 \
  --concurrency 2
```

This tests one classification task (`polarity`), one QA task (`germanquad`), and one text-pair task (`nli`) with just 20 samples each.

## Usage

### Local Models (Ollama)

```sh
uv run german-llm-eval \
  --base-url http://localhost:11434/v1 \
  --api-key ollama \
  --model qwen3.5:9b \
  --timeout 300 \
  --concurrency 2 \
  --output results/qwen3.5-9b.json
```

> **Tip:** For local models, use `--concurrency 2` to avoid overwhelming your GPU.

### OpenAI API

```sh
uv run german-llm-eval \
  --model gpt-4o \
  --api-key "$OPENAI_API_KEY" \
  --tasks germanquad,polarity \
  --max-samples 50
```

### Output to JSON

```sh
uv run german-llm-eval --model ... --output results.json
```

### List Available Tasks

```sh
uv run german-llm-eval --list-tasks
```

### CLI Options
| Flag | Description |
|---|---|
| `--base-url` | API base URL (default: `https://api.openai.com/v1`) |
| `--model` | Model name/ID to evaluate |
| `--api-key` | API key for authentication |
| `--tasks` | Comma-separated task names (default: all 19) |
| `--split` | Data split to use (default: `test`) |
| `--max-samples` | Max samples per task (0 = unlimited) |
| `--temperature` | Sampling temperature (default: `0.0`) |
| `--timeout` | Request timeout in seconds (default: `60`) |
| `--concurrency` | Parallel API requests (default: `5`) |
| `--output` | Path for JSON results export |

## Tasks

| Task | Type | Samples | Description |
|---|---|---|---|
| germanquad | QA | 2204 | German reading comprehension (SQuAD-style) |
| mlqa | QA | 4517 | Multi-lingual QA (German subset) |
| polarity | Classification | 2566 | Sentiment (positive/negative/neutral) |
| massive_intents | Classification | 1652 | Intent classification (126 classes) |
| offensive_lang | Classification | 3532 | Offensive language detection (4 classes) |
| flausch_classification | Classification | 9230 | Fake news detection |
| toxic_comments | Classification | 944 | Toxic comment detection |
| engaging_comments | Classification | 944 | Engaging comment classification |
| factclaiming_comments | Classification | 944 | Fact-claiming comment detection |
| nli | Pair Classification | 5010 | Natural language inference |
| pawsx | Pair Classification | 2000 | Paraphrase identification |
| quest_ans | Pair Classification | 10000 | Question-answer relevance |
| news_class | Pair Classification | 10000 | News text relationship |
| query_ad | Triple Classification | 10000 | Query-title-ad relevance |
| ner_europarl | NER | 858 | Named entities (EuParl corpus) |
| ner_legal | NER | 6673 | Named entities (legal documents) |
| ner_news | NER | 3007 | Named entities (news articles) |
| ner_biofid | NER | 1584 | Named entities (biomedical, BIOFID) |
| ner_wiki_news | NER | 5100 | Named entities (German Wikipedia news) |
