from __future__ import annotations

from german_llm_eval.loaders.base import (
    CSVClassificationLoader,
    NERBIOLOnlyLoader,
    QALoader,
    TextPairLoader,
    TextTripleLoader,
    TSVClassificationLoader,
    TSVTextPairLoader,
)
from german_llm_eval.tasks.base import (
    BinaryClassificationTask,
    MultiClassificationTask,
    NERPromptTask,
    QATask,
    TextPairClassificationTask,
    TextTripleClassificationTask,
)

TASK_DEFINITIONS: dict[str, tuple] = {}


def register(
    name: str,
    task_def: object,
    loader: object,
    data_subdir: str,
) -> None:
    TASK_DEFINITIONS[name] = (task_def, loader, data_subdir)


# --------------------------------------------------------------------------- #
# QA Tasks                                                                    #
# --------------------------------------------------------------------------- #

register(
    "germanquad",
    QATask("germanquad", "German QuA - German Question Answering", "exact_match"),
    QALoader(is_arrow=True),
    "GermanQuAD/test",
)

register(
    "mlqa",
    QATask("mlqa", "Multi-Lingual QA (German)", "exact_match"),
    QALoader(is_arrow=True),
    "XGlue/MLQA/test",
)

# --------------------------------------------------------------------------- #
# Text-Pair Classification                                                    #
# --------------------------------------------------------------------------- #

register(
    "nli",
    TextPairClassificationTask(
        "nli",
        "Natural Language Inference (German)",
        ["entailment", "contradiction", "neutral"],
    ),
    TextPairLoader(label_col=2),
    "XGlue/NLI",
)

register(
    "pawsx",
    TextPairClassificationTask(
        "pawsx",
        "PAWS-X Paraphrase Detection (German)",
        ["true", "false"],
    ),
    TSVTextPairLoader(text_a_col=1, text_b_col=2, label_col=3, has_header=True),
    "PAWSX",
)

register(
    "news_class",
    TextPairClassificationTask(
        "news_class",
        "News Classification (German)",
        ["news", "entertainment", "sports"],
    ),
    TextPairLoader(label_col=2),
    "XGlue/News_Class",
)

register(
    "quest_ans",
    TextPairClassificationTask(
        "quest_ans",
        "Question Answering Relevance (German)",
        ["true", "false"],
    ),
    TextPairLoader(label_col=2),
    "XGlue/Quest_Ans",
)

# --------------------------------------------------------------------------- #
# Triple Classification                                                       #
# --------------------------------------------------------------------------- #

register(
    "query_ad",
    TextTripleClassificationTask(
        "query_ad",
        "Query-Advertisement Relevance (German)",
        ["Good", "Bad"],
    ),
    TextTripleLoader(label_col=3),
    "XGlue/Query_Ad",
)

# --------------------------------------------------------------------------- #
# Single-Text Classification                                                  #
# --------------------------------------------------------------------------- #

register(
    "massive_intents",
    MultiClassificationTask(
        "massive_intents",
        "MASSIVE Intent Classification (German)",
        classes=[str(i) for i in range(126)],
    ),
    TSVClassificationLoader(text_col=0, label_col=1, has_header=False),
    "Massive/Intents",
)

register(
    "polarity",
    MultiClassificationTask(
        name="polarity",
        description="Sentiment classification (positive/negative/neutral)",
        classes=["positiv", "negativ", "neutral"],
    ),
    TSVClassificationLoader(text_col=1, label_col=3, has_header=False),
    "Germeval/2017",
)

register(
    "flausch_classification",
    BinaryClassificationTask(
        "flausch_classification",
        "Fake News Detection (German)",
        {"true": "true", "false": "false"},
    ),
    CSVClassificationLoader(text_col=2, label_col=3, has_header=True, delimiter=","),
    "Germeval/2025/FlauschErkennung/task1",
)

register(
    "offensive_lang",
    MultiClassificationTask(
        "offensive_lang",
        "Offensive Language Detection (German)",
        classes=["OTHER", "ABUSE", "INSULT", "PROFANITY"],
    ),
    TSVClassificationLoader(text_col=0, label_col=2, has_header=False),
    "Germeval/Offensive_Lang",
)

register(
    "toxic_comments",
    BinaryClassificationTask(
        "toxic_comments",
        "Toxic Comment Classification (German)",
        {"1": "true", "0": "false"},
    ),
    CSVClassificationLoader(text_col=1, label_col=2, has_header=True, delimiter=","),
    "Germeval/2021",
)

register(
    "engaging_comments",
    BinaryClassificationTask(
        "engaging_comments",
        "Engaging Comment Classification (German)",
        {"1": "true", "0": "false"},
    ),
    CSVClassificationLoader(text_col=1, label_col=3, has_header=True, delimiter=","),
    "Germeval/2021",
)

register(
    "factclaiming_comments",
    BinaryClassificationTask(
        "factclaiming_comments",
        "Fact-Claiming Comment Classification (German)",
        {"1": "true", "0": "false"},
    ),
    CSVClassificationLoader(text_col=1, label_col=4, has_header=True, delimiter=","),
    "Germeval/2021",
)

# --------------------------------------------------------------------------- #
# NER Tasks (prompt-based entity extraction)                                  #
# --------------------------------------------------------------------------- #

register(
    "ner_news",
    NERPromptTask("ner_news", entity_types=["PERSON", "LOC", "ORG"]),
    NERBIOLOnlyLoader(text_col=0, tag_col=1),
    "XGlue/NER_News",
)

register(
    "ner_europarl",
    NERPromptTask("ner_europarl", entity_types=["NC", "PC"]),
    NERBIOLOnlyLoader(text_col=0, tag_col=4),
    "NER/EuroParl",
)

register(
    "ner_legal",
    NERPromptTask(
        "ner_legal", entity_types=["PER", "LOC", "ORG", "GS", "RV", "UR", "LL"]
    ),
    NERBIOLOnlyLoader(text_col=0, tag_col=1),
    "NER/Legal",
)

register(
    "ner_biofid",
    NERPromptTask("ner_biofid", entity_types=["PER", "LOC", "ORG", "OTHER"]),
    NERBIOLOnlyLoader(text_col=0, tag_col=3),
    "NER/BioFID",
)

register(
    "ner_wiki_news",
    NERPromptTask("ner_wiki_news", entity_types=["MISC"]),
    NERBIOLOnlyLoader(text_col=1, tag_col=2),
    "NER/Wiki_News",
)

# --------------------------------------------------------------------------- #

ALL_TASKS = sorted(TASK_DEFINITIONS.keys())
