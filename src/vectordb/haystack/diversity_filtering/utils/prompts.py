"""RAG prompt templates for diversity filtering pipelines.

Provides dataset-specific prompt templates and document formatting utilities
for the RAG generation stage of diversity filtering pipelines.

Diverse retrieval results require carefully constructed prompts that guide
the LLM to synthesize information from multiple distinct sources. These
templates are optimized for different query types and domains.

Templates:
- TRIVIAQA_PROMPT: Optimized for factual trivia questions requiring concise,
  accurate answers based on retrieved passages.
- ARC_PROMPT: Supports multiple-choice reasoning with document evidence.
- POPQA_PROMPT: General-purpose template for popular/common questions.
- FACTSCORE_PROMPT: Structured fact-checking with verification/refutation.
- EARNINGS_CALLS_PROMPT: Financial domain template for earnings call analysis.

The format_documents() function prepares retrieved diverse documents for
injection into these prompts, handling various document formats.
"""

# Prompt template for answer generation given retrieved documents
RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided documents.

Question: {query}

Documents:
{documents}

Answer:"""

# Prompt template for TriviaQA
TRIVIAQA_PROMPT = """You are answering a trivia question. Use the provided documents to answer accurately and concisely.

Question: {query}

Retrieved Documents:
{documents}

Answer:"""

# Prompt template for ARC (AI2 Reasoning Challenge)
ARC_PROMPT = """You are answering a multiple-choice question. Use the provided documents to support your answer.

Question: {query}

Retrieved Documents:
{documents}

Answer:"""

# Prompt template for PopQA (Popular Questions)
POPQA_PROMPT = """You are answering a question about popular topics. Use the provided documents to answer clearly and helpfully.

Question: {query}

Retrieved Documents:
{documents}

Answer:"""

# Prompt template for FactScore (Fact Checking)
FACTSCORE_PROMPT = """You are fact-checking a statement. Use the provided documents to verify or refute the statement.

Statement: {query}

Retrieved Documents:
{documents}

Fact-check:"""

# Prompt template for Earnings Calls
EARNINGS_CALLS_PROMPT = """You are analyzing earnings call documents. Answer the question based on the provided excerpts.

Question: {query}

Retrieved Document Excerpts:
{documents}

Answer:"""

PROMPTS_BY_DATASET = {
    "triviaqa": TRIVIAQA_PROMPT,
    "arc": ARC_PROMPT,
    "popqa": POPQA_PROMPT,
    "factscore": FACTSCORE_PROMPT,
    "earnings_calls": EARNINGS_CALLS_PROMPT,
}


def get_prompt_template(dataset_name: str) -> str:
    """Get prompt template for a specific dataset.

    Args:
        dataset_name: Name of the dataset
            (triviaqa, arc, popqa, factscore, earnings_calls).

    Returns:
        Prompt template string.

    Raises:
        ValueError: If dataset not found.
    """
    if dataset_name not in PROMPTS_BY_DATASET:
        msg = f"Unknown dataset: {dataset_name}. Available: {list(PROMPTS_BY_DATASET.keys())}"
        raise ValueError(msg)
    return PROMPTS_BY_DATASET[dataset_name]


def format_documents(documents: list[dict]) -> str:
    """Format retrieved documents for RAG prompt.

    Args:
        documents: List of document dictionaries with 'content' or 'text' field.

    Returns:
        Formatted documents string.
    """
    formatted = []
    for i, doc in enumerate(documents, 1):
        content = doc.get("content") or doc.get("text") or str(doc)
        formatted.append(f"Document {i}: {content}")
    return "\n\n".join(formatted)
