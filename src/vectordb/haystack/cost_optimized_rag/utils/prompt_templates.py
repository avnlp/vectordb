"""RAG prompt templates for cost-optimized RAG pipelines.

Jinja2 templates optimized for minimal token usage while maintaining
effectiveness. Template selection affects both cost (token count) and
quality (response accuracy).

Token Usage Optimization:

    Template Selection Strategy:
        - RAG_ANSWER_TEMPLATE: Minimal tokens, standard RAG
        - RAG_ANSWER_WITH_SOURCES: Slightly longer, includes citations
        - COST_OPTIMIZED_RAG_TEMPLATE: Concise, cost-focused

    Cost Impact:
        - Template length affects input tokens
        - Document count (top_k) multiplies cost
        - Shorter documents = fewer tokens
        - Average: 500-2000 tokens per RAG query

    Template Characteristics:
        - No redundant whitespace
        - Jinja2 conditionals for flexibility
        - Clear instructions for LLM
        - Delimiters for document separation

Quality vs Cost Trade-offs:

    RAG_ANSWER_TEMPLATE:
        - Tokens: ~50 base + documents
        - Quality: Good for factual queries
        - Best for: General RAG use

    RAG_ANSWER_WITH_SOURCES_TEMPLATE:
        - Tokens: ~80 base + documents
        - Quality: Better attribution
        - Best for: When citations needed

    COST_OPTIMIZED_RAG_TEMPLATE:
        - Tokens: ~40 base + documents
        - Quality: Slightly reduced
        - Best for: High-volume, cost-sensitive

When to Use Each:
    - Standard RAG: RAG_ANSWER_TEMPLATE
    - Need citations: RAG_ANSWER_WITH_SOURCES_TEMPLATE
    - Cost critical: COST_OPTIMIZED_RAG_TEMPLATE

Template Variables:
    - documents: List of doc objects with content, meta
    - query: User query string
    - loop.index: Document position (sources template)
"""

RAG_ANSWER_TEMPLATE = """
Given the following documents, answer the question.

Documents:
{% for doc in documents %}
---
{{ doc.content }}
{% endfor %}
---

Question: {{ query }}

Provide a concise and accurate answer based only on the information in the documents.

Answer:
"""

RAG_ANSWER_WITH_SOURCES_TEMPLATE = """
Given the following documents, answer the question and cite your sources.

Documents:
{% for doc in documents %}
[{{ loop.index }}] {{ doc.content }}
{% endfor %}

Question: {{ query }}

Provide a concise answer and cite the document numbers that support your answer.

Answer:
"""

COST_OPTIMIZED_RAG_TEMPLATE = """
You are a helpful assistant that answers questions based on provided documents.

Context documents:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question: {{ query }}

Instructions:
- Answer based only on the provided context
- Be concise and direct
- If the answer is not in the context, say "I cannot find this information in the provided documents"

Answer:
"""
