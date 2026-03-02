# Haystack: Components

## 1. What This Feature Is

`vectordb.haystack.components` is the codebase's **reusable advanced-RAG component layer**. It exports five concrete classes that can be composed into larger Haystack pipelines:

| Component | Purpose | Use Case |
|-----------|---------|----------|
| **AgenticRouter** | LLM-based tool routing + self-reflection | Multi-tool RAG, answer refinement |
| **QueryEnhancer** | Query expansion (multi-query, HyDE, step-back) | Recall improvement |
| **ContextCompressor** | Query-aware context compression | Token budget control |
| **ResultMerger** | Deterministic fusion and deduplication | Hybrid retrieval fusion |
| **DeepEvalEvaluator** | DeepEval RAG metrics wrapper | Quality evaluation |

These are **framework-facing primitives** designed for composition into Haystack pipelines.

## 2. Why It Exists in Retrieval/RAG

RAG systems usually fail in **repeatable ways**:

| Failure Mode | Component Solution |
|--------------|-------------------|
| **Query too narrow** | `QueryEnhancer` expands intent |
| **Retrieved context noisy** | `ContextCompressor` shrinks context |
| **Multiple retrievers disagree** | `ResultMerger` fuses rankings |
| **Answers not refined** | `AgenticRouter` self-reflection loop |
| **Quality not measured** | `DeepEvalEvaluator` standardized metrics |

This module exists to handle those **quality gaps explicitly** with reusable components.

## 3. Component Lifecycle

Components follow a consistent lifecycle:

```mermaid
flowchart LR
    A[Initialize] --> B[Configure]
    B --> C[Execute]
    C --> D[Return Result]
    D --> E{Error?}
    E -->|Yes| F[Fallback]
    E -->|No| D
```

### Initialization

All components require **API key configuration**:

```python
from vectordb.haystack.components import AgenticRouter, QueryEnhancer, ContextCompressor

# All require GROQ_API_KEY env var or explicit api_key
router = AgenticRouter()  # Requires GROQ_API_KEY
enhancer = QueryEnhancer()  # Requires GROQ_API_KEY
compressor = ContextCompressor()  # Requires GROQ_API_KEY
```

### Execution Pattern

| Component | Input | Output |
|-----------|-------|--------|
| **AgenticRouter** | query, answer, context | tool selection, refined answer |
| **QueryEnhancer** | query, enhancement_type | enhanced queries |
| **ContextCompressor** | context, query, compression_type | compressed context |
| **ResultMerger** | ranked doc lists | fused doc list |
| **DeepEvalEvaluator** | query, context, answer | metric scores |

## 4. AgenticRouter Component

### Purpose

LLM-based tool routing plus self-reflection loop for answer improvement.

### Constructor

```python
from vectordb.haystack.components import AgenticRouter

router = AgenticRouter(
    api_key="groq-key",  # Or GROQ_API_KEY env var
    api_base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    temperature=0,  # Deterministic routing
    max_tokens=1024,
)
```

### Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| **select_tool(query)** | Choose tool: `retrieval | web_search | calculation | reasoning` | Tool name |
| **evaluate_answer_quality(query, answer, context)** | LLM judges relevance/completeness/grounding | Eval dict with scores |
| **should_refine_answer(eval_result, threshold)** | Check if avg score < threshold | Boolean |
| **self_reflect_loop(query, answer, context, max_iterations, quality_threshold)** | Evaluate → refine loop | Refined answer |

### Behavior

- **Tool validation**: Validates output against fixed list, falls back to `retrieval`
- **Evaluation JSON**: Expects `{relevance, completeness, grounding, issues, suggestions}`
- **Fallback on error**: Zero-score payload on parse/runtime error
- **Refinement loop**: Early exit when threshold met

## 5. QueryEnhancer Component

### Purpose

Query expansion using multi-query, HyDE-style hypothetical text generation, and step-back abstraction.

### Constructor

```python
from vectordb.haystack.components import QueryEnhancer

enhancer = QueryEnhancer(
    api_key="groq-key",
    model="llama-3.3-70b-versatile",
    temperature=0.7,  # Creative for query diversity
    max_tokens=1024,
)
```

### Enhancement Types

| Type | Method | Description |
|------|--------|-------------|
| **multi_query** | `generate_multi_queries(query, num_queries)` | Generate query variations |
| **hyde** | `generate_hypothetical_documents(query, num_docs)` | Generate hypothetical answers |
| **step_back** | `generate_step_back_query(query)` | Generate broader question |
| **enhance_query** | `enhance_query(query, enhancement_type, ...)` | Dispatch by type |

### Behavior

- **Enforces constraints**: `num_queries >= 1`, `num_docs >= 1`
- **Always includes original**: First item always original query
- **Fallback on error**: Returns `[query]` on LLM failure

## 6. ContextCompressor Component

### Purpose

Query-aware context compression (abstractive, extractive, relevance filtering).

### Constructor

```python
from vectordb.haystack.components import ContextCompressor

compressor = ContextCompressor(
    api_key="groq-key",
    model="llama-3.3-70b-versatile",
    temperature=0,  # Deterministic compression
    max_tokens=2048,
)
```

### Compression Types

| Type | Method | Description |
|------|--------|-------------|
| **abstractive** | `compress_abstractive(context, query, max_tokens)` | LLM summary |
| **extractive** | `compress_extractive(context, query, num_sentences)` | Select top sentences |
| **relevance_filter** | `filter_by_relevance(context, query, threshold)` | Filter by threshold |
| **compress** | `compress(context, query, compression_type, ...)` | Dispatch by type |

### Behavior

- **Logs compression ratio**: For abstractive compression
- **Returns original on error**: Fallback to full context
- **Validates config**: `validate_config(config)` lowercases type for validation

## 7. ResultMerger Component

### Purpose

Deterministic fusion and deduplication utilities for dense/sparse retrieval outputs.

### Usage (Stateless)

```python
from vectordb.haystack.components import ResultMerger
from haystack import Document

# RRF Fusion
dense_docs = [Document(content="A", score=0.8), Document(content="B", score=0.6)]
sparse_docs = [Document(content="A", score=12.0), Document(content="C", score=9.5)]

merged = ResultMerger.rrf_fusion(dense_docs, sparse_docs, k=60, top_k=10)

# Weighted Fusion
merged = ResultMerger.weighted_fusion(
    dense_docs, sparse_docs,
    dense_weight=0.6, sparse_weight=0.4,
    top_k=10,
)

# Deduplication
deduped = ResultMerger.deduplicate_by_content(merged, similarity_threshold=0.95)
# Note: Currently performs exact hash-based dedup; threshold parameter is reserved
```

### Key Methods

| Method | Purpose | Notes |
|--------|---------|-------|
| **stable_doc_id(doc)** | Generate consistent IDs | `meta["doc_id"]` → `doc.id` → SHA1(content) |
| **rrf_fusion(dense, sparse, k, top_k)** | Reciprocal Rank Fusion | Default k=60 |
| **rrf_fusion_many(ranked_lists, k, top_k)** | N-way RRF | For multiple retrievers |
| **weighted_fusion(dense, sparse, weights, top_k)** | Weighted score fusion | Auto-normalizes weights |
| **deduplicate_by_content(docs, threshold)** | Exact hash dedup | Threshold parameter reserved for future use |

### Behavior

- **Empty inputs**: Returns empty list
- **Duplicate handling**: Merges via stable IDs
- **Weight normalization**: Auto-normalizes if sum ≠ 1.0
- **Min-max normalization**: Normalizes scores to [0,1] or 0.5 when range collapses

## 8. DeepEvalEvaluator Component

### Purpose

Optional wrapper around DeepEval RAG metrics.

### Constructor

```python
from vectordb.haystack.components import DeepEvalEvaluator

# Lazily imports DeepEval; raises if not installed
evaluator = DeepEvalEvaluator()
# ValueError if deepeval not installed with install hint
```

### Metric Wrappers

| Method | Metric |
|--------|--------|
| **evaluate_contextual_recall** | Contextual Recall |
| **evaluate_contextual_precision** | Contextual Precision |
| **evaluate_answer_relevancy** | Answer Relevancy |
| **evaluate_faithfulness** | Faithfulness |

### Batch Evaluation

```python
metrics = evaluator.evaluate_all(
    query="What is RAG?",
    retrieval_context=["Context 1", "Context 2"],
    answer="RAG is Retrieval-Augmented Generation...",
    expected_output="Reference answer",  # Optional
)
```

### Behavior

- **Lazy import**: Raises `ValueError` with install hint if missing
- **Exception handling**: Per-metric exceptions → `score=0.0` with error details
- **Always computes**: Answer relevancy + faithfulness
- **Conditional**: Contextual recall/precision when `expected_output` provided

## 9. When to Use Components

### AgenticRouter

Use when:

- Multi-tool RAG workflows needed
- Answer refinement via self-reflection valuable
- Query-type-sensitive behavior required

### QueryEnhancer

Use when:

- Recall weak due to query phrasing
- Underspecified queries common
- Multi-query expansion beneficial

### ContextCompressor

Use when:

- Context too long/noisy for LLM
- Token budget constraints
- Query-aware trimming needed

### ResultMerger

Use when:

- Multiple retrievers need fusion
- Dense+sparse hybrid retrieval
- Deduplication across sources

### DeepEvalEvaluator

Use when:

- Standardized RAG metrics needed
- Quality tracking required
- Offline evaluation pipelines

## 10. When Not to Use Components

### Avoid When

| Component | Avoid When |
|-----------|------------|
| **AgenticRouter** | Strict fail-fast behavior needed |
| **QueryEnhancer** | No OpenAI-compatible endpoint |
| **ContextCompressor** | Indexing/storage logic needed |
| **ResultMerger** | Semantic deduplication required |
| **DeepEvalEvaluator** | DeepEval dependency unacceptable |

## 11. Configuration Semantics

### Key Runtime Knobs

| Component | Knob | Default | Impact |
|-----------|------|---------|--------|
| **AgenticRouter** | `temperature` | 0 | Deterministic routing |
| **AgenticRouter** | `max_tokens` | 1024 | Output length |
| **QueryEnhancer** | `temperature` | 0.7 | Query diversity |
| **QueryEnhancer** | `num_queries` | N/A | Expansion count |
| **ContextCompressor** | `temperature` | 0 | Deterministic compression |
| **ContextCompressor** | `max_tokens` | 2048 | Output length |
| **ResultMerger** | `k` (RRF) | 60 | Damping constant |
| **ResultMerger** | `weights` | [0.5, 0.5] | Fusion balance |

### API Key Configuration

All LLM-based components require API key:

```python
# Option 1: Environment variable
# export GROQ_API_KEY="your-key"

router = AgenticRouter()  # Reads GROQ_API_KEY

# Option 2: Explicit parameter
router = AgenticRouter(api_key="your-key")
```

### OpenAI-Compatible Endpoints

All components support any OpenAI-compatible endpoint:

```python
router = AgenticRouter(
    api_base_url="https://api.groq.com/openai/v1",  # Default
    # Or any OpenAI-compatible endpoint
    api_base_url="https://custom-endpoint.com/v1",
)
```

## 12. Failure Modes and Edge Cases

### AgenticRouter

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Missing API key** | Raises `ValueError` at init | Set GROQ_API_KEY |
| **Invalid tool output** | Falls back to `retrieval` | Accept fallback |
| **Invalid JSON from eval** | Zero-score fallback | Check LLM output |
| **Missing eval keys** | Treated as 0, forces refinement | Provide complete eval |
| **max_iterations=0** | Returns original answer | Set >0 for refinement |

### QueryEnhancer

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Missing API key** | Raises `ValueError` at init | Set GROQ_API_KEY |
| **Invalid num_queries/num_docs** | Raises `ValueError` | Use >= 1 |
| **LLM failure** | Returns original query | Accept fallback |
| **Invalid enhancement_type** | Raises `ValueError` | Use valid type |

### ContextCompressor

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Missing API key** | Raises `ValueError` at init | Set GROQ_API_KEY |
| **Invalid compression_type** | Raises `ValueError` | Use valid type |
| **LLM failure** | Returns original context | Accept fallback |
| **Empty context** | Handled gracefully | Not an error |

### ResultMerger

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Empty inputs** | Returns empty list | Not an error |
| **Duplicate content** | Merges via stable IDs | Expected behavior |
| **top_k=0** | Returns empty list | Use >0 |
| **Identical scores** | Normalizes to 0.5 | Expected behavior |
| **deduplicate_by_content** | Exact hash dedup (threshold unused) | Use different approach for semantic |

### DeepEvalEvaluator

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Missing deepeval** | Raises `ValueError` with install hint | `pip install deepeval` |
| **Per-metric exception** | Returns `score=0.0` with error | Check metric config |

## 13. Practical Usage Examples

### Example 1: Complete Component Pipeline

```python
from vectordb.haystack.components import (
    AgenticRouter,
    QueryEnhancer,
    ContextCompressor,
    ResultMerger,
)
from haystack import Document

# Initialize components
router = AgenticRouter()
enhancer = QueryEnhancer()
compressor = ContextCompressor()

# Enhance query
queries = enhancer.enhance_query(
    "What are retrieval failure modes?",
    enhancement_type="multi_query",
    num_queries=3,
)

# Example retriever outputs (from external retrievers)
dense_docs = [
    Document(content="Dense A", score=0.82),
    Document(content="Dense B", score=0.67),
]
sparse_docs = [
    Document(content="Dense A", score=12.0),
    Document(content="Sparse C", score=9.5),
]

# Fuse results
merged = ResultMerger.rrf_fusion(
    dense_docs,
    sparse_docs,
    k=60,
    top_k=10,
)

# Compress context
context = "\n\n".join(doc.content or "" for doc in merged)
compressed = compressor.compress(
    context,
    query=queries[0],
    compression_type="extractive",
    num_sentences=5,
)

# Generate answer (external generator)
answer = "Draft answer from your generator"

# Self-reflection
final_answer = router.self_reflect_loop(
    query=queries[0],
    answer=answer,
    context=compressed,
    max_iterations=2,
    quality_threshold=75,
)
```

### Example 2: RRF Fusion Many

```python
from vectordb.haystack.components import ResultMerger

# Results from 3 different retrievers
ranked_lists = [
    [doc1, doc2, doc3],  # Retriever 1
    [doc2, doc4, doc5],  # Retriever 2
    [doc1, doc3, doc6],  # Retriever 3
]

# Fuse with RRF
fused = ResultMerger.rrf_fusion_many(ranked_lists, k=60, top_k=10)
```

### Example 3: DeepEval Evaluation

```python
from vectordb.haystack.components import DeepEvalEvaluator

evaluator = DeepEvalEvaluator()

metrics = evaluator.evaluate_all(
    query="What is RAG?",
    retrieval_context=["Context 1", "Context 2"],
    answer="RAG is Retrieval-Augmented Generation...",
    expected_output="Reference answer text",  # Optional for contextual metrics
)

print(f"Contextual Recall: {metrics['contextual_recall']}")
print(f"Answer Relevancy: {metrics['answer_relevancy']}")
print(f"Faithfulness: {metrics['faithfulness']}")
```

### Example 4: Query Enhancement Types

```python
from vectordb.haystack.components import QueryEnhancer

enhancer = QueryEnhancer()

# Multi-query
queries = enhancer.enhance_query(
    "What causes climate change?",
    enhancement_type="multi_query",
    num_queries=5,
)

# HyDE
hypothetical_docs = enhancer.enhance_query(
    "What is quantum computing?",
    enhancement_type="hyde",
    num_docs=3,
)

# Step-back
step_back_queries = enhancer.enhance_query(
    "How does transformer attention work?",
    enhancement_type="step_back",
)
```

### Example 5: Context Compression Types

```python
from vectordb.haystack.components import ContextCompressor

compressor = ContextCompressor()

# Abstractive
compressed = compressor.compress(
    long_context,
    query="Summarize the key points",
    compression_type="abstractive",
    max_tokens=500,
)

# Extractive
compressed = compressor.compress(
    long_context,
    query="Find relevant sentences",
    compression_type="extractive",
    num_sentences=5,
)

# Relevance filtering
compressed = compressor.compress(
    long_context,
    query="Filter by relevance",
    compression_type="relevance_filter",
    relevance_threshold=0.7,
)
```

## 14. Source Walkthrough Map

### Public Export Surface

| File | Purpose |
|------|---------|
| `src/vectordb/haystack/components/__init__.py` | `__all__` exports |

### Component Implementations

| File | Component |
|------|-----------|
| `agentic_router.py` | Tool routing, quality evaluation, refinement loop |
| `query_enhancer.py` | Multi-query/HyDE/step-back transformations |
| `context_compressor.py` | Abstractive/extractive/relevance compression |
| `result_merger.py` | Stable IDs, RRF, weighted fusion, dedup |
| `evaluators.py` | DeepEval lazy import and metric wrappers |

### Test Files

| File | Coverage |
|------|----------|
| `tests/haystack/components/test_agentic_router.py` | Router tests |
| `tests/haystack/components/test_query_enhancer.py` | Enhancer tests |
| `tests/haystack/components/test_context_compressor.py` | Compressor tests |
| `tests/haystack/components/test_result_merger.py` | Merger tests |
| `tests/haystack/components/test_evaluators.py` | Evaluator tests |

### Module Reference

| File | Purpose |
|------|---------|
| `README.md` | High-level summary (runtime authority is in Python modules/tests) |

---

**Related Documentation**:

- **Agentic RAG** (`docs/haystack/agentic-rag.md`): Full agentic pipeline
- **Query Enhancement** (`docs/haystack/query-enhancement.md`): Query transformation
- **Contextual Compression** (`docs/haystack/contextual-compression.md`): Context trimming
- **Reranking** (`docs/haystack/reranking.md`): Post-retrieval scoring
