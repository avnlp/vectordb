# Agentic RAG (Retrieval-Augmented Generation) for LangChain

Agentic RAG combines multi-turn reasoning with vector search, document compression, answer reflection, and RAG generation. The agent makes routing decisions at each step to determine whether to search, reflect, or generate.

## Overview

This feature implements sophisticated agentic workflows that go beyond simple retrieval and generation. The agent orchestrates a multi-step process:

1. **Route**: Decide next action based on query and current state
2. **Search**: Retrieve documents from vector database
3. **Compress**: Extract relevant passages from retrieved documents
4. **Reflect**: Verify answer quality and iterate if needed
5. **Generate**: Create final RAG answer with ChatGroq

## Architecture

### Core Components

#### 1. AgenticRouter (`components/agentic_router.py`)
Routes queries to appropriate actions using LLM-based decision making.

**Features:**
- JSON-based structured routing decisions
- Context-aware: considers current state, iteration count, and available documents
- Max iterations safety: forces generation when iteration limit reached
- Reasoning capture: returns router's explanation for each decision

**Actions:**
- `search`: Retrieve documents from vector database
- `reflect`: Verify and improve answer quality
- `generate`: Create final answer (exits loop)

#### 2. ContextCompressor (`components/context_compressor.py`)
Extracts most relevant passages from retrieved documents.

**Modes:**
- `reranking`: Uses cross-encoder reranker (faster, more efficient)
- `llm_extraction`: Uses LLM to extract passages (more contextual)

#### 3. Base Pipeline (`base.py`)
Abstract base class defining the agentic RAG interface.

### Database-Specific Implementations

#### Search Pipelines (`search/`)
Orchestrate the full agentic workflow for each database:
- `PineconeAgenticRAGPipeline`
- `WeaviateAgenticRAGPipeline`
- `ChromaAgenticRAGPipeline`
- `MilvusAgenticRAGPipeline`
- `QdrantAgenticRAGPipeline`

**Key Methods:**
- `run(query, top_k, filters)`: Execute agentic RAG workflow

**Returns:**
```python
{
    "final_answer": str,          # Generated answer
    "documents": list[Document],  # Retrieved documents used
    "intermediate_steps": list,   # Actions taken during agentic loop
    "reasoning": list[str],       # Router reasoning for each step
}
```

#### Indexing Pipelines (`indexing/`)
Handle data indexing for agentic RAG:
- `PineconeAgenticRAGIndexingPipeline`
- `WeaviateAgenticRAGIndexingPipeline`
- `ChromaAgenticRAGIndexingPipeline`
- `MilvusAgenticRAGIndexingPipeline`
- `QdrantAgenticRAGIndexingPipeline`

## Workflow

### Agentic Loop

```
Input Query
    ↓
Embed Query
    ↓
[Loop: iteration = 1 to max_iterations]
    ↓
Router.route(query, has_docs, current_answer, iteration)
    ↓
    ├─→ action = "search"
    │       ↓
    │   Search Vector DB
    │       ↓
    │   Compress Documents
    │       ↓
    │   Store in state
    │
    ├─→ action = "reflect"
    │       ↓
    │   Verify Answer Quality
    │       ↓
    │   Update with reflection
    │
    └─→ action = "generate"
            ↓
        Generate Final Answer
            ↓
        Exit Loop
    ↓
Return {final_answer, documents, intermediate_steps, reasoning}
```

## Configuration

### YAML Structure

All configurations follow this pattern (`configs/{db}_{dataset}.yaml`):

```yaml
# Data loading
dataloader:
  type: "triviaqa|arc|popqa|factscore|earnings_calls"
  split: "test"
  limit: 100
  use_text_splitter: false

# Embeddings
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 32

# Database-specific config (varies by DB)
pinecone:  # or weaviate/chroma/milvus/qdrant
  api_key: "${PINECONE_API_KEY}"
  index_name: "lc-agentic-rag-triviaqa"
  ...

# Search parameters
search:
  top_k: 10

# RAG LLM (required for agentic RAG)
rag:
  enabled: true
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 2048

# Document reranking
reranker:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Agentic-specific settings
agentic:
  router_model: "llama-3.3-70b-versatile"
  max_iterations: 3
  compression_mode: "reranking"  # or "llm_extraction"

# Logging
logging:
  name: "lc_agentic_rag_pinecone_triviaqa"
  level: "INFO"
```

### Available Configurations

25 pre-built configurations (5 DBs × 5 datasets):

**Databases:**
- Pinecone
- Weaviate
- Chroma
- Milvus
- Qdrant

**Datasets:**
- TriviaQA
- ARC (AI2 Reasoning Challenge)
- PopQA
- FactScore
- Earnings Calls

## Usage

### Indexing

```python
from vectordb.langchain.agentic_rag.indexing import PineconeAgenticRAGIndexingPipeline

# Load config
config_path = "src/vectordb/langchain/agentic_rag/configs/pinecone_triviaqa.yaml"

# Create indexing pipeline
indexing = PineconeAgenticRAGIndexingPipeline(config_path)

# Run indexing
result = indexing.run()
print(f"Indexed {result['documents_indexed']} documents")
```

### Search / Agentic RAG

```python
from vectordb.langchain.agentic_rag.search import PineconeAgenticRAGPipeline

# Load config
config_path = "src/vectordb/langchain/agentic_rag/configs/pinecone_triviaqa.yaml"

# Create search pipeline
pipeline = PineconeAgenticRAGPipeline(config_path)

# Run agentic RAG
result = pipeline.run(
    query="What is the capital of France?",
    top_k=10,
    filters=None
)

# Results
print("Final Answer:", result["final_answer"])
print("Retrieved Documents:", len(result["documents"]))
print("Intermediate Steps:")
for step in result["intermediate_steps"]:
    print(f"  - Iteration {step['iteration']}: {step['action']} ({step['reasoning']})")
```

## Key Features

### 1. Multi-Turn Reasoning
- Agent can iterate up to `max_iterations` times
- Each iteration makes informed routing decisions
- Context from previous iterations influences next decision

### 2. Smart Document Compression
- **Reranking Mode**: Fast, efficient filtering using cross-encoders
- **LLM Extraction Mode**: More contextual, extracts specific passages
- Reduces context size for final generation

### 3. Answer Reflection
- LLM verifies answer quality against retrieved documents
- Suggestions for improvements
- Enables iterative refinement

### 4. Structured Routing
- JSON-based routing decisions
- Parser validation with error handling
- Clear reasoning captured for transparency

### 5. Multi-Database Support
- Consistent interface across all backends
- Database-specific optimizations where applicable
- Unified configuration format

## Configuration Options

### Agentic Section

```yaml
agentic:
  router_model: str              # LLM model for routing (default: llama-3.3-70b-versatile)
  max_iterations: int            # Max agentic loop iterations (default: 3, recommended: 2-5)
  compression_mode: str          # "reranking" or "llm_extraction" (default: reranking)
```

### Tuning Recommendations

**For Speed:**
```yaml
agentic:
  max_iterations: 2
  compression_mode: "reranking"
```

**For Quality:**
```yaml
agentic:
  max_iterations: 4
  compression_mode: "llm_extraction"
```

**Balanced:**
```yaml
agentic:
  max_iterations: 3
  compression_mode: "reranking"
```

## Output Structure

### Intermediate Steps

Each step in `intermediate_steps` contains:

```python
{
    "iteration": int,                    # Iteration number
    "action": "search|reflect|generate", # Action taken
    "reasoning": str,                    # Router's explanation
    "documents_retrieved": int,          # (if action="search") Number of docs
    "reflection": str,                   # (if action="reflect") Reflection text
    "answer_generated": bool,            # (if action="generate") True
}
```

## Error Handling

### Robust Router Parsing
- Validates JSON output from router
- Checks for required fields (`action`, `reasoning`)
- Validates action is one of: `search`, `reflect`, `generate`
- Raises `ValueError` with detailed error messages

### Graceful Fallbacks
- If no documents retrieved: uses LLM knowledge
- If router fails to format JSON: raises clear error with full response
- If max iterations reached: forces `generate` action

## Performance Considerations

### Latency
- Vector search: 50-200ms (varies by DB)
- Router decision: 500-1000ms (LLM call)
- Document compression: 100-500ms
- Answer generation: 1-3 seconds
- **Total typical latency: 2-5 seconds**

### Cost
- Router model calls: ~1-2 calls per query
- Compression model calls: ~1 call (if using LLM mode)
- Answer generation: 1 call
- Use `max_iterations=2` to minimize cost

## Known Limitations

1. **Router Reliability**: LLM routing decisions may be suboptimal; consider fallback strategies
2. **Context Length**: Very long document sets may exceed LLM context limits
3. **Latency**: Agentic loops add latency compared to simple retrieval
4. **Cost**: Multiple LLM calls increase operational costs

## Future Enhancements

- Tool-based routing with function calling
- Long-context model support (Claude 200K)
- Parallel search execution
- Caching for repeated queries
- Batch processing for high-throughput scenarios

## References

### Components
- `/src/vectordb/langchain/components/agentic_router.py` - Router implementation
- `/src/vectordb/langchain/components/context_compressor.py` - Document compression
- `/src/vectordb/langchain/utils/rag.py` - RAG helper utilities

### Database Wrappers
- Core database implementations in `/src/vectordb/{db}.py`

### Related Features
- Semantic Search: `/src/vectordb/langchain/semantic_search/`
- Query Enhancement: `/src/vectordb/langchain/query_enhancement/`
- Reranking: `/src/vectordb/langchain/reranking/`
