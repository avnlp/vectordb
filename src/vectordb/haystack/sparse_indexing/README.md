# Sparse Indexing (Haystack)

Sparse retrieval encodes documents and queries as high-dimensional token-weight vectors where each dimension corresponds to a vocabulary token. Only non-zero dimensions are stored, making storage efficient. Retrieval is based on lexical overlap: documents with many tokens that match the query receive high scores.

## How It Works

1. **Sparse encoding**: Each document is encoded by a sparse SentenceTransformers model (`SentenceTransformersSparseDocumentEmbedder`, typically a SPLADE model). SPLADE expands vocabulary weights beyond exact token matches by learning to upweight semantically related tokens during training, giving sparse retrieval some limited semantic generalization.
2. **Sparse index**: The sparse vector is stored in the backend alongside (or instead of) a dense vector. Backends that natively support sparse vectors (Milvus, Pinecone, Qdrant) use their native sparse index types; others use an inverted index approximation.
3. **Sparse query**: At query time, the same sparse encoder produces a query sparse vector (`SentenceTransformersSparseTextEmbedder`).
4. **Inner-product retrieval**: Documents are ranked by inner product between the query sparse vector and each document sparse vector, which rewards token overlap weighted by SPLADE's learned importance scores.

## When to Use It

- Keyword-heavy workloads: product codes, legal citations, medical terminology, technical identifiers.
- Domain jargon where exact terms matter and semantic paraphrasing is unhelpful or misleading.
- As the sparse leg of a hybrid pipeline (paired with dense retrieval in `hybrid_indexing/`).

## When Not to Use It

- Open-domain QA with varied natural-language phrasing where the query and documents rarely share exact vocabulary.
- Tasks where the document vocabulary is very different from typical query vocabulary (for example, dense academic writing vs. colloquial user queries).

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Strong lexical precision; limited semantic generalization beyond SPLADE learned expansion |
| Latency | Backend-dependent; sparse inner-product search is often competitive with ANN search |
| Cost | Sparse embedding inference is usually faster than dense; storage cost depends on vector sparsity |

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `sparse.model` | Model choice determines how aggressively vocabulary is expanded beyond exact matches |
| `search.top_k` | Sparse retrieval tends to have sharp relevance cutoffs; setting top_k too small loses hits |
| Score normalization | Some SPLADE models return unnormalized scores; verify the score range matches expectations |

## Common Pitfalls

- **Using sparse-only for paraphrase-heavy queries**: If users ask about "heart disease" and documents discuss "cardiovascular conditions", sparse retrieval fails without hybrid pairing.
- **Ignoring tokenization quirks**: Compound words, camelCase identifiers, and domain abbreviations may tokenize unexpectedly. Check SPLADE's vocabulary coverage for your domain.
- **Not pairing with dense in mixed workloads**: Sparse retrieval is rarely the best single-signal choice for general RAG. It works best as one leg of a hybrid pipeline.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `hybrid_indexing/` to pair sparse with dense retrieval for more robust coverage.
- Use `semantic_search/` as the baseline to determine whether your workload benefits more from semantic or lexical retrieval.
