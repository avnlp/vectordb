"""Sparse embeddings utilities using sentence-transformers SparseEncoder."""

from sentence_transformers import SparseEncoder


class SparseEmbedder:  # noqa: D101
    """Sparse embeddings using sentence-transformers SparseEncoder (SPLADE).

    This class wraps the official SparseEncoder from sentence-transformers,
    providing a consistent interface for generating sparse embeddings from text.
    """

    def __init__(
        self, model_name: str = "naver/splade-v2", device: str = "cpu"
    ) -> None:
        """Initialize sparse embedder with SparseEncoder.

        Args:
            model_name: Model name from HuggingFace (default: naver/splade-v2).
            device: Device to run model on ("cpu" or "cuda").

        Raises:
            OSError: If model cannot be loaded from HuggingFace.
        """
        self.model_name = model_name
        self.device = device
        self.encoder = SparseEncoder(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[dict[str, float]]:
        """Embed documents to sparse vectors using SparseEncoder.

        Args:
            texts: List of text documents to embed.

        Returns:
            List of sparse vectors as dicts {token_id: weight}.
            Each dict maps sparse token indices to their weights.
        """
        if not texts:
            return []

        # Use encode_document for document-specific embeddings
        # Returns torch.Tensor of shape (batch_size, vocab_size)
        sparse_embeddings_tensor = self.encoder.encode_document(texts)

        # Each row is a sparse vector - convert to dict format
        sparse_embeddings_list = []
        for embedding in sparse_embeddings_tensor:
            # embedding is a 1D tensor with vocab_size dimensions
            dense_list = embedding.tolist()

            sparse_embeddings_list.append(
                {
                    str(idx): float(weight)
                    for idx, weight in enumerate(dense_list)
                    if weight > 0  # Only include positive weights (sparsity)
                }
            )

        return sparse_embeddings_list

    def embed_query(self, text: str) -> dict[str, float]:
        """Embed query to sparse vector using SparseEncoder.

        Args:
            text: Query text to embed.

        Returns:
            Sparse vector as dict {token_id: weight}.
        """
        if not text:
            return {}

        # Use encode_query for query-specific embeddings
        # Returns torch.Tensor of shape (1, vocab_size)
        sparse_embedding_tensor = self.encoder.encode_query(text)

        dense_list = sparse_embedding_tensor[0].tolist()

        return {
            str(idx): float(weight)
            for idx, weight in enumerate(dense_list)
            if weight > 0
        }

    def embed_documents_normalized(self, texts: list[str]) -> list[dict[str, float]]:
        """Embed documents to normalized sparse vectors.

        Normalization ensures L2 norm = 1 for each vector.

        Args:
            texts: List of text documents.

        Returns:
            List of normalized sparse vectors {index: value}.
        """
        raw_embeddings = self.embed_documents(texts)
        normalized = []

        for sparse_vec in raw_embeddings:
            if not sparse_vec:
                normalized.append(sparse_vec)
                continue

            # Normalize weights (L2 normalization)
            total_squared = sum(v**2 for v in sparse_vec.values())
            if total_squared > 0:
                norm = total_squared**0.5
                normalized_vec = {k: v / norm for k, v in sparse_vec.items()}
            else:
                normalized_vec = sparse_vec

            normalized.append(normalized_vec)

        return normalized

    def embed_query_normalized(self, text: str) -> dict[str, float]:
        """Embed query to normalized sparse vector.

        Args:
            text: Query text.

        Returns:
            Normalized sparse vector {index: value}.
        """
        sparse_vec = self.embed_query(text)

        if not sparse_vec:
            return sparse_vec

        # L2 normalization
        total_squared = sum(v**2 for v in sparse_vec.values())
        if total_squared > 0:
            norm = total_squared**0.5
            return {k: v / norm for k, v in sparse_vec.items()}

        return sparse_vec
