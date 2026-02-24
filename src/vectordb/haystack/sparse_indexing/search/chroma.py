"""Chroma sparse search pipeline for keyword/BM25-style retrieval.

Note: Chroma OSS does not support sparse vectors. Only Chroma Cloud supports
sparse vectors. This implementation raises NotImplementedError to indicate
lack of support.
"""

from pathlib import Path
from typing import Any

from vectordb.haystack.utils import ConfigLoader
from vectordb.utils.logging import LoggerFactory


logger = LoggerFactory(logger_name=__name__).get_logger()


class ChromaSparseSearchPipeline:
    """Chroma sparse search pipeline.

    Chroma OSS does not support sparse vectors. Only Chroma Cloud supports
    sparse vectors. This implementation raises NotImplementedError to indicate
    lack of support.
    """

    def __init__(self, config_or_path: dict[str, Any] | str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
        """
        self.config = ConfigLoader.load(config_or_path)

        logger.warning(
            "Chroma sparse search is not supported in Chroma OSS. "
            "Only Chroma Cloud supports sparse vectors."
        )

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Search using sparse vectors.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Raises:
            NotImplementedError: Chroma OSS does not support sparse vectors.
        """
        raise NotImplementedError(
            "Chroma sparse search is not supported in Chroma OSS. "
            "Only Chroma Cloud supports sparse vectors. "
            "Consider using Chroma Cloud or switching to a different VectorDB."
        )
