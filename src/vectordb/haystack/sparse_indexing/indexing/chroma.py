"""Chroma sparse indexing pipeline for keyword/BM25-style search.

Note: Chroma OSS does not support sparse vectors. Only Chroma Cloud supports
sparse vectors. This implementation raises NotImplementedError to indicate
lack of support.
"""

from pathlib import Path
from typing import Any

from vectordb.haystack.utils import ConfigLoader
from vectordb.utils.logging import LoggerFactory


logger = LoggerFactory(logger_name=__name__).get_logger()


class ChromaSparseIndexingPipeline:
    """Chroma sparse indexing pipeline.

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
            "Chroma sparse indexing is not supported in Chroma OSS. "
            "Only Chroma Cloud supports sparse vectors."
        )

    def run(self) -> dict[str, Any]:
        """Run the indexing pipeline.

        Raises:
            NotImplementedError: Chroma OSS does not support sparse vectors.
        """
        raise NotImplementedError(
            "Chroma sparse indexing is not supported in Chroma OSS. "
            "Only Chroma Cloud supports sparse vectors. "
            "Consider using Chroma Cloud or switching to a different VectorDB."
        )
