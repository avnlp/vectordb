"""Dataset-specific loader implementations."""

from vectordb.dataloaders.datasets.arc import ARCLoader
from vectordb.dataloaders.datasets.earnings_calls import EarningsCallsLoader
from vectordb.dataloaders.datasets.factscore import FactScoreLoader
from vectordb.dataloaders.datasets.popqa import PopQALoader
from vectordb.dataloaders.datasets.triviaqa import TriviaQALoader


__all__ = [
    "ARCLoader",
    "EarningsCallsLoader",
    "FactScoreLoader",
    "PopQALoader",
    "TriviaQALoader",
]
