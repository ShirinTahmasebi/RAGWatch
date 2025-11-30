"""Corpus-only dataset catalog for ragwatch_client."""
from __future__ import annotations

from typing import Callable, Dict

from .base import CorpusDataset
from .hotpotqa import DATASET_NAME as HOTPOTQA_NAME, build_data_source as build_hotpotqa_data_source

# Map dataset names to callables that build corpus datasets.
DATA_SOURCES: Dict[str, Callable[[], CorpusDataset]] = {
    HOTPOTQA_NAME: build_hotpotqa_data_source,
}

__all__ = ["DATA_SOURCES"]
