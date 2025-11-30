"""Dataset catalog for ragwatch_client."""
from __future__ import annotations

from typing import Dict, Type

from .base import DatasetClient
from .hotpotqa import HotpotQADataset

# Map dataset IDs to their implementing classes. New datasets can be added by
# updating this dictionary.
DATASETS: Dict[str, Type[DatasetClient]] = {
	HotpotQADataset.id: HotpotQADataset,
}

__all__ = ["DATASETS"]
