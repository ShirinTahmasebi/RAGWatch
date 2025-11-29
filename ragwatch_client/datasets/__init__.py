"""Dataset registry for ragwatch_client."""
from __future__ import annotations

from .base import DatasetRegistry
from .hotpotqa import HotpotQADataset

registry = DatasetRegistry()
registry.register(HotpotQADataset())

__all__ = ["registry"]
