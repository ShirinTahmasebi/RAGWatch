"""HotpotQA corpus-only dataset components."""
from __future__ import annotations

from typing import Any, Dict, List

from ragwatch.utils import EnvKeys

from ..base import CorpusDataset
from . import data

DATASET_NAME = "hotpotqa"


class HotpotQADataSource(CorpusDataset):
    dataset_name = DATASET_NAME
    description = "LangChain demo pipeline answering HotpotQA questions."

    def load_questions(self) -> List[Dict[str, Any]]:
        return data.load_questions()

    def build_document_corpus(self) -> List[Dict[str, Any]]:
        return data.build_document_corpus()

def build_data_source() -> CorpusDataset:
    """Return the corpus-only dataset for downstream wiring."""

    return HotpotQADataSource()


__all__ = ["DATASET_NAME", "HotpotQADataSource", "build_data_source"]
