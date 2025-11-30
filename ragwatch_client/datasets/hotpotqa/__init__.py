"""HotpotQA corpus-only dataset components."""
from __future__ import annotations

from typing import Any, Dict, List

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

__all__ = ["DATASET_NAME", "HotpotQADataSource"]
