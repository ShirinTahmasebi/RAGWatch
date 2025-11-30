"""FAISS-backed implementation of the VectorStoreAdapter interface."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from ragwatch.utils import EnvKeys, env_path

from .base import VectorStoreAdapter


def _to_documents(docs: List[Dict[str, str]]) -> List[Document]:
    return [
        Document(page_content=doc["text"], metadata={k: v for k, v in doc.items() if k != "text"})
        for doc in docs
    ]


@dataclass(slots=True)
class FaissVectorStoreAdapter(VectorStoreAdapter):
    """Vectorstore adapter that persists embeddings using FAISS."""

    index_base_dir: str | Path | None = None
    embeddings: Optional[Embeddings] = None
    default_top_k: int = 5

    def load_if_exists(self, dataset_name: str):  # type: ignore[override]
        index_dir = self._resolve_index_dir(dataset_name)
        if not self._index_files_exist(index_dir):
            raise FileNotFoundError(f"FAISS index missing at: {index_dir}")

        embeddings = self.embeddings or OpenAIEmbeddings()
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def build(self, dataset_name: str, docs: List[Dict[str, str]]):  # type: ignore[override]
        if not docs:
            raise ValueError("Cannot build vectorstore without documents")

        index_dir = self._resolve_index_dir(dataset_name)
        index_dir.mkdir(parents=True, exist_ok=True)

        embeddings = self.embeddings or OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(_to_documents(docs), embeddings)
        vectorstore.save_local(str(index_dir))
        return vectorstore

    # ----- Helpers ---------------------------------------------------------
    def _resolve_index_dir(self, dataset_name: str) -> Path:
        base_dir = Path(self.index_base_dir).expanduser() if self.index_base_dir else env_path(EnvKeys.INDEX_DIR)
        return Path(base_dir) / dataset_name

    @staticmethod
    def _index_files_exist(index_dir: Path) -> bool:
        return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


__all__ = ["FaissVectorStoreAdapter"]
