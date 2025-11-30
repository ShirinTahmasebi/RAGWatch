"""HotpotQA dataset adapter for the generic ragwatch_client runner."""
from __future__ import annotations

from typing import Any, Dict, List

from ragwatch.utils import EnvKeys

from ..base import DatasetClient, DatasetResources
from . import data, pipeline, retriever


class HotpotQADataset(DatasetClient):
    id = "hotpotqa"
    description = "LangChain demo pipeline answering HotpotQA questions."
    default_dataset_name = "hotpotqa"
    default_pipeline_name = "v1"
    log_env_var = EnvKeys.HOTPOTQA_LOG_DIR
    default_top_k = retriever.DEFAULT_TOP_K

    def load_questions(self) -> List[Dict[str, Any]]:
        return data.load_questions()

    def prepare_resources(self) -> DatasetResources:
        docs = data.build_document_corpus()
        vectorstore, source = self._load_or_create_vectorstore(docs)
        retriever_instance = vectorstore.as_retriever(
            search_kwargs={"k": retriever.DEFAULT_TOP_K}
        )
        rag_chain = pipeline.build_rag_chain(retriever=retriever_instance)
        metadata = {
            "doc_count": len(docs),
            "top_k": retriever.DEFAULT_TOP_K,
            "vectorstore_source": source,
        }
        return DatasetResources(vectorstore=vectorstore, rag_chain=rag_chain, metadata=metadata)

    def _load_or_create_vectorstore(self, docs: List[Dict[str, Any]]):
        try:
            vectorstore = retriever.load_vectorstore()
            return vectorstore, "loaded"
        except FileNotFoundError:
            retriever.build_retriever(docs)
            vectorstore = retriever.load_vectorstore()
            return vectorstore, "built"
