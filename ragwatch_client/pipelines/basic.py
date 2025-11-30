"""Minimal LangChain-based RAG chain builders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from .base import RAGPipelineBuilder


def format_docs_compact(docs: List[Document]) -> str:
    """Default doc formatter used by the basic RAG chain."""

    if not docs:
        return "(no supporting documents retrieved)"
    return "\n\n".join(doc.page_content for doc in docs)


@dataclass(slots=True)
class BasicRAGPipelineBuilder(RAGPipelineBuilder):
    """Configurable builder for the default LangChain-style pipeline."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    system_prompt: str = "You answer questions using the provided context. Be concise."
    human_template: str = "Question: {question}\n\nContext:\n{context}"
    llm: Optional[BaseChatModel] = None
    format_docs: Callable[[List[Document]], str] = format_docs_compact

    def build_chain(self, retriever: BaseRetriever):  # type: ignore[override]
        llm = self.llm or ChatOpenAI(model=self.model, temperature=self.temperature)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", self.human_template),
            ]
        )

        rag_chain = (
            {"question": RunnablePassthrough(), "docs": retriever}
            | RunnableParallel(
                question=lambda x: x["question"],
                context=lambda x: self.format_docs(x["docs"]),
            )
            | prompt
            | llm
        )

        return rag_chain


def build_basic_rag_chain(
    retriever: BaseRetriever,
    **builder_kwargs,
):
    """Helper for legacy call sites that expect a function."""

    return BasicRAGPipelineBuilder(**builder_kwargs).build_chain(retriever)


__all__ = ["BasicRAGPipelineBuilder", "build_basic_rag_chain", "format_docs_compact"]
