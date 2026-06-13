"""HotpotQA dataset adapter."""

from ragwatch.core.schema import Document, QAExample, RAGDataset
from ragwatch.datasets.base import BaseQAAdapter
from ragwatch.datasets.hf_qa_loader import load_hf_dataset


class HotpotQAAdapter(BaseQAAdapter):
    """Adapter that loads HotpotQA from Hugging Face and produces a RAGDataset.

    Uses the 'fullwiki' configuration by default.
    HotpotQA provides multi-hop questions with supporting paragraphs.
    """

    def __init__(self, config: str = "fullwiki") -> None:
        self._config = config

    def load(
        self, split: str = "validation", max_examples: int | None = None
    ) -> RAGDataset:
        dataset = load_hf_dataset("hotpotqa/hotpot_qa", name=self._config, split=split)

        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        seen_contexts: dict[str, str] = {}  # text -> doc_id
        documents: list[Document] = []
        qa_examples: list[QAExample] = []

        for idx, row in enumerate(dataset):
            # HotpotQA context is a list of titles and sentences
            context_titles = row["context"]["title"]
            context_sentences = row["context"]["sentences"]

            # Supporting facts for metadata
            supporting_titles = row.get("supporting_facts", {}).get("title", [])

            # Build documents from context paragraphs
            example_doc_ids: list[str] = []
            for title, sentences in zip(context_titles, context_sentences):
                paragraph_text = " ".join(sentences)
                if not paragraph_text.strip():
                    continue

                if paragraph_text not in seen_contexts:
                    doc_id = f"hotpotqa_doc_{len(documents)}"
                    seen_contexts[paragraph_text] = doc_id
                    documents.append(
                        Document(
                            doc_id=doc_id,
                            text=paragraph_text,
                            metadata={
                                "source": "hotpot_qa",
                                "title": title,
                                "config": self._config,
                            },
                        )
                    )
                example_doc_ids.append(seen_contexts[paragraph_text])

            # Build QA example
            qa_examples.append(
                QAExample(
                    example_id=f"hotpotqa_q_{idx}",
                    question=row["question"],
                    answers=[row["answer"]],
                    metadata={
                        "dataset": "hotpot_qa",
                        "type": row.get("type", ""),
                        "level": row.get("level", ""),
                        "supporting_titles": supporting_titles,
                        "context_doc_ids": example_doc_ids,
                        "id": row.get("id", ""),
                    },
                )
            )

        return RAGDataset(
            name="hotpot_qa",
            corpus=documents,
            qa_examples=qa_examples,
            metadata={
                "split": split,
                "config": self._config,
                "num_examples": len(qa_examples),
            },
        )
