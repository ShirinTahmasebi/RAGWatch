"""SQuAD dataset adapter."""

from ragwatch.core.schema import Document, QAExample, RAGDataset
from ragwatch.datasets.base import BaseQAAdapter
from ragwatch.datasets.hf_qa_loader import load_hf_dataset


class SquadAdapter(BaseQAAdapter):
    """Adapter that loads SQuAD v1.1 from Hugging Face and produces a RAGDataset."""

    def load(
        self, split: str = "validation", max_examples: int | None = None
    ) -> RAGDataset:
        dataset = load_hf_dataset("rajpurkar/squad", split=split)

        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        # Deduplicate contexts to build the corpus
        seen_contexts: dict[str, str] = {}  # text -> doc_id
        documents: list[Document] = []
        qa_examples: list[QAExample] = []

        for idx, row in enumerate(dataset):
            context = row["context"]
            title = row["title"]

            # Deduplicate by exact context text
            if context not in seen_contexts:
                doc_id = f"squad_doc_{len(documents)}"
                seen_contexts[context] = doc_id
                documents.append(
                    Document(
                        doc_id=doc_id,
                        text=context,
                        metadata={"source": "squad", "title": title},
                    )
                )

            # Build QA example
            answers = row["answers"]["text"]
            qa_examples.append(
                QAExample(
                    example_id=f"squad_q_{idx}",
                    question=row["question"],
                    answers=answers,
                    metadata={
                        "dataset": "squad",
                        "title": title,
                        "context_doc_id": seen_contexts[context],
                        "answer_starts": row["answers"]["answer_start"],
                        "id": row["id"],
                    },
                )
            )

        return RAGDataset(
            name="squad",
            corpus=documents,
            qa_examples=qa_examples,
            metadata={"split": split, "num_examples": len(qa_examples)},
        )
