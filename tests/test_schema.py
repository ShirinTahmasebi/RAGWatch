import datetime as dt
import unittest

from ragwatch.models import RAGRunRecord, RetrievedDoc


class RAGRunRecordSchemaTest(unittest.TestCase):
    def test_run_record_instantiation(self) -> None:
        retrieved = [
            RetrievedDoc(
                doc_id="doc-1",
                score=0.87,
                source="vector_store",
                metadata={"chunk": 1},
            )
        ]

        record = RAGRunRecord(
            run_id="run-123",
            session_id="session-abc",
            timestamp=dt.datetime.utcnow(),
            dataset_name="hotpotqa",
            pipeline_name="hotpotqa_v1",
            question="Who wrote the novel?",
            answer="It was written by Jane Doe.",
            retrieved_docs=retrieved,
            latency_ms={"retrieve": 12.3, "generate": 120.5},
            token_usage={"prompt": 100, "completion": 42},
            extra={"escalated": True},
        )

        self.assertEqual(record.run_id, "run-123")
        self.assertEqual(record.retrieved_docs[0].doc_id, "doc-1")
        self.assertTrue(record.extra["escalated"])


if __name__ == "__main__":
    unittest.main()
