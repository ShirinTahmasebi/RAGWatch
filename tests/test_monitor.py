import json
import tempfile
import unittest
from pathlib import Path

from ragwatch import RAGMonitor


class DummyDoc:
    def __init__(self, **metadata):
        self.metadata = metadata


class RAGMonitorTest(unittest.TestCase):
    def test_session_logs_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = RAGMonitor(dataset_name="demo", pipeline_name="p1", log_dir=tmpdir)
            question = "What is retrieval-augmented generation?"

            with monitor.session(question=question, session_id="sess-1") as session:
                session.record_retrieval(
                    [
                        {
                            "doc_id": "doc-1",
                            "score": 0.42,
                            "source": "stub",
                            "metadata": {"foo": "bar"},
                        }
                    ]
                )
                session.record_answer(
                    "It combines retrieval with generation.",
                    latency_ms={"total": 123.0},
                    token_usage={"prompt": 10, "completion": 5},
                )
                session.set_gold_answer("A hybrid pipeline")
                session.add_extra(tag="unit-test")

            log_path = Path(tmpdir) / "demo_p1.jsonl"
            self.assertTrue(log_path.exists())
            payload = json.loads(log_path.read_text().strip())
            self.assertEqual(payload["session_id"], "sess-1")
            self.assertEqual(payload["question"], question)
            self.assertEqual(payload["answer"], "It combines retrieval with generation.")
            self.assertEqual(payload["extra"]["tag"], "unit-test")
            self.assertEqual(payload["extra"]["gold_answer"], "A hybrid pipeline")
            self.assertEqual(len(payload["retrieved_docs"]), 1)

    def test_record_retrieval_normalizes_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = RAGMonitor(dataset_name="demo", pipeline_name="p2", log_dir=tmpdir)
            doc = DummyDoc(doc_id="doc-2", source="dummy", title="Dummy Title")

            with monitor.session(question="Test?", session_id="sess-2") as session:
                session.record_retrieval([(doc, 0.99)])
                session.record_answer("ok")

            payload = json.loads((Path(tmpdir) / "demo_p2.jsonl").read_text().strip())
            retrieved = payload["retrieved_docs"]
            self.assertEqual(len(retrieved), 1)
            self.assertEqual(retrieved[0]["doc_id"], "doc-2")
            self.assertAlmostEqual(retrieved[0]["score"], 0.99)


if __name__ == "__main__":
    unittest.main()
