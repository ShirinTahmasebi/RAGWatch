import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ragwatch import RAGMonitor
from ragwatch.utils import EnvKeys


class DummyDoc:
    def __init__(self, **metadata):
        self.metadata = metadata


class RecordingStepLogger:
    def __init__(self):
        self.entries = []

    def log(self, stage: str, message: str, *args, **kwargs):
        rendered = message % args if args else message
        self.entries.append((stage, rendered))


class RAGMonitorTest(unittest.TestCase):
    def test_session_logs_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = RAGMonitor(dataset_name="demo", version="p1", log_dir=tmpdir)
            question = "What is retrieval-augmented generation?"

            with monitor.session(question=question, session_id="sess-1") as session:
                session.record_retrieval(
                    [
                        {
                            "doc_id": "doc-1",
                            "score": 0.42,
                            "source": "dummy",
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

            log_path = Path(tmpdir) / "demo" / "demo_p1.jsonl"
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
            monitor = RAGMonitor(dataset_name="demo", version="p2", log_dir=tmpdir)
            doc = DummyDoc(doc_id="doc-2", source="dummy", title="Dummy Title")

            with monitor.session(question="Test?", session_id="sess-2") as session:
                session.record_retrieval([(doc, 0.99)])
                session.record_answer("ok")

            payload = json.loads((Path(tmpdir) / "demo" / "demo_p2.jsonl").read_text().strip())
            retrieved = payload["retrieved_docs"]
            self.assertEqual(len(retrieved), 1)
            self.assertEqual(retrieved[0]["doc_id"], "doc-2")
            self.assertAlmostEqual(retrieved[0]["score"], 0.99)

    def test_session_emits_step_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = RecordingStepLogger()
            monitor = RAGMonitor(
                dataset_name="demo",
                version="p3",
                log_dir=tmpdir,
                step_logger=recorder,
            )
            with monitor.session(question="Why?", session_id="sess-3", log_steps=True) as session:
                session.record_retrieval(
                    [
                        {"doc_id": "doc-1", "score": 0.9, "source": "dummy", "metadata": {}},
                        {"doc_id": "doc-2", "score": 0.4, "source": "dummy2", "metadata": {}},
                    ],
                    top_k=5,
                )
                session.record_answer("Because", latency_ms={"total": 42.0})

        stages = [stage for stage, _ in recorder.entries]
        self.assertIn("session", stages)
        self.assertIn("retrieval", stages)
        self.assertIn("write", stages)

    def test_monitor_uses_env_log_dir_when_not_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {EnvKeys.LOG_DIR: tmpdir}, clear=False):
                monitor = RAGMonitor(dataset_name="demo", version="p4")
                with monitor.session(question="Env?", session_id="sess-env") as session:
                    session.record_answer("ok")
            log_path = Path(tmpdir) / "demo" / "demo_p4.jsonl"
            self.assertTrue(log_path.exists())

    def test_monitor_requires_log_dir_when_env_missing(self):
        original = os.environ.pop(EnvKeys.LOG_DIR, None)
        try:
            with self.assertRaises(RuntimeError):
                RAGMonitor(dataset_name="demo", version="p5")
        finally:
            if original is not None:
                os.environ[EnvKeys.LOG_DIR] = original


if __name__ == "__main__":
    unittest.main()
