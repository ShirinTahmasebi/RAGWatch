import json
import tempfile
from pathlib import Path
import unittest

from ragwatch import RAGWatchLogger


class RAGWatchLoggerTest(unittest.TestCase):
    def test_log_run_creates_jsonl_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = RAGWatchLogger(
                log_dir=str(log_dir),
                dataset_name="hotpotqa",
                pipeline_name="v1",
            )

            logger.log_run(
                question="What is RAG?",
                answer="A retrieval-augmented generation system.",
                retrieved_docs=[
                    {
                        "doc_id": "d1",
                        "score": 0.9,
                        "source": "dummy",
                        "metadata": {},
                    }
                ],
                latency_ms={"total": 123.4},
                token_usage={"prompt": 100, "completion": 50},
                session_id="test-session",
                extra={"escalated": False},
            )

            log_file = log_dir / "hotpotqa_v1.jsonl"
            self.assertTrue(log_file.exists())
            lines = log_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["question"], "What is RAG?")
            self.assertEqual(payload["retrieved_docs"][0]["doc_id"], "d1")


if __name__ == "__main__":
    unittest.main()
