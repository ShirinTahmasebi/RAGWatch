import datetime as dt
import json
import tempfile
from pathlib import Path
import unittest

from ragwatch.models import RAGRunRecord, RetrievedDoc
from ragwatch.utils import JSONLWriter


def _sample_run_record() -> RAGRunRecord:
    return RAGRunRecord(
        run_id="run-456",
        session_id="session-xyz",
        timestamp=dt.datetime.utcnow(),
        dataset_name="hotpotqa",
        version="hotpotqa_v1",
        question="Where was the author born?",
        answer="The author was born in Paris.",
        retrieved_docs=[
            RetrievedDoc(
                doc_id="doc-99",
                score=0.92,
                source="vector_store",
                metadata={"chunk": 3},
            )
        ],
        latency_ms={"retrieve": 11.2, "generate": 95.4},
        token_usage={"prompt": 88, "completion": 37},
        extra={"escalated": False},
    )


class JSONLWriterTest(unittest.TestCase):
    def test_write_appends_json_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "hotpotqa.jsonl"
            writer = JSONLWriter(str(log_path))

            record = _sample_run_record()
            writer.write(record)

            self.assertTrue(log_path.exists())
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["run_id"], record.run_id)
            self.assertEqual(parsed["retrieved_docs"][0]["doc_id"], "doc-99")


if __name__ == "__main__":
    unittest.main()
