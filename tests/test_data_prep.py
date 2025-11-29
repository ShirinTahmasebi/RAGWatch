import os
import unittest

from ragwatch_client.datasets.hotpotqa import data as data_prep


class HotpotQADataPrepTest(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_stub = os.environ.get("RAGWATCH_HOTPOTQA_USE_STUBS")
        os.environ["RAGWATCH_HOTPOTQA_USE_STUBS"] = "true"
        data_prep.clear_hotpotqa_cache()

    def tearDown(self) -> None:
        if self._orig_stub is None:
            os.environ.pop("RAGWATCH_HOTPOTQA_USE_STUBS", None)
        else:
            os.environ["RAGWATCH_HOTPOTQA_USE_STUBS"] = self._orig_stub
        data_prep.clear_hotpotqa_cache()


    def test_load_questions_returns_stub_entry(self) -> None:
        questions = data_prep.load_questions()
        self.assertGreaterEqual(len(questions), 1)
        sample = questions[0]
        self.assertEqual(sample["id"], "q1")
        self.assertIn("question", sample)
        self.assertIn("answer", sample)

    def test_build_document_corpus_returns_stub_doc(self) -> None:
        docs = data_prep.build_document_corpus()
        self.assertGreaterEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc["doc_id"], "d1")
        self.assertIn("text", doc)
        self.assertIn("title", doc)


if __name__ == "__main__":
    unittest.main()
