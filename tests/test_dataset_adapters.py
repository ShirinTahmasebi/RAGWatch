"""Tests for dataset adapters using mocked Hugging Face data."""

from unittest.mock import MagicMock, patch

from ragwatch.datasets.adapters.squad_adapter import SquadAdapter
from ragwatch.datasets.adapters.hotpotqa_adapter import HotpotQAAdapter


def _make_mock_squad_dataset():
    """Create a mock SQuAD dataset."""
    data = [
        {
            "id": "56be85543aeaaa14008c9063",
            "title": "Beyoncé",
            "context": "Beyoncé Giselle Knowles-Carter is an American singer.",
            "question": "What is Beyoncé's full name?",
            "answers": {"text": ["Beyoncé Giselle Knowles-Carter"], "answer_start": [0]},
        },
        {
            "id": "56be85543aeaaa14008c9064",
            "title": "Beyoncé",
            "context": "Beyoncé Giselle Knowles-Carter is an American singer.",
            "question": "What nationality is Beyoncé?",
            "answers": {"text": ["American"], "answer_start": [42]},
        },
        {
            "id": "56be85543aeaaa14008c9065",
            "title": "Frédéric Chopin",
            "context": "Chopin was a Polish composer and virtuoso pianist.",
            "question": "What instrument did Chopin play?",
            "answers": {"text": ["pianist"], "answer_start": [42]},
        },
    ]

    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: len(data)
    mock_ds.__iter__ = lambda self: iter(data)
    mock_ds.select = lambda rng: _make_mock_subset(data, rng)
    return mock_ds


def _make_mock_subset(data, rng):
    subset = [data[i] for i in rng]
    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: len(subset)
    mock_ds.__iter__ = lambda self: iter(subset)
    return mock_ds


def _make_mock_hotpotqa_dataset():
    """Create a mock HotpotQA dataset."""
    data = [
        {
            "id": "5a7a06935542990198eaf050",
            "question": "Which band has more members, Queensrÿche or The Bravery?",
            "answer": "Queensrÿche",
            "type": "comparison",
            "level": "medium",
            "context": {
                "title": ["Queensrÿche", "The Bravery"],
                "sentences": [
                    ["Queensrÿche is an American heavy metal band.", "They have five members."],
                    ["The Bravery was a band from New York.", "They had four members."],
                ],
            },
            "supporting_facts": {
                "title": ["Queensrÿche", "The Bravery"],
                "sent_id": [1, 1],
            },
        },
        {
            "id": "5a7a06935542990198eaf051",
            "question": "What city is The Bravery from?",
            "answer": "New York",
            "type": "bridge",
            "level": "easy",
            "context": {
                "title": ["The Bravery", "New York City"],
                "sentences": [
                    ["The Bravery was a band from New York.", "They had four members."],
                    ["New York City is in the state of New York.", "It is the largest city in the US."],
                ],
            },
            "supporting_facts": {
                "title": ["The Bravery"],
                "sent_id": [0],
            },
        },
    ]

    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: len(data)
    mock_ds.__iter__ = lambda self: iter(data)
    mock_ds.select = lambda rng: _make_mock_subset(data, rng)
    return mock_ds


class TestSquadAdapter:
    @patch("ragwatch.datasets.adapters.squad_adapter.load_hf_dataset")
    def test_load_produces_rag_dataset(self, mock_load_hf) -> None:
        mock_load_hf.return_value = _make_mock_squad_dataset()

        adapter = SquadAdapter()
        dataset = adapter.load(split="validation", max_examples=3)

        assert dataset.name == "squad"
        assert len(dataset.qa_examples) == 3
        # Two unique contexts -> 2 documents
        assert len(dataset.corpus) == 2

    @patch("ragwatch.datasets.adapters.squad_adapter.load_hf_dataset")
    def test_deduplicates_contexts(self, mock_load_hf) -> None:
        mock_load_hf.return_value = _make_mock_squad_dataset()

        adapter = SquadAdapter()
        dataset = adapter.load(split="validation")

        # First two questions share the same context
        doc_ids = [doc.doc_id for doc in dataset.corpus]
        assert len(doc_ids) == len(set(doc_ids))

    @patch("ragwatch.datasets.adapters.squad_adapter.load_hf_dataset")
    def test_qa_examples_have_metadata(self, mock_load_hf) -> None:
        mock_load_hf.return_value = _make_mock_squad_dataset()

        adapter = SquadAdapter()
        dataset = adapter.load(split="validation")

        example = dataset.qa_examples[0]
        assert example.metadata["dataset"] == "squad"
        assert "context_doc_id" in example.metadata
        assert "title" in example.metadata

    @patch("ragwatch.datasets.adapters.squad_adapter.load_hf_dataset")
    def test_max_examples_limits_output(self, mock_load_hf) -> None:
        mock_load_hf.return_value = _make_mock_squad_dataset()

        adapter = SquadAdapter()
        dataset = adapter.load(split="validation", max_examples=1)

        assert len(dataset.qa_examples) == 1


class TestHotpotQAAdapter:
    @patch("ragwatch.datasets.adapters.hotpotqa_adapter.load_hf_dataset")
    def test_load_produces_rag_dataset(self, mock_load_hf) -> None:
        mock_load_hf.return_value = _make_mock_hotpotqa_dataset()

        adapter = HotpotQAAdapter()
        dataset = adapter.load(split="validation", max_examples=2)

        assert dataset.name == "hotpot_qa"
        assert len(dataset.qa_examples) == 2
        assert len(dataset.corpus) >= 2

    @patch("ragwatch.datasets.adapters.hotpotqa_adapter.load_hf_dataset")
    def test_qa_examples_have_supporting_titles(self, mock_load_hf) -> None:
        mock_load_hf.return_value = _make_mock_hotpotqa_dataset()

        adapter = HotpotQAAdapter()
        dataset = adapter.load(split="validation")

        example = dataset.qa_examples[0]
        assert "supporting_titles" in example.metadata
        assert len(example.metadata["supporting_titles"]) > 0

    @patch("ragwatch.datasets.adapters.hotpotqa_adapter.load_hf_dataset")
    def test_deduplicates_contexts(self, mock_load_hf) -> None:
        mock_load_hf.return_value = _make_mock_hotpotqa_dataset()

        adapter = HotpotQAAdapter()
        dataset = adapter.load(split="validation")

        # "The Bravery" paragraph appears in both examples
        doc_ids = [doc.doc_id for doc in dataset.corpus]
        assert len(doc_ids) == len(set(doc_ids))
