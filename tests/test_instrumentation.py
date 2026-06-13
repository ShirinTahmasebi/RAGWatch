"""Tests for OpenTelemetry instrumentation."""

import pytest

from ragwatch.core.schema import Document, RAGRun
from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.generators.heuristic_generator import HeuristicGenerator
from ragwatch.retrievers.tfidf_retriever import TfidfRetriever

try:
    import opentelemetry  # noqa: F401

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


def _build_components():
    documents = [
        Document(doc_id="d1", text="Paris is the capital of France."),
        Document(doc_id="d2", text="Berlin is the capital of Germany."),
        Document(doc_id="d3", text="Tokyo is the capital of Japan."),
    ]
    retriever = TfidfRetriever()
    retriever.index(documents)
    generator = HeuristicGenerator()
    return retriever, generator


class TestBasicClientWithoutTracer:
    """Verify BasicRAGClient still works without a tracer."""

    def test_run_returns_rag_run(self) -> None:
        retriever, generator = _build_components()
        client = BasicRAGClient(retriever=retriever, generator=generator)
        result = client.run("capital of France", top_k=2)

        assert isinstance(result, RAGRun)
        assert result.query == "capital of France"
        assert result.latency_ms >= 0
        assert "traced" not in result.metadata

    def test_run_without_tracer_has_results(self) -> None:
        retriever, generator = _build_components()
        client = BasicRAGClient(retriever=retriever, generator=generator)
        result = client.run("capital of France", top_k=2)

        assert len(result.retrieved_documents) >= 1
        assert result.generation.answer


@pytest.mark.skipif(not HAS_OTEL, reason="opentelemetry not installed")
class TestBasicClientWithTracer:
    """Verify BasicRAGClient works with a tracer."""

    def test_run_with_none_exporter(self) -> None:
        from ragwatch.instrumentation.tracer import RAGWatchTracer

        retriever, generator = _build_components()
        tracer = RAGWatchTracer(service_name="test", exporter_type="none")
        client = BasicRAGClient(retriever=retriever, generator=generator, tracer=tracer)
        result = client.run("capital of France", top_k=2)

        assert isinstance(result, RAGRun)
        assert result.query == "capital of France"
        assert result.metadata.get("traced") is True
        tracer.shutdown()

    def test_traced_run_has_retrieved_documents(self) -> None:
        from ragwatch.instrumentation.tracer import RAGWatchTracer

        retriever, generator = _build_components()
        tracer = RAGWatchTracer(service_name="test", exporter_type="none")
        client = BasicRAGClient(retriever=retriever, generator=generator, tracer=tracer)
        result = client.run("capital of France", top_k=2)

        assert len(result.retrieved_documents) >= 1
        assert result.retrieved_documents[0].document.doc_id == "d1"
        tracer.shutdown()

    def test_traced_run_has_generation(self) -> None:
        from ragwatch.instrumentation.tracer import RAGWatchTracer

        retriever, generator = _build_components()
        tracer = RAGWatchTracer(service_name="test", exporter_type="none")
        client = BasicRAGClient(retriever=retriever, generator=generator, tracer=tracer)
        result = client.run("capital of France", top_k=2)

        assert result.generation.answer
        assert result.generation.generator_name == "heuristic"
        tracer.shutdown()

    def test_traced_run_metadata(self) -> None:
        from ragwatch.instrumentation.tracer import RAGWatchTracer

        retriever, generator = _build_components()
        tracer = RAGWatchTracer(service_name="test", exporter_type="none")
        client = BasicRAGClient(retriever=retriever, generator=generator, tracer=tracer)
        result = client.run("capital of France", top_k=3)

        assert result.metadata["top_k"] == 3
        assert result.metadata["retriever"] == "TfidfRetriever"
        assert result.metadata["generator"] == "HeuristicGenerator"
        assert result.metadata["num_retrieved_documents"] >= 1
        tracer.shutdown()


@pytest.mark.skipif(not HAS_OTEL, reason="opentelemetry not installed")
class TestInstrumentedClientDirect:
    """Test using InstrumentedRAGClient directly."""

    def test_instrument_client_factory(self) -> None:
        from ragwatch.instrumentation.instrumented_client import instrument_client
        from ragwatch.instrumentation.tracer import RAGWatchTracer

        retriever, generator = _build_components()
        tracer = RAGWatchTracer(service_name="test", exporter_type="none")
        client = instrument_client(retriever=retriever, generator=generator, tracer=tracer)
        result = client.run("capital of Germany", top_k=2)

        assert isinstance(result, RAGRun)
        assert result.metadata.get("traced") is True
        assert result.retrieved_documents[0].document.doc_id == "d2"
        tracer.shutdown()


@pytest.mark.skipif(not HAS_OTEL, reason="opentelemetry not installed")
class TestTracerConfiguration:
    """Test RAGWatchTracer configuration."""

    def test_invalid_exporter_type_raises(self) -> None:
        from ragwatch.instrumentation.tracer import RAGWatchTracer

        with pytest.raises(ValueError, match="Unknown exporter_type"):
            RAGWatchTracer(service_name="test", exporter_type="invalid")

    def test_console_exporter_creates_tracer(self) -> None:
        from ragwatch.instrumentation.tracer import RAGWatchTracer

        tracer = RAGWatchTracer(service_name="test-console", exporter_type="console")
        assert tracer.tracer is not None
        tracer.shutdown()


class TestMissingOtelDependency:
    """Test behavior when opentelemetry is not installed."""

    def test_import_without_otel_gives_helpful_error(self, monkeypatch) -> None:
        """Simulate missing opentelemetry by patching the import check."""
        import ragwatch.instrumentation.tracer as tracer_module

        def mock_check():
            raise ImportError(
                "OpenTelemetry packages are required for RAGWatch instrumentation. "
                "Install them with: pip install ragwatch[otel]"
            )

        monkeypatch.setattr(tracer_module, "_check_otel_installed", mock_check)

        with pytest.raises(ImportError, match="pip install ragwatch"):
            tracer_module.RAGWatchTracer(service_name="test")
