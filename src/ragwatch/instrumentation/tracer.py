"""RAGWatch tracer backed by OpenTelemetry."""

from __future__ import annotations

from typing import Any


def _check_otel_installed() -> None:
    try:
        import opentelemetry  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "OpenTelemetry packages are required for RAGWatch instrumentation. "
            "Install them with: pip install ragwatch[otel]"
        ) from e


class RAGWatchTracer:
    """Manages OpenTelemetry tracing for RAGWatch pipelines.

    Args:
        service_name: The service name for traces.
        exporter_type: One of "console", "otlp", or "none".
        otlp_endpoint: OTLP collector endpoint (only used when exporter_type="otlp").
    """

    def __init__(
        self,
        service_name: str = "ragwatch",
        exporter_type: str = "console",
        otlp_endpoint: str = "http://localhost:4317",
    ) -> None:
        _check_otel_installed()

        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if exporter_type == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif exporter_type == "otlp":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        elif exporter_type == "none":
            pass  # No exporter — spans are created but not exported
        else:
            raise ValueError(
                f"Unknown exporter_type: {exporter_type!r}. "
                "Supported: 'console', 'otlp', 'none'."
            )

        self._provider = provider
        self._tracer = provider.get_tracer("ragwatch", "0.3.0")

    @property
    def tracer(self):
        """Return the underlying OpenTelemetry tracer."""
        return self._tracer

    def start_span(self, name: str, attributes: dict[str, Any] | None = None):
        """Start a new span with optional attributes."""
        span = self._tracer.start_span(name, attributes=attributes)
        return span

    def shutdown(self) -> None:
        """Flush and shutdown the tracer provider."""
        self._provider.shutdown()
