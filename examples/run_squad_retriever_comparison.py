"""Compare retriever backends on SQuAD across multiple top-k settings.

Requirements:
    pip install -e ".[datasets]"            # TF-IDF only
    pip install -e ".[datasets,vectordb]"   # also Chroma + Qdrant + pgvector

Usage:
    python examples/run_squad_retriever_comparison.py

Output directory is mode-aware so base and semantic runs do not overwrite each
other:
    outputs/comparisons/squad_retrievers_base/              (default)
    outputs/comparisons/squad_retrievers_semantic_local/    (semantic, local)
    outputs/comparisons/squad_retrievers_semantic_openai/   (semantic, openai)
    outputs/comparisons/squad_retrievers_semantic_azure_openai/

Each directory contains:
    combined_kpis.csv
    comparison_summary.csv
    comparison_summary.json
    experiments/<retriever>_top<k>/...

Set RAGWATCH_COMPARISON_OUTPUT_DIR to override the output directory verbatim.

TF-IDF always runs. Chroma and Qdrant are included only if their optional
dependencies are installed. pgvector is included only when RAGWATCH_PGVECTOR_URL
is configured and Postgres is reachable; otherwise it is skipped gracefully.

To include pgvector:
    cp .env.template .env
    docker compose up -d postgres
    python examples/run_squad_retriever_comparison.py
"""

import csv
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.config.env import get_env, load_env
from ragwatch.core.schema import Document
from ragwatch.datasets.adapters.squad_adapter import SquadAdapter
from ragwatch.experiments.comparison import (
    ComparisonConfig,
    ComparisonRunner,
    RetrieverExperimentSpec,
    export_comparison,
)
from ragwatch.generators.heuristic_generator import HeuristicGenerator
from ragwatch.metrics.report import DBKPIEngine
from ragwatch.metrics.semantic import SemanticKPIEngine
from ragwatch.retrievers.tfidf_retriever import TfidfRetriever
from ragwatch.semantic.providers import (
    ENV_SEMANTIC_PROVIDER,
    PROVIDER_LOCAL,
    SemanticProviderConfigError,
    create_semantic_embedding_provider_from_env,
)

# Load .env so RAGWATCH_PGVECTOR_URL is available when present.
load_env()

PGVECTOR_COLLECTION_NAME = "squad_comparison_pgvector"

# Opt-in flag for including semantic KPIs in the comparison. Disabled by default
# to avoid accidental paid-API usage (OpenAI/Azure).
ENV_COMPUTE_SEMANTIC_KPIS = "RAGWATCH_COMPUTE_SEMANTIC_KPIS"

# Optional override for the comparison output directory. When set, it is used
# verbatim and the mode-specific directory naming is bypassed.
ENV_COMPARISON_OUTPUT_DIR = "RAGWATCH_COMPARISON_OUTPUT_DIR"

# Mode-aware output directory naming. Different modes write to different
# directories so a base run and a semantic run never overwrite each other.
COMPARISONS_ROOT = Path("outputs/comparisons")
COMPARISON_BASE_NAME = "squad_retrievers"
BASE_MODE_SUFFIX = "_base"
SEMANTIC_MODE_PREFIX = "_semantic_"

SEMANTIC_HINT = (
    "Continuing without semantic KPIs. To enable them, install dependencies and "
    'configure a provider:\n  pip install -e ".[semantic]"\n'
    "  RAGWATCH_COMPUTE_SEMANTIC_KPIS=true RAGWATCH_SEMANTIC_PROVIDER=local \\\n"
    "      python examples/run_squad_retriever_comparison.py"
)


def _semantic_kpis_requested() -> bool:
    """Return True if the opt-in env flag is set to a truthy value."""
    value = (get_env(ENV_COMPUTE_SEMANTIC_KPIS, default="false") or "").strip().lower()
    return value in ("1", "true", "yes", "on")


def try_create_semantic_kpi_engine() -> SemanticKPIEngine | None:
    """Create a semantic KPI engine from env, or None if unavailable.

    Returns ``None`` (printing a clear message) when the opt-in flag is unset or
    when the configured semantic provider cannot be created. Never raises, so
    the default comparison keeps working without semantic dependencies or keys.
    """
    if not _semantic_kpis_requested():
        return None

    try:
        provider = create_semantic_embedding_provider_from_env()
    except SemanticProviderConfigError as exc:
        print(f"Skipping semantic KPIs: {exc}")
        print(SEMANTIC_HINT)
        return None

    print("Semantic KPIs enabled for the comparison.")
    return SemanticKPIEngine(provider=provider)


def semantic_provider_name() -> str:
    """Return the configured semantic provider name (defaults to ``local``)."""
    return (
        get_env(ENV_SEMANTIC_PROVIDER, default=PROVIDER_LOCAL) or PROVIDER_LOCAL
    ).strip().lower()


def resolve_comparison_mode_name(
    semantic_enabled_successfully: bool,
    semantic_provider_name: str | None,
) -> str:
    """Return the mode-specific comparison name.

    Base runs use ``squad_retrievers_base``; successful semantic runs use
    ``squad_retrievers_semantic_<provider>``. If semantic KPIs were requested
    but could not be enabled, the base name is used because no semantic columns
    are present.
    """
    if semantic_enabled_successfully and semantic_provider_name:
        return f"{COMPARISON_BASE_NAME}{SEMANTIC_MODE_PREFIX}{semantic_provider_name}"
    return f"{COMPARISON_BASE_NAME}{BASE_MODE_SUFFIX}"


def resolve_comparison_output_dir(
    compute_semantic_kpis: bool,
    semantic_provider_name: str | None,
    semantic_enabled_successfully: bool,
) -> Path:
    """Resolve the comparison output directory for the current run mode.

    A ``RAGWATCH_COMPARISON_OUTPUT_DIR`` override, when set, is used verbatim.
    Otherwise the directory is chosen from the run mode so that base and
    semantic runs do not overwrite each other.
    """
    override = get_env(ENV_COMPARISON_OUTPUT_DIR)
    if override:
        print(f"Using custom comparison output directory: {override}")
        return Path(override)

    mode_name = resolve_comparison_mode_name(
        semantic_enabled_successfully, semantic_provider_name
    )
    return COMPARISONS_ROOT / mode_name


def try_create_pgvector_spec(
    corpus: list[Document],
    embedding_model: object | None,
    generator: HeuristicGenerator,
) -> RetrieverExperimentSpec | None:
    """Build a pgvector spec only when Postgres is configured and reachable.

    Returns ``None`` (and prints a clear reason) when ``RAGWATCH_PGVECTOR_URL``
    is unset, an embedding model is unavailable, or connecting/indexing fails.
    On success, the spec computes a DB KPI snapshot per setting.
    """
    db_url = get_env("RAGWATCH_PGVECTOR_URL")
    if not db_url:
        print(
            "Skipping pgvector: RAGWATCH_PGVECTOR_URL is not set or Postgres is "
            "not reachable.\n"
            "To include pgvector, run:\n"
            "  cp .env.template .env\n"
            "  docker compose up -d postgres"
        )
        return None

    if embedding_model is None:
        print("Skipping pgvector: no embedding model is available.")
        return None

    try:
        from ragwatch.retrievers.vector.pgvector_retriever import PgVectorRetriever

        pgvector = PgVectorRetriever(
            embedding_model=embedding_model,
            collection_name=PGVECTOR_COLLECTION_NAME,
            connection_string=db_url,
            dataset_name="squad_validation",
            reset_collection=True,
        )
        pgvector.index(corpus)
    except Exception as exc:  # noqa: BLE001
        print(
            "Skipping pgvector: RAGWATCH_PGVECTOR_URL is not set or Postgres is "
            f"not reachable ({exc}).\n"
            "To include pgvector, run:\n"
            "  cp .env.template .env\n"
            "  docker compose up -d postgres"
        )
        return None

    return RetrieverExperimentSpec(
        name="pgvector",
        client=BasicRAGClient(retriever=pgvector, generator=generator),
        retriever_name="pgvector",
        generator_name="heuristic",
        db_kpi_engine=DBKPIEngine(),
        compute_db_kpis=True,
    )



def _build_specs(
    corpus: list[Document], generator: HeuristicGenerator
) -> list[RetrieverExperimentSpec]:
    """Build retriever specs, including optional backends when available."""
    specs: list[RetrieverExperimentSpec] = []

    # --- TF-IDF (always available) ---
    tfidf = TfidfRetriever()
    tfidf.index(corpus)
    specs.append(
        RetrieverExperimentSpec(
            name="tfidf",
            client=BasicRAGClient(retriever=tfidf, generator=generator),
            retriever_name="tfidf",
            generator_name="heuristic",
        )
    )

    # --- Optional vector backends (Chroma, Qdrant) ---
    embedding_model = None
    try:
        from ragwatch.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingModel,
        )

        embedding_model = SentenceTransformerEmbeddingModel()
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping vector backends (no embedding model): {exc}")

    if embedding_model is not None:
        try:
            from ragwatch.retrievers.vector.chroma_retriever import ChromaRetriever

            chroma = ChromaRetriever(
                embedding_model=embedding_model,
                collection_name="ragwatch_squad_comparison",
            )
            chroma.index(corpus)
            specs.append(
                RetrieverExperimentSpec(
                    name="chroma",
                    client=BasicRAGClient(retriever=chroma, generator=generator),
                    retriever_name="chroma",
                    generator_name="heuristic",
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping Chroma: {exc}")

        try:
            from ragwatch.retrievers.vector.qdrant_retriever import QdrantRetriever

            qdrant = QdrantRetriever(
                embedding_model=embedding_model,
                collection_name="ragwatch_squad_comparison",
            )
            qdrant.index(corpus)
            specs.append(
                RetrieverExperimentSpec(
                    name="qdrant",
                    client=BasicRAGClient(retriever=qdrant, generator=generator),
                    retriever_name="qdrant",
                    generator_name="heuristic",
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping Qdrant: {exc}")

    # --- Optional pgvector backend (only when Postgres is reachable) ---
    pgvector_spec = try_create_pgvector_spec(corpus, embedding_model, generator)
    if pgvector_spec is not None:
        specs.append(pgvector_spec)

    return specs


def main() -> None:
    config = ComparisonConfig(
        comparison_name="squad_retrievers",
        dataset_name="squad_validation",
        top_k_values=[3, 5],
        max_examples=20,
    )

    # --- Load dataset ---
    print(f"Loading SQuAD (validation, max {config.max_examples} examples)...")
    adapter = SquadAdapter()
    dataset = adapter.load(split="validation", max_examples=config.max_examples)
    print(f"Corpus: {len(dataset.corpus)} documents")
    print(f"QA examples: {len(dataset.qa_examples)} questions\n")

    # --- Build retriever specs over the same corpus ---
    generator = HeuristicGenerator()
    specs = _build_specs(dataset.corpus, generator)
    print(f"Retrievers in comparison: {[s.name for s in specs]}\n")

    # --- Optionally enable semantic KPIs for every spec (opt-in via env) ---
    semantic_engine = try_create_semantic_kpi_engine()
    semantic_enabled = semantic_engine is not None
    provider_name = semantic_provider_name() if semantic_enabled else None
    if semantic_engine is not None:
        for spec in specs:
            spec.semantic_kpi_engine = semantic_engine
            spec.compute_semantic_kpis = True

    # --- Make the run mode-aware (name + output directory) ---
    config.comparison_name = resolve_comparison_mode_name(
        semantic_enabled_successfully=semantic_enabled,
        semantic_provider_name=provider_name,
    )
    output_dir = resolve_comparison_output_dir(
        compute_semantic_kpis=_semantic_kpis_requested(),
        semantic_provider_name=provider_name,
        semantic_enabled_successfully=semantic_enabled,
    )

    # --- Run comparison ---
    runner = ComparisonRunner()
    print("Running comparison...")
    result = runner.run(
        qa_examples=dataset.qa_examples, specs=specs, config=config
    )

    # --- Export ---
    export_comparison(result, output_dir)

    # --- Summary ---
    combined_path = output_dir / "combined_kpis.csv"
    with combined_path.open() as f:
        num_rows = max(sum(1 for _ in csv.reader(f)) - 1, 0)  # minus header

    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"  Comparison:       {config.comparison_name}")
    print(f"  Experiment keys:  {result.experiment_keys}")
    print(f"  Num settings:     {result.num_experiments}")
    print(f"  combined_kpis rows: {num_rows}")
    print(f"  Summary CSV:      {output_dir / 'comparison_summary.csv'}")
    print()


if __name__ == "__main__":
    main()
