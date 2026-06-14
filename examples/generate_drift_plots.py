"""Generate paper-quality static figures from drift-experiment outputs.

This is the dataset-agnostic plot entry point. The default input directory is
derived from ``RAGWATCH_DATASET`` and the semantic mode, mirroring
examples/run_drift_experiment.py:

    RAGWATCH_DATASET=squad    python examples/generate_drift_plots.py
    RAGWATCH_DATASET=hotpotqa python examples/generate_drift_plots.py

Requirements:
    pip install -e ".[plots]"

By default, reads the drift summary from the dataset-/mode-aware directory and
writes figures under it:
    outputs/drift/squad_tfidf_semantic_local/drift_summary.csv
        ->  outputs/drift/squad_tfidf_semantic_local/figures/

Set RAGWATCH_SEMANTIC_PROVIDER / RAGWATCH_DISABLE_SEMANTIC_KPIS to match the run
mode. Set RAGWATCH_DRIFT_OUTPUT_DIR to read from a different drift directory
verbatim. Set RAGWATCH_DRIFT_FIGURE_OUTPUT_DIR to write figures to a custom
directory (used verbatim) instead of <drift dir>/figures/.

If the drift outputs are missing, run this first:
    python examples/run_drift_experiment.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.config.env import get_env, load_env
from ragwatch.datasets.registry import dataset_name_from_env
from ragwatch.experiments.output_naming import drift_output_dir
from ragwatch.plots.drift_plots import generate_all_drift_plots
from ragwatch.semantic.providers import (
    resolve_semantic_provider_name,
    semantic_kpis_enabled,
)

# Load .env so the optional override variables are available when present.
load_env()

DEFAULT_DATASET = "squad"

# Optional overrides for the input drift directory and the output figures
# directory. When unset, dataset-/mode-aware defaults are used.
ENV_DRIFT_OUTPUT_DIR = "RAGWATCH_DRIFT_OUTPUT_DIR"
ENV_DRIFT_FIGURE_OUTPUT_DIR = "RAGWATCH_DRIFT_FIGURE_OUTPUT_DIR"


def resolve_drift_input_dir(default_dataset: str = DEFAULT_DATASET) -> Path:
    """Resolve the drift directory to read the summary from.

    A ``RAGWATCH_DRIFT_OUTPUT_DIR`` override, when set, is used verbatim.
    Otherwise the directory is derived from the dataset name and semantic mode.
    """
    override = get_env(ENV_DRIFT_OUTPUT_DIR)
    if override:
        return Path(override)
    dataset_name = dataset_name_from_env(default_dataset)
    semantic_enabled = semantic_kpis_enabled()
    provider_name = (
        resolve_semantic_provider_name() if semantic_enabled else None
    )
    return drift_output_dir(dataset_name, semantic_enabled, provider_name)


def resolve_figure_output_dir() -> Path | None:
    """Resolve a verbatim figures output directory override, if any.

    Returns ``None`` when no override is set so the default
    ``<drift dir>/figures/`` is used.
    """
    override = get_env(ENV_DRIFT_FIGURE_OUTPUT_DIR)
    if override:
        return Path(override)
    return None


def main(default_dataset: str = DEFAULT_DATASET) -> None:
    drift_dir = resolve_drift_input_dir(default_dataset)
    output_dir = resolve_figure_output_dir()

    if not (drift_dir / "drift_summary.csv").exists():
        print(
            f"Drift summary not found in {drift_dir}.\n"
            "Run python examples/run_drift_experiment.py first."
        )
        return

    print(f"Reading drift summary from: {drift_dir}")
    created = generate_all_drift_plots(drift_dir, output_dir)

    target_dir = output_dir if output_dir is not None else drift_dir / "figures"
    print(f"\nGenerated {len(created)} figures in {target_dir}:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
