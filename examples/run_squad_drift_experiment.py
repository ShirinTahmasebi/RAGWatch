"""Backward-compatible SQuAD drift experiment (thin wrapper).

The dataset-agnostic entry point examples/run_drift_experiment.py is the
preferred research script. This wrapper simply defaults the dataset to SQuAD so
the historical command keeps working:

    python examples/run_squad_drift_experiment.py

It is equivalent to:

    RAGWATCH_DATASET=squad python examples/run_drift_experiment.py

All behavior (semantic KPIs by default, RAGWATCH_DISABLE_SEMANTIC_KPIS opt-out,
RAGWATCH_SEMANTIC_PROVIDER, RAGWATCH_DRIFT_OUTPUT_DIR, and the
outputs/drift/squad_tfidf_<mode>/ output directory) is inherited from the
generic script.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "examples"))

from run_drift_experiment import main  # noqa: E402

DEFAULT_DATASET = "squad"


if __name__ == "__main__":
    main(default_dataset=DEFAULT_DATASET)
