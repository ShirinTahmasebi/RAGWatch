"""Query database-level KPIs from pgvector/Postgres.

Requirements:
    pip install -e ".[vectordb]"
    docker compose up -d postgres
    cp .env.template .env

Usage:
    python examples/query_pgvector_kpis.py
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.config.env import get_env, load_env

load_env()


def main() -> None:
    db_url = get_env("RAGWATCH_PGVECTOR_URL")
    if not db_url:
        print("ERROR: RAGWATCH_PGVECTOR_URL is not set.")
        print()
        print("To run this example:")
        print("  1. cp .env.template .env")
        print("  2. docker compose up -d postgres")
        print("  3. Index some documents first:")
        print("     python examples/run_squad_pgvector.py")
        print("  4. python examples/query_pgvector_kpis.py")
        sys.exit(1)

    from ragwatch.metrics.report import DBKPIEngine, to_flat_dict

    engine = DBKPIEngine()
    report = engine.compute(connection_string=db_url)

    print("=" * 60)
    print("RAGWatch Database KPIs")
    print("=" * 60)
    print()

    for kpi in report.results:
        print(f"  {kpi.name:30s} = {kpi.value}")
    print()

    # Flat dict
    flat = to_flat_dict(report)
    print("Flat dict:")
    print(json.dumps(flat, indent=2, default=str))
    print()


if __name__ == "__main__":
    main()
