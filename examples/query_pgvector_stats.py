"""Query pgvector database statistics.

Connects to PostgreSQL and prints basic statistics about stored documents,
embeddings, and collections. Useful for verifying what's in the database
and for future DB-based KPI calculations.

Requirements:
    pip install -e ".[vectordb]"
    docker compose up -d postgres
    cp .env.template .env

Usage:
    python examples/query_pgvector_stats.py
"""

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
        print("  4. python examples/query_pgvector_stats.py")
        sys.exit(1)

    from ragwatch.storage.postgres import get_connection

    conn = get_connection(db_url)

    print("=" * 60)
    print("RAGWatch Database Statistics")
    print("=" * 60)
    print()

    with conn.cursor() as cur:
        # Total documents
        cur.execute("SELECT COUNT(*) FROM ragwatch_documents")
        doc_count = cur.fetchone()[0]
        print(f"Total documents: {doc_count}")

        # Total embeddings
        cur.execute("SELECT COUNT(*) FROM ragwatch_embeddings")
        emb_count = cur.fetchone()[0]
        print(f"Total embeddings: {emb_count}")

        # Number of collections
        cur.execute("SELECT COUNT(DISTINCT collection_name) FROM ragwatch_embeddings")
        coll_count = cur.fetchone()[0]
        print(f"Number of collections: {coll_count}")

        print()

        # Documents per dataset
        cur.execute("""
            SELECT COALESCE(dataset_name, '(none)') AS ds, COUNT(*)
            FROM ragwatch_documents
            GROUP BY dataset_name
            ORDER BY COUNT(*) DESC
        """)
        rows = cur.fetchall()
        if rows:
            print("Documents per dataset:")
            for ds, count in rows:
                print(f"  {ds}: {count}")
            print()

        # Documents per source
        cur.execute("""
            SELECT COALESCE(source, '(none)') AS src, COUNT(*)
            FROM ragwatch_documents
            GROUP BY source
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
        if rows:
            print("Documents per source (top 10):")
            for src, count in rows:
                print(f"  {src}: {count}")
            print()

        # Average document text length
        cur.execute("SELECT AVG(LENGTH(text)) FROM ragwatch_documents")
        avg_len = cur.fetchone()[0]
        if avg_len is not None:
            print(f"Average document text length: {avg_len:.0f} chars")
            print()

        # Embeddings per collection
        cur.execute("""
            SELECT collection_name, embedding_model, embedding_dim, COUNT(*)
            FROM ragwatch_embeddings
            GROUP BY collection_name, embedding_model, embedding_dim
            ORDER BY COUNT(*) DESC
        """)
        rows = cur.fetchall()
        if rows:
            print("Embeddings per collection:")
            for coll, model, dim, count in rows:
                print(f"  {coll} (model={model}, dim={dim}): {count}")
            print()

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
