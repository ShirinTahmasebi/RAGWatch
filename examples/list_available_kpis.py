"""List all available KPIs from the RAGWatch catalog, grouped by category."""

from ragwatch.metrics.catalog import KPICategory, list_kpis_by_category


def main() -> None:
    print("=" * 60)
    print("RAGWatch KPI Catalog")
    print("=" * 60)
    print()

    for category in KPICategory:
        kpis = list_kpis_by_category(category)
        if not kpis:
            continue
        print(f"{category.value}")
        for defn in kpis:
            print(f"  - {defn.id.value}")
            print(f"      stage={defn.stage}, source={defn.source}")
            print(f"      {defn.description}")
        print()

    from ragwatch.metrics.catalog import list_kpis

    total = len(list_kpis())
    print(f"Total: {total} KPIs registered")


if __name__ == "__main__":
    main()
