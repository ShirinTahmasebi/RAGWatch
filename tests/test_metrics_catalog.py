"""Tests for the central KPI catalog."""

import pytest

from ragwatch.metrics.catalog import (
    KPI_CATALOG,
    KPICategory,
    KPIDefinition,
    KPIId,
    KPISource,
    KPIStage,
    get_kpi_definition,
    list_kpis,
    list_kpis_by_category,
    list_kpis_by_source,
    list_kpis_by_stage,
)


class TestCatalogCompleteness:
    """Every KPIId must exist in KPI_CATALOG."""

    def test_all_kpi_ids_have_definitions(self):
        for kpi_id in KPIId:
            assert kpi_id in KPI_CATALOG, f"{kpi_id} missing from KPI_CATALOG"

    def test_no_extra_keys_in_catalog(self):
        for key in KPI_CATALOG:
            assert key in KPIId, f"Unknown key {key} in KPI_CATALOG"

    def test_no_duplicate_ids(self):
        ids = list(KPI_CATALOG.keys())
        assert len(ids) == len(set(ids))


class TestKPIDefinitionValidity:
    """Every definition must have valid fields."""

    @pytest.mark.parametrize("kpi_id", list(KPIId), ids=lambda k: k.value)
    def test_has_non_empty_description(self, kpi_id: KPIId):
        defn = KPI_CATALOG[kpi_id]
        assert defn.description, f"{kpi_id} has empty description"

    @pytest.mark.parametrize("kpi_id", list(KPIId), ids=lambda k: k.value)
    def test_valid_category(self, kpi_id: KPIId):
        defn = KPI_CATALOG[kpi_id]
        assert defn.category in KPICategory

    @pytest.mark.parametrize("kpi_id", list(KPIId), ids=lambda k: k.value)
    def test_valid_stage(self, kpi_id: KPIId):
        defn = KPI_CATALOG[kpi_id]
        assert defn.stage in KPIStage

    @pytest.mark.parametrize("kpi_id", list(KPIId), ids=lambda k: k.value)
    def test_valid_source(self, kpi_id: KPIId):
        defn = KPI_CATALOG[kpi_id]
        assert defn.source in KPISource

    @pytest.mark.parametrize("kpi_id", list(KPIId), ids=lambda k: k.value)
    def test_id_field_matches_key(self, kpi_id: KPIId):
        defn = KPI_CATALOG[kpi_id]
        assert defn.id == kpi_id


class TestHelperFunctions:
    """Test catalog helper functions."""

    def test_list_kpis_returns_all(self):
        all_kpis = list_kpis()
        assert len(all_kpis) == len(KPIId)
        assert all(isinstance(d, KPIDefinition) for d in all_kpis)

    def test_get_kpi_definition_by_enum(self):
        defn = get_kpi_definition(KPIId.RETRIEVAL_SCORE_MEAN)
        assert defn.id == KPIId.RETRIEVAL_SCORE_MEAN

    def test_get_kpi_definition_by_string(self):
        defn = get_kpi_definition("retrieval_score_mean")
        assert defn.id == KPIId.RETRIEVAL_SCORE_MEAN

    def test_get_kpi_definition_invalid_raises(self):
        with pytest.raises((KeyError, ValueError)):
            get_kpi_definition("nonexistent_kpi")

    def test_list_kpis_by_category(self):
        retrieval = list_kpis_by_category(KPICategory.RETRIEVAL_QUALITY)
        assert len(retrieval) == 10
        assert all(d.category == KPICategory.RETRIEVAL_QUALITY for d in retrieval)

    def test_list_kpis_by_category_string(self):
        runtime = list_kpis_by_category("runtime")
        assert len(runtime) == 3
        assert all(d.category == KPICategory.RUNTIME for d in runtime)

    def test_list_kpis_by_stage(self):
        retrieval = list_kpis_by_stage(KPIStage.RETRIEVAL)
        assert len(retrieval) > 0
        assert all(d.stage == KPIStage.RETRIEVAL for d in retrieval)

    def test_list_kpis_by_stage_string(self):
        db = list_kpis_by_stage("database")
        assert len(db) == 7
        assert all(d.stage == KPIStage.DATABASE for d in db)

    def test_list_kpis_by_source(self):
        pg = list_kpis_by_source(KPISource.POSTGRES)
        assert len(pg) > 0
        assert all(d.source == KPISource.POSTGRES for d in pg)

    def test_list_kpis_by_source_string(self):
        meta = list_kpis_by_source("metadata")
        assert len(meta) == 3
        assert all(d.source == KPISource.METADATA for d in meta)


class TestMetricCatalogIntegration:
    """Default metric lists must reference valid KPI IDs."""

    def test_retrieval_metrics_have_valid_ids(self):
        from ragwatch.metrics.retrieval import DEFAULT_RETRIEVAL_METRICS

        for m in DEFAULT_RETRIEVAL_METRICS:
            assert m.kpi_id in KPI_CATALOG
            assert m.name == m.kpi_id.value

    def test_generation_metrics_have_valid_ids(self):
        from ragwatch.metrics.generation import DEFAULT_GENERATION_METRICS

        for m in DEFAULT_GENERATION_METRICS:
            assert m.kpi_id in KPI_CATALOG
            assert m.name == m.kpi_id.value

    def test_runtime_metrics_have_valid_ids(self):
        from ragwatch.metrics.runtime import DEFAULT_RUNTIME_METRICS

        for m in DEFAULT_RUNTIME_METRICS:
            assert m.kpi_id in KPI_CATALOG
            assert m.name == m.kpi_id.value

    def test_db_metrics_have_valid_ids(self):
        from ragwatch.metrics.db_stats import DEFAULT_DB_METRICS

        for m in DEFAULT_DB_METRICS:
            assert m.kpi_id in KPI_CATALOG
            assert m.name == m.kpi_id.value

    def test_metric_output_names_unchanged(self):
        """Verify that metric .name values match the original string names."""
        from ragwatch.metrics.retrieval import DEFAULT_RETRIEVAL_METRICS
        from ragwatch.metrics.generation import DEFAULT_GENERATION_METRICS
        from ragwatch.metrics.runtime import DEFAULT_RUNTIME_METRICS

        expected_names = [
            "num_retrieved_documents",
            "retrieval_score_min",
            "retrieval_score_max",
            "retrieval_score_mean",
            "retrieval_score_std",
            "retrieval_score_range",
            "retrieval_score_gap_top1_top2",
            "context_length_chars",
            "unique_retrieved_sources",
            "retrieval_redundancy",
            "answer_length_chars",
            "answer_length_words",
            "answer_to_context_length_ratio",
            "total_latency_ms",
            "retrieval_latency_ms",
            "generation_latency_ms",
        ]
        all_metrics = (
            DEFAULT_RETRIEVAL_METRICS
            + DEFAULT_GENERATION_METRICS
            + DEFAULT_RUNTIME_METRICS
        )
        actual_names = [m.name for m in all_metrics]
        assert actual_names == expected_names
