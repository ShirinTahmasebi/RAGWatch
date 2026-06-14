"""Lightweight checks for the observability infrastructure config.

These tests only read files; they never start Docker, containers, or network
connections. If PyYAML is available the compose/config files are parsed,
otherwise simple file-content checks are used.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import yaml

    HAVE_YAML = True
except ImportError:  # pragma: no cover - exercised only without PyYAML
    HAVE_YAML = False

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
OTEL_CONFIG = REPO_ROOT / "infra/observability/otel-collector-config.yaml"
PROMETHEUS_CONFIG = REPO_ROOT / "infra/observability/prometheus.yml"
GRAFANA_DATASOURCES = (
    REPO_ROOT
    / "infra/observability/grafana/provisioning/datasources/datasources.yml"
)
ENV_TEMPLATE = REPO_ROOT / ".env.template"

OBSERVABILITY_SERVICES = ["otel-collector", "prometheus", "tempo", "grafana"]
REQUIRED_PORTS = ["3000", "9090", "3200", "4317", "4318"]
OBSERVABILITY_ENV_VARS = [
    "RAGWATCH_OTEL_ENABLED",
    "RAGWATCH_OTEL_EXPORTER",
    "RAGWATCH_OTEL_ENDPOINT",
    "RAGWATCH_OTEL_SERVICE_NAME",
]


def test_root_compose_file_exists() -> None:
    assert COMPOSE_FILE.is_file()


def test_root_compose_contains_observability_services() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    for service in OBSERVABILITY_SERVICES:
        assert service in text, f"missing service: {service}"


@pytest.mark.skipif(not HAVE_YAML, reason="PyYAML not installed")
def test_observability_services_use_profile() -> None:
    config = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    services = config["services"]
    for service in OBSERVABILITY_SERVICES:
        assert service in services, f"missing service: {service}"
        profiles = services[service].get("profiles", [])
        assert "observability" in profiles, (
            f"{service} should use the observability profile"
        )
    # Postgres must remain available without any profile.
    assert "profiles" not in services["postgres"]


def test_observability_profile_present_in_compose_text() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    assert "observability" in text


def test_required_ports_present_in_compose() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    for port in REQUIRED_PORTS:
        assert port in text, f"missing port: {port}"


def test_otel_collector_config_exists() -> None:
    assert OTEL_CONFIG.is_file()
    if HAVE_YAML:
        config = yaml.safe_load(OTEL_CONFIG.read_text(encoding="utf-8"))
        assert "otlp" in config["receivers"]
        protocols = config["receivers"]["otlp"]["protocols"]
        assert "grpc" in protocols
        assert "http" in protocols


def test_prometheus_config_exists() -> None:
    assert PROMETHEUS_CONFIG.is_file()
    if HAVE_YAML:
        config = yaml.safe_load(PROMETHEUS_CONFIG.read_text(encoding="utf-8"))
        assert config["scrape_configs"]


def test_grafana_datasources_exists() -> None:
    assert GRAFANA_DATASOURCES.is_file()
    text = GRAFANA_DATASOURCES.read_text(encoding="utf-8")
    assert "http://prometheus:9090" in text
    assert "http://tempo:3200" in text


def test_env_template_contains_observability_vars() -> None:
    text = ENV_TEMPLATE.read_text(encoding="utf-8")
    for var in OBSERVABILITY_ENV_VARS:
        assert var in text, f"missing env var: {var}"
