from __future__ import annotations

from pif_research_platform.adapters.indicator_values import (
    BrowserRunner,
    ConnectorRegistry,
    FetchResult,
    IndicatorValueCollector,
    IndicatorValueConnector,
    UnsupportedIndicatorError,
)
from pif_research_platform.config import IndicatorDataSettings
from pif_research_platform.models import IndicatorSpec, TopicProfile


def _topic_profile() -> TopicProfile:
    return TopicProfile(
        domain="energy_transition",
        geography="Karnataka",
        research_goal="Assess EV charging readiness for the topic.",
        headline_metric_label="Transition readiness score",
        headline_unit="index",
        comparator_label="Previous readiness baseline",
        dimension_label="Capability attribution",
        keywords=["ev", "charging", "readiness", "infrastructure"],
        indicator_dimensions=["Infrastructure", "Demand", "Reliability", "Finance"],
        source_themes=["charging stations", "infrastructure rollout", "utilization", "project finance"],
    )


def _spec(indicator_id: str = "ACCESS") -> IndicatorSpec:
    return IndicatorSpec(
        indicator_id=indicator_id,
        label="Access-point expansion",
        frequency="quarterly",
        unit="count",
        sector_bucket="Infrastructure",
        rationale="Tracks network reach and density.",
        primary_source_name="Primary source",
        primary_source_url="https://example.org/primary",
        verification_source_name="Verification source",
        verification_source_url="https://example.org/verification",
    )


class StructuredConnector(IndicatorValueConnector):
    connector_id = "structured-live"

    def supports(self, topic_profile, spec) -> bool:
        del topic_profile
        return spec.indicator_id == "ACCESS"

    def fetch(self, context, *, browser_runner=None) -> FetchResult:
        del context, browser_runner
        return FetchResult(
            reference_period="2026-Q2",
            latest_value=42,
            prior_value=35,
            notes="Structured connector success.",
        )


class ErrorConnector(IndicatorValueConnector):
    connector_id = "error-live"

    def supports(self, topic_profile, spec) -> bool:
        del topic_profile
        return spec.indicator_id == "ACCESS"

    def fetch(self, context, *, browser_runner=None) -> FetchResult:
        del context, browser_runner
        raise UnsupportedIndicatorError("upstream portal unavailable")


class BrowserConnector(IndicatorValueConnector):
    connector_id = "browser-live"
    requires_browser = True

    def supports(self, topic_profile, spec) -> bool:
        del topic_profile
        return spec.indicator_id == "ACCESS"

    def fetch(self, context, *, browser_runner=None) -> FetchResult:
        assert browser_runner is not None
        return browser_runner.run(
            self.connector_id,
            lambda: FetchResult(
                reference_period="2026-Q2",
                latest_value=18,
                prior_value=12,
                notes=f"Browser connector success for {context.spec.indicator_id}.",
            ),
        )


class FakeBrowserRunner(BrowserRunner):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, connector_id: str, callback):
        self.calls.append(connector_id)
        return callback()


def test_collector_falls_back_to_fixture_when_no_connector_exists() -> None:
    collector = IndicatorValueCollector(
        settings=IndicatorDataSettings(provider="hybrid", fallback_provider="fixture"),
        registry=ConnectorRegistry(),
    )

    pack = collector.collect("Assess EV charging readiness in Karnataka", _topic_profile(), [_spec()], llm_runtime_label="template")

    assert pack.retrieval_mode == "fixture"
    assert pack.indicators[0].retrieval_mode == "fixture"
    assert pack.indicators[0].retrieval_status == "fallback"


def test_collector_uses_live_structured_connector_when_available() -> None:
    registry = ConnectorRegistry([StructuredConnector()])
    collector = IndicatorValueCollector(
        settings=IndicatorDataSettings(provider="live", fallback_provider="fixture"),
        registry=registry,
    )

    pack = collector.collect("Assess EV charging readiness in Karnataka", _topic_profile(), [_spec()], llm_runtime_label="template")

    assert pack.retrieval_mode == "live"
    assert pack.indicators[0].retrieval_mode == "live"
    assert pack.indicators[0].retrieval_status == "live"
    assert pack.indicators[0].connector_id == "structured-live"


def test_collector_supports_browser_connectors_via_runner() -> None:
    browser_runner = FakeBrowserRunner()
    collector = IndicatorValueCollector(
        settings=IndicatorDataSettings(
            provider="live",
            fallback_provider="fixture",
            allow_browser=True,
        ),
        registry=ConnectorRegistry([BrowserConnector()]),
        browser_runner=browser_runner,
    )

    pack = collector.collect("Assess EV charging readiness in Karnataka", _topic_profile(), [_spec()], llm_runtime_label="template")

    assert pack.retrieval_mode == "live"
    assert pack.indicators[0].connector_id == "browser-live"
    assert browser_runner.calls == ["browser-live"]


def test_collector_falls_back_when_live_connector_errors() -> None:
    collector = IndicatorValueCollector(
        settings=IndicatorDataSettings(provider="live", fallback_provider="fixture"),
        registry=ConnectorRegistry([ErrorConnector()]),
    )

    pack = collector.collect("Assess EV charging readiness in Karnataka", _topic_profile(), [_spec()], llm_runtime_label="template")

    assert pack.retrieval_mode == "fixture"
    assert pack.indicators[0].retrieval_status == "fallback"
    assert pack.indicators[0].connector_id == "error-live"


def test_collector_falls_back_when_browser_connector_is_disabled() -> None:
    collector = IndicatorValueCollector(
        settings=IndicatorDataSettings(
            provider="live",
            fallback_provider="fixture",
            allow_browser=False,
        ),
        registry=ConnectorRegistry([BrowserConnector()]),
        browser_runner=FakeBrowserRunner(),
    )

    pack = collector.collect("Assess EV charging readiness in Karnataka", _topic_profile(), [_spec()], llm_runtime_label="template")

    assert pack.retrieval_mode == "fixture"
    assert pack.indicators[0].retrieval_status == "fallback"
    assert pack.indicators[0].connector_id is None
