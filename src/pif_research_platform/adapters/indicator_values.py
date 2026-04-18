from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..config import IndicatorDataSettings
from ..models import IndicatorObservation, IndicatorPack, IndicatorSpec, TopicProfile, utc_now
from ..topic_intelligence import build_indicator_observations


class IndicatorCollectionError(RuntimeError):
    pass


class UnsupportedIndicatorError(IndicatorCollectionError):
    pass


@dataclass(frozen=True, slots=True)
class FetchContext:
    topic: str
    topic_profile: TopicProfile
    spec: IndicatorSpec
    timeout_seconds: float
    browser_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class FetchResult:
    reference_period: str
    latest_value: float
    prior_value: float
    notes: str = ""
    primary_source_name: str | None = None
    primary_source_url: str | None = None
    verification_source_name: str | None = None
    verification_source_url: str | None = None


class BrowserRunner:
    def run(self, connector_id: str, callback: Callable[[], FetchResult]) -> FetchResult:
        del connector_id
        return callback()


class IndicatorValueConnector:
    connector_id = "connector"
    requires_browser = False

    def supports(self, topic_profile: TopicProfile, spec: IndicatorSpec) -> bool:
        raise NotImplementedError

    def fetch(
        self,
        context: FetchContext,
        *,
        browser_runner: BrowserRunner | None = None,
    ) -> FetchResult:
        raise NotImplementedError


@dataclass(slots=True)
class ConnectorRegistry:
    connectors: list[IndicatorValueConnector] = field(default_factory=list)

    def register(self, connector: IndicatorValueConnector) -> None:
        self.connectors.append(connector)

    def resolve(self, topic_profile: TopicProfile, spec: IndicatorSpec) -> IndicatorValueConnector | None:
        for connector in self.connectors:
            if connector.supports(topic_profile, spec):
                return connector
        return None


@dataclass(slots=True)
class IndicatorValueCollector:
    settings: IndicatorDataSettings
    registry: ConnectorRegistry = field(default_factory=ConnectorRegistry)
    browser_runner: BrowserRunner | None = None

    def collect(
        self,
        topic: str,
        topic_profile: TopicProfile,
        indicator_plan: list[IndicatorSpec],
        *,
        llm_runtime_label: str,
    ) -> IndicatorPack:
        fixture_map = {
            item.indicator_id: item
            for item in build_indicator_observations(topic_profile, topic, indicator_plan)
        }
        observations: list[IndicatorObservation] = []
        live_count = 0
        fallback_count = 0

        for spec in indicator_plan:
            fixture = fixture_map.get(spec.indicator_id)
            if fixture is None:
                raise IndicatorCollectionError(f"Fixture data missing for indicator '{spec.indicator_id}'")
            observation = self._collect_one(topic, topic_profile, spec, fixture)
            observations.append(observation)
            if observation.retrieval_mode == "live":
                live_count += 1
            if observation.retrieval_status == "fallback":
                fallback_count += 1

        pack_mode = self._pack_retrieval_mode(observations)
        metadata = {
            "topic": topic,
            "domain": topic_profile.domain,
            "reference_period": observations[0].reference_period if observations else "Current cycle",
            "mode": pack_mode,
            "provider": self.settings.provider,
            "fallback_provider": self.settings.fallback_provider,
            "llm_runtime": llm_runtime_label,
            "browser_allowed": self.settings.allow_browser,
            "live_indicator_count": live_count,
            "fixture_indicator_count": len(observations) - live_count,
            "fallback_indicator_count": fallback_count,
            "note": self._pack_note(pack_mode, fallback_count),
        }
        return IndicatorPack(retrieval_mode=pack_mode, indicators=observations, metadata=metadata)

    def _collect_one(
        self,
        topic: str,
        topic_profile: TopicProfile,
        spec: IndicatorSpec,
        fixture: IndicatorObservation,
    ) -> IndicatorObservation:
        provider = self.settings.provider.lower()
        if provider == "fixture":
            return self._fixture_observation(spec, fixture, status="fixture", reason=None)

        connector = self.registry.resolve(topic_profile, spec)
        if connector is None:
            return self._fallback_or_raise(spec, fixture, "no registered live connector")

        if connector.requires_browser and not self.settings.allow_browser:
            return self._fallback_or_raise(spec, fixture, "browser-backed retrieval is disabled")

        context = FetchContext(
            topic=topic,
            topic_profile=topic_profile,
            spec=spec,
            timeout_seconds=self.settings.timeout_seconds,
            browser_timeout_seconds=self.settings.browser_timeout_seconds,
        )
        try:
            result = connector.fetch(context, browser_runner=self.browser_runner)
            self._validate_live_result(spec, result)
            return self._live_observation(spec, result, connector.connector_id)
        except Exception as exc:
            return self._fallback_or_raise(spec, fixture, str(exc), connector_id=connector.connector_id)

    def _fallback_or_raise(
        self,
        spec: IndicatorSpec,
        fixture: IndicatorObservation,
        reason: str,
        *,
        connector_id: str | None = None,
    ) -> IndicatorObservation:
        if self.settings.fallback_provider.lower() != "fixture":
            raise IndicatorCollectionError(
                f"Live retrieval failed for indicator '{spec.indicator_id}' and fixture fallback is disabled: {reason}"
            )
        observation = self._fixture_observation(spec, fixture, status="fallback", reason=reason)
        observation.connector_id = connector_id
        return observation

    def _fixture_observation(
        self,
        spec: IndicatorSpec,
        fixture: IndicatorObservation,
        *,
        status: str,
        reason: str | None,
    ) -> IndicatorObservation:
        note = "Deterministic fixture-backed observation used for offline-safe runs."
        if status == "fallback" and reason:
            note = f"Fixture fallback applied after live retrieval could not complete: {reason}."
        return self._build_observation(
            spec,
            reference_period=fixture.reference_period,
            latest_value=fixture.latest_value,
            prior_value=fixture.prior_value,
            notes=note,
            primary_source_name=fixture.primary_source_name,
            primary_source_url=fixture.primary_source_url,
            verification_source_name=fixture.verification_source_name,
            verification_source_url=fixture.verification_source_url,
            retrieval_mode="fixture",
            retrieval_status=status,
            connector_id=None,
        )

    def _live_observation(
        self,
        spec: IndicatorSpec,
        result: FetchResult,
        connector_id: str,
    ) -> IndicatorObservation:
        note = result.notes.strip() or "Live connector retrieved this observation; manual validation is still required."
        return self._build_observation(
            spec,
            reference_period=result.reference_period,
            latest_value=result.latest_value,
            prior_value=result.prior_value,
            notes=note,
            primary_source_name=result.primary_source_name or spec.primary_source_name,
            primary_source_url=result.primary_source_url or spec.primary_source_url,
            verification_source_name=result.verification_source_name or spec.verification_source_name,
            verification_source_url=result.verification_source_url or spec.verification_source_url,
            retrieval_mode="live",
            retrieval_status="live",
            connector_id=connector_id,
        )

    def _build_observation(
        self,
        spec: IndicatorSpec,
        *,
        reference_period: str,
        latest_value: float,
        prior_value: float,
        notes: str,
        primary_source_name: str,
        primary_source_url: str,
        verification_source_name: str,
        verification_source_url: str,
        retrieval_mode: str,
        retrieval_status: str,
        connector_id: str | None,
    ) -> IndicatorObservation:
        latest = float(latest_value)
        prior = float(prior_value)
        return IndicatorObservation(
            indicator_id=spec.indicator_id,
            label=spec.label,
            sector_bucket=spec.sector_bucket,
            reference_period=reference_period,
            latest_value=latest,
            unit=spec.unit,
            prior_value=prior,
            delta=round(latest - prior, 2),
            primary_source_name=primary_source_name,
            primary_source_url=primary_source_url,
            verification_source_name=verification_source_name,
            verification_source_url=verification_source_url,
            notes=notes,
            retrieval_mode=retrieval_mode,
            retrieval_status=retrieval_status,
            connector_id=connector_id,
            retrieved_at=utc_now(),
        )

    def _validate_live_result(self, spec: IndicatorSpec, result: FetchResult) -> None:
        if not result.reference_period.strip():
            raise UnsupportedIndicatorError(f"live connector returned no reference period for '{spec.indicator_id}'")
        for field_name in ("latest_value", "prior_value"):
            value = getattr(result, field_name)
            if value is None:
                raise UnsupportedIndicatorError(
                    f"live connector returned no {field_name} for '{spec.indicator_id}'"
                )

    def _pack_retrieval_mode(self, observations: list[IndicatorObservation]) -> str:
        modes = {item.retrieval_mode for item in observations}
        if modes == {"live"}:
            return "live"
        if modes == {"fixture"}:
            return "fixture"
        return "mixed"

    def _pack_note(self, pack_mode: str, fallback_count: int) -> str:
        if pack_mode == "live":
            return "All indicator observations were retrieved through live connectors and recorded with provenance."
        if pack_mode == "mixed":
            return (
                f"Indicator observations include live retrieval and fixture fallback; "
                f"{fallback_count} indicator(s) used deterministic fallback."
            )
        return "All indicator observations used deterministic fixtures with explicit provenance metadata."


def build_indicator_value_collector(settings: IndicatorDataSettings) -> IndicatorValueCollector:
    return IndicatorValueCollector(settings=settings, registry=ConnectorRegistry(), browser_runner=BrowserRunner())
