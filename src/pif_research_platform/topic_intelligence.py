from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import IndicatorObservation, IndicatorSpec, SourceRecord, TopicProfile


@dataclass(frozen=True, slots=True)
class DomainConfig:
    domain: str
    keyword_triggers: tuple[str, ...]
    headline_metric_label: str
    headline_unit: str
    comparator_label: str
    dimension_label: str
    research_goal_template: str
    indicator_dimensions: tuple[str, ...]
    indicator_templates: tuple[dict[str, str], ...]
    source_themes: tuple[str, ...]


DOMAIN_CONFIGS = [
    DomainConfig(
        domain="macroeconomy",
        keyword_triggers=("economy", "gdp", "gsdp", "gdsp", "inflation", "exports", "manufacturing", "demand", "nowcast"),
        headline_metric_label="Economic momentum estimate",
        headline_unit="% YoY",
        comparator_label="Recent reference estimate",
        dimension_label="Growth driver attribution",
        research_goal_template="Assess near-term macroeconomic momentum and identify the strongest demand, production, and financing drivers for {topic}.",
        indicator_dimensions=("Demand", "Production", "Mobility", "Finance", "Trade"),
        indicator_templates=(
            {"indicator_id": "TAX", "label": "Tax collections growth", "frequency": "monthly", "unit": "% YoY", "sector_bucket": "Demand", "rationale": "Proxy for formal demand and fiscal buoyancy."},
            {"indicator_id": "POWER", "label": "Electricity demand growth", "frequency": "monthly", "unit": "% YoY", "sector_bucket": "Production", "rationale": "Proxy for industrial and services throughput."},
            {"indicator_id": "REG", "label": "Registrations growth", "frequency": "monthly", "unit": "% YoY", "sector_bucket": "Demand", "rationale": "Captures household and fleet demand conditions."},
            {"indicator_id": "LOGISTICS", "label": "Goods movement growth", "frequency": "monthly", "unit": "% YoY", "sector_bucket": "Mobility", "rationale": "Tracks logistics and supply-chain intensity."},
            {"indicator_id": "CREDIT", "label": "Credit growth", "frequency": "monthly", "unit": "% YoY", "sector_bucket": "Finance", "rationale": "Signals financing conditions for firms and households."},
            {"indicator_id": "TRADE", "label": "External demand growth", "frequency": "monthly", "unit": "% YoY", "sector_bucket": "Trade", "rationale": "Tracks export-linked demand conditions."},
        ),
        source_themes=("mixed-frequency monitoring", "industrial activity", "financial conditions", "policy-note writing"),
    ),
    DomainConfig(
        domain="environment",
        keyword_triggers=(
            "air pollution",
            "air quality",
            "pollution",
            "emissions",
            "aqi",
            "pm2.5",
            "pm10",
            "environment",
            "environmental",
            "smog",
            "waste",
            "water quality",
            "climate",
            "heatwave",
        ),
        headline_metric_label="Environmental risk score",
        headline_unit="index",
        comparator_label="Previous risk baseline",
        dimension_label="Risk driver attribution",
        research_goal_template="Assess environmental exposure, emissions drivers, and implementation priorities for {topic}.",
        indicator_dimensions=("Exposure", "Emissions", "Mobility", "Enforcement", "Health"),
        indicator_templates=(
            {"indicator_id": "AQI", "label": "AQI pressure trend", "frequency": "daily", "unit": "index", "sector_bucket": "Exposure", "rationale": "Tracks the severity and persistence of poor-air episodes."},
            {"indicator_id": "PM25", "label": "PM2.5 concentration trend", "frequency": "daily", "unit": "ug/m3", "sector_bucket": "Exposure", "rationale": "Measures direct particulate exposure risk for residents."},
            {"indicator_id": "EMISSIONS", "label": "Industrial emission pressure", "frequency": "monthly", "unit": "index", "sector_bucket": "Emissions", "rationale": "Captures stationary-source emission pressure and compliance gaps."},
            {"indicator_id": "TRAFFIC", "label": "Transport emission pressure", "frequency": "monthly", "unit": "index", "sector_bucket": "Mobility", "rationale": "Tracks transport-linked pollution pressure and congestion spillovers."},
            {"indicator_id": "COMPLIANCE", "label": "Compliance enforcement coverage", "frequency": "monthly", "unit": "% share", "sector_bucket": "Enforcement", "rationale": "Measures whether inspection and enforcement intensity is keeping pace with the risk."},
            {"indicator_id": "HEALTH", "label": "Respiratory case-load pressure", "frequency": "monthly", "unit": "index", "sector_bucket": "Health", "rationale": "Signals whether environmental exposure is translating into health-system stress."},
        ),
        source_themes=("air quality management", "exposure assessment", "emissions control", "implementation enforcement"),
    ),
    DomainConfig(
        domain="energy_transition",
        keyword_triggers=("energy", "electricity", "grid", "renewable", "solar", "wind", "battery", "charging", "ev"),
        headline_metric_label="Transition readiness score",
        headline_unit="index",
        comparator_label="Previous readiness baseline",
        dimension_label="Capability attribution",
        research_goal_template="Assess transition readiness, operating constraints, and implementation sequencing for {topic}.",
        indicator_dimensions=("Infrastructure", "Demand", "Reliability", "Finance", "Delivery"),
        indicator_templates=(
            {"indicator_id": "CAPACITY", "label": "Installed clean-capacity growth", "frequency": "quarterly", "unit": "% YoY", "sector_bucket": "Infrastructure", "rationale": "Measures build-out pace of enabling infrastructure."},
            {"indicator_id": "UTILIZATION", "label": "Asset utilization improvement", "frequency": "quarterly", "unit": "index", "sector_bucket": "Reliability", "rationale": "Signals whether infrastructure is translating into usable service."},
            {"indicator_id": "ACCESS", "label": "Access-point expansion", "frequency": "quarterly", "unit": "count", "sector_bucket": "Infrastructure", "rationale": "Tracks network reach and density."},
            {"indicator_id": "ADOPTION", "label": "Adoption growth", "frequency": "quarterly", "unit": "% YoY", "sector_bucket": "Demand", "rationale": "Captures uptake by users or firms."},
            {"indicator_id": "FINANCE", "label": "Project-finance approvals", "frequency": "quarterly", "unit": "count", "sector_bucket": "Finance", "rationale": "Signals capital availability for scale-up."},
            {"indicator_id": "DELIVERY", "label": "Implementation completion rate", "frequency": "quarterly", "unit": "% share", "sector_bucket": "Delivery", "rationale": "Captures execution reliability of approved projects."},
        ),
        source_themes=("transition finance", "grid readiness", "infrastructure rollout", "delivery bottlenecks"),
    ),
    DomainConfig(
        domain="mobility",
        keyword_triggers=("mobility", "transport", "traffic", "congestion", "transit", "rail", "road", "freight"),
        headline_metric_label="Mobility resilience score",
        headline_unit="index",
        comparator_label="Previous operating baseline",
        dimension_label="System driver attribution",
        research_goal_template="Assess operational resilience, user demand, and infrastructure bottlenecks for {topic}.",
        indicator_dimensions=("Demand", "Capacity", "Reliability", "Logistics", "Finance"),
        indicator_templates=(
            {"indicator_id": "RIDERSHIP", "label": "Ridership growth", "frequency": "monthly", "unit": "% YoY", "sector_bucket": "Demand", "rationale": "Tracks system usage and service pull."},
            {"indicator_id": "CAPACITY", "label": "Network capacity additions", "frequency": "quarterly", "unit": "count", "sector_bucket": "Capacity", "rationale": "Signals supply expansion."},
            {"indicator_id": "RELIABILITY", "label": "On-time reliability", "frequency": "monthly", "unit": "% share", "sector_bucket": "Reliability", "rationale": "Measures service reliability."},
            {"indicator_id": "LOGISTICS", "label": "Freight throughput growth", "frequency": "monthly", "unit": "% YoY", "sector_bucket": "Logistics", "rationale": "Captures freight-side system performance."},
            {"indicator_id": "FINANCE", "label": "Mobility investment approvals", "frequency": "quarterly", "unit": "count", "sector_bucket": "Finance", "rationale": "Signals delivery pipeline strength."},
        ),
        source_themes=("ridership recovery", "network reliability", "freight corridors", "transport governance"),
    ),
    DomainConfig(
        domain="social_policy",
        keyword_triggers=("health", "education", "nutrition", "skills", "hospital", "school", "learning"),
        headline_metric_label="Service readiness score",
        headline_unit="index",
        comparator_label="Previous service baseline",
        dimension_label="Service driver attribution",
        research_goal_template="Assess service delivery readiness, operational gaps, and policy sequencing for {topic}.",
        indicator_dimensions=("Access", "Quality", "Capacity", "Finance", "Delivery"),
        indicator_templates=(
            {"indicator_id": "ACCESS", "label": "Service access improvement", "frequency": "quarterly", "unit": "% share", "sector_bucket": "Access", "rationale": "Tracks access expansion for target users."},
            {"indicator_id": "QUALITY", "label": "Outcome quality trend", "frequency": "quarterly", "unit": "index", "sector_bucket": "Quality", "rationale": "Captures whether service quality is improving."},
            {"indicator_id": "CAPACITY", "label": "Workforce capacity growth", "frequency": "quarterly", "unit": "% YoY", "sector_bucket": "Capacity", "rationale": "Measures operational staffing and capacity."},
            {"indicator_id": "FINANCE", "label": "Budget release efficiency", "frequency": "quarterly", "unit": "% share", "sector_bucket": "Finance", "rationale": "Signals whether allocations are reaching programmes on time."},
            {"indicator_id": "DELIVERY", "label": "Implementation completion rate", "frequency": "quarterly", "unit": "% share", "sector_bucket": "Delivery", "rationale": "Tracks execution reliability of interventions."},
        ),
        source_themes=("service delivery", "implementation quality", "targeting", "programme operations"),
    ),
    DomainConfig(
        domain="generic_policy",
        keyword_triggers=(),
        headline_metric_label="Policy momentum score",
        headline_unit="index",
        comparator_label="Previous baseline",
        dimension_label="Priority driver attribution",
        research_goal_template="Assess the strongest drivers, implementation bottlenecks, and policy actions for {topic}.",
        indicator_dimensions=("Demand", "Capacity", "Delivery", "Finance", "Risk"),
        indicator_templates=(
            {"indicator_id": "SIGNAL1", "label": "Leading demand signal", "frequency": "quarterly", "unit": "index", "sector_bucket": "Demand", "rationale": "Tracks headline directional movement relevant to the topic."},
            {"indicator_id": "SIGNAL2", "label": "Capacity expansion signal", "frequency": "quarterly", "unit": "index", "sector_bucket": "Capacity", "rationale": "Measures operating capacity improvement."},
            {"indicator_id": "SIGNAL3", "label": "Delivery reliability signal", "frequency": "quarterly", "unit": "% share", "sector_bucket": "Delivery", "rationale": "Captures execution performance."},
            {"indicator_id": "SIGNAL4", "label": "Financing signal", "frequency": "quarterly", "unit": "% YoY", "sector_bucket": "Finance", "rationale": "Tracks resourcing conditions for policy delivery."},
            {"indicator_id": "SIGNAL5", "label": "Risk pressure signal", "frequency": "quarterly", "unit": "index", "sector_bucket": "Risk", "rationale": "Captures operational or policy risk pressure."},
        ),
        source_themes=("policy implementation", "monitoring frameworks", "governance design", "delivery management"),
    ),
]


def infer_topic_profile(topic: str) -> TopicProfile:
    topic_lower = topic.lower().replace("gdsp", "gsdp")
    config = DOMAIN_CONFIGS[-1]
    matched_score = 0
    for candidate in DOMAIN_CONFIGS:
        score = sum(1 for keyword in candidate.keyword_triggers if keyword in topic_lower)
        if score > matched_score:
            matched_score = score
            config = candidate
    geography = extract_geography(topic)
    keywords = list(dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", topic_lower)))[:8]
    return TopicProfile(
        domain=config.domain,
        analysis_mode="indicator_composite",
        audience="Joint Secretary / Programme Officer level",
        geography=geography,
        research_goal=config.research_goal_template.format(topic=topic),
        headline_metric_label=config.headline_metric_label,
        headline_unit=config.headline_unit,
        comparator_label=config.comparator_label,
        dimension_label=config.dimension_label,
        keywords=keywords or ["policy", "implementation", "evidence"],
        indicator_dimensions=list(config.indicator_dimensions),
        source_themes=list(config.source_themes),
    )


def infer_topic_profile_with_llm(topic: str, local_llm: Any) -> TopicProfile:
    fallback = infer_topic_profile(topic)
    fallback_payload = {
        "domain": fallback.domain,
        "geography": fallback.geography,
        "research_goal": fallback.research_goal,
        "headline_metric_label": fallback.headline_metric_label,
        "headline_unit": fallback.headline_unit,
        "comparator_label": fallback.comparator_label,
        "dimension_label": fallback.dimension_label,
        "keywords": fallback.keywords,
        "indicator_dimensions": fallback.indicator_dimensions,
        "source_themes": fallback.source_themes or _fallback_source_themes(topic),
    }
    result = local_llm.complete_json(
        system_prompt=(
            "You design topic-specific research profiles for a policy analysis workflow. Return strict JSON with keys "
            "'domain', 'geography', 'research_goal', 'headline_metric_label', 'headline_unit', 'comparator_label', "
            "'dimension_label', 'keywords', 'indicator_dimensions', and 'source_themes'. Create a short snake_case "
            "domain derived from the actual query instead of forcing it into a fixed taxonomy. Choose a headline metric "
            "and dimensions that fit the topic directly. Do not frame non-economic topics as GDP or macroeconomy unless "
            "the query explicitly asks for that."
        ),
        user_prompt=(
            f"Topic: {topic}\n"
            f"Heuristic fallback domain: {fallback.domain}\n"
            f"Heuristic fallback research goal: {fallback.research_goal}\n"
            "Return JSON only."
        ),
        fallback=fallback_payload,
    )
    return _merge_topic_profile(topic, fallback, result)


def extract_geography(topic: str) -> str | None:
    pattern = re.compile(
        r"\b(?:for|in|of)\s+([A-Za-z][A-Za-z0-9&,\-\s]{1,80}?)(?=\s+(?:using|with|based|through|via)\b|[.,;:]|$)",
        re.IGNORECASE,
    )
    match = pattern.search(topic)
    if match:
        value = match.group(1).strip().rstrip(".")
        return value.title() if value.islower() else value
    return None


def build_indicator_specs(profile: TopicProfile, topic: str) -> list[IndicatorSpec]:
    config = next((item for item in DOMAIN_CONFIGS if item.domain == profile.domain), None)
    if config is not None:
        return _specs_from_templates(config.indicator_templates, topic)
    return _generic_specs_from_profile(profile, topic)


def build_indicator_specs_with_llm(
    profile: TopicProfile,
    topic: str,
    local_llm: Any,
    *,
    max_indicators: int = 6,
) -> list[IndicatorSpec]:
    fallback_specs = build_indicator_specs(profile, topic)[:max_indicators]
    fallback_payload = {
        "indicators": [
            {
                "indicator_id": item.indicator_id,
                "label": item.label,
                "frequency": item.frequency,
                "unit": item.unit,
                "sector_bucket": item.sector_bucket,
                "rationale": item.rationale,
            }
            for item in fallback_specs
        ]
    }
    result = local_llm.complete_json(
        system_prompt=(
            "You design indicator baskets for policy-analysis workflows. Return strict JSON with one key: "
            "'indicators'. The value must be a list of 5 or 6 objects, each containing 'indicator_id', 'label', "
            "'frequency', 'unit', 'sector_bucket', and 'rationale'. Make the basket specific to the topic and the "
            "stated dimensions. Use short uppercase IDs. Prefer publicly observable operational indicators or good "
            "proxies. Do not include source URLs or citation text."
        ),
        user_prompt=(
            f"Topic: {topic}\n"
            f"Domain: {profile.domain}\n"
            f"Research goal: {profile.research_goal}\n"
            f"Headline metric: {profile.headline_metric_label}\n"
            f"Dimensions: {', '.join(profile.indicator_dimensions)}\n"
            "Return JSON only."
        ),
        fallback=fallback_payload,
    )
    return _merge_indicator_specs(topic, profile, fallback_specs, result.get("indicators"))


def build_indicator_observations(profile: TopicProfile, topic: str, indicator_plan: list[IndicatorSpec]) -> list[IndicatorObservation]:
    rng = np.random.default_rng(topic_seed(topic))
    observations: list[IndicatorObservation] = []
    reference_period = "2026-Q1" if profile.headline_unit == "% YoY" else "2026-Cycle-1"
    for index, spec in enumerate(indicator_plan):
        latest, prior = _generate_observation_values(rng, spec.unit, index)
        observations.append(
            IndicatorObservation(
                indicator_id=spec.indicator_id,
                label=spec.label,
                sector_bucket=spec.sector_bucket,
                reference_period=reference_period,
                latest_value=latest,
                unit=spec.unit,
                prior_value=prior,
                delta=round(latest - prior, 2),
                primary_source_name=spec.primary_source_name,
                primary_source_url=spec.primary_source_url,
                verification_source_name=spec.verification_source_name,
                verification_source_url=spec.verification_source_url,
                notes="Offline local-fixture mode; switch to live connectors for production evidence retrieval.",
            )
        )
    return observations


def build_literature_sources(profile: TopicProfile, topic: str, max_results: int = 8) -> list[SourceRecord]:
    slug = slugify(topic)
    geography = profile.geography or "the target geography"
    themes = profile.source_themes or list(domain_config(profile.domain).source_themes) or _fallback_source_themes(topic)
    results: list[SourceRecord] = []
    for index in range(max_results):
        source_id = f"S{index + 1}"
        theme = themes[index % len(themes)]
        year = 2020 + (index % 5)
        title = f"{theme.title()} for {topic}"
        summary = (
            f"This source explains how {theme} influences implementation outcomes for {topic} in {geography}."
        )
        methodology = (
            "Comparative policy analysis with operational indicator mapping and implementation sequencing."
        )
        relevance = (
            f"Useful for framing {profile.dimension_label.lower()} and practical recommendations for {topic}."
        )
        authors = [f"Author {index + 1}A", f"Author {index + 1}B"]
        url = f"https://example.org/{slug}/source-{index + 1}"
        alternate = f"https://example.org/{slug}/source-{index + 1}-alt"
        apa = f"{', '.join(authors)} ({year}). {title}. Example Policy Lab. {url}"
        results.append(
            SourceRecord(
                source_id=source_id,
                title=title,
                authors=authors,
                year=year,
                source_type="Policy Report" if index % 2 else "Working Paper",
                url=url,
                alternate_url=alternate,
                publisher="Example Policy Lab",
                summary=summary,
                methodology=methodology,
                relevance=relevance,
                apa_citation=apa,
            )
        )
    return results


def domain_config(domain: str) -> DomainConfig:
    for config in DOMAIN_CONFIGS:
        if config.domain == domain:
            return config
    return DOMAIN_CONFIGS[-1]


def topic_seed(topic: str) -> int:
    digest = hashlib.sha256(topic.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "topic"


def _specs_from_templates(templates: tuple[dict[str, str], ...], topic: str) -> list[IndicatorSpec]:
    slug = slugify(topic)
    specs: list[IndicatorSpec] = []
    for template in templates:
        indicator_slug = slugify(template["label"])
        specs.append(
            IndicatorSpec(
                indicator_id=template["indicator_id"],
                label=template["label"],
                frequency=template["frequency"],
                unit=template["unit"],
                sector_bucket=template["sector_bucket"],
                rationale=template["rationale"],
                primary_source_name=f"{template['label']} primary source",
                primary_source_url=f"https://example.org/{slug}/{indicator_slug}/primary",
                verification_source_name=f"{template['label']} verification source",
                verification_source_url=f"https://example.org/{slug}/{indicator_slug}/verification",
            )
        )
    return specs


def _generic_specs_from_profile(profile: TopicProfile, topic: str) -> list[IndicatorSpec]:
    slug = slugify(topic)
    dimensions = profile.indicator_dimensions or ["Demand", "Capacity", "Delivery", "Finance", "Risk"]
    specs: list[IndicatorSpec] = []
    for index, bucket in enumerate(dimensions[:6], start=1):
        label = _generic_indicator_label(bucket)
        indicator_slug = slugify(label)
        specs.append(
            IndicatorSpec(
                indicator_id=_safe_indicator_id(bucket, index),
                label=label,
                frequency=_default_frequency_for_dimension(bucket),
                unit=_default_unit_for_dimension(bucket),
                sector_bucket=bucket,
                rationale=f"Tracks how {bucket.lower()} conditions are shaping the current topic.",
                primary_source_name=f"{label} primary source",
                primary_source_url=f"https://example.org/{slug}/{indicator_slug}/primary",
                verification_source_name=f"{label} verification source",
                verification_source_url=f"https://example.org/{slug}/{indicator_slug}/verification",
            )
        )
    return specs


def _merge_topic_profile(topic: str, fallback: TopicProfile, payload: dict[str, Any]) -> TopicProfile:
    domain = _sanitize_domain_name(str(payload.get("domain") or fallback.domain))
    geography = str(payload.get("geography") or fallback.geography or "").strip() or fallback.geography
    research_goal = str(payload.get("research_goal") or fallback.research_goal).strip() or fallback.research_goal
    headline_metric_label = str(payload.get("headline_metric_label") or fallback.headline_metric_label).strip() or fallback.headline_metric_label
    headline_unit = str(payload.get("headline_unit") or fallback.headline_unit).strip() or fallback.headline_unit
    comparator_label = str(payload.get("comparator_label") or fallback.comparator_label).strip() or fallback.comparator_label
    dimension_label = str(payload.get("dimension_label") or fallback.dimension_label).strip() or fallback.dimension_label
    keywords = _sanitize_text_list(payload.get("keywords"), fallback.keywords or _fallback_keywords(topic), max_items=8)
    indicator_dimensions = _sanitize_dimension_list(payload.get("indicator_dimensions"), fallback.indicator_dimensions)
    source_themes = _sanitize_text_list(payload.get("source_themes"), fallback.source_themes or _fallback_source_themes(topic), max_items=6)
    return TopicProfile(
        domain=domain,
        analysis_mode="indicator_composite",
        audience=fallback.audience,
        geography=geography,
        research_goal=research_goal,
        headline_metric_label=headline_metric_label,
        headline_unit=headline_unit,
        comparator_label=comparator_label,
        dimension_label=dimension_label,
        keywords=keywords,
        indicator_dimensions=indicator_dimensions,
        source_themes=source_themes,
    )


def _merge_indicator_specs(
    topic: str,
    profile: TopicProfile,
    fallback_specs: list[IndicatorSpec],
    payload: Any,
) -> list[IndicatorSpec]:
    if not isinstance(payload, list):
        return fallback_specs
    merged: list[IndicatorSpec] = []
    slug = slugify(topic)
    for index, item in enumerate(payload[:6], start=1):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        sector_bucket = str(item.get("sector_bucket") or "").strip()
        if not label or not sector_bucket:
            continue
        frequency = _normalize_frequency(str(item.get("frequency") or "quarterly"))
        unit = str(item.get("unit") or _default_unit_for_dimension(sector_bucket)).strip() or _default_unit_for_dimension(sector_bucket)
        rationale = str(item.get("rationale") or f"Tracks how {sector_bucket.lower()} conditions are shaping the topic.").strip()
        indicator_id = _safe_indicator_id(str(item.get("indicator_id") or sector_bucket), index)
        indicator_slug = slugify(label)
        merged.append(
            IndicatorSpec(
                indicator_id=indicator_id,
                label=label,
                frequency=frequency,
                unit=unit,
                sector_bucket=sector_bucket,
                rationale=rationale,
                primary_source_name=f"{label} primary source",
                primary_source_url=f"https://example.org/{slug}/{indicator_slug}/primary",
                verification_source_name=f"{label} verification source",
                verification_source_url=f"https://example.org/{slug}/{indicator_slug}/verification",
            )
        )
    return merged if len(merged) >= 4 else fallback_specs


def _sanitize_domain_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "generic_policy"


def _sanitize_text_list(value: Any, fallback: list[str], *, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return fallback[:max_items]
    items: list[str] = []
    for raw in value:
        text = str(raw).strip()
        if text and text not in items:
            items.append(text)
        if len(items) >= max_items:
            break
    return items or fallback[:max_items]


def _sanitize_dimension_list(value: Any, fallback: list[str]) -> list[str]:
    items = _sanitize_text_list(value, fallback, max_items=6)
    return [item.title() for item in items]


def _fallback_keywords(topic: str) -> list[str]:
    tokens = list(dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", topic.lower())))
    return tokens[:8] or ["policy", "implementation", "evidence"]


def _fallback_source_themes(topic: str) -> list[str]:
    keywords = _fallback_keywords(topic)
    themes = [
        "implementation design",
        "delivery bottlenecks",
        "monitoring frameworks",
        "governance capacity",
    ]
    for keyword in keywords[:2]:
        themes.insert(0, f"{keyword} policy evidence")
    deduped: list[str] = []
    for theme in themes:
        if theme not in deduped:
            deduped.append(theme)
    return deduped[:6]


def _generic_indicator_label(bucket: str) -> str:
    bucket_title = bucket.title()
    if any(token in bucket.lower() for token in ("risk", "pressure", "exposure", "emission")):
        return f"{bucket_title} pressure signal"
    if "delivery" in bucket.lower():
        return f"{bucket_title} performance signal"
    if "finance" in bucket.lower():
        return f"{bucket_title} conditions signal"
    return f"{bucket_title} trend signal"


def _safe_indicator_id(seed: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Z0-9]+", "", seed.upper())
    return (cleaned[:10] or f"SIG{index}")[:10]


def _normalize_frequency(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"daily", "weekly", "monthly", "quarterly", "annual", "yearly"}:
        return "annual" if normalized == "yearly" else normalized
    return "quarterly"


def _default_frequency_for_dimension(bucket: str) -> str:
    lowered = bucket.lower()
    if any(token in lowered for token in ("exposure", "air", "aqi")):
        return "daily"
    if any(token in lowered for token in ("mobility", "enforcement", "emissions", "demand", "production")):
        return "monthly"
    return "quarterly"


def _default_unit_for_dimension(bucket: str) -> str:
    lowered = bucket.lower()
    if "finance" in lowered:
        return "% YoY"
    if any(token in lowered for token in ("delivery", "coverage", "compliance", "quality")):
        return "% share"
    return "index"


def _generate_observation_values(rng: np.random.Generator, unit: str, index: int) -> tuple[float, float]:
    if unit == "% YoY":
        prior = round(4.5 + index * 0.6 + float(rng.uniform(-1.0, 1.0)), 2)
        latest = round(prior + float(rng.uniform(-1.4, 2.4)), 2)
        return latest, prior
    if unit == "% share":
        prior = round(42.0 + index * 1.8 + float(rng.uniform(-4.0, 4.0)), 2)
        latest = round(prior + float(rng.uniform(-3.0, 5.0)), 2)
        return latest, prior
    if unit == "count":
        prior = round(120 + index * 45 + float(rng.uniform(-30.0, 40.0)), 2)
        latest = round(prior + float(rng.uniform(-20.0, 80.0)), 2)
        return latest, prior
    prior = round(48.0 + index * 2.2 + float(rng.uniform(-5.0, 5.0)), 2)
    latest = round(prior + float(rng.uniform(-4.5, 6.5)), 2)
    return latest, prior
