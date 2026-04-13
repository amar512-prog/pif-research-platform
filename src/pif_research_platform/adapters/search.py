from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from urllib.parse import quote, urlparse
import xml.etree.ElementTree as ET

import requests

from ..config import SearchSettings
from ..models import IndicatorObservation, IndicatorSpec, SourceRecord, TopicProfile
from ..topic_intelligence import (
    build_indicator_observations,
    build_literature_sources,
    domain_config,
)


@dataclass(frozen=True, slots=True)
class IndicatorSourceHint:
    primary_source_name: str
    primary_source_url: str
    verification_source_name: str
    verification_source_url: str
    primary_domains: tuple[str, ...] = ()
    verification_domains: tuple[str, ...] = ()
    query_terms: tuple[str, ...] = ()


INDICATOR_SOURCE_HINTS: dict[str, dict[str, IndicatorSourceHint]] = {
    "macroeconomy": {
        "TAX": IndicatorSourceHint(
            primary_source_name="GST revenue collections portal",
            primary_source_url="https://www.gst.gov.in/",
            verification_source_name="PIB revenue release archive",
            verification_source_url="https://pib.gov.in/",
            primary_domains=("gst.gov.in", "cbic-gst.gov.in", "pib.gov.in"),
            verification_domains=("pib.gov.in", "cbic-gst.gov.in", "gst.gov.in"),
            query_terms=("gst revenue collections", "tax collections"),
        ),
        "POWER": IndicatorSourceHint(
            primary_source_name="GRID-INDIA demand data",
            primary_source_url="https://www.grid-india.in/",
            verification_source_name="Central Electricity Authority",
            verification_source_url="https://cea.nic.in/",
            primary_domains=("grid-india.in", "posoco.in", "cea.nic.in"),
            verification_domains=("cea.nic.in", "grid-india.in"),
            query_terms=("electricity demand", "peak power demand"),
        ),
        "REG": IndicatorSourceHint(
            primary_source_name="VAHAN dashboard",
            primary_source_url="https://vahan.parivahan.gov.in/vahan4dashboard/",
            verification_source_name="Parivahan Sewa",
            verification_source_url="https://parivahan.gov.in/",
            primary_domains=("vahan.parivahan.gov.in", "parivahan.gov.in"),
            verification_domains=("parivahan.gov.in", "india.gov.in"),
            query_terms=("vehicle registrations", "vahan dashboard"),
        ),
        "LOGISTICS": IndicatorSourceHint(
            primary_source_name="GST e-Way Bill portal",
            primary_source_url="https://ewaybillgst.gov.in/",
            verification_source_name="National Informatics Centre GST e-Way Bill project",
            verification_source_url="https://www.nic.gov.in/project/gst-e-way-bill-system/",
            primary_domains=("ewaybillgst.gov.in", "ewaybill2.gst.gov.in", "nic.gov.in"),
            verification_domains=("nic.gov.in", "gst.gov.in", "ewaybillgst.gov.in"),
            query_terms=("e-way bill", "goods movement"),
        ),
        "CREDIT": IndicatorSourceHint(
            primary_source_name="RBI Database on Indian Economy",
            primary_source_url="https://dbie.rbi.org.in/",
            verification_source_name="RBI sectoral credit releases",
            verification_source_url="https://www.rbi.org.in/",
            primary_domains=("dbie.rbi.org.in", "rbi.org.in"),
            verification_domains=("rbi.org.in", "dbie.rbi.org.in"),
            query_terms=("sectoral deployment of bank credit", "credit growth"),
        ),
        "TRADE": IndicatorSourceHint(
            primary_source_name="NIRYAT trade portal",
            primary_source_url="https://www.niryat.gov.in/",
            verification_source_name="Directorate General of Foreign Trade",
            verification_source_url="https://www.dgft.gov.in/",
            primary_domains=("niryat.gov.in", "dgft.gov.in", "commerce.gov.in"),
            verification_domains=("dgft.gov.in", "commerce.gov.in", "niryat.gov.in"),
            query_terms=("exports", "external trade"),
        ),
    },
    "environment": {
        "AQI": IndicatorSourceHint(
            "CPCB AQI India portal",
            "https://airquality.cpcb.gov.in/AQI_India/",
            "Delhi Pollution Control Committee",
            "https://www.dpcc.delhigovt.nic.in/",
            ("airquality.cpcb.gov.in", "cpcb.nic.in", "dpcc.delhigovt.nic.in"),
            ("dpcc.delhigovt.nic.in", "cpcb.nic.in"),
            ("AQI India", "air quality index"),
        ),
        "PM25": IndicatorSourceHint(
            "CPCB AQI India portal",
            "https://airquality.cpcb.gov.in/AQI_India/",
            "SAFAR India",
            "https://safar.tropmet.res.in/",
            ("airquality.cpcb.gov.in", "cpcb.nic.in"),
            ("safar.tropmet.res.in", "cpcb.nic.in", "dpcc.delhigovt.nic.in"),
            ("PM2.5 concentration", "particulate matter"),
        ),
        "EMISSIONS": IndicatorSourceHint(
            "Central Pollution Control Board",
            "https://cpcb.nic.in/",
            "Ministry of Environment, Forest and Climate Change",
            "https://moef.gov.in/",
            ("cpcb.nic.in", "moef.gov.in"),
            ("moef.gov.in", "cpcb.nic.in"),
            ("industrial emissions", "emission inventory"),
        ),
        "TRAFFIC": IndicatorSourceHint(
            "VAHAN dashboard",
            "https://vahan.parivahan.gov.in/vahan4dashboard/",
            "Parivahan Sewa",
            "https://parivahan.gov.in/",
            ("vahan.parivahan.gov.in", "parivahan.gov.in"),
            ("parivahan.gov.in", "india.gov.in"),
            ("vehicle registrations", "transport emissions"),
        ),
        "COMPLIANCE": IndicatorSourceHint(
            "Delhi Pollution Control Committee",
            "https://www.dpcc.delhigovt.nic.in/",
            "Central Pollution Control Board",
            "https://cpcb.nic.in/",
            ("dpcc.delhigovt.nic.in", "cpcb.nic.in"),
            ("cpcb.nic.in", "dpcc.delhigovt.nic.in"),
            ("pollution control compliance", "inspection action taken"),
        ),
        "HEALTH": IndicatorSourceHint(
            "Health Management Information System",
            "https://hmis.mohfw.gov.in/#!/",
            "Ministry of Health and Family Welfare",
            "https://main.mohfw.gov.in/",
            ("hmis.mohfw.gov.in", "main.mohfw.gov.in", "mohfw.gov.in"),
            ("main.mohfw.gov.in", "mohfw.gov.in"),
            ("respiratory cases", "public health surveillance"),
        ),
    },
    "energy_transition": {
        "CAPACITY": IndicatorSourceHint(
            "Ministry of New and Renewable Energy",
            "https://mnre.gov.in/",
            "Ministry of Power",
            "https://powermin.gov.in/",
            ("mnre.gov.in", "powermin.gov.in", "cea.nic.in"),
            ("powermin.gov.in", "mnre.gov.in", "cea.nic.in"),
            ("renewable capacity", "installed capacity"),
        ),
        "UTILIZATION": IndicatorSourceHint(
            "Central Electricity Authority",
            "https://cea.nic.in/",
            "Ministry of Power",
            "https://powermin.gov.in/",
            ("cea.nic.in", "powermin.gov.in"),
            ("powermin.gov.in", "cea.nic.in"),
            ("plant load factor", "asset utilization"),
        ),
        "ACCESS": IndicatorSourceHint(
            "e-Amrit",
            "https://e-amrit.niti.gov.in/",
            "NITI Aayog",
            "https://www.niti.gov.in/",
            ("e-amrit.niti.gov.in", "niti.gov.in"),
            ("niti.gov.in", "powermin.gov.in"),
            ("charging stations", "infrastructure rollout"),
        ),
        "ADOPTION": IndicatorSourceHint(
            "VAHAN dashboard",
            "https://vahan.parivahan.gov.in/vahan4dashboard/",
            "Parivahan Sewa",
            "https://parivahan.gov.in/",
            ("vahan.parivahan.gov.in", "parivahan.gov.in"),
            ("parivahan.gov.in", "india.gov.in"),
            ("ev registrations", "adoption"),
        ),
        "FINANCE": IndicatorSourceHint(
            "Ministry of New and Renewable Energy",
            "https://mnre.gov.in/",
            "India services portal",
            "https://www.india.gov.in/",
            ("mnre.gov.in", "india.gov.in"),
            ("india.gov.in", "mnre.gov.in"),
            ("project finance", "clean energy funding"),
        ),
        "DELIVERY": IndicatorSourceHint(
            "Ministry of New and Renewable Energy",
            "https://mnre.gov.in/",
            "Ministry of Power",
            "https://powermin.gov.in/",
            ("mnre.gov.in", "powermin.gov.in"),
            ("powermin.gov.in", "mnre.gov.in"),
            ("implementation status", "project completion"),
        ),
    },
    "mobility": {
        "RIDERSHIP": IndicatorSourceHint(
            "Ministry of Housing and Urban Affairs",
            "https://mohua.gov.in/",
            "Ministry of Road Transport and Highways",
            "https://morth.nic.in/",
            ("mohua.gov.in", "morth.nic.in"),
            ("morth.nic.in", "mohua.gov.in"),
            ("ridership", "transit demand"),
        ),
        "CAPACITY": IndicatorSourceHint(
            "Ministry of Road Transport and Highways",
            "https://morth.nic.in/",
            "PM Gati Shakti",
            "https://gati.gov.in/",
            ("morth.nic.in", "gati.gov.in"),
            ("gati.gov.in", "morth.nic.in"),
            ("capacity additions", "transport infrastructure"),
        ),
        "RELIABILITY": IndicatorSourceHint(
            "Indian Railways",
            "https://indianrailways.gov.in/",
            "Ministry of Road Transport and Highways",
            "https://morth.nic.in/",
            ("indianrailways.gov.in", "morth.nic.in"),
            ("morth.nic.in", "indianrailways.gov.in"),
            ("service reliability", "on-time performance"),
        ),
        "LOGISTICS": IndicatorSourceHint(
            "PM Gati Shakti",
            "https://gati.gov.in/",
            "Ministry of Commerce and Industry",
            "https://www.commerce.gov.in/",
            ("gati.gov.in", "commerce.gov.in"),
            ("commerce.gov.in", "gati.gov.in"),
            ("freight throughput", "logistics"),
        ),
        "FINANCE": IndicatorSourceHint(
            "Ministry of Road Transport and Highways",
            "https://morth.nic.in/",
            "NITI Aayog",
            "https://www.niti.gov.in/",
            ("morth.nic.in", "niti.gov.in"),
            ("niti.gov.in", "morth.nic.in"),
            ("transport financing", "investment approvals"),
        ),
    },
    "social_policy": {
        "ACCESS": IndicatorSourceHint(
            "Open Government Data Platform India",
            "https://www.data.gov.in/",
            "National Portal of India",
            "https://www.india.gov.in/",
            ("data.gov.in", "india.gov.in"),
            ("india.gov.in", "data.gov.in"),
            ("service access", "coverage"),
        ),
        "QUALITY": IndicatorSourceHint(
            "Open Government Data Platform India",
            "https://www.data.gov.in/",
            "National Portal of India",
            "https://www.india.gov.in/",
            ("data.gov.in", "india.gov.in"),
            ("india.gov.in", "data.gov.in"),
            ("quality outcomes", "performance"),
        ),
        "CAPACITY": IndicatorSourceHint(
            "National Portal of India",
            "https://www.india.gov.in/",
            "Open Government Data Platform India",
            "https://www.data.gov.in/",
            ("india.gov.in", "data.gov.in"),
            ("data.gov.in", "india.gov.in"),
            ("workforce capacity", "staffing"),
        ),
        "FINANCE": IndicatorSourceHint(
            "Union Budget",
            "https://www.indiabudget.gov.in/",
            "Open Government Data Platform India",
            "https://www.data.gov.in/",
            ("indiabudget.gov.in", "data.gov.in"),
            ("data.gov.in", "indiabudget.gov.in"),
            ("budget release", "programme finance"),
        ),
        "DELIVERY": IndicatorSourceHint(
            "Open Government Data Platform India",
            "https://www.data.gov.in/",
            "National Portal of India",
            "https://www.india.gov.in/",
            ("data.gov.in", "india.gov.in"),
            ("india.gov.in", "data.gov.in"),
            ("implementation status", "delivery"),
        ),
    },
    "generic_policy": {
        "SIGNAL1": IndicatorSourceHint(
            "Open Government Data Platform India",
            "https://www.data.gov.in/",
            "National Portal of India",
            "https://www.india.gov.in/",
            ("data.gov.in", "india.gov.in"),
            ("india.gov.in", "data.gov.in"),
            ("leading indicator", "policy data"),
        ),
        "SIGNAL2": IndicatorSourceHint(
            "Open Government Data Platform India",
            "https://www.data.gov.in/",
            "National Portal of India",
            "https://www.india.gov.in/",
            ("data.gov.in", "india.gov.in"),
            ("india.gov.in", "data.gov.in"),
            ("capacity data", "policy data"),
        ),
        "SIGNAL3": IndicatorSourceHint(
            "Open Government Data Platform India",
            "https://www.data.gov.in/",
            "National Portal of India",
            "https://www.india.gov.in/",
            ("data.gov.in", "india.gov.in"),
            ("india.gov.in", "data.gov.in"),
            ("delivery data", "implementation data"),
        ),
        "SIGNAL4": IndicatorSourceHint(
            "Open Government Data Platform India",
            "https://www.data.gov.in/",
            "National Portal of India",
            "https://www.india.gov.in/",
            ("data.gov.in", "india.gov.in"),
            ("india.gov.in", "data.gov.in"),
            ("financing data", "budget data"),
        ),
        "SIGNAL5": IndicatorSourceHint(
            "Open Government Data Platform India",
            "https://www.data.gov.in/",
            "National Portal of India",
            "https://www.india.gov.in/",
            ("data.gov.in", "india.gov.in"),
            ("india.gov.in", "data.gov.in"),
            ("risk data", "policy monitoring"),
        ),
    },
}


class BaseSearchAdapter:
    def search_literature(self, topic: str, topic_profile: TopicProfile, max_results: int = 8) -> list[SourceRecord]:
        raise NotImplementedError

    def resolve_indicator_plan(
        self,
        topic: str,
        topic_profile: TopicProfile,
        indicator_plan: list[IndicatorSpec],
    ) -> list[IndicatorSpec]:
        raise NotImplementedError

    def collect_indicator_values(
        self,
        topic: str,
        topic_profile: TopicProfile,
        indicator_plan: list[IndicatorSpec],
    ) -> list[IndicatorObservation]:
        raise NotImplementedError


class OfflineSearchAdapter(BaseSearchAdapter):
    """Deterministic fixtures so the prototype runs without external credentials."""

    def search_literature(self, topic: str, topic_profile: TopicProfile, max_results: int = 8) -> list[SourceRecord]:
        return build_literature_sources(topic_profile, topic, max_results=max_results)

    def resolve_indicator_plan(
        self,
        topic: str,
        topic_profile: TopicProfile,
        indicator_plan: list[IndicatorSpec],
    ) -> list[IndicatorSpec]:
        return [_apply_indicator_source_hint(topic_profile, spec) for spec in indicator_plan]

    def collect_indicator_values(
        self,
        topic: str,
        topic_profile: TopicProfile,
        indicator_plan: list[IndicatorSpec],
    ) -> list[IndicatorObservation]:
        observed = {
            item.indicator_id: item
            for item in build_indicator_observations(topic_profile, topic, indicator_plan)
        }
        return [observed[spec.indicator_id] for spec in indicator_plan if spec.indicator_id in observed]


@dataclass(slots=True)
class CrossrefSearchAdapter(BaseSearchAdapter):
    settings: SearchSettings
    fallback: BaseSearchAdapter | None = None

    def search_literature(self, topic: str, topic_profile: TopicProfile, max_results: int = 8) -> list[SourceRecord]:
        collected: list[SourceRecord] = []
        seen_keys: set[str] = set()
        rows_per_query = max(max_results * 3, 10)
        queries = self._candidate_queries(topic, topic_profile)
        for query in queries:
            if len(collected) >= max_results:
                break
            items = self._search_crossref(query, rows_per_query)
            for item in items:
                parsed = self._parse_crossref_item(item, topic, topic_profile)
                if parsed is None:
                    continue
                dedupe_key = parsed.alternate_url or parsed.url or parsed.title.lower()
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                parsed.source_id = f"S{len(collected) + 1}"
                collected.append(parsed)
                if len(collected) >= max_results:
                    break
        if collected:
            return collected[:max_results]
        if self.fallback is not None:
            return self.fallback.search_literature(topic, topic_profile, max_results=max_results)
        return []

    def resolve_indicator_plan(
        self,
        topic: str,
        topic_profile: TopicProfile,
        indicator_plan: list[IndicatorSpec],
    ) -> list[IndicatorSpec]:
        resolved: list[IndicatorSpec] = []
        for spec in indicator_plan:
            with_official_fallback = _apply_indicator_source_hint(topic_profile, spec)
            hint = _indicator_source_hint(topic_profile, with_official_fallback)
            primary_url = self._resolve_indicator_link(
                topic,
                topic_profile,
                with_official_fallback,
                domains=hint.primary_domains,
                query_terms=hint.query_terms,
            ) or hint.primary_source_url
            resolved.append(
                with_official_fallback.model_copy(
                    update={
                        "primary_source_url": primary_url,
                        "verification_source_url": hint.verification_source_url,
                    }
                )
            )
        return resolved

    def collect_indicator_values(
        self,
        topic: str,
        topic_profile: TopicProfile,
        indicator_plan: list[IndicatorSpec],
    ) -> list[IndicatorObservation]:
        if self.fallback is not None:
            return self.fallback.collect_indicator_values(topic, topic_profile, indicator_plan)
        observed = {
            item.indicator_id: item
            for item in build_indicator_observations(topic_profile, topic, indicator_plan)
        }
        return [observed[spec.indicator_id] for spec in indicator_plan if spec.indicator_id in observed]

    def _candidate_queries(self, topic: str, topic_profile: TopicProfile) -> list[str]:
        geography = topic_profile.geography or ""
        config = domain_config(topic_profile.domain)
        source_themes = tuple(topic_profile.source_themes or list(config.source_themes))
        normalized_topic = topic.replace("GDSP", "GSDP").replace("gdsp", "gsdp")
        queries = [
            normalized_topic,
            f"{normalized_topic} policy",
            f"{normalized_topic} evidence",
        ]
        if geography:
            queries.extend(
                [
                    f"{geography} {topic_profile.headline_metric_label}",
                    f"{geography} policy analysis {topic_profile.domain.replace('_', ' ')}",
                    f"{geography} {' '.join(topic_profile.keywords[:4])}",
                ]
            )
        if "gsdp" in normalized_topic.lower():
            geography_term = geography or "state"
            queries.extend(
                [
                    f"{geography_term} gross state domestic product",
                    f"{geography_term} economic survey gross state domestic product",
                    f"{geography_term} macroeconomic performance",
                ]
            )
        if "gdp" in normalized_topic.lower() and "gsdp" not in normalized_topic.lower():
            geography_term = geography or "economy"
            queries.extend(
                [
                    f"{geography_term} gross domestic product",
                    f"{geography_term} economic survey",
                ]
            )
        queries.extend(f"{normalized_topic} {theme}" for theme in source_themes[:4])
        return _dedupe_strings(queries)

    def _search_crossref(self, query: str, rows: int) -> list[dict]:
        headers = {"User-Agent": self.settings.user_agent}
        params = {
            "query.bibliographic": query,
            "rows": rows,
            "mailto": self.settings.mailto,
            "select": "DOI,URL,title,author,publisher,type,issued,published-print,published-online,container-title",
        }
        try:
            response = requests.get(
                f"{self.settings.crossref_base_url.rstrip('/')}/works",
                params=params,
                headers=headers,
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            message = payload.get("message", {})
            items = message.get("items", [])
            return items if isinstance(items, list) else []
        except Exception:
            return []

    def _parse_crossref_item(
        self,
        item: dict,
        topic: str,
        topic_profile: TopicProfile,
    ) -> SourceRecord | None:
        title_list = item.get("title") or []
        title = title_list[0].strip() if title_list else ""
        if not title:
            return None
        doi = (item.get("DOI") or "").strip()
        landing_url = self._normalize_landing_url(item, doi)
        if not landing_url:
            return None
        api_url = (
            f"{self.settings.crossref_base_url.rstrip('/')}/works/{quote(doi, safe='')}"
            if doi
            else landing_url
        )
        authors = self._authors(item.get("author") or [])
        year = self._year_from_item(item)
        publisher = (item.get("publisher") or "Unknown publisher").strip()
        source_type = self._humanize_type(item.get("type"))
        summary = (
            f"Crossref metadata match retrieved for {topic}. Review the full text to confirm direct analytical fit."
        )
        methodology = (
            "Metadata retrieved via the Crossref REST API. Methodological claims should be validated against the full text."
        )
        relevance = (
            f"Potentially relevant to {topic_profile.research_goal.lower()} because it surfaced for the current topic query."
        )
        return SourceRecord(
            source_id=f"S{sha1((doi or title).encode('utf-8')).hexdigest()[:6]}",
            title=title,
            authors=authors,
            year=year,
            source_type=source_type,
            url=landing_url,
            alternate_url=api_url,
            publisher=publisher,
            summary=summary,
            methodology=methodology,
            relevance=relevance,
            apa_citation=self._apa_citation(authors, year, title, publisher, landing_url),
        )

    def _normalize_landing_url(self, item: dict, doi: str) -> str | None:
        resource = item.get("resource") or {}
        primary = resource.get("primary") or {}
        candidate = (primary.get("URL") or item.get("URL") or "").strip()
        if candidate:
            return (
                candidate.replace("http://dx.doi.org/", "https://doi.org/")
                .replace("http://doi.org/", "https://doi.org/")
            )
        if doi:
            return f"https://doi.org/{doi}"
        return None

    def _authors(self, authors: list[dict]) -> list[str]:
        names: list[str] = []
        for author in authors[:6]:
            given = (author.get("given") or "").strip()
            family = (author.get("family") or "").strip()
            literal = (author.get("name") or "").strip()
            name = " ".join(part for part in [given, family] if part).strip() or literal
            if name:
                names.append(name)
        return names or ["Unknown author"]

    def _year_from_item(self, item: dict) -> int:
        for field in ["issued", "published-print", "published-online"]:
            value = item.get(field) or {}
            date_parts = value.get("date-parts") or []
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
                if isinstance(year, int):
                    return year
        return 2024

    def _humanize_type(self, raw_type: str | None) -> str:
        if not raw_type:
            return "Scholarly Source"
        return raw_type.replace("-", " ").title()

    def _apa_citation(
        self,
        authors: list[str],
        year: int,
        title: str,
        publisher: str,
        landing_url: str,
    ) -> str:
        return f"{', '.join(authors)} ({year}). {title}. {publisher}. {landing_url}"

    def _resolve_indicator_link(
        self,
        topic: str,
        topic_profile: TopicProfile,
        spec: IndicatorSpec,
        *,
        domains: tuple[str, ...],
        query_terms: tuple[str, ...],
    ) -> str | None:
        if self.settings.web_search_provider.lower() == "none":
            return None
        queries = self._indicator_queries(topic, topic_profile, spec, domains, query_terms)
        for query in queries[: self.settings.max_web_queries_per_indicator]:
            for url in self._search_web_urls(query):
                if _url_matches_domains(url, domains):
                    return url
        return None

    def _indicator_queries(
        self,
        topic: str,
        topic_profile: TopicProfile,
        spec: IndicatorSpec,
        domains: tuple[str, ...],
        query_terms: tuple[str, ...],
    ) -> list[str]:
        geography = topic_profile.geography or ""
        base_terms = list(query_terms) or [spec.label]
        queries: list[str] = []
        for term in base_terms[:3]:
            if geography:
                queries.append(f"{geography} {term}")
                queries.append(f"{geography} {term} official data")
            queries.append(f"{topic} {term}")
            queries.append(f"{term} official data")
            for domain in domains[:3]:
                if geography:
                    queries.append(f"{geography} {term} site:{domain}")
                queries.append(f"{term} site:{domain}")
        return _dedupe_strings(queries)

    def _search_web_urls(self, query: str) -> list[str]:
        if self.settings.web_search_provider.lower() != "bing_rss":
            return []
        try:
            response = requests.get(
                self.settings.web_search_base_url,
                params={
                    "q": query,
                    "format": "rss",
                    "setlang": self.settings.web_search_market,
                    "cc": self.settings.web_search_market.split("-")[-1].lower(),
                },
                headers={"User-Agent": self.settings.user_agent},
                timeout=self.settings.web_search_timeout_seconds,
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
            urls = [item.findtext("link", default="").strip() for item in root.findall("./channel/item")]
            return [url for url in urls if url]
        except Exception:
            return []


def _apply_indicator_source_hint(topic_profile: TopicProfile, spec: IndicatorSpec) -> IndicatorSpec:
    hint = _indicator_source_hint(topic_profile, spec)
    return spec.model_copy(
        update={
            "primary_source_name": hint.primary_source_name,
            "primary_source_url": hint.primary_source_url,
            "verification_source_name": hint.verification_source_name,
            "verification_source_url": hint.verification_source_url,
        }
    )


def _indicator_source_hint(topic_profile: TopicProfile, spec: IndicatorSpec) -> IndicatorSourceHint:
    domain_hints = INDICATOR_SOURCE_HINTS.get(topic_profile.domain, {})
    hint = domain_hints.get(spec.indicator_id)
    if hint:
        return hint
    return IndicatorSourceHint(
        primary_source_name=f"{spec.label} data portal",
        primary_source_url="https://www.data.gov.in/",
        verification_source_name="National Portal of India",
        verification_source_url="https://www.india.gov.in/",
        primary_domains=(),
        verification_domains=(),
        query_terms=(spec.label.lower(),),
    )


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(value.split())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def _normalize_host(value: str) -> str:
    return value.lower().removeprefix("www.")


def _url_matches_domains(url: str, allowed_domains: tuple[str, ...]) -> bool:
    if not allowed_domains:
        return _looks_official_url(url)
    host = _normalize_host(urlparse(url).netloc)
    for domain in allowed_domains:
        normalized = _normalize_host(domain)
        if host == normalized or host.endswith(f".{normalized}"):
            return True
    return False


def _looks_official_url(url: str) -> bool:
    host = _normalize_host(urlparse(url).netloc)
    return any(
        token in host
        for token in (
            ".gov",
            ".gov.in",
            ".nic.in",
            "rbi.org.in",
            "data.gov.in",
            "india.gov.in",
            "niti.gov.in",
            "worldbank.org",
            "imf.org",
            "who.int",
        )
    )


def build_search_adapter(settings: SearchSettings) -> BaseSearchAdapter:
    offline = OfflineSearchAdapter()
    provider = settings.provider.lower()
    if provider == "offline":
        return offline
    if provider in {"crossref", "live", "hybrid"}:
        return CrossrefSearchAdapter(
            settings=settings,
            fallback=offline if settings.fallback_provider.lower() == "offline" or provider == "hybrid" else None,
        )
    return offline
