from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .models import IndicatorObservation, IndicatorSpec, SourceRecord


@dataclass(frozen=True, slots=True)
class IndicatorFixture:
    indicator_id: str
    label: str
    frequency: str
    unit: str
    sector_bucket: str
    rationale: str
    latest_value: float
    prior_value: float
    reference_period: str
    primary_source_name: str
    primary_source_url: str
    verification_source_name: str
    verification_source_url: str


def build_indicator_fixtures() -> list[IndicatorFixture]:
    return [
        IndicatorFixture(
            indicator_id="GST",
            label="GST collections growth",
            frequency="monthly",
            unit="% YoY",
            sector_bucket="Demand",
            rationale="Tracks formal economy momentum and tax buoyancy.",
            latest_value=11.8,
            prior_value=10.6,
            reference_period="2026-Q1",
            primary_source_name="GSTN dashboard",
            primary_source_url="https://www.gst.gov.in/",
            verification_source_name="Ministry of Finance monthly review",
            verification_source_url="https://dea.gov.in/monthly-economic-review-table",
        ),
        IndicatorFixture(
            indicator_id="POWER",
            label="Electricity demand growth",
            frequency="monthly",
            unit="% YoY",
            sector_bucket="Production",
            rationale="Proxy for manufacturing throughput and services activity.",
            latest_value=6.9,
            prior_value=5.8,
            reference_period="2026-Q1",
            primary_source_name="National Power Portal",
            primary_source_url="https://npp.gov.in/",
            verification_source_name="CEA monthly report",
            verification_source_url="https://cea.nic.in/monthly-generation-report/",
        ),
        IndicatorFixture(
            indicator_id="VEHICLE",
            label="Vehicle registrations growth",
            frequency="monthly",
            unit="% YoY",
            sector_bucket="Demand",
            rationale="Captures consumption and logistics fleet renewal.",
            latest_value=9.7,
            prior_value=8.9,
            reference_period="2026-Q1",
            primary_source_name="VAHAN dashboard",
            primary_source_url="https://vahan.parivahan.gov.in/",
            verification_source_name="State transport bulletin",
            verification_source_url="https://transport.maharashtra.gov.in/",
        ),
        IndicatorFixture(
            indicator_id="EWAY",
            label="E-way bill growth",
            frequency="monthly",
            unit="% YoY",
            sector_bucket="Mobility",
            rationale="Tracks goods movement across the formal supply chain.",
            latest_value=10.5,
            prior_value=9.3,
            reference_period="2026-Q1",
            primary_source_name="E-way bill system",
            primary_source_url="https://ewaybillgst.gov.in/",
            verification_source_name="GST Council dashboard",
            verification_source_url="https://www.gstcouncil.gov.in/",
        ),
        IndicatorFixture(
            indicator_id="CREDIT",
            label="Bank credit growth",
            frequency="monthly",
            unit="% YoY",
            sector_bucket="Finance",
            rationale="Signals financing conditions for households and firms.",
            latest_value=11.1,
            prior_value=10.4,
            reference_period="2026-Q1",
            primary_source_name="RBI sectoral credit data",
            primary_source_url="https://rbi.org.in/",
            verification_source_name="RBI DBIE",
            verification_source_url="https://dbie.rbi.org.in/",
        ),
        IndicatorFixture(
            indicator_id="EXPORTS",
            label="Merchandise exports growth",
            frequency="monthly",
            unit="% YoY",
            sector_bucket="Trade",
            rationale="Proxy for external demand and industrial competitiveness.",
            latest_value=4.2,
            prior_value=3.6,
            reference_period="2026-Q1",
            primary_source_name="DGFT trade statistics",
            primary_source_url="https://www.dgft.gov.in/",
            verification_source_name="Commerce ministry dashboard",
            verification_source_url="https://commerce.gov.in/trade-statistics/",
        ),
    ]


def build_indicator_specs() -> list[IndicatorSpec]:
    return [
        IndicatorSpec(
            indicator_id=item.indicator_id,
            label=item.label,
            frequency=item.frequency,
            unit=item.unit,
            sector_bucket=item.sector_bucket,
            rationale=item.rationale,
            primary_source_name=item.primary_source_name,
            primary_source_url=item.primary_source_url,
            verification_source_name=item.verification_source_name,
            verification_source_url=item.verification_source_url,
        )
        for item in build_indicator_fixtures()
    ]


def build_indicator_observations() -> list[IndicatorObservation]:
    observations = []
    for item in build_indicator_fixtures():
        observations.append(
            IndicatorObservation(
                indicator_id=item.indicator_id,
                label=item.label,
                sector_bucket=item.sector_bucket,
                reference_period=item.reference_period,
                latest_value=item.latest_value,
                unit=item.unit,
                prior_value=item.prior_value,
                delta=round(item.latest_value - item.prior_value, 2),
                primary_source_name=item.primary_source_name,
                primary_source_url=item.primary_source_url,
                verification_source_name=item.verification_source_name,
                verification_source_url=item.verification_source_url,
                notes="Offline fixture mode; replace with live connector values for production use.",
            )
        )
    return observations


def build_literature_sources() -> list[SourceRecord]:
    raw = [
        (
            "S1",
            "Nowcasting Regional Output with High-Frequency Tax Signals",
            ["Ananya Deshmukh", "Rahul Menon"],
            2023,
            "Working Paper",
            "https://example.org/nowcasting-tax-signals",
            "https://example.org/alternate/tax-signals",
            "Policy Analytics Lab",
            "Tax receipts lead state-level growth by one quarter when formalisation is rising.",
            "Panel nowcasting with mixed-frequency regressors.",
            "Supports using GST data as a demand-side signal for Maharashtra.",
        ),
        (
            "S2",
            "Electricity Demand as a Real-Time Proxy for State Industrial Activity",
            ["Madhuri Rao"],
            2022,
            "Journal Article",
            "https://example.org/electricity-output",
            "https://example.org/alternate/electricity-output",
            "Journal of Applied Economic Measurement",
            "Electricity demand improves short-horizon industrial output forecasts.",
            "Distributed lag regression with seasonal adjustment.",
            "Justifies including power demand in the indicator basket.",
        ),
        (
            "S3",
            "Formal Logistics Indicators and Goods Movement in India",
            ["Karan Bhatia", "Nidhi Suri"],
            2024,
            "Policy Report",
            "https://example.org/logistics-indicators",
            "https://example.org/alternate/logistics-indicators",
            "Centre for Industrial Logistics",
            "E-way bills co-move with manufacturing dispatches and wholesale trade.",
            "Descriptive analysis with sector mapping.",
            "Supports a logistics-sensitive mobility bucket.",
        ),
        (
            "S4",
            "Credit Conditions and State-Level Investment Cycles",
            ["S. Iyer"],
            2021,
            "Working Paper",
            "https://example.org/credit-investment",
            "https://example.org/alternate/credit-investment",
            "Monetary Policy Institute",
            "Sectoral credit growth helps anticipate capex acceleration and services recovery.",
            "State-space investment indicator model.",
            "Provides justification for credit conditions in the nowcast.",
        ),
        (
            "S5",
            "Vehicle Registrations as a Composite Demand Indicator",
            ["Ritika Sharma", "Pranav Joshi"],
            2022,
            "Journal Article",
            "https://example.org/vehicle-demand",
            "https://example.org/alternate/vehicle-demand",
            "Transportation and Macroeconomy Review",
            "Passenger and commercial registrations carry separate consumer and logistics information.",
            "Composite indicator decomposition.",
            "Helps interpret vehicle registrations as both demand and supply-chain signals.",
        ),
        (
            "S6",
            "Mixed-Frequency State Domestic Product Nowcasting in India",
            ["Amitabh Sen", "Neha Borkar"],
            2024,
            "RBI Discussion Paper",
            "https://example.org/mixed-frequency-nowcasting",
            "https://example.org/alternate/mixed-frequency-nowcasting",
            "Reserve Bank Research Series",
            "Interpretable shrinkage models perform well when indicator count is modest and domain informed.",
            "Ridge-style shrinkage with macro feature engineering.",
            "Directly informs the modelling choice for the prototype.",
        ),
        (
            "S7",
            "Export Competitiveness and Maharashtra Manufacturing",
            ["Vivek Patil"],
            2023,
            "Policy Brief",
            "https://example.org/maharashtra-exports",
            "https://example.org/alternate/maharashtra-exports",
            "Trade Competitiveness Forum",
            "External demand matters most for transport equipment, chemicals, and refined products.",
            "Sectoral export mapping.",
            "Useful for explaining trade-bucket attribution.",
        ),
        (
            "S8",
            "From Monitoring to Action: Writing Policy Notes for Senior Government Readers",
            ["Leena Kapoor"],
            2020,
            "Handbook",
            "https://example.org/government-policy-writing",
            "https://example.org/alternate/government-policy-writing",
            "Public Policy Writing Lab",
            "Senior officials prefer short lead messages, decision framing, and implementer-specific recommendations.",
            "Practice-oriented writing guide.",
            "Informs the final report style and recommendation framing.",
        ),
    ]
    results = []
    for source_id, title, authors, year, source_type, url, alternate_url, publisher, summary, methodology, relevance in raw:
        author_text = ", ".join(authors)
        apa = f"{author_text} ({year}). {title}. {publisher}. {url}"
        results.append(
            SourceRecord(
                source_id=source_id,
                title=title,
                authors=list(authors),
                year=year,
                source_type=source_type,
                url=url,
                alternate_url=alternate_url,
                publisher=publisher,
                summary=summary,
                methodology=methodology,
                relevance=relevance,
                apa_citation=apa,
            )
        )
    return results


def build_historical_training_frame() -> pd.DataFrame:
    periods = pd.period_range("2019Q1", "2025Q4", freq="Q")
    rows: list[dict[str, float | str]] = []
    for idx, period in enumerate(periods):
        seasonal = math.sin(idx / 2.3)
        trend = 5.4 + (idx * 0.04)
        gst = 8.5 + 0.65 * seasonal + idx * 0.08
        power = 4.8 + 0.55 * math.cos(idx / 3.1) + idx * 0.03
        vehicle = 6.0 + 0.7 * math.sin(idx / 2.0 + 0.5) + idx * 0.04
        eway = 7.4 + 0.75 * math.cos(idx / 2.8 + 0.2) + idx * 0.07
        credit = 9.2 + 0.35 * math.sin(idx / 1.9) + idx * 0.05
        exports = 2.6 + 0.5 * math.cos(idx / 2.5 + 1.0) + idx * 0.03
        gsdp = (
            trend
            + 0.18 * gst
            + 0.14 * power
            + 0.17 * vehicle
            + 0.16 * eway
            + 0.11 * credit
            + 0.08 * exports
            - 4.9
        )
        rows.append(
            {
                "quarter": str(period),
                "GST": round(gst, 3),
                "POWER": round(power, 3),
                "VEHICLE": round(vehicle, 3),
                "EWAY": round(eway, 3),
                "CREDIT": round(credit, 3),
                "EXPORTS": round(exports, 3),
                "gsdp_growth": round(gsdp, 3),
            }
        )
    return pd.DataFrame(rows)


def latest_feature_row(indicators: Iterable[IndicatorObservation]) -> pd.DataFrame:
    data = {item.indicator_id: item.latest_value for item in indicators}
    return pd.DataFrame([data])

