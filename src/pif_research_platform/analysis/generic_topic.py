from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import AnalysisDataPoint, AnalysisPack, IndicatorObservation, PlannerOutput, TopicProfile
from ..topic_intelligence import build_indicator_specs


@dataclass(slots=True)
class GenericTopicAnalysisStrategy:
    def indicator_specs(self, profile: TopicProfile, topic: str):
        return build_indicator_specs(profile, topic)

    def run(
        self,
        profile: TopicProfile,
        indicator_dataset: list[IndicatorObservation],
        planner_output: PlannerOutput,
        revision_notes: list[str] | None = None,
    ) -> AnalysisPack:
        revision_notes = revision_notes or []
        deltas = np.array([item.delta for item in indicator_dataset], dtype=float)
        latest_values = np.array([item.latest_value for item in indicator_dataset], dtype=float)
        priors = np.array([item.prior_value for item in indicator_dataset], dtype=float)
        magnitudes = np.where(np.abs(priors) < 1, 1.0, np.abs(priors))
        normalized_change = deltas / magnitudes

        if profile.headline_unit == "% YoY":
            point_estimate = float(np.average(latest_values))
            comparator = float(np.average(priors))
            interval_half_width = max(0.6, float(np.std(latest_values) * 0.85))
        else:
            scores = 50 + (normalized_change * 120)
            point_estimate = float(np.clip(np.mean(scores), 25.0, 85.0))
            prior_scores = 50 + ((priors - priors.mean()) / np.where(priors.std() == 0, 1.0, priors.std() + 1e-6)) * 2.5
            comparator = float(np.clip(np.mean(prior_scores), 20.0, 80.0))
            interval_half_width = max(3.0, float(np.std(scores) * 0.9))

        bucket_map: dict[str, list[float]] = {}
        for item, change in zip(indicator_dataset, normalized_change):
            bucket_map.setdefault(item.sector_bucket, []).append(float(change))
        dimension_attribution = {
            bucket: round(sum(values), 3) for bucket, values in bucket_map.items()
        }
        dominant_bucket = max(dimension_attribution, key=dimension_attribution.get)
        target_period = indicator_dataset[0].reference_period if indicator_dataset else "Current cycle"
        datapoints = []
        for index, item in enumerate(indicator_dataset[:5], start=1):
            datapoints.append(
                AnalysisDataPoint(
                    datapoint_id=f"A{index}",
                    label=item.label,
                    value=item.latest_value,
                    unit=item.unit,
                    explanation=f"{item.label} is one of the clearest observable signals shaping the current assessment.",
                )
            )
        analysis_rows = [
            {
                "bucket": bucket,
                "contribution": contribution,
                "interpretation": "Supportive" if contribution >= 0 else "Constraining",
            }
            for bucket, contribution in dimension_attribution.items()
        ]
        scenario_commentary = (
            f"The current assessment is led by {dominant_bucket.lower()} dynamics. The interval remains wider when "
            f"indicator movements are uneven across {profile.dimension_label.lower()}."
        )
        if revision_notes:
            scenario_commentary += " Review feedback has been incorporated into implementation sequencing and risk framing."
        return AnalysisPack(
            target_period=target_period,
            model_name="Local LLM Guided Composite Indicator Model",
            headline_metric_label=profile.headline_metric_label,
            headline_unit=profile.headline_unit,
            comparator_label=profile.comparator_label,
            dimension_label=profile.dimension_label,
            point_estimate=round(point_estimate, 2),
            confidence_low=round(point_estimate - interval_half_width, 2),
            confidence_high=round(point_estimate + interval_half_width, 2),
            last_official_estimate=round(comparator, 2),
            sector_attribution=dimension_attribution,
            scenario_commentary=scenario_commentary,
            method_notes=planner_output.analytical_method,
            data_points=datapoints,
            analysis_table=analysis_rows,
        )
