from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..models import AnalysisDataPoint, AnalysisPack, IndicatorObservation, PlannerOutput
from ..sample_data import build_historical_training_frame, build_indicator_specs, latest_feature_row


@dataclass(slots=True)
class MaharashtraNowcastStrategy:
    alpha: float = 1.2

    def indicator_specs(self):
        return build_indicator_specs()

    def run(
        self,
        indicator_dataset: list[IndicatorObservation],
        planner_output: PlannerOutput,
        revision_notes: list[str] | None = None,
    ) -> AnalysisPack:
        revision_notes = revision_notes or []
        historical = build_historical_training_frame()
        feature_columns = ["GST", "POWER", "VEHICLE", "EWAY", "CREDIT", "EXPORTS"]
        x_train = historical[feature_columns].to_numpy(dtype=float)
        y_train = historical["gsdp_growth"].to_numpy(dtype=float)

        means = x_train.mean(axis=0)
        stds = x_train.std(axis=0, ddof=0)
        stds = np.where(stds == 0, 1.0, stds)
        x_scaled = (x_train - means) / stds
        x_design = np.hstack([np.ones((x_scaled.shape[0], 1)), x_scaled])

        regularizer = np.eye(x_design.shape[1])
        regularizer[0, 0] = 0.0
        beta = np.linalg.solve(x_design.T @ x_design + self.alpha * regularizer, x_design.T @ y_train)

        latest_df = latest_feature_row(indicator_dataset)
        latest_scaled = (latest_df[feature_columns].to_numpy(dtype=float) - means) / stds
        latest_design = np.hstack([np.ones((1, 1)), latest_scaled])
        prediction = float((latest_design @ beta)[0])

        fitted = x_design @ beta
        residuals = y_train - fitted
        sigma = float(np.sqrt(np.sum(residuals**2) / max(len(y_train) - x_design.shape[1], 1)))
        xtx_inv = np.linalg.pinv(x_design.T @ x_design + self.alpha * regularizer)
        leverage = float((latest_design @ xtx_inv @ latest_design.T)[0, 0])
        interval_half_width = 1.64 * sigma * float(np.sqrt(1 + leverage))

        coeffs = dict(zip(feature_columns, beta[1:]))
        current_z = dict(zip(feature_columns, latest_scaled[0]))
        raw_contrib = {name: coeffs[name] * current_z[name] for name in feature_columns}
        bucket_map = {
            "Demand": ["GST", "VEHICLE"],
            "Production": ["POWER"],
            "Mobility": ["EWAY"],
            "Finance": ["CREDIT"],
            "Trade": ["EXPORTS"],
        }
        sector_attribution = {
            bucket: round(sum(raw_contrib[name] for name in members), 3)
            for bucket, members in bucket_map.items()
        }
        last_official = float(historical["gsdp_growth"].iloc[-1])
        lead_bucket = max(sector_attribution, key=sector_attribution.get)

        datapoints = [
            AnalysisDataPoint(
                datapoint_id="A1",
                label="Headline Maharashtra GSDP nowcast",
                value=round(prediction, 2),
                unit="% YoY",
                explanation="Model-implied nowcast for 2026-Q1 based on the approved indicator basket.",
            ),
            AnalysisDataPoint(
                datapoint_id="A2",
                label="GST growth signal",
                value=float(latest_df["GST"].iloc[0]),
                unit="% YoY",
                explanation="Formal demand and tax buoyancy remain supportive.",
            ),
            AnalysisDataPoint(
                datapoint_id="A3",
                label="Electricity demand growth",
                value=float(latest_df["POWER"].iloc[0]),
                unit="% YoY",
                explanation="Production-side momentum remains positive.",
            ),
            AnalysisDataPoint(
                datapoint_id="A4",
                label="Vehicle registration growth",
                value=float(latest_df["VEHICLE"].iloc[0]),
                unit="% YoY",
                explanation="Consumer demand and logistics replacement remain firm.",
            ),
            AnalysisDataPoint(
                datapoint_id="A5",
                label="Lead sector contribution",
                value=round(sector_attribution[lead_bucket], 2),
                unit="standardized contribution",
                explanation=f"{lead_bucket} is the largest positive contributor in the current quarter.",
            ),
        ]

        scenario_commentary = (
            f"The nowcast is led by the {lead_bucket.lower()} bucket, while the 90% interval widens modestly "
            "because the prototype uses a compact training history and interpretable shrinkage rather than a "
            "black-box ensemble."
        )
        if revision_notes:
            scenario_commentary += (
                " Review feedback was incorporated by adding an implementation-sequencing section and clearer "
                "agency ownership in the narrative output."
            )

        analysis_rows: list[dict[str, Any]] = []
        for bucket, contribution in sector_attribution.items():
            analysis_rows.append(
                {
                    "bucket": bucket,
                    "contribution": round(contribution, 3),
                    "interpretation": "Positive" if contribution >= 0 else "Negative",
                }
            )

        return AnalysisPack(
            target_period="2026-Q1",
            model_name="Interpretable Ridge Nowcast",
            point_estimate=round(prediction, 2),
            confidence_low=round(prediction - interval_half_width, 2),
            confidence_high=round(prediction + interval_half_width, 2),
            last_official_estimate=round(last_official, 2),
            sector_attribution=sector_attribution,
            scenario_commentary=scenario_commentary,
            method_notes=planner_output.analytical_method,
            data_points=datapoints,
            analysis_table=analysis_rows,
        )
