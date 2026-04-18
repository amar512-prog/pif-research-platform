from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"


class RunStatus(str, Enum):
    WAITING = "waiting_for_checkpoint"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CheckpointDecision(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"


class CheckpointId(str, Enum):
    CONFIG = "config"
    PLAN = "plan"
    DATA = "data"
    ANALYSIS = "analysis"
    REVIEW = "review"


class RunStageId(str, Enum):
    CONFIG = "config"
    PLANNER = "planner"
    LITERATURE_REVIEW = "literature_review"
    DATA_COLLECTION = "data_collection"
    FACT_CHECKER = "fact_checker"
    ANALYSIS = "analysis"
    WRITER = "writer"
    QA_SYNTHESIS = "qa_synthesis"
    CRITICAL_REVIEWER = "critical_reviewer"
    FINALIZE = "finalize"


class SectionPlanItem(BaseModel):
    title: str
    target_words: int
    purpose: str


class TopicProfile(BaseModel):
    domain: str
    analysis_mode: Literal["indicator_composite", "policy_synthesis"] = "indicator_composite"
    audience: str = "Joint Secretary / Programme Officer level"
    geography: str | None = None
    research_goal: str
    headline_metric_label: str
    headline_unit: str
    comparator_label: str
    dimension_label: str
    keywords: list[str]
    indicator_dimensions: list[str]
    source_themes: list[str] = Field(default_factory=list)


class IndicatorSpec(BaseModel):
    indicator_id: str
    label: str
    frequency: str
    unit: str
    sector_bucket: str
    rationale: str
    primary_source_name: str
    primary_source_url: str
    verification_source_name: str
    verification_source_url: str


class SourceRecord(BaseModel):
    source_id: str
    title: str
    authors: list[str]
    year: int
    source_type: str
    url: str
    alternate_url: str
    publisher: str
    summary: str
    methodology: str
    relevance: str
    apa_citation: str


class IndicatorObservation(BaseModel):
    indicator_id: str
    label: str
    sector_bucket: str
    reference_period: str
    latest_value: float
    unit: str
    prior_value: float
    delta: float
    primary_source_name: str
    primary_source_url: str
    verification_source_name: str
    verification_source_url: str
    notes: str = ""
    retrieval_mode: Literal["fixture", "live"] = "fixture"
    retrieval_status: Literal["fixture", "live", "fallback"] = "fixture"
    connector_id: str | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)


class VerificationItem(BaseModel):
    item_id: str
    item_type: Literal["citation", "indicator"]
    status: Literal["verified", "flagged"]
    primary_source: str
    secondary_source: str
    note: str


class AnalysisDataPoint(BaseModel):
    datapoint_id: str
    label: str
    value: float
    unit: str
    explanation: str


class CriterionScore(BaseModel):
    criterion_id: int
    criterion: str
    score: float
    explanation: str


class PlannerOutput(BaseModel):
    problem_statement: str
    analytical_method: str
    section_plan: list[SectionPlanItem]
    indicator_plan: list[IndicatorSpec]
    planner_notes: str


class LiteraturePack(BaseModel):
    introduction_summary: str
    literature_review_summary: str
    sources: list[SourceRecord]


class IndicatorPack(BaseModel):
    retrieval_mode: Literal["fixture", "live", "mixed"] = "fixture"
    indicators: list[IndicatorObservation]
    metadata: dict[str, Any]


class VerificationPack(BaseModel):
    verified_allowlist: list[str]
    verification_items: list[VerificationItem]
    flagged_items: list[str]
    correction_summary: str


class AnalysisPack(BaseModel):
    target_period: str
    model_name: str
    headline_metric_label: str
    headline_unit: str
    comparator_label: str
    dimension_label: str
    point_estimate: float
    confidence_low: float
    confidence_high: float
    last_official_estimate: float
    sector_attribution: dict[str, float]
    scenario_commentary: str
    method_notes: str
    data_points: list[AnalysisDataPoint]
    analysis_table: list[dict[str, Any]]


class ReportDraft(BaseModel):
    version_label: str
    path: str
    markdown: str
    word_count: int


class QAPack(BaseModel):
    qa_notes: list[str]
    corrected_word_count: int
    citation_integrity_passed: bool
    recommendation_traceability_passed: bool
    revised_report_path: str


class ReviewScorecard(BaseModel):
    cycle_number: int
    criterion_scores: list[CriterionScore]
    composite_score: float
    passed: bool
    priority_fixes: list[str]
    reviewer_summary: str


class CheckpointRecord(BaseModel):
    checkpoint_id: str
    label: str
    status: Literal["pending", "waiting", "approved", "revised"] = "pending"
    last_decision: CheckpointDecision | None = None
    last_feedback: str | None = None
    waiting_payload: dict[str, Any] | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class RunCreateRequest(BaseModel):
    topic: str
    output_format: OutputFormat = OutputFormat.MARKDOWN
    target_word_count: int = 1800
    section_preferences: list[str] | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_word_count(self) -> "RunCreateRequest":
        if self.target_word_count < 800:
            raise ValueError("target_word_count must be at least 800")
        return self


class CheckpointSubmission(BaseModel):
    checkpoint_id: CheckpointId
    decision: CheckpointDecision
    feedback: str | None = None
    config_patch: dict[str, Any] | None = None


class StageRestartRequest(BaseModel):
    stage_id: RunStageId


class RunSummary(BaseModel):
    run_id: str
    topic: str
    status: RunStatus
    current_node: str
    active_checkpoint: str | None
    checkpoint_status: dict[str, CheckpointRecord]
    artifact_paths: dict[str, str]
    latest_score: float | None = None
    review_cycles: int = 0
    errors: list[str] = Field(default_factory=list)


class SerializedRunState(BaseModel):
    run_id: str
    topic: str
    output_format: OutputFormat
    target_word_count: int
    section_preferences: list[str] | None = None
    notes: str | None = None
    topic_profile: TopicProfile | None = None
    status: RunStatus = RunStatus.WAITING
    current_node: str = "config_checkpoint"
    active_checkpoint: str | None = CheckpointId.CONFIG.value
    section_plan: list[SectionPlanItem] = Field(default_factory=list)
    indicator_plan: list[IndicatorSpec] = Field(default_factory=list)
    source_registry: list[SourceRecord] = Field(default_factory=list)
    verified_allowlist: list[str] = Field(default_factory=list)
    indicator_dataset: list[IndicatorObservation] = Field(default_factory=list)
    planner_output: PlannerOutput | None = None
    literature_pack: LiteraturePack | None = None
    indicator_pack: IndicatorPack | None = None
    verification_pack: VerificationPack | None = None
    analysis_pack: AnalysisPack | None = None
    report_versions: list[ReportDraft] = Field(default_factory=list)
    qa_pack: QAPack | None = None
    review_cycles: list[ReviewScorecard] = Field(default_factory=list)
    checkpoint_status: dict[str, CheckpointRecord] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    revision_notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def to_summary(self) -> RunSummary:
        latest_score = self.review_cycles[-1].composite_score if self.review_cycles else None
        return RunSummary(
            run_id=self.run_id,
            topic=self.topic,
            status=self.status,
            current_node=self.current_node,
            active_checkpoint=self.active_checkpoint,
            checkpoint_status=self.checkpoint_status,
            artifact_paths=self.artifact_paths,
            latest_score=latest_score,
            review_cycles=len(self.review_cycles),
            errors=list(self.errors),
        )


class ResearchRunState(TypedDict, total=False):
    run_id: str
    topic: str
    output_format: str
    target_word_count: int
    section_preferences: list[str] | None
    notes: str | None
    topic_profile: dict[str, Any] | None
    status: str
    current_node: str
    active_checkpoint: str | None
    section_plan: list[dict[str, Any]]
    indicator_plan: list[dict[str, Any]]
    source_registry: list[dict[str, Any]]
    verified_allowlist: list[str]
    indicator_dataset: list[dict[str, Any]]
    planner_output: dict[str, Any] | None
    literature_pack: dict[str, Any] | None
    indicator_pack: dict[str, Any] | None
    verification_pack: dict[str, Any] | None
    analysis_pack: dict[str, Any] | None
    report_versions: list[dict[str, Any]]
    qa_pack: dict[str, Any] | None
    review_cycles: list[dict[str, Any]]
    checkpoint_status: dict[str, dict[str, Any]]
    artifact_paths: dict[str, str]
    revision_notes: list[str]
    errors: list[str]
    created_at: str
    updated_at: str
