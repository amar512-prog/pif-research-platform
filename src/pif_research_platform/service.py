from __future__ import annotations

import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .adapters.local_llm import LocalLLMAdapter
from .adapters.search import build_search_adapter
from .agents import AgentSuite
from .analysis.generic_topic import GenericTopicAnalysisStrategy
from .config import AppSettings
from .exporters import SimplePDFExporter
from .models import (
    CheckpointDecision,
    CheckpointId,
    CheckpointRecord,
    CheckpointSubmission,
    OutputFormat,
    RunCreateRequest,
    RunStageId,
    RunStatus,
    SerializedRunState,
)
from .storage import RunRepository
from .workflow import build_workflow


class RunService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.repo = RunRepository(settings)
        self.agent_suite = AgentSuite(
            settings=settings,
            repo=self.repo,
            search_adapter=build_search_adapter(settings.search),
            local_llm=LocalLLMAdapter(settings.local_llm),
            strategy=GenericTopicAnalysisStrategy(),
            pdf_exporter=SimplePDFExporter(),
        )
        self.graph = build_workflow(self.agent_suite).compile(checkpointer=InMemorySaver())

    def create_run(self, request: RunCreateRequest):
        run_id = uuid.uuid4().hex[:12]
        state = SerializedRunState(
            run_id=run_id,
            topic=request.topic,
            output_format=request.output_format,
            target_word_count=request.target_word_count,
            section_preferences=request.section_preferences,
            notes=request.notes,
            status=RunStatus.WAITING,
            current_node="config_checkpoint",
            active_checkpoint=CheckpointId.CONFIG.value,
            checkpoint_status={
                CheckpointId.CONFIG.value: CheckpointRecord(
                    checkpoint_id=CheckpointId.CONFIG.value,
                    label="Config Intake",
                    status="waiting",
                )
            },
        )
        self.repo.save_state(state)
        final_state = self._drive_graph(run_id, initial_state=state)
        return final_state.to_summary()

    def get_run(self, run_id: str):
        return self.repo.load_state(run_id).to_summary()

    def get_run_detail(self, run_id: str) -> SerializedRunState:
        return self.repo.load_state(run_id)

    def submit_checkpoint(self, run_id: str, submission: CheckpointSubmission):
        state = self.repo.load_state(run_id)
        if state.active_checkpoint != submission.checkpoint_id.value:
            raise ValueError(
                f"Run '{run_id}' is waiting on checkpoint '{state.active_checkpoint}', not '{submission.checkpoint_id.value}'"
            )
        final_state = self._resume_without_graph(state, submission)
        return final_state.to_summary()

    def restart_from_stage(self, run_id: str, stage_id: RunStageId):
        run = self.repo.load_state(run_id)
        truncated = self._truncate_after_stage(run, stage_id)
        restarted = self._execute_from_stage(truncated, stage_id)
        return restarted.to_summary()

    def _drive_graph(
        self,
        run_id: str,
        *,
        initial_state: SerializedRunState | None = None,
        resume_payload: dict | None = None,
    ) -> SerializedRunState:
        config = {"configurable": {"thread_id": run_id}}
        graph_input = initial_state.model_dump(mode="json") if initial_state else Command(resume=resume_payload)
        interrupt_payload = None
        try:
            for chunk in self.graph.stream(graph_input, config):
                if "__interrupt__" in chunk:
                    interrupt_payload = chunk["__interrupt__"][0].value
            snapshot = self.graph.get_state(config)
            merged = self.repo.load_state(run_id)
            if snapshot.values:
                merged = SerializedRunState.model_validate({**merged.model_dump(mode="json"), **snapshot.values})
            if interrupt_payload:
                checkpoint_id = interrupt_payload["checkpoint_id"]
                record = merged.checkpoint_status.get(checkpoint_id) or CheckpointRecord(
                    checkpoint_id=checkpoint_id,
                    label=interrupt_payload.get("label", checkpoint_id.title()),
                )
                record.status = "waiting"
                record.waiting_payload = interrupt_payload
                merged.checkpoint_status[checkpoint_id] = record
                merged.status = RunStatus.WAITING
                merged.active_checkpoint = checkpoint_id
                merged.current_node = f"{checkpoint_id}_checkpoint"
            else:
                merged.active_checkpoint = None
                merged.status = RunStatus.COMPLETED if not snapshot.next else RunStatus.RUNNING
            return self.repo.save_state(merged)
        except Exception as exc:  # pragma: no cover - handled in tests via failure state
            return self.repo.mark_failed(run_id, str(exc))

    def _resume_without_graph(
        self,
        run: SerializedRunState,
        submission: CheckpointSubmission,
    ) -> SerializedRunState:
        try:
            self._apply_checkpoint_submission(run, submission)
            checkpoint_id = submission.checkpoint_id
            decision = submission.decision

            if checkpoint_id == CheckpointId.CONFIG:
                if decision == CheckpointDecision.REVISE:
                    return self._set_waiting_checkpoint(run, CheckpointId.CONFIG)
                run = self.agent_suite.planner(run)
                return self._set_waiting_checkpoint(run, CheckpointId.PLAN)

            if checkpoint_id == CheckpointId.PLAN:
                if decision == CheckpointDecision.REVISE:
                    run = self.agent_suite.planner(run)
                    return self._set_waiting_checkpoint(run, CheckpointId.PLAN)
                run = self.agent_suite.literature_review(run)
                run = self.agent_suite.data_collection(run)
                run = self.agent_suite.fact_checker(run)
                return self._set_waiting_checkpoint(run, CheckpointId.DATA)

            if checkpoint_id == CheckpointId.DATA:
                if decision == CheckpointDecision.REVISE:
                    run = self.agent_suite.data_collection(run)
                    run = self.agent_suite.fact_checker(run)
                    return self._set_waiting_checkpoint(run, CheckpointId.DATA)
                run = self.agent_suite.analysis(run)
                return self._set_waiting_checkpoint(run, CheckpointId.ANALYSIS)

            if checkpoint_id == CheckpointId.ANALYSIS:
                if decision == CheckpointDecision.REVISE:
                    if submission.feedback:
                        run.revision_notes.append(submission.feedback)
                    run = self.agent_suite.analysis(run)
                    return self._set_waiting_checkpoint(run, CheckpointId.ANALYSIS)
                run = self.agent_suite.writer(run)
                run = self.agent_suite.qa_synthesis(run)
                run = self.agent_suite.critical_reviewer(run)
                return self._review_or_finalize(run)

            if checkpoint_id == CheckpointId.REVIEW:
                if submission.feedback:
                    run.revision_notes.append(submission.feedback)
                if decision == CheckpointDecision.REVISE:
                    return self._set_waiting_checkpoint(run, CheckpointId.REVIEW)
                run = self.agent_suite.writer(run)
                run = self.agent_suite.qa_synthesis(run)
                run = self.agent_suite.critical_reviewer(run)
                return self._review_or_finalize(run)

            raise ValueError(f"Unsupported checkpoint '{checkpoint_id.value}'")
        except Exception as exc:  # pragma: no cover - exercised via integration tests
            return self.repo.mark_failed(run.run_id, str(exc))

    def _apply_checkpoint_submission(
        self,
        run: SerializedRunState,
        submission: CheckpointSubmission,
    ) -> None:
        record = run.checkpoint_status.get(submission.checkpoint_id.value) or CheckpointRecord(
            checkpoint_id=submission.checkpoint_id.value,
            label=self._checkpoint_label(submission.checkpoint_id),
        )
        if submission.checkpoint_id == CheckpointId.CONFIG and submission.config_patch:
            patch = submission.config_patch
            if "target_word_count" in patch:
                run.target_word_count = int(patch["target_word_count"])
            if "section_preferences" in patch:
                run.section_preferences = list(patch["section_preferences"])
            if "notes" in patch:
                run.notes = patch["notes"]
            if "output_format" in patch:
                run.output_format = OutputFormat(patch["output_format"])
        record.status = "approved" if submission.decision == CheckpointDecision.APPROVE else "revised"
        record.last_decision = submission.decision
        record.last_feedback = submission.feedback
        record.waiting_payload = None
        run.checkpoint_status[submission.checkpoint_id.value] = record
        run.active_checkpoint = None
        run.status = RunStatus.RUNNING

    def _review_or_finalize(self, run: SerializedRunState) -> SerializedRunState:
        latest_review = run.review_cycles[-1] if run.review_cycles else None
        if latest_review and (latest_review.passed or len(run.review_cycles) >= self.settings.max_review_cycles):
            return self.agent_suite.finalize(run)
        return self._set_waiting_checkpoint(run, CheckpointId.REVIEW)

    def _set_waiting_checkpoint(self, run: SerializedRunState, checkpoint_id: CheckpointId) -> SerializedRunState:
        payload = self._checkpoint_payload(run, checkpoint_id)
        record = run.checkpoint_status.get(checkpoint_id.value) or CheckpointRecord(
            checkpoint_id=checkpoint_id.value,
            label=payload["label"],
        )
        record.status = "waiting"
        record.waiting_payload = payload
        run.checkpoint_status[checkpoint_id.value] = record
        run.status = RunStatus.WAITING
        run.active_checkpoint = checkpoint_id.value
        run.current_node = f"{checkpoint_id.value}_checkpoint"
        return self.repo.save_state(run)

    def _checkpoint_payload(self, run: SerializedRunState, checkpoint_id: CheckpointId) -> dict:
        if checkpoint_id == CheckpointId.CONFIG:
            return {
                "checkpoint_id": CheckpointId.CONFIG.value,
                "label": "Config Intake",
                "topic": run.topic,
                "output_format": run.output_format.value,
                "target_word_count": run.target_word_count,
                "section_preferences": run.section_preferences or [],
                "notes": run.notes,
            }
        if checkpoint_id == CheckpointId.PLAN:
            planner = run.planner_output
            return {
                "checkpoint_id": CheckpointId.PLAN.value,
                "label": "Research Plan Approval",
                "problem_statement": planner.problem_statement if planner else "",
                "analytical_method": planner.analytical_method if planner else "",
                "section_plan": [item.model_dump(mode="json") for item in (planner.section_plan if planner else [])],
                "indicator_plan": [item.model_dump(mode="json") for item in (planner.indicator_plan if planner else [])],
            }
        if checkpoint_id == CheckpointId.DATA:
            verification = run.verification_pack
            return {
                "checkpoint_id": CheckpointId.DATA.value,
                "label": "Verified Sources and Data Approval",
                "verified_allowlist": verification.verified_allowlist if verification else [],
                "flagged_items": verification.flagged_items if verification else [],
                "correction_summary": verification.correction_summary if verification else "",
            }
        if checkpoint_id == CheckpointId.ANALYSIS:
            analysis = run.analysis_pack
            return {
                "checkpoint_id": CheckpointId.ANALYSIS.value,
                "label": "Analytical Findings Approval",
                "point_estimate": analysis.point_estimate if analysis else None,
                "confidence_interval": [
                    analysis.confidence_low if analysis else None,
                    analysis.confidence_high if analysis else None,
                ],
                "sector_attribution": analysis.sector_attribution if analysis else {},
                "scenario_commentary": analysis.scenario_commentary if analysis else "",
            }
        if checkpoint_id == CheckpointId.REVIEW:
            latest_review = run.review_cycles[-1]
            return {
                "checkpoint_id": CheckpointId.REVIEW.value,
                "label": "Review Loop",
                "cycle_number": latest_review.cycle_number,
                "composite_score": latest_review.composite_score,
                "priority_fixes": latest_review.priority_fixes,
                "passed": latest_review.passed,
            }
        raise ValueError(f"Unsupported checkpoint '{checkpoint_id.value}'")

    def _checkpoint_label(self, checkpoint_id: CheckpointId) -> str:
        labels = {
            CheckpointId.CONFIG: "Config Intake",
            CheckpointId.PLAN: "Research Plan Approval",
            CheckpointId.DATA: "Verified Sources and Data Approval",
            CheckpointId.ANALYSIS: "Analytical Findings Approval",
            CheckpointId.REVIEW: "Review Loop",
        }
        return labels[checkpoint_id]

    def _execute_from_stage(self, run: SerializedRunState, stage_id: RunStageId) -> SerializedRunState:
        try:
            if stage_id == RunStageId.CONFIG:
                return self._set_waiting_checkpoint(run, CheckpointId.CONFIG)
            if stage_id == RunStageId.PLANNER:
                run = self.agent_suite.planner(run)
                return self._set_waiting_checkpoint(run, CheckpointId.PLAN)
            if stage_id == RunStageId.LITERATURE_REVIEW:
                run = self.agent_suite.literature_review(run)
                run = self.agent_suite.data_collection(run)
                run = self.agent_suite.fact_checker(run)
                return self._set_waiting_checkpoint(run, CheckpointId.DATA)
            if stage_id == RunStageId.DATA_COLLECTION:
                run = self.agent_suite.data_collection(run)
                run = self.agent_suite.fact_checker(run)
                return self._set_waiting_checkpoint(run, CheckpointId.DATA)
            if stage_id == RunStageId.FACT_CHECKER:
                run = self.agent_suite.fact_checker(run)
                return self._set_waiting_checkpoint(run, CheckpointId.DATA)
            if stage_id == RunStageId.ANALYSIS:
                run = self.agent_suite.analysis(run)
                return self._set_waiting_checkpoint(run, CheckpointId.ANALYSIS)
            if stage_id == RunStageId.WRITER:
                run = self.agent_suite.writer(run)
                run = self.agent_suite.qa_synthesis(run)
                run = self.agent_suite.critical_reviewer(run)
                return self._review_or_finalize(run)
            if stage_id == RunStageId.QA_SYNTHESIS:
                if not run.report_versions:
                    run = self.agent_suite.writer(run)
                run = self.agent_suite.qa_synthesis(run)
                run = self.agent_suite.critical_reviewer(run)
                return self._review_or_finalize(run)
            if stage_id == RunStageId.CRITICAL_REVIEWER:
                if not run.report_versions:
                    run = self.agent_suite.writer(run)
                if not run.qa_pack:
                    run = self.agent_suite.qa_synthesis(run)
                run = self.agent_suite.critical_reviewer(run)
                return self._review_or_finalize(run)
            if stage_id == RunStageId.FINALIZE:
                if not run.review_cycles and run.report_versions:
                    if not run.qa_pack:
                        run = self.agent_suite.qa_synthesis(run)
                    run = self.agent_suite.critical_reviewer(run)
                return self.agent_suite.finalize(run)
            raise ValueError(f"Unsupported stage '{stage_id.value}'")
        except Exception as exc:  # pragma: no cover - exercised via integration tests
            return self.repo.mark_failed(run.run_id, str(exc))

    def _truncate_after_stage(self, run: SerializedRunState, stage_id: RunStageId) -> SerializedRunState:
        run.errors = []
        run.status = RunStatus.RUNNING
        run.active_checkpoint = None
        self._remove_artifacts_after_stage(run, stage_id)

        if stage_id == RunStageId.CONFIG:
            run.topic_profile = None
            run.section_plan = []
            run.indicator_plan = []
            run.planner_output = None
            run.source_registry = []
            run.literature_pack = None
            run.indicator_dataset = []
            run.indicator_pack = None
            run.verified_allowlist = []
            run.verification_pack = None
            run.analysis_pack = None
            run.report_versions = []
            run.qa_pack = None
            run.review_cycles = []
            run.revision_notes = []
            run.checkpoint_status = {}
            return self.repo.save_state(run)

        if stage_id == RunStageId.PLANNER:
            run.section_plan = []
            run.indicator_plan = []
            run.planner_output = None
            run.source_registry = []
            run.literature_pack = None
            run.indicator_dataset = []
            run.indicator_pack = None
            run.verified_allowlist = []
            run.verification_pack = None
            run.analysis_pack = None
            run.report_versions = []
            run.qa_pack = None
            run.review_cycles = []
            run.revision_notes = []
            self._drop_checkpoints(run, CheckpointId.PLAN, CheckpointId.DATA, CheckpointId.ANALYSIS, CheckpointId.REVIEW)
            return self.repo.save_state(run)

        if stage_id == RunStageId.LITERATURE_REVIEW:
            run.source_registry = []
            run.literature_pack = None
            run.indicator_dataset = []
            run.indicator_pack = None
            run.verified_allowlist = []
            run.verification_pack = None
            run.analysis_pack = None
            run.report_versions = []
            run.qa_pack = None
            run.review_cycles = []
            run.revision_notes = []
            self._drop_checkpoints(run, CheckpointId.DATA, CheckpointId.ANALYSIS, CheckpointId.REVIEW)
            return self.repo.save_state(run)

        if stage_id == RunStageId.DATA_COLLECTION:
            run.indicator_dataset = []
            run.indicator_pack = None
            run.verified_allowlist = []
            run.verification_pack = None
            run.analysis_pack = None
            run.report_versions = []
            run.qa_pack = None
            run.review_cycles = []
            run.revision_notes = []
            self._drop_checkpoints(run, CheckpointId.DATA, CheckpointId.ANALYSIS, CheckpointId.REVIEW)
            return self.repo.save_state(run)

        if stage_id == RunStageId.FACT_CHECKER:
            run.verified_allowlist = []
            run.verification_pack = None
            run.analysis_pack = None
            run.report_versions = []
            run.qa_pack = None
            run.review_cycles = []
            run.revision_notes = []
            self._drop_checkpoints(run, CheckpointId.DATA, CheckpointId.ANALYSIS, CheckpointId.REVIEW)
            return self.repo.save_state(run)

        if stage_id == RunStageId.ANALYSIS:
            run.analysis_pack = None
            run.report_versions = []
            run.qa_pack = None
            run.review_cycles = []
            run.revision_notes = []
            self._drop_checkpoints(run, CheckpointId.ANALYSIS, CheckpointId.REVIEW)
            return self.repo.save_state(run)

        if stage_id == RunStageId.WRITER:
            run.report_versions = []
            run.qa_pack = None
            run.review_cycles = []
            run.revision_notes = []
            self._drop_checkpoints(run, CheckpointId.REVIEW)
            return self.repo.save_state(run)

        if stage_id == RunStageId.QA_SYNTHESIS:
            run.qa_pack = None
            run.review_cycles = []
            run.revision_notes = []
            self._drop_checkpoints(run, CheckpointId.REVIEW)
            return self.repo.save_state(run)

        if stage_id == RunStageId.CRITICAL_REVIEWER:
            run.review_cycles = []
            self._drop_checkpoints(run, CheckpointId.REVIEW)
            return self.repo.save_state(run)

        if stage_id == RunStageId.FINALIZE:
            return self.repo.save_state(run)

        raise ValueError(f"Unsupported stage '{stage_id.value}'")

    def _drop_checkpoints(self, run: SerializedRunState, *checkpoint_ids: CheckpointId) -> None:
        for checkpoint_id in checkpoint_ids:
            run.checkpoint_status.pop(checkpoint_id.value, None)

    def _remove_artifacts_after_stage(self, run: SerializedRunState, stage_id: RunStageId) -> None:
        if stage_id == RunStageId.FINALIZE:
            return
        keep_paths: set[str] = set()
        keep_keys = set(self._artifact_keys_to_keep(stage_id))
        for key, path in list(run.artifact_paths.items()):
            if key in keep_keys or any(key.startswith(prefix) for prefix in self._artifact_key_prefixes_to_keep(stage_id)):
                keep_paths.add(path)
                continue
            self._safe_delete_path(path)
            run.artifact_paths.pop(key, None)

        if stage_id in {
            RunStageId.CONFIG,
            RunStageId.PLANNER,
            RunStageId.LITERATURE_REVIEW,
            RunStageId.DATA_COLLECTION,
            RunStageId.FACT_CHECKER,
            RunStageId.ANALYSIS,
            RunStageId.WRITER,
        }:
            for draft in run.report_versions:
                if draft.path not in keep_paths:
                    self._safe_delete_path(draft.path)
        if stage_id in {
            RunStageId.CONFIG,
            RunStageId.PLANNER,
            RunStageId.LITERATURE_REVIEW,
            RunStageId.DATA_COLLECTION,
            RunStageId.FACT_CHECKER,
            RunStageId.ANALYSIS,
        }:
            self._safe_delete_path(str(self.repo.run_dir(run.run_id) / "artifacts" / "report.md"))
            self._safe_delete_path(str(self.repo.run_dir(run.run_id) / "artifacts" / "report.pdf"))
            self._safe_delete_path(str(self.repo.run_dir(run.run_id) / "artifacts" / "final_scores.json"))
        if stage_id in {
            RunStageId.CONFIG,
            RunStageId.PLANNER,
            RunStageId.LITERATURE_REVIEW,
            RunStageId.DATA_COLLECTION,
            RunStageId.FACT_CHECKER,
            RunStageId.ANALYSIS,
            RunStageId.WRITER,
            RunStageId.QA_SYNTHESIS,
            RunStageId.CRITICAL_REVIEWER,
        }:
            self._safe_delete_path(str(self.repo.run_dir(run.run_id) / "logs" / "final_scores.md"))

    def _artifact_keys_to_keep(self, stage_id: RunStageId) -> tuple[str, ...]:
        if stage_id == RunStageId.CONFIG:
            return ()
        if stage_id == RunStageId.PLANNER:
            return ("plan_log",)
        if stage_id == RunStageId.LITERATURE_REVIEW:
            return ("plan_log", "literature_log", "references_log")
        if stage_id == RunStageId.DATA_COLLECTION:
            return ("plan_log", "literature_log", "references_log", "data_collection_log", "indicator_workbook")
        if stage_id == RunStageId.FACT_CHECKER:
            return (
                "plan_log",
                "literature_log",
                "references_log",
                "data_collection_log",
                "indicator_workbook",
                "fact_check_log",
            )
        if stage_id == RunStageId.ANALYSIS:
            return (
                "plan_log",
                "literature_log",
                "references_log",
                "data_collection_log",
                "indicator_workbook",
                "fact_check_log",
                "analysis_log",
                "analysis_script",
            )
        if stage_id == RunStageId.WRITER:
            return (
                "plan_log",
                "literature_log",
                "references_log",
                "data_collection_log",
                "indicator_workbook",
                "fact_check_log",
                "analysis_log",
                "analysis_script",
                "report_draft",
            )
        if stage_id == RunStageId.QA_SYNTHESIS:
            return (
                "plan_log",
                "literature_log",
                "references_log",
                "data_collection_log",
                "indicator_workbook",
                "fact_check_log",
                "analysis_log",
                "analysis_script",
                "report_draft",
                "report_revised",
                "report",
                "report_pdf",
            )
        if stage_id == RunStageId.CRITICAL_REVIEWER:
            return (
                "plan_log",
                "literature_log",
                "references_log",
                "data_collection_log",
                "indicator_workbook",
                "fact_check_log",
                "analysis_log",
                "analysis_script",
                "report_draft",
                "report_revised",
                "report",
                "report_pdf",
            )
        return ()

    def _artifact_key_prefixes_to_keep(self, stage_id: RunStageId) -> tuple[str, ...]:
        if stage_id == RunStageId.FINALIZE:
            return ("review_cycle_",)
        return ()

    def _safe_delete_path(self, path: str | None) -> None:
        if not path:
            return
        target = Path(path)
        try:
            if target.exists():
                target.unlink()
        except IsADirectoryError:
            return
        except FileNotFoundError:
            return


def build_default_service(root_dir: str | Path | None = None) -> RunService:
    root = Path(root_dir).resolve() if root_dir else Path(__file__).resolve().parents[2]
    return RunService(AppSettings.from_root(root))
