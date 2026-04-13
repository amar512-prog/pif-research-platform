from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .agents import AgentSuite
from .models import CheckpointDecision, CheckpointId, CheckpointRecord, ResearchRunState, SerializedRunState


def build_workflow(agent_suite: AgentSuite):
    graph = StateGraph(ResearchRunState)

    def config_checkpoint(state: ResearchRunState) -> ResearchRunState:
        run = SerializedRunState.model_validate(state)
        payload = {
            "checkpoint_id": CheckpointId.CONFIG.value,
            "label": "Config Intake",
            "topic": run.topic,
            "output_format": run.output_format.value,
            "target_word_count": run.target_word_count,
            "section_preferences": run.section_preferences or [],
            "notes": run.notes,
        }
        response = interrupt(payload)
        run.checkpoint_status[CheckpointId.CONFIG.value] = run.checkpoint_status.get(
            CheckpointId.CONFIG.value
        ) or _checkpoint_record(CheckpointId.CONFIG.value, "Config Intake")
        if response.get("config_patch"):
            patch = response["config_patch"]
            if "target_word_count" in patch:
                run.target_word_count = int(patch["target_word_count"])
            if "section_preferences" in patch:
                run.section_preferences = list(patch["section_preferences"])
            if "notes" in patch:
                run.notes = patch["notes"]
            if "output_format" in patch:
                from .models import OutputFormat

                run.output_format = OutputFormat(patch["output_format"])
        record = run.checkpoint_status[CheckpointId.CONFIG.value]
        decision = CheckpointDecision(response["decision"])
        record.status = "approved" if decision == CheckpointDecision.APPROVE else "revised"
        record.last_decision = decision
        record.last_feedback = response.get("feedback")
        record.waiting_payload = None
        run.active_checkpoint = None
        return run.model_dump(mode="json")

    def plan_checkpoint(state: ResearchRunState) -> ResearchRunState:
        run = SerializedRunState.model_validate(state)
        planner = run.planner_output
        payload = {
            "checkpoint_id": CheckpointId.PLAN.value,
            "label": "Research Plan Approval",
            "problem_statement": planner.problem_statement if planner else "",
            "analytical_method": planner.analytical_method if planner else "",
            "section_plan": [item.model_dump(mode="json") for item in (planner.section_plan if planner else [])],
            "indicator_plan": [item.model_dump(mode="json") for item in (planner.indicator_plan if planner else [])],
        }
        response = interrupt(payload)
        run.checkpoint_status[CheckpointId.PLAN.value] = run.checkpoint_status.get(
            CheckpointId.PLAN.value
        ) or _checkpoint_record(CheckpointId.PLAN.value, "Research Plan Approval")
        record = run.checkpoint_status[CheckpointId.PLAN.value]
        decision = CheckpointDecision(response["decision"])
        record.status = "approved" if decision == CheckpointDecision.APPROVE else "revised"
        record.last_decision = decision
        record.last_feedback = response.get("feedback")
        record.waiting_payload = None
        run.active_checkpoint = None
        return run.model_dump(mode="json")

    def data_checkpoint(state: ResearchRunState) -> ResearchRunState:
        run = SerializedRunState.model_validate(state)
        verification = run.verification_pack
        payload = {
            "checkpoint_id": CheckpointId.DATA.value,
            "label": "Verified Sources and Data Approval",
            "verified_allowlist": verification.verified_allowlist if verification else [],
            "flagged_items": verification.flagged_items if verification else [],
            "correction_summary": verification.correction_summary if verification else "",
        }
        response = interrupt(payload)
        run.checkpoint_status[CheckpointId.DATA.value] = run.checkpoint_status.get(
            CheckpointId.DATA.value
        ) or _checkpoint_record(CheckpointId.DATA.value, "Verified Sources and Data Approval")
        record = run.checkpoint_status[CheckpointId.DATA.value]
        decision = CheckpointDecision(response["decision"])
        record.status = "approved" if decision == CheckpointDecision.APPROVE else "revised"
        record.last_decision = decision
        record.last_feedback = response.get("feedback")
        record.waiting_payload = None
        run.active_checkpoint = None
        return run.model_dump(mode="json")

    def analysis_checkpoint(state: ResearchRunState) -> ResearchRunState:
        run = SerializedRunState.model_validate(state)
        analysis = run.analysis_pack
        payload = {
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
        response = interrupt(payload)
        run.checkpoint_status[CheckpointId.ANALYSIS.value] = run.checkpoint_status.get(
            CheckpointId.ANALYSIS.value
        ) or _checkpoint_record(CheckpointId.ANALYSIS.value, "Analytical Findings Approval")
        record = run.checkpoint_status[CheckpointId.ANALYSIS.value]
        decision = CheckpointDecision(response["decision"])
        record.status = "approved" if decision == CheckpointDecision.APPROVE else "revised"
        record.last_decision = decision
        record.last_feedback = response.get("feedback")
        record.waiting_payload = None
        run.active_checkpoint = None
        return run.model_dump(mode="json")

    def review_checkpoint(state: ResearchRunState) -> ResearchRunState:
        run = SerializedRunState.model_validate(state)
        latest_review = run.review_cycles[-1]
        payload = {
            "checkpoint_id": CheckpointId.REVIEW.value,
            "label": "Review Loop",
            "cycle_number": latest_review.cycle_number,
            "composite_score": latest_review.composite_score,
            "priority_fixes": latest_review.priority_fixes,
            "passed": latest_review.passed,
        }
        response = interrupt(payload)
        run.checkpoint_status[CheckpointId.REVIEW.value] = run.checkpoint_status.get(
            CheckpointId.REVIEW.value
        ) or _checkpoint_record(CheckpointId.REVIEW.value, "Review Loop")
        record = run.checkpoint_status[CheckpointId.REVIEW.value]
        decision = CheckpointDecision(response["decision"])
        record.status = "approved" if decision == CheckpointDecision.APPROVE else "revised"
        record.last_decision = decision
        record.last_feedback = response.get("feedback")
        record.waiting_payload = None
        if response.get("feedback"):
            run.revision_notes.append(response["feedback"])
        run.active_checkpoint = None
        return run.model_dump(mode="json")

    graph.add_node("config_checkpoint", config_checkpoint)
    graph.add_node("planner", lambda state: agent_suite.planner(SerializedRunState.model_validate(state)).model_dump(mode="json"))
    graph.add_node("plan_checkpoint", plan_checkpoint)
    graph.add_node("literature_review", lambda state: agent_suite.literature_review(SerializedRunState.model_validate(state)).model_dump(mode="json"))
    graph.add_node("data_collection", lambda state: agent_suite.data_collection(SerializedRunState.model_validate(state)).model_dump(mode="json"))
    graph.add_node("fact_checker", lambda state: agent_suite.fact_checker(SerializedRunState.model_validate(state)).model_dump(mode="json"))
    graph.add_node("data_checkpoint", data_checkpoint)
    graph.add_node("analysis", lambda state: agent_suite.analysis(SerializedRunState.model_validate(state)).model_dump(mode="json"))
    graph.add_node("analysis_checkpoint", analysis_checkpoint)
    graph.add_node("writer", lambda state: agent_suite.writer(SerializedRunState.model_validate(state)).model_dump(mode="json"))
    graph.add_node("qa_synthesis", lambda state: agent_suite.qa_synthesis(SerializedRunState.model_validate(state)).model_dump(mode="json"))
    graph.add_node("critical_reviewer", lambda state: agent_suite.critical_reviewer(SerializedRunState.model_validate(state)).model_dump(mode="json"))
    graph.add_node("review_checkpoint", review_checkpoint)
    graph.add_node("finalize", lambda state: agent_suite.finalize(SerializedRunState.model_validate(state)).model_dump(mode="json"))

    graph.add_edge(START, "config_checkpoint")
    graph.add_conditional_edges("config_checkpoint", lambda state: "planner" if _decision(state, CheckpointId.CONFIG) == "approve" else "config_checkpoint", {"planner": "planner", "config_checkpoint": "config_checkpoint"})
    graph.add_edge("planner", "plan_checkpoint")
    graph.add_conditional_edges("plan_checkpoint", lambda state: "literature_review" if _decision(state, CheckpointId.PLAN) == "approve" else "planner", {"literature_review": "literature_review", "planner": "planner"})
    graph.add_edge("literature_review", "data_collection")
    graph.add_edge("data_collection", "fact_checker")
    graph.add_edge("fact_checker", "data_checkpoint")
    graph.add_conditional_edges("data_checkpoint", lambda state: "analysis" if _decision(state, CheckpointId.DATA) == "approve" else "data_collection", {"analysis": "analysis", "data_collection": "data_collection"})
    graph.add_edge("analysis", "analysis_checkpoint")
    graph.add_conditional_edges("analysis_checkpoint", lambda state: "writer" if _decision(state, CheckpointId.ANALYSIS) == "approve" else "analysis", {"writer": "writer", "analysis": "analysis"})
    graph.add_edge("writer", "qa_synthesis")
    graph.add_edge("qa_synthesis", "critical_reviewer")
    graph.add_conditional_edges(
        "critical_reviewer",
        lambda state: _review_route(state, agent_suite.settings.max_review_cycles),
        {"finalize": "finalize", "review_checkpoint": "review_checkpoint"},
    )
    graph.add_conditional_edges(
        "review_checkpoint",
        lambda state: "writer" if _decision(state, CheckpointId.REVIEW) == "approve" else "review_checkpoint",
        {"writer": "writer", "review_checkpoint": "review_checkpoint"},
    )
    graph.add_edge("finalize", END)
    return graph


def _checkpoint_record(checkpoint_id: str, label: str) -> CheckpointRecord:
    return _checkpoint_record_model(checkpoint_id, label)


def _checkpoint_record_model(checkpoint_id: str, label: str) -> CheckpointRecord:
    return CheckpointRecord(checkpoint_id=checkpoint_id, label=label)


def _decision(state: ResearchRunState, checkpoint_id: CheckpointId) -> str | None:
    run = SerializedRunState.model_validate(state)
    record = run.checkpoint_status.get(checkpoint_id.value)
    return record.last_decision.value if record and record.last_decision else None


def _review_route(state: ResearchRunState, max_cycles: int) -> str:
    run = SerializedRunState.model_validate(state)
    latest = run.review_cycles[-1]
    if latest.passed or len(run.review_cycles) >= max_cycles:
        return "finalize"
    return "review_checkpoint"
