import html
import os
import json
from urllib.parse import quote
import requests
import streamlit as st
from typing import Any

# 1. CONSTANTS & API CONFIG


def _secret_or_env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    try:
        if name in st.secrets:
            return str(st.secrets[name])
        api_section = st.secrets.get("api")
        if isinstance(api_section, dict):
            section_key_map = {
                "PIF_RA_API_BASE_URL": "base_url",
                "PIF_RA_API_TIMEOUT_SECONDS": "timeout_seconds",
            }
            section_key = section_key_map.get(name)
            if section_key and section_key in api_section:
                return str(api_section[section_key])
    except Exception:
        pass
    return default


API_BASE_URL = _secret_or_env("PIF_RA_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_TIMEOUT_SECONDS = int(_secret_or_env("PIF_RA_API_TIMEOUT_SECONDS", "180"))

STAGES = [
    {"id": "config", "label": "Config", "description": "Topic and output settings"},
    {"id": "planner", "label": "Planner", "description": "Creates research plan"},
    {"id": "literature_review", "label": "Literature Review", "description": "Searches & synthesizes"},
    {"id": "data_collection", "label": "Data Collection", "description": "Gathers HF indicators"},
    {"id": "fact_checker", "label": "Fact Checker", "description": "Verifies citations & data"},
    {"id": "analysis", "label": "Analysis", "description": "Econometric modelling"},
    {"id": "writer", "label": "Writer", "description": "Drafts policy report"},
    {"id": "qa_synthesis", "label": "QA + Synthesis", "description": "Checks consistency"},
    {"id": "critical_reviewer", "label": "Critical Reviewer", "description": "Scores against 9 criteria"},
    {"id": "finalize", "label": "Finalize", "description": "Final output bundle"},
]

# 2. UI STYLING
st.set_page_config(page_title="PIF Research Automation", layout="wide")
st.markdown("""
    <style>
    :root {
      --bg: #f8fafc;
      --text-main: #0f172a;
      --terminal-bg: #020617;
      --color-done: #10b981; 
      --color-current: #3b82f6; 
      --color-waiting: #f59e0b; 
      --color-pending: #94a3b8; 
    }
    .stApp { background: var(--bg); color: var(--text-main); }
    .timeline-card {
      background: white; border: 1px solid #cbd5e1; border-radius: 8px;
      padding: 0.8rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 5px;
    }
    .workspace-logs-container {
      background: var(--terminal-bg); color: #f8fafc; border-radius: 8px;
      padding: 1.5rem; font-family: 'Courier New', monospace; 
      height: 650px; overflow-y: auto; border: 1px solid #1e293b;
    }
    .log-header { color: #22c55e; font-weight: bold; font-size: 0.85rem; }
    .log-content { color: #38bdf8; font-size: 12px; white-space: pre-wrap; }
    .approval-box {
      background: #fffbeb; border: 2px solid #f59e0b; padding: 1.25rem;
      margin-top: 1rem; border-radius: 8px; color: #92400e;
    }
    </style>
""", unsafe_allow_html=True)

# 3. BACKEND HELPERS
def api_get(path: str):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()

def api_post(path: str, payload: dict):
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def artifact_download_url(run_id: str, artifact_key: str) -> str:
    return f"{API_BASE_URL}/runs/{quote(run_id)}/artifacts/{quote(artifact_key)}"


def artifact_label(artifact_key: str, artifact_path: str) -> str:
    filename = os.path.basename(artifact_path) or artifact_key
    return f"{artifact_key.replace('_', ' ').title()} ({filename})"

def stage_has_result(stage_id: str, run: dict, detail: dict) -> bool:
    """Explicit check to prevent graying out after approval"""
    if stage_id == "config": return True
    mapping = {
        "planner": "planner_output", "literature_review": "literature_pack",
        "data_collection": "indicator_pack", "fact_checker": "verification_pack",
        "analysis": "analysis_pack", "writer": "report_versions",
        "qa_synthesis": "qa_pack", "critical_reviewer": "review_cycles"
    }
    if stage_id in mapping:
        return bool(detail.get(mapping[stage_id]))
    if stage_id == "finalize":
        return run.get("status") == "completed"
    return False

def get_visuals(stage_id, run, detail, active_checkpoint):
    curr_node = run.get("current_node", "")
    # Priority 1: Waiting for Human
    if active_checkpoint and stage_id in curr_node:
        return "waiting", "var(--color-waiting)"
    # Priority 2: Active Processing
    if stage_id in curr_node:
        return "current", "var(--color-current)"
    # Priority 3: Completed (Persistence check)
    if stage_has_result(stage_id, run, detail):
        return "done", "var(--color-done)"
    return "pending", "var(--color-pending)"

# 4. SIDEBAR
with st.sidebar:
    st.header("Pipeline Control")
    with st.form("new_run"):
        topic = st.text_area("Topic", value="Assess EV charging readiness in Karnataka")
        fmt = st.selectbox("Format", ["markdown", "pdf"])
        words = st.number_input("Words", value=1800)
        notes = st.text_area("Notes", value="Demonstrate the full review loop, keep the report concise, and emphasize implementer ownership.")
        
        if st.form_submit_button("Start Pipeline", type="primary"):
            try:
                res = api_post("/runs", {"topic": topic, "output_format": fmt, "target_word_count": words, "notes": notes})
                st.session_state["run_id"] = res["run_id"]
                st.rerun()
            except Exception as e: st.error(str(e))

# 5. MAIN CONTENT
run_id = st.session_state.get("run_id")
if run_id:
    try:
        run = api_get(f"/runs/{run_id}")
        detail = api_get(f"/runs/{run_id}/detail")
        checkpoint = run.get("active_checkpoint")

        col_pipe, col_logs = st.columns([1, 2], gap="medium")

        with col_pipe:
            st.write("**Agent Timeline**")
            for i, stage in enumerate(STAGES):
                _, color = get_visuals(stage["id"], run, detail, checkpoint)
                c1, c2, c3 = st.columns([1, 0.3, 1])
                with c2: st.markdown(f"<div style='text-align:center; color:{color}; font-size:22px; line-height:50px;'>●</div>", unsafe_allow_html=True)
                target = c3 if (i % 2 != 0) else c1
                with target:
                    st.markdown(f'<div class="timeline-card" style="border-left:4px solid {color}"><div style="font-weight:800; font-size:13px;">{stage["label"]}</div><div style="color:var(--text-muted); font-size:11px;">{stage["description"]}</div></div>', unsafe_allow_html=True)

        with col_logs:
            st.write("**Workspace Logs**")
            # Concat Logic: Config -> Stages -> Artifacts
            log_str = f'<div class="workspace-logs-container">'
            
            # 1. Starting Config
            log_str += f'<span class="log-header">[CONFIG]</span><br><span class="log-content">{html.escape(json.dumps({"topic": detail.get("topic"), "format": detail.get("output_format"), "target": detail.get("target_word_count")}, indent=2))}</span><hr style="border-top:1px solid #1e293b; margin:10px 0;">'
            
            # 2. Stage Results
            keys = [("planner_output", "PLANNER"), ("literature_pack", "LIT_REVIEW"), ("indicator_pack", "DATA_COLLECTION"), ("verification_pack", "FACT_CHECK"), ("analysis_pack", "ANALYSIS"), ("report_versions", "WRITER"), ("qa_pack", "QA_SYNTHESIS"), ("review_cycles", "CRITICAL_REVIEW")]
            for k, title in keys:
                if detail.get(k):
                    data = detail[k][-1] if isinstance(detail[k], list) else detail[k]
                    log_str += f'<span class="log-header">[{title}]</span><br><span class="log-content">{html.escape(json.dumps(data, indent=2))}</span><hr style="border-top:1px solid #1e293b; margin:10px 0;">'

            # 3. Final Artifacts
            if run.get("artifact_paths"):
                log_str += f'<span class="log-header">[FINAL_ARTIFACTS]</span><br><span class="log-content">{html.escape(json.dumps(run["artifact_paths"], indent=2))}</span><br>'
            
            log_str += '</div>'
            st.markdown(log_str, unsafe_allow_html=True)

            if checkpoint:
                st.markdown(f'<div class="approval-box"><strong>Gatekeeper: {checkpoint.upper()}</strong><br>Review logs and submit decision.</div>', unsafe_allow_html=True)
                with st.form("hitl"):
                    fb = st.text_area("Feedback", height=180)
                    b1, b2 = st.columns(2)
                    if b1.form_submit_button("✅ Approve", type="primary", use_container_width=True):
                        api_post(f"/runs/{run_id}/checkpoint", {"checkpoint_id": checkpoint, "decision": "approve", "feedback": fb})
                        st.rerun()
                    if b2.form_submit_button("❌ Revise", use_container_width=True):
                        api_post(f"/runs/{run_id}/checkpoint", {"checkpoint_id": checkpoint, "decision": "revise", "feedback": fb})
                        st.rerun()

        if run.get("status") == "completed" and run.get("artifact_paths"):
            st.divider()
            st.subheader("Download Artifacts")
            st.caption("Final run outputs are available below.")
            for artifact_key, artifact_path in run["artifact_paths"].items():
                left, right = st.columns([3, 1])
                with left:
                    st.markdown(f"**{artifact_label(artifact_key, artifact_path)}**")
                    st.caption(artifact_path)
                with right:
                    st.link_button(
                        "Download",
                        artifact_download_url(run_id, artifact_key),
                        use_container_width=True,
                    )
    except Exception as e: st.error(f"Sync error: {e}")
