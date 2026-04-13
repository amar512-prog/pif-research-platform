# PIF Multi-Agent Research & Analysis Prototype

Assessment-grade prototype for the Pahle India Foundation CoDED AI Engineer brief. It implements a single-user, auditable multi-agent workflow for topic-generic policy research with a local-LLM-first runtime:

- FastAPI endpoints for `POST /runs`, `GET /runs/{run_id}`, and `POST /runs/{run_id}/checkpoint`
- LangGraph-driven checkpointed workflow with human approval gates
- Crossref-backed literature retrieval for real reference URLs, with offline fallback for deterministic testing
- Indicator-plan source resolution that prefers official data portals and can upgrade links via search-engine lookup
- Local narrative generation through an Ollama-compatible model sized for a `<=12 GB` setup
- A transparent composite-indicator analysis strategy that adapts to macro, energy, mobility, social-policy, and generic topics
- Artifact persistence under `runs/<run_id>/`
- Review-loop scoring with a bounded maximum of 3 cycles

## Project Layout

- `src/pif_research_platform/api.py`: FastAPI app and public endpoints
- `src/pif_research_platform/service.py`: run orchestration and graph driving
- `src/pif_research_platform/workflow.py`: LangGraph nodes, interrupts, and routes
- `src/pif_research_platform/agents.py`: planner, research, verification, analysis, writer, QA, reviewer
- `src/pif_research_platform/adapters/local_llm.py`: local LLM adapter with Ollama + template fallback
- `src/pif_research_platform/adapters/search.py`: Crossref-backed literature retrieval with offline fallback
- `src/pif_research_platform/topic_intelligence.py`: topic profiling and deterministic evidence fixtures
- `src/pif_research_platform/analysis/generic_topic.py`: topic-generic composite analysis strategy
- `streamlit_app.py`: lightweight checkpoint UI over the API
- `tests/test_platform.py`: service and API verification

## Setup

To make imports work cleanly from the environment:

```bash
./.conda-env/bin/pip install -r requirements.txt
```


## Local LLM Runtime

The default runtime expects a local Ollama-compatible model:

- Provider: `ollama`
- Model: `qwen3:14b`
- Memory target: `<=12 GB`
- Fallback mode: `template`
- Suggested pull: `ollama pull qwen3:14b`

Relevant environment variables:

```bash
export PIF_LOCAL_LLM_PROVIDER=ollama
export PIF_LOCAL_LLM_BASE_URL=http://127.0.0.1:11434
export PIF_LOCAL_LLM_MODEL=qwen3:14b
export PIF_LOCAL_LLM_MAX_GB=12
export PIF_LOCAL_LLM_READ_TIMEOUT=90
export PIF_LOCAL_LLM_FALLBACK=template
```

For deterministic tests or fully offline runs, set `PIF_LOCAL_LLM_PROVIDER=template`.
If you want a lighter Ollama fallback on smaller hardware, set `PIF_LOCAL_LLM_MODEL=qwen3:8b`.

## Literature Search Runtime

The default literature runtime is hybrid:

- Provider: `hybrid`
- Live source: Crossref REST API
- Fallback: `offline`

Relevant environment variables:

```bash
export PIF_SEARCH_PROVIDER=hybrid
export PIF_SEARCH_CROSSREF_BASE_URL=https://api.crossref.org
export PIF_WEB_SEARCH_PROVIDER=bing_rss
export PIF_WEB_SEARCH_BASE_URL=https://www.bing.com/search
export PIF_WEB_SEARCH_MARKET=en-IN
export PIF_SEARCH_TIMEOUT=12
export PIF_SEARCH_MAILTO=you@example.com
export PIF_SEARCH_USER_AGENT="pif-research-platform/0.1 (mailto:amaragnihotri1@hotmail.com)"
export PIF_SEARCH_FALLBACK=offline
```

Use `PIF_SEARCH_PROVIDER=offline` when you want deterministic fixture citations and official fallback URLs instead of live web lookups.

## Run The API

```bash
PYTHONPATH=src ./.conda-env/bin/python -m uvicorn pif_research_platform.api:app --app-dir src --reload
```

## Run The UI

```bash
PIF_RA_API_BASE_URL=http://127.0.0.1:8000 PYTHONPATH=src ./.conda-env/bin/streamlit run streamlit_app.py
```

## Test

```bash
PYTHONPATH=src ./.conda-env/bin/python -m pytest
```

## Notes

- Literature retrieval uses live Crossref lookups by default so report references resolve to real landing pages instead of demo URLs.
- Topic profiling is heuristic and supports multiple policy domains rather than only one hardcoded brief.
- `markdown` and `pdf` outputs are supported.
- Review-loop behavior is visible by design: the first reviewer pass typically requests stronger scenario framing and agency ownership, and the next pass should clear the benchmark.
