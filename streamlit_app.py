import html
import os
import json
from pathlib import Path
from urllib.parse import quote
import requests
import streamlit as st
from typing import Any

st.set_page_config(page_title="PIF Research Automation", layout="wide")

# 1. CONSTANTS & API CONFIG


def _has_streamlit_secrets_file() -> bool:
    return any(
        path.exists()
        for path in [
            Path.home() / ".streamlit" / "secrets.toml",
            Path.cwd() / ".streamlit" / "secrets.toml",
        ]
    )


def _secret_or_env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    if not _has_streamlit_secrets_file():
        return default
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
st.markdown("""
    <style>
    @property --surface-glow {
      syntax: "<percentage>";
      inherits: false;
      initial-value: 0%;
    }

    :root {
      color-scheme: light;
      accent-color: #2563eb;
      --bg: #eef4ff;
      --canvas: #f7fbff;
      --panel: rgba(255, 255, 255, 0.88);
      --panel-strong: #ffffff;
      --panel-border: rgba(148, 163, 184, 0.32);
      --text-main: #0f172a;
      --text-muted: #64748b;
      --terminal-bg: #020617;
      --terminal-line: #1e293b;
      --color-done: #10b981; 
      --color-current: #3b82f6; 
      --color-waiting: #f59e0b; 
      --color-pending: #94a3b8; 
      --blue: #2563eb;
      --blue-hover: #1d4ed8;
      --pink: #f9a8d4;
      --red-hover: #dc2626;
      --amber: #f59e0b;
      --green: #10b981;
      --violet: #7c3aed;
      --shadow-soft: 0 14px 38px rgba(15, 23, 42, 0.08);
      --shadow-card: 0 8px 22px rgba(15, 23, 42, 0.07);
      --radius-card: clamp(14px, 1.4vw, 22px);
      --space-card: clamp(0.85rem, 0.76rem + 0.45vw, 1.2rem);
      --font-sm: clamp(0.76rem, 0.72rem + 0.18vw, 0.88rem);
      --font-md: clamp(0.9rem, 0.84rem + 0.26vw, 1rem);
      --font-title: clamp(1.1rem, 0.95rem + 0.8vw, 1.75rem);
    }

    .stApp {
      background:
        radial-gradient(circle at 6% 4%, rgba(37, 99, 235, 0.14), transparent 28rem),
        radial-gradient(circle at 82% 12%, rgba(20, 184, 166, 0.14), transparent 30rem),
        radial-gradient(circle at 74% 80%, rgba(124, 58, 237, 0.10), transparent 26rem),
        linear-gradient(180deg, #f8fbff 0%, var(--bg) 58%, #f8fafc 100%);
      color: var(--text-main);
      font-family: "IBM Plex Sans", "Aptos", "Segoe UI", sans-serif;
      min-block-size: 100dvh;
    }

    .block-container {
      max-inline-size: min(100%, 1320px);     
      inline-size: 100%;
      margin-inline: auto;
      padding-block: clamp(1rem, 2vw, 2.4rem) 3rem;
      padding-inline: clamp(1.25rem, 2vw, 2.25rem);
      container: app-shell / inline-size;
    }

    section[data-testid="stSidebar"] {
      background:
        radial-gradient(circle at 20% 0%, rgba(96, 165, 250, 0.20), transparent 15rem),
        linear-gradient(180deg, #0f172a 0%, #111827 100%);
      border-right: 1px solid rgba(148, 163, 184, 0.24);
      
    }

    section[data-testid="stSidebar"][aria-expanded="true"] {
      inline-size: clamp(21rem, 27vw, 24rem) !important;
      min-inline-size: clamp(21rem, 27vw, 24rem) !important;
      max-inline-size: 24rem !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
      inline-size: 0 !important;
      min-inline-size: 0 !important;
      max-inline-size: 0 !important;
    }

    section[data-testid="stSidebar"] > div {
      padding-block-start: 1.15rem;
      padding-inline: clamp(0.9rem, 1.15vw, 1.1rem);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
      color: #e5edf8 !important;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stCaptionContainer {
      color: #a8b4c8 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stForm"] {
      background: rgba(255, 255, 255, 0.08);
      border-color: rgba(226, 232, 240, 0.12);
      box-shadow: 0 18px 36px rgba(2, 6, 23, 0.22);
    }

    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
      background: rgba(2, 6, 23, 0.88) !important;
      border-color: rgba(148, 163, 184, 0.34) !important;
      color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
      color: #f8fafc !important;
      fill: #f8fafc !important;
    }

    div[data-testid="stForm"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: var(--radius-card);
      box-shadow: var(--shadow-soft);
      backdrop-filter: blur(18px) saturate(120%);
    }

    textarea,
    input,
    div[data-baseweb="select"] > div {
      background: #0f172a !important;
      border: 1px solid #334155 !important;
      color: #f8fafc !important;
      border-radius: 12px !important;
      box-shadow: none !important;
      font-size: var(--font-md) !important;
    }

    textarea {
      field-sizing: content;
      min-block-size: 5lh;
      max-block-size: 42dvh;
      line-height: 1.55 !important;
    }

    textarea:focus,
    input:focus,
    div[data-baseweb="select"] > div:focus-within {
      border-color: #60a5fa !important;
      box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.22) !important;
    }

    textarea::placeholder,
    input::placeholder {
      color: #94a3b8 !important;
    }

    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3 {
      letter-spacing: -0.03em;
      text-wrap: balance;
    }

    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {
      text-wrap: pretty;
    }

    div[data-testid="stMarkdownContainer"] strong {
      color: color-mix(in oklab, var(--text-main), var(--blue) 12%);
    }

    .app-hero {
      position: relative;
      isolation: isolate;
      overflow: hidden;
      background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.92)),
        radial-gradient(circle at 84% 10%, rgba(96, 165, 250, 0.28), transparent 22rem);
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: clamp(20px, 2vw, 30px);
      box-shadow: 0 28px 70px rgba(15, 23, 42, 0.22);
      color: #f8fafc;
      padding: clamp(1.2rem, 1rem + 1.2vw, 2rem);
      margin-block-end: 1.35rem;
    }

    .app-hero::before {
      content: "";
      position: absolute;
      inset: auto -8% -44% auto;
      inline-size: 22rem;
      block-size: 22rem;
      border-radius: 50%;
      background: color-mix(in oklab, var(--blue), transparent 55%);
      filter: blur(18px);
      opacity: 0.7;
      z-index: -1;
    }

    .app-hero__eyebrow {
      color: #93c5fd;
      font-size: 0.75rem;
      font-weight: 900;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      margin-block-end: 0.4rem;
    }

    .app-hero__title {
      font-size: clamp(1.45rem, 1.05rem + 1.7vw, 2.8rem);
      line-height: 1.03;
      letter-spacing: -0.055em;
      font-weight: 900;
      max-inline-size: 24ch;
      text-wrap: balance;
    }

    .app-hero__meta {
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      margin-block-start: 1.15rem;
    }

    .meta-pill,
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      border-radius: 999px;
      padding: 0.4rem 0.65rem;
      font-size: 0.74rem;
      font-weight: 800;
      letter-spacing: 0.01em;
      white-space: nowrap;
    }

    .meta-pill {
      background: rgba(255, 255, 255, 0.10);
      border: 1px solid rgba(226, 232, 240, 0.15);
      color: #dbeafe;
    }

    .status-pill {
      text-transform: uppercase;
    }

    .status-pill--completed {
      background: rgba(16, 185, 129, 0.16);
      color: #bbf7d0;
      border: 1px solid rgba(74, 222, 128, 0.28);
    }

    .status-pill--running {
      background: rgba(59, 130, 246, 0.16);
      color: #bfdbfe;
      border: 1px solid rgba(96, 165, 250, 0.28);
    }

    .status-pill--failed {
      background: rgba(239, 68, 68, 0.18);
      color: #fecaca;
      border: 1px solid rgba(248, 113, 113, 0.32);
    }

    .status-pill--waiting_for_checkpoint {
      background: rgba(245, 158, 11, 0.18);
      color: #fde68a;
      border: 1px solid rgba(251, 191, 36, 0.32);
    }

    .section-label {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      margin: 0.25rem 0 0.75rem;
      color: var(--text-main);
      font-size: var(--font-sm);
      font-weight: 900;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .section-label__hint {
      color: var(--text-muted);
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.02em;
      text-transform: none;
    }

    .timeline-card {
      position: relative;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.92)),
        radial-gradient(circle at 0% 0%, color-mix(in oklab, var(--blue), white 78%), transparent 34%);
      border: 1px solid rgba(203, 213, 225, 0.9);
      border-radius: var(--radius-card);
      padding: var(--space-card);
      box-shadow: var(--shadow-card);
      margin-bottom: 0.55rem;
      transition:
        transform 180ms cubic-bezier(.2, .8, .2, 1),
        box-shadow 180ms ease,
        border-color 180ms ease,
        --surface-glow 220ms ease;
      overflow: clip;
      container: timeline-card / inline-size;
    }

    div[data-testid="stHorizontalBlock"]:has(.timeline-dot) {
      position: relative;
    }

    div[data-testid="stHorizontalBlock"]:has(.timeline-dot)::before {
      content: "";
      position: absolute;
      inset-block: -0.45rem;
      inset-inline-start: 50%;
      inline-size: 1px;
      translate: -50% 0;
      background: linear-gradient(180deg, transparent, rgba(148, 163, 184, 0.45), transparent);
      z-index: 0;
    }

    .timeline-card__top {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: start;
      gap: 0.6rem;
      position: relative;
      z-index: 1;
    }

    .timeline-card__title {
      font-weight: 900;
      font-size: var(--font-sm);
      color: var(--text-main);
      letter-spacing: -0.01em;
      min-inline-size: 0;
    }

    .timeline-card__desc {
      position: relative;
      z-index: 1;
      color: var(--text-muted);
      font-size: 0.74rem;
      line-height: 1.35;
      margin-block-start: 0.24rem;
      text-wrap: pretty;
    }

    .timeline-card__chip {
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.38);
      color: #475569;
      background: rgba(248, 250, 252, 0.8);
      padding: 0.22rem 0.45rem;
      font-size: 0.62rem;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      white-space: nowrap;
      justify-self: end;
      max-inline-size: 8.5ch;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .timeline-card__chip--done {
      color: #047857;
      background: rgba(16, 185, 129, 0.12);
      border-color: rgba(16, 185, 129, 0.28);
    }

    .timeline-card__chip--current {
      color: #1d4ed8;
      background: rgba(59, 130, 246, 0.12);
      border-color: rgba(59, 130, 246, 0.28);
    }

    .timeline-card__chip--waiting {
      color: #92400e;
      background: rgba(245, 158, 11, 0.14);
      border-color: rgba(245, 158, 11, 0.34);
    }

    .timeline-card__chip--pending {
      color: #64748b;
      background: rgba(241, 245, 249, 0.92);
    }

    .timeline-dot {
      display: grid;
      place-items: center;
      inline-size: 2rem;
      block-size: 2rem;
      margin-inline: auto;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid rgba(148, 163, 184, 0.42);
      box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
      font-size: 1.1rem;
      line-height: 1;
      position: relative;
      z-index: 1;
    }

    .timeline-card::after {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(
        90deg,
        color-mix(in oklab, var(--blue), transparent calc(100% - var(--surface-glow))),
        transparent 52%
      );
      opacity: 0.42;
      transition: opacity 160ms ease;
      pointer-events: none;
    }

    .timeline-card:hover {
      transform: translateY(-2px);
      border-color: rgba(59, 130, 246, 0.55);
      box-shadow: 0 18px 36px rgba(15, 23, 42, 0.12);
      --surface-glow: 10%;
    }

    .workspace-logs-container {
      background:
        linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(2, 6, 23, 0.98)),
        var(--terminal-bg);
      color: #f8fafc;
      border-radius: var(--radius-card);
      padding: clamp(1rem, 1.4vw, 1.5rem);
      font-family: "SFMono-Regular", "Cascadia Code", "Courier New", monospace; 
      block-size: min(650px, 72dvh);
      min-block-size: 420px;
      overflow-y: auto;
      overflow-wrap: anywhere;
      border: 1px solid rgba(96, 165, 250, 0.25);
      box-shadow: 0 22px 52px rgba(2, 6, 23, 0.28);
      scrollbar-color: #475569 #020617;
      scrollbar-width: thin;
    }

    .workspace-logs-container::selection {
      background: rgba(125, 211, 252, 0.26);
      color: #ffffff;
    }

    .workspace-logs-container::-webkit-scrollbar {
      width: 10px;
    }

    .workspace-logs-container::-webkit-scrollbar-track {
      background: #020617;
    }

    .workspace-logs-container::-webkit-scrollbar-thumb {
      background: #475569;
      border-radius: 999px;
    }

    .log-header {
      color: #34d399;
      font-weight: 800;
      font-size: var(--font-sm);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .log-content {
      color: #7dd3fc;
      font-size: clamp(0.72rem, 0.68rem + 0.15vw, 0.82rem);
      line-height: 1.62;
      white-space: pre-wrap;
    }

    .approval-box {
      background: linear-gradient(135deg, #fffbeb 0%, #fff7ed 100%);
      border: 1px solid #fbbf24;
      padding: var(--space-card);
      margin-block: 1.1rem 0.85rem;
      border-radius: var(--radius-card);
      color: #92400e;
      box-shadow: 0 14px 28px rgba(245, 158, 11, 0.13);
      text-wrap: pretty;
    }

    div[data-testid="stForm"]:has(button[data-testid="stBaseButton-primaryFormSubmit"]) {
      padding-top: 0.65rem;
    }

    div[data-testid="stForm"]:has(button[data-testid="stBaseButton-primaryFormSubmit"]) textarea {
      background: #111827 !important;
      border-color: rgba(96, 165, 250, 0.28) !important;
    }

    .artifact-card {
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid rgba(203, 213, 225, 0.72);
      border-radius: var(--radius-card);
      box-shadow: var(--shadow-card);
      padding: 0.95rem 1rem;
      margin-block: 0.45rem;
      min-block-size: 4.15rem;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 1rem;
      align-items: center;
      align-content: center;
    }

    .artifact-card__name {
      color: var(--text-main);
      font-weight: 900;
      letter-spacing: -0.01em;
    }

    .artifact-card__file {
      color: var(--text-muted);
      font-size: 0.78rem;
      overflow-wrap: anywhere;
    }

    .artifact-download {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-inline-size: 8.25rem;
      min-block-size: 2.65rem;
      border-radius: 12px;
      background: #0f172a;
      color: #ffffff !important;
      text-decoration: none !important;
      font-weight: 900;
      box-shadow: 0 10px 22px rgba(15, 23, 42, 0.14);
      transition: transform 160ms ease, background 160ms ease, box-shadow 160ms ease;
    }

    .artifact-download:hover {
      background: var(--blue);
      transform: translateY(-1px);
      box-shadow: 0 14px 30px rgba(37, 99, 235, 0.22);
    }

    @media (width <= 760px) {
      .artifact-card {
        grid-template-columns: 1fr;
      }

      .artifact-download {
        inline-size: 100%;
      }
    }

    .empty-state {
      min-block-size: 68dvh;
      display: grid;
      place-items: center;
      text-align: center;
    }

    .empty-state__panel {
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(203, 213, 225, 0.72);
      border-radius: clamp(22px, 2vw, 32px);
      box-shadow: var(--shadow-soft);
      padding: clamp(1.5rem, 1rem + 2vw, 3rem);
      max-inline-size: 54rem;
      backdrop-filter: blur(18px);
    }

    .empty-state__title {
      color: var(--text-main);
      font-size: clamp(1.5rem, 1rem + 2vw, 3rem);
      font-weight: 950;
      letter-spacing: -0.06em;
      line-height: 1.02;
      text-wrap: balance;
    }

    .empty-state__copy {
      color: var(--text-muted);
      font-size: clamp(0.95rem, 0.9rem + 0.25vw, 1.1rem);
      line-height: 1.6;
      margin-block-start: 0.8rem;
      text-wrap: pretty;
    }

    .stLinkButton a,
    div[data-testid="stLinkButton"] a {
      border-radius: 12px !important;
      background: #0f172a !important;
      color: #ffffff !important;
      border: 1px solid #1e293b !important;
      box-shadow: 0 10px 22px rgba(15, 23, 42, 0.12) !important;
      transition: transform 160ms ease, background 160ms ease !important;
    }

    .stLinkButton a:hover,
    div[data-testid="stLinkButton"] a:hover {
      background: #2563eb !important;
      border-color: #2563eb !important;
      transform: translateY(-1px);
    }

    button[kind="primary"],
    button[kind="primaryFormSubmit"],
    button[data-testid="stBaseButton-primaryFormSubmit"] {
      background: var(--blue) !important;
      border: 1px solid var(--blue) !important;
      color: #ffffff !important;
      border-radius: 12px !important;
      box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24) !important;
      font-weight: 800 !important;
      transition: transform 160ms ease, background 160ms ease, box-shadow 160ms ease !important;
      min-block-size: 2.55rem !important;
    }
    button[kind="primary"]:hover,
    button[kind="primaryFormSubmit"]:hover,
    button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
      background: var(--blue-hover) !important;
      border-color: var(--blue-hover) !important;
      color: #ffffff !important;
      transform: translateY(-1px);
      box-shadow: 0 14px 30px rgba(29, 78, 216, 0.32) !important;
    }
    button[kind="primary"]:focus,
    button[kind="primary"]:focus-visible,
    button[kind="primaryFormSubmit"]:focus,
    button[kind="primaryFormSubmit"]:focus-visible,
    button[data-testid="stBaseButton-primaryFormSubmit"]:focus,
    button[data-testid="stBaseButton-primaryFormSubmit"]:focus-visible {
      box-shadow: 0 0 0 0.2rem rgba(37, 99, 235, 0.25) !important;
    }
    button[kind="secondary"],
    button[kind="secondaryFormSubmit"],
    button[data-testid="stBaseButton-secondaryFormSubmit"] {
      background: var(--pink) !important;
      border: 1px solid #f472b6 !important;
      color: #831843 !important;
      border-radius: 12px !important;
      box-shadow: 0 10px 24px rgba(244, 114, 182, 0.20) !important;
      font-weight: 800 !important;
      transition: transform 160ms ease, background 160ms ease, box-shadow 160ms ease !important;
      min-block-size: 2.55rem !important;
    }
    button[kind="secondary"]:hover,
    button[kind="secondaryFormSubmit"]:hover,
    button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
      background: var(--red-hover) !important;
      border-color: var(--red-hover) !important;
      color: #ffffff !important;
      transform: translateY(-1px);
      box-shadow: 0 14px 30px rgba(220, 38, 38, 0.26) !important;
    }
    button[kind="secondary"]:focus,
    button[kind="secondary"]:focus-visible,
    button[kind="secondaryFormSubmit"]:focus,
    button[kind="secondaryFormSubmit"]:focus-visible,
    button[data-testid="stBaseButton-secondaryFormSubmit"]:focus,
    button[data-testid="stBaseButton-secondaryFormSubmit"]:focus-visible {
      box-shadow: 0 0 0 0.2rem rgba(244, 114, 182, 0.25) !important;
    }

    hr {
      border-color: var(--terminal-line) !important;
    }

    @supports not (color: color-mix(in oklab, white, black)) {
      div[data-testid="stMarkdownContainer"] strong {
        color: var(--text-main);
      }

      .timeline-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
      }

      .timeline-card::after {
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.06), transparent 52%);
      }
    }

    @container app-shell (width < 780px) {
      .workspace-logs-container {
        block-size: min(560px, 68dvh);
      }

      .timeline-card {
        padding: 0.82rem;
      }
    }

    @media (width <= 760px) {
      .block-container {
        padding-inline: 0.75rem;
      }

      .workspace-logs-container {
        min-block-size: 360px;
      }
    }

    @media (prefers-reduced-motion: reduce) {
      *,
      *::before,
      *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
      }

      .timeline-card:hover,
      .stLinkButton a:hover,
      div[data-testid="stLinkButton"] a:hover,
      button:hover {
        transform: none !important;
      }
    }

    @media (prefers-contrast: more) {
      :root {
        --panel-border: rgba(15, 23, 42, 0.74);
        --text-muted: #334155;
      }

      .timeline-card,
      .workspace-logs-container,
      .approval-box {
        border-width: 2px;
      }
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
    filename = os.path.basename(str(artifact_path)) or artifact_key
    return f"{artifact_key.replace('_', ' ').title()} ({filename})"


def artifact_filename(artifact_path: str) -> str:
    artifact_path_text = str(artifact_path)
    return os.path.basename(artifact_path_text) or artifact_path_text

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


def status_class(status: str | None) -> str:
    safe_status = (status or "running").replace(" ", "_").lower()
    if safe_status not in {"completed", "running", "failed", "waiting_for_checkpoint"}:
        return "running"
    return safe_status


def stage_chip_label(status: str) -> str:
    return {
        "done": "Done",
        "current": "Live",
        "waiting": "Review",
        "pending": "Pending",
    }.get(status, status.title())

# 4. SIDEBAR
with st.sidebar:
    st.header("Pipeline Control")
    st.caption(f"API: {API_BASE_URL}")
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
if not run_id:
    st.markdown(
        """
        <div class="empty-state">
          <div class="empty-state__panel">
            <div class="app-hero__eyebrow">PIF Research Automation</div>
            <div class="empty-state__title">Build an auditable multi-agent policy brief from one topic.</div>
            <div class="empty-state__copy">
              Start a pipeline from the sidebar to generate a plan, sources, indicators, analysis, review scorecard,
              and downloadable artifacts.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    try:
        run = api_get(f"/runs/{run_id}")
        detail = api_get(f"/runs/{run_id}/detail")
        checkpoint = run.get("active_checkpoint")
        status = run.get("status", "running")
        topic_title = detail.get("topic") or run.get("topic") or "Research automation run"
        current_node = run.get("current_node") or "Not started"
        checkpoint_label = checkpoint or "None"

        st.markdown(
            f"""
            <section class="app-hero">
              <div class="app-hero__eyebrow">Assessment Workspace</div>
              <div class="app-hero__title">{html.escape(topic_title)}</div>
              <div class="app-hero__meta">
                <span class="status-pill status-pill--{status_class(status)}">{html.escape(status.replace("_", " "))}</span>
                <span class="meta-pill">Run ID: {html.escape(str(run_id))}</span>
                <span class="meta-pill">Node: {html.escape(str(current_node))}</span>
                <span class="meta-pill">Checkpoint: {html.escape(str(checkpoint_label))}</span>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

        col_pipe, col_logs = st.columns([1, 2], gap="medium")

        with col_pipe:
            st.markdown(
                '<div class="section-label"><span>Agent Timeline</span><span class="section-label__hint">Live execution path</span></div>',
                unsafe_allow_html=True,
            )
            for i, stage in enumerate(STAGES):
                stage_state, color = get_visuals(stage["id"], run, detail, checkpoint)
                c1, c2, c3 = st.columns([1, 0.3, 1])
                with c2:
                    st.markdown(
                        f'<div class="timeline-dot" style="color:{color};">●</div>',
                        unsafe_allow_html=True,
                    )
                target = c3 if (i % 2 != 0) else c1
                with target:
                    st.markdown(
                        f"""
                        <div class="timeline-card" style="border-left:4px solid {color}">
                          <div class="timeline-card__top">
                            <div class="timeline-card__title">{html.escape(stage["label"])}</div>
                            <div class="timeline-card__chip timeline-card__chip--{stage_state}">
                              {html.escape(stage_chip_label(stage_state))}
                            </div>
                          </div>
                          <div class="timeline-card__desc">{html.escape(stage["description"])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with col_logs:
            st.markdown(
                '<div class="section-label"><span>Workspace Logs</span><span class="section-label__hint">Audit trail</span></div>',
                unsafe_allow_html=True,
            )
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

        if run.get("artifact_paths"):
            st.divider()
            artifact_hint = "Final output bundle" if run.get("status") == "completed" else "Generated so far"
            st.markdown(
                f'<div class="section-label"><span>Download Artifacts</span><span class="section-label__hint">{artifact_hint}</span></div>',
                unsafe_allow_html=True,
            )
            for artifact_key, artifact_path in run["artifact_paths"].items():
                download_url = artifact_download_url(run_id, artifact_key)
                st.markdown(
                    f"""
                    <div class="artifact-card">
                      <div>
                          <div class="artifact-card__name">{html.escape(artifact_label(artifact_key, artifact_path))}</div>
                          <div class="artifact-card__file">{html.escape(artifact_filename(artifact_path))}</div>
                      </div>
                      <a class="artifact-download" href="{html.escape(download_url)}" target="_blank" rel="noopener noreferrer">Download</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    except Exception as e: st.error(f"Sync error: {e}")
