from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SECTIONS = [
    "Executive Summary",
    "Research Question and Context",
    "Literature Review",
    "Indicator and Data Summary",
    "Methodology",
    "Key Findings",
    "Policy Recommendations",
    "Limitations",
    "References",
]


@dataclass(slots=True)
class LocalLLMSettings:
    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model_name: str = "qwen3:14b"
    max_model_memory_gb: int = 12
    connect_timeout_seconds: float = 0.35
    read_timeout_seconds: float = 90.0
    temperature: float = 0.2
    fallback_provider: str = "template"


@dataclass(slots=True)
class SearchSettings:
    provider: str = "hybrid"
    crossref_base_url: str = "https://api.crossref.org"
    web_search_provider: str = "bing_rss"
    web_search_base_url: str = "https://www.bing.com/search"
    web_search_market: str = "en-IN"
    web_search_timeout_seconds: float = 2.5
    max_web_queries_per_indicator: int = 1
    timeout_seconds: float = 12.0
    mailto: str = "research-platform@example.com"
    user_agent: str = "pif-research-platform/0.1 (mailto:research-platform@example.com)"
    max_results: int = 8
    fallback_provider: str = "offline"


@dataclass(slots=True)
class AppSettings:
    root_dir: Path
    runs_dir: Path
    database_path: Path
    default_word_count: int = 1800
    max_review_cycles: int = 3
    default_sections: list[str] = field(default_factory=lambda: list(DEFAULT_SECTIONS))
    local_llm: LocalLLMSettings = field(default_factory=LocalLLMSettings)
    search: SearchSettings = field(default_factory=SearchSettings)

    @classmethod
    def from_root(cls, root_dir: Path | str) -> "AppSettings":
        root = Path(root_dir).resolve()
        return cls(
            root_dir=root,
            runs_dir=root / "runs",
            database_path=root / "research_runs.db",
            local_llm=LocalLLMSettings(
                provider=os.environ.get("PIF_LOCAL_LLM_PROVIDER", "ollama"),
                base_url=os.environ.get("PIF_LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434"),
                model_name=os.environ.get("PIF_LOCAL_LLM_MODEL", "qwen3:14b"),
                max_model_memory_gb=int(os.environ.get("PIF_LOCAL_LLM_MAX_GB", "12")),
                connect_timeout_seconds=float(os.environ.get("PIF_LOCAL_LLM_CONNECT_TIMEOUT", "0.35")),
                read_timeout_seconds=float(os.environ.get("PIF_LOCAL_LLM_READ_TIMEOUT", "90.0")),
                temperature=float(os.environ.get("PIF_LOCAL_LLM_TEMPERATURE", "0.2")),
                fallback_provider=os.environ.get("PIF_LOCAL_LLM_FALLBACK", "template"),
            ),
            search=SearchSettings(
                provider=os.environ.get("PIF_SEARCH_PROVIDER", "hybrid"),
                crossref_base_url=os.environ.get("PIF_SEARCH_CROSSREF_BASE_URL", "https://api.crossref.org"),
                web_search_provider=os.environ.get("PIF_WEB_SEARCH_PROVIDER", "bing_rss"),
                web_search_base_url=os.environ.get("PIF_WEB_SEARCH_BASE_URL", "https://www.bing.com/search"),
                web_search_market=os.environ.get("PIF_WEB_SEARCH_MARKET", "en-IN"),
                web_search_timeout_seconds=float(os.environ.get("PIF_WEB_SEARCH_TIMEOUT", "2.5")),
                max_web_queries_per_indicator=int(os.environ.get("PIF_MAX_WEB_QUERIES_PER_INDICATOR", "1")),
                timeout_seconds=float(os.environ.get("PIF_SEARCH_TIMEOUT", "12.0")),
                mailto=os.environ.get("PIF_SEARCH_MAILTO", "research-platform@example.com"),
                user_agent=os.environ.get(
                    "PIF_SEARCH_USER_AGENT",
                    "pif-research-platform/0.1 (mailto:research-platform@example.com)",
                ),
                max_results=int(os.environ.get("PIF_SEARCH_MAX_RESULTS", "8")),
                fallback_provider=os.environ.get("PIF_SEARCH_FALLBACK", "offline"),
            ),
        )
