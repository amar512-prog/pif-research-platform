from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests

from ..config import LocalLLMSettings


@dataclass(slots=True)
class LocalLLMAdapter:
    settings: LocalLLMSettings
    _ollama_reachable: bool | None = None

    def runtime_label(self) -> str:
        return f"{self.settings.provider}:{self.settings.model_name}<=${self.settings.max_model_memory_gb}GB".replace("$", "")

    def complete(self, *, system_prompt: str, user_prompt: str, fallback: str) -> str:
        if self.settings.provider == "template":
            return fallback
        if self.settings.provider in {"ollama", "auto"}:
            result = self._ollama_complete(system_prompt=system_prompt, user_prompt=user_prompt)
            if result:
                return result
        return fallback

    def complete_json(self, *, system_prompt: str, user_prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
        fallback_text = json.dumps(fallback)
        text = self.complete(system_prompt=system_prompt, user_prompt=user_prompt, fallback=fallback_text)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else fallback
        except json.JSONDecodeError:
            return fallback

    def _ollama_complete(self, *, system_prompt: str, user_prompt: str) -> str | None:
        if self._ollama_reachable is False:
            return None
        payload = {
            "model": self.settings.model_name,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": self.settings.temperature,
            },
        }
        try:
            response = requests.post(
                f"{self.settings.base_url.rstrip('/')}/api/generate",
                json=payload,
                timeout=(
                    self.settings.connect_timeout_seconds,
                    self.settings.read_timeout_seconds,
                ),
            )
            response.raise_for_status()
            self._ollama_reachable = True
            data = response.json()
            text = data.get("response", "").strip()
            return text or None
        except Exception:
            self._ollama_reachable = False
            return None
