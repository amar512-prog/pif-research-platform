from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import AppSettings
from .models import RunStatus, SerializedRunState, utc_now


class RunRepository:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.settings.runs_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.settings.database_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_node TEXT NOT NULL,
                    active_checkpoint TEXT,
                    latest_score REAL,
                    output_format TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    state_path TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def run_dir(self, run_id: str) -> Path:
        path = self.settings.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "logs").mkdir(exist_ok=True)
        (path / "artifacts").mkdir(exist_ok=True)
        return path

    def save_state(self, state: SerializedRunState) -> SerializedRunState:
        state.updated_at = utc_now()
        run_dir = self.run_dir(state.run_id)
        state_path = run_dir / "state.json"
        manifest_path = run_dir / "run_manifest.json"
        state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        manifest = {
            "run_id": state.run_id,
            "topic": state.topic,
            "status": state.status.value,
            "current_node": state.current_node,
            "active_checkpoint": state.active_checkpoint,
            "artifact_paths": state.artifact_paths,
            "review_cycles": len(state.review_cycles),
            "updated_at": state.updated_at.isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        latest_score = state.review_cycles[-1].composite_score if state.review_cycles else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, topic, status, current_node, active_checkpoint, latest_score,
                    output_format, created_at, updated_at, state_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    topic = excluded.topic,
                    status = excluded.status,
                    current_node = excluded.current_node,
                    active_checkpoint = excluded.active_checkpoint,
                    latest_score = excluded.latest_score,
                    output_format = excluded.output_format,
                    updated_at = excluded.updated_at,
                    state_path = excluded.state_path
                """,
                (
                    state.run_id,
                    state.topic,
                    state.status.value,
                    state.current_node,
                    state.active_checkpoint,
                    latest_score,
                    state.output_format.value,
                    state.created_at.isoformat(),
                    state.updated_at.isoformat(),
                    str(state_path),
                ),
            )
            conn.commit()
        return state

    def load_state(self, run_id: str) -> SerializedRunState:
        state_path = self.run_dir(run_id) / "state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Run '{run_id}' does not exist")
        return SerializedRunState.model_validate_json(state_path.read_text(encoding="utf-8"))

    def write_markdown(self, run_id: str, relative_path: str, content: str) -> str:
        path = self.run_dir(run_id) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    def write_json(self, run_id: str, relative_path: str, payload: Any) -> str:
        path = self.run_dir(run_id) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    def mark_failed(self, run_id: str, error_message: str) -> SerializedRunState:
        state = self.load_state(run_id)
        state.status = RunStatus.FAILED
        state.current_node = "failed"
        state.active_checkpoint = None
        state.errors.append(error_message)
        return self.save_state(state)

