from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .models import CheckpointSubmission, RunCreateRequest, RunSummary, SerializedRunState, StageRestartRequest
from .service import RunService, build_default_service


def create_app(service: RunService | None = None) -> FastAPI:
    app = FastAPI(title="PIF Research Automation Prototype", version="0.1.0")
    run_service = service or build_default_service()

    @app.post("/runs", response_model=RunSummary)
    def create_run(request: RunCreateRequest) -> RunSummary:
        try:
            return run_service.create_run(request)
        except Exception as exc:  # pragma: no cover - framework serialization
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/runs/{run_id}", response_model=RunSummary)
    def get_run(run_id: str) -> RunSummary:
        try:
            return run_service.get_run(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/detail", response_model=SerializedRunState)
    def get_run_detail(run_id: str) -> SerializedRunState:
        try:
            return run_service.get_run_detail(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/artifacts/{artifact_key}")
    def download_artifact(run_id: str, artifact_key: str) -> FileResponse:
        try:
            state = run_service.get_run_detail(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        artifact_path = state.artifact_paths.get(artifact_key)
        if not artifact_path:
            raise HTTPException(status_code=404, detail=f"Artifact '{artifact_key}' not found for run '{run_id}'")
        path = Path(artifact_path).resolve()
        run_dir = run_service.repo.run_dir(run_id).resolve()
        try:
            path.relative_to(run_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Artifact path is outside the run directory") from exc
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail=f"Artifact file missing for key '{artifact_key}'")
        return FileResponse(path, filename=path.name)

    @app.post("/runs/{run_id}/checkpoint", response_model=RunSummary)
    def submit_checkpoint(run_id: str, submission: CheckpointSubmission) -> RunSummary:
        try:
            return run_service.submit_checkpoint(run_id, submission)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/restart-stage", response_model=RunSummary)
    def restart_stage(run_id: str, request: StageRestartRequest) -> RunSummary:
        try:
            return run_service.restart_from_stage(run_id, request.stage_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
