"""Local API: sessions, episode artifacts, bounded jobs, and replay.

Run on 127.0.0.1 only. There is deliberately no hardware execution endpoint.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import io
import json
from pathlib import Path
import threading
from typing import Annotated, Literal
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import pipeline
from .jobs import Jobs
from .storage import Store, MAX_ARCHIVE_BYTES, decode_npz

Seed = Annotated[int, Field(ge=0, le=2**31-1)]
ArtifactId = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EnvironmentConfig(StrictModel):
    friction: float = Field(default=1.0, ge=0.1, le=2)
    delay_steps: int = Field(default=0, ge=0, le=20)
    horizon: int = Field(default=600, ge=1, le=2000)
    object_position: list[float] | None = Field(default=None, min_length=2, max_length=2)
    object_position_bounds: list[list[float]] | None = Field(default=None, min_length=2, max_length=2)

    def settings(self):
        return self.model_dump(exclude_none=True)


class RolloutRequest(StrictModel):
    seeds: list[Seed] = Field(default_factory=lambda: list(range(5)), min_length=1, max_length=100)
    config: EnvironmentConfig = Field(default_factory=EnvironmentConfig)


class EvaluationRequest(RolloutRequest):
    seeds: list[Seed] = Field(default_factory=lambda: list(range(10000, 10005)), min_length=1, max_length=100)
    model_id: ArtifactId | None = None


class BenchmarkRequest(StrictModel):
    seeds: list[Seed] = Field(default_factory=lambda: list(range(10000, 10005)), min_length=1, max_length=100)
    conditions: list[EnvironmentConfig] = Field(min_length=1, max_length=12)
    model_ids: list[ArtifactId] = Field(default_factory=list, max_length=5)


class TrainingConfig(StrictModel):
    model_type: Literal["baseline_mlp", "act"] = "baseline_mlp"
    epochs: int = Field(default=20, ge=1, le=2000)
    batch_size: int = Field(default=64, ge=1, le=1024)
    chunk_size: int = Field(default=8, ge=1, le=128)
    hidden_dim: int = Field(default=64, ge=16, le=512, multiple_of=4)
    seed: int = Field(default=0, ge=0, le=2**31-1)
    val_fraction: float = Field(default=.2, ge=.01, le=.5)
    lr: float = Field(default=.001, ge=1e-7, le=.1)
    kl_weight: float = Field(default=.01, ge=0, le=100)
    device: Literal["cpu", "cuda"] = "cpu"


class TrainingRequest(StrictModel):
    episode_ids: list[ArtifactId] = Field(min_length=2, max_length=1000)
    config: TrainingConfig = Field(default_factory=TrainingConfig)


class SessionRequest(StrictModel):
    seed: int = Field(default=0, ge=0, le=2**31-1)
    config: EnvironmentConfig = Field(default_factory=EnvironmentConfig)


class StepRequest(StrictModel):
    action: list[float] | None = Field(default=None, min_length=1, max_length=128)
    expert: bool = False
    repeat: int = Field(default=1, ge=1, le=50)


def create_app(data_root: Path | str = "data") -> FastAPI:
    store = Store(data_root)
    sessions = {}
    session_lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(app):
        import torch
        torch.set_num_threads(1)
        app.state.jobs = Jobs(store)
        try:
            yield
        finally:
            app.state.jobs.close()
            for session in sessions.values():
                session["env"].close()
            sessions.clear()

    app = FastAPI(title="XLeRobot simulation workbench", version="0.1.0", lifespan=lifespan,
                  description="Simulation only. State-based learning, not camera-based ACT. See /docs for the interactive API.")
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])

    @app.middleware("http")
    async def local_origin(request, call_next):
        # Disallow browser cross-origin commands against a local unauthenticated service.
        origin = request.headers.get("origin")
        if origin:
            from urllib.parse import urlsplit
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc != request.headers.get("host"):
                return JSONResponse(status_code=403, content={"detail": "Cross-origin requests are disabled"})
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def bad_input(request, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def missing(request, exc):
        return JSONResponse(status_code=404, content={"detail": "Artifact not found"})

    @app.get("/")
    def index():
        return {"name": "XLeRobot simulation workbench", "interactive_api": "/docs",
                "hardware_enabled": False, "workflow": ["sessions or collection", "episodes", "train", "evaluate", "replay"]}

    @app.get("/health")
    def health():
        return {"status": "ok", "hardware_enabled": False, "cloud_enabled": False}

    @app.get("/environment")
    def environment():
        from .sim import ManipulationEnv
        env = ManipulationEnv()
        try:
            return {"metadata": env.metadata(), "defaults": env.config}
        finally:
            env.close()

    @app.get("/episodes")
    def episodes():
        return store.listing("episodes")

    @app.get("/episodes/{identifier}")
    def episode(identifier: str):
        return store.read("episodes", identifier)

    @app.get("/episodes/{identifier}/arrays")
    def arrays(identifier: str):
        values, metadata = store.load_episode(identifier)
        return {name: value.tolist() for name, value in values.items()}

    @app.get("/episodes/{identifier}/download")
    def download(identifier: str):
        store.load_episode(identifier)  # Verify checksum before export.
        return FileResponse(store.path("episodes", identifier) / "episode.npz",
                            filename=f"episode-{identifier}.npz", media_type="application/octet-stream")

    @app.post("/episodes/import")
    async def import_episode(request: Request, seed: int = 0, config_json: str = "{}"):
        """Upload raw NPZ bytes; accept only data reproducible by this simulator.

        Pass the source seed and JSON environment config as query parameters.
        No file path is accepted. Full manifest export is GET /episodes/{id}.
        """
        from .sim import ManipulationEnv
        if len(config_json) > 4096:
            raise ValueError("Environment config is too large")
        chunks, count = [], 0
        async for chunk in request.stream():
            count += len(chunk)
            if count > MAX_ARCHIVE_BYTES:
                raise HTTPException(413, "Episode archive exceeds 32 MiB")
            chunks.append(chunk)
        payload = b"".join(chunks)
        config = EnvironmentConfig.model_validate(json.loads(config_json)).settings()
        # CPU work goes off the event loop. No import may assert its own success.
        def verify_and_save():
            values = decode_npz(payload)
            env = ManipulationEnv(config)
            try:
                if values["actions"].shape[1] != env.metadata()["action_dim"]:
                    raise ValueError("Imported action dimension does not match the robot")
                if values["observations"].shape[1] != env.metadata()["observation_dim"]:
                    raise ValueError("Imported observation dimension does not match the robot")
                obs = env.reset(seed=seed)
                if not np.allclose(obs, values["observations"][0], atol=1e-5, rtol=0):
                    raise ValueError("Imported initial observation does not match seed/config")
                info = {}
                for index, action in enumerate(values["actions"]):
                    obs, reward, terminal, info = env.step(action)
                    if (not np.allclose(obs, values["observations"][index + 1], atol=1e-5, rtol=0)
                            or not np.isclose(reward, values["rewards"][index], atol=1e-5)
                            or bool(terminal) != bool(values["terminated"][index])):
                        raise ValueError(f"Imported transition {index} does not match simulation")
                    if terminal and index != len(values["actions"]) - 1:
                        raise ValueError("Imported episode continues beyond termination")
                return store.save_episode(values, {
                    "source": "imported_simulation", "seed": seed, "config": env.config,
                    "simulation": env.metadata(), "contract_sha256": pipeline.contract(env.metadata()),
                    "success": bool(info.get("success", False)), "outcome": info, "hardware": False,
                })
            finally:
                env.close()
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(verify_and_save)

    @app.get("/episodes/{identifier}/replay")
    def replay(identifier: str):
        return pipeline.replay(store, identifier)

    @app.get("/episodes/{identifier}/frame.png")
    async def frame(identifier: str):
        # GLFW context creation stays on uvicorn's main/event-loop thread.
        # ponytail: bounded replay blocks this local API; use a render process if interactive latency matters.
        return Response(pipeline.replay(store, identifier, image=True), media_type="image/png")

    @app.post("/jobs/collect", status_code=202)
    def collect(request: RolloutRequest):
        return app.state.jobs.submit("collect", pipeline.collect, seeds=request.seeds, config=request.config.settings())

    @app.post("/jobs/train", status_code=202)
    def train(request: TrainingRequest):
        return app.state.jobs.submit("train", pipeline.train_model,
                                     episode_ids=request.episode_ids, config=request.config.model_dump())

    @app.post("/jobs/evaluate", status_code=202)
    def evaluate(request: EvaluationRequest):
        return app.state.jobs.submit("evaluate", pipeline.evaluate, seeds=request.seeds,
                                     config=request.config.settings(), model_id=request.model_id)

    @app.get("/jobs")
    def jobs():
        return app.state.jobs.listing()

    @app.post("/jobs/benchmark", status_code=202)
    def benchmark(request: BenchmarkRequest):
        return app.state.jobs.submit("benchmark", pipeline.benchmark, seeds=request.seeds,
                                     conditions=[item.settings() for item in request.conditions], model_ids=request.model_ids)

    @app.get("/jobs/{identifier}")
    def job(identifier: str):
        return app.state.jobs.read(identifier)

    @app.post("/jobs/{identifier}/cancel")
    def cancel(identifier: str):
        return app.state.jobs.cancel(identifier)

    @app.get("/models")
    def models():
        return store.listing("models")

    @app.get("/models/{identifier}")
    def model(identifier: str):
        path = store.path("models", identifier)
        return {"provenance": json.loads((path / "pipeline.json").read_text("utf-8")),
                "training": json.loads((path / "metadata.json").read_text("utf-8"))}

    @app.get("/evaluations")
    def evaluations():
        return store.listing("evaluations")

    @app.get("/evaluations/{identifier}")
    def evaluation(identifier: str):
        return store.read("evaluations", identifier)

    def session(identifier):
        if identifier not in sessions:
            raise HTTPException(404, "Session not found; live sessions are not restored after restart")
        return sessions[identifier]

    @app.post("/sessions", status_code=201)
    def create_session(request: SessionRequest):
        from .sim import ManipulationEnv
        with session_lock:
            if len(sessions) >= 4:
                raise ValueError("Close an existing session (maximum 4)")
            env = ManipulationEnv(request.config.settings())
            identifier = uuid.uuid4().hex
            observation = env.reset(seed=request.seed)
            sessions[identifier] = {"env": env, "seed": request.seed,
                                    "observations": [observation.copy()], "actions": [],
                                    "rewards": [], "terminated": [], "outcome": {}, "source": "manual_simulation"}
            return {"id": identifier, "observation": observation.tolist(), "metadata": env.metadata()}

    @app.post("/sessions/{identifier}/step")
    def step(identifier: str, request: StepRequest):
        if request.expert == (request.action is not None):
            raise ValueError("Provide exactly one: action or expert=true")
        with session_lock:
            item = session(identifier)
            if item["terminated"] and item["terminated"][-1]:
                raise ValueError("Session terminated; save/close and start a new session")
            env = item["env"]
            if request.expert:
                item["source"] = "assisted_simulation"
            for _ in range(request.repeat):
                action = np.asarray(env.expert_action() if request.expert else request.action, dtype=np.float32)
                obs, reward, terminated, info = env.step(action)
                item["observations"].append(obs.copy())
                item["actions"].append(action.copy())
                item["rewards"].append(reward)
                item["terminated"].append(terminated)
                item["outcome"] = info
                if terminated:
                    break
            return {"observation": obs.tolist(), "reward": reward, "terminated": terminated,
                    "info": info, "steps": len(item["actions"])}

    @app.get("/sessions/{identifier}/frame.png")
    async def session_frame(identifier: str):
        with session_lock:
            env = session(identifier)["env"]
            stream = io.BytesIO()
            Image.fromarray(env.render()).save(stream, format="PNG")
            return Response(stream.getvalue(), media_type="image/png")

    @app.post("/sessions/{identifier}/save")
    def save_session(identifier: str):
        with session_lock:
            item = session(identifier)
            if not item["actions"]:
                raise ValueError("Record at least one transition before saving")
            env = item["env"]
            return store.save_episode({name: np.asarray(item[name]) for name in
                                       ("observations", "actions", "rewards", "terminated")}, {
                "source": item["source"], "seed": item["seed"], "config": env.config,
                "simulation": env.metadata(), "contract_sha256": pipeline.contract(env.metadata()),
                "success": bool(item["outcome"].get("success", False)), "outcome": item["outcome"], "hardware": False,
            })

    @app.delete("/sessions/{identifier}")
    def close_session(identifier: str):
        with session_lock:
            session(identifier)["env"].close()
            del sessions[identifier]
            return {"closed": identifier}

    return app
