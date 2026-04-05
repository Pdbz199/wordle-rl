from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
import sys

import bcrypt
import pytest
from fastapi.testclient import TestClient

WEBAPP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WEBAPP_ROOT.parent
if str(WEBAPP_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBAPP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.main import create_app
from shared.config import RuntimeSettings


class FakeRpcClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def call(self, payload: dict) -> dict:
        self.calls.append(payload)
        turns = payload.get("turns", [])
        solved = bool(turns and turns[-1]["pattern"] == "ggggg")
        return {
            "request_id": payload.get("request_id"),
            "turn": len(turns),
            "is_solved": solved,
            "candidate_count": 14855 - (len(turns) * 500),
            "next_guess": None if solved else "slate",
            "top_choices": [] if solved else [{"word": "slate", "probability": 0.42}],
        }


@pytest.fixture
def runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        app_username="tester",
        app_password_hash=bcrypt.hashpw(b"secret123", bcrypt.gensalt()).decode("utf-8"),
        session_secret="unit-test-session-secret",
        session_cookie_name="test-session",
        session_cookie_secure=False,
        rabbitmq_url="amqp://guest:guest@localhost:5672/%2F",
        rpc_queue="wordle_suggest",
        rpc_timeout_seconds=5.0,
        model_path="/tmp/model.zip",
        data_dir="/tmp/data",
        max_turns=6,
        top_k_default=10,
        top_k_max=25,
    )


@pytest.fixture
def fake_rpc_client() -> FakeRpcClient:
    return FakeRpcClient()


@pytest.fixture
def client(runtime_settings: RuntimeSettings, fake_rpc_client: FakeRpcClient) -> Generator[TestClient, None, None]:
    app = create_app(settings=runtime_settings, rpc_client=fake_rpc_client)
    with TestClient(app) as test_client:
        yield test_client
