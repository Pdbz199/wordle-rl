from __future__ import annotations

import base64
import os
import shutil
import subprocess
import time
from pathlib import Path

import bcrypt
import pytest
import requests


pytestmark = pytest.mark.integration


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "version"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _wait_for_health(base_url: str, timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("Timed out waiting for /health.")


@pytest.fixture(scope="module")
def compose_stack():
    if os.getenv("RUN_DOCKER_INTEGRATION") != "1":
        pytest.skip("Set RUN_DOCKER_INTEGRATION=1 to run Docker integration tests.")
    if not _docker_available():
        pytest.skip("Docker is not available in PATH.")

    webapp_dir = Path(__file__).resolve().parents[2]
    env_file = webapp_dir / ".env.integration"
    password = "integration-pass"
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    hash_b64 = base64.b64encode(password_hash.encode("utf-8")).decode("utf-8")

    env_file.write_text(
        "\n".join(
            [
                "APP_USERNAME=integration",
                f"APP_PASSWORD_HASH=base64:{hash_b64}",
                "SESSION_SECRET=integration-session-secret",
                "SESSION_COOKIE_NAME=wordle_rl_session",
                "SESSION_COOKIE_SECURE=false",
                "RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/%2F",
                "RPC_QUEUE=wordle_suggest",
                "RPC_TIMEOUT_SECONDS=4",
                "MODEL_PATH=/opt/wordle/runs/ppo/best_model/best_model.zip",
                "DATA_DIR=/opt/wordle/data",
                "MAX_TURNS=6",
                "TOP_K_DEFAULT=10",
                "TOP_K_MAX=25",
            ]
        ),
        encoding="utf-8",
    )

    compose_base = [
        "docker",
        "compose",
        "--env-file",
        env_file.name,
        "-f",
        "docker-compose.yml",
    ]

    try:
        subprocess.run(compose_base + ["up", "-d", "--build"], cwd=webapp_dir, check=True)
        _wait_for_health(base_url="http://127.0.0.1:8080")
        yield {
            "base_url": "http://127.0.0.1:8080",
            "username": "integration",
            "password": password,
            "compose_base": compose_base,
            "cwd": webapp_dir,
        }
    finally:
        subprocess.run(compose_base + ["down", "-v"], cwd=webapp_dir, check=False)
        if env_file.exists():
            env_file.unlink()


def _login(session: requests.Session, base_url: str, username: str, password: str) -> None:
    response = session.post(
        f"{base_url}/auth/login",
        json={"username": username, "password": password},
        timeout=5,
    )
    assert response.status_code == 200


def test_login_and_suggest_flow(compose_stack) -> None:
    session = requests.Session()
    _login(session, compose_stack["base_url"], compose_stack["username"], compose_stack["password"])

    response = session.post(
        f"{compose_stack['base_url']}/api/suggest",
        json={"turns": [], "top_k": 10},
        timeout=8,
    )
    assert response.status_code == 200
    body = response.json()
    assert "next_guess" in body
    assert "top_choices" in body
    assert "candidate_count" in body


def test_invalid_payload_returns_4xx(compose_stack) -> None:
    session = requests.Session()
    _login(session, compose_stack["base_url"], compose_stack["username"], compose_stack["password"])

    response = session.post(
        f"{compose_stack['base_url']}/api/suggest",
        json={"turns": [{"guess": "abc12", "pattern": "bbbbb"}], "top_k": 10},
        timeout=8,
    )
    assert response.status_code == 422


def test_worker_timeout_path_returns_504(compose_stack) -> None:
    session = requests.Session()
    _login(session, compose_stack["base_url"], compose_stack["username"], compose_stack["password"])

    subprocess.run(compose_stack["compose_base"] + ["stop", "worker"], cwd=compose_stack["cwd"], check=True)
    try:
        response = session.post(
            f"{compose_stack['base_url']}/api/suggest",
            json={"turns": [], "top_k": 10},
            timeout=10,
        )
        assert response.status_code == 504
    finally:
        subprocess.run(compose_stack["compose_base"] + ["start", "worker"], cwd=compose_stack["cwd"], check=True)
