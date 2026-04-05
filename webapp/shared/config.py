from __future__ import annotations

import base64
import os
from dataclasses import dataclass


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _normalize_password_hash(value: str) -> str:
    if value.startswith("base64:"):
        encoded = value[len("base64:") :]
        try:
            return base64.b64decode(encoded.encode("utf-8")).decode("utf-8")
        except Exception as exc:
            raise RuntimeError("APP_PASSWORD_HASH has invalid base64 payload.") from exc
    return value


def _optional_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean value for {name}: '{raw}'")


@dataclass(frozen=True)
class RuntimeSettings:
    app_username: str
    app_password_hash: str
    session_secret: str
    session_cookie_name: str
    session_cookie_secure: bool
    rabbitmq_url: str
    rpc_queue: str
    rpc_timeout_seconds: float
    model_path: str
    data_dir: str
    max_turns: int
    top_k_default: int
    top_k_max: int

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        top_k_default = int(os.getenv("TOP_K_DEFAULT", "10"))
        top_k_max = int(os.getenv("TOP_K_MAX", "25"))
        if top_k_default < 1:
            raise RuntimeError("TOP_K_DEFAULT must be at least 1.")
        if top_k_max < top_k_default:
            raise RuntimeError("TOP_K_MAX must be greater than or equal to TOP_K_DEFAULT.")

        return cls(
            app_username=_required_env("APP_USERNAME"),
            app_password_hash=_normalize_password_hash(_required_env("APP_PASSWORD_HASH")),
            session_secret=_required_env("SESSION_SECRET"),
            session_cookie_name=os.getenv("SESSION_COOKIE_NAME", "wordle_rl_session"),
            session_cookie_secure=_optional_bool("SESSION_COOKIE_SECURE", True),
            rabbitmq_url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/%2F"),
            rpc_queue=os.getenv("RPC_QUEUE", "wordle_suggest"),
            rpc_timeout_seconds=float(os.getenv("RPC_TIMEOUT_SECONDS", "8")),
            model_path=os.getenv("MODEL_PATH", "/opt/wordle/runs/ppo/best_model/best_model.zip"),
            data_dir=os.getenv("DATA_DIR", "/opt/wordle/data"),
            max_turns=int(os.getenv("MAX_TURNS", "6")),
            top_k_default=top_k_default,
            top_k_max=top_k_max,
        )
