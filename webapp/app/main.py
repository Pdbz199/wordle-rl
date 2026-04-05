from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.auth import verify_password
from app.rpc_client import RabbitRpcClient, RpcClientError, RpcTimeoutError
from app.schemas import AuthStatus, LoginRequest, SuggestRequest, SuggestResponse
from shared.config import RuntimeSettings


def create_app(
    settings: RuntimeSettings | None = None,
    rpc_client: RabbitRpcClient | None = None,
) -> FastAPI:
    runtime_settings = settings or RuntimeSettings.from_env()
    runtime_rpc_client = rpc_client or RabbitRpcClient(
        rabbitmq_url=runtime_settings.rabbitmq_url,
        request_queue=runtime_settings.rpc_queue,
        timeout_seconds=runtime_settings.rpc_timeout_seconds,
    )

    app = FastAPI(title="Wordle RL Web Assistant", version="1.0.0")
    app.add_middleware(
        SessionMiddleware,
        secret_key=runtime_settings.session_secret,
        session_cookie=runtime_settings.session_cookie_name,
        https_only=runtime_settings.session_cookie_secure,
        same_site="lax",
        max_age=7 * 24 * 60 * 60,
    )

    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    def require_authenticated(request: Request) -> str:
        username = request.session.get("user")
        if username != runtime_settings.app_username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required.",
            )
        return str(username)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/auth/status", response_model=AuthStatus)
    def auth_status(request: Request) -> AuthStatus:
        authenticated = request.session.get("user") == runtime_settings.app_username
        return AuthStatus(
            authenticated=authenticated,
            username=runtime_settings.app_username if authenticated else None,
        )

    @app.post("/auth/login", response_model=AuthStatus)
    def login(payload: LoginRequest, request: Request) -> AuthStatus:
        valid_username = payload.username == runtime_settings.app_username
        valid_password = verify_password(payload.password, runtime_settings.app_password_hash)

        if not (valid_username and valid_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials.",
            )

        request.session["user"] = runtime_settings.app_username
        return AuthStatus(authenticated=True, username=runtime_settings.app_username)

    @app.post("/auth/logout", response_model=AuthStatus)
    def logout(request: Request) -> AuthStatus:
        request.session.clear()
        return AuthStatus(authenticated=False, username=None)

    @app.post("/api/suggest", response_model=SuggestResponse)
    def suggest(payload: SuggestRequest, _username: str = Depends(require_authenticated)) -> SuggestResponse:
        request_id = str(uuid4())
        rpc_payload = {
            "request_id": request_id,
            "turns": [turn.model_dump() for turn in payload.turns],
            "top_k": payload.top_k,
        }

        try:
            rpc_response = runtime_rpc_client.call(rpc_payload)
        except RpcTimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Timed out waiting for the inference worker.",
            ) from exc
        except RpcClientError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Inference service unavailable: {exc}",
            ) from exc

        if "error" in rpc_response:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Inference worker error: {rpc_response['error']}",
            )

        try:
            return SuggestResponse.model_validate(rpc_response)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Inference worker returned an invalid response payload.",
            ) from exc

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> FileResponse:
        if full_path.startswith(("api", "auth", "health", "docs", "openapi", "redoc", "static")):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
        return FileResponse(frontend_dir / "index.html")

    return app
