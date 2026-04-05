# Wordle RL Web Assistant

This directory contains a password-protected Wordle helper web app for your RL model.

## What It Is

The web app lets you play real Wordle while the model suggests the next guess.

- You enter your guess in a Wordle-style board.
- You set feedback colors (gray/yellow/green) on the active row.
- The backend sends your current turns to a worker.
- The worker runs inference on the trained checkpoint and returns:
  - next best guess
  - ranked alternative guesses
  - remaining candidate count

The UI stores puzzle progress in browser localStorage. No database is required.

## How It Works

The stack uses Docker Compose with three services:

- `api` (FastAPI): serves UI, handles login/session cookie auth, calls worker via RabbitMQ RPC.
- `worker` (Python): loads one model and computes suggestions from turn history.
- `rabbitmq`: queues RPC requests from API to worker.

No backend API changes are required for UI updates.

## Local Run

From this directory (`webapp/`):

1. Copy env template:

```powershell
Copy-Item .env.example .env
```

2. Generate password hash (Docker method, no local installs needed):

```powershell
docker compose run --rm --no-deps api python scripts/generate_password_hash.py
```

3. Put the printed `base64:...` value into `.env` as `APP_PASSWORD_HASH`.

4. Set these values in `.env`:

- `APP_USERNAME`
- `APP_PASSWORD_HASH`
- `SESSION_SECRET` (long random secret)

5. Start the stack:

```powershell
docker compose up -d --build
```

6. Open:

- `http://127.0.0.1:8080`

7. Log in and play:

- Type letters from your physical keyboard or click the on-screen keyboard.
- Click tiles in the active row to cycle colors.
- Submit turn.
- Repeat using next suggestion/top choices.

## Configuration Guide

Important env vars in `.env`:

- `APP_USERNAME`: shared username for the app.
- `APP_PASSWORD_HASH`: bcrypt hash in `base64:` form from `scripts/generate_password_hash.py`.
- `SESSION_SECRET`: session signing key.
- `SESSION_COOKIE_SECURE`: `true` for HTTPS deployments, `false` for plain local HTTP.
- `MODEL_PATH`: default `/opt/wordle/runs/ppo/best_model/best_model.zip`.
- `DATA_DIR`: default `/opt/wordle/data`.
- `RABBITMQ_URL`: default works for this compose network.
- `RPC_QUEUE`: queue name used by API/worker.
- `RPC_TIMEOUT_SECONDS`: API timeout waiting for worker response.

## Remote Deployment (Friends/Family)

Recommended setup: Docker on your host + Nginx Proxy Manager (NPM) for public HTTPS.

1. Keep Compose port binding local-only (`127.0.0.1:8080:8000`) as defined.
2. Point NPM proxy host to:

- Forward Host/IP: host running Docker
- Forward Port: `8080`
- Scheme: `http`

3. Enable SSL in NPM for your domain.
4. Set `SESSION_COOKIE_SECURE=true` in `.env`.
5. Restart the stack:

```powershell
docker compose up -d
```

6. Optionally add NPM access-list auth for a second protection layer.

## Testing

Run non-integration tests in Docker:

```powershell
docker compose run --rm --no-deps api sh -lc "pip install --no-cache-dir -r requirements-dev.txt && pytest tests -m 'not integration'"
```

Run integration tests (optional, slower):

```powershell
$env:RUN_DOCKER_INTEGRATION = "1"
pytest tests/integration -m integration
```

## Troubleshooting

- Login always fails:
  - Verify `APP_PASSWORD_HASH` uses the `base64:` format produced by the script.
  - Verify `APP_USERNAME` matches what you enter.
- API returns `503` or `504`:
  - Check worker/RabbitMQ status:
  - `docker compose ps`
  - `docker compose logs worker --tail 100`
  - `docker compose logs rabbitmq --tail 100`
- Model load error:
  - Confirm model file exists at `MODEL_PATH`.
  - Confirm dictionary/model action-space compatibility.
- Session/cookie issues behind domain:
  - Use HTTPS in proxy and set `SESSION_COOKIE_SECURE=true`.
