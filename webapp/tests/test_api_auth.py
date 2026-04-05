from __future__ import annotations


def test_protected_route_requires_authentication(client) -> None:
    response = client.post("/api/suggest", json={"turns": [], "top_k": 10})
    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required."


def test_login_rejects_invalid_password(client) -> None:
    response = client.post(
        "/auth/login",
        json={"username": "tester", "password": "wrong-password"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials."


def test_login_allows_access_then_logout_removes_access(client, fake_rpc_client) -> None:
    login_response = client.post(
        "/auth/login",
        json={"username": "tester", "password": "secret123"},
    )
    assert login_response.status_code == 200
    assert login_response.json()["authenticated"] is True

    suggest_response = client.post("/api/suggest", json={"turns": [], "top_k": 10})
    assert suggest_response.status_code == 200
    assert fake_rpc_client.calls

    logout_response = client.post("/auth/logout")
    assert logout_response.status_code == 200
    assert logout_response.json()["authenticated"] is False

    denied_response = client.post("/api/suggest", json={"turns": [], "top_k": 10})
    assert denied_response.status_code == 401
