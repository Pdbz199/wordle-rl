from __future__ import annotations


def _login(client) -> None:
    response = client.post("/auth/login", json={"username": "tester", "password": "secret123"})
    assert response.status_code == 200


def test_invalid_guess_returns_422(client) -> None:
    _login(client)
    response = client.post(
        "/api/suggest",
        json={"turns": [{"guess": "ab12c", "pattern": "bbbbb"}], "top_k": 10},
    )
    assert response.status_code == 422


def test_invalid_pattern_returns_422(client) -> None:
    _login(client)
    response = client.post(
        "/api/suggest",
        json={"turns": [{"guess": "crate", "pattern": "gbzbg"}], "top_k": 10},
    )
    assert response.status_code == 422


def test_too_many_turns_returns_422(client) -> None:
    _login(client)
    turns = [{"guess": "crate", "pattern": "bbbbb"} for _ in range(7)]
    response = client.post("/api/suggest", json={"turns": turns, "top_k": 10})
    assert response.status_code == 422


def test_top_k_out_of_range_returns_422(client) -> None:
    _login(client)
    response = client.post("/api/suggest", json={"turns": [], "top_k": 99})
    assert response.status_code == 422


def test_turn_values_are_normalized_before_rpc(client, fake_rpc_client) -> None:
    _login(client)
    response = client.post(
        "/api/suggest",
        json={"turns": [{"guess": "CRATE", "pattern": "GxByx"}], "top_k": 10},
    )
    assert response.status_code == 200

    sent_turn = fake_rpc_client.calls[-1]["turns"][0]
    assert sent_turn["guess"] == "crate"
    assert sent_turn["pattern"] == "gbbyb"
