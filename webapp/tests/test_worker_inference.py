from __future__ import annotations

from pathlib import Path

import pytest

from shared.inference import InferenceEngine


@pytest.fixture(scope="module")
def engine() -> InferenceEngine:
    candidate_roots = [
        Path(__file__).resolve().parents[2],  # local repo layout
        Path("/opt/wordle"),                  # docker-compose mounted layout
        Path.cwd(),
    ]

    model_path = None
    data_dir = None
    for root in candidate_roots:
        maybe_model = root / "runs" / "ppo" / "best_model" / "best_model.zip"
        maybe_data = root / "data"
        if maybe_model.exists() and maybe_data.exists():
            model_path = maybe_model
            data_dir = maybe_data
            break

    if model_path is None or data_dir is None:
        pytest.skip("Model checkpoint or data directory not found in expected locations.")

    return InferenceEngine(
        model_path=str(model_path),
        data_dir=str(data_dir),
        max_turns=6,
        top_k_default=10,
        top_k_max=25,
    )


def test_engine_returns_valid_schema(engine: InferenceEngine) -> None:
    response = engine.suggest(turns=[], top_k=5, request_id="unit-1")
    assert response["request_id"] == "unit-1"
    assert response["turn"] == 0
    assert response["is_solved"] is False
    assert isinstance(response["candidate_count"], int)
    assert response["candidate_count"] > 0
    assert response["next_guess"] in engine.env.word_set
    assert 1 <= len(response["top_choices"]) <= 5

    for choice in response["top_choices"]:
        assert choice["word"] in engine.env.word_set
        assert 0.0 <= choice["probability"] <= 1.0


def test_engine_marks_solved_state_without_next_guess(engine: InferenceEngine) -> None:
    solved_word = engine.env.words[0]
    response = engine.suggest(
        turns=[{"guess": solved_word, "pattern": "ggggg"}],
        top_k=5,
        request_id="unit-2",
    )
    assert response["request_id"] == "unit-2"
    assert response["is_solved"] is True
    assert response["turn"] == 1
    assert response["next_guess"] is None
    assert response["top_choices"] == []


def test_engine_rejects_invalid_turn_payload(engine: InferenceEngine) -> None:
    with pytest.raises(ValueError):
        engine.suggest(turns=[{"guess": "ab12c", "pattern": "bbbbb"}], top_k=5, request_id="bad")
