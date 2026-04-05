from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from env.wordle import parse_pattern_string, pattern_to_string


class TurnIn(BaseModel):
    guess: str
    pattern: str

    @field_validator("guess")
    @classmethod
    def validate_guess(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 5 or not normalized.isalpha():
            raise ValueError("Guess must be exactly 5 alphabetic letters.")
        return normalized

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, value: str) -> str:
        parsed = parse_pattern_string(value.strip().lower())
        return pattern_to_string(parsed)


class SuggestRequest(BaseModel):
    turns: list[TurnIn] = Field(default_factory=list, max_length=6)
    top_k: int = Field(default=10, ge=1, le=25)


class TopChoice(BaseModel):
    word: str
    probability: float


class SuggestResponse(BaseModel):
    request_id: str | None
    turn: int
    is_solved: bool
    candidate_count: int
    next_guess: str | None
    top_choices: list[TopChoice]


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthStatus(BaseModel):
    authenticated: bool
    username: str | None = None
