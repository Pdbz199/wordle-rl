from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from env.wordle import WordleEnv, parse_pattern_string, pattern_to_string


def _validate_model_compatibility(model: MaskablePPO, env: WordleEnv, model_path: str) -> None:
    model_actions = getattr(model.action_space, "n", None)
    if model_actions != env.n_words:
        raise ValueError(
            f"Model '{model_path}' is incompatible with the current dictionary: "
            f"model action size={model_actions}, dictionary size={env.n_words}."
        )


@dataclass(frozen=True)
class NormalizedTurn:
    guess: str
    pattern: str


class InferenceEngine:
    """Reusable inference runner for Wordle assistant suggestions."""

    def __init__(
        self,
        model_path: str,
        data_dir: str,
        max_turns: int = 6,
        top_k_default: int = 10,
        top_k_max: int = 25,
    ) -> None:
        self.model_path = model_path
        self.model = MaskablePPO.load(model_path)
        self.env = WordleEnv(data_dir=data_dir, max_turns=max_turns)
        _validate_model_compatibility(self.model, self.env, model_path)
        self.top_k_default = max(1, int(top_k_default))
        self.top_k_max = max(self.top_k_default, int(top_k_max))

    def suggest(
        self,
        turns: Sequence[Mapping[str, Any]],
        top_k: int | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_turns = self._normalize_turns(turns)

        self.env.reset_assistant()
        for turn in normalized_turns:
            self.env.apply_feedback(guess_word=turn.guess, pattern=turn.pattern)

        solved = bool(normalized_turns and normalized_turns[-1].pattern == "ggggg")
        candidate_count = int(self.env.candidate_indices.size)

        next_guess: str | None = None
        top_choices: list[dict[str, Any]] = []

        if not solved and self.env.turn < self.env.max_turns:
            obs = self.env.get_obs()
            action_mask = self.env.action_masks()
            action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
            next_guess = self.env.words[int(action)]
            top_choices = self._rank_top_choices(obs=obs, action_mask=action_mask, top_k=top_k)

        return {
            "request_id": request_id,
            "turn": int(self.env.turn),
            "is_solved": solved,
            "candidate_count": candidate_count,
            "next_guess": next_guess,
            "top_choices": top_choices,
        }

    def _normalize_turns(self, turns: Sequence[Mapping[str, Any]]) -> list[NormalizedTurn]:
        if len(turns) > self.env.max_turns:
            raise ValueError(f"At most {self.env.max_turns} turns are allowed.")

        normalized: list[NormalizedTurn] = []
        for index, item in enumerate(turns):
            if not isinstance(item, Mapping):
                raise ValueError(f"Turn {index + 1} must be an object with guess/pattern.")

            raw_guess = item.get("guess")
            raw_pattern = item.get("pattern")
            if not isinstance(raw_guess, str):
                raise ValueError(f"Turn {index + 1} guess must be a string.")
            if not isinstance(raw_pattern, str):
                raise ValueError(f"Turn {index + 1} pattern must be a string.")

            guess = raw_guess.strip().lower()
            if len(guess) != 5 or not guess.isalpha():
                raise ValueError(f"Turn {index + 1} guess must be exactly 5 alphabetic letters.")

            pattern_values = parse_pattern_string(raw_pattern.strip().lower())
            pattern = pattern_to_string(pattern_values)
            normalized.append(NormalizedTurn(guess=guess, pattern=pattern))

        return normalized

    def _rank_top_choices(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        top_k: int | None,
    ) -> list[dict[str, Any]]:
        limit = self._normalize_top_k(top_k)
        if limit <= 0:
            return []

        valid_indices = np.flatnonzero(action_mask)
        if valid_indices.size == 0:
            return []

        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        batched_mask = np.asarray([action_mask], dtype=bool)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor, action_masks=batched_mask)
            probabilities = distribution.distribution.probs[0].detach().cpu().numpy()

        ranked = valid_indices[np.argsort(probabilities[valid_indices])[::-1]]
        selected = ranked[: min(limit, ranked.size)]

        return [
            {
                "word": self.env.words[int(index)],
                "probability": float(probabilities[int(index)]),
            }
            for index in selected
        ]

    def _normalize_top_k(self, top_k: int | None) -> int:
        if top_k is None:
            return self.top_k_default
        return max(1, min(int(top_k), self.top_k_max))
