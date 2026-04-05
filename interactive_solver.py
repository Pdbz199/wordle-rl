from __future__ import annotations

"""
Interactive Wordle assistant powered by a trained PPO agent.

This script lets you use the trained agent as a helper while playing Wordle on
your phone or in a browser. Each turn, the agent suggests a word based on its
learned policy. You enter the word you actually guessed and the feedback pattern
you received (e.g., `gybbg` for green-yellow-gray-gray-green), and the agent
updates its internal state and suggests the next guess.

The agent uses the same `WordleEnv` in "assistant mode" (no hidden target word).
Instead of auto-scoring guesses, you provide the feedback manually via the
`apply_feedback()` method.

Usage:

    python interactive_solver.py \
        --model runs/my-run/best_model/best_model.zip \
        --show-candidates
"""

import argparse

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from env.wordle import WordleEnv, parse_pattern_string


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for interactive solving mode."""
    parser = argparse.ArgumentParser(description="Interactive Wordle helper powered by a trained PPO agent.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .zip model")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--show-candidates", action="store_true")
    parser.add_argument("--candidate-limit", type=int, default=20)
    return parser.parse_args()


def prompt_pattern() -> str:
    """Prompt until valid Wordle feedback pattern is provided."""
    while True:
        try:
            raw = input("Feedback pattern (g=green, y=yellow, b=gray): ").strip().lower()
        except EOFError:
            return "bbbbb"
        try:
            parse_pattern_string(raw)
            return raw
        except ValueError as exc:
            print(f"Invalid pattern: {exc}")


def get_model_ranked_choices(
    model: MaskablePPO,
    env: WordleEnv,
    obs: np.ndarray,
    action_mask: np.ndarray,
    top_k: int,
) -> list[tuple[str, float]]:
    """
    Rank valid actions by the policy's probability distribution.

    Extracts the full action distribution from the policy network, applies the
    action mask, and returns the top-k words sorted by descending probability.
    This shows what the agent "wants" to guess and with what confidence.
    """
    if top_k <= 0:
        return []

    valid_indices = np.flatnonzero(action_mask)
    if valid_indices.size == 0:
        return []

    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    batched_mask = np.asarray([action_mask], dtype=bool)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor, action_masks=batched_mask)
        probs = dist.distribution.probs[0].detach().cpu().numpy()

    ranked_indices = valid_indices[np.argsort(probs[valid_indices])[::-1]]
    limited = ranked_indices[: min(top_k, ranked_indices.size)]
    return [(env.words[int(idx)], float(probs[int(idx)])) for idx in limited]


def format_probability(prob: float) -> str:
    """Format probability with enough precision for small policy masses."""
    pct = prob * 100.0
    if pct >= 1.0:
        return f"{pct:.1f}%"
    if pct >= 0.1:
        return f"{pct:.2f}%"
    if pct >= 0.01:
        return f"{pct:.3f}%"
    return f"{pct:.4f}%"


def validate_model_compatibility(model: MaskablePPO, env: WordleEnv, model_path: str) -> None:
    """Guard against loading checkpoints with mismatched action-space size."""
    model_actions = getattr(model.action_space, "n", None)
    if model_actions != env.n_words:
        raise ValueError(
            f"Model '{model_path}' is incompatible with the current dictionaries: "
            f"model action size={model_actions}, current guess vocabulary size={env.n_words}. "
            "Please retrain with the current dictionary data."
        )


def run_session(model: MaskablePPO, env: WordleEnv, show_candidates: bool, candidate_limit: int) -> None:
    """Run one interactive puzzle session until solved/quit/turn-limit."""
    env.reset_assistant()
    print("\nNew puzzle started.")
    print("Type guesses into your phone app, then enter the resulting pattern here.")
    print("Pattern example: gybbg")

    for turn in range(1, env.max_turns + 1):
        obs = env.get_obs()
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        suggestion = env.words[int(action)]
        print(f"\nTurn {turn} suggestion: {suggestion}")
        print(f"Current candidate count: {int(env.candidate_indices.size)}")

        try:
            guess_used = input("Guess used (Enter = suggested, 'quit' to exit): ").strip().lower()
        except EOFError:
            print("Input closed, ending interactive session.")
            return
        if guess_used == "quit":
            print("Ending interactive session.")
            return
        if guess_used == "":
            guess_used = suggestion

        pattern = prompt_pattern()
        result = env.apply_feedback(guess_word=guess_used, pattern=pattern)

        if result["solved"]:
            print(f"Solved in {result['turn']} turns.")
            return

        if show_candidates:
            next_obs = env.get_obs()
            next_mask = env.action_masks()
            ranked_choices = get_model_ranked_choices(
                model=model,
                env=env,
                obs=next_obs,
                action_mask=next_mask,
                top_k=candidate_limit,
            )
            rendered = ", ".join(f"{word} ({format_probability(prob)})" for word, prob in ranked_choices)
            print(f"Top {len(ranked_choices)} model choices: {rendered}")
        else:
            print(f"Candidates remaining: {result['candidate_count']}")

    print("Reached 6 turns without a full solve pattern.")


def main() -> None:
    """Program entrypoint."""
    args = parse_args()
    model = MaskablePPO.load(args.model)
    env = WordleEnv(data_dir=args.data_dir, max_turns=args.max_turns)
    validate_model_compatibility(model, env, args.model)

    while True:
        run_session(
            model=model,
            env=env,
            show_candidates=args.show_candidates,
            candidate_limit=args.candidate_limit,
        )
        try:
            again = input("\nStart another puzzle? (y/n): ").strip().lower()
        except EOFError:
            break
        if again not in {"y", "yes"}:
            break


if __name__ == "__main__":
    main()
