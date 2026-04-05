from __future__ import annotations

"""
Automated playback of a trained Wordle agent.

Loads a trained MaskablePPO checkpoint and runs it on a specified number of
Wordle episodes, printing the board state and results after each game. This is
useful for quickly inspecting the agent's behavior and verifying that a trained
model works correctly.

Usage:

    python play.py --model runs/my-run/best_model/best_model.zip --episodes 5
"""

import argparse

from sb3_contrib import MaskablePPO

from env.wordle import WordleEnv


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for automated gameplay playback."""
    parser = argparse.ArgumentParser(description="Play sample games with a trained Wordle PPO model.")
    parser.add_argument("--model", type=str, required=True, help="Path to .zip model file")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically instead of argmax.",
    )
    return parser.parse_args()


def validate_model_compatibility(model: MaskablePPO, env: WordleEnv, model_path: str) -> None:
    """Guard against loading checkpoints with mismatched action-space size."""
    model_actions = getattr(model.action_space, "n", None)
    if model_actions != env.n_words:
        raise ValueError(
            f"Model '{model_path}' is incompatible with the current dictionaries: "
            f"model action size={model_actions}, current guess vocabulary size={env.n_words}. "
            "Please retrain with the current dictionary data."
        )


def play() -> None:
    """
    Run episodes with the trained agent and print terminal board traces.

    Each episode uses a deterministic seed so results are reproducible.
    The agent selects actions using the policy's argmax (deterministic mode)
    unless `--stochastic` is passed.
    """
    args = parse_args()
    env = WordleEnv(data_dir=args.data_dir, max_turns=args.max_turns)
    model = MaskablePPO.load(args.model)
    validate_model_compatibility(model, env, args.model)

    wins = 0
    for episode_idx in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode_idx)
        done = False
        total_reward = 0.0
        final_info = {}
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            total_reward += float(reward)
            final_info = info

        wins += int(final_info.get("is_success", False))
        print(f"\nEpisode {episode_idx + 1}")
        env.render()
        print(
            f"Target: {env.target_word} | Reward: {total_reward:.3f} | "
            f"Solved: {bool(final_info.get('is_success', False))} | Turns: {final_info.get('num_turns', 0)}"
        )

    print(f"\nSolved {wins}/{args.episodes} episodes.")


if __name__ == "__main__":
    play()
