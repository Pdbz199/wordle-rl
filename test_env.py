from __future__ import annotations

"""
Smoke test for the Wordle environment.

Runs a series of episodes using a random (but valid) action policy to verify
that the environment's core mechanics work correctly: reset, step, action
masking, reward computation, and episode termination.

A random policy will almost never solve Wordle (the dictionary has ~15K words),
so the expected win rate is near zero. This script is for verifying correctness,
not performance.

Usage:

    python test_env.py --episodes 10 --seed 123
"""

import argparse

import numpy as np

from env.wordle import WordleEnv


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for environment smoke testing."""
    parser = argparse.ArgumentParser(description="Smoke test the Wordle environment.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data-dir", type=str, default="data")
    return parser.parse_args()


def test_environment() -> None:
    """Run random masked policy episodes and print summary lines."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = WordleEnv(data_dir=args.data_dir)
    wins = 0

    for episode_idx in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode_idx)
        done = False
        total_reward = 0.0
        final_info = {}

        while not done:
            mask = env.action_masks()
            valid_actions = np.flatnonzero(mask)
            if valid_actions.size == 0:
                raise RuntimeError("Action mask returned no valid actions.")
            action = int(rng.choice(valid_actions))
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            final_info = info

        wins += int(final_info.get("is_success", False))
        print(
            f"Episode {episode_idx + 1:02d}: solved={bool(final_info.get('is_success', False))} "
            f"turns={final_info.get('num_turns', 0)} reward={total_reward:.3f} "
            f"target={env.target_word}"
        )

    print(f"\nRandom policy solved {wins}/{args.episodes} episodes.")


if __name__ == "__main__":
    test_environment()
