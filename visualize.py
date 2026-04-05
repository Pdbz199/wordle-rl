from __future__ import annotations

"""
Render trained agent gameplay as an animated GIF.

Runs the agent on a set of Wordle episodes and renders each game state as a
Wordle-style board image (colored grid with letters). The frames are assembled
into an animated GIF for sharing or qualitative inspection of the agent's
strategy.

The color scheme matches the official Wordle game:
- Dark gray for empty/absent tiles, yellow for present, green for correct.

Usage:

    python visualize.py \
        --model runs/my-run/best_model/best_model.zip \
        --output runs/my-run/example_games.gif \
        --episodes 6
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sb3_contrib import MaskablePPO

from env.wordle import WordleEnv

# Wordle-style color palette (RGB tuples)
BG_COLOR = (16, 18, 20)         # Dark background
GRID_EMPTY = (58, 61, 64)       # Unfilled cell (no guess yet)
GRID_ABSENT = (95, 99, 103)     # Gray. Letter not in target
GRID_PRESENT = (201, 180, 88)   # Yellow. Letter in target, wrong position
GRID_CORRECT = (106, 170, 100)  # Green. Letter in correct position
TEXT_COLOR = (245, 245, 245)    # White letter text

# Map pattern characters to tile colors
GRID_COLORS = {".": GRID_EMPTY, "b": GRID_ABSENT, "y": GRID_PRESENT, "g": GRID_CORRECT}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for GIF rendering."""
    parser = argparse.ArgumentParser(description="Render trained Wordle PPO gameplay to GIF.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .zip model")
    parser.add_argument("--output", type=str, default="runs/example_games.gif")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fps", type=float, default=1.4)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-turns", type=int, default=6)
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


def draw_board_frame(snapshot: dict, episode_idx: int) -> np.ndarray:
    """Draw one board frame as an RGB numpy image."""
    cell = 64
    gap = 8
    top = 90
    left = 30
    width = left * 2 + (cell * 5) + (gap * 4)
    height = top + (cell * 6) + (gap * 5) + 90

    image = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    guesses: list[str] = snapshot["guesses"]
    patterns: list[str] = snapshot["patterns"]
    candidate_count: int = snapshot["candidate_count"]
    target_word = snapshot["target_word"] or "unknown"

    draw.text((left, 20), f"Episode {episode_idx + 1}", fill=TEXT_COLOR, font=font)
    draw.text((left, 40), f"Candidates: {candidate_count}", fill=TEXT_COLOR, font=font)
    draw.text((left, 60), f"Target: {target_word}", fill=TEXT_COLOR, font=font)

    for row in range(6):
        guess = guesses[row] if row < len(guesses) else "....."
        pattern = patterns[row] if row < len(patterns) else "....."
        for col in range(5):
            x0 = left + col * (cell + gap)
            y0 = top + row * (cell + gap)
            x1 = x0 + cell
            y1 = y0 + cell
            color = GRID_COLORS.get(pattern[col], GRID_EMPTY)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=6, fill=color)
            char = guess[col].upper() if guess[col].isalpha() else ""
            tw, th = draw.textbbox((0, 0), char, font=font)[2:]
            draw.text(
                (x0 + (cell - tw) / 2, y0 + (cell - th) / 2),
                char,
                fill=TEXT_COLOR,
                font=font,
            )
    return np.asarray(image, dtype=np.uint8)


def build_frames(
    model: MaskablePPO,
    env: WordleEnv,
    episodes: int,
    seed: int,
) -> list[np.ndarray]:
    """Roll out episodes and accumulate frame sequence for GIF export."""
    frames: list[np.ndarray] = []
    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        frames.append(draw_board_frame(env.state_snapshot(), episode_idx))

        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            frames.append(draw_board_frame(env.state_snapshot(), episode_idx))
            done = bool(terminated or truncated)

        # Hold the end-state frame a bit longer
        for _ in range(2):
            frames.append(draw_board_frame(env.state_snapshot(), episode_idx))
    return frames


def main() -> None:
    """Program entrypoint."""
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = MaskablePPO.load(args.model)
    env = WordleEnv(data_dir=args.data_dir, max_turns=args.max_turns)
    validate_model_compatibility(model, env, args.model)

    frames = build_frames(model=model, env=env, episodes=args.episodes, seed=args.seed)
    imageio.mimsave(output_path, frames, duration=1.0 / max(args.fps, 0.1))
    print(f"Wrote GIF: {output_path}")


if __name__ == "__main__":
    main()
