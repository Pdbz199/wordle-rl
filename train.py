from __future__ import annotations

"""
Training script for the Wordle RL agent using MaskablePPO.

This is the main entrypoint for running training experiments. It:

1. Seeds all RNGs (random, numpy, torch, CUDA) and optionally enables deterministic CUDA algorithms.
2. Creates vectorized environments to run in parallel to collect rollout data faster.
   A separate single-env is used for periodic evaluation.
3. Initializes MaskablePPO from `sb3-contrib` that respects action
   masks during both rollout and gradient updates. The policy is a
   2-layer MLP (256x256 hidden units) mapping the 66-dim observation
   to logits over ~15K dictionary words.
4. Trains with callbacks using `MaskableEvalCallback` to run periodic
   evaluations and saves the best checkpoint based on mean reward.
5. Saves final/best model checkpoints, evaluation metrics (JSON),
   training curves (PNG), full config (JSON), and an artifact manifest.

All artifacts are saved under `runs/<run-name>/` for easy comparison across experiments.
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env.wordle import WordleEnv


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments controlling training/evaluation behavior."""
    parser = argparse.ArgumentParser(description="Train a Wordle agent with MaskablePPO.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory that contains Wordle dictionary data.")
    parser.add_argument("--total-timesteps", type=int, default=120_000, help="Total PPO environment timesteps.")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--n-steps", type=int, default=256, help="Rollout length per environment per update.")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy regularization coefficient.")
    parser.add_argument("--max-turns", type=int, default=6, help="Maximum guesses allowed per episode.")
    parser.add_argument(
        "--mask-to-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask actions to currently feasible candidates (recommended).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device (e.g. auto/cpu/cuda).")
    parser.add_argument(
        "--deterministic-torch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable best-effort deterministic torch behavior.",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=None,
        help="Optional torch thread cap for CPU determinism/performance tuning.",
    )
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency (in env timesteps).")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Final evaluation episode count.")
    parser.add_argument("--eval-callback-episodes", type=int, default=30, help="Episodes per periodic eval callback.")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory for run artifacts.")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging (disabled by default to save storage).")
    parser.add_argument("--tensorboard-dir", type=str, default="wordle_tensorboard", help="TensorBoard log directory (only used when --tensorboard is passed).")
    parser.add_argument("--run-name", type=str, default=None, help="Run name override. Uses timestamp when omitted.")
    parser.add_argument(
        "--vec-env",
        type=str,
        choices=("auto", "dummy", "subproc"),
        default="auto",
        help="Vectorized environment backend.",
    )
    return parser.parse_args()


def set_global_seeds(seed: int, deterministic_torch: bool, torch_num_threads: int | None = None) -> None:
    """
    Set process-level seeds and deterministic settings.

    Notes:
    - This is a best-effort setup. Some GPU kernels remain nondeterministic.
    - `torch.use_deterministic_algorithms(..., warn_only=True)` is used so
      experiments keep running even if a nondeterministic op is encountered.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch_num_threads is not None and torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)

    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Keep benchmark enabled for possible performance gains when determinism is not required
        torch.backends.cudnn.benchmark = True


def make_run_dirs(args: argparse.Namespace) -> dict[str, Path]:
    """Create run directories and return canonical paths."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"ppo-{timestamp}"
    run_dir = Path(args.runs_dir) / run_name
    monitor_dir = run_dir / "monitor"
    eval_monitor_dir = run_dir / "eval_monitor"
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    eval_monitor_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_name": Path(run_name),
        "run_dir": run_dir,
        "monitor_dir": monitor_dir,
        "eval_monitor_dir": eval_monitor_dir,
    }


def save_run_config(args: argparse.Namespace, output_path: Path) -> None:
    """Persist full run configuration to JSON for reproducibility."""
    config: dict[str, Any] = vars(args).copy()
    config["created_at"] = datetime.now().isoformat()
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def load_monitor_data(monitor_dir: Path) -> pd.DataFrame | None:
    """
    Load monitor CSV files and return a unified episode table.

    The resulting dataframe includes:
    - `episode` index
    - cumulative `timesteps`
    - normalized success/turn columns when available
    """
    frames: list[pd.DataFrame] = []
    for monitor_file in sorted(monitor_dir.glob("*.monitor.csv")):
        frame = pd.read_csv(monitor_file, skiprows=1)
        if frame.empty:
            continue
        frame["source"] = monitor_file.name
        frames.append(frame)
    if not frames:
        return None

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("t").reset_index(drop=True)
    data["episode"] = np.arange(1, len(data) + 1)
    data["timesteps"] = data["l"].cumsum()
    if "is_success" in data.columns:
        data["is_success"] = pd.to_numeric(data["is_success"], errors="coerce").fillna(0.0)
    if "num_turns" in data.columns:
        data["num_turns"] = pd.to_numeric(data["num_turns"], errors="coerce").fillna(0.0)
    return data


def plot_training_metrics(monitor_dir: Path, output_path: Path) -> None:
    """
    Render reward and success-rate curves from monitor logs.

    A moving average (window=100 episodes) is plotted on top of raw episode reward.
    """
    data = load_monitor_data(monitor_dir)
    if data is None or data.empty:
        return

    reward_ma = data["r"].rolling(window=100, min_periods=1).mean()
    if "is_success" in data.columns:
        success_ma = data["is_success"].rolling(window=100, min_periods=1).mean()
    else:
        success_ma = pd.Series(np.zeros(len(data)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(data["timesteps"], data["r"], color="#8da0cb", alpha=0.25, linewidth=1)
    axes[0].plot(data["timesteps"], reward_ma, color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Wordle PPO Training Curves")
    axes[0].grid(alpha=0.25)

    axes[1].plot(data["timesteps"], success_ma, color="#2ca02c", linewidth=2)
    axes[1].set_ylabel("Win Rate (MA-100)")
    axes[1].set_xlabel("Environment Timesteps")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_model(
    model: MaskablePPO,
    data_dir: str,
    max_turns: int,
    n_episodes: int,
    seed: int,
    mask_to_candidates: bool,
) -> dict:
    """Run deterministic evaluation episodes and return summary metrics."""
    env = WordleEnv(data_dir=data_dir, max_turns=max_turns, mask_to_candidates=mask_to_candidates)
    rewards: list[float] = []
    wins = 0
    turns: list[int] = []

    for episode_idx in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        done = False
        episode_reward = 0.0
        final_info = {}
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            final_info = info
        rewards.append(episode_reward)
        wins += int(final_info.get("is_success", False))
        turns.append(int(final_info.get("num_turns", 0)))

    return {
        "episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "win_rate": float(wins / n_episodes),
        "mean_turns": float(np.mean(turns)),
    }


def train() -> None:
    """
    Full training pipeline: build envs, train MaskablePPO, evaluate, and save artifacts.

    The pipeline steps are:
    1. Seed all RNGs and create output directories.
    2. Build vectorized train/eval environments with Monitor wrappers.
    3. Initialize MaskablePPO with a 2-layer MLP policy.
    4. Train with periodic evaluation callbacks.
    5. Save final model, run final evaluation, and generate training plots.
    """
    args = parse_args()
    set_global_seeds(
        seed=args.seed,
        deterministic_torch=args.deterministic_torch,
        torch_num_threads=args.torch_num_threads,
    )

    # Create output directories and save config
    dirs = make_run_dirs(args)
    run_dir = dirs["run_dir"]
    monitor_dir = dirs["monitor_dir"]
    eval_monitor_dir = dirs["eval_monitor_dir"]
    run_name = dirs["run_name"].name
    # Only create TensorBoard log directory if explicitly enabled
    if args.tensorboard:
        tensorboard_dir = Path(args.tensorboard_dir)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_log = str(tensorboard_dir)
    else:
        tensorboard_log = None
    save_run_config(args=args, output_path=run_dir / "run_config.json")

    # Choose vectorized environment backend.
    # SubprocVecEnv uses multiprocessing for true parallelism but can fail on
    # Windows due to spawn-based process creation. DummyVecEnv runs envs
    # sequentially in-process (always safe, slightly slower)
    if args.vec_env == "dummy":
        vec_env_cls = DummyVecEnv
    elif args.vec_env == "subproc":
        vec_env_cls = SubprocVecEnv
    else:
        if os.name == "nt":
            vec_env_cls = DummyVecEnv
        else:
            vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv

    env_kwargs = {
        "data_dir": args.data_dir,
        "max_turns": args.max_turns,
        "mask_to_candidates": args.mask_to_candidates,
    }
    # Monitor wrappers log per-episode stats (reward, length, custom info keys)
    # to CSV files for later plotting
    monitor_kwargs = {"info_keywords": ("is_success", "num_turns", "candidate_count", "invalid_action")}

    train_env = make_vec_env(
        WordleEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
        monitor_dir=str(monitor_dir),
        monitor_kwargs=monitor_kwargs,
        vec_env_cls=vec_env_cls,
    )
    # Evaluation uses a single environment with a different seed range to
    # avoid evaluating on the same target-word sequence used during training
    eval_env = make_vec_env(
        WordleEnv,
        n_envs=1,
        seed=args.seed + 10_000,
        env_kwargs=env_kwargs,
        monitor_dir=str(eval_monitor_dir),
        monitor_kwargs=monitor_kwargs,
        vec_env_cls=DummyVecEnv,
    )

    # Initialize MaskablePPO.
    # net_arch=[256, 256] creates two hidden layers with 256 units each (ReLU).
    # The policy head outputs logits over the full dictionary (~15K words).
    # The value head outputs a single scalar V(s)
    policy_kwargs = {"net_arch": [256, 256]}
    model = MaskablePPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device=args.device,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
    )

    # Train with periodic evaluation.
    # eval_freq is divided by n_envs because SB3 counts per-env steps
    eval_callback = MaskableEvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval"),
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        n_eval_episodes=args.eval_callback_episodes,
        deterministic=True,
        warn=False,
    )

    print("Starting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        tb_log_name=run_name,
        progress_bar=False,
    )
    print("Training finished.")

    # Save model and run final evaluation
    final_model_path = run_dir / "wordle_maskable_ppo"
    model.save(str(final_model_path))
    print(f"Saved model: {final_model_path}.zip")

    # Run a larger final evaluation (default: 100 episodes) with a separate
    # seed range to get reliable performance estimates
    metrics = evaluate_model(
        model=model,
        data_dir=args.data_dir,
        max_turns=args.max_turns,
        n_episodes=args.eval_episodes,
        seed=args.seed + 20_000,
        mask_to_candidates=args.mask_to_candidates,
    )
    metrics_path = run_dir / "eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")

    # Generate training curves (reward + win rate over timesteps)
    metrics_png = run_dir / "training_metrics.png"
    plot_training_metrics(monitor_dir=monitor_dir, output_path=metrics_png)
    if metrics_png.exists():
        print(f"Saved training metrics plot: {metrics_png}")

    # Write a manifest listing all output artifacts for easy programmatic access
    manifest = {
        "run_name": run_name,
        "model_path": str(final_model_path.with_suffix(".zip")),
        "best_model_path": str(run_dir / "best_model" / "best_model.zip"),
        "metrics_json": str(metrics_path),
        "metrics_png": str(metrics_png),
        "monitor_dir": str(monitor_dir),
        "run_config": str(run_dir / "run_config.json"),
    }
    if tensorboard_log is not None:
        manifest["tensorboard_dir"] = tensorboard_log
    manifest_path = run_dir / "artifacts.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote artifact manifest: {manifest_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
