# Solving Wordle with Reinforcement Learning

A reinforcement learning agent that learns to play [Wordle](https://www.nytimes.com/games/wordle/index.html) using Maskable PPO (Proximal Policy Optimization with invalid-action masking).

The agent is trained from scratch. It learns which words to guess purely from reward signals, using a neural network policy that observes the game state and outputs a probability distribution over ~15,000 candidate words.

---

## Table of Contents

- [How Wordle Works](#how-wordle-works)
- [RL Formulation](#rl-formulation)
  - [Observation Space (State)](#observation-space-state)
  - [Action Space](#action-space)
  - [Action Masking](#action-masking)
  - [Reward Structure](#reward-structure)
- [Model Architecture](#model-architecture)
- [Setup](#setup)
- [Usage](#usage)
  - [Environment Smoke Test](#1-environment-smoke-test)
  - [Training](#2-training)
  - [Playing Sample Games](#3-playing-sample-games)
  - [Interactive Solver](#4-interactive-solver)
  - [Rendering GIFs](#5-rendering-gifs)
- [Training Details](#training-details)
- [Reproducibility](#reproducibility)

---

## How Wordle Works

Wordle is a word-guessing game where the player has 6 attempts to guess a hidden 5-letter word. After each guess, the game provides feedback for each letter:

- Green (g): The letter is in the correct position.
- Yellow (y): The letter is in the word but in the wrong position.
- Gray (b): The letter is not in the word (or all instances are already accounted for).

The scoring handles repeated letters correctly. For example, if the target is `"steel"` and you guess `"geese"`, only one `e` gets green (position 4) and the duplicate letters are scored against the remaining unmatched target letters.

---

## RL Formulation

We frame Wordle as a finite-horizon Markov Decision Process (MDP). Each game is one episode with at most 6 timesteps (guesses).

### Observation Space (State)

The agent receives a 66-dimensional float vector (all values normalized to `[0, 1]`) encoding the full game state:

| Dimensions | Size | Description |
|---|---|---|
| `[0–29]` | 30 | Guess letter features. For each of the 6 turns, 5 floats encode the guessed letters. Each letter is normalized as `(ord(ch) - ord('a') + 1) / 26`. Unplayed turns are zeros. |
| `[30–59]` | 30 | Feedback pattern features. For each of the 6 turns, 5 floats encode the Wordle feedback. Each tile color is normalized as `(color_value + 1) / 3`, where `absent=0`, `present=1`, `correct=2`. Unplayed turns are zeros. |
| `[60–85]` | 26 | Letter status tracker. One float per letter of the alphabet (`a`–`z`). Tracks the best-known status: `0.0` = unknown, `0.33` = absent (gray), `0.67` = present (yellow), `1.0` = correct (green). This gives the agent a summary of what it knows about each letter. |
| `[86]` | 1 | Turn Progress. Calculated as current turn divided by max turns (`turn / 6`), so the agent knows how many guesses remain. |

Why this encoding? The observation is intentionally compact and fully normalized. The letter status tracker (dims 60–85) acts as a compressed memory of all prior feedback, while the per-turn features (dims 0–59) preserve the exact sequence of guesses and responses. This gives the agent enough information to reconstruct the game state without requiring a massive observation space.

### Action Space

The action space is `Discrete(14,855)`. This translates to one action per word in the dictionary. When the agent selects action `i`, it guesses `words[i]`.

The dictionary is loaded from `data/words.txt`, which contains 14,855 valid 5-letter English words (one per line, lowercase, alphabetic only).

### Action Masking

Raw Wordle has a huge action space (~15K words), and most actions are suboptimal after the first guess because they contradict known feedback. To make learning tractable, we use action masking:

1. After each guess, the environment computes which dictionary words are still consistent with all observed feedback (i.e., words that would produce the same color patterns if they were the target).
2. The action mask is a boolean array of length 14,855. Only consistent candidate words have `True`; all others have `False`.
3. Previously guessed words are always masked out (no repeat guesses).
4. MaskablePPO applies this mask to the policy's logit output before the softmax, so the agent can only select from feasible words. This happens during both rollout collection and gradient updates.

The masking dramatically reduces the effective action space (often to a few hundred or fewer candidates by turn 3–4), making the learning problem tractable.

### Reward Structure

The reward signal is designed to encourage both information-gathering (narrowing down candidates) and fast solves:

```
reward = step_penalty                                           # -0.05 per guess (encourages fewer guesses)
       + info_gain_reward * (prev_candidates - curr_candidates) # +1.5 * fraction of candidates eliminated
             / prev_candidates                                  #   (rewards informative guesses)
       + [if solved] win_reward                                 # +5.0 for solving the puzzle
       + [if solved] (max_turns - turn - 1) * 0.5               # +0.5 per remaining turn (rewards early solves)
       + [if failed] loss_penalty                               # -2.0 for failing to solve in 6 guesses
```

| Component | Default Value | Purpose |
|---|---|---|
| `step_penalty` | `-0.05` | Small cost per guess to encourage efficiency |
| `info_gain_reward` | `+1.5` | Scaled by the fraction of candidates eliminated. The idea is to reward guesses that narrow the search space |
| `win_reward` | `+5.0` | Large bonus for correctly guessing the target word |
| Early solve bonus | `+0.5` per remaining turn | Extra reward for solving in fewer guesses (e.g., +2.0 for solving on turn 2) |
| `loss_penalty` | `-2.0` | Penalty for exhausting all 6 guesses without solving |
| `invalid_action_penalty` | `-1.5` | Penalty if the agent somehow picks a masked-out action (causes immediate episode termination) |

---

## Model Architecture

The agent uses MaskablePPO from [sb3-contrib](https://sb3-contrib.readthedocs.io/), which extends PPO with support for invalid-action masking.

Policy network (feedforward neural network):
- Input: 66-dim normalized observation vector
- Shared backbone: Two hidden layers of 256 units each with ReLU activations
- Policy head: Outputs 14,855 logits (one per dictionary word). Invalid actions are masked to `-inf` before softmax.
- Value head: Outputs a single scalar V(s) estimating the expected return from the current state.

Key PPO hyperparameters:

| Hyperparameter | Value | Description |
|---|---|---|
| Learning rate | `3e-4` | Adam optimizer step size |
| Discount (γ) | `0.99` | Future reward discount factor |
| Entropy coefficient | `0.01` | Encourages exploration by penalizing overly deterministic policies |
| Rollout length (n_steps) | `256` | Number of environment steps collected per update |
| Minibatch size | `256` | PPO minibatch size for gradient updates |
| Parallel environments | `4` | Vectorized environments for faster data collection |

---

## Setup

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Linux/macOS:
source .venv/bin/activate

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Optional: GPU PyTorch

`requirements.txt` pins CPU-only PyTorch for portability. For CUDA support:

```bash
pip install --force-reinstall --no-deps \
  torch==2.11.0+cu130 \
  torchvision==0.26.0+cu130 \
  torchaudio==2.11.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
```

---

## Usage

### 1. Environment Smoke Test

Verify the environment works correctly by running random-action episodes:

```bash
python test_env.py --episodes 10 --seed 123
```

This creates a `WordleEnv`, plays episodes with random (but valid) guesses, and prints per-episode results.

### 2. Training

Train a new agent from scratch:

```bash
python train.py \
  --total-timesteps 120000 \
  --n-envs 4 \
  --learning-rate 3e-4 \
  --ent-coef 0.01 \
  --eval-freq 10000 \
  --eval-episodes 100 \
  --run-name my-run \
  --seed 42
```

Training artifacts are saved to `runs/<run-name>/`:

| File | Description |
|---|---|
| `wordle_maskable_ppo.zip` | Final trained model checkpoint |
| `best_model/best_model.zip` | Best model (highest eval reward during training) |
| `eval_metrics.json` | Final evaluation results (win rate, mean reward, mean turns) |
| `training_metrics.png` | Reward and win-rate curves over training |
| `run_config.json` | Full hyperparameter configuration for reproducibility |
| `artifacts.json` | Manifest listing all output file paths |

To enable TensorBoard logging (disabled by default to save storage), add the `--tensorboard` flag:

```bash
python train.py --total-timesteps 120000 --run-name my-run --tensorboard

# Then monitor in real-time:
tensorboard --logdir wordle_tensorboard
```

### 3. Playing Sample Games

Watch the trained agent play Wordle in the terminal:

```bash
python play.py --model runs/my-run/best_model/best_model.zip --episodes 5
```

Each episode shows the agent's guesses, feedback patterns, and whether it solved the puzzle.

### 4. Interactive Solver

Use the trained agent as an assistant while playing Wordle on your phone or browser:

```bash
python interactive_solver.py \
  --model runs/my-run/best_model/best_model.zip \
  --show-candidates
```

The solver suggests a word each turn. You enter the actual word you guessed and the feedback pattern you received (e.g., `gybbg` for green-yellow-gray-gray-green). It then updates its internal state and suggests the next guess.

### 5. Rendering GIFs

Generate an animated GIF of the agent playing:

```bash
python visualize.py \
  --model runs/my-run/best_model/best_model.zip \
  --output runs/my-run/example_games.gif \
  --episodes 6
```

---

## Training Details

The training pipeline works as follows:

1. 4 parallel `WordleEnv` instances collect rollout data simultaneously. On Windows, `DummyVecEnv` (in-process) is used; on Linux/macOS, `SubprocVecEnv` (multiprocess) is used for better throughput.

2. Each environment runs for 256 steps (about 40–50 episodes per env), collecting `(observation, action, reward, mask)` tuples.

3. The collected rollouts are split into minibatches of 256. PPO computes the clipped surrogate objective and value loss, then updates the shared network via Adam.

4. Every 10,000 timesteps, the agent is evaluated on 30 episodes with a separate environment (deterministic action selection). The best-performing checkpoint is saved automatically.

5. After training completes, 100 episodes are run to compute final metrics (win rate, mean reward, mean turns to solve).

---

## Reproducibility

For fully reproducible results:

1. Use the same `data/words.txt` dictionary.
2. Set an explicit `--seed` (default: 42).
3. Keep `--deterministic-torch` enabled (default: on).
4. Pin package versions via `requirements.txt`.
5. Compare `run_config.json` files to verify identical settings.
