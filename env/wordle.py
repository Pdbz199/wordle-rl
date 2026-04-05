from __future__ import annotations

"""
Wordle Gymnasium environment and dictionary utilities.

This module provides:
- Dictionary loading and validation from a `words.txt` file.
- Wordle scoring logic that correctly handles repeated letters.
- `WordleEnv`: a Gymnasium environment compatible with sb3-contrib's MaskablePPO.
  It exposes an action-mask interface so the policy only considers valid guesses.
- Helper utilities for pattern parsing, used by the interactive solver and
  visualization scripts.

Key design decisions
--------------------
- After each guess, only words consistent with all prior feedback remain valid
  actions (hard-mode style). This reduces the effective action space from ~15K
  to often <100 by turn 3-4, making learning tractable for PPO.
- Shaped reward signal. This combines a small per-step penalty, an
  information-gain bonus (fraction of candidates eliminated), a win bonus
  (with early-solve scaling), and a loss penalty to produce a dense learning signal.
- Compact observation. A 66-dimensional float vector that encodes guess
  characters, feedback patterns, per-letter status, and turn progress all
  normalized to [0, 1] for stable neural network training.
"""

from enum import IntEnum
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np


class Color(IntEnum):
    """Wordle tile feedback colors.

    These integer codes are used throughout the codebase:
    - In `score_guess` return arrays.
    - In the observation's feedback-pattern features (normalized to [0, 1]).
    - In the `_letter_status_features` tracker.
    """

    ABSENT = 0    # Gray. Letter is not in the target word
    PRESENT = 1   # Yellow. Letter is in the target but in a different position
    CORRECT = 2   # Green. Letter is in the correct position


# Human-readable pattern symbols used by the CLI tools.
# "b" = black/gray (absent), "y" = yellow (present), "g" = green (correct)
PATTERN_TO_SYMBOLS = {Color.ABSENT: "b", Color.PRESENT: "y", Color.CORRECT: "g"}
SYMBOL_TO_PATTERN = {
    "b": Color.ABSENT,
    "x": Color.ABSENT,  # Alternative alias for gray
    "y": Color.PRESENT,
    "g": Color.CORRECT,
}


def load_words(words_path: str | Path) -> list[str]:
    """
    Load five-letter alphabetic words from a text file.

    Expected format: one token per line.
    """
    words: list[str] = []
    seen: set[str] = set()
    with open(words_path, encoding="utf-8") as handle:
        for raw_line in handle:
            word = raw_line.strip().lower()
            if len(word) != 5 or not word.isalpha():
                continue
            if word in seen:
                continue
            words.append(word)
            seen.add(word)
    if len(words) == 0:
        raise ValueError(f"No valid 5-letter words found in {words_path}")
    return words


def load_dictionary_words(
    data_dir: str | Path,
    words_file: str = "words.txt",
) -> list[str]:
    """
    Load the Wordle dictionary from `words.txt`.

    The dictionary file is required and must live in `data_dir`.
    """
    data_path = Path(data_dir)
    words_path = data_path / words_file
    if not words_path.exists():
        raise FileNotFoundError(f"Dictionary file '{words_file}' was not found in {data_path}.")
    return load_words(words_path)


def score_guess(guess: str, target: str) -> np.ndarray:
    """
    Score a guess against a target word using official Wordle rules.

    Uses a two-pass algorithm to handle repeated letters correctly:

    Pass 1: Exact matches (green):
      Compare each position; if `guess[i] == target[i]`, mark green and
      remove that target letter from the pool of available matches.

    Pass 2: Misplaced matches (yellow):
      For each non-green guess letter, check if it appears anywhere in the
      remaining (unmatched) target letters. If so, mark yellow and consume
      that target letter so it can't be matched again.

    Any letter not matched in either pass is marked gray (absent).

    Parameters
    ----------
    guess : str
        The 5-letter guess word.
    target : str
        The 5-letter target (secret) word.

    Returns
    -------
    np.ndarray
        Length-5 int8 array of `Color` values (0=absent, 1=present, 2=correct).

    Example
    -------
    >>> score_guess("rates", "steal")
    array([1, 0, 1, 1, 1])  # r=yellow, a=absent, t=yellow, e=yellow, s=yellow
    """
    if len(guess) != 5 or len(target) != 5:
        raise ValueError("guess and target must both be exactly 5 letters.")

    result = np.zeros(5, dtype=np.int8)
    remaining = list(target)

    # Pass 1: Mark exact positional matches (green)
    for idx, letter in enumerate(guess):
        if letter == remaining[idx]:
            result[idx] = Color.CORRECT.value
            remaining[idx] = None  # Consume this target letter

    # Pass 2: Mark misplaced matches (yellow) from leftover target letters
    for idx, letter in enumerate(guess):
        if result[idx] == Color.CORRECT.value:
            continue
        try:
            match_idx = remaining.index(letter)
        except ValueError:
            continue  # Letter not in remaining target letters -> stays absent (0)
        result[idx] = Color.PRESENT.value
        remaining[match_idx] = None  # Consume so duplicates aren't double-counted
    return result


def pattern_to_string(pattern: Sequence[int]) -> str:
    """Convert Color pattern values into compact `b/y/g` representation."""
    return "".join(PATTERN_TO_SYMBOLS[Color(int(value))] for value in pattern)


def parse_pattern_string(pattern: str) -> np.ndarray:
    """Parse compact pattern text (`b/y/g`) into integer pattern values."""
    cleaned = pattern.strip().lower()
    if len(cleaned) != 5:
        raise ValueError("Pattern must contain exactly 5 characters.")
    try:
        values = [SYMBOL_TO_PATTERN[ch] for ch in cleaned]
    except KeyError as exc:
        raise ValueError("Pattern can only contain g/y/b (or x for gray).") from exc
    return np.asarray([value.value for value in values], dtype=np.int8)


class WordleEnv(gym.Env):
    """
    Gymnasium environment for Wordle with action masks for MaskablePPO.

    This environment models one game of Wordle as a reinforcement learning episode.
    The agent selects word indices as actions; the environment scores each guess,
    updates the observation, computes a shaped reward, and provides an action mask
    restricting future guesses to words consistent with all prior feedback.

    Observation space
    -----------------
    A flat `Box(0, 1, shape=(66,), float32)` vector (for `max_turns=6`).
    See `_get_obs()` for the full layout.

    Action space
    ------------
    `Discrete(n_words)`. The agent picks an index into the dictionary word list.

    Action masking
    --------------
    `action_masks()` returns a boolean array. MaskablePPO uses this to zero out
    invalid logits before the softmax, so the agent can only pick feasible words.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the word list file.
    words_file : str
        Filename of the word list within `data_dir`.
    mask_to_candidates : bool
        If True (default), mask actions to only currently feasible candidate words
        (hard-mode). If False, only mask previously guessed words.
    max_turns : int
        Maximum number of guesses per episode (standard Wordle = 6).
    step_penalty : float
        Small negative reward applied every turn to encourage efficiency.
    info_gain_reward : float
        Multiplier for the candidate-elimination reward component.
    win_reward : float
        Bonus reward for correctly guessing the target word.
    loss_penalty : float
        Negative reward for failing to solve within `max_turns`.
    invalid_action_penalty : float
        Penalty for selecting a masked-out action (causes immediate termination).
    """

    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(
        self,
        data_dir: str = "data",
        words_file: str = "words.txt",
        mask_to_candidates: bool = True,
        max_turns: int = 6,
        step_penalty: float = -0.05,
        info_gain_reward: float = 1.5,
        win_reward: float = 5.0,
        loss_penalty: float = -2.0,
        invalid_action_penalty: float = -1.5,
    ) -> None:
        # Dictionary setup.
        # Load the word list and build fast lookup structures
        self.words = load_dictionary_words(data_dir=data_dir, words_file=words_file)
        self.word_to_index = {word: idx for idx, word in enumerate(self.words)}
        self.word_set = set(self.words)
        self.n_words = len(self.words)
        self.mask_to_candidates = mask_to_candidates

        # Reward configuration
        self.max_turns = max_turns
        self.step_penalty = step_penalty
        self.info_gain_reward = info_gain_reward
        self.win_reward = win_reward
        self.loss_penalty = loss_penalty
        self.invalid_action_penalty = invalid_action_penalty

        # Gymnasium spaces.
        # The action space is picking one word from the dictionary
        self.action_space = gym.spaces.Discrete(self.n_words)

        # Observation layout (66 dims for max_turns=6):
        #   [0  .. 29]  Guess character features      (max_turns * 5 = 30)
        #   [30 .. 59]  Feedback pattern features     (max_turns * 5 = 30)
        #   [60 .. 85]  Per-letter status tracker     (26, one per a-z)
        #   [86]        Normalized turn progress      (1)
        obs_size = (self.max_turns * 5) + (self.max_turns * 5) + 26 + 1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32,
        )

        # Episode state (initialized here, reset each episode)
        self.target_word: str | None = None
        self.turn = 0
        self.guess_indices = np.full(self.max_turns, -1, dtype=np.int32)
        self.pattern_matrix = np.full((self.max_turns, 5), -1, dtype=np.int8)
        # Candidate set: indices into self.words that are still consistent with
        # all observed feedback. Starts as the full dictionary each episode
        self.candidate_indices = np.arange(self.n_words, dtype=np.int32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Start a new episode.

        `options` may include:
        - target_word: fixed target for deterministic/testing scenarios.
        """
        super().reset(seed=seed)
        self._reset_episode_state()
        requested_target = options.get("target_word") if options else None
        if requested_target is None:
            # Randomly select a target word from the dictionary using the environment's RNG
            target_index = int(self.np_random.integers(0, self.n_words))
            self.target_word = self.words[target_index]
        else:
            # Validate the requested target word and set it if valid
            target_word = str(requested_target).strip().lower()
            if target_word not in self.word_set:
                raise ValueError(f"Unknown target_word '{target_word}'. Must be in the current dictionary.")
            self.target_word = target_word
        return self._get_obs(), {"candidate_count": int(self.candidate_indices.size)}

    def reset_assistant(self) -> np.ndarray:
        """Reset without picking a hidden target (for manual feedback mode)."""
        self._reset_episode_state()
        self.target_word = None
        return self._get_obs()

    def step(self, action: int):
        """
        Execute one guess and return the Gymnasium 5-tuple.

        Returns
        -------
        obs : np.ndarray
            Updated 66-dim observation vector.
        reward : float
            Shaped reward (see class docstring for the reward formula).
        terminated : bool
            True if the episode ended (solved, out of turns, or invalid action).
        truncated : bool
            Always False (no time-limit truncation beyond max_turns).
        info : dict
            Episode metadata: `is_success`, `num_turns`, `candidate_count`,
            `guess`, `pattern`, `invalid_action`.
        """
        if self.target_word is None:
            raise RuntimeError("step() requires a target word. Use reset() first.")
        if self.turn >= self.max_turns:
            raise RuntimeError("Episode already finished. Call reset().")
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is out of bounds.")

        # Invalid action check.
        # If MaskablePPO somehow selects a masked-out action, terminate immediately
        action_mask = self.action_masks()
        if not action_mask[action]:
            info = {
                "is_success": False,
                "num_turns": int(self.turn),
                "candidate_count": int(self.candidate_indices.size),
                "invalid_action": True,
            }
            return self._get_obs(), self.invalid_action_penalty, True, False, info

        # Score the guess against the target
        guess = self.words[action]
        pattern = score_guess(guess, self.target_word)
        self.guess_indices[self.turn] = action
        self.pattern_matrix[self.turn] = pattern

        # Filter candidates and compute information gain
        previous_count = max(int(self.candidate_indices.size), 1)
        self._filter_candidates(guess, pattern)
        current_count = int(self.candidate_indices.size)
        solved = bool(np.all(pattern == Color.CORRECT.value))

        # Compute shaped reward.
        # Base part of reward is a small per-step penalty to encourage fewer guesses
        reward = self.step_penalty
        # Information gain part of reward that is proportional to the fraction of candidates eliminated
        reduction_ratio = (previous_count - current_count) / previous_count
        reward += self.info_gain_reward * float(reduction_ratio)
        # Win bonus part of reward is fixed reward + extra bonus for each remaining turn (rewards early solves)
        if solved:
            reward += self.win_reward + (self.max_turns - self.turn - 1) * 0.5

        self.turn += 1
        terminated = solved
        # Loss penalty part of reward is applied when all turns are exhausted without solving
        if not terminated and self.turn >= self.max_turns:
            reward += self.loss_penalty
            terminated = True

        info = {
            "is_success": solved,
            "num_turns": int(self.turn) if terminated else 0,
            "candidate_count": current_count,
            "guess": guess,
            "pattern": pattern_to_string(pattern),
            "invalid_action": False,
        }
        return self._get_obs(), float(reward), terminated, False, info

    def action_masks(self) -> np.ndarray:
        """
        Return a boolean mask indicating which actions are currently valid.

        MaskablePPO calls this method at each step. The returned mask is applied
        to the policy's logit vector before softmax, effectively zeroing out
        the probability of invalid actions.

        Masking strategy (when `mask_to_candidates=True`):
        1. Start with only candidate words (words consistent with all feedback).
        2. Remove any previously guessed words (no repeat guesses).
        3. Safety fallback: if the mask is all-False, unmask everything.

        Returns
        -------
        np.ndarray
            Boolean array of shape `(n_words,)`.
        """
        if self.mask_to_candidates:
            # Only allow words that are still consistent with all prior feedback
            mask = np.zeros(self.n_words, dtype=bool)
            if self.candidate_indices.size > 0:
                mask[self.candidate_indices] = True
            else:
                mask[:] = True  # Fallback. This shouldn't happen, but never leave mask empty
        else:
            mask = np.ones(self.n_words, dtype=bool)

        # Exclude previously guessed words
        guessed = self.guess_indices[self.guess_indices >= 0]
        if guessed.size > 0:
            mask[guessed] = False

        # For safety, never return an all-False mask (would crash MaskablePPO)
        if not np.any(mask):
            mask[:] = True
        return mask

    def apply_feedback(self, guess_word: str, pattern: Sequence[int] | str) -> dict:
        """
        Update state from external feedback (interactive assistant mode).

        `pattern` can be:
        - sequence of ints in {0,1,2}
        - string of 5 chars using g/y/b (or x for gray)
        """
        if self.turn >= self.max_turns:
            raise RuntimeError("All turns are used. Reset assistant state first.")
        guess = guess_word.strip().lower()
        if guess not in self.word_to_index:
            raise ValueError(f"'{guess}' is not in the dictionary.")

        if isinstance(pattern, str):
            pattern_array = parse_pattern_string(pattern)
        else:
            pattern_array = np.asarray(pattern, dtype=np.int8)
            if pattern_array.shape != (5,) or not np.all(np.isin(pattern_array, [0, 1, 2])):
                raise ValueError("pattern must be length-5 values in {0,1,2}.")

        action = self.word_to_index[guess]
        self.guess_indices[self.turn] = action
        self.pattern_matrix[self.turn] = pattern_array
        self._filter_candidates(guess, pattern_array)
        solved = bool(np.all(pattern_array == Color.CORRECT.value))
        self.turn += 1

        return {
            "solved": solved,
            "turn": int(self.turn),
            "candidate_count": int(self.candidate_indices.size),
        }

    def get_candidates(self, limit: int | None = None) -> list[str]:
        """Return current candidate words that remain consistent with known feedback."""
        if self.candidate_indices.size == 0:
            return []
        words = [self.words[int(idx)] for idx in self.candidate_indices]
        if limit is None:
            return words
        return words[:limit]

    def get_obs(self) -> np.ndarray:
        """Public getter for the current observation vector."""
        return self._get_obs()

    def state_snapshot(self) -> dict:
        """Return serializable state summary used by rendering/visualization helpers."""
        guesses: list[str] = []
        patterns: list[str] = []
        for row in range(self.turn):
            guess_idx = int(self.guess_indices[row])
            guesses.append(self.words[guess_idx])
            patterns.append(pattern_to_string(self.pattern_matrix[row]))
        return {
            "turn": int(self.turn),
            "guesses": guesses,
            "patterns": patterns,
            "candidate_count": int(self.candidate_indices.size),
            "target_word": self.target_word,
        }

    def render(self):
        """Render a plaintext board suitable for terminal debugging."""
        for row in range(self.max_turns):
            guess_idx = int(self.guess_indices[row])
            if guess_idx < 0:
                print(f"{row + 1}: _____  .....")
                continue
            guess = self.words[guess_idx]
            pattern = pattern_to_string(self.pattern_matrix[row])
            print(f"{row + 1}: {guess}  {pattern}")
        print(f"Candidates remaining: {int(self.candidate_indices.size)}")

    def _reset_episode_state(self) -> None:
        """Reset all mutable episode state fields."""
        self.turn = 0
        self.guess_indices.fill(-1)
        self.pattern_matrix.fill(-1)
        self.candidate_indices = np.arange(self.n_words, dtype=np.int32)

    def _filter_candidates(self, guess_word: str, pattern: np.ndarray) -> None:
        """
        Filter the candidate set to words consistent with the observed feedback.

        For each remaining candidate, simulate ``score_guess(guess_word, candidate)``
        and keep only those that produce the exact same pattern. This is the core
        of the information-gain mechanism: each guess eliminates all candidates
        that would have produced a different feedback pattern.
        """
        if self.candidate_indices.size == 0:
            return
        next_candidates = [
            idx
            for idx in self.candidate_indices
            if np.array_equal(score_guess(guess_word, self.words[int(idx)]), pattern)
        ]
        self.candidate_indices = np.asarray(next_candidates, dtype=np.int32)

    def _letter_status_features(self) -> np.ndarray:
        """
        Build a 26-dimensional feature vector tracking the best-known status of each letter.

        Each letter (a-z) is assigned a status based on all feedback received so far:
        - 0 -> unknown (not yet guessed)
        - 1 -> absent (confirmed not in the target)
        - 2 -> present (in the target but position unknown)
        - 3 -> correct (position confirmed)

        Statuses only increase (e.g., a letter marked "present" won't revert to
        "absent" if a later guess also shows it as absent in a different position).

        The final values are normalized to [0, 1] by dividing by 3.
        """
        status = np.zeros(26, dtype=np.float32)
        for row in range(self.turn):
            guess = self.words[int(self.guess_indices[row])]
            pattern = self.pattern_matrix[row]
            for col, letter in enumerate(guess):
                letter_idx = ord(letter) - ord("a")
                color = Color(int(pattern[col]))
                # Only upgrade status, never downgrade
                if color == Color.CORRECT:
                    status[letter_idx] = 3.0
                elif color == Color.PRESENT and status[letter_idx] < 2.0:
                    status[letter_idx] = 2.0
                elif color == Color.ABSENT and status[letter_idx] < 1.0:
                    status[letter_idx] = 1.0
        return status / 3.0  # Normalize to [0, 1]

    def _get_obs(self) -> np.ndarray:
        """
        Build the flat observation vector passed to the policy network.

        The observation is a 66-dimensional float32 vector (for `max_turns=6`)
        with the following layout:

        ------------------------------------------------------------------
        Dimensions    Size      Content
        ------------------------------------------------------------------
        `[0..29]`   30          Guess characters. Each letter encoded
                                as `(ord(ch) - ord('a') + 1) / 26`.
                                Unplayed turns are all zeros.
        `[30..59]`  30          Feedback patterns. Each tile color
                                encoded as `(color + 1) / 3`.
                                Unplayed turns are all zeros.
        `[60..85]`  26          Letter status: one float per a-z,
                                from `_letter_status_features()`.
        `[86]`      1           Turn progress. `turn / max_turns`.
        ------------------------------------------------------------------

        All values are normalized to [0, 1] for stable neural network training.
        """
        guess_features = np.zeros((self.max_turns, 5), dtype=np.float32)
        pattern_features = np.zeros((self.max_turns, 5), dtype=np.float32)

        for row in range(self.turn):
            guess = self.words[int(self.guess_indices[row])]
            # Encode each letter as a normalized position in the alphabet (a=1/26, z=26/26)
            guess_features[row] = [(ord(ch) - ord("a") + 1) / 26.0 for ch in guess]
            # Encode feedback colors: absent(0) -> 0.33, present(1) -> 0.67, correct(2) -> 1.0
            pattern_features[row] = (self.pattern_matrix[row].astype(np.float32) + 1.0) / 3.0

        obs = np.concatenate([
            guess_features.flatten(),       # 30 dims. Guess letter encodings
            pattern_features.flatten(),     # 30 dims. Feedback pattern encodings
            self._letter_status_features(), # 26 dims. Per-letter knowledge summary
            np.asarray([self.turn / self.max_turns], dtype=np.float32),  # 1 dim. Turn progress
        ])
        return obs.astype(np.float32)
