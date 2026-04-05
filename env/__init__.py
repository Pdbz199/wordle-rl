"""Wordle RL environment package.

Re-exports the core environment class and utilities so they can be imported as::

    from env import WordleEnv, score_guess, Color
"""

from env.wordle import (
    Color,
    WordleEnv,
    load_dictionary_words,
    load_words,
    parse_pattern_string,
    pattern_to_string,
    score_guess,
)

__all__ = [
    "Color",
    "WordleEnv",
    "load_dictionary_words",
    "load_words",
    "parse_pattern_string",
    "pattern_to_string",
    "score_guess",
]
