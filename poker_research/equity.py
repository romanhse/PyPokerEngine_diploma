"""Fast, deterministic hold'em equity estimates backed by PHEvaluator."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator
from functools import lru_cache
from itertools import combinations

from phevaluator import evaluate_cards

RANKS = "23456789TJQKA"
SUITS = "cdhs"
STANDARD_DECK: tuple[str, ...] = tuple(rank + suit for rank in RANKS for suit in SUITS)
VALID_BOARD_LENGTHS = frozenset({0, 3, 4, 5})


def stable_seed(base_seed: int, *parts: object) -> int:
    """Create a process-stable 64-bit seed; unlike ``hash()``, this survives restarts."""

    payload = "|".join((str(base_seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def equity_vs_uniform(
    hole_cards: tuple[str, str],
    board_cards: tuple[str, ...],
    *,
    sample_count: int,
    seed: int,
    exact_river: bool = True,
) -> float:
    """Estimate showdown equity against a uniformly random legal opponent hand.

    The river is enumerated exactly by default. Earlier streets use reproducible
    Monte Carlo sampling. Ties count as half a win.
    """

    normalized_hole = tuple(sorted(hole_cards))
    normalized_board = tuple(sorted(board_cards))
    return _equity_vs_uniform_cached(
        normalized_hole,
        normalized_board,
        sample_count,
        seed,
        exact_river,
    )


@lru_cache(maxsize=32_768)
def _equity_vs_uniform_cached(
    hole_cards: tuple[str, str],
    board_cards: tuple[str, ...],
    sample_count: int,
    seed: int,
    exact_river: bool,
) -> float:
    _validate_cards(hole_cards, board_cards, sample_count)
    known = frozenset((*hole_cards, *board_cards))
    remaining = tuple(card for card in STANDARD_DECK if card not in known)
    missing_board = 5 - len(board_cards)

    if exact_river and missing_board == 0:
        samples: Iterable[tuple[str, ...]] = combinations(remaining, 2)
    else:
        draw_count = 2 + missing_board
        rng = random.Random(seed)
        samples = (tuple(rng.sample(remaining, draw_count)) for _ in range(sample_count))

    wins = 0.0
    total = 0
    for sample in samples:
        opponent_hole = sample[:2]
        runout = sample[2:]
        final_board = (*board_cards, *runout)
        hero_rank = int(evaluate_cards(*hole_cards, *final_board))
        opponent_rank = int(evaluate_cards(*opponent_hole, *final_board))
        wins += float(hero_rank < opponent_rank) + 0.5 * float(hero_rank == opponent_rank)
        total += 1

    if total == 0:
        raise RuntimeError("equity estimator produced no samples")
    return wins / total


def _validate_cards(
    hole_cards: tuple[str, str], board_cards: tuple[str, ...], sample_count: int
) -> None:
    if len(hole_cards) != 2:
        raise ValueError("hold'em requires exactly two hole cards")
    if len(board_cards) not in VALID_BOARD_LENGTHS:
        raise ValueError("board must contain 0, 3, 4, or 5 cards")
    cards = (*hole_cards, *board_cards)
    if len(set(cards)) != len(cards):
        raise ValueError("duplicate cards are not allowed")
    invalid = set(cards).difference(STANDARD_DECK)
    if invalid:
        raise ValueError(f"invalid cards: {sorted(invalid)}")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")


def iter_complete_boards(
    board_cards: tuple[str, ...], remaining_cards: Iterable[str]
) -> Iterator[tuple[str, ...]]:
    """Yield every legal board completion; useful for small deterministic tests."""

    missing = 5 - len(board_cards)
    for runout in combinations(tuple(remaining_cards), missing):
        yield (*board_cards, *runout)
