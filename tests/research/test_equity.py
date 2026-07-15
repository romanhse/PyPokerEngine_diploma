"""Tests for deterministic PHEvaluator-backed equity utilities."""

from __future__ import annotations

from itertools import combinations

import pytest

from poker_research.equity import (
    STANDARD_DECK,
    equity_vs_uniform,
    iter_complete_boards,
    stable_seed,
)


def test_royal_flush_on_board_is_an_exact_tie_against_every_range_combo() -> None:
    equity = equity_vs_uniform(
        ("2c", "3d"),
        ("Ah", "Kh", "Qh", "Jh", "Th"),
        sample_count=1,
        seed=7,
    )

    assert equity == 0.5


def test_monte_carlo_equity_is_reproducible_and_order_normalized() -> None:
    first = equity_vs_uniform(
        ("As", "Kd"),
        ("2c", "7h", "Tc"),
        sample_count=200,
        seed=12345,
    )
    replay = equity_vs_uniform(
        ("Kd", "As"),
        ("Tc", "2c", "7h"),
        sample_count=200,
        seed=12345,
    )

    assert first == replay
    assert 0.0 <= first <= 1.0


def test_different_seed_changes_a_small_monte_carlo_sample() -> None:
    first = equity_vs_uniform(
        ("9s", "8s"),
        (),
        sample_count=31,
        seed=1,
    )
    second = equity_vs_uniform(
        ("9s", "8s"),
        (),
        sample_count=31,
        seed=2,
    )

    assert first != second


@pytest.mark.parametrize(
    ("hole_cards", "board_cards", "sample_count", "message"),
    [
        (("As",), (), 10, "exactly two"),
        (("As", "Kd"), ("2c",), 10, "0, 3, 4, or 5"),
        (("As", "As"), (), 10, "duplicate"),
        (("As", "Kd"), ("As", "2c", "3d"), 10, "duplicate"),
        (("1s", "Kd"), (), 10, "invalid cards"),
        (("As", "Kd"), (), 0, "positive"),
    ],
)
def test_invalid_equity_inputs_are_rejected(
    hole_cards: tuple[str, ...],
    board_cards: tuple[str, ...],
    sample_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        equity_vs_uniform(  # type: ignore[arg-type]
            hole_cards,
            board_cards,
            sample_count=sample_count,
            seed=0,
        )


def test_complete_board_iterator_is_exhaustive_without_mutating_prefix() -> None:
    board = ("As", "Kd", "Qc")
    remaining = ("2c", "3c", "4c", "5c")
    completions = tuple(iter_complete_boards(board, remaining))

    assert len(completions) == len(tuple(combinations(remaining, 2))) == 6
    assert all(completion[:3] == board for completion in completions)
    assert len(set(completions)) == len(completions)


def test_standard_deck_and_stable_seed_invariants() -> None:
    assert len(STANDARD_DECK) == 52
    assert len(set(STANDARD_DECK)) == 52
    assert stable_seed(12, "hand", 4) == stable_seed(12, "hand", 4)
    assert stable_seed(12, "hand", 4) != stable_seed(12, "hand", 5)
    assert 0 <= stable_seed(12, "hand", 4) < 2**64
