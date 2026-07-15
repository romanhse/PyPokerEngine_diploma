"""Tests for duplicate-pair confidence intervals and power calculations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from poker_research.statistics import required_pair_count, summarize_duplicate_pairs


def test_summary_uses_pair_level_sample_statistics_and_one_sided_gate() -> None:
    summary = summarize_duplicate_pairs(
        [10.0, 20.0, 30.0, 40.0],
        practical_margin_bb100=5.0,
        bootstrap_resamples=2_000,
        bootstrap_seed=13,
    )

    assert summary.pair_count == 4
    assert summary.mean_bb100 == 25.0
    assert summary.std_bb100 == pytest.approx(12.9099444874)
    assert summary.standard_error == pytest.approx(6.4549722437)
    assert summary.two_sided_ci95 == pytest.approx((4.4573974324, 45.5426025676))
    assert summary.one_sided_lower95 == pytest.approx(9.8091043491)
    assert summary.t_statistic_vs_zero == pytest.approx(3.872983346)
    assert summary.one_sided_p_value_vs_zero == pytest.approx(0.0152331458)
    assert summary.significant_win is True
    assert summary.clears_practical_margin is True


def test_practical_margin_is_stricter_than_statistical_significance() -> None:
    summary = summarize_duplicate_pairs(
        [10.0, 20.0, 30.0, 40.0],
        practical_margin_bb100=10.0,
        bootstrap_resamples=1_000,
    )

    assert summary.significant_win is True
    assert summary.clears_practical_margin is False
    assert summary.one_sided_lower95_minus_margin < 0


@pytest.mark.parametrize(
    ("values", "expected_t", "expected_p", "significant"),
    [
        ([3.0, 3.0, 3.0], math.inf, 0.0, True),
        ([0.0, 0.0, 0.0], 0.0, 0.5, False),
        ([-3.0, -3.0, -3.0], -math.inf, 1.0, False),
    ],
)
def test_zero_variance_samples_have_explicit_inference_semantics(
    values: list[float], expected_t: float, expected_p: float, significant: bool
) -> None:
    summary = summarize_duplicate_pairs(values, bootstrap_resamples=1_000)

    assert summary.t_statistic_vs_zero == expected_t
    assert summary.one_sided_p_value_vs_zero == expected_p
    assert summary.significant_win is significant
    assert summary.bootstrap_ci95 == (values[0], values[0])


def test_bootstrap_interval_is_seed_reproducible() -> None:
    first = summarize_duplicate_pairs(
        [-8.0, -1.0, 3.0, 7.0, 15.0],
        bootstrap_resamples=3_000,
        bootstrap_seed=999,
    )
    replay = summarize_duplicate_pairs(
        [-8.0, -1.0, 3.0, 7.0, 15.0],
        bootstrap_resamples=3_000,
        bootstrap_seed=999,
    )

    assert first.bootstrap_ci95 == replay.bootstrap_ci95
    assert first.bootstrap_ci95[0] <= first.mean_bb100 <= first.bootstrap_ci95[1]


@pytest.mark.parametrize(
    ("values", "resamples", "message"),
    [
        ([1.0], 1_000, "at least two"),
        ([[1.0, 2.0], [3.0, 4.0]], 1_000, "one-dimensional"),
        ([1.0, float("nan")], 1_000, "finite"),
        ([1.0, float("inf")], 1_000, "finite"),
        ([1.0, 2.0], 999, "at least 1000"),
    ],
)
def test_summary_rejects_invalid_samples(
    values: list[float] | list[list[float]], resamples: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_duplicate_pairs(  # type: ignore[arg-type]
            values,
            bootstrap_resamples=resamples,
        )


def test_required_pair_count_matches_preregistered_normal_approximation() -> None:
    assert required_pair_count(100.0, 10.0, alpha=0.05, power=0.90) == 857
    assert required_pair_count(100.0, 20.0, alpha=0.05, power=0.90) < 857
    assert required_pair_count(120.0, 10.0, alpha=0.05, power=0.90) > 857


@pytest.mark.parametrize(
    ("standard_deviation", "effect", "alpha", "power"),
    [
        (0.0, 10.0, 0.05, 0.90),
        (-1.0, 10.0, 0.05, 0.90),
        (100.0, 0.0, 0.05, 0.90),
        (100.0, -1.0, 0.05, 0.90),
        (100.0, 10.0, 0.0, 0.90),
        (100.0, 10.0, 0.5, 0.90),
        (100.0, 10.0, 0.05, 0.5),
        (100.0, 10.0, 0.05, 1.0),
    ],
)
def test_required_pair_count_rejects_invalid_design_inputs(
    standard_deviation: float, effect: float, alpha: float, power: float
) -> None:
    with pytest.raises(ValueError):
        required_pair_count(
            standard_deviation,
            effect,
            alpha=alpha,
            power=power,
        )


def test_summary_accepts_numpy_sequence_without_copy_semantics_assumptions() -> None:
    values = np.array([1.0, 2.0, 3.0])
    summary = summarize_duplicate_pairs(values, bootstrap_resamples=1_000)

    assert summary.mean_bb100 == 2.0
