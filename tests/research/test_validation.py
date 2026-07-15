"""Tests for independent validation of completed run directories."""

from __future__ import annotations

from pathlib import Path

import pytest

import poker_research.policies as policies_module
from poker_research.arena import ArenaConfig
from poker_research.experiment import BenchmarkConfig, run_benchmark
from poker_research.policies import EquityValueConfig
from poker_research.validation import validate_run


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="validator-e2e",
        master_seed=7719,
        pair_count=2,
        practical_margin_bb100=0.0,
        bootstrap_resamples=1_000,
        bootstrap_seed=13,
        game=ArenaConfig(starting_stack=40, small_blind=1, big_blind=2),
        hero_policy="equity_value_v1",
        opponent_policy="calling_station_v1",
        hero_seed=5,
        equity_value=EquityValueConfig(preflop_samples=1, postflop_samples=1),
    )


def test_validate_run_accepts_complete_auditable_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(policies_module, "equity_vs_uniform", lambda *args, **kwargs: 0.65)
    output = tmp_path / "run"
    config_path = tmp_path / "frozen.toml"
    config_path.write_text("# frozen test config\n", encoding="utf-8")
    run_benchmark(_config(), output_directory=output, config_path=config_path)

    report = validate_run(output)

    assert report.passed is True
    assert all(check.passed for check in report.checks)
    assert {check.name for check in report.checks} >= {
        "artifact_hashes",
        "decision_contract",
        "duplicate_pairing",
        "pair_payoff_formula",
        "phh_count",
        "zero_sum_accounting",
    }


def test_validate_run_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(policies_module, "equity_vs_uniform", lambda *args, **kwargs: 0.65)
    output = tmp_path / "run"
    config_path = tmp_path / "frozen.toml"
    config_path.write_text("# frozen test config\n", encoding="utf-8")
    run_benchmark(_config(), output_directory=output, config_path=config_path)
    with (output / "pairs.csv").open("a", encoding="utf-8") as pairs:
        pairs.write("\n")

    report = validate_run(output)

    assert report.passed is False
    hash_check = next(check for check in report.checks if check.name == "artifact_hashes")
    assert hash_check.passed is False
    assert "pairs.csv" in hash_check.detail


def test_validate_run_reports_missing_files(tmp_path: Path) -> None:
    report = validate_run(tmp_path)

    assert report.passed is False
    assert report.checks[0].name == "required_files"
    assert report.checks[0].passed is False
