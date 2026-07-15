from __future__ import annotations

import json
import sys

from poker_research.cli import build_parser, main
from poker_research.neural import (
    EncoderConfig,
    MLPWeights,
    NumpyMLPPolicy,
    ObservationEncoder,
    save_checkpoint,
)


def test_catalog_command_parses_named_suite() -> None:
    args = build_parser().parse_args(["catalog", "--suite", "quick"])

    assert args.command == "catalog"
    assert args.suite == "quick"


def test_league_command_has_resource_conscious_defaults(tmp_path) -> None:
    args = build_parser().parse_args(["league", "--output", str(tmp_path / "league")])

    assert args.command == "league"
    assert args.suite == "quick"
    assert args.pairs == 10
    assert args.bootstrap_resamples == 2_000
    assert args.starting_stack == 200
    assert args.big_blind == 2
    assert args.allow_partial_debug is False


def test_explicit_league_policy_list_overrides_suite_at_runtime_parser_level(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "league",
            "--output",
            str(tmp_path / "league"),
            "--suite",
            "extended",
            "--policies",
            "tag_v1",
            "maniac_v1",
        ]
    )

    assert args.policies == ["tag_v1", "maniac_v1"]
    assert args.checkpoint == []


def test_imitation_command_defaults_to_resource_conscious_mps_auto(tmp_path) -> None:
    args = build_parser().parse_args(
        ["imitate", "--checkpoint", str(tmp_path / "policy.npz")]
    )

    assert args.command == "imitate"
    assert args.teacher == "equity_value_v1"
    assert args.deals_per_opponent == 100
    assert args.epochs == 12
    assert args.device == "auto"
    assert args.equity_samples == 64


def test_imitation_rejects_report_path_that_would_replace_checkpoint(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "same.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "poker-benchmark",
            "imitate",
            "--checkpoint",
            str(path),
            "--report",
            str(path),
        ],
    )

    try:
        main()
    except ValueError as error:
        assert "must be different" in str(error)
    else:
        raise AssertionError("same checkpoint/report path was accepted")
    assert not path.exists()


def test_tiny_imitation_cli_writes_safe_checkpoint_and_report(
    tmp_path, monkeypatch, capsys
) -> None:
    checkpoint = tmp_path / "tiny.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "poker-benchmark",
            "imitate",
            "--teacher",
            "tag_v1",
            "--opponents",
            "calling_station_v1",
            "--deals-per-opponent",
            "2",
            "--checkpoint",
            str(checkpoint),
            "--policy-name",
            "tiny_bc_v1",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--hidden-size",
            "8",
            "--equity-samples",
            "0",
            "--device",
            "cpu",
        ],
    )

    main()

    report_path = checkpoint.with_suffix(".training.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "exploratory_training_complete"
    assert report["teacher"] == "tag_v1"
    assert report["demonstration_count"] > 0
    assert report["history"][-1]["validation_illegal_predictions"] == 0
    assert NumpyMLPPolicy.from_checkpoint(checkpoint).name == "tiny_bc_v1"
    assert json.loads(capsys.readouterr().out)["checkpoint_sha256"] == report[
        "checkpoint_sha256"
    ]


def test_train_dqn_parser_has_m3_friendly_but_bounded_defaults(tmp_path) -> None:
    args = build_parser().parse_args(
        ["train-dqn", "--output", str(tmp_path / "dqn")]
    )

    assert args.command == "train-dqn"
    assert args.training_hands == 5_000
    assert args.replay_capacity == 100_000
    assert args.device == "auto"
    assert args.equity_samples == 32
    assert args.initial_checkpoint is None
    assert args.encoder_checkpoint is None


def test_train_dqn_rejects_two_initialization_sources(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "poker-benchmark",
            "train-dqn",
            "--output",
            str(output),
            "--initial-checkpoint",
            str(tmp_path / "weights.npz"),
            "--encoder-checkpoint",
            str(tmp_path / "encoder.npz"),
        ],
    )

    try:
        main()
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("expected mutually exclusive initialization sources to fail")
    assert not output.exists()


def test_tiny_train_dqn_cli_exports_resume_and_inference_artifacts(
    tmp_path, monkeypatch, capsys
) -> None:
    output = tmp_path / "dqn"
    encoder = ObservationEncoder(EncoderConfig(equity_samples=0))
    encoder_checkpoint = tmp_path / "encoder.npz"
    save_checkpoint(
        encoder_checkpoint,
        MLPWeights.random(encoder.feature_count, hidden_size=8, seed=11),
        encoder,
        policy_name="encoder_source_v1",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "poker-benchmark",
            "train-dqn",
            "--output",
            str(output),
            "--encoder-checkpoint",
            str(encoder_checkpoint),
            "--opponents",
            "calling_station_v1",
            "--training-hands",
            "2",
            "--validation-hands",
            "2",
            "--hidden-size",
            "8",
            "--batch-size",
            "1",
            "--replay-capacity",
            "32",
            "--min-replay-size",
            "1",
            "--updates-per-transition",
            "1",
            "--target-sync-interval",
            "2",
            "--epsilon-decay-decisions",
            "10",
            "--equity-samples",
            "0",
            "--device",
            "cpu",
            "--progress-every",
            "0",
        ],
    )

    main()

    report = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert report["status"] == "exploratory_training_complete"
    assert report["training_summary"]["hand_count"] == 2
    assert report["validation_summary"]["hand_count"] == 2
    assert report["optimization_steps"] > 0
    assert report["initial_checkpoint"] is None
    assert report["encoder_checkpoint"] == str(encoder_checkpoint)
    assert report["encoder_checkpoint_sha256"] == NumpyMLPPolicy.from_checkpoint(
        encoder_checkpoint
    ).checkpoint_sha256
    assert (output / "trainer.pt").is_file()
    assert NumpyMLPPolicy.from_checkpoint(output / "policy.npz").name == "double_dqn_v1"
    assert "trained 2/2 hands" in capsys.readouterr().out
