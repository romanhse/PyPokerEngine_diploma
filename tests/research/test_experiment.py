"""End-to-end tests for preregistered duplicate-poker experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest
from pokerkit import HandHistory

import poker_research.experiment as experiment_module
import poker_research.policies as policies_module
from poker_research.arena import ArenaConfig
from poker_research.experiment import (
    BenchmarkConfig,
    load_benchmark_config,
    run_benchmark,
)
from poker_research.neural import EncoderConfig, MLPWeights, ObservationEncoder, save_checkpoint
from poker_research.policies import EquityValueConfig


def _benchmark_config(**overrides: object) -> BenchmarkConfig:
    values: dict[str, object] = {
        "name": "pytest-duplicate-benchmark",
        "master_seed": 20260715,
        "pair_count": 2,
        "practical_margin_bb100": 0.0,
        "bootstrap_resamples": 1_000,
        "bootstrap_seed": 707,
        "game": ArenaConfig(starting_stack=40, small_blind=1, big_blind=2),
        "hero_policy": "equity_value_v1",
        "opponent_policy": "calling_station_v1",
        "hero_seed": 19,
        "equity_value": EquityValueConfig(preflop_samples=1, postflop_samples=1),
    }
    values.update(overrides)
    return BenchmarkConfig(**values)  # type: ignore[arg-type]


def _write_config(path: Path) -> None:
    path.write_text(
        """
[experiment]
name = "pytest-duplicate-benchmark"
mode = "debug"
master_seed = 20260715
pair_count = 2
practical_margin_bb100 = 0.0
bootstrap_resamples = 1000
bootstrap_seed = 707
policy_schedule_seed = 20260717

[game]
starting_stack = 40
small_blind = 1
big_blind = 2
ante = 0

[hero]
policy = "equity_value_v1"
seed = 19

[opponent]
policy = "calling_station_v1"

[equity_value]
preflop_samples = 1
postflop_samples = 1
""".lstrip(),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_benchmark_config_parses_frozen_protocol(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.toml"
    _write_config(config_path)

    config = load_benchmark_config(config_path)

    assert config.name == "pytest-duplicate-benchmark"
    assert config.mode == "debug"
    assert config.pair_count == 2
    assert config.game == ArenaConfig(starting_stack=40, small_blind=1, big_blind=2)
    assert config.hero_policy == "equity_value_v1"
    assert config.opponent_policy == "calling_station_v1"
    assert config.opponent_seed == 0
    assert config.hero_checkpoint is None
    assert config.opponent_checkpoint is None
    assert config.equity_value.preflop_samples == 1
    assert config.equity_value.postflop_samples == 1
    assert config.equity_value.call_margin == 0.015


def test_direct_benchmark_config_keeps_debug_mode_default() -> None:
    assert _benchmark_config().mode == "debug"


def test_toml_requires_explicit_valid_experiment_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmark.toml"
    _write_config(config_path)
    without_mode = config_path.read_text(encoding="utf-8").replace(
        'mode = "debug"\n', ""
    )
    config_path.write_text(without_mode, encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing required \[experiment\]\.mode"):
        load_benchmark_config(config_path)

    config_path.write_text(
        without_mode.replace(
            'name = "pytest-duplicate-benchmark"\n',
            'name = "pytest-duplicate-benchmark"\nmode = "publication"\n',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="debug.*pilot.*confirmatory"):
        load_benchmark_config(config_path)


@pytest.mark.parametrize(
    ("table", "location"),
    [
        ("[experiment]", r"\[experiment\]"),
        ("[game]", r"\[game\]"),
        ("[hero]", r"\[hero\]"),
        ("[opponent]", r"\[opponent\]"),
        ("[equity_value]", r"\[equity_value\]"),
    ],
)
def test_toml_rejects_unknown_section_keys(
    tmp_path: Path,
    table: str,
    location: str,
) -> None:
    config_path = tmp_path / "benchmark.toml"
    _write_config(config_path)
    source = config_path.read_text(encoding="utf-8").replace(
        f"{table}\n",
        f"{table}\nmisspelled_option = 1\n",
        1,
    )
    config_path.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match=rf"unknown.*{location}.*misspelled_option"):
        load_benchmark_config(config_path)


def test_toml_rejects_unknown_top_level_section_and_integer_coercion(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "benchmark.toml"
    _write_config(config_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n[unregistered]\nvalue = 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown.*top level.*unregistered"):
        load_benchmark_config(config_path)

    _write_config(config_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "pair_count = 2", "pair_count = 2.5"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="experiment.pair_count must be an integer"):
        load_benchmark_config(config_path)


@pytest.mark.parametrize(
    "critical_key",
    [
        "practical_margin_bb100",
        "bootstrap_resamples",
        "bootstrap_seed",
        "policy_schedule_seed",
    ],
)
def test_confirmatory_toml_requires_explicit_critical_fields(
    tmp_path: Path,
    critical_key: str,
) -> None:
    config_path = tmp_path / "confirmatory.toml"
    _write_config(config_path)
    lines = config_path.read_text(encoding="utf-8").replace(
        'mode = "debug"', 'mode = "confirmatory"'
    ).splitlines()
    source = "\n".join(line for line in lines if not line.startswith(f"{critical_key} ="))
    config_path.write_text(source + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"confirmatory.*explicitly set.*{critical_key}"):
        load_benchmark_config(config_path)


def test_repository_preregistrations_use_explicit_modes() -> None:
    config_directory = Path(__file__).resolve().parents[2] / "configs"
    pilot = load_benchmark_config(config_directory / "pilot_equity_vs_calling_station.toml")
    confirmatory = load_benchmark_config(
        config_directory / "confirmatory_equity_vs_calling_station.toml"
    )

    assert pilot.mode == "pilot"
    assert confirmatory.mode == "confirmatory"


def test_generic_catalog_policies_can_use_confirmatory_runner(tmp_path: Path) -> None:
    config = _benchmark_config(
        hero_policy="tag_v1",
        opponent_policy="loose_passive_v1",
        opponent_seed=23,
    )

    result = run_benchmark(config, output_directory=tmp_path / "generic")

    assert len(result.pairs) == 2
    assert result.summary.pair_count == 2
    assert (tmp_path / "generic" / "manifest.json").is_file()


def test_safe_checkpoint_is_hashed_copied_and_evaluated(tmp_path: Path) -> None:
    encoder = ObservationEncoder(
        EncoderConfig(starting_stack=40, big_blind=2, equity_samples=0)
    )
    checkpoint = tmp_path / "trained.npz"
    save_checkpoint(
        checkpoint,
        MLPWeights.random(encoder.feature_count, hidden_size=8, seed=1),
        encoder,
        policy_name="trained_policy_v1",
    )
    config = _benchmark_config(
        hero_policy="trained_policy_v1",
        hero_checkpoint=str(checkpoint),
    )
    output = tmp_path / "checkpoint-run"

    result = run_benchmark(config, output_directory=output)

    assert len(result.pairs) == 2
    copied = output / "hero_checkpoint.npz"
    assert copied.read_bytes() == checkpoint.read_bytes()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy_checkpoints"]["hero"]["sha256"] == _sha256(checkpoint)
    assert manifest["artifacts"]["hero_checkpoint.npz"] == _sha256(copied)


def test_checkpoint_policy_name_must_match_config(tmp_path: Path) -> None:
    encoder = ObservationEncoder(
        EncoderConfig(starting_stack=40, big_blind=2, equity_samples=0)
    )
    checkpoint = tmp_path / "wrong-name.npz"
    save_checkpoint(
        checkpoint,
        MLPWeights.random(encoder.feature_count, hidden_size=8),
        encoder,
        policy_name="inside_checkpoint_v1",
    )
    config = _benchmark_config(
        hero_policy="configured_name_v1",
        hero_checkpoint=str(checkpoint),
    )

    with pytest.raises(ValueError, match="does not match"):
        run_benchmark(config, output_directory=tmp_path / "not-created")
    failure = json.loads(
        (tmp_path / "not-created" / "failed_run.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert failure["completed_pairs"] == 0


def test_tiny_benchmark_writes_replayable_duplicate_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "benchmark.toml"
    output = tmp_path / "artifacts"
    _write_config(config_path)
    config = load_benchmark_config(config_path)
    monkeypatch.setattr(policies_module, "equity_vs_uniform", lambda *args, **kwargs: 0.65)
    progress: list[tuple[int, int]] = []

    result = run_benchmark(
        config,
        output_directory=output,
        config_path=config_path,
        progress=lambda done, total: progress.append((done, total)),
    )

    expected_files = {
        "decisions.jsonl",
        "errors.jsonl",
        "hands.csv",
        "hands.phhs",
        "manifest.json",
        "pairs.csv",
        "preregistered_config.toml",
        "summary.json",
    }
    assert {path.name for path in output.iterdir()} == expected_files
    assert progress == [(1, 2), (2, 2)]
    assert result.output_directory == output
    assert result.summary.pair_count == 2
    assert len(result.pairs) == 2
    assert (output / "preregistered_config.toml").read_bytes() == config_path.read_bytes()
    assert (output / "errors.jsonl").read_text(encoding="utf-8") == ""

    with (output / "hands.csv").open(encoding="utf-8", newline="") as csv_file:
        hand_rows = list(csv.DictReader(csv_file))
    assert len(hand_rows) == 4
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in hand_rows:
        grouped[row["pair_id"]].append(row)
        assert int(row["seat0_payoff"]) + int(row["seat1_payoff"]) == 0
        assert row["status"] == "complete"
        assert row["starting_stack"] == "40"
        assert row["big_blind"] == "2"
        assert len(row["seat0_hole"].split()) == 2
        assert len(row["seat1_hole"].split()) == 2
    for legs in grouped.values():
        assert [leg["leg"] for leg in legs] == ["a", "b"]
        assert legs[0]["deal_seed"] == legs[1]["deal_seed"]
        assert legs[0]["deck_hash"] == legs[1]["deck_hash"]
        assert (legs[0]["seat0_policy"], legs[0]["seat1_policy"]) == (
            "equity_value_v1",
            "calling_station_v1",
        )
        assert (legs[1]["seat0_policy"], legs[1]["seat1_policy"]) == (
            "calling_station_v1",
            "equity_value_v1",
        )
        assert (legs[0]["hero_seat"], legs[1]["hero_seat"]) == ("0", "1")

    decision_rows = [
        json.loads(line)
        for line in (output / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert decision_rows
    for row in decision_rows:
        assert row["decision"]["action"] in {
            action["name"] for action in row["legal_actions"]
        }
        assert sum(probability for _, probability in row["decision"]["probabilities"]) == 1.0

    with (output / "hands.phhs").open("rb") as phh_file:
        hand_histories = list(HandHistory.load_all(phh_file))
    assert len(hand_histories) == 4

    summary_payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload == result.summary.to_dict()
    with (output / "pairs.csv").open(encoding="utf-8", newline="") as csv_file:
        pair_rows = list(csv.DictReader(csv_file))
    assert len(pair_rows) == 2
    assert [float(row["hero_bb100"]) for row in pair_rows] == [
        pair.hero_bb100 for pair in result.pairs
    ]

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["schema_version"] == 1
    assert manifest["config"] == {
        "bootstrap_resamples": 1_000,
        "bootstrap_seed": 707,
        "equity_value": {
            "call_margin": 0.015,
            "flop_raise_equity": 0.61,
            "nut_equity": 0.84,
            "postflop_samples": 1,
            "preflop_raise_equity": 0.60,
            "preflop_samples": 1,
            "river_raise_equity": 0.53,
            "strong_equity": 0.70,
            "turn_raise_equity": 0.59,
        },
        "game": {
            "ante": 0,
            "big_blind": 2,
            "small_blind": 1,
            "starting_stack": 40,
        },
        "hero_checkpoint": None,
        "hero_policy": "equity_value_v1",
        "hero_seed": 19,
        "master_seed": 20260715,
        "mode": "debug",
        "name": "pytest-duplicate-benchmark",
        "opponent_checkpoint": None,
        "opponent_policy": "calling_station_v1",
        "opponent_seed": 0,
        "pair_count": 2,
        "policy_schedule_seed": 20_260_717,
        "practical_margin_bb100": 0.0,
    }
    assert len(manifest["source"]["tree_sha256"]) == 64
    assert manifest["source"]["files"].keys() >= {
        "poker_research/arena.py",
        "poker_research/policies.py",
        "pyproject.toml",
        "uv.lock",
    }
    assert "preregistered_config.toml" in manifest["artifacts"]
    for artifact, digest in manifest["artifacts"].items():
        assert digest == _sha256(output / artifact)


def test_benchmark_refuses_to_overwrite_nonempty_directory(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_benchmark(_benchmark_config(), output_directory=output)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_analysis_only_config_changes_do_not_change_play(
    tmp_path: Path,
) -> None:
    first = _benchmark_config(
        name="sensitivity-a",
        hero_policy="random_valid_v1",
        pair_count=3,
        practical_margin_bb100=0.0,
        bootstrap_seed=1,
    )
    second = _benchmark_config(
        name="sensitivity-b",
        hero_policy="random_valid_v1",
        pair_count=3,
        practical_margin_bb100=25.0,
        bootstrap_seed=999,
    )

    first_result = run_benchmark(first, output_directory=tmp_path / "first")
    second_result = run_benchmark(second, output_directory=tmp_path / "second")

    assert first_result.pairs == second_result.pairs

    def decisions(directory: Path) -> list[tuple[object, ...]]:
        rows = [
            json.loads(line)
            for line in (directory / "decisions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        return [
            (
                row["pair_id"],
                row["leg"],
                row["seat"],
                row["street"],
                row["decision"],
            )
            for row in rows
        ]

    assert decisions(tmp_path / "first") == decisions(tmp_path / "second")


def test_failed_run_preserves_structured_error_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failed"

    def fail_hand(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected engine failure")

    monkeypatch.setattr(experiment_module, "play_hand", fail_hand)
    with pytest.raises(RuntimeError, match="injected engine failure"):
        run_benchmark(_benchmark_config(), output_directory=output)

    failed = json.loads((output / "failed_run.json").read_text(encoding="utf-8"))
    error = json.loads((output / "errors.jsonl").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["completed_pairs"] == 0
    assert error["error_type"] == "RuntimeError"
    assert error["message"] == "injected engine failure"


def test_post_play_failure_is_also_marked_as_failed_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failed-statistics"

    def fail_statistics(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected statistics failure")

    monkeypatch.setattr(
        experiment_module,
        "summarize_duplicate_pairs",
        fail_statistics,
    )
    with pytest.raises(RuntimeError, match="injected statistics failure"):
        run_benchmark(_benchmark_config(), output_directory=output)

    failed = json.loads((output / "failed_run.json").read_text(encoding="utf-8"))
    error = json.loads((output / "errors.jsonl").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["completed_pairs"] == 2
    assert error["error_type"] == "RuntimeError"
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": ""}, "name"),
        ({"name": "contains whitespace"}, "name"),
        ({"pair_count": 1}, "at least two"),
        ({"bootstrap_resamples": 999}, "at least 1000"),
        ({"mode": "publication"}, "mode"),
        ({"practical_margin_bb100": float("nan")}, "finite"),
        ({"bootstrap_seed": -1}, "bootstrap_seed"),
        ({"opponent_policy": "equity_value_v1"}, "must differ"),
    ],
)
def test_invalid_benchmark_protocol_is_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _benchmark_config(**overrides).validate()


def test_unknown_policy_is_rejected_and_failure_is_logged(tmp_path: Path) -> None:
    output = tmp_path / "unknown-policy"

    with pytest.raises(ValueError, match="unknown policy"):
        run_benchmark(
            _benchmark_config(hero_policy="not_registered"),
            output_directory=output,
        )

    error = json.loads((output / "errors.jsonl").read_text(encoding="utf-8"))
    assert error["error_type"] == "ValueError"
