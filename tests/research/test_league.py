"""Tests for deterministic duplicate-poker policy leagues."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from pokerkit import HandHistory

from poker_research.arena import ArenaConfig
from poker_research.league import LeagueConfig, run_league
from poker_research.types import Decision, Observation
from tests.research.helpers import deterministic_decision


class _CheckCallPolicy:
    def __init__(self, name: str) -> None:
        self.name = name
        self.observations: list[Observation] = []

    def decide(self, observation: Observation) -> Decision:
        self.observations.append(observation)
        return deterministic_decision(observation, "check_call")


class _MinimumRaisePolicy(_CheckCallPolicy):
    def decide(self, observation: Observation) -> Decision:
        self.observations.append(observation)
        action = "raise_min" if observation.option("raise_min") is not None else "check_call"
        return deterministic_decision(observation, action)


class _FoldFacingBetPolicy(_CheckCallPolicy):
    def decide(self, observation: Observation) -> Decision:
        self.observations.append(observation)
        action = "fold" if observation.option("fold") is not None else "check_call"
        return deterministic_decision(observation, action)


class _ExplodingPolicy(_CheckCallPolicy):
    def decide(self, observation: Observation) -> Decision:
        raise RuntimeError(f"{self.name} exploded from seat {observation.seat}")


class _ExplodeFromBigBlindPolicy(_CheckCallPolicy):
    def decide(self, observation: Observation) -> Decision:
        if observation.seat == 0:
            raise RuntimeError("big-blind-only failure")
        return super().decide(observation)


class _ExplodeOnPairOnePolicy(_CheckCallPolicy):
    def decide(self, observation: Observation) -> Decision:
        if "-000001-" in observation.hand_id:
            raise RuntimeError("injected failure for pair one")
        return super().decide(observation)


def _config(**overrides: object) -> LeagueConfig:
    values: dict[str, object] = {
        "name": "pytest-league",
        "master_seed": 424242,
        "pair_count_per_matchup": 2,
        "practical_margin_bb100": 0.0,
        "bootstrap_resamples": 1_000,
        "bootstrap_seed": 1717,
        "game": ArenaConfig(starting_stack=20, small_blind=1, big_blind=2),
    }
    values.update(overrides)
    return LeagueConfig(**values)  # type: ignore[arg-type]


def _tracking_factories(
    names: tuple[str, ...],
) -> tuple[dict[str, object], Counter[str], dict[str, list[_CheckCallPolicy]]]:
    calls: Counter[str] = Counter()
    created: dict[str, list[_CheckCallPolicy]] = defaultdict(list)
    factories: dict[str, object] = {}
    for name in names:

        def factory(policy_name: str = name) -> _CheckCallPolicy:
            calls[policy_name] += 1
            policy = _CheckCallPolicy(policy_name)
            created[policy_name].append(policy)
            return policy

        factories[name] = factory
    return factories, calls, created


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_round_robin_smoke_uses_fresh_factories_and_writes_artifacts(tmp_path: Path) -> None:
    factories, calls, created = _tracking_factories(("gamma", "alpha", "beta"))
    output = tmp_path / "league"

    result = run_league(  # type: ignore[arg-type]
        factories,
        config=_config(),
        output_directory=output,
    )

    assert result.status == "complete"
    assert len(result.matchups) == 3
    assert len(result.pairs) == 6
    assert result.errors == ()
    assert [(matchup.policy_a, matchup.policy_b) for matchup in result.matchups] == [
        ("alpha", "beta"),
        ("alpha", "gamma"),
        ("beta", "gamma"),
    ]
    assert len({pair.deal_seed for pair in result.pairs}) == 6
    assert all(pair.policy_a_bb100 == 0.0 for pair in result.pairs)
    for matchup in result.matchups:
        assert matchup.status == "complete"
        assert matchup.successful_pair_count == 2
        assert matchup.failed_pair_count == 0
        assert matchup.summary is not None
        assert matchup.summary.mean_bb100 == 0.0
        assert matchup.summary.two_sided_ci95 == (0.0, 0.0)

    assert calls == Counter({"alpha": 8, "beta": 8, "gamma": 8})
    for policies in created.values():
        assert len({id(policy) for policy in policies}) == 8
        assert all(policy.observations for policy in policies)
        assert Counter(policy.observations[0].seat for policy in policies) == Counter({0: 4, 1: 4})

    assert [entry.policy for entry in result.leaderboard] == ["alpha", "beta", "gamma"]
    for rank, entry in enumerate(result.leaderboard, start=1):
        assert entry.rank == rank
        assert entry.scored_matchups == 2
        assert entry.matchup_draws == 2
        assert entry.matchup_wins == entry.matchup_losses == 0
        assert entry.successful_pair_count == 4
        assert entry.failed_pair_count == 0
        assert entry.summary is not None
        assert entry.summary.mean_bb100 == 0.0

    assert {path.name for path in output.iterdir()} == {
        "decisions.jsonl",
        "errors.jsonl",
        "hands.csv",
        "hands.phhs",
        "leaderboard.csv",
        "manifest.json",
        "matchups.csv",
        "pairs.csv",
        "summary.json",
    }
    assert (output / "errors.jsonl").read_text(encoding="utf-8") == ""
    with (output / "pairs.csv").open(encoding="utf-8", newline="") as csv_file:
        assert len(list(csv.DictReader(csv_file))) == 6
    with (output / "hands.csv").open(encoding="utf-8", newline="") as csv_file:
        hand_rows = list(csv.DictReader(csv_file))
    assert len(hand_rows) == 12
    assert all(row["pair_status"] == "complete_pair" for row in hand_rows)
    assert all(row["policy_rng_key"] for row in hand_rows)
    with (output / "hands.phhs").open("rb") as phh_file:
        assert len(list(HandHistory.load_all(phh_file))) == 12
    decision_rows = [
        json.loads(line)
        for line in (output / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert decision_rows
    assert all(
        row["decision"]["action"]
        in {option["name"] for option in row["legal_actions"]}
        for row in decision_rows
    )
    with (output / "matchups.csv").open(encoding="utf-8", newline="") as csv_file:
        matchup_rows = list(csv.DictReader(csv_file))
    assert len(matchup_rows) == 3
    assert all(row["mean_bb100"] == "0.0" for row in matchup_rows)
    assert "one_sided_p_value_vs_zero" not in matchup_rows[0]
    assert "significant_win" not in matchup_rows[0]
    with (output / "leaderboard.csv").open(encoding="utf-8", newline="") as csv_file:
        assert [row["policy"] for row in csv.DictReader(csv_file)] == [
            "alpha",
            "beta",
            "gamma",
        ]

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate_method"] == (
        "pooled_duplicate_pairs_equal_pair_weight_complete_matchups_only"
    )
    assert summary["partial_matchups_excluded_from_all_statistics"] is True
    assert summary["inferential_statistics"] == (
        "not_reported_exploratory_round_robin"
    )
    serialized_summary = summary["leaderboard"][0]["summary"]
    assert "one_sided_p_value_vs_zero" not in serialized_summary
    assert "significant_win" not in serialized_summary
    assert summary["successful_pair_count"] == 6
    assert summary["error_count"] == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for artifact, digest in manifest["artifacts"].items():
        assert digest == _sha256(output / artifact)


def test_duplicate_pair_perspective_and_leaderboard_are_antisymmetric(
    tmp_path: Path,
) -> None:
    result = run_league(
        {
            "aggressive": lambda: _MinimumRaisePolicy("aggressive"),
            "folder": lambda: _FoldFacingBetPolicy("folder"),
        },
        config=_config(pair_count_per_matchup=3),
        output_directory=tmp_path / "heads-up",
    )

    assert len(result.pairs) == 3
    assert all(pair.policy_a_payoff_leg_a == 1 for pair in result.pairs)
    assert all(pair.policy_a_payoff_leg_b == 2 for pair in result.pairs)
    assert all(pair.policy_a_net_chips == 3 for pair in result.pairs)
    assert all(pair.policy_a_bb100 == 75.0 for pair in result.pairs)
    matchup = result.matchups[0]
    assert matchup.summary is not None
    assert matchup.summary.mean_bb100 == 75.0
    assert [entry.policy for entry in result.leaderboard] == ["aggressive", "folder"]
    assert result.leaderboard[0].summary is not None
    assert result.leaderboard[1].summary is not None
    assert result.leaderboard[0].summary.mean_bb100 == 75.0
    assert result.leaderboard[1].summary.mean_bb100 == -75.0
    assert result.leaderboard[0].matchup_wins == 1
    assert result.leaderboard[1].matchup_losses == 1


def test_mapping_insertion_order_does_not_change_schedule_or_results(tmp_path: Path) -> None:
    forward, _, _ = _tracking_factories(("alpha", "beta", "gamma"))
    reverse, _, _ = _tracking_factories(("gamma", "beta", "alpha"))

    first = run_league(  # type: ignore[arg-type]
        forward,
        config=_config(),
        output_directory=tmp_path / "first",
    )
    replay = run_league(  # type: ignore[arg-type]
        reverse,
        config=_config(),
        output_directory=tmp_path / "replay",
    )

    assert first.run_id == replay.run_id
    assert first.pairs == replay.pairs
    assert first.matchups == replay.matchups
    assert first.leaderboard == replay.leaderboard


def test_progress_reports_every_attempt_in_schedule_order(tmp_path: Path) -> None:
    updates: list[tuple[int, int, str, str]] = []
    result = run_league(
        {
            "alpha": lambda: _CheckCallPolicy("alpha"),
            "beta": lambda: _CheckCallPolicy("beta"),
            "gamma": lambda: _CheckCallPolicy("gamma"),
        },
        config=_config(pair_count_per_matchup=2),
        output_directory=tmp_path / "progress",
        progress=lambda done, total, first, second: updates.append(
            (done, total, first, second)
        ),
    )

    assert result.status == "complete"
    assert [update[0] for update in updates] == list(range(1, 7))
    assert {update[1] for update in updates} == {6}
    assert updates[:2] == [(1, 6, "alpha", "beta"), (2, 6, "alpha", "beta")]


def test_policy_failures_are_isolated_and_other_matchups_finish(tmp_path: Path) -> None:
    result = run_league(
        {
            "alpha": lambda: _CheckCallPolicy("alpha"),
            "beta": lambda: _CheckCallPolicy("beta"),
            "broken": lambda: _ExplodingPolicy("broken"),
        },
        config=_config(),
        output_directory=tmp_path / "isolated",
    )

    assert result.status == "complete_with_errors"
    assert len(result.pairs) == 2
    assert len(result.errors) == 4
    assert {(pair.policy_a, pair.policy_b) for pair in result.pairs} == {("alpha", "beta")}
    assert {error.error_type for error in result.errors} == {"RuntimeError"}
    assert {error.failed_leg for error in result.errors} == {"a"}
    statuses = {
        (matchup.policy_a, matchup.policy_b): matchup.status for matchup in result.matchups
    }
    assert statuses == {
        ("alpha", "beta"): "complete",
        ("alpha", "broken"): "insufficient_data",
        ("beta", "broken"): "insufficient_data",
    }
    entries = {entry.policy: entry for entry in result.leaderboard}
    assert entries["alpha"].successful_pair_count == 2
    assert entries["alpha"].failed_pair_count == 2
    assert entries["beta"].successful_pair_count == 2
    assert entries["broken"].successful_pair_count == 0
    assert entries["broken"].failed_pair_count == 4
    assert entries["broken"].summary is None

    error_rows = [
        json.loads(line)
        for line in (tmp_path / "isolated" / "errors.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(error_rows) == 4
    manifest = json.loads(
        (tmp_path / "isolated" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete_with_errors"


def test_partial_matchup_is_not_summarized_or_ranked_from_survivors(
    tmp_path: Path,
) -> None:
    result = run_league(
        {
            "alpha": lambda: _CheckCallPolicy("alpha"),
            "fragile": lambda: _ExplodeOnPairOnePolicy("fragile"),
        },
        config=_config(pair_count_per_matchup=3),
        output_directory=tmp_path / "partial-invalid",
    )

    assert len(result.pairs) == 2
    assert len(result.errors) == 1
    matchup = result.matchups[0]
    assert matchup.status == "partial"
    assert matchup.successful_pair_count == 2
    assert matchup.failed_pair_count == 1
    assert matchup.summary is None
    for entry in result.leaderboard:
        assert entry.scored_matchups == 0
        assert entry.successful_pair_count == 0
        assert entry.failed_pair_count == 3
        assert entry.summary is None


def test_failure_in_second_duplicate_leg_is_attributed_correctly(tmp_path: Path) -> None:
    result = run_league(
        {
            "alpha": lambda: _CheckCallPolicy("alpha"),
            "seat_sensitive": lambda: _ExplodeFromBigBlindPolicy("seat_sensitive"),
        },
        config=_config(),
        output_directory=tmp_path / "leg-b",
    )

    assert result.pairs == ()
    assert len(result.errors) == 2
    assert {error.failed_leg for error in result.errors} == {"b"}
    assert all(error.message == "big-blind-only failure" for error in result.errors)


def test_factory_name_mismatch_becomes_isolated_pair_error(tmp_path: Path) -> None:
    result = run_league(
        {
            "alpha": lambda: _CheckCallPolicy("alpha"),
            "declared_name": lambda: _CheckCallPolicy("different_name"),
        },
        config=_config(),
        output_directory=tmp_path / "bad-name",
    )

    assert result.status == "complete_with_errors"
    assert result.pairs == ()
    assert len(result.errors) == 2
    assert all(error.error_type == "ValueError" for error in result.errors)
    assert all("returned policy named" in error.message for error in result.errors)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": ""}, "league name"),
        ({"name": "contains whitespace"}, "league name"),
        ({"pair_count_per_matchup": 1}, "at least two"),
        ({"bootstrap_resamples": 999}, "at least 1000"),
        ({"policy_schedule_seed": -1}, "policy_schedule_seed"),
        ({"practical_margin_bb100": float("nan")}, "finite"),
    ],
)
def test_invalid_league_config_is_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides).validate()


def test_league_validates_factory_mapping_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="at least two"):
        run_league(
            {"only": lambda: _CheckCallPolicy("only")},
            config=_config(),
            output_directory=output,
        )
    assert not output.exists()

    with pytest.raises(ValueError, match="non-empty strings"):
        run_league(
            {
                "": lambda: _CheckCallPolicy(""),
                "valid": lambda: _CheckCallPolicy("valid"),
            },
            config=_config(),
            output_directory=output,
        )


def test_league_refuses_to_overwrite_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    keep = output / "keep.txt"
    keep.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_league(
            {
                "alpha": lambda: _CheckCallPolicy("alpha"),
                "beta": lambda: _CheckCallPolicy("beta"),
            },
            config=_config(),
            output_directory=output,
        )

    assert keep.read_text(encoding="utf-8") == "keep me"
