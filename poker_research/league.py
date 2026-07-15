"""Deterministic round-robin leagues evaluated with duplicate poker.

Every policy is supplied as a factory and instantiated independently for each
leg.  A matchup observation is one duplicate pair (the same ordered deck with
the policies swapping seats), never an individual hand.  League statistics are
descriptive; confirmatory claims should still be preregistered separately.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, TypeAlias

from pokerkit import HandHistory

from poker_research.arena import ArenaConfig, HandResult, play_hand
from poker_research.equity import stable_seed
from poker_research.provenance import (
    dependency_versions,
    git_provenance,
    sha256_file,
    source_provenance,
)
from poker_research.statistics import EvaluationSummary, summarize_duplicate_pairs
from poker_research.types import Policy

PolicyFactory: TypeAlias = Callable[[], Policy]


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    """Frozen protocol shared by every round-robin matchup."""

    name: str = "policy-league"
    master_seed: int = 20260715
    pair_count_per_matchup: int = 100
    practical_margin_bb100: float = 0.0
    bootstrap_resamples: int = 20_000
    bootstrap_seed: int = 20260716
    policy_schedule_seed: int = 20260717
    game: ArenaConfig = ArenaConfig()

    def validate(self) -> None:
        self.game.validate()
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("league name must be non-empty and contain no whitespace")
        if self.pair_count_per_matchup < 2:
            raise ValueError("pair_count_per_matchup must be at least two")
        if self.bootstrap_resamples < 1_000:
            raise ValueError("bootstrap_resamples must be at least 1000")
        if self.policy_schedule_seed < 0:
            raise ValueError("policy_schedule_seed must be non-negative")
        if not math.isfinite(self.practical_margin_bb100):
            raise ValueError("practical_margin_bb100 must be finite")


@dataclass(frozen=True, slots=True)
class LeaguePairResult:
    """One independent duplicate-pair observation from policy A's perspective."""

    matchup_id: str
    policy_a: str
    policy_b: str
    pair_id: int
    deal_seed: int
    deck_hash: str
    policy_a_payoff_leg_a: int
    policy_a_payoff_leg_b: int
    policy_a_net_chips: int
    policy_a_bb100: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MatchupResult:
    """Pair-level estimate for one unordered policy matchup."""

    matchup_id: str
    policy_a: str
    policy_b: str
    planned_pair_count: int
    successful_pair_count: int
    failed_pair_count: int
    status: str
    summary: EvaluationSummary | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "matchup_id": self.matchup_id,
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "planned_pair_count": self.planned_pair_count,
            "successful_pair_count": self.successful_pair_count,
            "failed_pair_count": self.failed_pair_count,
            "status": self.status,
            "summary": _descriptive_summary(self.summary),
        }


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    """Aggregate pair-weighted result for one policy across all opponents."""

    rank: int
    policy: str
    opponent_count: int
    scored_matchups: int
    matchup_wins: int
    matchup_draws: int
    matchup_losses: int
    planned_pair_count: int
    successful_pair_count: int
    failed_pair_count: int
    summary: EvaluationSummary | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "policy": self.policy,
            "opponent_count": self.opponent_count,
            "scored_matchups": self.scored_matchups,
            "matchup_wins": self.matchup_wins,
            "matchup_draws": self.matchup_draws,
            "matchup_losses": self.matchup_losses,
            "planned_pair_count": self.planned_pair_count,
            "successful_pair_count": self.successful_pair_count,
            "failed_pair_count": self.failed_pair_count,
            "summary": _descriptive_summary(self.summary),
        }


@dataclass(frozen=True, slots=True)
class LeagueError:
    """A failed duplicate pair that was isolated from the rest of the league."""

    matchup_id: str
    policy_a: str
    policy_b: str
    pair_id: int
    deal_seed: int
    failed_leg: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LeagueHandRecord:
    """A completed hand, including orphaned first legs from failed pairs."""

    matchup_id: str
    policy_a: str
    policy_b: str
    pair_id: int
    leg: str
    policy_a_seat: int
    pair_status: str
    hand: HandResult


@dataclass(frozen=True, slots=True)
class LeagueResult:
    """In-memory result and the directory containing its durable artifacts."""

    run_id: str
    status: str
    output_directory: Path
    pairs: tuple[LeaguePairResult, ...]
    hands: tuple[LeagueHandRecord, ...]
    matchups: tuple[MatchupResult, ...]
    leaderboard: tuple[LeaderboardEntry, ...]
    errors: tuple[LeagueError, ...]


class _DuplicatePairFailure(RuntimeError):
    """Internal wrapper retaining the duplicate leg that failed."""

    def __init__(
        self,
        leg: str,
        cause: Exception,
        completed_hands: tuple[tuple[str, int, HandResult], ...] = (),
    ) -> None:
        super().__init__(str(cause))
        self.leg = leg
        self.cause = cause
        self.completed_hands = completed_hands


def run_league(
    policy_factories: Mapping[str, PolicyFactory],
    *,
    output_directory: Path,
    config: LeagueConfig | None = None,
    progress: Callable[[int, int, str, str], None] | None = None,
) -> LeagueResult:
    """Run every unordered policy matchup and continue after pair-level errors.

    Mapping keys are canonical policy names.  Each factory must return a policy
    whose ``name`` matches its key; factories are invoked anew for both legs of
    every duplicate pair.
    """

    rules = config or LeagueConfig()
    rules.validate()
    factories = dict(policy_factories)
    policy_names = _validate_factories(factories)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC)
    config_payload = {"league": asdict(rules), "policies": list(policy_names)}
    config_json = json.dumps(config_payload, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    run_id = f"{rules.name}-{config_hash[:12]}"
    all_pairs: list[LeaguePairResult] = []
    all_hands: list[LeagueHandRecord] = []
    matchup_results: list[MatchupResult] = []
    errors: list[LeagueError] = []
    completed_pair_attempts = 0
    total_pair_attempts = (
        len(policy_names) * (len(policy_names) - 1) // 2 * rules.pair_count_per_matchup
    )

    for matchup_index, (policy_a, policy_b) in enumerate(combinations(policy_names, 2)):
        matchup_id = f"matchup-{matchup_index:04d}"
        matchup_pairs: list[LeaguePairResult] = []
        for pair_id in range(rules.pair_count_per_matchup):
            deal_seed = stable_seed(
                rules.master_seed,
                "league_duplicate_pair",
                policy_a,
                policy_b,
                pair_id,
            )
            try:
                pair, played_hands = _play_duplicate_pair(
                    factories,
                    rules,
                    run_id=run_id,
                    matchup_id=matchup_id,
                    policy_a=policy_a,
                    policy_b=policy_b,
                    pair_id=pair_id,
                    deal_seed=deal_seed,
                )
            except _DuplicatePairFailure as failure:
                all_hands.extend(
                    LeagueHandRecord(
                        matchup_id=matchup_id,
                        policy_a=policy_a,
                        policy_b=policy_b,
                        pair_id=pair_id,
                        leg=leg,
                        policy_a_seat=policy_a_seat,
                        pair_status="invalid_pair",
                        hand=hand,
                    )
                    for leg, policy_a_seat, hand in failure.completed_hands
                )
                errors.append(
                    LeagueError(
                        matchup_id=matchup_id,
                        policy_a=policy_a,
                        policy_b=policy_b,
                        pair_id=pair_id,
                        deal_seed=deal_seed,
                        failed_leg=failure.leg,
                        error_type=type(failure.cause).__name__,
                        message=str(failure.cause),
                    )
                )
                completed_pair_attempts += 1
                if progress is not None:
                    progress(
                        completed_pair_attempts,
                        total_pair_attempts,
                        policy_a,
                        policy_b,
                    )
                continue
            matchup_pairs.append(pair)
            all_pairs.append(pair)
            all_hands.extend(
                LeagueHandRecord(
                    matchup_id=matchup_id,
                    policy_a=policy_a,
                    policy_b=policy_b,
                    pair_id=pair_id,
                    leg=leg,
                    policy_a_seat=policy_a_seat,
                    pair_status="complete_pair",
                    hand=hand,
                )
                for leg, policy_a_seat, hand in played_hands
            )
            completed_pair_attempts += 1
            if progress is not None:
                progress(completed_pair_attempts, total_pair_attempts, policy_a, policy_b)

        failed_count = rules.pair_count_per_matchup - len(matchup_pairs)
        summary = None
        if failed_count == 0:
            summary = _summarize(
                [pair.policy_a_bb100 for pair in matchup_pairs],
                practical_margin_bb100=rules.practical_margin_bb100,
                bootstrap_resamples=rules.bootstrap_resamples,
                bootstrap_seed=stable_seed(
                    rules.bootstrap_seed,
                    "league_matchup_bootstrap",
                    policy_a,
                    policy_b,
                ),
            )
        matchup_results.append(
            MatchupResult(
                matchup_id=matchup_id,
                policy_a=policy_a,
                policy_b=policy_b,
                planned_pair_count=rules.pair_count_per_matchup,
                successful_pair_count=len(matchup_pairs),
                failed_pair_count=failed_count,
                status=_matchup_status(len(matchup_pairs), failed_count),
                summary=summary,
            )
        )

    leaderboard = _build_leaderboard(
        policy_names,
        all_pairs,
        matchup_results,
        rules,
    )
    status = "complete" if not errors else "complete_with_errors"
    result = LeagueResult(
        run_id=run_id,
        status=status,
        output_directory=output_directory,
        pairs=tuple(all_pairs),
        hands=tuple(all_hands),
        matchups=tuple(matchup_results),
        leaderboard=leaderboard,
        errors=tuple(errors),
    )
    _validate_result(result, rules)
    _write_artifacts(result, rules, config_hash, started_at)
    return result


def _validate_factories(factories: Mapping[str, PolicyFactory]) -> tuple[str, ...]:
    if len(factories) < 2:
        raise ValueError("league requires at least two policy factories")
    for name, factory in factories.items():
        if not isinstance(name, str) or not name:
            raise ValueError("policy factory names must be non-empty strings")
        if not callable(factory):
            raise TypeError(f"policy factory for {name!r} must be callable")
    return tuple(sorted(factories))


def _fresh_policy(factory: PolicyFactory, expected_name: str) -> Policy:
    policy = factory()
    if policy.name != expected_name:
        raise ValueError(
            f"policy factory key {expected_name!r} returned policy named {policy.name!r}"
        )
    return policy


def _play_duplicate_pair(
    factories: Mapping[str, PolicyFactory],
    config: LeagueConfig,
    *,
    run_id: str,
    matchup_id: str,
    policy_a: str,
    policy_b: str,
    pair_id: int,
    deal_seed: int,
) -> tuple[LeaguePairResult, tuple[tuple[str, int, HandResult], ...]]:
    try:
        leg_a = play_hand(
            (
                _fresh_policy(factories[policy_a], policy_a),
                _fresh_policy(factories[policy_b], policy_b),
            ),
            deal_seed=deal_seed,
            hand_id=f"{run_id}-{matchup_id}-{pair_id:06d}-a",
            config=config.game,
            policy_rng_key=str(
                stable_seed(
                    config.policy_schedule_seed,
                    "league_policy_schedule_v1",
                    policy_a,
                    policy_b,
                    pair_id,
                    "a",
                )
            ),
        )
    except Exception as error:
        raise _DuplicatePairFailure("a", error) from error

    try:
        leg_b = play_hand(
            (
                _fresh_policy(factories[policy_b], policy_b),
                _fresh_policy(factories[policy_a], policy_a),
            ),
            deal_seed=deal_seed,
            hand_id=f"{run_id}-{matchup_id}-{pair_id:06d}-b",
            config=config.game,
            policy_rng_key=str(
                stable_seed(
                    config.policy_schedule_seed,
                    "league_policy_schedule_v1",
                    policy_a,
                    policy_b,
                    pair_id,
                    "b",
                )
            ),
        )
    except Exception as error:
        raise _DuplicatePairFailure("b", error, (("a", 0, leg_a),)) from error

    try:
        if leg_a.deck_hash != leg_b.deck_hash:
            raise RuntimeError("duplicate legs received different decks")
        payoff_a = leg_a.payoffs[0]
        payoff_b = leg_b.payoffs[1]
        net_chips = payoff_a + payoff_b
        bb100 = 50.0 * net_chips / config.game.big_blind
        return (
            LeaguePairResult(
                matchup_id=matchup_id,
                policy_a=policy_a,
                policy_b=policy_b,
                pair_id=pair_id,
                deal_seed=deal_seed,
                deck_hash=leg_a.deck_hash,
                policy_a_payoff_leg_a=payoff_a,
                policy_a_payoff_leg_b=payoff_b,
                policy_a_net_chips=net_chips,
                policy_a_bb100=bb100,
            ),
            (("a", 0, leg_a), ("b", 1, leg_b)),
        )
    except Exception as error:
        raise _DuplicatePairFailure(
            "validation",
            error,
            (("a", 0, leg_a), ("b", 1, leg_b)),
        ) from error


def _summarize(
    values: list[float],
    *,
    practical_margin_bb100: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> EvaluationSummary | None:
    if len(values) < 2:
        return None
    return summarize_duplicate_pairs(
        values,
        practical_margin_bb100=practical_margin_bb100,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def _matchup_status(successful_count: int, failed_count: int) -> str:
    if failed_count == 0:
        return "complete"
    if successful_count >= 2:
        return "partial"
    return "insufficient_data"


def _build_leaderboard(
    policy_names: tuple[str, ...],
    pairs: list[LeaguePairResult],
    matchups: list[MatchupResult],
    config: LeagueConfig,
) -> tuple[LeaderboardEntry, ...]:
    values: dict[str, list[float]] = {name: [] for name in policy_names}
    records = {
        name: {"scored": 0, "wins": 0, "draws": 0, "losses": 0}
        for name in policy_names
    }
    valid_matchup_ids = {
        matchup.matchup_id
        for matchup in matchups
        if matchup.status == "complete" and matchup.summary is not None
    }
    for pair in pairs:
        if pair.matchup_id not in valid_matchup_ids:
            continue
        values[pair.policy_a].append(pair.policy_a_bb100)
        values[pair.policy_b].append(-pair.policy_a_bb100)
    for matchup in matchups:
        if matchup.summary is None:
            continue
        records[matchup.policy_a]["scored"] += 1
        records[matchup.policy_b]["scored"] += 1
        mean = matchup.summary.mean_bb100
        if math.isclose(mean, 0.0, abs_tol=1e-12):
            records[matchup.policy_a]["draws"] += 1
            records[matchup.policy_b]["draws"] += 1
        elif mean > 0:
            records[matchup.policy_a]["wins"] += 1
            records[matchup.policy_b]["losses"] += 1
        else:
            records[matchup.policy_a]["losses"] += 1
            records[matchup.policy_b]["wins"] += 1

    opponent_count = len(policy_names) - 1
    planned_pair_count = opponent_count * config.pair_count_per_matchup
    unranked: list[LeaderboardEntry] = []
    for policy in policy_names:
        summary = _summarize(
            values[policy],
            practical_margin_bb100=config.practical_margin_bb100,
            bootstrap_resamples=config.bootstrap_resamples,
            bootstrap_seed=stable_seed(
                config.bootstrap_seed,
                "league_policy_bootstrap",
                policy,
            ),
        )
        successful_count = len(values[policy])
        record = records[policy]
        unranked.append(
            LeaderboardEntry(
                rank=0,
                policy=policy,
                opponent_count=opponent_count,
                scored_matchups=record["scored"],
                matchup_wins=record["wins"],
                matchup_draws=record["draws"],
                matchup_losses=record["losses"],
                planned_pair_count=planned_pair_count,
                successful_pair_count=successful_count,
                failed_pair_count=planned_pair_count - successful_count,
                summary=summary,
            )
        )

    ordered = sorted(
        unranked,
        key=lambda entry: (
            entry.summary is None,
            -(entry.summary.mean_bb100 if entry.summary is not None else 0.0),
            entry.policy,
        ),
    )
    return tuple(replace(entry, rank=rank) for rank, entry in enumerate(ordered, start=1))


def _validate_result(result: LeagueResult, config: LeagueConfig) -> None:
    """Reject internally inconsistent league artifacts before publishing a manifest."""

    pair_keys = {
        (pair.matchup_id, pair.pair_id): pair
        for pair in result.pairs
    }
    if len(pair_keys) != len(result.pairs):
        raise RuntimeError("league contains duplicate pair rows")
    hand_keys = {
        (record.matchup_id, record.pair_id, record.leg): record
        for record in result.hands
    }
    if len(hand_keys) != len(result.hands):
        raise RuntimeError("league contains duplicate hand rows")
    for record in result.hands:
        if sum(record.hand.payoffs) != 0:
            raise RuntimeError("league hand violates zero-sum payoff conservation")
        if sum(record.hand.finishing_stacks) != 2 * config.game.starting_stack:
            raise RuntimeError("league hand violates chip conservation")
        if record.hand.policy_names[record.policy_a_seat] != record.policy_a:
            raise RuntimeError("league hand policy-A seat does not match its policy names")
    for key, pair in pair_keys.items():
        try:
            leg_a = hand_keys[(*key, "a")]
            leg_b = hand_keys[(*key, "b")]
        except KeyError as error:
            raise RuntimeError("successful league pair is missing a duplicate leg") from error
        if leg_a.pair_status != "complete_pair" or leg_b.pair_status != "complete_pair":
            raise RuntimeError("successful league pair has an invalid hand status")
        if (leg_a.policy_a_seat, leg_b.policy_a_seat) != (0, 1):
            raise RuntimeError("duplicate league pair did not swap policy-A seats")
        if not (
            leg_a.hand.deal_seed == leg_b.hand.deal_seed == pair.deal_seed
        ):
            raise RuntimeError("duplicate league pair has inconsistent deal seeds")
        if not (
            leg_a.hand.deck_hash == leg_b.hand.deck_hash == pair.deck_hash
        ):
            raise RuntimeError("duplicate league pair has inconsistent deck hashes")
        if (
            leg_a.hand.payoffs[0] != pair.policy_a_payoff_leg_a
            or leg_b.hand.payoffs[1] != pair.policy_a_payoff_leg_b
        ):
            raise RuntimeError("league pair payoff does not match its hands")


def _hand_csv_row(record: LeagueHandRecord) -> dict[str, Any]:
    hand = record.hand
    return {
        "matchup_id": record.matchup_id,
        "policy_a": record.policy_a,
        "policy_b": record.policy_b,
        "pair_id": record.pair_id,
        "leg": record.leg,
        "pair_status": record.pair_status,
        "hand_id": hand.hand_id,
        "policy_rng_key": hand.policy_rng_key,
        "deal_seed": hand.deal_seed,
        "deck_hash": hand.deck_hash,
        "policy_a_seat": record.policy_a_seat,
        "seat0_policy": hand.policy_names[0],
        "seat1_policy": hand.policy_names[1],
        "seat0_payoff": hand.payoffs[0],
        "seat1_payoff": hand.payoffs[1],
        "seat0_hole": " ".join(hand.hole_cards[0]),
        "seat1_hole": " ".join(hand.hole_cards[1]),
        "board": " ".join(hand.board_cards),
        "starting_stack": hand.starting_stacks[0],
        "decision_count": len(hand.decisions),
        "total_latency_ms": sum(decision.latency_ms for decision in hand.decisions),
    }


def _policy_specs(result: LeagueResult) -> dict[str, dict[str, Any]]:
    """Lift stable checkpoint/prompt identifiers out of decision audit metadata."""

    stable_keys = (
        "checkpoint_sha256",
        "encoder_version",
        "expected_model_revision",
        "model",
        "model_revision",
        "prompt_version",
        "provider",
        "temperature",
    )
    values: dict[str, dict[str, set[str]]] = {
        entry.policy: {} for entry in result.leaderboard
    }
    for record in result.hands:
        for decision in record.hand.decisions:
            metadata = dict(decision.decision.metadata)
            policy_values = values.setdefault(decision.policy, {})
            for key in stable_keys:
                value = metadata.get(key)
                if value is not None:
                    policy_values.setdefault(key, set()).add(str(value))
    return {
        policy: {
            "name": policy,
            "observed_stable_metadata": {
                key: sorted(observed) for key, observed in sorted(fields.items())
            },
        }
        for policy, fields in sorted(values.items())
    }


_HAND_CSV_FIELDS = (
    "matchup_id",
    "policy_a",
    "policy_b",
    "pair_id",
    "leg",
    "pair_status",
    "hand_id",
    "policy_rng_key",
    "deal_seed",
    "deck_hash",
    "policy_a_seat",
    "seat0_policy",
    "seat1_policy",
    "seat0_payoff",
    "seat1_payoff",
    "seat0_hole",
    "seat1_hole",
    "board",
    "starting_stack",
    "decision_count",
    "total_latency_ms",
)


def _write_artifacts(
    result: LeagueResult,
    config: LeagueConfig,
    config_hash: str,
    started_at: datetime,
) -> None:
    output = result.output_directory
    _write_csv(
        output / "pairs.csv",
        [pair.to_dict() for pair in result.pairs],
        (
            "matchup_id",
            "policy_a",
            "policy_b",
            "pair_id",
            "deal_seed",
            "deck_hash",
            "policy_a_payoff_leg_a",
            "policy_a_payoff_leg_b",
            "policy_a_net_chips",
            "policy_a_bb100",
        ),
    )
    _write_csv(
        output / "matchups.csv",
        [_matchup_csv_row(matchup) for matchup in result.matchups],
        _SUMMARY_CSV_FIELDS,
    )
    _write_csv(
        output / "leaderboard.csv",
        [_leaderboard_csv_row(entry) for entry in result.leaderboard],
        _LEADERBOARD_CSV_FIELDS,
    )
    _write_csv(
        output / "hands.csv",
        [_hand_csv_row(record) for record in result.hands],
        _HAND_CSV_FIELDS,
    )
    decision_rows: list[dict[str, Any]] = []
    for record in result.hands:
        for decision in record.hand.decisions:
            decision_rows.append(
                {
                    "matchup_id": record.matchup_id,
                    "policy_a": record.policy_a,
                    "policy_b": record.policy_b,
                    "pair_id": record.pair_id,
                    "leg": record.leg,
                    "pair_status": record.pair_status,
                    "hand_id": record.hand.hand_id,
                    "policy_rng_key": record.hand.policy_rng_key,
                    "deal_seed": record.hand.deal_seed,
                    "deck_hash": record.hand.deck_hash,
                    **decision.to_dict(),
                }
            )
    _write_json_lines(output / "decisions.jsonl", decision_rows)
    with (output / "hands.phhs").open("wb") as phh_file:
        HandHistory.dump_all(
            (record.hand.hand_history for record in result.hands),
            phh_file,
        )
    _write_json_lines(output / "errors.jsonl", [error.to_dict() for error in result.errors])
    summary_payload = {
        "schema_version": 2,
        "run_id": result.run_id,
        "status": result.status,
        "aggregate_method": (
            "pooled_duplicate_pairs_equal_pair_weight_complete_matchups_only"
        ),
        "partial_matchups_excluded_from_all_statistics": True,
        "inferential_statistics": "not_reported_exploratory_round_robin",
        "config": asdict(config),
        "policies": [entry.policy for entry in sorted(result.leaderboard, key=lambda e: e.policy)],
        "successful_pair_count": len(result.pairs),
        "error_count": len(result.errors),
        "matchups": [matchup.to_dict() for matchup in result.matchups],
        "leaderboard": [entry.to_dict() for entry in result.leaderboard],
    }
    _write_json(output / "summary.json", summary_payload)

    artifact_names = (
        "decisions.jsonl",
        "errors.jsonl",
        "hands.csv",
        "hands.phhs",
        "leaderboard.csv",
        "matchups.csv",
        "pairs.csv",
        "summary.json",
    )
    finished_at = datetime.now(UTC)
    _write_json(
        output / "manifest.json",
        {
            "schema_version": 2,
            "run_id": result.run_id,
            "status": result.status,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
            "config_sha256": config_hash,
            "git": git_provenance(),
            "source": source_provenance(),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "dependencies": dependency_versions(),
            "policy_specs": _policy_specs(result),
            "artifacts": {name: sha256_file(output / name) for name in artifact_names},
        },
    )


_SUMMARY_CSV_FIELDS = (
    "matchup_id",
    "policy_a",
    "policy_b",
    "status",
    "planned_pair_count",
    "successful_pair_count",
    "failed_pair_count",
    "mean_bb100",
    "std_bb100",
    "standard_error",
    "ci95_low",
    "ci95_high",
)

_LEADERBOARD_CSV_FIELDS = (
    "rank",
    "policy",
    "opponent_count",
    "scored_matchups",
    "matchup_wins",
    "matchup_draws",
    "matchup_losses",
    "planned_pair_count",
    "successful_pair_count",
    "failed_pair_count",
    "mean_bb100",
    "std_bb100",
    "standard_error",
    "ci95_low",
    "ci95_high",
)


def _summary_csv_values(summary: EvaluationSummary | None) -> dict[str, Any]:
    if summary is None:
        return {
            "mean_bb100": None,
            "std_bb100": None,
            "standard_error": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    return {
        "mean_bb100": summary.mean_bb100,
        "std_bb100": summary.std_bb100,
        "standard_error": summary.standard_error,
        "ci95_low": summary.two_sided_ci95[0],
        "ci95_high": summary.two_sided_ci95[1],
    }


def _descriptive_summary(summary: EvaluationSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "pair_count": summary.pair_count,
        "mean_bb100": summary.mean_bb100,
        "std_bb100": summary.std_bb100,
        "standard_error": summary.standard_error,
        "two_sided_ci95": list(summary.two_sided_ci95),
        "bootstrap_ci95": list(summary.bootstrap_ci95),
    }


def _matchup_csv_row(matchup: MatchupResult) -> dict[str, Any]:
    return {
        "matchup_id": matchup.matchup_id,
        "policy_a": matchup.policy_a,
        "policy_b": matchup.policy_b,
        "status": matchup.status,
        "planned_pair_count": matchup.planned_pair_count,
        "successful_pair_count": matchup.successful_pair_count,
        "failed_pair_count": matchup.failed_pair_count,
        **_summary_csv_values(matchup.summary),
    }


def _leaderboard_csv_row(entry: LeaderboardEntry) -> dict[str, Any]:
    summary_values = _summary_csv_values(entry.summary)
    return {
        "rank": entry.rank,
        "policy": entry.policy,
        "opponent_count": entry.opponent_count,
        "scored_matchups": entry.scored_matchups,
        "matchup_wins": entry.matchup_wins,
        "matchup_draws": entry.matchup_draws,
        "matchup_losses": entry.matchup_losses,
        "planned_pair_count": entry.planned_pair_count,
        "successful_pair_count": entry.successful_pair_count,
        "failed_pair_count": entry.failed_pair_count,
        **summary_values,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
