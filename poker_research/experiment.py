"""Duplicate-poker experiment runner with immutable, auditable artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from pokerkit import HandHistory

from poker_research.arena import ArenaConfig, HandResult, play_hand
from poker_research.catalog import policy_factories
from poker_research.equity import stable_seed
from poker_research.neural import NumpyMLPPolicy
from poker_research.policies import CallingStationPolicy, EquityValueConfig, EquityValuePolicy
from poker_research.provenance import (
    dependency_versions,
    git_provenance,
    sha256_file,
    source_provenance,
)
from poker_research.statistics import EvaluationSummary, summarize_duplicate_pairs
from poker_research.types import Policy

ExperimentMode: TypeAlias = Literal["debug", "pilot", "confirmatory"]
EXPERIMENT_MODES = frozenset(("debug", "pilot", "confirmatory"))
_TOP_LEVEL_KEYS = frozenset(("experiment", "game", "hero", "opponent", "equity_value"))
_EXPERIMENT_KEYS = frozenset(
    (
        "name",
        "mode",
        "master_seed",
        "pair_count",
        "practical_margin_bb100",
        "bootstrap_resamples",
        "bootstrap_seed",
        "policy_schedule_seed",
    )
)
_GAME_KEYS = frozenset(("starting_stack", "small_blind", "big_blind", "ante"))
_POLICY_KEYS = frozenset(("policy", "seed", "checkpoint"))
_EQUITY_VALUE_KEYS = frozenset(
    (
        "preflop_samples",
        "postflop_samples",
        "call_margin",
        "preflop_raise_equity",
        "flop_raise_equity",
        "turn_raise_equity",
        "river_raise_equity",
        "strong_equity",
        "nut_equity",
    )
)
_CONFIRMATORY_EXPLICIT_KEYS = frozenset(
    (
        "practical_margin_bb100",
        "bootstrap_resamples",
        "bootstrap_seed",
        "policy_schedule_seed",
    )
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Frozen experiment protocol parsed from TOML."""

    name: str
    master_seed: int
    pair_count: int
    practical_margin_bb100: float
    bootstrap_resamples: int
    bootstrap_seed: int
    game: ArenaConfig
    hero_policy: str
    opponent_policy: str
    hero_seed: int
    equity_value: EquityValueConfig
    opponent_seed: int = 0
    hero_checkpoint: str | None = None
    opponent_checkpoint: str | None = None
    policy_schedule_seed: int = 20_260_717
    mode: ExperimentMode = "debug"

    def validate(self) -> None:
        self.game.validate()
        self.equity_value.validate()
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("experiment name must be non-empty and contain no whitespace")
        if self.mode not in EXPERIMENT_MODES:
            raise ValueError("mode must be 'debug', 'pilot' or 'confirmatory'")
        for field_name in (
            "master_seed",
            "bootstrap_seed",
            "hero_seed",
            "opponent_seed",
            "policy_schedule_seed",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if isinstance(self.pair_count, bool) or not isinstance(self.pair_count, int):
            raise ValueError("pair_count must be an integer")
        if self.pair_count < 2:
            raise ValueError("pair_count must be at least two")
        if isinstance(self.bootstrap_resamples, bool) or not isinstance(
            self.bootstrap_resamples, int
        ):
            raise ValueError("bootstrap_resamples must be an integer")
        if self.bootstrap_resamples < 1_000:
            raise ValueError("bootstrap_resamples must be at least 1000")
        if (
            isinstance(self.practical_margin_bb100, bool)
            or not isinstance(self.practical_margin_bb100, (int, float))
            or not math.isfinite(self.practical_margin_bb100)
            or self.practical_margin_bb100 < 0
        ):
            raise ValueError("practical_margin_bb100 must be finite and non-negative")
        if not self.hero_policy or not self.opponent_policy:
            raise ValueError("hero and opponent policy names must be non-empty")
        if self.hero_policy == self.opponent_policy:
            raise ValueError("hero and opponent policy names must differ")


@dataclass(frozen=True, slots=True)
class PairResult:
    pair_id: int
    deal_seed: int
    deck_hash: str
    hero_payoff_leg_a: int
    hero_payoff_leg_b: int
    hero_net_chips: int
    hero_bb100: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    output_directory: Path
    summary: EvaluationSummary
    pairs: tuple[PairResult, ...]


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Parse a closed preregistration schema without silently ignoring typos."""

    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "top level")
    experiment = _section(raw, "experiment", _EXPERIMENT_KEYS)
    game = _section(raw, "game", _GAME_KEYS)
    hero = _section(raw, "hero", _POLICY_KEYS)
    opponent = _section(raw, "opponent", _POLICY_KEYS)
    value = _section(raw, "equity_value", _EQUITY_VALUE_KEYS, required=False)

    raw_mode = _as_string(
        _required(experiment, "mode", "[experiment]"),
        "experiment.mode",
    )
    if raw_mode not in EXPERIMENT_MODES:
        raise ValueError("experiment.mode must be 'debug', 'pilot' or 'confirmatory'")
    mode = cast(ExperimentMode, raw_mode)
    if mode == "confirmatory":
        missing = sorted(_CONFIRMATORY_EXPLICIT_KEYS.difference(experiment))
        if missing:
            raise ValueError(
                "confirmatory [experiment] must explicitly set: " + ", ".join(missing)
            )

    config = BenchmarkConfig(
        name=_as_string(
            _required(experiment, "name", "[experiment]"),
            "experiment.name",
        ),
        master_seed=_as_integer(
            _required(experiment, "master_seed", "[experiment]"),
            "experiment.master_seed",
        ),
        pair_count=_as_integer(
            _required(experiment, "pair_count", "[experiment]"),
            "experiment.pair_count",
        ),
        practical_margin_bb100=_as_number(
            experiment.get("practical_margin_bb100", 0.0),
            "experiment.practical_margin_bb100",
        ),
        bootstrap_resamples=_as_integer(
            experiment.get("bootstrap_resamples", 20_000),
            "experiment.bootstrap_resamples",
        ),
        bootstrap_seed=_as_integer(
            experiment.get("bootstrap_seed", 20_260_715),
            "experiment.bootstrap_seed",
        ),
        game=ArenaConfig(
            starting_stack=_as_integer(
                game.get("starting_stack", 200), "game.starting_stack"
            ),
            small_blind=_as_integer(game.get("small_blind", 1), "game.small_blind"),
            big_blind=_as_integer(game.get("big_blind", 2), "game.big_blind"),
            ante=_as_integer(game.get("ante", 0), "game.ante"),
        ),
        hero_policy=_as_string(_required(hero, "policy", "[hero]"), "hero.policy"),
        opponent_policy=_as_string(
            _required(opponent, "policy", "[opponent]"),
            "opponent.policy",
        ),
        hero_seed=_as_integer(hero.get("seed", 0), "hero.seed"),
        equity_value=EquityValueConfig(
            preflop_samples=_as_integer(
                value.get("preflop_samples", 256),
                "equity_value.preflop_samples",
            ),
            postflop_samples=_as_integer(
                value.get("postflop_samples", 384),
                "equity_value.postflop_samples",
            ),
            call_margin=_as_number(
                value.get("call_margin", 0.015),
                "equity_value.call_margin",
            ),
            preflop_raise_equity=_as_number(
                value.get("preflop_raise_equity", 0.60),
                "equity_value.preflop_raise_equity",
            ),
            flop_raise_equity=_as_number(
                value.get("flop_raise_equity", 0.61),
                "equity_value.flop_raise_equity",
            ),
            turn_raise_equity=_as_number(
                value.get("turn_raise_equity", 0.59),
                "equity_value.turn_raise_equity",
            ),
            river_raise_equity=_as_number(
                value.get("river_raise_equity", 0.53),
                "equity_value.river_raise_equity",
            ),
            strong_equity=_as_number(
                value.get("strong_equity", 0.70),
                "equity_value.strong_equity",
            ),
            nut_equity=_as_number(
                value.get("nut_equity", 0.84),
                "equity_value.nut_equity",
            ),
        ),
        opponent_seed=_as_integer(opponent.get("seed", 0), "opponent.seed"),
        hero_checkpoint=_checkpoint_path(
            path,
            _as_optional_string(hero.get("checkpoint"), "hero.checkpoint"),
        ),
        opponent_checkpoint=_checkpoint_path(
            path,
            _as_optional_string(opponent.get("checkpoint"), "opponent.checkpoint"),
        ),
        policy_schedule_seed=_as_integer(
            experiment.get("policy_schedule_seed", 20_260_717),
            "experiment.policy_schedule_seed",
        ),
        mode=mode,
    )
    config.validate()
    return config


def _section(
    raw: Mapping[str, Any],
    name: str,
    allowed_keys: frozenset[str],
    *,
    required: bool = True,
) -> dict[str, Any]:
    if name not in raw:
        if required:
            raise ValueError(f"missing required [{name}] section")
        return {}
    value = raw[name]
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    _reject_unknown_keys(value, allowed_keys, f"[{name}]")
    return value


def _reject_unknown_keys(
    mapping: Mapping[str, Any],
    allowed_keys: frozenset[str],
    location: str,
) -> None:
    unknown = sorted(set(mapping).difference(allowed_keys))
    if unknown:
        raise ValueError(f"unknown key(s) in {location}: {', '.join(unknown)}")


def _required(section: Mapping[str, Any], key: str, location: str) -> object:
    if key not in section:
        raise ValueError(f"missing required {location}.{key}")
    return section[key]


def _as_string(value: object, qualified_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{qualified_name} must be a string")
    return value


def _as_optional_string(value: object | None, qualified_name: str) -> str | None:
    if value is None:
        return None
    return _as_string(value, qualified_name)


def _as_integer(value: object, qualified_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{qualified_name} must be an integer")
    return value


def _as_number(value: object, qualified_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{qualified_name} must be numeric")
    return float(value)


def _checkpoint_path(config_path: Path, raw_path: object | None) -> str | None:
    if raw_path is None:
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())


def run_benchmark(
    config: BenchmarkConfig,
    *,
    output_directory: Path,
    config_path: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> BenchmarkResult:
    """Run fixed-horizon duplicate pairs; never train or mutate a policy."""

    config.validate()
    checkpoint_provenance = _checkpoint_provenance(config)
    if output_directory.exists() and (
        not output_directory.is_dir() or any(output_directory.iterdir())
    ):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    config_payload = _config_payload(config)
    hash_payload: dict[str, Any] = dict(config_payload)
    if checkpoint_provenance:
        hash_payload["checkpoint_sha256"] = {
            role: payload["sha256"] for role, payload in checkpoint_provenance.items()
        }
    config_json = json.dumps(hash_payload, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    run_id = f"{config.name}-{config_hash[:12]}"
    pairs: list[PairResult] = []
    hands: list[tuple[int, str, int, HandResult]] = []
    errors: list[dict[str, Any]] = []
    try:
        if config_path is not None:
            (output_directory / "preregistered_config.toml").write_bytes(
                config_path.read_bytes()
            )
        for role, payload in checkpoint_provenance.items():
            source = Path(payload["path"])
            frozen_copy = output_directory / f"{role}_checkpoint.npz"
            frozen_copy.write_bytes(source.read_bytes())
            if sha256_file(frozen_copy) != payload["sha256"]:
                raise RuntimeError(f"{role} checkpoint changed while the run was starting")
        hero_factory, opponent_factory = _make_policy_factories(
            config,
            checkpoint_provenance,
        )
        for pair_id in range(config.pair_count):
            deal_seed = stable_seed(config.master_seed, "duplicate_pair", pair_id)
            hero_a, opponent_a = hero_factory(), opponent_factory()
            leg_a = play_hand(
                (hero_a, opponent_a),
                deal_seed=deal_seed,
                hand_id=f"{run_id}-{pair_id:06d}-a",
                config=config.game,
                policy_rng_key=str(
                    stable_seed(
                        config.policy_schedule_seed,
                        "benchmark_policy_schedule_v1",
                        pair_id,
                        "a",
                    )
                ),
            )
            opponent_b, hero_b = opponent_factory(), hero_factory()
            leg_b = play_hand(
                (opponent_b, hero_b),
                deal_seed=deal_seed,
                hand_id=f"{run_id}-{pair_id:06d}-b",
                config=config.game,
                policy_rng_key=str(
                    stable_seed(
                        config.policy_schedule_seed,
                        "benchmark_policy_schedule_v1",
                        pair_id,
                        "b",
                    )
                ),
            )
            if leg_a.deck_hash != leg_b.deck_hash:
                raise RuntimeError("duplicate legs received different decks")
            hero_payoff_a = leg_a.payoffs[0]
            hero_payoff_b = leg_b.payoffs[1]
            net_chips = hero_payoff_a + hero_payoff_b
            bb100 = 50.0 * net_chips / config.game.big_blind
            pair = PairResult(
                pair_id=pair_id,
                deal_seed=deal_seed,
                deck_hash=leg_a.deck_hash,
                hero_payoff_leg_a=hero_payoff_a,
                hero_payoff_leg_b=hero_payoff_b,
                hero_net_chips=net_chips,
                hero_bb100=bb100,
            )
            pairs.append(pair)
            hands.extend(((pair_id, "a", 0, leg_a), (pair_id, "b", 1, leg_b)))
            if progress is not None:
                progress(pair_id + 1, config.pair_count)
    except Exception as error:
        _record_failed_run(
            output_directory,
            run_id=run_id,
            config_hash=config_hash,
            started_at=started_at,
            completed_pairs=len(pairs),
            error=error,
            errors=errors,
        )
        raise

    try:
        summary = summarize_duplicate_pairs(
            [pair.hero_bb100 for pair in pairs],
            practical_margin_bb100=config.practical_margin_bb100,
            bootstrap_resamples=config.bootstrap_resamples,
            bootstrap_seed=config.bootstrap_seed,
        )
        _write_artifacts(
            output_directory,
            run_id,
            hands,
            pairs,
            summary,
            config.game.big_blind,
        )
        _write_json_lines(output_directory / "errors.jsonl", errors)
        finished_at = datetime.now(UTC)
        artifact_names = [
            "decisions.jsonl",
            "hands.csv",
            "hands.phhs",
            "pairs.csv",
            "summary.json",
            "errors.jsonl",
        ]
        if (output_directory / "preregistered_config.toml").is_file():
            artifact_names.append("preregistered_config.toml")
        artifact_names.extend(
            f"{role}_checkpoint.npz" for role in sorted(checkpoint_provenance)
        )
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
            "config_sha256": config_hash,
            "config": config_payload,
            "git": git_provenance(),
            "source": source_provenance(),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "dependencies": dependency_versions(),
            "policy_checkpoints": checkpoint_provenance,
            "artifacts": {
                name: sha256_file(output_directory / name) for name in artifact_names
            },
        }
        _write_json(output_directory / "manifest.json", manifest)
        return BenchmarkResult(output_directory, summary, tuple(pairs))
    except Exception as error:
        _record_failed_run(
            output_directory,
            run_id=run_id,
            config_hash=config_hash,
            started_at=started_at,
            completed_pairs=len(pairs),
            error=error,
            errors=errors,
        )
        raise


def _record_failed_run(
    output_directory: Path,
    *,
    run_id: str,
    config_hash: str,
    started_at: datetime,
    completed_pairs: int,
    error: Exception,
    errors: list[dict[str, Any]],
) -> None:
    errors.append(
        {
            "run_id": run_id,
            "completed_pairs": completed_pairs,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    )
    _write_json_lines(output_directory / "errors.jsonl", errors)
    _write_json(
        output_directory / "failed_run.json",
        {
            "run_id": run_id,
            "status": "failed",
            "started_at": started_at.isoformat(),
            "failed_at": datetime.now(UTC).isoformat(),
            "config_sha256": config_hash,
            "completed_pairs": completed_pairs,
        },
    )


def _make_policies(config: BenchmarkConfig) -> tuple[Policy, Policy]:
    provenance = _checkpoint_provenance(config)
    factories = _make_policy_factories(config, provenance)
    return factories[0](), factories[1]()


def _make_policy_factories(
    config: BenchmarkConfig,
    checkpoint_provenance: dict[str, dict[str, str]],
) -> tuple[Callable[[], Policy], Callable[[], Policy]]:
    return (
        _make_policy_factory(
            config.hero_policy,
            config.hero_seed,
            config,
            config.hero_checkpoint,
            checkpoint_provenance.get("hero"),
        ),
        _make_policy_factory(
            config.opponent_policy,
            config.opponent_seed,
            config,
            config.opponent_checkpoint,
            checkpoint_provenance.get("opponent"),
        ),
    )


def _make_policy_factory(
    name: str,
    seed: int,
    config: BenchmarkConfig,
    checkpoint: str | None,
    provenance: dict[str, str] | None,
) -> Callable[[], Policy]:
    if checkpoint is not None:
        if provenance is None:
            raise RuntimeError("checkpoint provenance was not computed")
        template = NumpyMLPPolicy.from_checkpoint(
            Path(checkpoint),
            expected_sha256=provenance["sha256"],
        )
        if template.name != name:
            raise ValueError(
                f"checkpoint policy name {template.name!r} does not match configured {name!r}"
            )

        def make_checkpoint_policy() -> NumpyMLPPolicy:
            return NumpyMLPPolicy(
                template.weights,
                encoder=template.encoder,
                name=template.name,
                seed=seed,
                checkpoint_sha256=template.checkpoint_sha256,
            )

        return make_checkpoint_policy
    if name == "calling_station_v1":
        return CallingStationPolicy
    if name == "equity_value_v1":
        return lambda: EquityValuePolicy(config.equity_value, seed=seed)
    try:
        factory = policy_factories(master_seed=seed, selected=(name,))[name]
    except ValueError as error:
        raise ValueError(f"unknown policy: {name}") from error
    return factory


def _checkpoint_provenance(config: BenchmarkConfig) -> dict[str, dict[str, str]]:
    checkpoints = {
        "hero": config.hero_checkpoint,
        "opponent": config.opponent_checkpoint,
    }
    provenance: dict[str, dict[str, str]] = {}
    for role, raw_path in checkpoints.items():
        if raw_path is None:
            continue
        path = Path(raw_path)
        if path.suffix != ".npz" or not path.is_file():
            raise ValueError(f"{role} checkpoint must be an existing .npz file: {path}")
        provenance[role] = {"path": str(path), "sha256": sha256_file(path)}
    return provenance


def _write_artifacts(
    output_directory: Path,
    run_id: str,
    hands: list[tuple[int, str, int, HandResult]],
    pairs: list[PairResult],
    summary: EvaluationSummary,
    big_blind: int,
) -> None:
    hand_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for pair_id, leg, hero_seat, hand in hands:
        hand_rows.append(
            {
                "run_id": run_id,
                "policy_rng_key": hand.policy_rng_key,
                "pair_id": pair_id,
                "leg": leg,
                "deal_seed": hand.deal_seed,
                "deck_hash": hand.deck_hash,
                "hero_seat": hero_seat,
                "hero_payoff_chips": hand.payoffs[hero_seat],
                "hero_payoff_bb": hand.payoffs[hero_seat] / big_blind,
                "seat0_policy": hand.policy_names[0],
                "seat1_policy": hand.policy_names[1],
                "seat0_payoff": hand.payoffs[0],
                "seat1_payoff": hand.payoffs[1],
                "seat0_hole": " ".join(hand.hole_cards[0]),
                "seat1_hole": " ".join(hand.hole_cards[1]),
                "board": " ".join(hand.board_cards),
                "starting_stack": hand.starting_stacks[0],
                "big_blind": big_blind,
                "decision_count": len(hand.decisions),
                "total_latency_ms": sum(record.latency_ms for record in hand.decisions),
                "status": "complete",
            }
        )
        for record in hand.decisions:
            decision_rows.append(
                {
                    "run_id": run_id,
                    "pair_id": pair_id,
                    "leg": leg,
                    "deal_seed": hand.deal_seed,
                    "deck_hash": hand.deck_hash,
                    **record.to_dict(),
                }
            )
    _write_csv(output_directory / "hands.csv", hand_rows)
    _write_json_lines(output_directory / "decisions.jsonl", decision_rows)
    _write_csv(output_directory / "pairs.csv", [asdict(pair) for pair in pairs])
    _write_json(output_directory / "summary.json", summary.to_dict())
    with (output_directory / "hands.phhs").open("wb") as phh_file:
        HandHistory.dump_all((hand.hand_history for _, _, _, hand in hands), phh_file)


def _config_payload(config: BenchmarkConfig) -> dict[str, Any]:
    return asdict(config)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
