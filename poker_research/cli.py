"""Command-line entry point for preregistered poker benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from poker_research.arena import ArenaConfig
from poker_research.catalog import policy_descriptors, policy_factories, suite_names
from poker_research.dqn import DoubleDQNTrainer, DQNConfig, DQNHandReport
from poker_research.equity import stable_seed
from poker_research.experiment import load_benchmark_config, run_benchmark
from poker_research.game_theory import run_cfr_verification
from poker_research.imitation import (
    BehaviorCloningConfig,
    Demonstration,
    collect_demonstrations,
    train_behavior_cloning,
)
from poker_research.league import LeagueConfig, run_league
from poker_research.neural import EncoderConfig, NumpyMLPPolicy, ObservationEncoder
from poker_research.statistics import required_pair_count
from poker_research.types import Policy
from poker_research.validation import validate_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poker-benchmark",
        description="Run and audit fixed-horizon duplicate-poker experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a preregistered TOML benchmark")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--progress-every", type=int, default=25)

    power = subparsers.add_parser("power", help="estimate required duplicate pairs")
    power.add_argument("--standard-deviation", type=float, required=True)
    power.add_argument("--effect", type=float, required=True)
    power.add_argument("--alpha", type=float, default=0.05)
    power.add_argument("--power", type=float, default=0.90)

    theory = subparsers.add_parser(
        "theory", help="verify CFR convergence and exact exploitability in a small game"
    )
    theory.add_argument("--game", choices=("kuhn_poker", "leduc_poker"), default="kuhn_poker")
    theory.add_argument("--algorithm", choices=("cfr", "cfr_plus"), default="cfr_plus")
    theory.add_argument("--iterations", type=int, default=10_000)
    theory.add_argument("--report-every", type=int, default=100)
    theory.add_argument("--output", type=Path)

    validate = subparsers.add_parser("validate-run", help="audit a completed run directory")
    validate.add_argument("directory", type=Path)

    catalog = subparsers.add_parser("catalog", help="list ready-to-run policy implementations")
    catalog.add_argument("--suite", choices=("quick", "standard", "extended"))

    league = subparsers.add_parser(
        "league",
        help="run an exploratory duplicate round-robin policy league",
    )
    league.add_argument("--output", type=Path, required=True)
    league.add_argument("--suite", choices=("quick", "standard", "extended"), default="quick")
    league.add_argument("--policies", nargs="+", help="explicit policy names; overrides --suite")
    league.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="safe neural .npz checkpoint; repeat to add multiple trained policies",
    )
    league.add_argument("--pairs", type=int, default=10, help="duplicate pairs per matchup")
    league.add_argument("--master-seed", type=int, default=20_260_715)
    league.add_argument("--bootstrap-resamples", type=int, default=2_000)
    league.add_argument("--bootstrap-seed", type=int, default=20_260_716)
    league.add_argument("--practical-margin", type=float, default=0.0)
    league.add_argument("--starting-stack", type=int, default=200)
    league.add_argument("--small-blind", type=int, default=1)
    league.add_argument("--big-blind", type=int, default=2)
    league.add_argument("--ante", type=int, default=0)
    league.add_argument("--progress-every", type=int, default=10)
    league.add_argument(
        "--allow-partial-debug",
        action="store_true",
        help="return success despite pair failures; partial matchups remain unscored",
    )

    imitate = subparsers.add_parser(
        "imitate",
        help="train a safe NumPy neural policy by cloning a catalog teacher",
    )
    imitate.add_argument("--teacher", default="equity_value_v1")
    imitate.add_argument(
        "--opponents",
        nargs="+",
        default=["calling_station_v1", "maniac_v1", "tight_passive_v1"],
    )
    imitate.add_argument("--deals-per-opponent", type=int, default=100)
    imitate.add_argument("--checkpoint", type=Path, required=True)
    imitate.add_argument("--report", type=Path)
    imitate.add_argument("--policy-name", default="behavior_cloning_v1")
    imitate.add_argument("--master-seed", type=int, default=20_260_715)
    imitate.add_argument("--epochs", type=int, default=12)
    imitate.add_argument("--batch-size", type=int, default=128)
    imitate.add_argument("--hidden-size", type=int, default=128)
    imitate.add_argument("--learning-rate", type=float, default=1e-3)
    imitate.add_argument("--validation-fraction", type=float, default=0.2)
    imitate.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    imitate.add_argument("--equity-samples", type=int, default=64)
    imitate.add_argument("--starting-stack", type=int, default=200)
    imitate.add_argument("--small-blind", type=int, default=1)
    imitate.add_argument("--big-blind", type=int, default=2)
    imitate.add_argument("--ante", type=int, default=0)

    dqn = subparsers.add_parser(
        "train-dqn",
        help="run exploratory Double-DQN training and export safe inference weights",
    )
    dqn.add_argument("--output", type=Path, required=True)
    dqn.add_argument(
        "--opponents",
        nargs="+",
        default=[
            "calling_station_v1",
            "maniac_v1",
            "loose_passive_v1",
            "tight_passive_v1",
            "tag_v1",
        ],
    )
    dqn.add_argument("--training-hands", type=int, default=5_000)
    dqn.add_argument("--validation-hands", type=int, default=500)
    dqn.add_argument("--policy-name", default="double_dqn_v1")
    dqn.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="optional safe .npz behavior-cloning/blueprint initialization",
    )
    dqn.add_argument(
        "--encoder-checkpoint",
        type=Path,
        help="reuse only a safe .npz encoder for a controlled random-init ablation",
    )
    dqn.add_argument("--master-seed", type=int, default=20_260_715)
    dqn.add_argument("--hidden-size", type=int, default=128)
    dqn.add_argument("--batch-size", type=int, default=128)
    dqn.add_argument("--replay-capacity", type=int, default=100_000)
    dqn.add_argument("--min-replay-size", type=int, default=1_000)
    dqn.add_argument("--updates-per-transition", type=int, default=1)
    dqn.add_argument("--target-sync-interval", type=int, default=1_000)
    dqn.add_argument("--learning-rate", type=float, default=1e-3)
    dqn.add_argument("--gamma", type=float, default=0.99)
    dqn.add_argument("--epsilon-start", type=float, default=1.0)
    dqn.add_argument("--epsilon-end", type=float, default=0.05)
    dqn.add_argument("--epsilon-decay-decisions", type=int, default=50_000)
    dqn.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    dqn.add_argument("--equity-samples", type=int, default=32)
    dqn.add_argument("--starting-stack", type=int, default=200)
    dqn.add_argument("--small-blind", type=int, default=1)
    dqn.add_argument("--big-blind", type=int, default=2)
    dqn.add_argument("--ante", type=int, default=0)
    dqn.add_argument("--progress-every", type=int, default=250)
    return parser


def _catalog_training_factory(name: str) -> Callable[[int, int], Policy]:
    def make(policy_seed: int, _seat: int) -> Policy:
        return policy_factories(master_seed=policy_seed, selected=(name,))[name]()

    return make


def _dqn_report_summary(reports: list[DQNHandReport]) -> dict[str, float | int]:
    payoffs = [report.hero_payoff_bb for report in reports]
    optimization = [metric for report in reports for metric in report.optimization]
    return {
        "hand_count": len(reports),
        "decision_count": sum(report.hero_decisions for report in reports),
        "transition_count": sum(report.transitions_added for report in reports),
        "mean_payoff_bb_per_hand": fmean(payoffs) if payoffs else 0.0,
        "optimization_count": len(optimization),
        "final_loss": optimization[-1].loss if optimization else 0.0,
        "final_mean_absolute_td_error": (
            optimization[-1].mean_absolute_td_error if optimization else 0.0
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "catalog":
        descriptors = policy_descriptors()
        names = tuple(descriptors) if args.suite is None else suite_names(args.suite)
        print(
            json.dumps(
                [descriptors[name].to_dict() for name in names],
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "league":
        checkpoint_names = tuple(
            NumpyMLPPolicy.from_checkpoint(path).name for path in args.checkpoint
        )
        selected = (
            tuple(args.policies)
            if args.policies
            else (*suite_names(args.suite), *checkpoint_names)
        )
        factories = policy_factories(
            master_seed=args.master_seed,
            selected=selected,
            checkpoints=tuple(args.checkpoint),
        )
        league_config = LeagueConfig(
            name=f"exploratory-{args.suite}-league",
            master_seed=args.master_seed,
            pair_count_per_matchup=args.pairs,
            practical_margin_bb100=args.practical_margin,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            game=ArenaConfig(
                starting_stack=args.starting_stack,
                small_blind=args.small_blind,
                big_blind=args.big_blind,
                ante=args.ante,
            ),
        )

        def show_league_progress(done: int, total: int, first: str, second: str) -> None:
            if done == total or (args.progress_every > 0 and done % args.progress_every == 0):
                print(
                    f"completed {done}/{total} duplicate pairs ({first} vs {second})",
                    flush=True,
                )

        league_result = run_league(
            factories,
            output_directory=args.output,
            config=league_config,
            progress=show_league_progress,
        )
        print(
            json.dumps(
                {
                    "run_id": league_result.run_id,
                    "status": league_result.status,
                    "policy_count": len(factories),
                    "matchup_count": len(league_result.matchups),
                    "successful_pair_count": len(league_result.pairs),
                    "error_count": len(league_result.errors),
                    "leaderboard": [
                        entry.to_dict() for entry in league_result.leaderboard
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        if league_result.errors and not args.allow_partial_debug:
            raise SystemExit(2)
        return

    if args.command == "imitate":
        if args.deals_per_opponent < 1:
            raise ValueError("deals-per-opponent must be positive")
        if args.teacher in args.opponents:
            raise ValueError("teacher and opponents must have distinct policy names")
        if len(args.opponents) != len(set(args.opponents)):
            raise ValueError("opponent policy names must be unique")
        report_path = args.report or args.checkpoint.with_suffix(".training.json")
        if report_path.resolve() == args.checkpoint.resolve():
            raise ValueError("checkpoint and report paths must be different")
        if report_path.exists():
            raise FileExistsError(f"refusing to overwrite training report: {report_path}")
        rules = ArenaConfig(
            starting_stack=args.starting_stack,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            ante=args.ante,
        )
        encoder = ObservationEncoder(
            EncoderConfig(
                starting_stack=args.starting_stack,
                big_blind=args.big_blind,
                equity_samples=args.equity_samples,
                equity_seed=stable_seed(args.master_seed, "imitation_encoder"),
            )
        )
        demonstrations: list[Demonstration] = []
        sample_counts: dict[str, int] = {}
        for opponent in args.opponents:
            deal_seeds = tuple(
                stable_seed(args.master_seed, "imitation_deal", opponent, index)
                for index in range(args.deals_per_opponent)
            )
            samples = collect_demonstrations(
                _catalog_training_factory(args.teacher),
                _catalog_training_factory(opponent),
                deal_seeds,
                encoder=encoder,
                arena_config=rules,
                hand_id_prefix=f"bc-{args.teacher}-vs-{opponent}",
                policy_seed=stable_seed(args.master_seed, "imitation_policy", opponent),
            )
            demonstrations.extend(samples)
            sample_counts[opponent] = len(samples)
        training_config = BehaviorCloningConfig(
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=stable_seed(args.master_seed, "imitation_torch") % (2**63),
            split_seed=stable_seed(args.master_seed, "imitation_split"),
            validation_fraction=args.validation_fraction,
            device=args.device,
            policy_name=args.policy_name,
        )
        training_result = train_behavior_cloning(
            demonstrations,
            encoder,
            args.checkpoint,
            config=training_config,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        training_report = {
            "schema_version": 1,
            "status": "exploratory_training_complete",
            "teacher": args.teacher,
            "opponents": list(args.opponents),
            "master_seed": args.master_seed,
            "deal_count": args.deals_per_opponent * len(args.opponents),
            "demonstration_count": len(demonstrations),
            "demonstrations_by_opponent": sample_counts,
            "arena": asdict(rules),
            "encoder": encoder.to_dict(),
            "training": asdict(training_config),
            "device": training_result.device,
            "train_deal_count": len(training_result.split.train_deal_seeds),
            "validation_deal_count": len(training_result.split.validation_deal_seeds),
            "history": [asdict(metrics) for metrics in training_result.history],
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": training_result.checkpoint_sha256,
        }
        report_path.write_text(
            json.dumps(training_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(training_report, indent=2, sort_keys=True))
        return

    if args.command == "train-dqn":
        if args.training_hands < 1 or args.validation_hands < 1:
            raise ValueError("training-hands and validation-hands must be positive")
        if len(args.opponents) != len(set(args.opponents)):
            raise ValueError("opponent policy names must be unique")
        if args.initial_checkpoint is not None and args.encoder_checkpoint is not None:
            raise ValueError("initial-checkpoint and encoder-checkpoint are mutually exclusive")
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty directory: {args.output}")
        args.output.mkdir(parents=True, exist_ok=True)
        rules = ArenaConfig(
            starting_stack=args.starting_stack,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            ante=args.ante,
        )
        initial_policy = (
            NumpyMLPPolicy.from_checkpoint(args.initial_checkpoint)
            if args.initial_checkpoint is not None
            else None
        )
        encoder_policy = (
            NumpyMLPPolicy.from_checkpoint(args.encoder_checkpoint)
            if args.encoder_checkpoint is not None
            else None
        )
        encoder = (
            initial_policy.encoder
            if initial_policy is not None
            else encoder_policy.encoder
            if encoder_policy is not None
            else ObservationEncoder(
                EncoderConfig(
                    starting_stack=args.starting_stack,
                    big_blind=args.big_blind,
                    equity_samples=args.equity_samples,
                    equity_seed=stable_seed(args.master_seed, "dqn_encoder"),
                )
            )
        )
        dqn_config = DQNConfig(
            hidden_size=args.hidden_size,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            replay_capacity=args.replay_capacity,
            batch_size=args.batch_size,
            min_replay_size=args.min_replay_size,
            updates_per_transition=args.updates_per_transition,
            target_sync_interval=args.target_sync_interval,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_decisions=args.epsilon_decay_decisions,
            master_seed=args.master_seed,
            device=args.device,
            policy_name=args.policy_name,
        )
        trainer = DoubleDQNTrainer(
            config=dqn_config,
            encoder=encoder,
            arena_config=rules,
            initial_weights=(initial_policy.weights if initial_policy is not None else None),
        )
        training_reports: list[DQNHandReport] = []
        trainer.train_mode()
        for index in range(args.training_hands):
            opponent = args.opponents[index % len(args.opponents)]
            training_reports.append(trainer.play(_catalog_training_factory(opponent)))
            completed = index + 1
            if completed == args.training_hands or (
                args.progress_every > 0 and completed % args.progress_every == 0
            ):
                print(f"trained {completed}/{args.training_hands} hands", flush=True)

        validation_reports: list[DQNHandReport] = []
        validation_opponents: list[str] = []
        trainer.eval_mode()
        for index in range(args.validation_hands):
            opponent = args.opponents[index % len(args.opponents)]
            validation_reports.append(trainer.play(_catalog_training_factory(opponent)))
            validation_opponents.append(opponent)

        trainer_path = args.output / "trainer.pt"
        inference_path = args.output / "policy.npz"
        trainer_sha256 = trainer.save_training_checkpoint(trainer_path)
        inference_sha256 = trainer.export_inference_checkpoint(inference_path)
        per_opponent_validation = {
            opponent: _dqn_report_summary(
                [
                    report
                    for report, report_opponent in zip(
                        validation_reports,
                        validation_opponents,
                        strict=True,
                    )
                    if report_opponent == opponent
                ]
            )
            for opponent in args.opponents
        }
        dqn_report = {
            "schema_version": 1,
            "status": "exploratory_training_complete",
            "evaluation_note": (
                "Validation is alternating-seat, held-out-deck diagnostic play; "
                "it is not duplicate confirmatory evidence."
            ),
            "opponents": list(args.opponents),
            "opponent_schedule": "round_robin_by_hand",
            "arena": asdict(rules),
            "encoder": encoder.to_dict(),
            "training": asdict(dqn_config),
            "device": trainer.device,
            "initial_checkpoint": (
                str(args.initial_checkpoint) if args.initial_checkpoint is not None else None
            ),
            "initial_checkpoint_sha256": (
                initial_policy.checkpoint_sha256 if initial_policy is not None else None
            ),
            "encoder_checkpoint": (
                str(args.encoder_checkpoint) if args.encoder_checkpoint is not None else None
            ),
            "encoder_checkpoint_sha256": (
                encoder_policy.checkpoint_sha256 if encoder_policy is not None else None
            ),
            "replay_size": len(trainer.replay),
            "environment_decisions": trainer.environment_decisions,
            "optimization_steps": trainer.optimization_steps,
            "training_summary": _dqn_report_summary(training_reports),
            "validation_summary": _dqn_report_summary(validation_reports),
            "validation_by_opponent": per_opponent_validation,
            "trainer_checkpoint": str(trainer_path),
            "trainer_checkpoint_sha256": trainer_sha256,
            "trainer_checkpoint_trusted_local_only": True,
            "inference_checkpoint": str(inference_path),
            "inference_checkpoint_sha256": inference_sha256,
        }
        report_path = args.output / "training.json"
        report_path.write_text(
            json.dumps(dqn_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(dqn_report, indent=2, sort_keys=True))
        return

    if args.command == "power":
        pair_count = required_pair_count(
            args.standard_deviation,
            args.effect,
            alpha=args.alpha,
            power=args.power,
        )
        print(json.dumps({"required_duplicate_pairs": pair_count}))
        return

    if args.command == "theory":
        points = run_cfr_verification(
            game_name=args.game,
            algorithm=args.algorithm,
            iterations=args.iterations,
            report_every=args.report_every,
        )
        if args.output is not None:
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite theory output: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(points[0].to_dict()))
                writer.writeheader()
                writer.writerows(point.to_dict() for point in points)
        print(json.dumps(points[-1].to_dict(), indent=2, sort_keys=True))
        return

    if args.command == "validate-run":
        report = validate_run(args.directory)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        if not report.passed:
            raise SystemExit(1)
        return

    config = load_benchmark_config(args.config)

    def show_progress(done: int, total: int) -> None:
        if done == total or (args.progress_every > 0 and done % args.progress_every == 0):
            print(f"completed {done}/{total} duplicate pairs", flush=True)

    benchmark_result = run_benchmark(
        config,
        output_directory=args.output,
        config_path=args.config,
        progress=show_progress,
    )
    print(json.dumps(benchmark_result.summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
