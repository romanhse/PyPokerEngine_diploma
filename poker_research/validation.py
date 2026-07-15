"""Independent validation of completed benchmark artifact directories."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from pokerkit import HandHistory


@dataclass(frozen=True, slots=True)
class QualityCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    directory: str
    passed: bool
    checks: tuple[QualityCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_run(directory: Path) -> DataQualityReport:
    """Validate provenance, duplicate design, accounting, decisions, and PHH count."""

    checks: list[QualityCheck] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(QualityCheck(name, passed, detail))

    required = {
        "decisions.jsonl",
        "errors.jsonl",
        "hands.csv",
        "hands.phhs",
        "manifest.json",
        "pairs.csv",
        "preregistered_config.toml",
        "summary.json",
    }
    missing = sorted(name for name in required if not (directory / name).is_file())
    add("required_files", not missing, "all present" if not missing else f"missing={missing}")
    if missing:
        return DataQualityReport(str(directory), False, tuple(checks))

    manifest = _read_json(directory / "manifest.json")
    summary = _read_json(directory / "summary.json")
    configured_pairs = int(manifest["config"]["pair_count"])
    big_blind = int(manifest["config"]["game"]["big_blind"])
    run_id = str(manifest["run_id"])
    add("manifest_status", manifest.get("status") == "complete", str(manifest.get("status")))

    artifact_hashes = manifest.get("artifacts", {})
    hash_failures = [
        name
        for name, expected in artifact_hashes.items()
        if _sha256(directory / name) != expected
    ]
    add(
        "artifact_hashes",
        not hash_failures and len(artifact_hashes) >= 6,
        f"verified={len(artifact_hashes)}, mismatched={hash_failures}",
    )

    pair_rows = _read_csv(directory / "pairs.csv")
    hand_rows = _read_csv(directory / "hands.csv")
    decision_rows = _read_json_lines(directory / "decisions.jsonl")
    error_rows = _read_json_lines(directory / "errors.jsonl")
    add("no_recorded_errors", not error_rows, f"errors={len(error_rows)}")
    add(
        "row_counts",
        len(pair_rows) == configured_pairs and len(hand_rows) == configured_pairs * 2,
        f"pairs={len(pair_rows)}, hands={len(hand_rows)}, configured_pairs={configured_pairs}",
    )
    add(
        "summary_pair_count",
        int(summary.get("pair_count", -1)) == configured_pairs,
        f"summary={summary.get('pair_count')}",
    )

    pair_ids = [int(row["pair_id"]) for row in pair_rows]
    deal_seeds = [int(row["deal_seed"]) for row in pair_rows]
    add(
        "unique_pairs_and_seeds",
        len(set(pair_ids)) == configured_pairs and len(set(deal_seeds)) == configured_pairs,
        f"unique_pairs={len(set(pair_ids))}, unique_seeds={len(set(deal_seeds))}",
    )

    hands_by_pair: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in hand_rows:
        hands_by_pair[int(row["pair_id"])].append(row)
    duplicate_failures: list[int] = []
    accounting_failures: list[str] = []
    for pair_id, legs in hands_by_pair.items():
        if (
            len(legs) != 2
            or {leg["leg"] for leg in legs} != {"a", "b"}
            or len({leg["deal_seed"] for leg in legs}) != 1
            or len({leg["deck_hash"] for leg in legs}) != 1
            or {(leg["leg"], leg["hero_seat"]) for leg in legs} != {("a", "0"), ("b", "1")}
        ):
            duplicate_failures.append(pair_id)
        for leg in legs:
            if int(leg["seat0_payoff"]) + int(leg["seat1_payoff"]) != 0:
                accounting_failures.append(f"{pair_id}{leg['leg']}")
    add(
        "duplicate_pairing",
        not duplicate_failures and len(hands_by_pair) == configured_pairs,
        f"invalid_pairs={duplicate_failures[:10]}",
    )
    add(
        "zero_sum_accounting",
        not accounting_failures,
        f"invalid_hands={accounting_failures[:10]}",
    )

    pairs_by_id = {int(row["pair_id"]): row for row in pair_rows}
    payoff_failures: list[int] = []
    for pair_id, legs in hands_by_pair.items():
        if pair_id not in pairs_by_id or len(legs) != 2:
            payoff_failures.append(pair_id)
            continue
        by_leg = {leg["leg"]: leg for leg in legs}
        hero_a = int(by_leg["a"]["hero_payoff_chips"])
        hero_b = int(by_leg["b"]["hero_payoff_chips"])
        pair = pairs_by_id[pair_id]
        expected_bb100 = 50.0 * (hero_a + hero_b) / big_blind
        if (
            int(pair["hero_payoff_leg_a"]) != hero_a
            or int(pair["hero_payoff_leg_b"]) != hero_b
            or abs(float(pair["hero_bb100"]) - expected_bb100) > 1e-9
        ):
            payoff_failures.append(pair_id)
    add("pair_payoff_formula", not payoff_failures, f"invalid_pairs={payoff_failures[:10]}")

    invalid_decisions: list[str] = []
    for row in decision_rows:
        legal_actions = cast(list[dict[str, Any]], row["legal_actions"])
        decision = cast(dict[str, Any], row["decision"])
        raw_probabilities = cast(list[list[Any]], decision["probabilities"])
        legal_names = {str(option["name"]) for option in legal_actions}
        probabilities: dict[str, float] = {
            str(name): float(probability) for name, probability in raw_probabilities
        }
        if (
            row.get("run_id") != run_id
            or decision["action"] not in legal_names
            or len(probabilities) != len(raw_probabilities)
            or set(probabilities) != legal_names
            or any(
                not math.isfinite(value) or value < 0 or value > 1
                for value in probabilities.values()
            )
            or not math.isclose(
                math.fsum(probabilities.values()),
                1.0,
                abs_tol=1e-9,
            )
            or probabilities.get(str(decision["action"]), 0.0) <= 0
        ):
            invalid_decisions.append(
                f"{row.get('pair_id')}:{row.get('leg')}:{row.get('index')}"
            )
    add(
        "decision_contract",
        bool(decision_rows) and not invalid_decisions,
        f"decisions={len(decision_rows)}, invalid={invalid_decisions[:10]}",
    )

    with (directory / "hands.phhs").open("rb") as phh_file:
        phh_count = sum(1 for _ in HandHistory.load_all(phh_file))
    add("phh_count", phh_count == len(hand_rows), f"phh={phh_count}, hands={len(hand_rows)}")

    passed = all(check.passed for check in checks)
    return DataQualityReport(str(directory), passed, tuple(checks))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object per line: {path}")
        rows.append(payload)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
