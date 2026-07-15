"""Reusable provenance helpers for durable research artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Hash a file without loading large artifacts into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance() -> dict[str, Any]:
    """Return the checked-out revision and whether any tracked/untracked files differ."""

    try:
        sha = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"sha": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def dependency_versions(
    packages: tuple[str, ...] = ("numpy", "phevaluator", "pokerkit", "scipy"),
) -> dict[str, str]:
    """Resolve installed versions for the packages that affect poker evaluation."""

    return {package: importlib.metadata.version(package) for package in packages}


def source_provenance() -> dict[str, Any]:
    """Hash the complete canonical research layer and its locked environment."""

    project_root = Path(__file__).resolve().parents[1]
    paths = sorted((project_root / "poker_research").glob("*.py"))
    paths.extend(project_root / name for name in ("pyproject.toml", "uv.lock"))
    files = {
        str(path.relative_to(project_root)): sha256_file(path)
        for path in paths
        if path.is_file()
    }
    combined = hashlib.sha256()
    for relative_path, digest in sorted(files.items()):
        combined.update(relative_path.encode())
        combined.update(b"\0")
        combined.update(digest.encode())
        combined.update(b"\0")
    return {"tree_sha256": combined.hexdigest(), "files": files}
