#!/usr/bin/env python3
"""Fail CI when packaging metadata drifts from the source tree."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PACKAGE_MARKERS = [
    "kalmanorix/benchmarks/__init__.py",
    "kalmanorix/experimental/__init__.py",
    "kalmanorix/internal/__init__.py",
]
EXPECTED_ENTRYPOINTS = [
    "kalmanorix-build-mixed-benchmark",
    "kalmanorix-run-benchmark",
    "kalmanorix-generate-report",
    "kalmanorix-run-canonical-benchmark",
    "kalmanorix-eval-routing",
    "kalmanorix-run-correlation-aware-fusion",
    "kalmanorix-run-uncertainty-calibration",
]


def _assert_no_committed_egg_info() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "*.egg-info", "*.egg-info/*"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if tracked:
        raise RuntimeError(
            "Generated egg-info files are committed. Remove them from version control:\n"
            f"{tracked}"
        )


def _build_dist(dist_dir: Path) -> tuple[Path, Path]:
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(dist_dir)],
        cwd=REPO_ROOT,
        check=True,
    )
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one sdist and one wheel, got sdists={sdists}, wheels={wheels}")
    return sdists[0], wheels[0]


def _assert_sdist_contains_markers(sdist: Path) -> None:
    with tarfile.open(sdist, "r:gz") as tf:
        names = {name.split("/", 1)[1] for name in tf.getnames() if "/" in name}
    missing = [path for path in EXPECTED_PACKAGE_MARKERS if f"src/{path}" not in names]
    if missing:
        raise RuntimeError(f"sdist missing expected package files: {missing}")


def _assert_wheel_contains_markers_and_entrypoints(wheel: Path) -> None:
    with zipfile.ZipFile(wheel, "r") as zf:
        names = set(zf.namelist())
        missing = [path for path in EXPECTED_PACKAGE_MARKERS if path not in names]
        if missing:
            raise RuntimeError(f"wheel missing expected package files: {missing}")

        entrypoints_path = next((name for name in names if name.endswith(".dist-info/entry_points.txt")), None)
        if entrypoints_path is None:
            raise RuntimeError("wheel is missing .dist-info/entry_points.txt")

        entrypoints = zf.read(entrypoints_path).decode("utf-8")
        missing_entrypoints = [ep for ep in EXPECTED_ENTRYPOINTS if f"{ep} =" not in entrypoints]
        if missing_entrypoints:
            raise RuntimeError(f"wheel missing expected console scripts: {missing_entrypoints}")


def main() -> None:
    _assert_no_committed_egg_info()

    with tempfile.TemporaryDirectory(prefix="kalmanorix-packaging-") as tmp:
        dist_dir = Path(tmp) / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        sdist, wheel = _build_dist(dist_dir)
        _assert_sdist_contains_markers(sdist)
        _assert_wheel_contains_markers_and_entrypoints(wheel)

        # Cleanup not strictly necessary due to TemporaryDirectory, but explicit for clarity.
        shutil.rmtree(dist_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
