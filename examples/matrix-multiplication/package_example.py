"""Assemble the standalone Phase 1C PYNQ deployment package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.runtime.verify_overlay import verify_artifacts


PACKAGE_LABEL_PATTERN = re.compile(
    r"(?:v[0-9]+\.[0-9]+\.[0-9]+|local-[0-9a-fA-F]{8,64})"
)
COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40,64}")

SOURCE_FILES = {
    "examples/matrix-multiplication/README.md": "README.md",
    "examples/matrix-multiplication/matrix_multiplication.ipynb": (
        "matrix_multiplication.ipynb"
    ),
    "examples/matrix-multiplication/run_on_board.py": "run_on_board.py",
    "examples/matrix-multiplication/runtime/matrix_multiplication.py": (
        "runtime/matrix_multiplication.py"
    ),
    "examples/matrix-multiplication/runtime/npu_package_init.py": (
        "src/runtime/__init__.py"
    ),
    "src/runtime/npu.py": "src/runtime/npu.py",
    "src/runtime/verify_overlay.py": "src/runtime/verify_overlay.py",
}
ARTIFACT_FILES = (
    "npu_matrix.bit",
    "npu_matrix.hwh",
    "npu_matrix.manifest.json",
)


class PackageError(RuntimeError):
    """The requested deployment package is incomplete or unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_release_tag(value: str) -> str:
    tag = value.strip()
    if PACKAGE_LABEL_PATTERN.fullmatch(tag) is None:
        raise PackageError(
            "package label must match vMAJOR.MINOR.PATCH or local-<commit>"
        )
    return tag


def _validated_commit(value: str) -> str:
    commit = value.strip().lower()
    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise PackageError("source commit must be a full hexadecimal object id")
    return commit


def _required_files(
    repository_root: Path, artifact_dir: Path
) -> list[tuple[Path, Path]]:
    files = [
        (repository_root / source, Path(destination))
        for source, destination in SOURCE_FILES.items()
    ]
    files.extend(
        (artifact_dir / name, Path("artifacts") / name) for name in ARTIFACT_FILES
    )
    missing = [str(source) for source, _ in files if not source.is_file()]
    if missing:
        raise PackageError(f"required package inputs are missing: {missing}")
    return files


def _copy_files(
    files: Iterable[tuple[Path, Path]], staging_dir: Path
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for source, relative_destination in files:
        destination = staging_dir / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "path": relative_destination.as_posix(),
                "size": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    return sorted(records, key=lambda record: str(record["path"]))


def build_package(
    *,
    repository_root: Path,
    artifact_dir: Path,
    output_dir: Path,
    release_tag: str,
    source_commit: str,
) -> dict[str, object]:
    """Build one complete package or fail without publishing a partial output."""

    repository_root = repository_root.resolve()
    artifact_dir = artifact_dir.resolve()
    output_dir = output_dir.resolve()
    tag = _validated_release_tag(release_tag)
    commit = _validated_commit(source_commit)
    if output_dir.exists():
        raise PackageError(f"output directory already exists: {output_dir}")

    overlay_manifest = verify_artifacts(artifact_dir)
    if str(overlay_manifest.get("source_commit", "")).lower() != commit:
        raise PackageError("overlay manifest source commit differs from release commit")
    files = _required_files(repository_root, artifact_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        records = _copy_files(files, staging_dir)
        package_manifest: dict[str, object] = {
            "schema_version": 1,
            "release_tag": tag,
            "source_commit": commit,
            "target_part": overlay_manifest["target_part"],
            "files": records,
        }
        (staging_dir / "package.manifest.json").write_text(
            json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging_dir.replace(output_dir)
        return package_manifest
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--release-tag")
    parser.add_argument("--source-commit")
    arguments = parser.parse_args()

    repository_root = arguments.repository_root.resolve()
    artifact_dir = (
        arguments.artifact_dir
        or repository_root / "build" / "vivado" / "npu_matrix_8x8" / "artifacts"
    ).resolve()
    overlay_manifest = verify_artifacts(artifact_dir)
    source_commit = arguments.source_commit or str(overlay_manifest["source_commit"])
    release_tag = arguments.release_tag or f"local-{source_commit[:8]}"
    output_dir = (
        arguments.output_dir
        or repository_root / "mount" / "matrix-multiplication" / release_tag
    ).resolve()

    build_package(
        repository_root=repository_root,
        artifact_dir=artifact_dir,
        output_dir=output_dir,
        release_tag=release_tag,
        source_commit=source_commit,
    )
    print(f"PASS: standalone matrix example package at {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
