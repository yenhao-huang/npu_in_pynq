"""Non-interactive Phase 1C acceptance runner for a packaged PYNQ overlay."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT if (PACKAGE_ROOT / "src").is_dir() else PACKAGE_ROOT.parents[1]
for import_root in (REPOSITORY_ROOT, PACKAGE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from runtime.matrix_multiplication import TiledMatrixMultiplier
from src.runtime import load_pynq_runtime
from src.runtime.verify_overlay import verify_artifacts


PASS_MARKER = "PASS: Phase 1C matrix multiplication example"


class BoardExampleError(RuntimeError):
    """The packaged Phase 1C example did not satisfy board acceptance."""


def _reference(a_matrix: np.ndarray, b_matrix: np.ndarray) -> np.ndarray:
    return (a_matrix.astype(np.int64) @ b_matrix.astype(np.int64)).astype(np.int32)


def _case_record(name: str, result: Any) -> dict[str, object]:
    metrics = result.metrics
    throughput = float(metrics.operations_per_second)
    return {
        "name": name,
        "status": "PASS",
        "shape": [metrics.m, metrics.k, metrics.n],
        "tile_count": metrics.tile_count,
        "elapsed_seconds": metrics.elapsed_seconds,
        "operation_count": metrics.operation_count,
        "operations_per_second": throughput if math.isfinite(throughput) else None,
    }


def execute_cases(
    runtime: Any,
    manifest: dict[str, object],
    *,
    release_tag: str,
) -> dict[str, object]:
    """Execute required cases through public runtimes and return PASS evidence."""

    multiplier = TiledMatrixMultiplier(runtime)
    normal_a = ((np.arange(runtime.max_m * 5).reshape(runtime.max_m, 5) % 17) - 8).astype(np.int8)
    normal_b = ((np.arange(5 * runtime.max_n).reshape(5, runtime.max_n) % 19) - 9).astype(np.int8)
    non_aligned_a = ((np.arange((runtime.max_m + 1) * 5).reshape(runtime.max_m + 1, 5) % 17) - 8).astype(np.int8)
    non_aligned_b = ((np.arange(5 * (runtime.max_n + 1)).reshape(5, runtime.max_n + 1) % 19) - 9).astype(np.int8)
    cases = (
        ("normal", normal_a, normal_b),
        ("non_aligned", non_aligned_a, non_aligned_b),
        ("repeated", -non_aligned_a, non_aligned_b),
    )

    records: list[dict[str, object]] = []
    for name, a_matrix, b_matrix in cases:
        try:
            result = multiplier.run(a_matrix, b_matrix, software_timeout=10.0)
            np.testing.assert_array_equal(result.output, _reference(a_matrix, b_matrix))
        except Exception as error:
            raise BoardExampleError(f"{name} matrix case failed") from error
        records.append(_case_record(name, result))

    if int(records[1]["tile_count"]) <= 1:
        raise BoardExampleError("non-aligned case did not exercise physical tiling")

    return {
        "schema_version": 1,
        "release_tag": release_tag,
        "source_commit": manifest.get("source_commit", "unknown"),
        "vivado_version": manifest.get("vivado_version", "unknown"),
        "target_part": manifest.get("target_part", "unknown"),
        "bit": manifest.get("bit", {}),
        "hwh": manifest.get("hwh", {}),
        "physical_limits": [runtime.max_m, runtime.max_n, runtime.max_k],
        "cases": records,
        "pass_marker": PASS_MARKER,
    }


def _write_evidence(path: Path, evidence: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    arguments = parser.parse_args()
    try:
        artifact_dir = arguments.artifact_dir.resolve()
        manifest = verify_artifacts(artifact_dir)
        runtime = load_pynq_runtime(artifact_dir / "npu_matrix.bit")
        evidence = execute_cases(
            runtime,
            manifest,
            release_tag=arguments.release_tag,
        )
        _write_evidence(arguments.evidence, evidence)
    except Exception as error:
        print(f"Phase 1C board example failed: {error}", file=sys.stderr)
        return 1
    print(PASS_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
