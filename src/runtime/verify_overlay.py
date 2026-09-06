"""Verify that a generated NPU BIT/HWH pair matches its build manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


EXPECTED_METADATA = {
    "accelerator": {
        "instance": "npu_matrix_accelerator_0",
        "parameters": {
            "ROWS": "2",
            "COLUMNS": "2",
            "MAX_K": "256",
            "C_BASEADDR": "0x43C00000",
            "C_HIGHADDR": "0x43C0FFFF",
        },
    },
    "dma": {
        "instance": "axi_dma_0",
        "parameters": {
            "C_INCLUDE_SG": "0",
            "C_INCLUDE_MM2S": "1",
            "C_INCLUDE_S2MM": "1",
            "C_M_AXIS_MM2S_TDATA_WIDTH": "8",
            "C_S_AXIS_S2MM_TDATA_WIDTH": "32",
            "C_BASEADDR": "0x40400000",
            "C_HIGHADDR": "0x4040FFFF",
        },
    },
}


class OverlayVerificationError(RuntimeError):
    """Generated artifacts or their provenance do not satisfy the contract."""


def _expected_metadata(array_size: int) -> dict[str, object]:
    if array_size not in (2, 8):
        raise OverlayVerificationError(
            f"unsupported array size {array_size}: expected 2 or 8"
        )
    expected = copy.deepcopy(EXPECTED_METADATA)
    parameters = expected["accelerator"]["parameters"]
    parameters["ROWS"] = str(array_size)
    parameters["COLUMNS"] = str(array_size)
    return expected


def _validated_commit(value: str) -> str:
    commit = value.strip()
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", commit) is None:
        raise OverlayVerificationError(
            "source_commit must be a full hexadecimal Git object id"
        )
    return commit.lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_parameters(hwh_path: Path, instance: str) -> dict[str, str]:
    root = ET.parse(hwh_path).getroot()
    for module in root.iter("MODULE"):
        if module.attrib.get("INSTANCE") == instance:
            return {
                parameter.attrib["NAME"].upper(): parameter.attrib.get("VALUE", "")
                for parameter in module.iter("PARAMETER")
                if "NAME" in parameter.attrib
            }
    raise OverlayVerificationError(f"HWH is missing module {instance}")


def inspect_hwh(hwh_path: Path, *, array_size: int = 2) -> dict[str, object]:
    observed: dict[str, object] = {}
    text = hwh_path.read_text(encoding="utf-8")
    for role, expectation in _expected_metadata(array_size).items():
        instance = str(expectation["instance"])
        parameters = _module_parameters(hwh_path, instance)
        expected_parameters = dict(expectation["parameters"])
        for name, expected in expected_parameters.items():
            actual = parameters.get(name)
            if actual is None or actual.lower() != expected.lower():
                raise OverlayVerificationError(
                    f"{instance} {name} expected {expected}, found {actual!r}"
                )
        observed[role] = {"instance": instance, "parameters": expected_parameters}
    required_connections = (
        'INSTANCE="npu_matrix_accelerator_0" PORT="s_axis_tdata"',
        'INSTANCE="npu_matrix_accelerator_0" PORT="m_axis_tdata"',
        'INSTANCE="npu_matrix_accelerator_0" PORT="irq"',
        'SLAVEBUSINTERFACE="S_AXI_HP0"',
        'VALUE="100000000"',
    )
    missing = [connection for connection in required_connections if connection not in text]
    if missing:
        raise OverlayVerificationError(f"HWH is missing connectivity metadata: {missing}")
    return observed


def _artifact_paths(artifact_dir: Path) -> tuple[Path, Path, Path]:
    bit_path = artifact_dir / "npu_matrix.bit"
    hwh_path = artifact_dir / "npu_matrix.hwh"
    manifest_path = artifact_dir / "npu_matrix.manifest.json"
    if bit_path.stem != hwh_path.stem:
        raise OverlayVerificationError("BIT and HWH basenames differ")
    for path in (bit_path, hwh_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise OverlayVerificationError(f"missing or empty artifact: {path}")
    return bit_path, hwh_path, manifest_path


def write_manifest(
    artifact_dir: Path,
    *,
    source_commit: str,
    vivado_version: str,
    array_size: int = 2,
) -> dict[str, object]:
    bit_path, hwh_path, manifest_path = _artifact_paths(artifact_dir)
    metadata = inspect_hwh(hwh_path, array_size=array_size)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_commit": _validated_commit(source_commit),
        "vivado_version": vivado_version.strip(),
        "target_part": "xc7z020clg400-1",
        "array_size": array_size,
        "bit": {
            "name": bit_path.name,
            "size": bit_path.stat().st_size,
            "mtime_ns": bit_path.stat().st_mtime_ns,
            "sha256": _sha256(bit_path),
        },
        "hwh": {
            "name": hwh_path.name,
            "size": hwh_path.stat().st_size,
            "mtime_ns": hwh_path.stat().st_mtime_ns,
            "sha256": _sha256(hwh_path),
        },
        "metadata": metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_artifacts(artifact_dir: Path) -> dict[str, object]:
    bit_path, hwh_path, manifest_path = _artifact_paths(artifact_dir)
    if not manifest_path.is_file():
        raise OverlayVerificationError("overlay provenance manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("target_part") != "xc7z020clg400-1":
        raise OverlayVerificationError("unsupported manifest schema or target part")
    _validated_commit(str(manifest.get("source_commit", "")))
    for label, path in (("bit", bit_path), ("hwh", hwh_path)):
        record = manifest.get(label)
        if not isinstance(record, dict):
            raise OverlayVerificationError(f"manifest lacks {label} record")
        if record.get("name") != path.name or record.get("size") != path.stat().st_size:
            raise OverlayVerificationError(f"{label} name or size does not match manifest")
        if record.get("sha256") != _sha256(path):
            raise OverlayVerificationError(f"{label} hash does not match manifest")
    array_size = manifest.get("array_size", 2)
    if isinstance(array_size, bool) or not isinstance(array_size, int):
        raise OverlayVerificationError("manifest array_size must be 2 or 8")
    observed = inspect_hwh(hwh_path, array_size=array_size)
    if manifest.get("metadata") != observed:
        raise OverlayVerificationError("HWH metadata differs from the build manifest")
    return manifest


def _default_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--source-commit", default=_default_commit())
    parser.add_argument("--vivado-version", default="unknown")
    parser.add_argument("--array-size", type=int, choices=(2, 8), default=2)
    arguments = parser.parse_args()
    artifact_dir = arguments.artifact_dir.resolve()
    if arguments.write_manifest:
        write_manifest(
            artifact_dir,
            source_commit=arguments.source_commit,
            vivado_version=arguments.vivado_version,
            array_size=arguments.array_size,
        )
    verify_artifacts(artifact_dir)
    print("PASS: npu_matrix BIT/HWH provenance and metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
