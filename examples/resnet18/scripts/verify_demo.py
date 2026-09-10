"""Validate the 8x8 ResNet demo inputs without loading or executing hardware."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from examples.resnet18.package_example import validate_workspace
from src.runtime.model import load_model_package
from src.runtime.verify_overlay import verify_artifacts


def validate_demo(artifact_dir: Path, model_dir: Path) -> dict:
    overlay = verify_artifacts(artifact_dir)
    if overlay.get("array_size") != 8:
        raise ValueError("ResNet demo requires an 8x8 overlay")
    validate_workspace(model_dir, ROOT / "examples/resnet18/model-source.json")
    model = load_model_package(model_dir / "resnet18.npu.json")
    return {
        "array_size": 8,
        "artifact_source_commit": overlay["source_commit"],
        "model_commands": len(model.graph.commands),
        "model_outputs": list(model.graph.outputs),
        "hardware_executed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path,
                        default=ROOT / "build/vivado/npu_matrix_8x8/artifacts")
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "examples/resnet18/model")
    args = parser.parse_args()
    print(json.dumps(validate_demo(args.artifact_dir, args.model_dir), indent=2))
    print("PASS: ResNet 8x8 demo inputs verified; hardware not executed")
