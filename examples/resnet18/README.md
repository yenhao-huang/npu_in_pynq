# ResNet-18 real-model workflow

This example downloads one pinned official TorchVision ResNet-18, converts it
to the repository's Phase 2A signed-INT8 format, checks a real
`(1, 224, 224, 3)` input through an independent integer reference and the
production model runtime, and creates a deterministic model archive.

The calibration image is synthetic and unlabeled. A host PASS proves importer
and runtime agreement; it is not ImageNet accuracy or physical-board evidence.

Steps 1 through 7 prepare and copy the release from the Windows development
host. Step 8 is the human demo and runs from the deployed `.ipynb` on the
PYNQ-Z1, where the `pynq` package, Overlay, MMIO, and DMA are available.

```
python -m venv build/resnet18-venv
.\build\resnet18-venv\Scripts\activate
```

## 1. Prepare the conversion environment

From the repository root in PowerShell:

```powershell
& build/resnet18-venv/Scripts/python.exe -m pip install `
  --extra-index-url https://download.pytorch.org/whl/cpu `
  -r examples/resnet18/requirements-convert.txt
```

Stop if dependency installation fails. PyTorch is used only on the conversion
host; the exported runtime remains NumPy-only.

## 2. Download the pinned checkpoint

```powershell
& build/resnet18-venv/Scripts/python.exe `
  examples/resnet18/scripts/download_model.py
```

Expected marker: `PASS: downloaded and verified ...resnet18-f37072fd.pth`.
The script rejects redirects to another host, incorrect length or SHA-256, and
an existing destination. Generated files stay under
`examples/resnet18/model/` and are ignored by Git.

## 3. Convert the model

```powershell
& build/resnet18-venv/Scripts/python.exe `
  examples/resnet18/scripts/convert_model.py
```

Expected markers report `resnet18.npu.json` and
`resnet18.conversion.json`. Stop on any source-schema, non-finite value,
quantization, accumulator certificate, or exporter error.

## 4. Validate the real model

```powershell
& examples/resnet18/scripts/verify.ps1 `
  -Python build/resnet18-venv/Scripts/python.exe
```

Expected marker: `PASS [real-model-host]`. The validator reloads the exported
package and runs all 1,814,073,344 MACs through `NPUModelRuntime`. Its
`stem.relu`, `layer1.1.relu`, and `logits` outputs must match the independent
vectorized integer reference exactly. It writes `model/acceptance.json` only
after every comparison passes.

## 5. Build or select trusted Vivado artifacts

This step requires a licensed Vivado host and cannot be replaced by CI fixture
artifacts:

```powershell
vivado -mode batch -nojournal -nolog `
  -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl
python -m src.runtime.verify_overlay `
  build/vivado/npu_matrix_8x8/artifacts
```

Stop unless the verification marker says the BIT/HWH provenance and metadata
passed and the artifact manifest identifies the intended source commit.

## 6. Build the model archive

```powershell
python examples/resnet18/package_example.py `
  --output-archive mount/resnet18/resnet18-model.zip
```

Expected marker: `PASS [real-model-host]`. The checkpoint itself is not
redistributed. Missing, stale, substituted, incomplete, or unvalidated model
workspaces publish no archive. Issue #7 combines this validated model boundary
with the matching trusted overlay for standalone board delivery.

## 7. Deploy to the PYNQ-Z1

`run_on_board.py` must not be launched from the Windows virtual environment.
Use the deployment wrapper to copy the required Python sources, ignored model
workspace, and verified Vivado artifacts to the board. The default
board endpoint is `xilinx@192.168.2.99`; override `-BoardHost` or
`-BoardUser` when necessary:

```powershell
& examples/resnet18/deploy_release.ps1 `
  -DeploymentId (Get-Date -Format yyyyMMdd-HHmmss) `
  -AllowArtifactCommitMismatch
```

This wrapper only creates an immutable release directory and copies the
example, shared runtime/export/model sources, Vivado artifacts, and deployment
metadata. It does not run the model, invoke `sudo`, claim a PASS, or retrieve
evidence. The `-AllowArtifactCommitMismatch` choice is recorded for the later
human validation; omit it when the artifacts were built from this exact
checkout.

## 8. Open the notebook and perform human validation

In the PYNQ Jupyter interface (e.g., http://192.168.2.99:9090/notebooks/), open the release directory printed by the
deployment wrapper, then open:

```text
examples/resnet18/resnet18.ipynb
```

Select the board's PYNQ Python kernel and run one cell at a time. The notebook
does not hide acceptance behind `run_on_board.py`. It separately exposes the
deployment provenance, model file digests, BIT/HWH verification, reconstructed
model graph, physical `NPURuntime` identity, execution metrics, and every
expected/actual output hash.

After reviewing those results, change `human_approves = False` to `True` in
the final cell and execute that cell. Only this explicit approval writes a new
`notebook-evidence-<UTC timestamp>.json` and prints one of these markers:

```text
PASS [physical-pynq-z1]: human-reviewed notebook demo
PASS [physical-pynq-z1-development]: human-reviewed notebook demo
```

The development marker means an artifact/check-out commit mismatch was
explicitly allowed. It is execution evidence, not trusted release acceptance.

For terminal-oriented verification, `run_on_board.py` remains an alternative
low-level entry point. Run it only on the PYNQ-Z1, using the commit values from
`deployment.json`:

```bash
source /etc/profile.d/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
cd /home/xilinx/jupyter_notebooks/npu_resnet18/releases/<deployment-id>
sudo XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
  examples/resnet18/run_on_board.py \
  --artifact-dir build/vivado/npu_matrix_8x8/artifacts \
  --expected-source-commit <40-character-artifact-commit> \
  --deployed-source-commit <40-character-deployed-commit> \
  --evidence board-evidence.json
```

The notebook calls the public verification and runtime APIs directly so each
boundary remains visible; the CLI composes the same checks for terminal and CD
use. Both routes require an actual `NPURuntime`, and host backends cannot emit
a physical PASS marker. Automated deployment and evidence collection belong
to the CD script under `.github/cd/`, not to this human demo workflow.

## Re-running generated steps

All download, conversion, validation, and archive outputs are intentionally
write-once. To repeat a step, choose a new output path or deliberately remove
only the corresponding ignored generated files after preserving any evidence
you need. The scripts never overwrite prior evidence silently.
