# Matrix multiplication deployment

This example defaults to an 8 x 8 systolic array. It builds the NPU overlay,
deploys it to PYNQ-Z1, runs three matrix tests, and downloads JSON evidence. Run all PowerShell commands from the
repository root.

## Requirements

- Vivado, Python, `ssh`, `scp`, and `tar` are available.
- The board is reachable as `pynq` or `192.168.2.99`.
- The Git worktree is clean before building deployable artifacts:

```powershell
git status --short
```

The command must print nothing. A dirty exploratory build may use
`-tclargs --allow-dirty`, but it deliberately does not publish deployable
artifacts.

## 1. Prepare the board once

```powershell
ssh pynq
sudo usermod -aG render,video xilinx
exit
ssh pynq id
```

The final output should include `render` and `video`. PYNQ also requires root
for `/dev/mem` MMIO, so local deployment will ask for the board sudo password.

## 2. Build and package

```powershell
vivado -mode batch -nojournal -nolog `
  -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl

python examples/matrix-multiplication/package_example.py
```

The package is created under
`mount/matrix-multiplication/local-<commit>`.

## 3. Deploy and test

```powershell
$package = Get-ChildItem mount/matrix-multiplication/local-* -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if (-not $package) { throw 'Build the package first.' }

$deploy = @{
  PackagePath = $package.FullName
  ReleaseTag = 'v0.0.0'
  DeploymentId = Get-Date -Format 'yyyyMMdd-HHmmss'
  EvidencePath = 'build/board/local-evidence.json'
  BoardHost = 'pynq'
}

& examples/matrix-multiplication/deploy_release.ps1 @deploy -DryRun
& examples/matrix-multiplication/deploy_release.ps1 @deploy -InteractiveSudo
```

Enter the board sudo password when prompted. Success prints:

```text
PASS: Phase 1C matrix multiplication example
PASS: deployed v0.0.0 as <deployment-id> and retrieved board evidence
```

Read the evidence:

```powershell
Get-Content build/board/local-evidence.json
```

## Release CD

`.github/workflows/cd.yml` uses the same deployment wrapper without
`-InteractiveSudo`. It therefore uses `sudo -n`; the dedicated board must have
a reviewed non-interactive sudo policy before GitHub Actions CD can pass. Never
store a password or private key in this repository.

## 8 x 8 notebook handoff

The default build publishes matching BIT/HWH/manifest files under
`build/vivado/npu_matrix_8x8/artifacts/`. Explicit `--array-size 2` builds
remain under `build/vivado/npu_matrix/`; pass that artifact directory to
`package_example.py --artifact-dir` when packaging a 2 x 2 demo.

To leave execution to the notebook operator, copy the complete generated
package to a dedicated directory under `/home/xilinx/jupyter_notebooks/`.
Keep `artifacts/`, `runtime/`, and `src/` alongside
`matrix_multiplication.ipynb`. Do not run the automated deploy-and-test
command in section 3 for this handoff.

Open `matrix_multiplication.ipynb` in the board's existing PYNQ Python kernel.
The notebook verifies artifact hashes before loading the overlay and prints
the physical limits; the default is `(8, 8, 256)`. It checks a full 8 x 8
output tile, a 9 x 9 output using four physical tiles, and a repeated job
against NumPy. Notebook execution programs the FPGA. Use a PYNQ kernel with
the board's existing MMIO/DMA permissions.
