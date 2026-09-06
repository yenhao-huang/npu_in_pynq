# NPU Matrix Overlay

The PYNQ-Z1 matrix overlay supports source-controlled 2x2 and 8x8 systolic
array configurations. The 2x2 target remains the default so existing build and
runtime workflows are unchanged.

## Select a configuration

Run Vivado from the repository root. Select the 8x8 target with:

```text
vivado -mode batch -source src/hw/vivado_tcl/npu_matrix/build_overlay.tcl \
  -tclargs --array-size 8
```

Use `--array-size 2` or omit the option for the default target. Only `2` and
`8` are accepted. The default build is written below
`build/vivado/npu_matrix/`; the 8x8 build uses
`build/vivado/npu_matrix_8x8/` so the two configurations cannot overwrite one
another.

Add `--elaborate-only` for block-design validation. Add `--allow-dirty` only
for an exploratory local implementation whose artifacts will not be
published.

## Verify the target

The repository simulation suite includes full 8x8 and masked 7x5 jobs with
deterministic pseudo-random signed INT8 operands. Run all open-source gates
with:

```text
make -C src/test lint sim
```

For a release candidate, run the 8x8 Vivado command without
`--elaborate-only`. The build must finish synthesis, implementation, routing,
DRC, and setup timing with nonnegative WNS before BIT/HWH artifacts are
published. The build evidence records the selected row and column counts.

Finally, deploy the matching BIT/HWH pair to a PYNQ-Z1 and run the matrix
example's board smoke test. Open-source simulation does not substitute for
Vivado timing/resource evidence or physical-board validation.

## 8x8 implementation evidence

Vivado 2026.1 implemented and routed the 8x8 target for
`xc7z020clg400-1` at 100 MHz on 2026-09-05. The routed timing report met all
specified constraints with setup WNS 0.079 ns and hold WHS 0.017 ns. The
implementation used 17,302 slice LUTs (32.52%), 8,528 slice registers (8.02%),
2 block RAM tiles (1.43%), and 64 DSPs (29.09%). There were no unrouted nets,
routing errors, setup failures, hold failures, or DRC errors.

These figures establish build feasibility for the PYNQ-Z1 target. A physical
matrix smoke test is still required for each release artifact because the
repository does not treat implementation evidence as board evidence.
