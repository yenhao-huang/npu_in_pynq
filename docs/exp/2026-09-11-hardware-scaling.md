# Hardware scaling investigation (#59)

Baseline: `c1c643e` on `dev`. Target: `xc7z020clg400-1` (PYNQ-Z1).
Tools: local Vivado 2026.1, Icarus Verilog 13.0, Verilator 5.050, Python 3.12.
Source change: `openspec/changes/optimize-larger-hardware/`.

## Findings and scope

The reproducible hardware bottleneck beyond 8x8 is operand-memory inference.
At 16x16, Vivado reports that each flat operand memory has too many read ports
and dissolves each into **32,768 register bits** before optimization. At 4x4
and 8x8 it uses distributed RAM, not BRAM. This is a structural implementation
cliff, separate from the per-transaction cycle count.

The controller also serializes data movement and computation. With K=256,
the 4/8/16 baseline tiles spend 2,048/4,096/8,192 cycles loading A and B;
for 16x16 this is 93.76% of the 8,737-cycle transaction. A single 8-bit input
stream cannot keep a growing array busy. The baseline's effective PE occupancy
at K=256 falls from 10.99% to 5.77% to 2.93% as the square array doubles.

The data does **not** demonstrate an abrupt execution-cycle jump specifically
at 8x8. Raw tile latency grows while each larger tile performs more work.
No physical 16x16 board run or user-provided failing workload was available;
these results identify and address RTL scaling limits, not an unmeasured
end-to-end ResNet regression. Production overlay selection still supports 2/8;
16 is evaluated as an experimental controller. Larger physical configurations
are not advertised by the existing overlay, and were not benchmarked.

## Implemented changes

1. Bank A by row and B by column with one synchronous read and one write per
   bank. Preserve the original boundary-register latency. Data memory/output
   registers have no reset; reset validity masks stale or uninitialized values.
2. Clear PEs during A loading. During B loading, advance compute only when the
   next full B row is resident. Hold the boundary registers and all PEs together
   during input stalls. Finish the original pipeline drain after B completes.
   This saves K-1 cycles for unstalled jobs without changing stream frames.
3. Add a configurable array DSP budget, default 192, with identical registered
   fabric multipliers for remaining PEs. A standalone PE and arrays through 8x8
   retain the existing DSP implementation. At 16x16, this explicitly leaves 28
   DSPs available for integration and uses 64 fabric multipliers. Vivado already
   spills excess requests automatically at its 220-DSP device ceiling; the
   budget controls the allocation rather than fixing a synthesis impossibility.

The register map, INT8 input/INT32 output format, saturating arithmetic, stream
length validation, output ordering/TLAST, backpressure and timeout behavior are
preserved. The cycle counter intentionally reports shorter execution.

## Cycle and throughput comparison

Full M=N=array-size tiles, no artificial stalls. Throughput counts **MACs**,
not two operations per MAC. Times and GMAC/s below assume 100 MHz; they are RTL
measurements converted using that clock, not physical board measurements.

| Array | K | Baseline cycles | Optimized cycles | Latency reduction | Baseline GMAC/s | Optimized GMAC/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4x4 | 1 | 34 | 34 | 0% | 0.0471 | 0.0471 |
| 4x4 | 32 | 313 | 282 | 9.90% | 0.1636 | 0.1816 |
| 4x4 | 256 | 2,329 | 2,074 | 10.95% | 0.1759 | 0.1975 |
| 8x8 | 1 | 98 | 98 | 0% | 0.0653 | 0.0653 |
| 8x8 | 32 | 625 | 594 | 4.96% | 0.3277 | 0.3448 |
| 8x8 | 256 | 4,433 | 4,178 | 5.75% | 0.3696 | 0.3921 |
| 16x16 | 1 | 322 | 322 | 0% | 0.0795 | 0.0795 |
| 16x16 | 32 | 1,345 | 1,314 | 2.30% | 0.6091 | 0.6234 |
| 16x16 | 256 | 8,737 | 8,482 | 2.92% | 0.7501 | 0.7726 |

At K=256 the corresponding baseline -> optimized latency is 23.29 -> 20.74 us,
44.33 -> 41.78 us, and 87.37 -> 84.82 us. Baseline no-stall latency is
`M*K + K*N + M*N + K + M + N + 1`; optimized latency subtracts `K-1`.
The benchmark asserts these counts as well as checking every output element.

For **equal total work**, a 16x256 by 256x16 matrix multiplication contains
65,536 MACs and requires 16, 4, or 1 full tiles. Summing the measured no-stall
tile costs gives 37,264 -> 33,184 cycles (4x4), 17,732 -> 16,712 (8x8), and
8,737 -> 8,482 (16x16). These are calculated controller-only totals; they exclude
host dispatch, DMA setup and inter-job idle time. They show that larger arrays
still improve equal-work cycle throughput, although far below ideal PE scaling.

## Bandwidth and remaining limits

The unchanged stream widths imply upper bounds of 100 MB/s ingress and
400 MB/s egress at 100 MHz, assuming continuous handshakes. A K=256 16x16 job
transfers 8,192 input bytes and 1,024 output bytes. Input alone costs 81.92 us.
After overlap, input loading is 96.58% of the 8,482-cycle job, so extra compute
parallelism or a deeper MAC pipeline alone cannot produce a large cycle gain.
K=1 instead spends 256/322 cycles serializing outputs. These are protocol
ceilings and phase counts, not measured DDR or board DMA bandwidth.

Wider/packed input streams, operand reuse across tiles, and double buffering
could address the remaining transfer limit. They require coordinated DMA,
runtime and buffering changes; the current changes preserve that interface.
The existing product pipeline remains intact. Explicit memory banking addresses
data movement/resource growth, and routed critical paths evaluate whether
additional internal pipelining is justified.

## Local synthesis and routing

All comparisons use the same standalone controller top, MAX_K=256 and 10 ns
clock. This excludes AXI-Lite registers, PS, DMA and their interconnect. External
ports have no OOC placement or I/O delay constraints, so the reported timing
only characterizes internal synchronous paths and is not full-overlay signoff.

Synthesis utilization:

| Array | Variant | LUT | FF | BRAM tiles | DSP |
| --- | --- | ---: | ---: | ---: | ---: |
| 4x4 | Baseline | 3,989 | 1,334 | 0 | 16 |
| 4x4 | Optimized | 2,108 | 1,257 | 4 | 16 |
| 8x8 | Baseline | 14,515 | 4,631 | 0 | 64 |
| 8x8 | Optimized | 6,831 | 4,413 | 8 | 64 |
| 16x16 | Baseline | **175,557** | 83,076 | 0 | 220 |
| 16x16 | Optimized | **29,068** | 17,201 | 16 | 192 |

The device has 53,200 LUTs, 106,400 FFs, 140 BRAM tiles and 220 DSPs. The
baseline 16x16 LUT demand is 330.0% of capacity; it cannot fit. Its simulation
throughput is therefore not physically realizable on this FPGA. That profiling
job was intentionally stopped after synthesis utilization, before implementation.
The profiling script now detects this LUT limit before attempting placement.
Optimized 16x16 reduces LUTs by 83.44% and FFs by 79.29%, replacing the flat
operand structures with 16 BRAM tiles. It uses 54.64% of LUTs and 87.27% of DSPs.

Routed internal timing at 100 MHz (10 ns):

| Array | Variant | Routed LUT | FF | BRAM | DSP | Setup WNS (ns) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4x4 | Baseline | 3,960 | 1,334 | 0 | 16 | +0.468 |
| 4x4 | Optimized | 2,088 | 1,257 | 4 | 16 | +0.473 |
| 8x8 | Baseline | 14,440 | 4,631 | 0 | 64 | +0.071 |
| 8x8 | Optimized | 6,798 | 4,413 | 8 | 64 | +0.493 |
| 16x16 | Baseline | Not placeable | — | — | — | N/A |
| 16x16 | Optimized | 29,016 | 17,201 | 16 | 192 | +0.444 |

The baseline 8x8 critical path runs from the load index to a distributed B RAM
write address: 9.903 ns data delay, 95.4% routing. After banking, its critical
path moves to dimension/error control (9.168 ns, 12 logic levels). The optimized
4x4 critical path is the 64-bit cycle/timeout control chain (9.238 ns, 20 logic
levels). These are report-derived critical paths, not assumptions about MACs.
At 16x16 the final critical path is a BRAM read through the first-column DSP
product into its pipeline register (9.547 ns data delay). All three optimized
sizes meet the 100 MHz internal setup, hold and pulse-width constraints. Routed
16x16 hold slack is +0.124 ns. Raw evidence is under `build/issue59/`.

Vivado comparisons used frozen baseline and optimized RTL snapshots. The only
later functional-source edit was an elaboration-time DSP_STYLE legal-value
assertion; the synthesized data path is unchanged. The final PE source also
passes `check_pe_dsp.tcl` with exactly one DSP, and the final-source benchmark
records its SHA-256 values. Baseline and optimized jobs ran concurrently during
part of this study, so synthesis wall time is not presented as a controlled
performance comparison.

## Experiment record

| Hypothesis / change | Evidence | Decision |
| --- | --- | --- |
| Profile serial baseline | 16x16 K=256: 8,192 load + 289 compute/transition + 256 output cycles | Target transfer overlap; distinguish equal-work scaling |
| Overlap B and compute | Same reference outputs; 255 cycles saved at K=256 for all sizes | Keep |
| Explicit synchronous banks | 16x16 synthesis uses 16 BRAM tiles; flat baseline emits multi-port RAM warnings | Keep |
| Bound DSP allocation | Banked-only synthesis: 27,550 LUT / 17,201 FF / 16 BRAM / 220 DSP; budgeted: 29,068 LUT / 17,201 FF / 16 BRAM / 192 DSP | Trade 1,518 LUTs for 28 DSPs of integration headroom |

The banked-only ablation was intentionally stopped after its synthesis reports
were saved to relieve local memory pressure; no routing result is claimed for it.

## Reproduction and correctness

Create an isolated baseline checkout at `c1c643e` (or extract its RTL under a
scratch root). Run the **new** benchmark against both source roots:

```text
python src/test/benchmark_scaling.py --baseline --rtl-root <baseline-root> --output build/issue59/baseline/benchmark
python src/test/benchmark_scaling.py --output build/issue59/optimized/benchmark
make -B -C src/test lint sim
python -m unittest discover -s src/test/tests -v
python -m unittest discover -s examples/matrix-multiplication/tests -v
```

The runner saves exact commands, source/testbench SHA-256 values, per-stage logs,
phase cycles, throughput and PE utilization in `metrics.json`. It rejects
nonzero exit status, error messages, missing PASS, and incorrect no-stall latency.
Tests cover signed INT8 extremes, K=1/32/256, partial tiles, sequential shape
changes, input gaps, output backpressure/stability, reset during overlapped B
loading, malformed B TLAST after partial compute, and successful recovery.
Existing regressions separately cover PE saturation and protocol errors/timeouts.
Final local lint and the complete discovered simulation suite pass; 129 core
Python tests and 17 matrix-example tests pass. The dedicated runner passes eight
matrix cases plus reset/error recovery at each of 4/8/16 for both source roots.

For each source root and SIZE in 4, 8, 16:

```text
vivado -mode batch -source src/hw/vivado_tcl/npu_matrix/profile_scaling.tcl -tclargs <source-root> <distinct-output-directory> <SIZE>
```

Run these local synthesis jobs sequentially on memory-limited hosts. Generated
Vivado projects, reports, checkpoints and logs stay ignored. No bitstream or
board deployment was produced for this investigation.
