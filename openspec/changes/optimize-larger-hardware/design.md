# Design and validation

## Hypotheses

1. The 8-bit input stream dominates large-K transaction time. Overlapping B
   loading and compute removes K-1 cycles without changing external traffic.
2. Independent synchronous A-row and B-column banks remove the multi-read-port
   inference cliff and replace mux/register structures with block RAM.
3. A 192-DSP budget permits 16x16 inference using 64 fabric multipliers while
   retaining identical arithmetic and the existing one-stage product pipeline.

## Scheduling invariant

During LOAD_A, clear the systolic array and invalidate boundary inputs. During
LOAD_B, load_outer is the count of fully received B rows. Advance compute_step
only when compute_step < load_outer. For a valid column c, the read reduction
index compute_step-c is therefore strictly less than the completed-row count.
All A rows are already resident. No valid read collides with a B write. Hold
the boundary data/valid registers and every PE together when no step is ready.
After B completes, preserve the in-flight pipeline through the one-cycle
transition state, then finish the original drain horizon K+M+N-1.

## Memory and numeric behavior

Each bank has one write port and one clocked read port, with no reset on memory
or data outputs. Validity is reset and masks old/uninitialized bank data. Every
valid location is written before use; repeated and partial jobs need no memory
clear. Runtime signed INT8 and per-add saturating INT32 behavior is unchanged.
DSP_STYLE selects synthesis implementation only; excess PEs retain all pipeline
registers and saturation logic. The default PE remains one DSP for standalone
checks and arrays up to 8x8 remain fully DSP based.

## Evaluation boundaries

Simulation measures cycle counts, phase occupancy, output reference agreement,
TLAST and backpressure stability. Run K=1/32/256 full tiles and changing partial
tiles with stalls at 4/8/16. Compare both per-tile latency and equal total work;
larger tiles perform more MACs and raw per-tile time is not sufficient evidence
of a scaling regression. Derive bandwidth ceilings from the actual stream widths.

Vivado uses xc7z020clg400-1 and the same 10 ns clock for baseline and optimized
out-of-context controllers. Report synthesis and routed utilization, WNS and
critical paths separately. Unfit configurations and timing failures remain
explicit. External I/O has no OOC placement/delay constraints; routed internal
timing is not full-overlay or physical-board proof. Do not run or modify board
deployments for this experiment.
