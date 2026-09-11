# Optimize larger hardware configurations

Tracking issue: #59. Baseline: dev commit c1c643e.

The controller serializes A loading, B loading, and compute. Flat operand
memories request one read port per array edge and cease inferring RAM at 16x16.
One requested DSP per PE also exceeds the Zynq-7020's 220 DSPs at that size;
Vivado automatically spills excess multipliers into fabric.

Preserve the INT8 operands, saturating INT32 accumulators, register map, stream
frames, timeout/error semantics, and existing deployable overlay sizes. Replace
flat operands with synchronous banks, overlap B loading with safe compute steps,
and bound DSP inference for larger arrays. Add reproducible simulation and local
Vivado profiling at 4x4, 8x8, and 16x16. A 16x16 controller experiment is not a
claim of a released 16x16 PS/DMA overlay or physical board acceptance.
