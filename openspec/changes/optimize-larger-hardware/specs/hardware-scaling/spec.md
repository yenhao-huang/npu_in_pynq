## ADDED Requirements

### Requirement: Banked operand storage
The matrix controller SHALL store A in independent row banks and B in
independent column banks with synchronous reads and validity-masked outputs.

#### Scenario: Sequential partial jobs
- **WHEN** a completed or aborted job is followed by a job with different M/N/K
- **THEN** every valid operand shall come from the new job and every output
  shall match signed INT8 multiplication with the existing INT32 saturation rule.

### Requirement: Safe overlap of loading and compute
The controller SHALL advance the array during B loading only when the next
required B row is completely resident, holding all pipeline stages together
when no new compute step is available.

#### Scenario: Input gaps and output backpressure
- **WHEN** the input stream contains gaps or the output consumer stalls
- **THEN** no product shall be duplicated or lost, and valid output data/TLAST
  shall remain stable until accepted.

#### Scenario: No-stall latency
- **WHEN** a valid M/N/K transaction runs without input or output stalls
- **THEN** it shall finish in M*K + K*N + M*N + M + N + 2 busy cycles.

### Requirement: Explicit DSP allocation
The systolic array SHALL support a DSP budget, defaulting to 192, and implement
excess PEs with fabric multipliers that retain the existing pipeline timing.

#### Scenario: A 16x16 controller
- **WHEN** the controller is synthesized for xc7z020clg400-1 with MAX_K=256
- **THEN** the default array shall use 192 DSPs and keep its remaining
  multipliers in fabric without changing arithmetic or stream-visible results.

### Requirement: Reproducible scaling evidence
The project SHALL provide self-checking 4x4, 8x8 and 16x16 benchmarks with
baseline and optimized cycle, throughput, utilization and timing evidence.

#### Scenario: Resource or timing limits
- **WHEN** a configuration exceeds device resources or fails timing
- **THEN** the evidence shall identify the failure explicitly and shall not
  represent simulation throughput as realizable hardware performance.
