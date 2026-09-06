## Purpose

Defines exact lowering of convolution and fully connected operators to bounded
Phase 1 matrix jobs, including safe K slicing and edge-tile behavior.

## ADDED Requirements

### Requirement: Bounded convolution lowering

The lowerer SHALL map convolution output positions to matrix M, flattened HWIO
kernel elements to K, and output channels to N. It SHALL partition M and N by
runtime-discovered physical limits and materialize only the current dense
signed INT8 input patch tile and weight tile before submitting each job through
the public Phase 1 runtime. It SHALL reject non-zero input zero points before
physical submission and SHALL fill convolution padding with signed INT8 zero.

#### Scenario: Edge tiles in all matrix dimensions
- **WHEN** convolution output positions or channels are not multiples of the physical M or N limits
- **THEN** only declared logical elements are submitted and the assembled output matches the quantized reference

### Requirement: Certified K slicing

K slicing SHALL be permitted only when every affected output channel carries
an exporter-generated proof that abs(bias) + 128 * sum(abs(weight)) is no
greater than 2147483647. Each K slice SHALL be no larger than the
runtime-discovered physical K limit, slices SHALL be processed in increasing K
order, partial signed INT32 results SHALL be combined in signed INT64, and the
certified final sum SHALL be representable as signed INT32 before bias and
requantization. Missing or invalid proof SHALL fail before physical submission.

#### Scenario: Unsafe accumulator bound
- **WHEN** an output channel's worst-case bound exceeds signed INT32
- **THEN** export fails and no executable package is emitted

#### Scenario: Multi-slice convolution
- **WHEN** flattened convolution K exceeds the physical maximum and all channel proofs are valid
- **THEN** ordered K-slice execution produces the same result as the bit-accurate unsliced reference

### Requirement: Fully connected lowering

The lowerer SHALL map a fully connected input to M=1, input features to K, and
output features to N, using the same M/N edge tiling and certified K-slicing
rules as convolution.

#### Scenario: Fully connected K exceeds the physical maximum
- **WHEN** a certified fully connected layer has more input features than physical MAX_K
- **THEN** it executes as ordered bounded K slices and matches the quantized reference

### Requirement: One finite execution deadline

All physical jobs and host operators for one model invocation SHALL share one
positive finite monotonic deadline. The runtime SHALL pass only the remaining
time to a physical matrix job and SHALL fail before starting further work when
the deadline is exhausted.

#### Scenario: Deadline expires between K slices
- **WHEN** the deadline is exhausted after one completed K slice
- **THEN** the runtime raises a timeout before submitting the next slice
