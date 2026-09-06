# Model contracts

`src/model/` is the production-visible source of truth for model data
structures and bit-accurate integer behavior. Exporters, runtimes, and tests
import the same definitions so that model conversion and execution do not
silently implement different numeric rules.

This directory contains source code and contracts. It does not contain trained
weights, datasets, exported model packages, bitstreams, or board results.

## Role in the system

```text
trained model or adapter
          |
          v
src/model contracts <--- contract tests
          |
          v
src/export package ---> src/runtime ---> NPU hardware
```

`src/model/` is framework-neutral. A future PyTorch or ONNX adapter may produce
these records, but framework-specific objects do not become part of the core
contract.

## Files

### `numeric.py`

[`src/model/numeric.py`](../../src/model/numeric.py) defines the shared integer
arithmetic contract:

- signed INT8 and INT32 ranges;
- INT8 and INT32 saturation;
- signed INT8 multiply with saturating INT32 accumulation;
- nearest rounding with exact ties away from zero;
- Q1.31 requantization to signed INT8; and
- deterministic dense INT8 matrix multiplication used as a reference.

The order of arithmetic operations is part of the contract. Changing when an
accumulator saturates or how a tie rounds may change hardware-visible results.

### `resnet.py`

[`src/model/resnet.py`](../../src/model/resnet.py) defines the immutable,
framework-neutral graph representation:

- `Quantization`;
- activation `TensorSpec` records;
- immutable weight and bias `ConstantTensor` records;
- commands for convolution, residual add, ReLU, max pooling, global average
  pooling, flatten, and fully connected layers; and
- the topologically ordered `QuantizedGraph`.

Constructing a graph validates identifiers, tensor layouts, batch size,
shapes, quantization compatibility, producer order, references, operator
parameters, per-channel requantization metadata, and bias types. Invalid or
unsupported graphs fail before export or hardware execution.

### `operators.py`

[`src/model/operators.py`](../../src/model/operators.py) implements the
bit-accurate Python reference for every supported command. These functions are
the executable meaning of the graph contract and provide expected results for
exporter, runtime, and hardware verification.

They are reference implementations, not an optimized inference engine.

### `__init__.py`

[`src/model/__init__.py`](../../src/model/__init__.py) is the supported public
import surface. Consumers should import model records and numeric functions
from `src.model` when possible instead of depending on private helpers.

## Supported Phase 2A model subset

Phase 2A accepts batch-one signed-INT8 activations. Convolution and pooling use
NHWC tensors; fully connected layers use NC tensors. Convolution weights use
HWIO layout, fully connected weights use IO layout, and bias tensors are signed
INT32.

Inputs to convolution and fully connected operators use symmetric INT8
quantization and therefore require `zero_point=0`. Non-zero input zero points
are rejected during graph validation, by the reference operators, and by
runtime lowering before any hardware job is submitted. Convolution padding
represents real zero with the signed INT8 value `0`. Output tensors may still
use a non-zero zero point as declared by their requantization contract.

Convolution is limited to `groups=1` and `dilation=(1, 1)`. Batch normalization
must be folded into a preceding convolution or fully connected operation.
Residual inputs must have identical shape, layout, and quantization.

The normative requirements and rejection scenarios are in the
[`quantized-resnet-operators` OpenSpec](../../openspec/changes/implement-phase2a-resnet-enablement/specs/quantized-resnet-operators/spec.md).

## Relationship to `src/test/model/`

`src/test/model/` predates the production model package and remains the Phase 0
hardware-verification reference. Shared numeric behavior is promoted into
`src/model/`; compatibility imports keep existing tests working during the
migration. Production code must not import from `src/test/`.

## What does not belong here

Do not place these artifacts under `src/model/`:

- trained `.pth`, `.onnx`, `.npz`, or similar weight files;
- generated model manifests or packed payloads;
- ImageNet or other datasets;
- Vivado projects, bitstreams, or hardware handoff files;
- benchmark output, acceptance evidence, or runtime logs; or
- framework-specific training pipelines.

Generated model packages belong outside Git. Conversion belongs in
`src/export/`, board execution belongs in `src/runtime/`, and verification
belongs in `src/test/`.

## Review rules

A change to `src/model/` is a contract change when it affects a public record,
accepted graph, tensor layout, arithmetic order, saturation point, rounding
rule, quantization rule, or operator result. Such a change should update the
corresponding OpenSpec requirement and focused tests in the same pull request.

At minimum, review should confirm:

1. exporter, runtime, and reference behavior still agree;
2. signed endpoints, overflow, saturation, and rounding boundaries are tested;
3. unsupported inputs fail before package output or physical runtime calls;
4. no production module imports from `src/test/`; and
5. no generated model or board artifact is committed.
