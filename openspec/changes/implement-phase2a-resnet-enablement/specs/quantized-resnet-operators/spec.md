## Purpose

Defines the tensor, quantization, and bit-accurate behavior of the operator
subset accepted for quantized batch-one ResNet-18 graphs.

## ADDED Requirements

### Requirement: Accepted graph and tensor contract

The system SHALL accept only acyclic batch-one graphs whose activations are
dense NHWC signed INT8 tensors and whose convolution weights are dense HWIO
signed INT8 tensors. Batch normalization SHALL be folded into a preceding
convolution or fully connected operator. Supported convolution parameters SHALL
be groups=1, dilation=1, positive stride, and explicit non-negative padding.
Unsupported operators, ranks, layouts, groups, dilation, or inconsistent tensor
shapes SHALL fail before package output or execution.

Inputs to convolution and fully connected operators SHALL use symmetric signed
INT8 quantization with zero point zero. Graph validation, export, reference
operators, and runtime lowering SHALL reject any non-zero input zero point
before execution.

#### Scenario: Unsupported grouped convolution
- **WHEN** a graph contains a convolution whose groups value is not one
- **THEN** export fails with an error identifying the operator and unsupported parameter

### Requirement: Quantized convolution

Convolution SHALL visit kernel height, kernel width, and input channel in
increasing order for every NHWC output position and HWIO output channel. Each
signed INT8 product SHALL use the Phase 0 ordered saturating INT32 MAC contract;
an optional signed INT32 bias SHALL then be saturating-added, and the result
    SHALL use the Phase 0 Q1.31 requantization contract. Padding elements SHALL
equal signed INT8 zero so that they represent real zero under the required
symmetric input contract.

#### Scenario: Strided padded convolution
- **WHEN** a batch-one 7x7 convolution uses stride two and explicit padding three
- **THEN** every output equals the bit-accurate reference using signed INT8 zero padding

#### Scenario: Asymmetric matrix input is rejected
- **WHEN** a convolution or fully connected input declares a non-zero zero point
- **THEN** graph validation fails before package output or execution

### Requirement: Residual add and activation

Residual add SHALL require both input tensors and the output tensor to have
identical shape, signed INT8 dtype, multiplier, shift, and zero point. It SHALL
compute each output as the signed INT8 saturation of
(lhs - zero_point) + (rhs - zero_point) + zero_point. ReLU SHALL preserve the
tensor quantization and clamp each signed INT8 value below its zero point to the
zero point.

#### Scenario: Incompatible residual quantization
- **WHEN** residual inputs do not share the required quantization parameters
- **THEN** export fails explicitly rather than silently rescaling either branch

### Requirement: Pooling and flatten

Max pooling SHALL support positive windows and strides with explicit
non-negative padding, treating padded values as -128, and SHALL preserve input
quantization. Global average pooling SHALL average each NHWC channel over all
spatial elements, round an exact tie away from zero, and saturate to signed
INT8 with the same quantization. Flatten SHALL convert a batch-one 1x1xC tensor
to a dense 1xC tensor without changing element order or quantization.

#### Scenario: Global average rounding
- **WHEN** a channel mean is exactly halfway between two signed integers
- **THEN** global average pooling selects the integer farther from zero

### Requirement: Fully connected output

Fully connected execution SHALL use ordered signed INT8 by signed INT8
saturating INT32 accumulation, optional saturating INT32 bias addition, and the
Phase 0 Q1.31 requantization contract for each output element.

#### Scenario: Fully connected signed endpoints
- **WHEN** inputs and weights contain -128 and 127 endpoint values
- **THEN** the output equals the bit-accurate quantized reference
