"""Immutable, framework-neutral contracts for quantized ResNet graphs."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import TypeAlias

from .numeric import INT8_MAX, INT8_MIN, INT32_MAX, INT32_MIN


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class GraphValidationError(ValueError):
    """A graph is unsupported, inconsistent, or not topologically ordered."""


def _identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a stable non-empty identifier")
    return value


def _integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _bounded(name: str, value: int, minimum: int, maximum: int) -> int:
    value = _integer(name, value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def _positive_tuple(name: str, values: tuple[int, ...], length: int) -> tuple[int, ...]:
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integers") from error
    if len(normalized) != length:
        raise ValueError(f"{name} must contain {length} values")
    for index, value in enumerate(normalized):
        _bounded(f"{name}[{index}]", value, 1, 0x7FFFFFFF)
    return normalized


def _padding(values: tuple[int, ...]) -> tuple[int, int, int, int]:
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise TypeError("padding must be an iterable of four integers") from error
    if len(normalized) != 4:
        raise ValueError("padding must contain top, bottom, left, and right")
    for index, value in enumerate(normalized):
        _bounded(f"padding[{index}]", value, 0, 0x7FFFFFFF)
    return normalized


@dataclass(frozen=True)
class Quantization:
    """Exact integer tensor-quantization identity used for compatibility."""

    multiplier_q31: int
    shift: int
    zero_point: int

    def __post_init__(self) -> None:
        _bounded("multiplier_q31", self.multiplier_q31, INT32_MIN, INT32_MAX)
        _bounded("shift", self.shift, 0, 31)
        _bounded("zero_point", self.zero_point, INT8_MIN, INT8_MAX)


@dataclass(frozen=True)
class TensorSpec:
    """One batch-one signed INT8 activation tensor."""

    name: str
    shape: tuple[int, ...]
    layout: str
    quantization: Quantization

    def __post_init__(self) -> None:
        _identifier("tensor name", self.name)
        shape = tuple(self.shape)
        object.__setattr__(self, "shape", shape)
        if self.layout == "NHWC":
            if len(shape) != 4:
                raise ValueError("NHWC tensors must have rank four")
        elif self.layout == "NC":
            if len(shape) != 2:
                raise ValueError("NC tensors must have rank two")
        else:
            raise ValueError(f"unsupported activation layout {self.layout!r}")
        if shape[0] != 1:
            raise ValueError("only batch-one tensors are supported")
        for index, dimension in enumerate(shape):
            _bounded(f"shape[{index}]", dimension, 1, 0x7FFFFFFF)
        if not isinstance(self.quantization, Quantization):
            raise TypeError("quantization must be a Quantization record")


@dataclass(frozen=True)
class ConstantTensor:
    """One immutable dense weight or bias tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    layout: str
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        _identifier("constant name", self.name)
        shape = tuple(self.shape)
        values = tuple(self.values)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "values", values)
        expected_layouts = {
            ("int8", "HWIO"): 4,
            ("int8", "IO"): 2,
            ("int32", "BIAS"): 1,
        }
        expected_rank = expected_layouts.get((self.dtype, self.layout))
        if expected_rank is None:
            raise ValueError(
                f"unsupported constant dtype/layout {self.dtype}/{self.layout}"
            )
        if len(shape) != expected_rank:
            raise ValueError(
                f"{self.layout} constants must have rank {expected_rank}"
            )
        for index, dimension in enumerate(shape):
            _bounded(f"shape[{index}]", dimension, 1, 0x7FFFFFFF)
        if math.prod(shape) != len(values):
            raise ValueError(
                f"constant {self.name!r} has {len(values)} values for shape {shape}"
            )
        minimum, maximum = (
            (INT8_MIN, INT8_MAX) if self.dtype == "int8"
            else (INT32_MIN, INT32_MAX)
        )
        for index, value in enumerate(values):
            _bounded(f"values[{index}]", value, minimum, maximum)


@dataclass(frozen=True)
class Conv2D:
    command_id: str
    input_id: str
    weight_id: str
    output_id: str
    multipliers_q31: tuple[int, ...]
    shifts: tuple[int, ...]
    bias_id: str | None = None
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    dilation: tuple[int, int] = (1, 1)
    groups: int = 1

    def __post_init__(self) -> None:
        for name in ("command_id", "input_id", "weight_id", "output_id"):
            _identifier(name, getattr(self, name))
        if self.bias_id is not None:
            _identifier("bias_id", self.bias_id)
        object.__setattr__(self, "stride", _positive_tuple("stride", self.stride, 2))
        object.__setattr__(self, "padding", _padding(self.padding))
        dilation = _positive_tuple("dilation", self.dilation, 2)
        if dilation != (1, 1):
            raise ValueError("only dilation=(1, 1) is supported")
        object.__setattr__(self, "dilation", dilation)
        if _integer("groups", self.groups) != 1:
            raise ValueError("only groups=1 convolution is supported")
        multipliers = tuple(self.multipliers_q31)
        shifts = tuple(self.shifts)
        if not multipliers or len(multipliers) != len(shifts):
            raise ValueError("requantization multipliers and shifts must align")
        for index, value in enumerate(multipliers):
            _bounded(f"multipliers_q31[{index}]", value, INT32_MIN, INT32_MAX)
        for index, value in enumerate(shifts):
            _bounded(f"shifts[{index}]", value, 0, 31)
        object.__setattr__(self, "multipliers_q31", multipliers)
        object.__setattr__(self, "shifts", shifts)


@dataclass(frozen=True)
class ResidualAdd:
    command_id: str
    lhs_id: str
    rhs_id: str
    output_id: str

    def __post_init__(self) -> None:
        for name in ("command_id", "lhs_id", "rhs_id", "output_id"):
            _identifier(name, getattr(self, name))


@dataclass(frozen=True)
class Relu:
    command_id: str
    input_id: str
    output_id: str

    def __post_init__(self) -> None:
        for name in ("command_id", "input_id", "output_id"):
            _identifier(name, getattr(self, name))


@dataclass(frozen=True)
class MaxPool:
    command_id: str
    input_id: str
    output_id: str
    window: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)

    def __post_init__(self) -> None:
        for name in ("command_id", "input_id", "output_id"):
            _identifier(name, getattr(self, name))
        object.__setattr__(self, "window", _positive_tuple("window", self.window, 2))
        object.__setattr__(self, "stride", _positive_tuple("stride", self.stride, 2))
        object.__setattr__(self, "padding", _padding(self.padding))


@dataclass(frozen=True)
class GlobalAveragePool:
    command_id: str
    input_id: str
    output_id: str

    def __post_init__(self) -> None:
        for name in ("command_id", "input_id", "output_id"):
            _identifier(name, getattr(self, name))


@dataclass(frozen=True)
class Flatten:
    command_id: str
    input_id: str
    output_id: str

    def __post_init__(self) -> None:
        for name in ("command_id", "input_id", "output_id"):
            _identifier(name, getattr(self, name))


@dataclass(frozen=True)
class FullyConnected:
    command_id: str
    input_id: str
    weight_id: str
    output_id: str
    multipliers_q31: tuple[int, ...]
    shifts: tuple[int, ...]
    bias_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("command_id", "input_id", "weight_id", "output_id"):
            _identifier(name, getattr(self, name))
        if self.bias_id is not None:
            _identifier("bias_id", self.bias_id)
        multipliers = tuple(self.multipliers_q31)
        shifts = tuple(self.shifts)
        if not multipliers or len(multipliers) != len(shifts):
            raise ValueError("requantization multipliers and shifts must align")
        for index, value in enumerate(multipliers):
            _bounded(f"multipliers_q31[{index}]", value, INT32_MIN, INT32_MAX)
        for index, value in enumerate(shifts):
            _bounded(f"shifts[{index}]", value, 0, 31)
        object.__setattr__(self, "multipliers_q31", multipliers)
        object.__setattr__(self, "shifts", shifts)


Command: TypeAlias = (
    Conv2D
    | ResidualAdd
    | Relu
    | MaxPool
    | GlobalAveragePool
    | Flatten
    | FullyConnected
)
_COMMAND_TYPES = (
    Conv2D,
    ResidualAdd,
    Relu,
    MaxPool,
    GlobalAveragePool,
    Flatten,
    FullyConnected,
)


@dataclass(frozen=True)
class QuantizedGraph:
    """A validated, topologically ordered Phase 2A command graph."""

    tensors: tuple[TensorSpec, ...]
    constants: tuple[ConstantTensor, ...]
    commands: tuple[Command, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("tensors", "constants", "commands", "inputs", "outputs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        self._validate()

    def _validate(self) -> None:
        tensor_map = self._unique_map("tensor", self.tensors)
        constant_map = self._unique_map("constant", self.constants)
        command_map = self._unique_map("command", self.commands, "command_id")
        all_ids = list(tensor_map) + list(constant_map) + list(command_map)
        if len(all_ids) != len(set(all_ids)):
            raise GraphValidationError("duplicate identifier across graph records")
        if not self.inputs or not self.outputs:
            raise GraphValidationError("graph inputs and outputs must not be empty")
        if len(self.inputs) != len(set(self.inputs)):
            raise GraphValidationError("duplicate graph input")
        if len(self.outputs) != len(set(self.outputs)):
            raise GraphValidationError("duplicate graph output")
        for name in self.inputs:
            if name not in tensor_map:
                raise GraphValidationError(f"input tensor {name!r} is not declared")
        available = set(self.inputs) | set(constant_map)
        produced = set(self.inputs)
        for command in self.commands:
            if not isinstance(command, _COMMAND_TYPES):
                raise GraphValidationError(
                    f"unsupported command record {type(command).__name__}"
                )
            output_id = command.output_id
            if output_id not in tensor_map:
                raise GraphValidationError(
                    f"output tensor {output_id!r} is not declared"
                )
            if output_id in produced:
                raise GraphValidationError(
                    f"tensor {output_id!r} has multiple definitions"
                )
            self._require_references(command, available)
            self._validate_command(command, tensor_map, constant_map)
            available.add(output_id)
            produced.add(output_id)
        missing_definitions = set(tensor_map) - produced
        if missing_definitions:
            missing = ", ".join(sorted(missing_definitions))
            raise GraphValidationError(f"tensor definitions are missing: {missing}")
        for name in self.outputs:
            if name not in produced:
                raise GraphValidationError(f"output tensor {name!r} is not available")

    @staticmethod
    def _unique_map(kind, records, attribute="name"):
        result = {}
        for record in records:
            if kind == "command" and not isinstance(record, _COMMAND_TYPES):
                continue
            expected = (
                _COMMAND_TYPES if kind == "command"
                else (TensorSpec,) if kind == "tensor"
                else (ConstantTensor,)
            )
            if not isinstance(record, expected):
                raise GraphValidationError(
                    f"unsupported {kind} record {type(record).__name__}"
                )
            name = getattr(record, attribute)
            if name in result:
                raise GraphValidationError(f"duplicate {kind} identifier {name!r}")
            result[name] = record
        return result

    @staticmethod
    def _require_references(command: Command, available: set[str]) -> None:
        if isinstance(command, ResidualAdd):
            references = (command.lhs_id, command.rhs_id)
        elif isinstance(command, (Conv2D, FullyConnected)):
            references = (command.input_id, command.weight_id)
            if command.bias_id is not None:
                references += (command.bias_id,)
        else:
            references = (command.input_id,)
        for reference in references:
            if reference not in available:
                raise GraphValidationError(
                    f"command {command.command_id!r} reference "
                    f"{reference!r} is not available"
                )

    @staticmethod
    def _same_tensor_contract(command_id, source, output) -> None:
        if source.shape != output.shape:
            raise GraphValidationError(
                f"command {command_id!r} output shape mismatch"
            )
        if source.layout != output.layout:
            raise GraphValidationError(
                f"command {command_id!r} output layout mismatch"
            )
        if source.quantization != output.quantization:
            raise GraphValidationError(
                f"command {command_id!r} output quantization mismatch"
            )

    def _validate_command(self, command, tensors, constants) -> None:
        output = tensors[command.output_id]
        if isinstance(command, ResidualAdd):
            tensor_references = (command.lhs_id, command.rhs_id)
            constant_references = ()
        elif isinstance(command, (Conv2D, FullyConnected)):
            tensor_references = (command.input_id,)
            constant_references = (command.weight_id,)
            if command.bias_id is not None:
                constant_references += (command.bias_id,)
        else:
            tensor_references = (command.input_id,)
            constant_references = ()
        for reference in tensor_references:
            if reference not in tensors:
                raise GraphValidationError(
                    f"command {command.command_id!r} reference "
                    f"{reference!r} is not an activation tensor"
                )
        for reference in constant_references:
            if reference not in constants:
                raise GraphValidationError(
                    f"command {command.command_id!r} reference "
                    f"{reference!r} is not a constant tensor"
                )
        if isinstance(command, Conv2D):
            source = tensors[command.input_id]
            weight = constants[command.weight_id]
            self._validate_matrix_input_quantization(command, source)
            if source.layout != "NHWC" or output.layout != "NHWC":
                raise GraphValidationError("convolution requires NHWC tensors")
            if weight.dtype != "int8" or weight.layout != "HWIO":
                raise GraphValidationError("convolution requires INT8 HWIO weights")
            _, height, width, channels = source.shape
            kernel_h, kernel_w, weight_channels, output_channels = weight.shape
            if channels != weight_channels:
                raise GraphValidationError("convolution input channel shape mismatch")
            top, bottom, left, right = command.padding
            numerator_h = height + top + bottom - kernel_h
            numerator_w = width + left + right - kernel_w
            if numerator_h < 0 or numerator_w < 0:
                raise GraphValidationError("convolution kernel exceeds padded input")
            expected = (
                1,
                numerator_h // command.stride[0] + 1,
                numerator_w // command.stride[1] + 1,
                output_channels,
            )
            if output.shape != expected:
                raise GraphValidationError(
                    f"command {command.command_id!r} output shape "
                    f"{output.shape} != {expected}"
                )
            self._validate_requantization(command, output_channels)
            self._validate_bias(command, constants, output_channels)
        elif isinstance(command, ResidualAdd):
            lhs = tensors[command.lhs_id]
            rhs = tensors[command.rhs_id]
            self._same_tensor_contract(command.command_id, lhs, rhs)
            self._same_tensor_contract(command.command_id, lhs, output)
        elif isinstance(command, Relu):
            self._same_tensor_contract(
                command.command_id, tensors[command.input_id], output
            )
        elif isinstance(command, MaxPool):
            source = tensors[command.input_id]
            if source.layout != "NHWC" or output.layout != "NHWC":
                raise GraphValidationError("max pool requires NHWC tensors")
            _, height, width, channels = source.shape
            top, bottom, left, right = command.padding
            numerator_h = height + top + bottom - command.window[0]
            numerator_w = width + left + right - command.window[1]
            if numerator_h < 0 or numerator_w < 0:
                raise GraphValidationError("pool window exceeds padded input")
            expected = (
                1,
                numerator_h // command.stride[0] + 1,
                numerator_w // command.stride[1] + 1,
                channels,
            )
            if output.shape != expected:
                raise GraphValidationError(
                    f"command {command.command_id!r} output shape mismatch"
                )
            if source.quantization != output.quantization:
                raise GraphValidationError("max pool quantization mismatch")
        elif isinstance(command, GlobalAveragePool):
            source = tensors[command.input_id]
            if source.layout != "NHWC" or output.layout != "NHWC":
                raise GraphValidationError(
                    "global average pool requires NHWC tensors"
                )
            expected = (1, 1, 1, source.shape[3])
            if output.shape != expected:
                raise GraphValidationError(
                    f"command {command.command_id!r} output shape mismatch"
                )
            if source.quantization != output.quantization:
                raise GraphValidationError(
                    "global average pool quantization mismatch"
                )
        elif isinstance(command, Flatten):
            source = tensors[command.input_id]
            if source.layout != "NHWC" or source.shape[1:3] != (1, 1):
                raise GraphValidationError("flatten requires a batch-one 1x1xC tensor")
            expected = (1, source.shape[3])
            if output.layout != "NC" or output.shape != expected:
                raise GraphValidationError(
                    f"command {command.command_id!r} output shape/layout mismatch"
                )
            if source.quantization != output.quantization:
                raise GraphValidationError("flatten quantization mismatch")
        elif isinstance(command, FullyConnected):
            source = tensors[command.input_id]
            weight = constants[command.weight_id]
            self._validate_matrix_input_quantization(command, source)
            if source.layout != "NC" or output.layout != "NC":
                raise GraphValidationError("fully connected requires NC tensors")
            if weight.dtype != "int8" or weight.layout != "IO":
                raise GraphValidationError(
                    "fully connected requires INT8 IO weights"
                )
            input_features, output_features = weight.shape
            expected = (1, output_features)
            if source.shape != (1, input_features) or output.shape != expected:
                raise GraphValidationError(
                    f"command {command.command_id!r} input/output shape mismatch"
                )
            self._validate_requantization(command, output_features)
            self._validate_bias(command, constants, output_features)
        else:
            raise GraphValidationError(
                f"unsupported command record {type(command).__name__}"
            )

    @staticmethod
    def _validate_matrix_input_quantization(command, source) -> None:
        if source.quantization.zero_point != 0:
            raise GraphValidationError(
                f"command {command.command_id!r} requires symmetric INT8 input "
                "with zero_point=0"
            )

    @staticmethod
    def _validate_requantization(command, channels) -> None:
        if (
            len(command.multipliers_q31) != channels
            or len(command.shifts) != channels
        ):
            raise GraphValidationError(
                f"command {command.command_id!r} requantization channel mismatch"
            )

    @staticmethod
    def _validate_bias(command, constants, channels) -> None:
        if command.bias_id is None:
            return
        bias = constants[command.bias_id]
        if (
            bias.dtype != "int32"
            or bias.layout != "BIAS"
            or bias.shape != (channels,)
        ):
            raise GraphValidationError(
                f"command {command.command_id!r} bias shape/type mismatch"
            )
