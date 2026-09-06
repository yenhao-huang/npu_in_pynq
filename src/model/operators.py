"""Integer-only golden operators for the supported ResNet-18 subset."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .numeric import (
    INT8_MAX,
    INT8_MIN,
    INT32_MAX,
    INT32_MIN,
    mac_int8_int32,
    requantize_int32_to_int8,
    round_ratio_away_from_zero,
    saturate_int8,
    saturating_add_int32,
)


def _integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _bounded(name: str, value: int, minimum: int, maximum: int) -> int:
    value = _integer(name, value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def _int8_array(name: str, value: np.ndarray, rank: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.int8:
        raise ValueError(f"{name} must use signed INT8 dtype")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if any(int(dimension) <= 0 for dimension in value.shape):
        raise ValueError(f"{name} dimensions must be positive")
    if rank in (2, 4) and int(value.shape[0]) != 1:
        raise ValueError(f"{name} must use batch size one")
    return value


def _pair(name: str, values: Sequence[int]) -> tuple[int, int]:
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must contain two integers") from error
    if len(normalized) != 2:
        raise ValueError(f"{name} must contain two integers")
    return tuple(_bounded(f"{name}[{index}]", value, 1, INT32_MAX)
                 for index, value in enumerate(normalized))


def _padding(values: Sequence[int]) -> tuple[int, int, int, int]:
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise TypeError("padding must contain four integers") from error
    if len(normalized) != 4:
        raise ValueError("padding must contain top, bottom, left, and right")
    return tuple(_bounded(f"padding[{index}]", value, 0, INT32_MAX)
                 for index, value in enumerate(normalized))


def _requantization(
    multipliers_q31: Sequence[int],
    shifts: Sequence[int],
    channels: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    try:
        multipliers = tuple(multipliers_q31)
        normalized_shifts = tuple(shifts)
    except TypeError as error:
        raise TypeError("requantization parameters must be integer sequences") from error
    if len(multipliers) != channels or len(normalized_shifts) != channels:
        raise ValueError(
            f"requantization requires one multiplier and shift for {channels} channels"
        )
    multipliers = tuple(
        _bounded(f"multipliers_q31[{index}]", value, INT32_MIN, INT32_MAX)
        for index, value in enumerate(multipliers)
    )
    normalized_shifts = tuple(
        _bounded(f"shifts[{index}]", value, 0, 31)
        for index, value in enumerate(normalized_shifts)
    )
    return multipliers, normalized_shifts


def _bias(value: np.ndarray | None, channels: int) -> np.ndarray:
    if value is None:
        return np.zeros((channels,), dtype=np.int32)
    if not isinstance(value, np.ndarray):
        raise TypeError("bias must be a NumPy array")
    if value.dtype != np.int32 or value.shape != (channels,):
        raise ValueError(f"bias must be signed INT32 with shape ({channels},)")
    return value


def conv2d_int8(
    source: np.ndarray,
    weights: np.ndarray,
    *,
    multipliers_q31: Sequence[int],
    shifts: Sequence[int],
    output_zero_point: int,
    bias: np.ndarray | None = None,
    stride: Sequence[int] = (1, 1),
    padding: Sequence[int] = (0, 0, 0, 0),
    input_zero_point: int = 0,
) -> np.ndarray:
    """Return bit-accurate batch-one NHWC convolution for HWIO weights."""

    source = _int8_array("source", source, 4)
    if not isinstance(weights, np.ndarray):
        raise TypeError("weights must be a NumPy array")
    if weights.dtype != np.int8 or weights.ndim != 4:
        raise ValueError("weights must be rank-four signed INT8 HWIO data")
    if any(int(dimension) <= 0 for dimension in weights.shape):
        raise ValueError("weight dimensions must be positive")
    _, height, width, input_channels = map(int, source.shape)
    kernel_h, kernel_w, weight_channels, output_channels = map(int, weights.shape)
    if input_channels != weight_channels:
        raise ValueError("source and weight input channels must match")
    stride_h, stride_w = _pair("stride", stride)
    top, bottom, left, right = _padding(padding)
    input_zero_point = _bounded(
        "input_zero_point", input_zero_point, INT8_MIN, INT8_MAX
    )
    if input_zero_point != 0:
        raise ValueError(
            "input_zero_point must be zero for symmetric INT8 convolution"
        )
    output_zero_point = _bounded(
        "output_zero_point", output_zero_point, INT8_MIN, INT8_MAX
    )
    multipliers, normalized_shifts = _requantization(
        multipliers_q31, shifts, output_channels
    )
    normalized_bias = _bias(bias, output_channels)
    numerator_h = height + top + bottom - kernel_h
    numerator_w = width + left + right - kernel_w
    if numerator_h < 0 or numerator_w < 0:
        raise ValueError("kernel exceeds padded input")
    output_h = numerator_h // stride_h + 1
    output_w = numerator_w // stride_w + 1
    output = np.empty(
        (1, output_h, output_w, output_channels), dtype=np.int8, order="C"
    )

    for output_y in range(output_h):
        for output_x in range(output_w):
            for output_channel in range(output_channels):
                accumulator = 0
                for kernel_y in range(kernel_h):
                    input_y = output_y * stride_h + kernel_y - top
                    for kernel_x in range(kernel_w):
                        input_x = output_x * stride_w + kernel_x - left
                        for input_channel in range(input_channels):
                            activation = (
                                int(source[0, input_y, input_x, input_channel])
                                if 0 <= input_y < height and 0 <= input_x < width
                                else input_zero_point
                            )
                            accumulator = mac_int8_int32(
                                accumulator,
                                activation,
                                int(
                                    weights[
                                        kernel_y,
                                        kernel_x,
                                        input_channel,
                                        output_channel,
                                    ]
                                ),
                            )
                accumulator = saturating_add_int32(
                    accumulator, int(normalized_bias[output_channel])
                )
                output[0, output_y, output_x, output_channel] = (
                    requantize_int32_to_int8(
                        accumulator,
                        multipliers[output_channel],
                        normalized_shifts[output_channel],
                        output_zero_point,
                    )
                )
    return output


def fully_connected_int8(
    source: np.ndarray,
    weights: np.ndarray,
    *,
    multipliers_q31: Sequence[int],
    shifts: Sequence[int],
    output_zero_point: int,
    bias: np.ndarray | None = None,
    input_zero_point: int = 0,
) -> np.ndarray:
    """Return bit-accurate batch-one NC by IO fully connected output."""

    source = _int8_array("source", source, 2)
    if not isinstance(weights, np.ndarray):
        raise TypeError("weights must be a NumPy array")
    if weights.dtype != np.int8 or weights.ndim != 2:
        raise ValueError("weights must be rank-two signed INT8 IO data")
    if any(int(dimension) <= 0 for dimension in weights.shape):
        raise ValueError("weight dimensions must be positive")
    input_features, output_features = map(int, weights.shape)
    if source.shape != (1, input_features):
        raise ValueError("source feature count must match weights")
    input_zero_point = _bounded(
        "input_zero_point", input_zero_point, INT8_MIN, INT8_MAX
    )
    if input_zero_point != 0:
        raise ValueError(
            "input_zero_point must be zero for symmetric INT8 fully connected"
        )
    output_zero_point = _bounded(
        "output_zero_point", output_zero_point, INT8_MIN, INT8_MAX
    )
    multipliers, normalized_shifts = _requantization(
        multipliers_q31, shifts, output_features
    )
    normalized_bias = _bias(bias, output_features)
    output = np.empty((1, output_features), dtype=np.int8, order="C")
    for output_feature in range(output_features):
        accumulator = 0
        for input_feature in range(input_features):
            accumulator = mac_int8_int32(
                accumulator,
                int(source[0, input_feature]),
                int(weights[input_feature, output_feature]),
            )
        accumulator = saturating_add_int32(
            accumulator, int(normalized_bias[output_feature])
        )
        output[0, output_feature] = requantize_int32_to_int8(
            accumulator,
            multipliers[output_feature],
            normalized_shifts[output_feature],
            output_zero_point,
        )
    return output


def residual_add_int8(
    lhs: np.ndarray, rhs: np.ndarray, *, zero_point: int
) -> np.ndarray:
    """Add identically quantized signed INT8 residual tensors."""

    if not isinstance(lhs, np.ndarray) or not isinstance(rhs, np.ndarray):
        raise TypeError("residual inputs must be NumPy arrays")
    if lhs.dtype != np.int8 or rhs.dtype != np.int8:
        raise ValueError("residual inputs must use signed INT8 dtype")
    if lhs.shape != rhs.shape or lhs.ndim not in (2, 4):
        raise ValueError("residual inputs must have identical NC or NHWC shape")
    if lhs.shape[0] != 1 or any(int(size) <= 0 for size in lhs.shape):
        raise ValueError("residual inputs must be non-empty batch-one tensors")
    zero_point = _bounded("zero_point", zero_point, INT8_MIN, INT8_MAX)
    result = (
        lhs.astype(np.int16)
        + rhs.astype(np.int16)
        - np.int16(zero_point)
    )
    return np.ascontiguousarray(np.clip(result, INT8_MIN, INT8_MAX), dtype=np.int8)


def relu_int8(source: np.ndarray, *, zero_point: int) -> np.ndarray:
    """Clamp a signed INT8 tensor below its quantized real-zero value."""

    if not isinstance(source, np.ndarray):
        raise TypeError("source must be a NumPy array")
    if source.dtype != np.int8 or source.ndim not in (2, 4):
        raise ValueError("source must be signed INT8 NC or NHWC data")
    if source.shape[0] != 1 or any(int(size) <= 0 for size in source.shape):
        raise ValueError("source must be a non-empty batch-one tensor")
    zero_point = _bounded("zero_point", zero_point, INT8_MIN, INT8_MAX)
    return np.array(
        np.maximum(source, np.int8(zero_point)),
        dtype=np.int8,
        order="C",
        copy=True,
    )


def max_pool_int8(
    source: np.ndarray,
    *,
    window: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int] = (0, 0, 0, 0),
) -> np.ndarray:
    """Return signed INT8 NHWC max pooling with -128 padding."""

    source = _int8_array("source", source, 4)
    window_h, window_w = _pair("window", window)
    stride_h, stride_w = _pair("stride", stride)
    top, bottom, left, right = _padding(padding)
    _, height, width, channels = map(int, source.shape)
    numerator_h = height + top + bottom - window_h
    numerator_w = width + left + right - window_w
    if numerator_h < 0 or numerator_w < 0:
        raise ValueError("pool window exceeds padded input")
    output_h = numerator_h // stride_h + 1
    output_w = numerator_w // stride_w + 1
    output = np.full(
        (1, output_h, output_w, channels), INT8_MIN, dtype=np.int8, order="C"
    )
    for output_y in range(output_h):
        for output_x in range(output_w):
            for channel in range(channels):
                maximum = INT8_MIN
                for window_y in range(window_h):
                    input_y = output_y * stride_h + window_y - top
                    for window_x in range(window_w):
                        input_x = output_x * stride_w + window_x - left
                        value = (
                            int(source[0, input_y, input_x, channel])
                            if 0 <= input_y < height and 0 <= input_x < width
                            else INT8_MIN
                        )
                        maximum = max(maximum, value)
                output[0, output_y, output_x, channel] = maximum
    return output


def global_average_pool_int8(source: np.ndarray) -> np.ndarray:
    """Average every NHWC channel, rounding exact ties away from zero."""

    source = _int8_array("source", source, 4)
    _, height, width, channels = map(int, source.shape)
    denominator = height * width
    output = np.empty((1, 1, 1, channels), dtype=np.int8, order="C")
    for channel in range(channels):
        total = int(np.sum(source[0, :, :, channel], dtype=np.int64))
        output[0, 0, 0, channel] = saturate_int8(
            round_ratio_away_from_zero(total, denominator)
        )
    return output


def flatten_int8(source: np.ndarray) -> np.ndarray:
    """Convert owned batch-one 1x1xC NHWC data to owned 1xC NC data."""

    source = _int8_array("source", source, 4)
    if source.shape[1:3] != (1, 1):
        raise ValueError("flatten requires spatial dimensions 1x1")
    return np.array(
        source.reshape(1, int(source.shape[3])),
        dtype=np.int8,
        order="C",
        copy=True,
    )
