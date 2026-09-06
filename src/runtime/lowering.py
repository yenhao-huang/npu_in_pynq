"""Certified bounded matrix lowering over the public Phase 1 runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from numbers import Real
import time
from typing import Any

import numpy as np

from src.model.numeric import (
    INT8_MAX,
    INT8_MIN,
    INT32_MAX,
    INT32_MIN,
    requantize_int32_to_int8,
)


class LoweringValidationError(ValueError):
    """Inputs or safety evidence cannot be lowered without contract drift."""


class MatrixTileError(RuntimeError):
    """One bounded physical matrix tile failed."""

    def __init__(self, row_range, column_range, k_range, cause):
        self.row_range = row_range
        self.column_range = column_range
        self.k_range = k_range
        super().__init__(
            "physical tile "
            f"M[{row_range[0]}:{row_range[1]}] "
            f"N[{column_range[0]}:{column_range[1]}] "
            f"K[{k_range[0]}:{k_range[1]}] failed: {cause}"
        )


@dataclass(frozen=True)
class LoweringMetrics:
    physical_jobs: int
    mac_count: int
    elapsed_seconds: float
    physical_cycles: int | None = None


@dataclass(frozen=True)
class LoweringResult:
    output: np.ndarray
    metrics: LoweringMetrics


def _integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _bounded(name: str, value: int, minimum: int, maximum: int) -> int:
    value = _integer(name, value)
    if not minimum <= value <= maximum:
        raise LoweringValidationError(
            f"{name} must be in [{minimum}, {maximum}], got {value}"
        )
    return value


def _pair(name: str, values: Sequence[int]) -> tuple[int, int]:
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer pair") from error
    if len(normalized) != 2:
        raise LoweringValidationError(f"{name} must contain two values")
    return tuple(
        _bounded(f"{name}[{index}]", value, 1, INT32_MAX)
        for index, value in enumerate(normalized)
    )


def _padding(values: Sequence[int]) -> tuple[int, int, int, int]:
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise TypeError("padding must contain four integers") from error
    if len(normalized) != 4:
        raise LoweringValidationError(
            "padding must contain top, bottom, left, and right"
        )
    return tuple(
        _bounded(f"padding[{index}]", value, 0, INT32_MAX)
        for index, value in enumerate(normalized)
    )


def _requantization(multipliers, shifts, channels):
    try:
        multipliers = tuple(multipliers)
        shifts = tuple(shifts)
    except TypeError as error:
        raise TypeError("requantization parameters must be sequences") from error
    if len(multipliers) != channels or len(shifts) != channels:
        raise LoweringValidationError(
            f"requantization requires {channels} channel values"
        )
    multipliers = tuple(
        _bounded(f"multipliers_q31[{index}]", value, INT32_MIN, INT32_MAX)
        for index, value in enumerate(multipliers)
    )
    shifts = tuple(
        _bounded(f"shifts[{index}]", value, 0, 31)
        for index, value in enumerate(shifts)
    )
    return multipliers, shifts


def _bias(value, channels):
    if value is None:
        return np.zeros((channels,), dtype=np.int32)
    if not isinstance(value, np.ndarray):
        raise TypeError("bias must be a NumPy array")
    if value.dtype != np.int32 or value.shape != (channels,):
        raise LoweringValidationError(
            f"bias must be signed INT32 with shape ({channels},)"
        )
    return value


def _certificate(weights, bias, evidence):
    output_channels = int(weights.shape[-1])
    if evidence is None:
        raise LoweringValidationError("accumulator safety certificate is required")
    try:
        evidence = tuple(evidence)
    except TypeError as error:
        raise TypeError("accumulator certificate must be a sequence") from error
    if len(evidence) != output_channels:
        raise LoweringValidationError(
            "accumulator certificate channel count mismatch"
        )
    flattened = weights.reshape(-1, output_channels)
    expected = tuple(
        abs(int(bias[channel]))
        + 128 * sum(abs(int(value)) for value in flattened[:, channel])
        for channel in range(output_channels)
    )
    for channel, (reported, required) in enumerate(zip(evidence, expected)):
        reported = _integer(f"accumulator_bounds[{channel}]", reported)
        if reported != required or reported > INT32_MAX:
            raise LoweringValidationError(
                f"accumulator certificate mismatch for channel {channel}: "
                f"reported {reported}, required {required}"
            )
    return expected


class MatrixLowerer:
    """Lower convolution and fully connected work to bounded matrix jobs."""

    def __init__(
        self,
        runtime: Any,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime = runtime
        if not callable(getattr(runtime, "run", None)):
            raise TypeError("runtime must expose a callable run method")
        try:
            self.max_m = int(runtime.max_m)
            self.max_n = int(runtime.max_n)
            self.max_k = int(runtime.max_k)
        except (AttributeError, TypeError, ValueError) as error:
            raise LoweringValidationError(
                "runtime physical limits are missing"
            ) from error
        if self.max_m <= 0 or self.max_n <= 0 or self.max_k <= 0:
            raise LoweringValidationError(
                "runtime physical limits must be positive"
            )
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        self.monotonic = monotonic

    def conv2d(
        self,
        source: np.ndarray,
        weights: np.ndarray,
        *,
        accumulator_bounds: Sequence[int] | None,
        multipliers_q31: Sequence[int],
        shifts: Sequence[int],
        output_zero_point: int,
        bias: np.ndarray | None = None,
        stride: Sequence[int] = (1, 1),
        padding: Sequence[int] = (0, 0, 0, 0),
        input_zero_point: int = 0,
        hardware_timeout_cycles: int = 1_000_000,
        software_timeout: float = 5.0,
    ) -> LoweringResult:
        """Lower batch-one NHWC convolution with on-demand patch tiles."""

        self._int8_array("source", source, 4)
        if not isinstance(weights, np.ndarray):
            raise TypeError("weights must be a NumPy array")
        if (
            weights.dtype != np.int8
            or weights.ndim != 4
            or any(int(size) <= 0 for size in weights.shape)
        ):
            raise LoweringValidationError(
                "weights must be non-empty rank-four signed INT8 HWIO data"
            )
        _, height, width, input_channels = map(int, source.shape)
        kernel_h, kernel_w, weight_channels, output_channels = map(
            int, weights.shape
        )
        if input_channels != weight_channels:
            raise LoweringValidationError(
                "source and weight input channels must match"
            )
        stride_h, stride_w = _pair("stride", stride)
        top, bottom, left, right = _padding(padding)
        input_zero_point = _bounded(
            "input_zero_point", input_zero_point, INT8_MIN, INT8_MAX
        )
        if input_zero_point != 0:
            raise LoweringValidationError(
                "input_zero_point must be zero for symmetric INT8 convolution"
            )
        output_zero_point = _bounded(
            "output_zero_point", output_zero_point, INT8_MIN, INT8_MAX
        )
        multipliers, normalized_shifts = _requantization(
            multipliers_q31, shifts, output_channels
        )
        normalized_bias = _bias(bias, output_channels)
        _certificate(
            weights, normalized_bias, accumulator_bounds
        )
        numerator_h = height + top + bottom - kernel_h
        numerator_w = width + left + right - kernel_w
        if numerator_h < 0 or numerator_w < 0:
            raise LoweringValidationError("kernel exceeds padded input")
        output_h = numerator_h // stride_h + 1
        output_w = numerator_w // stride_w + 1
        logical_m = output_h * output_w
        logical_k = kernel_h * kernel_w * input_channels
        logical_n = output_channels
        weight_matrix = np.ascontiguousarray(
            weights.reshape(logical_k, logical_n), dtype=np.int8
        )

        def make_a(
            row_start: int,
            row_stop: int,
            k_start: int,
            k_stop: int,
        ) -> np.ndarray:
            tile = np.empty(
                (row_stop - row_start, k_stop - k_start),
                dtype=np.int8,
                order="C",
            )
            kernel_plane = kernel_w * input_channels
            for row_offset, position in enumerate(range(row_start, row_stop)):
                output_y, output_x = divmod(position, output_w)
                for k_offset, reduction_index in enumerate(
                    range(k_start, k_stop)
                ):
                    kernel_y, remainder = divmod(
                        reduction_index, kernel_plane
                    )
                    kernel_x, input_channel = divmod(
                        remainder, input_channels
                    )
                    input_y = output_y * stride_h + kernel_y - top
                    input_x = output_x * stride_w + kernel_x - left
                    tile[row_offset, k_offset] = (
                        source[0, input_y, input_x, input_channel]
                        if 0 <= input_y < height and 0 <= input_x < width
                        else input_zero_point
                    )
            return tile

        matrix, jobs, cycles, elapsed = self._run_matrix(
            make_a,
            weight_matrix,
            normalized_bias,
            multipliers,
            normalized_shifts,
            output_zero_point,
            logical_m,
            logical_n,
            logical_k,
            hardware_timeout_cycles,
            software_timeout,
        )
        output = np.array(
            matrix.reshape(1, output_h, output_w, output_channels),
            dtype=np.int8,
            order="C",
            copy=True,
        )
        return LoweringResult(
            output,
            LoweringMetrics(
                physical_jobs=jobs,
                mac_count=logical_m * logical_n * logical_k,
                physical_cycles=cycles,
                elapsed_seconds=elapsed,
            ),
        )

    def fully_connected(
        self,
        source: np.ndarray,
        weights: np.ndarray,
        *,
        accumulator_bounds: Sequence[int] | None,
        multipliers_q31: Sequence[int],
        shifts: Sequence[int],
        output_zero_point: int,
        bias: np.ndarray | None = None,
        input_zero_point: int = 0,
        hardware_timeout_cycles: int = 1_000_000,
        software_timeout: float = 5.0,
    ) -> LoweringResult:
        """Lower batch-one NC by IO fully connected work."""

        self._int8_array("source", source, 2)
        if not isinstance(weights, np.ndarray):
            raise TypeError("weights must be a NumPy array")
        if (
            weights.dtype != np.int8
            or weights.ndim != 2
            or any(int(size) <= 0 for size in weights.shape)
        ):
            raise LoweringValidationError(
                "weights must be non-empty rank-two signed INT8 IO data"
            )
        logical_k, logical_n = map(int, weights.shape)
        if source.shape != (1, logical_k):
            raise LoweringValidationError(
                "source feature count must match weights"
            )
        input_zero_point = _bounded(
            "input_zero_point", input_zero_point, INT8_MIN, INT8_MAX
        )
        if input_zero_point != 0:
            raise LoweringValidationError(
                "input_zero_point must be zero for symmetric INT8 fully connected"
            )
        output_zero_point = _bounded(
            "output_zero_point", output_zero_point, INT8_MIN, INT8_MAX
        )
        multipliers, normalized_shifts = _requantization(
            multipliers_q31, shifts, logical_n
        )
        normalized_bias = _bias(bias, logical_n)
        _certificate(weights, normalized_bias, accumulator_bounds)
        source_matrix = np.ascontiguousarray(source, dtype=np.int8)
        weight_matrix = np.ascontiguousarray(weights, dtype=np.int8)

        def make_a(row_start, row_stop, k_start, k_stop):
            return np.ascontiguousarray(
                source_matrix[row_start:row_stop, k_start:k_stop],
                dtype=np.int8,
            )

        matrix, jobs, cycles, elapsed = self._run_matrix(
            make_a,
            weight_matrix,
            normalized_bias,
            multipliers,
            normalized_shifts,
            output_zero_point,
            1,
            logical_n,
            logical_k,
            hardware_timeout_cycles,
            software_timeout,
        )
        return LoweringResult(
            np.array(matrix, dtype=np.int8, order="C", copy=True),
            LoweringMetrics(
                physical_jobs=jobs,
                mac_count=logical_n * logical_k,
                physical_cycles=cycles,
                elapsed_seconds=elapsed,
            ),
        )

    @staticmethod
    def _int8_array(name, value, rank):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if (
            value.dtype != np.int8
            or value.ndim != rank
            or any(int(size) <= 0 for size in value.shape)
        ):
            raise LoweringValidationError(
                f"{name} must be non-empty rank-{rank} signed INT8 data"
            )
        if value.shape[0] != 1:
            raise LoweringValidationError(
                f"{name} must use batch size one"
            )

    @staticmethod
    def _timeouts(hardware_timeout_cycles, software_timeout):
        if (
            isinstance(hardware_timeout_cycles, bool)
            or not isinstance(hardware_timeout_cycles, int)
            or hardware_timeout_cycles <= 0
        ):
            raise LoweringValidationError(
                "hardware timeout must be a positive integer"
            )
        if isinstance(software_timeout, bool) or not isinstance(
            software_timeout, Real
        ):
            raise TypeError(
                "software timeout must be a positive finite number"
            )
        software_timeout = float(software_timeout)
        if not math.isfinite(software_timeout) or software_timeout <= 0.0:
            raise LoweringValidationError(
                "software timeout must be a positive finite number"
            )
        return hardware_timeout_cycles, software_timeout

    def _run_matrix(
        self,
        make_a,
        weights,
        bias,
        multipliers,
        shifts,
        output_zero_point,
        logical_m,
        logical_n,
        logical_k,
        hardware_timeout_cycles,
        software_timeout,
    ):
        hardware_timeout_cycles, timeout = self._timeouts(
            hardware_timeout_cycles, software_timeout
        )
        start = float(self.monotonic())
        if not math.isfinite(start):
            raise RuntimeError("monotonic clock returned a non-finite value")
        deadline = start + timeout
        output = np.empty((logical_m, logical_n), dtype=np.int8, order="C")
        jobs = 0
        cycle_total = 0
        cycles_available = True
        for row_start in range(0, logical_m, self.max_m):
            row_stop = min(row_start + self.max_m, logical_m)
            for column_start in range(0, logical_n, self.max_n):
                column_stop = min(
                    column_start + self.max_n, logical_n
                )
                accumulator = np.zeros(
                    (row_stop - row_start, column_stop - column_start),
                    dtype=np.int64,
                )
                for k_start in range(0, logical_k, self.max_k):
                    k_stop = min(k_start + self.max_k, logical_k)
                    now = float(self.monotonic())
                    if not math.isfinite(now) or now < start:
                        raise RuntimeError("monotonic clock is invalid")
                    remaining = deadline - now
                    if remaining <= 0.0:
                        raise TimeoutError("bounded matrix lowering timed out")
                    a_tile = np.ascontiguousarray(
                        make_a(row_start, row_stop, k_start, k_stop),
                        dtype=np.int8,
                    )
                    b_tile = np.ascontiguousarray(
                        weights[
                            k_start:k_stop,
                            column_start:column_stop,
                        ],
                        dtype=np.int8,
                    )
                    try:
                        physical_result = self.runtime.run(
                            a_tile,
                            b_tile,
                            hardware_timeout_cycles=hardware_timeout_cycles,
                            software_timeout=remaining,
                        )
                    except Exception as error:
                        raise MatrixTileError(
                            (row_start, row_stop),
                            (column_start, column_stop),
                            (k_start, k_stop),
                            error,
                        ) from error
                    physical_metrics = getattr(
                        self.runtime, "last_metrics", None
                    )
                    physical_cycles = getattr(
                        physical_metrics, "cycles", None
                    )
                    if (
                        isinstance(physical_cycles, bool)
                        or not isinstance(physical_cycles, (int, np.integer))
                        or int(physical_cycles) < 0
                    ):
                        cycles_available = False
                    else:
                        cycle_total += int(physical_cycles)
                    partial = np.asarray(physical_result)
                    expected_shape = (
                        row_stop - row_start,
                        column_stop - column_start,
                    )
                    if (
                        partial.dtype != np.int32
                        or partial.shape != expected_shape
                    ):
                        raise RuntimeError(
                            "physical runtime returned an incompatible "
                            f"result: dtype={partial.dtype} "
                            f"shape={partial.shape}, expected int32 "
                            f"{expected_shape}"
                        )
                    accumulator += partial.astype(np.int64)
                    jobs += 1
                accumulator += bias[column_start:column_stop].astype(
                    np.int64
                )
                if np.any(accumulator < INT32_MIN) or np.any(
                    accumulator > INT32_MAX
                ):
                    raise RuntimeError(
                        "certified matrix accumulator exceeded signed INT32"
                    )
                for row_offset in range(accumulator.shape[0]):
                    for column_offset in range(accumulator.shape[1]):
                        channel = column_start + column_offset
                        output[
                            row_start + row_offset,
                            channel,
                        ] = requantize_int32_to_int8(
                            int(accumulator[row_offset, column_offset]),
                            multipliers[channel],
                            shifts[channel],
                            output_zero_point,
                        )
        finish = float(self.monotonic())
        if not math.isfinite(finish) or finish < start:
            raise RuntimeError("monotonic clock is invalid")
        if finish > deadline:
            raise TimeoutError("bounded matrix lowering timed out")
        return (
            output,
            jobs,
            cycle_total if cycles_available else None,
            finish - start,
        )
