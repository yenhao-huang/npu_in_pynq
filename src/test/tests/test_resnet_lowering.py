import unittest
from types import SimpleNamespace

import numpy as np

from src.model.numeric import INT32_MAX
from src.model.operators import conv2d_int8, fully_connected_int8
from src.runtime.lowering import (
    LoweringValidationError,
    MatrixLowerer,
)


def bounds_for(weights, bias=None):
    output_channels = int(weights.shape[-1])
    flattened = weights.reshape(-1, output_channels)
    if bias is None:
        bias = np.zeros((output_channels,), dtype=np.int32)
    return tuple(
        abs(int(bias[channel]))
        + 128 * sum(abs(int(value)) for value in flattened[:, channel])
        for channel in range(output_channels)
    )


class FakeRuntime:
    def __init__(self, *, max_m=2, max_n=2, max_k=3, cycles=None):
        self.max_m = max_m
        self.max_n = max_n
        self.max_k = max_k
        self.calls = []
        self.bad_result = None
        self.cycles = None if cycles is None else iter(cycles)

    def run(
        self,
        a_matrix,
        b_matrix,
        *,
        hardware_timeout_cycles,
        software_timeout,
    ):
        self.calls.append(
            (
                np.array(a_matrix, copy=True),
                np.array(b_matrix, copy=True),
                hardware_timeout_cycles,
                software_timeout,
            )
        )
        if self.bad_result is not None:
            return self.bad_result
        if self.cycles is not None:
            value = next(self.cycles)
            self.last_metrics = (
                None if value is None else SimpleNamespace(cycles=value)
            )
        return np.asarray(
            a_matrix.astype(np.int64) @ b_matrix.astype(np.int64),
            dtype=np.int32,
        )


class StepClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return float(next(self.values))


class ConvolutionLoweringTests(unittest.TestCase):
    def test_m_n_k_edges_padding_and_dense_calls_match_golden(self):
        runtime = FakeRuntime(max_m=2, max_n=2, max_k=3)
        lowerer = MatrixLowerer(runtime)
        source = np.array(
            [
                [
                    [[1, -2], [3, 4], [-5, 6]],
                    [[7, 8], [-9, 10], [11, -12]],
                    [[13, -14], [15, 16], [-17, 18]],
                ]
            ],
            dtype=np.int8,
        )
        weights = np.array(
            [
                [
                    [[1, 2, -3], [4, -5, 6]],
                    [[-7, 8, 9], [10, 11, -12]],
                ],
                [
                    [[13, -14, 15], [-16, 17, 18]],
                    [[19, 20, -21], [22, -23, 24]],
                ],
            ],
            dtype=np.int8,
        )
        bias = np.array([5, -7, 11], dtype=np.int32)
        multipliers = (INT32_MAX, INT32_MAX, 1 << 30)
        shifts = (0, 1, 0)
        result = lowerer.conv2d(
            source,
            weights,
            accumulator_bounds=bounds_for(weights, bias),
            multipliers_q31=multipliers,
            shifts=shifts,
            output_zero_point=-3,
            bias=bias,
            stride=(1, 1),
            padding=(1, 0, 1, 0),
            input_zero_point=0,
            software_timeout=10.0,
        )
        expected = conv2d_int8(
            source,
            weights,
            multipliers_q31=multipliers,
            shifts=shifts,
            output_zero_point=-3,
            bias=bias,
            stride=(1, 1),
            padding=(1, 0, 1, 0),
            input_zero_point=0,
        )
        np.testing.assert_array_equal(result.output, expected)
        self.assertEqual(result.metrics.physical_jobs, 30)
        self.assertEqual(len(runtime.calls), 30)
        self.assertEqual(result.metrics.mac_count, 3 * 3 * 3 * 8)
        for a_matrix, b_matrix, hardware_timeout, software_timeout in runtime.calls:
            self.assertEqual(a_matrix.dtype, np.int8)
            self.assertEqual(b_matrix.dtype, np.int8)
            self.assertTrue(a_matrix.flags.c_contiguous)
            self.assertTrue(b_matrix.flags.c_contiguous)
            self.assertLessEqual(a_matrix.shape[0], runtime.max_m)
            self.assertLessEqual(b_matrix.shape[1], runtime.max_n)
            self.assertLessEqual(a_matrix.shape[1], runtime.max_k)
            self.assertEqual(a_matrix.shape[1], b_matrix.shape[0])
            self.assertGreater(hardware_timeout, 0)
            self.assertGreater(software_timeout, 0.0)

    def test_missing_or_invalid_certificate_prevents_calls(self):
        runtime = FakeRuntime()
        lowerer = MatrixLowerer(runtime)
        source = np.ones((1, 1, 1, 4), dtype=np.int8)
        weights = np.ones((1, 1, 4, 1), dtype=np.int8)
        kwargs = dict(
            multipliers_q31=(INT32_MAX,),
            shifts=(0,),
            output_zero_point=0,
        )
        with self.assertRaisesRegex(LoweringValidationError, "certificate"):
            lowerer.conv2d(
                source, weights, accumulator_bounds=None, **kwargs
            )
        with self.assertRaisesRegex(LoweringValidationError, "certificate"):
            lowerer.conv2d(
                source, weights, accumulator_bounds=(1,), **kwargs
            )
        self.assertEqual(runtime.calls, [])

    def test_nonzero_input_zero_point_prevents_convolution_calls(self):
        runtime = FakeRuntime()
        weights = np.ones((1, 1, 1, 1), dtype=np.int8)
        with self.assertRaisesRegex(LoweringValidationError, "symmetric INT8"):
            MatrixLowerer(runtime).conv2d(
                np.ones((1, 1, 1, 1), dtype=np.int8),
                weights,
                accumulator_bounds=bounds_for(weights),
                multipliers_q31=(INT32_MAX,),
                shifts=(0,),
                output_zero_point=0,
                input_zero_point=3,
            )
        self.assertEqual(runtime.calls, [])

    def test_incompatible_physical_result_is_rejected(self):
        runtime = FakeRuntime()
        runtime.bad_result = np.zeros((1, 1), dtype=np.int8)
        lowerer = MatrixLowerer(runtime)
        source = np.ones((1, 1, 1, 1), dtype=np.int8)
        weights = np.ones((1, 1, 1, 1), dtype=np.int8)
        with self.assertRaisesRegex(RuntimeError, "physical runtime"):
            lowerer.conv2d(
                source,
                weights,
                accumulator_bounds=bounds_for(weights),
                multipliers_q31=(INT32_MAX,),
                shifts=(0,),
                output_zero_point=0,
            )


class FullyConnectedLoweringTests(unittest.TestCase):
    def test_nonzero_input_zero_point_prevents_fully_connected_calls(self):
        runtime = FakeRuntime()
        weights = np.ones((1, 1), dtype=np.int8)
        with self.assertRaisesRegex(LoweringValidationError, "symmetric INT8"):
            MatrixLowerer(runtime).fully_connected(
                np.ones((1, 1), dtype=np.int8),
                weights,
                accumulator_bounds=bounds_for(weights),
                multipliers_q31=(INT32_MAX,),
                shifts=(0,),
                output_zero_point=0,
                input_zero_point=-2,
            )
        self.assertEqual(runtime.calls, [])

    def test_m_one_n_edge_and_k_slices_match_golden(self):
        runtime = FakeRuntime(max_m=2, max_n=2, max_k=3)
        lowerer = MatrixLowerer(runtime)
        source = np.array([[-128, -7, -1, 0, 1, 9, 127]], dtype=np.int8)
        weights = np.array(
            [
                [1, 2, 3],
                [-4, 5, 6],
                [7, -8, 9],
                [10, 11, -12],
                [-13, 14, 15],
                [16, -17, 18],
                [19, 20, -21],
            ],
            dtype=np.int8,
        )
        bias = np.array([5, -6, 7], dtype=np.int32)
        multipliers = (INT32_MAX, 1 << 30, INT32_MAX)
        shifts = (0, 0, 1)
        result = lowerer.fully_connected(
            source,
            weights,
            accumulator_bounds=bounds_for(weights, bias),
            multipliers_q31=multipliers,
            shifts=shifts,
            output_zero_point=4,
            bias=bias,
        )
        expected = fully_connected_int8(
            source,
            weights,
            multipliers_q31=multipliers,
            shifts=shifts,
            output_zero_point=4,
            bias=bias,
        )
        np.testing.assert_array_equal(result.output, expected)
        self.assertEqual(result.metrics.physical_jobs, 6)
        self.assertEqual(len(runtime.calls), 6)

    def test_cycle_sum_requires_telemetry_from_every_physical_job(self):
        source = np.ones((1, 3), dtype=np.int8)
        weights = np.ones((3, 2), dtype=np.int8)
        kwargs = dict(
            accumulator_bounds=bounds_for(weights),
            multipliers_q31=(INT32_MAX, INT32_MAX),
            shifts=(0, 0),
            output_zero_point=0,
        )
        complete = MatrixLowerer(
            FakeRuntime(max_n=1, max_k=1, cycles=(1, 2, 3, 4, 5, 6))
        ).fully_connected(source, weights, **kwargs)
        self.assertEqual(complete.metrics.physical_cycles, 21)

        incomplete = MatrixLowerer(
            FakeRuntime(max_n=1, max_k=1, cycles=(1, None, 3, 4, 5, 6))
        ).fully_connected(source, weights, **kwargs)
        self.assertIsNone(incomplete.metrics.physical_cycles)


class DeadlineTests(unittest.TestCase):
    def test_deadline_expiry_prevents_next_submission(self):
        runtime = FakeRuntime(max_m=1, max_n=1, max_k=8)
        clock = StepClock((0.0, 0.1, 1.1))
        lowerer = MatrixLowerer(runtime, monotonic=clock)
        source = np.ones((1, 1, 2, 1), dtype=np.int8)
        weights = np.ones((1, 1, 1, 1), dtype=np.int8)
        with self.assertRaises(TimeoutError):
            lowerer.conv2d(
                source,
                weights,
                accumulator_bounds=bounds_for(weights),
                multipliers_q31=(INT32_MAX,),
                shifts=(0,),
                output_zero_point=0,
                software_timeout=1.0,
            )
        self.assertEqual(len(runtime.calls), 1)
        self.assertAlmostEqual(runtime.calls[0][3], 0.9)

    def test_final_completion_after_deadline_is_rejected(self):
        runtime = FakeRuntime(max_m=1, max_n=1, max_k=8)
        clock = StepClock((0.0, 0.1, 1.1))
        lowerer = MatrixLowerer(runtime, monotonic=clock)
        source = np.ones((1, 1, 1, 1), dtype=np.int8)
        weights = np.ones((1, 1, 1, 1), dtype=np.int8)
        with self.assertRaises(TimeoutError):
            lowerer.conv2d(
                source,
                weights,
                accumulator_bounds=bounds_for(weights),
                multipliers_q31=(INT32_MAX,),
                shifts=(0,),
                output_zero_point=0,
                software_timeout=1.0,
            )
        self.assertEqual(len(runtime.calls), 1)


if __name__ == "__main__":
    unittest.main()
