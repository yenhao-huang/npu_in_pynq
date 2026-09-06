import unittest

import numpy as np

from src.model.numeric import INT32_MAX
from src.model.operators import (
    conv2d_int8,
    flatten_int8,
    fully_connected_int8,
    global_average_pool_int8,
    max_pool_int8,
    relu_int8,
    residual_add_int8,
)


def round_away(numerator, denominator):
    magnitude, remainder = divmod(abs(int(numerator)), int(denominator))
    if remainder * 2 >= denominator:
        magnitude += 1
    return -magnitude if numerator < 0 else magnitude


def requantize(value, multiplier, shift, zero_point):
    rounded = round_away(int(value) * int(multiplier), 1 << (31 + int(shift)))
    return max(-128, min(127, rounded + int(zero_point)))


def scalar_conv(
    source,
    weights,
    multipliers,
    shifts,
    output_zero_point,
    bias,
    stride,
    padding,
    input_zero_point,
):
    _, height, width, channels = source.shape
    kernel_h, kernel_w, _, output_channels = weights.shape
    stride_h, stride_w = stride
    top, bottom, left, right = padding
    out_h = (height + top + bottom - kernel_h) // stride_h + 1
    out_w = (width + left + right - kernel_w) // stride_w + 1
    output = np.empty((1, out_h, out_w, output_channels), dtype=np.int8)
    for oy in range(out_h):
        for ox in range(out_w):
            for oc in range(output_channels):
                accumulator = 0
                for ky in range(kernel_h):
                    iy = oy * stride_h + ky - top
                    for kx in range(kernel_w):
                        ix = ox * stride_w + kx - left
                        for ic in range(channels):
                            value = (
                                int(source[0, iy, ix, ic])
                                if 0 <= iy < height and 0 <= ix < width
                                else input_zero_point
                            )
                            accumulator += value * int(weights[ky, kx, ic, oc])
                            accumulator = max(-(1 << 31), min((1 << 31) - 1, accumulator))
                accumulator = max(
                    -(1 << 31),
                    min((1 << 31) - 1, accumulator + int(bias[oc])),
                )
                output[0, oy, ox, oc] = requantize(
                    accumulator,
                    multipliers[oc],
                    shifts[oc],
                    output_zero_point,
                )
    return output


class ConvolutionTests(unittest.TestCase):
    def test_strided_padded_per_channel_requantization_matches_scalar(self):
        source = np.array(
            [[[[1], [-2], [3]], [[4], [-5], [6]], [[7], [8], [-9]]]],
            dtype=np.int8,
        )
        weights = np.array(
            [
                [[[2, -3]], [[-1, 4]]],
                [[[5, 1]], [[-2, -2]]],
            ],
            dtype=np.int8,
        )
        bias = np.array([7, -11], dtype=np.int32)
        multipliers = (INT32_MAX, 1 << 30)
        shifts = (0, 0)
        expected = scalar_conv(
            source,
            weights,
            multipliers,
            shifts,
            3,
            bias,
            (2, 2),
            (1, 0, 1, 0),
            0,
        )
        actual = conv2d_int8(
            source,
            weights,
            multipliers_q31=multipliers,
            shifts=shifts,
            output_zero_point=3,
            bias=bias,
            stride=(2, 2),
            padding=(1, 0, 1, 0),
            input_zero_point=0,
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.int8)
        self.assertTrue(actual.flags.c_contiguous)

    def test_nonzero_input_zero_point_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "symmetric INT8 convolution"):
            conv2d_int8(
                np.ones((1, 1, 1, 1), dtype=np.int8),
                np.ones((1, 1, 1, 1), dtype=np.int8),
                multipliers_q31=(INT32_MAX,),
                shifts=(0,),
                output_zero_point=0,
                input_zero_point=1,
            )

    def test_signed_endpoint_convolution(self):
        source = np.array([[[[-128, 127]]]], dtype=np.int8)
        weights = np.array([[[[127], [-128]]]], dtype=np.int8)
        actual = conv2d_int8(
            source,
            weights,
            multipliers_q31=(INT32_MAX,),
            shifts=(0,),
            output_zero_point=0,
        )
        expected_accumulator = -128 * 127 + 127 * -128
        self.assertEqual(int(actual[0, 0, 0, 0]), requantize(
            expected_accumulator, INT32_MAX, 0, 0
        ))


class HostOperatorTests(unittest.TestCase):
    def test_residual_relu_and_pooling(self):
        lhs = np.array([[[[-128, 120], [5, -7]], [[80, 12], [-3, 127]]]], dtype=np.int8)
        rhs = np.array([[[[-10, 20], [6, -8]], [[90, -30], [4, 9]]]], dtype=np.int8)
        added = residual_add_int8(lhs, rhs, zero_point=-3)
        expected_add = np.clip(
            lhs.astype(np.int16) + rhs.astype(np.int16) + 3,
            -128,
            127,
        ).astype(np.int8)
        np.testing.assert_array_equal(added, expected_add)
        activated = relu_int8(added, zero_point=-3)
        np.testing.assert_array_equal(activated, np.maximum(expected_add, -3))
        pooled = max_pool_int8(
            activated,
            window=(2, 2),
            stride=(1, 1),
            padding=(1, 0, 1, 0),
        )
        padded = np.pad(
            activated,
            ((0, 0), (1, 0), (1, 0), (0, 0)),
            constant_values=-128,
        )
        expected_pool = np.empty((1, 2, 2, 2), dtype=np.int8)
        for output_y in range(2):
            for output_x in range(2):
                expected_pool[0, output_y, output_x, :] = np.max(
                    padded[0, output_y:output_y + 2, output_x:output_x + 2, :],
                    axis=(0, 1),
                )
        np.testing.assert_array_equal(pooled, expected_pool)

    def test_global_average_ties_away_from_zero_and_flatten(self):
        source = np.array(
            [[[[0, 0], [1, -1]], [[0, 0], [1, -1]]]],
            dtype=np.int8,
        )
        averaged = global_average_pool_int8(source)
        np.testing.assert_array_equal(
            averaged,
            np.array([[[[1, -1]]]], dtype=np.int8),
        )
        flattened = flatten_int8(averaged)
        np.testing.assert_array_equal(flattened, np.array([[1, -1]], dtype=np.int8))
        self.assertTrue(flattened.flags.owndata)

    def test_invalid_host_operator_inputs_fail(self):
        with self.assertRaises(ValueError):
            residual_add_int8(
                np.zeros((1, 1, 1, 1), dtype=np.int8),
                np.zeros((1, 2, 1, 1), dtype=np.int8),
                zero_point=0,
            )
        with self.assertRaises(ValueError):
            relu_int8(np.zeros((1, 1), dtype=np.float32), zero_point=0)


class FullyConnectedTests(unittest.TestCase):
    def test_nonzero_input_zero_point_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "symmetric INT8 fully connected"):
            fully_connected_int8(
                np.ones((1, 1), dtype=np.int8),
                np.ones((1, 1), dtype=np.int8),
                multipliers_q31=(INT32_MAX,),
                shifts=(0,),
                output_zero_point=0,
                input_zero_point=-1,
            )

    def test_per_channel_output_matches_scalar_reference(self):
        source = np.array([[-128, -1, 0, 1, 127]], dtype=np.int8)
        weights = np.array(
            [[127, -128], [-2, 3], [5, -7], [11, 13], [-128, 127]],
            dtype=np.int8,
        )
        bias = np.array([19, -23], dtype=np.int32)
        multipliers = (INT32_MAX, 1 << 30)
        actual = fully_connected_int8(
            source,
            weights,
            multipliers_q31=multipliers,
            shifts=(0, 1),
            output_zero_point=-5,
            bias=bias,
        )
        expected = np.empty((1, 2), dtype=np.int8)
        for column in range(2):
            accumulator = 0
            for index in range(5):
                accumulator += int(source[0, index]) * int(weights[index, column])
                accumulator = max(-(1 << 31), min((1 << 31) - 1, accumulator))
            accumulator = max(
                -(1 << 31),
                min((1 << 31) - 1, accumulator + int(bias[column])),
            )
            expected[0, column] = requantize(
                accumulator, multipliers[column], (0, 1)[column], -5
            )
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
