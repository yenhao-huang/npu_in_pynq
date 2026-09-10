import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from src.export.resnet import export_model
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
from src.model.package import REQUIRED_ABI_MAJOR, REQUIRED_CAPABILITIES
from src.model.resnet import (
    ConstantTensor,
    Conv2D,
    Flatten,
    FullyConnected,
    GlobalAveragePool,
    MaxPool,
    Quantization,
    QuantizedGraph,
    Relu,
    ResidualAdd,
    TensorSpec,
)
from src.runtime.model import (
    ModelExecutionError,
    ModelLoadError,
    ModelRuntimeError,
    NPUModelRuntime,
    load_model_package,
)


Q = Quantization(INT32_MAX, 0, 0)


def activation(name, shape, layout="NHWC"):
    return TensorSpec(name, shape, layout, Q)


def resnet_sequence():
    tensors = (
        activation("input", (1, 2, 2, 1)),
        activation("stem", (1, 2, 2, 2)),
        activation("stem_relu", (1, 2, 2, 2)),
        activation("pool", (1, 2, 2, 2)),
        activation("main", (1, 2, 2, 2)),
        activation("added", (1, 2, 2, 2)),
        activation("block_relu", (1, 2, 2, 2)),
        activation("average", (1, 1, 1, 2)),
        activation("flat", (1, 2), "NC"),
        activation("logits", (1, 2), "NC"),
    )
    constants = (
        ConstantTensor("stem_w", (1, 1, 1, 2), "int8", "HWIO", (2, -1)),
        ConstantTensor("main_w", (1, 1, 2, 2), "int8", "HWIO", (1, 2, -2, 1)),
        ConstantTensor("fc_w", (2, 2), "int8", "IO", (1, -1, 2, 1)),
        ConstantTensor("stem_b", (2,), "int32", "BIAS", (1, -2)),
        ConstantTensor("main_b", (2,), "int32", "BIAS", (0, 1)),
        ConstantTensor("fc_b", (2,), "int32", "BIAS", (3, -3)),
    )
    rq = (INT32_MAX, INT32_MAX)
    commands = (
        Conv2D("stem_conv", "input", "stem_w", "stem", rq, (0, 0), bias_id="stem_b"),
        Relu("stem_relu_cmd", "stem", "stem_relu"),
        MaxPool("pool_cmd", "stem_relu", "pool", (1, 1), (1, 1)),
        Conv2D("main_conv", "pool", "main_w", "main", rq, (0, 0), bias_id="main_b"),
        ResidualAdd("residual", "main", "pool", "added"),
        Relu("block_relu_cmd", "added", "block_relu"),
        GlobalAveragePool("average_cmd", "block_relu", "average"),
        Flatten("flatten_cmd", "average", "flat"),
        FullyConnected("fc", "flat", "fc_w", "logits", rq, (0, 0), bias_id="fc_b"),
    )
    return QuantizedGraph(tensors, constants, commands, ("input",), ("logits",))


class FakeRuntime:
    abi_major = REQUIRED_ABI_MAJOR
    capabilities = REQUIRED_CAPABILITIES
    max_m = 2
    max_n = 1
    max_k = 1

    def __init__(self, *, cycles=None):
        self.calls = []
        self.fail_at = None
        self.cycles = cycles

    def run(self, a, b, **timeouts):
        call = len(self.calls) + 1
        self.calls.append((a.copy(), b.copy(), dict(timeouts)))
        if self.fail_at == call:
            raise TimeoutError("injected physical failure")
        if self.cycles is not None:
            self.last_metrics = SimpleNamespace(cycles=self.cycles)
        return np.asarray(a, dtype=np.int32) @ np.asarray(b, dtype=np.int32)


def golden(source):
    constants = {
        item.name: np.asarray(
            item.values, dtype=np.dtype(item.dtype)
        ).reshape(item.shape)
        for item in resnet_sequence().constants
    }
    stem = conv2d_int8(
        source,
        constants["stem_w"],
        multipliers_q31=(INT32_MAX,) * 2,
        shifts=(0, 0),
        output_zero_point=0,
        bias=constants["stem_b"],
    )
    stem_relu = relu_int8(stem, zero_point=0)
    pool = max_pool_int8(stem_relu, window=(1, 1), stride=(1, 1))
    main = conv2d_int8(
        pool,
        constants["main_w"],
        multipliers_q31=(INT32_MAX,) * 2,
        shifts=(0, 0),
        output_zero_point=0,
        bias=constants["main_b"],
    )
    added = residual_add_int8(main, pool, zero_point=0)
    block_relu = relu_int8(added, zero_point=0)
    average = global_average_pool_int8(block_relu)
    flat = flatten_int8(average)
    return fully_connected_int8(
        flat,
        constants["fc_w"],
        multipliers_q31=(INT32_MAX,) * 2,
        shifts=(0, 0),
        output_zero_point=0,
        bias=constants["fc_b"],
    )


class ModelRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        exported = export_model(resnet_sequence(), Path(self.temporary.name) / "resnet")
        self.manifest_path = exported.manifest_path
        self.payload_path = exported.payload_path

    def runtime(self, physical=None):
        physical = physical or FakeRuntime()
        return physical, NPUModelRuntime(physical, load_model_package(self.manifest_path))

    def test_export_load_execute_matches_every_golden_operator(self):
        model = load_model_package(self.manifest_path)
        with self.assertRaises(ValueError):
            model.constants["stem_w"].flags.writeable = True
        physical = FakeRuntime()
        runtime = NPUModelRuntime(physical, model)
        source = np.array([[[[-3], [2]], [[4], [1]]]], dtype=np.int8)
        result = runtime.run({"input": source})
        np.testing.assert_array_equal(result.outputs["logits"], golden(source))
        self.assertGreater(len(physical.calls), 2)
        self.assertEqual(result.metrics.physical_jobs, len(physical.calls))
        self.assertEqual(result.metrics.operation_count, 2 * result.metrics.mac_count)
        self.assertIsNone(result.metrics.physical_cycles)
        self.assertEqual(sum(count for _, count in result.metrics.command_counts), 9)
        self.assertIn(("conv2d", 2), result.metrics.command_counts)

    def test_outputs_are_owned_and_repeat_runs_clear_reused_arena(self):
        _, runtime = self.runtime()
        first_input = np.full((1, 2, 2, 1), 5, dtype=np.int8)
        second_input = np.full((1, 2, 2, 1), -7, dtype=np.int8)
        first = runtime.run({"input": first_input}).outputs["logits"]
        snapshot = first.copy()
        second = runtime.run({"input": second_input}).outputs["logits"]
        np.testing.assert_array_equal(first, snapshot)
        np.testing.assert_array_equal(first, golden(first_input))
        np.testing.assert_array_equal(second, golden(second_input))
        self.assertFalse(np.shares_memory(first, second))

    def test_8x8_resnet_sequence_matches_2x2_with_fewer_jobs(self):
        source = np.array([[[[-3], [2]], [[4], [1]]]], dtype=np.int8)
        results = {}
        for size in (2, 8):
            physical = FakeRuntime()
            physical.max_m = physical.max_n = size
            physical.max_k = 256
            _, runtime = self.runtime(physical)
            results[size] = runtime.run({"input": source})
            np.testing.assert_array_equal(results[size].outputs["logits"], golden(source))
            self.assertTrue(all(a.shape[0] <= size and b.shape[1] <= size
                                for a, b, _ in physical.calls))
        self.assertLess(results[8].metrics.physical_jobs, results[2].metrics.physical_jobs)

    def test_bounded_capture_is_owned_and_rejects_invalid_names_preflight(self):
        physical, runtime = self.runtime(FakeRuntime(cycles=7))
        source = np.ones((1, 2, 2, 1), dtype=np.int8)
        result = runtime.run(
            {"input": source}, capture_tensors=("stem_relu", "added")
        )
        self.assertEqual(tuple(result.captures), ("stem_relu", "added"))
        self.assertEqual(
            result.metrics.physical_cycles,
            result.metrics.physical_jobs * 7,
        )
        added = result.captures["added"]
        self.assertTrue(added.flags.c_contiguous)
        self.assertTrue(added.flags.owndata)
        runtime.run({"input": np.full_like(source, -4)})
        np.testing.assert_array_equal(added, result.captures["added"])
        with self.assertRaises(TypeError):
            result.captures["other"] = added

        calls = len(physical.calls)
        for requested in (("missing",), ("input",), ("added", "added")):
            with self.subTest(requested=requested):
                with self.assertRaises(ValueError):
                    runtime.run({"input": source}, capture_tensors=requested)
        self.assertEqual(len(physical.calls), calls)

    def test_no_capture_and_missing_physical_telemetry_are_explicit(self):
        _, runtime = self.runtime()
        result = runtime.run(
            {"input": np.ones((1, 2, 2, 1), dtype=np.int8)}
        )
        self.assertEqual(dict(result.captures), {})
        self.assertIsNone(result.metrics.physical_cycles)

    def test_mid_model_failure_is_contextual_and_next_run_recovers(self):
        physical, runtime = self.runtime()
        source = np.ones((1, 2, 2, 1), dtype=np.int8)
        physical.fail_at = 3
        with self.assertRaises(ModelExecutionError) as raised:
            runtime.run({"input": source})
        self.assertEqual(raised.exception.command_id, "stem_conv")
        self.assertIn("physical tile M[2:4] N[0:1] K[0:1]", str(raised.exception))
        physical.fail_at = None
        np.testing.assert_array_equal(
            runtime.run({"input": source}).outputs["logits"], golden(source)
        )

    def test_bad_input_fails_before_any_physical_call(self):
        physical, runtime = self.runtime()
        with self.assertRaises(ValueError):
            runtime.run({"input": np.ones((1, 2, 2, 1), dtype=np.int16)})
        self.assertEqual(physical.calls, [])

    def test_corrupt_payload_and_unsupported_command_fail_loading(self):
        payload = bytearray(self.payload_path.read_bytes())
        payload[0] ^= 1
        self.payload_path.write_bytes(payload)
        with self.assertRaisesRegex(ModelLoadError, "digest"):
            load_model_package(self.manifest_path)

        exported = export_model(resnet_sequence(), Path(self.temporary.name) / "resnet")
        manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
        manifest["commands"][0]["op"] = "mystery"
        exported.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ModelLoadError, "unsupported command"):
            load_model_package(exported.manifest_path)

    def test_tampered_memory_and_certificate_are_recomputed(self):
        for field, pattern in (("memory", "memory plan"), ("certificate", "certificate")):
            with self.subTest(field=field):
                exported = export_model(resnet_sequence(), Path(self.temporary.name) / "resnet")
                manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
                if field == "memory":
                    manifest["memory"]["allocations"][0]["last_use"] += 1
                else:
                    manifest["accumulator_certificates"][0]["bounds"][0] += 1
                exported.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ModelLoadError, pattern):
                    load_model_package(exported.manifest_path)

    def test_incompatible_abi_and_arena_fail_before_execution(self):
        model = load_model_package(self.manifest_path)
        physical = FakeRuntime()
        physical.abi_major += 1
        with self.assertRaisesRegex(ModelRuntimeError, "ABI"):
            NPUModelRuntime(physical, model)
        self.assertEqual(physical.calls, [])
        physical.abi_major = REQUIRED_ABI_MAJOR
        with self.assertRaisesRegex(ModelRuntimeError, "arena"):
            NPUModelRuntime(physical, model, arena_limit_bytes=0)
        self.assertEqual(physical.calls, [])


if __name__ == "__main__":
    unittest.main()
