"""Fail-closed loading and sequential execution of Phase 2A model packages."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import struct
import time
from types import MappingProxyType
from typing import Any

import numpy as np

from src.export.planner import MemoryPlan, TensorAllocation, plan_memory
from src.model.operators import (
    flatten_int8,
    global_average_pool_int8,
    max_pool_int8,
    relu_int8,
    residual_add_int8,
)
from src.model.package import PackageValidationError, validate_package_data
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
from src.runtime.lowering import MatrixLowerer


class ModelLoadError(PackageValidationError):
    """A package cannot be reconstructed into its certified graph."""


class ModelRuntimeError(RuntimeError):
    """A loaded package is incompatible with the selected physical runtime."""


class ModelExecutionError(RuntimeError):
    """A model command failed without publishing partial output."""

    def __init__(self, command_id: str, cause: BaseException) -> None:
        self.command_id = command_id
        super().__init__(f"model command {command_id!r} failed: {cause}")


@dataclass(frozen=True)
class LoadedModel:
    graph: QuantizedGraph
    constants: Mapping[str, np.ndarray]
    memory_plan: MemoryPlan
    accumulator_certificates: Mapping[str, tuple[int, ...]]
    required_abi_major: int
    required_capabilities: int


@dataclass(frozen=True)
class ModelMetrics:
    command_counts: tuple[tuple[str, int], ...]
    physical_jobs: int
    mac_count: int
    operation_count: int
    physical_cycles: int | None
    elapsed_seconds: float


@dataclass(frozen=True)
class ModelResult:
    outputs: Mapping[str, np.ndarray]
    metrics: ModelMetrics
    captures: Mapping[str, np.ndarray] = field(
        default_factory=lambda: MappingProxyType({})
    )


_COMMAND_FIELDS = {
    "conv2d": (
        Conv2D,
        (
            "command_id", "input_id", "weight_id", "output_id",
            "multipliers_q31", "shifts", "bias_id", "stride",
            "padding", "dilation", "groups",
        ),
    ),
    "fully_connected": (
        FullyConnected,
        (
            "command_id", "input_id", "weight_id", "output_id",
            "multipliers_q31", "shifts", "bias_id",
        ),
    ),
    "residual_add": (ResidualAdd, ("command_id", "lhs_id", "rhs_id", "output_id")),
    "relu": (Relu, ("command_id", "input_id", "output_id")),
    "max_pool": (MaxPool, ("command_id", "input_id", "output_id", "window", "stride", "padding")),
    "global_average_pool": (GlobalAveragePool, ("command_id", "input_id", "output_id")),
    "flatten": (Flatten, ("command_id", "input_id", "output_id")),
}
_COMMAND_NAMES = {
    command_type: name
    for name, (command_type, _fields) in _COMMAND_FIELDS.items()
}


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelLoadError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ModelLoadError(f"non-finite JSON value {value!r}")


def _decode_manifest(data: bytes) -> dict:
    try:
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelLoadError(f"manifest JSON is invalid: {error}") from error
    if not isinstance(value, dict):
        raise ModelLoadError("manifest root must be an object")
    return value


def _payload_path(manifest_path: Path) -> Path:
    suffix = ".npu.json"
    if not manifest_path.name.endswith(suffix):
        raise ModelLoadError("manifest path must end in .npu.json")
    return manifest_path.with_name(manifest_path.name[: -len(suffix)] + ".npu.bin")


def _constant(entry: dict, payload: bytes) -> tuple[ConstantTensor, np.ndarray]:
    try:
        name = entry["name"]
        shape = tuple(entry["shape"])
        dtype_name = entry["dtype"]
        layout = entry["layout"]
        offset = entry["offset"]
        size = entry["size"]
    except KeyError as error:
        raise ModelLoadError(f"constant is missing {error.args[0]!r}") from error
    raw = payload[offset : offset + size]
    storage = bytes(raw)
    if dtype_name == "int8":
        values = tuple(value if value < 128 else value - 256 for value in storage)
        array = np.frombuffer(storage, dtype=np.int8).reshape(shape)
    elif dtype_name == "int32":
        values = tuple(item[0] for item in struct.iter_unpack("<i", storage))
        array = np.frombuffer(storage, dtype="<i4").reshape(shape)
    else:
        raise ModelLoadError(f"unsupported constant dtype {dtype_name!r}")
    record = ConstantTensor(name, shape, dtype_name, layout, values)
    array.flags.writeable = False
    return record, array


def _command(entry: dict):
    op = entry.get("op")
    specification = _COMMAND_FIELDS.get(op)
    if specification is None:
        raise ModelLoadError(f"unsupported command op {op!r}")
    command_type, fields = specification
    allowed = set(fields) | {"op"}
    unknown = set(entry) - allowed
    missing = set(fields) - set(entry)
    if unknown or missing:
        raise ModelLoadError(
            f"command {entry.get('command_id')!r} fields mismatch; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    values = {field: entry[field] for field in fields}
    for field in ("multipliers_q31", "shifts", "stride", "padding", "dilation", "window"):
        if field in values:
            values[field] = tuple(values[field])
    return command_type(**values)


def _expected_bounds(graph: QuantizedGraph) -> dict[str, tuple[int, ...]]:
    constants = {constant.name: constant for constant in graph.constants}
    result = {}
    for command in graph.commands:
        if not isinstance(command, (Conv2D, FullyConnected)):
            continue
        weight = constants[command.weight_id]
        channels = weight.shape[-1]
        bias = constants[command.bias_id].values if command.bias_id else (0,) * channels
        flattened = np.asarray(weight.values, dtype=np.int8).reshape(-1, channels)
        result[command.command_id] = tuple(
            abs(int(bias[channel]))
            + 128 * sum(abs(int(value)) for value in flattened[:, channel])
            for channel in range(channels)
        )
    return result


def _manifest_plan(manifest: dict) -> MemoryPlan:
    memory = manifest["memory"]
    try:
        allocations = tuple(
            TensorAllocation(
                name=item["name"],
                offset=item["offset"],
                logical_bytes=item["logical_bytes"],
                allocated_bytes=item["allocated_bytes"],
                first_definition=item["first_definition"],
                last_use=item["last_use"],
            )
            for item in memory["allocations"]
        )
        return MemoryPlan(
            allocations=allocations,
            arena_bytes=memory["arena_bytes"],
            alignment_bytes=memory["alignment_bytes"],
        )
    except (KeyError, TypeError) as error:
        raise ModelLoadError(f"memory plan is incomplete: {error}") from error


def load_model_package(
    manifest_path: str | Path,
    payload_path: str | Path | None = None,
) -> LoadedModel:
    """Read, reconstruct, and independently verify a package before execution."""

    manifest_path = Path(manifest_path)
    payload_path = (
        Path(payload_path)
        if payload_path is not None
        else _payload_path(manifest_path)
    )
    try:
        manifest_bytes = manifest_path.read_bytes()
        payload = payload_path.read_bytes()
    except OSError as error:
        raise ModelLoadError(f"model package cannot be read: {error}") from error
    manifest = _decode_manifest(manifest_bytes)
    try:
        validate_package_data(manifest, payload)
        tensors = tuple(
            TensorSpec(
                item["name"],
                tuple(item["shape"]),
                item["layout"],
                Quantization(**item["quantization"]),
            )
            for item in manifest["tensors"]
        )
        constant_pairs = tuple(_constant(item, payload) for item in manifest["constants"])
        commands = tuple(_command(item) for item in manifest["commands"])
        graph_record = manifest["graph"]
        graph = QuantizedGraph(
            tensors=tensors,
            constants=tuple(pair[0] for pair in constant_pairs),
            commands=commands,
            inputs=tuple(graph_record["inputs"]),
            outputs=tuple(graph_record["outputs"]),
        )
        declared_plan = _manifest_plan(manifest)
        computed_plan = plan_memory(graph)
        if declared_plan != computed_plan:
            raise ModelLoadError("memory plan does not match reconstructed graph")
        declared_certificates = {
            item["command_id"]: tuple(item["bounds"])
            for item in manifest["accumulator_certificates"]
        }
        expected_certificates = _expected_bounds(graph)
        if declared_certificates != expected_certificates:
            raise ModelLoadError("accumulator certificates do not match constants")
        required = manifest["required_abi"]
    except ModelLoadError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ModelLoadError(f"model reconstruction failed: {error}") from error
    return LoadedModel(
        graph=graph,
        constants=MappingProxyType({pair[0].name: pair[1] for pair in constant_pairs}),
        memory_plan=computed_plan,
        accumulator_certificates=MappingProxyType(expected_certificates),
        required_abi_major=required["major"],
        required_capabilities=required["capabilities"],
    )


class NPUModelRuntime:
    """Execute one validated graph through host operators and MatrixLowerer."""

    def __init__(
        self,
        physical_runtime: Any,
        model: LoadedModel,
        *,
        arena_limit_bytes: int | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(model, LoadedModel):
            raise TypeError("model must be a LoadedModel")
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        try:
            abi_major = int(physical_runtime.abi_major)
            capabilities = int(physical_runtime.capabilities)
        except (AttributeError, TypeError, ValueError) as error:
            raise ModelRuntimeError("runtime ABI metadata is missing") from error
        if abi_major != model.required_abi_major:
            raise ModelRuntimeError(
                f"runtime ABI major {abi_major} != required {model.required_abi_major}"
            )
        missing = model.required_capabilities & ~capabilities
        if missing:
            raise ModelRuntimeError(f"runtime is missing capabilities 0x{missing:08x}")
        if arena_limit_bytes is not None:
            if (
                isinstance(arena_limit_bytes, bool)
                or not isinstance(arena_limit_bytes, int)
                or arena_limit_bytes < 0
            ):
                raise ValueError("arena limit must be a non-negative integer")
            if model.memory_plan.arena_bytes > arena_limit_bytes:
                raise ModelRuntimeError(
                    f"activation arena requires {model.memory_plan.arena_bytes} bytes, "
                    f"only {arena_limit_bytes} available"
                )
        self._lowerer = MatrixLowerer(physical_runtime, monotonic=monotonic)
        self._model = model
        self._monotonic = monotonic
        self._arena = np.empty(model.memory_plan.arena_bytes, dtype=np.uint8)
        self._tensor_specs = {tensor.name: tensor for tensor in model.graph.tensors}

    @property
    def input_names(self) -> tuple[str, ...]:
        """Declared model inputs in package order."""

        return self._model.graph.inputs

    @property
    def output_names(self) -> tuple[str, ...]:
        """Declared model outputs in package order."""

        return self._model.graph.outputs

    def _views(self) -> dict[str, np.ndarray]:
        return {
            tensor.name: self._arena[
                allocation.offset : allocation.offset + allocation.logical_bytes
            ].view(np.int8).reshape(tensor.shape)
            for tensor in self._model.graph.tensors
            for allocation in (self._model.memory_plan.allocation(tensor.name),)
        }

    def run(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        hardware_timeout_cycles: int = 1_000_000,
        software_timeout: float = 5.0,
        capture_tensors: tuple[str, ...] | None = None,
    ) -> ModelResult:
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping")
        expected_names = set(self._model.graph.inputs)
        if set(inputs) != expected_names:
            raise ValueError(f"input names must be exactly {sorted(expected_names)}")
        validated_inputs = {}
        for name in self._model.graph.inputs:
            value = inputs[name]
            spec = self._tensor_specs[name]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.int8
                or value.shape != spec.shape
            ):
                raise ValueError(
                    f"input {name!r} must be signed INT8 with shape {spec.shape}"
                )
            validated_inputs[name] = value
        if capture_tensors is None:
            capture_names = ()
        elif not isinstance(capture_tensors, tuple):
            raise TypeError("capture_tensors must be a tuple of tensor names")
        else:
            capture_names = capture_tensors
        if any(not isinstance(name, str) for name in capture_names):
            raise TypeError("capture tensor names must be strings")
        if len(capture_names) != len(set(capture_names)):
            raise ValueError("capture tensor names must be unique")
        produced_names = {
            command.output_id for command in self._model.graph.commands
        }
        for name in capture_names:
            if name not in produced_names:
                raise ValueError(
                    f"capture tensor {name!r} is not a produced activation"
                )
        if isinstance(software_timeout, bool) or not isinstance(software_timeout, (int, float)):
            raise TypeError("software timeout must be numeric")
        software_timeout = float(software_timeout)
        if not math.isfinite(software_timeout) or software_timeout <= 0:
            raise ValueError("software timeout must be positive and finite")
        start = float(self._monotonic())
        if not math.isfinite(start):
            raise RuntimeError("monotonic clock returned a non-finite value")
        deadline = start + software_timeout
        self._arena.fill(0)
        views = self._views()
        for name, value in validated_inputs.items():
            np.copyto(views[name], value)
        counts = Counter()
        physical_jobs = 0
        mac_count = 0
        physical_cycles = 0
        cycles_available = True
        captured = {}
        graph = self._model.graph
        tensor_specs = self._tensor_specs
        constants = self._model.constants
        for command in graph.commands:
            now = float(self._monotonic())
            remaining = deadline - now
            if not math.isfinite(now) or now < start or remaining <= 0:
                raise ModelExecutionError(
                    command.command_id,
                    TimeoutError("model deadline expired"),
                )
            try:
                if isinstance(command, Conv2D):
                    result = self._lowerer.conv2d(
                        views[command.input_id], constants[command.weight_id],
                        accumulator_bounds=self._model.accumulator_certificates[
                            command.command_id
                        ],
                        multipliers_q31=command.multipliers_q31,
                        shifts=command.shifts,
                        output_zero_point=tensor_specs[
                            command.output_id
                        ].quantization.zero_point,
                        input_zero_point=tensor_specs[
                            command.input_id
                        ].quantization.zero_point,
                        bias=constants.get(command.bias_id), stride=command.stride,
                        padding=command.padding, hardware_timeout_cycles=hardware_timeout_cycles,
                        software_timeout=remaining,
                    )
                    output = result.output
                    physical_jobs += result.metrics.physical_jobs
                    mac_count += result.metrics.mac_count
                    if result.metrics.physical_cycles is None:
                        cycles_available = False
                    else:
                        physical_cycles += result.metrics.physical_cycles
                elif isinstance(command, FullyConnected):
                    result = self._lowerer.fully_connected(
                        views[command.input_id], constants[command.weight_id],
                        accumulator_bounds=self._model.accumulator_certificates[
                            command.command_id
                        ],
                        multipliers_q31=command.multipliers_q31,
                        shifts=command.shifts,
                        output_zero_point=tensor_specs[
                            command.output_id
                        ].quantization.zero_point,
                        input_zero_point=tensor_specs[
                            command.input_id
                        ].quantization.zero_point,
                        bias=constants.get(command.bias_id),
                        hardware_timeout_cycles=hardware_timeout_cycles,
                        software_timeout=remaining,
                    )
                    output = result.output
                    physical_jobs += result.metrics.physical_jobs
                    mac_count += result.metrics.mac_count
                    if result.metrics.physical_cycles is None:
                        cycles_available = False
                    else:
                        physical_cycles += result.metrics.physical_cycles
                elif isinstance(command, ResidualAdd):
                    output = residual_add_int8(
                        views[command.lhs_id], views[command.rhs_id],
                        zero_point=tensor_specs[command.output_id].quantization.zero_point,
                    )
                elif isinstance(command, Relu):
                    output = relu_int8(
                        views[command.input_id],
                        zero_point=tensor_specs[command.output_id].quantization.zero_point,
                    )
                elif isinstance(command, MaxPool):
                    output = max_pool_int8(
                        views[command.input_id],
                        window=command.window,
                        stride=command.stride,
                        padding=command.padding,
                    )
                elif isinstance(command, GlobalAveragePool):
                    output = global_average_pool_int8(views[command.input_id])
                elif isinstance(command, Flatten):
                    output = flatten_int8(views[command.input_id])
                else:  # QuantizedGraph already makes this unreachable.
                    raise RuntimeError(f"unsupported command {type(command).__name__}")
                if output.dtype != np.int8 or output.shape != views[command.output_id].shape:
                    raise RuntimeError("command produced an incompatible tensor")
                np.copyto(views[command.output_id], output)
                if command.output_id in capture_names:
                    captured[command.output_id] = np.array(
                        views[command.output_id],
                        dtype=np.int8,
                        order="C",
                        copy=True,
                    )
                counts[_COMMAND_NAMES[type(command)]] += 1
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise ModelExecutionError(command.command_id, error) from error
        finish = float(self._monotonic())
        if not math.isfinite(finish) or finish < start or finish > deadline:
            terminal = graph.commands[-1].command_id if graph.commands else "<graph>"
            raise ModelExecutionError(terminal, TimeoutError("model deadline expired"))
        outputs = MappingProxyType({
            name: np.array(views[name], dtype=np.int8, order="C", copy=True)
            for name in graph.outputs
        })
        metrics = ModelMetrics(
            command_counts=tuple(sorted(counts.items())),
            physical_jobs=physical_jobs,
            mac_count=mac_count,
            operation_count=2 * mac_count,
            physical_cycles=(
                physical_cycles if cycles_available else None
            ),
            elapsed_seconds=finish - start,
        )
        captures = MappingProxyType(
            {name: captured[name] for name in capture_names}
        )
        return ModelResult(
            outputs=outputs,
            metrics=metrics,
            captures=captures,
        )
