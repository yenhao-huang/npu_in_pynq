import unittest

from src.model.numeric import INT32_MAX
from src.model.resnet import (
    ConstantTensor,
    Conv2D,
    Flatten,
    FullyConnected,
    GlobalAveragePool,
    GraphValidationError,
    MaxPool,
    Quantization,
    QuantizedGraph,
    Relu,
    ResidualAdd,
    TensorSpec,
)


Q = Quantization(multiplier_q31=INT32_MAX, shift=0, zero_point=0)


def activation(name, shape, layout="NHWC", quantization=Q):
    return TensorSpec(
        name=name,
        shape=shape,
        layout=layout,
        quantization=quantization,
    )


class RecordValidationTests(unittest.TestCase):
    def test_records_are_immutable_and_validate_scalar_ranges(self):
        tensor = activation("input", (1, 3, 3, 1))
        with self.assertRaises((AttributeError, TypeError)):
            tensor.shape = (1, 1, 1, 1)
        with self.assertRaises(ValueError):
            Quantization(multiplier_q31=1 << 31, shift=0, zero_point=0)
        with self.assertRaises(ValueError):
            activation("bad", (1, 3, 3), layout="NHWC")
        with self.assertRaises(ValueError):
            ConstantTensor(
                name="bad_weight",
                shape=(1, 1, 1, 2),
                dtype="int8",
                layout="HWIO",
                values=(1,),
            )

    def test_unsupported_convolution_parameters_fail(self):
        kwargs = dict(
            command_id="conv",
            input_id="input",
            weight_id="weight",
            output_id="output",
            multipliers_q31=(INT32_MAX,),
            shifts=(0,),
        )
        with self.assertRaises(ValueError):
            Conv2D(**kwargs, groups=2)
        with self.assertRaises(ValueError):
            Conv2D(**kwargs, dilation=(2, 1))
        with self.assertRaises(ValueError):
            Conv2D(**kwargs, stride=(0, 1))


class WholeGraphValidationTests(unittest.TestCase):
    def _valid_graph(self):
        tensors = (
            activation("input", (1, 3, 3, 1)),
            activation("conv_out", (1, 3, 3, 1)),
            activation("relu_out", (1, 3, 3, 1)),
            activation("pool_out", (1, 2, 2, 1)),
            activation("avg_out", (1, 1, 1, 1)),
            activation("flat_out", (1, 1), layout="NC"),
            activation("logits", (1, 2), layout="NC"),
        )
        constants = (
            ConstantTensor(
                name="conv_weight",
                shape=(1, 1, 1, 1),
                dtype="int8",
                layout="HWIO",
                values=(2,),
            ),
            ConstantTensor(
                name="fc_weight",
                shape=(1, 2),
                dtype="int8",
                layout="IO",
                values=(3, -4),
            ),
        )
        commands = (
            Conv2D(
                command_id="conv",
                input_id="input",
                weight_id="conv_weight",
                output_id="conv_out",
                multipliers_q31=(INT32_MAX,),
                shifts=(0,),
            ),
            Relu("relu", "conv_out", "relu_out"),
            MaxPool(
                "pool",
                "relu_out",
                "pool_out",
                window=(2, 2),
                stride=(2, 2),
                padding=(0, 1, 0, 1),
            ),
            GlobalAveragePool("avg", "pool_out", "avg_out"),
            Flatten("flatten", "avg_out", "flat_out"),
            FullyConnected(
                command_id="fc",
                input_id="flat_out",
                weight_id="fc_weight",
                output_id="logits",
                multipliers_q31=(INT32_MAX, INT32_MAX),
                shifts=(0, 0),
            ),
        )
        return QuantizedGraph(
            tensors=tensors,
            constants=constants,
            commands=commands,
            inputs=("input",),
            outputs=("logits",),
        )

    def test_valid_resnet_operator_sequence(self):
        graph = self._valid_graph()
        self.assertEqual(graph.outputs, ("logits",))
        self.assertEqual(tuple(command.command_id for command in graph.commands),
                         ("conv", "relu", "pool", "avg", "flatten", "fc"))

    def test_matrix_operators_reject_asymmetric_input_quantization(self):
        asymmetric = Quantization(
            multiplier_q31=INT32_MAX,
            shift=0,
            zero_point=7,
        )
        valid = self._valid_graph()
        conv_tensors = tuple(
            activation(
                tensor.name,
                tensor.shape,
                tensor.layout,
                asymmetric if tensor.name == "input" else tensor.quantization,
            )
            for tensor in valid.tensors
        )
        with self.assertRaisesRegex(GraphValidationError, "symmetric INT8 input"):
            QuantizedGraph(
                tensors=conv_tensors,
                constants=valid.constants,
                commands=valid.commands,
                inputs=valid.inputs,
                outputs=valid.outputs,
            )

        tensors = (
            activation("input", (1, 1), "NC", asymmetric),
            activation("output", (1, 1), "NC"),
        )
        constants = (
            ConstantTensor("weight", (1, 1), "int8", "IO", (1,)),
        )
        with self.assertRaisesRegex(GraphValidationError, "symmetric INT8 input"):
            QuantizedGraph(
                tensors=tensors,
                constants=constants,
                commands=(
                    FullyConnected(
                        "fc",
                        "input",
                        "weight",
                        "output",
                        (INT32_MAX,),
                        (0,),
                    ),
                ),
                inputs=("input",),
                outputs=("output",),
            )

    def test_duplicate_ids_and_unknown_commands_fail(self):
        valid = self._valid_graph()
        with self.assertRaisesRegex(GraphValidationError, "duplicate"):
            QuantizedGraph(
                tensors=valid.tensors + (valid.tensors[0],),
                constants=valid.constants,
                commands=valid.commands,
                inputs=valid.inputs,
                outputs=valid.outputs,
            )
        with self.assertRaisesRegex(GraphValidationError, "unsupported command"):
            QuantizedGraph(
                tensors=(activation("input", (1, 1, 1, 1)),),
                constants=(),
                commands=(object(),),
                inputs=("input",),
                outputs=("input",),
            )

    def test_future_reference_cycle_and_shape_mismatch_fail(self):
        tensors = (
            activation("input", (1, 1, 1, 1)),
            activation("a", (1, 1, 1, 1)),
            activation("b", (1, 1, 1, 1)),
        )
        with self.assertRaisesRegex(GraphValidationError, "not available"):
            QuantizedGraph(
                tensors=tensors,
                constants=(),
                commands=(Relu("a_from_b", "b", "a"), Relu("b_from_a", "a", "b")),
                inputs=("input",),
                outputs=("b",),
            )

        valid = self._valid_graph()
        bad_tensors = tuple(
            activation("conv_out", (1, 2, 3, 1))
            if tensor.name == "conv_out"
            else tensor
            for tensor in valid.tensors
        )
        with self.assertRaisesRegex(GraphValidationError, "shape"):
            QuantizedGraph(
                tensors=bad_tensors,
                constants=valid.constants,
                commands=valid.commands,
                inputs=valid.inputs,
                outputs=valid.outputs,
            )

    def test_reference_kind_mismatch_fails_explicitly(self):
        tensors = (
            activation("input", (1, 1, 1, 1)),
            activation("output", (1, 1, 1, 1)),
        )
        constants = (
            ConstantTensor(
                name="constant",
                shape=(1, 1, 1, 1),
                dtype="int8",
                layout="HWIO",
                values=(1,),
            ),
        )
        with self.assertRaisesRegex(GraphValidationError, "activation tensor"):
            QuantizedGraph(
                tensors=tensors,
                constants=constants,
                commands=(Relu("relu", "constant", "output"),),
                inputs=("input",),
                outputs=("output",),
            )

    def test_residual_requires_identical_shape_and_quantization(self):
        other_q = Quantization(multiplier_q31=INT32_MAX - 1, shift=0, zero_point=0)
        tensors = (
            activation("lhs", (1, 1, 1, 1)),
            activation("rhs", (1, 1, 1, 1), quantization=other_q),
            activation("out", (1, 1, 1, 1)),
        )
        with self.assertRaisesRegex(GraphValidationError, "quantization"):
            QuantizedGraph(
                tensors=tensors,
                constants=(),
                commands=(ResidualAdd("add", "lhs", "rhs", "out"),),
                inputs=("lhs", "rhs"),
                outputs=("out",),
            )


if __name__ == "__main__":
    unittest.main()
