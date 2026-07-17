import json
import tempfile
import unittest
from pathlib import Path


class TorchGraphLoweringTests(unittest.TestCase):
    def test_missing_parameter_role_fails_before_weight_loading(self) -> None:
        try:
            import torch
            from swiftlm_coreai.errors import ExportError
            from swiftlm_coreai.lowering import TorchGraphLowerer
        except ImportError as error:
            self.skipTest(f"Lowering dependencies unavailable: {error}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _structural_payload()
            payload["rootRegion"]["operations"][0]["parameterBindings"] = []
            document = _load_document(root, payload)

            with self.assertRaises(ExportError) as context:
                TorchGraphLowerer(document, object(), torch)

            self.assertEqual(context.exception.code, "unsupported_lowering")
            self.assertIn("embedding_table", context.exception.message)

    def test_unsupported_semantics_fail_before_graph_construction(self) -> None:
        try:
            import torch
            from swiftlm_coreai.errors import ExportError
            from swiftlm_coreai.lowering import TorchGraphLowerer
        except ImportError as error:
            self.skipTest(f"Lowering dependencies unavailable: {error}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _structural_payload()
            parallel = payload["rootRegion"]["operations"][2]["kind"]
            parallel["merge"] = {"concat": {}}
            document = _load_document(root, payload)

            with self.assertRaises(ExportError) as context:
                TorchGraphLowerer(document, object(), torch)

            self.assertEqual(context.exception.code, "unsupported_lowering")
            self.assertIn("has no axis contract", context.exception.message)

    def test_structural_operations_follow_swift_ir_semantics(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
            from swiftlm_coreai.lowering import TorchGraphLowerer
            from swiftlm_coreai.weights import SafetensorWeightStore
        except ImportError as error:
            self.skipTest(f"Lowering dependencies unavailable: {error}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(
                {
                    "embedding.weight": torch.eye(2),
                    "scale.two": torch.full((2,), 2.0),
                    "scale.three": torch.full((2,), 3.0),
                    "linear.weight": torch.eye(2),
                    "head.weight": torch.eye(2),
                },
                root / "model.safetensors",
            )
            document = _load_document(root, _structural_payload())
            weights = SafetensorWeightStore(root, torch, torch.float32)
            model = TorchGraphLowerer(document, weights, torch).make_stateless_model()

            input_ids = torch.tensor([[0, 1]], dtype=torch.int32)
            position_ids = torch.tensor([[0, 1]], dtype=torch.int32)
            actual = model(input_ids, position_ids)
            expected = torch.eye(2).unsqueeze(0) * 12

            torch.testing.assert_close(actual, expected)

    def test_parameter_bindings_drive_graph_execution(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
            from swiftlm_coreai.document import ExportDocument
            from swiftlm_coreai.lowering import TorchGraphLowerer
            from swiftlm_coreai.weights import SafetensorWeightStore
        except ImportError as error:
            self.skipTest(f"Lowering dependencies unavailable: {error}")

        payload = {
            "formatVersion": 2,
            "metadata": {
                "name": "tiny",
                "modelType": "test",
                "target": "macos_dynamic",
                "maxContextLength": 8,
                "vocabSize": 4,
            },
            "program": {
                "source": "swift_lmir",
                "execution": "stateless",
                "functions": [
                    {
                        "name": "main",
                        "inputs": [
                            {
                                "name": "input_ids",
                                "dataType": "int32",
                                "dimensions": [
                                    {"kind": "fixed", "size": 1},
                                    {
                                        "kind": "dynamic",
                                        "symbol": "sequence_length",
                                        "minimum": 1,
                                        "maximum": 8,
                                    },
                                ],
                            },
                            {
                                "name": "position_ids",
                                "dataType": "int32",
                                "dimensions": [
                                    {"kind": "fixed", "size": 1},
                                    {
                                        "kind": "dynamic",
                                        "symbol": "sequence_length",
                                        "minimum": 1,
                                        "maximum": 8,
                                    },
                                ],
                            },
                        ],
                        "outputs": [
                            {
                                "name": "logits",
                                "dataType": "float16",
                                "dimensions": [
                                    {"kind": "fixed", "size": 1},
                                    {
                                        "kind": "dynamic",
                                        "symbol": "sequence_length",
                                        "minimum": 1,
                                        "maximum": 8,
                                    },
                                    {"kind": "fixed", "size": 4},
                                ],
                            }
                        ],
                        "states": [],
                    }
                ],
            },
            "rootRegion": {
                "parameters": [],
                "operations": [
                    {
                        "key": 0,
                        "operands": [],
                        "results": [0],
                        "parameterBindings": [
                            {"role": "embedding_table", "tensorName": "embedding.weight"}
                        ],
                        "stateBindings": [],
                        "kind": {
                            "tag": "primitive",
                            "primitive": {
                                "opcode": "token_embedding",
                                "attributes": {"vocabSize": 4, "embeddingSize": 2},
                            },
                        },
                    },
                    {
                        "key": 1,
                        "operands": [0],
                        "results": [1],
                        "parameterBindings": [
                            {"role": "weight", "tensorName": "head.weight"}
                        ],
                        "stateBindings": [],
                        "kind": {
                            "tag": "primitive",
                            "primitive": {
                                "opcode": "output_head",
                                "attributes": {
                                    "inputSize": 2,
                                    "vocabSize": 4,
                                    "tiedToEmbedding": False,
                                    "bias": False,
                                },
                            },
                        },
                    },
                ],
                "results": [1],
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document_path = root / "document.json"
            document_path.write_text(json.dumps(payload), encoding="utf-8")
            embedding = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
                dtype=torch.float32,
            )
            head = torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [-1.0, 1.0], [0.5, -0.5]],
                dtype=torch.float32,
            )
            save_file(
                {"embedding.weight": embedding, "head.weight": head},
                root / "model.safetensors",
            )
            document = ExportDocument.load(document_path)
            weights = SafetensorWeightStore(root, torch, torch.float32)
            model = TorchGraphLowerer(document, weights, torch).make_stateless_model()

            input_ids = torch.tensor([[0, 2]], dtype=torch.int64)
            position_ids = torch.arange(2, dtype=torch.int64).unsqueeze(0)
            actual = model(input_ids, position_ids)
            expected = torch.nn.functional.linear(embedding[input_ids], head)

            torch.testing.assert_close(actual, expected)

    def test_top_k_moe_matches_independent_reference(self) -> None:
        self._assert_moe_matches_reference(
            gate_kind="topK",
            normalize=True,
            use_expert_bias=False,
            routed_scaling_factor=0.75,
        )

    def test_sigmoid_top_k_moe_uses_bias_only_for_selection(self) -> None:
        self._assert_moe_matches_reference(
            gate_kind="sigmoidTopK",
            normalize=True,
            use_expert_bias=True,
            routed_scaling_factor=1.25,
        )

    def test_moe_graph_exports_through_coreai_torch(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
            from swiftlm_coreai.lowering import TorchGraphLowerer
            from swiftlm_coreai.program import export_torch_module
            from swiftlm_coreai.weights import SafetensorWeightStore
        except ImportError as error:
            self.skipTest(f"Core AI export dependencies unavailable: {error}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(_moe_tensors(torch, True), root / "model.safetensors")
            document = _load_document(
                root,
                _moe_payload(
                    gate_kind="sigmoidTopK",
                    normalize=True,
                    use_expert_bias=True,
                    routed_scaling_factor=1.0,
                ),
            )
            weights = SafetensorWeightStore(root, torch, torch.float32)
            model = TorchGraphLowerer(document, weights, torch).make_stateless_model()
            output = root / "moe.aimodel"

            result = export_torch_module(
                model,
                {
                    "input_ids": torch.tensor([[0, 1]], dtype=torch.int32),
                    "position_ids": torch.tensor([[0, 1]], dtype=torch.int32),
                },
                output,
                input_names=("input_ids", "position_ids"),
                output_names=("logits",),
            )

            self.assertEqual(result, output)
            self.assertTrue((output / "metadata.json").is_file())

    def test_unsupported_moe_expert_contract_fails_before_weight_loading(self) -> None:
        try:
            import torch
            from swiftlm_coreai.errors import ExportError
            from swiftlm_coreai.lowering import TorchGraphLowerer
        except ImportError as error:
            self.skipTest(f"Lowering dependencies unavailable: {error}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _moe_payload(
                gate_kind="topK",
                normalize=True,
                use_expert_bias=False,
                routed_scaling_factor=1.0,
            )
            attributes = payload["rootRegion"]["operations"][1]["kind"]["primitive"][
                "attributes"
            ]
            attributes["expertMLP"]["gating"] = {"geglu": {}}
            document = _load_document(root, payload)

            with self.assertRaises(ExportError) as context:
                TorchGraphLowerer(document, object(), torch)

            self.assertEqual(context.exception.code, "unsupported_lowering")
            self.assertIn("gating 'geglu'", context.exception.message)

    def _assert_moe_matches_reference(
        self,
        *,
        gate_kind: str,
        normalize: bool,
        use_expert_bias: bool,
        routed_scaling_factor: float,
    ) -> None:
        try:
            import torch
            from safetensors.torch import save_file
            from swiftlm_coreai.lowering import TorchGraphLowerer
            from swiftlm_coreai.weights import SafetensorWeightStore
        except ImportError as error:
            self.skipTest(f"Lowering dependencies unavailable: {error}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = _moe_tensors(torch, use_expert_bias)
            save_file(tensors, root / "model.safetensors")
            document = _load_document(
                root,
                _moe_payload(
                    gate_kind=gate_kind,
                    normalize=normalize,
                    use_expert_bias=use_expert_bias,
                    routed_scaling_factor=routed_scaling_factor,
                ),
            )
            weights = SafetensorWeightStore(root, torch, torch.float32)
            model = TorchGraphLowerer(document, weights, torch).make_stateless_model()

            input_ids = torch.tensor([[0, 1]], dtype=torch.int32)
            position_ids = torch.tensor([[0, 1]], dtype=torch.int32)
            hidden = tensors["embedding.weight"][input_ids.long()]
            expected = _reference_moe(
                hidden,
                tensors,
                torch,
                gate_kind=gate_kind,
                normalize=normalize,
                use_expert_bias=use_expert_bias,
                routed_scaling_factor=routed_scaling_factor,
            )
            actual = model(input_ids, position_ids)

            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_stateful_short_convolution_matches_full_sequence(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
            from swiftlm_coreai.document import ExportDocument
            from swiftlm_coreai.lowering import TorchGraphLowerer
            from swiftlm_coreai.weights import SafetensorWeightStore
        except ImportError as error:
            self.skipTest(f"Lowering dependencies unavailable: {error}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embedding = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
                dtype=torch.float32,
            )
            save_file(
                {
                    "embedding.weight": embedding,
                    "conv.in_proj": torch.tensor(
                        [
                            [0.5, 0.1],
                            [0.2, -0.3],
                            [0.4, 0.7],
                            [-0.2, 0.6],
                            [0.8, -0.1],
                            [0.3, 0.5],
                        ],
                        dtype=torch.float32,
                    ),
                    "conv.weight": torch.tensor(
                        [[[0.2, 0.3, 0.5]], [[-0.1, 0.4, 0.6]]],
                        dtype=torch.float32,
                    ),
                    "conv.out_proj": torch.tensor(
                        [[0.6, -0.2], [0.1, 0.9]],
                        dtype=torch.float32,
                    ),
                    "head.weight": torch.tensor(
                        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-0.5, 1.0]],
                        dtype=torch.float32,
                    ),
                },
                root / "model.safetensors",
            )
            weights = SafetensorWeightStore(root, torch, torch.float32)
            stateless = _load_document(root, _conv_payload("stateless"))
            stateful = _load_document(root, _conv_payload("stateful"))
            full_model = TorchGraphLowerer(stateless, weights, torch).make_stateless_model()
            decode_model = TorchGraphLowerer(stateful, weights, torch).make_stateful_model()

            tokens = torch.tensor([[0, 2, 1]], dtype=torch.int32)
            cache = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
            for index in range(tokens.shape[1]):
                prefix = tokens[:, : index + 1]
                positions = torch.arange(index + 1, dtype=torch.int32).unsqueeze(0)
                expected = full_model(prefix, positions)[:, -1:, :]
                actual = decode_model(tokens[:, index : index + 1], positions, (cache,))
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _load_document(root: Path, payload: dict):
    from swiftlm_coreai.document import ExportDocument

    path = root / f"{payload['program']['execution']}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ExportDocument.load(path)


def _moe_tensors(torch, use_expert_bias: bool) -> dict:
    tensors = {
        "embedding.weight": torch.eye(2, dtype=torch.float32),
        "router.weight": torch.tensor(
            [[4.0, -1.0], [2.0, 3.0], [-3.0, 1.0]],
            dtype=torch.float32,
        ),
        "head.weight": torch.eye(2, dtype=torch.float32),
    }
    for expert_index in range(3):
        offset = float(expert_index + 1)
        tensors[f"expert.{expert_index}.gate"] = torch.tensor(
            [
                [0.2 * offset, -0.1],
                [0.3, 0.15 * offset],
                [-0.25, 0.1 * offset],
            ],
            dtype=torch.float32,
        )
        tensors[f"expert.{expert_index}.up"] = torch.tensor(
            [
                [0.4, 0.2 * offset],
                [-0.3 * offset, 0.5],
                [0.1, 0.35 * offset],
            ],
            dtype=torch.float32,
        )
        tensors[f"expert.{expert_index}.down"] = torch.tensor(
            [
                [0.5, -0.2 * offset, 0.3],
                [0.1 * offset, 0.4, -0.25],
            ],
            dtype=torch.float32,
        )
    if use_expert_bias:
        tensors["expert.bias"] = torch.tensor([-10.0, -10.0, 10.0])
    return tensors


def _reference_moe(
    hidden,
    tensors: dict,
    torch,
    *,
    gate_kind: str,
    normalize: bool,
    use_expert_bias: bool,
    routed_scaling_factor: float,
):
    router_logits = torch.nn.functional.linear(hidden, tensors["router.weight"]).float()
    if gate_kind == "topK":
        if normalize:
            selected_logits, selected_indices = torch.topk(router_logits, 2, dim=-1)
            selected_scores = torch.softmax(selected_logits, dim=-1)
        else:
            routing_scores = torch.softmax(router_logits, dim=-1)
            selected_scores, selected_indices = torch.topk(routing_scores, 2, dim=-1)
    else:
        routing_scores = torch.sigmoid(router_logits)
        selection_scores = routing_scores
        if use_expert_bias:
            selection_scores = selection_scores + tensors["expert.bias"]
        _, selected_indices = torch.topk(selection_scores, 2, dim=-1)
        selected_scores = torch.gather(routing_scores, -1, selected_indices)
        if normalize:
            selected_scores = selected_scores / selected_scores.sum(dim=-1, keepdim=True)

    output = torch.zeros_like(hidden)
    for batch_index in range(hidden.shape[0]):
        for token_index in range(hidden.shape[1]):
            value = hidden[batch_index, token_index]
            for selected_index in range(selected_indices.shape[-1]):
                expert_index = selected_indices[batch_index, token_index, selected_index].item()
                gate = torch.nn.functional.linear(
                    value,
                    tensors[f"expert.{expert_index}.gate"],
                )
                up = torch.nn.functional.linear(
                    value,
                    tensors[f"expert.{expert_index}.up"],
                )
                expert_output = torch.nn.functional.linear(
                    torch.nn.functional.silu(gate) * up,
                    tensors[f"expert.{expert_index}.down"],
                )
                output[batch_index, token_index] += (
                    expert_output
                    * selected_scores[batch_index, token_index, selected_index]
                    * routed_scaling_factor
                )
    return output


def _moe_payload(
    *,
    gate_kind: str,
    normalize: bool,
    use_expert_bias: bool,
    routed_scaling_factor: float,
) -> dict:
    sequence = {
        "kind": "dynamic",
        "symbol": "sequence_length",
        "minimum": 1,
        "maximum": 8,
    }
    expert_bindings = []
    for expert_index in range(3):
        for role, suffix in (
            ("gate_proj", "gate"),
            ("up_proj", "up"),
            ("down_proj", "down"),
        ):
            expert_bindings.append(
                {
                    "role": f"expert_{expert_index}_{role}",
                    "tensorName": f"expert.{expert_index}.{suffix}",
                }
            )
    parameter_bindings = [
        {"role": "router", "tensorName": "router.weight"},
        *expert_bindings,
    ]
    if use_expert_bias:
        parameter_bindings.append({"role": "expert_bias", "tensorName": "expert.bias"})

    return {
        "formatVersion": 2,
        "metadata": {
            "name": "moe",
            "modelType": "test",
            "target": "macos_dynamic",
            "maxContextLength": 8,
            "vocabSize": 2,
        },
        "program": {
            "source": "swift_lmir",
            "execution": "stateless",
            "functions": [
                {
                    "name": "main",
                    "inputs": [
                        {
                            "name": "input_ids",
                            "dataType": "int32",
                            "dimensions": [
                                {"kind": "fixed", "size": 1},
                                sequence,
                            ],
                        },
                        {
                            "name": "position_ids",
                            "dataType": "int32",
                            "dimensions": [
                                {"kind": "fixed", "size": 1},
                                sequence,
                            ],
                        },
                    ],
                    "outputs": [
                        {
                            "name": "logits",
                            "dataType": "float32",
                            "dimensions": [
                                {"kind": "fixed", "size": 1},
                                sequence,
                                {"kind": "fixed", "size": 2},
                            ],
                        }
                    ],
                    "states": [],
                }
            ],
        },
        "rootRegion": {
            "parameters": [],
            "operations": [
                {
                    "key": 0,
                    "operands": [],
                    "results": [0],
                    "parameterBindings": [
                        {"role": "embedding_table", "tensorName": "embedding.weight"}
                    ],
                    "stateBindings": [],
                    "kind": {
                        "tag": "primitive",
                        "primitive": {
                            "opcode": "token_embedding",
                            "attributes": {"vocabSize": 2, "embeddingSize": 2},
                        },
                    },
                },
                {
                    "key": 1,
                    "operands": [0],
                    "results": [1],
                    "parameterBindings": parameter_bindings,
                    "stateBindings": [],
                    "kind": {
                        "tag": "primitive",
                        "primitive": {
                            "opcode": "moe",
                            "attributes": {
                                "expertCount": 3,
                                "expertsPerToken": 2,
                                "gateKind": {gate_kind: {}},
                                "normalizeRoutingWeights": normalize,
                                "routedScalingFactor": routed_scaling_factor,
                                "useExpertBias": use_expert_bias,
                                "expertMLP": {
                                    "inputSize": 2,
                                    "outputSize": 2,
                                    "intermediateSize": 3,
                                    "activation": {"silu": {}},
                                    "gating": {"swiglu": {}},
                                    "bias": False,
                                },
                            },
                        },
                    },
                },
                {
                    "key": 2,
                    "operands": [1],
                    "results": [2],
                    "parameterBindings": [
                        {"role": "weight", "tensorName": "head.weight"}
                    ],
                    "stateBindings": [],
                    "kind": {
                        "tag": "primitive",
                        "primitive": {
                            "opcode": "output_head",
                            "attributes": {
                                "inputSize": 2,
                                "vocabSize": 2,
                                "tiedToEmbedding": False,
                                "bias": False,
                            },
                        },
                    },
                },
            ],
            "results": [2],
        },
    }


def _conv_payload(execution: str) -> dict:
    stateful = execution == "stateful"
    sequence_dimension = (
        {"kind": "fixed", "size": 1}
        if stateful
        else {
            "kind": "dynamic",
            "symbol": "sequence_length",
            "minimum": 1,
            "maximum": 8,
        }
    )
    states = []
    state_bindings = []
    if stateful:
        states = [
            {
                "name": "convCache",
                "dataType": "float32",
                "dimensions": [
                    {"kind": "fixed", "size": 1},
                    {"kind": "fixed", "size": 1},
                    {"kind": "fixed", "size": 2},
                    {"kind": "fixed", "size": 3},
                ],
            }
        ]
        state_bindings = [
            {"role": "conv_cache", "state": "convCache", "axisIndex": 0}
        ]
    return {
        "formatVersion": 2,
        "metadata": {
            "name": "conv",
            "modelType": "test",
            "target": "macos_dynamic",
            "maxContextLength": 8,
            "vocabSize": 4,
        },
        "program": {
            "source": "swift_lmir",
            "execution": execution,
            "functions": [
                {
                    "name": "main",
                    "inputs": [
                        {
                            "name": "input_ids",
                            "dataType": "int32",
                            "dimensions": [
                                {"kind": "fixed", "size": 1},
                                sequence_dimension,
                            ],
                        },
                        {
                            "name": "position_ids",
                            "dataType": "int32",
                            "dimensions": [
                                {"kind": "fixed", "size": 1},
                                {
                                    "kind": "dynamic",
                                    "symbol": "position_length" if stateful else "sequence_length",
                                    "minimum": 1,
                                    "maximum": 8,
                                },
                            ],
                        },
                    ],
                    "outputs": [
                        {
                            "name": "logits",
                            "dataType": "float32",
                            "dimensions": [
                                {"kind": "fixed", "size": 1},
                                sequence_dimension,
                                {"kind": "fixed", "size": 4},
                            ],
                        }
                    ],
                    "states": states,
                }
            ],
        },
        "rootRegion": {
            "parameters": [],
            "operations": [
                {
                    "key": 0,
                    "operands": [],
                    "results": [0],
                    "parameterBindings": [
                        {"role": "embedding_table", "tensorName": "embedding.weight"}
                    ],
                    "stateBindings": [],
                    "kind": {
                        "tag": "primitive",
                        "primitive": {
                            "opcode": "token_embedding",
                            "attributes": {"vocabSize": 4, "embeddingSize": 2},
                        },
                    },
                },
                {
                    "key": 1,
                    "operands": [0],
                    "results": [1],
                    "parameterBindings": [
                        {"role": "in_proj", "tensorName": "conv.in_proj"},
                        {"role": "conv_weight", "tensorName": "conv.weight"},
                        {"role": "out_proj", "tensorName": "conv.out_proj"},
                    ],
                    "stateBindings": state_bindings,
                    "kind": {
                        "tag": "primitive",
                        "primitive": {
                            "opcode": "short_conv",
                            "attributes": {"hiddenSize": 2, "kernelSize": 3},
                        },
                    },
                },
                {
                    "key": 2,
                    "operands": [1],
                    "results": [2],
                    "parameterBindings": [
                        {"role": "weight", "tensorName": "head.weight"}
                    ],
                    "stateBindings": [],
                    "kind": {
                        "tag": "primitive",
                        "primitive": {
                            "opcode": "output_head",
                            "attributes": {
                                "inputSize": 2,
                                "vocabSize": 4,
                                "tiedToEmbedding": False,
                                "bias": False,
                            },
                        },
                    },
                },
            ],
            "results": [2],
        },
    }


def _structural_payload() -> dict:
    def layer_scale_region(parameter: int, result: int, tensor: str) -> dict:
        return {
            "parameters": [parameter],
            "operations": [
                {
                    "key": result,
                    "operands": [parameter],
                    "results": [result],
                    "parameterBindings": [
                        {"role": "layer_scalar", "tensorName": tensor}
                    ],
                    "stateBindings": [],
                    "kind": {
                        "tag": "primitive",
                        "primitive": {
                            "opcode": "layer_scale",
                            "attributes": {"dimension": 2},
                        },
                    },
                }
            ],
            "results": [result],
        }

    def linear_region(parameter: int, result: int) -> dict:
        return {
            "parameters": [parameter],
            "operations": [
                {
                    "key": result,
                    "operands": [parameter],
                    "results": [result],
                    "parameterBindings": [
                        {"role": "weight", "tensorName": "linear.weight"}
                    ],
                    "stateBindings": [],
                    "kind": {
                        "tag": "primitive",
                        "primitive": {
                            "opcode": "linear",
                            "attributes": {
                                "inputSize": 2,
                                "outputSize": 2,
                                "bias": False,
                            },
                        },
                    },
                }
            ],
            "results": [result],
        }

    sequence = {
        "kind": "dynamic",
        "symbol": "sequence_length",
        "minimum": 1,
        "maximum": 8,
    }
    return {
        "formatVersion": 2,
        "metadata": {
            "name": "structural",
            "modelType": "test",
            "target": "macos_dynamic",
            "maxContextLength": 8,
            "vocabSize": 2,
        },
        "program": {
            "source": "swift_lmir",
            "execution": "stateless",
            "functions": [
                {
                    "name": "main",
                    "inputs": [
                        {
                            "name": "input_ids",
                            "dataType": "int32",
                            "dimensions": [
                                {"kind": "fixed", "size": 1},
                                sequence,
                            ],
                        },
                        {
                            "name": "position_ids",
                            "dataType": "int32",
                            "dimensions": [
                                {"kind": "fixed", "size": 1},
                                sequence,
                            ],
                        },
                    ],
                    "outputs": [
                        {
                            "name": "logits",
                            "dataType": "float32",
                            "dimensions": [
                                {"kind": "fixed", "size": 1},
                                sequence,
                                {"kind": "fixed", "size": 2},
                            ],
                        }
                    ],
                    "states": [],
                }
            ],
        },
        "rootRegion": {
            "parameters": [],
            "operations": [
                {
                    "key": 0,
                    "operands": [],
                    "results": [0],
                    "parameterBindings": [
                        {"role": "embedding_table", "tensorName": "embedding.weight"}
                    ],
                    "stateBindings": [],
                    "kind": {
                        "tag": "primitive",
                        "primitive": {
                            "opcode": "token_embedding",
                            "attributes": {"vocabSize": 2, "embeddingSize": 2},
                        },
                    },
                },
                {
                    "key": 1,
                    "operands": [0],
                    "results": [1],
                    "parameterBindings": [],
                    "stateBindings": [],
                    "kind": {
                        "tag": "repeating",
                        "count": 2,
                        "body": {
                            "parameters": [10],
                            "operations": [
                                {
                                    "key": 11,
                                    "operands": [10],
                                    "results": [11],
                                    "parameterBindings": [],
                                    "stateBindings": [],
                                    "kind": {
                                        "tag": "conditional",
                                        "condition": {
                                            "layerIndices": {"_0": [0]}
                                        },
                                        "then": layer_scale_region(20, 21, "scale.two"),
                                        "else": layer_scale_region(30, 31, "scale.three"),
                                    },
                                }
                            ],
                            "results": [11],
                        },
                    },
                },
                {
                    "key": 2,
                    "operands": [1],
                    "results": [2],
                    "parameterBindings": [],
                    "stateBindings": [],
                    "kind": {
                        "tag": "parallel",
                        "merge": {"add": {}},
                        "branches": [
                            linear_region(40, 41),
                            linear_region(50, 51),
                        ],
                    },
                },
                {
                    "key": 3,
                    "operands": [2],
                    "results": [3],
                    "parameterBindings": [
                        {"role": "weight", "tensorName": "head.weight"}
                    ],
                    "stateBindings": [],
                    "kind": {
                        "tag": "primitive",
                        "primitive": {
                            "opcode": "output_head",
                            "attributes": {
                                "inputSize": 2,
                                "vocabSize": 2,
                                "tiedToEmbedding": False,
                                "bias": False,
                            },
                        },
                    },
                },
            ],
            "results": [3],
        },
    }


if __name__ == "__main__":
    unittest.main()
