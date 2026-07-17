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
