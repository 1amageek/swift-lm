import json
import tempfile
import unittest
from pathlib import Path


class AttentionLoweringTests(unittest.TestCase):
    def test_packed_query_gate_is_split_within_each_head(self) -> None:
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
            tensors = _attention_tensors(torch)
            save_file(tensors, root / "model.safetensors")
            path = root / "document.json"
            path.write_text(json.dumps(_attention_payload()), encoding="utf-8")
            document = ExportDocument.load(path)
            weights = SafetensorWeightStore(root, torch, torch.float32)
            model = TorchGraphLowerer(document, weights, torch).make_stateless_model()
            input_ids = torch.tensor([[0, 1]], dtype=torch.int32)
            positions = torch.tensor([[0, 1]], dtype=torch.int32)

            hidden = tensors["embedding.weight"][input_ids.long()]
            expected = _reference_attention(hidden, tensors, torch)
            actual = model(input_ids, positions)

            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def _reference_attention(hidden, tensors: dict, torch):
    batch_size, sequence_length, _ = hidden.shape
    packed_query = torch.nn.functional.linear(hidden, tensors["attention.q_proj"])
    packed_query = packed_query.view(batch_size, sequence_length, 2, 4)
    query, gate = packed_query.chunk(2, dim=-1)
    key = torch.nn.functional.linear(hidden, tensors["attention.k_proj"])
    value = torch.nn.functional.linear(hidden, tensors["attention.v_proj"])
    key = key.view(batch_size, sequence_length, 2, 2)
    value = value.view(batch_size, sequence_length, 2, 2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    scores = torch.matmul(query, key.transpose(-1, -2)) * (2**-0.5)
    causal_mask = torch.triu(
        torch.full((sequence_length, sequence_length), float("-inf")),
        diagonal=1,
    )
    probabilities = torch.softmax(scores + causal_mask, dim=-1)
    output = torch.matmul(probabilities, value)
    output = output.transpose(1, 2).reshape(batch_size, sequence_length, 4)
    output = output * torch.sigmoid(gate.reshape(batch_size, sequence_length, 4))
    return torch.nn.functional.linear(output, tensors["attention.o_proj"])


def _attention_tensors(torch) -> dict:
    return {
        "embedding.weight": torch.tensor(
            [[0.5, -1.0, 0.25, 2.0], [-0.75, 1.5, 0.5, -0.25]],
            dtype=torch.float32,
        ),
        "attention.q_proj": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-4.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "attention.k_proj": torch.eye(4, dtype=torch.float32),
        "attention.v_proj": torch.tensor(
            [
                [1.0, 0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -0.5],
                [0.0, 0.0, 0.25, 1.0],
            ],
            dtype=torch.float32,
        ),
        "attention.o_proj": torch.eye(4, dtype=torch.float32),
    }


def _attention_payload() -> dict:
    sequence = {
        "kind": "dynamic",
        "symbol": "sequence_length",
        "minimum": 1,
        "maximum": 8,
    }
    return {
        "formatVersion": 2,
        "metadata": {
            "name": "packed-query-gate",
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
                        {
                            "role": "embedding_table",
                            "tensorName": "embedding.weight",
                        }
                    ],
                    "stateBindings": [],
                    "kind": {
                        "tag": "primitive",
                        "primitive": {
                            "opcode": "token_embedding",
                            "attributes": {"vocabSize": 2, "embeddingSize": 4},
                        },
                    },
                },
                {
                    "key": 1,
                    "operands": [0],
                    "results": [1],
                    "parameterBindings": [
                        {"role": "q_proj", "tensorName": "attention.q_proj"},
                        {"role": "k_proj", "tensorName": "attention.k_proj"},
                        {"role": "v_proj", "tensorName": "attention.v_proj"},
                        {"role": "o_proj", "tensorName": "attention.o_proj"},
                    ],
                    "stateBindings": [],
                    "kind": {
                        "tag": "primitive",
                        "primitive": {
                            "opcode": "attention",
                            "attributes": {
                                "hiddenSize": 4,
                                "headCount": 2,
                                "kvHeadCount": 2,
                                "headDimension": 2,
                                "causal": True,
                                "bias": False,
                                "attentionScale": None,
                                "rope": None,
                                "qkNorm": {"none": {}},
                                "qkNormEpsilon": 1e-6,
                                "outputGate": {"sigmoidPackedInQProj": {}},
                                "valueProjectionSource": {"dedicatedProjection": {}},
                                "valueNorm": None,
                                "window": None,
                                "sharedKeyValueSourceLayerIndex": None,
                            },
                        },
                    },
                },
            ],
            "results": [1],
        },
    }
