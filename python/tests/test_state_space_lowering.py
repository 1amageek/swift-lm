import json
import tempfile
import unittest
from pathlib import Path


class StateSpaceLoweringTests(unittest.TestCase):
    def test_gated_deltanet_matches_independent_recurrent_reference(self) -> None:
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
            tensors = _state_space_tensors(torch)
            save_file(tensors, root / "model.safetensors")
            document = _load_document(root, _state_space_payload("stateless"))
            weights = SafetensorWeightStore(root, torch, torch.float32)
            model = TorchGraphLowerer(document, weights, torch).make_stateless_model()
            input_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
            position_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)

            actual = model(input_ids, position_ids)
            hidden = tensors["embedding.weight"][input_ids.long()]
            expected, _, _ = _reference_state_space(hidden, tensors, torch)

            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_stateful_gated_deltanet_matches_stateless_sequence(self) -> None:
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
            save_file(_state_space_tensors(torch), root / "model.safetensors")
            stateless_document = _load_document(root, _state_space_payload("stateless"))
            stateful_document = _load_document(root, _state_space_payload("stateful"))
            weights = SafetensorWeightStore(root, torch, torch.float32)
            stateless = TorchGraphLowerer(
                stateless_document,
                weights,
                torch,
            ).make_stateless_model()
            stateful = TorchGraphLowerer(
                stateful_document,
                weights,
                torch,
            ).make_stateful_model()
            input_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
            positions = torch.tensor([[0, 1, 2]], dtype=torch.int32)
            states = (
                torch.zeros((1, 1, 10, 3), dtype=torch.float32),
                torch.zeros((1, 1, 2, 2, 3), dtype=torch.float32),
            )

            expected = stateless(input_ids, positions)
            for index in range(input_ids.shape[1]):
                actual = stateful(
                    input_ids[:, index : index + 1],
                    positions[:, : index + 1],
                    states,
                )
                torch.testing.assert_close(
                    actual,
                    expected[:, index : index + 1],
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_unsupported_state_space_variant_fails_before_weight_loading(self) -> None:
        try:
            import torch
            from swiftlm_coreai.errors import ExportError
            from swiftlm_coreai.lowering import TorchGraphLowerer
        except ImportError as error:
            self.skipTest(f"Lowering dependencies unavailable: {error}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _state_space_payload("stateless")
            attributes = payload["rootRegion"]["operations"][1]["kind"]["primitive"][
                "attributes"
            ]
            attributes["variant"] = "mamba"
            document = _load_document(root, payload)

            with self.assertRaises(ExportError) as context:
                TorchGraphLowerer(document, object(), torch)

            self.assertEqual(context.exception.code, "unsupported_lowering")
            self.assertIn("state-space variant 'mamba'", context.exception.message)


def _reference_state_space(hidden, tensors: dict, torch):
    batch_size, sequence_length, _ = hidden.shape
    mixed = torch.nn.functional.linear(hidden, tensors["state.in_proj_qkv"])
    mixed = mixed.transpose(-1, -2)
    padded = torch.nn.functional.pad(mixed, (2, 0))
    convolved = torch.nn.functional.conv1d(
        padded,
        tensors["state.conv_weight"],
        groups=10,
    )
    convolved = torch.nn.functional.silu(convolved).transpose(-1, -2)
    query, key, value = torch.split(convolved, (2, 2, 6), dim=-1)
    query = query.reshape(batch_size, sequence_length, 1, 2).repeat_interleave(2, dim=2)
    key = key.reshape(batch_size, sequence_length, 1, 2).repeat_interleave(2, dim=2)
    value = value.reshape(batch_size, sequence_length, 2, 3)
    query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
    query = query * (2**-0.5)

    beta = torch.sigmoid(torch.nn.functional.linear(hidden, tensors["state.in_proj_b"]))
    decay_input = torch.nn.functional.linear(hidden, tensors["state.in_proj_a"])
    decay = -tensors["state.A_log"].exp() * torch.nn.functional.softplus(
        decay_input + tensors["state.dt_bias"]
    )
    gate = torch.nn.functional.linear(hidden, tensors["state.in_proj_z"])
    gate = gate.reshape(batch_size, sequence_length, 2, 3)
    state = torch.zeros((batch_size, 2, 2, 3), dtype=torch.float32)
    outputs = []
    for token_index in range(sequence_length):
        token_key = key[:, token_index]
        token_value = value[:, token_index]
        state = state * decay[:, token_index].exp().unsqueeze(-1).unsqueeze(-1)
        memory = (state * token_key.unsqueeze(-1)).sum(dim=-2)
        delta = (token_value - memory) * beta[:, token_index].unsqueeze(-1)
        state = state + token_key.unsqueeze(-1) * delta.unsqueeze(-2)
        token_query = query[:, token_index]
        outputs.append((state * token_query.unsqueeze(-1)).sum(dim=-2))
    recurrence = torch.stack(outputs, dim=1)
    variance = recurrence.pow(2).mean(dim=-1, keepdim=True)
    normalized = recurrence * torch.rsqrt(variance + 1e-5)
    normalized = normalized * tensors["state.scale"]
    normalized = normalized * torch.nn.functional.silu(gate)
    normalized = normalized.reshape(batch_size, sequence_length, 6)
    output = torch.nn.functional.linear(normalized, tensors["state.out_proj"])
    logits = torch.nn.functional.linear(output, tensors["head.weight"])
    return logits, padded[..., -3:], state


def _state_space_tensors(torch) -> dict:
    generator = torch.Generator().manual_seed(7)

    def random(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.2

    return {
        "embedding.weight": random((4, 4)),
        "state.in_proj_qkv": random((10, 4)),
        "state.in_proj_z": random((6, 4)),
        "state.in_proj_b": random((2, 4)),
        "state.in_proj_a": random((2, 4)),
        "state.out_proj": random((4, 6)),
        "state.scale": random((3,)) + 1,
        "state.conv_weight": random((10, 1, 3)),
        "state.dt_bias": random((2,)),
        "state.A_log": random((2,)),
        "head.weight": torch.eye(4, dtype=torch.float32),
    }


def _load_document(root: Path, payload: dict):
    from swiftlm_coreai.document import ExportDocument

    path = root / f"{payload['program']['execution']}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ExportDocument.load(path)


def _state_space_payload(execution: str) -> dict:
    stateful = execution == "stateful"
    sequence = (
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
                "name": "stateSpaceConvCache",
                "dataType": "float32",
                "dimensions": [
                    {"kind": "fixed", "size": 1},
                    {"kind": "fixed", "size": 1},
                    {"kind": "fixed", "size": 10},
                    {"kind": "fixed", "size": 3},
                ],
            },
            {
                "name": "stateSpaceRecurrentState",
                "dataType": "float32",
                "dimensions": [
                    {"kind": "fixed", "size": 1},
                    {"kind": "fixed", "size": 1},
                    {"kind": "fixed", "size": 2},
                    {"kind": "fixed", "size": 2},
                    {"kind": "fixed", "size": 3},
                ],
            },
        ]
        state_bindings = [
            {"role": "conv_cache", "state": "stateSpaceConvCache", "axisIndex": 0},
            {
                "role": "recurrent_state",
                "state": "stateSpaceRecurrentState",
                "axisIndex": 0,
            },
        ]

    parameters = [
        ("in_proj_qkv", "state.in_proj_qkv"),
        ("in_proj_z", "state.in_proj_z"),
        ("in_proj_b", "state.in_proj_b"),
        ("in_proj_a", "state.in_proj_a"),
        ("out_proj", "state.out_proj"),
        ("scale", "state.scale"),
        ("conv_weight", "state.conv_weight"),
        ("dt_bias", "state.dt_bias"),
        ("A_log", "state.A_log"),
    ]
    return {
        "formatVersion": 2,
        "metadata": {
            "name": "state-space",
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
                                sequence,
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
                                sequence,
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
                            "attributes": {"vocabSize": 4, "embeddingSize": 4},
                        },
                    },
                },
                {
                    "key": 1,
                    "operands": [0],
                    "results": [1],
                    "parameterBindings": [
                        {"role": role, "tensorName": name} for role, name in parameters
                    ],
                    "stateBindings": state_bindings,
                    "kind": {
                        "tag": "primitive",
                        "primitive": {
                            "opcode": "state_space",
                            "attributes": {
                                "hiddenSize": 4,
                                "numHeads": 2,
                                "groupCount": 1,
                                "keyHeadDim": 2,
                                "valueHeadDim": 3,
                                "convKernelSize": 3,
                                "normEpsilon": 1e-5,
                                "variant": "gated_deltanet",
                                "computeDType": "float32",
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
                                "inputSize": 4,
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


if __name__ == "__main__":
    unittest.main()
