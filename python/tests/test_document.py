import json
import tempfile
import unittest
from pathlib import Path

from swiftlm_coreai.document import ExportDocument
from swiftlm_coreai.errors import ExportError


class ExportDocumentTests(unittest.TestCase):
    def test_valid_document(self) -> None:
        payload = {
            "formatVersion": 2,
            "metadata": {
                "name": "tiny",
                "modelType": "llama",
                "target": "macos_dynamic",
                "maxContextLength": 128,
                "vocabSize": 32,
            },
            "program": _stateless_program(),
            "rootRegion": {
                "parameters": [],
                "operations": [
                    {
                        "key": 0,
                        "operands": [],
                        "results": [0],
                        "parameterBindings": [],
                        "stateBindings": [],
                        "kind": {
                            "tag": "primitive",
                            "primitive": {"opcode": "rms_norm", "attributes": {}},
                        },
                    }
                ],
                "results": [0],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            document = ExportDocument.load(path)
            self.assertEqual(document.model_type, "llama")

    def test_undefined_value_is_rejected(self) -> None:
        payload = {
            "formatVersion": 2,
            "metadata": {
                "name": "tiny",
                "modelType": "llama",
                "target": "macos_dynamic",
                "maxContextLength": 128,
                "vocabSize": 32,
            },
            "program": _stateless_program(),
            "rootRegion": {
                "parameters": [],
                "operations": [
                    {
                        "key": 0,
                        "operands": [4],
                        "results": [0],
                        "parameterBindings": [],
                        "stateBindings": [],
                        "kind": {
                            "tag": "primitive",
                            "primitive": {"opcode": "rms_norm", "attributes": {}},
                        },
                    }
                ],
                "results": [0],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ExportError):
                ExportDocument.load(path)

    def test_unknown_state_binding_is_rejected(self) -> None:
        payload = {
            "formatVersion": 2,
            "metadata": {
                "name": "tiny",
                "modelType": "test",
                "target": "macos_dynamic",
                "maxContextLength": 8,
                "vocabSize": 4,
            },
            "program": _stateful_program(),
            "rootRegion": {
                "parameters": [],
                "operations": [
                    {
                        "key": 0,
                        "operands": [],
                        "results": [0],
                        "parameterBindings": [],
                        "stateBindings": [
                            {"role": "conv_cache", "state": "missing", "axisIndex": 0}
                        ],
                        "kind": {
                            "tag": "primitive",
                            "primitive": {"opcode": "short_conv", "attributes": {}},
                        },
                    }
                ],
                "results": [0],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ExportError) as context:
                ExportDocument.load(path)
            self.assertEqual(context.exception.code, "invalid_graph")

    def test_duplicate_parameter_binding_role_is_rejected(self) -> None:
        payload = {
            "formatVersion": 2,
            "metadata": {
                "name": "tiny",
                "modelType": "test",
                "target": "macos_dynamic",
                "maxContextLength": 8,
                "vocabSize": 4,
            },
            "program": _stateless_program(),
            "rootRegion": {
                "parameters": [],
                "operations": [
                    {
                        "key": 0,
                        "operands": [],
                        "results": [0],
                        "parameterBindings": [
                            {"role": "weight", "tensorName": "first.weight"},
                            {"role": "weight", "tensorName": "second.weight"},
                        ],
                        "stateBindings": [],
                        "kind": {
                            "tag": "primitive",
                            "primitive": {"opcode": "linear", "attributes": {}},
                        },
                    }
                ],
                "results": [0],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ExportError) as context:
                ExportDocument.load(path)
            self.assertEqual(context.exception.code, "invalid_graph")


def _stateless_program() -> dict:
    return {
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
                    }
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
    }


def _stateful_program() -> dict:
    program = _stateless_program()
    program["execution"] = "stateful"
    program["functions"][0]["states"] = [
        {
            "name": "convCache",
            "dataType": "float16",
            "dimensions": [
                {"kind": "fixed", "size": 1},
                {"kind": "fixed", "size": 1},
                {"kind": "fixed", "size": 2},
                {"kind": "fixed", "size": 3},
            ],
        }
    ]
    return program


if __name__ == "__main__":
    unittest.main()
