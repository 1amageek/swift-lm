import json
import tempfile
import unittest
from pathlib import Path

from swiftlm_coreai.document import ExportDocument
from swiftlm_coreai.errors import ExportError


class ExportDocumentTests(unittest.TestCase):
    def test_valid_document(self) -> None:
        payload = {
            "formatVersion": 1,
            "metadata": {
                "name": "tiny",
                "modelType": "llama",
                "target": "macos_dynamic",
                "maxContextLength": 128,
                "vocabSize": 32,
            },
            "rootRegion": {
                "parameters": [],
                "operations": [
                    {
                        "key": 0,
                        "operands": [],
                        "results": [0],
                        "parameterBindings": [],
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
            "formatVersion": 1,
            "metadata": {
                "name": "tiny",
                "modelType": "llama",
                "target": "macos_dynamic",
                "maxContextLength": 128,
                "vocabSize": 32,
            },
            "rootRegion": {
                "parameters": [],
                "operations": [
                    {
                        "key": 0,
                        "operands": [4],
                        "results": [0],
                        "parameterBindings": [],
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


if __name__ == "__main__":
    unittest.main()
