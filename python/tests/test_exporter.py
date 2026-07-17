import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from swiftlm_coreai.exporter import export_model


class ExporterTests(unittest.TestCase):
    def test_export_uses_swift_ir_for_every_model_family(self) -> None:
        for model_type in ("lfm2", "llama", "qwen3"):
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                document = root / "graph.json"
                output = root / "output"
                document.write_text(
                    json.dumps(_document(model_type=model_type, execution="stateless")),
                    encoding="utf-8",
                )
                expected = output / "model"

                with patch(
                    "swiftlm_coreai.exporter.export_ir_language_model",
                    return_value=expected,
                ) as exporter:
                    result = export_model(document, "unused", output)

                self.assertEqual(result, expected)
                exporter.assert_called_once_with(
                    unittest.mock.ANY,
                    "unused",
                    output,
                    overwrite=False,
                )

    def test_stateful_execution_is_read_from_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "graph.json"
            output = root / "output"
            document.write_text(
                json.dumps(_document(model_type="lfm2", execution="stateful")),
                encoding="utf-8",
            )
            expected = output / "model"

            with patch(
                "swiftlm_coreai.exporter.export_ir_language_model",
                return_value=expected,
            ) as exporter:
                result = export_model(document, "unused", output)

            self.assertEqual(result, expected)
            exported_document = exporter.call_args.args[0]
            self.assertEqual(exported_document.execution, "stateful")


def _document(*, model_type: str, execution: str) -> dict:
    states = []
    if execution == "stateful":
        states = [
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
    return {
        "formatVersion": 2,
        "metadata": {
            "name": "model",
            "modelType": model_type,
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
                                {"kind": "fixed", "size": 1},
                            ],
                        },
                        {
                            "name": "position_ids",
                            "dataType": "int32",
                            "dimensions": [
                                {"kind": "fixed", "size": 1},
                                {"kind": "fixed", "size": 1},
                            ],
                        },
                    ],
                    "outputs": [
                        {
                            "name": "logits",
                            "dataType": "float16",
                            "dimensions": [
                                {"kind": "fixed", "size": 1},
                                {"kind": "fixed", "size": 1},
                                {"kind": "fixed", "size": 4},
                            ],
                        }
                    ],
                    "states": states,
                }
            ],
        },
        "rootRegion": {"parameters": [], "operations": [], "results": []},
    }


if __name__ == "__main__":
    unittest.main()
