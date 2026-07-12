import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from swiftlm_coreai.errors import ExportError
from swiftlm_coreai.exporter import export_model


class ExporterTests(unittest.TestCase):
    def test_lfm2_uses_low_level_export(self) -> None:
        payload = {
            "formatVersion": 1,
            "metadata": {
                "name": "lfm2",
                "modelType": "lfm2",
                "target": "macos_dynamic",
                "maxContextLength": 128,
                "vocabSize": 32,
            },
            "rootRegion": {
                "parameters": [],
                "operations": [],
                "results": [],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "graph.json"
            output = Path(directory) / "output"
            document.write_text(json.dumps(payload), encoding="utf-8")

            expected = output / "lfm2"
            with patch("swiftlm_coreai.exporter.export_lfm2_model", return_value=expected) as exporter:
                result = export_model(document, "unused", output)

            self.assertEqual(result, expected)
            exporter.assert_called_once_with(
                "unused",
                output,
                output_name="lfm2",
                max_context_length=128,
                overwrite=False,
            )

    def test_unregistered_model_is_rejected_before_model_download(self) -> None:
        payload = {
            "formatVersion": 1,
            "metadata": {
                "name": "llama",
                "modelType": "llama",
                "target": "macos_dynamic",
                "maxContextLength": 128,
                "vocabSize": 32,
            },
            "rootRegion": {"parameters": [], "operations": [], "results": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "graph.json"
            output = Path(directory) / "output"
            document.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ExportError) as context:
                export_model(document, "unused", output)

            self.assertEqual(context.exception.code, "unsupported_model_type")

    def test_lfm2_rejects_static_ios_target(self) -> None:
        payload = {
            "formatVersion": 1,
            "metadata": {
                "name": "lfm2",
                "modelType": "lfm2",
                "target": "ios_static",
                "maxContextLength": 128,
                "vocabSize": 32,
            },
            "rootRegion": {"parameters": [], "operations": [], "results": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "graph.json"
            output = Path(directory) / "output"
            document.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ExportError) as context:
                export_model(document, "unused", output)

            self.assertEqual(context.exception.code, "unsupported_target")


if __name__ == "__main__":
    unittest.main()
