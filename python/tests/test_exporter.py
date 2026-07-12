import json
import tempfile
import unittest
from pathlib import Path

from swiftlm_coreai.errors import ExportError
from swiftlm_coreai.exporter import export_model


class ExporterTests(unittest.TestCase):
    def test_lfm2_requires_low_level_export(self) -> None:
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

            with self.assertRaises(ExportError) as context:
                export_model(document, "unused", output)

            self.assertEqual(context.exception.code, "low_level_export_required")


if __name__ == "__main__":
    unittest.main()
