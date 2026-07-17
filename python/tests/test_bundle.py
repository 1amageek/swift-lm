import json
import tempfile
import unittest
from pathlib import Path

from swiftlm_coreai.bundle import validate_language_bundle, write_language_bundle_metadata
from swiftlm_coreai.errors import ExportError


class BundleTests(unittest.TestCase):
    def test_writes_and_validates_language_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "model"
            asset = bundle / "model.aimodel"
            asset.mkdir(parents=True)
            tokenizer = bundle / "tokenizer"
            tokenizer.mkdir()
            (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")

            metadata_path = write_language_bundle_metadata(
                bundle,
                name="model",
                model_id="org/model",
                vocab_size=128,
                max_context_length=4096,
            )

            self.assertTrue(metadata_path.is_file())
            metadata = validate_language_bundle(
                bundle,
                expected_name="model",
                expected_model_id="org/model",
                expected_vocab_size=128,
                expected_max_context_length=4096,
            )
            self.assertEqual(metadata["metadata_version"], "0.2")
            self.assertEqual(metadata["assets"]["main"], "model.aimodel")

    def test_rejects_asset_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "model"
            bundle.mkdir()
            payload = {
                "metadata_version": "0.2",
                "kind": "llm",
                "name": "model",
                "assets": {"main": "../model.aimodel"},
                "language": {
                    "tokenizer": "org/model",
                    "vocab_size": 128,
                    "max_context_length": 4096,
                },
            }
            (bundle / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ExportError) as context:
                validate_language_bundle(
                    bundle,
                    expected_name="model",
                    expected_model_id="org/model",
                    expected_vocab_size=128,
                    expected_max_context_length=4096,
                )

            self.assertEqual(context.exception.code, "invalid_bundle")

    def test_requires_embedded_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "model"
            asset = bundle / "model.aimodel"
            asset.mkdir(parents=True)
            write_language_bundle_metadata(
                bundle,
                name="model",
                model_id="org/model",
                vocab_size=128,
                max_context_length=4096,
            )
            (bundle / "tokenizer").mkdir()
            (bundle / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
            (bundle / "tokenizer" / "tokenizer.json").unlink()

            with self.assertRaises(ExportError) as context:
                validate_language_bundle(
                    bundle,
                    expected_name="model",
                    expected_model_id="org/model",
                    expected_vocab_size=128,
                    expected_max_context_length=4096,
                )

            self.assertEqual(context.exception.code, "invalid_bundle")


if __name__ == "__main__":
    unittest.main()
