import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from swiftlm_coreai.errors import ExportError
from swiftlm_coreai.ir_export import _copy_tokenizer_assets, _validate_source_contract


class IRExportTests(unittest.TestCase):
    def test_tokenizer_assets_are_copied_without_loading_transformers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "tokenizer.json").write_text("{}", encoding="utf-8")
            (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (source / "chat_template.jinja").write_text("template", encoding="utf-8")
            (source / "config.json").write_text("{}", encoding="utf-8")

            _copy_tokenizer_assets(source, output)

            self.assertEqual((output / "tokenizer.json").read_text(), "{}")
            self.assertEqual((output / "chat_template.jinja").read_text(), "template")
            self.assertFalse((output / "config.json").exists())

    def test_tokenizer_export_requires_tokenizer_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            with self.assertRaises(ExportError) as context:
                _copy_tokenizer_assets(root, root / "output")

            self.assertEqual(context.exception.code, "tokenizer_export_failed")

    def test_multimodal_source_uses_text_config_vocab_size(self) -> None:
        document = SimpleNamespace(
            model_type="qwen3_5",
            metadata={"vocabSize": 248320},
        )
        config = {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "vocab_size": 248320,
            },
        }

        _validate_source_contract(document, config)

    def test_multimodal_source_reports_nested_vocab_mismatch(self) -> None:
        document = SimpleNamespace(
            model_type="qwen3_5",
            metadata={"vocabSize": 248320},
        )
        config = {
            "model_type": "qwen3_5",
            "text_config": {"vocab_size": 1000},
        }

        with self.assertRaises(ExportError) as context:
            _validate_source_contract(document, config)

        self.assertEqual(context.exception.code, "vocab_size_mismatch")


if __name__ == "__main__":
    unittest.main()
