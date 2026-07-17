"""Bridge validated swift-lm documents to Core AI exporters."""

from __future__ import annotations

from pathlib import Path

from .bundle import validate_language_bundle
from .document import ExportDocument
from .errors import ExportError
from .lfm2 import export_lfm2_model

LOW_LEVEL_ONLY_MODEL_TYPES = {"lfm2", "lfm2_moe"}
APPLE_HIGH_LEVEL_MODEL_TYPES = {
    "gemma3",
    "gemma3_text",
    "gpt_oss",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen2_5",
    "qwen3",
    "qwen3_moe",
    "qwen3_vl",
}


def validate_document(path: Path) -> ExportDocument:
    """Load and validate a Swift-produced export document."""
    return ExportDocument.load(path)


def export_model(
    document_path: Path,
    model_id: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
    stateful: bool = False,
) -> Path:
    """Export a validated document using the family-specific Core AI pipeline.

    The Swift document is a required preflight contract. The exporter refuses
    to infer a different model family or platform from the Hugging Face bundle.
    Apple-registry Transformer families use Apple's official pipeline; LFM2
    families use the low-level Hugging Face Torch adapter.
    """
    document = validate_document(document_path)
    if document.model_type in LOW_LEVEL_ONLY_MODEL_TYPES:
        if document.target != "macos_dynamic":
            raise ExportError(
                "unsupported_target",
                f"{document.model_type} low-level export currently requires macos_dynamic",
            )
        return export_lfm2_model(
            model_id,
            output_dir,
            output_name=document.metadata["name"],
            max_context_length=document.metadata["maxContextLength"],
            overwrite=overwrite,
            stateful=stateful,
        )
    if stateful:
        raise ExportError(
            "unsupported_stateful_model",
            f"stateful export is only supported for {sorted(LOW_LEVEL_ONLY_MODEL_TYPES)}",
        )
    if document.model_type not in APPLE_HIGH_LEVEL_MODEL_TYPES:
        raise ExportError(
            "unsupported_model_type",
            f"{document.model_type} is not supported by Apple's high-level Core AI model registry",
        )
    try:
        from transformers import AutoConfig
        from coreai_models.export.pipeline import ExportConfig, export_model as apple_export
    except ImportError as error:
        raise ExportError(
            "missing_coreai_dependencies",
            "Install the pinned Python dependencies from python/pyproject.toml",
        ) from error

    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception as error:
        raise ExportError("hf_config_load_failed", f"Could not load {model_id}") from error

    actual_model_type = getattr(config, "model_type", None)
    if actual_model_type != document.model_type:
        raise ExportError(
            "model_type_mismatch",
            f"document declares {document.model_type!r}, HF config declares {actual_model_type!r}",
        )

    variant = "macOS" if document.target == "macos_dynamic" else "iOS"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_config = ExportConfig(
        hf_model_id=model_id,
        variant=variant,
        max_context_length=document.metadata["maxContextLength"],
        compute_precision="float16",
        compression="none",
        output_dir=str(output_dir),
        output_name=document.metadata["name"],
        overwrite=overwrite,
    )
    try:
        result = apple_export(export_config)
    except Exception as error:
        raise ExportError("coreai_export_failed", str(error)) from error
    bundle_path = Path(result)
    validate_language_bundle(
        bundle_path,
        expected_name=document.metadata["name"],
        expected_model_id=model_id,
        expected_vocab_size=document.metadata["vocabSize"],
        expected_max_context_length=document.metadata["maxContextLength"],
    )
    return bundle_path
