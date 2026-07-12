"""Bridge from validated swift-lm documents to Apple's Core AI exporter."""

from __future__ import annotations

from pathlib import Path

from .document import ExportDocument
from .errors import ExportError

LOW_LEVEL_ONLY_MODEL_TYPES = {"lfm2", "lfm2_moe"}


def validate_document(path: Path) -> ExportDocument:
    """Load and validate a Swift-produced export document."""
    return ExportDocument.load(path)


def export_model(
    document_path: Path,
    model_id: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export a validated document using Apple's official Core AI model pipeline.

    The Swift document is a required preflight contract. The exporter refuses
    to infer a different model family or platform from the Hugging Face bundle.
    """
    document = validate_document(document_path)
    if document.model_type in LOW_LEVEL_ONLY_MODEL_TYPES:
        raise ExportError(
            "low_level_export_required",
            f"{document.model_type} requires export_torch_module because its hybrid state layout is not in Apple's high-level registry",
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
    return Path(result)
