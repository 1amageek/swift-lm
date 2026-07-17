"""Bridge validated swift-lm documents to Core AI exporters."""

from __future__ import annotations

from pathlib import Path

from .document import ExportDocument
from .ir_export import export_ir_language_model


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
    """Export a validated Swift-authored program without family dispatch."""
    document = validate_document(document_path)
    return export_ir_language_model(
        document,
        model_id,
        output_dir,
        overwrite=overwrite,
    )
