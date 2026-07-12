"""Low-level Core AI program export for custom stateful PyTorch modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import ExportError


def export_torch_module(
    module: Any,
    reference_inputs: Mapping[str, Any],
    output_path: Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    state_names: Sequence[str] = (),
    dynamic_shapes: Mapping[str, Any] | None = None,
) -> Path:
    """Export a PyTorch module as an optimized Core AI source asset.

    This is the escape hatch for model families that are not in Apple's
    high-level registry. State names are explicit and must be mutated by the
    module during forward execution so Core AI can surface them as states.
    """
    try:
        import coreai_torch
        import torch
        from coreai_torch import TorchConverter
    except ImportError as error:
        raise ExportError(
            "missing_coreai_dependencies",
            "Install the pinned Python dependencies from python/pyproject.toml",
        ) from error

    if not output_path.name.endswith(".aimodel"):
        raise ExportError("invalid_output", "output_path must end with .aimodel")
    if not input_names:
        raise ExportError("invalid_contract", "at least one input is required")
    missing_inputs = [name for name in input_names if name not in reference_inputs]
    if missing_inputs:
        raise ExportError("invalid_contract", f"missing reference inputs: {missing_inputs}")
    overlap = set(input_names) & set(state_names)
    if overlap:
        raise ExportError("invalid_contract", f"inputs and states overlap: {sorted(overlap)}")

    def export_fn(model: Any) -> Any:
        with torch.no_grad():
            exported = torch.export.export(
                model,
                args=(),
                kwargs=dict(reference_inputs),
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
        return exported.run_decompositions(coreai_torch.get_decomp_table())

    try:
        module.eval()
        converter = TorchConverter()
        converter.add_pytorch_module(
            module,
            export_fn=export_fn,
            input_names=tuple(input_names),
            output_names=tuple(output_names),
            state_names=tuple(state_names),
        )
        program = converter.to_coreai()
        program.optimize()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        program.save_asset(output_path)
    except Exception as error:
        raise ExportError("coreai_program_export_failed", str(error)) from error
    return output_path
