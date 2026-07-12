"""Export Hugging Face LFM2 models through the low-level Core AI path."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .errors import ExportError
from .program import export_torch_module


SUPPORTED_MODEL_TYPES = {"lfm2", "lfm2_moe"}


def export_lfm2_model(
    model_id: str,
    output_dir: Path,
    *,
    output_name: str | None = None,
    max_context_length: int | None = None,
    compute_precision: str = "float16",
    overwrite: bool = False,
) -> Path:
    """Export an LFM2 model as a dynamic, stateless Core AI bundle.

    The exported function recomputes the supplied sequence on every call. This
    is the correctness-first LFM2 route while Core AI state mutation for the
    hybrid convolution and attention cache is still a separate contract.
    """
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ExportError(
            "missing_coreai_dependencies",
            "Install the pinned Python dependencies from python/pyproject.toml",
        ) from error

    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception as error:
        raise ExportError("hf_config_load_failed", f"Could not load {model_id}") from error

    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ExportError(
            "model_type_mismatch",
            f"LFM2 exporter requires model_type in {sorted(SUPPORTED_MODEL_TYPES)}, got {model_type!r}",
        )

    native_context_length = getattr(config, "max_position_embeddings", None)
    context_length = (
        max_context_length
        if max_context_length is not None
        else native_context_length
    )
    if context_length is None or context_length <= 0:
        raise ExportError("invalid_context_length", "LFM2 max context length must be positive")
    if native_context_length is not None and context_length > native_context_length:
        raise ExportError(
            "invalid_context_length",
            f"max_context_length {context_length} exceeds model maximum {native_context_length}",
        )

    dtype = _resolve_dtype(torch, compute_precision)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as error:
        raise ExportError("hf_model_load_failed", f"Could not load {model_id}") from error

    if model_type == "lfm2_moe":
        _replace_moe_experts_with_dense_routing(model, torch)

    class StatelessLFM2(torch.nn.Module):
        def __init__(self, source_model: Any) -> None:
            super().__init__()
            self.model = source_model

        def forward(self, input_ids: Any, position_ids: Any) -> Any:
            return self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
                return_dict=False,
            )[0]

    reference_sequence_length = min(2, context_length)
    sequence_length = torch.export.Dim(
        "sequence_length",
        min=1,
        max=context_length,
    )
    reference_inputs = {
        "input_ids": torch.ones((1, reference_sequence_length), dtype=torch.int64),
        "position_ids": torch.arange(reference_sequence_length, dtype=torch.int64).unsqueeze(0),
    }
    dynamic_shapes = {
        "input_ids": {1: sequence_length},
        "position_ids": {1: sequence_length},
    }

    name = output_name or _default_output_name(model_id)
    bundle_path = output_dir / name
    asset_path = bundle_path / f"{name}.aimodel"
    if asset_path.exists():
        if not overwrite:
            raise ExportError("output_exists", f"Output already exists: {asset_path}")
        shutil.rmtree(asset_path)
    bundle_path.mkdir(parents=True, exist_ok=True)

    try:
        export_torch_module(
            StatelessLFM2(model),
            reference_inputs,
            asset_path,
            input_names=["input_ids", "position_ids"],
            output_names=["logits"],
            dynamic_shapes=dynamic_shapes,
        )
        tokenizer.save_pretrained(bundle_path / "tokenizer")
    except ExportError:
        raise
    except Exception as error:
        raise ExportError("lfm2_export_failed", str(error)) from error
    return bundle_path


def _resolve_dtype(torch: Any, value: str) -> Any:
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[value]
    except KeyError as error:
        raise ExportError(
            "invalid_compute_precision",
            "compute_precision must be float16, bfloat16, or float32",
        ) from error


def _replace_moe_experts_with_dense_routing(model: Any, torch: Any) -> None:
    """Replace dynamic expert enumeration with Core AI-lowerable dense routing.

    The Hugging Face reference uses ``one_hot``/``where``/``index_add`` to
    visit only active experts. That path introduces ``histc`` and
    ``empty_permuted`` into ``torch.export`` on the current beta. Evaluating
    every expert and applying the same top-k weights is semantically equivalent
    and keeps the graph within Core AI's supported tensor operators.
    """

    class DenseExperts(torch.nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.num_experts = source.num_experts
            self.hidden_dim = source.hidden_dim
            self.gate_up_proj = source.gate_up_proj
            self.down_proj = source.down_proj

        def forward(self, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
            token_count = hidden_states.shape[0]
            expert_inputs = hidden_states.unsqueeze(0).expand(
                self.num_experts, token_count, self.hidden_dim
            )
            gate_up = torch.bmm(expert_inputs, self.gate_up_proj.transpose(1, 2))
            gate, up = gate_up.chunk(2, dim=-1)
            expert_hidden = torch.nn.functional.silu(gate) * up
            expert_outputs = torch.bmm(expert_hidden, self.down_proj.transpose(1, 2))

            expert_weights = []
            for expert_index in range(self.num_experts):
                selected = (top_k_index == expert_index).to(top_k_weights.dtype)
                expert_weights.append((selected * top_k_weights).sum(dim=-1))
            weights = torch.stack(expert_weights, dim=0)
            return (expert_outputs * weights.unsqueeze(-1)).sum(dim=0).to(hidden_states.dtype)

    for module in list(model.modules()):
        if module.__class__.__name__ == "Lfm2MoeSparseMoeBlock":
            module.experts = DenseExperts(module.experts)


def _default_output_name(model_id: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id.rsplit("/", 1)[-1]).strip("_")
    if not name:
        raise ExportError("invalid_output", "model_id does not provide an output name")
    return name
