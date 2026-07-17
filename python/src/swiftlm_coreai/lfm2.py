"""Export Hugging Face LFM2 models through the low-level Core AI path."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .bundle import validate_language_bundle, write_language_bundle_metadata
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
    stateful: bool = False,
) -> Path:
    """Export an LFM2 model as a Core AI bundle.

    The default route recomputes the supplied sequence on every call. The
    stateful route exposes attention and short-convolution caches as mutable
    Core AI states and supports repeated single-token calls. Each call receives
    one input token and the complete prefix position range.
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

    name = output_name or _default_output_name(model_id)
    bundle_path = output_dir / name
    asset_path = bundle_path / f"{name}.aimodel"
    if asset_path.exists():
        if not overwrite:
            raise ExportError("output_exists", f"Output already exists: {asset_path}")
        shutil.rmtree(asset_path)
    bundle_path.mkdir(parents=True, exist_ok=True)

    if stateful:
        try:
            _export_stateful_program(
                model,
                config,
                torch,
                asset_path,
                max_context_length=context_length,
            )
            tokenizer.save_pretrained(bundle_path / "tokenizer")
        except ExportError:
            raise
        except Exception as error:
            raise ExportError("lfm2_stateful_export_failed", str(error)) from error
        _finalize_bundle(
            bundle_path,
            model_id=model_id,
            name=name,
            vocab_size=config.vocab_size,
            max_context_length=context_length,
        )
        return bundle_path

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
    _finalize_bundle(
        bundle_path,
        model_id=model_id,
        name=name,
        vocab_size=config.vocab_size,
        max_context_length=context_length,
    )
    return bundle_path


def _finalize_bundle(
    bundle_path: Path,
    *,
    model_id: str,
    name: str,
    vocab_size: int,
    max_context_length: int,
) -> None:
    """Add and verify the Apple language bundle contract after asset export."""
    write_language_bundle_metadata(
        bundle_path,
        name=name,
        model_id=model_id,
        vocab_size=vocab_size,
        max_context_length=max_context_length,
    )
    validate_language_bundle(
        bundle_path,
        expected_name=name,
        expected_model_id=model_id,
        expected_vocab_size=vocab_size,
        expected_max_context_length=max_context_length,
    )


def export_lfm2_stateful_model(
    model_id: str,
    output_dir: Path,
    *,
    output_name: str | None = None,
    max_context_length: int | None = None,
    compute_precision: str = "float16",
    overwrite: bool = False,
) -> Path:
    """Export an LFM2 model with mutable KV and convolution Core AI states."""
    return export_lfm2_model(
        model_id,
        output_dir,
        output_name=output_name,
        max_context_length=max_context_length,
        compute_precision=compute_precision,
        overwrite=overwrite,
        stateful=True,
    )


def _export_stateful_program(
    source_model: Any,
    config: Any,
    torch: Any,
    asset_path: Path,
    *,
    max_context_length: int,
) -> None:
    """Export the hybrid model with explicit mutable Core AI cache tensors."""
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.primitives.macos.cache import KVCache

    model = _make_stateful_model(source_model, torch, KVCache)
    layer_count = config.num_hidden_layers
    key_value_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    trace_cache_length = min(max_context_length, 4096)
    sequence_length = torch.export.Dim("sequence_length", min=1, max=max_context_length)
    cache_shapes = {}
    if trace_cache_length < max_context_length:
        cache_shapes = {
            3: torch.export.Dim(
                "cache_length",
                min=trace_cache_length,
                max=max_context_length,
            )
        }
    reference_inputs = {
        "input_ids": torch.ones((1, 1), dtype=torch.int64),
        "position_ids": torch.arange(2, dtype=torch.int64).unsqueeze(0),
        "k_cache": torch.zeros(
            layer_count,
            1,
            key_value_heads,
            trace_cache_length,
            head_dim,
            dtype=next(source_model.parameters()).dtype,
        ),
        "v_cache": torch.zeros(
            layer_count,
            1,
            key_value_heads,
            trace_cache_length,
            head_dim,
            dtype=next(source_model.parameters()).dtype,
        ),
        "conv_cache": torch.zeros(
            layer_count,
            1,
            config.hidden_size,
            config.conv_L_cache,
            dtype=next(source_model.parameters()).dtype,
        ),
    }
    dynamic_shapes = {
        "input_ids": {},
        "position_ids": {1: sequence_length},
        "k_cache": cache_shapes,
        "v_cache": cache_shapes,
        "conv_cache": {},
    }
    program = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids"),
        output_names=("logits",),
        state_names=("keyCache", "valueCache", "convCache"),
    )
    program.optimize()
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(asset_path)


def _make_stateful_model(source_model: Any, torch: Any, kv_cache_type: Any) -> Any:
    """Adapt the Hugging Face hybrid layers to explicit Core AI cache objects."""
    from coreai_models.primitives._ops import mutable_slice_update
    from coreai_models.primitives.macos.sdpa import SDPA

    class StatefulCache:
        def __init__(self, k_cache: Any, v_cache: Any, conv_cache: Any, sequence_length: Any) -> None:
            self.kv_cache = kv_cache_type(k_cache, v_cache)
            self.conv_cache = conv_cache
            self.sequence_length = sequence_length

        def update(self, key_states: Any, value_states: Any, layer_idx: int, *args: Any, **kwargs: Any):
            query_length = key_states.shape[-2]
            offset = self.sequence_length - query_length
            return self.kv_cache.update_and_fetch(
                layer_idx,
                offset,
                key_states,
                value_states,
                seq_len=self.sequence_length,
                query_len=query_length,
            )

        def get_conv_state(self, layer_idx: int) -> Any:
            return self.conv_cache.narrow(0, layer_idx, 1).squeeze(0)

        def update_conv_state(self, state: Any, layer_idx: int) -> Any:
            layer = self.conv_cache.narrow(0, layer_idx, 1).squeeze(0)
            updated = torch.cat((layer[..., 1:], state[..., -1:]), dim=-1)
            layer_index = torch.tensor((layer_idx,), dtype=torch.int32, device=state.device)
            layer_index_end = torch.tensor((layer_idx + 1,), dtype=torch.int32, device=state.device)
            begin = torch.cat(
                [
                    layer_index,
                    torch.zeros(3, dtype=torch.int32, device=state.device),
                ]
            )
            end = torch.cat(
                [
                    layer_index_end,
                    torch.tensor(updated.shape, dtype=torch.int32, device=state.device),
                ]
            )
            mutable_slice_update(
                x=self.conv_cache,
                update=updated.unsqueeze(0),
                begin=begin,
                end=end,
            )
            return updated

        def has_previous_state(self, layer_idx: int | None = None) -> bool:
            return True

    def rotate_half(value: Any) -> Any:
        midpoint = value.shape[-1] // 2
        return torch.cat((-value[..., midpoint:], value[..., :midpoint]), dim=-1)

    def apply_rotary(query: Any, key: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (
            (query * cos) + (rotate_half(query) * sin),
            (key * cos) + (rotate_half(key) * sin),
        )

    def repeat_kv(value: Any, repetitions: int) -> Any:
        if repetitions == 1:
            return value
        batch, heads, sequence, dimension = value.shape
        return value[:, :, None, :, :].expand(
            batch, heads, repetitions, sequence, dimension
        ).reshape(batch, heads * repetitions, sequence, dimension)

    class StatefulAttention(torch.nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.layer_idx = source.layer_idx
            self.head_dim = source.head_dim
            self.num_key_value_groups = source.num_key_value_groups
            self.scaling = source.scaling
            self.q_proj = source.q_proj
            self.k_proj = source.k_proj
            self.v_proj = source.v_proj
            self.out_proj = source.out_proj
            self.q_layernorm = source.q_layernorm
            self.k_layernorm = source.k_layernorm
            self.sdpa = SDPA(scale=self.scaling, is_causal=True)

        def forward(
            self,
            hidden_states: Any,
            position_embeddings: tuple[Any, Any],
            attention_mask: Any = None,
            past_key_values: Any = None,
            **kwargs: Any,
        ) -> tuple[Any, None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query = self.q_layernorm(self.q_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
            key = self.k_layernorm(self.k_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
            value = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_length = query.shape[-2]
            offset = position_embeddings[0].shape[-2] - query_length
            cos = cos.narrow(-2, offset, query_length)
            sin = sin.narrow(-2, offset, query_length)
            query, key = apply_rotary(query, key, cos, sin)
            key, value = past_key_values.update(key, value, self.layer_idx)
            key = repeat_kv(key, self.num_key_value_groups)
            value = repeat_kv(value, self.num_key_value_groups)
            output = self.sdpa(query, key, value)
            output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            return self.out_proj(output), None

    class StatefulShortConv(torch.nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.layer_idx = source.layer_idx
            self.in_proj = source.in_proj
            self.out_proj = source.out_proj
            self.conv = source.conv
            self.cache_length = source.L_cache

        def forward(
            self,
            hidden_states: Any,
            past_key_values: Any = None,
            attention_mask: Any = None,
        ) -> Any:
            if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
                hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
            bcx = self.in_proj(hidden_states).transpose(-1, -2)
            gate, candidate, value = bcx.chunk(3, dim=-2)
            mixed = gate * value
            previous = past_key_values.get_conv_state(self.layer_idx)
            full = torch.cat((previous, mixed), dim=-1)
            convolution = torch.nn.functional.conv1d(
                full,
                self.conv.weight,
                self.conv.bias,
                groups=full.shape[1],
            )
            convolution = convolution[..., 1:]
            past_key_values.update_conv_state(full[..., -self.cache_length :], self.layer_idx)
            output = candidate * convolution
            output = self.out_proj(output.transpose(-1, -2).contiguous())
            return output

    class StatefulLFM2(torch.nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.model = source.model
            self.lm_head = source.lm_head
            for layer in self.model.layers:
                if layer.is_attention_layer:
                    layer.self_attn = StatefulAttention(layer.self_attn)
                else:
                    layer.conv = StatefulShortConv(layer.conv)
            self.rotary_embedding = getattr(self.model, "rotary_emb", None)
            if self.rotary_embedding is None:
                self.rotary_embedding = self.model.pos_emb

        def forward(
            self,
            input_ids: Any,
            position_ids: Any,
            k_cache: Any,
            v_cache: Any,
            conv_cache: Any,
        ) -> Any:
            hidden_states = self.model.embed_tokens(input_ids)
            position_embeddings = self.rotary_embedding(hidden_states, position_ids=position_ids)
            cache = StatefulCache(
                k_cache,
                v_cache,
                conv_cache,
                position_ids.shape[-1],
            )
            for layer in self.model.layers:
                hidden_states = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=cache,
                )
            hidden_states = self.model.embedding_norm(hidden_states)
            return self.lm_head(hidden_states)

    return StatefulLFM2(source_model)


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
