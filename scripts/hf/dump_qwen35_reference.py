#!/usr/bin/env python3
"""Dump Qwen3.5 HuggingFace reference tensors for Metal correctness tests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from transformers import __version__ as TRANSFORMERS_VERSION


DEFAULT_INPUT_TOKENS = [
    248045,
    846,
    198,
    3710,
    369,
    279,
    6511,
    314,
    6124,
    30,
    248046,
    198,
    248045,
    74455,
    198,
    248068,
    271,
    248069,
    271,
]

REFERENCE_CASES = [
    DEFAULT_INPUT_TOKENS,
    DEFAULT_INPUT_TOKENS[:8],
]


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    linear_block_ordinals = parse_ordinal_list(args.linear_block_ordinals)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=not args.allow_download,
    )
    model.eval()

    captures: dict[str, torch.Tensor] = {}
    current_phase = {"name": "ref.case_0.prefill"}
    hooks = register_hooks(model, captures, current_phase, linear_block_ordinals)

    captures["ref.meta.schema_version"] = torch.tensor([5], dtype=torch.int32)
    captures["ref.meta.case_count"] = torch.tensor([len(REFERENCE_CASES)], dtype=torch.int32)
    captures["ref.meta.decode_steps"] = torch.tensor([args.decode_steps], dtype=torch.int32)
    captures["ref.meta.linear_block_ordinals"] = torch.tensor(
        sorted(linear_block_ordinals),
        dtype=torch.int32,
    )
    captures["ref.meta.config_sha256"] = config_sha256_tensor(model_path, model)
    captures["ref.meta.torch_version_utf8"] = utf8_tensor(torch.__version__)
    captures["ref.meta.transformers_version_utf8"] = utf8_tensor(TRANSFORMERS_VERSION)
    captures["ref.meta.fast_backend_available"] = torch.tensor(
        [1 if fast_backend_available() else 0],
        dtype=torch.int32,
    )

    with torch.inference_mode():
        for case_index, tokens in enumerate(REFERENCE_CASES):
            capture_case(case_index, tokens, model, captures, current_phase, args.decode_steps)

    for hook in hooks:
        hook.remove()

    save_file(captures, str(output_path))
    print(f"Wrote {len(captures)} tensors to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("SWIFTLM_QWEN35_REFERENCE_MODEL"),
        help="Path or HuggingFace id for Qwen/Qwen3.5-0.8B.",
    )
    parser.add_argument(
        "--output",
        default="TestData/qwen35_reference.safetensors",
        help="Output safetensors path.",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=2,
        help="Number of greedy decode steps to dump after prefill.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download the model when it is not cached.",
    )
    parser.add_argument(
        "--linear-block-ordinals",
        default="0",
        help="Comma-separated linear-attention ordinals to dump with block-boundary tensors.",
    )
    return parser.parse_args()


def parse_ordinal_list(value: str) -> set[int]:
    ordinals: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ordinal = int(item)
        if ordinal < 0:
            raise ValueError(f"linear block ordinal must be non-negative: {ordinal}")
        ordinals.add(ordinal)
    return ordinals


def resolve_model_path(model: str | None) -> Path | str:
    if model:
        expanded = Path(model).expanduser()
        return expanded if expanded.exists() else model

    cache_root = Path("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots").expanduser()
    if cache_root.exists():
        for snapshot in sorted(cache_root.iterdir()):
            if (snapshot / "config.json").exists():
                return snapshot

    return "Qwen/Qwen3.5-0.8B"


def config_sha256_tensor(model_path: Path | str, model: Any) -> torch.Tensor:
    if isinstance(model_path, Path):
        config_path = model_path / "config.json"
        if config_path.exists():
            digest = hashlib.sha256(config_path.read_bytes()).digest()
            return torch.tensor(list(digest), dtype=torch.uint8)

    config_json = model.config.to_json_string(use_diff=False).encode("utf-8")
    digest = hashlib.sha256(config_json).digest()
    return torch.tensor(list(digest), dtype=torch.uint8)


def utf8_tensor(value: str) -> torch.Tensor:
    return torch.tensor(list(value.encode("utf-8")), dtype=torch.uint8)


def fast_backend_available() -> bool:
    return (
        importlib.util.find_spec("fla") is not None
        and importlib.util.find_spec("causal_conv1d") is not None
    )


def capture_case(
    case_index: int,
    tokens: list[int],
    model: Any,
    captures: dict[str, torch.Tensor],
    current_phase: dict[str, str],
    decode_steps: int,
) -> None:
    prefix = f"ref.case_{case_index}"
    input_ids = torch.tensor([tokens], dtype=torch.long)
    captures[f"{prefix}.meta.input_tokens"] = input_ids.to(dtype=torch.int32).cpu()
    captures[f"{prefix}.meta.prefill_token_count"] = torch.tensor([len(tokens)], dtype=torch.int32)

    current_phase["name"] = f"{prefix}.prefill"
    outputs = model(input_ids=input_ids, use_cache=True)
    logits_last = outputs.logits[:, -1, :]
    captures[f"{prefix}.prefill.logits_last"] = to_reference_tensor(logits_last)
    next_token = torch.argmax(logits_last, dim=-1).to(dtype=torch.int32)
    captures[f"{prefix}.prefill.next_token"] = next_token.cpu()
    capture_cache(f"{prefix}.prefill", outputs.past_key_values, captures)

    past_key_values = outputs.past_key_values
    decode_token = next_token.to(dtype=torch.long).view(1, 1)
    for step in range(decode_steps):
        current_phase["name"] = f"{prefix}.decode_{step}"
        outputs = model(
            input_ids=decode_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits_last = outputs.logits[:, -1, :]
        captures[f"{prefix}.decode_{step}.logits_last"] = to_reference_tensor(logits_last)
        next_token = torch.argmax(logits_last, dim=-1).to(dtype=torch.int32)
        captures[f"{prefix}.decode_{step}.next_token"] = next_token.cpu()
        capture_cache(f"{prefix}.decode_{step}", outputs.past_key_values, captures)

        past_key_values = outputs.past_key_values
        decode_token = next_token.to(dtype=torch.long).view(1, 1)


def register_hooks(
    model: Any,
    captures: dict[str, torch.Tensor],
    current_phase: dict[str, str],
    linear_block_ordinals: set[int],
) -> list[Any]:
    hooks: list[Any] = []
    base_model = model.model

    hooks.append(
        base_model.embed_tokens.register_forward_hook(
            make_hook(captures, current_phase, "embedding")
        )
    )
    hooks.append(
        base_model.norm.register_forward_hook(
            make_hook(captures, current_phase, "final_hidden")
        )
    )

    linear_ordinal = 0
    for layer_index, layer in enumerate(base_model.layers):
        hooks.append(
            layer.register_forward_hook(
                make_hook(captures, current_phase, f"layer_{layer_index}.after_layer")
            )
        )
        hooks.append(
            layer.input_layernorm.register_forward_hook(
                make_hook(captures, current_phase, f"layer_{layer_index}.input_norm")
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(
                make_hook(captures, current_phase, f"layer_{layer_index}.post_attention_norm")
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                make_hook(captures, current_phase, f"layer_{layer_index}.after_mlp")
            )
        )

        if hasattr(layer, "linear_attn"):
            if linear_ordinal in linear_block_ordinals:
                hooks.extend(
                    register_linear_attention_block_hooks(
                        layer.linear_attn,
                        captures,
                        current_phase,
                        linear_ordinal,
                        layer_index,
                    )
                )
            hooks.append(
                layer.linear_attn.register_forward_hook(
                    make_hook(captures, current_phase, f"layer_{layer_index}.after_op")
                )
            )
            linear_ordinal += 1
        if hasattr(layer, "self_attn"):
            hooks.append(
                layer.self_attn.register_forward_hook(
                    make_hook(captures, current_phase, f"layer_{layer_index}.after_op")
                )
            )

    return hooks


def register_linear_attention_block_hooks(
    block: Any,
    captures: dict[str, torch.Tensor],
    current_phase: dict[str, str],
    linear_ordinal: int,
    layer_index: int,
) -> list[Any]:
    prefix = f"linear_ordinal_{linear_ordinal}.block"
    captures[f"ref.meta.linear_ordinal_{linear_ordinal}.layer_index"] = torch.tensor(
        [layer_index],
        dtype=torch.int32,
    )
    return [
        block.in_proj_qkv.register_forward_hook(
            make_hook(captures, current_phase, f"{prefix}.projected_qkv")
        ),
        block.in_proj_z.register_forward_hook(
            make_hook(captures, current_phase, f"{prefix}.projected_z")
        ),
        block.in_proj_b.register_forward_hook(
            make_hook(captures, current_phase, f"{prefix}.projected_beta")
        ),
        block.in_proj_a.register_forward_hook(
            make_hook(captures, current_phase, f"{prefix}.projected_alpha")
        ),
        block.conv1d.register_forward_hook(
            make_conv_silu_hook(captures, current_phase, f"{prefix}.conv_silu")
        ),
        block.norm.register_forward_hook(
            make_hook(captures, current_phase, f"{prefix}.gated_recurrent_output")
        ),
        block.out_proj.register_forward_hook(
            make_hook(captures, current_phase, f"{prefix}.out_projection")
        ),
    ]


def make_conv_silu_hook(
    captures: dict[str, torch.Tensor],
    current_phase: dict[str, str],
    suffix: str,
) -> Any:
    def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        tensor = first_tensor(output)
        input_tensor = first_tensor(inputs)
        if tensor is None or input_tensor is None:
            return
        seq_len = input_tensor.shape[-1]
        activated = torch.nn.functional.silu(tensor[:, :, :seq_len]).transpose(1, 2)
        captures[f"{current_phase['name']}.{suffix}"] = to_reference_tensor(activated)

    return hook


def make_hook(
    captures: dict[str, torch.Tensor],
    current_phase: dict[str, str],
    suffix: str,
) -> Any:
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = first_tensor(output)
        if tensor is None:
            return
        captures[f"{current_phase['name']}.{suffix}"] = to_reference_tensor(tensor)

    return hook


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def capture_cache(prefix: str, cache: Any, captures: dict[str, torch.Tensor]) -> None:
    if hasattr(cache, "layers"):
        attention_ordinal = 0
        linear_ordinal = 0
        for layer_index, layer in enumerate(cache.layers):
            conv_states = getattr(layer, "conv_states", None)
            recurrent_states = getattr(layer, "recurrent_states", None)
            if conv_states is not None:
                captures[f"{prefix}.conv_state.{layer_index}"] = to_conv_state_tensor(conv_states)
                captures[f"{prefix}.linear_ordinal_{linear_ordinal}.conv_state"] = to_conv_state_tensor(conv_states)
            if recurrent_states is not None:
                captures[f"{prefix}.recurrent_state.{layer_index}"] = to_reference_tensor(
                    recurrent_states.squeeze(0)
                )
                captures[f"{prefix}.linear_ordinal_{linear_ordinal}.recurrent_state"] = (
                    to_reference_tensor(recurrent_states.squeeze(0))
                )
            if conv_states is not None or recurrent_states is not None:
                linear_ordinal += 1

            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                continue
            captures[f"{prefix}.attn_layer_{layer_index}.keys"] = to_reference_tensor(keys.squeeze(0))
            captures[f"{prefix}.attn_layer_{layer_index}.values"] = to_reference_tensor(values.squeeze(0))
            captures[f"{prefix}.attn_ordinal_{attention_ordinal}.keys"] = to_reference_tensor(keys.squeeze(0))
            captures[f"{prefix}.attn_ordinal_{attention_ordinal}.values"] = to_reference_tensor(values.squeeze(0))
            attention_ordinal += 1


def to_conv_state_tensor(state: torch.Tensor) -> torch.Tensor:
    squeezed = state.squeeze(0)
    if squeezed.ndim == 2:
        squeezed = squeezed.transpose(0, 1)
    return to_reference_tensor(squeezed)


def to_reference_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()


if __name__ == "__main__":
    main()
