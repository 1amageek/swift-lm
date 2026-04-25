#!/usr/bin/env python3
"""Dump LFM2 reference intermediate tensors for Metal comparison.

Runs LFM2.5-1.2B-Thinking on CPU through HuggingFace transformers,
captures intermediate values at key points, and saves to safetensors.
The Swift ReferenceComparisonTests loads this file and compares
against Metal inference output to identify divergence points.

Usage:
    python3 scripts/hf/dump_lfm2_reference.py --output Tests/MetalCompilerTests/lfm2_reference.safetensors
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb


# Fixed input tokens (matches existing DecodeTests.swift)
DEFAULT_INPUT_TOKENS = [1, 1, 6, 6423, 708]
NUM_DECODE_STEPS = 3
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"


def load_model_from(model_id):
    """Load the LFM2 model and tokenizer in bfloat16 on CPU."""
    print(f"Loading model: {model_id}")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.eval()
    print(f"Model loaded: {config.num_hidden_layers} layers, "
          f"hidden={config.hidden_size}, vocab={config.vocab_size}")
    return model, config


def identify_conv_layers(config):
    """Return indices of conv layers from layer_types."""
    layer_types = config.layer_types
    conv_indices = [i for i, t in enumerate(layer_types) if t == "conv"]
    attn_indices = [i for i, t in enumerate(layer_types) if t != "conv"]
    print(f"Conv layers: {conv_indices} ({len(conv_indices)} total)")
    print(f"Attention layers: {attn_indices} ({len(attn_indices)} total)")
    return conv_indices


def conv_state_from_cache_layer(past_key_values, layer_index):
    """Read a conv-state tensor from either the old or new transformers cache API."""
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_index]
        if hasattr(layer, "conv_states") and layer.conv_states is not None:
            return layer.conv_states[0].detach().clone().cpu()
    if hasattr(past_key_values, "conv_cache"):
        return past_key_values.conv_cache[layer_index][0].detach().clone().cpu()
    raise AttributeError(f"Conv cache not found for layer {layer_index}")


def kv_cache_from_cache_layer(past_key_values, layer_index):
    """Read key/value tensors from either the old or new transformers cache API."""
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_index]
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            return (
                layer.keys[0].detach().clone().cpu(),
                layer.values[0].detach().clone().cpu(),
            )
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return (
            past_key_values.key_cache[layer_index][0].detach().clone().cpu(),
            past_key_values.value_cache[layer_index][0].detach().clone().cpu(),
        )
    raise AttributeError(f"KV cache not found for layer {layer_index}")


def run_prefill_with_hooks(model, config, input_tokens):
    """Run prefill and capture intermediate tensors via forward hooks."""
    captures = {}
    handles = []
    conv_layer_indices = identify_conv_layers(config)
    attention_layer_indices = [i for i in range(config.num_hidden_layers) if i not in conv_layer_indices]

    # Hook: embedding output
    def embed_hook(module, input, output):
        captures["ref.prefill.embedding"] = output[0].detach().cpu()
    handles.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    # Hook: each decoder layer (captures after_mlp = full layer output)
    for i in range(config.num_hidden_layers):
        def layer_hook(module, input, output, idx=i):
            hidden = output[0] if isinstance(output, tuple) else output
            captures[f"ref.prefill.layer_{idx}.after_mlp"] = hidden.detach().cpu()
        handles.append(model.model.layers[i].register_forward_hook(layer_hook))

    # Hook: conv/attention sub-modules (captures after_op = before MLP residual)
    for i in range(config.num_hidden_layers):
        layer = model.model.layers[i]
        if i in conv_layer_indices:
            def conv_hook(module, input, output, idx=i):
                captures[f"ref.prefill.layer_{idx}.after_op"] = output.detach().cpu()
            handles.append(layer.conv.register_forward_hook(conv_hook))
        else:
            def attn_hook(module, input, output, idx=i):
                attn_out = output[0] if isinstance(output, tuple) else output
                captures[f"ref.prefill.layer_{idx}.after_op"] = attn_out.detach().cpu()
            handles.append(layer.self_attn.register_forward_hook(attn_hook))

        def mlp_hook(module, input, output, idx=i):
            mlp_out = output[0] if isinstance(output, tuple) else output
            captures[f"ref.prefill.layer_{idx}.mlp_out"] = mlp_out.detach().cpu()
        handles.append(layer.feed_forward.register_forward_hook(mlp_hook))

        def ffn_norm_hook(module, input, output, idx=i):
            captures[f"ref.prefill.layer_{idx}.after_op_residual"] = input[0].detach().cpu()
            captures[f"ref.prefill.layer_{idx}.ffn_norm"] = output.detach().cpu()
        handles.append(layer.ffn_norm.register_forward_hook(ffn_norm_hook))

    # Hook: final norm
    def norm_hook(module, input, output):
        captures["ref.prefill.final_hidden"] = output.detach().cpu()
    handles.append(model.model.embedding_norm.register_forward_hook(norm_hook))

    # Run prefill
    input_ids = torch.tensor([input_tokens], dtype=torch.long)
    print(f"Running prefill with {len(input_tokens)} tokens: {input_tokens}")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

    # Capture logits (last position only)
    logits_last = outputs.logits[0, -1, :].detach().cpu()
    captures["ref.prefill.logits_last"] = logits_last

    # Capture conv_state from cache
    past = outputs.past_key_values
    conv_idx = 0
    kv_idx = 0
    for i in range(config.num_hidden_layers):
        if i in conv_layer_indices:
            # Python layout: [batch=1, hidden_size, L_cache]
            # Swift layout:  [L_cache, hidden_size] (temporal-first)
            conv_state = conv_state_from_cache_layer(past, i)  # [hidden, L_cache]
            conv_state_swift = conv_state.permute(1, 0).contiguous()  # [L_cache, hidden]
            captures[f"ref.prefill.conv_state.{conv_idx}"] = conv_state_swift
            conv_idx += 1
        elif i in attention_layer_indices:
            # Python layout after batch squeeze: [kv_heads, seq, head_dim].
            key_cache, value_cache = kv_cache_from_cache_layer(past, i)
            captures[f"ref.prefill.kv_cache.{kv_idx}.keys"] = key_cache.contiguous()
            captures[f"ref.prefill.kv_cache.{kv_idx}.values"] = value_cache.contiguous()
            kv_idx += 1

    # Clean up hooks
    for h in handles:
        h.remove()

    # Report
    argmax = logits_last.argmax().item()
    print(f"Prefill logits: argmax={argmax}, max={logits_last.max().item():.4f}")
    print(f"Conv states captured: {conv_idx}")
    print(f"KV cache layers captured: {kv_idx}")
    print(f"Captures: {len(captures)} tensors")

    return captures, outputs.past_key_values, argmax


def run_decode_steps(model, config, past_key_values, first_token, num_steps, input_tokens):
    """Run decode steps and capture intermediate tensors."""
    captures = {}
    conv_layer_indices = identify_conv_layers(config)
    current_token = first_token
    past = past_key_values
    position = len(input_tokens)

    for step in range(num_steps):
        handles = []
        q_norm_outputs = {}
        k_norm_outputs = {}

        # Hook: each decoder layer (captures after_mlp = full layer output)
        for i in range(config.num_hidden_layers):
            def layer_hook(module, input, output, idx=i, s=step):
                hidden = output[0] if isinstance(output, tuple) else output
                captures[f"ref.decode_{s}.layer_{idx}.after_mlp"] = hidden[0, 0].detach().cpu()
            handles.append(model.model.layers[i].register_forward_hook(layer_hook))

        # Hook: conv/attention sub-modules (captures after_op = before MLP residual)
        for i in range(config.num_hidden_layers):
            layer = model.model.layers[i]
            if i in conv_layer_indices:
                def conv_hook(module, input, output, idx=i, s=step):
                    captures[f"ref.decode_{s}.layer_{idx}.after_op"] = output[0, 0].detach().cpu()
                handles.append(layer.conv.register_forward_hook(conv_hook))
            else:
                def attn_hook(module, input, output, idx=i, s=step):
                    attn_out = output[0] if isinstance(output, tuple) else output
                    captures[f"ref.decode_{s}.layer_{idx}.after_op"] = attn_out[0, 0].detach().cpu()
                handles.append(layer.self_attn.register_forward_hook(attn_hook))

                def o_proj_pre_hook(module, input, idx=i, s=step):
                    captures[f"ref.decode_{s}.layer_{idx}.attn_pre_o_proj"] = input[0][0, 0].detach().cpu()
                handles.append(layer.self_attn.out_proj.register_forward_pre_hook(o_proj_pre_hook))

            def mlp_hook(module, input, output, idx=i, s=step):
                mlp_out = output[0] if isinstance(output, tuple) else output
                captures[f"ref.decode_{s}.layer_{idx}.mlp_out"] = mlp_out[0, 0].detach().cpu()
            handles.append(layer.feed_forward.register_forward_hook(mlp_hook))

            def ffn_norm_hook(module, input, output, idx=i, s=step):
                captures[f"ref.decode_{s}.layer_{idx}.after_op_residual"] = input[0][0, 0].detach().cpu()
                captures[f"ref.decode_{s}.layer_{idx}.ffn_norm"] = output[0, 0].detach().cpu()
            handles.append(layer.ffn_norm.register_forward_hook(ffn_norm_hook))

            if i not in conv_layer_indices:
                def q_norm_hook(module, input, output, idx=i):
                    q_norm_outputs[idx] = output.detach().clone()
                handles.append(layer.self_attn.q_layernorm.register_forward_hook(q_norm_hook))

                def k_norm_hook(module, input, output, idx=i):
                    k_norm_outputs[idx] = output.detach().clone()
                handles.append(layer.self_attn.k_layernorm.register_forward_hook(k_norm_hook))

        # Hook: final norm
        def norm_hook(module, input, output, s=step):
            captures[f"ref.decode_{s}.final_hidden"] = output[0, 0].detach().cpu()
        handles.append(model.model.embedding_norm.register_forward_hook(norm_hook))

        # Run single decode step
        input_ids = torch.tensor([[current_token]], dtype=torch.long)
        cache_position = torch.tensor([position], dtype=torch.long)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                cache_position=cache_position,
            )

        # Capture logits
        logits = outputs.logits[0, 0, :].detach().cpu()
        captures[f"ref.decode_{step}.logits"] = logits
        captures[f"ref.decode_{step}.input_token"] = torch.tensor(
            [current_token], dtype=torch.int32)

        position_ids = torch.tensor([[position]], dtype=torch.long)
        for i in range(config.num_hidden_layers):
            if i in conv_layer_indices:
                continue
            q_norm = q_norm_outputs[i]
            k_norm = k_norm_outputs[i]
            captures[f"ref.decode_{step}.layer_{i}.q_norm"] = q_norm[0, 0].detach().cpu()
            captures[f"ref.decode_{step}.layer_{i}.k_norm"] = k_norm[0, 0].detach().cpu()

            cos, sin = model.model.rotary_emb(q_norm, position_ids=position_ids)
            q_rope, k_rope = apply_rotary_pos_emb(
                q_norm.transpose(1, 2),
                k_norm.transpose(1, 2),
                cos,
                sin,
            )
            captures[f"ref.decode_{step}.layer_{i}.q_rope"] = q_rope[0, :, 0, :].detach().cpu()
            captures[f"ref.decode_{step}.layer_{i}.k_rope"] = k_rope[0, :, 0, :].detach().cpu()

        # Capture conv_state
        past = outputs.past_key_values
        conv_idx = 0
        kv_idx = 0
        for i in range(config.num_hidden_layers):
            if i in conv_layer_indices:
                conv_state = conv_state_from_cache_layer(past, i)
                conv_state_swift = conv_state.permute(1, 0).contiguous()
                captures[f"ref.decode_{step}.conv_state.{conv_idx}"] = conv_state_swift
                conv_idx += 1
            else:
                key_cache, value_cache = kv_cache_from_cache_layer(past, i)
                captures[f"ref.decode_{step}.kv_cache.{kv_idx}.keys"] = key_cache.contiguous()
                captures[f"ref.decode_{step}.kv_cache.{kv_idx}.values"] = value_cache.contiguous()
                captures[f"ref.decode_{step}.kv_cache.{kv_idx}.current_key"] = (
                    key_cache[:, position, :].contiguous()
                )
                captures[f"ref.decode_{step}.kv_cache.{kv_idx}.current_value"] = (
                    value_cache[:, position, :].contiguous()
                )
                kv_idx += 1

        # Clean up hooks
        for h in handles:
            h.remove()

        next_token = logits.argmax().item()
        print(f"Decode step {step}: input={current_token} → output={next_token} "
              f"(logits max={logits.max().item():.4f}@{logits.argmax().item()})")

        current_token = next_token
        position += 1

    return captures


def save_reference(all_captures, output_path, input_tokens):
    """Save all captured tensors to safetensors."""
    # Add input tokens
    all_captures["ref.input_tokens"] = torch.tensor(
        input_tokens, dtype=torch.int32)

    # Ensure all tensors are contiguous
    for key in all_captures:
        if isinstance(all_captures[key], torch.Tensor):
            all_captures[key] = all_captures[key].contiguous()

    # Squeeze batch dimension where present
    for key in list(all_captures.keys()):
        t = all_captures[key]
        if t.dim() >= 2 and t.shape[0] == 1:
            # Prefill: [1, seqLen, hidden] → [seqLen, hidden]
            # Decode: [1, 1, hidden] → [hidden] (squeeze twice if needed)
            all_captures[key] = t.squeeze(0)

    print(f"\nSaving {len(all_captures)} tensors to {output_path}")
    for key, tensor in sorted(all_captures.items()):
        print(f"  {key}: {list(tensor.shape)} {tensor.dtype}")

    save_file(all_captures, str(output_path))
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Dump LFM2 reference tensors")
    parser.add_argument(
        "--output", type=str,
        default="TestData/lfm2_reference.safetensors",
        help="Output safetensors file path",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_ID,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--input-tokens",
        type=str,
        default=json.dumps(DEFAULT_INPUT_TOKENS),
        help="JSON array of input token IDs to run through prefill/decode",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=NUM_DECODE_STEPS,
        help="Number of greedy decode steps to capture after prefill",
    )
    args = parser.parse_args()

    model_id = args.model
    input_tokens = json.loads(args.input_tokens)
    if not isinstance(input_tokens, list) or not all(isinstance(token, int) for token in input_tokens):
        raise ValueError("--input-tokens must decode to a JSON array of integers")

    # Load model
    model, config = load_model_from(model_id)

    # Run prefill with hooks
    prefill_captures, past_key_values, first_token = run_prefill_with_hooks(
        model, config, input_tokens)

    # Run decode steps
    decode_captures = run_decode_steps(
        model, config, past_key_values, first_token, args.decode_steps, input_tokens)

    # Merge and save
    all_captures = {**prefill_captures, **decode_captures}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_reference(all_captures, output_path, input_tokens)

    print("\nDone. Run Swift comparison test:")
    print(f"  xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' "
          f"-only-testing 'MetalCompilerTests/ReferenceComparisonTests'")


if __name__ == "__main__":
    main()
