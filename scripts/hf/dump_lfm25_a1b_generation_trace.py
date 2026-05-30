#!/usr/bin/env python3
"""Dump deterministic HuggingFace generation traces for LFM2.5-8B-A1B."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "LiquidAI/LFM2.5-8B-A1B"
DEFAULT_PROMPT = "What is the capital of Japan? Answer with exactly one word."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    return parser.parse_args()


def configure_eager_experts(model):
    if hasattr(model, "config"):
        model.config._experts_implementation = "eager"
    for layer in getattr(model.model, "layers", []):
        feed_forward = getattr(layer, "feed_forward", None)
        experts = getattr(feed_forward, "experts", None)
        if experts is not None and hasattr(experts, "config"):
            experts.config._experts_implementation = "eager"


def main():
    args = parse_args()
    dtype = torch.bfloat16
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        device_map=None,
    )
    configure_eager_experts(model)
    model.eval()
    model.to(device)

    messages = [{"role": "user", "content": args.prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
        )

    prompt_length = int(inputs["input_ids"].shape[-1])
    generated_ids = generated[0, prompt_length:].detach().cpu().tolist()
    payload = {
        "schema": 1,
        "model": args.model,
        "prompt": args.prompt,
        "rendered_prompt": rendered,
        "prompt_token_ids": inputs["input_ids"][0].detach().cpu().tolist(),
        "generated_token_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "dtype": "bfloat16",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {output}")
    print(f"generated_token_ids={generated_ids}")
    print(payload["generated_text"])


if __name__ == "__main__":
    main()
