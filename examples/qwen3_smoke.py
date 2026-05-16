"""Qwen3 smoke test for the PyTorch/HF adapter.

Run with:
    python examples/qwen3_smoke.py --model-id Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from rllm.core.types import GenerationConfig, PromptBatch
from rllm.models.hf import HFCausalLMActor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Solve: 1 + 1 =")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    actor = HFCausalLMActor.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).to(args.device)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    prompts = PromptBatch(encoded["input_ids"], encoded["attention_mask"]).to(args.device)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
    )
    sequences = actor.generate(prompts, generation_config)
    attention_mask = torch.ones_like(sequences)
    logprobs = actor.logprobs(sequences, attention_mask)
    print(tokenizer.decode(sequences[0], skip_special_tokens=True))
    print(f"logprobs_shape={tuple(logprobs.shape)}")


if __name__ == "__main__":
    main()

