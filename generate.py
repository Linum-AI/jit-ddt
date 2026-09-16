# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: generate.py
Description: Command-line text-to-image generation with JiT-DDT. Writes one PNG per seed.

    python generate.py --weights /path/to/jit-ddt-weights --qwen_model_path Qwen/Qwen3.5-4B \\
        --prompt "$PROMPT" --seeds 42,123
"""

import argparse
import os
from typing import List

import torch
from PIL import Image

from jit_ddt import (
    JitDDT, QwenTextEncoder, SamplerConfig, active_attention_backend, generate,
    set_attention_backend,
)
from jit_ddt.attention import BACKENDS


# --------------------------------
# CLI
# --------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    defaults = SamplerConfig()
    parser = argparse.ArgumentParser(description="Generate images with JiT-DDT")
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Hugging Face Hub id, or a local directory with model.safetensors + config.json")
    parser.add_argument(
        "--qwen_model_path", type=str, required=True,
        help="Local directory or Hugging Face id of Qwen3.5-4B")
    parser.add_argument("--prompt", type=str, required=True, help="Caption")
    parser.add_argument(
        "--negative_prompt", type=str, default=defaults.negative_prompt,
        help="Caption for the unconditional branch")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")
    parser.add_argument("--height", type=int, default=defaults.height, help="Multiple of 64")
    parser.add_argument("--width", type=int, default=defaults.width, help="Multiple of 64")
    parser.add_argument("--sampling_steps", type=int, default=defaults.sampling_steps)
    parser.add_argument("--guidance_scale", type=float, default=defaults.guidance_scale)
    parser.add_argument(
        "--batch_seeds", action="store_true",
        help="Sample all seeds in one batch (faster; per-image numerics differ slightly from "
             "one-seed-per-call, which is how the reference images were made)")
    parser.add_argument(
        "--attention_backend", type=str, default="auto", choices=BACKENDS,
        help="auto: FlashAttention-3 if installed, else PyTorch SDPA")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Where to write PNGs")
    return parser.parse_args()


def main() -> None:
    """Generate images."""
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    sampler = SamplerConfig(
        height=args.height,
        width=args.width,
        sampling_steps=args.sampling_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
    )

    torch.set_float32_matmul_precision("high")
    set_attention_backend(backend=args.attention_backend)
    print(f"attention backend: {active_attention_backend()}")
    model = JitDDT.from_pretrained(weights_dir=args.weights, device="cuda")
    text_encoder = QwenTextEncoder(
        model_path=args.qwen_model_path,
        extraction_layers=model.config.text_layers,
        max_length=model.config.text_len)

    os.makedirs(args.out_dir, exist_ok=True)
    seed_groups: List[List[int]] = [seeds] if args.batch_seeds else [[s] for s in seeds]
    for group in seed_groups:
        images = generate(
            model=model, text_encoder=text_encoder, prompt=args.prompt, seeds=group,
            sampler=sampler)
        for seed, image in zip(group, images):
            path = os.path.join(args.out_dir, f"seed_{seed}.png")
            save_png(image=image, path=path)
            print(f"wrote {path}")


# --------------------------------
# OUTPUT
# --------------------------------

def save_png(image: torch.Tensor, path: str) -> None:
    """
    Write a (3, 1, H, W) uint8 tensor as a PNG.

    Args:
        image (torch.Tensor):
            uint8 image tensor.
        path (str):
            Destination path.
    """
    array = image[:, 0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


if __name__ == "__main__":
    main()
