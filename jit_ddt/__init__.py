# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: __init__.py
Description: JiT-DDT — a pixel-space text-to-image diffusion transformer (encoder + decoder DiT).
"""

from jit_ddt.attention import active_attention_backend, set_attention_backend
from jit_ddt.config import JitDDTConfig, SamplerConfig
from jit_ddt.model import JitDDT, JitDDTOutput
from jit_ddt.sampler import generate
from jit_ddt.text_encoder import QwenTextEncoder

__all__ = [
    "JitDDT",
    "JitDDTConfig",
    "JitDDTOutput",
    "QwenTextEncoder",
    "SamplerConfig",
    "active_attention_backend",
    "generate",
    "set_attention_backend",
]
