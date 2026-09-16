# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: config.py
Description: Architecture and sampler configuration for JiT-DDT. The defaults are exactly the
    released model (Linum experiment 47b); the architecture dataclass exists so the shapes are
    documented in one place and can be round-tripped through the `config.json` shipped with
    the weights, not because other shapes are supported.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class JitDDTConfig:
    """
    JiT-DDT architecture. Two DiTs (encoder + decoder) share every hyperparameter below except
    their patch sizes: the encoder patchifies 64x64 pixels into one token and predicts an
    8x downsampled image; the decoder patchifies 32x32 and predicts at full resolution.
    """

    dim: int = 2944
    num_heads: int = 23
    num_layers: int = 11
    ffn_dim: int = 7936               # SwiGLU target width; hidden = round256(2/3 * ffn_dim)
    refiner_blocks: int = 2           # Per-stream pre-conditioner depth (text + image, each DiT)
    refiner_ffn_dim: int = 4096
    text_dim: int = 7680              # 3 x 2560: Qwen3.5-4B hidden states, layers (7, 15, 27)
    text_len: int = 256
    in_channels: int = 3
    out_channels: int = 3
    freq_dim: int = 256               # Sinusoidal timestep embedding width
    eps: float = 1e-6
    bottleneck_dim: int = 256         # BottleneckPatchEmbed intermediate channels
    encoder_patch: Tuple[int, int, int] = (1, 64, 64)
    encoder_output_patch: Tuple[int, int, int] = (1, 8, 8)
    decoder_patch: Tuple[int, int, int] = (1, 32, 32)
    rope_max_positions: int = 1024
    text_layers: Tuple[int, int, int] = (7, 15, 27)

    @property
    def head_dim(self) -> int:
        """Per-head width (128 for the released model)."""
        return self.dim // self.num_heads

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JitDDTConfig":
        """
        Build a config from a plain dict (e.g. the `architecture` block of `config.json`).

        Args:
            data (Dict[str, Any]):
                Field values; tuple fields may be given as lists.

        Returns:
            JitDDTConfig:
                The config.
        """
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise KeyError(f"Unknown JitDDTConfig fields: {unknown}")
        clean = {
            k: (tuple(v) if isinstance(v, list) else v) for k, v in data.items()}
        return cls(**clean)

    @classmethod
    def from_json(cls, path: str) -> "JitDDTConfig":
        """
        Read the `architecture` block of a `config.json`.

        Args:
            path (str):
                Path to `config.json`.

        Returns:
            JitDDTConfig:
                The config.
        """
        with open(path) as fh:
            data = json.load(fh)
        return cls.from_dict(data["architecture"])

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a JSON-friendly dict.

        Returns:
            Dict[str, Any]:
                Field values with tuples as lists.
        """
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


@dataclass
class SamplerConfig:
    """
    Sampling settings. The defaults are the ones the released model was validated with during
    training: 50 uniform Euler steps from t=1 (noise) to t=0, adaptive projected guidance
    (APG) at scale 15, initial noise scaled by 2.
    """

    height: int = 512
    width: int = 512
    sampling_steps: int = 50
    guidance_scale: float = 15.0
    apg_momentum: float = -0.75
    apg_eta: float = 0.0
    apg_rescale: float = 10.0
    noise_scale: float = 2.0
    negative_prompt: str = "watermark, signature, logo, copyright, url"
    num_timesteps: int = field(default=1000, repr=False)   # Timestep-embedding scale (fixed)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a JSON-friendly dict.

        Returns:
            Dict[str, Any]:
                Field values.
        """
        return asdict(self)
