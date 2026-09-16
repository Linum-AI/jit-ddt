# Copyright 2026 Linum Inc. Licensed under the Apache License, Version 2.0.

"""
File: loss.py
Description: Reference implementation of the JiT-DDT training objective, for readers of the
    paper/code. This file is documentation in executable form: every term is written as it was
    computed during training, with the released model's hyperparameters as defaults. It is NOT
    a training script — no data, optimizer, or distributed machinery is shipped — and the two
    frozen perception networks it references (DINOv2 ViT-B/14 with registers, LPIPS-VGG) are
    passed in as callables rather than shipped. The PixelREPA masked transformer adapter IS
    implemented here (PixelRepaAdapter, verified against the internal one) but its weights are
    not released — it is a training-time head that plays no part in sampling.

    Notation: x_0 is the clean image in [-1, 1], eps ~ N(0, I), t in (0, 1) with t=0 clean and
    t=1 noise. Shapes are (B, 3, 1, H, W) — one frame — throughout.

    Total loss (both DiTs trained jointly, gradients flow from the decoder loss back through
    the encoder's plan tokens; nothing is stop-gradiented):

        L = L_dec + L_enc + 0.1 * L_pixel_repa + 0.1 * L_lpips + 0.01 * L_pdino
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from jit_ddt.attention import build_varlen_metadata
from jit_ddt.model import (
    Block,
    build_rope_table,
    compute_rotary_frequencies,
    sinusoidal_embedding_1d,
)


@dataclass(frozen=True)
class LossConfig:
    """Objective hyperparameters of the released model (Linum experiment 47b)."""

    noise_scale: float = 2.0          # x_1 = noise_scale * eps  (JiT-style scaled noise)
    t_eps: float = 0.1                # Clamp on t in the 1/t^2 velocity weighting
    num_timesteps: int = 1000         # Integer timestep fed to the model = int(t * 1000)
    # Timestep distribution: logit-normal, one draw shared by encoder and decoder.
    #   Stage A (first ~80M samples): mean 0.8,  std 0.8   (noise-heavy)
    #   Stage B (next  ~70M samples): mean -0.2, std 1.0   (shifted toward clean)
    logit_normal_mean: float = -0.2
    logit_normal_std: float = 1.0
    encoder_weight: float = 1.0
    decoder_weight: float = 1.0
    encoder_downsample: int = 8       # Encoder predicts x_0 at 1/8 resolution (patch 64 -> 8)
    pixel_repa_weight: float = 0.1
    pixel_repa_mask_ratio: float = 0.2
    pixel_repa_block: int = 5         # Encoder block (1-indexed, of 11) whose output is aligned
    lpips_weight: float = 0.1
    pdino_weight: float = 0.01
    perceptual_gate_t: float = 0.7    # Perceptual terms only for samples with t <= 0.7
    text_dropout: float = 0.05        # Caption -> all-zeros embedding with this probability
    ema_half_life_steps: int = 6594   # EMA beta = exp(-ln 2 / half_life) = 0.99989489
    ema_start_step: int = 7321


# --------------------------------
# FORWARD PROCESS
# --------------------------------

def sample_timesteps(batch_size: int, cfg: LossConfig, device: torch.device) -> torch.Tensor:
    """
    Logit-normal timesteps, t = sigmoid(mean + std * z), clamped away from 0 and 1.

    Args:
        batch_size (int):
            Number of samples.
        cfg (LossConfig):
            Objective hyperparameters.
        device (torch.device):
            Device.

    Returns:
        torch.Tensor:
            t of shape (B,), shared by the encoder and the decoder.
    """
    z = torch.randn(batch_size, device=device)
    t = torch.sigmoid(z * cfg.logit_normal_std + cfg.logit_normal_mean)
    return t.clamp(1e-3, 1 - 1e-3)


def add_noise(
        x_0: torch.Tensor,
        t: torch.Tensor,
        cfg: LossConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linear interpolation x_t = (1 - t) x_0 + t x_1 with scaled Gaussian noise x_1.

    Args:
        x_0 (torch.Tensor):
            Clean images (B, 3, 1, H, W) in [-1, 1].
        t (torch.Tensor):
            Timesteps (B,).
        cfg (LossConfig):
            Objective hyperparameters.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            x_t, and the integer timesteps int(t * num_timesteps) the model consumes.
    """
    x_1 = torch.randn_like(x_0) * cfg.noise_scale
    t5 = t.view(-1, 1, 1, 1, 1)
    x_t = (1 - t5) * x_0 + t5 * x_1
    return x_t, (t * cfg.num_timesteps).long()


def drop_text(text: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """
    Classifier-free-guidance dropout: whole captions replaced by zeros (lengths unchanged).

    Args:
        text (torch.Tensor):
            Caption embeddings (B, L, D).
        cfg (LossConfig):
            Objective hyperparameters.

    Returns:
        torch.Tensor:
            Embeddings with a 5 % subset of samples zeroed.
    """
    keep = torch.bernoulli(torch.full((text.shape[0],), 1 - cfg.text_dropout, device=text.device))
    return text * keep.view(-1, 1, 1)


# --------------------------------
# DIFFUSION LOSSES (x-prediction, velocity-space MSE)
# --------------------------------

def velocity_mse(
        x_pred: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
        cfg: LossConfig) -> torch.Tensor:
    """
    The model predicts x_0; the loss is the MSE of the implied velocity.

    With x_t = (1 - t) x_0 + t x_1 the true velocity is v = x_1 - x_0 and the predicted one
    is (x_t - x_pred) / t, so ||v_pred - v||^2 = ||x_pred - x_0||^2 / t^2. The 1/t^2 weight is
    clamped at t_eps (max weight 100x).

    Args:
        x_pred (torch.Tensor):
            Predicted clean image, same shape as x_0.
        x_0 (torch.Tensor):
            Target clean image.
        t (torch.Tensor):
            Continuous timesteps (B,).
        cfg (LossConfig):
            Objective hyperparameters.

    Returns:
        torch.Tensor:
            Scalar loss (mean over all elements).
    """
    t5 = t.view(-1, 1, 1, 1, 1).clamp_min(cfg.t_eps)
    return (((x_pred - x_0) / t5) ** 2).mean()


def encoder_target(x_0: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """
    The encoder's regression target: x_0 antialiased-bilinear downsampled 8x.

    Args:
        x_0 (torch.Tensor):
            Clean images (B, 3, 1, H, W).
        cfg (LossConfig):
            Objective hyperparameters.

    Returns:
        torch.Tensor:
            (B, 3, 1, H/8, W/8).
    """
    h, w = x_0.shape[-2] // cfg.encoder_downsample, x_0.shape[-1] // cfg.encoder_downsample
    small = F.interpolate(
        x_0[:, :, 0], size=(h, w), mode="bilinear", align_corners=False, antialias=True)
    return small.unsqueeze(2)


# --------------------------------
# PIXEL-REPA (representation alignment on the encoder)
# --------------------------------

class PixelRepaAdapter(nn.Module):
    """
    PixelREPA masked transformer adapter (training-only; 241M parameters at the released
    width). Reads the encoder's hidden state after block 5 — the full in-context sequence
    [image tokens | caption tokens] — replaces a random 20 % of the image tokens with a learned
    mask token, runs two AdaLN DiT blocks (same block as the model, 16 heads, FFN 8192, its own
    timestep pipeline, 3-D RoPE with captions at identity), and projects the image tokens to
    DINOv2's 768 dims through Linear -> LayerNorm -> SiLU -> Linear.
    """

    def __init__(
            self,
            dim: int = 2944,
            num_heads: int = 16,
            ffn_dim: int = 8192,
            num_blocks: int = 2,
            freq_dim: int = 256,
            eps: float = 1e-6,
            out_dim: int = 768,
            mask_ratio: float = 0.2,
            rope_max_positions: int = 1024):
        """
        Build the adapter (defaults are the released model's).

        Args:
            dim (int):
                Width; equals the encoder's width (no input projection).
            num_heads (int):
                Attention heads (head dim 184).
            ffn_dim (int):
                SwiGLU target width (hidden 5632).
            num_blocks (int):
                Number of DiT blocks.
            freq_dim (int):
                Sinusoidal timestep width.
            eps (float):
                q/k RMSNorm epsilon.
            out_dim (int):
                Target feature width (768 for DINOv2-B).
            mask_ratio (float):
                Fraction of image tokens replaced by the mask token.
            rope_max_positions (int):
                RoPE table length.
        """
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.time_embed = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            Block(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, eps=eps, modulated=True)
            for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, out_dim))
        head_dim = dim // num_heads
        with torch.device("cpu"):
            self.freqs = torch.cat([
                compute_rotary_frequencies(
                    max_seq_len=rope_max_positions, dim=head_dim - 4 * (head_dim // 6)),
                compute_rotary_frequencies(
                    max_seq_len=rope_max_positions, dim=2 * (head_dim // 6)),
                compute_rotary_frequencies(
                    max_seq_len=rope_max_positions, dim=2 * (head_dim // 6)),
            ], dim=1)

    def forward(
            self,
            tokens: torch.Tensor,
            mask: torch.Tensor,
            t: torch.Tensor,
            grid: Tuple[int, int, int]) -> torch.Tensor:
        """
        Project masked encoder tokens to DINO space.

        Args:
            tokens (torch.Tensor):
                Encoder hidden state after the tapped block, (B, L_img + L_text, dim).
            mask (torch.Tensor):
                The encoder's validity mask for that sequence, (B, L_img + L_text).
            t (torch.Tensor):
                Integer timesteps (B,).
            grid (Tuple[int, int, int]):
                Image token grid (f, h, w); L_img = f * h * w.

        Returns:
            torch.Tensor:
                (B, L_img, out_dim).
        """
        if self.freqs.device != tokens.device:
            self.freqs = self.freqs.to(tokens.device)
        f_p, h_p, w_p = grid
        l_img = f_p * h_p * w_p
        x_img = tokens[:, :l_img]
        keep = torch.rand(x_img.shape[0], l_img, device=tokens.device) >= self.mask_ratio
        x_img = torch.where(keep.unsqueeze(-1), x_img, self.mask_token.to(x_img.dtype))
        x = torch.cat([x_img, tokens[:, l_img:]], dim=1)

        e = self.time_embed(sinusoidal_embedding_1d(dim=self.freq_dim, position=t, dtype=x.dtype))
        e0 = self.time_proj(e).unflatten(1, (6, self.dim))
        meta = build_varlen_metadata(mask=mask)
        rope = build_rope_table(
            freqs=self.freqs, f_s=f_p, h_s=h_p, w_s=w_p, text_len=tokens.shape[1] - l_img,
            extra_positions=None)
        for block in self.blocks:
            x = block(x=x, e=e0, mask=mask, rope_table=rope, varlen_meta=meta)
        return self.head(x[:, :l_img])


def pixel_repa_loss(
        block_tokens: torch.Tensor,
        block_mask: torch.Tensor,
        t_int: torch.Tensor,
        grid: Tuple[int, int, int],
        adapter: PixelRepaAdapter,
        dino_tokens: torch.Tensor) -> torch.Tensor:
    """
    Negative cosine similarity between DINOv2 patch features of the CLEAN image and the
    adapter's projection of the encoder's block-5 tokens (PixelREPA). Averaged over ALL image
    tokens, masked and kept alike.

    DINO targets: the clean image at 512x512 is resized to the encoder's 8x8 token grid times
    DINO's 14-pixel patch (112x112), ImageNet-normalized, and read at DINOv2's final block with
    the CLS + register tokens dropped — 64 tokens matching the encoder's 64.

    Args:
        block_tokens (torch.Tensor):
            Encoder hidden state after block `pixel_repa_block`, (B, L_img + L_text, dim).
        block_mask (torch.Tensor):
            The encoder's validity mask for that sequence, (B, L_img + L_text).
        t_int (torch.Tensor):
            Integer timesteps (B,).
        grid (Tuple[int, int, int]):
            Image token grid (f, h, w).
        adapter (PixelRepaAdapter):
            The masked adapter.
        dino_tokens (torch.Tensor):
            DINOv2 features of x_0, (B, L_img, 768).

    Returns:
        torch.Tensor:
            Scalar in [-1, 1]; lower is better.
    """
    z = adapter(tokens=block_tokens, mask=block_mask, t=t_int, grid=grid)
    return -(F.normalize(dino_tokens, dim=-1) * F.normalize(z, dim=-1)).sum(-1).mean()


# --------------------------------
# PERCEPTUAL LOSSES ON THE DECODER PREDICTION
# --------------------------------

def perceptual_gate(t: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    """
    Per-sample gate: perceptual terms apply only when t <= 0.7 (the prediction is a
    plausible image). Terms are averaged over the gated-in samples, not the whole batch.

    Args:
        t (torch.Tensor):
            Timesteps (B,).
        cfg (LossConfig):
            Objective hyperparameters.

    Returns:
        torch.Tensor:
            Float gate (B,).
    """
    return (t <= cfg.perceptual_gate_t).float()


def gated_mean(per_sample: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Mean over the samples the gate lets through.

    Args:
        per_sample (torch.Tensor):
            Per-sample losses (B,).
        gate (torch.Tensor):
            Float gate (B,).

    Returns:
        torch.Tensor:
            Scalar.
    """
    return (per_sample * gate).sum() / gate.sum().clamp_min(1.0)


def lpips_loss(
        x_pred: torch.Tensor,
        x_0: torch.Tensor,
        lpips_vgg: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        gate: torch.Tensor,
        crop: int = 224) -> torch.Tensor:
    """
    LPIPS (VGG) between the decoder's x_0 prediction and the clean image on one shared random
    224x224 crop per batch. Inputs stay in [-1, 1].

    Args:
        x_pred (torch.Tensor):
            Decoder prediction (B, 3, 1, H, W).
        x_0 (torch.Tensor):
            Clean image (B, 3, 1, H, W).
        lpips_vgg (Callable):
            Frozen LPIPS network: (pred, target) 4-D in [-1, 1] -> (B, 1, 1, 1).
        gate (torch.Tensor):
            Output of perceptual_gate.
        crop (int):
            Crop size. Default 224.

    Returns:
        torch.Tensor:
            Scalar.
    """
    pred, target = x_pred[:, :, 0], x_0[:, :, 0]
    if min(pred.shape[-2:]) < crop:   # bicubic upsize so the shorter side reaches the crop
        scale = crop / min(pred.shape[-2:])
        size = (max(crop, round(pred.shape[-2] * scale)), max(crop, round(pred.shape[-1] * scale)))
        pred = F.interpolate(pred, size=size, mode="bicubic", align_corners=False)
        target = F.interpolate(target, size=size, mode="bicubic", align_corners=False)
    top = int(torch.randint(0, pred.shape[-2] - crop + 1, (1,)))
    left = int(torch.randint(0, pred.shape[-1] - crop + 1, (1,)))
    pred = pred[..., top:top + crop, left:left + crop]
    target = target[..., top:top + crop, left:left + crop]
    return gated_mean(per_sample=lpips_vgg(pred, target).view(-1), gate=gate)


def pdino_loss(
        x_pred: torch.Tensor,
        x_0: torch.Tensor,
        dino_features: Callable[[torch.Tensor], torch.Tensor],
        gate: torch.Tensor) -> torch.Tensor:
    """
    Perceptual DINO loss: 1 - cosine similarity between DINOv2 patch features of the
    prediction (with gradient) and of the clean image (no gradient), averaged over tokens.
    Images are resized so the longer side is <= 224 (rounded to DINO's 14-pixel patch) and
    ImageNet-normalized.

    Args:
        x_pred (torch.Tensor):
            Decoder prediction (B, 3, 1, H, W).
        x_0 (torch.Tensor):
            Clean image (B, 3, 1, H, W).
        dino_features (Callable):
            Frozen DINOv2 feature extractor: 4-D image in [-1, 1] -> (B, N, 768).
        gate (torch.Tensor):
            Output of perceptual_gate.

    Returns:
        torch.Tensor:
            Scalar.
    """
    feat_pred = dino_features(x_pred[:, :, 0])
    with torch.no_grad():
        feat_target = dino_features(x_0[:, :, 0])
    per_sample = (1 - F.cosine_similarity(feat_pred, feat_target, dim=-1)).mean(dim=-1)
    return gated_mean(per_sample=per_sample, gate=gate)


# --------------------------------
# TOTAL
# --------------------------------

def jit_ddt_loss(
        model: Callable,
        x_0: torch.Tensor,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        cfg: LossConfig,
        adapter: Optional[PixelRepaAdapter] = None,
        dino_tokens_clean: Optional[torch.Tensor] = None,
        lpips_vgg: Optional[Callable] = None,
        dino_features: Optional[Callable] = None) -> Dict[str, torch.Tensor]:
    """
    One training step's losses, as computed for the released model.

    Args:
        model (Callable):
            JiT-DDT forward returning an object with `.x_hat_dec`, `.x_hat_enc` and, for the
            REPA tap, `.encoder_block_tokens` / `.encoder_block_mask` (the encoder's block-5
            hidden state [image | caption] and its validity mask). The released JitDDT
            returns only the first two; the tap is a training-time hook.
        x_0 (torch.Tensor):
            Clean images (B, 3, 1, H, W) in [-1, 1].
        text (torch.Tensor):
            Caption embeddings (B, 256, 7680).
        text_lens (torch.Tensor):
            Caption lengths (B,).
        cfg (LossConfig):
            Objective hyperparameters.
        adapter (Optional[PixelRepaAdapter]):
            PixelREPA masked adapter (training-only module).
        dino_tokens_clean (Optional[torch.Tensor]):
            DINOv2 tokens of x_0 on the encoder grid.
        lpips_vgg (Optional[Callable]):
            Frozen LPIPS-VGG.
        dino_features (Optional[Callable]):
            Frozen DINOv2 feature extractor.

    Returns:
        Dict[str, torch.Tensor]:
            Every term plus "total".
    """
    t = sample_timesteps(batch_size=x_0.shape[0], cfg=cfg, device=x_0.device)
    x_t, t_int = add_noise(x_0=x_0, t=t, cfg=cfg)
    out = model(x_t=x_t, text=drop_text(text=text, cfg=cfg), text_lens=text_lens, t=t_int)

    losses = {
        "decoder": cfg.decoder_weight * velocity_mse(
            x_pred=out.x_hat_dec, x_0=x_0, t=t, cfg=cfg),
        "encoder": cfg.encoder_weight * velocity_mse(
            x_pred=out.x_hat_enc, x_0=encoder_target(x_0=x_0, cfg=cfg), t=t, cfg=cfg),
    }
    if adapter is not None and dino_tokens_clean is not None:
        grid = (1, x_0.shape[-2] // 64, x_0.shape[-1] // 64)        # encoder patch 64
        losses["pixel_repa"] = cfg.pixel_repa_weight * pixel_repa_loss(
            block_tokens=out.encoder_block_tokens, block_mask=out.encoder_block_mask,
            t_int=t_int, grid=grid, adapter=adapter, dino_tokens=dino_tokens_clean)
    gate = perceptual_gate(t=t, cfg=cfg)
    if lpips_vgg is not None:
        losses["lpips"] = cfg.lpips_weight * lpips_loss(
            x_pred=out.x_hat_dec, x_0=x_0, lpips_vgg=lpips_vgg, gate=gate)
    if dino_features is not None:
        losses["pdino"] = cfg.pdino_weight * pdino_loss(
            x_pred=out.x_hat_dec, x_0=x_0, dino_features=dino_features, gate=gate)
    losses["total"] = sum(losses.values())
    return losses


def ema_beta(cfg: LossConfig) -> float:
    """
    EMA decay used to produce the released weights: shadow = beta * shadow + (1 - beta) * w,
    every step from `ema_start_step` on.

    Args:
        cfg (LossConfig):
            Objective hyperparameters.

    Returns:
        float:
            beta (0.99989489 for a 6594-step half-life).
    """
    return math.exp(-math.log(2.0) / cfg.ema_half_life_steps)
