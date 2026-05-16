"""Helpers for next-token logprob accounting."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Gather label-aligned next-token logprobs.

    Returns a tensor with shape `[batch, seq_len - 1]` where item `t` is the
    logprob assigned to `input_ids[:, t + 1]` by logits from position `t`.
    """

    if logits.ndim != 3:
        raise ValueError(f"logits must have rank 3, got {tuple(logits.shape)}")
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have rank 2, got {tuple(input_ids.shape)}")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must agree on batch and sequence dimensions")
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean over true mask elements, returning zero for an empty mask."""

    if values.shape != mask.shape:
        raise ValueError(f"values and mask shapes differ: {tuple(values.shape)} != {tuple(mask.shape)}")
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum().clamp_min(eps)
    return (values * mask_f).sum() / denom


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Mask logits outside the top-k set."""

    if top_k is None or top_k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, top_k, dim=-1)
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)

