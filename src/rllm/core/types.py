"""Shared tensor batch types used by algorithms and backends."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import torch


Metadata = Mapping[str, Any]


def _check_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}")


def _check_same_shape(name: str, tensor: torch.Tensor, expected: torch.Size) -> None:
    if tensor.shape != expected:
        raise ValueError(f"{name} must have shape {tuple(expected)}, got {tuple(tensor.shape)}")


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling configuration for autoregressive rollout generation."""

    max_new_tokens: int
    temperature: float = 1.0
    top_k: int | None = None
    do_sample: bool = True
    eos_token_id: int | None = None
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when set")


@dataclass(frozen=True)
class PromptBatch:
    """A padded batch of prompts."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    metadata: Sequence[Metadata] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_rank("input_ids", self.input_ids, 2)
        _check_rank("attention_mask", self.attention_mask, 2)
        _check_same_shape("attention_mask", self.attention_mask, self.input_ids.shape)
        if self.input_ids.dtype != torch.long:
            raise ValueError("input_ids must be torch.long")
        if self.metadata and len(self.metadata) != self.batch_size:
            raise ValueError("metadata length must match batch size")

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def device(self) -> torch.device:
        return self.input_ids.device

    def to(self, device: torch.device | str) -> "PromptBatch":
        return replace(
            self,
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
        )


@dataclass(frozen=True)
class RolloutBatch:
    """Generated sequences plus token-level RL accounting tensors.

    `input_ids` has shape `[batch, seq_len]`. Token logprob tensors have shape
    `[batch, seq_len - 1]`, aligned to labels `input_ids[:, 1:]`. `action_mask`
    selects generated response tokens in that label-aligned space.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    old_logprobs: torch.Tensor
    prompt_lengths: torch.Tensor
    group_ids: torch.Tensor
    ref_logprobs: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    values: torch.Tensor | None = None
    old_values: torch.Tensor | None = None
    metadata: Sequence[Metadata] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_rank("input_ids", self.input_ids, 2)
        _check_rank("attention_mask", self.attention_mask, 2)
        _check_same_shape("attention_mask", self.attention_mask, self.input_ids.shape)
        if self.input_ids.dtype != torch.long:
            raise ValueError("input_ids must be torch.long")

        logprob_shape = torch.Size((self.input_ids.shape[0], self.input_ids.shape[1] - 1))
        _check_same_shape("action_mask", self.action_mask, logprob_shape)
        _check_same_shape("old_logprobs", self.old_logprobs, logprob_shape)
        if self.action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be torch.bool")

        batch_shape = torch.Size((self.input_ids.shape[0],))
        _check_same_shape("prompt_lengths", self.prompt_lengths, batch_shape)
        _check_same_shape("group_ids", self.group_ids, batch_shape)
        if self.prompt_lengths.dtype != torch.long:
            raise ValueError("prompt_lengths must be torch.long")
        if self.group_ids.dtype != torch.long:
            raise ValueError("group_ids must be torch.long")

        for name in (
            "ref_logprobs",
            "advantages",
            "returns",
            "values",
            "old_values",
        ):
            tensor = getattr(self, name)
            if tensor is not None:
                _check_same_shape(name, tensor, logprob_shape)
        if self.rewards is not None:
            _check_same_shape("rewards", self.rewards, batch_shape)
        if self.metadata and len(self.metadata) != self.batch_size:
            raise ValueError("metadata length must match batch size")

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def device(self) -> torch.device:
        return self.input_ids.device

    @property
    def response_lengths(self) -> torch.Tensor:
        return self.action_mask.sum(dim=1)

    def to(self, device: torch.device | str) -> "RolloutBatch":
        kwargs = {
            field_name: getattr(self, field_name).to(device)
            for field_name in (
                "input_ids",
                "attention_mask",
                "action_mask",
                "old_logprobs",
                "prompt_lengths",
                "group_ids",
            )
        }
        optional_names = (
            "ref_logprobs",
            "rewards",
            "advantages",
            "returns",
            "values",
            "old_values",
        )
        for name in optional_names:
            tensor = getattr(self, name)
            kwargs[name] = None if tensor is None else tensor.to(device)
        return replace(self, **kwargs)

    def with_updates(self, **kwargs: Any) -> "RolloutBatch":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class AlgorithmStats:
    """Scalar metrics returned by algorithm loss functions."""

    loss: torch.Tensor
    policy_loss: torch.Tensor | None = None
    value_loss: torch.Tensor | None = None
    kl: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    clip_fraction: torch.Tensor | None = None
    mean_reward: torch.Tensor | None = None
    extra: Mapping[str, torch.Tensor] = field(default_factory=dict)

