"""Abstract interfaces for rllm components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch

from rllm.core.types import AlgorithmStats, GenerationConfig, PromptBatch, RolloutBatch


class Actor(ABC):
    """Policy model capable of generation and token logprob evaluation."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device containing the actor parameters."""

    @abstractmethod
    def forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return next-token logits with shape `[batch, seq_len, vocab]`."""

    @abstractmethod
    def logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return label-aligned token logprobs with shape `[batch, seq_len - 1]`."""

    @abstractmethod
    def generate(self, prompts: PromptBatch, config: GenerationConfig) -> torch.Tensor:
        """Return unpadded generated sequences for the provided prompts."""

    @abstractmethod
    def backbone_state_dict(self) -> Mapping[str, torch.Tensor]:
        """Return the state dict for the transformer backbone only."""

    def forward_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return backbone hidden states when the backend exposes them."""

        raise NotImplementedError(f"{type(self).__name__} does not expose backbone hidden states")


class ReferencePolicy(ABC):
    """Frozen policy used to compute reference logprobs and KL penalties."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the reference policy device."""

    @abstractmethod
    def logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return label-aligned token logprobs."""


class Critic(ABC):
    """Value model with a backbone that can be checked against an actor."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the critic device."""

    @abstractmethod
    def values(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return label-aligned token values with shape `[batch, seq_len - 1]`."""

    @abstractmethod
    def backbone_state_dict(self) -> Mapping[str, torch.Tensor]:
        """Return the state dict for the transformer backbone only."""

    def forward_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return backbone hidden states when the backend exposes them."""

        raise NotImplementedError(f"{type(self).__name__} does not expose backbone hidden states")


class RewardProvider(ABC):
    """Scores generated responses."""

    @abstractmethod
    def score(self, prompts: PromptBatch, rollouts: RolloutBatch) -> torch.Tensor:
        """Return one scalar reward per rollout sequence."""


class RolloutGenerator(ABC):
    """Builds rollout batches from prompts."""

    @abstractmethod
    def generate(self, prompts: PromptBatch, config: GenerationConfig) -> RolloutBatch:
        """Generate responses and fill rollout accounting tensors."""


class RLAlgorithm(ABC):
    """Algorithm object that can annotate rollouts and compute a loss."""

    @abstractmethod
    def prepare_rollouts(self, rollouts: RolloutBatch) -> RolloutBatch:
        """Add algorithm-specific tensors such as advantages or returns."""

    @abstractmethod
    def loss(self, rollouts: RolloutBatch, new_logprobs: torch.Tensor) -> tuple[torch.Tensor, AlgorithmStats]:
        """Compute the differentiable objective for the latest policy."""
