"""Group Relative Policy Optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rllm.core.interfaces import RLAlgorithm
from rllm.core.types import AlgorithmStats, RolloutBatch
from rllm.utils.logprobs import masked_mean


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for GRPO loss computation."""

    clip_ratio: float = 0.2
    beta_kl: float = 0.04
    advantage_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.clip_ratio <= 0:
            raise ValueError("clip_ratio must be positive")
        if self.beta_kl < 0:
            raise ValueError("beta_kl must be non-negative")
        if self.advantage_eps <= 0:
            raise ValueError("advantage_eps must be positive")


def compute_group_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize scalar rewards inside each prompt group."""

    if rewards.ndim != 1 or group_ids.ndim != 1:
        raise ValueError("rewards and group_ids must be rank-1 tensors")
    if rewards.shape != group_ids.shape:
        raise ValueError("rewards and group_ids must have the same shape")

    advantages = torch.zeros_like(rewards)
    for group_id in torch.unique(group_ids, sorted=True):
        mask = group_ids == group_id
        group_rewards = rewards[mask]
        mean = group_rewards.mean()
        std = group_rewards.std(unbiased=False)
        advantages[mask] = (group_rewards - mean) / (std + eps)
    return advantages


def broadcast_sequence_advantages(
    sequence_advantages: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Broadcast one advantage per sequence to generated-token positions."""

    if sequence_advantages.ndim != 1:
        raise ValueError("sequence_advantages must be rank-1")
    if sequence_advantages.shape[0] != action_mask.shape[0]:
        raise ValueError("sequence_advantages batch dimension must match action_mask")
    return sequence_advantages[:, None].to(dtype=torch.float32) * action_mask.to(dtype=torch.float32)


def reference_kl(new_logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """Non-negative token-level KL estimator against a reference policy."""

    log_ratio = ref_logprobs - new_logprobs
    return torch.exp(log_ratio) - log_ratio - 1.0


def grpo_loss(
    rollouts: RolloutBatch,
    new_logprobs: torch.Tensor,
    config: GRPOConfig | None = None,
) -> tuple[torch.Tensor, AlgorithmStats]:
    """Compute clipped GRPO objective for the latest policy logprobs."""

    config = config or GRPOConfig()
    if new_logprobs.shape != rollouts.old_logprobs.shape:
        raise ValueError("new_logprobs shape must match old_logprobs")

    if rollouts.advantages is None:
        if rollouts.rewards is None:
            raise ValueError("rollouts need rewards or precomputed advantages")
        sequence_advantages = compute_group_advantages(
            rollouts.rewards,
            rollouts.group_ids,
            eps=config.advantage_eps,
        )
        advantages = broadcast_sequence_advantages(sequence_advantages, rollouts.action_mask)
    else:
        advantages = rollouts.advantages

    mask = rollouts.action_mask
    old_logprobs = rollouts.old_logprobs.detach()
    advantages = advantages.detach().to(device=new_logprobs.device, dtype=new_logprobs.dtype)
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    policy_loss = -masked_mean(surrogate, mask)

    if rollouts.ref_logprobs is None or config.beta_kl == 0:
        kl = torch.zeros((), device=new_logprobs.device, dtype=new_logprobs.dtype)
    else:
        kl_tokens = reference_kl(new_logprobs, rollouts.ref_logprobs.to(new_logprobs.device))
        kl = masked_mean(kl_tokens, mask)

    loss = policy_loss + config.beta_kl * kl
    clip_fraction = masked_mean(((ratio - 1.0).abs() > config.clip_ratio).to(new_logprobs.dtype), mask)
    mean_reward = None if rollouts.rewards is None else rollouts.rewards.mean()
    stats = AlgorithmStats(
        loss=loss.detach(),
        policy_loss=policy_loss.detach(),
        kl=kl.detach(),
        clip_fraction=clip_fraction.detach(),
        mean_reward=None if mean_reward is None else mean_reward.detach(),
    )
    return loss, stats


class GRPO(RLAlgorithm):
    """State-light GRPO algorithm object."""

    def __init__(self, config: GRPOConfig | None = None) -> None:
        self.config = config or GRPOConfig()

    def prepare_rollouts(self, rollouts: RolloutBatch) -> RolloutBatch:
        if rollouts.rewards is None:
            raise ValueError("GRPO requires scalar rewards")
        sequence_advantages = compute_group_advantages(
            rollouts.rewards,
            rollouts.group_ids,
            eps=self.config.advantage_eps,
        )
        advantages = broadcast_sequence_advantages(sequence_advantages, rollouts.action_mask)
        return rollouts.with_updates(advantages=advantages)

    def loss(self, rollouts: RolloutBatch, new_logprobs: torch.Tensor) -> tuple[torch.Tensor, AlgorithmStats]:
        return grpo_loss(rollouts, new_logprobs, self.config)

