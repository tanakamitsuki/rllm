"""Proximal Policy Optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rllm.core.interfaces import RLAlgorithm
from rllm.core.types import AlgorithmStats, RolloutBatch
from rllm.utils.logprobs import masked_mean


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for PPO loss and return computation."""

    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    gamma: float = 1.0
    lam: float = 0.95

    def __post_init__(self) -> None:
        if self.clip_ratio <= 0:
            raise ValueError("clip_ratio must be positive")
        if self.value_clip_ratio <= 0:
            raise ValueError("value_clip_ratio must be positive")
        if self.value_coef < 0 or self.entropy_coef < 0:
            raise ValueError("loss coefficients must be non-negative")
        if not (0 < self.gamma <= 1):
            raise ValueError("gamma must be in (0, 1]")
        if not (0 <= self.lam <= 1):
            raise ValueError("lam must be in [0, 1]")


def terminal_rewards_to_token_rewards(
    terminal_rewards: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Place each scalar sequence reward on its final generated token."""

    if terminal_rewards.ndim != 1:
        raise ValueError("terminal_rewards must be rank-1")
    if terminal_rewards.shape[0] != action_mask.shape[0]:
        raise ValueError("terminal_rewards batch dimension must match action_mask")

    rewards = torch.zeros(action_mask.shape, device=terminal_rewards.device, dtype=terminal_rewards.dtype)
    for row in range(action_mask.shape[0]):
        action_positions = torch.nonzero(action_mask[row], as_tuple=False).flatten()
        if action_positions.numel() > 0:
            rewards[row, int(action_positions[-1].item())] = terminal_rewards[row]
    return rewards * action_mask.to(dtype=terminal_rewards.dtype)


def generalized_advantage_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute token-level GAE advantages and returns."""

    if rewards.shape != values.shape or rewards.shape != action_mask.shape:
        raise ValueError("rewards, values, and action_mask must share shape")

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    next_values = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    next_mask = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(rewards.shape[1])):
        mask_t = action_mask[:, t].to(dtype=rewards.dtype)
        delta = rewards[:, t] + gamma * next_values * next_mask - values[:, t]
        last_gae = (delta + gamma * lam * last_gae * next_mask) * mask_t
        advantages[:, t] = last_gae
        next_values = values[:, t]
        next_mask = mask_t

    returns = advantages + values
    return advantages, returns


def ppo_loss(
    rollouts: RolloutBatch,
    new_logprobs: torch.Tensor,
    values: torch.Tensor,
    config: PPOConfig | None = None,
) -> tuple[torch.Tensor, AlgorithmStats]:
    """Compute clipped PPO policy and value losses."""

    config = config or PPOConfig()
    if new_logprobs.shape != rollouts.old_logprobs.shape:
        raise ValueError("new_logprobs shape must match old_logprobs")
    if values.shape != rollouts.old_logprobs.shape:
        raise ValueError("values shape must match old_logprobs")
    if rollouts.advantages is None or rollouts.returns is None:
        raise ValueError("PPO requires precomputed advantages and returns")

    mask = rollouts.action_mask
    advantages = rollouts.advantages.detach().to(device=new_logprobs.device, dtype=new_logprobs.dtype)
    returns = rollouts.returns.detach().to(device=values.device, dtype=values.dtype)
    old_values = rollouts.old_values
    if old_values is None:
        old_values = values.detach()
    else:
        old_values = old_values.detach().to(device=values.device, dtype=values.dtype)

    ratio = torch.exp(new_logprobs - rollouts.old_logprobs.detach())
    clipped_ratio = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
    policy_loss = -masked_mean(torch.minimum(ratio * advantages, clipped_ratio * advantages), mask)

    value_unclipped = (values - returns).pow(2)
    values_clipped = old_values + (values - old_values).clamp(-config.value_clip_ratio, config.value_clip_ratio)
    value_clipped = (values_clipped - returns).pow(2)
    value_loss = 0.5 * masked_mean(torch.maximum(value_unclipped, value_clipped), mask)

    entropy = -masked_mean(new_logprobs, mask)
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    clip_fraction = masked_mean(((ratio - 1.0).abs() > config.clip_ratio).to(new_logprobs.dtype), mask)
    mean_reward = None if rollouts.rewards is None else rollouts.rewards.mean()
    stats = AlgorithmStats(
        loss=loss.detach(),
        policy_loss=policy_loss.detach(),
        value_loss=value_loss.detach(),
        entropy=entropy.detach(),
        clip_fraction=clip_fraction.detach(),
        mean_reward=None if mean_reward is None else mean_reward.detach(),
    )
    return loss, stats


class PPO(RLAlgorithm):
    """PPO algorithm object."""

    def __init__(self, config: PPOConfig | None = None) -> None:
        self.config = config or PPOConfig()

    def prepare_rollouts(self, rollouts: RolloutBatch) -> RolloutBatch:
        if rollouts.rewards is None or rollouts.values is None:
            raise ValueError("PPO requires scalar rewards and old token values")
        token_rewards = terminal_rewards_to_token_rewards(rollouts.rewards, rollouts.action_mask)
        advantages, returns = generalized_advantage_estimation(
            token_rewards,
            rollouts.values,
            rollouts.action_mask,
            gamma=self.config.gamma,
            lam=self.config.lam,
        )
        return rollouts.with_updates(
            advantages=advantages,
            returns=returns,
            old_values=rollouts.values.detach(),
        )

    def loss(
        self,
        rollouts: RolloutBatch,
        new_logprobs: torch.Tensor,
        values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AlgorithmStats]:
        if values is None:
            raise ValueError("PPO.loss requires current values")
        return ppo_loss(rollouts, new_logprobs, values, self.config)
