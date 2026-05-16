"""Rule-based reward providers for RLVR-style validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from rllm.core.interfaces import RewardProvider
from rllm.core.types import PromptBatch, RolloutBatch


@dataclass(frozen=True)
class RewardExample:
    """One generated response presented to a Python reward function."""

    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    group_id: int
    metadata: Mapping[str, Any]


RuleRewardFn = Callable[[RewardExample], float]


class RuleRewardProvider(RewardProvider):
    """Adapt a deterministic Python callable to the reward interface."""

    def __init__(self, reward_fn: RuleRewardFn) -> None:
        self.reward_fn = reward_fn

    def score(self, prompts: PromptBatch, rollouts: RolloutBatch) -> torch.Tensor:
        rewards: list[float] = []
        for row in range(rollouts.batch_size):
            group_id = int(rollouts.group_ids[row].item())
            prompt_length = int(rollouts.prompt_lengths[row].item())
            sequence_length = int(rollouts.attention_mask[row].sum().item())
            prompt_ids = rollouts.input_ids[row, :prompt_length].detach().cpu()
            response_ids = rollouts.input_ids[row, prompt_length:sequence_length].detach().cpu()
            metadata = prompts.metadata[group_id] if prompts.metadata else {}
            rewards.append(
                float(
                    self.reward_fn(
                        RewardExample(
                            prompt_ids=prompt_ids,
                            response_ids=response_ids,
                            group_id=group_id,
                            metadata=metadata,
                        )
                    )
                )
            )
        return torch.tensor(rewards, device=rollouts.device, dtype=torch.float32)

