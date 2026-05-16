"""Single-process rollout generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rllm.core.interfaces import Actor, ReferencePolicy, RewardProvider, RolloutGenerator
from rllm.core.types import GenerationConfig, PromptBatch, RolloutBatch


@dataclass(frozen=True)
class LocalRolloutConfig:
    """Configuration for local rollout collection."""

    num_generations: int = 4

    def __post_init__(self) -> None:
        if self.num_generations <= 0:
            raise ValueError("num_generations must be positive")


class LocalRolloutGenerator(RolloutGenerator):
    """Generate response groups on one process with a PyTorch actor."""

    def __init__(
        self,
        actor: Actor,
        reward_provider: RewardProvider,
        *,
        generation_actor: Actor | None = None,
        reference_policy: ReferencePolicy | None = None,
        config: LocalRolloutConfig | None = None,
    ) -> None:
        self.actor = actor
        self.generation_actor = generation_actor or actor
        self.reward_provider = reward_provider
        self.reference_policy = reference_policy
        self.config = config or LocalRolloutConfig()

    @torch.no_grad()
    def generate(self, prompts: PromptBatch, config: GenerationConfig) -> RolloutBatch:
        device = self.generation_actor.device
        prompts = prompts.to(device)
        sequences: list[torch.Tensor] = []
        prompt_lengths: list[int] = []
        group_ids: list[int] = []
        metadata: list[dict[str, Any]] = []

        for group_id in range(prompts.batch_size):
            prompt_length = int(prompts.attention_mask[group_id].sum().item())
            if prompt_length <= 0:
                raise ValueError("empty prompts are not supported")
            prompt_ids = prompts.input_ids[group_id, :prompt_length].unsqueeze(0)
            prompt_mask = torch.ones_like(prompt_ids)
            prompt = PromptBatch(prompt_ids, prompt_mask)
            for generation_index in range(self.config.num_generations):
                sequence = self.generation_actor.generate(prompt, config).squeeze(0).detach()
                sequences.append(sequence)
                prompt_lengths.append(prompt_length)
                group_ids.append(group_id)
                metadata.append({"group_id": group_id, "generation_index": generation_index})

        max_len = max(int(sequence.numel()) for sequence in sequences)
        batch_size = len(sequences)
        input_ids = torch.full(
            (batch_size, max_len),
            config.pad_token_id,
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)
        for row, sequence in enumerate(sequences):
            length = int(sequence.numel())
            input_ids[row, :length] = sequence
            attention_mask[row, :length] = 1

        action_mask = torch.zeros((batch_size, max_len - 1), device=device, dtype=torch.bool)
        prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device, dtype=torch.long)
        for row, prompt_length in enumerate(prompt_lengths):
            sequence_length = int(attention_mask[row].sum().item())
            action_mask[row, prompt_length - 1 : sequence_length - 1] = True

        old_logprobs = self.generation_actor.logprobs(input_ids, attention_mask).detach()
        ref_logprobs = None
        if self.reference_policy is not None:
            ref_logprobs = self.reference_policy.logprobs(input_ids, attention_mask).detach().to(device)

        rollouts = RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            prompt_lengths=prompt_lengths_tensor,
            group_ids=torch.tensor(group_ids, device=device, dtype=torch.long),
            metadata=metadata,
        )
        rewards = self.reward_provider.score(prompts, rollouts)
        return rollouts.with_updates(rewards=rewards)
