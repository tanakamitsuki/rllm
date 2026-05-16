"""Tiny GRPO rule-reward example.

Run with:
    python examples/tiny_grpo_rule.py
"""

from __future__ import annotations

import torch

from rllm.core.identity import assert_bitwise_identical_backbones
from rllm.core.types import GenerationConfig, PromptBatch
from rllm.models.torch_causal_lm import (
    FrozenReferencePolicy,
    TinyCausalLMConfig,
    build_tiny_actor_critic,
)
from rllm.rewards.rule import RewardExample, RuleRewardProvider
from rllm.rollouts.local import LocalRolloutConfig, LocalRolloutGenerator
from rllm.trainers.grpo import GRPOTrainer


TARGET_TOKEN = 7


def reward_fn(example: RewardExample) -> float:
    return 1.0 if example.response_ids.numel() > 0 and int(example.response_ids[0]) == TARGET_TOKEN else 0.0


def main() -> None:
    torch.manual_seed(0)
    actor, critic = build_tiny_actor_critic(
        TinyCausalLMConfig(vocab_size=16, hidden_size=32, num_heads=4, max_position_embeddings=32),
        seed=0,
    )
    assert_bitwise_identical_backbones(actor, critic)
    reference = FrozenReferencePolicy(actor)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=3e-3)
    rollout_generator = LocalRolloutGenerator(
        actor,
        RuleRewardProvider(reward_fn),
        reference_policy=reference,
        config=LocalRolloutConfig(num_generations=4),
    )
    trainer = GRPOTrainer(actor, optimizer, rollout_generator)
    prompts = PromptBatch(
        input_ids=torch.tensor([[1, 2], [1, 3]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
    )
    generation_config = GenerationConfig(max_new_tokens=1, temperature=1.0, top_k=8, pad_token_id=0)

    for step in range(10):
        stats, _ = trainer.step(prompts, generation_config)
        reward = float(stats.mean_reward.item()) if stats.mean_reward is not None else 0.0
        print(f"step={step:02d} loss={float(stats.loss):.4f} mean_reward={reward:.3f}")


if __name__ == "__main__":
    main()

