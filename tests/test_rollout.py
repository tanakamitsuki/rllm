import copy

import pytest

torch = pytest.importorskip("torch")

from rllm.core.types import GenerationConfig, PromptBatch
from rllm.models.torch_causal_lm import (
    FrozenReferencePolicy,
    TinyCausalLMConfig,
    build_tiny_actor_critic,
)
from rllm.rewards.rule import RewardExample, RuleRewardProvider
from rllm.rollouts.local import LocalRolloutConfig, LocalRolloutGenerator


def test_local_rollout_generator_builds_grouped_rollouts() -> None:
    actor, _ = build_tiny_actor_critic(
        TinyCausalLMConfig(vocab_size=16, hidden_size=16, num_heads=4, max_position_embeddings=16),
        seed=3,
    )
    reference = FrozenReferencePolicy(actor)

    def reward_fn(example: RewardExample) -> float:
        return float(example.response_ids.numel())

    generator = LocalRolloutGenerator(
        actor,
        RuleRewardProvider(reward_fn),
        reference_policy=reference,
        config=LocalRolloutConfig(num_generations=2),
    )
    prompts = PromptBatch(
        input_ids=torch.tensor([[1, 2], [3, 0]]),
        attention_mask=torch.tensor([[1, 1], [1, 0]]),
    )

    rollouts = generator.generate(
        prompts,
        GenerationConfig(max_new_tokens=2, do_sample=False, pad_token_id=0),
    )

    assert rollouts.batch_size == 4
    assert rollouts.rewards is not None
    assert rollouts.ref_logprobs is not None
    assert torch.equal(rollouts.group_ids, torch.tensor([0, 0, 1, 1]))
    assert torch.equal(rollouts.response_lengths, torch.tensor([2, 2, 2, 2]))


def test_local_rollout_generator_records_generation_actor_logprobs() -> None:
    actor, _ = build_tiny_actor_critic(
        TinyCausalLMConfig(vocab_size=16, hidden_size=16, num_heads=4, max_position_embeddings=16),
        seed=5,
    )
    generation_actor = copy.deepcopy(actor)
    with torch.no_grad():
        generation_actor.lm_head.weight[0, 0].add_(0.5)
    generator = LocalRolloutGenerator(
        actor,
        RuleRewardProvider(lambda example: 0.0),
        generation_actor=generation_actor,
        config=LocalRolloutConfig(num_generations=1),
    )
    prompts = PromptBatch(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.tensor([[1, 1]]),
    )

    rollouts = generator.generate(
        prompts,
        GenerationConfig(max_new_tokens=1, do_sample=False, pad_token_id=0),
    )

    expected = generation_actor.logprobs(rollouts.input_ids, rollouts.attention_mask)
    actor_recomputed = actor.logprobs(rollouts.input_ids, rollouts.attention_mask)
    assert torch.allclose(rollouts.old_logprobs, expected)
    assert not torch.allclose(rollouts.old_logprobs, actor_recomputed)
