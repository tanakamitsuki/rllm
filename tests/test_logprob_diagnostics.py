import copy

import pytest

torch = pytest.importorskip("torch")

from rllm.core.types import GenerationConfig, PromptBatch
from rllm.diagnostics.logprobs import (
    actor_generator_logprob_diff,
    assert_actor_generator_logprobs_close,
)
from rllm.models.torch_causal_lm import TinyCausalLMConfig, build_tiny_actor_critic
from rllm.rewards.rule import RuleRewardProvider
from rllm.rollouts.local import LocalRolloutConfig, LocalRolloutGenerator


def _prompts() -> PromptBatch:
    return PromptBatch(
        input_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
    )


def _reward_provider() -> RuleRewardProvider:
    return RuleRewardProvider(lambda example: float(example.response_ids.numel()))


def test_actor_generator_logprob_diff_is_zero_for_identical_models() -> None:
    actor, _ = build_tiny_actor_critic(
        TinyCausalLMConfig(vocab_size=16, hidden_size=16, num_heads=4, max_position_embeddings=16),
        seed=11,
    )
    generation_actor = copy.deepcopy(actor)
    generator = LocalRolloutGenerator(
        actor,
        _reward_provider(),
        generation_actor=generation_actor,
        config=LocalRolloutConfig(num_generations=2),
    )
    rollouts = generator.generate(
        _prompts(),
        GenerationConfig(max_new_tokens=2, do_sample=False, pad_token_id=0),
    )

    stats = assert_actor_generator_logprobs_close(actor, rollouts)

    assert stats.num_tokens == 8
    assert stats.max_abs_diff.item() == pytest.approx(0.0)
    assert stats.mean_abs_diff.item() == pytest.approx(0.0)


def test_actor_generator_logprob_diff_detects_backend_drift() -> None:
    actor, _ = build_tiny_actor_critic(
        TinyCausalLMConfig(vocab_size=16, hidden_size=16, num_heads=4, max_position_embeddings=16),
        seed=13,
    )
    generation_actor = copy.deepcopy(actor)
    with torch.no_grad():
        generation_actor.lm_head.weight[0, 0].add_(0.5)

    generator = LocalRolloutGenerator(
        actor,
        _reward_provider(),
        generation_actor=generation_actor,
        config=LocalRolloutConfig(num_generations=2),
    )
    rollouts = generator.generate(
        _prompts(),
        GenerationConfig(max_new_tokens=2, do_sample=False, pad_token_id=0),
    )

    stats = actor_generator_logprob_diff(actor, rollouts)

    assert stats.num_tokens == 8
    assert stats.max_abs_diff.item() > 0
    with pytest.raises(AssertionError, match="actor/generator logprobs differ"):
        assert_actor_generator_logprobs_close(actor, rollouts, atol=0.0, rtol=0.0)

