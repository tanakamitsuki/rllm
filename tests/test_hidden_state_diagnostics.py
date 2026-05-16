import pytest

torch = pytest.importorskip("torch")

from rllm.diagnostics.hidden_states import (
    actor_critic_hidden_state_diff,
    assert_actor_critic_hidden_states_close,
)
from rllm.models.torch_causal_lm import TinyCausalLMConfig, build_tiny_actor_critic


def test_actor_critic_hidden_states_match_for_identical_backbones() -> None:
    actor, critic = build_tiny_actor_critic(TinyCausalLMConfig(hidden_size=16, num_heads=4), seed=17)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)

    stats = assert_actor_critic_hidden_states_close(actor, critic, input_ids, attention_mask)

    assert stats.max_abs_diff.item() == pytest.approx(0.0)
    assert stats.mean_abs_diff.item() == pytest.approx(0.0)


def test_actor_critic_hidden_state_diff_detects_backbone_drift() -> None:
    actor, critic = build_tiny_actor_critic(TinyCausalLMConfig(hidden_size=16, num_heads=4), seed=19)
    with torch.no_grad():
        critic.backbone.embed_tokens.weight[1, 0].add_(0.5)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    stats = actor_critic_hidden_state_diff(actor, critic, input_ids, attention_mask)

    assert stats.max_abs_diff.item() > 0
    with pytest.raises(AssertionError, match="actor/critic hidden states differ"):
        assert_actor_critic_hidden_states_close(actor, critic, input_ids, attention_mask, atol=0.0, rtol=0.0)
