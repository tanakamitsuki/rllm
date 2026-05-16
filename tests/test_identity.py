import pytest

torch = pytest.importorskip("torch")

from rllm.core.identity import assert_bitwise_identical_backbones
from rllm.models.torch_causal_lm import TinyCausalLMConfig, build_tiny_actor_critic


def test_tiny_actor_and_critic_backbones_start_bit_identical() -> None:
    actor, critic = build_tiny_actor_critic(TinyCausalLMConfig(hidden_size=16, num_heads=4), seed=7)
    assert_bitwise_identical_backbones(actor, critic)


def test_identity_check_fails_after_backbone_mutation() -> None:
    actor, critic = build_tiny_actor_critic(TinyCausalLMConfig(hidden_size=16, num_heads=4), seed=7)
    with torch.no_grad():
        first_parameter = next(critic.backbone.parameters())
        first_parameter.view(-1)[0].add_(1.0)

    with pytest.raises(AssertionError, match="values differ"):
        assert_bitwise_identical_backbones(actor, critic)

