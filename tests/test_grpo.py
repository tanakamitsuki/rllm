import pytest

torch = pytest.importorskip("torch")

from rllm.algorithms.grpo import GRPO, compute_group_advantages, grpo_loss, reference_kl
from rllm.core.types import RolloutBatch


def _rollout_batch() -> RolloutBatch:
    input_ids = torch.tensor([[1, 2, 3], [1, 4, 5], [6, 7, 0], [6, 8, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0]])
    action_mask = torch.tensor([[True, True], [True, True], [True, False], [True, False]])
    old_logprobs = torch.zeros((4, 2))
    return RolloutBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        action_mask=action_mask,
        old_logprobs=old_logprobs,
        ref_logprobs=torch.zeros_like(old_logprobs),
        rewards=torch.tensor([1.0, 0.0, 0.5, 0.5]),
        prompt_lengths=torch.tensor([1, 1, 1, 1]),
        group_ids=torch.tensor([0, 0, 1, 1]),
    )


def test_group_advantages_normalize_within_prompt_group() -> None:
    advantages = compute_group_advantages(
        torch.tensor([1.0, 0.0, 0.5, 0.5]),
        torch.tensor([0, 0, 1, 1]),
    )
    assert torch.allclose(advantages[:2], torch.tensor([1.0, -1.0]), atol=1e-5)
    assert torch.allclose(advantages[2:], torch.zeros(2), atol=1e-5)


def test_reference_kl_is_zero_when_logprobs_match() -> None:
    logprobs = torch.tensor([[0.1, -0.2]])
    assert torch.allclose(reference_kl(logprobs, logprobs), torch.zeros_like(logprobs))


def test_grpo_loss_is_finite_and_prepares_advantages() -> None:
    rollouts = _rollout_batch()
    algorithm = GRPO()
    prepared = algorithm.prepare_rollouts(rollouts)
    new_logprobs = torch.zeros_like(rollouts.old_logprobs, requires_grad=True)

    loss, stats = grpo_loss(prepared, new_logprobs)

    assert torch.isfinite(loss)
    assert stats.policy_loss is not None
    loss.backward()
    assert new_logprobs.grad is not None

