import pytest

torch = pytest.importorskip("torch")

from rllm.algorithms.ppo import generalized_advantage_estimation, terminal_rewards_to_token_rewards


def test_terminal_rewards_land_on_last_true_action() -> None:
    action_mask = torch.tensor(
        [
            [False, True, True, False],
            [True, False, False, False],
        ]
    )
    rewards = terminal_rewards_to_token_rewards(torch.tensor([2.0, 3.0]), action_mask)
    expected = torch.tensor(
        [
            [0.0, 0.0, 2.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(rewards, expected)


def test_gae_respects_action_mask_boundaries() -> None:
    rewards = torch.tensor([[0.0, 1.0, 0.0]])
    values = torch.zeros_like(rewards)
    action_mask = torch.tensor([[True, True, False]])

    advantages, returns = generalized_advantage_estimation(
        rewards,
        values,
        action_mask,
        gamma=1.0,
        lam=1.0,
    )

    assert torch.equal(advantages, torch.tensor([[1.0, 1.0, 0.0]]))
    assert torch.equal(returns, advantages)

