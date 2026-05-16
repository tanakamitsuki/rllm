"""Actor/critic backbone hidden-state consistency diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rllm.core.interfaces import Actor, Critic


@dataclass(frozen=True)
class HiddenStateDiffStats:
    """Summary of actor-vs-critic backbone hidden-state differences."""

    max_abs_diff: torch.Tensor
    mean_abs_diff: torch.Tensor
    rms_diff: torch.Tensor
    num_values: int

    def as_floats(self) -> dict[str, float | int]:
        return {
            "max_abs_diff": float(self.max_abs_diff.item()),
            "mean_abs_diff": float(self.mean_abs_diff.item()),
            "rms_diff": float(self.rms_diff.item()),
            "num_values": self.num_values,
        }


@torch.no_grad()
def actor_critic_hidden_state_diff(
    actor: Actor,
    critic: Critic,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> HiddenStateDiffStats:
    """Compare actor and critic backbone outputs before their separate heads."""

    actor_hidden = actor.forward_hidden_states(
        input_ids.to(actor.device),
        attention_mask.to(actor.device),
    )
    critic_hidden = critic.forward_hidden_states(
        input_ids.to(critic.device),
        attention_mask.to(critic.device),
    ).to(actor_hidden.device)
    attention_mask = attention_mask.to(actor_hidden.device).bool()
    hidden_mask = attention_mask[..., None].expand_as(actor_hidden)
    diff = (actor_hidden - critic_hidden)[hidden_mask]
    abs_diff = diff.abs()
    return HiddenStateDiffStats(
        max_abs_diff=abs_diff.max(),
        mean_abs_diff=abs_diff.mean(),
        rms_diff=diff.square().mean().sqrt(),
        num_values=int(hidden_mask.sum().item()),
    )


def assert_actor_critic_hidden_states_close(
    actor: Actor,
    critic: Critic,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> HiddenStateDiffStats:
    """Assert that actor and critic backbones produce matching hidden states."""

    with torch.no_grad():
        actor_hidden = actor.forward_hidden_states(
            input_ids.to(actor.device),
            attention_mask.to(actor.device),
        )
        critic_hidden = critic.forward_hidden_states(
            input_ids.to(critic.device),
            attention_mask.to(critic.device),
        ).to(actor_hidden.device)
    hidden_mask = attention_mask.to(actor_hidden.device).bool()[..., None].expand_as(actor_hidden)
    stats = actor_critic_hidden_state_diff(actor, critic, input_ids, attention_mask)
    if not torch.allclose(actor_hidden[hidden_mask], critic_hidden[hidden_mask], atol=atol, rtol=rtol):
        values = stats.as_floats()
        raise AssertionError(
            "actor/critic hidden states differ on non-padding tokens: "
            f"max_abs_diff={values['max_abs_diff']:.6g}, "
            f"mean_abs_diff={values['mean_abs_diff']:.6g}, "
            f"rms_diff={values['rms_diff']:.6g}, "
            f"num_values={values['num_values']}, "
            f"atol={atol}, rtol={rtol}"
        )
    return stats

