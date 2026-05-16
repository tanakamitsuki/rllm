"""Actor/generator logprob consistency diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rllm.core.interfaces import Actor
from rllm.core.types import RolloutBatch


@dataclass(frozen=True)
class LogprobDiffStats:
    """Summary of masked actor-vs-generator logprob differences."""

    max_abs_diff: torch.Tensor
    mean_abs_diff: torch.Tensor
    rms_diff: torch.Tensor
    num_tokens: int

    def as_floats(self) -> dict[str, float | int]:
        return {
            "max_abs_diff": float(self.max_abs_diff.item()),
            "mean_abs_diff": float(self.mean_abs_diff.item()),
            "rms_diff": float(self.rms_diff.item()),
            "num_tokens": self.num_tokens,
        }


def _stats_from_logprobs(
    actor_logprobs: torch.Tensor,
    generator_logprobs: torch.Tensor,
    action_mask: torch.Tensor,
) -> LogprobDiffStats:
    if not bool(action_mask.any()):
        zero = torch.zeros((), device=actor_logprobs.device, dtype=actor_logprobs.dtype)
        return LogprobDiffStats(zero, zero, zero, 0)

    diff = (actor_logprobs - generator_logprobs)[action_mask]
    abs_diff = diff.abs()
    return LogprobDiffStats(
        max_abs_diff=abs_diff.max(),
        mean_abs_diff=abs_diff.mean(),
        rms_diff=diff.square().mean().sqrt(),
        num_tokens=int(action_mask.sum().item()),
    )


@torch.no_grad()
def actor_generator_logprob_diff(actor: Actor, rollouts: RolloutBatch) -> LogprobDiffStats:
    """Compare actor recomputation with generator-recorded rollout logprobs."""

    actor_logprobs = actor.logprobs(
        rollouts.input_ids.to(actor.device),
        rollouts.attention_mask.to(actor.device),
    )
    generator_logprobs = rollouts.old_logprobs.to(actor_logprobs.device)
    action_mask = rollouts.action_mask.to(actor_logprobs.device)
    return _stats_from_logprobs(actor_logprobs, generator_logprobs, action_mask)


def assert_actor_generator_logprobs_close(
    actor: Actor,
    rollouts: RolloutBatch,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> LogprobDiffStats:
    """Assert that actor and generator logprobs match on response tokens."""

    with torch.no_grad():
        actor_logprobs = actor.logprobs(
            rollouts.input_ids.to(actor.device),
            rollouts.attention_mask.to(actor.device),
        )
    generator_logprobs = rollouts.old_logprobs.to(actor_logprobs.device)
    action_mask = rollouts.action_mask.to(actor_logprobs.device)
    stats = _stats_from_logprobs(actor_logprobs, generator_logprobs, action_mask)

    if bool(action_mask.any()):
        actor_tokens = actor_logprobs[action_mask]
        generator_tokens = generator_logprobs[action_mask]
        if not torch.allclose(actor_tokens, generator_tokens, atol=atol, rtol=rtol):
            values = stats.as_floats()
            raise AssertionError(
                "actor/generator logprobs differ on generated tokens: "
                f"max_abs_diff={values['max_abs_diff']:.6g}, "
                f"mean_abs_diff={values['mean_abs_diff']:.6g}, "
                f"rms_diff={values['rms_diff']:.6g}, "
                f"num_tokens={values['num_tokens']}, "
                f"atol={atol}, rtol={rtol}"
            )
    return stats
