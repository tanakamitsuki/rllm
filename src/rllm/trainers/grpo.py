"""Minimal single-process GRPO trainer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import torch

from rllm.algorithms.grpo import GRPO, GRPOConfig
from rllm.core.interfaces import Actor
from rllm.core.types import AlgorithmStats, GenerationConfig, PromptBatch, RolloutBatch
from rllm.diagnostics.logprobs import assert_actor_generator_logprobs_close
from rllm.rollouts.local import LocalRolloutGenerator


class _Optimizer(Protocol):
    def zero_grad(self, set_to_none: bool = ...) -> None: ...

    def step(self) -> None: ...


@dataclass(frozen=True)
class GRPOTrainerConfig:
    """Training-step options for the local GRPO trainer."""

    max_grad_norm: float | None = 1.0
    verify_generator_logprobs: bool = False
    logprob_atol: float = 1e-6
    logprob_rtol: float = 1e-5


class GRPOTrainer:
    """A compact trainer that wires local rollouts to the GRPO loss."""

    def __init__(
        self,
        actor: Actor,
        optimizer: _Optimizer,
        rollout_generator: LocalRolloutGenerator,
        *,
        algorithm: GRPO | None = None,
        algorithm_config: GRPOConfig | None = None,
        config: GRPOTrainerConfig | None = None,
    ) -> None:
        self.actor = actor
        self.optimizer = optimizer
        self.rollout_generator = rollout_generator
        self.algorithm = algorithm or GRPO(algorithm_config)
        self.config = config or GRPOTrainerConfig()

    def step(
        self,
        prompts: PromptBatch,
        generation_config: GenerationConfig,
    ) -> tuple[AlgorithmStats, RolloutBatch]:
        rollouts = self.rollout_generator.generate(prompts, generation_config)
        logprob_stats = None
        if self.config.verify_generator_logprobs:
            logprob_stats = assert_actor_generator_logprobs_close(
                self.actor,
                rollouts,
                atol=self.config.logprob_atol,
                rtol=self.config.logprob_rtol,
            )
        rollouts = self.algorithm.prepare_rollouts(rollouts)
        new_logprobs = self.actor.logprobs(rollouts.input_ids, rollouts.attention_mask)
        loss, stats = self.algorithm.loss(rollouts, new_logprobs)
        if logprob_stats is not None:
            stats = replace(
                stats,
                extra={
                    **stats.extra,
                    "generator_logprob_max_abs_diff": logprob_stats.max_abs_diff.detach(),
                    "generator_logprob_mean_abs_diff": logprob_stats.mean_abs_diff.detach(),
                    "generator_logprob_rms_diff": logprob_stats.rms_diff.detach(),
                },
            )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm is not None:
            parameters = getattr(self.actor, "parameters", None)
            if parameters is not None:
                torch.nn.utils.clip_grad_norm_(parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return stats, rollouts
