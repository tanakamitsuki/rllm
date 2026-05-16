# rllm Initial Design

## Goals

`rllm` starts as a compact RL framework for language models with a PyTorch local
runtime and clear adapter seams for future engines such as Megatron or vLLM. The
first useful path is GRPO on rule rewards. PPO is specified on the same rollout
objects so it can reuse token logprobs, masks, reference KL, and reward plumbing.

## Architecture

- `rllm.core.interfaces` defines the public contracts: `Actor`, `Critic`,
  `ReferencePolicy`, `RewardProvider`, `RolloutGenerator`, and `RLAlgorithm`.
- `rllm.core.types` owns tensor batch structures and trainer stats.
- `rllm.models.torch_causal_lm` contains local PyTorch actor/critic wrappers and
  a tiny causal LM used for cheap validation.
- `rllm.algorithms.grpo` implements grouped reward normalization and GRPO loss.
- `rllm.algorithms.ppo` implements PPO loss and GAE utilities.
- `rllm.rollouts.local` builds single-process rollouts from an actor, reference
  policy, and reward provider.
- `rllm.rewards.rule` adapts deterministic Python reward functions.
- `rllm.diagnostics.logprobs` compares generator-recorded rollout logprobs
  against actor recomputation on the same generated tokens.
- `rllm.diagnostics.hidden_states` compares actor and critic backbone outputs
  before their task-specific heads.

## Actor/Critic Invariant

Actor and critic transformer backbones must be bit-identical at initialization:
same module class, same state-dict keys, same shapes, same dtypes, and
`torch.equal` values. PPO critic-only parameters live outside the backbone, for
example in a value head. This keeps the logits/logprob path maximally stable for
low-precision experiments.

The rollout path has a second numerical invariant: the generator backend records
`old_logprobs`, and the actor must be able to recompute matching logprobs for the
same sampled tokens. This is the early warning check for token alignment,
attention-mask handling, or backend inference drift once rollout generation is
served by a separate engine.

Because the critic has a value head rather than an LM head, actor-versus-critic
`logprobs` are not defined. Their comparable runtime invariant is equality of
shared-backbone hidden states on the same non-padding tokens.

## Algorithm Behavior

GRPO groups multiple responses per prompt. A scalar reward is computed per
response, normalized within the prompt group, then broadcast over generated
tokens through `action_mask`. The loss uses PPO-style ratio clipping and an
optional reference KL term.

PPO consumes the same rollout layout, adds token-level values and returns, and
optimizes clipped policy and value losses. Terminal scalar rewards can be placed
on the final generated token before running GAE.

## Validation Ladder

1. Unit tests for masks, shapes, grouped advantages, PPO GAE, loss finiteness,
   and backbone identity.
2. Tiny causal LM plus deterministic rule reward.
3. Qwen smoke test loading `Qwen/Qwen3-0.6B` with Transformers `>=4.51.0`.
