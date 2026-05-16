# Learning Path

This guide is a reading order for `rllm`, not just a file index. Follow it from
top to bottom the first time. Each step gives you one idea to understand, the
files to read, and a small check that proves the idea is real.

## 0. Build the mental model

`rllm` has one main loop:

1. Turn prompts into several sampled responses.
2. Score each response with a reward provider.
3. Convert rewards into algorithm-specific training targets.
4. Recompute current actor logprobs on the sampled tokens.
5. Backpropagate a policy loss and update the actor.

The first complete path is GRPO:

```text
PromptBatch
  -> LocalRolloutGenerator
  -> RolloutBatch with rewards and old_logprobs
  -> GRPO.prepare_rollouts()
  -> actor.logprobs()
  -> GRPO.loss()
  -> GRPOTrainer.step()
```

Read [design.md](design.md) once before diving into code. It explains why the
actor and critic backbones are kept identical and why generator/actor logprob
checks matter.

## 1. Learn the tensor layout first

Read:

- `src/rllm/core/types.py`
- `src/rllm/utils/logprobs.py`
- `tests/test_logprobs.py`
- `tests/test_rollout.py`

What to learn:

- `PromptBatch` is just padded input text plus optional metadata.
- `RolloutBatch.input_ids` has shape `[batch, seq_len]`.
- Token logprobs have shape `[batch, seq_len - 1]` because each position scores
  the next token.
- `action_mask` lives in that label-aligned logprob space, not raw token space.

Why this matters:

Most RL-on-language bugs are alignment bugs. If the first generated token is
masked at the wrong column, every later loss can look valid while training the
wrong thing.

Try:

```powershell
python -m pytest tests/test_logprobs.py tests/test_rollout.py
```

Then sketch one prompt with two generated tokens on paper and mark which
`action_mask` columns should be true.

## 2. See how components stay swappable

Read:

- `src/rllm/core/interfaces.py`
- `src/rllm/models/torch_causal_lm.py`
- `src/rllm/models/hf.py`
- `src/rllm/rewards/rule.py`

What to learn:

- The trainer depends on `Actor`, `RewardProvider`, and `RolloutGenerator`
  interfaces, not one concrete backend.
- The tiny PyTorch model is for fast validation.
- The Hugging Face actor is only an adapter around loaded checkpoints; the
  rollout accounting still stays inside `rllm`.
- `RuleRewardProvider` turns a simple Python function into a reward backend.

Try:

```powershell
python examples/tiny_grpo_rule.py
```

While reading its output, identify which object owns generation, which object
owns reward scoring, and which object owns optimization.

## 3. Follow one rollout from prompt to reward

Read:

- `src/rllm/rollouts/local.py`
- `src/rllm/rewards/rule.py`
- `tests/test_rollout.py`

What to learn:

- One input prompt becomes one GRPO group.
- `num_generations` creates sibling responses for that prompt.
- `old_logprobs` come from the actor that actually sampled the response.
- Rewards are attached after the `RolloutBatch` exists so the reward provider can
  inspect prompt metadata and response tokens together.

Questions to answer before moving on:

1. Why does `group_id` repeat across several rollout rows?
2. Why is the first generated token masked at `prompt_length - 1`?
3. Why is `generation_actor` allowed to differ from `actor`?

## 4. Understand GRPO before touching PPO

Read:

- `src/rllm/algorithms/grpo.py`
- `tests/test_grpo.py`
- `tests/test_trainer_grpo.py`

What to learn:

- GRPO normalizes rewards within each prompt group.
- If all sibling responses receive the same reward, the group has zero relative
  advantage and produces no policy signal.
- Sequence-level advantages are broadcast across generated tokens only.
- The policy objective is a clipped ratio loss plus optional reference KL.

Try:

```python
import torch
from rllm.algorithms.grpo import compute_group_advantages

rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
group_ids = torch.tensor([0, 0, 1, 1])
print(compute_group_advantages(rewards, group_ids))
```

Expected lesson:

- Group `0` has a useful comparison.
- Group `1` has no useful comparison because both responses tie.

## 5. Read the trainer as orchestration, not magic

Read:

- `src/rllm/trainers/grpo.py`
- `src/rllm/diagnostics/logprobs.py`
- `tests/test_trainer_grpo.py`
- `tests/test_logprob_diagnostics.py`

What to learn:

- `GRPOTrainer.step()` is intentionally short.
- It collects rollouts, optionally verifies actor/generator logprob equality,
  asks GRPO to attach advantages, recomputes current actor logprobs, and updates.
- Generator logprob diagnostics are there to catch backend drift before you trust
  a rollout engine such as vLLM or another inference stack.

Try:

```python
from rllm.diagnostics.logprobs import actor_generator_logprob_diff

rollouts = rollout_generator.generate(prompts, generation_config)
print(actor_generator_logprob_diff(actor, rollouts).as_floats())
```

If the same backend generated and rescored the tokens, the differences should be
zero or extremely close to zero.

## 6. Learn the backbone invariants

Read:

- `src/rllm/core/identity.py`
- `src/rllm/diagnostics/hidden_states.py`
- `tests/test_identity.py`
- `tests/test_hidden_state_diagnostics.py`

What to learn:

- Actor and critic backbones must start bit-identical.
- PPO can add a value head without changing the shared transformer core.
- Because actor and critic heads differ, hidden states are the right runtime
  object to compare, not actor-vs-critic logprobs.

Try:

```powershell
python -m pytest tests/test_identity.py tests/test_hidden_state_diagnostics.py
```

Then intentionally perturb one critic backbone weight in a scratch experiment
and confirm that the hidden-state diff becomes nonzero.

## 7. Move from toy training to real training

Read:

- `examples/train_qwen3_arithmetic_grpo.py`
- `examples/train_qwen3_gsm8k_grpo.py`
- `tests/test_gsm8k_example.py`

Run:

```powershell
python examples/train_qwen3_arithmetic_grpo.py
python examples/train_qwen3_gsm8k_grpo.py --require-signal
```

What to watch:

- `mean_reward`
- `mean_abs_advantage`
- `nonzero_advantage_fraction`
- `generator_logprob_max_abs_diff`
- before/after exact match

Interpretation:

- `loss ~= 0` is not automatically bad in GRPO.
- `mean_abs_advantage = 0` means a sampled group has no relative reward signal.
- `generator_logprob_max_abs_diff = 0` means rollout generation and actor
  rescoring agree on the sampled tokens.
- On GSM8K, reward is granted only when the response contains the requested
  `#### <integer>` final-answer marker. This keeps the validation honest.

## 8. Only then study PPO

Read:

- `src/rllm/algorithms/ppo.py`
- `tests/test_ppo.py`

What to learn:

- PPO reuses the same rollout tensor layout.
- Scalar terminal rewards are placed on the final generated token.
- GAE walks backward through generated tokens to compute advantages and returns.
- The critic-specific value head is extra structure outside the actor-identical
  backbone.

At this point, compare GRPO and PPO in your own words:

- What data do both algorithms share?
- What extra tensors does PPO need?
- What is the engineering cost of introducing the critic?

## 9. Suggested experiments

Once the reading path feels comfortable, these are good next exercises:

1. Add a `marker_rate` metric to the GSM8K example so formatting quality and
   answer quality are visible separately.
2. Add a new deterministic rule-reward task that is not arithmetic.
3. Implement a second rollout backend that calls the same actor interface but
   stores `old_logprobs` from a different inference path, then use the
   diagnostics to prove equivalence.
4. Add a PPO trainer that mirrors the shape of `GRPOTrainer`.
5. Replace text metadata with a small multimodal placeholder batch and note
   which parts of the algorithm code remain unchanged.

## 10. A compact reading order

If you later want a quick refresher, use this order:

1. `docs/design.md`
2. `src/rllm/core/types.py`
3. `src/rllm/rollouts/local.py`
4. `src/rllm/algorithms/grpo.py`
5. `src/rllm/trainers/grpo.py`
6. `src/rllm/models/torch_causal_lm.py`
7. `src/rllm/diagnostics/logprobs.py`
8. `examples/train_qwen3_gsm8k_grpo.py`
9. `src/rllm/algorithms/ppo.py`

That sequence follows the actual data flow, which is usually the fastest way to
make a framework feel small enough to hold in your head.
