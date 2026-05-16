# rllm

`rllm` is a small PyTorch-first reinforcement learning framework for language
models. The first implementation target is single-GPU RLVR/RLHF experimentation:
GRPO is the primary path, PPO shares the same rollout and log-probability
infrastructure, and Hugging Face is used only for tokenizer/config/checkpoint
loading.

## Current Scope

- Abstract interfaces for actors, critics, reference policies, reward providers,
  rollout generators, and algorithms.
- PyTorch local backends for causal language models.
- A tiny causal LM for cheap algorithm validation.
- GRPO loss, grouped reward normalization, and reference-KL penalty.
- PPO loss and generalized advantage estimation utilities.
- Bitwise identity checks for actor and critic transformer backbones.
- Actor-versus-generator logprob drift diagnostics for rollout consistency.
- End-to-end GRPO trainer coverage on deterministic and sampled tiny rule-reward tasks.
- Examples for a tiny rule-reward GRPO run and a Qwen3 smoke test.

The initial Qwen target is `Qwen/Qwen3-0.6B`. Transformers `>=4.51.0` is required
for Qwen3 model support.

## Install

```powershell
python -m pip install -e .[dev]
```

For the Qwen smoke test, install the optional runtime helpers too:

```powershell
python -m pip install -e .[dev,qwen]
```

## Quick Shape

```python
import torch

from rllm.algorithms.grpo import GRPOConfig, compute_group_advantages, grpo_loss
from rllm.core.types import RolloutBatch

rewards = torch.tensor([1.0, 0.0, 0.5, 0.5])
group_ids = torch.tensor([0, 0, 1, 1])
advantages = compute_group_advantages(rewards, group_ids)
```

## Validation

```powershell
python -m pytest
python examples/tiny_grpo_rule.py
```

To verify rollout-generator and actor logprobs on the same sampled tokens:

```python
from rllm.diagnostics.logprobs import actor_generator_logprob_diff

rollouts = rollout_generator.generate(prompts, generation_config)
stats = actor_generator_logprob_diff(actor, rollouts)
print(stats.as_floats())
```

For training-time enforcement, enable `GRPOTrainerConfig(verify_generator_logprobs=True)`.

The Qwen smoke test is intentionally explicit because it may download a large
checkpoint:

```powershell
python examples/qwen3_smoke.py --model-id Qwen/Qwen3-0.6B
```

## Design Notes

The actor and critic are allowed to have different output heads, but their
transformer backbones must be identical at initialization. For PPO, value
estimation is isolated in an external value projection so logits/logprobs can be
compared across actor, reference, and critic-backbone paths without hidden
structural drift.
