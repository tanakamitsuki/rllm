"""Hugging Face checkpoint adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from rllm.core.interfaces import Actor
from rllm.core.types import GenerationConfig, PromptBatch
from rllm.models.torch_causal_lm import TorchCausalLMActor
from rllm.utils.logprobs import apply_top_k, token_logprobs


class HFCausalLMActor(nn.Module, Actor):
    """Actor wrapper for `transformers.AutoModelForCausalLM`.

    Hugging Face is used to load checkpoints and tokenizers; logprob accounting
    and the fallback generation loop stay in PyTorch.
    """

    def __init__(self, model: nn.Module, *, backbone_attr: str = "model") -> None:
        super().__init__()
        self.model = model
        self.backbone_attr = backbone_attr

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> "HFCausalLMActor":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        return cls(model)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return token_logprobs(self.forward_logits(input_ids, attention_mask), input_ids)

    @torch.no_grad()
    def generate(self, prompts: PromptBatch, config: GenerationConfig) -> torch.Tensor:
        input_ids = prompts.input_ids.to(self.device)
        attention_mask = prompts.attention_mask.to(self.device)
        finished = torch.zeros(input_ids.shape[0], device=self.device, dtype=torch.bool)

        # Keep a small PyTorch generation loop here rather than delegating to
        # `transformers.generate()`: rollout accounting must use the same token
        # path that later supplies `old_logprobs`.
        for _ in range(config.max_new_tokens):
            logits = self.forward_logits(input_ids, attention_mask)[:, -1, :] / config.temperature
            logits = apply_top_k(logits, config.top_k)
            if config.do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            was_finished = finished
            if config.eos_token_id is not None:
                next_token = torch.where(
                    was_finished,
                    torch.full_like(next_token, config.pad_token_id),
                    next_token,
                )
                finished = finished | (next_token == config.eos_token_id)

            input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)
            attention_mask = torch.cat([attention_mask, (~was_finished)[:, None].to(attention_mask.dtype)], dim=1)
            if config.eos_token_id is not None and bool(finished.all()):
                break
        return input_ids

    def backbone_state_dict(self) -> Mapping[str, torch.Tensor]:
        backbone = getattr(self.model, self.backbone_attr, None)
        if backbone is None:
            raise AttributeError(
                f"model has no `{self.backbone_attr}` backbone attribute; pass the correct backbone_attr"
            )
        return backbone.state_dict()


def actor_from_torch_causal_lm(
    backbone: nn.Module,
    lm_head: nn.Module,
    *,
    pad_token_id: int = 0,
    eos_token_id: int | None = None,
) -> TorchCausalLMActor:
    """Small helper for callers that already own PyTorch modules."""

    return TorchCausalLMActor(
        backbone,
        lm_head,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
