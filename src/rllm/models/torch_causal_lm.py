"""Local PyTorch causal LM backends used for validation and small runs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from rllm.core.interfaces import Actor, Critic, ReferencePolicy
from rllm.core.types import GenerationConfig, PromptBatch
from rllm.utils.logprobs import apply_top_k, token_logprobs


@dataclass(frozen=True)
class TinyCausalLMConfig:
    """Configuration for the validation-only tiny causal LM."""

    vocab_size: int = 32
    hidden_size: int = 32
    num_layers: int = 2
    num_heads: int = 4
    max_position_embeddings: int = 128
    dropout: float = 0.0
    pad_token_id: int = 0
    eos_token_id: int | None = None


class TinyTransformerBackbone(nn.Module):
    """A compact causal transformer backbone with no LM or value head."""

    def __init__(self, config: TinyCausalLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_heads,
                    dim_feedforward=config.hidden_size * 4,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.embed_tokens(input_ids) + self.embed_positions(positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = attention_mask == 0
        for layer in self.layers:
            hidden = layer(hidden, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return self.norm(hidden)


class TorchCausalLMActor(nn.Module, Actor):
    """Actor wrapper around a PyTorch backbone and language-model head."""

    def __init__(
        self,
        backbone: nn.Module,
        lm_head: nn.Module,
        *,
        pad_token_id: int = 0,
        eos_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.lm_head = lm_head
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_hidden_states(input_ids, attention_mask)
        return self.lm_head(hidden)

    def forward_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids, attention_mask)

    def logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(input_ids, attention_mask)
        return token_logprobs(logits, input_ids)

    @torch.no_grad()
    def generate(self, prompts: PromptBatch, config: GenerationConfig) -> torch.Tensor:
        input_ids = prompts.input_ids.to(self.device)
        attention_mask = prompts.attention_mask.to(self.device)
        finished = torch.zeros(input_ids.shape[0], device=self.device, dtype=torch.bool)
        eos_token_id = config.eos_token_id if config.eos_token_id is not None else self.eos_token_id
        pad_token_id = config.pad_token_id if config.pad_token_id is not None else self.pad_token_id

        for _ in range(config.max_new_tokens):
            logits = self.forward_logits(input_ids, attention_mask)[:, -1, :] / config.temperature
            logits = apply_top_k(logits, config.top_k)
            if config.do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            was_finished = finished
            if eos_token_id is not None:
                # Finished rows append padding while unfinished rows append the
                # newly sampled token. Their attention entries become zero on
                # later steps, preserving a rectangular batch without treating
                # post-EOS padding as model actions.
                next_token = torch.where(was_finished, torch.full_like(next_token, pad_token_id), next_token)
                finished = finished | (next_token == eos_token_id)

            input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)
            attention_mask = torch.cat([attention_mask, (~was_finished)[:, None].to(attention_mask.dtype)], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break

        return input_ids

    def backbone_state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.backbone.state_dict()


class TorchCausalLMCritic(nn.Module, Critic):
    """Critic with an actor-identical backbone and a separate value head."""

    def __init__(self, backbone: nn.Module, value_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = value_head

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def values(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_hidden_states(input_ids, attention_mask)
        return self.value_head(hidden[:, :-1, :]).squeeze(-1)

    def forward_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids, attention_mask)

    def backbone_state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.backbone.state_dict()


class FrozenReferencePolicy(nn.Module, ReferencePolicy):
    """Frozen copy of an actor used for reference logprobs."""

    def __init__(self, actor: TorchCausalLMActor) -> None:
        super().__init__()
        self.actor = copy.deepcopy(actor)
        self.actor.eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return self.actor.device

    @torch.no_grad()
    def logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.actor.logprobs(input_ids.to(self.device), attention_mask.to(self.device))


def build_tiny_actor_critic(
    config: TinyCausalLMConfig | None = None,
    *,
    seed: int | None = None,
) -> tuple[TorchCausalLMActor, TorchCausalLMCritic]:
    """Build an actor/critic pair with bit-identical backbones."""

    if seed is not None:
        torch.manual_seed(seed)
    config = config or TinyCausalLMConfig()
    actor_backbone = TinyTransformerBackbone(config)
    actor = TorchCausalLMActor(
        actor_backbone,
        nn.Linear(config.hidden_size, config.vocab_size, bias=False),
        pad_token_id=config.pad_token_id,
        eos_token_id=config.eos_token_id,
    )
    # Deep-copy the already initialized backbone instead of reinitializing a new
    # one. This gives the critic the same parameter names, dtypes, shapes, and
    # values at bit level while still allowing a distinct value head.
    critic_backbone = copy.deepcopy(actor_backbone)
    critic = TorchCausalLMCritic(critic_backbone, nn.Linear(config.hidden_size, 1))
    return actor, critic
