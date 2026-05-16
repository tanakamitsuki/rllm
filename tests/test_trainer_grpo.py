import pytest

torch = pytest.importorskip("torch")
from torch import nn

from rllm.algorithms.grpo import GRPO, GRPOConfig
from rllm.core.interfaces import Actor
from rllm.core.types import GenerationConfig, PromptBatch
from rllm.rewards.rule import RewardExample, RuleRewardProvider
from rllm.rollouts.local import LocalRolloutConfig, LocalRolloutGenerator
from rllm.trainers.grpo import GRPOTrainer, GRPOTrainerConfig
from rllm.utils.logprobs import token_logprobs


class ScriptedBanditActor(nn.Module, Actor):
    """One-step actor that emits a fixed action pattern for stable trainer tests."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self._script = [1, 2, 1, 2]
        self._cursor = 0

    @property
    def device(self) -> torch.device:
        return self.logits.device

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.logits.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)

    def logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return token_logprobs(self.forward_logits(input_ids, attention_mask), input_ids)

    @torch.no_grad()
    def generate(self, prompts: PromptBatch, config: GenerationConfig) -> torch.Tensor:
        action = self._script[self._cursor % len(self._script)]
        self._cursor += 1
        action_ids = torch.full(
            (prompts.batch_size, 1),
            action,
            device=prompts.device,
            dtype=torch.long,
        )
        return torch.cat([prompts.input_ids, action_ids], dim=1)

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return {}


class SamplingBanditActor(ScriptedBanditActor):
    """One-step categorical policy that samples from its own logits."""

    @torch.no_grad()
    def generate(self, prompts: PromptBatch, config: GenerationConfig) -> torch.Tensor:
        probs = torch.softmax(self.logits, dim=0)
        action_ids = torch.multinomial(probs, num_samples=prompts.batch_size, replacement=True).to(
            device=prompts.device,
            dtype=torch.long,
        )
        return torch.cat([prompts.input_ids, action_ids[:, None]], dim=1)


def test_grpo_trainer_step_updates_policy_toward_rewarded_action() -> None:
    actor = ScriptedBanditActor()

    def reward_fn(example: RewardExample) -> float:
        return 1.0 if int(example.response_ids[0]) == 1 else 0.0

    rollout_generator = LocalRolloutGenerator(
        actor,
        RuleRewardProvider(reward_fn),
        config=LocalRolloutConfig(num_generations=4),
    )
    trainer = GRPOTrainer(
        actor,
        torch.optim.SGD(actor.parameters(), lr=0.5),
        rollout_generator,
        algorithm=GRPO(GRPOConfig(beta_kl=0.0)),
        config=GRPOTrainerConfig(max_grad_norm=None),
    )
    prompts = PromptBatch(
        input_ids=torch.tensor([[0]], dtype=torch.long),
        attention_mask=torch.tensor([[1]], dtype=torch.long),
    )
    generation_config = GenerationConfig(max_new_tokens=1, do_sample=False, pad_token_id=0)

    before = torch.softmax(actor.logits.detach(), dim=0)[1].item()
    stats, rollouts = trainer.step(prompts, generation_config)
    after = torch.softmax(actor.logits.detach(), dim=0)[1].item()

    assert rollouts.advantages is not None
    assert torch.equal(rollouts.rewards, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert stats.mean_reward is not None
    assert stats.mean_reward.item() == pytest.approx(0.5)
    assert after > before


def test_grpo_trainer_step_changes_actor_parameters() -> None:
    actor = ScriptedBanditActor()
    rollout_generator = LocalRolloutGenerator(
        actor,
        RuleRewardProvider(lambda example: 1.0 if int(example.response_ids[0]) == 1 else 0.0),
        config=LocalRolloutConfig(num_generations=4),
    )
    trainer = GRPOTrainer(
        actor,
        torch.optim.SGD(actor.parameters(), lr=0.5),
        rollout_generator,
        algorithm=GRPO(GRPOConfig(beta_kl=0.0)),
        config=GRPOTrainerConfig(max_grad_norm=1.0),
    )
    prompts = PromptBatch(
        input_ids=torch.tensor([[0]], dtype=torch.long),
        attention_mask=torch.tensor([[1]], dtype=torch.long),
    )
    generation_config = GenerationConfig(max_new_tokens=1, do_sample=False, pad_token_id=0)

    before = actor.logits.detach().clone()
    trainer.step(prompts, generation_config)

    assert not torch.equal(actor.logits.detach(), before)


def test_grpo_trainer_can_verify_generator_logprobs() -> None:
    actor = ScriptedBanditActor()
    generation_actor = ScriptedBanditActor()
    rollout_generator = LocalRolloutGenerator(
        actor,
        RuleRewardProvider(lambda example: 1.0 if int(example.response_ids[0]) == 1 else 0.0),
        generation_actor=generation_actor,
        config=LocalRolloutConfig(num_generations=4),
    )
    trainer = GRPOTrainer(
        actor,
        torch.optim.SGD(actor.parameters(), lr=0.5),
        rollout_generator,
        algorithm=GRPO(GRPOConfig(beta_kl=0.0)),
        config=GRPOTrainerConfig(
            max_grad_norm=None,
            verify_generator_logprobs=True,
            logprob_atol=0.0,
            logprob_rtol=0.0,
        ),
    )
    prompts = PromptBatch(
        input_ids=torch.tensor([[0]], dtype=torch.long),
        attention_mask=torch.tensor([[1]], dtype=torch.long),
    )
    generation_config = GenerationConfig(max_new_tokens=1, do_sample=False, pad_token_id=0)

    trainer.step(prompts, generation_config)


@pytest.mark.slow
def test_grpo_trainer_improves_rewarded_action_probability() -> None:
    torch.manual_seed(0)
    actor = SamplingBanditActor()
    rollout_generator = LocalRolloutGenerator(
        actor,
        RuleRewardProvider(lambda example: 1.0 if int(example.response_ids[0]) == 1 else 0.0),
        config=LocalRolloutConfig(num_generations=64),
    )
    trainer = GRPOTrainer(
        actor,
        torch.optim.SGD(actor.parameters(), lr=0.5),
        rollout_generator,
        algorithm=GRPO(GRPOConfig(beta_kl=0.0)),
        config=GRPOTrainerConfig(max_grad_norm=None),
    )
    prompts = PromptBatch(
        input_ids=torch.tensor([[0]], dtype=torch.long),
        attention_mask=torch.tensor([[1]], dtype=torch.long),
    )
    generation_config = GenerationConfig(max_new_tokens=1, do_sample=True, pad_token_id=0)

    before = torch.softmax(actor.logits.detach(), dim=0)[1].item()
    for _ in range(20):
        trainer.step(prompts, generation_config)
    after = torch.softmax(actor.logits.detach(), dim=0)[1].item()

    assert after > before + 0.20
    assert after > 0.60
