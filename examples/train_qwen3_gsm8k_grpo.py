"""Short GRPO run on a real GSM8K slice.

Run with:
    python examples/train_qwen3_gsm8k_grpo.py
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoTokenizer

from rllm.algorithms.grpo import GRPOConfig
from rllm.core.types import GenerationConfig, PromptBatch, RolloutBatch
from rllm.models.hf import HFCausalLMActor
from rllm.rewards.rule import RewardExample, RuleRewardProvider
from rllm.rollouts.local import LocalRolloutConfig, LocalRolloutGenerator
from rllm.trainers.grpo import GRPOTrainer, GRPOTrainerConfig


INTEGER_PATTERN = re.compile(r"-?\d[\d,]*")
FINAL_ANSWER_PATTERN = re.compile(r"####\s*(-?\d[\d,]*)")


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    answer: int


@dataclass(frozen=True)
class ResponseScore:
    predicted: int | None
    correct: bool
    reward: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset-id", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-examples", type=int, default=32)
    parser.add_argument("--eval-examples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--print-rollouts",
        type=int,
        default=4,
        help="Number of sampled training responses to print after each step; use 0 to disable.",
    )
    parser.add_argument(
        "--train-scope",
        choices=("lm_head", "all"),
        default="lm_head",
        help="`lm_head` is the fast default; `all` updates every actor parameter.",
    )
    parser.add_argument(
        "--require-signal",
        action="store_true",
        help="Exit with an error if no nonzero GRPO advantage is observed across the run.",
    )
    return parser.parse_args()


def parse_gsm8k_answer(answer_text: str) -> int:
    match = FINAL_ANSWER_PATTERN.search(answer_text)
    if match is None:
        raise ValueError(f"GSM8K answer does not contain a `####` final answer marker: {answer_text!r}")
    return int(match.group(1).replace(",", ""))


def load_gsm8k_split(
    dataset_id: str,
    dataset_config: str,
    split: str,
    *,
    limit: int,
) -> list[GSM8KExample]:
    from datasets import load_dataset

    rows = load_dataset(dataset_id, dataset_config, split=f"{split}[:{limit}]")
    return [
        GSM8KExample(
            question=str(row["question"]),
            answer=parse_gsm8k_answer(str(row["answer"])),
        )
        for row in rows
    ]


def extract_last_integer(text: str) -> int | None:
    matches = INTEGER_PATTERN.findall(text)
    return None if not matches else int(matches[-1].replace(",", ""))


def score_response(response_text: str, answer: int) -> ResponseScore:
    predicted = extract_last_integer(response_text)
    correct = predicted == answer
    return ResponseScore(predicted=predicted, correct=correct, reward=1.0 if correct else 0.0)


def format_prompt(tokenizer: object, question: str) -> str:
    content = (
        "Solve the math word problem. "
        "We only score the final numeric answer, so put the final answer at the end.\n\n"
        f"{question}"
    )
    messages = [{"role": "user", "content": content}]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return content
    try:
        return apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_prompt_batch(
    examples: Iterable[GSM8KExample],
    tokenizer: object,
    *,
    device: str,
) -> PromptBatch:
    rows = list(examples)
    prompt_texts = [format_prompt(tokenizer, row.question) for row in rows]
    encoded = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    metadata = [{"answer": row.answer, "question": row.question} for row in rows]
    return PromptBatch(encoded["input_ids"], encoded["attention_mask"], metadata=metadata).to(device)


def make_reward_provider(tokenizer: object) -> RuleRewardProvider:
    def reward_fn(example: RewardExample) -> float:
        answer = int(example.metadata["answer"])
        response_text = tokenizer.decode(example.response_ids, skip_special_tokens=True).strip()
        return score_response(response_text, answer).reward

    return RuleRewardProvider(reward_fn)


def set_trainable_scope(actor: HFCausalLMActor, scope: str) -> int:
    if scope == "all":
        for parameter in actor.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        output_embeddings = actor.model.get_output_embeddings()
        if output_embeddings is None:
            raise ValueError("model does not expose output embeddings for `lm_head` training")
        for parameter in output_embeddings.parameters():
            parameter.requires_grad_(True)
    return sum(parameter.numel() for parameter in actor.parameters() if parameter.requires_grad)


@torch.no_grad()
def evaluate_exact_match(
    actor: HFCausalLMActor,
    tokenizer: object,
    examples: list[GSM8KExample],
    generation_config: GenerationConfig,
) -> tuple[float, list[tuple[str, str, int, int | None]]]:
    reward_provider = make_reward_provider(tokenizer)
    generator = LocalRolloutGenerator(
        actor,
        reward_provider,
        config=LocalRolloutConfig(num_generations=1),
    )
    correct = 0
    samples: list[tuple[str, str, int, int | None]] = []
    for row in examples:
        prompts = build_prompt_batch([row], tokenizer, device=str(actor.device))
        rollouts = generator.generate(prompts, generation_config)
        response_length = int(rollouts.attention_mask[0].sum().item())
        prompt_length = int(rollouts.prompt_lengths[0].item())
        response_ids = rollouts.input_ids[0, prompt_length:response_length]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        score = score_response(response_text, row.answer)
        correct += int(score.correct)
        samples.append((row.question, response_text, row.answer, score.predicted))
    return correct / len(examples), samples


def print_rollout_samples(
    rollouts: RolloutBatch,
    prompts: PromptBatch,
    tokenizer: object,
    *,
    limit: int,
) -> None:
    if limit <= 0:
        return
    print("train_group_rewards:")
    for group_id in torch.unique(rollouts.group_ids, sorted=True).tolist():
        group_mask = rollouts.group_ids == group_id
        group_rewards = rollouts.rewards[group_mask] if rollouts.rewards is not None else None
        answer = int(prompts.metadata[group_id]["answer"])
        question = str(prompts.metadata[group_id]["question"])
        if group_rewards is None:
            reward_summary = "n/a"
        else:
            reward_summary = (
                f"mean={float(group_rewards.mean().item()):.3f} "
                f"min={float(group_rewards.min().item()):.3f} "
                f"max={float(group_rewards.max().item()):.3f}"
            )
        print(f"- group={group_id} target={answer} rewards={reward_summary} question={question!r}")

    print("train_rollouts:")
    rows_by_group: dict[int, list[int]] = {}
    for row in range(rollouts.batch_size):
        group_id = int(rollouts.group_ids[row].item())
        rows_by_group.setdefault(group_id, []).append(row)

    ordered_rows: list[int] = []
    group_ids = sorted(rows_by_group)
    sample_index = 0
    while len(ordered_rows) < min(limit, rollouts.batch_size):
        added = False
        for group_id in group_ids:
            group_rows = rows_by_group[group_id]
            if sample_index < len(group_rows):
                ordered_rows.append(group_rows[sample_index])
                added = True
                if len(ordered_rows) == min(limit, rollouts.batch_size):
                    break
        if not added:
            break
        sample_index += 1

    for row in ordered_rows:
        group_id = int(rollouts.group_ids[row].item())
        answer = int(prompts.metadata[group_id]["answer"])
        question = str(prompts.metadata[group_id]["question"])
        response_length = int(rollouts.attention_mask[row].sum().item())
        prompt_length = int(rollouts.prompt_lengths[row].item())
        response_ids = rollouts.input_ids[row, prompt_length:response_length]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        score = score_response(response_text, answer)
        reward_text = "n/a" if rollouts.rewards is None else f"{float(rollouts.rewards[row].item()):.1f}"
        generation_index = int(rollouts.metadata[row]["generation_index"])
        print(
            f"- group={group_id} sample={generation_index} "
            f"target={answer} predicted={score.predicted} "
            f"correct={score.correct} reward={reward_text} "
            f"question={question!r} response={response_text!r}"
        )


def sample_batch(
    examples: list[GSM8KExample],
    *,
    batch_size: int,
    rng: random.Random,
) -> list[GSM8KExample]:
    if batch_size >= len(examples):
        return list(examples)
    return rng.sample(examples, k=batch_size)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    actor = HFCausalLMActor.from_pretrained(args.model_id, torch_dtype=dtype).to(args.device)
    actor.eval()
    trainable_parameters = set_trainable_scope(actor, args.train_scope)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    train_examples = load_gsm8k_split(
        args.dataset_id,
        args.dataset_config,
        "train",
        limit=args.train_examples,
    )
    eval_examples = load_gsm8k_split(
        args.dataset_id,
        args.dataset_config,
        "test",
        limit=args.eval_examples,
    )
    reward_provider = make_reward_provider(tokenizer)
    rollout_generator = LocalRolloutGenerator(
        actor,
        reward_provider,
        config=LocalRolloutConfig(num_generations=args.num_generations),
    )
    trainer = GRPOTrainer(
        actor,
        optimizer,
        rollout_generator,
        algorithm_config=GRPOConfig(beta_kl=0.0),
        config=GRPOTrainerConfig(
            max_grad_norm=1.0,
            verify_generator_logprobs=True,
            logprob_atol=1e-5,
            logprob_rtol=1e-5,
        ),
    )
    sample_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    eval_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    before_accuracy, _ = evaluate_exact_match(actor, tokenizer, eval_examples, eval_config)
    print(
        f"model={args.model_id} dataset={args.dataset_id}/{args.dataset_config} "
        f"train_examples={len(train_examples)} eval_examples={len(eval_examples)} "
        f"train_scope={args.train_scope} trainable_params={trainable_parameters:,}"
    )
    print(f"before_exact_match={before_accuracy:.3f}")

    saw_nonzero_advantage = False
    for step in range(args.steps):
        batch = sample_batch(train_examples, batch_size=args.batch_size, rng=rng)
        prompts = build_prompt_batch(batch, tokenizer, device=args.device)
        stats, rollouts = trainer.step(prompts, sample_config)
        diff = float(stats.extra["generator_logprob_max_abs_diff"].item())
        mean_abs_advantage = float(stats.extra["mean_abs_advantage"].item())
        nonzero_advantage_fraction = float(stats.extra["nonzero_advantage_fraction"].item())
        saw_nonzero_advantage = saw_nonzero_advantage or nonzero_advantage_fraction > 0.0
        reward = 0.0 if stats.mean_reward is None else float(stats.mean_reward.item())
        print(
            f"step={step + 1:02d} "
            f"loss={float(stats.loss):.4f} "
            f"mean_reward={reward:.3f} "
            f"mean_abs_advantage={mean_abs_advantage:.3f} "
            f"nonzero_advantage_fraction={nonzero_advantage_fraction:.3f} "
            f"generator_logprob_max_abs_diff={diff:.3e}"
        )
        print_rollout_samples(rollouts, prompts, tokenizer, limit=args.print_rollouts)

    after_accuracy, samples = evaluate_exact_match(actor, tokenizer, eval_examples, eval_config)
    print(f"after_exact_match={after_accuracy:.3f}")
    print("sample_generations:")
    for question, response, answer, predicted in samples[: min(4, len(samples))]:
        print(f"- target={answer} predicted={predicted} question={question!r} response={response!r}")

    if args.require_signal and not saw_nonzero_advantage:
        raise RuntimeError(
            "no nonzero GRPO advantage observed; increase `--num-generations`, "
            "temperature, or choose a broader training slice"
        )


if __name__ == "__main__":
    main()

