import pytest

torch = pytest.importorskip("torch")

from rllm.utils.logprobs import masked_mean, token_logprobs


def test_token_logprobs_are_label_aligned() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 2.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        ]
    )
    input_ids = torch.tensor([[0, 1, 2]])

    logprobs = token_logprobs(logits, input_ids)

    expected_0 = torch.log_softmax(logits[:, 0, :], dim=-1)[0, 1]
    expected_1 = torch.log_softmax(logits[:, 1, :], dim=-1)[0, 2]
    assert logprobs.shape == (1, 2)
    assert torch.allclose(logprobs, torch.stack([expected_0, expected_1]).unsqueeze(0))


def test_masked_mean_ignores_false_positions() -> None:
    values = torch.tensor([[1.0, 100.0], [3.0, 100.0]])
    mask = torch.tensor([[True, False], [True, False]])
    assert masked_mean(values, mask).item() == pytest.approx(2.0)

