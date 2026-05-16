import importlib.util
import sys
from pathlib import Path

import pytest


pytest.importorskip("torch")
pytest.importorskip("transformers")


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "train_qwen3_gsm8k_grpo.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_qwen3_gsm8k_grpo", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_gsm8k_answer_reads_final_marker() -> None:
    module = _load_module()
    assert module.parse_gsm8k_answer("Some work here.\n#### 1,234") == 1234


def test_score_response_requires_marked_final_answer() -> None:
    module = _load_module()
    score = module.score_response("He starts with 3 and ends with 42.\n#### 42", 42)
    assert score.predicted == 42
    assert score.has_final_marker is True
    assert score.correct is True
    assert score.reward == 1.0


def test_score_response_ignores_unmarked_numbers() -> None:
    module = _load_module()
    score = module.score_response("He starts with 3 and ends with 42.", 42)
    assert score.predicted is None
    assert score.has_final_marker is False
    assert score.correct is False
    assert score.reward == 0.0
