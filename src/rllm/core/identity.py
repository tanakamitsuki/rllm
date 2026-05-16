"""Actor/critic backbone identity checks."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def assert_bitwise_identical_state_dicts(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    *,
    left_name: str = "left",
    right_name: str = "right",
) -> None:
    """Raise when two state dicts are not structurally and bitwise identical."""

    left_keys = list(left.keys())
    right_keys = list(right.keys())
    if left_keys != right_keys:
        missing_left = sorted(set(right_keys) - set(left_keys))
        missing_right = sorted(set(left_keys) - set(right_keys))
        raise AssertionError(
            f"state-dict keys differ: missing from {left_name}={missing_left}, "
            f"missing from {right_name}={missing_right}"
        )

    for key in left_keys:
        left_tensor = left[key]
        right_tensor = right[key]
        if left_tensor.shape != right_tensor.shape:
            raise AssertionError(
                f"{key} shape differs: {tuple(left_tensor.shape)} != {tuple(right_tensor.shape)}"
            )
        if left_tensor.dtype != right_tensor.dtype:
            raise AssertionError(f"{key} dtype differs: {left_tensor.dtype} != {right_tensor.dtype}")
        if left_tensor.device.type != right_tensor.device.type:
            raise AssertionError(
                f"{key} device type differs: {left_tensor.device.type} != {right_tensor.device.type}"
            )
        if not torch.equal(left_tensor.detach().cpu(), right_tensor.detach().cpu()):
            raise AssertionError(f"{key} values differ")


def assert_bitwise_identical_backbones(actor: object, critic: object) -> None:
    """Check actor and critic `backbone_state_dict()` outputs."""

    actor_state = actor.backbone_state_dict()  # type: ignore[attr-defined]
    critic_state = critic.backbone_state_dict()  # type: ignore[attr-defined]
    assert_bitwise_identical_state_dicts(actor_state, critic_state, left_name="actor", right_name="critic")

