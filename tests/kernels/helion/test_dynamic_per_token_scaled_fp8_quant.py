# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the dynamic_per_token_scaled_fp8_quant helion kernel

Run `pytest tests/kernels/helion/test_dynamic_per_token_scaled_fp8_quant.py`.
"""

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.dynamic_per_token_scaled_fp8_quant import (
    _pick_cache,
    baseline,
    dynamic_per_token_scaled_fp8_quant,
    pick_config,
)
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _generate_fake_input(num_tokens: int, hidden_size: int) -> tuple[Any, ...]:
    with FakeTensorMode():
        input = torch.randn(
            num_tokens,
            hidden_size,
            device=current_platform.device_type,
            dtype=torch.bfloat16,
        )
        result = torch.empty(
            input.shape, device=input.device, dtype=current_platform.fp8_dtype()
        )
        scale = torch.empty((num_tokens, 1), device=input.device, dtype=torch.float32)
        scale_ub = torch.mean(input).to(torch.float32)
        args = (result, input, scale, scale_ub)
        return args


def _reference_dynamic_per_token_scaled_fp8_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    scale_ub: torch.Tensor | None,
) -> None:
    """Device-agnostic reference for dynamic_per_token_scaled_fp8_quant.

    baseline() in vllm/kernels/helion/ops/dynamic_per_token_scaled_fp8_quant.py
    calls torch.ops._C.dynamic_per_token_scaled_fp8_quant, which is CUDA-only.
    On platforms without that custom op (e.g. XPU), use QuantFP8's native
    (plain PyTorch) per-token dynamic quantization path instead: it computes
    the same per-token absmax -> optional scale_ub clamp -> divide by fp8 max
    -> clamp to min_scaling_factor -> reciprocal-multiply-and-clamp math as
    both the CUDA kernel and this kernel's own Helion implementation, and it
    is also the exact function vLLM's CustomOp dispatch already uses on XPU
    for this case (QuantFP8.forward_xpu delegates to forward_native for
    non-group, per-token quantization).

    Requires an active VllmConfig context (QuantFP8 is a CustomOp/nn.Module);
    callers must use the `default_vllm_config` pytest fixture.
    """
    quant_op = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
    quantized, computed_scale = quant_op.forward_native(
        input, scale=None, scale_ub=scale_ub
    )
    result.copy_(quantized)
    scale.copy_(computed_scale)


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestDynamicPerTokenScaledFp8QuantConfigPicker:
    def setup_method(self):
        _pick_cache.clear()

    def test_config_picker_exact_match(self):
        config_keys = [
            CaseKey({"hidden_size": 2048, "num_tokens": 16}),
            CaseKey({"hidden_size": 4096, "num_tokens": 16}),
        ]

        args = _generate_fake_input(16, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey({"hidden_size": 4096, "num_tokens": 16})

    def test_config_picker_closest_match(self):
        config_keys = [
            CaseKey({"hidden_size": 2048, "num_tokens": 16}),
            CaseKey({"hidden_size": 2048, "num_tokens": 32}),
            CaseKey({"hidden_size": 4096, "num_tokens": 16}),
            CaseKey({"hidden_size": 4096, "num_tokens": 32}),
        ]

        args = _generate_fake_input(20, 3000)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey({"hidden_size": 2048, "num_tokens": 32})

    def test_config_picker_no_configs(self):
        config_keys: list[dict] = []

        args = _generate_fake_input(16, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            CaseKey({"hidden_size": 2048, "num_tokens": 16}),
            CaseKey({"hidden_size": 4096, "num_tokens": 16}),
        ]

        args = _generate_fake_input(32, 8192)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey({"hidden_size": 4096, "num_tokens": 16})


class TestDynamicPerTokenScaledFp8QuantCorrectness:
    @pytest.mark.parametrize("num_tokens", [1, 7, 4096])
    @pytest.mark.parametrize("hidden_size", [17, 1024, 1025, 1026, 5137, 8193])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
    @pytest.mark.parametrize("has_scale_ub", [True, False])
    @pytest.mark.parametrize("seed", [0])
    def test_dynamic_per_token_fp8_quant(
        self,
        default_vllm_config,
        num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        has_scale_ub: bool,
        seed: int,
    ) -> None:
        skip_if_platform_unsupported("dynamic_per_token_scaled_fp8_quant")
        set_random_seed(seed)
        device = current_platform.device_type

        x = (
            torch.rand(num_tokens, hidden_size, dtype=dtype, device=device) + 1e-6
        )  # avoid nans

        scale_ub = (
            torch.mean(x).to(dtype=torch.float32, device=device)
            if has_scale_ub
            else None
        )

        ref_out = torch.empty(x.shape, device=device, dtype=FP8_DTYPE)
        ref_scales = torch.empty((x.shape[0], 1), device=device, dtype=torch.float32)
        if current_platform.is_cuda():
            baseline(ref_out, x, ref_scales, scale_ub)
        else:
            # baseline() calls torch.ops._C.dynamic_per_token_scaled_fp8_quant,
            # which is CUDA-only; use the portable QuantFP8-based reference on
            # other platforms (e.g. XPU).
            _reference_dynamic_per_token_scaled_fp8_quant(
                ref_out, x, ref_scales, scale_ub
            )

        ops_out = torch.empty(x.shape, device=device, dtype=FP8_DTYPE)
        ops_scales = torch.empty((x.shape[0], 1), device=device, dtype=torch.float32)
        dynamic_per_token_scaled_fp8_quant(ops_out, x, ops_scales, scale_ub)

        torch.testing.assert_close(ref_scales, ops_scales)
        # allow 1 ULP difference
        assert (
            ref_out.view(torch.uint8).to(torch.int16)
            - ops_out.view(torch.uint8).to(torch.int16)
        ).abs().max() <= 1


class TestDynamicPerTokenScaledFp8QuantIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "dynamic_per_token_scaled_fp8_quant" in registered_kernels

        kernel_wrapper = registered_kernels["dynamic_per_token_scaled_fp8_quant"]
        assert kernel_wrapper.op_name == "dynamic_per_token_scaled_fp8_quant"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["result", "scale"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("dynamic_per_token_scaled_fp8_quant")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["dynamic_per_token_scaled_fp8_quant"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_fake_input(16, 4096)
        assert fake_impl(*args) is None
