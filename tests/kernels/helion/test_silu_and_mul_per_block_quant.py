# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the silu_and_mul_per_block_quant helion kernel
Run `pytest tests/kernels/helion/test_silu_and_mul_per_block_quant.py`.

XPU known issue, PRECISE root cause (see below for how this was narrowed
down from an earlier, less precise "eager bucketing" theory):
TestSiluAndMulPerBlockQuantCorrectness cases with is_scale_transposed=False
and num_tokens in (7, 4096) currently fail when run after num_tokens=1 has
already triggered compilation for the same specialized (hidden_size,
group_size, quant_dtype, is_scale_transposed) combination.

Confirmed root cause (via a side-by-side diff of
BoundKernel.to_triton_code() for a num_tokens=1 bind vs. a num_tokens=7
bind, same specialized combination): when Helion's FIRST compile for a
given specialized combination happens to see num_tokens=1 -- which
coincides with this kernel's hl.tile([num_tokens, ...], block_size=[1,
None, group_size])'s FIXED block_size=1 for that dimension -- Helion fully
specializes num_tokens as a compile-time constant (`_BLOCK_SIZE_0 =
tl.constexpr(1)`, `num_blocks_0 = 1`, and `num_tokens` is dropped from the
compiled kernel's parameter list entirely) instead of keeping it symbolic.
Subsequent calls with a *different* num_tokens then reuse this
now-permanently-1-sized compiled kernel (vLLM's config/key only
discriminates on `hl.specialize()`'d params, and num_tokens intentionally
isn't one), silently computing only the first token's row correctly.

This contradicts CompileEnvironment's own stated intent
(helion/_compiler/compile_environment.py: "For dynamic kernels, keep 0/1
tensor dimensions symbolic so a kernel first seen with size 0 or 1 can be
reused for larger sizes" -- `specialize_zero_one=settings.static_shapes`,
which vLLM forces to False) -- so this is a genuine Helion bug where that
guarantee doesn't hold when a hl.tile()'s fixed block_size happens to
equal the first concrete size seen for that dimension. Confirmed this is
specific to that coincidence, not general "eager bucketing": running
num_tokens=7 (or any value != 1) FIRST for a given specialized combination
makes every subsequent num_tokens value (including 1) compute correctly.

A warm-up call with a safe (!=1) num_tokens before real use is a possible
workaround, but only protects the *specific* specialized combination it
warms up (hidden_size/group_size/quant_dtype/is_scale_transposed are all
hl.specialize()'d, so e.g. warming up hidden_size=256 does not protect
hidden_size=1024) -- a general fix would need to warm up every
specialized combination the deployment cares about, which is out of
scope here. Recommend filing this as an upstream Helion issue (the
to_triton_code() diff above is a minimal, concrete repro) as the real
fix. Per project decision, this kernel intentionally has NO XPU config
committed so it remains disabled (HelionKernelWrapper._disabled=True)
rather than shipping something that silently corrupts output for
num_tokens=1 users. Not caused by the device-portability fix or the
portable reference implementation below (both verified correct in
isolated single-shape calls -- see
_reference_silu_and_mul_per_block_quant).
"""

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.silu_and_mul_per_block_quant import (
    _pick_cache,
    baseline,
    pick_config,
    silu_and_mul_per_block_quant,
)
from vllm.kernels.helion.utils import get_int8_min_max, get_int8_min_scaling_factor
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _reference_silu_and_mul_per_block_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> None:
    """Device-agnostic reference for silu_and_mul_per_block_quant.

    baseline() calls torch.ops._C.silu_and_mul_per_block_quant, which is
    CUDA-only. On other platforms (e.g. XPU), use this instead.

    The activation half reuses SiluAndMul.forward_native (portable vLLM
    layer). The per-[1, group_size] quantization is done manually here
    rather than via QuantFP8: QuantFP8's dynamic group-quant path
    (_quantize_group_native) only ever produces FP8 output and does not
    accept scale_ub, while this kernel supports both int8 and FP8 output
    plus an optional scale_ub clamp. The math below mirrors the raw
    kernel body in silu_and_mul_per_block_quant() (see that file).

    is_scale_transposed is a dummy parameter on this kernel (kept only
    for torch.ops._C interface parity, per upstream PR review) -- it
    only affects the `scales` tensor's physical layout, not the values
    written, so no special handling is needed here.
    """
    num_tokens, two_intermediate_size = input.shape
    intermediate_size = two_intermediate_size // 2
    groups_per_row = intermediate_size // group_size

    silu_out = SiluAndMul.forward_native(input)  # [num_tokens, intermediate_size]
    x = silu_out.to(torch.float32).view(num_tokens, groups_per_row, group_size)

    quant_dtype = out.dtype
    if quant_dtype == torch.int8:
        qmin, qmax = get_int8_min_max()
        min_scaling_factor = get_int8_min_scaling_factor()
    else:
        qmin, qmax = get_fp8_min_max()
        min_scaling_factor = 1.0 / (qmax * 512.0)

    s_blk = x.abs().amax(dim=-1)
    if scale_ub is not None:
        s_blk = s_blk.clamp(max=scale_ub.to(torch.float32))
    s_blk = (s_blk * (1.0 / qmax)).clamp(min=min_scaling_factor)
    scales.copy_(s_blk)

    y = x / s_blk.unsqueeze(-1)
    if quant_dtype == torch.int8:
        y = y.round()
    y = y.clamp(qmin, qmax).to(quant_dtype)
    out.copy_(y.view(num_tokens, intermediate_size))


def _generate_fake_input(
    num_tokens: int, intermediate_size: int, group_size: int
) -> tuple[Any, ...]:
    with FakeTensorMode():
        in_dtype: torch.dtype = torch.bfloat16
        out_dtype: torch.dtype = current_platform.fp8_dtype()
        scale_dtype: torch.dtype = torch.float32
        device = current_platform.device_type
        input = torch.randn(
            num_tokens, 2 * intermediate_size, device=device, dtype=in_dtype
        )
        result = torch.empty(
            num_tokens, intermediate_size, device=input.device, dtype=out_dtype
        )
        scale = torch.empty(
            (num_tokens, intermediate_size // group_size),
            device=input.device,
            dtype=scale_dtype,
        )
        scale_ub = torch.mean(input).to(scale_dtype)
        args = (
            result,
            input,
            scale,
            group_size,
            scale_ub,
            False,
        )
        return args


class TestSiluAndMulPerBlockQuantConfigPicker:
    def setup_method(self):
        _pick_cache.clear()

    def test_config_picker_exact_match(self):
        config_keys = [
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 16}),
        ]

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"intermediate_size": 4096, "group_size": 128, "num_tokens": 16}
        )

    def test_config_picker_closest_match(self):
        config_keys = [
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 32}),
            CaseKey({"intermediate_size": 2048, "group_size": 128, "num_tokens": 16}),
            CaseKey({"intermediate_size": 2048, "group_size": 128, "num_tokens": 32}),
            CaseKey({"intermediate_size": 4096, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 64, "num_tokens": 32}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 32}),
        ]

        args = _generate_fake_input(20, 3000, 70)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"intermediate_size": 2048, "group_size": 64, "num_tokens": 32}
        )

    def test_config_picker_no_configs(self):
        config_keys: list[dict] = []

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 32}),
            CaseKey({"intermediate_size": 2048, "group_size": 128, "num_tokens": 16}),
            CaseKey({"intermediate_size": 2048, "group_size": 128, "num_tokens": 32}),
            CaseKey({"intermediate_size": 4096, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 64, "num_tokens": 32}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 32}),
        ]

        args = _generate_fake_input(64, 8192, 256)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"intermediate_size": 4096, "group_size": 128, "num_tokens": 32}
        )


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestSiluAndMulPerBlockQuantCorrectness:
    @pytest.mark.parametrize("num_tokens", [1, 7, 4096])
    @pytest.mark.parametrize("hidden_size", [1024, 2048, 5120])
    @pytest.mark.parametrize("group_size", [64, 128])
    @pytest.mark.parametrize("is_scale_transposed", [False, True])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("quant_dtype", [current_platform.fp8_dtype(), torch.int8])
    @pytest.mark.parametrize("has_scale_ub", [True, False])
    @pytest.mark.parametrize("seed", [0])
    def test_silu_and_mul_per_block_quant(
        self,
        num_tokens: int,
        hidden_size: int,
        group_size: int,
        is_scale_transposed: bool,
        dtype: torch.dtype,
        quant_dtype: torch.dtype,
        has_scale_ub: bool,
        seed: int,
    ) -> None:
        skip_if_platform_unsupported("silu_and_mul_per_block_quant")
        set_random_seed(seed)

        if hidden_size % group_size != 0:
            return

        if has_scale_ub and quant_dtype != FP8_DTYPE:
            # skip
            return

        device = current_platform.device_type
        scale = 1 / hidden_size
        x = torch.randn(num_tokens, 2 * hidden_size, dtype=dtype, device=device) * scale

        if has_scale_ub:
            act = torch.nn.functional.silu(x[:, :hidden_size]) * x[:, hidden_size:]
            act_abs = act.abs().float()
            scale_ub = 0.5 * (act_abs.mean() + act_abs.amax())
        else:
            scale_ub = None

        ref_out = torch.empty(num_tokens, hidden_size, device=device, dtype=quant_dtype)

        if is_scale_transposed:
            ref_scales = torch.empty(
                (hidden_size // group_size, x.shape[0]),
                device=device,
                dtype=torch.float32,
            ).t()
        else:
            ref_scales = torch.empty(
                (x.shape[0], hidden_size // group_size),
                device=device,
                dtype=torch.float32,
            )

        ops_out = ref_out.clone()
        ops_scales = ref_scales.clone()

        if current_platform.is_cuda():
            baseline(ref_out, x, ref_scales, group_size, scale_ub, is_scale_transposed)
        else:
            # baseline() calls torch.ops._C.silu_and_mul_per_block_quant, which
            # is CUDA-only. Use the portable reference on other platforms
            # (e.g. XPU).
            _reference_silu_and_mul_per_block_quant(
                ref_out, x, ref_scales, group_size, scale_ub, is_scale_transposed
            )
        silu_and_mul_per_block_quant(
            ops_out, x, ops_scales, group_size, scale_ub, is_scale_transposed
        )

        if current_platform.is_cuda():
            torch.testing.assert_close(ref_scales, ops_scales)
            # allow 1 ULP difference
            assert (
                ref_out.view(torch.uint8).to(torch.int16)
                - ops_out.view(torch.uint8).to(torch.int16)
            ).abs().max() <= 1
        else:
            # _reference_silu_and_mul_per_block_quant computes the activation
            # via SiluAndMul.forward_native in the input's native dtype
            # (bf16/fp16), while the Helion kernel upcasts both halves to
            # float32 before the sigmoid/multiply -- a small but real
            # numerical difference. The CUDA path's strict raw-code 1-ULP
            # check (calibrated for two near-bit-identical float32
            # computations) is too tight for this, so compare dequantized
            # values with a loose tolerance instead, analogous to
            # _reference_scaled_mm's use of a loose rtol/atol in
            # test_scaled_mm.py.
            torch.testing.assert_close(ref_scales, ops_scales, rtol=0.05, atol=1e-6)

            def _dequant(vals: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
                n = vals.shape[0]
                return (
                    vals.to(torch.float32).view(n, -1, group_size)
                    * scales.unsqueeze(-1)
                ).view(n, hidden_size)

            torch.testing.assert_close(
                _dequant(ref_out, ref_scales),
                _dequant(ops_out, ops_scales),
                rtol=0.05,
                atol=1e-3,
            )


class TestSiluAndMulPerBlockQuantIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "silu_and_mul_per_block_quant" in registered_kernels

        kernel_wrapper = registered_kernels["silu_and_mul_per_block_quant"]
        assert kernel_wrapper.op_name == "silu_and_mul_per_block_quant"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["out", "scales"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("silu_and_mul_per_block_quant")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["silu_and_mul_per_block_quant"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_fake_input(16, 4096, 128)
        assert fake_impl(*args) is None
