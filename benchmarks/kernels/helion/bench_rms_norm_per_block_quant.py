# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level benchmark for the Helion ``rms_norm_per_block_quant`` kernel
on Intel XPU.

Three-way comparison, matching the blog's own methodology
(https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/):

  a) the Helion kernel
  b) ``torch.compile(native_impl)`` -- native_impl composes
     ``RMSNorm.forward_native`` + ``QuantFP8(group_shape=GroupShape(1,
     128)).forward_native``, matching the ``Layer`` class in the reference
     end-to-end-benchmark dev branch
     (xiaohongchen1991/vllm@d9b0566a54737c1b0a8e3714ef7a3ee40a4442c2,
     vllm/kernels/helion/ops/rms_norm_per_block_quant.py), adapted to call
     ``RMSNorm(...).forward_native(...)`` since that commit's
     ``RMSNorm.forward_static(x, epsilon, hidden_size, dtype, weight,
     residual)`` (a bare staticmethod) no longer exists on current main.
     Wrapped with the same ``torch.compile(..., fullgraph=True,
     dynamic=False, backend="inductor", options={...})`` call as the
     reference commit.
  c) ``torch.ops._C.rms_norm_per_block_quant`` -- this kernel's own
     ``baseline()`` calls this directly. CUDA-only; probed once at import
     time, reports N/A here (confirmed unavailable in this environment).

Shapes: one per Qwen3 model hidden_size (1.7B/8B/32B), num_tokens=128,
group_size=128 -- matching the task's requested representative set.

Usage:
    python benchmarks/kernels/helion/bench_rms_norm_per_block_quant.py
    python benchmarks/kernels/helion/bench_rms_norm_per_block_quant.py \
        --autotune-effort full --autotune-budget-seconds 120
"""

from __future__ import annotations

import argparse
import dataclasses
import math

import torch
from helion._testing import assert_close_with_mismatch_tolerance

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.kernels.helion.ops.rms_norm_per_block_quant import (
    rms_norm_per_block_quant as _wrapper,
)
from vllm.kernels.helion.register import create_helion_decorated_kernel
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.triton_utils import triton

FP8 = current_platform.fp8_dtype()
BF16 = torch.bfloat16
DEVICE = current_platform.device_type

# One shape per Qwen3 model hidden_size.
HIDDEN_SIZES: dict[str, int] = {
    "Qwen3-1.7B": 2048,
    "Qwen3-8B": 4096,
    "Qwen3-32B": 5120,
}
NUM_TOKENS = 128
GROUP_SIZE = 128
EPS = 1e-6

# torch.compile options copied verbatim from the reference commit.
_INDUCTOR_OPTIONS = {
    "enable_auto_functionalized_v2": False,
    "size_asserts": False,
    "alignment_asserts": False,
    "scalar_asserts": False,
    "combo_kernels": True,
    "benchmark_combo_kernel": True,
}

_layer_cache: dict[tuple[int, torch.dtype, float], tuple] = {}


def _get_layers(hidden_size: int, dtype: torch.dtype, epsilon: float):
    key = (hidden_size, dtype, epsilon)
    if key not in _layer_cache:
        with set_current_vllm_config(VllmConfig()):
            rms_norm_layer = RMSNorm(hidden_size, eps=epsilon, dtype=dtype).to(DEVICE)
            quant_layer = QuantFP8(static=False, group_shape=GroupShape(1, GROUP_SIZE))

            def native_impl(input, weight, scale_ub, residual):
                rms_norm_layer.weight.data.copy_(weight)
                normed, new_residual = rms_norm_layer.forward(input, residual)
                quantized, scale = quant_layer.forward(normed, None)
                return quantized, new_residual, scale

            compiled = torch.compile(
                native_impl,
                fullgraph=True,
                dynamic=False,
                backend="inductor",
                options=_INDUCTOR_OPTIONS,
            )
        _layer_cache[key] = (rms_norm_layer, quant_layer, native_impl, compiled)
    return _layer_cache[key]


def torch_compile_baseline(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    """(b): torch.compile(native_impl), adapted from the reference commit."""
    _, _, _, compiled = _get_layers(input.shape[-1], input.dtype, epsilon)
    quantized, new_residual, s = compiled(input, weight, scale_ub, residual)
    result.copy_(quantized)
    scale.copy_(s)
    if residual is not None:
        residual.copy_(new_residual)


def rms_norm_per_block_quant_eager(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    """Eager (uncompiled) native_impl -- used only as the Helion autotuner's
    fast per-candidate-config accuracy check, not as a reported baseline."""
    _, _, native_impl, _ = _get_layers(input.shape[-1], input.dtype, epsilon)
    quantized, new_residual, s = native_impl(input, weight, scale_ub, residual)
    result.copy_(quantized)
    scale.copy_(s)
    if residual is not None:
        residual.copy_(new_residual)


def torch_ops_c_baseline(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    """(c): the kernel's own baseline() -- torch.ops._C, CUDA-only."""
    torch.ops._C.rms_norm_per_block_quant(
        result,
        input,
        weight,
        scale,
        epsilon,
        scale_ub,
        residual,
        group_size,
        is_scale_transposed,
    )


def _probe_torch_ops_c() -> bool:
    try:
        x = torch.randn(2, GROUP_SIZE, dtype=BF16, device=DEVICE)
        w = torch.ones(GROUP_SIZE, dtype=BF16, device=DEVICE)
        r = torch.empty_like(x, dtype=FP8)
        s = torch.empty((2, 1), dtype=torch.float32, device=DEVICE)
        res = torch.randn_like(x)
        torch_ops_c_baseline(r, x, w, s, EPS, None, res, GROUP_SIZE, False)
        torch.xpu.synchronize()
        return True
    except Exception:
        return False


def make_inputs(num_tokens: int, hidden_size: int, group_size: int):
    scale = 1.0 / hidden_size
    input = (torch.randn(num_tokens, hidden_size, dtype=BF16, device=DEVICE)) * scale
    result = torch.empty(input.shape, dtype=FP8, device=DEVICE)
    quant_scale = torch.empty(
        (num_tokens, hidden_size // group_size), dtype=torch.float32, device=DEVICE
    )
    residual = torch.randn_like(input) * scale
    weight = torch.normal(
        mean=1.0, std=1.0, size=(hidden_size,), dtype=BF16, device=DEVICE
    )
    return (result, input, weight, quant_scale, EPS, None, residual, group_size, False)


def clone_args(args: tuple) -> tuple:
    result, input, weight, scale, epsilon, scale_ub, residual, group_size, is_st = args
    return (
        result.clone(),
        input,
        weight,
        scale.clone(),
        epsilon,
        scale_ub,
        residual.clone() if residual is not None else None,
        group_size,
        is_st,
    )


def dequantize(x: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    num_tokens, hidden_size = x.shape
    groups_per_row = hidden_size // group_size
    x_grouped = x.float().view(num_tokens, groups_per_row, group_size)
    return (x_grouped * scale.float().unsqueeze(-1)).view(num_tokens, hidden_size)


def rel_err(
    out: torch.Tensor,
    out_scale: torch.Tensor,
    ref: torch.Tensor,
    ref_scale: torch.Tensor,
    group_size: int,
) -> float:
    o = dequantize(out, out_scale, group_size)
    r = dequantize(ref, ref_scale, group_size)
    return ((o - r).abs().max() / (r.abs().max() + 1e-6)).item()


def bench(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


def _autotune_accuracy_check(actual: object, expected: object) -> None:
    """Custom autotune-candidate accuracy check -- unchanged from before,
    see git history for the full explanation (Helion's default bitwise-exact
    fp8 check is too strict for legitimate 1-ULP rounding-order differences).
    """
    if actual is None or expected is None:
        return
    result_a, _, _, scale_a, _, _, residual_a, _, _ = actual
    result_e, _, _, scale_e, _, _, residual_e, _, _ = expected

    torch.testing.assert_close(scale_a, scale_e)
    if residual_a is not None:
        torch.testing.assert_close(residual_a, residual_e)

    assert_close_with_mismatch_tolerance(
        result_a,
        result_e,
        atol=0.0,
        rtol=0.0,
        max_mismatch_pct=0.01,
        max_abs_diff=5.0,
        max_rel_diff=0.5,
    )


def build_kernel(autotune_effort: str, autotune_budget_seconds: int):
    """Rebuild rms_norm_per_block_quant with a live per-shape autotuner
    (bypassing the single dummy preset config) and an XPU-compatible
    autotune baseline + relaxed accuracy-check tolerance.
    """
    settings = dataclasses.replace(
        _wrapper.helion_settings,
        autotune_baseline_fn=rms_norm_per_block_quant_eager,
        autotune_baseline_accuracy_check_fn=_autotune_accuracy_check,
    )
    return create_helion_decorated_kernel(
        _wrapper.raw_kernel_func,
        settings,
        extra_kwargs={
            "autotune_effort": autotune_effort,
            "autotune_budget_seconds": autotune_budget_seconds,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--autotune-effort", default="quick", choices=["none", "quick", "full"]
    )
    parser.add_argument("--autotune-budget-seconds", type=int, default=45)
    args = parser.parse_args()

    assert current_platform.is_xpu(), "This benchmark targets Intel XPU"

    torch_ops_c_available = _probe_torch_ops_c()
    if not torch_ops_c_available:
        print(
            "torch.ops._C.rms_norm_per_block_quant is not available on "
            "this platform (CUDA-only); reporting torch.compile(native) "
            "only, no torch.ops._C column.\n"
        )

    kernel = build_kernel(args.autotune_effort, args.autotune_budget_seconds)

    rows = []
    for name, hidden_size in HIDDEN_SIZES.items():
        base_args = make_inputs(NUM_TOKENS, hidden_size, GROUP_SIZE)
        group_size = GROUP_SIZE

        kernel_args = clone_args(base_args)
        kernel(*kernel_args)
        torch.xpu.synchronize()

        compiled_args = clone_args(base_args)
        torch_compile_baseline(*compiled_args)
        torch.xpu.synchronize()

        err = rel_err(
            kernel_args[0],
            kernel_args[3],
            compiled_args[0],
            compiled_args[3],
            group_size,
        )

        bench_kernel_args = clone_args(base_args)
        bench_compiled_args = clone_args(base_args)
        kern_ms = bench(lambda a=bench_kernel_args: kernel(*a))
        compiled_ms = bench(lambda a=bench_compiled_args: torch_compile_baseline(*a))
        speedup_vs_compiled = compiled_ms / kern_ms if kern_ms > 0 else 0.0

        ops_c_ms = None
        speedup_vs_ops_c = None
        if torch_ops_c_available:
            bench_opsc_args = clone_args(base_args)
            ops_c_ms = bench(lambda a=bench_opsc_args: torch_ops_c_baseline(*a))
            speedup_vs_ops_c = ops_c_ms / kern_ms if kern_ms > 0 else 0.0

        rows.append(
            (
                name,
                err,
                compiled_ms,
                ops_c_ms,
                kern_ms,
                speedup_vs_ops_c,
                speedup_vs_compiled,
            )
        )
        ops_c_str = f"{ops_c_ms:9.5f}" if ops_c_ms is not None else "      N/A"
        speedup_c_str = (
            f"{speedup_vs_ops_c:6.3f}x" if speedup_vs_ops_c is not None else "    N/A"
        )
        print(
            f"{name:12s} num_tokens={NUM_TOKENS:5d} hidden_size={hidden_size:6d} "
            f"group_size={group_size:4d}  rel_err={err:.4f}  "
            f"compiled_ms={compiled_ms:9.5f}  ops_c_ms={ops_c_str}  "
            f"helion_ms={kern_ms:9.5f}  "
            f"speedup_vs_compiled={speedup_vs_compiled:6.3f}x  "
            f"speedup_vs_ops_c={speedup_c_str}"
        )

    speedups_compiled = [r[-1] for r in rows if r[-1] > 0]
    if speedups_compiled:
        geo = math.exp(sum(math.log(s) for s in speedups_compiled) / len(rows))
        print(
            f"\ngeomean speedup vs torch.compile(native) over "
            f"{len(rows)} shapes: {geo:.3f}x"
        )
    speedups_ops_c = [r[-2] for r in rows if r[-2] is not None and r[-2] > 0]
    if speedups_ops_c:
        geo_c = math.exp(sum(math.log(s) for s in speedups_ops_c) / len(rows))
        print(f"geomean speedup vs torch.ops._C over {len(rows)} shapes: {geo_c:.3f}x")
    else:
        print("torch.ops._C: N/A on this platform")
    max_err = max((r[1] for r in rows), default=0.0)
    print(f"max rel_err across all shapes: {max_err:.4f}")


if __name__ == "__main__":
    main()
