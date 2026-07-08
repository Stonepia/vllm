# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level benchmark for the Helion ``silu_and_mul_dynamic_per_token_quant``
kernel on Intel XPU.

Three-way comparison, matching the blog's own methodology
(https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/):

  a) the Helion kernel
  b) ``torch.compile(native_impl)`` -- native_impl composes
     ``SiluAndMul.forward_native`` + ``QuantFP8(group_shape=
     GroupShape.PER_TOKEN)``, matching the ``Layer`` class in the reference
     end-to-end-benchmark dev branch exactly
     (xiaohongchen1991/vllm@d9b0566a54737c1b0a8e3714ef7a3ee40a4442c2,
     vllm/kernels/helion/ops/silu_and_mul_dynamic_per_token_quant.py).
     Wrapped with the same ``torch.compile(..., fullgraph=True,
     dynamic=False, backend="inductor", options={...})`` call.
  c) ``torch.ops._C.dynamic_per_token_scaled_fp8_quant`` (what this kernel's
     own ``baseline()`` calls, via ``ops.scaled_fp8_quant(...,
     use_per_token_if_dynamic=True)``) -- CUDA-only; probed once at import
     time, reports N/A here (confirmed unavailable: this environment's
     SiluAndMul CustomOp cannot even be instantiated on XPU since its
     underlying torch.ops._C.silu_and_mul isn't registered either, same
     vllm_xpu_kernels ABI gap noted elsewhere).

Shapes: one per Qwen3 model's gate_up projection, following
bench_scaled_mm.py's B_SHAPES convention (gate_up [K, N] dims halved, since
silu_and_mul halves the width):
  Qwen3-1.7B gate_up N=12288 -> intermediate_size=6144
  Qwen3-8B   gate_up N=24576 -> intermediate_size=12288
  Qwen3-32B  gate_up N=51200 -> intermediate_size=25600

Usage:
    python benchmarks/kernels/helion/bench_silu_and_mul_dynamic_per_token_quant.py
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from functools import partial

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.kernels.helion.ops.silu_and_mul_dynamic_per_token_quant import (
    silu_and_mul_dynamic_per_token_quant as _wrapper,
)
from vllm.kernels.helion.register import create_helion_decorated_kernel
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.triton_utils import triton

FP8 = current_platform.fp8_dtype()
BF16 = torch.bfloat16
DEVICE = current_platform.device_type

# intermediate_size per Qwen3 model's gate_up projection (halved N dim).
SHAPES: dict[str, int] = {
    "Qwen3-1.7B/gate_up": 6144,
    "Qwen3-8B/gate_up": 12288,
    "Qwen3-32B/gate_up": 25600,
}
NUM_TOKENS = 128

# torch.compile options copied verbatim from the reference commit.
_INDUCTOR_OPTIONS = {
    "enable_auto_functionalized_v2": False,
    "size_asserts": False,
    "alignment_asserts": False,
    "scalar_asserts": False,
    "combo_kernels": True,
    "benchmark_combo_kernel": True,
}

# QuantFP8 (a CustomOp/nn.Module) needs an active VllmConfig context to
# construct; forward itself does not need the context afterward.
with set_current_vllm_config(VllmConfig()):
    _QUANT_OP = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)


def _native_impl(input: torch.Tensor, scale_ub: torch.Tensor | None):
    """Matches the reference commit's Layer.forward exactly."""
    act_result = SiluAndMul.forward_native(input)
    result, scale = _QUANT_OP.forward(act_result, None, scale_ub)
    return result, scale


with set_current_vllm_config(VllmConfig()):
    _COMPILED_NATIVE = torch.compile(
        _native_impl,
        fullgraph=True,
        dynamic=False,
        backend="inductor",
        options=_INDUCTOR_OPTIONS,
    )


def silu_and_mul_dynamic_per_token_quant_eager(
    result: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    scale_ub: torch.Tensor | None = None,
) -> None:
    """Eager native_impl -- used only as the Helion autotuner's fast
    per-candidate-config accuracy check, not as a reported baseline.

    Also replicates the "overwrite first half of input in-place" step from
    this kernel's own baseline() (in
    vllm/kernels/helion/ops/silu_and_mul_dynamic_per_token_quant.py): the
    Helion kernel mutates `input` too (registered with mutates_args
    including "input"), and Helion's autotuner accuracy-check compares the
    full post-call args tuple, not just the return value -- this mutation
    isn't part of the real computation, it exists purely so that check
    passes when this function is used as autotune_baseline_fn (see
    build_kernel() below). torch_compile_baseline (the actual reported
    benchmark baseline) does NOT do this, matching the reference commit,
    since it isn't used for the autotuner's args-tuple comparison.
    """
    silu_and_mul_out = SiluAndMul.forward_native(input)
    intermediate_size = silu_and_mul_out.shape[1]
    input[:, :intermediate_size].copy_(silu_and_mul_out.to(input.dtype))
    out, scale_out = _QUANT_OP.forward_native(
        silu_and_mul_out, scale=None, scale_ub=scale_ub
    )
    result.copy_(out)
    scale.copy_(scale_out)


def torch_compile_baseline(
    result: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    scale_ub: torch.Tensor | None = None,
) -> None:
    """(b): torch.compile(native_impl), matching the reference commit."""
    out, scale_out = _COMPILED_NATIVE(input, scale_ub)
    result.copy_(out)
    scale.copy_(scale_out)


def torch_ops_c_baseline(
    result: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    scale_ub: torch.Tensor | None = None,
) -> None:
    """(c): the kernel's own baseline() -- torch.ops._C, CUDA-only."""
    import vllm._custom_ops as ops

    silu_and_mul_out = SiluAndMul.forward_native(input)
    out, scale_out = ops.scaled_fp8_quant(
        silu_and_mul_out, scale=None, scale_ub=scale_ub, use_per_token_if_dynamic=True
    )
    result.copy_(out)
    scale.copy_(scale_out)


def _probe_torch_ops_c() -> bool:
    try:
        x = torch.rand(2, 16, dtype=BF16, device=DEVICE) + 1e-6
        r = torch.empty(2, 8, dtype=FP8, device=DEVICE)
        s = torch.empty((2, 1), dtype=torch.float32, device=DEVICE)
        torch_ops_c_baseline(r, x.clone(), s, None)
        torch.xpu.synchronize()
        return True
    except Exception:
        return False


def make_inputs(
    num_tokens: int, intermediate_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (input_src, scale_ub); callers clone input_src per run since
    both the kernel and the eager reference mutate the first half in place.
    """
    input_src = (
        torch.rand(num_tokens, 2 * intermediate_size, dtype=BF16, device=DEVICE) + 1e-6
    )
    scale_ub = torch.mean(SiluAndMul.forward_native(input_src)).to(torch.float32)
    return input_src, scale_ub


def make_output_buffers(
    num_tokens: int, intermediate_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    result = torch.empty(num_tokens, intermediate_size, dtype=FP8, device=DEVICE)
    scale = torch.empty((num_tokens, 1), dtype=torch.float32, device=DEVICE)
    return result, scale


def bench(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


def rel_err(
    out: torch.Tensor,
    out_scale: torch.Tensor,
    ref: torch.Tensor,
    ref_scale: torch.Tensor,
) -> float:
    o = out.float() * out_scale.float()
    r = ref.float() * ref_scale.float()
    return ((o - r).abs().max() / (r.abs().max() + 1e-6)).item()


def build_kernel(autotune_effort: str, autotune_budget_seconds: int):
    """Rebuild silu_and_mul_dynamic_per_token_quant with a live per-shape
    autotuner (bypassing the single dummy preset config) and an
    XPU-compatible autotune baseline.
    """
    settings = replace(
        _wrapper.helion_settings,
        autotune_baseline_fn=silu_and_mul_dynamic_per_token_quant_eager,
        autotune_baseline_atol=1.0,
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
            "torch.ops._C.dynamic_per_token_scaled_fp8_quant (via "
            "ops.scaled_fp8_quant) is not available on this platform "
            "(CUDA-only); reporting torch.compile(native) only, no "
            "torch.ops._C column.\n"
        )

    kernel = build_kernel(args.autotune_effort, args.autotune_budget_seconds)

    rows = []
    autotune_wall_seconds = 0.0
    for name, intermediate_size in SHAPES.items():
        input_src, scale_ub = make_inputs(NUM_TOKENS, intermediate_size)

        kernel_input = input_src.clone()
        kernel_result, kernel_scale = make_output_buffers(NUM_TOKENS, intermediate_size)

        t0 = time.perf_counter()
        kernel(kernel_result, kernel_input, kernel_scale, scale_ub)
        torch.xpu.synchronize()
        shape_autotune_s = time.perf_counter() - t0
        autotune_wall_seconds += shape_autotune_s

        compiled_input = input_src.clone()
        compiled_result, compiled_scale = make_output_buffers(
            NUM_TOKENS, intermediate_size
        )
        torch_compile_baseline(
            compiled_result, compiled_input, compiled_scale, scale_ub
        )
        torch.xpu.synchronize()

        err = rel_err(kernel_result, kernel_scale, compiled_result, compiled_scale)

        compiled_ms = bench(
            partial(
                torch_compile_baseline,
                compiled_result,
                compiled_input,
                compiled_scale,
                scale_ub,
            )
        )
        kern_ms = bench(
            partial(kernel, kernel_result, kernel_input, kernel_scale, scale_ub)
        )
        speedup_vs_compiled = compiled_ms / kern_ms if kern_ms > 0 else 0.0

        ops_c_ms = None
        speedup_vs_ops_c = None
        if torch_ops_c_available:
            opsc_input = input_src.clone()
            opsc_result, opsc_scale = make_output_buffers(NUM_TOKENS, intermediate_size)
            ops_c_ms = bench(
                partial(
                    torch_ops_c_baseline, opsc_result, opsc_input, opsc_scale, scale_ub
                )
            )
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
            f"{name:20s} num_tokens={NUM_TOKENS:5d} "
            f"intermediate_size={intermediate_size:6d}  rel_err={err:.4f}  "
            f"compiled_ms={compiled_ms:9.5f}  ops_c_ms={ops_c_str}  "
            f"helion_ms={kern_ms:9.5f}  "
            f"speedup_vs_compiled={speedup_vs_compiled:6.3f}x  "
            f"speedup_vs_ops_c={speedup_c_str}  (autotune {shape_autotune_s:7.2f}s)"
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
    print(
        f"total autotuning wall time for {len(rows)} shapes: "
        f"{autotune_wall_seconds:.2f}s"
    )


if __name__ == "__main__":
    main()
