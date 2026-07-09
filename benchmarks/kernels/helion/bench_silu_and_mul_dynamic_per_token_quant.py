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

num_tokens=128 by default; ``--full`` sweeps the blog's full 14-value
num_tokens grid (1..8192) per SHAPE_AUDIT.md.

Usage:
    python benchmarks/kernels/helion/bench_silu_and_mul_dynamic_per_token_quant.py
    python benchmarks/kernels/helion/bench_silu_and_mul_dynamic_per_token_quant.py \
        --full

Crash-safe full-sweep mode (see ``run_full_sweep.sh``):
    python bench_silu_and_mul_dynamic_per_token_quant.py --full --list-cases
    python bench_silu_and_mul_dynamic_per_token_quant.py --full \\
        --only-case <name> --sweep-file benchmark_logs/sweep_....jsonl
    python bench_silu_and_mul_dynamic_per_token_quant.py --full \\
        --report-from-sweep benchmark_logs/sweep_....jsonl
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterator
from dataclasses import replace
from functools import partial

import torch
from bench_utils import (
    add_crash_safe_args,
    append_sweep_result,
    bench_with_xpu_graph_fallback,
    load_sweep_results,
    print_case_name,
    print_detailed_report,
)

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.kernels.helion.ops.silu_and_mul_dynamic_per_token_quant import (
    silu_and_mul_dynamic_per_token_quant as _wrapper,
)
from vllm.kernels.helion.register import create_helion_decorated_kernel
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

FP8 = current_platform.fp8_dtype()
BF16 = torch.bfloat16
DEVICE = current_platform.device_type

# intermediate_size per Qwen3 model's gate_up projection (halved N dim).
SHAPES: dict[str, int] = {
    "Qwen3-1.7B/gate_up": 6144,
    "Qwen3-8B/gate_up": 12288,
    "Qwen3-32B/gate_up": 25600,
}
NUM_TOKENS_FULL = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
NUM_TOKENS_QUICK = [128]

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


def bench(fn) -> tuple[float, bool]:
    return bench_with_xpu_graph_fallback(fn)


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


def iter_cases(num_tokens_list: list[int]) -> Iterator[tuple[str, str, int, int]]:
    """Yields (case_name, name, intermediate_size, num_tokens) for every
    case."""
    for name, intermediate_size in SHAPES.items():
        for num_tokens in num_tokens_list:
            case_name = f"intermediate_size_{intermediate_size}_num_tokens_{num_tokens}"
            yield case_name, name, intermediate_size, num_tokens


def measure_case(
    kernel, intermediate_size: int, num_tokens: int, torch_ops_c_available: bool
) -> tuple[float, float, float | None, float, float | None, float, bool, float]:
    """Measures one (intermediate_size, num_tokens) case. Returns (err,
    compiled_ms, ops_c_ms, kern_ms, speedup_vs_ops_c, speedup_vs_compiled,
    xpu_graph_enabled, shape_autotune_s). Raises on failure -- deliberately
    left uncaught here; see run_full_sweep.sh for how a raised exception
    from --only-case is handled from outside the process."""
    input_src, scale_ub = make_inputs(num_tokens, intermediate_size)

    kernel_input = input_src.clone()
    kernel_result, kernel_scale = make_output_buffers(num_tokens, intermediate_size)

    t0 = time.perf_counter()
    kernel(kernel_result, kernel_input, kernel_scale, scale_ub)
    torch.xpu.synchronize()
    shape_autotune_s = time.perf_counter() - t0

    compiled_input = input_src.clone()
    compiled_result, compiled_scale = make_output_buffers(num_tokens, intermediate_size)
    torch_compile_baseline(compiled_result, compiled_input, compiled_scale, scale_ub)
    torch.xpu.synchronize()

    err = rel_err(kernel_result, kernel_scale, compiled_result, compiled_scale)

    compiled_ms, compiled_graph = bench(
        partial(
            torch_compile_baseline,
            compiled_result,
            compiled_input,
            compiled_scale,
            scale_ub,
        )
    )
    kern_ms, kern_graph = bench(
        partial(kernel, kernel_result, kernel_input, kernel_scale, scale_ub)
    )
    xpu_graph_enabled = compiled_graph and kern_graph
    speedup_vs_compiled = compiled_ms / kern_ms if kern_ms > 0 else 0.0

    ops_c_ms = None
    speedup_vs_ops_c = None
    if torch_ops_c_available:
        opsc_input = input_src.clone()
        opsc_result, opsc_scale = make_output_buffers(num_tokens, intermediate_size)
        ops_c_ms, _ = bench(
            partial(torch_ops_c_baseline, opsc_result, opsc_input, opsc_scale, scale_ub)
        )
        speedup_vs_ops_c = ops_c_ms / kern_ms if kern_ms > 0 else 0.0

    return (
        err,
        compiled_ms,
        ops_c_ms,
        kern_ms,
        speedup_vs_ops_c,
        speedup_vs_compiled,
        xpu_graph_enabled,
        shape_autotune_s,
    )


def _print_case_result(
    name: str,
    intermediate_size: int,
    num_tokens: int,
    err: float,
    compiled_ms: float,
    ops_c_ms: float | None,
    kern_ms: float,
    speedup_vs_ops_c: float | None,
    speedup_vs_compiled: float,
    xpu_graph_enabled: bool,
    shape_autotune_s: float,
) -> None:
    ops_c_str = f"{ops_c_ms:9.5f}" if ops_c_ms is not None else "      N/A"
    speedup_c_str = (
        f"{speedup_vs_ops_c:6.3f}x" if speedup_vs_ops_c is not None else "    N/A"
    )
    xpu_graph_str = "enabled" if xpu_graph_enabled else "disabled (fallback)"
    print(
        f"{name:20s} num_tokens={num_tokens:5d} "
        f"intermediate_size={intermediate_size:6d}  rel_err={err:.4f}  "
        f"compiled_ms={compiled_ms:9.5f}  ops_c_ms={ops_c_str}  "
        f"helion_ms={kern_ms:9.5f}  "
        f"speedup_vs_compiled={speedup_vs_compiled:6.3f}x  "
        f"speedup_vs_ops_c={speedup_c_str}  xpu_graph={xpu_graph_str}  "
        f"(autotune {shape_autotune_s:7.2f}s)"
    )


def _print_summary_and_report(
    rows: list[
        tuple[str, float, float, float | None, float, float | None, float, bool, float]
    ],
    detailed_rows: list[tuple[str, float, float]],
    n_not_attempted: int = 0,
) -> None:
    speedups_compiled = [r[-3] for r in rows if r[-3] > 0]
    if speedups_compiled:
        geo = math.exp(sum(math.log(s) for s in speedups_compiled) / len(rows))
        print(
            f"\ngeomean speedup vs torch.compile(native) over "
            f"{len(rows)} shapes: {geo:.3f}x"
        )
    speedups_ops_c = [r[-4] for r in rows if r[-4] is not None and r[-4] > 0]
    if speedups_ops_c:
        geo_c = math.exp(sum(math.log(s) for s in speedups_ops_c) / len(rows))
        print(f"geomean speedup vs torch.ops._C over {len(rows)} shapes: {geo_c:.3f}x")
    else:
        print("torch.ops._C: N/A on this platform")
    max_err = max((r[1] for r in rows), default=0.0)
    print(f"max rel_err across all shapes: {max_err:.4f}")
    n_graph_enabled = sum(1 for r in rows if r[-2])
    print(f"xpu_graph enabled for {n_graph_enabled}/{len(rows)} shapes")
    autotune_wall_seconds = sum(r[-1] for r in rows)
    print(
        f"total autotuning wall time for {len(rows)} shapes: "
        f"{autotune_wall_seconds:.2f}s"
    )
    if n_not_attempted:
        print(f"{n_not_attempted} case(s) FAILED or not yet attempted (see above)")

    print_detailed_report(
        hardware=current_platform.get_device_name(),
        baseline_name="torch.compile(native_impl)",
        rows=detailed_rows,
    )


def _report_from_sweep(sweep_file: str, num_tokens_list: list[int]) -> None:
    results = load_sweep_results(sweep_file)
    rows: list[tuple] = []
    detailed_rows: list[tuple[str, float, float]] = []
    n_not_attempted = 0
    for case_name, name, intermediate_size, num_tokens in iter_cases(num_tokens_list):
        rec = results.get(case_name)
        if rec is None or rec.get("status") != "ok":
            n_not_attempted += 1
            reason = "not attempted" if rec is None else "failed"
            print(f"{case_name}: {reason}")
            continue
        rows.append(
            (
                name,
                rec["rel_err"],
                rec["compiled_ms"],
                rec["ops_c_ms"],
                rec["kernel_ms"],
                rec["speedup_vs_ops_c"],
                rec["speedup_vs_compiled"],
                rec["xpu_graph_enabled"],
                rec["shape_autotune_s"],
            )
        )
        detailed_rows.append((case_name, rec["compiled_ms"], rec["kernel_ms"]))
    _print_summary_and_report(rows, detailed_rows, n_not_attempted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--autotune-effort", default="quick", choices=["none", "quick", "full"]
    )
    parser.add_argument("--autotune-budget-seconds", type=int, default=45)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Sweep the full blog num_tokens grid (14 values, 1..8192) "
        "instead of the default representative num_tokens=128.",
    )
    add_crash_safe_args(parser)
    args = parser.parse_args()

    num_tokens_list = NUM_TOKENS_FULL if args.full else NUM_TOKENS_QUICK

    if args.list_cases:
        for case_name, *_ in iter_cases(num_tokens_list):
            print_case_name(case_name)
        return

    if args.report_from_sweep:
        _report_from_sweep(args.report_from_sweep, num_tokens_list)
        return

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

    if args.only_case:
        if not args.sweep_file:
            raise SystemExit("--only-case requires --sweep-file")
        for case_name, name, intermediate_size, num_tokens in iter_cases(
            num_tokens_list
        ):
            if case_name != args.only_case:
                continue
            (
                err,
                compiled_ms,
                ops_c_ms,
                kern_ms,
                speedup_vs_ops_c,
                speedup_vs_compiled,
                xpu_graph_enabled,
                shape_autotune_s,
            ) = measure_case(
                kernel, intermediate_size, num_tokens, torch_ops_c_available
            )
            _print_case_result(
                name,
                intermediate_size,
                num_tokens,
                err,
                compiled_ms,
                ops_c_ms,
                kern_ms,
                speedup_vs_ops_c,
                speedup_vs_compiled,
                xpu_graph_enabled,
                shape_autotune_s,
            )
            append_sweep_result(
                args.sweep_file,
                case_name,
                name=name,
                intermediate_size=intermediate_size,
                num_tokens=num_tokens,
                rel_err=err,
                compiled_ms=compiled_ms,
                ops_c_ms=ops_c_ms,
                kernel_ms=kern_ms,
                speedup_vs_ops_c=speedup_vs_ops_c,
                speedup_vs_compiled=speedup_vs_compiled,
                xpu_graph_enabled=xpu_graph_enabled,
                shape_autotune_s=shape_autotune_s,
            )
            return
        raise SystemExit(f"Unknown --only-case {args.only_case!r}")

    rows = []
    detailed_rows: list[tuple[str, float, float]] = []
    for case_name, name, intermediate_size, num_tokens in iter_cases(num_tokens_list):
        (
            err,
            compiled_ms,
            ops_c_ms,
            kern_ms,
            speedup_vs_ops_c,
            speedup_vs_compiled,
            xpu_graph_enabled,
            shape_autotune_s,
        ) = measure_case(kernel, intermediate_size, num_tokens, torch_ops_c_available)
        detailed_rows.append((case_name, compiled_ms, kern_ms))
        rows.append(
            (
                name,
                err,
                compiled_ms,
                ops_c_ms,
                kern_ms,
                speedup_vs_ops_c,
                speedup_vs_compiled,
                xpu_graph_enabled,
                shape_autotune_s,
            )
        )
        _print_case_result(
            name,
            intermediate_size,
            num_tokens,
            err,
            compiled_ms,
            ops_c_ms,
            kern_ms,
            speedup_vs_ops_c,
            speedup_vs_compiled,
            xpu_graph_enabled,
            shape_autotune_s,
        )

    _print_summary_and_report(rows, detailed_rows)


if __name__ == "__main__":
    main()
