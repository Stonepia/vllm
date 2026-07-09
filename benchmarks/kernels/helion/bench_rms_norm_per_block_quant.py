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

Shapes: one per Qwen3 model hidden_size (1.7B/8B/32B), group_size=128,
num_tokens=128 by default; ``--full`` sweeps the blog's full 14-value
num_tokens grid (1..8192) per SHAPE_AUDIT.md.

Usage:
    python benchmarks/kernels/helion/bench_rms_norm_per_block_quant.py
    python benchmarks/kernels/helion/bench_rms_norm_per_block_quant.py \
        --autotune-effort full --autotune-budget-seconds 120

Crash-safe full-sweep mode (see ``run_full_sweep.sh``):
    python bench_rms_norm_per_block_quant.py --full --list-cases
    python bench_rms_norm_per_block_quant.py --full \\
        --only-case <name> --sweep-file benchmark_logs/sweep_....jsonl
    python bench_rms_norm_per_block_quant.py --full \\
        --report-from-sweep benchmark_logs/sweep_....jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from collections.abc import Iterator

import torch
from bench_utils import (
    add_crash_safe_args,
    append_sweep_result,
    bench_with_xpu_graph_fallback,
    load_sweep_results,
    print_case_name,
    print_detailed_report,
)
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

FP8 = current_platform.fp8_dtype()
BF16 = torch.bfloat16
DEVICE = current_platform.device_type

# One shape per Qwen3 model hidden_size.
HIDDEN_SIZES: dict[str, int] = {
    "Qwen3-1.7B": 2048,
    "Qwen3-8B": 4096,
    "Qwen3-32B": 5120,
}
NUM_TOKENS_FULL = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
NUM_TOKENS_QUICK = [128]
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


def bench(fn) -> tuple[float, bool]:
    return bench_with_xpu_graph_fallback(fn)


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


def iter_cases(num_tokens_list: list[int]) -> Iterator[tuple[str, str, int, int]]:
    """Yields (case_name, name, hidden_size, num_tokens) for every case."""
    for name, hidden_size in HIDDEN_SIZES.items():
        for num_tokens in num_tokens_list:
            case_name = (
                f"hidden_size_{hidden_size}_group_size_{GROUP_SIZE}_"
                f"num_tokens_{num_tokens}"
            )
            yield case_name, name, hidden_size, num_tokens


def measure_case(
    kernel, hidden_size: int, num_tokens: int, torch_ops_c_available: bool
) -> tuple[float, float, float | None, float, float | None, float, bool]:
    """Measures one (hidden_size, num_tokens) case. Returns (err,
    compiled_ms, ops_c_ms, kern_ms, speedup_vs_ops_c, speedup_vs_compiled,
    xpu_graph_enabled). Raises on failure -- deliberately left uncaught
    here; see run_full_sweep.sh for how a raised exception from
    --only-case is handled from outside the process."""
    base_args = make_inputs(num_tokens, hidden_size, GROUP_SIZE)

    kernel_args = clone_args(base_args)
    kernel(*kernel_args)
    torch.xpu.synchronize()

    compiled_args = clone_args(base_args)
    torch_compile_baseline(*compiled_args)
    torch.xpu.synchronize()

    err = rel_err(
        kernel_args[0], kernel_args[3], compiled_args[0], compiled_args[3], GROUP_SIZE
    )

    bench_kernel_args = clone_args(base_args)
    bench_compiled_args = clone_args(base_args)
    kern_ms, kern_graph = bench(lambda a=bench_kernel_args: kernel(*a))
    compiled_ms, compiled_graph = bench(
        lambda a=bench_compiled_args: torch_compile_baseline(*a)
    )
    xpu_graph_enabled = compiled_graph and kern_graph
    speedup_vs_compiled = compiled_ms / kern_ms if kern_ms > 0 else 0.0

    ops_c_ms = None
    speedup_vs_ops_c = None
    if torch_ops_c_available:
        bench_opsc_args = clone_args(base_args)
        ops_c_ms, _ = bench(lambda a=bench_opsc_args: torch_ops_c_baseline(*a))
        speedup_vs_ops_c = ops_c_ms / kern_ms if kern_ms > 0 else 0.0

    return (
        err,
        compiled_ms,
        ops_c_ms,
        kern_ms,
        speedup_vs_ops_c,
        speedup_vs_compiled,
        xpu_graph_enabled,
    )


def _print_case_result(
    name: str,
    hidden_size: int,
    num_tokens: int,
    err: float,
    compiled_ms: float,
    ops_c_ms: float | None,
    kern_ms: float,
    speedup_vs_ops_c: float | None,
    speedup_vs_compiled: float,
    xpu_graph_enabled: bool,
) -> None:
    ops_c_str = f"{ops_c_ms:9.5f}" if ops_c_ms is not None else "      N/A"
    speedup_c_str = (
        f"{speedup_vs_ops_c:6.3f}x" if speedup_vs_ops_c is not None else "    N/A"
    )
    xpu_graph_str = "enabled" if xpu_graph_enabled else "disabled (fallback)"
    print(
        f"{name:12s} num_tokens={num_tokens:5d} hidden_size={hidden_size:6d} "
        f"group_size={GROUP_SIZE:4d}  rel_err={err:.4f}  "
        f"compiled_ms={compiled_ms:9.5f}  ops_c_ms={ops_c_str}  "
        f"helion_ms={kern_ms:9.5f}  "
        f"speedup_vs_compiled={speedup_vs_compiled:6.3f}x  "
        f"speedup_vs_ops_c={speedup_c_str}  xpu_graph={xpu_graph_str}"
    )


def _print_summary_and_report(
    rows: list[
        tuple[str, float, float, float | None, float, float | None, float, bool]
    ],
    detailed_rows: list[tuple[str, float, float]],
    n_not_attempted: int = 0,
) -> None:
    speedups_compiled = [r[-2] for r in rows if r[-2] > 0]
    if speedups_compiled:
        geo = math.exp(sum(math.log(s) for s in speedups_compiled) / len(rows))
        print(
            f"\ngeomean speedup vs torch.compile(native) over "
            f"{len(rows)} shapes: {geo:.3f}x"
        )
    speedups_ops_c = [r[-3] for r in rows if r[-3] is not None and r[-3] > 0]
    if speedups_ops_c:
        geo_c = math.exp(sum(math.log(s) for s in speedups_ops_c) / len(rows))
        print(f"geomean speedup vs torch.ops._C over {len(rows)} shapes: {geo_c:.3f}x")
    else:
        print("torch.ops._C: N/A on this platform")
    max_err = max((r[1] for r in rows), default=0.0)
    print(f"max rel_err across all shapes: {max_err:.4f}")
    n_graph_enabled = sum(1 for r in rows if r[-1])
    print(f"xpu_graph enabled for {n_graph_enabled}/{len(rows)} shapes")
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
    for case_name, name, hidden_size, num_tokens in iter_cases(num_tokens_list):
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
            "torch.ops._C.rms_norm_per_block_quant is not available on "
            "this platform (CUDA-only); reporting torch.compile(native) "
            "only, no torch.ops._C column.\n"
        )

    kernel = build_kernel(args.autotune_effort, args.autotune_budget_seconds)

    if args.only_case:
        if not args.sweep_file:
            raise SystemExit("--only-case requires --sweep-file")
        for case_name, name, hidden_size, num_tokens in iter_cases(num_tokens_list):
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
            ) = measure_case(kernel, hidden_size, num_tokens, torch_ops_c_available)
            _print_case_result(
                name,
                hidden_size,
                num_tokens,
                err,
                compiled_ms,
                ops_c_ms,
                kern_ms,
                speedup_vs_ops_c,
                speedup_vs_compiled,
                xpu_graph_enabled,
            )
            append_sweep_result(
                args.sweep_file,
                case_name,
                name=name,
                hidden_size=hidden_size,
                num_tokens=num_tokens,
                rel_err=err,
                compiled_ms=compiled_ms,
                ops_c_ms=ops_c_ms,
                kernel_ms=kern_ms,
                speedup_vs_ops_c=speedup_vs_ops_c,
                speedup_vs_compiled=speedup_vs_compiled,
                xpu_graph_enabled=xpu_graph_enabled,
            )
            return
        raise SystemExit(f"Unknown --only-case {args.only_case!r}")

    rows = []
    detailed_rows: list[tuple[str, float, float]] = []
    for case_name, name, hidden_size, num_tokens in iter_cases(num_tokens_list):
        (
            err,
            compiled_ms,
            ops_c_ms,
            kern_ms,
            speedup_vs_ops_c,
            speedup_vs_compiled,
            xpu_graph_enabled,
        ) = measure_case(kernel, hidden_size, num_tokens, torch_ops_c_available)
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
            )
        )
        _print_case_result(
            name,
            hidden_size,
            num_tokens,
            err,
            compiled_ms,
            ops_c_ms,
            kern_ms,
            speedup_vs_ops_c,
            speedup_vs_compiled,
            xpu_graph_enabled,
        )

    _print_summary_and_report(rows, detailed_rows)


if __name__ == "__main__":
    main()
