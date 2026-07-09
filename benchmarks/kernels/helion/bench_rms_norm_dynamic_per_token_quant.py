# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level benchmark for the Helion ``rms_norm_dynamic_per_token_quant``
kernel on Intel XPU.

Three-way comparison, matching the blog's own methodology
(https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/):

  a) the Helion kernel
  b) ``torch.compile(native_impl)`` -- native_impl composes
     ``RMSNorm.forward_native`` + ``QuantFP8.forward_native``, matching the
     ``Layer`` class in the reference end-to-end-benchmark dev branch
     (xiaohongchen1991/vllm@d9b0566a54737c1b0a8e3714ef7a3ee40a4442c2,
     vllm/kernels/helion/ops/rms_norm_dynamic_per_token_quant.py), adapted to
     call ``RMSNorm(...).forward_native(...)`` since that commit's
     ``RMSNorm.forward_static(x, epsilon, hidden_size, dtype, weight,
     residual)`` (a bare staticmethod taking weight as a plain argument) no
     longer exists on current main -- ``RMSNorm`` is now a stateful
     ``CustomOp``/``nn.Module``. Wrapped with the same
     ``torch.compile(..., fullgraph=True, dynamic=False, backend="inductor",
     options={...})`` call as the reference commit.
  c) ``torch.ops._C.rms_norm_dynamic_per_token_quant`` -- this kernel's own
     ``baseline()`` (in
     vllm/kernels/helion/ops/rms_norm_dynamic_per_token_quant.py) calls this
     directly. CUDA-only; probed once at import time, reported as N/A when
     unavailable (confirmed unavailable in this environment).

Shapes: one per Qwen3 model hidden_size (1.7B/8B/32B), at num_tokens=128
by default; ``--full`` sweeps the blog's full 14-value num_tokens grid
(1..8192) per SHAPE_AUDIT.md.

Timing: uses ``torch.xpu.XPUGraph`` capture/replay (an XPU port of
``triton.testing.do_bench_cudagraph``, see ``bench_utils.py``) to match the
blog's own dispatch-overhead-free methodology; falls back to plain
``triton.testing.do_bench`` (flagged in the output) if graph capture fails
for a given case.

Usage:
    python benchmarks/kernels/helion/bench_rms_norm_dynamic_per_token_quant.py
    python benchmarks/kernels/helion/bench_rms_norm_dynamic_per_token_quant.py --full

Crash-safe full-sweep mode (see ``run_full_sweep.sh``):
    python bench_rms_norm_dynamic_per_token_quant.py --full --list-cases
    python bench_rms_norm_dynamic_per_token_quant.py --full \\
        --only-case <name> --sweep-file benchmark_logs/sweep_....jsonl
    python bench_rms_norm_dynamic_per_token_quant.py --full \\
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

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.kernels.helion.ops.rms_norm_dynamic_per_token_quant import (
    rms_norm_dynamic_per_token_quant as _wrapper,
)
from vllm.kernels.helion.register import create_helion_decorated_kernel
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

FP8 = current_platform.fp8_dtype()
BF16 = torch.bfloat16
DEVICE = current_platform.device_type
EPSILON = 1e-6

AUTOTUNE_EFFORT = "quick"
AUTOTUNE_BUDGET_SECONDS = 45

# One [num_tokens, hidden_size] shape per Qwen3 model's hidden_size.
# NUM_TOKENS_QUICK (default) keeps the original single representative
# num_tokens; --full sweeps NUM_TOKENS_FULL, the blog's full grid (verified
# against generate_inputs()/a reference table, see SHAPE_AUDIT.md).
NUM_TOKENS_FULL = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
NUM_TOKENS_QUICK = [128]
HIDDEN_SIZES: dict[str, int] = {
    "Qwen3-1.7B": 2048,
    "Qwen3-8B": 4096,
    "Qwen3-32B": 5120,
}

# torch.compile options copied verbatim from the reference commit.
_INDUCTOR_OPTIONS = {
    "enable_auto_functionalized_v2": False,
    "size_asserts": False,
    "alignment_asserts": False,
    "scalar_asserts": False,
    "combo_kernels": True,
    "benchmark_combo_kernel": True,
}

# Cache of (RMSNorm, QuantFP8, compiled_fn) per (hidden_size, dtype, epsilon)
# so the timed loop doesn't pay module-construction/compile overhead per call.
_layer_cache: dict[tuple[int, torch.dtype, float], tuple] = {}


def _get_layers(hidden_size: int, dtype: torch.dtype, epsilon: float):
    key = (hidden_size, dtype, epsilon)
    if key not in _layer_cache:
        with set_current_vllm_config(VllmConfig()):
            rms_norm_layer = RMSNorm(hidden_size, eps=epsilon, dtype=dtype).to(DEVICE)
            quant_layer = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

            def native_impl(input, weight, scale_ub, residual):
                rms_norm_layer.weight.data.copy_(weight)
                normed, new_residual = rms_norm_layer.forward(input, residual)
                quantized, token_scale = quant_layer.forward(normed, None, scale_ub)
                return quantized, new_residual, token_scale

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
) -> None:
    """(b): torch.compile(native_impl), adapted from the reference commit."""
    _, _, _, compiled = _get_layers(input.shape[-1], input.dtype, epsilon)
    quantized, new_residual, token_scale = compiled(input, weight, scale_ub, residual)
    result.copy_(quantized)
    scale.copy_(token_scale)
    if residual is not None:
        residual.copy_(new_residual)


def rms_norm_dynamic_per_token_quant_eager(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> None:
    """Eager (uncompiled) native_impl -- used only as the Helion autotuner's
    fast per-candidate-config accuracy check, not as a reported baseline."""
    rms_norm_layer, quant_layer, native_impl, _ = _get_layers(
        input.shape[-1], input.dtype, epsilon
    )
    quantized, new_residual, token_scale = native_impl(
        input, weight, scale_ub, residual
    )
    result.copy_(quantized)
    scale.copy_(token_scale)
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
) -> None:
    """(c): the kernel's own baseline() -- torch.ops._C, CUDA-only."""
    torch.ops._C.rms_norm_dynamic_per_token_quant(
        result, input, weight, scale, epsilon, scale_ub, residual
    )


def _probe_torch_ops_c() -> bool:
    try:
        x = torch.randn(2, 8, dtype=BF16, device=DEVICE)
        w = torch.ones(8, dtype=BF16, device=DEVICE)
        r = torch.empty_like(x, dtype=FP8)
        s = torch.empty((2, 1), dtype=torch.float32, device=DEVICE)
        res = torch.randn_like(x)
        torch_ops_c_baseline(r, x, w, s, EPSILON, None, res)
        torch.xpu.synchronize()
        return True
    except Exception:
        return False


def make_inputs(num_tokens: int, hidden_size: int):
    input_ = torch.randn(num_tokens, hidden_size, device=DEVICE, dtype=BF16)
    result = torch.empty(input_.shape, device=DEVICE, dtype=FP8)
    scale = torch.empty((num_tokens, 1), device=DEVICE, dtype=torch.float32)
    scale_ub = torch.mean(input_).to(torch.float32)
    residual = torch.randn_like(input_)
    weight = torch.normal(
        mean=1.0, std=1.0, size=(hidden_size,), dtype=BF16, device=DEVICE
    )
    return result, input_, weight, scale, EPSILON, scale_ub, residual


def bench(fn) -> tuple[float, bool]:
    return bench_with_xpu_graph_fallback(fn)


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    o, r = out.float(), ref.float()
    return ((o - r).abs().max() / (r.abs().max() + 1e-6)).item()


def build_kernel(autotune_effort: str = AUTOTUNE_EFFORT):
    """Rebuild rms_norm_dynamic_per_token_quant with a live per-shape
    autotuner (bypassing the single dummy preset config) and an
    XPU-compatible autotune baseline.
    """
    settings = dataclasses.replace(
        _wrapper.helion_settings,
        autotune_baseline_fn=rms_norm_dynamic_per_token_quant_eager,
    )
    return create_helion_decorated_kernel(
        _wrapper.raw_kernel_func,
        settings,
        extra_kwargs={
            "autotune_effort": autotune_effort,
            "autotune_budget_seconds": AUTOTUNE_BUDGET_SECONDS,
        },
    )


def iter_cases(num_tokens_list: list[int]) -> Iterator[tuple[str, str, int, int]]:
    """Yields (case_name, name, hidden_size, num_tokens) for every case."""
    for name, hidden_size in HIDDEN_SIZES.items():
        for num_tokens in num_tokens_list:
            case_name = f"hidden_size_{hidden_size}_num_tokens_{num_tokens}"
            yield case_name, name, hidden_size, num_tokens


def measure_case(
    kernel, hidden_size: int, num_tokens: int, torch_ops_c_available: bool
) -> tuple[float, float, float | None, float, float | None, float, bool]:
    """Measures one (hidden_size, num_tokens) case. Returns (err,
    compiled_ms, ops_c_ms, kern_ms, speedup_vs_ops_c, speedup_vs_compiled,
    xpu_graph_enabled). Raises on failure -- deliberately left uncaught
    here; see run_full_sweep.sh for how a raised exception from
    --only-case is handled from outside the process."""
    result, input_, weight, scale, epsilon, scale_ub, residual = make_inputs(
        num_tokens, hidden_size
    )

    kernel_result, kernel_scale, kernel_residual = (
        result.clone(),
        scale.clone(),
        residual.clone(),
    )
    kernel(
        kernel_result, input_, weight, kernel_scale, epsilon, scale_ub, kernel_residual
    )

    compiled_result, compiled_scale, compiled_residual = (
        result.clone(),
        scale.clone(),
        residual.clone(),
    )
    torch_compile_baseline(
        compiled_result,
        input_,
        weight,
        compiled_scale,
        epsilon,
        scale_ub,
        compiled_residual,
    )
    torch.xpu.synchronize()
    err = max(
        rel_err(kernel_result, compiled_result),
        rel_err(kernel_scale, compiled_scale),
    )

    compiled_buf = (
        result.clone(),
        input_,
        weight,
        scale.clone(),
        epsilon,
        scale_ub,
        residual.clone(),
    )
    kernel_buf = (
        result.clone(),
        input_,
        weight,
        scale.clone(),
        epsilon,
        scale_ub,
        residual.clone(),
    )

    compiled_ms, compiled_graph = bench(
        lambda buf=compiled_buf: torch_compile_baseline(*buf)
    )
    kern_ms, kern_graph = bench(lambda buf=kernel_buf: kernel(*buf))
    xpu_graph_enabled = compiled_graph and kern_graph
    speedup_vs_compiled = compiled_ms / kern_ms if kern_ms > 0 else 0.0

    ops_c_ms = None
    speedup_vs_ops_c = None
    if torch_ops_c_available:
        opsc_buf = (
            result.clone(),
            input_,
            weight,
            scale.clone(),
            epsilon,
            scale_ub,
            residual.clone(),
        )
        ops_c_ms, _ = bench(lambda buf=opsc_buf: torch_ops_c_baseline(*buf))
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
        f"{name:12s} num_tokens={num_tokens:5d} hidden_size={hidden_size:6d}  "
        f"rel_err={err:.4f}  compiled_ms={compiled_ms:9.5f}  ops_c_ms={ops_c_str}  "
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
        "--autotune-effort", default=AUTOTUNE_EFFORT, choices=["none", "quick", "full"]
    )
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
            "torch.ops._C.rms_norm_dynamic_per_token_quant is not available "
            "on this platform (CUDA-only); reporting torch.compile(native) "
            "only, no torch.ops._C column.\n"
        )

    kernel = build_kernel(args.autotune_effort)

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
