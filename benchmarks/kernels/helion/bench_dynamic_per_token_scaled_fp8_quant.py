# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level benchmark for the Helion ``dynamic_per_token_scaled_fp8_quant``
kernel on Intel XPU.

Three-way comparison, matching the blog's own methodology
(https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/):

  a) the Helion kernel
  b) ``torch.compile(native_impl)`` -- the native_impl (``QuantFP8``) and the
     exact ``torch.compile(..., fullgraph=True, dynamic=False,
     backend="inductor", options={...})`` wrapping are ported from the
     reference end-to-end-benchmark dev branch
     (xiaohongchen1991/vllm@d9b0566a54737c1b0a8e3714ef7a3ee40a4442c2,
     vllm/kernels/helion/ops/dynamic_per_token_scaled_fp8_quant.py), not
     reimplemented here.
  c) ``torch.ops._C.dynamic_per_token_scaled_fp8_quant`` -- this kernel's own
     ``baseline()`` (in
     vllm/kernels/helion/ops/dynamic_per_token_scaled_fp8_quant.py) calls this
     directly. It is CUDA-only: not registered at all in this environment
     (vllm_xpu_kernels' native XPU implementation exists but is unusable here
     due to a libsycl ABI mismatch -- see RESULTS.md). Probed once at import
     time; reported as N/A when unavailable, per (b) as the fallback.

Shapes: one per Qwen3 model hidden_size (1.7B/8B/32B), num_tokens=128 by
default (a representative decode/short-prefill batch size); ``--full``
sweeps the blog's full 14-value num_tokens grid (1..8192) per
SHAPE_AUDIT.md.

Usage:
    python benchmarks/kernels/helion/bench_dynamic_per_token_scaled_fp8_quant.py
    python benchmarks/kernels/helion/bench_dynamic_per_token_scaled_fp8_quant.py --full

Crash-safe full-sweep mode (see ``run_full_sweep.sh``):
    python bench_dynamic_per_token_scaled_fp8_quant.py --full --list-cases
    python bench_dynamic_per_token_scaled_fp8_quant.py --full \\
        --only-case <name> --sweep-file benchmark_logs/sweep_....jsonl
    python bench_dynamic_per_token_scaled_fp8_quant.py --full \\
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
from vllm.kernels.helion.ops.dynamic_per_token_scaled_fp8_quant import (
    dynamic_per_token_scaled_fp8_quant as _wrapper,
)
from vllm.kernels.helion.register import create_helion_decorated_kernel
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

FP8 = current_platform.fp8_dtype()
BF16 = torch.bfloat16
DEVICE = current_platform.device_type

# Hidden sizes match each Qwen3 model's hidden_size (the width of the
# activation tensor that gets quantized, e.g. before a down_proj/o_proj
# matmul).
HIDDEN_SIZES: dict[str, int] = {
    "Qwen3-1.7B": 2048,
    "Qwen3-8B": 4096,
    "Qwen3-32B": 5120,
}
NUM_TOKENS_FULL = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
NUM_TOKENS_QUICK = [128]

# QuantFP8 is a CustomOp/nn.Module; it requires an active VllmConfig context
# to construct (but not to call forward_native/the compiled forward
# afterwards). Constructed once in main().
_QUANT_OP: QuantFP8 | None = None
_COMPILED_NATIVE = None

# torch.compile options copied verbatim from the reference commit -- these
# match the blog's own documented benchmark setup exactly.
_INDUCTOR_OPTIONS = {
    "enable_auto_functionalized_v2": False,
    "size_asserts": False,
    "alignment_asserts": False,
    "scalar_asserts": False,
    "combo_kernels": True,
    "benchmark_combo_kernel": True,
}


def dynamic_per_token_scaled_fp8_quant_native(
    result: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    scale_ub: torch.Tensor | None,
) -> None:
    """Eager QuantFP8.forward_native -- used only as the Helion autotuner's
    fast per-candidate-config accuracy check (see build_kernel()), not as a
    reported benchmark baseline (that's torch_compile_baseline below)."""
    assert _QUANT_OP is not None
    quantized, computed_scale = _QUANT_OP.forward_native(
        input, scale=None, scale_ub=scale_ub
    )
    result.copy_(quantized)
    scale.copy_(computed_scale)


def torch_compile_baseline(
    result: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    scale_ub: torch.Tensor | None,
) -> None:
    """(b): torch.compile(native_impl), ported from the reference commit."""
    assert _COMPILED_NATIVE is not None
    quantized, computed_scale = _COMPILED_NATIVE(input, None, scale_ub)
    result.copy_(quantized)
    scale.copy_(computed_scale)


def torch_ops_c_baseline(
    result: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    scale_ub: torch.Tensor | None,
) -> None:
    """(c): the kernel's own baseline() -- torch.ops._C, CUDA-only."""
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(result, input, scale, scale_ub)


def _probe_torch_ops_c() -> bool:
    try:
        x = torch.randn(2, 8, dtype=BF16, device=DEVICE)
        r = torch.empty_like(x, dtype=FP8)
        s = torch.empty((2, 1), dtype=torch.float32, device=DEVICE)
        torch_ops_c_baseline(r, x, s, None)
        torch.xpu.synchronize()
        return True
    except Exception:
        return False


def make_inputs(
    num_tokens: int, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input = torch.randn(num_tokens, hidden_size, dtype=BF16, device=DEVICE)
    result = torch.empty(input.shape, dtype=FP8, device=DEVICE)
    scale = torch.empty((num_tokens, 1), dtype=torch.float32, device=DEVICE)
    scale_ub = torch.mean(input).to(torch.float32)
    return result, input, scale, scale_ub


def bench(fn) -> tuple[float, bool]:
    return bench_with_xpu_graph_fallback(fn)


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    o, r = out.float(), ref.float()
    return ((o - r).abs().max() / (r.abs().max() + 1e-6)).item()


def build_kernel(autotune_effort: str, autotune_budget_seconds: float):
    """Rebuild the kernel with a live per-shape autotuner (bypassing the
    single dummy preset config) and an XPU-compatible autotune baseline.

    The registered op's own helion_settings.autotune_baseline_fn points at
    baseline() (torch.ops._C.*, CUDA-only); swap in
    dynamic_per_token_scaled_fp8_quant_native so the autotuner's
    per-candidate-config accuracy check works on XPU.
    """
    settings = dataclasses.replace(
        _wrapper.helion_settings,
        autotune_baseline_fn=dynamic_per_token_scaled_fp8_quant_native,
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
    result, input, scale, scale_ub = make_inputs(num_tokens, hidden_size)
    call_args = (result, input, scale, scale_ub)

    kernel(*call_args)
    kernel_out, kernel_scale = result.clone(), scale.clone()

    compiled_out = torch.empty_like(result)
    compiled_scale = torch.empty_like(scale)
    torch_compile_baseline(compiled_out, input, compiled_scale, scale_ub)
    torch.xpu.synchronize()

    err = max(rel_err(kernel_out, compiled_out), rel_err(kernel_scale, compiled_scale))

    compiled_ms, compiled_graph = bench(
        lambda co=compiled_out, i=input, cs=compiled_scale, su=scale_ub: (
            torch_compile_baseline(co, i, cs, su)
        )
    )
    kern_ms, kern_graph = bench(lambda ca=call_args: kernel(*ca))
    xpu_graph_enabled = compiled_graph and kern_graph
    speedup_vs_compiled = compiled_ms / kern_ms if kern_ms > 0 else 0.0

    ops_c_ms = None
    speedup_vs_ops_c = None
    if torch_ops_c_available:
        opsc_out = torch.empty_like(result)
        opsc_scale = torch.empty_like(scale)
        torch_ops_c_baseline(opsc_out, input, opsc_scale, scale_ub)
        torch.xpu.synchronize()
        ops_c_ms, _ = bench(
            lambda oo=opsc_out, i=input, os_=opsc_scale, su=scale_ub: (
                torch_ops_c_baseline(oo, i, os_, su)
            )
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
        f"rel_err={err:.4f}  compiled_ms={compiled_ms:9.5f}  "
        f"ops_c_ms={ops_c_str}  helion_ms={kern_ms:9.5f}  "
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
    parser.add_argument("--autotune-budget-seconds", type=float, default=45.0)
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

    global _QUANT_OP, _COMPILED_NATIVE
    with set_current_vllm_config(VllmConfig()):
        _QUANT_OP = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
        _COMPILED_NATIVE = torch.compile(
            _QUANT_OP.forward,
            fullgraph=True,
            dynamic=False,
            backend="inductor",
            options=_INDUCTOR_OPTIONS,
        )

    torch_ops_c_available = _probe_torch_ops_c()
    if not torch_ops_c_available:
        print(
            "torch.ops._C.dynamic_per_token_scaled_fp8_quant is not available "
            "on this platform (CUDA-only); reporting torch.compile(native) "
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

    rows: list[tuple[str, float, float, float | None, float, float | None, float, bool]]
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
