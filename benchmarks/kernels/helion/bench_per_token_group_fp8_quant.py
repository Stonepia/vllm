# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level benchmark for the Helion ``per_token_group_fp8_quant`` kernel
on Intel XPU.

Three-way comparison, matching the blog's own methodology
(https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/):

  a) the Helion kernel
  b) ``torch.compile(native_impl)`` -- native_impl is
     ``QuantFP8(static=False, group_shape=GroupShape(1, group_size))``,
     matching the reference end-to-end-benchmark dev branch exactly
     (xiaohongchen1991/vllm@d9b0566a54737c1b0a8e3714ef7a3ee40a4442c2,
     vllm/kernels/helion/ops/per_token_group_fp8_quant.py: ``compiled_layer =
     torch.compile(layer.forward, fullgraph=True, dynamic=False,
     backend="inductor", options={...})``, ``baseline()`` there returns
     ``compiled_layer(input, None)``). Note this reference implementation
     itself only takes ``input``/``scale`` -- QuantFP8's own internal
     eps/clamp constants are used, not this kernel's explicit
     eps/fp8_min/fp8_max/scale_ue8m0 parameters (a difference already
     present in the reference commit, not introduced here; inert for
     realistic non-near-zero inputs, same as this kernel's own correctness
     tests already tolerate).
  c) ``torch.ops._C.per_token_group_fp8_quant`` -- this kernel's own
     ``baseline()`` (in vllm/kernels/helion/ops/per_token_group_fp8_quant.py)
     calls this directly. CUDA-only: not registered at all in this
     environment (confirmed via hasattr check) -- vLLM's own
     fp8_utils.per_token_group_quant_fp8()'s is_xpu() branch unconditionally
     dispatches to this same missing op too, a pre-existing production gap
     unrelated to this task. Probed once at import time; reports N/A here.

Autotune accuracy-check tolerance: Helion's default per-candidate-config
accuracy check is too tight for this kernel's legitimate fp8-rounding-
boundary noise (see _accuracy_check below); overridden the same way as
before, just pointed at the new QuantFP8-based reference.

Shapes: one per Qwen3 model hidden_size (1.7B=2048, 8B=4096, 32B=5120),
group_size=128 for all three, num_tokens=128 by default; ``--full`` sweeps
the blog's full 14-value num_tokens grid (1..8192) per SHAPE_AUDIT.md.

Usage:
    python benchmarks/kernels/helion/bench_per_token_group_fp8_quant.py
    python benchmarks/kernels/helion/bench_per_token_group_fp8_quant.py --full

Crash-safe full-sweep mode (see ``run_full_sweep.sh``):
    python bench_per_token_group_fp8_quant.py --full --list-cases
    python bench_per_token_group_fp8_quant.py --full \\
        --only-case <name> --sweep-file benchmark_logs/sweep_....jsonl
    python bench_per_token_group_fp8_quant.py --full \\
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
from vllm.kernels.helion.ops.per_token_group_fp8_quant import (
    per_token_group_fp8_quant as _wrapper,
)
from vllm.kernels.helion.register import create_helion_decorated_kernel
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    get_fp8_min_max,
)
from vllm.platforms import current_platform

FP8 = current_platform.fp8_dtype()
BF16 = torch.bfloat16
DEVICE = current_platform.device_type
FP8_MIN, FP8_MAX = get_fp8_min_max()
EPS = 1e-10
GROUP_SIZE = 128

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

# torch.compile options copied verbatim from the reference commit.
_INDUCTOR_OPTIONS = {
    "enable_auto_functionalized_v2": False,
    "size_asserts": False,
    "alignment_asserts": False,
    "scalar_asserts": False,
    "combo_kernels": True,
    "benchmark_combo_kernel": True,
}

_QUANT_OP: QuantFP8 | None = None
_COMPILED_NATIVE = None


def per_token_group_fp8_quant_eager(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    """Eager QuantFP8.forward_native -- used only as the Helion autotuner's
    fast per-candidate-config accuracy check, not as a reported baseline."""
    assert _QUANT_OP is not None
    quantized, scale = _QUANT_OP.forward_native(input, scale=None)
    output_q.copy_(quantized)
    output_s.copy_(scale)


def torch_compile_baseline(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    """(b): torch.compile(native_impl), ported from the reference commit."""
    assert _COMPILED_NATIVE is not None
    quantized, scale = _COMPILED_NATIVE(input, None)
    output_q.copy_(quantized)
    output_s.copy_(scale)


def torch_ops_c_baseline(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    """(c): the kernel's own baseline() -- torch.ops._C, CUDA-only."""
    torch.ops._C.per_token_group_fp8_quant(
        input,
        output_q,
        output_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        dummy_is_scale_transposed,
        dummy_is_tma_aligned,
    )


def _probe_torch_ops_c() -> bool:
    if not hasattr(torch.ops._C, "per_token_group_fp8_quant"):
        return False
    try:
        x = torch.randn(2, GROUP_SIZE, dtype=BF16, device=DEVICE)
        q = torch.empty_like(x, dtype=FP8)
        s = torch.empty((2, 1), dtype=torch.float32, device=DEVICE)
        torch_ops_c_baseline(x, q, s, GROUP_SIZE, EPS, FP8_MIN, FP8_MAX, False)
        torch.xpu.synchronize()
        return True
    except Exception:
        return False


def make_inputs(
    num_tokens: int, hidden_size: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input = torch.randn(num_tokens, hidden_size, dtype=BF16, device=DEVICE)
    output_q = torch.empty(input.shape, dtype=FP8, device=DEVICE)
    output_s = torch.empty(
        (num_tokens, hidden_size // group_size), dtype=torch.float32, device=DEVICE
    )
    return input, output_q, output_s


def bench(fn) -> tuple[float, bool]:
    return bench_with_xpu_graph_fallback(fn)


def _accuracy_check(actual: object, expected: object) -> None:
    """Custom autotune_baseline_accuracy_check_fn -- see the module docstring
    of the earlier version of this file (git history) for the full
    explanation of why Helion's default check is too tight for this
    kernel's legitimate fp8-rounding-boundary noise; unchanged logic, just
    pointed at the new QuantFP8-based native_impl.
    """
    if actual is None or expected is None:
        return
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        pairs = [(actual, expected)]
    else:
        pairs = [
            (a, e)
            for a, e in zip(actual, expected, strict=True)
            if isinstance(a, torch.Tensor) and isinstance(e, torch.Tensor)
        ]
    for a, e in pairs:
        assert_close_with_mismatch_tolerance(
            a.to(torch.float32),
            e.to(torch.float32),
            atol=1e-5,
            rtol=1e-3,
            max_mismatch_pct=0.10,
            max_abs_diff=None,
            max_rel_diff=None,
        )


def rel_err(
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    ref_q: torch.Tensor,
    ref_s: torch.Tensor,
    group_size: int,
) -> float:
    """Relative error of the fully dequantized (value * scale) output."""
    num_tokens, hidden_size = output_q.shape
    out_q_grouped = output_q.float().view(num_tokens, -1, group_size)
    ref_q_grouped = ref_q.float().view(num_tokens, -1, group_size)
    out_deq = out_q_grouped * output_s.float().unsqueeze(-1)
    ref_deq = ref_q_grouped * ref_s.float().unsqueeze(-1)
    return ((out_deq - ref_deq).abs().max() / (ref_deq.abs().max() + 1e-6)).item()


def build_kernel(autotune_effort: str, autotune_budget_seconds: float):
    """Rebuild per_token_group_fp8_quant with a live per-shape autotuner
    (bypassing the single dummy preset config) and an XPU-compatible
    autotune baseline + relaxed accuracy-check tolerance.
    """
    settings = dataclasses.replace(
        _wrapper.helion_settings,
        autotune_baseline_fn=per_token_group_fp8_quant_eager,
        autotune_baseline_accuracy_check_fn=_accuracy_check,
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
    input, output_q, output_s = make_inputs(num_tokens, hidden_size, GROUP_SIZE)
    call_args = (
        input,
        output_q,
        output_s,
        GROUP_SIZE,
        EPS,
        FP8_MIN,
        FP8_MAX,
        False,  # scale_ue8m0
        False,  # dummy_is_scale_transposed
        False,  # dummy_is_tma_aligned
    )

    kernel(*call_args)
    kernel_q, kernel_s = output_q.clone(), output_s.clone()

    compiled_q, compiled_s = torch.empty_like(output_q), torch.empty_like(output_s)
    torch_compile_baseline(
        input, compiled_q, compiled_s, GROUP_SIZE, EPS, FP8_MIN, FP8_MAX, False
    )
    torch.xpu.synchronize()
    err = rel_err(kernel_q, kernel_s, compiled_q, compiled_s, GROUP_SIZE)

    compiled_ms, compiled_graph = bench(
        lambda i=input, cq=compiled_q, cs=compiled_s: torch_compile_baseline(
            i, cq, cs, GROUP_SIZE, EPS, FP8_MIN, FP8_MAX, False
        )
    )
    kern_ms, kern_graph = bench(lambda ca=call_args: kernel(*ca))
    xpu_graph_enabled = compiled_graph and kern_graph
    speedup_vs_compiled = compiled_ms / kern_ms if kern_ms > 0 else 0.0

    ops_c_ms = None
    speedup_vs_ops_c = None
    if torch_ops_c_available:
        opsc_q, opsc_s = torch.empty_like(output_q), torch.empty_like(output_s)
        torch_ops_c_baseline(
            input, opsc_q, opsc_s, GROUP_SIZE, EPS, FP8_MIN, FP8_MAX, False
        )
        torch.xpu.synchronize()
        ops_c_ms, _ = bench(
            lambda i=input, oq=opsc_q, os_=opsc_s: torch_ops_c_baseline(
                i, oq, os_, GROUP_SIZE, EPS, FP8_MIN, FP8_MAX, False
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
        f"{name:10s} num_tokens={num_tokens:5d} hidden_size={hidden_size:5d} "
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
        _QUANT_OP = QuantFP8(static=False, group_shape=GroupShape(1, GROUP_SIZE))
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
            "torch.ops._C.per_token_group_fp8_quant is not available on "
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
