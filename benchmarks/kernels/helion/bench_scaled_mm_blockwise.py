# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level benchmark for the Helion ``scaled_mm_blockwise`` kernel on
Intel XPU.

Companion to ``bench_scaled_mm.py``. Default sweeps a representative
subset: one [K, N] projection shape per Qwen3 model (qkv_proj), a single
num_tokens value, and DeepSeek-style blockwise fp8 quantization
(group_m=1, group_k=128, group_n=128). ``--full`` sweeps the complete
grid: all 12 [K, N] shapes (matching ``bench_scaled_mm.py``'s ``B_SHAPES``
exactly) x this kernel's own 12-value num_tokens list
(4, 8, ..., 8192 -- verified against a real reference table, see
SHAPE_AUDIT.md; unlike ``scaled_mm``, this kernel's own
``generate_inputs()`` list *is* the right source here).

Baseline note: unlike ``scaled_mm``, this kernel's own ``baseline()`` (see
vllm/kernels/helion/ops/scaled_mm_blockwise.py) is already pure portable
PyTorch math (dequantize with ``group_broadcast`` + ``torch.mm``) -- it
does not call CUTLASS or any custom op, so it runs fine on XPU as-is and
needs no substitute for the autotuner's per-config accuracy gate.

For the *performance* comparison, this benchmark uses ``torch._scaled_mm``
with block-wise ``scale_a``/``scale_b`` (verified on this XPU build to
accept the same 1x128 / 128x128 DeepSeek-style scale shapes as this kernel,
and to match ``baseline()`` numerically) as the native reference, rather
than ``baseline()`` itself: ``baseline()`` upcasts the full operands to
float32 and does the group-scale broadcast in Python before an fp32
``torch.mm``, which is not representative of a real optimized alternative.
This mirrors the ``scaled_mm_native`` pattern in ``bench_scaled_mm.py``.

Usage:
    python benchmarks/kernels/helion/bench_scaled_mm_blockwise.py
    python benchmarks/kernels/helion/bench_scaled_mm_blockwise.py --full

Crash-safe full-sweep mode (see ``run_full_sweep.sh``):
    python bench_scaled_mm_blockwise.py --full --list-cases
    python bench_scaled_mm_blockwise.py --full \\
        --only-case <name> --sweep-file benchmark_logs/sweep_....jsonl
    python bench_scaled_mm_blockwise.py --full \\
        --report-from-sweep benchmark_logs/sweep_....jsonl
"""

from __future__ import annotations

import argparse
import math
import time
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

from vllm.kernels.helion.ops.scaled_mm_blockwise import (
    scaled_mm_blockwise as _wrapper,
)
from vllm.kernels.helion.register import create_helion_decorated_kernel
from vllm.platforms import current_platform

FP8 = current_platform.fp8_dtype()
BF16 = torch.bfloat16
DEVICE = current_platform.device_type

# [K, N] projection dims -- full 12-shape grid, identical to
# bench_scaled_mm.py's B_SHAPES (verified against this kernel's own
# generate_inputs(), see SHAPE_AUDIT.md). Default (--full not given) only
# sweeps the first 3 (one qkv_proj shape per model), matching the previous
# representative subset.
B_SHAPES: dict[str, tuple[int, int]] = {
    "Qwen3-1.7B/qkv_proj": (2048, 4096),
    "Qwen3-1.7B/out_proj": (2048, 2048),
    "Qwen3-1.7B/gate_up": (2048, 12288),
    "Qwen3-1.7B/down_proj": (6144, 2048),
    "Qwen3-8B/qkv_proj": (4096, 6144),
    "Qwen3-8B/out_proj": (4096, 4096),
    "Qwen3-8B/gate_up": (4096, 24576),
    "Qwen3-8B/down_proj": (12288, 4096),
    "Qwen3-32B/qkv_proj": (5120, 10240),
    "Qwen3-32B/out_proj": (5120, 5120),
    "Qwen3-32B/gate_up": (5120, 51200),
    "Qwen3-32B/down_proj": (25600, 5120),
}
B_SHAPES_QUICK = {
    k: v
    for k, v in B_SHAPES.items()
    if k in ("Qwen3-1.7B/qkv_proj", "Qwen3-8B/qkv_proj", "Qwen3-32B/qkv_proj")
}
NUM_TOKENS_FULL = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
NUM_TOKENS_QUICK = [128]
GROUP_M, GROUP_K, GROUP_N = 1, 128, 128
AUTOTUNE_EFFORT = "quick"
AUTOTUNE_BUDGET_SECONDS = 45


def scaled_mm_bw_native(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    group_m: int,
    group_k: int,
    group_n: int,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """XPU-native blockwise baseline: torch._scaled_mm (see module docstring).

    torch._scaled_mm's block-wise scaling recipe expects group shapes
    matching this kernel's own group_m=1/group_k=128/group_n=128 scheme,
    so scale_a/scale_b are passed through unchanged.
    """
    assert group_m == 1 and group_k == 128 and group_n == 128
    out = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)
    if bias is not None:
        out = out + bias
    return out


def make_inputs(
    M: int, K: int, N: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
    scale = 1.0 / math.sqrt(K)
    a = (scale * (0.5 + torch.rand(M, K, dtype=torch.float32, device=DEVICE))).to(FP8)
    b = (
        (scale * (0.5 + torch.rand(N, K, dtype=torch.float32, device=DEVICE)))
        .to(FP8)
        .t()
    )
    num_group_m = M // GROUP_M
    num_group_k = K // GROUP_K
    num_group_n = N // GROUP_N
    scale_a = 0.5 + torch.rand(
        num_group_m, num_group_k, dtype=torch.float32, device=DEVICE
    )
    scale_b = 0.5 + torch.rand(
        num_group_k, num_group_n, dtype=torch.float32, device=DEVICE
    )
    # make scales M-major / K-major, matching generate_inputs()'s layout
    scale_a = scale_a.t().contiguous().t()
    scale_b = scale_b.t().contiguous().t()
    # bias not yet supported by scaled_mm_blockwise (kernel asserts bias is
    # None); keep None to avoid failure during benchmarking.
    bias = None
    return a, b, scale_a, scale_b, bias


def bench(fn) -> tuple[float, bool]:
    return bench_with_xpu_graph_fallback(fn)


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    o, r = out.float(), ref.float()
    return ((o - r).abs().max() / (r.abs().max() + 1e-6)).item()


def build_kernel(autotune_effort: str, autotune_budget_seconds: float):
    """Rebuild scaled_mm_blockwise with a live per-shape autotuner
    (bypassing the single dummy preset config installed for production
    dispatch). No autotune_baseline_fn override is needed here -- see
    module docstring.
    """
    return create_helion_decorated_kernel(
        _wrapper.raw_kernel_func,
        _wrapper.helion_settings,
        extra_kwargs={
            "autotune_effort": autotune_effort,
            "autotune_budget_seconds": autotune_budget_seconds,
        },
    )


def iter_cases(
    b_shapes: dict[str, tuple[int, int]], num_tokens_list: list[int]
) -> Iterator[tuple[str, str, int, int, int]]:
    """Yields (case_name, name, M, K, N) for every case in the grid."""
    for name, (K, N) in b_shapes.items():
        for M in num_tokens_list:
            case_name = f"{name.replace('/', '_')}_M_{M}_K_{K}_N_{N}"
            yield case_name, name, M, K, N


def measure_case(
    kernel, M: int, K: int, N: int
) -> tuple[float, float, float, float, bool, float]:
    """Measures one (M, K, N) case. Returns (err, base_ms, kern_ms,
    speedup, xpu_graph_enabled, autotune_s). Raises on failure --
    deliberately left uncaught here; see run_full_sweep.sh for how a
    raised exception from --only-case is handled from outside the
    process."""
    a, b, scale_a, scale_b, bias = make_inputs(M, K, N)
    call_args = (a, b, scale_a, scale_b, GROUP_M, GROUP_K, GROUP_N, BF16, bias)

    # First call for this shape triggers autotuning + compilation.
    torch.xpu.synchronize()
    t0 = time.time()
    out = kernel(*call_args)
    torch.xpu.synchronize()
    autotune_s = time.time() - t0

    ref = scaled_mm_bw_native(*call_args)
    torch.xpu.synchronize()
    err = rel_err(out, ref)

    base_ms, base_graph = bench(lambda ca=call_args: scaled_mm_bw_native(*ca))
    kern_ms, kern_graph = bench(lambda ca=call_args: kernel(*ca))
    xpu_graph_enabled = base_graph and kern_graph
    speedup = base_ms / kern_ms if kern_ms > 0 else 0.0

    return err, base_ms, kern_ms, speedup, xpu_graph_enabled, autotune_s


def _print_case_result(
    name: str,
    M: int,
    K: int,
    N: int,
    err: float,
    base_ms: float,
    kern_ms: float,
    speedup: float,
    xpu_graph_enabled: bool,
) -> None:
    xpu_graph_str = "enabled" if xpu_graph_enabled else "disabled (fallback)"
    print(
        f"{name:22s} M={M:5d} K={K:6d} N={N:6d}  rel_err={err:.4f}  "
        f"native_ms={base_ms:9.5f}  helion_ms={kern_ms:9.5f}  "
        f"speedup={speedup:6.3f}x  xpu_graph={xpu_graph_str}"
    )


def _print_summary_and_report(
    rows: list[tuple[str, int, int, int, float, float, float, float, bool]],
    detailed_rows: list[tuple[str, float, float]],
    autotune_wall_time: float,
    n_not_attempted: int = 0,
) -> None:
    speedups = [r[-2] for r in rows if r[-2] > 0]
    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(
            f"\ngeomean speedup vs torch._scaled_mm (XPU native) over "
            f"{len(rows)} shapes: {geo:.3f}x"
        )
    max_err = max((r[4] for r in rows), default=0.0)
    print(f"max rel_err across all shapes: {max_err:.4f}")
    n_graph_enabled = sum(1 for r in rows if r[-1])
    print(f"xpu_graph enabled for {n_graph_enabled}/{len(rows)} shapes")
    print(
        f"autotuning wall-clock time for {len(rows)} shapes: {autotune_wall_time:.1f}s"
    )
    if n_not_attempted:
        print(f"{n_not_attempted} case(s) FAILED or not yet attempted (see above)")

    print_detailed_report(
        hardware=current_platform.get_device_name(),
        baseline_name="torch._scaled_mm (CUTLASS-equivalent)",
        rows=detailed_rows,
    )


def _report_from_sweep(
    sweep_file: str, b_shapes: dict[str, tuple[int, int]], num_tokens_list: list[int]
) -> None:
    results = load_sweep_results(sweep_file)
    rows: list[tuple] = []
    detailed_rows: list[tuple[str, float, float]] = []
    autotune_wall_time = 0.0
    n_not_attempted = 0
    for case_name, name, M, K, N in iter_cases(b_shapes, num_tokens_list):
        rec = results.get(case_name)
        if rec is None or rec.get("status") != "ok":
            n_not_attempted += 1
            reason = "not attempted" if rec is None else "failed"
            print(f"{case_name}: {reason}")
            continue
        rows.append(
            (
                name,
                M,
                K,
                N,
                rec["rel_err"],
                rec["baseline_ms"],
                rec["kernel_ms"],
                rec["speedup"],
                rec["xpu_graph_enabled"],
            )
        )
        autotune_wall_time += rec["autotune_s"]
        detailed_rows.append((case_name, rec["baseline_ms"], rec["kernel_ms"]))
    _print_summary_and_report(rows, detailed_rows, autotune_wall_time, n_not_attempted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Sweep the full grid (12 [K,N] shapes x 12 num_tokens = 144 "
        "shapes). Default sweeps a representative subset (3 qkv_proj "
        "shapes x num_tokens=128).",
    )
    add_crash_safe_args(parser)
    args = parser.parse_args()

    b_shapes = B_SHAPES if args.full else B_SHAPES_QUICK
    num_tokens_list = NUM_TOKENS_FULL if args.full else NUM_TOKENS_QUICK

    if args.list_cases:
        for case_name, *_ in iter_cases(b_shapes, num_tokens_list):
            print_case_name(case_name)
        return

    if args.report_from_sweep:
        _report_from_sweep(args.report_from_sweep, b_shapes, num_tokens_list)
        return

    assert current_platform.is_xpu(), "This benchmark targets Intel XPU"

    kernel = build_kernel(AUTOTUNE_EFFORT, AUTOTUNE_BUDGET_SECONDS)

    if args.only_case:
        if not args.sweep_file:
            raise SystemExit("--only-case requires --sweep-file")
        for case_name, name, M, K, N in iter_cases(b_shapes, num_tokens_list):
            if case_name != args.only_case:
                continue
            err, base_ms, kern_ms, speedup, xpu_graph_enabled, autotune_s = (
                measure_case(kernel, M, K, N)
            )
            _print_case_result(
                name, M, K, N, err, base_ms, kern_ms, speedup, xpu_graph_enabled
            )
            append_sweep_result(
                args.sweep_file,
                case_name,
                name=name,
                M=M,
                K=K,
                N=N,
                rel_err=err,
                baseline_ms=base_ms,
                kernel_ms=kern_ms,
                speedup=speedup,
                xpu_graph_enabled=xpu_graph_enabled,
                autotune_s=autotune_s,
            )
            return
        raise SystemExit(f"Unknown --only-case {args.only_case!r}")

    rows: list[tuple[str, int, int, int, float, float, float, float, bool]] = []
    detailed_rows: list[tuple[str, float, float]] = []
    autotune_wall_time = 0.0
    for case_name, name, M, K, N in iter_cases(b_shapes, num_tokens_list):
        err, base_ms, kern_ms, speedup, xpu_graph_enabled, autotune_s = measure_case(
            kernel, M, K, N
        )
        autotune_wall_time += autotune_s
        detailed_rows.append((case_name, base_ms, kern_ms))
        rows.append((name, M, K, N, err, base_ms, kern_ms, speedup, xpu_graph_enabled))
        _print_case_result(
            name, M, K, N, err, base_ms, kern_ms, speedup, xpu_graph_enabled
        )

    _print_summary_and_report(rows, detailed_rows, autotune_wall_time)


if __name__ == "__main__":
    main()
