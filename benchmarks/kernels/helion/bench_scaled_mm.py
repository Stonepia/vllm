# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level benchmark for the Helion ``scaled_mm`` kernel on Intel XPU.

Reproduces the ``scaled_mm`` row of blog Table 2
(https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/)
on Intel XPU.

Baseline note: the blog (and vllm/kernels/helion/ops/scaled_mm.py's own
``baseline()``) compares against CUTLASS (``ops.cutlass_scaled_mm``),
which is CUDA-only. CUTLASS has no XPU equivalent, and the native-XPU
implementation that vllm_xpu_kernels would otherwise provide is
unavailable in this environment (libsycl.so.8/.so.9 ABI mismatch --
see the "[XPU] Make vllm_xpu_kernels custom-op imports optional"
commit). Per project decision, this benchmark uses ``torch._scaled_mm``
(the native PyTorch fp8 scaled-matmul op, which does run on XPU) as
the comparison baseline instead. This mirrors what the blog itself
does for its "vs torch.compile" columns on other kernels -- comparing
against the best available portable/native reference rather than a
CUDA-specific vendor library.

Shapes match blog Table 1 (Qwen3-1.7B/8B/32B qkv_proj/out_proj/
gate_up/down_proj [K, N] projection dims) x num_tokens powers-of-two.

Usage:
    python benchmarks/kernels/helion/bench_scaled_mm.py
    python benchmarks/kernels/helion/bench_scaled_mm.py --full --autotune-effort full

Crash-safe full-sweep mode (see ``run_full_sweep.sh`` and RESULTS.md's
"What's not done" -- some large shapes OOM the XPU when benchmarked, and a
crash while measuring one shape shouldn't lose the rest of the sweep):
    python benchmarks/kernels/helion/bench_scaled_mm.py --full --list-cases
    python benchmarks/kernels/helion/bench_scaled_mm.py --full \\
        --only-case <name> --sweep-file benchmark_logs/sweep_scaled_mm.jsonl
    python benchmarks/kernels/helion/bench_scaled_mm.py --full \\
        --report-from-sweep benchmark_logs/sweep_scaled_mm.jsonl
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

from vllm.kernels.helion.ops.scaled_mm import scaled_mm as _wrapper
from vllm.kernels.helion.register import create_helion_decorated_kernel
from vllm.platforms import current_platform

FP8 = current_platform.fp8_dtype()
BF16 = torch.bfloat16
DEVICE = current_platform.device_type

# [K, N] projection dims, blog Tab. 1 ("Projection layer [K, N] dimensions
# for each Qwen3 model").
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
NUM_TOKENS_FULL = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
NUM_TOKENS_QUICK = [16, 128, 1024]


def scaled_mm_native(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """XPU-native baseline: torch._scaled_mm (see module docstring)."""
    out = torch._scaled_mm(
        a, b, scale_a=scale_a, scale_b=scale_b.T, out_dtype=out_dtype
    )
    if bias is not None:
        out = out + bias
    return out


def make_inputs(
    M: int, K: int, N: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = 1.0 / math.sqrt(K)
    a = (scale * (0.5 + torch.rand(M, K, dtype=torch.float32, device=DEVICE))).to(FP8)
    b = (
        (scale * (0.5 + torch.rand(N, K, dtype=torch.float32, device=DEVICE)))
        .to(FP8)
        .t()
    )
    scale_a = 0.5 + torch.rand((M, 1), dtype=torch.float32, device=DEVICE)
    scale_b = 0.5 + torch.rand((N, 1), dtype=torch.float32, device=DEVICE)
    bias = 0.5 * (torch.rand(N, dtype=BF16, device=DEVICE) - 0.5)
    return a, b, scale_a, scale_b, bias


def bench(fn) -> tuple[float, bool]:
    return bench_with_xpu_graph_fallback(fn)


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    o, r = out.float(), ref.float()
    return ((o - r).abs().max() / (r.abs().max() + 1e-6)).item()


def build_kernel(autotune_effort: str):
    """Rebuild scaled_mm with a live per-shape autotuner (bypassing the
    single dummy preset config) and an XPU-compatible autotune baseline.

    The registered op's own helion_settings.autotune_baseline_fn points at
    baseline() (CUTLASS, CUDA-only); swap in scaled_mm_native so the
    autotuner's per-candidate-config accuracy check works on XPU.
    """
    settings = dataclasses.replace(
        _wrapper.helion_settings,
        autotune_baseline_fn=scaled_mm_native,
    )
    return create_helion_decorated_kernel(
        _wrapper.raw_kernel_func,
        settings,
        extra_kwargs={"autotune_effort": autotune_effort},
    )


def iter_cases(
    num_tokens_list: list[int],
) -> Iterator[tuple[str, str, int, int, int]]:
    """Yields (case_name, name, M, K, N) for every case in the grid.

    Single source of truth for case naming/ordering, shared by --list-cases,
    --only-case's lookup, the in-process default loop, and
    --report-from-sweep -- so they can never disagree about what a case
    named e.g. "Qwen3-32B_down_proj_M_128_K_25600_N_5120" refers to.
    """
    for name, (K, N) in B_SHAPES.items():
        for M in num_tokens_list:
            case_name = f"{name.replace('/', '_')}_M_{M}_K_{K}_N_{N}"
            yield case_name, name, M, K, N


def measure_case(
    kernel, name: str, M: int, K: int, N: int
) -> tuple[float, float, float, float, bool]:
    """Measures one (name, M, K, N) case. Returns (err, base_ms, kern_ms,
    speedup, xpu_graph_enabled). Raises on failure (XPU OOM/DEVICE_LOST
    etc.) -- deliberately left uncaught here; see run_full_sweep.sh for how
    a raised exception from --only-case is handled from outside the
    process."""
    a, b, scale_a, scale_b, bias = make_inputs(M, K, N)
    call_args = (a, b, scale_a, scale_b, BF16, bias)

    out = kernel(*call_args)
    ref = scaled_mm_native(*call_args)
    torch.xpu.synchronize()
    err = rel_err(out, ref)

    base_ms, base_graph = bench(lambda ca=call_args: scaled_mm_native(*ca))
    kern_ms, kern_graph = bench(lambda ca=call_args: kernel(*ca))
    xpu_graph_enabled = base_graph and kern_graph
    speedup = base_ms / kern_ms if kern_ms > 0 else 0.0
    return err, base_ms, kern_ms, speedup, xpu_graph_enabled


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
    if n_not_attempted:
        print(f"{n_not_attempted} case(s) FAILED or not yet attempted (see above)")

    print_detailed_report(
        hardware=current_platform.get_device_name(),
        baseline_name="torch._scaled_mm (CUTLASS-equivalent)",
        rows=detailed_rows,
    )


def _report_from_sweep(sweep_file: str, num_tokens_list: list[int]) -> None:
    """Reads a --sweep-file built up by many --only-case invocations (see
    run_full_sweep.sh) and prints the same summary/detailed report the
    in-process loop below would, without re-measuring anything."""
    results = load_sweep_results(sweep_file)
    rows: list[tuple[str, int, int, int, float, float, float, float, bool]] = []
    detailed_rows: list[tuple[str, float, float]] = []
    n_not_attempted = 0
    for case_name, name, M, K, N in iter_cases(num_tokens_list):
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
        detailed_rows.append((case_name, rec["baseline_ms"], rec["kernel_ms"]))
    _print_summary_and_report(rows, detailed_rows, n_not_attempted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Sweep the full blog shape grid (14 num_tokens x 12 [K,N] = "
        "168 shapes; slow, full autotuning per blog's Caveats section can "
        "take a long time). Default sweeps a representative subset.",
    )
    parser.add_argument(
        "--autotune-effort", default="quick", choices=["none", "quick", "full"]
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
    kernel = build_kernel(args.autotune_effort)

    if args.only_case:
        if not args.sweep_file:
            raise SystemExit("--only-case requires --sweep-file")
        for case_name, name, M, K, N in iter_cases(num_tokens_list):
            if case_name != args.only_case:
                continue
            err, base_ms, kern_ms, speedup, xpu_graph_enabled = measure_case(
                kernel, name, M, K, N
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
            )
            return
        raise SystemExit(f"Unknown --only-case {args.only_case!r}")

    rows: list[tuple[str, int, int, int, float, float, float, float, bool]] = []
    detailed_rows: list[tuple[str, float, float]] = []
    for case_name, name, M, K, N in iter_cases(num_tokens_list):
        err, base_ms, kern_ms, speedup, xpu_graph_enabled = measure_case(
            kernel, name, M, K, N
        )
        detailed_rows.append((case_name, base_ms, kern_ms))
        rows.append((name, M, K, N, err, base_ms, kern_ms, speedup, xpu_graph_enabled))
        _print_case_result(
            name, M, K, N, err, base_ms, kern_ms, speedup, xpu_graph_enabled
        )

    _print_summary_and_report(rows, detailed_rows)


if __name__ == "__main__":
    main()
