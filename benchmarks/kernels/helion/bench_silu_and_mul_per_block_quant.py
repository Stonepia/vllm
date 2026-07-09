# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level benchmark for the Helion ``silu_and_mul_per_block_quant``
kernel on Intel XPU.

** NOTE **: this kernel is currently kept DISABLED on XPU (no
intel_arc_pro_b70.json config committed) due to an unresolved Helion bug --
see RESULTS.md and tests/kernels/helion/test_silu_and_mul_per_block_quant.py
for the full root-cause. This benchmark still runs (it builds a kernel
directly via create_helion_decorated_kernel, bypassing the disabled
registered wrapper) and is safe to use as long as num_tokens is held fixed
across shapes within one process (as it is here: NUM_TOKENS=128 throughout,
only intermediate_size -- which is hl.specialize()'d, giving each shape its
own independently-compiled kernel -- varies).

Three-way comparison, matching the blog's own methodology
(https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/):

  a) the Helion kernel
  b) ``torch.compile(native_impl)`` -- native_impl composes
     ``SiluAndMul.forward_native`` + ``QuantFP8(group_shape=GroupShape(1,
     128))``, matching the ``Layer`` class in the reference
     end-to-end-benchmark dev branch
     (xiaohongchen1991/vllm@d9b0566a54737c1b0a8e3714ef7a3ee40a4442c2,
     vllm/kernels/helion/ops/silu_and_mul_per_block_quant.py). Wrapped with
     the same ``torch.compile(..., fullgraph=True, dynamic=False,
     backend="inductor", options={...})`` call as the reference commit.
  c) ``torch.ops._C.silu_and_mul_per_block_quant`` -- this kernel's own
     ``baseline()`` calls this directly. CUDA-only; probed once at import
     time, reports N/A here (confirmed unavailable in this environment).

Shapes: one per Qwen3 model's intermediate_size, derived the same way as
bench_scaled_mm.py's gate_up [K, N] dims (N / 2, since gate_up's output is
the concatenated [gate, up] projection): 1.7B->6144, 8B->12288, 32B->25600.
num_tokens=128, group_size=128 (matches the values already used in this
kernel's own generate_inputs()).

``--full`` sweeps the blog's full 14-value num_tokens grid (1..8192) per
SHAPE_AUDIT.md, but -- because of the known bug above -- ONLY via the
crash-safe per-case path (--list-cases/--only-case/--report-from-sweep,
see run_full_sweep.sh), never via the plain in-process default loop: each
--only-case subprocess builds a fresh kernel and calls it for exactly one
num_tokens value, so the "a different num_tokens reuses the wrong
compiled specialization" failure mode above cannot arise (there's no
second, different-num_tokens call in that process to corrupt). Passing
--full without one of those three flags is refused with an explanatory
error rather than silently producing wrong numbers for num_tokens != the
first one tried.

Usage:
    python benchmarks/kernels/helion/bench_silu_and_mul_per_block_quant.py

Crash-safe full-sweep mode (see ``run_full_sweep.sh``):
    python bench_silu_and_mul_per_block_quant.py --full --list-cases
    python bench_silu_and_mul_per_block_quant.py --full \\
        --only-case <name> --sweep-file benchmark_logs/sweep_....jsonl
    python bench_silu_and_mul_per_block_quant.py --full \\
        --report-from-sweep benchmark_logs/sweep_....jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
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

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.kernels.helion.ops.silu_and_mul_per_block_quant import (
    silu_and_mul_per_block_quant as _wrapper,
)
from vllm.kernels.helion.register import create_helion_decorated_kernel
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.platforms import current_platform

BF16 = torch.bfloat16
FP8 = current_platform.fp8_dtype()
DEVICE = current_platform.device_type
GROUP_SIZE = 128
NUM_TOKENS_FULL = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
NUM_TOKENS_QUICK = [128]
AUTOTUNE_EFFORT = "quick"
AUTOTUNE_BUDGET_SECONDS = 45

# intermediate_size per Qwen3 model, derived from bench_scaled_mm.py's
# B_SHAPES gate_up [K, N] dims / 2 (N is the concatenated [gate, up] width):
#   Qwen3-1.7B/gate_up N=12288 -> intermediate_size 6144
#   Qwen3-8B/gate_up   N=24576 -> intermediate_size 12288
#   Qwen3-32B/gate_up  N=51200 -> intermediate_size 25600
INTERMEDIATE_SIZES: dict[str, int] = {
    "Qwen3-1.7B": 6144,
    "Qwen3-8B": 12288,
    "Qwen3-32B": 25600,
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

with set_current_vllm_config(VllmConfig()):
    _QUANT_OP = QuantFP8(static=False, group_shape=GroupShape(1, GROUP_SIZE))


def _native_impl(input: torch.Tensor):
    """Matches the reference commit's Layer.forward exactly."""
    act_result = SiluAndMul.forward_native(input)
    result, scale = _QUANT_OP.forward(act_result, None)
    return result, scale


with set_current_vllm_config(VllmConfig()):
    _COMPILED_NATIVE = torch.compile(
        _native_impl,
        fullgraph=True,
        dynamic=False,
        backend="inductor",
        options=_INDUCTOR_OPTIONS,
    )


def silu_and_mul_per_block_quant_eager(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None,
    is_scale_transposed: bool,
) -> None:
    """Eager native_impl -- used only as the Helion autotuner's fast
    per-candidate-config accuracy check, not as a reported baseline.
    is_scale_transposed is a dummy parameter on this kernel (interface
    parity with torch.ops._C only) and needs no special handling.
    """
    quantized, scale = _native_impl(input)
    out.copy_(quantized)
    scales.copy_(scale)


def torch_compile_baseline(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None,
    is_scale_transposed: bool,
) -> None:
    """(b): torch.compile(native_impl), matching the reference commit."""
    quantized, scale = _COMPILED_NATIVE(input)
    out.copy_(quantized)
    scales.copy_(scale)


def torch_ops_c_baseline(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    scale_ub: torch.Tensor | None,
    is_scale_transposed: bool,
) -> None:
    """(c): the kernel's own baseline() -- torch.ops._C, CUDA-only."""
    torch.ops._C.silu_and_mul_per_block_quant(
        out, input, scales, group_size, scale_ub, is_scale_transposed
    )


def _probe_torch_ops_c() -> bool:
    try:
        x = torch.randn(2, 2 * GROUP_SIZE, dtype=BF16, device=DEVICE)
        o = torch.empty(2, GROUP_SIZE, dtype=FP8, device=DEVICE)
        s = torch.empty((2, 1), dtype=torch.float32, device=DEVICE)
        torch_ops_c_baseline(o, x, s, GROUP_SIZE, None, False)
        torch.xpu.synchronize()
        return True
    except Exception:
        return False


def make_inputs(
    num_tokens: int, intermediate_size: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor | None, bool]:
    input = torch.randn(num_tokens, 2 * intermediate_size, dtype=BF16, device=DEVICE)
    out = torch.empty(num_tokens, intermediate_size, dtype=FP8, device=DEVICE)
    scales = torch.empty(
        num_tokens,
        intermediate_size // group_size,
        dtype=torch.float32,
        device=DEVICE,
    )
    return out, input, scales, group_size, None, False


def bench(fn) -> tuple[float, bool]:
    return bench_with_xpu_graph_fallback(fn)


def rel_err(
    out: torch.Tensor,
    ref: torch.Tensor,
    scales: torch.Tensor,
    ref_scales: torch.Tensor,
    group_size: int,
) -> float:
    n, h = out.shape
    o = (out.float().view(n, -1, group_size) * scales.unsqueeze(-1)).view(n, h)
    r = (ref.float().view(n, -1, group_size) * ref_scales.unsqueeze(-1)).view(n, h)
    return ((o - r).abs().max() / (r.abs().max() + 1e-6)).item()


def build_kernel(autotune_effort: str, autotune_budget_seconds: int):
    """Rebuild the kernel with a live per-shape autotuner (bypassing the
    single dummy preset config, and the disabled registered wrapper -- see
    module docstring) and an XPU-compatible autotune baseline.
    """
    settings = dataclasses.replace(
        _wrapper.helion_settings,
        autotune_baseline_fn=silu_and_mul_per_block_quant_eager,
        autotune_baseline_atol=8.0,
        autotune_baseline_rtol=0.1,
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
    for name, intermediate_size in INTERMEDIATE_SIZES.items():
        for num_tokens in num_tokens_list:
            case_name = (
                f"intermediate_size_{intermediate_size}_group_size_{GROUP_SIZE}_"
                f"num_tokens_{num_tokens}"
            )
            yield case_name, name, intermediate_size, num_tokens


def measure_case(
    kernel, intermediate_size: int, num_tokens: int, torch_ops_c_available: bool
) -> tuple[float, float, float | None, float, float | None, float, bool, float]:
    """Measures one (intermediate_size, num_tokens) case. Returns (err,
    compiled_ms, ops_c_ms, kern_ms, speedup_vs_ops_c, speedup_vs_compiled,
    xpu_graph_enabled, shape_duration). Raises on failure -- deliberately
    left uncaught here; see run_full_sweep.sh for how a raised exception
    from --only-case is handled from outside the process."""
    out, input, scales, group_size, scale_ub, is_scale_transposed = make_inputs(
        num_tokens, intermediate_size, GROUP_SIZE
    )
    call_args = (out, input, scales, group_size, scale_ub, is_scale_transposed)

    shape_start = time.time()
    kernel(*call_args)  # first call for this shape triggers autotuning
    torch.xpu.synchronize()
    shape_duration = time.time() - shape_start

    compiled_out = torch.empty_like(out)
    compiled_scales = torch.empty_like(scales)
    torch_compile_baseline(
        compiled_out,
        input,
        compiled_scales,
        group_size,
        scale_ub,
        is_scale_transposed,
    )
    err = rel_err(out, compiled_out, scales, compiled_scales, group_size)

    compiled_ms, compiled_graph = bench(
        lambda o=compiled_out,
        i=input,
        s=compiled_scales,
        g=group_size,
        su=scale_ub,
        trp=is_scale_transposed: (torch_compile_baseline(o, i, s, g, su, trp))
    )
    kern_ms, kern_graph = bench(lambda call_args=call_args: kernel(*call_args))
    xpu_graph_enabled = compiled_graph and kern_graph
    speedup_vs_compiled = compiled_ms / kern_ms if kern_ms > 0 else 0.0

    ops_c_ms = None
    speedup_vs_ops_c = None
    if torch_ops_c_available:
        opsc_out = torch.empty_like(out)
        opsc_scales = torch.empty_like(scales)
        ops_c_ms, _ = bench(
            lambda o=opsc_out,
            i=input,
            s=opsc_scales,
            g=group_size,
            su=scale_ub,
            trp=is_scale_transposed: (torch_ops_c_baseline(o, i, s, g, su, trp))
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
        shape_duration,
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
    shape_duration: float,
) -> None:
    ops_c_str = f"{ops_c_ms:9.5f}" if ops_c_ms is not None else "      N/A"
    speedup_c_str = (
        f"{speedup_vs_ops_c:6.3f}x" if speedup_vs_ops_c is not None else "    N/A"
    )
    xpu_graph_str = "enabled" if xpu_graph_enabled else "disabled (fallback)"
    print(
        f"{name:12s} intermediate_size={intermediate_size:6d} "
        f"num_tokens={num_tokens:5d}  rel_err={err:.4f}  "
        f"compiled_ms={compiled_ms:9.5f}  ops_c_ms={ops_c_str}  "
        f"helion_ms={kern_ms:9.5f}  "
        f"speedup_vs_compiled={speedup_vs_compiled:6.3f}x  "
        f"speedup_vs_ops_c={speedup_c_str}  xpu_graph={xpu_graph_str}  "
        f"(autotune+bench wall: {shape_duration:.1f}s)"
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
    wall_total = sum(r[-1] for r in rows)
    print(
        f"total wall-clock time (autotune + benchmark, {len(rows)} shapes): "
        f"{wall_total:.1f}s"
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
                rec["shape_duration"],
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
        "instead of the default representative num_tokens=128. Only valid "
        "combined with --list-cases/--only-case/--report-from-sweep (see "
        "module docstring for why the plain in-process loop can't safely "
        "do this for this specific kernel).",
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

    if args.full and not args.only_case:
        raise SystemExit(
            "--full requires --only-case (or --list-cases/--report-from-sweep) "
            "for this kernel: looping multiple num_tokens values for the same "
            "specialization in one process can silently reuse a compiled "
            "kernel hardcoded for num_tokens=1 -- see the module docstring "
            "and RESULTS.md's 'real bug found, root-caused, NOT fixed' section."
        )

    assert current_platform.is_xpu(), "This benchmark targets Intel XPU"

    torch_ops_c_available = _probe_torch_ops_c()
    if not torch_ops_c_available:
        print(
            "torch.ops._C.silu_and_mul_per_block_quant is not available on "
            "this platform (CUDA-only); reporting torch.compile(native) "
            "only, no torch.ops._C column.\n"
        )

    kernel = build_kernel(args.autotune_effort, AUTOTUNE_BUDGET_SECONDS)

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
                shape_duration,
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
                shape_duration,
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
                shape_duration=shape_duration,
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
            shape_duration,
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
                shape_duration,
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
            shape_duration,
        )

    _print_summary_and_report(rows, detailed_rows)


if __name__ == "__main__":
    main()
