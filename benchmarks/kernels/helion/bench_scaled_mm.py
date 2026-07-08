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
"""

from __future__ import annotations

import argparse
import dataclasses
import math

import torch

from vllm.kernels.helion.ops.scaled_mm import scaled_mm as _wrapper
from vllm.kernels.helion.register import create_helion_decorated_kernel
from vllm.platforms import current_platform
from vllm.triton_utils import triton

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


def bench(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


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
    args = parser.parse_args()

    assert current_platform.is_xpu(), "This benchmark targets Intel XPU"

    num_tokens_list = NUM_TOKENS_FULL if args.full else NUM_TOKENS_QUICK
    kernel = build_kernel(args.autotune_effort)

    rows: list[tuple[str, int, int, int, float, float, float, float]] = []
    for name, (K, N) in B_SHAPES.items():
        for M in num_tokens_list:
            a, b, scale_a, scale_b, bias = make_inputs(M, K, N)
            call_args = (a, b, scale_a, scale_b, BF16, bias)

            out = kernel(*call_args)
            ref = scaled_mm_native(*call_args)
            torch.xpu.synchronize()
            err = rel_err(out, ref)

            base_ms = bench(lambda ca=call_args: scaled_mm_native(*ca))
            kern_ms = bench(lambda ca=call_args: kernel(*ca))
            speedup = base_ms / kern_ms if kern_ms > 0 else 0.0
            rows.append((name, M, K, N, err, base_ms, kern_ms, speedup))
            print(
                f"{name:22s} M={M:5d} K={K:6d} N={N:6d}  rel_err={err:.4f}  "
                f"native_ms={base_ms:9.5f}  helion_ms={kern_ms:9.5f}  "
                f"speedup={speedup:6.3f}x"
            )

    speedups = [r[-1] for r in rows if r[-1] > 0]
    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(
            f"\ngeomean speedup vs torch._scaled_mm (XPU native) over "
            f"{len(rows)} shapes: {geo:.3f}x"
        )
    max_err = max((r[4] for r in rows), default=0.0)
    print(f"max rel_err across all shapes: {max_err:.4f}")


if __name__ == "__main__":
    main()
