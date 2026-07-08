# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level benchmark for the Helion ``fused_qk_norm_rope`` kernel on
Intel XPU.

Three-way comparison, matching the blog's own methodology
(https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/):

  a) the Helion kernel
  b) ``torch.compile(native_impl)`` -- unlike the other kernels, this
     kernel's own ``baseline()`` (in
     vllm/kernels/helion/ops/fused_qk_norm_rope.py) is ALREADY fully
     portable pure PyTorch (``vllm.ir.ops.rms_norm`` +
     ``RotaryEmbedding.forward_static``, no CUDA-only op), matching what the
     reference end-to-end-benchmark dev branch's custom ``Layer`` class
     computes (xiaohongchen1991/vllm@d9b0566a54737c1b0a8e3714ef7a3ee40a4442c2,
     vllm/kernels/helion/ops/fused_qk_norm_rope.py) -- so ``baseline()`` is
     wrapped in ``torch.compile(..., fullgraph=True, dynamic=False,
     backend="inductor", options={...})`` directly (same options as the
     reference commit), with no separate native_impl needed.
  c) ``torch.ops._C.fused_qk_norm_rope`` -- CUDA-only; probed once at import
     time, reports N/A here (confirmed unavailable in this environment).

Shapes: one per Qwen3 model (1.7B/8B/32B). ``num_q_heads``/``num_kv_heads``
reproduce this kernel's own ``generate_inputs()`` ``num_heads_pair`` list
``[(16, 8), (32, 8), (64, 8)]``, which is independently cross-checked
against the Qwen3 qkv_proj ``[K, N]`` dims already used in
``bench_scaled_mm.py``: ``N = (num_q_heads + 2 * num_kv_heads) * head_dim``
with ``head_dim=128`` gives N=4096/6144/10240 for Qwen3-1.7B/8B/32B,
matching exactly. ``num_tokens=128`` is used for all three shapes (a
representative single value, not a full grid).

** FIXED data race on Intel XPU (see fused_qk_norm_rope.py) **: this
kernel used to exhibit a data race on Intel XPU (repeated invocations
with bit-identical inputs produced different, nondeterministic
outputs). Root cause: the kernel stored the RMSNorm result into
``qkv`` and then immediately reloaded an overlapping region of that
same buffer for the RoPE step, instead of reusing the value already
computed in registers. Fixed by reading from the already-computed
``x_blk`` (via ``torch.gather``) instead of re-reading ``qkv``.
Verified deterministic (0 mismatched elements across repeated calls)
at every previously-failing shape, and all 102 pytest cases pass
across repeated full-suite runs. ``rel_err`` reported below is now a
reliable correctness signal.

Usage:
    python benchmarks/kernels/helion/bench_fused_qk_norm_rope.py
    python benchmarks/kernels/helion/bench_fused_qk_norm_rope.py \
        --autotune-effort full --autotune-budget-seconds 120
"""

from __future__ import annotations

import argparse
import math

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.kernels.helion.ops.fused_qk_norm_rope import (
    _compute_cos_sin_cache,
    baseline,
)
from vllm.kernels.helion.ops.fused_qk_norm_rope import (
    fused_qk_norm_rope as _wrapper,
)
from vllm.kernels.helion.register import create_helion_decorated_kernel
from vllm.platforms import current_platform
from vllm.triton_utils import triton

BF16 = torch.bfloat16
DEVICE = current_platform.device_type
HEAD_DIM = 128
NUM_TOKENS = 128
EPS = 1e-6
IS_NEOX = True

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
    _COMPILED_BASELINE = torch.compile(
        baseline,
        fullgraph=True,
        dynamic=False,
        backend="inductor",
        options=_INDUCTOR_OPTIONS,
    )

# (num_q_heads, num_kv_heads) per Qwen3 model. These match this kernel's own
# generate_inputs() num_heads_pair list exactly, and independently reproduce
# the Qwen3 qkv_proj [K, N] dims from bench_scaled_mm.py's B_SHAPES:
#   N = (num_q_heads + 2 * num_kv_heads) * head_dim
#   Qwen3-1.7B: N=4096 -> 4096/128=32 = 16 + 2*8  -> (16, 8)
#   Qwen3-8B:   N=6144 -> 6144/128=48 = 32 + 2*8  -> (32, 8)
#   Qwen3-32B:  N=10240 -> 10240/128=80 = 64 + 2*8 -> (64, 8)
MODEL_HEADS: dict[str, tuple[int, int]] = {
    "Qwen3-1.7B": (16, 8),
    "Qwen3-8B": (32, 8),
    "Qwen3-32B": (64, 8),
}


def torch_ops_c_baseline(qkv: torch.Tensor, *rest) -> None:
    """(c): the CUDA-only fused custom op, if it existed -- this kernel's
    baseline() doesn't call it (it's already portable), but the blog's
    Table 2 has a separate "vs torch.ops._C" column for this kernel, so we
    probe for it here too rather than assuming it doesn't exist."""
    torch.ops._C.fused_qk_norm_rope(qkv, *rest)


def _probe_torch_ops_c() -> bool:
    if not hasattr(torch.ops._C, "fused_qk_norm_rope"):
        return False
    try:
        qkv, rest = make_inputs(16, 8)
        torch_ops_c_baseline(qkv, *rest)
        torch.xpu.synchronize()
        return True
    except Exception:
        return False


def make_inputs(num_q_heads: int, num_kv_heads: int) -> tuple[torch.Tensor, tuple]:
    total_dim = (num_q_heads + 2 * num_kv_heads) * HEAD_DIM
    qkv = torch.empty(NUM_TOKENS, total_dim, dtype=BF16, device=DEVICE).uniform_(
        -0.1, 0.1
    )
    positions = torch.arange(NUM_TOKENS, dtype=torch.long, device=DEVICE)
    q_weight = torch.empty(HEAD_DIM, dtype=BF16, device=DEVICE).uniform_(0.8, 1.2)
    k_weight = torch.empty(HEAD_DIM, dtype=BF16, device=DEVICE).uniform_(0.8, 1.2)
    cos_sin_cache = _compute_cos_sin_cache(40960, HEAD_DIM, device=DEVICE).to(BF16)
    rest = (
        num_q_heads,
        num_kv_heads,
        num_kv_heads,
        HEAD_DIM,
        EPS,
        q_weight,
        k_weight,
        cos_sin_cache,
        IS_NEOX,
        positions,
    )
    return qkv, rest


def bench(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    o, r = out.float(), ref.float()
    return ((o - r).abs().max() / (r.abs().max() + 1e-6)).item()


def build_kernel(autotune_effort: str, autotune_budget_seconds: int):
    """Live per-shape autotuner.

    baseline() is fully portable pure PyTorch (see module docstring), so
    unlike scaled_mm's benchmark, no autotune_baseline_fn override is
    needed -- the kernel's own helion_settings (with its unmodified
    baseline()-based autotune_baseline_fn) is reused as-is; only
    autotune_effort/autotune_budget_seconds are overridden via extra_kwargs.
    """
    return create_helion_decorated_kernel(
        _wrapper.raw_kernel_func,
        _wrapper.helion_settings,
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
            "torch.ops._C.fused_qk_norm_rope is not available on this "
            "platform (CUDA-only); reporting torch.compile(native) only, "
            "no torch.ops._C column.\n"
        )

    kernel = build_kernel(args.autotune_effort, args.autotune_budget_seconds)

    rows = []
    for name, (num_q_heads, num_kv_heads) in MODEL_HEADS.items():
        qkv, rest = make_inputs(num_q_heads, num_kv_heads)

        compiled_qkv = qkv.clone()
        _COMPILED_BASELINE(compiled_qkv, *rest)

        # First call also triggers autotuning/compilation for this
        # specialized shape.
        test_qkv = qkv.clone()
        kernel(test_qkv, *rest)
        torch.xpu.synchronize()
        err = rel_err(test_qkv, compiled_qkv)

        compiled_buf = qkv.clone()
        compiled_ms = bench(
            lambda buf=compiled_buf, r=rest: _COMPILED_BASELINE(buf, *r)
        )

        kern_buf = qkv.clone()
        kern_ms = bench(lambda buf=kern_buf, r=rest: kernel(buf, *r))
        speedup_vs_compiled = compiled_ms / kern_ms if kern_ms > 0 else 0.0

        ops_c_ms = None
        speedup_vs_ops_c = None
        if torch_ops_c_available:
            opsc_buf = qkv.clone()
            ops_c_ms = bench(lambda buf=opsc_buf, r=rest: torch_ops_c_baseline(buf, *r))
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
            f"{name:12s} num_q_heads={num_q_heads:3d} num_kv_heads={num_kv_heads:2d} "
            f"num_tokens={NUM_TOKENS:5d}  rel_err={err:.4f}  "
            f"compiled_ms={compiled_ms:9.5f}  ops_c_ms={ops_c_str}  "
            f"helion_ms={kern_ms:9.5f}  "
            f"speedup_vs_compiled={speedup_vs_compiled:6.3f}x  "
            f"speedup_vs_ops_c={speedup_c_str}"
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


if __name__ == "__main__":
    main()
