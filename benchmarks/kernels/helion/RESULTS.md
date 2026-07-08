# Helion Kernel XPU Reproduction — Results

Reproduction of the kernel-level benchmarks from the PyTorch blog post
["Portable vLLM Model Inference Kernels in
Helion"](https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/)
(Table 2) and [issue #32962](https://github.com/vllm-project/vllm/issues/32962)
on Intel XPU, instead of the original NVIDIA H100/B200.

- **Branch**: `helion_xpu_kernels`
- **Hardware**: 2x Intel Arc Pro B70 (canonical platform key `intel_arc_pro_b70`)
- **Software**: `torch==2.14.0.dev20260629+xpu`, `helion==1.2.1.dev20+g875b3acbd`,
  `triton-xpu==3.7.2`, `vllm-xpu-kernels==0.1.10.1` (prebuilt custom ops present
  but unusable in this environment -- see Environment Notes)

## Summary table (mirrors blog Table 2)

Measured with `autotune_effort=quick` (`autotune_budget_seconds=45` where
supported by the script), each kernel run sequentially and standalone (not
concurrently), via `benchmarks/kernels/helion/run_all_7.sh`/`run_remaining.sh`.
`HELION_CACHE_DIR` was redirected to a project-local `.helion_cache/` so
autotuning artifacts persist across sessions. Full logs: `benchmark_logs/`.

| Kernel | Status | vs `torch.compile(native)` | vs `torch.ops._C` / CUTLASS | max rel_err |
| --- | --- | --- | --- | --- |
| `scaled_mm` | Enabled | N/A | 0.357x (1 shape only) | 0.0000 |
| `scaled_mm_blockwise` | Enabled | N/A | 0.537x | ≤0.006 |
| `dynamic_per_token_scaled_fp8_quant` | Enabled | 1.510x | N/A | 0.0000 |
| `rms_norm_dynamic_per_token_quant` | Enabled | 11.686x | N/A | 0.0000 |
| `per_token_group_fp8_quant` | Enabled | 1.192x | N/A | 0.0643 |
| `rms_norm_per_block_quant` | Enabled (production path works); **benchmark's live autotuning search fails, see below** | N/A | N/A | N/A |
| `silu_and_mul_dynamic_per_token_quant` | Enabled | 1.828x | N/A | 0.0714 |
| `silu_and_mul_per_block_quant` | **Disabled** for production dispatch; standalone benchmark works (bypasses the disabled registry entry) | 2.571x | N/A | 0.0004 |
| `fused_qk_norm_rope` | Enabled (bug fixed) | 8.131x | N/A | 0.0028 |

`torch.ops._C` is N/A across the board: confirmed unavailable (CUDA-only,
not registered) for all 7 non-GEMM kernels in this environment. `scaled_mm`/
`scaled_mm_blockwise` have no `torch.compile` column, matching the blog's own
Table 2 (CUTLASS-only comparison for GEMM kernels) -- their `torch._scaled_mm`
comparison is unaffected by anything below.

Per-shape numbers, autotuning wall time, and full stdout for every kernel are
in `benchmark_logs/bench_<kernel>.log`.

**Directional comparison with the blog's CUDA results**: on H100/B200,
`scaled_mm`/`scaled_mm_blockwise` were competitive with or beat CUTLASS
(0.74-1.08x), while non-GEMM kernels won more modestly (1.13-2.3x vs
`torch.ops._C`). On this XPU, the two GEMM kernels currently *lose* to the
native op (0.36-0.54x), while the non-GEMM kernels show much larger apparent
wins (1.2-11.7x vs `torch.compile`) -- not directly comparable
magnitude-for-magnitude to the blog's `torch.ops._C` column, since XPU has no
`torch.ops._C` implementation available to compare against at all here.

## Kernel status details

### 7 kernels ported + enabled cleanly

`scaled_mm`, `scaled_mm_blockwise`, `dynamic_per_token_scaled_fp8_quant`,
`rms_norm_dynamic_per_token_quant`, `per_token_group_fp8_quant`,
`rms_norm_per_block_quant`, `silu_and_mul_dynamic_per_token_quant`. Each has:
correctness tests passing on XPU (independently re-verified for the ones
that needed non-trivial config fixes, not just trusted from the porting
pass), a dummy `CaseKey.default()` XPU config (generated via Helion's own
`config_spec.default_config()`, never hand-copied from the NVIDIA configs),
and a `benchmarks/kernels/helion/bench_<kernel>.py` script. (`rms_norm_per_block_quant`'s
production config/correctness are fine, but its benchmark script hits a
separate, unrelated autotuning-search issue -- see its own subsection below.)

### `fused_qk_norm_rope` — real bug found AND fixed

Root cause: the kernel wrote its RMSNorm result into the packed `qkv`
buffer, then immediately re-read a subset of that same buffer for the RoPE
step, instead of reusing the value already in registers. This read-after-
write pattern produced non-deterministic output on Intel XPU (confirmed:
repeated calls with bit-identical inputs gave different results, ~4% of
elements, up to 32/102 pytest cases failing). **Fixed** by reading from the
already-computed value via `torch.gather` instead of re-reading `qkv` (see
`vllm/kernels/helion/ops/fused_qk_norm_rope.py`, commit `364de928a`). All
102 tests pass, confirmed clean across 3 repeated full-suite runs, plus a
standalone determinism stress test at 6 shapes. This is a genuine kernel
bug independent of XPU vs. CUDA semantics (removes an unnecessary,
hazardous global-memory round-trip) and is a good candidate for upstream
contribution back to PR #44010, independent of XPU enablement.

### `silu_and_mul_per_block_quant` — real bug found, root-caused, NOT fixed

Kept disabled (no XPU config committed). Root cause (see the detailed
docstring in `tests/kernels/helion/test_silu_and_mul_per_block_quant.py`
and commit `420e08043` for the full evidence): the kernel tiles `num_tokens`
with a fixed `hl.tile(..., block_size=1)`. When Helion's first-ever compile
for a given specialized (hidden_size/group_size/quant_dtype/
is_scale_transposed) combination happens to see `num_tokens=1` -- which
coincides with that fixed `block_size=1` -- Helion fully hardcodes
`num_tokens` as a compile-time constant (`_BLOCK_SIZE_0 = tl.constexpr(1)`,
grid size `1`, `num_tokens` dropped entirely from the compiled kernel's
signature) instead of keeping it symbolic as intended. This contradicts
Helion's own documented `specialize_zero_one=False` guarantee for dynamic
kernels. Subsequent calls with a different `num_tokens` then reuse this
permanently-1-sized compiled kernel and silently under-compute. Confirmed
via a side-by-side `to_triton_code()` diff between a `num_tokens=1` bind
and a `num_tokens=7` bind -- a minimal, concrete repro for an upstream
Helion bug report. A warm-up-call workaround was tested and found
insufficient in general (it only protects the one specialized combination
it warms up); a full fix needs either an upstream Helion fix or a
comprehensive per-specialization warm-up mechanism in vLLM, both out of
scope here.

### `rms_norm_per_block_quant` — benchmark's autotuning search hits an XPU Triton gap, and fails unsafely

The kernel itself is fine for production use: calling it directly with its
committed dummy default config (`vllm/kernels/helion/configs/rms_norm_per_block_quant/intel_arc_pro_b70.json`,
`block_sizes=[32, 32]`) works correctly (verified standalone) and its
correctness test suite passes in full. The problem is specific to
`bench_rms_norm_per_block_quant.py`'s live re-autotuning (via
`create_helion_decorated_kernel`, used so the benchmark measures a genuinely
searched config rather than always reusing the one dummy default -- same as
the other 6 kernels' scripts).

Root cause: Helion's `LFBOTreeSearch` seeds its initial population from the
kernel's existing config ("`Starting with seed/default configs`" /
"`Initial population: 1 total`" in the logs), then explores nearby variants.
One such variant it tries for this kernel/shape (`block_sizes=[32, 16],
num_warps=4` -- close to, but not identical to, the working
`[32, 32]` default) compiles to Triton code that attempts to add two
`Float8_e4m3fn` tensors directly, which XPU's backend does not implement:

```text
NotImplementedError: "add_xpu" not implemented for 'Float8_e4m3fn'
```

This alone would just be a bad candidate to reject during search. The actual
bug: encountering this error while autotuning **hangs the whole process
indefinitely** instead of raising cleanly, when `autotune_ignore_errors`/
`HELION_AUTOTUNE_IGNORE_ERRORS` is unset (Helion's default). Confirmed
reproducible twice, independently: the original run sat at 0% CPU on a
`futex_do_wait` for **~12 hours** before being caught and killed manually; a
clean retry (fresh process, no stale state) hung again within ~3 minutes,
CPU time frozen, and was killed after confirming the pattern. Both times the
`TritonError` had already been formatted and logged (visible in
`benchmark_logs/bench_rms_norm_per_block_quant_noignore.log`) before the
hang, so the failure is classified correctly (not caught by Helion's
"unrecoverable error" patterns, which would at least `raise` cleanly) --
something in the unwind/cleanup path after that specific error hangs on XPU.
This looks like a genuine Helion/XPU-backend issue, not something specific
to vLLM's kernel code, and is a good candidate for an upstream bug report.

Setting `HELION_AUTOTUNE_IGNORE_ERRORS=1` avoids the hang (skips the bad
candidate instead of raising) but doesn't produce a benchmark number here:
with only one seed candidate in the initial population and no working
mutation found within the `quick`/45s budget, the search reports
`helion.exc.NoConfigFound` and exits (cleanly, in ~12s) rather than
completing. Re-running with a larger autotune budget/effort, or fixing the
underlying XPU Triton gap, are both out of scope for this benchmarking pass.
`benchmarks/kernels/helion/run_remaining.sh` sets this flag (plus a
`timeout` safety net) for exactly this reason.

## Environment notes (apply to all kernels)

- **`vllm_xpu_kernels` ABI mismatch**: the installed wheel needs
  `libsycl.so.8`; this environment only has `libsycl.so.9`. Confirmed via a
  real (not soname-only) ABI break: a naive symlink workaround fails with
  `undefined symbol: sycl::exception constructor`. `vllm/platforms/xpu.py`
  was patched to make this import optional (commit `7db60a5c3`) so XPU
  platform detection still works; any code path needing the prebuilt custom
  ops (e.g. `torch.ops._C.rms_norm_dynamic_per_token_quant` and friends, which
  the disassembled `_C.abi3.so` confirms ARE implemented natively for XPU,
  just inaccessible here) fails explicitly instead.
- Per project decision, CUDA-only baselines (CUTLASS, `torch.ops._C.*`) were
  replaced with the best available portable/native alternative for
  benchmarking on XPU: the blog's own three-way methodology (Helion vs.
  `torch.compile(native_impl)` vs. `torch.ops._C`, reporting N/A rather than
  omitting a column when an op is unavailable) is followed exactly, with
  `native_impl` ported verbatim from the reference commit rather than
  reinvented -- see the Summary table callout above.
- Each kernel's own `baseline()` function (in `vllm/kernels/helion/ops/*.py`)
  was left byte-identical to its upstream source; portable substitutes were
  only introduced at the point of use (test files, benchmark scripts).

## Autotuning economics (important)

A single shape at `autotune_effort="quick"` (Helion's default preset) took
**~6.5 minutes** for `scaled_mm` (153 configs searched). A `autotune_budget_seconds`
cap (a genuine Helion `Settings` field, separate from the `none`/`quick`/`full`
effort presets) brings this down to **~45-60s/shape** with still-real tuning,
at some cost to how thoroughly the search converges. Even so, several
kernels' "final verification" phase (a post-search re-benchmark step,
*not* bounded by `autotune_budget_seconds`) added anywhere from 0 to
10+ minutes unpredictably. See `AUTOTUNING_HANDOFF.md` for how to run a
proper full sweep later.

The 7-non-GEMM-kernel sequential re-run (`run_all_7.sh`/`run_remaining.sh`,
results now in the Summary table above) took **~2h7m of actual kernel time**
(sum of all 7 scripts' wall time) across two invocations, plus a lost
**~12h** to the `rms_norm_per_block_quant` hang described above before it
was caught -- per-kernel wall time ranged from 12s (that same kernel's
clean-failure retry) to ~34 min (`dynamic_per_token_scaled_fp8_quant`, its
first invocation with a cold Triton/Helion cache).

## What's not done (flagging explicitly)

- `rms_norm_per_block_quant`'s benchmark: no working speedup number (see its
  subsection above) -- would need either a larger autotune budget/effort
  (untested whether that avoids the specific bad candidate) or an upstream
  Helion fix for the XPU hang-on-error behavior.
- Only 1 shape benchmarked for `scaled_mm` (vs. 3 for the other kernels) --
  discovered the 6.5-min/shape cost before settling on the budget-capped
  approach; not revisited afterward.
- Real per-shape autotuned configs (vs. the single dummy default per
  kernel) -- see `AUTOTUNING_HANDOFF.md`.
- Clean, sequential (non-contended) timing re-measurement for `scaled_mm`/
  `scaled_mm_blockwise` specifically -- those two numbers in the Summary
  table still predate the sequential re-run (gathered under parallel-subagent
  GPU contention); the 7 non-GEMM kernels' numbers above are from the clean
  sequential re-run.
- End-to-end serving benchmarks (blog Figs. 1-3, Table 3) -- out of scope
  per an earlier explicit scope decision; the Helion kernel registry isn't
  wired into vLLM's real fusion passes/forward pass anywhere (upstream or
  here), so this would be a separate, larger integration effort.
- `vllm_xpu_kernels`'s ABI mismatch itself isn't fixed (would need a
  rebuild against this environment's oneAPI/libsycl, or a newer released
  wheel) -- worked around per project decision, not fixed.
