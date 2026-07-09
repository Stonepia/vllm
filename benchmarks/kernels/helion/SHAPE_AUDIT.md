# Shape/config audit: current benchmark coverage vs. each kernel's own grid

Written in response to: "why do the benchmarks only cover so few shapes?"
(compared to e.g. the blog's own `num_tokens=1..8192` sweep for
`silu_and_mul_per_block_quant`). This document is the fact-finding step
before deciding what to extend and how -- **for review before any script
changes are made.**

## Source of truth

Each kernel's own `vllm/kernels/helion/ops/<kernel>.py` defines a
`generate_inputs()` function, registered via `register_kernel(...,
input_generator=generate_inputs)`. This is the kernel author's own
declaration of "the shapes this kernel is autotuned and measured across" --
used to build the committed `intel_arc_pro_b70.json` config file. **This is
the ground truth used below, not my benchmark scripts' docstrings**, which
in one case (see Correction below) turned out to be a wrong assumption
rather than something actually checked against this source.

Cross-check: `silu_and_mul_per_block_quant`'s `generate_inputs()`
(`intermediate_size_list=[6144,12288,25600]` x
`num_tokens_list=[1,2,4,...,8192]`, 14 values) produces exactly the 42 cases
in the CUDA reference table you pasted (same case names, same order). I
take this as good evidence `generate_inputs()` is the right ground truth for
the other 8 kernels too, though I only have an independent reference table
to cross-check for this one kernel -- flagging that as the limit of my
certainty here.

## Correction #1: GEMM kernels don't share the non-GEMM `num_tokens` list

I initially told you the "full" grid was 630 combinations across all 9
kernels, assuming every kernel uses the same 14-value `num_tokens_list`
(`[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]`) as the 7 non-GEMM
kernels, including for the two GEMM kernels. **That assumption was wrong for
both GEMM kernels** -- checking their own `generate_inputs()` directly:

- `scaled_mm`'s own `m_size_list = [1, 2, 4, 8, 16, 32, 64, 128]` -- **8**
  values, capped at 128, never 1024/2048/4096/8192.
- `scaled_mm_blockwise`'s own `m_list = [4, 8, 16, 32, 64, 128, 256, 512,
  1024, 2048, 4096, 8192]` -- **12** values, starting at 4 (no 1 or 2).

This means my own `bench_scaled_mm.py`'s `NUM_TOKENS_FULL` (14 values,
including up to 8192) does not match this kernel's own declared shape grid,
and `NUM_TOKENS_QUICK = [16, 128, 1024]` includes M=1024, a value
`scaled_mm`'s own `generate_inputs()` never tests. Corrected full-parity
total (as of this correction): 534, not 630. **This total turned out to
also be wrong -- see Correction #2 below.**

## Correction #2: `scaled_mm`'s real grid is 168, not 96

You then pasted the actual CUDA reference table for `scaled_mm` itself
(14 `num_tokens` values `1..8192` x the same 12 `[K,N]` shapes as
`scaled_mm.py`'s own `B_SHAPES` -- verified by generating the 12 case-name
prefixes from `B_SHAPES` directly and matching them against your table's
first 12 rows, exact match, same order: `hidden_size`=K, `feature_size`=N).
That's **168** combinations, not the 96 I derived from `generate_inputs()`'s
`m_size_list = [1,2,4,8,16,32,64,128]`.

So `generate_inputs()` is *not* the same grid the blog's Table 2 benchmark
actually swept for this kernel -- it appears to be a separate, smaller grid
used only for autotuning the production config JSON (`pick_config`'s
nearest-match fallback at serving time presumably covers the gaps).
Correction #1 above, which relied on `generate_inputs()` as ground truth,
was itself wrong for `scaled_mm`. Good news: this means my very first
`bench_scaled_mm.py` script (`NUM_TOKENS_FULL = [1,...,8192]`, 14 values)
was actually right all along, and `NUM_TOKENS_QUICK = [16, 128, 1024]`
is a valid subset of the real grid (1024 *is* one of the 14 values) --
no script bug there after all.

I have **not** verified `scaled_mm_blockwise`'s 144 figure against an actual
reference table the way I just did for `scaled_mm` -- it's still only
backed by that kernel's own `generate_inputs()`, which just got shown to be
the wrong source for its sibling kernel. Flagging this explicitly rather
than asserting it with more confidence than I have: **if you have a
reference table for `scaled_mm_blockwise` too, it would be worth checking
the same way.**

One more thing worth noting from your pasted table: at this GPU (which one
-- H100 or B200? I don't know from the paste alone), `scaled_mm`'s speedup
is below 1.0x (loses to baseline) for most shapes at `num_tokens >= 16`,
recovering close to/above 1.0x only at `num_tokens` 1-8 for a few shapes.
That's a real, useful data point for RESULTS.md's cross-hardware comparison
once I know which GPU it's from -- it shows this kernel doesn't clearly
beat baseline on the reference hardware either, just less severely than on
this XPU.

## Confirmation: `scaled_mm_blockwise`'s 144 figure is correct

You then pasted the actual reference table for `scaled_mm_blockwise` too.
Same cross-check as `scaled_mm`: 12 `[K,N]` shapes matching `B_SHAPES`
exactly, and critically the `num_tokens` values present are `4, 8, 16, 32,
64, 128, 256, 512, 1024, 2048, 4096, 8192` -- **12 values, starting at 4,
with no 1 or 2** -- an exact match to this kernel's own `generate_inputs()`
`m_list`, unlike `scaled_mm` where `generate_inputs()` was wrong. So unlike
`scaled_mm`, `generate_inputs()` *is* the right source here: **144
confirmed**, no correction needed to that row.

Substantive finding worth carrying into RESULTS.md: this table shows a
clear crossover pattern -- the kernel wins (1.1x-2.0x) at small
`num_tokens` (4-64), crosses below 1.0x somewhere around `num_tokens`
128-256, and settles around 0.65-0.73x at the largest `num_tokens`
(1024-8192), fairly consistently across all 12 shapes. Directionally
similar to `scaled_mm`'s own "worse at larger num_tokens" pattern (and to
what we saw on XPU for both GEMM kernels), but much less severe here --
never worse than ~0.6x, vs. XPU's ~0.05-0.09x for `scaled_mm` at large M.
Still don't know which GPU this table (or the `scaled_mm` one) is from --
would help to attribute this correctly in RESULTS.md.

## Confirmation: `silu_and_mul_dynamic_per_token_quant`'s 42 figure is correct

You pasted its reference table too: `intermediate_size_6144_num_tokens_1`,
`intermediate_size_12288_num_tokens_1`, `intermediate_size_25600_num_tokens_1`
as the first 3 rows -- exact match to this kernel's own `generate_inputs()`
(`intermediate_size_list=[6144,12288,25600]` x the same 14-value
`num_tokens_list`). Second independent confirmation (after kernel 7) that
the shared 14-value list is right for the non-GEMM kernels.

One more gap this table surfaces, separate from shapes: it has
`baseline_peak(MB)` / `kernel_peak(MB)` / `mem_improve(x)` columns -- peak
memory, not just timing. None of our benchmark scripts measure or report
memory at all right now (always exactly 1.000x here, i.e. no memory
difference for this particular kernel, but that won't necessarily hold for
every kernel). Flagging as a separate, additional gap from the shape-count
one -- not fixing it now, just noting it exists so it doesn't get lost.

## What's now fully verified vs. still assumed

- **Verified against a real reference table**: kernel 6
  (`silu_and_mul_dynamic_per_token_quant`), kernel 7
  (`silu_and_mul_per_block_quant`), kernel 8 (`scaled_mm_blockwise`, 144),
  kernel 9 (`scaled_mm`, 168).
- **Still only inferred from `generate_inputs()`, not independently
  verified**: kernels 1-5 (`rms_norm_dynamic_per_token_quant`,
  `fused_qk_norm_rope`, `dynamic_per_token_scaled_fp8_quant`,
  `per_token_group_fp8_quant`, `rms_norm_per_block_quant`). Two independent
  confirmations of the same 14-value list (kernels 6 and 7) is reasonably
  good indirect evidence for these 5 too, since they all use that identical
  list -- but not the same as a direct check.

## Per-kernel table (corrected)

| # | Kernel | Shape axis (values) | `num_tokens`/M values | Full-parity combos | Currently measured | Gap |
| - | --- | --- | --- | --- | --- | --- |
| 1 | `rms_norm_dynamic_per_token_quant` | hidden_size: 2048, 4096, 5120 | 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192 (14) | 42 | 3 | 39 |
| 2 | `fused_qk_norm_rope` | (q_heads,kv_heads): (16,8),(32,8),(64,8) | same 14 | 42 | 3 | 39 |
| 3 | `dynamic_per_token_scaled_fp8_quant` | hidden_size: 2048, 4096, 5120 | same 14 | 42 | 3 | 39 |
| 4 | `per_token_group_fp8_quant` | hidden_size: 2048, 4096, 5120 (group_size=128 fixed) | same 14 | 42 | 3 | 39 |
| 5 | `rms_norm_per_block_quant` | hidden_size: 2048, 4096, 5120 (group_size=128 fixed) | same 14 | 42 | 3 (0 working -- autotune bug) | 39 |
| 6 | `silu_and_mul_dynamic_per_token_quant` | intermediate_size: 6144, 12288, 25600 | same 14 | 42 | 3 | 39 |
| 7 | `silu_and_mul_per_block_quant` | intermediate_size: 6144, 12288, 25600 (group_size=128 fixed) | same 14 | 42 | 3 | 39 |
| 8 | `scaled_mm_blockwise` | **12** `[K,N]` shapes (full grid: all 3 models x qkv/out/gate_up/down_proj) -- **verified against your pasted reference table** | 4,8,16,32,64,128,256,512,1024,2048,4096,8192 (12) | 144 | 3 (only qkv_proj shapes, only M=128) | 141 |
| 9 | `scaled_mm` | 12 `[K,N]` shapes (same full grid) -- **verified against your pasted reference table** | 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192 (14) | **168** | 34 (of the 36 our "quick" mode attempts; 2 OOM) | 134 |

**Total: 58 of 606 full-parity combinations measured (~10%).**

Notes:

- Kernels 1-7's `num_tokens_list` is identical across all 7 (verified by
  reading each file, not assumed from one and copied), and independently
  cross-checked against your pasted reference table for kernel 7.
- `scaled_mm_blockwise`'s current benchmark has two independent gaps: only
  3 of 12 `[K,N]` shapes (missing `out_proj`/`gate_up`/`down_proj` for every
  model entirely, not just a thinner M sweep), and a single fixed M=128.
- `scaled_mm`'s 34/36-measured figure comes from `NUM_TOKENS_QUICK = [16,
  128, 1024]` (3 values x 12 shapes = 36) -- now confirmed to be a valid
  subset sample of the real 14-value grid, not an out-of-grid value as I
  wrongly claimed in Correction #1.
- `rms_norm_per_block_quant` still has 0 working data points regardless of
  grid size (separate, already-documented autotuning bug).

## What I have not yet done

- Not yet decided/implemented anything about extending the sweeps to these
  full grids -- this document is the fact-finding step, for your review,
  before that decision.
- Not yet designed the OOM-recovery mechanism (separate follow-up, since
  `scaled_mm`/`scaled_mm_blockwise`'s largest shapes at their largest M
  values are the most likely to repeat the `down_proj` OOM crash already
  seen).
