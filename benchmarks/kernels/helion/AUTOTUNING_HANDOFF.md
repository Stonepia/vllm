# Handoff: Running Full Helion Autotuning on XPU

This is a practical how-to for a **future session** that wants to go beyond
the single dummy `CaseKey.default()` config each kernel currently has, and
produce real, per-shape tuned `intel_arc_pro_b70.json` configs (matching the
fidelity of the existing `nvidia_h100.json`/`nvidia_b200.json` files), and/or
get clean (non-preliminary) kernel-level benchmark numbers. See
`RESULTS.md` in this same directory for what's already been done and why
this wasn't done in the original session.

## TL;DR time budget

- A single shape at `autotune_effort="quick"` (Helion's built-in preset, no
  budget cap) took **~6.5 minutes** for `scaled_mm` (153 configs searched).
- Capping with `autotune_budget_seconds=45` brought this to **~45-60s/shape**
  with still-real tuning -- but several kernels' "final verification" phase
  (a post-search re-benchmark, NOT bounded by `autotune_budget_seconds`) can
  add anywhere from 0 to 10+ minutes unpredictably on top of that.
- The blog's own "Caveats" section says a full-effort sweep of all 168
  `scaled_mm` shapes "can take an entire day" on H100. Expect XPU to be
  similar or slower (younger/less-optimized Triton-XPU backend).
- **Budget realistically**: a few hours to a day, per kernel, for a proper
  full sweep across the blog's complete shape grid (12 `[K,N]` shapes x 14
  `num_tokens` values for GEMM kernels; hidden_size x num_tokens for the
  rest). Plan to run this **unattended/overnight**, not interactively.

## Two ways to autotune

### Option A: the official script (recommended for "real" configs)

`scripts/autotune_helion_kernels.py` is vLLM's production autotuning
entrypoint -- it's what actually produced `nvidia_h100.json`/
`nvidia_b200.json`. **It currently hard-gates on CUDA and needs one small
fix before it'll run on XPU:**

```python
# scripts/autotune_helion_kernels.py, check_requirements(), ~line 77
def check_requirements() -> bool:
    if not torch.cuda.is_available():          # <-- change this
        logger.error("CUDA is not available. Helion autotuning requires GPU.")
        return False
    ...
```

Change to something like:

```python
def check_requirements() -> bool:
    from vllm.platforms import current_platform
    if not (torch.cuda.is_available() or current_platform.is_xpu()):
        logger.error("No supported accelerator (CUDA/XPU) available.")
        return False
    ...
```

That's the only hard gate; `get_canonical_gpu_name()` (used right after)
already resolves to `intel_arc_pro_b70` correctly (the `_GPU_NAME_ALIASES`
entry for this was added in this effort -- see `vllm/kernels/helion/utils.py`).
Every kernel's own `generate_inputs()` in `vllm/kernels/helion/ops/*.py`
already uses `current_platform.device_type` (fixed in this effort for all 9
kernels), so no further per-kernel device-string fixes should be needed.

Usage once patched:

```bash
source ~/torch-xpu-env/.venv/bin/activate   # or wherever your vLLM+helion env lives

# List available kernels first:
python scripts/autotune_helion_kernels.py --list

# Autotune ONE kernel, quick effort (fastest way to get a real per-shape
# config file rather than the single dummy default currently committed):
python scripts/autotune_helion_kernels.py --kernels scaled_mm --autotune-effort quick

# Full effort (matches the blog's methodology -- LFBOTreeSearch,
# initial_population=FROM_RANDOM, copies=5, max_generations=20). SLOW.
python scripts/autotune_helion_kernels.py --kernels scaled_mm --autotune-effort full

# All 9 kernels (drop --kernels). Run with nohup/screen/tmux, this will
# take a long time:
nohup python scripts/autotune_helion_kernels.py --autotune-effort quick \
    > autotune_all.log 2>&1 &
```

Notes on this script's behavior (as of this branch):

- It iterates every `CaseKey` produced by each kernel's own `generate_inputs()`
  (the SAME full shape grid used to build the NVIDIA configs -- e.g.
  `scaled_mm`'s is 8 `num_tokens` values x 12 `[K,N]` shapes = 96 combos, not
  the full 168 the blog mentions for its exhaustive H100 sweep; check each
  kernel's `generate_inputs()` if you want to expand/shrink the grid).
- It SKIPS keys that already have a saved config for the target platform
  unless you pass `--force` -- so it will NOT touch/overwrite the existing
  single dummy `CaseKey.default()` entry already committed for 8 of the 9
  kernels (that entry's key is different from any real per-shape key, so it
  coexists fine either way).
- It saves incrementally (`config_manager.save_configs(...)` after each
  successful key), so you can safely Ctrl-C and resume later; it'll pick up
  from where it left off given the skip-existing-keys behavior above.
- It does NOT expose `--autotune-budget-seconds` today. If you want the
  bounded-budget behavior (see below), either add that CLI passthrough
  yourself (it just needs to reach `HelionKernelWrapper.run_autotune()`'s
  `extra_kwargs`), or use Option B instead for budget-capped runs.

### Option B: the benchmark scripts (quicker iteration, ad-hoc)

Each `benchmarks/kernels/helion/bench_<kernel>.py` script (all 9 exist,
including the 2 kept-disabled kernels) already supports:

```bash
python benchmarks/kernels/helion/bench_scaled_mm.py \
    --autotune-effort quick --autotune-budget-seconds 45   # current defaults

python benchmarks/kernels/helion/bench_scaled_mm.py \
    --full --autotune-effort full --autotune-budget-seconds 300
```

`--full` switches from the 3-shape representative subset (1.7B/8B/32B) to
the complete `NUM_TOKENS_FULL` grid x all `[K,N]`/hidden_size shapes -- this
is what you want for real full-grid numbers. These scripts do their OWN
live autotuning per shape (via `create_helion_decorated_kernel(...,
extra_kwargs={"autotune_effort":..., "autotune_budget_seconds":...})`) and
do NOT save configs anywhere -- they're for benchmark numbers, not config
generation. Use Option A if you want the actual `intel_arc_pro_b70.json`
files updated with real per-shape entries.

## Recommended plan for a full, clean run

1. Apply the one-line fix to `scripts/autotune_helion_kernels.py` above.
2. Make sure NOTHING else is using the GPUs (check `nvidia-smi`-equivalent
   for XPU, e.g. `xpu-smi` if installed, or just `ps aux | grep -i bench`/
   `pytest`) -- autotuning's own internal benchmarking is timing-sensitive,
   contention will produce misleading "best" configs.
3. Run Option A per kernel, one at a time, in the background:
   ```bash
   for k in scaled_mm scaled_mm_blockwise dynamic_per_token_scaled_fp8_quant \
            rms_norm_dynamic_per_token_quant per_token_group_fp8_quant \
            rms_norm_per_block_quant silu_and_mul_dynamic_per_token_quant \
            fused_qk_norm_rope; do
     echo "=== $k ===" >> autotune_all.log
     python scripts/autotune_helion_kernels.py --kernels "$k" \
       --autotune-effort quick >> autotune_all.log 2>&1
   done
   ```
   (`silu_and_mul_per_block_quant` is deliberately excluded -- see below.)
4. Once done, re-run each `bench_<kernel>.py --full` (WITHOUT
   `--autotune-effort`/`--autotune-budget-seconds` overrides doesn't apply
   here since those scripts do their own live autotuning independent of the
   saved config; if you want the benchmark to exercise the NEWLY saved
   per-shape configs via the real production dispatch path instead of a
   fresh live autotune, you'd need to adapt the benchmark script to call
   through `get_registered_kernels()[name]` instead of building a fresh
   kernel via `create_helion_decorated_kernel` -- not currently how they're
   written, since they were built to work even when the kernel is
   `_disabled` due to missing configs).
5. Update `RESULTS.md`'s summary table with the new, clean numbers.

## `silu_and_mul_per_block_quant` — do not naively autotune this one yet

This kernel has an **unresolved Helion bug** (see `RESULTS.md` and
`tests/kernels/helion/test_silu_and_mul_per_block_quant.py`'s module
docstring for the full, evidenced root cause): if the very first compile
for a given specialized shape combination happens to see `num_tokens=1`,
Helion incorrectly hardcodes `num_tokens` as a compile-time constant, and
every subsequent DIFFERENT `num_tokens` value silently computes garbage.

If you run the official autotune script against this kernel, it WILL
process `num_tokens=1` at some point in its shape grid, and depending on
processing order, could either (a) generate a subtly-wrong config that
happens to look fine for whichever shape got tuned last, or (b) pollute
Helion's process-local compile cache such that OTHER shapes tuned
afterward in the same process are affected. Before autotuning this kernel:

- Either fix the underlying Helion bug (file/check for an upstream fix
  first -- search Helion's issue tracker for `specialize_zero_one`,
  `hl.tile` fixed `block_size` + first-call specialization interactions),
- Or patch `vllm/kernels/helion/ops/silu_and_mul_per_block_quant.py`'s
  `generate_inputs()` to exclude `num_tokens=1` from the grid AND run each
  DISTINCT `(hidden_size, group_size)` combination in its OWN fresh Python
  process (`autotune_kernel()` in the script processes all keys for a
  kernel within one process, which is exactly the condition that triggers
  this bug across different `num_tokens` values sharing a compile cache).

## Sanity-check after any full sweep

Re-run the full pytest suite per kernel (`pytest
tests/kernels/helion/test_<kernel>.py`) after generating new configs --
correctness could regress if a real per-shape config exposes an edge case
the single dummy default happened not to hit (this exact thing happened for
`rms_norm_per_block_quant` in this effort: the FIRST dummy config, bound
with only one of two optional-tensor features active, hit a genuine Helion
compile error for the other feature combination -- see that kernel's commit
message for the full story). If you hit something similar, the fix pattern
is: bind the config-generation example with the "maximal" combination of
optional/boolean kernel arguments active (residual, scale_ub, etc.) so the
resulting config's internal indexing/load-store bookkeeping covers every
code path, not just the one exercised by whichever combination you happened
to bind with.
