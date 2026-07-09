#!/usr/bin/env zsh
# Crash-safe full-grid sweep: one subprocess per (kernel, case), so a
# per-shape XPU fault (OOM/DEVICE_LOST -- see RESULTS.md's "What's not
# done") only loses that one case, not the rest of the kernel's sweep.
#
# Why per-case, not per-kernel-with-relaunch: a hard crash (segfault,
# driver abort) or a timeout-killed hang leaves no Python code running to
# record its own failure -- confirmed empirically that even a same-process
# fallback path failed right after a DEVICE_LOST fault. Recording "this
# case failed" has to come from something OUTSIDE the crashed process. A
# bash loop over a list enumerated up front does this for free: it
# observes each case's subprocess exit code from the parent shell (valid
# no matter how the child died) and always proceeds to the next list
# element regardless of the previous one's outcome. No per-process
# "resume" bookkeeping is needed because there's nothing to resume --
# each case is independent, and the full case list is known before any
# of them run.
#
# See SHAPE_AUDIT.md for why the full grid differs per kernel (each
# kernel's own generate_inputs()/reference table, not a shared assumption).
set -uo pipefail

cd /home/tongsu/vllm
source ~/torch-xpu-env/.venv/bin/activate

export HELION_CACHE_DIR=/home/tongsu/vllm/.helion_cache
export HELION_AUTOTUNE_IGNORE_ERRORS=1
LOGDIR=/home/tongsu/vllm/benchmark_logs
mkdir -p "$LOGDIR"

# Per-case, not per-kernel: a hang now only costs one case (this timeout),
# not a whole kernel's sweep, so this can be much shorter than the old
# 12h PER_KERNEL_TIMEOUT while still comfortably covering the slowest
# legitimate from-scratch autotune seen so far (~6 min, Qwen3-32B/down_proj
# M=16). Generous headroom for a never-tried larger shape needing more.
PER_CASE_TIMEOUT="${PER_CASE_TIMEOUT:-7200}" # 2h, overridable for testing

DRIVER_LOG="$LOGDIR/driver_fullsweep.log"

# run_kernel_full_sweep <kernel> [extra bench_<kernel>.py args, e.g. --full]
run_kernel_full_sweep() {
  local kernel="$1"
  shift
  local extra_args=("$@")
  local sweep_file="$LOGDIR/sweep_${kernel}.jsonl"
  local case_log="$LOGDIR/bench_${kernel}_fullsweep.log"
  local script="benchmarks/kernels/helion/bench_${kernel}.py"

  : > "$case_log"
  echo "=== $kernel: listing cases at $(date) ===" | tee -a "$DRIVER_LOG"
  local cases
  # vLLM's own import-time logging goes to stdout in this environment, not
  # just stderr (confirmed directly -- corrupted an earlier run's case
  # count and fed a warning message in as a bogus --only-case argument).
  # bench_utils.py's print_case_name() tags real case lines with a unique
  # "CASE:" prefix specifically so this grep can't be fooled by whatever
  # other noise ends up on stdout.
  cases=$(python "$script" "${extra_args[@]}" --list-cases 2>/dev/null \
    | sed -n 's/^CASE://p') || {
    echo "=== $kernel: --list-cases itself failed, aborting this kernel ===" \
      | tee -a "$DRIVER_LOG"
    return 1
  }
  local total
  total=$(printf '%s\n' "$cases" | wc -l)
  echo "=== $kernel: $total cases ===" | tee -a "$DRIVER_LOG"

  local n=0
  local case
  while IFS= read -r case; do
    n=$((n + 1))
    if [ -f "$sweep_file" ] \
      && jq -e --arg c "$case" 'select(.case == $c)' "$sweep_file" >/dev/null 2>&1; then
      continue # already resolved (ok or failed) by a previous invocation
    fi

    echo "=== $kernel [$n/$total]: $case starting at $(date) ===" >> "$case_log"
    timeout "$PER_CASE_TIMEOUT" python "$script" "${extra_args[@]}" \
      --only-case "$case" --sweep-file "$sweep_file" >> "$case_log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "=== $kernel [$n/$total]: $case FAILED rc=$rc at $(date) ===" \
        | tee -a "$DRIVER_LOG"
      # Recorded here, from OUTSIDE the (possibly hard-crashed) subprocess
      # -- not relying on it having recorded anything about itself.
      printf '{"case": %s, "status": "failed", "rc": %d}\n' \
        "$(printf '%s' "$case" | jq -R .)" "$rc" >> "$sweep_file"
    fi
  done <<< "$cases"

  echo "=== $kernel: sweep done at $(date), building report ===" | tee -a "$DRIVER_LOG"
  python "$script" "${extra_args[@]}" --report-from-sweep "$sweep_file" \
    > "$LOGDIR/bench_${kernel}_fullsweep_report.log" 2>&1
}

echo "=== run_full_sweep started at $(date) ===" > "$DRIVER_LOG"
# Cheapest/lowest-risk kernels (already-cached, small tensors) first;
# scaled_mm_blockwise/scaled_mm last since they're the biggest grids
# (144/168 shapes) and the ones known to hit OOM on some shapes.
run_kernel_full_sweep rms_norm_dynamic_per_token_quant --full
run_kernel_full_sweep fused_qk_norm_rope --full
run_kernel_full_sweep dynamic_per_token_scaled_fp8_quant --full
run_kernel_full_sweep per_token_group_fp8_quant --full
run_kernel_full_sweep rms_norm_per_block_quant --full
run_kernel_full_sweep silu_and_mul_dynamic_per_token_quant --full
run_kernel_full_sweep silu_and_mul_per_block_quant --full
run_kernel_full_sweep scaled_mm_blockwise --full
run_kernel_full_sweep scaled_mm --full
echo "=== run_full_sweep ALL DONE at $(date) ===" | tee -a "$DRIVER_LOG"
