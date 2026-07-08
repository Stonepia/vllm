#!/usr/bin/env zsh
# Re-run of the remaining/needs-retry kernels after the first attempt hung
# on rms_norm_per_block_quant (see driver.log + bench_rms_norm_per_block_quant.log
# from the first run: an autotuner candidate compiled Triton code that adds
# two Float8_e4m3fn tensors directly, which XPU's backend doesn't implement
# ("add_xpu" not implemented for 'Float8_e4m3fn'); encountering this hung the
# whole process for ~12h instead of a clean crash -- a genuine XPU-specific
# Helion autotuner issue, not something specific to our benchmark scripts).
#
# Two safety nets added vs. the first run:
#   1. HELION_AUTOTUNE_IGNORE_ERRORS=1 -- per Helion's own error message,
#      this makes the autotuner skip a failing candidate config and move on,
#      rather than raise (the raise path is what preceded the hang).
#   2. `timeout 3600` per kernel -- defense in depth in case some other,
#      different failure mode also hangs instead of crashing cleanly.
set -uo pipefail

cd /home/tongsu/vllm
source ~/torch-xpu-env/.venv/bin/activate

export HELION_CACHE_DIR=/home/tongsu/vllm/.helion_cache
export HELION_AUTOTUNE_IGNORE_ERRORS=1
LOGDIR=/home/tongsu/vllm/benchmark_logs
PER_KERNEL_TIMEOUT=3600

KERNELS=(
  rms_norm_per_block_quant
  silu_and_mul_per_block_quant
  fused_qk_norm_rope
)

echo "=== run_remaining started at $(date) ===" > "$LOGDIR/driver2.log"
for k in "${KERNELS[@]}"; do
  echo "=== $k: starting at $(date) ===" | tee -a "$LOGDIR/driver2.log"
  t0=$(date +%s)
  timeout "$PER_KERNEL_TIMEOUT" python "benchmarks/kernels/helion/bench_${k}.py" > "$LOGDIR/bench_${k}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "=== $k: finished at $(date), rc=$rc, elapsed=$((t1 - t0))s ===" | tee -a "$LOGDIR/driver2.log"
done
echo "=== run_remaining ALL DONE at $(date) ===" | tee -a "$LOGDIR/driver2.log"
