#!/usr/bin/env zsh
# Sequential re-run of the 7 non-GEMM Helion kernels' benchmarks, using each
# script's own default settings (autotune_effort=quick, budget_seconds=45
# where supported) to stay comparable with the prior preliminary numbers in
# RESULTS.md -- only the baseline methodology changed (3-way restructure),
# not the autotuning budget.
#
# Helion's autotune search cache is redirected (via HELION_CACHE_DIR) to a
# project-local, gitignored directory so results persist across sessions
# instead of living in /tmp.
set -uo pipefail

cd /home/tongsu/vllm
source ~/torch-xpu-env/.venv/bin/activate

export HELION_CACHE_DIR=/home/tongsu/vllm/.helion_cache
LOGDIR=/home/tongsu/vllm/benchmark_logs

KERNELS=(
  dynamic_per_token_scaled_fp8_quant
  rms_norm_dynamic_per_token_quant
  per_token_group_fp8_quant
  rms_norm_per_block_quant
  silu_and_mul_dynamic_per_token_quant
  silu_and_mul_per_block_quant
  fused_qk_norm_rope
)

echo "=== run_all_7 started at $(date) ===" > "$LOGDIR/driver.log"
for k in "${KERNELS[@]}"; do
  echo "=== $k: starting at $(date) ===" | tee -a "$LOGDIR/driver.log"
  t0=$(date +%s)
  python "benchmarks/kernels/helion/bench_${k}.py" > "$LOGDIR/bench_${k}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "=== $k: finished at $(date), rc=$rc, elapsed=$((t1 - t0))s ===" | tee -a "$LOGDIR/driver.log"
done
echo "=== run_all_7 ALL DONE at $(date) ===" | tee -a "$LOGDIR/driver.log"
