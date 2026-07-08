#!/usr/bin/env zsh
# Sequential re-run of all 9 Helion kernels' benchmarks with the corrected
# XPUGraph-based timing methodology (bench_utils.py's do_bench_xpu_graph,
# an XPU port of triton.testing.do_bench_cudagraph -- the blog's own
# methodology, which plain do_bench doesn't match: it measures Python/
# dispatch overhead alongside real kernel time, which inflated the
# "vs torch.compile" speedups reported by the previous run by roughly
# 4-5x for the fastest kernels -- see RESULTS.md).
#
# HELION_AUTOTUNE_IGNORE_ERRORS=1 + a per-kernel timeout are kept from the
# previous run's incident (rms_norm_per_block_quant's autotuner hanging
# instead of raising on an XPU Triton gap -- see RESULTS.md).
set -uo pipefail

cd /home/tongsu/vllm
source ~/torch-xpu-env/.venv/bin/activate

export HELION_CACHE_DIR=/home/tongsu/vllm/.helion_cache
export HELION_AUTOTUNE_IGNORE_ERRORS=1
LOGDIR=/home/tongsu/vllm/benchmark_logs
PER_KERNEL_TIMEOUT=3600

KERNELS=(
  rms_norm_dynamic_per_token_quant
  fused_qk_norm_rope
  dynamic_per_token_scaled_fp8_quant
  per_token_group_fp8_quant
  rms_norm_per_block_quant
  silu_and_mul_dynamic_per_token_quant
  silu_and_mul_per_block_quant
  scaled_mm_blockwise
  scaled_mm
)

echo "=== run_all_9_xpugraph started at $(date) ===" > "$LOGDIR/driver3.log"
for k in "${KERNELS[@]}"; do
  echo "=== $k: starting at $(date) ===" | tee -a "$LOGDIR/driver3.log"
  t0=$(date +%s)
  timeout "$PER_KERNEL_TIMEOUT" python "benchmarks/kernels/helion/bench_${k}.py" > "$LOGDIR/bench_${k}_xpugraph.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "=== $k: finished at $(date), rc=$rc, elapsed=$((t1 - t0))s ===" | tee -a "$LOGDIR/driver3.log"
done
echo "=== run_all_9_xpugraph ALL DONE at $(date) ===" | tee -a "$LOGDIR/driver3.log"
