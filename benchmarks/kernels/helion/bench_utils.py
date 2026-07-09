# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared XPU benchmarking helpers for the Helion kernel-level benchmarks.

``triton.testing.do_bench_cudagraph`` -- what the blog's own methodology
uses ("we enabled CudaGraph mode via triton.testing.do_bench_cudagraph ...
to get rid of noises like dispatch overhead") -- is hardcoded to
``torch.cuda.*`` APIs and does not run on XPU. Without it, plain
``triton.testing.do_bench`` measures Python/dispatch overhead per call
alongside actual kernel time. For the sub-millisecond kernels here, that
overhead is a large fraction of what's measured, and it inflates the
``torch.compile`` baseline (which has more per-call dispatch/guard-checking
overhead than Helion's thinner wrapper) far more than the Helion kernel,
producing misleadingly large "speedups" -- confirmed directly: e.g.
``rms_norm_dynamic_per_token_quant`` measured at 13.1x speedup with plain
``do_bench`` dropped to ~2.4x once dispatch overhead was eliminated, closely
matching the blog's own H100/B200 numbers for the same kernel (1.18-1.24x).

``do_bench_xpu_graph`` below is a same-algorithm XPU port of
``do_bench_cudagraph``, using ``torch.xpu.XPUGraph`` (XPU's direct analog of
``torch.cuda.CUDAGraph`` -- confirmed present and working on this platform),
so both sides of every comparison in this benchmark suite get the same
dispatch-overhead-free measurement the blog's methodology intends.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def do_bench_xpu_graph(
    fn: Callable[[], object],
    rep: int = 20,
    n_retries: int = 10,
    return_mode: str = "median",
) -> float:
    """XPU port of ``triton.testing.do_bench_cudagraph``.

    Captures ``fn`` called back-to-back ``n_repeat`` times into a single
    ``torch.xpu.XPUGraph`` (``n_repeat`` chosen so the total replay is
    roughly ``rep`` ms, same estimation approach ``do_bench_cudagraph``
    uses), then times ``n_retries`` replays of that graph. Capturing
    multiple repeats into one graph -- rather than replaying a
    single-call graph in a Python loop -- is what actually eliminates
    per-call dispatch overhead from the measurement.

    Args:
        fn: Zero-argument callable to benchmark.
        rep: Target total replay duration in ms, used to pick how many
            calls to capture per graph.
        n_retries: Number of times to replay the captured graph for the
            returned statistic.
        return_mode: One of "median", "mean", "min", "max".

    Returns:
        Per-call time in ms, aggregated across ``n_retries`` graph replays
        according to ``return_mode``.
    """
    assert return_mode in ("median", "mean", "min", "max")

    # Warm up on a side stream first -- same prerequisite CUDA graph
    # capture has (lets any first-call compilation/allocation happen
    # outside the captured region).
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            fn()
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()

    # Estimate single-call time to size n_repeat (mirrors
    # do_bench_cudagraph's own estimation step).
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        fn()
    end_event.record()
    torch.xpu.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    n_repeat = max(1, int(rep / estimate_ms)) if estimate_ms > 0 else 1000

    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(n_repeat):
            fn()
    torch.xpu.synchronize()

    samples: list[float] = []
    for _ in range(n_retries):
        start_event = torch.xpu.Event(enable_timing=True)
        end_event = torch.xpu.Event(enable_timing=True)
        start_event.record()
        graph.replay()
        end_event.record()
        torch.xpu.synchronize()
        samples.append(start_event.elapsed_time(end_event) / n_repeat)

    samples.sort()
    if return_mode == "median":
        return samples[len(samples) // 2]
    if return_mode == "mean":
        return sum(samples) / len(samples)
    if return_mode == "min":
        return samples[0]
    return samples[-1]


def bench_with_xpu_graph_fallback(
    fn: Callable[[], object],
    warmup: int = 25,
    rep: int = 100,
) -> tuple[float, bool]:
    """Try XPUGraph-based benchmarking; fall back to plain ``do_bench``.

    Returns ``(ms, xpu_graph_enabled)``. Falls back (rather than raising)
    for cases that can't be graph-captured -- e.g. a kernel invocation
    that triggers host-side control flow or a fresh compile inside the
    captured region -- so a single unsupported shape/kernel doesn't take
    down the whole benchmark run. Callers should surface
    ``xpu_graph_enabled`` (e.g. as a column) rather than silently
    reporting a possibly-inflated plain-``do_bench`` number as if it were
    the corrected one.
    """
    from triton.testing import do_bench

    try:
        return do_bench_xpu_graph(fn, rep=rep), True
    except Exception:
        return do_bench(fn, warmup=warmup, rep=rep, return_mode="median"), False


def print_detailed_report(
    hardware: str,
    baseline_name: str,
    rows: list[tuple[str, float, float]],
) -> None:
    """Print a detailed per-case report.

    ``rows`` is a list of ``(case_name, baseline_ms, kernel_ms)`` tuples;
    speedup is computed here so callers don't need to.

    Format matches::

        Hardware: H100
        Baseline: torch.compile

        case                       | baseline_ms | kernel_ms | speedup(x)
        ---------------------------+-------------+-----------+-----------
        hidden_size_2048_ntok_1    | 0.003       | 0.002     | 1.622
    """
    print(f"\nHardware: {hardware}")
    print(f"Baseline: {baseline_name}\n")

    name_width = max([len("case")] + [len(name) for name, _, _ in rows]) + 1
    header = (
        f"{'case':<{name_width}}| {'baseline_ms':<11} | {'kernel_ms':<9} | speedup(x)"
    )
    print(header)
    print("-" * name_width + "+" + "-" * 13 + "+" + "-" * 11 + "+" + "-" * 11)
    for name, baseline_ms, kernel_ms in rows:
        speedup = baseline_ms / kernel_ms if kernel_ms > 0 else 0.0
        print(
            f"{name:<{name_width}}| {baseline_ms:<11.3f} | {kernel_ms:<9.3f} | "
            f"{speedup:.3f}"
        )


CASE_LINE_PREFIX = "CASE:"


def print_case_name(case_name: str) -> None:
    """Prints one ``--list-cases`` line, tagged with a unique, unambiguous
    prefix (``run_full_sweep.sh`` greps for it and strips it).

    vLLM's own logging (import-time WARNING/INFO messages from
    registering all 9 Helion kernels) goes to stdout, not stderr, in this
    environment -- confirmed directly: a plain ``print(case_name)`` here
    got ~30 log lines mixed into ``run_full_sweep.sh``'s ``$(...)``
    capture of ``--list-cases``'s output, corrupting the case count and,
    worse, silently turning one of those log lines into a bogus
    "--only-case <warning text>" argument on the next invocation. Only
    excluding known noise patterns (e.g. lines starting with "WARNING"/
    "INFO") would be fragile against whatever other noise shows up next;
    positively tagging the lines we actually want is robust regardless.
    """
    print(f"{CASE_LINE_PREFIX}{case_name}")


def append_sweep_result(path: str, case: str, **fields: Any) -> None:
    """Append one case's result to a JSON-lines crash-safe sweep file.

    Used by a script's ``--only-case`` mode to durably persist a result the
    instant it's computed. flush()+fsync() so the write survives even if
    the process is killed immediately after (e.g. a subsequent case in the
    same ``run_full_sweep.sh`` loop iteration crashes -- this write is for
    an already-completed case, not the crashing one, but the same
    durability guarantee applies uniformly).

    Each line is a JSON object: ``{"case": ..., "status": "ok", **fields}``.
    ``run_full_sweep.sh`` appends its own ``{"case": ..., "status":
    "failed", "rc": ...}`` line directly (via jq/printf, not this function)
    when a case's subprocess exits non-zero -- deliberately *not* relying
    on the crashed process to have recorded anything about its own
    failure, since a hard crash (segfault, driver abort) or a
    timeout-killed hang leaves no Python code running to do that. See
    ``SHAPE_AUDIT.md`` / RESULTS.md for why this split (durable
    self-reporting on success, external observation on failure) is the
    design, not an oversight.
    """
    rec = {"case": case, "status": "ok", **fields}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_sweep_results(path: str) -> dict[str, dict[str, Any]]:
    """Load a JSON-lines crash-safe sweep file into ``{case: record}``.

    Tolerant of a missing file (returns ``{}``) and of unparsable lines
    (skipped -- e.g. a line torn by a hard kill mid-write; vanishingly
    unlikely for a single short ``write()`` syscall followed by fsync, but
    cheap to guard against). If a case appears more than once (shouldn't
    happen in normal operation, but e.g. a manually-edited file), the last
    occurrence wins.
    """
    results: dict[str, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return results
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        results[rec["case"]] = rec
    return results


def add_crash_safe_args(parser: argparse.ArgumentParser) -> None:
    """Adds the 4 crash-safe-sweep CLI args shared identically by all 9
    kernels' bench scripts, so ``run_full_sweep.sh`` can drive any of them
    the same way. See ``bench_scaled_mm.py`` for the reference
    implementation of how a script wires these up in its own ``main()``:
    ``--list-cases`` and ``--report-from-sweep`` return immediately;
    ``--only-case`` measures one case and appends to ``--sweep-file``,
    letting any exception propagate uncaught (a fresh subprocess, not this
    one, measures the next case -- see run_full_sweep.sh and RESULTS.md's
    "What's not done" for why).
    """
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print every case name (one per line) and exit. No XPU use, "
        "no measurement -- for a driver script to enumerate what to run.",
    )
    parser.add_argument(
        "--only-case",
        default=None,
        help="Measure just this one case (see --list-cases for names); "
        "appends its result to --sweep-file and exits. Requires "
        "--sweep-file. For use by run_full_sweep.sh's crash-safe "
        "per-case driver -- a case that OOMs/crashes only takes down "
        "this one invocation, not the rest of the sweep.",
    )
    parser.add_argument(
        "--sweep-file",
        default=None,
        help="JSON-lines file --only-case appends its result to, or "
        "--report-from-sweep reads a completed sweep from.",
    )
    parser.add_argument(
        "--report-from-sweep",
        default=None,
        metavar="SWEEP_FILE",
        help="Print the aggregate report (geomean + detailed table) from "
        "an already-completed --sweep-file, instead of measuring "
        "anything.",
    )
