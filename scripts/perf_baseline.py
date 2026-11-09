#!/usr/bin/env python3
"""Record and compare identity-stamped performance baselines.

A *baseline* is one measured result plus everything needed to decide whether a
later measurement is comparable with it:

- **source**: the exact git commit; a dirty tree is refused unless explicitly
  allowed, and then the diff digest is recorded;
- **build**: CMake build type, CPU ISA envelope, compiler and GPU backends from
  the build directory's ``CMakeCache.txt``;
- **host**: architecture, kernel, CPU model, logical CPUs, page size and NUMA
  node layout;
- **workload**: the measured document's own identity (benchmark prompt digest
  and token counts, or a kernel harness name and parameters);
- **metrics**: normalized ``{value, unit, better, samples}`` entries.

Two input kinds are accepted:

``llaminar-benchmark``
    The ``llaminar.benchmark.v1`` document written by
    ``llaminar2 benchmark --benchmark-json-output``. Prefill and decode
    throughput and decode latency become metrics; the complete document is
    retained verbatim under ``raw``.
``kernel-metrics``
    A generic document ``{"harness": str, "parameters": {...}, "metrics":
    {name: {"value", "unit", "better": "higher"|"lower", "samples"?}}}`` for
    ``Perf__`` harnesses.

``diff`` refuses to compare baselines whose workload, build envelope or host
CPU differ, unless each differing field is named with ``--allow-identity-change``.
A metric regresses only when it moves in the worse direction by more than the
larger of ``--threshold-pct`` and twice the combined robust relative spread
(1.4826 x median absolute deviation over the median) of both sample sets. Exit codes: 0 comparable without regression,
1 at least one regression, 2 incomparable or invalid input.

Baselines are evidence, not source: store them with the measurement record
(tracker evidence or the corpora repository), never in this source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "llaminar.perf-baseline.v1"
KINDS = ("llaminar-benchmark", "kernel-metrics")


class BaselineError(Exception):
    """Invalid input, refused identity or incomparable baselines."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(source_dir: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(source_dir), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise BaselineError(f"git {' '.join(args)} failed in {source_dir}: {result.stderr.strip()}")
    return result.stdout


def source_identity(source_dir: Path, allow_dirty: bool) -> dict[str, Any]:
    """Return the exact commit; a dirty tree is fatal unless explicitly allowed."""

    commit = _git(source_dir, "rev-parse", "HEAD").strip()
    status = _git(source_dir, "status", "--porcelain", "--untracked-files=no")
    dirty = bool(status.strip())
    if dirty and not allow_dirty:
        raise BaselineError("source tree has uncommitted tracked changes; commit them or pass --allow-dirty")
    identity: dict[str, Any] = {"commit": commit, "dirty": dirty}
    if dirty:
        identity["diff_sha256"] = _sha256_bytes(_git(source_dir, "diff", "HEAD").encode())
    return identity


def build_identity(build_dir: Path) -> dict[str, Any]:
    """Read the build envelope from CMakeCache.txt; missing keys stay explicit nulls."""

    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        raise BaselineError(f"{cache} not found; --build-dir must be a configured CMake build tree")
    values: dict[str, str] = {}
    for line in cache.read_text(errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        key_type, value = line.split("=", 1)
        values[key_type.split(":", 1)[0]] = value
    # CMake records the detected compiler version per CMake release, not in the cache.
    for compiler_file in sorted(build_dir.glob("CMakeFiles/*/CMakeCXXCompiler.cmake")):
        for line in compiler_file.read_text(errors="replace").splitlines():
            if line.startswith("set(CMAKE_CXX_COMPILER_VERSION "):
                values.setdefault("CMAKE_CXX_COMPILER_VERSION", line.split('"')[1])
    return {
        "build_type": values.get("CMAKE_BUILD_TYPE") or None,
        "cpu_isa": values.get("LLAMINAR_CPU_ISA") or None,
        "cxx_compiler": values.get("CMAKE_CXX_COMPILER") or None,
        "cxx_compiler_version": values.get("CMAKE_CXX_COMPILER_VERSION") or None,
        "have_cuda": values.get("HAVE_CUDA") or None,
        "have_rocm": values.get("HAVE_ROCM") or None,
    }


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        fields = {}
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if ":" in line:
                key, value = (part.strip() for part in line.split(":", 1))
                fields.setdefault(key, value)
        for key in ("model name", "CPU part"):
            if key in fields:
                implementer = fields.get("CPU implementer")
                return f"{implementer}:{fields[key]}" if key == "CPU part" and implementer else fields[key]
    if sys.platform == "darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    return platform.processor() or None


def _numa_nodes() -> list[dict[str, Any]]:
    root = Path("/sys/devices/system/node")
    nodes = []
    if root.is_dir():
        for node in sorted(root.glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
            cpulist = (node / "cpulist").read_text().strip() if (node / "cpulist").is_file() else ""
            mem_kb = None
            meminfo = node / "meminfo"
            if meminfo.is_file():
                for line in meminfo.read_text().splitlines():
                    if "MemTotal:" in line:
                        mem_kb = int(line.split(":", 1)[1].split()[0])
            nodes.append({"node": int(node.name[4:]), "cpus": cpulist, "mem_total_kb": mem_kb})
    return nodes


def host_identity() -> dict[str, Any]:
    """Describe the measuring host; NUMA nodes are empty where the kernel publishes none."""

    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "page_size": os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else None,
        "numa_nodes": _numa_nodes(),
    }


def _metric(value: Any, unit: str, better: str, samples: Sequence[float] | None = None) -> dict[str, Any]:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise BaselineError(f"metric value {value!r} is not a finite number")
    if better not in ("higher", "lower"):
        raise BaselineError(f"metric direction {better!r} must be 'higher' or 'lower'")
    metric: dict[str, Any] = {"value": float(value), "unit": unit, "better": better}
    if samples:
        metric["samples"] = [float(s) for s in samples]
    return metric


def extract(kind: str, document: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (workload identity, metrics) for one supported input document."""

    if kind == "llaminar-benchmark":
        if document.get("schema") != "llaminar.benchmark.v1":
            raise BaselineError(f"expected schema llaminar.benchmark.v1, got {document.get('schema')!r}")
        try:
            tokens, throughput = document["tokens"], document["throughput_tokens_per_sec"]
            latency = document["decode_latency_ms"]
            workload = {
                "prompt_sha256": document["prompt"]["sha256"],
                "prefill_tokens": tokens["prefill"],
                "decode_tokens": tokens["decode"],
            }
            metrics = {
                "prefill_tokens_per_sec": _metric(throughput["prefill"], "tok/s", "higher"),
                "decode_tokens_per_sec": _metric(throughput["decode"], "tok/s", "higher"),
                "decode_latency_mean_ms": _metric(latency["mean"], "ms", "lower"),
                "decode_latency_p50_ms": _metric(latency["p50"], "ms", "lower"),
            }
        except (KeyError, TypeError) as error:
            raise BaselineError(f"llaminar.benchmark.v1 document lacks {error}") from error
        return workload, metrics
    if kind == "kernel-metrics":
        harness, metrics = document.get("harness"), document.get("metrics")
        if not isinstance(harness, str) or not harness:
            raise BaselineError("kernel-metrics document requires a non-empty 'harness'")
        if not isinstance(metrics, Mapping) or not metrics:
            raise BaselineError("kernel-metrics document requires a non-empty 'metrics' object")
        normalized = {}
        for name, entry in metrics.items():
            if not isinstance(entry, Mapping):
                raise BaselineError(f"metric {name!r} must be an object")
            normalized[name] = _metric(entry.get("value"), str(entry.get("unit", "")), entry.get("better"),
                                       entry.get("samples"))
        return {"harness": harness, "parameters": document.get("parameters", {})}, normalized
    raise BaselineError(f"unknown kind {kind!r}; expected one of {KINDS}")


def record(kind: str, input_path: Path, source_dir: Path, build_dir: Path, *, allow_dirty: bool = False,
           diagnostic: bool = False, command: str | None = None) -> dict[str, Any]:
    """Build one baseline document. Non-Release builds require --diagnostic and never certify."""

    raw_bytes = input_path.read_bytes()
    try:
        document = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise BaselineError(f"{input_path} is not JSON: {error}") from error
    workload, metrics = extract(kind, document)
    build = build_identity(build_dir)
    if build["build_type"] != "Release" and not diagnostic:
        raise BaselineError(f"build type {build['build_type']!r} is not Release; pass --diagnostic for a "
                            "non-certifying record")
    return {
        "schema": SCHEMA,
        "kind": kind,
        "certifying": build["build_type"] == "Release" and not diagnostic,
        "source": source_identity(source_dir, allow_dirty),
        "build": build,
        "host": host_identity(),
        "command": command,
        "workload": workload,
        "metrics": metrics,
        "raw_sha256": _sha256_bytes(raw_bytes),
        "raw": document,
    }


# Identity fields that must match before two baselines are comparable.
IDENTITY_FIELDS = {
    "kind": lambda b: b["kind"],
    "workload": lambda b: b["workload"],
    "build_type": lambda b: b["build"]["build_type"],
    "cpu_isa": lambda b: b["build"]["cpu_isa"],
    "host_cpu": lambda b: b["host"]["cpu_model"],
    "host_page_size": lambda b: b["host"]["page_size"],
}


def _relative_spread(metric: Mapping[str, Any]) -> float:
    """Robust relative spread: 1.4826 x median absolute deviation over the median.

    Timing samples carry occasional scheduler stalls; a standard deviation lets
    one stall widen the tolerance arbitrarily, while the scaled MAD estimates
    the same spread for normal noise and ignores isolated outliers.
    """

    samples = metric.get("samples") or []
    if len(samples) < 2:
        return 0.0
    median = statistics.median(samples)
    if median == 0:
        return 0.0
    return 1.4826 * statistics.median(abs(sample - median) for sample in samples) / abs(median)


def diff(baseline: Mapping[str, Any], candidate: Mapping[str, Any], threshold_pct: float,
         allowed_changes: Sequence[str] = ()) -> dict[str, Any]:
    """Compare two baselines; raise BaselineError when they are not comparable."""

    for document in (baseline, candidate):
        if document.get("schema") != SCHEMA:
            raise BaselineError(f"expected schema {SCHEMA}, got {document.get('schema')!r}")
    unknown = set(allowed_changes) - set(IDENTITY_FIELDS)
    if unknown:
        raise BaselineError(f"unknown identity fields {sorted(unknown)}; choose from {sorted(IDENTITY_FIELDS)}")
    changed = [name for name, get in IDENTITY_FIELDS.items() if get(baseline) != get(candidate)]
    refused = [name for name in changed if name not in allowed_changes]
    if refused:
        raise BaselineError(f"baselines are not comparable; differing identity: {refused}")
    rows = []
    for name in sorted(set(baseline["metrics"]) | set(candidate["metrics"])):
        if name not in baseline["metrics"] or name not in candidate["metrics"]:
            raise BaselineError(f"metric {name!r} exists in only one baseline")
        base, cand = baseline["metrics"][name], candidate["metrics"][name]
        if base["better"] != cand["better"] or base["unit"] != cand["unit"]:
            raise BaselineError(f"metric {name!r} changed unit or direction")
        if base["value"] == 0:
            raise BaselineError(f"metric {name!r} has a zero baseline value")
        change_pct = 100.0 * (cand["value"] - base["value"]) / abs(base["value"])
        noise_pct = 200.0 * math.hypot(_relative_spread(base), _relative_spread(cand))
        tolerance_pct = max(threshold_pct, noise_pct)
        worse = change_pct < 0 if base["better"] == "higher" else change_pct > 0
        if abs(change_pct) <= tolerance_pct:
            verdict = "within-tolerance"
        else:
            verdict = "regressed" if worse else "improved"
        rows.append({"metric": name, "unit": base["unit"], "baseline": base["value"], "candidate": cand["value"],
                     "change_pct": change_pct, "tolerance_pct": tolerance_pct, "verdict": verdict})
    return {
        "baseline_commit": baseline["source"]["commit"],
        "candidate_commit": candidate["source"]["commit"],
        "allowed_identity_changes": changed,
        "certifying": bool(baseline["certifying"] and candidate["certifying"]),
        "metrics": rows,
        "regressions": [row["metric"] for row in rows if row["verdict"] == "regressed"],
    }


def _print_diff(report: Mapping[str, Any]) -> None:
    print(f"baseline {report['baseline_commit'][:12]} -> candidate {report['candidate_commit'][:12]}"
          f" ({'certifying' if report['certifying'] else 'NON-CERTIFYING'})")
    if report["allowed_identity_changes"]:
        print(f"allowed identity changes: {', '.join(report['allowed_identity_changes'])}")
    for row in report["metrics"]:
        print(f"  {row['metric']:<28} {row['baseline']:>12.4f} -> {row['candidate']:>12.4f} {row['unit']:<6}"
              f" {row['change_pct']:+7.2f}% (tol {row['tolerance_pct']:.2f}%) {row['verdict']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="stamp one measurement with source, build and host identity")
    rec.add_argument("--kind", choices=KINDS, required=True)
    rec.add_argument("--input", type=Path, required=True)
    rec.add_argument("--build-dir", type=Path, required=True)
    rec.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1])
    rec.add_argument("--output", type=Path, required=True)
    rec.add_argument("--allow-dirty", action="store_true")
    rec.add_argument("--diagnostic", action="store_true", help="permit a non-Release, non-certifying record")
    rec.add_argument("--run-command", help="exact command that produced --input, recorded verbatim")
    cmp_ = sub.add_parser("diff", help="compare a candidate baseline with a reference baseline")
    cmp_.add_argument("--baseline", type=Path, required=True)
    cmp_.add_argument("--candidate", type=Path, required=True)
    cmp_.add_argument("--threshold-pct", type=float, default=2.0)
    cmp_.add_argument("--allow-identity-change", action="append", default=[], choices=sorted(IDENTITY_FIELDS))
    cmp_.add_argument("--json", action="store_true", help="print the machine-readable report")
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            document = record(args.kind, args.input, args.source_dir, args.build_dir, allow_dirty=args.allow_dirty,
                              diagnostic=args.diagnostic, command=args.run_command)
            args.output.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
            return 0
        if args.threshold_pct < 0:
            raise BaselineError("--threshold-pct must be non-negative")
        report = diff(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()),
                      args.threshold_pct, args.allow_identity_change)
        print(json.dumps(report, indent=1)) if args.json else _print_diff(report)
        return 1 if report["regressions"] else 0
    except (BaselineError, OSError, json.JSONDecodeError) as error:
        print(f"perf_baseline: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
