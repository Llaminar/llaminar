#!/usr/bin/env python3
"""Record and compare identity-bound performance baselines.

A baseline document (schema ``llaminar.perf-baseline.v2``) is one measured
result plus the identity needed to decide whether a later result is
comparable with it. Two ways of producing one:

``run``
    Executes the measuring command itself. The executable must live inside
    ``--build-dir``; it and every shared library it loads from that build tree
    are SHA-256 digested before and after the run and must not change. The
    command must exit 0 and must create ``--result-json``, which may not exist
    beforehand. Declared inputs (``--input-file NAME=PATH``: model, saved plan,
    prompt file, ...) are digested into the workload identity, together with
    every ``LLAMINAR_*``/``OMP_*``/``GOMP_*``/``KMP_*`` environment variable
    and the exact argv. Only a ``run`` from a clean tree in a ``Release`` build
    receives grade ``bound``.
``import``
    Wraps an already existing result. Nothing ties that file to the build or
    host it is stamped with, so its grade is always ``unbound``.

``bound`` is a local self-attestation of how the record was produced, not a
tamper-proof certificate and not the repository's image certification.

Input kinds:

``llaminar-benchmark``
    The ``llaminar.benchmark.v1`` document of ``llaminar2 benchmark
    --benchmark-json-output``. ``success``, ``prefill_success`` and
    ``decode_success`` must be true. The workload identity is the prompt,
    token counts, iteration/warmup counts, runtime state architecture,
    execution path and primary device, and the complete
    ``runtime_configuration`` (sampling, MTP, prefix cache, ...). The model is
    NOT in that document: declare it with ``--input-file model=PATH``. The
    generated token ids form a separate ``output`` identity.
``kernel-metrics``
    ``{"harness": str, "parameters": {...}, "metrics": {name: {"value",
    "unit", "better": "higher"|"lower", "samples"?}}}`` from ``Perf__``
    harnesses; harness and parameters are the workload identity.

``diff`` validates both documents completely, then refuses to compare unless
every identity field matches or is named with ``--allow-identity-change``.
A metric regresses when it moves in the worse direction by more than
``max(--threshold-pct, 2 * hypot(spread_a, spread_b))``, where a spread is the
robust relative spread ``1.4826 * MAD / median`` of that metric's samples.
Exit codes: 0 comparable without regression, 1 regression, 2 refused or
invalid input. Store baselines with the measurement evidence, never in this
source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "llaminar.perf-baseline.v2"
KINDS = ("llaminar-benchmark", "kernel-metrics")
GRADES = ("bound", "unbound")
LAUNCH_ENV_PREFIXES = ("LLAMINAR_", "OMP_", "GOMP_", "KMP_")


class BaselineError(Exception):
    """Invalid input, refused identity or incomparable baselines."""


def _reject_constant(name: str) -> None:
    raise BaselineError(f"non-finite JSON constant {name} is not allowed")


def load_json(path: Path) -> Any:
    """Parse strict JSON: NaN/Infinity literals are rejected, not silently accepted."""

    try:
        return json.loads(path.read_text(), parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise BaselineError(f"{path} is not JSON: {error}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(source_dir: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(source_dir), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise BaselineError(f"git {' '.join(args)} failed in {source_dir}: {result.stderr.strip()}")
    return result.stdout


def source_identity(source_dir: Path, allow_dirty: bool) -> dict[str, Any]:
    """Return the exact commit; a dirty tree is refused unless explicitly allowed."""

    commit = _git(source_dir, "rev-parse", "HEAD").strip()
    dirty = bool(_git(source_dir, "status", "--porcelain", "--untracked-files=no").strip())
    if dirty and not allow_dirty:
        raise BaselineError("source tree has uncommitted tracked changes; commit them or pass --allow-dirty")
    identity: dict[str, Any] = {"commit": commit, "dirty": dirty}
    if dirty:
        identity["diff_sha256"] = hashlib.sha256(_git(source_dir, "diff", "HEAD").encode()).hexdigest()
    return identity


def build_identity(build_dir: Path) -> dict[str, Any]:
    """Read the build envelope from CMakeCache.txt and CMake's detected compiler file."""

    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        raise BaselineError(f"{cache} not found; --build-dir must be a configured CMake build tree")
    values: dict[str, str] = {}
    for line in cache.read_text(errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        key_type, value = line.split("=", 1)
        values[key_type.split(":", 1)[0]] = value
    for compiler_file in sorted(build_dir.glob("CMakeFiles/*/CMakeCXXCompiler.cmake")):
        for line in compiler_file.read_text(errors="replace").splitlines():
            if line.startswith("set(CMAKE_CXX_COMPILER_VERSION "):
                values.setdefault("CMAKE_CXX_COMPILER_VERSION", line.split('"')[1])
    build_type = values.get("CMAKE_BUILD_TYPE") or None
    return {
        "build_type": build_type,
        "cpu_isa": values.get("LLAMINAR_CPU_ISA") or None,
        "cxx_compiler": values.get("CMAKE_CXX_COMPILER") or None,
        "cxx_compiler_version": values.get("CMAKE_CXX_COMPILER_VERSION") or None,
        "cxx_flags": " ".join(filter(None, (values.get("CMAKE_CXX_FLAGS"),
                                            values.get(f"CMAKE_CXX_FLAGS_{(build_type or '').upper()}")))),
        "backends": {"cuda": values.get("HAVE_CUDA") or None, "rocm": values.get("HAVE_ROCM") or None},
        "shared_core": values.get("LLAMINAR_SHARED_CORE") or None,
    }


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        fields: dict[str, str] = {}
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if ":" in line:
                key, value = (part.strip() for part in line.split(":", 1))
                fields.setdefault(key, value)
        if "model name" in fields:
            return fields["model name"]
        if "CPU part" in fields:
            return f"{fields.get('CPU implementer', '?')}:{fields['CPU part']}"
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
            if (node / "meminfo").is_file():
                for line in (node / "meminfo").read_text().splitlines():
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
        "page_size": os.sysconf("SC_PAGE_SIZE"),
        "numa_nodes": _numa_nodes(),
    }


def _finite(value: Any, what: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise BaselineError(f"{what} {value!r} is not a finite number")
    return float(value)


def _metric(entry: Any, name: str) -> dict[str, Any]:
    """Validate and normalize one metric; shared by input extraction and document validation."""

    if not isinstance(entry, Mapping):
        raise BaselineError(f"metric {name!r} must be an object")
    value = _finite(entry.get("value"), f"metric {name!r} value")
    if value <= 0:
        raise BaselineError(f"metric {name!r} value {value} must be positive")
    unit, better = entry.get("unit"), entry.get("better")
    if not isinstance(unit, str) or not unit:
        raise BaselineError(f"metric {name!r} requires a non-empty unit")
    if better not in ("higher", "lower"):
        raise BaselineError(f"metric {name!r} direction {better!r} must be 'higher' or 'lower'")
    metric: dict[str, Any] = {"value": value, "unit": unit, "better": better}
    samples = entry.get("samples")
    if samples is not None:
        if not isinstance(samples, list) or not samples:
            raise BaselineError(f"metric {name!r} samples must be a non-empty list")
        metric["samples"] = [_finite(s, f"metric {name!r} sample") for s in samples]
    return metric


def _metrics(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not raw:
        raise BaselineError("metrics must be a non-empty object")
    return {str(name): _metric(entry, str(name)) for name, entry in raw.items()}


def extract(kind: str, document: Mapping[str, Any]) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Return (workload identity, output identity, metrics) for one input document."""

    if kind == "llaminar-benchmark":
        if document.get("schema") != "llaminar.benchmark.v1":
            raise BaselineError(f"expected schema llaminar.benchmark.v1, got {document.get('schema')!r}")
        for flag in ("success", "prefill_success", "decode_success"):
            if document.get(flag) is not True:
                raise BaselineError(f"benchmark {flag} is {document.get(flag)!r}; failed runs are not baselines")
        try:
            state, tokens = document["runtime_state"], document["tokens"]
            workload = {
                "prompt": document["prompt"],
                "tokens": {key: tokens[key] for key in ("prefill", "decode", "decode_after_prefill")},
                "measurement_iterations": document["measurement_iterations"],
                "warmup_iterations": document["warmup_iterations"],
                "runtime_state": {key: state[key] for key in ("architecture", "execution_path", "primary_device")},
                "runtime_configuration": document["runtime_configuration"],
            }
            output = {"generated_token_ids_sha256": hashlib.sha256(
                json.dumps(document["generated_token_ids"]).encode()).hexdigest()}
            throughput, latency = document["throughput_tokens_per_sec"], document["decode_latency_ms"]
            metrics = _metrics({
                "prefill_tokens_per_sec": {"value": throughput["prefill"], "unit": "tok/s", "better": "higher"},
                "decode_tokens_per_sec": {"value": throughput["decode"], "unit": "tok/s", "better": "higher"},
                "decode_latency_mean_ms": {"value": latency["mean"], "unit": "ms", "better": "lower"},
                "decode_latency_p50_ms": {"value": latency["p50"], "unit": "ms", "better": "lower"},
            })
        except (KeyError, TypeError) as error:
            raise BaselineError(f"llaminar.benchmark.v1 document lacks {error}") from error
        return workload, output, metrics
    if kind == "kernel-metrics":
        harness, parameters = document.get("harness"), document.get("parameters")
        if not isinstance(harness, str) or not harness:
            raise BaselineError("kernel-metrics document requires a non-empty 'harness'")
        if not isinstance(parameters, Mapping):
            raise BaselineError("kernel-metrics document requires a 'parameters' object")
        return {"harness": harness, "parameters": parameters}, None, _metrics(document.get("metrics"))
    raise BaselineError(f"unknown kind {kind!r}; expected one of {KINDS}")


def _declared_inputs(specs: Sequence[str]) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise BaselineError(f"--input-file expects NAME=PATH, got {spec!r}")
        if name in inputs:
            raise BaselineError(f"--input-file {name!r} declared twice")
        if not Path(path).is_file():
            raise BaselineError(f"--input-file {name}: {path} is not a file")
        inputs[name] = _sha256_file(Path(path))
    return inputs


def _build_tree_closure(executable: Path, build_dir: Path) -> dict[str, str]:
    """Digest the executable and every shared library it resolves inside the build tree."""

    paths = {executable}
    if shutil.which("ldd"):
        listing = subprocess.run(["ldd", str(executable)], capture_output=True, text=True).stdout
        for line in listing.splitlines():
            parts = line.split("=>")
            target = (parts[1] if len(parts) > 1 else parts[0]).strip().split(" (")[0]
            if target.startswith("/") and Path(target).resolve().is_relative_to(build_dir):
                paths.add(Path(target).resolve())
    return {str(path.relative_to(build_dir)): _sha256_file(path) for path in sorted(paths)}


def _document(kind: str, raw: Mapping[str, Any], *, grade: str, source: dict, build: dict, inputs: dict,
              launch: dict | None) -> dict[str, Any]:
    workload, output, metrics = extract(kind, raw)
    workload = {**workload, "declared_inputs": inputs}
    if launch is not None:
        workload["launch_env"] = launch["env"]
    document = {"schema": SCHEMA, "kind": kind, "grade": grade, "source": source, "build": build,
                "host": host_identity(), "launch": launch, "workload": workload, "output": output,
                "metrics": metrics, "raw": raw}
    validate_document(document)
    return document


def run(kind: str, build_dir: Path, result_json: Path, command: Sequence[str], *, source_dir: Path,
        inputs: Sequence[str] = (), allow_dirty: bool = False) -> dict[str, Any]:
    """Execute one measurement and bind its result to the executable that produced it."""

    if not command:
        raise BaselineError("run requires the measuring command after --")
    build_dir = build_dir.resolve()
    found = shutil.which(command[0])
    if not found:
        raise BaselineError(f"executable {command[0]!r} not found")
    executable = Path(found).resolve()
    if not executable.is_relative_to(build_dir):
        raise BaselineError(f"executable {executable} is not inside --build-dir {build_dir}")
    if result_json.exists():
        raise BaselineError(f"{result_json} already exists; a run must create its result")
    source, build, declared = source_identity(source_dir, allow_dirty), build_identity(build_dir), _declared_inputs(inputs)
    before = _build_tree_closure(executable, build_dir)
    env = {key: value for key, value in sorted(os.environ.items()) if key.startswith(LAUNCH_ENV_PREFIXES)}
    started = time.time()
    completed = subprocess.run(list(command))
    finished = time.time()
    if completed.returncode != 0:
        raise BaselineError(f"measuring command exited {completed.returncode}")
    if _build_tree_closure(executable, build_dir) != before:
        raise BaselineError("executable or build-tree library changed during the run")
    if not result_json.is_file():
        raise BaselineError(f"measuring command did not create {result_json}")
    grade = "bound" if build["build_type"] == "Release" and not source["dirty"] else "unbound"
    launch = {"argv": list(command), "env": env, "binaries": before, "started_unix": started,
              "finished_unix": finished, "result_sha256": _sha256_file(result_json)}
    return _document(kind, load_json(result_json), grade=grade, source=source, build=build, inputs=declared,
                     launch=launch)


def import_result(kind: str, input_path: Path, build_dir: Path, *, source_dir: Path, inputs: Sequence[str] = (),
                  allow_dirty: bool = False) -> dict[str, Any]:
    """Wrap an existing result. Nothing binds it to this build or host: always unbound."""

    return _document(kind, load_json(input_path), grade="unbound", source=source_identity(source_dir, allow_dirty),
                     build=build_identity(build_dir), inputs=_declared_inputs(inputs), launch=None)


def validate_document(document: Any) -> None:
    """Complete structural and numeric validation; every entry point uses it."""

    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA:
        raise BaselineError(f"expected a {SCHEMA} document")
    if document.get("kind") not in KINDS or document.get("grade") not in GRADES:
        raise BaselineError("document kind or grade is invalid")
    for section in ("source", "build", "host", "workload"):
        if not isinstance(document.get(section), Mapping):
            raise BaselineError(f"document lacks the {section!r} object")
    if not isinstance(document["source"].get("commit"), str) or not isinstance(document["source"].get("dirty"), bool):
        raise BaselineError("source identity is incomplete")
    if document["grade"] == "bound" and (document.get("launch") is None or document["source"]["dirty"]
                                         or document["build"].get("build_type") != "Release"):
        raise BaselineError("a bound document requires a recorded launch, a clean tree and a Release build")
    _metrics(document.get("metrics"))


# Identity that must match (or be named with --allow-identity-change) before comparison.
IDENTITY_FIELDS = {
    "kind": lambda b: b["kind"],
    "workload": lambda b: {k: v for k, v in b["workload"].items() if k not in ("declared_inputs", "launch_env")},
    "declared_inputs": lambda b: b["workload"].get("declared_inputs"),
    "launch_env": lambda b: b["workload"].get("launch_env"),
    "output": lambda b: b.get("output"),
    "build_type": lambda b: b["build"].get("build_type"),
    "cpu_isa": lambda b: b["build"].get("cpu_isa"),
    "compiler": lambda b: (b["build"].get("cxx_compiler"), b["build"].get("cxx_compiler_version")),
    "cxx_flags": lambda b: b["build"].get("cxx_flags"),
    "backends": lambda b: b["build"].get("backends"),
    "host_cpu": lambda b: b["host"].get("cpu_model"),
    "host_topology": lambda b: (b["host"].get("logical_cpus"), b["host"].get("numa_nodes")),
    "host_page_size": lambda b: b["host"].get("page_size"),
    "host_kernel": lambda b: b["host"].get("kernel"),
}


def _relative_spread(metric: Mapping[str, Any]) -> float:
    """Robust relative spread 1.4826*MAD/median; isolated stalls cannot widen or fake a verdict."""

    samples = metric.get("samples") or []
    if len(samples) < 2:
        return 0.0
    median = statistics.median(samples)
    return 0.0 if median == 0 else 1.4826 * statistics.median(abs(s - median) for s in samples) / abs(median)


def diff(baseline: Any, candidate: Any, threshold_pct: float, allowed_changes: Sequence[str] = ()) -> dict[str, Any]:
    """Compare two validated baselines; raise BaselineError when invalid or not comparable."""

    validate_document(baseline)
    validate_document(candidate)
    threshold_pct = _finite(threshold_pct, "--threshold-pct")
    if threshold_pct < 0:
        raise BaselineError("--threshold-pct must be non-negative")
    unknown = set(allowed_changes) - set(IDENTITY_FIELDS)
    if unknown:
        raise BaselineError(f"unknown identity fields {sorted(unknown)}; choose from {sorted(IDENTITY_FIELDS)}")
    changed = [name for name, get in IDENTITY_FIELDS.items() if get(baseline) != get(candidate)]
    refused = [name for name in changed if name not in allowed_changes]
    if refused:
        raise BaselineError(f"baselines are not comparable; differing identity: {refused}")
    if set(baseline["metrics"]) != set(candidate["metrics"]):
        raise BaselineError("baselines carry different metric sets")
    rows = []
    for name in sorted(baseline["metrics"]):
        base, cand = baseline["metrics"][name], candidate["metrics"][name]
        if base["better"] != cand["better"] or base["unit"] != cand["unit"]:
            raise BaselineError(f"metric {name!r} changed unit or direction")
        change_pct = 100.0 * (cand["value"] - base["value"]) / base["value"]
        tolerance_pct = max(threshold_pct, 200.0 * math.hypot(_relative_spread(base), _relative_spread(cand)))
        worse = change_pct < 0 if base["better"] == "higher" else change_pct > 0
        verdict = "within-tolerance" if abs(change_pct) <= tolerance_pct else ("regressed" if worse else "improved")
        rows.append({"metric": name, "unit": base["unit"], "baseline": base["value"], "candidate": cand["value"],
                     "change_pct": change_pct, "tolerance_pct": tolerance_pct, "verdict": verdict})
    return {
        "baseline_commit": baseline["source"]["commit"],
        "candidate_commit": candidate["source"]["commit"],
        "grade": "bound" if baseline["grade"] == candidate["grade"] == "bound" else "unbound",
        "allowed_identity_changes": changed,
        "metrics": rows,
        "regressions": [row["metric"] for row in rows if row["verdict"] == "regressed"],
    }


def _print_diff(report: Mapping[str, Any]) -> None:
    print(f"baseline {report['baseline_commit'][:12]} -> candidate {report['candidate_commit'][:12]} "
          f"(grade {report['grade']})")
    if report["allowed_identity_changes"]:
        print(f"declared identity changes: {', '.join(report['allowed_identity_changes'])}")
    for row in report["metrics"]:
        print(f"  {row['metric']:<28} {row['baseline']:>12.4f} -> {row['candidate']:>12.4f} {row['unit']:<6}"
              f" {row['change_pct']:+7.2f}% (tol {row['tolerance_pct']:.2f}%) {row['verdict']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "import"):
        p = sub.add_parser(name)
        p.add_argument("--kind", choices=KINDS, required=True)
        p.add_argument("--build-dir", type=Path, required=True)
        p.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1])
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--input-file", action="append", default=[], metavar="NAME=PATH")
        p.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree; the record is unbound")
        if name == "run":
            p.add_argument("--result-json", type=Path, required=True, help="file the command must create")
            p.add_argument("measure", nargs=argparse.REMAINDER, help="-- COMMAND [ARGS...]")
        else:
            p.add_argument("--input", type=Path, required=True)
    cmp_ = sub.add_parser("diff")
    cmp_.add_argument("--baseline", type=Path, required=True)
    cmp_.add_argument("--candidate", type=Path, required=True)
    cmp_.add_argument("--threshold-pct", type=float, default=2.0)
    cmp_.add_argument("--allow-identity-change", action="append", default=[], choices=sorted(IDENTITY_FIELDS))
    cmp_.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command in ("run", "import"):
            if args.command == "run":
                command = args.measure[1:] if args.measure[:1] == ["--"] else args.measure
                document = run(args.kind, args.build_dir, args.result_json, command, source_dir=args.source_dir,
                               inputs=args.input_file, allow_dirty=args.allow_dirty)
            else:
                document = import_result(args.kind, args.input, args.build_dir, source_dir=args.source_dir,
                                         inputs=args.input_file, allow_dirty=args.allow_dirty)
            args.output.write_text(json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n")
            return 0
        report = diff(load_json(args.baseline), load_json(args.candidate), args.threshold_pct,
                      args.allow_identity_change)
        print(json.dumps(report, indent=1)) if args.json else _print_diff(report)
        return 1 if report["regressions"] else 0
    except (BaselineError, OSError) as error:
        print(f"perf_baseline: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
