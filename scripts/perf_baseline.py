#!/usr/bin/env python3
"""Record and compare identity-bound performance baselines.

A baseline document (schema ``llaminar.perf-baseline.v4``) is one measured
result plus the identity needed to decide whether a later result is comparable.

``run`` (Linux only)
    Executes the measuring command itself through ONE admitted executable: the
    command's first word is resolved to a real path inside ``--build-dir``, and
    exactly that resolved path is executed and verified. Before and after the
    run, the executable and every shared library ``ldd`` resolves for it inside
    the build tree are SHA-256 digested, as are the declared input files; any
    change is fatal, and a missing or failing ``ldd`` is fatal. ``ldd`` reports
    the loader's initial resolution only: libraries opened later with
    ``dlopen`` are not observed. The command must exit 0 and must create
    ``--result-json``, which may not exist beforehand. Argv, working directory,
    CPU affinity and the launch environment (``LLAMINAR_*``, ``OMP_*``,
    ``GOMP_*``, ``KMP_*``, ``HSA_*``, ``NCCL_*``, ``RCCL_*``, ``LD_LIBRARY_PATH``,
    ``LD_PRELOAD`` and the GPU visibility variables) are recorded.

    Output roles. The fresh result destination is normalized to ``<result-json>``
    only at DECLARED output positions, never by value or substring: the
    command words named with ``--result-arg-index N`` (0 is the executable),
    the values of options named with ``--result-arg-option OPT`` (``OPT=v`` or
    ``OPT v``), the variables named with ``--result-env VAR``, and, for
    ``llaminar-benchmark``, the typed field
    ``runtime_configuration.configuration.benchmark_json_output_path``. Each
    declared value must denote the result file physically: its ``realpath``
    (symlinks resolved before ``..``, relative to the launch directory) must
    equal the result's; otherwise the run is refused. Every undeclared word or
    variable stays verbatim, even when it equals the result path.
    Declared inputs (``--input-file NAME=PATH``) keep their launch-visible path:
    path-to-target binding and target bytes are checked before and after.
``import``
    Wraps an existing result; there is no launch record. A benchmark's
    ``benchmark_json_output_path`` is replaced by ``<unauthenticated-output>``:
    the field is typed as the producer's output destination, but its value
    cannot be authenticated for an import.

Grades: ``bound`` only for a ``run`` from a clean tree, in a ``Release`` build,
without ``--allow-dirty``; everything else is ``unbound``. ``bound`` is a local
self-attestation of how the record was produced: not tamper-proof, not compiler
provenance (build fields are read from CMake's cache), not image certification.

Input kinds:

``llaminar-benchmark``
    ``llaminar.benchmark.v1`` from ``llaminar2 benchmark --benchmark-json-output``.
    All three success flags must be true and ``--input-file model=PATH`` is
    required. Identity: prompt, token and iteration counts, runtime state and the
    complete ``runtime_configuration``; generated token ids form a separate
    ``output`` identity. Throughput and latency are the metrics.
``kernel-metrics``
    ``{"harness", "parameters", "metrics", "observations"?}``. ``harness`` and
    ``parameters`` are identity; ``metrics`` are compared; ``observations``
    (per-run timings and other measured context) are kept but never compared.

``diff`` validates both documents completely, then compares only when every
identity field matches. A field that is unknown (absent or null) in either
document never matches. ``--allow-identity-change FIELD`` waives exactly one
field. A metric regresses when it moves the wrong way by more than
``max(--threshold-pct, 2 * hypot(spread_a, spread_b))``, spread being the robust
relative spread ``1.4826 * MAD / median`` of its samples. Exit codes: 0 comparable
without regression, 1 regression, 2 refused or invalid input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA = "llaminar.perf-baseline.v4"
KINDS = ("llaminar-benchmark", "kernel-metrics")
GRADES = ("bound", "unbound")
ENV_PREFIXES = ("LLAMINAR_", "OMP_", "GOMP_", "KMP_", "HSA_", "NCCL_", "RCCL_")
ENV_EXACT = ("LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
             "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
RESULT_PLACEHOLDER = "<result-json>"
HEX64 = re.compile(r"[0-9a-f]{64}")
RESULT_PLACEHOLDER_IMPORT = "<unauthenticated-output>"
OUTPUT_FIELD = ("runtime_configuration", "configuration", "benchmark_json_output_path")


class BaselineError(Exception):
    """Invalid input, refused identity or incomparable baselines."""


def _reject_constant(name: str) -> None:
    raise BaselineError(f"non-finite JSON constant {name} is not allowed")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise BaselineError(f"JSON number {text} is out of range")
    return value


def load_json(path: Path) -> Any:
    """Parse strict JSON: NaN/Infinity literals and out-of-range numbers (1e999) are rejected."""

    try:
        return json.loads(path.read_text(), parse_constant=_reject_constant, parse_float=_finite_float)
    except json.JSONDecodeError as error:
        raise BaselineError(f"{path} is not JSON: {error}") from error


def _require_finite_tree(value: Any, where: str = "$") -> None:
    """Reject any non-finite float anywhere in a document, not only in metrics."""

    if isinstance(value, float):
        _require(math.isfinite(value), f"{where} is not finite")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_tree(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{where}[{index}]")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------
# Typed field checks shared by every producer and by validate_document().

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineError(message)


def _text(value: Any, what: str, *, allow_empty: bool = False) -> str:
    _require(isinstance(value, str) and (allow_empty or bool(value)), f"{what} must be a{'' if allow_empty else ' non-empty'} string")
    return value


def _int(value: Any, what: str, minimum: int = 0) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
             f"{what} must be an integer >= {minimum}")
    return value


def _finite(value: Any, what: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
             f"{what} {value!r} is not a finite number")
    return float(value)


def _positive(value: Any, what: str) -> float:
    number = _finite(value, what)
    _require(number > 0, f"{what} {number} must be positive")
    return number


def _hex64(value: Any, what: str) -> str:
    _require(isinstance(value, str) and HEX64.fullmatch(value) is not None, f"{what} must be a SHA-256 hex digest")
    return value


def _object(value: Any, what: str, *, allow_empty: bool = False) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping) and (allow_empty or bool(value)),
             f"{what} must be a{'n' if allow_empty else ' non-empty'} object")
    return value


def _metric(entry: Any, name: str) -> dict[str, Any]:
    """Finite positive value, non-empty unit, direction, and finite positive samples."""

    entry = _object(entry, f"metric {name!r}")
    metric: dict[str, Any] = {"value": _positive(entry.get("value"), f"metric {name!r} value"),
                              "unit": _text(entry.get("unit"), f"metric {name!r} unit"),
                              "better": entry.get("better")}
    _require(metric["better"] in ("higher", "lower"), f"metric {name!r} direction must be 'higher' or 'lower'")
    samples = entry.get("samples")
    if samples is not None:
        _require(isinstance(samples, list) and bool(samples), f"metric {name!r} samples must be a non-empty list")
        metric["samples"] = [_positive(s, f"metric {name!r} sample") for s in samples]
    return metric


def _metrics(raw: Any) -> dict[str, Any]:
    return {str(name): _metric(entry, str(name)) for name, entry in _object(raw, "metrics").items()}


def _check_build(build: Any) -> None:
    build = _object(build, "build")
    for key in ("build_type", "cpu_isa", "cxx_compiler", "cxx_compiler_version"):
        _text(build.get(key), f"build.{key}")
    _text(build.get("cxx_flags"), "build.cxx_flags", allow_empty=True)
    backends = _object(build.get("backends"), "build.backends")
    for key in ("cuda", "rocm"):
        _text(backends.get(key), f"build.backends.{key}")


def _check_host(host: Any) -> None:
    host = _object(host, "host")
    for key in ("machine", "kernel", "cpu_model"):
        _text(host.get(key), f"host.{key}")
    _int(host.get("logical_cpus"), "host.logical_cpus", 1)
    _int(host.get("page_size"), "host.page_size", 1)
    nodes = host.get("numa_nodes")
    _require(isinstance(nodes, list), "host.numa_nodes must be a list")
    for index, node in enumerate(nodes):
        node = _object(node, f"host.numa_nodes[{index}]")
        _int(node.get("node"), f"host.numa_nodes[{index}].node")
        _text(node.get("cpus"), f"host.numa_nodes[{index}].cpus", allow_empty=True)
        _int(node.get("mem_total_kb"), f"host.numa_nodes[{index}].mem_total_kb")


def _check_workload(kind: str, workload: Any) -> None:
    workload = _object(workload, "workload")
    inputs = _object(workload.get("declared_inputs"), "workload.declared_inputs", allow_empty=True)
    for name, binding in inputs.items():
        binding = _object(binding, f"declared input {name!r}")
        _text(binding.get("path"), f"declared input {name!r} path")
        _text(binding.get("target"), f"declared input {name!r} target")
        _hex64(binding.get("sha256"), f"declared input {name!r} sha256")
    if kind == "llaminar-benchmark":
        _require("model" in inputs, "a llaminar-benchmark record requires the declared input 'model'")
        prompt = _object(workload.get("prompt"), "workload.prompt")
        _hex64(prompt.get("sha256"), "workload.prompt.sha256")
        tokens = _object(workload.get("tokens"), "workload.tokens")
        for key in ("prefill", "decode", "decode_after_prefill"):
            _int(tokens.get(key), f"workload.tokens.{key}")
        _int(workload.get("measurement_iterations"), "workload.measurement_iterations", 1)
        _int(workload.get("warmup_iterations"), "workload.warmup_iterations")
        state = _object(workload.get("runtime_state"), "workload.runtime_state")
        for key in ("architecture", "execution_path", "primary_device"):
            _text(state.get(key), f"workload.runtime_state.{key}")
        envelope = _object(workload.get("runtime_configuration"), "workload.runtime_configuration")
        _require(envelope.get("kind") == "llaminar.orchestration-config", "workload.runtime_configuration kind is unsupported")
        _int(envelope.get("schema_version"), "workload.runtime_configuration.schema_version", 1)
        configuration = _object(envelope.get("configuration"), "workload.runtime_configuration.configuration")
        _require(configuration.get(OUTPUT_FIELD[-1]) in (RESULT_PLACEHOLDER, RESULT_PLACEHOLDER_IMPORT),
                 "persisted benchmark_json_output_path must be a normalization placeholder")
    else:
        _text(workload.get("harness"), "workload.harness")
        _object(workload.get("parameters"), "workload.parameters")


def _check_launch(launch: Any) -> None:
    launch = _object(launch, "launch")
    argv = launch.get("argv")
    _require(isinstance(argv, list) and bool(argv) and all(isinstance(a, str) for a in argv), "launch.argv must be a non-empty string list")
    env = _object(launch.get("env"), "launch.env", allow_empty=True)
    _require(all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), "launch.env must map strings")
    _text(launch.get("cwd"), "launch.cwd")
    affinity = launch.get("cpu_affinity")
    _require(isinstance(affinity, list) and bool(affinity)
             and all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in affinity)
             and affinity == sorted(set(affinity)),
             "launch.cpu_affinity must be a non-empty sorted list of unique non-negative CPU ids")
    binaries = _object(launch.get("binaries"), "launch.binaries")
    for name, digest in binaries.items():
        _hex64(digest, f"launch.binaries[{name!r}]")
    _text(launch.get("loader_observation"), "launch.loader_observation")
    _hex64(launch.get("result_sha256"), "launch.result_sha256")
    started, finished = _finite(launch.get("started_unix"), "launch.started_unix"), _finite(launch.get("finished_unix"), "launch.finished_unix")
    _require(finished >= started, "launch.finished_unix precedes started_unix")


def validate_document(document: Any) -> None:
    """Complete typed validation of a baseline document; every entry point uses it."""

    document = _object(document, "baseline document")
    _require_finite_tree(document)
    _require(document.get("schema") == SCHEMA, f"expected schema {SCHEMA}")
    kind, grade = document.get("kind"), document.get("grade")
    _require(kind in KINDS and grade in GRADES, "document kind or grade is invalid")
    source = _object(document.get("source"), "source")
    _require(isinstance(source.get("commit"), str) and bool(re.fullmatch(r"[0-9a-f]{40}", source["commit"])),
             "source.commit must be a full commit id")
    _require(isinstance(source.get("dirty"), bool) and isinstance(source.get("allow_dirty"), bool), "source flags are incomplete")
    _check_build(document.get("build"))
    _check_host(document.get("host"))
    _check_workload(kind, document.get("workload"))
    if document.get("launch") is not None:
        _check_launch(document["launch"])
    if grade == "bound":
        _require(document.get("launch") is not None, "a bound document requires a launch record")
        _require(not source["dirty"] and not source["allow_dirty"], "a bound document requires a clean tree without --allow-dirty")
        _require(document["build"]["build_type"] == "Release", "a bound document requires a Release build")
    if kind == "llaminar-benchmark":
        _hex64(_object(document.get("output"), "output").get("generated_token_ids_sha256"), "output digest")
        expected = RESULT_PLACEHOLDER if document.get("launch") is not None else RESULT_PLACEHOLDER_IMPORT
        _require(document["workload"]["runtime_configuration"]["configuration"][OUTPUT_FIELD[-1]] == expected,
                 "benchmark_json_output_path placeholder does not match how the record was produced")
    _metrics(document.get("metrics"))
    if kind == "kernel-metrics" and document.get("observations") is not None:
        _object(document["observations"], "observations", allow_empty=True)


# ---------------------------------------------------------------------------------------------
# Identity observation.

def _git(source_dir: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(source_dir), *args], capture_output=True, text=True)
    _require(result.returncode == 0, f"git {' '.join(args)} failed in {source_dir}: {result.stderr.strip()}")
    return result.stdout


def source_identity(source_dir: Path, allow_dirty: bool) -> dict[str, Any]:
    """Exact commit; a dirty tree is refused unless --allow-dirty (which forces unbound)."""

    commit = _git(source_dir, "rev-parse", "HEAD").strip()
    dirty = bool(_git(source_dir, "status", "--porcelain", "--untracked-files=no").strip())
    _require(not dirty or allow_dirty, "source tree has uncommitted tracked changes; commit them or pass --allow-dirty")
    identity: dict[str, Any] = {"commit": commit, "dirty": dirty, "allow_dirty": allow_dirty}
    if dirty:
        identity["diff_sha256"] = hashlib.sha256(_git(source_dir, "diff", "HEAD").encode()).hexdigest()
    return identity


def build_identity(build_dir: Path) -> dict[str, Any]:
    """Build envelope as recorded in CMakeCache.txt and CMake's detected-compiler file (not compiler provenance)."""

    cache = build_dir / "CMakeCache.txt"
    _require(cache.is_file(), f"{cache} not found; --build-dir must be a configured CMake build tree")
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
    build = {
        "build_type": build_type,
        "cpu_isa": values.get("LLAMINAR_CPU_ISA") or None,
        "cxx_compiler": values.get("CMAKE_CXX_COMPILER") or None,
        "cxx_compiler_version": values.get("CMAKE_CXX_COMPILER_VERSION") or None,
        "cxx_flags": " ".join(filter(None, (values.get("CMAKE_CXX_FLAGS"),
                                            values.get(f"CMAKE_CXX_FLAGS_{(build_type or '').upper()}")))),
        "backends": {"cuda": values.get("HAVE_CUDA") or None, "rocm": values.get("HAVE_ROCM") or None},
    }
    _check_build(build)
    return build


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
            return result.stdout.strip() or None
    return None


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
            _require(mem_kb is not None, f"{node}/meminfo publishes no MemTotal")
            nodes.append({"node": int(node.name[4:]), "cpus": cpulist, "mem_total_kb": mem_kb})
    return nodes


def host_identity() -> dict[str, Any]:
    """Measuring host; an unobservable CPU model is fatal rather than recorded as unknown."""

    host = {"hostname": platform.node(), "machine": platform.machine(), "kernel": platform.release(),
            "cpu_model": _cpu_model(), "logical_cpus": os.cpu_count(), "page_size": os.sysconf("SC_PAGE_SIZE"),
            "numa_nodes": _numa_nodes()}
    _check_host(host)
    return host


def extract(kind: str, document: Any) -> tuple[dict[str, Any], Any, dict[str, Any], Any]:
    """Return (workload identity without declared inputs, output identity, metrics, observations)."""

    document = _object(document, "input document")
    if kind == "llaminar-benchmark":
        _require(document.get("schema") == "llaminar.benchmark.v1",
                 f"expected schema llaminar.benchmark.v1, got {document.get('schema')!r}")
        for flag in ("success", "prefill_success", "decode_success"):
            _require(document.get(flag) is True, f"benchmark {flag} is {document.get(flag)!r}; failed runs are not baselines")
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
            token_ids = document["generated_token_ids"]
            _require(isinstance(token_ids, list) and all(isinstance(t, int) and not isinstance(t, bool) for t in token_ids),
                     "generated_token_ids must be an integer list")
            output = {"generated_token_ids_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()}
            throughput, latency = document["throughput_tokens_per_sec"], document["decode_latency_ms"]
            metrics = _metrics({
                "prefill_tokens_per_sec": {"value": throughput["prefill"], "unit": "tok/s", "better": "higher"},
                "decode_tokens_per_sec": {"value": throughput["decode"], "unit": "tok/s", "better": "higher"},
                "decode_latency_mean_ms": {"value": latency["mean"], "unit": "ms", "better": "lower"},
                "decode_latency_p50_ms": {"value": latency["p50"], "unit": "ms", "better": "lower"},
            })
        except (KeyError, TypeError) as error:
            raise BaselineError(f"llaminar.benchmark.v1 document lacks {error}") from error
        return workload, output, metrics, None
    if kind == "kernel-metrics":
        observations = document.get("observations")
        if observations is not None:
            _object(observations, "observations", allow_empty=True)
        return ({"harness": document.get("harness"), "parameters": document.get("parameters")}, None,
                _metrics(document.get("metrics")), observations)
    raise BaselineError(f"unknown kind {kind!r}; expected one of {KINDS}")


def _declared_inputs(specs: Sequence[str], kind: str) -> dict[str, str]:
    """Parse NAME=PATH declarations, keeping the launch-visible path; the kind's required inputs are checked here."""

    inputs: dict[str, str] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        _require(bool(sep and name and path), f"--input-file expects NAME=PATH, got {spec!r}")
        _require(name not in inputs, f"--input-file {name!r} declared twice")
        _require(Path(path).is_file(), f"--input-file {name}: {path} is not a file")
        inputs[name] = path
    if kind == "llaminar-benchmark":
        _require("model" in inputs, "llaminar-benchmark requires --input-file model=PATH")
    return inputs


def _bind_inputs(inputs: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Observe each launch-visible path, the target it resolves to, and the target's bytes."""

    bindings = {}
    for name, path in sorted(inputs.items()):
        target = Path(path).resolve()
        _require(target.is_file(), f"--input-file {name}: {path} no longer resolves to a file")
        bindings[name] = {"path": path, "target": str(target), "sha256": _sha256_file(target)}
    return bindings


def _build_tree_closure(executable: Path, build_dir: Path) -> dict[str, str]:
    """Digest the executable and the build-tree libraries of its initial loader resolution.

    A missing ldd, a failing ldd or an unresolved library is fatal: an
    unobserved closure is never reported as empty.
    """

    ldd = shutil.which("ldd")
    _require(ldd is not None, "run requires ldd to observe the executable's library closure")
    listing = subprocess.run([ldd, str(executable)], capture_output=True, text=True)
    _require(listing.returncode == 0, f"ldd failed for {executable}: {(listing.stderr or listing.stdout).strip()}")
    paths = {executable}
    for line in listing.stdout.splitlines():
        _require("not found" not in line, f"unresolved library for {executable}: {line.strip()}")
        parts = line.split("=>")
        target = (parts[1] if len(parts) > 1 else parts[0]).strip().split(" (")[0]
        if target.startswith("/") and Path(target).resolve().is_relative_to(build_dir):
            paths.add(Path(target).resolve())
    return {str(path.relative_to(build_dir)): _sha256_file(path) for path in sorted(paths)}


class ResultDestination:
    """The result file in its raw supplied spelling plus its physical identity."""

    def __init__(self, raw: str) -> None:
        _require(isinstance(raw, str) and bool(raw), "--result-json must be a non-empty path")
        self.raw = raw
        self.physical = os.path.realpath(raw)

    def denotes(self, value: Any) -> bool:
        """Physical identity: realpath resolves symlinks before '..', relative to the launch cwd."""

        return isinstance(value, str) and bool(value) and os.path.realpath(value) == self.physical


def _normalize_argv(arguments: Sequence[str], destination: ResultDestination, indices: Sequence[int],
                    options: Sequence[str]) -> list[str]:
    """Replace only declared output positions; every other word stays verbatim.

    ``arguments`` is the complete command with the executable at index 0.
    """

    normalized = list(arguments)
    declared = set()
    for index in indices:
        _require(0 < index < len(arguments), f"--result-arg-index {index} is outside the command")
        declared.add(index)
    for index, word in enumerate(arguments):
        for option in options:
            if word.startswith(option + "="):
                declared.add((index, option))
            elif word == option:
                _require(index + 1 < len(arguments), f"--result-arg-option {option} has no value")
                declared.add(index + 1)
    for entry in declared:
        if isinstance(entry, tuple):
            index, option = entry
            _require(destination.denotes(arguments[index][len(option) + 1:]),
                     f"{option} does not denote the result file")
            normalized[index] = f"{option}={RESULT_PLACEHOLDER}"
        else:
            _require(destination.denotes(arguments[entry]), f"command word {entry} does not denote the result file")
            normalized[entry] = RESULT_PLACEHOLDER
    for option in options:
        _require(any(word == option or word.startswith(option + "=") for word in arguments),
                 f"--result-arg-option {option} does not occur in the command")
    return normalized


def _launch_environment(result_env: Sequence[str], destination: ResultDestination) -> dict[str, str]:
    """Recorded launch variables; only declared --result-env variables carry the placeholder."""

    env = {key: value for key, value in sorted(os.environ.items()) if key.startswith(ENV_PREFIXES) or key in ENV_EXACT}
    for name in result_env:
        _require(destination.denotes(os.environ.get(name)), f"--result-env {name} does not denote the result file")
        env[name] = RESULT_PLACEHOLDER
    return env


def _benchmark_output_field(raw: Any) -> str:
    """Validate the supported configuration envelope and return its typed output-path string."""

    raw = _object(raw, "benchmark document")
    envelope = _object(raw.get("runtime_configuration"), "runtime_configuration")
    _require(envelope.get("kind") == "llaminar.orchestration-config",
             "runtime_configuration is not a llaminar.orchestration-config document")
    _int(envelope.get("schema_version"), "runtime_configuration.schema_version", 1)
    configuration = _object(envelope.get("configuration"), "runtime_configuration.configuration")
    return _text(configuration.get(OUTPUT_FIELD[-1]), "runtime_configuration.configuration.benchmark_json_output_path")


def _normalize_output_field(kind: str, raw: Any, destination: ResultDestination | None) -> Any:
    """Replace the benchmark's typed output field: authenticated for run, marked unauthenticated for import."""

    if kind != "llaminar-benchmark":
        return raw
    value = _benchmark_output_field(raw)
    if destination is None:
        replacement = RESULT_PLACEHOLDER_IMPORT
    else:
        _require(destination.denotes(value), f"benchmark_json_output_path {value!r} does not denote the result file")
        replacement = RESULT_PLACEHOLDER
    normalized = json.loads(json.dumps(raw))
    normalized["runtime_configuration"]["configuration"][OUTPUT_FIELD[-1]] = replacement
    return normalized


def _assemble(kind: str, raw: Any, *, grade: str, source: dict, build: dict, inputs: Mapping[str, Any],
              launch: dict | None, destination: ResultDestination | None) -> dict[str, Any]:
    workload, output, metrics, observations = extract(kind, _normalize_output_field(kind, raw, destination))
    document = {"schema": SCHEMA, "kind": kind, "grade": grade, "source": source, "build": build,
                "host": host_identity(), "launch": launch, "workload": {**workload, "declared_inputs": dict(inputs)},
                "output": output, "metrics": metrics, "observations": observations, "raw": raw}
    validate_document(document)
    return document


def run(kind: str, build_dir: Path, result_json: str, command: Sequence[str], *, source_dir: Path,
        inputs: Sequence[str] = (), allow_dirty: bool = False, result_env: Sequence[str] = (),
        result_arg_indices: Sequence[int] = (), result_arg_options: Sequence[str] = ()) -> dict[str, Any]:
    """Execute one measurement through one admitted executable and bind the result to it."""

    _require(sys.platform.startswith("linux"), "run is supported on Linux only (it requires ldd)")
    _require(bool(command), "run requires the measuring command after --")
    build_dir = build_dir.resolve()
    found = shutil.which(command[0])
    _require(found is not None, f"executable {command[0]!r} not found")
    executable = Path(found).resolve()
    _require(executable.is_relative_to(build_dir), f"executable {executable} is not inside --build-dir {build_dir}")
    destination = ResultDestination(result_json)
    _require(not os.path.lexists(destination.physical), f"{result_json} already exists; a run must create its result")
    declared = _declared_inputs(inputs, kind)
    source, build = source_identity(source_dir, allow_dirty), build_identity(build_dir)
    env = _launch_environment(result_env, destination)
    argv_identity = _normalize_argv([str(executable.relative_to(build_dir)), *command[1:]], destination,
                                    result_arg_indices, result_arg_options)
    inputs_before, binaries_before = _bind_inputs(declared), _build_tree_closure(executable, build_dir)
    argv = [str(executable), *command[1:]]
    started = time.time()
    completed = subprocess.run(argv)
    finished = time.time()
    _require(completed.returncode == 0, f"measuring command exited {completed.returncode}")
    _require(_build_tree_closure(executable, build_dir) == binaries_before,
             "executable or build-tree library changed during the run")
    _require(_bind_inputs(declared) == inputs_before,
             "a declared input path was retargeted or its bytes changed during the run")
    result_file = Path(destination.physical)
    _require(result_file.is_file(), f"measuring command did not create {result_json}")
    grade = "bound" if build["build_type"] == "Release" and not source["dirty"] and not allow_dirty else "unbound"
    launch = {
        "argv": argv_identity,
        "env": env, "cwd": os.getcwd(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "binaries": binaries_before,
        "loader_observation": "ldd initial resolution; dlopen-loaded libraries not observed",
        "started_unix": started, "finished_unix": finished, "result_sha256": _sha256_file(result_file),
    }
    return _assemble(kind, load_json(result_file), grade=grade, source=source, build=build, inputs=inputs_before,
                     launch=launch, destination=destination)


def import_result(kind: str, input_path: Path, build_dir: Path, *, source_dir: Path, inputs: Sequence[str] = (),
                  allow_dirty: bool = False) -> dict[str, Any]:
    """Wrap an existing result; nothing binds it to this build or host, so it is always unbound."""

    declared = _declared_inputs(inputs, kind)
    return _assemble(kind, load_json(input_path), grade="unbound", source=source_identity(source_dir, allow_dirty),
                     build=build_identity(build_dir), inputs=_bind_inputs(declared), launch=None, destination=None)


# ---------------------------------------------------------------------------------------------
# Comparison.

def _launch_field(key: str) -> Callable[[Mapping[str, Any]], Any]:
    return lambda b: (b.get("launch") or {}).get(key)


IDENTITY_FIELDS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "kind": lambda b: b["kind"],
    "workload": lambda b: {k: v for k, v in b["workload"].items() if k != "declared_inputs"},
    "declared_inputs": lambda b: {name: binding["sha256"] for name, binding in b["workload"]["declared_inputs"].items()},
    "output": lambda b: b.get("output") if b["kind"] == "llaminar-benchmark" else "n/a",
    "argv": _launch_field("argv"),
    "launch_env": _launch_field("env"),
    "cwd": _launch_field("cwd"),
    "cpu_affinity": _launch_field("cpu_affinity"),
    "build_type": lambda b: b["build"]["build_type"],
    "cpu_isa": lambda b: b["build"]["cpu_isa"],
    "compiler": lambda b: [b["build"]["cxx_compiler"], b["build"]["cxx_compiler_version"]],
    "cxx_flags": lambda b: b["build"]["cxx_flags"],
    "backends": lambda b: b["build"]["backends"],
    "host_cpu": lambda b: b["host"]["cpu_model"],
    "host_topology": lambda b: [b["host"]["logical_cpus"], b["host"]["numa_nodes"]],
    "host_page_size": lambda b: b["host"]["page_size"],
    "host_kernel": lambda b: b["host"]["kernel"],
}


_HALF_MAX = sys.float_info.max / 2


def _median(values: Sequence[float]) -> float:
    """Median of non-negative values with a correctly rounded even-count midpoint.

    Range split: while ``high <= max/2`` the sum ``low + high`` cannot overflow
    and ``(low + high) / 2`` is correctly rounded (halving is exact for normal
    results; in the subnormal range the sum is exact and halving rounds once,
    ties to even). Only at the large end (``high > max/2``) is
    ``low/2 + high/2`` used: ``high/2`` is exact there, and ``low/2`` is exact
    unless ``low`` is subnormal, in which case it may round to 0. That loss is
    harmless: such a ``low`` is far below half an ulp of ``high/2``, so the
    correctly rounded midpoint is ``high/2`` either way, and otherwise the sum
    of two exact halves rounds once. No branch adds an error term that could
    invent spread for constant or near-constant samples.
    """

    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    low, high = ordered[middle - 1], ordered[middle]
    if high <= _HALF_MAX:
        return (low + high) / 2
    return low / 2 + high / 2


def _derived(value: float, what: str) -> float:
    """Never publish an unrepresentable derived value."""

    _require(math.isfinite(value), f"{what} is not representable for these inputs")
    return value


def _relative_spread(metric: Mapping[str, Any], name: str) -> float:
    """Robust relative spread 1.4826 * MAD / median, computed on ratios to the median (range-safe)."""

    samples = metric.get("samples") or []
    if len(samples) < 2:
        return 0.0
    median = _median(samples)
    _require(math.isfinite(median) and median > 0, f"metric {name!r} sample median is not positive and finite")
    ratios = [_derived(sample / median, f"metric {name!r} sample ratio") for sample in samples]
    return _derived(1.4826 * _median([abs(ratio - 1.0) for ratio in ratios]), f"metric {name!r} spread")


def diff(baseline: Any, candidate: Any, threshold_pct: float, allowed_changes: Sequence[str] = ()) -> dict[str, Any]:
    """Compare two validated baselines; raise BaselineError when invalid or not comparable."""

    validate_document(baseline)
    validate_document(candidate)
    threshold_pct = _finite(threshold_pct, "--threshold-pct")
    _require(threshold_pct >= 0, "--threshold-pct must be non-negative")
    unknown_fields = set(allowed_changes) - set(IDENTITY_FIELDS)
    _require(not unknown_fields, f"unknown identity fields {sorted(unknown_fields)}; choose from {sorted(IDENTITY_FIELDS)}")
    differing, unknown = [], []
    for name, get in IDENTITY_FIELDS.items():
        a, b = get(baseline), get(candidate)
        if a is None or b is None:
            unknown.append(name)
        elif a != b:
            differing.append(name)
    refused = [name for name in differing + unknown if name not in allowed_changes]
    _require(not refused, f"baselines are not comparable; differing identity: {sorted(n for n in refused if n in differing)}"
                          f", unknown identity: {sorted(n for n in refused if n in unknown)}")
    _require(set(baseline["metrics"]) == set(candidate["metrics"]), "baselines carry different metric sets")
    rows = []
    for name in sorted(baseline["metrics"]):
        base, cand = baseline["metrics"][name], candidate["metrics"][name]
        _require(base["better"] == cand["better"] and base["unit"] == cand["unit"], f"metric {name!r} changed unit or direction")
        # Divide before subtracting/multiplying: the ratio of two positive finite values
        # stays representable where their difference may not.
        ratio = _derived(cand["value"] / base["value"], f"metric {name!r} ratio")
        change_pct = _derived((ratio - 1.0) * 100.0, f"metric {name!r} change")
        tolerance_pct = _derived(max(threshold_pct, 200.0 * math.hypot(_relative_spread(base, name),
                                                                        _relative_spread(cand, name))),
                                 f"metric {name!r} tolerance")
        worse = change_pct < 0 if base["better"] == "higher" else change_pct > 0
        verdict = "within-tolerance" if abs(change_pct) <= tolerance_pct else ("regressed" if worse else "improved")
        rows.append({"metric": name, "unit": base["unit"], "baseline": base["value"], "candidate": cand["value"],
                     "change_pct": change_pct, "tolerance_pct": tolerance_pct, "verdict": verdict})
    return {"baseline_commit": baseline["source"]["commit"], "candidate_commit": candidate["source"]["commit"],
            "grade": "bound" if baseline["grade"] == candidate["grade"] == "bound" else "unbound",
            "waived_identity": sorted(n for n in differing + unknown if n in allowed_changes),
            "metrics": rows, "regressions": [row["metric"] for row in rows if row["verdict"] == "regressed"]}


def _print_diff(report: Mapping[str, Any]) -> None:
    print(f"baseline {report['baseline_commit'][:12]} -> candidate {report['candidate_commit'][:12]} "
          f"(grade {report['grade']})")
    if report["waived_identity"]:
        print(f"waived identity fields: {', '.join(report['waived_identity'])}")
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
        p.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree; the record is always unbound")
        if name == "run":
            p.add_argument("--result-json", required=True, help="file the command must create (spelling preserved)")
            p.add_argument("--result-arg-index", action="append", default=[], type=int, metavar="N",
                           help="command word N (0 = executable) is the result path")
            p.add_argument("--result-arg-option", action="append", default=[], metavar="OPT",
                           help="the value of option OPT (OPT=v or OPT v) is the result path")
            p.add_argument("--result-env", action="append", default=[], metavar="VAR",
                           help="environment variable that carries the result path (typed output role)")
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
                               inputs=args.input_file, allow_dirty=args.allow_dirty, result_env=args.result_env,
                               result_arg_indices=args.result_arg_index, result_arg_options=args.result_arg_option)
            else:
                document = import_result(args.kind, args.input, args.build_dir, source_dir=args.source_dir,
                                         inputs=args.input_file, allow_dirty=args.allow_dirty)
            args.output.write_text(json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n")
            return 0
        report = diff(load_json(args.baseline), load_json(args.candidate), args.threshold_pct,
                      args.allow_identity_change)
        print(json.dumps(report, indent=1, allow_nan=False)) if args.json else _print_diff(report)
        return 1 if report["regressions"] else 0
    except (BaselineError, OSError, ValueError) as error:
        print(f"perf_baseline: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
