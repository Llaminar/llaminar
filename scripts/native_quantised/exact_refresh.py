#!/usr/bin/env python3
"""Collect finite exact overlays without fitting or replacing generic programs.

This entrypoint routes to the existing CPU execution-exact and GPU prefill
transactions. Decode uses the existing native trainers, common strong adapters,
exact emitters and isolated profiler collector. M1 publication and its owning
rebuild precede grouped evidence; a partial stage is never a complete sweep.
No corpus is downloaded, no learner is launched, and no backend is substituted.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "tests/v2/performance/kernels"
sys.path.insert(0, str(KERNELS))

from native_quantised_dispatch.corpus_provenance import (  # noqa: E402
    mapping_digest, require_release_trainer, trainer_provenance,
    _find_cmake_cache, _cmake_cache_value,
)
from native_quantised_dispatch.format_registry import FORMAT_SPECS  # noqa: E402
from native_quantised_dispatch.shape_manifest import load_shape_manifest  # noqa: E402
from native_quantised_dispatch.modern_sweep_scope import modern_sweep_shape_names  # noqa: E402
from native_quantised_dispatch.validation import CANONICAL_VERIFIER_M  # noqa: E402


def parser() -> argparse.ArgumentParser:
    """Declare exact-only controls; old learner options have no executable route."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--backend", choices=("cpu", "cuda", "rocm", "all"), required=True)
    result.add_argument("--profile", choices=("all",), default="all",
                        help="Selected shared production exact inventory; no private model list")
    result.add_argument("--exact-overlays", action="store_true", help="Explicit spelling of the only workflow")
    result.add_argument("--surface", choices=("decode", "dense-prefill", "moe-prefill", "all"), default="all")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--dry-run", action="store_true", help="Pure command plan; no binary/device discovery")
    result.add_argument("--install", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--model-scope", choices=("modern", "all-retained"),
                        help="New jobs default to modern; resume preserves the saved inventory")
    result.add_argument("--reuse-completed-root", type=Path,
                        help="Authenticate original completed CPU owners or GPU decode shapes before collecting missing jobs")
    result.add_argument("--emit-only", action="store_true")
    result.add_argument("--sudo-tool", action="store_true",
                        help="Use existing noninteractive sudo support only for isolated profiler owners")
    result.add_argument("--stage", choices=("decode", "rows", "all", "expert-only"), default="all")
    result.add_argument("--prefill-row-policy", choices=("exact-points", "two-anchor"), default="exact-points")
    result.add_argument("--tournament", choices=("full", "adaptive_v1"), default="full")
    result.add_argument("--expert-model-family")
    result.add_argument("--expert-tp-degree", type=int)
    result.add_argument("--expert-layout", choices=("complete", "intermediate_shard", "down_columns"))
    result.add_argument("--expert-isa", choices=("AVX2:AVX2", "AVX512:AVX2", "AVX512:AVX512"))
    result.add_argument("--expert-purpose", choices=("decode", "mtp_verifier", "prefill"), action="append")
    result.add_argument("--expert-role-format", action="append", default=[])
    result.add_argument("--exclude-expert-role-format", action="append", default=[])
    result.add_argument("--threads", type=int,
                        help="Explicit physical CPU worker budget; required for native CPU collection")
    result.add_argument("--warmups", type=int, default=5)
    result.add_argument("--samples", type=int, default=30)
    result.add_argument("--repeats", type=int, default=8)
    result.add_argument("--cpu-list")
    result.add_argument("--batch-limit", type=int)
    result.add_argument("--cpu-avx2-sweep-bin", type=Path,
        default=ROOT / "build_v2_release_avx2/tests/v2/v2_perf_cpu_native_quantised_gemv")
    result.add_argument("--cpu-avx512-sweep-bin", type=Path,
        default=ROOT / "build_v2_release_avx512/tests/v2/v2_perf_cpu_native_quantised_gemv")
    result.add_argument("--fused-avx2-bin", type=Path,
        default=ROOT / "build_v2_release_avx2/tests/v2/v2_perf_cpu_fused_verifier")
    result.add_argument("--fused-avx512-bin", type=Path,
        default=ROOT / "build_v2_release_avx512/tests/v2/v2_perf_cpu_fused_verifier")
    for backend, binary in (("cuda", "v2_perf_cuda_native_quantised_decode_trainer"),
                            ("rocm", "v2_perf_native_quantised_throughput")):
        result.add_argument(f"--{backend}-bin", type=Path, default=ROOT / "build_v2_release/tests/v2" / binary)
        result.add_argument(f"--{backend}-devices", default="0")
        result.add_argument(f"--{backend}-context", type=Path,
            help="Native device/compiler context JSON: compiler_id,architecture_class,device_name,driver_runtime")
        result.add_argument(f"--{backend}-profiler", type=Path,
            help="Explicit selected ncu/rocprof executable; required for native exact publication")
        result.add_argument(f"--{backend}-dense-prefill-bin", type=Path)
        result.add_argument(f"--{backend}-moe-prefill-bin", type=Path)
    result.add_argument("--cuda-base-policy-json", type=Path,
                        help="Original authenticated policy artifact paired with the retained Auto include")
    return result


def run(command: list[str], *, env: dict[str, str] | None = None,
        log: Path | None = None) -> None:
    """Execute one owned command, keeping diagnostics out of timing labels."""
    if log is None:
        subprocess.run(command, check=True, env=env)
    else:
        with log.open("w") as stream:
            subprocess.run(command, check=True, env=env, stdout=stream, stderr=subprocess.STDOUT)


def module(name: str, *args: object) -> list[str]:
    """Use the existing common collector, never a copied policy implementation."""
    return [sys.executable, "-m", f"native_quantised_dispatch.{name}", *map(str, args)]


def model_scope(args: argparse.Namespace, backend: str) -> str:
    """Keep historical resumed inventories intact; new work selects modern models.

    A wrapper selection receipt owns orchestration scope, independently of the
    lower owners' original producer manifests. Historical plans without that
    receipt retain their original inventory. Scope changes use a new directory;
    this marker never qualifies or relabels numerical/performance evidence.
    """
    saved: set[str] = set()
    root = args.output_dir / backend
    selection = root / "collection-selection.json"
    has_selection = selection.is_file()
    if has_selection:
        if not args.resume:
            raise ValueError("existing exact selection requires --resume")
        payload = json.loads(selection.read_text())
        if payload.get("schema") != "native-quantised-exact-selection-v1":
            raise ValueError("unsupported exact selection receipt")
        saved.add(payload.get("model_scope"))
    if args.resume:
        if backend == "cpu":
            paths = (root / "cpu-exact-plan.json",)
        else:
            paths = tuple((root / "decode").glob("*/collection-contract.json"))
            # Lower prefill owners retain their own producer/plan identity.
            # Only an absent wrapper marker needs historical scope inference.
            if not has_selection:
                paths += tuple(root.glob("*-prefill/*.plan.json"))
                paths += tuple(root.glob("*-prefill/corpus.json"))
        for path in paths:
            if path.is_file():
                saved.add(json.loads(path.read_text()).get("model_scope", "all-retained"))
    if not saved <= {"modern", "all-retained"} or len(saved) > 1:
        raise ValueError("saved exact inventories disagree about model scope")
    if saved and args.model_scope is not None and saved != {args.model_scope}:
        raise ValueError("changing a saved model scope requires a new output directory")
    return args.model_scope or next(iter(saved), "modern")


def write_gpu_selection(args: argparse.Namespace, backend: str, scope: str) -> None:
    """Persist orchestration selection without modifying lower evidence owners."""
    path = args.output_dir / backend / "collection-selection.json"
    expected = {"schema": "native-quantised-exact-selection-v1", "model_scope": scope}
    if path.is_file():
        if json.loads(path.read_text()) != expected:
            raise ValueError("saved exact selection changed")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.inprogress")
    temporary.write_text(json.dumps(expected, indent=2) + "\n")
    temporary.replace(path)


def decode_shape_names(scope: str) -> tuple[str, ...]:
    """Select collection names without altering manifest or installed policy keys."""
    manifest = load_shape_manifest()
    return (modern_sweep_shape_names(manifest) if scope == "modern"
            else manifest.names(production_only=True))


def cpu_command(args: argparse.Namespace) -> list[str]:
    """Delegate M1/rebuild/rows ownership to the canonical CPU transaction."""
    command = module("cpu_exact_sweep", "--output-dir", args.output_dir / "cpu",
        "--avx2-bin", args.cpu_avx2_sweep_bin, "--avx512-bin", args.cpu_avx512_sweep_bin,
        "--fused-avx2-bin", args.fused_avx2_bin, "--fused-avx512-bin", args.fused_avx512_bin,
        "--threads", args.threads if args.threads is not None else "<physical-worker-budget>", "--warmups", args.warmups,
        "--samples", args.samples, "--repeats", args.repeats, "--stage", args.stage)
    for enabled, flag in ((args.dry_run, "--plan-only"), (args.install, "--install"),
                          (args.resume, "--resume"), (args.emit_only, "--emit-only"),
                          (args.sudo_tool, "--sudo-tool")):
        if enabled:
            command.append(flag)
    command += ["--model-scope", model_scope(args, "cpu")]
    if args.reuse_completed_root is not None:
        command += ["--reuse-completed-root", str(args.reuse_completed_root)]
    if args.stage == "expert-only":
        command += ["--prefill-row-policy", args.prefill_row_policy, "--tournament", args.tournament]
        for name in ("expert_model_family", "expert_tp_degree", "expert_layout", "expert_isa"):
            value=getattr(args,name)
            if value is not None:command += ["--"+name.replace("_","-"),str(value)]
        for name in ("expert_purpose", "expert_role_format", "exclude_expert_role_format"):
            for value in getattr(args,name) or ():command += ["--"+name.replace("_","-"),value]
    if args.cpu_list:
        command += ["--cpu-list", args.cpu_list]
    if args.batch_limit is not None:
        command += ["--batch-limit", str(args.batch_limit)]
    return command


def include_path(backend: str) -> Path:
    """Resolve the one installed generated program owned by this backend."""
    return ROOT / f"src/v2/kernels/{backend}/gemm" / (
        "CUDANativeQuantisedGemvDispatchHeuristicGenerated.inc" if backend == "cuda"
        else "ROCmNativeQuantisedDecodeDispatchGenerated.inc")


def prefill_include(backend: str, surface: str) -> Path:
    """Resolve the existing dense/whole-MoE exact overlay authority."""
    vendor = "CUDA" if backend == "cuda" else "ROCm"
    kind = "Dense" if surface == "dense-prefill" else "MoE"
    return ROOT / f"src/v2/kernels/{backend}/gemm/{vendor}{kind}ProductionPrefillOverlayGenerated.inc"


def digest(path: Path) -> str:
    """Hash a named immutable artifact without reading any model payload."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def owning_build_root(binary: Path) -> Path:
    """Reject snapshots/foreign source owners before publishing staged code."""
    cache = _find_cmake_cache(binary.resolve())
    source = _cmake_cache_value(cache, "CMAKE_HOME_DIRECTORY") if cache else None
    if cache is None or source is None or Path(source).resolve() != (ROOT / "src/v2").resolve():
        raise ValueError("exact publication requires the selected trainer's owning build of this source tree")
    return cache.parent


def analyzer(backend: str) -> Path:
    """Return the existing exact emitter/adapter entrypoint."""
    subdir = "cuda/gemm" if backend == "cuda" else "rocm"
    return KERNELS / subdir / f"analyze_{backend}_native_quantised_decode_trainer.py"


def merge_csv(destination: Path, sources: list[Path]) -> None:
    """Stream disjoint native lane outputs and require one exact header."""
    temporary = destination.with_suffix(destination.suffix + ".inprogress")
    header = None
    with temporary.open("w", newline="") as output:
        writer = csv.writer(output)
        for source in sources:
            with source.open(newline="") as stream:
                rows = csv.reader(stream)
                current = next(rows)
                if header is None:
                    header = current
                    writer.writerow(header)
                elif current != header:
                    raise ValueError("native exact lanes disagree on their CSV schema")
                writer.writerows(rows)
    temporary.replace(destination)


def context(args: argparse.Namespace, backend: str, binary: Path) -> dict[str, str]:
    """Bind explicit native device facts to the actual Release executable closure."""
    path = getattr(args, backend + "_context")
    if path is None:
        raise ValueError(f"--{backend}-context is required; native device facts cannot be invented")
    raw = json.loads(path.read_text())
    fields = ("compiler_id", "architecture_class", "device_name", "driver_runtime")
    if not isinstance(raw, dict) or any(not isinstance(raw.get(key), str) or not raw[key].strip() for key in fields):
        raise ValueError("native context requires compiler/device/architecture/runtime facts")
    producer = trainer_provenance(binary, backend)
    require_release_trainer(producer)
    result = {key: raw[key] for key in fields}
    result["git_revision"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    result["build_id"] = mapping_digest(producer)
    # The exact installed include and native kernel bodies own serial arithmetic.
    bodies = sorted((ROOT / f"src/v2/kernels/{backend}/gemm").glob("*NativeQuantised*"))
    result["serial_m1_policy_hash"] = mapping_digest({str(p.relative_to(ROOT)): digest(p)
                                                   for p in bodies if p.is_file()})
    return result


def collect(args: argparse.Namespace, backend: str, binary: Path,
            directory: Path, ms: str, names: tuple[str, ...], devices: tuple[int, ...],
            facts: dict[str, str]) -> tuple[Path, Path]:
    """Reuse native trainer environment contracts with disjoint shape lanes."""
    prefix = "LLAMINAR_" + backend.upper() + "_NATIVE_QUANTISED_DECODE_"
    formats = ",".join(spec.label for spec in FORMAT_SPECS)
    if not devices or not names:
        raise ValueError("native exact lanes require nonempty disjoint production shape slices")
    from native_quantised_dispatch.gpu_completed_sweep import (
        retain_completed_decode_shapes, resume_retained_decode_shapes, verify_retained_decode_sources,
    )
    aggregate, timing = directory / "aggregate.csv", directory / "timing.csv"
    contract = {"build": facts, "shapes": names, "m": ms, "devices": devices,
                "formats": formats, "warmups": args.warmups, "samples": args.samples,
                "repeats": args.repeats}
    if model_scope(args, backend) == "modern":
        contract["model_scope"] = "modern"
    contract = json.loads(json.dumps(contract))
    contract_path, complete = directory / "collection-contract.json", directory / "collection-complete.json"
    if contract_path.exists():
        if not args.resume or json.loads(contract_path.read_text()) != contract:
            raise ValueError("existing exact collection requires --resume and an identical source/build contract")
        if complete.exists():
            saved = json.loads(complete.read_text())
            expected = {"aggregate": digest(aggregate), "timing": digest(timing)}
            reuse_receipt = directory / "collection-reuse.json"
            if reuse_receipt.is_file():
                expected["retained_originals"] = digest(reuse_receipt)
            if saved != expected:
                raise ValueError("retained exact timing evidence changed")
            verify_retained_decode_sources(directory)
            return aggregate, timing
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    retained = []
    missing = names
    if (directory / "collection-reuse.json").is_file():
        retained, missing = resume_retained_decode_shapes(directory, backend, contract, args.reuse_completed_root)
    elif args.reuse_completed_root is not None:
        retained, missing = retain_completed_decode_shapes(args.reuse_completed_root, backend, contract, directory)
    if (directory / "collection-reuse.json").is_file():
        print(f"{backend}: retained {len(names) - len(missing)} original shapes; "
              f"collecting {len(missing)} missing shapes; receipt={directory / 'collection-reuse.json'}")
    active_devices = devices[:min(len(devices), len(missing))]
    def lane(index: int, device: int) -> tuple[Path, Path]:
        """Own one native process and immutable disjoint production-shape slice."""
        part, samples = directory / f"lane{index}.csv", directory / f"lane{index}.timing.csv"
        env = {key: value for key, value in os.environ.items() if not key.startswith(prefix)}
        env.update({prefix + "FORMATS": formats, prefix + "SHAPES": ",".join(missing[index::len(active_devices)]),
            prefix + "M": ms, prefix + "EXECUTION_MODES": "eager,graph_captured",
            prefix + "MAX_CASES": "2147483647", prefix + "WARMUPS": str(args.warmups),
            prefix + "SAMPLES": str(args.samples), prefix + "CSV": str(part),
            prefix + "TIMING_CSV": str(samples)})
        env["CUDA_VISIBLE_DEVICES" if backend == "cuda" else "ROCR_VISIBLE_DEVICES"] = str(device)
        if backend == "cuda":
            env[prefix + "TIMED_REPLAYS"] = str(args.repeats)
        test = "*TrainerCsv_StrongDecode_AllFormats" if backend == "cuda" else "*TrainerCsv_CodebookTagged*"
        run([str(binary), "--gtest_filter=" + test], env=env, log=directory / f"lane{index}.log")
        return part, samples
    outputs = list(retained)
    if active_devices:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_devices)) as workers:
            outputs.extend(workers.map(lambda pair: lane(*pair), enumerate(active_devices)))
    merge_csv(aggregate, [row[0] for row in outputs])
    merge_csv(timing, [row[1] for row in outputs])
    verify_retained_decode_sources(directory)
    completion = {"aggregate": digest(aggregate), "timing": digest(timing)}
    reuse_receipt = directory / "collection-reuse.json"
    if reuse_receipt.is_file():
        completion["retained_originals"] = digest(reuse_receipt)
    complete.write_text(json.dumps(completion, indent=2) + "\n")
    return aggregate, timing


def profile(args: argparse.Namespace, backend: str, binary: Path, directory: Path, common: Path) -> None:
    """Use the existing isolated profiler transaction after unprofiled timing."""
    requests, evidence = directory / "requests.json", directory / "evidence.json"
    run(module("profiler_evidence", "emit-requests", "--observation", common, "--output", requests))
    command = module("profiler_collectors", "--backend", backend, "--binary", binary,
        "--tool", getattr(args, backend + "_profiler"), "--requests", requests,
        "--output", evidence, "--raw-directory", directory / "raw", "--resume",
        "--device-lanes", len(getattr(args, backend + "_devices").split(",")))
    if args.sudo_tool:
        command.append("--sudo-tool")
    run(command)
    run(module("profiler_evidence", "validate-evidence", "--requests", requests, "--evidence", evidence))


def gpu_decode(args: argparse.Namespace, backend: str) -> None:
    """Collect M1, rebuild its exact program, then verify retained grouped rows."""
    binary = getattr(args, backend + "_bin").resolve()
    devices = tuple(int(value) for value in getattr(args, backend + "_devices").split(","))
    if devices != tuple(range(len(devices))) or not devices:
        raise ValueError("the existing profiler lane API requires contiguous GPU ordinals starting at zero")
    if getattr(args, backend + "_profiler") is None:
        raise ValueError(f"--{backend}-profiler is required; exact publication needs isolated kernel evidence")
    if args.stage == "all" and not args.install:
        raise ValueError("complete GPU decode/grouped transaction requires --install for the intermediate M1 rebuild")
    build_root = owning_build_root(binary) if args.install else None
    destination = args.output_dir / backend / "decode"
    destination.mkdir(parents=True, exist_ok=True)
    names = decode_shape_names(model_scope(args, backend))
    base = include_path(backend)
    staged = destination / "m1.inc"
    receipt_path = destination / "m1-stage.json"
    resumed_m1 = (args.resume and receipt_path.is_file() and staged.is_file()
                  and digest(base) == digest(staged))
    if args.stage in {"decode", "all"} and not resumed_m1:
        phase = destination / "m1"
        phase.mkdir(exist_ok=True)
        facts = context(args, backend, binary)
        aggregate, timing = collect(args, backend, binary, phase, "1", names, devices, facts)
        retained_base = destination / "base-before.inc"
        if retained_base.exists() and digest(retained_base) != digest(base):
            raise ValueError("retained generic base changed during exact M1 collection")
        if not retained_base.exists():
            shutil.copyfile(base, retained_base)
        command = [sys.executable, str(analyzer(backend)), "--input", str(aggregate),
            "--timing-sidecar", str(timing), "--output", str(staged), "--exact-only",
            "--common-observations", str(phase / "common.csv"), "--profile", "production",
            "--run-id", f"{backend}-exact-m1", "--model-scope", model_scope(args, backend)]
        if backend == "cuda":
            if args.cuda_base_policy_json is None:
                raise ValueError("CUDA M1 exact retention requires --cuda-base-policy-json")
            command += ["--retain-auto-include", str(retained_base), "--retain-auto-policy-json",
                        str(args.cuda_base_policy_json), "--require-fast-m1-complete"]
        else:
            command += ["--base-include", str(retained_base), "--require-complete"]
        for key, value in facts.items():
            command += ["--" + key.replace("_", "-"), value]
        run(command)
        profile(args, backend, binary, phase, phase / "common.csv")
        receipt_path.write_text(json.dumps({"m1_sha256": digest(staged),
            "retained_base_sha256": digest(retained_base), "old_build_id": facts["build_id"],
            "generic_refit": False, "status": "awaiting_decode_rebuild"}, indent=2) + "\n")
        if not args.install:
            print(f"{backend}: awaiting_decode_rebuild; staged artifact only, no grouped qualification")
            return
        temporary = base.with_suffix(base.suffix + ".inprogress")
        shutil.copyfile(staged, temporary)
        temporary.replace(base)
        run(["cmake", "--build", str(build_root), "--parallel", "--target", binary.name])
    if args.stage in {"rows", "all"}:
        receipt = json.loads(receipt_path.read_text())
        if digest(base) != receipt["m1_sha256"] or digest(staged) != receipt["m1_sha256"]:
            raise ValueError("grouped exact stage changed its installed frozen M1 base")
        facts = context(args, backend, binary)
        if facts["build_id"] == receipt["old_build_id"]:
            raise ValueError("grouped exact stage requires the owning M1 rebuild")
        phase = destination / "rows"
        phase.mkdir(exist_ok=True)
        ms = ",".join(map(str, sorted(CANONICAL_VERIFIER_M)))
        aggregate, timing = collect(args, backend, binary, phase, ms, names, devices, facts)
        output = destination / "complete.inc"
        command = [sys.executable, str(analyzer(backend)), "--input", str(aggregate),
            "--timing-sidecar", str(timing), "--output", str(output), "--exact-only",
            "--common-observations", str(phase / "common.csv"), "--profile", "production",
            "--run-id", f"{backend}-exact-grouped", "--model-scope", model_scope(args, backend), "--retention-receipt", str(phase / "retention.json")]
        if backend == "cuda":
            command += ["--retain-grouped-base-include", str(staged), "--exact-refresh-shapes", *names,
                "--exact-refresh-m", *map(str, sorted(CANONICAL_VERIFIER_M)),
                "--exact-refresh-modes", "eager", "graph_captured"]
        else:
            command += ["--base-include", str(staged), "--verify-retained-exact",
                        "--retained-base-sha256", receipt["m1_sha256"]]
        for key, value in facts.items():
            command += ["--" + key.replace("_", "-"), value]
        run(command)
        profile(args, backend, binary, phase, phase / "common.csv")
        if args.install:
            temporary = base.with_suffix(base.suffix + ".inprogress")
            shutil.copyfile(output, temporary)
            temporary.replace(base)
            run(["cmake", "--build", str(build_root), "--parallel", "--target", binary.name])
        (destination / "complete.json").write_text(json.dumps({"status": "complete",
            "m1_sha256": receipt["m1_sha256"], "final_sha256": digest(output),
            "measured_build_id": facts["build_id"], "new_generic_certification": False,
            "model_and_image_qualification": "pending"}, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Validate a finite transaction before creating output or launching work."""
    cli = parser()
    supplied = sys.argv[1:] if argv is None else argv
    retired = ("--generic", "--freeze-generic", "--certify-generic", "--fit-",
               "--maximum-p95-regret", "--minimum-passing-domain", "--policy-fitter",
               "--skip-scorer-tests", "--corpus-root", "--workspace-root", "--no-lfs-pull")
    if any(token.startswith(retired) for token in supplied):
        cli.error("generic fitting/refitting controls are retired; only finite exact overlays are supported")
    args = cli.parse_args(supplied)
    if min(args.warmups, args.samples, args.repeats) <= 0 or (args.threads is not None and args.threads <= 0):
        cli.error("threads and timing cardinalities must be positive")
    if args.install and (args.warmups < 5 or args.samples < 30):
        cli.error("installable exact evidence requires at least 5 warmups and 30 samples")
    expert_selection=any(getattr(args,name) for name in ("expert_model_family", "expert_tp_degree", "expert_layout", "expert_isa", "expert_purpose", "expert_role_format", "exclude_expert_role_format"))
    if args.stage == "expert-only":
        if args.backend != "cpu":cli.error("expert-only first slice requires --backend cpu")
        if any(getattr(args,name) is None for name in ("expert_model_family", "expert_tp_degree", "expert_layout", "expert_isa")):
            cli.error("expert-only requires explicit catalog model/TP/layout and build:runtime ISA")
    elif expert_selection or args.tournament != "full" or args.prefill_row_policy != "exact-points":
        cli.error("adaptive expert selection belongs to --stage expert-only")
    backends = ("cpu", "cuda", "rocm") if args.backend == "all" else (args.backend,)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(KERNELS)
    os.environ["PYTHONPATH"] = str(KERNELS)
    commands = []
    scopes = {backend: model_scope(args, backend) for backend in backends}
    if args.reuse_completed_root and any(backend != "cpu" for backend in backends) and args.surface != "decode":
        cli.error("GPU completed-root reuse supports --surface decode (including MTP); historical MoE receipts lack actual hardware identity, so resume their original corpus instead")
    for backend in backends:
        if backend == "cpu":
            if not args.dry_run and args.threads is None:
                cli.error("--threads must name the physical CPU worker budget")
            if args.surface not in {"decode", "all"}:
                cli.error("CPU dense-prefill exact installation is not implemented; the expert transaction is separate")
            commands.append(cpu_command(args))
            continue
        if args.surface in {"decode", "all"}:
            if not args.dry_run:
                if args.emit_only:
                    cli.error("GPU emit-only is not supported by the native exact transaction")
                for suffix in ("context", "profiler"):
                    if getattr(args, backend + "_" + suffix) is None:
                        cli.error(f"--{backend}-{suffix} is required before exact native collection")
                if backend == "cuda" and args.stage != "rows" and args.cuda_base_policy_json is None:
                    cli.error("--cuda-base-policy-json is required before retained Auto exact refresh")
                if args.stage == "all" and not args.install:
                    cli.error("complete GPU exact collection requires intermediate --install and owning rebuild")
            commands.append(["GPU_EXACT_TRANSACTION", backend, "M1", "owning_rebuild", "grouped_M2..16,31"])
        for surface in ("dense-prefill", "moe-prefill"):
            if args.surface not in {surface, "all"}:
                continue
            if surface == "dense-prefill" and scopes[backend] == "modern":
                if args.surface == "dense-prefill":
                    cli.error("ordinary dense-prefill collection is parked; modern scope covers decode/MTP and experts")
                continue
            binary = getattr(args, backend + "_" + surface.replace("-", "_") + "_bin")
            if binary is None and not args.dry_run:
                cli.error(f"--{backend}-{surface}-bin is required for the selected surface")
            if args.install and not args.dry_run:
                owning_build_root(binary)
            work = args.output_dir / backend / surface
            sweep = "production_" + surface.replace("-", "_") + "_sweep"
            shared = ["--backend", backend, "--output-dir", work]
            emitter_extra = []
            if surface == "dense-prefill":
                shared += ["--warmup-runs", args.warmups, "--bench-runs", args.samples]
                emitter_extra = ["--warmup-runs", args.warmups, "--bench-runs", args.samples]
            if not args.emit_only:
                commands.append(module(sweep, "run", *shared,
                    "--binary", binary or f"<{backend}-{surface}-Release-trainer>",
                    "--devices", getattr(args, backend + "_devices")))
            commands.append(module(sweep, "combine", *shared))
            output = work / "overlay.inc"
            emitter = "dense_production_overlay" if surface == "dense-prefill" else "moe_production_overlay"
            commands.append(module(emitter, "--backend", backend, "--work-dir", work,
                "--output", output, "--summary-csv", work / "summary.csv",
                "--retain-base-include", prefill_include(backend, surface),
                "--retention-receipt", work / "retention.json", *emitter_extra))
            if args.install:
                commands.append(["INSTALL_EXACT_PREFILL", str(output),
                    str(prefill_include(backend, surface)), str(binary)])
    if args.dry_run:
        print(json.dumps({"workflow": "exact-overlays", "generic_fit": False,
            "production_shapes": {backend: len(decode_shape_names(scopes[backend])) for backend in backends},
            "model_scope": scopes,
            "formats": len(FORMAT_SPECS), "commands": commands}, indent=2))
        return 0
    for backend in backends:
        if backend != "cpu":
            write_gpu_selection(args, backend, scopes[backend])
    for command in commands:
        if command[0] == "GPU_EXACT_TRANSACTION":
            gpu_decode(args, command[1])
        elif command[0] == "INSTALL_EXACT_PREFILL":
            output, base, binary = map(Path, command[1:])
            temporary = base.with_suffix(base.suffix + ".inprogress")
            shutil.copyfile(output, temporary)
            temporary.replace(base)
            run(["cmake", "--build", str(owning_build_root(binary)),
                 "--parallel", "--target", binary.name])
        else:
            run(command, env=env)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
