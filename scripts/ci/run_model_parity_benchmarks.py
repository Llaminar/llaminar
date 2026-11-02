#!/usr/bin/env python3
"""Benchmark canonical E2E candidates on an image or local Release diagnostic.

The canonical matrix exports a separate production-default benchmark policy
for each E2E candidate. HTTP stress arguments never enter timing. This runner
owns only a versioned workload, timing validation and the high-water comparison.
It never commits results, lowers a baseline, or mints a production certificate;
the full pipeline owns those operations after all correctness gates pass.
Full same-image E2E evidence is required unless --diagnostic explicitly selects
a one-off experiment that cannot certify an image.
``--diagnostic-binary`` checks the exact canonical matrix before a new image is
published; it cannot produce or consume an image E2E receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import uuid

import docker_paths
import run_model_parity_e2e as e2e
from model_parity_inventory import map_predictor_source
from production_artifacts import (digest, image_identity, positive, ratchet,
                                  validate_image_e2e, validate_manifest, write_json)

ROOT = Path(__file__).resolve().parents[2]


def hardware_identity() -> dict:
    """Identify compute/topology without keying ratchets on changing clocks."""
    cpu = json.loads(subprocess.check_output(["lscpu", "--json"], text=True))["lscpu"]
    fields = ("Architecture:", "CPU(s):", "Model name:", "Socket(s):",
              "Core(s) per socket:", "Thread(s) per core:", "NUMA node(s):")
    selected = {item["field"]: item["data"] for item in cpu
                if item["field"] in fields or re.fullmatch(r"NUMA node\d+ CPU\(s\):", item["field"])}
    pci = subprocess.check_output(["lspci", "-Dnn"], text=True)
    devices = sorted(line for line in pci.splitlines()
                     if any(kind in line for kind in ("VGA compatible controller", "3D controller", "Display controller")))
    return {"cpu": selected, "accelerators": devices}


def validate_workload(workload: dict) -> str:
    """Keep workload selection explicit and forbid empty/shortened evidence."""
    if workload.get("schema") != 1 or not workload.get("prompt_unit"):
        raise ValueError("invalid production benchmark workload")
    for field in ("prompt_repetitions", "decode_tokens", "warmup_iterations", "measurement_iterations"):
        if type(workload.get(field)) is not int or workload[field] <= 0:
            raise ValueError(f"invalid benchmark workload {field}")
    if workload["measurement_iterations"] < 3:
        raise ValueError("certification needs at least three measured requests")
    return workload["prompt_unit"] * workload["prompt_repetitions"]


def benchmark_profile(record: dict) -> dict:
    """Require producer-owned production defaults; never salvage stale HTTP argv.

    The matrix owns model/topology and selected execution modes. Missing or
    malformed metadata means discovery must be rebuilt, not that the consumer
    may reconstruct policy by stripping a growing blacklist of stress flags.
    """
    if not record.get("e2e"):
        raise ValueError("only E2E-tagged cells may be benchmarked")
    profile = record.get("benchmark")
    if (not isinstance(profile, dict) or type(profile.get("schema")) is not int
            or profile["schema"] != 1 or profile.get("policy") != "production_defaults"
            or type(profile.get("context_length")) is not int or profile["context_length"] <= 0
            or profile["context_length"] != record["e2e"]["context_length"]
            or not isinstance(profile.get("args"), list) or not profile["args"]
            or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in profile["args"])):
        raise ValueError("invalid or missing production-default benchmark policy; rebuild/export canonical metadata")
    return profile


def benchmark_arguments(record: dict, model: str, output: str, workload: dict) -> list[str]:
    """Use the exported benchmark intent verbatim; add only the fixed workload."""
    prompt = validate_workload(workload)
    profile = benchmark_profile(record)
    return ["benchmark", *profile["args"], "-m", model,
            "--context-length", str(profile["context_length"]),
            "--prompt", prompt, "-n", str(workload["decode_tokens"]),
            "--seed", str(workload["seed"]), "--temperature", str(workload["temperature"]),
            "--top-k", str(workload["top_k"]), "--top-p", str(workload["top_p"]),
            "--benchmark-json-output", output]


def validate_measurement(data: dict, workload: dict) -> dict:
    """Use medians of complete unprofiled requests, never rounded console text."""
    prompt = validate_workload(workload).encode()
    if (data.get("schema") != "llaminar.benchmark.v1" or data.get("success") is not True
            or data.get("prefill_success") is not True or data.get("decode_success") is not True):
        raise ValueError("benchmark did not complete production prefill and decode")
    if (data.get("prompt", {}).get("sha256") != hashlib.sha256(prompt).hexdigest()
            or data["prompt"].get("bytes") != len(prompt)):
        raise ValueError("benchmark prompt identity mismatch")
    count = workload["measurement_iterations"]
    rows = data.get("iterations", [])
    if (data.get("measurement_iterations") != count or len(rows) != count
            or data.get("warmup_iterations") != workload["warmup_iterations"]):
        raise ValueError("benchmark omitted warmup or measured requests")
    samples = {"prefill": [], "decode": []}
    token_counts = set()
    for row in rows:
        if row["tokens"]["decode"] != workload["decode_tokens"]:
            raise ValueError("benchmark decode was truncated (including early EOS)")
        token_counts.add(positive(row["tokens"]["prefill"]))
        for phase, metric in (("prefill", "prefill"), ("decode", "decode_after_prefill")):
            samples[phase].append(positive(row["throughput_tokens_per_sec"][metric]))
    if len(token_counts) != 1:
        raise ValueError("benchmark prompt tokenization changed between requests")
    return {"tokens_per_second": {key: statistics.median(values) for key, values in samples.items()},
            "samples": samples, "prefill_tokens": int(token_counts.pop())}


def run_cell(image: str | None, backends: str, directory: Path, model: Path,
             record: dict, workload: dict, *, diagnostic_binary: Path | None = None) -> dict:
    """Run one exact workload, isolating a certified image or a local diagnostic.

    The binary route is explicitly non-certifying. Its process group includes
    self-launched MPI followers so a timeout retires the entire local cell.
    """
    name = "llaminar-benchmark-" + uuid.uuid4().hex
    output = directory / "benchmark.json"
    if diagnostic_binary is not None:
        command = [str(diagnostic_binary), *benchmark_arguments(
            record, str(model), str(output), workload)]
    else:
        if image is None:
            raise ValueError("certifying benchmark cell requires an immutable image")
        command = ["docker", "run", "--rm", "--name", name,
                   *docker_paths.device_args(image, backends),
                   *docker_paths.mounts([(model.parent, str(model.parent), True),
                                         (directory, "/benchmark-output", False)]),
                   "-e", f"LLAMINAR_BENCHMARK_ITERATIONS={workload['measurement_iterations']}",
                   "-e", f"LLAMINAR_BENCHMARK_WARMUP_ITERATIONS={workload['warmup_iterations']}",
                   image, *benchmark_arguments(record, str(model), "/benchmark-output/benchmark.json", workload)]
    write_json(directory / "invocation.json", {"argv": command})
    if diagnostic_binary is not None:
        environment = os.environ.copy()
        environment["LLAMINAR_BENCHMARK_ITERATIONS"] = str(workload["measurement_iterations"])
        environment["LLAMINAR_BENCHMARK_WARMUP_ITERATIONS"] = str(workload["warmup_iterations"])
        with (directory / "benchmark.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            try:
                return_code = process.wait(timeout=e2e.parity.EXACT_CELL_TIMEOUT_SECONDS)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
    else:
        try:
            with (directory / "benchmark.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               timeout=e2e.parity.EXACT_CELL_TIMEOUT_SECONDS)
        finally:
            # A dead docker client does not necessarily retire server/MPI children.
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30, check=False)
    return json.loads(output.read_text())


def declared_panel_eligibility(panel: dict, exact: str, record: dict, workload: dict) -> dict:
    """Require genuine completed numerical acceptance of this identical exported policy/workload."""
    paired = panel.get("paired_workload")
    if (panel.get("mode") not in ("local_paired_canonical_accuracy", "local_fixed_history_diagnostic")
            or panel.get("complete") is not True or panel.get("passed") is not True
            or panel.get("observations_complete") is not True
            or (paired is not None and (paired.get("exact_cell") != exact or paired.get("source_export") != record))):
        raise ValueError("declared benchmark requires green numerical panel for the identical canonical cell")
    profile = record.get("diagnostic_benchmark", {})
    if (profile.get("schema") != 1 or profile.get("policy") != "diagnostic_declared"
            or profile.get("certification_eligible") is not False or not profile.get("args")
            or any(type(arg) is not str or not arg or "\0" in arg for arg in profile["args"])):
        raise ValueError("unsupported declared benchmark policy")
    cells = panel.get("cells", [])
    if not cells or any(cell.get("passed") is not True or cell.get("complete") is not True
            or cell.get("observations_complete") is not True or cell.get("native_retired") is not True
            or cell.get("returncode") != 0 or cell.get("driver_window_clean") is not True
            or cell.get("reset_byte_equal") is not True for cell in cells):
        raise ValueError("numerical panel has incomplete native/driver/cleanup proof")
    validate_workload(workload)
    return profile


def declared_config_identity(selection, observed: dict, parser: Path, environment: dict) -> dict:
    """Use the paired owner's public-parser projection, including observer-owned controls."""
    from types import SimpleNamespace
    return selection.require_public_projection(SimpleNamespace(subprocess=subprocess),
        SimpleNamespace(binary=parser), environment, observed)


def declared_runtime_projection(parser: Path, configuration: dict, environment: dict, path: Path) -> dict:
    """Project a complete resolved document through the existing model-free C++ parser."""
    import model_kld_corpus_execution as owner
    from model_kld_corpus import identity, unchanged
    write_json(path, configuration)
    pins = {str(path): identity(path)}
    arguments = ['--config', str(path)]
    described = json.loads(subprocess.check_output([str(parser), '--describe-config', '--', *arguments],
        env={**environment, 'LLAMINAR_FORCE_CPU_ONLY_STARTUP':'1',
            'LLAMINAR_SKIP_CUDA_STARTUP':'1', 'LLAMINAR_SKIP_ROCM_STARTUP':'1'}, text=True))
    unchanged(pins)
    intent = described.get('qfn_canonical_intent')
    if (described.get('schema') != 'llaminar.model-kld-described-config.v1'
            or described.get('engine_argv') != arguments
            or described.get('runtime_config_digest') != owner.metadata_digest(described.get('config'))
            or not isinstance(intent, dict) or not intent):
        raise ValueError("resolved benchmark document lacks its original typed parser projection")
    return intent


def declared_runtime_configuration(data: dict, canonical: dict, accepted: dict, workload: dict,
        output: Path, parser: Path, environment: dict) -> None:
    """Compare resolved placement with accepted native ownership and other controls with canonical intent."""
    actual = data.get('runtime_configuration')
    if not isinstance(actual, dict) or not isinstance(accepted, dict):
        raise ValueError("missing original accepted or benchmark runtime configuration")
    accepted_intent = declared_runtime_projection(parser, accepted, environment, output/'accepted-native-config.json')
    actual_intent = declared_runtime_projection(parser, actual, environment, output/'benchmark-runtime-config.json')
    if actual_intent != accepted_intent:
        raise ValueError("actual benchmark resolved placement differs from accepted native ownership")
    expected = json.loads(json.dumps(canonical['config']))
    actual_controls = json.loads(json.dumps(actual))
    expected['configuration'].update(benchmark_mode=True, benchmark_json_output_path=str(output/'benchmark.json'),
        prompt=validate_workload(workload), prompt_was_explicitly_provided=True,
        n_predict=workload['decode_tokens'], seed=workload['seed'],
        temperature=workload['temperature'], top_k=workload['top_k'], top_p=workload['top_p'])
    # Partition using the original C++ projection's keys, not a Python list of
    # normalization exceptions. Resolved placement above is exact; all controls
    # outside that projection remain canonical plus the explicit workload.
    for field in accepted_intent:
        expected['configuration'].pop(field, None)
        actual_controls['configuration'].pop(field, None)
    if actual_controls != expected:
        raise ValueError("actual benchmark controls differ from canonical policy/workload")


def declared_source_pins(selection) -> dict:
    """Retain the existing original source stat authority for every shard/predictor, without payload reads."""
    import model_kld_corpus_execution as owner
    pins = {**selection.corpus['inputs'], **selection.pins}
    source = selection.corpus['manifest']['source']
    for descriptor in source['shards'] + ([source['shared_predictor']] if 'shared_predictor' in source else []):
        path = Path(descriptor['path']).resolve(strict=True)
        pin = owner.original_source_identity(path, descriptor)
        if path.name != descriptor['name'] or pin[3] != descriptor['bytes']:
            raise ValueError("original benchmark source extent/name changed")
        pins[str(path)] = pin
    return pins


def run_declared_guardian(command: list[str], environment: dict, output: Path) -> None:
    """Join the original MPI guardian on success and TERM/drain before returning any failure."""
    from mpi_session_retirement import require_retired
    base = output/'mpi-session'
    if command[1:4] != ['--session-base', str(base), '--']:
        raise ValueError("benchmark guardian does not own its exact declared session base")
    base.mkdir(mode=0o700, exist_ok=False)
    process = None
    try:
        with (output/'benchmark.log').open('x') as log:
            process = subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
            code = process.wait(timeout=e2e.parity.EXACT_CELL_TIMEOUT_SECONDS)
        if code:
            raise ValueError("declared benchmark/native session retirement failed")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=36)
        require_retired([output/'mpi-session'],timeout_seconds=0)


def finish_declared_driver(driver, output: Path) -> None:
    """Require the same canonical clean driver window after actual native/session retirement."""
    clean = driver.finish(output/'driver-state.json',output/'canary.driver-diagnostics.json')
    driver.validate_evidence(output)
    if not clean:
        raise ValueError("declared benchmark driver window failed")


def declared_inventory(args, binary: Path):
    """Read the source-bound ALL export from Integration; Release owns execution, never a second matrix."""
    from types import SimpleNamespace
    import model_parity_inventory as inventory
    from model_kld_corpus import identity, unchanged
    if args.manifest is None:
        raise ValueError("declared benchmark requires the original all-cell source manifest")
    path = args.manifest.resolve(strict=True)
    pins = {str(path): identity(path)}
    selected = inventory.discover(SimpleNamespace(build_dir=binary.parent, manifest=path,
        source_revision=args.source_revision, backend='.*', campaign='.*', cell=re.escape(args.declared_cell)),
        inventory.InventoryScope.ALL)
    unchanged(pins)
    if len(selected) != 1 or selected[0][1] != args.declared_cell:
        raise ValueError("declared benchmark needs one exact source-manifest cell")
    return selected[0], pins


def run_declared_diagnostic(args) -> int:
    """Execute the existing benchmark mode under original paired/parser/process/driver owners; never certify."""
    import model_kld_corpus_execution as paired_owner
    from model_kld_corpus import identity, unchanged
    import gpu_driver_diagnostics as driver
    if (not args.diagnostic or args.diagnostic_binary is None or args.image or args.e2e_report
            or not all((args.panel_report, args.corpus_root, args.parser_binary))):
        raise ValueError("declared diagnostics require local Release binary, parser, corpus and green panel; no image/E2E admission")
    binary = args.diagnostic_binary.resolve(strict=True)
    e2e.release_binary_cpu_isa(binary)
    parser = args.parser_binary.resolve(strict=True)
    if parser.parent != binary.parent:
        raise ValueError("benchmark parser must belong to the same original Release build")
    e2e.release_binary_cpu_isa(parser)
    (_campaign, exact, record), inventory_pins = declared_inventory(args,binary)
    selection = paired_owner.PairedCorpusSelection.from_export(exact, record, args.corpus_root, ROOT)
    panel = json.loads(args.panel_report.read_text())
    workload = json.loads(args.workload.read_text())
    profile = declared_panel_eligibility(panel, exact, record, workload)
    cell_directory = args.panel_report.parent / selection.topology['id']
    observed = json.loads((cell_directory/'described-config.json').read_text())
    paired_owner.validate_described_config(selection.topology, observed)
    if panel.get('corpus_manifest_digest') != paired_owner.metadata_digest(selection.corpus['manifest']):
        raise ValueError("numerical panel corpus identity differs from original paired obligation")
    if selection.documents.get('pilot') is not None and panel.get('absolute_pilot_digest') != paired_owner.metadata_digest(selection.documents['pilot']):
        raise ValueError("numerical panel acceptance document differs from original paired obligation")
    controller = json.loads((cell_directory/'controller.json').read_text())
    if (controller.get('complete') is not True or controller.get('native_retired') is not True
            or controller.get('passed') is not True or controller.get('returncode') != 0
            or controller.get('topology_id') != selection.topology['id']):
        raise ValueError("numerical panel original process/config retirement is incomplete")
    driver.validate_evidence(cell_directory)
    native = json.loads((cell_directory/'native-observation.json').read_text())
    cell = next((entry for entry in panel['cells'] if entry.get('topology_id') == selection.topology['id']), None)
    if (cell is None or native.get('complete') is not True
            or native.get('topology_id') != selection.topology['id']
            or native.get('corpus_manifest_digest') != paired_owner.metadata_digest(selection.corpus['manifest'])
            or native.get('source_digest') != paired_owner.metadata_digest(selection.corpus['manifest']['source'])
            or native.get('runtime_config_digest') != selection.topology['runtime_config_digest']
            or native.get('arithmetic') != selection.topology['arithmetic']
            or native.get('native_identity') != cell.get('native_identity')
            or native.get('actual_owner') != cell.get('actual_owner')
            or not all(native.get('proofs',{}).get(name) is True for name in paired_owner.corpus_v2.PROOFS)
            or panel.get('binary',{}).get('sha256') != native.get('native_identity',{}).get('executable_sha256')):
        raise ValueError("numerical panel lacks original native source/config/publication proof")

    if json.loads(selection.paths['benchmark'].read_text()) != workload:
        raise ValueError("benchmark workload differs from the original paired obligation")
    pins = {**declared_source_pins(selection), **inventory_pins}
    for path in (cell_directory/'described-config.json',cell_directory/'controller.json',cell_directory/'native-observation.json'):
        pins[str(path)] = identity(path)

    for path in (binary, parser, args.panel_report.resolve(strict=True), args.workload.resolve(strict=True)):
        pins[str(path)] = identity(path)
    if args.list:
        unchanged(pins)
        print(f"[declared-diagnostic-benchmark] eligible {exact}")
        return 0
    output = args.report.parent / (args.report.stem + '-artifacts')
    output.mkdir(parents=True, exist_ok=False)
    environment = {key: os.environ[key] for key in ('PATH','HOME','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES',
        'CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL','HWLOC_COMPONENTS',
        'OMP_NUM_THREADS','OMP_PLACES','OMP_PROC_BIND','OPENBLAS_NUM_THREADS') if key in os.environ}
    environment.update(LLAMINAR_BENCHMARK_ITERATIONS=str(workload['measurement_iterations']),
        LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=str(workload['warmup_iterations']))
    environment.update({paired_owner.SELECTORS[key]:value for key,value in selection.topology['arithmetic'].items()})
    canonical = declared_config_identity(selection, observed, parser, environment)
    write_json(output/'described-config.json', canonical)
    helper = binary.parent/'llaminar_mpi_session'
    if not helper.is_file() or not os.access(helper,os.X_OK):
        raise ValueError("missing original MPI session retirement helper")
    prompt = validate_workload(workload)
    engine = ['benchmark',*profile['args'],'-m',record['model'],'--context-length',str(profile['context_length']),
        '--prompt',prompt,'-n',str(workload['decode_tokens']),'--seed',str(workload['seed']),
        '--temperature',str(workload['temperature']),'--top-k',str(workload['top_k']),
        '--top-p',str(workload['top_p']),'--benchmark-json-output',str(output/'benchmark.json')]
    command = [str(helper),'--session-base',str(output/'mpi-session'),'--',str(binary),*engine]
    write_json(output/'invocation.json',dict(argv=command,environment=environment,exact_cell=exact))
    result = dict(schema=1, policy='diagnostic_declared', diagnostic=True, certification_eligible=False,
        exact_cell=exact, source_export=record, complete=False, passed=False,
        panel_report=str(args.panel_report), workload=workload)
    armed = False
    try:
        unchanged(pins)
        driver.begin(output/'driver-state.json'); armed=True
        run_declared_guardian(command,environment,output)
        result['native_retired'] = True
        data = json.loads((output/'benchmark.json').read_text())
        measured = validate_measurement(data,workload)
        declared_runtime_configuration(data,canonical,native['actual_owner']['actual_config'],workload,output,parser,environment)
        unchanged(pins)
        result.update(measurement=measured)
        armed=False # The exact driver owner is finished once, including failed validation.
        finish_declared_driver(driver,output)
        result.update(complete=True,passed=True,driver_window_clean=True)
        return 0
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        try:
            if armed:
                result['driver_window_clean'] = bool(driver.finish(output/'driver-state.json',output/'canary.driver-diagnostics.json'))
        except BaseException as error:
            result['driver_cleanup_error'] = repr(error)
            raise
        finally:
            write_json(args.report,result)


def main(argv: list[str] | None = None) -> int:
    """Emit independent cell progress and preserve partial evidence on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--declared-cell", help="exact paired numerical-eligible diagnostic cell; never image certification")
    parser.add_argument("--panel-report", type=Path)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--parser-binary", type=Path)
    execution = parser.add_mutually_exclusive_group(required=True)
    execution.add_argument("--image", help="immutable candidate image for certification")
    execution.add_argument("--diagnostic-binary", type=Path,
                           help="local Release executable; diagnostic only, never an image certificate")
    admission = parser.add_mutually_exclusive_group()
    admission.add_argument("--e2e-report", type=Path,
                           help="passing full E2E report for this exact image and manifest")
    admission.add_argument("--diagnostic", action="store_true",
                           help="explicit one-off experiment; never eligible for certification")
    parser.add_argument("--cell", default=".*", help="diagnostic selector; partial runs cannot certify an image")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "benchmarks/production/high_water.json")
    parser.add_argument("--workload", type=Path, default=ROOT / "benchmarks/production/workload.json")
    parser.add_argument("--model-ramdisk-root", type=Path, default=Path("/mnt/llaminar-production-parity"))
    args = parser.parse_args(argv)
    if args.diagnostic_binary is not None and not args.diagnostic:
        parser.error("--diagnostic-binary requires --diagnostic")
    if args.declared_cell:
        return run_declared_diagnostic(args)
    if args.manifest is None or any((args.panel_report,args.corpus_root,args.parser_binary)):
        parser.error("ordinary benchmark requires manifest; declared inputs require --declared-cell")
    manifest = json.loads(args.manifest.read_text())
    all_cells = validate_manifest(manifest, args.source_revision)
    cells = [row for row in all_cells if re.fullmatch(args.cell, row["case"])]
    if not cells:
        raise ValueError("no eligible benchmark cells selected")
    for row in cells:
        benchmark_profile(row["configuration"])
        print(f"[production-benchmark] selected {row['case']}", flush=True)
    if args.list:
        return 0
    if not args.diagnostic and args.e2e_report is None:
        parser.error("benchmarks require --e2e-report from the full server suite; "
                     "use --diagnostic only for an explicit one-off experiment")
    if args.diagnostic_binary is not None:
        binary = args.diagnostic_binary.resolve(strict=True)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("diagnostic binary must be an executable file")
        # Diagnostic/certifying is report provenance, not an instruction set.
        # Reuse HTTP's Release/ISA authority so native runs compare against the
        # matching image baseline instead of silently creating a new ISA key.
        cpu_isa = e2e.release_binary_cpu_isa(binary)
        runtime = {"id": f"local-release:{binary}",
                   "labels": {"org.llaminar.cpu_isa": cpu_isa.value}}
    else:
        binary = None
        runtime = image_identity(args.image)
        # Explicit diagnostics may compare an older immutable implementation
        # under the current workload policy. Keep both revisions in evidence;
        # this never satisfies same-source/same-image certification admission.
        if (not args.diagnostic and
                runtime["labels"].get("org.opencontainers.image.revision") != args.source_revision):
            raise ValueError("benchmark image has the wrong source revision")
    e2e_digest = None
    if args.e2e_report is not None:
        evidence = json.loads(args.e2e_report.read_text())
        validate_image_e2e(evidence, manifest, runtime["id"])
        e2e_digest = digest(evidence)
    workload = json.loads(args.workload.read_text())
    validate_workload(workload)
    hardware = hardware_identity()
    baseline = json.loads(args.baseline.read_text())
    report = {"schema": 1, "image": runtime["id"], "source_revision": args.source_revision,
              "runtime_source_revision": runtime["labels"].get("org.opencontainers.image.revision"),
              "manifest_digest": digest(manifest), "hardware": hardware,
              "diagnostic": args.diagnostic, "e2e_report_digest": e2e_digest,
              "workload": workload, "complete": False, "passed": False, "cells": []}
    args.report = args.report.resolve()
    run_root = args.report.parent / ("benchmarks-" + str(time.time_ns()))
    run_root.mkdir(parents=True)
    staging = [e2e.parity.CampaignCell(row["campaign"], e2e.parity.CampaignGroup(row["backends"], "ALL"),
                                     model_files=tuple(row["model_files"])) for row in cells]
    started = time.monotonic()
    try:
        with e2e.parity.model_staging_workspace(args.model_ramdisk_root, Path("cache"), None) as workspace:
            staged, _ = e2e.parity.stage_models_in_ramdisk(staging, workspace.models, None,
                                                         persistent=workspace.persistent)
            workspace.protect_published_models()
            paths = {Path(item.source_path).resolve(): workspace.models / item.filename for item in staged}
            for index, row in enumerate(cells, 1):
                print(f"[production-benchmark] {index}/{len(cells)} RUN {row['case']}", flush=True)
                directory = run_root / str(index)
                directory.mkdir()
                config = row["configuration"]
                model = paths[Path(config["model"]).resolve()]
                data = run_cell(runtime["id"] if binary is None else None,
                                row["backends"], directory, model, map_predictor_source(config, lambda path: str(paths[Path(path).resolve()])), workload,
                                diagnostic_binary=binary)
                measured = validate_measurement(data, workload)
                # Preserve the model filename/size, not machine-specific mount text.
                identity = {"case": row["case"], "configuration": {**config, "model": model.name},
                            "model_bytes": model.stat().st_size, "hardware": hardware,
                            "model_files": {Path(name).name: Path(name).stat().st_size for name in row["model_files"]},
                            "cpu_isa": runtime["labels"]["org.llaminar.cpu_isa"],
                            "workload": workload, "prefill_tokens": measured["prefill_tokens"]}
                report["cells"].append({"case": row["case"], "identity": identity,
                                        "artifacts": str(directory), **measured})
                print(f"[production-benchmark] {index}/{len(cells)} PASS {row['case']} "
                      f"prefill={measured['tokens_per_second']['prefill']:.2f} "
                      f"decode={measured['tokens_per_second']['decode']:.2f} tok/s", flush=True)
                write_json(args.report, report)
        proposed, comparisons = ratchet(baseline, report["cells"])
        for comparison in comparisons:
            prior = comparison["high_water"]
            delta = "new" if prior is None else f"{100 * (comparison['current'] / prior - 1):+.2f}%"
            print(f"[production-benchmark] {comparison['status']} {comparison['case']} "
                  f"{comparison['phase']}={comparison['current']:.2f} tok/s delta={delta}", flush=True)
        report.update(complete=not args.diagnostic and len(cells) == len(all_cells), comparisons=comparisons,
                      passed=all(item["passed"] for item in comparisons),
                      proposed_high_water=proposed, baseline_digest=digest(baseline))
        return 0 if report["passed"] else 1
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.report, report)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"[production-benchmark] ERROR: {error}", file=sys.stderr)
        sys.exit(1)
