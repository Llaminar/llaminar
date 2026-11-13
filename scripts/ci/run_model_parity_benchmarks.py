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
published; it cannot produce or consume an image E2E receipt. Explicit declared
focused panels retain their diagnostic prerequisite scope and reuse the original
complete-row accuracy evaluator; they never become image/canonical acceptance.
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


def declared_panel_eligibility(panel: dict, exact: str, record: dict, workload: dict, *,
        allow_focused_diagnostic: bool = False) -> dict:
    """Require genuine completed numerical acceptance of this identical exported policy/workload."""
    paired = panel.get("paired_workload")
    focused = panel.get("mode") == "local_paired_focused_diagnostic"
    if focused:
        # This explicit diagnostic receipt never becomes a canonical green.
        # Its original row evaluator is replayed below after source admission.
        prerequisites = panel.get("prerequisites", {})
        declaration = record.get("paired_workload")
        accuracy = declaration.get("accuracy", {}) if isinstance(declaration, dict) else {}
        policy = accuracy.get("policy") if isinstance(accuracy, dict) else None
        verdict = "REQUIRED_GROUPS_GREEN" if policy == "initial_absolute_pilot" else "GREEN"
        if (not allow_focused_diagnostic or policy not in ("initial_absolute_pilot", "calibrated_revision")
                or panel.get("schema") != 2 or panel.get("status") != "DIAGNOSTIC_PANEL_COMPLETE"
                or panel.get("accuracy_verdict") != verdict or panel.get("verdict") != "DIAGNOSTIC_" + verdict
                or panel.get("diagnostic_passed") is not True or "passed" not in panel or panel["passed"] is not None
                or panel.get("certification_eligible") is not False or panel.get("full_preflight") is not False
                or not isinstance(prerequisites, dict) or prerequisites.get("available") is not True
                or prerequisites.get("mode") != "focused_DeviceFree_diagnostic"
                or prerequisites.get("full_preflight") is not False
                or prerequisites.get("native_backend_regressions") != "not_covered_by_DeviceFree_selection"
                or not isinstance(paired, dict) or not isinstance(panel.get("cells"), list) or len(panel["cells"]) != 1):
            raise ValueError("focused numerical panel is not an explicit noncertifying green diagnostic")
        accepted = True
    else:
        accepted = (panel.get("mode") in ("local_paired_canonical_accuracy", "local_fixed_history_diagnostic")
            and panel.get("passed") is True and "diagnostic_passed" not in panel
            and not str(panel.get("status", "")).startswith("DIAGNOSTIC_")
            and not str(panel.get("verdict", "")).startswith("DIAGNOSTIC_"))
    if (not accepted or panel.get("complete") is not True
            or panel.get("observations_complete") is not True
            or (paired is not None and (paired.get("exact_cell") != exact or paired.get("source_export") != record))):
        raise ValueError("declared benchmark requires green numerical panel for the identical canonical cell")
    from model_kld_corpus_execution import declared_benchmark_profile
    profile = declared_benchmark_profile(record)
    cells = panel.get("cells", [])
    if not cells or any(cell.get("passed") is not True or cell.get("complete") is not True
            or cell.get("observations_complete") is not True or cell.get("native_retired") is not True
            or cell.get("returncode") != 0 or cell.get("driver_window_clean") is not True
            or cell.get("reset_byte_equal") is not True for cell in cells):
        raise ValueError("numerical panel has incomplete native/driver/cleanup proof")
    validate_workload(workload)
    return profile


def declared_focused_accuracy_scope(panel: dict, selection, cell: dict) -> dict:
    """Replay the original row/threshold evaluator; preserve focused prerequisite limits.

    This reads scored metadata only. It never reads logits, changes an input
    verdict, invents calibration or runs prerequisites to repair their scope.
    """
    import model_kld_corpus_execution as owner
    documents = selection.documents
    summary = owner.corpus_v2.summarize(selection.corpus['manifest'], cell.get('metrics'),
        documents.get('baseline'), documents.get('approval'),
        calibration=documents.get('calibration'), absolute_pilot=documents.get('pilot'))
    if (summary.get('passed') is not True or summary != cell.get('numerical_summary')
            or summary.get('native_identity') != cell.get('native_identity')
            or panel.get('calibrated') is not summary.get('calibrated')
            or panel.get('scope') != summary.get('scope')
            or (documents.get('pilot') is not None and panel.get('whole_corpus_accepted') is not False)):
        raise ValueError("focused numerical panel differs from original complete row/policy evaluation")
    return dict(mode=panel['mode'], status=panel['status'], verdict=panel['verdict'],
        accuracy_verdict=panel['accuracy_verdict'], diagnostic_passed=True,
        scope=summary['scope'], calibrated=summary['calibrated'],
        full_preflight=False, certification_eligible=False,
        prerequisites=panel['prerequisites'])


def declared_config_identity(selection, observed: dict, parser: Path, environment: dict) -> dict:
    """Use the paired owner's public-parser projection, including observer-owned controls."""
    from types import SimpleNamespace
    canonical = selection.require_public_projection(SimpleNamespace(subprocess=subprocess),
        SimpleNamespace(binary=parser), environment, observed)
    # The shared QFN projection intentionally excludes observer prefix controls.
    # Benchmark startup still must retain this original source policy: otherwise
    # a disabled-prefix cell can open the caller's persistent archive by default.
    if (canonical['config']['configuration'].get('prefix_cache') !=
            selection.topology['expected_config']['configuration'].get('prefix_cache')):
        raise ValueError("Declared benchmark changed original source prefix-cache policy")
    return canonical


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


def declared_inventory(args, binary: Path, *, scope=None):
    """Read an explicit source inventory; canonical scope remains ALL by default."""
    from types import SimpleNamespace
    import model_parity_inventory as inventory
    from model_kld_corpus import identity, unchanged
    if args.manifest is None:
        raise ValueError("declared benchmark requires the original all-cell source manifest")
    path = args.manifest.resolve(strict=True)
    pins = {str(path): identity(path)}
    selected = inventory.discover(SimpleNamespace(build_dir=binary.parent, manifest=path,
        source_revision=args.source_revision, backend='.*', campaign='.*', cell=re.escape(args.declared_cell)),
        inventory.InventoryScope.ALL if scope is None else scope)
    unchanged(pins)
    if len(selected) != 1 or selected[0][1] != args.declared_cell:
        raise ValueError("declared benchmark needs one exact source-manifest cell")
    return selected[0], pins


# One source-bound, passive policy for every declared diagnostic candidate.
DECLARED_PASSIVE_POLICY = 'passive_lookup_setup_terminal_v1'
DECLARED_PASSIVE_FILTERS = ('immutable_lookup', 'ngram_lookup',
    'immutable_lookup_setup.admitted_owner', 'moe_placement.routed_expert_weight_selection')


def declared_environment(caller: dict, workload: dict, arithmetic: dict, output: Path) -> dict:
    """Bind passive collection before startup; unrelated caller profiling/loader controls never survive."""
    import model_kld_corpus_execution as owner
    environment = {key: caller[key] for key in ('PATH','HOME','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES',
        'CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL','HWLOC_COMPONENTS',
        'OMP_NUM_THREADS','OMP_PLACES','OMP_PROC_BIND','OPENBLAS_NUM_THREADS') if key in caller}
    environment.update(LLAMINAR_BENCHMARK_ITERATIONS=str(workload['measurement_iterations']),
        LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=str(workload['warmup_iterations']),
        LLAMINAR_PERF_STATS_JSON=str(output/'passive-perfstats-rank-{rank}.json'),
        LLAMINAR_PERF_STATS_FILTER=','.join(DECLARED_PASSIVE_FILTERS),
        LLAMINAR_PERF_STATS_CPU_STAGE_TIMING='0', LLAMINAR_PERF_STATS_GPU_STAGE_TIMING='0',
        LLAMINAR_GPU_STAGE_TIMING='0', LLAMINAR_GPU_STAGE_TIMING_DETAIL='0')
    environment.update({owner.SELECTORS[key]:value for key,value in arithmetic.items()})
    return environment


def _passive_uint(value, name: str) -> int:
    """Require exact unsigned metadata/counters, never bools, lossy floats or negative observations."""
    if isinstance(value, str):
        if not re.fullmatch(r'0|[1-9][0-9]*', value):
            raise ValueError(f'invalid passive {name}')
        return int(value)
    if isinstance(value, bool) or not isinstance(value, (int,float)) or value < 0 or int(value) != value:
        raise ValueError(f'invalid passive {name}')
    return int(value)


def _passive_device(value: str) -> str:
    """Normalize only the original DeviceId backend spelling, preserving physical ordinal."""
    if str(value).upper()=='CPU':return 'CPU:0'
    match = re.fullmatch(r'(CPU|CUDA|ROCM):([0-9]+)', str(value).upper())
    if match is None or match[1]=='CPU' and int(match[2])!=0:
        raise ValueError('invalid passive physical device')
    return f'{match[1]}:{int(match[2])}'


def _expert_member_device(member: dict) -> str:
    """Project a topology member to the native expert publisher's DeviceId.

    CPU topology ordinals select sockets, while their rank-local DeviceId is
    CPU. The existing global participant identity preserves each distinct
    expert bank; GPU ordinals remain physical device identities unchanged.
    """
    ordinal = _passive_uint(member['device_ordinal'], 'expert member ordinal')
    return _passive_device('CPU' if member['device_type'] == 'CPU'
                           else f"{member['device_type']}:{ordinal}")


def _declared_continuation(config: dict) -> list[tuple[int,str]]:
    """Project original serialized continuation membership; do not create a topology registry."""
    plan = config.get('moe_routed_expert_plan') or {}
    name = plan.get('continuation_domain')
    domains = [row for row in plan.get('dense_domains',[]) if row.get('name') == name]
    if len(domains) != 1:
        raise ValueError('passive adaptive evidence lacks original continuation domain')
    domain = domains[0]
    rank = _passive_uint(domain.get('owner_rank'),'continuation owner rank')
    result = [(rank,_passive_device(f"{p['device_type']}:{p['device_ordinal']}")) for p in domain.get('participants',[])]
    if not result or len(set(result)) != len(result) or any(device.startswith('CPU:') for _,device in result):
        raise ValueError('passive adaptive evidence has incomplete/foreign continuation membership')
    return result


def _validate_declared_passive_evidence(data: dict) -> dict:
    """Authenticate original setup partitions and measured terminal schemas, not locality or disk activity.

    The source-bound BenchmarkRunner's pre-measurement reset owns scope. Counter
    totals alone cannot detect same-key warmup contamination; no guessed token or
    iteration multiplier is used as a replacement for that original owner.
    """
    stats = data.get('perf_stats')
    if not isinstance(stats,dict) or stats.get('schema') != 'llaminar.perf_stats.v1' or stats.get('enabled') is not True:
        raise ValueError('declared passive collection is missing or disabled')
    records = stats.get('records')
    if not isinstance(records,list):
        raise ValueError('declared passive records are incomplete')
    config = (data.get('runtime_configuration') or {}).get('configuration')
    if not isinstance(config,dict):
        raise ValueError('declared passive evidence lacks original resolved configuration')
    policy = config.get('ngram_table_policy') or {}
    adaptive = policy.get('storage') == 'AdaptiveVRAMRAMSSD'
    observed_fixed = 'observed_pipeline_binding' in data
    expert = []
    initial_banks = []
    setup = []
    terminal = {}
    submissions = []
    resets = {}
    local_setup = []
    local_terminal = []
    for record in records:
        if not isinstance(record,dict) or record.get('kind') != 'counter':
            raise ValueError('passive evidence contains a non-counter observation')
        domain,name,phase = record.get('domain'),record.get('name'),record.get('phase')
        if domain not in ('immutable_lookup','ngram_lookup','immutable_lookup_setup','moe_placement'):
            raise ValueError('passive evidence includes an unrequested instrumentation family')
        count = _passive_uint(record.get('count'),'record count')
        value = _passive_uint(record.get('value'),'record value')
        if not count or any(record.get(k,0) != 0 for k in ('total_ns','min_ns','max_ns')):
            raise ValueError('passive evidence includes clocks or an empty observation')
        if domain=='immutable_lookup' and (name=='failures' or str(name).endswith('_failures') or name=='adaptive_failed_frames') and value:
            raise ValueError('passive lookup reports failed work')
        device = _passive_device(record.get('device'))
        tags = record.get('tags')
        if not isinstance(tags,dict):
            raise ValueError('passive evidence lacks original tags')
        if domain == 'moe_placement':
            if name != 'routed_expert_weight_selection' or phase not in ('load','model_setup'):
                raise ValueError('passive expert evidence is not original immutable setup')
            if tags.get('owner_order') != {'Ordinal':'ordinal','Random':'random'}.get(config.get('routed_expert_owner_order')) or tags.get('selection_layout') not in ('contiguous','noncontiguous'):
                raise ValueError('passive expert selection policy differs from original config')
            layer = _passive_uint(tags.get('layer'),'expert layer')
            participant = _passive_uint(tags.get('participant'),'expert participant')
            if value % count:
                raise ValueError('passive expert selection mixed incompatible setup populations')
            expert.append(dict(device=device,layer=layer,participant=participant,owner_order=tags['owner_order'],
                selection_layout=tags['selection_layout'],observations=count,selected_experts_per_observation=value//count))
            if phase=='model_setup':
                required={'layer','owner_order','participant','tier','tier_participant_count','selection_layout',
                          'frozen_owner_map_match','publication','layer_role'}
                if (set(tags)!=required or tags.get('publication')!='initial_epoch_bank' or
                        tags.get('frozen_owner_map_match')!='true' or
                        tags.get('layer_role') not in ('main_transformer','mtp_predictor') or count!=1):
                    raise ValueError('passive initial expert bank publication is not frozen original setup')
                initial_banks.append(dict(device=device,layer=layer,participant=participant,
                    tier=_passive_uint(tags['tier'],'initial expert tier'),
                    tier_participant_count=_passive_uint(tags['tier_participant_count'],'initial tier participants'),
                    layer_role=tags['layer_role'],selected=value//count))

        elif domain == 'immutable_lookup_setup':
            if name != 'admitted_owner' or phase != 'setup' or count != 1 or value != 1:
                raise ValueError('passive admitted setup is missing, repeated or not immutable')
            if tags.get('role') in ('local_gpu_cache','local_ready'):
                local_setup.append((device,tags))
            else:
                setup.append((device,tags))
        elif domain == 'ngram_lookup':
            if str(name).startswith('local_'):
                local_terminal.append(record)
                continue
            if phase != 'hosted_transaction' or name not in ('retained_source_native_invocations','retained_remote_native_enqueues') or tags != {'policy':'retained_hosted'}:
                raise ValueError('passive native submission evidence is not the original submission schema')
            # These are submission observations, never a native payload byte oracle.
            submissions.append(dict(name=name,device=device,count=count,value=value,tags=tags))
        elif observed_fixed and domain == 'immutable_lookup' and str(name).startswith('fixed_head_pipeline_'):
            # The shared fixed pipeline validator below compares every actual
            # measured terminal field with the native byte/receipt proof.
            from model_parity_inventory import FIXED_NATIVE_PIPELINE_METRICS
            if name not in FIXED_NATIVE_PIPELINE_METRICS:
                raise ValueError('fixed pipeline measured evidence contains a nonterminal field')
        elif name in ('reset_terminal_closes','reset_terminal_waits'):
            if phase!='request_reset' or tags or value!=count:
                raise ValueError('passive reset observation is not the original terminal infrastructure')
            group=resets.setdefault(device,{})
            if name in group:raise ValueError('duplicate passive reset observation')
            group[name]=(count,value)
        else:
            if phase != 'generation_terminal' and not (not adaptive and phase=='lookup_completion'):
                raise ValueError('passive lookup evidence includes warmup/nonterminal scope')
            key=(device,tuple(sorted(tags.items())))
            group=terminal.setdefault(key,{})
            if name in group:
                raise ValueError('duplicate passive terminal counter')
            group[name]=(count,value)
    for group in resets.values():
        if set(group)!= {'reset_terminal_closes','reset_terminal_waits'} or len(set(group.values()))!=1:
            raise ValueError('passive reset terminal close/wait proof is incomplete')
    for device in {r['device'] for r in submissions}:
        group=[r for r in submissions if r['device']==device]
        if len(group)!=2 or len({r['name'] for r in group})!=2 or len({r['count'] for r in group})!=1:
            raise ValueError('passive native submission tuple is incomplete')
    if config.get('moe_routed_expert_plan') and not expert:
        raise ValueError('missing original routed expert selection setup')
    if len({(x['device'],x['layer'],x['participant']) for x in expert}) != len(expert):
        raise ValueError('duplicate passive expert setup identities')
    plan=config.get('moe_routed_expert_plan') or {}
    if plan:
        tiers=plan.get('routed_tiers')
        if not isinstance(tiers,list) or not tiers:
            raise ValueError('passive expert setup lacks original frozen tier projection')
        if initial_banks:
            if len(initial_banks)!=len(expert):
                raise ValueError('passive expert setup mixes legacy sliced and initial-bank populations')
            expected_selection=set();selected_totals={};selected_banks={};roles={};first_participant=0
            for tier_index,tier in enumerate(tiers):
                counts=tier.get('resolved_live_experts_per_layer')
                domains=[d for d in plan.get('domains',[]) if d.get('name')==tier.get('domain')]
                if not isinstance(counts,list) or len(domains)!=1:
                    raise ValueError('passive initial bank lacks original admitted tier/domain')
                for layer,count in enumerate(counts):
                    _passive_uint(count,'resolved expert population')
                    actual_layer=_passive_uint(plan.get('first_model_layer'),'first model layer')+layer
                    for member_index,member in enumerate(domains[0].get('participants',[])):
                        device=_expert_member_device(member)
                        expected_selection.add((device,actual_layer,tier_index,first_participant+member_index))
                first_participant+=len(domains[0].get('participants',[]))
            seen=set()
            for bank in initial_banks:
                tier_index=bank['tier']
                if tier_index>=len(tiers):raise ValueError('foreign initial expert bank tier')
                tier=tiers[tier_index]
                domain=next(d for d in plan['domains'] if d['name']==tier['domain'])
                key=(bank['device'],bank['layer'],tier_index,bank['participant'])
                if key not in expected_selection or key in seen or bank['tier_participant_count']!=len(domain['participants']):
                    raise ValueError('passive initial expert bank has foreign or duplicate membership')
                seen.add(key)
                # MoEExpertOwnerMap::buildTierParticipants assigns global IDs
                # by original tier order/member order. Preserve that identity,
                # including distinct CPU participants sharing a DeviceId string.
                role=bank['layer_role'];layer=bank['layer']
                if roles.setdefault(layer,role)!=role:raise ValueError('passive initial expert layer role changed')
                total_key=(tier_index,layer)
                selected_totals[total_key]=selected_totals.get(total_key,0)+bank['selected']
                selected_banks.setdefault(total_key,[]).append(bank['selected'])
            if seen!=expected_selection:raise ValueError('passive initial expert bank coverage is partial')
            first=_passive_uint(plan['first_model_layer'],'first model layer')
            for (tier_index,layer),total in selected_totals.items():
                tier=tiers[tier_index]
                domain=next(d for d in plan['domains'] if d['name']==tier['domain'])
                compute=domain.get('routed_compute_policy');phase=domain.get('routed_phase_policy')
                population=tier['resolved_live_experts_per_layer'][layer-first]
                # Registry observes physical expert IDs, not phase route counts.
                # Tensor shards and full replicas repeat those IDs per member;
                # phase-aware replicated decode likewise requires full residents.
                if phase=='PrefillApportionedDecodeReplicated' and compute in ('Apportioned','Replicated'):
                    replicated=True
                elif phase=='Uniform' and compute in ('Apportioned','Replicated','TensorSharded','GateUpOwnedDownColumns'):
                    # Initial banks enumerate movable source expert IDs. Gate/up
                    # follows one owner just like whole-expert apportionment;
                    # separately retained fixed down slices do not replicate
                    # these movable-bank IDs on every participant.
                    replicated=compute not in ('Apportioned','GateUpOwnedDownColumns')
                else:
                    raise ValueError('unsupported passive initial-bank compute/phase policy: '+str(compute)+'/'+str(phase))
                if (replicated and any(value!=population for value in selected_banks[(tier_index,layer)])) or \
                        (not replicated and total!=population):
                    raise ValueError('passive initial expert bank population differs from frozen tier policy')
            main=[layer for layer,role in roles.items() if role=='main_transformer']
            predictor=[layer for layer,role in roles.items() if role=='mtp_predictor']
            if predictor and not config.get('shared_predictor'):
                raise ValueError('passive initial predictor bank has no original predictor source')
            if not main or min(main)!=first or (predictor and min(predictor)<=max(main)):
                raise ValueError('passive initial expert main/predictor role ordering changed')
        else:
            expected_selection=set()
            for tier in tiers:
                counts=tier.get('resolved_live_experts_per_layer')
                domains=[d for d in plan.get('domains',[]) if d.get('name')==tier.get('domain')]
                if not isinstance(counts,list) or len(domains)!=1:
                    raise ValueError('passive expert setup lacks original admitted layer/domain membership')
                participants=domains[0].get('participants',[])
                for layer,count in enumerate(counts):
                    if not _passive_uint(count,'resolved expert population'):continue
                    for participant,p in enumerate(participants):
                        expected_selection.add((_expert_member_device(p),
                            _passive_uint(plan.get('first_model_layer'),'first model layer')+layer,participant))
            if not expected_selection.issubset({(r['device'],r['layer'],r['participant']) for r in expert}):
                raise ValueError('passive expert physical selection coverage is partial')
            declared_members={(_expert_member_device(p),index)
                for domain in plan.get('domains',[]) for index,p in enumerate(domain.get('participants',[]))}
            if any((r['device'],r['participant']) not in declared_members for r in expert):
                raise ValueError('passive expert selection has a foreign original participant')


    if not adaptive:
        if local_setup or local_terminal:raise ValueError('ordinary lookup cannot claim local hot cache evidence')
        if setup:raise ValueError('ordinary lookup cannot claim adaptive admitted setup')
        if observed_fixed:
            if terminal or submissions or not expert:
                raise ValueError('fixed native pipeline measured evidence lacks expert owners or includes hosted lookup')
            from model_parity_inventory import require_observed_pipeline_terminal
            return dict(collection_policy=DECLARED_PASSIVE_POLICY,adaptive=False,
                expert_selection=expert,fixed_native_pipeline=require_observed_pipeline_terminal(data,records))
        return dict(collection_policy=DECLARED_PASSIVE_POLICY,adaptive=False,
            expert_selection=expert,qualification_limits=['ordinary lookup schema remains its original mode contract'])
    request=policy['gpu_adaptive']
    backing=request.get('source_backing')
    if backing not in ('SSD','RAM'):
        raise ValueError('unsupported adaptive source backing in passive evidence')
    mode='gpu_subordinate_full_ram' if backing=='RAM' else 'host_adaptive_ram_ssd'
    continuation=_declared_continuation(config)
    placement=request.get('placement') or {}
    arm=placement.get('mode')
    if arm=='ConsumerLocal':
        expected={key:(request['maximum_gpu_cache_rows'],request['gpu_cache_physical_byte_limit']) for key in continuation}
        coordinators=set(continuation)
    elif arm=='DeclaredRankLocal':
        expected={}
        for owner in placement.get('owners',[]):
            p=owner['participant'];key=(_passive_uint(p['world_rank'],'owner rank'),_passive_device(f"{p['backend']}:{p['ordinal']}"))
            if key in expected:raise ValueError('duplicate original configured cache owner')
            expected[key]=(owner['maximum_cache_rows'],owner['cache_physical_byte_limit'])
        p=placement['coordinator'];coordinators={(_passive_uint(p['world_rank'],'coordinator rank'),_passive_device(f"{p['backend']}:{p['ordinal']}"))}
        if not coordinators.issubset(set(continuation)):
            raise ValueError('configured coordinator is not an original continuation participant')
    else:
        raise ValueError('unsupported adaptive placement in passive evidence')
    seen=set();cache={};roles={};source_identity=None;host=None
    numeric=('world_rank','admitted_slots','payload_bytes','metadata_bytes','source_storage_bytes','table_storage_bytes','lookup_staging_bytes',
        'source_generation','source_file_offset','source_rows','source_row_elements','source_row_bytes')
    for device,tags in setup:
        if any(not isinstance(tags.get(name),str) for name in numeric):raise ValueError('passive setup numeric tags are not original strings')
        values={name:_passive_uint(tags.get(name),'setup '+name) for name in numeric}
        if tags.get('storage') != mode or not values['source_generation'] or not values['source_rows'] or not values['source_row_elements'] or not values['source_row_bytes']:
            raise ValueError('passive setup source/backing identity is incomplete')
        identity=tuple(tags.get(k) for k in ('source_path','source_tensor','source_format','source_generation','source_file_offset','source_rows','source_row_elements','source_row_bytes'))
        if any(v is None or v=='' for v in identity) or source_identity is not None and identity!=source_identity:
            raise ValueError('passive setup combines different original sources')
        source_identity=identity
        if 'local_storage_identity' in tags or 'local_consumer' in tags:
            raise ValueError('legacy passive setup cannot carry local root lineage')
        key=(values['world_rank'],device);role=tags.get('role')
        if role not in ('gpu_cache','gpu_policy','gpu_source_staging','shared_host_source') or (key,role) in seen:
            raise ValueError('passive setup duplicates or invents a physical role')
        seen.add((key,role));roles.setdefault(role,set()).add(key)
        if not values['lookup_staging_bytes']:raise ValueError('passive setup omitted its mandatory original staging')
        if values['table_storage_bytes'] != values['payload_bytes']+values['metadata_bytes']+values['source_storage_bytes']:
            raise ValueError('passive setup physical byte partition is inconsistent')
        if role=='gpu_cache':
            if key not in expected or values['source_storage_bytes']:
                raise ValueError('passive cache owner is foreign')
            maximum,ceiling=map(lambda x:_passive_uint(x,'configured cache ceiling'),expected[key])
            slots=values['admitted_slots']
            if slots>maximum or values['table_storage_bytes']>ceiling or bool(slots)!=bool(values['table_storage_bytes']):
                raise ValueError('passive cache grant violates original configured bounds')
            if maximum and ceiling and not slots:
                raise ValueError('positive cache comparison has no actual positive admitted grant')
            cache[key]=values
        elif role=='shared_host_source':
            if device!='CPU:0' or values['world_rank'] not in {rank for rank,_ in coordinators} or host is not None or not values['source_storage_bytes']:
                raise ValueError('passive setup must retain one original host source')
            host=values
            host_request=request['host']
            if values['admitted_slots']>_passive_uint(host_request['maximum_cache_rows'],'host cache bound'):
                raise ValueError('passive host cache grant exceeds original request')
            if values['payload_bytes']+values['metadata_bytes']>_passive_uint(host_request['cache_physical_byte_limit'],'host byte ceiling'):
                raise ValueError('passive host cache bytes exceed original request')
            if host_request['maximum_cache_rows'] and host_request['cache_physical_byte_limit'] and not values['admitted_slots']:
                raise ValueError('positive RAM cache comparison has no actual admitted grant')
            if backing=='RAM' and any(values[k] for k in ('admitted_slots','payload_bytes','metadata_bytes')):
                raise ValueError('full RAM source has a redundant cache grant')
        elif key not in coordinators or values['admitted_slots'] or values['payload_bytes'] or values['source_storage_bytes']:
            raise ValueError('passive source/policy root is foreign or aliases cache/source payload')
    if set(cache)!=set(expected) or roles.get('gpu_policy')!=coordinators or roles.get('gpu_source_staging')!=coordinators or host is None:
        raise ValueError('passive setup is partial')
    row_bytes=_passive_uint(setup[0][1]['source_row_bytes'],'original source row width')
    device_names={device for _,device in coordinators}
    device_metrics=('adaptive_lookup_calls','adaptive_completed_live_ids','adaptive_cache_hit_rows','adaptive_cache_miss_rows',
        'adaptive_source_rows','adaptive_source_native_bytes','adaptive_promotions','adaptive_evictions','adaptive_failed_frames')
    host_metrics=('intervals','requests','ids','id_bytes','native_bytes','metadata_bytes','acknowledgement_bytes','terminal_bytes','failures',
        'window_hits','window_misses','window_evictions','adaptive_batches','ram_rows','source_rows','duplicate_rows','cache_admissions',
        'cache_evictions','cache_admission_rejections','all_pinned_responses')
    device_seen=set();host_seen=set();observations=[]
    for (device,tag_items),group in terminal.items():
        tags=dict(tag_items)
        if tags.get('transport')!='pooled_subordinate':raise ValueError('passive adaptive terminal transport changed')
        if tags.get('authority')=='terminal_device_generation_control':
            names=device_metrics
            if tags.get('storage')!='vram' or device not in {d for _,d in continuation}:
                raise ValueError('foreign passive device terminal owner')
            if device in device_names:device_seen.add(device)
        else:
            names=host_metrics;host_seen.add(device)
            if tags.get('storage')!=mode or device not in device_names:raise ValueError('foreign passive host terminal source')
        if set(group)!=set(names) or len({group[name][0] for name in names})!=1:
            raise ValueError('passive terminal tuple is incomplete or mixes intervals')
        values={name:group[name][1] for name in names}
        if any(value for name,value in values.items() if name in ('failures','adaptive_failed_frames')):
            raise ValueError('passive measured lookup reports failed work')
        if names==device_metrics:
            # Original participant controllers publish the fixed terminal tuple
            # even when this participant owns no pooled-source execution.
            if device not in device_names and any(values.values()):
                raise ValueError('non-coordinator passive device reports source work')
            if values['adaptive_source_native_bytes']!=values['adaptive_source_rows']*row_bytes:
                raise ValueError('passive device native bytes differ from original row width')
            if values['adaptive_source_rows']>values['adaptive_cache_miss_rows'] or values['adaptive_cache_hit_rows']+values['adaptive_cache_miss_rows']>values['adaptive_completed_live_ids']:
                raise ValueError('passive device rows do not partition completed live work')
        else:
            if values['native_bytes']!=values['ids']*row_bytes or values['id_bytes']!=values['ids']*8:
                raise ValueError('passive host native/id bytes differ from original row/channel width')
            if values['ram_rows']+values['source_rows']+values['duplicate_rows']!=values['ids'] or values['cache_evictions']>values['cache_admissions']:
                raise ValueError('passive source rows do not partition completed responses')
            if backing=='RAM' and any(values[k] for k in ('ram_rows','cache_admissions','cache_evictions','cache_admission_rejections',
                    'all_pinned_responses','window_hits','window_misses','window_evictions')):
                raise ValueError('full RAM terminal tuple reports cache or mapping work')
        observations.append(dict(device=device,tags=tags,count=next(iter(group.values()))[0],counters=values))
    if device_seen!=device_names or host_seen!=device_names:
        raise ValueError('passive measured terminal owner coverage is incomplete')
    return dict(collection_policy=DECLARED_PASSIVE_POLICY,adaptive=True,backing=backing,
        admitted_gpu_cache_class=('positive' if all(r['admitted_slots'] for r in cache.values()) else 'c0' if not any(r['admitted_slots'] for r in cache.values()) else 'mixed'),
        measured_scope='original_benchmark_reset_before_measurement',setup=[dict(device=d,tags=t) for d,t in setup],
        expert_selection=sorted(expert,key=lambda x:(x['device'],x['layer'],x['participant'])),
        terminal_observations=observations,native_submission_observations=submissions,reset_terminal_observations=resets,
        local_hot=validate_declared_local_hot_evidence(request,continuation,coordinators,local_setup,local_terminal,source_identity),
        qualification_limits=['aggregate_gpu_hits_do_not_identify_locality','native_submissions_are_not_wire_bytes',
            'mapping_window_misses_are_not_ssd_io','startup_full_ram_preload_is_outside_measured_window'])



# Stable native terminal publication schema: original-L1 IDs, native rows, then
# receipt-completed bytes. These are not a row-format/size or admission table.
LOCAL_TERMINAL_NAMES = (
    'local_read_original_ids','local_hit_original_ids','local_all_hit_wait_avoided_original_ids','local_consumed_original_ids',
    'local_future_on_time_native_rows','local_future_late_native_rows','local_future_useful_native_rows','local_future_wasted_native_rows',
    'local_promotions','local_evictions','local_request_key_payload_bytes','local_native_result_payload_bytes',
    'local_lookup_control_header_bytes','local_acknowledgement_bytes','local_request_admission_bytes')


def validate_declared_local_hot_evidence(request,continuation,coordinators,setup,records,source_identity):
    """Bind completed native L1 windows to original cold roots; never infer locality or time.

    This consumes the same fifteen-field publication contract as the native
    terminal observer. Initial storage windows occur once per storage, not once
    per request; subsequent windows retain their actual aggregated count.
    """
    local=request.get('local_hot_cache')
    if local is None:
        if setup or records:raise ValueError('shared-only lookup cannot claim local cache observations')
        return None
    if request.get('placement',{}).get('mode')!='DeclaredRankLocal':
        raise ValueError('local hot cache requires original declared shared placement')
    expected={key:(local['maximum_cache_rows'],local['cache_physical_byte_limit']) for key in continuation}
    overridden=set()
    for override in local.get('consumer_overrides',[]):
        p=override['participant']
        key=(_passive_uint(p['world_rank'],'local consumer rank'),_passive_device(f"{p['backend']}:{p['ordinal']}"))
        if key not in expected or key in overridden:raise ValueError('foreign or duplicate local ceiling override')
        overridden.add(key);expected[key]=(override['maximum_cache_rows'],override['cache_physical_byte_limit'])
    cache={};ready={};storage_ids=set()
    numeric=('world_rank','admitted_slots','payload_bytes','metadata_bytes','source_storage_bytes','table_storage_bytes','lookup_staging_bytes',
        'source_generation','source_file_offset','source_rows','source_row_elements','source_row_bytes','local_storage_identity')
    for device,tags in setup:
        if any(not isinstance(tags.get(name),str) for name in numeric):raise ValueError('local setup lost original numeric tags')
        values={name:_passive_uint(tags[name],'local setup '+name) for name in numeric}
        identity=tuple(tags.get(k) for k in ('source_path','source_tensor','source_format','source_generation','source_file_offset','source_rows','source_row_elements','source_row_bytes'))
        consumer=_passive_device(tags.get('local_consumer'))
        key=(values['world_rank'],consumer);role=tags.get('role')
        if key not in expected or identity!=source_identity or tags.get('storage')!=('gpu_subordinate_full_ram' if request['source_backing']=='RAM' else 'host_adaptive_ram_ssd') or not values['local_storage_identity'] or not values['lookup_staging_bytes']:
            raise ValueError('local setup changed original source, consumer or live root')
        if values['table_storage_bytes']!=values['payload_bytes']+values['metadata_bytes']+values['source_storage_bytes']:
            raise ValueError('local setup physical byte partition is inconsistent')
        if role=='local_gpu_cache':
            maximum,ceiling=(_passive_uint(v,'local ceiling') for v in expected[key])
            if device!=consumer or key in cache or values['local_storage_identity'] in storage_ids or values['source_storage_bytes'] or \
                values['admitted_slots']>maximum or values['table_storage_bytes']>ceiling or \
                bool(values['admitted_slots'])!=bool(values['table_storage_bytes']) or \
                values['payload_bytes']!=values['admitted_slots']*values['source_row_bytes'] or \
                (maximum and ceiling and not values['admitted_slots']):
                raise ValueError('local native cache differs from original admitted ceiling/root')
            cache[key]=values;storage_ids.add(values['local_storage_identity'])
        elif role=='local_ready':
            if device!='CPU:0' or key in ready or any(values[k] for k in ('admitted_slots','payload_bytes','metadata_bytes','source_storage_bytes','table_storage_bytes')):
                raise ValueError('local ready page is foreign, duplicated or aliases native cache')
            ready[key]=values
        else:raise ValueError('unknown local setup role')
    if set(cache)!=set(expected) or set(ready)!=set(expected) or any(cache[k]['local_storage_identity']!=ready[k]['local_storage_identity'] for k in expected):
        raise ValueError('local setup requires one exact GPU/ready pair per original consumer')
    windows={};reads={key:0 for key in expected}
    for record in records:
        name=record['name'];device=_passive_device(record['device']);tags=record['tags'];count=_passive_uint(record['count'],'local count')
        matches=[key for key in expected if key[1]==device]
        if len(matches)!=1 or name not in LOCAL_TERMINAL_NAMES or record['phase']!='generation_terminal':
            raise ValueError('local terminal changed original participant or field')
        key=matches[0];index=LOCAL_TERMINAL_NAMES.index(name);window=tags.get('window')
        required={'authority':'completed_local_hot_terminal','storage_identity':str(cache[key]['local_storage_identity']),
            'unit':'original_l1_ids' if index<4 else 'native_rows' if index<10 else 'bytes',
            'wire_attribution':'receive_completed_only','window':window}
        value=_passive_uint(record['value'],'local total')
        minimum=record.get('sequence_minimum_words');maximum=record.get('sequence_maximum_words')
        if tags!=required or window not in ('since_original_storage_initialization','between_completed_terminals') or \
            (window=='since_original_storage_initialization' and count!=1) or not count or value>9007199254740991 or \
            _passive_uint(record.get('sequence_word_count'),'local exact sequence word count')!=count or not isinstance(minimum,list) or not isinstance(maximum,list) or len(minimum)!=1 or len(maximum)!=1:
            raise ValueError('local terminal changed root, window, units or completed publication')
        low=_passive_uint(minimum[0],'local minimum');high=_passive_uint(maximum[0],'local maximum')
        if low>high or not low*count<=value<=high*count:raise ValueError('local terminal aggregate is not its completed observations')
        group=windows.setdefault((key,window),{})
        if name in group:raise ValueError('duplicate local terminal field')
        group[name]=(count,value)
    observations=[]
    for (key,window),group in windows.items():
        if set(group)!=set(LOCAL_TERMINAL_NAMES) or len({count for count,_ in group.values()})!=1:
            raise ValueError('local terminal window is incomplete')
        v=[group[name][1] for name in LOCAL_TERMINAL_NAMES];width=cache[key]['source_row_bytes']
        if any(v[4:8]):raise ValueError('current-only compact delivery cannot report future native work')
        if v[1]>v[0] or v[2]>v[1] or v[3]>v[0] or (not cache[key]['admitted_slots'] and any(v[i] for i in (1,8,9))):
            raise ValueError('local terminal original-L1 work or C0 churn is impossible')
        if key in coordinators:
            # Native protocol Request88 / Result96; self produces no wire.
            if v[11] or v[14] or v[10]%8 or v[12]%88 or v[13]%96 or v[12]//88!=v[13]//96:
                raise ValueError('local coordinator wire is not completed foreign receives')
        elif v[10] or v[13] or v[11]%width or v[12]%96 or v[14]%40 or v[11]//width>v[0]-v[1]:
            raise ValueError('local reader wire exceeds completed native misses')
        reads[key]+=v[0]
        observations.append(dict(world_rank=key[0],device=key[1],storage_identity=cache[key]['local_storage_identity'],
            window=window,count=next(iter(group.values()))[0],counters={name:group[name][1] for name in LOCAL_TERMINAL_NAMES}))
    if any(not value for value in reads.values()):raise ValueError('local terminal omitted useful work on an original consumer')
    return dict(setup=[dict(device=device,tags=tags) for device,tags in setup],terminal_observations=observations,
        qualification_limits=['original_l1_ids_are_not_compact_pool_rows','lifetime_windows_are_not_requests','legacy_source_and_local_windows_are_not_implicitly_paired',
            'future_shared_misses_not_supported','physical_budget_pairing_requires_final_original_pma_costs'])

def validate_declared_passive_evidence(data: dict) -> dict:
    """Reject incomplete external documents through the original passive schema owner."""
    try:
        return _validate_declared_passive_evidence(data)
    except (KeyError,TypeError,OverflowError) as error:
        raise ValueError('incomplete declared passive evidence schema') from error


def require_matched_original_launch(original: dict, current: dict, *,
                                    panel_binary: Path, release_binary: Path) -> dict:
    """Match MPI placement semantics while authenticating each build's actual helper.

    Integration accuracy and Release timing intentionally own distinct session
    executables. Their path/stat identities remain independently pinned across
    timing; rankfile, MPI implementation, population, binding and overlap policy
    must match exactly. Neither helper identity is evidence about the other.
    """
    from model_kld_corpus import identity
    if (not isinstance(original,dict) or not isinstance(current,dict)
            or set(original) != set(current) or 'session_launcher' not in original
            or {k:v for k,v in original.items() if k != 'session_launcher'} !=
               {k:v for k,v in current.items() if k != 'session_launcher'}):
        raise ValueError("Release observed pipeline changed its paired MPI rankfile, launcher or binding")
    pins = {}
    for launch,binary_owner in ((original,panel_binary),(current,release_binary)):
        helper = launch['session_launcher']
        if (not isinstance(helper,dict) or set(helper) != {'path','stat'}
                or not isinstance(helper['path'],str) or not helper['path']):
            raise ValueError("Observed pipeline omitted its original build-owned session helper")
        path = Path(helper['path']).resolve(strict=True)
        expected = (Path(binary_owner).resolve(strict=True).parent/'llaminar_mpi_session').resolve(strict=True)
        if path != expected:
            raise ValueError("Observed pipeline session helper escaped its original binary build owner")
        if not path.is_file() or not os.access(path,os.X_OK):
            raise ValueError("Observed pipeline session helper is not an executable file")
        pin = identity(path)
        if json.loads(json.dumps(pin)) != helper['stat']:
            raise ValueError("Observed pipeline build-owned session helper changed after admission")
        pins[str(path)] = pin
    return pins


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
    panel_path = args.panel_report.resolve(strict=True)
    panel_pin = identity(panel_path)
    panel = json.loads(panel_path.read_text())
    unchanged({str(panel_path):panel_pin})
    focused = panel.get('mode') == 'local_paired_focused_diagnostic'
    import model_parity_inventory as inventory
    paired_source=(panel.get('paired_workload') or {}).get('source_export',{})
    observed_pipeline='observed_pipeline_binding' in paired_source
    if observed_pipeline and focused:
        raise ValueError("Observed pipeline timing requires complete production prerequisites")
    scope = inventory.InventoryScope.LOCAL_DIAGNOSTIC if focused or observed_pipeline else inventory.InventoryScope.ALL
    (_campaign, exact, record), inventory_pins = declared_inventory(args,binary,scope=scope)
    output = args.report.parent / (args.report.stem + '-artifacts')
    if not args.list:
        output.mkdir(parents=True, exist_ok=False)
    selection = paired_owner.PairedCorpusSelection.from_export(exact, record, args.corpus_root, ROOT,
        metadata_policy=(paired_owner.PairedMetadataPolicy.SOURCE_BOUND_DIAGNOSTIC if focused else
            paired_owner.PairedMetadataPolicy.SOURCE_BOUND_OBSERVED_PIPELINE if observed_pipeline else
            paired_owner.PairedMetadataPolicy.CANONICAL_ROOT),
        configuration_directory=None if args.list else output)
    workload = json.loads(args.workload.read_text())
    profile = declared_panel_eligibility(panel, exact, record, workload,allow_focused_diagnostic=focused)
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

    if observed_pipeline:
        inventory.require_observed_pipeline_runtime(record,native.get('observed_pipeline_binding'))
        if (native.get('observed_pipeline_binding') != cell.get('observed_pipeline_binding')
                or native.get('fixed_native_pipeline_terminal') != cell.get('fixed_native_pipeline_terminal')
                or not native.get('fixed_native_pipeline_terminal')):
            raise ValueError("numerical panel omitted the original fixed pipeline terminal evidence")
    accuracy_scope = declared_focused_accuracy_scope(panel,selection,cell) if focused else None
    if json.loads(selection.paths['benchmark'].read_text()) != workload:
        raise ValueError("benchmark workload differs from the original paired obligation")
    pins = {**declared_source_pins(selection), **inventory_pins}
    for path in (cell_directory/'described-config.json',cell_directory/'controller.json',cell_directory/'native-observation.json'):
        pins[str(path)] = identity(path)

    pins[str(panel_path)] = panel_pin
    for path in (binary, parser, args.workload.resolve(strict=True)):
        pins[str(path)] = identity(path)
    if args.list:
        unchanged(pins)
        print(f"[declared-diagnostic-benchmark] eligible {exact}")
        if focused:
            print("[declared-diagnostic-benchmark] focused_accuracy_only full_preflight=false certification_eligible=false")
        return 0
    environment = declared_environment(os.environ,workload,selection.topology['arithmetic'],output)
    canonical = declared_config_identity(selection, observed, parser, environment)
    write_json(output/'described-config.json', canonical)
    helper = binary.parent/'llaminar_mpi_session'
    if not helper.is_file() or not os.access(helper,os.X_OK):
        raise ValueError("missing original MPI session retirement helper")
    prompt = validate_workload(workload)
    # Retain the identical source document in this existing artifact directory;
    # command evidence never depends on a temporary parser input after return.
    # from_export already retained the original representation and translated
    # its authenticated numerical observer's disabled-prefix policy. Rebuilding
    # raw profile argv here would lose that exact source-owned projection.
    engine_arguments=selection.canonical_arguments
    engine = ['benchmark',*engine_arguments,
        '--prompt',prompt,'-n',str(workload['decode_tokens']),'--seed',str(workload['seed']),
        '--temperature',str(workload['temperature']),'--top-k',str(workload['top_k']),
        '--top-p',str(workload['top_p']),'--benchmark-json-output',str(output/'benchmark.json')]
    command = [str(helper),'--session-base',str(output/'mpi-session'),'--',str(binary),*engine]
    invocation = dict(environment=environment,exact_cell=exact,collection_policy=DECLARED_PASSIVE_POLICY)
    if observed_pipeline:
        from types import SimpleNamespace
        from run_model_kld_canaries import original_local_launch
        original_launch = controller.get('native_launch') or {}
        rankfile = (original_launch.get('rankfile') or {}).get('path')
        if not rankfile or original_launch.get('schema') != 'llaminar.original-mpi-rankfile-launch.v1':
            raise ValueError("Release observed pipeline requires its paired original discovery MPI launch")
        command, launch_pins, launch = original_local_launch(
            SimpleNamespace(mpi_rankfile=Path(rankfile),binary=binary), selection, None,
            output, [str(binary),*engine], environment, prepare_session=False)
        pins.update(require_matched_original_launch(original_launch,json.loads(json.dumps(launch)),
            panel_binary=Path(panel['binary']['path']),release_binary=binary))
        pins.update(launch_pins)
        invocation['native_launch'] = launch
    write_json(output/'invocation.json',dict(invocation,argv=command))
    result = dict(schema=1, policy='diagnostic_declared', diagnostic=True, certification_eligible=False,
        exact_cell=exact, source_export=record, complete=False, passed=False,
        collection_policy=DECLARED_PASSIVE_POLICY,
        panel_report=str(args.panel_report), workload=workload)
    if focused:
        result.update(full_preflight=False,accuracy_prerequisite=accuracy_scope)
    armed = False
    try:
        unchanged(pins)
        driver.begin(output/'driver-state.json'); armed=True
        run_declared_guardian(command,environment,output)
        result['native_retired'] = True
        data = json.loads((output/'benchmark.json').read_text())
        measured = validate_measurement(data,workload)
        declared_runtime_configuration(data,canonical,native['actual_owner']['actual_config'],workload,output,parser,environment)
        if observed_pipeline:
            runtime_binding=data.get('observed_pipeline_binding')
            inventory.require_observed_pipeline_runtime(record,runtime_binding)
            if runtime_binding != native.get('observed_pipeline_binding'):
                raise ValueError("Release observed fixed ranges differ from the accepted numerical cell")
            prompt_ids = data.get('prompt',{}).get('token_ids')
            generated_ids = [row.get('generated_token_ids') for row in data['iterations']]
            if (not isinstance(prompt_ids,list) or len(prompt_ids) != measured['prefill_tokens']
                    or any(type(token) is not int or token < 0 for token in prompt_ids)
                    or any(not isinstance(ids,list) or len(ids) != workload['decode_tokens']
                           or any(type(token) is not int or token < 0 for token in ids) for ids in generated_ids)):
                raise ValueError("Release observed pipeline omitted exact prompt or accepted output token IDs")
            result.update(observed_pipeline_binding=runtime_binding,prompt_token_ids=prompt_ids,
                          accepted_token_ids=generated_ids)
        passive = validate_declared_passive_evidence(data)
        unchanged(pins)
        result.update(measurement=measured,passive_evidence=passive)
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
