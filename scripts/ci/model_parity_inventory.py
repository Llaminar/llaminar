#!/usr/bin/env python3
## @file model_parity_inventory.py
# @brief Authenticate source-owned CTest/GoogleTest metadata once for inventory and qualification consumers.
"""Export canonical model cells without executing numerical or generation tests.

CTest owns exact membership; GoogleTest's typed JSON owns each configuration.
Routine generation regression projects MTP-off and dynamic MTP, while HTTP certification,
remote MPI certification and benchmarks consume their explicitly tagged subsets. No consumer
may reconstruct a topology from a test name or maintain another axis expander.
Numerical-only candidates keep their complete CTest/GoogleTest membership proof
but do not enter canonical runtime/control projections. Discovery never loads
model tensors, stages weights, or initializes devices.
"""
from __future__ import annotations

import argparse
import dataclasses
from enum import Enum
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import run_production_parity_campaigns as parity
from production_artifacts import write_json

ROOT = Path(__file__).resolve().parents[2]


def cross_host_scenarios(record: dict) -> list[dict]:
    """Validate emitted remote cases without generating a second topology matrix.

    Missing metadata is stale, not an empty successful cloud campaign. Every
    declared topology must carry both public frontend routes. Model path and
    HTTP workload remain in the parent record so remounting cannot create a
    second, stale model identity. This is discovery validation, not live proof.
    """
    scenarios = record.get("cross_host_e2e")
    if not isinstance(scenarios, list):
        raise ValueError("canonical cell omitted its explicit cross-host E2E eligibility; rebuild/export stale metadata")
    if scenarios and not isinstance(record.get("e2e"), dict):
        raise ValueError("cross-host E2E eligibility requires an HTTP-tagged source cell")
    seen, routes = set(), {}
    for scenario in scenarios:
        if (not isinstance(scenario, dict) or type(scenario.get("schema")) is not int
                or scenario["schema"] != 1 or "model" in scenario
                or not isinstance(scenario.get("id"), str) or not scenario["id"]
                or any(ch in scenario["id"] for ch in "\x00\r\n")
                or scenario["id"] in seen):
            raise ValueError("invalid or duplicate cross-host E2E identity")
        seen.add(scenario["id"])
        topology = scenario.get("topology")
        if (not isinstance(topology, dict) or topology.get("kind") != "cross-host-expert-overlay"
                or topology.get("continuation_backend") not in ("cuda", "rocm")
                or type(topology.get("continuation_devices")) is not int or topology["continuation_devices"] != 1
                or type(topology.get("remote_cpu_hosts")) is not int or not 0 < topology["remote_cpu_hosts"] < 2**31 - 1
                or type(topology.get("cpu_ranks_per_host")) is not int or topology["cpu_ranks_per_host"] != 1
                or type(topology.get("execution_ranks")) is not int
                or topology["execution_ranks"] != topology["remote_cpu_hosts"] + 1
                or type(topology.get("continuation_priority")) is not int
                or type(topology.get("remote_priority")) is not int
                or topology["continuation_priority"] >= topology["remote_priority"]):
            raise ValueError("invalid cross-host E2E topology")
        frontend = scenario.get("frontend")
        arguments = scenario.get("server_policy_args")
        planning = scenario.get("planning")
        if (planning != {"mode": "auto", "strategy": "expert-overlay", "device_counts": {
                topology["continuation_backend"]: topology["continuation_devices"],
                "cpu": topology["remote_cpu_hosts"]},
                "mpi_ranks": topology["execution_ranks"]}
                or any(type(count) is not int for count in planning["device_counts"].values())):
            raise ValueError("cross-host E2E automatic constraints differ from its typed topology")
        if (frontend not in ("plan-apply", "auto-serve")
                or scenario.get("movement_evidence") != "required" or scenario.get("owner_order") != "ordinal"
                or not isinstance(arguments, list) or not arguments
                or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in arguments)):
            raise ValueError("invalid cross-host E2E frontend/policy")
        key = json.dumps(topology, sort_keys=True)
        selected = routes.setdefault(key, {})
        if frontend in selected:
            raise ValueError("duplicate cross-host E2E frontend route")
        selected[frontend] = (arguments, planning)
    for selected in routes.values():
        if set(selected) != {"plan-apply", "auto-serve"}:
            raise ValueError("cross-host E2E topology must prove both public frontend routes")
        if selected["plan-apply"] != selected["auto-serve"]:
            raise ValueError("cross-host E2E routes must preserve identical inference policy")
    return scenarios


def select_cross_host_scenario(record: dict, exact: str) -> dict:
    """Select an existing typed remote case; never construct a topology from its name.

    Both frontend routes are validated before narrowing. A diagnostic harness
    cannot admit a hand-edited single-route profile as canonical eligibility.
    The caller must keep the selected immutable record and bind its actual
    resolved runtime/image separately; discovery is not execution evidence.
    """
    if (not isinstance(record, dict) or type(record.get("model_parity_schema")) is not int
            or record["model_parity_schema"] != 1 or not isinstance(record.get("id"), str)
            or not record["id"] or not isinstance(record.get("model"), str) or not record["model"]
            or not isinstance(exact, str) or not exact):
        raise ValueError("cross-host harness requires a canonical configuration and exact case")
    matches = [row for row in cross_host_scenarios(record) if row["id"] == exact]
    if len(matches) != 1:
        raise ValueError("requested cross-host case is absent from canonical eligibility")
    return matches[0]


def select_cross_host_scenario_from_manifest(document: dict, exact: str) -> dict:
    """Select one remote case from the revision-bound manifest projection.

    The HTTP harness receives the exported document, not an individual source
    configuration. Keep that distinction explicit so callers cannot
    accidentally pass a manifest to :func:`select_cross_host_scenario` and
    bypass the per-row schema checks. Exactly one parent row must own the
    requested scenario; duplicate ownership is a malformed projection.
    """
    if (not isinstance(document, dict) or type(document.get("schema")) is not int
            or document["schema"] != 1
            or document.get("scope") != InventoryScope.CROSS_HOST_E2E.value
            or not isinstance(document.get("source_revision"), str) or not document["source_revision"]
            or not isinstance(document.get("cells"), list)
            or not isinstance(exact, str) or not exact):
        raise ValueError("cross-host manifest must contain schema 1, cross-host scope, cells and an exact case")
    matches = []
    for parent in document["cells"]:
        if not isinstance(parent, dict) or not isinstance(parent.get("configuration"), dict):
            raise ValueError("cross-host manifest contains a malformed parent cell")
        # Validate the complete source row even when this exact case is absent;
        # malformed siblings must not disappear merely because the harness
        # narrowed the requested scenario.
        config = parent["configuration"]
        if (type(config.get("model_parity_schema")) is not int or config["model_parity_schema"] != 1
                or not isinstance(config.get("id"), str) or not config["id"]
                or not isinstance(config.get("model"), str) or not config["model"]):
            raise ValueError("cross-host manifest contains an invalid source configuration")
        for scenario in cross_host_scenarios(config):
            if scenario["id"] == exact:
                matches.append(scenario)
    if len(matches) != 1:
        raise ValueError("requested cross-host case is absent or duplicated in the canonical manifest")
    return matches[0]


class InventoryScope(str, Enum):
    """Select a projection of existing cells, never a different configuration."""
    ALL = "all"
    LOCAL_DIAGNOSTIC = "local-diagnostic"
    GENERATION = "generation"
    E2E = "e2e"
    CROSS_HOST_E2E = "cross-host-e2e"

    def accepts(self, record: dict) -> bool:
        """Require an explicit eligibility field even for an untagged cell."""
        if "e2e" not in record:
            raise ValueError("canonical cell omitted its explicit E2E eligibility")
        if record["e2e"] is not None and not isinstance(record["e2e"], dict):
            raise ValueError("canonical E2E eligibility must be a profile or null")
        # Schema1 originally contains only canonical records. Keep that default
        # byte-compatible; a source-owned numerical diagnostic must opt out
        # explicitly. Unknown values are stale/malformed, never an exclusion.
        role = record.get("inventory_role", "canonical")
        if not isinstance(role, str) or role not in ("canonical", "numerical_diagnostic"):
            raise ValueError("canonical cell has an unknown inventory role")
        if role == "numerical_diagnostic":
            if record["e2e"] is not None or record.get("cross_host_e2e"):
                raise ValueError("numerical diagnostic cannot declare runtime certification eligibility")
            # Only an explicit local diagnostic can inspect these source-owned
            # mathematical candidates. Ordinary/runtime inventories stay closed.
            return self == InventoryScope.LOCAL_DIAGNOSTIC
        if self == InventoryScope.CROSS_HOST_E2E:
            return bool(cross_host_scenarios(record))
        if self == InventoryScope.GENERATION:
            # Project producer-owned policy; never parse names/argv or create
            # another topology. Fixed depths remain mathematical diagnostics.
            from generation_regression_http import MTPPolicy, generation_profile
            return MTPPolicy(generation_profile(record)["mtp_policy"]) in (MTPPolicy.OFF, MTPPolicy.DYNAMIC)
        return self in (InventoryScope.ALL, InventoryScope.LOCAL_DIAGNOSTIC) or record["e2e"] is not None


def model_source_fields(record: dict) -> dict[str, str]:
    """Read explicit GGUF source roles without guessing filenames or MTP mode."""
    sources = {"model": record.get("model")}
    if "shared_predictor" in record:
        sources["shared_predictor"] = record["shared_predictor"]
    if any(not isinstance(path, str) or not path for path in sources.values()):
        raise ValueError("canonical cell has an invalid explicit GGUF source")
    return sources


def map_predictor_source(record: dict, transform) -> dict:
    """Remap the explicit predictor and its typed argv without changing policies.

    The caller supplies the admitted staging/mount/portable mapping. This
    helper makes no source selection or allocation decision, and rejects a
    foreign argument before a consumer can launch model materialization.
    """
    source = model_source_fields(record).get("shared_predictor")
    result = json.loads(json.dumps(record))
    # These are the producer-owned execution argv surfaces. Request bodies
    # and user/tool JSON are data and must never be interpreted as CLI policy.
    surfaces = [(result.get("runtime"), "server_args"),
                (result.get("e2e"), "server_args"),
                (result.get("benchmark"), "args"),
                (result.get("diagnostic_benchmark"), "args")]
    surfaces += [(scenario, "server_policy_args") for scenario in result.get("cross_host_e2e", [])]
    for owner, key in surfaces:
        if not isinstance(owner, dict) or key not in owner:
            continue
        args = owner[key]
        if not isinstance(args, list):
            raise ValueError("canonical execution argv must be a list")
        positions = [i for i, item in enumerate(args) if item == "--mtp-model"]
        if source is None:
            if positions:
                raise ValueError("canonical argv selected an undeclared shared predictor")
        elif (len(positions) != 1 or positions[0] + 1 >= len(args)
              or not isinstance(args[positions[0] + 1], str)
              or Path(args[positions[0] + 1]).resolve() != Path(source).resolve()):
            raise ValueError("canonical argv selected a foreign shared predictor")
        else:
            args[positions[0] + 1] = transform(source)
    for field,argv_field in (("runtime","server_args"),("diagnostic_benchmark","args")):
        owner=result.get(field)
        if isinstance(owner,dict) and "configuration_document" in owner:
            config=declared_document_configuration(owner,argv_field)
            declared=model_source_fields(record)
            if (not isinstance(config.get("model_path"),str)
                    or Path(config["model_path"]).resolve()!=Path(declared["model"]).resolve()
                    or config.get("shared_predictor")!=source):
                raise ValueError("lossless runtime/benchmark document selected a foreign original GGUF source")
            config["model_path"]=transform(declared["model"])
            if source is not None:config["shared_predictor"]=transform(source)
    if source is not None:
        result["shared_predictor"] = transform(source)
    return result


def declared_document_configuration(owner: dict, argv_field: str) -> dict:
    """Validate one existing lossless public document, without interpreting its C++ policy."""
    if (not isinstance(owner,dict)
            or owner.get("configuration_policy")!="lossless_declared_document_v1"
            or argv_field in owner):
        raise ValueError("lossless configuration conflicts with argv or has an unknown policy")
    document=owner.get("configuration_document")
    if (not isinstance(document,dict) or document.get("kind")!="llaminar.orchestration-config"
            or type(document.get("schema_version")) is not int or document["schema_version"]<=0
            or not isinstance(document.get("configuration"),dict)):
        raise ValueError("canonical input omitted its actual lossless configuration document")
    return document["configuration"]


def _runtime_document_configuration(record: dict) -> dict:
    """Retain the original server-document owner and exact representation contract."""
    return declared_document_configuration(record.get("runtime"),"server_args")


def require_observed_pipeline_document(record: dict) -> dict:
    """Authenticate the bound public document against the original physical receipt."""
    binding = record.get("observed_pipeline_binding")
    if not isinstance(binding, dict) or binding.get("schema") != 1 or binding.get("cell") != record.get("id") \
            or binding.get("active") is not True or binding.get("parity_work_succeeded") is not True \
            or binding.get("terminal_join_completed") is not True:
        raise ValueError("observed pipeline requires its completed original active receipt")
    runtime, benchmark = record.get("runtime"), record.get("diagnostic_benchmark")
    config = declared_document_configuration(runtime, "server_args")
    if config != declared_document_configuration(benchmark, "args") \
            or config.get("execution_rank_selection") != binding.get("selected_discovery_ranks") \
            or config.get("mpi_procs") != binding.get("discovery_size") \
            or config.get("planning_mode") != "Apply":
        raise ValueError("observed pipeline document substituted original rank selection")
    stages, domains = config.get("pp_stage_definitions"), config.get("domain_definitions")
    sizes, participants = binding.get("stage_sizes"), binding.get("participants")
    if not isinstance(stages, list) or not isinstance(domains, list) or not isinstance(sizes, list) \
            or not isinstance(participants, list) or len(stages) != len(sizes) or len(domains) != len(sizes):
        raise ValueError("observed pipeline document has incomplete stage geometry")
    for index, (stage, size) in enumerate(zip(stages, sizes)):
        members = [p for p in participants if p.get("stage") == index]
        domain = [d for d in domains if d.get("name") == stage.get("domain_name")]
        if len(domain) != 1 or len(members) != size or stage.get("stage_id") != index:
            raise ValueError("observed pipeline document changed original stage membership")
        domain = domain[0]
        if domain.get("owner_rank") != 0 or domain.get("scope") != "RANK_LOCAL" \
                or len(domain.get("devices", [])) != size:
            raise ValueError("observed pipeline document changed the selected local stage owner")
        for expected, actual in zip(members, domain["devices"]):
            if actual != {"hostname": binding["original_hostname"], "numa_node": binding["original_numa_node"],
                          "device_type": {"cuda": "CUDA", "rocm": "ROCm"}[expected["backend"]],
                          "device_ordinal": expected["ordinal"]}:
                raise ValueError("observed pipeline document changed an original physical endpoint")
    return binding


def require_observed_pipeline_runtime(record: dict, actual: dict) -> None:
    """Compare each actual native stage and original resource to the completed binding."""
    expected = require_observed_pipeline_document(record)
    keys = ("discovery_size", "selected_discovery_ranks", "original_owner", "original_hostname",
            "original_numa_node", "execution_rank", "stage_sizes", "participants")
    if not isinstance(actual, dict) or any(actual.get(key) != expected[key] for key in keys):
        raise ValueError("runtime substituted original selected rank, stage, UUID or NUMA ownership")
    owners, consumers, ple = actual.get("fixed_owners"), actual.get("ple_consumers"), actual.get("ple_stage")
    if (not isinstance(owners, list) or not isinstance(consumers, list) or type(ple) is not int
            or not 0 <= ple < len(expected["stage_sizes"]) or len(owners) != len(expected["participants"])
            or len(consumers) != expected["stage_sizes"][ple]):
        raise ValueError("runtime omitted the actual fixed owner or PLE consumer roster")
    native_device = lambda participant: participant["backend"].upper() + ":" + str(participant["ordinal"])
    # Native perf/device spelling may preserve ROCm's mixed-case enum spelling.
    stage_devices = [[native_device(p) for p in expected["participants"] if p["stage"] == stage]
                     for stage in range(len(expected["stage_sizes"]))]
    if [value.upper() for value in consumers] != stage_devices[ple]:
        raise ValueError("runtime PLE roster differs from original stage membership")
    next_head = 0
    seen = set()
    for owner in owners:
        stage, member = owner.get("stage"), owner.get("participant")
        reader, heads = owner.get("consumer"), owner.get("heads")
        device = owner.get("device", "").upper()
        if (type(stage) is not int or not 0 <= stage < len(stage_devices)
                or type(member) is not int or not 0 <= member < len(stage_devices[stage])
                or type(reader) is not int or not 0 <= reader < len(consumers)
                or device != stage_devices[stage][member] or device in seen
                or (stage == ple and reader != member)
                or owner.get("first_head") != next_head or type(heads) is not int or heads <= 0
                or type(owner.get("native_bytes")) is not int or owner["native_bytes"] <= 0):
            raise ValueError("runtime substituted a fixed storage range or original reader")
        seen.add(device)
        next_head += heads


FIXED_NATIVE_PIPELINE_METRICS = frozenset('fixed_head_pipeline_' + name for name in (
    'calls', 'live_ids', 'native_bytes', 'failures', 'request_native_bytes',
    'response_native_bytes', 'gather_native_bytes', 'completed_receipts'))


def require_observed_pipeline_terminal(document: dict, records: list[dict]) -> dict:
    """Match the native validator's completed terminal tuple to the same exported raw observations."""
    binding = document.get("observed_pipeline_binding", {})
    report = document.get("fixed_native_pipeline_terminal")
    if not isinstance(report, dict) or report.get("schema") != "llaminar.fixed-native-pipeline-terminal.v1":
        raise ValueError("missing native fixed pipeline terminal proof")
    owners = report.get("owners")
    if not isinstance(owners, dict) or not owners or set(owners) != {r["device"] for r in binding.get("fixed_owners", [])}:
        raise ValueError("fixed pipeline terminal proof changed its original storage roster")
    totals = {}
    observations = None
    for record in records:
        if record.get("domain") != "immutable_lookup":
            continue
        tags = record.get("tags", {})
        if record.get("name") not in FIXED_NATIVE_PIPELINE_METRICS:
            if (record.get("phase") == "generation_terminal"
                    and tags.get("authority") == "terminal_device_generation_control"):
                raise ValueError("fixed pipeline terminal contains a legacy immutable lookup layout")
            continue
        name, device = record["name"], record.get("device")
        value, count = record.get("value"), record.get("count")
        words = record.get("sequence_word_count")
        if (record.get("kind") != "counter" or record.get("phase") != "request_terminal"
                or tags.get("authority") != "pipeline_generation_terminal" or tags.get("request") != "0"
                or device not in owners or type(count) is not int or count <= 0
                or type(words) is not int or words != count
                or (observations is not None and observations != count)
                or type(value) not in (int, float) or not 0 <= value <= 2**53-1 or int(value) != value):
            raise ValueError("fixed pipeline terminal record has a foreign device, phase or count")
        fields = totals.setdefault(device, {})
        if name in fields:
            raise ValueError("fixed pipeline terminal repeats an original owner metric")
        fields[name] = int(value)
        observations = count
    if any(not isinstance(fields, dict) or set(fields) != FIXED_NATIVE_PIPELINE_METRICS
           or any(type(value) is not int or value < 0 for value in fields.values()) for fields in owners.values()) \
            or totals != owners:
        raise ValueError("fixed pipeline terminal totals differ from the actual exported device observations")
    return report


def bind_observed_pipeline_artifacts(exact: str, declared: dict, directory: Path) -> dict:
    """Bind one discovered candidate to its actual completed numerical artifact, without new discovery."""
    from run_qfn_source_admission import validate_observed_pipeline_receipts
    runtime = declared.get("runtime", {})
    request = runtime.get("observed_pipeline_request")
    if declared.get("inventory_role") != "numerical_diagnostic" or \
            runtime.get("configuration_policy") != "observed_pipeline_binding_required_v1" or \
            not isinstance(request, dict) or set(request) != {"backend", "native_isa", "mpi_ranks", "stage_sizes"}:
        raise ValueError("observed pipeline artifacts require one original unbound numerical declaration")
    rank_root = directory.resolve(strict=True) / "observed-pipeline"
    original = {"id": declared["id"], "mpi_ranks": request["mpi_ranks"], "participants": [],
                "observed_pipeline_request": {key: request[key] for key in ("backend", "native_isa", "stage_sizes")}}
    receipts = validate_observed_pipeline_receipts(original, lambda rank: rank_root / f"rank-{rank}",
        "parity_work_succeeded", inactive_artifacts=("observed-pipeline-config.json",))
    active = next(binding for binding in receipts.values() if binding["active"])
    path = rank_root / f"rank-{active['discovery_rank']}" / "observed-pipeline-config.json"
    if path.is_symlink():
        raise ValueError("observed pipeline configuration must be owned by its original active artifact directory")
    exported = json.loads(path.read_text())
    if exported.get("schema") != 1 or exported.get("exact_cell") != exact or exported.get("declaration") != declared:
        raise ValueError("observed pipeline configuration belongs to another original declaration")
    configuration = exported.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("observed pipeline omitted its lossless runtime configuration")
    unchanged_fields = set(declared) - {"runtime"}
    if any(configuration.get(field) != declared[field] for field in unchanged_fields) or any(
            configuration.get("runtime", {}).get(field) != value for field, value in runtime.items()
            if field not in ("configuration_policy", "observed_pipeline_request")):
        raise ValueError("observed pipeline changed the original source or inference policy")
    bound = {**configuration, "observed_pipeline_binding": active}
    require_observed_pipeline_document(bound)
    return bound


def runtime_server_arguments(record: dict, directory: Path, model: str) -> list[str]:
    """Materialize the declared document once for the existing public --config consumer.

    Source remapping must already have authenticated the original GGUF fields.
    The production parser remains the sole config-schema/policy validator; no
    participant ceilings are translated to unsupported CLI options or dropped.
    """
    runtime = record.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("canonical runtime is missing")
    if "configuration_document" not in runtime:
        if "configuration_policy" in runtime:
            raise ValueError("runtime representation has no declared document")
        arguments = runtime.get("server_args")
        if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
            raise ValueError("canonical execution argv must be a string list")
        return list(arguments)
    config = _runtime_document_configuration(record)
    if (not isinstance(config.get("model_path"), str)
            or Path(config["model_path"]).resolve() != Path(model).resolve()
            or config.get("shared_predictor") != record.get("shared_predictor")):
        raise ValueError("staged runtime document selected a foreign GGUF source")
    path = directory / "runtime-config.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(runtime["configuration_document"], stream, sort_keys=True, indent=2)
        stream.write("\n")
    return ["--config", str(path.resolve())]


def parse_parameters(output: str) -> dict[str, dict]:
    """Read complete typed parameters; console output can truncate these values."""
    records = {}
    for suite in json.loads(output)["testsuites"]:
        for test in suite["testsuite"]:
            if not test["name"].startswith("ProductionParity/"):
                continue
            exact = suite["name"] + "." + test["name"]
            record = json.loads(test["value_param"])
            if record.get("model_parity_schema") != 1:
                raise ValueError(f"unsupported parameter schema: {exact}")
            if "shared_predictor" in record:
                model_source_fields(record)
            InventoryScope.ALL.accepts(record)
            if exact in records:
                raise ValueError(f"duplicate parameter: {exact}")
            records[exact] = record
    return records


def source_revision(args: argparse.Namespace) -> str:
    """Containers without Git receive the revision of their admitted snapshot."""
    return getattr(args, "source_revision", None) or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def matrix_binary(campaign: parity.CampaignCell) -> Path:
    """Resolve one registered executable, independently of launcher topology."""
    binaries = [Path(arg) for arg in campaign.command
                if Path(arg).name.startswith("v2_integration_parity_")
                and Path(arg).name.endswith("_matrix")]
    if len(binaries) != 1:
        raise ValueError(f"ambiguous matrix executable: {campaign.name}")
    return binaries[0]


def discover_parameters(registered: list[parity.CampaignCell], campaigns: list[parity.CampaignCell]) -> dict[str, dict]:
    """Read original typed GoogleTest JSON once/binary and authenticate complete CTest membership.

    This metadata-only boundary does not stat/map model files, project routes,
    choose policies or derive topology from names. Numerical, HTTP and inventory
    consumers share this same reader rather than maintaining another parser.
    """
    registered_by_binary = {}
    registered_names = set()
    for campaign in registered:
        names = registered_by_binary.setdefault(matrix_binary(campaign), set())
        for exact in campaign.gtest_cases:
            if exact in registered_names:
                raise ValueError(f"duplicate canonical CTest cell in {campaign.name}: {exact}")
            registered_names.add(exact)
            names.add(exact)
    inventories = {}
    for campaign in campaigns:
        binary = matrix_binary(campaign)
        if binary not in inventories:
            with tempfile.TemporaryDirectory(prefix="llaminar-cell-discovery-") as directory:
                output = Path(directory) / "parameters.json"
                subprocess.run(
                    [str(binary), "--gtest_list_tests", f"--gtest_output=json:{output}"],
                    cwd=campaign.working_directory,
                    env={**os.environ, "LLAMINAR_FORCE_CPU_ONLY_STARTUP": "1",
                         "HWLOC_COMPONENTS": "-gl,-opencl"},
                    check=True, capture_output=True, text=True, timeout=30)
                inventories[binary] = parse_parameters(output.read_text())
            actual = set(inventories[binary])
            expected = registered_by_binary[binary]
            if actual != expected:
                raise ValueError(f"CTest/GoogleTest inventory mismatch for {binary}: "
                                 f"unregistered={sorted(actual - expected)} "
                                 f"missing_from_binary={sorted(expected - actual)}")
    return {exact:record for inventory in inventories.values() for exact,record in inventory.items()}


def discover(args: argparse.Namespace, scope: InventoryScope) -> list[tuple[parity.CampaignCell, str, dict]]:
    """Join exact CTest membership to typed metadata once per matrix executable.

    A saved manifest declares whether it covers all cells or only E2E tags.
    An E2E-only manifest must never masquerade as the full generation matrix.
    Model stat identity handles mount aliases without reading GGUF payloads.
    """
    if not isinstance(scope, InventoryScope):
        raise TypeError("inventory scope must be explicit and typed")
    selected = []
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        if manifest.get("schema") != 1 or manifest.get("source_revision") != source_revision(args):
            raise ValueError("canonical manifest must be generated from this source revision")
        # Existing E2E manifests predate the general projection. They remain
        # valid E2E inputs, but cannot assert coverage of untagged cells.
        declared = InventoryScope(manifest.get("scope", "e2e"))
        if declared == InventoryScope.LOCAL_DIAGNOSTIC and scope != InventoryScope.LOCAL_DIAGNOSTIC:
            raise ValueError("a local diagnostic manifest cannot become a canonical runtime inventory")
        if scope == InventoryScope.ALL and declared != InventoryScope.ALL:
            raise ValueError("the full matrix requires an all-cell manifest")
        if scope == InventoryScope.GENERATION and declared not in (InventoryScope.ALL, InventoryScope.GENERATION):
            raise ValueError("generation requires the full inventory or its explicit generation projection")
        seen = set()
        for row in manifest["cells"]:
            exact = row["case"]
            if exact in seen:
                raise ValueError(f"duplicate canonical manifest cell: {exact}")
            seen.add(exact)
            record = row["configuration"]
            if record.get("model_parity_schema") != 1 or not declared.accepts(record):
                raise ValueError(f"invalid configuration in {declared.value} manifest: {exact}")
            if any(path not in row.get("model_files", [])
                   for path in model_source_fields(record).values()):
                raise ValueError(f"canonical manifest omitted a declared GGUF source: {exact}")
            if (scope.accepts(record) and re.fullmatch(args.backend, row["backends"])
                    and re.fullmatch(args.campaign, row["campaign"])
                    and re.fullmatch(args.cell, exact)):
                campaign = parity.CampaignCell(row["campaign"],
                    parity.CampaignGroup(row["backends"], "ALL"),
                    model_files=tuple(row["model_files"]))
                selected.append((campaign, exact, record))
    else:
        # CTest membership chooses which binaries can contain the requested
        # cells. An exact diagnostic must not initialize unrelated matrices.
        # Still audit the complete registration of each chosen binary below,
        # including siblings excluded by the cell/backend/campaign selectors;
        # otherwise a filtered diagnostic could conceal a dropped CTest case.
        registered = parity.discover_campaigns(args.build_dir)
        campaigns = [campaign for campaign in registered
                     if re.fullmatch(args.backend, campaign.group.backends)
                     and re.fullmatch(args.campaign, campaign.name)
                     and any(re.fullmatch(args.cell, exact)
                             for exact in campaign.gtest_cases)]
        parameters=discover_parameters(registered,campaigns)
        for campaign in campaigns:
            for exact in campaign.gtest_cases:
                if exact not in parameters:
                    raise ValueError(f"stale CTest registration, rebuild {matrix_binary(campaign)}: {exact}")
                record = parameters[exact]
                if not scope.accepts(record) or not re.fullmatch(args.cell, exact):
                    continue
                # Keep the target and explicit predictor together, including
                # MTP-off controls. Both use the same admitted shard/identity
                # authority; unrelated model families remain excluded.
                source_matches = {}
                for role, source in model_source_fields(record).items():
                    matches = [path for path in campaign.model_files if Path(source).samefile(path)]
                    if not matches:
                        raise ValueError(f"{role} is not in the canonical GGUF manifest: {exact}")
                    source_matches[role] = matches[0]
                model_campaign = dataclasses.replace(campaign, model_files=tuple(source_matches.values()))
                model_campaign = dataclasses.replace(model_campaign, model_files=tuple(
                    str(path) for path in parity.selected_model_files([model_campaign])))
                # Shard expansion resolves mount aliases for every source role.
                record = {**record, **{role: next(path for path in model_campaign.model_files
                          if Path(path).samefile(source)) for role, source in source_matches.items()}}
                selected.append((model_campaign, exact, record))
    if not selected:
        raise ValueError(f"no {scope.value} canonical cells selected")
    return selected


def export_manifest(selected: list, revision: str, scope: InventoryScope) -> dict:
    """Retain the projection's scope and full configurations for other drivers."""
    if not selected or len({exact for _, exact, _ in selected}) != len(selected):
        raise ValueError("cannot export an empty or duplicate canonical inventory")
    for _, exact, record in selected:
        if not scope.accepts(record):
            raise ValueError(f"cell is outside the exported scope: {exact}")
    return {"schema": 1, "source_revision": revision, "scope": scope.value,
            "cells": [{"campaign": campaign.name, "backends": campaign.group.backends,
                       "case": exact, "configuration": record,
                       "model_files": list(campaign.model_files)}
                      for campaign, exact, record in selected]}


def main(argv: list[str] | None = None) -> int:
    """List/export metadata only; this command cannot run or certify inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build_v2_integration")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scope", type=InventoryScope, choices=list(InventoryScope), default=InventoryScope.ALL)
    parser.add_argument("--export-manifest", type=Path)
    parser.add_argument("--observed-pipeline-artifacts", type=Path,
                        help="Bind one exact local numerical cell to its completed original-rank artifacts")
    parser.add_argument("--source-revision")
    parser.add_argument("--backend", default=".*")
    parser.add_argument("--campaign", default=".*")
    parser.add_argument("--cell", default=".*")
    args = parser.parse_args(argv)
    selected = discover(args, args.scope)
    if args.observed_pipeline_artifacts is not None:
        if args.manifest or args.scope != InventoryScope.LOCAL_DIAGNOSTIC or len(selected) != 1:
            parser.error("observed pipeline binding requires one original local diagnostic cell")
        campaign, exact, record = selected[0]
        selected = [(campaign, exact, bind_observed_pipeline_artifacts(exact, record, args.observed_pipeline_artifacts))]
    if args.export_manifest:
        write_json(args.export_manifest, export_manifest(selected, source_revision(args), args.scope))
    for _, exact, record in selected:
        if args.scope == InventoryScope.CROSS_HOST_E2E:
            for remote in record["cross_host_e2e"]:
                print(remote["id"])
        else:
            print(exact)
    if args.scope == InventoryScope.CROSS_HOST_E2E:
        count = sum(len(record["cross_host_e2e"]) for _, _, record in selected)
        print(f"[model-parity-inventory] scope={args.scope.value} cells={count} source_cells={len(selected)}")
    else:
        print(f"[model-parity-inventory] scope={args.scope.value} cells={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
