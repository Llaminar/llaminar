"""Replay the public MoE planner against provisioned GGUFs on CUDA TP2 or CUDA TP2 plus CPU TP2."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def require(condition, message):
    """Keep failures explicit even under Python's optimized mode."""
    if not condition:
        raise RuntimeError(message)


def verify_plan(document, model, topology, predictor, *, maintenance="Dynamic"):
    """Check the saved Apply document and the actual execution domain it selected."""
    require(document["kind"] == "llaminar.orchestration-config", "wrong output document kind")
    cfg = document["configuration"]
    require(cfg["planning_mode"] == "Apply", "planner did not publish an Apply configuration")
    require(Path(cfg["model_path"]).resolve() == model, "planner selected another source")
    require(cfg["shared_predictor"] is None, "planner unexpectedly selected a sidecar")
    embedded = predictor == "embedded"
    require(cfg["mtp_activation_policy"] == ("Enabled" if embedded else "Disabled")
            and cfg["mtp"]["enabled"] == embedded, "automatic MTP did not follow the source")
    if embedded:
        require(cfg["mtp"]["depth_policy"]["mode"] == "Dynamic",
                "automatic admission did not select dynamic depth")
    require(cfg["moe_rebalance"]["mode"] == maintenance, "unexpected residency maintenance policy")
    require(cfg["routed_expert_compute_policy"] == "TensorSharded", "default expert layout changed")
    plan = cfg["moe_routed_expert_plan"]
    mixed = topology == "mixed"
    expected_domains = 2 if mixed else 1
    require(plan["enabled"] and len(plan["domains"]) == len(plan["routed_tiers"]) == expected_domains,
            "unexpected routed cohort count")
    require(plan["authority_execution"] == ("HostResident" if mixed else "DeviceResident"),
            "routed authority does not match the execution topology")
    domains = {domain["name"]: domain for domain in plan["domains"]}
    tiers = plan["routed_tiers"]
    gpu = domains[plan["continuation_domain"]]
    require(gpu["name"] == plan["shared_expert_domain"], "shared experts left the CUDA continuation")
    require(plan["continuation_domain_spec"]["dense_policy"] == "TensorParallel",
            "expected dense TP on the CUDA continuation")
    require(gpu["scope"] == "RANK_LOCAL" and gpu["routed_compute_policy"] == "TensorSharded",
            "expected a rank-local sharded CUDA domain")
    devices = gpu["participants"]
    require(len(devices) == 2 and all(device["device_type"] == "CUDA" for device in devices)
            and len({device["device_ordinal"] for device in devices}) == 2,
            "expected two distinct CUDA endpoints")
    host = devices[0]["hostname"]
    require(host and {device["hostname"] for device in devices} == {host}
            and devices[0]["numa_node"] >= 0
            and {device["numa_node"] for device in devices} == {devices[0]["numa_node"]},
            "CUDA endpoints do not share one original host and NUMA node")
    require(tiers[0]["domain"] == gpu["name"] and tiers[0]["fallback"] == (not mixed),
            "CUDA tier must be first and the sole fallback only in the single-tier plan")
    transport = "MappedSparse" if mixed else "NativeCollective"
    require(plan["observed_route_transport"] == {"decode": transport, "ordinary_prefill": transport},
            "routing transport did not match the original cohorts")
    if mixed:
        cpu = domains[tiers[1]["domain"]]
        require(cpu["name"] != gpu["name"] and tiers[1]["fallback"]
                and cpu["scope"] == "NODE_LOCAL" and cpu["routed_compute_policy"] == "TensorSharded"
                and set(cpu["world_ranks"]) == {0, 1} and len(cpu["world_ranks"]) == 2,
                "expected a separate node-local sharded CPU tier with one member per MPI rank")
        endpoints = cpu["participants"]
        require(len(endpoints) == 2 and all(endpoint["device_type"] == "CPU"
                and endpoint["hostname"] == host and endpoint["numa_node"] >= 0 for endpoint in endpoints)
                and len({endpoint["numa_node"] for endpoint in endpoints}) == 2
                and devices[0]["numa_node"] in {endpoint["numa_node"] for endpoint in endpoints},
                "CPU tier must cover both original NUMA nodes on the continuation host")


def main():
    """Run plan-only in the owning CTest MPI process tree and read its saved document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--topology", required=True, choices=("cuda", "mixed"))
    parser.add_argument("--predictor", required=True, choices=("absent", "embedded"))
    args = parser.parse_args()
    mixed = args.topology == "mixed"
    model = args.model.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="llaminar-moe-auto-tp2-") as root:
        selected = Path(root) / "selected.json"
        subprocess.run([
            "mpirun", "-np", "2", "--bind-to", "socket", "--map-by", "socket",
            "--mca", "mpi_leave_pinned", "1", "--mca", "orte_allowed_exit_without_sync", "1", "--",
            str(args.binary.resolve(strict=True)), "plan", "--auto", "--mpi-procs", "2",
            "--only-backends", "cuda,cpu" if mixed else "cuda",
            "--auto-device-counts", "cuda=2,cpu=2" if mixed else "cuda=2",
            "-m", str(model), "--plan-workload", "2048,256", "-o", str(selected),
        ], check=True)
        with selected.open(encoding="utf-8") as receipt:
            verify_plan(json.load(receipt), model, args.topology, args.predictor)
    print(f"Verified {args.topology} tensor-sharded Apply configuration with {args.predictor} predictor.")


if __name__ == "__main__":
    main()
