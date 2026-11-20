"""Compare ordinary and dynamic MTP in one retained 122B CUDA/CPU benchmark runner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from check_automatic_moe_tp2_plan import require, verify_plan


PROMPT = (
    "<|im_start|>user\nIn Python, what does len([1, 2, 3]) return? Answer briefly."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n"
)
PROMPT_IDS = [
    248045, 846, 198, 623, 12654, 11, 1092, 1503, 2343, 2477,
    16, 11, 220, 17, 11, 220, 18, 2387, 460, 30,
    21134, 25899, 13, 248046, 198, 248045, 74455, 198, 248068, 198,
]
SAMPLING = {"temperature": 0, "top_k": 1, "top_p": 1, "seed": 42}
GENERATED_TOKENS = 320
SENTINEL_TOKENS = 269
POLICIES = {
    "ordinary": {"mode": "off"},
    "grouped": {"mode": "dynamic", "minimum_depth": 1, "maximum_depth": 15, "initial_depth": 0},
}


def read_json(path):
    """Read one complete native receipt."""
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def verify_results(results, model, campaign_digest):
    """Bind both native cells to one campaign and compare all committed tokens."""
    summary = read_json(results / "summary.json")
    require(summary["schema"] == "llaminar.benchmark-campaign-result.v1"
            and summary["success"] and summary["complete"]
            and summary["cell_count"] == summary["completed_cells"] == len(POLICIES)
            and summary["document_sha256"] == campaign_digest,
            "native campaign did not complete both requested cells")
    require(summary["cells"] == [
        {"id": cell, "file": f"{cell}.json", "success": True} for cell in POLICIES
    ], "native summary does not name the exact two evidence files")

    records = {}
    for index, (cell, policy) in enumerate(POLICIES.items()):
        row = records[cell] = read_json(results / f"{cell}.json")
        campaign = row["campaign"]
        require(campaign["schema"] == "llaminar.benchmark-campaign.v1"
                and campaign["document_sha256"] == campaign_digest
                and campaign["cell_id"] == cell and campaign["cell_index"] == index
                and campaign["prompt_id"] == "exact-chat" and campaign["partition"] == "holdout",
                f"{cell} receipt belongs to a different campaign, prompt, or cell")
        require(row["architecture"] == "qwen35moe" and row["success"]
                and row["prefill_success"] and row["decode_success"]
                and row["measurement_iterations"] == 1 and row["warmup_iterations"] == 0,
                f"{cell} did not complete the original measured model request")
        require(row["config"]["sampling"] == SAMPLING
                and row["prompt"]["token_ids"] == PROMPT_IDS
                and row["prompt"]["sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest(),
                f"{cell} prompt history or sampling changed")
        ids = row["generated_token_ids"]
        require(len(ids) >= SENTINEL_TOKENS, f"{cell} ended before the 269-token regression boundary")
        require(row["tokens"]["prefill"] == len(PROMPT_IDS)
                and len(ids) == row["tokens"]["decode"] == GENERATED_TOKENS
                and all(type(token) is int and token >= 0 for token in ids),
                f"{cell} did not retain all 320 valid committed token IDs")

        verify_plan(row["runtime_configuration"], model, "mixed", "embedded", maintenance="Off")
        require(row["runtime_state"]["execution_path"] == "rank-orchestrator"
                and row["runtime_state"]["moe_runtime_movement_epoch"] == 1,
                f"{cell} did not retain its initial fixed placement (zero moves)")
        actual = row["request_policy"]
        mtp = row["mtp"]
        active = cell == "grouped"
        require(actual["mode"] == policy["mode"] and actual["enabled"] is active
                and actual["verify_mode"] == "greedy" and mtp["config_enabled"] is active,
                f"{cell} request did not execute its assigned MTP policy")
        if active:
            require((actual["minimum_depth"], actual["maximum_depth"], actual["initial_depth"]) == (1, 15, 0)
                    and mtp["request"]["adaptive_depth_enabled"]
                    and mtp["verifier_runs"] == mtp["transaction_commits"] > 0
                    and mtp["accepted_tokens"] > 0 and mtp["draft_steps"] > 0
                    and mtp["transaction_validation_failures"] == 0 and not mtp["bypassed"],
                    "dynamic MTP did not commit actual device verifier work")
        else:
            require(all(mtp[counter] == 0 for counter in (
                "verifier_runs", "verifier_token_count", "draft_steps", "accepted_tokens", "transaction_commits",
            )), "ordinary arm submitted speculative verifier work")

    ordinary = records["ordinary"]
    grouped = records["grouped"]
    require(ordinary["runtime_configuration_sha256"] == grouped["runtime_configuration_sha256"]
            and ordinary["runtime_configuration"] == grouped["runtime_configuration"],
            "the two requests did not use the same retained runtime configuration")
    differences = (i for i, pair in enumerate(zip(
        ordinary["generated_token_ids"], grouped["generated_token_ids"], strict=True
    )) if pair[0] != pair[1])
    first = next(differences, None)
    require(first is None, f"dynamic MTP diverged from the ordinary token at zero-based index {first}")
    return grouped["mtp"]["verifier_runs"]


def main():
    """Run a single public campaign under the configured, owned two-rank MPI session."""
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("binary", "model", "mpi-session", "mpiexec", "evidence-dir"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence = Path(tempfile.mkdtemp(prefix="fixed-placement-122b-", dir=args.evidence_dir.resolve()))
    results = evidence / "results"
    campaign = evidence / "campaign.json"
    campaign.write_text(json.dumps({
        "schema": "llaminar.benchmark-campaign.v1",
        "output_directory": str(results),
        "sampling": SAMPLING,
        "prompts": [{"id": "exact-chat", "partition": "holdout", "text": PROMPT}],
        "cells": [{"id": cell, "prompt_id": "exact-chat", "mtp": policy} for cell, policy in POLICIES.items()],
    }), encoding="utf-8")
    digest = hashlib.sha256(campaign.read_bytes()).hexdigest()
    print(f"122B campaign evidence at {evidence}", flush=True)
    with tempfile.TemporaryDirectory(prefix="runtime-", dir=evidence) as scratch:
        cache = Path(scratch) / "cache"
        sessions = Path(scratch) / "mpi"
        sessions.mkdir()
        command = [
            str(args.mpi_session.resolve(strict=True)), "--session-base", str(sessions), "--",
            str(args.mpiexec.resolve(strict=True)), "-np", "2", "--bind-to", "socket", "--map-by", "socket",
            "--mca", "mpi_leave_pinned", "1", "--mca", "orte_allowed_exit_without_sync", "1", "--",
            str(args.binary.resolve(strict=True)), "benchmark", "--auto", "--mpi-procs", "2",
            "--only-backends", "cuda,cpu", "--only-strategies", "expert-overlay",
            "--auto-device-counts", "cuda=2,cpu=2", "-m", str(model), "--plan-workload", "2048,256",
            "--prefix-cache-disk-dir", str(cache), "--moe-residency-maintenance", "off",
            "--temperature", "0", "--top-k", "1", "--top-p", "1", "--seed", str(SAMPLING["seed"]),
            "-n", str(GENERATED_TOKENS), "--benchmark-campaign", str(campaign),
        ]
        environment = dict(os.environ, LLAMINAR_BENCHMARK_ITERATIONS="1",
                           LLAMINAR_BENCHMARK_WARMUP_ITERATIONS="0")
        (evidence / "argv.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        subprocess.run(command, env=environment, check=True)
        verifier_runs = verify_results(results, model, digest)
    print(f"Matched {GENERATED_TOKENS} committed tokens, verifier runs 0 vs {verifier_runs}, placement epoch 1.")


if __name__ == "__main__":
    main()
