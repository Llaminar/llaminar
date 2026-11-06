#!/usr/bin/env python3
"""Saved QFN PLE input provenance and terminal-evidence consistency for pooled replay.

This extends the pooled-ngram trace tooling, not cache execution. Export reads
canonical GGUF descriptors and already-published integer snapshots; it never
loads weights, runs a model, invents future/rejected MTP work, or emits timings.
Native cache execution and source/receipt authentication remain owning native
fixture obligations. A consistent counter document alone is not a certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

SCHEMA = 1
HEADS = 16
COUNTERS = (
    "read_original_ids", "local_hit_original_ids", "all_hit_wait_avoided_original_ids",
    "consumed_original_ids", "future_on_time_native_rows", "future_late_native_rows",
    "future_useful_native_rows", "future_wasted_native_rows",
)
MODES = {"consumer_local", "shared_only", "combined"}


def require(condition: bool, detail: str) -> None:
    """Reject incomplete trace/evidence without substituting another workload."""
    if not condition:
        raise ValueError(detail)


def natural(value, name: str, *, positive: bool = False) -> int:
    """Read an exact bounded integer; booleans and floating counts are not IDs."""
    require(type(value) is int and (value > 0 if positive else value >= 0)
            and value <= (1 << 64) - 1, f"invalid {name}")
    return value


def load_document(path: Path) -> dict:
    """Reject duplicate JSON authority fields rather than accepting last-write wins."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON field {key}")
            result[key] = value
        return result
    result = json.loads(path.read_text(), object_pairs_hook=unique)
    require(isinstance(result, dict), "expected trace/config object")
    return result


def sha256(path: Path) -> str:
    """Bind small input metadata/snapshots; never hash the full model payload."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_trace(trace: dict) -> dict:
    """Validate real saved rows separately from explicitly authored adversarial controls.

    Current/consumed describes reference evaluation only. It grants no native
    consumption, cache-hit, promotion, future-work or performance credit.
    """
    require(trace.get("schema") == SCHEMA, "unsupported QFN trace schema")
    origin = trace.get("origin")
    require(origin in {"saved_pytorch_ple_indices", "adversarial_metadata_control"}, "unknown trace origin")
    require(trace.get("native_execution_proven") is False and trace.get("timing_qualified") is False,
            "saved input traces cannot certify native execution or timing")
    source = trace["source"]
    rows = natural(source["rows"], "source rows", positive=True)
    natural(source["row_elements"], "source row width", positive=True)
    natural(source["native_row_bytes"], "native row bytes", positive=True)
    require(bool(source["tensor"]) and bool(source["format"]), "missing source descriptor")
    maximum = natural(trace["maximum_original_ids"], "retained query bound", positive=True)
    require(maximum % HEADS == 0, "retained full-query bound must cover complete sixteen-head rows")
    histories = {h["id"]: h for h in trace["histories"]}
    require(len(histories) == len(trace["histories"]) and histories, "duplicate/empty history identity")
    for history in histories.values():
        for name in ("prompt_ids", "continuation_ids"):
            require(isinstance(history[name], list) and history[name], "missing original token history")
            for token in history[name]:
                natural(token, "token")
    seen = set()
    for frame in trace["frames"]:
        require(frame["id"] not in seen, "duplicate frame identity")
        seen.add(frame["id"])
        history = histories[frame["history_id"]]
        phase = frame["phase"]
        require(phase in {"prefill", "decode"}, "unsupported original phase")
        step = natural(frame["step"], "phase step")
        if phase == "prefill":
            require(step == 0, "prefill has no decode step")
            expected = history["prompt_ids"]
        else:
            require(step < len(history["continuation_ids"]), "decode input missing from immutable teacher history")
            expected = [history["continuation_ids"][step]]
        require(frame["tokens"] == expected, "frame changed original input tokens")
        natural(frame["layer"], "PLE source layer")
        ids = frame["ids"]
        require(len(ids) == len(expected) * HEADS and len(ids) <= maximum,
                "live hash rows disagree with original tokens or retained query bound")
        for row in ids:
            require(natural(row, "source row") < rows, "hash row exceeds exact source extent")
        require(frame["intent"] in {"current", "future"} and
                frame["outcome"] in {"consumed", "cancelled", "discarded"}, "unknown input intent/outcome")
        if origin == "saved_pytorch_ple_indices":
            require(frame["intent"] == "current" and frame["outcome"] == "consumed",
                    "saved reference does not observe speculative or cancelled MTP work")
            require(len(frame.get("snapshot_sha256", "")) == 64 and bool(frame.get("snapshot_path")),
                    "saved rows lack their original snapshot provenance")
    require(seen, "empty trace")
    coverage = trace["unsupported_coverage"]
    if origin == "saved_pytorch_ple_indices":
        require({"native_execution", "future_read_ahead", "cancelled_mtp", "timing"}.issubset(coverage),
                "reference trace omitted unsupported native/speculative coverage")
    return {"frames": len(seen), "histories": len(histories),
            "original_ids": sum(len(f["ids"]) for f in trace["frames"]),
            "scope": "input_and_consistency_only", "native_execution_proven": False,
            "timing_qualified": False}


def validate_terminal_metrics(trace: dict, evidence: dict) -> dict:
    """Check actual terminal fields without simulating residency or predicting hits.

    Evidence adapters must obtain these values from completed production counters
    and PMA declarations. This checks arithmetic, bounded staging and demand
    credit only, and deliberately never returns a native/performance certificate.
    """
    validate_trace(trace)
    require(evidence.get("schema") == SCHEMA and evidence.get("scope") == "terminal_consistency_only",
            "unsupported terminal evidence scope")
    by_id = {f["id"]: f for f in trace["frames"]}
    seen = set()
    row_bytes = trace["source"]["native_row_bytes"]
    capacity = trace["maximum_original_ids"] * row_bytes
    for sample in evidence["frames"]:
        key = (sample["mode"], sample["consumer"], sample["frame_id"])
        require(key not in seen and sample["mode"] == "combined",
                "LocalHot terminal counters apply only to the actual combined path")
        seen.add(key)
        frame = by_id[sample["frame_id"]]
        counters = sample["counter_deltas"]
        require(set(counters) == set(COUNTERS), "terminal counter fields are incomplete or renamed")
        for name, value in counters.items():
            natural(value, name)
        original = len(frame["ids"])
        unique = len(set(frame["ids"]))
        require(counters["read_original_ids"] <= original and
                counters["local_hit_original_ids"] <= counters["read_original_ids"] and
                counters["all_hit_wait_avoided_original_ids"] <= counters["local_hit_original_ids"],
                "local hit/original-ID accounting is inconsistent")
        consumed = frame["outcome"] == "consumed"
        require(not consumed or counters["read_original_ids"] == original, "consumed input lacks its original complete read")
        require(counters["consumed_original_ids"] == (original if consumed else 0),
                "unconsumed work received demand credit or consumed work lost credit")
        fetched = natural(sample["fetched_native_rows"], "fetched native rows")
        payload = natural(sample["native_payload_bytes"], "native payload bytes")
        require(fetched <= unique and payload == fetched * row_bytes,
                "wire extent includes capacity/padding or duplicates beyond actual unique rows")
        peak = natural(sample["peak_response_bytes"], "peak response bytes")
        response_capacity = natural(sample["response_capacity_bytes"], "native response capacity")
        require(payload <= peak <= capacity and response_capacity == capacity,
                "response staging differs from its single admitted bounded native frame")
        require(counters["future_useful_native_rows"] <=
                counters["future_on_time_native_rows"] + counters["future_late_native_rows"],
                "useful future rows lack their completion/adoption classification")
        require(counters["future_useful_native_rows"] + counters["future_wasted_native_rows"] <= fetched,
                "future native-row credit exceeds actual fetched extent")
        if frame["intent"] == "current":
            require(all(counters[k] == 0 for k in COUNTERS if k.startswith("future_")),
                    "ordinary current demand masquerades as read-ahead")
        if not consumed:
            require(not sample["promotions"] and not sample["evictions"] and
                    not counters["future_useful_native_rows"], "speculative work changed hot policy or earned useful credit")
        natural(sample["promotions"], "completed promotions")
        natural(sample["evictions"], "completed evictions")
    require(seen, "empty terminal evidence")
    return {"samples": len(seen), "scope": "terminal_consistency_only", "timing_qualified": False}


def validate_equal_budgets(configurations: list[dict]) -> None:
    """Require identical authored PMA ceilings per original allocator across modes.

    Complete mandatory+optional BOM bytes may differ. They must fit the same
    original capacity for each CPU/GPU, not merely an equal cross-device sum.
    """
    require(len(configurations) == 3 and {x["mode"] for x in configurations} == MODES,
            "matched comparison requires ConsumerLocal, shared-only and combined")
    reference = None
    for config in configurations:
        limits = {}
        for resource in config["pma_resources"]:
            key = (natural(resource["world_rank"], "world rank"), resource["device"])
            require(key not in limits and resource["device"], "duplicate/foreign PMA allocator")
            capacity = natural(resource["capacity_bytes"], "PMA physical budget", positive=True)
            require(natural(resource["complete_bom_bytes"], "complete PMA BOM") <= capacity,
                    "complete mandatory/optional BOM exceeds original capacity")
            limits[key] = capacity
        require(limits, "missing original PMA resources")
        if reference is None:
            reference = limits
        require(limits == reference, "comparison changed original allocator populations or physical budgets")


def metadata(path: Path) -> dict:
    """Read the existing snapshot's scalar identity without accepting duplicates."""
    result = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        require(key not in result, f"duplicate snapshot metadata {key}")
        result[key] = value.strip()
    return result


def describe_source(source_root: Path, model: Path, checkpoint_revision: str) -> dict:
    """Read original native format geometry through the canonical GGUF parser.

    The parser owns format/block arithmetic. No conservative size estimator or
    duplicate exporter format table can authorize a saved physical row stride.
    Only metadata is parsed; weight bytes are not decoded, copied or hashed.
    """
    parser_path = source_root / "python/reference/loaders/gguf_parser.py"
    spec = importlib.util.spec_from_file_location("qfn_trace_gguf_parser", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with module.GGUFParser(model) as parser:
        parser.parse()
        tensors = [x for x in parser.tensors if x.name == "per_layer_token_embd.weight"]
        require(len(tensors) == 1 and len(tensors[0].shape) == 2, "missing original PLE source tensor")
        tensor = tensors[0]
        rows, width = tensor.shape
        total_bytes = parser._calculate_tensor_size(tensor)
        require(total_bytes % rows == 0, "native source row extent is not exact")
        source = {"tensor": tensor.name, "rows": rows, "row_elements": width,
                  "format": tensor.type.name, "native_row_bytes": total_bytes // rows,
                  "model_path": str(model), "checkpoint_revision": checkpoint_revision}
    return source


def export_trace(source_root: Path, manifest_path: Path, reference: Path,
                 history_id: str, layer: int, maximum: int) -> dict:
    """Import one real saved full-stage history using canonical source descriptors.

    GGUFParser parses split headers only; no tensor-data reader is invoked. The
    snapshot metadata must bind the exact original prompt/teacher/source before
    integer row arrays can become input traces. Missing captures are errors.
    """
    import numpy as np
    manifest = load_document(manifest_path)
    matches = [h for h in manifest["histories"] if h["id"] == history_id]
    require(len(matches) == 1, "missing/duplicate frozen history")
    history = matches[0]
    meta_path = reference / "metadata.txt"
    meta = metadata(meta_path)
    prompt = history["prompt_ids"]
    teachers = history["continuation_ids"]
    decode_steps = natural(history["decode_rows"], "declared decode rows")
    require([int(x) for x in meta["token_ids"].split(",")] == prompt and
            [int(x) for x in meta["decode_tokens"].split(",")] == teachers and
            int(meta["decode_steps"]) == decode_steps and
            meta["qwen4exp_frozen_history_id"] == history_id, "snapshot changed frozen prompt/teacher identity")
    model = Path(manifest["source"]["shards"][0]["path"])
    require(Path(meta["Model"]).resolve() == model.resolve() and
            meta["qwen4exp_checkpoint_revision"] == manifest["source"]["checkpoint_revision"],
            "snapshot changed original model/checkpoint identity")
    source = describe_source(source_root, model, manifest["source"]["checkpoint_revision"])
    frames = []
    for step in range(decode_steps + 1):
        prefill = step == 0
        snapshot = reference / (("" if prefill else f"decode_step{step-1}_") + f"layer{layer}_PLE_HASH_INDICES.npy")
        values = np.load(snapshot, allow_pickle=False)
        tokens = prompt if prefill else [teachers[step-1]]
        require(values.dtype == np.dtype("int64") and values.shape == (1, len(tokens), HEADS),
                f"invalid original integer PLE snapshot {snapshot}")
        frames.append({"id": f"{history_id}:{'prefill' if prefill else 'decode'}:{0 if prefill else step-1}:layer{layer}",
                       "history_id": history_id, "phase": "prefill" if prefill else "decode",
                       "step": 0 if prefill else step-1, "layer": layer, "tokens": tokens,
                       "ids": values.reshape(-1).tolist(), "intent": "current", "outcome": "consumed",
                       "snapshot_path": str(snapshot.resolve()), "snapshot_sha256": sha256(snapshot)})
    trace = {"schema": SCHEMA, "origin": "saved_pytorch_ple_indices", "source": source,
             "native_execution_proven": False, "timing_qualified": False,
             "maximum_original_ids": maximum, "histories": [{k: history[k] for k in ("id", "prompt_ids", "continuation_ids")}],
             "frames": frames,
             "provenance": {"manifest_path": str(manifest_path.resolve()), "manifest_sha256": sha256(manifest_path),
                            "snapshot_metadata_path": str(meta_path.resolve()), "snapshot_metadata_sha256": sha256(meta_path),
                            "frozen_corpus_digest": meta["qwen4exp_frozen_corpus_digest"],
                            "frozen_source_digest": meta["qwen4exp_frozen_source_digest"]},
             "unsupported_coverage": ["native_execution", "future_read_ahead", "cancelled_mtp", "timing",
                                      "warm_changed_conversation", "prefill_scan_pollution", "ordinary_mtp_reset"]}
    validate_trace(trace)
    return trace


def verify_saved_files(trace: dict) -> None:
    """Authenticate the external small input files/rows, never model weights.

    Adversarial metadata controls deliberately cannot be authenticated as real
    QFN traces. Full native provenance remains the existing runtime observer's
    responsibility, not a claim made by this offline input document.
    """
    import numpy as np
    validate_trace(trace)
    require(trace["origin"] == "saved_pytorch_ple_indices", "adversarial controls are not saved QFN input evidence")
    provenance = trace["provenance"]
    for stem in ("manifest", "snapshot_metadata"):
        require(sha256(Path(provenance[stem + "_path"])) == provenance[stem + "_sha256"],
                f"changed original {stem} input")
    manifest = load_document(Path(provenance["manifest_path"]))
    actual_source = describe_source(Path(__file__).resolve().parents[5],
        Path(manifest["source"]["shards"][0]["path"]), manifest["source"]["checkpoint_revision"])
    require(trace["source"] == actual_source, "trace changed original native source descriptor/row bytes")
    original = {h["id"]: h for h in manifest["histories"]}
    for history in trace["histories"]:
        require(all(original[history["id"]][key] == history[key]
                    for key in ("prompt_ids", "continuation_ids")), "trace changed original manifest token history")
    for frame in trace["frames"]:
        path = Path(frame["snapshot_path"])
        require(sha256(path) == frame["snapshot_sha256"], "changed original integer snapshot")
        values = np.load(path, allow_pickle=False)
        require(values.dtype == np.dtype("int64") and values.shape == (1, len(frame["tokens"]), HEADS)
                and values.reshape(-1).tolist() == frame["ids"], "trace altered original PLE hash rows")


def campaign_from_selection(selection, history_ids: list[str], output_directory: Path,
                            *, repetitions: int = 2, seed: int | None = None) -> dict:
    """Project authenticated original natural histories into the existing campaign.

    This is metadata only. The public campaign runner tokenizes each natural
    prompt once and verifies these exact expected IDs before its first request.
    Repeated cells reuse that prompt; neither teacher tokens nor cache hits are
    injected. One selection fixes the source/configuration/arithmetic policy.
    An explicitly supplied positive seed fixes the benchmark sampling cohort;
    the public runtime must receive the same seed. An unspecified source seed
    is rejected rather than emitted as an unusable or random campaign.
    """
    require(type(repetitions) is int and 1 <= repetitions <= 100, "invalid campaign repetitions")
    require(history_ids and len(set(history_ids)) == len(history_ids), "duplicate/empty selected history")
    manifest = selection.corpus["manifest"]
    exact = selection.exact
    require(selection.topology["canonical_cell"] == exact and ".ProductionParity/" in exact,
            "campaign changed original source case")
    histories = {h["id"]: h for h in manifest["histories"]}
    require(len(histories) == len(manifest["histories"]), "duplicate original history provenance")
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # Match the existing corpus owner's canonical digest, not a payload digest.
    prompts = []
    for history_id in history_ids:
        require(history_id in histories, "unknown original history")
        history = histories[history_id]
        require(all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", history[k])
                    for k in ("id", "context_group")), "invalid original history identity")
        tokens = history["prompt_ids"]
        require(type(tokens) is list and tokens and all(type(t) is int and 0 <= t <= 2147483647 for t in tokens),
                "invalid original expected token IDs")
        require(type(history["prompt"]) is str and history["prompt"], "missing original natural text")
        prompts.append(dict(id=history_id, partition="holdout", text=history["prompt"], frozen_history=dict(
            schema="llaminar.benchmark-frozen-history.v1", corpus_manifest_digest=digest(manifest),
            source_digest=digest(manifest["source"]), canonical_cell=exact, history_id=history_id,
            context_group=history["context_group"], prompt_ids=tokens)))
    # The source-owned paired scalar case is Off. Do not create an MTP policy or
    # sampling law from corpus teacher continuations. Reuse its public intent.
    config = selection.topology["expected_config"]["configuration"]
    require(selection.record["runtime"]["generation"]["mtp_policy"] == "off", "unsupported campaign MTP source policy")
    sampling = {key: config[key] for key in ("temperature", "top_k", "top_p", "seed")}
    campaign_seed = sampling["seed"] if seed is None else seed
    require(type(campaign_seed) is int and 1 <= campaign_seed <= 2147483647,
            "campaign requires a positive 32-bit seed; supply --seed and pass the same seed to benchmark")
    sampling["seed"] = campaign_seed
    return dict(schema="llaminar.benchmark-campaign.v1", output_directory=str(output_directory), sampling=sampling,
                prompts=prompts, cells=[dict(id=f"r{repeat}-{h}", prompt_id=h, mtp=dict(mode="off"))
                    for repeat in range(repetitions) for h in history_ids])


def export_campaign(source_root: Path, exported: Path, exact: str, corpus_root: Path,
                    document_root: Path, history_ids: list[str], output_directory: Path, repetitions: int,
                    *, seed: int | None = None) -> dict:
    """Use the original paired source/corpus validator; never launch a model.

    The resulting campaign is a diagnostic real-text workload, not the declared
    synthetic benchmark or a numerical/certification substitute. Source-bound
    local metadata retains the canonical reference-payload containment checks.
    """
    sys.path.insert(0, str(source_root / "scripts/ci"))
    try:
        import model_parity_inventory as inventory
        import model_kld_corpus_execution as paired
        from model_kld_corpus import unchanged
        records = inventory.parse_parameters(exported.read_text())
        require(exact in records, "unknown original exported source cell")
        selection = paired.PairedCorpusSelection.from_export(exact, records[exact], corpus_root, document_root,
            metadata_policy=paired.PairedMetadataPolicy.SOURCE_BOUND_DIAGNOSTIC)
        result = campaign_from_selection(selection, history_ids, output_directory, repetitions=repetitions, seed=seed)
        unchanged(selection.pins)
        return result
    finally:
        sys.path.pop(0)


def main() -> None:
    """Run explicit export/consistency operations; never execute a performance actor."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--source-root", type=Path, required=True)
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--reference-directory", type=Path, required=True)
    export.add_argument("--history-id", required=True)
    export.add_argument("--layer", type=int, required=True)
    export.add_argument("--maximum-original-ids", type=int, required=True)
    export.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("trace", type=Path)
    campaign = sub.add_parser("campaign", help="metadata-only original natural-token benchmark campaign")
    campaign.add_argument("--source-root", type=Path, required=True)
    campaign.add_argument("--export", type=Path, required=True)
    campaign.add_argument("--exact-cell", required=True)
    campaign.add_argument("--corpus-root", type=Path, required=True)
    campaign.add_argument("--document-root", type=Path, required=True)
    campaign.add_argument("--history-id", action="append", required=True)
    campaign.add_argument("--repetitions", type=int, default=2)
    campaign.add_argument("--seed", type=int,
                          help="positive benchmark cohort seed; also pass this seed to the public benchmark runtime")
    campaign.add_argument("--campaign-output-directory", type=Path, required=True)
    campaign.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "campaign":
        document = export_campaign(args.source_root, args.export, args.exact_cell, args.corpus_root,
            args.document_root, args.history_id, args.campaign_output_directory, args.repetitions, seed=args.seed)
        with args.output.open("x") as output:
            json.dump(document, output, indent=2)
            output.write("\n")
        print(json.dumps(dict(scope="diagnostic_real_text_campaign_metadata", native_execution_proven=False,
            timing_qualified=False, prompts=len(document["prompts"]), cells=len(document["cells"]))))
        return
    if args.command == "export":
        trace = export_trace(args.source_root, args.manifest, args.reference_directory,
                             args.history_id, args.layer, args.maximum_original_ids)
        # Output must be explicit and fresh; no corpus or prior evidence is overwritten.
        with args.output.open("x") as output:
            json.dump(trace, output, indent=2)
            output.write("\n")
    else:
        trace = load_document(args.trace)
        verify_saved_files(trace)
    print(json.dumps(validate_trace(trace), sort_keys=True))


if __name__ == "__main__":
    main()
