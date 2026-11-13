"""Retain complete native GPU decode shape evidence under its original owner.

The existing collection contract is the compatibility boundary: build/compiler,
architecture/device/runtime, serial arithmetic policy, phase, formats and device
ordinals must match. A retained superset may cover fewer newly requested shapes;
raw candidate/sample values and original artifacts remain independently bound.
No cross-build compatibility or missing native hardware facts are inferred.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile

from .corpus_provenance import sha256_file


def _compatible(original: dict, wanted: dict) -> bool:
    """Require the existing actual producer/surface identity and timing floors."""
    try:
        return (all(original[key] == wanted[key] for key in
                    ("build", "devices", "formats", "m", "repeats"))
                and all(type(original[key]) is int and original[key] >= wanted[key]
                        for key in ("warmups", "samples")))
    except (KeyError, TypeError):
        return False


def _project_csv(source: Path, destination: Path, shapes: set[str]) -> None:
    """Copy original row values for selected shapes, preserving raw sample IDs."""
    with source.open(newline="") as stream, destination.open("w", newline="") as output:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "shape" not in reader.fieldnames:
            raise ValueError("original GPU artifact has no physical shape identity")
        writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["shape"] in shapes:
                writer.writerow(row)


def _validated_shapes(backend: str, original: Path, aggregate: Path, timing: Path,
                      contract: dict, shapes: tuple[str, ...], wanted: dict) -> tuple[str, ...]:
    """Reapply original strong adapters and complete per-shape production gates.

    Adapters authenticate every raw double, launch route and timing generation.
    No screening row can acquire the full production sample floor. An invalid
    supported candidate rejects its whole shape, even if another candidate won.
    """
    from .adaptive_tournament import is_promotion_grade
    from .adapters.evidence import raw_corpus_id
    from .corpus import ObservationCorpus
    from .profiles import MeasurementProfile, require_promotion_timing
    from .schema import ExecutionMode, SemanticContract
    from .shape_manifest import load_shape_manifest
    from .validation import require_exact_overlay_scope
    if backend == "cuda":
        from .adapters.cuda_decode import CUDADecodeAdapterContext as Context, adapt_cuda_decode_csv as adapt
        from cuda.gemm.analyze_cuda_native_quantised_decode_trainer import _require_shape_reachable_candidate_coverage as candidates
    elif backend == "rocm":
        from .adapters.rocm_decode import ROCmDecodeAdapterContext as Context, adapt_rocm_decode_csv as adapt
        from .candidate_registry import rocm_native_quantised_decode_registry
        from .validation import require_registry_candidate_coverage
        def candidates(corpus):
            require_registry_candidate_coverage(corpus, rocm_native_quantised_decode_registry())
    else:
        raise ValueError("unsupported GPU completed collection backend")
    ms = tuple(int(value) for value in contract["m"].split(","))
    semantic = (SemanticContract.FAST if ms == (1,)
                else SemanticContract.VERIFIER_SERIAL_M1_BITWISE)
    context = Context(profile=MeasurementProfile.PRODUCTION,
        run_id=f"{backend}-exact-{'m1' if ms == (1,) else 'grouped'}",
        corpus_id=raw_corpus_id((original / "aggregate.csv", original / "timing.csv")),
        raw_timing_sidecar_retained=True, **contract["build"])
    corpus = adapt((aggregate,), context, timing_sidecars=(timing,))
    if backend == "cuda":
        with timing.open(newline="") as stream:
            if any(int(row["timed_replays"]) != contract["repeats"] for row in csv.DictReader(stream)):
                raise ValueError("original CUDA raw replay divisor disagrees with its contract")
    manifest = load_shape_manifest()
    by_shape: dict[str, list] = {name: [] for name in shapes}
    for row in corpus:
        by_shape[row.shape_name].append(row)
    accepted = []
    for name, rows in by_shape.items():
        try:
            cell = ObservationCorpus(rows)
            require_exact_overlay_scope(cell, manifest, shape_names=(name,),
                m_values=ms, execution_modes=tuple(ExecutionMode), contract=semantic)
            candidates(cell)
            if any(not is_promotion_grade(row) for row in cell):
                continue
            require_promotion_timing(cell, label="original completed GPU shape",
                minimum_warmups=contract["warmups"], minimum_samples=contract["samples"])
            if any(row.supported and not (row.numerical_correctness and row.repeat_equal
                    and row.forced_route_ok and row.workspace_ok and row.explicit_stream_ok
                    and (row.execution_mode != ExecutionMode.GRAPH_CAPTURED or row.graph_capture_ok)
                    and (semantic != SemanticContract.VERIFIER_SERIAL_M1_BITWISE or row.bitwise_equal))
                   for row in cell):
                continue
            accepted.append(name)
        except ValueError:
            # A failed/incomplete shape is new missing work, never retained proof.
            continue
    return tuple(accepted)


def retain_completed_decode_shapes(source_root: Path, backend: str, wanted: dict,
                                   directory: Path) -> tuple[list[tuple[Path, Path]], tuple[str, ...]]:
    """Reference admissible original shapes and leave only genuinely missing jobs.

    Search only the explicit root and existing decode phase locations. Retained
    references cannot become a second-generation producer. Each original native
    artifact and contract is hashed before and after validation, then bound by
    the new collection's completion receipt through collection-reuse.json.
    """
    source_root, directory = Path(source_root), Path(directory)
    if not source_root.is_dir():
        raise ValueError("requested original GPU collection root is not a directory")
    phase = "m1" if wanted["m"] == "1" else "rows"
    roots = (source_root, source_root / phase, source_root / "decode" / phase,
             source_root / backend / "decode" / phase)
    seen: set[Path] = set()
    remaining = list(wanted["shapes"])
    outputs, sources, declined = [], [], []
    retained = directory / "retained"
    for root in roots:
        root = root.resolve()
        if root in seen or root == directory.resolve():
            continue
        seen.add(root)
        if not (root / "collection-contract.json").is_file():
            continue
        try:
            if (root / "collection-reuse.json").exists():
                raise ValueError("reference is not an original native collection")
            files = {name: root / filename for name, filename in (
                ("contract", "collection-contract.json"), ("complete", "collection-complete.json"),
                ("aggregate", "aggregate.csv"), ("timing", "timing.csv"))}
            hashes = {name: sha256_file(path) for name, path in files.items()}
            contract = json.loads(files["contract"].read_text())
            if not _compatible(contract, wanted):
                raise ValueError("original producer/hardware/phase/format/device/timing contract differs")
            completion = json.loads(files["complete"].read_text())
            if completion != {name: hashes[name] for name in ("aggregate", "timing")}:
                raise ValueError("original native completion is incomplete or its raw bytes changed")
            names = contract["shapes"]
            if not isinstance(names, list) or len(set(names)) != len(names):
                raise ValueError("original shape inventory is not unique")
            selected = tuple(name for name in remaining if name in names)
            if not selected:
                continue
            retained.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=retained) as temporary:
                aggregate, timing = Path(temporary) / "aggregate.csv", Path(temporary) / "timing.csv"
                for source, target in ((files["aggregate"], aggregate), (files["timing"], timing)):
                    _project_csv(source, target, set(selected))
                selected = _validated_shapes(backend, root, aggregate, timing, contract, selected, wanted)
            if not selected:
                raise ValueError("no complete full-grade correct shape remains")
            target = retained / str(len(sources))
            target.mkdir(exist_ok=True)
            aggregate, timing = target / "aggregate.csv", target / "timing.csv"
            for source, destination in ((files["aggregate"], aggregate), (files["timing"], timing)):
                _project_csv(source, destination, set(selected))
            if any(sha256_file(files[name]) != digest for name, digest in hashes.items()):
                raise ValueError("original evidence changed while validating reuse")
            sources.append({"shapes": list(selected), "original_contract": contract,
                "artifacts": {str(files[name]): digest for name, digest in hashes.items()},
                "aggregate": str(aggregate.resolve()), "timing": str(timing.resolve()),
                "retained_artifacts": {str(path.resolve()): sha256_file(path) for path in (aggregate, timing)}})
            outputs.append((aggregate, timing))
            remaining = [name for name in remaining if name not in selected]
        except (OSError, ValueError, KeyError, TypeError) as error:
            declined.append({"directory": str(root), "reason": str(error)})
    if not sources and not declined:
        declined.append({"directory": str(source_root.resolve()),
                         "reason": "no original native collection covers the requested phase/shapes"})
    receipt = {"schema": "native-quantised-original-gpu-shapes-v1", "backend": backend,
               "source_root": str(source_root.resolve()), "requested_contract": wanted,
               "sources": sources, "declined": declined}
    (directory / "collection-reuse.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return outputs, tuple(remaining)


def verify_retained_decode_sources(directory: Path) -> None:
    """Reject changed or missing original references before publication/resume."""
    path = Path(directory) / "collection-reuse.json"
    if not path.exists():
        return
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != "native-quantised-original-gpu-shapes-v1":
        raise ValueError("unsupported original GPU shape receipt")
    for source in receipt["sources"]:
        for field in ("artifacts", "retained_artifacts"):
            for name, expected in source[field].items():
                if sha256_file(Path(name)) != expected:
                    raise ValueError("retained original GPU shape artifact changed")


def resume_retained_decode_shapes(directory: Path, backend: str, wanted: dict,
                                  source_root: Path | None = None) -> tuple[list[tuple[Path, Path]], tuple[str, ...]]:
    """Resume validated references after interruption without paying for them again."""
    directory = Path(directory)
    verify_retained_decode_sources(directory)
    receipt = json.loads((directory / "collection-reuse.json").read_text())
    if receipt.get("backend") != backend or receipt.get("requested_contract") != wanted:
        raise ValueError("retained original GPU scope changed")
    if source_root is not None and receipt.get("source_root") != str(Path(source_root).resolve()):
        raise ValueError("retained original GPU root changed")
    outputs, covered = [], set()
    for source in receipt["sources"]:
        names = source["shapes"]
        if (not names or len(set(names)) != len(names) or covered.intersection(names)
                or not set(names) <= set(wanted["shapes"])):
            raise ValueError("retained original GPU shape coverage is inconsistent")
        aggregate, timing = Path(source["aggregate"]), Path(source["timing"])
        if (set(source["retained_artifacts"]) != {str(aggregate), str(timing)}
                or aggregate.name != "aggregate.csv" or timing.name != "timing.csv"
                or aggregate.parent != timing.parent
                or aggregate.parent.parent != (directory / "retained").resolve()):
            raise ValueError("retained original GPU artifact ownership is inconsistent")
        outputs.append((aggregate, timing))
        covered.update(names)
    return outputs, tuple(name for name in wanted["shapes"] if name not in covered)
