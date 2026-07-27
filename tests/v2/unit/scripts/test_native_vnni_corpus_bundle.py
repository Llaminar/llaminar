"""Regression tests for immutable NativeVNNI corpus publication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.v2.performance.kernels.native_vnni_dispatch.corpus_bundle import (
    CorpusBundleError,
    LFS_POINTER_PREFIX,
    MANIFEST_NAME,
    canonical_refresh_arguments,
    configuration_digest,
    inventory_generation_digest,
    missing_required_payloads,
    required_payloads,
    seal_corpus,
    verify_corpus,
)
from tests.v2.performance.kernels.native_vnni_dispatch.qwen_release_geometry import (
    QWEN_RELEASE_CATALOG_PATH,
)
from tests.v2.performance.kernels.native_vnni_dispatch.shape_manifest import (
    MANIFEST_PATH,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
INVENTORY_SOURCES = (MANIFEST_PATH, QWEN_RELEASE_CATALOG_PATH)


def _seal(directory: Path) -> dict[str, object]:
    """Seal a minimal representative corpus for one test."""

    for relative in required_payloads("cuda", "all"):
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"fixture for {relative}\n", encoding="utf-8")
    return seal_corpus(
        directory,
        backend="cuda",
        profile="all",
        architecture="sm86-rtx3090",
        repository_root=REPOSITORY_ROOT,
        inventory_sources=INVENTORY_SOURCES,
        refresh_arguments=("--cuda-measurement-lanes", "2"),
    )


def test_round_trip_authenticates_payload_and_inventory(tmp_path: Path) -> None:
    """A complete bundle can be verified repeatedly without mutation."""

    (tmp_path / "timing.csv").write_text("shape,median_us\nA,1.0\n")
    (tmp_path / "profiler.json").write_text('{"complete": true}\n')
    sealed = _seal(tmp_path)

    verified = verify_corpus(
        tmp_path / MANIFEST_NAME,
        repository_root=REPOSITORY_ROOT,
    )

    assert verified["corpus_id"] == sealed["corpus_id"]
    assert verified["refresh_arguments"] == [
        "--cuda-measurement-lanes",
        "2",
    ]
    assert verified["architecture"] == "sm86-rtx3090"


def test_backend_measurement_source_changes_inventory_generation(
    tmp_path: Path,
) -> None:
    """A backend plan gets a new workspace without changing shared shapes."""

    source = tmp_path / "cpu_prefill_plan.py"
    source.write_text("M_VALUES = (64, 256, 512)\n", encoding="utf-8")
    first = inventory_generation_digest((source,), tmp_path)
    source.write_text("M_VALUES = (64, 256, 512, 1024)\n", encoding="utf-8")
    second = inventory_generation_digest((source,), tmp_path)

    assert first != second


@pytest.mark.parametrize(
    "checkpoint_arguments",
    (
        ("--cpu-batch-limit", "10"),
        ("--cpu-batch-limit=10",),
        ("--resume-cpu-partials",),
        ("--cpu-batch-limit", "10", "--resume-cpu-partials"),
    ),
)
def test_checkpoint_controls_do_not_change_collection_identity(
    checkpoint_arguments: tuple[str, ...],
) -> None:
    """Changing checkpoint scheduling must resume the same evidence corpus."""

    evidence_arguments = ("--cpu-formats", "Q8_0,Q4_0")

    assert canonical_refresh_arguments(
        (*evidence_arguments, *checkpoint_arguments)
    ) == evidence_arguments
    assert configuration_digest(
        (*evidence_arguments, *checkpoint_arguments)
    ) == configuration_digest(evidence_arguments)


def test_checkpoint_option_requires_its_value() -> None:
    """Malformed execution controls fail before selecting a workspace."""

    with pytest.raises(CorpusBundleError, match="missing value"):
        configuration_digest(("--cpu-batch-limit",))


def test_rocm_development_resume_keeps_the_paid_corpus_identity() -> None:
    """A post-profiler continuation must reopen the existing staging root."""

    evidence_arguments = ("--rocm-formats", "Q8_0,Q4_0")
    continuation_arguments = (
        "--reuse-rocm-development",
        "--rocm-development-build-change-audit",
        "profiler lifetime guard only; candidate arithmetic unchanged",
    )

    assert canonical_refresh_arguments(
        (*evidence_arguments, *continuation_arguments)
    ) == evidence_arguments
    assert configuration_digest(
        (*evidence_arguments, *continuation_arguments)
    ) == configuration_digest(evidence_arguments)


def test_fit_only_evidence_floors_do_not_change_collection_identity() -> None:
    """Promotion thresholds may mine one immutable timing corpus repeatedly."""

    baseline = ("--cpu-formats", "Q8_0")
    adjusted = (
        *baseline,
        "--cpu-minimum-promotion-warmups",
        "7",
        "--cpu-minimum-promotion-samples=50",
    )

    assert configuration_digest(adjusted) == configuration_digest(baseline)


def test_sealed_manifest_omits_checkpoint_controls(tmp_path: Path) -> None:
    """Published provenance contains semantics, not one run's stop point."""

    (tmp_path / "timing.csv").write_text("shape,median_us\nA,1.0\n")
    for relative in required_payloads("cuda", "all"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"fixture for {relative}\n", encoding="utf-8")
    sealed = seal_corpus(
        tmp_path,
        backend="cuda",
        profile="all",
        architecture="sm86-rtx3090",
        repository_root=REPOSITORY_ROOT,
        inventory_sources=INVENTORY_SOURCES,
        refresh_arguments=(
            "--cuda-measurement-lanes",
            "2",
            "--cpu-batch-limit",
            "10",
            "--resume-cpu-partials",
        ),
    )

    assert sealed["refresh_arguments"] == ["--cuda-measurement-lanes", "2"]


def test_collection_completeness_distinguishes_checkpoint_from_publication(
    tmp_path: Path,
) -> None:
    """Turnkey publication waits for every backend production payload."""

    required = required_payloads("cpu-prefill", "all")
    for relative in required[:-1]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")

    assert missing_required_payloads(
        tmp_path,
        backend="cpu-prefill",
        profile="all",
    ) == (required[-1],)

    final_path = tmp_path / required[-1]
    final_path.write_text("final fixture\n", encoding="utf-8")
    assert missing_required_payloads(
        tmp_path,
        backend="cpu-prefill",
        profile="all",
    ) == ()


def test_payload_corruption_is_rejected(tmp_path: Path) -> None:
    """Timing evidence cannot change after publication."""

    payload = tmp_path / "timing.csv"
    payload.write_text("shape,median_us\nA,1.0\n")
    _seal(tmp_path)
    payload.write_text("shape,median_us\nA,2.0\n")

    with pytest.raises(CorpusBundleError, match="payload digest changed"):
        verify_corpus(
            tmp_path / MANIFEST_NAME,
            repository_root=REPOSITORY_ROOT,
        )


def test_unresolved_lfs_payload_is_rejected(tmp_path: Path) -> None:
    """A pointer-only checkout must request Git LFS materialization."""

    pointer = (
        LFS_POINTER_PREFIX
        + b"oid sha256:0123456789abcdef\n"
        + b"size 123\n"
    )
    (tmp_path / "timing.csv").write_bytes(pointer)

    with pytest.raises(CorpusBundleError, match="unresolved Git LFS pointer"):
        _seal(tmp_path)


def test_partial_publication_is_rejected(tmp_path: Path) -> None:
    """Interrupted atomic outputs never become corpus evidence."""

    (tmp_path / "timing.csv").write_text("shape,median_us\nA,1.0\n")
    (tmp_path / "timing.csv.inprogress").write_text("partial")

    with pytest.raises(CorpusBundleError, match="partial corpus artifact"):
        _seal(tmp_path)


def test_incomplete_production_transaction_cannot_be_sealed(tmp_path: Path) -> None:
    """A checkpoint is not publishable merely because its files are immutable."""

    (tmp_path / "cuda_decode_m1.development.csv").write_text("partial\n")

    with pytest.raises(CorpusBundleError, match="production corpus is incomplete"):
        seal_corpus(
            tmp_path,
            backend="cuda",
            profile="all",
            architecture="sm86-rtx3090",
            repository_root=REPOSITORY_ROOT,
            inventory_sources=INVENTORY_SOURCES,
        )


def test_cpu_bundle_requires_decode_dependency_and_grouped_policy() -> None:
    """A publishable CPU corpus owns both sides of the staged dependency."""

    payloads = set(required_payloads("cpu", "all"))

    assert "CPUNativeVNNIDecodePolicyGenerated.inc" in payloads
    assert "cpu_decode_m1_policy.json" in payloads
    assert "cpu_decode_m1_common_observations.csv" in payloads
    assert "cpu_decode_final_profiler_evidence.json" in payloads
    assert "CPUNativeVNNIVerifierRowsPolicyGenerated.inc" in payloads
    assert "cpu_verifier_rows_policy.json" in payloads


def test_manifest_identity_cannot_be_rewritten(tmp_path: Path) -> None:
    """Changing refresh provenance invalidates the self-authenticating ID."""

    (tmp_path / "timing.csv").write_text("shape,median_us\nA,1.0\n")
    _seal(tmp_path)
    manifest_path = tmp_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["refresh_arguments"] = ["--cuda-measurement-lanes", "1"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(CorpusBundleError, match="configuration digest"):
        verify_corpus(manifest_path, repository_root=REPOSITORY_ROOT)


def test_missing_manifest_reports_bundle_error(tmp_path: Path) -> None:
    """Callers receive one stable domain error for an absent checkout."""

    with pytest.raises(CorpusBundleError, match="cannot read corpus file"):
        verify_corpus(
            tmp_path / MANIFEST_NAME,
            repository_root=REPOSITORY_ROOT,
        )


def test_refresh_argument_types_are_not_silently_coerced(tmp_path: Path) -> None:
    """Manifest arguments retain exact ordered string identity."""

    (tmp_path / "timing.csv").write_text("shape,median_us\nA,1.0\n")
    _seal(tmp_path)
    manifest_path = tmp_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["refresh_arguments"] = ["--cuda-measurement-lanes", 2]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(CorpusBundleError, match="arguments must be strings"):
        verify_corpus(manifest_path, repository_root=REPOSITORY_ROOT)


def test_file_record_types_are_not_silently_coerced(tmp_path: Path) -> None:
    """Sizes and digests remain canonical typed manifest fields."""

    (tmp_path / "timing.csv").write_text("shape,median_us\nA,1.0\n")
    _seal(tmp_path)
    manifest_path = tmp_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["size_bytes"] = str(
        manifest["files"][0]["size_bytes"]
    )
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(CorpusBundleError, match="file record values are invalid"):
        verify_corpus(manifest_path, repository_root=REPOSITORY_ROOT)
