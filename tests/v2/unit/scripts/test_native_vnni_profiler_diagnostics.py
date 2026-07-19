"""Focused regressions for NativeVNNI profiler-signal diagnostics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.profiler_diagnostics import (  # noqa: E402
    diagnose_profiler_catalog,
)
from native_vnni_dispatch.profiler_model import (
    PhysicalCandidateKey,
    ProfilerCandidateDescriptor,
    ProfilerFeatureCatalog,
)  # noqa: E402
from native_vnni_dispatch.schema import Backend, ExecutionMode  # noqa: E402


def _descriptor(
    candidate: str,
    *,
    complete_metric: float,
    partial_metric: float | None,
    duration_reliable: bool,
) -> ProfilerCandidateDescriptor:
    """Build one exact CPU launch descriptor for a two-candidate contest."""

    key = PhysicalCandidateKey(
        backend=Backend.CPU,
        architecture_class="cpu-test|build=AVX2|runtime=AVX2|threads=2",
        operation_kind="NativeVNNIPrefillProjection",
        bundle_signature="single",
        prepared_family_id="q4",
        packing_abi="test",
        runtime_codebook_id=2,
        effective_candidate_id=candidate,
        execution_mode=ExecutionMode.EAGER,
        m=32,
        projection_n_vector=(64,),
        aggregate_n=64,
        k=64,
    )
    features: dict[str, float | str] = {
        "metric.cpu.instructions_per_million_macs_log1p": complete_metric,
        "profile.cpu.duration_features_reliable": float(duration_reliable),
    }
    if partial_metric is not None:
        features["metric.cpu.llc_load_miss_fraction"] = partial_metric
    return ProfilerCandidateDescriptor(
        key=key,
        anchor_m=32,
        anchor_n=64,
        anchor_k=64,
        features=features,
    )


class NativeVNNIProfilerDiagnosticsTest(unittest.TestCase):
    """Prove complete-contest and candidate-variation accounting."""

    def test_reports_only_complete_candidate_contests_as_model_signal(self) -> None:
        first = _descriptor(
            "candidate-a",
            complete_metric=1.0,
            partial_metric=0.25,
            duration_reliable=True,
        )
        second = _descriptor(
            "candidate-b",
            complete_metric=1.5,
            partial_metric=None,
            duration_reliable=False,
        )
        catalog = ProfilerFeatureCatalog(
            corpus_digest="sha256:corpus",
            request_manifest_digest="sha256:requests",
            evidence_manifest_digest="sha256:evidence",
            descriptors={first.key: first, second.key: second},
        )

        report = diagnose_profiler_catalog(
            catalog,
            source_formats=(name for name in ("Q4_0", "Q4_0")),
        )

        self.assertEqual(report["source_format_count"], 1)
        self.assertEqual(report["multi_candidate_contest_count"], 1)
        complete = report["metrics"][
            "metric.cpu.instructions_per_million_macs_log1p"
        ]
        self.assertEqual(complete["complete_candidate_contests"], 1)
        self.assertEqual(complete["varying_complete_contests"], 1)
        partial = report["metrics"]["metric.cpu.llc_load_miss_fraction"]
        self.assertEqual(partial["complete_candidate_contests"], 0)
        self.assertEqual(report["cpu_duration_reliability"]["reliable"], 1)
        self.assertEqual(report["cpu_duration_reliability"]["unreliable"], 1)


if __name__ == "__main__":
    unittest.main()
