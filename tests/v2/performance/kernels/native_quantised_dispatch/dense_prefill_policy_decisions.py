"""Publish selective WMMA improvements against the actual frozen installed policy.

Every planned runtime key owns one typed install or retain decision. Retention
preserves exact rows, compatible rows and absence literally; it is not a new
certificate for the installed policy. Only changed, profitable native launches
create profiler obligations. Decisions are reconstructed from authenticated raw
timings whenever profiling or publication needs them, closing partial-inventory
and stale-decision holes without treating Auto as a concrete generated row.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import os
from pathlib import Path
import statistics
import tempfile
from typing import Sequence

from . import dense_production_overlay as overlay
from . import production_dense_prefill_sweep as sweep
from .corpus_provenance import mapping_digest, sha256_file
from .dense_prefill_policy_binding import _digest, validate_installed_policy
from .format_registry import FORMAT_SPECS

RULE = "changed-alias-strict-median-improvement-v1"
SCHEMA = "rocm-dense-prefill-policy-decisions-v1"


class RetentionReason(str, Enum):
    """A retained policy has no new profitable physical launch to certify."""

    NO_PROFITABLE_CHANGE = "no-wmma-improves-every-changed-alias"


@dataclass(frozen=True)
class RetainInstalled:
    """Keep the original lookup behavior, including a missing exact row."""

    key: overlay.DenseOverlayKey
    source_formats: tuple[str, ...]
    shape_names: tuple[str, ...]
    reason: RetentionReason = RetentionReason.NO_PROFITABLE_CHANGE


@dataclass(frozen=True)
class InstallNativeWmma:
    """Replace one key with an independently qualified concrete native launch."""

    entry: overlay.DenseOverlayEntry

    @property
    def key(self) -> overlay.DenseOverlayKey:
        """Expose the same runtime-key authority as a retention decision."""
        return self.entry.key


Decision = InstallNativeWmma | RetainInstalled


def validate_decision_totality(decisions: Sequence[Decision], cells: Sequence[sweep.DensePrefillCell]) -> None:
    """Require one typed decision and every original source alias for every key."""
    expected = overlay._expected_overlay_inventory(cells, "rocm")
    if any(not isinstance(decision, (InstallNativeWmma, RetainInstalled)) for decision in decisions):
        raise ValueError("selective policy requires typed install or retain decisions")
    keys = tuple(decision.key for decision in decisions)
    if keys != tuple(sorted(expected)):
        raise ValueError("selective decisions must cover every planned key exactly once in order")
    for decision in decisions:
        item = decision.entry if isinstance(decision, InstallNativeWmma) else decision
        if (item.source_formats, item.shape_names) != expected[decision.key]:
            raise ValueError("selective decision source alias or shape coverage mismatch")
        aliases = tuple(sorted(source.label for source in FORMAT_SPECS
                               if source.runtime_codebook("rocm") == decision.key.execution_codebook))
        if item.source_formats != aliases:
            raise ValueError("selective decisions require complete original source aliases")
        if isinstance(decision, InstallNativeWmma):
            overlay._validate_prepared_identity(decision.entry)
            if (decision.entry.backend != "rocm" or decision.entry.launch[5] != 4
                    or decision.entry.candidate_id not in sweep.ROCM_WMMA_CANDIDATE_IDS):
                raise ValueError("selective installation requires an implemented native WMMA candidate")
        elif not isinstance(decision.reason, RetentionReason):
            raise ValueError("retention requires an explicit typed reason")


@dataclass(frozen=True)
class DensePolicyDecisions:
    """A complete decision inventory bound to original policy, producer and clocks."""

    plan_digest: str
    installed_policy_digest: str
    producer_digest: str
    observations_digest: str
    decisions: tuple[Decision, ...]

    @property
    def installations(self) -> tuple[overlay.DenseOverlayEntry, ...]:
        """Project only changed rows; totality still belongs to all decisions."""
        return tuple(decision.entry for decision in self.decisions if isinstance(decision, InstallNativeWmma))

    def canonical_mapping(self) -> dict[str, object]:
        """Serialize all retained and installed keys with the explicit scoring rule."""
        return dict(schema=SCHEMA, profitability_rule=RULE, plan_digest=self.plan_digest,
                    installed_policy_digest=self.installed_policy_digest,
                    producer_digest=self.producer_digest, observations_digest=self.observations_digest,
                    decisions=[dict(action="install-native-wmma" if isinstance(item, InstallNativeWmma)
                                    else "retain-installed", **asdict(item)) for item in self.decisions])

    @property
    def digest(self) -> str:
        """Bind even all-retain outcomes to the complete original observations."""
        return mapping_digest(self.canonical_mapping())


def collect_policy_decisions(root: Path, *, warmups: int, samples: int) -> DensePolicyDecisions:
    """Choose among eligible WMMA candidates without concretizing retained Auto.

    Eligibility is deliberately per alias: each physically changed launch must
    strictly improve the raw-sample median over actual Auto. Identical launches
    do not count as improvements and at least one alias must change. Ties and
    regressions retain policy. Among eligible candidates, geometric-mean gain,
    worst-alias gain and candidate identity define a deterministic ordering.
    Profiling and a later real installed-Auto qualification remain separate gates.
    """
    cells, plan, grouped = overlay.collect_dense_observations(
        root, "rocm", warmup_runs=warmups, bench_runs=samples)
    if (plan["schema_version"] != sweep.BOUND_SWEEP_PLAN_SCHEMA or
            any(cell.candidate_scope is not sweep.DenseCandidateScope.INSTALLED_VS_NATIVE_WMMA for cell in cells)):
        raise ValueError("selective decisions require an original bound native-WMMA plan")
    binding = plan["installed_policy"]
    validate_installed_policy(binding, plan["trainer_provenance"])
    decisions: list[Decision] = []
    artifacts = {}
    for key, observations in sorted(grouped.items()):
        aliases = tuple(sorted(cell.source_format.label for cell, _, _ in observations))
        shapes = tuple(sorted({cell.shape.name for cell, _, _ in observations}))
        contenders = []
        for candidate in sweep.ROCM_WMMA_CANDIDATE_IDS:
            launches = {overlay._launch_tuple("rocm", rows[candidate]) for _, rows, _ in observations}
            if len(launches) != 1:
                raise ValueError("WMMA candidate resolves to conflicting physical alias launches")
            launch = next(iter(launches))
            gains = []
            changed = False
            for _, rows, timing in observations:
                if launch == overlay._launch_tuple("rocm", rows["Auto"]):
                    gains.append(1.)
                else:
                    changed = True
                    gains.append(statistics.median(timing["Auto"]) / statistics.median(timing[candidate]))
                    if gains[-1] <= 1.:
                        break
            else:
                if changed:
                    contenders.append((overlay._geometric_mean(gains), min(gains), candidate, launch))
        if contenders:
            gain, _, candidate, launch = min(contenders, key=lambda item: (-item[0], -item[1], item[2]))
            regrets = [statistics.median(timing[candidate]) /
                       min(statistics.median(values) for values in timing.values())
                       for _, _, timing in observations]
            decisions.append(InstallNativeWmma(overlay.DenseOverlayEntry(
                "rocm", key, aliases, shapes, candidate, launch,
                overlay._geometric_mean(regrets)-1., max(regrets)-1., gain)))
        else:
            decisions.append(RetainInstalled(key, aliases, shapes))
        for cell, _, _ in observations:
            paths = sweep.production_dense_prefill_cell_paths(root, cell)
            artifacts[cell.cell_id] = {path.name: sha256_file(path)
                                      for path in (paths.aggregate, paths.timing, paths.manifest)}
    validate_decision_totality(decisions, cells)
    return DensePolicyDecisions(plan["plan_digest"], binding["binding_digest"],
                                mapping_digest(plan["trainer_provenance"]), mapping_digest(artifacts), tuple(decisions))


def compose_policy(base: str, decisions: DensePolicyDecisions) -> tuple[str, dict[str, object]]:
    """Preserve the complete frozen envelope and every row outside the install set.

    This is pure composition, not publication admission. The publication owner
    always reconstructs decisions from the original authenticated cells first.
    Empty installation sets return the exact base bytes and a no-change receipt.
    """
    prefix, original, suffix = overlay._split_generated_overlay(base, "rocm", preserve_envelope=True)
    entries = decisions.installations
    refreshed = {}
    if entries:
        if ("NativeQuantisedArchitecture architecture{};" not in prefix and
                '#include "kernels/rocm/gemm/ROCmDenseProductionPrefillOverlayTypes.h"' not in prefix):
            raise ValueError("frozen selector lacks architecture storage for qualified WMMA publication")
        _, refreshed, _ = overlay._split_generated_overlay(overlay.render_dense_overlay(entries), "rocm")
    rows = original | refreshed
    encoded = prefix + "".join(rows[key] for key in sorted(rows)) + suffix if entries else base
    receipt = dict(schema="rocm-dense-prefill-selective-publication-v1",
                   outcome="installed" if entries else "no-change",
                   decision_inventory=decisions.canonical_mapping(), decision_digest=decisions.digest,
                   base_sha256=_digest(base), output_sha256=_digest(encoded),
                   installed_keys=len(refreshed), retained_planned_keys=len(decisions.decisions)-len(refreshed),
                   original_keys=len(original), output_keys=len(rows),
                   retained_policy_evidence="original installed policy; not recertified")
    return encoded, receipt


def _atomic_write(path: Path, text: str) -> None:
    """Publish a complete artifact, with the receipt published last by the caller."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="." + path.name + "-", dir=path.parent) as temporary:
        staging = Path(temporary) / "artifact"
        with staging.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)


def validate_publication_paths(root: Path, cells: Sequence[sweep.DensePrefillCell],
                               destinations: Sequence[Path], *, sources: Sequence[Path] = (),
                               replace_base: Path | None = None) -> None:
    """Admit all outputs before any write, preserving original evidence and policy.

    Reserve conventional staging names too, so a caller cannot publish a final
    artifact onto another transaction's in-progress evidence. Actual selective
    writes use exclusive temporary directories and cannot follow old symlinks.
    Only the first final destination may intentionally replace the frozen base.
    """
    protected = {root / sweep.SWEEP_PLAN_FILENAME, *sources}
    protected.update(path.with_name(path.name + ".inprogress") for path in tuple(protected))
    for cell in cells:
        paths = sweep.production_dense_prefill_cell_paths(root, cell)
        protected.update((paths.aggregate, paths.timing, paths.manifest, paths.log,
                          paths.aggregate_staging, paths.timing_staging, paths.manifest_staging, paths.log_staging))
    protected = {path.resolve() for path in protected}
    resolved = set()
    for index, destination in enumerate(destinations):
        for staged, path in enumerate((destination, destination.with_name(destination.name + ".inprogress"))):
            actual = path.resolve()
            if actual in resolved:
                raise ValueError("publication destinations and staging paths must be distinct")
            resolved.add(actual)
            if actual in protected:
                raise ValueError("publication must not overwrite original timing or profiler evidence")
            if replace_base is not None and actual == replace_base.resolve() and (index != 0 or staged):
                raise ValueError("only the final policy output may replace the frozen base include")


def write_policy(root: Path, output: Path, *, base_include: Path, receipt_path: Path,
                 warmups: int, samples: int, profiler_requests: Path | None = None,
                 profiler_evidence: Path | None = None,
                 summary_output: Path | None = None) -> DensePolicyDecisions:
    """Reconstruct complete decisions and authenticate profile evidence before writes."""
    decisions = collect_policy_decisions(root, warmups=warmups, samples=samples)
    cells, plan = sweep.load_planned_dense_prefill_cells(root / sweep.SWEEP_PLAN_FILENAME, "rocm")
    if plan["plan_digest"] != decisions.plan_digest:
        raise ValueError("publication plan changed during qualification")
    destinations = (output, receipt_path) + ((summary_output,) if summary_output is not None else ())
    validate_publication_paths(root, cells, destinations, replace_base=base_include,
                               sources=tuple(path for path in (profiler_requests, profiler_evidence) if path is not None))
    base = plan["installed_policy"]["base_text"]
    if base_include.read_bytes() != base.encode("utf-8"):
        raise ValueError("publication base differs from the original compiled policy")
    if decisions.installations:
        if profiler_requests is None or profiler_evidence is None:
            raise ValueError("changed WMMA entries require winner and actual-Auto profiling for every alias")
        from .native_prefill_profiler import require_profile
        require_profile(root, profiler_requests, profiler_evidence, warmups=warmups, samples=samples)
    elif profiler_requests is not None or profiler_evidence is not None:
        raise ValueError("all-retain publication has no new profiler obligations")
    encoded, receipt = compose_policy(base, decisions)
    # Profiling validation may be lengthy. Do not overwrite a concurrently
    # changed policy with the frozen generation whose observations were read.
    if base_include.read_bytes() != base.encode("utf-8"):
        raise ValueError("publication base changed during qualification")
    if collect_policy_decisions(root, warmups=warmups, samples=samples) != decisions:
        raise ValueError("publication decisions changed during qualification")
    _atomic_write(output, encoded)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".dense-summary-", dir=summary_output.parent) as temporary:
            staging = Path(temporary) / "summary.csv"
            overlay.write_overlay_summary(staging, decisions.installations)
            os.replace(staging, summary_output)
    _atomic_write(receipt_path, json.dumps(receipt, indent=2) + "\n")
    return decisions
