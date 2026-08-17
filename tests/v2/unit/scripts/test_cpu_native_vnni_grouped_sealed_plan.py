#!/usr/bin/env python3
"""Regress fresh-leaf CPU grouped verifier certification plans."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.corpus import GenericDomain  # noqa: E402
from native_vnni_dispatch.cpu_grouped_decode_sealed_plan import (  # noqa: E402
    CPU_GROUPED_SEALED_GEOMETRY_PREFIX,
    build_cpu_grouped_sealed_plan,
    certify_cpu_grouped_sealed_pairs,
    cpu_grouped_burned_seal_costs,
    read_cpu_grouped_sealed_plan,
    validate_cpu_grouped_sealed_plan,
    write_cpu_grouped_sealed_plan,
)
from native_vnni_dispatch.paired_confirmation import (  # noqa: E402
    CPU_ISOLATED_PAIRED_PROTOCOL_VERSION,
    CPU_PROCESS_ISOLATED_TIMING_SCOPE,
    PAIRED_PROTOCOL_VERSION,
    REQUIRED_COLUMNS,
    REQUIRED_COLUMNS_V4,
)
from native_vnni_dispatch.cpu_sealed_paired import (  # noqa: E402
    resolve_sealed_paired_evidence_paths,
)
from native_vnni_dispatch.schema import (  # noqa: E402
    AspectBucket,
    Backend,
    ExecutionMode,
    SemanticContract,
)
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    GenericDispatchRule,
)


ARCHITECTURE = "test-host|build=AVX512|runtime=AVX512|threads=28"
FROZEN_DIGEST = "sha256:" + "a" * 64
DEVELOPMENT_DIGEST = "sha256:" + "b" * 64
SEALED_BUILD_ID = "sha256:" + "d" * 64


class _Development:
    """Minimal immutable corpus surface needed by the seal planner."""

    def __init__(self, rows, domains=(), digest=DEVELOPMENT_DIGEST):
        self._rows = tuple(rows)
        self._domains = tuple(domains)
        self._digest = digest

    def __iter__(self):
        return iter(self._rows)

    def digest(self) -> str:
        return self._digest

    def generic_domains(self):
        return self._domains

    def rows_for_generic_domain(self, domain):
        """Return the synthetic forceability evidence owned by one domain."""

        return tuple(
            row for row in self._rows
            if getattr(row, "domain", None) == domain
        )


class _Manifest:
    """Minimal shape inventory with a deterministic transaction digest."""

    maximum_cpu_measurement_weight_elements = 16_000_000
    shapes = (SimpleNamespace(n=256, k=256),)

    @staticmethod
    def digest() -> str:
        return "sha256:" + "c" * 64


class CPUSealedPairedEvidenceInventoryTest(unittest.TestCase):
    """Protect bounded, complete post-freeze evidence discovery."""

    def test_directory_expands_complete_shards_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index in (2, 1):
                stem = f"shard-{index:04d}.cpu.fixture"
                (directory / f"{stem}.requests.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                (directory / f"{stem}.csv").write_text(
                    "header\n", encoding="utf-8"
                )

            paths = resolve_sealed_paired_evidence_paths((), (directory,))

        self.assertEqual(
            [path.name for path in paths],
            [
                "shard-0001.cpu.fixture.csv",
                "shard-0002.cpu.fixture.csv",
            ],
        )

    def test_directory_rejects_missing_and_inprogress_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "shard-0000.cpu.fixture.requests.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                resolve_sealed_paired_evidence_paths((), (directory,))

            (directory / "shard-0000.cpu.fixture.csv.inprogress").write_text(
                "header\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "unfinished"):
                resolve_sealed_paired_evidence_paths((), (directory,))


def _rule(
    index: int,
    *,
    m: int,
    codebook: int,
    candidate: str,
) -> GenericDispatchRule:
    """Create one complementary all-geometry grouped leaf for a unit domain."""

    return GenericDispatchRule(
        domain=GenericDomain(
            backend=Backend.CPU,
            architecture_class=ARCHITECTURE,
            semantic_contract=(
                SemanticContract.VERIFIER_SERIAL_M1_BITWISE
            ),
            operation_kind="NativeVNNIVerifierRows",
            bundle_signature="single",
            prepared_family_id=f"NativeVNNI_cpu_CB{codebook}",
            packing_abi=f"native-vnni-cpu-cb{codebook}-v1",
            runtime_codebook_id=codebook,
            execution_mode=ExecutionMode.EAGER,
            m=m,
            aspect_bucket=AspectBucket.BALANCED,
            all_aspects=True,
        ),
        predicates=(),
        candidate_id=candidate,
        arithmetic_fingerprint=f"sha256:test-grouped-{index}",
        development_shape_groups=(f"group-{index}",),
        development_max_regret=0.01,
        development_p95_regret=0.01,
        development_mean_regret=0.01,
    )


def _observation(
    rule: GenericDispatchRule,
    *,
    shape_group_id: str,
    n: int,
    k: int,
    source_format: str,
    candidate_id: str,
    forceable: bool,
) -> SimpleNamespace:
    """Create the minimum complete grouped candidate proof used by the plan."""

    return SimpleNamespace(
        domain=rule.domain,
        shape_group_id=shape_group_id,
        aggregate_n=n,
        k=k,
        source_format=source_format,
        candidate_id=candidate_id,
        effective_candidate_id=candidate_id,
        observed_candidate_id=candidate_id,
        supported=forceable,
        forced_route_ok=forceable,
        route_counter_ok=forceable,
        workspace_ok=True,
        explicit_stream_ok=True,
        execution_mode=ExecutionMode.EAGER,
        graph_capture_ok=False,
        semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
        serial_m1_policy_hash="sha256:" + "1" * 64,
        bitwise_equal=True,
        repeat_equal=True,
        mismatch_count=0,
        numerical_correctness=True,
        ordered_reduction=True,
        uses_atomic_reduction=False,
    )


class CPUGroupedSealedPlanTest(unittest.TestCase):
    """Prove grouped plans cover leaves, aliases, M, and candidates exactly."""

    def setUp(self) -> None:
        self.rules = (
            _rule(
                0,
                m=2,
                codebook=4,
                candidate="cpu.nvnni.verifier.pairwise",
            ),
            _rule(
                1,
                m=31,
                codebook=0,
                candidate="cpu.nvnni.verifier.wide_rows",
            ),
        )
        rows = []
        for source_format in ("IQ4_NL", "IQ4_XS"):
            rows.extend((
                _observation(
                    self.rules[0],
                    shape_group_id="group-0",
                    n=512,
                    k=512,
                    source_format=source_format,
                    candidate_id="cpu.nvnni.verifier.pairwise",
                    forceable=True,
                ),
                _observation(
                    self.rules[0],
                    shape_group_id="group-0",
                    n=512,
                    k=512,
                    source_format=source_format,
                    candidate_id="cpu.nvnni.verifier.wide_rows",
                    forceable=False,
                ),
            ))
        for candidate_id in (
            "cpu.nvnni.verifier.pairwise",
            "cpu.nvnni.verifier.wide_rows",
        ):
            rows.append(_observation(
                self.rules[1],
                shape_group_id="group-1",
                n=768,
                k=512,
                source_format="Q4_0",
                candidate_id=candidate_id,
                forceable=True,
            ))
        self.development = _Development(
            rows, tuple(rule.domain for rule in self.rules)
        )
        self.manifest = _Manifest()

    @staticmethod
    def _write_paired_evidence(
        path: Path,
        plan,
        *,
        process_isolated: bool = True,
    ) -> None:
        """Write strict byte-exact paired rows for every planned edge.

        CPU evidence capable of promoting a policy must identify a one-rank,
        process-isolated timing environment.  The optional legacy form exists
        solely to prove that final certification rejects historical co-run
        evidence whose execution context cannot be authenticated.
        """

        rows = []
        pair_count = 30
        for item in plan.requests:
            request = item.request
            for pair_index in range(pair_count):
                order = (
                    ("selected", "exact")
                    if pair_index % 2 == 0
                    else ("exact", "selected")
                )
                for within_pair_order, role in enumerate(order):
                    candidate = (
                        request.selected_candidate_id
                        if role == "selected"
                        else request.exact_candidate_id
                    )
                    latency = 12.0 if role == "selected" else 10.0
                    row = {
                        "protocol_version": (
                            CPU_ISOLATED_PAIRED_PROTOCOL_VERSION
                            if process_isolated
                            else PAIRED_PROTOCOL_VERSION
                        ),
                        "request_id": request.request_id,
                        "backend": "cpu",
                        "architecture_class": request.architecture_class,
                        "phase": "verifier_rows",
                        "source_format": request.source_format,
                        "source_codebook": str(request.source_codebook),
                        "execution_codebook": str(request.execution_codebook),
                        "shape": request.shape,
                        "execution_mode": request.execution_mode,
                        "m": str(request.m),
                        "n": str(request.n),
                        "k": str(request.k),
                        "pair_index": str(pair_index),
                        "configured_pair_count": str(pair_count),
                        "within_pair_order": str(within_pair_order),
                        "cell_order_seed": "12345",
                        "pair_order_seed": str(50_000 + pair_index),
                        "candidate_role": role,
                        "candidate_id": candidate,
                        "timed_replays": "16",
                        "latency_us": f"{latency:.9f}",
                        "latency_us_hex": latency.hex(),
                        "warmup_count": "5",
                        "bit_mismatches": "0",
                        "first_bit_mismatch": "0",
                        "repeat_byte_mismatches": "0",
                        "max_abs": "0",
                        "relative_l2": "0",
                        "cosine": "1",
                        "symmetric_kld": "0",
                        "grouped_output_digest": "fnv1a64:1111111111111111",
                        "serial_output_digest": "fnv1a64:1111111111111111",
                        "graph_capture_ok": "1",
                        "workspace_ok": "1",
                        "explicit_stream_ok": "1",
                        "route_counter_ok": "1",
                        "observed_candidate_id": candidate,
                        "observed_path": "grouped-verifier-rows",
                        "observed_tile_n": "0",
                        "observed_cpt": "0",
                        "observed_effective_kb": "0",
                        "serial_m1_candidate_id": "cpu.serial.production",
                        "numerical_correctness": "1",
                        "correctness_pass": "1",
                    }
                    if process_isolated:
                        row.update({
                            "timing_scope": CPU_PROCESS_ISOLATED_TIMING_SCOPE,
                            "mpi_world_size": "1",
                        })
                    rows.append(row)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=sorted(
                    REQUIRED_COLUMNS_V4
                    if process_isolated
                    else REQUIRED_COLUMNS
                ),
            )
            writer.writeheader()
            writer.writerows(rows)

    def _plan(self):
        return build_cpu_grouped_sealed_plan(
            self.rules,
            FROZEN_DIGEST,
            self.development,
            self.manifest,
            SEALED_BUILD_ID,
        )

    def test_plan_is_fresh_and_exhaustive_for_aliases_and_m(self) -> None:
        plan = self._plan()

        self.assertEqual(len(plan.rule_witnesses), 2)
        self.assertEqual({item.m for item in plan.rule_witnesses}, {2, 31})
        self.assertTrue(all(
            item.shape.startswith(CPU_GROUPED_SEALED_GEOMETRY_PREFIX)
            for item in plan.rule_witnesses
        ))
        self.assertTrue(all(
            (item.n, item.k) not in {(512, 512), (768, 512), (256, 256)}
            for item in plan.rule_witnesses
        ))
        # CPU codebook 4 owns IQ4_NL and IQ4_XS; codebook 0 owns Q4_0.
        self.assertEqual(len(plan.requests), 3)
        self.assertEqual(
            {item.request.source_format for item in plan.requests},
            {"IQ4_NL", "IQ4_XS", "Q4_0"},
        )
        self.assertEqual(
            {
                item.request.source_format
                for item in plan.requests
                if item.request.selected_candidate_id
                == item.request.exact_candidate_id
            },
            {"IQ4_NL", "IQ4_XS"},
        )
        validate_cpu_grouped_sealed_plan(
            plan, self.rules, self.development, self.manifest, SEALED_BUILD_ID
        )

    def test_round_trip_preserves_self_digest_and_shards(self) -> None:
        plan = self._plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "plan.json"
            shards = write_cpu_grouped_sealed_plan(
                plan_path, plan, root / "requests", max_requests_per_shard=2
            )
            restored = read_cpu_grouped_sealed_plan(plan_path)

        self.assertEqual(restored.canonical_mapping(), plan.canonical_mapping())
        self.assertEqual(len(shards), 2)

    def test_single_forceable_domain_is_route_witnessed_at_zero_regret(
        self,
    ) -> None:
        """A normalized schedule cannot become a fictitious challenger."""

        plan = self._plan()
        with tempfile.TemporaryDirectory() as temporary:
            evidence_path = Path(temporary) / "paired.csv"
            self._write_paired_evidence(evidence_path, plan)
            report = certify_cpu_grouped_sealed_pairs(
                self.rules,
                FROZEN_DIGEST,
                plan,
                (evidence_path,),
                bootstrap_replicates=1_000,
                workers=1,
            )

        self.assertEqual(report.unexercised_rule_count, 0)
        self.assertEqual(len(report.cells), 3)
        single_cells = tuple(
            cell for cell in report.cells
            if cell.domain == self.rules[0].domain
        )
        self.assertEqual(len(single_cells), 2)
        self.assertTrue(all(
            cell.selected_candidate_id == cell.exact_candidate_id
            and cell.observed_worst_surface_regret == 0.0
            and cell.simultaneous_95pct_upper_regret == 0.0
            for cell in single_cells
        ))

    def test_seal_rejects_legacy_cpu_corun_timing_evidence(self) -> None:
        """Unknown CPU co-run context may inform fitting but cannot certify."""

        plan = self._plan()
        with tempfile.TemporaryDirectory() as temporary:
            evidence_path = Path(temporary) / "paired.csv"
            self._write_paired_evidence(
                evidence_path,
                plan,
                process_isolated=False,
            )
            with self.assertRaisesRegex(
                ValueError,
                "unauthenticated co-run environment",
            ):
                certify_cpu_grouped_sealed_pairs(
                    self.rules,
                    FROZEN_DIGEST,
                    plan,
                    (evidence_path,),
                    bootstrap_replicates=1_000,
                    workers=1,
                )

    def test_burned_seal_becomes_generic_costs_and_not_a_reusable_reserve(
        self,
    ) -> None:
        """Inspected timings inform the next tree and burn their geometry."""

        plan = self._plan()
        with tempfile.TemporaryDirectory() as temporary:
            evidence_path = Path(temporary) / "paired.csv"
            self._write_paired_evidence(evidence_path, plan)
            costs = cpu_grouped_burned_seal_costs(
                self.development,
                plan,
                (evidence_path,),
            )

        self.assertEqual(set(costs), {rule.domain for rule in self.rules})
        by_domain = {
            domain: {cost.candidate_id: cost for cost in domain_costs}
            for domain, domain_costs in costs.items()
        }
        for rule in self.rules:
            selected = by_domain[rule.domain][rule.candidate_id]
            challengers = tuple(
                cost
                for candidate, cost in by_domain[rule.domain].items()
                if candidate != rule.candidate_id
            )
            if challengers:
                self.assertAlmostEqual(selected.max_surface_regret, 0.2)
                self.assertEqual(challengers[0].max_surface_regret, 0.0)
            else:
                self.assertEqual(selected.max_surface_regret, 0.0)

        burned_dimensions = {
            f"burned-{index}": (witness.n, witness.k)
            for index, witness in enumerate(plan.rule_witnesses)
        }
        next_plan = build_cpu_grouped_sealed_plan(
            self.rules,
            FROZEN_DIGEST,
            self.development,
            self.manifest,
            SEALED_BUILD_ID,
            burned_dimensions,
        )
        self.assertTrue(all(
            (witness.n, witness.k) not in set(burned_dimensions.values())
            for witness in next_plan.rule_witnesses
        ))
        with self.assertRaisesRegex(ValueError, "visible to fitting"):
            validate_cpu_grouped_sealed_plan(
                plan,
                self.rules,
                self.development,
                self.manifest,
                SEALED_BUILD_ID,
                burned_dimensions,
            )

    def test_legacy_corun_seal_remains_development_only_evidence(self) -> None:
        """A burned v3 seal may refine a fit but can never certify one."""

        plan = self._plan()
        with tempfile.TemporaryDirectory() as temporary:
            evidence_path = Path(temporary) / "paired.csv"
            self._write_paired_evidence(
                evidence_path,
                plan,
                process_isolated=False,
            )
            costs = cpu_grouped_burned_seal_costs(
                self.development,
                plan,
                (evidence_path,),
            )

        self.assertEqual(set(costs), {rule.domain for rule in self.rules})

    def test_burned_seal_informs_an_additively_extended_corpus(self) -> None:
        """Historical paired costs survive later generic geometry additions."""

        plan = self._plan()
        extended = _Development(
            (*tuple(self.development), SimpleNamespace(
                shape_group_id="later-development-geometry",
                aggregate_n=1024,
                k=768,
            )),
            tuple(rule.domain for rule in self.rules),
            digest="sha256:" + "d" * 64,
        )
        self.assertNotEqual(plan.development_corpus_digest, extended.digest())
        with tempfile.TemporaryDirectory() as temporary:
            evidence_path = Path(temporary) / "paired.csv"
            self._write_paired_evidence(evidence_path, plan)
            costs = cpu_grouped_burned_seal_costs(
                extended,
                plan,
                (evidence_path,),
            )

        self.assertEqual(set(costs), {rule.domain for rule in self.rules})

    def test_missing_source_alias_or_challenger_is_rejected(self) -> None:
        plan = self._plan()
        incomplete = replace(plan, requests=plan.requests[1:])

        with self.assertRaisesRegex(ValueError, "challenger matrix incomplete"):
            validate_cpu_grouped_sealed_plan(
                incomplete,
                self.rules,
                self.development,
                self.manifest,
                SEALED_BUILD_ID,
            )

    def test_witness_m_cannot_be_relabelled(self) -> None:
        plan = self._plan()
        changed = replace(
            plan,
            rule_witnesses=(
                replace(plan.rule_witnesses[0], m=3),
                plan.rule_witnesses[1],
            ),
        )

        with self.assertRaisesRegex(ValueError, "does not exercise rule 0"):
            validate_cpu_grouped_sealed_plan(
                changed,
                self.rules,
                self.development,
                self.manifest,
                SEALED_BUILD_ID,
            )

    def test_sealed_build_change_burns_the_grouped_plan(self) -> None:
        """A rebuilt trainer cannot reuse a prior post-freeze holdout."""

        plan = self._plan()
        with self.assertRaisesRegex(ValueError, "sealed build digest changed"):
            validate_cpu_grouped_sealed_plan(
                plan,
                self.rules,
                self.development,
                self.manifest,
                "sha256:" + "e" * 64,
            )


if __name__ == "__main__":
    unittest.main()
