#!/usr/bin/env python3
"""Regression tests for failed-fit CPU prefill refinement transactions."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch import cpu_prefill_generic_refinement  # noqa: E402
from native_vnni_dispatch.cpu_prefill_generic_refinement import (  # noqa: E402
    _sha256_mapping,
    build_cpu_prefill_generic_refinement_plan,
    build_cpu_prefill_generic_refinement_probe_plan,
    development_fit_unpromoted_domain_count,
    read_cpu_prefill_generic_refinement_plan,
    read_cpu_prefill_generic_refinement_probe_plan,
    validate_cpu_prefill_generic_refinement_plan_source,
    write_cpu_prefill_generic_refinement_plan,
    write_cpu_prefill_generic_refinement_probe_plan,
)
from native_vnni_dispatch.cpu_prefill_route_manifest import (  # noqa: E402
    CPU_PREFILL_FULL_K_BUNDLE,
    CPUPrefillSerialRoute,
    CPUPrefillSerialRouteManifest,
)
from native_vnni_dispatch.cpu_prefill_split_manifest import (  # noqa: E402
    load_cpu_prefill_split_manifest,
)
from native_vnni_dispatch.prefill_matrix import (  # noqa: E402
    CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS,
    cpu_prefill_maximum_weight_elements_for_m,
)
from native_vnni_dispatch.shape_manifest import (  # noqa: E402
    ShapePartition,
    ShapeRole,
    load_declared_shape_manifest,
    load_shape_manifest,
)


ISA_REGIME = "avx512-build.avx512-runtime"
ARCHITECTURE = "test-host|build=AVX512|runtime=AVX512|threads=28"


class CPUNativeVNNIPrefillGenericRefinementTest(unittest.TestCase):
    """Prove refinement plans are complete, fresh, and route authenticated."""

    @staticmethod
    def _eligible_shapes():
        split = load_cpu_prefill_split_manifest()
        manifest = load_shape_manifest()
        sealed = split.shape_names(sealed=True)
        return tuple(
            shape
            for shape in manifest.shapes
            if shape.role == ShapeRole.CERTIFICATION
            and not shape.exact_overlay
            and shape.name not in sealed
            and shape.prefill_partition != ShapePartition.SEALED
            and shape.work_items
            <= CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS
        )

    @classmethod
    def _routes(cls) -> CPUPrefillSerialRouteManifest:
        return CPUPrefillSerialRouteManifest(
            CPUPrefillSerialRoute(
                execution_codebook=0,
                shape_name=shape.name,
                n=shape.n,
                k=shape.k,
                isa_regime=ISA_REGIME,
                payload_bytes=16,
                threads=28,
                k_tiles=0,
                bundle_signature=CPU_PREFILL_FULL_K_BUNDLE,
            )
            for shape in cls._eligible_shapes()
        )

    @staticmethod
    def _domain() -> dict[str, object]:
        return {
            "all_aspects": True,
            "architecture_class": ARCHITECTURE,
            "aspect_bucket": "balanced",
            "backend": "cpu",
            "bundle_signature": CPU_PREFILL_FULL_K_BUNDLE,
            "execution_mode": "eager",
            "m": 64,
            "operation_kind": "NativeVNNIPrefillProjection",
            "packing_abi": "native-vnni-cpu-cb0-v1",
            "prepared_family_id": "NativeVNNI_cpu_CB0",
            "runtime_codebook_id": 0,
            "semantic_contract": "VerifierSerialM1Bitwise",
        }

    @classmethod
    def _write_fit(
        cls,
        path: Path,
        *,
        include_cv: bool,
        state: str = "development_fit_diagnostic_noninstallable",
        final_fit_target: tuple[int, int] | None = None,
    ) -> None:
        domain = cls._domain()
        shapes = cls._eligible_shapes()
        middle = shapes[len(shapes) // 2]
        cross_validation = []
        if include_cv:
            cross_validation.append({
                "domain": domain,
                "cells": [],
                "max_regret": 0.08,
                "worst_aggregate_n": middle.n,
                "worst_k": middle.k,
            })
        policy = {
            "policy_abi": "native-vnni-dispatch-v1",
            "learner_version": "test",
            "feature_schema_version": "test",
            "generic_rules": [],
            "unpromoted_domains": [domain],
            "cross_validation": cross_validation,
            "metadata": {
                "development_corpus_digest": "sha256:development",
            },
        }
        artifact = {
            "state": state,
            "policy": policy,
            "policy_digest": _sha256_mapping(policy),
            "frozen_generic_policy_digest": "sha256:generic",
        }
        if final_fit_target is not None:
            diagnostics = [{
                "domain": domain,
                "rejection_stage": "final_fit_p95",
                "final_worst_aggregate_n": final_fit_target[0],
                "final_worst_k": final_fit_target[1],
                "final_worst_p95_regret": 0.04,
            }]
            artifact["promotion_diagnostics"] = diagnostics
            artifact["promotion_diagnostics_digest"] = _sha256_mapping({
                "promotion_diagnostics": diagnostics,
            })
        path.write_text(json.dumps(artifact), encoding="utf-8")

    def test_boundary_plan_round_trip_reconstructs_every_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit = root / "fit.json"
            output = root / "plan.json"
            self._write_fit(fit, include_cv=True)
            routes = self._routes()
            split = load_cpu_prefill_split_manifest()

            plan = build_cpu_prefill_generic_refinement_plan(
                fit, routes, split
            )
            self.assertEqual(len(plan.obligations), 1)
            self.assertEqual(len(plan.obligations[0].selected_shapes), 4)
            self.assertEqual(plan.obligations[0].threads, 28)
            self.assertEqual(len(plan.records), 4)
            manifest = load_shape_manifest()
            self.assertTrue(all(
                manifest.by_name(name).work_items
                <= cpu_prefill_maximum_weight_elements_for_m(
                    plan.obligations[0].m
                )
                for name in plan.obligations[0].selected_shapes
            ))
            write_cpu_prefill_generic_refinement_plan(output, plan)

            restored = read_cpu_prefill_generic_refinement_plan(
                output, routes, split
            )
            self.assertEqual(restored.canonical_mapping(), plan.canonical_mapping())
            self.assertEqual(restored.digest(), plan.digest())

    def test_generated_probe_recovers_an_exhausted_static_neighborhood(self) -> None:
        """Later rounds retain four fresh, production-route-probed CV groups."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit = root / "fit.json"
            probe_path = root / "probe.json"
            plan_path = root / "plan.json"
            self._write_fit(fit, include_cv=True)

            raw = json.loads(fit.read_text(encoding="utf-8"))
            raw["policy"]["cross_validation"][0]["cells"] = [
                {
                    "runtime_key": {
                        "aggregate_n": shape.n,
                        "k": shape.k,
                    }
                }
                for shape in self._eligible_shapes()
            ]
            raw["policy_digest"] = _sha256_mapping(raw["policy"])
            fit.write_text(json.dumps(raw), encoding="utf-8")

            probe = build_cpu_prefill_generic_refinement_probe_plan(fit)
            self.assertGreaterEqual(len(probe.shapes), 4)
            self.assertTrue(all(
                shape.name.startswith("CPUPrefillAutoRefine_")
                for shape in probe.shapes
            ))
            write_cpu_prefill_generic_refinement_probe_plan(probe_path, probe)
            restored_probe = read_cpu_prefill_generic_refinement_probe_plan(
                probe_path,
                fit,
            )
            self.assertEqual(restored_probe.digest(), probe.digest())

            probe_routes = CPUPrefillSerialRouteManifest(
                CPUPrefillSerialRoute(
                    execution_codebook=0,
                    shape_name=shape.name,
                    n=shape.n,
                    k=shape.k,
                    isa_regime=ISA_REGIME,
                    payload_bytes=16,
                    threads=28,
                    k_tiles=0,
                    bundle_signature=CPU_PREFILL_FULL_K_BUNDLE,
                )
                for shape in probe.shapes
            )
            base_routes = self._routes()
            plan = build_cpu_prefill_generic_refinement_plan(
                fit,
                base_routes,
                load_cpu_prefill_split_manifest(),
                probe_plan=probe,
                probe_route_manifest=probe_routes,
            )

            selected = plan.obligations[0].selected_shapes
            self.assertEqual(len(selected), 4)
            self.assertTrue(all(
                name.startswith("CPUPrefillAutoRefine_") for name in selected
            ))
            self.assertEqual(
                {shape.name for shape in plan.refinement_shapes},
                set(selected),
            )
            self.assertEqual(len(plan.refinement_routes), 4)

            write_cpu_prefill_generic_refinement_plan(plan_path, plan)
            restored = read_cpu_prefill_generic_refinement_plan(
                plan_path,
                base_routes,
                load_cpu_prefill_split_manifest(),
            )
            self.assertEqual(restored.canonical_mapping(), plan.canonical_mapping())

    def test_generated_plan_rejects_tampered_embedded_route(self) -> None:
        """A resumed v3 plan cannot rewrite a generated C++ route witness."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit = root / "fit.json"
            plan_path = root / "plan.json"
            self._write_fit(fit, include_cv=True)
            raw_fit = json.loads(fit.read_text(encoding="utf-8"))
            raw_fit["policy"]["cross_validation"][0]["cells"] = [
                {
                    "runtime_key": {
                        "aggregate_n": shape.n,
                        "k": shape.k,
                    }
                }
                for shape in self._eligible_shapes()
            ]
            raw_fit["policy_digest"] = _sha256_mapping(raw_fit["policy"])
            fit.write_text(json.dumps(raw_fit), encoding="utf-8")
            probe = build_cpu_prefill_generic_refinement_probe_plan(fit)
            probe_routes = CPUPrefillSerialRouteManifest(
                CPUPrefillSerialRoute(
                    execution_codebook=0,
                    shape_name=shape.name,
                    n=shape.n,
                    k=shape.k,
                    isa_regime=ISA_REGIME,
                    payload_bytes=16,
                    threads=28,
                    k_tiles=0,
                    bundle_signature=CPU_PREFILL_FULL_K_BUNDLE,
                )
                for shape in probe.shapes
            )
            base_routes = self._routes()
            plan = build_cpu_prefill_generic_refinement_plan(
                fit,
                base_routes,
                load_cpu_prefill_split_manifest(),
                probe_plan=probe,
                probe_route_manifest=probe_routes,
            )
            write_cpu_prefill_generic_refinement_plan(plan_path, plan)
            raw = json.loads(plan_path.read_text(encoding="utf-8"))
            raw["refinement_routes"][0]["n"] += 1
            plan_path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "route changed"):
                read_cpu_prefill_generic_refinement_plan(
                    plan_path,
                    base_routes,
                    load_cpu_prefill_split_manifest(),
                )

    def test_final_fit_failure_refines_its_leaf_not_the_cv_worst(self) -> None:
        """A CV-passing final-tree miss must use final-fit target geometry."""

        with tempfile.TemporaryDirectory() as temporary:
            fit = Path(temporary) / "fit.json"
            shapes = self._eligible_shapes()
            target = shapes[len(shapes) // 3]
            self._write_fit(
                fit,
                include_cv=True,
                final_fit_target=(target.n, target.k),
            )

            plan = build_cpu_prefill_generic_refinement_plan(
                fit,
                self._routes(),
                load_cpu_prefill_split_manifest(),
            )

            obligation = plan.obligations[0]
            self.assertEqual(obligation.worst_n, target.n)
            self.assertEqual(obligation.worst_k, target.k)
            self.assertEqual(obligation.cv_max_regret, 0.04)

    def test_refinement_plan_binds_final_fit_target_diagnostics(self) -> None:
        """A rewritten final-leaf target invalidates an existing v2 plan."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit = root / "fit.json"
            observations = root / "observations.csv"
            target = self._eligible_shapes()[0]
            self._write_fit(
                fit,
                include_cv=True,
                final_fit_target=(target.n, target.k),
            )
            plan = build_cpu_prefill_generic_refinement_plan(
                fit,
                self._routes(),
                load_cpu_prefill_split_manifest(),
            )
            raw = json.loads(fit.read_text(encoding="utf-8"))
            raw["promotion_diagnostics"][0][
                "final_worst_aggregate_n"
            ] += 64
            raw["promotion_diagnostics_digest"] = _sha256_mapping({
                "promotion_diagnostics": raw["promotion_diagnostics"],
            })
            fit.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "source fit changed",
            ), mock.patch.object(
                cpu_prefill_generic_refinement,
                "read_observation_csv",
            ) as reader:
                validate_cpu_prefill_generic_refinement_plan_source(
                    plan,
                    fit,
                    observations,
                )
            reader.assert_not_called()

    def test_v1_refinement_plan_remains_byte_stable_and_readable(self) -> None:
        """Authenticated historical plans keep their original canonical IR."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit = root / "fit.json"
            legacy_path = root / "legacy-plan.json"
            self._write_fit(fit, include_cv=True)
            routes = self._routes()
            split = load_cpu_prefill_split_manifest()
            plan = build_cpu_prefill_generic_refinement_plan(
                fit,
                routes,
                split,
            )
            legacy = plan.canonical_mapping()
            legacy["schema_version"] = getattr(
                cpu_prefill_generic_refinement,
                "LEGACY_CPU_PREFILL_GENERIC_REFINEMENT_SCHEMA",
            )
            legacy.pop("source_promotion_diagnostics_digest")
            legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

            restored = read_cpu_prefill_generic_refinement_plan(
                legacy_path,
                routes,
                split,
            )

            self.assertIsNone(
                restored.source_promotion_diagnostics_digest
            )
            self.assertEqual(restored.canonical_mapping(), legacy)

    def test_pre_release_shape_manifest_identity_remains_readable(self) -> None:
        """Adding release overlays must not invalidate referenced old shapes."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit = root / "fit.json"
            historical_path = root / "historical-plan.json"
            self._write_fit(fit, include_cv=True)
            routes = self._routes()
            split = load_cpu_prefill_split_manifest()
            plan = build_cpu_prefill_generic_refinement_plan(
                fit,
                routes,
                split,
            )
            historical = plan.canonical_mapping()
            historical["shape_manifest_digest"] = (
                load_declared_shape_manifest().digest()
            )
            historical_path.write_text(
                json.dumps(historical),
                encoding="utf-8",
            )

            restored = read_cpu_prefill_generic_refinement_plan(
                historical_path,
                routes,
                split,
            )

            self.assertEqual(
                restored.shape_manifest_digest,
                "sha256:253fa231865269283241960b39bc733a141d0ff3a8ca421ca1318db476b28231",
            )

    def test_domain_without_cv_receives_three_geometry_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fit = Path(temporary) / "fit.json"
            self._write_fit(
                fit,
                include_cv=False,
                state="frozen_development",
            )
            plan = build_cpu_prefill_generic_refinement_plan(
                fit,
                self._routes(),
                load_cpu_prefill_split_manifest(),
            )

            self.assertEqual(plan.source_fit_state, "frozen_development")
            self.assertEqual(len(plan.obligations[0].selected_shapes), 3)
            self.assertEqual(len(plan.records), 3)
            selected = [
                load_shape_manifest().by_name(name).work_items
                for name in plan.obligations[0].selected_shapes
            ]
            self.assertLessEqual(max(selected), min(selected) * 2.1)
            self.assertEqual(development_fit_unpromoted_domain_count(fit), 1)

    def test_existing_plan_reauthenticates_fit_and_observation_corpus(self) -> None:
        """A resumed collection cannot pair a plan with foreign evidence."""

        with tempfile.TemporaryDirectory() as temporary:
            fit = Path(temporary) / "fit.json"
            observations = Path(temporary) / "observations.csv"
            self._write_fit(fit, include_cv=True)
            plan = build_cpu_prefill_generic_refinement_plan(
                fit,
                self._routes(),
                load_cpu_prefill_split_manifest(),
            )
            collapsed = mock.Mock()
            collapsed.digest.return_value = plan.source_development_corpus_digest
            corpus = mock.Mock()
            corpus.with_collapsed_aspect_domains.return_value = collapsed
            with mock.patch.object(
                cpu_prefill_generic_refinement,
                "read_observation_csv",
                return_value=corpus,
            ) as reader:
                validate_cpu_prefill_generic_refinement_plan_source(
                    plan,
                    fit,
                    observations,
                )

            reader.assert_called_once_with((observations,))

    def test_existing_plan_rejects_foreign_fit_before_reading_observations(self) -> None:
        """Policy provenance is checked before opening a potentially huge CSV."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit = root / "fit.json"
            foreign_fit = root / "foreign-fit.json"
            self._write_fit(fit, include_cv=True)
            self._write_fit(foreign_fit, include_cv=True)
            plan = build_cpu_prefill_generic_refinement_plan(
                fit,
                self._routes(),
                load_cpu_prefill_split_manifest(),
            )
            raw = json.loads(foreign_fit.read_text(encoding="utf-8"))
            raw["policy"]["learner_version"] = "foreign"
            raw["policy_digest"] = _sha256_mapping(raw["policy"])
            foreign_fit.write_text(json.dumps(raw), encoding="utf-8")
            with mock.patch.object(
                cpu_prefill_generic_refinement,
                "read_observation_csv",
            ) as reader:
                with self.assertRaisesRegex(ValueError, "source fit changed"):
                    validate_cpu_prefill_generic_refinement_plan_source(
                        plan,
                        foreign_fit,
                        root / "observations.csv",
                    )

            reader.assert_not_called()

    def test_existing_plan_rejects_foreign_observation_corpus(self) -> None:
        """Matching fit JSON cannot authorize a different adapted corpus."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit = root / "fit.json"
            self._write_fit(fit, include_cv=True)
            plan = build_cpu_prefill_generic_refinement_plan(
                fit,
                self._routes(),
                load_cpu_prefill_split_manifest(),
            )
            collapsed = mock.Mock()
            collapsed.digest.return_value = "sha256:foreign-development"
            corpus = mock.Mock()
            corpus.with_collapsed_aspect_domains.return_value = collapsed
            with mock.patch.object(
                cpu_prefill_generic_refinement,
                "read_observation_csv",
                return_value=corpus,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "source observations changed",
                ):
                    validate_cpu_prefill_generic_refinement_plan_source(
                        plan,
                        fit,
                        root / "observations.csv",
                    )

    def test_large_m_refinement_excludes_32b_class_geometries(self) -> None:
        """M=16384 refinement remains inside the small-model work envelope."""

        with tempfile.TemporaryDirectory() as temporary:
            fit = Path(temporary) / "fit.json"
            domain = self._domain()
            domain["m"] = 16384
            policy = {
                "policy_abi": "native-vnni-dispatch-v1",
                "learner_version": "test",
                "feature_schema_version": "test",
                "generic_rules": [],
                "unpromoted_domains": [domain],
                "cross_validation": [],
                "metadata": {
                    "development_corpus_digest": "sha256:development",
                },
            }
            fit.write_text(json.dumps({
                "state": "development_fit_diagnostic_noninstallable",
                "policy": policy,
                "policy_digest": _sha256_mapping(policy),
                "frozen_generic_policy_digest": "sha256:generic",
            }), encoding="utf-8")

            plan = build_cpu_prefill_generic_refinement_plan(
                fit,
                self._routes(),
                load_cpu_prefill_split_manifest(),
            )
            cap = cpu_prefill_maximum_weight_elements_for_m(16384)
            manifest = load_shape_manifest()
            self.assertTrue(all(
                manifest.by_name(name).work_items <= cap
                for name in plan.obligations[0].selected_shapes
            ))
            self.assertTrue(all(record.n * record.k <= cap for record in plan.records))

    def test_refinement_pool_excludes_decode_sized_geometries(self) -> None:
        """Ordinary prefill never borrows the larger shared decode envelope."""

        eligible = {shape.name for shape in self._eligible_shapes()}
        self.assertNotIn("V3VerifierSealed_Tall_9216x32768", eligible)
        self.assertTrue(all(
            shape.work_items <= CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS
            for shape in self._eligible_shapes()
        ))

    def test_reader_rejects_records_not_derived_from_obligations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit = root / "fit.json"
            output = root / "plan.json"
            self._write_fit(fit, include_cv=True)
            routes = self._routes()
            split = load_cpu_prefill_split_manifest()
            plan = build_cpu_prefill_generic_refinement_plan(
                fit, routes, split
            )
            write_cpu_prefill_generic_refinement_plan(output, plan)
            raw = json.loads(output.read_text(encoding="utf-8"))
            raw["records"][0]["m_values"] = [64, 256]
            output.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "do not match"):
                read_cpu_prefill_generic_refinement_plan(output, routes, split)

    def test_split_authorizes_only_fresh_non_overlay_development_shapes(self) -> None:
        split = load_cpu_prefill_split_manifest()
        eligible = "Generic_VeryWide_Boundary"
        corpus = [
            SimpleNamespace(shape_name=name)
            for name in (*split.development_shapes, eligible)
        ]
        split.require_partition(
            corpus,
            sealed=False,
            additional_development_shapes=(eligible,),
        )

        with self.assertRaisesRegex(ValueError, "not eligible"):
            split.require_partition(
                corpus,
                sealed=False,
                additional_development_shapes=("0.5B_AttnOut",),
            )
        with self.assertRaisesRegex(ValueError, "not eligible"):
            split.require_partition(
                corpus,
                sealed=False,
                additional_development_shapes=(split.sealed_shapes[0],),
            )
        burned_name = split.sealed_shapes[0]
        burned_shape = load_shape_manifest().by_name(burned_name)
        burned_corpus = [
            SimpleNamespace(shape_name=name)
            for name in (*split.development_shapes, burned_name)
        ]
        split.require_partition(
            burned_corpus,
            sealed=False,
            additional_development_shapes=(burned_name,),
            additional_development_geometries={
                burned_name: (burned_shape.n, burned_shape.k)
            },
            burned_sealed_development_shapes=(burned_name,),
        )
        with self.assertRaisesRegex(ValueError, "cannot be refined"):
            split.require_partition(
                [],
                sealed=True,
                additional_development_shapes=(eligible,),
            )

    def test_split_accepts_only_plan_owned_generated_refinement_geometry(self) -> None:
        """Dynamic boundary points require their authenticated N/K mapping."""

        split = load_cpu_prefill_split_manifest()
        generated = "CPUPrefillAutoRefine_N2496_K9152"
        corpus = [
            SimpleNamespace(shape_name=name)
            for name in (*split.development_shapes, generated)
        ]
        split.require_partition(
            corpus,
            sealed=False,
            additional_development_shapes=(generated,),
            additional_development_geometries={generated: (2496, 9152)},
        )

        with self.assertRaisesRegex(ValueError, "unknown"):
            split.require_partition(
                corpus,
                sealed=False,
                additional_development_shapes=(generated,),
            )
        with self.assertRaisesRegex(ValueError, "not eligible"):
            split.require_partition(
                corpus,
                sealed=False,
                additional_development_shapes=(generated,),
                additional_development_geometries={generated: (2496, 9153)},
            )


if __name__ == "__main__":
    unittest.main()
