"""Regressions for the sealed CPU NativeVNNI M=1 policy compiler."""

from __future__ import annotations

import sys
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
CPU_ANALYZER_ROOT = KERNEL_PERF_ROOT / "cpu"
for root in (KERNEL_PERF_ROOT, CPU_ANALYZER_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from analyze_cpu_native_vnni_decode_trainer import (  # noqa: E402
    CPUDecodeGenericRule,
    DECODE_BUNDLES,
    REQUIRED_ISA_REGIMES,
    DecodePolicyEntry,
    _candidate_policy,
    _pack_key,
    generate_include,
    validate_total_policy,
)
from native_vnni_dispatch.corpus import GenericDomain  # noqa: E402
from native_vnni_dispatch.format_registry import FORMAT_SPECS  # noqa: E402
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402
from native_vnni_dispatch.schema import (  # noqa: E402
    AspectBucket,
    Backend,
    ExecutionMode,
    SemanticContract,
)
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    FeatureAxis,
    FeaturePredicate,
    FeatureThreshold,
    GenericDispatchRule,
)


def total_rules() -> list[CPUDecodeGenericRule]:
    """Build one no-split cross-aspect leaf per runtime dispatch domain."""

    candidate_id = "cpu.nvnni.decode.n_chunk_grid.nbc4"
    codebooks = sorted({spec.runtime_codebook("cpu") for spec in FORMAT_SPECS})
    rules = []
    for build, runtime in sorted(REQUIRED_ISA_REGIMES):
        architecture = (
            f"unit|build={build}|runtime={runtime}|threads=28"
        )
        for codebook in codebooks:
            for bundle in DECODE_BUNDLES:
                rule = GenericDispatchRule(
                    domain=GenericDomain(
                        backend=Backend.CPU,
                        architecture_class=architecture,
                        semantic_contract=SemanticContract.FAST,
                        operation_kind="NativeVNNIFastM1Projection",
                        bundle_signature=bundle,
                        prepared_family_id=f"unit-family-{codebook}",
                        packing_abi=f"unit-packing-{codebook}",
                        runtime_codebook_id=codebook,
                        execution_mode=ExecutionMode.EAGER,
                        m=1,
                        aspect_bucket=AspectBucket.BALANCED,
                        all_aspects=True,
                    ),
                    predicates=(),
                    candidate_id=candidate_id,
                    arithmetic_fingerprint="unit-byte-exact-arithmetic",
                    development_shape_groups=("unit",),
                    development_max_regret=0.0,
                    development_p95_regret=0.0,
                    development_mean_regret=0.0,
                )
                rules.append(CPUDecodeGenericRule(
                    build_isa=build,
                    runtime_isa=runtime,
                    threads=28,
                    rule=rule,
                ))
    return rules


class CPUDecodeTrainerTest(unittest.TestCase):
    """Protect selector ABI, exact precedence, and generic totality."""

    def test_candidate_registry_maps_only_physical_nbc_widths(self) -> None:
        self.assertEqual(
            _candidate_policy("cpu.nvnni.decode.n_chunk_grid.nbc1"),
            "Nbc1",
        )
        self.assertEqual(
            _candidate_policy("cpu.nvnni.decode.n_chunk_grid.nbc16"),
            "Nbc16",
        )

    def test_exact_key_distinguishes_frozen_kpart_regime(self) -> None:
        full_k = _pack_key(18, False, 5120, 16384)
        kpart = _pack_key(18, True, 5120, 16384)

        self.assertNotEqual(full_k, kpart)
        self.assertEqual(full_k ^ kpart, 1 << 55)

    def test_generated_include_keeps_exact_overlay_before_generic_rules(self) -> None:
        entry = DecodePolicyEntry(
            build_isa="AVX512",
            runtime_isa="AVX2",
            threads=28,
            codebook=18,
            n=5120,
            k=16384,
            serial_kpart=True,
            policy="Nbc4",
            candidate_id="cpu.nvnni.decode.n_chunk_grid.nbc4",
            shape_names=("UnitShape",),
            max_surface_regret=0.0,
            max_cv=0.0,
        )
        include = generate_include(
            [entry],
            [
                item for item in total_rules()
                if item.build_isa == "AVX512"
                and item.runtime_isa == "AVX2"
            ],
            corpus_digest="sha256:" + "1" * 64,
            registry_digest="sha256:" + "2" * 64,
            profile=MeasurementProfile.QUICK,
        )

        exact_case = f"case 0x{_pack_key(18, True, 5120, 16384):016x}ULL"
        self.assertIn("LLAMINAR_CPU_NVNNI_DECODE_POLICY_CERTIFIED 0", include)
        self.assertLess(include.index(exact_case), include.index("work_items"))
        self.assertIn("serial_kpart", include)

    def test_totality_requires_every_cross_aspect_kpart_bundle(self) -> None:
        rules = total_rules()
        validate_total_policy(rules)
        missing_bundle = [
            item for item in rules
            if not (
                item.build_isa == "AVX2"
                and item.runtime_isa == "AVX2"
                and item.rule.domain.bundle_signature == DECODE_BUNDLES[1]
            )
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"not total: missing_count=18 .*serial-kpart",
        ):
            validate_total_policy(missing_bundle)

    def test_generated_selector_is_total_and_thread_parametric(self) -> None:
        """Exact overlays stay scoped while generic wave rules use runtime T."""

        architecture = "unit|build=AVX512|runtime=AVX512|threads=28"
        domain = GenericDomain(
            backend=Backend.CPU,
            architecture_class=architecture,
            semantic_contract=SemanticContract.FAST,
            operation_kind="NativeVNNIFastM1Projection",
            bundle_signature=DECODE_BUNDLES[0],
            prepared_family_id="unit-family-0",
            packing_abi="unit-packing-0",
            runtime_codebook_id=0,
            execution_mode=ExecutionMode.EAGER,
            m=1,
            aspect_bucket=AspectBucket.BALANCED,
            all_aspects=True,
        )
        threshold = FeatureThreshold(
            FeatureAxis.N_PARALLEL_WAVES_64,
            numerator=1,
            parallelism_width=28,
        )

        def rule(
            require_less_equal: bool,
            candidate_id: str,
        ) -> CPUDecodeGenericRule:
            return CPUDecodeGenericRule(
                build_isa="AVX512",
                runtime_isa="AVX512",
                threads=28,
                rule=GenericDispatchRule(
                    domain=domain,
                    predicates=(
                        FeaturePredicate(threshold, require_less_equal),
                    ),
                    candidate_id=candidate_id,
                    arithmetic_fingerprint="unit-byte-exact-arithmetic",
                    development_shape_groups=("unit",),
                    development_max_regret=0.0,
                    development_p95_regret=0.0,
                    development_mean_regret=0.0,
                ),
            )

        exact = DecodePolicyEntry(
            build_isa="AVX512",
            runtime_isa="AVX512",
            threads=28,
            codebook=0,
            n=1280,
            k=1024,
            serial_kpart=False,
            policy="Nbc1",
            candidate_id="cpu.nvnni.decode.n_chunk_grid.nbc1",
            shape_names=("MeasuredT28",),
            max_surface_regret=0.0,
            max_cv=0.0,
        )
        generated = generate_include(
            [exact],
            [
                rule(True, "cpu.nvnni.decode.n_chunk_grid.nbc2"),
                rule(False, "cpu.nvnni.decode.n_chunk_grid.nbc8"),
            ],
            corpus_digest="sha256:" + "1" * 64,
            registry_digest="sha256:" + "2" * 64,
            profile=MeasurementProfile.QUICK,
        )
        self.assertIn("static_cast<long long>(threads)", generated)
        source = "\n".join((
            generated,
            "#include <climits>",
            "int main() {",
            "  using namespace llaminar2::cpu::native_vnni::generated;",
            "  CPUNativeVNNIDecodePolicy policy{};",
            "  const int widths[] = {1, 2, 3, 7, 27, 28, 31, 56, 112, INT_MAX};",
            "  for (const int threads : widths) {",
            "    if (!selectCPUNativeVNNIDecodeGeneratedPolicy(",
            "            CPUNativeVNNIDecodeBuildISA::AVX512,",
            "            CPUNativeVNNIDecodeRuntimeISA::AVX512,",
            "            threads, 0, 1280, 1024, false, 0, policy))",
            "      return 1;",
            "    const auto expected = threads == 28",
            "        ? CPUNativeVNNIDecodePolicy::Nbc1",
            "        : (threads >= 20 ? CPUNativeVNNIDecodePolicy::Nbc2",
            "                         : CPUNativeVNNIDecodePolicy::Nbc8);",
            "    if (policy != expected)",
            "      return 2;",
            "  }",
            "  if (selectCPUNativeVNNIDecodeGeneratedPolicy(",
            "          CPUNativeVNNIDecodeBuildISA::AVX512,",
            "          CPUNativeVNNIDecodeRuntimeISA::AVX512,",
            "          0, 0, 1280, 1024, false, 0, policy))",
            "    return 3;",
            "  return 0;",
            "}",
        ))
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "cpu-decode-thread-totality"
            compiled = subprocess.run(
                ["g++", "-std=c++20", "-x", "c++", "-o", str(binary), "-"],
                input=source,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run(
                [str(binary)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(executed.returncode, 0, executed.stderr.decode())


if __name__ == "__main__":
    unittest.main()
