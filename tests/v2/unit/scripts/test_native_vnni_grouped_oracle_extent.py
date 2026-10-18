#!/usr/bin/env python3
"""Guard complete-row serial-equivalence oracles in every grouped trainer.

The performance trainers are promotion authorities: a green CSV row can place
its candidate into generated production dispatch.  Each oracle must therefore
compare the complete grouped publication, not merely row zero.  This focused
source contract complements the GPU/CPU integration sweeps by making an
accidental `N`-only comparison fail in the fast unit gate. The same device-free
module validates the emitted Q6 coefficient type so raw signed source bytes
cannot be silently described as floating half scales.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


class NativeVNNIGroupedOracleExtentTest(unittest.TestCase):
    """Prove every policy-producing grouped oracle covers its full tensor."""

    def read(self, relative: str) -> str:
        """Read one repository source file with a stable absolute root."""

        return (REPO_ROOT / relative).read_text(encoding="utf-8")

    def test_dense_decode_trainers_compare_m_times_n_values(self) -> None:
        """CUDA, ROCm, and CPU grouped decode must certify every row."""

        sources = (
            self.read(
                "tests/v2/performance/kernels/cuda/gemm/"
                "Perf__CUDANativeVNNIDecodeTrainer.cpp"
            ),
            self.read(
                "tests/v2/performance/kernels/rocm/"
                "Perf__NativeVNNI_Throughput.cpp"
            ),
            self.read(
                "tests/v2/performance/kernels/cpu/native_vnni/"
                "Perf__CPUNativeVNNI_GEMV.cpp"
            ),
        )
        complete_extent = (
            "static_cast<size_t>(M) * static_cast<size_t>(N)"
        )
        # CUDA names the local function parameters in lower case; ROCm and CPU
        # use upper case. Both spellings must visibly multiply both dimensions.
        self.assertIn(
            "static_cast<size_t>(m) * static_cast<size_t>(n)",
            sources[0],
        )
        for source in sources[1:]:
            self.assertIn(complete_extent, source)

    def test_grouped_moe_trainer_compares_the_downloaded_tensor(self) -> None:
        """ROCm grouped MoE gate/up and down evidence must include all rows."""

        source = self.read(
            "tests/v2/performance/kernels/moe/"
            "Perf__ROCmMoEVerifierPrefill.cpp"
        )
        self.assertGreaterEqual(source.count("actual.size());"), 2)
        self.assertNotIn(
            "serial,\n                            "
            "static_cast<size_t>(shape.d_model)",
            source,
        )



class NativeVNNISecondaryMetadataTypeTest(unittest.TestCase):
    """Prove source-factor metadata types through the actual manifest reader."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load only the standard-library comparator; never launch its emitters."""
        path = REPO_ROOT / "tests/v2/integration/kernels/moe/compare_cross_vendor_native_vnni_arithmetic.py"
        spec = importlib.util.spec_from_file_location("native_vnni_metadata_comparator", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load the NativeVNNI artifact reader")
        cls.comparator = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.comparator
        spec.loader.exec_module(cls.comparator)

    def manifest(self, format_name: str, element_type: str):
        """Parse one fresh metadata-only prepared secondary declaration."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifacts.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["backend", "format", "artifact", "bytes", "element_type", "rows", "routes", "width"])
                writer.writerow(["cuda", format_name, "prepared.variant0.down.mins.bin", 4, element_type, 0, 0, 0])
            return self.comparator._read_artifact_manifest(path, "cuda", format_name)

    def test_q6_requires_raw_u16(self) -> None:
        """Original signed coefficient pairs remain raw words in the manifest."""
        result = self.manifest("Q6_K", "u16")
        self.assertEqual(result["prepared.variant0.down.mins.bin"].element_type, "u16")

    def test_q6_rejects_old_half_tag(self) -> None:
        """A half-scale tag cannot reinterpret the new Q6 source coefficients."""
        with self.assertRaisesRegex(ValueError, "requires u16"):
            self.manifest("Q6_K", "fp16_bits")

    def test_iq2s_requires_raw_u16(self) -> None:
        """Original IQ2_S coefficient bytes are not derived half scales."""
        result = self.manifest("IQ2_S", "u16")
        self.assertEqual(result["prepared.variant0.down.mins.bin"].element_type, "u16")

    def test_iq2s_rejects_old_half_tag(self) -> None:
        """Old metadata labelling must not disguise the new source factors."""
        with self.assertRaisesRegex(ValueError, "requires u16"):
            self.manifest("IQ2_S", "fp16_bits")

    def test_iq3_source_secondary_is_raw(self):
        """IQ3 source nibbles remain raw metadata in actual manifests."""
        self.manifest("IQ3_S", "u16")

    def test_iq3_derived_half_secondary_is_rejected(self):
        """Stale source typing cannot disguise old narrowed factors."""
        with self.assertRaises(ValueError):
            self.manifest("IQ3_S", "fp16_bits")

    def test_iq4xs_requires_raw_u16(self) -> None:
        """Original six-bit source coefficients must not be labelled half minima."""
        result = self.manifest("IQ4_XS", "u16")
        self.assertEqual(result["prepared.variant0.down.mins.bin"].element_type, "u16")

    def test_iq4xs_rejects_old_half_tag(self) -> None:
        """Stale IQ4_XS derived-scale typing is rejected before payload comparison."""
        with self.assertRaisesRegex(ValueError, "requires u16"):
            self.manifest("IQ4_XS", "fp16_bits")

    def test_other_dual_scale_stays_half(self) -> None:
        """Unchanged IQ2_XS secondary publication remains a derived half."""
        result = self.manifest("IQ2_XS", "fp16_bits")
        self.assertEqual(result["prepared.variant0.down.mins.bin"].element_type, "fp16_bits")

    def test_other_dual_scale_rejects_raw_q6_tag(self) -> None:
        """Raw coefficient typing is not a blanket relaxation for other formats."""
        with self.assertRaisesRegex(ValueError, "requires fp16_bits"):
            self.manifest("IQ2_XS", "u16")


if __name__ == "__main__":
    unittest.main()
