#!/usr/bin/env python3
"""Guard complete-row serial-equivalence oracles in every grouped trainer.

The performance trainers are promotion authorities: a green CSV row can place
its candidate into generated production dispatch.  Each oracle must therefore
compare the complete grouped publication, not merely row zero.  This focused
source contract complements the GPU/CPU integration sweeps by making an
accidental `N`-only comparison fail in the fast unit gate.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
