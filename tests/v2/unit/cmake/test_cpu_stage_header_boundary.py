#!/usr/bin/env python3
"""@file test_cpu_stage_header_boundary.py
@brief Keep cache-publication orchestration independent of ISA-only intrinsic headers.

The genuine native GNU14 full-engine compile reproduced an unused unconditional
immintrin.h include in KVCacheAppendStage.cpp. Numerical codecs already own their
native implementations; the stage must not introduce another ISA dependency.
This source architecture regression complements the actual-TU compiler proof;
it does not qualify cache arithmetic, leases, runtime inference or performance.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[4]
STAGE = ROOT / 'src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp'


def require_stage_header_boundary(source):
    """Reject direct intrinsic headers; actual codecs retain their own compiler envelope."""
    code = re.sub(r'/\*.*?\*/|//[^\n]*', '', source, flags=re.DOTALL)
    forbidden = re.findall(r'^\s*#\s*include\s*[<"]([A-Za-z0-9_]*intrin\.h|arm_neon\.h)[>"]',
                           code, flags=re.MULTILINE)
    if forbidden:
        raise ValueError('Cache publication stage must delegate ISA arithmetic, not include ' + ','.join(forbidden))


class CPUStageHeaderBoundaryTests(unittest.TestCase):
    """Device-free stage/kernel authority separation, including old-source controls."""

    def test_cache_append_delegates_isa_specific_arithmetic(self):
        """The real stage consumes typed native codecs without a direct intrinsic SDK."""
        require_stage_header_boundary(STAGE.read_text(encoding='utf-8'))

    def test_x86_and_arm_direct_header_controls_are_rejected(self):
        """The rule is architectural, not an ARM-only header-removal exception."""
        source = STAGE.read_text(encoding='utf-8')
        for name in ('immintrin.h', 'xmmintrin.h', 'x86intrin.h', 'arm_neon.h'):
            with self.subTest(header=name), self.assertRaisesRegex(ValueError, name):
                require_stage_header_boundary(source + f'\n#include <{name}>\n')
        require_stage_header_boundary(source + '\n/* #include <immintrin.h> */\n')


if __name__ == '__main__':
    unittest.main()
