#!/usr/bin/env python3
"""Run provenance-bound per-candidate NativeVNNI profiler collection."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.profiler_collectors import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
