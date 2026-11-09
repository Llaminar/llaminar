#!/usr/bin/env python3
"""Reject the retired NativeQuantised generic-fit performance campaign.

Exact-only tuning uses the shared refresh transaction. Historical policy
readers and separate native arithmetic/replay integration tests retain their
own modules; this former entrypoint cannot launch fitting or GPU work.
"""
from __future__ import annotations

from pathlib import Path
import sys

KERNELS = Path(__file__).resolve().parents[1]
if str(KERNELS) not in sys.path:
    sys.path.insert(0, str(KERNELS))


def main() -> int:
    """Emit the canonical retirement diagnostic before discovery or work."""
    from native_quantised_dispatch.workflow_cli import retired_training_cli
    return retired_training_cli('Perf__NativeQuantisedPolicyCrossValidation')


if __name__ == "__main__":
    raise SystemExit(main())
