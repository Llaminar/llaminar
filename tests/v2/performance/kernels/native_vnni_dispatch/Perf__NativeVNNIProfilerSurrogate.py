#!/usr/bin/env python3
"""Measure the production CUDA profiler-surrogate training transaction.

NativeVNNI policy fitting learns one leakage-controlled profiler surface for
each held-out geometry set.  The production implementation uses XGBoost's CUDA
histogram algorithm, while the unit suite replaces it with a deterministic
stub so unit tests never initialize a GPU.  This performance test exercises the
real learner on a production-sized dense feature matrix, verifies bitwise
repeatability, and reports the wall-clock rate used to project complete fits.

The synthetic matrix intentionally contains nonlinear geometry, work-size,
candidate, and profiler interactions.  It is not a policy-quality oracle; the
authenticated corpus and sealed cross-validation gate own that proof.  Its
purpose is to prevent the learner transaction from quietly returning to a
long-running host implementation or losing deterministic CUDA execution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.profiler_model import (  # noqa: E402
    _fit_profiler_surrogate,
)


def _fixture(
    *,
    training_rows: int,
    prediction_rows: int,
    feature_count: int,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic dense evidence with candidate/profile structure."""

    total_rows = training_rows + prediction_rows
    row = np.arange(total_rows, dtype=np.float32)[:, np.newaxis]
    column = np.arange(feature_count, dtype=np.float32)[np.newaxis, :]
    # The modular term represents categorical candidate/configuration inputs;
    # smooth trigonometric terms represent geometry and normalized counters.
    matrix = (
        np.remainder(row * (column + 3.0), 97.0) / 97.0
        + np.sin(row * 0.0007 + column * 0.071)
        + np.cos(row * 0.00013 * (column + 1.0))
    ).astype(np.float32, copy=False)
    training_matrix = np.ascontiguousarray(matrix[:training_rows])
    prediction_matrix = np.ascontiguousarray(matrix[training_rows:])

    target_column = np.arange(target_count, dtype=np.float32)[np.newaxis, :]
    targets = (
        np.sin(row[:training_rows] * (0.00031 + target_column * 0.000017))
        + np.cos(row[:training_rows] * 0.00011 + target_column * 0.19)
        + training_matrix[:, :target_count] * 0.23
        + training_matrix[:, -target_count:] * 0.07
    ).astype(np.float32, copy=False)
    return training_matrix, np.ascontiguousarray(targets), prediction_matrix


def main() -> int:
    """Run two real CUDA fits and require deterministic, bounded execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--peer-device",
        type=int,
        help="second same-architecture CUDA device used for parity",
    )
    parser.add_argument("--training-rows", type=int, default=65536)
    parser.add_argument("--prediction-rows", type=int, default=16384)
    parser.add_argument("--features", type=int, default=160)
    parser.add_argument("--targets", type=int, default=12)
    parser.add_argument("--target-seconds", type=float, default=30.0)
    arguments = parser.parse_args()
    if arguments.device < 0:
        parser.error("device must be non-negative")
    if arguments.peer_device is not None and (
        arguments.peer_device < 0 or arguments.peer_device == arguments.device
    ):
        parser.error("peer-device must be a distinct non-negative ordinal")
    if min(
        arguments.training_rows,
        arguments.prediction_rows,
        arguments.features,
        arguments.targets,
    ) <= 1:
        parser.error("matrix dimensions and target count must exceed one")
    if arguments.targets > arguments.features:
        parser.error("targets cannot exceed feature count")
    if arguments.target_seconds <= 0.0:
        parser.error("target-seconds must be positive")

    fixture_started = time.perf_counter()
    training_matrix, targets, prediction_matrix = _fixture(
        training_rows=arguments.training_rows,
        prediction_rows=arguments.prediction_rows,
        feature_count=arguments.features,
        target_count=arguments.targets,
    )
    fixture_seconds = time.perf_counter() - fixture_started
    device = f"cuda:{arguments.device}"

    fit_seconds = []
    predictions = []
    for _ in range(2):
        started = time.perf_counter()
        predictions.append(_fit_profiler_surrogate(
            training_matrix,
            targets,
            prediction_matrix,
            device=device,
        ))
        fit_seconds.append(time.perf_counter() - started)

    peer_seconds = None
    if arguments.peer_device is not None:
        peer_started = time.perf_counter()
        peer_prediction = _fit_profiler_surrogate(
            training_matrix,
            targets,
            prediction_matrix,
            device=f"cuda:{arguments.peer_device}",
        )
        peer_seconds = time.perf_counter() - peer_started
        if not np.array_equal(predictions[0], peer_prediction):
            maximum_delta = float(np.max(np.abs(
                predictions[0] - peer_prediction
            )))
            raise RuntimeError(
                "CUDA profiler surrogate changed across peer devices: "
                f"maximum_delta={maximum_delta:.9g}"
            )

    if not np.array_equal(predictions[0], predictions[1]):
        maximum_delta = float(np.max(np.abs(
            predictions[0] - predictions[1]
        )))
        raise RuntimeError(
            "CUDA profiler surrogate is not bitwise deterministic: "
            f"maximum_delta={maximum_delta:.9g}"
        )
    if not np.isfinite(predictions[0]).all():
        raise RuntimeError("CUDA profiler surrogate emitted non-finite values")

    warm_seconds = fit_seconds[-1]
    report = {
        "device": device,
        "training_rows": arguments.training_rows,
        "prediction_rows": arguments.prediction_rows,
        "features": arguments.features,
        "targets": arguments.targets,
        "fixture_seconds": fixture_seconds,
        "cold_fit_seconds": fit_seconds[0],
        "warm_fit_seconds": warm_seconds,
        "peer_device": (
            None
            if arguments.peer_device is None
            else f"cuda:{arguments.peer_device}"
        ),
        "peer_fit_seconds": peer_seconds,
        "training_rows_per_second": arguments.training_rows / warm_seconds,
        "bitwise_repeatable": True,
        "peer_device_bitwise_equal": arguments.peer_device is not None,
        "target_seconds": arguments.target_seconds,
    }
    print(json.dumps(report, sort_keys=True))
    if warm_seconds > arguments.target_seconds:
        raise RuntimeError(
            "CUDA profiler surrogate missed its per-surface economy target: "
            f"observed={warm_seconds:.3f}s target={arguments.target_seconds:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
