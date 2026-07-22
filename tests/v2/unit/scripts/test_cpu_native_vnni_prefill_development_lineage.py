#!/usr/bin/env python3
"""Regression tests for additive CPU prefill split migrations."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.cpu_prefill_development_lineage import (  # noqa: E402
    _print_launch_records,
    build_cpu_prefill_development_lineage_plan,
    read_cpu_prefill_development_lineage_plan,
    write_cpu_prefill_development_lineage_plan,
)
from native_vnni_dispatch.cpu_prefill_training_plan import (  # noqa: E402
    CPUPrefillSourceTrainingRecord,
)
from native_vnni_dispatch.shape_manifest import (  # noqa: E402
    load_shape_manifest,
)


SOURCE_SHAPE = "0.5B_AttnOut"
ADDED_SHAPE = "35BMoE_Expert_Down"
REGIME = "avx2-build.avx2-runtime"
THREADS = 28


class CPUNativeVNNIPrefillDevelopmentLineageTest(unittest.TestCase):
    """Prove migrations preserve source identity and remain strictly additive."""

    @staticmethod
    def _split(name: str, shapes: tuple[str, ...]):
        return SimpleNamespace(
            development_shapes=shapes,
            sealed_shapes=("sealed",),
            sealed_m_values=(64,),
            digest=lambda: f"sha256:{name}",
            shape_names=lambda *, sealed: (
                frozenset(("sealed",)) if sealed else frozenset(shapes)
            ),
        )

    @staticmethod
    def _routes():
        manifest = load_shape_manifest()

        def route_for(_codebook: int, shape_name: str, _regime: str):
            shape = manifest.by_name(shape_name)
            return SimpleNamespace(n=shape.n, k=shape.k)

        return SimpleNamespace(
            digest=lambda: "sha256:routes",
            route_for=route_for,
            thread_count=lambda: THREADS,
        )

    @staticmethod
    def _write_source(path: Path) -> None:
        shape = load_shape_manifest().by_name(SOURCE_SHAPE)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "source_format",
                "shape",
                "m",
                "n",
                "k",
                "build_isa",
                "runtime_isa_effective",
                "threads",
            ))
            writer.writeheader()
            # Candidate rows repeat a process-cell identity. The lineage must
            # collapse those rows without inventing duplicate launch cells.
            for _ in range(2):
                writer.writerow({
                    "source_format": "Q4_0",
                    "shape": SOURCE_SHAPE,
                    "m": 64,
                    "n": shape.n,
                    "k": shape.k,
                    "build_isa": "AVX2",
                    "runtime_isa_effective": "AVX2",
                    "threads": THREADS,
                })

    @staticmethod
    def _increment() -> CPUPrefillSourceTrainingRecord:
        shape = load_shape_manifest().by_name(ADDED_SHAPE)
        return CPUPrefillSourceTrainingRecord(
            source_format="Q4_0",
            shape_name=ADDED_SHAPE,
            n=shape.n,
            k=shape.k,
            isa_regime=REGIME,
            runtime_isa="avx2",
            m_values=(64, 256),
        )

    def test_round_trip_binds_source_bytes_and_exact_increment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate = root / "source.csv"
            timing = root / "source.timing.csv"
            output = root / "lineage.json"
            self._write_source(aggregate)
            timing.write_text("sample\n1\n", encoding="utf-8")
            source_split = self._split("source", (SOURCE_SHAPE,))
            target_split = self._split(
                "target",
                (SOURCE_SHAPE, ADDED_SHAPE),
            )
            refinement = SimpleNamespace(
                digest=lambda: "sha256:refinement",
                split_manifest_digest="sha256:source",
                records=(),
            )

            with mock.patch(
                "native_vnni_dispatch.cpu_prefill_development_lineage."
                "cpu_prefill_source_training_records",
                return_value=(self._increment(),),
            ):
                plan = build_cpu_prefill_development_lineage_plan(
                    aggregate,
                    timing,
                    source_split,
                    target_split,
                    self._routes(),
                    self._routes(),
                    (refinement,),
                )
            self.assertEqual(plan.added_production_shapes, (ADDED_SHAPE,))
            self.assertEqual(plan.thread_count, THREADS)
            self.assertEqual(len(plan.source_records), 1)
            self.assertEqual(plan.source_records[0].m_values, (64,))
            self.assertEqual(plan.increment_records, (self._increment(),))
            launch_output = StringIO()
            with redirect_stdout(launch_output):
                _print_launch_records(plan)
            added_shape = load_shape_manifest().by_name(ADDED_SHAPE)
            self.assertEqual(
                launch_output.getvalue().splitlines(),
                [
                    f"thread_count\t{THREADS}",
                    "\t".join((
                        "Q4_0",
                        ADDED_SHAPE,
                        str(added_shape.n),
                        str(added_shape.k),
                        REGIME,
                        "avx2",
                        "64,256",
                    )),
                ],
            )
            write_cpu_prefill_development_lineage_plan(output, plan)

            restored = read_cpu_prefill_development_lineage_plan(
                output,
                aggregate,
                timing,
                source_split,
                target_split,
                self._routes(),
                self._routes(),
                (refinement,),
            )
            self.assertEqual(restored.canonical_mapping(), plan.canonical_mapping())
            self.assertEqual(restored.digest(), plan.digest())

            historical = plan.canonical_mapping()
            historical["shape_manifest_digest"] = "sha256:" + "a" * 64
            output.write_text(
                json.dumps(historical),
                encoding="utf-8",
            )
            restored_historical = read_cpu_prefill_development_lineage_plan(
                output,
                aggregate,
                timing,
                source_split,
                target_split,
                self._routes(),
                self._routes(),
                (refinement,),
            )
            self.assertEqual(
                restored_historical.shape_manifest_digest,
                historical["shape_manifest_digest"],
            )

            timing.write_text("sample\n2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_timing_digest"):
                read_cpu_prefill_development_lineage_plan(
                    output,
                    aggregate,
                    timing,
                    source_split,
                    target_split,
                    self._routes(),
                    self._routes(),
                    (refinement,),
                )

    def test_source_threads_must_match_authenticated_routes(self) -> None:
        """An additive increment cannot silently change OpenMP geometry."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate = root / "source.csv"
            timing = root / "source.timing.csv"
            self._write_source(aggregate)
            timing.write_text("sample\n1\n", encoding="utf-8")
            with aggregate.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["threads"] = "36"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "thread count disagrees"):
                build_cpu_prefill_development_lineage_plan(
                    aggregate,
                    timing,
                    self._split("source", (SOURCE_SHAPE,)),
                    self._split("target", (SOURCE_SHAPE, ADDED_SHAPE)),
                    self._routes(),
                    self._routes(),
                    (),
                )

    def test_target_may_not_relabel_or_remove_source_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate = root / "source.csv"
            timing = root / "source.timing.csv"
            self._write_source(aggregate)
            timing.write_text("sample\n1\n", encoding="utf-8")
            source_split = self._split("source", (SOURCE_SHAPE,))
            non_additive = self._split("target", (ADDED_SHAPE,))

            with self.assertRaisesRegex(ValueError, "strictly extend"):
                build_cpu_prefill_development_lineage_plan(
                    aggregate,
                    timing,
                    source_split,
                    non_additive,
                    self._routes(),
                    self._routes(),
                    (),
                )


if __name__ == "__main__":
    unittest.main()
