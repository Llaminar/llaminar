#!/usr/bin/env python3
"""Authenticate heavy Unit fixtures against CTest's actual CPU reservations.

The compiled quantized-kernel sweep and its reservation share one CMake width.
Numerical Python uses one worker per library inside its one-core reservation;
fresh native-library probes verify that the configured environment reaches the
actual NumPy/PyTorch runtimes. No models, accelerators or source scans are used.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


BUILD: Path
SWEEP_THREADS: int
LOGICAL_PER_CORE: int
POOL_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def fixture_properties(document, name):
    """Require exactly one registered owner, including its scheduler properties."""
    matches = [row for row in document["tests"] if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"missing or duplicate fixture: {name}")
    return {row["name"]: row["value"] for row in matches[0]["properties"]}


def validate_reservations(document):
    """Reject under-reserved explicit sweeps and unbounded numerical worker pools."""
    sweep = fixture_properties(document, "V2_Unit_CPUQuantizedGemmParity")
    if sweep.get("PROCESSORS", 1) < SWEEP_THREADS * LOGICAL_PER_CORE:
        raise ValueError("kernel sweep exceeds its physical CPU reservation")
    reference = fixture_properties(document, "V2_Unit_Python_Reference_All")
    if reference.get("PROCESSORS", 1) != LOGICAL_PER_CORE:
        raise ValueError("reference fixture must reserve one physical core")
    environment = reference.get("ENVIRONMENT", [])
    for name in (*POOL_VARIABLES, "LLAMINAR_FORCE_CPU_ONLY_STARTUP"):
        values = [item.partition("=")[2] for item in environment if item.startswith(name + "=")]
        if values != ["1"]:
            raise ValueError(f"reference pool ownership is ambiguous: {name}={values}")
    return dict(item.split("=", 1) for item in environment)


class UnitCpuBudgetTests(unittest.TestCase):
    """Exercise configured ownership and fresh native-runtime negative controls."""

    @classmethod
    def setUpClass(cls):
        """Read the configured CTest inventory without launching any registered test."""
        cls.document = json.loads(subprocess.check_output(
            ["ctest", "--test-dir", str(BUILD), "--show-only=json-v1"], text=True, timeout=10))

    def test_complete_cpu_reservations(self):
        """The real registrations must account for their maximum live worker demand."""
        self.assertTrue(validate_reservations(self.document))

    def test_missing_and_under_reserved_owners_are_rejected(self):
        """An absent fixture or the previous four-core reservation cannot certify the sweep."""
        for mutation in ("missing", "processors"):
            with self.subTest(mutation=mutation):
                document = copy.deepcopy(self.document)
                test = next(row for row in document["tests"] if row["name"] == "V2_Unit_CPUQuantizedGemmParity")
                if mutation == "missing":
                    document["tests"].remove(test)
                else:
                    test["properties"] = [row for row in test["properties"] if row["name"] != "PROCESSORS"]
                    test["properties"].append({"name": "PROCESSORS", "value": 4 * LOGICAL_PER_CORE})
                with self.assertRaises(ValueError):
                    validate_reservations(document)

    def test_missing_conflicting_and_oversized_library_pools_are_rejected(self):
        """Each library owns an explicit width; inherited defaults cannot enlarge it."""
        for variable in POOL_VARIABLES:
            for replacement in ([], [variable + "=2"], [variable + "=1", variable + "=2"]):
                with self.subTest(variable=variable, replacement=replacement):
                    document = copy.deepcopy(self.document)
                    test = next(row for row in document["tests"] if row["name"] == "V2_Unit_Python_Reference_All")
                    environment = next(row for row in test["properties"] if row["name"] == "ENVIRONMENT")
                    environment["value"] = [item for item in environment["value"]
                        if not item.startswith(variable + "=")] + replacement
                    with self.assertRaisesRegex(ValueError, "pool ownership"):
                        validate_reservations(document)

    def test_real_numerical_libraries_obey_the_registered_environment(self):
        """Fresh native runtimes obey the fixture environment and detect enlarged pools."""
        environment = {**os.environ, **validate_reservations(self.document)}
        probe = ("import json,numpy,torch,threadpoolctl\n"
                 "numpy.dot(numpy.ones((8,8)),numpy.ones((8,8)))\n"
                 "torch.mm(torch.ones((8,8)),torch.ones((8,8)))\n"
                 "print(json.dumps({'torch':torch.get_num_threads(),'pools':threadpoolctl.threadpool_info()}))\n")
        # PyTorch gives MKL_NUM_THREADS precedence over OMP_NUM_THREADS even
        # when its active pool is libgomp. Mutate the authority each native
        # runtime actually consumes; the inventory checks cover every setting.
        for mutation in (None, "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            with self.subTest(mutation=mutation):
                selected = {**environment, **({mutation: "2"} if mutation else {})}
                result = json.loads(subprocess.check_output([sys.executable, "-c", probe],
                    env=selected, text=True, timeout=20))
                self.assertTrue(result["pools"])
                widths = [result["torch"], *(row["num_threads"] for row in result["pools"])]
                self.assertEqual(all(width == 1 for width in widths), mutation is None, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--sweep-threads", type=int, required=True)
    parser.add_argument("--logical-per-core", type=int, required=True)
    args, remaining = parser.parse_known_args()
    BUILD = args.build_dir.resolve(strict=True)
    SWEEP_THREADS, LOGICAL_PER_CORE = args.sweep_threads, args.logical_per_core
    unittest.main(argv=[sys.argv[0], *remaining])
