#!/usr/bin/env python3
"""Check actual native-program GTests against their disjoint bounded CTest gates.

Listing tests executes no bodies or native APIs. The generated CTest inventory,
not a second copied case table, supplies the real commands and properties.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import unittest

COVERAGE = "V2_Integration_PooledNativeProgramRegistrationCoverage"
RETIRED = "V2_Integration_PooledNativeProgramDiagnosticReplay"
REQUIRED_ENV = {
    "LLAMINAR_FORCE_CPU_ONLY_STARTUP": "1",
    "LLAMINAR_SKIP_CUDA_STARTUP": "1",
    "LLAMINAR_SKIP_ROCM_STARTUP": "1",
    "GTEST_DEATH_TEST_STYLE": "threadsafe",
}


def listed_cases(text):
    """Read every actual compiled case, rejecting disabled or duplicate names."""
    suite = None
    result = []
    for line in text.splitlines():
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_/]*)\.(?:\s*#.*)?", line)
        if match:
            suite = match[1]
        elif suite and line.startswith("  "):
            case = line.strip().split("#", 1)[0].strip()
            if case:
                result.append(suite + "." + case)
    if not result or len(set(result)) != len(result):
        raise ValueError("Compiled native-program cases are empty or duplicated")
    if any("DISABLED_" in case for case in result):
        raise ValueError("Compiled native-program case is disabled")
    return result


def verify(cases, inventory, binary):
    """Prove one explicit 20-repeat/30s threadsafe preflight per compiled test, no gaps."""
    errors = []
    observed = []
    binary = Path(binary).resolve()
    for test in inventory.get("tests", []):
        name = test["name"]
        if name == RETIRED:
            errors.append("Retired broad native-program replay is still registered")
        if name == COVERAGE:
            continue
        properties = {item["name"]: item["value"] for item in test.get("properties", [])}
        if "ProductionTestPreflight" not in properties.get("LABELS", []):
            continue
        command = test.get("command", [])
        if not any(Path(arg).resolve() == binary for arg in command):
            continue
        filters = [arg.split("=", 1)[1] for arg in command if arg.startswith("--gtest_filter=")]
        if len(filters) != 1 or not re.fullmatch(r"[A-Za-z0-9_./]+", filters[0]):
            errors.append(name + ": gate needs one exact case, without wildcard/union")
            continue
        observed.append(filters[0])
        expected_name = "V2_Integration_PooledNativeProgram_" + filters[0].split(".", 1)[-1] + "Replay"
        if name != expected_name:
            errors.append(name + ": noncanonical case registration name")
        if [arg for arg in command if arg.startswith("--gtest_repeat=")] != ["--gtest_repeat=20"]:
            errors.append(name + ": original 20 replays are not preserved")
        if properties.get("TIMEOUT") != 30:
            errors.append(name + ": canonical 30s timeout changed")
        environment = dict(item.split("=", 1) for item in properties.get("ENVIRONMENT", []) if "=" in item)
        for key, value in REQUIRED_ENV.items():
            if environment.get(key) != value:
                errors.append(name + ": missing exact " + key)
        styles = [arg for arg in command if arg.startswith("--gtest_death_test_style=")]
        if styles and styles != ["--gtest_death_test_style=threadsafe"]:
            errors.append(name + ": command overrides threadsafe style")
    duplicate = sorted(case for case, count in Counter(observed).items() if count != 1)
    missing = sorted(set(cases) - set(observed))
    extra = sorted(set(observed) - set(cases))
    if duplicate:
        errors.append("Duplicate preflight cases: " + ", ".join(duplicate))
    if missing:
        errors.append("Missing preflight cases: " + ", ".join(missing))
    if extra:
        errors.append("Foreign preflight cases: " + ", ".join(extra))
    return errors


class RegistrationCoverageTests(unittest.TestCase):
    """Recorded CTest metadata mutations exercise the same installed verifier."""
    def setUp(self):
        self.binary = "/recorded/native-program"
        self.cases = ["PooledNativeProgramDiagnostic.Alpha", "PooledNativeProgramDiagnostic.Beta"]
        self.inventory = {"tests": [self.entry(case) for case in self.cases]}

    def entry(self, case):
        return {"name": "V2_Integration_PooledNativeProgram_" + case.split(".", 1)[-1] + "Replay", "command": [self.binary, "--gtest_filter=" + case, "--gtest_repeat=20"],
                "properties": [{"name": "LABELS", "value": ["ProductionTestPreflight"]},
                               {"name": "TIMEOUT", "value": 30.0},
                               {"name": "ENVIRONMENT", "value": [key + "=" + value for key, value in REQUIRED_ENV.items()]}]}

    def errors(self):
        return verify(self.cases, self.inventory, self.binary)

    def test_exact_generated_partition(self):
        self.assertEqual(self.errors(), [])

    def test_missing_case(self):
        self.inventory["tests"].pop()
        self.assertTrue(any("Missing" in error for error in self.errors()))

    def test_duplicate_case(self):
        self.inventory["tests"].append(copy.deepcopy(self.inventory["tests"][0]))
        self.assertTrue(any("Duplicate" in error for error in self.errors()))

    def test_foreign_case(self):
        self.inventory["tests"].append(self.entry("PooledNativeProgramDiagnostic.Foreign"))
        self.assertTrue(any("Foreign" in error for error in self.errors()))

    def test_retired_broad_gate(self):
        self.inventory["tests"].append({"name": RETIRED})
        self.assertTrue(any("Retired broad" in error for error in self.errors()))

    def test_wildcard_and_union(self):
        for query in ["PooledNativeProgramDiagnostic.*", ":".join(self.cases)]:
            with self.subTest(query=query):
                self.inventory["tests"][0]["command"][1] = "--gtest_filter=" + query
                self.assertTrue(any("exact case" in error for error in self.errors()))

    def test_changed_replay_or_timeout(self):
        self.inventory["tests"][0]["command"][2] = "--gtest_repeat=1"
        self.inventory["tests"][0]["properties"][1]["value"] = 60
        self.assertTrue(any("replays" in error for error in self.errors()))
        self.assertTrue(any("timeout" in error for error in self.errors()))

    def test_fast_style_and_device_startup(self):
        self.inventory["tests"][0]["command"].append("--gtest_death_test_style=fast")
        self.inventory["tests"][0]["properties"][2]["value"] = []
        self.assertTrue(any("threadsafe style" in error for error in self.errors()))
        self.assertTrue(any("LLAMINAR_FORCE_CPU_ONLY_STARTUP" in error for error in self.errors()))

    def test_noncanonical_name_and_missing_preflight_label(self):
        self.inventory["tests"][0]["name"] = "UnrelatedGate"
        self.assertTrue(any("noncanonical" in error for error in self.errors()))
        self.inventory["tests"][0]["properties"][0]["value"] = ["Integration"]
        self.assertTrue(any("Missing" in error for error in self.errors()))

    def test_gtest_listing_rejects_missing_disabled_or_duplicate(self):
        for text in ["", "Suite.\n  DISABLED_Alpha\n", "Suite.\n  Alpha\n  Alpha\n"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                listed_cases(text)
        self.assertEqual(listed_cases("Suite.\n  Alpha\n  Beta\n"), ["Suite.Alpha", "Suite.Beta"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--ctest", type=Path)
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        return not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RegistrationCoverageTests)).wasSuccessful()
    if not args.binary or not args.ctest or not args.build_dir:
        parser.error("actual binary, CTest and generated build directory are required")
    env = dict(os.environ, **REQUIRED_ENV)
    for key in ("GTEST_TOTAL_SHARDS", "GTEST_SHARD_INDEX"):
        env.pop(key, None)
    output = subprocess.run([str(args.binary), "--gtest_list_tests", "--gtest_filter=*"], env=env,
                            check=True, capture_output=True, text=True, timeout=10).stdout
    cases = listed_cases(output)
    data = subprocess.run([str(args.ctest), "--show-only=json-v1", "--test-dir", str(args.build_dir)], env=env,
                          check=True, capture_output=True, text=True, timeout=10).stdout
    errors = verify(cases, json.loads(data), args.binary)
    if errors:
        raise RuntimeError("; ".join(errors))
    print(json.dumps({"schema": "llaminar.native-program-registration-coverage.v1", "cases": cases,
                      "count": len(cases), "replays_per_case": 20, "timeout_seconds": 30}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
