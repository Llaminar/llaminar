#!/usr/bin/env python3
"""Require every original production-body counterexample to fail all20 replays.

The fixture executable compiles the authenticated original body against the
same native API doubles and positive assertions. No vendor runtime is linked.
An unrelated crash, skip, missing case or one accidental aggregate failure
cannot satisfy this control.
"""
import argparse
import subprocess

CASES = (
    "SavedCurrentFailureCannotInventOrdinalZero",
    "FailedDeviceHandleObservationCannotBecomeInactiveContext",
    "FailedPrimaryStateObservationCannotReleaseExistingContext",
    "FailedCapacitySelectionRestoresCallerAndNeverInventsFreeVRAM",
    "FailedMemoryObservationRestoresAndCannotPublishPoisonedOutput",
    "FailedExplicitRestorationPropagatesWithoutRetry",
    "FailedExplicitPrimaryReleaseIsCheckedAndEveryEligibleReleaseAttempted",
    "FailedPeerQueryCannotBecomeAnAbsentEdge",
)


def main() -> None:
    """Accept only all eight actual original failures, each observed20 times."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()
    names = tuple("ROCmEnumerationFailures." + case for case in CASES)
    result = subprocess.run(
        [args.binary, "--gtest_color=no", "--gtest_repeat=20", "--gtest_filter=" + ":".join(names)],
        text=True, capture_output=True, timeout=30, check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 1 or "[  SKIPPED ]" in output:
        raise AssertionError(f"Original control terminated incorrectly: {result.returncode}\n{output}")
    for name in names:
        runs = output.count("[ RUN      ] " + name + "\n")
        failures = output.count("[  FAILED  ] " + name + " (")
        if runs != 20 or failures != 20:
            raise AssertionError(f"Original {name}: runs={runs}, failures={failures}; expected20 each\n{output}")
    if "[  PASSED  ]" in output and "[  PASSED  ] 0 tests." not in output:
        raise AssertionError("Unexpected original-code pass\n" + output)
    print("Eight actual original production-body defects rejected in every20-case replay (160 expected failures).")


if __name__ == "__main__":
    main()
