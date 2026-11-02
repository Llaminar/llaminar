#!/usr/bin/env python3
"""Run every selected original-source cohort case/repetition in its own canonical test process.

ProcessRuntime obligations deliberately survive validated context reset. Fresh case
sources therefore require fresh processes; this driver does not release tokens,
change death-test style, widen the enclosing canonical timeout, or hide failures.
"""
import fnmatch
import subprocess
import sys


def main():
    binary, *arguments = sys.argv[1:]
    selected = "*"
    repetitions = 1
    passthrough = []
    for arg in arguments:
        if arg.startswith("--gtest_filter="):
            selected = arg.split("=", 1)[1]
        elif arg.startswith("--gtest_repeat="):
            repetitions = int(arg.split("=", 1)[1])
        else:
            passthrough.append(arg)
    if repetitions <= 0:
        raise ValueError("Original cohort replay requires a finite positive population")
    listed = subprocess.run([binary, "--gtest_list_tests"], capture_output=True, text=True, check=True)
    names, suite = [], ""
    for line in listed.stdout.splitlines():
        if line and not line[0].isspace() and line.endswith("."):
            suite = line
        elif line.startswith("  ") and suite:
            names.append(suite + line.strip().split(" #", 1)[0])
    include, _, exclude = selected.partition("-")
    cases = [name for name in names if any(fnmatch.fnmatchcase(name, pattern) for pattern in include.split(":"))
             and not any(fnmatch.fnmatchcase(name, pattern) for pattern in exclude.split(":") if pattern)]
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("Original cohort selection has no exact unique compiled cases")
    for repeat in range(repetitions):
        for name in cases:
            print(f"Original source process replay {repeat + 1}/{repetitions}: {name}", flush=True)
            subprocess.run([binary, *passthrough, "--gtest_death_test_style=threadsafe", "--gtest_repeat=1", "--gtest_filter=" + name], check=True)


if __name__ == "__main__":
    main()
