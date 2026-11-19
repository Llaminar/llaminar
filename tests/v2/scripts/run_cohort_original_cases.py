#!/usr/bin/env python3
## @file run_cohort_original_cases.py
#  @brief Preserve the CTest process owner while isolating native replay lifetimes.
"""Run every selected original-source cohort case/repetition in its own canonical test process.

ProcessRuntime obligations deliberately survive validated context reset. Fresh case
sources therefore require fresh processes; this driver does not release tokens,
change death-test style, widen the enclosing canonical timeout, or hide failures.
"""
import fnmatch
import itertools
import os
import subprocess
import sys


def main():
    binary, *arguments = sys.argv[1:]
    selected = "*"
    repetitions = 1
    passthrough = []
    variants = {}
    for arg in arguments:
        if arg.startswith("--gtest_filter="):
            selected = arg.split("=", 1)[1]
        elif arg.startswith("--gtest_repeat="):
            repetitions = int(arg.split("=", 1)[1])
        elif arg.startswith("--original-case-env="):
            key, separator, values = arg.split("=", 1)[1].partition("=")
            if not separator or not key or key in variants or any(not value for value in values.split(",")):
                raise ValueError("Original cohort replay needs a unique environment name and nonempty values")
            variants[key] = values.split(",")
        else:
            passthrough.append(arg)
    if repetitions <= 0:
        raise ValueError("Original cohort replay requires a finite positive population")
    # The enclosing native guardian deliberately makes its authenticated loan
    # descriptors inheritable. Preserve that exec contract for discovery and
    # replay; Python's own files and pipes are close-on-exec by default.
    listed = subprocess.run([binary, "--gtest_list_tests"], capture_output=True, text=True,
                            check=True, close_fds=False)
    names, suite = [], ""
    for raw_line in listed.stdout.splitlines():
        # GTest prints two spaces before a suite type or case value comment.
        # Preserve indentation but remove the delimiter and its entire margin.
        line = raw_line.split(" #", 1)[0].rstrip()
        if line and not line[0].isspace() and line.endswith("."):
            suite = line
        elif line.startswith("  ") and suite:
            names.append(suite + line.strip())
    include, _, exclude = selected.partition("-")
    cases = [name for name in names if any(fnmatch.fnmatchcase(name, pattern) for pattern in include.split(":"))
             and not any(fnmatch.fnmatchcase(name, pattern) for pattern in exclude.split(":") if pattern)]
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("Original cohort selection has no exact unique compiled cases")
    for repeat in range(repetitions):
        for name in cases:
            for values in itertools.product(*variants.values()):
                overrides = dict(zip(variants, values))
                detail = f" {overrides}" if overrides else ""
                print(f"Original source process replay {repeat + 1}/{repetitions}: {name}{detail}", flush=True)
                subprocess.run([binary, *passthrough, "--gtest_death_test_style=threadsafe",
                                "--gtest_repeat=1", "--gtest_filter=" + name],
                               check=True, close_fds=False, env={**os.environ, **overrides})


if __name__ == "__main__":
    main()
