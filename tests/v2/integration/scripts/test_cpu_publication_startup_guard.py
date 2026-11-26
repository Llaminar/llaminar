#!/usr/bin/env python3
"""Prove MPI startup cannot masquerade as a publication rejection.

The canonical CTest session owner encloses this driver and every binary it
starts. Parent and threadsafe death-child injection must fail with the MPI
tripwire diagnostic, never pass the intended publication-admission assertions.
"""
import argparse
import os
import subprocess
import sys


def main():
    """Exercise both startup positions through the actual publication fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()
    command = [args.binary,
               "--gtest_filter=CPUNativeQuantisedPublicationAdmission.EveryEntryRejectsUnsealedAndStaleBytes"]
    for mode, expected in (("parent", 86), ("death-child", 1)):
        env = dict(os.environ, LLAMINAR_TEST_PUBLICATION_FORCE_MPI=mode)
        run = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
        print(f"startup={mode} exit={run.returncode}\n{run.stdout}", flush=True)
        if run.returncode != expected or "unexpected MPI initialization" not in run.stdout:
            return 1
        if mode == "death-child":
            # A guard death is not the expected 'never sealed' or stale-seal
            # failure. GTest must mark the actual admission fixture as failed.
            if "[  FAILED  ]" not in run.stdout or "never sealed" not in run.stdout:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
