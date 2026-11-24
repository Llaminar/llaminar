#!/usr/bin/env python3
## @file run_cuda_moe_overlay_prefix_arrival_bom.py
#  @brief Compare the real ExpertOverlay prefix lifecycle against a cold control.
"""Give each policy its own process-runtime PMA root, and compare their receipts."""
import os
import subprocess
import sys


def run_policy(binary, mode):
    """Return the admitted graph bytes and both requests' exact completion IDs."""
    command = [binary, "--pipeline-native-family=cuda", "--pipeline-native-degree=1",
               "--gtest_filter=MoEOverlayArrivalOwnerLifecycle.PrefixRestoreKeepsEpochOwnedArrivalsOutOfNativeBOM"]
    # Keep the enclosing native process guardian's loan descriptors on exec.
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            close_fds=False, env={**os.environ, "LLAMINAR_TEST_MOE_ARRIVAL_PREFIX_CACHE": mode})
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    prefix = "CUDA ExpertOverlay prefix arrival BOM receipt: "
    records = [line.removeprefix(prefix) for line in result.stdout.splitlines() if line.startswith(prefix)]
    if len(records) != 1:
        raise RuntimeError(f"Prefix cache {mode}: expected one complete native receipt, got {len(records)}")
    receipt = tuple(map(int, records[0].split()))
    if len(receipt) != 7 or receipt[0] <= 0 or any(token < 0 for token in receipt[1:]):
        raise RuntimeError(f"Prefix cache {mode}: invalid native receipt {receipt}")
    return receipt


def main():
    """Reject any allocation or token difference between matched native runs."""
    binary, = sys.argv[1:]
    off = run_policy(binary, "off")
    on = run_policy(binary, "on")
    if on != off:
        raise RuntimeError(f"Prefix restore changed native graph bytes or completion IDs: off={off}, on={on}")
    print(f"Prefix restore retained {on[0]} NativeGraphExecutable bytes and {len(on) - 1} exact completion IDs")


if __name__ == "__main__":
    main()
