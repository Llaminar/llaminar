# IQ3_S block-local plane offsets: validation

The IQ3_S block decoder re-derives its column-derived plane addresses per K
block for ASIMD and non-four-row tiles. The empty register constraint is tied
to the block to prevent loop-invariant hoisting; it changes no arithmetic.
Four-row SDOT/SMMLA remain unchanged because the measured constraint did not
improve them. Companion count files were regenerated with
`CPUNativeTileCountRegeneration.py`; the source-bound roles entry is mechanical.
No key mapping, stack baseline, register budget or checker grammar changed.

## Native GNU-14 results

Spark `.80`, `~/pr73-arm-stack-repair/runs/iq3-checked`, compared with `v12`:

- Stack failures: NEON 6 -> 5, DOTPROD 22 -> 20, I8MM 26 -> 24.
- No key worsened and no new failing key appeared. Release/Integration agree.
- Fused stage passes in all envelopes. I8MM admission still failing:
  IQ3_S `4:2:1:3:11:0` spills 4 > 2; Q5_K `4:2:1:3:7:1` spills 2 > 0.
- NativeNEON byte tests pass 21/21/23; extreme-byte 1, compact units 14,
  Q6/IQ4 density 5 and Q5 single-scale 2 all pass.
- Codegen attribute policy: 2703 files, 22 attributes, zero violations.

## Evidence-suite worker-retirement diagnosis

Both unchanged HEAD `9e4dfb181` (archived into `evidence-baseline`) and candidate
were run with the identical command in the GNU-14 container:

```sh
cd /src && python3 tests/v2/unit/cmake/test_cpu_native_tile_register_evidence.py
```

Without Docker `--init`, both fail exactly one of 224 tests:
`ParallelAnalysisTest.test_no_worker_survives_a_killed_parent` (41.730 s baseline,
41.435 s candidate). Bash execs its final command, making Python PID 1. After
the test kills the mapping parent, all three workers are dead zombies adopted
by PID 1. A separate process-table probe captured PPID 1 / STAT Z for all three.
Python PID 1 does not reap these orphan descendants; `kill(pid, 0)` therefore
still succeeds. This is not a running worker or a kernel regression.

Adding Docker's standard `--init` orphan reaper, keeping the command, test and
all timeouts unchanged, passes both full suites: 224/224 baseline (30.828 s)
and 224/224 candidate (31.183 s). Logs are `evidence-baseline.log`,
`evidence-candidate.log`, `evidence-evidence-baseline-init.log` and
`evidence-src-init.log` in the remote scratch root. No source/test workaround
was installed. The scratch container launcher must use a real init when
executing this orphan-retirement test with Python as its final command.

Candidate kernel SHA256:
`0cc8e68b9876e2e32df996fa7f30fdc717c01dc4644887b52f0eb42da76e6df8`.
All three local companion JSON hashes match the validated remote source.
