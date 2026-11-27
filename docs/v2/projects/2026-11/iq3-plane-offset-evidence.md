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

## Follow-on: bound SMMLA decode live ranges

The three- and four-row IQ3_S SMMLA tiles now walk group pairs with a two-pair
unroll. The original three-row walk used a one-pair unroll; four rows fully
expanded the paired-index decoder. This retains shared decoded weights across
all rows and the exact INT32 dot and FP partition order, while shortening the
decode live ranges that compete with outer FP accumulators.

Final evidence: remote `runs/iq3-pair-final`, GNU-14, full canonical NEON and
DOTPROD/I8MM Release and Integration configure/build/verify commands:

- **IQ3_S admission fixed:** `4:2:1:3:11:0` FP spills 4 -> 2, meeting the
  unchanged budget of 2. Admission still failing: only I8MM Q5_K `4:2:1:3:7:1` (2 > 0 spills);
  NEON and DOTPROD have no admission failures.
- I8MM stack failures 24 -> 23. IQ3_S `3:2:1:3:11:0` excess 1351 -> 0;
  `4:2:1:3:11:0` excess 2040 -> 1350. NEON 5 and DOTPROD 20 unchanged.
  No key worsened; both Release/Integration failing sets and per-key failure
  records agree exactly.
- Fused witnesses pass in all envelopes. The committed regeneration tool
  changes only I8MM backend-native `4:2:0:3:11:0` FMLA site count 5 -> 8;
  ordered-add fingerprints remain unchanged. The backend-native compiler
  duplicates sites with the bounded group walk; byte contracts still pass.
- Native byte suites again pass 21/21/23, plus extreme-byte 1, compact 14,
  Q6/IQ4 5 and Q5 2. Full evidence suite with `--init` passes 224/224
  (29.795 s); attribute policy remains zero violations.
- No mapping, baseline, budget, grammar or production-test changes.

Rejected probes: an empty barrier on all four pair accumulators did not alter
admission; a one-pair loop for four rows met admission but raised total scalar
stack excess 2040 -> 2900, so it was not accepted. Two-pair unrolling improves
both admission and total weighted stack traffic. No end-to-end latency or
hardware-counter performance claim is made by this compilation/byte gate.

Local commands: `/tmp/pr73-iq3/iter2.sh iq3-c` and `iq3-d` (diagnostic only),
`/tmp/pr73-iq3/commitprep2.sh iq3-pair-walk`, followed by a comment-only source
clarification, role rebind and `/tmp/pr73-iq3/validate.sh`. Final test log is
`evidence-pair-final-init.log` in the remote scratch root.
