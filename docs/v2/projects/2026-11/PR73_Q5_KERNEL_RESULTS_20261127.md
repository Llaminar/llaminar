# PR73 Q5 kernel results — 2026-11-27

## Final disposition

The owner stopped further Q5 timing work and deferred timing-only ancillary
code-generation drift, conditional on the existing byte-identity evidence.
**Timing is unmeasured; no performance qualification or neutrality is claimed.**

The Q5 implementation was restored to the original `e4add4e20` scalar-row
contribution program. Checkpoint `2285fac2d` has the exact original
`CPUNativeNEONPreparedProjection.h` SHA-256
`b775b6eba2e41631c5e1754d7e62359d01a60198635c4f57c9ce834861fba008`.
The original integration15 evidence records native Release/Integration admission
failures **0/0/0**, passing fused witnesses and byte suites, and unresolved stack
failures **5/20/23**. Those are the retained original-program results, not a new
qualification of subsequent, independently owned IQ3 changes.

No compact K-loop call was removed by this work. No original stack failure was
removed by a retained Q5 change. The original compact12 reconstruction call and
two compact17 ancillary body changes remain timing-only open observations,
now deferred by the owner rather than certified economical.

## Bounded experiments, all rejected

Each implementation was committed and parent-pushed before native verification.

| Experiment | Actual result | Disposition |
|---|---|---|
| Per-row duplicated-lane vector widening (`ab701c4a0`) | Q5 key `4:2:1:3:7:1` acquired two spills against budget zero; all three compact helpers and string COMDAT still differed | Reverted |
| Q5-local, nonshared vector-conversion helper (`c1668921d`) | Isolated Q5-only admission 0/0/0 and fused witnesses passed, but the same five bodies still differed: Q5, three compact helpers, string COMDAT | Removed |
| Local scalar half conversion (`a7c867cc3`) | Four actual `fcvt s,h` instructions emitted; existing witness grammar rejected them, so I8MM admission was unavailable. The same five bodies still differed | Removed; no checker expansion |

The compact comparison used retained pre-Q5
`integration15-controls/4b551-noQ5-NEON_I8MM.o`; the earlier integration15
shared-header control established its emitted-body equivalence to the 58d77
comparison source. In the local-vector experiment compact12 still called
`coded_metadata::reconstruct<12,I8MM>` at offset `0x674`.

Frozen numerical results were independent of these economy failures:
- Failed per-row vector candidate: DotProd 22/22 and I8MM 24/24 native CTests passed.
- Local-vector helper: I8MM prepared-projection Unit and preflight 2/2 passed.
- Scalar helper: I8MM prepared-projection Unit and preflight 2/2 passed, including
  all 65,536 half encodings under an explicitly admitted production FPCR profile.

The retained exhaustive test now checks the existing native half conversion
against portable bits and explicitly asserts the production FPCR admission.
It has no dependency on a removed experimental helper. Its existing Unit and
ProductionTestPreflight registrations were preserved.

## Measurement stopped without a performance claim

The existing core-free panel tournament was minimally extended to admit compact
formats only at Strips=1; Strips=2/4 remain rejected. Its parser/publication
self-test passed with zero failed controls.

A CPU-only GNU14 NEON_I8MM Release build of the existing `llaminar2_core` and
panel target completed. Before any paired timing, the existing instruction
normalizer found a body mismatch for compact17 native contiguous SDOT:
588 normalized operations in the tournament versus 536 in the actual
`MoEExpertComputeStage.cpp.o` production copy. Consequently no core-free timing
was used to certify production. No 160-pair experiment or performance sampling
was performed. No ThreadScaling source change was made.

Available GGUF models on the Spark were DeepSeek-V4 IQ1_M/IQ3_XXS and a BF16
vision projector, not a supported matching Q5_K model pair. No model was
converted or downloaded. Ordinary perf access was denied at paranoid=4;
`sudo -n perf stat` worked without changing host settings, but no kernel timing
claim was derived from that availability check.

Source inspection also distinguished installed-engine reachability from the
supported explicit low-level ISA API: the I8MM engine AUTO path selects I8MM
before the SDOT branch, whereas explicit `ISAPath::NEON_DOTPROD` remains admitted
at the low-level boundary. No new production ISA selector was introduced and
no supported explicit kernel path was retired or relabeled unsupported.

## Evidence ownership and residual boundaries

Worker-owned host root: `factory@192.168.178.80:~/pr73-arm-stack-repair`.
The separate `~/pr73-arm-stack-q5` root belongs to the other kernel lane;
it was read for retained comparison evidence, never modified by this worker.
Container `/repair` aliases alone do not identify the host root; actual Docker
mounts were checked when apparent concurrent compiler paths were observed.

Retained repair-root evidence:
- `runs/row-widen*`, including failed objects;
- `runs/q5-row-local*`, including combined-candidate objects;
- `runs/q5-only-local*`, isolated Q5-local-helper objects;
- `runs/q5-scalar-half*`, scalar witness failure and exhaustive-byte results;
- `runs/perf-restored-B-release-build.log` and
  `runs/perf-restored-B-panel-contract.log`;
- `numtree/build/`, the completed Release production archive and panel binary.

On stop, no worker container remained active. Combined scratch use was
20,892,163,459 bytes, below 20 GiB. No third source copy was created and no local
result payload was committed.

The scratch's existing stack gate rejected comparisons because its running
checker identity `22af142b535b` did not match table identity `38199a641092`.
Neither checker, table, budget nor key map was changed to bypass that rejection.
This is distinct from actual admission failures and does not establish a new
per-key stack no-regression result. Release/Integration parity and remaining
stack reduction for independently changing kernels remain with their owning
qualification lanes.
