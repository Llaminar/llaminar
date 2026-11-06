# Six-GPU moving QFN MTP: focused checkpoint

The fixed-depth-1 case passes on two RTX 3090 continuation GPUs and four MI50
expert followers, with RAM ngrams and dynamic ordinal expert movement. This
checkpoint includes the original stall repairs and the peer-cancellation fix.
Full Unit and ProductionTestPreflight were deliberately not run for this
iteration, as requested by the user.

## Scope and result

The unchanged four-shard Qwen3.8-Flash-Next UD-Q4_K_XL model and its Q8_0 learned
predictor run with FP32 activations, FP16 KV, eight CUDA-resident experts and
the remaining 504 experts on the four ROCm followers. Retained MTP capacity is
three; this executed case selects fixed depth one. The continuation is CUDA TP2;
the expert domains use their declared apportioned ownership. RAM ngram backing
does not create a CPU expert participant.

The isolated checkpoint source is projected from `6c16488d1`, rather than from
the entire experimental worktree. Its normal owning builds include CPU, CUDA
SM80/86/89/90 and HIP gfx906/gfx1100, with the configured SDK, exact NCCL/RCCL
libraries and spill guards unchanged. The reviewed scope excludes unfinished
NativeGroup/no-reset candidates and recurrent-depth implementation work.

| Focused validation | Result |
|---|---|
| Owning host suites and explicit lifecycle regressions | 48 registrations pass in 12.17 s |
| Peer cancellation before successor admission | CUDA SM86, MI50 and gfx1100 pass; each capture replays 20 repetitions of abort-before-generation, valid successor and ready-abort |
| Follower predecessor ordering | CUDA and ROCm pass, including the deterministic missing-edge negative control, repeated capture and stale-receipt checks |
| Device-authored migration cycles | Actual CUDA/ROCm producer regression passes |
| Finite placement lifecycle | CUDA2, ROCm2 and ROCm4 captured regressions pass |
| Exact six-GPU model case | Pass, 315.40 s including shutdown; no skipped cases or new driver diagnostics |

The model result contains **1,208 unique stage/row comparisons**, all byte
identical to actual serial inference, across four prefix histories. One accepted
and three rejected draft checkpoints execute. Four complete physical expert
migration transactions pass the required movement-evidence assertions. All six
original device contexts retire normally and their canonical allocation counts
return to zero. The four MI50s return to their 10,866,688-byte baseline. Peak
sampled process RSS is 48.89 GiB.

This is `native_serial_grouped_exact_functional` qualification. It is not a new
PyTorch final-logit KL panel, frozen-history numerical qualification, Release
performance measurement, complete preflight receipt or image certificate.
Depth three and dynamic MTP recurrence remain unqualified.

## Repairs retained in the checkpoint

- Physical migration consumes the device-authored cycle identity instead of
  reconstructing a different decomposition that can exceed admitted lane demand.
  Canonical cycle geometry and capacity remain unchanged.
- The original publication ring retains every accepted transaction and its
  submission handle through aggregate retirement. Source failures remain sticky;
  a missing original failure terminal is fatal rather than entering an unwind
  that can wait forever. Completed submission generations form a monotonic prefix.
- Different follower capture streams wait on the exact predecessor event before
  reusing shared epoch/workspace state. Metadata publication and withdrawal are
  synchronized without a host completion wait under the metadata lock. An
  already-observed current event remains a valid device dependency; stale,
  relabelled and foreign receipts reject.
- Both native peer-epoch waits now observe the existing absorbing abort while
  waiting for a successor generation. The successful post-generation signal
  acquire is preserved. A rejected signal-first draft could accept an old
  positive signal as publication of a new descriptor; a controlled interleaving
  reproduces that defect. The accepted branch preserves the original final CUDA
  register counts and introduces no memory spills on the shipped targets.
- Source admission distinguishes host RAM ngram storage from routed CPU expert
  computation. Complete predictor graph identity, original epoch/source ownership
  and the typed finite placement boundary remain part of the same focused closure.
- The QFN test filters include the controller telemetry domain required by their
  existing movement assertions. Completed native, mapped and CPU movement evidence
  retains the same strict requirements; partial or stale evidence cannot pass.

Every production defect has an explicit focused ProductionTestPreflight entry.
The entries are installed for future canonical gates; this iteration ran only
the selected registrations above.

## Storage observation and cleanup

Shutdown persisted eight RAM cache entries, writing 1,424,404,544 payload bytes.
That write took 106.40 s while the process was in I/O wait and the host reported
substantial I/O pressure. It completed successfully; there was no missing GPU
terminal or driver failure. The earlier active-source run completed in 210.16 s.
These timings include diagnostic snapshots and different storage conditions;
they are not an inference throughput comparison or a resolved storage-performance
claim.

After native retirement, 1,424,419,115 bytes of this run's owned prefix-cache files
were removed. Logs, CSVs, resource observations and failure controls are retained.
No model, reference-cache payload or unrelated process state was removed.

## Remaining recurrent work

Repeated R1 follower captures need terminal receipts over the original admitted
slots and device handoff between packet invocations. That implementation remains
private until its native capture and fixed3/dynamic model tests pass. It must
observe the complete aggregate before clearing any packet metadata or original
protocol slot, preserve cancellation and use only actual live payload extents.

The source already has a complete aggregate frontier for this hosted policy:
each sidecar's capture event joins the scheduler stream; the complete branch
publishes an immutable scheduler ticket and ready event; RankOrchestrator observes
all continuation participants before the coordinator retires the sequence.
Additional per-draft source event populations or host fences are unnecessary.
The recurrent integration still needs actual next-request reset/replay proof.

Local evidence: `/tmp/qfn-fixed1-projection-build-r710/` owns the initial coherent
build, `/tmp/qfn-peer-abort-integration-r712/` owns the accepted cancellation
integration, and `/tmp/qfn-cpu-joint-qualified-r224/focused713-*` through
`focused721-*` own the focused results. The model directory's `artifact-audit.json`
and `cleanup.json` record comparison freshness, exactness, retirement and cleanup.
Private recurrence packages are intentionally outside this checkpoint.
