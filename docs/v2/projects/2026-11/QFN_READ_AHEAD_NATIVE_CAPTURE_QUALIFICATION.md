# QFN read-ahead native capture qualification

This is a qualification record, not a tuning dashboard. It records a focused
component slice; it does not certify the full model, complete preflight, or a
production throughput policy.

## Captured read-ahead

The CUDA and ROCm side-stream and parent-terminal families now have supported
opaque resource allowances in `NativeAuxiliaryQualifiedLimits.h`. Selection is
by backend and typed resource program. Application, driver and hardware
fingerprints remain provenance, not eligibility keys or measured-size ceilings.

| Family, per original population | Host allowance | Device allowance | Lifetime |
| --- | ---: | ---: | --- |
| One side stream and two ordering events | 4 MiB | 16 MiB | Process runtime |
| Independent parent terminal event | 512 KiB | 2 MiB | Process runtime |

Graph executable storage and explicit payloads remain independently admitted.
The side allowance leaves substantial room above the observed queue granularity;
the parent uses the established one-event allowance units. Handle destruction
and context reset do not prove that opaque process pools released their backing,
so the original PMA retains those commitments. These are conservative support
commitments, not claims of exact or universally measured native allocation sizes.

The canonical raw diagnostic ran separately on an RTX 3090 (`sm_86`) and one
MI50 (`gfx906:sramecc+:xnack-`). Each backend passed 20 fresh processes, another
20 successful family lifetimes in one process, and three partial-creation
failure controls: 24 processes, 40 successful side-family lifetimes, 800 captured
replays and 43 parent-event lifetimes. Every process completed original context
retirement; both complete driver windows were clean.

| Observation, maximum checkpoint delta | CUDA | ROCm |
| --- | ---: | ---: |
| Parent creation/first-use heap | 1,888 B | 2,000 B |
| Side creation/first-use heap | 1,664 B | 30,800 B |
| Side creation/first-use visible device-memory decrease | 0 B | 2 MiB |
| Side plus graph/diagnostic bookkeeping heap | 261,888 B | 67,360 B |

Zero visible device-memory decrease does not mean zero physical backing.
Checkpoint heap/device observations are not an exact attribution or a live
accounting authority. The complete observations and source/runtime identities
are retained at `/tmp/qfn-read-ahead-resource-qualification-r447/`; the reviewed
`observations.json` evidence digest is
`220fa9fcb0cd4b0fc31d0b25c0215941eba3bafe15b55213ce4681336d438279`.
The existing diagnostic script is
`scripts/diagnostics/run_read_ahead_resource_calibration.py`; its target is
`v2_perf_read_ahead_resource_calibration`. Both raw runs used `--cold-processes
20 --warm-lifetimes 20 --replays 20 --creator-rejection-controls` inside the
canonical GPU driver-observation window. They took 197.2 s and 191.5 s.

The captured fixture now admits its worker through the original
`NativeContextPreparation`, combines that source's mandatory BOM with its
family, graph and payload declarations, and retains the same PMA through
retirement. Its executable uses the canonical GPU process entrypoint. A first
native run exposed the former plain-GoogleTest entrypoint, which lacked that
lexical process owner; the preserved failure is `focused367`.

All four explicit registrations passed in `focused369`, without skips or driver
findings, and now belong to `ProductionTestPreflight`:

- `V2_Integration_ReadAheadCapturedOverlap_CUDA`
- `V2_Integration_ReadAheadCapturedOverlap_ROCm`
- `V2_Integration_ReadAheadCapturedSnapshot_CUDA`
- `V2_Integration_ReadAheadCapturedSnapshot_ROCm`

The same retained production parent proves causal fork/join ordering, exact
production decoder bytes, untouched inactive tails, invalid/empty counts,
changing queries, repeated reset/replay and immutable snapshot publication.
Ordinary storage is released; exact original opaque commitments remain charged
through their declared process lifetime. No timing-event family was added.
Device-free eligibility and lifecycle regressions passed in the ten-test
`focused366` selection, including fingerprint-independent selection and the
original-source/underbudget rejection cases.

The tiny matched replay diagnostic measured CUDA serial/read-ahead at
11.96/8.68 microseconds and ROCm at 30.59/35.35 microseconds. These are one-row,
width-32 component observations, not full-model performance measurements or a
reason to promote a scheduling default.

Eight owning translation-unit syntax checks also passed against the isolated
feature checkpoint; the affected ownership implementations match the tested
Integration composition. Receipts are retained in
`/tmp/qfn-read-ahead-checkpoint-syntax-r450/`.

This slice does not renew the old complete prerequisite receipt. It does not
qualify full-model scheduling, Dynamic expert publication or incoming-fragment
use; those remain separate model and performance obligations.
