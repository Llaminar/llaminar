# gfx1100 half-operand prefill investigation

Status: deferred after reviewing its value to the goal. rocBLAS remains the
production implementation. Three custom WMMA variants passed component
correctness but lost every measured complete-operation cell. Their source,
binaries, tests and evidence are archived locally; the uninstalled candidate
and its temporary test registrations were removed from the feature tree.
The user explicitly directed the campaign to stay focused and avoid competing
with rocBLAS unless there is a compelling reason.

## Decision and scope

The sampled rocBLAS kernel uses vector mixed-FP16 FMA rather than WMMA. That ISA
observation alone does not establish a performance problem. Replacing the
library is not required to close the four primary prefill opportunities or the
added quantized MTP work. Reopen only with production attribution showing a
material bottleneck and a credible reason the library cannot address it.

The attempted boundary was the ordinary, explicitly declared `FP16RNE` path in
`ROCmFloatingPointGemmKernel::runFP16OperandProjection`: existing half-A bank,
prepared half weights, FP32 accumulation/output, one shared cast for fused
consumers. Fixed serial/verifier, strict FP32 and native FP32-activation paths
were excluded. No Automatic matrix keys were installed.

## Measured result

Each variant used six cells: M64/128/512, N8192/K2048, standalone or two independent
consumers sharing one cast. The second fused consumer owned N8209. Each cell
used31 alternating AB/BA pairs with eight complete operations per graph.
Internal captured events excluded host submission; each sampled launch started
with poisoned banks and its outputs were observed before any subsequent launch.
Library/matrix ratios below1 mean the library was faster.

| Candidate | Logical VGPRs | LDS bytes | Library/matrix range | M512 single/fused |
|---|---:|---:|---:|---:|
|Initial wave64 M32/N64|130|6144|0.521–0.792×|0.779× /0.792×|
|Padded wave64, shorter operand lifetime|79|6528|0.627–0.753×|0.718× /0.753×|
|Explicit wave32 M32/N64|64|6528|0.546–0.750×|0.664× /0.709×|

Every build compiled the shipping gfx906/gfx1100 target set with zero private
storage or VGPR memory spills. gfx906 emitted no matrix entry. Initial profiling
found45.71% LDS conflicts and25.79% occupancy. Padding reduced conflicts to13.64%
and raised occupancy to42.25%; wave32 reached77.61% occupancy. None of those
resource improvements produced a complete-operation win. A larger M64/N128 tile
was built and archived, but was not runtime-tested or timed before deferral.

Five component replay cases passed: exact row/column/K tails, shared casts,
subnormals, FP32 values above half range, epilogues, cancellation-sensitive finite
operands, changed/reset inputs, production geometry, and imported timing-parent
identity. Independent half-RNE/double references preserved existing exact
witnesses; finite inputs used the FP32 forward-error bound. Three focused
Integration registrations passed for the initial and wave32 versions, including
the existing physical-half and vendor-module inventory controls. These are
component observations, not production/model certification.

## Retained evidence

All paths are relative to the ignored
`/tmp/llaminar-gfx1100-wmma-prefill-evidence` root.

| Evidence | Receipt/archive |
|---|---|
|Initial replay|`half-prefill-replay-1794747971519819543/receipt.json`|
|Initial timing|`half-prefill-timing-1794747992316805938/receipt.json`|
|Padded replay|`half-prefill-replay-1794748402693922996/receipt.json`|
|Padded timing|`half-prefill-timing-1794748410779875482/receipt.json`|
|Wave32 replay|`half-prefill-replay-1794748953889454905/receipt.json`|
|Wave32 timing|`half-prefill-timing-1794748962430746467/receipt.json`|
|Integration controls|`half-prefill-focused-integration.log`, `half-prefill-wave32-focused-integration.log`|
|Initial archive,96 files|`half-prefill-initial-snapshot/snapshot.json`; SHA256 `48ece5d0c4116e8ace30c7995c6689b938f511ea38b8e48b4b27ff6120f02a83`|
|Padded archive,38 files|`half-prefill-padded-snapshot/snapshot.json`; SHA256 `adc1f1e2ccd660f52a230fb003408c811c7714391dda274eebdc3ff82714f1e5`|
|Wave32 archive,38 files|`half-prefill-wave32-snapshot/snapshot.json`; SHA256 `d7b6d633d935720da44ad53dcc3a962a7c7a3584ead7064c1bc067c9589c1786`|
|Unmeasured larger tile,12 files|`half-prefill-wide-unmeasured-snapshot/snapshot.json`; SHA256 `0bba079dc05f73e7e2b50f991fdc653f7767bf8a58b22d2ce4ff0197369fefaa`|

An earlier one-case exact-tail run passed natively but its collector failed to
serialize the receipt because the hash helper expected a `Path`. Its original
log/reader remains preserved; the authenticated replay above supersedes it.
