# PLE pipeline admission and failure retirement

2026-10-31. Focused repairs and complete model-free production preflight pass.

The CPU/NUMA/mixed-placement source batch r29 passed 40 of 43 cases. All three
failures used the original `ROCm0→ROCm1→CPU0` pipeline with layer ownership
`[0,1)`, `[1,2)`, `[2,48)`, resident VRAM ngrams on the PLE participant, and
fixed1, fixed3 or Dynamic MTP. Active tests retain capacity15 and original
main/predictor weights. No model, context, layer count or table policy was
reduced to make these cases fit.

## Graph entry defect

`Qwen4ExpAuxiliaryState::appendVerifierPLEPrior` required a nonempty list of
earlier local producers. PLE is the first model operation on the middle
participant, so its real snapshot legitimately begins an empty graph.
Construction failed before the native pipeline composer could attach ingress.

The snapshot now accepts empty predecessors only when the fragment contains
no nodes. An omitted dependency in a nonempty fragment still rejects. Named
producers must exist and belong to the auxiliary state's actual participant.
The snapshot remains the real root, and its move-only prior capability orders
hash/convolution consumers and candidate retention. No dummy stage is added.

After model construction, the existing `PipelineForwardGraphEdges::append`
attaches metadata, verifier checkpoint and activation receive before every
model root. Egress still follows every original leaf. SnapshotPrior reads
accepted history and live-row metadata; it does not read candidate
normalization before that producer runs. This fix changes graph construction
only: no allocation, transfer, synchronization or kernel arithmetic is added.

## Failure receipt defect

The specialized ngram admission fixture called `ASSERT_TRUE(initialize())`
inside its test body. On failure, destruction retired owners, but the assertion
returned before explicit cleanup evidence could be written.

It now uses the existing `observeQwen4ExpAdmissionThenRetire` sequencer. All
observer aliases unwind before the original final-owner retirement. Fatal
assertions remain GTest failures, exceptions retain their identity, and cleanup
receipts are written only after successful retirement. The existing retirement
regression file was also missing from CMake; it is now compiled into Unit and
explicitly registered in production preflight.

## Evidence

- Original core: both new PLE regressions fail for the intended causes—root
  rejection and acceptance of foreign producer identities.
- Original-core real-model control: initialization still fails, but the repaired
  fixture emits valid retirement evidence, removes its prefix directory and
  leaves its actual MPI session base empty. Driver diagnostics are clean.
- Fixed core: all three original PLE/MTP cases pass in r31, with unchanged
  runtime/source metadata, final retirement, empty sessions and clean drivers.
- Eight focused CTest suites pass, including actual CPU snapshot/retention
  replays over live16/3/0/1/12, reset, inactive-row poison, exact copy-byte
  counters, invalid predecessors and the existing native pipeline root/leaf
  topology checks for both GPU backend identities and rows1–16.
- Complete canonical prerequisites: **3,267 passed, zero failed, 29 unchanged
  hardware-unqualified**, across 3,296 registered tests in 88.3 minutes. Unit
  is 855/855, host 1,333/1,333, CUDA 273/273, ROCm 378/380 and mixed/exclusive
  428/455. No new unavailable-hardware cases appear. All 13,596 fenced source
  files and the selected runtime identities are unchanged; the full driver
  window is clean and all 2,989 actual owned session bases are empty. The
  expanded real-model admission matrix and numerical inference remain separate.

Explicit new preflight entries are `Qwen4ExpPLEPipelineIngress`,
`NativePipelineMTPModelRoots`, and `Qwen4ExpSourceAdmissionRetirement`, each
under the `V2_Integration_` prefix. Existing broad lifecycle coverage remains.

Local evidence:

- `/tmp/qfn-ple-pipeline-ingress-r1/`: original/fixed replay XML, build logs,
  complete Unit XML, canonical full-gate transaction, `full-gate-audit.json`
  and `full-gate-session-retirement.json`.
- `/tmp/qfn-ple-pipeline-admission-review-r1/REVIEW.md`: independent ordering
  and retirement review; no scoped blocker found.
- `parity-results/qwen-flash-next/source-admission-cpu-numa-mixed-placement-r29/`
- `parity-results/qwen-flash-next/source-admission-ple-retirement-original-control-r30/`
- `parity-results/qwen-flash-next/source-admission-ple-pipeline-mtp-recheck-r31/`

## Separate replicated-decode capacity boundary

The ROCm TP4 diagnosis found no remaining concrete duplicate physical charge.
All 49 routed expert banks require 24,720,179,200 bytes per device: the passing
48-bank Off configuration uses 24,051,712,000, and the shared predictor adds
668,467,200 bytes. Whole-expert ownership and the gate/up-owned, down-column
policy account for the same expert payload through different physical slices.

The complete recorded M64 active/full15 BOM remains larger than each MI50:

| Replicated-decode expert policy | Required bytes/device | Observed available bytes/device | Deficit |
|---|---:|---:|---:|
| GateUpOwnedDownColumns | 36,387,235,528 | 33,992,736,768 | 2,394,498,760 |
| Apportioned | 36,391,635,944 | 33,992,736,768 | 2,398,899,176 |

The earlier prepared-root duplicate-charge repair is already included. Staging
and captured inventory are real owners; subtracting them again would admit
unbacked storage. Even the entire physical 32GiB is below these requirements.
The final small expert allocation named by the rejection is not the total
capacity deficit.

Original positive cases therefore retain their full request on an authored
48GiB physical-card class. Separate 32GiB-cap cases retain the same requests
and require their exact routed-expert capacity rejection. See
[source provisioning](../../../testing/QFN_SOURCE_PROVISIONING.md).
All ten CUDA/ROCm typed capacity controls pass in
`source-admission-typed-capacity-controls-r32/`, including cleanup and driver
checks. These expected rejections do not qualify the larger-card positives.
Detailed arithmetic and independent review remain under
`/tmp/qfn-rocm-tp4-replicated-mtp-capacity-r1/` and
`/tmp/qfn-rocm-provisioning-review-r1/`.
