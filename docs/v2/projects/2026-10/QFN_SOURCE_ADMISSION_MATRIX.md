# QFN source-admission matrix

Updated 2026-10-31. **Size limit: 15,000 UTF-8 bytes.** This is a bounded status
and coverage dashboard; detailed logs remain in linked local evidence.

## Contract

These tests initialize the real production runner from the original four-shard
Qwen3.8 Flash Next UD-Q4_K_XL source, with all 48 layers and 512 experts per
bank. They exercise public configuration, prepared weights, actual TP/PP
ownership, ngram owners and PhysicalMemoryAuthority, then normal final-owner
retirement and deletion of fixture-owned prefix files. They do not load or
generate a PyTorch reference, enable snapshots, or execute a numerical request.
Source identity uses GGUF/file metadata, never a payload hash.

Admission success does not certify inference, arithmetic, movement, capture
replay or throughput. Those remain independent obligations of the existing
numerical and generation cases. In particular, a Dynamic declaration here
proves setup only; it cannot prove an expert moved during serving.

## Canonical inventory

Typed vectors and generated CTest own exact identities; the counts below
summarize them.

| Family | Positive cases | Scope |
|---|---:|---|
| Main | 20 | CPU; CUDA/CPU and ROCm/CPU ExpertOverlay; mixed-vendor PP; RAM/mmap |
| Host-adaptive | 20 | Same existing compositions; zero and positive RAM cache over SSD |
| Fixed-head table shards | 64 | CUDA/ROCm TP2/4/8 plus CPU experts, reversed TP2; RAM controls and VRAM shards; Static/Dynamic, ordinal/random |
| GPU-adaptive | 4 | CUDA/ROCm GPU1 plus CPU experts; zero and positive VRAM/RAM caches over SSD |
| All-GPU ROCm TP4 | 1 | Internally sharded experts; RAM ngrams |
| ROCm TP4 policy controls | 3 | Existing dense and routed compute policy alternatives |
| New CPU NUMA | 8 | TP2/PP2; RAM, mmap, host-adaptive zero and positive cache |
| **Supported-intent positive** | **120** | Declaration is not native qualification |
| Composed candidates | 20 | Four original GPU compositions and additive CUDA2→CPU2 15/33 layers; four host-table policies |
| **Total positive obligations** | **140** | No reference-directory dependency |

New compositions: CPU NUMA0+1 TP2; CPU0→CPU1 PP2; ROCm4 as
PP(TP2, TP2); CUDA2→CPU2 as PP(TP2, TP2); ROCm2→CPU2 in that same composition;
and CUDA2→ROCm2→CPU2 as PP(TP2, TP2, TP2). PP layer boundaries are explicit
and checked against the final execution plan. All are rank-local declarations.
The four GPU compositions have separate `CandidateSourceAdmission` tests,
labelled `UnqualifiedCandidate`. They still require successful native admission;
they are neither disabled nor inverted into expected failures. Declared Dynamic
residency remains intact, including on CPU TP stages.

**Twelve separate negative cases** retain requests for
`GateUpOwnedDownColumns` across a GPU expert tier and a CPU expert tier,
crossing CUDA/ROCm, TP2/4/8 and RAM/fixed-head VRAM ngrams. Production currently
supports this expert slicing policy within one native GPU tier; cross-tier
slice publication is unimplemented. These tests require that exact rejection,
with no changed expert policy, device count or storage tier. Table-head
sharding and internal expert sharding are distinct axes.

The 155 original seeds expand over **Off, fixed1, fixed3, and Dynamic**. Two
additional four-policy CUDA TP2 families plus six ROCm replicated-decode capacity controls bring the total to **634**:
496 supported-intent positives, 80 composed candidates and 58 rejection controls.
The added families cover four-expert VRAM with retained MTP capacity3 and
eight-expert admission under an explicit 13GiB public GPU cap. Original names,
sources and full15 identities remain unchanged. Discovery authenticates all 634 identities. Active cells retain depth15
except the explicit depth3 CUDA family. Both use the original R1 predictor;
Dynamic keeps automatic initial depth, default controller cadence and the
production speculative verifier.

Active setup checks original main/predictor sources, real prepared source48
weights, PMA owners, and exact serving participant/role/row/outcome families.
`mtp-setup.csv` explicitly reports deferred ordinary CPU forward setup and
`controller_execution_proved=0`. Source admission does not execute the depth
controller or certify numerical inference.

Sources: [typed extensions](../../../../tests/v2/integration/parity/qwen4exp/Qwen4ExpSourceAdmissionCases.h),
[fixture registration](../../../../tests/v2/integration/parity/qwen4exp/Test__Qwen4Exp_Production_Parity.cpp),
[lifetime and owner proof](../../../../tests/v2/integration/parity/qwen4exp/Qwen4ExpSourceAdmission.h),
[actual plan and source metadata proof](../../../../tests/v2/integration/parity/qwen4exp/Qwen4ExpSourceAdmissionEvidence.h).

## Current qualification

**The complete source matrix is not qualified.**

| Scope | Evidence and result |
|---|---|
| Last complete prerequisites | Current PLE/capacity slice: **3,267 passes, zero failures, 29 unchanged hardware-unqualified**. Unit855/855; source/runtime unchanged, drivers clean, 2,989 owned session bases empty. [Evidence](QFN_PLE_PIPELINE_ADMISSION_REPAIRS.md). |
| Current native slices | **CUDA16/CPU32 and ROCm16/CPU32: 32 host-tier/MTP identities pass**. RAM retains r10/r20; mmap and zero/positive host-adaptive cache pass 12/12 per backend in r23/r24. Full48/active retained15, retirement, prefix cleanup and driver checks pass on each recorded runtime. |
| Rechecked failures | **6/6 pass in r26**: mixed CUDA/ROCm/CPU PP fixed3/Dynamic; ROCm TP4 random dynamic expert placement with sharded VRAM ngrams fixed1/fixed3; CPU NUMA PP2 host-adaptive fixed1/fixed3. Original full48/retained15 policies, retirement, empty session bases and driver checks pass. |
| CUDA TP2 placement sweep | **16/16 pass in r27**: eight GPU experts, RAM ngrams, Static/Dynamic × ordinal/random × all four MTP modes. Original full48/active retained15, exact retirement, empty sessions and clean driver window. `source-admission-cuda-tp2-ram-policies-r27/`. |
| ROCm TP4 policy sweep | **6/12 pass in r28**. Active replicated decode needs about 2.23GiB more per MI50; no remaining duplicate charge found. Six original positives require 48GiB cards; all six new 32GiB-cap controls pass in r32. All r28 cases retire; driver checks pass. `source-admission-rocm-tp4-ram-policies-r28/`. |
| CPU/NUMA/mixed placement | **40/43 pass in r29**. [All three PLE-only PP failures are repaired](QFN_PLE_PIPELINE_ADMISSION_REPAIRS.md): **3/3 pass in r31** with original full15, cleanup and clean driver evidence. Focused entry/replay and failure-retirement controls pass. `source-admission-cpu-numa-mixed-placement-r29/`, `source-admission-ple-pipeline-mtp-recheck-r31/`. |
| Prepared physical roots | r19 shares the runtime's exact identity predicate with planning, removing replicated-decode sidecar double charges. The original CUDA16/CPU32 capacity failure is resolved by r20 native evidence. ROCm24/24 retains a 317,565,660-byte active-capacity deficit and separate native-domain composition work. |
| CUDA capacity controls | Four-expert VRAM retained3 positives and eight-expert 13GiB-cap rejections pass 8/8 with exact MTP/cap evidence and retirement. `source-admission-cuda-capacity-controls-r5/`. |
| CUDA15/CPU33 | **16/16 native pass**: four storage policies × all four MTP modes; retirement and driver checks pass. `source-admission-cuda-cpu15-full15-r12/`. |
| Retained coverage | 634 declared, 484 eligible, 150 hardware-unqualified. r32: **197 historical passes**, 4 eligible failures, 283 unexecuted. Not current-binary certification. `/tmp/qfn-source-admission-coverage-r32-r1/`. |
| Native repairs | [NCCL init](QFN_NCCL_GIN_INITIALIZATION.md), [I/O retirement](QFN_IMMUTABLE_WORKER_RETIREMENT.md) and [PMA preparation/permits](QFN_NATIVE_ADMISSION_PREPARATION.md) have focused evidence. Opaque-resource implementation and native retirement propagation remain open. |
| GPU targets / ownership | SM80/86/89/90 and gfx906/gfx1100 compile; startup/exit/capture pass 20 repeats. Eight mixed-ISA defects fixed. Model qualification open; MMA deferred. [Status](ROCM_KERNEL_TARGETS_AND_DISPATCH.md). |
| Pooled local provider | **16/16 original CUDA/ROCm source cases pass**: zero/positive VRAM+RAM caches over SSD × Off/fixed1/fixed3/Dynamic, all 48 layers and active retained15. **62 focused replays pass**, including XTX and 23 GGUF formats. Driver and retirement checks pass. [Cold-admission proof](QFN_POOLED_LOOKUP_COLD_ADMISSION.md). Distributed placement remains unqualified. |

Native evidence directories above are under
`parity-results/qwen-flash-next/` unless an absolute path is given. Earlier
MTP and automatic-placement receipts remain independent earlier evidence;
they do not certify the expanded matrix.

[Composed capacity replay](QFN_COMPOSED_CAPACITY_REPLAY_20261029.md) records
earlier deficits; CUDA16/CPU32 RAM now passes unchanged. An explicit ROCm25/23
split fits the recorded metadata budget, but has no installed case or native
qualification; new native transport storage must also fit before admission.

Physical provisioning is independent of current free memory. Original active/full15
CUDA TP2 eight-expert VRAM positives require 32GiB cards; the six replicated-decode
ROCm TP4 positives require 48GiB. Smaller cards keep the separate fitting cases
and precise capacity-rejection controls. See [exact BOMs and provisioning](../../../testing/QFN_SOURCE_PROVISIONING.md).

The requested [pooled tiering extension](QFN_POOLED_NGRAM_CONTINUATIONS.md)
keeps ngram ownership independent of continuation topology, including cross-vendor
CPU/CUDA/ROCm. Current GPU-adaptive storage only supports a single GPU; TP/PP and
distributed tiering are implementation work, not an installed capability.

| Area | Installed contract / regression evidence |
|---|---|
| CPU identity and locality | Exact domain/rank/participant sources, native CPU allocator identity and NUMA policy; reversed order and foreign identities tested. |
| Prepared weights | Participant graph families reuse prepared engines. The actual backing retains its PMA lease through views, movement and final retirement; no duplicate live ledger. |
| Exact physical geometry | Shared immutable declarations supply arena, serial/grouped expert and router roots. Ten native encodings and floating storage retain allocator-rounded bytes; logical payload and temporary packing demand are separate. |
| CPU request and MTP capacity | Ordinary request banks, retained predictors and explicit execution phases have distinct declared geometry. TP1/2/4/8 schema sweeps cover ownership and retained capacity. |
| KV communication | Indexed CPU gathering sends exact live spans. Partial/empty work, 1/2/4/8 members, replay, aliases and synchronized abort have explicit regressions. |
| Source guards | Per-call allocation/preparation checks complement runtime ownership enforcement; deliberate bypass mutations must fail. Narrow infrastructure boundaries do not exempt entire subsystems. |
| Test-only admission | Mock fixtures author their actual BOM. Concrete production models reject test overrides at factory and direct rank entrypoints. |
| Native materialization | CPU arena, workspace and routed weights must have live claims when planned; service-only CPU resources do not invent compute buffers. Exact bytes and final-owner lifetime are tested separately. |

The [accounting repair note](QFN_CPU_MATERIALIZATION_ACCOUNTING_GAP.md)
retains earlier ownership changes and evidence. All 80 composed candidate
obligations remain enabled; passing RAM cases do not qualify other placements.

## Running and CI boundaries

Build the existing `v2_integration_parity_qwen4exp_production_matrix` target.
Use CTest JSON discovery to select exact `FullSourceAdmission`,
`CandidateSourceAdmission`, `RejectUnsupportedSourceAdmission` or
`RejectCapacitySourceAdmission` registrations.
Run through CTest, retaining
its exact MPI process, backend-isolation, model prerequisites and integration
lock. Set `LLAMINAR_PRODUCTION_PARITY_ARTIFACT_ROOT` to a fresh absolute result
directory. `LLAMINAR_SKIP_MODEL_DOWNLOAD=1` may reuse already installed sources;
it does not waive required source files. Collect all results in a focused batch
and retain a passive driver window around native GPU work.

The real-model cases have `Admission;RealModel;FullModel` labels, with
`NegativeAdmission` added to expected rejections, `CapacityAdmission` to the
explicit-cap probes, and `UnqualifiedCandidate` to
unfinished positive compositions. All remain enabled and outside model-free
`ProductionTestPreflight`. Fast discovery, declaration/negative-control, CPU TP
validator and MTP setup-evidence replay regressions are explicitly registered
in preflight. Real-model admission is wired into both feature PRs targeting
`develop` and `develop`→`master` release gates. Typed hardware eligibility and explicit physical provisioning must
be recorded; every selected case must run without skips. Unqualified candidates
remain positive obligations and must fail the gate until implemented. Numerical
campaigns keep their existing reference, full-stage and path-evidence contracts.

Local admission uses the portable default `V2_QFN_LOCAL_PAIRED_BINDINGS=`.
Numerical profile pins remain unchanged; source admission retains the authored
topology/arithmetic defaults and independent Off/active controls.

Each completed native case retains its declared `admission.csv`,
`rejection.csv` or `capacity-rejection.csv`, plus
`retirement.csv`; successful materialization also retains the actual PMA
attestation. Require the CTest result and all applicable evidence together.
Temporary prefix directories must be absent after final retirement.

## Remaining obligations

1. Complete preflight for the PLE/retirement/provisioning slice; focused native controls pass.
   CUDA/ROCm16→CPU32 now pass all host-table tiers and MTP modes.
2. Implement pooled publication, row delivery and retirement in parallel with
   admission testing. Integrate through the [shared completion plan](QFN_POOLED_COMPLETION_PLAN.md);
   distributed serving remains unqualified until native capture passes.
3. Run remaining locally available placement/ownership/MTP cases. The six
   implemented-topology historical failures now pass; broader coverage is open.
4. Resolve native ROCm PP(TP2, TP2) composition and active capacity failures.
   Qualify CPU-tail placements outside CUDA16/32, ROCm16/32 and CUDA15/33
   families; preserve authored full15 positives. CUDA4/8 and ROCm8
   remain unqualified on this CUDA2/ROCm4/two-NUMA host.
5. Qualify the active MTP product across table sharding/adaptive tiers and
   composed graphs. Numerical/controller execution remains separate; existing
   paired-source MTP execution admissions remain intact.
