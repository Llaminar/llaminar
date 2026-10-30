# QFN source-admission matrix

Updated 2026-10-30. **Size limit: 15,000 UTF-8 bytes.** This is a bounded status
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

The typed case vectors and generated CTest registration own exact identities.
The counts below summarize them; do not maintain another runner-side manifest.

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

The six new compositions are CPU NUMA0+1 TP2; CPU0→CPU1 PP2; ROCm4 as
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

The original vectors plus the 15/33 layer family contain **152 Off identities**. Three special probes cover
CPU Automatic, CUDA Automatic under an explicit 24 GiB cap, and whole-table
ROCm VRAM in a PLE-owning pipeline stage. These form **155 Off seeds**.

Every seed now expands over **Off, fixed1, fixed3, and Dynamic**: **620 declared
base source-admission cases**, comprising 492 supported-intent positives, 80
composed candidates and 48 precise rejections. Two additional
authored four-policy families bring the total to **628**: 496 supported-intent
positives, 80 candidates, and 52 rejection controls. CUDA TP2 adds a four-expert
VRAM family with retained MTP capacity3 and an eight-expert family under an
explicit 13GiB public GPU cap. Original full15 identities remain unchanged. Generated discovery
and 11 focused declaration/evidence regressions pass. Existing Off names
and source closures remain unchanged. Base active cells retain depth15 capacity; the bounded CUDA family retains
depth3. Both use the original R1 predictor; Dynamic keeps automatic initial depth and the default
controller cadence. The production speculative verifier remains selected.

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

Remote checkpoint `6f4c742e7` contains stage-owned CPU maintenance, NUMA
transfer planning, exact MTP routed-parent/workspace ownership and the distinct
CUDA/CPU 15/33 layer family.
**The complete source matrix is not qualified.**

| Scope | Evidence and result |
|---|---|
| Last complete prerequisites | **r13 green**: 790/790 Unit; 2,258 preflight passes and 29 unavailable-hardware skips; zero failures. Complete 3,077-registration gate took 5,088.69s. Driver window clean. `/tmp/qfn-admission-r13-prerequisites/` and sibling `-observer/` authenticate checkpoint `6f4c742e7`. |
| Focused repair proof | 13/13 affected preflight checks pass; original workspace implementation fails all three ownership regressions. `/tmp/qfn-cuda-cpu-fit-r12-focused.xml`. Earlier CI-builder failure and scoped closure remain in `/tmp/qfn-admission-r5-{prerequisites,focused-closure}/`. |
| Current native slice | r10: **21/28 pass; 28/28 retire**, unchanged runtime and clean driver window. Eight CPU TP2/PP2 and four all-GPU ROCm TP4 RAM controls pass across Off/fixed1/fixed3/Dynamic. Nine composed RAM candidates pass: CUDA2→CPU2 Off; ROCm2→CPU2 and CUDA2→ROCm2→CPU2 in all four modes. `source-admission-cpu-mtp-r10/`. |
| Workspace-owner native repeat | r11: 6/11 pass; all retire, runtime unchanged, driver clean. Earlier GPU stages no longer pay for CPU-tail head/predictor workspace: 1,121,648,640 B/GPU saved. Original CUDA16/CPU32 active cases still exceed observed available VRAM by 137,388,696 B. ROCm PP still needs native composition/capacity work. `source-admission-workspace-owner-r11/`. |
| Original-code controls | Reproduce NUMA allocator duplication, MTP routed-parent duplication and incorrect CPU global-layer certification. `/tmp/qfn-cpu-numa-original-control-r10/`, `/tmp/qfn-mtp-routed-replica-owner-r1/`, `/tmp/qfn-cpu-offset-original-control-r8/`. |
| CUDA capacity controls | Four-expert VRAM retained3 positives and eight-expert 13GiB-cap rejections pass 8/8 with exact MTP/cap evidence and retirement. `source-admission-cuda-capacity-controls-r5/`. |
| Added CUDA15/CPU33 family | **16/16 native pass**: four storage policies × Off/fixed1/fixed3/Dynamic, all with exact retirement, unchanged runtime and clean driver checks. `source-admission-cuda-cpu15-full15-r12/`. |
| Current discovery | 628 declared, 484 selected, 144 hardware-unqualified; r12 `hardware-selection.json` retains exact identities. This is not full-matrix execution. |
| Native init repair | [NCCL GIN fix](QFN_NCCL_GIN_INITIALIZATION.md): source replay and CUDA capture/reset pass; full native admission remains open. |

Three r10 intervals conservatively overlapped private CUDA property queries.
`source-admission-cpu-mtp-r10-quiet/` repeats their same two passes and one
capacity failure, with 3/3 retirement, unchanged runtime and clean driver checks.
The original overlap record remains `/tmp/qfn-source-r10-query-overlap.json`.

Native evidence directories above are under
`parity-results/qwen-flash-next/` unless an absolute path is given. Earlier
MTP and automatic-placement receipts remain independent earlier evidence;
they do not certify the expanded matrix.

The added CUDA15/CPU33 family retains all expert policies, full15 MTP capacity
and every host-table placement. It complements the unchanged CUDA16/CPU32
positive. Its complete source-admission family now passes; inference and
numerical qualification remain separate.

[Composed capacity replay](QFN_COMPOSED_CAPACITY_REPLAY_20261029.md) records
the remaining original CUDA16/CPU32 and ROCm24/24 deficits. An explicit ROCm25/23
split fits the recorded metadata budget, but has no installed case or native
qualification; new native transport storage must also fit before admission.

The 24 original active/full15 CUDA TP2 eight-expert fixed-head VRAM positives
retain their identities and require an authored **32GiB actual total VRAM per
participating GPU**. The recorded exact M64 BOM is 27,479,004,096 bytes versus
24,852,365,312 available locally. This pure-TP policy already excluded routed
predictor parents from its separate sidecar BOM; the replicated-decode repair
must not be subtracted from that total. The provisioning class is independent
of current free memory. Smaller cards leave these positives unqualified;
Off, four-expert retained3 and explicit-cap controls remain mandatory. See
[source provisioning](../../../testing/QFN_SOURCE_PROVISIONING.md).

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

1. Resolve remaining composed capacity/native-domain failures while preserving
   original full15 requests; CUDA15/CPU33 now passes its complete family.
2. Keep the complete gate green as further implementations land; r13 qualifies
   the installed checkpoint, not the private native-transport/pooled work.
3. Run remaining locally available placement/ownership/MTP cases without
   equating a sampled batch with the full matrix.
4. Resolve native ROCm PP(TP2, TP2) composition and active capacity failures.
   Qualify CPU-tail placement variants beyond the passing RAM slice; preserve
   authored full15 positives while correcting ownership. CUDA4/8 and ROCm8
   remain unqualified on this CUDA2/ROCm4/two-NUMA host.
5. Qualify the active MTP product across table sharding/adaptive tiers and
   composed graphs. Numerical/controller execution remains separate; existing
   paired-source MTP execution admissions remain intact.
