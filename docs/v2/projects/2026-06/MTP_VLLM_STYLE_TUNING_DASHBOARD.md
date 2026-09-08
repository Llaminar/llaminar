# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CPU, CUDA, and ROCm. Keep implementation
detail in project handoffs.

RAG: **G** correct and economical, **A** correct but untuned/stale, **R**
failing or not yet proven. Token equality alone is not verifier parity proof.

## Current State

September 8 WIP checkpoint: additive packed-prefill exact dispatch finishes its
clean bracket at **1186.524 prefill / 46.485 decode tok/s**, with unchanged
tokens and weight/workspace bytes. Pinned llama.cpp confirmation is
**1204.301 / 45.757**: decode leads, prefill still trails. All 147 format/shape
cells and 45 isolated zero-spill launch profiles pass. Full Unit is **638/638**;
the 112 preflight integrations and twelve model cells are still in progress:
`/tmp/qwen38-staged-prefill-proof.{json,log}`. The final gate is not yet claimed
green. Dynamic's physical verifier-width fix and its per-backend >=90% target
remain pending. See the
[phase investigation](../2026-09/2026-09-07-cuda-mtp-off-phase-comparison.md).

September 8 floating-prefill follow-up: shared-operand projection dispatch
improves clean CUDA prefill by about 2.3% in a control/retest bracket, ending at
**1146.532 prefill / 46.495 decode tok/s**, with identical tokens and unchanged
weight/workspace bytes. A refreshed same-pinned llama.cpp run gives
**1208.976 / 45.787**: decode still leads, prefill still trails. All native
floating formats have byte/resource and boundary-economy evidence; the complete
affected gate passes **638 Unit + 112 preflight + twelve model cells + 106 CSVs**
in 576.680 seconds: `/tmp/qwen38-tiny-shared-proof.{json,log}`.
This does not change dynamic-depth execution; its maximum-width verifier
remains the pending target. See the September 8 section of the
[phase investigation](../2026-09/2026-09-07-cuda-mtp-off-phase-comparison.md).

September 8 parallel-attention checkpoint: deterministic mode now retains the
normal ordered KV-split policy on CUDA and ROCm, including removal of HIP's
device-side single-split override. Clean CUDA MTP-off decode is **46.576 tok/s**
(43.977 before), ahead of the pinned llama.cpp 45.689 comparison; prefill is
1126.644 and still trails. ROCm's refreshed MTP-off baseline is **30.804 tok/s**.
No weight/activation precision or attention workspace increase. Deterministic
and normal parallel serial/grouped outputs are byte-identical in 288 captured
configurations per vendor; old single-split vs new model tokens first differ
at index 231, so do not claim old/new output byte identity.

The consolidated ROCm request-cache tests pass 20 repeats and now join the
preflight label alongside ROCm grouped/context attention. The expanded gate
passes **638 Unit + 112 preflight + all twelve Qwen3.8 CUDA/ROCm cells**, with
106 CSV artifacts in 574.176 seconds:
`/tmp/qwen38-parallel-deterministic-proof.{json,log}`.
The full ROCm attention binary is 59/61: two legacy real-Qwen2 fixture setup
failures lack a PhysicalMemoryAuthority and remain explicitly open outside
this model-free preflight. Dynamic physical verifier width is **not fixed yet**.
Re-establish fixed-depth winners after the attention change before certifying
the dynamic >=90% objective; the measurements below predate that change.

September 8: fixed-depth 1/2/3 now measures **64.193/67.226/65.640 tok/s CUDA**
and **33.436/38.418/38.041 ROCm**, with identical tokens within each backend.
Depth 2 leads this neighborhood; the complete depth-through-15 search is pending.
The startup parser now refreshes the canonical kernel-policy snapshot after
`--deterministic`: a reproducing Unit passes twenty repeats and all 181 parser
tests pass. Corrected direct profiles match canonical output tokens. They expose
the next ordinary decode target: deterministic mode forces attention to one KV
split on both vendors. CUDA attention measures 1.956 ms/token in the matched
trace, not the earlier unmatched 0.428 ms. Full Unit/preflight/model validation
of startup publication passed: 638 Unit registrations, 109 preflight integrations,
and all twelve CUDA/ROCm model cells with 106 validated CSV artifacts in 491.887 s.
At that earlier checkpoint, neither attention parallelism nor dynamic
execution-width selection had changed.

**Starting after the green attention decode slice:** tune dynamic-depth MTP on
CUDA and ROCm to **at least 90% of each backend's best fixed-depth decode** for
the same Qwen3.8-27B / exact 512-token prompt. Initial fixed-2/dynamic-15
measurements are **67.226/23.950 tok/s CUDA** and **38.418/13.990 ROCm**:
dynamic delivers only 35.6% and 36.4% respectively. A diagnostic ceiling of 2
recovers 65.367 CUDA / 37.176 ROCm with unchanged tokens and adaptive counters.
This isolates a large capacity-dependent execution tax, not a finished fix;
the production solution must retain depth 15. September 8 normal-bootstrap
captured-event evidence localizes 99.6% of the ROCm capacity-dependent loop
increase to its verifier: 121 identical-count replays take 51.506 ms each at
physical M3 versus 144.917 ms at M16. Outputs and adaptive counters match.
Best-fixed-depth inventory, per-kernel attribution and implementation remain
pending; do not tune controller thresholds around this execution-width tax.
The [project plan](MTP_VLLM_STYLE_PROJECT_PLAN.md#dynamic-depth-economy-cuda-and-rocm)
defines the matched Release bracket, fixed-depth search through 15, unchanged
precision/memory constraints, controller-cost evidence and correctness gates.

September 7 23:40 UTC, attention decode spill cleanup: **638 Unit,
109 preflight, twelve CUDA/ROCm Qwen3.8 cells and 106 CSVs pass** in
489.473 seconds. The final linked FP16 decoder has zero stack/local bytes and
zero measured spills; Release model speed is unchanged within noise at
1123.878/43.977 tok/s. The ordinary full-generation binding and original
llama.cpp throughput target remain open. The maximum-width dynamic verifier
tail is the source-audit lead for the measured capacity tax; exact kernel
attribution and the production fix are still pending.

September 7 22:28 UTC, ordinary live-frontier publication: **638 Unit,
108 preflight and all twelve CUDA/ROCm Qwen3.8 cells pass**, with 106 validated
CSVs, in 827.206 seconds including the Unit rebuild. Both backends share the
response/position/next-condition transition and pass captured aliasing and
twenty-reset proofs; isolated profiles show zero scratch/spills. The ordinary
model loop is not yet connected and both off cells still report no complete
generation-loop certificate. Reuse the existing sampler/logical-state buffers
for the pending production binding. Last retained Release throughput remains
1117.762/43.930 tok/s against pinned llama.cpp's 1152.615/45.689; the goal stays
open. See the [current phase investigation](../2026-09/2026-09-07-cuda-mtp-off-phase-comparison.md)
for the fresh receipt, per-cell timings, scope limits and lifecycle map.

September 7, ordinary entry/terminal composition: CUDA can now place its
captured prefill sample before the first WHILE predicate. One immutable
admission and shared validator authenticate ordinary/MTP terminal accounting;
the seven pure controller tests and focused native-prologue proof pass. The
fresh selected gate is **635/635 Unit, 96/96 preflight, 6/6 CUDA Qwen3.8 cells
and 53 CSV artifacts**, in 379.455 seconds. One stale source-count assertion
in the first prerequisite run was replaced by explicit lowering boundaries and
the prologue-to-predicate edge. Ordinary model-loop wiring and a new throughput
measurement are not complete; the off cell still truthfully reports no
generation-loop certificate. See `qwen38-ordinary-contract-proof-v2` in the
tuning receipt for the durable run identity and per-cell timings.

September 7, ordinary controller foundation: a typed ordinary generation policy
now shares the existing 44-byte admission ABI and 46-word resident ledger.
All **635 Unit, 96 preflight and six CUDA Qwen3.8 cells pass**, with 53 validated
CSV artifacts, in 397.622 seconds. Captured CUDA/ROCm publication/reset tests
pass; isolated publication profiles show zero spills on both backends.
This does **not** yet replace the model's host-per-token loop or
change the last measured throughput. See the current tuning receipt below for
the exact first-sample, pending-condition and EOS lifecycle.

September 7, MTP-off follow-up: retained source confirms **1075.182 / 42.475
tok/s** prefill/after-prefill decode, unchanged output bytes and VRAM. The
exhaustive Q5 unpack primitive joins Unit; **634/634 Unit, 96/96 preflight,
6/6 CUDA Qwen3.8 cells and 53 CSV artifacts pass** in 392.645 seconds. Slower
CTA-local reduction and balanced-unpack prototypes were removed. Attribution
also exposed a structural gap: ordinary decode captures each forward but
still submits/samples through the host per token; complete device-owned
generation is wired only for MTP. Existing MTP-off math/capture certificates
do not prove that stricter full-generation contract. The
[tuning receipt](../2026-09/2026-09-06-cuda-qwen38-mtp-off-tuning.md)
maps the current/target lifecycle and records rejected experiments. The
llama.cpp target remains unmet; no precision, weight, workspace or generated
dispatch-policy change was used to improve the numbers.

September 6, MTP-off focus: the user has made non-speculative CUDA Qwen3.8
prefill/decode the active comparison target. A BK64 partition-boundary cursor
improves unprofiled Release prefill **957.946 -> 1065.861 tok/s (+11.27%)**;
after-prefill decode is unchanged at **42.426 tok/s**, versus llama.cpp
**1175.520 / 45.664 tok/s**. All 256 output IDs and physical workspace bytes
are unchanged. IQ1_M retains its original economical register schedule after
rejecting a spill-free but slower one-CTA variant. **Unit 633/633, preflight
96/96, and all six CUDA Qwen3.8 cells with 53 validated CSV artifacts pass**;
the selected aggregate completes in 380.84 seconds using cached model/reference
data. This is a green correctness slice, not a completed performance goal.
Next attribute Q5 projection instructions and ordered-reducer publication;
do not change arithmetic, precision, weights or workspace to close the gap.
See the
[MTP-off tuning receipt](../2026-09/2026-09-06-cuda-qwen38-mtp-off-tuning.md).

September 6, CUDA Qwen3.8 tuning: small floating projections now preserve the
fixed reduction with one block barrier for decode and no block barriers for
prefill. The first Release sample improves 512-token prefill **930.700 ->
957.651 tok/s**, while MTP3 decode is effectively unchanged at **63.701 tok/s**.
Final-source confirmation gives 948.547/63.507 tok/s; the prefill gain is 1.9–2.9%,
not a fixed best-case result. Outputs and workspace bytes are unchanged. All
1,076 isolated byte-oracle cases, Unit 633/633 and preflight 94/94 pass. Fresh
llama.cpp measures 981.901/70.600 tok/s. Its output diverges at token 77 and it
accepts more drafts, so the remaining decode gap needs transaction/acceptance
attribution as well as kernel timing. The external goal is still open. See the
[focused tuning receipt](../2026-09/2026-09-06-cuda-qwen38-tiny-projection-tuning.md)
for profiles, exact commands, rejected candidates and remaining work. The fresh
canonical CUDA parity matrix passes all six MTP cells and all 53 CSV artifacts;
MTP3 draft-head cosines are 0.999910/0.999922. A separate Release 512-prompt /
256-output serial run matches every MTP3 output token in all repetitions:
42.587 tok/s serial versus 63.507 with MTP3. No mathematical defect was exposed;
the short-prompt HF comparisons and longer token witness remain distinct proofs.
Earlier
campaign progress below is historical; the September certification dashboard
records thirteen individually green E2E cells with their repeat gate paused.

September 6, 13:40 UTC: **A, final-parent attachment; R, aggregate.** CUDA2/CPU2
fails setup because its one retained parent cannot use the former direct-capture
fork/join interface. The shared CUDA/HIP native-DAG attachment removes those
paired events and recording flags. Focused tests pass on both GPUs; 22 transfer,
87 engine and three CPU-only DAG tests pass. At 13:51 twenty fresh-process
stress rounds pass (520 test executions), Unit is 633/633, preflight is 93/93
and Release is rebuilt. The CPU-tier retry passes all behavior and clean
shutdown but times out during its 740281-record artifact handling. Offline
validation passes. Single-owner collection/validation preserves byte-identical
evidence and improves isolated processing from 18.929 s to 10.819 s; 106 focused
tests and Unit 633/633 pass. The fresh online retry passes all 43/43 checks and
740078 evidence records in 597.234 s, including clean logs/shutdown and zero
residual GPU memory. The old timeout remains red. Its 2.766-second runtime
margin is narrow; this is not a full repeat/economy certificate. Ornith 1.5
LocalTP 2xCUDA passes 43/43 checks in 109.319 s, including all eight long checks
and 34439 evidence records, with clean shutdown and zero residual VRAM.
Qwen3.6 MoE NodeTP 2xCPU passes 43/43 in 572.911 s, all eight long checks and
877663 evidence records, with clean shutdown and no GPU use. Ornith NodeTP
2xCPU also passes 43/43 in 572.858 s, all eight long checks and 838794 records,
with clean shutdown and no GPU use. Four of thirteen have fresh passing
receipts. The remaining-nine run's first cell, 122B ROCm2/CPU2, passes 43/43 in
552.792 s with all eight long checks, 700812 records, clean shutdown/logs and
full VRAM release. ROCm4/CPU2 then passes 43/43 in 512.371 s with 558140
records, and Qwen3.6 MoE CUDA1 passes 43/43 in 55.196 s with 7692 records.
Both retain all eight long checks, clean shutdown/logs and exact VRAM return.
Qwen3.8 dense CUDA1 then passes 43/43 in 158.479 s, with all eight long checks,
5757 records, clean shutdown/logs and exact VRAM return. Eight of thirteen are
fresh green; mixed 122B CUDA2/ROCm4 runs next.

After all thirteen individual cells are green, pause the twenty-repeat gate for
a fixed-depth-three Release comparison against then-current upstream llama.cpp:
Qwen3.6 MoE 35B and Qwen3.8 dense 27B on CUDA1 and ROCm1, identical inputs,
with separate prefill/decode wins required in every comparable cell.

Mixed 122B CUDA2/ROCm4 passes 43/43 in 389.070 s with all eight long checks,
107654 records, clean shutdown/logs, exact VRAM return and no recurrence of the
long transfer warnings. Nine of thirteen are fresh green; Ornith RCCL ROCm2
runs next.

Ornith RCCL ROCm2 passes 43/43 in 159.072 s with all eight long checks, 52878
records, clean shutdown/logs and exact VRAM return. Ten of thirteen are fresh
green; Qwen3.6 MoE ROCm1 runs next.

September 6, 13:03 UTC: **G, targeted mixed 122B E2E; R, full aggregate.** The bounded
transfer service and TP-worker capture fix pass Unit 633/633, preflight 93/93,
and 20 symmetric fresh-process repetitions. The latest retry passes readiness
and two needles, then fails at MTP depth 14: hosted verifier replay explicitly
omits the branch authority installed at capture. A device-free reproducer is
red-before. The engine now resolves the policy from the same runner on capture
and replay, retaining strict rejection of changed ownership. The 200-replay
regression, all 87 engine tests, both builds, Unit 633/633 and preflight 93/93
pass. The exact retry passes 43/43 checks and all eight long checks in 394.907 s,
with clean shutdown, zero residual VRAM and no warnings. It commits 170 moves
(36 promotions/36 demotions/98 same-priority), 3.209 GB; remote endpoint p95 is
410.561 ms versus the prior 26.870 s. Other cells, CPU-quantized progress,
service accounting and full-suite repeat gates remain open. No end-to-end
throughput speedup is claimed. The September handoff owns diagrams/receipts.

September 6: **R, full HTTP aggregate not certified.** The HTTP/SSE delimiter
fix and two event-ordering fixes pass Unit 633/633, preflight 92/92, and twenty
symmetric GPU process repetitions. Arithmetic now requires natural EOS, not a
correct final number inside a length-exhausted loop. MTP-on/off agree after the
event fixes. The remaining arithmetic-B loop also reproduces in the independent
CPU/FP32 reference on the exact GGUF and native forced prefix. Our model schemas
omitted Qwen's paragraph boundary before the forced stop phrase. Correcting
only those bytes changes the reference to `14` plus EOS. A shared model policy
and three-schema red-before regression are installed; Unit 633/633 and preflight
92/92 pass. Twenty full CUDA E2E repetitions pass 20/20, each 43/43 checks in
54.49–55.60 s. The fresh aggregate passes both CUDA single-device cells, then
mixed 122B CUDA2/ROCm4 times out preparing transaction-36 physical transfers
during near-boundary prefill. Earlier needles/long generation pass; missing
PerfStats and shutdown errors are consequences of MPI abort. Fatal-only
projection diagnostics are building; root cause is not yet established.
Detailed receipts and the lifecycle map live in the September E2E handoff.

Historical follow-up: the 2048-operation CUDA KV archive pressure regression passes
20/20; it does not reproduce the mixed-model stall. A topology-wide diagnostic
rerun instead exposes an earlier lost empty-command race: CUDA opens snapshot
17 and clears command 16 before ROCm acquires it. The shared CUDA/HIP and CPU
fix retains sealed commands until the existing next-snapshot fan-in and keeps
new phase intent separate. CPU regression is red-before/green-after; a retained
real-GPU completion/open pair passes 20 delayed-follower replays with each
vendor as authority. Full gates are rebuilding. No new E2E cell is certified;
the original archive stall and CUDA2/CPU2 economy timeout remain open. See the
September certification handoff for exact receipts and the lifecycle diagram.

Latest 2026-09-05: NCCL resumed-capture regression is fixed (20/20), installed
under the canonical dependency SONAME, and gated by Unit 632/632 and preflight
90/90. Paired Release collective replay ratio is 0.99907 fixed/control.
CUDA2+CPU2 122B now passes startup and shared-prefix requests but fails forced
SSE decode: placement floor 19 versus an acquired epoch-18 reader. KV-only MTP
now publishes KV readiness without acquiring expert residency. Typed ownership
and observer-only maintenance waits pass CUDA/HIP 20/20 each, Unit 632/632,
preflight 90/90. The E2E retry clears SSE and four needle/JSON checks, then hits
the 600-second watchdog in structured generation. No stale epoch through 127;
throughput and pending-DMA diagnostic attribution remain open. Not an E2E pass.
Command-local DMA age/throttle now passes Unit 633/633 and preflight 90/90;
warnings retain severity and expose their exact command age. CPU sampling shows
substantial OpenMP waiting and NativeVNNI projection work, not a proven tuning
win. 122B ROCm4+CPU2 passes all eight behavioral checks unchanged, including
2048-token generation and 7595/8192 context, but is not certified: file-size RAM
heuristics and shutdown/rank-export defects lose final evidence. Canonical
memory attestation, rank-qualified collection and zero-only clean shutdown are
implemented; 71 focused harness regressions, Unit 633/633 and preflight 90/90
pass. The exact ROCm4+CPU2 retry fails its first MTP answer after readiness:
catch-up terminal refresh consumes prefill lengths instead of verifier geometry.
The redundant refreshes are now removed: typed scratch retirement precedes the
existing captured accepted-state publication. The initial API also drops its
irrelevant row count. Two-stream captured regressions pass 20/20 per backend;
both builds, Unit 633/633 and preflight 90/90 pass. Exact ROCm4/CPU2 passes all
behavior and clean teardown (514.54 s). Its original receipt rejects two valid
evidence types: retained parents and mapped activation collectives. Corrected
validators pass 77 focused tests and revalidate all 550186 saved records.
The fresh end-to-end receipt is green in 523.24 s: all eight long checks,
39/39 harness assertions, 555186 validated all-rank records, clean shutdown,
and zero post-teardown VRAM delta. ROCm2/CPU2 also passes all eight long checks
and 39 assertions in 597.62 s, with 705832 validated records and clean teardown.
Its 2.38-second watchdog margin is fragile. Four historical passes still need
refresh, and the remaining failing cells are not certified.

Small-helper admission evidence: 128 retained 64-kernel graphs consume 80 MiB
cold / 0 MiB warm on CUDA and 256 MiB per lifetime on ROCm. Focused certificates
and both full graph suites pass. Typed owner classification and capture-shape
enforcement are now implemented: depth-15/request-one has 82 bounded helpers
and 25 general auxiliaries, with the same 107 total owners. CUDA admission
decreases 1640 MiB per runner; ROCm admission is unchanged. Both builds, four
focused graph suites, Unit 633/633 and preflight 90/90 pass. Dense Qwen3.8 CUDA
now passes full E2E in 151.16 s: 39 assertions, all eight long checks, 2048
generated tokens, clean shutdown and zero VRAM delta. Seven of nine cells have
successful receipts; earlier shared-change refreshes are still due. The
CUDA2/ROCm4 122B retry reproduces its prior prefix-progress stall in 282.98 s:
first needle passes, then ROCm prepared transaction 28 waits on CUDA at 27;
the CUDA request thread is blocked in KV archive launch. CUDA debugger attach
fails internally and terminates a separate diagnostic, providing no device
kernel evidence. No new runtime workaround was installed. Model-free
archive/controller queue-ordering reduction is next.

2026-09-05 E2E update: CPU2 Qwen3.6 MoE reaches server readiness after the
distributed-TP memory fix, but its first MTP request exposed a constant sparse
operation ID. A host-owner sequence shared by retained sidecar variants is
under validation; the production-runner regression reproduced step `0` on all
20 executions before the fix. Runner and real-MPI regressions now pass 20/20
repeats each, with Unit 632/632 and preflight 89/89. CPU E2E advances through
the first long needle but exposes a separate grouped-policy cache collision
(M=258 aliases M=2); its full-width identity fix passes 20 focused repeats,
CPU all-format grouped-verifier integration, and refreshed Unit 632/632.
Refreshed preflight passes 89/89. CPU2 E2E is now fully green: 40/40 checks,
all eight full-context proofs, 2,048 completion tokens, clean shutdown in
572.2 s. Economy remains close to the 600-second cell watchdog.
Mixed122B remains red in prefix KV archive submission.
122B CUDA2+CPU2 now clears CPU expert preparation after first-touch allocation
isolation and fault-resolving NUMA certification fixes. Large concurrent NUMA
and guard/accounting regressions pass 20 repeats; refreshed Unit is 632/632 and
preflight 89/89. The rejected prefill fragment contains a non-clonable CUDA
conditional. Direct recording into the eventual parent passes 20-repeat graph
tests, refreshed Unit 632/632, and preflight 89/89, but real E2E now fails during
the second fragment's NCCL rooted reduction. Repeated capture-to-graph sessions
keep the same CUDA capture ID; NCCL 2.28.9's ID-only strong-stream cache reuses
a stream that stopped capturing. Add this missing NCCL/direct-recording
intersection to the integration gate before completing the fix. No fresh
full-cell pass is claimed for this topology yet.
See [current evidence and lifecycle maps](../2026-09/2026-09-05-model-parity-e2e-certification.md).

| Goal | Completion | Remaining proof |
|---|---:|---|
| SingleDevice fully device-resident MTP | 97% | refresh d1 economy and stochastic matrix |
| LocalTP fully device-resident MTP | 99% | remote participants and economy matrix |
| ExpertParallel fully device-resident MTP | 97% | mirrored-head batching across every EP mode |

- Homogeneous CUDA/ROCm execution is full-graph only. Device parameters select
  regimes inside one immutable capture; attention never requests recapture.
- GPU KV append, TurboQuant, verifier publication, and collectives use
  persistent workspace and explicit events. Hot paths have no allocation,
  blocking sync, segmentation, or intermediate host observation.
- Prefix restore preserves graph addresses through producer events. GPU
  FFN/MTP APIs reject null publication streams. CPU and GPU MTP terminal-hidden
  mailboxes are capacity-complete arena owners whose addresses remain fixed for
  the runner lifetime.

## Canonical CUDA2/ROCm2 LLEP Target

`LLEP` is the ordinary large-prefill routed-row assignment axis, not a
shorthand for an entire MoE execution mode. The active tuning target is this
explicit tuple:

| Axis | Required policy |
|---|---|
| Dense/shared trunk | tensor parallel |
| Routed expert storage/compute | apportioned whole experts |
| Routed phase | uniform |
| Grouped verifier/decode assignment | static owner |
| Ordinary large-prefill assignment | least-loaded resident (LLEP) |
| Small-prefill regime | static owner below the explicit routed-row threshold |
| MTP terminal norm/head | mirrored full vocabulary |
| Durable residency maintenance | off |
| Hot expert replica cache | off |
| Transport | one fully captured NCCL/RCCL graph |

The production large-prefill boundary is currently `M * top_k >= 8192` routed
rows. Setting it to zero is reserved for focused transfer-path proof. This is a
declared work-regime switch: grouped verifier/decode remains static-owner and
must never inherit least-loaded prefill assignment. Dynamic whole-expert
residency maintenance and hot replicas are independent experiments and are not
part of the canonical LLEP economy row.

Canonical tests must prove the tuple through PerfStats: static-owner grouped
verifier calls, least-loaded current-batch prefill when the threshold is met,
mirrored terminal-head execution, graph-captured NCCL/RCCL, no segmented
execution, and no durable maintenance or hot-cache activity. Performance
tuning starts with stochastic fixed depth 3 so graph/communication economics
are isolated; dynamic depth is tuned only after that baseline is sound.

## Production Matrix

| Mode | Device | Dense greedy/stoch | MoE greedy/stoch | Status |
|---|---|:---:|:---:|---|
| SingleDevice | CPU | R/R | A/A | refresh paused |
| SingleDevice | CUDA | A/G | A/R | dense d3 wins; d1/MoE need tuning |
| SingleDevice | ROCm | A/A | A/A | dense d3 wins; d1/MoE need tuning |
| LocalTP | CUDA2 | A/A | A/A | Dynamic-maintenance/current-batch-LLEP matrix green; perf active |
| LocalTP | ROCm2 | A/A | A/A | Dynamic-maintenance/current-batch-LLEP matrix green; perf pending |
| LocalTP | ROCm4 | A/R | R/R | full refresh pending |
| NodeLocalTP | CPU2 | A/A | A/A | MoE long-context green; economy red |
| ExpertParallel | GPU+CPU | A/R | G/R | explicit Dynamic/LLEP greedy+prefix green |

## Correctness Proof

- The 2026-08-12 Qwen3.6 MoE SingleDevice production-checkpoint campaign is
  green on CPU, CUDA, and ROCm for fixed depths 1, 2, and 3 plus dynamic depth.
  The three `ALL_PRECISIONS` campaign targets take `138.14 s`, `104.14 s`, and
  `112.85 s` respectively from a warm authenticated reference pack: `355.13 s`
  (`5 min 55.13 s`) for the complete three-backend slice. A forced reference
  refresh followed by the same CUDA and ROCm runs takes `511.04 s`
  (`8 min 31.04 s`). The previous CUDA campaign took `1524.20 s`; retaining one
  graph-native serial oracle and one plan-certified immutable
  `ModelContext`/`PreparedWeightStore` per campaign removes repeated model
  loading while every depth still owns fresh request, arena, stream, graph,
  controller, and prefix state. CUDA and ROCm CSV evidence records full prefill
  and decode graph capture plus replay, zero segmentation, and prepared-weight
  reuse only after the first depth. Token traces prove actual depths 1/2/3;
  dynamic runs begin at depth 3, evaluate and update the device-owned policy,
  demote once, and finish at depth 2.
- The same campaign now authenticates `moe_router_snapshot_schema: 1` and MTP
  sidecar schema 5, so `MOE_ROUTER_OUTPUT` always means the complete
  post-softmax distribution. Probability checkpoints use direct row-wise
  symmetric KL rather than an erroneous second softmax. Every active MTP graph
  context also writes a `counterfactual_vs_mtp_router` row to the existing
  `mtp_sidecar_snapshot_breakdown.csv`: an independent double-accumulation
  projection loads the real GGUF gate and evaluates it on the exact live
  production `FFN_NORM` input. Across all three backends the worst causal
  relative L2 is below `3.3e-7` and worst symmetric KL below `6.9e-14`, proving
  the router kernel itself when recurrent upstream drift is larger. The
  device-free `V2_Unit_SnapshotCapture` gate rejects malformed distributions,
  proves worst-row behavior, and prevents reintroducing a second softmax.
- All-format grouped attention, TurboQuant, MoE routing, GDN, short-conv,
  stochastic target preparation, and draft publication sweeps are serial-row
  byte exact on their production backends and M ranges.
- The 2026-08-07 partial-terminal prefix/MTP failure was a CPU graph-lifetime
  defect. Scalar MTP first allocated a one-row terminal-hidden mailbox; later
  grouped publication replaced it with a wider tensor while cached CPU stages
  retained the freed address. Every backend now reserves
  `max(batch capacity, request_count * (depth + 1))` rows in one arena owner
  before graph construction, and runtime validation forbids allocation,
  rebinding, or reassignment. A focused scalar-to-four-row pointer-stability
  regression, typed CPU grouped-verifier routing tests, retained-producer role
  tests, and all six CPU/CUDA/ROCm ordinary-partial plus MTP-partial restore
  E2Es pass (`6/6`, `86.83 s` wall time). The KV lifecycle source sanitizer
  rejects reintroduction of the mutable-owner pattern. `GlobalOrchestrator`
  now preserves the typed grouped-verifier request across every execute and
  transfer step in GlobalTP/GlobalPP rank plans instead of erasing that role
  through generic `forward()` dispatch.
- The follow-on device-free gate exposed a deterministic MPI/OpenMP Q8 embedding
  repack defect: a ceil-divided 256-element grouped unpack crossed row boundaries
  whenever `d_model` contained fewer than eight Q8 blocks or had a block tail.
  Full-table and vocabulary-range repacks now share one implementation that uses
  grouped unpack only for complete eight-block groups and the bounded block API
  for every tail. Instrumented regressions prove short rows never enter grouped
  unpack while complete 256-element rows retain it. The canonical MPI-wrapped
  embedding suite passed 20 consecutive runs, the complete Integration tree
  rebuilt cleanly, and the unit/source gate passed `591/591` in `135.43 s`.
- CUDA MoE grouped verifier passed every native format at M=2..16 and M=31
  through the real router/expert path after the small-M grouping fusion.
- Release CUDA2/ROCm2 passed all eight Dynamic/LLEP cells and `166/166` checks
  through 2048 tokens with full capture and clean VRAM release.
- Release CUDA2 LLEP + RAM prefix + stochastic dynamic d4..15 passed `23/23`
  canonical server checks at context 4096 and 1024 output tokens.
  This includes deterministic prefix replay, forced movement, long recall,
  prefill replay, no segmented execution, clean shutdown, and PerfStats path
  assertions.
- The 2026-08-04 canonical grouped-verifier gate passed `89/89` explicit CPU,
  CUDA, CUDA2, ROCm, and ROCm2 cells. CUDA/ROCm KV publication is isolated by
  all 14 cache/source format pairs plus converted-read, logical-restore,
  adversarial, and TurboQuant lifecycle cells; each process uses the production
  graph-capture transaction and exact stream/event ownership.
- TurboQuant codebooks are uploaded once per cache/device on its construction
  stream and covered by the cache constructor's initialization fence. Launch
  wrappers cannot perform codebook upload, and no process-global ready flag can
  incorrectly alias initialization across devices. The complete Integration
  tree rebuilt cleanly and the device-free unit/source gate passed `585/585`.
- Expert-overlay policy is now represented as independent typed axes throughout
  config, graph lowering, fixtures, and E2E registration. The model-free
  canonical-tuple regression proves dense TP, apportioned/uniform routed work,
  static-owner grouped verification, least-loaded current-batch prefill,
  mirrored MTP head, and both durable maintenance and hot replicas Off. The
  complete Integration tree rebuilt all `1413/1413` edges and the device-free
  unit/source gate passed `585/585` on 2026-08-04.
- LocalTP routed-plus-shared publication now has one typed canonical transaction:
  every participant publishes router slots plus one rank-indexed shared bank,
  one rooted sum transports that payload, the root finalizes routes, shared
  banks, sigmoid gate, and residual in fixed serial order, and one broadcast
  publishes the terminal hidden rows. The previous routed reduce/broadcast plus
  independent shared allreduce/gate/combine transaction is absent from this
  policy. The production graph-lowering regression proves one reduce and one
  broadcast, exact producer dependencies, root-only finalization, and no old
  shared collective or epilogue nodes.
- The canonical publication kernels are byte exact against repeated production
  M=1 arithmetic for every `M=1..16,31`, full Qwen width 2048, and ragged/vector
  boundary widths `1,3,4,5,255,256,257,512,513`. Captured M=16 graphs replay at
  live M=11 without changing inactive rows. The complete CUDA and ROCm MoE
  grouped-verifier suites, including all native expert formats and real-path
  graph capture, passed in `189.83 s` and `109.47 s`, respectively. The full
  Integration tree then rebuilt all 971 affected targets and the complete
  device-free unit/source gate passed `585/585` in `143.32 s`. The strengthened
  source policy also forbids compute stages from inspecting raw tensor
  coherence after exact-stream publication; graph capture records that edge in
  its dependency ledger and replay owns the authority transition.
- The 2026-08-05 CurrentBatchLLEP long-context failure was a runtime-table
  identity defect, not a collective timeout. Ordinary prefill and durable
  static decode had aliased one placement table, allowing transient
  least-loaded assignment to leak into the next decode layer. Qwen3.5 MoE now
  declares typed `MainDecodeDurablePlacement`, `CurrentBatchLLEPPrefill`, and
  `MTPDepth` table roles; graph keys, prefix runtime state v4, bindings, and
  source/unit tests preserve those identities independently. CUDA2 and ROCm2
  CurrentBatchLLEP long-context cells passed after the change in `239.50 s` and
  `455.36 s`, respectively, with their canonical full-capture and PerfStats
  assertions enabled.
- A full CUDA FlashAttention fixture then exposed a second device-state defect:
  the cached-token parameter writer treated one shared FA2 parameter record as
  one logical query row, publishing `kv_len=1` for ordinary multirow prefill.
  CUDA and ROCm now distinguish shared prefill/M=1 records from row-local
  grouped-verifier records. Focused device-memory regressions cover initial and
  continuation prefill, active-row padded prefill, M=1 decode, and M=4 grouped
  verification. The complete CUDA and ROCm FlashAttention fixtures passed in
  `69.60 s` and `64.05 s`; CUDA captured cache growth/reset and ROCm all-format
  captured request-cache proofs remain byte exact. GPU cache views also require
  an explicit backend-qualified `DeviceId`, preventing a HIP allocation from
  being silently labeled as CUDA storage. The final Integration tree rebuilt
  all 784 affected targets and the device-free unit/source gate passed
  `585/585` in `135.90 s`.
- Long-context prefill now has an explicit `M=262145` totality gate on CPU,
  CUDA, and ROCm. The CPU cell uses Qwen2.5-0.5B geometry and one native Q8_1
  KV layer, processes exactly `64 * 4096 + 1` real rows, and proves native-byte
  first/final-row identity in `0.52 s`. Each GPU cell captures one 4096-row
  graph and replays it 64 times plus the one-row tail without per-chunk host
  slicing, transfer, allocation, or synchronization. A profiler-amplified
  final-tail miss exposed an unordered diagnostic D2H; the harness now records
  the terminal graph producer event and consumes it on the explicit observation
  stream before the sole terminal readback. CUDA and ROCm focused gates pass,
  including the exact `262145` device KV count. The CUDA materializer winner is
  32 threads at approximately `3.55 us`, 35 registers, and zero spills. The
  ROCm winner is 32 threads at `2.56 us` median and `2.734 us` pooled mean versus
  `2.72/2.890 us` for 256 threads; ISA evidence shows vectorized dwordx4
  loads/stores, 16 VGPRs, 52 logical SGPRs, and zero scratch, spills, LDS,
  barriers, or atomics.
- The CUDA NativeVNNI grouped WIDE/DIRECT path now publishes its only K
  partition directly from the weight-reuse CTA. Explicit RN multiply/add
  operations preserve the former store/load publication boundary byte for byte,
  while the obsolete publication kernel and partials-workspace dependency are
  gone. The canonical all-format grouped-verifier integration gate passed in
  `115.86 s`, covering native formats, FP32/FP16/BF16, LM-head M16, large-K
  KPAR, MoE projections, and fused SwiGLU against serial rows. On the
  Qwen3.6-35B IQ3_S terminal head (`248320 x 2048`, M4), Nsight reports one
  kernel instead of two, 48 registers/thread, zero local loads/stores, and an
  unchanged `464.0 us` counter-replay producer versus `464.5 us` before fusion;
  canonical latency remains approximately `414 us` and is therefore neutral.
- The shared release geometry catalog now exposes 12 unique MTP-specific
  matrices: `H x 2H` plus `248320 x H` for every distinct Qwen3.5/3.6 hidden
  width. The turnkey `qwen-mtp-head` profile sweeps all formats and M=1..16,31.
  A focused Qwen3.6-35B IQ3_S tournament proved all grouped M=2..16 rows byte
  exact and retained the installed R2/R4/R8 policy. The terminal M1 KPAR
  challenger was rejected despite a `1.22%` isolated win because changing the
  serial family would reintroduce grouped reduction overhead; the existing
  WIDE serial route remains the economical whole-transaction choice.

## CUDA LLEP Economy

- Fixed-d3 control: Qwen3.6-35B-A3B IQ3_S, stochastic sampling, fixed prompt,
  425 prefill + 256 decode tokens, three iterations after one warmup.
- Stable M=4 grouping removed four launches/layer and moved verifier compute
  nodes `1536 -> 1376` per device. Direct FP32 route publication then removed
  one conversion kernel plus one memcpy/layer: compute nodes are now `1336`,
  captured prefill graphs are 80 total nodes/device smaller, and the native hot
  transaction is `1403` kernels including 67 NCCL/control sidecars.
- Runtime expert publication now uses one deterministic two-level warp/wave
  scan for counts, offsets, and active ranks instead of the former quadratic
  per-expert scan and second publication phase. Across M=2..31 the fused plan is
  `1.14x..1.26x` faster on CUDA and `1.16x..1.72x` faster on ROCm; at M=4 it
  moved `12.16 -> 9.76 us` and `30.28 -> 19.43 us`, respectively.
- The canonical routed-plus-shared transport transaction is `1.075x..1.151x`
  faster than the former three-collective transaction for M=2..16 in the
  isolated CUDA2 NCCL harness. Its fixed-order finalizer moved from `12.86` to
  `9.82 us` on CUDA with a 256-column tile and from `19.84` to `15.36 us` on
  ROCm with a 128-column tile. CUDA uses 30 registers/thread, 1.02 KiB shared
  memory, and zero spills; ROCm uses 24 VGPR, 48 SGPR, 512 bytes LDS, and zero
  scratch/spills. Narrower CUDA and wider ROCm alternatives were measured and
  rejected.
- The matched Release CUDA2 fixed-d3 benchmark validates the complete graph
  win: prefill improved `583.34 -> 654.43 tok/s` (`1.122x`) and decode improved
  `172.99 -> 180.43 tok/s` (`1.043x`). Generated token IDs, generated text
  bytes, and `79.25%` stochastic acceptance are identical to the baseline;
  verifier transaction validation failures remain zero. Both participants
  captured one complete prefill graph (`6387/6347` nodes), replayed it without
  recapture or segmentation, and all decode contexts report full-graph replay
  with deferred event completion. The mirrored LocalTP PerfStats records show
  participant-local outcomes with no outcome collective and no host shadow;
  the only D2H boundary is terminal response materialization.
- The M=4 CUDA fused kernel uses 40 registers/thread and 4.23 KiB shared memory,
  has zero spills, and retains 100% theoretical occupancy. The ROCm wave64
  kernel uses 45 VGPR, 89 SGPR, and 4,232 bytes LDS with zero private segment
  and zero spills. The post-change Integration tree rebuilt cleanly, the
  canonical grouped-verifier gate passed `89/89` in 834.31 seconds, and the
  device-free unit/source gate passed `585/585` in 134.99 seconds.
- Isolated M=4 grouping is `5.99 us`: 38 registers/thread, 4.10 KiB shared,
  zero spills, and 100% theoretical per-SM
  occupancy. Achieved whole-GPU occupancy is intentionally low for this single
  dependency block; splitting it would restore launch/dependency overhead.
- Matched decode improved `156.97 -> 164.93 -> 165.92 tok/s`; acceptance stayed
  `61.57%` with zero validation failures. Prefill is `179.51 tok/s`; decode is
  `3.43%` below the historical `171.81 tok/s` promotion target.
- Direct CUDA router publication is 40 registers, zero spills, 100% theoretical
  and 83% achieved occupancy. ROCm is 18 VGPR with no scratch or spills.
- Cooperative CUDA Top-K 40 cut target/draft distribution to `0.238/0.214 ms`;
  it is spill-free and byte exact through M=16.
- Qwen `N=512,K=2048` grouped projection is byte exact across all 21 CUDA
  formats at M=2..31 and gains `1.69x..2.21x`; its eight-CTA geometry remains a target.
- CUDA GDN is byte exact through M=31; at `d_k=d_v=128` it reaches `17.73 us`,
  541 GB/s, and zero spills.
- This RTX 3090 pair has no peer access; mirrored verification avoids its
  measured `941 us/layer` rooted M=5 NCCL collective.
- The post-fix long-context parity harness remains unsuitable as an economy
  result: CUDA2 CurrentBatchLLEP takes `239.50 s` and ROCm2 takes `455.36 s`,
  versus `31.72 s` for the earlier CUDA2 dynamic-maintenance control. The ROCm
  trace attributes roughly `194 s` to CPU-bound model preparation/loading and
  includes one `131.6 s` mapped-download wait plus per-step snapshot downloads.
  The next profile must separate test-only parity observation from the Release
  production graph before attributing those waits to LLEP itself; neither cost
  is accepted as part of the target device-resident inference transaction.

Matched llama.cpp master comparison, tok/s:

| Backend/model | d1 L/LC | d3 L/LC |
|---|---:|---:|
| CUDA dense 27B | `36.24/59.64` | `64.84/60.91` |
| CUDA MoE 35B | `93.01/153.92` | `159.61/171.81` |
| ROCm dense 27B | `20.08/25.15` | `39.28/22.66` |
| ROCm MoE 35B | `46.33/73.93` | `74.94/95.84` |

### Reproducible CUDA1 SingleDevice Reference

The active CUDA1 control uses the Release binary, a 434-token production-valid
Qwen chat prompt, stochastic fixed-depth-3 verification, one capture/warmup,
and three measured replays. MTP prefill population is inside the measured
prefill transaction; model load, graph construction, and warmup are outside it.

```bash
env \
  LLAMINAR_BENCHMARK_ITERATIONS=3 \
  LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
  LLAMINAR_PREFILL_GRAPH_REQUIRED=1 \
  LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES=1 \
  LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=0 \
  /workspaces/llaminar/build_v2_release/llaminar2 benchmark \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  -d cuda:0 --context-length 4096 -n 256 \
  --benchmark-json-output /tmp/llaminar-cuda1-qwen36-35b-mtp-d3.json \
  --prompt-file /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed_chat.txt \
  --seed 123 --temperature 0.8 --top-k 40 --top-p 0.9 \
  --mtp --mtp-draft-tokens 3 --mtp-depth-policy fixed \
  --mtp-verify-mode speculative-sampling \
  --moe-residency-maintenance off
```

The 2026-08-07 pre-router-tuning baseline was `2271.66 tok/s` prefill
(`191.05 ms`) and `189.60 tok/s` decode. A captured, byte-exact tournament over
every production bucket from M=64 through M=4096 replaced the underfilled
Qwen-35B `64x64` FP32 router tile with a typed bucket policy. At the canonical
M=512 bucket, `32x32` reduced isolated replay from approximately `426 us` to
`267 us`; all candidate outputs matched the installed arithmetic byte for byte.
Two post-install Release runs measured `2336.31/2335.26 tok/s` prefill
(`185.76/185.85 ms`) and `189.52/189.68 tok/s` decode. Both retained exactly
`450/672` stochastic accepts (`66.96%`), zero transaction-validation failures,
and one complete 1611-node captured graph.

Nsight Compute confirms the physical improvement. The former `64x64` node used
40 registers, zero spills, 32 blocks, 16.67% achieved occupancy, and `466.53 us`.
The installed `32x32` node uses 39 registers, zero spills, 4.10 KiB shared
memory, 128 blocks, 25.24% achieved occupancy, and `280.29 us`; measured SM and
memory throughput both reach 42.25%. The next dominant target is grouped IMMA:
gate/up/SwiGLU plus down projection account for approximately `78.45 ms` of the
complete prefill replay, versus roughly `11.2 ms` for the tuned router.

The follow-on all-format grouped-IMMA tournament covered the three concrete
Qwen3.6 35B MoE gate/up and down codebook tuples over every production bucket:
63 cells, 24 physical candidates per cell, and 1,512 authenticated records.
Every candidate matched the serial arithmetic byte for byte. The installed
policy uses paired gate/up projection with 32-column tiles through the smaller
buckets, 64-column tiles where they win at larger M, and 32-column down tiles.
Nsight Compute reports 64 registers, zero spills, and `64.19%/65.42%` achieved
occupancy for the representative M=512/M=1024 winners. The resulting Release
control measures `2445.92 tok/s` prefill (`177.44 ms`) and `189.69 tok/s`
decode with unchanged `450/672` stochastic acceptance (`66.96%`), zero
transaction-validation failures, and one complete captured verifier graph.

The first full-context HTTP gate then exposed a separate graph-family workspace
defect at Q8 `M=256,N=512,K=2048`: exact dispatch is non-monotonic in M, but
planning had assumed the largest bucket represented every smaller bucket. The
M=4096 direct winner therefore declared zero canonical-reducer scratch even
though M=64/128/256 use the 16-partition public-M1 tree. Planning now computes
a typed envelope over every installed cell through the graph family's maximum,
caches it with the complete mutable policy identity, and fails during planning
if the envelope cannot be proven. A focused CUDA integration regression covers
all 16 execution codebooks and every captured prefill bucket; the Q8 case proves
the 32 MiB four-slot allocation and the exact-overlay cache-key transition.

The corrected Release live-server cell passed `19/19`: repeated prefill replay,
three-position and strict-JSON needle recall, 1,024-token stochastic generation,
cache reset, `3827/4096` near-boundary context, oversized rejection, clean exit,
and complete VRAM release. PerfStats captured 2,702 records with zero segmented
execution, 240 paired-IMMA exact-overlay calls, all 40 grouped-verifier layers,
436 depth-three verifier transactions over 1,744 rows, and eight real executions
of the formerly failing Q8 canonical bucket. The only D2H publications were the
14 terminal request results. Peak process VRAM was 21,710 MiB on the 24 GiB
RTX 3090.

The final linked Release binary repeated the same `19/19` full-tier result and
2,702-record PerfStats contract after the fail-fast planner change. The complete
Integration tree rebuilt successfully (`792/792` targets), and the device-free
unit gate passed `591/591` in 139.32 seconds.

The next CUDA1 prefill slice pipelined the causal GDN recurrence instead of
changing its arithmetic. Two 16-byte-vectorized `cp.async` stages overlap row
`t+1` Q/K publication with the exact row-`t` recurrence; the measured launch is
64 threads with a ten-block launch bound. At the production Qwen M=425 shape,
latency moved from `568.893 us` to `449.331 us` (`1.266x`). Nsight reports 96
registers/thread, zero spill requests, 41.67% theoretical and 25.94% achieved
occupancy, 90.91% L2 hit rate, and `8.39` warp cycles per issued instruction.
The D_K=64/D_K=128 M-totality, captured M=3, M=2/M=4 snapshot, runtime-M
publication, and unequal-request continuation tests all remain byte exact. The
matched Release model replay consequently moved `2749.13 -> 2806.76 tok/s`
prefill (`+2.10%`) while decode remained `190.36 tok/s`; stochastic acceptance
remained exactly `450/672` (`66.96%`) with zero transaction-validation failures.

CUDA FA2 has a shared capture-time physical K/V tile policy backed by a
complete 288-cell Release tournament: every released Qwen attention geometry,
TP=1/2/4/8 where the head split is legal, M=64/128/512, and KV horizons from 64
through 131072. `TILE_KV=64` won `0/288`; HD128/HD256 remain within `3.149%`
with the 16-row tile, while HD64 query groups of width three or six select 32.
The installed generic rule passed all 288 cells with `2.481%` p95 and `3.149%`
maximum regret. Its device-free totality gate covers 48/64/100/164 KiB shared
memory ceilings and fail-closed profiling overrides.

The follow-on slice now implements the real byte-exact K/V-context mode and
its fixed-order device merge. One immutable captured transaction contains the
direct query-sequence root and an IF-only context body; the live device-owned
K/V count publishes the CUDA conditional predicate, so neither mode selection
nor intermediate attention state crosses the host. The capture plan owns the
maximum persistent partial-output and `(m,l)` workspace envelope, and invalid
or incomplete geometry fails closed instead of selecting another launch path.

The certified device-adaptive tournament covers 720 model/TP/M/KV domains:
all listed Qwen 2.5 and Qwen 3.5/3.6 geometries, legal TP=1/2/4/8 splits,
HD=64/128/256, M=17/32/64/128, and live K/V lengths
256/512/1024/8192/131072. Mean regret is `0.574%`, p95 is `3.108%`, and maximum
regret is `5.511%`. The five domains above 5% are all Qwen2.5-0.5B, KV=256
launch-floor cases; the worst is `62.935 us` versus `59.648 us`, an absolute
`3.287 us` difference within the explicit `3.5 us` native-conditional budget.
An HD256 PV8 challenger was also measured and retired: at Qwen3.6-35B TP8,
M=64, KV=131072 it regressed context execution by approximately `77%` and
direct execution by approximately `33%` versus PV4.

Nsight Compute on the retained PV4 context kernel reports `4.51 ms`, 128
registers/thread, 43.78 KiB dynamic shared memory, zero local-memory spilling
requests, `24.86%` achieved versus `25%` theoretical occupancy, and 11.93
active warps/SM. Compute and memory-pipe throughput are both `32.91%`; the
remaining limiter is fixed-tree barrier latency rather than spills or an
underfilled grid. The policy unit gate passed `15/15`, and the CUDA attention
integration binary passed `54/54`, including adaptive replay, fixed direct and
context byte equality, every native K/V format, grouped M=2..16, TP boundaries,
and TurboQuant-adjacent cache paths.

The linked Release Qwen3.6-35B CUDA1 server cell then passed `20/20` with
dynamic stochastic MTP depth 1..15 and the full long-context tier. It proved
seeded stochastic prefix replay, repeated captured prefill replay, three needle
positions, strict multi-needle JSON, 1,024-token generation, cache reset,
3,827/4,096 near-boundary context, oversized rejection, clean exit, and full
VRAM release. Its 3,365 PerfStats records prove complete homogeneous CUDA graph
capture with no segmented execution and no forbidden intermediate D2H. The
run also exposed and fixed a harness-only asymmetry: CUDA terminal-ledger
validation had rejected valid mixed greedy/stochastic compact-outcome records,
while the ROCm validator already accepted both audited sources and required at
least one stochastic record. The focused policy regression now enforces that
same fail-closed rule for CUDA.

The post-FA2 fixed-depth-3 Release control measures `2785.83 tok/s` prefill and
`214.25 tok/s` decode with `73.18%` stochastic acceptance and zero transaction
validation failures. The matched no-MTP control is `2805.78 tok/s` prefill, so
population of the shifted MTP KV state costs only `0.71%`; main-model prefill,
not the MTP sidecar, owns the remaining throughput gap.

```bash
env LLAMINAR_E2E_LONG_CONTEXT=1 \
  LLAMINAR_E2E_LONG_CONTEXT_TIER=full \
  LLAMINAR_E2E_CONTEXT_LENGTH=4096 \
  LLAMINAR_E2E_LONG_MAX_TOKENS=1024 \
  LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS=900 \
  LLAMINAR_E2E_PERF_STATS=1 \
  LLAMINAR_E2E_PERF_STATS_GPU_STAGE_TIMING=1 \
  tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_release/llaminar2 \
  --suite '/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf|cuda:0|64|--moe-residency-maintenance off --mtp --mtp-draft-tokens 4 --mtp-min-draft-tokens 1 --mtp-initial-draft-tokens 4 --mtp-max-draft-tokens 15 --mtp-depth-policy dynamic --mtp-depth-window 4 --mtp-depth-min-samples 4 --mtp-depth-promote-windows 1 --mtp-verify-mode speculative-sampling|qwen36-moe-cuda1-dynamic-mtp-long-context|prefill-graph-probe,require-prefill-graph-capture,stochastic-mtp-probe,non-thinking-only'
```

### Reproducible ROCm1 SingleDevice Reference

The active ROCm SingleDevice tuning control uses the Qwen3.6-35B-A3B MoE
model, a byte-stable 434-token Qwen assistant-generation prompt, stochastic
sampling, and MTP depth 3. Llaminar measures three steady-state graph replays
after one warmup; model loading, arena construction, graph capture, and the
warmup are outside the timing sample. MTP prefill is enabled, so each measured
prefill includes population of both the main and MTP KV caches.

`qwen36_mtp_fixed_chat.txt` is the production-valid benchmark prompt. Its
2,511 bytes have SHA-256
`63d628982074c2785953dfde6f327e4f0146162a5c51396a48b1f59a239a97ae` and
contain the exact Qwen user/assistant wrapper generated from the model's GGUF
chat template. The unwrapped `qwen36_mtp_fixed.txt` remains useful as a raw
tokenizer/continuation diagnostic, but the model closes that incomplete turn
with `<|im_end|>` as its first greedy token. Acceptance and throughput from
that malformed framing are not promotion evidence.

```bash
env \
  LLAMINAR_BENCHMARK_ITERATIONS=3 \
  LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
  LLAMINAR_PREFILL_GRAPH_REQUIRED=1 \
  LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES=1 \
  LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=0 \
  /workspaces/llaminar/build_v2_release/llaminar2 benchmark \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  -d rocm:0 --context-length 4096 -n 256 \
  --benchmark-json-output /tmp/llaminar-rocm1-qwen36-35b-mtp-d3.json \
  --prompt-file /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed_chat.txt \
  --seed 123 --temperature 0.8 --top-k 40 --top-p 0.9 \
  --mtp --mtp-draft-tokens 3 --mtp-depth-policy fixed \
  --mtp-verify-mode speculative-sampling \
  --moe-residency-maintenance off
```

The 2026-08-07 post-exact-GDN/grouped-verifier stochastic baseline is
`1301.15 tok/s` prefill and `120.39 tok/s` decode, with `72.27%` draft
acceptance and zero transaction-validation failures. Relative to the active
`1400/125 tok/s` targets, the remaining shortfall is `7.06%` for prefill and
`3.69%` for stochastic decode. On the same prompt at temperature zero,
Llaminar measured `1302.46 tok/s` prefill and `145.36 tok/s` decode with
`82.94%` acceptance.

The temperature-zero correctness matrix compares 256 generated token IDs from
serial and fixed-depth-3 MTP. Llaminar is identical for all 256 tokens on both
the production-valid prompt and the older raw diagnostic prompt. On the same
434-token production-valid input, llama.cpp MTP first diverges from its own
serial decode at generated token 147; it is therefore a useful performance
reference but not a byte-exact grouped-verifier oracle. The two engines' serial
lanes first differ at generated token 26, which is an ordinary cross-engine
kernel-math branch and is independent of Llaminar's internal MTP equivalence.

The focused Release ROCm1 fixed-depth-3 live-server gate passed `19/19`. It
proved repeated captured-prefill replay; beginning, middle, and end needle
recall; strict multi-needle JSON; a non-degenerate 1024-token completion; cache
reset; valid use of `3827/4096` context tokens; oversized-context rejection;
clean shutdown/VRAM release; and `8,405` PerfStats records. The direct grouped
verifier operation-equivalence matrix remains green on both CUDA and ROCm for
M=1..4, the first production transaction/publication, and the M=6
resident-sidecar device target.

The 2026-08-08 ROCm FA2 policy slice installs the same explicit
query-sequence/K/V-context capture-time choice used by CUDA. Its complete
2,040-domain Qwen catalog covers every released dense/MoE attention geometry,
legal TP=1/2/4/8 split, tuned context horizons through 128K, and the production
head-dimension envelope. Mean regret is `0.371%`, maximum regret is `2.663%`,
and no catalog domain exceeds 5%. The independent phase-grid tournament
selected 60 resident phase blocks and retained a `3.387%` maximum regret.
`rocprof`/ISA evidence for
the installed kernels reports zero scratch: HD128 direct and phase use 68
VGPRs, HD256 phase uses 80 VGPRs, and the ordered reducer uses 32 VGPRs.

Timing exploration remains capped at 128K, but correctness and capacity do
not. Device-free CUDA/ROCm policy regressions exhaust every positive capacity
through one million and prove extremal dispatch totality through `INT_MAX`.
Real captured 256K transactions then prove CUDA direct/context byte equality
for both compiled native K/V types (FP16 and FP32), and ROCm direct/context
equality for FP32, FP16, BF16, and Q8_1 K/V storage. This exposed a
non-monotonic arena-sizing defect:
the largest M selected direct query execution and declared no partial
workspace, while an intermediate M selected context execution and needed the
family maximum. The ROCm launch policy now computes the complete family
workspace envelope before capture; runtime rebinding or allocation is not used.

The linked Release Qwen3.6-35B ROCm1 server cell subsequently passed `19/19`
again at fixed MTP depth 3 and the full long-context tier, producing `3,842`
PerfStats records with clean shutdown and complete VRAM release. Every FA2
capture requested `geometry_selected`: M=256 selected the actual context graph
with 60 phase blocks and 1,024 reducer blocks, while M=1,536/2,048/4,096 selected
query-sequence execution. The server gate now rejects a legacy query-only ROCm
request policy or malformed/missing per-backend FA2 capture evidence.
The complete Integration target set rebuilt cleanly, and the final device-free
unit/source-policy checkpoint passed `593/593` after synchronizing the stale
CUDA MoE router boundary regression with its installed measured overlay.

The 2026-08-09 CPU FA2 slice replaced cache-percentage magic numbers with a
typed, empirically certified K/V tile policy. Code-generation ISA and runtime
dispatch ISA are independent policy axes: native AVX2 passed 81 domains at
`2.859%` p95 regret (`4.120%` maximum), native AVX-512 passed at `3.371%`
(`5.124%` maximum), and an AVX-512 build forced through AVX2 runtime dispatch
passed at `2.733%` (`10.079%` maximum). The mixed-profile maximum is one
Q16_1, HD128, M=1 domain; it remains explicit evidence rather than being
hidden by an alias to the native AVX2 policy. CPU certification runs must be
isolated: simultaneous socket-wide tournaments measurably perturb shared
power and memory behavior and are not valid policy evidence.

The production byte-totality gate independently covers 34 distinct
participant geometries induced by every listed Qwen dense/MoE geometry and
legal TP=1/2/4/8 split, all nine native K/V formats, decode M=1, grouped
M=2/4/8/15, and prefill M=32/128/257 over a 512-row prefix. Every one of the
2,448 domains executes all seven compiled physical tiles through the optimized
kernel, for 17,136 candidate executions, and is byte identical to the fixed
canonical arithmetic. The complete gate passed in `326.68 s`; the focused
policy, Q16, tournament-driver, and production-kernel checks pass in `1.46 s`.

### CPU Qwen3.6-35B MoE Release Matrix

The 2026-08-09 control uses the 434-token canonical chat prompt
(`sha256:63d628982074c2785953dfde6f327e4f0146162a5c51396a48b1f59a239a97ae`),
256 stochastic output tokens, seed 123, temperature 0.8, top-k 40, top-p 0.9,
one warmup, and three measured iterations. Single-socket rows use 28 physical
cores. Dual-socket rows use two MPI ranks pinned one per socket with 28 physical
cores each. PSS is the observed warm process-tree value, not model-file size.

| ISA | Topology | MTP | Prefill tok/s | Decode tok/s | Acceptance | MTP decode / baseline | PSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| AVX-512 | 1 socket | off | 110.84 | 15.69 | - | 1.000x | 42,239 |
| AVX-512 | 1 socket | fixed d3 | 108.77 | 10.71 | 79.91% | 0.683x | 43,680 |
| AVX-512 | 2 sockets | off | 32.07 | 20.09 | - | 1.000x | 44,904 |
| AVX-512 | 2 sockets | fixed d3 | 31.81 | 14.77 | 78.77% | 0.735x | 46,355 |
| AVX2 | 1 socket | off | 71.21 | 14.28 | - | 1.000x | 42,640 |
| AVX2 | 1 socket | fixed d3 | 69.90 | 9.08 | 74.09% | 0.636x | 44,084 |
| AVX2 | 2 sockets | off | 27.05 | 19.83 | - | 1.000x | 45,585 |
| AVX2 | 2 sockets | fixed d3 | 26.92 | 14.36 | 79.81% | 0.724x | 46,433 |

All eight cells completed with zero transaction rollbacks and zero transaction
validation failures. AVX-512 improves single-socket prefill by `1.557x` and
decode by `1.099x` over AVX2. Dual-socket baseline decode scales by `1.281x`
on AVX-512 and `1.389x` on AVX2, but dual-socket prefill regresses to only
`0.289x` and `0.380x` of one socket. MTP is also unequivocally uneconomical in
this control despite high proposal acceptance; its loss is verifier/draft
execution cost, not rejection or rollback pathology.

The first structural dual-prefill defect is explicit in the collective code.
Same-node CPU TP selects `ShmemSpinBackend`, whose native allreduce is capped at
8,192 elements. A 434-row by 2,048-hidden activation has 888,832 elements, so
large prefill allreduces leave the shared-memory implementation and enter its
MPI fallback, while decode-sized 2,048-element reductions remain native. This
matches the measured phase split but still requires per-stage timing before
attributing the entire deficit. The fix gate is an economical, total native
same-node allreduce for all positive payload sizes; retaining the fallback is
not an acceptable final architecture.

Both AVX-512 and AVX2 Release binaries then passed the full 4,096-context CPU2
MoE server tier `20/20`, including 2,048-token generation, boundary handling,
RAM prefix restore with MTP state, and CPU FA2 policy PerfStats. The complete
Integration target set rebuilt cleanly, and the device-free unit/source gate
passed `592/592` in `159.17 s`. Runtime ISA reporting now reads
`activeISALevel()` rather than claiming every CPU run is AVX-512, and the
canonical matrix wrapper admits CPU2 MoE through the UPI same-node policy
instead of rejecting the lane or forcing MPI-only transport.

The llama.cpp CLI command below intentionally receives the unwrapped source
file because `--conversation` applies the GGUF chat template itself. Its
effective 434-token prompt is byte-equivalent to Llaminar's checked-in
`qwen36_mtp_fixed_chat.txt`; passing the preformatted file here would wrap it
twice.

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  /workspaces/llama.cpp-reference/build-rocm/bin/llama-cli \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 3 \
  -f /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed.txt \
  -n 256 -c 4096 -ngl all --split-mode none --main-gpu 0 --fit off \
  --flash-attn on --seed 123 --temp 0.8 --top-k 40 --top-p 0.9 --min-p 0 \
  --repeat-penalty 1 --dry-multiplier 0 --perf \
  --conversation --single-turn --no-display-prompt
```

On current llama.cpp master `f9e832c10e94`, three runs measured prefill at
`408.7`, `405.3`, and `403.0 tok/s`, and generation at `78.4`, `77.7`, and
`78.4 tok/s`. The medians are therefore `405.3 tok/s` prefill and
`78.4 tok/s` generation.

### Reproducible Llaminar CUDA2 LLEP Reference

This is the canonical clean-throughput command for the explicit apportioned
LLEP policy tuple. It uses stochastic fixed depth 3 and the same prompt and
sampling controls as the llama.cpp reference below. The 425-token prompt is
below the production `M * top_k >= 8192` LLEP-prefill boundary, so this row
isolates grouped decode and two-device communication economy; use a longer
prompt to measure least-loaded current-batch expert movement itself.

```bash
env \
  LLAMINAR_BENCHMARK_ITERATIONS=3 \
  LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
  LLAMINAR_PREFILL_GRAPH_REQUIRED=1 \
  LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES=1 \
  LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=0 \
  /workspaces/llaminar/build_v2_release/llaminar2 benchmark \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --context-length 4096 -n 256 \
  --benchmark-json-output /tmp/llaminar-cuda2-llep-d3-clean.json \
  --prompt-file /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed.txt \
  --seed 123 --temperature 0.8 --top-k 40 --top-p 0.9 \
  --mtp --mtp-draft-tokens 3 --mtp-depth-policy fixed \
  --mtp-verify-mode speculative-sampling \
  --mtp-terminal-head-policy mirrored-full-vocabulary \
  --moe-release-raw-expert-weights \
  --moe-residency-maintenance off --moe-hot-expert-cache off \
  --moe-routed-expert-placement tiered-overlay \
  --moe-routed-expert-continuation-domain qwen36_moe_cuda_hot \
  --moe-routed-expert-base-model-domain qwen36_moe_cuda_hot \
  --moe-routed-expert-shared-domain qwen36_moe_cuda_hot \
  --moe-routed-expert-residency static-by-id \
  --moe-continuation-dense-policy tensor-parallel \
  --moe-routed-expert-domain \
    'qwen36_moe_cuda_hot=cuda:0,cuda:1;scope=local;backend=nccl;routed_compute=apportioned;routed_phase=uniform;routed_decode_assignment=static-owner;routed_prefill_assignment=least-loaded-resident;owner=0' \
  --moe-routed-expert-tier \
    'hot@qwen36_moe_cuda_hot;priority=0;max-experts-per-layer=256;memory-mb=8192'
```

The 2026-08-05 post-correctness refresh measured `682.13 tok/s` prefill and
`124.07 tok/s` decode, with `52.89%` stochastic draft acceptance and zero
transaction-validation failures. Both devices captured and replayed one full
prefill graph (`5947/5907` nodes). This is the active economy baseline; the
older `180.43 tok/s` row had `79.25%` acceptance and must not be treated as
current until the acceptance regression is explained.

### Reproducible llama.cpp CUDA Reference

```bash
CUDA_VISIBLE_DEVICES=0 \
  /workspaces/llama.cpp-reference/build-cuda/bin/llama-cli \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 3 \
  -f /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed.txt \
  -n 256 -c 4096 -ngl all --split-mode none --main-gpu 0 --fit off \
  --flash-attn on --seed 123 --temp 0.8 --top-k 40 --top-p 0.9 --min-p 0 \
  --repeat-penalty 1 --dry-multiplier 0 --perf \
  --conversation --single-turn --no-display-prompt
```

The CUDA reference was rebuilt from current llama.cpp master `f9e832c10e94`
before measurement. Three serial runs produced
`1488.2/1411.4/1460.4 tok/s` prefill, for a `1460.4 tok/s` median. Three
fixed-depth-3 MTP runs produced `1250.5/1290.8/1304.8 tok/s`, for a
`1290.8 tok/s` median; llama.cpp therefore pays an `11.61%` MTP prefill tax on
this control. Llaminar is `1.92x` faster than the serial control and `2.16x`
faster than the MTP control while paying only a `0.71%` MTP prefill tax.
Promotion retains the stricter historical `171.81 tok/s` decode target and
requires at least `2245 tok/s` prefill.

### AIME25 HTTP Math And Multi-Turn Gate

`scripts/benchmarks/aime25_http_benchmark.py` owns a Release HTTP server and
runs the pinned 30-problem `math-ai/aime25` corpus as two requests per problem:
an initial solution followed by an independent review carrying the complete
message history. The runner authenticates the dataset, model, executable,
prompts, sampling policy, and server arguments; proves history with a nonce
canary; and durably checkpoints each first turn and reviewed result.

The first ROCm1 live probe passed server startup, model loading, and the
two-request history canary. Problem 0's first turn completed, but its review
exposed a production chunked-prefill admission defect: a 4096-row chunk followed
by a 156-row terminal tail was admitted into the fixed 4096 graph bucket, then
`ForwardExecutionEngine` incorrectly reapplied the 256-row raw-prompt minimum to
the already admitted tail. The staged fix makes positive `bucket_seq_len` an
explicit scheduler-admission contract. Focused device-free regressions, the
shared CUDA/ROCm 256K+1 graph-cache test, and both complete 12-case backend graph
cache suites are green. Each backend captures one 4096-row graph, replays the 64
full chunks, and admits the final one-row tail without eager execution or
recapture.

The follow-on audit found that the same minimum also made ordinary short GPU
prompts permanently eager. The staged structural fix redefines it as a minimum
*physical padded bucket* for raw prompts: short prompts are coalesced into that
bucket and remain graph-cache owned, exact smaller graphs remain capturable,
and scheduler-admitted tails retain their fixed transaction bucket. The old
parity-only graph-disable escape hatch is removed. The AIME runner now stops the
owned server before publishing success and authenticates PerfStats evidence for
prefill and decode capture plus replay, zero segmented/manual graph execution,
and zero intermediate D2H. The workflow is backend-neutral across `cpu:N`,
`cuda:N`, and `rocm:N`; GPU-only execution invariants are required for CUDA and
ROCm without imposing them on CPU. Its 15 device-free workflow regressions and
registered CTest gate are green. The full Release tree rebuilt cleanly; the
fresh 30-problem ROCm run is the remaining live gate.

The exact replay canary also certifies shifted MTP KV restore rather than only
ordinary prompt KV reuse: owned-server arguments force depth-three MTP plus RAM
prefix storage, the replay must preserve deterministic response bytes, and the
completed PerfStats ledger must contain a restore tagged
`includes_mtp_state=true`. The underlying partial-terminal restore path is now
green on CPU, CUDA, and ROCm as described in the correctness ledger above.

The CUDA dense-prefill corpus is complete across all 21 source formats:
`33,075/33,075` authenticated cells produced 25,200 exact runtime overlays from
Release scorer evidence. The installed generated include compiled in both the
complete Integration and Release trees. Mixed-format MoE overlay collection
follows the AIME live proof.

## Next Gates

1. Profile the complete canonical CUDA2 d3 graph with `nsys`, then use `ncu` on
   its hottest non-collective kernels. Attribute NCCL primitives, verifier
   compute, prefill expert movement, and tiny control launches separately;
   retain full graph capture and exact output bytes throughout.
2. Close the prefill gap from `654.43 tok/s` to at least llama.cpp's current
   `1122.5 tok/s` median, then pursue the `2245 tok/s` two-times target with
   longer prompts that expose LLEP's theoretical crossover.
3. Keep native geometry for large kernels and fuse adjacent tiny/control work.
   `V2_Perf_CUDAPersistentVerifierGeometry` found a one-kernel CTA proxy `1.25x`
   faster, but full-lane/ALU8 proxies only `0.964x/0.952x`; ncu attributes
   `63-68%` of issue stalls to cooperative barriers. Zero spills.
4. Tune dynamic depth/hysteresis through depth 15 under deterministic prompt
   scenarios without regressing the fixed-d3 `180.43 tok/s` reference.
5. Repeat the economy pass for ROCm and the remaining SingleDevice/EP lanes.
6. After ROCm SingleDevice MTP is green, replace the current MTP KV-only
   prefill lowering with a dedicated captured K/V-only stage. Preserve the
   exact hidden/embedding norms, MTP projection, attention-input norm, K/V
   projection, K norm/RoPE, and cache-format append, while eliminating Q
   projection/gating, Q split/norm/RoPE, Q scratch, and unshiftable or padded
   row work. Promotion requires byte-identical MTP KV payloads across cache
   formats and prefix restore, plus CUDA/ROCm profiler evidence for occupancy,
   registers/VGPRs, zero spills, and end-to-end prefill gain.
