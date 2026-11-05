# QFN moving CPU shard MTP: focused checkpoint

Base: `a4c2b2fc8b6fc7d3dbce31d6daaa275af75274a7` on
`feat/qfn-ngram-continuation`. This slice retains the original model, precision,
topology, reference inputs, byte contracts and movement policy.

## Defects and resulting behavior

Dynamic movement can retire the initial CPU expert bank before the first full
MTP predictor graph is constructed. Graph construction previously tried to
register and borrow that retired bank again. Graph declarations now retain the
original immutable source and workspace description. Each execution packet
leases its actual epoch bank, and both normal and exceptional exits clear
transient engine references before releasing that lease.

Cold admission binds the actual original ModelContext and publishes its prepared
CPU sources after weight preparation and before either graph consumer starts.
This preserves strict rejection of attempts to resurrect a retired initial
bank. The distinct CompleteAtConstruction contract remains unchanged.

A shared serial arena has a representative physical layout covering multiple
layers. It is not a layer's weight source. The stage validates exact ordered
participants, rank/NUMA identity and every shard projection against that layout;
the original per-layer family and packet retain weight-source authority. The
regression prepares two distinct GGUF layers and checks actual execution after
bank retirement, including different source values that expose accidental use
of the representative layer's weights.

The decode command now reports its original exception before coordinated abort
guards unwind. Canonical moving fixed1/fixed3/dynamic-depth tests give each
primary, adaptive and acceptance comparison its own finite placement scope and
serial oracle. Movement remains enabled between comparisons.

## Focused validation

The isolated candidate is built normally with CMake, Ninja and the spill guard
for CUDA SM80/86/89/90 and ROCm gfx906/gfx1100. Its source is the scoped patch on
the published base. It does not use substituted object files.

Seven registered host checks pass without skips, including the complete owning
registry and residency-authority suites, cold/late source lifecycles and the
two-layer geometry regression. Original implementation controls reproduce the
lost diagnostic and invalid layout-identity checks. Focused new regressions are
explicitly registered in ProductionTestPreflight.

Actual model topology: one MI50 continuation, two tensor-sharded CPU expert
members, one MPI rank per physical socket and 28 physical workers per rank.
The original four-part Qwen3.8-Flash-Next UD-Q4_K_XL GGUF and shared Q8 predictor
are unchanged; ngrams reside in RAM. Expert placement is DynamicOrdinal and
retained MTP draft capacity is three.

| MTP mode | Native result | Exact verifier rows | Published movement waves / edges |
|---|---|---:|---:|
| Fixed 1 | PASS, focused610 | 2,718 | 21 / 210 |
| Fixed 3 | PASS, focused611 | 5,285 | 22 / 220 |
| Dynamic depth | PASS, focused613 | 4,530 | 23 / 230 |

All three model cases include actual accepted speculative work, exact
serial tokens, canonical movement-ledger validation, physical transfer/execution
joins and complete two-rank retirement. Drivers are clean. Fixed1, fixed3 and
dynamic depth took 240.99, 259.02 and 264.49 seconds including setup and diagnostic
snapshots; these are not inference throughput measurements.

All six final captured CPUJoint prefill/decode/verifier checks pass on CUDA and
ROCm against this exact isolated candidate, without skips or driver findings.
The dynamic model case records actual depths three and two, three controller
updates, exact adaptive/acceptance tokens and same-frontier state. Its complete
layer matrix covers four verifier rows; the depth-two adaptive comparison checks
final tokens/state without a separate three-row layer matrix.

## Evidence and limits

- Isolated build and host checks:
  `/tmp/qfn-moving-mtp-checkpoint-r592/`, especially `r598-supplement`,
  `r602-supplement` and `focused-r602-checkpoint/summary.json`.
- Model receipts and all canonical CSVs:
  `/tmp/qfn-cpu-joint-qualified-r224/focused610-checkpoint-mtp-fixed1`,
  `focused611-checkpoint-mtp-fixed3`, `focused613-checkpoint-mtp-dynamic`, and the
  corresponding `*-mtp-artifacts` directories.
- Final CUDA/ROCm captured checks:
  `/tmp/qfn-cpu-joint-qualified-r224/focused615-checkpoint-native-cpu-joint`.
- Independent canonical audits:
  `/tmp/qfn-moving-mtp-fixed1-audit-r612/`,
  `/tmp/qfn-moving-mtp-fixed3-audit-r614/` and
  `/tmp/qfn-moving-mtp-dynamic-audit-r616/`.
- Cold and geometry regressions and original failing controls:
  `/tmp/qfn-cold-cpu-joint-admission-r598/` and
  `/tmp/qfn-serial-cpu-source-geometry-r602/`.

The logs show finite full/chained predictor materialization followed by reuse.
Detailed per-binding graph cache counters were not exported, so this does not
claim complete cold graph preparation or a performance improvement. The earlier
MTP-off numerical checkpoint remains documented in
[its evidence note](QFN_CPU_JOINT_DYNAMIC_FOCUSED_CHECKPOINT_20261105.md).

Owned prefix-cache payloads are removed after successful process retirement;
CSV, protocol, resource and driver evidence is retained. No cache payload is
hashed or read to perform that cleanup.

Full preflight is deliberately not run for this iteration. This is a focused
functional checkpoint, not full model-matrix, numerical-corpus, image or release
certification. Pending ngram transport/model work remains a separate slice.
