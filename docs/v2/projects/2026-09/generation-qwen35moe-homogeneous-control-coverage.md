# Qwen3.5 MoE homogeneous overlay controls — 2026-09-11

Eight previously unseen generation controls pass on their first attempts in
`parity-results/generation-qwen35moe-homogeneous-unseen-01/`. The real model is
`Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`, with FP32 activations, FP16 KV and MTP off.
The canonical definitions expand two-socket CPU NodeTP and two-ROCm local TP
over Static/Dynamic and Ordinal/Random placement. This document records results;
it is not another matrix definition or an approved token baseline.

The sequential, fail-fast batch completes in **1,211.572 seconds**. All eight
cells complete the four seed-4242, 384-token fresh/full/partial/full probes and
all eight HTTP-harness checks. The saved observations retain exact prompt and
completion token IDs, terminal hybrid-prefix outcomes, complete runtime
configuration, server logs and terminal PerfStats.

## Timing evidence

| Topology | Policy | Initial placement | Whole cell (s) | Four HTTP requests (s) |
|---|---|---|---:|---:|
| Two-socket CPU NodeTP | Static | Ordinal | 165.681 | 151.919 |
| Two-socket CPU NodeTP | Dynamic | Ordinal | 188.580 | 172.554 |
| Two-socket CPU NodeTP | Static | Random | 165.480 | 150.749 |
| Two-socket CPU NodeTP | Dynamic | Random | 187.336 | 170.337 |
| Two-ROCm local TP | Static | Ordinal | 86.298 | 47.623 |
| Two-ROCm local TP | Dynamic | Ordinal | 163.261 | 123.721 |
| Two-ROCm local TP | Static | Random | 88.401 | 47.547 |
| Two-ROCm local TP | Dynamic | Random | 165.891 | 124.023 |

These are correctness-workload timings, not a production benchmark certificate.
The existing Dynamic parity profile uses a one-token maintenance window and
minimum period, permissive movement floors and a 65,536-token payoff horizon.
That policy was not overridden during these runs. Dynamic is slower than Static
over this short observed horizon, especially on ROCm; do not characterize the
green functional result as an economy win. Attribute and tune the cost in the
subsequent benchmark work without weakening movement or token evidence.

## Placement, movement and token invariance

Setup `moe_placement/routed_expert_weight_selection` records distinguish actual
contiguous ordinal loading from noncontiguous random loading on both backends.
The GPU initial-bank records also report `frozen_owner_map_match=true` for the
two participants and forty main-model layers.

Static records contain no committed migration/arrival/physical-byte evidence.
Dynamic CPU records contain committed migration edges and nonzero remote
projection payload bytes explicitly tagged `purpose=placement_change`; the
transport-profile copies are excluded from that byte observation. Dynamic
ROCm uses the homogeneous device placement completion diagnostics under
`moe_rebalance`: applied arrivals and useful payload bytes are nonzero. Both
ROCm Dynamic runs report 52,436,992 useful payload bytes and 16 accumulated
wave-applied arrivals. These are diagnostic counter values, not a reconstructed
unique movement ledger; counters must not become inference or certification
state authority.

After the live cells complete, read-only comparisons of their persisted
`generation/observations.json` files use the same `observation_traces()` and
`compare_tokens()` helpers as the generation driver. Relative to Static/Ordinal
**within each topology/backend**, all four request streams match exactly for
Dynamic/Ordinal, Static/Random and Dynamic/Random. This independently checks
policy/placement token invariance without another inference run. It does not
assert bitwise identity across CPU and ROCm arithmetic, bound KL error, or
replace the independent HF/reference proof.

All full and partial probes prove main-model hybrid recurrent-state restore
in their public terminal summaries. CPU Dynamic/Ordinal, for example, advances
observed placement epochs between requests while retaining valid full and
partial restores; its four token streams still equal Static's. Both GPU Static
and Dynamic paths pass native graph/transport validation and release VRAM on
clean shutdown.

## Gate reuse and certification boundary

The entire batch reuses
`hybrid-q8-prefix-prerequisites-01/prerequisites.json`: 647 Unit and 137
ProductionParityPreflight registrations for the unchanged build. Recorded
repeated prerequisite cost is zero. The sealed persistent tmpfs entry is reused
with zero copied model bytes. No production edit, rebuild, reconfiguration,
prompt change, precision change or retry is needed for this slice.

These remain unapproved controls. Generation's complete per-request movement
ledger/provenance integration, remaining controls and MTP comparisons, full
numerical evidence, E2E suites, economy gates and both ISA Docker certificates
are not established by these eight passes. At this checkpoint, the canonical
175-control inventory has 87 distinct recorded attempts: 72 latest observations
green, 15 older reds retained and 88 still unseen. Continue unseen acquisition
before rerunning expensive complete campaigns.
