# Grouped prefill operation qualification

## Serving boundary

The canonical-route component tournament measures a useful public operation:
grouping, expert gate/up, SwiGLU, down publication into original route rows, and
a separate ordered fold. Its result does not establish ordinary single-device
prefill performance. Ordinary Qwen prefill normally consumes the router's
published Q8 hidden rows and folds directly into token outputs.

`QwenGraph` selects direct ordered folding for ordinary prefill and binds the
shared router publication. `ROCmMoEKernel::executeGroupedPrefillProjection`
consumes that publication when its source, device, geometry and capture ownership
match. Selected-route and split-projection operations have distinct contracts.
The broad `Unmeasured` operation value must never act as an Automatic admission
key: it also describes other publication and projection combinations.

## Component proof

The ordinary-operation fixture must retain a real router and its Q8 producer,
separate router output tensors, the expert kernel, prepared weights, workspace,
device live-count scalar, and complete captured operation. Controlled expert
route inputs remain explicit injected data; they do not claim to be the
router-selected distribution. Every timed transaction includes the real routing
operation, controlled expert grouping, and direct token publication. Qwen
ordinary-operation qualification uses the all-local shard1 cases; shard2/4
controls intentionally discard routes and cannot represent ordinary single-device
work economics.

The actual imported timing parent must prove:

- Every repeated router producer reads the intended FP32 input and writes the
  published Q8/scales addresses using the retained device count.
- Native gate/up nodes consume those exact addresses with original-token
  indexing and the grouping authority's index pointer.
- No separate expert hidden-row quantizer was recorded.
- The router-reuse submission counter increases during capture and stays fixed
  during replay. It describes host submissions, not device branch executions.
- Each candidate repairs poisoned active Q8/scales and output bytes. Inactive Q8
  tails remain poisoned because the producer does not own those rows.
- Omitting all repeated Q8 producers breaks the output; restoring them recovers
  exact bytes without another candidate repairing the banks.
- For mixed expert counts, independently omitting the ordered and native down
  branches breaks token output. Existing canonical-route tests additionally
  prove the surviving experts' individual contribution bytes.

Cross full, empty, partial, row-floor and restored live work in retained graphs.
Inspect each timing sample before launching another candidate. Keep source-row
serial oracles and all-codebook canonical replay alongside this operation proof.

## Automatic selection

Qualify ordinary-operation timings before installing an Automatic entry. Use
one immutable key containing the full-width Complete/TokenRows/AllRoutes
operation, physical bucket, H/I/E/top-k geometry, all four physical/source masks,
original-token versus route-slot addressing, and published-Q8 versus internal
quantization. Keep physical IQ4 variants and source identities distinct. Resolve
the key once against cached native architecture/resource capabilities.

Live expert counts remain device-owned. Existing whole-expert row admission
selects ordered or native arithmetic from the same count in both projections;
no host histogram, count download or recapture is needed.

PerfStats must consume a caller-owned successful-submission result from the
low-level pipeline. Return the actual selected engine, row floor, role geometry
and admission after all submissions succeed. Do not independently reselect a
legacy tile in the reporting layer or infer the executed branch from capacity.

## Current status

The ordinary-operation fixture and its focused all-codebook/Qwen replay
registrations are implemented. Normal Release/Integration builds and both focused
replay registrations pass (24 native cases,51.87s). The ordinary Release timing
campaign is complete: five cases ×17 device-live points ×six policies ×31 samples
=15,810 qualified timing rows. The roster is Dot4, pure N32/N64, adaptive
N32Floor16/N64Floor16, and WideFloor16. Every sample times eight complete captured
transactions with internal event boundaries, then observes output and published
Q8/scales before another candidate can repair them.

The five cases are all-local Qwen shard1 uniform, hotset and power-law at
H2048/I512/E256/top-k8/capacity512, plus uniform compact Q4_0 and IQ2_S at
H512/I256/E8/top-k8/capacity512. Route IDs and weights are explicit injected
fixture inputs; the actual production router writes separate route outputs and
publishes the real Q8 bank consumed by the expert pipeline. These timings include
router work and direct TokenRows publication, not canonical-route folding or a
real model's chosen routing distribution.

At full512 live rows, speedup over the same operation with explicit Dot4 is:

| Case | Fastest measured policy | Speedup |
|---|---|---:|
| Qwen uniform | N32 | 1.20750× |
| Qwen hotset | N32 | 1.15204× |
| Qwen power-law | WideFloor16 | 1.05358× |
| Compact Q4_0 | N32 | 1.05295× |
| Compact IQ2_S | N32 | 1.04256× |

Sparse work still favors Dot4; pure N32 loses to Dot4 at full Qwen power-law
(0.97897×). There is no universal winning candidate. Population-based selection
is a design proposal, not an installed or timed selector; compact mixed-count
work and unsampled transition populations must not acquire performance claims.

Evidence is
`/tmp/llaminar-gfx1100-wmma-prefill-evidence/grouped-serving-timings/campaign.json`,
producer SHA256
`90dfa911df343259ac4ab22783bb91eeeda7368c1958d84b8e33ac46549e9610`.
The campaign records complete canonical timing and restored AUTO power. This
ordinary-operation corpus is separate from the canonical-route shard2 controls.
The rejected bounded511-worker experiment is archived and the previously
profiled fine-grid fence kernel is restored. Grouped Automatic remains Dot4.
The narrower128-column gate/up experiment is complete but unpromoted: both
full-target builds,72-variant ISA/resource inspection,186 native cases and15,810
timing rows pass. Normalized wide gains change only0.9988–1.0138× across the five
cases and leave the three-family requirement intact. No runtime hardware-counter
qualification is claimed; tooling and the88-file archive are retained in the
[detailed investigation](GFX1100_GROUPED_WMMA_ADMISSION.md).
The seven experimental source files are restored exactly to `f6e727942`; the
profiled GU256 fence remains the baseline. Working binaries still contain GU128
until rebuilt and cannot certify that restored source. Device-population
Automatic selection and its route-owned/count-threshold lifecycle gates are
next, still unimplemented; no admission row or Automatic installation is claimed.
