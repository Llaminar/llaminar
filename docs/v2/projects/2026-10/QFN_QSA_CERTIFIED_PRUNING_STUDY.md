# QFN certified sparse-index pruning study

Date: 2026-10-24. Source checkpoint: `d34ac0796`.

## Scope and decision

This evaluates the supplied candidate-reuse proposal and the subsequent
page-ball bound against learned QFN operands. No reuse state,
page metadata, selection kernel, or production policy has been installed.
The completed 4K, 16K, 64K and 128K pilots find no economical configuration
among the tested global-radius temporal and contiguous-page ball certificates.
The stronger mathematical bound is useful, but its remaining uncertainty is
too large on these inputs. Retain the evidence; do not install these candidates.
The [completed-study manifest and frozen private sources](../../../../parity-results/qwen-flash-next/qsa192-193-study-source-r1/manifest.json)
retain the exact capture, evaluation, adversarial tests and arithmetic-boundary
reproduction. Tensor payloads were neither duplicated nor hashed for that archive.
The qualified four-MI50 TP4 result remains **820.777 prefill / 41.562 decode
tok/s**, against Castagna's **42.959 decode tok/s** target.

The accepted 512-prompt/256-output throughput workload does not rank sparse
blocks: QSA begins ranking at 2,052 visible tokens. These ideas therefore target
long-context indexing, not a direct improvement to that short-context score.
The [tuning dashboard](QFN_TUNING_DASHBOARD.md) retains the main target.

## Two bounds

For the four prepared query heads, define `a_A = sum(q_h for h in A)`, including
the empty subset. The identity

\[
\sum_{h=1}^{4}\operatorname{ReLU}(q_h^T k)
 = \max_{A\subseteq\{1,2,3,4\}}a_A^T k
\]

gives a temporal drift bound

\[
\Delta = c R_0 \max_A\left\|\sum_{h\in A}(q_h-q_{0,h})\right\|_2,
\]

where `c` is the exact stored FP32 score-scale literal. This is no larger over
the reals than `c R_0 sum_h ||q_h-q_0h||`. It preserves cancellation between
heads. Only sixteen subset norms are involved; a maximum of squared norms
followed by one square root is sufficient. Each reuse attempt retains the
original anchor and includes every newly completed four-token block.

The related page bound for keys in a ball of center `z` and radius `r` is

\[
U_{\mathrm{page}}=c\max_A(a_A^Tz+r\|a_A\|_2).
\]

A page may be skipped only when its upper bound is strictly below the current
512th score among keys already evaluated. Equality is retained, preserving the
original global-ID tie rule. The four head-center dot products can be shared
when forming the sixteen subset-center products. Query subset norms are shared
across pages.

These are real-arithmetic statements. A production certificate additionally
needs outward geometry/norm bounds and an allowance for the actual FP32 score
kernel, reduction order, scaling and FTZ behavior. Float64 evaluation and exact
ID observations alone do not establish that certificate. The cited Lean file
`/tmp/qwen-ops-audit-20261024/sparse/SparseCertificate.lean` is absent from this
workspace; no compiled Lean proof is claimed here.

## Measurement contract

The native observer uses the existing Integration runner, model tokenizer,
admitted snapshot BOM, and captured forced-token parity transaction on four
MI50s. It takes real source-text prefixes at 4K, 16K, 64K and 128K, then 33
following source tokens: the first publishes a pending token and the remaining calls provide
32 actual decode rows. All twelve sparse layers are observed. This is teacher
forcing, not natural generation, a model-accuracy gate, or a timing benchmark.

For every row the observer exports the actual post-normalization/post-RoPE
FP32 index queries, completed pooled keys and selected token IDs. The canonical
INT32 selection checkpoint is exposed as FP32 diagnostic values; the observer
checks the exact integer range, integrality, causal order and sentinel boundary
before writing int64 files. Query metadata authenticates the original row.
One logical replica is exported; this does not certify all-rank value equality.

The offline analysis checks every historical key prefix byte-for-byte before
reusing the final key table. It compares native selections to an independent
float64 full scan, reporting any disagreement. Both temporal bounds run on the
same operands and are also compared at identical anchors. Independent evolving
histories are labeled separately. Failed attempts charge their candidate scoring
and the subsequent complete scan. Vacuous successes that retain all keys cannot
count as useful pruning. Norms, selection, gathering and control costs remain
additional costs, not implied speedups.

The page experiment follows the temporal experiment on the same observations.
It uses contiguous completed pages and scores the live open page. Its threshold
must come from actually scored keys, with seeding included in the work count.
Metadata construction, query bounds and page ordering must be reported even
when the historical-key scoring count improves.

## Collection diagnostics

The first private capture attempted ordinary complete-generation snapshots.
That path executes retained D2D snapshot nodes but does not publish them through
the host snapshot getter at completion. The failed receipt is retained under
`qsa192-learned-python-code-c4096-r1`; this diagnostic publication gap is pending,
and no inference-math defect is inferred from it. The explicit captured parity
transaction already has the required publication boundary.

The second private capture reached actual query/key observations but rejected
the selection checkpoint because the reader incorrectly required a typed INT64
publication. Its retained `r2` receipt documents the reader failure. The revised
reader explicitly follows the canonical INT32-as-FP32 checkpoint contract; it
does not change the model or production snapshot API.

## Results

### Temporal reuse: 4K Python source

The [native capture](../../../../parity-results/qwen-flash-next/qsa192-learned-python-code-c4096-r3/result.json)
passes with 32 rows from all twelve layers, normal process retirement and a
clean driver window. Every observed historical key prefix is byte-identical
to the corresponding final prefix. All **384/384** native selected-ID sets
match the independent float64 full-score oracle in the
[analysis](../../../../parity-results/qwen-flash-next/qsa192-study-python-code-c4096-r1/result.json).

The stronger bound is substantially tighter: its ratio to the original bound
has a median of layer medians of **0.48136** at the same anchors. Nevertheless,
both bounds certify **0/372** attempted nonvacuous steps for each shortlist
size 512, 640, 768, 896 and 1,000. These are repeated configurations on the
same observations, not independent samples.

| Retained candidates | Original useful certificates | Subset-bound useful certificates | Added score-block work, including failed attempts |
|---|---:|---:|---:|
| 512 | 0/372 | 0/372 | 48.29% |
| 768 | 0/372 | 0/372 | 72.42% |
| 1,000 | 0/372 | 0/372 | 94.28% |
| 1,024, initially all blocks | 0 | 0 | 0%; all 372 successes are vacuous |

At shortlist 768, the per-layer median available score margin is only
**0.748–1.655**, while the strengthened drift allowance is **12.567–17.386**.
The certificate is too loose on this history despite the mathematical
improvement. These counts exclude additional selection, norm, gathering and
control work; they are not timings. The fixed maximum-anchor-age sweep also
retains all failed attempts and separate vacuous controls.

### Temporal reuse: prose at 4K and 16K

The same procedure passes on architecture prose at
[4K](../../../../parity-results/qwen-flash-next/qsa192-study-architecture-prose-c4096-r1/result.json)
and [16K](../../../../parity-results/qwen-flash-next/qsa192-study-architecture-prose-c16384-r1/result.json).
Each has 384/384 native selections matching the independent oracle and immutable
historical key prefixes. The three histories therefore cover **1,152 observed
layer/step selections**. They are not a whole-model accuracy certification.

Both prose histories also produce **zero useful temporal certificates** for
every tested shortlist and anchor-age policy. At 4K the same-anchor coefficient
ratio is **0.47287**; at 16K it is **0.47419** (median of layer medians for the
nonvacuous configurations). The 16K sweep covers 512, 768, 1,024, 1,536, 2,048,
3,072 and 4,095 retained candidates. Even excluding just one of the initial
4,096 blocks never certifies a reuse step. Keeping all initial blocks succeeds
vacuously and saves no score work.

At 16K, shortlist 1,024 adds **24.20%** score-block work after failed attempts;
its per-layer median available margin is **0.147–0.974**, versus a strengthened
drift allowance of **10.170–18.205**. No production temporal-reuse kernel is
justified by these pilots. Conservative FP32 allowances would widen this
certificate further. The requested 64K and 128K measurements and their results
are recorded below.

### Page pruning: initial 4K and 16K results

The online page experiment passes on
[4K Python](../../../../parity-results/qwen-flash-next/qsa193-pages-python-code-c4096-r1/result.json)
and [16K prose](../../../../parity-results/qwen-flash-next/qsa193-pages-architecture-prose-c16384-r1/result.json).
Pages of 16, 32, 64 and 128 pooled blocks (64–512 tokens) all prune **zero**
keys. This remains zero even with an optimistic final full-scan threshold, so
improving threshold seeding cannot rescue these cases. Each configuration
scores 100% of visible blocks and adds geometry/metadata work. The exact final
selected IDs match the independent oracle.

The [matched 4K prose repeat](../../../../parity-results/qwen-flash-next/qsa193-pages-architecture-prose-c4096-r1/result.json)
also prunes zero keys at all four sizes. The
[oracle-only small-page screen](../../../../parity-results/qwen-flash-next/qsa193-tiny-pages-oracle-three-captures-r1/result.json)
finds that two-block pages can prune at most **0.04105%** of keys for 4K Python,
**0.00203%** for 4K prose and **0.19818%** for 16K prose; four- and eight-block
pages prune none. These are optimistic potentials, not an executable algorithm.
Two-block pages require more than 50% pruning merely to offset their four
center dot products per page, before norms, metadata, selection and control.

### 64K extension

The [native 64K run](../../../../parity-results/qwen-flash-next/qsa192-learned-architecture-prose-c65536-r1/result.json)
passes on all four MI50s, with 32 observations per sparse layer, clean driver
and retirement checks, and GPU usage returning to its pre-run level. It uses
the same binary and precision as the pilots. The longer source file comprises
96 real documentation files, deduplicated by resolved path and source content;
no repeated-token padding supplies the context.

The [temporal analysis](../../../../parity-results/qwen-flash-next/qsa192-study-architecture-prose-c65536-r1/result.json)
has 384/384 native selections matching the independent oracle. Every shortlist
from 512 through 12,288 gives zero certificates. At 16,383 candidates, excluding
one initial block, the original bound passes 0/372 attempts and the subset bound
passes **4/372**. Failed attempts still make the stronger policy perform
**95.81% more score-block work** than a full scan. The same-anchor coefficient
ratio is **0.47185**; keeping all 16,384 initial blocks is again vacuous.

The [online page analysis](../../../../parity-results/qwen-flash-next/qsa193-pages-architecture-prose-c65536-r1/result.json)
prunes zero keys for pages of 16, 32, 64 and 128 blocks, including with the ideal
oracle threshold. The [small-page screen](../../../../parity-results/qwen-flash-next/qsa193-tiny-pages-oracle-c65536-r1/result.json)
does improve with context: two-block pages can optimistically prune **3.75986%**
of keys and four-block pages **0.02606%**; eight-block pages still prune none.
These remain below the respective 50%, 25% and 12.5% head-dot break-even points.
For two-block pages, even the optimistic count including center dots is
36,810,576 versus the baseline's 25,171,584 head dots, before other overhead.

### 128K extension

The [native 128K run](../../../../parity-results/qwen-flash-next/qsa192-learned-architecture-prose-c131072-r1/result.json)
passes with the same binary, settings, long real-text source and 32-step/12-layer
inventory. Driver and retirement checks pass; all four cards return to their
pre-run VRAM usage. The [temporal analysis](../../../../parity-results/qwen-flash-next/qsa192-study-architecture-prose-c131072-r1/result.json)
again finds no certificates for useful shortlist sizes, from 512 through 24,576.
At 32,767 candidates, excluding just one initial block, the subset bound passes
23/372 attempts but still adds **90.87%** score-block work. The same-anchor
coefficient ratio for the smaller shortlists is **0.46791**. Counts here use no
scheduled anchor expiry; the separate eight-step expiry sweeps are retained
and also provide no positive net score-work savings.

The [online page experiment](../../../../parity-results/qwen-flash-next/qsa193-pages-architecture-prose-c131072-r1/result.json)
prunes one 16-block page across 12,584,352 examined blocks: **0.000127%**.
Pages of 32, 64 and 128 blocks prune none. The
[small-page oracle screen](../../../../parity-results/qwen-flash-next/qsa193-tiny-pages-oracle-c131072-r1/result.json)
shows improved but still uneconomical potential:

| Pooled blocks per page | Optimistic fraction of keys skipped | Minimum fraction to offset center head-dot work |
|---:|---:|---:|
| 2 | 14.8724% | More than 50% |
| 4 | 0.49983% | More than 25% |
| 8 | 0.01386% | More than 12.5% |

These thresholds exclude additional norms, metadata, sorting, selection and
control. Two-block pages still require 68,019,360 head dots in the optimistic
count versus 50,337,408 for baseline scoring. No runtime speedup is inferred.

Across all five histories, **1,919/1,920** native selected-ID sets match the
independent float64 score oracle. The single difference is layer 47, decode
observation 30, at 128K. Float64 favors block 30,858 over 24,896 by
`4.3120617388e-7`, only **0.9043 FP32 ULP** at that score. A CPU reproduction of
the actual FP32 FMA, 32-lane reduction, head-sum and scaling order produces the
same `0x40e74af7` score for both; the smaller-ID tie rule selects 24,896. All
512 reproduced IDs match the native observation. Rounding only the final
float64 score would not reproduce the tie: the earlier arithmetic roundpoints
matter. This is a source-order explanation consistent with native selection,
not a measurement of GPU score bits or an engine defect. The
[boundary reproduction](../../../../parity-results/qwen-flash-next/qsa192-193-study-source-r1/qfn-qsa128-rank-boundary-private/REPORT.md)
retains the exact operations and evidence. The original 383/384
float64 agreement in the 128K artifacts is preserved.

The native runs are diagnostics, not throughput measurements. Sampled peak
VRAM is **27.879 GiB/card at 64K** and **29.504 GiB/card at 128K**. Snapshot
collection peaks at **127.24/302.67 GiB RSS**, respectively; these are not
ordinary-serving capacity or leak measurements. Exported evidence consumes
about **3.03/6.08 GiB** of disk space. Passive resource sampling, typed export
budgets, source/runtime identities and driver checks are retained with each run.

## Observed geometry

The three initial histories have median consecutive head drift of **0.66–0.71**
times the preceding head's norm. Their median top-512 overlap is **65.9–84.2%**;
overlap alone cannot certify all selected keys. A layer's maximum key norm is
typically only **1–2%** above its median (worst observed ratio 1.040), so a large
key-norm outlier is not the main explanation for these loose temporal bounds.

For pages of 16 blocks the median radius is about **0.68** times the page's
median key norm, rising to **0.83–0.86** at 128 blocks. These are broad balls,
consistent with a large worst-direction term in the page bound. This is an
observed limitation of these histories and page geometries, not an impossibility
result for other partitions or certificates. No numerical-error cause is inferred.
