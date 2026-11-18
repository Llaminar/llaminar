# Four-MI50 QFN inference phases against Castagna Veloce

The fixed-depth verifier is the largest measured gap on the 512-token review
prompt. Llaminar needs 25.06 ms of device activity per four-row verification.
Castagna needs 16.97 ms including its native peer reductions. The two engines
are within 0.011 ms per draft token on average. These measurements use the
tuned Llaminar Release candidate. Its individual switches remain experimental.

## Matched workload

Both engines ran the original four UD-Q4_K_XL shards of Qwen3.8-Flash-Next
and, for speculation, the shared Q8_0 predictor. They ran tensor sharding on
the same four 60-CU gfx906 cards behind one switch (PCI buses 8c, 91, 97, 9a)
with FP16 KV, RAM ngrams, an empty prompt cache, greedy generation and exact
matched input IDs. Reported throughput is the median of three unprofiled
256-token generations after warmup. Decode counts the 255 tokens after the
first. Native timings come from separate rocprofv3 traces.

The Llaminar core SHA256 is
`731498df6d777a77372aa68dcff49b9ad67fb26e5f3822744057e2c45f4c800b`.
Castagna is revision `3483d462715d61f21ed5877836be35a724446359`.
Each profile reproduced its own unprofiled token history and draft work.
The two engines generate different histories even without MTP. At fixed
depth Llaminar matched all six of its ordinary histories. Castagna diverged
from its own ordinary decoded history at review token 224 and code token 14
(zero-based). Routed work can therefore differ between engines.

## Fixed depth three on review

The table gives mean per-device native time in milliseconds. For the two
phase totals, overlapping dispatch intervals on one device count once. Stage
breakdowns sum named kernel durations. Each number uses its own physical
per-round or per-draft count. All four devices contribute equally.

| Boundary | Llaminar | Castagna | Llaminar minus Castagna |
| --- | ---: | ---: | ---: |
| Complete verifier, per four target rows | 25.056 | 16.970 | +8.086 |
| HC, projections, activation preparation, routing and experts | 19.221 | 12.616 | +6.605 |
| Target peer reductions | 2.886 | 2.539 | +0.347 |
| Gated delta net and convolution | 1.357 | 0.962 | +0.395 |
| Attention, indexer and RoPE | 0.707 | 0.234 | +0.473 |
| Target vocabulary projection | 0.284 | 0.313 | −0.028 |
| Draft prediction, per proposed token | 0.988 | 0.977 | +0.010 |
| Accepted-state publication, per verification | 1.133 | — | — |

Llaminar completed 74 verifier rounds, proposed 222 tokens and accepted 182.
Castagna completed 76 rounds, proposed 228 and accepted 181. All 76 native
target head boundaries contained exactly 96 peer reductions on each card.
Castagna issues those reductions outside its captured compute graphs.
Counting only the repeated captured portion gives 14.35 ms for the target
and 0.61 ms for the draft, substantially understating its cost. The complete
target column includes the two eager rounds. The complete draft column
includes all work on the fork's draft queue between target boundaries.
Llaminar's retained graph markers identify verification, draft and accepted
state publication separately. The fork trace has no independently verified
accepted-state publication interval.

The previous Llaminar verifier took 28.75 ms. The selected group of changes
saved 3.70 ms (12.9%). Its HC/projection/router/expert subtotal dropped from
23.21 to 19.22 ms. Ordinary projection kernels, routed experts, and grouped
routing still use 4.92, 5.10 and 4.14 ms respectively per verifier. Castagna
fuses these roles, so comparison below the combined subtotal requires a matched
operator replay. Each Llaminar captured verifier issues 2,295 dispatches
versus 485 in Castagna’s captured target compute. The complete Castagna target
queue issues 588 dispatches per round on PCI 8c.

For the 512-token prefill, the first-to-last observed device work spanned
510.04 ms in Llaminar and 607.08 ms in Castagna. Mean dispatch interval
unions were 508.61 and 349.04 ms. Castagna used physical input batches of
320, 188 and 4. Llaminar used one 512-row prefill. The profiler perturbs host
submission and device overlap. Use the unprofiled throughput for end-to-end
speed comparisons and separate native intervals for attribution.

## Ordinary decode on review

Each trace has all 255 post-prefill vocabulary boundaries on all four devices.
The table gives mean device-active milliseconds per token for each stage.
The 36 fused GDN input projections per token are charged to the
HC/projection group. Their native symbol contains the word “attention”.

| Stage | Llaminar | Castagna | Llaminar minus Castagna |
| --- | ---: | ---: | ---: |
| HC, projections, activation preparation, routing and experts | 8.219 | 6.747 | +1.472 |
| Attention, indexer and RoPE | 0.650 | 0.204 | +0.446 |
| Gated delta net and convolution | 0.745 | 0.600 | +0.145 |
| Vocabulary projection | 0.247 | 0.232 | +0.015 |
| Communication | 1.969 | 2.439 | −0.470 |

This identifies two distinct tuning opportunities. The verifier and ordinary
decode both spend more GPU time on HC/projection/routed expert computation.
Ordinary decode already leads on interconnect kernel time. For the 512-token
prefill, 49 gather kernels, 48 ordered folds and 49 FP16 reductions account
for most of Llaminar's 81.57 ms of communication kernel activity. Castagna's
communication kernels take 10.43 ms. The pinned fork defaults to 16-bit staging for large reductions and enables
peer DMA. Both ordinary kernel traces omit copy records. Compare actual wire
formats, copy events and payloads before interpreting the 71 ms gap as
communication latency.

## Unprofiled comparison

Each pair of numbers is prefill / decode in tokens per second.

| Mode and prompt | Llaminar | Castagna first control | Castagna quiet repeat |
| --- | ---: | ---: | ---: |
| Off, review 512 | 1038.42 / 86.92 | 878.32 / 86.17 | 876.82 / 86.38 |
| Off, code 2048 | 1008.12 / 80.63 | 1003.65 / 75.93 | 1003.40 / 75.48 |
| Fixed 3, review 512 | 999.16 / 118.68 | 774.28 / 114.89 | 785.74 / 117.62 |
| Fixed 3, code 2048 | 970.95 / 97.16 | 888.35 / 106.31 | 892.87 / 108.39 |
| Dynamic, review 512 | 1002.89 / 116.99 | — | — |
| Dynamic, code 2048 | 970.22 / 95.08 | — | — |

The first Llaminar fixed-depth run measured 108.91 and 95.05 decode tok/s.
Its code prefill samples of 683, 643 and 968 tok/s overlapped the Release
build. After the build finished, the same executable and settings produced
970.17, 970.95 and 971.00 tok/s with identical histories and draft work.
The table uses this repeat. A separate Castagna run during the build measured 85.71 and 82.16 decode
tok/s. The quiet fork repeat recovered to 117.62 and 108.39. Against that
repeat Llaminar leads by 0.90% on review decode and trails by 10.36% on
code decode. With the verifier improvements disabled on the same quiet
Llaminar ELF, decode measured 105.96 and 87.72 tok/s. The two
ordinary Castagna controls agree within 0.6% on all four rates. The ordinary
code prefill lead is about 0.5%, so a distribution over independent runs is
especially important there. The fork's fixed-depth code requests required
78–81 verification rounds versus Llaminar's 86. Report both per-round
kernel cost and total rounds for this prompt.

## Fixed-depth follow-up

A second Release image compares three additional changes on top of the complete
previous verifier policy. It uses register-based router top-K, a fused short-context
identity-indexer preparation, and vector GDN accepted-state publication. The
two arms loaded the same executable and core
`23fc618630bad1dc69b7b816e67248295ecf5c600504cebd43a3ecc5460f16e0`.
All six measured outputs matched the ordinary same-image generations. All
five draft and verifier work counters matched the fixed-depth control in each
iteration. Both arms passed driver and retirement checks before the host reboot.

| Fixed 3, unprofiled | Control prefill / decode | Candidate prefill / decode | Castagna prefill / decode |
| --- | ---: | ---: | ---: |
| Review 512 | 1002.48 / 119.20 | 1002.74 / 122.94 | 785.74 / 117.62 |
| Code 2048 | 970.54 / 97.96 | 970.94 / 100.20 | 892.87 / 108.39 |

The candidate gained 3.14% and 2.29% in decode over its control. Review is
4.53% above the quiet fork median. Code is 7.55% below it and needs another
8.17% throughput gain to reach parity. The control and treatment executed
74 review and 86 code verifications per generation. The contributions of the
three individual changes still require native phase measurements. The dynamic
comparison was interrupted by a host reboot and has no paired result.

For the earlier core, the 2048-token trace produced all 256 expected IDs and
exactly reproduced all three unprofiled token histories, all five work counters
and the 86 M4 verification boundaries. Its original profiling runner rejected
the trace because it required a 512-token input. Offline replay of the corrected
workload check accepts both genuine prompt receipts and rejects altered input
counts, generated IDs and MTP counters. This gives **diagnostic** stage timings
of 26.664 ms per verification and 1.032 ms per draft. The independently
qualified fork trace measured 18.222 ms and 1.208 ms respectively over its
81 verifications and 243 proposals. A passing native rerun is still required
before using the Llaminar code trace as a phase gate.

On ordinary review decode each Llaminar GPU executes 12 attention producers
per token, while each Castagna GPU executes six. The pinned fork rotates pairs
of whole GQA-group owners. Llaminar divides query heads across four participants
in each layer. Both perform 72 query-head/layer instances per GPU. Mean
device-active time therefore answers a different question from elapsed time
across the active participants. [Source comparison and ownership](../2026-10/QFN_CASTAGNA_STAGE_COMPARISON.md)

## Reproduction

The local evidence root for native phase attribution is
`/tmp/qfn-mi50-current-stage-pair`. It contains the read-only SQLite readers,
per-device classified results, native queue bounds and timing receipts. The
two original model trace roots retain the device mapping, loader, source,
token and GPU retirement receipts. Re-run analysis on those receipts with:

```bash
P=/tmp/qfn-mi50-current-stage-pair
D=/tmp/qfn-mi50-grouped-stack-final-r1
M=/tmp/qfn-mi50-retained-real-tp4/r7-all-tp-live/full-model
python3 "$P/phase-queues.py" review512 "$D/quartet-r1-fixed3-trace" \
  "$M/trace-fork-fixed3-r1" "$P/review-complete-replay.json"
```

The three-change unprofiled reproduction uses the frozen source and archive
members recorded beside the Release image. A fresh cohort name is required.

```bash
X=/tmp/qfn-mi50-phase-tuning-r2
flock -x "$M/grouped-mtp-suite.lock" bash "$X/run-phases.sh" execute replay-01
```

The clean pre-reboot evidence is under `$X/quartet-r1-fixed3-{control,phase}`.
The matching ordinary arm is `$X/quartet-r1-off`. The next measurements are
the full-model grouped FFN A/B, native attribution on both prompts and the
dynamic control/treatment pair. Use fresh same-boot controls after a reboot.
