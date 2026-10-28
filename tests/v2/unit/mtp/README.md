# MTP controller trace replay

`v2_test_mtp_depth_trace_replay` invokes the actual resident observation transition
in `SamplingMath.h` or actual host `MTPDepthController`. It loads no model and
initializes no accelerator. Authored golden inputs specify decisions and terminal
values independently; replay does not generate its own expected oracle.
Controller replay, malformed inputs, codec/reset, publication witnesses and the
CLI have explicit `ProductionTestPreflight` registrations as well as Unit coverage.

```sh
cmake --build build_v2_integration --parallel --target v2_test_mtp_depth_trace_replay
ctest --test-dir build_v2_integration --output-on-failure --no-tests=error \
  -R '^V2_Integration_MTPDepthTraceReplay'
```

## Capture and publication ownership

Use one fresh scalar enabled-MTP request with positive depth, at most256 generated
tokens, warmups0 and measurements1. Set `LLAMINAR_MTP_DEPTH_TRANSACTION_TRACE=1`,
`LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=0`, `LLAMINAR_BENCHMARK_ITERATIONS=1`,
`LLAMINAR_PERF_STATS_JSON` and `LLAMINAR_PERF_STATS_FILTER=mtp`. Leave CPU/GPU stage
timing disabled. These diagnostics are not throughput evidence.

The exact `mtp.depth_transaction_trace.{initial,record,final}` families retain
source-authored inputs. Group by actual participant and observation identity;
never merge requests or count TP participants as independent observations.
Native records use the existing exclusive diagnostic bank and terminal copy;
CPU records come from actual host publication and the actual controller. Competing
first-transaction diagnostics, CPU bypass and request batching are unsupported
in this bounded trace mode and fail admission explicitly.

Let O be the source output list length including its condition row, L its actual
already-emitted leading count, P the actual published main-state rows, and D=O−P.
Both L and D are exactly0 or1 at the admitted publication boundary. Actual newly
emitted output is Y=O−L; actual published progress is V=P=Y+L−D. Typed publication
witnesses validate the actual O/L/P/stop counts before narrowing to bools. Neither
acceptance nor rollback nor bonus-ready presence supplies missing publication data.
A stopped correction can have raw D1 even though its next live carry is0.

## V3 format and independent terminal witnesses

Normalized requests use `llaminar.mtp.depth_trace_replay_request.v3`; native trace
header version is3 and CPU initial policy schema is
`llaminar.mtp.cpu_depth_transaction_trace_initial.v3`. Initial and terminal native
controls remain108 words; trace3520B fits the existing exclusive3912B bank.
Each record stays12B and each transaction object has exactly `ordinal`, `packed`,
`token_witness`, and `evaluated_windows`. Bits29/30/31 encode actual L/raw D/stop.
The owner bit fixes the token-witness unit:

- Native resident: actual cumulative committed response count; consecutive
  differences supply Y. Actual initial/final controls supply independent Y and V
  endpoints through ResponseTokenCount and PublishedStateCommitCount. Carry,
  model-stop, budget, terminal count and last-emission witnesses must also match.
- CPU host: actual transaction Y, never a fabricated cumulative response count.
  Requested/effective depth preserve their five-bit fields. The actual fresh host
  policy retains round-trip floating values and explicit initial_incoming_debt.

The CPU terminal object requires transactions/current_depth/evaluated_windows,
emitted_tokens/published_state_tokens/model_stopped, and utility_probes/keeps/returns.
The ordinary same-request MTP summary supplies independent transaction/depth/window
and utility totals. Its verifier Y is ONLY the exact `mtp.output_tokens` counter
with source=committed_host_mtp_transaction. Its V is ONLY the successful
`mtp.grouped_outcome_host_state_publications` ledger, weighted by its exact
accepted_state_count tag; require scalar request1 and publication-event count
equal to verifier_runs. The explicit stopped_on_output tag supplies the verifier
stop witness. Both exact families survive measured serialization; lookalikes and
stale records do not. Untagged direct outputs remain in general output metrics
but cannot become verifier witnesses. In particular a CPU host-visible first stop
can complete without a verifier and must not be fabricated as a trace record.
Never construct independent endpoints from trace length, last record, trace sums,
accepted-plus-runs, physically computed rows or shifted-KV publication counts.

The one-request collector epoch is part of this contract. A normalizer must retain
producer IDs, receipt/config provenance and exact framing. The standalone tool
compares the53 controller-owned static/window/learner/report words against baseline
terminal state. It separately checks publication endpoints; it does not rerun model,
sampler, QSA or main-state publishers. Candidate output exposes no final projection
past an unsupported depth, and CPU replay never invents a resident control row.

## Baseline and candidate replay

```sh
build_v2_integration/tests/v2/v2_test_mtp_depth_trace_replay \
  --replay-depth-trace /absolute/normalized-request.json
```

`intent: "baseline"` requires this executable's identical compiled source/table
identity and reproduces every decision and owned endpoint. Missing, corrupt,
duplicated, truncated, unordered, out-of-scope or identity-mismatched inputs fail.
The CLI refuses to replace existing output files. Fixtures compile into the tool;
installed device-free preflight needs no SDK, model or corpus download.

Use `intent: "candidate"` explicitly to assess another implementation/table only
while each requested depth remains observed. The trace/control/publication ABI
and immutable backend/model/verification/scope must still agree. Candidate replay
reports both identities, never certifies an old baseline as current, and stops at
the first depth requiring unobserved feedback. It cannot invent a counterfactual
response or establish model economics. Fresh uninstrumented runs own performance.

The genuine TP4 captures in `MTPDepthTraceRecorded.h` and
`MTPDepthTraceDirectionalRecorded.h` remain byte-identical historical v2 evidence.
They are now explicit incompatibility regressions, not v3 positive replay proof.
V3 cannot infer their missing publication/debt/stop bits, even with candidate
intent or unchanged high bits. Never pad, relabel, upgrade or restamp them. Retain
their original exact replay binary/receipts. The genuine v3 captures below supply
separate recorded-input coverage; authored v3 golden/native tests complement
those observations.

## Genuine survival v3 TRAIN inputs

`MTPDepthTraceSurvivalRecorded.h` retains the original natural Python TRAIN
request from each sampling law: greedy71 transactions/15 windows and stochastic57
transactions/12 windows. Both real paths include probes, keeps and returns; the
greedy path visits depths1..6, stochastic3..7. All four TP participants agreed on
raw transaction inputs, exact ordered decisions/windows and the53 owned terminal
words before a single representative participant was embedded. Full raw initial
and terminal controls and original producer identity remain intact. Source
receipt/config identities are preserved; only test intent/output differ.

The expected ordered chronology is compacted losslessly from independently saved
production baseline output. After a controller change those original source IDs
must not be rewritten. `V2_Integration_MTPDepthTraceReplayHistoricalSurvivalV3`
rejects old-source baseline intent and verifies that the confirmation candidate
stops at the first changed selector, transaction 4, without borrowing later
feedback. Truncated-tail, independent V/Y/carry and incompatible ABI negatives
remain covered. These TRAIN inputs are historical correctness evidence, not
held-out or throughput evidence; v2 headers remain unchanged and incompatible.

## Proposal confirmation replay

`V2_Integration_MTPDepthTraceReplayConfirmation` uses a new independently authored
16-transaction, 19-token case through actual host and resident controllers. It
checks distinct completed windows at transactions 4/8/12, actual target/streak
chronology, the third window's reference and a final clipped transaction that
leaves the trial incomplete. Resident endpoints still compare all53 owned words;
reset, truncation and malformed endpoints fail explicitly. Real source-matched
captures and baseline replay remain necessary to qualify a new controller.


## Genuine confirmation v3 TRAIN inputs

`MTPDepthTraceConfirmationRecorded.h` separately retains predetermined natural
Python TRAIN inputs from both laws after all10 captures and40 strict baseline
replays passed. Greedy has58 transactions/13 evaluated windows, one actual
probe/return and final depth5; stochastic has56 transactions/13 windows, one
probe/keep and final depth6. Both publish256 response tokens and256 state rows,
with one clipped final transaction. Confirmation recommendations change while
active depth remains5; three disjoint matching proposals start a trial at
transaction28 (greedy) or44 (stochastic). Pending compares at32/48 respectively.
Each complete original initial/control/record/producer identity is preserved;
only test intent/output differ. Expected decisions, promotion_streak, windows
and53-word terminal projection come from saved native baseline output, not a
newly generated controller oracle. All four TP participants agreed before
ROCm:0 was embedded; they are not four independent samples.

`V2_Integration_MTPDepthTraceReplayRecordedConfirmationV3` repeats both genuine
paths three times through the actual production replay, requires the complete
observed depth path and retained chronology/terminal, and rejects corrupted
tail/publication/carry/ABI inputs. An unsupported future selector fails this
positive regression instead of borrowing later observations. Original survival
v3 and historical v2 headers remain byte-identical with their existing explicit
incompatibility/frontier tests. Real recorded inputs complement the authored
stop/partial/reset and native capture tests; they make no throughput, held-out
or policy-training claim.
