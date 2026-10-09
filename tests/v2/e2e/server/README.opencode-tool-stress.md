# OpenCode tool calling stress

The default `--gate stress` release policy requires complete phases, successful tool
coverage, intact protocol evidence, and model tool errors strictly below 5% of
unique emitted tool calls. The cell aggregate and each long coding session must
both satisfy that boundary; exactly 5% fails. Known client-domain rejections
count as failed tool attempts, and their original arguments/errors remain in
the evidence. Unknown client errors, corrupted arguments, missing continuations,
transport faults and engine failures remain hard failures regardless of rate.
Task quality and app acceptance stay visible independently of this engine gate.
Phase receipts include duration and exchange count; token progress artifacts
distinguish active reasoning from a stalled session without a generation deadline.

`opencode_tool_stress.py` runs the installed OpenCode CLI against a live Release
server. OpenCode supplies its own system prompt, tool schemas, argument
validation, tool execution, and continuation messages. The harness records the
client version and relays its request bodies and JSON/SSE responses unchanged.

The qualified client is OpenCode 1.18.34. The published-suite controller installs
its pinned native npm artifact with `scripts/ci/install_opencode_client.py`, which
verifies artifact integrity and the actual executable version before publication.
For local use, pass `--destination /path/to/opencode` to that installer, then run:

```bash
python3 tests/v2/e2e/server/opencode_tool_stress.py \
  --base-url http://127.0.0.1:8080 \
  --model Qwen3.8-27B-IQ4_XS \
  --opencode /path/to/opencode \
  --context-length 262144 --max-tokens 32768 \
  --iterations 1 --concurrency 1 \
  --output parity-results/opencode-stress
```

The endpoint must report ready before any client starts; readiness failures
fail immediately. The output directory must be new. Each session uses a standalone temporary
workspace so repository discovery and ancestor `AGENTS.md` files cannot redirect
the coding tools. The harness retains its prompt, expected bytes, client events,
raw exchanges, and final workspace. It publishes `stress.json` after each session;
cancellation leaves incomplete evidence. `workspace-location.json` records the
standalone workspace before client work. Exceptions preserve that directory;
it is removed only after a successful copy into the session artifacts.
String fixtures cover JSON-looking
contents, quotes, code, whitespace, empty strings, Unicode and protocol tags.
Prompts enter the real client through UTF-8 stdin. OpenCode 1.18.34's positional
argument path adds literal quote escaping to messages, which changes quoted
fixtures before they reach the model. Every phase authenticates its intended
prompt against the first request's latest user text and records both hashes;
changed or missing text fails even if the model happens to produce the expected
file. The recorder still forwards the client's request bytes unchanged.
Each exchange also writes `exchange-NNNN.sse` as bytes arrive and publishes
`exchange-NNNN.progress.json` with activity times, separate reasoning/text/tool
argument counts, short tails, tool names and repeated-phrase signals. A signal
is diagnostic evidence, not an automatic failure or cancellation. SSE delta
counts are fragments, not model token counts; terminal `usage` supplies the
actual completion count. For exact per-token IDs and text previews, start the
server with `LLAMINAR_TRACE_GENERATED_TOKENS=1` and retain `docker logs -f` output.
This opt-in trace includes terminal token IDs and their disposition
(`text`, `stop_token` or `runner_complete`). It uses already surfaced output
tokens and adds no device reads. Terminal control tokens remain absent from
the client-facing generated text.
Trace-enabled runs diagnose behavior; use untraced runs for timing claims.
Qualification drivers can use `runtime_stats_observer.RuntimeStatsObserver`
around a workload, supplying a validated, read-only `/stats` reader with a finite
HTTP observation timeout. It writes successful snapshots to the requested JSONL
file and failed polls immediately to its `.errors.jsonl` sidecar. A transport
failure leaves qualification failed while later scheduled reads retain further
evidence; invalid statistics stop observation. The `.result.json` report keeps
the original failure and successful/failed poll counts. The observer never
resubmits generation, resets counters, or imposes a model-turn deadline.

Add `--measure-prefix-reuse` with `--concurrency 1` and an exclusive server to
measure checkpoint effectiveness during the real app workload. Each exchange
retains a `prefix_measurement` joined to its exact native request sequence and
wire token counts. The session's `prefix-reuse.json` reports restored-token
ratios and TTFT mean/median/p95 separately for ordinary requests, compaction and
the first request after compaction. It also records RAM/disk occupancy at request
boundaries and total committed tier traffic across the observation interval,
including background writes completed while tools run. Resource observation
still owns between-request physical peaks; pending archive writes require the
retired server's final PerfStats and storage audit.

The recorder forwards the original request and SSE bytes. Two small `/stats`
probes surround each measured request; an EOF-to-terminal-publication wait
applies only after generation has ended. Missing observations, resets, competing
requests, changed tier admission or mismatched token usage fail the measurement.
A session with no compaction reports an empty compaction cohort. Compare the
same model/topology, context, cache budgets and tracing settings on the baseline
and candidate; a stochastic session's TTFT change alone is not a controlled
performance claim. Retain captured requests for matched replay of transitions.

`resource_usage_observer.py` additionally journals host-PID process identities,
anonymous/file/shared RSS, swap, CUDA process memory, ROCm DRM mappings, cgroups,
and filesystem inode/block usage. Declare its exact `backends` inventory (`cpu`, `cuda`, `rocm`) in the observer
manifest; unselected vendors are never probed. Run it in the inference host's PID namespace;
container-local `/proc` cannot authenticate host PIDs. Cache payload observation
uses inode metadata only. The lifecycle owner seals the baseline before starting
the workload and binds each native rank and GPU UUID/PCI address independently
of enumeration order. Keep observation running through a final sample after the
workload, before retiring the server.

After retirement, `resource_growth_evidence.py --binding BINDING.json
--observations observations.jsonl --perf perf.json --output resources.json`
joins that binding to the complete rank-qualified initialized memory-authority
records. It checks each process's anonymous RSS plus swap, each GPU independently,
and every configured cache instance. Limits use the sealed baseline and remaining
admitted materialization; they never increase after a stats reset or a later peak.
Missing baseline/end coverage, observation gaps, PID reuse, OOM events and cache
ownership changes fail the receipt. File-backed RSS, shared driver mappings and
physical archive peaks remain separate diagnostics. This receipt does not replace
the metadata-only archive ownership audit after native retirement. Partial
observations from an already running session cannot certify its earlier lifetime.

Requested streaming usage must arrive once before DONE, and the real SDK must
report positive input counts so automatic context compaction can run. The
summary request is a separate, tool-free exchange. It retains the pending tool
join until the resumed coding request supplies the recent assistant/tool tail;
missing or changed continuation arguments still fail. Reports count these
auxiliary requests as `compaction_requests`, and native audits include their
tokens and reasoning even when the request omits `tools`. OpenCode's
[compaction implementation](https://github.com/anomalyco/opencode/blob/v1.18.34/packages/opencode/src/session/compaction.ts)
owns this history transition.
The live workload is the ten-phase app development session. Device-free parser
fixtures retain primitive string, Unicode and error cases; the live harness does
not schedule separate primitive sessions. The final app sentinel checks exact
JSON-looking file bytes through authenticated `write` and `read` calls.

Ten turns share one OpenCode session. Coding turns have no default elapsed-time,
socket-inactivity or agent-step limit. Prefill and a buffered tool call can keep
generating work without publishing SSE bytes; exact token traces remain useful
during those quiet intervals. The client provider explicitly sets `timeout`,
`headerTimeout` and `chunkTimeout` to `false`: OpenCode 1.18.34 otherwise applies
independent five-minute header and SSE-idle deadlines, even with no total limit.
The [upstream timeout implementation](https://github.com/anomalyco/opencode/blob/v1.18.34/packages/opencode/src/provider/provider.ts#L94)
owns these settings. `--request-timeout` optionally bounds upstream
socket inactivity, independently of total generation time. Readiness remains
bounded to five seconds.
An explicitly supplied `--session-timeout` is an operator stop limit; reaching it
records a harness interruption and incomplete evidence, not a proven model or
protocol defect. The agent builds a standard-library
Python/SQLite task app, then adds CRUD, HTML, emoji handling, input validation,
filtering, pagination, persistence and concurrent requests. Each turn has its
own prompt and client events. Independent HTTP acceptance checks launch the
generated CLI, verify the API and exact Unicode bytes, exercise twelve concurrent
creates and restart persistence, then run the authored tests. Configure the
client context to match the server's admitted context. The output limit includes
reasoning tokens; the example reserves 32,768 tokens within the total context.
The final review explicitly requests the `glob` tool for `**/*.py`, `grep` for
`def ` in Python files, and `read` for a discovered file. These are planned
review operations; omitted tools still fail the coverage gate.
Each prompt asks for a small increment and a status report after at most three
repair attempts, retaining any failing tests for the next turn. This is coding
task scope, not an enforced generation deadline or automatic retry policy. The
model's unresolved defects remain visible in independent app acceptance results.
For this model, start the server at its 262,144-token model maximum when physical
memory admission succeeds, with `--prefix-cache-ram-budget-mb 16384` and
`--prefix-cache-disk-budget-mb 32768` for the RAM and disk tiers.
The RAM budget applies to each participant; same-model TP participants share
the disk archive and its disk budget.
An app need not cross the archive's maintenance threshold. Report compaction as
unexercised when absent; any observed compaction must still copy zero payload
bytes under the archive worker. Dedicated `PrefixArchiveBackgroundPublication`
and `PrefixArchiveBoundedPayloadStorage` preflight tests exercise that lifecycle.

The disk budget limits live payload. The archive stores a metadata journal and
immutable payload files in its adjacent `.blocks` directory. Durable eviction
unlinks retired payloads immediately; compaction copies metadata only. Allow
filesystem headroom for one in-flight write, metadata, and the specific payload
inodes retained by readers. Old archive format versions are rejected intact;
use a fresh cache namespace when qualifying an archive format change.
The coding case allows the client's normal temporary test files outside the
isolated app workspace. Network browsing and delegated agents stay disabled for
this local workload. The final sentinel
also verifies that a JSON-looking file content remains a string tool argument.
One app session runs by default; `--iterations` repeats the full conversation.

Each session reports `protocol_passed` and `task_passed` independently. The
optional `--gate strict` requires both. The default stress gate also enforces
complete phases and the model tool-error allowance described above. For focused
engine/parser investigation, use
`--gate protocol`: incorrect model-authored code, wrong requested file bytes,
and recognized tool-domain outcomes remain task failures while the protocol is
graded independently. `passed` always retains strict success; `gate_passed`
owns the selected CLI exit criterion. Both scores and all errors are retained.

SDK/schema, HTTP, framing, unknown tool errors, changed client inputs and client
failures remain fatal in either gate, even if the agent later repairs them.
Validation independently checks required keys, declared string types, call
identities, assistant/tool history joins and normal termination. Recognized
edit-match, unchanged-edit, missing-file and external-directory denial outcomes require exact
wire/client argument agreement and remain in `execution_errors`. This labels a
tool outcome; it does not establish whether the model or a client rule caused
it. OpenCode's exact `glob` ripgrep execution error is also retained as a failed
external operation after matching valid wire/client inputs. It does not prove
the underlying filesystem/process cause or successful search coverage.
An invalid-escape `grep` diagnostic is a task failure only when it echoes the
complete authenticated regex and has ripgrep's exact diagnostic and caret
format. Changed escapes, mismatched patterns and unknown diagnostics remain
protocol failures. A rejected regex never counts as successful `grep` coverage.
Permissions are unchanged. Successful executions must cover every requested
tool for **each selected case** across its iterations. Missing sessions or tool
coverage prevent qualification even when every observed response is well formed.
An output-budget failure remains a failure; it is never retried into a pass.

Protocol success alone does not prove numerical engine accuracy or native-text
parsing fidelity. Join saved wire responses to the server's generated token IDs,
inspect any suspect arguments against decoded native output, and retain matched
generation controls, MTP/capture PerfStats and driver diagnostics. Model task
scores remain visible alongside that independent runtime evidence.
`authenticate_native_qwen_calls` provides an independent inverse-rendering
check: complete wire values must occur in the joined native call with all
required arguments. It matches literal string bytes before consuming structural
delimiters, so production parser replay alone cannot certify a corrupted value.
The caller must authenticate the raw token/request join and thinking boundary.
Increase `--concurrency` to exercise independent clients against the same server.

`prefix_archive_metadata_audit.py` audits a retired server's archive against its
committed metadata and inode inventory. The lifecycle owner must first prove
all native server processes and archive writers have retired. The audit rejects
orphaned, missing, aliased-generation or incorrectly sized payload files and
requires live bytes to fit the disk budget. It never opens payload contents;
format checks apply only to the small journal records. This is storage ownership
evidence, separate from the native reader's metadata validation.

`opencode_native_audit.py` provides the complete app-response audit. Published
image checks select `--decoder-image sha256:...` using the exact serving image
ID; its installed helper consumes stdin with the complete model shard directory
mounted read-only. Local diagnostics may explicitly select `--decoder PATH`.
These modes are mutually exclusive and neither substitutes another decoder.
Also supply the recorded `--exchanges` directory, complete `--server-log`,
recorded `--server-start-unix`, the served `--model` and a new `--output` directory.
The decoder loads GGUF/tokenizer metadata and the model's generation policy,
then replays the production streaming parser without running inference. The
independent Python check authenticates every Qwen XML argument and compares
ordinary text and reasoning byte for byte. Tool-free compaction responses stay
in the join. Missing, ambiguous or unaccounted native responses fail, including
unrecorded control requests. Keep separate control-server evidence or explicitly
retain a log scope bounded by authenticated control frontiers. An app-only
server log requires no exclusions. Other native grammars fail with a precise
diagnostic until their independent inverse-rendering proof is implemented.

Models with complete MTP heads enable dynamic MTP by default. Use `--no-mtp` on
the server for the explicit ordinary-decoding control; preserve topology and
other runtime settings between comparisons. Bracket every GPU server lifetime
with `gpu_driver_diagnostics.py begin` and `finish`, retain complete server logs
and enabled PerfStats, and verify the intended captured/MTP path actually ran.
`begin` arms an owned continuous collector; `finish` takes a final observation
after server teardown and retires that exact process. Keep its records journal
beside the report. Collected log rotation is harmless, but an observation gap,
reader failure, owner loss or missing journal fails the driver proof. These
observer handshakes impose no deadline on model loading or coding turns.
A failed close remains failed and retires only the authenticated observer
process using its native process handle. Closed state is published only after
retirement; an unavailable cleanup operation retains the live admission and
original failure for diagnosis.
Run the complete Unit and `ProductionTestPreflight` prerequisites after fixes.
This focused harness does not issue an image or model-matrix certificate.

The decoder's `--describe-model GGUF` mode reports the model's training context,
reasoning default and the production prefix block alignment using metadata only.
`opencode_context_admission.py` tries that exact maximum first, then searches
aligned candidates only after a typed PMA capacity rejection. It retains the
32,768-token output budget and proves adjacent admitted/rejected bounds. A
planning error, driver fault or incomplete retirement must stop the probe owner;
none can be classified as a capacity rejection from exit status alone.

`resource_growth_evidence.py` joins all rank-qualified PerfStats documents to a
sealed OS observation window. Startup PID evidence and Linux `NSpid` bind ranks
to host process lifetimes. GPU UUIDs and native PCI endpoints bind memory counters
to admitted devices; ordinals, process order and upstream PCIe bridges never
select owners. MPI inventory exchange retains that complete endpoint identity.
The observer reads bounded process metadata, including anonymous/file/shared RSS
and swap. No process environment or model/cache payload is needed for this join.
The resource receipt reports process/GPU growth and cache capacity; physical
archive ownership still needs its separate metadata-only retirement audit.

`opencode_certification_evidence.py` authenticates the complete ten-phase app
receipt for CI. It recomputes the stress verdict and checks uninterrupted phase
order, exact authored prompt identities, one continuing session, unique tool
counts and full native-audit coverage. It requires the pinned client, admitted
context, reasoning output budget and absence of generation deadlines. Failed
app acceptance remains visible independently of engine qualification. The
published suite and release publisher share benchmark-bundle admission so the
coding gate cannot consume stale, partial or failed benchmark evidence.

The master PR workflow runs the canonical coding matrix only after both
published benchmark lanes pass. `scripts/ci/run_opencode_matrix.py` derives
every blessed E2E cell on both ISAs; `run_model_parity_opencode.py` consumes one
exact case without creating a separate model/topology list. `opencode_cell_runtime.py`
owns context probes, serving, the real app, observers, native retirement and
retired cache reclamation. Production placement and MTP intent come from that
cell's canonical benchmark/runtime policies.

Both the inner owner and outer Docker driver prove native retirement. Containers
inherit a unique owner label and an init process for signal forwarding. Cleanup
uses exact daemon identities and successful inventory queries; a client exit,
localized error message or unchecked removal cannot prove absence. Exceptions
and operator cancellation retain their original verdict, even when exceptional
cleanup succeeds. A surviving or unproven parent blocks child enumeration and
the next admission.

`scripts/ci/opencode_evidence.py` independently replays the complete app/context
and runtime validators and requires every resource, driver, archive and cleanup
receipt. The final matrix cannot omit an ISA or a canonical case. Its compact
JSON retains all validator inputs, prefix measurements and app-quality results;
the release publisher revalidates it before promoting the benchmarked images.
