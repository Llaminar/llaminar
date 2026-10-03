# Public HTTP lifecycle hammer

Use this diagnostic when an intermittent request/reset/prefix/MTP defect has
survived the requested fresh-lifetime loop. Extended stress must be in scope;
passing a bounded loop alone does not establish the original fault's cause.
The warm-server hammer complements, but does not replace, full fresh HTTP E2E
lifetimes and their startup/teardown evidence. For intermittent retirement
defects, use the same driver's fresh-lifetime mode so shutdown is exercised
after each finite workload, rather than only when a warm request loop fails.

## One lifecycle and one configuration authority

`scripts/ci/run_http_lifecycle_hammer.py` selects exactly one existing cell from
the canonical exported E2E manifest. It reuses the normal model-staging lease,
complete GGUF shard inventory, Release/image admission, public serve arguments,
readiness, driver diagnostics, memory checks, and retirement. The mature
`test_server_e2e.sh` owns the server; `http_lifecycle_hammer.py` only drives its
HTTP workload. Do not maintain another topology table, launch MPI manually, or
reload the model/re-run prerequisites for every request cycle.

Freeze the active harness and runtime source while a lifetime is running.
Bash can read its script lazily: rewriting that file in place can produce a
late parser error even after all inference checks succeeded. Preserve such an
attempt as a harness/infrastructure failure, not a native stall or a clean
full-lifetime pass. Run replacement lifetimes after the harness is stable.

The harness's driver checkpoint is one-shot: `finish` closes it and belongs
after server retirement. Midpoint inspection must only read a snapshot and
compare it with the armed cursor (the observer's `read_snapshot`/`new_records`
functions do not mutate the checkpoint). Never call `finish` on the canonical
state while inference is live or reuse that partial interval as a lifetime
certificate.

## Preserve the failing sequence

Supply schema-1 JSON with a nonempty, ordered `requests` array. Each item has a
unique safe `name`, the exact public request `body`, and an independently known
`expected_numeric_answer` string. This initial diagnostic oracle is for short
arithmetic/recall sequences; it is not a mathematical-parity or long-generation
oracle. Do not infer expected answers from the failing candidate's output.

```json
{
  "schema": 1,
  "requests": [
    {
      "name": "streaming_thinking_after_restore",
      "body": {
        "messages": [
          {"role": "system", "content": "Reply with only the numeric answer."},
          {"role": "user", "content": "What is 1+1?"}
        ],
        "max_tokens": 200,
        "temperature": 0.0,
        "enable_thinking": true,
        "thinking_budget_tokens": 16,
        "stream": true
      },
      "expected_numeric_answer": "2"
    }
  ]
}
```

In a real reproduction, include the earlier requests in their original order:
prefix producers, exact/partial restores, independent request resets and sampler
or reasoning-policy changes can all be prerequisites. Preserve prompt bytes,
budgets and sampling settings. This workload does not alter the cell's expert
movement, graph, MTP, precision or capacity policies.

```bash
python3 scripts/ci/run_http_lifecycle_hammer.py \
  --build-dir build_v2_integration \
  --binary /absolute/path/to/Release/llaminar2 \
  --manifest /absolute/path/to/canonical-e2e-manifest.json \
  --cell 'exact-registered-cell-selector' \
  --sequence /absolute/path/to/preserved-http-sequence.json \
  --model-ramdisk-root /mnt/llaminar-production-parity \
  --persistent-model-cache-dir cache \
  --output /absolute/path/to/new-diagnostic-directory
```

`--cell` is a full-match regular expression against the manifest's complete
`case` identity, including its suite/test prefix, not only the final parameter
suffix or campaign binary. Inspect the canonical names first:

```bash
jq -r '.cells[].case' /absolute/path/to/canonical-e2e-manifest.json
```

Select exactly one complete name and escape regex punctuation (for example,
write `[.]` for a literal suite/test separator). Zero or multiple selections
are admission errors before model staging; do not broaden the selector or
invent another cell definition to make a diagnostic launch succeed.

`--container-image` uses the same immutable-image admission as ordinary E2E.
The defaults, `--cycle-limit 0 --lifetime-limit 1`, keep one warm server until
the first failure or an explicit interruption. A positive cycle limit bounds
the workload within each lifetime. Add `--lifetime-limit 0` to repeat fresh
server lifetimes until failure, or a positive lifetime limit for a bounded
diagnostic. Repeated lifetimes require a positive cycle limit so they can reach
normal retirement; invalid geometry is rejected before discovery or staging.
For example, append `--cycle-limit 2 --lifetime-limit 0 --prefix-pressure` to
the command above to repeatedly pressure prefix reset/restore, retire that
server, and start another using the same staged weights and unchanged cell.

One staging lease covers the entire diagnostic, not each lifetime. Every
server still goes through the existing harness's readiness, HTTP workload,
retirement and final driver checkpoint. A failed launch, request, retirement,
or incomplete driver report stops the diagnostic without retry. Bounded
success remains non-certifying. There is no global stress deadline; every
HTTP call retains a total deadline, alongside the normal native collective,
server-readiness and retirement watchdogs.

## Target prefix-cache lifecycle edges

Add `--prefix-pressure` to the same command when prefix restore/reset is a
suspected prerequisite. It derives long-prompt geometry from the selected
cell's E2E definition and recurrent/MTP state obligations from that model's
canonical generation definition. Missing typed metadata fails before staging;
the runner does not infer hybrid state or speculative policy from a model name.
It does not change server arguments, cache budgets, precision, expert movement
or capture policy.

Each cycle uses the mature long-needle builder, rotating beginning/middle/end
placements and a recorded lifetime/cycle nonce. It submits:

1. A new long prompt, requiring a genuinely cold admission and the known needle
   answer.
2. Its exact repeat, requiring full restoration.
3. An extension of that complete conversation, requiring partial restoration
   at the earlier actual token boundary. As in the canonical generation probe,
   append an assistant turn without a newer user turn: thinking templates can
   strip historic reasoning markers when another user message is added. Always
   authenticate the earlier full prefix with returned token IDs.
4. An exact repeat of the extension, requiring full restoration.
5. The unchanged preserved short JSON/SSE sequence.
6. Full restores of both long prompts after those large-to-small request resets.

The non-streaming prefix probes request original token IDs and terminal runtime
summaries. The shared `generation_regression_http.py` consumer proves actual
KV/GDN recurrent-state and shifted MTP-state restoration, not lookup eligibility
or optional PerfStats counters. Full repeats additionally compare every output
token and termination against the completed producer. Needle answers remain
an independent accuracy oracle; repeat equality alone is not mathematical parity.

Each lifetime's `hammer/progress.json` counts completed fresh/full/partial
probes and observed storage tiers; numbered results retain source ordinals,
token counts and full request-local prefix/MTP outcomes. The nonce prevents durable cache entries from
an earlier diagnostic lifetime manufacturing a fresh admission. Successively
new long inputs pressure the existing bounded RAM/disk tiers. That does not by
itself prove eviction, hydration or archive compaction: require their actual
runtime/tier/retirement evidence before claiming those paths were exercised.

A legitimate long cold prefill can exceed the short-request attachment
threshold. The hammer therefore leaves optional debugger callbacks dormant for
cold prefix producers; their normal HTTP/native watchdogs remain active. Full
and partial restore probes retain the configured passive callback. Choose its
threshold from observed healthy restore latency. Never raise a native timeout
or attach during known healthy long work merely to obtain a diagnostic trace.

## Failure evidence and optional attachment

Each lifetime's `hammer/progress.json` is atomically published before and after
each request. It retains completed cycles/requests, the active request and the
first failure.
Numbered request, response, headers, curl-error and result files are never
overwritten by another cycle. SSE bytes reach the response file as they arrive,
including an incomplete stream on timeout. HTTP failures, malformed/error SSE,
missing `[DONE]`, truncation and incorrect arithmetic all stop the loop without
a retry. An interruption retains its active request rather than manufacturing
a clean result. The outer `report.json` records the active/completed lifetime
ordinals and separate terminal outcomes, and explicitly declares
`certified: false`. Shared `server-args.json` and `hammer-configuration.json` remain
immutable; every lifetime has its own `harness.log`, request artifacts and
driver report under `lifetimes/000001/`, `lifetimes/000002/`, and so on.

If live native state is needed before the protocol's fatal timeout, supply
`--on-stall-argv FILE`, a JSON argv array, and an appropriate `--stall-seconds`.
The callback is dormant during healthy requests and runs only once after a
request exceeds that threshold. It receives:

- `LLAMINAR_HTTP_STALL_REQUEST` and `LLAMINAR_HTTP_STALL_PROGRESS`;
- `LLAMINAR_HTTP_HAMMER_OWNER_PID` and its Linux start-time identity;
- `LLAMINAR_HTTP_HAMMER_BINARY` for local-executable authentication.

Authenticate process birth, exact executable/runtime ABI and ancestry before
inspecting any rank; never attach to every similarly named host process. For
containers, authenticate the actual container/process namespace rather than
assuming the local binary path matches the image. Bound the diagnostic itself;
the runner stops its owned callback group after ten seconds. Use passive host
metadata/thread snapshots, not GPU debug traps, inferior function calls, tensor
payload reads, replacement kernels, or steady-state tracing. Choose a threshold
appropriate to the preserved short request, not a legitimate long prefill.
An attachment-induced failure is a diagnostic result, not an unobserved
stability certificate or proof that the original native fault recurred.

Focused tooling coverage is registered in both `V2_Unit_HTTPLifecycleHammer`
and `V2_Integration_HTTPLifecycleHammer` (`ProductionTestPreflight`). The latter
uses a loopback HTTP peer; it neither loads a model nor measures performance.
