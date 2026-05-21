# Handover: Long-Context Server E2E Test Plan

Date: 2026-05-21
Workspace: `/workspaces/llaminar`
Status: implementation plan for next agent
Primary target: `llaminar2 serve` E2E server tests
Main harness: `tests/v2/e2e/server/test_server_e2e.sh`

## Goal

Add objective long-context and long-horizon generation coverage to the existing
server E2E suite. The purpose is to catch bugs that only appear after medium-long
prefill and decode lengths, especially KV cache, GDN hybrid cache, cache reset,
attention mask, RoPE/position, and long decode degeneration issues.

The tests should not depend on subjective human judgment. They should convert
"the model stayed coherent" into simple pass/fail checks using deterministic
prompts, structured outputs, response metadata, and degeneration heuristics.

## Existing Harness Facts

- The current server E2E harness starts `llaminar2 serve`, waits for `/health`,
  sends OpenAI-compatible `/v1/chat/completions` requests with `curl`, validates
  response format, checks streaming, checks memory, scans logs, then shuts the
  server down.
- Requests already use `temperature: 0.0` in `make_chat_payload`, which makes
  exact-answer tests practical.
- `ChatCompletionHandler` returns useful usage metadata:
  `prompt_tokens`, `completion_tokens`, `total_tokens`, `context_window`, and
  `context_used`.
- `ChatCompletionHandler` rejects prompts larger than the context window with
  HTTP 400 and a message that says to use `-c <size>`.
- The CLI supports `-c` / `--context-length`; default context length is 4096.
- The current validator assumes `finish_reason == "stop"`. Long decode tests
  that intentionally run to `max_tokens` should accept `finish_reason ==
  "length"` when that is the expected cap.

## Recommended Shape

Keep the existing smoke tests intact and add a separate long-context path. The
cleanest implementation is a Python helper invoked by the shell harness, for
example:

```text
tests/v2/e2e/server/long_context_checks.py
```

The shell script should start the server as it does today, then optionally call
the Python helper against the live port. Python will make prompt generation,
JSON validation, usage checks, and repetition metrics much easier than Bash.

Suggested controls:

```text
LLAMINAR_E2E_LONG_CONTEXT=1          # enable long-context checks
LLAMINAR_E2E_LONG_CONTEXT_TIER=lite  # lite or full
LLAMINAR_E2E_CONTEXT_LENGTH=4096     # passed to serve as -c
LLAMINAR_E2E_LONG_MAX_TOKENS=512     # lite default; full can use 1000
```

The precommit hook should use a low-cost "lite" tier. The full 1k prompt plus
1k generation tier is better suited for nightly/manual validation or a focused
developer command.

## Test Tiering

### Tier 1: Precommit Long-Lite

Purpose: catch obvious medium-context KV/cache regressions without making every
commit painfully slow.

Recommended scope:

- One small model suite, initially `models/qwen2.5-1.5b-instruct-q8_0.gguf`.
- CPU plus one available GPU backend if runtime is acceptable. If not, start
  with CPU only and add GPU to nightly.
- Explicit `-c 2048` or `-c 4096` on the server.
- Prompt size target: at least 900 prompt tokens, verified via response usage.
- Completion target: 256 to 512 tokens for structured generation.
- Request timeout should be higher than current smoke tests.

Required checks:

- Long needle recall.
- Multi-needle JSON recall.
- Structured long generation with anti-degeneration metrics.
- Long request followed by independent short request to verify cache reset.

### Tier 2: Full Long-Context / Nightly

Purpose: stress the actual target scenario of 1k prompt tokens and 1k generated
tokens across long decode horizons.

Recommended scope:

- `-c 4096` or larger if needed.
- Prompt size target: 1000 to 1500 prompt tokens.
- Completion target: 900 to 1100 tokens.
- Include Qwen3.5 dense or MoE GDN/hybrid-cache models when available.
- Include ROCm for the Qwen3.5 MoE path because many hybrid-cache failures have
  historically surfaced there.

For thinking models, prefer `enable_thinking=false` for deterministic long
content checks, or set a small `thinking_budget_tokens` so the test does not
spend most of its horizon in hidden reasoning.

## Objective Test Oracles

### 1. Long Needle-In-Haystack Recall

Generate a long prompt containing many similar records and one target record:

```text
Record 000: audit code is 104382.
Record 001: audit code is 883019.
...
Record 073: audit code is 492771.
...

Question: Return only the audit code for Record 073.
```

Run target placements near the beginning, middle, and end of the long prompt.

Pass criteria:

- HTTP status is 200.
- `usage.prompt_tokens >= configured_min_prompt_tokens`.
- Assistant content contains the exact target code.
- Assistant content does not contain nearby distractor codes.
- Completion is not empty.
- No server WARN/ERROR lines are produced by the request.

This catches prompt-position bugs, attention mask mistakes, RoPE/position drift,
and cache corruption during long prefill.

### 2. Multi-Needle Strict JSON Recall

Place multiple facts across the prompt and require strict JSON output:

```text
Return exactly this JSON shape and no prose:
{"alpha":"...","middle":"...","omega":"..."}
```

Pass criteria:

- Response content parses as JSON after trimming optional code fences.
- Keys match exactly: `alpha`, `middle`, `omega`.
- Values match the sentinel values placed in beginning, middle, and end prompt
  regions.
- No extra keys are present.

This is stronger than a single needle because partial attention failures often
recover one anchor but lose others.

### 3. Long Structured Generation

Use a prompt that asks for a long, machine-checkable response:

```text
Write 150 numbered lines.
Each line must use this exact format:
001 | one sentence about reliable inference
002 | one sentence about reliable inference
...
End with END_OF_REPORT.
```

For the full tier, request around 1000 tokens. For precommit, request 256 to
512 tokens and lower the required line count accordingly.

Pass criteria:

- `usage.completion_tokens >= min_completion_tokens`, unless the model stops
  naturally after satisfying the full structure.
- `finish_reason` is either `length` for capped long generation or `stop` if the
  requested sentinel is reached.
- At least the configured minimum number of numbered lines is present.
- Line numbers mostly increase and do not reset repeatedly.
- `END_OF_REPORT` appears only when the prompt asks for a complete report and
  `max_tokens` is high enough to allow it.
- No replacement character `U+FFFD` in content.
- No obvious server error strings in content.
- Degeneration metrics pass.

Suggested degeneration metrics:

- No repeated character run longer than 32 characters.
- No repeated word run longer than 12 words.
- Unique 4-gram ratio above 0.20 for responses longer than 200 words.
- Duplicate non-empty line ratio below 0.30.
- The most common 8-token phrase does not occupy more than 20 percent of the
  response.

These thresholds should start loose. The goal is to catch collapse, not grade
style.

### 4. Long Request Followed By Cache-Reset Probe

After a long prompt and long decode, send a second independent short request:

```text
System: You are a calculator. Reply only with the numeric answer.
User: What is 6+7?
```

Pass criteria:

- The second request returns `13`.
- The second request does not contain any sentinel from the long prompt.
- Usage for the second request is small, proving the long conversation did not
  leak into the new request.

This directly targets KV/GDN cache clear and persistent-state corruption bugs.

### 5. Context Boundary Checks

Add one valid near-boundary request and one invalid oversized request.

Valid request pass criteria:

- `usage.context_window == configured_context_length`.
- `usage.context_used` is close to but below `context_window`.
- Request succeeds.

Invalid request pass criteria:

- HTTP status is 400.
- Error type is `invalid_request_error`.
- Error parameter is `messages`, if present.

This catches off-by-one context accounting issues and verifies the server fails
loudly before attempting impossible prefill.

## Suggested Python Helper Interface

The helper can be called by the shell harness after server startup:

```bash
python3 tests/v2/e2e/server/long_context_checks.py \
  --base-url "http://127.0.0.1:${port}" \
  --tag "${tag}" \
  --tier "${LLAMINAR_E2E_LONG_CONTEXT_TIER:-lite}" \
  --min-prompt-tokens 900 \
  --long-max-tokens "${LLAMINAR_E2E_LONG_MAX_TOKENS:-512}" \
  --thinking-model "${thinking_model}"
```

The helper should print concise PASS/FAIL lines and exit non-zero on failure so
the existing Bash counters can either consume its status or wrap it as one
aggregate test.

Request payload defaults:

```json
{
  "messages": [...],
  "max_tokens": 512,
  "temperature": 0.0,
  "enable_thinking": false
}
```

For thinking-model coverage, add a separate optional mode with:

```json
{
  "enable_thinking": true,
  "thinking_budget_tokens": 32
}
```

## Shell Harness Changes

Recommended changes in `tests/v2/e2e/server/test_server_e2e.sh`:

1. Add environment/config variables:

```bash
LONG_CONTEXT_ENABLED="${LLAMINAR_E2E_LONG_CONTEXT:-0}"
LONG_CONTEXT_TIER="${LLAMINAR_E2E_LONG_CONTEXT_TIER:-lite}"
CONTEXT_LENGTH="${LLAMINAR_E2E_CONTEXT_LENGTH:-4096}"
LONG_MAX_TOKENS="${LLAMINAR_E2E_LONG_MAX_TOKENS:-512}"
```

2. Start `serve` with explicit context length when long context is enabled:

```bash
LLAMINAR_LOG_LEVEL="$LOG_LEVEL" "$BINARY" serve --port "$port" \
  -c "$CONTEXT_LENGTH" $device_flag -m "$model" >"$log_path" 2>&1 &
```

3. After the existing streaming/error checks, run the Python helper only when
   enabled:

```bash
if [ "$LONG_CONTEXT_ENABLED" = "1" ]; then
    python3 "$SCRIPT_DIR/long_context_checks.py" \
      --base-url "http://127.0.0.1:${port}" \
      --tag "$tag" \
      --tier "$LONG_CONTEXT_TIER" \
      --min-prompt-tokens 900 \
      --long-max-tokens "$LONG_MAX_TOKENS" \
      --thinking-model "$thinking_model"
fi
```

4. Relax `validate_chat_response_format` or add a separate long-response
   validator that accepts `finish_reason in {"stop", "length"}`.

5. Increase `REQUEST_TIMEOUT` for long tests or add a separate
   `LLAMINAR_E2E_LONG_REQUEST_TIMEOUT`.

## Precommit Integration

Do not immediately make the full 1k/1k tier mandatory in precommit. Suggested
initial precommit wiring:

```bash
LLAMINAR_E2E_LONG_CONTEXT=1 \
LLAMINAR_E2E_LONG_CONTEXT_TIER=lite \
LLAMINAR_E2E_CONTEXT_LENGTH=2048 \
LLAMINAR_E2E_LONG_MAX_TOKENS=256 \
tests/v2/e2e/server/test_server_e2e.sh \
  --binary "$BUILD_V2_INTEGRATION/llaminar2" \
  --backends "cpu"
```

Once runtime is known and stable, broaden to one GPU backend in precommit or
keep GPU long tests in nightly/manual runs.

## Full/Nightly Command Example

```bash
LLAMINAR_LOG_LEVEL=WARN \
LLAMINAR_E2E_LONG_CONTEXT=1 \
LLAMINAR_E2E_LONG_CONTEXT_TIER=full \
LLAMINAR_E2E_CONTEXT_LENGTH=4096 \
LLAMINAR_E2E_LONG_MAX_TOKENS=1000 \
LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_e2e_long_context \
bash tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_integration/llaminar2 \
  --suite "models/qwen2.5-1.5b-instruct-q8_0.gguf|cpu,cuda:0,rocm:0|10" \
  --port 19800
```

For Qwen3.5/GDN/MoE nightly coverage, add explicit suites only when the models
exist locally:

```bash
--suite "models/Qwen3.5-4B-Q8_0.gguf|cpu|200"
--suite "models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|rocm:0|200"
```

## Implementation Checklist

- Add `long_context_checks.py` with deterministic prompt generators and
  validators.
- Add helper functions for POSTing chat completions, extracting content, usage,
  and finish reason.
- Implement needle recall checks for beginning/middle/end target positions.
- Implement multi-needle strict JSON recall.
- Implement structured long generation plus degeneration metrics.
- Implement post-long-request cache-reset probe.
- Implement context boundary success/failure checks.
- Wire optional long-context execution into `test_server_e2e.sh`.
- Add environment variables for tier, context length, max tokens, and timeout.
- Make long-response validation accept expected `length` finish reasons.
- Add focused documentation to the test script header.
- Validate with a lite run before touching precommit.
- Only after runtime is measured, decide whether precommit runs CPU-only or adds
  one GPU backend.

## Notes And Pitfalls

- Avoid using an LLM judge. It makes pass/fail flaky and adds external
  dependencies.
- Do not compare long free-form prose exactly across backends. Use exact checks
  only for structured recall outputs; use length/shape/repetition metrics for
  long prose.
- Verify actual prompt length with `usage.prompt_tokens`; approximating token
  count from words is not reliable enough.
- For capped long generation, `finish_reason == "length"` is often the desired
  evidence that the decode loop reached the requested horizon.
- Keep thresholds loose at first. The test should catch corrupted or collapsed
  generation, not reject merely bland wording.
- After a long request, always run an independent short answer test. That is the
  simplest E2E signal for KV/GDN cache cleanup regressions.