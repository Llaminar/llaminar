# Handover: Qwen3.5 CPU TP E2E Regression Fix

Date: 2026-05-14
Branch context: `feat/qwen35-moe`
Primary goal for next agent: review, keep, and commit the Qwen3.5 dense CPU tensor-parallel regression fix plus the new unit regression coverage. The focused parity and server E2E checks now pass.

## Executive Summary

The Qwen3.5 dense E2E server suite regressed only on CPU shorthand (`-d cpu`), which bootstraps two MPI CPU ranks / NodeLocalTP. Single-socket CPU (`-d cpu:0`) still passed.

The real correctness bug was in GDN recurrence state indexing under TP. Qwen3.5 GDN value heads are not necessarily in the same coordinate system as FA/Q attention heads. The graph used `config_.head_start` as the global V-head offset when `qkv_column_parallel` was active. For Qwen3.5-4B dense, rank 1 has Q-head start 18 but GDN V-head start 16. That made rank 1 read the wrong recurrent state shard, causing early `ATTENTION_OUTPUT` drift and garbage server completions.

A smaller server-template issue was also fixed while investigating: `/v1/chat/completions` parsed `enable_thinking`, but tokenizer chat-template rendering always passed `enable_thinking=true`, so deterministic E2E requests could produce hidden reasoning instead of normal content.

## Current Dirty Tree

Intentional modified files in this slice:

- `src/v2/app/modes/ChatCompletionHandler.cpp`
- `src/v2/models/qwen35/Qwen35Graph.cpp`
- `src/v2/models/qwen35/Qwen35Graph.h`
- `src/v2/models/qwen35/Qwen35Schema.h`
- `src/v2/utils/Tokenizer.cpp`
- `src/v2/utils/Tokenizer.h`
- `tests/v2/e2e/server/test_server_e2e.sh`
- `tests/v2/unit/loaders/Test__WeightSlicer.cpp`
- `tests/v2/unit/models/qwen35/Test__Qwen35Schema.cpp`
- `docs/v2/HANDOVER_QWEN35_CPU_TP_REGRESSION_2026-05-14.md` (this file)

There may be unrelated user or prior-agent changes elsewhere. Do not revert unrelated files.

## Reproduction Before the Fix

Focused server E2E that initially failed:

```bash
tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_integration/llaminar2 \
  --suite "$PWD/models/Qwen3.5-4B-Q8_0.gguf|cpu|200" \
  --port 19180
```

Initial symptoms:

- `-d cpu` failed 3/10 checks.
- Single-turn and multi-turn responses had empty `content` and populated `reasoning_content` before the template plumbing fix.
- After `enable_thinking=false` was correctly wired, responses still produced wrong content on `-d cpu`.
- The same suite with `cpu:0` passed 10/10, proving the server/model path could work on single-device CPU.

Focused parity that initially failed:

```bash
ctest --test-dir build_v2_integration \
  -R "Qwen35NodeLocalTPParityTest_PrefillParity_NodeLocalTP_2xMPI_CPU_4B$" \
  --output-on-failure -V
```

Failure signature before the fix:

```text
Early layers passed: 1/6
LM_HEAD cosine ~= 0.608880
LM_HEAD KL ~= 4.0315
Layer 0 avg cosine ~= 0.816
Layer 0 ATTENTION_OUTPUT cosine ~= 0.950
Layer 0 FFN_DOWN cosine ~= 0.550
```

The 0.8B dense NodeLocalTP prefill parity also failed hard, so this was not specific to the 4B weights.

## Root Cause

Qwen3.5 GDN layers have independent GDN value-head dimensions. For 4B dense:

```text
FA/Q attention heads: 36
GDN key heads:       16
GDN value heads:     32
d_state:             128
```

In TP=2:

```text
rank 0 Q heads: [0,18),  GDN V heads: [0,16)
rank 1 Q heads: [18,36), GDN V heads: [16,32)
```

The graph used the Q-head start (`18`) as `global_v_head_offset` on rank 1, but recurrence state indexing needed the V-head start (`16`). This corrupted the GDN recurrent-state lookup only in cross-rank TP paths.

The relevant code was in `Qwen35Graph::buildGDNAttentionGraph()` around `GDNRecurrenceStage::Params::global_v_head_offset`.

## Implemented Fixes

### 1. GDN V-Head Offset Resolution

Files:

- `src/v2/models/qwen35/Qwen35Graph.h`
- `src/v2/models/qwen35/Qwen35Graph.cpp`

Added `Qwen35Graph::resolveGDNGlobalVHeadOffset(...)` as a pure helper and routed graph construction through it.

Resolution order:

1. Prefer frozen binding slice metadata from the value projection (`attn_gate.weight`): `slice.row_start / d_v`.
2. Fall back to proportional mapping from Q-head assignment into V-head space when `GraphConfig::tp_config` exists.
3. Fall back to `mpi_ctx->rank() * n_v_heads` for older global TP contexts.
4. Fall back to `tp_ctx->myIndex() * n_v_heads` for local TP contexts without binding metadata.

This prevents using `GraphConfig::head_start` directly for GDN recurrence state selection.

### 2. Qwen3.5 GDN Value-Head Sharding Metadata

File:

- `src/v2/models/qwen35/Qwen35Schema.h`

Changed GDN value-head-sized weights from `WeightDimensionType::Heads` to `WeightDimensionType::ProportionalHeads`:

- `attn_gate.weight`: `ColumnParallel + ProportionalHeads`
- `ssm_out.weight`: `InputParallel + ProportionalHeads`

This matches the existing `ssm_alpha`, `ssm_beta`, `ssm_dt.bias`, and `ssm_a` treatment and keeps value-head slices aligned to `d_state`.

### 3. Chat Template `enable_thinking` Plumbing

Files:

- `src/v2/utils/Tokenizer.h`
- `src/v2/utils/Tokenizer.cpp`
- `src/v2/app/modes/ChatCompletionHandler.cpp`
- `tests/v2/e2e/server/test_server_e2e.sh`

Added overloads for:

- `ITokenizer::encodeChat(..., bool enable_thinking)`
- `ITokenizer::applyTemplate(..., bool enable_thinking)`
- `BPETokenizer` implementations of both

The old overloads still default to `true` for compatibility.

`ChatCompletionHandler::setupInference()` now passes `request.enable_thinking` into tokenizer rendering.

The E2E server script now sends `"enable_thinking": false` for deterministic arithmetic and streaming checks.

### 4. Unit Regression Tests

Files:

- `tests/v2/unit/models/qwen35/Test__Qwen35Schema.cpp`
- `tests/v2/unit/loaders/Test__WeightSlicer.cpp`

New Qwen3.5 schema tests pin:

- `attn_gate.weight` sharding as `ColumnParallel + ProportionalHeads`
- `ssm_out.weight` sharding as `InputParallel + ProportionalHeads`
- `resolveGDNGlobalVHeadOffset()` prefers the frozen value-projection slice (`16`) over stale Q-head start (`18`)
- fallback mapping converts Q-head assignment into V-head coordinate space

New WeightSlicer test pins Qwen3.5-4B TP=2 value-head slicing:

- rank 1 `attn_gate.weight` starts at `16 * 128`
- rank 1 `ssm_out.weight` starts at `16 * 128`
- both starts are aligned to `d_state`
- both differ from the stale `head_start * (4096 / 36)` calculation

## Verification Completed

Build and unit regression tests:

```bash
cmake --build build_v2_integration \
  --target v2_test_qwen35_schema v2_unit_weight_slicer \
  --parallel

ctest --test-dir build_v2_integration \
  -R "^V2_Unit_(Qwen35Schema|WeightSlicer)$" \
  --output-on-failure --parallel
```

Result:

```text
100% tests passed, 0 tests failed out of 3
```

Focused parity checks:

```bash
cmake --build build_v2_integration --parallel

ctest --test-dir build_v2_integration \
  -R "Qwen35NodeLocalTPParityTest_PrefillParity_NodeLocalTP_2xMPI_CPU_4B$" \
  --output-on-failure -V

ctest --test-dir build_v2_integration \
  -R "Qwen35NodeLocalTPParityTest_PrefillParity_NodeLocalTP_2xMPI_CPU_08B$" \
  --output-on-failure -V
```

Final 4B result after the helper cleanup:

```text
Layer 0 avg cosine: 0.999782
LM_HEAD cosine:     0.998205
LM_HEAD KL:         0.0037
Top1:               100.0%
Test passed
```

0.8B also passed:

```text
LM_HEAD cosine: 0.998949
LM_HEAD KL:     0.0024
Top1:           100.0%
Test passed
```

Final server E2E:

```bash
tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_integration/llaminar2 \
  --suite "$PWD/models/Qwen3.5-4B-Q8_0.gguf|cpu|200" \
  --port 19230
```

Result:

```text
ALL PASSED: 10/10 tests passed
```

Final hygiene:

```bash
git diff --check
```

Result: clean.

VS Code diagnostics reported no errors in the touched files.

## Useful Artifacts

Generated parity CSVs were written under the current commit hash directory:

```text
tests/v2/integration/parity/results/fb9a5b26/Qwen35NodeLocalTP_Qwen35NodeLocalTPParityTest_PrefillParity_NodeLocalTP_2xMPI_CPU_4B/
tests/v2/integration/parity/results/fb9a5b26/Qwen35NodeLocalTP_Qwen35NodeLocalTPParityTest_PrefillParity_NodeLocalTP_2xMPI_CPU_08B/
```

Useful files:

```text
prefill_layers.csv
prefill_summary.csv
prefill_stages.csv
test_log.txt
```

Temporary logs used during investigation may still exist:

```text
/tmp/qwen35_nodelocal_4b_after_offset.log
/tmp/qwen35_nodelocal_08b_after_offset.log
/tmp/qwen35_nodelocal_4b_final.log
/tmp/qwen35_4b_cpu_e2e_final.log
```

Do not depend on `/tmp` logs surviving across sessions.

## Suggested Next Steps

1. Review the diff as one logical patch.

   The Qwen3.5 TP fix and the `enable_thinking` plumbing are related by the same failing E2E reproduction, but they are separable if you prefer two commits.

2. Optionally run a broader unit sweep before commit:

   ```bash
   ctest --test-dir build_v2_integration \
     -R "^V2_Unit_" \
     --output-on-failure --parallel
   ```

3. Optionally rerun the focused server E2E on `cpu:0` as a sanity control:

   ```bash
   tests/v2/e2e/server/test_server_e2e.sh \
     --binary build_v2_integration/llaminar2 \
     --suite "$PWD/models/Qwen3.5-4B-Q8_0.gguf|cpu:0|200" \
     --port 19240
   ```

   This already passed during investigation before the final patch, and the important regression target is `-d cpu` / NodeLocalTP.

4. Commit without reverting unrelated files.

   Suggested commit split if desired:

   - `fix(v2): align qwen35 gdn tp value-head offsets`
   - `fix(v2): honor chat enable_thinking template flag`

## Cautions for the Next Agent

- Do not replace the value-head offset helper with `config_.head_start`. That reintroduces the regression.
- `attn_gate.weight` exists on both GDN and FA layers; this fix is specifically about the GDN Z/value projection path where output dim is `n_v_heads * d_state`.
- `ProportionalHeads` is intentional for GDN value-head-sized weights because `n_v_heads` may differ from `n_heads`.
- Single-device CPU passing is not enough coverage for this bug. Always verify a NodeLocalTP/GlobalTP path (`-d cpu` or the NodeLocalTP parity test).
- The E2E script now sends `enable_thinking=false`; if that is removed, Qwen3.5 can put arithmetic answers in `reasoning_content` and make content-based checks misleading.
- Keep using the Integration build for parity/unit verification. Do not artificially limit build or test parallelism.
