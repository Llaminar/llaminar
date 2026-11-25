# Serving-step replay fixture

`qwen2.5-0.5b-instruct-q4_0.cpu.trace` is a recorded trace of the production
`ServingSession` (continuous batching) on the real model, consumed by the
device-free recorded-input replay `V2_Unit_ServingStepReplay` (also the
`ProductionTestPreflight` entry `V2_Integration_ServingStepReplay`).

Format and exactness rules: `tests/v2/utils/ServingStepTrace.h` (commands,
admissions with resolved seeds, every shared forward with each row's top-64
logits as hex floats, every step's events).

Provenance:

- model: `qwen2.5-0.5b-instruct-q4_0.gguf`
  (sha256 `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed`),
  CPU, context 1024, capacity 3, prefix cache off;
- recorded by `ContinuousBatchingCPU.RecordedServingTraceReplaysExactly`
  (which replays its own recording) on source `bc3f1f527` plus the
  one-chunk-per-prefill-transaction change, Docker Desktop aarch64:

  ```bash
  LLAMINAR_CB_RECORD_TRACE=<path> LLAMINAR_CB_TEST_MODEL=<model> \
    v2_integration_continuous_batching_cpu \
    --gtest_filter='*RecordedServingTraceReplaysExactly*'
  ```

Re-record only when the session's step contract changes deliberately; a replay
failure otherwise means the production step logic changed behaviour.
