# Proposal: Coherence System Hardening

**Date**: April 1, 2026  
**Status**: ✅ **IMPLEMENTED** (April 2, 2026)  
**Context**: The GPU token duplication bug (commit `74de820e`) had a **two-part root cause** that took extensive manual instrumentation to find. Both parts were "silent" — no error, no crash, just wrong output. This proposal addresses three questions:

1. How to make this class of bugs structurally impossible
2. How to make them fast to diagnose if they do occur
3. What regression tests to add

---

## Implementation Summary

| Priority | Item | Status | Files Changed |
|----------|------|--------|---------------|
| **P0** | allocateOnDevice state preservation tests | ✅ | `tests/v2/unit/tensors/Test__TensorAllocateOnDevice.cpp` (+2 tests) |
| **P0** | mark_dirty always-true tests | ✅ | `tests/v2/unit/execution/.../Test__StageRunPolicy.cpp` (+2 tests) |
| **P1** | Make mark_dirty unconditional in executor | ✅ | `DeviceGraphExecutor.cpp`, `DeviceGraphExecutor.h` |
| **P1** | Arena/tensor state mismatch assertion | ✅ | `memory/CoherenceTracker.cpp` |
| **P2** | Validated state transitions (applyCoherenceOp_) | ✅ | `TensorClasses.h`, `TensorBase.cpp`, `TransferEngine.cpp` |
| **P2** | Coherence audit log (ring buffer) | ✅ | `tensors/CoherenceAuditLog.h` (new), `TensorClasses.h`, `DebugEnv.h` |
| **P2** | CoherenceAuditLog unit test | ✅ | `tests/v2/unit/tensors/Test__CoherenceAuditLog.cpp` (new, 10 tests) |

**Migration summary**: 10 `setCoherenceState_()` calls migrated to `applyCoherenceOp_()` across `TensorBase.cpp` (5) and `TransferEngine.cpp` (5). ~8 calls remain as raw `setCoherenceState_()` for residency-mode changes (MAPPED init, BAR init, fresh allocation, secondary promote) where no single `CoherenceOp` maps cleanly.

**Test results**: 341/341 unit tests pass (was 340 — new CoherenceAuditLog test added 1).

---

## Table of Contents

- [Root Cause Recap](#root-cause-recap)
- [Bug Classification](#bug-classification)
- [Part 1: Architectural Hardening](#part-1-architectural-hardening)
  - [1A. Validated State Transitions (eliminate raw setCoherenceState\_)](#1a-validated-state-transitions)
  - [1B. Mandatory Mark-Dirty — decouple from coherence flag](#1b-mandatory-mark-dirty)
  - [1C. Compile-Time Policy Coupling via static_assert](#1c-compile-time-policy-coupling)
- [Part 2: Tracing and Diagnostic Infrastructure](#part-2-tracing-and-diagnostic-infrastructure)
  - [2A. Coherence Audit Log (ring buffer)](#2a-coherence-audit-log)
  - [2B. Round-Trip Canary](#2b-round-trip-canary)
  - [2C. Cross-Step Staleness Detector](#2c-cross-step-staleness-detector)
- [Part 3: Regression Tests](#part-3-regression-tests)
  - [3A. Unit: allocateOnDevice preserves state](#3a-allocateondevice-preserves-state)
  - [3B. Unit: mark_dirty independent of coherence](#3b-mark_dirty-independent-of-coherence)
  - [3C. Integration: fast-decode staleness canary](#3c-fast-decode-staleness-canary)
  - [3D. Integration: multi-step logit divergence](#3d-multi-step-logit-divergence)
- [Implementation Priority](#implementation-priority)

---

## Root Cause Recap

**Bug 1 — `allocateOnDevice` state reset**: `TensorBase::allocateOnDevice()` unconditionally set `HOST_AUTHORITATIVE` for already-allocated buffers. This was called by `prepareForWrite()` every step. Result: output tensors that were correctly `DEVICE_AUTHORITATIVE` got reset to `HOST_AUTHORITATIVE`, and the next `prepareForRead()` uploaded stale host data over correct GPU results.

**Bug 2 — `mark_dirty` silently gated on `coherence`**: The `StageBufferContract` was only computed when `policy.coherence=true`. The `mark_dirty` block checked `use_contract`, which was false when `coherence=false`. So `fastDecode()` with `mark_dirty=true` was a no-op — outputs were never marked `DEVICE_AUTHORITATIVE`.

Both bugs were **completely silent**. No assertions fired, no logs, no crashes. The only symptom was identical logits across decode steps — the model repeated the same token endlessly.

---

## Bug Classification

These bugs belong to two distinct classes:

| Class | Description | Instance |
|-------|-------------|----------|
| **Invalid state transition** | Raw `setCoherenceState_()` bypasses the transition table, allowing impossible state jumps | `allocateOnDevice` setting `HOST_AUTHORITATIVE` on a `DEVICE_AUTHORITATIVE` tensor |
| **Policy coupling violation** | A feature flag silently depends on another flag's side effects, not on explicit preconditions | `mark_dirty` depending on `coherence` producing a non-empty contract |

---

## Part 1: Architectural Hardening

### 1A. Validated State Transitions

**Problem**: `setCoherenceState_()` is a raw setter with 30 call sites. It bypasses the `constexpr` transition table in `CoherenceState.h`. Any caller can set any state from any state — the transition table exists but is advisory.

**Solution**: Replace `setCoherenceState_()` with `applyCoherenceOp_()` that routes through the transition table, and fail hard on invalid transitions.

```cpp
// TensorClasses.h — replace the raw setter
void applyCoherenceOp_(CoherenceOp op)
{
    auto result = coherenceTransition(coherence_state_, op);
    if (!result.valid)
    {
        // In Debug/Integration: hard fail with full context
        LOG_ERROR("[COHERENCE] Invalid transition: " 
                  << toString(coherence_state_) << " + " << toString(op)
                  << " → INVALID  tensor=" << (void*)this
                  << " shape=" << shapeString());
#if LLAMINAR_ASSERTIONS_ACTIVE
        LLAMINAR_UNREACHABLE("Invalid coherence transition: " 
                             << toString(coherence_state_) << " + " << toString(op));
#endif
        return; // In Release: log + no-op (defensive)
    }
    coherence_state_ = result.new_state;
}
```

**Migration**: Each of the ~30 `setCoherenceState_` call sites maps to a specific `CoherenceOp`:

| Current call | Maps to |
|---|---|
| `setCoherenceState_(SYNCED)` (after upload) | `applyCoherenceOp_(CoherenceOp::UPLOAD)` |
| `setCoherenceState_(SYNCED)` (after download) | `applyCoherenceOp_(CoherenceOp::DOWNLOAD)` |
| `setCoherenceState_(DEVICE_AUTHORITATIVE)` | `applyCoherenceOp_(CoherenceOp::MARK_DEVICE_DIRTY)` |
| `setCoherenceState_(HOST_AUTHORITATIVE)` (mutable host access) | `applyCoherenceOp_(CoherenceOp::MUTABLE_HOST_ACCESS)` |
| `setCoherenceState_(HOST_ONLY)` (release device) | `applyCoherenceOp_(CoherenceOp::RELEASE_DEVICE)` |
| `setCoherenceState_(MAPPED)` | Keep as raw set — `MAPPED` is a residency mode, not a transition |

**Key property**: This would have caught Bug 1. `allocateOnDevice()` reuse-path was doing `setCoherenceState_(HOST_AUTHORITATIVE)` on a `DEVICE_AUTHORITATIVE` tensor. There is no single `CoherenceOp` that makes this transition — it would require `MUTABLE_HOST_ACCESS` (which implies host data was actually modified), and the transition table correctly routes `DEVICE_AUTHORITATIVE + MUTABLE_HOST_ACCESS → SYNCED` (with implicit download). The caller would have to choose an op, and none of them silently produce `HOST_AUTHORITATIVE` from `DEVICE_AUTHORITATIVE` without a download.

**Escape hatch for allocateOnDevice specifically**: The reuse path now correctly does *nothing* to coherence state. Under the new system this is explicit — `allocateOnDevice()` simply doesn't call `applyCoherenceOp_` when the buffer already exists. The fresh-allocation path calls `applyCoherenceOp_(MUTABLE_HOST_ACCESS)` because the new GPU buffer has undefined contents and the host data is authoritative.

### 1B. Mandatory Mark-Dirty

**Problem**: `mark_dirty` was silently gated on the `coherence` flag because the contract was only computed when `coherence=true`. The fix (computing contract when `coherence || mark_dirty`) is correct but fragile — it's a non-obvious coupling that could regress.

**Solution**: Make `mark_dirty` not optional. Every execution path that runs GPU stages MUST mark outputs. Redefine the policy:

```cpp
struct StageRunPolicy
{
    bool input_coherence = true;   // Upload inputs, allocate outputs
    bool weight_coherence = true;  // Upload weights
    // mark_dirty is REMOVED as a toggle — it is always-on for GPU stages
    bool validation = true;        // NaN/Inf output validation
    bool profiling = true;         // Per-stage timing
    ...
};
```

The `runStage()` contract computation becomes unconditional when an arena exists:

```cpp
// Contract is ALWAYS fetched when we have an arena.
// input_coherence controls whether prepareForRead/prepareForWrite are called.
// Marking outputs dirty is unconditional — this is correctness, not overhead.
const StageBufferContract contract = arena_ 
    ? node.stage->bufferContract() 
    : StageBufferContract{};
const bool use_contract = !contract.empty() && arena_ != nullptr;

// After execute:
if (success && use_contract)
{
    // Always mark outputs — this is O(N) flag updates, zero data transfer
    for (const auto &binding : contract.allWrites())
    {
        if (need_event)
            arena_->markWritten(binding.id, target_device, stream);
        else
            arena_->markWrittenFlagsOnly(binding.id, target_device);
    }
}
```

**Key property**: This eliminates Bug 2 structurally. There is no combination of policy flags that can prevent outputs from being marked. The overhead of `markWrittenFlagsOnly()` is negligible (~10ns per buffer, ~30 buffers per layer = ~300ns/layer — invisible vs. the ~2ms decode step).

### 1C. Compile-Time Policy Coupling

**Problem**: Policy flags have implicit dependencies. `mark_dirty` depended on `coherence` through the contract, which was not expressed in the type system.

**Solution**: Add a `constexpr` self-check to `StageRunPolicy`:

```cpp
struct StageRunPolicy
{
    ...
    
    /// Compile-time sanity check.
    constexpr void validate() const
    {
        // If we want validation, we need some form of coherence so outputs
        // are accessible on host:
        if (validation && !input_coherence)
        {
            // This is fine — validation reads outputs via data() which
            // triggers D2H. But worth documenting.
        }
    }
    
    static StageRunPolicy fastDecode()
    {
        StageRunPolicy p;
        p.input_coherence = false;  // Buffers already on device
        p.weight_coherence = false; // Weights already uploaded
        p.validation = false;
        p.profiling = false;
        p.stage_dump = false;
        p.snapshot_callback = false;
        p.pointer_validation = false;
        p.timeline = true;
        // NOTE: mark_dirty is not a toggle — always happens
        return p;
    }
};
```

---

## Part 2: Tracing and Diagnostic Infrastructure

### 2A. Coherence Audit Log

**Problem**: The only way to diagnose the bug was to add `fprintf` to `setCoherenceState_()` and `CoherenceTracker::prepareForRead()`, grep through thousands of lines, and manually reason about state transitions. This took multiple iterations.

**Solution**: A per-tensor ring buffer that records the last N coherence operations, enabled by environment variable. Zero overhead when disabled.

```cpp
// CoherenceAuditLog.h

struct CoherenceAuditEntry
{
    TensorCoherenceState from_state;
    TensorCoherenceState to_state;
    CoherenceOp op;          // Which operation caused this
    const char *caller;       // __func__ or explicit label
    uint64_t timestamp_ns;    // std::chrono steady_clock
    uint32_t thread_id;
};

// Per-tensor ring buffer, compile-time controlled
class CoherenceAuditLog
{
public:
    static constexpr size_t RING_SIZE = 32;  // Last 32 transitions per tensor
    
    void record(TensorCoherenceState from, TensorCoherenceState to,
                CoherenceOp op, const char *caller);
    
    // Dump to stderr on demand (e.g., when verification fails)
    void dump(const char *tensor_name, void *tensor_ptr) const;
    
    // Check if a specific transition pattern occurred
    bool hasTransition(TensorCoherenceState from, TensorCoherenceState to) const;
    
private:
    std::array<CoherenceAuditEntry, RING_SIZE> ring_{};
    size_t head_ = 0;
    size_t count_ = 0;
};
```

**Integration with `applyCoherenceOp_`**:

```cpp
void applyCoherenceOp_(CoherenceOp op, const char *caller = __builtin_FUNCTION())
{
    auto from = coherence_state_;
    auto result = coherenceTransition(coherence_state_, op);
    
#if LLAMINAR_ASSERTIONS_ACTIVE
    if (debugEnv().coherence.audit_log)
    {
        coherence_log_.record(from, result.new_state, op, caller);
    }
    
    if (!result.valid)
    {
        coherence_log_.dump(name_.c_str(), this);
        LLAMINAR_UNREACHABLE("Invalid coherence transition");
    }
#endif
    
    coherence_state_ = result.new_state;
}
```

**Activation**: `LLAMINAR_COHERENCE_AUDIT=1`. When the tensor verification system detects a NaN/staleness/zero, it automatically dumps the audit log for all involved tensors.

**Integration with TensorVerification**: When a `VerificationFailure` is thrown (NaN detected, zero tensor, etc.), the handler dumps the coherence audit logs for all stage inputs and outputs. This gives immediate visibility into *how* the tensor arrived at its current state:

```
╔══════════════════════════════════════════════════════════════╗
║               COHERENCE AUDIT LOG                            ║
║ Tensor: attn_output (0x7f1234)                               ║
╠══════════════════════════════════════════════════════════════╣
║ #28  HOST_ONLY → SYNCED         UPLOAD        TensorBase::ensureOnDevice  t=0.000ms
║ #29  SYNCED → DEVICE_AUTH       MARK_DIRTY    markWrittenFlagsOnly        t=2.341ms
║ #30  DEVICE_AUTH → HOST_AUTH    MUTABLE_HOST  allocateOnDevice  ← BUG!   t=4.102ms
║ #31  HOST_AUTH → SYNCED         UPLOAD        prepareForRead              t=4.103ms
╚══════════════════════════════════════════════════════════════╝
```

Entry #30 would immediately reveal the `allocateOnDevice` bug: a `DEVICE_AUTHORITATIVE → HOST_AUTHORITATIVE` transition labeled `MUTABLE_HOST_ACCESS` in `allocateOnDevice` — the function name alone makes the problem obvious.

### 2B. Round-Trip Canary

**Problem**: The token duplication manifested as *identical logits across decode steps*. This is an observable invariant violation: in autoregressive decoding, consecutive steps MUST produce different logit distributions (because different positions and different input tokens are processed).

**Solution**: A lightweight staleness check in the decode loop that compares logit fingerprints between consecutive steps.

```cpp
// StalenessCanary.h

class StalenessCanary
{
public:
    /// Call after each decode step with the logit tensor.
    /// Returns true if logits are suspiciously similar to the previous step.
    bool checkForStaleness(const float *logits, size_t vocab_size)
    {
        // Compute a fast fingerprint: hash of first 64 + last 64 logits + argmax
        uint64_t fingerprint = computeFingerprint(logits, vocab_size);
        
        bool stale = (step_count_ > 0 && fingerprint == prev_fingerprint_);
        if (stale)
        {
            consecutive_stale_++;
            if (consecutive_stale_ >= 2)
            {
                LOG_ERROR("[STALENESS_CANARY] Logits unchanged for " 
                          << consecutive_stale_ << " consecutive decode steps "
                          << "— probable coherence bug. "
                          << "fingerprint=0x" << std::hex << fingerprint
                          << " step=" << step_count_);
#if LLAMINAR_ASSERTIONS_ACTIVE
                // Dump coherence audit logs for all arena buffers
                if (arena_) arena_->dumpCoherenceAuditLogs();
                LLAMINAR_ASSERT(false, "Stale logits detected — see coherence audit log above");
#endif
            }
        }
        else
        {
            consecutive_stale_ = 0;
        }
        
        prev_fingerprint_ = fingerprint;
        step_count_++;
        return stale;
    }
    
private:
    uint64_t prev_fingerprint_ = 0;
    size_t step_count_ = 0;
    int consecutive_stale_ = 0;
    
    static uint64_t computeFingerprint(const float *logits, size_t n)
    {
        // FNV-1a of first 64 floats + last 64 floats + argmax position
        // This is O(128) — negligible vs decode step cost
        uint64_t hash = 14695981039346656037ULL;
        size_t sample_count = std::min(n, size_t(64));
        
        for (size_t i = 0; i < sample_count; ++i)
            hash = fnv1a_update(hash, logits[i]);
        for (size_t i = n - sample_count; i < n; ++i)
            hash = fnv1a_update(hash, logits[i]);
            
        // Include argmax position
        size_t argmax = 0;
        float max_val = logits[0];
        for (size_t i = 1; i < n; ++i)
        {
            if (logits[i] > max_val) { max_val = logits[i]; argmax = i; }
        }
        hash ^= argmax;
        hash *= 1099511628211ULL;
        
        return hash;
    }
};
```

**Integration point**: The canary lives in `DeviceGraphOrchestrator` (or `ForwardExecutionEngine`) and is checked after each decode step's logit tensor is ready. In Debug/Integration builds it asserts; in Release it logs a warning.

**Why this catches the bug class**: Token duplication is the *only observable symptom* of coherence bugs that silently replay stale data. If the canary fires, you know immediately that something is replaying old data, and the coherence audit log tells you exactly which tensor and which transition went wrong.

### 2C. Cross-Step Staleness Detector

**Problem**: Bug 1 was specifically about `prepareForWrite` + `allocateOnDevice` resetting state. A more targeted detector can catch this pattern directly.

**Solution**: In Debug/Integration builds, `CoherenceTracker::prepareForRead()` can verify that the tensor's coherence state is consistent with the arena's expectation:

```cpp
// CoherenceTracker.cpp — enhanced prepareForRead

bool CoherenceTracker::prepareForRead(TensorBase *tensor, CoherenceState &state, 
                                       DeviceId target)
{
    if (!tensor || !state.isActive())
        return true;

#if LLAMINAR_ASSERTIONS_ACTIVE
    // Staleness invariant: if arena says DEVICE is authoritative, the tensor
    // MUST NOT be HOST_AUTHORITATIVE. If it is, something reset the tensor's
    // state without going through the arena (e.g., allocateOnDevice bug).
    if (state.authority == Authority::DEVICE && 
        tensor->coherenceState() == TensorCoherenceState::HOST_AUTHORITATIVE)
    {
        LOG_ERROR("[COHERENCE_INVARIANT] Arena says DEVICE authoritative but tensor is "
                  "HOST_AUTHORITATIVE — coherence state was reset outside arena control. "
                  "tensor=" << (void*)tensor << " target=" << target.to_string());
        tensor->dumpCoherenceAuditLog();
        LLAMINAR_ASSERT(false, "Arena/tensor coherence state mismatch — see audit log");
    }
#endif

    if (!state.needsTransferTo(target, tensor->coherenceState()))
        return true;

    // ... existing transfer logic
}
```

**Key property**: This assertion directly encodes the invariant that Bug 1 violated: "if the arena says a buffer was last written by the GPU, the tensor must agree." It fires at the exact moment stale data would be uploaded, not after the damage is done.

---

## Part 3: Regression Tests

### 3A. allocateOnDevice preserves state

**File**: `tests/v2/unit/tensors/Test__TensorAllocateOnDevice.cpp` (extend existing)

```cpp
// Test that allocateOnDevice does NOT reset DEVICE_AUTHORITATIVE state
// when the buffer is already allocated on the target device.
// This is the regression test for Bug 1 (commit 74de820e).
TEST(Test__TensorAllocateOnDevice, ReusedAllocationPreservesDeviceAuthoritativeState)
{
    auto tensor = TestTensorFactory::createFP32({32, 32});
    DeviceId gpu = firstAvailableGPU();
    if (!gpu.is_gpu()) GTEST_SKIP() << "No GPU available";
    
    // First allocation
    ASSERT_TRUE(tensor->allocateOnDevice(gpu));
    
    // Simulate: GPU kernel wrote to this buffer
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, gpu);
    ASSERT_EQ(tensor->coherenceState(), TensorCoherenceState::DEVICE_AUTHORITATIVE);
    
    // Second allocateOnDevice (reuse path) — must NOT reset state
    ASSERT_TRUE(tensor->allocateOnDevice(gpu));
    EXPECT_EQ(tensor->coherenceState(), TensorCoherenceState::DEVICE_AUTHORITATIVE)
        << "allocateOnDevice must not reset DEVICE_AUTHORITATIVE for reused buffers";
}

// Same test for SYNCED state
TEST(Test__TensorAllocateOnDevice, ReusedAllocationPreservesSyncedState)
{
    auto tensor = TestTensorFactory::createFP32({32, 32});
    DeviceId gpu = firstAvailableGPU();
    if (!gpu.is_gpu()) GTEST_SKIP() << "No GPU available";
    
    ASSERT_TRUE(tensor->ensureOnDevice(gpu));  // Upload → SYNCED
    ASSERT_EQ(tensor->coherenceState(), TensorCoherenceState::SYNCED);
    
    ASSERT_TRUE(tensor->allocateOnDevice(gpu));
    EXPECT_EQ(tensor->coherenceState(), TensorCoherenceState::SYNCED)
        << "allocateOnDevice must not reset SYNCED for reused buffers";
}
```

### 3B. mark_dirty independent of coherence

**File**: `tests/v2/unit/execution/local_execution/graph/Test__StageRunPolicy.cpp` (extend existing)

This tests the actual executor code path, not just flag values. Uses a mock arena and mock stage to verify that `runStage()` marks outputs dirty regardless of the `coherence` flag.

```cpp
// Regression test for Bug 2: mark_dirty must work when coherence=false.
// Verifies that the contract is computed and outputs are marked dirty
// even when input_coherence is disabled.
TEST(Test__StageRunPolicy, MarkDirtyWorksWithoutCoherence)
{
    // Setup: mock arena with a single output buffer  
    auto arena = std::make_shared<MockBufferArena>();
    BufferId output_id = arena->registerBuffer("test_output", ...);
    
    // Mock stage that declares the output in its contract
    auto stage = std::make_shared<MockStageWithContract>();
    stage->setContract(StageBufferContract::build()
        .addOutput(output_id)
        .done());
    
    // Build graph with single node
    ComputeGraph graph;
    graph.addNode("test_stage", stage);
    
    // Execute with coherence=false but mark_dirty=true (fastDecode policy)
    DeviceGraphExecutor executor;
    executor.setArena(arena);
    
    auto policy = StageRunPolicy::fastDecode();
    ASSERT_FALSE(policy.coherence);  // Precondition: no input coherence
    ASSERT_TRUE(policy.mark_dirty);  // Precondition: marking enabled
    
    executor.runStage(graph.getNode("test_stage"), ctx, policy);
    
    // Verify: output was marked written
    EXPECT_TRUE(arena->wasMarkedWritten(output_id))
        << "Output must be marked dirty even when coherence=false";
}
```

### 3C. Integration: fast-decode staleness canary

**File**: `tests/v2/integration/execution/coherence/Test__FastDecodeCoherence.cpp` (new)

This is the highest-value regression test — it directly reproduces the end-to-end symptom of the token duplication bug.

```cpp
// End-to-end test: run 3 decode steps on GPU and verify that logit
// fingerprints differ between steps. This catches any coherence bug
// that causes the model to replay the same computation.
TEST(Test__FastDecodeCoherence, LogitsDivergeAcrossDecodeSteps)
{
    auto model = loadModel("qwen2.5-0.5b-instruct-q4_0.gguf");
    DeviceId gpu = firstAvailableGPU();
    if (!gpu.is_gpu()) GTEST_SKIP() << "No GPU available";
    
    auto runner = InferenceRunnerFactory::create(model, gpu);
    
    // Prefill
    auto prompt_tokens = tokenize("What is the capital of France?");
    runner->prefill(prompt_tokens);
    
    // Decode 3 steps, collect logit fingerprints
    std::vector<uint64_t> fingerprints;
    for (int step = 0; step < 3; ++step)
    {
        auto logits = runner->decodeStep();
        fingerprints.push_back(StalenessCanary::computeFingerprint(
            logits.data(), logits.size()));
    }
    
    // Each step MUST produce different logits (autoregressive property)
    EXPECT_NE(fingerprints[0], fingerprints[1])
        << "Decode steps 0 and 1 produced identical logits — coherence bug";
    EXPECT_NE(fingerprints[1], fingerprints[2])
        << "Decode steps 1 and 2 produced identical logits — coherence bug";
}
```

### 3D. Integration: multi-step logit divergence

**File**: `tests/v2/integration/execution/coherence/Test__FastDecodeCoherence.cpp` (same file)

A more targeted version that compares fast-decode vs full-coherence results:

```cpp
// Compare outputs between full() and fastDecode() policies.
// They should produce the same tokens (within GPU non-determinism).
// If fastDecode produces different/repeated tokens, the fast path
// has a coherence bug.
TEST(Test__FastDecodeCoherence, FastDecodeMatchesFullCoherence)
{
    auto model = loadModel("qwen2.5-0.5b-instruct-q4_0.gguf");
    DeviceId gpu = firstAvailableGPU();
    if (!gpu.is_gpu()) GTEST_SKIP() << "No GPU available";
    
    auto prompt_tokens = tokenize("The capital of France is");
    
    // Run with full coherence (ground truth)
    std::vector<int> full_tokens;
    {
        auto runner = InferenceRunnerFactory::create(model, gpu);
        runner->setPolicy(StageRunPolicy::full());
        runner->prefill(prompt_tokens);
        for (int i = 0; i < 5; ++i)
            full_tokens.push_back(runner->decodeSample());
    }
    
    // Run with fast decode (the path being tested)
    std::vector<int> fast_tokens;
    {
        auto runner = InferenceRunnerFactory::create(model, gpu);
        runner->setPolicy(StageRunPolicy::fastDecode());
        runner->prefill(prompt_tokens);
        for (int i = 0; i < 5; ++i)
            fast_tokens.push_back(runner->decodeSample());
    }
    
    // At least the first token should match (greedy, temp=0)
    EXPECT_EQ(full_tokens[0], fast_tokens[0])
        << "First decode token differs between full and fast-decode paths";
    
    // All tokens should be non-repeating
    for (size_t i = 1; i < fast_tokens.size(); ++i)
    {
        // Repeating the SAME token 3+ times is suspicious
        if (fast_tokens[i] == fast_tokens[i-1] && i >= 2 && fast_tokens[i] == fast_tokens[i-2])
        {
            FAIL() << "Fast-decode produced 3+ identical tokens in a row: "
                   << fast_tokens[i] << " at positions " << (i-2) << "-" << i
                   << " — probable coherence bug";
        }
    }
}
```

---

## Implementation Priority

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| **P0** | 3A. allocateOnDevice state preservation test | ~30 min | Direct regression test for Bug 1 |
| **P0** | 3B. mark_dirty independence test | ~1 hr | Direct regression test for Bug 2 |
| **P0** | 3C. Fast-decode staleness canary test | ~2 hr | Catches entire bug CLASS, not just these two instances |
| **P1** | 1B. Make mark_dirty unconditional | ~1 hr | Eliminates Bug 2 structurally |
| **P1** | 2C. Arena/tensor state mismatch assertion | ~30 min | Catches Bug 1 at the moment of damage |
| **P2** | 1A. Validated state transitions | ~4 hr | Eliminates invalid transitions across 30 call sites |
| **P2** | 2A. Coherence audit log | ~3 hr | Radically faster diagnosis of any future coherence bug |
| **P2** | 2B. Round-trip canary in decode loop | ~1 hr | Runtime detection in all builds |
| **P3** | 1C. Compile-time policy coupling | ~30 min | Nice to have, documents intent |
| **P3** | 3D. Full vs fast-decode comparison test | ~2 hr | Belt-and-suspenders |

**Recommended order**: P0 tests first (guard against regression), then P1 structural fixes (prevent recurrence), then P2 diagnostic infrastructure (faster future debugging).
