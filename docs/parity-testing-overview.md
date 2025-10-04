# Parity Testing in Llaminar: Framework Overview

## Relationship to Existing Tests

Llaminar has two complementary approaches to validating correctness against llama.cpp:

### 1. End-to-End Golden Test (Existing)
**File**: `tests/test_prefill_attention_golden.cpp`

**What it does:**
- Runs complete inference through both Llaminar and llama.cpp
- Compares final outputs: logits and pre-LM hidden states
- Validates overall pipeline correctness

**When to use:**
- Regular regression testing
- Verifying model loading and execution work
- Confirming end-to-end numerical accuracy

**Limitations:**
- Cannot identify which specific stage diverges
- Limited visibility into intermediate computations
- Harder to debug when something goes wrong

### 2. Parity Test Framework (New)
**Files**: `tests/parity_test_framework.{h,cpp}`, `tests/test_parity_framework.cpp`

**What it does:**
- Captures intermediate tensor states at 18 pipeline stages
- Enables stage-by-stage comparison with llama.cpp
- Provides detailed diagnostics when divergence occurs

**When to use:**
- Debugging pipeline divergence
- Validating new kernel implementations
- Understanding which stage introduces errors
- Developing new model architecture support

**Limitations:**
- Requires manual integration (adding capture hooks)
- llama.cpp intermediate states not readily available
- More complex to set up than end-to-end test

## Recommended Workflow

```
┌─────────────────────────────────────┐
│  1. Start with Golden Test          │
│  (test_prefill_attention_golden)    │
│                                      │
│  Does end-to-end parity pass?       │
└──────────────┬──────────────────────┘
               │
               ├─ YES ──► You're done! Pipeline is correct
               │
               └─ NO ──► Continue to step 2
                         
┌─────────────────────────────────────┐
│  2. Use Parity Framework             │
│  (test_parity_framework)             │
│                                      │
│  Add capture hooks at suspected      │
│  divergence points                   │
└──────────────┬──────────────────────┘
               │
               └─ Identify diverging stage
                  
┌─────────────────────────────────────┐
│  3. Debug Specific Stage             │
│                                      │
│  - Examine top-k differences         │
│  - Check tensor shapes               │
│  - Validate intermediate values      │
│  - Fix the bug                       │
└──────────────┬──────────────────────┘
               │
               └─ Return to step 1 to verify fix
```

## Quick Reference

### If you want to...

**Verify overall correctness:**
```bash
mpirun -np 2 ./build/test_prefill_attention_golden
```

**Debug which stage is wrong:**
```bash
# 1. Add hooks in pipeline code
if (parity::LlaminarSnapshotHook::is_enabled()) {
    parity::LlaminarSnapshotHook::capture(...);
}

# 2. Run with capture enabled
export LLAMINAR_PARITY_CAPTURE=1
mpirun -np 2 ./build/test_parity_framework
```

**Compare specific tensors:**
```cpp
auto result = SnapshotComparator::compare(
    reference, test, 
    ComparisonTolerance(1e-3f, 1e-4)
);
if (!result.passed()) {
    SnapshotComparator::log_top_differences(...);
}
```

**Test a new model architecture:**
1. Run golden test first (end-to-end)
2. If divergence found, add parity hooks
3. Use framework to isolate the issue
4. Document expected tolerances for new architecture

## Framework Capabilities Matrix

| Capability | Golden Test | Parity Framework |
|------------|-------------|------------------|
| End-to-end validation | ✅ Primary | ✅ Available |
| Intermediate state capture | ❌ No | ✅ Yes (18 stages) |
| Stage-level debugging | ❌ Limited | ✅ Detailed |
| Zero integration cost | ✅ Ready to use | ⚠️ Requires hooks |
| llama.cpp reference | ✅ Built-in | ⚠️ Partial (endpoints) |
| Custom architecture support | ⚠️ Model-specific | ✅ Extensible |
| Tolerance configuration | ✅ Global | ✅ Per-stage |
| Diff visualization | ✅ Basic | ✅ Top-k details |
| MPI compatibility | ✅ Full | ✅ Full |

## Integration Patterns

### Pattern 1: Quick Validation (No Code Changes)
```bash
# Use existing golden test
mpirun -np 2 ./build/test_prefill_attention_golden

# Check for:
# [PARITY_LOGITS] max_abs=... rel_l2=...
# PASSED if within tolerance
```

### Pattern 2: Targeted Debugging (Minimal Hooks)
```cpp
// Add just one hook at suspected stage
if (parity::LlaminarSnapshotHook::is_enabled()) {
    parity::LlaminarSnapshotHook::capture(
        parity::PipelineStage::ATTENTION_OUTPUT,
        layer_idx,
        data,
        seq_len,
        d_model
    );
}

// Run test and examine that specific stage
```

### Pattern 3: Comprehensive Validation (Full Instrumentation)
```cpp
// Add hooks at all major stages
// - After embedding
// - After each layer's attention
// - After each layer's FFN
// - After final norm

// Run full parity suite
// Generates complete stage-by-stage report
```

## Example Debugging Session

```bash
# Step 1: Initial test shows divergence
$ mpirun -np 2 ./build/test_prefill_attention_golden
[PARITY_LOGITS] max_abs=0.125 rel_l2=0.023
FAILED - Exceeds tolerance

# Step 2: Add hooks and capture
$ export LLAMINAR_PARITY_CAPTURE=1
$ mpirun -np 2 ./build/test_parity_framework

# Step 3: Examine output
[PARITY_FINAL_HIDDEN] max_abs=0.002 rel_l2=0.0004  ✓ OK
[PARITY_LOGITS] max_abs=0.125 rel_l2=0.023         ✗ FAIL

# Conclusion: Final norm OK, but LM head projection wrong
# → Check output projection weights or matmul

# Step 4: Add more granular hooks
# (Add hook after LM head matmul but before softmax)

# Step 5: Fix issue and verify
$ mpirun -np 2 ./build/test_prefill_attention_golden
[PARITY_LOGITS] max_abs=0.0012 rel_l2=0.00023
PASSED ✓
```

## Documentation Map

```
docs/
├── parity-test-framework.md
│   └── Complete API reference, architecture, metrics
│
├── parity-integration-guide.md
│   └── Step-by-step: How to add hooks and write tests
│
├── pipeline-vs-llama-cpp-comparison.md
│   └── Background: Pipeline analysis (existing)
│
└── THIS FILE (parity-testing-overview.md)
    └── High-level guide: When to use what

tests/
├── test_prefill_attention_golden.cpp
│   └── Existing: End-to-end golden test
│
├── test_parity_framework.cpp
│   └── New: Stage-by-stage parity validation
│
└── README_PARITY.md
    └── Quick start guide for parity framework
```

## Key Takeaways

1. **Start simple**: Always run the golden test first
2. **Add hooks selectively**: Don't instrument everything at once
3. **Use framework for debugging**: When golden test fails, parity framework helps isolate the issue
4. **Trust the framework**: Once parity passes, you can be confident in correctness
5. **Extend thoughtfully**: Framework is designed for easy extension to new architectures

## Getting Help

**For end-to-end testing:**
- See existing `test_prefill_attention_golden.cpp`
- Review `docs/pipeline-vs-llama-cpp-comparison.md`

**For stage-by-stage testing:**
- Read `docs/parity-test-framework.md` (API reference)
- Follow `docs/parity-integration-guide.md` (integration steps)
- Check `tests/test_parity_framework.cpp` (examples)

**For general questions:**
- Review this overview document
- Check `tests/README_PARITY.md` (quick start)

## Future Enhancements

Potential improvements to the testing infrastructure:

1. **Automatic hook generation**: Code generation for capture hooks
2. **llama.cpp instrumentation**: Patches to extract all intermediate states
3. **Continuous parity monitoring**: Track parity metrics over time
4. **Visual diff tools**: Web UI for comparing tensor snapshots
5. **Fuzzing integration**: Random input testing with parity validation

---

**Remember**: The parity framework complements, not replaces, the golden test. Use both together for comprehensive validation.
