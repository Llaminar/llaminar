# ATTENTION_CONTEXT Documentation Update Summary

**Date**: January 2025  
**Feature**: ATTENTION_CONTEXT snapshot capability for parity testing  
**Status**: ✅ Complete - Feature implemented and documented

## Overview

Successfully documented the `ATTENTION_CONTEXT` pipeline stage snapshot feature across all parity testing documentation. This feature enables precise isolation of attention divergence by capturing intermediate results **before the output projection (W_o)**.

## What is ATTENTION_CONTEXT?

`ATTENTION_CONTEXT` captures the attention context vectors (the result of `attention_weights @ value_states`) immediately before the output projection is applied:

```
Attention Pipeline:
  Q @ K^T → scores
  softmax(scores) → attention_weights
  attention_weights @ V → ATTENTION_CONTEXT ✨ (captured here)
  ATTENTION_CONTEXT @ W_o → ATTENTION_OUTPUT
```

## Documentation Updates

### 1. PARITY_FRAMEWORK_SUMMARY.md (+165 lines)

**Changes**:
- Marked ATTENTION_CONTEXT in pipeline stages list with ✨ indicator
- Added comprehensive "Enhanced Attention Stage Debugging" section covering:
  - What ATTENTION_CONTEXT captures and why it matters
  - C++ implementation details with code examples
  - Python reference implementation with code examples
  - Enum synchronization between C++ and Python
  - Complete usage workflow (4 steps)
  - Validation results table showing rel_l2 < 3e-06 achievement
  - Debugging workflow with decision tree
  - Available attention sub-stages list
  - Key benefits enumeration
  - Future enhancement ideas

**Key Sections Added**:
- Implementation code snippets (C++ and Python)
- Usage instructions with bash examples
- Validation results table
- Debugging workflow decision tree
- Benefits and future enhancements

### 2. .github/instructions/parity-test-framework.instructions.md (+142 lines)

**Changes**:
- Enhanced ATTENTION_CONTEXT listing in "Detailed Attention Substages" with:
  - Critical debugging purpose
  - Verified accuracy metrics
  - Debugging workflow decision tree
  - Implementation locations
- Added comprehensive "ATTENTION_CONTEXT Debugging Workflow" section covering:
  - Quick diagnosis table (CONTEXT vs OUTPUT results → diagnosis)
  - Scenario 1: CONTEXT ✅ OUTPUT ❌ (output projection issue)
    - Investigation checklist
    - Common fixes
  - Scenario 2: CONTEXT ❌ (earlier attention issue)
    - Binary search approach
    - Granular capture recommendations
  - Complete example debugging session with terminal output
  - Validation reference data for comparison

**Key Sections Added**:
- Diagnosis decision table
- Two debugging scenarios with detailed steps
- Example debugging session walkthrough
- Reference validation values

### 3. tests/README_PARITY.md (+50 lines)

**Changes**:
- Enhanced environment variables section with PyTorch-specific vars
- Added "Key Pipeline Stages" section listing:
  - Global stages
  - Per-layer stages
  - Attention sub-stages with ATTENTION_CONTEXT highlighted ✨
- Added "Quick ATTENTION_CONTEXT Debugging" section with 3-step workflow:
  - Generate PyTorch reference
  - Run Llaminar with parity capture
  - Interpret results

**Key Sections Added**:
- Complete pipeline stages reference
- Quick debugging workflow
- Result interpretation guide

## Feature Implementation Status

### C++ Side ✅
- **File**: `src/pipeline_stages.h`
  - Enum: `ATTENTION_CONTEXT` (line 45)
  - String conversion: `stage_to_string()` (lines 92-93)
  - String parsing: `string_to_stage()` (lines 146-147)

- **File**: `src/kernels/MPIAttentionKernel.cpp`
  - Primary capture: Lines 800-845
  - Alternative capture: Line 1409
  - Debug logging for layer 0, dimension 842
  - Rank 0 only to avoid duplicates

### Python Side ✅
- **File**: `python/reference/pipeline_stages.py`
  - Enum: `ATTENTION_CONTEXT = auto()` (line 36)
  - String mapping: Line 67
  - Documentation comment

- **File**: `python/reference/generate_test_snapshots.py`
  - Capture: Line 191
  - After `torch.matmul(attn_weights, value_states)`
  - Before `o_proj` application

- **File**: `python/reference/verify_structure.py`
  - Includes ATTENTION_CONTEXT in expected stages (line 122)

### Validation ✅
- **Relative L2 Error**: 2.6e-06 (threshold: 1e-05) ✅
- **Max Absolute Difference**: 5.3e-07
- **Status**: Perfect parity with PyTorch reference

## Key Benefits

1. **Pinpoint Divergence**: Isolates whether issue is in attention mechanism or output projection
2. **Verified Correct**: ATTENTION_CONTEXT achieves rel_l2 < 3e-06 vs PyTorch
3. **Easy Integration**: Single capture hook, works with existing parity framework
4. **Synchronized**: C++ and Python implementations perfectly aligned
5. **Debug-Friendly**: Includes value logging for specific dimensions to aid verification

## Usage Example

```bash
# 1. Generate PyTorch reference snapshots
cd python/reference
python generate_test_snapshots.py \
    --model qwen2.5-0.5b-instruct \
    --tokens 1,2,3,4,5 \
    --output snapshots.npz

# 2. Enable parity capture
export LLAMINAR_PARITY_CAPTURE=1
export PYTORCH_SNAPSHOT_DIR=python/reference/

# 3. Run test
./build/test_parity_framework --gtest_filter="*OpenBLASPrefillVsPyTorch"

# 4. Interpret results
# ATTENTION_CONTEXT ✅ + ATTENTION_OUTPUT ❌ → Output projection issue
# ATTENTION_CONTEXT ❌ → Earlier attention issue
```

## Debugging Decision Tree

```
┌─────────────────────────────┐
│ Run parity test             │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Check ATTENTION_CONTEXT     │
└─────────────┬───────────────┘
              │
        ┌─────┴─────┐
        │           │
        ▼           ▼
      ✅ PASS     ❌ FAIL
        │           │
        ▼           ▼
┌───────────────┐ ┌──────────────────────┐
│ Check OUTPUT  │ │ Check earlier stages │
└───┬───────────┘ │ (Q/K/V/RoPE/scores)  │
    │             └──────────────────────┘
    ▼
  ❌ FAIL
    │
    ▼
┌──────────────────────────────┐
│ Focus on output projection:  │
│ - Weight loading/orientation │
│ - Transpose flags            │
│ - Dimension reversal         │
└──────────────────────────────┘
```

## Documentation Files Updated

| File | Lines Added | Description |
|------|-------------|-------------|
| `PARITY_FRAMEWORK_SUMMARY.md` | +165 | Comprehensive feature documentation with examples |
| `.github/instructions/parity-test-framework.instructions.md` | +142 | Debugging workflows and troubleshooting |
| `tests/README_PARITY.md` | +50 | Quick reference and usage guide |
| **Total** | **+357** | Complete documentation coverage |

## Related Files (No Changes Needed - Already Implemented)

- ✅ `src/pipeline_stages.h` - Enum definition
- ✅ `src/kernels/MPIAttentionKernel.cpp` - Capture implementation
- ✅ `python/reference/pipeline_stages.py` - Python enum
- ✅ `python/reference/generate_test_snapshots.py` - PyTorch capture
- ✅ `python/reference/verify_structure.py` - Structure validation

## Next Steps (Optional Enhancements)

Future improvements that could be added:

1. **Visual Diff Tools**: Web UI for tensor comparison
2. **Separate Q/K/V Captures**: Pre-RoPE projection snapshots
3. **Per-Head Validation**: Capture attention scores for each head independently
4. **Statistical Validation**: Distribution comparison in addition to element-wise
5. **Automatic Bisection**: Auto-detect first diverging layer

## Conclusion

The ATTENTION_CONTEXT snapshot feature is now fully documented across all relevant documentation files:

- ✅ **Feature Purpose**: Clear explanation of what it captures and why
- ✅ **Implementation**: Code locations and examples provided
- ✅ **Usage**: Step-by-step workflows with bash commands
- ✅ **Validation**: Proven accuracy metrics documented
- ✅ **Debugging**: Decision trees and troubleshooting guides
- ✅ **Integration**: Works seamlessly with existing parity framework

Developers can now:
- Understand what ATTENTION_CONTEXT captures
- Use it to debug attention divergence efficiently
- Interpret results correctly
- Follow established debugging workflows
- Extend it for new use cases

**Documentation Status**: ✅ Complete and ready for use
