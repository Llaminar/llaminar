# Prefill Causal Mask Bug - Debug Handover Document

**Date**: December 14, 2025  
**Status**: Bug identified, root cause being investigated  
**Priority**: Critical - blocks prefill optimization work

## Executive Summary

We discovered a **critical bug** in the JIT fused attention kernel's causal masking for multi-tile prefill. The bug causes Q2 (query at position 2) to incorrectly attend to K3/V3 (key/value at position 3) when it should be masked out by the causal constraint.

**Root Cause Identified**: The `position_offset` spill slot appears to be getting corrupted between tiles. For Tile 0, `position_offset` correctly reads as 0. For Tile 1, it incorrectly reads as 2 (which is the `tile_start` value).

## Files Under Investigation

### Primary Files
| File | Purpose |
|------|---------|
| [src/v2/kernels/cpu/jit/q8_1/JitFusedAttentionWo.cpp](src/v2/kernels/cpu/jit/q8_1/JitFusedAttentionWo.cpp) | JIT kernel generator - contains the bug |
| [src/v2/kernels/cpu/jit/q8_1/JitFusedAttentionWo.h](src/v2/kernels/cpu/jit/q8_1/JitFusedAttentionWo.h) | JIT kernel header with debug buffer definitions |
| [tests/v2/unit/attention/Test__JitFusedAttentionWo_Debug.cpp](tests/v2/unit/attention/Test__JitFusedAttentionWo_Debug.cpp) | Debug test file with multiple isolation tests |

### Supporting Files
| File | Purpose |
|------|---------|
| [src/v2/kernels/cpu/jit/q8_1/JitMicrokernelBase.h](src/v2/kernels/cpu/jit/q8_1/JitMicrokernelBase.h) | Base class with `stack_frame_size()` |
| [docs/v2/projects/2025-12/PREFILL_OPTIMIZATION_PLAN.md](PREFILL_OPTIMIZATION_PLAN.md) | Overall optimization plan |

## Bug Description

### Symptoms
- **Single-tile prefill works correctly** (head_dim=32, q_tile_size=4)
- **Multi-tile prefill fails** (head_dim=64, q_tile_size=2)
- Q2 outputs ~2.5 instead of ~2.0 (attending to all 4 KV positions instead of 3)
- Decode mode works correctly (uses different code path)

### Test Configuration
```cpp
JitAttentionConfig config;
config.head_dim = 64;        // Results in q_tile_size = 2
config.num_heads = 1;
config.num_kv_heads = 1;
config.batch_size = 4;       // 4 queries
config.causal = true;
config.mode = AttentionMode::PREFILL;

// Test with V[0]=0, V[1]=0, V[2]=0, V[3]=1000
// Q2 should output 0 (only sees V[0..2])
// Q2 actually outputs ~250 (sees V[3] too!)
```

### Debug Trace Output
We added runtime tracing that captures intermediate values. Key finding:

```
Tile 0 (tile_start=0):
  kv=3: q_start=3, q_local=3, pos_off=0  ✓ CORRECT

Tile 1 (tile_start=2):
  kv=3: q_start=1, q_local=0, pos_off=2  ✗ BUG! pos_off should be 0!
```

The `position_offset` value is being corrupted from 0 to 2 between tiles.

## Current Hypothesis

The stack spill slot for `position_offset` (offset 808) is being overwritten somewhere between:
1. Initial load at function entry (correctly stores 0)
2. Second tile's KV loop (incorrectly reads 2)

The value 2 is suspiciously equal to `tile_start` for Tile 1, suggesting either:
- Memory corruption/overlap between spill slots
- Incorrect offset calculation in one of the emit functions
- Stack frame miscalculation

### Stack Layout (verified)
```
spill_vars_offset = 784
├── tile_start_spill     = 784  (gets updated each tile: 0 → 2)
├── tile_size_spill      = 792
├── seq_len_kv_spill     = 800
├── position_offset_spill = 808  ← GETTING CORRUPTED
├── kv_idx_spill         = 816
└── q_local_spill        = 824
```

## Debug Infrastructure Added

### 1. Global Debug Buffer
Added to `JitFusedAttentionWo.h`:
```cpp
struct JitDebugBuffer {
    int64_t data[256 * 5];  // tile_start, kv_idx, q_local_start, q, skipped
    int count = 0;
    bool enabled = false;
};
extern JitDebugBuffer g_jit_debug_buffer;

struct JitDebugBuffer2 {
    int64_t data[256 * 8];  // tile_start, kv_idx, q_start, q_local_start, pos_off, head_idx, 0, 0
    int count = 0;
    bool enabled = false;
};
extern JitDebugBuffer2 g_jit_debug_buffer2;
```

### 2. JIT Debug Code Injection
In `generate_prefill_kernel()` around line 640-700, we inject code to capture runtime values:
- Pushes registers to preserve state
- Writes tile_start, kv_idx, q_start, q_local_start, position_offset to buffer
- Pops registers to restore state

### 3. Test Debug Output
The test dumps the buffer contents after kernel execution to trace the values.

## Tests to Run

### Primary Debug Test
```bash
cd /workspaces/llaminar
cmake --build build_v2 --parallel
./build_v2/tests/v2/v2_test_jit_fused_attention_wo_debug \
  --gtest_filter="Test__JitFusedAttentionWo_Debug.CausalCheckIsolation"
```

### All Debug Tests
```bash
./build_v2/tests/v2/v2_test_jit_fused_attention_wo_debug
```

### Key Test Cases
| Test | Purpose | Expected |
|------|---------|----------|
| `CausalCheckIsolation` | 4 queries, multi-tile, V[3]=1000 | Q2 should be ~0, currently ~250 |
| `Tile1OnlyCausal` | 2 queries with position_offset=2 | **PASSES** - single tile works |
| `SingleTileCausal` | head_dim=32, all queries in 1 tile | **PASSES** |
| `CausalMaskTrace_SingleHead` | Traces q_local_start calculation | For analysis |

## Debugging Commands

### Build Debug Version
```bash
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build_v2 --parallel
```

### Run with GDB
```bash
gdb --args ./build_v2/tests/v2/v2_test_jit_fused_attention_wo_debug \
  --gtest_filter="Test__JitFusedAttentionWo_Debug.CausalCheckIsolation"
```

### Disassemble JIT Code
The test `DumpJitCode_MultiTile` dumps the generated binary:
```bash
./build_v2/tests/v2/v2_test_jit_fused_attention_wo_debug \
  --gtest_filter="*DumpJitCode*"
objdump -D -b binary -m i386:x86-64 /tmp/jit_prefill_multi_tile.bin | less
```

### Search for Stack Writes
```bash
grep -n "ptr\[rsp" src/v2/kernels/cpu/jit/q8_1/JitFusedAttentionWo.cpp | grep -v "//"
```

## Next Steps

1. **Find the corruption source**: Search all emit functions for writes to stack offset 808 or nearby
2. **Verify stack layout**: Add assertions to verify spill slot values haven't been corrupted
3. **Check emit functions**: Particularly `emit_prefill_normalize_project` and `emit_prefill_wo_projection_tile`
4. **Consider stack alignment**: Verify 16-byte alignment requirements aren't causing issues

## Key Code Sections

### q_local_start Calculation (lines ~608-627)
```cpp
// q_start = max(0, kv_idx - position_offset)
mov(r10, ptr[rsp + kv_idx_spill]);
if (config_.causal) {
    sub(r10, ptr[rsp + position_offset_spill]);  // ← position_offset read here
    xor_(r11, r11);
    cmp(r10, 0);
    cmovl(r10, r11);
}

// q_local_start = max(0, q_start - tile_start)
mov(rax, ptr[rsp + tile_start_spill]);
sub(r10, rax);
xor_(r11, r11);
cmp(r10, 0);
cmovl(r10, r11);
```

### Causal Check in Softmax (lines ~893-900)
```cpp
cmp(reg_q_local_start, q);  // reg_q_local_start is r10
jg(skip_label, T_NEAR);     // if q_local_start > q, skip

cmp(r8, q);                 // r8 = tile_size
jle(skip_label, T_NEAR);    // if tile_size <= q, skip
```

### Tile Loop Increment (lines ~743-749)
```cpp
mov(rax, ptr[rsp + tile_start_spill]);
add(rax, q_tile_size_);
mov(ptr[rsp + tile_start_spill], rax);
jmp("prefill_tile_loop", T_NEAR);
```

## Reference: Correct Behavior

For 4 queries with causal masking:
- Q0 (position 0): attend to K0 only
- Q1 (position 1): attend to K0, K1
- Q2 (position 2): attend to K0, K1, K2
- Q3 (position 3): attend to K0, K1, K2, K3

With q_tile_size=2:
- Tile 0 (tile_start=0): Q0, Q1
- Tile 1 (tile_start=2): Q2, Q3

For Tile 1, kv_idx=3:
- `q_start = max(0, kv_idx - position_offset) = max(0, 3 - 0) = 3`
- `q_local_start = max(0, q_start - tile_start) = max(0, 3 - 2) = 1`
- Q2 (q=0): `1 > 0` → SKIP ✓
- Q3 (q=1): `1 > 1` → PROCESS ✓

## Contact

This investigation was conducted as part of the prefill optimization effort documented in [PREFILL_OPTIMIZATION_PLAN.md](PREFILL_OPTIMIZATION_PLAN.md).
