# Testing Strategy: Online Softmax V2 Implementation

## Overview

This document outlines the testing strategy for the V2 online softmax rewrite before integrating it into the full attention pipeline. The goal is to **prove correctness in isolation** using real captured data.

---

## 1. Test Levels

### Level 1: Unit Tests (Synthetic Data)
**Purpose**: Verify mathematical correctness of individual functions.

| Test | Description |
|------|-------------|
| `rescale_int64_basic` | Verify `__int128` rescale matches FP64 reference |
| `rescale_int64_edge_cases` | Overflow, underflow, zero handling |
| `chunked_accumulation_vs_naive` | Chunk + dump matches single-pass INT64 |
| `weight_sum_tracking` | `sum_w_scaled` == Σ(w >> shift) exactly |
| `finalization_division` | Final divide produces correct weighted average |

### Level 2: Microkernel Replay (Real Data)
**Purpose**: Verify V2 produces same output as V1 on real pipeline inputs.

| Test | Description |
|------|-------------|
| `V1_vs_V2_decode_parity` | Same Q/K/V/scales → same context |
| `V1_vs_V2_prefill_parity` | Same inputs → same context per row |
| `V2_vs_FP32_reference` | V2 output ≈ FP32 softmax (expected tolerance) |

### Level 3: Integration (Full Pipeline)
**Purpose**: Verify the integrated pipeline still passes E2E parity.

---

## 2. Stage Dump Capture Strategy

### Required Tensors for Replay

To replay the `flash_decode_process_kv_block` microkernel, we need:

| Tensor | Type | Shape | Source |
|--------|------|-------|--------|
| `Q` | `Q16_1Block_64` | `[blocks_per_row]` | Per-head Q projection output |
| `K` | `Q16_1Block_64` | `[kv_len × blocks_per_row]` | KV cache |
| `V` | `Q16_1Block_64` | `[kv_len × blocks_per_row]` | KV cache |
| `q_scale` | `float` | scalar | From `Q16IntegerAttentionParams` |
| `kv_scale` | `float` | scalar | From KV cache block headers |
| `k_position_scales` | `float[]` | `[kv_len]` | Optional: per-position K scales |

### Capture Command

```bash
# Capture attention inputs for layer 0, decode iteration 0
LLAMINAR_STAGE_DUMP_ENABLED=1 \
LLAMINAR_STAGE_DUMP_TYPES=FUSED_ATTENTION_WO \
LLAMINAR_STAGE_DUMP_LAYERS=0 \
LLAMINAR_STAGE_DUMP_ITERATION=0 \
LLAMINAR_STAGE_DUMP_INPUTS=1 \
LLAMINAR_STAGE_DUMP_OUTPUTS=1 \
./build_v2/llaminar2 -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello world" -n 5
```

### Dump Directory Structure

```
/tmp/llaminar_stage_dumps/
└── stage_0001_FUSED_ATTENTION_WO_layer0_rank0/
    ├── metadata.txt           # Human-readable config
    ├── inputs/
    │   ├── Q.bin              # Q blocks (binary Q16_1 data)
    │   ├── K.bin              # K cache blocks
    │   ├── V.bin              # V cache blocks
    │   ├── q_scales.bin       # Per-head Q scales
    │   ├── kv_scales.bin      # Per-KV-head scales
    │   └── k_position_scales.bin  # Optional per-position K scales
    └── outputs/
        ├── context.bin        # INT32 context before Wo
        └── wo_output.bin      # Q16_1 Wo projection output
```

---

## 3. Replay Test Design

### Test File: `Test__OnlineSoftmaxV2_Replay.cpp`

```cpp
/**
 * @file Test__OnlineSoftmaxV2_Replay.cpp
 * @brief Replay test: V2 softmax vs V1 and FP32 reference using real pipeline data
 *
 * This test proves the V2 deferred normalization implementation produces
 * equivalent results to V1 before integrating into the full pipeline.
 */

#include <gtest/gtest.h>
#include "kernels/cpu/attention/q16_1/ref/microkernels/OnlineSoftmax.h"
#include "tensors/BlockStructures.h"
#include <fstream>
#include <cmath>

using namespace llaminar2::kernels::q16_1::microkernels;
using namespace llaminar2;

namespace {

// Qwen2-0.5B config
constexpr int HEAD_DIM = 64;
constexpr int NUM_HEADS = 14;
constexpr int NUM_KV_HEADS = 2;

/**
 * @brief Load Q16_1Block data from binary dump
 */
template <typename BlockType>
std::vector<BlockType> loadQ16Blocks(const std::string& path, size_t num_blocks) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path);
    
    std::vector<BlockType> blocks(num_blocks);
    file.read(reinterpret_cast<char*>(blocks.data()), num_blocks * sizeof(BlockType));
    return blocks;
}

/**
 * @brief FP32 reference: compute softmax(Q @ K^T) @ V for single head
 */
std::vector<float> fp32_reference_attention(
    const float* Q,    // [head_dim]
    const float* K,    // [kv_len × head_dim]
    const float* V,    // [kv_len × head_dim]
    int kv_len,
    int head_dim)
{
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Compute scores
    std::vector<float> scores(kv_len);
    float max_score = -INFINITY;
    for (int k = 0; k < kv_len; ++k) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += Q[d] * K[k * head_dim + d];
        }
        scores[k] = dot * scale;
        max_score = std::max(max_score, scores[k]);
    }
    
    // Softmax
    float sum = 0.0f;
    for (int k = 0; k < kv_len; ++k) {
        scores[k] = std::exp(scores[k] - max_score);
        sum += scores[k];
    }
    for (int k = 0; k < kv_len; ++k) {
        scores[k] /= sum;
    }
    
    // Weighted sum
    std::vector<float> context(head_dim, 0.0f);
    for (int k = 0; k < kv_len; ++k) {
        for (int d = 0; d < head_dim; ++d) {
            context[d] += scores[k] * V[k * head_dim + d];
        }
    }
    
    return context;
}

} // namespace

class Test__OnlineSoftmaxV2_Replay : public ::testing::Test {
protected:
    static constexpr const char* DUMP_DIR = 
        "/tmp/llaminar_stage_dumps/stage_0001_FUSED_ATTENTION_WO_layer0_rank0";
    
    void SetUp() override {
        std::ifstream test_file(std::string(DUMP_DIR) + "/metadata.txt");
        if (!test_file.good()) {
            GTEST_SKIP() << "No dump data. Run with LLAMINAR_STAGE_DUMP_ENABLED=1 first.";
        }
    }
};

/**
 * @brief Verify V2 matches V1 output on real captured data
 *
 * This is the KEY test: if V1 == V2, we can swap implementations safely.
 */
TEST_F(Test__OnlineSoftmaxV2_Replay, V2_Matches_V1_Decode) {
    // Load captured Q/K/V blocks
    // ... (implementation follows pattern from HybridQ16AttentionReplay)
    
    // Run V1 (current implementation)
    std::vector<int32_t> context_v1(HEAD_DIM);
    OnlineSoftmaxState state_v1;
    state_v1.init(qk_scale);
    // ... process blocks with V1 ...
    
    // Run V2 (new implementation)
    std::vector<int64_t> context_accum_v2(HEAD_DIM, 0);
    OnlineSoftmaxStateV2 state_v2;
    state_v2.init(qk_scale);
    // ... process blocks with V2 ...
    
    std::vector<int32_t> context_v2(HEAD_DIM);
    flash_decode_finalize_v2(context_accum_v2.data(), context_v2.data(), 
                              state_v2, HEAD_DIM);
    
    // Compare
    for (int d = 0; d < HEAD_DIM; ++d) {
        EXPECT_EQ(context_v1[d], context_v2[d]) 
            << "Mismatch at dimension " << d;
    }
}

/**
 * @brief Verify V2 is within expected tolerance of FP32 reference
 */
TEST_F(Test__OnlineSoftmaxV2_Replay, V2_Matches_FP32_Reference) {
    // Load and dequantize Q/K/V to FP32
    // ... 
    
    // Compute FP32 reference
    auto ref_context = fp32_reference_attention(Q_fp32, K_fp32, V_fp32, 
                                                  kv_len, HEAD_DIM);
    
    // Compute V2
    // ...
    
    // Dequantize V2 context to FP32
    std::vector<float> v2_context_fp32(HEAD_DIM);
    for (int d = 0; d < HEAD_DIM; ++d) {
        v2_context_fp32[d] = context_v2[d] * pv_scale;
    }
    
    // Check cosine similarity (expected: > 0.99 for typical Q16 precision)
    float cos_sim = cosineSimilarity(ref_context.data(), v2_context_fp32.data(), HEAD_DIM);
    EXPECT_GT(cos_sim, 0.99f) << "V2 diverges from FP32 reference";
}
```

---

## 4. Synthetic Unit Test: Deferred Normalization Correctness

### Test File: `Test__OnlineSoftmaxV2_Unit.cpp`

```cpp
/**
 * @brief Unit test proving deferred normalization is mathematically equivalent
 *
 * Key property: 
 *   (Σ w_i × V_i) / (Σ w_i)  ==  running_average_approach
 *
 * We verify this with synthetic data before using real pipeline data.
 */
TEST(Test__OnlineSoftmaxV2_Unit, DeferredNormalization_EquivalentToRunningAverage) {
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 128;
    
    // Generate synthetic scores and V values
    std::vector<int32_t> scores(KV_LEN);
    std::vector<std::array<int16_t, HEAD_DIM>> V(KV_LEN);
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> score_dist(-10000, 10000);
    std::uniform_int_distribution<int16_t> v_dist(-1000, 1000);
    
    for (int k = 0; k < KV_LEN; ++k) {
        scores[k] = score_dist(rng);
        for (int d = 0; d < HEAD_DIM; ++d) {
            V[k][d] = v_dist(rng);
        }
    }
    
    // ===== V1: Running average approach =====
    // (Current slow FP implementation)
    std::vector<int32_t> context_v1(HEAD_DIM, 0);
    double l_processed = 0.0;
    int32_t max_score = std::numeric_limits<int32_t>::min();
    
    for (int k = 0; k < KV_LEN; ++k) {
        max_score = std::max(max_score, scores[k]);
    }
    
    for (int k = 0; k < KV_LEN; ++k) {
        // Simulate exp2 weight (simplified)
        double weight = std::exp2(static_cast<double>(scores[k] - max_score) / 1000.0);
        double l_new = l_processed + weight;
        
        for (int d = 0; d < HEAD_DIM; ++d) {
            double numerator = static_cast<double>(context_v1[d]) * l_processed 
                             + weight * V[k][d];
            context_v1[d] = static_cast<int32_t>(std::round(numerator / l_new));
        }
        l_processed = l_new;
    }
    
    // ===== V2: Deferred normalization approach =====
    // (New fast integer implementation)
    std::vector<int64_t> context_v2(HEAD_DIM, 0);
    int64_t sum_w = 0;
    
    for (int k = 0; k < KV_LEN; ++k) {
        // Same weight calculation
        int64_t weight = static_cast<int64_t>(
            std::exp2(static_cast<double>(scores[k] - max_score) / 1000.0) * 1000);
        sum_w += weight;
        
        for (int d = 0; d < HEAD_DIM; ++d) {
            context_v2[d] += weight * V[k][d];
        }
    }
    
    // Finalize: single division
    std::vector<int32_t> context_v2_final(HEAD_DIM);
    for (int d = 0; d < HEAD_DIM; ++d) {
        context_v2_final[d] = static_cast<int32_t>(context_v2[d] / sum_w);
    }
    
    // ===== Compare =====
    // Due to FP precision differences, allow small tolerance
    for (int d = 0; d < HEAD_DIM; ++d) {
        EXPECT_NEAR(context_v1[d], context_v2_final[d], 2)
            << "Mismatch at dimension " << d 
            << " (V1=" << context_v1[d] << " V2=" << context_v2_final[d] << ")";
    }
}

/**
 * @brief Verify __int128 rescale correctness
 */
TEST(Test__OnlineSoftmaxV2_Unit, Rescale128_MatchesFP64) {
    // Test values that would overflow 64-bit multiply
    int64_t value = (1LL << 50);  // Large accumulator
    int32_t scale_num = (1 << 28);  // Large scale
    int scale_shift = 25;
    
    // FP64 reference
    double fp_result = static_cast<double>(value) * scale_num / (1ULL << scale_shift);
    
    // __int128 implementation
    __int128 product = static_cast<__int128>(value) * scale_num;
    int64_t int_result = static_cast<int64_t>(product >> scale_shift);
    
    // Should match within rounding
    EXPECT_NEAR(static_cast<double>(int_result), fp_result, 1.0);
}

/**
 * @brief Verify chunk accumulation doesn't overflow INT32
 */
TEST(Test__OnlineSoftmaxV2_Unit, ChunkAccumulation_NoOverflow) {
    constexpr int CHUNK_SIZE = 60;
    constexpr int HEAD_DIM = 128;
    constexpr int32_t MAX_WEIGHT = 1023;  // 10-bit after shift
    constexpr int16_t MAX_V = 32767;
    
    int32_t chunk_accum[HEAD_DIM] = {0};
    
    // Worst case: max weight × max V × chunk_size
    for (int k = 0; k < CHUNK_SIZE; ++k) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            int32_t product = MAX_WEIGHT * MAX_V;
            
            // Verify no overflow before accumulation
            ASSERT_LT(std::abs(static_cast<int64_t>(chunk_accum[d]) + product), 
                      static_cast<int64_t>(INT32_MAX));
            
            chunk_accum[d] += product;
        }
    }
    
    // Verify final values
    int64_t expected_max = static_cast<int64_t>(CHUNK_SIZE) * MAX_WEIGHT * MAX_V;
    EXPECT_LT(expected_max, static_cast<int64_t>(INT32_MAX))
        << "Chunk overflow detected! Need smaller CHUNK_SIZE.";
}
```

---

## 5. Test Execution Plan

### Phase 1: Unit Tests (Before Any Code Changes)

```bash
# Build and run unit tests for V2 functions
ctest --test-dir build_v2 -R "V2_Unit_OnlineSoftmaxV2" --output-on-failure
```

**Expected**: All pass, proving the math is correct.

### Phase 2: Generate Stage Dumps

```bash
# Capture attention inputs from integration test
LLAMINAR_STAGE_DUMP_ENABLED=1 \
LLAMINAR_STAGE_DUMP_NAMES=fused_attn \
LLAMINAR_STAGE_DUMP_LAYERS=0,1,2 \
ctest --test-dir build_v2_integration -R "V2_Integration_HybridQ16Pipeline" --verbose
```

### Phase 3: Replay Tests (V1 vs V2)

```bash
# Run replay test comparing V1 and V2
ctest --test-dir build_v2 -R "V2_Unit_OnlineSoftmaxV2_Replay" --output-on-failure
```

**Expected**: V2 output == V1 output (exact match for same inputs).

### Phase 4: Performance Comparison

```bash
# Benchmark V1 vs V2 on same inputs
./build_v2_release/tests/v2/v2_perf_online_softmax_v1_vs_v2
```

**Expected**: V2 10-100× faster due to eliminated FP divisions.

### Phase 5: Full Integration

```bash
# Replace V1 with V2 in pipeline, run full test suite
ctest --test-dir build_v2_integration -R "V2_Integration_" --output-on-failure
ctest --test-dir build_v2_e2e_release -R "V2_E2E_" --output-on-failure
```

---

## 6. Files to Create

| File | Purpose |
|------|---------|
| `tests/v2/unit/microkernels/Test__OnlineSoftmaxV2_Unit.cpp` | Synthetic unit tests |
| `tests/v2/integration/replay/Test__OnlineSoftmaxV2_Replay.cpp` | Replay using stage dumps |
| `tests/v2/performance/Test__OnlineSoftmaxV1_vs_V2.cpp` | Performance comparison |

---

## 7. Success Criteria

| Criterion | Metric |
|-----------|--------|
| **V1 == V2** | Exact INT32 match on all captured inputs |
| **V2 ≈ FP32** | Cosine similarity > 0.99 |
| **No overflow** | All accumulations stay within INT32/INT64 bounds |
| **Performance** | V2 prefill 10× faster than V1 |
| **E2E parity** | Same token predictions as before |

---

## 8. Risk Mitigation

1. **Keep V1 code**: Add V2 as new functions with `_v2` suffix, don't delete V1
2. **Feature flag**: Allow runtime selection of V1 vs V2 for A/B testing
3. **Snapshot comparison**: Capture snapshots from both and compare programmatically

---

*Document version: 1.0*
*Created: 2026-01-05*
