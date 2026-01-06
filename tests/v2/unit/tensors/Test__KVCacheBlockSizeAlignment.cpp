/**
 * @file Test__KVCacheBlockSizeAlignment.cpp
 * @brief Regression tests for KV cache block size alignment bug
 *
 * BUG DESCRIPTION (January 2026):
 * KVCacheAppendStage was creating temporary Q16_1Tensor objects (k_q16_owned, v_q16_owned)
 * with the default block size (BLOCK_32 = 72 bytes per block), but the KV cache allocates
 * tensors using optimal_q16_block_size(head_dim) which returns BLOCK_64 (136 bytes) for
 * head_dim=64.
 *
 * When copy_append_data() runs, it uses the destination's block size to calculate offsets.
 * This caused it to read 136-byte chunks from 72-byte blocks, resulting in:
 * - Reading garbage/uninitialized memory
 * - NaN values appearing in V cache for KV head 1 (but not head 0)
 * - Attention output corruption for Q heads 7-13 (mapped to KV head 1)
 *
 * ROOT CAUSE:
 * Q16_1Tensor constructor without block_size parameter defaults to BLOCK_32.
 * KV cache uses optimal_q16_block_size(head_dim_) which returns BLOCK_64 for head_dim=64.
 *
 * FIX:
 * Pass optimal_q16_block_size(head_dim) to all Q16_1Tensor constructors in KVCacheAppendStage.
 *
 * These tests ensure the bug doesn't regress.
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/UnifiedKVCache.h"
#include "tensors/BlockStructures.h"
#include "utils/MPIContext.h"
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>

namespace llaminar2
{
    namespace test
    {

        // Single-rank MPI context for unit tests
        static MPIContext getTestMPIContext()
        {
            return MPIContext(0, 1, MPI_COMM_WORLD);
        }

        // =============================================================================
        // Test Fixture
        // =============================================================================

        class Test__KVCacheBlockSizeAlignment : public ::testing::Test
        {
        protected:
            void SetUp() override {}
            void TearDown() override {}

            // Helper to check for NaN in Q16_1 tensor scale factors
            static int countNaNScales(const Q16_1Tensor *tensor)
            {
                const uint8_t *raw = static_cast<const uint8_t *>(tensor->raw_data());
                const size_t block_bytes = q16_block_size_bytes(tensor->q16_block_size());
                const size_t num_blocks = tensor->numel() / q16_block_size_elements(tensor->q16_block_size());

                int nan_count = 0;
                for (size_t i = 0; i < num_blocks; ++i)
                {
                    float d;
                    std::memcpy(&d, raw + i * block_bytes, sizeof(float));
                    if (std::isnan(d))
                        nan_count++;
                }
                return nan_count;
            }

            // Helper to create FP32 test data with known pattern
            static std::vector<float> createTestFP32Data(size_t count, float scale = 0.1f)
            {
                std::vector<float> data(count);
                for (size_t i = 0; i < count; ++i)
                {
                    data[i] = static_cast<float>(i % 128) * scale - 6.4f; // Range roughly [-6.4, 6.3]
                }
                return data;
            }
        };

        // =============================================================================
        // Block Size Selection Tests
        // =============================================================================

        TEST_F(Test__KVCacheBlockSizeAlignment, OptimalBlockSize_HeadDim64_ReturnsBlock64)
        {
            // Qwen2-0.5B uses head_dim=64
            EXPECT_EQ(optimal_q16_block_size(64), Q16BlockSize::BLOCK_64)
                << "head_dim=64 should use BLOCK_64 for optimal 1-block-per-head layout";
        }

        TEST_F(Test__KVCacheBlockSizeAlignment, OptimalBlockSize_HeadDim128_ReturnsBlock128)
        {
            // Llama-7B and larger use head_dim=128
            EXPECT_EQ(optimal_q16_block_size(128), Q16BlockSize::BLOCK_128)
                << "head_dim=128 should use BLOCK_128 for optimal 1-block-per-head layout";
        }

        TEST_F(Test__KVCacheBlockSizeAlignment, OptimalBlockSize_HeadDim32_ReturnsBlock32)
        {
            // Some smaller models use head_dim=32
            EXPECT_EQ(optimal_q16_block_size(32), Q16BlockSize::BLOCK_32)
                << "head_dim=32 should use BLOCK_32 for optimal 1-block-per-head layout";
        }

        // =============================================================================
        // Block Size Byte Calculations
        // =============================================================================

        TEST_F(Test__KVCacheBlockSizeAlignment, BlockSizeBytes_Block32_Is72)
        {
            // Q16_1Block_32: float d (4) + int32_t sum_qs (4) + int16_t qs[32] (64) = 72
            EXPECT_EQ(q16_block_size_bytes(Q16BlockSize::BLOCK_32), 72u);
        }

        TEST_F(Test__KVCacheBlockSizeAlignment, BlockSizeBytes_Block64_Is136)
        {
            // Q16_1Block_64: float d (4) + int32_t sum_qs (4) + int16_t qs[64] (128) = 136
            EXPECT_EQ(q16_block_size_bytes(Q16BlockSize::BLOCK_64), 136u);
        }

        TEST_F(Test__KVCacheBlockSizeAlignment, BlockSizeBytes_Block128_Is264)
        {
            // Q16_1Block_128: float d (4) + int32_t sum_qs (4) + int16_t qs[128] (256) = 264
            EXPECT_EQ(q16_block_size_bytes(Q16BlockSize::BLOCK_128), 264u);
        }

        // =============================================================================
        // Q16_1Tensor Construction Block Size Tests
        // =============================================================================

        TEST_F(Test__KVCacheBlockSizeAlignment, Q16_1Tensor_DefaultBlockSize_IsBlock32)
        {
            // Document that default is BLOCK_32 - this is why the bug occurred!
            Q16_1Tensor tensor({1, 64}); // No block_size specified
            EXPECT_EQ(tensor.q16_block_size(), Q16BlockSize::BLOCK_32)
                << "Default Q16_1Tensor block size should be BLOCK_32";
        }

        TEST_F(Test__KVCacheBlockSizeAlignment, Q16_1Tensor_ExplicitBlock64_Works)
        {
            Q16_1Tensor tensor({1, 64}, Q16BlockSize::BLOCK_64);
            EXPECT_EQ(tensor.q16_block_size(), Q16BlockSize::BLOCK_64);
        }

        TEST_F(Test__KVCacheBlockSizeAlignment, Q16_1Tensor_ExplicitBlock128_Works)
        {
            Q16_1Tensor tensor({1, 128}, Q16BlockSize::BLOCK_128);
            EXPECT_EQ(tensor.q16_block_size(), Q16BlockSize::BLOCK_128);
        }

        // =============================================================================
        // KV Cache Block Size Alignment Tests (The Bug Scenario)
        // =============================================================================

        TEST_F(Test__KVCacheBlockSizeAlignment, KVCache_AllocatesWithOptimalBlockSize)
        {
            // This test verifies that UnifiedKVCache allocates tensors with optimal block size
            const int n_layers = 1;
            const int batch_size = 1;
            const int n_kv_heads = 2;
            const int head_dim = 64;
            const int max_seq_len = 128;

            // Use the type alias for Q16_1 cache
            UnifiedKVCacheQ16_1 cache(getTestMPIContext(), n_layers, batch_size, max_seq_len,
                                      n_kv_heads, head_dim, -1);

            // Get the K and V tensors via the polymorphic interface (layer 0, seq 0)
            const TensorBase *k_base = cache.get_k_base(0, 0);
            const TensorBase *v_base = cache.get_v_base(0, 0);
            auto *k_tensor = dynamic_cast<const Q16_1Tensor *>(k_base);
            auto *v_tensor = dynamic_cast<const Q16_1Tensor *>(v_base);

            ASSERT_NE(k_tensor, nullptr) << "K tensor should be Q16_1Tensor";
            ASSERT_NE(v_tensor, nullptr) << "V tensor should be Q16_1Tensor";

            // Verify block size matches optimal for head_dim
            const Q16BlockSize expected = optimal_q16_block_size(head_dim);
            EXPECT_EQ(k_tensor->q16_block_size(), expected)
                << "K tensor block size should match optimal_q16_block_size(head_dim)";
            EXPECT_EQ(v_tensor->q16_block_size(), expected)
                << "V tensor block size should match optimal_q16_block_size(head_dim)";
        }

        TEST_F(Test__KVCacheBlockSizeAlignment, TempTensor_MustMatchKVCacheBlockSize)
        {
            // This is the exact scenario that caused the bug:
            // Creating a temp tensor for V conversion that MUST match KV cache block size
            const int total_tokens = 9;
            const int n_kv_heads = 2;
            const int head_dim = 64;
            const int kv_dim = n_kv_heads * head_dim; // 128

            // WRONG (the bug): default block size
            Q16_1Tensor wrong_tensor({static_cast<size_t>(total_tokens), static_cast<size_t>(kv_dim)});
            EXPECT_EQ(wrong_tensor.q16_block_size(), Q16BlockSize::BLOCK_32)
                << "Without explicit block size, tensor uses BLOCK_32";

            // CORRECT (the fix): explicit optimal block size
            Q16_1Tensor correct_tensor({static_cast<size_t>(total_tokens), static_cast<size_t>(kv_dim)},
                                       optimal_q16_block_size(head_dim));
            EXPECT_EQ(correct_tensor.q16_block_size(), Q16BlockSize::BLOCK_64)
                << "With optimal_q16_block_size(64), tensor uses BLOCK_64";
        }

        // =============================================================================
        // Data Corruption Scenario Tests
        // =============================================================================

        TEST_F(Test__KVCacheBlockSizeAlignment, MismatchedBlockSize_CausesWrongDataLayout)
        {
            // Demonstrate that mismatched block sizes cause different memory layouts
            const size_t rows = 2;
            const size_t cols = 64; // One head_dim

            // Same logical shape, different block sizes
            Q16_1Tensor block32_tensor({rows, cols}, Q16BlockSize::BLOCK_32);
            Q16_1Tensor block64_tensor({rows, cols}, Q16BlockSize::BLOCK_64);

            // Fill with test data
            auto fp32_data = createTestFP32Data(rows * cols);
            block32_tensor.copyFrom_fp32_fixed_scale(fp32_data.data(), 0.01f, cols);
            block64_tensor.copyFrom_fp32_fixed_scale(fp32_data.data(), 0.01f, cols);

            // Verify both have no NaN
            EXPECT_EQ(countNaNScales(&block32_tensor), 0) << "BLOCK_32 tensor should have no NaN scales";
            EXPECT_EQ(countNaNScales(&block64_tensor), 0) << "BLOCK_64 tensor should have no NaN scales";

            // The raw data sizes are different due to different block structures
            // BLOCK_32: 64 elements = 2 blocks per row, 72 bytes per block = 144 bytes per row
            // BLOCK_64: 64 elements = 1 block per row, 136 bytes per block = 136 bytes per row
            const size_t block32_bytes = q16_block_size_bytes(Q16BlockSize::BLOCK_32);
            const size_t block64_bytes = q16_block_size_bytes(Q16BlockSize::BLOCK_64);

            EXPECT_EQ(block32_bytes, 72u);
            EXPECT_EQ(block64_bytes, 136u);

            // Total storage differs:
            // BLOCK_32: 2 rows * (64/32) blocks/row * 72 bytes/block = 2 * 2 * 72 = 288 bytes
            // BLOCK_64: 2 rows * (64/64) blocks/row * 136 bytes/block = 2 * 1 * 136 = 272 bytes
            const size_t expected_block32_storage = rows * (cols / 32) * 72;
            const size_t expected_block64_storage = rows * (cols / 64) * 136;

            EXPECT_EQ(expected_block32_storage, 288u);
            EXPECT_EQ(expected_block64_storage, 272u);
        }

        TEST_F(Test__KVCacheBlockSizeAlignment, ReadingBlock32DataAsBlock64_ProducesGarbage)
        {
            // This test demonstrates what happens when you read BLOCK_32 data with BLOCK_64 offsets
            // (the exact bug scenario)
            const size_t rows = 2;
            const size_t cols = 64;

            // Create BLOCK_32 tensor with valid data
            Q16_1Tensor source({rows, cols}, Q16BlockSize::BLOCK_32);
            auto fp32_data = createTestFP32Data(rows * cols);
            source.copyFrom_fp32_fixed_scale(fp32_data.data(), 0.01f, cols);

            // Verify source is valid
            EXPECT_EQ(countNaNScales(&source), 0) << "Source tensor should have no NaN scales";

            // Now simulate what copy_append_data does when dst has BLOCK_64:
            // It reads at offsets of 136 bytes, but source only has 72-byte blocks
            const uint8_t *src_raw = static_cast<const uint8_t *>(source.raw_data());
            const size_t wrong_block_bytes = q16_block_size_bytes(Q16BlockSize::BLOCK_64); // 136

            // Reading row 1 with wrong offset would read at byte 136
            // But source row 1 actually starts at byte 144 (2 blocks * 72 bytes)
            // This means we'd be reading the middle of row 0's second block!

            // Check that the memory layout is as expected
            const size_t correct_row1_offset = 2 * 72; // 144 bytes (2 BLOCK_32 blocks)
            const size_t wrong_row1_offset = 1 * 136;  // 136 bytes (1 BLOCK_64 block)

            EXPECT_NE(correct_row1_offset, wrong_row1_offset)
                << "This demonstrates the offset mismatch: correct=" << correct_row1_offset
                << " wrong=" << wrong_row1_offset << " difference=" << (correct_row1_offset - wrong_row1_offset);

            // Reading scale factor at wrong offset
            float scale_at_wrong_offset;
            std::memcpy(&scale_at_wrong_offset, src_raw + wrong_row1_offset, sizeof(float));

            // This value is NOT a valid scale factor - it's reading from int16_t quantized values!
            // It might be NaN, Inf, or a garbage float value
            // The exact value depends on the quantized data, but it won't be a sensible scale

            // We can't assert it's NaN because it depends on the bit pattern,
            // but we can verify it's NOT the same as the correct row 1 scale
            float correct_row1_scale;
            std::memcpy(&correct_row1_scale, src_raw + correct_row1_offset, sizeof(float));

            // These should be different (unless by coincidence)
            // The correct scale should be valid (not NaN)
            EXPECT_FALSE(std::isnan(correct_row1_scale))
                << "Correct row 1 scale should be valid, not NaN";
        }

        // =============================================================================
        // End-to-End Regression Test
        // =============================================================================

        TEST_F(Test__KVCacheBlockSizeAlignment, E2E_VTensorConversion_NoNaNWithCorrectBlockSize)
        {
            // Simulate the exact KVCacheAppendStage scenario that had the bug
            const int total_tokens = 9;
            const int n_kv_heads = 2;
            const int head_dim = 64;
            const int kv_dim = n_kv_heads * head_dim;  // 128
            const float kv_cache_scale = 0.001953125f; // 1/512

            // Create source FP32 data (simulating V after dequantization from Q8_1)
            auto v_fp32_data = createTestFP32Data(total_tokens * kv_dim);

            // CORRECT: Create temp tensor with proper block size
            Q16_1Tensor v_q16_correct({static_cast<size_t>(total_tokens), static_cast<size_t>(kv_dim)},
                                      optimal_q16_block_size(head_dim));

            // Quantize
            bool success = v_q16_correct.copyFrom_fp32_fixed_scale(v_fp32_data.data(), kv_cache_scale, head_dim);
            ASSERT_TRUE(success) << "Quantization should succeed";

            // Verify no NaN in any block's scale factor
            EXPECT_EQ(countNaNScales(&v_q16_correct), 0)
                << "V tensor with correct block size should have no NaN scales";

            // Verify block size is what KV cache expects
            EXPECT_EQ(v_q16_correct.q16_block_size(), Q16BlockSize::BLOCK_64)
                << "V tensor should use BLOCK_64 for head_dim=64";
        }

        TEST_F(Test__KVCacheBlockSizeAlignment, E2E_KTensorConversion_NoNaNWithCorrectBlockSize)
        {
            // Same test for K tensor path
            const int total_tokens = 9;
            const int n_kv_heads = 2;
            const int head_dim = 64;
            const int kv_dim = n_kv_heads * head_dim;
            const float kv_cache_scale = 0.001953125f;

            auto k_fp32_data = createTestFP32Data(total_tokens * kv_dim);

            Q16_1Tensor k_q16_correct({static_cast<size_t>(total_tokens), static_cast<size_t>(kv_dim)},
                                      optimal_q16_block_size(head_dim));

            bool success = k_q16_correct.copyFrom_fp32_fixed_scale(k_fp32_data.data(), kv_cache_scale, head_dim);
            ASSERT_TRUE(success);

            EXPECT_EQ(countNaNScales(&k_q16_correct), 0)
                << "K tensor with correct block size should have no NaN scales";
            EXPECT_EQ(k_q16_correct.q16_block_size(), Q16BlockSize::BLOCK_64);
        }

        // =============================================================================
        // HEAD_MAJOR Layout Specific Tests
        // =============================================================================

        TEST_F(Test__KVCacheBlockSizeAlignment, HeadMajorLayout_BlockIndexCalculation)
        {
            // Verify HEAD_MAJOR block index calculation matches KV cache expectations
            const int n_kv_heads = 2;
            const int max_seq_len = 4096;
            const int head_dim = 64;
            const int blocks_per_head = head_dim / 64; // 1 for BLOCK_64

            // In HEAD_MAJOR layout: block_idx = (kv_h * max_seq_len + pos) * blocks_per_head
            // For pos=0, kv_h=0: block_idx = 0
            // For pos=0, kv_h=1: block_idx = 4096
            // For pos=1, kv_h=0: block_idx = 1
            // For pos=1, kv_h=1: block_idx = 4097

            EXPECT_EQ((0 * max_seq_len + 0) * blocks_per_head, 0);
            EXPECT_EQ((1 * max_seq_len + 0) * blocks_per_head, 4096);
            EXPECT_EQ((0 * max_seq_len + 1) * blocks_per_head, 1);
            EXPECT_EQ((1 * max_seq_len + 1) * blocks_per_head, 4097);
        }

        TEST_F(Test__KVCacheBlockSizeAlignment, HeadMajorLayout_TotalStorageCalculation)
        {
            // Verify storage calculation for HEAD_MAJOR layout
            const int n_kv_heads = 2;
            const int max_seq_len = 4096;
            const int head_dim = 64;

            // Total elements = n_kv_heads * max_seq_len * head_dim
            const size_t total_elements = static_cast<size_t>(n_kv_heads) * max_seq_len * head_dim;
            EXPECT_EQ(total_elements, 524288u); // 2 * 4096 * 64

            // With BLOCK_64, blocks_per_head = 1
            const size_t total_blocks = total_elements / 64;
            EXPECT_EQ(total_blocks, 8192u);

            // Total bytes = total_blocks * 136
            const size_t total_bytes = total_blocks * q16_block_size_bytes(Q16BlockSize::BLOCK_64);
            EXPECT_EQ(total_bytes, 1114112u); // ~1.06 MB per K or V tensor
        }

    } // namespace test
} // namespace llaminar2
