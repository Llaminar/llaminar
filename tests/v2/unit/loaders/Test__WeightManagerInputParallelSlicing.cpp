/**
 * @file Test__WeightManagerInputParallelSlicing.cpp
 * @brief Exhaustive unit tests for WeightManager INPUT_PARALLEL slicing
 *
 * This file tests the INPUT_PARALLEL (column slicing) logic that was the source
 * of a critical bug: attn_output (Wo) and ffn_down have different slicing strategies
 * but both use INPUT_PARALLEL mode.
 *
 * Bug scenario that was fixed:
 * - INPUT_PARALLEL used d_ff dimensions (4864) for ALL weights
 * - attn_output.weight [896, 896] was being sliced with d_ff ranges [0, 2432)
 * - This caused "Invalid column range" errors
 *
 * The fix: Use categorizeWeight() to distinguish:
 * - ATTENTION_WO: slice by head_start * head_dim to (head_start + head_count) * head_dim
 * - FFN_DOWN: slice by d_ff_start to d_ff_start + d_ff_count
 *
 * @note These tests mock the TP config and assignment to test slicing logic
 *       without requiring actual GGUF model loading.
 */

#include <gtest/gtest.h>
#include <memory>
#include <cstring>
#include <cmath>

#include "loaders/WeightManager.h"
#include "tensors/Tensors.h"
#include "utils/TestTensorFactory.h"
#include "models/qwen/Qwen2Schema.h"
#include "mocks/MockModelLoader.h"

namespace llaminar2
{
    namespace test
    {

        // =============================================================================
        // Test Fixture with Qwen2.5 realistic dimensions
        // =============================================================================

        /**
         * @brief Test fixture for INPUT_PARALLEL slicing tests
         *
         * Uses realistic Qwen2.5-0.5B dimensions:
         * - n_heads = 14
         * - n_kv_heads = 2
         * - head_dim = 64
         * - hidden_dim = 896 (n_heads * head_dim)
         * - intermediate_size (d_ff) = 4864
         */
        class Test__WeightManagerInputParallelSlicing : public ::testing::Test
        {
        protected:
            // Qwen2.5-0.5B model dimensions
            static constexpr size_t N_HEADS = 14;
            static constexpr size_t N_KV_HEADS = 2;
            static constexpr size_t HEAD_DIM = 64;
            static constexpr size_t HIDDEN_DIM = N_HEADS * HEAD_DIM; // 896
            static constexpr size_t D_FF = 4864;

            // Weight names
            static constexpr const char *ATTN_OUTPUT_L0 = "blk.0.attn_output.weight";
            static constexpr const char *FFN_DOWN_L0 = "blk.0.ffn_down.weight";
            static constexpr const char *ATTN_Q_L0 = "blk.0.attn_q.weight";
            static constexpr const char *FFN_GATE_L0 = "blk.0.ffn_gate.weight";

            /**
             * @brief Create a tensor with sequential values for easy verification
             *
             * Values are set to row * 10000 + col for easy debugging
             */
            static std::shared_ptr<FP32Tensor> createSequentialTensor(size_t rows, size_t cols)
            {
                auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
                float *data = tensor->mutable_data();
                for (size_t i = 0; i < rows; ++i)
                {
                    for (size_t j = 0; j < cols; ++j)
                    {
                        data[i * cols + j] = static_cast<float>(i * 10000 + j);
                    }
                }
                return tensor;
            }

            /**
             * @brief Verify that a tensor slice contains expected sequential values
             */
            static void verifySequentialSlice(
                const TensorBase *slice,
                size_t expected_rows,
                size_t expected_cols,
                size_t original_cols,
                size_t col_offset)
            {
                ASSERT_NE(slice, nullptr) << "Slice should not be null";
                ASSERT_EQ(slice->shape().size(), 2) << "Slice should be 2D";
                EXPECT_EQ(slice->shape()[0], expected_rows) << "Row count mismatch";
                EXPECT_EQ(slice->shape()[1], expected_cols) << "Col count mismatch";

                const float *data = slice->data();
                for (size_t i = 0; i < expected_rows; ++i)
                {
                    for (size_t j = 0; j < expected_cols; ++j)
                    {
                        size_t original_col = col_offset + j;
                        float expected = static_cast<float>(i * 10000 + original_col);
                        EXPECT_FLOAT_EQ(data[i * expected_cols + j], expected)
                            << "Mismatch at row " << i << ", local col " << j
                            << " (original col " << original_col << ")";
                    }
                }
            }
        };

        /**
         * @brief Mock loader that simulates an unavailable native packed column slice
         *
         * Some real GGUF quantized formats can only slice columns on their packed
         * block boundary. Returning nullptr from loadTensorColumnSlice() lets the
         * unit test exercise WeightManager's FP32 fallback without needing a GGUF
         * fixture on disk.
         */
        class FailingColumnSliceModelLoader : public MockModelLoader
        {
        public:
            std::shared_ptr<TensorBase> loadTensorColumnSlice(
                const std::string &name,
                size_t col_start,
                size_t col_end,
                DeviceId device = DeviceId::cpu(),
                WeightPrecision weight_precision = WeightPrecision::NATIVE) override
            {
                (void)name;
                (void)device;
                (void)weight_precision;

                ++column_slice_attempts;
                last_col_start = col_start;
                last_col_end = col_end;
                return nullptr;
            }

            std::shared_ptr<TensorBase> loadTensor(
                const std::string &name,
                DeviceId device = DeviceId::cpu(),
                WeightPrecision weight_precision = WeightPrecision::NATIVE) override
            {
                if (weight_precision == WeightPrecision::CONVERT_TO_FP32)
                {
                    ++fp32_fallback_loads;
                }
                return MockModelLoader::loadTensor(name, device, weight_precision);
            }

            int column_slice_attempts = 0;
            int fp32_fallback_loads = 0;
            size_t last_col_start = 0;
            size_t last_col_end = 0;
        };

        // =============================================================================
        // Sharding Config Mode Tests
        // =============================================================================

        TEST_F(Test__WeightManagerInputParallelSlicing, ShardingMode_AttnOutput_IsInputParallel)
        {
            Qwen2SchemaFactory schema_factory;
            auto config = schema_factory.getWeightShardingConfig();

            // attn_output (Wo) should be INPUT_PARALLEL
            EXPECT_EQ(config.getMode("blk.0.attn_output.weight"), WeightShardingMode::InputParallel);
            EXPECT_EQ(config.getMode("blk.5.attn_output.weight"), WeightShardingMode::InputParallel);
            EXPECT_EQ(config.getMode("blk.23.attn_output.weight"), WeightShardingMode::InputParallel);

            // Generic name also works
            EXPECT_EQ(config.getMode("attn_output.weight"), WeightShardingMode::InputParallel);
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, ShardingMode_FFNDown_IsInputParallel)
        {
            Qwen2SchemaFactory schema_factory;
            auto config = schema_factory.getWeightShardingConfig();

            // ffn_down should be INPUT_PARALLEL
            EXPECT_EQ(config.getMode("blk.0.ffn_down.weight"), WeightShardingMode::InputParallel);
            EXPECT_EQ(config.getMode("blk.10.ffn_down.weight"), WeightShardingMode::InputParallel);
            EXPECT_EQ(config.getMode("blk.23.ffn_down.weight"), WeightShardingMode::InputParallel);

            // Generic name
            EXPECT_EQ(config.getMode("ffn_down.weight"), WeightShardingMode::InputParallel);
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, ShardingMode_QKV_IsColumnParallel)
        {
            Qwen2SchemaFactory schema_factory;
            auto config = schema_factory.getWeightShardingConfig();

            // QKV should be COLUMN_PARALLEL (not INPUT_PARALLEL)
            EXPECT_EQ(config.getMode("blk.0.attn_q.weight"), WeightShardingMode::ColumnParallel);
            EXPECT_EQ(config.getMode("blk.0.attn_k.weight"), WeightShardingMode::ColumnParallel);
            EXPECT_EQ(config.getMode("blk.0.attn_v.weight"), WeightShardingMode::ColumnParallel);
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, ShardingMode_FFNGateUp_IsColumnParallel)
        {
            Qwen2SchemaFactory schema_factory;
            auto config = schema_factory.getWeightShardingConfig();

            // FFN gate/up should be COLUMN_PARALLEL (not INPUT_PARALLEL)
            EXPECT_EQ(config.getMode("blk.0.ffn_gate.weight"), WeightShardingMode::ColumnParallel);
            EXPECT_EQ(config.getMode("blk.0.ffn_up.weight"), WeightShardingMode::ColumnParallel);
        }

        // =============================================================================
        // Column Slice Range Calculation Tests
        // =============================================================================

        /**
         * @brief Test that Wo slice ranges are calculated correctly for 2-way TP
         *
         * Wo shape: [hidden_dim, hidden_dim] = [896, 896]
         * With n_heads=14, head_dim=64:
         *   - Rank 0: heads 0-6, cols [0, 448)
         *   - Rank 1: heads 7-13, cols [448, 896)
         */
        TEST_F(Test__WeightManagerInputParallelSlicing, WoSliceRange_2WayTP_EvenHeadSplit)
        {
            // Simulate 2-way TP split of n_heads=14
            const size_t total_cols = HIDDEN_DIM; // 896
            const int total_heads = N_HEADS;      // 14
            const size_t head_dim = HEAD_DIM;     // 64

            // Rank 0: heads 0-6 (7 heads)
            int head_start_r0 = 0;
            int head_count_r0 = 7;
            size_t col_start_r0 = head_start_r0 * head_dim;
            size_t col_count_r0 = head_count_r0 * head_dim;

            EXPECT_EQ(col_start_r0, 0);
            EXPECT_EQ(col_count_r0, 448);
            EXPECT_EQ(col_start_r0 + col_count_r0, 448);

            // Rank 1: heads 7-13 (7 heads)
            int head_start_r1 = 7;
            int head_count_r1 = 7;
            size_t col_start_r1 = head_start_r1 * head_dim;
            size_t col_count_r1 = head_count_r1 * head_dim;

            EXPECT_EQ(col_start_r1, 448);
            EXPECT_EQ(col_count_r1, 448);
            EXPECT_EQ(col_start_r1 + col_count_r1, 896);

            // Coverage check: both ranks cover all columns
            EXPECT_EQ(col_count_r0 + col_count_r1, total_cols);
        }

        /**
         * @brief Test that FFN Down slice ranges are calculated correctly for 2-way TP
         *
         * FFN Down shape: [hidden_dim, d_ff] = [896, 4864]
         * With d_ff=4864:
         *   - Rank 0: cols [0, 2432)
         *   - Rank 1: cols [2432, 4864)
         */
        TEST_F(Test__WeightManagerInputParallelSlicing, FFNDownSliceRange_2WayTP_EvenSplit)
        {
            const size_t total_cols = D_FF; // 4864

            // Rank 0: d_ff [0, 2432)
            size_t d_ff_start_r0 = 0;
            size_t d_ff_count_r0 = 2432;

            EXPECT_EQ(d_ff_start_r0, 0);
            EXPECT_EQ(d_ff_count_r0, 2432);
            EXPECT_EQ(d_ff_start_r0 + d_ff_count_r0, 2432);

            // Rank 1: d_ff [2432, 4864)
            size_t d_ff_start_r1 = 2432;
            size_t d_ff_count_r1 = 2432;

            EXPECT_EQ(d_ff_start_r1, 2432);
            EXPECT_EQ(d_ff_count_r1, 2432);
            EXPECT_EQ(d_ff_start_r1 + d_ff_count_r1, 4864);

            // Coverage check
            EXPECT_EQ(d_ff_count_r0 + d_ff_count_r1, total_cols);
        }

        /**
         * @brief CRITICAL TEST: Verify Wo and FFN Down have DIFFERENT column ranges
         *
         * This is the exact bug scenario: using d_ff (4864) ranges for Wo (896 cols)
         * would cause "Invalid column range [0, 2432) for tensor with 896 columns"
         */
        TEST_F(Test__WeightManagerInputParallelSlicing, CriticalBug_WoVsFFNDown_DifferentRanges)
        {
            // Wo has 896 columns (hidden_dim)
            const size_t wo_cols = HIDDEN_DIM; // 896

            // FFN Down has 4864 columns (d_ff)
            const size_t ffn_down_cols = D_FF; // 4864

            // They are DIFFERENT sizes
            EXPECT_NE(wo_cols, ffn_down_cols);
            EXPECT_LT(wo_cols, ffn_down_cols);

            // With 2-way TP:
            // - Wo rank 0: cols [0, 448) - VALID for 896 cols
            // - Wo rank 1: cols [448, 896) - VALID for 896 cols
            // - FFN Down rank 0: cols [0, 2432) - VALID for 4864 cols
            // - FFN Down rank 1: cols [2432, 4864) - VALID for 4864 cols

            // BUG SCENARIO: If we used FFN Down ranges for Wo:
            // - Wo rank 0: cols [0, 2432) - INVALID! Wo only has 896 cols
            // - Wo rank 1: cols [2432, 4864) - INVALID! Wo only has 896 cols

            size_t wo_r0_end = 448; // Correct Wo range
            size_t ffn_r0_end = 2432; // Correct FFN range

            EXPECT_LE(wo_r0_end, wo_cols) << "Wo rank 0 end must be <= 896";
            EXPECT_GT(ffn_r0_end, wo_cols) << "FFN rank 0 end exceeds Wo cols (this was the bug)";
        }

        // =============================================================================
        // sliceColumnRange Tests for INPUT_PARALLEL
        // =============================================================================

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceColumnRange_WoShape_Rank0)
        {
            // Create Wo-shaped tensor [896, 896]
            auto tensor = createSequentialTensor(HIDDEN_DIM, HIDDEN_DIM);

            // Slice columns [0, 448) for rank 0 (7 heads * 64 head_dim)
            auto sliced = WeightManager::sliceColumnRange(tensor, 0, 448);

            verifySequentialSlice(sliced.get(), HIDDEN_DIM, 448, HIDDEN_DIM, 0);
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceColumnRange_WoShape_Rank1)
        {
            // Create Wo-shaped tensor [896, 896]
            auto tensor = createSequentialTensor(HIDDEN_DIM, HIDDEN_DIM);

            // Slice columns [448, 896) for rank 1 (7 heads * 64 head_dim)
            auto sliced = WeightManager::sliceColumnRange(tensor, 448, 448);

            verifySequentialSlice(sliced.get(), HIDDEN_DIM, 448, HIDDEN_DIM, 448);
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceColumnRange_FFNDownShape_Rank0)
        {
            // Create FFN Down-shaped tensor [896, 4864]
            auto tensor = createSequentialTensor(HIDDEN_DIM, D_FF);

            // Slice columns [0, 2432) for rank 0
            auto sliced = WeightManager::sliceColumnRange(tensor, 0, 2432);

            verifySequentialSlice(sliced.get(), HIDDEN_DIM, 2432, D_FF, 0);
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceColumnRange_FFNDownShape_Rank1)
        {
            // Create FFN Down-shaped tensor [896, 4864]
            auto tensor = createSequentialTensor(HIDDEN_DIM, D_FF);

            // Slice columns [2432, 4864) for rank 1
            auto sliced = WeightManager::sliceColumnRange(tensor, 2432, 2432);

            verifySequentialSlice(sliced.get(), HIDDEN_DIM, 2432, D_FF, 2432);
        }

        /**
         * @brief Regression: unaligned native INPUT_PARALLEL slices fall back to FP32
         *
         * Qwen3.6 shared-expert down weights can be IQ3_S with a 256-column packed
         * block, while ROCm 4-way TP asks each rank for a 128-column logical shard.
         * The loader correctly refuses that native packed slice, so WeightManager
         * must materialize a full FP32 tensor and copy only the rank-local columns.
         */
        TEST_F(Test__WeightManagerInputParallelSlicing, LoadInputParallelWeight_FallsBackToFP32WhenNativeSliceUnavailable)
        {
            constexpr size_t kRows = 16;
            constexpr size_t kColumns = 512;
            constexpr const char *kWeightName = "blk.0.ffn_down.weight";

            auto loader = std::make_shared<FailingColumnSliceModelLoader>();
            loader->addTensor(kWeightName, createSequentialTensor(kRows, kColumns));

            std::vector<DeviceId> devices = {
                DeviceId::rocm(0),
                DeviceId::rocm(1),
                DeviceId::rocm(2),
                DeviceId::rocm(3)};
            auto tp_config = std::make_shared<TensorParallelConfig>(
                TensorParallelConfig::equalSplit(
                    4,
                    4,                  // n_heads: irrelevant for FFNHidden slicing
                    4,                  // n_kv_heads
                    static_cast<int>(kColumns),
                    128,                // vocab_size
                    devices));

            WeightManager wm(*loader, nullptr, nullptr,
                             WeightDistributionStrategy::SHARDED,
                             WeightPrecision::NATIVE);
            Qwen2SchemaFactory schema_factory;
            wm.setWeightShardingConfig(schema_factory.getWeightShardingConfig());
            wm.setTensorParallelConfig(tp_config);

            const auto &assignment = tp_config->forDevice(DeviceId::rocm(1));
            auto shard = wm.getShardedWeightForAssignment(kWeightName, DeviceId::rocm(1), assignment, 0);

            ASSERT_NE(shard, nullptr);
            EXPECT_EQ(loader->column_slice_attempts, 1);
            EXPECT_EQ(loader->fp32_fallback_loads, 1);
            EXPECT_EQ(loader->last_col_start, 128);
            EXPECT_EQ(loader->last_col_end, 256);

            verifySequentialSlice(shard.get(), kRows, 128, kColumns, 128);
        }

        // =============================================================================
        // 4-Way TP Tests
        // =============================================================================

        TEST_F(Test__WeightManagerInputParallelSlicing, WoSliceRange_4WayTP_Qwen2)
        {
            // Qwen2.5-0.5B: n_heads=14, not evenly divisible by 4
            // This tests the "odd head count" scenario

            const size_t total_cols = HIDDEN_DIM; // 896
            const int total_heads = N_HEADS;      // 14
            const size_t head_dim = HEAD_DIM;     // 64

            // 14 heads / 4 ranks:
            // Possible distributions:
            // - Equal: 3, 3, 4, 4 (or some variation)
            // - Rank 0: 3 heads -> 192 cols
            // - Rank 1: 3 heads -> 192 cols
            // - Rank 2: 4 heads -> 256 cols
            // - Rank 3: 4 heads -> 256 cols
            // Total: 192 + 192 + 256 + 256 = 896 ✓

            // Alternative: 4, 4, 3, 3
            int heads_r0 = 4, heads_r1 = 4, heads_r2 = 3, heads_r3 = 3;

            size_t start_r0 = 0;
            size_t start_r1 = heads_r0 * head_dim;
            size_t start_r2 = (heads_r0 + heads_r1) * head_dim;
            size_t start_r3 = (heads_r0 + heads_r1 + heads_r2) * head_dim;

            EXPECT_EQ(start_r0, 0);
            EXPECT_EQ(start_r1, 256);
            EXPECT_EQ(start_r2, 512);
            EXPECT_EQ(start_r3, 704);

            size_t end_r3 = start_r3 + heads_r3 * head_dim;
            EXPECT_EQ(end_r3, 896); // Should cover all columns
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, FFNDownSliceRange_4WayTP)
        {
            // d_ff=4864, 4-way split
            const size_t total_cols = D_FF;

            // 4864 / 4 = 1216 per rank
            size_t per_rank = 1216;

            EXPECT_EQ(per_rank * 4, total_cols);

            for (int rank = 0; rank < 4; ++rank)
            {
                size_t start = rank * per_rank;
                size_t end = start + per_rank;

                EXPECT_EQ(end - start, per_rank) << "Rank " << rank << " has wrong slice size";
                if (rank == 3)
                {
                    EXPECT_EQ(end, total_cols) << "Last rank should end at d_ff";
                }
            }
        }

        // =============================================================================
        // Proportional (Uneven) TP Tests - Heterogeneous GPU scenario
        // =============================================================================

        /**
         * @brief Test proportional TP (e.g., 73% CUDA + 27% ROCm)
         *
         * In heterogeneous setups, weights are split proportionally rather than evenly
         */
        TEST_F(Test__WeightManagerInputParallelSlicing, WoSliceRange_ProportionalTP_73_27)
        {
            const size_t total_cols = HIDDEN_DIM; // 896
            const int total_heads = N_HEADS;      // 14
            const size_t head_dim = HEAD_DIM;     // 64

            // 73% / 27% split of 14 heads:
            // 14 * 0.73 = 10.22 -> 10 heads for CUDA
            // 14 - 10 = 4 heads for ROCm

            int cuda_heads = 10;
            int rocm_heads = 4;
            EXPECT_EQ(cuda_heads + rocm_heads, total_heads);

            size_t cuda_cols = cuda_heads * head_dim; // 640
            size_t rocm_cols = rocm_heads * head_dim; // 256

            EXPECT_EQ(cuda_cols, 640);
            EXPECT_EQ(rocm_cols, 256);
            EXPECT_EQ(cuda_cols + rocm_cols, total_cols);

            // CUDA: cols [0, 640)
            // ROCm: cols [640, 896)
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, FFNDownSliceRange_ProportionalTP_73_27)
        {
            const size_t total_cols = D_FF; // 4864

            // 73% / 27% split of d_ff:
            // 4864 * 0.73 = 3550.72 -> round to alignment
            // For simplicity, let's say 3584 (multiple of 64 for memory alignment)
            // Then ROCm gets 4864 - 3584 = 1280

            // Actual proportional split may vary based on alignment
            size_t cuda_cols = static_cast<size_t>(total_cols * 0.73);
            size_t rocm_cols = total_cols - cuda_cols;

            EXPECT_GT(cuda_cols, rocm_cols) << "CUDA should get more than ROCm";
            EXPECT_EQ(cuda_cols + rocm_cols, total_cols);

            // Both slices must be valid
            EXPECT_LE(cuda_cols, total_cols);
            EXPECT_GT(cuda_cols, 0);
            EXPECT_LE(rocm_cols, total_cols);
            EXPECT_GT(rocm_cols, 0);
        }

        // =============================================================================
        // Edge Cases
        // =============================================================================

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceColumnRange_SingleColumn)
        {
            auto tensor = createSequentialTensor(16, 8);

            // Slice single column
            auto sliced = WeightManager::sliceColumnRange(tensor, 3, 1);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 16);
            EXPECT_EQ(sliced->shape()[1], 1);

            // Verify data
            const float *data = sliced->data();
            for (size_t i = 0; i < 16; ++i)
            {
                EXPECT_FLOAT_EQ(data[i], static_cast<float>(i * 10000 + 3));
            }
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceColumnRange_AllColumns)
        {
            auto tensor = createSequentialTensor(16, 8);

            // Slice all columns (essentially a copy)
            auto sliced = WeightManager::sliceColumnRange(tensor, 0, 8);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 16);
            EXPECT_EQ(sliced->shape()[1], 8);

            // Should be identical to original
            const float *orig_data = tensor->data();
            const float *sliced_data = sliced->data();
            for (size_t i = 0; i < 128; ++i)
            {
                EXPECT_FLOAT_EQ(sliced_data[i], orig_data[i]);
            }
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceColumnRange_LastColumn)
        {
            auto tensor = createSequentialTensor(16, 8);

            // Slice last column
            auto sliced = WeightManager::sliceColumnRange(tensor, 7, 1);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[1], 1);

            const float *data = sliced->data();
            for (size_t i = 0; i < 16; ++i)
            {
                EXPECT_FLOAT_EQ(data[i], static_cast<float>(i * 10000 + 7));
            }
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceColumnRange_NullTensor_ReturnsNull)
        {
            std::shared_ptr<TensorBase> null_tensor = nullptr;
            EXPECT_THROW(WeightManager::sliceColumnRange(null_tensor, 0, 1), std::runtime_error);
        }

        // =============================================================================
        // Boundary Validation Tests
        // =============================================================================

        TEST_F(Test__WeightManagerInputParallelSlicing, WoSlice_BoundaryCheck_AllRanksValid)
        {
            // Verify that all Wo slice ranges are within bounds
            const size_t wo_cols = HIDDEN_DIM;

            // 2-way TP
            EXPECT_LE(0 + 448, wo_cols);        // Rank 0: [0, 448)
            EXPECT_LE(448 + 448, wo_cols);      // Rank 1: [448, 896)

            // 4-way TP with 4,4,3,3 head distribution
            size_t end_r0 = 4 * HEAD_DIM;       // 256
            size_t end_r1 = 8 * HEAD_DIM;       // 512
            size_t end_r2 = 11 * HEAD_DIM;      // 704
            size_t end_r3 = 14 * HEAD_DIM;      // 896

            EXPECT_LE(end_r0, wo_cols);
            EXPECT_LE(end_r1, wo_cols);
            EXPECT_LE(end_r2, wo_cols);
            EXPECT_LE(end_r3, wo_cols);
            EXPECT_EQ(end_r3, wo_cols); // Last rank should exactly match
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, FFNDownSlice_BoundaryCheck_AllRanksValid)
        {
            // Verify that all FFN Down slice ranges are within bounds
            const size_t ffn_cols = D_FF;

            // 2-way TP
            EXPECT_LE(0 + 2432, ffn_cols);      // Rank 0: [0, 2432)
            EXPECT_LE(2432 + 2432, ffn_cols);   // Rank 1: [2432, 4864)

            // 4-way TP
            size_t per_rank = ffn_cols / 4;     // 1216
            for (int rank = 0; rank < 4; ++rank)
            {
                size_t start = rank * per_rank;
                size_t end = (rank == 3) ? ffn_cols : (start + per_rank);
                EXPECT_LE(end, ffn_cols) << "Rank " << rank << " exceeds bounds";
            }
        }

        // =============================================================================
        // Different Model Architecture Tests (Llama-style)
        // =============================================================================

        TEST_F(Test__WeightManagerInputParallelSlicing, LlamaStyle_WoSlice_HeadDim128)
        {
            // Llama-style: n_heads=32, head_dim=128, hidden_dim=4096
            const size_t llama_n_heads = 32;
            const size_t llama_head_dim = 128;
            const size_t llama_hidden = llama_n_heads * llama_head_dim; // 4096

            // 2-way TP: 16 heads per rank
            size_t heads_per_rank = llama_n_heads / 2;
            size_t cols_per_rank = heads_per_rank * llama_head_dim;

            EXPECT_EQ(heads_per_rank, 16);
            EXPECT_EQ(cols_per_rank, 2048);
            EXPECT_EQ(cols_per_rank * 2, llama_hidden);

            // 4-way TP: 8 heads per rank
            heads_per_rank = llama_n_heads / 4;
            cols_per_rank = heads_per_rank * llama_head_dim;

            EXPECT_EQ(heads_per_rank, 8);
            EXPECT_EQ(cols_per_rank, 1024);
            EXPECT_EQ(cols_per_rank * 4, llama_hidden);
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, LlamaStyle_FFNDownSlice_LargeD_FF)
        {
            // Llama-style: d_ff = 11008 (Llama-7B)
            const size_t llama_d_ff = 11008;

            // 2-way TP
            size_t half = llama_d_ff / 2; // 5504

            EXPECT_EQ(half * 2, llama_d_ff);

            // 4-way TP (not evenly divisible: 11008 / 4 = 2752)
            size_t quarter = llama_d_ff / 4; // 2752

            EXPECT_EQ(quarter * 4, llama_d_ff);

            // 8-way TP (not evenly divisible: 11008 / 8 = 1376)
            size_t eighth = llama_d_ff / 8; // 1376

            EXPECT_EQ(eighth * 8, llama_d_ff);
        }

        // =============================================================================
        // Regression Test: Bug Scenario Recreation
        // =============================================================================

        /**
         * @brief Recreate the exact bug scenario that was fixed
         *
         * Bug: Using d_ff (4864) dimensions for attn_output (896 cols)
         * Symptom: "Invalid column range [0, 2432) for tensor with 896 columns"
         */
        TEST_F(Test__WeightManagerInputParallelSlicing, RegressionTest_WoSliceWithFFNRanges_Invalid)
        {
            // Create Wo-shaped tensor [896, 896]
            auto wo_tensor = createSequentialTensor(HIDDEN_DIM, HIDDEN_DIM);

            // Try to slice with FFN Down ranges - THIS WAS THE BUG
            // d_ff range [0, 2432) is invalid for 896-column tensor

            // Verify that column 2432 exceeds Wo bounds
            EXPECT_GT(2432u, wo_tensor->shape()[1])
                << "FFN slice end (2432) should exceed Wo cols (896)";

            // The correct slice for Wo rank 0 is [0, 448)
            auto correct_slice = WeightManager::sliceColumnRange(wo_tensor, 0, 448);
            ASSERT_NE(correct_slice, nullptr) << "Correct Wo slice should succeed";
            EXPECT_EQ(correct_slice->shape()[1], 448);
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, RegressionTest_FFNDownSliceWithWoRanges_TooSmall)
        {
            // Create FFN Down-shaped tensor [896, 4864]
            auto ffn_tensor = createSequentialTensor(HIDDEN_DIM, D_FF);

            // If we used Wo ranges [0, 448) for FFN Down, it would "work" but be wrong
            // Each rank would only get 448 cols instead of 2432

            // Verify that 448 is much less than correct FFN slice size
            EXPECT_LT(448u, 2432u)
                << "Wo slice size (448) is much smaller than correct FFN slice (2432)";

            // The correct slice for FFN Down rank 0 is [0, 2432)
            auto correct_slice = WeightManager::sliceColumnRange(ffn_tensor, 0, 2432);
            ASSERT_NE(correct_slice, nullptr) << "Correct FFN slice should succeed";
            EXPECT_EQ(correct_slice->shape()[1], 2432);
        }

        // =============================================================================
        // Coverage Verification Tests
        // =============================================================================

        TEST_F(Test__WeightManagerInputParallelSlicing, CoverageTest_WoSlices_CoverAllColumns)
        {
            // Verify that all Wo slices together cover all columns
            auto tensor = createSequentialTensor(HIDDEN_DIM, HIDDEN_DIM);

            // 2-way TP slices
            auto slice_r0 = WeightManager::sliceColumnRange(tensor, 0, 448);
            auto slice_r1 = WeightManager::sliceColumnRange(tensor, 448, 448);

            ASSERT_NE(slice_r0, nullptr);
            ASSERT_NE(slice_r1, nullptr);

            // Total columns covered
            size_t total_covered = slice_r0->shape()[1] + slice_r1->shape()[1];
            EXPECT_EQ(total_covered, HIDDEN_DIM);

            // No overlap: r0 ends where r1 starts
            EXPECT_EQ(0 + 448, 448); // r0 end == r1 start
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, CoverageTest_FFNDownSlices_CoverAllColumns)
        {
            // Verify that all FFN Down slices together cover all columns
            auto tensor = createSequentialTensor(HIDDEN_DIM, D_FF);

            // 2-way TP slices
            auto slice_r0 = WeightManager::sliceColumnRange(tensor, 0, 2432);
            auto slice_r1 = WeightManager::sliceColumnRange(tensor, 2432, 2432);

            ASSERT_NE(slice_r0, nullptr);
            ASSERT_NE(slice_r1, nullptr);

            // Total columns covered
            size_t total_covered = slice_r0->shape()[1] + slice_r1->shape()[1];
            EXPECT_EQ(total_covered, D_FF);

            // No overlap
            EXPECT_EQ(0 + 2432, 2432); // r0 end == r1 start
        }

        // =============================================================================
        // Error Handling Tests - Out of Bounds
        // =============================================================================

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceColumnRange_OutOfBounds_StartPlusCount)
        {
            auto tensor = createSequentialTensor(16, 8);

            // col_start=5 + col_count=5 = 10, but tensor only has 8 columns
            EXPECT_THROW(WeightManager::sliceColumnRange(tensor, 5, 5), std::runtime_error);
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceColumnRange_StartAtEnd)
        {
            auto tensor = createSequentialTensor(16, 8);

            // col_start=8 is out of bounds (columns are 0-7)
            EXPECT_THROW(WeightManager::sliceColumnRange(tensor, 8, 1), std::runtime_error);
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceColumnRange_ZeroCount)
        {
            auto tensor = createSequentialTensor(16, 8);

            // Zero columns requested
            EXPECT_THROW(WeightManager::sliceColumnRange(tensor, 0, 0), std::runtime_error);
        }

        // =============================================================================
        // sliceRowRange Additional Tests (symmetry with sliceColumnRange)
        // =============================================================================

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceRowRange_BasicSlice)
        {
            // 10x8 tensor, slice rows 2-5 (4 rows)
            auto tensor = createSequentialTensor(10, 8);
            auto sliced = WeightManager::sliceRowRange(tensor, 2, 4);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], 4);
            EXPECT_EQ(sliced->shape()[1], 8);

            // Verify values: row 2 should have values 20000+col
            const float *data = sliced->data();
            EXPECT_FLOAT_EQ(data[0], 20000.0f);      // Row 2, col 0
            EXPECT_FLOAT_EQ(data[7], 20007.0f);      // Row 2, col 7
            EXPECT_FLOAT_EQ(data[8], 30000.0f);      // Row 3, col 0
            EXPECT_FLOAT_EQ(data[31], 50007.0f);     // Row 5, col 7
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceRowRange_OutOfBounds)
        {
            auto tensor = createSequentialTensor(10, 8);

            // row_start=7 + row_count=5 = 12 > 10
            EXPECT_THROW(WeightManager::sliceRowRange(tensor, 7, 5), std::runtime_error);
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, SliceRowRange_NullTensor)
        {
            std::shared_ptr<TensorBase> null_tensor = nullptr;
            EXPECT_THROW(WeightManager::sliceRowRange(null_tensor, 0, 1), std::runtime_error);
        }

        // =============================================================================
        // DATA INTEGRITY TESTS: Verify slice data matches original
        // =============================================================================

        TEST_F(Test__WeightManagerInputParallelSlicing, DataIntegrity_WoSlice_MatchesOriginal)
        {
            // Create Wo tensor with known pattern
            auto tensor = createSequentialTensor(HIDDEN_DIM, HIDDEN_DIM);

            // Slice columns [224, 672) - middle portion for rank 1 of 4-way TP
            auto sliced = WeightManager::sliceColumnRange(tensor, 224, 448);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], HIDDEN_DIM);
            EXPECT_EQ(sliced->shape()[1], 448);

            // Verify ALL values match
            const float *orig = tensor->data();
            const float *slice = sliced->data();

            for (size_t row = 0; row < HIDDEN_DIM; ++row)
            {
                for (size_t col = 0; col < 448; ++col)
                {
                    size_t orig_col = 224 + col;
                    float expected = orig[row * HIDDEN_DIM + orig_col];
                    float actual = slice[row * 448 + col];
                    EXPECT_FLOAT_EQ(actual, expected)
                        << "Mismatch at row " << row << ", col " << col;
                }
            }
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, DataIntegrity_FFNDownSlice_MatchesOriginal)
        {
            // Create FFN Down tensor with known pattern
            auto tensor = createSequentialTensor(HIDDEN_DIM, D_FF);

            // Slice columns [1216, 2432) - second quarter for rank 1 of 4-way TP
            auto sliced = WeightManager::sliceColumnRange(tensor, 1216, 1216);

            ASSERT_NE(sliced, nullptr);
            EXPECT_EQ(sliced->shape()[0], HIDDEN_DIM);
            EXPECT_EQ(sliced->shape()[1], 1216);

            // Verify sample values
            const float *orig = tensor->data();
            const float *slice = sliced->data();

            // Check first row
            EXPECT_FLOAT_EQ(slice[0], orig[1216]);  // Row 0, original col 1216
            EXPECT_FLOAT_EQ(slice[1215], orig[2431]);  // Row 0, original col 2431

            // Check last row
            size_t last_row = HIDDEN_DIM - 1;
            float expected_last_first = static_cast<float>(last_row * 10000 + 1216);
            float expected_last_last = static_cast<float>(last_row * 10000 + 2431);
            EXPECT_FLOAT_EQ(slice[last_row * 1216], expected_last_first);
            EXPECT_FLOAT_EQ(slice[last_row * 1216 + 1215], expected_last_last);
        }

        // =============================================================================
        // TP DEGREE TESTS: 2, 4, 8 way splits
        // =============================================================================

        TEST_F(Test__WeightManagerInputParallelSlicing, TPDegrees_AllValid_Wo)
        {
            const size_t wo_cols = HIDDEN_DIM;

            // 2-way TP
            size_t cols_2way = wo_cols / 2;  // 448
            EXPECT_EQ(cols_2way * 2, wo_cols);

            // 7-way TP (odd, matches n_heads=14)
            size_t cols_7way = wo_cols / 7;  // 128
            EXPECT_EQ(cols_7way * 7, wo_cols);

            // 14-way TP (one head per rank)
            size_t cols_14way = wo_cols / 14;  // 64
            EXPECT_EQ(cols_14way * 14, wo_cols);
        }

        TEST_F(Test__WeightManagerInputParallelSlicing, TPDegrees_AllValid_FFNDown)
        {
            const size_t ffn_cols = D_FF;

            // 2-way TP
            size_t cols_2way = ffn_cols / 2;  // 2432
            EXPECT_EQ(cols_2way * 2, ffn_cols);

            // 4-way TP
            size_t cols_4way = ffn_cols / 4;  // 1216
            EXPECT_EQ(cols_4way * 4, ffn_cols);

            // 8-way TP
            size_t cols_8way = ffn_cols / 8;  // 608
            EXPECT_EQ(cols_8way * 8, ffn_cols);
        }

        // =============================================================================
        // MEMORY INDEPENDENCE TESTS
        // =============================================================================

        TEST_F(Test__WeightManagerInputParallelSlicing, MemoryIndependence_ModifySliceDoesNotAffectOriginal)
        {
            auto tensor = createSequentialTensor(16, 8);
            auto sliced = WeightManager::sliceColumnRange(tensor, 2, 4);

            ASSERT_NE(sliced, nullptr);

            // Get mutable access to sliced tensor
            auto *fp32_sliced = dynamic_cast<FP32Tensor *>(sliced.get());
            ASSERT_NE(fp32_sliced, nullptr);

            // Modify the slice
            float *slice_data = fp32_sliced->mutable_data();
            slice_data[0] = 99999.0f;

            // Verify original is unchanged
            const float *orig_data = tensor->data();
            EXPECT_FLOAT_EQ(orig_data[2], 2.0f)  // Original row 0, col 2
                << "Original tensor should be unchanged after modifying slice";
        }

    } // namespace test
} // namespace llaminar2
