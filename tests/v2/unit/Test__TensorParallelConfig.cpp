/**
 * @file Test__TensorParallelConfig.cpp
 * @brief Unit tests for TensorParallelConfig infrastructure
 *
 * Tests the configuration structures that enable proportional tensor parallelism
 * for heterogeneous GPU configurations (e.g., NVIDIA 73% + AMD 27%).
 */

#include <gtest/gtest.h>
#include "config/TensorParallelConfig.h"

namespace llaminar2
{
    namespace test
    {

        class Test__TensorParallelConfig : public ::testing::Test
        {
        protected:
            // Qwen2.5-7B parameters (common test case)
            static constexpr int QWEN_7B_HEADS = 28;
            static constexpr int QWEN_7B_KV_HEADS = 4;
            static constexpr int QWEN_7B_D_FF = 18944;
            static constexpr int QWEN_7B_VOCAB = 151936;

            // Smaller test parameters
            static constexpr int TEST_HEADS = 28;
            static constexpr int TEST_KV_HEADS = 4;
            static constexpr int TEST_D_FF = 1024;
            static constexpr int TEST_VOCAB = 32000;
        };

        // =============================================================================
        // Equal Split Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig, EqualSplit_TwoDevices)
        {
            auto config = TensorParallelConfig::equalSplit(
                2, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_EQ(config.worldSize(), 2);
            EXPECT_FALSE(config.isProportional());
            EXPECT_TRUE(config.validate());

            // Check totals match
            EXPECT_EQ(config.totalHeads(), TEST_HEADS);
            EXPECT_EQ(config.totalKVHeads(), TEST_KV_HEADS);
            EXPECT_EQ(config.totalDFF(), TEST_D_FF);
            EXPECT_EQ(config.totalVocab(), TEST_VOCAB);

            // Check equal split: 28 heads / 2 = 14 each
            const auto &rank0 = config.forRank(0);
            const auto &rank1 = config.forRank(1);

            EXPECT_EQ(rank0.head_count, 14);
            EXPECT_EQ(rank1.head_count, 14);
            EXPECT_EQ(rank0.head_start, 0);
            EXPECT_EQ(rank1.head_start, 14);

            // KV heads: 4 / 2 = 2 each
            EXPECT_EQ(rank0.kv_head_count, 2);
            EXPECT_EQ(rank1.kv_head_count, 2);
        }

        TEST_F(Test__TensorParallelConfig, EqualSplit_WithRemainder)
        {
            // 28 heads, 3 devices: can't divide evenly
            auto config = TensorParallelConfig::equalSplit(
                3, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_EQ(config.worldSize(), 3);
            EXPECT_TRUE(config.validate());
            EXPECT_EQ(config.totalHeads(), TEST_HEADS);

            // Check distribution handles remainder
            const auto &rank0 = config.forRank(0);
            const auto &rank1 = config.forRank(1);
            const auto &rank2 = config.forRank(2);

            // 28 / 3 = 9.33, so we expect 10 + 9 + 9 or 9 + 10 + 9, etc.
            int total = rank0.head_count + rank1.head_count + rank2.head_count;
            EXPECT_EQ(total, TEST_HEADS);

            // Each rank should have at least floor(28/3) = 9 heads
            EXPECT_GE(rank0.head_count, 9);
            EXPECT_GE(rank1.head_count, 9);
            EXPECT_GE(rank2.head_count, 9);
        }

        TEST_F(Test__TensorParallelConfig, EqualSplit_WithCustomDevices)
        {
            std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};

            auto config = TensorParallelConfig::equalSplit(
                2, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB, devices);

            EXPECT_EQ(config.worldSize(), 2);

            // Check devices match
            const auto &cuda_assignment = config.forDevice(DeviceId::cuda(0));
            const auto &rocm_assignment = config.forDevice(DeviceId::rocm(0));

            EXPECT_TRUE(cuda_assignment.device.is_cuda());
            EXPECT_TRUE(rocm_assignment.device.is_rocm());
            EXPECT_EQ(cuda_assignment.local_rank, 0);
            EXPECT_EQ(rocm_assignment.local_rank, 1);
        }

        // =============================================================================
        // Proportional Split Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig, ProportionalSplit_73_27)
        {
            // NVIDIA gets 73%, AMD gets 27%
            std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
            std::vector<float> fractions = {0.73f, 0.27f};

            auto config = TensorParallelConfig::proportionalSplit(
                devices, fractions, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_EQ(config.worldSize(), 2);
            EXPECT_TRUE(config.isProportional());
            EXPECT_TRUE(config.validate());

            const auto &cuda = config.forDevice(DeviceId::cuda(0));
            const auto &rocm = config.forDevice(DeviceId::rocm(0));

            // 28 heads × 73% ≈ 20.44 → round to 20
            // 28 heads × 27% ≈ 7.56 → 8 (to make total = 28)
            EXPECT_EQ(cuda.head_count + rocm.head_count, TEST_HEADS);

            // CUDA should get more heads
            EXPECT_GT(cuda.head_count, rocm.head_count);

            // Work fractions should be normalized
            EXPECT_NEAR(cuda.work_fraction, 0.73f, 0.01f);
            EXPECT_NEAR(rocm.work_fraction, 0.27f, 0.01f);

            // Ranges should be contiguous
            EXPECT_EQ(cuda.head_start, 0);
            EXPECT_EQ(rocm.head_start, cuda.headEnd());
        }

        TEST_F(Test__TensorParallelConfig, ProportionalSplit_GQAAlignment)
        {
            // Test that KV heads maintain proper ratio with Q heads
            std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
            std::vector<float> fractions = {0.70f, 0.30f};

            auto config = TensorParallelConfig::proportionalSplit(
                devices, fractions, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            const auto &cuda = config.forDevice(DeviceId::cuda(0));
            const auto &rocm = config.forDevice(DeviceId::rocm(0));

            // Total KV heads must sum correctly
            EXPECT_EQ(cuda.kv_head_count + rocm.kv_head_count, TEST_KV_HEADS);

            // KV heads should follow similar ratio as Q heads
            // (might not be exact due to small numbers)
            if (TEST_KV_HEADS >= 4)
            {
                // Larger share should have more KV heads
                EXPECT_GE(cuda.kv_head_count, rocm.kv_head_count);
            }
        }

        TEST_F(Test__TensorParallelConfig, ProportionalSplit_FractionsNormalized)
        {
            // Fractions don't sum to 1.0 - should be normalized
            std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
            std::vector<float> fractions = {73.0f, 27.0f}; // Sum = 100, not 1.0

            auto config = TensorParallelConfig::proportionalSplit(
                devices, fractions, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            const auto &cuda = config.forDevice(DeviceId::cuda(0));
            const auto &rocm = config.forDevice(DeviceId::rocm(0));

            // Should still be normalized to ~0.73 and ~0.27
            EXPECT_NEAR(cuda.work_fraction, 0.73f, 0.01f);
            EXPECT_NEAR(rocm.work_fraction, 0.27f, 0.01f);
        }

        // =============================================================================
        // Single Device Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig, SingleDevice)
        {
            auto config = TensorParallelConfig::singleDevice(
                DeviceId::cuda(0), TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_EQ(config.worldSize(), 1);
            EXPECT_FALSE(config.isProportional());
            EXPECT_TRUE(config.validate());

            const auto &assignment = config.forRank(0);
            EXPECT_EQ(assignment.head_count, TEST_HEADS);
            EXPECT_EQ(assignment.kv_head_count, TEST_KV_HEADS);
            EXPECT_EQ(assignment.d_ff_count, TEST_D_FF);
            EXPECT_EQ(assignment.vocab_count, TEST_VOCAB);
            EXPECT_EQ(assignment.work_fraction, 1.0f);
        }

        TEST_F(Test__TensorParallelConfig, SingleDevice_CPU)
        {
            auto config = TensorParallelConfig::singleDevice(
                DeviceId::cpu(), TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_EQ(config.worldSize(), 1);
            EXPECT_TRUE(config.forRank(0).device.is_cpu());
        }

        // =============================================================================
        // Validation Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig, Validate_OverlappingRanges)
        {
            // Manually create overlapping assignments
            DeviceShardingAssignment a1;
            a1.device = DeviceId::cuda(0);
            a1.local_rank = 0;
            a1.head_start = 0;
            a1.head_count = 16; // Ends at 16
            a1.kv_head_start = 0;
            a1.kv_head_count = 2;
            a1.d_ff_start = 0;
            a1.d_ff_count = 512;
            a1.vocab_start = 0;
            a1.vocab_count = 16000;

            DeviceShardingAssignment a2;
            a2.device = DeviceId::cuda(1);
            a2.local_rank = 1;
            a2.head_start = 14; // Overlaps! Should start at 16
            a2.head_count = 14;
            a2.kv_head_start = 2;
            a2.kv_head_count = 2;
            a2.d_ff_start = 512;
            a2.d_ff_count = 512;
            a2.vocab_start = 16000;
            a2.vocab_count = 16000;

            TensorParallelConfig config({a1, a2});

            EXPECT_FALSE(config.validate());
            EXPECT_TRUE(config.validationError().find("Overlapping") != std::string::npos);
        }

        TEST_F(Test__TensorParallelConfig, Validate_GapInRanges)
        {
            // Create assignments with a gap
            DeviceShardingAssignment a1;
            a1.device = DeviceId::cuda(0);
            a1.local_rank = 0;
            a1.head_start = 0;
            a1.head_count = 10; // Ends at 10
            a1.kv_head_start = 0;
            a1.kv_head_count = 2;
            a1.d_ff_start = 0;
            a1.d_ff_count = 512;
            a1.vocab_start = 0;
            a1.vocab_count = 16000;

            DeviceShardingAssignment a2;
            a2.device = DeviceId::cuda(1);
            a2.local_rank = 1;
            a2.head_start = 12; // Gap! Should be 10
            a2.head_count = 10;
            a2.kv_head_start = 2;
            a2.kv_head_count = 2;
            a2.d_ff_start = 512;
            a2.d_ff_count = 512;
            a2.vocab_start = 16000;
            a2.vocab_count = 16000;

            TensorParallelConfig config({a1, a2});

            EXPECT_FALSE(config.validate());
            EXPECT_TRUE(config.validationError().find("Gap") != std::string::npos);
        }

        TEST_F(Test__TensorParallelConfig, Validate_DuplicateDevice)
        {
            DeviceShardingAssignment a1;
            a1.device = DeviceId::cuda(0);
            a1.local_rank = 0;
            a1.head_start = 0;
            a1.head_count = 14;
            a1.kv_head_start = 0;
            a1.kv_head_count = 2;
            a1.d_ff_start = 0;
            a1.d_ff_count = 512;
            a1.vocab_start = 0;
            a1.vocab_count = 16000;

            DeviceShardingAssignment a2;
            a2.device = DeviceId::cuda(0); // Same device!
            a2.local_rank = 1;
            a2.head_start = 14;
            a2.head_count = 14;
            a2.kv_head_start = 2;
            a2.kv_head_count = 2;
            a2.d_ff_start = 512;
            a2.d_ff_count = 512;
            a2.vocab_start = 16000;
            a2.vocab_count = 16000;

            TensorParallelConfig config({a1, a2});

            EXPECT_FALSE(config.validate());
            EXPECT_TRUE(config.validationError().find("Duplicate") != std::string::npos);
        }

        // =============================================================================
        // Accessor Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig, ForDevice_NotFound)
        {
            auto config = TensorParallelConfig::equalSplit(
                2, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            // Default devices are cuda(0), cuda(1)
            EXPECT_THROW(config.forDevice(DeviceId::rocm(0)), std::out_of_range);
            EXPECT_THROW(config.forDevice(DeviceId::cpu()), std::out_of_range);
        }

        TEST_F(Test__TensorParallelConfig, ForRank_OutOfBounds)
        {
            auto config = TensorParallelConfig::equalSplit(
                2, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_THROW(config.forRank(-1), std::out_of_range);
            EXPECT_THROW(config.forRank(2), std::out_of_range);
            EXPECT_THROW(config.forRank(100), std::out_of_range);
        }

        // =============================================================================
        // D_FF Alignment Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig, DFFAlignment_32Aligned)
        {
            // d_ff should be aligned to 32-element boundaries
            std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
            std::vector<float> fractions = {0.73f, 0.27f};

            auto config = TensorParallelConfig::proportionalSplit(
                devices, fractions, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            const auto &cuda = config.forDevice(DeviceId::cuda(0));
            const auto &rocm = config.forDevice(DeviceId::rocm(0));

            // Both d_ff counts should be divisible by 32
            EXPECT_EQ(cuda.d_ff_count % 32, 0)
                << "CUDA d_ff_count=" << cuda.d_ff_count << " not 32-aligned";
            EXPECT_EQ(rocm.d_ff_count % 32, 0)
                << "ROCm d_ff_count=" << rocm.d_ff_count << " not 32-aligned";

            // Total should still equal original (might have slight adjustment)
            EXPECT_EQ(cuda.d_ff_count + rocm.d_ff_count, TEST_D_FF);
        }

        TEST_F(Test__TensorParallelConfig, DFFAlignment_LargeDFF)
        {
            // Test with Qwen2.5-7B's actual d_ff = 18944
            std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
            std::vector<float> fractions = {0.73f, 0.27f};

            auto config = TensorParallelConfig::proportionalSplit(
                devices, fractions,
                QWEN_7B_HEADS, QWEN_7B_KV_HEADS, QWEN_7B_D_FF, QWEN_7B_VOCAB);

            const auto &cuda = config.forDevice(DeviceId::cuda(0));
            const auto &rocm = config.forDevice(DeviceId::rocm(0));

            EXPECT_EQ(cuda.d_ff_count % 32, 0);
            EXPECT_EQ(rocm.d_ff_count % 32, 0);
            EXPECT_EQ(cuda.d_ff_count + rocm.d_ff_count, QWEN_7B_D_FF);
        }

        // =============================================================================
        // Edge Case Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig, EqualSplit_InvalidWorldSize)
        {
            EXPECT_THROW(
                TensorParallelConfig::equalSplit(0, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB),
                std::invalid_argument);

            EXPECT_THROW(
                TensorParallelConfig::equalSplit(-1, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB),
                std::invalid_argument);
        }

        TEST_F(Test__TensorParallelConfig, EqualSplit_MoreDevicesThanHeads)
        {
            // Can't split 4 heads across 8 devices
            EXPECT_THROW(
                TensorParallelConfig::equalSplit(8, 4, 2, TEST_D_FF, TEST_VOCAB),
                std::invalid_argument);
        }

        TEST_F(Test__TensorParallelConfig, ProportionalSplit_NegativeFraction)
        {
            std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
            std::vector<float> fractions = {0.73f, -0.27f}; // Negative!

            EXPECT_THROW(
                TensorParallelConfig::proportionalSplit(
                    devices, fractions, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB),
                std::invalid_argument);
        }

        TEST_F(Test__TensorParallelConfig, ProportionalSplit_ZeroFractions)
        {
            std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
            std::vector<float> fractions = {0.0f, 0.0f}; // All zero!

            EXPECT_THROW(
                TensorParallelConfig::proportionalSplit(
                    devices, fractions, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB),
                std::invalid_argument);
        }

        TEST_F(Test__TensorParallelConfig, ProportionalSplit_MismatchedSizes)
        {
            std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
            std::vector<float> fractions = {0.73f}; // Only one fraction!

            EXPECT_THROW(
                TensorParallelConfig::proportionalSplit(
                    devices, fractions, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB),
                std::invalid_argument);
        }

        // =============================================================================
        // String Representation Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig, ToString_ContainsRelevantInfo)
        {
            auto config = TensorParallelConfig::equalSplit(
                2, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            std::string str = config.toString();

            EXPECT_TRUE(str.find("world_size=2") != std::string::npos);
            EXPECT_TRUE(str.find("heads=" + std::to_string(TEST_HEADS)) != std::string::npos);
            EXPECT_TRUE(str.find("Rank 0") != std::string::npos);
            EXPECT_TRUE(str.find("Rank 1") != std::string::npos);
        }

        TEST_F(Test__TensorParallelConfig, DeviceShardingAssignment_ToString)
        {
            auto config = TensorParallelConfig::singleDevice(
                DeviceId::cuda(0), TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            std::string str = config.forRank(0).toString();

            EXPECT_TRUE(str.find("Rank 0") != std::string::npos);
            EXPECT_TRUE(str.find("CUDA:0") != std::string::npos);
            EXPECT_TRUE(str.find("heads") != std::string::npos);
        }

        // =============================================================================
        // Real-World Configuration Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig, Qwen2_5_7B_HeterogeneousConfig)
        {
            // RTX 3090 (CUDA) + MI50 (ROCm) configuration
            // CUDA gets ~73% based on relative performance
            std::vector<DeviceId> devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
            std::vector<float> fractions = {0.73f, 0.27f};

            auto config = TensorParallelConfig::proportionalSplit(
                devices, fractions,
                QWEN_7B_HEADS, QWEN_7B_KV_HEADS, QWEN_7B_D_FF, QWEN_7B_VOCAB);

            EXPECT_TRUE(config.validate()) << config.validationError();
            EXPECT_TRUE(config.isProportional());

            // Verify totals
            EXPECT_EQ(config.totalHeads(), QWEN_7B_HEADS);
            EXPECT_EQ(config.totalKVHeads(), QWEN_7B_KV_HEADS);
            EXPECT_EQ(config.totalDFF(), QWEN_7B_D_FF);
            EXPECT_EQ(config.totalVocab(), QWEN_7B_VOCAB);

            // CUDA should have majority
            const auto &cuda = config.forDevice(DeviceId::cuda(0));
            const auto &rocm = config.forDevice(DeviceId::rocm(0));

            EXPECT_GT(cuda.head_count, rocm.head_count);
            EXPECT_GT(cuda.d_ff_count, rocm.d_ff_count);
            EXPECT_GT(cuda.vocab_count, rocm.vocab_count);

            // Log the configuration for inspection
            std::cout << config.toString() << std::endl;
        }

    } // namespace test
} // namespace llaminar2
