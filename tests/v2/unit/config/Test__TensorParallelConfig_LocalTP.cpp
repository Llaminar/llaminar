/**
 * @file Test__TensorParallelConfig_LocalTP.cpp
 * @brief Unit tests for TensorParallelConfig::fromLocalTPContext()
 *
 * Tests the factory method that creates TensorParallelConfig from ILocalTPContext,
 * enabling LOCAL TP weight sharding with proportional work distribution for
 * heterogeneous GPU configurations.
 */

#include <gtest/gtest.h>
#include "config/TensorParallelConfig.h"
#include "collective/ILocalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "tensors/ITensor.h"
#include <vector>
#include <cmath>
#include <atomic>
#include <memory>

namespace llaminar2
{
    namespace test
    {

        /**
         * @brief Lightweight mock for ILocalTPContext
         *
         * Implements the minimum interface needed for testing fromLocalTPContext().
         * Collective operations are no-ops since we're only testing configuration generation.
         */
        class MockLocalTPContext : public ILocalTPContext
        {
        public:
            MockLocalTPContext(
                std::vector<GlobalDeviceAddress> devices,
                std::vector<float> weights = {},
                CollectiveBackendType backend = CollectiveBackendType::HOST)
                : devices_(std::move(devices)), weights_(std::move(weights)), backend_(backend)
            {
                // Normalize weights if provided, else create equal weights
                if (weights_.empty())
                {
                    weights_.resize(devices_.size(), 1.0f / static_cast<float>(devices_.size()));
                }
                else
                {
                    // Normalize to sum to 1.0
                    float sum = 0.0f;
                    for (float w : weights_)
                        sum += w;
                    if (sum > 0.0f)
                    {
                        for (float &w : weights_)
                            w /= sum;
                    }
                }
            }

            // =====================================================================
            // ILocalTPContext Implementation - Configuration
            // =====================================================================

            const std::vector<GlobalDeviceAddress> &devices() const override { return devices_; }
            const std::vector<float> &weights() const override { return weights_; }
            CollectiveBackendType backend() const override { return backend_; }
            int degree() const override { return static_cast<int>(devices_.size()); }
            int myIndex() const override { return 0; }

            // =====================================================================
            // ILocalTPContext Implementation - Collective Operations (no-ops)
            // =====================================================================

            bool allreduce(TensorBase * /*tensor*/) override { return true; }
            bool allreduce(TensorBase *tensor, const std::string & /*stage_name*/, size_t /*count*/ = 0) override { return allreduce(tensor); }
            bool allreduce(const TensorBase * /*input*/, TensorBase * /*output*/) override { return true; }
            bool allgather(const TensorBase * /*local_shard*/, TensorBase * /*global_tensor*/) override { return true; }
            bool gatherFromDevices(const std::vector<const TensorBase *> & /*shards*/, TensorBase * /*output*/) override { return true; }
            bool reduceScatter(const TensorBase * /*input*/, TensorBase * /*output_shard*/) override { return true; }

            // =====================================================================
            // ILocalTPContext Implementation - Synchronization (no-op)
            // =====================================================================

            void synchronize() override {}

            // =====================================================================
            // ILocalTPContext Implementation - Device Management
            // =====================================================================

            int indexForDevice(const GlobalDeviceAddress &device) const override
            {
                for (size_t i = 0; i < devices_.size(); ++i)
                {
                    if (devices_[i] == device)
                        return static_cast<int>(i);
                }
                return -1;
            }

            const GlobalDeviceAddress &deviceAt(int index) const override
            {
                if (index < 0 || index >= static_cast<int>(devices_.size()))
                    throw std::out_of_range("MockLocalTPContext::deviceAt: index out of range");
                return devices_[index];
            }

            float weightForDevice(const GlobalDeviceAddress &device) const override
            {
                int idx = indexForDevice(device);
                return (idx >= 0) ? weights_[idx] : 0.0f;
            }

            // =====================================================================
            // ILocalTPContext Implementation - Sharding Utilities
            // =====================================================================

            int headsForDevice(const GlobalDeviceAddress &device, int total_heads) const override
            {
                float w = weightForDevice(device);
                return static_cast<int>(w * total_heads + 0.5f);
            }

            std::pair<int, int> rowRangeForDevice(
                const GlobalDeviceAddress &device, int total_rows) const override
            {
                int idx = indexForDevice(device);
                if (idx < 0)
                    return {0, 0};

                float cumulative = 0.0f;
                for (int i = 0; i < idx; ++i)
                    cumulative += weights_[i];

                int start = static_cast<int>(cumulative * total_rows);
                int end = static_cast<int>((cumulative + weights_[idx]) * total_rows);
                return {start, end};
            }

            std::pair<int, int> colRangeForDevice(
                const GlobalDeviceAddress &device, int total_cols) const override
            {
                return rowRangeForDevice(device, total_cols);
            }

            // =====================================================================
            // ILocalTPContext Implementation - BAR Registry (no-ops for tests)
            // =====================================================================

            void registerBARBackedOutput(
                const std::string & /*stage_name*/,
                const GlobalDeviceAddress & /*device*/,
                TensorBase * /*tensor*/) override
            {
                // No-op for unit tests
            }

            bool hasBARBackedOutputs(const std::string & /*stage_name*/) const override { return false; }
            void clearBARBackedOutputs() override {}
            bool reserveTempBufferBytes(size_t /*bytes*/) override { return true; }

            // =====================================================================
            // ILocalTPContext Implementation - Broadcast (no-op)
            // =====================================================================
            bool broadcast(TensorBase * /*tensor*/, int /*source_device_index*/ = 0) override { return true; }

            void requestAbort() override {}
            bool isAbortRequested() const override { return false; }

        private:
            std::vector<GlobalDeviceAddress> devices_;
            std::vector<float> weights_;
            CollectiveBackendType backend_;
        };

        // =============================================================================
        // Test Fixture
        // =============================================================================

        class Test__TensorParallelConfig_LocalTP : public ::testing::Test
        {
        protected:
            // Qwen2.5-0.5B-style parameters (commonly used test model)
            static constexpr int QWEN_05B_HEADS = 14;
            static constexpr int QWEN_05B_KV_HEADS = 2;
            static constexpr int QWEN_05B_D_FF = 4864;
            static constexpr int QWEN_05B_VOCAB = 151936;

            // Smaller test parameters for basic validation
            static constexpr int TEST_HEADS = 14;
            static constexpr int TEST_KV_HEADS = 2;
            static constexpr int TEST_D_FF = 1024;
            static constexpr int TEST_VOCAB = 32000;
        };

        // =============================================================================
        // fromLocalTPContext - Equal Split Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig_LocalTP, FromLocalTPContext_EqualSplit_TwoDevices)
        {
            // Create mock with 2 devices, no weights (equal split)
            std::vector<GlobalDeviceAddress> devices = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::cuda(1)};
            MockLocalTPContext ctx(devices);

            auto config = TensorParallelConfig::fromLocalTPContext(
                ctx, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_EQ(config.worldSize(), 2);
            EXPECT_TRUE(config.validate()) << config.validationError();
            EXPECT_EQ(config.totalHeads(), TEST_HEADS);
            EXPECT_EQ(config.totalKVHeads(), TEST_KV_HEADS);
            EXPECT_EQ(config.totalDFF(), TEST_D_FF);
            EXPECT_EQ(config.totalVocab(), TEST_VOCAB);

            // Verify device 0 assignment
            const auto &dev0 = config.forRank(0);
            EXPECT_EQ(dev0.head_start, 0);
            EXPECT_EQ(dev0.head_count, 7);
            EXPECT_EQ(dev0.kv_head_start, 0);
            EXPECT_EQ(dev0.kv_head_count, 1);
            EXPECT_TRUE(dev0.device.is_cuda());
            EXPECT_EQ(dev0.device.ordinal, 0);

            // Verify device 1 assignment
            const auto &dev1 = config.forRank(1);
            EXPECT_EQ(dev1.head_start, 7);
            EXPECT_EQ(dev1.head_count, 7);
            EXPECT_EQ(dev1.kv_head_start, 1);
            EXPECT_EQ(dev1.kv_head_count, 1);
            EXPECT_TRUE(dev1.device.is_cuda());
            EXPECT_EQ(dev1.device.ordinal, 1);

            // Verify vocab split (should be roughly equal)
            EXPECT_EQ(dev0.vocab_count + dev1.vocab_count, TEST_VOCAB);
        }

        TEST_F(Test__TensorParallelConfig_LocalTP, FromLocalTPContext_EqualSplit_HeterogeneousDevices)
        {
            // Mixed CUDA + ROCm configuration
            std::vector<GlobalDeviceAddress> devices = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::rocm(0)};
            MockLocalTPContext ctx(devices);

            auto config = TensorParallelConfig::fromLocalTPContext(
                ctx, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_EQ(config.worldSize(), 2);
            EXPECT_TRUE(config.validate()) << config.validationError();

            // Verify device types preserved
            const auto &cuda = config.forDevice(DeviceId::cuda(0));
            const auto &rocm = config.forDevice(DeviceId::rocm(0));

            EXPECT_TRUE(cuda.device.is_cuda());
            EXPECT_TRUE(rocm.device.is_rocm());
            EXPECT_EQ(cuda.head_count + rocm.head_count, TEST_HEADS);
        }

        // =============================================================================
        // fromLocalTPContext - Proportional Split Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig_LocalTP, FromLocalTPContext_ProportionalSplit_73_27)
        {
            // Typical heterogeneous config: NVIDIA 73%, AMD 27%
            std::vector<GlobalDeviceAddress> devices = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::rocm(0)};
            std::vector<float> weights = {0.73f, 0.27f};
            MockLocalTPContext ctx(devices, weights);

            auto config = TensorParallelConfig::fromLocalTPContext(
                ctx, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_EQ(config.worldSize(), 2);
            EXPECT_TRUE(config.isProportional());
            EXPECT_TRUE(config.validate()) << config.validationError();

            const auto &cuda = config.forDevice(DeviceId::cuda(0));
            const auto &rocm = config.forDevice(DeviceId::rocm(0));

            // CUDA should get majority (~73% of 14 heads ≈ 10)
            EXPECT_GT(cuda.head_count, rocm.head_count);
            EXPECT_EQ(cuda.head_count + rocm.head_count, TEST_HEADS);

            // Work fractions should be approximately preserved
            EXPECT_NEAR(cuda.work_fraction, 0.73f, 0.05f);
            EXPECT_NEAR(rocm.work_fraction, 0.27f, 0.05f);

            // Vocab should also be split proportionally
            float cuda_vocab_fraction = static_cast<float>(cuda.vocab_count) / TEST_VOCAB;
            EXPECT_NEAR(cuda_vocab_fraction, 0.73f, 0.05f);
        }

        TEST_F(Test__TensorParallelConfig_LocalTP, FromLocalTPContext_ProportionalSplit_60_40)
        {
            // 60/40 split
            std::vector<GlobalDeviceAddress> devices = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::cuda(1)};
            std::vector<float> weights = {0.6f, 0.4f};
            MockLocalTPContext ctx(devices, weights);

            auto config = TensorParallelConfig::fromLocalTPContext(
                ctx, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_TRUE(config.validate()) << config.validationError();
            EXPECT_TRUE(config.isProportional());

            const auto &dev0 = config.forRank(0);
            const auto &dev1 = config.forRank(1);

            // 60% of 14 = 8.4, 40% of 14 = 5.6
            // After rounding: expect 8 or 9 + 6 or 5 = 14
            EXPECT_EQ(dev0.head_count + dev1.head_count, TEST_HEADS);
            EXPECT_GE(dev0.head_count, dev1.head_count);
        }

        // =============================================================================
        // fromLocalTPContext - Three Device Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig_LocalTP, FromLocalTPContext_ThreeDevices_EqualSplit)
        {
            // 3-way equal split to test remainder handling
            std::vector<GlobalDeviceAddress> devices = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::cuda(1),
                GlobalDeviceAddress::rocm(0)};
            MockLocalTPContext ctx(devices);

            auto config = TensorParallelConfig::fromLocalTPContext(
                ctx, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_EQ(config.worldSize(), 3);
            EXPECT_TRUE(config.validate()) << config.validationError();

            // 14 heads / 3 with GQA alignment (n_kv_heads=2) → multiples of 2
            // Expected: [6, 4, 4] (first device gets extra aligned chunk)
            int total_heads = 0;
            for (int i = 0; i < 3; ++i)
            {
                const auto &a = config.forRank(i);
                total_heads += a.head_count;
                EXPECT_EQ(a.head_count % TEST_KV_HEADS, 0)
                    << "Rank " << i << " head_count=" << a.head_count
                    << " must be divisible by n_kv_heads=" << TEST_KV_HEADS;
            }
            EXPECT_EQ(total_heads, TEST_HEADS);
        }

        TEST_F(Test__TensorParallelConfig_LocalTP, FromLocalTPContext_ThreeDevices_ProportionalSplit)
        {
            // 50% / 30% / 20% split
            std::vector<GlobalDeviceAddress> devices = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::cuda(1),
                GlobalDeviceAddress::rocm(0)};
            std::vector<float> weights = {0.5f, 0.3f, 0.2f};
            MockLocalTPContext ctx(devices, weights);

            auto config = TensorParallelConfig::fromLocalTPContext(
                ctx, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_EQ(config.worldSize(), 3);
            EXPECT_TRUE(config.isProportional());
            EXPECT_TRUE(config.validate()) << config.validationError();

            const auto &dev0 = config.forRank(0);
            const auto &dev1 = config.forRank(1);
            const auto &dev2 = config.forRank(2);

            // Verify ordering: dev0 > dev1 > dev2
            EXPECT_GE(dev0.head_count, dev1.head_count);
            EXPECT_GE(dev1.head_count, dev2.head_count);
            EXPECT_EQ(dev0.head_count + dev1.head_count + dev2.head_count, TEST_HEADS);
        }

        // =============================================================================
        // forDevice Lookup Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig_LocalTP, ForDevice_LookupWorks)
        {
            std::vector<GlobalDeviceAddress> devices = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::rocm(0)};
            MockLocalTPContext ctx(devices);

            auto config = TensorParallelConfig::fromLocalTPContext(
                ctx, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            // Lookup by DeviceId should work
            const auto &cuda_assignment = config.forDevice(DeviceId::cuda(0));
            EXPECT_TRUE(cuda_assignment.device.is_cuda());
            EXPECT_EQ(cuda_assignment.local_rank, 0);

            const auto &rocm_assignment = config.forDevice(DeviceId::rocm(0));
            EXPECT_TRUE(rocm_assignment.device.is_rocm());
            EXPECT_EQ(rocm_assignment.local_rank, 1);

            // Non-existent device should throw
            EXPECT_THROW(config.forDevice(DeviceId::cuda(1)), std::out_of_range);
        }

        TEST_F(Test__TensorParallelConfig_LocalTP, ForRank_MatchesDeviceOrder)
        {
            std::vector<GlobalDeviceAddress> devices = {
                GlobalDeviceAddress::rocm(0), // Note: ROCm first
                GlobalDeviceAddress::cuda(0)};
            MockLocalTPContext ctx(devices);

            auto config = TensorParallelConfig::fromLocalTPContext(
                ctx, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            // Rank 0 should be ROCm (first in device list)
            const auto &rank0 = config.forRank(0);
            EXPECT_TRUE(rank0.device.is_rocm());

            // Rank 1 should be CUDA (second in device list)
            const auto &rank1 = config.forRank(1);
            EXPECT_TRUE(rank1.device.is_cuda());
        }

        // =============================================================================
        // D_FF Alignment Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig_LocalTP, DFFAlignment_32Aligned)
        {
            std::vector<GlobalDeviceAddress> devices = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::rocm(0)};
            std::vector<float> weights = {0.73f, 0.27f};
            MockLocalTPContext ctx(devices, weights);

            auto config = TensorParallelConfig::fromLocalTPContext(
                ctx, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            const auto &cuda = config.forDevice(DeviceId::cuda(0));
            const auto &rocm = config.forDevice(DeviceId::rocm(0));

            // Both d_ff counts should be divisible by 32 (quantization block boundary)
            EXPECT_EQ(cuda.d_ff_count % 32, 0)
                << "CUDA d_ff_count=" << cuda.d_ff_count << " not 32-aligned";
            EXPECT_EQ(rocm.d_ff_count % 32, 0)
                << "ROCm d_ff_count=" << rocm.d_ff_count << " not 32-aligned";

            // Total should still equal original
            EXPECT_EQ(cuda.d_ff_count + rocm.d_ff_count, TEST_D_FF);
        }

        // =============================================================================
        // Real-World Configuration Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig_LocalTP, Qwen2_5_0_5B_HeterogeneousConfig)
        {
            // RTX 3090 (CUDA) + MI50 (ROCm) with performance-weighted split
            std::vector<GlobalDeviceAddress> devices = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::rocm(0)};
            std::vector<float> weights = {0.73f, 0.27f};
            MockLocalTPContext ctx(devices, weights);

            auto config = TensorParallelConfig::fromLocalTPContext(
                ctx, QWEN_05B_HEADS, QWEN_05B_KV_HEADS, QWEN_05B_D_FF, QWEN_05B_VOCAB);

            EXPECT_TRUE(config.validate()) << config.validationError();
            EXPECT_TRUE(config.isProportional());

            // Verify totals
            EXPECT_EQ(config.totalHeads(), QWEN_05B_HEADS);
            EXPECT_EQ(config.totalKVHeads(), QWEN_05B_KV_HEADS);
            EXPECT_EQ(config.totalDFF(), QWEN_05B_D_FF);
            EXPECT_EQ(config.totalVocab(), QWEN_05B_VOCAB);

            const auto &cuda = config.forDevice(DeviceId::cuda(0));
            const auto &rocm = config.forDevice(DeviceId::rocm(0));

            // CUDA should have majority
            EXPECT_GT(cuda.head_count, rocm.head_count);
            EXPECT_GT(cuda.d_ff_count, rocm.d_ff_count);
            EXPECT_GT(cuda.vocab_count, rocm.vocab_count);

            // Ranges should be contiguous
            EXPECT_EQ(cuda.head_start, 0);
            EXPECT_EQ(rocm.head_start, cuda.headEnd());

            // Log configuration for inspection
            std::cout << "Qwen2.5-0.5B Heterogeneous Config:\n"
                      << config.toString() << std::endl;
        }

        // =============================================================================
        // Edge Case and Error Tests
        // =============================================================================

        TEST_F(Test__TensorParallelConfig_LocalTP, SingleDevice_DegreesTo1)
        {
            std::vector<GlobalDeviceAddress> devices = {GlobalDeviceAddress::cuda(0)};
            MockLocalTPContext ctx(devices);

            auto config = TensorParallelConfig::fromLocalTPContext(
                ctx, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_EQ(config.worldSize(), 1);
            EXPECT_TRUE(config.validate());

            const auto &assignment = config.forRank(0);
            EXPECT_EQ(assignment.head_count, TEST_HEADS);
            EXPECT_EQ(assignment.kv_head_count, TEST_KV_HEADS);
            EXPECT_EQ(assignment.d_ff_count, TEST_D_FF);
            EXPECT_EQ(assignment.vocab_count, TEST_VOCAB);
            EXPECT_NEAR(assignment.work_fraction, 1.0f, 0.001f);
        }

        TEST_F(Test__TensorParallelConfig_LocalTP, WeightsNormalized)
        {
            // Weights that don't sum to 1.0 should be normalized
            std::vector<GlobalDeviceAddress> devices = {
                GlobalDeviceAddress::cuda(0),
                GlobalDeviceAddress::rocm(0)};
            std::vector<float> weights = {73.0f, 27.0f}; // Sum = 100

            MockLocalTPContext ctx(devices, weights);

            // Context should normalize weights
            EXPECT_NEAR(ctx.weights()[0], 0.73f, 0.001f);
            EXPECT_NEAR(ctx.weights()[1], 0.27f, 0.001f);

            auto config = TensorParallelConfig::fromLocalTPContext(
                ctx, TEST_HEADS, TEST_KV_HEADS, TEST_D_FF, TEST_VOCAB);

            EXPECT_TRUE(config.validate()) << config.validationError();

            const auto &cuda = config.forDevice(DeviceId::cuda(0));
            EXPECT_NEAR(cuda.work_fraction, 0.73f, 0.01f);
        }

    } // namespace test
} // namespace llaminar2
