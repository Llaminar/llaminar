/**
 * @file Test__CollectiveContext_ROCm.cpp
 * @brief Integration tests for CollectiveContext with ROCm/RCCL backend
 *
 * Tests the ROCm-specific pipeline: CollectiveContext -> BackendRouter -> RCCL
 * Verifies backend selection and GPU collective execution on AMD GPUs.
 *
 * Split from Test__CollectiveContext_GPU.cpp to isolate ROCm tests into their
 * own binary. This is critical because ROCm CLR has a known bug where repeated
 * RCCL ncclCommDestroy calls corrupt internal memory object tracking, causing
 * "Memobj map does not have ptr: 0x0" crashes. By isolating ROCm tests, we
 * minimize the number of RCCL init/destroy cycles within a single process.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "execution/local_execution/collective/CollectiveContext.h"
#include "collective/BackendRouter.h"
#include "collective/ICollectiveBackend.h"
#include "tensors/TensorClasses.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "backends/DeviceId.h"
#include "backends/BackendManager.h"
#include "utils/Logger.h"

#include <iostream>
#include <cmath>
#include <numeric>

#ifdef HAVE_ROCM

namespace llaminar2
{

    // =========================================================================
    // Test Fixture
    //
    // CRITICAL: ROCm CLR has a known bug where repeated RCCL ncclCommDestroy
    // calls corrupt internal memory object tracking. To avoid this, we cache
    // CollectiveContext instances in static storage so RCCL communicators are
    // only created once per unique inventory configuration and destroyed at
    // process exit (not between tests).
    // =========================================================================

    class CollectiveContextROCmTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            auto *rocm_backend = getROCmBackend();
            if (rocm_backend == nullptr)
            {
                GTEST_SKIP() << "ROCm backend not available";
            }

            rocm_count_ = rocm_backend->deviceCount();
            if (rocm_count_ == 0)
            {
                GTEST_SKIP() << "No ROCm GPUs available";
            }

            std::cout << "CollectiveContext ROCm Test: Found "
                      << rocm_count_ << " ROCm GPU(s)" << std::endl;
            for (int i = 0; i < rocm_count_; ++i)
            {
                std::cout << "  ROCm GPU " << i << ": " << rocm_backend->deviceName(i)
                          << std::endl;
            }
        }

        void TearDown() override
        {
            auto *rocm_backend = getROCmBackend();
            if (rocm_backend != nullptr)
            {
                for (int i = 0; i < rocm_count_; ++i)
                {
                    rocm_backend->synchronize(i);
                }
            }
        }

        /**
         * @brief Get a cached CollectiveContext for the full ROCm inventory.
         *
         * Reuses the same context (and therefore the same RCCL communicators)
         * across all tests to avoid repeated init/destroy cycles.
         */
        CollectiveContext *getROCmContext()
        {
            if (!rocm_ctx_)
            {
                auto inventory = buildROCmInventory();
                rocm_ctx_ = CollectiveContextFactory::createIntraNode(inventory, nullptr);
            }
            return rocm_ctx_.get();
        }

        /**
         * @brief Get a cached CollectiveContext for a single-ROCm-GPU inventory.
         */
        CollectiveContext *getSingleROCmContext()
        {
            if (!single_rocm_ctx_)
            {
                auto inventory = buildSingleROCmInventory();
                single_rocm_ctx_ = CollectiveContextFactory::createIntraNode(inventory, nullptr);
            }
            return single_rocm_ctx_.get();
        }

        int rocm_count_ = 0;

    private:
        ClusterInventory buildROCmInventory()
        {
            ClusterInventory inv;
            RankInventory rank_inv;
            rank_inv.rank = 0;
            rank_inv.node_id = 0;
            rank_inv.local_rank = 0;
            rank_inv.hostname = "localhost";

            auto *rocm_backend = getROCmBackend();
            if (rocm_backend != nullptr)
            {
                for (int i = 0; i < rocm_count_; ++i)
                {
                    DeviceInfo gpu;
                    gpu.type = DeviceType::ROCm;
                    gpu.local_device_id = i;
                    gpu.memory_bytes = rocm_backend->deviceMemoryTotal(i);
                    gpu.name = rocm_backend->deviceName(i);
                    rank_inv.gpus.push_back(gpu);
                }
            }

            inv.ranks.push_back(rank_inv);
            inv.world_size = 1;
            inv.buildNodeAggregations();
            return inv;
        }

        ClusterInventory buildSingleROCmInventory()
        {
            ClusterInventory inv;
            RankInventory rank_inv;
            rank_inv.rank = 0;
            rank_inv.node_id = 0;
            rank_inv.local_rank = 0;
            rank_inv.hostname = "localhost";

            auto *rocm_backend = getROCmBackend();
            if (rocm_backend != nullptr && rocm_count_ > 0)
            {
                DeviceInfo gpu;
                gpu.type = DeviceType::ROCm;
                gpu.local_device_id = 0;
                gpu.memory_bytes = rocm_backend->deviceMemoryTotal(0);
                gpu.name = rocm_backend->deviceName(0);
                rank_inv.gpus.push_back(gpu);
            }

            inv.ranks.push_back(rank_inv);
            inv.world_size = 1;
            inv.buildNodeAggregations();
            return inv;
        }

        // Static context caching - RCCL communicators live for the process lifetime
        static inline std::unique_ptr<CollectiveContext> rocm_ctx_;
        static inline std::unique_ptr<CollectiveContext> single_rocm_ctx_;
    };

    // =========================================================================
    // Backend Selection
    // =========================================================================

    TEST_F(CollectiveContextROCmTest, BackendSelection)
    {
        auto *ctx = getROCmContext();
        ASSERT_NE(ctx, nullptr);

#ifdef HAVE_RCCL
        EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::RCCL))
            << "RCCL backend should be available with ROCm GPUs present";
#else
        EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::HOST))
            << "HOST backend should be available as fallback";
#endif
    }

    // =========================================================================
    // AllReduce
    // =========================================================================

    TEST_F(CollectiveContextROCmTest, AllReduce)
    {
        auto *ctx = getSingleROCmContext();
        ASSERT_NE(ctx, nullptr);

        constexpr size_t TENSOR_SIZE = 64;
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());

        float *data = tensor->mutable_data();
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            data[i] = static_cast<float>(i + 1);
        }

        DeviceId rocm_device = DeviceId::rocm(0);
        bool result = ctx->executeAllreduce(tensor.get(), TENSOR_SIZE, rocm_device);
        EXPECT_TRUE(result) << "AllReduce on ROCm device routing failed";

        const float *result_data = tensor->data();
        ASSERT_NE(result_data, nullptr);
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            EXPECT_FALSE(std::isnan(result_data[i])) << "NaN at index " << i;
            EXPECT_FALSE(std::isinf(result_data[i])) << "Inf at index " << i;
        }
    }

    // =========================================================================
    // AllGather
    // =========================================================================

    TEST_F(CollectiveContextROCmTest, AllGather)
    {
        auto *ctx = getROCmContext();
        ASSERT_NE(ctx, nullptr);

        constexpr size_t TENSOR_SIZE = 8;
        const size_t num_devices = ctx->localDevices().size();

        auto local_input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());
        auto full_output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{TENSOR_SIZE * num_devices}, DeviceId::cpu());

        float *input_data = local_input->mutable_data();
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            input_data[i] = static_cast<float>(i + 1);
        }

        float *output_data = full_output->mutable_data();
        std::fill(output_data, output_data + TENSOR_SIZE * num_devices, 0.0f);

        DeviceId rocm_device = DeviceId::rocm(0);
        bool result = ctx->executeAllgather(
            local_input.get(), full_output.get(), TENSOR_SIZE, rocm_device);
        EXPECT_TRUE(result) << "AllGather on ROCm device routing failed";

        const float *result_data = full_output->data();
        for (size_t i = 0; i < TENSOR_SIZE * num_devices; ++i)
        {
            EXPECT_FALSE(std::isnan(result_data[i])) << "NaN at index " << i;
            EXPECT_FALSE(std::isinf(result_data[i])) << "Inf at index " << i;
        }
    }

    // =========================================================================
    // Broadcast
    // =========================================================================

    TEST_F(CollectiveContextROCmTest, Broadcast)
    {
        auto *ctx = getROCmContext();
        ASSERT_NE(ctx, nullptr);

        constexpr size_t TENSOR_SIZE = 64;
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());

        float *data = tensor->mutable_data();
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            data[i] = static_cast<float>(i + 1);
        }

        DeviceId rocm_device = DeviceId::rocm(0);
        bool result = ctx->executeBroadcast(tensor.get(), TENSOR_SIZE, 0, rocm_device);
        EXPECT_TRUE(result) << "Broadcast on ROCm device routing failed";

        const float *result_data = tensor->data();
        ASSERT_NE(result_data, nullptr);
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            EXPECT_FLOAT_EQ(result_data[i], static_cast<float>(i + 1))
                << "Data mismatch at index " << i;
        }
    }

} // namespace llaminar2

#endif // HAVE_ROCM
