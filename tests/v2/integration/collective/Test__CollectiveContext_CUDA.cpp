/**
 * @file Test__CollectiveContext_CUDA.cpp
 * @brief Integration tests for CollectiveContext with CUDA/NCCL backend
 *
 * Tests the CUDA-specific pipeline: CollectiveContext -> BackendRouter -> NCCL
 * Verifies backend selection and GPU collective execution on NVIDIA GPUs.
 *
 * Split from Test__CollectiveContext_GPU.cpp to isolate CUDA tests from ROCm
 * tests, avoiding ROCm CLR state corruption from repeated RCCL init/destroy
 * cycles within a single test binary.
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

#ifdef HAVE_CUDA

namespace llaminar2
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class CollectiveContextCUDATest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            auto *cuda_backend = getCUDABackend();
            if (cuda_backend == nullptr)
            {
                GTEST_SKIP() << "CUDA backend not available";
            }

            cuda_count_ = cuda_backend->deviceCount();
            if (cuda_count_ == 0)
            {
                GTEST_SKIP() << "No CUDA GPUs available";
            }

            std::cout << "CollectiveContext CUDA Test: Found "
                      << cuda_count_ << " CUDA GPU(s)" << std::endl;
            for (int i = 0; i < cuda_count_; ++i)
            {
                std::cout << "  CUDA GPU " << i << ": " << cuda_backend->deviceName(i)
                          << std::endl;
            }
        }

        void TearDown() override
        {
            auto *cuda_backend = getCUDABackend();
            if (cuda_backend != nullptr)
            {
                for (int i = 0; i < cuda_count_; ++i)
                {
                    cuda_backend->synchronize(i);
                }
            }
        }

        ClusterInventory buildCUDAInventory()
        {
            ClusterInventory inv;
            RankInventory rank_inv;
            rank_inv.rank = 0;
            rank_inv.node_id = 0;
            rank_inv.local_rank = 0;
            rank_inv.hostname = "localhost";

            auto *cuda_backend = getCUDABackend();
            if (cuda_backend != nullptr)
            {
                for (int i = 0; i < cuda_count_; ++i)
                {
                    DeviceInfo gpu;
                    gpu.type = DeviceType::CUDA;
                    gpu.local_device_id = i;
                    gpu.memory_bytes = cuda_backend->deviceMemoryTotal(i);
                    gpu.name = cuda_backend->deviceName(i);
                    gpu.supports_p2p = true;
                    rank_inv.gpus.push_back(gpu);
                }
            }

            inv.ranks.push_back(rank_inv);
            inv.world_size = 1;
            inv.buildNodeAggregations();
            return inv;
        }

        ClusterInventory buildSingleCUDAInventory()
        {
            ClusterInventory inv;
            RankInventory rank_inv;
            rank_inv.rank = 0;
            rank_inv.node_id = 0;
            rank_inv.local_rank = 0;
            rank_inv.hostname = "localhost";

            auto *cuda_backend = getCUDABackend();
            if (cuda_backend != nullptr && cuda_count_ > 0)
            {
                DeviceInfo gpu;
                gpu.type = DeviceType::CUDA;
                gpu.local_device_id = 0;
                gpu.memory_bytes = cuda_backend->deviceMemoryTotal(0);
                gpu.name = cuda_backend->deviceName(0);
                rank_inv.gpus.push_back(gpu);
            }

            inv.ranks.push_back(rank_inv);
            inv.world_size = 1;
            inv.buildNodeAggregations();
            return inv;
        }

        int cuda_count_ = 0;
    };

    // =========================================================================
    // Backend Selection
    // =========================================================================

    TEST_F(CollectiveContextCUDATest, BackendSelection)
    {
        auto inventory = buildCUDAInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(inventory, nullptr);
        ASSERT_NE(ctx, nullptr);

#ifdef HAVE_NCCL
        EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::NCCL))
            << "NCCL backend should be available with CUDA GPUs present";
#else
        EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::HOST))
            << "HOST backend should be available as fallback";
#endif
    }

    // =========================================================================
    // AllReduce
    // =========================================================================

    TEST_F(CollectiveContextCUDATest, AllReduce)
    {
        auto inventory = buildSingleCUDAInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(inventory, nullptr);
        ASSERT_NE(ctx, nullptr);

        constexpr size_t TENSOR_SIZE = 64;
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());

        float *data = tensor->mutable_data();
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            data[i] = static_cast<float>(i + 1);
        }

        DeviceId cuda_device = DeviceId::cuda(0);
        bool result = ctx->executeAllreduce(tensor.get(), TENSOR_SIZE, cuda_device);
        EXPECT_TRUE(result) << "AllReduce on CUDA device routing failed";

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

    TEST_F(CollectiveContextCUDATest, AllGather)
    {
        auto inventory = buildCUDAInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(inventory, nullptr);
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

        DeviceId cuda_device = DeviceId::cuda(0);
        bool result = ctx->executeAllgather(
            local_input.get(), full_output.get(), TENSOR_SIZE, cuda_device);
        EXPECT_TRUE(result) << "AllGather on CUDA device routing failed";

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

    TEST_F(CollectiveContextCUDATest, Broadcast)
    {
        auto inventory = buildCUDAInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(inventory, nullptr);
        ASSERT_NE(ctx, nullptr);

        constexpr size_t TENSOR_SIZE = 64;
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());

        float *data = tensor->mutable_data();
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            data[i] = static_cast<float>(i + 1);
        }

        DeviceId cuda_device = DeviceId::cuda(0);
        bool result = ctx->executeBroadcast(tensor.get(), TENSOR_SIZE, 0, cuda_device);
        EXPECT_TRUE(result) << "Broadcast on CUDA device routing failed";

        const float *result_data = tensor->data();
        ASSERT_NE(result_data, nullptr);
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            EXPECT_FLOAT_EQ(result_data[i], static_cast<float>(i + 1))
                << "Data mismatch at index " << i;
        }
    }

} // namespace llaminar2

#endif // HAVE_CUDA
