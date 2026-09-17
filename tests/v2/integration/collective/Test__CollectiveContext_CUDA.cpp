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
#include "../../utils/ObservedCollectiveInventory.h"

#include <iostream>
#include <cmath>
#include <numeric>

#ifdef HAVE_CUDA

namespace llaminar2
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    /** @brief Real-device collective fixture backed by canonical hardware observation. */
    class CollectiveContextCUDATest : public ::testing::Test
    {
    protected:
        /** @brief Require the suite's real backend before constructing participants. */
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

        /** @brief Join the fixture's device work before releasing test resources. */
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

        /** @return Selected real participants, with canonical UUID/P2P evidence. */
        ClusterInventory buildCUDAInventory()
        {
            return test::observedLocalCollectiveInventory(cuda_count_, 0);
        }

        /** @return Selected real participants, with canonical UUID/P2P evidence. */
        ClusterInventory buildSingleCUDAInventory()
        {
            return test::observedLocalCollectiveInventory(1, 0);
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

    TEST_F(CollectiveContextCUDATest, RejectsHostAllReduceRoutedAsCUDA)
    {
        auto inventory = buildSingleCUDAInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(inventory, nullptr);
        ASSERT_NE(ctx, nullptr);

        constexpr size_t TENSOR_SIZE = 64;
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());

        DeviceId cuda_device = DeviceId::cuda(0);
        EXPECT_THROW(
            ctx->executeAllreduce(
                tensor.get(), TENSOR_SIZE, cuda_device),
            std::invalid_argument);
    }

    // =========================================================================
    // AllGather
    // =========================================================================

    TEST_F(CollectiveContextCUDATest, RejectsHostAllGatherRoutedAsCUDA)
    {
        auto inventory = buildCUDAInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(inventory, nullptr);
        ASSERT_NE(ctx, nullptr);

        constexpr size_t TENSOR_SIZE = 8;
        auto local_input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());
        auto full_output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{TENSOR_SIZE * 2}, DeviceId::cpu());

        DeviceId cuda_device = DeviceId::cuda(0);
        EXPECT_THROW(
            ctx->executeAllgather(
                local_input.get(),
                full_output.get(),
                TENSOR_SIZE,
                cuda_device),
            std::invalid_argument);
    }

    // =========================================================================
    // Broadcast
    // =========================================================================

    TEST_F(CollectiveContextCUDATest, RejectsHostBroadcastRoutedAsCUDA)
    {
        auto inventory = buildCUDAInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(inventory, nullptr);
        ASSERT_NE(ctx, nullptr);

        constexpr size_t TENSOR_SIZE = 64;
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());

        DeviceId cuda_device = DeviceId::cuda(0);
        EXPECT_THROW(
            ctx->executeBroadcast(
                tensor.get(), TENSOR_SIZE, 0, cuda_device),
            std::invalid_argument);
    }

} // namespace llaminar2

#endif // HAVE_CUDA
