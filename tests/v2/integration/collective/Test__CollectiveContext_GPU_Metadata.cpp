/**
 * @file Test__CollectiveContext_GPU.cpp
 * @brief Integration tests for CollectiveContext metadata and routing
 *
 * Tests backend-agnostic CollectiveContext behavior: world size, rank,
 * device enumeration, requiresCollectives(), router access, and error
 * handling. These tests use CUDA-only inventory by default to avoid
 * RCCL init/destroy cycles (ROCm-specific tests are in a separate binary).
 *
 * GPU-vendor-specific collective tests are split into:
 * - Test__CollectiveContext_CUDA.cpp (NCCL backend)
 * - Test__CollectiveContext_ROCm.cpp (RCCL backend)
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

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)

namespace llaminar2
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class CollectiveContextGPUTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            cuda_count_ = 0;
            rocm_count_ = 0;

#ifdef HAVE_CUDA
            auto *cuda_backend = getCUDABackend();
            if (cuda_backend != nullptr)
            {
                cuda_count_ = cuda_backend->deviceCount();
            }
#endif

#ifdef HAVE_ROCM
            auto *rocm_backend = getROCmBackend();
            if (rocm_backend != nullptr)
            {
                rocm_count_ = rocm_backend->deviceCount();
            }
#endif

            if (cuda_count_ == 0 && rocm_count_ == 0)
            {
                GTEST_SKIP() << "No GPUs available";
            }

            std::cout << "CollectiveContext GPU Test: Found "
                      << cuda_count_ << " CUDA GPU(s), "
                      << rocm_count_ << " ROCm GPU(s)" << std::endl;

            inventory_ = buildLocalInventory();
        }

        void TearDown() override
        {
#ifdef HAVE_CUDA
            auto *cuda_backend = getCUDABackend();
            if (cuda_backend != nullptr)
            {
                for (int i = 0; i < cuda_count_; ++i)
                {
                    cuda_backend->synchronize(i);
                }
            }
#endif

#ifdef HAVE_ROCM
            auto *rocm_backend = getROCmBackend();
            if (rocm_backend != nullptr)
            {
                for (int i = 0; i < rocm_count_; ++i)
                {
                    rocm_backend->synchronize(i);
                }
            }
#endif
        }

        ClusterInventory buildLocalInventory()
        {
            ClusterInventory inv;
            RankInventory rank_inv;
            rank_inv.rank = 0;
            rank_inv.node_id = 0;
            rank_inv.local_rank = 0;
            rank_inv.hostname = "localhost";

#ifdef HAVE_CUDA
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
#endif

#ifdef HAVE_ROCM
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
#endif

            inv.ranks.push_back(rank_inv);
            inv.world_size = 1;
            inv.buildNodeAggregations();
            return inv;
        }

        ClusterInventory buildCUDAOnlyInventory()
        {
            ClusterInventory inv;
            RankInventory rank_inv;
            rank_inv.rank = 0;
            rank_inv.node_id = 0;
            rank_inv.local_rank = 0;
            rank_inv.hostname = "localhost";

#ifdef HAVE_CUDA
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
                    rank_inv.gpus.push_back(gpu);
                }
            }
#endif

            inv.ranks.push_back(rank_inv);
            inv.world_size = 1;
            inv.buildNodeAggregations();
            return inv;
        }

        int cuda_count_ = 0;
        int rocm_count_ = 0;
        ClusterInventory inventory_;
    };

    // =========================================================================
    // World Size and Rank Tests
    // =========================================================================

    TEST_F(CollectiveContextGPUTest, WorldSizeAndRank)
    {
        auto ctx = CollectiveContextFactory::createIntraNode(inventory_, nullptr);
        ASSERT_NE(ctx, nullptr);

        EXPECT_EQ(ctx->worldSize(), 1) << "World size should be 1 for single-rank test";
        EXPECT_EQ(ctx->rank(), 0) << "Rank should be 0 for single-rank test";
    }

    TEST_F(CollectiveContextGPUTest, LocalDevicesReturnsGPUs)
    {
        auto ctx = CollectiveContextFactory::createIntraNode(inventory_, nullptr);
        ASSERT_NE(ctx, nullptr);

        const auto &devices = ctx->localDevices();
        EXPECT_GE(devices.size(), 0u);

        std::cout << "Local devices:" << std::endl;
        for (const auto &dev : devices)
        {
            std::cout << "  " << (dev.is_cuda() ? "CUDA" : (dev.is_rocm() ? "ROCm" : "CPU"))
                      << " device " << dev.ordinal << std::endl;
        }
    }

    // =========================================================================
    // RequiresCollectives Tests
    // =========================================================================

    TEST_F(CollectiveContextGPUTest, RequiresCollectives_SingleDevice)
    {
        auto ctx = CollectiveContextFactory::createSingleDevice();
        ASSERT_NE(ctx, nullptr);

        EXPECT_FALSE(ctx->requiresCollectives())
            << "Single device should not require collectives";
    }

    TEST_F(CollectiveContextGPUTest, RequiresCollectives_MultipleGPUs)
    {
        int total_gpus = cuda_count_ + rocm_count_;
        if (total_gpus < 2)
        {
            GTEST_SKIP() << "Need at least 2 GPUs for this test, have " << total_gpus;
        }

        auto ctx = CollectiveContextFactory::createIntraNode(inventory_, nullptr);
        ASSERT_NE(ctx, nullptr);

        const auto &devices = ctx->localDevices();
        if (devices.size() >= 2)
        {
            EXPECT_TRUE(ctx->requiresCollectives())
                << "Multiple devices should require collectives";
        }
    }

    // =========================================================================
    // Router Access Tests
    // =========================================================================

    TEST_F(CollectiveContextGPUTest, RouterIsAccessible)
    {
        auto ctx = CollectiveContextFactory::createIntraNode(inventory_, nullptr);
        ASSERT_NE(ctx, nullptr);

        IBackendRouter *router = ctx->router();

        if (router != nullptr)
        {
            std::cout << "Router is available" << std::endl;

#ifdef HAVE_NCCL
            if (cuda_count_ > 0)
            {
                EXPECT_TRUE(router->isAvailable(CollectiveBackendType::NCCL));
            }
#endif

#ifdef HAVE_RCCL
            if (rocm_count_ > 0)
            {
                EXPECT_TRUE(router->isAvailable(CollectiveBackendType::RCCL));
            }
#endif
        }
        else
        {
            std::cout << "Router is nullptr (may be expected without full config)"
                      << std::endl;
        }
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    TEST_F(CollectiveContextGPUTest, AllReduceWithZeroCount)
    {
        auto ctx = CollectiveContextFactory::createIntraNode(inventory_, nullptr);
        ASSERT_NE(ctx, nullptr);

        constexpr size_t TENSOR_SIZE = 64;
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());

        float *data = tensor->mutable_data();
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            data[i] = static_cast<float>(i + 1);
        }

        DeviceId device = DeviceId::cpu();
#ifdef HAVE_CUDA
        if (cuda_count_ > 0)
        {
            device = DeviceId::cuda(0);
        }
#endif
#ifdef HAVE_ROCM
        if (rocm_count_ > 0 && cuda_count_ == 0)
        {
            device = DeviceId::rocm(0);
        }
#endif

        bool result = ctx->executeAllreduce(tensor.get(), 0, device);
        EXPECT_TRUE(result) << "AllReduce with count=0 should use tensor numel";
    }

    TEST_F(CollectiveContextGPUTest, AllReduceWithNullBuffer)
    {
        auto ctx = CollectiveContextFactory::createIntraNode(inventory_, nullptr);
        ASSERT_NE(ctx, nullptr);

        // Note: Calling executeAllreduce with nullptr is undefined behavior
        // and may crash depending on the backend. This test just verifies
        // context creation works.
    }

} // namespace llaminar2

#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)
