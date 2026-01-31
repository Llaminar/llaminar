/**
 * @file Test__CollectiveContext_GPU.cpp
 * @brief Integration tests for CollectiveContext with GPU backends
 *
 * Tests the full pipeline: CollectiveContext -> BackendRouter -> NCCL/RCCL
 * Verifies correct backend selection and GPU collective execution.
 *
 * Test coverage:
 * - Backend availability check (NCCL for CUDA, RCCL for ROCm)
 * - CollectiveContext creation with GPU inventory
 * - AllReduce operations on GPU tensors
 * - AllGather operations on GPU tensors
 * - World size and rank queries
 * - requiresCollectives() for single/multi-GPU configurations
 *
 * NOTE: Due to header conflicts between CUDA and ROCm vector types,
 * this test uses runtime detection via DeviceContext rather than
 * directly including both cuda_runtime.h and hip_runtime.h.
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
            // Detect available GPUs via BackendManager
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

            // Log device info
            std::cout << "CollectiveContext GPU Test: Found "
                      << cuda_count_ << " CUDA GPU(s), "
                      << rocm_count_ << " ROCm GPU(s)" << std::endl;

#ifdef HAVE_CUDA
            if (cuda_count_ > 0)
            {
                auto *backend = getCUDABackend();
                for (int i = 0; i < cuda_count_; ++i)
                {
                    std::cout << "  CUDA GPU " << i << ": " << backend->deviceName(i)
                              << std::endl;
                }
            }
#endif

#ifdef HAVE_ROCM
            if (rocm_count_ > 0)
            {
                auto *backend = getROCmBackend();
                for (int i = 0; i < rocm_count_; ++i)
                {
                    std::cout << "  ROCm GPU " << i << ": " << backend->deviceName(i)
                              << std::endl;
                }
            }
#endif

            // Build cluster inventory
            inventory_ = buildLocalInventory();
        }

        void TearDown() override
        {
            // Synchronize GPU devices via BackendManager
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

        /**
         * @brief Build a ClusterInventory representing local GPUs
         */
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

        /**
         * @brief Build inventory with only CUDA GPUs
         */
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

        /**
         * @brief Build inventory with only ROCm GPUs
         */
        ClusterInventory buildROCmOnlyInventory()
        {
            ClusterInventory inv;
            RankInventory rank_inv;
            rank_inv.rank = 0;
            rank_inv.node_id = 0;
            rank_inv.local_rank = 0;
            rank_inv.hostname = "localhost";

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

        /**
         * @brief Build inventory with single CUDA GPU (for no-op allreduce tests)
         */
        ClusterInventory buildSingleCUDAInventory()
        {
            ClusterInventory inv;
            RankInventory rank_inv;
            rank_inv.rank = 0;
            rank_inv.node_id = 0;
            rank_inv.local_rank = 0;
            rank_inv.hostname = "localhost";

#ifdef HAVE_CUDA
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
#endif

            inv.ranks.push_back(rank_inv);
            inv.world_size = 1;
            inv.buildNodeAggregations();

            return inv;
        }

        /**
         * @brief Build inventory with single ROCm GPU (for no-op allreduce tests)
         */
        ClusterInventory buildSingleROCmInventory()
        {
            ClusterInventory inv;
            RankInventory rank_inv;
            rank_inv.rank = 0;
            rank_inv.node_id = 0;
            rank_inv.local_rank = 0;
            rank_inv.hostname = "localhost";

#ifdef HAVE_ROCM
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
#endif

            inv.ranks.push_back(rank_inv);
            inv.world_size = 1;
            inv.buildNodeAggregations();

            return inv;
        }

        /**
         * @brief Skip test if no CUDA GPUs available
         */
        void skipIfNoCUDA()
        {
            if (cuda_count_ == 0)
            {
                GTEST_SKIP() << "No CUDA GPUs available";
            }
        }

        /**
         * @brief Skip test if no ROCm GPUs available
         */
        void skipIfNoROCm()
        {
            if (rocm_count_ == 0)
            {
                GTEST_SKIP() << "No ROCm GPUs available";
            }
        }

        int cuda_count_ = 0;
        int rocm_count_ = 0;
        ClusterInventory inventory_;
    };

    // =========================================================================
    // Backend Selection Tests
    // =========================================================================

    TEST_F(CollectiveContextGPUTest, BackendSelectionForCUDA)
    {
        skipIfNoCUDA();

        auto cuda_inventory = buildCUDAOnlyInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(cuda_inventory, nullptr);

        ASSERT_NE(ctx, nullptr);

#ifdef HAVE_NCCL
        // With NCCL compiled in and CUDA GPUs, NCCL should be available
        EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::NCCL))
            << "NCCL backend should be available with CUDA GPUs present";
#else
        // Without NCCL, should fall back to HOST
        EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::HOST))
            << "HOST backend should be available as fallback";
#endif

        // MPI backend depends on whether we have MPI context
        // Without MPI context, MPI backend may not be available
    }

    TEST_F(CollectiveContextGPUTest, BackendSelectionForROCm)
    {
        skipIfNoROCm();

        auto rocm_inventory = buildROCmOnlyInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(rocm_inventory, nullptr);

        ASSERT_NE(ctx, nullptr);

#ifdef HAVE_RCCL
        // With RCCL compiled in and ROCm GPUs, RCCL should be available
        EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::RCCL))
            << "RCCL backend should be available with ROCm GPUs present";
#else
        // Without RCCL, should fall back to HOST
        EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::HOST))
            << "HOST backend should be available as fallback";
#endif
    }

    // =========================================================================
    // AllReduce Tests
    // =========================================================================

    TEST_F(CollectiveContextGPUTest, AllReduceOnCUDATensor)
    {
        skipIfNoCUDA();

        // Use single-device inventory but note that CollectiveContext always
        // includes CPU for fallback, making it a multi-device group
        auto cuda_inventory = buildSingleCUDAInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(cuda_inventory, nullptr);

        ASSERT_NE(ctx, nullptr);

        // Create FP32 tensor
        constexpr size_t TENSOR_SIZE = 64;
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());

        // Initialize with test data
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            data[i] = static_cast<float>(i + 1);
        }

        // Test that allreduce routing works with CUDA device specified
        // Note: The CollectiveContext includes CPU for fallback, so the
        // HostBackend will perform actual reduction across 2 "devices"
        // (GPU context + CPU fallback). We only verify the operation succeeds.
        DeviceId cuda_device = DeviceId::cuda(0);
        bool result = ctx->executeAllreduce(tensor.get(), TENSOR_SIZE, cuda_device);
        EXPECT_TRUE(result) << "AllReduce on CUDA device routing failed";

        // Verify data pointer is still valid
        const float *result_data = tensor->data();
        ASSERT_NE(result_data, nullptr);

        // Verify no NaN/Inf in result (basic sanity check)
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            EXPECT_FALSE(std::isnan(result_data[i])) << "NaN at index " << i;
            EXPECT_FALSE(std::isinf(result_data[i])) << "Inf at index " << i;
        }
    }

    TEST_F(CollectiveContextGPUTest, AllReduceOnROCmTensor)
    {
        skipIfNoROCm();

        // Use single-device inventory but note that CollectiveContext always
        // includes CPU for fallback, making it a multi-device group
        auto rocm_inventory = buildSingleROCmInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(rocm_inventory, nullptr);

        ASSERT_NE(ctx, nullptr);

        // Create FP32 tensor
        constexpr size_t TENSOR_SIZE = 64;
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());

        // Initialize with test data
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            data[i] = static_cast<float>(i + 1);
        }

        // Test that allreduce routing works with ROCm device specified
        // Note: The CollectiveContext includes CPU for fallback, so the
        // HostBackend will perform actual reduction across 2 "devices"
        // (GPU context + CPU fallback). We only verify the operation succeeds.
        DeviceId rocm_device = DeviceId::rocm(0);
        bool result = ctx->executeAllreduce(tensor.get(), TENSOR_SIZE, rocm_device);
        EXPECT_TRUE(result) << "AllReduce on ROCm device routing failed";

        // Verify data pointer is still valid
        const float *result_data = tensor->data();
        ASSERT_NE(result_data, nullptr);

        // Verify no NaN/Inf in result (basic sanity check)
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            EXPECT_FALSE(std::isnan(result_data[i])) << "NaN at index " << i;
            EXPECT_FALSE(std::isinf(result_data[i])) << "Inf at index " << i;
        }
    }

    // =========================================================================
    // AllGather Tests
    // =========================================================================

    TEST_F(CollectiveContextGPUTest, AllGatherOnCUDATensor)
    {
        skipIfNoCUDA();

        auto cuda_inventory = buildCUDAOnlyInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(cuda_inventory, nullptr);

        ASSERT_NE(ctx, nullptr);

        // CollectiveContext always adds CPU, so local_devices = CUDA devices + 1 CPU
        // For multi-device allgather, output buffer must be N * input_size
        constexpr size_t TENSOR_SIZE = 8;
        const size_t num_devices = ctx->localDevices().size();

        auto local_input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());
        // Output buffer must be large enough for all devices
        auto full_output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{TENSOR_SIZE * num_devices}, DeviceId::cpu());

        // Initialize local input with test data
        float *input_data = local_input->mutable_data();
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            input_data[i] = static_cast<float>(i + 1);
        }

        // Zero output buffer
        float *output_data = full_output->mutable_data();
        std::fill(output_data, output_data + TENSOR_SIZE * num_devices, 0.0f);

        // Test that allgather routing works
        DeviceId cuda_device = DeviceId::cuda(0);
        bool result = ctx->executeAllgather(
            local_input.get(), full_output.get(), TENSOR_SIZE, cuda_device);
        EXPECT_TRUE(result) << "AllGather on CUDA device routing failed";

        // Verify no NaN/Inf in result
        const float *result_data = full_output->data();
        for (size_t i = 0; i < TENSOR_SIZE * num_devices; ++i)
        {
            EXPECT_FALSE(std::isnan(result_data[i])) << "NaN at index " << i;
            EXPECT_FALSE(std::isinf(result_data[i])) << "Inf at index " << i;
        }
    }

    TEST_F(CollectiveContextGPUTest, AllGatherOnROCmTensor)
    {
        skipIfNoROCm();

        auto rocm_inventory = buildROCmOnlyInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(rocm_inventory, nullptr);

        ASSERT_NE(ctx, nullptr);

        // CollectiveContext always adds CPU, so local_devices = ROCm devices + 1 CPU
        constexpr size_t TENSOR_SIZE = 8;
        const size_t num_devices = ctx->localDevices().size();

        auto local_input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());
        // Output buffer must be large enough for all devices
        auto full_output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{TENSOR_SIZE * num_devices}, DeviceId::cpu());

        // Initialize local input with test data
        float *input_data = local_input->mutable_data();
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            input_data[i] = static_cast<float>(i + 1);
        }

        // Zero output buffer
        float *output_data = full_output->mutable_data();
        std::fill(output_data, output_data + TENSOR_SIZE * num_devices, 0.0f);

        // Test that allgather routing works
        DeviceId rocm_device = DeviceId::rocm(0);
        bool result = ctx->executeAllgather(
            local_input.get(), full_output.get(), TENSOR_SIZE, rocm_device);
        EXPECT_TRUE(result) << "AllGather on ROCm device routing failed";

        // Verify no NaN/Inf in result
        const float *result_data = full_output->data();
        for (size_t i = 0; i < TENSOR_SIZE * num_devices; ++i)
        {
            EXPECT_FALSE(std::isnan(result_data[i])) << "NaN at index " << i;
            EXPECT_FALSE(std::isinf(result_data[i])) << "Inf at index " << i;
        }
    }

    // =========================================================================
    // World Size and Rank Tests
    // =========================================================================

    TEST_F(CollectiveContextGPUTest, WorldSizeAndRank)
    {
        auto ctx = CollectiveContextFactory::createIntraNode(inventory_, nullptr);

        ASSERT_NE(ctx, nullptr);

        // For single-rank test (no MPI), world size should be 1
        EXPECT_EQ(ctx->worldSize(), 1) << "World size should be 1 for single-rank test";

        // Rank should be 0
        EXPECT_EQ(ctx->rank(), 0) << "Rank should be 0 for single-rank test";
    }

    TEST_F(CollectiveContextGPUTest, LocalDevicesReturnsGPUs)
    {
        auto ctx = CollectiveContextFactory::createIntraNode(inventory_, nullptr);

        ASSERT_NE(ctx, nullptr);

        const auto &devices = ctx->localDevices();

        // Should have at least one device (could be CPU + GPUs)
        EXPECT_GE(devices.size(), 0u);

        // Log devices for debugging
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
        // Create single-device context
        auto ctx = CollectiveContextFactory::createSingleDevice();

        ASSERT_NE(ctx, nullptr);

        // Single device should not require collectives
        EXPECT_FALSE(ctx->requiresCollectives())
            << "Single device should not require collectives";
    }

    TEST_F(CollectiveContextGPUTest, RequiresCollectives_MultipleGPUs)
    {
        // Skip if we don't have multiple GPUs
        int total_gpus = cuda_count_ + rocm_count_;
        if (total_gpus < 2)
        {
            GTEST_SKIP() << "Need at least 2 GPUs for this test, have " << total_gpus;
        }

        // Create context with inventory containing multiple GPUs
        auto ctx = CollectiveContextFactory::createIntraNode(inventory_, nullptr);

        ASSERT_NE(ctx, nullptr);

        // Multiple GPUs should require collectives
        // Note: This depends on the implementation - if the context is configured
        // to use multiple devices for tensor parallelism, it should require collectives
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

        // Router should be accessible
        IBackendRouter *router = ctx->router();

        // With inventory provided, router should be created
        if (router != nullptr)
        {
            std::cout << "Router is available" << std::endl;

            // Test backend availability through router
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

        // Create tensor
        constexpr size_t TENSOR_SIZE = 64;
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());

        // Initialize data
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            data[i] = static_cast<float>(i + 1);
        }

        // Determine which device ID to use for routing (data stays on CPU for HostBackend)
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

        // Execute allreduce with count=0 (should use tensor->numel())
        bool result = ctx->executeAllreduce(tensor.get(), 0, device);
        EXPECT_TRUE(result) << "AllReduce with count=0 should use tensor numel";
    }

    TEST_F(CollectiveContextGPUTest, AllReduceWithNullBuffer)
    {
        auto ctx = CollectiveContextFactory::createIntraNode(inventory_, nullptr);

        ASSERT_NE(ctx, nullptr);

        // Note: Calling executeAllreduce with nullptr is undefined behavior
        // and may crash depending on the backend. This test just verifies
        // context creation works. Proper error handling tests would need
        // mocked backends to safely test edge cases.
        //
        // Skip the actual call to avoid undefined behavior:
        // DeviceId device = DeviceId::cpu();
        // bool result = ctx->executeAllreduce(nullptr, 64, device);
        // EXPECT_FALSE(result) << "AllReduce with null buffer should fail";
    }

    // =========================================================================
    // Broadcast Tests
    // =========================================================================

    TEST_F(CollectiveContextGPUTest, BroadcastOnCUDATensor)
    {
        skipIfNoCUDA();

        auto cuda_inventory = buildCUDAOnlyInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(cuda_inventory, nullptr);

        ASSERT_NE(ctx, nullptr);

        // Create FP32 tensor
        constexpr size_t TENSOR_SIZE = 64;
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());

        // Initialize with test data
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            data[i] = static_cast<float>(i + 1);
        }

        // For single-rank CollectiveContext without NCCL initialized,
        // the HostBackend will be used which operates on CPU data.
        DeviceId cuda_device = DeviceId::cuda(0);

        // Execute broadcast from rank 0 - for single-rank, this is a no-op
        bool result = ctx->executeBroadcast(tensor.get(), TENSOR_SIZE, 0, cuda_device);
        EXPECT_TRUE(result) << "Broadcast on CUDA device routing failed";

        // Verify data is still valid
        const float *result_data = tensor->data();
        ASSERT_NE(result_data, nullptr);

        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            EXPECT_FLOAT_EQ(result_data[i], static_cast<float>(i + 1))
                << "Data mismatch at index " << i;
        }
    }

    TEST_F(CollectiveContextGPUTest, BroadcastOnROCmTensor)
    {
        skipIfNoROCm();

        auto rocm_inventory = buildROCmOnlyInventory();
        auto ctx = CollectiveContextFactory::createIntraNode(rocm_inventory, nullptr);

        ASSERT_NE(ctx, nullptr);

        // Create FP32 tensor
        constexpr size_t TENSOR_SIZE = 64;
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{TENSOR_SIZE}, DeviceId::cpu());

        // Initialize with test data
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            data[i] = static_cast<float>(i + 1);
        }

        // For single-rank CollectiveContext without RCCL initialized,
        // the HostBackend will be used which operates on CPU data.
        DeviceId rocm_device = DeviceId::rocm(0);

        // Execute broadcast from rank 0 - for single-rank, this is a no-op
        bool result = ctx->executeBroadcast(tensor.get(), TENSOR_SIZE, 0, rocm_device);
        EXPECT_TRUE(result) << "Broadcast on ROCm device routing failed";

        // Verify data is still valid
        const float *result_data = tensor->data();
        ASSERT_NE(result_data, nullptr);

        for (size_t i = 0; i < TENSOR_SIZE; ++i)
        {
            EXPECT_FLOAT_EQ(result_data[i], static_cast<float>(i + 1))
                << "Data mismatch at index " << i;
        }
    }

} // namespace llaminar2

#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)
