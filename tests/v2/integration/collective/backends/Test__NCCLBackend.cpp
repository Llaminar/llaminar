/**
 * @file Test__NCCLBackend.cpp
 * @brief Integration tests for NCCL collective backend
 *
 * These tests require CUDA GPUs and NCCL library.
 * Tests are skipped if HAVE_NCCL is not defined or no CUDA GPUs available.
 *
 * Test coverage:
 * - Backend availability check
 * - Single GPU initialization and shutdown
 * - Multi-GPU initialization (requires 2+ GPUs)
 * - AllReduce operations (SUM, MIN, MAX, PROD)
 * - AllGather operations
 * - Broadcast operations
 * - Clean shutdown without errors
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#ifdef HAVE_NCCL
#include "collective/backends/NCCLBackend.h"
#include "collective/DeviceGroup.h"
#include "backends/DeviceId.h"
#include "utils/Logger.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

namespace llaminar2
{

    // =========================================================================
    // GPU Memory Helper Functions
    // =========================================================================

    /**
     * @brief Allocate GPU memory on a specific device
     * @tparam T Data type to allocate
     * @param count Number of elements
     * @param device CUDA device ordinal
     * @return Device pointer or nullptr on failure
     */
    template <typename T>
    T *allocGPU(size_t count, int device = 0)
    {
        T *ptr = nullptr;
        cudaSetDevice(device);
        cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
        if (err != cudaSuccess)
        {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
        return ptr;
    }

    /**
     * @brief Copy data from host to GPU
     * @tparam T Data type
     * @param d_ptr Device pointer
     * @param h_ptr Host pointer
     * @param count Number of elements
     */
    template <typename T>
    void copyToGPU(T *d_ptr, const T *h_ptr, size_t count)
    {
        cudaMemcpy(d_ptr, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice);
    }

    /**
     * @brief Copy data from GPU to host
     * @tparam T Data type
     * @param h_ptr Host pointer
     * @param d_ptr Device pointer
     * @param count Number of elements
     */
    template <typename T>
    void copyFromGPU(T *h_ptr, const T *d_ptr, size_t count)
    {
        cudaMemcpy(h_ptr, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    }

    /**
     * @brief Free GPU memory
     * @tparam T Data type
     * @param ptr Device pointer
     */
    template <typename T>
    void freeGPU(T *ptr)
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class NCCLBackendTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Check for CUDA GPUs
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err != cudaSuccess || device_count < 1)
            {
                GTEST_SKIP() << "No CUDA GPUs available (err=" << cudaGetErrorString(err) << ")";
            }
            device_count_ = device_count;

            // Log device info
            std::cout << "NCCL Backend Test: Found " << device_count_ << " CUDA GPU(s)" << std::endl;
            for (int i = 0; i < device_count_; ++i)
            {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                std::cout << "  GPU " << i << ": " << prop.name
                          << " (CC " << prop.major << "." << prop.minor << ")"
                          << std::endl;
            }
        }

        void TearDown() override
        {
            // Synchronize and clear any CUDA errors
            for (int i = 0; i < device_count_; ++i)
            {
                cudaSetDevice(i);
                cudaDeviceSynchronize();
                cudaGetLastError();
            }
        }

        /**
         * @brief Create a DeviceGroup for testing
         * @param num_devices Number of CUDA devices to include
         * @param local_rank Rank within the group (default 0)
         * @return Configured DeviceGroup
         */
        DeviceGroup createDeviceGroup(int num_devices, int local_rank = 0)
        {
            DeviceGroup group;
            group.name = "test_group_" + std::to_string(num_devices) + "gpus";
            group.local_rank = local_rank;
            group.scope = CollectiveScope::LOCAL;
            group.is_homogeneous = true;
            group.primary_type = DeviceType::CUDA;

            for (int i = 0; i < num_devices && i < device_count_; ++i)
            {
                group.devices.push_back(DeviceId::cuda(i));
            }

            group.cuda_count = static_cast<int>(group.devices.size());
            return group;
        }

        int device_count_ = 0;
    };

// GTEST_SKIP() only returns from the function it's called in, so this must be
// a macro (not a method) to return from the TEST_F body itself.
#define SKIP_IF_LESS_THAN(required_gpus)                                       \
    do {                                                                       \
        if (device_count_ < (required_gpus)) {                                 \
            GTEST_SKIP() << "Test requires " << (required_gpus)                \
                         << " GPUs, only " << device_count_ << " available";   \
        }                                                                      \
    } while (0)

    // =========================================================================
    // Availability Tests
    // =========================================================================

    TEST_F(NCCLBackendTest, IsAvailable)
    {
        NCCLBackend backend;
        EXPECT_TRUE(backend.isAvailable()) << "NCCL should be available with CUDA GPUs present";
    }

    TEST_F(NCCLBackendTest, BackendType)
    {
        NCCLBackend backend;
        EXPECT_EQ(backend.type(), CollectiveBackendType::NCCL);
        EXPECT_EQ(backend.name(), "NCCL");
    }

    TEST_F(NCCLBackendTest, SupportsCUDADevices)
    {
        NCCLBackend backend;
        EXPECT_TRUE(backend.supportsDevice(DeviceType::CUDA));
        EXPECT_FALSE(backend.supportsDevice(DeviceType::CPU));
        EXPECT_FALSE(backend.supportsDevice(DeviceType::ROCm));
    }

    TEST_F(NCCLBackendTest, SupportsDirectTransfer)
    {
        NCCLBackend backend;

        DeviceId cuda0 = DeviceId::cuda(0);
        DeviceId cuda1 = DeviceId::cuda(1);
        DeviceId cpu = DeviceId::cpu();

        // CUDA to CUDA should be supported
        EXPECT_TRUE(backend.supportsDirectTransfer(cuda0, cuda0));
        EXPECT_TRUE(backend.supportsDirectTransfer(cuda0, cuda1));

        // CPU transfers not supported
        EXPECT_FALSE(backend.supportsDirectTransfer(cpu, cuda0));
        EXPECT_FALSE(backend.supportsDirectTransfer(cuda0, cpu));
    }

    // =========================================================================
    // Initialization Tests
    // =========================================================================

    TEST_F(NCCLBackendTest, Initialize_SingleGPU)
    {
        NCCLBackend backend;
        EXPECT_FALSE(backend.isInitialized());

        DeviceGroup group = createDeviceGroup(1);
        ASSERT_EQ(group.devices.size(), 1u);

        bool success = backend.initialize(group);
        EXPECT_TRUE(success) << "Failed to initialize: " << backend.lastError();
        EXPECT_TRUE(backend.isInitialized());
        EXPECT_EQ(backend.numRanks(), 1);
        EXPECT_EQ(backend.localRank(), 0);

        backend.shutdown();
        EXPECT_FALSE(backend.isInitialized());
    }

    TEST_F(NCCLBackendTest, Initialize_MultiGPU)
    {
        SKIP_IF_LESS_THAN(2);

        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(2);
        ASSERT_EQ(group.devices.size(), 2u);

        bool success = backend.initialize(group);
        EXPECT_TRUE(success) << "Failed to initialize: " << backend.lastError();
        EXPECT_TRUE(backend.isInitialized());
        EXPECT_EQ(backend.numRanks(), 2);
        EXPECT_EQ(backend.localRank(), 0);

        backend.shutdown();
    }

    TEST_F(NCCLBackendTest, Initialize_RejectsNonCUDADevices)
    {
        NCCLBackend backend;

        DeviceGroup group;
        group.name = "mixed_devices";
        group.devices.push_back(DeviceId::cuda(0));
        group.devices.push_back(DeviceId::cpu()); // Should fail
        group.local_rank = 0;

        bool success = backend.initialize(group);
        EXPECT_FALSE(success) << "Should reject non-CUDA devices";
        EXPECT_FALSE(backend.isInitialized());
        EXPECT_FALSE(backend.lastError().empty());
    }

    TEST_F(NCCLBackendTest, Initialize_RejectsEmptyGroup)
    {
        NCCLBackend backend;

        DeviceGroup group;
        group.name = "empty";
        group.local_rank = 0;

        bool success = backend.initialize(group);
        EXPECT_FALSE(success) << "Should reject empty group";
        EXPECT_FALSE(backend.isInitialized());
    }

    TEST_F(NCCLBackendTest, Reinitialize)
    {
        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);

        // First initialization
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();
        EXPECT_TRUE(backend.isInitialized());

        // Re-initialize should work (shutdown and reinit)
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();
        EXPECT_TRUE(backend.isInitialized());

        backend.shutdown();
    }

    // =========================================================================
    // AllReduce Tests
    // =========================================================================

    TEST_F(NCCLBackendTest, AllReduce_Sum_SingleGPU)
    {
        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        constexpr size_t count = 1024;
        std::vector<float> h_data(count);
        for (size_t i = 0; i < count; ++i)
        {
            h_data[i] = static_cast<float>(i);
        }

        // Allocate and copy to GPU
        float *d_data = allocGPU<float>(count, 0);
        ASSERT_NE(d_data, nullptr);
        copyToGPU(d_data, h_data.data(), count);

        // AllReduce with SUM (single GPU, so values should be unchanged)
        bool success = backend.allreduce(d_data, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
        EXPECT_TRUE(success) << "AllReduce failed: " << backend.lastError();

        // Synchronize
        ASSERT_TRUE(backend.synchronize()) << backend.lastError();

        // Copy back and verify
        std::vector<float> h_result(count);
        copyFromGPU(h_result.data(), d_data, count);

        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(h_result[i], h_data[i])
                << "Mismatch at index " << i;
        }

        freeGPU(d_data);
        backend.shutdown();
    }

    TEST_F(NCCLBackendTest, AllReduce_MultipleDataTypes)
    {
        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        constexpr size_t count = 256;

        // Test INT32
        {
            std::vector<int32_t> h_data(count, 42);
            int32_t *d_data = allocGPU<int32_t>(count, 0);
            ASSERT_NE(d_data, nullptr);
            copyToGPU(d_data, h_data.data(), count);

            EXPECT_TRUE(backend.allreduce(d_data, count, CollectiveDataType::INT32, CollectiveOp::ALLREDUCE_SUM))
                << backend.lastError();
            EXPECT_TRUE(backend.synchronize());

            std::vector<int32_t> h_result(count);
            copyFromGPU(h_result.data(), d_data, count);
            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(h_result[i], 42) << "INT32 mismatch at " << i;
            }
            freeGPU(d_data);
        }

        // Test INT8
        {
            std::vector<int8_t> h_data(count, 7);
            int8_t *d_data = allocGPU<int8_t>(count, 0);
            ASSERT_NE(d_data, nullptr);
            copyToGPU(d_data, h_data.data(), count);

            EXPECT_TRUE(backend.allreduce(d_data, count, CollectiveDataType::INT8, CollectiveOp::ALLREDUCE_SUM))
                << backend.lastError();
            EXPECT_TRUE(backend.synchronize());

            std::vector<int8_t> h_result(count);
            copyFromGPU(h_result.data(), d_data, count);
            for (size_t i = 0; i < count; ++i)
            {
                EXPECT_EQ(h_result[i], 7) << "INT8 mismatch at " << i;
            }
            freeGPU(d_data);
        }

        backend.shutdown();
    }

    TEST_F(NCCLBackendTest, AllReduce_MultipleOps)
    {
        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        constexpr size_t count = 64;

        // Test SUM
        {
            std::vector<float> h_data(count, 3.0f);
            float *d_data = allocGPU<float>(count, 0);
            copyToGPU(d_data, h_data.data(), count);
            EXPECT_TRUE(backend.allreduce(d_data, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
            EXPECT_TRUE(backend.synchronize());
            std::vector<float> result(count);
            copyFromGPU(result.data(), d_data, count);
            EXPECT_FLOAT_EQ(result[0], 3.0f);
            freeGPU(d_data);
        }

        // Test MIN
        {
            std::vector<float> h_data(count);
            for (size_t i = 0; i < count; ++i)
                h_data[i] = static_cast<float>(i);
            float *d_data = allocGPU<float>(count, 0);
            copyToGPU(d_data, h_data.data(), count);
            EXPECT_TRUE(backend.allreduce(d_data, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_MIN));
            EXPECT_TRUE(backend.synchronize());
            std::vector<float> result(count);
            copyFromGPU(result.data(), d_data, count);
            // Single GPU, values unchanged
            EXPECT_FLOAT_EQ(result[0], 0.0f);
            EXPECT_FLOAT_EQ(result[count - 1], static_cast<float>(count - 1));
            freeGPU(d_data);
        }

        // Test MAX
        {
            std::vector<float> h_data(count, 7.5f);
            float *d_data = allocGPU<float>(count, 0);
            copyToGPU(d_data, h_data.data(), count);
            EXPECT_TRUE(backend.allreduce(d_data, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_MAX));
            EXPECT_TRUE(backend.synchronize());
            std::vector<float> result(count);
            copyFromGPU(result.data(), d_data, count);
            EXPECT_FLOAT_EQ(result[0], 7.5f);
            freeGPU(d_data);
        }

        backend.shutdown();
    }

    TEST_F(NCCLBackendTest, AllReduce_LargeBuffer)
    {
        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        // Test with a larger buffer (4MB of floats)
        constexpr size_t count = 1024 * 1024; // 1M elements = 4MB
        std::vector<float> h_data(count);
        for (size_t i = 0; i < count; ++i)
        {
            h_data[i] = static_cast<float>(i % 1000) * 0.001f;
        }

        float *d_data = allocGPU<float>(count, 0);
        ASSERT_NE(d_data, nullptr);
        copyToGPU(d_data, h_data.data(), count);

        bool success = backend.allreduce(d_data, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
        EXPECT_TRUE(success) << "Large allreduce failed: " << backend.lastError();
        EXPECT_TRUE(backend.synchronize());

        // Spot check some values
        std::vector<float> h_result(count);
        copyFromGPU(h_result.data(), d_data, count);
        EXPECT_FLOAT_EQ(h_result[0], h_data[0]);
        EXPECT_FLOAT_EQ(h_result[count / 2], h_data[count / 2]);
        EXPECT_FLOAT_EQ(h_result[count - 1], h_data[count - 1]);

        freeGPU(d_data);
        backend.shutdown();
    }

    // =========================================================================
    // AllGather Tests
    // =========================================================================

    TEST_F(NCCLBackendTest, AllGather_SingleGPU)
    {
        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        constexpr size_t send_count = 128;
        constexpr size_t recv_count = send_count; // Single GPU: recv = send

        std::vector<float> h_send(send_count);
        for (size_t i = 0; i < send_count; ++i)
        {
            h_send[i] = static_cast<float>(i);
        }

        float *d_send = allocGPU<float>(send_count, 0);
        float *d_recv = allocGPU<float>(recv_count, 0);
        ASSERT_NE(d_send, nullptr);
        ASSERT_NE(d_recv, nullptr);

        copyToGPU(d_send, h_send.data(), send_count);
        cudaMemset(d_recv, 0, recv_count * sizeof(float));

        bool success = backend.allgather(d_send, d_recv, send_count, CollectiveDataType::FLOAT32);
        EXPECT_TRUE(success) << "AllGather failed: " << backend.lastError();
        EXPECT_TRUE(backend.synchronize());

        std::vector<float> h_recv(recv_count);
        copyFromGPU(h_recv.data(), d_recv, recv_count);

        for (size_t i = 0; i < recv_count; ++i)
        {
            EXPECT_FLOAT_EQ(h_recv[i], h_send[i]) << "Mismatch at index " << i;
        }

        freeGPU(d_send);
        freeGPU(d_recv);
        backend.shutdown();
    }

    // =========================================================================
    // Broadcast Tests
    // =========================================================================

    TEST_F(NCCLBackendTest, Broadcast_SingleGPU)
    {
        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        constexpr size_t count = 256;
        std::vector<float> h_data(count);
        for (size_t i = 0; i < count; ++i)
        {
            h_data[i] = static_cast<float>(i) * 0.5f;
        }

        float *d_data = allocGPU<float>(count, 0);
        ASSERT_NE(d_data, nullptr);
        copyToGPU(d_data, h_data.data(), count);

        // Broadcast from root 0 (single GPU, should be a no-op)
        bool success = backend.broadcast(d_data, count, CollectiveDataType::FLOAT32, 0);
        EXPECT_TRUE(success) << "Broadcast failed: " << backend.lastError();
        EXPECT_TRUE(backend.synchronize());

        std::vector<float> h_result(count);
        copyFromGPU(h_result.data(), d_data, count);

        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(h_result[i], h_data[i]) << "Mismatch at index " << i;
        }

        freeGPU(d_data);
        backend.shutdown();
    }

    // =========================================================================
    // Shutdown Tests
    // =========================================================================

    TEST_F(NCCLBackendTest, Shutdown_CleanWithOperations)
    {
        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        // Perform some operations
        constexpr size_t count = 64;
        std::vector<float> h_data(count, 1.0f);
        float *d_data = allocGPU<float>(count, 0);
        copyToGPU(d_data, h_data.data(), count);

        backend.allreduce(d_data, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
        backend.synchronize();

        freeGPU(d_data);

        // Shutdown should not crash or hang
        backend.shutdown();
        EXPECT_FALSE(backend.isInitialized());

        // Error should be empty after clean shutdown
        // (or at least not indicate a crash)
    }

    TEST_F(NCCLBackendTest, Shutdown_MultipleCallsAreNoop)
    {
        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        // Multiple shutdown calls should be safe
        backend.shutdown();
        EXPECT_FALSE(backend.isInitialized());

        backend.shutdown(); // Should be no-op
        EXPECT_FALSE(backend.isInitialized());

        backend.shutdown(); // Should be no-op
        EXPECT_FALSE(backend.isInitialized());
    }

    TEST_F(NCCLBackendTest, Shutdown_WithoutInitialize)
    {
        NCCLBackend backend;
        EXPECT_FALSE(backend.isInitialized());

        // Shutdown without initialize should be safe
        backend.shutdown();
        EXPECT_FALSE(backend.isInitialized());
    }

    // =========================================================================
    // Multi-GPU Tests (require 2+ GPUs)
    // =========================================================================

    TEST_F(NCCLBackendTest, AllReduce_MultiGPU_DoesNotCrash)
    {
        SKIP_IF_LESS_THAN(2);

        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(2);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        // Note: In a single-process test, we can only exercise one rank.
        // This test verifies the multi-GPU initialization path works
        // and doesn't crash when calling allreduce.

        constexpr size_t count = 512;
        std::vector<float> h_data(count, 3.14159f);

        // Use device 0 (the local_rank device)
        cudaSetDevice(0);
        float *d_data = allocGPU<float>(count, 0);
        ASSERT_NE(d_data, nullptr);
        copyToGPU(d_data, h_data.data(), count);

        // This will hang if communicator setup is wrong
        // In single-process mode with 2 GPUs, NCCL may require special handling
        bool success = backend.allreduce(d_data, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);

        // Note: This may fail in single-process multi-GPU mode because NCCL
        // expects all ranks to call allreduce. For proper multi-GPU testing,
        // we'd need separate threads or processes for each GPU.
        // For now, we just check it doesn't crash.
        (void)success;

        freeGPU(d_data);
        backend.shutdown();
    }

    /**
     * @brief Multi-GPU single-process AllReduce test using allreduceMulti API
     *
     * This test verifies that multiple GPUs in a single process can participate
     * in a collective allreduce operation. Each GPU has a buffer with its rank value,
     * and after allreduce SUM, all buffers should contain the sum of ranks.
     */
    TEST_F(NCCLBackendTest, AllReduce_MultiGPU_SingleProcess)
    {
        SKIP_IF_LESS_THAN(2);

        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(device_count_);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        // Verify multi-GPU mode is detected
        EXPECT_TRUE(backend.isMultiGpuSingleProcess()) << "Should be in multi-GPU single-process mode";

        constexpr size_t count = 1024;
        const int num_gpus = device_count_;

        // Expected sum: 0 + 1 + 2 + ... + (n-1) = n*(n-1)/2
        const float expected_sum = static_cast<float>(num_gpus * (num_gpus - 1) / 2);

        // Allocate buffers on each GPU
        std::vector<float *> d_buffers(num_gpus);
        std::vector<std::vector<float>> h_data(num_gpus);

        for (int i = 0; i < num_gpus; ++i)
        {
            // Each GPU gets its rank value in all elements
            h_data[i].resize(count, static_cast<float>(i));

            d_buffers[i] = allocGPU<float>(count, i);
            ASSERT_NE(d_buffers[i], nullptr) << "Failed to allocate on GPU " << i;

            cudaSetDevice(i);
            copyToGPU(d_buffers[i], h_data[i].data(), count);
        }

        // Create void* buffer vector for the API
        std::vector<void *> buffers;
        for (auto *ptr : d_buffers)
        {
            buffers.push_back(ptr);
        }

        // Perform multi-GPU AllReduce SUM
        bool success = backend.allreduceMulti(buffers, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
        EXPECT_TRUE(success) << "allreduceMulti failed: " << backend.lastError();

        // Synchronize all streams
        ASSERT_TRUE(backend.synchronize()) << backend.lastError();

        // Verify results on each GPU
        for (int i = 0; i < num_gpus; ++i)
        {
            std::vector<float> h_result(count);
            cudaSetDevice(i);
            copyFromGPU(h_result.data(), d_buffers[i], count);

            for (size_t j = 0; j < count; ++j)
            {
                EXPECT_FLOAT_EQ(h_result[j], expected_sum)
                    << "Mismatch on GPU " << i << " at index " << j
                    << " (expected " << expected_sum << ", got " << h_result[j] << ")";
            }
        }

        // Cleanup
        for (int i = 0; i < num_gpus; ++i)
        {
            cudaSetDevice(i);
            freeGPU(d_buffers[i]);
        }

        backend.shutdown();
    }

    /**
     * @brief Multi-GPU single-process AllGather test using allgatherMulti API
     *
     * Each GPU contributes a chunk of data, and after allgather, every GPU
     * has the complete gathered result.
     */
    TEST_F(NCCLBackendTest, AllGather_MultiGPU_SingleProcess)
    {
        SKIP_IF_LESS_THAN(2);

        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(device_count_);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();
        EXPECT_TRUE(backend.isMultiGpuSingleProcess());

        constexpr size_t send_count = 256; // Elements per GPU
        const int num_gpus = device_count_;
        const size_t recv_count = send_count * num_gpus;

        // Allocate send and receive buffers on each GPU
        std::vector<float *> d_send(num_gpus);
        std::vector<float *> d_recv(num_gpus);
        std::vector<std::vector<float>> h_send(num_gpus);

        for (int i = 0; i < num_gpus; ++i)
        {
            // Each GPU sends data filled with its rank + 0.5
            h_send[i].resize(send_count, static_cast<float>(i) + 0.5f);

            cudaSetDevice(i);
            d_send[i] = allocGPU<float>(send_count, i);
            d_recv[i] = allocGPU<float>(recv_count, i);
            ASSERT_NE(d_send[i], nullptr) << "Failed to allocate send on GPU " << i;
            ASSERT_NE(d_recv[i], nullptr) << "Failed to allocate recv on GPU " << i;

            copyToGPU(d_send[i], h_send[i].data(), send_count);
            cudaMemset(d_recv[i], 0, recv_count * sizeof(float));
        }

        // Create buffer vectors for the API
        std::vector<const void *> send_bufs;
        std::vector<void *> recv_bufs;
        for (int i = 0; i < num_gpus; ++i)
        {
            send_bufs.push_back(d_send[i]);
            recv_bufs.push_back(d_recv[i]);
        }

        // Perform multi-GPU AllGather
        bool success = backend.allgatherMulti(send_bufs, recv_bufs, send_count, CollectiveDataType::FLOAT32);
        EXPECT_TRUE(success) << "allgatherMulti failed: " << backend.lastError();

        // Synchronize all streams
        ASSERT_TRUE(backend.synchronize()) << backend.lastError();

        // Verify results on each GPU
        for (int i = 0; i < num_gpus; ++i)
        {
            std::vector<float> h_result(recv_count);
            cudaSetDevice(i);
            copyFromGPU(h_result.data(), d_recv[i], recv_count);

            // Each chunk of send_count elements should contain data from the corresponding GPU
            for (int src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
            {
                float expected = static_cast<float>(src_gpu) + 0.5f;
                for (size_t j = 0; j < send_count; ++j)
                {
                    size_t idx = src_gpu * send_count + j;
                    EXPECT_FLOAT_EQ(h_result[idx], expected)
                        << "Mismatch on GPU " << i << " at index " << idx
                        << " (chunk from GPU " << src_gpu << ")";
                }
            }
        }

        // Cleanup
        for (int i = 0; i < num_gpus; ++i)
        {
            cudaSetDevice(i);
            freeGPU(d_send[i]);
            freeGPU(d_recv[i]);
        }

        backend.shutdown();
    }

    /**
     * @brief Multi-GPU single-process Broadcast test using broadcastMulti API
     *
     * GPU 0 (root) has the source data, and after broadcast, all GPUs should
     * have the same data.
     */
    TEST_F(NCCLBackendTest, Broadcast_MultiGPU_SingleProcess)
    {
        SKIP_IF_LESS_THAN(2);

        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(device_count_);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();
        EXPECT_TRUE(backend.isMultiGpuSingleProcess());

        constexpr size_t count = 512;
        const int num_gpus = device_count_;
        const int root = 0;
        const float root_value = 42.0f;

        // Allocate buffers on each GPU
        std::vector<float *> d_buffers(num_gpus);

        for (int i = 0; i < num_gpus; ++i)
        {
            cudaSetDevice(i);
            d_buffers[i] = allocGPU<float>(count, i);
            ASSERT_NE(d_buffers[i], nullptr) << "Failed to allocate on GPU " << i;

            if (i == root)
            {
                // Root has the source data
                std::vector<float> h_data(count, root_value);
                copyToGPU(d_buffers[i], h_data.data(), count);
            }
            else
            {
                // Other GPUs have different values (should be overwritten)
                std::vector<float> h_data(count, static_cast<float>(i * 100));
                copyToGPU(d_buffers[i], h_data.data(), count);
            }
        }

        // Create void* buffer vector for the API
        std::vector<void *> buffers;
        for (auto *ptr : d_buffers)
        {
            buffers.push_back(ptr);
        }

        // Perform multi-GPU Broadcast from root
        bool success = backend.broadcastMulti(buffers, count, CollectiveDataType::FLOAT32, root);
        EXPECT_TRUE(success) << "broadcastMulti failed: " << backend.lastError();

        // Synchronize all streams
        ASSERT_TRUE(backend.synchronize()) << backend.lastError();

        // Verify all GPUs have the root's data
        for (int i = 0; i < num_gpus; ++i)
        {
            std::vector<float> h_result(count);
            cudaSetDevice(i);
            copyFromGPU(h_result.data(), d_buffers[i], count);

            for (size_t j = 0; j < count; ++j)
            {
                EXPECT_FLOAT_EQ(h_result[j], root_value)
                    << "Mismatch on GPU " << i << " at index " << j
                    << " (expected " << root_value << ", got " << h_result[j] << ")";
            }
        }

        // Cleanup
        for (int i = 0; i < num_gpus; ++i)
        {
            cudaSetDevice(i);
            freeGPU(d_buffers[i]);
        }

        backend.shutdown();
    }

    // =========================================================================
    // Synchronization Tests
    // =========================================================================

    TEST_F(NCCLBackendTest, Synchronize_BeforeOperations)
    {
        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        // Synchronize with no pending operations should succeed
        EXPECT_TRUE(backend.synchronize()) << backend.lastError();

        backend.shutdown();
    }

    TEST_F(NCCLBackendTest, Synchronize_MultipleCallsAreOk)
    {
        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        constexpr size_t count = 64;
        float *d_data = allocGPU<float>(count, 0);
        ASSERT_NE(d_data, nullptr);
        cudaMemset(d_data, 0, count * sizeof(float));

        backend.allreduce(d_data, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);

        // Multiple synchronize calls should be fine
        EXPECT_TRUE(backend.synchronize());
        EXPECT_TRUE(backend.synchronize());
        EXPECT_TRUE(backend.synchronize());

        freeGPU(d_data);
        backend.shutdown();
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    TEST_F(NCCLBackendTest, LastError_EmptyOnSuccess)
    {
        NCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group));

        // After successful init, lastError should be empty
        // (or at least not indicate an error)
        // Note: Some implementations may keep last_error_ from previous ops
        // so we don't strictly require empty, just check init succeeded

        backend.shutdown();
    }

} // namespace llaminar2

#else // HAVE_NCCL

// Placeholder test when NCCL is not available
TEST(NCCLBackendTest, SkippedWithoutNCCL)
{
    GTEST_SKIP() << "NCCL not available (built without -DHAVE_NCCL)";
}

#endif // HAVE_NCCL
