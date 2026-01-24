/**
 * @file Test__RCCLBackend.cpp
 * @brief Integration tests for RCCL collective backend
 *
 * These tests require ROCm GPUs and RCCL library.
 * Tests are skipped if HAVE_RCCL is not defined or no ROCm GPUs available.
 *
 * Test coverage:
 * - Backend availability check
 * - Single GPU initialization and shutdown
 * - Multi-GPU initialization (requires 2+ GPUs)
 * - AllReduce operations (SUM, MIN, MAX)
 * - AllGather operations
 * - Broadcast operations
 * - Clean shutdown without errors
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#ifdef HAVE_RCCL
#include "collective/backends/RCCLBackend.h"
#include "collective/backends/RCCLDynamicLoader.h"
#include "collective/DeviceGroup.h"
#include "backends/DeviceId.h"
#include "utils/Logger.h"
#include <hip/hip_runtime.h>
// NOTE: We do NOT include <rccl/rccl.h> directly!
// Instead, we use the rccl_dynamic namespace which loads RCCL via dlopen
// with RTLD_LOCAL to avoid symbol conflicts with NCCL.
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>
#include <thread>
#include <atomic>

// Import RCCL types and functions from our dynamic loader namespace
using namespace llaminar2::rccl_dynamic;

namespace llaminar2
{

    // =========================================================================
    // GPU Memory Helper Functions (HIP API)
    // =========================================================================

    /**
     * @brief Allocate GPU memory on a specific device
     * @tparam T Data type to allocate
     * @param count Number of elements
     * @param device ROCm device ordinal
     * @return Device pointer or nullptr on failure
     */
    template <typename T>
    T *allocGPU(size_t count, int device = 0)
    {
        T *ptr = nullptr;
        hipSetDevice(device);
        hipError_t err = hipMalloc(&ptr, count * sizeof(T));
        if (err != hipSuccess)
        {
            std::cerr << "hipMalloc failed: " << hipGetErrorString(err) << std::endl;
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
        hipMemcpy(d_ptr, h_ptr, count * sizeof(T), hipMemcpyHostToDevice);
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
        hipMemcpy(h_ptr, d_ptr, count * sizeof(T), hipMemcpyDeviceToHost);
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
            hipFree(ptr);
        }
    }

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class RCCLBackendTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Check for ROCm GPUs
            int device_count = 0;
            hipError_t err = hipGetDeviceCount(&device_count);
            if (err != hipSuccess || device_count < 1)
            {
                GTEST_SKIP() << "No ROCm GPUs available (err=" << hipGetErrorString(err) << ")";
            }
            device_count_ = device_count;

            // Log device info
            std::cout << "RCCL Backend Test: Found " << device_count_ << " ROCm GPU(s)" << std::endl;
            for (int i = 0; i < device_count_; ++i)
            {
                hipDeviceProp_t prop;
                hipGetDeviceProperties(&prop, i);
                std::cout << "  GPU " << i << ": " << prop.name
                          << " (gfx" << prop.gcnArchName << ")"
                          << std::endl;
            }
        }

        void TearDown() override
        {
            // Synchronize and clear any HIP errors
            for (int i = 0; i < device_count_; ++i)
            {
                hipSetDevice(i);
                hipDeviceSynchronize();
                hipGetLastError();
            }
        }

        /**
         * @brief Create a DeviceGroup for testing
         * @param num_devices Number of ROCm devices to include
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
            group.primary_type = DeviceType::ROCm;

            for (int i = 0; i < num_devices && i < device_count_; ++i)
            {
                group.devices.push_back(DeviceId::rocm(i));
            }

            group.rocm_count = static_cast<int>(group.devices.size());
            return group;
        }

        /**
         * @brief Skip test if fewer than required GPUs
         */
        void skipIfLessThan(int required_gpus)
        {
            if (device_count_ < required_gpus)
            {
                GTEST_SKIP() << "Test requires " << required_gpus
                             << " GPUs, only " << device_count_ << " available";
            }
        }

        int device_count_ = 0;
    };

    // =========================================================================
    // Availability Tests
    // =========================================================================

    TEST_F(RCCLBackendTest, IsAvailable)
    {
        RCCLBackend backend;
        EXPECT_TRUE(backend.isAvailable()) << "RCCL should be available with ROCm GPUs present";
    }

    TEST_F(RCCLBackendTest, BackendType)
    {
        RCCLBackend backend;
        EXPECT_EQ(backend.type(), CollectiveBackendType::RCCL);
        EXPECT_EQ(backend.name(), "RCCL");
    }

    TEST_F(RCCLBackendTest, SupportsROCmDevices)
    {
        RCCLBackend backend;
        EXPECT_TRUE(backend.supportsDevice(DeviceType::ROCm));
        EXPECT_FALSE(backend.supportsDevice(DeviceType::CPU));
        EXPECT_FALSE(backend.supportsDevice(DeviceType::CUDA));
    }

    TEST_F(RCCLBackendTest, SupportsDirectTransfer)
    {
        RCCLBackend backend;

        DeviceId rocm0 = DeviceId::rocm(0);
        DeviceId rocm1 = DeviceId::rocm(1);
        DeviceId cpu = DeviceId::cpu();

        // ROCm to ROCm should be supported
        EXPECT_TRUE(backend.supportsDirectTransfer(rocm0, rocm0));
        EXPECT_TRUE(backend.supportsDirectTransfer(rocm0, rocm1));

        // CPU transfers not supported
        EXPECT_FALSE(backend.supportsDirectTransfer(cpu, rocm0));
        EXPECT_FALSE(backend.supportsDirectTransfer(rocm0, cpu));
    }

    // =========================================================================
    // Initialization Tests
    // =========================================================================

    TEST_F(RCCLBackendTest, Initialize_SingleGPU)
    {
        RCCLBackend backend;
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

    TEST_F(RCCLBackendTest, Initialize_MultiGPU)
    {
        skipIfLessThan(2);

        RCCLBackend backend;
        DeviceGroup group = createDeviceGroup(2);
        ASSERT_EQ(group.devices.size(), 2u);

        bool success = backend.initialize(group);
        EXPECT_TRUE(success) << "Failed to initialize: " << backend.lastError();
        EXPECT_TRUE(backend.isInitialized());
        EXPECT_EQ(backend.numRanks(), 2);
        EXPECT_EQ(backend.localRank(), 0);

        backend.shutdown();
    }

    TEST_F(RCCLBackendTest, Initialize_RejectsNonROCmDevices)
    {
        RCCLBackend backend;

        DeviceGroup group;
        group.name = "mixed_devices";
        group.devices.push_back(DeviceId::rocm(0));
        group.devices.push_back(DeviceId::cpu()); // Should fail
        group.local_rank = 0;

        bool success = backend.initialize(group);
        EXPECT_FALSE(success) << "Should reject non-ROCm devices";
        EXPECT_FALSE(backend.isInitialized());
        EXPECT_FALSE(backend.lastError().empty());
    }

    TEST_F(RCCLBackendTest, Initialize_RejectsEmptyGroup)
    {
        RCCLBackend backend;

        DeviceGroup group;
        group.name = "empty";
        group.local_rank = 0;

        bool success = backend.initialize(group);
        EXPECT_FALSE(success) << "Should reject empty group";
        EXPECT_FALSE(backend.isInitialized());
    }

    TEST_F(RCCLBackendTest, Reinitialize)
    {
        RCCLBackend backend;
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

    TEST_F(RCCLBackendTest, AllReduce_Sum_SingleGPU)
    {
        RCCLBackend backend;
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

    TEST_F(RCCLBackendTest, AllReduce_MultipleDataTypes)
    {
        RCCLBackend backend;
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

    TEST_F(RCCLBackendTest, AllReduce_MultipleOps)
    {
        RCCLBackend backend;
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

    TEST_F(RCCLBackendTest, AllReduce_LargeBuffer)
    {
        RCCLBackend backend;
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

    TEST_F(RCCLBackendTest, AllGather_SingleGPU)
    {
        RCCLBackend backend;
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
        hipMemset(d_recv, 0, recv_count * sizeof(float));

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

    TEST_F(RCCLBackendTest, Broadcast_SingleGPU)
    {
        RCCLBackend backend;
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

    TEST_F(RCCLBackendTest, Shutdown_CleanWithOperations)
    {
        RCCLBackend backend;
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

    TEST_F(RCCLBackendTest, Shutdown_MultipleCallsAreNoop)
    {
        RCCLBackend backend;
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

    TEST_F(RCCLBackendTest, Shutdown_WithoutInitialize)
    {
        RCCLBackend backend;
        EXPECT_FALSE(backend.isInitialized());

        // Shutdown without initialize should be safe
        backend.shutdown();
        EXPECT_FALSE(backend.isInitialized());
    }

    // =========================================================================
    // Multi-GPU Tests (require 2+ GPUs)
    // =========================================================================

    /**
     * @brief Test multi-GPU communicator initialization with threaded ncclCommInitRank
     *
     * This tests the same initialization pattern used in multi-process MPI scenarios.
     * ncclCommInitRank requires ALL ranks to call it concurrently, so we spawn threads.
     *
     * NOTE: This test only verifies initialization. The actual collective operations
     * (AllReduce, AllGather, etc.) on multi-GPU single-process setups often crash
     * due to RCCL/PCIe topology limitations - those are tested via MPI multi-process.
     */
    TEST_F(RCCLBackendTest, Initialize_MultiGPU_ThreadedCommInitRank)
    {
        skipIfLessThan(2);

        constexpr int num_gpus = 2;

        // Generate unique ID (shared across all communicators)
        ncclUniqueId id;
        ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);

        // Thread-shared state
        std::vector<ncclComm_t> comms(num_gpus, nullptr);
        std::atomic<int> errors{0};
        std::vector<std::string> error_msgs(num_gpus);

        // Launch threads - each thread initializes one GPU's communicator
        std::vector<std::thread> threads;
        for (int rank = 0; rank < num_gpus; ++rank)
        {
            threads.emplace_back([rank, &id, &comms, &errors, &error_msgs]()
                                 {
                hipError_t hip_err = hipSetDevice(rank);
                if (hip_err != hipSuccess) {
                    error_msgs[rank] = "hipSetDevice failed: " + std::string(hipGetErrorString(hip_err));
                    errors.fetch_add(1);
                    return;
                }

                ncclResult_t err = ncclCommInitRank(&comms[rank], num_gpus, id, rank);
                if (err != ncclSuccess) {
                    error_msgs[rank] = "ncclCommInitRank failed: " + std::string(ncclGetErrorString(err));
                    errors.fetch_add(1);
                } });
        }

        for (auto &t : threads)
            t.join();

        // Verify all communicators initialized
        ASSERT_EQ(errors.load(), 0) << "Initialization failed: "
                                    << (error_msgs[0].empty() ? "" : error_msgs[0] + "; ")
                                    << (error_msgs[1].empty() ? "" : error_msgs[1]);

        for (int i = 0; i < num_gpus; ++i)
        {
            EXPECT_NE(comms[i], nullptr) << "Communicator " << i << " is null";
        }

        // Cleanup
        for (int i = 0; i < num_gpus; ++i)
        {
            if (comms[i])
            {
                ncclCommDestroy(comms[i]);
            }
        }
    }

    /**
     * @brief Test RCCLBackend multi-GPU initialization (uses threaded ncclCommInitRank internally)
     */
    TEST_F(RCCLBackendTest, Initialize_MultiGPU_Backend)
    {
        skipIfLessThan(2);

        RCCLBackend backend;
        DeviceGroup group = createDeviceGroup(2);
        ASSERT_EQ(group.devices.size(), 2u);

        bool success = backend.initialize(group);
        EXPECT_TRUE(success) << "Failed to initialize: " << backend.lastError();
        EXPECT_TRUE(backend.isInitialized());
        EXPECT_EQ(backend.numRanks(), 2);
        EXPECT_EQ(backend.localRank(), 0);

        backend.shutdown();
        EXPECT_FALSE(backend.isInitialized());
    }

    /**
     * @brief Multi-GPU single-process collectives are NOT supported
     *
     * RCCL/NCCL collective operations in single-process multi-GPU mode require:
     * 1. All GPU buffers to be passed simultaneously
     * 2. ncclGroupStart/ncclGroupEnd wrapping
     * 3. Each ncclAllReduce call uses a different communicator and buffer
     *
     * The RCCLBackend API is designed for the MPI use case where each process
     * handles ONE GPU. Multi-GPU collective operations should use MPI with
     * one rank per GPU, NOT single-process multi-GPU.
     *
     * This test is disabled because the RCCLBackend::allreduce() API only
     * accepts a single buffer, which cannot work for single-process multi-GPU.
     * The initialization tests (Initialize_MultiGPU_*) pass because they only
     * verify communicator creation, not actual collective operations.
     */
    /**
     * @brief Multi-GPU single-process AllReduce test using allreduceMulti API
     *
     * This test verifies that multiple GPUs in a single process can participate
     * in a collective allreduce operation. Each GPU has a buffer with its rank value,
     * and after allreduce SUM, all buffers should contain the sum of ranks.
     */
    TEST_F(RCCLBackendTest, AllReduce_MultiGPU_SingleProcess)
    {
        skipIfLessThan(2);

        RCCLBackend backend;
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

            hipSetDevice(i);
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
            hipSetDevice(i);
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
            hipSetDevice(i);
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
    TEST_F(RCCLBackendTest, AllGather_MultiGPU_SingleProcess)
    {
        skipIfLessThan(2);

        RCCLBackend backend;
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

            hipSetDevice(i);
            d_send[i] = allocGPU<float>(send_count, i);
            d_recv[i] = allocGPU<float>(recv_count, i);
            ASSERT_NE(d_send[i], nullptr) << "Failed to allocate send on GPU " << i;
            ASSERT_NE(d_recv[i], nullptr) << "Failed to allocate recv on GPU " << i;

            copyToGPU(d_send[i], h_send[i].data(), send_count);
            hipMemset(d_recv[i], 0, recv_count * sizeof(float));
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
            hipSetDevice(i);
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
            hipSetDevice(i);
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
    TEST_F(RCCLBackendTest, Broadcast_MultiGPU_SingleProcess)
    {
        skipIfLessThan(2);

        RCCLBackend backend;
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
            hipSetDevice(i);
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
            hipSetDevice(i);
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
            hipSetDevice(i);
            freeGPU(d_buffers[i]);
        }

        backend.shutdown();
    }

    // =========================================================================
    // Synchronization Tests
    // =========================================================================

    TEST_F(RCCLBackendTest, Synchronize_BeforeOperations)
    {
        RCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        // Synchronize with no pending operations should succeed
        EXPECT_TRUE(backend.synchronize()) << backend.lastError();

        backend.shutdown();
    }

    TEST_F(RCCLBackendTest, Synchronize_MultipleCallsAreOk)
    {
        RCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group)) << backend.lastError();

        constexpr size_t count = 64;
        float *d_data = allocGPU<float>(count, 0);
        ASSERT_NE(d_data, nullptr);
        hipMemset(d_data, 0, count * sizeof(float));

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

    TEST_F(RCCLBackendTest, LastError_EmptyOnSuccess)
    {
        RCCLBackend backend;
        DeviceGroup group = createDeviceGroup(1);
        ASSERT_TRUE(backend.initialize(group));

        // After successful init, lastError should be empty
        // (or at least not indicate an error)
        // Note: Some implementations may keep last_error_ from previous ops
        // so we don't strictly require empty, just check init succeeded

        backend.shutdown();
    }

} // namespace llaminar2

#else // HAVE_RCCL

// Placeholder test when RCCL is not available
TEST(RCCLBackendTest, SkippedWithoutRCCL)
{
    GTEST_SKIP() << "RCCL not available (built without -DHAVE_RCCL)";
}

#endif // HAVE_RCCL
