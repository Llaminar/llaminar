/**
 * @file Test__BackendCoordinatorIntegration.cpp
 * @brief Integration tests for NCCLBackend/RCCLBackend coordinator delegation
 * @author GitHub Copilot
 * @date February 2026
 *
 * Tests that verify NCCLBackend and RCCLBackend properly delegate to their
 * respective coordinators (NCCLCoordinator, RCCLCoordinator) for multi-GPU
 * single-process collective operations.
 *
 * These tests complement the direct coordinator tests by verifying:
 * - Backend initialization with multi-GPU DeviceGroup
 * - Backend lifecycle (initialize → operations → shutdown)
 * - Proper delegation of allreduceMulti, allgatherMulti, broadcastMulti
 * - Error handling when coordinator fails
 *
 * NOTE: These tests require GPU hardware. Tests will be skipped if:
 * - No CUDA devices available (NCCL tests)
 * - No ROCm devices available (RCCL tests)
 * - Less than 2 GPUs available (multi-GPU tests)
 *
 * IMPORTANT: CUDA and HIP headers cannot be included in the same translation unit
 * due to type conflicts. This test uses helper functions to abstract GPU operations
 * and keeps CUDA/HIP code in separate #ifdef blocks.
 *
 * @see NCCLBackend for NCCL backend implementation
 * @see RCCLBackend for RCCL backend implementation
 * @see NCCLCoordinator, RCCLCoordinator for coordinator implementations
 */

#include <gtest/gtest.h>

#include "collective/backends/NCCLBackend.h"
#include "collective/backends/RCCLBackend.h"
#include "collective/DeviceGroup.h"
#include "collective/ICollectiveBackend.h"
#include "backends/DeviceId.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

using namespace llaminar2;

// =============================================================================
// NCCL Backend Coordinator Integration Tests (CUDA-only section)
// =============================================================================

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

namespace gpu_helper_cuda
{

    void *cudaAlloc(int device_id, size_t bytes)
    {
        cudaSetDevice(device_id);
        void *ptr = nullptr;
        if (cudaMalloc(&ptr, bytes) != cudaSuccess)
        {
            return nullptr;
        }
        return ptr;
    }

    void cudaFreeHelper(int device_id, void *ptr)
    {
        cudaSetDevice(device_id);
        cudaFree(ptr);
    }

    void cudaCopyH2D(int device_id, void *dst, const void *src, size_t bytes)
    {
        cudaSetDevice(device_id);
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    void cudaCopyD2H(int device_id, void *dst, const void *src, size_t bytes)
    {
        cudaSetDevice(device_id);
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    void cudaSyncAll(int device_count)
    {
        for (int i = 0; i < device_count; ++i)
        {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
    }

    int cudaGetCount()
    {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess)
        {
            return 0;
        }
        return count;
    }

} // namespace gpu_helper_cuda

class Test__NCCLBackendCoordinatorIntegration : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cuda_device_count_ = gpu_helper_cuda::cudaGetCount();

        if (cuda_device_count_ > 0)
        {
            std::cout << "  CUDA devices available: " << cuda_device_count_ << std::endl;
            for (int i = 0; i < cuda_device_count_; ++i)
            {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                std::cout << "    GPU " << i << ": " << prop.name << std::endl;
            }
        }
    }

    void TearDown() override
    {
        gpu_helper_cuda::cudaSyncAll(cuda_device_count_);
    }

    DeviceGroup createCUDADeviceGroup(int num_gpus)
    {
        DeviceGroupBuilder builder;
        builder.setName("test_cuda_group");
        builder.setScope(CollectiveScope::LOCAL);
        builder.setLocalRank(0);

        for (int i = 0; i < num_gpus && i < cuda_device_count_; ++i)
        {
            builder.addDevice(DeviceId::cuda(i));
        }

        return builder.build();
    }

    void *allocateDeviceBuffer(int device_id, size_t bytes)
    {
        return gpu_helper_cuda::cudaAlloc(device_id, bytes);
    }

    void freeDeviceBuffer(int device_id, void *ptr)
    {
        gpu_helper_cuda::cudaFreeHelper(device_id, ptr);
    }

    void copyHostToDevice(int device_id, void *dst, const void *src, size_t bytes)
    {
        gpu_helper_cuda::cudaCopyH2D(device_id, dst, src, bytes);
    }

    void copyDeviceToHost(int device_id, void *dst, const void *src, size_t bytes)
    {
        gpu_helper_cuda::cudaCopyD2H(device_id, dst, src, bytes);
    }

    void fillDeviceBuffer(int device_id, float *buffer, size_t count, float value)
    {
        std::vector<float> host_data(count, value);
        copyHostToDevice(device_id, buffer, host_data.data(), count * sizeof(float));
    }

    int cuda_device_count_ = 0;
};

// -----------------------------------------------------------------------------
// Initialization and Lifecycle Tests
// -----------------------------------------------------------------------------

TEST_F(Test__NCCLBackendCoordinatorIntegration, SingleGPUInitialization)
{
    if (cuda_device_count_ < 1)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    NCCLBackend backend;

    EXPECT_FALSE(backend.isInitialized());
    EXPECT_FALSE(backend.isMultiGpuSingleProcess());

    DeviceGroup group = createCUDADeviceGroup(1);
    ASSERT_EQ(group.size(), 1u);

    ASSERT_TRUE(backend.initialize(group)) << "Failed: " << backend.lastError();

    EXPECT_TRUE(backend.isInitialized());
    EXPECT_FALSE(backend.isMultiGpuSingleProcess());
    EXPECT_EQ(backend.numRanks(), 1);
    EXPECT_EQ(backend.localRank(), 0);

    backend.shutdown();
    EXPECT_FALSE(backend.isInitialized());
}

TEST_F(Test__NCCLBackendCoordinatorIntegration, MultiGPUInitialization)
{
    if (cuda_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 CUDA devices for multi-GPU test";
    }

    NCCLBackend backend;

    DeviceGroup group = createCUDADeviceGroup(2);
    ASSERT_EQ(group.size(), 2u);

    ASSERT_TRUE(backend.initialize(group)) << "Failed: " << backend.lastError();

    EXPECT_TRUE(backend.isInitialized());
    EXPECT_TRUE(backend.isMultiGpuSingleProcess()) << "Multi-GPU should use coordinator";
    EXPECT_EQ(backend.numRanks(), 2);

    backend.shutdown();
    EXPECT_FALSE(backend.isInitialized());
}

TEST_F(Test__NCCLBackendCoordinatorIntegration, ShutdownIdempotency)
{
    if (cuda_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 CUDA devices";
    }

    NCCLBackend backend;
    DeviceGroup group = createCUDADeviceGroup(2);

    ASSERT_TRUE(backend.initialize(group)) << "Failed: " << backend.lastError();

    backend.shutdown();
    EXPECT_FALSE(backend.isInitialized());

    // Second shutdown should be safe (idempotent)
    backend.shutdown();
    EXPECT_FALSE(backend.isInitialized());
}

TEST_F(Test__NCCLBackendCoordinatorIntegration, ReInitialization)
{
    if (cuda_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 CUDA devices";
    }

    NCCLBackend backend;
    DeviceGroup group = createCUDADeviceGroup(2);

    ASSERT_TRUE(backend.initialize(group)) << "Failed: " << backend.lastError();
    EXPECT_TRUE(backend.isInitialized());

    // Re-initialization (should shutdown first and succeed)
    ASSERT_TRUE(backend.initialize(group)) << "Re-init failed: " << backend.lastError();
    EXPECT_TRUE(backend.isInitialized());

    backend.shutdown();
}

// -----------------------------------------------------------------------------
// Multi-GPU Allreduce Tests
// -----------------------------------------------------------------------------

TEST_F(Test__NCCLBackendCoordinatorIntegration, MultiGPUAllreduceSumFloat32)
{
    if (cuda_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 CUDA devices for multi-GPU allreduce";
    }

    const int num_gpus = std::min(cuda_device_count_, 2);
    const size_t count = 1024;

    NCCLBackend backend;
    DeviceGroup group = createCUDADeviceGroup(num_gpus);

    ASSERT_TRUE(backend.initialize(group)) << "Failed: " << backend.lastError();
    ASSERT_TRUE(backend.isMultiGpuSingleProcess());

    std::vector<void *> buffers(num_gpus);
    for (int i = 0; i < num_gpus; ++i)
    {
        buffers[i] = allocateDeviceBuffer(i, count * sizeof(float));
        ASSERT_NE(buffers[i], nullptr) << "Failed to allocate buffer on GPU " << i;

        fillDeviceBuffer(i, static_cast<float *>(buffers[i]), count, static_cast<float>(i + 1));
    }

    ASSERT_TRUE(backend.allreduceMulti(buffers, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM))
        << "allreduceMulti failed: " << backend.lastError();

    ASSERT_TRUE(backend.synchronize()) << "Synchronize failed: " << backend.lastError();

    // Expected sum: 1 + 2 = 3 for 2 GPUs
    float expected_sum = 0.0f;
    for (int i = 0; i < num_gpus; ++i)
    {
        expected_sum += static_cast<float>(i + 1);
    }

    for (int i = 0; i < num_gpus; ++i)
    {
        std::vector<float> host_result(count);
        copyDeviceToHost(i, host_result.data(), buffers[i], count * sizeof(float));

        for (size_t j = 0; j < count; ++j)
        {
            EXPECT_NEAR(host_result[j], expected_sum, 1e-5f)
                << "Mismatch at GPU " << i << ", element " << j;
        }

        freeDeviceBuffer(i, buffers[i]);
    }

    backend.shutdown();
}

TEST_F(Test__NCCLBackendCoordinatorIntegration, MultiGPUAllreduceLargeBuffer)
{
    if (cuda_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 CUDA devices";
    }

    const int num_gpus = std::min(cuda_device_count_, 2);
    const size_t count = 1024 * 1024; // 1M elements = 4MB per GPU

    NCCLBackend backend;
    DeviceGroup group = createCUDADeviceGroup(num_gpus);

    ASSERT_TRUE(backend.initialize(group)) << "Failed: " << backend.lastError();

    std::vector<void *> buffers(num_gpus);
    for (int i = 0; i < num_gpus; ++i)
    {
        buffers[i] = allocateDeviceBuffer(i, count * sizeof(float));
        ASSERT_NE(buffers[i], nullptr);
        fillDeviceBuffer(i, static_cast<float *>(buffers[i]), count, static_cast<float>(i + 1));
    }

    ASSERT_TRUE(backend.allreduceMulti(buffers, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM))
        << "allreduceMulti failed: " << backend.lastError();

    ASSERT_TRUE(backend.synchronize());

    float expected_sum = static_cast<float>(num_gpus * (num_gpus + 1) / 2);

    for (int i = 0; i < num_gpus; ++i)
    {
        std::vector<float> host_result(count);
        copyDeviceToHost(i, host_result.data(), buffers[i], count * sizeof(float));

        EXPECT_NEAR(host_result[0], expected_sum, 1e-5f);
        EXPECT_NEAR(host_result[count - 1], expected_sum, 1e-5f);

        freeDeviceBuffer(i, buffers[i]);
    }

    backend.shutdown();
}

// -----------------------------------------------------------------------------
// Multi-GPU Allgather Tests
// -----------------------------------------------------------------------------

TEST_F(Test__NCCLBackendCoordinatorIntegration, MultiGPUAllgatherFloat32)
{
    if (cuda_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 CUDA devices for multi-GPU allgather";
    }

    const int num_gpus = std::min(cuda_device_count_, 2);
    const size_t send_count = 256;
    const size_t recv_count = send_count * num_gpus;

    NCCLBackend backend;
    DeviceGroup group = createCUDADeviceGroup(num_gpus);

    ASSERT_TRUE(backend.initialize(group)) << "Failed: " << backend.lastError();

    std::vector<void *> send_buffers(num_gpus);
    std::vector<void *> recv_buffers(num_gpus);
    std::vector<const void *> send_buffers_const(num_gpus);

    for (int i = 0; i < num_gpus; ++i)
    {
        send_buffers[i] = allocateDeviceBuffer(i, send_count * sizeof(float));
        recv_buffers[i] = allocateDeviceBuffer(i, recv_count * sizeof(float));
        ASSERT_NE(send_buffers[i], nullptr);
        ASSERT_NE(recv_buffers[i], nullptr);

        fillDeviceBuffer(i, static_cast<float *>(send_buffers[i]), send_count, static_cast<float>(i + 1) * 10.0f);
        send_buffers_const[i] = send_buffers[i];
    }

    ASSERT_TRUE(backend.allgatherMulti(send_buffers_const, recv_buffers, send_count, CollectiveDataType::FLOAT32))
        << "allgatherMulti failed: " << backend.lastError();

    ASSERT_TRUE(backend.synchronize());

    for (int recv_gpu = 0; recv_gpu < num_gpus; ++recv_gpu)
    {
        std::vector<float> host_result(recv_count);
        copyDeviceToHost(recv_gpu, host_result.data(), recv_buffers[recv_gpu], recv_count * sizeof(float));

        for (int src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            float expected_value = static_cast<float>(src_gpu + 1) * 10.0f;
            size_t offset = src_gpu * send_count;

            for (size_t j = 0; j < send_count; ++j)
            {
                EXPECT_NEAR(host_result[offset + j], expected_value, 1e-5f)
                    << "Mismatch at recv_gpu=" << recv_gpu << ", src_gpu=" << src_gpu << ", element=" << j;
            }
        }
    }

    for (int i = 0; i < num_gpus; ++i)
    {
        freeDeviceBuffer(i, send_buffers[i]);
        freeDeviceBuffer(i, recv_buffers[i]);
    }

    backend.shutdown();
}

// -----------------------------------------------------------------------------
// Multi-GPU Broadcast Tests
// -----------------------------------------------------------------------------

TEST_F(Test__NCCLBackendCoordinatorIntegration, MultiGPUBroadcastFromRoot0)
{
    if (cuda_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 CUDA devices for multi-GPU broadcast";
    }

    const int num_gpus = std::min(cuda_device_count_, 2);
    const size_t count = 512;
    const int root = 0;

    NCCLBackend backend;
    DeviceGroup group = createCUDADeviceGroup(num_gpus);

    ASSERT_TRUE(backend.initialize(group)) << "Failed: " << backend.lastError();

    std::vector<void *> buffers(num_gpus);
    for (int i = 0; i < num_gpus; ++i)
    {
        buffers[i] = allocateDeviceBuffer(i, count * sizeof(float));
        ASSERT_NE(buffers[i], nullptr);

        if (i == root)
        {
            fillDeviceBuffer(i, static_cast<float *>(buffers[i]), count, 42.0f);
        }
        else
        {
            fillDeviceBuffer(i, static_cast<float *>(buffers[i]), count, 0.0f);
        }
    }

    ASSERT_TRUE(backend.broadcastMulti(buffers, count, CollectiveDataType::FLOAT32, root))
        << "broadcastMulti failed: " << backend.lastError();

    ASSERT_TRUE(backend.synchronize());

    for (int i = 0; i < num_gpus; ++i)
    {
        std::vector<float> host_result(count);
        copyDeviceToHost(i, host_result.data(), buffers[i], count * sizeof(float));

        for (size_t j = 0; j < count; ++j)
        {
            EXPECT_NEAR(host_result[j], 42.0f, 1e-5f)
                << "Mismatch at GPU " << i << ", element " << j;
        }

        freeDeviceBuffer(i, buffers[i]);
    }

    backend.shutdown();
}

// -----------------------------------------------------------------------------
// Error Handling Tests
// -----------------------------------------------------------------------------

TEST_F(Test__NCCLBackendCoordinatorIntegration, AllreduceFailsWhenNotInitialized)
{
    NCCLBackend backend;

    std::vector<void *> buffers = {nullptr};
    EXPECT_FALSE(backend.allreduceMulti(buffers, 100, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_FALSE(backend.lastError().empty());
}

TEST_F(Test__NCCLBackendCoordinatorIntegration, AllreduceFailsWithWrongBufferCount)
{
    if (cuda_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 CUDA devices";
    }

    NCCLBackend backend;
    DeviceGroup group = createCUDADeviceGroup(2);
    ASSERT_TRUE(backend.initialize(group));

    // Pass wrong number of buffers (1 instead of 2)
    std::vector<void *> buffers = {nullptr};
    EXPECT_FALSE(backend.allreduceMulti(buffers, 100, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

    backend.shutdown();
}

#endif // HAVE_CUDA

// =============================================================================
// RCCL Backend Coordinator Integration Tests (ROCm-only section)
// =============================================================================
// NOTE: This section is only compiled when HAVE_ROCM is defined AND HAVE_CUDA
// is NOT defined, to avoid header conflicts. When both are defined, only
// NCCL tests are run from this file.

#if defined(HAVE_ROCM) && !defined(HAVE_CUDA)

#include <hip/hip_runtime.h>

namespace gpu_helper_hip
{

    void *hipAlloc(int device_id, size_t bytes)
    {
        hipSetDevice(device_id);
        void *ptr = nullptr;
        if (hipMalloc(&ptr, bytes) != hipSuccess)
        {
            return nullptr;
        }
        return ptr;
    }

    void hipFreeHelper(int device_id, void *ptr)
    {
        hipSetDevice(device_id);
        hipFree(ptr);
    }

    void hipCopyH2D(int device_id, void *dst, const void *src, size_t bytes)
    {
        hipSetDevice(device_id);
        hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
        hipDeviceSynchronize();
    }

    void hipCopyD2H(int device_id, void *dst, const void *src, size_t bytes)
    {
        hipSetDevice(device_id);
        hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
    }

    void hipSyncAll(int device_count)
    {
        for (int i = 0; i < device_count; ++i)
        {
            hipSetDevice(i);
            hipDeviceSynchronize();
        }
    }

    int hipGetCount()
    {
        int count = 0;
        if (hipGetDeviceCount(&count) != hipSuccess)
        {
            return 0;
        }
        return count;
    }

} // namespace gpu_helper_hip

class Test__RCCLBackendCoordinatorIntegration : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rocm_device_count_ = gpu_helper_hip::hipGetCount();

        if (rocm_device_count_ > 0)
        {
            std::cout << "  ROCm devices available: " << rocm_device_count_ << std::endl;
            for (int i = 0; i < rocm_device_count_; ++i)
            {
                hipDeviceProp_t prop;
                hipGetDeviceProperties(&prop, i);
                std::cout << "    GPU " << i << ": " << prop.name << std::endl;
            }
        }
    }

    void TearDown() override
    {
        gpu_helper_hip::hipSyncAll(rocm_device_count_);
    }

    DeviceGroup createROCmDeviceGroup(int num_gpus)
    {
        DeviceGroupBuilder builder;
        builder.setName("test_rocm_group");
        builder.setScope(CollectiveScope::LOCAL);
        builder.setLocalRank(0);

        for (int i = 0; i < num_gpus && i < rocm_device_count_; ++i)
        {
            builder.addDevice(DeviceId::rocm(i));
        }

        return builder.build();
    }

    void *allocateDeviceBuffer(int device_id, size_t bytes)
    {
        return gpu_helper_hip::hipAlloc(device_id, bytes);
    }

    void freeDeviceBuffer(int device_id, void *ptr)
    {
        gpu_helper_hip::hipFreeHelper(device_id, ptr);
    }

    void copyHostToDevice(int device_id, void *dst, const void *src, size_t bytes)
    {
        gpu_helper_hip::hipCopyH2D(device_id, dst, src, bytes);
    }

    void copyDeviceToHost(int device_id, void *dst, const void *src, size_t bytes)
    {
        gpu_helper_hip::hipCopyD2H(device_id, dst, src, bytes);
    }

    void fillDeviceBuffer(int device_id, float *buffer, size_t count, float value)
    {
        std::vector<float> host_data(count, value);
        copyHostToDevice(device_id, buffer, host_data.data(), count * sizeof(float));
    }

    int rocm_device_count_ = 0;
};

// -----------------------------------------------------------------------------
// Initialization and Lifecycle Tests
// -----------------------------------------------------------------------------

TEST_F(Test__RCCLBackendCoordinatorIntegration, SingleGPUInitialization)
{
    if (rocm_device_count_ < 1)
    {
        GTEST_SKIP() << "No ROCm devices available";
    }

    RCCLBackend backend;

    EXPECT_FALSE(backend.isInitialized());
    EXPECT_FALSE(backend.isMultiGpuSingleProcess());

    DeviceGroup group = createROCmDeviceGroup(1);
    ASSERT_EQ(group.size(), 1u);

    ASSERT_TRUE(backend.initialize(group)) << "Failed: " << backend.lastError();

    EXPECT_TRUE(backend.isInitialized());
    EXPECT_FALSE(backend.isMultiGpuSingleProcess());

    backend.shutdown();
    EXPECT_FALSE(backend.isInitialized());
}

TEST_F(Test__RCCLBackendCoordinatorIntegration, MultiGPUInitialization)
{
    if (rocm_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices for multi-GPU test";
    }

    RCCLBackend backend;

    DeviceGroup group = createROCmDeviceGroup(2);
    ASSERT_EQ(group.size(), 2u);

    ASSERT_TRUE(backend.initialize(group)) << "Failed: " << backend.lastError();

    EXPECT_TRUE(backend.isInitialized());
    EXPECT_TRUE(backend.isMultiGpuSingleProcess()) << "Multi-GPU should use coordinator";

    backend.shutdown();
    EXPECT_FALSE(backend.isInitialized());
}

TEST_F(Test__RCCLBackendCoordinatorIntegration, ShutdownIdempotency)
{
    if (rocm_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices";
    }

    RCCLBackend backend;
    DeviceGroup group = createROCmDeviceGroup(2);

    ASSERT_TRUE(backend.initialize(group));

    backend.shutdown();
    EXPECT_FALSE(backend.isInitialized());

    backend.shutdown();
    EXPECT_FALSE(backend.isInitialized());
}

// -----------------------------------------------------------------------------
// Multi-GPU Allreduce Tests
// -----------------------------------------------------------------------------

TEST_F(Test__RCCLBackendCoordinatorIntegration, MultiGPUAllreduceSumFloat32)
{
    if (rocm_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices";
    }

    const int num_gpus = std::min(rocm_device_count_, 2);
    const size_t count = 1024;

    RCCLBackend backend;
    DeviceGroup group = createROCmDeviceGroup(num_gpus);

    ASSERT_TRUE(backend.initialize(group)) << "Failed: " << backend.lastError();
    ASSERT_TRUE(backend.isMultiGpuSingleProcess());

    std::vector<void *> buffers(num_gpus);
    for (int i = 0; i < num_gpus; ++i)
    {
        buffers[i] = allocateDeviceBuffer(i, count * sizeof(float));
        ASSERT_NE(buffers[i], nullptr);
        fillDeviceBuffer(i, static_cast<float *>(buffers[i]), count, static_cast<float>(i + 1));
    }

    ASSERT_TRUE(backend.allreduceMulti(buffers, count, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM))
        << "allreduceMulti failed: " << backend.lastError();

    ASSERT_TRUE(backend.synchronize());

    float expected_sum = 0.0f;
    for (int i = 0; i < num_gpus; ++i)
    {
        expected_sum += static_cast<float>(i + 1);
    }

    for (int i = 0; i < num_gpus; ++i)
    {
        std::vector<float> host_result(count);
        copyDeviceToHost(i, host_result.data(), buffers[i], count * sizeof(float));

        for (size_t j = 0; j < count; ++j)
        {
            EXPECT_NEAR(host_result[j], expected_sum, 1e-5f)
                << "Mismatch at GPU " << i << ", element " << j;
        }

        freeDeviceBuffer(i, buffers[i]);
    }

    backend.shutdown();
}

// -----------------------------------------------------------------------------
// Multi-GPU Allgather Tests
// -----------------------------------------------------------------------------

TEST_F(Test__RCCLBackendCoordinatorIntegration, MultiGPUAllgatherFloat32)
{
    if (rocm_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices";
    }

    const int num_gpus = std::min(rocm_device_count_, 2);
    const size_t send_count = 256;
    const size_t recv_count = send_count * num_gpus;

    RCCLBackend backend;
    DeviceGroup group = createROCmDeviceGroup(num_gpus);

    ASSERT_TRUE(backend.initialize(group)) << "Failed: " << backend.lastError();

    std::vector<void *> send_buffers(num_gpus);
    std::vector<void *> recv_buffers(num_gpus);
    std::vector<const void *> send_buffers_const(num_gpus);

    for (int i = 0; i < num_gpus; ++i)
    {
        send_buffers[i] = allocateDeviceBuffer(i, send_count * sizeof(float));
        recv_buffers[i] = allocateDeviceBuffer(i, recv_count * sizeof(float));
        ASSERT_NE(send_buffers[i], nullptr);
        ASSERT_NE(recv_buffers[i], nullptr);

        fillDeviceBuffer(i, static_cast<float *>(send_buffers[i]), send_count, static_cast<float>(i + 1) * 10.0f);
        send_buffers_const[i] = send_buffers[i];
    }

    ASSERT_TRUE(backend.allgatherMulti(send_buffers_const, recv_buffers, send_count, CollectiveDataType::FLOAT32))
        << "allgatherMulti failed: " << backend.lastError();

    ASSERT_TRUE(backend.synchronize());

    for (int recv_gpu = 0; recv_gpu < num_gpus; ++recv_gpu)
    {
        std::vector<float> host_result(recv_count);
        copyDeviceToHost(recv_gpu, host_result.data(), recv_buffers[recv_gpu], recv_count * sizeof(float));

        for (int src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
        {
            float expected_value = static_cast<float>(src_gpu + 1) * 10.0f;
            size_t offset = src_gpu * send_count;

            for (size_t j = 0; j < send_count; ++j)
            {
                EXPECT_NEAR(host_result[offset + j], expected_value, 1e-5f)
                    << "Mismatch at recv_gpu=" << recv_gpu << ", src_gpu=" << src_gpu << ", element=" << j;
            }
        }
    }

    for (int i = 0; i < num_gpus; ++i)
    {
        freeDeviceBuffer(i, send_buffers[i]);
        freeDeviceBuffer(i, recv_buffers[i]);
    }

    backend.shutdown();
}

// -----------------------------------------------------------------------------
// Multi-GPU Broadcast Tests
// -----------------------------------------------------------------------------

TEST_F(Test__RCCLBackendCoordinatorIntegration, MultiGPUBroadcastFromRoot0)
{
    if (rocm_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices";
    }

    const int num_gpus = std::min(rocm_device_count_, 2);
    const size_t count = 512;
    const int root = 0;

    RCCLBackend backend;
    DeviceGroup group = createROCmDeviceGroup(num_gpus);

    ASSERT_TRUE(backend.initialize(group)) << "Failed: " << backend.lastError();

    std::vector<void *> buffers(num_gpus);
    for (int i = 0; i < num_gpus; ++i)
    {
        buffers[i] = allocateDeviceBuffer(i, count * sizeof(float));
        ASSERT_NE(buffers[i], nullptr);

        if (i == root)
        {
            fillDeviceBuffer(i, static_cast<float *>(buffers[i]), count, 42.0f);
        }
        else
        {
            fillDeviceBuffer(i, static_cast<float *>(buffers[i]), count, 0.0f);
        }
    }

    ASSERT_TRUE(backend.broadcastMulti(buffers, count, CollectiveDataType::FLOAT32, root))
        << "broadcastMulti failed: " << backend.lastError();

    ASSERT_TRUE(backend.synchronize());

    for (int i = 0; i < num_gpus; ++i)
    {
        std::vector<float> host_result(count);
        copyDeviceToHost(i, host_result.data(), buffers[i], count * sizeof(float));

        for (size_t j = 0; j < count; ++j)
        {
            EXPECT_NEAR(host_result[j], 42.0f, 1e-5f)
                << "Mismatch at GPU " << i << ", element " << j;
        }

        freeDeviceBuffer(i, buffers[i]);
    }

    backend.shutdown();
}

// -----------------------------------------------------------------------------
// Error Handling Tests
// -----------------------------------------------------------------------------

TEST_F(Test__RCCLBackendCoordinatorIntegration, AllreduceFailsWhenNotInitialized)
{
    RCCLBackend backend;

    std::vector<void *> buffers = {nullptr};
    EXPECT_FALSE(backend.allreduceMulti(buffers, 100, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
    EXPECT_FALSE(backend.lastError().empty());
}

TEST_F(Test__RCCLBackendCoordinatorIntegration, AllreduceFailsWithWrongBufferCount)
{
    if (rocm_device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ROCm devices";
    }

    RCCLBackend backend;
    DeviceGroup group = createROCmDeviceGroup(2);
    ASSERT_TRUE(backend.initialize(group));

    std::vector<void *> buffers = {nullptr};
    EXPECT_FALSE(backend.allreduceMulti(buffers, 100, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));

    backend.shutdown();
}

#endif // defined(HAVE_ROCM) && !defined(HAVE_CUDA)

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
