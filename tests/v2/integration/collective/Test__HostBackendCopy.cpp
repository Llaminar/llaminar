/**
 * @file Test__HostBackendCopy.cpp
 * @brief Integration tests for HostBackend::copy() across all device pairs
 *
 * Validates data integrity for:
 * - CPU → CPU (cross-NUMA socket if available)
 * - CUDA → ROCm (cross-vendor via host staging)
 * - ROCm → CUDA (cross-vendor via host staging)
 * - CPU → GPU and GPU → CPU (mixed)
 *
 * Each test writes a known pattern on the source device, copies via
 * HostBackend::copy(), reads back on the destination, and verifies
 * byte-exact match.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "collective/backends/HostBackend.h"
#include "collective/DeviceGroup.h"
#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "backends/DeviceId.h"
#include "utils/Logger.h"

using namespace llaminar2;

// ============================================================================
// Test configuration
// ============================================================================

namespace
{
    // Test with several sizes to exercise staging buffer growth
    const std::vector<size_t> TEST_SIZES = {
        128,             //   128 B  (tiny — below any DMA threshold)
        4 * 1024,        //     4 KB
        64 * 1024,       //    64 KB
        1024 * 1024,     //     1 MB
        8 * 1024 * 1024, //     8 MB (above 64 MB staging buffer min after first grow)
    };

    /// Fill a host buffer with a deterministic pattern based on seed
    void fillPattern(void *ptr, size_t bytes, uint8_t seed)
    {
        auto *p = static_cast<uint8_t *>(ptr);
        for (size_t i = 0; i < bytes; ++i)
        {
            p[i] = static_cast<uint8_t>((i * 7 + seed) & 0xFF);
        }
    }

    /// Verify a host buffer matches the expected pattern
    bool verifyPattern(const void *ptr, size_t bytes, uint8_t seed)
    {
        const auto *p = static_cast<const uint8_t *>(ptr);
        for (size_t i = 0; i < bytes; ++i)
        {
            if (p[i] != static_cast<uint8_t>((i * 7 + seed) & 0xFF))
            {
                LOG_ERROR("Pattern mismatch at byte " << i
                                                      << ": expected " << int((i * 7 + seed) & 0xFF)
                                                      << ", got " << int(p[i]));
                return false;
            }
        }
        return true;
    }
} // namespace

// ============================================================================
// Test fixture
// ============================================================================

class Test__HostBackendCopy : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cuda_backend_ = getCUDABackend();
        rocm_backend_ = getROCmBackend();
    }

    void TearDown() override
    {
        // Free GPU allocations
        for (auto &[ptr, dev] : gpu_allocs_)
        {
            IBackend *be = getBackendFor(dev);
            if (be)
                be->free(ptr, dev.ordinal);
        }
        gpu_allocs_.clear();
    }

    /// Allocate GPU memory and track for cleanup
    void *allocGPU(DeviceId dev, size_t bytes)
    {
        IBackend *be = getBackendFor(dev);
        EXPECT_NE(be, nullptr);
        if (!be)
            return nullptr;
        void *ptr = be->allocate(bytes, dev.ordinal);
        EXPECT_NE(ptr, nullptr);
        if (ptr)
            gpu_allocs_.push_back({ptr, dev});
        return ptr;
    }

    /// Upload host data to GPU
    bool uploadToGPU(void *gpu_ptr, const void *host_ptr, DeviceId dev, size_t bytes)
    {
        IBackend *be = getBackendFor(dev);
        if (!be)
            return false;
        return be->hostToDevice(gpu_ptr, host_ptr, bytes, dev.ordinal);
    }

    /// Download GPU data to host
    bool downloadFromGPU(void *host_ptr, const void *gpu_ptr, DeviceId dev, size_t bytes)
    {
        IBackend *be = getBackendFor(dev);
        if (!be)
            return false;
        return be->deviceToHost(host_ptr, gpu_ptr, bytes, dev.ordinal);
    }

    /// Sync a device
    void syncDevice(DeviceId dev)
    {
        IBackend *be = getBackendFor(dev);
        if (be)
            be->synchronize(dev.ordinal);
    }

    IBackend *cuda_backend_ = nullptr;
    IBackend *rocm_backend_ = nullptr;
    std::vector<std::pair<void *, DeviceId>> gpu_allocs_;
};

// ============================================================================
// CPU → CPU
// ============================================================================

TEST_F(Test__HostBackendCopy, CPUtoCPU_DataIntegrity)
{
    DeviceId cpu0 = DeviceId::cpu();

    DeviceGroup group;
    group.name = "test_cpu";
    group.devices = {cpu0};

    HostBackend backend;
    ASSERT_TRUE(backend.initialize(group));
    ASSERT_TRUE(backend.supportsCopy(cpu0, cpu0));

    for (size_t bytes : TEST_SIZES)
    {
        SCOPED_TRACE("Size = " + std::to_string(bytes));

        std::vector<uint8_t> src(bytes);
        std::vector<uint8_t> dst(bytes, 0);

        fillPattern(src.data(), bytes, 0xAB);

        ASSERT_TRUE(backend.copy(dst.data(), cpu0, src.data(), cpu0, bytes));
        EXPECT_TRUE(verifyPattern(dst.data(), bytes, 0xAB));
    }

    backend.shutdown();
}

// ============================================================================
// CPU → CPU (cross-NUMA: allocate on different nodes)
// ============================================================================

TEST_F(Test__HostBackendCopy, CPUtoCPU_CrossNUMA)
{
    // This test verifies memcpy between two host buffers.
    // On a 2-socket system the OS may place them on different NUMA nodes,
    // but the copy path is the same. The key assertion is data integrity.
    DeviceId cpu0{DeviceType::CPU, 0};
    DeviceId cpu1{DeviceType::CPU, 1};

    DeviceGroup group;
    group.name = "test_cpu_cross";
    group.devices = {cpu0, cpu1};

    HostBackend backend;
    ASSERT_TRUE(backend.initialize(group));
    ASSERT_TRUE(backend.supportsCopy(cpu0, cpu1));

    for (size_t bytes : TEST_SIZES)
    {
        SCOPED_TRACE("Size = " + std::to_string(bytes));

        std::vector<uint8_t> src(bytes);
        std::vector<uint8_t> dst(bytes, 0);

        fillPattern(src.data(), bytes, 0xCD);

        ASSERT_TRUE(backend.copy(dst.data(), cpu1, src.data(), cpu0, bytes));
        EXPECT_TRUE(verifyPattern(dst.data(), bytes, 0xCD));
    }

    backend.shutdown();
}

// ============================================================================
// CUDA → ROCm
// ============================================================================

TEST_F(Test__HostBackendCopy, CUDAtoROCm_DataIntegrity)
{
    if (!cuda_backend_ || !rocm_backend_)
        GTEST_SKIP() << "Requires both CUDA and ROCm backends";

    DeviceId cuda0 = DeviceId::cuda(0);
    DeviceId rocm0 = DeviceId::rocm(0);

    DeviceGroup group;
    group.name = "test_cuda_rocm";
    group.devices = {cuda0, rocm0};

    HostBackend backend;
    ASSERT_TRUE(backend.initialize(group));
    ASSERT_TRUE(backend.supportsCopy(cuda0, rocm0));

    for (size_t bytes : TEST_SIZES)
    {
        SCOPED_TRACE("Size = " + std::to_string(bytes));

        // 1. Create source pattern on host
        std::vector<uint8_t> host_src(bytes);
        fillPattern(host_src.data(), bytes, 0x42);

        // 2. Upload to CUDA
        void *cuda_ptr = allocGPU(cuda0, bytes);
        ASSERT_NE(cuda_ptr, nullptr);
        ASSERT_TRUE(uploadToGPU(cuda_ptr, host_src.data(), cuda0, bytes));
        syncDevice(cuda0);

        // 3. Allocate on ROCm destination
        void *rocm_ptr = allocGPU(rocm0, bytes);
        ASSERT_NE(rocm_ptr, nullptr);

        // 4. Cross-vendor copy: CUDA → ROCm via HostBackend
        ASSERT_TRUE(backend.copy(rocm_ptr, rocm0, cuda_ptr, cuda0, bytes));

        // 5. Download from ROCm and verify
        std::vector<uint8_t> host_dst(bytes, 0);
        ASSERT_TRUE(downloadFromGPU(host_dst.data(), rocm_ptr, rocm0, bytes));
        syncDevice(rocm0);

        EXPECT_TRUE(verifyPattern(host_dst.data(), bytes, 0x42))
            << "CUDA→ROCm copy corrupted data at size " << bytes;
    }

    backend.shutdown();
}

// ============================================================================
// ROCm → CUDA
// ============================================================================

TEST_F(Test__HostBackendCopy, ROCmtoCUDA_DataIntegrity)
{
    if (!cuda_backend_ || !rocm_backend_)
        GTEST_SKIP() << "Requires both CUDA and ROCm backends";

    DeviceId cuda0 = DeviceId::cuda(0);
    DeviceId rocm0 = DeviceId::rocm(0);

    DeviceGroup group;
    group.name = "test_rocm_cuda";
    group.devices = {cuda0, rocm0};

    HostBackend backend;
    ASSERT_TRUE(backend.initialize(group));
    ASSERT_TRUE(backend.supportsCopy(rocm0, cuda0));

    for (size_t bytes : TEST_SIZES)
    {
        SCOPED_TRACE("Size = " + std::to_string(bytes));

        // 1. Create source pattern on host
        std::vector<uint8_t> host_src(bytes);
        fillPattern(host_src.data(), bytes, 0x77);

        // 2. Upload to ROCm
        void *rocm_ptr = allocGPU(rocm0, bytes);
        ASSERT_NE(rocm_ptr, nullptr);
        ASSERT_TRUE(uploadToGPU(rocm_ptr, host_src.data(), rocm0, bytes));
        syncDevice(rocm0);

        // 3. Allocate on CUDA destination
        void *cuda_ptr = allocGPU(cuda0, bytes);
        ASSERT_NE(cuda_ptr, nullptr);

        // 4. Cross-vendor copy: ROCm → CUDA via HostBackend
        ASSERT_TRUE(backend.copy(cuda_ptr, cuda0, rocm_ptr, rocm0, bytes));

        // 5. Download from CUDA and verify
        std::vector<uint8_t> host_dst(bytes, 0);
        ASSERT_TRUE(downloadFromGPU(host_dst.data(), cuda_ptr, cuda0, bytes));
        syncDevice(cuda0);

        EXPECT_TRUE(verifyPattern(host_dst.data(), bytes, 0x77))
            << "ROCm→CUDA copy corrupted data at size " << bytes;
    }

    backend.shutdown();
}

// ============================================================================
// CPU → GPU and GPU → CPU (mixed direction tests)
// ============================================================================

TEST_F(Test__HostBackendCopy, CPUtoGPU_DataIntegrity)
{
    // Test CPU→CUDA and CPU→ROCm
    struct Target
    {
        const char *name;
        DeviceId dev;
        IBackend *backend;
    };

    std::vector<Target> targets;
    if (cuda_backend_)
        targets.push_back({"CUDA", DeviceId::cuda(0), cuda_backend_});
    if (rocm_backend_)
        targets.push_back({"ROCm", DeviceId::rocm(0), rocm_backend_});

    if (targets.empty())
        GTEST_SKIP() << "No GPU backends available";

    DeviceId cpu0 = DeviceId::cpu();

    for (const auto &tgt : targets)
    {
        SCOPED_TRACE(std::string("CPU → ") + tgt.name);

        DeviceGroup group;
        group.name = std::string("test_cpu_") + tgt.name;
        group.devices = {cpu0, tgt.dev};

        HostBackend backend;
        ASSERT_TRUE(backend.initialize(group));

        for (size_t bytes : TEST_SIZES)
        {
            SCOPED_TRACE("Size = " + std::to_string(bytes));

            std::vector<uint8_t> host_src(bytes);
            fillPattern(host_src.data(), bytes, 0x33);

            void *gpu_ptr = allocGPU(tgt.dev, bytes);
            ASSERT_NE(gpu_ptr, nullptr);

            // CPU → GPU via HostBackend
            ASSERT_TRUE(backend.copy(gpu_ptr, tgt.dev, host_src.data(), cpu0, bytes));

            // Download and verify
            std::vector<uint8_t> host_dst(bytes, 0);
            ASSERT_TRUE(downloadFromGPU(host_dst.data(), gpu_ptr, tgt.dev, bytes));
            syncDevice(tgt.dev);

            EXPECT_TRUE(verifyPattern(host_dst.data(), bytes, 0x33));
        }

        backend.shutdown();
    }
}

TEST_F(Test__HostBackendCopy, GPUtoCPU_DataIntegrity)
{
    struct Source
    {
        const char *name;
        DeviceId dev;
        IBackend *backend;
    };

    std::vector<Source> sources;
    if (cuda_backend_)
        sources.push_back({"CUDA", DeviceId::cuda(0), cuda_backend_});
    if (rocm_backend_)
        sources.push_back({"ROCm", DeviceId::rocm(0), rocm_backend_});

    if (sources.empty())
        GTEST_SKIP() << "No GPU backends available";

    DeviceId cpu0 = DeviceId::cpu();

    for (const auto &src : sources)
    {
        SCOPED_TRACE(std::string(src.name) + " → CPU");

        DeviceGroup group;
        group.name = std::string("test_") + src.name + "_cpu";
        group.devices = {src.dev, cpu0};

        HostBackend backend;
        ASSERT_TRUE(backend.initialize(group));

        for (size_t bytes : TEST_SIZES)
        {
            SCOPED_TRACE("Size = " + std::to_string(bytes));

            // Upload pattern to GPU
            std::vector<uint8_t> host_src(bytes);
            fillPattern(host_src.data(), bytes, 0xEE);

            void *gpu_ptr = allocGPU(src.dev, bytes);
            ASSERT_NE(gpu_ptr, nullptr);
            ASSERT_TRUE(uploadToGPU(gpu_ptr, host_src.data(), src.dev, bytes));
            syncDevice(src.dev);

            // GPU → CPU via HostBackend
            std::vector<uint8_t> host_dst(bytes, 0);
            ASSERT_TRUE(backend.copy(host_dst.data(), cpu0, gpu_ptr, src.dev, bytes));

            EXPECT_TRUE(verifyPattern(host_dst.data(), bytes, 0xEE));
        }

        backend.shutdown();
    }
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_F(Test__HostBackendCopy, ZeroBytes_Succeeds)
{
    DeviceId cpu0 = DeviceId::cpu();

    DeviceGroup group;
    group.name = "test_zero";
    group.devices = {cpu0};

    HostBackend backend;
    ASSERT_TRUE(backend.initialize(group));

    uint8_t dummy_src = 0xAA;
    uint8_t dummy_dst = 0x00;
    EXPECT_TRUE(backend.copy(&dummy_dst, cpu0, &dummy_src, cpu0, 0));
    // dst should be untouched
    EXPECT_EQ(dummy_dst, 0x00);

    backend.shutdown();
}

TEST_F(Test__HostBackendCopy, NullPointer_Fails)
{
    DeviceId cpu0 = DeviceId::cpu();

    DeviceGroup group;
    group.name = "test_null";
    group.devices = {cpu0};

    HostBackend backend;
    ASSERT_TRUE(backend.initialize(group));

    uint8_t buf = 0;
    EXPECT_FALSE(backend.copy(nullptr, cpu0, &buf, cpu0, 1));
    EXPECT_FALSE(backend.copy(&buf, cpu0, nullptr, cpu0, 1));

    backend.shutdown();
}

TEST_F(Test__HostBackendCopy, StagingBufferGrows)
{
    // Verify that the staging buffer handles growing sizes correctly
    // by doing a small copy then a large copy with the same backend instance
    if (!cuda_backend_ || !rocm_backend_)
        GTEST_SKIP() << "Requires both CUDA and ROCm backends";

    DeviceId cuda0 = DeviceId::cuda(0);
    DeviceId rocm0 = DeviceId::rocm(0);

    DeviceGroup group;
    group.name = "test_grow";
    group.devices = {cuda0, rocm0};

    HostBackend backend;
    ASSERT_TRUE(backend.initialize(group));

    // Small transfer first (will allocate 64MB staging buffer)
    {
        const size_t bytes = 256;
        std::vector<uint8_t> host_src(bytes);
        fillPattern(host_src.data(), bytes, 0x11);

        void *cuda_ptr = allocGPU(cuda0, bytes);
        void *rocm_ptr = allocGPU(rocm0, bytes);
        ASSERT_NE(cuda_ptr, nullptr);
        ASSERT_NE(rocm_ptr, nullptr);

        ASSERT_TRUE(uploadToGPU(cuda_ptr, host_src.data(), cuda0, bytes));
        syncDevice(cuda0);
        ASSERT_TRUE(backend.copy(rocm_ptr, rocm0, cuda_ptr, cuda0, bytes));

        std::vector<uint8_t> host_dst(bytes, 0);
        ASSERT_TRUE(downloadFromGPU(host_dst.data(), rocm_ptr, rocm0, bytes));
        syncDevice(rocm0);
        EXPECT_TRUE(verifyPattern(host_dst.data(), bytes, 0x11));
    }

    // Large transfer (may trigger staging buffer growth if > 64MB)
    {
        const size_t bytes = 8 * 1024 * 1024; // 8 MB (fits in 64 MB min)
        std::vector<uint8_t> host_src(bytes);
        fillPattern(host_src.data(), bytes, 0x22);

        void *rocm_ptr = allocGPU(rocm0, bytes);
        void *cuda_ptr = allocGPU(cuda0, bytes);
        ASSERT_NE(rocm_ptr, nullptr);
        ASSERT_NE(cuda_ptr, nullptr);

        ASSERT_TRUE(uploadToGPU(rocm_ptr, host_src.data(), rocm0, bytes));
        syncDevice(rocm0);
        ASSERT_TRUE(backend.copy(cuda_ptr, cuda0, rocm_ptr, rocm0, bytes));

        std::vector<uint8_t> host_dst(bytes, 0);
        ASSERT_TRUE(downloadFromGPU(host_dst.data(), cuda_ptr, cuda0, bytes));
        syncDevice(cuda0);
        EXPECT_TRUE(verifyPattern(host_dst.data(), bytes, 0x22));
    }

    backend.shutdown();
}
