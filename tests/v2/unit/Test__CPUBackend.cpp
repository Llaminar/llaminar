/**
 * @file Test__CPUBackend.cpp
 * @brief Unit tests for CPUBackend
 *
 * Tests the CPU/NUMA backend implementation of IBackend interface.
 * Validates device enumeration, memory queries, allocation/deallocation,
 * and transfer operations.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "backends/CPUBackend.h"
#include "backends/BackendManager.h"
#include "backends/DeviceId.h"

#include <cstring>
#include <vector>

namespace llaminar2
{
    namespace test
    {

        // ============================================================================
        // Device Enumeration Tests
        // ============================================================================

        TEST(Test__CPUBackend, DeviceCountIsOne)
        {
            CPUBackend backend(0);
            EXPECT_EQ(backend.deviceCount(), 1);
        }

        TEST(Test__CPUBackend, DeviceCountIsOneForAnyNumaNode)
        {
            // Even with different NUMA nodes, each rank sees exactly 1 device
            CPUBackend backend0(0);
            CPUBackend backend1(1);
            CPUBackend backendNeg(-1);

            EXPECT_EQ(backend0.deviceCount(), 1);
            EXPECT_EQ(backend1.deviceCount(), 1);
            EXPECT_EQ(backendNeg.deviceCount(), 1);
        }

        TEST(Test__CPUBackend, BackendNameIsCPU)
        {
            CPUBackend backend(0);
            EXPECT_EQ(backend.backendName(), "CPU");
        }

        TEST(Test__CPUBackend, DeviceNameIncludesNuma)
        {
            CPUBackend backend(0);
            std::string name = backend.deviceName(0);
            EXPECT_THAT(name, testing::HasSubstr("CPU:NUMA"));
            EXPECT_THAT(name, testing::HasSubstr("0"));
        }

        TEST(Test__CPUBackend, DeviceNameShowsNumaNode)
        {
            CPUBackend backend0(0);
            CPUBackend backend1(1);

            EXPECT_EQ(backend0.deviceName(0), "CPU:NUMA0");
            EXPECT_EQ(backend1.deviceName(0), "CPU:NUMA1");
        }

        TEST(Test__CPUBackend, DeviceNameShowsAllForNoNuma)
        {
            CPUBackend backend(-1);
            EXPECT_EQ(backend.deviceName(0), "CPU:NUMAALL");
        }

        TEST(Test__CPUBackend, DeviceNameInvalidDeviceId)
        {
            CPUBackend backend(0);
            EXPECT_EQ(backend.deviceName(1), "Invalid Device");
            EXPECT_EQ(backend.deviceName(-1), "Invalid Device");
        }

        // ============================================================================
        // Memory Query Tests
        // ============================================================================

        TEST(Test__CPUBackend, MemoryTotalIsPositive)
        {
            CPUBackend backend(0);
            size_t total = backend.deviceMemoryTotal(0);
            EXPECT_GT(total, 0);
        }

        TEST(Test__CPUBackend, MemoryFreeIsPositive)
        {
            CPUBackend backend(0);
            size_t free = backend.deviceMemoryFree(0);
            EXPECT_GT(free, 0);
        }

        TEST(Test__CPUBackend, MemoryFreeIsLessThanOrEqualToTotal)
        {
            CPUBackend backend(0);
            size_t total = backend.deviceMemoryTotal(0);
            size_t free = backend.deviceMemoryFree(0);
            EXPECT_LE(free, total);
        }

        TEST(Test__CPUBackend, MemoryTotalInvalidDeviceId)
        {
            CPUBackend backend(0);
            EXPECT_EQ(backend.deviceMemoryTotal(1), 0);
            EXPECT_EQ(backend.deviceMemoryTotal(-1), 0);
        }

        TEST(Test__CPUBackend, MemoryFreeInvalidDeviceId)
        {
            CPUBackend backend(0);
            EXPECT_EQ(backend.deviceMemoryFree(1), 0);
            EXPECT_EQ(backend.deviceMemoryFree(-1), 0);
        }

        TEST(Test__CPUBackend, MemoryReasonableSize)
        {
            CPUBackend backend(-1); // System-wide
            size_t total = backend.deviceMemoryTotal(0);

            // Should be at least 100 MB and less than 100 TB
            EXPECT_GT(total, 100ULL * 1024 * 1024);               // > 100 MB
            EXPECT_LT(total, 100ULL * 1024 * 1024 * 1024 * 1024); // < 100 TB
        }

        // ============================================================================
        // Memory Allocation Tests
        // ============================================================================

        TEST(Test__CPUBackend, AllocateAndFree)
        {
            CPUBackend backend(0);
            void *ptr = backend.allocate(4096, 0);
            ASSERT_NE(ptr, nullptr);

            // Write to memory to verify it's usable
            std::memset(ptr, 0xAB, 4096);

            backend.free(ptr, 0);
        }

        TEST(Test__CPUBackend, AllocateZeroBytes)
        {
            CPUBackend backend(0);
            void *ptr = backend.allocate(0, 0);
            EXPECT_EQ(ptr, nullptr);
        }

        TEST(Test__CPUBackend, AllocateLargeBuffer)
        {
            CPUBackend backend(0);

            // Allocate 10 MB
            const size_t size = 10 * 1024 * 1024;
            void *ptr = backend.allocate(size, 0);
            ASSERT_NE(ptr, nullptr);

            // Write pattern to verify it's usable
            char *data = static_cast<char *>(ptr);
            for (size_t i = 0; i < size; i += 4096)
            {
                data[i] = static_cast<char>(i & 0xFF);
            }

            backend.free(ptr, 0);
        }

        TEST(Test__CPUBackend, AllocateInvalidDeviceId)
        {
            CPUBackend backend(0);
            EXPECT_EQ(backend.allocate(4096, 1), nullptr);
            EXPECT_EQ(backend.allocate(4096, -1), nullptr);
        }

        TEST(Test__CPUBackend, FreeNullptrIsNoop)
        {
            CPUBackend backend(0);
            backend.free(nullptr, 0); // Should not crash
        }

        TEST(Test__CPUBackend, AllocateAlignment)
        {
            CPUBackend backend(0);
            void *ptr = backend.allocate(4096, 0);
            ASSERT_NE(ptr, nullptr);

            // Should be 64-byte aligned for AVX-512
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            EXPECT_EQ(addr % 64, 0);

            backend.free(ptr, 0);
        }

        // ============================================================================
        // Memset Tests
        // ============================================================================

        TEST(Test__CPUBackend, MemsetWorks)
        {
            CPUBackend backend(0);
            void *ptr = backend.allocate(256, 0);
            ASSERT_NE(ptr, nullptr);

            EXPECT_TRUE(backend.memset(ptr, 0x42, 256, 0));

            char *data = static_cast<char *>(ptr);
            for (int i = 0; i < 256; ++i)
            {
                EXPECT_EQ(static_cast<unsigned char>(data[i]), 0x42);
            }

            backend.free(ptr, 0);
        }

        TEST(Test__CPUBackend, MemsetZeroBytes)
        {
            CPUBackend backend(0);
            void *ptr = backend.allocate(256, 0);
            ASSERT_NE(ptr, nullptr);

            EXPECT_TRUE(backend.memset(ptr, 0x42, 0, 0));

            backend.free(ptr, 0);
        }

        TEST(Test__CPUBackend, MemsetNullptr)
        {
            CPUBackend backend(0);
            EXPECT_TRUE(backend.memset(nullptr, 0x42, 0, 0));
        }

        TEST(Test__CPUBackend, MemsetInvalidDeviceId)
        {
            CPUBackend backend(0);
            void *ptr = backend.allocate(256, 0);
            ASSERT_NE(ptr, nullptr);

            EXPECT_FALSE(backend.memset(ptr, 0x42, 256, 1));

            backend.free(ptr, 0);
        }

        // ============================================================================
        // Transfer Operation Tests
        // ============================================================================

        TEST(Test__CPUBackend, HostToDeviceMemcpy)
        {
            CPUBackend backend(0);

            char src[256], dst[256];
            std::memset(src, 0x42, sizeof(src));
            std::memset(dst, 0, sizeof(dst));

            EXPECT_TRUE(backend.hostToDevice(dst, src, sizeof(src), 0));
            EXPECT_EQ(std::memcmp(src, dst, sizeof(src)), 0);
        }

        TEST(Test__CPUBackend, DeviceToHostMemcpy)
        {
            CPUBackend backend(0);

            char src[256], dst[256];
            std::memset(src, 0x42, sizeof(src));
            std::memset(dst, 0, sizeof(dst));

            EXPECT_TRUE(backend.deviceToHost(dst, src, sizeof(src), 0));
            EXPECT_EQ(std::memcmp(src, dst, sizeof(src)), 0);
        }

        TEST(Test__CPUBackend, MemcpyZeroBytes)
        {
            CPUBackend backend(0);

            char src[256], dst[256];
            std::memset(src, 0x42, sizeof(src));
            std::memset(dst, 0, sizeof(dst));

            EXPECT_TRUE(backend.hostToDevice(dst, src, 0, 0));
            EXPECT_TRUE(backend.deviceToHost(dst, src, 0, 0));

            // dst should still be zeros
            EXPECT_EQ(dst[0], 0);
        }

        TEST(Test__CPUBackend, MemcpyInvalidDeviceId)
        {
            CPUBackend backend(0);

            char src[256], dst[256];
            EXPECT_FALSE(backend.hostToDevice(dst, src, sizeof(src), 1));
            EXPECT_FALSE(backend.deviceToHost(dst, src, sizeof(src), 1));
        }

        // ============================================================================
        // Synchronize Tests
        // ============================================================================

        TEST(Test__CPUBackend, SynchronizeIsNoop)
        {
            CPUBackend backend(0);
            EXPECT_TRUE(backend.synchronize(0));
        }

        TEST(Test__CPUBackend, SynchronizeInvalidDeviceId)
        {
            CPUBackend backend(0);
            EXPECT_FALSE(backend.synchronize(1));
            EXPECT_FALSE(backend.synchronize(-1));
        }

        // ============================================================================
        // SetDevice Tests
        // ============================================================================

        TEST(Test__CPUBackend, SetDeviceZeroSucceeds)
        {
            CPUBackend backend(0);
            EXPECT_TRUE(backend.setDevice(0));
        }

        TEST(Test__CPUBackend, InvalidDeviceIdFails)
        {
            CPUBackend backend(0);
            EXPECT_FALSE(backend.setDevice(1)); // Only device 0 exists
            EXPECT_FALSE(backend.setDevice(-1));
            EXPECT_FALSE(backend.setDevice(100));
        }

        // ============================================================================
        // Capability Query Tests
        // ============================================================================

        TEST(Test__CPUBackend, SupportsBF16)
        {
            CPUBackend backend(0);
            EXPECT_TRUE(backend.supportsBF16(0));
        }

        TEST(Test__CPUBackend, SupportsFP16)
        {
            CPUBackend backend(0);
            EXPECT_TRUE(backend.supportsFP16(0));
        }

        TEST(Test__CPUBackend, SupportsINT8)
        {
            CPUBackend backend(0);
            EXPECT_TRUE(backend.supportsINT8(0));
        }

        TEST(Test__CPUBackend, CapabilityInvalidDeviceId)
        {
            CPUBackend backend(0);
            EXPECT_FALSE(backend.supportsBF16(1));
            EXPECT_FALSE(backend.supportsFP16(1));
            EXPECT_FALSE(backend.supportsINT8(1));
        }

        // ============================================================================
        // GEMM Tests
        // ============================================================================

        TEST(Test__CPUBackend, GemmIQ4NLNotImplemented)
        {
            CPUBackend backend(0);
            // gemmIQ4NL should return false (use CPU kernel directly)
            EXPECT_FALSE(backend.gemmIQ4NL(nullptr, nullptr, nullptr, 0, 0, 0, 0));
        }

        // ============================================================================
        // NUMA Node Tests
        // ============================================================================

        TEST(Test__CPUBackend, NumaNodeAccessor)
        {
            CPUBackend backend0(0);
            CPUBackend backend1(1);
            CPUBackend backendNeg(-1);

            EXPECT_EQ(backend0.numaNode(), 0);
            EXPECT_EQ(backend1.numaNode(), 1);
            EXPECT_EQ(backendNeg.numaNode(), -1);
        }

        // ============================================================================
        // BackendManager Integration Tests
        // ============================================================================

        TEST(Test__CPUBackend, BackendManagerGetCPUBackend)
        {
            // Initialize CPU backend
            initCPUBackend(0);

            IBackend *backend = getCPUBackend();
            ASSERT_NE(backend, nullptr);
            EXPECT_EQ(backend->backendName(), "CPU");
            EXPECT_EQ(backend->deviceCount(), 1);
        }

        TEST(Test__CPUBackend, BackendManagerHasCPUBackend)
        {
            initCPUBackend(0);
            EXPECT_TRUE(hasCPUBackend());
        }

        TEST(Test__CPUBackend, BackendManagerGetBackendForCPU)
        {
            initCPUBackend(0);

            IBackend *backend = getBackendFor(DeviceId::cpu());
            ASSERT_NE(backend, nullptr);
            EXPECT_EQ(backend->backendName(), "CPU");
        }

        TEST(Test__CPUBackend, BackendManagerGetBackendForInvalid)
        {
            // Test that invalid DeviceId returns nullptr or the CPU backend falls back
            DeviceId invalid = DeviceId::invalid();

            // Invalid ordinal should still be handled gracefully
            // The behavior depends on whether CPU is initialized
            IBackend *backend = getBackendFor(invalid);
            // For invalid devices, we still return CPU backend (type is CPU, ordinal is -1)
            // The backend itself will reject operations on invalid ordinals
        }

        // ============================================================================
        // Integration Test: Full Allocation Workflow
        // ============================================================================

        TEST(Test__CPUBackend, FullAllocationWorkflow)
        {
            CPUBackend backend(0);

            // 1. Check memory available
            size_t total = backend.deviceMemoryTotal(0);
            size_t free = backend.deviceMemoryFree(0);
            ASSERT_GT(total, 0);
            ASSERT_GT(free, 0);

            // 2. Allocate memory
            const size_t alloc_size = 1024 * 1024; // 1 MB
            void *ptr = backend.allocate(alloc_size, 0);
            ASSERT_NE(ptr, nullptr);

            // 3. Set device (no-op for CPU)
            EXPECT_TRUE(backend.setDevice(0));

            // 4. Initialize with memset
            EXPECT_TRUE(backend.memset(ptr, 0, alloc_size, 0));

            // 5. Host to device (memcpy)
            std::vector<char> src_data(alloc_size, 0xAB);
            EXPECT_TRUE(backend.hostToDevice(ptr, src_data.data(), alloc_size, 0));

            // 6. Synchronize (no-op for CPU)
            EXPECT_TRUE(backend.synchronize(0));

            // 7. Device to host (memcpy)
            std::vector<char> dst_data(alloc_size, 0);
            EXPECT_TRUE(backend.deviceToHost(dst_data.data(), ptr, alloc_size, 0));

            // 8. Verify data
            EXPECT_EQ(std::memcmp(src_data.data(), dst_data.data(), alloc_size), 0);

            // 9. Free memory
            backend.free(ptr, 0);
        }

    } // namespace test
} // namespace llaminar2
