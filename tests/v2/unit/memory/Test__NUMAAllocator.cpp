/**
 * @file Test__NUMAAllocator.cpp
 * @brief Unit tests for NUMAAllocator class
 *
 * Tests NUMA-aware memory allocation including:
 * - Singleton access
 * - Local and node-specific allocation
 * - First-touch initialization
 * - Array allocation with RAII
 * - NUMABuffer wrapper
 * - Statistics tracking
 *
 * @author David Sanftenberg
 * @date 2026-01-21
 */

#include <gtest/gtest.h>
#include "memory/NUMAAllocator.h"

#include <cstring>
#include <thread>
#include <vector>
#include <numeric>

using namespace llaminar2;

// ============================================================================
// Singleton Tests
// ============================================================================

TEST(Test__NUMAAllocator, SingletonExists)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();
    // Just verify we can access the singleton
    EXPECT_GE(allocator.numNUMANodes(), 1);
}

TEST(Test__NUMAAllocator, SingletonIsSame)
{
    NUMAAllocator &a1 = NUMAAllocator::instance();
    NUMAAllocator &a2 = NUMAAllocator::instance();
    EXPECT_EQ(&a1, &a2) << "Singleton should return same instance";
}

// ============================================================================
// Basic Allocation Tests
// ============================================================================

TEST(Test__NUMAAllocator, AllocateLocalReturnsNonNull)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    void *ptr = allocator.allocateLocal(1024);
    ASSERT_NE(ptr, nullptr) << "allocateLocal should return non-null for valid size";

    // Clean up
    allocator.free(ptr, 1024);
}

TEST(Test__NUMAAllocator, AllocateLocalZeroBytes)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    void *ptr = allocator.allocateLocal(0);
    EXPECT_EQ(ptr, nullptr) << "allocateLocal with 0 bytes should return nullptr";
}

TEST(Test__NUMAAllocator, AllocateOnNodeReturnsNonNull)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    // Allocate on node 0 (always valid)
    void *ptr = allocator.allocateOnNode(2048, 0);
    ASSERT_NE(ptr, nullptr) << "allocateOnNode(0) should return non-null";

    allocator.free(ptr, 2048);
}

TEST(Test__NUMAAllocator, AllocateOnNodeLocalReturnsNonNull)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    // -1 means allocate on local node
    void *ptr = allocator.allocateOnNode(4096, -1);
    ASSERT_NE(ptr, nullptr) << "allocateOnNode(-1) should return non-null (local allocation)";

    allocator.free(ptr, 4096);
}

TEST(Test__NUMAAllocator, AllocateOnNodeInvalidNode)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    void *ptr = allocator.allocateOnNode(1024, 999);
    EXPECT_EQ(ptr, nullptr) << "Invalid NUMA node must fail instead of falling back";
}

TEST(Test__NUMAAllocator, AllocateOnNodeNegativeInvalid)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    // -2 is invalid (only -1 is special for local)
    void *ptr = allocator.allocateOnNode(1024, -2);
    EXPECT_EQ(ptr, nullptr) << "Negative invalid node must fail instead of falling back";
}

// ============================================================================
// Free Tests
// ============================================================================

TEST(Test__NUMAAllocator, FreeDoesNotCrash)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    void *ptr = allocator.allocateLocal(8192);
    ASSERT_NE(ptr, nullptr);

    // This should not crash
    allocator.free(ptr, 8192);

    SUCCEED() << "free() completed without crash";
}

TEST(Test__NUMAAllocator, FreeNullDoesNotCrash)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    // Freeing nullptr should be safe
    allocator.free(nullptr, 0);
    allocator.free(nullptr, 1024);

    SUCCEED() << "free(nullptr) completed without crash";
}

// ============================================================================
// First-Touch Initialization Tests
// ============================================================================

TEST(Test__NUMAAllocator, AllocateAndTouchInitializesMemory)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    constexpr size_t size = 16384; // 4 pages
    constexpr uint8_t init_value = 0xAB;

    void *ptr = allocator.allocateAndTouch(size, 0, init_value);
    ASSERT_NE(ptr, nullptr);

    // Verify initialization
    uint8_t *data = static_cast<uint8_t *>(ptr);
    bool all_initialized = true;
    for (size_t i = 0; i < size && all_initialized; ++i)
    {
        if (data[i] != init_value)
        {
            all_initialized = false;
            FAIL() << "Byte at offset " << i << " is " << static_cast<int>(data[i])
                   << " instead of " << static_cast<int>(init_value);
        }
    }

    EXPECT_TRUE(all_initialized) << "All bytes should be initialized to init_value";

    allocator.free(ptr, size);
}

TEST(Test__NUMAAllocator, AllocateAndTouchZeroInit)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    constexpr size_t size = 8192;

    void *ptr = allocator.allocateAndTouch(size, 0, 0);
    ASSERT_NE(ptr, nullptr);

    // Verify zero initialization
    uint8_t *data = static_cast<uint8_t *>(ptr);
    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(data[i], 0) << "Byte at offset " << i << " should be 0";
    }

    allocator.free(ptr, size);
}

// ============================================================================
// Array Allocation Tests
// ============================================================================

TEST(Test__NUMAAllocator, AllocateArrayFloat)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    constexpr size_t count = 1024;
    auto arr = allocator.allocateArray<float>(count, 0);

    ASSERT_NE(arr.get(), nullptr) << "allocateArray<float> should succeed";

    // Verify we can write and read
    for (size_t i = 0; i < count; ++i)
    {
        arr[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(arr[i], static_cast<float>(i));
    }

    // unique_ptr automatically frees on scope exit
}

TEST(Test__NUMAAllocator, AllocateArrayDouble)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    constexpr size_t count = 512;
    auto arr = allocator.allocateArray<double>(count, -1); // local node

    ASSERT_NE(arr.get(), nullptr) << "allocateArray<double> should succeed";

    // Verify alignment (doubles need 8-byte alignment, we provide 64)
    uintptr_t addr = reinterpret_cast<uintptr_t>(arr.get());
    EXPECT_EQ(addr % 64, 0u) << "Array should be 64-byte aligned";

    // Verify usability
    arr[0] = 3.14159;
    arr[count - 1] = 2.71828;
    EXPECT_DOUBLE_EQ(arr[0], 3.14159);
    EXPECT_DOUBLE_EQ(arr[count - 1], 2.71828);
}

TEST(Test__NUMAAllocator, AllocateArrayZeroCount)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    auto arr = allocator.allocateArray<int>(0, 0);
    EXPECT_EQ(arr.get(), nullptr) << "allocateArray with count=0 should return null unique_ptr";
}

// ============================================================================
// Query Method Tests
// ============================================================================

TEST(Test__NUMAAllocator, GetCurrentNUMANode)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    int node = allocator.getCurrentNUMANode();
    EXPECT_GE(node, 0) << "Current NUMA node should be >= 0";
    EXPECT_LT(node, allocator.numNUMANodes()) << "Current NUMA node should be < numNUMANodes";
}

TEST(Test__NUMAAllocator, GetNUMANodeForAddress)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    // Allocate on node 0
    void *ptr = allocator.allocateOnNode(4096, 0);
    ASSERT_NE(ptr, nullptr);

    // Touch the memory to ensure it's mapped
    std::memset(ptr, 0, 4096);

    if (allocator.isNUMAAvailable())
    {
        int node = allocator.getNUMANodeForAddress(ptr);
        // Node should be valid (may not be 0 if system migrated it)
        EXPECT_GE(node, -1) << "getNUMANodeForAddress should return >= -1";
        if (node >= 0)
        {
            EXPECT_LT(node, allocator.numNUMANodes());
        }
    }
    else
    {
        // Without NUMA, should return -1 (cannot determine)
        int node = allocator.getNUMANodeForAddress(ptr);
        EXPECT_EQ(node, -1);
    }

    allocator.free(ptr, 4096);
}

TEST(Test__NUMAAllocator, GetNUMANodeForNullAddress)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    int node = allocator.getNUMANodeForAddress(nullptr);
    EXPECT_EQ(node, -1) << "getNUMANodeForAddress(nullptr) should return -1";
}

TEST(Test__NUMAAllocator, IsNUMAAvailable)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    // Just verify the method doesn't crash and returns a boolean
    bool available = allocator.isNUMAAvailable();
    SUCCEED() << "isNUMAAvailable() returned " << (available ? "true" : "false");
}

TEST(Test__NUMAAllocator, NumNUMANodesAtLeastOne)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    EXPECT_GE(allocator.numNUMANodes(), 1) << "Should have at least 1 NUMA node";
}

// ============================================================================
// NUMABuffer Tests
// ============================================================================

TEST(Test__NUMAAllocator, NUMABufferConstruction)
{
    NUMABuffer<float> buffer(1024, 0);

    EXPECT_TRUE(buffer.valid()) << "NUMABuffer should be valid after construction";
    EXPECT_EQ(buffer.size(), 1024u);
    EXPECT_NE(buffer.data(), nullptr);
}

TEST(Test__NUMAAllocator, NUMABufferDefaultConstruction)
{
    NUMABuffer<float> buffer;

    EXPECT_FALSE(buffer.valid()) << "Default-constructed NUMABuffer should be invalid";
    EXPECT_EQ(buffer.size(), 0u);
    EXPECT_EQ(buffer.data(), nullptr);
}

TEST(Test__NUMAAllocator, NUMABufferAccess)
{
    constexpr size_t count = 256;
    NUMABuffer<int> buffer(count, -1); // local NUMA node

    ASSERT_TRUE(buffer.valid());

    // Write via operator[]
    for (size_t i = 0; i < count; ++i)
    {
        buffer[i] = static_cast<int>(i * 2);
    }

    // Read via operator[]
    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(buffer[i], static_cast<int>(i * 2));
    }

    // Read via data()
    const int *raw = buffer.data();
    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(raw[i], static_cast<int>(i * 2));
    }
}

TEST(Test__NUMAAllocator, NUMABufferMove)
{
    NUMABuffer<double> buffer1(512, 0);
    ASSERT_TRUE(buffer1.valid());

    buffer1[0] = 42.0;

    // Move construct
    NUMABuffer<double> buffer2(std::move(buffer1));

    EXPECT_FALSE(buffer1.valid()) << "Moved-from buffer should be invalid";
    EXPECT_TRUE(buffer2.valid()) << "Moved-to buffer should be valid";
    EXPECT_EQ(buffer2.size(), 512u);
    EXPECT_DOUBLE_EQ(buffer2[0], 42.0);
}

TEST(Test__NUMAAllocator, NUMABufferMoveAssign)
{
    NUMABuffer<float> buffer1(256, 0);
    NUMABuffer<float> buffer2(128, 0);

    ASSERT_TRUE(buffer1.valid());
    ASSERT_TRUE(buffer2.valid());

    buffer1[0] = 1.0f;
    buffer2[0] = 2.0f;

    // Move assign
    buffer2 = std::move(buffer1);

    EXPECT_FALSE(buffer1.valid());
    EXPECT_TRUE(buffer2.valid());
    EXPECT_EQ(buffer2.size(), 256u);
    EXPECT_FLOAT_EQ(buffer2[0], 1.0f);
}

TEST(Test__NUMAAllocator, NUMABufferIterators)
{
    constexpr size_t count = 64;
    NUMABuffer<int> buffer(count, 0);

    // Fill using iterators
    int value = 0;
    for (auto &elem : buffer)
    {
        elem = value++;
    }

    // Verify using iterators
    value = 0;
    for (const auto &elem : buffer)
    {
        EXPECT_EQ(elem, value++);
    }

    // Verify begin/end
    EXPECT_EQ(buffer.end() - buffer.begin(), static_cast<ptrdiff_t>(count));
}

TEST(Test__NUMAAllocator, NUMABufferNUMANode)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    // Allocate on local node (-1)
    NUMABuffer<float> buffer(1024, -1);

    if (allocator.isNUMAAvailable())
    {
        // Should have resolved to actual node
        int node = buffer.numaNode();
        EXPECT_GE(node, 0);
        EXPECT_LT(node, allocator.numNUMANodes());
    }
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(Test__NUMAAllocator, GetNodeStats)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    // Get stats for node 0
    auto stats = allocator.getNodeStats(0);

    // allocated_by_us should be >= 0
    EXPECT_GE(stats.allocated_by_us, 0u);

    // If NUMA available, total_bytes should be > 0
    if (allocator.isNUMAAvailable())
    {
        EXPECT_GT(stats.total_bytes, 0u) << "NUMA node should have memory";
        EXPECT_LE(stats.free_bytes, stats.total_bytes);
    }
}

TEST(Test__NUMAAllocator, GetNodeStatsTracksAllocation)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    constexpr size_t alloc_size = 65536; // 64KB

    auto stats_before = allocator.getNodeStats(0);

    void *ptr = allocator.allocateOnNode(alloc_size, 0);
    ASSERT_NE(ptr, nullptr);

    auto stats_during = allocator.getNodeStats(0);

    allocator.free(ptr, alloc_size);

    auto stats_after = allocator.getNodeStats(0);

    // During allocation, our tracked amount should increase
    EXPECT_GE(stats_during.allocated_by_us, stats_before.allocated_by_us);

    // After free, should return close to before (accounting for concurrent allocations)
    // This is a weak check due to potential concurrent activity
    EXPECT_LE(stats_after.allocated_by_us, stats_during.allocated_by_us);
}

// ============================================================================
// Alignment Tests
// ============================================================================

TEST(Test__NUMAAllocator, AllocationIsAligned)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    // Test various alignments
    std::vector<size_t> alignments = {64, 128, 256, 512};

    for (size_t alignment : alignments)
    {
        void *ptr = allocator.allocateOnNode(4096, 0, alignment);
        ASSERT_NE(ptr, nullptr);

        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        EXPECT_EQ(addr % alignment, 0u)
            << "Allocation should be " << alignment << "-byte aligned";

        allocator.free(ptr, 4096);
    }
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST(Test__NUMAAllocator, ConcurrentAllocation)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    constexpr int num_threads = 4;
    constexpr int allocs_per_thread = 100;
    constexpr size_t alloc_size = 1024;

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&allocator, &success_count]()
                             {
            for (int i = 0; i < allocs_per_thread; ++i) {
                void* ptr = allocator.allocateLocal(alloc_size);
                if (ptr) {
                    // Touch memory
                    std::memset(ptr, 0, alloc_size);
                    allocator.free(ptr, alloc_size);
                    ++success_count;
                }
            } });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    EXPECT_EQ(success_count.load(), num_threads * allocs_per_thread)
        << "All concurrent allocations should succeed";
}

// ============================================================================
// Large Allocation Tests
// ============================================================================

TEST(Test__NUMAAllocator, LargeAllocation)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();

    // Allocate 100MB
    constexpr size_t size = 100 * 1024 * 1024;

    void *ptr = allocator.allocateOnNode(size, 0);

    // This might fail on memory-constrained systems, so allow nullptr
    if (ptr)
    {
        // Touch first and last page
        uint8_t *data = static_cast<uint8_t *>(ptr);
        data[0] = 0xAA;
        data[size - 1] = 0xBB;

        EXPECT_EQ(data[0], 0xAA);
        EXPECT_EQ(data[size - 1], 0xBB);

        allocator.free(ptr, size);
        SUCCEED() << "Large allocation (100MB) succeeded";
    }
    else
    {
        GTEST_SKIP() << "System doesn't have enough memory for 100MB allocation";
    }
}
