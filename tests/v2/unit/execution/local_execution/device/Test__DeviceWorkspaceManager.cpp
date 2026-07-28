/**
 * @file Test__DeviceWorkspaceManager.cpp
 * @brief Unit tests for DeviceWorkspaceManager
 *
 * Tests the GPU workspace buffer management system using CPU backend
 * (no actual GPU required for unit tests).
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <set>
#include <cstring>

#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "backends/BackendManager.h"
#include "kernels/common/DeviceResidentRouterGateCache.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

using namespace llaminar2;

namespace
{
    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old = std::getenv(name);
            if (old)
            {
                had_old_ = true;
                old_value_ = old;
            }
            if (value)
                setenv(name, value, 1);
            else
                unsetenv(name);
            mutableDebugEnv().reload();
        }

        ~ScopedEnv()
        {
            if (had_old_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

    private:
        std::string name_;
        bool had_old_ = false;
        std::string old_value_;
    };

    const PerfStatRecord *findMemoryCounter(
        const std::vector<PerfStatRecord> &records,
        const std::string &name,
        const std::string &tag_name = {},
        const std::string &tag_value = {})
    {
        auto it = std::find_if(records.begin(), records.end(), [&](const PerfStatRecord &record)
                               {
                                   if (record.kind != PerfStatRecord::Kind::Counter ||
                                       record.domain != "memory" ||
                                       record.name != name)
                                   {
                                       return false;
                                   }
                                   if (tag_name.empty())
                                       return true;
                                   auto tag_it = record.tags.find(tag_name);
                                   return tag_it != record.tags.end() && tag_it->second == tag_value;
                               });
        return it == records.end() ? nullptr : &(*it);
    }
}

/**
 * @brief Test fixture for DeviceWorkspaceManager tests
 *
 * Uses CPU device so tests run without GPU hardware.
 */
class Test__DeviceWorkspaceManager : public ::testing::Test
{
protected:
    // Use CPU device for unit tests (no GPU required)
    DeviceId device = DeviceId::cpu();
    size_t budget = 1024 * 1024; // 1MB

    void SetUp() override
    {
        // Ensure CPU backend is initialized
        if (!hasCPUBackend())
        {
            initCPUBackend(-1); // System-wide memory (no NUMA binding)
        }
    }
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(Test__DeviceWorkspaceManager, ConstructionSetsDevice)
{
    DeviceWorkspaceManager mgr(device, budget);
    EXPECT_EQ(mgr.device(), device);
    EXPECT_EQ(mgr.budget(), budget);
    EXPECT_FALSE(mgr.isAllocated());
    EXPECT_EQ(mgr.used(), 0);
    EXPECT_EQ(mgr.remaining(), budget);
    EXPECT_EQ(mgr.bufferCount(), 0);
}

TEST_F(Test__DeviceWorkspaceManager, ConstructionWithZeroBudget)
{
    DeviceWorkspaceManager mgr(device, 0);
    EXPECT_EQ(mgr.budget(), 0);
    EXPECT_EQ(mgr.remaining(), 0);
    EXPECT_FALSE(mgr.isAllocated());
}

TEST_F(Test__DeviceWorkspaceManager, ManagerIdsAreUniqueAcrossInstances)
{
    DeviceWorkspaceManager first(device, budget);
    DeviceWorkspaceManager second(device, budget);

    EXPECT_NE(first.id(), 0u);
    EXPECT_NE(second.id(), 0u);
    EXPECT_NE(first.id(), second.id())
        << "Kernel workspace scratch caches depend on manager identity, not just host pointer equality";
}

// ============================================================================
// Simple Allocation Tests
// ============================================================================

TEST_F(Test__DeviceWorkspaceManager, AllocateSimpleBuffer)
{
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"test_buffer", 4096, 256, true});

    ASSERT_TRUE(mgr.allocate(reqs));
    EXPECT_TRUE(mgr.isAllocated());
    EXPECT_TRUE(mgr.hasBuffer("test_buffer"));
    EXPECT_EQ(mgr.bufferCount(), 1);

    void *ptr = mgr.getBuffer("test_buffer");
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(mgr.getBufferSize("test_buffer"), 4096);

    // Verify buffer is usable by writing to it
    std::memset(ptr, 0xAB, 4096);
}

TEST_F(Test__DeviceWorkspaceManager, AllocateMultipleBuffers)
{
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"buffer_a", 1024, 256, true});
    reqs.buffers.push_back({"buffer_b", 2048, 256, true});
    reqs.buffers.push_back({"buffer_c", 4096, 256, true});

    ASSERT_TRUE(mgr.allocate(reqs));
    EXPECT_EQ(mgr.bufferCount(), 3);

    void *a = mgr.getBuffer("buffer_a");
    void *b = mgr.getBuffer("buffer_b");
    void *c = mgr.getBuffer("buffer_c");

    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);

    // Verify buffers don't overlap
    EXPECT_NE(a, b);
    EXPECT_NE(b, c);
    EXPECT_NE(a, c);

    // Verify sizes
    EXPECT_EQ(mgr.getBufferSize("buffer_a"), 1024);
    EXPECT_EQ(mgr.getBufferSize("buffer_b"), 2048);
    EXPECT_EQ(mgr.getBufferSize("buffer_c"), 4096);
}

TEST_F(Test__DeviceWorkspaceManager, EmitsStructuredMemoryCountersForWorkspaceLayout)
{
    ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"alpha_visible", 1024, 256, true});
    reqs.buffers.push_back({"beta_visible", 2048, 512, false});

    ASSERT_TRUE(mgr.allocate(reqs));
    mgr.release();

    const auto records = PerfStatsCollector::snapshot({"memory"});
    const auto *block = findMemoryCounter(records, "workspace_block_bytes");
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(block->device, device.to_string());
    EXPECT_GE(block->value, 1024.0 + 2048.0);
    EXPECT_EQ(block->tags.at("buffer_count"), "2");

    const auto *alpha = findMemoryCounter(records, "workspace_suballoc_bytes", "name", "alpha_visible");
    ASSERT_NE(alpha, nullptr);
    EXPECT_DOUBLE_EQ(alpha->value, 1024.0);
    EXPECT_EQ(alpha->tags.at("required"), "true");
    EXPECT_EQ(alpha->tags.at("alignment"), "256");

    const auto *beta = findMemoryCounter(records, "workspace_suballoc_bytes", "name", "beta_visible");
    ASSERT_NE(beta, nullptr);
    EXPECT_DOUBLE_EQ(beta->value, 2048.0);
    EXPECT_EQ(beta->tags.at("required"), "false");
    EXPECT_EQ(beta->tags.at("alignment"), "512");

    const auto *release = findMemoryCounter(records, "workspace_release_bytes");
    ASSERT_NE(release, nullptr);
    EXPECT_EQ(release->tags.at("buffer_count"), "2");
}

// ============================================================================
// Budget Tests
// ============================================================================

TEST_F(Test__DeviceWorkspaceManager, BudgetExceededFailsForRequired)
{
    size_t small_budget = 1024; // 1KB
    DeviceWorkspaceManager mgr(device, small_budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"too_big", 10000, 256, true}); // Required, too big

    EXPECT_FALSE(mgr.allocate(reqs));
    EXPECT_FALSE(mgr.isAllocated());
}

TEST_F(Test__DeviceWorkspaceManager, OptionalBufferSkippedWhenOverBudget)
{
    size_t small_budget = 1024; // 1KB
    DeviceWorkspaceManager mgr(device, small_budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"small_required", 256, 256, true});
    reqs.buffers.push_back({"big_optional", 10000, 256, false}); // Optional, too big

    ASSERT_TRUE(mgr.allocate(reqs)); // Should succeed (optional skipped)
    EXPECT_TRUE(mgr.isAllocated());
    EXPECT_TRUE(mgr.hasBuffer("small_required"));
    EXPECT_FALSE(mgr.hasBuffer("big_optional")); // Skipped
    EXPECT_EQ(mgr.bufferCount(), 1);
}

TEST_F(Test__DeviceWorkspaceManager, ZeroBudgetRejectsAllocation)
{
    DeviceWorkspaceManager mgr(device, 0); // Zero budget

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"tiny", 1, 1, true});

    EXPECT_FALSE(mgr.allocate(reqs));
}

TEST_F(Test__DeviceWorkspaceManager, UsedBytesTracked)
{
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"buf1", 1024, 256, true});
    reqs.buffers.push_back({"buf2", 2048, 256, true});

    ASSERT_TRUE(mgr.allocate(reqs));

    // Used should be >= sum of buffers (may be more due to alignment)
    EXPECT_GE(mgr.used(), 1024 + 2048);
    EXPECT_LE(mgr.used(), mgr.budget());
    EXPECT_GT(mgr.remaining(), 0);
}

// ============================================================================
// Alignment Tests
// ============================================================================

TEST_F(Test__DeviceWorkspaceManager, AlignmentRespected)
{
    DeviceWorkspaceManager mgr(device, budget);

    // Note: CPUBackend allocates with 64-byte base alignment, so we test
    // alignments up to 64 bytes. GPU backends provide higher alignment.
    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"aligned_16", 100, 16, true});
    reqs.buffers.push_back({"aligned_64", 100, 64, true});

    ASSERT_TRUE(mgr.allocate(reqs));

    void *ptr16 = mgr.getBuffer("aligned_16");
    void *ptr64 = mgr.getBuffer("aligned_64");

    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr16) % 16, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr64) % 64, 0);
}

TEST_F(Test__DeviceWorkspaceManager, DifferentAlignments)
{
    DeviceWorkspaceManager mgr(device, budget);

    // Note: CPUBackend allocates with 64-byte base alignment, so we can only
    // guarantee alignments up to 64 bytes in suballocated buffers.
    // GPU backends (CUDA/ROCm) typically provide much higher base alignment.
    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"align_1", 17, 1, true});    // Minimal alignment
    reqs.buffers.push_back({"align_16", 33, 16, true});  // 16-byte alignment
    reqs.buffers.push_back({"align_64", 100, 64, true}); // 64-byte alignment (max for CPU)

    ASSERT_TRUE(mgr.allocate(reqs));

    void *ptr1 = mgr.getBuffer("align_1");
    void *ptr16 = mgr.getBuffer("align_16");
    void *ptr64 = mgr.getBuffer("align_64");

    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr16, nullptr);
    ASSERT_NE(ptr64, nullptr);

    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr16) % 16, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr64) % 64, 0);
}

// ============================================================================
// Release Tests
// ============================================================================

TEST_F(Test__DeviceWorkspaceManager, ReleaseFreesMemory)
{
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"test", 4096, 256, true});

    ASSERT_TRUE(mgr.allocate(reqs));
    EXPECT_TRUE(mgr.isAllocated());

    mgr.release();

    EXPECT_FALSE(mgr.isAllocated());
    EXPECT_EQ(mgr.bufferCount(), 0);
    EXPECT_EQ(mgr.getBuffer("test"), nullptr);
    EXPECT_EQ(mgr.used(), 0);
}

TEST_F(Test__DeviceWorkspaceManager, DoubleReleaseIsSafe)
{
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"test", 4096, 256, true});

    ASSERT_TRUE(mgr.allocate(reqs));

    mgr.release();
    mgr.release(); // Should not crash
    mgr.release(); // Should not crash

    EXPECT_FALSE(mgr.isAllocated());
}

TEST_F(Test__DeviceWorkspaceManager, ReallocateAfterRelease)
{
    DeviceWorkspaceManager mgr(device, budget);

    // First allocation
    WorkspaceRequirements reqs1;
    reqs1.buffers.push_back({"first", 1024, 256, true});
    ASSERT_TRUE(mgr.allocate(reqs1));
    EXPECT_TRUE(mgr.hasBuffer("first"));

    mgr.release();

    // Second allocation with different buffers
    WorkspaceRequirements reqs2;
    reqs2.buffers.push_back({"second_a", 2048, 256, true});
    reqs2.buffers.push_back({"second_b", 4096, 256, true});
    ASSERT_TRUE(mgr.allocate(reqs2));

    EXPECT_FALSE(mgr.hasBuffer("first")); // Old buffer gone
    EXPECT_TRUE(mgr.hasBuffer("second_a"));
    EXPECT_TRUE(mgr.hasBuffer("second_b"));
    EXPECT_EQ(mgr.bufferCount(), 2);
}

// ============================================================================
// Buffer Access Tests
// ============================================================================

TEST_F(Test__DeviceWorkspaceManager, GetNonexistentBufferReturnsNull)
{
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"exists", 1024, 256, true});

    ASSERT_TRUE(mgr.allocate(reqs));

    EXPECT_EQ(mgr.getBuffer("nonexistent"), nullptr);
    EXPECT_EQ(mgr.getBufferSize("nonexistent"), 0);
    EXPECT_FALSE(mgr.hasBuffer("nonexistent"));
}

TEST_F(Test__DeviceWorkspaceManager, BufferNamesReturned)
{
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"alpha", 1024, 256, true});
    reqs.buffers.push_back({"beta", 1024, 256, true});
    reqs.buffers.push_back({"gamma", 1024, 256, true});

    ASSERT_TRUE(mgr.allocate(reqs));

    auto names = mgr.bufferNames();
    EXPECT_EQ(names.size(), 3);

    // Check all names present (order may vary)
    std::set<std::string> name_set(names.begin(), names.end());
    EXPECT_TRUE(name_set.count("alpha"));
    EXPECT_TRUE(name_set.count("beta"));
    EXPECT_TRUE(name_set.count("gamma"));
}

TEST_F(Test__DeviceWorkspaceManager, BufferNamesEmptyWhenNotAllocated)
{
    DeviceWorkspaceManager mgr(device, budget);
    auto names = mgr.bufferNames();
    EXPECT_TRUE(names.empty());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(Test__DeviceWorkspaceManager, EmptyRequirementsSucceeds)
{
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs; // Empty

    EXPECT_TRUE(mgr.allocate(reqs));
    EXPECT_TRUE(mgr.isAllocated());
    EXPECT_EQ(mgr.bufferCount(), 0);
    EXPECT_EQ(mgr.used(), 0);
}

TEST_F(Test__DeviceWorkspaceManager, SingleByteBuffer)
{
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"tiny", 1, 1, true});

    ASSERT_TRUE(mgr.allocate(reqs));
    EXPECT_TRUE(mgr.hasBuffer("tiny"));
    EXPECT_EQ(mgr.getBufferSize("tiny"), 1);

    // Should be writable
    char *ptr = static_cast<char *>(mgr.getBuffer("tiny"));
    ASSERT_NE(ptr, nullptr);
    *ptr = 'X';
}

TEST_F(Test__DeviceWorkspaceManager, LargeBuffer)
{
    size_t large_budget = 100 * 1024 * 1024; // 100MB
    DeviceWorkspaceManager mgr(device, large_budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"large", 64 * 1024 * 1024, 256, true}); // 64MB

    ASSERT_TRUE(mgr.allocate(reqs));
    EXPECT_TRUE(mgr.hasBuffer("large"));
    EXPECT_EQ(mgr.getBufferSize("large"), 64 * 1024 * 1024);

    void *ptr = mgr.getBuffer("large");
    ASSERT_NE(ptr, nullptr);
}

TEST_F(Test__DeviceWorkspaceManager, ExactBudgetAllocation)
{
    size_t exact_budget = 4096;
    DeviceWorkspaceManager mgr(device, exact_budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"exact", 4096, 1, true}); // Exactly budget size

    ASSERT_TRUE(mgr.allocate(reqs));
    EXPECT_TRUE(mgr.hasBuffer("exact"));
    EXPECT_EQ(mgr.used(), 4096);
    EXPECT_EQ(mgr.remaining(), 0);
}

TEST_F(Test__DeviceWorkspaceManager, OneBytesOverBudget)
{
    size_t exact_budget = 4096;
    DeviceWorkspaceManager mgr(device, exact_budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"over", 4097, 1, true}); // One byte over

    EXPECT_FALSE(mgr.allocate(reqs));
}

TEST_F(Test__DeviceWorkspaceManager, AllOptionalBuffers)
{
    size_t small_budget = 100;
    DeviceWorkspaceManager mgr(device, small_budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"opt1", 1000, 1, false}); // Optional, too big
    reqs.buffers.push_back({"opt2", 2000, 1, false}); // Optional, too big

    // Should succeed with no buffers allocated
    ASSERT_TRUE(mgr.allocate(reqs));
    EXPECT_TRUE(mgr.isAllocated());
    EXPECT_EQ(mgr.bufferCount(), 0);
    EXPECT_FALSE(mgr.hasBuffer("opt1"));
    EXPECT_FALSE(mgr.hasBuffer("opt2"));
}

TEST_F(Test__DeviceWorkspaceManager, MixedRequiredOptional)
{
    size_t budget = 2048;
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"req1", 512, 256, true});    // Required, fits
    reqs.buffers.push_back({"opt1", 512, 256, false});   // Optional, fits
    reqs.buffers.push_back({"req2", 256, 256, true});    // Required, fits
    reqs.buffers.push_back({"opt2", 10000, 256, false}); // Optional, too big

    ASSERT_TRUE(mgr.allocate(reqs));
    EXPECT_TRUE(mgr.hasBuffer("req1"));
    EXPECT_TRUE(mgr.hasBuffer("opt1"));
    EXPECT_TRUE(mgr.hasBuffer("req2"));
    EXPECT_FALSE(mgr.hasBuffer("opt2")); // Skipped
    EXPECT_EQ(mgr.bufferCount(), 3);
}

TEST_F(Test__DeviceWorkspaceManager, DoubleAllocateWithoutReleaseFails)
{
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"first", 1024, 256, true});

    ASSERT_TRUE(mgr.allocate(reqs));

    // Second allocate without release should fail
    WorkspaceRequirements reqs2;
    reqs2.buffers.push_back({"second", 2048, 256, true});

    EXPECT_FALSE(mgr.allocate(reqs2));

    // Original allocation should still be intact
    EXPECT_TRUE(mgr.hasBuffer("first"));
    EXPECT_FALSE(mgr.hasBuffer("second"));
}

/**
 * @brief Append-only growth preserves every address a captured graph can own.
 */
TEST_F(Test__DeviceWorkspaceManager, ExtendRetainsOldStorageAndStableNames)
{
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements initial;
    initial.buffers.push_back({"growing", 1024, 64, true});
    initial.buffers.push_back({"stable", 1024, 64, true});
    ASSERT_TRUE(mgr.allocate(initial));

    auto *old_growing =
        static_cast<unsigned char *>(mgr.getBuffer("growing"));
    auto *old_stable =
        static_cast<unsigned char *>(mgr.getBuffer("stable"));
    ASSERT_NE(old_growing, nullptr);
    ASSERT_NE(old_stable, nullptr);
    old_growing[0] = 0x5a;
    old_stable[0] = 0xa5;

    WorkspaceRequirements extension;
    extension.buffers.push_back({"growing", 4096, 64, true});
    extension.buffers.push_back({"stable", 1024, 64, true});
    extension.buffers.push_back({"new_name", 2048, 64, true});
    ASSERT_TRUE(mgr.extend(extension));

    EXPECT_NE(mgr.getBuffer("growing"), old_growing);
    EXPECT_EQ(mgr.getBufferSize("growing"), 4096u);
    EXPECT_EQ(mgr.getBuffer("stable"), old_stable);
    EXPECT_EQ(mgr.getBufferSize("stable"), 1024u);
    EXPECT_NE(mgr.getBuffer("new_name"), nullptr);
    EXPECT_EQ(old_growing[0], 0x5a)
        << "Superseded storage must remain live for captured graph executables";
    EXPECT_EQ(old_stable[0], 0xa5);
}

TEST_F(Test__DeviceWorkspaceManager, ExtendFailsClosedWhenRequiredGrowthExceedsBudget)
{
    DeviceWorkspaceManager mgr(device, 4096);

    WorkspaceRequirements initial;
    initial.buffers.push_back({"initial", 3072, 64, true});
    ASSERT_TRUE(mgr.allocate(initial));
    void *initial_address = mgr.getBuffer("initial");

    WorkspaceRequirements extension;
    extension.buffers.push_back({"required_growth", 2048, 64, true});
    EXPECT_FALSE(mgr.extend(extension));
    EXPECT_EQ(mgr.getBuffer("initial"), initial_address);
    EXPECT_FALSE(mgr.hasBuffer("required_growth"));
}

// ============================================================================
// Buffer Content Tests
// ============================================================================

TEST_F(Test__DeviceWorkspaceManager, BuffersAreIndependent)
{
    DeviceWorkspaceManager mgr(device, budget);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"buf_a", 256, 256, true});
    reqs.buffers.push_back({"buf_b", 256, 256, true});

    ASSERT_TRUE(mgr.allocate(reqs));

    char *a = static_cast<char *>(mgr.getBuffer("buf_a"));
    char *b = static_cast<char *>(mgr.getBuffer("buf_b"));

    // Fill buffers with different patterns
    std::memset(a, 0xAA, 256);
    std::memset(b, 0xBB, 256);

    // Verify patterns are independent
    for (int i = 0; i < 256; i++)
    {
        EXPECT_EQ(static_cast<unsigned char>(a[i]), 0xAA);
        EXPECT_EQ(static_cast<unsigned char>(b[i]), 0xBB);
    }
}

// ============================================================================
// Persistent Graph Metadata Slot Tests
// ============================================================================

TEST_F(Test__DeviceWorkspaceManager, PersistentMetadataLeasesAreExclusiveAndReusable)
{
    DeviceWorkspaceManager mgr(device, budget);

    auto first = mgr.acquirePersistentSlot("moe_gateup_descriptors", 2);
    auto second = mgr.acquirePersistentSlot("moe_gateup_descriptors", 2);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->slot(), 0u);
    EXPECT_EQ(second->slot(), 1u);
    EXPECT_EQ(
        mgr.acquirePersistentSlot("moe_gateup_descriptors", 2),
        nullptr)
        << "A full graph-metadata domain must fail closed instead of aliasing "
           "an address already captured by another kernel";

    first.reset();
    auto replacement = mgr.acquirePersistentSlot("moe_gateup_descriptors", 2);
    ASSERT_NE(replacement, nullptr);
    EXPECT_EQ(replacement->slot(), 0u)
        << "Destroying an owner must return its immutable metadata slot";

    auto independent_domain =
        mgr.acquirePersistentSlot("moe_down_descriptors", 2);
    ASSERT_NE(independent_domain, nullptr);
    EXPECT_EQ(independent_domain->slot(), 0u)
        << "Separate physical descriptor tables have separate ownership domains";
}

TEST_F(Test__DeviceWorkspaceManager, PersistentSlotAddressesUseTheDeclaredTableStride)
{
    DeviceWorkspaceManager mgr(device, budget);
    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"persistent_router_table", 1024, 256, true});
    ASSERT_TRUE(mgr.allocate(reqs));

    auto first = mgr.acquirePersistentSlot("router_gate", 2);
    auto second = mgr.acquirePersistentSlot("router_gate", 2);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    void *small_payload = mgr.getPersistentSlotBuffer(
        "persistent_router_table",
        /*slot_capacity=*/2,
        first->slot(),
        /*payload_bytes=*/64);
    void *large_payload = mgr.getPersistentSlotBuffer(
        "persistent_router_table",
        /*slot_capacity=*/2,
        second->slot(),
        /*payload_bytes=*/384);
    ASSERT_NE(small_payload, nullptr);
    ASSERT_NE(large_payload, nullptr);

    const auto byte_distance =
        static_cast<char *>(large_payload) -
        static_cast<char *>(small_payload);
    EXPECT_EQ(byte_distance, 512)
        << "Different payload sizes must not change the common slot stride";
    EXPECT_EQ(
        mgr.getPersistentSlotBuffer(
            "persistent_router_table",
            /*slot_capacity=*/2,
            second->slot(),
            /*payload_bytes=*/513),
        nullptr)
        << "A payload wider than its fixed slot must fail before publication";
}

TEST_F(Test__DeviceWorkspaceManager, RouterGateIdentityIgnoresRequestLocalTensorWrappers)
{
    constexpr std::uint64_t kWorkspaceGeneration = 37;
    const void *const device_weight =
        reinterpret_cast<const void *>(std::uintptr_t{0x100000});
    const auto expected = DeviceResidentRouterGateCacheKey::make(
        kWorkspaceGeneration,
        device_weight,
        /*model_width=*/2048,
        /*expert_count=*/256);

    /*
     * Rebuilt graph views may have arbitrary host addresses. Simulate many
     * requests and prove none of those wrapper identities can alter the
     * prepared device-weight key.
     */
    for (int request = 0; request < 256; ++request)
    {
        int request_local_tensor_wrapper = request;
        (void)request_local_tensor_wrapper;
        EXPECT_EQ(
            DeviceResidentRouterGateCacheKey::make(
                kWorkspaceGeneration,
                device_weight,
                /*model_width=*/2048,
                /*expert_count=*/256),
            expected);
    }

    EXPECT_NE(
        DeviceResidentRouterGateCacheKey::make(
            kWorkspaceGeneration + 1,
            device_weight,
            /*model_width=*/2048,
            /*expert_count=*/256),
        expected);
    EXPECT_NE(
        DeviceResidentRouterGateCacheKey::make(
            kWorkspaceGeneration,
            reinterpret_cast<const void *>(std::uintptr_t{0x200000}),
            /*model_width=*/2048,
            /*expert_count=*/256),
        expected);
    EXPECT_NE(
        DeviceResidentRouterGateCacheKey::make(
            kWorkspaceGeneration,
            device_weight,
            /*model_width=*/4096,
            /*expert_count=*/256),
        expected);
}

TEST_F(Test__DeviceWorkspaceManager, ImmutablePublicationIsSharedAcrossGraphLocalOwners)
{
    DeviceWorkspaceManager mgr(device, budget);
    const PersistentWorkspacePublicationKey router_gate{
        .word0 = mgr.id(),
        .word1 = 0x100000,
        .word2 = 2048,
        .word3 = 256,
    };
    int factory_calls = 0;

    /*
     * A long-context server run constructs many graph-local routed-pipeline
     * kernel owners. All of them consume the same immutable router conversion;
     * ownership must therefore converge on one workspace publication rather
     * than exhausting one slot per graph object.
     */
    std::shared_ptr<void> expected_publication;
    for (int graph_owner = 0; graph_owner < 256; ++graph_owner)
    {
        const auto result = mgr.getOrCreatePersistentPublication(
            "rocm_moe_router_q8_gate_cache",
            router_gate,
            /*slot_capacity=*/2,
            [&](size_t slot) -> std::shared_ptr<void>
            {
                ++factory_calls;
                EXPECT_EQ(slot, 0u);
                return std::make_shared<int>(17);
            });
        ASSERT_TRUE(result);
        EXPECT_EQ(result.slot, 0u);
        EXPECT_EQ(result.created, graph_owner == 0);
        if (!expected_publication)
            expected_publication = result.publication;
        EXPECT_EQ(result.publication, expected_publication);
    }
    EXPECT_EQ(factory_calls, 1)
        << "The immutable device bytes must be published once, not once per graph-local kernel";

    const auto second = mgr.getOrCreatePersistentPublication(
        "rocm_moe_router_q8_gate_cache",
        PersistentWorkspacePublicationKey{
            .word0 = mgr.id(),
            .word1 = 0x200000,
            .word2 = 2048,
            .word3 = 256,
        },
        /*slot_capacity=*/2,
        [](size_t slot) -> std::shared_ptr<void>
        {
            EXPECT_EQ(slot, 1u);
            return std::make_shared<int>(23);
        });
    ASSERT_TRUE(second);
    EXPECT_TRUE(second.created);
    EXPECT_EQ(second.slot, 1u);

    EXPECT_FALSE(mgr.getOrCreatePersistentPublication(
        "rocm_moe_router_q8_gate_cache",
        PersistentWorkspacePublicationKey{
            .word0 = mgr.id(),
            .word1 = 0x300000,
            .word2 = 2048,
            .word3 = 256,
        },
        /*slot_capacity=*/2,
        [](size_t) -> std::shared_ptr<void>
        {
            return std::make_shared<int>(29);
        }))
        << "A genuinely distinct immutable weight must still fail closed when "
           "the physical table is full";
}

TEST_F(Test__DeviceWorkspaceManager, FailedImmutablePublicationRollsBackItsSlot)
{
    DeviceWorkspaceManager mgr(device, budget);
    const PersistentWorkspacePublicationKey key{
        .word0 = mgr.id(),
        .word1 = 0x100000,
        .word2 = 2048,
        .word3 = 256,
    };

    EXPECT_FALSE(mgr.getOrCreatePersistentPublication(
        "cuda_moe_router_q8_gate_cache",
        key,
        /*slot_capacity=*/1,
        [](size_t) -> std::shared_ptr<void>
        {
            return {};
        }));

    const auto retry = mgr.getOrCreatePersistentPublication(
        "cuda_moe_router_q8_gate_cache",
        key,
        /*slot_capacity=*/1,
        [](size_t slot) -> std::shared_ptr<void>
        {
            EXPECT_EQ(slot, 0u);
            return std::make_shared<int>(31);
        });
    ASSERT_TRUE(retry);
    EXPECT_TRUE(retry.created);
    EXPECT_EQ(retry.slot, 0u)
        << "A failed event or conversion setup must not poison the immutable publication namespace";
}

TEST_F(Test__DeviceWorkspaceManager, ReleaseInvalidatesOldMetadataLeaseNamespace)
{
    DeviceWorkspaceManager mgr(device, budget);

    auto old_lease = mgr.acquirePersistentSlot("moe_expert_masks", 1);
    ASSERT_NE(old_lease, nullptr);
    EXPECT_EQ(old_lease->slot(), 0u);

    mgr.release();

    auto new_lease = mgr.acquirePersistentSlot("moe_expert_masks", 1);
    ASSERT_NE(new_lease, nullptr);
    EXPECT_EQ(new_lease->slot(), 0u)
        << "A replacement device allocation must not inherit occupied slots "
           "from graph owners tied to the released allocation";

    old_lease.reset();
    EXPECT_EQ(
        mgr.acquirePersistentSlot("moe_expert_masks", 1),
        nullptr)
        << "Destroying an obsolete lease must not release a slot in the new namespace";
}

// ============================================================================
// Device Tests
// ============================================================================

TEST_F(Test__DeviceWorkspaceManager, DeviceReturnsCorrectValue)
{
    {
        DeviceWorkspaceManager mgr(DeviceId::cpu(), budget);
        EXPECT_TRUE(mgr.device().is_cpu());
    }

    // Note: CUDA/ROCm device tests would require actual GPU
    // These test just the DeviceId storage, not actual allocation
    {
        DeviceWorkspaceManager mgr(DeviceId::cuda(0), budget);
        EXPECT_TRUE(mgr.device().is_cuda());
        EXPECT_EQ(mgr.device().ordinal, 0);
    }

    {
        DeviceWorkspaceManager mgr(DeviceId::rocm(1), budget);
        EXPECT_TRUE(mgr.device().is_rocm());
        EXPECT_EQ(mgr.device().ordinal, 1);
    }
}
