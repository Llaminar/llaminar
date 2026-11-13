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
 * - Typed final-weight placement rejects invalid identities before allocation
 * - Actual allocator placement lifecycle checks under scoped syscall/allocation faults
 * - Native happy-path CPU and memory-policy restoration before publication
 *
 * @author David Sanftenberg
 * @date 2026-01-21
 */

#include <gtest/gtest.h>
#include "memory/NUMAAllocator.h"
#include "memory/CPUWeightStoragePlacement.h"
#include "memory/NUMAPageQuery.h"
#include "utils/Logger.h"
#include "tensors/AlignedVector.h"

#include <cstdlib>
#include <algorithm>
#include <array>
#include <dlfcn.h>
#include <limits>
#include <new>
#include <numa.h>
#include <cerrno>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>
#include <numeric>

#include <sched.h>
#include <numaif.h>
#include <unistd.h>

using namespace llaminar2;

namespace
{
/** @brief Deterministic failures at the actual production placement boundary. */
enum class PlacementFault
{
    None, BeforeBindingAllocation, RecordInsertionAllocation, RejectPolicy, RejectPolicyAndRestoreAffinity,
    RestoreAffinity, RestorePolicy, ReadAffinity, ReadPolicy,
    ContradictAffinity, ContradictPolicy
};

/** @brief Authored NUMA facts plus real caller CPU-mask observations, scoped to one fixture thread. */
struct PlacementProbe final
{
    PlacementFault fault = PlacementFault::None;
    cpu_set_t incoming{};
    std::array<unsigned long, 64> nodes{};
    int mode = MPOL_PREFERRED;
    int target_node = 0;
    unsigned affinity_writes = 0;
    unsigned policy_writes = 0;
    unsigned borrowed_allocations = 0;
    bool borrowed = false;
    bool affinity_restored = false;
    bool policy_restored = false;
    bool reject_borrowed_allocations = false;
    bool fail_next_allocation = false;
    bool injected_allocation = false;
    void *storage = nullptr;
    bool freed_storage = false;
};
thread_local PlacementProbe *placement_probe = nullptr;

/** @brief Resolve the selected native operation before fault injection is armed. */
template<class Function>
Function nativePlacementSymbol(const char *name) noexcept
{
    auto *symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) std::_Exit(91);
    return reinterpret_cast<Function>(symbol);
}
using NativeAffinityWrite = int (*)(pid_t, size_t, const cpu_set_t *);
using NativeAffinityRead = int (*)(pid_t, size_t, cpu_set_t *);
using NativePolicyWrite = long (*)(int, const unsigned long *, unsigned long);
using NativePolicyRead = long (*)(int *, unsigned long *, unsigned long, void *, unsigned);
using NativeNodeCPUs = int (*)(int, bitmask *);
using NativePageQuery = long (*)(int, unsigned long, void **, const int *, int *, int);
using NativeAlignedAllocation = void *(*)(size_t, size_t);
using NativeFree = void (*)(void *);

/** @return Exact native scheduler writer, unmodified outside this fixture's TLS scope. */
NativeAffinityWrite nativeAffinityWrite() noexcept
{ static const auto f = nativePlacementSymbol<NativeAffinityWrite>("sched_setaffinity"); return f; }
/** @return Exact native scheduler reader. */
NativeAffinityRead nativeAffinityRead() noexcept
{ static const auto f = nativePlacementSymbol<NativeAffinityRead>("sched_getaffinity"); return f; }
/** @return Selected libnuma policy writer. */
NativePolicyWrite nativePolicyWrite() noexcept
{ static const auto f = nativePlacementSymbol<NativePolicyWrite>("set_mempolicy"); return f; }
/** @return Selected libnuma policy reader. */
NativePolicyRead nativePolicyRead() noexcept
{ static const auto f = nativePlacementSymbol<NativePolicyRead>("get_mempolicy"); return f; }
/** @return Selected topology CPU query. */
NativeNodeCPUs nativeNodeCPUs() noexcept
{ static const auto f = nativePlacementSymbol<NativeNodeCPUs>("numa_node_to_cpus"); return f; }
/** @return Selected physical page query. */
NativePageQuery nativePageQuery() noexcept
{ static const auto f = nativePlacementSymbol<NativePageQuery>("move_pages"); return f; }
/** @return Native allocation without replacing the production storage path. */
NativeAlignedAllocation nativeAlignedAllocation() noexcept
{ static const auto f = nativePlacementSymbol<NativeAlignedAllocation>("aligned_alloc"); return f; }
/** @return Native release without borrowing the allocation fault seam. */
NativeFree nativeFree() noexcept
{
    // free() can run before main. Detect resolver reentry before entering the
    // C++ initialization guard, so loader bootstrap cannot recurse or hang.
    static thread_local bool resolving = false;
    if (resolving)
    {
        constexpr char message[] = "Fatal: recursive native free resolution in NUMA fixture\n";
        const auto written = ::write(STDERR_FILENO, message, sizeof(message) - 1);
        (void)written;
        std::_Exit(90);
    }
    static const auto function = [&] {
        resolving = true;
        const auto result = nativePlacementSymbol<NativeFree>("free");
        resolving = false;
        return result;
    }();
    return function;
}

/** @brief Arm only after resolving every delegate; cleanup cannot leave this test caller pinned. */
class PlacementFaultScope final
{
public:
    /** @brief Author policy/page observations while keeping CPU writes native. */
    explicit PlacementFaultScope(PlacementProbe &probe) : probe_(probe)
    {
        (void)nativeAffinityWrite(); (void)nativeAffinityRead();
        (void)nativePolicyWrite(); (void)nativePolicyRead(); (void)nativeNodeCPUs();
        (void)nativePageQuery(); (void)nativeAlignedAllocation(); (void)nativeFree();
        if (placement_probe || nativeAffinityRead()(0, sizeof(probe_.incoming), &probe_.incoming) != 0)
            std::_Exit(92);
        probe_.nodes[0] = 1;
        placement_probe = &probe_;
    }
    /** @brief Retire the seam and repair only fixture-owned resources if an old-code control leaked them. */
    ~PlacementFaultScope()
    {
        placement_probe = nullptr;
        if (nativeAffinityWrite()(0, sizeof(probe_.incoming), &probe_.incoming) != 0)
            std::_Exit(93);
        if (probe_.storage && !probe_.freed_storage) nativeFree()(probe_.storage);
    }
    PlacementFaultScope(const PlacementFaultScope &) = delete;
    PlacementFaultScope &operator=(const PlacementFaultScope &) = delete;
private:
    PlacementProbe &probe_;
};

/** @return A native caller snapshot even on a single-CPU/container partition. */
cpu_set_t placementCallerMask()
{
    cpu_set_t mask{};
    if (nativeAffinityRead()(0, sizeof(mask), &mask) != 0) std::_Exit(94);
    return mask;
}
} // namespace

/** @brief Observe the actual public allocator's allocation window; unarmed new retains standard semantics. */
void *operator new(std::size_t bytes)
{
    if (auto *probe = placement_probe)
    {
        if (probe->borrowed) ++probe->borrowed_allocations;
        if ((probe->borrowed && probe->reject_borrowed_allocations) || probe->fail_next_allocation ||
            (probe->fault == PlacementFault::BeforeBindingAllocation && probe->storage && !probe->injected_allocation))
        {
            probe->fail_next_allocation = false;
            probe->injected_allocation = true;
            throw std::bad_alloc();
        }
    }
    for (;;)
    {
        if (void *memory = std::malloc(bytes ? bytes : 1)) return memory;
        const auto handler = std::get_new_handler();
        if (!handler) throw std::bad_alloc();
        handler();
    }
}
/** @brief Pair native global-new storage with the real free path. */
void operator delete(void *memory) noexcept { std::free(memory); }
/** @brief Sized deletion preserves the same native release semantics. */
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }

/** @brief Real allocation, observed only while this fixture owns its calling thread. */
extern "C" void *aligned_alloc(size_t alignment, size_t bytes) noexcept
{
    void *memory = nativeAlignedAllocation()(alignment, bytes);
    if (placement_probe && !placement_probe->storage) placement_probe->storage = memory;
    return memory;
}
/** @brief Observe exact storage retirement without manufacturing a production accounting ledger. */
extern "C" void free(void *memory) noexcept
{
    if (placement_probe && memory && memory == placement_probe->storage) placement_probe->freed_storage = true;
    nativeFree()(memory);
}
/** @brief All successful CPU writes remain real scheduler operations. */
extern "C" int sched_setaffinity(pid_t pid, size_t bytes, const cpu_set_t *mask) noexcept
{
    auto *probe = placement_probe;
    if (!probe || pid != 0) return nativeAffinityWrite()(pid, bytes, mask);
    const bool restoring = probe->affinity_writes++ != 0;
    if (restoring && (probe->fault == PlacementFault::RestoreAffinity || probe->fault == PlacementFault::RejectPolicyAndRestoreAffinity)) { errno = EIO; return -1; }
    const int result = nativeAffinityWrite()(pid, bytes, mask);
    if (result == 0) { probe->borrowed = !restoring; probe->affinity_restored = restoring; }
    return result;
}
/** @brief Native CPU readback, with isolated contradictory/error facts only in death tests. */
extern "C" int sched_getaffinity(pid_t pid, size_t bytes, cpu_set_t *mask) noexcept
{
    const int result = nativeAffinityRead()(pid, bytes, mask);
    auto *probe = placement_probe;
    if (!probe || !probe->affinity_restored || result != 0 || pid != 0) return result;
    if (probe->fault == PlacementFault::ReadAffinity) { errno = EIO; return -1; }
    if (probe->fault == PlacementFault::ContradictAffinity) CPU_ZERO(mask);
    return result;
}
/** @brief Author a target CPU within the caller's real allowed mask; this injects lifecycle faults, without topology certification. */
extern "C" int numa_node_to_cpus(int node, bitmask *mask)
{
    auto *probe = placement_probe;
    if (!probe) return nativeNodeCPUs()(node, mask);
    probe->target_node = node;
    numa_bitmask_clearall(mask);
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &probe->incoming)) { numa_bitmask_setbit(mask, cpu); return 0; }
    errno = EINVAL; return -1;
}
/** @brief Deterministic policy publication/rejection through the actual production scope. */
extern "C" long set_mempolicy(int mode, const unsigned long *nodes, unsigned long max_nodes)
{
    auto *probe = placement_probe;
    if (!probe) return nativePolicyWrite()(mode, nodes, max_nodes);
    const bool restoring = probe->policy_writes++ != 0;
    if ((!restoring && (probe->fault == PlacementFault::RejectPolicy || probe->fault == PlacementFault::RejectPolicyAndRestoreAffinity)) ||
        (restoring && probe->fault == PlacementFault::RestorePolicy)) { errno = EIO; return -1; }
    const size_t words = (max_nodes + sizeof(unsigned long)*8-1)/(sizeof(unsigned long)*8);
    if (words > probe->nodes.size()) std::_Exit(95);
    probe->mode = mode; probe->nodes.fill(0);
    if (nodes) std::copy_n(nodes, words, probe->nodes.begin());
    probe->policy_restored = restoring;
    return 0;
}
/** @brief Retain and read back authored policy facts; unarmed physical address queries stay native. */
extern "C" long get_mempolicy(int *mode, unsigned long *nodes, unsigned long max_nodes, void *address, unsigned flags)
{
    auto *probe = placement_probe;
    if (!probe || flags) return nativePolicyRead()(mode, nodes, max_nodes, address, flags);
    if (probe->policy_restored && probe->fault == PlacementFault::ReadPolicy) { errno = EIO; return -1; }
    const size_t words = (max_nodes + sizeof(unsigned long)*8-1)/(sizeof(unsigned long)*8);
    if (words > probe->nodes.size()) std::_Exit(96);
    if (mode) *mode = probe->mode;
    if (nodes) std::copy_n(probe->nodes.begin(), words, nodes);
    if (probe->policy_restored && probe->fault == PlacementFault::ContradictPolicy && mode) *mode = MPOL_DEFAULT;
    return 0;
}
/** @brief Supply controlled page-query facts after the actual first-touch/restore path completes. */
extern "C" long move_pages(int pid, unsigned long count, void **pages, const int *nodes, int *status, int flags)
{
    auto *probe = placement_probe;
    if (!probe) return nativePageQuery()(pid, count, pages, nodes, status, flags);
    for (unsigned long i = 0; i < count; ++i) status[i] = probe->target_node;
    if (probe->fault == PlacementFault::RecordInsertionAllocation) probe->fail_next_allocation = true;
    return 0;
}

TEST(Test__NUMAAllocator, FirstTouchPlacementPreparesBeforeBorrowingAndRestoresExactCaller)
{
    auto &allocator = NUMAAllocator::instance();
    const auto incoming = placementCallerMask();
    for (int lifetime = 0; lifetime < 20; ++lifetime)
    {
        PlacementProbe probe;
        probe.reject_borrowed_allocations = true;
        bool threw = false;
        cpu_set_t after{};
        {
            PlacementFaultScope scope(probe);
            try
            {
                void *memory = allocator.allocateOnNode(4096, -1, 4096);
                EXPECT_NE(memory, nullptr);
                if (memory) allocator.free(memory, 4096);
            }
            catch (const std::bad_alloc &) { threw = true; }
            // Observe old-code failures without also faulting GTest diagnostics.
            probe.reject_borrowed_allocations = false;
            after = placementCallerMask();
            EXPECT_FALSE(probe.borrowed);
            EXPECT_TRUE(probe.policy_restored);
            EXPECT_EQ(probe.mode, MPOL_PREFERRED);
            EXPECT_EQ(probe.nodes[0], 1ul);
        }
        EXPECT_FALSE(threw);
        EXPECT_EQ(probe.borrowed_allocations, 0u);
        EXPECT_TRUE(CPU_EQUAL(&incoming, &after));
        EXPECT_TRUE(probe.freed_storage);
    }
}

TEST(Test__NUMAAllocator, FirstTouchPlacementPolicyRejectionRestoresOnlyPublishedCPU)
{
    auto &allocator = NUMAAllocator::instance();
    const auto incoming = placementCallerMask();
    PlacementProbe probe; probe.fault = PlacementFault::RejectPolicy;
    cpu_set_t after{};
    {
        PlacementFaultScope scope(probe);
        EXPECT_EQ(allocator.allocateOnNode(4096, -1, 4096), nullptr);
        after = placementCallerMask();
        EXPECT_EQ(probe.policy_writes, 1u);
        EXPECT_FALSE(probe.borrowed);
        EXPECT_EQ(probe.mode, MPOL_PREFERRED);
        EXPECT_EQ(probe.nodes[0], 1ul);
    }
    EXPECT_TRUE(CPU_EQUAL(&incoming, &after));
    EXPECT_TRUE(probe.freed_storage);
}

TEST(Test__NUMAAllocator, FirstTouchPlacementExceptionsReleaseUnpublishedStorage)
{
    auto &allocator = NUMAAllocator::instance();
    const int node = allocator.getCurrentNUMANode();
    const auto prior_level = Logger::getInstance().getLogLevel();
    Logger::getInstance().setLogLevel(LogLevel::ERROR);
    for (const auto fault : {PlacementFault::BeforeBindingAllocation, PlacementFault::RecordInsertionAllocation})
    {
        const auto before = allocator.getNodeStats(node).allocated_by_us;
        const auto incoming = placementCallerMask();
        PlacementProbe probe; probe.fault = fault;
        bool threw = false; cpu_set_t after{};
        {
            PlacementFaultScope scope(probe);
            try { (void)allocator.allocateOnNode(4096, -1, 4096); }
            catch (const std::bad_alloc &) { threw = true; }
            // Observe old-code failures without also faulting GTest diagnostics.
            probe.reject_borrowed_allocations = false;
            after = placementCallerMask();
            EXPECT_TRUE(probe.freed_storage);
        }
        EXPECT_TRUE(threw);
        EXPECT_TRUE(probe.injected_allocation);
        EXPECT_TRUE(CPU_EQUAL(&incoming, &after));
        EXPECT_EQ(allocator.getNodeStats(node).allocated_by_us, before);
    }
    Logger::getInstance().setLogLevel(prior_level);
}

TEST(Test__NUMAAllocator, FirstTouchPlacementRestorationErrorsAndContradictionsAreFatal)
{
    for (const auto fault : {PlacementFault::RestoreAffinity, PlacementFault::RestorePolicy,
                             PlacementFault::ReadAffinity, PlacementFault::ReadPolicy,
                             PlacementFault::ContradictAffinity, PlacementFault::ContradictPolicy})
    {
        EXPECT_DEATH(( [&] {
            auto &allocator = NUMAAllocator::instance();
            PlacementProbe probe; probe.fault = fault;
            PlacementFaultScope scope(probe);
            void *memory = allocator.allocateOnNode(4096, -1, 4096);
            if (memory) allocator.free(memory, 4096);
        }()), "cannot restore exact first-touch caller placement");
    }
    // A rejected policy still owns the already-published CPU write. The same
    // fatal behavior is required while acquire() is rolling that write back.
    EXPECT_DEATH(( [] {
        auto &allocator = NUMAAllocator::instance();
        PlacementProbe probe; probe.fault = PlacementFault::RejectPolicyAndRestoreAffinity;
        PlacementFaultScope scope(probe);
        (void)allocator.allocateOnNode(4096, -1, 4096);
    }()), "cannot restore exact first-touch caller placement");
}


TEST(Test__NUMAAllocator, CPUWeightPlacementRejectsNegativeNodeBeforeAllocation)
{
    EXPECT_THROW(CPUWeightStoragePlacement::onNode(-1), std::invalid_argument);
    EXPECT_THROW(CPUWeightStoragePlacement::onNode(-42), std::invalid_argument);
    auto empty = CPUWeightStoragePlacement::local().allocate<uint8_t>(0);
    EXPECT_TRUE(empty.empty());
}

TEST(Test__NUMAAllocator, BatchedNUMAObservationRequiresActualResolvedNode)
{
    uint8_t retained_page = 0xa5;
    int resolutions = 0;
    const auto resolve = [&](const void *address) {
        EXPECT_EQ(address, &retained_page);
        ++resolutions;
        return 7; // Deliberately differs from both ordinary fixture nodes.
    };
    EXPECT_EQ(resolveObservedNUMANode(0, &retained_page, resolve), 0);
    EXPECT_EQ(resolveObservedNUMANode(1, &retained_page, resolve), 1);
    EXPECT_EQ(resolveObservedNUMANode(-EFAULT, &retained_page, resolve), -EFAULT);
    EXPECT_EQ(resolveObservedNUMANode(-EACCES, &retained_page, resolve), -EACCES);
    EXPECT_EQ(resolutions, 0);
    EXPECT_EQ(resolveObservedNUMANode(-ENOENT, &retained_page, resolve), 7);
    EXPECT_EQ(resolutions, 1);
    EXPECT_EQ(resolveObservedNUMANode(-ENOENT, &retained_page,
        [](const void *) { return -EFAULT; }), -EFAULT);
    EXPECT_EQ(retained_page, 0xa5); // Completion never re-touches or repairs bytes.
}

TEST(Test__NUMAAllocator, PhysicalNodeLookupRejectsInaccessibleGuard)
{
    const auto page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    auto guarded = AlignedVector<uint8_t>::pageMappedUninitialized(page);
    guarded.front() = 0xa5;
    const auto *guard = reinterpret_cast<const void *>(
        reinterpret_cast<uintptr_t>(guarded.data()) - page);
    EXPECT_EQ(NUMAAllocator::instance().getNUMANodeForAddress(guard), -1);
    EXPECT_EQ(guarded.front(), 0xa5);
}

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

TEST(Test__NUMAAllocator, PrepareExternalReceiveRangeUsesCertifiedFirstTouch)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();
    if (!allocator.isNUMAAvailable())
        GTEST_SKIP() << "NUMA policy APIs are unavailable";

    constexpr size_t page_size = 4096;
    constexpr size_t bytes = 3 * page_size;
    std::unique_ptr<void, decltype(&std::free)> storage(
        std::aligned_alloc(page_size, bytes), &std::free);
    ASSERT_NE(storage, nullptr);

    cpu_set_t incoming_affinity{};
    ASSERT_EQ(sched_getaffinity(
                  0, sizeof(incoming_affinity), &incoming_affinity),
              0);

    constexpr unsigned long word_bits = std::numeric_limits<unsigned long>::digits;
    const unsigned long policy_bits = std::max(word_bits, static_cast<unsigned long>(numa_max_node() + 1));
    std::vector<unsigned long> incoming_nodes((policy_bits + word_bits - 1) / word_bits, 0);
    int incoming_mode = MPOL_DEFAULT;
    ASSERT_EQ(get_mempolicy(&incoming_mode, incoming_nodes.data(), policy_bits, nullptr, 0), 0);

    // Repeatedly alternate nodes so recycled resident pages cannot make a
    // policy-only implementation appear correct. Each preparation must also
    // restore the exact launcher/container affinity before returning.
    const int rounds = allocator.numNUMANodes() > 1 ? 32 : 1;
    for (int round = 0; round < rounds; ++round)
    {
        const int target_node = round % allocator.numNUMANodes();
        ASSERT_TRUE(allocator.prepareExternalReceiveRangeOnNode(
            storage.get(), bytes, target_node));

        cpu_set_t restored_affinity{};
        ASSERT_EQ(sched_getaffinity(
                      0, sizeof(restored_affinity), &restored_affinity),
                  0);
        EXPECT_TRUE(CPU_EQUAL(&incoming_affinity, &restored_affinity));
        std::vector<unsigned long> restored_nodes(incoming_nodes.size(), 0);
        int restored_mode = MPOL_DEFAULT;
        ASSERT_EQ(get_mempolicy(&restored_mode, restored_nodes.data(), policy_bits, nullptr, 0), 0);
        EXPECT_EQ(restored_mode, incoming_mode);
        EXPECT_EQ(restored_nodes, incoming_nodes);

        // Simulate the transport's full overwrite after preparation.
        std::memset(storage.get(), 0xA5, bytes);
        const auto *data = static_cast<const uint8_t *>(storage.get());
        for (size_t offset = 0; offset < bytes; offset += page_size)
        {
            EXPECT_EQ(
                allocator.getNUMANodeForAddress(data + offset),
                target_node)
                << "round " << round << " page offset " << offset
                << " ignored certified first touch";
        }
    }
}

TEST(Test__NUMAAllocator, FirstTouchOverridesInheritedRemoteMemoryPolicy)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();
    if (!allocator.isNUMAAvailable() || allocator.numNUMANodes() < 2)
        GTEST_SKIP() << "Two NUMA nodes are required for conflicting-policy proof";

    constexpr size_t bytes = 3u * 4096u;
    constexpr unsigned long max_nodes = sizeof(unsigned long) * 8u;
    const unsigned long preferred_node_zero = 1ul;
    int policy_error = 0;
    bool allocated_on_requested_node = false;
    int restored_mode = -1;
    unsigned long restored_nodes = 0;

    // A separate thread lets the kernel discard this deliberately conflicting
    // policy at exit, even if an assertion later fails in the test process.
    std::thread worker([&] {
        if (set_mempolicy(MPOL_PREFERRED, &preferred_node_zero, max_nodes) != 0)
        {
            policy_error = errno;
            return;
        }

        void *memory = allocator.allocateOnNode(bytes, 1, 4096);
        if (memory)
        {
            allocated_on_requested_node =
                allocator.getNUMANodeForAddress(memory) == 1;
            allocator.free(memory, bytes);
        }

        if (get_mempolicy(&restored_mode, &restored_nodes, max_nodes,
                          nullptr, 0) != 0)
            policy_error = errno;
    });
    worker.join();

    if (policy_error != 0)
        GTEST_SKIP() << "This environment cannot set/query thread NUMA policy: "
                     << std::strerror(policy_error);
    EXPECT_TRUE(allocated_on_requested_node)
        << "First touch must target the requested NUMA node despite an "
           "inherited policy preferring a remote node";
    EXPECT_EQ(restored_mode, MPOL_PREFERRED);
    EXPECT_EQ(restored_nodes & preferred_node_zero, preferred_node_zero)
        << "First-touch policy must be restored for subsequent allocations";
}

TEST(Test__NUMAAllocator, DedicatedMappingsSurviveAlternatingNodeChurn)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();
    if (!allocator.isNUMAAvailable())
        GTEST_SKIP() << "NUMA policy APIs are unavailable";

    const size_t page_size =
        NUMAAllocator::externalRangeAlignment();
    constexpr size_t pages_per_round = 384;
    const int rounds = allocator.numNUMANodes() > 1 ? 32 : 4;
    for (int round = 0; round < rounds; ++round)
    {
        const int target_node = round % allocator.numNUMANodes();
        auto storage =
            AlignedVector<std::uint8_t>::pageMappedUninitialized(
                pages_per_round * page_size - 17u);
        ASSERT_TRUE(allocator.prepareExternalReceiveRangeOnNode(
            storage.data(), storage.size(), target_node));

        for (size_t offset = 0;
             offset < storage.allocationBytes();
             offset += page_size)
        {
            ASSERT_EQ(
                allocator.getNUMANodeForAddress(storage.data() + offset),
                target_node)
                << "round " << round << " page offset " << offset;
        }
    }
}

TEST(Test__NUMAAllocator, HugePageAdvisedMappingsRetainExactFirstTouchNode)
{
    NUMAAllocator &allocator = NUMAAllocator::instance();
    if (!allocator.isNUMAAvailable())
        GTEST_SKIP() << "NUMA policy APIs are unavailable";

    const size_t page_size =
        NUMAAllocator::externalRangeAlignment();
    constexpr size_t mapping_bytes = 4u * 1024u * 1024u;
    const int rounds = allocator.numNUMANodes() > 1 ? 16 : 4;
    for (int round = 0; round < rounds; ++round)
    {
        const int target_node = round % allocator.numNUMANodes();
        auto storage =
            AlignedVector<std::uint8_t>::pageMappedUninitialized(
                mapping_bytes);
        ASSERT_TRUE(allocator.prepareExternalReceiveRangeOnNode(
            storage.data(), storage.size(), target_node));

        for (size_t offset = 0; offset < storage.allocationBytes();
             offset += page_size)
        {
            ASSERT_EQ(
                allocator.getNUMANodeForAddress(storage.data() + offset),
                target_node)
                << "round " << round << " page offset " << offset
                << " escaped the exact first-touch node";
        }
    }
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
