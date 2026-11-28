/**
 * @file Test__NUMAFirstTouchBoundaryNative.cpp
 * @brief Real-kernel native regressions for the first-touch boundary.
 *
 * Explicit ProductionTestPreflight registrations (these are NOT
 * device-free Unit tests):
 *
 * 1. SingleNodeTransitionProof — on any allowed topology (one RAM node
 *    suffices): the production populate sequence through a recording
 *    boundary instance that CALLS THROUGH to the real default ops.
 *    Records the real syscalls with results/errno (errno preserved
 *    across the recording activity), performs REAL kernel-policy
 *    probes immediately before and after the actual touch with masks
 *    sized from the kernel's COMPLETE possible-node geometry, matches
 *    the exact touch address/length/stride and the one-store-per-page
 *    placement content, and QUERIES AND ASSERTS the restored policy
 *    AND the restored affinity (complete nodemask words, complete
 *    cpu_set_t) against the captured originals.
 *
 * 2. RemoteRAMPlacementMultiRAMNode — the genuine remote-placement
 *    obligation under the memory-node contract (>= 2 allowed
 *    RAM-bearing nodes AND a target node with a CPU usable under the
 *    process's permitted affinity). A PREREQUISITE positive control
 *    first proves that the inherited STRICT remote MPOL_BIND actually
 *    places pages remotely; then the production sequence's target
 *    placement, the PUBLIC allocator composition
 *    (prepareExternalReceiveRangeOnNode), and the omit-bind mutation
 *    negative (isolated to the TARGET bind; restoration stays real)
 *    are all proven. The omit control must place every page REMOTELY
 *    (proven by the SAME everyPageOnNode validator used for the
 *    production proof on the remote node) AND be rejected by that
 *    validator for the target node. The original thread state is
 *    verified after EVERY case. On machines without the prerequisite
 *    the test FAILS explicitly (never skip-as-pass).
 *
 * 3. QueryValidator* — per-page-error, empty-extent and mixed-placement
 *    negatives for the shared placement query/validators: a query that
 *    reports any negative per-page status, an empty extent, or a
 *    mixed placement is REJECTED; it can never certify a proof.
 *
 * Safety invariants: every mapping is RAII-owned; every mask is sized
 * from the kernel's complete possible-node geometry and preallocated
 * BEFORE any mutation (node indices are bounds-checked through the
 * shared helpers before any indexing or operation); the thread-state
 * guard fails closed when capture fails, restores on every path,
 * verifies the COMPLETE restored policy+affinity before disarming, and
 * terminates if restoration cannot be verified; recorder storage and
 * probe buffers are preallocated — no allocation happens during the
 * observed sequence; the placement query validates every per-page
 * status.
 */

#include <gtest/gtest.h>
#include "memory/FirstTouchBoundary.h"
#include "memory/NUMAAllocator.h"
#include "utils/HostPageSize.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <string>
#include <vector>

#include <sys/mman.h>

namespace
{
    const size_t page_bytes = llaminar2::hostPageBytes();

    /// The kernel's COMPLETE possible-node geometry: every mask in
    /// this file is sized from it, preallocated before any mutation.
    /// An unavailable topology is an EXPLICIT rejection (the tests
    /// ASSERT and stop) — never a silent mapping to node 0.
    struct NativeGeometry
    {
        int max_node = -1;         ///< kernel possible max node
        size_t words = 0;          ///< nodemask word count
        unsigned long bits = 0;    ///< Linux maxnode operand, including the extra ABI unit.
    };

    /// Acquire the geometry with an explicit rejection on failure.
    /// Call through ASSERT_NO_FATAL_FAILURE so the test stops before
    /// any mutation when the topology is unavailable.
    void requireNativeGeometry(NativeGeometry &out)
    {
        const int max_node = numa_max_possible_node();
        ASSERT_GE(max_node, 0)
            << "EXPLICIT REJECTION: numa_max_possible_node() failed; "
               "the native proofs refuse to proceed on an unavailable "
               "topology (no silent single-node substitution)";
        out.max_node = max_node;
        out.words = llaminar2::firstTouchMaskWordCount(max_node);
        out.bits = static_cast<unsigned long>(max_node) + 2;
    }

    #define ASSERT_NODE_IN_GEOMETRY(geom, node)                               \
        do                                                                     \
        {                                                                      \
            ASSERT_GE((node), 0);                                               \
            ASSERT_LE(                                                         \
                (node), (geom).max_node)                                       \
                << "node " << (node) << " exceeds the kernel's "               \
                   "possible-node geometry " << (geom).max_node                \
                << " — this test cannot address it";                           \
        } while (false)

    // ------------------------------------------------------------------
    // RAII-owned anonymous mapping (page-aligned, page-rounded).
    // ------------------------------------------------------------------
    class Mapping
    {
    public:
        explicit Mapping(size_t page_count)
            : base_(::mmap(nullptr, page_count * page_bytes,
                  PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)),
              bytes_(page_count * page_bytes)
        {
            if (base_ == MAP_FAILED)
                base_ = nullptr;
        }

        ~Mapping()
        {
            if (base_)
                (void)::munmap(base_, bytes_);
        }

        Mapping(const Mapping &) = delete;
        Mapping &operator=(const Mapping &) = delete;

        bool valid() const { return base_ != nullptr; }
        void *get() const { return base_; }
        size_t bytes() const { return bytes_; }

    private:
        void *base_;
        size_t bytes_;
    };

    // ------------------------------------------------------------------
    // Thread-state guard: capture-verify-restore with VERIFIED
    // restoration of the COMPLETE original policy and affinity on
    // every path.
    // ------------------------------------------------------------------
    class ThreadStateGuard
    {
    public:
        /// @param geometry Complete possible-node geometry; ALL guard
        ///        storage (captured policy AND the verification
        ///        scratch) is preallocated HERE, at construction —
        ///        before any mutation — so no restoration or unwind
        ///        path ever depends on a fresh allocation.
        explicit ThreadStateGuard(const NativeGeometry &geometry)
            : geometry_(geometry),
              original_nodes_(geometry.words, 0),
              verify_nodes_(geometry.words, 0)
        {
            if (::get_mempolicy(&original_mode_, original_nodes_.data(),
                    geometry.bits, nullptr, 0) != 0)
                capture_failed_ = true;
            if (::sched_getaffinity(
                    0, sizeof(original_affinity_), &original_affinity_) != 0)
                capture_failed_ = true;
        }

        bool captured() const { return !capture_failed_; }

        int original_mode() const { return original_mode_; }
        const std::vector<unsigned long> &original_nodes() const
        {
            return original_nodes_;
        }
        const cpu_set_t &original_affinity() const
        {
            return original_affinity_;
        }

        /// Explicit VERIFIED restoration (happy path): sets the policy,
        /// then re-queries and compares the COMPLETE nodemask words and
        /// mode, then compares the complete affinity — all BEFORE
        /// disarming the destructor.
        void restoreAndVerify()
        {
            const bool has_nodes =
                !llaminar2::firstTouchMaskEmpty(
                    original_nodes_.data(), geometry_.words);
            ASSERT_EQ(::set_mempolicy(original_mode_,
                          has_nodes ? original_nodes_.data() : nullptr,
                          has_nodes ? geometry_.bits : 0),
                0)
                << "policy restoration failed: " << std::strerror(errno);
            int mode = -1;
            std::fill(verify_nodes_.begin(), verify_nodes_.end(), 0);
            ASSERT_EQ(::get_mempolicy(&mode, verify_nodes_.data(),
                          geometry_.bits, nullptr, 0),
                0)
                << "policy verification query failed: "
                << std::strerror(errno);
            ASSERT_EQ(mode, original_mode_)
                << "restored policy mode mismatch";
            ASSERT_TRUE(llaminar2::firstTouchMasksEqual(
                verify_nodes_.data(), original_nodes_.data(),
                geometry_.words))
                << "restored policy nodemask mismatch (complete "
                   "comparison)";
            cpu_set_t current{};
            ASSERT_EQ(::sched_getaffinity(
                          0, sizeof(current), &current),
                0);
            ASSERT_EQ(memcmp(&current, &original_affinity_,
                      sizeof(current)),
                0)
                << "affinity restoration mismatch";
            restored_ = true;
        }

        ~ThreadStateGuard()
        {
            if (capture_failed_ || restored_)
                return; // nothing mutated / already verified
            // Assertion-failure path: restoration still runs and is
            // VERIFIED — an unrestorable or unverifiable thread state
            // is fatal, never silently discarded.
            const bool has_nodes =
                !llaminar2::firstTouchMaskEmpty(
                    original_nodes_.data(), geometry_.words);
            const int policy_errno =
                ::set_mempolicy(original_mode_,
                    has_nodes ? original_nodes_.data() : nullptr,
                    has_nodes ? geometry_.bits : 0) == 0
                    ? 0
                    : errno;
            int affinity_errno = 0;
            int verify_errno = 0;
            if (::sched_setaffinity(0, sizeof(original_affinity_),
                    &original_affinity_) != 0)
                affinity_errno = errno;
            else
            {
                // Verify the restoration actually re-established the
                // complete original state, using the PREALLOCATED
                // scratch (no allocation on the unwind path).
                int mode = -1;
                std::fill(verify_nodes_.begin(), verify_nodes_.end(), 0);
                if (::get_mempolicy(&mode, verify_nodes_.data(),
                        geometry_.bits, nullptr, 0) != 0 ||
                    mode != original_mode_ ||
                    !llaminar2::firstTouchMasksEqual(
                        verify_nodes_.data(), original_nodes_.data(),
                        geometry_.words))
                {
                    verify_errno = errno != 0 ? errno : EFAULT;
                }
                else
                {
                    cpu_set_t current{};
                    if (::sched_getaffinity(
                            0, sizeof(current), &current) != 0 ||
                        memcmp(&current, &original_affinity_,
                            sizeof(current)) != 0)
                        verify_errno = errno != 0 ? errno : EFAULT;
                }
            }
            if (policy_errno != 0 || affinity_errno != 0 ||
                verify_errno != 0)
            {
                fprintf(stderr,
                    "FATAL: native test thread-state restoration failed "
                    "verification (policy_errno=%d affinity_errno=%d "
                    "verify_errno=%d)\n",
                    policy_errno, affinity_errno, verify_errno);
                std::terminate();
            }
        }

    private:
        NativeGeometry geometry_;
        int original_mode_ = MPOL_DEFAULT;
        std::vector<unsigned long> original_nodes_;
        std::vector<unsigned long> verify_nodes_; // preallocated scratch
        cpu_set_t original_affinity_{};
        bool capture_failed_ = false;
        bool restored_ = false;
    };

    /// Verify the CURRENT thread state equals the expected mode and
    /// COMPLETE nodemask words and the expected affinity (used after
    /// every case).
    ::testing::AssertionResult threadStateIs(const NativeGeometry &geom,
        int expected_mode,
        const std::vector<unsigned long> &expected_nodes,
        const cpu_set_t &expected_affinity)
    {
        int mode = -1;
        std::vector<unsigned long> nodes(geom.words, 0);
        if (::get_mempolicy(&mode, nodes.data(), geom.bits,
                nullptr, 0) != 0)
            return ::testing::AssertionFailure()
                << "get_mempolicy failed: " << std::strerror(errno);
        if (mode != expected_mode ||
            !llaminar2::firstTouchMasksEqual(nodes.data(),
                expected_nodes.data(), geom.words))
            return ::testing::AssertionFailure()
                << "policy mode/nodes mismatch against the complete "
                   "expected mask";
        cpu_set_t current{};
        if (::sched_getaffinity(0, sizeof(current), &current) != 0)
            return ::testing::AssertionFailure()
                << "sched_getaffinity failed: " << std::strerror(errno);
        if (memcmp(&current, &expected_affinity, sizeof(current)) != 0)
            return ::testing::AssertionFailure()
                << "affinity does not equal the expected mask";
        return ::testing::AssertionSuccess();
    }

    /// One strict single-node MPOL_BIND policy mask for @p node
    /// (allocated OUTSIDE the observed sequence; node bounds-checked
    /// first).
    std::vector<unsigned long> strictBindMask(
        const NativeGeometry &geom, int node)
    {
        EXPECT_GE(node, 0);
        EXPECT_LE(node, geom.max_node);
        std::vector<unsigned long> words(geom.words, 0);
        if (node >= 0 && node <= geom.max_node)
            EXPECT_TRUE(llaminar2::firstTouchMaskSetNode(
                words.data(), geom.words, node));
        return words;
    }

    /// Set a strict single-node MPOL_BIND as the inherited policy.
    int setStrictBindMasked(const unsigned long *words,
        const NativeGeometry &geom)
    {
        return ::set_mempolicy(MPOL_BIND, words, geom.bits);
    }

    // ------------------------------------------------------------------
    // Shared per-page placement query (move_pages with nodes == NULL
    // only REPORTS placement; it never moves application bytes). The
    // ONE validator used by the positive control, the production
    // proof, the public allocator proof, and the omit-bind mutant.
    // It REJECTS: syscall errors, ANY negative per-page status (a
    // per-page -errno is a query error, never a placement), and empty
    // extents. A rejected query can never certify a proof.
    // ------------------------------------------------------------------
    constexpr size_t kMaxQueryPages = 16;

    struct PlacementQuery
    {
        bool query_ok = false;
        int node[kMaxQueryPages] = {-1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1};
        size_t count = 0;
        int query_errno = 0;
    };

    /// Address-level query: validates every per-page status.
    PlacementQuery queryPlacementAddresses(void *addresses[],
        size_t page_count)
    {
        PlacementQuery out;
        if (page_count == 0 || page_count > kMaxQueryPages)
        {
            out.query_errno = EINVAL; // empty or oversized extent
            return out;
        }
        int statuses[kMaxQueryPages];
        for (size_t i = 0; i < page_count; ++i)
            statuses[i] = -1;
        if (::move_pages(0, static_cast<unsigned long>(page_count),
                addresses, nullptr, statuses, 0) != 0)
        {
            out.query_errno = errno;
            return out;
        }
        for (size_t i = 0; i < page_count; ++i)
        {
            if (statuses[i] < 0)
            {
                // A per-page error (e.g. -ENOENT for an unmapped page)
                // is a REJECTED query, never a placement observation.
                out.query_errno = -statuses[i];
                return out;
            }
        }
        for (size_t i = 0; i < page_count; ++i)
            out.node[i] = statuses[i];
        out.query_ok = true;
        out.count = page_count;
        return out;
    }

    PlacementQuery queryPlacement(const void *base, size_t page_count)
    {
        if (page_count == 0 || page_count > kMaxQueryPages)
        {
            PlacementQuery out;
            out.query_errno = EINVAL;
            return out;
        }
        void *addresses[kMaxQueryPages];
        for (size_t i = 0; i < page_count; ++i)
            addresses[i] = static_cast<uint8_t *>(
                               const_cast<void *>(base)) +
                (i * page_bytes);
        return queryPlacementAddresses(addresses, page_count);
    }

    /// The ONE placement predicate: every queried page sits on @p node,
    /// with NO query error. Used unchanged by the production proof
    /// (must hold) and the omit-bind control (must hold for the remote
    /// node and must be REJECTED for the target node).
    ::testing::AssertionResult everyPageOnNode(
        const PlacementQuery &query, int node)
    {
        if (!query.query_ok)
            return ::testing::AssertionFailure()
                << "placement query rejected: errno " << query.query_errno;
        if (query.count == 0)
            return ::testing::AssertionFailure()
                << "placement query has an empty extent";
        for (size_t i = 0; i < query.count; ++i)
            if (query.node[i] != node)
                return ::testing::AssertionFailure()
                    << "page " << i << " on node " << query.node[i]
                    << " instead of " << node;
        return ::testing::AssertionSuccess();
    }

    // ------------------------------------------------------------------
    // Native recorder: wraps the REAL default ops callables, records
    // every call with result/errno into FIXED preallocated storage,
    /// and probes the REAL kernel policy immediately around the actual
    /// touch into PREALLOCATED complete-geometry buffers. No
    /// allocation and no throw during the observed sequence.
    // ------------------------------------------------------------------
    constexpr size_t kMaxCalls = 16;

    struct NativeRecorder
    {
        struct Call
        {
            const char *name;
            int result;
            int call_errno;
        };
        Call calls[kMaxCalls];
        size_t call_count = 0;
        bool overflow = false;

        /// Preallocated probe buffers (complete geometry, allocated at
        /// recorder construction — BEFORE any mutation).
        std::vector<unsigned long> probe_before;
        std::vector<unsigned long> probe_after;
        unsigned long probe_bits = 0; // geometry bit count for probes

        struct TouchObservation
        {
            const volatile void *base = nullptr;
            size_t bytes = 0;
            size_t stride = 0;
            int policy_before_mode = -1;
            int policy_after_mode = -1;
        };
        TouchObservation touches[1];
        size_t touch_count = 0;

        /// @param words Complete-geometry word count; the probe
        ///        buffers are preallocated HERE, before any mutation.
        explicit NativeRecorder(const NativeGeometry &geom)
            : probe_before(geom.words, 0),
              probe_after(geom.words, 0),
              probe_bits(geom.bits)
        {
        }

        void append(const Call &call)
        {
            if (call_count < kMaxCalls)
                calls[call_count++] = call;
            else
                overflow = true;
        }
    };

    /// Record with errno preserved across the recording activity.
    void recordCall(NativeRecorder &recorder, const char *name,
        int result)
    {
        const int saved_errno = errno;
        recorder.append(NativeRecorder::Call{
            name, result, result == 0 ? 0 : saved_errno});
        errno = saved_errno;
    }

    int nativeGetAffinity(void *context, cpu_set_t *mask)
    {
        const auto &ops = llaminar2::firstTouchDefaultOps();
        const int result = ops.getThreadAffinity(nullptr, mask);
        recordCall(*static_cast<NativeRecorder *>(context),
            "getaffinity", result);
        return result;
    }

    int nativeSetAffinity(void *context, const cpu_set_t *mask)
    {
        const auto &ops = llaminar2::firstTouchDefaultOps();
        const int result = ops.setThreadAffinity(nullptr, mask);
        recordCall(*static_cast<NativeRecorder *>(context),
            "setaffinity", result);
        return result;
    }

    int nativeGetPolicy(void *context, int *mode, unsigned long *nodemask,
        unsigned long max_nodes, void *addr, int flags)
    {
        const auto &ops = llaminar2::firstTouchDefaultOps();
        const int result =
            ops.getMemoryPolicy(nullptr, mode, nodemask, max_nodes, addr,
                flags);
        recordCall(*static_cast<NativeRecorder *>(context),
            "getpolicy", result);
        return result;
    }

    int nativeSetPolicy(void *context, int mode,
        const unsigned long *nodemask, unsigned long max_nodes)
    {
        const auto &ops = llaminar2::firstTouchDefaultOps();
        const int result =
            ops.setMemoryPolicy(nullptr, mode, nodemask, max_nodes);
        recordCall(*static_cast<NativeRecorder *>(context),
            "setpolicy", result);
        return result;
    }

    int nativeNodeCpuSet(void *context, int node, cpu_set_t *mask)
    {
        const auto &ops = llaminar2::firstTouchDefaultOps();
        const int result = ops.nodeCpuSet(nullptr, node, mask);
        recordCall(*static_cast<NativeRecorder *>(context),
            "nodecpuset", result);
        return result;
    }

    void nativeTouch(void *context, volatile void *base, size_t bytes,
        size_t stride)
    {
        auto *recorder = static_cast<NativeRecorder *>(context);
        const int saved_errno = errno;
        NativeRecorder::TouchObservation observation;
        observation.base = base;
        observation.bytes = bytes;
        observation.stride = stride;
        // REAL kernel-policy probe immediately BEFORE the actual touch
        // into the PREALLOCATED complete-geometry buffer.
        if (::get_mempolicy(&observation.policy_before_mode,
                recorder->probe_before.data(), recorder->probe_bits,
                nullptr, 0) != 0)
            observation.policy_before_mode = -1;
        // The ACTUAL touch (the production page-store loop).
        llaminar2::firstTouchDefaultOps().touchRange(
            nullptr, base, bytes, stride);
        // REAL kernel-policy probe immediately AFTER the touch.
        if (::get_mempolicy(&observation.policy_after_mode,
                recorder->probe_after.data(), recorder->probe_bits,
                nullptr, 0) != 0)
            observation.policy_after_mode = -1;
        errno = saved_errno; // probes must not clobber observed errno
        recorder->append(NativeRecorder::Call{"touch", 0, 0});
        if (recorder->touch_count == 0)
            recorder->touches[recorder->touch_count++] = observation;
    }

    llaminar2::FirstTouchBoundaryOps nativeRecordingOps(
        NativeRecorder &recorder)
    {
        return llaminar2::FirstTouchBoundaryOps{
            &recorder,
            nativeGetAffinity,
            nativeSetAffinity,
            nativeGetPolicy,
            nativeSetPolicy,
            nativeNodeCpuSet,
            nativeTouch};
    }

    // ------------------------------------------------------------------
    // Topology observation: COMPLETE masks (libnuma bitmaps and the
    // kernel's own node attributes), never one-word assumptions.
    // ------------------------------------------------------------------

    /// Allowed memory nodes as reported by the kernel policy mask
    /// (complete bitmask via libnuma).
    std::vector<int> allowedMemoryNodes()
    {
        struct bitmask *mask = numa_get_mems_allowed();
        if (!mask)
            return {};
        std::vector<int> nodes;
        for (unsigned int node = 0; node < mask->size; ++node)
            if (numa_bitmask_isbitset(mask, node))
                nodes.push_back(static_cast<int>(node));
        numa_free_nodemask(mask);
        return nodes;
    }

    /// Parse a kernel node list ("0", "0-3", "0-3,5", "0-3,5-7").
    std::vector<int> parseNodeList(const std::string &text)
    {
        std::vector<int> nodes;
        size_t position = 0;
        while (position < text.size())
        {
            const size_t comma = text.find(',', position);
            const std::string token = text.substr(
                position, comma == std::string::npos
                              ? std::string::npos
                              : comma - position);
            const size_t dash = token.find('-');
            if (dash == std::string::npos)
            {
                if (!token.empty())
                    nodes.push_back(std::stoi(token));
            }
            else
            {
                const int lo = std::stoi(token.substr(0, dash));
                const int hi = std::stoi(token.substr(dash + 1));
                for (int node = lo; node <= hi; ++node)
                    nodes.push_back(node);
            }
            if (comma == std::string::npos)
                break;
            position = comma + 1;
        }
        return nodes;
    }

    /// The RAM-bearing subset of the allowed nodes, per the kernel's
    /// own has_normal_memory attribute (memoryless allowed nodes hold
    /// no RAM and never qualify).
    std::vector<int> ramBearingAllowedNodes(
        const std::vector<int> &allowed)
    {
        FILE *file =
            fopen("/sys/devices/system/node/has_normal_memory", "r");
        if (!file)
            return {};
        char line[512] = {};
        const bool read = fgets(line, sizeof(line), file) != nullptr;
        fclose(file);
        if (!read)
            return {};
        std::vector<int> result;
        for (const int node : parseNodeList(line))
            if (std::find(allowed.begin(), allowed.end(), node) !=
                allowed.end())
                result.push_back(node);
        return result;
    }

    /// A node's CPUs that are ALSO usable under the process's
    /// permitted affinity (a node whose CPUs are all outside the
    /// permitted set cannot host the first-touch thread).
    cpu_set_t usableNodeCpus(int node, bool &usable)
    {
        cpu_set_t node_mask{};
        cpu_set_t permitted{};
        cpu_set_t usable_mask{};
        CPU_ZERO(&node_mask);
        CPU_ZERO(&permitted);
        CPU_ZERO(&usable_mask);
        usable = false;
        if (llaminar2::firstTouchDefaultOps().nodeCpuSet(
                nullptr, node, &node_mask) != 0)
            return usable_mask;
        if (::sched_getaffinity(0, sizeof(permitted), &permitted) != 0)
            return usable_mask;
        CPU_AND(&usable_mask, &node_mask, &permitted);
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &usable_mask))
            {
                usable = true;
                break;
            }
        return usable_mask;
    }
} // namespace

// ---------------------------------------------------------------------------
// Query-validator negatives: a rejected query can never certify a
// proof. Per-page errors, empty extents, and mixed placements must all
// be rejected by the shared validator.
// ---------------------------------------------------------------------------

TEST(FirstTouchBoundaryNative, QueryValidatorRejectsPerPageErrors)
{
    // OWNED RESERVATION control with EXPLICIT fixture determinism. One
    // positively certified resident page, plus a NON-FIRST owned page
    // whose NONRESIDENCY is established by a CHECKED local operation
    // on that owned mapping (madvise MADV_DONTNEED). No cross-page
    // placement uniformity is assumed — two independent prefaults may
    // legitimately land on different nodes under an inherited
    // interleave policy — and no untouched-page nonresidency is
    // assumed (a larger anonymous/THP allocation could populate an
    // untouched page). The shared query must REJECT the whole query: a
    // per-page error is never a placement.
    Mapping mapping(2);
    ASSERT_TRUE(mapping.valid());
    // Prefault both pages (real residency; nodes may legitimately
    // differ — no equality assumption).
    auto *heads = static_cast<volatile uint8_t *>(mapping.get());
    heads[0] = 1;
    heads[page_bytes] = 1;
    // Positive certification of ONE resident page through the same
    // query+validator used everywhere.
    const PlacementQuery valid_query = queryPlacement(mapping.get(), 1);
    ASSERT_TRUE(valid_query.query_ok)
        << "the prefaulted first page must query successfully (errno "
        << valid_query.query_errno << ")";
    ASSERT_EQ(valid_query.count, 1u);
    ASSERT_GE(valid_query.node[0], 0);
    ASSERT_TRUE(everyPageOnNode(valid_query, valid_query.node[0]))
        << "the resident page must certify through the same validator "
           "before the rejection proof";
    // EXPLICITLY establish nonresidency of the NON-FIRST owned page
    // with a checked local operation on our own mapping: DONTNEED
    // zaps the private anonymous page, so the following query reports
    // a per-page -ENOENT for it deterministically.
    ASSERT_EQ(::madvise(
                  static_cast<uint8_t *>(mapping.get()) + page_bytes,
                  page_bytes, MADV_DONTNEED),
        0)
        << "madvise(MADV_DONTNEED) on the owned non-first page failed: "
        << std::strerror(errno);
    // The resident page still certifies ALONE after the zap.
    const PlacementQuery still_valid = queryPlacement(mapping.get(), 1);
    ASSERT_TRUE(still_valid.query_ok)
        << "the certified resident page must remain certifiable after "
           "the DONTNEED on the other page (errno "
        << still_valid.query_errno << ")";
    ASSERT_EQ(still_valid.count, 1u);
    ASSERT_TRUE(everyPageOnNode(still_valid, still_valid.node[0]));
    // Adding the zapped NON-FIRST page rejects the whole query with
    // the per-page ENOENT.
    const PlacementQuery query = queryPlacement(mapping.get(), 2);
    EXPECT_FALSE(query.query_ok)
        << "a query with a negative per-page status must be rejected";
    EXPECT_EQ(query.query_errno, ENOENT);
    EXPECT_EQ(query.count, 0u)
        << "a rejected query reports no placement observations";
}

TEST(FirstTouchBoundaryNative, QueryValidatorRejectsEmptyExtent)
{
    Mapping mapping(1);
    ASSERT_TRUE(mapping.valid());
    const PlacementQuery query = queryPlacement(mapping.get(), 0);
    EXPECT_FALSE(query.query_ok)
        << "an empty extent must be rejected";
    EXPECT_EQ(query.query_errno, EINVAL);
    // The shared predicate must reject it too — an empty query can
    // never certify "every page on node N".
    EXPECT_FALSE(everyPageOnNode(query, 0))
        << "the validator must reject an empty extent";
}

TEST(FirstTouchBoundaryNative, QueryValidatorRejectsMixedPlacement)
{
    // Fabricated MIXED placement (validator negative; the query
    // predicates must reject it regardless of how it arose): one page
    // on the target node, one elsewhere.
    PlacementQuery query;
    query.query_ok = true;
    query.count = 2;
    query.node[0] = 0;
    query.node[1] = 1;
    EXPECT_FALSE(everyPageOnNode(query, 0))
        << "a mixed placement must fail the validator for node 0";
    EXPECT_FALSE(everyPageOnNode(query, 1))
        << "a mixed placement must fail the validator for node 1";
    // Sanity of the predicate itself: a uniform placement passes.
    query.node[1] = 0;
    EXPECT_TRUE(everyPageOnNode(query, 0));
}

TEST(FirstTouchBoundaryNative, SingleNodeTransitionProof)
{
    // Any allowed topology with at least one RAM-bearing allowed node
    // whose CPUs intersect the permitted affinity (a single RAM node
    // suffices). The proof exercises the REAL production sequence with
    // REAL syscalls through the recording boundary.
    NativeGeometry geometry;
    ASSERT_NO_FATAL_FAILURE(requireNativeGeometry(geometry));
    ThreadStateGuard guard(geometry);
    ASSERT_TRUE(guard.captured())
        << "thread-state capture failed: nothing is mutated and the "
           "proof cannot proceed";

    const std::vector<int> allowed = allowedMemoryNodes();
    ASSERT_FALSE(allowed.empty())
        << "kernel reports no allowed memory nodes";

    int target_node = -1;
    for (const int node : ramBearingAllowedNodes(allowed))
    {
        ASSERT_NODE_IN_GEOMETRY(geometry, node);
        bool usable = false;
        (void)usableNodeCpus(node, usable);
        if (usable)
        {
            target_node = node;
            break;
        }
    }
    ASSERT_GE(target_node, 0)
        << "no RAM-bearing allowed node with a CPU usable under the "
           "process's permitted affinity exists for the native "
           "transition proof";
    ASSERT_NODE_IN_GEOMETRY(geometry, target_node);

    // Inherited, distinguishable policy: INTERLEAVE over the target
    // node (valid on a single-RAM-node machine) — intentionally
    // different from the sequence's MPOL_BIND so restoration is
    // observable. All mask storage is allocated HERE, before any
    // mutation, through the bounds-checked shared helpers.
    std::vector<unsigned long> inherited_nodes =
        strictBindMask(geometry, target_node);
    ASSERT_EQ(::set_mempolicy(MPOL_INTERLEAVE, inherited_nodes.data(),
                  geometry.bits),
        0)
        << "cannot establish an inherited distinguishable policy: "
        << std::strerror(errno);

    constexpr size_t kPages = 5;
    Mapping pages(kPages);
    ASSERT_TRUE(pages.valid()) << std::strerror(errno);

    // Non-zero content so the real touch's per-page store is
    // observable; the production touch performs ONE store per page
    // (the placement operation). Full-range zeroing belongs to the
    // subsequent transport, not to the sequence.
    memset(pages.get(), 0xAA, pages.bytes());

    NativeRecorder recorder(geometry); // probe buffers preallocated
    const auto ops = nativeRecordingOps(recorder);
    const llaminar2::FirstTouchGeometry sequence_geometry{
        std::max(target_node, geometry.max_node)};
    const auto result = llaminar2::firstTouchPopulateRange(
        pages.get(), pages.bytes(), page_bytes, target_node,
        sequence_geometry, ops);

    ASSERT_TRUE(result.completed)
        << "populate phase " << static_cast<int>(result.phase)
        << " errno " << result.failure_errno;
    ASSERT_EQ(result.phase, llaminar2::FirstTouchPhase::Completed);
    EXPECT_FALSE(recorder.overflow);

    // Exactly ONE touch observation, with the EXACT address/length and
    // the EXACT production stride.
    ASSERT_EQ(recorder.touch_count, 1u);
    const auto &touch = recorder.touches[0];
    EXPECT_EQ(touch.base, pages.get());
    EXPECT_EQ(touch.bytes, pages.bytes());
    EXPECT_EQ(touch.stride, page_bytes);

    // REAL kernel-policy probes around the actual touch: the policy
    // was MPOL_BIND{target} immediately before AND immediately after
    // the touch (the touch executed inside the bind window), compared
    // over the COMPLETE nodemask words.
    EXPECT_EQ(touch.policy_before_mode, MPOL_BIND)
        << "kernel probe before the touch must report MPOL_BIND";
    EXPECT_EQ(touch.policy_after_mode, MPOL_BIND)
        << "kernel probe after the touch must still report MPOL_BIND";
    EXPECT_TRUE(llaminar2::firstTouchMaskHasNode(
        recorder.probe_before.data(), geometry.words, target_node))
        << "the pre-touch probe mask must contain the target node";
    EXPECT_TRUE(llaminar2::firstTouchMaskHasNode(
        recorder.probe_after.data(), geometry.words, target_node))
        << "the post-touch probe mask must contain the target node";
    // The bind names EXACTLY the target node: the probe mask equals
    // the single-bit target mask over all words.
    const std::vector<unsigned long> target_mask =
        strictBindMask(geometry, target_node);
    EXPECT_TRUE(llaminar2::firstTouchMasksEqual(
        recorder.probe_before.data(), target_mask.data(),
        geometry.words))
        << "the pre-touch policy mask must be exactly {target}";
    EXPECT_TRUE(llaminar2::firstTouchMasksEqual(
        recorder.probe_after.data(), target_mask.data(),
        geometry.words))
        << "the post-touch policy mask must be exactly {target}";

    // Content: every page's first byte is zeroed (one store per page);
    // the untouched interior bytes keep the pre-touch sentinel.
    const auto *bytes = static_cast<const uint8_t *>(pages.get());
    for (size_t page = 0; page < kPages; ++page)
        ASSERT_EQ(bytes[page * page_bytes], 0)
            << "page " << page << " was not touched";
    ASSERT_EQ(bytes[1], 0xAA)
        << "interior bytes belong to the transport, not the sequence";

    // Restoration is QUERIED AND ASSERTED for BOTH components over the
    // COMPLETE masks: the kernel reports the inherited policy AND the
    // exact original affinity again (probe outside the sequence).
    EXPECT_TRUE(threadStateIs(geometry, MPOL_INTERLEAVE,
        inherited_nodes, guard.original_affinity()))
        << "the sequence must restore the inherited policy and the "
           "exact captured affinity";

    // Verify capture, bind, touch, and both restoration setters followed by
    // the production guard's policy and affinity readbacks.
    static constexpr const char *kExpected[] = {
        "getaffinity", "getpolicy", "nodecpuset", "setaffinity",
        "setpolicy", "touch", "setpolicy", "setaffinity",
        "getpolicy", "getaffinity"};
    ASSERT_EQ(recorder.call_count, std::size(kExpected))
        << "unexpected boundary call count";
    for (size_t i = 0; i < std::size(kExpected); ++i)
    {
        EXPECT_STREQ(recorder.calls[i].name, kExpected[i]);
        EXPECT_EQ(recorder.calls[i].result, 0)
            << recorder.calls[i].name << " failed with errno "
            << recorder.calls[i].call_errno;
    }

    guard.restoreAndVerify();
}

TEST(FirstTouchBoundaryNative, RemoteRAMPlacementMultiRAMNode)
{
    // The genuine remote-RAM obligation under the memory-node
    // contract. Eligibility: >= 2 allowed RAM-bearing nodes AND a
    // target with a CPU usable under the permitted affinity. On
    // machines without the prerequisite this test FAILS explicitly —
    // absent hardware is an unqualified environment blocker, never a
    // skip-as-pass.
    NativeGeometry geometry;
    ASSERT_NO_FATAL_FAILURE(requireNativeGeometry(geometry));
    ThreadStateGuard guard(geometry);
    ASSERT_TRUE(guard.captured())
        << "thread-state capture failed: nothing is mutated and the "
           "proof cannot proceed";

    const std::vector<int> allowed = allowedMemoryNodes();
    ASSERT_FALSE(allowed.empty())
        << "kernel reports no allowed memory nodes";

    const std::vector<int> ram_nodes = ramBearingAllowedNodes(allowed);
    ASSERT_FALSE(ram_nodes.empty())
        << "cannot resolve RAM-bearing allowed nodes (allowed mask has "
        << allowed.size() << " entries)";
    ASSERT_GE(ram_nodes.size(), 2u)
        << "PREREQUISITE NOT MET: the memory-node contract requires at "
           "least two allowed RAM-bearing nodes (allowed mask has "
        << allowed.size()
        << " entries; RAM-bearing among them: " << ram_nodes.size()
        << "). This machine is EXPLICITLY UNQUALIFIED for the "
           "remote-RAM placement obligation; this failure is the "
           "environment blocker, not a skip.";

    int target_node = -1;
    for (const int node : ram_nodes)
    {
        ASSERT_NODE_IN_GEOMETRY(geometry, node);
        bool usable = false;
        (void)usableNodeCpus(node, usable);
        if (usable)
        {
            target_node = node;
            break;
        }
    }
    ASSERT_GE(target_node, 0)
        << "PREREQUISITE NOT MET: no RAM-bearing allowed node has a "
           "CPU usable under the process's permitted affinity for the "
           "first-touch thread";
    ASSERT_NODE_IN_GEOMETRY(geometry, target_node);

    const int remote_node = *std::find_if(ram_nodes.begin(),
        ram_nodes.end(),
        [target_node](int node) { return node != target_node; });
    ASSERT_NODE_IN_GEOMETRY(geometry, remote_node);
    const std::vector<unsigned long> remote_mask =
        strictBindMask(geometry, remote_node);
    const llaminar2::FirstTouchGeometry sequence_geometry{
        std::max(std::max(target_node, remote_node),
            geometry.max_node)};
    constexpr size_t kPages = 4;

    // ------------------------------------------------------------------
    // PREREQUISITE POSITIVE CONTROL: the inherited STRICT remote bind
    // must ACTUALLY place pages remotely. Only if this control
    // establishes remote placement do the override proofs below mean
    // anything.
    // ------------------------------------------------------------------
    ASSERT_EQ(setStrictBindMasked(remote_mask.data(), geometry), 0)
        << "cannot establish the strict remote MPOL_BIND control: "
        << std::strerror(errno);
    {
        Mapping control_pages(kPages);
        ASSERT_TRUE(control_pages.valid());
        volatile auto *heads =
            static_cast<volatile uint8_t *>(control_pages.get());
        for (size_t offset = 0; offset < control_pages.bytes();
             offset += page_bytes)
            heads[offset] = 1; // plain touch under the inherited bind
        const PlacementQuery query =
            queryPlacement(control_pages.get(), kPages);
        ASSERT_TRUE(everyPageOnNode(query, remote_node))
            << "PREREQUISITE CONTROL FAILED: the strict remote MPOL_BIND "
               "did not place the control pages remotely; the "
               "remote-placement obligation cannot be certified on this "
               "evidence";
    }

    // ------------------------------------------------------------------
    // Production proof: with the strict remote bind still inherited,
    // the production sequence must place its fresh mapping on the
    // TARGET node (the override) and restore the inherited state.
    // ------------------------------------------------------------------
    {
        Mapping proof_pages(kPages);
        ASSERT_TRUE(proof_pages.valid());
        const auto proof = llaminar2::firstTouchPopulateRange(
            proof_pages.get(), proof_pages.bytes(), page_bytes,
            target_node, sequence_geometry,
            llaminar2::firstTouchDefaultOps());
        ASSERT_TRUE(proof.completed)
            << "production populate failed in phase "
            << static_cast<int>(proof.phase) << " errno "
            << proof.failure_errno;
        const PlacementQuery query =
            queryPlacement(proof_pages.get(), kPages);
        EXPECT_TRUE(everyPageOnNode(query, target_node))
            << "production first touch must place every page on the "
               "target node despite the inherited remote bind";
        EXPECT_TRUE(threadStateIs(geometry, MPOL_BIND, remote_mask,
            guard.original_affinity()))
            << "the inherited strict remote bind and the original "
               "affinity must be restored after the production proof";
    }

    // ------------------------------------------------------------------
    // PUBLIC allocator coverage: the production COMPOSITION
    // (revocation + sequence + move_pages certification), not only the
    // extracted inner routine, must override the inherited remote bind.
    // ------------------------------------------------------------------
    {
        ASSERT_EQ(setStrictBindMasked(remote_mask.data(), geometry), 0)
            << "cannot re-establish the inherited strict remote bind: "
            << std::strerror(errno);
        Mapping public_pages(kPages);
        ASSERT_TRUE(public_pages.valid());
        ASSERT_TRUE(llaminar2::NUMAAllocator::instance()
                        .prepareExternalReceiveRangeOnNode(
                            public_pages.get(), public_pages.bytes(),
                            target_node))
            << "the public allocator composition failed under the "
               "inherited strict remote bind";
        const PlacementQuery query =
            queryPlacement(public_pages.get(), kPages);
        EXPECT_TRUE(everyPageOnNode(query, target_node))
            << "the public allocator composition must place every page "
               "on the target node despite the inherited remote bind";
        EXPECT_TRUE(threadStateIs(geometry, MPOL_BIND, remote_mask,
            guard.original_affinity()))
            << "the public allocator composition must restore the "
               "inherited strict remote bind and the original affinity";
    }

    // Preserve the allocation-path (not just transport-range) proof under
    // ARM's inherited preferred-policy contract, using the observed RAM nodes
    // qualified by the strict-bind control above.
    {
        ASSERT_EQ(::set_mempolicy(MPOL_PREFERRED, remote_mask.data(), geometry.bits), 0)
            << "cannot establish a conflicting preferred policy: " << std::strerror(errno);
        ASSERT_TRUE(threadStateIs(geometry, MPOL_PREFERRED, remote_mask,
            guard.original_affinity()));
        auto &allocator = llaminar2::NUMAAllocator::instance();
        const size_t bytes = kPages * page_bytes;
        const auto free_allocation = [&allocator, bytes](void *ptr) { allocator.free(ptr, bytes); };
        const std::unique_ptr<void, decltype(free_allocation)> allocation(
            allocator.allocateOnNode(bytes, target_node, page_bytes), free_allocation);
        ASSERT_NE(allocation, nullptr);
        EXPECT_TRUE(everyPageOnNode(queryPlacement(allocation.get(), kPages), target_node));
        EXPECT_TRUE(threadStateIs(geometry, MPOL_PREFERRED, remote_mask,
            guard.original_affinity()));
    }

    // ------------------------------------------------------------------
    // Omit-bind mutation negative: under the same inherited strict
    // remote bind, a test-local ops variant omits ONLY the TARGET bind
    // (the bind whose mask names exactly the target node, compared
    // over ALL words); every other call — including restoration of
    // the inherited MPOL_BIND{remote} — stays REAL. Detection uses
    // the SAME everyPageOnNode validator as the production proof: the
    // mutant must place every page REMOTELY (validator holds for the
    // remote node) and must be REJECTED for the target node.
    // ------------------------------------------------------------------
    {
        ASSERT_EQ(setStrictBindMasked(remote_mask.data(), geometry), 0)
            << "cannot re-establish the inherited strict remote bind: "
            << std::strerror(errno);
        // Preallocated target mask (complete geometry, before any
        // mutation in this case).
        std::vector<unsigned long> target_words =
            strictBindMask(geometry, target_node);

        const auto omit_policy = [](
            void *context, int mode, const unsigned long *nodemask,
            unsigned long max_nodes) -> int {
            const auto *words =
                static_cast<const std::vector<unsigned long> *>(
                    context);
            if (mode == MPOL_BIND && nodemask)
            {
                // Omit ONLY the bind whose mask names exactly the
                // target node over ALL words; restoration of the
                // inherited remote MPOL_BIND stays REAL.
                const size_t words_available = max_nodes >= 2
                    ? (static_cast<size_t>(max_nodes) - 2) / llaminar2::kFirstTouchMaskWordBits + 1
                    : 0;
                bool is_target =
                    words_available == words->size();
                for (size_t word = 0;
                     is_target && word < words_available; ++word)
                    if (nodemask[word] != (*words)[word])
                        is_target = false;
                if (is_target)
                    return 0; // the omitted target bind
            }
            return llaminar2::firstTouchDefaultOps().setMemoryPolicy(
                nullptr, mode, nodemask, max_nodes);
        };

        // The omit variant carries its own context (the target-node
        // mask); the const members require complete construction.
        const llaminar2::FirstTouchBoundaryOps bound_omit_ops{
            &target_words,
            llaminar2::firstTouchDefaultOps().getThreadAffinity,
            llaminar2::firstTouchDefaultOps().setThreadAffinity,
            llaminar2::firstTouchDefaultOps().getMemoryPolicy,
            omit_policy,
            llaminar2::firstTouchDefaultOps().nodeCpuSet,
            llaminar2::firstTouchDefaultOps().touchRange};

        Mapping mutant_pages(kPages);
        ASSERT_TRUE(mutant_pages.valid());
        const auto mutant = llaminar2::firstTouchPopulateRange(
            mutant_pages.get(), mutant_pages.bytes(), page_bytes,
            target_node, sequence_geometry, bound_omit_ops);
        ASSERT_TRUE(mutant.completed)
            << "the omit-bind variant still completes (the omission is "
               "silent to the sequence)";
        const PlacementQuery query =
            queryPlacement(mutant_pages.get(), kPages);
        // DETECTION through the SAME shared validator: with the target
        // bind omitted the pages stayed REMOTE (the inherited strict
        // bind owns placement) — the validator HOLDS for the remote
        // node and is REJECTED for the target node.
        EXPECT_TRUE(everyPageOnNode(query, remote_node))
            << "with the target bind omitted, every page must stay on "
               "the inherited remote node";
        EXPECT_FALSE(everyPageOnNode(query, target_node))
            << "the SAME validator that certifies production placement "
               "must reject the mutant for the target node";
        EXPECT_TRUE(threadStateIs(geometry, MPOL_BIND, remote_mask,
            guard.original_affinity()))
            << "the omit-bind variant's REAL restoration must "
               "re-establish the inherited remote bind and the "
               "original affinity";
    }

    guard.restoreAndVerify();
}
