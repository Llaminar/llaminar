/**
 * @file Test__NUMAFirstTouchBoundary.cpp
 * @brief Deterministic device-free state-machine/ordering tests for the
 *        ONE production first-touch populate sequence through the typed
 *        OS boundary.
 *
 * Every test binds its OWN boundary instance with per-instance recording
 * context (no mutable global or TLS recorder). The recording instance
 * implements a tiny in-test thread state machine (affinity + memory
 * policy) with FIXED preallocated bounded storage, so no recording
 * activity can allocate or throw during the sequence. The tests prove,
 * deterministically and without any kernel/hardware dependency (the
 * topology geometry is supplied explicitly through FirstTouchGeometry):
 *   - the exact operation order acquire/bind -> touch(exact span) ->
 *     restore -> independent exact readback;
 *   - the touch executes under the bound MPOL_BIND{target} policy;
 *   - every precondition and capture/resolution failure returns with
 *     the thread exactly unmutated;
 *   - every bind failure returns ONLY after both owned components were
 *     restored independently, with the failing call's errno captured
 *     at the call site (success-path activity deliberately clobbers
 *     errno and must not corrupt the attribution);
 *   - a callback exception in the FIRST-mutation window still restores
 *     both components during unwind;
 *   - restoration failure, failed readback or silent state loss is terminal;
 *   - compound failures preserve the primary operation and independent cleanup evidence;
 *   - validators reject omitted bind and touch-before-bind observations.
 *
 * Real-kernel call-through proof (probes at the actual touch, content
 * match, move_pages certification) lives in the explicit native
 * integration/preflight regressions, NOT here.
 */

#include <gtest/gtest.h>
#include "memory/FirstTouchBoundary.h"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <csignal>
#include <cstring>
#include <sched.h>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <numaif.h>
#include <unistd.h>

#include <sys/mman.h>
#include <sys/wait.h>

using llaminar2::FirstTouchBoundaryOps;
using llaminar2::FirstTouchGeometry;
using llaminar2::FirstTouchPhase;
using llaminar2::firstTouchMaskEmpty;
using llaminar2::firstTouchMaskHasNode;
using llaminar2::firstTouchMaskSetNode;
using llaminar2::firstTouchMaskWordCount;
using llaminar2::firstTouchMasksEqual;
using llaminar2::firstTouchPopulateRange;

namespace
{
    constexpr size_t kMaxEvents = 32;
    using PolicyMask = std::array<unsigned long, 2>;
    enum class PolicyRestoreEffect { Exact, Omit, WrongMode, DropHighNode };
    enum class AffinityRestoreEffect { Exact, WrongLowMask, DropHighCpu };

    /// The per-instance observer state machine: records every boundary
    /// call with its arguments AND models the thread's affinity/policy
    /// state (the fake "kernel"). FIXED bounded storage, entirely
    /// outside the observed span; no recording activity allocates.
    struct RecordingState
    {
        enum class Kind
        {
            GetAffinity,
            SetAffinity,
            GetPolicy,
            SetPolicy,
            NodeCpuSet,
            Touch
        };
        struct Event
        {
            Kind kind;
            int mode = -1;
            PolicyMask nodes{};
            unsigned long maxnode = 0;
            bool has_mask = false;
            int node = -1;
            const volatile void *base = nullptr; // Touch base
            size_t bytes = 0;                    // Touch span
            size_t page_bytes = 0;               // Touch stride
            int policy_at_touch = -1;            // policy mode when Touch ran
            PolicyMask nodes_at_touch{};
        };
        Event events[kMaxEvents];
        size_t event_count = 0;
        bool overflow = false; // an event beyond kMaxEvents was dropped

        cpu_set_t affinity{};       // current thread affinity state
        int policy_mode = MPOL_DEFAULT;
        PolicyMask policy_nodes{};
        unsigned long expected_maxnode = 5; // logical bits 0–3 plus the native operand's extra unit
        size_t policy_words = 1;
        int node_cpus_word = 0;     // CPUs reported for a node
        bool node_cpus_empty = false; // NodeCpuSet reports no CPU

        // Failure injection (all per-instance, none global).
        int fail_get_affinity_errno = 0;
        int fail_get_policy_errno = 0;
        int fail_readback_policy_errno = 0;
        bool throw_on_affinity_readback = false;
        int fail_node_cpu_set_errno = 0;
        int fail_bind_affinity_errno = 0;  // the affinity BIND fails
        int fail_bind_errno = 0;           // the MPOL_BIND set fails
        bool fail_policy_restore = false;
        // Fail the Nth setThreadAffinity call (1-based; 2 = the
        // explicit restore after a successful bind). The injected
        // errno is EPERM — deliberately distinguishable from the
        // policy-part EIO so fatal diagnostics identify the exact
        // failing part.
        int fail_set_affinity_on_call = 0;
        // Deliberately clobber errno after a SUCCESSFUL operation: the
        // sequence must capture failure errnos at their own call sites.
        int clobber_errno_on_success = 0;
        // Exception injection windows.
        bool throw_on_set_affinity = false;
        bool throw_on_touch = false;
        bool throw_after_policy_restore = false;
        bool throw_after_affinity_restore = false;
        PolicyRestoreEffect policy_restore = PolicyRestoreEffect::Exact;
        AffinityRestoreEffect affinity_restore = AffinityRestoreEffect::Exact;

        int get_policy_calls = 0;
        int get_affinity_calls = 0;
        int set_affinity_calls = 0;
        bool policy_bind_pending = false;

        // Retained bounded call evidence for the fatal-path controls:
        // an anonymous owned descriptor (test-local, acquired BEFORE
        // any mutation) written with a robust errno-preserving write
        // and a stack buffer — no allocation, no throw, no path to
        // unlink. Death-test children inherit the descriptor, so the
        // parent can prove BOTH restoration attempts executed before
        // the fatal termination. A failed or truncated recording is
        // VISIBLE (trace_failed), never silent.
        int trace_fd = -1;
        bool trace_failed = false;

        // DEATH-TEST FIXTURES set this: inside a death-test child a
        // RECORDING failure must become a DISTINCT immediate child
        // failure (exit 83 + an unambiguous stderr line) that can
        // NEVER satisfy the expected fatal-death regex — the parent's
        // EXPECT_DEATH then fails and the unusable evidence is
        // rejected, never silently accepted.
        bool abort_on_trace_failure = false;

        void trace(const char *text)
        {
            const bool ok = traceWrite(text);
            if (!ok)
            {
                trace_failed = true; // visible in-process
                if (abort_on_trace_failure)
                {
                    (void)dprintf(2, "trace-recording-failed\n");
                    _exit(83); // distinct child failure
                }
            }
        }

        /// Errno-preserving robust write. A ZERO-PROGRESS write is
        /// rejected (never looped on), EINTR is retried, short writes
        /// are completed, and every failure is reported to the caller
        /// without corrupting the errno under observation.
        bool traceWrite(const char *text)
        {
            if (trace_fd < 0)
                return false; // failed acquisition: visible
            const int saved_errno = errno;
            char line[256];
            const int formatted =
                snprintf(line, sizeof(line), "%s\n", text);
            bool ok = formatted > 0 &&
                static_cast<size_t>(formatted) < sizeof(line);
            if (ok)
            {
                size_t done = 0;
                const size_t length = static_cast<size_t>(formatted);
                while (done < length)
                {
                    const ssize_t written = ::write(
                        trace_fd, line + done, length - done);
                    if (written < 0)
                    {
                        if (errno == EINTR)
                            continue;
                        ok = false; // error outcome: visible
                        break;
                    }
                    if (written == 0)
                    {
                        ok = false; // ZERO PROGRESS: reject, never
                        break;      // loop forever
                    }
                    done += static_cast<size_t>(written);
                }
            }
            errno = saved_errno;
            return ok;
        }

        void record(const Event &event)
        {
            if (event_count < kMaxEvents)
                events[event_count++] = event;
            else
                overflow = true;
        }

        void record(Kind kind) { record(Event{kind}); }

        size_t count() const { return event_count; }
        const Event &at(size_t index) const { return events[index]; }
    };

    cpu_set_t maskFromWord(unsigned long word)
    {
        cpu_set_t mask{};
        CPU_ZERO(&mask);
        for (int bit = 0; bit < 64; ++bit)
            if (word & (1ul << bit))
                CPU_SET(bit, &mask);
        return mask;
    }

    unsigned long wordFromMask(const cpu_set_t &mask)
    {
        unsigned long word = 0;
        for (int bit = 0; bit < 64; ++bit)
            if (CPU_ISSET(bit, &mask))
                word |= 1ul << bit;
        return word;
    }

    /// Complete cpu_set_t equality over EVERY CPU in CPU_SETSIZE: a
    /// mask differing only in a high CPU (>= 64) is a mismatch, which
    /// a low-word comparison would silently accept.
    bool affinityEquals(const cpu_set_t &a, const cpu_set_t &b)
    {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &a) != CPU_ISSET(cpu, &b))
                return false;
        return true;
    }

    void clobberErrnoOnSuccess(RecordingState &state)
    {
        if (state.clobber_errno_on_success)
            errno = state.clobber_errno_on_success;
    }

    int recordGetAffinity(void *context, cpu_set_t *mask)
    {
        auto &state = *static_cast<RecordingState *>(context);
        state.record(RecordingState::Kind::GetAffinity);
        if (++state.get_affinity_calls == 1)
        {
            if (state.fail_get_affinity_errno)
            {
                errno = state.fail_get_affinity_errno;
                return -1;
            }
        }
        else
        {
            if (state.throw_on_affinity_readback)
            {
                state.trace("affinity_readback:exception");
                errno = ENOSPC;
                throw std::runtime_error("affinity readback threw");
            }
            state.trace("affinity_readback:ok");
        }
        *mask = state.affinity;
        clobberErrnoOnSuccess(state);
        return 0;
    }

    int recordSetAffinity(void *context, const cpu_set_t *mask)
    {
        auto &state = *static_cast<RecordingState *>(context);
        state.record(RecordingState::Kind::SetAffinity);
        const bool binding = ++state.set_affinity_calls == 1;
        if (binding && state.throw_on_set_affinity)
        {
            state.affinity = *mask;
            state.trace("affinity_call:1:throw");
            throw std::runtime_error("injected first-mutation failure");
        }
        if ((binding && state.fail_bind_affinity_errno) ||
            state.fail_set_affinity_on_call == state.set_affinity_calls)
        {
            const bool bind_failure = binding && state.fail_bind_affinity_errno != 0;
            errno = bind_failure ? state.fail_bind_affinity_errno : EPERM;
            state.trace(bind_failure ? "affinity_call:bind:fail"
                                      : "affinity_call:restore:fail:1");
            return -1;
        }
        state.affinity = *mask;
        if (binding)
        {
            state.policy_bind_pending = true;
            state.trace("affinity_call:1:ok");
        }
        else
        {
            switch (state.affinity_restore)
            {
            case AffinityRestoreEffect::Exact:
                break;
            case AffinityRestoreEffect::WrongLowMask:
                state.affinity = maskFromWord(0x0f0ful);
                break;
            case AffinityRestoreEffect::DropHighCpu:
                static_assert(CPU_SETSIZE > 100);
                CPU_CLR(100, &state.affinity);
                break;
            }
            if (state.throw_after_affinity_restore)
            {
                state.trace("affinity_call:restore:exception");
                errno = ENOSPC;
                throw std::runtime_error("affinity restore threw after mutation");
            }
            state.trace("affinity_call:restore:ok");
        }
        clobberErrnoOnSuccess(state);
        return 0;
    }

    int recordGetPolicy(void *context, int *mode, unsigned long *nodemask,
        unsigned long maxnode, void *addr, int flags)
    {
        auto &state = *static_cast<RecordingState *>(context);
        RecordingState::Event event{RecordingState::Kind::GetPolicy};
        event.maxnode = maxnode;
        event.has_mask = nodemask != nullptr;
        state.record(event);
        const bool capture = ++state.get_policy_calls == 1;
        if (!mode || !nodemask || maxnode != state.expected_maxnode || addr || flags ||
            state.policy_words == 0 || state.policy_words > state.policy_nodes.size())
        {
            errno = EINVAL;
            return -1;
        }
        // Injections can leave plausible outputs behind on error. Only a
        // successful call is comparable.
        *mode = state.policy_mode;
        for (size_t word = 0; word < state.policy_words; ++word)
            nodemask[word] = state.policy_nodes[word];
        const int failure = capture ? state.fail_get_policy_errno : state.fail_readback_policy_errno;
        if (failure != 0)
        {
            if (!capture)
                state.trace("policy_readback:rejected");
            errno = failure;
            return -1;
        }
        if (!capture)
            state.trace("policy_readback:ok");
        clobberErrnoOnSuccess(state);
        return 0;
    }

    int recordSetPolicy(void *context, int mode,
        const unsigned long *nodemask, unsigned long maxnode)
    {
        auto &state = *static_cast<RecordingState *>(context);
        const bool binding = state.policy_bind_pending;
        state.policy_bind_pending = false;
        RecordingState::Event event{RecordingState::Kind::SetPolicy};
        event.mode = mode;
        event.maxnode = maxnode;
        event.has_mask = nodemask != nullptr;
        // The fixture binds literal wire extents independently of the
        // sequence, before reading the supplied storage.
        if ((nodemask && (maxnode != state.expected_maxnode ||
                         state.policy_words == 0 || state.policy_words > event.nodes.size())) ||
            (!nodemask && (maxnode != 0 || binding)))
        {
            state.record(event);
            errno = EINVAL;
            return -1;
        }
        if (nodemask)
            for (size_t word = 0; word < state.policy_words; ++word)
                event.nodes[word] = nodemask[word];
        state.record(event);
        if (binding && state.fail_bind_errno)
        {
            errno = state.fail_bind_errno;
            state.trace("policy_bind:fail");
            return -1;
        }
        if (!binding && state.fail_policy_restore)
        {
            errno = EIO;
            state.trace("policy_restore:fail:5");
            return -1;
        }
        if (binding || state.policy_restore != PolicyRestoreEffect::Omit)
        {
            state.policy_mode = mode;
            state.policy_nodes = event.nodes;
            if (!binding && state.policy_restore == PolicyRestoreEffect::WrongMode)
                state.policy_mode = MPOL_PREFERRED;
            if (!binding && state.policy_restore == PolicyRestoreEffect::DropHighNode)
                state.policy_nodes[1] &= ~1ul;
        }
        if (!binding)
        {
            if (state.throw_after_policy_restore)
            {
                state.trace("policy_restore:exception");
                errno = ENOSPC;
                throw std::runtime_error("policy restore threw after mutation");
            }
            state.trace("policy_restore:ok");
        }
        clobberErrnoOnSuccess(state);
        return 0;
    }

    int recordNodeCpuSet(void *context, int numa_node, cpu_set_t *mask)
    {
        auto &state = *static_cast<RecordingState *>(context);
        RecordingState::Event event{RecordingState::Kind::NodeCpuSet};
        event.node = numa_node;
        state.record(event);
        if (state.fail_node_cpu_set_errno)
        {
            errno = state.fail_node_cpu_set_errno;
            return -1;
        }
        CPU_ZERO(mask);
        if (!state.node_cpus_empty)
            CPU_SET(state.node_cpus_word, mask);
        clobberErrnoOnSuccess(state);
        return 0;
    }

    void recordTouch(void *context, volatile void *base, size_t bytes,
        size_t page_bytes)
    {
        auto &state = *static_cast<RecordingState *>(context);
        RecordingState::Event event{RecordingState::Kind::Touch};
        event.base = base;
        event.bytes = bytes;
        event.page_bytes = page_bytes;
        event.policy_at_touch = state.policy_mode;
        event.nodes_at_touch = state.policy_nodes;
        state.record(event);
        if (state.throw_on_touch)
        {
            state.trace("touch:exception");
            errno = ENOSPC;
            throw std::runtime_error("injected touch failure");
        }
    }

    /** @brief Bind the recorder and only the deliberately varied policy setter. */
    FirstTouchBoundaryOps recordingOps(RecordingState &state,
        int (*set_policy)(void *, int, const unsigned long *, unsigned long) = &recordSetPolicy)
    {
        return FirstTouchBoundaryOps{
            &state,
            &recordGetAffinity,
            &recordSetAffinity,
            &recordGetPolicy,
            set_policy,
            &recordNodeCpuSet,
            &recordTouch};
    }

    // ------------------------------------------------------------------
    // The shared validators: the SAME checks run for the positive proof
    // and for every mutation control (the controls must FAIL them while
    // the observer machinery stays intact).
    // ------------------------------------------------------------------

    struct Validation
    {
        bool ok = false;
        std::string failure;
    };

    /// Validate one populate record: exact order, exact touch span,
    /// touch under MPOL_BIND{target}, and FULL restoration — the final
    /// affinity must EQUAL the captured original mask (a nonempty but
    /// wrong mask is a failure, not a pass).
    Validation validatePopulateRecord(const RecordingState &state,
        const volatile void *expected_base, size_t expected_bytes,
        size_t expected_page_bytes, int target_node,
        int original_mode, unsigned long original_nodes,
        const cpu_set_t &original_affinity)
    {
        Validation out;
        static constexpr RecordingState::Kind expected[]{
            RecordingState::Kind::GetAffinity,
            RecordingState::Kind::GetPolicy,
            RecordingState::Kind::NodeCpuSet,
            RecordingState::Kind::SetAffinity,
            RecordingState::Kind::SetPolicy, // MPOL_BIND{target}
            RecordingState::Kind::Touch,
            RecordingState::Kind::SetPolicy, // restore
            RecordingState::Kind::SetAffinity,
            RecordingState::Kind::GetPolicy,
            RecordingState::Kind::GetAffinity};
        constexpr size_t expected_count = sizeof(expected) / sizeof(expected[0]);
        if (state.overflow || state.count() != expected_count)
        {
            out.failure = "event count " + std::to_string(state.count()) +
                " != " + std::to_string(expected_count);
            return out;
        }
        for (size_t i = 0; i < expected_count; ++i)
            if (state.at(i).kind != expected[i])
            {
                out.failure = "event " + std::to_string(i) +
                    " kind mismatch";
                return out;
            }
        // Bind must name exactly the target node.
        if (state.at(4).mode != MPOL_BIND ||
            state.at(4).nodes != PolicyMask{1ul << static_cast<unsigned long>(target_node), 0})
        {
            out.failure = "bind policy did not name the exact target node";
            return out;
        }
        // The touch must cover the EXACT span with the EXACT stride.
        if (state.at(5).base != expected_base ||
            state.at(5).bytes != expected_bytes ||
            state.at(5).page_bytes != expected_page_bytes)
        {
            out.failure = "touch span does not match the exact allocation";
            return out;
        }
        // The touch executed under the bound policy.
        if (state.at(5).policy_at_touch != MPOL_BIND ||
            state.at(5).nodes_at_touch != PolicyMask{1ul << static_cast<unsigned long>(target_node), 0})
        {
            out.failure =
                "touch executed under a policy other than MPOL_BIND{target}";
            return out;
        }
        // Restore must re-establish the exact inherited policy.
        if (state.at(6).mode != original_mode ||
            state.at(6).nodes != PolicyMask{original_nodes, 0})
        {
            out.failure = "policy restoration does not match the captured "
                          "inherited policy";
            return out;
        }
        if (state.policy_mode != original_mode ||
            state.policy_nodes != PolicyMask{original_nodes, 0})
        {
            out.failure = "final policy state is not the inherited policy";
            return out;
        }
        // Restore must re-establish the EXACT original affinity mask
        // over the COMPLETE cpu_set_t (a high-CPU-only mismatch is a
        // failure, never a pass).
        if (!affinityEquals(state.affinity, original_affinity))
        {
            out.failure = "final affinity does not equal the captured "
                          "original affinity mask (complete cpu_set_t "
                          "comparison)";
            return out;
        }
        out.ok = true;
        return out;
    }

    /// Standard fixture: inherited MPOL_INTERLEAVE{0}, affinity
    /// 0x00ff, node 1 reports CPU 5. Explicit test geometry (max node
    /// 3) keeps the sequence wholly topology-independent.
    struct Fixture
    {
        RecordingState state;
        static constexpr int kTargetNode = 1;
        static constexpr FirstTouchGeometry kGeometry{3};
        static constexpr unsigned long kOriginalAffinityWord = 0x00fful;

        Fixture()
        {
            state.policy_mode = MPOL_INTERLEAVE;
            state.policy_nodes = {1ul, 0};
            state.affinity = originalAffinity();
            state.node_cpus_word = 5;
        }

        static cpu_set_t originalAffinity()
        {
            return maskFromWord(kOriginalAffinityWord);
        }
    };
} // namespace

TEST(FirstTouchBoundary, PopulateSequenceOrdersBindsTouchesAndRestores)
{
    Fixture fixture;
    uint8_t pages[3 * 4096]{};
    const auto ops = recordingOps(fixture.state);
    const auto result = firstTouchPopulateRange(
        pages, sizeof(pages), 4096, Fixture::kTargetNode,
        Fixture::kGeometry, ops);

    ASSERT_TRUE(result.completed);
    ASSERT_EQ(result.phase, FirstTouchPhase::Completed);
    EXPECT_FALSE(fixture.state.overflow);
    const auto validation = validatePopulateRecord(
        fixture.state, pages, sizeof(pages), 4096, Fixture::kTargetNode,
        MPOL_INTERLEAVE, 1ul, Fixture::originalAffinity());
    ASSERT_TRUE(validation.ok) << validation.failure;
}

TEST(FirstTouchBoundary, PreconditionRejectsIncompleteOpsAndGeometry)
{
    {
        RecordingState state;
        uint8_t pages[4096]{};
        // An incomplete instance IS constructible (const members
        // forbid REBINDING, not incomplete construction); the sequence
        // must reject it at entry.
        const FirstTouchBoundaryOps incomplete{};
        const auto result = firstTouchPopulateRange(
            pages, sizeof(pages), 4096, 0, FirstTouchGeometry{3},
            incomplete);
        EXPECT_FALSE(result.completed);
        EXPECT_EQ(result.failure_errno, EINVAL);
        EXPECT_EQ(state.count(), 0u)
            << "no boundary interaction may happen on a rejected binding";
    }
    {
        Fixture fixture;
        uint8_t pages[4096]{};
        const auto ops = recordingOps(fixture.state);
        // Geometry below the target node is rejected.
        const auto result = firstTouchPopulateRange(
            pages, sizeof(pages), 4096, Fixture::kTargetNode,
            FirstTouchGeometry{0}, ops);
        EXPECT_FALSE(result.completed);
        EXPECT_EQ(result.failure_errno, EINVAL);
        EXPECT_EQ(fixture.state.count(), 0u);
    }
    {
        Fixture fixture;
        uint8_t pages[4096]{};
        const auto ops = recordingOps(fixture.state);
        const auto result = firstTouchPopulateRange(
            pages, sizeof(pages), 4096, Fixture::kTargetNode,
            FirstTouchGeometry{-1}, ops);
        EXPECT_FALSE(result.completed);
        EXPECT_EQ(result.failure_errno, EINVAL);
        EXPECT_EQ(fixture.state.count(), 0u);
    }
}

TEST(FirstTouchBoundary, CaptureAffinityFailureLeavesThreadUnmutated)
{
    Fixture fixture;
    fixture.state.fail_get_affinity_errno = EIO;
    uint8_t pages[4096]{};
    const auto ops = recordingOps(fixture.state);
    const auto result = firstTouchPopulateRange(
        pages, sizeof(pages), 4096, Fixture::kTargetNode,
        Fixture::kGeometry, ops);
    ASSERT_FALSE(result.completed);
    EXPECT_EQ(result.phase, FirstTouchPhase::CaptureAffinity);
    EXPECT_EQ(result.failure_errno, EIO);
    // Nothing after the failed capture may have executed.
    ASSERT_EQ(fixture.state.count(), 1u);
}

TEST(FirstTouchBoundary, CapturePolicyFailureLeavesThreadUnmutated)
{
    Fixture fixture;
    fixture.state.fail_get_policy_errno = EIO;
    uint8_t pages[4096]{};
    const auto ops = recordingOps(fixture.state);
    const auto result = firstTouchPopulateRange(
        pages, sizeof(pages), 4096, Fixture::kTargetNode,
        Fixture::kGeometry, ops);
    ASSERT_FALSE(result.completed);
    EXPECT_EQ(result.phase, FirstTouchPhase::CapturePolicy);
    EXPECT_EQ(result.failure_errno, EIO);
    // getaffinity succeeded but mutated nothing; no later call ran.
    ASSERT_EQ(fixture.state.count(), 2u);
    EXPECT_EQ(fixture.state.at(0).kind,
        RecordingState::Kind::GetAffinity);
    EXPECT_EQ(fixture.state.at(1).kind, RecordingState::Kind::GetPolicy);
}

TEST(FirstTouchBoundary, NodeCpuResolutionFailureIsAttributed)
{
    Fixture fixture;
    fixture.state.fail_node_cpu_set_errno = EIO;
    uint8_t pages[4096]{};
    const auto ops = recordingOps(fixture.state);
    const auto result = firstTouchPopulateRange(
        pages, sizeof(pages), 4096, Fixture::kTargetNode,
        Fixture::kGeometry, ops);
    ASSERT_FALSE(result.completed);
    EXPECT_EQ(result.phase, FirstTouchPhase::ResolveNodeCpus);
    EXPECT_EQ(result.failure_errno, EIO);
    // Nothing mutated: no SetAffinity/SetPolicy/Touch ran.
    for (size_t i = 0; i < fixture.state.count(); ++i)
    {
        EXPECT_NE(fixture.state.at(i).kind,
            RecordingState::Kind::SetAffinity);
        EXPECT_NE(fixture.state.at(i).kind,
            RecordingState::Kind::SetPolicy);
        EXPECT_NE(fixture.state.at(i).kind, RecordingState::Kind::Touch);
    }
}

TEST(FirstTouchBoundary, EmptyTargetCpuSetFailsWithoutMutation)
{
    Fixture fixture;
    fixture.state.node_cpus_empty = true; // node reports no CPU
    uint8_t pages[4096]{};
    const auto ops = recordingOps(fixture.state);
    const auto result = firstTouchPopulateRange(
        pages, sizeof(pages), 4096, Fixture::kTargetNode,
        Fixture::kGeometry, ops);
    ASSERT_FALSE(result.completed);
    EXPECT_EQ(result.phase, FirstTouchPhase::ResolveNodeCpus);
    EXPECT_EQ(result.failure_errno, EINVAL);
    for (size_t i = 0; i < fixture.state.count(); ++i)
        EXPECT_NE(fixture.state.at(i).kind, RecordingState::Kind::Touch);
}

TEST(FirstTouchBoundary, BindAffinityFailureRestoresStateAndPreservesErrno)
{
    Fixture fixture;
    fixture.state.fail_bind_affinity_errno = EIO;
    // Deliberately clobber errno after every SUCCESSFUL operation: the
    // bind errno must survive the restoration activity that follows it.
    fixture.state.clobber_errno_on_success = ENOENT;
    uint8_t pages[4096]{};
    const auto ops = recordingOps(fixture.state);
    const auto result = firstTouchPopulateRange(
        pages, sizeof(pages), 4096, Fixture::kTargetNode,
        Fixture::kGeometry, ops);
    ASSERT_FALSE(result.completed);
    EXPECT_EQ(result.phase, FirstTouchPhase::BindAffinity);
    // The bind errno survives the subsequent restoration activity.
    EXPECT_EQ(result.failure_errno, EIO);
    // Both owned components were restored independently (the policy was
    // never mutated; re-setting the original is the verified no-op that
    // proves full restoration on the return path).
    const size_t count = fixture.state.count();
    ASSERT_EQ(count, 8u);
    EXPECT_EQ(fixture.state.at(4).kind, RecordingState::Kind::SetPolicy);
    EXPECT_EQ(fixture.state.at(5).kind, RecordingState::Kind::SetAffinity);
    EXPECT_EQ(fixture.state.at(6).kind, RecordingState::Kind::GetPolicy);
    EXPECT_EQ(fixture.state.at(7).kind, RecordingState::Kind::GetAffinity);
    EXPECT_EQ(fixture.state.policy_mode, MPOL_INTERLEAVE);
    EXPECT_EQ(fixture.state.policy_nodes, (PolicyMask{1ul, 0}));
    EXPECT_TRUE(affinityEquals(fixture.state.affinity,
        Fixture::originalAffinity()));
    for (size_t i = 0; i < count; ++i)
        EXPECT_NE(fixture.state.at(i).kind, RecordingState::Kind::Touch);
}

TEST(FirstTouchBoundary, BindPolicyFailureRestoresBothPartsPreservesErrno)
{
    Fixture fixture;
    fixture.state.fail_bind_errno = EIO;
    // Successful restoration calls clobber errno; the captured bind
    // errno must be the one reported.
    fixture.state.clobber_errno_on_success = ENOENT;
    uint8_t pages[4096]{};
    const auto ops = recordingOps(fixture.state);
    const auto result = firstTouchPopulateRange(
        pages, sizeof(pages), 4096, Fixture::kTargetNode,
        Fixture::kGeometry, ops);
    ASSERT_FALSE(result.completed);
    EXPECT_EQ(result.phase, FirstTouchPhase::BindPolicy);
    EXPECT_EQ(result.failure_errno, EIO)
        << "the bind errno must survive restoration activity that "
           "clobbers errno";
    ASSERT_EQ(fixture.state.count(), 9u);
    EXPECT_EQ(fixture.state.at(3).kind,
        RecordingState::Kind::SetAffinity); // affinity bind succeeded
    EXPECT_EQ(fixture.state.at(4).kind, RecordingState::Kind::SetPolicy);
    EXPECT_EQ(fixture.state.at(5).kind,
        RecordingState::Kind::SetPolicy); // policy part restored
    EXPECT_EQ(fixture.state.at(6).kind,
        RecordingState::Kind::SetAffinity); // affinity part restored
    EXPECT_EQ(fixture.state.at(7).kind, RecordingState::Kind::GetPolicy);
    EXPECT_EQ(fixture.state.at(8).kind, RecordingState::Kind::GetAffinity);
    EXPECT_EQ(fixture.state.policy_mode, MPOL_INTERLEAVE);
    EXPECT_EQ(fixture.state.policy_nodes, (PolicyMask{1ul, 0}));
    EXPECT_TRUE(affinityEquals(fixture.state.affinity,
        Fixture::originalAffinity()));
    for (size_t i = 0; i < fixture.state.count(); ++i)
        EXPECT_NE(fixture.state.at(i).kind, RecordingState::Kind::Touch);
}

TEST(FirstTouchBoundary, FirstMutationExceptionWindowStillRestores)
{
    // A callback exception in the FIRST-mutation window (the armed
    // guard exists and the affinity bind has REALLY mutated the fake
    // thread state before throwing) must restore both owned
    // components during unwind — no exception can bypass restoration,
    // and the recovery is EXACT (not merely "some affinity").
    Fixture fixture;
    fixture.state.throw_on_set_affinity = true;
    uint8_t pages[4096]{};
    const auto ops = recordingOps(fixture.state);
    bool threw = false;
    try
    {
        (void)firstTouchPopulateRange(
            pages, sizeof(pages), 4096, Fixture::kTargetNode,
            Fixture::kGeometry, ops);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    ASSERT_TRUE(threw);
    const size_t count = fixture.state.count();
    ASSERT_EQ(count, 8u);
    EXPECT_EQ(fixture.state.at(4).kind, RecordingState::Kind::SetPolicy);
    EXPECT_EQ(fixture.state.at(5).kind, RecordingState::Kind::SetAffinity);
    EXPECT_EQ(fixture.state.at(6).kind, RecordingState::Kind::GetPolicy);
    EXPECT_EQ(fixture.state.at(7).kind, RecordingState::Kind::GetAffinity);
    EXPECT_EQ(fixture.state.policy_mode, MPOL_INTERLEAVE);
    EXPECT_EQ(fixture.state.policy_nodes, (PolicyMask{1ul, 0}));
    EXPECT_TRUE(affinityEquals(fixture.state.affinity,
        Fixture::originalAffinity()));
    // No touch may have executed.
    for (size_t i = 0; i < count; ++i)
        EXPECT_NE(fixture.state.at(i).kind, RecordingState::Kind::Touch);
}

TEST(FirstTouchBoundary, UnwindOutOfTouchStillRestores)
{
    Fixture fixture;
    fixture.state.throw_on_touch = true;
    uint8_t pages[4096]{};
    const auto ops = recordingOps(fixture.state);
    bool threw = false;
    try
    {
        (void)firstTouchPopulateRange(
            pages, sizeof(pages), 4096, Fixture::kTargetNode,
            Fixture::kGeometry, ops);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    ASSERT_TRUE(threw);
    ASSERT_EQ(fixture.state.count(), 10u);
    EXPECT_EQ(fixture.state.at(5).kind, RecordingState::Kind::Touch);
    EXPECT_EQ(fixture.state.at(6).kind, RecordingState::Kind::SetPolicy);
    EXPECT_EQ(fixture.state.at(7).kind, RecordingState::Kind::SetAffinity);
    EXPECT_EQ(fixture.state.at(8).kind, RecordingState::Kind::GetPolicy);
    EXPECT_EQ(fixture.state.at(9).kind, RecordingState::Kind::GetAffinity);
    EXPECT_EQ(fixture.state.policy_mode, MPOL_INTERLEAVE);
    EXPECT_EQ(fixture.state.policy_nodes, (PolicyMask{1ul, 0}));
    EXPECT_TRUE(affinityEquals(fixture.state.affinity,
        Fixture::originalAffinity()));
}

// ---------------------------------------------------------------------------
// Fatal-path controls: a restoration failure is FATAL — the result is
// never returned, the terminal diagnostic names both per-part errnos,
// and no best-effort continuation exists. Each control isolates ONE
// failing part and then both together, including during unwind. The
// parts fail with DISTINGUISHABLE errnos (policy EIO, affinity EPERM),
// the death regexes match the EXACT per-part diagnostic, and retained
// bounded call evidence (a preopened trace file written with write(2),
// inherited by the death-test child) proves BOTH restoration attempts
// executed before the termination — skipping either attempt fails the
// tests.
// ---------------------------------------------------------------------------

namespace
{
    /// Status of a retained-evidence read: the fatal-path validator
    /// rejects on ANY non-Ok status, so a truncated or errored capture
    /// can never satisfy a death proof even if it contains the
    /// required markers.
    enum class TraceReadStatus
    {
        Ok,
        Truncated,
        ReadError,
        SeekError,
        InvalidDescriptor
    };

    struct TraceReadResult
    {
        TraceReadStatus status = TraceReadStatus::InvalidDescriptor;
        std::string content;
    };

    inline const char *traceStatusName(TraceReadStatus status)
    {
        switch (status)
        {
        case TraceReadStatus::Ok:
            return "Ok";
        case TraceReadStatus::Truncated:
            return "Truncated";
        case TraceReadStatus::ReadError:
            return "ReadError";
        case TraceReadStatus::SeekError:
            return "SeekError";
        case TraceReadStatus::InvalidDescriptor:
            return "InvalidDescriptor";
        }
        return "Unknown";
    }

    /// Owned anonymous trace buffer for fatal-path call evidence.
    /// Acquired (memfd) BEFORE any mutation; the death-test child
    /// inherits the descriptor. There is NO pathname: failed
    /// acquisition can never delete anything, and the evidence is read
    /// through the RETAINED descriptor (never a pathname reopen) with
    /// a hard bound.
    class TraceFile
    {
    public:
        TraceFile() : fd_(::memfd_create("ftb-trace", 0)) {}

        /// Injection constructor for the failed-acquisition and
        /// recording-failure negatives (the test owns the descriptor;
        /// a negative fd models failed acquisition).
        explicit TraceFile(int injected_fd) : fd_(injected_fd) {}

        ~TraceFile()
        {
            if (fd_ >= 0)
                (void)close(fd_);
            // Anonymous: no path exists, nothing is ever unlinked —
            // including on failed acquisition (fd_ < 0).
        }

        TraceFile(const TraceFile &) = delete;
        TraceFile &operator=(const TraceFile &) = delete;

        bool valid() const { return fd_ >= 0; }
        int fd() const { return fd_; }

        /// Read the retained evidence through the RETAINED descriptor
        /// with a HARD bound (8 KiB). The read STATUS is part of the
        /// result: a bounded-out read reports Truncated (with a
        /// visible trailing marker), a failed read reports ReadError
        /// (with the partial prefix — which the shared validator
        /// REJECTS on status), and a failed acquisition reports
        /// InvalidDescriptor. No silent prefix return remains.
        TraceReadResult read() const
        {
            TraceReadResult result;
            if (fd_ < 0)
                return result; // InvalidDescriptor
            if (::lseek(fd_, 0, SEEK_SET) == static_cast<off_t>(-1))
            {
                result.status = TraceReadStatus::SeekError;
                return result;
            }
            constexpr size_t kBound = 8192;
            char buffer[512];
            size_t total = 0;
            bool read_error = false;
            while (total < kBound)
            {
                const size_t want =
                    sizeof(buffer) < (kBound - total) ? sizeof(buffer)
                                                      : (kBound - total);
                const ssize_t got = ::read(fd_, buffer, want);
                if (got < 0)
                {
                    if (errno == EINTR)
                        continue;
                    read_error = true;
                    break;
                }
                if (got == 0)
                    break;
                result.content.append(buffer, static_cast<size_t>(got));
                total += static_cast<size_t>(got);
            }
            if (read_error)
                result.status = TraceReadStatus::ReadError;
            else if (total == kBound)
            {
                result.content += "[truncated]";
                result.status = TraceReadStatus::Truncated;
            }
            else
                result.status = TraceReadStatus::Ok;
            return result;
        }

    private:
        int fd_;
    };

    /// THE shared death-proof evidence validator: it consumes a
    /// TraceReadResult VALUE (callers read the retained descriptor
    /// ONCE) — the capture/read STATUS is part of the proof. The
    /// result must have status Ok (never truncated, never a read
    /// error, never a failed acquisition) AND contain both required
    /// call lines in order. Every fatal-path death test asserts
    /// through THIS validator, and the validator-control negatives
    /// pass COUNTEREXAMPLE VALUES through the same entry point.
    inline ::testing::AssertionResult expectFatalEvidenceInOrder(
        const TraceReadResult &result, const char *first_line,
        const char *second_line)
    {
        if (result.status != TraceReadStatus::Ok)
            return ::testing::AssertionFailure()
                << "trace evidence is UNUSABLE (status="
                << traceStatusName(result.status) << ") — a failed or "
                   "truncated capture must be REJECTED, not read as a "
                   "prefix: " << result.content;
        const size_t first_at = result.content.find(first_line);
        if (first_at == std::string::npos)
            return ::testing::AssertionFailure()
                << "missing evidence line '" << first_line
                << "': " << result.content;
        const size_t second_at = result.content.find(second_line);
        if (second_at == std::string::npos)
            return ::testing::AssertionFailure()
                << "missing evidence line '" << second_line
                << "': " << result.content;
        if (first_at >= second_at)
            return ::testing::AssertionFailure()
                << "evidence order violated ('" << first_line
                << "' must precede '" << second_line
                << "'): " << result.content;
        return ::testing::AssertionSuccess();
    }

    /** @brief Require the actual terminal signal, diagnostic and complete once-only trace. */
    void expectPlacementAbort(Fixture &fixture, const char *diagnostic, const char *expected_trace,
        int target_node = Fixture::kTargetNode, FirstTouchGeometry geometry = Fixture::kGeometry)
    {
        TraceFile trace;
        ASSERT_TRUE(trace.valid());
        fixture.state.trace_fd = trace.fd();
        fixture.state.abort_on_trace_failure = true;
        fixture.state.clobber_errno_on_success = ENOENT;
        uint8_t pages[2 * 4096]{};
        const auto ops = recordingOps(fixture.state);
        EXPECT_EXIT(
            { (void)firstTouchPopulateRange(pages, sizeof(pages), 4096, target_node, geometry, ops); },
            ::testing::KilledBySignal(SIGABRT), diagnostic);
        const auto result = trace.read();
        EXPECT_TRUE(expectFatalEvidenceInOrder(result, "policy_restore:", "affinity_call:restore:"));
        if (result.status == TraceReadStatus::Ok)
            EXPECT_EQ(result.content, expected_trace);
    }
} // namespace

TEST(FirstTouchBoundary, RestorePolicyFailureIsFatal)
{
    Fixture fixture;
    fixture.state.fail_policy_restore = true;
    expectPlacementAbort(fixture,
        "primary_phase=completed primary=ok policy_set=rejected:status=-1:errno=5 affinity_set=ok "
        "policy_read=ok policy_state=mismatch affinity_read=ok affinity_state=equal",
        "affinity_call:1:ok\npolicy_restore:fail:5\naffinity_call:restore:ok\n"
        "policy_readback:ok\naffinity_readback:ok\n");
}

TEST(FirstTouchBoundary, RestoreAffinityFailureIsFatal)
{
    Fixture fixture;
    fixture.state.fail_set_affinity_on_call = 2;
    expectPlacementAbort(fixture,
        "primary_phase=completed primary=ok policy_set=ok affinity_set=rejected:status=-1:errno=1 "
        "policy_read=ok policy_state=equal affinity_read=ok affinity_state=mismatch",
        "affinity_call:1:ok\npolicy_restore:ok\naffinity_call:restore:fail:1\n"
        "policy_readback:ok\naffinity_readback:ok\n");
}

TEST(FirstTouchBoundary, RestoreBothPartsFailureIsFatal)
{
    Fixture fixture;
    fixture.state.fail_policy_restore = true;
    fixture.state.fail_set_affinity_on_call = 2;
    expectPlacementAbort(fixture,
        "primary_phase=completed primary=ok policy_set=rejected:status=-1:errno=5 "
        "affinity_set=rejected:status=-1:errno=1 policy_read=ok policy_state=mismatch "
        "affinity_read=ok affinity_state=mismatch",
        "affinity_call:1:ok\npolicy_restore:fail:5\naffinity_call:restore:fail:1\n"
        "policy_readback:ok\naffinity_readback:ok\n");
}

TEST(FirstTouchBoundary, UnwindOutOfTouchWithFailedRestoreIsFatal)
{
    Fixture fixture;
    fixture.state.throw_on_touch = true;
    fixture.state.fail_policy_restore = true;
    expectPlacementAbort(fixture,
        "primary_phase=touch primary=exception policy_set=rejected:status=-1:errno=5 affinity_set=ok "
        "policy_read=ok policy_state=mismatch affinity_read=ok affinity_state=equal",
        "affinity_call:1:ok\ntouch:exception\npolicy_restore:fail:5\naffinity_call:restore:ok\n"
        "policy_readback:ok\naffinity_readback:ok\n");
}

TEST(FirstTouchBoundary, BindRejectionAndBothRestoreFailuresRetainEachOutcome)
{
    Fixture fixture;
    fixture.state.fail_bind_errno = EACCES;
    fixture.state.fail_policy_restore = true;
    fixture.state.fail_set_affinity_on_call = 2;
    expectPlacementAbort(fixture,
        "primary_phase=bind-policy primary=rejected:status=-1:errno=13 "
        "policy_set=rejected:status=-1:errno=5 affinity_set=rejected:status=-1:errno=1 "
        "policy_read=ok policy_state=equal affinity_read=ok affinity_state=mismatch",
        "affinity_call:1:ok\npolicy_bind:fail\npolicy_restore:fail:5\naffinity_call:restore:fail:1\n"
        "policy_readback:ok\naffinity_readback:ok\n");
}

TEST(FirstTouchBoundary, RejectedRestoreIsFatalEvenWhenBothReadsMatch)
{
    Fixture fixture;
    fixture.state.fail_bind_affinity_errno = EACCES;
    fixture.state.fail_policy_restore = true;
    expectPlacementAbort(fixture,
        "primary_phase=bind-affinity primary=rejected:status=-1:errno=13 "
        "policy_set=rejected:status=-1:errno=5 affinity_set=ok "
        "policy_read=ok policy_state=equal affinity_read=ok affinity_state=equal",
        "affinity_call:bind:fail\npolicy_restore:fail:5\naffinity_call:restore:ok\n"
        "policy_readback:ok\naffinity_readback:ok\n");
}

TEST(FirstTouchBoundary, ThrowingRestoresAreFatalEvenWhenBothReadsMatch)
{
    Fixture fixture;
    fixture.state.throw_after_policy_restore = true;
    fixture.state.throw_after_affinity_restore = true;
    expectPlacementAbort(fixture,
        "primary_phase=completed primary=ok policy_set=exception affinity_set=exception "
        "policy_read=ok policy_state=equal affinity_read=ok affinity_state=equal",
        "affinity_call:1:ok\npolicy_restore:exception\naffinity_call:restore:exception\n"
        "policy_readback:ok\naffinity_readback:ok\n");
}

TEST(FirstTouchBoundary, RejectedPolicyReadbackStillQueriesAffinityBeforeFatal)
{
    Fixture fixture;
    fixture.state.fail_readback_policy_errno = EBADF;
    expectPlacementAbort(fixture,
        "primary_phase=completed primary=ok policy_set=ok affinity_set=ok "
        "policy_read=rejected:status=-1:errno=9 policy_state=not-compared affinity_read=ok affinity_state=equal",
        "affinity_call:1:ok\npolicy_restore:ok\naffinity_call:restore:ok\n"
        "policy_readback:rejected\naffinity_readback:ok\n");
}

TEST(FirstTouchBoundary, ThrowingAffinityReadbackIsFatalDuringUnwind)
{
    Fixture fixture;
    fixture.state.throw_on_touch = true;
    fixture.state.throw_on_affinity_readback = true;
    expectPlacementAbort(fixture,
        "primary_phase=touch primary=exception policy_set=ok affinity_set=ok "
        "policy_read=ok policy_state=equal affinity_read=exception affinity_state=not-compared",
        "affinity_call:1:ok\ntouch:exception\npolicy_restore:ok\naffinity_call:restore:ok\n"
        "policy_readback:ok\naffinity_readback:exception\n");
}

TEST(FirstTouchBoundary, BothReadbackFailuresRetainedDuringUnwind)
{
    Fixture fixture;
    fixture.state.throw_on_touch = true;
    fixture.state.fail_readback_policy_errno = EBADF;
    fixture.state.throw_on_affinity_readback = true;
    expectPlacementAbort(fixture,
        "primary_phase=touch primary=exception policy_set=ok affinity_set=ok "
        "policy_read=rejected:status=-1:errno=9 policy_state=not-compared "
        "affinity_read=exception affinity_state=not-compared",
        "affinity_call:1:ok\ntouch:exception\npolicy_restore:ok\naffinity_call:restore:ok\n"
        "policy_readback:rejected\naffinity_readback:exception\n");
}

// ---------------------------------------------------------------------------
// Trace-evidence helper negatives: the fatal-path evidence machinery
// itself must fail visibly and safely — failed acquisition deletes
// nothing, recording failures are visible and never corrupt the errno
// under observation, and evidence reads are hard-bounded.
// ---------------------------------------------------------------------------

TEST(FirstTouchBoundary, TraceFileFailedAcquisitionIsSafeAndVisible)
{
    // A failed acquisition (negative descriptor) is representable and
    // safe: nothing exists to delete (anonymous, no pathname), tracing
    // flags the failure instead of silently doing nothing, and reads
    // through the retained descriptor are empty — never a reopen of a
    // path that does not exist.
    TraceFile failed(-1);
    EXPECT_FALSE(failed.valid());
    RecordingState state;
    state.trace_fd = failed.fd();
    errno = EIO;
    state.trace("policy_restore:fail:5");
    EXPECT_EQ(errno, EIO)
        << "recording activity must not corrupt the observed errno";
    EXPECT_TRUE(state.trace_failed)
        << "a failed acquisition must be VISIBLE, never silent";
    const TraceReadResult result = failed.read();
    EXPECT_EQ(result.status, TraceReadStatus::InvalidDescriptor);
    EXPECT_TRUE(result.content.empty());
    // The destructor of the failed instance deletes nothing and cannot
    // crash (exercised by leaving this scope).
}

TEST(FirstTouchBoundary, TraceFileRecordingFailureIsVisiblePreservesErrno)
{
    // A descriptor that exists but CANNOT be written (a read-only fd)
    // models a recording failure: the write fails, the failure is
    // visible, and the errno under observation survives the attempt.
    const int read_only =
        ::open("/etc/hostname", O_RDONLY | O_CLOEXEC);
    ASSERT_GE(read_only, 0);
    {
        TraceFile ro(read_only); // owns the descriptor
        ASSERT_TRUE(ro.valid());
        RecordingState state;
        state.trace_fd = ro.fd();
        errno = EIO; // the errno under observation
        state.trace("policy_restore:fail:5");
        EXPECT_EQ(errno, EIO)
            << "the failed recording attempt must not corrupt the "
               "observed errno";
        EXPECT_TRUE(state.trace_failed)
            << "a failed recording must be VISIBLE";
    } // descriptor closed exactly once by the owner
}

TEST(FirstTouchBoundary, TraceFileReadIsHardBounded)
{
    TraceFile trace;
    ASSERT_TRUE(trace.valid());
    RecordingState state;
    state.trace_fd = trace.fd();
    // Write well past the hard bound: the read must stop at the bound
    // with a visible truncation status and marker, never grow
    // unbounded.
    for (int i = 0; i < 200; ++i)
        state.trace("0123456789012345678901234567890123456789");
    const TraceReadResult result = trace.read();
    EXPECT_EQ(result.status, TraceReadStatus::Truncated);
    constexpr size_t kBound = 8192;
    EXPECT_LE(result.content.size(),
        kBound + strlen("[truncated]"));
    EXPECT_NE(result.content.find("[truncated]"), std::string::npos)
        << "a bounded-out read must be visible";
    EXPECT_FALSE(state.trace_failed);
}

// The capture/read STATUS must gate the death proofs: evidence that
// CONTAINS the required markers is still REJECTED when the capture is
// truncated or errored — the old substring checks alone would pass.
TEST(FirstTouchBoundary,
    TraceEvidenceValidatorRejectsTruncatedEvidenceContainingMarkers)
{
    TraceFile trace;
    ASSERT_TRUE(trace.valid());
    RecordingState state;
    state.trace_fd = trace.fd();
    // The required fatal-path markers are written FIRST...
    state.trace("policy_restore:fail:5");
    state.trace("affinity_call:restore:ok");
    // ...then enough content to overrun the hard bound.
    for (int i = 0; i < 300; ++i)
        state.trace("0123456789012345678901234567890123456789");
    const TraceReadResult result = trace.read();
    ASSERT_EQ(result.status, TraceReadStatus::Truncated);
    // The markers ARE present in the truncated content...
    EXPECT_NE(result.content.find("policy_restore:fail:5"),
        std::string::npos);
    EXPECT_NE(result.content.find("affinity_call:restore:ok"),
        std::string::npos);
    // ...but the SAME shared death-proof validator must REJECT the
    // VALUE (read once; no second consumption).
    EXPECT_FALSE(expectFatalEvidenceInOrder(
        result, "policy_restore:fail:5", "affinity_call:restore:ok"));
    EXPECT_FALSE(state.trace_failed);
}

// VALIDATOR CONTROL (value counterexample, honestly labeled): a
// TraceReadResult with status ReadError whose content CONTAINS BOTH
// required markers is passed through the SAME shared validator entry
// point the real death tests use — it must be REJECTED on status
// FIRST. An ignore-status mutant would find both markers in the
// content and wrongly pass; this control proves that mutant fails.
// No descriptors are opened here at all: the counterexample is a
// plain value, exactly like the truncated control above.
TEST(FirstTouchBoundary,
    TraceEvidenceValidatorRejectsReadErrorValueContainingMarkers)
{
    TraceReadResult broken;
    broken.status = TraceReadStatus::ReadError;
    broken.content =
        "policy_restore:fail:5\n"
        "affinity_call:restore:ok\n";
    // Both required markers ARE present in the errored content...
    EXPECT_NE(broken.content.find("policy_restore:fail:5"),
        std::string::npos);
    EXPECT_NE(broken.content.find("affinity_call:restore:ok"),
        std::string::npos);
    // ...but the SAME shared status-first validator must REJECT it.
    EXPECT_FALSE(expectFatalEvidenceInOrder(
        broken, "policy_restore:fail:5", "affinity_call:restore:ok"));
}

TEST(FirstTouchBoundary, TraceEvidenceValidatorRejectsReadError)
{
    // A write-only descriptor: the trace writes SUCCEED (the markers
    // are recorded), but reading the retained descriptor FAILS — the
    // old silent-prefix readAll would have returned an empty string
    // while the read had errored; the validator must reject on the
    // ReadError status.
    const int write_only = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
    ASSERT_GE(write_only, 0);
    TraceFile wo(write_only); // owns the descriptor
    RecordingState state;
    state.trace_fd = wo.fd();
    state.trace("policy_restore:fail:5");
    state.trace("affinity_call:restore:ok");
    EXPECT_FALSE(state.trace_failed)
        << "the recording itself succeeds on the write-only "
           "descriptor";
    const TraceReadResult result = wo.read();
    EXPECT_EQ(result.status, TraceReadStatus::ReadError);
    EXPECT_FALSE(expectFatalEvidenceInOrder(
        result, "policy_restore:fail:5", "affinity_call:restore:ok"));
}

namespace
{
/// EXPECT_EXIT predicate: the child terminated with the DISTINCT
/// trace-recording-failure exit.
bool exitedWithTraceRecordingFailure(int status)
{
    return WIFEXITED(status) && WEXITSTATUS(status) == 83;
}
} // namespace

TEST(FirstTouchBoundary, TraceRecordingFailureProducesDistinctChildFailure)
{
    // Inside a death-test child, a RECORDING failure must not be able
    // to masquerade as the expected fatal death: the child fails
    // immediately with a distinct exit (83) and an unambiguous stderr
    // line, so the fatal death regex can never match and the parent's
    // EXPECT_DEATH would FAIL — a recording failure is REJECTED, not
    // silently accepted.
    const int read_only = ::open("/etc/hostname", O_RDONLY | O_CLOEXEC);
    ASSERT_GE(read_only, 0);
    // RAII owner: the parent closes the descriptor exactly once; the
    // death-test child only inherits it (no parent-side leak).
    TraceFile owner(read_only);
    ASSERT_TRUE(owner.valid());
    EXPECT_EXIT(
        {
            RecordingState state;
            state.trace_fd = owner.fd();
            state.abort_on_trace_failure = true;
            errno = EIO; // the errno under observation
            state.trace("policy_restore:fail:5");
            errno = EIO;
            state.trace("affinity_call:restore:ok");
            FAIL() << "a recording failure must terminate the child "
                      "before this point";
        },
        exitedWithTraceRecordingFailure, "trace-recording-failed");
}

namespace
{
    /** @brief Accept the bind request without performing it, preserving the recorder. */
    int omitBindSetPolicy(void *context, int mode,
        const unsigned long *nodemask, unsigned long maxnode)
    {
        auto &state = *static_cast<RecordingState *>(context);
        if (!state.policy_bind_pending)
            return recordSetPolicy(context, mode, nodemask, maxnode);
        state.policy_bind_pending = false;
        RecordingState::Event event{RecordingState::Kind::SetPolicy};
        event.mode = mode;
        event.maxnode = maxnode;
        event.has_mask = nodemask != nullptr;
        if (!nodemask || maxnode != state.expected_maxnode || state.policy_words != 1)
        {
            state.record(event);
            errno = EINVAL;
            return -1;
        }
        event.nodes[0] = nodemask[0];
        state.record(event);
        return 0;
    }
} // namespace

TEST(FirstTouchBoundary, MutationControlOmitBindFailsTheValidator)
{
    Fixture fixture;
    uint8_t pages[2 * 4096]{};
    const auto ops = recordingOps(fixture.state, &omitBindSetPolicy);
    ASSERT_TRUE(firstTouchPopulateRange(
                    pages, sizeof(pages), 4096, Fixture::kTargetNode,
                    Fixture::kGeometry, ops)
                    .completed);
    const auto validation = validatePopulateRecord(
        fixture.state, pages, sizeof(pages), 4096, Fixture::kTargetNode,
        MPOL_INTERLEAVE, 1ul, Fixture::originalAffinity());
    EXPECT_FALSE(validation.ok);
    EXPECT_NE(validation.failure.find("touch executed under a policy other than MPOL_BIND"),
        std::string::npos) << validation.failure;
}

TEST(FirstTouchBoundary, SilentPolicyRestoreOmissionIsFatal)
{
    Fixture fixture;
    fixture.state.policy_restore = PolicyRestoreEffect::Omit;
    expectPlacementAbort(fixture,
        "primary_phase=completed primary=ok policy_set=ok affinity_set=ok "
        "policy_read=ok policy_state=mismatch affinity_read=ok affinity_state=equal",
        "affinity_call:1:ok\npolicy_restore:ok\naffinity_call:restore:ok\n"
        "policy_readback:ok\naffinity_readback:ok\n");
}

TEST(FirstTouchBoundary, SilentPolicyModeChangeIsFatalWithMaskIntact)
{
    Fixture fixture;
    fixture.state.policy_restore = PolicyRestoreEffect::WrongMode;
    expectPlacementAbort(fixture,
        "primary_phase=completed primary=ok policy_set=ok affinity_set=ok "
        "policy_read=ok policy_state=mismatch affinity_read=ok affinity_state=equal",
        "affinity_call:1:ok\npolicy_restore:ok\naffinity_call:restore:ok\n"
        "policy_readback:ok\naffinity_readback:ok\n");
}

TEST(FirstTouchBoundary, SilentAffinityReplacementIsFatal)
{
    Fixture fixture;
    fixture.state.affinity_restore = AffinityRestoreEffect::WrongLowMask;
    expectPlacementAbort(fixture,
        "primary_phase=completed primary=ok policy_set=ok affinity_set=ok "
        "policy_read=ok policy_state=equal affinity_read=ok affinity_state=mismatch",
        "affinity_call:1:ok\npolicy_restore:ok\naffinity_call:restore:ok\n"
        "policy_readback:ok\naffinity_readback:ok\n");
}

TEST(FirstTouchBoundary, SilentHighCpuLossIsFatal)
{
    Fixture fixture;
    CPU_SET(100, &fixture.state.affinity);
    fixture.state.affinity_restore = AffinityRestoreEffect::DropHighCpu;
    expectPlacementAbort(fixture,
        "primary_phase=completed primary=ok policy_set=ok affinity_set=ok "
        "policy_read=ok policy_state=equal affinity_read=ok affinity_state=mismatch",
        "affinity_call:1:ok\npolicy_restore:ok\naffinity_call:restore:ok\n"
        "policy_readback:ok\naffinity_readback:ok\n");
}

TEST(FirstTouchBoundary, SilentHighPolicyBitLossDuringUnwindIsFatal)
{
    Fixture fixture;
    fixture.state.expected_maxnode = 66;
    fixture.state.policy_words = 2;
    fixture.state.policy_nodes = {1ul, 1ul};
    fixture.state.policy_restore = PolicyRestoreEffect::DropHighNode;
    fixture.state.throw_on_touch = true;
    expectPlacementAbort(fixture,
        "primary_phase=touch primary=exception policy_set=ok affinity_set=ok "
        "policy_read=ok policy_state=mismatch affinity_read=ok affinity_state=equal",
        "affinity_call:1:ok\ntouch:exception\npolicy_restore:ok\naffinity_call:restore:ok\n"
        "policy_readback:ok\naffinity_readback:ok\n", 64, FirstTouchGeometry{64});
}

TEST(FirstTouchBoundary, MutationControlTouchBeforeBindFailsTheValidator)
{
    Fixture fixture;
    const FirstTouchBoundaryOps ops = recordingOps(fixture.state);

    // Deliberately misordered adapter calls exercise the validator,
    // not a second production state machine.
    uint8_t pages[2 * 4096]{};
    cpu_set_t original{};
    cpu_set_t target{};
    cpu_set_t actual_affinity{};
    PolicyMask original_nodes{};
    const PolicyMask target_nodes{1ul << 1, 0};
    PolicyMask actual_nodes{};
    int original_mode = -1;
    int actual_mode = -1;
    constexpr unsigned long maxnode = 5;
    ASSERT_EQ(ops.getThreadAffinity(ops.context, &original), 0);
    ASSERT_EQ(ops.getMemoryPolicy(ops.context, &original_mode,
                  original_nodes.data(), maxnode, nullptr, 0), 0);
    ASSERT_EQ(ops.nodeCpuSet(ops.context, 1, &target), 0);
    ASSERT_EQ(ops.setThreadAffinity(ops.context, &target), 0);
    ops.touchRange(ops.context, pages, sizeof(pages), 4096);
    ASSERT_EQ(ops.setMemoryPolicy(ops.context, MPOL_BIND, target_nodes.data(), maxnode), 0);
    ASSERT_EQ(ops.setMemoryPolicy(ops.context, original_mode, original_nodes.data(), maxnode), 0);
    ASSERT_EQ(ops.setThreadAffinity(ops.context, &original), 0);
    ASSERT_EQ(ops.getMemoryPolicy(ops.context, &actual_mode, actual_nodes.data(), maxnode, nullptr, 0), 0);
    ASSERT_EQ(ops.getThreadAffinity(ops.context, &actual_affinity), 0);
    ASSERT_EQ(actual_mode, original_mode);
    ASSERT_EQ(actual_nodes, original_nodes);
    ASSERT_TRUE(affinityEquals(actual_affinity, original));

    const auto validation = validatePopulateRecord(
        fixture.state, pages, sizeof(pages), 4096, Fixture::kTargetNode,
        MPOL_INTERLEAVE, 1ul, Fixture::originalAffinity());
    EXPECT_FALSE(validation.ok);
    EXPECT_EQ(validation.failure, "event 4 kind mismatch");
}

TEST(FirstTouchBoundary, NativeMaxnodePreservesLastNodeAcrossWordBoundary)
{
    struct TestLayout
    {
        int top_node;
        unsigned long maxnode;
        size_t words;
        PolicyMask original;
        PolicyMask target;
    };
    constexpr TestLayout layouts[]{
        {63, 65, 1, {(1ul << 63) | 1ul, 0}, {1ul << 63, 0}},
        {64, 66, 2, {1ul, 1ul}, {0, 1ul}},
    };
    for (const auto &layout : layouts)
    {
        SCOPED_TRACE(layout.top_node);
        Fixture fixture;
        fixture.state.expected_maxnode = layout.maxnode;
        fixture.state.policy_words = layout.words;
        fixture.state.policy_nodes = layout.original;
        uint8_t pages[2 * 4096]{};
        const auto ops = recordingOps(fixture.state);
        const auto result = firstTouchPopulateRange(pages, sizeof(pages), 4096,
            layout.top_node, FirstTouchGeometry{layout.top_node}, ops);

        ASSERT_TRUE(result.completed);
        ASSERT_EQ(fixture.state.count(), 10u);
        ASSERT_FALSE(fixture.state.overflow);
        const auto &capture = fixture.state.at(1);
        const auto &bind = fixture.state.at(4);
        const auto &touch = fixture.state.at(5);
        const auto &restore = fixture.state.at(6);
        const auto &readback = fixture.state.at(8);
        EXPECT_EQ(capture.kind, RecordingState::Kind::GetPolicy);
        EXPECT_EQ(bind.kind, RecordingState::Kind::SetPolicy);
        EXPECT_EQ(touch.kind, RecordingState::Kind::Touch);
        EXPECT_EQ(restore.kind, RecordingState::Kind::SetPolicy);
        EXPECT_EQ(readback.kind, RecordingState::Kind::GetPolicy);
        for (const auto *event : {&capture, &bind, &restore, &readback})
        {
            EXPECT_TRUE(event->has_mask);
            EXPECT_EQ(event->maxnode, layout.maxnode);
        }
        EXPECT_EQ(bind.mode, MPOL_BIND);
        EXPECT_EQ(bind.nodes, layout.target);
        EXPECT_EQ(touch.policy_at_touch, MPOL_BIND);
        EXPECT_EQ(touch.nodes_at_touch, layout.target);
        EXPECT_EQ(restore.mode, MPOL_INTERLEAVE);
        EXPECT_EQ(restore.nodes, layout.original);
        EXPECT_EQ(fixture.state.policy_mode, MPOL_INTERLEAVE);
        EXPECT_EQ(fixture.state.policy_nodes, layout.original);
        EXPECT_TRUE(affinityEquals(fixture.state.affinity, Fixture::originalAffinity()));
    }
}

TEST(FirstTouchBoundary, EmptyPolicyRestoreUsesNullButReadbackUsesFullMask)
{
    Fixture fixture;
    fixture.state.policy_mode = MPOL_DEFAULT;
    fixture.state.policy_nodes = {};
    fixture.state.expected_maxnode = 66;
    fixture.state.policy_words = 2;
    uint8_t pages[4096]{};
    const auto ops = recordingOps(fixture.state);
    ASSERT_TRUE(firstTouchPopulateRange(pages, sizeof(pages), 4096, 64, FirstTouchGeometry{64}, ops).completed);
    ASSERT_EQ(fixture.state.count(), 10u);
    const auto &capture = fixture.state.at(1);
    const auto &restore = fixture.state.at(6);
    const auto &readback = fixture.state.at(8);
    EXPECT_EQ(capture.kind, RecordingState::Kind::GetPolicy);
    EXPECT_EQ(restore.kind, RecordingState::Kind::SetPolicy);
    EXPECT_EQ(readback.kind, RecordingState::Kind::GetPolicy);
    EXPECT_EQ(restore.mode, MPOL_DEFAULT);
    EXPECT_FALSE(restore.has_mask);
    EXPECT_EQ(restore.maxnode, 0ul);
    EXPECT_TRUE(capture.has_mask);
    EXPECT_TRUE(readback.has_mask);
    EXPECT_EQ(capture.maxnode, 66ul);
    EXPECT_EQ(readback.maxnode, 66ul);
    EXPECT_EQ(fixture.state.policy_mode, MPOL_DEFAULT);
    EXPECT_EQ(fixture.state.policy_nodes, PolicyMask{});
    EXPECT_TRUE(affinityEquals(fixture.state.affinity, Fixture::originalAffinity()));
}

// ---------------------------------------------------------------------------
// Complete multi-word nodemask helpers: device-free coverage of the
// word arithmetic shared by the production sequence and the native
// proofs. Node ids 0/63/64/127/128/200 and sparse combinations are
// covered; out-of-bounds ids are rejected WITHOUT touching the mask.
// ---------------------------------------------------------------------------

TEST(FirstTouchBoundaryMasks, WordCountsCoverTheCompleteNodeSpace)
{
    // Negative geometry is INVALID: 0 words, never a plausible
    // single-node result (production rejects it with EINVAL first).
    EXPECT_EQ(llaminar2::firstTouchMaskWordCount(-1), 0u);
    EXPECT_EQ(llaminar2::firstTouchMaskWordCount(0), 1u);
    EXPECT_EQ(llaminar2::firstTouchMaskWordCount(1), 1u);
    EXPECT_EQ(llaminar2::firstTouchMaskWordCount(63), 1u);
    EXPECT_EQ(llaminar2::firstTouchMaskWordCount(64), 2u);
    EXPECT_EQ(llaminar2::firstTouchMaskWordCount(127), 2u);
    EXPECT_EQ(llaminar2::firstTouchMaskWordCount(128), 3u);
    EXPECT_EQ(llaminar2::firstTouchMaskWordCount(200), 4u);
}

TEST(FirstTouchBoundaryMasks, SetAndQueryNodeBitsAcrossAllWords)
{
    constexpr size_t words = 4;
    for (const int node : {0, 1, 63, 64, 65, 127, 128, 200})
    {
        unsigned long mask[words] = {0, 0, 0, 0};
        ASSERT_TRUE(llaminar2::firstTouchMaskSetNode(mask, words, node))
            << "node " << node;
        EXPECT_TRUE(llaminar2::firstTouchMaskHasNode(mask, words, node))
            << "node " << node;
        // Exactly that node's bit, in the right word.
        unsigned long expected[words] = {0, 0, 0, 0};
        expected[static_cast<size_t>(node) / 64] =
            1ul << (static_cast<unsigned long>(node) % 64);
        EXPECT_TRUE(llaminar2::firstTouchMasksEqual(mask, expected, words))
            << "node " << node;
    }
    // Sparse combination of distant nodes in one mask.
    unsigned long sparse[words] = {0, 0, 0, 0};
    EXPECT_TRUE(llaminar2::firstTouchMaskSetNode(sparse, words, 5));
    EXPECT_TRUE(llaminar2::firstTouchMaskSetNode(sparse, words, 130));
    EXPECT_TRUE(llaminar2::firstTouchMaskHasNode(sparse, words, 5));
    EXPECT_TRUE(llaminar2::firstTouchMaskHasNode(sparse, words, 130));
    EXPECT_FALSE(llaminar2::firstTouchMaskHasNode(sparse, words, 6));
    EXPECT_FALSE(llaminar2::firstTouchMaskEmpty(sparse, words));
}

TEST(FirstTouchBoundaryMasks, OutOfBoundsIsRejectedWithoutMutation)
{
    unsigned long mask[2] = {0, 0};
    EXPECT_FALSE(llaminar2::firstTouchMaskSetNode(mask, 2, 128));
    EXPECT_FALSE(llaminar2::firstTouchMaskSetNode(mask, 2, 129));
    EXPECT_FALSE(llaminar2::firstTouchMaskSetNode(mask, 2, -1));
    // The mask is untouched by rejected operations.
    EXPECT_TRUE(llaminar2::firstTouchMaskEmpty(mask, 2));
    EXPECT_FALSE(llaminar2::firstTouchMaskHasNode(mask, 2, 128));
    EXPECT_FALSE(llaminar2::firstTouchMaskHasNode(mask, 2, -1));
}

TEST(FirstTouchBoundaryMasks, EqualityAndEmptinessCoverAllWords)
{
    // A difference in the SECOND word only — the exact class a
    // first-word-only comparison silently accepts.
    const unsigned long a[2] = {1, 2};
    const unsigned long b[2] = {1, 3};
    const unsigned long c[2] = {1, 2};
    EXPECT_FALSE(llaminar2::firstTouchMasksEqual(a, b, 2));
    EXPECT_TRUE(llaminar2::firstTouchMasksEqual(a, c, 2));
    const unsigned long e[2] = {0, 5};
    EXPECT_FALSE(llaminar2::firstTouchMaskEmpty(e, 2));
    const unsigned long f[2] = {0, 0};
    EXPECT_TRUE(llaminar2::firstTouchMaskEmpty(f, 2));
}
