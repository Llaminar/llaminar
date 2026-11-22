/**
 * @file FirstTouchBoundary.cpp
 * @brief Restore complete calling-thread placement before returning from first touch.
 *
 * Acquisition and mask storage precede the first setter. Once armed, the
 * guard owns both prior settings even if a setter rejects or throws.
 * Return and unwind require exact readback after the two restores.
 */

#include "FirstTouchBoundary.h"

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <numa.h>
#include <numaif.h>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /** @brief Backing extent and Linux wire extent for the same logical mask. */
        struct PolicyMaskLayout
        {
            std::size_t words;
            unsigned long maxnode;
        };

        /** @return A layout preserving every possible node bit without overflow. */
        std::optional<PolicyMaskLayout> policyMaskLayout(int highest_node) noexcept
        {
            static_assert(kFirstTouchMaskWordBits == 64, "Linux policy copyout uses the supported LP64 ABI");
            if (!std::in_range<unsigned long>(highest_node))
                return std::nullopt;
            const auto highest = static_cast<unsigned long>(highest_node);
            if (highest > std::numeric_limits<unsigned long>::max() - 2)
                return std::nullopt;
            const auto logical_bits = highest + 1;
            if (!std::in_range<std::size_t>(logical_bits))
                return std::nullopt;
            // Linux imports maxnode-1 bits; the extra wire unit is not a node bit.
            return PolicyMaskLayout{firstTouchMaskWordCount(highest_node), logical_bits + 1};
        }

        struct Succeeded {};
        struct NativeRejection { int status; int error; };
        struct CallbackException {};
        using OperationOutcome = std::variant<Succeeded, NativeRejection, CallbackException>;

        /** @brief Keep the operation which entered restoration separate from cleanup. */
        struct PrimaryOutcome
        {
            FirstTouchPhase phase = FirstTouchPhase::BindAffinity;
            OperationOutcome operation = Succeeded{};
        };

        /** @brief Capture one cleanup call without letting its failure suppress the next. */
        template <class Operation>
        OperationOutcome observeCleanup(Operation operation) noexcept
        {
            try
            {
                const int status = operation();
                if (status != 0)
                    return NativeRejection{status, errno};
                return Succeeded{};
            }
            catch (...)
            {
                return CallbackException{};
            }
        }

        /** @brief Format a diagnostic on the stack; only a native rejection has errno. */
        std::array<char, 80> describeOutcome(const OperationOutcome &outcome) noexcept
        {
            std::array<char, 80> text{};
            if (const auto *failure = std::get_if<NativeRejection>(&outcome))
                std::snprintf(text.data(), text.size(), "rejected:status=%d:errno=%d", failure->status, failure->error);
            else
                std::snprintf(text.data(), text.size(), "%s",
                    std::holds_alternative<Succeeded>(outcome) ? "ok" : "exception");
            return text;
        }

        /** @brief Comparison is forbidden when the corresponding observation was unsuccessful. */
        enum class Readback { NotCompared, Equal, Mismatch };

        /** @return The observed comparison, without inferring state from a query error. */
        const char *readbackName(Readback result) noexcept
        {
            switch (result)
            {
                case Readback::NotCompared: return "not-compared";
                case Readback::Equal: return "equal";
                case Readback::Mismatch: return "mismatch";
            }
            std::terminate();
        }

        /** @brief Restore and verify both captured settings once at scope exit. */
        class RestoreThreadPlacement final
        {
        public:
            /** @brief Arm before the first setter, borrowing only preallocated state. */
            RestoreThreadPlacement(const FirstTouchBoundaryOps &ops, const cpu_set_t &affinity,
                                   int policy, const std::vector<unsigned long> &mask,
                                   std::vector<unsigned long> &readback_mask, unsigned long maxnode,
                                   const PrimaryOutcome &primary) noexcept
                : ops_(ops), affinity_(affinity), policy_(policy), mask_(mask),
                  readback_mask_(readback_mask), maxnode_(maxnode), primary_(primary) {}
            RestoreThreadPlacement(const RestoreThreadPlacement &) = delete;
            RestoreThreadPlacement &operator=(const RestoreThreadPlacement &) = delete;

            /** @brief Restore policy then affinity; query both once before any continuation. */
            ~RestoreThreadPlacement() noexcept
            {
                const int interrupted_errno = errno;
                const bool empty_mask = firstTouchMaskEmpty(mask_.data(), mask_.size());
                const auto policy_set = observeCleanup([&] {
                    return ops_.setMemoryPolicy(ops_.context, policy_,
                        empty_mask ? nullptr : mask_.data(), empty_mask ? 0 : maxnode_);
                });
                const auto affinity_set = observeCleanup([&] {
                    return ops_.setThreadAffinity(ops_.context, &affinity_);
                });
                const auto policy_read = observeCleanup([&] {
                    return ops_.getMemoryPolicy(ops_.context, &readback_policy_,
                        readback_mask_.data(), maxnode_, nullptr, 0);
                });
                const auto affinity_read = observeCleanup([&] {
                    return ops_.getThreadAffinity(ops_.context, &readback_affinity_);
                });
                const auto policy_state = !std::holds_alternative<Succeeded>(policy_read) ? Readback::NotCompared :
                    (readback_policy_ == policy_ && readback_mask_ == mask_ ? Readback::Equal : Readback::Mismatch);
                const auto affinity_state = !std::holds_alternative<Succeeded>(affinity_read) ? Readback::NotCompared :
                    (CPU_EQUAL(&affinity_, &readback_affinity_) ? Readback::Equal : Readback::Mismatch);
                if (!std::holds_alternative<Succeeded>(policy_set) || !std::holds_alternative<Succeeded>(affinity_set) ||
                    policy_state != Readback::Equal || affinity_state != Readback::Equal)
                {
                    std::fprintf(stderr,
                        "Fatal: cannot restore first-touch thread placement: primary_phase=%s primary=%s "
                        "policy_set=%s affinity_set=%s policy_read=%s policy_state=%s affinity_read=%s affinity_state=%s\n",
                        firstTouchPhaseName(primary_.phase), describeOutcome(primary_.operation).data(),
                        describeOutcome(policy_set).data(), describeOutcome(affinity_set).data(),
                        describeOutcome(policy_read).data(), readbackName(policy_state),
                        describeOutcome(affinity_read).data(), readbackName(affinity_state));
                    std::terminate();
                }
                errno = interrupted_errno;
            }

        private:
            const FirstTouchBoundaryOps &ops_;
            const cpu_set_t &affinity_;
            const int policy_;
            const std::vector<unsigned long> &mask_;
            std::vector<unsigned long> &readback_mask_;
            const unsigned long maxnode_;
            const PrimaryOutcome &primary_;
            int readback_policy_ = -1;
            cpu_set_t readback_affinity_{};
        };

        /** @brief Resolve the target CPU set without installing any thread state. */
        int linuxNodeCpuSet(void *, int node, cpu_set_t *affinity)
        {
            auto *cpus = numa_allocate_cpumask();
            if (!cpus)
            {
                errno = ENOMEM;
                return -1;
            }
            const int status = numa_node_to_cpus(node, cpus);
            const int observed_errno = errno;
            if (status == 0)
            {
                CPU_ZERO(affinity);
                for (unsigned long cpu = 0; cpu < cpus->size && cpu < CPU_SETSIZE; ++cpu)
                    if (numa_bitmask_isbitset(cpus, cpu) != 0)
                        CPU_SET(static_cast<int>(cpu), affinity);
            }
            numa_free_cpumask(cpus);
            errno = observed_errno;
            return status;
        }

        /** @brief Establish page residency with one store at each exact stride. */
        void linuxTouchRange(void *, volatile void *base, std::size_t bytes, std::size_t page_bytes)
        {
            auto *pages = static_cast<volatile std::uint8_t *>(base);
            for (std::size_t offset = 0; offset < bytes; offset += page_bytes)
                pages[offset] = 0;
        }
    }

    const char *firstTouchPhaseName(FirstTouchPhase phase) noexcept
    {
        switch (phase)
        {
            case FirstTouchPhase::Precondition: return "precondition";
            case FirstTouchPhase::CaptureAffinity: return "capture-affinity";
            case FirstTouchPhase::CapturePolicy: return "capture-policy";
            case FirstTouchPhase::ResolveNodeCpus: return "resolve-node-cpus";
            case FirstTouchPhase::BindAffinity: return "bind-affinity";
            case FirstTouchPhase::BindPolicy: return "bind-policy";
            case FirstTouchPhase::Touch: return "touch";
            case FirstTouchPhase::Completed: return "completed";
        }
        std::terminate();
    }

    const FirstTouchBoundaryOps &firstTouchDefaultOps() noexcept
    {
        static constexpr FirstTouchBoundaryOps linux_ops{
            .getThreadAffinity = [](void *, cpu_set_t *mask) {
                return ::sched_getaffinity(0, sizeof(*mask), mask);
            },
            .setThreadAffinity = [](void *, const cpu_set_t *mask) {
                return ::sched_setaffinity(0, sizeof(*mask), mask);
            },
            .getMemoryPolicy = [](void *, int *mode, unsigned long *mask, unsigned long maxnode, void *addr, int flags) {
                return static_cast<int>(::get_mempolicy(mode, mask, maxnode, addr, flags));
            },
            .setMemoryPolicy = [](void *, int mode, const unsigned long *mask, unsigned long maxnode) {
                return static_cast<int>(::set_mempolicy(mode, mask, maxnode));
            },
            .nodeCpuSet = linuxNodeCpuSet,
            .touchRange = linuxTouchRange,
        };
        return linux_ops;
    }

    FirstTouchPopulateResult firstTouchPopulateRange(
        volatile void *base, std::size_t bytes, std::size_t page_bytes,
        int numa_node, FirstTouchGeometry geometry, const FirstTouchBoundaryOps &ops)
    {
        if (!base || bytes == 0 || page_bytes == 0 || bytes % page_bytes != 0 ||
            numa_node < 0 || numa_node > geometry.max_node ||
            !ops.getThreadAffinity || !ops.setThreadAffinity || !ops.getMemoryPolicy ||
            !ops.setMemoryPolicy || !ops.nodeCpuSet || !ops.touchRange)
            return {FirstTouchPhase::Precondition, EINVAL};

        const auto layout = policyMaskLayout(geometry.max_node);
        if (!layout)
            return {FirstTouchPhase::Precondition, EINVAL};
        std::vector<unsigned long> original_nodes(layout->words);
        std::vector<unsigned long> target_nodes(layout->words);
        std::vector<unsigned long> readback_nodes(layout->words);
        if (!firstTouchMaskSetNode(target_nodes.data(), target_nodes.size(), numa_node))
            return {FirstTouchPhase::Precondition, EINVAL};

        cpu_set_t original_affinity{};
        if (ops.getThreadAffinity(ops.context, &original_affinity) != 0)
            return {FirstTouchPhase::CaptureAffinity, errno};
        int original_policy = MPOL_DEFAULT;
        if (ops.getMemoryPolicy(ops.context, &original_policy, original_nodes.data(), layout->maxnode, nullptr, 0) != 0)
            return {FirstTouchPhase::CapturePolicy, errno};
        cpu_set_t target_affinity{};
        if (ops.nodeCpuSet(ops.context, numa_node, &target_affinity) != 0)
            return {FirstTouchPhase::ResolveNodeCpus, errno};
        if (CPU_COUNT(&target_affinity) == 0)
            return {FirstTouchPhase::ResolveNodeCpus, EINVAL};

        PrimaryOutcome primary;
        const RestoreThreadPlacement restore(ops, original_affinity, original_policy,
            original_nodes, readback_nodes, layout->maxnode, primary);
        try
        {
            int status = ops.setThreadAffinity(ops.context, &target_affinity);
            if (status != 0)
            {
                const int primary_errno = errno;
                primary.operation = NativeRejection{status, primary_errno};
                return {primary.phase, primary_errno};
            }

            primary.phase = FirstTouchPhase::BindPolicy;
            status = ops.setMemoryPolicy(ops.context, MPOL_BIND, target_nodes.data(), layout->maxnode);
            if (status != 0)
            {
                const int primary_errno = errno;
                primary.operation = NativeRejection{status, primary_errno};
                return {primary.phase, primary_errno};
            }

            primary.phase = FirstTouchPhase::Touch;
            ops.touchRange(ops.context, base, bytes, page_bytes);
            primary.phase = FirstTouchPhase::Completed;
            return {primary.phase, 0};
        }
        catch (...)
        {
            primary.operation = CallbackException{};
            throw;
        }
    }
}
