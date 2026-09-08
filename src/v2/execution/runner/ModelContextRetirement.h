/**
 * @file ModelContextRetirement.h
 * @brief Exact final-owner retirement for reusable prepared model contexts.
 *
 * A reusable model contract deliberately keeps prepared GPU weights and sealed
 * workspace backing alive after its last runner has retired.  Destroying the
 * final shared pointers releases those allocations, but it does not prove that
 * CUDA/HIP runtime generations, graph caches, worker streams, or driver-visible
 * capacity have reached the next model-admission boundary.  This API consumes
 * that final ownership edge through TransferEngine's two-phase retirement
 * protocol so JIT replacement and short-lived campaign processes have the same
 * deterministic reclamation semantics.
 */

#pragma once

#include "execution/runner/IOrchestrationRunner.h"

#include <utility>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Complete proof emitted after one prepared model authority retires.
     *
     * CPU-only contracts produce an empty device receipt list after releasing
     * their host owners.  Every GPU row in the sealed allocation BOM produces
     * exactly one runtime-generation receipt.
     */
    struct ModelContextRetirementReceipt
    {
        std::vector<DeviceMemoryReclamationReceipt> device_receipts;

        /** @return Number of physical GPU generations retired. */
        [[nodiscard]] std::size_t retiredDeviceCount() const noexcept
        {
            return device_receipts.size();
        }
    };

    /**
     * @brief Typed pending boundary between allocation sealing and retirement.
     *
     * A sealed CPU-only model has no GPU ticket by construction. A GPU model
     * must own one or more TransferEngine tickets. Keeping those states in a
     * typed plan prevents callers from passing an ambiguous empty vector into
     * the deliberately strict GPU batch API.
     */
    class PendingExclusiveModelRetirement final
    {
    public:
        /** Exact physical authority represented by this pending plan. */
        enum class Kind
        {
            HostOnly,   ///< CPU owners retire through ordinary host ownership.
            DeviceBatch ///< One or more GPU runtime generations must retire.
        };

        PendingExclusiveModelRetirement(
            PendingExclusiveModelRetirement &&) noexcept = default;
        PendingExclusiveModelRetirement &operator=(
            PendingExclusiveModelRetirement &&) noexcept = default;
        PendingExclusiveModelRetirement(
            const PendingExclusiveModelRetirement &) = delete;
        PendingExclusiveModelRetirement &operator=(
            const PendingExclusiveModelRetirement &) = delete;

        /**
         * @brief Capture the exact retirement authority while allocations live.
         * @param retention Sealed unique GPU allocation BOM; empty means CPU-only.
         * @return Pending typed plan whose completion must follow owner release.
         * @throws std::logic_error for an invalid or duplicate GPU row.
         */
        [[nodiscard]] static PendingExclusiveModelRetirement begin(
            const std::vector<ModelDeviceMemoryRetention> &retention);

        /** @return Whether this plan represents host-only ownership. */
        [[nodiscard]] Kind kind() const noexcept { return kind_; }

        /** @return Number of GPU generations awaiting completion. */
        [[nodiscard]] std::size_t pendingDeviceCount() const noexcept
        {
            return tickets_.size();
        }

        /**
         * @brief Complete retirement after every allocation owner was released.
         * @return Host-only or per-device runtime-generation evidence.
         * @throws std::logic_error when called twice or for malformed state.
         * @throws std::runtime_error when exact backend retirement fails.
         */
        [[nodiscard]] ModelContextRetirementReceipt complete();

    private:
        PendingExclusiveModelRetirement(
            Kind kind,
            std::vector<ExclusiveModelRetirementTicket> tickets)
            : kind_(kind), tickets_(std::move(tickets))
        {
        }

        Kind kind_ = Kind::HostOnly;
        std::vector<ExclusiveModelRetirementTicket> tickets_;
        bool completed_ = false;
    };

    /**
     * @brief Retire the exclusive final owner of a reusable model contract.
     *
     * The contract must be at its typed `Reusable` boundary and must be the
     * sole owner of both ModelContext and reusable workspace backing.  Tickets
     * are captured while those owners are live, then the owners are released,
     * and finally each CUDA/HIP runtime generation is reset and certified.
     * This function is intentionally independent of a cache implementation: a
     * production JIT cache may run it on a background worker and join that task
     * before an admission requiring the reclaimed capacity, while a terminating
     * process may call it directly before backend shutdown.
     *
     * @param contract Exclusive reusable contract.  On success its allocation
     *        owners and lifecycle authority are cleared and it cannot be reused.
     * @return Per-device runtime-generation retirement evidence.
     * @throws std::invalid_argument for an incomplete contract.
     * @throws std::logic_error when the contract is not reusable or not the
     *         exclusive final owner.
     * @throws std::runtime_error when exact backend retirement is incomplete.
     */
    [[nodiscard]] ModelContextRetirementReceipt
    retireExclusiveModelContextReuseContract(
        ModelContextReuseContract &contract);

} // namespace llaminar2
