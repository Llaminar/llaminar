/**
 * @file MoEOverlayDevicePhysicalMovement.h
 * @brief Physical interpretation of an authenticated device-authored MoE wave.
 *
 * The all-GPU ExpertOverlay controller is the sole placement authority.  A
 * host transport worker may still need resolved endpoints to move weight
 * bytes, but it must not reconstruct histograms, desired ownership, or policy.
 * This file defines that narrow boundary: it translates one immutable command
 * batch plus the frozen topology into physical migrations, closed durable
 * cycles, shadow-slot demand, and a cross-rank transaction fingerprint.
 */

#pragma once

#include "MoEOverlayDeviceControllerTopology.h"
#include "MoEOverlayDeviceTransportProtocol.h"
#include "MoEOverlayDistributedResidencyProtocol.h"
#include "MoEOverlayResidencyAuthority.h"

#include <cstdint>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Fair bounded host-progress cursor for one parallel physical wave.
     *
     * Every physical operation is submitted before this cursor is used, so
     * limiting host polls does not serialize device DMA or MPI requests.  The
     * cursor only bounds how many driver/MPI progress calls one maintenance
     * quantum may issue.  This prevents a large wave from repeatedly querying
     * every CUDA/HIP event and starving the inference thread's graph launches.
     *
     * Readiness is stored as bytes rather than `vector<bool>` so callers can
     * expose it as a stable span without proxy references.  A non-zero byte is
     * terminal-ready.  Each operation is inspected at most once per quantum,
     * and the rotating cursor prevents a perpetually pending early operation
     * from starving later work.
     */
    class MoEOverlayPhysicalWavePollCursor final
    {
    public:
        /**
         * @brief Bind immutable wave geometry and the per-quantum poll budget.
         * @param operation_count Number of independently submitted operations.
         * @param maximum_polls_per_quantum Topology-derived host progress cap.
         * @throws std::invalid_argument when either value is zero.
         */
        MoEOverlayPhysicalWavePollCursor(
            std::size_t operation_count,
            std::size_t maximum_polls_per_quantum);

        /** @brief Start a new quantum in which each operation is inspected once. */
        void beginQuantum() noexcept { inspected_this_quantum_ = 0u; }

        /**
         * @brief Select the next not-ready operation in fair cyclic order.
         * @param ready Exact byte readiness vector for the complete wave.
         * @return Pending operation index, or no value after one complete scan.
         * @throws std::invalid_argument when @p ready has the wrong size.
         */
        [[nodiscard]] std::optional<std::size_t> nextPending(
            std::span<const std::uint8_t> ready);

        /** @return Maximum progress calls admitted in one maintenance quantum. */
        [[nodiscard]] std::size_t maximumPollsPerQuantum() const noexcept
        {
            return maximum_polls_per_quantum_;
        }

    private:
        const std::size_t operation_count_;
        const std::size_t maximum_polls_per_quantum_;
        std::size_t next_operation_ = 0u;
        std::size_t inspected_this_quantum_ = 0u;
    };

    /**
     * @brief Host-owned byte-transport view of one device-authored transaction.
     *
     * This value contains physical facts only.  `migrations` is a projection of
     * authenticated device commands, never a host-authored placement proposal.
     * Dynamic movements form closed capacity-preserving cycles.  LLEP arrivals
     * are copies and therefore reserve shadow slots without vacating sources or
     * manufacturing durable cycles.  Assignment-only commands remain covered
     * by `command_count` and the fingerprint but require no weight operation.
     */
    struct MoEOverlayDevicePhysicalMovementBatch
    {
        MoEOverlayDeviceControllerTransactionKind kind =
            MoEOverlayDeviceControllerTransactionKind::Invalid;
        std::uint64_t topology_fingerprint = 0u;
        std::uint64_t transaction_id = 0u;
        std::uint64_t base_epoch = 0u;
        std::uint64_t candidate_epoch = 0u;
        std::uint64_t command_digest = 0u;
        std::uint64_t packed_weight_bytes = 0u;
        std::uint32_t command_count = 0u;
        std::uint32_t participant_count = 0u;
        std::uint32_t num_layers = 0u;
        std::uint32_t num_experts = 0u;
        MoEOverlayResidencyTransactionFingerprint transaction_fingerprint;
        std::vector<MoEOverlayTierMigration> migrations;
        std::vector<MoEOverlayTierMigrationCycle> migration_cycles;
        std::vector<MoEOverlayTierShadowRequirement> shadow_requirements;

        /**
         * @return Whether identities, epochs, physical bytes, cycles, and slot
         *         requirements form one self-consistent transport batch.
         */
        [[nodiscard]] bool valid() const noexcept;

        /** @return Whether this batch contains at least one weight arrival. */
        [[nodiscard]] bool movesWeights() const noexcept
        {
            return !migrations.empty();
        }
    };

    /**
     * @brief Resolve physical work from one authenticated device command batch.
     *
     * Integer tier priority is interpreted only as an ordering: a movement to
     * a smaller number is a promotion, a larger number is a demotion, and equal
     * priorities describe same-tier skew rebalancing.  No tier name, backend,
     * rank, socket, or continuation-domain position receives special meaning.
     *
     * @param command Exact immutable bytes acquired from the mapped transport.
     * @param topology Frozen setup-time participant and priority catalogue.
     * @return Complete physical movement and reservation contract.
     * @throws std::invalid_argument for stale identity, malformed commands,
     *         non-capacity-preserving Dynamic movement, or incomplete topology.
     */
    [[nodiscard]] MoEOverlayDevicePhysicalMovementBatch
    makeMoEOverlayDevicePhysicalMovementBatch(
        const MoEOverlayDeviceTransportCommandBatch &command,
        const MoEOverlayDeviceControllerTopology &topology);
} // namespace llaminar2
