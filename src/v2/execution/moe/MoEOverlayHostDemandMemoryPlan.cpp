/**
 * @file MoEOverlayHostDemandMemoryPlan.cpp
 * @brief Checked composition of canonical route-bank, snapshot and mailbox sizes.
 *
 * Individual payload owners define their layout. This planner only composes
 * their simultaneous lifetimes into a typed BOM before automatic expert filling.
 * It performs no device discovery, allocation, free-memory subtraction or policy
 * adaptation. Runtime owners claim their actual payloads from the admitted PMA.
 */
#include "MoEOverlayHostDemandMemoryPlan.h"
#include "MoEOverlayDistributedResidencyProtocol.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @return Checked setup arithmetic without narrowing an overflowing BOM. */
        std::size_t checkedDemandBytes(unsigned __int128 bytes)
        {
            if (bytes > std::numeric_limits<std::size_t>::max())
                throw std::overflow_error("ExpertOverlay host routing-evidence BOM exceeds size_t");
            return static_cast<std::size_t>(bytes);
        }
    }

    MoEOverlayHostDemandMemoryPlan::MoEOverlayHostDemandMemoryPlan(
        MoEOverlayHostDemandGeometry geometry) : geometry_(geometry)
    {
        if (geometry.num_layers <= 0 || geometry.num_experts <= 0 ||
            geometry.top_k <= 0 || geometry.top_k > geometry.num_experts ||
            geometry.top_k > DecodeExpertHistogram::MAX_TOP_K ||
            geometry.initial_window_rows <= 0 || geometry.maximum_window_rows < 0 ||
            (geometry.maximum_window_rows != 0 &&
             geometry.maximum_window_rows < geometry.initial_window_rows) ||
            geometry.maximum_invocation_rows <= 0)
            throw std::invalid_argument("ExpertOverlay host routing-evidence requires complete model/window/batch geometry");

        capacity_ = {
            static_cast<std::uint32_t>(std::max(geometry.initial_window_rows, geometry.maximum_window_rows)),
            static_cast<std::uint32_t>(geometry.maximum_invocation_rows),
            static_cast<std::uint32_t>(geometry.top_k)};
        // Snapshot and wire authorities validate route extent and their own ABI.
        // Do not reproduce their struct padding or serialization arithmetic here.
        const auto one_observation = DecodeExpertTransactionWindow::maximumAllocationBytes(
            capacity_, geometry.num_layers, geometry.num_experts);
        constexpr std::size_t mutable_rcu_banks = 2;
        constexpr std::size_t forecast_observation = 1;
        constexpr std::size_t incoming_observation = 1;
        constexpr std::size_t retiring_observation = 1;
        mutable_bank_bytes_ = checkedDemandBytes(
            static_cast<unsigned __int128>(mutable_rcu_banks) * geometry.num_layers * capacity_.allocationBytes());
        observation_bytes_ = checkedDemandBytes(
            static_cast<unsigned __int128>(forecast_observation + incoming_observation + retiring_observation) *
            one_observation);
        switch (geometry.publication)
        {
        case MoEOverlayDemandPublicationScope::ProcessLocal:
            break;
        case MoEOverlayDemandPublicationScope::Distributed:
            mailbox_bytes_ = moeOverlayDistributedResidencyProposalWireBytes(
                geometry.num_layers, geometry.num_experts, &capacity_);
            if (mailbox_bytes_ > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::overflow_error("ExpertOverlay routing-evidence mailbox exceeds the MPI count ABI");
            break;
        default:
            throw std::invalid_argument("ExpertOverlay routing-evidence has an invalid publication scope");
        }
        total_bytes_ = checkedDemandBytes(
            static_cast<unsigned __int128>(mutable_bank_bytes_) + observation_bytes_ + mailbox_bytes_);
    }

    ExpertHistogramTransactionConfig MoEOverlayHostDemandMemoryPlan::bind(
        const DecodeExpertHistogramConfig &histogram,
        std::shared_ptr<PhysicalMemoryAuthority> memory) const
    {
        if (!memory || histogram.num_layers != geometry_.num_layers ||
            histogram.num_experts != geometry_.num_experts ||
            histogram.top_k != geometry_.top_k ||
            histogram.window_size != geometry_.initial_window_rows)
            throw std::invalid_argument("ExpertOverlay live demand differs from its admitted model/window identity");
        // The histogram claims its banks from this PMA. Retained snapshots and
        // the MPI mailbox inherit the same capacity; none recomputes a bound.
        return {capacity_, std::move(memory)};
    }

    const moe_overlay_economy::TransactionDemandCapacity &
    MoEOverlayHostDemandMemoryPlan::capacity() const noexcept { return capacity_; }
    const MoEOverlayHostDemandGeometry &
    MoEOverlayHostDemandMemoryPlan::geometry() const noexcept { return geometry_; }
    std::size_t MoEOverlayHostDemandMemoryPlan::mutableBankBytes() const noexcept { return mutable_bank_bytes_; }
    std::size_t MoEOverlayHostDemandMemoryPlan::observationBytes() const noexcept { return observation_bytes_; }
    std::size_t MoEOverlayHostDemandMemoryPlan::mailboxBytes() const noexcept { return mailbox_bytes_; }
    std::size_t MoEOverlayHostDemandMemoryPlan::allocationBytes() const noexcept { return total_bytes_; }
}
