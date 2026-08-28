/**
 * @file MoEOverlayPinnedRouteEvidenceViews.h
 * @brief Typed diagnostic views of one request-pinned ExpertOverlay route epoch.
 *
 * Production routing owns two complementary device-resident projections: the
 * overlay-wide expert placement banks and the invocation-local participant
 * assignment for every route slot. This class exposes non-owning tensor views
 * of those exact values to stage-dump capture without introducing a host
 * mirror, a copy, or a second route authority. The backing allocations remain
 * owned by the model runtime table for the entire retained-graph lifetime.
 */

#pragma once

#include "../IComputeStage.h"
#include "../../../execution/moe/MoEOverlayActivationPacketABI.h"
#include "../../../execution/moe/MoEOverlayNodeLocalRouteExchangeABI.h"

#include <cstdint>
#include <memory>

namespace llaminar2
{
    class ITensor;

    /**
     * @brief Non-owning device views for a single layer's acquired route epoch.
     *
     * Construction is intentionally strict. A caller either supplies the
     * complete device-owned route contract or does not construct this object;
     * partial banks, undersized route ledgers, and ambiguous devices cannot
     * become apparently valid diagnostic evidence.
     */
    class MoEOverlayPinnedRouteEvidenceViews final
    {
    public:
        /** @brief Complete graph-bound geometry and device authority. */
        struct Params
        {
            DeviceId device = DeviceId::invalid();
            MoEDomainRouteAssignmentLedger domain_route_assignment{};
            /** Final post-readiness-filter weight for every original route slot. */
            const float *runtime_route_weights = nullptr;
            MoEOverlayRoutePlacementDeviceBinding overlay_route_placement{};
            std::uint32_t physical_rows = 0u;
            std::uint32_t top_k = 0u;
        };

        /**
         * @brief Bind seven pure-device tensor views without copying any value.
         *
         * @param params Exact route-slot geometry and request-pinned authority.
         * @throws std::invalid_argument when any pointer, device, or capacity is
         *         incomplete for the declared route geometry.
         */
        explicit MoEOverlayPinnedRouteEvidenceViews(Params params);

        MoEOverlayPinnedRouteEvidenceViews(
            const MoEOverlayPinnedRouteEvidenceViews &) = delete;
        MoEOverlayPinnedRouteEvidenceViews &operator=(
            const MoEOverlayPinnedRouteEvidenceViews &) = delete;
        MoEOverlayPinnedRouteEvidenceViews(
            MoEOverlayPinnedRouteEvidenceViews &&) = delete;
        MoEOverlayPinnedRouteEvidenceViews &operator=(
            MoEOverlayPinnedRouteEvidenceViews &&) = delete;

        /**
         * @brief Append stable named outputs to an existing stage dump.
         *
         * Snapshot capture performs any diagnostic copy later on the stage's
         * exact producer stream. This method only describes already-live device
         * addresses and therefore has no inference-time side effect.
         *
         * @param info Stage-owned dump description to extend.
         */
        void appendOutputs(StageDumpInfo &info) const;

        /** @return Number of physical router slots described by the views. */
        [[nodiscard]] std::uint32_t routeSlots() const noexcept
        {
            return physical_rows_ * top_k_;
        }

        /** @return Number of experts represented by each placement bank. */
        [[nodiscard]] std::uint32_t expertCount() const noexcept
        {
            return expert_count_;
        }

    private:
        std::uint32_t physical_rows_ = 0u;
        std::uint32_t top_k_ = 0u;
        std::uint32_t expert_count_ = 0u;
        /** Final domain-local participant for each original router slot. */
        std::unique_ptr<ITensor> domain_route_assignment_device_view_;
        /** Exact weight consumed by grouped execution for each route slot. */
        std::unique_ptr<ITensor> runtime_route_weights_device_view_;
        /** Durable global expert-placement bank zero. */
        std::unique_ptr<ITensor> overlay_route_participants_bank0_device_view_;
        /** Epoch carried by durable placement bank zero. */
        std::unique_ptr<ITensor> overlay_route_bank0_epoch_device_view_;
        /** Durable global expert-placement bank one. */
        std::unique_ptr<ITensor> overlay_route_participants_bank1_device_view_;
        /** Epoch carried by durable placement bank one. */
        std::unique_ptr<ITensor> overlay_route_bank1_epoch_device_view_;
        /** Bank acquired by this request's immutable device ticket. */
        std::unique_ptr<ITensor> overlay_route_selected_bank_device_view_;
    };
}
