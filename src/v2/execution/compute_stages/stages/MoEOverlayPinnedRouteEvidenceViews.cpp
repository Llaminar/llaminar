/**
 * @file MoEOverlayPinnedRouteEvidenceViews.cpp
 * @brief Implements zero-copy diagnostic views of pinned ExpertOverlay routes.
 */

#include "MoEOverlayPinnedRouteEvidenceViews.h"

#include "../../../tensors/GpuTensorView.h"

#include <limits>
#include <stdexcept>

namespace llaminar2
{
    MoEOverlayPinnedRouteEvidenceViews::
        MoEOverlayPinnedRouteEvidenceViews(Params params)
    {
        const std::uint64_t route_slots =
            static_cast<std::uint64_t>(params.physical_rows) *
            static_cast<std::uint64_t>(params.top_k);
        if (!params.device.is_gpu() || params.physical_rows == 0u ||
            params.top_k == 0u || route_slots == 0u ||
            route_slots >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::uint32_t>::max()) ||
            !params.domain_route_assignment.validFor(
                static_cast<std::uint32_t>(route_slots)) ||
            !params.runtime_route_weights.validFor(
                params.physical_rows, params.top_k) ||
            !params.overlay_route_placement.valid())
        {
            throw std::invalid_argument(
                "Pinned ExpertOverlay route evidence requires one complete GPU route epoch, final runtime weights, and exact positive geometry");
        }

        physical_rows_ = params.physical_rows;
        top_k_ = params.top_k;
        expert_count_ = params.overlay_route_placement.expert_count;

        /* These views never own or move storage. They only give the snapshot
         * machinery typed geometry for allocations already retained by the
         * runtime table and ordered by the enclosing stage's producer stream. */
        domain_route_assignment_device_view_ =
            std::make_unique<GpuTensorView>(
                const_cast<std::int32_t *>(
                    params.domain_route_assignment.participant_ids),
                static_cast<std::size_t>(physical_rows_),
                static_cast<std::size_t>(top_k_),
                TensorType::INT32,
                params.device);
        runtime_route_weights_device_view_ =
            std::make_unique<GpuTensorView>(
                const_cast<float *>(params.runtime_route_weights.weights),
                static_cast<std::size_t>(physical_rows_),
                static_cast<std::size_t>(top_k_),
                TensorType::FP32,
                params.device);
        overlay_route_participants_bank0_device_view_ =
            std::make_unique<GpuTensorView>(
                const_cast<std::int32_t *>(
                    params.overlay_route_placement.banks[0]
                        .route_participants),
                1u,
                static_cast<std::size_t>(expert_count_),
                TensorType::INT32,
                params.device);
        overlay_route_bank0_epoch_device_view_ =
            std::make_unique<GpuTensorView>(
                const_cast<std::uint32_t *>(
                    params.overlay_route_placement.banks[0].epoch),
                1u,
                1u,
                TensorType::INT32,
                params.device);
        overlay_route_participants_bank1_device_view_ =
            std::make_unique<GpuTensorView>(
                const_cast<std::int32_t *>(
                    params.overlay_route_placement.banks[1]
                        .route_participants),
                1u,
                static_cast<std::size_t>(expert_count_),
                TensorType::INT32,
                params.device);
        overlay_route_bank1_epoch_device_view_ =
            std::make_unique<GpuTensorView>(
                const_cast<std::uint32_t *>(
                    params.overlay_route_placement.banks[1].epoch),
                1u,
                1u,
                TensorType::INT32,
                params.device);
        overlay_route_selected_bank_device_view_ =
            std::make_unique<GpuTensorView>(
                const_cast<std::uint32_t *>(
                    &params.overlay_route_placement.status->bank),
                1u,
                1u,
                TensorType::INT32,
                params.device);
    }

    void MoEOverlayPinnedRouteEvidenceViews::appendOutputs(
        StageDumpInfo &info) const
    {
        info.addOutput(
                "domain_route_participant_ids",
                domain_route_assignment_device_view_.get(),
                static_cast<std::size_t>(physical_rows_),
                static_cast<std::size_t>(top_k_))
            .addOutput(
                "runtime_route_weights",
                runtime_route_weights_device_view_.get(),
                static_cast<std::size_t>(physical_rows_),
                static_cast<std::size_t>(top_k_))
            .addOutput(
                "overlay_route_participants_bank0",
                overlay_route_participants_bank0_device_view_.get(),
                1u,
                static_cast<std::size_t>(expert_count_))
            .addOutput(
                "overlay_route_bank0_epoch",
                overlay_route_bank0_epoch_device_view_.get(),
                1u,
                1u)
            .addOutput(
                "overlay_route_participants_bank1",
                overlay_route_participants_bank1_device_view_.get(),
                1u,
                static_cast<std::size_t>(expert_count_))
            .addOutput(
                "overlay_route_bank1_epoch",
                overlay_route_bank1_epoch_device_view_.get(),
                1u,
                1u)
            .addOutput(
                "overlay_route_selected_bank",
                overlay_route_selected_bank_device_view_.get(),
                1u,
                1u);
    }
}
