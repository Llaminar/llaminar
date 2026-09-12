/**
 * @file MTPServingForwardCaptureGeometry.h
 * @brief Setup-owned verifier geometry and shared checkpoint capture ordering.
 *
 * A retained runner separates its immutable graph capacity from the execution
 * policy chosen by each request. Both forward construction and snapshot-arena
 * ordering consume this geometry; neither may derive capacity from a shallow
 * request and later grow storage beneath an already captured pointer.
 */
#pragma once

#include "MTPDeviceGenerationPolicy.h"
#include "MTPVerifierPolicy.h"

namespace llaminar2
{
    /**
     * @brief Immutable row geometry for setup-owned MTP forward capture.
     *
     * The same value controls both graph construction and diagnostic-arena
     * materialization order. Keeping it typed prevents the serving-family
     * caller and the grouped-verifier builder from independently deriving
     * different fixed/dynamic depth envelopes.
     */
    struct MTPServingForwardCaptureGeometry
    {
        bool enabled = false; ///< Whether this runner retains an MTP family.
        int draft_depth = 0; ///< Maximum draft rows represented by capture.
        int verifier_rows = 0; ///< Physical grouped-verifier row capacity.

        /** @return Whether the enabled/disabled geometry is complete. */
        [[nodiscard]] bool valid() const noexcept
        {
            return !enabled ||
                   (draft_depth > 0 && verifier_rows > draft_depth);
        }
    };

    /**
     * @brief Resolve setup geometry from the admitted retained MTP envelope.
     * @param mtp Frozen capacity and initial request policy.
     * @param maximum_verifier_rows Arena-owned physical row capacity.
     * @return Complete disabled or enabled capture geometry.
     */
    [[nodiscard]] inline MTPServingForwardCaptureGeometry
    resolveMTPServingForwardCaptureGeometry(
        const MTPRuntimeConfig &mtp,
        int maximum_verifier_rows)
    {
        if (!mtp.enabled)
            return {};

        const auto depth_policy =
            resolveMTPDeviceGenerationDepthPolicy(mtp);
        const bool dynamic_depth =
            depth_policy.mode ==
            sampling_math::DeviceGenerationPolicyMode::Dynamic;
        // Initial depth is request data, not graph capacity. A shallow first
        // request can retain a deeper family; materialize that family's widest
        // verifier before another graph freezes the shared snapshot address.
        const int draft_depth = resolveMTPRetainedDraftCapacity(mtp);
        const int logical_verifier_rows = draft_depth + 1;
        const auto physical_policy =
            dynamic_depth
                ? MTPVerifierPhysicalWidthPolicy::DynamicDeviceEnvelope
                : MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket;
        const int verifier_rows =
            depth_policy.valid() && draft_depth > 0
                ? mtpVerifierPhysicalPaddedSeqLen(
                      /*request_count=*/1,
                      logical_verifier_rows,
                      maximum_verifier_rows,
                      physical_policy)
                : 0;
        return MTPServingForwardCaptureGeometry{
            .enabled = true,
            .draft_depth = draft_depth,
            .verifier_rows = verifier_rows,
        };
    }

    /**
     * @brief Select which wide checkpoint graph freezes the shared address.
     *
     * Snapshot storage for a fixed graph topology grows monotonically with
     * physical rows. A grouped verifier wins a row-count tie because its
     * terminal outcome publications are additional to the model stages.
     * This setup-only decision has no effect when snapshots are disabled.
     *
     * @param snapshots_enabled Whether captured diagnostic copies exist.
     * @param largest_prefill_rows Largest admitted prefill bucket.
     * @param mtp_geometry Frozen grouped-verifier geometry.
     * @return true when grouped verification must be captured first.
     */
    [[nodiscard]] inline bool materializeMTPWideCheckpointLaneFirst(
        bool snapshots_enabled,
        int largest_prefill_rows,
        const MTPServingForwardCaptureGeometry &mtp_geometry) noexcept
    {
        return snapshots_enabled && mtp_geometry.enabled &&
               mtp_geometry.valid() && largest_prefill_rows > 0 &&
               mtp_geometry.verifier_rows >= largest_prefill_rows;
    }

}
