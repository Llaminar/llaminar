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
     * @brief Resolve physical width ownership from the declared execution role.
     * @param role Immutable standalone or complete-parent execution ownership.
     * @param depth_mode Admitted request controller mode.
     * @return Exact hosted rows, native dynamic envelope, or standalone buckets.
     * @throws std::invalid_argument for an unknown role or depth mode.
     */
    [[nodiscard]] inline MTPVerifierPhysicalWidthPolicy
    resolveMTPVerifierPhysicalWidthPolicy(
        MTPVerifierExecutionRole role, MTPDepthPolicyMode depth_mode)
    {
        if (depth_mode != MTPDepthPolicyMode::Fixed &&
            depth_mode != MTPDepthPolicyMode::Observe &&
            depth_mode != MTPDepthPolicyMode::Dynamic)
            throw std::invalid_argument("MTP verifier depth mode is unknown");
        switch (role)
        {
        case MTPVerifierExecutionRole::HostedCapturedTransactions:
            return MTPVerifierPhysicalWidthPolicy::ExactLogicalRows;
        case MTPVerifierExecutionRole::NativeConditionalParent:
            return depth_mode == MTPDepthPolicyMode::Dynamic
                ? MTPVerifierPhysicalWidthPolicy::DynamicDeviceEnvelope
                : MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket;
        case MTPVerifierExecutionRole::Standalone:
            return MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket;
        }
        throw std::invalid_argument("MTP verifier execution role is unknown");
    }

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
        MTPVerifierGraphFamily family = MTPVerifierGraphFamily::BoundedBuckets; ///< Admitted identity set.

        /** @return Whether the enabled/disabled geometry is complete. */
        [[nodiscard]] bool valid() const noexcept
        {
            return !enabled ||
                   (draft_depth > 0 && verifier_rows > draft_depth &&
                    mtpVerifierGraphFamilyContains(verifier_rows, verifier_rows, family));
        }
    };

    /**
     * @brief Resolve setup geometry from the admitted retained MTP envelope.
     * @param mtp Frozen capacity and initial request policy.
     * @param maximum_verifier_rows Arena-owned physical row capacity.
     * @param family Immutable graph identities selected from execution ownership.
     * @return Complete disabled or enabled capture geometry.
     * @throws std::invalid_argument For an unknown retained family.
     */
    [[nodiscard]] inline MTPServingForwardCaptureGeometry
    resolveMTPServingForwardCaptureGeometry(
        const MTPRuntimeConfig &mtp,
        int maximum_verifier_rows,
        MTPVerifierGraphFamily family = MTPVerifierGraphFamily::BoundedBuckets)
    {
        (void)mtpVerifierGraphFamilyWidthCount(maximum_verifier_rows, family);
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
            family == MTPVerifierGraphFamily::ExactRows
                ? MTPVerifierPhysicalWidthPolicy::ExactLogicalRows
                : dynamic_depth
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
            .family = family,
        };
    }

    /**
     * @brief Enumerate the complete admitted scalar verifier family largest first.
     * @param mtp Immutable graph capacity and initial serving policy.
     * @param maximum_verifier_rows Exact arena-owned per-request row capacity.
     * @param family_kind Immutable graph identities selected before physical admission.
     * @return Every admitted physical width; empty when execution is disabled.
     * @throws std::invalid_argument when arena and retained capacity disagree.
     *
     * The first graph fixes shared snapshot addresses at their largest extent.
     * The canonical family count and member enumeration also price physical
     * admission. Exact hosted transactions retain every row width; unchanged
     * native/standalone roles retain bounded identities. No live request grows
     * storage, records another graph, or changes the admitted family.
     */
    [[nodiscard]] inline std::vector<MTPServingForwardCaptureGeometry>
    resolveMTPServingForwardCaptureFamily(
        const MTPRuntimeConfig &mtp, int maximum_verifier_rows,
        MTPVerifierGraphFamily family_kind = MTPVerifierGraphFamily::BoundedBuckets)
    {
        const auto widest = resolveMTPServingForwardCaptureGeometry(mtp, maximum_verifier_rows, family_kind);
        if (!widest.enabled) return {};
        if (!widest.valid() || widest.draft_depth != maximum_verifier_rows - 1 ||
            widest.verifier_rows != maximum_verifier_rows)
            throw std::invalid_argument("MTP serving verifier family differs from admitted arena capacity");
        std::vector<MTPServingForwardCaptureGeometry> family;
        const auto count = mtpVerifierGraphFamilyWidthCount(maximum_verifier_rows, family_kind);
        family.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            const int width = mtpVerifierGraphFamilyWidthAt(index, maximum_verifier_rows, family_kind);
            family.push_back({.enabled = true, .draft_depth = width - 1,
                              .verifier_rows = width, .family = family_kind});
        }
        return family;
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
