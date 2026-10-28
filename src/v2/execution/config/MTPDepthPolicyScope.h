/**
 * @file MTPDepthPolicyScope.h
 * @brief Frozen execution and loaded-model geometry for measured depth policy.
 *
 * An all-zero scope explicitly retains the installed generic policy. A complete
 * scope selects an offline-measured overlay only for exactly matching geometry;
 * no execution mode, device state, or request capacity changes when it misses.
 */
#pragma once
#include "../mtp/MTPVerifierPhysicalWidthPolicy.h"
#include "MTPDepthDefaults.h"
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MTP_SCOPE_HD __host__ __device__
#else
#define LLAMINAR_MTP_SCOPE_HD
#endif
namespace llaminar2
{
    /** @brief Admission-only keys; every non-generic scope has complete geometry. */
    struct MTPDepthPolicyScope
    {
        MTPVerifierPhysicalWidthPolicy verifier_width_policy = MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket;
        int tp_width = 0; ///< Actual uniform TP participants, never a CLI request.
        int pp_stages = 0; ///< Actual local PP stages; remote/composed scopes remain generic.
        int retained_depth = 0; ///< Retained capacity, independent of active fixed depth.
        int hidden_dim = 0; ///< Loaded main-model activation width.
        int main_layers = 0; ///< Original main layers, excluding predictor blocks.
        int expert_count = 0; ///< Loaded routed bank; zero for dense.
        int experts_per_token = 0; ///< Loaded routed selection; zero for dense.

        int compute_capability_major = 0; ///< Frozen inventory architecture pair, never a device-name hash.
        int compute_capability_minor = 0; ///< CUDA native SM or ROCm gfx pair from discovery.

        /** @return Exact immutable identity on host and device. */
        LLAMINAR_MTP_SCOPE_HD bool operator==(const MTPDepthPolicyScope &) const = default;
        /** @return Whether this is an explicitly complete measured scope. */
        LLAMINAR_MTP_SCOPE_HD constexpr bool specified() const { return tp_width != 0; }
        /** @return Whether this is wholly generic or wholly specified and valid. */
        LLAMINAR_MTP_SCOPE_HD constexpr bool valid() const
        {
            const bool known = verifier_width_policy == MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket ||
                verifier_width_policy == MTPVerifierPhysicalWidthPolicy::DynamicDeviceEnvelope ||
                verifier_width_policy == MTPVerifierPhysicalWidthPolicy::ExactLogicalRows;
            if (!specified())
                return verifier_width_policy == MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket &&
                    pp_stages == 0 && retained_depth == 0 && hidden_dim == 0 && main_layers == 0 &&
                    expert_count == 0 && experts_per_token == 0 && compute_capability_major == 0 && compute_capability_minor == 0;
            return known && tp_width > 0 && pp_stages > 0 && retained_depth > 0 && retained_depth <= kMaximumSupportedMTPDraftDepth &&
                hidden_dim > 0 && main_layers > 0 && compute_capability_major > 0 && compute_capability_minor >= 0 && ((expert_count == 0 && experts_per_token == 0) ||
                (expert_count > 0 && experts_per_token > 0 && experts_per_token <= expert_count));
        }
    };
    /** @return Canonical serialization of the physical execution contract. */
    inline constexpr const char *mtpDepthPolicyWidthPolicyName(MTPVerifierPhysicalWidthPolicy policy)
    {
        switch (policy)
        {
        case MTPVerifierPhysicalWidthPolicy::BoundedLogicalBucket: return "bounded_logical_bucket";
        case MTPVerifierPhysicalWidthPolicy::DynamicDeviceEnvelope: return "dynamic_device_envelope";
        case MTPVerifierPhysicalWidthPolicy::ExactLogicalRows: return "exact_logical_rows";
        }
        return "invalid";
    }
}
#undef LLAMINAR_MTP_SCOPE_HD
