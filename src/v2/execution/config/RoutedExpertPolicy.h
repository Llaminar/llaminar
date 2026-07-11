/**
 * @file RoutedExpertPolicy.h
 * @brief Canonical policy axes for distributing and assigning routed MoE experts.
 *
 * A former umbrella label described several incompatible execution strategies
 * in Llaminar. In particular, it was used both for
 * assigning whole experts to different participants and for tensor-sharding
 * every expert across those participants.  Those strategies have different
 * weight layouts, collective requirements, and state-publication lifetimes, so
 * they must never share an enum value or rely on a caller-specific meaning.
 *
 * This header defines the two orthogonal routed-expert policy axes used by
 * configuration, graph construction, and runtime execution:
 *
 * 1. RoutedExpertComputePolicy says where an expert's weights and GEMMs live.
 * 2. RoutedExpertAssignmentPolicy says which eligible resident participant
 *    executes a router-selected row.
 *
 * Dense/shared-model tensor parallelism is intentionally not represented here;
 * it is described independently by DenseParallelPolicy.
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>

namespace llaminar2
{
    /**
     * @enum RoutedExpertComputePolicy
     * @brief Physical weight and GEMM distribution for each routed expert.
     *
     * This axis answers only how an expert is represented and computed inside
     * one execution domain. It does not select the participant for a routed row,
     * describe the hardware scope, or choose the dense-model policy.
     */
    enum class RoutedExpertComputePolicy : uint8_t
    {
        Unspecified = 0, ///< Domain declaration did not provide a routed compute policy.

        /** Every participant owns and computes every routed expert in full. */
        Replicated,

        /**
         * Whole experts are assigned to participants.  One eligible participant
         * computes a selected expert's complete gate/up/down path, and routed
         * partial outputs are combined across the domain.
         */
        Apportioned,

        /**
         * Every participant owns a tensor shard of every routed expert.  All
         * participants compute each selected expert and reduce its partial down
         * projection.  This is routed-expert tensor parallelism, not whole-expert
         * apportionment.
         */
        TensorSharded,
    };

    /**
     * @enum RoutedExpertAssignmentPolicy
     * @brief Scheduling policy among residents that hold a complete expert.
     *
     * Assignment is meaningful after routing has produced an expert ID. It may
     * select among eligible complete residents, but it must not change the
     * router's expert ID or imply a different weight-distribution policy.
     */
    enum class RoutedExpertAssignmentPolicy : uint8_t
    {
        Unspecified = 0, ///< Domain declaration did not provide an assignment policy.
        StaticOwner,    ///< Use the canonical owner/resident selected by placement.

        /**
         * Preserve router expert IDs while selecting the least-loaded eligible
         * resident participant for each row in the current scheduling window.
         */
        LeastLoadedResident,
    };

    /**
     * @brief Return the canonical configuration spelling for a compute policy.
     * @param policy Typed compute-distribution value to render.
     * @return Stable lowercase spelling used by CLI, YAML, and diagnostics.
     */
    inline const char *routedExpertComputePolicyToString(
        RoutedExpertComputePolicy policy)
    {
        switch (policy)
        {
        case RoutedExpertComputePolicy::Unspecified:
            return "unspecified";
        case RoutedExpertComputePolicy::Replicated:
            return "replicated";
        case RoutedExpertComputePolicy::Apportioned:
            return "apportioned";
        case RoutedExpertComputePolicy::TensorSharded:
            return "tensor-sharded";
        }
        return "unknown";
    }

    /**
     * @brief Return the canonical configuration spelling for an assignment policy.
     * @param policy Typed routed-row scheduling value to render.
     * @return Stable lowercase spelling used by CLI, YAML, and diagnostics.
     */
    inline const char *routedExpertAssignmentPolicyToString(
        RoutedExpertAssignmentPolicy policy)
    {
        switch (policy)
        {
        case RoutedExpertAssignmentPolicy::Unspecified:
            return "unspecified";
        case RoutedExpertAssignmentPolicy::StaticOwner:
            return "static-owner";
        case RoutedExpertAssignmentPolicy::LeastLoadedResident:
            return "least-loaded-resident";
        }
        return "unknown";
    }

    /**
     * @brief Normalize case and separator style for one policy token.
     * @param value User-provided CLI or YAML token.
     * @return Lowercase token with underscores represented as hyphens.
     *
     * Normalization handles spelling mechanics only. The parsers below still
     * compare against the canonical semantic names and intentionally do not map
     * old umbrella labels onto one of the new policy axes.
     */
    inline std::string normalizeRoutedExpertPolicyToken(std::string value)
    {
        std::transform(
            value.begin(),
            value.end(),
            value.begin(),
            [](unsigned char c)
            {
                return static_cast<char>(std::tolower(c));
            });
        std::replace(value.begin(), value.end(), '_', '-');
        return value;
    }

    /**
     * @brief Parse a canonical routed-expert compute policy.
     * @param value CLI or YAML value naming the physical expert distribution.
     * @return The typed policy, or `std::nullopt` when the value is not one of
     *         `replicated`, `apportioned`, or `tensor-sharded`.
     */
    inline std::optional<RoutedExpertComputePolicy> parseRoutedExpertComputePolicy(
        const std::string &value)
    {
        const std::string normalized = normalizeRoutedExpertPolicyToken(value);
        if (normalized == "replicated")
            return RoutedExpertComputePolicy::Replicated;
        if (normalized == "apportioned")
            return RoutedExpertComputePolicy::Apportioned;
        if (normalized == "tensor-sharded")
            return RoutedExpertComputePolicy::TensorSharded;
        return std::nullopt;
    }

    /**
     * @brief Parse a canonical routed-row assignment policy.
     * @param value CLI or YAML value naming resident row scheduling.
     * @return The typed policy, or `std::nullopt` when the value is not
     *         `static-owner` or `least-loaded-resident`.
     */
    inline std::optional<RoutedExpertAssignmentPolicy> parseRoutedExpertAssignmentPolicy(
        const std::string &value)
    {
        const std::string normalized = normalizeRoutedExpertPolicyToken(value);
        if (normalized == "static-owner")
            return RoutedExpertAssignmentPolicy::StaticOwner;
        if (normalized == "least-loaded-resident")
            return RoutedExpertAssignmentPolicy::LeastLoadedResident;
        return std::nullopt;
    }

} // namespace llaminar2
