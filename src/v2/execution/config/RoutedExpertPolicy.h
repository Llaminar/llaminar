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
 * This header defines the orthogonal routed-expert policy axes used by
 * configuration, graph construction, and runtime execution:
 *
 * 1. RoutedExpertComputePolicy says where an expert's weights and GEMMs live.
 * 2. RoutedExpertPhasePolicy says whether a phase executes one assigned copy
 *    or every complete local replica.
 * 3. RoutedExpertWorkloadAssignmentPolicy says which eligible complete
 *    resident executes a row independently for decode and prefill work.
 *
 * Dense/shared-model tensor parallelism is intentionally not represented here;
 * it is described independently by DenseParallelPolicy.
 */

#pragma once

#include "backends/DeviceType.h"

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
     * @enum RoutedExpertPhasePolicy
     * @brief Phase-specific execution over physically resident routed experts.
     *
     * This axis is deliberately separate from RoutedExpertComputePolicy. A
     * phase policy may choose which complete resident computes a row, but it
     * cannot change the physical weight representation promised by the compute
     * policy. In particular, apportioned prefill plus replicated decode requires
     * every participant to own every complete expert.
     */
    enum class RoutedExpertPhasePolicy : uint8_t
    {
        Unspecified = 0, ///< Domain declaration did not state a phase policy.

        /** Use the routed compute policy uniformly in prefill and decode. */
        Uniform,

        /**
         * Apportion complete experts across participants for ordinary prefill,
         * then execute serial decode and grouped verifier rows from each
         * participant's complete local replica. This preserves LLEP's large-M
         * work sharing while removing tiny routed collectives from decode.
         */
        PrefillApportionedDecodeReplicated,
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
     * @struct RoutedExpertWorkloadAssignmentPolicy
     * @brief Independent row-assignment policy for decode and prefill graphs.
     *
     * Grouped MTP verification is decode work even though one verifier graph
     * carries several rows. Treating every `M > 1` graph as prefill used to
     * lower costly LLEP planning into tiny speculative transactions. This
     * value object makes the semantic phase explicit before graph lowering:
     * serial decode and grouped verifier rows use @ref decode, while ordinary
     * prompt/batched-prefill rows use @ref prefill.
     */
    struct RoutedExpertWorkloadAssignmentPolicy
    {
        ///< Assignment for M=1 decode and grouped serial-equivalent verification.
        RoutedExpertAssignmentPolicy decode =
            RoutedExpertAssignmentPolicy::Unspecified;

        ///< Assignment for ordinary prefill and explicitly batched prompt work.
        RoutedExpertAssignmentPolicy prefill =
            RoutedExpertAssignmentPolicy::Unspecified;

        /** @brief Compare both workload-specific assignment axes. */
        bool operator==(
            const RoutedExpertWorkloadAssignmentPolicy &other) const = default;
    };

    /**
     * @brief Resolve an omitted assignment to canonical-owner expert parallelism.
     * @param policy Possibly unspecified assignment policy.
     * @return `StaticOwner` for an omitted policy; otherwise the input value.
     */
    inline RoutedExpertAssignmentPolicy resolveRoutedExpertAssignmentPolicy(
        RoutedExpertAssignmentPolicy policy) noexcept
    {
        return policy == RoutedExpertAssignmentPolicy::Unspecified
                   ? RoutedExpertAssignmentPolicy::StaticOwner
                   : policy;
    }

    /**
     * @enum RoutedExpertRowExecutionPolicy
     * @brief Graph-lowered execution contract for one router-selected row.
     *
     * This is not another user-facing placement axis. Graph construction
     * derives it from the physical routed-expert tier, its phase policy, and
     * the graph phase being built. Keeping the result typed prevents a stage
     * from receiving a fully replicated runtime bank while still applying the
     * participant-assignment policy intended for apportioned prefill.
     *
     * The distinction also owns the collective boundary. Participant-assigned
     * rows publish one partial contribution across the domain, whereas a fully
     * replicated local row is already complete on every participant and must
     * never be reduced with another identical result.
     */
    enum class RoutedExpertRowExecutionPolicy : uint8_t
    {
        /** Exactly one eligible participant executes each selected row. */
        ParticipantAssigned = 0,

        /** Every participant executes every selected row from its full replica. */
        FullyReplicatedLocal,
    };

    /**
     * @enum MoEParticipantPublicationPolicy
     * @brief Graph-lowered publication transaction for partial MoE branches.
     *
     * This policy is derived from model topology after routed-row execution and
     * dense/shared placement are known. It is not a user-facing expert-placement
     * axis: it says how already-computed participant evidence becomes one visible
     * FFN result. Keeping the choice typed prevents graph construction from
     * independently deciding routed and shared collectives and accidentally
     * changing their arithmetic or communication transaction.
     */
    enum class MoEParticipantPublicationPolicy : uint8_t
    {
        /** Routed and shared branches each publish through their own collective. */
        IndependentBranchCollectives = 0,

        /**
         * Routed slots and rank-addressed shared banks use one rooted reduction.
         * The root folds both banks in fixed order, applies the shared gate, and
         * broadcasts only the final combined row.
         */
        CanonicalRootedRankBanks,
    };

    /**
     * @enum MoERouteAccumulationPolicy
     * @brief Arithmetic layout used to accumulate router-selected expert rows.
     *
     * This policy is orthogonal to expert placement and participant
     * publication. It says whether one kernel owns the complete ordered route
     * fold or whether route dots are exposed as independent device work before
     * a dedicated reducer reproduces the same increasing-route addition tree.
     * Keeping the choice typed makes the extra reducer an explicit graph edge
     * instead of an implicit backend convention.
     */
    enum class MoERouteAccumulationPolicy : uint8_t
    {
        /** One output lane visits router slots in increasing order. */
        DirectOrderedFold = 0,

        /**
         * Each router slot publishes one FP32 row, then a following device
         * kernel folds those rows in increasing router-slot order.
         */
        IndependentRouteSlotsThenOrderedFold,
    };

    /**
     * @enum MoERouteAccumulationWorkload
     * @brief Graph workload class relevant to route-accumulation lowering.
     *
     * Ordinary prefill and one-row decode retain their backend's direct ordered
     * publication. A grouped verifier is allowed to expose independent route
     * rows only when the complete graph topology can preserve serial route
     * order in a following device reducer.
     */
    enum class MoERouteAccumulationWorkload : uint8_t
    {
        Ordinary = 0,
        GroupedVerifier,
    };

    /**
     * @struct MoERouteAccumulationSelection
     * @brief Immutable inputs for selecting one graph accumulation topology.
     *
     * The graph builder supplies this value once while constructing a captured
     * executable. Runtime kernels do not inspect host state or mode-shift
     * between accumulation trees after capture.
     */
    struct MoERouteAccumulationSelection
    {
        DeviceType backend = DeviceType::CPU; ///< Backend that owns the graph.
        int participant_count = 0; ///< Number of graph participants in the domain.
        MoERouteAccumulationWorkload workload =
            MoERouteAccumulationWorkload::Ordinary; ///< Fixed captured workload.
    };

    /**
     * @brief Select the route-accumulation topology for a captured MoE graph.
     * @param selection Typed backend, participant, and workload context.
     * @return One immutable production accumulation policy.
     *
     * gfx906 benefits materially from publishing independent route dots for a
     * small grouped verifier: doing so exposes route-level parallelism while a
     * compact reducer restores increasing-router-slot FP32 addition order. The
     * current CUDA kernel already owns an economical deterministic ordered
     * publication, and multi-participant graphs have a separate canonical
     * rank-bank lowering, so neither enters this single-device ROCm topology.
     */
    [[nodiscard]] constexpr MoERouteAccumulationPolicy
    selectMoERouteAccumulationPolicy(
        const MoERouteAccumulationSelection &selection) noexcept
    {
        return selection.backend == DeviceType::ROCm &&
                       selection.participant_count == 1 &&
                       selection.workload ==
                           MoERouteAccumulationWorkload::GroupedVerifier
                   ? MoERouteAccumulationPolicy::
                         IndependentRouteSlotsThenOrderedFold
                   : MoERouteAccumulationPolicy::DirectOrderedFold;
    }

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
     * @brief Return the canonical configuration spelling for a phase policy.
     * @param policy Typed phase-specific execution policy.
     * @return Stable lowercase spelling used by CLI, YAML, and diagnostics.
     */
    inline const char *routedExpertPhasePolicyToString(
        RoutedExpertPhasePolicy policy)
    {
        switch (policy)
        {
        case RoutedExpertPhasePolicy::Unspecified:
            return "unspecified";
        case RoutedExpertPhasePolicy::Uniform:
            return "uniform";
        case RoutedExpertPhasePolicy::PrefillApportionedDecodeReplicated:
            return "prefill-apportioned-decode-replicated";
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
     * @brief Return the canonical diagnostic spelling for a lowered row policy.
     * @param policy Graph-resolved routed-row execution contract.
     * @return Stable lowercase spelling used by graph and stage diagnostics.
     */
    inline const char *routedExpertRowExecutionPolicyToString(
        RoutedExpertRowExecutionPolicy policy)
    {
        switch (policy)
        {
        case RoutedExpertRowExecutionPolicy::ParticipantAssigned:
            return "participant-assigned";
        case RoutedExpertRowExecutionPolicy::FullyReplicatedLocal:
            return "fully-replicated-local";
        }
        return "unknown";
    }

    /**
     * @brief Return the stable diagnostic spelling for MoE publication policy.
     * @param policy Graph-lowered participant publication transaction.
     * @return Lowercase spelling used by graph diagnostics and PerfStats.
     */
    inline const char *moeParticipantPublicationPolicyToString(
        MoEParticipantPublicationPolicy policy)
    {
        switch (policy)
        {
        case MoEParticipantPublicationPolicy::IndependentBranchCollectives:
            return "independent-branch-collectives";
        case MoEParticipantPublicationPolicy::CanonicalRootedRankBanks:
            return "canonical-rooted-rank-banks";
        }
        return "unknown";
    }

    /**
     * @brief Return the stable diagnostic spelling for route accumulation.
     * @param policy Typed local route-accumulation contract.
     * @return Lowercase spelling used by graph diagnostics and PerfStats.
     */
    inline const char *moeRouteAccumulationPolicyToString(
        MoERouteAccumulationPolicy policy)
    {
        switch (policy)
        {
        case MoERouteAccumulationPolicy::DirectOrderedFold:
            return "direct-ordered-fold";
        case MoERouteAccumulationPolicy::IndependentRouteSlotsThenOrderedFold:
            return "independent-route-slots-then-ordered-fold";
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
     * @brief Parse a canonical routed-expert phase policy.
     * @param value CLI or YAML token naming phase-specific execution.
     * @return Typed policy, or `std::nullopt` for an unknown spelling.
     */
    inline std::optional<RoutedExpertPhasePolicy> parseRoutedExpertPhasePolicy(
        const std::string &value)
    {
        const std::string normalized = normalizeRoutedExpertPolicyToken(value);
        if (normalized == "uniform")
            return RoutedExpertPhasePolicy::Uniform;
        if (normalized == "prefill-apportioned-decode-replicated")
            return RoutedExpertPhasePolicy::PrefillApportionedDecodeReplicated;
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
