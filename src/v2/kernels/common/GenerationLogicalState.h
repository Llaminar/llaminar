/**
 * @file GenerationLogicalState.h
 * @brief Typed initial frontier for ordinary and speculative device generation.
 *
 * Unsampled prefill has valid logits and a position, but deliberately no pending
 * condition token. A sampled frontier additionally owns that token. Both use
 * the same existing logical-state rows and one initialization kernel, with no
 * synthetic forward, host state mirror or speculative acceptance transaction.
 */
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_LOGICAL_HD __host__ __device__
#else
#define LLAMINAR_LOGICAL_HD
#endif

namespace llaminar2
{
/** @brief Exactly which producer has completed at this generation boundary. */
enum class GenerationInitialFrontier : uint8_t
{
    UnsampledLogits, ///< Prefill/restore completed; no token has been sampled.
    SampledCondition, ///< A device sampler has produced the pending input token.
};

/** @brief Borrowed, disjoint rows in the canonical logical-state allocation. */
struct GenerationLogicalStateRows
{
    int32_t *base_cached_tokens = nullptr;
    int32_t *target_positions = nullptr;
    int32_t *accepted_state_counts = nullptr;
    int32_t *next_condition_tokens = nullptr;
    int32_t *all_drafts_accepted_flags = nullptr;
    int32_t *stopped_flags = nullptr;
    int32_t *publication_ok_flags = nullptr;

    /** @return Whether every field has its existing arena-owned destination. */
    LLAMINAR_LOGICAL_HD bool valid() const noexcept
    {
        return base_cached_tokens && target_positions && accepted_state_counts &&
            next_condition_tokens && all_drafts_accepted_flags && stopped_flags && publication_ok_flags;
    }
};

/**
 * @brief Complete initialization contract, shared by host validation and GPU code.
 *
 * Position input may alias its target output. Other rows are independently owned.
 * The source kind, rather than a dummy token or a nullable-pointer convention,
 * determines whether a sampled condition must exist. All addresses are borrowed.
 */
struct GenerationLogicalStateInitialization
{
    GenerationInitialFrontier source = GenerationInitialFrontier::UnsampledLogits;
    const int32_t *positions = nullptr; ///< Device-owned prefill/restore positions.
    const int32_t *sampled_tokens = nullptr; ///< Required only for SampledCondition.
    int request_count = 0;
    GenerationLogicalStateRows output;

    /** @return Whether the source and destination bindings agree before enqueue. */
    LLAMINAR_LOGICAL_HD bool valid() const noexcept
    {
        return request_count > 0 && positions && output.valid() &&
            ((source == GenerationInitialFrontier::UnsampledLogits && !sampled_tokens) ||
             (source == GenerationInitialFrontier::SampledCondition && sampled_tokens));
    }

    /**
     * @brief Initialize one exclusively owned request row without advancing KV.
     * @param request Row ordinal, assigned to a single GPU lane or CPU test owner.
     * @return False for malformed geometry or poisoned device input.
     *
     * Read inputs before any writes to support position-row reuse. Invalid input
     * publishes an unhealthy frontier and no usable condition; consumers cannot
     * treat an invalid sampled token as the legitimate unsampled-logits state.
     */
    LLAMINAR_LOGICAL_HD bool initializeRequest(int request) const noexcept
    {
        if (!valid() || request < 0 || request >= request_count)
            return false;
        const int32_t position = positions[request];
        const int32_t condition = source == GenerationInitialFrontier::SampledCondition
            ? sampled_tokens[request] : -1;
        const bool healthy = position >= 0 &&
            (source == GenerationInitialFrontier::UnsampledLogits || condition >= 0);
        output.base_cached_tokens[request] = position;
        output.target_positions[request] = position;
        output.accepted_state_counts[request] = 0;
        output.next_condition_tokens[request] = healthy ? condition : -1;
        output.all_drafts_accepted_flags[request] = 0;
        output.stopped_flags[request] = 0;
        output.publication_ok_flags[request] = healthy ? 1 : 0;
        return healthy;
    }
};
} // namespace llaminar2
#undef LLAMINAR_LOGICAL_HD
