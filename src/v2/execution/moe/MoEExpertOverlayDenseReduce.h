/**
 * @file MoEExpertOverlayDenseReduce.h
 * @brief Correctness-first dense partial reduction for MoE expert-overlay tiers.
 */

#pragma once

#include "MoEExpertParallelPlan.h"

#include <cstddef>
#include <string>
#include <vector>

namespace llaminar2
{
    class IDeviceContext;
    class ITensor;

    struct MoEExpertOverlayDensePartial
    {
        std::string tier_name;
        std::string source_domain;
        const ITensor *tensor = nullptr;
    };

    struct MoEExpertOverlayDenseReduceRequest
    {
        const MoEExpertParallelPlan *plan = nullptr;
        std::vector<MoEExpertOverlayDensePartial> partials;
        ITensor *output = nullptr;
        size_t rows = 0;
        size_t cols = 0;
    };

    /**
     * @brief Host-side model-light reducer for validating overlay tier contracts.
     *
    * The production Bridge Phase 7A GPU/ROCm path is intentionally not wired here. This
     * helper validates that dense FP32 partials are returned from the domains
     * declared by the overlay plan, then reduces them into the continuation
     * output through MoEExpertParallelReduceStage.
     */
    class MoEExpertOverlayDenseReduce
    {
    public:
        static std::vector<std::string> validateRequest(
            const MoEExpertOverlayDenseReduceRequest &request);

        static bool reduceToContinuation(
            const MoEExpertOverlayDenseReduceRequest &request,
            IDeviceContext *ctx);
    };

} // namespace llaminar2