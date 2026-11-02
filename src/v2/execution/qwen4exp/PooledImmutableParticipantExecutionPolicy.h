/**
 * @file PooledImmutableParticipantExecutionPolicy.h
 * @brief Admit recurring participant execution inside its original retained MTP envelope.
 *
 * Ordinary and speculative forwards share the original whole-request storage
 * interval. Request depth cannot enlarge the retained predictor/verifier family;
 * the existing device controller continues to select live rows and branches.
 */
#pragma once
#include "execution/config/RuntimeConfig.h"
#include "kernels/common/SamplingMath.h"
#include <stdexcept>
namespace llaminar2::qwen4exp {
/** @brief Shared setup/admission validation for the singleton hosted source. */
class PooledImmutableParticipantExecutionPolicy final {
public:
    /** @brief Validate retained capacity and the initial request without repricing it.
     * @param config Original runtime graph/arena envelope and request policy.
     * @throws std::invalid_argument For invalid capacity, batching or request policy. */
    static void requireConfiguration(const MTPRuntimeConfig& config){
        if(config.graph_capacity_draft_tokens<0 ||
           config.graph_capacity_draft_tokens>sampling_math::DeviceGenerationPolicy::kMaximumSupportedDraftDepth ||
           config.max_request_batch!=1)
            throw std::invalid_argument("Recurring remote participants require one request and a valid retained MTP capacity");
        if(const auto error=validateMTPRequestPolicy(makeMTPRequestPolicy(config),config))
            throw std::invalid_argument(*error);
        if(config.enabled && (config.draft_tokens<=0 ||
           resolveMTPMaximumExecutionDraftDepth(config)>sampling_math::DeviceGenerationPolicy::kMaximumSupportedDraftDepth ||
           (config.graph_capacity_draft_tokens>0 &&
            resolveMTPMaximumExecutionDraftDepth(config)>config.graph_capacity_draft_tokens)))
            throw std::invalid_argument("Recurring remote MTP execution exceeds its original retained graph capacity");
    }
    /** @brief Authenticate a controller policy against the original retained family.
     * @param policy Actual request controller policy, including adaptive ceiling.
     * @param retained Original graph/arena capacity; never inferred from live rows.
     * @throws std::invalid_argument For invalid or out-of-envelope execution. */
    static void requireAdmission(const sampling_math::DeviceGenerationPolicy& policy,
                                 const MTPRuntimeConfig& retained){
        requireConfiguration(retained);
        if(!policy.valid() ||
           (!policy.isOrdinary() && !policy.isForwardOnly() &&
            policy.maximum_depth>resolveMTPRetainedDraftCapacity(retained)))
            throw std::invalid_argument("Recurring remote execution requires an exact valid policy within its original retained MTP family");
    }
};
}
