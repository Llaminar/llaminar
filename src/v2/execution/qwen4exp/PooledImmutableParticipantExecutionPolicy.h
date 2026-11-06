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
#include <memory>
#include <stdexcept>
namespace llaminar2 { struct GraphConfig; struct MoEExpertOverlayExecutionPlan; class PreparedGPUContextUsePermit; }
namespace llaminar2::qwen4exp {
class PooledImmutableSourceGeometry;
class PooledImmutableLocalHotTransportGeometry;
/** @brief Shared setup/admission validation for the hosted source and its original readers. */
class PooledImmutableParticipantExecutionPolicy final {
public:
    /**
     * @brief Decide whether the original source needs retained cross-device participants.
     * @param source Checked source geometry with independent storage-owner and compute-reader populations.
     * @return True when results have multiple readers or any storage owner differs from the coordinator.
     * @note Cache capacity does not determine communication: a query-only reader still
     * consumes results, and a zero-capacity remote owner still participates in the protocol.
     */
    static bool requiresRetainedParticipants(const PooledImmutableSourceGeometry& source) noexcept;
    /**
     * @brief Admit the combined local/shared participant program against its original pool geometry.
     * @param source Original pool source and its admitted query envelope.
     * @param localHot Optional complete local-cache transport declaration.
     * @return True for a matching combined program, including a single consumer or zero cache capacity.
     * @throws std::invalid_argument If the table, original pool envelope or consumer roster changed.
     */
    static bool requiresRetainedParticipants(const PooledImmutableSourceGeometry& source,
        const PooledImmutableLocalHotTransportGeometry* localHot);
    /** @brief Policy projection only; actual reader membership is authenticated by its retained cohort. */
    enum class GraphRole { Source, ReplicatedReader };
    /**
     * @brief Require the original complete rank-local source execution boundary.
     * @param original Same original resolved ExpertOverlay owner descriptor.
     * @throws std::invalid_argument For absent continuation ownership, distributed/native-only
     *         execution, or a no-CPU declaration outside the original mixed-GPU contract.
     * @note Existing CPU hosted sources keep their original lifecycle. A mixed-GPU
     *       source still requires its exact native auxiliary graph/seal later.
     */
    static void requireHostedSourceExecution(const MoEExpertOverlayExecutionPlan& original);
    /**
     * @brief Check the complete hosted graph descriptor without treating a derived value as identity.
     * @param original Source graph's retained immutable execution descriptor.
     * @param participant Actual participant graph, after original model/cohort membership validation.
     * @param role Source keeps its exact descriptor; a sibling reader derives an equal descriptor.
     * @throws std::invalid_argument For missing, foreign, partial or changed execution policy.
     * @note This grants no source, communicator, native storage or execution capability.
     */
    static void requireGraphDescriptor(
        const std::shared_ptr<const MoEExpertOverlayExecutionPlan>& original,
        const GraphConfig& participant, GraphRole role);
    /**
     * @brief Check an original prepared physical owner against the actual expert execution descriptor.
     * @param original Same immutable descriptor retained by the original source/cohort.
     * @param owner Exact prepared source permit, not a discovered device/TP-domain guess.
     * @return True only for an owned, locally addressable noncontinuation expert participant at its original rank.
     * This projection grants no worker, parent or storage capability; full
     * source/PMA/capture authentication remains mandatory before attachment.
     */
    static bool declaresExpertOwner(const MoEExpertOverlayExecutionPlan& original,
        const PreparedGPUContextUsePermit& owner);
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
