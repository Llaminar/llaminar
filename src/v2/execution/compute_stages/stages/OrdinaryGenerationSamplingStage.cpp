/**
 * @file OrdinaryGenerationSamplingStage.cpp
 * @brief Reuse backend samplers without a host-owned ordinary-generation frontier.
 *
 * Host iteration below runs only while recording graph nodes. Replays contain
 * native kernels and no host callbacks, allocations, transfers or stream waits.
 * Scratch is shared because each request's compact distribution is consumed
 * before the next request overwrites it on the same exact stream.
 */
#include "OrdinaryGenerationSamplingStage.h"
#include "../../../backends/IBackend.h"
#include "../../../memory/BufferId.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
OrdinaryGenerationSamplingStage::OrdinaryGenerationSamplingStage(Params params)
    : IComputeStage(params.device_id), params_(std::move(params))
{
    validate();
}

void OrdinaryGenerationSamplingStage::validate() const
{
    const auto &p = params_;
    const auto &w = p.workspace;
    const auto &publication = p.publication;
    if (!p.device_id.is_gpu() || !p.backend || p.backend->backendDeviceType() != p.device_id.type ||
        !p.logits || p.vocab_size <= 0 ||
        p.logits_row_stride < p.vocab_size || !publication.valid() ||
        publication.history.vocab_size != p.vocab_size ||
        !w.values || !w.indices || w.capacity <= 0 ||
        !w.partial_values || !w.partial_indices || w.partial_capacity <= 0 ||
        publication.sampled_tokens == publication.frontier.next_condition_tokens ||
        publication.sampled_tokens == publication.frontier.cached_tokens ||
        publication.sampled_tokens == publication.frontier.stopped_flags ||
        publication.sampled_tokens == publication.frontier.publication_ok_flags)
        throw std::invalid_argument("OrdinaryGenerationSamplingStage: incomplete or aliased resident binding");
    if (const auto *policy = std::get_if<Stochastic>(&p.policy))
    {
        if (policy->top_k <= 0 || policy->top_k > sampling_math::kMaxTopK ||
            policy->top_k > p.vocab_size || policy->top_k > w.capacity ||
            !std::isfinite(policy->top_p) || policy->top_p <= 0 || policy->top_p > 1 ||
            !std::isfinite(policy->temperature) || policy->temperature <= 0 ||
            !policy->seeds)
            throw std::invalid_argument("OrdinaryGenerationSamplingStage: invalid admitted stochastic policy");
    }
}

bool OrdinaryGenerationSamplingStage::execute(IDeviceContext *ctx)
{
    if (!ensureContext(ctx, "OrdinaryGenerationSamplingStage"))
        return false;
    void *const stream = requireGPUStream();
    if (!stream)
        return false;
    const auto &p = params_;
    const auto &w = p.workspace;
    const auto &publication = p.publication;
    // Sampling owns the scratch token, not the committed next-condition row.
    auto *const samples = publication.sampled_tokens;
    for (int request = 0; request < publication.request_count; ++request)
    {
        auto *logits = p.logits + static_cast<size_t>(request) * p.logits_row_stride;
        const auto *history = publication.history.counts +
            static_cast<size_t>(request) * publication.history.request_stride;
        if (const auto *policy = std::get_if<Stochastic>(&p.policy))
        {
            // A forward owns fresh logits for this transaction. Consume them
            // once with the canonical penalty arithmetic; never reapply the
            // transform to already-processed logits or reconstruct history on
            // the host. Null verifier tokens means durable history only.
            if (p.penalties && !p.backend->enqueueApplyMTPPenaltiesToF32RowsDevice(
                    logits, 1, p.vocab_size, p.logits_row_stride, nullptr,
                    history, p.penalties, p.device_id.gpu_ordinal(), stream))
                return false;
            if (!p.backend->enqueueBuildTopKTopPDistributionF32Device(
                    logits, p.vocab_size, policy->top_k, policy->top_p, policy->temperature,
                    p.device_id.gpu_ordinal(), stream, w.indices, w.values,
                    w.partial_values, w.partial_indices, w.partial_capacity))
                return false;
            // The prefill frontier already names its logits. Decode has just
            // consumed the pending condition, so its logits name position + 1.
            // Publication advances that same position only after sampling.
            const int offset = publication.source == sampling_math::OrdinaryGenerationSampleSource::DecodeLogits ? 1 : 0;
            if (!p.backend->enqueueSampleDistributionF32Device(
                    w.indices, w.values, policy->top_k, 0.0F,
                    p.device_id.gpu_ordinal(), stream, samples + request, nullptr,
                    0, publication.frontier.cached_tokens + request, offset, policy->seeds + request))
                return false;
        }
        else if (p.penalties)
        {
            // Greedy selection already has a fused history-aware reducer: no
            // extra transform kernel or processed-logit allocation is needed.
            if (!p.backend->enqueueArgmaxF32RowsWithHistoryDevice(
                    logits, 1, p.vocab_size, GenerationPenaltyHistory::committed(history, p.penalties),
                    p.device_id.gpu_ordinal(), stream, w.values, samples + request,
                    w.partial_values, w.partial_indices, w.partial_capacity))
                return false;
        }
        else if (!p.backend->enqueueArgmaxF32BatchedRowsDevice(
                     logits, 1, p.vocab_size, p.device_id.gpu_ordinal(), stream,
                     w.values, samples + request, w.partial_values, w.partial_indices, w.partial_capacity))
            return false;
    }
    return p.backend->enqueuePublishOrdinaryGenerationSample(publication, p.device_id.gpu_ordinal(), stream);
}

bool OrdinaryGenerationSamplingStage::supportsBackend(ComputeBackendType backend) const
{
    return backend == ComputeBackendType::GPU_CUDA || backend == ComputeBackendType::GPU_ROCM;
}

StageBufferContract OrdinaryGenerationSamplingStage::bufferContract() const
{
    StageBufferContract contract;
    contract.inputs.reserve(7);
    contract.outputs.reserve(10);
    contract.addInput(BufferId::LOGITS);
    contract.addInput(BufferId::MTP_VERIFIER_STOP_TOKENS);
    contract.addInput(BufferId::MTP_LOGICAL_SEQUENCE_STATE);
    contract.addInput(BufferId::MTP_GENERATION_CONTROL);
    contract.addInput(BufferId::MTP_GENERATED_TOKEN_COUNTS);
    if (params_.penalties)
    {
        contract.addInput(BufferId::MTP_GREEDY_PENALTY_POLICY);
        if (std::holds_alternative<Stochastic>(params_.policy))
            contract.addOutput(BufferId::LOGITS);
    }
    if (std::holds_alternative<Stochastic>(params_.policy))
        contract.addInput(BufferId::SAMPLING_REQUEST_SEEDS);
    contract.addOutput(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS);
    contract.addOutput(BufferId::STOCHASTIC_TARGET_TOKEN_IDS);
    contract.addOutput(BufferId::STOCHASTIC_TARGET_PROBS);
    contract.addOutput(BufferId::STOCHASTIC_TOPK_PARTIAL_VALS);
    contract.addOutput(BufferId::STOCHASTIC_TOPK_PARTIAL_IDXS);
    contract.addOutput(BufferId::MTP_GENERATION_RESPONSE_TOKENS);
    contract.addOutput(BufferId::MTP_GENERATION_CONTROL);
    contract.addOutput(BufferId::MTP_LOGICAL_SEQUENCE_STATE);
    contract.addOutput(BufferId::MTP_GENERATED_TOKEN_COUNTS);
    return contract;
}

StageDumpInfo OrdinaryGenerationSamplingStage::buildDumpInfoImpl() const
{
    StageDumpInfo info;
    info.addScalarInt("requests", params_.publication.request_count);
    info.addScalarInt("vocab_size", params_.vocab_size);
    info.addScalarBool("stochastic", std::holds_alternative<Stochastic>(params_.policy));
    info.addScalarInt("sample_source", static_cast<int>(params_.publication.source));
    return info;
}
} // namespace llaminar2
