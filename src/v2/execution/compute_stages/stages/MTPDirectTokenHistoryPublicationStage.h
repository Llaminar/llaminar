/**
 * @file MTPDirectTokenHistoryPublicationStage.h
 * @brief Retained device-only history commit for one newly emitted MTP token.
 *
 * Direct responses and explicit policy tokens have no grouped outcome whose
 * publication could update durable sampling history. This single-node helper
 * consumes the original target slot and increments the existing arena history
 * with the same backend arithmetic used by ordinary forced-token publication.
 * The same node normalizes only the existing policy's device-owned history bit
 * for the successor fresh-logits boundary. Nonterminal direct tokens enter the
 * committed main row before any successor sampling; terminal direct tokens have
 * no successor. It owns no payload, sampler or controller mirror. Carried
 * outputs never launch this graph and retain their grouped publication state.
 */
#pragma once
#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "backends/IBackend.h"
#include "memory/BufferId.h"
#include "execution/mtp/MTPVerifierOutcomeGraph.h"

namespace llaminar2
{
/** @brief Publish one target-slot token into the device-owned generation history. */
class MTPDirectTokenHistoryPublicationStage final : public IComputeStage
{
public:
    /** @brief Immutable original scalar source and durable history bindings. */
    struct Params
    {
        STAGE_PARAMS_COMMON_FIELDS;
        IBackend *backend = nullptr; ///< Exact participant backend.
        const int32_t *token_device = nullptr; ///< Original target slot zero.
        int32_t *counts_device = nullptr; ///< Existing packed prompt/generated history.
        MTPGreedyPenaltyPolicy *policy_device = nullptr; ///< Original mutable history-frontier owner.
        int vocab_size = 0; ///< Exact admitted vocabulary extent.
    };
    static_assert(StageParamsRequired<Params>);

    /** @brief Retain immutable arena pointers; no memory is allocated by this stage. */
    explicit MTPDirectTokenHistoryPublicationStage(Params params)
        : IComputeStage(params.device_id), params_(params) {}

    /** @brief Record the existing history kernel on the explicit graph stream.
     * @param ctx Exact execution context supplied by the retained graph owner.
     * @return Whether the original scalar/history binding was enqueued.
     */
    bool execute(IDeviceContext *ctx) override
    {
        if (!ensureContext(ctx, "MTPDirectTokenHistoryPublicationStage") ||
            !params_.device_id.is_gpu() || !params_.backend || !params_.token_device ||
            !params_.counts_device || !params_.policy_device || params_.vocab_size <= 0 ||
            params_.backend->backendDeviceType() != params_.device_id.type)
            return false;
        void *const stream = requireGPUStream();
        if (!stream) return false;
        return params_.backend->enqueueCommitMTPDirectTokenHistoryDevice(
            params_.token_device, params_.vocab_size, params_.counts_device, params_.policy_device,
            params_.device_id.gpu_ordinal(), stream);
    }
    /** @return Dedicated diagnostic stage identity. */
    ComputeStageType type() const override { return ComputeStageType::MTP_DIRECT_TOKEN_HISTORY_PUBLICATION; }
    /** @return Stable graph node label. */
    std::string name() const override { return "mtp_direct_token_history_publication"; }
    /** @return One integer histogram increment. */
    size_t estimatedFlops() const override { return 1; }
    /** @return Scalar/history traffic plus one resident history-frontier write. */
    size_t estimatedMemoryBytes() const override { return 4 * sizeof(int32_t); }
    /** @return Native GPU backends that implement the existing kernel. */
    bool supportsBackend(ComputeBackendType backend) const override
    { return backend == ComputeBackendType::GPU_CUDA || backend == ComputeBackendType::GPU_ROCM; }
    /** @return True: recording contains no allocation, transfer or host callback. */
    bool isGraphCapturable() const override { return true; }
    /** @return This stage has no cross-participant communication. */
    bool isCollectiveStage() const override { return false; }
    /** @return Arena/event ownership is published by the enclosing helper. */
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
    /** @return Immutable source metadata for exact retained-cache authentication. */
    const Params &getParams() const noexcept { return params_; }
    /** @return Exact read/write ownership; history contents remain replay data. */
    StageBufferContract bufferContract() const override
    {
        StageBufferContract contract;
        contract.addInput(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS);
        contract.addInput(BufferId::MTP_GENERATED_TOKEN_COUNTS);
        contract.addOutput(BufferId::MTP_GENERATED_TOKEN_COUNTS);
        contract.addInput(BufferId::MTP_GREEDY_PENALTY_POLICY);
        contract.addOutput(BufferId::MTP_GREEDY_PENALTY_POLICY);
        return contract;
    }
protected:
    /** @return Bounded geometry metadata, never a live token or count readback. */
    StageDumpInfo buildDumpInfoImpl() const override
    {
        StageDumpInfo info;
        info.addScalarInt("vocab_size", params_.vocab_size);
        info.addScalarInt("target_slot", 0);
        return info;
    }
private:
    Params params_;
};
} // namespace llaminar2
