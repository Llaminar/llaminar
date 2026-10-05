/**
 * @file NativeVocabularyAllGatherStage.h
 * @brief Participant-local captured publication of a complete vocabulary.
 *
 * The LM head remains column-sharded. An explicit NCCL/RCCL graph collective
 * transports only device-authorized live rows, then an exact-copy transpose
 * publishes row-major logits for participant-local verifier and sampler work.
 * The enclosing graph and PMA retain all tensor and workspace owners.
 */
#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "backends/GlobalDeviceAddress.h"
#include "collective/NativeCollectiveRows.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "memory/BufferId.h"
#include <string>
#include <vector>

namespace llaminar2
{
    class ILocalTPContext;

    /** @brief Lossless native gather and column assembly on one exact stream. */
    class NativeVocabularyAllGatherStage final : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        /** @brief Immutable participant geometry and already admitted tensor owners. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;
            ILocalTPContext *tp_ctx = nullptr;
            ITensor *local_logits = nullptr;
            ITensor *full_logits = nullptr;
            DeviceRowRange rows = DeviceRowRange::fullyActive(1);
            int local_vocabulary = 0;
            int vocabulary = 0;
            int participant = -1;
            BufferId input_id = BufferId::LOGITS_LOCAL;
            BufferId output_id = BufferId::LOGITS;
            std::string stage_name;
            unsigned long long *payload_bytes = nullptr; ///< Optional device-owned transport evidence.
        };
        static_assert(StageParamsRequired<Params>);

        /** @brief Validate frozen native membership, exact row prefixes and disjoint FP32 owners.
         * @param params Complete graph declaration; setup performs no GPU work.
         * @throws std::invalid_argument For malformed ownership or unsupported unequal shards.
         * @throws std::logic_error For changing native membership. */
        explicit NativeVocabularyAllGatherStage(Params params);
        /** @brief Enqueue the live-row gather and exact assembly, publishing the full tensor.
         * @param ctx Exact participant context with a prepared non-null stream.
         * @return Native collective and assembly enqueue success.
         * @throws std::logic_error For stale membership or missing admitted scratch.
         * @throws std::invalid_argument For foreign contexts or overlapping physical banks. */
        bool execute(IDeviceContext *ctx) override;
        /** @return Explicit native graph collective classification. */
        ComputeStageType type() const override { return ComputeStageType::NATIVE_ALLGATHER; }
        /** @return Stable terminal collective identity. */
        std::string name() const override { return params_.stage_name; }
        /** @return True; every participant must record the same graph edge. */
        bool requiresAllreduce() const override { return true; }
        /** @return True only for the declared compiled CUDA/HIP backend. */
        bool supportsBackend(ComputeBackendType backend) const override;
        /** @return Native capture eligibility after frozen membership validation. */
        bool isGraphCapturable() const override;
        /** @return Zero; transport and assembly preserve representations without arithmetic. */
        std::size_t estimatedFlops() const override { return 0; }
        /** @return Admitted traffic upper bound; replay wire extent is device-counted. */
        std::size_t estimatedMemoryBytes() const override;
        /** @return Frozen operation rows and vocabulary axes, independent of parent capacity. */
        StageBufferRequirements getBufferRequirements() const override;
        /** @return Read-only local logits and the complete row-major output publication. */
        StageBufferContract bufferContract() const override;
        /** @return Tensor and physical geometry metadata without reading device state. */
        StageDumpInfo buildDumpInfoImpl() const override;
        /** @return The executor prepares the full output owner before recording. */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }
        /** @brief Preview the retained family's transpose bank before publication.
         * @param m Operation rows requested by setup, never KV capacity. Larger
         *          family members are priced without changing this stage's rows.
         * @param n Ignored; complete vocabulary is immutable.
         * @param k Ignored; assembly has no reduction dimension.
         * @return The common requirement for the larger of M and this stage's rows.
         * @throws std::invalid_argument For a nonpositive operation row count. */
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        /** @brief Borrow setup-owned scratch; passing null retires the binding.
         * @param workspace Existing physical-memory-authority-backed workspace. */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override { workspace_ = workspace; }
        /** @return Whether setup has published the persistent workspace binding. */
        bool hasWorkspace() const override { return workspace_ != nullptr; }
        /** @return Borrowed workspace owner, never a second allocation authority. */
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }
        /** @return Immutable owners used to authenticate complete terminal graph publication. */
        const Params &params() const noexcept { return params_; }

    private:
        /** @brief Reject changed communicators or unsupported participant ownership. */
        void validateMembership() const;
        /** @brief Authenticate disjoint physical banks when their storage is materialized. */
        void validateBuffers() const;
        const Params params_;
        const std::vector<GlobalDeviceAddress> members_;
        const NativeCollectiveRows rows_;
        DeviceWorkspaceManager *workspace_ = nullptr;
    };
}
