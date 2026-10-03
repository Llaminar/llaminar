/**
 * @file NativeAllGatherStage.h
 * @brief Participant-local, lossless byte allgather inside a native GPU graph.
 *
 * Projection-partitioned MoE exchanges already encoded intermediate rows and
 * finished output columns. Neither exchange is a floating-point reduction or a
 * strided MPI gather. This stage exposes the native collective explicitly in the
 * graph, borrowing arena storage and the existing communicator. It owns no
 * allocation, placement state, host rendezvous, precision policy or transport
 * alternative. Consumers interpret the rank-major bytes in a separate stage.
 */
#pragma once

#include "CapturedAllGatherStage.h"
#include "../StageParamsBase.h"
#include "backends/GlobalDeviceAddress.h"
#include "memory/BufferId.h"
#include "collective/NativeCollectiveRows.h"

#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    class ILocalTPContext;

    /**
     * @brief Exact-stream native allgather with explicit input/output ownership.
     *
     * The enclosing runner retains the communicator, tensors and arena through
     * graph retirement. Physical rank strides are frozen at construction;
     * device-counted row declarations clip useful payloads on every replay.
     * A participant is a communicator coordinate, not a physical GPU ordinal.
     */
    class NativeAllGatherStage final : public CapturedAllGatherStage
    {
    public:
        /** @brief Immutable graph declaration; both buffers must be arena-bound. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;
            ILocalTPContext *tp_ctx = nullptr; ///< Existing native group, retained by the runner.
            ITensor *local_input = nullptr; ///< Read-only contiguous local byte prefix.
            ITensor *rank_major_output = nullptr; ///< Disjoint receive bank, in communicator order.
            std::size_t bytes_per_participant = 0; ///< Exact physical message, not tensor capacity.
            int participant = -1; ///< Index in tp_ctx->devices(), independent of ordinal.
            std::string stage_name; ///< Stable graph/diagnostic identity.
            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> output_buffer_id;
            std::optional<NativeCollectiveRows> live_rows; ///< INT8 row geometry; never allocation padding.
        };
        static_assert(StageParamsRequired<Params>);

        /**
         * @brief Freeze topology and prove byte geometry without device work.
         * @param params Complete participant-local graph declaration.
         * @throws std::invalid_argument for incomplete, aliased or non-native declarations.
         * @throws std::overflow_error when receive/traffic extents cannot be represented.
         */
        explicit NativeAllGatherStage(Params params);

        /**
         * @brief Enqueue the sole native byte collective and publish its exact stream.
         * @param ctx Owning graph execution context; a missing context is rejected.
         * @return True only when the native enqueue succeeds.
         * @throws std::logic_error for changed membership or an unbound stream.
         * @throws std::invalid_argument for overlapping or overflowing byte ranges.
         * @throws std::runtime_error for missing or foreign device storage.
         */
        bool execute(IDeviceContext *ctx) override;

        /**
         * @brief Authenticate immutable membership and resident byte banks before enqueue.
         *
         * Both ordinary recording and a graph-visible collective fork call this
         * on the producer frontier. The latter then transfers that frontier to
         * its exact auxiliary stream through TransferEngine; neither path can
         * skip physical-range, device, or publication validation.
         * @param execution Exact producer-stream authority, before any fork.
         * @throws std::logic_error for changed membership or a foreign device.
         * @throws std::invalid_argument for aliased or incomplete byte banks.
         * @throws std::runtime_error for missing resident storage/publication.
         */
        void validateEnqueue(const StageGPUExecution &execution) const override;

        /** @return The same immutable arena owners used by ordinary native execution. */
        CapturedAllGatherBuffers exchangeBuffers() const override
        { return {params_.local_input, params_.rank_major_output,
            *params_.input_buffer_id, *params_.output_buffer_id}; }
        /** @brief Enqueue unchanged NCCL/RCCL bytes on an authenticated fork.
         * @param input Exact source/event acquisition produced by TransferEngine.
         * @return Native collective enqueue success; publication belongs to the join.
         * @throws std::invalid_argument For a foreign source or device. */
        bool enqueueAcquiredInput(const AcquiredDeviceTransferInput &input) const override;

        /** @return Dedicated native collective classification, never the MPI gather path. */
        ComputeStageType type() const override { return ComputeStageType::NATIVE_ALLGATHER; }
        /** @return Frozen graph identity used in native collective diagnostics. */
        std::string name() const override { return params_.stage_name; }
        /** @return True so collective scheduling includes this explicit graph edge. */
        bool requiresAllreduce() const override { return true; }
        /** @return Whether the compiled backend matches the declared participant. */
        bool supportsBackend(ComputeBackendType backend) const override;
        /** @return Native capability after authenticating the original membership. */
        bool isGraphCapturable() const override;
        /** @return Zero; copying representations performs no model arithmetic. */
        std::size_t estimatedFlops() const override { return 0; }
        /** @return Local input plus complete receive extent, not a capacity ledger. */
        std::size_t estimatedMemoryBytes() const override { return traffic_bytes_; }
        /** @return Arena-visible input/output shape requirements. */
        StageBufferRequirements getBufferRequirements() const override;
        /** @return Read-only local input and disjoint receive publication. */
        StageBufferContract bufferContract() const override;
        /** @return Tensor/geometry diagnostics without device readback. */
        StageDumpInfo buildDumpInfoImpl() const override;
        /** @return The graph executor prepares output storage before recording. */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }
        /** @return Immutable parameters for graph construction and regressions. */
        const Params &params() const noexcept { return params_; }

    private:
        /** @brief Reject changed/native-incompatible membership without discovering hardware. */
        void validateMembership() const;
        /** @brief Check declared capacity; no backend allocation or transfer is permitted. */
        void validateBuffers() const;
        /** @brief Submit exact bytes; the caller owns input acquisition and output publication. */
        bool enqueueOnStream(void *stream) const;

        const Params params_;
        const std::vector<GlobalDeviceAddress> members_;
        std::size_t receive_bytes_ = 0;
        std::size_t traffic_bytes_ = 0;
    };
} // namespace llaminar2
