/**
 * @file NativeAllGatherStage.h
 * @brief Participant-local, lossless byte allgather inside a native GPU graph.
 *
 * Projection-partitioned MoE exchanges already encoded intermediate rows and
 * finished output columns. Neither exchange is a floating-point reduction or a
 * strided MPI gather. This stage exposes the native collective explicitly in the
 * graph, borrowing arena storage and the existing communicator. A typed terminal
 * may specialize diagnostic projection semantics while inheriting this exact
 * transport, validation and capture/publication implementation. It owns no
 * allocation, placement state, host rendezvous, precision policy or transport
 * alternative. Consumers interpret the rank-major bytes in a separate stage.
 */
#pragma once

#include "CapturedAllGatherStage.h"
#include "../StageParamsBase.h"
#include "execution/mtp/DeviceGenerationCollectiveReceipt.h"
#include "backends/GlobalDeviceAddress.h"
#include "memory/BufferId.h"
#include "collective/NativeCollectiveRows.h"
#include "tensors/TensorClasses.h"

#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    class ILocalTPContext;
    class TensorBase;
    class IBackend;

    /**
     * @brief Exact-stream native allgather with explicit input/output ownership.
     *
     * The enclosing runner retains the communicator, tensors and arena through
     * graph retirement. Physical rank strides are frozen at construction;
     * device-counted row declarations clip useful payloads on every replay.
     * A participant is a communicator coordinate, not a physical GPU ordinal.
     */
    class NativeAllGatherStage : public CapturedAllGatherStage
    {
    public:
        /**
         * @brief Arena-owned transport prefix resolved after ordinary materialization.
         *
         * The scalar is produced by a graph node, not sampled by the host. A
         * canonical INT32 root and byte offset keep owner/event authority visible
         * while its cold device pointer does not yet exist. This transport count
         * cannot replace an original request/history count.
         */
        struct ArenaRows
        {
            TensorBase *owner = nullptr; ///< Existing canonical arena root; enclosing graph retains it.
            BufferId buffer = BufferId::_COUNT; ///< Exact count producer/consumer role.
            std::size_t byte_offset = 0; ///< Aligned scalar offset inside owner.
            int capacity = 0; ///< Positive admitted physical rows, independent of live value.
            std::size_t bytes_per_row = 0; ///< Exact native bytes per live row.
        };

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
            std::optional<NativeCollectiveRows> live_rows; ///< Raw immutable row declaration; excludes arena_rows.
            std::optional<ArenaRows> arena_rows; ///< Cold arena count declaration; excludes live_rows.
            std::optional<DeviceGenerationCollectiveReceipt> payload_receipt; ///< Typed original control-row observation; requires arena_rows.
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
         * @brief Seal an already prepared arena count, without allocating or reading it.
         * @param ctx Exact participant context owning the operation.
         * @param stream Explicit upcoming graph stream.
         * @return True after the immutable physical count identity is sealed.
         * @throws std::logic_error For preparation during capture or changed sealed storage.
         * @throws std::invalid_argument For missing/foreign/unprepared count storage.
         */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        /** @return Only arena-count declarations require setup before native recording. */
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        { return params_.arena_rows ? GraphLaunchPreparationPolicy::CaptureOnly : GraphLaunchPreparationPolicy::None; }
        /** @return Cold declarations remain eligible for ordinary executor preparation. */
        bool supportsGraphCaptureAfterLaunchPreparation() const override
        { return supportsLazyPrefillGraphCapturePreflight(); }
        /** @return Membership/compiled backend eligibility, independent of cold count storage. */
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        /** @return Counted banks clip padded rows; fixed byte messages retain their existing contract. */
        bool supportsPaddedPrefillRealLengthContract() const override
        { return params_.arena_rows.has_value() || params_.live_rows.has_value(); }
        /** @return Complete immutable declaration; sealed storage drift is fatal. */
        std::uint64_t graphCaptureVariantSignature() const override;

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
            *params_.input_buffer_id, *params_.output_buffer_id,
            params_.arena_rows ? params_.arena_rows->owner : nullptr,
            params_.payload_receipt ? params_.payload_receipt->owner() : nullptr}; }
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
        /** @brief Validate count declaration and any existing physical scalar extent. */
        void validateCountDeclaration() const;
        /** @brief Reject missing/stale prepared count owner, pointer, device or runtime generation. */
        void validateCountSeal() const;
        /** @return Sealed counted geometry; never read scalar contents or substitute another authority. */
        NativeCollectiveRows arenaRows() const;
        /** @brief Submit exact bytes; the caller owns input acquisition and output publication. */
        bool enqueueOnStream(void *stream) const;

        const Params params_;
        const std::vector<GlobalDeviceAddress> members_;
        std::size_t receive_bytes_ = 0;
        std::size_t traffic_bytes_ = 0;
        /** @brief Capture-only metadata; no live count value is cached here. */
        struct CountSeal
        {
            const void *base;
            std::size_t owner_bytes;
            IBackend *backend;
            std::uint64_t runtime_generation;
        };
        std::optional<CountSeal> count_seal_; ///< Immutable after first preparation; ordinary reset never rebinds it.
    };
} // namespace llaminar2
