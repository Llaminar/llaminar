/**
 * @file MoEOverlayActivationPacketStages.h
 * @brief Typed graph stages for node-local ExpertOverlay activation packets.
 *
 * These stages are the declarative bridge between model graphs and the
 * backend-neutral activation packet ABI. Topology planning resolves an exact
 * mapped lane and participant role during setup; each stage records its packet
 * kernel and adjacent mapped-timeline edge into one complete endpoint graph on
 * the exact non-null stream. No stage chooses a backend, rank, tier,
 * continuation owner, or participant count.
 *
 * The four operations remain separate types so their read/write authority and
 * lifecycle cannot be confused: continuation dispatch pack, follower dispatch
 * consume, follower return pack, and continuation return consume. Their mapped
 * pointers are model-lifetime graph identity and must outlive every retained
 * parent transaction that imports these stages.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "../../../memory/BufferId.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace llaminar2
{
    class IBackend;
    class IMoEKernel;
    class TensorBase;

    /**
     * @brief Pack one planner-targeted sparse dispatch packet on continuation.
     *
     * The router-owned indices, route weights, and hidden rows are the same
     * typed arena tensors consumed by the ordinary local MoE path. Both
     * placement-bank target arrays remain stable raw device allocations in the
     * model runtime table. The request ticket selects one bank on device, so
     * migration cannot expose a host-side owner decision or require a second
     * route representation.
     */
    class MoEOverlayActivationDispatchPackStage final : public IComputeStage
    {
    public:
        /** @brief Immutable capture geometry and source bindings. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;
            MoEOverlayMappedActivationDeviceLane lane;
            TensorBase *hidden = nullptr;
            TensorBase *routing_indices = nullptr;
            TensorBase *routing_weights = nullptr;
            /** Ticket, banks, and geometry resolved from one runtime table. */
            MoEOverlayRoutePlacementDeviceBinding placement{};
            const std::int32_t *active_row_count_device = nullptr;
            std::optional<BufferId> hidden_buffer_id;
            std::optional<BufferId> routing_indices_buffer_id;
            std::optional<BufferId> routing_weights_buffer_id;
            std::int32_t physical_rows = 0;
            std::uint32_t stage_ordinal = 0u;
            std::int32_t model_layer_index = -1;
        };

        static_assert(StageParamsRequired<Params>);

        /** @brief Construct one independently owned packet-kernel launch site. */
        explicit MoEOverlayActivationDispatchPackStage(Params params);
        ~MoEOverlayActivationDispatchPackStage() override;

        /** @return Immutable lane, layer, and graph-family capture identity. */
        [[nodiscard]] const Params &getParams() const noexcept { return params_; }

        /** @brief Enqueue deterministic row-major route compaction. */
        bool execute(IDeviceContext *ctx) override;
        /** @return MOE_OVERLAY_ACTIVATION_DISPATCH_PACK. */
        ComputeStageType type() const override;
        /** @return Stable diagnostic operation name. */
        std::string name() const override;
        /** @return Whether CUDA/ROCm supplies the packet kernel. */
        bool supportsBackend(ComputeBackendType backend) const override;
        /** @return Whether every final device address is capture-ready. */
        bool isGraphCapturable() const override;
        /** @return Whether arena prebinding can complete capture readiness. */
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        /** @return Whether immutable packet geometry admits cold prefill capture. */
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        /** @return Whether device-owned live rows make padded capture exact. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        /** @return Whether padding is excluded by the device live-row scalar. */
        bool supportsPaddedPrefillRealLengthContract() const override;
        /** @brief Bind and validate the exact future capture stream. */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        /** @return CaptureOnly; no mutable per-replay host metadata exists. */
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override;
        /** @return Typed arena reads for hidden/routing producers. */
        StageBufferContract bufferContract() const override;
        /** @return Stable tensor geometry used by workspace planning. */
        StageBufferRequirements getBufferRequirements() const override;
        /** @return Complete capture-readiness diagnostic. */
        std::string graphCaptureReadinessDebugString() const override;

    protected:
        /** @return Diagnostic description of tensor inputs and immutable geometry. */
        StageDumpInfo buildDumpInfoImpl() const override;

    private:
        /** @return Whether topology and shape are valid before device allocation. */
        bool hasStaticContract() const noexcept;
        /** @return Whether every captured device pointer is final. */
        bool hasFixedContract() const noexcept;

        Params params_;
        std::unique_ptr<IMoEKernel> moe_kernel_;
    };

    /**
     * @brief Own one layer's immutable continuation-to-follower lane transaction.
     *
     * Decode uses a fused direct-mapped kernel and therefore needs no auxiliary
     * resources. Wider prefill/verifier rows fork every lane from one exact
     * continuation stream, keep dispatch/return acquisition on context-owned
     * auxiliary streams, and join those streams only after continuation-local
     * expert work. Events are layer-owned because native graph nodes retain event
     * identity; streams are topology-bounded and shared by lane ordinal.
     *
     * This object is the only mutable ordering authority shared by the fork and
     * join stages. It stores no host mirror of packet progress and performs no
     * inference-time allocation, polling, or synchronization.
     */
    class MoEOverlayActivationLaneBatchState final
    {
    public:
        /**
         * @brief Freeze canonical lane order and graph geometry.
         *
         * @throws std::invalid_argument when lanes disagree on device, shape,
         *         stage identity, or target participant uniqueness.
         */
        MoEOverlayActivationLaneBatchState(
            DeviceId device,
            std::vector<MoEOverlayMappedActivationDeviceLane> lanes,
            std::int32_t physical_rows,
            std::uint32_t stage_ordinal,
            std::int32_t model_layer_index);

        /** @brief Destroy layer-owned events; auxiliary streams remain context-owned. */
        ~MoEOverlayActivationLaneBatchState();

        MoEOverlayActivationLaneBatchState(
            const MoEOverlayActivationLaneBatchState &) = delete;
        MoEOverlayActivationLaneBatchState &operator=(
            const MoEOverlayActivationLaneBatchState &) = delete;

        /** @return Exact planner-selected continuation device. */
        [[nodiscard]] DeviceId device() const noexcept { return device_; }
        /** @return Canonical participant order used for dispatch and FP32 fold. */
        [[nodiscard]] const std::vector<MoEOverlayMappedActivationDeviceLane> &
        lanes() const noexcept
        {
            return lanes_;
        }
        /** @return Fixed captured row capacity for this graph family. */
        [[nodiscard]] std::int32_t physicalRows() const noexcept
        {
            return physical_rows_;
        }
        /** @return Ordinal authenticated by the mapped activation epoch. */
        [[nodiscard]] std::uint32_t stageOrdinal() const noexcept
        {
            return stage_ordinal_;
        }
        /** @return Declarative model-layer identity. */
        [[nodiscard]] std::int32_t modelLayerIndex() const noexcept
        {
            return model_layer_index_;
        }
        /** @return Shared hidden width of every lane. */
        [[nodiscard]] std::int32_t dModel() const noexcept;
        /** @return Shared top-k route width of every lane. */
        [[nodiscard]] std::int32_t topK() const noexcept;
        /** @return Whether this geometry uses forked side-stream transactions. */
        [[nodiscard]] bool usesAsynchronousLanes() const noexcept;

        /**
         * @brief Create persistent events and resolve bounded auxiliary streams.
         *
         * This setup-only operation is idempotent for the frozen identity and
         * must run before native capture begins.
         */
        bool materializePersistentResources();
        /** @return Whether every required stream/event is bound to this identity. */
        [[nodiscard]] bool persistentResourcesReady() const noexcept;
        /** @brief Release layer-owned events without destroying shared streams. */
        void release();

        /** @return Main-to-lane fork event, or nullptr for fused one-row decode. */
        [[nodiscard]] void *producerReadyEvent() const noexcept
        {
            return producer_ready_event_;
        }
        /** @return Context-owned auxiliary stream for canonical lane @p index. */
        [[nodiscard]] void *laneStream(std::size_t index) const;
        /** @return Layer-owned return-import completion event for lane @p index. */
        [[nodiscard]] void *laneReturnReadyEvent(std::size_t index) const;
        /** @return Number of distinct shared physical dispatch publications. */
        [[nodiscard]] std::size_t sharedDispatchPayloadGroupCount() const noexcept
        {
            return shared_dispatch_payload_groups_.size();
        }
        /** @return Whether lane @p index consumes a shared physical payload. */
        [[nodiscard]] bool laneUsesSharedDispatchPayload(
            std::size_t index) const;
        /**
         * @return Whether lane @p index owns the one D2H publication for its
         *         mapped rank-pair group.
         */
        [[nodiscard]] bool lanePublishesSharedDispatchPayload(
            std::size_t index) const;
        /** @return Number of participant lanes sharing lane @p index's payload. */
        [[nodiscard]] std::size_t sharedDispatchPayloadLaneCount(
            std::size_t index) const;
        /**
         * @return Layer-owned copy-completion event shared by lane @p index's
         *         publication group.
         */
        [[nodiscard]] void *sharedDispatchPayloadReadyEvent(
            std::size_t index) const;

    private:
        /** @brief One physical activation publication and all CSR consumers. */
        struct SharedDispatchPayloadGroup
        {
            /** Registered mapping identity; different channels never alias. */
            const MappedHostTransferRegion *mapped_region = nullptr;
            /** Fixed destination offset within @ref mapped_region. */
            std::size_t destination_offset = 0u;
            /** First canonical lane that enqueues the physical D2H copy. */
            std::size_t publisher_lane_index = 0u;
            /** Number of lane-local timeline publications covered by the copy. */
            std::size_t lane_count = 0u;
            /** Producer-stream copy completion consumed by other lane streams. */
            void *ready_event = nullptr;
        };

        DeviceId device_ = DeviceId::invalid();
        std::vector<MoEOverlayMappedActivationDeviceLane> lanes_;
        std::int32_t physical_rows_ = 0;
        std::uint32_t stage_ordinal_ = 0u;
        std::int32_t model_layer_index_ = -1;
        IBackend *event_backend_ = nullptr;
        int event_device_ordinal_ = -1;
        void *producer_ready_event_ = nullptr;
        std::vector<void *> lane_streams_;
        std::vector<void *> lane_return_ready_events_;
        /** Per-lane group index, or -1 for a direct compact packet. */
        std::vector<std::int32_t> lane_shared_dispatch_payload_groups_;
        /** Unique groups in first-canonical-lane order. */
        std::vector<SharedDispatchPayloadGroup>
            shared_dispatch_payload_groups_;
    };

    /**
     * @brief Fork a topology-sized remote-lane transaction from continuation.
     *
     * Exact one-row direct mappings retain the single-grid decode fast path.
     * Wider rows publish the physical activation matrix once per rank-pair,
     * then compact lane metadata, publication, return acquisition, and direct
     * mapped return writes proceed independently on persistent lane streams.
     * The public graph stream does not wait here, allowing continuation-local
     * experts to overlap the complete remote transaction.
     */
    class MoEOverlayActivationDispatchPackBatchStage final
        : public IComputeStage
    {
    public:
        /** @brief Immutable shared sources and canonical target-lane order. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;
            std::shared_ptr<MoEOverlayActivationLaneBatchState> transaction;
            TensorBase *hidden = nullptr;
            TensorBase *routing_indices = nullptr;
            TensorBase *routing_weights = nullptr;
            MoEOverlayRoutePlacementDeviceBinding placement{};
            const std::int32_t *active_row_count_device = nullptr;
            std::optional<BufferId> hidden_buffer_id;
            std::optional<BufferId> routing_indices_buffer_id;
            std::optional<BufferId> routing_weights_buffer_id;
        };

        static_assert(StageParamsRequired<Params>);

        /** @brief Retain one canonical lane batch for setup-time preparation. */
        explicit MoEOverlayActivationDispatchPackBatchStage(Params params);
        ~MoEOverlayActivationDispatchPackBatchStage() override;

        /** @return Immutable topology and tensor bindings used by this batch. */
        [[nodiscard]] const Params &getParams() const noexcept { return params_; }

        /** @brief Enqueue fused decode or a non-blocking side-stream fork. */
        bool execute(IDeviceContext *ctx) override;
        /** @return MOE_OVERLAY_ACTIVATION_DISPATCH_PACK. */
        ComputeStageType type() const override;
        /** @return Stable batch-stage diagnostic name. */
        std::string name() const override;
        /** @return Whether CUDA/ROCm provides the batch kernel. */
        bool supportsBackend(ComputeBackendType backend) const override;
        /** @return Whether tensor addresses and device descriptors are fixed. */
        bool isGraphCapturable() const override;
        /** @return Whether setup can prepare immutable descriptor storage. */
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        /** @return Whether the frozen geometry is statically valid. */
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        /** @return Whether the device live-row scalar makes padding exact. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        /** @return Whether inactive padded rows are excluded on device. */
        bool supportsPaddedPrefillRealLengthContract() const override;
        /** @brief Prepare decode descriptors or persistent fork/join resources. */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        /** @return CaptureOnly when the static contract is complete. */
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override;
        /** @return Arena inputs shared by every lane. */
        StageBufferContract bufferContract() const override;
        /** @return Shared input geometry for workspace planning. */
        StageBufferRequirements getBufferRequirements() const override;
        /** @return Complete topology and descriptor readiness diagnostic. */
        std::string graphCaptureReadinessDebugString() const override;

    protected:
        /** @return Diagnostic view of shared tensors and batch identity. */
        StageDumpInfo buildDumpInfoImpl() const override;

    private:
        /** @return Whether lane topology and source tensor geometry are valid. */
        bool hasStaticContract() const noexcept;
        /** @return Whether every persistent device address is capture-ready. */
        bool hasFixedContract() const noexcept;
        /** @brief Materialize the persistent device descriptor array once. */
        bool prepareDescriptors(void *stream);

        Params params_;
        std::unique_ptr<IMoEKernel> moe_kernel_;
        std::unique_ptr<TensorBase> descriptor_storage_;
    };

    /**
     * @brief Expand one shared sparse dispatch packet into follower tensors.
     *
     * Unused rows and route slots are cleared by the backend kernel. The
     * device-local live-row scalar is published in the same stream before the
     * following real expert stage consumes it.
     */
    class MoEOverlayActivationDispatchConsumeStage final : public IComputeStage
    {
    public:
        /** @brief Immutable lane and fixed-capacity follower destinations. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;
            MoEOverlayMappedActivationDeviceLane lane;
            TensorBase *hidden = nullptr;
            TensorBase *routing_indices = nullptr;
            TensorBase *routing_weights = nullptr;
            std::int32_t *active_row_count_device = nullptr;
            BufferId hidden_buffer_id = BufferId::NORMALIZED;
            BufferId routing_indices_buffer_id = BufferId::MOE_EXPERT_INDICES;
            BufferId routing_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;
            std::int32_t physical_rows = 0;
            std::uint32_t stage_ordinal = 0u;
            std::int32_t model_layer_index = -1;
        };

        static_assert(StageParamsRequired<Params>);

        /** @brief Construct one independently owned packet-kernel launch site. */
        explicit MoEOverlayActivationDispatchConsumeStage(Params params);
        ~MoEOverlayActivationDispatchConsumeStage() override;

        /** @return Immutable lane, layer, and graph-family capture identity. */
        [[nodiscard]] const Params &getParams() const noexcept { return params_; }

        /** @brief Enqueue descriptor validation and fixed-capacity expansion. */
        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override;
        std::string name() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        /** @return Whether immutable packet geometry admits cold prefill capture. */
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        /** @return Whether descriptor-owned live rows make padded capture exact. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        /** @return true because the dispatch descriptor owns the live prefix. */
        bool supportsPaddedPrefillRealLengthContract() const override;
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override;
        StageBufferContract bufferContract() const override;
        StageBufferRequirements getBufferRequirements() const override;
        std::string graphCaptureReadinessDebugString() const override;

    protected:
        StageDumpInfo buildDumpInfoImpl() const override;

    private:
        bool hasStaticContract() const noexcept;
        bool hasFixedContract() const noexcept;

        Params params_;
        std::unique_ptr<IMoEKernel> moe_kernel_;
    };

    /** @brief Pack follower-local expert outputs into one shared return lane. */
    class MoEOverlayActivationReturnPackStage final : public IComputeStage
    {
    public:
        /** @brief Immutable lane and fixed-capacity local output binding. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;
            MoEOverlayMappedActivationDeviceLane lane;
            TensorBase *local_output = nullptr;
            BufferId local_output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
            std::int32_t physical_rows = 0;
            std::uint32_t stage_ordinal = 0u;
            std::int32_t model_layer_index = -1;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MoEOverlayActivationReturnPackStage(Params params);
        ~MoEOverlayActivationReturnPackStage() override;

        /** @return Immutable lane, layer, and graph-family capture identity. */
        [[nodiscard]] const Params &getParams() const noexcept { return params_; }

        /** @brief Enqueue compact return publication in dispatch row order. */
        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override;
        std::string name() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        /** @return Whether immutable packet geometry admits cold prefill capture. */
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        /** @return Whether descriptor-owned live rows make padded capture exact. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        /** @return true because only dispatch-live rows enter the return lane. */
        bool supportsPaddedPrefillRealLengthContract() const override;
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override;
        StageBufferContract bufferContract() const override;
        StageBufferRequirements getBufferRequirements() const override;
        std::string graphCaptureReadinessDebugString() const override;

    protected:
        StageDumpInfo buildDumpInfoImpl() const override;

    private:
        bool hasStaticContract() const noexcept;
        bool hasFixedContract() const noexcept;

        Params params_;
        std::unique_ptr<IMoEKernel> moe_kernel_;
    };

    /**
     * @brief Fold one returned participant lane into continuation output.
     *
     * Parent lowering appends these stages in planner-canonical participant
     * order. The kernel gives each output element one writer, preserving that
     * exact FP32 fold order regardless of follower completion timing.
     */
    class MoEOverlayActivationReturnConsumeStage final : public IComputeStage
    {
    public:
        /** @brief Immutable lane and continuation accumulation binding. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;
            MoEOverlayMappedActivationDeviceLane lane;
            TensorBase *dense_output = nullptr;
            BufferId dense_output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
            std::int32_t physical_rows = 0;
            std::uint32_t stage_ordinal = 0u;
            std::int32_t model_layer_index = -1;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MoEOverlayActivationReturnConsumeStage(Params params);
        ~MoEOverlayActivationReturnConsumeStage() override;

        /** @return Immutable lane, layer, and graph-family capture identity. */
        [[nodiscard]] const Params &getParams() const noexcept { return params_; }

        /** @brief Enqueue descriptor validation and deterministic FP32 fold. */
        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override;
        std::string name() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        /** @return Whether immutable packet geometry admits cold prefill capture. */
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        /** @return Whether descriptor-owned live rows make padded capture exact. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        /** @return true because only returned live-row identities are folded. */
        bool supportsPaddedPrefillRealLengthContract() const override;
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override;
        StageBufferContract bufferContract() const override;
        StageBufferRequirements getBufferRequirements() const override;
        std::string graphCaptureReadinessDebugString() const override;

    protected:
        StageDumpInfo buildDumpInfoImpl() const override;

    private:
        bool hasStaticContract() const noexcept;
        bool hasFixedContract() const noexcept;

        Params params_;
        std::unique_ptr<IMoEKernel> moe_kernel_;
    };

    /**
     * @brief Join independent lane returns and fold them in canonical order.
     *
     * Exact one-row decode retains the topology-sized gather kernel. Wider-row
     * transactions wait only for the layer-owned import events already queued
     * by the fork stage, validate/index all lanes in one topology-parallel
     * kernel, then fold them in planner order with one writer per output
     * element. Completion timing therefore cannot perturb FP32 arithmetic.
     */
    class MoEOverlayActivationReturnConsumeBatchStage final
        : public IComputeStage
    {
    public:
        /** @brief Immutable canonical lanes and continuation accumulator. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;
            std::shared_ptr<MoEOverlayActivationLaneBatchState> transaction;
            TensorBase *dense_output = nullptr;
            BufferId dense_output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
        };

        static_assert(StageParamsRequired<Params>);

        /** @brief Retain one canonical return batch for setup preparation. */
        explicit MoEOverlayActivationReturnConsumeBatchStage(Params params);
        ~MoEOverlayActivationReturnConsumeBatchStage() override;

        /** @return Immutable canonical lane order and accumulator binding. */
        [[nodiscard]] const Params &getParams() const noexcept { return params_; }

        /** @brief Enqueue parallel gather and deterministic fold. */
        bool execute(IDeviceContext *ctx) override;
        /** @return MOE_OVERLAY_ACTIVATION_RETURN_CONSUME. */
        ComputeStageType type() const override;
        /** @return Stable batch-stage diagnostic name. */
        std::string name() const override;
        /** @return Whether CUDA/ROCm provides both kernels. */
        bool supportsBackend(ComputeBackendType backend) const override;
        /** @return Whether descriptors and scratch addresses are fixed. */
        bool isGraphCapturable() const override;
        /** @return Whether setup can prepare persistent batch storage. */
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        /** @return Whether the frozen geometry is statically valid. */
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        /** @return Whether returned live-row metadata makes padding exact. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        /** @return Whether only descriptor-live rows enter the canonical fold. */
        bool supportsPaddedPrefillRealLengthContract() const override;
        /** @brief Allocate descriptors and gather scratch before capture. */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        /** @return CaptureOnly when the static contract is complete. */
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override;
        /** @return In/out continuation accumulator contract. */
        StageBufferContract bufferContract() const override;
        /** @return Accumulator geometry for workspace planning. */
        StageBufferRequirements getBufferRequirements() const override;
        /** @return Complete topology and storage readiness diagnostic. */
        std::string graphCaptureReadinessDebugString() const override;

    protected:
        /** @return Diagnostic view of accumulator and lane count. */
        StageDumpInfo buildDumpInfoImpl() const override;

    private:
        /** @return Whether lane topology and one-row output geometry are valid. */
        bool hasStaticContract() const noexcept;
        /** @return Whether every persistent device address is capture-ready. */
        bool hasFixedContract() const noexcept;
        /** @brief Materialize descriptors and topology-sized scratch once. */
        bool prepareStorage(void *stream);

        Params params_;
        std::unique_ptr<IMoEKernel> moe_kernel_;
        /** Persistent planner-ordered launch descriptors for either batch geometry. */
        std::unique_ptr<TensorBase> descriptor_storage_;
        /** One-row gathered values; unused by multi-row transactions. */
        std::unique_ptr<TensorBase> gathered_rows_;
        /** One-row live counts or multi-row authenticated-lane flags. */
        std::unique_ptr<TensorBase> lane_live_rows_;
        /** Multi-row `[lane, destination_row] -> compact_row` lookup. */
        std::unique_ptr<TensorBase> lane_row_to_compact_;
    };
} // namespace llaminar2
