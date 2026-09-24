/**
 * @file MoEOverlayNodeLocalRankBatchTransport.h
 * @brief Same-node shared activation channel for rank-batched MoE work.
 *
 * This transport is valid only when topology proves that the continuation and
 * endpoint MPI ranks share one physical node.  Fixed participant dispatch and
 * return arrays live in one POSIX shared-memory mapping and are accessed through
 * exact device-visible aliases. Device-owned timeline words order every compact
 * packet without a CPU copy, MPI payload, host wait, or default stream.
 *
 * Inter-node execution is intentionally outside this class.  The typed factory
 * below selects the MPI implementation for a rank pair on different hosts and
 * never tries to emulate shared pages across a network.
 */

#pragma once

#include "MoEOverlayActivationChannelPlan.h"
#include "MoEOverlayActivationEpochProtocol.h"
#include "MoEOverlayActivationPacketABI.h"
#include "MoEOverlayRankBatchTransport.h"
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{
    class TensorBase;

    /**
     * @brief Complete geometry-selected dispatch payload embedded by a lane.
     *
     * The value binds the only three facts that may vary with retained graph
     * geometry: hidden layout, exact hidden pointer, and mapped publication
     * offset. Packet metadata is compact in both layouts. Keeping these facts
     * indivisible prevents a consumer from combining a current descriptor with
     * the other layout's stable-but-stale matrix.
     */
    struct MoEOverlayMappedActivationDispatchPayloadView
    {
        /** Geometry and hidden-row interpretation selected as one value. */
        MoEOverlayActivationPayloadSelection selection{};
        /** Exact metadata and selected hidden-matrix aliases. */
        MoEOverlayMappedDispatchDeviceView packet{};
        /** Region-relative destination used by bulk payload publication. */
        std::size_t hidden_payload_offset = 0u;

        /** @return Whether every address and geometry relation is complete. */
        [[nodiscard]] bool valid() const noexcept
        {
            return selection.valid() && packet.valid() &&
                   packet.row_capacity >=
                       static_cast<std::size_t>(selection.physical_rows);
        }

        /** @return Whether this view requires one rank-pair bulk publication. */
        [[nodiscard]] bool requiresBulkPublication() const noexcept
        {
            return valid() && selection.usesSharedPhysicalRows();
        }
    };

    /**
     * @brief Complete immutable device binding for one participant/graph lane.
     *
     * The topology planner chooses @ref device; this structure assigns no
     * meaning to CUDA, ROCm, or CPU.  GPU graph stages use the device aliases
     * and signal offsets, while a CPU endpoint sees the same host addresses.
     * Keeping the mapped-region lifetime in the binding makes it impossible for
     * a retained graph to outlive driver registration of any embedded address.
     */
    struct MoEOverlayMappedActivationDeviceLane
    {
        DeviceId device = DeviceId::invalid(); ///< Exact local endpoint selected by planning.
        int target_participant_id = -1; ///< Logical participant owning packet compute.
        std::size_t graph_family_ordinal = 0u; ///< Main or one learned MTP graph family.
        std::shared_ptr<const MappedHostTransferRegion> mapped_region; ///< Registration lifetime.
        /**
         * Activation-only physical-row matrix shared by every participant in
         * this rank-pair mapping. Direct compact packets do not access it.
         */
        float *shared_dispatch_hidden_rows_fp32 = nullptr;
        MoEOverlayActivationEpochControl *control_host = nullptr; ///< Scheduler/watchdog view.
        MoEOverlayActivationEpochControl *control_device = nullptr; ///< Exact endpoint alias.
        MoEOverlayActivationDeviceEpochGrant *grant_device = nullptr; ///< Endpoint-private hot-path state.
        /** Setup publication event consumed once at each stage-zero graph edge. */
        void *grant_initialization_event = nullptr;
        MoEOverlayMappedDispatchDeviceView dispatch; ///< Shared continuation-to-follower packet.
        MoEOverlayMappedReturnDeviceView returned; ///< Shared follower-to-continuation packet.
        std::size_t dispatch_hidden_offset = 0u; ///< Region offset of the mapped dispatch matrix.
        /** Region offset of the shared physical-row dispatch matrix. */
        std::size_t shared_dispatch_hidden_offset = 0u;
        /** Region offset of the rank-pair canonical route-return matrix. */
        std::size_t return_output_offset = 0u;
        std::size_t admission_signal_offset = 0u; ///< Scheduler release word for graph admission.
        std::array<std::size_t, kMoEOverlayActivationBufferCount>
            dispatch_signal_offsets{}; ///< Region-relative 64-bit publication words.
        std::array<std::size_t, kMoEOverlayActivationBufferCount>
            return_signal_offsets{}; ///< Region-relative 64-bit return words.

        /** @return Whether graph capture can safely embed every address. */
        [[nodiscard]] bool valid() const noexcept
        {
            const auto containsFP32Matrix =
                [this](std::size_t offset,
                       std::size_t rows,
                       std::int32_t columns)
            {
                if (!mapped_region || columns <= 0)
                    return false;
                const auto width = static_cast<std::size_t>(columns);
                constexpr auto maximum = static_cast<std::size_t>(-1);
                if (rows > maximum / width ||
                    rows * width > maximum / sizeof(float))
                {
                    return false;
                }
                return mapped_region->contains(
                    offset, rows * width * sizeof(float));
            };
            return device.is_valid() && target_participant_id >= 0 &&
                   mapped_region && mapped_region->isBound() && control_host &&
                   control_device && grant_device &&
                   (!device.is_gpu() || grant_initialization_event) &&
                   shared_dispatch_hidden_rows_fp32 && dispatch.valid() &&
                   returned.valid() &&
                   containsFP32Matrix(
                       dispatch_hidden_offset,
                       dispatch.row_capacity,
                       dispatch.d_model) &&
                   containsFP32Matrix(
                       shared_dispatch_hidden_offset,
                       dispatch.row_capacity,
                       dispatch.d_model) &&
                   containsFP32Matrix(
                       return_output_offset,
                       returned.route_slot_capacity,
                       returned.d_model) &&
                   mapped_region->contains(
                       admission_signal_offset, sizeof(std::uint64_t));
        }

        /**
         * @brief Resolve one indivisible dispatch view for fixed geometry.
         *
         * Timeline words and packet metadata remain mapped for every geometry.
         * One-row decode writes the compact participant matrix. Multi-row
         * prefill publishes the source activation once per rank-pair and binds
         * the returned view to that shared physical matrix. No caller is
         * permitted to override the pointer or reinterpret the layout later.
         *
         * @param physical_rows Exact padded rows embedded by the graph.
         * @return Capture-stable payload view, invalid for bad geometry.
         */
        [[nodiscard]] MoEOverlayMappedActivationDispatchPayloadView
        dispatchPayload(std::int32_t physical_rows) const noexcept
        {
            MoEOverlayMappedActivationDispatchPayloadView result{
                .selection = MoEOverlayActivationPayloadSelection::
                    forPhysicalRows(physical_rows),
                .packet = dispatch,
                .hidden_payload_offset = dispatch_hidden_offset,
            };
            if (result.selection.usesSharedPhysicalRows())
            {
                result.packet.hidden_rows_fp32 =
                    shared_dispatch_hidden_rows_fp32;
                result.hidden_payload_offset =
                    shared_dispatch_hidden_offset;
            }
            return result;
        }

        /**
         * @brief Build the CPU follower's sparse view from the selected lane.
         *
         * This is valid only for a planner-declared CPU endpoint because the
         * packet aliases must be directly host-addressable. The returned object
         * owns no storage; this lane and its mapped-region lifetime must outlive
         * every local-expert use.
         *
         * @param physical_rows Exact rows authenticated by the request ticket.
         * @return Sparse CPU view carrying the same pointer/layout selection as
         *         the device launch, or an invalid empty view on mismatch.
         */
        [[nodiscard]] MoEOverlaySparseRows hostDispatchPayload(
            std::int32_t physical_rows) const noexcept
        {
            const auto payload = dispatchPayload(physical_rows);
            if (!device.is_cpu() || !payload.valid())
                return {};
            MoEOverlaySparseRows rows;
            rows.target_participant = target_participant_id;
            rows.d_model = payload.packet.d_model;
            rows.top_k = payload.packet.top_k;
            rows.row_capacity = payload.packet.row_capacity;
            rows.entry_capacity = payload.packet.entry_capacity;
            rows.hidden_row_capacity =
                static_cast<std::size_t>(payload.selection.physical_rows);
            rows.hidden_payload_layout = payload.selection.layout;
            rows.row_ids_host = payload.packet.row_ids;
            rows.entry_offsets_host = payload.packet.entry_offsets;
            rows.expert_ids_host = payload.packet.expert_ids;
            rows.route_weights_host = payload.packet.route_weights;
            rows.original_route_slots_host =
                payload.packet.original_route_slots;
            rows.compact_route_slots_host =
                payload.packet.compact_route_slots;
            rows.hidden_rows_fp32 = payload.packet.hidden_rows_fp32;
            return rows;
        }
    };

    /** Element layout of one graph-private activation-protocol allocation. */
    enum class MoEOverlayGraphStorageType : std::uint8_t
    {
        Int32 = 0, ///< Four-byte descriptor words or integer scratch.
        FP32 = 1, ///< Canonical FP32 accumulation scratch.
    };

    /**
     * @brief Own one capture-stable device tensor outside compute-stage lifecycle.
     *
     * Packet stages describe immutable launch metadata and scratch geometry,
     * but they must never allocate or upload GPU storage themselves. This
     * setup owner allocates through TransferEngine when the graph transaction
     * is constructed, retains the host bytes needed by asynchronous metadata
     * publication, and exposes only explicit-stream preparation/validation to
     * the stages. Device addresses remain fixed until every retained graph
     * sharing this owner has been destroyed.
     */
    class MoEOverlayPersistentGraphStorage final
    {
    public:
        /** Complete immutable allocation identity. */
        struct Config
        {
            DeviceId device = DeviceId::invalid(); ///< Exact allocation owner.
            MoEOverlayGraphStorageType type =
                MoEOverlayGraphStorageType::Int32; ///< Element representation.
            std::vector<std::size_t> shape; ///< Positive tensor dimensions.
            bool immutable_input = false; ///< True for host-authored metadata.
            std::string identity; ///< Stable diagnostic role.
        };

        /**
         * @brief Allocate the complete capture-stable storage during setup.
         * @throws std::invalid_argument for an incomplete identity or shape.
         * @throws std::runtime_error when the canonical transfer authority
         *         cannot materialize the requested GPU allocation.
         */
        explicit MoEOverlayPersistentGraphStorage(Config config);
        ~MoEOverlayPersistentGraphStorage();

        MoEOverlayPersistentGraphStorage(
            const MoEOverlayPersistentGraphStorage &) = delete;
        MoEOverlayPersistentGraphStorage &operator=(
            const MoEOverlayPersistentGraphStorage &) = delete;

        /**
         * @brief Publish immutable host bytes on the exact setup stream.
         *
         * The first call copies and uploads the complete allocation. Later
         * calls must name byte-identical metadata and merely enqueue the
         * existing producer-event dependency on @p stream.
         */
        bool publishImmutableBytes(
            const void *bytes,
            std::size_t byte_count,
            void *stream);

        /** @brief Validate/enqueue this immutable input on an exact stream. */
        bool requireInput(void *stream) const;
        /** @brief Validate this graph-private output on an exact stream. */
        bool requireOutput(void *stream) const;

        /** @return Stable device address, or nullptr before complete setup. */
        [[nodiscard]] void *deviceData() const noexcept;
        /** @return Immutable allocation capacity in bytes. */
        [[nodiscard]] std::size_t sizeBytes() const noexcept;
        /** @return Whether the allocation has a stable device address. */
        [[nodiscard]] bool allocated() const noexcept;
        /** @return Whether immutable input bytes have been published. */
        [[nodiscard]] bool published() const noexcept { return published_; }

    private:
        Config config_;
        std::unique_ptr<TensorBase> tensor_;
        bool published_ = false;
    };

    /**
     * @brief One immutable retained-graph manifest sharing a rank-batch mapping.
     *
     * Main prefill, decode, and grouped verification share one ordered full-model
     * manifest. Each MTP sidecar owns a separate one-layer manifest. Keeping the
     * allowed role mask beside the layer list prevents a valid ticket for one
     * retained family from arming another family's device control.
     */
    struct MoEOverlayActivationGraphFamilyManifest
    {
        /** Bit N admits @ref MoEOverlayInferenceGraphRole value N. */
        std::uint32_t graph_role_mask = 0u;
        /** Exact strictly increasing model-layer order embedded by the graph. */
        std::vector<std::int32_t> model_layer_indices;

        /** @return Whether roles and ordered stage geometry are non-empty. */
        [[nodiscard]] bool valid() const noexcept
        {
            return graph_role_mask != 0u &&
                   MoEOverlayActivationEpochProtocol::stageManifestDigest(
                       model_layer_indices) != 0u;
        }
    };

    /**
     * @brief Resolve canonical main and MTP activation-channel graph families.
     *
     * The main family admits prefill, decode, and grouped verification over the
     * complete main layer sequence. Every learned MTP source layer becomes one
     * independently authenticated draft family in depth order.
     *
     * @param graph_family Exact transaction graph geometry shared by both ranks.
     * @return Main manifest followed by one manifest per MTP depth.
     * @throws std::invalid_argument when @p graph_family is incomplete.
     */
    [[nodiscard]] std::vector<MoEOverlayActivationGraphFamilyManifest>
    makeMoEOverlayActivationGraphFamilyManifests(
        const MoEOverlayInferenceGraphFamilyIdentity &graph_family);

    /**
     * @brief Complete immutable contract used by topology-aware transport selection.
     */
    struct MoEOverlayRankBatchTransportConfig
    {
        std::shared_ptr<IMPIContext> mpi_ctx; ///< World/topology authority.
        int source_world_rank = -1; ///< Continuation rank.
        int target_world_rank = -1; ///< Endpoint-owner rank.
        std::shared_ptr<MoEOverlayRankBatchWireWorkspace> workspace;
        size_t max_rows_per_participant = 0; ///< Fixed row storage per endpoint.
        size_t max_entries_per_participant = 0; ///< Fixed CSR route storage per endpoint.
        int d_model = 0; ///< Hidden/output width.
        int top_k = 0; ///< Maximum routes per token row.
        int tier_index = -1; ///< Stable tier index; never interpreted as preference.
        int domain_ordinal = -1; ///< Stable routed-domain ordinal.
        /** Stable topology identity, normally the canonical participant list. */
        std::string channel_identity;
        size_t transaction_slot_count = 4096; ///< Fixed replay ledger slots.
        size_t asynchronous_send_slot_count = 4; ///< MPI-only stable send ring size.
        /** Exact rank-pair graph identity also authenticated by scheduler tickets. */
        MoEOverlayInferenceTopologyIdentity transaction_topology;
        /** Planner-selected continuation endpoint; backend type is deliberately absent. */
        MoEOverlayActivationLaneEndpoint source_endpoint;
        /** Integer preference of the target tier; smaller values are preferred. */
        int target_tier_priority = 0;
        /** Complete retained main/MTP graph-family manifests in canonical order. */
        std::vector<MoEOverlayActivationGraphFamilyManifest>
            activation_graph_families;
        /** Planner-certified mapping offsets and first-touch ownership. */
        MoEOverlayNodeLocalActivationLayout activation_layout;
        /**
         * Exact process-local participant/device lanes embedded by graphs.
         *
         * Capacity admission and preflight obtain this list from
         * @ref MoEOverlayActivationChannelPlanner. Keeping participant identity
         * beside device identity prevents an unpriced Cartesian product from
         * being materialized when one target rank owns several accelerators.
         * The list is never mixed into POSIX mapping identity because the two
         * endpoint ranks deliberately own different local lanes.
         */
        std::vector<MoEOverlayActivationLocalLaneBinding> local_lanes;
    };

    /**
     * @brief Graph-facing capability for mapped node-local activation epochs.
     *
     * This interface describes a physical capability, not a concrete transport
     * class or backend.  A production graph builder may bind retained packet
     * stages only when the topology-selected rank-batch transport implements
     * this contract.  In particular, the portable MPI transport deliberately
     * does not expose mapped device aliases and therefore cannot accidentally
     * enter the node-local device-owned epoch path.
     *
     * The returned objects are immutable setup identity except for the epoch
     * control itself, whose mutation is owned by the authenticated activation
     * protocol.  Implementations must retain every mapping and driver
     * registration for at least as long as any lane returned from this object.
     */
    class IMoEOverlayMappedActivationTransport
    {
    public:
        /** @brief Polymorphic destruction through the capability boundary. */
        virtual ~IMoEOverlayMappedActivationTransport() = default;

        /** @return Number of independently authenticated retained graph families. */
        [[nodiscard]] virtual size_t activationGraphFamilyCount() const noexcept = 0;

        /**
         * @brief Resolve a model layer to its exact ordinal in one retained family.
         *
         * Callers must not infer this ordinal from a global layer number: pipeline
         * partitions and MTP families can carry arbitrary ordered subsets.
         *
         * @throws std::out_of_range for an unknown family or absent model layer.
         */
        [[nodiscard]] virtual std::uint32_t activationStageOrdinal(
            size_t graph_family_ordinal,
            std::int32_t model_layer_index) const = 0;

        /**
         * @brief Return the exact shared epoch control for one target/family lane.
         * @throws std::out_of_range for an unknown participant or graph family.
         */
        [[nodiscard]] virtual MoEOverlayActivationEpochControl &
        activationEpochControl(
            int target_participant_id,
            size_t graph_family_ordinal) const = 0;

        /**
         * @brief Return the immutable scheduler contract for one mapped lane.
         *
         * The scheduler must validate the shared control against the same
         * topology recipe that initialized it. Exposing that recipe through
         * the capability boundary avoids reconstructing channel nonces, lane
         * ordinals, or role masks from raw ABI bytes in a model runner.
         *
         * @throws std::out_of_range for an unknown participant or family.
         */
        [[nodiscard]] virtual MoEOverlayActivationEpochConfig
        activationEpochConfig(
            int target_participant_id,
            size_t graph_family_ordinal) const = 0;

        /**
         * @return Mapping authority whose lifetime makes every embedded alias valid.
         */
        [[nodiscard]] virtual std::shared_ptr<const MappedHostTransferRegion>
        mappedTransferRegion() const noexcept = 0;

        /**
         * @brief Resolve every captured address for one planner-declared endpoint.
         *
         * This is a setup-only operation.  It never arms an epoch, waits for a
         * signal, or reads mutable packet payloads.
         *
         * @param target_participant_id Logical endpoint represented by the lane.
         * @param graph_family_ordinal Main/MTP retained graph-family ordinal.
         * @param device Exact process-local CPU/CUDA/ROCm endpoint.
         * @return Complete stable device binding for packet graph construction.
         * @throws std::invalid_argument or std::out_of_range on topology mismatch.
         */
        [[nodiscard]] virtual MoEOverlayMappedActivationDeviceLane
        activationDeviceLane(
            int target_participant_id,
            size_t graph_family_ordinal,
            DeviceId device) const = 0;
    };

    /**
     * @brief Same-host shared row transport with authenticated publication epochs.
     */
    class MoEOverlayNodeLocalRankBatchTransport final
        : public IMoEOverlayRankBatchTransport,
          public IMoEOverlayMappedActivationTransport
    {
    public:
        /**
         * @brief Map or create the exact shared channel and validate topology.
         * @throws std::invalid_argument for invalid geometry or rank identity.
         * @throws std::runtime_error when ranks are not node-local or mapping fails.
         */
        explicit MoEOverlayNodeLocalRankBatchTransport(
            MoEOverlayRankBatchTransportConfig config);

        /** @brief Unmap local pages; source also unlinks the run-scoped name. */
        ~MoEOverlayNodeLocalRankBatchTransport() override;

        MoEOverlayNodeLocalRankBatchTransport(
            const MoEOverlayNodeLocalRankBatchTransport &) = delete;
        MoEOverlayNodeLocalRankBatchTransport &operator=(
            const MoEOverlayNodeLocalRankBatchTransport &) = delete;

        /** @return NodeLocalSharedRows. */
        MoEOverlayRankBatchTransportKind kind() const noexcept override
        {
            return MoEOverlayRankBatchTransportKind::NodeLocalSharedRows;
        }

        /** @brief Publish/consume shared dispatch metadata around in-place arrays. */
        MoEOverlayCollectiveResult exchangeDispatch(
            const MoEOverlayRankBatchKey &key,
            std::span<const MoEOverlaySparseRows *const> outbound,
            std::span<MoEOverlaySparseRows *const> inbound) override;

        /** @brief Publish/consume shared return metadata around in-place arrays. */
        MoEOverlayCollectiveResult exchangeReturn(
            const MoEOverlayRankBatchKey &key,
            std::span<const MoEOverlayReturnRows *const> outbound,
            std::span<MoEOverlayReturnRows *const> inbound) override;

        int sourceWorldRank() const noexcept override;
        int targetWorldRank() const noexcept override;
        int localWorldRank() const noexcept override;
        const std::vector<int> &participantIds() const noexcept override;
        bool hasSharedRowStorage() const noexcept override { return true; }
        MoEOverlaySparseRows sharedDispatchRows(
            int participant_id) const override;
        MoEOverlayReturnRows sharedReturnRows(
            int participant_id) const override;

        /** @return POSIX mapping bytes, including metadata and both payload directions. */
        size_t mappedBytes() const noexcept;

        /** @return Number of independently authenticated retained graph families. */
        [[nodiscard]] size_t activationGraphFamilyCount() const noexcept override;

        /** @copydoc IMoEOverlayMappedActivationTransport::activationStageOrdinal */
        [[nodiscard]] std::uint32_t activationStageOrdinal(
            size_t graph_family_ordinal,
            std::int32_t model_layer_index) const override;

        /**
         * @brief Return the exact shared control for one target/family lane.
         * @throws std::out_of_range for an unknown participant or family.
         */
        [[nodiscard]] MoEOverlayActivationEpochControl &activationEpochControl(
            int target_participant_id,
            size_t graph_family_ordinal) const override;

        /** @copydoc IMoEOverlayMappedActivationTransport::activationEpochConfig */
        [[nodiscard]] MoEOverlayActivationEpochConfig activationEpochConfig(
            int target_participant_id,
            size_t graph_family_ordinal) const override;

        /** @return Canonical mapped-region authority retained by this transport. */
        [[nodiscard]] std::shared_ptr<const MappedHostTransferRegion>
        mappedTransferRegion() const noexcept override;

        /**
         * @brief Resolve a shared host subrange to one declared local device alias.
         * @throws std::out_of_range when the host subrange is outside the mapping.
         * @throws std::invalid_argument when @p device was not planner-declared.
         */
        [[nodiscard]] void *mappedDeviceAlias(
            DeviceId device,
            const void *shared_host_address,
            size_t bytes) const;

        /**
         * @brief Resolve every captured address for one local device endpoint.
         *
         * The returned lane is setup-only immutable graph identity. It does not
         * arm a transaction, wait for a signal, or inspect mutable payload data.
         * The device must have appeared in this process's planner-provided
         * `local_lanes` set when the mapping was registered.
         *
         * @param target_participant_id Exact participant packet within the rank batch.
         * @param graph_family_ordinal Main/MTP retained graph family ordinal.
         * @param device Exact local CPU/CUDA/ROCm endpoint consuming the binding.
         * @return Complete mapped lane with device aliases and timeline offsets.
         * @throws std::invalid_argument or std::out_of_range for topology mismatch.
         */
        [[nodiscard]] MoEOverlayMappedActivationDeviceLane activationDeviceLane(
            int target_participant_id,
            size_t graph_family_ordinal,
            DeviceId device) const override;

        /**
         * @brief Resolve a shared host subrange to its immutable mapping offset.
         * @throws std::out_of_range when the range is not wholly inside the mapping.
         */
        [[nodiscard]] size_t mappedOffset(
            const void *shared_host_address,
            size_t bytes) const;

    private:
        class Mapping;
        class DeviceGrantStorage;

        /** @brief Claim a fixed replay-ledger slot or reject stale reuse. */
        bool claimTransaction(
            const MoEOverlayRankBatchKey &key,
            std::vector<std::optional<MoEOverlayRankBatchKey>> &ledger,
            std::string *error);

        MoEOverlayRankBatchTransportConfig config_;
        TransferEngine transfer_engine_;
        std::shared_ptr<Mapping> mapping_;
        /** Destroyed before @ref mapping_ so drivers unregister live pages first. */
        std::shared_ptr<MappedHostTransferRegion> mapped_region_;
        /**
         * Endpoint-private grants allocated before capture and retained for the
         * complete transport lifetime.  They are deliberately separate from
         * the mapped channel: each local device advances its own grant after a
         * single authenticated admission acquire.
         */
        std::vector<std::unique_ptr<DeviceGrantStorage>> device_grants_;
        std::vector<std::optional<MoEOverlayRankBatchKey>> dispatch_ledger_;
        std::vector<std::optional<MoEOverlayRankBatchKey>> return_ledger_;
        std::optional<MoEOverlayRankBatchKey> pending_return_dispatch_key_;
        std::atomic_flag in_flight_ = ATOMIC_FLAG_INIT;
    };

    /**
     * @brief Resolve the physical transport from authoritative rank locality.
     * @throws std::runtime_error when real topology is unavailable.
     */
    MoEOverlayRankBatchTransportKind resolveMoEOverlayRankBatchTransportKind(
        const IMPIContext &mpi_ctx,
        int source_world_rank,
        int target_world_rank);

    /**
     * @brief Construct the first-class node-local or inter-node implementation.
     */
    std::shared_ptr<IMoEOverlayRankBatchTransport>
    createMoEOverlayRankBatchTransport(
        MoEOverlayRankBatchTransportConfig config);

} // namespace llaminar2
