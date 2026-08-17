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

#include "MoEOverlayActivationEpochProtocol.h"
#include "MoEOverlayActivationPacketABI.h"
#include "MoEOverlayRankBatchTransport.h"
#include "transfer/TransferEngine.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Captured node-local payload movement selected from fixed geometry.
     *
     * Timeline words and packet metadata remain mapped in both modes. A
     * one-row transaction is compacted into each lane. Multi-row prefill
     * publishes one physical activation matrix per rank pair and each follower
     * reads only the rows selected by its compact metadata. Both variants avoid
     * per-lane bounce matrices and are immutable capture identity.
     */
    enum class MoEOverlayActivationPayloadPath : std::uint8_t
    {
        DirectMapped = 0, ///< Packet owns compact mapped hidden rows.
        SharedPhysicalMapped = 1, ///< Packet selects rows from one shared physical matrix.
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
        MoEOverlayMappedDispatchDeviceView dispatch; ///< Shared continuation-to-follower packet.
        MoEOverlayMappedReturnDeviceView returned; ///< Shared follower-to-continuation packet.
        std::size_t dispatch_hidden_offset = 0u; ///< Region offset of the mapped dispatch matrix.
        /** Region offset of the shared physical-row dispatch matrix. */
        std::size_t shared_dispatch_hidden_offset = 0u;
        std::size_t return_output_offset = 0u; ///< Region offset of the mapped return matrix.
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
            const auto matrixBytes = [](
                                         std::size_t rows,
                                         std::int32_t columns,
                                         std::size_t *bytes)
            {
                if (!bytes || columns <= 0)
                    return false;
                const auto width = static_cast<std::size_t>(columns);
                constexpr auto maximum = static_cast<std::size_t>(-1);
                if (rows > maximum / width ||
                    rows * width > maximum / sizeof(float))
                {
                    return false;
                }
                *bytes = rows * width * sizeof(float);
                return true;
            };
            std::size_t dispatch_matrix_bytes = 0u;
            std::size_t return_matrix_bytes = 0u;
            const bool matrix_geometry_valid =
                matrixBytes(
                    dispatch.row_capacity,
                    dispatch.d_model,
                    &dispatch_matrix_bytes) &&
                matrixBytes(
                    returned.row_capacity,
                    returned.d_model,
                    &return_matrix_bytes);
            return device.is_valid() && target_participant_id >= 0 &&
                   mapped_region && mapped_region->isBound() && control_host &&
                   control_device && grant_device &&
                   shared_dispatch_hidden_rows_fp32 && dispatch.valid() &&
                   returned.valid() && matrix_geometry_valid &&
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
                       returned.row_capacity,
                       returned.d_model) &&
                   mapped_region->contains(
                       admission_signal_offset, sizeof(std::uint64_t));
        }

        /**
         * @brief Select the immutable payload path for one captured row geometry.
         *
         * @param physical_rows Exact padded rows embedded by this graph.
         * @return Compact direct mapping for one-row decode, otherwise the one
         *         shared physical activation mapping for prefill.
         */
        [[nodiscard]] MoEOverlayActivationPayloadPath payloadPath(
            std::int32_t physical_rows) const noexcept
        {
            return physical_rows == 1
                       ? MoEOverlayActivationPayloadPath::DirectMapped
                       : MoEOverlayActivationPayloadPath::SharedPhysicalMapped;
        }

        /**
         * @brief Resolve the hidden-matrix interpretation for fixed geometry.
         *
         * Bulk DMA publishes the original physical activation once per shared
         * rank-pair channel. Direct mapped packets retain compact rows because
         * that avoids a full-matrix transfer for decode-sized transactions.
         *
         * @param physical_rows Exact padded rows embedded by the graph.
         * @return Capture-stable hidden payload layout for this lane.
         */
        [[nodiscard]] MoEOverlayActivationHiddenPayloadLayout
        hiddenPayloadLayout(std::int32_t physical_rows) const noexcept
        {
            return payloadPath(physical_rows) ==
                           MoEOverlayActivationPayloadPath::SharedPhysicalMapped
                       ? MoEOverlayActivationHiddenPayloadLayout::
                             SharedPhysicalRows
                       : MoEOverlayActivationHiddenPayloadLayout::CompactRows;
        }
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
        /**
         * Process-local devices that register this mapping.
         *
         * This list is never mixed into POSIX mapping identity because the two
         * ranks deliberately see different local endpoint sets.
         */
        std::vector<DeviceId> local_devices;
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
         * `local_devices` set when the mapping was registered.
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
