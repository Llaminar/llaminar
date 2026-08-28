/**
 * @file MoEOverlayPhysicalResidencyFabric.h
 * @brief Pre-materialized local physical fabric for ExpertOverlay migration.
 *
 * The residency authority decides which complete experts move; this class owns
 * the process-local physical consequences of that decision.  It allocates a
 * bounded inactive-slot budget for every local participant/layer, creates the
 * exact CPU/GPU transfer lanes before serving starts, and converts one global
 * transaction into projection operations that can be event-polled by the
 * maintenance worker.  It never reloads a tensor, creates a host mirror of a
 * GPU source, or waits on an inference stream.
 */

#pragma once

#include "ExpertWeightFormat.h"
#include "MoEOverlayParticipantMigration.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    struct GGUFModel;
    class MoEOverlayMPIRemoteProjectionTransport;
    class MappedTransferProgressEpoch;
    struct MoEOverlayDevicePhysicalMovementBatch;
    struct MoEOverlayResidencyExecutionFingerprint;

    /**
     * @brief Device-independent shape and source provenance for one projection.
     *
     * This is the model-wide contract used to materialize an inactive slot on
     * a participant that initially owns no experts.  It deliberately excludes
     * the current CPU/GPU execution layout: the destination tier derives that
     * layout from the original GGUF source identity.
     */
    struct MoEOverlayProjectionWeightManifest
    {
        ExpertTierWeightProjection projection =
            ExpertTierWeightProjection::Gate;
        int N = 0; ///< Logical output columns.
        int K = 0; ///< Logical reduction dimension, divisible by 32.
        ExpertWeightFormat format; ///< Exact quantized or floating provenance.

        /** @return Whether role, geometry, and provenance are catalogued. */
        [[nodiscard]] bool valid() const noexcept;

        /** @brief Compare the complete device-independent projection contract. */
        bool operator==(
            const MoEOverlayProjectionWeightManifest &) const = default;
    };

    /** @brief Complete gate/up/down materialization contract for one model layer. */
    struct MoEOverlayLayerWeightManifest
    {
        int layer_idx = -1;
        std::array<MoEOverlayProjectionWeightManifest, 3> projections;

        /** @return Whether every role appears exactly once with valid metadata. */
        [[nodiscard]] bool valid() const noexcept;

        /** @brief Compare layer identity and all three projection contracts. */
        bool operator==(const MoEOverlayLayerWeightManifest &) const = default;
    };

    /**
     * @brief Derive the complete routed-expert slot contract from GGUF metadata.
     *
     * Every rank parses the same tensor directory even when sparse payload
     * loading leaves a receiving tier with no resident expert in one layer.
     * This builder uses only tensor names, shapes, and quantization types; it
     * never reads weight payload bytes or constructs a host execution mirror.
     * The physical fabric subsequently authenticates every local prepared
     * engine against this model-owned contract.
     *
     * @param model Parsed model tensor directory.
     * @param num_layers Exact main-transformer layer count.
     * @param num_experts Exact routed-expert count in every layer.
     * @return One valid gate/up/down manifest per layer in layer order.
     * @throws std::invalid_argument For absent/non-3D/unsupported expert weights,
     *         inconsistent expert counts, or device-kernel-incompatible shapes.
     */
    [[nodiscard]] std::vector<MoEOverlayLayerWeightManifest>
    buildMoEOverlayLayerWeightManifestFromGGUF(
        const GGUFModel &model,
        int num_layers,
        int num_experts);

    /** @brief Process-local evidence for the materialized residency fabric. */
    struct MoEOverlayPhysicalResidencyFabricStats
    {
        std::uint64_t endpoint_layer_pools = 0;
        /** Loader-owned live slots enrolled for bounded post-retire reuse. */
        std::uint64_t adopted_initial_slots = 0;
        /** Bootstrap assignments actually returned after an old ticket barrier. */
        std::uint64_t adopted_initial_slots_recycled = 0;
        std::uint64_t cpu_shadow_slots = 0;
        std::uint64_t gpu_shadow_slots = 0;
        std::uint64_t persistent_transfer_lanes = 0;
        /** Same-device or driver-authorized direct GPU peer lanes. */
        std::uint64_t direct_gpu_peer_lanes = 0;
        /** Same-backend GPU lanes explicitly relayed because P2P is unavailable. */
        std::uint64_t same_backend_no_peer_relay_lanes = 0;
        /** CUDA/ROCm lanes using the portable host relay. */
        std::uint64_t cross_backend_gpu_relay_lanes = 0;
        /** Largest number of lanes installed for one directed edge/projection. */
        std::uint64_t maximum_parallel_edge_lanes = 0;
        /** Local GPU projection operations that reserved distinct pool lanes. */
        std::uint64_t parallel_lane_reservations = 0;
        /** Fatal attempts to exceed or overlap a pre-materialized lane pool. */
        std::uint64_t parallel_lane_pool_exhaustions = 0;
        /** Number of admitted operations that encountered an occupied lane. */
        std::uint64_t serialized_lane_deferrals = 0;
        /** CPU projection workers installed for the widest address edge. */
        std::uint64_t maximum_parallel_cpu_copy_lanes = 0;
        /** CPU projection operations that reserved distinct background workers. */
        std::uint64_t parallel_cpu_copy_lane_reservations = 0;
        /** Fatal CPU worker-pool reservations beyond the admitted wave BOM. */
        std::uint64_t parallel_cpu_copy_lane_pool_exhaustions = 0;
        /** Largest simultaneous CPU worker set released by one wave barrier. */
        std::uint64_t maximum_concurrent_cpu_copy_operations = 0;
        /** CPU workers still armed or copying when the snapshot was read. */
        std::uint64_t active_cpu_copy_operations = 0;
        std::uint64_t waves_prepared = 0;
        std::uint64_t waves_deferred = 0;
        std::uint64_t waves_failed = 0;
        std::uint64_t projection_operations_prepared = 0;
        std::uint64_t cpu_copy_operations = 0;
        std::uint64_t remote_cpu_operations = 0;
        std::uint64_t remote_gpu_cpu_operations = 0;
        std::uint64_t remote_gpu_blob_operations = 0;
        std::uint64_t gpu_cpu_operations = 0;
        std::uint64_t same_backend_gpu_operations = 0;
        std::uint64_t heterogeneous_gpu_operations = 0;
        std::uint64_t inference_stream_waits = 0;
        std::uint64_t blocking_synchronizations = 0;
    };

    /**
     * @brief Canonical process-local weight banks transferred back to a model context.
     *
     * Dynamic movement may return the logical owner map to its prepared placement
     * while a resident still occupies an over-provisioned migration shadow slot.
     * This value is produced only after every old inference reader and maintenance
     * operation has drained.  Every retained triplet then aliases one of the exact
     * loader-era allocations enrolled by the physical fabric, so destroying the
     * fabric cannot leave a shadow arena pinned by the reusable model registry.
     */
    struct MoEOverlayReusableContextSeal
    {
        /** Published logical epoch whose owner map was physically canonicalized. */
        std::uint64_t source_epoch = 0;
        /** Complete immutable banks for every process-local participant. */
        std::vector<MoEOverlayParticipantResidencyBank> local_banks;
        /** Residents already occupying an adopted loader-era allocation. */
        std::size_t retained_canonical_experts = 0;
        /** Residents copied asynchronously out of a temporary shadow allocation. */
        std::size_t compacted_shadow_experts = 0;

        /** @return Whether this value names a positive epoch and complete banks. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /**
     * @brief Local slot, engine, and lane authority for arbitrary-tier moves.
     *
     * `create()` is a model-setup operation.  It requires the complete initial
     * participant banks because those live prepared engines are the authority
     * for projection geometry and source-format provenance.  GPU shadow slots
     * are deliberately capacity-shaped for every catalogued execution format;
     * each arrival still receives its exact decoder and arithmetic-policy
     * identities.  CPU slots retain the exact model source identity and final
     * NativeVNNI representation.
     *
     * Slot and engine ownership remain process-local. When a private remote
     * projection transport is supplied, the fabric composes those local slots
     * with transaction-wide MPI operations and returns the same fixed operation
     * cardinality on every rank. Remote admission is whole-wave atomic and no
     * network request begins before the outer reservation consensus.
     */
    class MoEOverlayPhysicalResidencyFabric final
        : public IMoEOverlayParticipantTransferProvider,
          public IMoEOverlayTransferProgressAuthority,
          public std::enable_shared_from_this<
              MoEOverlayPhysicalResidencyFabric>
    {
    public:
        /**
         * @brief Opaque model-lifetime maps, lanes, pools, and counters.
         *
         * The declaration is public so translation-unit helpers can construct
         * the implementation without exposing any of its fields in this
         * header. Ownership remains private through @ref impl_.
         */
        struct Impl;

        /** @brief Immutable model-lifetime materialization policy. */
        struct Config
        {
            /** Complete process-local initial-bank registry. */
            std::shared_ptr<MoEOverlayParticipantResidencyRegistry> registry;
            /** Exact epoch and owner map installed in @ref registry. */
            std::shared_ptr<const MoEOverlayResidencySnapshot> initial_snapshot;
            /**
             * Optional globally-published layer manifest.
             *
             * A process with at least one resident expert in every layer can
             * derive this contract from its live initial banks.  Distributed
             * composition supplies it explicitly when a process hosts an empty
             * receiving tier/layer whose only initial sources are remote.  Any
             * supplied entry is authenticated against every local resident.
             */
            std::vector<MoEOverlayLayerWeightManifest> layer_weight_manifest;
            /**
             * Private cross-rank data plane, required by remote migrations.
             * Null is valid only for a topology whose movement endpoints are
             * always process-local. The runner owns communicator lifetime and
             * must destroy this fabric before destroying the transport.
             */
            std::shared_ptr<MoEOverlayMPIRemoteProjectionTransport>
                remote_projection_transport;
            /**
             * Maximum incoming experts retained per participant/layer wave.
             * A proposal requiring more is a configuration error, not transient
             * backpressure, because runtime allocation is forbidden.
             */
            std::size_t shadow_slots_per_endpoint_layer = 1;
            /** Bounded CPU-format or GPU host-relay staging bytes per lane. */
            std::size_t staging_capacity_bytes = 4u * 1024u * 1024u;
            /**
             * Maximum closed migration cycles admitted in one wave.
             *
             * The fabric combines this with logical-participant multiplicity
             * per physical device and pre-materializes the resulting lane
             * pools. It must equal the residency authority's scheduling cap.
             */
            std::size_t maximum_concurrent_cycles = 1;
            /**
             * Collect exact device/host timing evidence on local transfer lanes.
             * Dynamic production enables this for economy certification;
             * static residency leaves timing events unmaterialized.
             */
            bool collect_economy_measurements = false;
            /** Stable topology label attached to PerfStats evidence. */
            std::string perf_device;
        };

        /**
         * @brief Materialize every local pool, stream, event, and staging slot.
         * @param config Complete initial-bank and capacity contract.
         * @return Shared provider suitable for a participant wave factory.
         * @throws std::invalid_argument for incomplete topology or capacity.
         * @throws std::runtime_error for absent engines/devices or allocation
         *         and lane-materialization failures.
         */
        static std::shared_ptr<MoEOverlayPhysicalResidencyFabric> create(
            Config config);

        /** @brief Drain-free destruction; no lane may still own runtime work. */
        ~MoEOverlayPhysicalResidencyFabric() override;

        MoEOverlayPhysicalResidencyFabric(
            const MoEOverlayPhysicalResidencyFabric &) = delete;
        MoEOverlayPhysicalResidencyFabric &operator=(
            const MoEOverlayPhysicalResidencyFabric &) = delete;

        /**
         * @brief Atomically reserve a local closed wave and create its operations.
         * @param transaction Exact old/new epoch and complete migration cycles.
         * @param local_destination_participants Sorted ids owned by this process.
         * @return Started projection work, typed transient backpressure, or an
         *         exact fatal topology/capacity diagnostic.
         */
        MoEOverlayParticipantPreparedTransfers prepareTransfers(
            const MoEOverlayResidencyTransaction &transaction,
            const std::vector<int> &local_destination_participants) override;

        /**
         * @brief Prepare real weight arrivals for a device-authored Dynamic wave.
         *
         * The supplied batch has already projected the immutable device command
         * into physical endpoints and closed cycles. This method consumes those
         * facts only; it cannot inspect histograms or author an owner map. A
         * zero-movement command is completed by the transport protocol without
         * entering the physical fabric, and transient LLEP lifetime management
         * uses its separately typed restoration path.
         *
         * @param batch Authenticated non-empty durable device movement.
         * @param local_destination_participants Sorted ids owned by this process.
         * @return Started asynchronous projection work, typed backpressure, or
         *         a fatal physical topology/capacity diagnostic.
         */
        MoEOverlayParticipantPreparedTransfers prepareDeviceTransfers(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            const std::vector<int> &local_destination_participants);

        /**
         * @brief Retain completed destination triplets before device RCU apply.
         * @param batch Exact begun device movement identity.
         * @param prepared Physical operations after all projection polls are Ready.
         * @param error Optional exact rejection diagnostic.
         * @return True after every process-local destination lifetime is staged.
         */
        [[nodiscard]] bool stageDevicePreparedTransfers(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            const MoEOverlayParticipantPreparedTransfers &prepared,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Make staged lifetimes durable before bounded device RCU apply.
         * @param batch Exact staged device movement identity.
         * @param error Optional exact rejection diagnostic.
         * @return True after the physical ledger advances to the candidate epoch.
         *
         * The physical ledger owns allocation lifetimes only. Advancing it
         * does not expose the candidate to inference; the device publication
         * epoch remains the sole runtime selector and admission authority.
         */
        [[nodiscard]] bool publishDevicePreparedTransfers(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Release old physical sources after device reader retirement.
         * @param batch Exact published device movement identity.
         * @param error Optional exact rejection diagnostic.
         * @return True after departed slots are recyclable and the wave is closed.
         */
        [[nodiscard]] bool retireDevicePreviousSources(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Forget an unpublished physical wave after operation abort drains.
         * @param batch Exact begun or staged device movement identity.
         * @param error Optional exact rejection diagnostic.
         * @return True when no prepared destination remains publishable.
         */
        [[nodiscard]] bool abortDeviceTransfers(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Return departed bootstrap slots to the recyclable live arena.
         *
         * The participant transaction invokes this only after publication and
         * the old epoch's inference-ticket barrier. Arrived slots are released
         * by their lease aliases; this method handles initial loader-owned
         * slots that predate the lease arena.
         */
        void retirePreviousSources(
            std::uint64_t retired_epoch,
            const std::vector<MoEOverlayTierMigration> &migrations) noexcept override;

        /**
         * @brief Return the shared retained relay epoch for one local GPU.
         * @param device Exact process-local GPU endpoint.
         * @return Shared epoch, or null when that GPU has no host-relay edge.
         *
         * Device executors retain this handle as a cache-owned captured branch.
         * The physical fabric remains the sole authority for topology-sized
         * command-slot accounting.
         */
        [[nodiscard]] std::shared_ptr<MappedTransferProgressEpoch>
        transferProgressEpoch(DeviceId device) const noexcept;

        /**
         * @brief Enumerate exact GPUs whose lane BOM requires mapped progress.
         * @return Stable device-sorted setup inventory.
         *
         * Orchestration uses this code-owned inventory to install every epoch
         * before native serving graphs are captured. It must not infer demand
         * again from controller bindings, because host-authoritative overlays
         * intentionally have no device-controller runtime table.
         */
        [[nodiscard]] std::vector<DeviceId> transferProgressDevices() const;

        /**
         * @brief Enqueue every outstanding mapped GPU relay epoch once.
         * @param error Optional exact device/worker submission diagnostic.
         * @return True after every non-idle epoch is enqueued without waiting
         *         for device completion.
         *
         * The inventory is keyed by physical device, so several lanes and
         * projections sharing an epoch never create duplicate launch authority.
         */
        [[nodiscard]] bool submitOutstandingTransferProgress(
            std::string *error = nullptr) noexcept override;

        /**
         * @brief Canonicalize the restored placement for model-context reuse.
         * @param published_epoch Exact quiescent participant-bank epoch.
         * @param error Optional precise lifecycle, capacity, or transfer failure.
         * @return A complete terminal seal, or no value on a fatal invariant.
         *
         * The caller must already own the typed terminal context-seal lifecycle:
         * no inference ticket, migration wave, retirement, or abort may remain.
         * This method first proves that every current resident matches the initial
         * prepared owner map, then copies only non-canonical residents through the
         * setup-owned CPU/GPU lanes.  GPU completion is event-polled and CPU work
         * uses the retained background workers; no stream or device synchronization
         * is introduced.  The returned banks, rather than the transient published
         * banks, are the sole legal source for the reusable model registry.
         */
        [[nodiscard]] std::optional<MoEOverlayReusableContextSeal>
        sealReusableInitialPlacement(
            std::uint64_t published_epoch,
            std::string *error = nullptr) noexcept;

        /** @return Race-safe cumulative model-lifetime proof counters. */
        [[nodiscard]] MoEOverlayPhysicalResidencyFabricStats stats()
            const noexcept;

    private:
        /**
         * @brief Common physical implementation after either authority validates.
         *
         * Every argument is transport identity or byte/slot work. Keeping the
         * host transaction and device command wrappers outside this method
         * prevents the shared data plane from acquiring a second policy input.
         */
        MoEOverlayParticipantPreparedTransfers preparePhysicalTransfers(
            MoEOverlayResidencyTransactionPurpose purpose,
            std::uint64_t expected_epoch,
            std::uint64_t candidate_epoch,
            const MoEOverlayResidencyExecutionFingerprint &
                execution_fingerprint,
            const std::vector<MoEOverlayTierMigration> &migrations,
            const std::vector<MoEOverlayTierMigrationCycle> &migration_cycles,
            const std::vector<MoEOverlayTierShadowRequirement> &
                shadow_requirements,
            const MoEOverlayDevicePhysicalMovementBatch *device_batch,
            const std::vector<int> &local_destination_participants);

        /** @brief Retain a completely materialized implementation. */
        explicit MoEOverlayPhysicalResidencyFabric(
            Config config,
            std::unique_ptr<Impl> impl) noexcept;

        Config config_;
        std::unique_ptr<Impl> impl_;
    };
} // namespace llaminar2
