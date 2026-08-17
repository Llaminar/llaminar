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
#include <string>
#include <vector>

namespace llaminar2
{
    struct GGUFModel;
    class MoEOverlayMPIRemoteProjectionTransport;

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
            /** Bounded CPU-format or heterogeneous-blob staging bytes per lane. */
            std::size_t staging_capacity_bytes = 4u * 1024u * 1024u;
            /** Bytes kept free after each GPU shadow-slot arena allocation. */
            std::size_t gpu_vram_safety_margin_bytes = 0;
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

        /** @return Race-safe cumulative model-lifetime proof counters. */
        [[nodiscard]] MoEOverlayPhysicalResidencyFabricStats stats()
            const noexcept;

    private:
        /** @brief Retain a completely materialized implementation. */
        explicit MoEOverlayPhysicalResidencyFabric(
            Config config,
            std::unique_ptr<Impl> impl) noexcept;

        Config config_;
        std::unique_ptr<Impl> impl_;
    };
} // namespace llaminar2
