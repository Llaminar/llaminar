/**
 * @file MoEOverlayNodeLocalRouteExchange.h
 * @brief Topology-generic mapped fabric for same-node canonical MoE routes.
 *
 * This owner creates one persistent mapped lane for every non-root device in a
 * continuation domain.  It is intentionally independent of CUDA, ROCm, MPI
 * rank number, tier name, and participant count.  Planning supplies exact
 * `DeviceId` endpoints and stable logical participant ids; TransferEngine
 * resolves the possibly different mapped alias on every endpoint.
 *
 * The object is shared by all participant-local graph builders in one
 * RankOrchestrator.  Materialization is idempotent and thread-safe because
 * those builders are created concurrently.  Capacity is the model-lifetime
 * graph envelope, never the first request's row count, so no captured pointer
 * can be invalidated by a larger prefill bucket later.
 */

#pragma once

#include "MoEOverlayNodeLocalRouteExchangeABI.h"
#include "backends/DeviceId.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class DeviceTransferBuffer;
    class MappedHostTransferRegion;

    /** Logical participant and exact process-local accelerator endpoint. */
    struct MoENodeLocalRouteEndpoint
    {
        int participant_id = -1;
        DeviceId device = DeviceId::invalid();

        bool operator==(const MoENodeLocalRouteEndpoint &) const = default;
    };

    /**
     * @brief Persistent node-local route lanes shared by captured device graphs.
     *
     * One object serves all graph roles and row buckets for a continuation
     * domain.  Payload storage is reused only after the root publishes its
     * device-owned acknowledgement; no inference hot-path method allocates,
     * registers pages, changes topology, or touches host memory.
     */
    class MoEOverlayNodeLocalRouteExchange final
    {
    public:
        /** Immutable process-local device topology known by RankOrchestrator. */
        struct Config
        {
            std::vector<DeviceId> devices;
            DeviceId root_device = DeviceId::invalid();
            std::string identity;
        };

        /**
         * @brief Construct an unmaterialized route fabric.
         * @param config Complete local device set and exact root device.
         * @throws std::invalid_argument for invalid/duplicate/non-GPU devices.
         */
        explicit MoEOverlayNodeLocalRouteExchange(Config config);

        /** Release page registrations before the retained mmap lifetime. */
        ~MoEOverlayNodeLocalRouteExchange();

        MoEOverlayNodeLocalRouteExchange(
            const MoEOverlayNodeLocalRouteExchange &) = delete;
        MoEOverlayNodeLocalRouteExchange &operator=(
            const MoEOverlayNodeLocalRouteExchange &) = delete;
        MoEOverlayNodeLocalRouteExchange(
            MoEOverlayNodeLocalRouteExchange &&) = delete;
        MoEOverlayNodeLocalRouteExchange &operator=(
            MoEOverlayNodeLocalRouteExchange &&) = delete;

        /**
         * @brief Install participant identity and allocate the full graph envelope.
         *
         * Concurrent calls with the same arguments are idempotent.  Any change
         * in participant/device identity or capacity is fatal because captured
         * graphs may already embed the original aliases.
         *
         * @param endpoints Every continuation participant exactly once.
         * @param root_participant Stable logical id hosted by `root_device`.
         * @param max_rows Largest physical graph row bucket.
         * @param top_k Original router slots per row.
         * @param d_model FP32 elements in one routed output row.
         */
        void materialize(
            std::vector<MoENodeLocalRouteEndpoint> endpoints,
            int root_participant,
            std::uint32_t max_rows,
            std::uint32_t top_k,
            std::uint32_t d_model);

        /** @return Whether immutable mapped lanes have been registered. */
        [[nodiscard]] bool materialized() const noexcept;

        /** @return Exact configured root accelerator. */
        [[nodiscard]] DeviceId rootDevice() const noexcept
        {
            return config_.root_device;
        }

        /** @return Stable logical root id after materialization, otherwise -1. */
        [[nodiscard]] int rootParticipant() const noexcept;

        /** @return Full route-slot capacity shared by every peer lane. */
        [[nodiscard]] std::uint32_t routeCapacity() const noexcept;

        /** @return Hidden width shared by every peer lane. */
        [[nodiscard]] std::uint32_t dModel() const noexcept;

        /** @return Maximum physical rows retained by every captured graph. */
        [[nodiscard]] std::uint32_t maxRows() const noexcept;

        /**
         * @brief Resolve one producer lane through that producer's exact alias.
         * @param producer_device Non-root endpoint executing the publication.
         * @return Capture-stable mapped binding for producer kernels.
         * @throws std::logic_error if topology is not materialized or mismatched.
         */
        [[nodiscard]] MoENodeLocalRoutePeerDeviceBinding producerBinding(
            DeviceId producer_device) const;

        /**
         * @brief Resolve every peer lane through the root's exact device alias.
         * @param root_device Must equal the configured root.
         * @return Bindings sorted by stable producer participant id.
         * @throws std::logic_error if topology is not materialized or mismatched.
         */
        [[nodiscard]] std::vector<MoENodeLocalRoutePeerDeviceBinding>
        rootPeerBindings(DeviceId root_device) const;

        /**
         * @brief Resolve the dense root-publication channel for one endpoint.
         *
         * The returned aliases are capture-stable and endpoint-correct. Root
         * and peer roles derive from immutable device identity, never from the
         * caller's requested collective operation.
         *
         * @param device Exact continuation endpoint embedding the binding.
         * @return Device-facing control, payload, and acknowledgement aliases.
         * @throws std::logic_error when materialization or identity is invalid.
         */
        [[nodiscard]] MoENodeLocalDensePublicationDeviceBinding
        densePublicationBinding(DeviceId device) const;

        /**
         * @brief Retain the registered mapped region used by dense publication.
         * @return Shared registration lifetime for captured transfer nodes.
         * @throws std::logic_error before complete materialization.
         */
        [[nodiscard]] std::shared_ptr<const MappedHostTransferRegion>
        mappedRegion() const;

        /** @return Byte offset of the shared dense FP32 payload matrix. */
        [[nodiscard]] std::size_t densePublicationPayloadOffset() const;

        /** @return Maximum dense FP32 elements admitted by setup. */
        [[nodiscard]] std::size_t densePublicationElementCapacity() const;

        /** @return Immutable endpoint identities sorted by participant id. */
        [[nodiscard]] std::vector<MoENodeLocalRouteEndpoint> endpoints() const;

        /** @return Human-readable topology, capacity, and allocation evidence. */
        [[nodiscard]] std::string diagnostics() const;

        /**
         * @brief Publish a terminal sentinel before endpoint streams are drained.
         *
         * This is a shutdown/error path only. It never participates in a live
         * inference transaction and exists so teardown can release a producer
         * or root kernel that is waiting for a peer whose graph failed. The
         * owning orchestrator must still drain endpoint streams before allowing
         * this fabric's mapped pages to be unregistered.
         */
        void abortForShutdown() noexcept;

    private:
        /** Byte offsets for one non-root lane within the shared allocation. */
        struct LaneLayout
        {
            DeviceId producer_device = DeviceId::invalid();
            int producer_participant = -1;
            std::size_t control_offset = 0u;
            std::size_t slot_epochs_offset = 0u;
            std::size_t payload_offset = 0u;
            std::size_t root_staging_offset = 0u;
        };

        /** @return Endpoint-specific binding; caller must hold `mutex_`. */
        [[nodiscard]] MoENodeLocalRoutePeerDeviceBinding bindingLocked(
            const LaneLayout &lane,
            DeviceId alias_device) const;

        Config config_;
        mutable std::mutex mutex_;
        bool materialized_ = false;
        int root_participant_ = -1;
        std::uint32_t route_capacity_ = 0u;
        std::uint32_t max_rows_ = 0u;
        std::uint32_t d_model_ = 0u;
        std::size_t mapping_bytes_ = 0u;
        std::size_t dense_publication_control_offset_ = 0u;
        std::size_t dense_publication_peers_offset_ = 0u;
        std::size_t dense_publication_payload_offset_ = 0u;
        std::size_t dense_publication_element_capacity_ = 0u;
        std::vector<MoENodeLocalRouteEndpoint> endpoints_;
        std::vector<LaneLayout> lanes_;
        std::shared_ptr<void> mapping_lifetime_;
        std::shared_ptr<MappedHostTransferRegion> mapped_region_;
        std::shared_ptr<DeviceTransferBuffer> root_staging_;
    };
} // namespace llaminar2
