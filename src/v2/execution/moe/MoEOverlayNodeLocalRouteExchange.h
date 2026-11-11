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
#include "planning/PhysicalMemoryAuthority.h"

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
    class DeviceWorkspaceManager;
    class WorkspaceBufferLease;
    struct ClusterInventory;
    namespace test { class NodeLocalRouteExchangeReplayAccess; }
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

        /** @brief Exact shared layout declaration; only declareStorage may issue it. */
        class StorageDeclaration final
        {
        public:
            /** @return Original host mapping plus exact root workspace BOM, without allocation. */
            [[nodiscard]] PhysicalMemoryPlan bom(PhysicalMemoryResource host, PhysicalMemoryResource root_gpu) const;
            /** @return Page-aligned logical mapped protocol capacity, excluding arena slack. */
            [[nodiscard]] std::size_t mappingBytes() const noexcept { return mapping_bytes_; }
            /** @return Exact logical root staging capacity, excluding workspace alignment. */
            [[nodiscard]] std::size_t rootBytes() const noexcept { return root_bytes_; }
        private:
            friend class MoEOverlayNodeLocalRouteExchange;
            StorageDeclaration(std::vector<DeviceId> devices, DeviceId root, std::string identity,
                std::uint32_t rows, std::uint32_t top_k, std::uint32_t width,
                std::size_t mapping_bytes, std::size_t root_bytes)
                : devices_(std::move(devices)), root_(root), identity_(std::move(identity)), rows_(rows), top_k_(top_k),
                  width_(width), mapping_bytes_(mapping_bytes), root_bytes_(root_bytes) {}
            std::vector<DeviceId> devices_;
            DeviceId root_;
            std::string identity_;
            std::uint32_t rows_, top_k_, width_;
            std::size_t mapping_bytes_, root_bytes_;
        };
        /** @brief Price the same checked protocol geometry later materialized by the owner. */
        [[nodiscard]] static StorageDeclaration declareStorage(const Config&, std::uint32_t max_rows,
            std::uint32_t top_k, std::uint32_t d_model);
        /**
         * @brief Bind the complete admitted storage to its original rank-owned GPU cohort.
         * @param config Exact continuation endpoints and root.
         * @param declaration Original maximum graph geometry used by the admission BOM.
         * @param memory Sole original source authority; missing authority is invalid.
         * @param inventory Original complete hardware publication in that authority's namespace.
         * @throws std::invalid_argument for any ownership, geometry or declaration mismatch.
         */
        MoEOverlayNodeLocalRouteExchange(Config config, StorageDeclaration declaration,
            std::shared_ptr<PhysicalMemoryAuthority> memory,
            std::shared_ptr<const ClusterInventory> inventory);

        /**
         * @brief Reject a reused owner from another original source, inventory or retained graph envelope.
         * @param declaration Same original exact topology/domain/geometry used by admission.
         * @param memory Actual original model authority, compared by ownership identity.
         * @param inventory Exact retained original hardware publication, never a reconstructed view.
         * @throws std::logic_error for a raw-test owner, foreign authority or changed capture identity.
         */
        void requireOriginalAdmission(const StorageDeclaration& declaration,
            const std::shared_ptr<PhysicalMemoryAuthority>& memory,
            const std::shared_ptr<const ClusterInventory>& inventory) const;

        /** Release the TransferEngine-owned mapping after captured users drain. */
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
        friend class test::NodeLocalRouteExchangeReplayAccess;
        /** @brief Explicit test-only constructor: no production caller can omit admission. */
        explicit MoEOverlayNodeLocalRouteExchange(Config config);
        /** @brief Closed allocation authority; only the test friend can issue isolated storage. */
        enum class StorageOwnership { OriginalAdmitted, IsolatedTestReplay };
        /** @brief Share pure topology validation without a nullable production authority arm. */
        MoEOverlayNodeLocalRouteExchange(Config, StorageOwnership);
        /** @brief Pure endpoint validation shared by cold declaration and materialization. */
        static void validateConfig(const Config&);
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

        /** @brief Checked layout used identically by the quote and allocation. */
        struct StorageLayout {
            std::vector<LaneLayout> lanes;
            std::uint32_t route_capacity;
            std::size_t mapping_bytes, root_bytes, dense_control, dense_peers, dense_payload,
                dense_elements, dense_payload_bytes;
        };
        /** @return Full endpoint-independent physical layout; no native operation. */
        static StorageLayout storageLayout(std::size_t participants, std::uint32_t rows,
            std::uint32_t top_k, std::uint32_t width);
        /** @return Endpoint-specific binding; caller must hold `mutex_`. */
        [[nodiscard]] MoENodeLocalRoutePeerDeviceBinding bindingLocked(
            const LaneLayout &lane,
            DeviceId alias_device) const;

        Config config_;
        const StorageOwnership ownership_;
        // Original authority precedes all physical owners, so it retires last.
        std::shared_ptr<PhysicalMemoryAuthority> memory_;
        std::shared_ptr<const ClusterInventory> inventory_;
        std::unique_ptr<StorageDeclaration> declaration_;
        std::unique_ptr<DeviceWorkspaceManager> root_workspace_;
        std::shared_ptr<const WorkspaceBufferLease> root_lease_;
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
        std::shared_ptr<MappedHostTransferRegion> mapped_region_;
        std::shared_ptr<DeviceTransferBuffer> root_staging_;
    };
} // namespace llaminar2
