/**
 * @file PlanningCommunicationService.h
 * @brief Bounded topology-bound native, host-MPI and GPU/host startup evidence.
 *
 * Physical groups, process ownership and locality come from discovery. Actual
 * captured NCCL/RCCL observations and completed host-MPI exchanges keep separate
 * typed identities: neither substitutes for GPU DMA or mapped kernel access.
 * The GPU/host primitives have separate mechanism/direction/first-touch identity;
 * they are not complete activation-packet timings. No model warmup or per-candidate probe occurs.
 */
#pragma once
#include "PlanningLocalTPMeasurement.h"
#include "PlanningMPITransferMeasurement.h"
#include "PlanningHostDeviceMeasurement.h"
#include "PlanningPublication.h"
#include "config/OrchestrationPlanningPolicy.h"
#include <optional>

namespace llaminar2
{
    /** @brief One common observer owns this whole physical native collective. */
    struct PlanningNativeCollectiveSample
    {
        int discovery_rank;
        int physical_node;
        std::vector<std::string> uuids; ///< Same order as request devices, never rank-local aliases.
        PlanningLocalTPRequest request;
    };

    /**
     * @brief Immutable bounded sample basis, not the Cartesian product of candidate domains.
     *
     * Native samples cover maximal homogeneous groups visible together to a
     * rank, deduplicated by physical machine/backend/UUID set. Smaller candidate
     * subsets are not advertised as measured. Host MPI samples cover directed
     * eligible rank pairs with control, outbound-heavy and return-heavy exchanges.
     * The two directions share no assumed bandwidth or half-RTT latency. An
     * explicitly single-device search needs neither kind of communication.
     */
    class PlanningCommunicationSamplePlan final
    {
    public:
        /**
         * @brief Resolve actual discovery ownership without allocating or probing.
         * @param inventory Same canonical publication used by candidate construction.
         * @param request Hard compute/strategy constraints, never preference-based filtering.
         * @param hidden_width Positive model activation width.
         * @param prefill_rows Positive bounded sample rows; not an inference context limit.
         * @param precisions Native wire precisions to measure; nonempty, distinct and valid.
         */
        static PlanningCommunicationSamplePlan resolve(const ClusterInventory &inventory,
            const AutomaticOrchestrationRequest &request, int hidden_width, int prefill_rows,
            std::span<const PlanningAllreducePrecision> precisions);
        /** @return Native groups and their complete observing process ownership. */
        const std::vector<PlanningNativeCollectiveSample> &native() const noexcept { return native_; }
        /** @return Directed exact host request/reply geometries, with no inferred reverse symmetry. */
        const std::vector<PlanningMPITransferRequest> &mpi() const noexcept { return mpi_; }
        /** @return GPU/host primitives for each permitted rank-visible GPU and that rank's first-touch scope. */
        const std::vector<PlanningHostDeviceRequest> &hostDevice() const noexcept { return host_device_; }
        /** @return Complete checked plan and membership for all-rank agreement, excluding measured times. */
        std::vector<uint8_t> serialize() const;
    private:
        /** @brief Only validated inventory resolution creates a plan. */
        PlanningCommunicationSamplePlan() = default;
        std::vector<PlanningNativeCollectiveSample> native_;
        std::vector<PlanningMPITransferRequest> mpi_;
        std::vector<PlanningHostDeviceRequest> host_device_;
        std::vector<int> rank_nodes_;
    };

    /** @brief Completed native observation bound to its physical group and reporting rank. */
    struct PlanningNativeCollectiveService
    {
        PlanningNativeCollectiveSample sample;
        PlanningLocalTPObservations observation;
    };

    /** @brief Complete process-local observations collected in the same failure-atomic rank rounds. */
    struct PlanningLocalCommunicationEvidence
    {
        std::vector<PlanningNativeCollectiveService> native;
        std::vector<PlanningHostDeviceObservations> host_device;
    };

    /** @brief One complete immutable evidence batch; no execution or allocation owners escape. */
    class PlanningCommunicationService final
    {
    public:
        /** @return Strict versioned native receipts; sample indices bind the complete group identity. */
        static std::vector<uint8_t> encodeLocal(const PlanningCommunicationSamplePlan &plan,
            std::span<const std::pair<size_t, PlanningLocalTPObservations>> observations,
            std::span<const std::pair<size_t, PlanningHostDeviceObservations>> host_device);
        /** @return Complete local evidence authenticated against transport-supplied rank IDs. */
        static PlanningLocalCommunicationEvidence acceptLocal(const PlanningCommunicationSamplePlan &plan,
            int world_size, std::span<const RankPlanningSample> receipts);
        /**
         * @brief Require complete native request, phase, graph and ordered endpoint evidence.
         * @param service One receipt already bound to its observing process by the collector.
         *
         * Cost composition reuses this same validation; it does not maintain a
         * second looser interpretation of a native observation. Physical UUID
         * and rank membership are additionally checked against its inventory.
         */
        static void validateNative(const PlanningNativeCollectiveService &service);
        /**
         * @brief Collect short native, GPU/host and MPI samples on discovery membership, root result only.
         * @param mpi Live discovery communicator; null denotes an explicitly local process.
         * @param inventory That communicator's canonical immutable inventory publication.
         * @param request Validated hard participant/strategy constraints.
         * @param hidden_width Model activation width, not a guessed allocation size.
         * @param prefill_rows Bounded physical rows to observe alongside M=1.
         * @param precisions Explicit native wire precision family to measure.
         * @return Complete evidence at discovery root; absence on followers after consensus.
         * @throws std::exception if agreement, admission, execution or evidence validation fails.
         *
         * Plan agreement and PMA admission finish before any device or packet
         * work. Physical-node rounds avoid sampling two overlapping reporters
         * simultaneously. Native and GPU/host graphs retire before MPI payload reuse. Root
         * cannot receive a partial catalog, and measurement failures are fatal
         * to this preparation transaction rather than silently assigned a cost.
         */
        static std::optional<PlanningCommunicationService> collect(const std::shared_ptr<IMPIContext> &mpi,
            const ClusterInventory &inventory, const AutomaticOrchestrationRequest &request,
            int hidden_width, int prefill_rows, std::span<const PlanningAllreducePrecision> precisions);
        /** @return Exact measured native groups; no prediction for an unseen subset is implied. */
        const std::vector<PlanningNativeCollectiveService> &native() const noexcept { return native_; }
        /** @return Completed host-MPI exchanges carrying canonical physical locality. */
        const std::vector<PlanningMPITransferObservation> &mpi() const noexcept { return mpi_; }
        /** @return Distinct DMA/mapped-kernel byte primitives, including host first-touch identity. */
        const std::vector<PlanningHostDeviceObservations> &hostDevice() const noexcept { return host_device_; }
    private:
        /** @brief Publish only after both complete measurement families passed consensus. */
        PlanningCommunicationService(std::vector<PlanningNativeCollectiveService> native,
            std::vector<PlanningMPITransferObservation> mpi, std::vector<PlanningHostDeviceObservations> host_device)
            : native_(std::move(native)), mpi_(std::move(mpi)), host_device_(std::move(host_device)) {}
        std::vector<PlanningNativeCollectiveService> native_;
        std::vector<PlanningMPITransferObservation> mpi_;
        std::vector<PlanningHostDeviceObservations> host_device_;
    };
}
