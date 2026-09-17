/**
 * @file PlanningMPITransferMeasurement.h
 * @brief Bounded, measured host MPI exchange costs on discovery membership.
 *
 * One root describes exact directed request/reply payloads. All ranks prepare
 * before traffic starts, using their canonical physical-memory authority.
 * Completed initiator-local round trips are gathered and authenticated at root.
 * This is a startup observation, not an inference transport or a device-state
 * controller. It must not price NCCL/RCCL, GPU copies, or arbitrary payload sizes.
 */
#pragma once

#include "backends/DeviceId.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace llaminar2
{
    class IMPIContext;
    class PhysicalMemoryAuthority;

    /**
     * @brief Exact ordered exchange geometry in the discovery rank namespace.
     *
     * Values are validated collectively, not by a constructor that could throw
     * on one rank before peers enter preparation. Both payloads must be positive
     * MPI-representable byte counts; reversing endpoints is a separate sample.
     */
    struct PlanningMPITransferRequest
    {
        int initiator = -1;
        int responder = -1;
        size_t request_bytes = 0;
        size_t reply_bytes = 0;
        bool operator==(const PlanningMPITransferRequest &) const = default;
    };

    /** @brief A completed exchange observation, never a fabricated one-way rate. */
    class PlanningMPITransferObservation final
    {
    public:
        /** @return Authenticated endpoint order and the exact timed payload sizes. */
        const PlanningMPITransferRequest &request() const noexcept { return request_; }
        /** @return Discovery physical membership; timings never classify locality. */
        const RankConnectionTopology &topology() const noexcept { return topology_; }
        /** @return Complete observed request/reply latency, including MPI progress. */
        double secondsPerExchange() const noexcept { return seconds_; }

    private:
        friend class PlanningMPITransferMeasurement;
        /** @brief Seal positive finite elapsed time only after every sample completed. */
        PlanningMPITransferObservation(PlanningMPITransferRequest request, RankConnectionTopology topology, double seconds);
        PlanningMPITransferRequest request_;
        RankConnectionTopology topology_;
        double seconds_;
    };

    /** @brief One short startup exchange batch; storage and preparation are untimed. */
    class PlanningMPITransferMeasurement final
    {
    public:
        /**
         * @brief Exact reusable host-payload BOM contribution for one rank.
         * @param requests Validated ordered sample list, which may be empty.
         * @param rank Discovery rank whose local storage is being planned.
         * @param world_size Actual discovery membership.
         * @return Largest request-plus-reply pair involving this rank, or zero.
         * @throws std::invalid_argument for malformed membership or sample geometry.
         *
         * Add this demand under ActivationTransportStaging before materializing.
         * Samples are sequential, so summing every pair would overallocate. No
         * device capacity or available-byte arithmetic is performed here.
         */
        static size_t workspaceBytes(std::span<const PlanningMPITransferRequest> requests,
            int rank, int world_size);

        /**
         * @brief Measure exact host MPI request/reply exchanges and gather at root.
         * @param mpi Live discovery context; all its ranks must enter this call.
         * @param memory Existing rank-bound authority admitting the workspace demand.
         *        May be null only for a rank absent from every request.
         * @param host_device CPU allocator whose admitted region is first-touched by this rank.
         * @param describe Root-only producer; returns the same geometry used for its BOM.
         * @return Complete ordered observations on root, an empty vector on followers.
         * @throws std::runtime_error on any preparation, admission, or evidence failure.
         *
         * One warmup precedes three timed exchanges per geometry. Receives are
         * posted before sends; the responder replies only after request completion.
         * Only the initiator's clock measures a round trip. Private communicator
         * ownership isolates tags from inventory and inference. MPI failure or a
         * 30-second stalled request aborts that communicator: pending buffers must
         * never be freed underneath MPI. Ordinary pre-traffic failures reach the
         * existing initialization consensus and allocate/send nothing on peers.
         *
         * Observations cover host MPI only. They include endpoint progress and
         * message startup, exclude remote compute and GPU DMA, and cannot stand
         * in for native GPU collectives or whole-model throughput measurements.
         */
        static std::vector<PlanningMPITransferObservation> measure(
            const std::shared_ptr<IMPIContext> &mpi,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory, DeviceId host_device,
            const std::function<std::vector<PlanningMPITransferRequest>()> &describe);
    };
}
