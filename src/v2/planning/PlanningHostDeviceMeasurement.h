/**
 * @file PlanningHostDeviceMeasurement.h
 * @brief Bounded captured GPU/host link observations, distinct from MPI and native collectives.
 *
 * One rank's setup thread first-touches its own mapped pages before GPU work is
 * submitted. Exact streams then measure both directions through the production
 * DMA and mapped-copy-kernel mechanisms. These are byte-service primitives, not
 * complete activation-packet protocol timings or measured model throughput.
 */
#pragma once
#include "PlanningExecutionMeasurement.h"
#include "PhysicalMemoryAuthority.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "transfer/MappedTransferProgressABI.h"
#include <vector>

namespace llaminar2
{
    /** @brief Mechanisms with separate observations; no failed-operation substitution. */
    enum class PlanningHostTransferMechanism { DMA, MappedKernel };

    /** @brief Immutable GPU visibility, host first-touch scope and two bounded payload sizes. */
    class PlanningHostDeviceRequest final
    {
    public:
        /**
         * @brief Resolve one rank-visible GPU and its rank-local host allocator.
         * @param rank Canonical process/physical-node/NUMA publication.
         * @param device Exact CUDA/ROCm ordinal in that process.
         * @param bulk_bytes Positive larger sample, greater than the one-byte control sample.
         * @throws std::invalid_argument for missing identities or unrepresentable payloads.
         */
        static PlanningHostDeviceRequest fromInventory(const RankInventory &rank, DeviceId device, size_t bulk_bytes);
        /** @return Process whose setup thread owns host first touch. */
        int rank() const noexcept { return rank_; }
        /** @return Physical machine from inventory, not inferred from link timing. */
        int physicalNode() const noexcept { return node_; }
        /** @return Declared rank host scope; -1 remains explicitly unqualified, not GPU-local. */
        int hostNumaNode() const noexcept { return host_numa_; }
        /** @return GPU ordinal visible to the observing process. */
        DeviceId device() const noexcept { return device_; }
        /** @return Physical GPU identity within this node and backend. */
        const std::string &uuid() const noexcept { return uuid_; }
        /** @return Larger exact payload; allocation padding is never timed as traffic. */
        size_t bulkBytes() const noexcept { return bulk_bytes_; }
        bool operator==(const PlanningHostDeviceRequest &) const = default;
    private:
        /** @brief Only validated inventory projection may establish first-touch/endpoint identity. */
        PlanningHostDeviceRequest(int rank, int node, int host_numa, DeviceId device, std::string uuid, size_t bulk)
            : rank_(rank), node_(node), host_numa_(host_numa), device_(device), uuid_(std::move(uuid)), bulk_bytes_(bulk) {}
        int rank_, node_, host_numa_;
        DeviceId device_;
        std::string uuid_;
        size_t bulk_bytes_;
    };

    /** @brief One completed payload/direction/mechanism coordinate. */
    struct PlanningHostDevicePhaseObservation
    {
        PlanningHostTransferMechanism mechanism;
        MappedTransferDirection direction;
        size_t payload_bytes;
        size_t graph_nodes;
        PlanningServiceObservation service;
    };
    /** @brief Complete immutable two-size, two-direction, two-mechanism evidence. */
    struct PlanningHostDeviceObservations
    {
        PlanningHostDeviceRequest request;
        std::vector<PlanningHostDevicePhaseObservation> phases;
    };

    /** @brief Setup-only producer with one reusable staging family and no model warmup. */
    class PlanningHostDeviceMeasurement final
    {
    public:
        /**
         * @brief Add exact two mapped host buffers, one device buffer and two retained graphs to PMA.
         * @param request Same sealed geometry passed to measure.
         * @param host Rank-local CPU allocator; never charged to the GPU's RAM.
         * @param execution Exact rank-visible GPU resource.
         * @param builder Canonical BOM owner; no parallel admission arithmetic.
         */
        static void contributeMemory(const PlanningHostDeviceRequest &request, PhysicalMemoryResource host,
            PhysicalMemoryResource execution, PhysicalMemoryPlanBuilder &builder);
        /**
         * @brief Measure from the affined rank setup caller, never a GPU worker or MPI progress callback.
         * @param request Immutable inventory/first-touch/payload identity.
         * @param memory Rank-bound authority admitting the complete declared family.
         * @return Eight positive native-event observations after graph/storage retirement.
         *
         * Host publication changes the source before each H2D invocation,
         * outside timing. Final byte verification and diagnostic readback are
         * also untimed. The normal GPU worker records/replays exact-stream
         * graphs; all copies go through TransferEngine, with no stream sync.
         */
        static PlanningHostDeviceObservations measure(const PlanningHostDeviceRequest &request,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory);
        /** @brief Reject incomplete/duplicate coordinates, foreign units, empty graphs or wrong completed work. */
        static void validate(const PlanningHostDeviceObservations &observation);
    };
}
