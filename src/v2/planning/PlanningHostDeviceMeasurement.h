/**
 * @file PlanningHostDeviceMeasurement.h
 * @brief Bounded captured GPU/host link observations, distinct from MPI and native collectives.
 *
 * The declared host rank first-touches its shared pages before the NUMA-local
 * GPU executor registers them. Original all-rank agreement fences preparation
 * and retirement; neither role may substitute another rank's scope. Exact
 * streams then measure both directions through the production
 * DMA and mapped-copy-kernel mechanisms. These are byte-service primitives, not
 * complete activation-packet protocol timings or measured model throughput.
 */
#pragma once
#include "PlanningExecutionMeasurement.h"
#include "PhysicalMemoryAuthority.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "transfer/MappedTransferProgressABI.h"
#include <vector>
#include <optional>
#include <memory>

namespace llaminar2
{
    class IMPIContext;
    /** @brief Mechanisms with separate observations; no failed-operation substitution. */
    enum class PlanningHostTransferMechanism { DMA, MappedKernel };

    /** @brief Native executor role; visibility alone cannot authorize this GPU. */
    struct PlanningHostDeviceGPUExecutor final
    {
        int rank;
        DeviceId device;
        bool operator==(const PlanningHostDeviceGPUExecutor &) const = default;
    };
    /** @brief Setup caller that owns physical host first touch, never GPU execution. */
    struct PlanningHostDeviceFirstTouch final
    {
        int rank;
        bool operator==(const PlanningHostDeviceFirstTouch &) const = default;
    };
    /** @brief Immutable distinct GPU-executor and same-machine host-placement roles. */
    class PlanningHostDeviceRequest final
    {
    public:
        /** @brief Resolve both roles from one original inventory namespace.
         * @param inventory Original complete discovery publication.
         * @param gpu NUMA-local native executor and its original device ordinal.
         * @param host Original first-touch rank, with a known NUMA on the same physical machine.
         * @param bulk_bytes Existing positive larger byte sample, bounded by INT_MAX.
         * @return Sealed GPU UUID and observed host NUMA without another query.
         * @throws std::invalid_argument For foreign/unknown ownership or geometry. */
        static PlanningHostDeviceRequest fromInventory(const ClusterInventory &inventory,
            PlanningHostDeviceGPUExecutor gpu, PlanningHostDeviceFirstTouch host, size_t bulk_bytes);
        /** @return Rank permitted to own the native context/stream/measurement. */
        int gpuRank() const noexcept { return gpu_.rank; }
        /** @return Original setup caller that creates/first-touches the host pages. */
        int hostRank() const noexcept { return host_.rank; }
        /** @return Original physical machine shared by both roles. */
        int physicalNode() const noexcept { return node_; }
        /** @return Original known host NUMA; no unknown or rank-derived surrogate. */
        int hostNumaNode() const noexcept { return host_numa_; }
        /** @return Original GPU ordinal in its executor's visible namespace. */
        DeviceId device() const noexcept { return gpu_.device; }
        /** @return Original physical GPU identity, scoped to machine/backend. */
        const std::string &uuid() const noexcept { return uuid_; }
        /** @return Live larger sample extent, not page-rounded allocation bytes. */
        size_t bulkBytes() const noexcept { return bulk_bytes_; }
        /** @return Exact metadata identity used by existing planning agreement/receipts. */
        std::vector<uint8_t> serialize() const;
        bool operator==(const PlanningHostDeviceRequest &) const = default;
    private:
        PlanningHostDeviceRequest(PlanningHostDeviceGPUExecutor gpu, PlanningHostDeviceFirstTouch host,
            int node, int numa, std::string uuid, size_t bulk)
            : gpu_(gpu), host_(host), node_(node), host_numa_(numa), uuid_(std::move(uuid)), bulk_bytes_(bulk) {}
        PlanningHostDeviceGPUExecutor gpu_;
        PlanningHostDeviceFirstTouch host_;
        int node_, host_numa_;
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
        /** @brief Quote the two shared host buffers on their first-touch owner.
         * @param request Same sealed identity used by collect.
         * @param host Original CPU resource on request.hostRank().
         * @param builder Canonical PMA BOM, never another rank's ledger. */
        static void contributeHostMemory(const PlanningHostDeviceRequest &request,
            PhysicalMemoryResource host, PhysicalMemoryPlanBuilder &builder);
        /** @brief Quote GPU buffer and two graphs on the native executor.
         * @param request Same sealed identity used by collect.
         * @param execution Original GPU resource on request.gpuRank().
         * @param builder Canonical PMA BOM; no duplicate host-page charge. */
        static void contributeExecutionMemory(const PlanningHostDeviceRequest &request,
            PhysicalMemoryResource execution, PhysicalMemoryPlanBuilder &builder);
        /** @return Exact page-rounded pair extent shared by admission and mmap. */
        static size_t hostAllocationBytes(const PlanningHostDeviceRequest &request);
        /** @brief Collect one original sample with both roles and all-rank retirement.
         * @param request Immutable executor/first-touch/byte identity, agreed by every rank.
         * @param mpi Existing discovery communicator; null means its one-rank local form.
         * @param memory This caller's rank-bound physical admission.
         * @return Eight native observations only on the GPU owner; no result on other roles.
         *
         * Host first touch precedes registration through existing all-rank
         * agreement. Native clocks retain the original warmup/three-invocation
         * mechanism; source publication and byte checks remain outside timing.
         * The final agreement proves GPU graphs/registrations have retired
         * before the host owner releases any shared page or PMA claim.
         */
        static std::optional<PlanningHostDeviceObservations> collect(const PlanningHostDeviceRequest &request,
            const std::shared_ptr<IMPIContext> &mpi, const std::shared_ptr<PhysicalMemoryAuthority> &memory);
        /** @brief Reject incomplete/duplicate coordinates, foreign units, empty graphs or wrong completed work. */
        static void validate(const PlanningHostDeviceObservations &observation);
    };
}
