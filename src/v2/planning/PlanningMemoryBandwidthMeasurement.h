/**
 * @file PlanningMemoryBandwidthMeasurement.h
 * @brief Bounded streaming-memory observations on real CPU/CUDA/ROCm endpoints.
 *
 * Inventory supplies cache and worker geometry. Each of three FP32 streams is
 * four times the observed cache capacity, preventing a repeated small matrix
 * from masquerading as DRAM service. The existing production residual-add
 * kernel supplies two reads and one write per element; the result is useful
 * streaming bytes/second, not a hardware bus-counter measurement or compute
 * throughput. PhysicalMemoryAuthority owns every host/device allocation.
 */
#pragma once
#include "planning/PlanningExecutionMeasurement.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "execution/mpi_orchestration/DeviceInventory.h"

namespace llaminar2
{
    /** @brief Immutable cache-exceeding stream geometry from one rank's actual inventory. */
    class PlanningMemoryBandwidthRequest final
    {
    public:
        /**
         * @brief Resolve an exact local endpoint without rediscovery or capacity arithmetic.
         * @param rank Canonical observing rank, including its CPU workshare and GPU properties.
         * @param device CPU allocator zero or a GPU ordinal visible on that rank.
         * @throws std::invalid_argument for missing cache/worker/endpoint evidence.
         * @throws std::overflow_error for a stream extent outside the kernel ABI.
         *
         * CPU cache coverage includes the selected sockets' LLCs and the team's
         * private L2s. Unknown cache geometry fails; it is not replaced by a
         * guessed buffer size or a fraction of free RAM/VRAM.
         */
        static PlanningMemoryBandwidthRequest fromInventory(const RankInventory &rank, DeviceId device);
        /** @return Exact rank-local allocator, never an inferred physical host. */
        DeviceId device() const noexcept { return device_; }
        /** @return Discovery rank owning the observation and physical admission. */
        int rank() const noexcept { return rank_; }
        /** @return Observed aggregate cache capacity that every stream exceeds. */
        size_t cacheBytes() const noexcept { return cache_bytes_; }
        /** @return Equal extent of each input/output stream. */
        size_t streamBytes() const noexcept { return stream_bytes_; }
        /** @return Two input reads plus one output write; no write-allocate traffic is invented. */
        size_t usefulBytes() const noexcept { return 3 * stream_bytes_; }
        /** @return Exact CPU workshare, or zero for a GPU. */
        int workers() const noexcept { return workers_; }
        /** @return CPU policy identity; default/unused for a GPU. */
        const CPUExecutionGeometry &cpuGeometry() const noexcept { return cpu_; }
    private:
        /** @brief Only validated inventory projection may seal the sample geometry. */
        PlanningMemoryBandwidthRequest(DeviceId device, int rank, size_t cache_bytes,
            size_t stream_bytes, int workers, CPUExecutionGeometry cpu)
            : device_(device), rank_(rank), cache_bytes_(cache_bytes), stream_bytes_(stream_bytes), workers_(workers), cpu_(cpu) {}
        DeviceId device_;
        int rank_;
        size_t cache_bytes_, stream_bytes_;
        int workers_;
        CPUExecutionGeometry cpu_;
    };

    /** @brief Completed native streaming observation; no tensor or backend lifetime escapes. */
    struct PlanningMemoryBandwidthObservation
    {
        PlanningMemoryBandwidthRequest request;
        size_t graph_nodes; ///< Zero for CPU; a nonempty retained graph on either GPU backend.
        PlanningServiceObservation service;
    };

    /** @brief One preparation/capture followed by one warmup and three timed streaming passes. */
    class PlanningMemoryBandwidthMeasurement final
    {
    public:
        /**
         * @brief Add exact staging, stream payload and graph-family demands to the sole PMA.
         * @param request Validated endpoint/cache geometry.
         * @param host Same-rank CPU allocator used for first-touch initialization.
         * @param execution Exact endpoint allocator from the same inventory.
         * @param builder Canonical physical BOM; this method never subtracts available capacity.
         */
        static void contributeMemory(const PlanningMemoryBandwidthRequest &request,
            PhysicalMemoryResource host, PhysicalMemoryResource execution, PhysicalMemoryPlanBuilder &builder);
        /**
         * @brief Observe production streaming service with setup outside the timed interval.
         * @param request Exact geometry admitted by contributeMemory.
         * @param memory Rank-bound physical authority retaining every sample allocation.
         * @return Completed useful-byte observation after all private resources retire.
         *
         * CPU runs on the affined setup caller with its published OpenMP team.
         * GPU runs on its ordinary owning worker, binds an exact stream and
         * replays one retained native graph. There is no eager GPU operation,
         * default stream, hot-path allocation or whole-model warmup.
         */
        static PlanningMemoryBandwidthObservation measure(const PlanningMemoryBandwidthRequest &request,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory);
    };
}
