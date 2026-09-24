/**
 * @file PlanningLocalTPMeasurement.h
 * @brief Bounded native rank-local collective observations for topology costing.
 *
 * A request names one concrete homogeneous local GPU group. LocalTPContext
 * selects and executes its production NCCL/RCCL protocol, including precision
 * conversion. The receipt describes collective latency at exact payload sizes,
 * not a host-MPI rate, a per-link bandwidth, or an inference benchmark. Physical
 * machine/rank identity remains the enclosing inventory collector's authority.
 */
#pragma once
#include "planning/PhysicalMemoryAuthority.h"
#include "planning/CollectiveMemoryEstimator.h"
#include <span>
#include <vector>

namespace llaminar2
{
    /** @brief Wire representation only; the model-facing activation remains FP32. */
    enum class PlanningAllreducePrecision { FP32, FP16 };

    /** @brief Validated same-process native GPU group and two positive payload geometries. */
    class PlanningLocalTPRequest final
    {
    public:
        /**
         * @brief Reject duplicate, mixed-backend, CPU and single-device groups before allocation.
         * @param devices Rank-local ordinals in production collective order; never remote ordinals.
         * @param hidden_width Logical FP32 activation columns.
         * @param prefill_rows Positive physical prefill rows; decode always uses one row.
         * @param precision Explicit production collective transport precision.
         */
        PlanningLocalTPRequest(std::vector<DeviceId> devices, int hidden_width,
            int prefill_rows, PlanningAllreducePrecision precision);
        /** @return Immutable native collective order. */
        const std::vector<DeviceId> &devices() const noexcept { return devices_; }
        /** @return Logical activation width, independent of allocation padding. */
        int hiddenWidth() const noexcept { return hidden_width_; }
        /** @return Exact requested prefill sample geometry. */
        int prefillRows() const noexcept { return prefill_rows_; }
        /** @return Explicit wire representation; no runtime environment override. */
        PlanningAllreducePrecision precision() const noexcept { return precision_; }
        /** @return NCCL for CUDA or RCCL for ROCm; never a substitute host transport. */
        CollectiveBackendType backend() const noexcept;
        /** @return Exact in-place FP32 allocation extent at the maximum rows. */
        size_t activationBytes() const noexcept;
        /** @return Exact logical reduction payload; not aggregate wire traffic. */
        size_t payloadBytes(int rows) const;
    private:
        std::vector<DeviceId> devices_;
        int hidden_width_, prefill_rows_;
        PlanningAllreducePrecision precision_;
    };

    /** @brief One endpoint's completed event timing of the same coupled collective. */
    struct PlanningLocalTPEndpointObservation final
    {
        DeviceId device;
        size_t graph_nodes;
        double seconds_per_collective;
    };

    /** @brief Complete collective phase, retaining all endpoints instead of an optimistic mean. */
    struct PlanningLocalTPPhaseObservation final
    {
        int rows;
        size_t payload_bytes;
        std::vector<PlanningLocalTPEndpointObservation> endpoints;
        /** @return Slowest endpoint's completed latency; endpoints are not independent transfers. */
        double secondsPerCollective() const;
    };

    /** @brief Completed two-phase native observations with immutable request identity. */
    struct PlanningLocalTPObservations final
    {
        PlanningLocalTPRequest request;
        std::vector<PlanningLocalTPPhaseObservation> phases;
    };

    /** @brief Admit once, capture once per payload, observe briefly, retire before returning. */
    class PlanningLocalTPMeasurement final
    {
    public:
        /**
         * @brief Compose ordinary LocalTP scratch, FP32 payloads and retained graph-family BOM.
         * @param request Validated local group; no backend initialization occurs here.
         * @param host Same-rank CPU staging allocator.
         * @param devices Exact resources in the request's collective order.
         * @param builder Sole canonical aggregate builder; this method does not admit capacity.
         */
        static void contributeMemory(const PlanningLocalTPRequest &request,
            PhysicalMemoryResource host, std::span<const PhysicalMemoryResource> devices,
            PhysicalMemoryPlanBuilder &builder);
        /**
         * @brief Measure actual captured LocalTP on the normal owning GPU workers.
         * @param request Same immutable group/payload policy used for admission.
         * @param memory Rank-bound authority containing the complete contributed BOM.
         * @return Both complete phases after native graphs, payloads and collective borrows retire.
         * @throws std::exception on admission, preparation, capture or asynchronous failure.
         *
         * Invoke from the rank setup owner, not inside a GPU worker. All workers
         * are submitted before joining any one of them. Each phase uses one
         * warmup and three event-timed retained launches; there is no eager
         * collective, model warmup, alternative transport or stream-wide wait.
         */
        static PlanningLocalTPObservations measure(const PlanningLocalTPRequest &request,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory);
    };
}
