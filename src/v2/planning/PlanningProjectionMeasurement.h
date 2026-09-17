/**
 * @file PlanningProjectionMeasurement.h
 * @brief Production-representation projection observations, independent of expert service.
 *
 * A sealed matrix selection preserves the source model format and TP shard.
 * The canonical loader contract alone selects any runtime FP32 promotion; it is
 * not a precision override. Source and converted storage have separate PMA
 * lifetimes. Preparation uses the production factory/pool, and both GPU phases are native
 * captured graphs. These are repeated-matrix kernel measurements, not full-model
 * latency, DRAM bandwidth, fused-FFN service or an assumed MTP acceptance rate.
 * A separately typed source-free FP32 probe uses that same execution lifecycle;
 * it is an arithmetic ranking proxy, never evidence for a model weight format.
 */
#pragma once
#include "PlanningMatrixSample.h"
#include "PlanningExpertSample.h"
#include "PlanningExecutionMeasurement.h"
#include "PhysicalMemoryAuthority.h"
#include "backends/CPUExecutionGeometry.h"
#include <optional>

namespace llaminar2
{
    enum class ISALevel : uint8_t;

    /** @brief Exact observing CPU workshare; absent for an accelerator observation. */
    struct PlanningProjectionCPUObservation
    {
        int workers;
        ISALevel isa, compiled_isa;
        CPUExecutionGeometry execution;
    };

    /** @brief One ordinary (not grouped-verifier) projection at its exact M. */
    struct PlanningProjectionPhaseObservation
    {
        int rows;
        size_t graph_nodes; ///< Zero on CPU, nonzero for every CUDA/HIP sample.
        PlanningServiceObservation service;
    };

    /** @brief Immutable completed observations; no allocations or execution handles escape. */
    struct PlanningProjectionObservations
    {
        PlanningMatrixSamplePlan source;
        DeviceId device;
        size_t prepared_bytes;
        std::optional<PlanningProjectionCPUObservation> cpu;
        std::vector<PlanningProjectionPhaseObservation> phases;
    };

    /**
     * @brief Bounded source-free FP32 arithmetic basis, independent of model quantization.
     *
     * A fixed four-MiB matrix supplies effective floating arithmetic service at
     * M=1 and M=64. This is neither a hardware peak nor measured attention/GDN
     * latency. No model, candidate count, or context length enlarges this probe.
     */
    struct PlanningFP32ArithmeticPlan final
    {
        static constexpr int kN = 1024;
        static constexpr int kK = 1024;
        static constexpr int kPrefillRows = 64;
    };

    /** @brief Floating arithmetic evidence cannot be relabeled as a GGUF projection. */
    struct PlanningFP32ArithmeticObservations
    {
        DeviceId device;
        size_t prepared_bytes;
        std::optional<PlanningProjectionCPUObservation> cpu;
        std::vector<PlanningProjectionPhaseObservation> phases;
    };

    /** @brief Bounded decode/prefill sampling through the ordinary projection ABI. */
    class PlanningProjectionMeasurement final
    {
    public:
        /**
         * @brief Contribute source, preparation, tensors, workspace and native graphs.
         * @param plan Exact source format/geometry published before device preparation.
         * @param origin Distinguishes GGUF-reader staging from an already received payload.
         * @param host Rank's CPU staging allocator.
         * @param execution Same-rank CPU or exact owning GPU allocator.
         * @param builder Canonical physical BOM, never a second capacity ledger.
         * @param prefill_rows Positive actual prefill sample M; decode always uses M=1.
         * @param cpu Rank-observed cache/ISA and worker geometry; required only on CPU.
         *
         * GPU calls run on their exact device worker. This declares storage but
         * reads no weights, allocates no payload and executes no model arithmetic.
         */
        static void contributeMemory(const PlanningMatrixSamplePlan &plan, PlanningSampleOrigin origin,
            PhysicalMemoryResource host, PhysicalMemoryResource execution, PhysicalMemoryPlanBuilder &builder,
            int prefill_rows, const std::optional<PlanningProjectionCPUObservation> &cpu = {});

        /**
         * @brief Prepare once, observe M=1 and prefill M, then retire private resources.
         * @param sample Loaded native source owner, retained for this whole transaction.
         * @param device Exact execution device; GPU callers must own its normal worker.
         * @param memory Rank-bound authority containing the contributed complete BOM.
         * @param prefill_rows Same positive physical M admitted by the contributor.
         * @return Completed kernel observations with source, worker and cache provenance.
         * @throws std::exception on admission, ownership, preparation or execution failure.
         *
         * CPU uses BackendNative ordinary projections, not GPUAlignedExpert.
         * GPU timing excludes I/O, preparation and capture and never executes eager
         * arithmetic. No source tensor, host topology or live runner is modified.
         */
        static PlanningProjectionObservations measure(const PlanningLoadedMatrixSample &sample,
            DeviceId device, const std::shared_ptr<PhysicalMemoryAuthority> &memory, int prefill_rows);

        /**
         * @brief Admit source-free FP32 operands through the same preparation/graph BOM.
         * @param host Rank CPU allocator for immutable probe initialization.
         * @param execution Same-rank observing CPU or exact owning GPU allocator.
         * @param builder Sole physical admission builder; no model source is charged.
         * @param cpu Exact CPU observer when execution is CPU, absent on GPU.
         */
        static void contributeMemory(PlanningFP32ArithmeticPlan, PhysicalMemoryResource host,
            PhysicalMemoryResource execution, PhysicalMemoryPlanBuilder &builder,
            const std::optional<PlanningProjectionCPUObservation> &cpu = {});

        /**
         * @brief Measure one fixed FP32 basis without reading or modifying model weights.
         * @param device Exact CPU/GPU endpoint, with the same ownership rules as source sampling.
         * @param memory Authority containing the complete source-free BOM.
         * @return M-specific effective arithmetic evidence after private resources retire.
         *
         * Initialization and preparation precede timing. CPU uses ordinary native
         * projections; GPU reuses the same exact-stream captured projection path.
         */
        static PlanningFP32ArithmeticObservations measure(PlanningFP32ArithmeticPlan,
            DeviceId device, const std::shared_ptr<PhysicalMemoryAuthority> &memory);
    };
}
