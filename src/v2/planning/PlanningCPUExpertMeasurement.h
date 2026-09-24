/**
 * @file PlanningCPUExpertMeasurement.h
 * @brief Bounded GGUF-backed CPU expert observations through production service.
 *
 * One full gate/up/down triplet is loaded and prepared once, then the same
 * grouped CPU stage used by ExpertOverlay measures decode, prefill and verifier
 * service. No model runner, router, histogram, prefix state or synthetic model
 * inference is constructed. The observation retains exact source geometry,
 * formats and worker/ISA identity; it is not a generic CPU bandwidth number.
 */
#pragma once

#include "planning/PlanningModelMetadata.h"
#include "planning/PlanningExpertSample.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "planning/OrchestrationPerformanceEvidence.h"
#include "execution/moe/ExpertHistogramSource.h"
#include <array>
#include "backends/CPUExecutionGeometry.h"

namespace llaminar2
{
    enum class ISALevel : uint8_t;

    /** @brief Exact phase geometry and completed production-FFN timing. */
    struct PlanningCPUExpertPhaseObservation final
    {
        ExpertHistogramSource phase;
        int rows;
        PlanningServiceObservation service;
    };

    /**
     * @brief Immutable results of repeatedly executing one prepared source expert.
     *
     * Repeated weights may fit in cache. Consumers must not label the observed
     * rate as streaming memory bandwidth, remote-link bandwidth or a different
     * shape/format's service rate. Physical rank/node identity comes from the
     * authenticated inventory/publication enclosing this rank-local operation.
     */
    struct PlanningCPUExpertObservations final
    {
        DeviceId device;
        int worker_threads;
        ISALevel isa;
        ISALevel compiled_isa;
        size_t input_width;
        size_t intermediate_width;
        size_t prepared_bytes;
        std::array<std::string, 3> formats;
        PlanningExpertSampleRequest source;
        std::vector<PlanningCPUExpertPhaseObservation> phases;
    };

    /** @brief One admitted setup transaction; no private capacity or service cache. */
    class PlanningCPUExpertMeasurement final
    {
    public:
        /**
         * @brief Contribute all overlapping source/prepared/workspace storage.
         * @param source Retained root/local GGUF metadata authority.
         * @param request One complete expert triplet from a single logical layer.
         * @param resource Observed exact CPU allocator, never a fabricated capacity.
         * @param builder Canonical aggregate builder; this method does not admit it.
         * @param execution Rank-published CPU cache/ISA observation, never discovery-root CPUID.
         * @param workers Admitted physical-core workshare width for the observer.
         *
         * Three source and prepared matrices overlap. Reader and packer staging
         * are serial and therefore share their maximum, as do phase workspaces.
         */
        static void contributeMemory(const PlanningModelSource &source,
            const PlanningExpertSampleRequest &request, PhysicalMemoryResource resource,
            PhysicalMemoryPlanBuilder &builder, CPUExecutionGeometry execution, int workers);

        /** @brief Contribute the same BOM from published metadata; followers need no reader staging. */
        static void contributeMemory(const PlanningExpertSamplePlan &plan, PlanningSampleOrigin origin,
            PhysicalMemoryResource resource, PhysicalMemoryPlanBuilder &builder,
            CPUExecutionGeometry execution, int workers);

        /**
         * @brief Load, prepare, observe and retire one full native CPU expert.
         * @param source Same retained GGUF metadata used for the admitted BOM.
         * @param request Exact source expert identity.
         * @param device CPU allocator on the affined production setup thread.
         * @param memory Rank-bound authority containing the complete sample BOM.
         * @return Completed observations only; no source/prepared bytes escape.
         * @throws std::exception on geometry, source, admission or execution failure.
         *
         * The caller establishes the ordinary rank CPU/NUMA policy. This method
         * neither changes worker budgets nor starts another executor. Dynamic
         * OpenMP teams and nested invocation are rejected because they would
         * mislabel the measured worker geometry. No whole-model execution occurs.
         */
        static PlanningCPUExpertObservations measure(const PlanningModelSource &source,
            const PlanningExpertSampleRequest &request, DeviceId device,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory);

        /**
         * @brief Prepare and time an already admitted triplet without filesystem access.
         * @param sample Retains native source bytes and their PMA claims for this entire call.
         * @param device Affined CPU allocator and production worker geometry.
         * @param memory Same rank authority admitting preparation and execution workspace.
         * @return Completed observations with the sample's sealed source identity.
         */
        static PlanningCPUExpertObservations measure(const PlanningLoadedExpertSample &sample,
            DeviceId device, const std::shared_ptr<PhysicalMemoryAuthority> &memory);
    };
}
