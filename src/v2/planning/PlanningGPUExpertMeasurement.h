/**
 * @file PlanningGPUExpertMeasurement.h
 * @brief Source-backed, captured GPU expert service observations for planning.
 *
 * The production weight pool and MoE stage own preparation and execution. Only
 * Startup source samplers observe only DecodeToken and PrefillChunk; live
 * measurePrepared per-phase MTP calibration remains independent. Only
 * completed timings and exact source/device identity escape the transaction;
 * no runner, live routing histogram, sampler, KV cache or topology is created.
 */
#pragma once
#include "PlanningModelMetadata.h"
#include "PlanningExpertSample.h"
#include "PhysicalMemoryAuthority.h"
#include "OrchestrationPerformanceEvidence.h"
#include "execution/moe/ExpertHistogramSource.h"
#include "execution/moe/MoEOverlayRouteTarget.h"
#include "collective/NativeLocalTPContextBorrow.h"
#include <span>

namespace llaminar2
{
    class MoEOverlayPreparedExpertPayload;
    class MoEOverlayFixedDownProjectionBank;
    class RetainedParticipantGraphExecutable;
    class ILocalTPContext;
    class MoEOverlayCPUServiceMeasurement;
    /** @brief Borrow actual immutable owner-domain facts through one calibration transaction. */
    struct PlanningPreparedOwnerDispatch
    {
        const ILocalTPContext* domain = nullptr; ///< Original physical communicator, never reconstructed membership.
        int top_k = 0; ///< Original routed width, distinct from the singleton sample route.
        std::span<const MoEOverlayPreparedExpertPayload> resident_sources; ///< Original complete local bank for source-policy resolution.
        bool has_live_rows = false; ///< Actual graph row policy, not the sample capacity.
    };
    /** @brief One complete captured FFN, including routing/activation work. */
    struct PlanningGPUExpertPhaseObservation final
    {
        ExpertHistogramSource phase;
        int rows;
        size_t graph_nodes;
        PlanningServiceObservation service;
    };

    /**
     * @brief Qualified repeated-expert measurements, not streaming bandwidth.
     *
     * Rank and physical-node identities belong to the enclosing authenticated
     * inventory. A faster measured link or device never changes those facts.
     */
    struct PlanningGPUExpertObservations final
    {
        DeviceId device;
        size_t input_width;
        size_t intermediate_width;
        size_t prepared_bytes;
        std::array<std::string, 3> formats;
        PlanningExpertSampleRequest source;
        std::vector<PlanningGPUExpertPhaseObservation> phases;
    };

    class MoEExpertShardCohort;
    class MoEOverlayParticipantResidencyRegistry;
    struct RankInventory;
    /**
     * @brief Original complete-cohort observations with deliberately separate units.
     *
     * Complete service is a root-native interval for the actual captured cohort.
     * Compute is the sum of concurrent member maxima, not summed member timings.
     * Communication is measured once per input-ready frame at its explicit H-row
     * extent, with empty control measured independently. None is implicitly
     * convertible to an expert activation rate or a migration transfer price.
     */
    struct PlanningGPUCohortServiceObservation {
        MoEOverlayRouteTarget target;
        int layer=-1,expert=-1;
        ExpertHistogramSource phase=ExpertHistogramSource::DecodeToken;
        uint64_t rows_per_frame=0,measured_frames=0;
        uint64_t complete_service_nanoseconds=0;
        uint64_t concurrent_compute_nanoseconds=0;
        uint64_t live_frame_communication_nanoseconds=0;
        uint64_t empty_frame_communication_nanoseconds=0;
        size_t participants=0;
    };

    /** @brief One bounded setup transaction on the exact device's normal worker. */
    class PlanningGPUExpertMeasurement final
    {
    public:
        /**
         * @brief Reject unsupported projection arithmetic from metadata alone.
         * @param request Authenticated gate/up/down source tensor identities.
         * @param description Their resolved format and geometry, before payload I/O.
         * @throws std::invalid_argument For uncatalogued or incompatible formats.
         *
         * This uses the live GPU expert payload's canonical validation contract,
         * not a second planning capability table. CPU expert sampling remains a
         * separate native implementation and does not acquire GPU restrictions.
         */
        static void validateSourceArithmetic(const PlanningExpertSampleRequest &request,
            const PlanningExpertSampleDescription &description);

        /**
         * @brief Contribute source, native weights, staging, workspace and graphs.
         * @param source Retained metadata authority; this operation reads no payload.
         * @param request Full source triplet, shared with CPU sampling.
         * @param host Exact CPU staging allocator on the same rank as the GPU.
         * @param gpu Exact observed GPU allocator, never an invented capacity.
         * @param builder Canonical aggregate builder; admission remains its caller's job.
         *
         * Invoke on the owning GPU worker: dispatch-specific scratch depends on
         * the actual local card. This performs no payload read or GPU allocation,
         * and must never be used to query a remote ordinal from the root rank.
         */
        static void contributeMemory(const PlanningModelSource &source,
            const PlanningExpertSampleRequest &request, PhysicalMemoryResource host,
            PhysicalMemoryResource gpu, PhysicalMemoryPlanBuilder &builder);

        /** @brief Same exact-worker BOM from published source identity; no remote file access. */
        static void contributeMemory(const PlanningExpertSamplePlan &plan, PlanningSampleOrigin origin,
            PhysicalMemoryResource host, PhysicalMemoryResource gpu, PhysicalMemoryPlanBuilder &builder);

        /**
         * @brief Acquire one original General owner for serial cold prepared-kernel definitions.
         * @param device Original prepared GPU.
         * @param memory Same admitted source authority whose one-slot BOM preceded admission.
         * @return Closed calibration-purpose owner; no serving graph can enter this purpose.
         * @throws std::exception For stale/missing original source or insufficient admission.
         */
        static std::shared_ptr<RetainedParticipantGraphExecutable> beginPreparedCalibration(
            DeviceId device,const std::shared_ptr<PhysicalMemoryAuthority>& memory);

        /**
         * @brief Capture/time one original resident expert without reading or repacking model weights.
         * @param payload Complete gate/up/down family borrowed under its original bank lease.
         * @param device Exact original prepared GPU, owned by the same admitted authority.
         * @param layer Original logical layer identity.
         * @param phase Original priced production phase, independent of row count.
         * @param width Full expert input/output width.
         * @param intermediate Full expert intermediate width.
         * @param memory Same original admitted PMA for scratch and General graph obligations.
         * @param declared_workspace_bytes Same original source estimator's admitted serial workspace envelope.
         * @param calibration Same original cold-only owner reused after each exact terminal/native reset.
         * @param fixed_down Original fixed column bank, required only for GateUp policy metadata; never executed.
         * @param source_expert Original expert ID selecting that bank's immutable descriptor.
         * @param owner_dispatch Actual production domain/operation/source facts for owner launch-policy resolution.
         * @return One completed GPU-event interval spanning three invocations, in nanoseconds, excluding warmup.
         * @throws std::exception For stale source, geometry, admission, capture or timing failure.
         * @note Caller runs off-worker; this method records on the original worker,
         * then joins and destroys native graph handles before releasing any bank/view storage.
         */
        static uint64_t measurePrepared(const MoEOverlayPreparedExpertPayload &payload,
            DeviceId device, int layer, ExpertHistogramSource phase, int width, int intermediate,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory, size_t declared_workspace_bytes,
            const std::shared_ptr<RetainedParticipantGraphExecutable> &calibration,
            std::shared_ptr<const MoEOverlayFixedDownProjectionBank> fixed_down = {}, int source_expert = -1,
            PlanningPreparedOwnerDispatch owner_dispatch = {});

    private:
        friend class MoEOverlayCPUServiceMeasurement;
        /**
         * @brief Observe one genuine complete original GPU cohort without serving mutation.
         * @param registry Same model/source-bound initial banks; all member readers span capture/replay.
         * @param epoch Original installed initial epoch.
         * @param cohort Same original source cohort retained by every member family.
         * @param layer Original global layer. @param expert Same original resident expert on every member.
         * @param phase Explicit original arithmetic phase, independent of capacity.
         * @param native Original once-per-domain native borrow; singleton is an identity.
         * @param memory Sole original PMA. @param profile Original complete source workspace metadata.
         * @param inventory Same rank's physical GPU compute geometry.
         * @param calibrations One same-source cold calibration owner per original ordered member.
         * @return Separate complete, concurrent compute and per-frame communication observations.
         * @throws std::exception For incomplete/foreign sources, domains, admission or native progress.
         * @note No serving runtime, histogram, placement bank or communicator is created or mutated.
         */
        static PlanningGPUCohortServiceObservation measurePreparedCohort(
            const MoEOverlayParticipantResidencyRegistry& registry,uint64_t epoch,
            const std::shared_ptr<const MoEExpertShardCohort>& cohort,int layer,int expert,
            ExpertHistogramSource phase,const NativeLocalTPContextBorrow& native,
            const std::shared_ptr<PhysicalMemoryAuthority>& memory,const ModelMemoryProfile& profile,
            const RankInventory& inventory,
            std::span<const std::shared_ptr<RetainedParticipantGraphExecutable>> calibrations);

    public:
        /**
         * @brief Prepare once and measure the retained decode and prefill graphs used by startup ranking.
         * @param source Same retained source used for admission.
         * @param request Exact triplet named in the sample BOM.
         * @param device CUDA or ROCm device whose worker invokes this operation.
         * @param memory Sole rank-bound authority for every physical allocation.
         * @return Completed observations after all sample-owned buffers retire.
         * @throws std::exception for unsupported geometry, admission, preparation or execution failure.
         *
         * Invoke through the ordinary GPU worker, before live inference. Timings
         * exclude I/O, preparation and capture. No eager execution is permitted.
         */
        static PlanningGPUExpertObservations measure(const PlanningModelSource &source,
            const PlanningExpertSampleRequest &request, DeviceId device,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory);

        /**
         * @brief Prepare and capture an admitted source triplet on its exact GPU worker.
         * @param sample Native bytes and sealed source provenance, independent of their transport.
         * @param device Exact owning CUDA/HIP worker; no remote ordinal lookup is allowed.
         * @param memory Same rank authority for prepared weights, staging, workspace and graphs.
         * @return Completed captured observations; no files or live runner are opened.
         */
        static PlanningGPUExpertObservations measure(const PlanningLoadedExpertSample &sample, DeviceId device,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory);
    };
}
