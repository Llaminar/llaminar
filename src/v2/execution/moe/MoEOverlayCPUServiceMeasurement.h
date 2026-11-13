/**
 * @file MoEOverlayCPUServiceMeasurement.h
 * @brief Existing prepared-expert service producer for startup economy readiness.
 *
 * Cold resident experts need not receive any user route. This setup-only producer
 * measures an exact resident FFN without a router, histogram, KV cache, sampler
 * or placement writer. Its immutable observations supplement absent service
 * classes in the existing certifier; they never become live inference totals.
 */
#pragma once

#include "MoEOverlayEconomyCalibrationPlanner.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "memory/CPUStorageRoot.h"
#include "planning/ModelMemoryProfile.h"
#include "MoEOverlayRouteTarget.h"
#include "MoEOverlayPreparedTargetService.h"
#include <optional>
#include <span>

namespace llaminar2
{
    struct RankInventory;
    class IGlobalTPContext;
    class ILocalTPContext;
    /**
     * @brief Setup producer on the same rank-affined OpenMP execution as CPU serving.
     *
     * All calls finish before maintenance and request admission. One bank lease
     * spans each observation, and private floating execution bindings prevent
     * workspace mutation in already prepared serving engines. The common missing-
     * class producer delegates GPU execution to the existing captured planning
     * measurement owner; GPU timings are device-produced, never CPU proxies.
     */
    class MoEOverlayCPUServiceMeasurement final
    {
    public:
        static constexpr int kMeasuredSamples = 3; ///< Bounded completed FFN observations.
        static constexpr int kMaximumRows = 8; ///< Largest retained service-sample input family.
        /** @brief Complete model dimensions for a singleton full-expert observation. */
        struct Geometry
        {
            int d_model = 0;
            int intermediate = 0;
        };

        /**
         * @brief Maximum concurrently allocated setup payload on one CPU resource.
         * @param geometry Full expert dimensions, not a tensor-parallel shard.
         * @param invocation_workspace_bytes Canonical merged kernel requirements for this sample family.
         * @return Exact maximum of the serial phase scratch families.
         * @throws std::invalid_argument for nonpositive dimensions.
         */
        [[nodiscard]] static size_t allocationBytes(Geometry geometry, size_t invocation_workspace_bytes);

        /** @brief Actual FP32 roots shared by GPU admission and materialization. */
        struct GPUStorageLayout
        {
            CPUStorageRoot<float> input, indices, weights, output;
            /** @return Complete physical host-root bytes, including allocator rounding. */
            size_t hostBytes() const;
            /** @return Exact GPU logical payload bytes for the four original tensor copies. */
            size_t deviceBytes() const;
        };
        /**
         * @brief Declare the actual producer's GPU probe tensors, with no materialization.
         * @param geometry Full original FFN dimensions.
         * @param rows Positive declared phase extent bounded by the existing family.
         * @return Four CPU0 storage roots used unchanged by the actual measurement.
         * @throws std::exception For invalid geometry or overflow.
         */
        [[nodiscard]] static GPUStorageLayout gpuStorageLayout(Geometry geometry, int rows);

        /** @brief Serial GPU probe's complete existing host/device tensor and invocation envelope. */
        struct GPUAllocation
        {
            size_t host_bytes = 0; ///< Four actual CPUStorageRoot FP32 input/output/route banks.
            size_t device_bytes = 0; ///< Four GPU tensor copies plus original stage workspace.
        };
        /**
         * @brief Declare one serial GPU prepared-service scratch family before admission.
         * @param geometry Complete original expert dimensions.
         * @param invocation_workspace_bytes Shared estimator's merged source-format requirements.
         * @return Exact host-root and GPU-copy/workspace byte contributions.
         * @throws std::exception For invalid geometry or checked arithmetic overflow.
         */
        [[nodiscard]] static GPUAllocation gpuAllocationBytes(Geometry geometry, size_t invocation_workspace_bytes);
        /** @return Exact bounded row count used for this observed production phase. */
        [[nodiscard]] static int rowsForPhase(ExpertHistogramSource source);

        /**
         * @brief Observe one privately owned, completely prepared CPU expert.
         * @param source Canonical gate/up/down engines retained for this call.
         * @param device Exact CPU allocator/worker, never a GPU proxy.
         * @param layer Logical layer for the production stage's diagnostic identity.
         * @param phase Production phase selecting its bounded representative batch.
         * @param geometry Full, unsharded expert dimensions.
         * @param memory Authority admitting the existing grouped workspace and tensors.
         * @return Sum of completed FFN nanoseconds across kMeasuredSamples, after warmup.
         * @throws std::exception for invalid geometry, ownership, admission or execution.
         *
         * Used both by startup planning and by the published-bank certifier below.
         * This does not publish a bank or alter routing. The original engines are
         * never rebound; private execution views use the same production stage,
         * arithmetic and explicit workspace as resident expert service.
         */
        [[nodiscard]] static uint64_t measurePrepared(
            const MoEOverlayPreparedExpertPayload &source, DeviceId device, int layer,
            ExpertHistogramSource phase, Geometry geometry,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory);

        /**
         * @brief Explicit setup-only boundary, not serving dispatch-to-publication.
         *
         * Inputs and route metadata are ready before the rendezvous. The root
         * clock includes the original-domain start release, prepared fragment
         * FFNs, rooted gather/combine and canonical ordered-FMA output fold.
         * Serving dispatch, activation ingress and terminal device ingress are
         * excluded. Whole measurePrepared() has no collective release and must
         * not be silently mixed into a dispatch-inclusive price.
         */
        enum class PreparedComputeBoundary { InputReadyStartReleaseToCanonicalOutput };

        /** @brief Passive root-clock evidence; it grants no service/placement readiness. */
        struct PreparedComputeObservation
        {
            MoEOverlayRouteTarget target;
            int layer = -1, expert = -1;
            ExpertHistogramSource phase = ExpertHistogramSource::DecodeToken;
            PreparedComputeBoundary boundary = PreparedComputeBoundary::InputReadyStartReleaseToCanonicalOutput;
            uint64_t logical_rows_per_invocation = 0; ///< Logical rows once, never multiplied by members.
            uint64_t measured_invocations = 0;
            uint64_t total_root_wall_nanoseconds = 0; ///< Sum of root intervals, never peer-clock maxima.
        };

        /**
         * @brief Quote the same serial setup roots used by the original-cohort probe.
         * @param cohort Original checked full source/cuts/member metadata.
         * @param member Original domain member, not a CPU device ordinal.
         * @param phase Exact existing representative row family.
         * @param invocation_workspace_bytes Existing merged stage/kernel requirement.
         * @return Physical ExecutionWorkspace bytes, including root-only gather/fold storage.
         * @throws std::exception For invalid source/phase/member or overflow.
         */
        [[nodiscard]] static size_t cpuJointAllocationBytes(
            const std::shared_ptr<const MoEExpertShardCohort>& cohort, size_t member,
            ExpertHistogramSource phase, size_t invocation_workspace_bytes);

        /**
         * @brief Measure one actual prepared CPU cohort at the explicitly named boundary.
         * @param registry Original model/source-bound participant banks.
         * @param epoch Installed original initial epoch; no candidate or fabricated readiness.
         * @param participant Exact original local global participant ID.
         * @param layer Original global model layer.
         * @param expert Same original logical expert selected on every member.
         * @param phase Same production phase on every member.
         * @param original_domain SAME live admitted native CPU expert domain shared owner.
         * @param memory Sole authority admitting exact setup roots and invocation workspace.
         * @param diagnostic_output Optional exact root rows*H CPU span, empty on contributors; filled only after all timed work succeeds. It owns no storage and is excluded from timing/traffic.
         * @return Root-only completed evidence; contributors return nullopt after their final gather.
         * @throws std::exception For missing source/bank/domain/admission or failed computation.
         * @note All original members enter this explicit job in the same order. No
         * new communicator, executor, histogram or active service price is installed.
         * Original source and bank readers span every stage and collective.
         * After the native owner is authenticated, any rank-local rejection
         * requests that same domain's existing fatal abort; callers cannot
         * continue a partially prepared collective job. A foreign native-owner
         * alias is rejected without entering or aborting that foreign domain.
         */
        [[nodiscard]] static std::optional<PreparedComputeObservation> measurePreparedCPUJoint(
            const MoEOverlayParticipantResidencyRegistry& registry, uint64_t epoch,
            int participant, int layer, int expert, ExpertHistogramSource phase,
            const std::shared_ptr<IGlobalTPContext>& original_domain,
            const std::shared_ptr<PhysicalMemoryAuthority>& memory,
            std::span<float> diagnostic_output = {});

        /**
         * @brief Execute deterministic original prepared-only target/class/phase jobs.
         * @param registry Original model-bound banks. @param plan Checked original declarations.
         * @param geometry Full original H/F. @param original_domain Existing CPU expert owner, null only on nonmembers.
         * @param memory Original PMA. @param model_profile Original workspace source for GPU targets.
         * @param rank_inventory Same rank's actual backend geometry for GPU targets.
         * @param owner_domain Original local projection communicator for owner policy.
         * @param owner_top_k Actual model routed width for that policy.
         * @return Only actual locally rooted samples; cohort contributors emit no rows.
         * @note This does not import live packet timing or install a service price.
         */
        static std::vector<MoEOverlayPreparedTargetServiceTotals> measurePreparedTargets(
            const MoEOverlayParticipantResidencyRegistry& registry,
            const MoEOverlayPreparedTargetServicePlan& plan,Geometry geometry,
            const std::shared_ptr<IGlobalTPContext>& original_domain,
            const std::shared_ptr<PhysicalMemoryAuthority>& memory,
            const ModelMemoryProfile* model_profile=nullptr,const RankInventory* rank_inventory=nullptr,
            const ILocalTPContext* owner_domain=nullptr,int owner_top_k=0);

        /**
         * @brief Measure missing priced CPU/GPU class/phase coordinates from the initial bank.
         * @param registry Local CPU/GPU residency endpoints in canonical ID order.
         * @param epoch Exact published startup epoch; no candidate bank is legal.
         * @param catalog Immutable exact gate/up/down format-equivalence classes.
         * @param topology Phases actually priced by the retained serving policy.
         * @param geometry Full expert FFN dimensions.
         * @param memory Sole rank-local physical admission/claim authority.
         * @param model_profile Original source workspace metadata, mandatory for GPU participants.
         * @param rank_inventory Original same-rank physical SM/CU observations, mandatory for GPU participants.
         * @param owner_domain Original local projection communicator for owner policy.
         * @param owner_top_k Actual model routed width for that policy.
         * @return Complete local matrix with only genuine prepared observations populated.
         * @throws std::exception for a missing bank, priced class without resident weights, invalid source or failed FFN.
         *
         * Call on the production rank's affined setup thread, not an arbitrary
         * worker with a different OpenMP/NUMA policy. No new thread pool or CPU
         * worker budget is selected here. Live observations take precedence in
         * the existing certifier, and are never overwritten by these samples.
         */
        [[nodiscard]] static std::vector<MoEOverlayParticipantLayerServiceTotals> measure(
            const MoEOverlayParticipantResidencyRegistry &registry,
            uint64_t epoch,
            const MoEOverlayEconomyCalibrationLayerCatalog &catalog,
            const ExpertHistogramProductionTopology &topology,
            Geometry geometry,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory,
            const ModelMemoryProfile *model_profile = nullptr,
            const RankInventory *rank_inventory = nullptr,
            const ILocalTPContext* owner_domain=nullptr,int owner_top_k=0);
    };
}
