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

namespace llaminar2
{
    struct RankInventory;
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
         * @brief Measure missing priced CPU/GPU class/phase coordinates from the initial bank.
         * @param registry Local CPU/GPU residency endpoints in canonical ID order.
         * @param epoch Exact published startup epoch; no candidate bank is legal.
         * @param catalog Immutable exact gate/up/down format-equivalence classes.
         * @param topology Phases actually priced by the retained serving policy.
         * @param geometry Full expert FFN dimensions.
         * @param memory Sole rank-local physical admission/claim authority.
         * @param model_profile Original source workspace metadata, mandatory for GPU participants.
         * @param rank_inventory Original same-rank physical SM/CU observations, mandatory for GPU participants.
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
            const RankInventory *rank_inventory = nullptr);
    };
}
