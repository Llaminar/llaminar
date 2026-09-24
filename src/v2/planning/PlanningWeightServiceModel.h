/**
 * @file PlanningWeightServiceModel.h
 * @brief Bounded source-family service estimates from authenticated device observations.
 *
 * This setup-only model combines repeated native-kernel observations with an
 * independently measured streaming-memory limit. Its output is a weight-work
 * estimate, not a model benchmark or an allocation decision. Attention, KV/GDN,
 * communication, dependency scheduling and MTP acceptance remain separate; a
 * caller must not advertise this component as complete request latency.
 */
#pragma once
#include "PlanningForwardWeightWork.h"
#include "PlanningKernelServiceCatalog.h"

namespace llaminar2
{
    class AutomaticPlanningPreparation;

    /** @brief One source representative of an ordinary executed format or complete expert family. */
    using PlanningWeightServiceSample = std::variant<PlanningModelSampleRequest, PlanningExpertSampleRequest>;

    /**
     * @brief Immutable, reusable component estimates; no sampling occurs per candidate.
     *
     * Shape extrapolation is deliberately explicit: use effective measured
     * operations/second for the same executed format and operation family, then
     * apply the independent streaming-byte lower bound. This is bounded ranking,
     * not an exhaustive per-shape performance simulator. Endpoint aliasing and
     * CPU worker scope remain owned by the authenticated catalogs.
     */
    class PlanningWeightServiceModel final
    {
    public:
        /**
         * @brief Select the largest aggregate-arithmetic source shape per format/expert triplet.
         * @param model Root-published immutable GGUF directory, including main-layer boundary.
         * @return Stable ordered requests independent of tensor directory iteration order.
         *
         * Repeated layers do not create more samples. Ordinary matrices use at
         * most 1024 real source rows while preserving their complete K/codebook;
         * canonical loader promotions select the executed family without
         * manufacturing substitute source tensors or changing inference policy;
         * experts remain complete native triplets. Learned MTP sidecars are not
         * main-forward work. This reads no payload and estimates no capacity.
         */
        static std::vector<PlanningWeightServiceSample> samples(const PlanningModelMetadata &model);

        /**
         * @brief Gather bounded source and streaming observations once on discovery membership.
         * @param context Common plan/serve preparation transaction, before membership narrows.
         * @return Complete immutable evidence on root only; peers return absence.
         *
         * Uses the existing source publication, physical admission, worker and
         * failure-consensus owners. Root alone reads source bytes. The caller's
         * preparation completion consensus covers final model construction.
         */
        static std::optional<PlanningWeightServiceModel> collect(const AutomaticPlanningPreparation &context);

        /**
         * @brief Seal independently authenticated memory and source-family observations.
         * @param memory Source-free streaming catalog, not cached matrix service.
         * @param kernels Ordinary projection and complete-expert catalogs.
         * @throws std::invalid_argument for wrong/duplicate families or missing endpoints.
         */
        PlanningWeightServiceModel(PlanningKernelServiceCatalog memory,
            std::vector<PlanningKernelServiceCatalog> kernels);

        /**
         * @brief Estimate an ordinary projection using its actual prepared format and local N/K.
         * @param rank Discovery rank, not compact execution rank.
         * @param device Exact inventory endpoint visible to that rank.
         * @param weight Compiled source-native/FP32 representation and shard geometry.
         * @param rows Actual projection rows; terminal heads normally use one.
         * @return Estimated seconds including the weight streaming bound.
         * @throws std::exception for missing format/service or nonmatrix work.
         */
        double projectionSeconds(int rank, DeviceId device, const PlanningWeightOperand &weight, size_t rows) const;

        /**
         * @brief Estimate this endpoint's complete routed FFNs under explicit uniform top-k routing.
         * @param work Compiled exact quotas/replication policy; not aggregate physical capacity.
         * @param token_rows Routing rows, not rows per individual expert.
         * @return Estimated seconds; zero only for explicitly zero routed ownership/work.
         *
         * Charge expected nonempty groups, not K*T separate weight reads. No
         * future promotion benefit, observed hotness or MTP acceptance is invented.
         */
        double expertSeconds(int rank, DeviceId device,
            const PlanningRoutedExpertWeightWork &work, size_t token_rows) const;

        /** @return Streaming service time for explicitly supplied traffic, not allocation capacity. */
        double memorySeconds(int rank, DeviceId device, double traffic_bytes) const;

        /** @return Immutable streaming evidence for joining other compute families by physical observer. */
        const PlanningKernelServiceCatalog &streamingEvidence() const noexcept { return memory_; }

    private:
        /** @return Required format-specific projection catalog; absence is fatal, never another codebook. */
        const PlanningKernelServiceCatalog &projection(std::string_view format) const;
        /** @return Required complete-FFN format triplet; ordinary GEMM cannot substitute for it. */
        const PlanningKernelServiceCatalog &expert(const std::array<std::string, 3> &formats) const;
        PlanningKernelServiceCatalog memory_;
        std::vector<PlanningKernelServiceCatalog> kernels_;
    };
}
