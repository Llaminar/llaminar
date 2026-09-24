/**
 * @file AutomaticOrchestrationPlanner.h
 * @brief One streaming, cost-ranked selection over canonical admitted candidates.
 *
 * The planner owns selection, not hardware discovery, probing or physical
 * accounting. Its evaluator prices only complete admitted candidates for one
 * explicit workload. A returned value always retains the winning admission,
 * exact apply configuration and cost evidence. Enumeration order, capacity
 * and backend-specific core counts cannot substitute for a performance score.
 */
#pragma once

#include "planning/OrchestrationCandidateAdmission.h"
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

namespace llaminar2
{
    /**
     * @brief Explicitly estimated request latency, with its input-evidence description.
     *
     * Even when its inputs are measured kernel/link samples, this value is a
     * prediction, not a measured model benchmark. Decode is the mean cost over
     * this workload's generation horizon, including its growing KV context.
     */
    class OrchestrationCostEstimate final
    {
    public:
        /**
         * @brief Reject missing, nonfinite, nonpositive or overflowing costs.
         * @param workload Exact prompt and generation geometry used by the evaluator.
         * @param prefill_seconds Complete prompt latency, not seconds per token.
         * @param decode_seconds_per_token Mean generation latency over the stated horizon.
         * @param evidence Human-readable provenance of the evaluator's inputs and method.
         */
        OrchestrationCostEstimate(OrchestrationPlanningWorkload workload,
            double prefill_seconds, double decode_seconds_per_token, std::string evidence);
        /** @return Workload identity, which selection checks before comparing costs. */
        const OrchestrationPlanningWorkload &workload() const noexcept { return workload_; }
        /** @return Predicted prompt latency in seconds. */
        double prefillSeconds() const noexcept { return prefill_seconds_; }
        /** @return Predicted mean seconds per generated token. */
        double decodeSecondsPerToken() const noexcept { return decode_seconds_per_token_; }
        /** @return Complete request cost, already checked to be finite. */
        double requestSeconds() const noexcept;
        /** @return Input and method provenance, never a claim of actual model throughput. */
        const std::string &evidence() const noexcept { return evidence_; }
    private:
        OrchestrationPlanningWorkload workload_;
        double prefill_seconds_;
        double decode_seconds_per_token_;
        std::string evidence_;
    };

    /** @brief Counts of completed selection work, not a second model/capacity ledger. */
    struct OrchestrationSelectionCounts
    {
        size_t proposed = 0;
        size_t capacity_rejected = 0;
        size_t evaluated = 0;
    };

    /** @brief Successful automatic selection retains its exact winning physical admission. */
    class SelectedAutomaticOrchestration final
    {
    public:
        /** @return Winning compiled configuration, ranks and canonical physical certificate. */
        const AdmittedOrchestrationCandidate &candidate() const noexcept { return candidate_; }
        /** @return Estimated latency and provenance that caused this choice. */
        const OrchestrationCostEstimate &cost() const noexcept { return cost_; }
        /** @return Complete search accounting; no candidate is silently treated as a pass. */
        const OrchestrationSelectionCounts &counts() const noexcept { return counts_; }
    private:
        friend class AutomaticOrchestrationPlanner;
        /** @brief Only completed selection may construct a winning result. */
        SelectedAutomaticOrchestration(AdmittedOrchestrationCandidate candidate,
            OrchestrationCostEstimate cost, OrchestrationSelectionCounts counts);
        AdmittedOrchestrationCandidate candidate_;
        OrchestrationCostEstimate cost_;
        OrchestrationSelectionCounts counts_;
    };

    /** @brief Shared automatic-selection algorithm, independent of command and MPI lifecycle. */
    class AutomaticOrchestrationPlanner final
    {
    public:
        /** @brief Setup evaluator with immutable evidence; it must not launch model inference per proposal. */
        using Evaluate = std::function<OrchestrationCostEstimate(
            const AdmittedOrchestrationCandidate &, const OrchestrationPlanningWorkload &)>;
        /** @brief Streaming diagnostic for canonical physical-capacity rejection. */
        using CapacityRejected = std::function<void(OrchestrationStrategy, std::string_view)>;

        /**
         * @brief Minimize complete request latency across the existing candidate catalog.
         * @param request Valid automatic intent; the caller's configuration is not mutated.
         * @param source One retained metadata-only model owner for the complete search.
         * @param inventory Canonical discovery observation, not another hardware query.
         * @param memory Exact production capture/upload admission policy.
         * @param workload Explicit objective geometry; must fit the request context.
         * @param evaluate Required cost authority; missing/failed estimates are fatal.
         * @param rejected Optional observer of rejected physical admissions.
         * @return The fastest estimated admitted candidate and complete search counts.
         * @throws PhysicalMemoryCapacityExhausted if every proposed candidate exhausts capacity.
         * @throws std::invalid_argument for malformed intent, workload or cost evidence.
         *
         * Only typed physical exhaustion permits a rejected candidate. Compiler,
         * backend and evaluator defects propagate; they cannot silently turn into
         * a CPU/serial/different-topology execution. Hints break exact cost ties
         * only, then fewer distinct compute endpoints win (single-device before
         * multi-device); the complete serialized apply configuration breaks any
         * remaining tie independently of enumeration order. Endpoint count never
         * overrides a lower request cost or substitutes for measured link costs.
         * No candidate model is loaded.
         */
        static SelectedAutomaticOrchestration select(const OrchestrationConfig &request,
            const PlanningModelSource &source, const ClusterInventory &inventory,
            const OrchestrationCandidateMemoryPolicy &memory,
            const OrchestrationPlanningWorkload &workload, const Evaluate &evaluate,
            const CapacityRejected &rejected = {});
    };
}
