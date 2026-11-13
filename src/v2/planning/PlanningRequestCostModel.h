/**
 * @file PlanningRequestCostModel.h
 * @brief Bounded measured request ranking shared by automatic plan and serve.
 *
 * Samples are collected once on discovery membership, never once per proposal.
 * The pure evaluator combines compiled main-forward operands, live state work
 * and topology-bound communication primitives. It is deliberately an estimate:
 * uniform routing, mean-context rooflines and conservative communication
 * composition are reported explicitly, not presented as model benchmarks.
 * Like GPUs on one original rank/NUMA owner share median compute predictions
 * over their retained raw observations; physical admission and links stay exact.
 */
#pragma once
#include "AutomaticOrchestrationPlanner.h"
#include "PlanningWeightServiceModel.h"
#include "PlanningCommunicationCost.h"
#include <unordered_map>
#include <map>

namespace llaminar2
{
    class AutomaticPlanningPreparation;

    /** @brief Immutable measured basis; no device, graph, payload or allocation owner escapes setup. */
    class PlanningRequestCostModel final
    {
    public:
        /**
         * @brief Collect bounded compute/streaming/link evidence before rank selection.
         * @param context Shared plan/serve discovery transaction with root-owned GGUF source.
         * @return Complete root-only evaluator; followers return absence after collection.
         * @throws std::exception on incomplete evidence or any sampling failure.
         *
         * There is no candidate inference warmup, learned acceptance assumption,
         * hidden retry, or replacement backend. Preparation uses the existing
         * admitted samplers and failure-atomic publication transactions.
         */
        static std::optional<AutomaticOrchestrationPlanner::Evaluate> prepare(const AutomaticPlanningPreparation &context);

        /**
         * @brief Bind the model and one authenticated complete observation set.
         * @param model Exact immutable metadata used by candidate compilation.
         * @param inventory Physical discovery identity; measurements cannot redefine locality.
         * @param weights Source-family compute and independent streaming observations.
         * @param arithmetic Dedicated FP32 proxy, not a quantized model weight rate.
         * @param communication Completed topology-bound primitive observations.
         * @throws std::invalid_argument for mismatched observer/CPU workshare identity.
         */
        PlanningRequestCostModel(PlanningModelMetadata model, ClusterInventory inventory,
            PlanningWeightServiceModel weights, PlanningKernelServiceCatalog arithmetic,
            PlanningCommunicationCost communication);

        /**
         * @brief Price an admitted candidate for the stated complete request.
         * @param candidate Exact compiled membership, TP/PP geometry and expert quotas.
         * @param workload Positive prefill/generation objective inside admitted context.
         * @return Predicted phase times and explicit assumptions, never benchmark results.
         *
         * Distinct compute endpoints overlap at a layer's join; work on one
         * endpoint is additive. Layers and PP boundaries remain dependent.
         * Communication is charged separately without crediting unproved overlap.
         * Single-device plans have exactly zero interconnect demand. The model
         * never credits future migration or MTP acceptance without evidence.
         */
        OrchestrationCostEstimate evaluate(const AdmittedOrchestrationCandidate &candidate,
            const OrchestrationPlanningWorkload &workload) const;

    private:
        /** @brief Request-local estimate and protocol qualifications, never a persistent runtime ledger. */
        struct InvocationCost;
        /** @return One phase invocation using live rows/context and compiled ownership. */
        InvocationCost invocation(const AdmittedOrchestrationCandidate &candidate,
            PlanningMainForwardPhase phase, int rows, double context) const;
        /** @return Source-native or canonically promoted bytes for logical parameter reads only. */
        double parameterBytes(const PlanningWeightOperand &weight) const;
        /** @return Qualified FP32/streaming roofline for explicitly supplied non-weight work. */
        double scalarSeconds(int rank, DeviceId device, int rows, double operations, double bytes) const;
        /** @brief Evaluate identical local work against same-class measured services; CPU and unknown classes stay exact.
         * @param rank Original authenticated discovery owner, never execution rank.
         * @param device Actual admitted endpoint whose immutable cohort supplies compute evidence.
         * @param estimate Complete same-family/precision/geometry prediction at its authenticated original rank/device.
         * @return Median prediction, not a synthesized or aliased endpoint measurement.
         */
        double medianSeconds(int rank,DeviceId device,const std::function<double(int,DeviceId)> &estimate) const;
        /** @brief Build immutable like-GPU classes from actual inventory and authenticated streaming observers. */
        void prepareComputeCohorts();
        std::map<std::pair<int,DeviceId>,std::vector<DeviceId>> compute_cohorts_;
        std::string compute_cohort_evidence_;
        PlanningModelMetadata model_;
        ClusterInventory inventory_;
        PlanningWeightServiceModel weights_;
        PlanningKernelServiceCatalog arithmetic_;
        PlanningCommunicationCost communication_;
        std::unordered_map<std::string, double> source_bytes_per_element_;
    };
}
