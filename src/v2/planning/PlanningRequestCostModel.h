/**
 * @file PlanningRequestCostModel.h
 * @brief Bounded measured request ranking shared by automatic plan and serve.
 *
 * Samples are collected once on discovery membership, never once per proposal.
 * The pure evaluator combines compiled main-forward operands, live state work
 * and topology-bound communication primitives. It is deliberately an estimate:
 * uniform routing, mean-context rooflines and conservative communication
 * composition are reported explicitly, not presented as model benchmarks.
 */
#pragma once
#include "AutomaticOrchestrationPlanner.h"
#include "PlanningWeightServiceModel.h"
#include "PlanningCommunicationCost.h"
#include <unordered_map>

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
        PlanningModelMetadata model_;
        ClusterInventory inventory_;
        PlanningWeightServiceModel weights_;
        PlanningKernelServiceCatalog arithmetic_;
        PlanningCommunicationCost communication_;
        std::unordered_map<std::string, double> source_bytes_per_element_;
    };
}
