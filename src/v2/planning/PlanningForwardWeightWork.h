/**
 * @file PlanningForwardWeightWork.h
 * @brief Phase-specific main-forward operands from compiled participant ownership.
 *
 * Physical residency is not execution work: retained MTP predictors, expert
 * replicas and alternate decode weights must not all be charged to every row.
 * This metadata-only projection keeps ordinary projections, embedding lookups,
 * non-projection parameters and complete routed experts distinct. It is an operand
 * inventory, not an execution DAG, allocation ledger or measured latency.
 */
#pragma once
#include "planning/WeightShardGeometry.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace llaminar2
{
    class AdmittedOrchestrationCandidate;
    class PlanningModelMetadata;

    /** @brief Main-model invocation; retained sidecar capacity is not another phase. */
    enum class PlanningMainForwardPhase { Prefill, Decode };

    /** @brief Semantic work family, shared by operand compilation and representative sampling. */
    enum class PlanningOrdinaryWeightKind { Projection, Router, Embedding, NonProjection, Unclassified };
    /** @return The one main-forward classification of a semantic role; unknown roles stay explicit. */
    PlanningOrdinaryWeightKind planningOrdinaryWeightKind(WeightRole role) noexcept;

    /** @brief A source operand and its canonical prepared scalar representation. */
    struct PlanningWeightOperand
    {
        std::string source_name;
        WeightRole role;
        int layer;
        std::string source_format;
        ModelPreparedWeightRepresentation representation;
        WeightShardGeometry geometry;

        /** @return Actual runtime scalar/codebook label, not an assumed source format. */
        std::string_view executionFormat() const noexcept
        {
            return representation == ModelPreparedWeightRepresentation::FP32
                ? std::string_view{"F32"} : std::string_view{source_format};
        }
    };

    /** @brief One ordinary projection; actual fused scheduling remains graph-owned. */
    struct PlanningProjectionWeight { PlanningWeightOperand weight; };
    /** @brief Router projection/top-k service, not the ordinary GEMM kernel family. */
    struct PlanningRouterWeight { PlanningWeightOperand weight; };
    /** @brief Indexed embedding access, never vocabulary-sized GEMM work per token. */
    struct PlanningEmbeddingWeight { PlanningWeightOperand weight; };
    /** @brief Norm, gate, bias or recurrent parameter; no GEMM shape is implied. */
    struct PlanningNonProjectionWeight { PlanningWeightOperand weight; };
    /** @brief Explicitly unresolved semantics; a cost consumer must not price this as zero. */
    struct PlanningUnclassifiedWeight { PlanningWeightOperand weight; };
    using PlanningOrdinaryWeightWork = std::variant<PlanningProjectionWeight, PlanningRouterWeight,
        PlanningEmbeddingWeight, PlanningNonProjectionWeight, PlanningUnclassifiedWeight>;

    /** @brief How a participant issues complete expert FFNs in this graph phase. */
    enum class PlanningExpertExecution
    {
        OwnedExperts, ///< One owner executes each selected expert in its admitted quota.
        ReplicatedExperts, ///< Every replica executes the same selected experts; no division by TP degree.
        BalancedReplicaAssignment, ///< Prefill assigns whole experts among eligible complete replicas.
    };

    /**
     * @brief Explicit uniform-routing prediction, not observed traffic or allocated storage.
     *
     * Tokens independently choose K distinct experts uniformly from E. Complete
     * replicas are charged independently; an assigned-replica phase additionally
     * assumes balanced expert assignment, not measured LLEP/histogram behavior.
     * Expectations may be fractional even though each real invocation is integral.
     */
    struct PlanningExpertWorkExpectation
    {
        double routed_rows;
        double nonempty_experts;
    };

    /** @brief Validated phase-specific ownership, separate from physical copy accounting. */
    class PlanningExpertExecutionShare final
    {
    public:
        /**
         * @brief Bind one tier's expert population to its actual execution semantics.
         * @param execution Resolved graph-phase policy; never inferred from device count.
         * @param experts Population eligible on this participant, possibly zero.
         * @param assignment_participants Assignment degree only for BalancedReplicaAssignment;
         *        other policies require one because their eligible population is already local.
         */
        PlanningExpertExecutionShare(PlanningExpertExecution execution, int experts, int assignment_participants = 1);
        /** @return Graph-phase semantics, including whether balance is only an assumption. */
        PlanningExpertExecution execution() const noexcept { return execution_; }
        /** @return Eligible complete experts, not tensor elements or physical byte capacity. */
        int experts() const noexcept { return experts_; }
        /** @return Explicit assignment divisor; replicated decode always returns one. */
        int assignmentParticipants() const noexcept { return assignment_participants_; }
        /**
         * @brief Expected complete expert rows/calls under a declared uniform top-k model.
         * @param model_experts Positive total router population E.
         * @param top_k Distinct experts selected by each token, in [1,E].
         * @param token_rows Logical rows in one routing batch, including zero for no work.
         * @return E[rows] and E[nonempty experts], with stable small-probability arithmetic.
         *
         * The call count is not K times T: repeated selections of one expert form
         * one grouped invocation. No histogram, migration benefit, independent
         * expert selections within a token, or MTP acceptance is invented here.
         */
        PlanningExpertWorkExpectation uniformExpectation(int model_experts, int top_k, size_t token_rows) const;
    private:
        PlanningExpertExecution execution_;
        int experts_, assignment_participants_;
    };

    /**
     * @brief One complete expert FFN shape at a routed layer.
     *
     * Each operand describes exactly one expert, independent of resident copies.
     * Routing multiplies invocation rows using a separately declared distribution;
     * capacity quotas are neither observed hotness nor per-token expert traffic.
     */
    struct PlanningRoutedExpertWeightWork
    {
        int layer;
        int expert_count;
        int routes_per_token;
        std::array<PlanningWeightOperand, 3> gate_up_down;
        std::vector<PlanningExpertExecutionShare> execution_shares; ///< Distinct logical tier bindings on this endpoint.

        /** @return Sum of this endpoint's disjoint tier populations under uniform routing. */
        PlanningExpertWorkExpectation uniformExpectation(size_t token_rows) const;
    };

    /** @brief Exact rank/device ownership and phase-selected model operands. */
    struct PlanningParticipantWeightWork
    {
        int execution_rank;
        int discovery_rank;
        DeviceId device;
        int first_layer;
        int last_layer;
        int activation_rows;
        std::vector<PlanningOrdinaryWeightWork> ordinary;
        std::vector<PlanningRoutedExpertWeightWork> routed;
    };

    /**
     * @brief Compile main-forward weight work without loading any tensor payload.
     * @param model Same immutable model descriptor used for candidate admission.
     * @param candidate Successfully admitted compiler output, not a raw topology string.
     * @param phase Select the installed prefill or serial-decode weight authority.
     * @return Participant-local operands, preserving both rank namespaces.
     * @throws std::exception for missing matrix geometry, incomplete expert triplets
     *         or inconsistent compiled ownership. Unknown roles remain explicit.
     *
     * Attention/KV/GDN arithmetic, collectives, MTP acceptance, execution order and
     * fused overlap are deliberately not invented here. A complete cost evaluator
     * must additionally price those graph operations and its routing assumptions.
     */
    std::vector<PlanningParticipantWeightWork> compilePlanningForwardWeightWork(
        const PlanningModelMetadata &model, const AdmittedOrchestrationCandidate &candidate,
        PlanningMainForwardPhase phase);
}
