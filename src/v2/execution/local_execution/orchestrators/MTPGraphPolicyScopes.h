/**
 * @file MTPGraphPolicyScopes.h
 * @brief Internal RAII declarations for paired MTP builder/execution policies.
 *
 * Serving-family preparation, workspace declaration and live forwarding use
 * these same guards. The guards retain no graph or execution data and restore
 * the existing policy at their original boundaries, including failed setup.
 */
#pragma once

#include "execution/mtp/MTPVerifierOutcomeGraph.h"
#include <vector>

namespace llaminar2
{
class IGraphBuilder;
namespace dgo_detail
{
/**
 * @brief Install the declarative live-MTP-condition graph policy for one scope.
 *
 * Runtime replay, setup-only executable materialization, and workspace
 * family declaration must all select the same GDN/logits namespace for
 * a device-resident MTP condition.  Centralizing the paired builder and
 * orchestrator flags here prevents one lifecycle path from restoring
 * only half of that policy after graph construction.
 */
class ScopedLiveMTPRequestBatchConditionPolicy
{
public:
    /**
     * @param builder Declarative graph builder receiving the policy.
     * @param orchestrator_flag Matching execution-host policy flag.
     * @param enable Whether this transaction is a live MTP condition.
     */
    ScopedLiveMTPRequestBatchConditionPolicy(
        IGraphBuilder *builder,
        bool *orchestrator_flag,
        bool enable);

    /** @brief Restore the ordinary condition policy on scope exit. */
    ~ScopedLiveMTPRequestBatchConditionPolicy();

    ScopedLiveMTPRequestBatchConditionPolicy(
        const ScopedLiveMTPRequestBatchConditionPolicy &) = delete;
    ScopedLiveMTPRequestBatchConditionPolicy &operator=(
        const ScopedLiveMTPRequestBatchConditionPolicy &) = delete;

    /// Return whether the requested policy was installed completely.
    bool ready() const noexcept { return ready_; }

private:
    IGraphBuilder *builder_ = nullptr;
    bool *orchestrator_flag_ = nullptr;
    bool active_ = false;
    bool ready_ = false;
};

/**
 * @brief Install one immutable grouped-verifier graph topology.
 *
 * Serving-family setup must capture the same declarative policy that a
 * live verifier later selects, while leaving request data untouched.
 * This scope changes the builder and execution-host mirrors together and
 * restores both on every exit.  Partial installation is rolled back as
 * well, so a failed setup cannot leak verifier topology into the next
 * family member.
 */
class ScopedGroupedMTPVerifierGraphPolicy
{
public:
    /**
     * @param builder Declarative model graph receiving the policy.
     * @param compute_all_position_logits Matching execution-host flag.
     * @param compute_row_indexed_logits Matching row-selection flag.
     * @param row_indexed_logit_count Matching physical output-row count.
     * @param outcome_mode Matching terminal verifier topology.
     * @param physical_rows Immutable captured verifier width.
     * @param requested_outcome_mode Greedy or external stochastic outcome.
     */
    ScopedGroupedMTPVerifierGraphPolicy(
        IGraphBuilder *builder,
        bool *compute_all_position_logits,
        bool *compute_row_indexed_logits,
        int *row_indexed_logit_count,
        MTPVerifierOutcomeGraphMode *outcome_mode,
        int physical_rows,
        MTPVerifierOutcomeGraphMode requested_outcome_mode);

    /** @brief Restore the ordinary forward-graph policy. */
    ~ScopedGroupedMTPVerifierGraphPolicy();

    ScopedGroupedMTPVerifierGraphPolicy(
        const ScopedGroupedMTPVerifierGraphPolicy &) = delete;
    ScopedGroupedMTPVerifierGraphPolicy &operator=(
        const ScopedGroupedMTPVerifierGraphPolicy &) = delete;

    /** @return True only when builder and host mirrors agree. */
    [[nodiscard]] bool ready() const noexcept { return ready_; }

private:
    IGraphBuilder *builder_ = nullptr; ///< Declarative topology owner.
    bool *compute_all_position_logits_ = nullptr; ///< Host policy mirror.
    bool *compute_row_indexed_logits_ = nullptr; ///< Host row-policy mirror.
    int *row_indexed_logit_count_ = nullptr; ///< Captured physical rows.
    MTPVerifierOutcomeGraphMode *outcome_mode_ = nullptr; ///< Terminal topology.
    std::vector<int> selected_rows_; ///< Immutable identity row ordering.
    bool touched_builder_ = false; ///< Whether rollback is required.
    bool ready_ = false; ///< Complete paired-policy installation.
};


} // namespace dgo_detail
} // namespace llaminar2
