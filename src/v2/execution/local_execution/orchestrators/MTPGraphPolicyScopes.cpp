/**
 * @file MTPGraphPolicyScopes.cpp
 * @brief Single implementation of the paired MTP graph-policy lifecycle.
 *
 * Both physical orchestrator translation units consume these definitions.
 * Installation, partial-failure rollback and destruction preserve the original
 * builder calls and host topology flags; no request or device state is copied.
 */
#include "MTPGraphPolicyScopes.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include <numeric>

namespace llaminar2::dgo_detail
{
ScopedLiveMTPRequestBatchConditionPolicy::ScopedLiveMTPRequestBatchConditionPolicy(
    IGraphBuilder *builder,
    bool *orchestrator_flag,
    bool enable)
    : builder_(builder),
      orchestrator_flag_(orchestrator_flag),
      active_(enable)
{
    if (!active_)
    {
        ready_ = true;
        return;
    }
    ready_ = builder_ && orchestrator_flag_ &&
             builder_->setLiveMTPRequestBatchCondition(true);
    if (ready_)
        *orchestrator_flag_ = true;
}

ScopedLiveMTPRequestBatchConditionPolicy::~ScopedLiveMTPRequestBatchConditionPolicy()
{
    if (!active_ || !ready_)
        return;
    (void)builder_->setLiveMTPRequestBatchCondition(false);
    *orchestrator_flag_ = false;
}

ScopedGroupedMTPVerifierGraphPolicy::ScopedGroupedMTPVerifierGraphPolicy(
    IGraphBuilder *builder,
    bool *compute_all_position_logits,
    bool *compute_row_indexed_logits,
    int *row_indexed_logit_count,
    MTPVerifierOutcomeGraphMode *outcome_mode,
    int physical_rows,
    MTPVerifierOutcomeGraphMode requested_outcome_mode)
    : builder_(builder),
      compute_all_position_logits_(
          compute_all_position_logits),
      compute_row_indexed_logits_(
          compute_row_indexed_logits),
      row_indexed_logit_count_(row_indexed_logit_count),
      outcome_mode_(outcome_mode)
{
    if (!builder_ || !compute_all_position_logits_ ||
        !compute_row_indexed_logits_ ||
        !row_indexed_logit_count_ || !outcome_mode_ ||
        physical_rows <= 0 ||
        (*outcome_mode_ !=
             MTPVerifierOutcomeGraphMode::Disabled) ||
        *compute_all_position_logits_ ||
        *compute_row_indexed_logits_ ||
        *row_indexed_logit_count_ != 0)
    {
        return;
    }

    selected_rows_.resize(
        static_cast<std::size_t>(physical_rows));
    std::iota(
        selected_rows_.begin(),
        selected_rows_.end(),
        0);

    touched_builder_ = true;
    ready_ =
        builder_->setComputeAllPositionLogits(true) &&
        builder_->setComputeRowIndexedAllPositionLogits(
            true, physical_rows) &&
        builder_->setRowIndexedAllPositionLogitRows(
            selected_rows_) &&
        builder_->setGroupedMTPVerifier(true) &&
        builder_->setMTPVerifierOutcomeGraphMode(
            requested_outcome_mode);
    if (!ready_)
        return;

    *compute_all_position_logits_ = true;
    *compute_row_indexed_logits_ = true;
    *row_indexed_logit_count_ = physical_rows;
    *outcome_mode_ = requested_outcome_mode;
}

ScopedGroupedMTPVerifierGraphPolicy::~ScopedGroupedMTPVerifierGraphPolicy()
{
    if (touched_builder_ && builder_)
    {
        (void)builder_->setMTPVerifierOutcomeGraphMode(
            MTPVerifierOutcomeGraphMode::Disabled);
        (void)builder_->setGroupedMTPVerifier(false);
        (void)builder_->setRowIndexedAllPositionLogitRows({});
        (void)builder_->setComputeRowIndexedAllPositionLogits(
            false, 0);
        (void)builder_->setComputeAllPositionLogits(false);
    }
    if (compute_all_position_logits_)
        *compute_all_position_logits_ = false;
    if (compute_row_indexed_logits_)
        *compute_row_indexed_logits_ = false;
    if (row_indexed_logit_count_)
        *row_indexed_logit_count_ = 0;
    if (outcome_mode_)
    {
        *outcome_mode_ =
            MTPVerifierOutcomeGraphMode::Disabled;
    }
}
} // namespace llaminar2::dgo_detail
