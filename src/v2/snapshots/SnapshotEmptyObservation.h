/**
 * @file SnapshotEmptyObservation.h
 * @brief Exact diagnostic presence/shape proof for an observed empty numerical state.
 *
 * An empty state has no cosine or error distribution. Its immutable publication
 * owner and exact mathematical columns distinguish it from missing evidence.
 */
#pragma once
#include "SnapshotInfo.h"
namespace llaminar2
{
/** @brief Authenticate an owned empty floating publication against reference columns.
 * @param observation Immutable terminal diagnostic view, never live model state.
 * @param columns Positive exact reference row width.
 * @return True only for a present, correctly shaped, zero-element numerical state.
 */
inline bool matchesFloatingEmptyObservation(const SnapshotInfo &observation,std::size_t columns) noexcept
{
    return columns>0&&observation.lifetime_owner&&observation.rows==0&&observation.cols==columns&&
        observation.size==0&&!observation.isInteger();
}
}
