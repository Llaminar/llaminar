/**
 * @file PrefixAuxiliaryStateIO.h
 * @brief Exact native model auxiliary sections at canonical prefix-cache I/O boundaries.
 *
 * Immutable archive coordinates determine all copy extents. Local rollback
 * checkpoints are deliberately excluded: their capacity includes unpublished
 * tail slots and must never become a RAM/SSD communication extent.
 */
#pragma once
#include "PrefixStorageBackend.h"
#include "../../kernels/IKVCache.h"
namespace llaminar2
{
/** @brief Explicit main versus shifted predictor archive ownership. */
enum class PrefixAuxiliaryLane { Main, ShiftedMTP };
/** @brief Export only completed pools and an explicitly selected endpoint.
 * @param cache Canonical cache and attached model state owner.
 * @param sequence Exact admitted request slot.
 * @param lane Main or shifted predictor logical coordinates.
 * @param handle Retained host/device archive with its immutable exact geometry.
 * @param stream Exact GPU producer stream, or null for CPU.
 */
void exportPrefixAuxiliaryState(const IKVCache &cache,int sequence,PrefixAuxiliaryLane lane,
    PrefixBlockHandle &handle,void *stream);
/** @brief Import compact archive bytes before publishing the restored canonical frontier.
 * @param cache Canonical cache retaining the model auxiliary state owner.
 * @param sequence Exact admitted request slot.
 * @param lane Main or shifted predictor logical coordinates.
 * @param handle Ready archive whose immutable key authenticates all token extents.
 * @param stream Exact ordered GPU consumer stream, or null for CPU.
 */
void importPrefixAuxiliaryState(IKVCache &cache,int sequence,PrefixAuxiliaryLane lane,
    const PrefixBlockHandle &handle,void *stream);
}
