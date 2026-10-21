/**
 * @file QwenSparseSnapshotMemory.h
 * @brief Selected main-QSA native state and authored metadata in the existing snapshot BOM.
 *
 * Live reference pool rows cannot price a captured full-capacity state slot.
 * This pure projection uses the same source shapes and layout as real stage
 * descriptors. Only PMA owns reservation, materialization and release.
 */
#pragma once
#include "GraphSnapshotMemoryCapacity.h"
#include "planning/ModelMemoryProfile.h"
#include "snapshots/Qwen4ExpSnapshotGeometry.h"
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include <algorithm>
#include <limits>
#include <span>

namespace llaminar2
{
/** @brief Disjoint additional components, without repricing ordinary token payloads. */
struct QwenSparseSnapshotMemory
{
    std::size_t native_state_payload_bytes = 0;
    std::size_t authored_metadata_bytes = 0;
    /** @return Checked total contribution to the single GraphSnapshotArena owner. */
    std::size_t totalBytes() const { return snapshot_storage_detail::add(native_state_payload_bytes, authored_metadata_bytes); }
};
/** @brief Checked owner/request product; no row count or pool frontier is read. */
inline std::size_t qwenSparseSnapshotMultiply(std::size_t a, std::size_t b)
{
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("QSA native snapshot owner extent overflows size_t");
    return a * b;
}
/**
 * @brief Resolve exact native capacities and bounded slot overhead for selected main producers.
 * @param model Authenticated metadata, including explicit main sparse layer identities.
 * @param mainLayers Exact main decoder layer count; a predictor is not a sparse main layer.
 * @param first First resident main layer, inclusive.
 * @param last Last resident main layer, inclusive.
 * @param context Native admitted per-request context capacity.
 * @param requests Actual request capacity; tokenRows is per request, not flattened twice.
 * @param tokenRows Maximum physical token rows in an admitted ordinary graph.
 * @param owners Existing retained forward arena inventory, never a token-derived multiplier.
 * @param selected Frozen source-output selection authenticated against reference headers.
 * @return Native pooled payload plus source-authored metadata/slot padding exactly once.
 *
 * Token query/selection payload remains in the caller's reference envelope.
 * Each selected pool's live NPY payload was excluded there and is replaced
 * here by its native context-capacity bank. A semantic alias is one selected
 * output; different producer occurrences/owners remain distinct. Dense0 has no
 * sparse main source identity, and CPU callers never enter this GPU projection.
 */
inline QwenSparseSnapshotMemory qwenSparseSnapshotMemory(
    const Qwen4ExpMemoryGeometry &model, int mainLayers, int first, int last, int context, int requests,
    int tokenRows, std::size_t owners, std::span<const GraphSnapshotSourceOutput> selected)
{
    QwenSparseSnapshotMemory result;
    if (selected.empty()) return result;
    if (mainLayers <= 0 || first < 0 || last < first ||
        last >= mainLayers || context <= 0 || requests <= 0 || tokenRows <= 0 || !owners)
        throw std::invalid_argument("QSA snapshots require exact sparse-main geometry and retained arena owners");
    if (model.indexer_head_dimension != qwen4exp::QwenSparseIndexDim ||
        model.indexer_query_heads != qwen4exp::QwenSparseIndexHeads ||
        model.compression_ratio != qwen4exp::QwenSparsePool ||
        model.maximum_selected_tokens != qwen4exp::QwenSparseSelected)
        throw std::invalid_argument("QSA snapshots disagree with authenticated native indexer geometry");
    const auto instances = qwenSparseSnapshotMultiply(static_cast<std::size_t>(requests), owners);
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const auto &source = selected[index];
        if (std::find(selected.begin(), selected.begin() + index, source) != selected.begin() + index)
            throw std::invalid_argument("QSA snapshot source selection contains a duplicate producer");
        if (source.layer() >= mainLayers ||
            std::find(model.sparse_decoder_layers.begin(), model.sparse_decoder_layers.end(),
                static_cast<unsigned>(source.layer())) == model.sparse_decoder_layers.end())
            throw std::invalid_argument("QSA snapshot source is absent from the admitted sparse-main manifest");
        if (source.layer() < first || source.layer() > last) continue;
        const auto role = source.role() == GraphSnapshotSourceRole::QwenSparseQuery ? Qwen4ExpSnapshotOutput::PreparedQuery :
            source.role() == GraphSnapshotSourceRole::QwenSparsePooledKeys ? Qwen4ExpSnapshotOutput::PooledKeys :
            Qwen4ExpSnapshotOutput::SelectedIndices;
        const auto shape = qwen4ExpSparseSnapshotShape(role, tokenRows, context, model.maximum_selected_tokens);
        const auto layout = snapshotStorageLayout(shape.storageShape());
        if (source.replacesReferencePayload())
            result.native_state_payload_bytes = snapshot_storage_detail::add(result.native_state_payload_bytes,
                qwenSparseSnapshotMultiply(layout.payload_storage_bytes, instances));
        // Relative payload-to-header padding and absolute slot alignment are
        // both physical bytes. The bound is per actual selected descriptor;
        // it does not spend unrelated payload slack on missing headers.
        result.authored_metadata_bytes = snapshot_storage_detail::add(result.authored_metadata_bytes,
            qwenSparseSnapshotMultiply(qwen4ExpSnapshotMetadataBytes(shape), instances));
    }
    (void)result.totalBytes();
    return result;
}
} // namespace llaminar2
