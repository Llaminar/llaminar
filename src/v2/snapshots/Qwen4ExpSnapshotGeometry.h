/**
 * @file Qwen4ExpSnapshotGeometry.h
 * @brief Source-owned QFN authored output geometry for descriptors and admission.
 *
 * An entry denotes an actual producer output, not a semantic alias or storage
 * bank. Two observations of the same tensor at different producers are two
 * entries. This geometry reads no count, owns no storage and requests no memory.
 */
#pragma once
#include "SnapshotStorageLayout.h"
#include <algorithm>
#include <array>
#include <vector>

namespace llaminar2
{
/** @brief Closed authored-output roles emitted by QFN production stages. */
enum class Qwen4ExpSnapshotOutput { BranchNormalized, TerminalHiddenInput, PreparedQuery,
    SelectedCount, PooledKeys, SelectedIndices, RetainedRawKey, PredictorFC };
/** @brief One actual output's pointer-free source shape and typed producer role. */
struct Qwen4ExpSnapshotOutputShape
{
    Qwen4ExpSnapshotOutput role;
    DiagnosticSnapshotCopyGeometry geometry;
    /** @return The canonical physical slot shape, including native INT32 units. */
    SnapshotStorageShape storageShape() const
    {
        // The canonical metadata validator proves all dimensions and products.
        (void)diagnosticSnapshotMetadataCapacity(geometry);
        return {static_cast<std::size_t>(geometry.rows)*geometry.columns*geometry.elementBytes,geometry};
    }
};
/**
 * @brief Describe one branch norm output or its distinct, read-only wide input.
 * @param rows Admitted token rows, never current live rows.
 * @param hidden Collapsed model width; four branches form the observed width.
 * @param role Exactly normalized output or terminal raw input.
 */
inline Qwen4ExpSnapshotOutputShape qwen4ExpBranchSnapshotShape(
    int rows,int hidden,Qwen4ExpSnapshotOutput role)
{
    if(hidden<=0 || hidden>std::numeric_limits<int>::max()/4 ||
       (role!=Qwen4ExpSnapshotOutput::BranchNormalized && role!=Qwen4ExpSnapshotOutput::TerminalHiddenInput))
        throw std::invalid_argument("QFN branch snapshot requires its exact wide producer role and positive width");
    Qwen4ExpSnapshotOutputShape shape{role,{DiagnosticSnapshotCopyKind::TokenRows,rows,4*hidden,4}};
    (void)shape.storageShape();return shape;
}
/** @brief One distinct logical [M,4H] observation at the per-branch predictor FC producer. */
inline Qwen4ExpSnapshotOutputShape qwen4ExpPredictorFCSnapshotShape(int rows,int hidden)
{
    auto shape=qwen4ExpBranchSnapshotShape(rows,hidden,Qwen4ExpSnapshotOutput::BranchNormalized);
    shape.role=Qwen4ExpSnapshotOutput::PredictorFC;
    return shape;
}
/**
 * @brief Describe one real indexer/retention output using native source dimensions.
 * @param role The actual producer output, independent of semantic aliases.
 * @param rows Per-request admitted token rows.
 * @param context Native admitted context, not a live cache frontier.
 * @param selected Exact maximum expanded selection width from the model contract.
 */
inline Qwen4ExpSnapshotOutputShape qwen4ExpSparseSnapshotShape(
    Qwen4ExpSnapshotOutput role,int rows,int context,int selected)
{
    if(context<=0 || selected<=0)throw std::invalid_argument("QFN sparse snapshot requires admitted context and selection width");
    DiagnosticSnapshotCopyGeometry g{DiagnosticSnapshotCopyKind::TokenRows,rows,1,4};
    switch(role) {
    case Qwen4ExpSnapshotOutput::PreparedQuery:g.columns=4*128;break;
    case Qwen4ExpSnapshotOutput::SelectedCount:break;
    case Qwen4ExpSnapshotOutput::PooledKeys:
        g={DiagnosticSnapshotCopyKind::StatePrefixRows,context/4+(context%4!=0),128,4,context,4};break;
    case Qwen4ExpSnapshotOutput::SelectedIndices:g.kind=DiagnosticSnapshotCopyKind::CountedSelectionRows;g.columns=selected;break;
    case Qwen4ExpSnapshotOutput::RetainedRawKey:g.columns=128;break;
    default:throw std::invalid_argument("QFN sparse snapshot role is not a sparse producer output");
    }
    Qwen4ExpSnapshotOutputShape shape{role,g};(void)shape.storageShape();return shape;
}
/** @brief Exact authored indexer output inventory; aliases cannot multiply entries. */
inline auto qwen4ExpIndexerSnapshotShapes(int rows,int context,int selected)
{
    return std::array{qwen4ExpSparseSnapshotShape(Qwen4ExpSnapshotOutput::PreparedQuery,rows,context,selected),
        qwen4ExpSparseSnapshotShape(Qwen4ExpSnapshotOutput::SelectedCount,rows,context,selected),
        qwen4ExpSparseSnapshotShape(Qwen4ExpSnapshotOutput::PooledKeys,rows,context,selected),
        qwen4ExpSparseSnapshotShape(Qwen4ExpSnapshotOutput::SelectedIndices,rows,context,selected)};
}
/**
 * @brief Validate an actual authored descriptor against its source-owned shape.
 * @throws std::logic_error If a producer changes geometry without updating its BOM source.
 */
inline void requireQwen4ExpSnapshotShape(const Qwen4ExpSnapshotOutputShape &shape,
    std::size_t rows,std::size_t columns,std::size_t elementBytes)
{
    (void)shape.storageShape();
    if(rows!=static_cast<std::size_t>(shape.geometry.rows)||columns!=static_cast<std::size_t>(shape.geometry.columns)||
       elementBytes!=static_cast<std::size_t>(shape.geometry.elementBytes))
        throw std::logic_error("QFN authored snapshot descriptor diverges from source-owned geometry");
}
/**
 * @brief Bound only metadata/internal padding/leading alignment beyond payload.
 *
 * Every output is charged separately, including temporal reuse of one tensor.
 * A prior slot may end at any byte alignment; the explicit alignment bound is
 * the canonical required_alignment-1, never incidental spare payload capacity.
 */
inline std::size_t qwen4ExpSnapshotMetadataBytes(const Qwen4ExpSnapshotOutputShape &shape)
{
    const auto layout=snapshotStorageLayout(shape.storageShape());
    return snapshot_storage_detail::add(layout.storage_bytes-layout.payload_storage_bytes,layout.required_alignment-1);
}
} // namespace llaminar2
