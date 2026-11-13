/**
 * @file QwenSparseTransactionPrior.h
 * @brief Typed packed verifier-prior bytes with exact metadata/payload extents.
 *
 * These ephemeral arena bytes belong to one immutable verifier transaction,
 * never the prefix archive or a second accepted frontier. The transaction lease
 * excludes reset, ordinary commit and a new snapshot until terminal retirement.
 */
#pragma once
#include "Qwen4ExpCheckpointGeometry.h"
#include "QwenSparseTerminalCounters.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace llaminar2::qwen4exp {
/** @brief Execution-owned causal transaction base; CPU host or native GPU words. */
struct QwenSparseReplay {
    int prefix_tokens=0; ///< Accepted logical tokens before this transaction.
    int live_rows=0; ///< Current rows bounded by the admitted row capacity.
    int position_origin=0; ///< Absolute position of logical cache index zero.
};
/** @brief Packed INT32 arena codec; FP32 payload is exposed only through this typed record. */
struct alignas(16) QwenSparseTransactionPrior {
    QwenSparseReplay replay; ///< Immutable original preappend metadata.
    int row_capacity; ///< Authenticated original physical verifier geometry.
    float partial[Qwen4ExpCheckpointGeometry::TailRows*Qwen4ExpCheckpointGeometry::IndexDimension]; ///< Only prefix%4 rows have payload.
};
static_assert(offsetof(QwenSparseTransactionPrior,partial)==Qwen4ExpCheckpointGeometry::TailHeaderBytes);
static_assert(sizeof(QwenSparseTransactionPrior)==Qwen4ExpCheckpointGeometry::sparseLayerBytes() && sizeof(QwenSparseTransactionPrior)/sizeof(std::int32_t)==388);
static_assert(std::is_trivially_copyable_v<QwenSparseTransactionPrior>);
/** @brief Exact native traffic observation; required metadata is distinct from live payload. */
struct QwenSparsePriorTraffic {
    std::uint32_t metadata_bytes; ///< Required replay/geometry publication, always16.
    std::uint32_t payload_bytes; ///< Exactly live>0 ? (prefix%4)*128*4 :0.
};
static_assert(sizeof(QwenSparsePriorTraffic)==8);
/** @brief Borrowed snapshot operands; no allocator, pool or native KV bank is representable. */
struct QwenSparsePriorSnapshot {
    const QwenSparseReplay* replay; ///< Original ordered preappend producer.
    const float* accepted_partial; ///< Original accepted raw tail, at most3 live rows.
    QwenSparseTransactionPrior* output; ///< Distinct admitted packed INT32 transaction bank.
    QwenSparsePriorTraffic* traffic; ///< Distinct exact8-byte native observation.
    int row_capacity,token_capacity; ///< Immutable admitted graph/cache geometry.
    QwenSparseWorkCounterRow work_counters; ///< Optional original request row, never a graph-owned counter mirror.
};
/** @brief Reject incomplete or aliased immutable bindings without reading native state. */
void validateQwenSparsePriorSnapshot(const QwenSparsePriorSnapshot& params);
/** @brief Copy only authenticated live original tail rows on the CPU authority. */
void cpuQwenSparseSnapshotPrior(const QwenSparsePriorSnapshot& params);
/** @brief Enqueue the same exact snapshot on an explicit non-default CUDA stream. */
void cudaQwenSparseSnapshotPrior(const QwenSparsePriorSnapshot& params,void* stream);
/** @brief Enqueue the same exact snapshot on an explicit non-default HIP stream. */
void rocmQwenSparseSnapshotPrior(const QwenSparsePriorSnapshot& params,void* stream);
} // namespace llaminar2::qwen4exp
