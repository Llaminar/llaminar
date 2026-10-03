/**
 * @file SnapshotPublication.h
 * @brief Producer-owned completeness of an immutable diagnostic tensor.
 *
 * A finalizer can publish a complete value under the same semantic key as an
 * earlier TP partial. Completeness travels with the captured bytes, not with
 * the model's static sharding table, so collectors never reduce that value a
 * second time. A producer may instead expose contiguous output-column shards
 * in participant order when its physical layout differs from the schema's
 * whole-expert partial sums. This contract affects diagnostics only, not
 * inference state or graph execution.
 */
#pragma once

#include <cstdint>

namespace llaminar2
{
    /** @brief Producer-owned assembly contract attached to the captured bytes. */
    enum class SnapshotPublication : uint8_t
    {
        SchemaPartition, ///< Assemble using the graph/schema's ordinary TP layout.
        CompleteValue,   ///< The named finalizer already assembled the semantic value.
        ColumnPartition, ///< Join contiguous output columns in TP participant order; require every participant.
        RowPartition,    ///< Copy explicitly owned logical rows once; never sum or infer ownership from zeros.
    };
}
