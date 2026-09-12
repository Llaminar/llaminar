/**
 * @file SnapshotPublication.h
 * @brief Producer-owned completeness of an immutable diagnostic tensor.
 *
 * A finalizer can publish a complete value under the same semantic key as an
 * earlier TP partial. Completeness travels with the captured bytes, not with
 * the model's static sharding table, so collectors never reduce that value a
 * second time. This contract affects diagnostics only, not inference state.
 */
#pragma once

#include <cstdint>

namespace llaminar2
{
    /** @brief Whether schema assembly is still required for this publication. */
    enum class SnapshotPublication : uint8_t
    {
        SchemaPartition, ///< Assemble using the graph/schema's ordinary TP layout.
        CompleteValue,   ///< The named finalizer already assembled the semantic value.
    };
}
