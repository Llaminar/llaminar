/**
 * @file GraphArenaStorageLayout.h
 * @brief Resolved graph-root inventory shared by admission and arena registration.
 *
 * Schema buffers and explicitly declared continuation mailboxes form one list
 * of physical roots. Alias annotations do not merge allocations. The layout
 * owns declarations only; PhysicalMemoryAuthority owns every live allocation.
 */
#pragma once

#include "BufferId.h"
#include "execution/debug/BufferRole.h"
#include <span>
#include <vector>

namespace llaminar2
{
struct GraphSchema;
struct GraphResolverConfig;

/** @brief Additional roots retained by orchestration beside the model schema. */
struct GraphArenaSupplementalStorage
{
    bool full_prefill_kv_handoff = false; ///< Separate full-width K/V banks for phase-split prefill.
    size_t terminal_hidden_rows = 0; ///< Zero when no retained MTP capacity exists.
    size_t terminal_hidden_columns = 0; ///< Residual width, including the QFN trunk.
};

/** @brief One independently allocated arena root and its stable graph identity. */
struct GraphArenaStorageEntry
{
    BufferId id;
    BufferDescriptor descriptor;
};

/** @brief Immutable resolved inventory, independent of model execution and live accounting. */
class GraphArenaStorageLayout final
{
public:
    /**
     * @brief Resolve the production schema and continuation roots once.
     * @param schema Actual model buffer declarations.
     * @param resolver Exact participant geometry and name mappings.
     * @param supplemental Explicit roots owned outside the model schema.
     * @return Complete physical inventory; unused quadratic attention scratch is omitted.
     * @throws std::invalid_argument For unknown or duplicate buffer identities.
     */
    static GraphArenaStorageLayout resolve(const GraphSchema &schema,
        const GraphResolverConfig &resolver, GraphArenaSupplementalStorage supplemental = {});

    /** @return Every physical root in registration order; borrowed tensors are not included. */
    [[nodiscard]] std::span<const GraphArenaStorageEntry> entries() const noexcept { return entries_; }

    /**
     * @brief Price CPU roots through the same declarations used by BufferArena.
     * @return Exact sum of allocator-rounded physical extents.
     * @throws std::invalid_argument If any declaration targets a non-CPU allocator.
     */
    [[nodiscard]] size_t cpuPhysicalBytes() const;

private:
    std::vector<GraphArenaStorageEntry> entries_; ///< Immutable after resolve returns.
};
}
