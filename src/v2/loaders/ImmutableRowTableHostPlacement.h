/**
 * @file ImmutableRowTableHostPlacement.h
 * @brief Immutable RAM locality policy, separate from the process CPU allocator.
 *
 * Task policy preserves Linux inherited affinity/memory policy and first touch;
 * it makes no exact-node promise. Exact-node policy certifies fresh RAM before
 * source reads. Neither policy changes PMA identities, budgets or mmap page-cache
 * locality. Rank observation -1 means whole-host/task policy, never node zero.
 */
#pragma once
#include <optional>
#include <stdexcept>
namespace llaminar2
{
/** @brief Setup-only native-table RAM placement; owns no storage or capacity. */
class ImmutableRowTableHostPlacement final
{
public:
    /** @return Explicit inherited Linux task/system first-touch policy. */
    static ImmutableRowTableHostPlacement taskPolicy() noexcept { return ImmutableRowTableHostPlacement(std::nullopt); }
    /**
     * @brief Require physical placement on one exact node before source reads.
     * @param node Nonnegative observed physical NUMA node.
     * @return Checked immutable exact-node policy.
     * @throws std::invalid_argument If the node is negative.
     */
    static ImmutableRowTableHostPlacement onNode(int node)
    {
        if (node < 0) throw std::invalid_argument("Native table locality requires a nonnegative exact NUMA node");
        return ImmutableRowTableHostPlacement(node);
    }
    /**
     * @brief Project the canonical rank observation without inventing an allocator.
     * @param node Whole-host -1 or an exact nonnegative NUMA affinity.
     * @return Explicit task policy or checked exact-node policy.
     * @throws std::invalid_argument For sentinel values below -1.
     */
    static ImmutableRowTableHostPlacement fromObservedNUMANode(int node)
    {
        return node == -1 ? taskPolicy() : onNode(node);
    }
    /** @return Exact node when requested; absence deliberately promises no locality. */
    std::optional<int> numaNode() const noexcept { return node_; }
    /** @return Whether first touch follows the existing Linux task/system policy. */
    bool usesTaskPolicy() const noexcept { return !node_; }
    bool operator==(const ImmutableRowTableHostPlacement &) const = default;
private:
    /** @brief Only checked factories may construct the immutable policy. */
    explicit ImmutableRowTableHostPlacement(std::optional<int> node) noexcept : node_(node) {}
    std::optional<int> node_;
};
}
