/**
 * @file CPURouterQ8StorageLayout.h
 * @brief Source-authenticated CPU router publication storage without allocation.
 *
 * Hidden width remains part of identity even when two widths round to the same
 * Q8 block count. Workspace families may merge row capacity only, and grouped
 * scratch reuses this exact declaration for its existing router root. The root
 * allocator owns physical claims; this metadata is not a live memory ledger.
 */
#pragma once
#include "CPUStorageRoot.h"
#include "tensors/BlockStructures.h"
#include <algorithm>
#include <limits>

namespace llaminar2
{
/** @brief Complete immutable physical geometry for one CPU router publication. */
struct CPURouterQ8StorageLayout
{
    CPUStorageRoot<Q8_1Block> root; ///< Exact backing owner, allocator and policy.
    size_t row_capacity = 0; ///< Maximum simultaneous published hidden rows.
    int d_model = 0; ///< Exact source width, including the final partial block.
    bool operator==(const CPURouterQ8StorageLayout &) const = default;

    /**
     * @brief Declare one root without materializing storage.
     * @param rows Maximum published hidden rows.
     * @param hidden_width Exact source hidden width.
     * @param device Exact CPU allocator identity.
     * @param placement Physical first-touch policy.
     * @param initialization Producer initialization policy.
     * @return Complete structurally comparable publication declaration.
     * @throws std::invalid_argument For nonpositive geometry or a non-CPU allocator.
     * @throws std::overflow_error For an unrepresentable physical root.
     */
    static CPURouterQ8StorageLayout declare(size_t rows, int hidden_width, DeviceId device,
        CPUStoragePlacement placement = CPUStoragePlacement::local(),
        CPUStorageInitialization initialization = CPUStorageInitialization::Zero)
    {
        if (!rows || hidden_width <= 0 || !device.is_cpu())
            throw std::invalid_argument("CPU router publication requires positive rows, width and exact CPU allocator");
        const size_t blocks = (size_t(hidden_width) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        if (rows > std::numeric_limits<size_t>::max() / blocks)
            throw std::overflow_error("CPU router publication extent overflows");
        return {CPUStorageRoot<Q8_1Block>(rows * blocks, PhysicalMemoryOwner::ExecutionWorkspace,
            device, placement, initialization), rows, hidden_width};
    }

    /**
     * @brief Reject partial or inconsistent source and root geometry.
     * @throws std::invalid_argument For a declaration differing from its canonical root.
     * @throws std::overflow_error For an unrepresentable physical extent.
     */
    void validate() const
    {
        if (*this != declare(row_capacity, d_model, root.allocator(), root.placement(), root.initialization()))
            throw std::invalid_argument("CPU router publication declaration is not canonical");
    }

    /**
     * @brief Merge row capacity within one source and physical allocation policy.
     * @param other Complete serial participant declaration.
     * @return Canonical maximum row capacity with unchanged source identity.
     * @throws std::invalid_argument For incomplete or incompatible source identity.
     * @throws std::overflow_error For an unrepresentable physical extent.
     */
    [[nodiscard]] CPURouterQ8StorageLayout merge(const CPURouterQ8StorageLayout &other) const
    {
        validate();
        other.validate();
        if (d_model != other.d_model || root.allocator() != other.root.allocator() ||
            root.placement() != other.root.placement() || root.initialization() != other.root.initialization())
            throw std::invalid_argument("CPU router publications have incompatible source geometry or allocator");
        return declare(std::max(row_capacity, other.row_capacity), d_model,
            root.allocator(), root.placement(), root.initialization());
    }

    /**
     * @brief Authenticate a request against this frozen envelope.
     * @param requested Complete source-authenticated request.
     * @return Whether its rows fit with unchanged source identity.
     * @throws std::invalid_argument For incomplete or incompatible source identity.
     * @throws std::overflow_error For an unrepresentable physical extent.
     */
    [[nodiscard]] bool contains(const CPURouterQ8StorageLayout &requested) const
    {
        return merge(requested) == *this;
    }

    /** @return Exact independently rounded root extent, without claiming it. */
    [[nodiscard]] size_t physicalBytes() const { return root.physicalBytes(); }
};
}
