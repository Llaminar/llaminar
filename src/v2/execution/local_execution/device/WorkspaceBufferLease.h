/**
 * @file WorkspaceBufferLease.h
 * @brief Retained physical ownership of one bounded named workspace region.
 *
 * A captured collective may consume metadata already owned by a workspace.
 * Retaining that exact allocation avoids a second mailbox and a replay-time
 * copy. The allocation's original PhysicalMemoryAuthority claim stays live
 * until its last lease retires. This type neither allocates device memory nor
 * owns a second accounting ledger, and does not make concurrent reuse safe:
 * the enclosing graph still orders every producer and consumer explicitly.
 */
#pragma once

#include "backends/DeviceId.h"
#include <cstddef>
#include <memory>

namespace llaminar2
{
class IBackend;
class DeviceWorkspaceManager;
namespace detail { struct WorkspaceAllocation; }

/** @brief Immutable allocation/extent identity retained by a captured operation. */
class WorkspaceBufferLease final
{
public:
    /** @brief Drop this reference; the last physical owner releases bytes then their claim. */
    ~WorkspaceBufferLease();
    WorkspaceBufferLease(const WorkspaceBufferLease &) = delete;
    WorkspaceBufferLease &operator=(const WorkspaceBufferLease &) = delete;

    /** @return Exact physical device that allocated the retained region. */
    [[nodiscard]] DeviceId device() const noexcept;
    /** @return Process-lifetime backend owner, frozen when storage was allocated. */
    [[nodiscard]] IBackend *backend() const noexcept;
    /** @return Size of this named subregion, never the whole physical block. */
    [[nodiscard]] size_t sizeBytes() const noexcept { return bytes_; }
    /** @return Whether a positive byte range fits without overflow. */
    [[nodiscard]] bool contains(size_t offset, size_t bytes) const noexcept;
    /** @brief Resolve bytes only within this immutable region.
     * @param offset Byte offset relative to the named region.
     * @return Stable backend address, including a legal one-past-end address.
     * @throws std::out_of_range for an offset beyond the retained extent. */
    [[nodiscard]] void *data(size_t offset = 0) const;

private:
    friend class DeviceWorkspaceManager;
    /** @brief Bind only after the manager validates the name and exact allocation.
     * @param allocation Original physical allocation and canonical claim.
     * @param offset Region start relative to that allocation.
     * @param bytes Positive extent within the named buffer. */
    WorkspaceBufferLease(std::shared_ptr<detail::WorkspaceAllocation> allocation,
        size_t offset, size_t bytes);

    std::shared_ptr<detail::WorkspaceAllocation> allocation_;
    const size_t offset_;
    const size_t bytes_;
};
}
