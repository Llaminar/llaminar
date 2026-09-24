/**
 * @file NUMAPageQuery.h
 * @brief Interpret Linux batched NUMA observations without confusing migration with absence.
 *
 * move_pages queries do not fault migration entries. An ENOENT observation must
 * therefore be completed by the kernel's fault-resolving physical-node lookup.
 * No requested/preferred node is an input: unresolved or wrong placement cannot
 * be manufactured into success. Other errors remain fatal to certification.
 */
#pragma once

#include <cerrno>
#include <utility>

namespace llaminar2
{
    /**
     * @brief Complete one physical-page observation, with at most one resolving lookup.
     * @tparam ResidentLookup Callable returning the physical node, or a negative error.
     * @param batch_status Node or negative status returned by move_pages.
     * @param address Retained page whose observation is being completed.
     * @param lookup Kernel-backed fault-resolving physical-node query.
     * @return Actual node or unchanged error; never a requested placement assumption.
     */
    template <typename ResidentLookup>
    int resolveObservedNUMANode(int batch_status, const void *address, ResidentLookup &&lookup)
    {
        return batch_status == -ENOENT
            ? std::forward<ResidentLookup>(lookup)(address)
            : batch_status;
    }
}
