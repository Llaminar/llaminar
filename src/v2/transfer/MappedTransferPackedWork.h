/**
 * @file MappedTransferPackedWork.h
 * @brief Immutable packed-weight parameters for the canonical transfer service.
 *
 * This is a node-local device work description, not an MoE placement policy or
 * another transfer owner. A permanent progress slot retains the source,
 * destination and staging leases; its existing generation receipt owns their
 * reuse. Repacking and copying therefore cannot be separated by a native queue
 * which is itself waiting for the inference graph to finish.
 */
#pragma once

#include "MappedTransferProgressABI.h"
#include "execution/moe/ExpertTierWeightDeviceLayout.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llaminar2
{
    /**
     * @brief Scalar conversion geometry and already-owned GPU addresses.
     *
     * The command's ordinary source/destination fields name the GPU payload
     * and mapped CPU bytes according to direction. This descriptor supplies
     * only the additional separated metadata and the lane's existing device
     * staging. Conversion reuses the same unit arithmetic as native repacking;
     * staging keeps scattered format accesses off the PCIe link.
     */
    struct MappedTransferPackedWork
    {
        MappedTransferWorkKind kind = MappedTransferWorkKind::Bytes;
        std::uint32_t first_unit = 0u;
        ExpertTierWeightDeviceLayout layout;
        std::uint64_t scales = 0u;
        std::uint64_t mins = 0u;
        std::uint64_t emins = 0u;
        std::uint64_t device_staging = 0u;
    };

    static_assert(offsetof(MappedTransferPackedWork, kind) == 0u);
    static_assert(sizeof(MappedTransferPackedWork) <=
                  sizeof(std::uint64_t) * kMappedTransferWorkWords);
    static_assert(std::is_standard_layout_v<MappedTransferPackedWork>);
    static_assert(std::is_trivially_copyable_v<MappedTransferPackedWork>);
} // namespace llaminar2
