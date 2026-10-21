/**
 * @file CapturedTransferChannelBatch.h
 * @brief Bounded native argument packet for independent original channel endpoints.
 *
 * This packet owns no allocation, cursor, progress or live count. TransferEngine
 * validates retained bindings before constructing it. Three entries bound native
 * kernel argument storage; larger exchanges use successive complete packets.
 * Each entry keeps its original protocol and endpoint-local copy coordinates.
 */
#pragma once
#include "CapturedTransferChannelProtocol.h"

namespace llaminar2
{
    /** @brief One original message plus its exact immutable copy aliases. */
    struct CapturedTransferChannelBatchEntry
    {
        CapturedTransferChannelDeviceBinding binding;
        void *destination = nullptr;
        const void *source = nullptr;
        unsigned copy_blocks = 0; ///< Backend-owned cold geometry, never a live extent.
        bool vector_aligned = false; ///< Frozen alias alignment, not count alignment.
    };

    /** @brief Stack/native-argument packet, retained by native graph recording by value. */
    struct CapturedTransferChannelBatch
    {
        static constexpr unsigned kCapacity = 3;
        CapturedTransferChannelBatchEntry entries[kCapacity]{};
        unsigned count = 0;

        /** @return Whether every active entry names a distinct same-role original endpoint. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            if (!count || count > kCapacity) return false;
            for (unsigned i = 0; i < count; ++i)
            {
                const auto &entry = entries[i];
                if (!entry.binding.valid() || !entry.destination || !entry.source ||
                    entry.binding.role != entries[0].binding.role) return false;
                for (unsigned j = 0; j < i; ++j)
                    if (entry.binding.cursor == entries[j].binding.cursor ||
                        entry.binding.control == entries[j].binding.control) return false;
            }
            return true;
        }
    };
    static_assert(std::is_trivially_copyable_v<CapturedTransferChannelBatch>);
    // Leave ample space under the CUDA/HIP common 4 KiB kernel-argument bound.
    static_assert(sizeof(CapturedTransferChannelBatch) < 3072);
}
