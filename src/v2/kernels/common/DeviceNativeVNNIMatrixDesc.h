/**
 * @file DeviceNativeVNNIMatrixDesc.h
 * @brief Backend-neutral device ABI for one prepared NativeVNNI matrix.
 *
 * Grouped GEMM/GEMV kernels select expert weights from device-resident tables.
 * This deliberately small standard-layout descriptor is the single ABI shared
 * by tensor preparation, MoE runtime placement, and backend launchers. Keeping
 * it outside the broad tensor-kernel interface lets CUDA and HIP translation
 * units consume the exact type without importing host-only tensor machinery or
 * maintaining hand-copied structures that can silently drift.
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace llaminar2
{
    /**
     * @brief Device-readable descriptor for one prepared NativeVNNI matrix.
     *
     * Pointers name immutable device allocations. Shape and codebook metadata
     * are copied alongside them so a grouped kernel can reject a descriptor
     * whose table slot does not match the captured projection. The allocation
     * fields describe reusable transfer slots; ordinary immutable weights leave
     * them zero because their physical allocation follows @ref codebook_id.
     */
    struct DeviceNativeVNNIMatrixDesc
    {
        const uint8_t *payload = nullptr; ///< Compact codebook payload blocks.
        const void *scales = nullptr;     ///< Per-block primary FP16 scale bits.
        const void *mins = nullptr;       ///< Optional minima or secondary scales.
        const void *emins = nullptr;      ///< Optional packed extended minima.
        int n = 0;                        ///< Output-column count.
        int k = 0;                        ///< Reduction width.
        uint32_t blocks_per_row = 0;      ///< Number of 32-value K blocks.
        uint8_t codebook_id = 0;          ///< NativeVNNI execution codebook.
        uint8_t allocation_payload_bytes_per_block = 0;
        uint8_t allocation_has_mins = 0;
        uint8_t allocation_has_emins = 0;

        /** @return Whether the descriptor names a minimally valid matrix. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return payload && scales && n > 0 && k > 0 && blocks_per_row > 0;
        }
    };

    static_assert(
        std::is_standard_layout_v<DeviceNativeVNNIMatrixDesc>,
        "NativeVNNI device descriptors must remain a stable standard-layout ABI");
    static_assert(
        std::is_trivially_copyable_v<DeviceNativeVNNIMatrixDesc>,
        "NativeVNNI device descriptors must remain directly uploadable");
} // namespace llaminar2
