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
        uint8_t source_codebook_id = 0; ///< Original GGUF arithmetic-policy codebook.
        uint8_t source_is_superblock = 0; ///< Original source block-family discriminator.
        uint8_t source_identity_present = 0; ///< Whether the two source fields are authoritative.
        uint8_t reserved = 0;             ///< Reserved ABI byte; producers write zero.
        uint32_t reserved_tail = 0;       ///< Explicit tail padding for byte-stable publication.

        /** @return Whether the descriptor names a minimally valid matrix. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return payload && scales && n > 0 && k > 0 && blocks_per_row > 0;
        }

        /**
         * @brief Return the codebook whose FP32 reduction tree must be retained.
         *
         * CPU-tier promotion can normalize several compact formats into one
         * expanded accelerator representation. The execution codebook selects
         * the decoder, while the source codebook keeps migration from changing
         * split-K partition boundaries and therefore output bits mid-request.
         */
        [[nodiscard]] constexpr uint8_t arithmeticPolicyCodebookId() const noexcept
        {
            return source_identity_present ? source_codebook_id : codebook_id;
        }
    };

    static_assert(
        std::is_standard_layout_v<DeviceNativeVNNIMatrixDesc>,
        "NativeVNNI device descriptors must remain a stable standard-layout ABI");
    static_assert(
        std::is_trivially_copyable_v<DeviceNativeVNNIMatrixDesc>,
        "NativeVNNI device descriptors must remain directly uploadable");
    static_assert(
        sizeof(DeviceNativeVNNIMatrixDesc) == 56,
        "NativeVNNI device descriptors must retain the backend ABI size");
} // namespace llaminar2
