/**
 * @file NativeVnniScalePlane.h
 * @brief Single authority for the byte width of prepared NativeVNNI metadata planes.
 *
 * Prepared NativeVNNI matrices keep each 32-value block's metadata in planes
 * that are separate from the payload: a primary scale plane, an optional
 * secondary plane (minimum, source coefficient or high-half scale) and an
 * optional Q2_K effective-minimum plane. Pool allocation, BOM estimates,
 * expert transfer descriptors, slot directories, tier layouts and on-device
 * rebalance packing all derive plane extents from these functions. A codebook
 * can then adopt a narrower primary plane (MXFP4's native E8M0 byte, issue
 * #26) without a second size table.
 *
 * The functions are `__host__ __device__`. CUDA builds do not pass
 * `--expt-relaxed-constexpr`, and calling a host-only constexpr function from
 * device code under `-rdc` can make the kernel silently never run.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_NVNNI_PLANE_HD __host__ __device__
#else
#define LLAMINAR_NVNNI_PLANE_HD
#endif

namespace llaminar2
{
    /** @brief Bytes per block of the secondary plane (always one 16-bit word). */
    inline constexpr std::size_t kNativeVnniSecondaryPlaneBytes = 2;
    /** @brief Bytes per block of the Q2_K effective-minimum plane. */
    inline constexpr std::size_t kNativeVnniEminPlaneBytes = 4;

    /**
     * @brief Bytes per 32-value block in a physical codebook's primary scale plane.
     * @param physical_codebook Device/CPU execution codebook (not the arithmetic policy).
     * @return Primary-plane width. Every codebook currently stores one FP16 word.
     */
    LLAMINAR_NVNNI_PLANE_HD constexpr std::size_t nativeVnniScalePlaneBytes(
        std::uint8_t physical_codebook) noexcept
    {
        (void)physical_codebook;
        return 2;
    }

    /**
     * @brief Exact byte extents of one prepared matrix's metadata planes.
     * @param block_count Number of 32-value blocks (rows * K / 32).
     * @param physical_codebook Execution codebook selecting the primary width.
     * @param has_secondary Whether the secondary plane exists.
     * @param has_emins Whether the Q2_K effective-minimum plane exists.
     */
    struct NativeVnniMetadataPlaneBytes
    {
        std::size_t scales = 0;
        std::size_t secondary = 0;
        std::size_t emins = 0;

        LLAMINAR_NVNNI_PLANE_HD static constexpr NativeVnniMetadataPlaneBytes of(
            std::size_t block_count, std::uint8_t physical_codebook,
            bool has_secondary, bool has_emins) noexcept
        {
            return {
                block_count * nativeVnniScalePlaneBytes(physical_codebook),
                has_secondary ? block_count * kNativeVnniSecondaryPlaneBytes : 0u,
                has_emins ? block_count * kNativeVnniEminPlaneBytes : 0u,
            };
        }
    };

    static_assert(NativeVnniMetadataPlaneBytes::of(10, 25, true, false).secondary == 20);
} // namespace llaminar2

#undef LLAMINAR_NVNNI_PLANE_HD
