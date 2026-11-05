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

// Device qualifiers (__host__, __device__, __forceinline__) come from the
// backend runtime header; include it so this header is self-contained.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

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
     * @return 1 for MXFP4 (codebook 25), which stores its native E8M0 byte;
     *         2 (one FP16 word) for every other codebook.
     */
    LLAMINAR_NVNNI_PLANE_HD constexpr std::size_t nativeVnniScalePlaneBytes(
        std::uint8_t physical_codebook) noexcept
    {
        return physical_codebook == 25 ? 1 : 2;
    }

    /**
     * @brief Typed primary-plane width used for allocation capacity.
     *
     * An enum class, so pool planners cannot receive an unrelated integer and
     * every allocation states its plane width explicitly.
     */
    enum class NativeVnniScalePlaneWidth : std::uint8_t
    {
        E8M0Byte = 1, ///< MXFP4's native exponent byte.
        FP16Word = 2, ///< One 16-bit word (every other codebook; the widest plane).
    };

    /** @return Plane width of physical codebook @p physical_codebook. */
    LLAMINAR_NVNNI_PLANE_HD constexpr NativeVnniScalePlaneWidth nativeVnniScalePlaneWidth(
        std::uint8_t physical_codebook) noexcept
    {
        return nativeVnniScalePlaneBytes(physical_codebook) == 1 ? NativeVnniScalePlaneWidth::E8M0Byte
                                                                 : NativeVnniScalePlaneWidth::FP16Word;
    }

    /** @return The wider of two plane capacities (union of formats sharing a slot). */
    LLAMINAR_NVNNI_PLANE_HD constexpr NativeVnniScalePlaneWidth widestScalePlane(
        NativeVnniScalePlaneWidth a, NativeVnniScalePlaneWidth b) noexcept
    {
        return static_cast<std::uint8_t>(a) >= static_cast<std::uint8_t>(b) ? a : b;
    }

    /** @return Bytes per block of a plane capacity. */
    LLAMINAR_NVNNI_PLANE_HD constexpr std::size_t scalePlaneBytes(NativeVnniScalePlaneWidth width) noexcept
    {
        return static_cast<std::size_t>(width);
    }

    /**
     * @brief 16-bit storage words that hold a primary plane of @p block_count blocks.
     *
     * Host-packed planes keep `uint16_t` storage so device pointer arithmetic
     * stays word-aligned; a 1-byte plane uses the first block_count bytes.
     */
    LLAMINAR_NVNNI_PLANE_HD constexpr std::size_t nativeVnniScalePlaneStorageWords(
        std::uint8_t physical_codebook, std::size_t block_count) noexcept
    {
        return (block_count * nativeVnniScalePlaneBytes(physical_codebook) + 1) / 2;
    }

    /**
     * @brief Exact live bytes of a primary plane holding @p block_count blocks.
     *
     * This is the transfer and device-view length. The storage-word count above
     * is host capacity only: for an odd 1-byte plane it includes one byte that
     * belongs to no block and must never be transferred.
     */
    LLAMINAR_NVNNI_PLANE_HD constexpr std::size_t nativeVnniScalePlaneLiveBytes(
        std::uint8_t physical_codebook, std::size_t block_count) noexcept
    {
        return block_count * nativeVnniScalePlaneBytes(physical_codebook);
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
            return of(block_count, nativeVnniScalePlaneWidth(physical_codebook), has_secondary, has_emins);
        }

        /** @brief Plane extents for an explicit primary capacity (slots shared by formats). */
        LLAMINAR_NVNNI_PLANE_HD static constexpr NativeVnniMetadataPlaneBytes of(
            std::size_t block_count, NativeVnniScalePlaneWidth width,
            bool has_secondary, bool has_emins) noexcept
        {
            return {
                block_count * scalePlaneBytes(width),
                has_secondary ? block_count * kNativeVnniSecondaryPlaneBytes : 0u,
                has_emins ? block_count * kNativeVnniEminPlaneBytes : 0u,
            };
        }
    };

    static_assert(NativeVnniMetadataPlaneBytes::of(10, 25, true, false).secondary == 20);
    static_assert(NativeVnniMetadataPlaneBytes::of(10, 25, false, false).scales == 10);
    static_assert(NativeVnniMetadataPlaneBytes::of(10, 4, false, false).scales == 20);
} // namespace llaminar2

#undef LLAMINAR_NVNNI_PLANE_HD
