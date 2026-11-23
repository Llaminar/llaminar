/**
 * @file NativeQuantisedScaleLoad.h
 * @brief Single read path for the prepared NativeQuantised primary scale plane.
 *
 * Three separate steps, so staged copies never reuse plane addressing:
 * loadStagedPrimary() reads the stored element from the global plane and
 * normalizes it to a 16-bit staged word; registers and LDS carry that word;
 * decodeStagedPrimary() turns it into the FP32 scale consumed by the existing
 * FP32 contribution contract. primaryScale() composes the two for direct
 * global reads. Raw byte-plane copies (cp.async) are not staged words until
 * their copy owner normalizes them.
 *
 * - **2-byte plane (every codebook today).** The loader returns
 *   `widen(plane[linear])`, using the caller's existing backend FP16 widening
 *   (CUDA fp16_bits_to_float, HIP __half2float, CPU F16C). Instruction
 *   sequences and arithmetic are unchanged.
 * - **1-byte plane (MXFP4's native E8M0 byte, issue #26).** The loader builds
 *   the normal FP32 scale 2^(e-128) from bits (mxfp4::normalScaleFP32Bits).
 *   Byte 0 is the reserved all-zero-block marker and yields +0. On e in
 *   [104, 143] this equals the widened FP16 scale bit for bit, so contribution
 *   trees and certified schedules stay valid; outside that window no FP16 form
 *   exists, which is why the loader returns FP32, not FP16 bits.
 *
 * Kernels keep a `const uint16_t *` plane base for ABI stability; the element
 * width is owned by nativeQuantisedScalePlaneBytes(), never by the pointer type.
 */
#pragma once

// Device qualifiers (__host__, __device__, __forceinline__) come from the
// backend runtime header; include it so this header is self-contained.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include "tensors/MXFP4Format.h"
#include "tensors/NativeQuantisedSuperblockMetadata.h"
#include "kernels/common/DeviceNativeQuantisedMatrixDesc.h"
#include "tensors/NativeQuantisedScalePlane.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_NATIVE_QUANTISED_SCALE_HD __host__ __device__ __forceinline__
#else
#define LLAMINAR_NATIVE_QUANTISED_SCALE_HD inline
#endif

namespace llaminar2::native_quantised_scale
{
    /** @brief Reinterpret IEEE-754 binary32 bits as float on host or device. */
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD float floatFromBits(std::uint32_t bits)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return __uint_as_float(bits);
#else
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
#endif
    }

    /**
     * @brief FP32 scale of an ADMITTED native E8M0 plane byte.
     * @param e Stored byte: 0 marks an all-zero block, otherwise e satisfies
     *        mxfp4::exponentFitsNativeE8M0 (e in [2, 252]).
     * @pre Validation belongs to the producer/admission authority (preparation,
     *      repack, tier CPU->GPU inverse); this hot-path decoder does not
     *      re-check and maps an unadmitted byte to a meaningless power of two.
     */
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD float e8m0Scale(std::uint8_t e)
    {
        return e == 0 ? 0.0f : floatFromBits(mxfp4::normalScaleFP32Bits(e));
    }

    /**
     * @brief Load block @p linear from the global plane as a normalized staged word.
     * @tparam Codebook Physical execution codebook (never the arithmetic policy).
     * @param plane Primary scale plane base address.
     * @param linear Block-major linear block index.
     * @return The staged word that registers and LDS slots carry: the original
     *         FP16 bits for 2-byte planes; for 1-byte planes the E8M0 byte is
     *         normalized to mxfp4::powerOfTwoBF16Word (0 stays 0), so staged
     *         decode is a branch-free placement. A staged word is never a raw
     *         plane byte. A stage that holds raw plane bytes instead (CUDA BK64
     *         async metadata for 1-byte planes) must be read through a
     *         consumer that applies powerOfTwoBF16Word to each byte.
     */
    template <std::uint8_t Codebook>
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD std::uint16_t loadStagedPrimary(const std::uint16_t *plane, std::size_t linear)
    {
        if constexpr (nativeQuantisedScalePlaneBytes(Codebook) == 1)
            return mxfp4::powerOfTwoBF16Word(reinterpret_cast<const std::uint8_t *>(plane)[linear]);
        else
            return plane[linear];
    }

    /** @brief Runtime-codebook form of loadStagedPrimary(). */
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD std::uint16_t loadStagedPrimary(
        std::uint8_t codebook, const std::uint16_t *plane, std::size_t linear)
    {
        return nativeQuantisedScalePlaneBytes(codebook) == 1
                   ? mxfp4::powerOfTwoBF16Word(reinterpret_cast<const std::uint8_t *>(plane)[linear])
                   : plane[linear];
    }

    /**
     * @brief Decode a staged word (see loadStagedPrimary) to the FP32 scale.
     * @tparam Codebook Physical execution codebook that produced @p staged.
     * @param staged Normalized staged word.
     * @param widen Caller's backend FP16-bits-to-FP32 widening (2-byte planes only),
     *        so existing expressions and reduction order are preserved.
     */
    template <std::uint8_t Codebook, class Widen>
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD float decodeStagedPrimary(std::uint16_t staged, Widen widen)
    {
        if constexpr (nativeQuantisedScalePlaneBytes(Codebook) == 1)
            return floatFromBits(std::uint32_t{staged} << 16);
        else
            return widen(staged);
    }

    /** @brief Runtime-codebook form of decodeStagedPrimary(). */
    template <class Widen>
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD float decodeStagedPrimary(std::uint8_t codebook, std::uint16_t staged, Widen widen)
    {
        return nativeQuantisedScalePlaneBytes(codebook) == 1 ? floatFromBits(std::uint32_t{staged} << 16)
                                                        : widen(staged);
    }

    /**
     * @brief FP32 scale of a staged native E8M0 word (1-byte planes only).
     * @param staged mxfp4::powerOfTwoBF16Word of an admitted byte, or 0.
     *
     * For call sites that keep their original FP16 expression verbatim in an
     * `if constexpr` else-branch instead of passing a widening callback (for
     * example decoder bodies textually included into a class).
     */
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD float stagedNativeScale(std::uint16_t staged)
    {
        return floatFromBits(std::uint32_t{staged} << 16);
    }

    /**
     * @brief Store a CPU-unit scale word into the GPU plane (tier CPU -> GPU inverse).
     * @param codebook Physical execution codebook of the destination plane.
     * @param plane Destination primary plane base address.
     * @param linear Block-major linear block index.
     * @param cpu_word CPU unit word: FP16 for 2-byte planes, otherwise
     *        mxfp4::powerOfTwoBF16Word (the staged word of a 1-byte plane).
     * @return False when a 1-byte plane receives a word that is not +0 or an
     *         admitted power of two; nothing is stored then.
     */
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD bool storePrimaryFromCpuWord(
        std::uint8_t codebook, std::uint16_t *plane, std::size_t linear, std::uint16_t cpu_word)
    {
        if (nativeQuantisedScalePlaneBytes(codebook) == 1)
        {
            std::uint8_t e = 0;
            if (!mxfp4::e8m0FromPowerOfTwoBF16Word(cpu_word, e))
                return false;
            reinterpret_cast<std::uint8_t *>(plane)[linear] = e;
            return true;
        }
        plane[linear] = cpu_word;
        return true;
    }

    using native_quantised_superblock::BlockAddress;
    using native_quantised_superblock::StagedBlockMetadata;

    /**
     * @brief Device metadata planes of one prepared matrix, with their addressing phase.
     *
     * `primary` is the primary word plane (per-block representation) or the
     * per-block code plane (superblock-coded). `secondary` is the secondary
     * word plane or the per-superblock d/dmin plane. `emins` is the Q2_K
     * effective-minimum plane of the per-block representation. The phase is
     * native_quantised_superblock::superblockPhase() of the matrix's first
     * prepared K block; it is zero for the per-block representation and for
     * every matrix that starts on a source superblock boundary.
     */
    struct NativeQuantisedMetadataPlanes
    {
        const std::uint16_t *primary = nullptr;
        const void *secondary = nullptr;
        const std::uint32_t *emins = nullptr;
        std::uint32_t superblock_phase = 0;
        NativeQuantisedMetadataRepresentation representation = NativeQuantisedMetadataRepresentation::PerBlockPlanes;
        std::uint32_t code_bit_shift = 0; ///< First code's bit offset inside `primary` (coded only).

        /** @return Whether the planes hold superblock-coded metadata. */
        LLAMINAR_NATIVE_QUANTISED_SCALE_HD bool coded() const
        {
            return representation == NativeQuantisedMetadataRepresentation::SuperblockCoded;
        }
    };

    /** @return The typed metadata planes and phase named by a matrix descriptor. */
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD NativeQuantisedMetadataPlanes metadataPlanes(
        const DeviceNativeQuantisedMatrixDesc &desc)
    {
        return {static_cast<const std::uint16_t *>(desc.scales), desc.mins,
                static_cast<const std::uint32_t *>(desc.emins), nativeQuantisedLayoutPhase(desc.metadata_layout),
                nativeQuantisedLayoutRepresentation(desc.metadata_layout),
                nativeQuantisedLayoutCodeBitShift(desc.metadata_layout)};
    }

    /** @return Address of block-major block (@p block, @p column) in an @p n-column matrix. */
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD BlockAddress blockMajorAddress(
        const NativeQuantisedMetadataPlanes &planes, std::uint32_t block, std::uint32_t column, std::uint32_t n)
    {
        return native_quantised_superblock::blockMajorAddress(block, column, n, planes.superblock_phase);
    }

    /** @return Address of row-major block (@p block, @p column) with @p blocks_per_row blocks per row. */
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD BlockAddress rowMajorAddress(
        const NativeQuantisedMetadataPlanes &planes, std::uint32_t block, std::uint32_t column,
        std::uint32_t blocks_per_row)
    {
        return native_quantised_superblock::rowMajorAddress(block, column, blocks_per_row, planes.superblock_phase);
    }

    /**
     * @brief Load one block's staged metadata words through the representation authority.
     * @tparam Codebook Physical execution codebook.
     * @param planes Matrix planes, phase and representation.
     * @param address Block address (linear plus superblock index).
     * @return The staged primary word (loadStagedPrimary semantics), the raw
     *         secondary word and the emin word, bit-identical across representations.
     *
     * The representation branch is uniform for a whole matrix, so it never
     * diverges within a warp/wavefront; incapable codebooks compile only the
     * per-block path.
     */
    template <std::uint8_t Codebook>
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD StagedBlockMetadata loadBlockMetadata(
        const NativeQuantisedMetadataPlanes &planes, BlockAddress address)
    {
        if constexpr (nativeQuantisedSuperblockCodable(Codebook))
        {
            if (planes.coded())
                return native_quantised_superblock::loadBlockMetadata<Codebook>(planes.primary, planes.secondary,
                                                                                address, planes.code_bit_shift);
        }
        StagedBlockMetadata out;
        out.primary = loadStagedPrimary<Codebook>(planes.primary, address.linear);
        out.secondary = planes.secondary ? static_cast<const std::uint16_t *>(planes.secondary)[address.linear]
                                         : std::uint16_t{0};
        // Only Q2_K owns an effective-minimum plane; never keep the pointer live otherwise.
        if constexpr (Codebook == 10)
            out.emin = planes.emins ? planes.emins[address.linear] : std::uint32_t{0};
        return out;
    }

    /** @brief Load and decode block @p linear directly from the global plane. */
    template <std::uint8_t Codebook, class Widen>
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD float primaryScale(const std::uint16_t *plane, std::size_t linear, Widen widen)
    {
        return decodeStagedPrimary<Codebook>(loadStagedPrimary<Codebook>(plane, linear), widen);
    }

    /** @brief Runtime-codebook form of primaryScale(). */
    template <class Widen>
    LLAMINAR_NATIVE_QUANTISED_SCALE_HD float primaryScale(
        std::uint8_t codebook, const std::uint16_t *plane, std::size_t linear, Widen widen)
    {
        return decodeStagedPrimary(codebook, loadStagedPrimary(codebook, plane, linear), widen);
    }
} // namespace llaminar2::native_quantised_scale

#undef LLAMINAR_NATIVE_QUANTISED_SCALE_HD
