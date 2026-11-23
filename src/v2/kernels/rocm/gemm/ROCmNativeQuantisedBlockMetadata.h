/**
 * @file ROCmNativeQuantisedBlockMetadata.h
 * @brief ROCm readers' single access path for prepared NativeQuantised block metadata.
 *
 * A prepared matrix stores its per-32 metadata in one of two representations
 * (NativeQuantisedScalePlane.h): per-block planes (primary FP16 word,
 * secondary word and, for Q2_K, a packed effective-minimum word) or
 * superblock-coded planes (bit-packed per-block source codes plus one FP16 d,
 * or d|dmin, per source superblock; docs/v2/projects/2026-11/
 * GPU_SUPERBLOCK_METADATA_PLAN.md). The coded form keeps GPU weights at GGUF
 * file density.
 *
 * Every ROCm decoder reads block metadata through blockMetadata(), which
 * returns the identical staged words for both representations
 * (native_quantised_scale::loadBlockMetadata, proved exhaustively by
 * Test__NativeQuantisedSuperblockMetadata and on device by
 * Test__ROCmSuperblockCodedMetadata). Decoder arithmetic, instruction order
 * and output bytes are therefore unchanged; only the metadata fetch differs.
 *
 * IQ3_XXS, IQ2_XS and IQ2_XXS coded payloads store their scale code in the
 * even-parity top bit of each seven-bit sign selector (and IQ2_XS also in the
 * spare upper nibble of its qh byte). Decoders recompute that parity with
 * restoreSignParity*() before using a sign byte. The recomputation is
 * idempotent for per-block payloads, whose sign bytes already carry even
 * parity (ksigns_iq2xs), so it is applied unconditionally and the decode
 * stays branch-free.
 */
#pragma once

// Host C++ translation units use the plane type and admission only; the
// device readers below compile in HIP translation units.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include "kernels/common/NativeQuantisedScaleLoad.h"
#include "tensors/NativeQuantisedSuperblockMetadata.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__HIPCC__)
#define LLAMINAR_ROCM_METADATA_HD __host__ __device__ __forceinline__
#else
#define LLAMINAR_ROCM_METADATA_HD inline
#endif

namespace llaminar2::rocm_native_metadata
{
    using MetadataPlanes = ::llaminar2::native_quantised_scale::NativeQuantisedMetadataPlanes;
    using StagedBlockMetadata = ::llaminar2::native_quantised_superblock::StagedBlockMetadata;
    using BlockAddress = ::llaminar2::native_quantised_superblock::BlockAddress;

    /** @return Whether codebook @p cb's payload sign selectors carry even parity (IQ3_XXS, IQ2_XS, IQ2_XXS). */
    LLAMINAR_ROCM_METADATA_HD constexpr bool signParityCoded(std::uint8_t cb) noexcept
    {
        return cb == 12 || cb == 14 || cb == 15;
    }

#if defined(__HIPCC__)
    /**
     * @brief Restore the even-parity top bit of one sign byte of a parity-coded codebook.
     * @tparam CB Physical codebook; other codebooks return @p sign unchanged.
     */
    template <std::uint8_t CB>
    __device__ __forceinline__ std::uint8_t restoreSignParity(std::uint8_t sign)
    {
        if constexpr (signParityCoded(CB))
            return static_cast<std::uint8_t>((sign & 0x7Fu) | ((__popc(sign & 0x7Fu) & 1u) << 7));
        else
            return sign;
    }

    /**
     * @brief Restore the parity bit of the sign bytes selected by @p ByteMask in a payload word.
     * @tparam CB Physical codebook; other codebooks return @p word unchanged.
     * @tparam ByteMask 0xFF in every byte lane that holds a sign selector (other
     *         lanes, such as an IQ2_XS qh byte, pass through untouched).
     *
     * Per byte: fold the seven low bits to their parity in bit 0 (each fold
     * stays inside its byte for bits 0..3), then place it in bit 7.
     */
    template <std::uint8_t CB, std::uint32_t ByteMask = 0xFFFFFFFFu>
    __device__ __forceinline__ std::uint32_t restoreSignParityWord(std::uint32_t word)
    {
        if constexpr (signParityCoded(CB))
        {
            const std::uint32_t low = word & 0x7F7F7F7Fu;
            std::uint32_t parity = low ^ (low >> 4);
            parity ^= parity >> 2;
            parity ^= parity >> 1;
            const std::uint32_t restored = low | ((parity & 0x01010101u) << 7);
            return (restored & ByteMask) | (word & ~ByteMask);
        }
        else
            return word;
    }

    /** @return Block-major address of block (@p block, @p column) in an @p n-column matrix. */
    __device__ __forceinline__ BlockAddress blockAddress(const MetadataPlanes &planes, std::uint32_t block,
                                                         std::uint32_t column, std::uint32_t n)
    {
        return ::llaminar2::native_quantised_scale::blockMajorAddress(planes, block, column, n);
    }

    /**
     * @brief Staged metadata words of block-major block (@p block, @p column).
     * @tparam CB Physical execution codebook.
     * @return primary (staged), secondary and emin words, bit-identical for both representations.
     */
    template <std::uint8_t CB>
    __device__ __forceinline__ StagedBlockMetadata blockMetadata(const MetadataPlanes &planes, std::uint32_t block,
                                                                 std::uint32_t column, std::uint32_t n)
    {
        return ::llaminar2::native_quantised_scale::loadBlockMetadata<CB>(planes,
                                                                         blockAddress(planes, block, column, n));
    }

    /**
     * @brief Staged metadata words with the representation fixed at compile time.
     * @tparam CB Physical execution codebook.
     * @tparam Coded True only in instantiations launched for a superblock-coded
     *         matrix (host-selected from its uniform layout). The per-block
     *         instantiation keeps exactly the original plane loads, so tuned
     *         per-block kernels retain their code and register budget.
     * @param planes Matrix planes. @param linear Block-major index (block * n + column).
     * @param block K32 block. @param column Output column. @param n Block-major pitch.
     * @param loaded_payload This block's payload words already in registers
     *        (code-in-payload formats read their code from them).
     */
    template <std::uint8_t CB, bool Coded>
    __device__ __forceinline__ StagedBlockMetadata blockMetadataAs(const MetadataPlanes &planes, std::size_t linear,
                                                                   std::uint32_t block, std::uint32_t column,
                                                                   std::uint32_t n,
                                                                   const std::uint32_t *loaded_payload = nullptr)
    {
        if constexpr (Coded)
        {
            namespace sb = ::llaminar2::native_quantised_superblock;
            static_assert(nativeQuantisedSuperblockCodable(CB), "only codable codebooks have a coded instantiation");
            // The caller's live block-major index is reused. The superblock
            // index is formed in 32 bits: a superblock plane holds one word per
            // 256 weights, so 2^32 entries would need a single matrix of more
            // than 10^12 weights, far beyond any admitted device allocation.
            const sb::BlockAddress address{
                linear, static_cast<std::uint32_t>((block + planes.superblock_phase) / sb::kBlocksPerSuperblock) * n +
                            column};
            std::uint32_t code;
            if constexpr (sb::superblockCodeInPayload(CB))
            {
                // The caller already holds this block's payload in registers;
                // read the embedded code from them instead of re-loading it.
                code = loaded_payload ? sb::codeFromPayload<CB>(reinterpret_cast<const std::uint8_t *>(loaded_payload))
                                      : sb::loadCode<CB>(planes.primary, address.linear);
            }
            else if constexpr (sb::superblockCodeBits(CB) == 16u || sb::superblockCodeBits(CB) == 8u)
            {
                // Byte-aligned widths: a run's code offset is a multiple of 8
                // bits, so the code bit shift is zero and the code is exactly
                // one aligned element of the plane (identical to loadCode).
                using Element = std::conditional_t<sb::superblockCodeBits(CB) == 16u, std::uint16_t, std::uint8_t>;
                code = static_cast<const Element *>(static_cast<const void *>(planes.primary))[address.linear];
            }
            else
                code = sb::loadCode<CB>(planes.primary, address.linear, planes.code_bit_shift);
            return sb::decodeSuperblockMetadata<CB>(code, sb::loadSuperblock<CB>(planes.secondary, address.superblock));
        }
        else
        {
            // Exactly the original per-block reads: callers consume only the
            // planes host admission proved present, so unconsumed loads are
            // dead after inlining and never dereference an absent plane.
            (void)block;
            (void)column;
            (void)n;
            StagedBlockMetadata out;
            out.primary = ::llaminar2::native_quantised_scale::loadStagedPrimary<CB>(planes.primary, linear);
            out.secondary = static_cast<const std::uint16_t *>(planes.secondary)[linear];
            if constexpr (CB == 10)
                out.emin = planes.emins[linear];
            return out;
        }
    }

    /** @brief Staged metadata words at an already computed block address. */
    template <std::uint8_t CB>
    __device__ __forceinline__ StagedBlockMetadata blockMetadataAt(const MetadataPlanes &planes, BlockAddress address)
    {
        return ::llaminar2::native_quantised_scale::loadBlockMetadata<CB>(planes, address);
    }

#endif // __HIPCC__

    /**
     * @brief Planes from raw plane pointers and a packed metadata layout byte.
     * @param primary Primary (per-block) or code plane. @param secondary Secondary or superblock plane.
     * @param emins Q2_K effective-minimum plane (per-block only).
     * @param layout Packed layout byte (nativeQuantisedMetadataLayout) carried by
     *        descriptors and kernel-argument records.
     */
    LLAMINAR_ROCM_METADATA_HD MetadataPlanes planesFromLayout(const void *primary, const void *secondary,
                                                              const void *emins, std::uint8_t layout) noexcept
    {
        return {static_cast<const std::uint16_t *>(primary), secondary, static_cast<const std::uint32_t *>(emins),
                nativeQuantisedLayoutPhase(layout), nativeQuantisedLayoutRepresentation(layout),
                nativeQuantisedLayoutCodeBitShift(layout)};
    }

    /** @return The packed layout byte of @p planes (inverse of planesFromLayout). */
    LLAMINAR_ROCM_METADATA_HD std::uint8_t packedLayout(const MetadataPlanes &planes) noexcept
    {
        return nativeQuantisedMetadataLayout(planes.representation, planes.superblock_phase, planes.code_bit_shift);
    }

    /**
     * @brief Per-block (layout zero) planes from raw plane pointers.
     * @param scales Primary plane. @param mins Optional secondary plane.
     * @param emins Optional Q2_K effective-minimum plane.
     * @return Planes in the per-block representation; never use this for a
     *         matrix whose producer declared a superblock-coded layout.
     */
    LLAMINAR_ROCM_METADATA_HD MetadataPlanes perBlockMetadataPlanes(const void *scales, const void *mins,
                                                                    const void *emins) noexcept
    {
        return planesFromLayout(scales, mins, emins, 0);
    }

    /**
     * @brief Host-side admission of a matrix's metadata layout for ROCm readers.
     * @return Whether @p planes names a valid layout for @p codebook with the planes it requires.
     */
    LLAMINAR_ROCM_METADATA_HD bool metadataPlanesAdmitted(std::uint8_t codebook, const MetadataPlanes &planes) noexcept
    {
        if (!nativeQuantisedLayoutValid(codebook, packedLayout(planes)) || !planes.primary)
            return false;
        // A coded matrix always owns a superblock plane in the secondary slot.
        return !planes.coded() || planes.secondary != nullptr;
    }

    /**
     * @brief Whether @p planes carry every plane a decoder consumes.
     * @param requires_secondary The per-block decoder reads a secondary word.
     * @param requires_emins The per-block decoder reads a Q2_K emin word.
     *
     * A superblock-coded matrix reconstructs both from its code and
     * superblock planes, so it needs no per-block secondary or emin plane.
     */
    LLAMINAR_ROCM_METADATA_HD bool metadataPlanesComplete(std::uint8_t codebook, const MetadataPlanes &planes,
                                                 bool requires_secondary, bool requires_emins) noexcept
    {
        if (!metadataPlanesAdmitted(codebook, planes))
            return false;
        if (planes.coded())
            return true;
        return (!requires_secondary || planes.secondary) && (!requires_emins || planes.emins);
    }

    /**
     * @brief Descriptor form of metadataPlanesComplete() for grouped/MoE readers.
     * @param desc Prepared expert matrix descriptor.
     * @param requires_secondary The per-block decoder reads a secondary word.
     * @param requires_emins The per-block decoder reads a Q2_K emin word.
     */
    LLAMINAR_ROCM_METADATA_HD bool descriptorMetadataComplete(const DeviceNativeQuantisedMatrixDesc &desc,
                                                                      bool requires_secondary, bool requires_emins) noexcept
    {
        return metadataPlanesComplete(desc.codebook_id, ::llaminar2::native_quantised_scale::metadataPlanes(desc),
                                      requires_secondary, requires_emins);
    }
} // namespace llaminar2::rocm_native_metadata

#undef LLAMINAR_ROCM_METADATA_HD
