/**
 * @file NativeVnniScaleLoad.h
 * @brief Single read path for the prepared NativeVNNI primary scale plane.
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
 * width is owned by nativeVnniScalePlaneBytes(), never by the pointer type.
 */
#pragma once

#include "tensors/MXFP4Format.h"
#include "tensors/NativeVnniScalePlane.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_NVNNI_SCALE_HD __host__ __device__ __forceinline__
#else
#define LLAMINAR_NVNNI_SCALE_HD inline
#endif

namespace llaminar2::native_vnni_scale
{
    /** @brief Reinterpret IEEE-754 binary32 bits as float on host or device. */
    LLAMINAR_NVNNI_SCALE_HD float floatFromBits(std::uint32_t bits)
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
    LLAMINAR_NVNNI_SCALE_HD float e8m0Scale(std::uint8_t e)
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
     *         plane byte: cp.async copies of the raw byte plane must be
     *         normalized by their copy owner before publication.
     */
    template <std::uint8_t Codebook>
    LLAMINAR_NVNNI_SCALE_HD std::uint16_t loadStagedPrimary(const std::uint16_t *plane, std::size_t linear)
    {
        if constexpr (nativeVnniScalePlaneBytes(Codebook) == 1)
            return mxfp4::powerOfTwoBF16Word(reinterpret_cast<const std::uint8_t *>(plane)[linear]);
        else
            return plane[linear];
    }

    /** @brief Runtime-codebook form of loadStagedPrimary(). */
    LLAMINAR_NVNNI_SCALE_HD std::uint16_t loadStagedPrimary(
        std::uint8_t codebook, const std::uint16_t *plane, std::size_t linear)
    {
        return nativeVnniScalePlaneBytes(codebook) == 1
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
    LLAMINAR_NVNNI_SCALE_HD float decodeStagedPrimary(std::uint16_t staged, Widen widen)
    {
        if constexpr (nativeVnniScalePlaneBytes(Codebook) == 1)
            return floatFromBits(std::uint32_t{staged} << 16);
        else
            return widen(staged);
    }

    /** @brief Runtime-codebook form of decodeStagedPrimary(). */
    template <class Widen>
    LLAMINAR_NVNNI_SCALE_HD float decodeStagedPrimary(std::uint8_t codebook, std::uint16_t staged, Widen widen)
    {
        return nativeVnniScalePlaneBytes(codebook) == 1 ? floatFromBits(std::uint32_t{staged} << 16)
                                                        : widen(staged);
    }

    /** @brief Load and decode block @p linear directly from the global plane. */
    template <std::uint8_t Codebook, class Widen>
    LLAMINAR_NVNNI_SCALE_HD float primaryScale(const std::uint16_t *plane, std::size_t linear, Widen widen)
    {
        return decodeStagedPrimary<Codebook>(loadStagedPrimary<Codebook>(plane, linear), widen);
    }

    /** @brief Runtime-codebook form of primaryScale(). */
    template <class Widen>
    LLAMINAR_NVNNI_SCALE_HD float primaryScale(
        std::uint8_t codebook, const std::uint16_t *plane, std::size_t linear, Widen widen)
    {
        return decodeStagedPrimary(codebook, loadStagedPrimary(codebook, plane, linear), widen);
    }
} // namespace llaminar2::native_vnni_scale

#undef LLAMINAR_NVNNI_SCALE_HD
