/**
 * @file NativeVnniScaleLoad.h
 * @brief Single read path for the prepared NativeVNNI primary scale plane.
 *
 * Three separate steps, so staged copies never reuse plane addressing:
 * loadEncodedPrimary() reads the stored element from the global plane into a
 * 16-bit carrier; staging (registers, LDS, cp.async tiles) moves that carrier
 * unchanged; decodePrimary() turns a carrier into the FP32 scale consumed by
 * the existing FP32 contribution contract. primaryScale() composes the two
 * for direct global reads.
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
     * @brief FP32 scale of a native E8M0 plane byte.
     * @param e Stored byte: 0 marks an all-zero block, otherwise e in [2, 254].
     */
    LLAMINAR_NVNNI_SCALE_HD float e8m0Scale(std::uint8_t e)
    {
        return e == 0 ? 0.0f : floatFromBits(mxfp4::normalScaleFP32Bits(e));
    }

    /**
     * @brief Load one encoded primary-scale element from the global plane.
     * @tparam Codebook Physical execution codebook (never the arithmetic policy).
     * @param plane Primary scale plane base address.
     * @param linear Block-major linear block index.
     * @return A 16-bit carrier: FP16 bits for 2-byte planes, the zero-extended
     *         E8M0 byte for 1-byte planes. Staging (registers, LDS, cp.async
     *         tiles) moves this carrier; only decodePrimary() interprets it.
     */
    template <std::uint8_t Codebook>
    LLAMINAR_NVNNI_SCALE_HD std::uint16_t loadEncodedPrimary(const std::uint16_t *plane, std::size_t linear)
    {
        if constexpr (nativeVnniScalePlaneBytes(Codebook) == 1)
            return reinterpret_cast<const std::uint8_t *>(plane)[linear];
        else
            return plane[linear];
    }

    /** @brief Runtime-codebook form of loadEncodedPrimary(). */
    LLAMINAR_NVNNI_SCALE_HD std::uint16_t loadEncodedPrimary(
        std::uint8_t codebook, const std::uint16_t *plane, std::size_t linear)
    {
        return nativeVnniScalePlaneBytes(codebook) == 1
                   ? std::uint16_t{reinterpret_cast<const std::uint8_t *>(plane)[linear]}
                   : plane[linear];
    }

    /**
     * @brief Decode an encoded carrier (from a plane or from staging) to the FP32 scale.
     * @tparam Codebook Physical execution codebook that produced @p encoded.
     * @param encoded Carrier returned by loadEncodedPrimary(), possibly staged.
     * @param widen Caller's backend FP16-bits-to-FP32 widening (2-byte planes only),
     *        so existing expressions and reduction order are preserved.
     */
    template <std::uint8_t Codebook, class Widen>
    LLAMINAR_NVNNI_SCALE_HD float decodePrimary(std::uint16_t encoded, Widen widen)
    {
        if constexpr (nativeVnniScalePlaneBytes(Codebook) == 1)
            return e8m0Scale(static_cast<std::uint8_t>(encoded));
        else
            return widen(encoded);
    }

    /** @brief Runtime-codebook form of decodePrimary(). */
    template <class Widen>
    LLAMINAR_NVNNI_SCALE_HD float decodePrimary(std::uint8_t codebook, std::uint16_t encoded, Widen widen)
    {
        return nativeVnniScalePlaneBytes(codebook) == 1 ? e8m0Scale(static_cast<std::uint8_t>(encoded))
                                                        : widen(encoded);
    }

    /** @brief Load and decode block @p linear directly from the global plane. */
    template <std::uint8_t Codebook, class Widen>
    LLAMINAR_NVNNI_SCALE_HD float primaryScale(const std::uint16_t *plane, std::size_t linear, Widen widen)
    {
        return decodePrimary<Codebook>(loadEncodedPrimary<Codebook>(plane, linear), widen);
    }

    /** @brief Runtime-codebook form of primaryScale(). */
    template <class Widen>
    LLAMINAR_NVNNI_SCALE_HD float primaryScale(
        std::uint8_t codebook, const std::uint16_t *plane, std::size_t linear, Widen widen)
    {
        return decodePrimary(codebook, loadEncodedPrimary(codebook, plane, linear), widen);
    }
} // namespace llaminar2::native_vnni_scale

#undef LLAMINAR_NVNNI_SCALE_HD
