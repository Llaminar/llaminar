#ifndef LLAMINAR2_TENSORS_NATIVEVNNIFORMATINFO_H
#define LLAMINAR2_TENSORS_NATIVEVNNIFORMATINFO_H

/**
 * @file NativeVnniFormatInfo.h
 * @brief Intrinsic format metadata for native-VNNI quantization formats
 *
 * Describes encoding properties that are invariant for each quantization format:
 * payload size, symmetry, super-block layout, and magnitude bounds. These are
 * returned by IINT8Unpackable::vnniFormatInfo() on each tensor class, replacing
 * the external dispatch table.
 *
 * Codebook IDs match the NativeVNNIFormat enum in ROCmGemvKernel_native_VNNI.hip.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace llaminar2
{
    /**
     * @brief Metadata descriptor for a native-VNNI quantization format
     *
     * Each native-VNNI format (≤6-bit) has fixed metadata that determines:
     * - How the GEMV/GEMM GPU kernel decodes the payload (codebook_id)
     * - How many bytes each block's payload occupies (payload_bytes)
     * - Whether min-value correction is needed (is_asymmetric)
     * - Whether 256→8×32 sub-block decomposition is needed (is_superblock)
     * - Conservative element magnitude for INT8 re-quantization (max_abs_factor)
     */
    struct NativeVnniFormatInfo
    {
        uint8_t codebook_id;  ///< Kernel dispatch ID (0=Q4_0, 4=IQ4_NL, 5=Q4_1, ...)
        int payload_bytes;    ///< Bytes per 32-element block payload
        bool is_asymmetric;   ///< True if format has min-value offset
        bool is_superblock;   ///< True if format has 256-element super-blocks (8×32)
        bool has_emins;       ///< True if format needs separate embedded mins (Q2_K only)
        float max_abs_factor; ///< Conservative max |element| / scale
    };

    /**
     * @brief Byte lengths of the four persistent native-VNNI storage regions.
     *
     * GPU preparation stores each logical 32-value execution block as one
     * compressed payload plus FP16 scale metadata. Asymmetric formats add an
     * FP16 minimum and Q2_K adds one packed effective-minimum word. Keeping
     * this calculation beside the format catalog makes the loader, memory
     * planner, and tests consume exactly the same storage contract.
     */
    struct NativeVnniPackedRegionSizes
    {
        size_t payload_bytes = 0;
        size_t scales_bytes = 0;
        size_t mins_bytes = 0;
        size_t emins_bytes = 0;

        [[nodiscard]] constexpr size_t logicalBytes() const noexcept
        {
            return payload_bytes + scales_bytes + mins_bytes + emins_bytes;
        }
    };

    /**
     * @brief Canonical native-VNNI format catalog.
     *
     * These constants are the sole host-side declaration of payload geometry.
     * Tensor classes advertise one of these entries, the device loader uses
     * the advertised entry to allocate its pool, and preflight planning looks
     * up the same entry by GGUF format name.
     */
    namespace native_vnni_formats
    {
        inline constexpr NativeVnniFormatInfo IQ4_NL{
            4, 16, false, false, false, 127.0f};
        inline constexpr NativeVnniFormatInfo Q8_0{
            19, 32, false, false, false, 127.0f};
        inline constexpr NativeVnniFormatInfo Q8_1{
            20, 32, false, false, false, 127.0f};
        inline constexpr NativeVnniFormatInfo Q4_0{
            0, 16, false, false, false, 8.0f};
        inline constexpr NativeVnniFormatInfo Q4_1{
            5, 16, true, false, false, 15.0f};
        inline constexpr NativeVnniFormatInfo Q5_0{
            6, 20, false, false, false, 16.0f};
        inline constexpr NativeVnniFormatInfo Q5_1{
            7, 20, true, false, false, 31.0f};
        inline constexpr NativeVnniFormatInfo Q6_K{
            8, 24, true, true, false, 32.0f};
        inline constexpr NativeVnniFormatInfo Q2_K{
            10, 8, true, true, true, 3.0f};
        inline constexpr NativeVnniFormatInfo Q5_K{
            7, 20, true, true, false, 31.0f};
        inline constexpr NativeVnniFormatInfo Q3_K{
            9, 12, true, true, false, 4.0f};
        inline constexpr NativeVnniFormatInfo Q4_K{
            5, 16, true, true, false, 15.0f};
        inline constexpr NativeVnniFormatInfo Q8_K{
            21, 32, false, true, false, 127.0f};
        inline constexpr NativeVnniFormatInfo IQ4_XS{
            4, 16, false, true, false, 127.0f};
        inline constexpr NativeVnniFormatInfo IQ2_XXS{
            15, 8, false, true, false, 43.0f};
        inline constexpr NativeVnniFormatInfo IQ2_XS{
            14, 9, true, true, false, 43.0f};
        inline constexpr NativeVnniFormatInfo IQ3_XXS{
            12, 12, false, true, false, 62.0f};
        inline constexpr NativeVnniFormatInfo IQ2_S{
            13, 9, true, true, false, 43.0f};
        inline constexpr NativeVnniFormatInfo IQ3_S{
            11, 13, false, true, false, 15.0f};
        inline constexpr NativeVnniFormatInfo IQ1_S{
            16, 6, true, true, false, 1.125f};
        inline constexpr NativeVnniFormatInfo IQ1_M{
            17, 6, true, true, false, 1.125f};

        /**
         * @brief Resolve a GGUF quantization name to its canonical descriptor.
         *
         * Family suffixes such as Q4_K_S and Q5_K_M describe model-level
         * quantization choices while retaining the same execution block
         * geometry, so they intentionally map to the common family entry.
         */
        [[nodiscard]] inline constexpr const NativeVnniFormatInfo *
        forQuantType(std::string_view quant_type) noexcept
        {
            if (quant_type == "IQ4_NL")
                return &IQ4_NL;
            if (quant_type == "Q8_0")
                return &Q8_0;
            if (quant_type == "Q8_1")
                return &Q8_1;
            if (quant_type == "Q4_0")
                return &Q4_0;
            if (quant_type == "Q4_1")
                return &Q4_1;
            if (quant_type == "Q5_0")
                return &Q5_0;
            if (quant_type == "Q5_1")
                return &Q5_1;
            if (quant_type == "Q6_K")
                return &Q6_K;
            if (quant_type == "Q2_K")
                return &Q2_K;
            if (quant_type == "Q5_K" ||
                quant_type == "Q5_K_S" ||
                quant_type == "Q5_K_M")
                return &Q5_K;
            if (quant_type == "Q3_K" ||
                quant_type == "Q3_K_S" ||
                quant_type == "Q3_K_M" ||
                quant_type == "Q3_K_L")
                return &Q3_K;
            if (quant_type == "Q4_K" ||
                quant_type == "Q4_K_S" ||
                quant_type == "Q4_K_M")
                return &Q4_K;
            if (quant_type == "Q8_K")
                return &Q8_K;
            if (quant_type == "IQ4_XS")
                return &IQ4_XS;
            if (quant_type == "IQ2_XXS")
                return &IQ2_XXS;
            if (quant_type == "IQ2_XS")
                return &IQ2_XS;
            if (quant_type == "IQ3_XXS")
                return &IQ3_XXS;
            if (quant_type == "IQ2_S")
                return &IQ2_S;
            if (quant_type == "IQ3_S")
                return &IQ3_S;
            if (quant_type == "IQ1_S")
                return &IQ1_S;
            if (quant_type == "IQ1_M")
                return &IQ1_M;
            return nullptr;
        }
    } // namespace native_vnni_formats

    /**
     * @brief Compute the exact logical pool regions for one packed matrix.
     *
     * Native-VNNI kernels require K to consist of complete 32-value execution
     * blocks. The loader enforces that shape contract before calling the pool;
     * this pure helper mirrors the resulting persistent allocation sizes.
     */
    [[nodiscard]] inline constexpr NativeVnniPackedRegionSizes
    nativeVnniPackedRegionSizes(
        size_t rows,
        size_t columns,
        const NativeVnniFormatInfo &format) noexcept
    {
        const size_t block_count = rows * (columns / 32);
        return {
            .payload_bytes =
                block_count * static_cast<size_t>(format.payload_bytes),
            .scales_bytes = block_count * sizeof(uint16_t),
            .mins_bytes =
                format.is_asymmetric ? block_count * sizeof(uint16_t) : 0,
            .emins_bytes =
                format.has_emins ? block_count * sizeof(uint32_t) : 0,
        };
    }

} // namespace llaminar2

#endif // LLAMINAR2_TENSORS_NATIVEVNNIFORMATINFO_H
