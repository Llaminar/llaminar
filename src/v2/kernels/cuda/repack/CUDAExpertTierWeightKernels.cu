/**
 * @file CUDAExpertTierWeightKernels.cu
 * @brief CUDA kernels for direct GPU/CPU ExpertOverlay weight streaming.
 *
 * One thread block owns one complete 64-column by 32-K CPU unit. This gives
 * every output byte exactly one writer, including Q6_K high-bit planes, and
 * makes arbitrary ordered chunks independent. Padding bytes are initialized
 * deterministically so the transmitted CPU representation is byte-exact.
 */

#include "kernels/cuda/repack/CUDAExpertTierWeightKernels.h"
#include "kernels/cuda/gemm/CUDANativeVNNIDecodeCommon.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace llaminar2
{
    namespace
    {
        constexpr int kColumnsPerUnit = 64;

        __device__ __constant__ std::int8_t kTierIQ4ValuesCUDA[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10,
            1, 13, 25, 38, 53, 69, 89, 113};

        /** Return the byte offset of one column/value in CPU interleaving. */
        __device__ __forceinline__ std::size_t interleavedValueOffset(
            int local_column,
            int value_index)
        {
            const int group = value_index / 4;
            const int index = value_index % 4;
            const int zmm = local_column / 16;
            const int lane = local_column % 16;
            return static_cast<std::size_t>(group) * 256u +
                   static_cast<std::size_t>(zmm) * 64u +
                   static_cast<std::size_t>(lane) * 4u +
                   static_cast<std::size_t>(index);
        }

        /** Initialize every byte, including padded columns and metadata gaps. */
        __device__ __forceinline__ void initializeCpuUnit(
            std::uint8_t *unit,
            std::uint32_t stride)
        {
            for (std::uint32_t offset = threadIdx.x; offset < stride;
                 offset += blockDim.x)
            {
                unit[offset] = 0;
            }
        }

        /** Decode one nibble solely for the CPU compensation vector. */
        template <std::uint8_t Codebook>
        __device__ __forceinline__ int decodeNibbleForCompensation(
            std::uint8_t value)
        {
            if constexpr (Codebook == 0)
                return static_cast<int>(value) - 8;
            if constexpr (Codebook == 4)
                return static_cast<int>(kTierIQ4ValuesCUDA[value]);
            return static_cast<int>(value);
        }

        /** GPU-separated nibble projection to final CPU unit bytes. */
        template <std::uint8_t Codebook, bool Asymmetric>
        __global__ void gpuToCpuNibbleUnitsCUDA(
            const std::uint8_t *__restrict__ payload,
            const std::uint16_t *__restrict__ scales,
            const std::uint16_t *__restrict__ mins,
            int N,
            int blocks_per_row,
            std::uint32_t first_unit,
            std::uint32_t cpu_stride,
            std::uint8_t *__restrict__ cpu_chunk)
        {
            const std::uint32_t global_unit = first_unit + blockIdx.x;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb =
                static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x);
            std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(blockIdx.x) * cpu_stride;

            initializeCpuUnit(unit, cpu_stride);
            __syncthreads();

            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (n >= N)
                return;
            const std::size_t linear =
                static_cast<std::size_t>(kb) * N + n;
            const std::uint8_t *source = payload + linear * 16u;
            int compensation = 0;
            for (int value = 0; value < 16; ++value)
            {
                const std::uint8_t packed = source[value];
                unit[interleavedValueOffset(local_column, value)] = packed;
                compensation += decodeNibbleForCompensation<Codebook>(
                    static_cast<std::uint8_t>(packed & 0x0fu));
                compensation += decodeNibbleForCompensation<Codebook>(
                    static_cast<std::uint8_t>(packed >> 4));
            }
            reinterpret_cast<std::int16_t *>(unit + 1024u)[local_column] =
                static_cast<std::int16_t>(compensation);
            reinterpret_cast<std::uint16_t *>(unit + 1152u)[local_column] =
                scales[linear];
            if constexpr (Asymmetric)
            {
                reinterpret_cast<std::uint16_t *>(unit + 1280u)[local_column] =
                    mins[linear];
            }
        }

        /**
         * @brief Extract one signed decoded byte from eight packed DP4A groups.
         * @param groups NativeVNNI decoder output, four values per group.
         * @param value_index Logical value in the 32-element execution block.
         * @return Sign-extended quantized value.
         */
        __device__ __forceinline__ int decodedGroupValue(
            const std::int32_t (&groups)[8],
            int value_index)
        {
            const std::uint32_t packed =
                static_cast<std::uint32_t>(groups[value_index / 4]);
            const std::uint8_t byte = static_cast<std::uint8_t>(
                packed >> (8 * (value_index % 4)));
            return static_cast<int>(static_cast<std::int8_t>(byte));
        }

        /**
         * @brief Convert FP16 storage bits to FP32 without changing the bits.
         * @param bits Serialized scale/minimum metadata.
         * @return FP32 value used by CPU-preparation arithmetic.
         */
        __device__ __forceinline__ float tierHalfToFloat(std::uint16_t bits)
        {
            return __half2float(*reinterpret_cast<const __half *>(&bits));
        }

        /**
         * @brief Round one FP32 metadata value to production FP16 storage.
         * @param value Final scale or minimum computed by the transcode.
         * @return IEEE FP16 bits using round-to-nearest-even.
         */
        __device__ __forceinline__ std::uint16_t tierFloatToHalf(float value)
        {
            return __half_as_ushort(__float2half_rn(value));
        }

        /**
         * @brief Reconstruct one source-format value from decoded integer data.
         * @tparam Codebook Accelerator execution codebook.
         * @param value_index Position in the logical 32-value block.
         * @param quantized Integer decoded by the production NativeVNNI decoder.
         * @param payload Original compact payload, needed for IQ1_M deltas.
         * @param primary Low/whole-block scale.
         * @param secondary High-half scale or additive minimum.
         * @param emins Packed Q2_K low/high additive minima.
         * @return FP32 value represented by the accelerator source arrays.
         *
         * Explicit rounded operations keep the conversion independent of
         * compiler FMA choices.  This is preparation work, not GEMM arithmetic,
         * but the resulting INT8 bytes must still be deterministic.
         */
        template <std::uint8_t Codebook>
        __device__ __forceinline__ float sourceValue(
            int value_index,
            int quantized,
            const std::uint8_t *payload,
            float primary,
            float secondary,
            std::uint32_t emins)
        {
            const float q = static_cast<float>(quantized);
            if constexpr (Codebook == 9 || Codebook == 13 || Codebook == 14)
            {
                const float scale = value_index < 16 ? primary : secondary;
                return __fmul_rn(q, scale);
            }
            else if constexpr (Codebook == 10)
            {
                const float scale = value_index < 16 ? primary : secondary;
                const std::uint16_t minimum_bits =
                    value_index < 16
                        ? static_cast<std::uint16_t>(emins)
                        : static_cast<std::uint16_t>(emins >> 16);
                return __fadd_rn(
                    __fmul_rn(q, scale),
                    tierHalfToFloat(minimum_bits));
            }
            else if constexpr (Codebook == 16)
            {
                return __fadd_rn(__fmul_rn(q, primary), secondary);
            }
            else if constexpr (Codebook == 17)
            {
                const float scale = value_index < 16 ? primary : secondary;
                const int subgroup = value_index / 8;
                const std::uint8_t qh = payload[4 + subgroup / 2];
                const int delta_bit = (subgroup & 1) == 0 ? 3 : 7;
                constexpr float kDelta = 0.125f;
                const float delta = (qh & (1u << delta_bit))
                                        ? -kDelta
                                        : kDelta;
                return __fmul_rn(__fadd_rn(q, delta), scale);
            }
            else
            {
                return __fmul_rn(q, primary);
            }
        }

        /** Clamp a prepared signed-byte value to the CPU execution interval. */
        __device__ __forceinline__ std::int8_t clampPreparedInt8(int value)
        {
            value = value < -128 ? -128 : value;
            value = value > 127 ? 127 : value;
            return static_cast<std::int8_t>(value);
        }

        /**
         * @brief Build the canonical Q16 multiplier for an IQ2 grid half.
         * @param factor Published source scale for this sixteen-value half.
         * @param inverse_scale Reciprocal of the destination Q8 scale.
         * @return Positive Q16 multiplier rounded before integer application.
         *
         * CPU AVX-512 preparation intentionally rounds only this multiplier;
         * the later integer multiply is truncated. Keeping those two edges
         * separate prevents GPU demotion from silently reverting to FP32
         * round-to-nearest and changing migrated weights by one byte.
         */
        __device__ __forceinline__ std::int32_t tierIQ2Q16RatioCUDA(
            float factor,
            float inverse_scale)
        {
            const float scaled = __fmul_rn(
                __fmul_rn(factor, inverse_scale), 65536.0f);
            return static_cast<std::int32_t>(__fadd_rn(scaled, 0.5f));
        }

        /**
         * @brief Apply one canonical Q16 IQ2 transcode while retaining sign.
         * @param signed_grid_value Signed integer decoded from an IQ2 grid.
         * @param ratio Positive Q16 multiplier for the value's source half.
         * @return Signed Q8 payload byte in an integer container.
         *
         * The unsigned shift exactly mirrors AVX-512 `mullo` followed by
         * `srli`; sign application deliberately happens after truncation.
         */
        __device__ __forceinline__ int applyTierIQ2Q16RatioCUDA(
            int signed_grid_value,
            std::int32_t ratio)
        {
            const std::uint32_t magnitude = static_cast<std::uint32_t>(
                signed_grid_value < 0 ? -signed_grid_value : signed_grid_value);
            const std::uint32_t prepared =
                (magnitude * static_cast<std::uint32_t>(ratio)) >> 16;
            return signed_grid_value < 0
                       ? -static_cast<int>(prepared)
                       : static_cast<int>(prepared);
        }

        /**
         * @brief Transcode one non-reversible source codebook into final CPU
         * expanded-INT8 unit bytes.
         * @tparam Codebook One of 6, 7, 9-17, raw-INT8 19, or normalized
         *         asymmetric INT8+minimum 23.
         *
         * Each thread owns one logical output column.  The shared production
         * decoder reconstructs source integers; direct Q5/Q8 formats retain
         * those integers, IQ formats are requantized to Q8_0, and Q3_K/Q2_K
         * use the CPU packer's affine two-half transcode.  Padded columns remain
         * deterministic zeros from `initializeCpuUnit`.
         */
        template <std::uint8_t Codebook>
        __global__ void gpuToCpuDecodedUnitsCUDA(
            const std::uint8_t *__restrict__ payload,
            const std::uint16_t *__restrict__ scales,
            const std::uint16_t *__restrict__ mins,
            const std::uint32_t *__restrict__ emins,
            int N,
            int blocks_per_row,
            std::uint32_t first_unit,
            std::uint32_t cpu_stride,
            std::uint8_t *__restrict__ cpu_chunk)
        {
            const std::uint32_t global_unit = first_unit + blockIdx.x;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb = static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x);
            std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(blockIdx.x) * cpu_stride;

            initializeCpuUnit(unit, cpu_stride);
            __syncthreads();

            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (n >= N)
                return;
            const std::size_t linear =
                static_cast<std::size_t>(kb) * N + n;
            constexpr int payload_bytes =
                cuda_native_vnni::CodebookTraits<Codebook>::payload_bytes;
            const std::uint8_t *source =
                payload + linear * static_cast<std::size_t>(payload_bytes);
            std::int32_t decoded_groups[8]{};
            cuda_native_vnni::decode_groups<Codebook>(
                source, decoded_groups);

            const float primary = tierHalfToFloat(scales[linear]);
            const float secondary =
                mins == nullptr ? 0.0f : tierHalfToFloat(mins[linear]);
            const std::uint32_t effective_mins =
                emins == nullptr ? 0u : emins[linear];
            float output_scale = primary;
            float output_minimum = secondary;

            if constexpr (Codebook == 9 || Codebook == 10)
            {
                // Q3_K and Q2_K have independent half-block affine domains.
                // Reproduce the CPU packer's theoretical union so byte output
                // does not depend on whether extrema occur in this payload.
                float minimum0 = 0.0f;
                float maximum0 = 0.0f;
                float minimum1 = 0.0f;
                float maximum1 = 0.0f;
                if constexpr (Codebook == 9)
                {
                    minimum0 = primary >= 0.0f
                                   ? __fmul_rn(primary, -4.0f)
                                   : __fmul_rn(primary, 3.0f);
                    maximum0 = primary >= 0.0f
                                   ? __fmul_rn(primary, 3.0f)
                                   : __fmul_rn(primary, -4.0f);
                    minimum1 = secondary >= 0.0f
                                   ? __fmul_rn(secondary, -4.0f)
                                   : __fmul_rn(secondary, 3.0f);
                    maximum1 = secondary >= 0.0f
                                   ? __fmul_rn(secondary, 3.0f)
                                   : __fmul_rn(secondary, -4.0f);
                }
                else
                {
                    minimum0 = tierHalfToFloat(
                        static_cast<std::uint16_t>(effective_mins));
                    minimum1 = tierHalfToFloat(
                        static_cast<std::uint16_t>(effective_mins >> 16));
                    maximum0 = __fadd_rn(
                        __fmul_rn(primary, 3.0f), minimum0);
                    maximum1 = __fadd_rn(
                        __fmul_rn(secondary, 3.0f), minimum1);
                }
                const float global_minimum = fminf(minimum0, minimum1);
                const float global_maximum = fmaxf(maximum0, maximum1);
                const float range =
                    __fsub_rn(global_maximum, global_minimum);
                if (range < 1.0e-5f)
                {
                    output_scale = 0.0f;
                    output_minimum = global_minimum;
                    for (int value = 0; value < 32; ++value)
                    {
                        unit[interleavedValueOffset(
                            local_column, value)] = 0x80u;
                    }
                }
                else
                {
                    output_scale = __fdiv_rn(range, 255.0f);
                    output_minimum = __fadd_rn(
                        global_minimum,
                        __fmul_rn(128.0f, output_scale));
                    const float inverse_scale =
                        __fdiv_rn(1.0f, output_scale);
                    const float bias = __fmul_rn(
                        -output_minimum, inverse_scale);
                    for (int value = 0; value < 32; ++value)
                    {
                        const float actual = sourceValue<Codebook>(
                            value,
                            decodedGroupValue(decoded_groups, value),
                            source,
                            primary,
                            secondary,
                            effective_mins);
                        const int quantized = __float2int_rn(__fadd_rn(
                            __fmul_rn(actual, inverse_scale), bias));
                        unit[interleavedValueOffset(local_column, value)] =
                            static_cast<std::uint8_t>(
                                clampPreparedInt8(quantized));
                    }
                }
            }
            else if constexpr (Codebook == 13 || Codebook == 15)
            {
                // IQ2_S and IQ2_XXS use the CPU AVX-512 Q16 transcode
                // contract. Find the integer grid maxima first, round one Q16
                // ratio per half, then truncate each integer product before
                // applying its sign. This remains one thread per output column
                // and adds no synchronization or staging traffic.
                int maximum_grid_low = 0;
                int maximum_grid_high = 0;
                for (int value = 0; value < 32; ++value)
                {
                    const int signed_grid =
                        decodedGroupValue(decoded_groups, value);
                    const int magnitude =
                        signed_grid < 0 ? -signed_grid : signed_grid;
                    if (value < 16)
                        maximum_grid_low =
                            magnitude > maximum_grid_low
                                ? magnitude
                                : maximum_grid_low;
                    else
                        maximum_grid_high =
                            magnitude > maximum_grid_high
                                ? magnitude
                                : maximum_grid_high;
                }

                const float factor_low = primary;
                const float factor_high =
                    Codebook == 13 ? secondary : primary;
                const float maximum_low = __fmul_rn(
                    factor_low, static_cast<float>(maximum_grid_low));
                const float maximum_high = __fmul_rn(
                    factor_high, static_cast<float>(maximum_grid_high));
                const float maximum_absolute =
                    fmaxf(maximum_low, maximum_high);
                output_minimum = 0.0f;
                if (maximum_absolute < 1.0e-6f)
                {
                    output_scale = 0.0f;
                    for (int value = 0; value < 32; ++value)
                    {
                        unit[interleavedValueOffset(
                            local_column, value)] = 0u;
                    }
                }
                else
                {
                    output_scale = fminf(
                        __fdiv_rn(maximum_absolute, 127.0f),
                        65504.0f);
                    const float inverse_scale =
                        __fdiv_rn(1.0f, output_scale);
                    const std::int32_t ratio_low =
                        tierIQ2Q16RatioCUDA(factor_low, inverse_scale);
                    const std::int32_t ratio_high =
                        tierIQ2Q16RatioCUDA(factor_high, inverse_scale);
                    for (int value = 0; value < 32; ++value)
                    {
                        const int quantized = applyTierIQ2Q16RatioCUDA(
                            decodedGroupValue(decoded_groups, value),
                            value < 16 ? ratio_low : ratio_high);
                        unit[interleavedValueOffset(local_column, value)] =
                            static_cast<std::uint8_t>(
                                static_cast<std::int8_t>(quantized));
                    }
                }
            }
            else if constexpr (
                Codebook == 11 || Codebook == 12 || Codebook == 14 ||
                Codebook == 16 || Codebook == 17)
            {
                // IQ formats are normalized exactly as Q8_0: choose one
                // block-wide absmax scale, then round each represented value.
                float maximum_absolute = 0.0f;
                for (int value = 0; value < 32; ++value)
                {
                    const float actual = sourceValue<Codebook>(
                        value,
                        decodedGroupValue(decoded_groups, value),
                        source,
                        primary,
                        secondary,
                        effective_mins);
                    maximum_absolute =
                        fmaxf(maximum_absolute, fabsf(actual));
                }
                output_minimum = 0.0f;
                if (maximum_absolute < 1.0e-6f)
                {
                    output_scale = 0.0f;
                    for (int value = 0; value < 32; ++value)
                    {
                        unit[interleavedValueOffset(
                            local_column, value)] = 0u;
                    }
                }
                else
                {
                    output_scale = fminf(
                        __fdiv_rn(maximum_absolute, 127.0f),
                        65504.0f);
                    const float inverse_scale =
                        __fdiv_rn(1.0f, output_scale);
                    for (int value = 0; value < 32; ++value)
                    {
                        const float actual = sourceValue<Codebook>(
                            value,
                            decodedGroupValue(decoded_groups, value),
                            source,
                            primary,
                            secondary,
                            effective_mins);
                        int quantized = __float2int_rn(
                            __fmul_rn(actual, inverse_scale));
                        quantized = quantized < -127 ? -127 : quantized;
                        quantized = quantized > 127 ? 127 : quantized;
                        unit[interleavedValueOffset(local_column, value)] =
                            static_cast<std::uint8_t>(
                                static_cast<std::int8_t>(quantized));
                    }
                }
            }
            else
            {
                // Q5_0/Q5_1/Q5_K and raw Q8 already use the CPU's integer
                // domain, so conversion is only a deterministic transpose.
                for (int value = 0; value < 32; ++value)
                {
                    unit[interleavedValueOffset(local_column, value)] =
                        static_cast<std::uint8_t>(static_cast<std::int8_t>(
                            decodedGroupValue(decoded_groups, value)));
                }
            }

            // CPU VNNI kernels shift activations to unsigned and require the
            // exact signed-weight sum for every expanded block.
            int compensation = 0;
            for (int value = 0; value < 32; ++value)
            {
                compensation += static_cast<std::int8_t>(
                    unit[interleavedValueOffset(local_column, value)]);
            }
            reinterpret_cast<std::int16_t *>(unit + 2048u)[local_column] =
                static_cast<std::int16_t>(compensation);
            reinterpret_cast<std::uint16_t *>(unit + 2176u)[local_column] =
                tierFloatToHalf(output_scale);
            if constexpr (
                Codebook == 7 || Codebook == 9 || Codebook == 10 ||
                Codebook == 13 || Codebook == 14 || Codebook == 16 ||
                Codebook == 17 ||
                Codebook == kNativeVnniExpandedInt8MinCodebook)
            {
                reinterpret_cast<std::uint16_t *>(unit + 2304u)[local_column] =
                    tierFloatToHalf(output_minimum);
            }
        }

        /** GPU-separated native Q6_K projection to CPU dual-scale unit bytes. */
        __global__ void gpuToCpuQ6UnitsCUDA(
            const std::uint8_t *__restrict__ payload,
            const std::uint16_t *__restrict__ scales,
            const std::uint16_t *__restrict__ secondary_scales,
            int N,
            int blocks_per_row,
            std::uint32_t first_unit,
            std::uint32_t cpu_stride,
            std::uint8_t *__restrict__ cpu_chunk)
        {
            const std::uint32_t global_unit = first_unit + blockIdx.x;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb =
                static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x);
            std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(blockIdx.x) * cpu_stride;

            initializeCpuUnit(unit, cpu_stride);
            __syncthreads();

            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (n < N)
            {
                const std::size_t linear =
                    static_cast<std::size_t>(kb) * N + n;
                const std::uint8_t *source = payload + linear * 24u;
                for (int value = 0; value < 16; ++value)
                {
                    unit[interleavedValueOffset(local_column, value)] =
                        source[value];
                }
                reinterpret_cast<std::uint16_t *>(unit + 1536u)[local_column] =
                    scales[linear];
                reinterpret_cast<std::uint16_t *>(unit + 1664u)[local_column] =
                    secondary_scales[linear];
            }

            if (threadIdx.x < 32)
            {
                const int group = static_cast<int>(threadIdx.x) / 4;
                const int zmm = static_cast<int>(threadIdx.x) % 4;
                std::uint64_t low_plane = 0;
                std::uint64_t high_plane = 0;
                for (int lane = 0; lane < 16; ++lane)
                {
                    const int plane_n =
                        n_chunk * kColumnsPerUnit + zmm * 16 + lane;
                    if (plane_n >= N)
                        continue;
                    const std::size_t linear =
                        static_cast<std::size_t>(kb) * N + plane_n;
                    const std::uint8_t packed_high =
                        payload[linear * 24u + 16u + group];
                    for (int index = 0; index < 4; ++index)
                    {
                        const std::uint64_t bit =
                            static_cast<std::uint64_t>(lane * 4 + index);
                        const std::uint8_t high = static_cast<std::uint8_t>(
                            (packed_high >> (index * 2)) & 0x03u);
                        low_plane |= static_cast<std::uint64_t>(high & 1u) << bit;
                        high_plane |= static_cast<std::uint64_t>(high >> 1u) << bit;
                    }
                }
                std::uint64_t *planes = reinterpret_cast<std::uint64_t *>(
                    unit + 1024u + static_cast<std::size_t>(group) * 64u +
                    static_cast<std::size_t>(zmm) * 16u);
                planes[0] = low_plane;
                planes[1] = high_plane;
            }
        }

        /** CPU nibble units to separated GPU projection. */
        template <bool Asymmetric>
        __global__ void cpuToGpuNibbleUnitsCUDA(
            const std::uint8_t *__restrict__ cpu_chunk,
            int N,
            int blocks_per_row,
            std::uint32_t first_unit,
            std::uint32_t cpu_stride,
            std::uint8_t *__restrict__ payload,
            std::uint16_t *__restrict__ scales,
            std::uint16_t *__restrict__ mins)
        {
            const std::uint32_t global_unit = first_unit + blockIdx.x;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb =
                static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x);
            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (n >= N)
                return;

            const std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(blockIdx.x) * cpu_stride;
            const std::size_t linear =
                static_cast<std::size_t>(kb) * N + n;
            std::uint8_t *destination = payload + linear * 16u;
            for (int value = 0; value < 16; ++value)
            {
                destination[value] =
                    unit[interleavedValueOffset(local_column, value)];
            }
            scales[linear] =
                reinterpret_cast<const std::uint16_t *>(unit + 1152u)[local_column];
            if constexpr (Asymmetric)
            {
                mins[linear] = reinterpret_cast<const std::uint16_t *>(
                    unit + 1280u)[local_column];
            }
        }

        /** CPU expanded INT8 units to separated GPU projection. */
        template <bool Asymmetric>
        __global__ void cpuToGpuExpandedUnitsCUDA(
            const std::uint8_t *__restrict__ cpu_chunk,
            int N,
            int blocks_per_row,
            std::uint32_t first_unit,
            std::uint32_t cpu_stride,
            std::uint8_t *__restrict__ payload,
            std::uint16_t *__restrict__ scales,
            std::uint16_t *__restrict__ mins)
        {
            const std::uint32_t global_unit = first_unit + blockIdx.x;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb =
                static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x);
            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (n >= N)
                return;

            const std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(blockIdx.x) * cpu_stride;
            const std::size_t linear =
                static_cast<std::size_t>(kb) * N + n;
            std::uint8_t *destination = payload + linear * 32u;
            for (int value = 0; value < 32; ++value)
            {
                destination[value] =
                    unit[interleavedValueOffset(local_column, value)];
            }
            scales[linear] =
                reinterpret_cast<const std::uint16_t *>(unit + 2176u)[local_column];
            if constexpr (Asymmetric)
            {
                mins[linear] = reinterpret_cast<const std::uint16_t *>(
                    unit + 2304u)[local_column];
            }
        }

        /** CPU native Q6_K units to separated GPU projection. */
        __global__ void cpuToGpuQ6UnitsCUDA(
            const std::uint8_t *__restrict__ cpu_chunk,
            int N,
            int blocks_per_row,
            std::uint32_t first_unit,
            std::uint32_t cpu_stride,
            std::uint8_t *__restrict__ payload,
            std::uint16_t *__restrict__ scales,
            std::uint16_t *__restrict__ secondary_scales)
        {
            const std::uint32_t global_unit = first_unit + blockIdx.x;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb =
                static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x);
            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (n >= N)
                return;

            const std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(blockIdx.x) * cpu_stride;
            const std::size_t linear =
                static_cast<std::size_t>(kb) * N + n;
            std::uint8_t *destination = payload + linear * 24u;
            for (int value = 0; value < 16; ++value)
            {
                destination[value] =
                    unit[interleavedValueOffset(local_column, value)];
            }
            const int zmm = local_column / 16;
            const int lane = local_column % 16;
            for (int group = 0; group < 8; ++group)
            {
                const std::uint64_t *planes =
                    reinterpret_cast<const std::uint64_t *>(
                        unit + 1024u + static_cast<std::size_t>(group) * 64u +
                        static_cast<std::size_t>(zmm) * 16u);
                std::uint8_t packed_high = 0;
                for (int index = 0; index < 4; ++index)
                {
                    const std::uint64_t bit =
                        static_cast<std::uint64_t>(lane * 4 + index);
                    const std::uint8_t high = static_cast<std::uint8_t>(
                        ((planes[0] >> bit) & 1u) |
                        (((planes[1] >> bit) & 1u) << 1u));
                    packed_high |= static_cast<std::uint8_t>(
                        high << (index * 2));
                }
                destination[16 + group] = packed_high;
            }
            scales[linear] =
                reinterpret_cast<const std::uint16_t *>(unit + 1536u)[local_column];
            secondary_scales[linear] =
                reinterpret_cast<const std::uint16_t *>(unit + 1664u)[local_column];
        }

        /** Validate one complete ordered chunk without touching the runtime. */
        bool validChunk(
            const ExpertTierWeightDeviceLayout &layout,
            std::uint32_t first_unit,
            std::uint32_t unit_count,
            std::size_t cpu_chunk_capacity,
            const void *stream) noexcept
        {
            return layout.valid() && stream != nullptr && unit_count != 0 &&
                   first_unit < layout.unit_count &&
                   unit_count <= layout.unit_count - first_unit &&
                   unit_count <= layout.maximum_units_per_chunk &&
                   cpu_chunk_capacity >= layout.chunkBytes(unit_count);
        }
    } // namespace

    bool launchGpuToCpuExpertTierChunkCUDA(
        const ExpertTierGpuConstProjectionView &source,
        const ExpertTierWeightDeviceLayout &layout,
        std::uint32_t first_unit,
        std::uint32_t unit_count,
        std::uint8_t *device_cpu_chunk,
        std::size_t device_cpu_chunk_capacity,
        void *stream) noexcept
    {
        if (layout.direction !=
                ExpertTierWeightConversionDirection::GpuToCpu ||
            device_cpu_chunk == nullptr || !source.validFor(layout) ||
            !validChunk(
                layout, first_unit, unit_count,
                device_cpu_chunk_capacity, stream) ||
            cudaPeekAtLastError() != cudaSuccess)
        {
            return false;
        }

        const dim3 grid(unit_count);
        const dim3 block(kColumnsPerUnit);
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        if (layout.cpu_encoding ==
            cpu::native_vnni::CPUNativeVNNIEncoding::NibbleLUT)
        {
            if (layout.gpu_codebook_id == 0)
            {
                gpuToCpuNibbleUnitsCUDA<0, false><<<grid, block, 0, cuda_stream>>>(
                    source.payload, source.scales, source.mins, layout.N,
                    layout.blocks_per_row, first_unit, layout.cpu_block_stride,
                    device_cpu_chunk);
            }
            else if (layout.gpu_codebook_id == 4)
            {
                gpuToCpuNibbleUnitsCUDA<4, false><<<grid, block, 0, cuda_stream>>>(
                    source.payload, source.scales, source.mins, layout.N,
                    layout.blocks_per_row, first_unit, layout.cpu_block_stride,
                    device_cpu_chunk);
            }
            else
            {
                gpuToCpuNibbleUnitsCUDA<5, true><<<grid, block, 0, cuda_stream>>>(
                    source.payload, source.scales, source.mins, layout.N,
                    layout.blocks_per_row, first_unit, layout.cpu_block_stride,
                    device_cpu_chunk);
            }
        }
        else if (layout.cpu_encoding ==
                 cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8)
        {
#define LLAMINAR_LAUNCH_DECODED_SOURCE(CODEBOOK)                         \
    gpuToCpuDecodedUnitsCUDA<CODEBOOK><<<grid, block, 0, cuda_stream>>>( \
        source.payload, source.scales, source.mins, source.emins,         \
        layout.N, layout.blocks_per_row, first_unit,                     \
        layout.cpu_block_stride, device_cpu_chunk)
            switch (layout.gpu_codebook_id)
            {
            case 6:
                LLAMINAR_LAUNCH_DECODED_SOURCE(6);
                break;
            case 7:
                LLAMINAR_LAUNCH_DECODED_SOURCE(7);
                break;
            case 9:
                LLAMINAR_LAUNCH_DECODED_SOURCE(9);
                break;
            case 10:
                LLAMINAR_LAUNCH_DECODED_SOURCE(10);
                break;
            case 11:
                LLAMINAR_LAUNCH_DECODED_SOURCE(11);
                break;
            case 12:
                LLAMINAR_LAUNCH_DECODED_SOURCE(12);
                break;
            case 13:
                LLAMINAR_LAUNCH_DECODED_SOURCE(13);
                break;
            case 14:
                LLAMINAR_LAUNCH_DECODED_SOURCE(14);
                break;
            case 15:
                LLAMINAR_LAUNCH_DECODED_SOURCE(15);
                break;
            case 16:
                LLAMINAR_LAUNCH_DECODED_SOURCE(16);
                break;
            case 17:
                LLAMINAR_LAUNCH_DECODED_SOURCE(17);
                break;
            case 19:
                LLAMINAR_LAUNCH_DECODED_SOURCE(19);
                break;
            case kNativeVnniExpandedInt8MinCodebook:
                LLAMINAR_LAUNCH_DECODED_SOURCE(
                    kNativeVnniExpandedInt8MinCodebook);
                break;
            default:
                return false;
            }
#undef LLAMINAR_LAUNCH_DECODED_SOURCE
        }
        else
        {
            gpuToCpuQ6UnitsCUDA<<<grid, block, 0, cuda_stream>>>(
                source.payload, source.scales, source.mins, layout.N,
                layout.blocks_per_row, first_unit, layout.cpu_block_stride,
                device_cpu_chunk);
        }
        return cudaPeekAtLastError() == cudaSuccess;
    }

    bool launchCpuToGpuExpertTierChunkCUDA(
        const std::uint8_t *device_cpu_chunk,
        std::size_t device_cpu_chunk_bytes,
        const ExpertTierWeightDeviceLayout &layout,
        std::uint32_t first_unit,
        std::uint32_t unit_count,
        const ExpertTierGpuMutableProjectionView &destination,
        void *stream) noexcept
    {
        if (layout.direction !=
                ExpertTierWeightConversionDirection::CpuToGpu ||
            device_cpu_chunk == nullptr || !destination.validFor(layout) ||
            !validChunk(
                layout, first_unit, unit_count,
                device_cpu_chunk_bytes, stream) ||
            cudaPeekAtLastError() != cudaSuccess)
        {
            return false;
        }

        const dim3 grid(unit_count);
        const dim3 block(kColumnsPerUnit);
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        if (layout.cpu_encoding ==
            cpu::native_vnni::CPUNativeVNNIEncoding::NibbleLUT)
        {
            if (layout.cpu_is_asymmetric != 0)
            {
                cpuToGpuNibbleUnitsCUDA<true><<<grid, block, 0, cuda_stream>>>(
                    device_cpu_chunk, layout.N, layout.blocks_per_row,
                    first_unit, layout.cpu_block_stride, destination.payload,
                    destination.scales, destination.mins);
            }
            else
            {
                cpuToGpuNibbleUnitsCUDA<false><<<grid, block, 0, cuda_stream>>>(
                    device_cpu_chunk, layout.N, layout.blocks_per_row,
                    first_unit, layout.cpu_block_stride, destination.payload,
                    destination.scales, destination.mins);
            }
        }
        else if (layout.cpu_encoding ==
                 cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8)
        {
            if (layout.cpu_is_asymmetric != 0)
            {
                cpuToGpuExpandedUnitsCUDA<true><<<grid, block, 0, cuda_stream>>>(
                    device_cpu_chunk, layout.N, layout.blocks_per_row,
                    first_unit, layout.cpu_block_stride, destination.payload,
                    destination.scales, destination.mins);
            }
            else
            {
                cpuToGpuExpandedUnitsCUDA<false><<<grid, block, 0, cuda_stream>>>(
                    device_cpu_chunk, layout.N, layout.blocks_per_row,
                    first_unit, layout.cpu_block_stride, destination.payload,
                    destination.scales, destination.mins);
            }
        }
        else
        {
            cpuToGpuQ6UnitsCUDA<<<grid, block, 0, cuda_stream>>>(
                device_cpu_chunk, layout.N, layout.blocks_per_row,
                first_unit, layout.cpu_block_stride, destination.payload,
                destination.scales, destination.mins);
        }
        return cudaPeekAtLastError() == cudaSuccess;
    }
} // namespace llaminar2
