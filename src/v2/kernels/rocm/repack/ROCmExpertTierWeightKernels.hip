/**
 * @file ROCmExpertTierWeightKernels.hip
 * @brief HIP kernels for direct GPU/CPU ExpertOverlay weight streaming.
 *
 * CUDA and ROCm consume the same separated GPU representation. This backend
 * therefore implements the identical one-block-per-CPU-unit arithmetic and
 * differs only in runtime launch/error handling. Complete unit ownership keeps
 * writes deterministic on both 32- and 64-lane hardware execution widths.
 */

#include "kernels/rocm/repack/ROCmExpertTierWeightKernels.h"
#include "backends/rocm/HipDeviceGuard.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace llaminar2
{
    namespace
    {
        constexpr int kColumnsPerUnit = 64;

        __device__ __constant__ std::int8_t kTierIQ4ValuesROCm[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10,
            1, 13, 25, 38, 53, 69, 89, 113};

        // HIP device constants are translation-unit local without GPU RDC.
        // These copies belong to tier conversion and are published once by the
        // central ROCm IQ-table initialization authority before graph capture.
        __device__ __constant__ std::uint32_t kTierIQ3SGridROCm[512];
        __device__ __constant__ std::uint32_t kTierIQ3XXSGridROCm[256];
        __device__ __constant__ std::uint64_t kTierIQ2SGridROCm[1024];
        __device__ __constant__ std::uint64_t kTierIQ2XSGridROCm[512];
        __device__ __constant__ std::uint64_t kTierIQ2XXSGridROCm[256];
        __device__ __constant__ std::uint64_t kTierIQ1SGridROCm[2048];

        __device__ __forceinline__ void initializeCpuUnitROCm(std::uint8_t *, std::uint32_t);

        /**
         * @brief Lossless native-payload transpose for all compact multi-scale formats.
         * @tparam ToCpu True for demotion, false for promotion.
         * @param source Immutable separated GPU source (demotion only).
         * @param destination Mutable separated GPU destination (promotion only).
         * @param cpu_source Complete received CPU units (promotion only).
         * @param cpu_destination Complete outgoing CPU units (demotion only).
         * @param layout Validated immutable format and stride contract.
         * @param first_unit Absolute unit origin of this bounded transfer chunk.
         *
         * One CTA owns a unit; each lane owns one output column. Padding and
         * unused metadata are zeroed before valid lanes publish. No floating
         * arithmetic, lookup, allocation or synchronization with inference is
         * required: scale/minimum bits and IQ1_M delta signs move unchanged.
         */
        template <bool ToCpu>
        __global__ void compactMultiScaleUnitsROCm(
            ExpertTierGpuConstProjectionView source,
            ExpertTierGpuMutableProjectionView destination,
            const std::uint8_t *cpu_source,
            std::uint8_t *cpu_destination,
            ExpertTierWeightDeviceLayout layout,
            std::uint32_t first_unit)
        {
            const unsigned unit_index = first_unit + blockIdx.x;
            const int column = static_cast<int>(threadIdx.x);
            const int n = static_cast<int>(unit_index / layout.blocks_per_row) * 64 + column;
            const int kb = static_cast<int>(unit_index % layout.blocks_per_row);
            const size_t offset = static_cast<size_t>(blockIdx.x) * layout.cpu_block_stride;
            if constexpr (ToCpu)
            {
                initializeCpuUnitROCm(cpu_destination + offset, layout.cpu_block_stride);
                __syncthreads();
            }
            if (n >= layout.N)
                return;
            const size_t linear = static_cast<size_t>(kb) * layout.N + n;
            const int payload_bytes = layout.gpu_payload_bytes_per_block;
            for (int byte = 0; byte < payload_bytes; ++byte)
            {
                const size_t cpu_offset = offset + (byte / 4) * 256 + column * 4 + byte % 4;
                if constexpr (ToCpu)
                    cpu_destination[cpu_offset] = source.payload[linear * payload_bytes + byte];
                else
                    destination.payload[linear * payload_bytes + byte] = cpu_source[cpu_offset];
            }
            const size_t metadata = offset + layout.cpu_data_stride;
            if constexpr (ToCpu)
            {
                reinterpret_cast<uint16_t *>(cpu_destination + metadata)[column] = source.scales[linear];
                reinterpret_cast<uint16_t *>(cpu_destination + metadata + 128)[column] = source.mins[linear];
                if (layout.gpu_has_emins)
                    reinterpret_cast<uint32_t *>(cpu_destination + metadata + 256)[column] = source.emins[linear];
            }
            else
            {
                destination.scales[linear] = reinterpret_cast<const uint16_t *>(cpu_source + metadata)[column];
                destination.mins[linear] = reinterpret_cast<const uint16_t *>(cpu_source + metadata + 128)[column];
                if (layout.gpu_has_emins)
                    destination.emins[linear] = reinterpret_cast<const uint32_t *>(cpu_source + metadata + 256)[column];
            }
        }

        /**
         * @brief Compile-time properties of one accelerator source codebook.
         * @tparam Codebook NativeVNNI codebook identifier stored in the stream
         *         manifest.
         *
         * The stream kernel reads actual accelerator source blocks, not a
         * normalized CPU projection. Keeping payload geometry here beside the
         * decoder makes a new codebook fail compilation until its byte contract
         * is explicit.
         */
        template <std::uint8_t Codebook>
        struct TierSourceTraitsROCm
        {
            static constexpr bool is_5bit = Codebook == 6 || Codebook == 7;
            static constexpr bool is_3bit = Codebook == 9;
            static constexpr bool is_2bit = Codebook == 10;
            static constexpr bool is_iq3 = Codebook == 11 || Codebook == 12;
            static constexpr bool is_iq2 =
                Codebook == 13 || Codebook == 14 || Codebook == 15;
            static constexpr bool is_iq1 = Codebook == 16 || Codebook == 17;
            static constexpr bool is_int8 =
                Codebook == 19 ||
                Codebook == kNativeVnniExpandedInt8MinCodebook;
            static constexpr int payload_bytes =
                is_int8                           ? 32
                : is_5bit                        ? 20
                : is_3bit                        ? 12
                : is_2bit                        ? 8
                : Codebook == 11                 ? 13
                : Codebook == 12                 ? 12
                : Codebook == 13 || Codebook == 14 ? 9
                : Codebook == 15                 ? 8
                : is_iq1                         ? 6
                                                   : 0;
            static constexpr bool supported =
                is_5bit || is_3bit || is_2bit || is_iq3 || is_iq2 ||
                is_iq1 || is_int8;
        };

        /**
         * @brief Apply four IQ sign bits to four packed positive magnitudes.
         * @param grid4 Four signed-byte magnitudes packed in a 32-bit word.
         * @param sign_low_nibble One conditional-negation bit per byte.
         * @return Four independently signed bytes in their original positions.
         *
         * Multiplication scatters the four selector bits into byte-low bits.
         * Shift/subtract expands each selector to 0x00 or 0xff, after which
         * XOR-plus-carry is an independent two's-complement negate per byte.
         * This is the same SWAR identity used by ROCm NativeVNNI inference.
         */
        __device__ __forceinline__ std::uint32_t applyTierIQSignsROCm(
            std::uint32_t grid4,
            std::uint8_t sign_low_nibble)
        {
            const std::uint32_t spread =
                (static_cast<std::uint32_t>(sign_low_nibble & 0x0fu) *
                 0x00204081u) &
                0x01010101u;
            std::uint32_t mask = 0;
            asm("v_lshlrev_b32 %0, 8, %1\n\t"
                "v_sub_u32 %0, %0, %1"
                : "=v"(mask)
                : "v"(spread));
            return (grid4 ^ mask) + spread;
        }

        /**
         * @brief Read one IQ3 entry from this translation unit's immutable LUT.
         * @tparam Codebook IQ3_S (11) or IQ3_XXS (12).
         * @param index Source payload's resolved grid index.
         * @return Four positive magnitudes packed as bytes.
         */
        template <std::uint8_t Codebook>
        __device__ __forceinline__ std::uint32_t tierIQ3LookupROCm(int index)
        {
            static_assert(Codebook == 11 || Codebook == 12);
            if constexpr (Codebook == 11)
                return kTierIQ3SGridROCm[index];
            return kTierIQ3XXSGridROCm[index];
        }

        /**
         * @brief Read one IQ2 entry from this translation unit's immutable LUT.
         * @tparam Codebook IQ2_S (13), IQ2_XS (14), or IQ2_XXS (15).
         * @param index Source payload's resolved grid index.
         * @return Eight positive magnitudes packed as bytes.
         */
        template <std::uint8_t Codebook>
        __device__ __forceinline__ std::uint64_t tierIQ2LookupROCm(int index)
        {
            static_assert(Codebook >= 13 && Codebook <= 15);
            if constexpr (Codebook == 13)
                return kTierIQ2SGridROCm[index];
            if constexpr (Codebook == 14)
                return kTierIQ2XSGridROCm[index];
            return kTierIQ2XXSGridROCm[index];
        }

        /**
         * @brief Read one shared IQ1_S/IQ1_M grid entry.
         * @param index Eleven-bit source payload grid index.
         * @return Eight signed {-1,0,+1} values packed as bytes.
         */
        __device__ __forceinline__ std::uint64_t tierIQ1LookupROCm(int index)
        {
            return kTierIQ1SGridROCm[index];
        }

        /**
         * @brief Decode one compact accelerator block into eight INT8x4 groups.
         * @tparam Codebook One of 6, 7, 9 through 17, raw-INT8 19, or
         *         normalized asymmetric INT8+minimum 23.
         * @param payload Start of one production-packed NativeVNNI block.
         * @param groups Receives 32 decoded integers in logical K order.
         *
         * This is the integer decode used by the production ROCm grouped MoE
         * path, specialized here against tier-local IQ tables. It intentionally
         * stops before applying scales: later preparation chooses either the
         * CPU affine representation or normalized Q8 representation.
         */
        template <std::uint8_t Codebook>
        __device__ __forceinline__ void decodeTierSourceGroupsROCm(
            const std::uint8_t *payload,
            std::int32_t (&groups)[8])
        {
            using Traits = TierSourceTraitsROCm<Codebook>;
            static_assert(Traits::supported);

            if constexpr (Traits::is_5bit)
            {
                const std::uint32_t high_bits =
                    *reinterpret_cast<const std::uint32_t *>(payload + 16);
#pragma unroll
                for (int group = 0; group < 4; ++group)
                {
                    const std::uint32_t raw =
                        *reinterpret_cast<const std::uint32_t *>(
                            payload + group * 4);
                    const std::uint32_t low_high_nibble =
                        (high_bits >> (group * 4)) & 0x0fu;
                    const std::uint32_t high_high_nibble =
                        (high_bits >> (group * 4 + 16)) & 0x0fu;
                    const std::uint32_t low_deposit =
                        ((low_high_nibble & 1u) << 4) |
                        ((low_high_nibble & 2u) << 11) |
                        ((low_high_nibble & 4u) << 18) |
                        ((low_high_nibble & 8u) << 25);
                    const std::uint32_t high_deposit =
                        ((high_high_nibble & 1u) << 4) |
                        ((high_high_nibble & 2u) << 11) |
                        ((high_high_nibble & 4u) << 18) |
                        ((high_high_nibble & 8u) << 25);
                    std::uint32_t low =
                        (raw & 0x0f0f0f0fu) | low_deposit;
                    std::uint32_t high =
                        ((raw >> 4) & 0x0f0f0f0fu) | high_deposit;
                    if constexpr (Codebook == 6)
                    {
                        // XOR guards byte lanes from cross-byte borrow while
                        // subtracting the symmetric Q5 zero point of sixteen.
                        low ^= 0x80808080u;
                        low -= 0x10101010u;
                        low ^= 0x80808080u;
                        high ^= 0x80808080u;
                        high -= 0x10101010u;
                        high ^= 0x80808080u;
                    }
                    groups[group] = static_cast<std::int32_t>(low);
                    groups[group + 4] = static_cast<std::int32_t>(high);
                }
            }
            else if constexpr (Traits::is_3bit)
            {
                constexpr std::uint32_t kBitSpread = 0x00204081u;
                const std::uint32_t high_bits =
                    *reinterpret_cast<const std::uint32_t *>(payload + 8);
#pragma unroll
                for (int half = 0; half < 2; ++half)
                {
                    const std::uint32_t raw =
                        *reinterpret_cast<const std::uint32_t *>(
                            payload + half * 4);
#pragma unroll
                    for (int group = 0; group < 4; ++group)
                    {
                        const std::uint32_t low_two =
                            (raw >> (group * 2)) & 0x03030303u;
                        const std::uint32_t four_high_bits =
                            (high_bits >> ((half * 4 + group) * 4)) & 0x0fu;
                        const std::uint32_t expanded_high =
                            ((four_high_bits * kBitSpread) & 0x01010101u) << 2;
                        std::uint32_t decoded =
                            (low_two | expanded_high) ^ 0x80808080u;
                        decoded -= 0x04040404u;
                        decoded ^= 0x80808080u;
                        groups[half * 4 + group] =
                            static_cast<std::int32_t>(decoded);
                    }
                }
            }
            else if constexpr (Traits::is_2bit)
            {
#pragma unroll
                for (int half = 0; half < 2; ++half)
                {
                    const std::uint32_t raw =
                        *reinterpret_cast<const std::uint32_t *>(
                            payload + half * 4);
                    const int base = half * 4;
                    groups[base] = static_cast<std::int32_t>(
                        raw & 0x03030303u);
                    groups[base + 1] = static_cast<std::int32_t>(
                        (raw >> 2) & 0x03030303u);
                    groups[base + 2] = static_cast<std::int32_t>(
                        (raw >> 4) & 0x03030303u);
                    groups[base + 3] = static_cast<std::int32_t>(
                        (raw >> 6) & 0x03030303u);
                }
            }
            else if constexpr (Traits::is_iq3)
            {
                const std::uint8_t high_index_bits =
                    Codebook == 11 ? payload[8] : 0;
                constexpr int kSignBase = Codebook == 11 ? 9 : 8;
#pragma unroll
                for (int group = 0; group < 8; ++group)
                {
                    int index = payload[group];
                    if constexpr (Codebook == 11)
                    {
                        index |= static_cast<int>(
                                     (high_index_bits >> group) & 1u)
                                 << 8;
                    }
                    const std::uint32_t grid =
                        tierIQ3LookupROCm<Codebook>(index);
                    const std::uint8_t signs =
                        payload[kSignBase + group / 2];
                    const std::uint8_t sign_nibble =
                        (group & 1) != 0
                            ? static_cast<std::uint8_t>(signs >> 4)
                            : static_cast<std::uint8_t>(signs & 0x0fu);
                    groups[group] = static_cast<std::int32_t>(
                        applyTierIQSignsROCm(grid, sign_nibble));
                }
            }
            else if constexpr (Traits::is_iq2)
            {
                const std::uint8_t high_index_bits =
                    Codebook == 15 ? 0 : payload[4];
                constexpr int kSignBase = Codebook == 15 ? 4 : 5;
#pragma unroll
                for (int lookup = 0; lookup < 4; ++lookup)
                {
                    int index = payload[lookup];
                    if constexpr (Codebook == 13)
                    {
                        index |= static_cast<int>(
                                     (high_index_bits >> (2 * lookup)) & 3u)
                                 << 8;
                    }
                    else if constexpr (Codebook == 14)
                    {
                        index |= static_cast<int>(
                                     (high_index_bits >> lookup) & 1u)
                                 << 8;
                    }
                    const std::uint64_t grid =
                        tierIQ2LookupROCm<Codebook>(index);
                    const std::uint8_t signs = payload[kSignBase + lookup];
                    groups[lookup * 2] = static_cast<std::int32_t>(
                        applyTierIQSignsROCm(
                            static_cast<std::uint32_t>(grid),
                            signs & 0x0fu));
                    groups[lookup * 2 + 1] = static_cast<std::int32_t>(
                        applyTierIQSignsROCm(
                            static_cast<std::uint32_t>(grid >> 32),
                            signs >> 4));
                }
            }
            else if constexpr (Codebook == 16)
            {
                const std::uint16_t high_index_bits =
                    static_cast<std::uint16_t>(payload[4]) |
                    (static_cast<std::uint16_t>(payload[5]) << 8);
#pragma unroll
                for (int lookup = 0; lookup < 4; ++lookup)
                {
                    const int index =
                        payload[lookup] |
                        (static_cast<int>(
                             (high_index_bits >> (3 * lookup)) & 7u)
                         << 8);
                    const std::uint64_t grid = tierIQ1LookupROCm(index);
                    groups[lookup * 2] = static_cast<std::int32_t>(
                        static_cast<std::uint32_t>(grid));
                    groups[lookup * 2 + 1] = static_cast<std::int32_t>(
                        static_cast<std::uint32_t>(grid >> 32));
                }
            }
            else if constexpr (Codebook == 17)
            {
                const std::uint8_t high0 = payload[4];
                const std::uint8_t high1 = payload[5];
                const int indices[4] = {
                    payload[0] | (static_cast<int>(high0 & 0x07u) << 8),
                    payload[1] |
                        (static_cast<int>((high0 >> 4) & 0x07u) << 8),
                    payload[2] | (static_cast<int>(high1 & 0x07u) << 8),
                    payload[3] |
                        (static_cast<int>((high1 >> 4) & 0x07u) << 8)};
#pragma unroll
                for (int lookup = 0; lookup < 4; ++lookup)
                {
                    const std::uint64_t grid =
                        tierIQ1LookupROCm(indices[lookup]);
                    groups[lookup * 2] = static_cast<std::int32_t>(
                        static_cast<std::uint32_t>(grid));
                    groups[lookup * 2 + 1] = static_cast<std::int32_t>(
                        static_cast<std::uint32_t>(grid >> 32));
                }
            }
            else
            {
                static_assert(
                    Codebook == 19 ||
                    Codebook == kNativeVnniExpandedInt8MinCodebook);
                // Normalized symmetric and asymmetric INT8 share the same
                // direct 32-byte payload.  Codebook 23's affine minimum is
                // carried separately and is restored with metadata below.
#pragma unroll
                for (int group = 0; group < 8; ++group)
                {
                    groups[group] =
                        *reinterpret_cast<const std::int32_t *>(
                            payload + group * 4);
                }
            }
        }

        /**
         * @brief Locate one column/value byte within a CPU VNNI unit.
         * @param local_column Column index from zero through 63.
         * @param value_index Prepared payload byte index for that column.
         * @return Byte offset from the start of the unit.
         */
        __device__ __forceinline__ std::size_t interleavedValueOffsetROCm(
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

        /**
         * @brief Deterministically initialize a complete final CPU unit.
         * @param unit Start of this block's exclusively owned unit.
         * @param stride Total data and metadata bytes in the unit.
         */
        __device__ __forceinline__ void initializeCpuUnitROCm(
            std::uint8_t *unit,
            std::uint32_t stride)
        {
            for (std::uint32_t offset = threadIdx.x; offset < stride;
                 offset += blockDim.x)
            {
                unit[offset] = 0;
            }
        }

        /**
         * @brief Decode one compact value for the CPU compensation vector.
         * @tparam Codebook Nibble execution codebook 0, 4, or 5.
         * @param value Four-bit quantized value.
         * @return Signed/unsigned decoded integer consumed by CPU VNNI.
         */
        template <std::uint8_t Codebook>
        __device__ __forceinline__ int decodeNibbleForCompensationROCm(
            std::uint8_t value)
        {
            if constexpr (Codebook == 0)
                return static_cast<int>(value) - 8;
            if constexpr (Codebook == 4)
                return static_cast<int>(kTierIQ4ValuesROCm[value]);
            return static_cast<int>(value);
        }

        /**
         * @brief Transpose separated compact nibbles into final CPU units.
         * @tparam Codebook Nibble execution codebook.
         * @tparam Asymmetric Whether the minimum region is present.
         */
        template <std::uint8_t Codebook, bool Asymmetric>
        __global__ void gpuToCpuNibbleUnitsROCm(
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
            const int kb = static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x);
            std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(blockIdx.x) * cpu_stride;

            // Zero padding and metadata gaps before real columns publish bytes.
            initializeCpuUnitROCm(unit, cpu_stride);
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
                unit[interleavedValueOffsetROCm(local_column, value)] = packed;
                compensation += decodeNibbleForCompensationROCm<Codebook>(
                    static_cast<std::uint8_t>(packed & 0x0fu));
                compensation += decodeNibbleForCompensationROCm<Codebook>(
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
         * @brief Extract one sign-extended byte from decoded INT8x4 groups.
         * @param groups Eight packed groups returned by the source decoder.
         * @param value_index Logical value from zero through 31.
         * @return Sign-extended decoded integer.
         */
        __device__ __forceinline__ int decodedTierGroupValueROCm(
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
         * @brief Losslessly expand one single-scale block per logical row.
         * @tparam Codebook One of 6, 7, 11, 12, 15, 16, 19, or normalized 23.
         * @param payload Production-packed source payload array.
         * @param scales Source primary/low-half FP16 scale array.
         * @param mins Optional source secondary-scale/minimum array.
         * @param emins Optional packed Q2_K effective-minimum array.
         * @param N Logical output rows in the GPU projection.
         * @param blocks_per_row Number of 32-value K blocks per row.
         * @param first_unit First global 64-column CPU unit in this chunk.
         * @param cpu_stride Complete byte stride of one CPU unit.
         * @param cpu_chunk Preallocated device staging output.
         *
         * Every thread owns one logical column. Q5 and Q8 blocks transpose
         * their exact integer domain. Single-scale IQuant grids preserve their
         * original scale/minimum bits without floating-point conversion. All
         * metadata is published only after the final byte domain is known.
         */
        template <std::uint8_t Codebook>
        __global__ void gpuToCpuDecodedUnitsROCm(
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
            const int k_block =
                static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x);
            std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(blockIdx.x) * cpu_stride;

            // Whole-unit initialization makes padded N columns and metadata
            // gaps deterministic, including when a chunk starts mid-matrix.
            initializeCpuUnitROCm(unit, cpu_stride);
            __syncthreads();

            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (n >= N)
                return;
            const std::size_t linear =
                static_cast<std::size_t>(k_block) * N + n;
            constexpr int kPayloadBytes =
                TierSourceTraitsROCm<Codebook>::payload_bytes;
            const std::uint8_t *source =
                payload + linear * static_cast<std::size_t>(kPayloadBytes);
            std::int32_t decoded_groups[8]{};
            decodeTierSourceGroupsROCm<Codebook>(source, decoded_groups);

            static_assert(Codebook == 6 || Codebook == 7 || Codebook == 11 ||
                          Codebook == 12 || Codebook == 15 || Codebook == 16 ||
                          Codebook == 19 || Codebook == kNativeVnniExpandedInt8MinCodebook);
            // These single-scale formats retain their exact integer domain.
            // Multi-scale sources use the native transposition kernel instead.
            for (int value = 0; value < 32; ++value)
            {
                unit[interleavedValueOffsetROCm(local_column, value)] =
                    static_cast<std::uint8_t>(static_cast<std::int8_t>(
                        decodedTierGroupValueROCm(decoded_groups, value)));
            }

            // CPU kernels shift activations to unsigned before VNNI. Store the
            // exact signed weight sum that compensates that shift.
            int compensation = 0;
            for (int value = 0; value < 32; ++value)
            {
                compensation += static_cast<std::int8_t>(
                    unit[interleavedValueOffsetROCm(local_column, value)]);
            }
            reinterpret_cast<std::int16_t *>(unit + 2048u)[local_column] =
                static_cast<std::int16_t>(compensation);
            reinterpret_cast<std::uint16_t *>(unit + 2176u)[local_column] =
                scales[linear];
            if constexpr (
                Codebook == 7 || Codebook == 16 ||
                Codebook == kNativeVnniExpandedInt8MinCodebook)
            {
                reinterpret_cast<std::uint16_t *>(
                    unit + 2304u)[local_column] =
                    mins[linear];
            }
        }

        /** @brief Transpose separated native Q6_K into dual-scale CPU units. */
        __global__ void gpuToCpuQ6UnitsROCm(
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
            const int kb = static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x);
            std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(blockIdx.x) * cpu_stride;

            initializeCpuUnitROCm(unit, cpu_stride);
            __syncthreads();
            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (n < N)
            {
                const std::size_t linear =
                    static_cast<std::size_t>(kb) * N + n;
                const std::uint8_t *source = payload + linear * 24u;
                for (int value = 0; value < 16; ++value)
                {
                    unit[interleavedValueOffsetROCm(local_column, value)] =
                        source[value];
                }
                reinterpret_cast<std::uint16_t *>(unit + 1536u)[local_column] =
                    scales[linear];
                reinterpret_cast<std::uint16_t *>(unit + 1664u)[local_column] =
                    secondary_scales[linear];
            }

            // Thirty-two threads each own one (group, 16-column ZMM) plane pair.
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

        /**
         * @brief Deinterleave CPU compact-nibble units into separated GPU data.
         * @tparam Asymmetric Whether the minimum region is present.
         */
        template <bool Asymmetric>
        __global__ void cpuToGpuNibbleUnitsROCm(
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
            const int kb = static_cast<int>(global_unit % blocks_per_row);
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
                    unit[interleavedValueOffsetROCm(local_column, value)];
            }
            scales[linear] =
                reinterpret_cast<const std::uint16_t *>(unit + 1152u)[local_column];
            if constexpr (Asymmetric)
            {
                mins[linear] = reinterpret_cast<const std::uint16_t *>(
                    unit + 1280u)[local_column];
            }
        }

        /**
         * @brief Deinterleave CPU expanded-INT8 units into separated GPU data.
         * @tparam Asymmetric Whether the minimum region is present.
         */
        template <bool Asymmetric>
        __global__ void cpuToGpuExpandedUnitsROCm(
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
            const int kb = static_cast<int>(global_unit % blocks_per_row);
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
                    unit[interleavedValueOffsetROCm(local_column, value)];
            }
            scales[linear] =
                reinterpret_cast<const std::uint16_t *>(unit + 2176u)[local_column];
            if constexpr (Asymmetric)
            {
                mins[linear] = reinterpret_cast<const std::uint16_t *>(
                    unit + 2304u)[local_column];
            }
        }

        /** @brief Deinterleave CPU native-Q6 units into separated GPU data. */
        __global__ void cpuToGpuQ6UnitsROCm(
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
            const int kb = static_cast<int>(global_unit % blocks_per_row);
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
                    unit[interleavedValueOffsetROCm(local_column, value)];
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

        /**
         * @brief Validate a complete chunk before any HIP runtime interaction.
         * @param layout Immutable device layout.
         * @param first_unit First global unit in the chunk.
         * @param unit_count Number of complete units.
         * @param cpu_chunk_capacity Available staging bytes.
         * @param stream Exact stream, which must be non-null.
         * @return Whether range and capacity satisfy the capture identity.
         */
        bool validChunkROCm(
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

    bool initializeExpertTierIQGridTablesROCm(
        int device_id,
        const void *iq3s_grid,
        const void *iq3xxs_grid,
        const void *iq2s_grid,
        const void *iq2xs_grid,
        const void *iq2xxs_grid,
        const void *iq1s_grid) noexcept
    {
        if (iq3s_grid == nullptr || iq3xxs_grid == nullptr ||
            iq2s_grid == nullptr || iq2xs_grid == nullptr ||
            iq2xxs_grid == nullptr || iq1s_grid == nullptr ||
            HipDeviceGuard::setDevice(device_id) != hipSuccess)
        {
            return false;
        }

        // Each copy targets this translation unit's private constant symbol.
        // Publication is synchronous and occurs during weight preparation, so
        // captured tier-stream launches only read immutable device state.
        if (hipMemcpyToSymbol(
                kTierIQ3SGridROCm,
                iq3s_grid,
                sizeof(kTierIQ3SGridROCm)) != hipSuccess)
            return false;
        if (hipMemcpyToSymbol(
                kTierIQ3XXSGridROCm,
                iq3xxs_grid,
                sizeof(kTierIQ3XXSGridROCm)) != hipSuccess)
            return false;
        if (hipMemcpyToSymbol(
                kTierIQ2SGridROCm,
                iq2s_grid,
                sizeof(kTierIQ2SGridROCm)) != hipSuccess)
            return false;
        if (hipMemcpyToSymbol(
                kTierIQ2XSGridROCm,
                iq2xs_grid,
                sizeof(kTierIQ2XSGridROCm)) != hipSuccess)
            return false;
        if (hipMemcpyToSymbol(
                kTierIQ2XXSGridROCm,
                iq2xxs_grid,
                sizeof(kTierIQ2XXSGridROCm)) != hipSuccess)
            return false;
        return hipMemcpyToSymbol(
                   kTierIQ1SGridROCm,
                   iq1s_grid,
                   sizeof(kTierIQ1SGridROCm)) == hipSuccess;
    }

    /**
     * @brief Validate and launch one asynchronous GPU-to-CPU-format HIP chunk.
     * @return `true` when the exact specialized kernel was enqueued.
     */
    bool launchGpuToCpuExpertTierChunkROCm(
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
            !validChunkROCm(
                layout, first_unit, unit_count,
                device_cpu_chunk_capacity, stream) ||
            hipPeekAtLastError() != hipSuccess)
        {
            return false;
        }

        const dim3 grid(unit_count);
        const dim3 block(kColumnsPerUnit);
        auto hip_stream = reinterpret_cast<hipStream_t>(stream);
        if (layout.cpu_encoding == cpu::native_vnni::CPUNativeVNNIEncoding::CompactMultiScale)
        {
            compactMultiScaleUnitsROCm<true><<<grid, block, 0, hip_stream>>>(
                source, {}, nullptr, device_cpu_chunk, layout, first_unit);
            return hipPeekAtLastError() == hipSuccess;
        }
        if (layout.cpu_encoding ==
            cpu::native_vnni::CPUNativeVNNIEncoding::NibbleLUT)
        {
            if (layout.gpu_codebook_id == 0)
            {
                hipLaunchKernelGGL(
                    (gpuToCpuNibbleUnitsROCm<0, false>), grid, block, 0,
                    hip_stream, source.payload, source.scales, source.mins,
                    layout.N, layout.blocks_per_row, first_unit,
                    layout.cpu_block_stride, device_cpu_chunk);
            }
            else if (layout.gpu_codebook_id == 4)
            {
                hipLaunchKernelGGL(
                    (gpuToCpuNibbleUnitsROCm<4, false>), grid, block, 0,
                    hip_stream, source.payload, source.scales, source.mins,
                    layout.N, layout.blocks_per_row, first_unit,
                    layout.cpu_block_stride, device_cpu_chunk);
            }
            else
            {
                hipLaunchKernelGGL(
                    (gpuToCpuNibbleUnitsROCm<5, true>), grid, block, 0,
                    hip_stream, source.payload, source.scales, source.mins,
                    layout.N, layout.blocks_per_row, first_unit,
                    layout.cpu_block_stride, device_cpu_chunk);
            }
        }
        else if (layout.cpu_encoding ==
                 cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8)
        {
#define LLAMINAR_LAUNCH_ROCM_TIER_SOURCE(CODEBOOK)                    \
    hipLaunchKernelGGL(                                               \
        (gpuToCpuDecodedUnitsROCm<CODEBOOK>), grid, block, 0,         \
        hip_stream, source.payload, source.scales, source.mins,       \
        source.emins, layout.N, layout.blocks_per_row, first_unit,    \
        layout.cpu_block_stride, device_cpu_chunk)
            switch (layout.gpu_codebook_id)
            {
            case 6:
                LLAMINAR_LAUNCH_ROCM_TIER_SOURCE(6);
                break;
            case 7:
                LLAMINAR_LAUNCH_ROCM_TIER_SOURCE(7);
                break;
            case 11:
                LLAMINAR_LAUNCH_ROCM_TIER_SOURCE(11);
                break;
            case 12:
                LLAMINAR_LAUNCH_ROCM_TIER_SOURCE(12);
                break;
            case 15:
                LLAMINAR_LAUNCH_ROCM_TIER_SOURCE(15);
                break;
            case 16:
                LLAMINAR_LAUNCH_ROCM_TIER_SOURCE(16);
                break;
            case 19:
                LLAMINAR_LAUNCH_ROCM_TIER_SOURCE(19);
                break;
            case kNativeVnniExpandedInt8MinCodebook:
                LLAMINAR_LAUNCH_ROCM_TIER_SOURCE(
                    kNativeVnniExpandedInt8MinCodebook);
                break;
            default:
                return false;
            }
#undef LLAMINAR_LAUNCH_ROCM_TIER_SOURCE
        }
        else
        {
            hipLaunchKernelGGL(
                gpuToCpuQ6UnitsROCm, grid, block, 0, hip_stream,
                source.payload, source.scales, source.mins, layout.N,
                layout.blocks_per_row, first_unit, layout.cpu_block_stride,
                device_cpu_chunk);
        }
        return hipPeekAtLastError() == hipSuccess;
    }

    /**
     * @brief Validate and launch one asynchronous CPU-format-to-GPU HIP chunk.
     * @return `true` when the exact specialized kernel was enqueued.
     */
    bool launchCpuToGpuExpertTierChunkROCm(
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
            !validChunkROCm(
                layout, first_unit, unit_count,
                device_cpu_chunk_bytes, stream) ||
            hipPeekAtLastError() != hipSuccess)
        {
            return false;
        }

        const dim3 grid(unit_count);
        const dim3 block(kColumnsPerUnit);
        auto hip_stream = reinterpret_cast<hipStream_t>(stream);
        if (layout.cpu_encoding ==
            cpu::native_vnni::CPUNativeVNNIEncoding::CompactMultiScale)
        {
            compactMultiScaleUnitsROCm<false><<<grid, block, 0, hip_stream>>>(
                {}, destination, device_cpu_chunk, nullptr, layout, first_unit);
            return hipPeekAtLastError() == hipSuccess;
        }
        if (layout.cpu_encoding ==
            cpu::native_vnni::CPUNativeVNNIEncoding::NibbleLUT)
        {
            if (layout.cpu_is_asymmetric != 0)
            {
                hipLaunchKernelGGL(
                    (cpuToGpuNibbleUnitsROCm<true>), grid, block, 0,
                    hip_stream, device_cpu_chunk, layout.N,
                    layout.blocks_per_row, first_unit,
                    layout.cpu_block_stride, destination.payload,
                    destination.scales, destination.mins);
            }
            else
            {
                hipLaunchKernelGGL(
                    (cpuToGpuNibbleUnitsROCm<false>), grid, block, 0,
                    hip_stream, device_cpu_chunk, layout.N,
                    layout.blocks_per_row, first_unit,
                    layout.cpu_block_stride, destination.payload,
                    destination.scales, destination.mins);
            }
        }
        else if (layout.cpu_encoding ==
                 cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8)
        {
            if (layout.cpu_is_asymmetric != 0)
            {
                hipLaunchKernelGGL(
                    (cpuToGpuExpandedUnitsROCm<true>), grid, block, 0,
                    hip_stream, device_cpu_chunk, layout.N,
                    layout.blocks_per_row, first_unit,
                    layout.cpu_block_stride, destination.payload,
                    destination.scales, destination.mins);
            }
            else
            {
                hipLaunchKernelGGL(
                    (cpuToGpuExpandedUnitsROCm<false>), grid, block, 0,
                    hip_stream, device_cpu_chunk, layout.N,
                    layout.blocks_per_row, first_unit,
                    layout.cpu_block_stride, destination.payload,
                    destination.scales, destination.mins);
            }
        }
        else
        {
            hipLaunchKernelGGL(
                cpuToGpuQ6UnitsROCm, grid, block, 0, hip_stream,
                device_cpu_chunk, layout.N, layout.blocks_per_row, first_unit,
                layout.cpu_block_stride, destination.payload,
                destination.scales, destination.mins);
        }
        return hipPeekAtLastError() == hipSuccess;
    }
} // namespace llaminar2
