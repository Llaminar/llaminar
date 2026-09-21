/**
 * @file CUDAExpertTierWeightKernels.cu
 * @brief CUDA kernels for direct GPU/CPU ExpertOverlay weight streaming.
 *
 * One thread block owns one complete 64-column by 32-K CPU unit. This gives
 * every output byte exactly one writer, including Q6_K high-bit planes, and
 * makes arbitrary ordered chunks independent. Padding bytes are initialized
 * deterministically so the transmitted CPU representation is byte-exact.
 * Native launches and the graph-resident transfer worker call the same unit
 * functions. Conversion is therefore not a separately queued prerequisite of
 * background copying and cannot be stranded behind a future inference reader.
 */

#include "kernels/cuda/repack/CUDAExpertTierWeightKernels.h"
#include "kernels/cuda/gemm/CUDANativeVNNIDecodeCommon.cuh"
#include "backends/cuda/MappedTransferServiceCUDA.h"
#include "backends/cuda/MappedTransferServiceDevice.cuh"
#include "transfer/MappedTransferPackedWork.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace llaminar2
{
    namespace
    {
        constexpr int kColumnsPerUnit = 64;

        __device__ __constant__ __align__(16) std::int8_t kTierIQ4ValuesCUDA[16] = {
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

        /**
         * @brief Gather CPU-interleaved words into aligned GPU payload vectors.
         * @tparam Bytes Complete physical payload width, a multiple of 16 bytes.
         * @param unit One CPU-format unit in device-local staging.
         * @param column This group's unique logical column, from zero through 63.
         * @tparam Alignment Guaranteed physical block alignment (Q6_K uses eight).
         * @param destination Packed GPU block with the declared aligned base/stride.
         *
         * Word loads are coalesced across columns. Vector stores avoid issuing
         * 16/32 separate byte stores per column while retaining every payload bit.
         * The same helper serves native repacking and the background worker.
         */
        template <unsigned Bytes, unsigned Alignment = 16u>
        __device__ __forceinline__ void gatherCpuPayloadVectors(
            const std::uint8_t *unit, int column, std::uint8_t *destination)
        {
            static_assert(Bytes % sizeof(uint4) == 0);
#pragma unroll
            for (unsigned offset = 0u; offset < Bytes; offset += sizeof(uint4))
            {
                const uint4 payload{
                    *reinterpret_cast<const unsigned *>(unit + interleavedValueOffset(column, offset)),
                    *reinterpret_cast<const unsigned *>(unit + interleavedValueOffset(column, offset + 4u)),
                    *reinterpret_cast<const unsigned *>(unit + interleavedValueOffset(column, offset + 8u)),
                    *reinterpret_cast<const unsigned *>(unit + interleavedValueOffset(column, offset + 12u))};
                if constexpr (Alignment >= alignof(uint4))
                    *reinterpret_cast<uint4 *>(destination + offset) = payload;
                else
                {
                    static_assert(Alignment >= alignof(uint2));
                    reinterpret_cast<uint2 *>(destination + offset)[0] = uint2{payload.x, payload.y};
                    reinterpret_cast<uint2 *>(destination + offset)[1] = uint2{payload.z, payload.w};
                }
            }
        }

        /** One 64-lane unit group initializes its bytes and metadata gaps. */
        __device__ __forceinline__ void initializeCpuUnit(
            std::uint8_t *unit,
            std::uint32_t stride)
        {
            for (std::uint32_t offset = threadIdx.x % kColumnsPerUnit; offset < stride;
                 offset += kColumnsPerUnit)
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
            {
                // Divergent constant-memory indices serialize each warp by
                // distinct index. Read four warp-uniform words, then select
                // the nibble's exact signed byte in registers instead.
                const auto *words = reinterpret_cast<const unsigned *>(kTierIQ4ValuesCUDA);
                const unsigned packed = value < 4u ? words[0] : value < 8u ? words[1]
                    : value < 12u ? words[2] : words[3];
                return static_cast<std::int8_t>(packed >> ((value & 3u) * 8u));
            }
            return static_cast<int>(value);
        }

        /**
         * @brief Lossless native-payload transpose for all compact multi-scale formats.
         * @tparam ToCpu True for demotion, false for promotion.
         * @param source Immutable separated GPU source (demotion only).
         * @param destination Mutable separated GPU destination (promotion only).
         * @param cpu_source Complete received CPU units (promotion only).
         * @param cpu_destination Complete outgoing CPU units (demotion only).
         * @param layout Validated immutable format and stride contract.
         * @param first_unit Absolute unit origin of this bounded transfer chunk.
         * @param local_unit Chunk-relative unit selected by this 64-lane group.
         * @param active False for an unused tail group; it still joins CTA barriers.
         *
         * One CTA owns a unit; each lane owns one output column. Padding and
         * unused metadata are zeroed before valid lanes publish. No floating
         * arithmetic, lookup, allocation or synchronization with inference is
         * required: scale/minimum bits and IQ1_M delta signs move unchanged.
         */
        template <bool ToCpu>
        __device__ __forceinline__ void compactMultiScaleUnitsCUDAUnit(
            ExpertTierGpuConstProjectionView source,
            ExpertTierGpuMutableProjectionView destination,
            const std::uint8_t *cpu_source,
            std::uint8_t *cpu_destination,
            ExpertTierWeightDeviceLayout layout,
            std::uint32_t first_unit,
            std::uint32_t local_unit, bool active = true)
        {
            const unsigned unit_index = first_unit + local_unit;
            const int column = static_cast<int>(threadIdx.x % kColumnsPerUnit);
            const int n = static_cast<int>(unit_index / layout.blocks_per_row) * 64 + column;
            const int kb = static_cast<int>(unit_index % layout.blocks_per_row);
            const size_t offset = static_cast<size_t>(local_unit) * layout.cpu_block_stride;
            if constexpr (ToCpu)
            {
                if (active) initializeCpuUnit(cpu_destination + offset, layout.cpu_block_stride);
                __syncthreads();
            }
            if (!active || n >= layout.N)
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

        /** Native entry retains the existing grid and forwards one complete unit. */
        template <bool ToCpu>
        __global__ void compactMultiScaleUnitsCUDA(
            ExpertTierGpuConstProjectionView source,
            ExpertTierGpuMutableProjectionView destination,
            const std::uint8_t *cpu_source,
            std::uint8_t *cpu_destination,
            ExpertTierWeightDeviceLayout layout,
            std::uint32_t first_unit)
        {
            compactMultiScaleUnitsCUDAUnit<ToCpu>(
                source, destination, cpu_source, cpu_destination, layout, first_unit, blockIdx.x);
        }

        /** GPU-separated nibble projection to final CPU unit bytes. */
        template <std::uint8_t Codebook, bool Asymmetric>
        __device__ __forceinline__ void gpuToCpuNibbleUnitsCUDAUnit(
            const std::uint8_t *__restrict__ payload,
            const std::uint16_t *__restrict__ scales,
            const std::uint16_t *__restrict__ mins,
            int N,
            int blocks_per_row,
            std::uint32_t first_unit,
            std::uint32_t cpu_stride,
            std::uint8_t *__restrict__ cpu_chunk,
            std::uint32_t local_unit, bool active = true)
        {
            const std::uint32_t global_unit = first_unit + local_unit;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb =
                static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x % kColumnsPerUnit);
            std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(local_unit) * cpu_stride;

            if (active) initializeCpuUnit(unit, cpu_stride);
            __syncthreads();

            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (!active || n >= N)
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

        /** Native entry retains the existing grid and forwards one complete unit. */
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
            gpuToCpuNibbleUnitsCUDAUnit<Codebook, Asymmetric>(
                payload, scales, mins, N, blocks_per_row, first_unit, cpu_stride, cpu_chunk, blockIdx.x);
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
         * @brief Losslessly expand a single-scale source into CPU unit bytes.
         * @tparam Codebook One of 6, 7, 11, 12, 15, 16, raw-INT8 19, or normalized
         *         asymmetric INT8+minimum 23.
         *
         * Each thread owns one logical output column.  The shared production
         * decoder reconstructs source integers; direct Q5/Q8 formats retain
         * those integers, single-scale IQ formats preserve their native grids
         * and scale/minimum without any floating-point conversion. Padded columns remain
         * deterministic zeros from `initializeCpuUnit`.
         */
        template <std::uint8_t Codebook>
        __device__ __forceinline__ void gpuToCpuDecodedUnitsCUDAUnit(
            const std::uint8_t *__restrict__ payload,
            const std::uint16_t *__restrict__ scales,
            const std::uint16_t *__restrict__ mins,
            const std::uint32_t *__restrict__ emins,
            int N,
            int blocks_per_row,
            std::uint32_t first_unit,
            std::uint32_t cpu_stride,
            std::uint8_t *__restrict__ cpu_chunk,
            std::uint32_t local_unit, bool active = true)
        {
            const std::uint32_t global_unit = first_unit + local_unit;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb = static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x % kColumnsPerUnit);
            std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(local_unit) * cpu_stride;

            if (active) initializeCpuUnit(unit, cpu_stride);
            __syncthreads();

            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (!active || n >= N)
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

            static_assert(Codebook == 6 || Codebook == 7 || Codebook == 11 ||
                          Codebook == 12 || Codebook == 15 || Codebook == 16 ||
                          Codebook == 19 || Codebook == kNativeVnniExpandedInt8MinCodebook);
            // These single-scale formats retain their exact integer domain.
            // Multi-scale sources use the native transposition kernel instead.
            for (int group = 0; group < 8; ++group)
            {
                *reinterpret_cast<std::uint32_t *>(
                    unit + interleavedValueOffset(local_column, group * 4)) =
                    static_cast<std::uint32_t>(decoded_groups[group]);
            }

            // CPU VNNI kernels shift activations to unsigned and require the
            // exact signed-weight sum for every expanded block.
            int compensation = 0;
            for (int value = 0; value < 32; ++value)
            {
                compensation += decodedGroupValue(decoded_groups, value);
            }
            reinterpret_cast<std::int16_t *>(unit + 2048u)[local_column] =
                static_cast<std::int16_t>(compensation);
            reinterpret_cast<std::uint16_t *>(unit + 2176u)[local_column] =
                scales[linear];
            if constexpr (
                Codebook == 7 || Codebook == 16 ||
                Codebook == kNativeVnniExpandedInt8MinCodebook)
            {
                reinterpret_cast<std::uint16_t *>(unit + 2304u)[local_column] =
                    mins[linear];
            }
        }

        /** Native entry retains the existing grid and forwards one complete unit. */
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
            gpuToCpuDecodedUnitsCUDAUnit<Codebook>(
                payload, scales, mins, emins, N, blocks_per_row, first_unit, cpu_stride, cpu_chunk, blockIdx.x);
        }

        /** GPU-separated native Q6_K projection to CPU dual-scale unit bytes. */
        __device__ __forceinline__ void gpuToCpuQ6UnitsCUDAUnit(
            const std::uint8_t *__restrict__ payload,
            const std::uint16_t *__restrict__ scales,
            const std::uint16_t *__restrict__ secondary_scales,
            int N,
            int blocks_per_row,
            std::uint32_t first_unit,
            std::uint32_t cpu_stride,
            std::uint8_t *__restrict__ cpu_chunk,
            std::uint32_t local_unit, bool active = true)
        {
            const std::uint32_t global_unit = first_unit + local_unit;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb =
                static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x % kColumnsPerUnit);
            std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(local_unit) * cpu_stride;

            if (active) initializeCpuUnit(unit, cpu_stride);
            __syncthreads();
            if (!active) return;

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

            if (local_column < 32)
            {
                const int group = local_column / 4;
                const int zmm = local_column % 4;
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

        /** Native entry retains the existing grid and forwards one complete unit. */
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
            gpuToCpuQ6UnitsCUDAUnit(
                payload, scales, secondary_scales, N, blocks_per_row, first_unit, cpu_stride, cpu_chunk, blockIdx.x);
        }

        /** CPU nibble units to separated GPU projection. */
        template <bool Asymmetric>
        __device__ __forceinline__ void cpuToGpuNibbleUnitsCUDAUnit(
            const std::uint8_t *__restrict__ cpu_chunk,
            int N,
            int blocks_per_row,
            std::uint32_t first_unit,
            std::uint32_t cpu_stride,
            std::uint8_t *__restrict__ payload,
            std::uint16_t *__restrict__ scales,
            std::uint16_t *__restrict__ mins,
            std::uint32_t local_unit)
        {
            const std::uint32_t global_unit = first_unit + local_unit;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb =
                static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x % kColumnsPerUnit);
            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (n >= N)
                return;

            const std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(local_unit) * cpu_stride;
            const std::size_t linear =
                static_cast<std::size_t>(kb) * N + n;
            std::uint8_t *destination = payload + linear * 16u;
            gatherCpuPayloadVectors<16>(unit, local_column, destination);
            scales[linear] =
                reinterpret_cast<const std::uint16_t *>(unit + 1152u)[local_column];
            if constexpr (Asymmetric)
            {
                mins[linear] = reinterpret_cast<const std::uint16_t *>(
                    unit + 1280u)[local_column];
            }
        }

        /** Native entry retains the existing grid and forwards one complete unit. */
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
            cpuToGpuNibbleUnitsCUDAUnit<Asymmetric>(
                cpu_chunk, N, blocks_per_row, first_unit, cpu_stride, payload, scales, mins, blockIdx.x);
        }

        /** CPU expanded INT8 units to separated GPU projection. */
        template <bool Asymmetric>
        __device__ __forceinline__ void cpuToGpuExpandedUnitsCUDAUnit(
            const std::uint8_t *__restrict__ cpu_chunk,
            int N,
            int blocks_per_row,
            std::uint32_t first_unit,
            std::uint32_t cpu_stride,
            std::uint8_t *__restrict__ payload,
            std::uint16_t *__restrict__ scales,
            std::uint16_t *__restrict__ mins,
            std::uint32_t local_unit)
        {
            const std::uint32_t global_unit = first_unit + local_unit;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb =
                static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x % kColumnsPerUnit);
            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (n >= N)
                return;

            const std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(local_unit) * cpu_stride;
            const std::size_t linear =
                static_cast<std::size_t>(kb) * N + n;
            std::uint8_t *destination = payload + linear * 32u;
            gatherCpuPayloadVectors<32>(unit, local_column, destination);
            scales[linear] =
                reinterpret_cast<const std::uint16_t *>(unit + 2176u)[local_column];
            if constexpr (Asymmetric)
            {
                mins[linear] = reinterpret_cast<const std::uint16_t *>(
                    unit + 2304u)[local_column];
            }
        }

        /** Native entry retains the existing grid and forwards one complete unit. */
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
            cpuToGpuExpandedUnitsCUDAUnit<Asymmetric>(
                cpu_chunk, N, blocks_per_row, first_unit, cpu_stride, payload, scales, mins, blockIdx.x);
        }

        /** CPU native Q6_K units to separated GPU projection. */
        __device__ __forceinline__ void cpuToGpuQ6UnitsCUDAUnit(
            const std::uint8_t *__restrict__ cpu_chunk,
            int N,
            int blocks_per_row,
            std::uint32_t first_unit,
            std::uint32_t cpu_stride,
            std::uint8_t *__restrict__ payload,
            std::uint16_t *__restrict__ scales,
            std::uint16_t *__restrict__ secondary_scales,
            std::uint32_t local_unit)
        {
            const std::uint32_t global_unit = first_unit + local_unit;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb =
                static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x % kColumnsPerUnit);
            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (n >= N)
                return;

            const std::uint8_t *unit =
                cpu_chunk + static_cast<std::size_t>(local_unit) * cpu_stride;
            const std::size_t linear =
                static_cast<std::size_t>(kb) * N + n;
            std::uint8_t *destination = payload + linear * 24u;
            gatherCpuPayloadVectors<16, 8>(unit, local_column, destination);
            const int zmm = local_column / 16;
            const int lane = local_column % 16;
            std::uint64_t high_bytes = 0u;
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
                high_bytes |= static_cast<std::uint64_t>(packed_high) << (8u * group);
            }
            *reinterpret_cast<std::uint64_t *>(destination + 16u) = high_bytes;
            scales[linear] =
                reinterpret_cast<const std::uint16_t *>(unit + 1536u)[local_column];
            secondary_scales[linear] =
                reinterpret_cast<const std::uint16_t *>(unit + 1664u)[local_column];
        }

        /** Native entry retains the existing grid and forwards one complete unit. */
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
            cpuToGpuQ6UnitsCUDAUnit(
                cpu_chunk, N, blocks_per_row, first_unit, cpu_stride, payload, scales, secondary_scales, blockIdx.x);
        }

        /**
         * @brief One operation implementation under the shared GPU claim lifecycle.
         *
         * The descriptor is immutable until the service receipt releases it.
         * A quantum contains whole CPU units; the existing staging allocation
         * absorbs scattered unit accesses while only coalesced copies cross the
         * mapped-host link. No launch, allocation or host wait occurs here.
         */
        struct CUDAQueuedTransferPayload
        {
            /**
             * @brief Access parameters after the common worker checked every complement.
             * @param command Immutable snapshot held by the device claim.
             * @return Typed scalar snapshot; no host pointer is followed.
             */
            __device__ static __forceinline__ MappedTransferPackedWork work(
                const MappedTransferProgressCommand &command)
            {
                // Copy the object representation instead of aliasing an integer
                // array as a different C++ object. Fixed-size inlining keeps the
                // used fields in registers; no allocation or runtime copy API.
                MappedTransferPackedWork result;
                memcpy(&result, command.work, sizeof(result));
                return result;
            }

            /**
             * @brief Reject malformed dispatch geometry before division or dereference.
             * @param command Complement-validated snapshot of one generation.
             * @return Whether the operation has a complete supported device dispatch.
             */
            __device__ static __forceinline__ bool valid(
                const MappedTransferProgressCommand &command)
            {
                const auto &description = work(command);
                if (description.kind == MappedTransferWorkKind::Bytes) return true;
                const bool to_cpu = description.kind == MappedTransferWorkKind::PackedGpuToCpu;
                if (!to_cpu && description.kind != MappedTransferWorkKind::PackedCpuToGpu)
                    return false;
                const auto &layout = description.layout;
                // Full format/footprint validation belongs to the canonical
                // host layout. These device checks protect the dynamic loop;
                // exact word complements cover its immutable format metadata.
                if (!description.device_staging || !description.scales ||
                    (layout.gpu_is_asymmetric && !description.mins) ||
                    (layout.gpu_has_emins && !description.emins) ||
                    layout.N <= 0 || layout.K <= 0 || layout.K % 32 != 0 ||
                    layout.blocks_per_row != layout.K / 32 ||
                    !layout.cpu_block_stride || command.bytes % layout.cpu_block_stride ||
                    (to_cpu ? layout.direction != ExpertTierWeightConversionDirection::GpuToCpu
                            : layout.direction != ExpertTierWeightConversionDirection::CpuToGpu))
                    return false;
                const auto units = command.bytes / layout.cpu_block_stride;
                if (!units || description.first_unit >= layout.unit_count ||
                    units > layout.unit_count - description.first_unit ||
                    units > layout.maximum_units_per_chunk)
                    return false;
                // This is dispatch totality, not a second footprint authority.
                // Never acknowledge bytes for a codebook with no device branch.
                using Encoding = cpu::native_vnni::CPUNativeVNNIEncoding;
                switch (layout.cpu_encoding)
                {
                case Encoding::NibbleLUT:
                    return layout.gpu_codebook_id == 0 || layout.gpu_codebook_id == 4 ||
                        layout.gpu_codebook_id == 5;
                case Encoding::Q6KNativeDualScale:
                    return layout.gpu_codebook_id == 8;
                case Encoding::CompactMultiScale:
                    return layout.gpu_codebook_id == 9 || layout.gpu_codebook_id == 10 ||
                        layout.gpu_codebook_id == 13 || layout.gpu_codebook_id == 14 ||
                        layout.gpu_codebook_id == 17;
                case Encoding::ExpandedInt8:
                    if (!to_cpu)
                        return layout.gpu_codebook_id == 19 ||
                            layout.gpu_codebook_id == kNativeVnniExpandedInt8MinCodebook;
                    switch (layout.gpu_codebook_id)
                    {
                    case 6: case 7: case 11: case 12: case 15: case 16: case 19:
                    case kNativeVnniExpandedInt8MinCodebook:
                        return true;
                    default:
                        return false;
                    }
                }
                return false;
            }

            /**
             * @brief Cooperatively copy a bounded range, including unaligned tails.
             * @param destination Unique writable extent owned by this command.
             * @param source Immutable readable extent owned by this command.
             * @param bytes Exact extent; all CTA lanes must pass the same value.
             */
            __device__ static __forceinline__ void copy(
                unsigned char *destination, const unsigned char *source, std::uint64_t bytes)
            {
                if (((reinterpret_cast<std::uintptr_t>(destination) |
                      reinterpret_cast<std::uintptr_t>(source)) & 15u) == 0u)
                {
                    const auto vectors = bytes / sizeof(uint4);
                    for (auto index = static_cast<std::uint64_t>(threadIdx.x);
                         index < vectors; index += blockDim.x)
                        reinterpret_cast<uint4 *>(destination)[index] =
                            reinterpret_cast<const uint4 *>(source)[index];
                    for (auto index = vectors * sizeof(uint4) + threadIdx.x;
                         index < bytes; index += blockDim.x)
                        destination[index] = source[index];
                }
                else
                    for (auto index = static_cast<std::uint64_t>(threadIdx.x);
                         index < bytes; index += blockDim.x)
                        destination[index] = source[index];
            }

            /**
             * @brief Convert one complete unit using the native launch arithmetic.
             * @param command Validated source, destination and staging geometry.
             * @param unit Chunk-local unit index; the descriptor owns its global origin.
             * @param active Whether this 64-lane group owns a complete unit in the batch.
             */
            __device__ static __forceinline__ void convertUnit(
                const MappedTransferProgressCommand &command, std::uint32_t unit, bool active)
            {
                const auto &description = work(command);
                const auto &layout = description.layout;
                auto *chunk = reinterpret_cast<std::uint8_t *>(description.device_staging);
                const bool to_cpu = description.kind == MappedTransferWorkKind::PackedGpuToCpu;
                const auto gpu_address = to_cpu ? command.source_address : command.destination_address;
                ExpertTierGpuConstProjectionView source{};
                source.payload = reinterpret_cast<const std::uint8_t *>(gpu_address);
                source.scales = reinterpret_cast<const std::uint16_t *>(description.scales);
                source.mins = reinterpret_cast<const std::uint16_t *>(description.mins);
                source.emins = reinterpret_cast<const std::uint32_t *>(description.emins);
                ExpertTierGpuMutableProjectionView destination{};
                destination.payload = reinterpret_cast<std::uint8_t *>(gpu_address);
                destination.scales = reinterpret_cast<std::uint16_t *>(description.scales);
                destination.mins = reinterpret_cast<std::uint16_t *>(description.mins);
                destination.emins = reinterpret_cast<std::uint32_t *>(description.emins);
                using Encoding = cpu::native_vnni::CPUNativeVNNIEncoding;
                if (layout.cpu_encoding == Encoding::CompactMultiScale)
                {
                    if (to_cpu)
                        compactMultiScaleUnitsCUDAUnit<true>(source, {}, nullptr, chunk,
                            layout, description.first_unit, unit, active);
                    else
                        compactMultiScaleUnitsCUDAUnit<false>({}, destination, chunk, nullptr,
                            layout, description.first_unit, unit, active);
                    return;
                }
                if (to_cpu)
                {
#define LLAMINAR_TRANSFER_DEMOTE_UNIT(FUNCTION, ...) \
    FUNCTION<__VA_ARGS__>(source.payload, source.scales, source.mins, layout.N, \
        layout.blocks_per_row, description.first_unit, layout.cpu_block_stride, chunk, unit, active)
                    if (layout.cpu_encoding == Encoding::NibbleLUT)
                    {
                        if (layout.gpu_codebook_id == 0)
                            LLAMINAR_TRANSFER_DEMOTE_UNIT(gpuToCpuNibbleUnitsCUDAUnit, 0, false);
                        else if (layout.gpu_codebook_id == 4)
                            LLAMINAR_TRANSFER_DEMOTE_UNIT(gpuToCpuNibbleUnitsCUDAUnit, 4, false);
                        else
                            LLAMINAR_TRANSFER_DEMOTE_UNIT(gpuToCpuNibbleUnitsCUDAUnit, 5, true);
                    }
                    else if (layout.cpu_encoding == Encoding::ExpandedInt8)
                    {
#define LLAMINAR_TRANSFER_DECODED_UNIT(CODEBOOK) \
    case CODEBOOK: gpuToCpuDecodedUnitsCUDAUnit<CODEBOOK>(source.payload, source.scales, \
        source.mins, source.emins, layout.N, layout.blocks_per_row, description.first_unit, \
        layout.cpu_block_stride, chunk, unit, active); break
                        switch (layout.gpu_codebook_id)
                        {
                        LLAMINAR_TRANSFER_DECODED_UNIT(6);
                        LLAMINAR_TRANSFER_DECODED_UNIT(7);
                        LLAMINAR_TRANSFER_DECODED_UNIT(11);
                        LLAMINAR_TRANSFER_DECODED_UNIT(12);
                        LLAMINAR_TRANSFER_DECODED_UNIT(15);
                        LLAMINAR_TRANSFER_DECODED_UNIT(16);
                        LLAMINAR_TRANSFER_DECODED_UNIT(19);
                        LLAMINAR_TRANSFER_DECODED_UNIT(kNativeVnniExpandedInt8MinCodebook);
                        }
#undef LLAMINAR_TRANSFER_DECODED_UNIT
                    }
                    else
                        gpuToCpuQ6UnitsCUDAUnit(source.payload, source.scales, source.mins,
                            layout.N, layout.blocks_per_row, description.first_unit,
                            layout.cpu_block_stride, chunk, unit, active);
#undef LLAMINAR_TRANSFER_DEMOTE_UNIT
                    return;
                }
                // Promotion has no internal CTA barrier. Demotion above must
                // let unused tail groups join its initialization barrier first.
                if (!active) return;
#define LLAMINAR_TRANSFER_PROMOTE_UNIT(FUNCTION, ASYMMETRIC) \
    FUNCTION<ASYMMETRIC>(chunk, layout.N, layout.blocks_per_row, description.first_unit, \
        layout.cpu_block_stride, destination.payload, destination.scales, destination.mins, unit)
                if (layout.cpu_encoding == Encoding::NibbleLUT)
                {
                    if (layout.cpu_is_asymmetric)
                        LLAMINAR_TRANSFER_PROMOTE_UNIT(cpuToGpuNibbleUnitsCUDAUnit, true);
                    else
                        LLAMINAR_TRANSFER_PROMOTE_UNIT(cpuToGpuNibbleUnitsCUDAUnit, false);
                }
                else if (layout.cpu_encoding == Encoding::ExpandedInt8)
                {
                    if (layout.cpu_is_asymmetric)
                        LLAMINAR_TRANSFER_PROMOTE_UNIT(cpuToGpuExpandedUnitsCUDAUnit, true);
                    else
                        LLAMINAR_TRANSFER_PROMOTE_UNIT(cpuToGpuExpandedUnitsCUDAUnit, false);
                }
                else
                    cpuToGpuQ6UnitsCUDAUnit(chunk, layout.N, layout.blocks_per_row,
                        description.first_unit, layout.cpu_block_stride,
                        destination.payload, destination.scales, destination.mins, unit);
#undef LLAMINAR_TRANSFER_PROMOTE_UNIT
            }

            /**
             * @brief Advance one coalesced byte quantum or complete-unit batch.
             * @param command Immutable, validated snapshot held under the GPU claim.
             * @param position Already-published byte extent of this generation.
             * @param quantum Maximum ordinary batch size; one whole unit always fits.
             * @return Exact completed extent, which every CTA lane computes identically.
             */
            __device__ static __forceinline__ std::uint64_t advance(
                const MappedTransferProgressCommand &command, std::uint64_t position,
                std::uint64_t quantum)
            {
                const auto remaining = command.bytes - position;
                const auto &description = work(command);
                auto *destination = reinterpret_cast<unsigned char *>(command.destination_address);
                const auto *source = reinterpret_cast<const unsigned char *>(command.source_address);
                if (description.kind == MappedTransferWorkKind::Bytes)
                {
                    const auto count = remaining < quantum ? remaining : quantum;
                    copy(destination + position, source + position, count);
                    return count;
                }
                const auto stride = description.layout.cpu_block_stride;
                const auto batch_units = quantum / stride ? quantum / stride : 1u;
                const auto available_units = remaining / stride;
                const auto units = available_units < batch_units ? available_units : batch_units;
                const auto count = units * stride;
                auto *staging = reinterpret_cast<unsigned char *>(description.device_staging);
                if (description.kind == MappedTransferWorkKind::PackedCpuToGpu)
                    copy(staging + position, source + position, count);
                __syncthreads();
                const auto unit_groups = blockDim.x / kColumnsPerUnit;
                for (std::uint64_t batch = 0u; batch < units; batch += unit_groups)
                {
                    // Every group owns one whole unit and preserves the native
                    // arithmetic. All groups take the same number of barriers;
                    // absent tail groups use a valid address but perform no IO.
                    const auto offset = batch + threadIdx.x / kColumnsPerUnit;
                    const bool active = offset < units;
                    const auto unit = position / stride + (active ? offset : 0u);
                    convertUnit(command, static_cast<std::uint32_t>(unit), active);
                    __syncthreads();
                }
                if (description.kind == MappedTransferWorkKind::PackedGpuToCpu)
                    copy(destination + position, staging + position, count);
                return count;
            }
        };

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

    bool prepareMappedTransferServiceCUDA() noexcept
    {
        cudaFuncAttributes attributes{};
        return cudaFuncGetAttributes(&attributes,
            mappedTransferServiceKernel<CUDAQueuedTransferPayload>) == cudaSuccess;
    }

    bool launchMappedTransferServiceCUDA(
        const MappedTransferProgressCommand *commands,
        MappedTransferProgressCompletion *completions,
        MappedTransferServiceCursor *cursors, std::size_t capacity,
        std::size_t maximum_bytes, const std::uint64_t *wake,
        MappedTransferServiceRun run, void *stream) noexcept
    {
        if (!stream || !commands || !completions || !cursors || !capacity ||
            !maximum_bytes ||
            (run != MappedTransferServiceRun::CapturedInterval &&
             run != MappedTransferServiceRun::FinitePass) ||
            (run == MappedTransferServiceRun::CapturedInterval && !wake) ||
            cudaPeekAtLastError() != cudaSuccess)
            return false;
        // Four independent 64-lane unit groups expose enough outstanding reads
        // for mapped-host upload without adding service CTAs/SMs. Native repack
        // launches still use one unit per CTA and the exact same arithmetic.
        const auto blocks = run == MappedTransferServiceRun::CapturedInterval
            ? 1u : static_cast<unsigned>(capacity < 4u ? capacity : 4u);
        mappedTransferServiceKernel<CUDAQueuedTransferPayload><<<blocks, 256u, 0u,
            reinterpret_cast<cudaStream_t>(stream)>>>(
                commands, completions, cursors, capacity, maximum_bytes, wake, run);
        return cudaGetLastError() == cudaSuccess;
    }

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
        if (layout.cpu_encoding == cpu::native_vnni::CPUNativeVNNIEncoding::CompactMultiScale)
        {
            compactMultiScaleUnitsCUDA<true><<<grid, block, 0, cuda_stream>>>(
                source, {}, nullptr, device_cpu_chunk, layout, first_unit);
            return cudaPeekAtLastError() == cudaSuccess;
        }
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
            case 11:
                LLAMINAR_LAUNCH_DECODED_SOURCE(11);
                break;
            case 12:
                LLAMINAR_LAUNCH_DECODED_SOURCE(12);
                break;
            case 15:
                LLAMINAR_LAUNCH_DECODED_SOURCE(15);
                break;
            case 16:
                LLAMINAR_LAUNCH_DECODED_SOURCE(16);
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
            cpu::native_vnni::CPUNativeVNNIEncoding::CompactMultiScale)
        {
            compactMultiScaleUnitsCUDA<false><<<grid, block, 0, cuda_stream>>>(
                {}, destination, device_cpu_chunk, nullptr, layout, first_unit);
            return cudaPeekAtLastError() == cudaSuccess;
        }
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
