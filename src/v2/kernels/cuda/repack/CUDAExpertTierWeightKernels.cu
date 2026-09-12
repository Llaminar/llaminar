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
        __global__ void compactMultiScaleUnitsCUDA(
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
                initializeCpuUnit(cpu_destination + offset, layout.cpu_block_stride);
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

            static_assert(Codebook == 6 || Codebook == 7 || Codebook == 11 ||
                          Codebook == 12 || Codebook == 15 || Codebook == 16 ||
                          Codebook == 19 || Codebook == kNativeVnniExpandedInt8MinCodebook);
            // These single-scale formats retain their exact integer domain.
            // Multi-scale sources use the native transposition kernel instead.
            for (int value = 0; value < 32; ++value)
            {
                unit[interleavedValueOffset(local_column, value)] =
                    static_cast<std::uint8_t>(static_cast<std::int8_t>(
                        decodedGroupValue(decoded_groups, value)));
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
                scales[linear];
            if constexpr (
                Codebook == 7 || Codebook == 16 ||
                Codebook == kNativeVnniExpandedInt8MinCodebook)
            {
                reinterpret_cast<std::uint16_t *>(unit + 2304u)[local_column] =
                    mins[linear];
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
