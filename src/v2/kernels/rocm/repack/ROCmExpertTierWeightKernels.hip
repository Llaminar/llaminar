/**
 * @file ROCmExpertTierWeightKernels.hip
 * @brief HIP kernels for direct GPU/CPU ExpertOverlay weight streaming.
 *
 * CUDA and ROCm consume the same separated GPU representation. This backend
 * therefore implements the identical one-block-per-CPU-unit arithmetic and
 * differs only in runtime launch/error handling. Complete unit ownership keeps
 * writes deterministic on both 32- and 64-lane hardware execution widths.
 * Coded formats such as IQ3_S retain their source payload and metadata planes,
 * without decoding or re-encoding signed grids at the tier boundary.
 */

#include "kernels/rocm/repack/ROCmExpertTierWeightKernels.h"
#include "kernels/common/NativeQuantisedScaleLoad.h"
#include "tensors/MXFP4Format.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedNibbleLUT.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace llaminar2
{
    namespace
    {
        constexpr int kColumnsPerUnit = 64;

        __device__ __constant__ std::int8_t kTierIQ4ValuesROCm[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10,
            1, 13, 25, 38, 53, 69, 89, 113};

        __device__ __forceinline__ void initializeCpuUnitROCm(std::uint8_t *, std::uint32_t);

        /** @brief Transpose a compensation-free CPU unit without decoding payloads or staging scale words. */
        template <bool ToCpu>
        __device__ __forceinline__ void nativeSingleScaleUnitsROCmUnit(
            ExpertTierGpuConstProjectionView source, ExpertTierGpuMutableProjectionView destination,
            const uint8_t *wire_in, uint8_t *wire_out, ExpertTierWeightDeviceLayout layout,
            uint32_t first_unit, uint32_t local_unit, bool active = true)
        {
            if (!active)
                return;
            const auto unit = first_unit + local_unit;
            const unsigned column = threadIdx.x % kColumnsPerUnit;
            const unsigned columns = expertTierWireColumns(layout, unit);
            if (column >= columns)
                return;
            const auto block = unit % layout.blocks_per_row + layout.gpu_first_column / 32;
            const auto row = unit / layout.blocks_per_row * 64 + layout.gpu_first_row + column;
            const auto linear = size_t(block) * layout.gpu_N + row;
            const auto offset = expertTierWireOffset(layout, first_unit, local_unit);
            const unsigned bytes = layout.gpu_payload_bytes_per_block;
            const bool q5 = layout.cpu_encoding == cpu::native_quantised::CPUNativeQuantisedEncoding::Q5NativeSingleScale;
            for (unsigned byte = 0; byte < (q5 ? 16u : bytes); ++byte)
            {
                const auto wire_byte = offset + (byte / 4) * (4 * columns) + column * 4 + byte % 4;
                if constexpr (ToCpu)
                    wire_out[wire_byte] = source.payload[linear * bytes + byte];
                else
                    destination.payload[linear * bytes + byte] = wire_in[wire_byte];
            }
            if (q5)
            {
                const auto qh = offset + 16 * columns + 4 * column;
                for (unsigned byte = 0; byte < 4; ++byte)
                    if constexpr (ToCpu)
                        wire_out[qh + byte] = expertTierQ5FifthFromGpu(source.payload + linear * bytes + 16, byte);
                    else
                        destination.payload[linear * bytes + 16 + byte] = expertTierQ5FifthFromCpu(wire_in + qh, byte);
            }
            const unsigned width = nativeQuantisedScalePlaneBytes(layout.gpu_codebook_id);
            const auto primary = offset + bytes * columns + column * width;
            if (width == 1)
            {
                const auto exponent = ToCpu ? reinterpret_cast<const uint8_t *>(source.scales)[linear] : wire_in[primary];
                if (exponent != 0 && !mxfp4::exponentFitsNativeE8M0(exponent))
                {
                    __builtin_trap();
                    return;
                }
            }
            for (unsigned byte = 0; byte < width; ++byte)
                if constexpr (ToCpu)
                    wire_out[primary + byte] = reinterpret_cast<const uint8_t *>(source.scales)[linear * width + byte];
                else
                    reinterpret_cast<uint8_t *>(destination.scales)[linear * width + byte] = wire_in[primary + byte];
            if (layout.gpu_is_asymmetric)
            {
                const auto secondary = offset + (bytes + width) * columns + column * 2;
                for (unsigned byte = 0; byte < 2; ++byte)
                    if constexpr (ToCpu)
                        wire_out[secondary + byte] = reinterpret_cast<const uint8_t *>(source.mins)[linear * 2 + byte];
                    else
                        reinterpret_cast<uint8_t *>(destination.mins)[linear * 2 + byte] = wire_in[secondary + byte];
            }
        }

        /** @brief Launch one native wire unit per ROCm block. */
        template <bool ToCpu>
        __global__ void nativeSingleScaleUnitsROCm(
            ExpertTierGpuConstProjectionView source, ExpertTierGpuMutableProjectionView destination,
            const uint8_t *wire_in, uint8_t *wire_out, ExpertTierWeightDeviceLayout layout, uint32_t first_unit)
        {
            nativeSingleScaleUnitsROCmUnit<ToCpu>(source, destination, wire_in, wire_out, layout, first_unit, blockIdx.x);
        }

        /**
         * @brief Transpose one live coded CPU wire unit and its GPU plane coordinates.
         *
         * Payload lanes are disjoint, including the short final payload plane.
         * On download one lane owns each complete packed code byte (or 16-bit
         * field), so unused boundary bits are zero without initialization or
         * CTA synchronization. On upload, adjacent resident codes may belong
         * to different units or pooled matrices; only their own bits change.
         */
        template <bool ToCpu>
        __device__ __forceinline__ void compactMultiScaleUnitsROCmUnit(
            ExpertTierGpuConstProjectionView source,
            ExpertTierGpuMutableProjectionView destination,
            const std::uint8_t *cpu_source,
            std::uint8_t *cpu_destination,
            ExpertTierWeightDeviceLayout layout,
            std::uint32_t first_unit,
            std::uint32_t local_unit, bool active = true)
        {
            namespace sb = native_quantised_superblock;
            if (!active)
                return;
            const unsigned unit = first_unit + local_unit;
            const unsigned column = threadIdx.x % kColumnsPerUnit;
            const unsigned columns = expertTierWireColumns(layout, unit);
            if (column >= columns)
                return;
            const unsigned kb = unit % layout.blocks_per_row;
            const unsigned gpu_kb = kb + layout.gpu_first_column / 32;
            const unsigned gpu_row = (unit / layout.blocks_per_row) * 64 + layout.gpu_first_row + column;
            const auto address = sb::blockMajorAddress(gpu_kb, gpu_row, layout.gpu_N, layout.gpu_superblock_phase);
            const auto offset = expertTierWireOffset(layout, first_unit, local_unit);
            const unsigned payload_bytes = layout.gpu_payload_bytes_per_block;
            const auto planes = expertTierCodedPlanes(layout.gpu_codebook_id);
            const unsigned code_bits = planes.low_bits + planes.high_bits;
            const size_t metadata = offset + payload_bytes * columns;
            const auto bit_shift = layout.gpu_code_bit_shift;
            for (unsigned byte = 0; byte < payload_bytes; ++byte)
            {
                const auto cpu_offset = offset + expertTierCodedPayloadOffset(payload_bytes, columns, column, byte);
                const auto gpu_offset = address.linear * payload_bytes + byte;
                if constexpr (ToCpu)
                    cpu_destination[cpu_offset] = source.payload[gpu_offset];
                else
                    destination.payload[gpu_offset] = cpu_source[cpu_offset];
            }
            if (code_bits)
            {
                if constexpr (ToCpu)
                {
                    const auto *codes = reinterpret_cast<const uint8_t *>(source.scales);
                    const auto put = [&](size_t plane, unsigned width, unsigned code_shift) {
                        if (!width)
                            return;
                        const unsigned field_count = width >= 8 ? 1 : 8 / width;
                        if (column % field_count != 0)
                            return;
                        uint32_t word = 0;
                        for (unsigned i = 0; i < field_count && column + i < columns; ++i)
                        {
                            const auto code = sb::loadPackedBits(codes, (address.linear + i) * code_bits + bit_shift, code_bits);
                            word |= ((code >> code_shift) & ((1u << width) - 1)) << (i * width);
                        }
                        auto *out = cpu_destination + plane + size_t(column) * width / 8;
                        out[0] = uint8_t(word);
                        if (width == 16)
                            out[1] = uint8_t(word >> 8);
                    };
                    put(metadata, planes.low_bits, 0);
                    put(metadata + planes.bytes(columns, planes.low_bits), planes.high_bits, planes.low_bits);
                }
                else
                {
                    const auto code = expertTierLoadCodedColumn(cpu_source + metadata, planes, columns, column);
                    const auto byte_address = reinterpret_cast<uintptr_t>(destination.scales);
                    // A pooled descriptor may point to an interior, unaligned
                    // byte. Align the atomic word down and preserve other bits.
                    sb::storePackedBits(reinterpret_cast<void *>(byte_address & ~uintptr_t{3}),
                        address.linear * code_bits + bit_shift + (byte_address & 3u) * 8, code_bits, code);
                }
            }
            const auto wire = layout.cpuStorage().wireGeometry(columns);
            if (wire.closesSuperblock(kb))
            {
                const auto field = sb::superblockFieldBytes(layout.gpu_codebook_id);
                const auto gpu = address.superblock * field;
                const auto cpu = offset + wire.block_stride + column * field;
                for (unsigned byte = 0; byte < field; ++byte)
                {
                    if constexpr (ToCpu)
                        cpu_destination[cpu + byte] = reinterpret_cast<const uint8_t *>(source.mins)[gpu + byte];
                    else
                        reinterpret_cast<uint8_t *>(destination.mins)[gpu + byte] = cpu_source[cpu + byte];
                }
            }
        }

        /** Native entry retains the existing grid and forwards one complete unit. */
        template <bool ToCpu>
        __global__ void compactMultiScaleUnitsROCm(
            ExpertTierGpuConstProjectionView source,
            ExpertTierGpuMutableProjectionView destination,
            const std::uint8_t *cpu_source,
            std::uint8_t *cpu_destination,
            ExpertTierWeightDeviceLayout layout,
            std::uint32_t first_unit)
        {
            compactMultiScaleUnitsROCmUnit<ToCpu>(
                source, destination, cpu_source, cpu_destination, layout, first_unit, blockIdx.x);
        }

        /**
         * @brief Locate one column/value byte within a CPU VNNI unit.
         * @param local_column Column index from zero through 63.
         * @param value_index Prepared payload byte index for that column.
         * @return Byte offset from the start of the unit.
         */
        __device__ __forceinline__ std::size_t interleavedValueOffsetROCm(
            int local_column,
            int value_index, unsigned columns=64)
        {
            const int group = value_index / 4;
            const int index = value_index % 4;
            const int zmm = local_column / 16;
            const int lane = local_column % 16;
            return static_cast<std::size_t>(group) * (columns*4u) +
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
         * @tparam Codebook Nibble execution codebook 0, 4, 5 or 25 (MXFP4).
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
            if constexpr (Codebook == kNativeQuantisedMXFP4Codebook)
            {
                // Register lookup of the doubled E2M1 values (no constant-memory table).
                constexpr auto lut = rocm::nibble_lut::words<kNativeQuantisedMXFP4Codebook>();
                const std::uint32_t packed = value < 4u ? lut.lo_a : value < 8u ? lut.lo_b
                    : value < 12u ? lut.hi_a : lut.hi_b;
                return static_cast<int>(static_cast<std::int8_t>(packed >> ((value & 3u) * 8u)));
            }
            return static_cast<int>(value);
        }

        /**
         * @brief Transpose separated compact nibbles into final CPU units.
         * @tparam Codebook Nibble execution codebook.
         * @tparam HasSecondaryMetadata Whether a raw secondary word region is present.
         * IQ4_XS copies encoded coefficients unchanged; ordinary asymmetric
         * formats copy their existing binary16 minima.
         */
        template <std::uint8_t Codebook, bool HasSecondaryMetadata>
        __global__ void gpuToCpuNibbleUnitsROCm(
            const std::uint8_t *__restrict__ payload,
            const std::uint16_t *__restrict__ scales,
            const std::uint16_t *__restrict__ mins,
            ExpertTierWeightDeviceLayout layout,
            std::uint32_t first_unit,
            std::uint8_t *__restrict__ cpu_chunk)
        {
            const std::uint32_t global_unit = first_unit + blockIdx.x;
            const int N=layout.N,blocks_per_row=layout.blocks_per_row;
            const unsigned columns=expertTierWireColumns(layout,global_unit);
            const unsigned cpu_stride=(layout.cpu_block_stride/64)*columns;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb = static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x);
            std::uint8_t *unit =
                cpu_chunk + expertTierWireOffset(layout,first_unit,blockIdx.x);

            // Zero padding and metadata gaps before real columns publish bytes.
            initializeCpuUnitROCm(unit, cpu_stride);
            __syncthreads();
            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (n >= N)
                return;

            const std::size_t linear =
                static_cast<size_t>(kb+layout.gpu_first_column/32)*layout.gpu_N+n+layout.gpu_first_row;
            const std::uint8_t *source = payload + linear * 16u;
            int compensation = 0;
            for (int value = 0; value < 16; ++value)
            {
                const std::uint8_t packed = source[value];
                unit[interleavedValueOffsetROCm(local_column, value, columns)] = packed;
                compensation += decodeNibbleForCompensationROCm<Codebook>(
                    static_cast<std::uint8_t>(packed & 0x0fu));
                compensation += decodeNibbleForCompensationROCm<Codebook>(
                    static_cast<std::uint8_t>(packed >> 4));
            }
            reinterpret_cast<std::int16_t *>(unit + (1024u/64)*columns)[local_column] =
                static_cast<std::int16_t>(compensation);
            // CPU unit word: FP16 for 2-byte planes, BF16 power of two for E8M0 planes.
            reinterpret_cast<std::uint16_t *>(unit + (1152u/64)*columns)[local_column] =
                ::llaminar2::native_quantised_scale::loadStagedPrimary<Codebook>(scales, linear);
            if constexpr (HasSecondaryMetadata)
            {
                reinterpret_cast<std::uint16_t *>(unit + (1280u/64)*columns)[local_column] =
                    mins[linear];
            }
        }

        /**
         * @brief Transpose Q6 payload and original factor words into CPU units.
         * @details Metadata is copied verbatim: original half d and two signed
         * subscales. CPU and GPU consumers share that encoding; no half conversion
         * or reconstruction occurs at this movement boundary.
         * @param payload Borrowed compact GPU Q6 payload plane.
         * @param scales Borrowed original half d words.
         * @param secondary_scales Borrowed packed signed coefficient words.
         * @param N Logical matrix rows, excluding allocation capacity.
         * @param blocks_per_row Logical32-value block count per row.
         * @param first_unit First native CPU unit in this bounded transfer.
         * @param cpu_stride Existing1792-byte CPU unit stride.
         * @param cpu_chunk Destination admitted native CPU transfer bytes.
         */
        __global__ void gpuToCpuQ6UnitsROCm(
            const std::uint8_t *__restrict__ payload,
            const std::uint16_t *__restrict__ scales,
            const std::uint16_t *__restrict__ secondary_scales,
            ExpertTierWeightDeviceLayout layout,
            std::uint32_t first_unit,
            std::uint8_t *__restrict__ cpu_chunk)
        {
            const unsigned global_unit=first_unit+blockIdx.x;
            const unsigned columns=expertTierWireColumns(layout,global_unit);
            const unsigned column=static_cast<unsigned>(threadIdx.x);
            auto *unit=cpu_chunk+expertTierWireOffset(layout,first_unit,blockIdx.x);
            if(true) initializeCpuUnitROCm(unit,(layout.cpu_block_stride/64)*columns);
            __syncthreads();
            if(column>=columns)return;
            const unsigned n=(global_unit/layout.blocks_per_row)*64+column;
            const unsigned kb=global_unit%layout.blocks_per_row;
            const size_t linear=size_t(kb+layout.gpu_first_column/32)*layout.gpu_N+n+layout.gpu_first_row;
            for(unsigned value=0;value<16;++value) {
                const size_t encoded=(value/4)*(columns*4)+column*4+value%4;
                unit[encoded]=payload[linear*24+value];
            }
            if(columns==64) {
                // Preserve the existing complete-unit bit-plane representation.
                // One lane owns each full plane pair; there is no shared-word RMW.
                if(column<32) {
                    const unsigned group=column/4,zmm=column%4;
                    uint64_t low=0,high=0;
                    for(unsigned lane=0;lane<16;++lane) {
                        const unsigned source_n=(global_unit/layout.blocks_per_row)*64+zmm*16+lane+layout.gpu_first_row;
                        const size_t source_index=size_t(kb+layout.gpu_first_column/32)*layout.gpu_N+source_n;
                        const unsigned original=payload[source_index*24+16+group];
                        for(unsigned index=0;index<4;++index) {
                            low|=uint64_t((original>>(2*index))&1)<<(lane*4+index);
                            high|=uint64_t((original>>(2*index+1))&1)<<(lane*4+index);
                        }
                    }
                    auto* planes=reinterpret_cast<uint64_t*>(unit+1024+group*64+zmm*16);
                    planes[0]=low;planes[1]=high;
                }
            } else for(unsigned group=0;group<8;++group) {
                unsigned converted=0;const unsigned original=payload[linear*24+16+group];
                for(unsigned index=0;index<4;++index) {
                    converted|=((original>>(2*index))&1)<<index;
                    converted|=((original>>(2*index+1))&1)<<(4+index);
                }
                unit[16*columns+group*columns+column]=static_cast<uint8_t>(converted);
            }
            reinterpret_cast<uint16_t*>(unit+24*columns)[column]=scales[linear];
            reinterpret_cast<uint16_t*>(unit+26*columns)[column]=secondary_scales[linear];
        }

        /**
         * @brief Deinterleave CPU compact-nibble units into separated GPU data.
         * @tparam HasSecondaryMetadata Whether a raw secondary word region is present.
         * IQ4_XS copies encoded coefficients unchanged; ordinary asymmetric
         * formats copy their existing binary16 minima.
         */
        template <bool HasSecondaryMetadata>
        __global__ void cpuToGpuNibbleUnitsROCm(
            const std::uint8_t *__restrict__ cpu_chunk,
            ExpertTierWeightDeviceLayout layout,
            std::uint32_t first_unit,
            std::uint8_t *__restrict__ payload,
            std::uint16_t *__restrict__ scales,
            std::uint16_t *__restrict__ mins)
        {
            const std::uint32_t global_unit = first_unit + blockIdx.x;
            const int N=layout.N,blocks_per_row=layout.blocks_per_row;
            const unsigned columns=expertTierWireColumns(layout,global_unit);
            const unsigned cpu_stride=(layout.cpu_block_stride/64)*columns;
            const int n_chunk =
                static_cast<int>(global_unit / blocks_per_row);
            const int kb = static_cast<int>(global_unit % blocks_per_row);
            const int local_column = static_cast<int>(threadIdx.x);
            const int n = n_chunk * kColumnsPerUnit + local_column;
            if (n >= N)
                return;

            const std::uint8_t *unit =
                cpu_chunk + expertTierWireOffset(layout,first_unit,blockIdx.x);
            const std::size_t linear =
                static_cast<size_t>(kb+layout.gpu_first_column/32)*layout.gpu_N+n+layout.gpu_first_row;
            std::uint8_t *destination = payload + linear * 16u;
            for (int value = 0; value < 16; ++value)
            {
                destination[value] =
                    unit[interleavedValueOffsetROCm(local_column, value, columns)];
            }
            // Exact inverse of the GPU->CPU scale word; a malformed unit is fatal.
            if (!::llaminar2::native_quantised_scale::storePrimaryFromCpuWord(
                    layout.gpu_codebook_id, scales, linear,
                    reinterpret_cast<const std::uint16_t *>(unit + (1152u/64)*columns)[local_column]))
            {
                __builtin_trap();
                return;
            }
            if constexpr (HasSecondaryMetadata)
            {
                mins[linear] = reinterpret_cast<const std::uint16_t *>(
                    unit + (1280u/64)*columns)[local_column];
            }
        }

        /**
         * @brief Deinterleave source-factor CPU Q6 units into separated GPU data.
         * @details Preserve both metadata words exactly across repeated tier cycles;
         * the second word is packed signed coefficients, never a binary16 value.
         * @param cpu_chunk Borrowed source-factor native CPU transfer units.
         * @param N Logical output rows, excluding padded CPU columns.
         * @param blocks_per_row Logical32-value block count per row.
         * @param first_unit First native unit represented by this transfer.
         * @param cpu_stride Existing1792-byte CPU unit stride.
         * @param payload Destination compact GPU Q6 payload plane.
         * @param scales Destination original half d words.
         * @param secondary_scales Destination packed signed coefficient words.
         */
        __global__ void cpuToGpuQ6UnitsROCm(
            const std::uint8_t *__restrict__ cpu_chunk,
            ExpertTierWeightDeviceLayout layout,
            std::uint32_t first_unit,
            std::uint8_t *__restrict__ payload,
            std::uint16_t *__restrict__ scales,
            std::uint16_t *__restrict__ secondary_scales)
        {
            const unsigned global_unit=first_unit+blockIdx.x;
            const unsigned columns=expertTierWireColumns(layout,global_unit);
            const unsigned column=static_cast<unsigned>(threadIdx.x);
            auto *unit=cpu_chunk+expertTierWireOffset(layout,first_unit,blockIdx.x);
            if(column>=columns)return;
            const unsigned n=(global_unit/layout.blocks_per_row)*64+column;
            const unsigned kb=global_unit%layout.blocks_per_row;
            const size_t linear=size_t(kb+layout.gpu_first_column/32)*layout.gpu_N+n+layout.gpu_first_row;
            for(unsigned value=0;value<16;++value) {
                const size_t encoded=(value/4)*(columns*4)+column*4+value%4;
                payload[linear*24+value]=unit[encoded];
            }
            for(unsigned group=0;group<8;++group) {
                unsigned original;
                if(columns==64) {
                    const auto* planes=reinterpret_cast<const uint64_t*>(unit+1024+group*64+(column/16)*16);
                    const auto bit=(column%16)*4;
                    original=static_cast<unsigned>(((planes[0]>>bit)&15)|(((planes[1]>>bit)&15)<<4));
                } else original=unit[16*columns+group*columns+column];
                unsigned converted=0;
                for(unsigned index=0;index<4;++index) {
                    converted|=((original>>index)&1)<<(2*index);
                    converted|=((original>>(4+index))&1)<<(2*index+1);
                }
                payload[linear*24+16+group]=static_cast<uint8_t>(converted);
            }
            scales[linear]=reinterpret_cast<const uint16_t*>(unit+24*columns)[column];
            secondary_scales[linear]=reinterpret_cast<const uint16_t*>(unit+26*columns)[column];
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
                   cpu_chunk_capacity >= layout.chunkBytes(unit_count,first_unit);
        }
        /** @brief Copy one logical bit span without replacing bits owned by neighboring matrices. */
        __global__ void copyCodedExpertBitsROCm(const uint8_t *source, void *destination, size_t bits,
            unsigned source_shift, unsigned destination_shift)
        {
            namespace sb = native_quantised_superblock;
            const size_t bit = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
            if (bit >= bits)
                return;
            const auto width = static_cast<unsigned>(bits - bit < 8 ? bits - bit : 8);
            const auto address = reinterpret_cast<uintptr_t>(destination);
            auto *word = reinterpret_cast<void *>(address & ~uintptr_t{3});
            const auto value = sb::loadPackedBits(source, bit + source_shift, width);
            sb::storePackedBits(word, bit + destination_shift + (address & 3u) * 8, width, value);
        }
        /** @brief Copy one live block; only the first live block of each source superblock writes its header. */
        __global__ void copyCodedPeerRectangleROCm(ROCmCodedPeerProjectionRectangle r)
        {
            namespace sb = native_quantised_superblock;
            const auto item = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (item >= size_t(r.rows) * r.blocks)
                return;
            const unsigned n = item % r.rows;
            const unsigned kb = item / r.rows;
            const auto from = sb::blockMajorAddress(r.src_block + kb, r.src_row + n, r.src_n,
                nativeQuantisedLayoutPhase(r.source.metadata_layout));
            const auto to = sb::blockMajorAddress(r.dst_block + kb, r.dst_row + n, r.dst_n,
                nativeQuantisedLayoutPhase(r.destination.metadata_layout));
            for (unsigned byte = 0; byte < r.payload_bytes; ++byte)
                r.destination.payload[to.linear * r.payload_bytes + byte] = r.source.payload[from.linear * r.payload_bytes + byte];
            const auto bits = sb::superblockCodePlaneBits(r.codebook);
            if (bits)
            {
                const auto src_bit = from.linear * bits + nativeQuantisedLayoutCodeBitShift(r.source.metadata_layout);
                const auto code = sb::loadPackedBits(reinterpret_cast<const uint8_t *>(r.source.scales), src_bit, bits);
                const auto address = reinterpret_cast<uintptr_t>(r.destination.scales);
                auto *aligned = reinterpret_cast<void *>(address & ~uintptr_t{3});
                const auto dst_bit = to.linear * bits + nativeQuantisedLayoutCodeBitShift(r.destination.metadata_layout) + (address & 3u) * 8;
                sb::storePackedBits(aligned, dst_bit, bits, code);
            }
            if (kb == 0 || sb::superblockPhase(r.src_block + kb + nativeQuantisedLayoutPhase(r.source.metadata_layout)) == 0)
            {
                const auto field = sb::superblockFieldBytes(r.codebook);
                const auto *input = reinterpret_cast<const uint8_t *>(r.source.mins) + from.superblock * field;
                auto *output = reinterpret_cast<uint8_t *>(r.destination.mins) + to.superblock * field;
                for (unsigned byte = 0; byte < field; ++byte)
                    output[byte] = input[byte];
            }
        }
    } // namespace

    namespace detail
    {
    bool launchCodedExpertBitCopyROCm(const void *source, void *destination, size_t bits,
        unsigned source_shift, unsigned destination_shift, void *stream) noexcept
    {
        if (reinterpret_cast<uintptr_t>(stream) <= 2 || !source || !destination || !bits ||
            source == destination || source_shift > 7 || destination_shift > 7)
            return false;
        constexpr unsigned threads = 128;
        const auto blocks = (bits - 1) / (threads * 8) + 1;
        if (blocks > std::numeric_limits<int>::max())
            return false;
        copyCodedExpertBitsROCm<<<static_cast<unsigned>(blocks), threads, 0, static_cast<hipStream_t>(stream)>>>(
            static_cast<const uint8_t *>(source), destination, bits, source_shift, destination_shift);
        return hipPeekAtLastError() == hipSuccess;
    }

    } // namespace detail

    bool launchROCmCodedPeerProjectionRectangle(ROCmCodedPeerProjectionRectangle rectangle, void *stream) noexcept
    {
        const auto &r = rectangle;
        if (reinterpret_cast<uintptr_t>(stream) <= 2 || !r.rows || !r.blocks || !r.payload_bytes ||
            r.src_row > r.src_n || r.rows > r.src_n - r.src_row ||
            r.dst_row > r.dst_n || r.rows > r.dst_n - r.dst_row ||
            !r.source.payload || !r.destination.payload || !r.source.scales || !r.destination.scales ||
            !r.source.mins || !r.destination.mins || !r.source.metadata_layout || !r.destination.metadata_layout ||
            !nativeQuantisedLayoutValid(r.codebook, r.source.metadata_layout) ||
            !nativeQuantisedLayoutValid(r.codebook, r.destination.metadata_layout) ||
            native_quantised_superblock::superblockPhase(r.src_block + nativeQuantisedLayoutPhase(r.source.metadata_layout)) !=
                native_quantised_superblock::superblockPhase(r.dst_block + nativeQuantisedLayoutPhase(r.destination.metadata_layout)))
            return false;
        constexpr size_t threads = 128;
        const auto grid = (size_t(rectangle.rows) * rectangle.blocks - 1) / threads + 1;
        if (grid > std::numeric_limits<int>::max())
            return false;
        copyCodedPeerRectangleROCm<<<unsigned(grid), threads, 0, static_cast<hipStream_t>(stream)>>>(rectangle);
        return hipPeekAtLastError() == hipSuccess;
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
        if (layout.cpuStorage().coded())
        {
            auto coded_layout = layout;
            coded_layout.gpu_code_bit_shift = nativeQuantisedLayoutCodeBitShift(source.metadata_layout);
            compactMultiScaleUnitsROCm<true><<<grid, block, 0, hip_stream>>>(
                source, {}, nullptr, device_cpu_chunk, coded_layout, first_unit);
            return hipPeekAtLastError() == hipSuccess;
        }
        if (expertTierPreservesNativeSingleScale(layout.cpu_encoding, layout.cpu_codebook_id))
        {
            nativeSingleScaleUnitsROCm<true><<<grid, block, 0, hip_stream>>>(
                source, {}, nullptr, device_cpu_chunk, layout, first_unit);
            return hipPeekAtLastError() == hipSuccess;
        }
        if (layout.cpu_encoding == cpu::native_quantised::CPUNativeQuantisedEncoding::NibbleLUTSourceScale)
        {
            hipLaunchKernelGGL((gpuToCpuNibbleUnitsROCm<4, true>), grid, block, 0,
                hip_stream, source.payload, source.scales, source.mins, layout, first_unit, device_cpu_chunk);
            return hipPeekAtLastError() == hipSuccess;
        }
        if (layout.cpu_encoding ==
            cpu::native_quantised::CPUNativeQuantisedEncoding::NibbleLUT)
        {
            if (layout.gpu_codebook_id == 0)
            {
                hipLaunchKernelGGL(
                    (gpuToCpuNibbleUnitsROCm<0, false>), grid, block, 0,
                    hip_stream, source.payload, source.scales, source.mins,
                    layout, first_unit, device_cpu_chunk);
            }
            else if (layout.gpu_codebook_id == 4)
            {
                hipLaunchKernelGGL(
                    (gpuToCpuNibbleUnitsROCm<4, false>), grid, block, 0,
                    hip_stream, source.payload, source.scales, source.mins,
                    layout, first_unit, device_cpu_chunk);
            }
            else if (layout.gpu_codebook_id == kNativeQuantisedMXFP4Codebook)
            {
                // Symmetric FP16 scale, no secondary plane; only the lookup differs from 4.
                hipLaunchKernelGGL(
                    (gpuToCpuNibbleUnitsROCm<kNativeQuantisedMXFP4Codebook, false>), grid, block, 0,
                    hip_stream, source.payload, source.scales, source.mins,
                    layout, first_unit, device_cpu_chunk);
            }
            else if (layout.gpu_codebook_id == 5)
            {
                hipLaunchKernelGGL(
                    (gpuToCpuNibbleUnitsROCm<5, true>), grid, block, 0,
                    hip_stream, source.payload, source.scales, source.mins,
                    layout, first_unit, device_cpu_chunk);
            }
            else
            {
                // A nibble layout names exactly one of the decoders above; never
                // reinterpret an unknown codebook as Q4_1 with a nonexistent min plane.
                return false;
            }
        }
        else if (layout.cpu_encoding == cpu::native_quantised::CPUNativeQuantisedEncoding::Q6KNativeDualScale)
        {
            hipLaunchKernelGGL(
                gpuToCpuQ6UnitsROCm, grid, block, 0, hip_stream,
                source.payload, source.scales, source.mins, layout, first_unit,
                device_cpu_chunk);
        }
        else
            return false;
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
        if (layout.cpuStorage().coded())
        {
            auto coded_layout = layout;
            coded_layout.gpu_code_bit_shift = nativeQuantisedLayoutCodeBitShift(destination.metadata_layout);
            compactMultiScaleUnitsROCm<false><<<grid, block, 0, hip_stream>>>(
                {}, destination, device_cpu_chunk, nullptr, coded_layout, first_unit);
            return hipPeekAtLastError() == hipSuccess;
        }
        if (expertTierPreservesNativeSingleScale(layout.cpu_encoding, layout.cpu_codebook_id))
        {
            nativeSingleScaleUnitsROCm<false><<<grid, block, 0, hip_stream>>>(
                {}, destination, device_cpu_chunk, nullptr, layout, first_unit);
            return hipPeekAtLastError() == hipSuccess;
        }
        if (layout.cpu_encoding == cpu::native_quantised::CPUNativeQuantisedEncoding::NibbleLUTSourceScale)
        {
            hipLaunchKernelGGL((cpuToGpuNibbleUnitsROCm<true>), grid, block, 0,
                hip_stream, device_cpu_chunk, layout, first_unit, destination.payload, destination.scales, destination.mins);
            return hipPeekAtLastError() == hipSuccess;
        }
        if (layout.cpu_encoding ==
            cpu::native_quantised::CPUNativeQuantisedEncoding::NibbleLUT)
        {
            if (layout.cpu_is_asymmetric != 0)
            {
                hipLaunchKernelGGL(
                    (cpuToGpuNibbleUnitsROCm<true>), grid, block, 0,
                    hip_stream, device_cpu_chunk, layout, first_unit, destination.payload,
                    destination.scales, destination.mins);
            }
            else
            {
                hipLaunchKernelGGL(
                    (cpuToGpuNibbleUnitsROCm<false>), grid, block, 0,
                    hip_stream, device_cpu_chunk, layout, first_unit, destination.payload,
                    destination.scales, destination.mins);
            }
        }
        else if (layout.cpu_encoding == cpu::native_quantised::CPUNativeQuantisedEncoding::Q6KNativeDualScale)
        {
            hipLaunchKernelGGL(
                cpuToGpuQ6UnitsROCm, grid, block, 0, hip_stream,
                device_cpu_chunk, layout, first_unit, destination.payload,
                destination.scales, destination.mins);
        }
        else
            return false;
        return hipPeekAtLastError() == hipSuccess;
    }
} // namespace llaminar2
