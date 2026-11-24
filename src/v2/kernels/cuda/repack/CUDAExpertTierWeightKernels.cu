/**
 * @file CUDAExpertTierWeightKernels.cu
 * @brief CUDA kernels for direct GPU/CPU ExpertOverlay weight streaming.
 *
 * One thread block owns one complete 64-column by 32-K CPU unit. This gives
 * every output byte exactly one writer and makes arbitrary ordered chunks
 * independent. Resident CPU expert units are either superblock-coded (the
 * compact family, Q4_K, Q6_K and IQ4_XS) or native single-scale (nibble, Q5
 * and Q8); those two converters are the only ones, and any other encoding
 * fails closed.
 * Native launches and the graph-resident transfer worker call the same unit
 * functions. Conversion is therefore not a separately queued prerequisite of
 * background copying and cannot be stranded behind a future inference reader.
 * IQ4_XS physical format 24 preserves the original half base and raw six-bit
 * coefficient in separate 16-bit planes. Its LUT payload and source4 reduction
 * schedule are unchanged; the coefficient is storage metadata, never a minimum.
 */

#include "kernels/cuda/repack/CUDAExpertTierWeightKernels.h"
#include "kernels/common/NativeQuantisedScaleLoad.h"
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

        // The coded and native single-scale converters own every resident unit:
        // Q6_K and IQ4_XS are superblock-coded, nibble units are compensation-free.
        // The retired compensated-nibble and per-block Q6 converters stay retired.
        static_assert(cpu::native_quantised::preparedSuperblockCoded(
                          cpu::native_quantised::CPUNativeQuantisedEncoding::Q6KNativeDualScale) &&
                      cpu::native_quantised::preparedSuperblockCoded(
                          cpu::native_quantised::CPUNativeQuantisedEncoding::NibbleLUTSourceScale) &&
                      expertTierPreservesNativeSingleScale(
                          cpu::native_quantised::CPUNativeQuantisedEncoding::NibbleLUT, 0));

        /**
         * @brief Locate one payload byte using the actual live wire-column stride.
         * @param local_column Live column within this compact unit.
         * @param value_index Native payload byte within that column.
         * @param columns Actual live columns; retained64-column capacity is never the wire stride.
         * @return Byte offset in this unit's compact encoded payload.
         */
        __device__ __forceinline__ std::size_t interleavedValueOffset(
            int local_column,
            int value_index, unsigned columns)
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

        /** @brief Transpose a compensation-free CPU unit without decoding payloads or staging scale words. */
        template <bool ToCpu>
        __device__ __forceinline__ void nativeSingleScaleUnitsCUDAUnit(
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
                const auto wire_byte = offset + interleavedValueOffset(column, byte, columns);
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
                    __trap();
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

        /** @brief Launch one native wire unit per CUDA block. */
        template <bool ToCpu>
        __global__ void nativeSingleScaleUnitsCUDA(
            ExpertTierGpuConstProjectionView source, ExpertTierGpuMutableProjectionView destination,
            const uint8_t *wire_in, uint8_t *wire_out, ExpertTierWeightDeviceLayout layout, uint32_t first_unit)
        {
            nativeSingleScaleUnitsCUDAUnit<ToCpu>(source, destination, wire_in, wire_out, layout, first_unit, blockIdx.x);
        }

        /** @brief Update only the live code bits at a possibly unaligned pooled plane address. */
        __device__ __forceinline__ void storeResidentCode(void *plane, size_t bit, unsigned width, uint32_t code)
        {
            const auto address = reinterpret_cast<uintptr_t>(plane);
            native_quantised_superblock::storePackedBits(reinterpret_cast<void *>(address & ~uintptr_t{3}),
                bit + (address & 3u) * 8, width, code);
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
        __device__ __forceinline__ void compactMultiScaleUnitsCUDAUnit(
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
                    storeResidentCode(destination.scales, address.linear * code_bits + bit_shift, code_bits, code);
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
                const bool coded = layout.cpuStorage().coded();
                if ((!description.device_staging && !coded) || !description.scales ||
                    ((coded || layout.gpu_is_asymmetric) && !description.mins) ||
                    (!coded && layout.gpu_has_emins && !description.emins) ||
                    layout.N <= 0 || layout.K <= 0 || layout.K % 32 != 0 ||
                    layout.blocks_per_row != layout.K / 32 ||
                    !layout.cpu_block_stride || description.first_unit>=layout.unit_count ||
                    (to_cpu ? layout.direction != ExpertTierWeightConversionDirection::GpuToCpu
                            : layout.direction != ExpertTierWeightConversionDirection::CpuToGpu))
                    return false;
                const auto first_bytes=expertTierWirePrefix(layout,description.first_unit);
                const auto complete_bytes=expertTierWirePrefix(layout,layout.unit_count);
                if(command.bytes>complete_bytes-first_bytes)return false;
                const auto final_unit=expertTierWireUnitAt(layout,first_bytes+command.bytes);
                if(expertTierWirePrefix(layout,final_unit)!=first_bytes+command.bytes)return false;
                const auto units=final_unit-description.first_unit;
                if (!units || description.first_unit >= layout.unit_count ||
                    units > layout.unit_count - description.first_unit ||
                    units > layout.maximum_units_per_chunk)
                    return false;
                // This is dispatch totality, not a second footprint authority.
                // Never acknowledge bytes for a codebook with no device branch.
                if (coded)
                    return nativeQuantisedSuperblockCodable(layout.gpu_codebook_id) && layout.gpu_superblock_phase < 8 &&
                        layout.gpu_code_bit_shift < 8;
                // Scalar layout admission authenticated the exact native provenance and
                // planes. Q6_K and IQ4_XS units are coded above; nothing else has a branch.
                return expertTierPreservesNativeSingleScale(layout.cpu_encoding, layout.cpu_codebook_id);
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
                const bool to_cpu = description.kind == MappedTransferWorkKind::PackedGpuToCpu;
                auto *chunk = reinterpret_cast<std::uint8_t *>(description.device_staging ? description.device_staging :
                    (to_cpu ? command.destination_address : command.source_address));
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
                if (layout.cpuStorage().coded())
                {
                    if (to_cpu)
                        compactMultiScaleUnitsCUDAUnit<true>(source, {}, nullptr, chunk,
                            layout, description.first_unit, unit, active);
                    else
                        compactMultiScaleUnitsCUDAUnit<false>({}, destination, chunk, nullptr,
                            layout, description.first_unit, unit, active);
                    return;
                }
                // valid() admitted only coded or native single-scale units, and
                // neither converter has an internal CTA barrier.
                if (to_cpu)
                    nativeSingleScaleUnitsCUDAUnit<true>(source, {}, nullptr, chunk, layout, description.first_unit, unit, active);
                else
                    nativeSingleScaleUnitsCUDAUnit<false>({}, destination, chunk, nullptr, layout, description.first_unit, unit, active);
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
                const auto& layout=description.layout;
                const auto chunk_origin=expertTierWirePrefix(layout,description.first_unit);
                const auto first=expertTierWireUnitAt(layout,chunk_origin+position);
                const auto final=expertTierWireUnitAt(layout,chunk_origin+command.bytes);
                const auto geometry = layout.cpuStorage().wireGeometry(expertTierWireColumns(layout, first));
                const auto stride = geometry.block_stride + geometry.header_bytes;
                const auto batch_units=quantum/stride?quantum/stride:1u;
                const auto full_units=static_cast<unsigned>(layout.N/64)*layout.blocks_per_row;
                const auto phase_end=first<full_units && full_units<final?full_units:final;
                const auto available_units=phase_end-first;
                const auto units=available_units<batch_units?available_units:batch_units;
                const auto count = expertTierWireOffset(layout, first, units);
                auto *staging = reinterpret_cast<unsigned char *>(description.device_staging);
                if (staging && description.kind == MappedTransferWorkKind::PackedCpuToGpu)
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
                    const auto unit = first-description.first_unit+(active ? offset : 0u);
                    convertUnit(command, static_cast<std::uint32_t>(unit), active);
                    __syncthreads();
                }
                if (staging && description.kind == MappedTransferWorkKind::PackedGpuToCpu)
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
                   cpu_chunk_capacity >= layout.chunkBytes(unit_count,first_unit);
        }
        /** @brief Copy one logical bit span without replacing bits owned by neighboring matrices. */
        __global__ void copyCodedExpertBitsCUDA(const uint8_t *source, void *destination, size_t bits,
            unsigned source_shift, unsigned destination_shift)
        {
            namespace sb = native_quantised_superblock;
            const size_t bit = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
            if (bit >= bits)
                return;
            const auto width = static_cast<unsigned>(bits - bit < 8 ? bits - bit : 8);
            const auto value = sb::loadPackedBits(source, bit + source_shift, width);
            storeResidentCode(destination, bit + destination_shift, width, value);
        }
        /** @brief Copy one live block; only the first live block of each source superblock writes its header. */
        __global__ void copyCodedPeerRectangleCUDA(CUDACodedPeerProjectionRectangle r)
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
                const auto dst_bit = to.linear * bits + nativeQuantisedLayoutCodeBitShift(r.destination.metadata_layout);
                storeResidentCode(r.destination.scales, dst_bit, bits, code);
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
    bool launchCodedExpertBitCopyCUDA(const void *source, void *destination, size_t bits,
        unsigned source_shift, unsigned destination_shift, void *stream) noexcept
    {
        if (!stream || !source || !destination || !bits || source == destination)
            return false;
        constexpr unsigned threads = 128;
        const auto blocks = (bits - 1) / (threads * 8) + 1;
        if (blocks > std::numeric_limits<int>::max())
            return false;
        copyCodedExpertBitsCUDA<<<static_cast<unsigned>(blocks), threads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t *>(source), destination, bits, source_shift, destination_shift);
        return cudaPeekAtLastError() == cudaSuccess;
    }

    } // namespace detail

    bool launchCUDACodedPeerProjectionRectangle(CUDACodedPeerProjectionRectangle rectangle, void *stream) noexcept
    {
        if (!stream || !rectangle.rows || !rectangle.blocks || !rectangle.payload_bytes)
            return false;
        constexpr size_t threads = 128;
        const auto grid = (size_t(rectangle.rows) * rectangle.blocks - 1) / threads + 1;
        if (grid > std::numeric_limits<int>::max())
            return false;
        copyCodedPeerRectangleCUDA<<<unsigned(grid), threads, 0, static_cast<cudaStream_t>(stream)>>>(rectangle);
        return cudaPeekAtLastError() == cudaSuccess;
    }

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
        if (layout.cpuStorage().coded())
        {
            auto coded_layout = layout;
            coded_layout.gpu_code_bit_shift = nativeQuantisedLayoutCodeBitShift(source.metadata_layout);
            compactMultiScaleUnitsCUDA<true><<<grid, block, 0, cuda_stream>>>(
                source, {}, nullptr, device_cpu_chunk, coded_layout, first_unit);
            return cudaPeekAtLastError() == cudaSuccess;
        }
        if (expertTierPreservesNativeSingleScale(layout.cpu_encoding, layout.cpu_codebook_id))
        {
            nativeSingleScaleUnitsCUDA<true><<<grid, block, 0, cuda_stream>>>(
                source, {}, nullptr, device_cpu_chunk, layout, first_unit);
            return cudaPeekAtLastError() == cudaSuccess;
        }
        // Q6_K and IQ4_XS units are superblock-coded and NibbleLUT units are
        // native single-scale, both handled above; any other encoding fails closed.
        return false;
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
        if (layout.cpuStorage().coded())
        {
            auto coded_layout = layout;
            coded_layout.gpu_code_bit_shift = nativeQuantisedLayoutCodeBitShift(destination.metadata_layout);
            compactMultiScaleUnitsCUDA<false><<<grid, block, 0, cuda_stream>>>(
                {}, destination, device_cpu_chunk, nullptr, coded_layout, first_unit);
            return cudaPeekAtLastError() == cudaSuccess;
        }
        if (expertTierPreservesNativeSingleScale(layout.cpu_encoding, layout.cpu_codebook_id))
        {
            nativeSingleScaleUnitsCUDA<false><<<grid, block, 0, cuda_stream>>>(
                {}, destination, device_cpu_chunk, nullptr, layout, first_unit);
            return cudaPeekAtLastError() == cudaSuccess;
        }
        // Q6_K and IQ4_XS units are superblock-coded and NibbleLUT units are
        // native single-scale, both handled above; any other encoding fails closed.
        return false;
    }
} // namespace llaminar2
