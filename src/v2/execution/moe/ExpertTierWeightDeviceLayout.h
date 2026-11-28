/**
 * @file ExpertTierWeightDeviceLayout.h
 * @brief Allocation-free device contract for streamed expert-weight repacking.
 *
 * The structures in this file deliberately contain only scalar geometry and
 * IQ3_S expanded source grids retain original factors and promote exactly to
 * physical11; inverse-grid membership is checked before publication.
 * already-owned pointers. CUDA and HIP conversion launchers consume them
 * directly, allowing isolated correctness, timing, and compiler-resource
 * tests without coupling a kernel translation unit to model orchestration.
 */

#pragma once

#include "kernels/cpu/gemm/CPUNativeQuantisedEncoding.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "MoEExpertProjectionOwnership.h"
#include <algorithm>
#include <stdexcept>
#include <span>
#include <string>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace llaminar2
{
    /**
     * @brief Checked overlap of two admitted projections in original GEMM coordinates.
     *
     * This is geometry, not residency authority. The caller retains the original
     * source/destination epoch leases. Local origins are derived from the same
     * original rectangles so equal physical shapes cannot hide different cuts.
     */
    class ExpertTierWeightProjectionIntersection final
    {
    public:
        using Projection = MoEExpertProjectionOwnership::Projection;
        /**
         * @brief Intersect original projections without allocating or rebuilding ownership.
         * @param source Original source member projection.
         * @param destination Original destination member projection.
         * @return Exact overlap; an empty overlap submits no lane work.
         * @throws std::invalid_argument For foreign original dimensions or out-of-bounds local rectangles.
         */
        static ExpertTierWeightProjectionIntersection between(Projection source, Projection destination)
        {
            const auto valid=[](const Projection& p) {
                const bool known=p.residence==MoEProjectionResidence::ExpertOwner ||
                    p.residence==MoEProjectionResidence::ParticipantOutputSlice ||
                    p.residence==MoEProjectionResidence::ParticipantInputSlice;
                return known && p.source_rows>0 && p.source_columns>0 && p.first_row>=0 && p.rows>0 &&
                    p.first_column>=0 && p.columns>0 && p.first_row<=p.source_rows &&
                    p.rows<=p.source_rows-p.first_row && p.first_column<=p.source_columns &&
                    p.columns<=p.source_columns-p.first_column;
            };
            if(!valid(source) || !valid(destination) || source.source_rows!=destination.source_rows ||
               source.source_columns!=destination.source_columns)
                throw std::invalid_argument("Expert transfer requires compatible original projection rectangles");
            return ExpertTierWeightProjectionIntersection(source,destination);
        }
        /** @return Original source projection, with its independent local strides. */
        const Projection& source() const noexcept { return source_; }
        /** @return Original destination projection, with its independent local strides. */
        const Projection& destination() const noexcept { return destination_; }
        /** @return Original overlap row origin. */
        int firstRow() const noexcept { return std::max(source_.first_row,destination_.first_row); }
        /** @return Original overlap K origin. */
        int firstColumn() const noexcept { return std::max(source_.first_column,destination_.first_column); }
        /** @return Actual overlap rows, excluding both allocations' padding. */
        int rows() const noexcept { return std::max(0,std::min(source_.first_row+source_.rows,destination_.first_row+destination_.rows)-firstRow()); }
        /** @return Actual overlap K columns, excluding nonintersecting native blocks. */
        int columns() const noexcept { return std::max(0,std::min(source_.first_column+source_.columns,destination_.first_column+destination_.columns)-firstColumn()); }
        /** @return Whether this pair requires no transfer. */
        bool empty() const noexcept { return rows()==0 || columns()==0; }
        /** @return Overlap origin relative to the original source local allocation. */
        int sourceRowOrigin() const noexcept { return firstRow()-source_.first_row; }
        /** @return Overlap K origin relative to the original source local allocation. */
        int sourceColumnOrigin() const noexcept { return firstColumn()-source_.first_column; }
        /** @return Overlap origin relative to the original destination local allocation. */
        int destinationRowOrigin() const noexcept { return firstRow()-destination_.first_row; }
        /** @return Overlap K origin relative to the original destination local allocation. */
        int destinationColumnOrigin() const noexcept { return firstColumn()-destination_.first_column; }
        /** @brief Compare complete original projection and overlap identity. */
        bool operator==(const ExpertTierWeightProjectionIntersection&) const = default;
    private:
        /**
         * @brief Retain validated original projections; all offsets remain derived.
         * @param source Original source member rectangle.
         * @param destination Original destination member rectangle.
         */
        ExpertTierWeightProjectionIntersection(Projection source,Projection destination)
            :source_(source),destination_(destination) {}
        Projection source_,destination_;
    };

    // Tuple matching is exact only when each source identity names one catalog entry.
    static_assert([] {
        constexpr auto &formats = native_quantised_formats::kAllSourceFormats;
        for (std::size_t i = 0; i < formats.size(); ++i)
            for (std::size_t j = i + 1; j < formats.size(); ++j)
                if (formats[i].metadata->codebook_id == formats[j].metadata->codebook_id &&
                    formats[i].metadata->is_superblock == formats[j].metadata->is_superblock)
                    return false;
        return true;
    }(), "Expert source identities must be unique");

    /**
     * @brief Reject original K origins the resident coded CPU header cannot represent.
     * @param source Authenticated original format, not the device decoder ID.
     * @param original_k_origin First retained K value in original source coordinates.
     * @throws std::invalid_argument If a coded root would straddle two source superblocks.
     */
    inline void requireExpertTierResidentCpuOrigin(const NativeQuantisedFormatInfo &source, int original_k_origin)
    {
        constexpr auto values_per_superblock = 32 * native_quantised_superblock::kBlocksPerSuperblock;
        if (!cpu::native_quantised::migrationUsesSuperblockCodes(source) || original_k_origin % values_per_superblock == 0)
            return;
        std::string label = std::to_string(source.codebook_id);
        for (const auto &entry : native_quantised_formats::kAllSourceFormats)
            if (entry.metadata->codebook_id == source.codebook_id && entry.metadata->is_superblock == source.is_superblock)
                label = entry.quant_type;
        throw std::invalid_argument("Coded CPU expert " + label + " K origin " + std::to_string(original_k_origin) +
            " is not aligned to the " + std::to_string(values_per_superblock) + "-value source superblock");
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_TIER_WEIGHT_HD __host__ __device__
#else
#define LLAMINAR_TIER_WEIGHT_HD
#endif
    /** @return Whether the resident CPU unit retains a native single-scale payload without compensation. */
    [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD constexpr bool expertTierPreservesNativeSingleScale(
        cpu::native_quantised::CPUNativeQuantisedEncoding encoding, int codebook) noexcept
    {
        using Encoding = cpu::native_quantised::CPUNativeQuantisedEncoding;
        return encoding == Encoding::NibbleLUT || encoding == Encoding::Q5NativeSingleScale ||
            (encoding == Encoding::SignedInt8SingleScale && cpu::native_quantised::isSignedInt8SingleScaleSource(codebook));
    }

    /** @return One CPU Q5 byte holding the fifth bits for two four-lane groups. */
    [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD inline uint8_t expertTierQ5FifthFromGpu(const uint8_t *bits, unsigned group) noexcept
    {
        const auto shift = (group % 2) * 4;
        return uint8_t(((bits[group / 2] >> shift) & 15) | (((bits[2 + group / 2] >> shift) & 15) << 4));
    }

    /** @return One original GPU Q5 fifth-bit byte reconstructed from its two CPU byte lanes. */
    [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD inline uint8_t expertTierQ5FifthFromCpu(const uint8_t *paired, unsigned byte) noexcept
    {
        const auto shift = (byte / 2) * 4;
        return uint8_t(((paired[(byte % 2) * 2] >> shift) & 15) | (((paired[(byte % 2) * 2 + 1] >> shift) & 15) << 4));
    }

    /** @brief Live code planes in their native low/high order, independently rounded per plane. */
    struct ExpertTierCodedPlanes
    {
        unsigned low_bits = 0, high_bits = 0;
        /** @return Exact bytes of one plane for the specified live columns. */
        [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD static constexpr size_t bytes(unsigned columns, unsigned bits) noexcept
        { return (size_t(columns) * bits + 7) / 8; }
        /** @return Bytes of both live code planes, excluding GPU allocation guard bytes. */
        [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD constexpr size_t bytes(unsigned columns) const noexcept
        { return bytes(columns, low_bits) + bytes(columns, high_bits); }
    };

    /** @return Native code-plane widths for a physical GPU codebook. */
    [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD constexpr ExpertTierCodedPlanes expertTierCodedPlanes(uint8_t codebook) noexcept
    {
        const auto bits = native_quantised_superblock::superblockCodePlaneBits(codebook);
        return bits == 12 ? ExpertTierCodedPlanes{8, 4} : bits == 6 ? ExpertTierCodedPlanes{4, 2}
             : ExpertTierCodedPlanes{bits, 0};
    }

    /** @return Byte offset of one column's payload byte in a resident or compact coded unit. */
    [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD constexpr size_t expertTierCodedPayloadOffset(
        unsigned payload_bytes, unsigned columns, unsigned column, unsigned byte) noexcept
    {
        const auto tail = payload_bytes % 4;
        const auto whole = payload_bytes - tail;
        return byte < whole ? size_t(byte / 4) * columns * 4 + column * 4 + byte % 4
            : size_t(whole) * columns + column * tail + byte - whole;
    }

    /** @return A column's complete code from the live split CPU planes (0 when in payload). */
    [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD inline uint32_t expertTierLoadCodedColumn(
        const uint8_t *plane, ExpertTierCodedPlanes planes, unsigned columns, unsigned column) noexcept
    {
        using native_quantised_superblock::loadPackedBits;
        const auto low = planes.low_bits ? loadPackedBits(plane, size_t(column) * planes.low_bits, planes.low_bits) : 0;
        return planes.high_bits ? low | (loadPackedBits(plane + planes.bytes(columns, planes.low_bits),
            size_t(column) * planes.high_bits, planes.high_bits) << planes.low_bits) : low;
    }

    /** @brief Root and wire geometry with exact coded per-plane extents. */
    [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD constexpr cpu::native_quantised::CPUNativeQuantisedBlockGeometry
    expertTierBlockGeometry(unsigned blocks, unsigned columns, unsigned phase, unsigned data_stride,
        unsigned block_stride, bool coded, uint8_t codebook) noexcept
    {
        return coded ? cpu::native_quantised::CPUNativeQuantisedBlockGeometry{
            .block_stride = size_t(data_stride / 64) * columns + expertTierCodedPlanes(codebook).bytes(columns),
            .header_bytes = size_t(native_quantised_superblock::superblockFieldBytes(codebook)) * columns,
            .blocks_per_row = blocks, .phase = phase}
            : cpu::native_quantised::CPUNativeQuantisedBlockGeometry{
                .block_stride = size_t(block_stride / 64) * columns, .blocks_per_row = blocks};
    }

    /** @return Exact encoded byte prefix; a unit closing its wire superblock includes its header. */
    template<class Layout>
    [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD constexpr size_t expertTierWirePrefix(const Layout &layout, unsigned units) noexcept
    {
        const auto cpu = layout.cpuStorage();
        const auto blocks = static_cast<unsigned>(cpu.blocks_per_row);
        const auto full = cpu.wire_columns_per_unit;
        const auto complete = static_cast<unsigned>(cpu.N) / full;
        const auto full_units = complete * blocks;
        const auto geometry = cpu.wireGeometry(full);
        if (units <= full_units)
            return geometry.blockOffset(units / blocks, units % blocks);
        const auto tail = cpu.wireGeometry(static_cast<unsigned>(cpu.N) % full);
        const auto tail_units = units - full_units;
        return geometry.matrixBytes(complete) + tail.blockOffset(tail_units / blocks, tail_units % blocks);
    }

    /**
     * @brief Native CPU root and bounded encoded wire geometry, with no GPU metadata.
     *
     * This describes an original local CPU allocation plus its intersection.
     * It grants no model/epoch authority. Native source leases and the existing
     * manifest remain the caller's authority. Small send lanes choose fewer
     * live columns per wire unit; final resident CPU tiles remain 64 columns.
     */
    struct ExpertTierWeightCPUStorageLayout
    {
        int32_t N=0,K=0,blocks_per_row=0;
        int32_t cpu_N=0,cpu_K=0,cpu_first_row=0,cpu_first_column=0;
        uint8_t cpu_codebook_id=0,cpu_is_asymmetric=0,cpu_is_superblock=0;
        cpu::native_quantised::CPUNativeQuantisedEncoding cpu_encoding=cpu::native_quantised::CPUNativeQuantisedEncoding::SignedInt8SingleScale;
        uint32_t cpu_data_stride=0,cpu_block_stride=0,unit_count=0,maximum_units_per_chunk=0;
        unsigned wire_columns_per_unit=64;
        /** @return The CPU-only endpoint of this layout. */
        [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD constexpr const ExpertTierWeightCPUStorageLayout &cpuStorage() const noexcept
        { return *this; }
        /** @return Whether this root contains codes and superblock words. */
        [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD constexpr bool coded() const noexcept
        { return cpu::native_quantised::preparedSuperblockCoded(cpu_encoding); }
        /** @return The physical codec ID (IQ4_XS uses the aliased physical ID24). */
        [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD constexpr uint8_t codedCodebook() const noexcept
        { return cpu_codebook_id == 4 && cpu_is_superblock ? kNativeQuantisedIQ4XSSourceScaleCodebook : cpu_codebook_id; }
        /** @return Geometry of a full CPU root, whose first source block is superblock-aligned. */
        [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD constexpr auto rootGeometry() const noexcept
        {
            return cpu::native_quantised::CPUNativeQuantisedBlockGeometry{
                .block_stride = cpu_block_stride,
                .header_bytes = coded() ? 64u * native_quantised_superblock::superblockFieldBytes(codedCodebook()) : 0u,
                .blocks_per_row = static_cast<size_t>(cpu_K / 32)};
        }
        /** @return Geometry of one wire row with exactly @p columns live columns and the intersection's K phase. */
        [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD constexpr auto wireGeometry(unsigned columns) const noexcept
        {
            const auto phase = native_quantised_superblock::superblockPhase(cpu_first_column / 32);
            return expertTierBlockGeometry(blocks_per_row, columns, phase, cpu_data_stride, cpu_block_stride, coded(), codedCodebook());
        }
        /** @return Largest live unit; one closing block includes one complete header. */
        [[nodiscard]] constexpr size_t maximumWireUnitBytes() const noexcept
        {
            const auto geometry = wireGeometry(N < int(wire_columns_per_unit) ? N : wire_columns_per_unit);
            return geometry.block_stride + geometry.header_bytes;
        }
        /** @return Whether original local bounds, native physical traits and wire counts agree. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            const auto *source = native_quantised_formats::forSourceIdentity(cpu_codebook_id, cpu_is_superblock != 0);
            if (!source || ((coded() || cpu::native_quantised::migrationUsesSuperblockCodes(*source)) &&
                    cpu_encoding != cpu::native_quantised::preparedEncodingForSource(*source)))
                return false;
            if(N<=0 || K<=0 || K%32 || blocks_per_row!=K/32 || cpu_N<=0 ||
               cpu_N>std::numeric_limits<int32_t>::max()-63 || cpu_K<=0 || cpu_K%32 ||
               cpu_first_row<0 || cpu_first_column<0 || cpu_first_column%32 ||
               cpu_first_row>cpu_N || N>cpu_N-cpu_first_row || cpu_first_column>cpu_K || K>cpu_K-cpu_first_column ||
               wire_columns_per_unit==0 || wire_columns_per_unit>64 || cpu_is_asymmetric>1 || cpu_is_superblock>1 ||
               !cpu::native_quantised::isKnownPreparedEncoding(cpu_encoding) ||
                cpu_data_stride!=cpu::native_quantised::preparedDataStride(cpu_encoding,cpu_codebook_id) ||
               cpu_block_stride!=cpu::native_quantised::preparedInterleavedBlockStride(cpu_encoding,cpu_is_asymmetric!=0,cpu_codebook_id))return false;
            const uint64_t expected=((uint64_t(N)+wire_columns_per_unit-1)/wire_columns_per_unit)*blocks_per_row;
            const auto row_bytes = rootGeometry().rowBytes();
            return row_bytes && ((uint64_t(cpu_N)+63)/64)<=std::numeric_limits<size_t>::max()/row_bytes && expected<=std::numeric_limits<uint32_t>::max() && unit_count==expected &&
                maximum_units_per_chunk>0 && maximum_units_per_chunk<=unit_count;
        }
        /**
         * @brief Quote the largest actual chunk in the admitted ordered schedule.
         *
         * Coded closing units include a superblock header, unlike the first
         * unit. Inspect complete ranges, including the compact final N tile,
         * rather than treating the first chunk as a fixed byte stride.
         * @return Exact staging capacity required by this schedule, or zero for invalid counts.
         */
        [[nodiscard]] constexpr size_t maximumChunkBytes() const noexcept
        {
            if (!maximum_units_per_chunk || maximum_units_per_chunk > unit_count)
                return 0;
            size_t maximum = 0;
            for (uint32_t first = 0; first < unit_count;)
            {
                const auto count = std::min(maximum_units_per_chunk, unit_count - first);
                maximum = std::max(maximum, chunkBytes(count, first));
                first += count;
            }
            return maximum;
        }
        /** @return Complete final CPU allocation bytes, independent of communicated extent. */
        constexpr size_t cpuStorageBytes() const noexcept
        { return rootGeometry().matrixBytes((cpu_N + 63) / 64); }
        /**
         * @brief Quote exact live encoded bytes for one ordered wire chunk.
         * @param units Number of complete live-column/K-block units.
         * @param first_unit First unit in the original intersection order.
         * @return Compact encoded bytes, excluding inactive columns.
         */
        constexpr size_t chunkBytes(uint32_t units,uint32_t first_unit=0) const noexcept
        { return units ? expertTierWirePrefix(*this, first_unit + units) - expertTierWirePrefix(*this, first_unit) : 0; }
    };

    /** Direction in which one streamed CPU-format chunk is converted. */
    enum class ExpertTierWeightConversionDirection : std::uint8_t
    {
        GpuToCpu = 1, ///< Source GPU decodes/reformats bytes before D2H transfer.
        CpuToGpu = 2, ///< Destination GPU reformats arriving CPU bytes after H2D.
    };

    /**
     * @brief Complete immutable geometry needed by a repack kernel launch.
     *
     * `unit_count` indexes64-column CPU tiles, but each tail tile communicates
     * only its actual logical columns. Final CPU roots retain their native tile
     * padding independently; GPU arrays retain their full local logical extent.
     */
    struct ExpertTierWeightDeviceLayout
    {
        // Direction determines whether GPU metadata is a source or destination.
        ExpertTierWeightConversionDirection direction =
            ExpertTierWeightConversionDirection::CpuToGpu;
        // Logical and padded matrix geometry used to map a unit to (N, K block).
        std::int32_t N = 0;
        std::int32_t K = 0;
        std::int32_t N_padded = 0;
        std::int32_t blocks_per_row = 0;
        // Both endpoint allocations have independent local matrix strides.
        std::int32_t cpu_N = 0, cpu_K = 0, cpu_first_row = 0, cpu_first_column = 0;
        std::int32_t gpu_N = 0, gpu_K = 0, gpu_first_row = 0, gpu_first_column = 0;
        // Source provenance disambiguates shared execution codebooks.
        std::uint8_t cpu_codebook_id = 0;
        std::uint8_t gpu_source_codebook_id = 0;
        std::uint8_t gpu_source_is_superblock = 0;
        // Endpoint format metadata selects one compile-time backend kernel.
        std::uint8_t gpu_codebook_id = 0;
        std::uint8_t gpu_payload_bytes_per_block = 0;
        std::uint8_t cpu_is_asymmetric = 0;
        std::uint8_t cpu_is_superblock = 0;
        std::uint8_t gpu_is_asymmetric = 0;
        std::uint8_t gpu_has_emins = 0;
        std::uint8_t gpu_superblock_phase = 0;
        std::uint8_t gpu_code_bit_shift = 0; ///< Live descriptor bit shift for captured transfer work.
        cpu::native_quantised::CPUNativeQuantisedEncoding cpu_encoding =
            cpu::native_quantised::CPUNativeQuantisedEncoding::SignedInt8SingleScale;
        // CPU-unit and persistent staging capacity; no device allocation occurs.
        std::uint32_t cpu_data_stride = 0;
        std::uint32_t cpu_block_stride = 0;
        std::uint32_t unit_count = 0;
        std::uint32_t maximum_units_per_chunk = 0;

        /** @return Original CPU-only physical storage projection of this GPU-edge contract. */
        [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD constexpr ExpertTierWeightCPUStorageLayout cpuStorage() const noexcept
        {
            return {.N=N,.K=K,.blocks_per_row=blocks_per_row,.cpu_N=cpu_N,.cpu_K=cpu_K,
                .cpu_first_row=cpu_first_row,.cpu_first_column=cpu_first_column,
                .cpu_codebook_id=cpu_codebook_id,.cpu_is_asymmetric=cpu_is_asymmetric,.cpu_is_superblock=cpu_is_superblock,
                .cpu_encoding=cpu_encoding,.cpu_data_stride=cpu_data_stride,.cpu_block_stride=cpu_block_stride,
                .unit_count=unit_count,.maximum_units_per_chunk=maximum_units_per_chunk,.wire_columns_per_unit=64};
        }

        /**
         * @brief Check the device tuple against compile-time catalog entries without device host-table reads.
         * @return Whether original provenance and every physical GPU plane match one catalog source.
         */
        [[nodiscard]] LLAMINAR_TIER_WEIGHT_HD constexpr bool gpuFormatMatchesSource() const noexcept
        {
            return [this]<std::size_t... Index>(std::index_sequence<Index...>) {
                return (([this] {
                    constexpr auto source = *native_quantised_formats::kAllSourceFormats[Index].metadata;
                    constexpr auto physical = canonicalDeviceVnniFormat(source);
                    return gpu_source_codebook_id == source.codebook_id &&
                        gpu_source_is_superblock == source.is_superblock &&
                        gpu_codebook_id == physical.codebook_id &&
                        gpu_payload_bytes_per_block == physical.payload_bytes_per_block &&
                        gpu_is_asymmetric == physical.is_asymmetric && gpu_has_emins == physical.has_emins;
                }()) || ...);
            }(std::make_index_sequence<native_quantised_formats::kAllSourceFormats.size()>{});
        }

        /**
         * @brief Validate all scalar geometry and format relationships.
         * @return Whether the layout can safely drive a device kernel.
         */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            using Encoding = cpu::native_quantised::CPUNativeQuantisedEncoding;
            const bool gpu_to_cpu =
                direction == ExpertTierWeightConversionDirection::GpuToCpu;
            const bool cpu_to_gpu =
                direction == ExpertTierWeightConversionDirection::CpuToGpu;
            // CPU kernels consume whole 32-K blocks and 64-column N units.
            if ((!gpu_to_cpu && !cpu_to_gpu) ||
                N <= 0 || N>std::numeric_limits<int32_t>::max()-63 ||
                cpu_N>std::numeric_limits<int32_t>::max()-63 || K <= 0 || (K % 32) != 0 ||
                blocks_per_row != K / 32 || N_padded < N ||
                (N_padded % 64) != 0 || N_padded != ((N + 63) / 64) * 64 ||
                !cpu::native_quantised::isKnownPreparedEncoding(cpu_encoding) ||
                cpu_is_asymmetric>1 || cpu_is_superblock>1 || gpu_source_is_superblock>1 ||
                gpu_is_asymmetric>1 || gpu_has_emins>1 ||
                cpu_N<=0 || cpu_K<=0 || gpu_N<=0 || gpu_K<=0 || cpu_K%32 || gpu_K%32 ||
                cpu_first_row<0 || gpu_first_row<0 || cpu_first_column<0 || gpu_first_column<0 ||
                cpu_first_column%32 || gpu_first_column%32 || cpu_first_row>cpu_N || gpu_first_row>gpu_N ||
                cpu_first_column>cpu_K || gpu_first_column>gpu_K || N>cpu_N-cpu_first_row ||
                N>gpu_N-gpu_first_row || K>cpu_K-cpu_first_column || K>gpu_K-gpu_first_column)
            {
                return false;
            }
            // Redundant wire strides must agree with the canonical encoding.
            if (cpu_data_stride !=
                    cpu::native_quantised::preparedDataStride(cpu_encoding, cpu_codebook_id) ||
                cpu_block_stride !=
                    cpu::native_quantised::preparedInterleavedBlockStride(
                        cpu_encoding, cpu_is_asymmetric != 0, cpu_codebook_id))
            {
                return false;
            }
            // Tile indices cover retained CPU storage; compact tail bytes omit inactive N.
            const std::uint64_t expected_units =
                static_cast<std::uint64_t>(N_padded / 64) *
                static_cast<std::uint64_t>(blocks_per_row);
            if (expected_units == 0 ||
                expected_units > std::numeric_limits<std::uint32_t>::max() ||
                unit_count != expected_units || maximum_units_per_chunk == 0 ||
                maximum_units_per_chunk > unit_count)
            {
                return false;
            }

            const auto *source = native_quantised_formats::forSourceIdentity(
                gpu_source_codebook_id, gpu_source_is_superblock != 0);
            if (!source || cpu_codebook_id != gpu_source_codebook_id ||
                cpu_is_superblock != gpu_source_is_superblock || cpu_is_asymmetric != source->is_asymmetric ||
                cpu_encoding != cpu::native_quantised::preparedEncodingForSource(*source))
                return false;

            const auto cpu = cpuStorage();
            if (!cpu.valid() || !gpuFormatMatchesSource())
                return false;
            if (!cpu.coded())
                return gpu_superblock_phase == 0 && gpu_code_bit_shift == 0;
            return gpu_superblock_phase < native_quantised_superblock::kBlocksPerSuperblock && gpu_code_bit_shift < 8 &&
                native_quantised_superblock::superblockPhase(gpu_superblock_phase + gpu_first_column / 32) ==
                    cpu.wireGeometry(64).phase;
        }

        /**
         * @brief Compute exact communicated bytes for ordered live native columns.
         * @param units Number of consecutive native-block tiles.
         * @param first_unit Intersection-relative first tile; determines compact tail width.
         * @return Required bytes for the chunk.
         */
        [[nodiscard]] constexpr std::size_t chunkBytes(
            std::uint32_t units, std::uint32_t first_unit = 0) const noexcept
        {
            return cpuStorage().chunkBytes(units, first_unit);
        }
        /** @return Exact final CPU allocation extent, independent of transfer bytes. */
        [[nodiscard]] constexpr std::size_t cpuStorageBytes() const noexcept
        {
            return cpuStorage().cpuStorageBytes();
        }
    };

    /** @return GPU wire units retain the native 64-column launch grouping. */
    LLAMINAR_TIER_WEIGHT_HD inline unsigned expertTierWireColumnQuantum(const ExpertTierWeightDeviceLayout&) noexcept {return 64;}
    /** @return Actual CPU-only wire quantum selected within original send-lane capacity. */
    inline unsigned expertTierWireColumnQuantum(const ExpertTierWeightCPUStorageLayout& layout) noexcept {return layout.wire_columns_per_unit;}
    /** @brief Actual native columns in one intersection unit, excluding resident padding. */
    template<class Layout>
    LLAMINAR_TIER_WEIGHT_HD inline unsigned expertTierWireColumns(const Layout& layout,unsigned unit) noexcept
    {
        const unsigned quantum=expertTierWireColumnQuantum(layout);
        const unsigned first=(unit/static_cast<unsigned>(layout.blocks_per_row))*quantum;
        return static_cast<unsigned>(layout.N)-first<quantum?static_cast<unsigned>(layout.N)-first:quantum;
    }
    /** @brief Staging byte origin of a chunk-relative unit, including compact tail planes. */
    template<class Layout>
    LLAMINAR_TIER_WEIGHT_HD inline size_t expertTierWireOffset(const Layout& layout,unsigned first,unsigned local) noexcept
    {return expertTierWirePrefix(layout,first+local)-expertTierWirePrefix(layout,first);}
    /** @brief Inverse prefix for a validated exact unit boundary; never used for arbitrary byte offsets. */
    template<class Layout>
    LLAMINAR_TIER_WEIGHT_HD inline unsigned expertTierWireUnitAt(const Layout& layout,size_t bytes) noexcept
    {
        const auto cpu = layout.cpuStorage();
        if (!cpu.coded())
        {
            const unsigned regular = (cpu.N / cpu.wire_columns_per_unit) * cpu.blocks_per_row;
            const auto stride = cpu.wireGeometry(cpu.wire_columns_per_unit).block_stride;
            const size_t prefix = size_t(regular) * stride;
            if (bytes <= prefix)
                return static_cast<unsigned>(bytes / stride);
            return regular + static_cast<unsigned>((bytes - prefix) /
                cpu.wireGeometry(cpu.N % cpu.wire_columns_per_unit).block_stride);
        }
        unsigned low = 0, high = layout.unit_count;
        while (low < high)
        {
            const unsigned mid = low + (high - low) / 2;
            if (expertTierWirePrefix(layout, mid) < bytes)
                low = mid + 1;
            else
                high = mid;
        }
        return low;
    }
#undef LLAMINAR_TIER_WEIGHT_HD

    /**
     * @brief Move only live native column planes between an admitted CPU root and staging.
     * @param layout Exact endpoint/intersection and native encoding contract.
     * @param cpu_root Complete original local CPU allocation (including resident padding).
     * @param first_unit First intersection-relative unit in the ordered chunk.
     * @param units Nonempty bounded chunk population.
     * @param compact_chunk Exact live encoded bytes; never capacity padding.
     * @return False before writes for invalid geometry, bounds or aliasing.
     */
    bool gatherExpertTierCPUWeightChunk(const ExpertTierWeightDeviceLayout& layout,
        std::span<const std::uint8_t> cpu_root,std::uint32_t first_unit,std::uint32_t units,
        std::span<std::uint8_t> compact_chunk) noexcept;
    /** @brief Inverse native plane copy; only original intersecting CPU columns change. */
    bool scatterExpertTierCPUWeightChunk(const ExpertTierWeightDeviceLayout& layout,
        std::span<const std::uint8_t> compact_chunk,std::uint32_t first_unit,std::uint32_t units,
        std::span<std::uint8_t> cpu_root) noexcept;

    /** @brief Gather the same encoded planes for a CPU-native-only original root. */
    bool gatherExpertTierCPUWeightChunk(const ExpertTierWeightCPUStorageLayout& layout,
        std::span<const std::uint8_t> cpu_root,std::uint32_t first_unit,std::uint32_t units,
        std::span<std::uint8_t> compact_chunk) noexcept;
    /** @brief Scatter the same encoded planes without a GPU format or endpoint. */
    bool scatterExpertTierCPUWeightChunk(const ExpertTierWeightCPUStorageLayout& layout,
        std::span<const std::uint8_t> compact_chunk,std::uint32_t first_unit,std::uint32_t units,
        std::span<std::uint8_t> cpu_root) noexcept;

    /** Read-only common CUDA/ROCm separated projection view. */
    struct ExpertTierGpuConstProjectionView
    {
        // Pointers name already-owned separated device allocations.
        const std::uint8_t *payload = nullptr;
        const std::uint16_t *scales = nullptr;
        const std::uint16_t *mins = nullptr;
        const std::uint32_t *emins = nullptr;
        // Capacities are exact so stale/undersized descriptors fail before launch.
        std::size_t payload_bytes = 0;
        std::size_t scales_bytes = 0;
        std::size_t mins_bytes = 0;
        std::size_t emins_bytes = 0;
        uint8_t metadata_layout = 0;

        /**
         * @brief Validate read-only regions against a scalar layout.
         * @param layout Expected geometry, formats, and region sizes.
         * @return Whether pointers and capacities exactly satisfy `layout`.
         */
        [[nodiscard]] constexpr bool validFor(
            const ExpertTierWeightDeviceLayout &layout) const noexcept
        {
            // Validate scalar arithmetic before deriving any byte count.
            if (!layout.valid() || payload == nullptr || scales == nullptr)
                return false;
            // GPU arrays omit padded N and are block-major over logical rows.
            if (layout.cpuStorage().coded())
            {
                using namespace native_quantised_superblock;
                if (!nativeQuantisedLayoutValid(layout.gpu_codebook_id, metadata_layout) ||
                    nativeQuantisedLayoutRepresentation(metadata_layout) != NativeQuantisedMetadataRepresentation::SuperblockCoded ||
                    nativeQuantisedLayoutPhase(metadata_layout) != layout.gpu_superblock_phase)
                    return false;
                const auto blocks = static_cast<unsigned>(layout.gpu_K / 32);
                const auto bits = size_t(layout.gpu_N) * blocks * superblockCodePlaneBits(layout.gpu_codebook_id);
                const auto bytes = planeBytes(layout.gpu_codebook_id, layout.gpu_N, blocks, layout.gpu_superblock_phase);
                const auto shift = nativeQuantisedLayoutCodeBitShift(metadata_layout);
                return payload_bytes == size_t(layout.gpu_N) * blocks * layout.gpu_payload_bytes_per_block &&
                    scales_bytes == (bits ? (bits + shift + 7) / 8 : 0) && (bits || (!shift && scales == reinterpret_cast<const uint16_t *>(payload))) &&
                    mins && mins_bytes == bytes.superblock && !emins && !emins_bytes;
            }
            if (metadata_layout)
                return false;
            const std::size_t blocks =
                static_cast<std::size_t>(layout.gpu_N) *
                static_cast<std::size_t>(layout.gpu_K/32);
            return payload_bytes ==
                       blocks * layout.gpu_payload_bytes_per_block &&
                   scales_bytes == blocks * nativeQuantisedScalePlaneBytes(layout.gpu_codebook_id) &&
                   mins_bytes ==
                       (layout.gpu_is_asymmetric != 0
                            ? blocks * kNativeQuantisedSecondaryPlaneBytes
                            : 0u) &&
                   emins_bytes ==
                       (layout.gpu_has_emins != 0
                            ? blocks * sizeof(std::uint32_t)
                            : 0u) &&
                   (layout.gpu_is_asymmetric == 0 || mins != nullptr) &&
                   (layout.gpu_has_emins == 0 || emins != nullptr);
        }
    };

    /** Writable common CUDA/ROCm separated projection view. */
    struct ExpertTierGpuMutableProjectionView
    {
        // Mutable counterpart used only by CPU-to-GPU arrival conversion.
        std::uint8_t *payload = nullptr;
        std::uint16_t *scales = nullptr;
        std::uint16_t *mins = nullptr;
        std::uint32_t *emins = nullptr;
        // Capacities must describe the complete logical GPU projection.
        std::size_t payload_bytes = 0;
        std::size_t scales_bytes = 0;
        std::size_t mins_bytes = 0;
        std::size_t emins_bytes = 0;
        uint8_t metadata_layout = 0;

        /**
         * @brief Convert mutable ownership to a read-only launch view.
         * @return A view with identical pointer identity and capacity.
         */
        [[nodiscard]] constexpr ExpertTierGpuConstProjectionView asConst()
            const noexcept
        {
            // Pointer values are preserved; only write permission is removed.
            return ExpertTierGpuConstProjectionView{
                .payload = payload,
                .scales = scales,
                .mins = mins,
                .emins = emins,
                .payload_bytes = payload_bytes,
                .scales_bytes = scales_bytes,
                .mins_bytes = mins_bytes,
                .emins_bytes = emins_bytes,
                .metadata_layout = metadata_layout,
            };
        }

        /**
         * @brief Validate writable regions against a scalar layout.
         * @param layout Expected geometry, formats, and region sizes.
         * @return Whether pointers and capacities exactly satisfy `layout`.
         */
        [[nodiscard]] constexpr bool validFor(
            const ExpertTierWeightDeviceLayout &layout) const noexcept
        {
            return asConst().validFor(layout);
        }
    };
} // namespace llaminar2
