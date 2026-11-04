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

#include "kernels/cpu/gemm/CPUNativeVNNIPreparedFootprint.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "MoEExpertProjectionOwnership.h"
#include <algorithm>
#include <stdexcept>
#include <span>

#include <cstddef>
#include <cstdint>
#include <limits>

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

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_TIER_WEIGHT_HD __host__ __device__
#else
#define LLAMINAR_TIER_WEIGHT_HD
#endif
    /**
     * @brief Common exact encoded-byte prefix; allocation padding is never traffic.
     * @param rows Logical intersection N rows.
     * @param blocks Intersection K blocks.
     * @param stride Canonical resident 64-column unit bytes.
     * @param columns_per_unit Positive live-column wire quantum, at most 64.
     * @param count Complete wire units in this prefix.
     * @return Exact encoded payload/metadata bytes for the logical prefix.
     */
    LLAMINAR_TIER_WEIGHT_HD constexpr size_t expertTierEncodedPrefix(
        int rows,int blocks,uint32_t stride,unsigned columns_per_unit,uint64_t count) noexcept
    {
        const uint64_t full=uint64_t(rows/columns_per_unit)*blocks;
        const uint64_t regular=count<full?count:full;
        return size_t(regular)*(stride/64)*columns_per_unit+
            size_t(count-regular)*(stride/64)*(rows%columns_per_unit);
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
        cpu::native_vnni::CPUNativeVNNIEncoding cpu_encoding=cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8;
        uint32_t cpu_data_stride=0,cpu_block_stride=0,unit_count=0,maximum_units_per_chunk=0;
        unsigned wire_columns_per_unit=64;
        /** @return Whether original local bounds, native physical traits and wire counts agree. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            if(N<=0 || K<=0 || K%32 || blocks_per_row!=K/32 || cpu_N<=0 ||
               cpu_N>std::numeric_limits<int32_t>::max()-63 || cpu_K<=0 || cpu_K%32 ||
               cpu_first_row<0 || cpu_first_column<0 || cpu_first_column%32 ||
               cpu_first_row>cpu_N || N>cpu_N-cpu_first_row || cpu_first_column>cpu_K || K>cpu_K-cpu_first_column ||
               wire_columns_per_unit==0 || wire_columns_per_unit>64 || cpu_is_asymmetric>1 || cpu_is_superblock>1 ||
               !native_vnni_formats::forSourceIdentity(cpu_codebook_id,cpu_is_superblock!=0) ||
               !cpu::native_vnni::isKnownPreparedEncoding(cpu_encoding) ||
               cpu_data_stride!=cpu::native_vnni::preparedDataStride(cpu_encoding) ||
               cpu_block_stride!=cpu::native_vnni::preparedInterleavedBlockStride(cpu_encoding,cpu_is_asymmetric!=0))return false;
            const uint64_t expected=((uint64_t(N)+wire_columns_per_unit-1)/wire_columns_per_unit)*blocks_per_row;
            const uint64_t root_units=((uint64_t(cpu_N)+63)/64)*(cpu_K/32);
            return root_units<=std::numeric_limits<size_t>::max()/cpu_block_stride && expected<=std::numeric_limits<uint32_t>::max() && unit_count==expected &&
                maximum_units_per_chunk>0 && maximum_units_per_chunk<=unit_count;
        }
        /** @return Complete final CPU allocation bytes, independent of communicated extent. */
        constexpr size_t cpuStorageBytes() const noexcept
        {return size_t((cpu_N+63)/64)*(cpu_K/32)*cpu_block_stride;}
        /**
         * @brief Quote exact live encoded bytes for one ordered wire chunk.
         * @param units Number of complete live-column/K-block units.
         * @param first_unit First unit in the original intersection order.
         * @return Compact encoded bytes, excluding inactive columns.
         */
        constexpr size_t chunkBytes(uint32_t units,uint32_t first_unit=0) const noexcept
        {return expertTierEncodedPrefix(N,blocks_per_row,cpu_block_stride,wire_columns_per_unit,uint64_t(first_unit)+units)-
                expertTierEncodedPrefix(N,blocks_per_row,cpu_block_stride,wire_columns_per_unit,first_unit);}
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
        cpu::native_vnni::CPUNativeVNNIEncoding cpu_encoding =
            cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8;
        // CPU-unit and persistent staging capacity; no device allocation occurs.
        std::uint32_t cpu_data_stride = 0;
        std::uint32_t cpu_block_stride = 0;
        std::uint32_t unit_count = 0;
        std::uint32_t maximum_units_per_chunk = 0;

        /** @return Original CPU-only physical storage projection of this GPU-edge contract. */
        [[nodiscard]] constexpr ExpertTierWeightCPUStorageLayout cpuStorage() const noexcept
        {
            return {.N=N,.K=K,.blocks_per_row=blocks_per_row,.cpu_N=cpu_N,.cpu_K=cpu_K,
                .cpu_first_row=cpu_first_row,.cpu_first_column=cpu_first_column,
                .cpu_codebook_id=cpu_codebook_id,.cpu_is_asymmetric=cpu_is_asymmetric,.cpu_is_superblock=cpu_is_superblock,
                .cpu_encoding=cpu_encoding,.cpu_data_stride=cpu_data_stride,.cpu_block_stride=cpu_block_stride,
                .unit_count=unit_count,.maximum_units_per_chunk=maximum_units_per_chunk,.wire_columns_per_unit=64};
        }

        /**
         * @brief Validate all scalar geometry and format relationships.
         * @return Whether the layout can safely drive a device kernel.
         */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            using Encoding = cpu::native_vnni::CPUNativeVNNIEncoding;
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
                !cpu::native_vnni::isKnownPreparedEncoding(cpu_encoding) ||
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
                    cpu::native_vnni::preparedDataStride(cpu_encoding) ||
                cpu_block_stride !=
                    cpu::native_vnni::preparedInterleavedBlockStride(
                        cpu_encoding, cpu_is_asymmetric != 0))
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

            // Both directions retain source provenance even when CPU→GPU
            // promotion normalizes an expanded source into codebook 19 or 23.
            const NativeVnniFormatInfo *source_format =
                native_vnni_formats::forSourceIdentity(
                    gpu_source_codebook_id,
                    gpu_source_is_superblock != 0);
            if (source_format == nullptr ||
                cpu_codebook_id != gpu_source_codebook_id ||
                cpu_is_superblock != gpu_source_is_superblock ||
                ((cpu_codebook_id == 4 && cpu_is_superblock != 0) &&
                 cpu_encoding == Encoding::NibbleLUT) ||
                ((cpu_codebook_id == 8 ||
                  hasCompactMultiScaleVnniPayload(cpu_codebook_id)) &&
                 cpu_encoding == Encoding::ExpandedInt8))
            {
                return false;
            }

            if (gpu_to_cpu)
            {
                // A demotion kernel must decode the actual accelerator source,
                // including Q2_K effective minima and dual-scale codebooks.
                // A previous CPU promotion may instead have normalized an
                // ExpandedInt8 source to physical codebook 19/23. That is a
                // first-class live representation, not a provenance change.
                const bool canonical_source =
                    gpu_codebook_id == canonicalDeviceVnniCodebookId(*source_format) &&
                    gpu_payload_bytes_per_block ==
                        source_format->payload_bytes &&
                    gpu_is_asymmetric ==
                        static_cast<std::uint8_t>(
                            canonicalDeviceVnniFormat(*source_format).is_asymmetric) &&
                    gpu_has_emins ==
                        static_cast<std::uint8_t>(source_format->has_emins);
                const bool normalized_source =
                    cpu_encoding == Encoding::ExpandedInt8 &&
                    gpu_payload_bytes_per_block == 32 &&
                    gpu_has_emins == 0 &&
                    gpu_is_asymmetric == cpu_is_asymmetric &&
                    gpu_codebook_id ==
                        (cpu_is_asymmetric != 0
                             ? kNativeVnniExpandedInt8MinCodebook
                             : static_cast<std::uint8_t>(19));
                if ((!canonical_source && !normalized_source) ||
                    cpu_is_asymmetric !=
                        static_cast<std::uint8_t>(
                            source_format->is_asymmetric))
                {
                    return false;
                }

                const Encoding expected_encoding =
                    gpu_codebook_id == kNativeVnniIQ4XSSourceScaleCodebook
                        ? Encoding::NibbleLUTSourceScale
                        : gpu_codebook_id == 11 ? Encoding::IQ3SExpandedSourceScale
                        : gpu_codebook_id == 0 || gpu_codebook_id == 4 ||
                            gpu_codebook_id == 5
                        ? Encoding::NibbleLUT
                        : (gpu_codebook_id == 8
                               ? Encoding::Q6KNativeDualScale
                               : hasCompactMultiScaleVnniPayload(gpu_codebook_id)
                                     ? Encoding::CompactMultiScale
                                     : Encoding::ExpandedInt8);
                return cpu_encoding == expected_encoding;
            }

            // Promotion consumes final CPU bytes. Expanded CPU encodings have
            // deliberately lost their compact source representation and must
            // therefore target the normalized signed-INT8 codebooks.
            if (gpu_has_emins != 0 && cpu_encoding != Encoding::CompactMultiScale)
                return false;
            switch (cpu_encoding)
            {
            case Encoding::IQ3SExpandedSourceScale:
                return cpu_codebook_id == 11 && cpu_is_superblock == 1 &&
                       gpu_codebook_id == 11 && gpu_payload_bytes_per_block == 13 &&
                       cpu_is_asymmetric == 0 && gpu_is_asymmetric == 1;
            case Encoding::NibbleLUTSourceScale:
                return cpu_codebook_id == 4 && cpu_is_superblock != 0 &&
                       gpu_codebook_id == kNativeVnniIQ4XSSourceScaleCodebook &&
                       gpu_payload_bytes_per_block == 16 &&
                       cpu_is_asymmetric == 0 && gpu_is_asymmetric == 1;
            case Encoding::NibbleLUT:
                return gpu_payload_bytes_per_block == 16 &&
                       gpu_is_asymmetric == cpu_is_asymmetric &&
                       (((gpu_codebook_id == 0 || gpu_codebook_id == 4) &&
                         cpu_is_asymmetric == 0) ||
                        (gpu_codebook_id == 5 && cpu_is_asymmetric != 0));
            case Encoding::ExpandedInt8:
                return gpu_payload_bytes_per_block == 32 &&
                       gpu_codebook_id ==
                           (cpu_is_asymmetric != 0
                                ? kNativeVnniExpandedInt8MinCodebook
                                : static_cast<std::uint8_t>(19)) &&
                       gpu_is_asymmetric == cpu_is_asymmetric;
            case Encoding::Q6KNativeDualScale:
                return cpu_codebook_id == 8 &&
                       gpu_source_codebook_id == 8 &&
                       gpu_codebook_id == 8 &&
                       gpu_payload_bytes_per_block == 24 &&
                       cpu_is_asymmetric != 0 && gpu_is_asymmetric != 0;
            case Encoding::CompactMultiScale:
                return hasCompactMultiScaleVnniPayload(cpu_codebook_id) &&
                       gpu_codebook_id == cpu_codebook_id &&
                       gpu_payload_bytes_per_block == source_format->payload_bytes &&
                       cpu_is_asymmetric != 0 && gpu_is_asymmetric != 0 &&
                       gpu_has_emins == static_cast<uint8_t>(source_format->has_emins);
            }
            return false;
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
            return expertTierEncodedPrefix(N,blocks_per_row,cpu_block_stride,64,uint64_t(first_unit)+units)-
                expertTierEncodedPrefix(N,blocks_per_row,cpu_block_stride,64,first_unit);
        }
        /** @return Exact final CPU allocation extent, independent of transfer bytes. */
        [[nodiscard]] constexpr std::size_t cpuStorageBytes() const noexcept
        {
            return static_cast<std::size_t>((cpu_N+63)/64)*(cpu_K/32)*cpu_block_stride;
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
    /** @brief Byte prefix of exact live column planes in the canonical ordered stream. */
    template<class Layout>
    LLAMINAR_TIER_WEIGHT_HD inline size_t expertTierWirePrefix(const Layout& layout,unsigned units) noexcept
    {
        return expertTierEncodedPrefix(layout.N,layout.blocks_per_row,layout.cpu_block_stride,expertTierWireColumnQuantum(layout),units);
    }
    /** @brief Staging byte origin of a chunk-relative unit, including compact tail planes. */
    template<class Layout>
    LLAMINAR_TIER_WEIGHT_HD inline size_t expertTierWireOffset(const Layout& layout,unsigned first,unsigned local) noexcept
    {return expertTierWirePrefix(layout,first+local)-expertTierWirePrefix(layout,first);}
    /** @brief Inverse prefix for a validated exact unit boundary; never used for arbitrary byte offsets. */
    template<class Layout>
    LLAMINAR_TIER_WEIGHT_HD inline unsigned expertTierWireUnitAt(const Layout& layout,size_t bytes) noexcept
    {
        const auto quantum=expertTierWireColumnQuantum(layout);
        const unsigned regular=static_cast<unsigned>(layout.N/quantum)*layout.blocks_per_row;
        const size_t stride=(layout.cpu_block_stride/64)*quantum,prefix=size_t(regular)*stride;
        return bytes<prefix?static_cast<unsigned>(bytes/stride):
            regular+static_cast<unsigned>((bytes-prefix)/((layout.cpu_block_stride/64)*(layout.N%quantum?layout.N%quantum:quantum)));
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
            const std::size_t blocks =
                static_cast<std::size_t>(layout.gpu_N) *
                static_cast<std::size_t>(layout.gpu_K/32);
            return payload_bytes ==
                       blocks * layout.gpu_payload_bytes_per_block &&
                   scales_bytes == blocks * sizeof(std::uint16_t) &&
                   mins_bytes ==
                       (layout.gpu_is_asymmetric != 0
                            ? blocks * sizeof(std::uint16_t)
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
