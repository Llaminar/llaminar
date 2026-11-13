/**
 * @file MoEOverlayRemoteProjectionProtocol.cpp
 * @brief Canonical wire encoding and adversarial validation for remote weights.
 */

#include "MoEOverlayRemoteProjectionProtocol.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace llaminar2
{
    namespace
    {
        constexpr std::uint64_t kFNVOffset = 14695981039346656037ull;
        constexpr std::uint64_t kFNVPrime = 1099511628211ull;

        /** @brief Store a rejection diagnostic only when requested. */
        bool reject(std::string *error, std::string message) noexcept
        {
            if (error)
                *error = std::move(message);
            return false;
        }

        /** @brief Mix one byte into the stable manifest digest. */
        void hashByte(std::uint64_t &hash, std::uint8_t value) noexcept
        {
            hash ^= value;
            hash *= kFNVPrime;
        }

        /** @brief Mix an integral or enum scalar in little-endian order. */
        template <typename Value>
        void hashScalar(std::uint64_t &hash, Value value) noexcept
        {
            using Raw = typename std::conditional_t<
                std::is_enum_v<Value>,
                std::underlying_type<Value>,
                std::type_identity<Value>>::type;
            using Unsigned = std::make_unsigned_t<Raw>;
            Unsigned bits = static_cast<Unsigned>(value);
            for (std::size_t byte = 0; byte < sizeof(Unsigned); ++byte)
            {
                hashByte(hash, static_cast<std::uint8_t>(bits & 0xffu));
                bits >>= 8u;
            }
        }

        /** @brief Hash every semantic field of one remote endpoint identity. */
        void hashIdentity(
            std::uint64_t &hash,
            const MoEOverlayRemoteProjectionIdentity &identity) noexcept
        {
            hashScalar(hash, identity.expected_epoch);
            hashScalar(hash, identity.candidate_epoch);
            hashScalar(hash, identity.execution_fingerprint.low);
            hashScalar(hash, identity.execution_fingerprint.high);
            hashScalar(hash, identity.migration_index);
            hashScalar(hash, identity.layer_idx);
            hashScalar(hash, identity.expert_id);
            hashScalar(hash, identity.projection);
            hashScalar(hash, identity.source_participant);
            hashScalar(hash, identity.destination_participant);
            hashScalar(hash, identity.source_world_rank);
            hashScalar(hash, identity.destination_world_rank);
            hashScalar(hash, identity.source_device.type);
            hashScalar(hash, identity.source_device.ordinal);
            hashScalar(hash, identity.destination_device.type);
            hashScalar(hash, identity.destination_device.ordinal);
        }

        /** @brief Return whether an endpoint is supported by the tier fabric. */
        bool supportedDevice(DeviceId device) noexcept
        {
            return device.is_cpu() || device.is_cuda() || device.is_rocm();
        }

        /** @brief Sum region sizes without accepting uint64 overflow. */
        bool sumRegions(
            const std::array<
                std::uint64_t,
                kMoEOverlayRemoteProjectionRegionCount> &regions,
            std::uint64_t &total) noexcept
        {
            total = 0;
            for (const std::uint64_t bytes : regions)
            {
                if (bytes > std::numeric_limits<std::uint64_t>::max() - total)
                    return false;
                total += bytes;
            }
            return true;
        }

        /** @brief Validate current GPU execution format against provenance. */
        bool validGpuExecutionFormat(
            const MoEOverlayRemoteProjectionManifest &manifest,
            const NativeQuantisedFormatInfo &source) noexcept
        {
            const std::uint8_t canonical =
                canonicalDeviceVnniCodebookId(source);
            if (manifest.gpu_codebook_id == canonical)
            {
                return manifest.gpu_payload_bytes_per_block ==
                           source.payload_bytes &&
                       manifest.gpu_is_asymmetric ==
                           static_cast<std::uint8_t>(canonicalDeviceVnniFormat(source).is_asymmetric) &&
                       manifest.gpu_has_emins ==
                           static_cast<std::uint8_t>(source.has_emins);
            }

            /*
             * A CPU promotion may no longer possess compact source bits.  Its
             * destination GPU then executes normalized signed bytes while the
             * source identity remains the mathematical GGUF provenance.
             */
            if (cpu::native_quantised::preparedEncodingForSource(source) !=
                cpu::native_quantised::CPUNativeQuantisedEncoding::ExpandedInt8)
            {
                return false;
            }
            const std::uint8_t normalized =
                source.is_asymmetric
                    ? kNativeQuantisedExpandedInt8MinCodebook
                    : static_cast<std::uint8_t>(19);
            return manifest.gpu_codebook_id == normalized &&
                   manifest.gpu_payload_bytes_per_block == 32u &&
                   manifest.gpu_is_asymmetric ==
                       static_cast<std::uint8_t>(source.is_asymmetric) &&
                   manifest.gpu_has_emins == 0u;
        }

        /** @brief Seal a constructed manifest and reject impossible capacities. */
        MoEOverlayRemoteProjectionManifest seal(
            MoEOverlayRemoteProjectionManifest manifest)
        {
            if (manifest.maximum_chunk_bytes == 0)
            {
                throw std::invalid_argument(
                    "Remote ExpertOverlay projection requires positive staging capacity");
            }
            if (!sumRegions(manifest.region_bytes, manifest.total_bytes) ||
                manifest.total_bytes == 0)
            {
                throw std::overflow_error(
                    "Remote ExpertOverlay projection byte total is invalid");
            }
            if(manifest.source_projection.source_rows==0) {
                manifest.source_projection={MoEProjectionResidence::ExpertOwner,manifest.N,manifest.K,0,manifest.N,0,manifest.K};
                manifest.destination_projection=manifest.source_projection;
            }
            manifest.manifest_hash = manifest.computedHash();
            std::string error;
            if (!manifest.valid(&error))
                throw std::invalid_argument(error);
            return manifest;
        }

        /** @brief Populate fields shared by every physical packing. */
        MoEOverlayRemoteProjectionManifest baseManifest(
            const MoEOverlayRemoteProjectionIdentity &identity,
            NativeQuantisedSourceIdentity source_identity,
            int N,
            int K,
            std::uint32_t maximum_chunk_bytes)
        {
            if (!identity.valid() || !source_identity.present || N <= 0 ||
                K <= 0 || (K % 32) != 0 || maximum_chunk_bytes == 0)
            {
                throw std::invalid_argument(
                    "Remote ExpertOverlay projection has invalid identity, provenance, geometry, or capacity");
            }
            const auto *source = native_quantised_formats::forSourceIdentity(
                source_identity.codebook_id,
                source_identity.is_superblock);
            if (!source)
            {
                throw std::invalid_argument(
                    "Remote ExpertOverlay projection provenance is not catalogued");
            }

            MoEOverlayRemoteProjectionManifest manifest;
            manifest.format_kind = ExpertWeightFormatKind::NativeQuantised;
            manifest.identity = identity;
            manifest.N = N;
            manifest.K = K;
            manifest.N_padded = ((N + 63) / 64) * 64;
            manifest.blocks_per_row = K / 32;
            manifest.source_codebook_id = source_identity.codebook_id;
            manifest.source_is_superblock =
                source_identity.is_superblock ? 1u : 0u;
            manifest.cpu_codebook_id = source_identity.codebook_id;
            manifest.cpu_is_asymmetric = source->is_asymmetric ? 1u : 0u;
            manifest.cpu_is_superblock =
                source_identity.is_superblock ? 1u : 0u;
            manifest.cpu_encoding =
                cpu::native_quantised::preparedEncodingForSource(*source);
            // Wire strides derive from the CPU endpoint's canonical layout
            // tuple (the protocol carries no prepared-layout field until its
            // versioned A3 update, so the endpoint's layout is canonical by
            // version statement); the section authority, not the encoding
            // family, owns the arithmetic.
            manifest.cpu_data_stride =
                cpu::native_quantised::preparedDataStride(
                    cpu::native_quantised::kHostPreparedLayout,
                    manifest.cpu_encoding, manifest.cpu_is_asymmetric != 0, *source);
            manifest.cpu_block_stride = static_cast<std::uint32_t>(
                cpu::native_quantised::preparedInterleavedBlockStride(
                    cpu::native_quantised::kHostPreparedLayout,
                    manifest.cpu_encoding,
                    manifest.cpu_is_asymmetric != 0, *source));
            manifest.maximum_chunk_bytes = maximum_chunk_bytes;
            return manifest;
        }

        /** @brief Write one scalar in canonical little-endian order. */
        template <typename Value>
        void writeLittleEndian(
            std::span<std::uint8_t> destination,
            std::size_t &offset,
            Value value) noexcept
        {
            using Raw = typename std::conditional_t<
                std::is_enum_v<Value>,
                std::underlying_type<Value>,
                std::type_identity<Value>>::type;
            using Unsigned = std::make_unsigned_t<Raw>;
            Unsigned bits = static_cast<Unsigned>(value);
            for (std::size_t byte = 0; byte < sizeof(Unsigned); ++byte)
            {
                destination[offset++] =
                    static_cast<std::uint8_t>(bits & 0xffu);
                bits >>= 8u;
            }
        }

        /** @brief Read one scalar from canonical little-endian bytes. */
        template <typename Value>
        Value readLittleEndian(
            std::span<const std::uint8_t> packet,
            std::size_t &offset) noexcept
        {
            using Raw = typename std::conditional_t<
                std::is_enum_v<Value>,
                std::underlying_type<Value>,
                std::type_identity<Value>>::type;
            using Unsigned = std::make_unsigned_t<Raw>;
            Unsigned bits = 0;
            for (std::size_t byte = 0; byte < sizeof(Unsigned); ++byte)
            {
                bits |= static_cast<Unsigned>(packet[offset++])
                        << (byte * 8u);
            }
            return static_cast<Value>(bits);
        }

        /** @brief Encode or decode one DeviceId without relying on enum ABI. */
        void writeDevice(
            std::span<std::uint8_t> destination,
            std::size_t &offset,
            DeviceId device) noexcept
        {
            writeLittleEndian<std::int32_t>(
                destination, offset,
                static_cast<std::int32_t>(device.type));
            writeLittleEndian<std::int32_t>(
                destination, offset, device.ordinal);
        }

        /** @brief Decode one DeviceId; manifest validation rejects unknown types. */
        DeviceId readDevice(
            std::span<const std::uint8_t> packet,
            std::size_t &offset) noexcept
        {
            const auto type = readLittleEndian<std::int32_t>(packet, offset);
            const auto ordinal = readLittleEndian<std::int32_t>(packet, offset);
            return DeviceId(static_cast<DeviceType>(type), ordinal);
        }

        /** @brief Return true only when every byte in a reserved range is zero. */
        template <std::size_t Size>
        bool allZero(const std::uint8_t (&bytes)[Size]) noexcept
        {
            return std::all_of(
                std::begin(bytes), std::end(bytes),
                [](std::uint8_t byte) { return byte == 0; });
        }
    } // namespace

    bool MoEOverlayRemoteProjectionIdentity::valid() const noexcept
    {
        return expected_epoch > 0 &&
               candidate_epoch == expected_epoch + 1 &&
               execution_fingerprint.valid() && layer_idx >= 0 &&
               expert_id >= 0 &&
               static_cast<std::uint8_t>(projection) <=
                   static_cast<std::uint8_t>(
                       ExpertTierWeightProjection::Down) &&
               allZero(reserved_identity) && source_participant >= 0 &&
               destination_participant >= 0 &&
               source_participant != destination_participant &&
               source_world_rank >= 0 && destination_world_rank >= 0 &&
               source_world_rank != destination_world_rank &&
               supportedDevice(source_device) &&
               supportedDevice(destination_device);
    }

    std::uint64_t MoEOverlayRemoteProjectionManifest::computedHash()
        const noexcept
    {
        std::uint64_t hash = kFNVOffset;
        hashScalar(hash, magic);
        hashScalar(hash, abi_version);
        hashScalar(hash, packing);
        hashScalar(hash, format_kind);
        hashIdentity(hash, identity);
        hashScalar(hash, N);
        hashScalar(hash, K);
        hashScalar(hash, N_padded);
        hashScalar(hash, blocks_per_row);
        hashScalar(hash, source_codebook_id);
        hashScalar(hash, source_is_superblock);
        hashScalar(hash, cpu_codebook_id);
        hashScalar(hash, gpu_codebook_id);
        hashScalar(hash, gpu_payload_bytes_per_block);
        hashScalar(hash, cpu_is_asymmetric);
        hashScalar(hash, cpu_is_superblock);
        hashScalar(hash, gpu_is_asymmetric);
        hashScalar(hash, gpu_has_emins);
        hashScalar(hash, cpu_encoding);
        hashScalar(hash, cpu_data_stride);
        hashScalar(hash, cpu_block_stride);
        hashScalar(hash, cpu_wire_columns_per_unit);
        for (const auto bytes : region_bytes)
            hashScalar(hash, bytes);
        hashScalar(hash, total_bytes);
        hashScalar(hash, maximum_chunk_bytes);
        for(const auto& owned:{source_projection,destination_projection}) {
            hashScalar(hash,static_cast<int32_t>(owned.residence));
            for(const auto field:{owned.source_rows,owned.source_columns,owned.first_row,owned.rows,owned.first_column,owned.columns})
                hashScalar(hash,field);
        }
        return hash == 0 ? 0x9e3779b97f4a7c15ull : hash;
    }

    MoEOverlayRemoteProjectionManifest MoEOverlayRemoteProjectionManifest::forFloatingIntersection(
        const ExpertTierWeightProjectionIntersection& intersection) const
    {
        if(!valid() || !carriesFloatingBytes() || intersection.empty() ||
           intersection.source().rows!=source_projection.rows || intersection.source().columns!=source_projection.columns)
            throw std::invalid_argument("Floating remote intersection differs from its original source allocation");
        auto result=*this;result.source_projection=intersection.source();result.destination_projection=intersection.destination();
        result.N=intersection.rows();result.K=intersection.columns();
        // Floating wire geometry is compact. The original endpoint projections
        // retain their independent allocation strides; no padded rows travel.
        result.N_padded=result.N;
        const ExpertWeightFormat format{.kind=format_kind};
        result.region_bytes[0]=uint64_t(result.N)*result.K*format.floatingElementBytes();
        return seal(result);
    }

    MoEOverlayRemoteProjectionManifest MoEOverlayRemoteProjectionManifest::forCPUIntersection(
        const ExpertTierWeightProjectionIntersection& intersection) const
    {
        if(!valid() || !carriesCpuBytes() || !identity.source_device.is_cpu() || !identity.destination_device.is_cpu() ||
           intersection.empty() || intersection.source().rows!=source_projection.rows ||
           intersection.source().columns!=source_projection.columns || intersection.firstColumn()%32 || intersection.columns()%32 ||
           intersection.sourceColumnOrigin()%32 || intersection.destinationColumnOrigin()%32)
            throw std::invalid_argument("Native CPU remote intersection differs from its original source or native K groups");
        auto result=*this; result.source_projection=intersection.source();result.destination_projection=intersection.destination();
        result.N=intersection.rows();result.K=intersection.columns();result.N_padded=(result.N+63)/64*64;
        result.blocks_per_row=result.K/32;
        result.region_bytes[0]=uint64_t(result.N)*result.blocks_per_row*(result.cpu_block_stride/64);
        return seal(result);
    }

    bool MoEOverlayRemoteProjectionManifest::matchesGpuDestinationProjection(
        const ExpertTierWeightProjectionIntersection &original,
        ExpertWeightFormat destination_format,
        int destination_rows, int destination_columns) const noexcept
    {
        if (!valid() || !identity.destination_device.is_gpu() || !destination_format.valid() ||
            source_projection != original.source() || destination_projection != original.destination() ||
            destination_projection.rows != destination_rows ||
            destination_projection.columns != destination_columns)
            return false;
        if (destination_format.isFloating())
            return carriesFloatingBytes() && format_kind == destination_format.kind;
        return (carriesGpuBytes() || carriesCpuBytes()) &&
            source_codebook_id == destination_format.native_quantised.codebook_id &&
            (source_is_superblock != 0) == destination_format.native_quantised.is_superblock;
    }

    bool MoEOverlayRemoteProjectionManifest::valid(
        std::string *error) const noexcept
    {
        if (magic != kMagic || abi_version != kABIVersion ||
            !identity.valid() ||
            !allZero(reserved_format) || reserved_size != 0)
        {
            return reject(
                error,
                "Remote ExpertOverlay projection has invalid ABI, identity, or reserved fields");
        }
        if (N <= 0 || K <= 0 || maximum_chunk_bytes == 0)
        {
            return reject(
                error,
                "Remote ExpertOverlay projection geometry or chunk capacity is invalid");
        }
        try {
            const auto overlap=ExpertTierWeightProjectionIntersection::between(source_projection,destination_projection);
            if(overlap.empty() || overlap.rows()!=N || overlap.columns()!=K)
                return reject(error,"Remote projection original ownership intersection disagrees with payload geometry");
        } catch(const std::invalid_argument&) {
            return reject(error,"Remote projection original ownership rectangles are invalid");
        }
        std::uint64_t summed = 0;
        if (!sumRegions(region_bytes, summed) || summed == 0 ||
            summed != total_bytes)
        {
            return reject(
                error,
                "Remote ExpertOverlay projection region byte totals are invalid");
        }

        if (carriesFloatingBytes())
        {
            const ExpertWeightFormat format{.kind = format_kind};
            const auto element_bytes = format.floatingElementBytes();
            const auto n = static_cast<std::uint64_t>(N);
            const auto k = static_cast<std::uint64_t>(K);
            const bool byte_count_valid = element_bytes != 0u &&
                n <= std::numeric_limits<std::uint64_t>::max() / k &&
                n * k <= std::numeric_limits<std::uint64_t>::max() /
                             element_bytes;
            if (!format.isFloating() || N_padded != N ||
                blocks_per_row != 0 || source_codebook_id != 0 ||
                source_is_superblock != 0 || cpu_codebook_id != 0 ||
                gpu_codebook_id != 0 ||
                gpu_payload_bytes_per_block != 0 ||
                cpu_is_asymmetric != 0 || cpu_is_superblock != 0 ||
                gpu_is_asymmetric != 0 || gpu_has_emins != 0 ||
                cpu_data_stride != 0 || cpu_block_stride != 0 || cpu_wire_columns_per_unit != 0 ||
                !byte_count_valid ||
                region_bytes != std::array<std::uint64_t, 4>{
                    n * k * element_bytes, 0u, 0u, 0u})
            {
                return reject(
                    error,
                    "Contiguous floating remote projection disagrees with its precision, geometry, or byte extent");
            }
        }
        else
        {
            if (format_kind != ExpertWeightFormatKind::NativeQuantised ||
                (K % 32) != 0 ||
                N_padded != ((N + 63) / 64) * 64 ||
                blocks_per_row != K / 32)
            {
                return reject(
                    error,
                    "Remote ExpertOverlay NativeQuantised projection geometry or format kind is invalid");
            }
            const auto *source = native_quantised_formats::forSourceIdentity(
                source_codebook_id, source_is_superblock != 0);
            if (!source || cpu_codebook_id != source_codebook_id ||
                cpu_is_asymmetric !=
                    static_cast<std::uint8_t>(source->is_asymmetric) ||
                cpu_is_superblock != source_is_superblock ||
                cpu_encoding !=
                    cpu::native_quantised::preparedEncodingForSource(*source) ||
                cpu_data_stride !=
                    cpu::native_quantised::preparedDataStride(cpu_encoding) ||
                cpu_block_stride !=
                    cpu::native_quantised::preparedInterleavedBlockStride(
                        cpu_encoding, cpu_is_asymmetric != 0))
            {
                return reject(
                    error,
                    "Remote ExpertOverlay projection provenance and CPU format disagree");
            }

            if (carriesCpuBytes())
            {
            if (!identity.source_device.is_cpu() &&
                !identity.destination_device.is_cpu())
            {
                return reject(
                    error,
                    "CPU-format remote projection does not cross a CPU endpoint");
            }
            const auto overlap=ExpertTierWeightProjectionIntersection::between(source_projection,destination_projection);
            if(source_projection.rows>std::numeric_limits<int32_t>::max()-63 ||
               destination_projection.rows>std::numeric_limits<int32_t>::max()-63 ||
               source_projection.columns%32 || destination_projection.columns%32 ||
               overlap.sourceColumnOrigin()%32 || overlap.destinationColumnOrigin()%32)
                return reject(error,"Native CPU root projections must retain original32-element K groups");
            if(cpu_wire_columns_per_unit==0 || cpu_wire_columns_per_unit>64 ||
               ((uint64_t(N)+cpu_wire_columns_per_unit-1)/cpu_wire_columns_per_unit)*blocks_per_row>UINT32_MAX)
                return reject(error,"Native CPU compact wire unit population exceeds its bounded ABI");
            // The checked32-bit unit population and canonical per-column stride
            // make this product representable before any endpoint is admitted.
            const std::uint64_t expected_bytes = static_cast<uint64_t>(N)*blocks_per_row*(cpu_block_stride/64);
            if (region_bytes[0] != expected_bytes || region_bytes[1] || region_bytes[2] || region_bytes[3] ||
                maximum_chunk_bytes < (cpu_block_stride/64)*std::min<uint32_t>(cpu_wire_columns_per_unit,N))
                return reject(error,"CPU-format remote projection requires exact live native planes and bounded column groups");
            if (identity.source_device.is_gpu() ||
                identity.destination_device.is_gpu())
            {
                if (!validGpuExecutionFormat(*this, *source))
                {
                    return reject(
                        error,
                        "CPU-format remote projection has an invalid GPU endpoint format");
                }
                /*
                 * Device repack kernels consume indivisible 64-column by
                 * 32-K NativeQuantised units.  Reject a malformed wire capacity at
                 * manifest authentication rather than accepting it and
                 * discovering the partial-unit boundary after MPI has begun
                 * mutating destination state.
                 */
                if (cpu_wire_columns_per_unit != 64 || cpu_block_stride == 0 ||
                    maximum_chunk_bytes < cpu_block_stride ||
                    (maximum_chunk_bytes % cpu_block_stride) != 0)
                {
                    return reject(
                        error,
                        "CPU/GPU remote projection chunks must contain complete NativeQuantised units");
                }
            }
            else if (gpu_codebook_id != 0 ||
                     gpu_payload_bytes_per_block != 0 ||
                     gpu_is_asymmetric != 0 || gpu_has_emins != 0)
            {
                return reject(
                    error,
                    "CPU-to-CPU remote projection unexpectedly names a GPU format");
            }
            }
            else if (carriesGpuBytes())
            {
            if (cpu_wire_columns_per_unit != 0 || !identity.source_device.is_gpu() ||
                !identity.destination_device.is_gpu() ||
                !validGpuExecutionFormat(*this, *source))
            {
                return reject(
                    error,
                    "GPU blob remote projection requires two compatible GPU endpoints");
            }
            const std::uint64_t blocks =
                static_cast<std::uint64_t>(N) *
                static_cast<std::uint64_t>(blocks_per_row);
            const std::array<std::uint64_t, 4> expected{
                blocks * gpu_payload_bytes_per_block,
                blocks * sizeof(std::uint16_t),
                gpu_is_asymmetric ? blocks * sizeof(std::uint16_t) : 0u,
                gpu_has_emins ? blocks * sizeof(std::uint32_t) : 0u,
            };
            if (region_bytes != expected)
            {
                return reject(
                    error,
                    "GPU blob remote projection separated-region sizes disagree with its format");
            }
            }
            else
            {
                return reject(
                    error,
                    "Remote ExpertOverlay projection packing is unknown");
            }
        }

        if (manifest_hash == 0 || manifest_hash != computedHash())
        {
            return reject(
                error,
                "Remote ExpertOverlay projection manifest authentication failed");
        }
        if (error)
            error->clear();
        return true;
    }

    namespace {
        /** @brief Validate actual final CPU root metadata against its own canonical physical geometry. */
        void requireCanonicalCpuRoot(const cpu::native_quantised::CPUNativeQuantisedPackedWeights &root,
            const MoEOverlayRemoteProjectionManifest &format)
        {
            if (root.N_padded != format.N_padded || root.blocks_per_row != format.blocks_per_row ||
                root.encoding != format.cpu_encoding || root.data_stride != static_cast<int>(format.cpu_data_stride) ||
                root.interleaved_block_stride != format.cpu_block_stride || root.native_interleaved.size() !=
                    uint64_t(format.N_padded / 64) * format.blocks_per_row * format.cpu_block_stride)
                throw std::invalid_argument("Remote CPU root does not match its genuine canonical final storage");
        }
        /** @brief Seal CPU-only wire geometry without claiming any source payload owner. */
        MoEOverlayRemoteProjectionManifest sealCpuWire(MoEOverlayRemoteProjectionManifest manifest,
            uint32_t maximum_chunk_bytes)
        {
            manifest.packing = MoEOverlayRemoteProjectionPacking::CpuNativeQuantisedInterleaved;
            if (maximum_chunk_bytes < manifest.cpu_block_stride / 64)
                throw std::invalid_argument("Native CPU capacity cannot contain one exact encoded column");
            manifest.cpu_wire_columns_per_unit = std::min<uint32_t>(64, maximum_chunk_bytes / (manifest.cpu_block_stride / 64));
            manifest.region_bytes[0] = uint64_t(manifest.N) * manifest.blocks_per_row * (manifest.cpu_block_stride / 64);
            return seal(std::move(manifest));
        }
    }

    MoEOverlayRemoteProjectionManifest
    makeMoEOverlayRemoteCpuProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const cpu::native_quantised::CPUNativeQuantisedPackedWeights &source,
        std::uint32_t maximum_chunk_bytes)
    {
        const NativeQuantisedSourceIdentity source_identity{
            .codebook_id = source.codebook_id,
            .is_superblock = source.is_superblock,
            .present = true,
        };
        auto manifest = baseManifest(
            identity,
            source_identity,
            source.N,
            source.K,
            maximum_chunk_bytes);
        if (!identity.source_device.is_cpu() || !identity.destination_device.is_cpu())
            throw std::invalid_argument("Native CPU wire requires its actual original CPU endpoints");
        requireCanonicalCpuRoot(source, manifest);
        return sealCpuWire(std::move(manifest), maximum_chunk_bytes);
    }

    MoEOverlayRemoteProjectionManifest makeMoEOverlayRemoteCpuDestinationProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const cpu::native_quantised::CPUNativeQuantisedPackedWeights &destination,
        const ExpertTierWeightProjectionIntersection &intersection, uint32_t capacity)
    {
        if (!identity.source_device.is_cpu() || !identity.destination_device.is_cpu() || intersection.empty() ||
            destination.N != intersection.destination().rows || destination.K != intersection.destination().columns)
            throw std::invalid_argument("Native receiver expectation changed its genuine original destination");
        const NativeQuantisedSourceIdentity provenance{.codebook_id = destination.codebook_id,
            .is_superblock = destination.is_superblock, .present = true};
        const auto actual = baseManifest(identity, provenance, destination.N, destination.K, capacity);
        requireCanonicalCpuRoot(destination, actual);
        auto expected = baseManifest(identity, provenance, intersection.source().rows,
            intersection.source().columns, capacity);
        return sealCpuWire(std::move(expected), capacity).forCPUIntersection(intersection);
    }

    MoEOverlayRemoteProjectionManifest
    makeMoEOverlayRemoteCpuProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const ExpertTierWeightStreamManifest &stream,
        std::uint32_t maximum_chunk_bytes)
    {
        std::string stream_error;
        if (!stream.valid(&stream_error))
            throw std::invalid_argument(stream_error);
        const NativeQuantisedSourceIdentity source_identity{
            .codebook_id = stream.gpu_source_codebook_id,
            .is_superblock = stream.gpu_source_is_superblock != 0,
            .present = true,
        };
        const std::uint64_t aligned_chunk_bytes =
            static_cast<std::uint64_t>(stream.cpu_block_stride) *
            stream.maximum_units_per_chunk;
        if (aligned_chunk_bytes == 0 ||
            aligned_chunk_bytes > maximum_chunk_bytes ||
            aligned_chunk_bytes >
                std::numeric_limits<std::uint32_t>::max())
        {
            throw std::invalid_argument(
                "Remote CPU/GPU projection network capacity cannot contain the stream's complete-unit chunk");
        }
        auto manifest = baseManifest(
            identity,
            source_identity,
            stream.N,
            stream.K,
            static_cast<std::uint32_t>(aligned_chunk_bytes));
        const bool identity_is_gpu_to_cpu =
            identity.source_device.is_gpu() &&
            identity.destination_device.is_cpu();
        const bool identity_is_cpu_to_gpu =
            identity.source_device.is_cpu() &&
            identity.destination_device.is_gpu();
        const bool stream_is_gpu_to_cpu =
            stream.source_packing ==
            ExpertTierWeightPacking::GpuSeparatedNativeQuantised;
        if ((!identity_is_gpu_to_cpu && !identity_is_cpu_to_gpu) ||
            identity_is_gpu_to_cpu != stream_is_gpu_to_cpu ||
            stream.residency_epoch !=
                (stream_is_gpu_to_cpu ? identity.expected_epoch
                                      : identity.candidate_epoch) ||
            stream.layer_idx != identity.layer_idx ||
            stream.expert_id != identity.expert_id ||
            stream.projection != identity.projection)
        {
            throw std::invalid_argument(
                "Remote CPU/GPU projection stream disagrees with transaction identity or direction");
        }

        manifest.packing = MoEOverlayRemoteProjectionPacking::
            CpuNativeQuantisedInterleaved;
        manifest.N_padded = stream.N_padded;
        manifest.blocks_per_row = stream.blocks_per_row;
        manifest.cpu_codebook_id = stream.cpu_codebook_id;
        manifest.gpu_codebook_id = stream.gpu_codebook_id;
        manifest.gpu_payload_bytes_per_block =
            stream.gpu_payload_bytes_per_block;
        manifest.cpu_is_asymmetric = stream.cpu_is_asymmetric;
        manifest.cpu_is_superblock = stream.cpu_is_superblock;
        manifest.gpu_is_asymmetric = stream.gpu_is_asymmetric;
        manifest.gpu_has_emins = stream.gpu_has_emins;
        manifest.cpu_encoding = stream.cpu_encoding;
        manifest.cpu_data_stride = stream.cpu_data_stride;
        manifest.cpu_block_stride = stream.cpu_block_stride;
        manifest.cpu_wire_columns_per_unit = 64;
        manifest.source_projection=stream.source_projection;
        manifest.destination_projection=stream.destination_projection;
        manifest.region_bytes[0] = stream.total_stream_bytes;
        return seal(std::move(manifest));
    }

    MoEOverlayRemoteProjectionManifest
    makeMoEOverlayRemoteGpuProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const GpuExpertPackedDescriptor &source,
        NativeQuantisedSourceIdentity source_identity,
        std::uint32_t maximum_chunk_bytes)
    {
        if (!source.valid() || !identity.source_device.is_gpu() ||
            !identity.destination_device.is_gpu())
        {
            throw std::invalid_argument(
                "Remote GPU blob projection requires valid GPU endpoints and source arrays");
        }
        auto manifest = baseManifest(
            identity,
            source_identity,
            source.n,
            source.k,
            maximum_chunk_bytes);
        manifest.packing = MoEOverlayRemoteProjectionPacking::
            GpuSeparatedNativeQuantised;
        manifest.gpu_codebook_id = source.codebook_id;
        manifest.gpu_payload_bytes_per_block =
            source.payload_bytes_per_block;
        manifest.gpu_is_asymmetric = source.is_asymmetric ? 1u : 0u;
        manifest.gpu_has_emins = source.has_emins ? 1u : 0u;
        manifest.region_bytes = {
            source.vnni_bytes,
            source.scales_bytes,
            source.mins_bytes,
            source.emins_bytes,
        };
        return seal(std::move(manifest));
    }

    namespace {
        /** @brief Shared floating metadata builder; only the caller can authenticate actual storage. */
        MoEOverlayRemoteProjectionManifest floatingProjectionGeometry(
            const MoEOverlayRemoteProjectionIdentity &identity, TensorType type,
            int rows, int columns, uint32_t capacity)
        {
            const auto format = ExpertWeightFormat::floating(type);
            if (!identity.valid() || !format.valid() || rows <= 0 || columns <= 0 || capacity == 0)
                throw std::invalid_argument("Floating wire geometry requires original endpoints, precision and bounds");
            MoEOverlayRemoteProjectionManifest manifest;
            manifest.packing = MoEOverlayRemoteProjectionPacking::ContiguousFloating;
            manifest.format_kind = format.kind;
            manifest.identity = identity;
            manifest.N = rows; manifest.K = columns; manifest.N_padded = rows;
            manifest.cpu_encoding = cpu::native_quantised::CPUNativeQuantisedEncoding::NibbleLUT;
            manifest.region_bytes[0] = uint64_t(rows) * columns * format.floatingElementBytes();
            manifest.maximum_chunk_bytes = capacity;
            return seal(std::move(manifest));
        }
    }
    MoEOverlayRemoteProjectionManifest makeMoEOverlayRemoteFloatingProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const ContiguousFloatingPointWeightDescriptor &source, uint32_t capacity)
    {
        if (!source.valid()) throw std::invalid_argument("Floating remote source requires its genuine retained storage");
        return floatingProjectionGeometry(identity, source.type, source.n, source.k, capacity);
    }
    MoEOverlayRemoteProjectionManifest makeMoEOverlayRemoteFloatingDestinationProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const ContiguousFloatingPointWeightDescriptor &destination,
        const ExpertTierWeightProjectionIntersection &intersection, uint32_t capacity)
    {
        if (!destination.valid() || intersection.empty() || destination.n != intersection.destination().rows ||
            destination.k != intersection.destination().columns)
            throw std::invalid_argument("Floating receiver expectation changed its actual original destination");
        return floatingProjectionGeometry(identity, destination.type, intersection.source().rows,
            intersection.source().columns, capacity).forFloatingIntersection(intersection);
    }

    ExpertTierWeightCPUStorageLayout remoteCpuProjectionStorageLayout(
        const MoEOverlayRemoteProjectionManifest& manifest,MoEOverlayRemoteCPUStorageEndpoint endpoint)
    {
        const bool source=endpoint==MoEOverlayRemoteCPUStorageEndpoint::Source;
        if((!source && endpoint!=MoEOverlayRemoteCPUStorageEndpoint::Destination) || !manifest.valid() || !manifest.carriesCpuBytes() ||
           !(source?manifest.identity.source_device:manifest.identity.destination_device).is_cpu())
            throw std::invalid_argument("Native CPU storage layout requires an authenticated actual CPU endpoint");
        const auto& owned=source?manifest.source_projection:manifest.destination_projection;
        ExpertTierWeightCPUStorageLayout format{
            .cpu_codebook_id = manifest.cpu_codebook_id, .cpu_is_asymmetric = manifest.cpu_is_asymmetric,
            .cpu_is_superblock = manifest.cpu_is_superblock, .cpu_encoding = manifest.cpu_encoding,
            .cpu_data_stride = manifest.cpu_data_stride, .cpu_block_stride = manifest.cpu_block_stride,
            .wire_columns_per_unit = manifest.cpu_wire_columns_per_unit};
        return cpuExpertTierStorageLayoutForIntersection(format, owned,
            ExpertTierWeightProjectionIntersection::between(manifest.source_projection, manifest.destination_projection),
            manifest.maximum_chunk_bytes);
    }

    ExpertTierWeightDeviceLayout remoteCpuProjectionDeviceLayout(
        const MoEOverlayRemoteProjectionManifest &manifest)
    {
        std::string error;
        if (!manifest.valid(&error) || !manifest.carriesCpuBytes())
        {
            throw std::invalid_argument(
                error.empty()
                    ? "Remote CPU/GPU projection layout requires a valid CPU wire manifest"
                    : std::move(error));
        }
        const bool gpu_to_cpu = manifest.identity.source_device.is_gpu() &&
                                manifest.identity.destination_device.is_cpu();
        const bool cpu_to_gpu = manifest.identity.source_device.is_cpu() &&
                                manifest.identity.destination_device.is_gpu();
        if (!gpu_to_cpu && !cpu_to_gpu)
        {
            throw std::invalid_argument(
                "Remote CPU projection device layout requires exactly one GPU endpoint");
        }
        if (manifest.cpu_block_stride == 0 ||
            (manifest.maximum_chunk_bytes % manifest.cpu_block_stride) != 0)
        {
            throw std::invalid_argument(
                "Remote CPU/GPU projection chunks must contain complete NativeQuantised units");
        }

        const std::uint64_t units =
            static_cast<std::uint64_t>(manifest.N_padded / 64) *
            static_cast<std::uint64_t>(manifest.blocks_per_row);
        const std::uint64_t maximum_units =
            manifest.maximum_chunk_bytes / manifest.cpu_block_stride;
        if (units == 0 ||
            units > std::numeric_limits<std::uint32_t>::max() ||
            maximum_units == 0 || maximum_units > units)
        {
            throw std::invalid_argument(
                "Remote CPU/GPU projection unit geometry exceeds the device layout ABI");
        }

        ExpertTierWeightDeviceLayout layout{
            .direction = gpu_to_cpu
                             ? ExpertTierWeightConversionDirection::GpuToCpu
                             : ExpertTierWeightConversionDirection::CpuToGpu,
            .N = manifest.N,
            .K = manifest.K,
            .N_padded = manifest.N_padded,
            .blocks_per_row = manifest.blocks_per_row,
            .cpu_N = (gpu_to_cpu?manifest.destination_projection:manifest.source_projection).rows,
            .cpu_K = (gpu_to_cpu?manifest.destination_projection:manifest.source_projection).columns,
            .cpu_first_row = std::max(manifest.source_projection.first_row,manifest.destination_projection.first_row)-(gpu_to_cpu?manifest.destination_projection:manifest.source_projection).first_row,
            .cpu_first_column = std::max(manifest.source_projection.first_column,manifest.destination_projection.first_column)-(gpu_to_cpu?manifest.destination_projection:manifest.source_projection).first_column,
            .gpu_N = (gpu_to_cpu?manifest.source_projection:manifest.destination_projection).rows,
            .gpu_K = (gpu_to_cpu?manifest.source_projection:manifest.destination_projection).columns,
            .gpu_first_row = std::max(manifest.source_projection.first_row,manifest.destination_projection.first_row)-(gpu_to_cpu?manifest.source_projection:manifest.destination_projection).first_row,
            .gpu_first_column = std::max(manifest.source_projection.first_column,manifest.destination_projection.first_column)-(gpu_to_cpu?manifest.source_projection:manifest.destination_projection).first_column,
            .cpu_codebook_id = manifest.cpu_codebook_id,
            .gpu_source_codebook_id = manifest.source_codebook_id,
            .gpu_source_is_superblock = manifest.source_is_superblock,
            .gpu_codebook_id = manifest.gpu_codebook_id,
            .gpu_payload_bytes_per_block =
                manifest.gpu_payload_bytes_per_block,
            .cpu_is_asymmetric = manifest.cpu_is_asymmetric,
            .cpu_is_superblock = manifest.cpu_is_superblock,
            .gpu_is_asymmetric = manifest.gpu_is_asymmetric,
            .gpu_has_emins = manifest.gpu_has_emins,
            .cpu_encoding = manifest.cpu_encoding,
            // Remote-manifest wire semantics: the protocol's CPU bytes are
            // canonical (a versioned statement; the A3 protocol update
            // carries the endpoint layout explicitly).
            .cpu_layout =
                cpu::native_quantised::CPUNativeQuantisedLayout::InterleavedCanonical,
            .cpu_data_stride = manifest.cpu_data_stride,
            .cpu_block_stride = manifest.cpu_block_stride,
            .unit_count = static_cast<std::uint32_t>(units),
            .maximum_units_per_chunk =
                static_cast<std::uint32_t>(maximum_units),
        };
        // Derive the GPU conversion contract's wire planes from the CPU
        // endpoint layout's tables once, here; an endpoint layout without
        // complete tables fails closed before any repack kernel runs.
        if (!layout.deriveCpuEndpointWirePlanes() || !layout.valid())
        {
            throw std::invalid_argument(
                "Remote CPU/GPU projection fields do not form a supported device repack layout");
        }
        return layout;
    }

    bool encodeMoEOverlayRemoteProjectionManifest(
        const MoEOverlayRemoteProjectionManifest &manifest,
        std::span<std::uint8_t> destination,
        std::string *error) noexcept
    {
        if (destination.size() !=
                MoEOverlayRemoteProjectionManifest::kWireBytes ||
            !manifest.valid(error))
        {
            if (destination.size() !=
                MoEOverlayRemoteProjectionManifest::kWireBytes)
            {
                return reject(
                    error,
                    "Remote ExpertOverlay manifest destination has the wrong size");
            }
            return false;
        }
        std::fill(destination.begin(), destination.end(), 0u);
        std::size_t offset = 0;
        writeLittleEndian(destination, offset, manifest.magic);
        writeLittleEndian(destination, offset, manifest.abi_version);
        writeLittleEndian(destination, offset, manifest.packing);
        writeLittleEndian(destination, offset, manifest.format_kind);
        const auto &identity = manifest.identity;
        writeLittleEndian(destination, offset, identity.expected_epoch);
        writeLittleEndian(destination, offset, identity.candidate_epoch);
        writeLittleEndian(
            destination, offset, identity.execution_fingerprint.low);
        writeLittleEndian(
            destination, offset, identity.execution_fingerprint.high);
        writeLittleEndian(destination, offset, identity.migration_index);
        writeLittleEndian(destination, offset, identity.layer_idx);
        writeLittleEndian(destination, offset, identity.expert_id);
        writeLittleEndian(destination, offset, identity.projection);
        for (const auto byte : identity.reserved_identity)
            writeLittleEndian(destination, offset, byte);
        writeLittleEndian(destination, offset, identity.source_participant);
        writeLittleEndian(destination, offset, identity.destination_participant);
        writeLittleEndian(destination, offset, identity.source_world_rank);
        writeLittleEndian(destination, offset, identity.destination_world_rank);
        writeDevice(destination, offset, identity.source_device);
        writeDevice(destination, offset, identity.destination_device);
        writeLittleEndian(destination, offset, manifest.N);
        writeLittleEndian(destination, offset, manifest.K);
        writeLittleEndian(destination, offset, manifest.N_padded);
        writeLittleEndian(destination, offset, manifest.blocks_per_row);
        writeLittleEndian(destination, offset, manifest.source_codebook_id);
        writeLittleEndian(destination, offset, manifest.source_is_superblock);
        writeLittleEndian(destination, offset, manifest.cpu_codebook_id);
        writeLittleEndian(destination, offset, manifest.gpu_codebook_id);
        writeLittleEndian(
            destination, offset, manifest.gpu_payload_bytes_per_block);
        writeLittleEndian(destination, offset, manifest.cpu_is_asymmetric);
        writeLittleEndian(destination, offset, manifest.cpu_is_superblock);
        writeLittleEndian(destination, offset, manifest.gpu_is_asymmetric);
        writeLittleEndian(destination, offset, manifest.gpu_has_emins);
        writeLittleEndian(destination, offset, manifest.cpu_encoding);
        for (const auto byte : manifest.reserved_format)
            writeLittleEndian(destination, offset, byte);
        writeLittleEndian(destination, offset, manifest.cpu_data_stride);
        writeLittleEndian(destination, offset, manifest.cpu_block_stride);
        writeLittleEndian(destination, offset, manifest.cpu_wire_columns_per_unit);
        for (const auto bytes : manifest.region_bytes)
            writeLittleEndian(destination, offset, bytes);
        writeLittleEndian(destination, offset, manifest.total_bytes);
        writeLittleEndian(destination, offset, manifest.maximum_chunk_bytes);
        writeLittleEndian(destination, offset, manifest.reserved_size);
        writeLittleEndian(destination, offset, manifest.manifest_hash);
        for(const auto& owned:{manifest.source_projection,manifest.destination_projection}) {
            writeLittleEndian<int32_t>(destination,offset,static_cast<int32_t>(owned.residence));
            for(const auto field:{owned.source_rows,owned.source_columns,owned.first_row,owned.rows,owned.first_column,owned.columns})
                writeLittleEndian<int32_t>(destination,offset,field);
        }
        if (offset != destination.size())
            return reject(error, "Remote ExpertOverlay manifest encoder size drift");
        if (error)
            error->clear();
        return true;
    }

    bool decodeMoEOverlayRemoteProjectionManifest(
        std::span<const std::uint8_t> packet,
        MoEOverlayRemoteProjectionManifest *manifest,
        std::string *error) noexcept
    {
        if (!manifest ||
            packet.size() != MoEOverlayRemoteProjectionManifest::kWireBytes)
        {
            if (manifest)
                *manifest = {};
            return reject(
                error,
                "Remote ExpertOverlay manifest packet or destination is invalid");
        }
        *manifest = {};
        std::size_t offset = 0;
        manifest->magic = readLittleEndian<std::uint32_t>(packet, offset);
        manifest->abi_version =
            readLittleEndian<std::uint16_t>(packet, offset);
        manifest->packing =
            readLittleEndian<MoEOverlayRemoteProjectionPacking>(packet, offset);
        manifest->format_kind =
            readLittleEndian<ExpertWeightFormatKind>(packet, offset);
        auto &identity = manifest->identity;
        identity.expected_epoch =
            readLittleEndian<std::uint64_t>(packet, offset);
        identity.candidate_epoch =
            readLittleEndian<std::uint64_t>(packet, offset);
        identity.execution_fingerprint.low =
            readLittleEndian<std::uint64_t>(packet, offset);
        identity.execution_fingerprint.high =
            readLittleEndian<std::uint64_t>(packet, offset);
        identity.migration_index =
            readLittleEndian<std::uint64_t>(packet, offset);
        identity.layer_idx = readLittleEndian<std::int32_t>(packet, offset);
        identity.expert_id = readLittleEndian<std::int32_t>(packet, offset);
        identity.projection =
            readLittleEndian<ExpertTierWeightProjection>(packet, offset);
        for (auto &byte : identity.reserved_identity)
            byte = readLittleEndian<std::uint8_t>(packet, offset);
        identity.source_participant =
            readLittleEndian<std::int32_t>(packet, offset);
        identity.destination_participant =
            readLittleEndian<std::int32_t>(packet, offset);
        identity.source_world_rank =
            readLittleEndian<std::int32_t>(packet, offset);
        identity.destination_world_rank =
            readLittleEndian<std::int32_t>(packet, offset);
        identity.source_device = readDevice(packet, offset);
        identity.destination_device = readDevice(packet, offset);
        manifest->N = readLittleEndian<std::int32_t>(packet, offset);
        manifest->K = readLittleEndian<std::int32_t>(packet, offset);
        manifest->N_padded = readLittleEndian<std::int32_t>(packet, offset);
        manifest->blocks_per_row =
            readLittleEndian<std::int32_t>(packet, offset);
        manifest->source_codebook_id =
            readLittleEndian<std::uint8_t>(packet, offset);
        manifest->source_is_superblock =
            readLittleEndian<std::uint8_t>(packet, offset);
        manifest->cpu_codebook_id =
            readLittleEndian<std::uint8_t>(packet, offset);
        manifest->gpu_codebook_id =
            readLittleEndian<std::uint8_t>(packet, offset);
        manifest->gpu_payload_bytes_per_block =
            readLittleEndian<std::uint8_t>(packet, offset);
        manifest->cpu_is_asymmetric =
            readLittleEndian<std::uint8_t>(packet, offset);
        manifest->cpu_is_superblock =
            readLittleEndian<std::uint8_t>(packet, offset);
        manifest->gpu_is_asymmetric =
            readLittleEndian<std::uint8_t>(packet, offset);
        manifest->gpu_has_emins =
            readLittleEndian<std::uint8_t>(packet, offset);
        manifest->cpu_encoding =
            readLittleEndian<
                cpu::native_quantised::CPUNativeQuantisedEncoding>(packet, offset);
        for (auto &byte : manifest->reserved_format)
            byte = readLittleEndian<std::uint8_t>(packet, offset);
        manifest->cpu_data_stride =
            readLittleEndian<std::uint32_t>(packet, offset);
        manifest->cpu_block_stride =
            readLittleEndian<std::uint32_t>(packet, offset);
        manifest->cpu_wire_columns_per_unit = readLittleEndian<std::uint32_t>(packet,offset);
        for (auto &bytes : manifest->region_bytes)
            bytes = readLittleEndian<std::uint64_t>(packet, offset);
        manifest->total_bytes =
            readLittleEndian<std::uint64_t>(packet, offset);
        manifest->maximum_chunk_bytes =
            readLittleEndian<std::uint32_t>(packet, offset);
        manifest->reserved_size =
            readLittleEndian<std::uint32_t>(packet, offset);
        manifest->manifest_hash =
            readLittleEndian<std::uint64_t>(packet, offset);
        for(auto* owned:{&manifest->source_projection,&manifest->destination_projection}) {
            owned->residence=static_cast<MoEProjectionResidence>(readLittleEndian<int32_t>(packet,offset));
            for(auto* field:{&owned->source_rows,&owned->source_columns,&owned->first_row,&owned->rows,&owned->first_column,&owned->columns})
                *field=readLittleEndian<int32_t>(packet,offset);
        }
        if (offset != packet.size() || !manifest->valid(error))
        {
            if (offset != packet.size())
                reject(error, "Remote ExpertOverlay manifest decoder size drift");
            *manifest = {};
            return false;
        }
        if (error)
            error->clear();
        return true;
    }

    bool MoEOverlayRemoteProjectionChunkHeader::structurallyValid()
        const noexcept
    {
        return manifest_hash != 0 && payload_hash != 0 &&
               region < kMoEOverlayRemoteProjectionRegionCount &&
               allZero(reserved_region) && payload_bytes > 0 &&
               final_chunk <= 1u && allZero(reserved_final);
    }

    bool encodeMoEOverlayRemoteProjectionChunkHeader(
        const MoEOverlayRemoteProjectionChunkHeader &header,
        std::span<std::uint8_t> destination,
        std::string *error) noexcept
    {
        if (destination.size() !=
                MoEOverlayRemoteProjectionChunkHeader::kWireBytes ||
            !header.structurallyValid())
        {
            return reject(
                error,
                "Remote ExpertOverlay chunk header or destination is invalid");
        }
        std::fill(destination.begin(), destination.end(), 0u);
        std::size_t offset = 0;
        writeLittleEndian(destination, offset, header.manifest_hash);
        writeLittleEndian(destination, offset, header.payload_hash);
        writeLittleEndian(destination, offset, header.sequence);
        writeLittleEndian(destination, offset, header.region);
        for (const auto byte : header.reserved_region)
            writeLittleEndian(destination, offset, byte);
        writeLittleEndian(destination, offset, header.region_offset);
        writeLittleEndian(destination, offset, header.payload_bytes);
        writeLittleEndian(destination, offset, header.final_chunk);
        for (const auto byte : header.reserved_final)
            writeLittleEndian(destination, offset, byte);
        if (offset != destination.size())
            return reject(error, "Remote ExpertOverlay chunk encoder size drift");
        if (error)
            error->clear();
        return true;
    }

    bool decodeMoEOverlayRemoteProjectionChunkHeader(
        std::span<const std::uint8_t> packet,
        MoEOverlayRemoteProjectionChunkHeader *header,
        std::string *error) noexcept
    {
        if (!header ||
            packet.size() !=
                MoEOverlayRemoteProjectionChunkHeader::kWireBytes)
        {
            if (header)
                *header = {};
            return reject(
                error,
                "Remote ExpertOverlay chunk packet or destination is invalid");
        }
        *header = {};
        std::size_t offset = 0;
        header->manifest_hash =
            readLittleEndian<std::uint64_t>(packet, offset);
        header->payload_hash =
            readLittleEndian<std::uint64_t>(packet, offset);
        header->sequence =
            readLittleEndian<std::uint64_t>(packet, offset);
        header->region = readLittleEndian<std::uint8_t>(packet, offset);
        for (auto &byte : header->reserved_region)
            byte = readLittleEndian<std::uint8_t>(packet, offset);
        header->region_offset =
            readLittleEndian<std::uint64_t>(packet, offset);
        header->payload_bytes =
            readLittleEndian<std::uint32_t>(packet, offset);
        header->final_chunk =
            readLittleEndian<std::uint8_t>(packet, offset);
        for (auto &byte : header->reserved_final)
            byte = readLittleEndian<std::uint8_t>(packet, offset);
        if (offset != packet.size() || !header->structurallyValid())
        {
            *header = {};
            return reject(
                error,
                "Remote ExpertOverlay chunk packet is malformed");
        }
        if (error)
            error->clear();
        return true;
    }

    MoEOverlayRemoteProjectionChunkCursor::
        MoEOverlayRemoteProjectionChunkCursor(
            MoEOverlayRemoteProjectionManifest manifest,
            std::array<
                std::span<const std::uint8_t>,
                kMoEOverlayRemoteProjectionRegionCount> regions)
        : manifest_(std::move(manifest)), regions_(regions)
    {
        std::string error;
        if (!manifest_.valid(&error))
            throw std::invalid_argument(error);
        for (std::size_t region = 0; region < regions_.size(); ++region)
        {
            const ExpertWeightFormat format{.kind=manifest_.format_kind};
            const size_t expected=region==0 && manifest_.carriesFloatingBytes()
                ? size_t(manifest_.source_projection.rows)*manifest_.source_projection.columns*format.floatingElementBytes()
                : requiresCPUStaging() && region==0 ? remoteCpuProjectionStorageLayout(manifest_,MoEOverlayRemoteCPUStorageEndpoint::Source).cpuStorageBytes():manifest_.region_bytes[region];
            if (regions_[region].size() != expected)
            {
                throw std::invalid_argument(
                    "Remote ExpertOverlay chunk cursor regions disagree with manifest");
            }
        }
        seekNextRegion();
    }

    void MoEOverlayRemoteProjectionChunkCursor::seekNextRegion() noexcept
    {
        while (region_ < regions_.size() &&
               region_offset_ == manifest_.region_bytes[region_])
        {
            ++region_;
            region_offset_ = 0;
        }
    }

    std::optional<MoEOverlayRemoteProjectionChunkView>
    MoEOverlayRemoteProjectionChunkCursor::takeNext(std::span<uint8_t> staging) noexcept
    {
        seekNextRegion();
        if (region_ == regions_.size())
            return std::nullopt;
        const std::uint64_t remaining =
            manifest_.region_bytes[region_] - region_offset_;
        std::uint64_t bytes=std::min<std::uint64_t>(remaining,manifest_.maximum_chunk_bytes);
        std::span<const uint8_t> payload;
        if(requiresCPUStaging()) {
            // The native root remains immutable. Reuse exactly the transport's
            // already-admitted send lane until its original MPI ACK retires it.
            const auto layout=remoteCpuProjectionStorageLayout(manifest_,MoEOverlayRemoteCPUStorageEndpoint::Source);
            const auto first=expertTierWireUnitAt(layout,region_offset_);
            const auto count=std::min(layout.maximum_units_per_chunk,layout.unit_count-first);
            bytes=layout.chunkBytes(count,first);
            if(staging.size()<bytes || !gatherExpertTierCPUWeightChunk(layout,regions_[0],first,count,staging.first(bytes)))
                return std::nullopt;
            payload=staging.first(bytes);
        } else if(manifest_.carriesFloatingBytes()) {
            const ExpertWeightFormat format{.kind=manifest_.format_kind};
            const size_t element=format.floatingElementBytes(),row_bytes=size_t(manifest_.K)*element;
            const auto& owned=manifest_.source_projection;
            if(owned.columns!=manifest_.K || manifest_.destination_projection.columns!=manifest_.K)
                bytes=std::min(bytes,row_bytes-region_offset_%row_bytes);
            const auto first_row=std::max(owned.first_row,manifest_.destination_projection.first_row)-owned.first_row;
            const auto first_col=std::max(owned.first_column,manifest_.destination_projection.first_column)-owned.first_column;
            const size_t offset=(size_t(first_row)+region_offset_/row_bytes)*owned.columns*element+size_t(first_col)*element+region_offset_%row_bytes;
            payload=regions_[0].subspan(offset,bytes);
        } else payload=regions_[region_].subspan(static_cast<size_t>(region_offset_),static_cast<size_t>(bytes));
        const bool final = emitted_bytes_ + bytes == manifest_.total_bytes;
        MoEOverlayRemoteProjectionChunkView result{
            .header = {
                .manifest_hash = manifest_.manifest_hash,
                .payload_hash = expertTierWeightBytesHash(payload),
                .sequence = sequence_,
                .region = static_cast<std::uint8_t>(region_),
                .region_offset = region_offset_,
                .payload_bytes = static_cast<std::uint32_t>(bytes),
                .final_chunk = static_cast<std::uint8_t>(final ? 1u : 0u),
            },
            .payload = payload,
        };
        ++sequence_;
        region_offset_ += bytes;
        emitted_bytes_ += bytes;
        seekNextRegion();
        return result;
    }

    bool MoEOverlayRemoteProjectionChunkCursor::complete() const noexcept
    {
        return emitted_bytes_ == manifest_.total_bytes;
    }

    MoEOverlayRemoteProjectionChunkValidator::
        MoEOverlayRemoteProjectionChunkValidator(
            MoEOverlayRemoteProjectionManifest manifest)
        : manifest_(std::move(manifest))
    {
        std::string error;
        if (!manifest_.valid(&error))
            throw std::invalid_argument(error);
        seekNextRegion();
    }

    void MoEOverlayRemoteProjectionChunkValidator::seekNextRegion() noexcept
    {
        while (region_ < manifest_.region_bytes.size() &&
               region_offset_ == manifest_.region_bytes[region_])
        {
            ++region_;
            region_offset_ = 0;
        }
    }

    bool MoEOverlayRemoteProjectionChunkValidator::accept(
        const MoEOverlayRemoteProjectionChunkHeader &header,
        std::span<const std::uint8_t> payload,
        std::string *error) noexcept
    {
        if (aborted_)
            return reject(error, "Remote ExpertOverlay chunk validator is aborted");
        if (complete_)
            return reject(error, "Remote ExpertOverlay projection already completed");
        seekNextRegion();
        if (!header.structurallyValid() ||
            header.manifest_hash != manifest_.manifest_hash ||
            header.sequence != next_sequence_ ||
            region_ >= manifest_.region_bytes.size() ||
            header.region != region_ ||
            header.region_offset != region_offset_ ||
            header.payload_bytes != payload.size() ||
            header.payload_bytes > manifest_.maximum_chunk_bytes ||
            header.payload_bytes >
                manifest_.region_bytes[region_] - region_offset_)
        {
            return reject(
                error,
                "Remote ExpertOverlay chunk identity, order, or range is invalid");
        }
        if (header.payload_hash != expertTierWeightBytesHash(payload))
            return reject(error, "Remote ExpertOverlay chunk payload authentication failed");

        const bool must_be_final =
            received_bytes_ + payload.size() == manifest_.total_bytes;
        if ((header.final_chunk != 0) != must_be_final)
        {
            return reject(
                error,
                "Remote ExpertOverlay chunk has an invalid final marker");
        }

        ++next_sequence_;
        region_offset_ += payload.size();
        received_bytes_ += payload.size();
        seekNextRegion();
        complete_ = must_be_final;
        if (error)
            error->clear();
        return true;
    }
} // namespace llaminar2
