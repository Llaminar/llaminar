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
            hashScalar(hash, identity.transaction_fingerprint.low);
            hashScalar(hash, identity.transaction_fingerprint.high);
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
            const NativeVnniFormatInfo &source) noexcept
        {
            const std::uint8_t canonical =
                canonicalDeviceVnniCodebookId(source.codebook_id);
            if (manifest.gpu_codebook_id == canonical)
            {
                return manifest.gpu_payload_bytes_per_block ==
                           source.payload_bytes &&
                       manifest.gpu_is_asymmetric ==
                           static_cast<std::uint8_t>(source.is_asymmetric) &&
                       manifest.gpu_has_emins ==
                           static_cast<std::uint8_t>(source.has_emins);
            }

            /*
             * A CPU promotion may no longer possess compact source bits.  Its
             * destination GPU then executes normalized signed bytes while the
             * source identity remains the mathematical GGUF provenance.
             */
            if (cpu::native_vnni::preparedEncodingForCodebook(
                    source.codebook_id) !=
                cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8)
            {
                return false;
            }
            const std::uint8_t normalized =
                source.is_asymmetric
                    ? kNativeVnniExpandedInt8MinCodebook
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
            manifest.manifest_hash = manifest.computedHash();
            std::string error;
            if (!manifest.valid(&error))
                throw std::invalid_argument(error);
            return manifest;
        }

        /** @brief Populate fields shared by every physical packing. */
        MoEOverlayRemoteProjectionManifest baseManifest(
            const MoEOverlayRemoteProjectionIdentity &identity,
            NativeVnniSourceIdentity source_identity,
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
            const auto *source = native_vnni_formats::forSourceIdentity(
                source_identity.codebook_id,
                source_identity.is_superblock);
            if (!source)
            {
                throw std::invalid_argument(
                    "Remote ExpertOverlay projection provenance is not catalogued");
            }

            MoEOverlayRemoteProjectionManifest manifest;
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
                cpu::native_vnni::preparedEncodingForCodebook(
                    source_identity.codebook_id);
            manifest.cpu_data_stride =
                cpu::native_vnni::preparedDataStride(manifest.cpu_encoding);
            manifest.cpu_block_stride = static_cast<std::uint32_t>(
                cpu::native_vnni::preparedInterleavedBlockStride(
                    manifest.cpu_encoding,
                    manifest.cpu_is_asymmetric != 0));
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
               transaction_fingerprint.valid() && layer_idx >= 0 &&
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
        for (const auto bytes : region_bytes)
            hashScalar(hash, bytes);
        hashScalar(hash, total_bytes);
        hashScalar(hash, maximum_chunk_bytes);
        return hash == 0 ? 0x9e3779b97f4a7c15ull : hash;
    }

    bool MoEOverlayRemoteProjectionManifest::valid(
        std::string *error) const noexcept
    {
        if (magic != kMagic || abi_version != kABIVersion ||
            reserved_protocol != 0 || !identity.valid() ||
            !allZero(reserved_format) || reserved_size != 0)
        {
            return reject(
                error,
                "Remote ExpertOverlay projection has invalid ABI, identity, or reserved fields");
        }
        if (N <= 0 || K <= 0 || (K % 32) != 0 ||
            N_padded != ((N + 63) / 64) * 64 ||
            blocks_per_row != K / 32 || maximum_chunk_bytes == 0)
        {
            return reject(
                error,
                "Remote ExpertOverlay projection geometry or chunk capacity is invalid");
        }
        const auto *source = native_vnni_formats::forSourceIdentity(
            source_codebook_id, source_is_superblock != 0);
        if (!source || cpu_codebook_id != source_codebook_id ||
            cpu_is_asymmetric !=
                static_cast<std::uint8_t>(source->is_asymmetric) ||
            cpu_is_superblock != source_is_superblock ||
            cpu_encoding !=
                cpu::native_vnni::preparedEncodingForCodebook(
                    source_codebook_id) ||
            cpu_data_stride !=
                cpu::native_vnni::preparedDataStride(cpu_encoding) ||
            cpu_block_stride !=
                cpu::native_vnni::preparedInterleavedBlockStride(
                    cpu_encoding, cpu_is_asymmetric != 0))
        {
            return reject(
                error,
                "Remote ExpertOverlay projection provenance and CPU format disagree");
        }

        std::uint64_t summed = 0;
        if (!sumRegions(region_bytes, summed) || summed == 0 ||
            summed != total_bytes)
        {
            return reject(
                error,
                "Remote ExpertOverlay projection region byte totals are invalid");
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
            const std::uint64_t expected_units =
                static_cast<std::uint64_t>(N_padded / 64) *
                static_cast<std::uint64_t>(blocks_per_row);
            const std::uint64_t expected_bytes =
                expected_units * cpu_block_stride;
            if (region_bytes[0] != expected_bytes ||
                region_bytes[1] != 0 || region_bytes[2] != 0 ||
                region_bytes[3] != 0)
            {
                return reject(
                    error,
                    "CPU-format remote projection does not contain exact final units");
            }
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
                 * 32-K NativeVNNI units.  Reject a malformed wire capacity at
                 * manifest authentication rather than accepting it and
                 * discovering the partial-unit boundary after MPI has begun
                 * mutating destination state.
                 */
                if (cpu_block_stride == 0 ||
                    maximum_chunk_bytes < cpu_block_stride ||
                    (maximum_chunk_bytes % cpu_block_stride) != 0)
                {
                    return reject(
                        error,
                        "CPU/GPU remote projection chunks must contain complete NativeVNNI units");
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
            if (!identity.source_device.is_gpu() ||
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

    MoEOverlayRemoteProjectionManifest
    makeMoEOverlayRemoteCpuProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const cpu::native_vnni::CPUNativeVNNIPackedWeights &source,
        std::uint32_t maximum_chunk_bytes)
    {
        const NativeVnniSourceIdentity source_identity{
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
        if (!identity.source_device.is_cpu() ||
            !identity.destination_device.is_cpu() || source.N_padded !=
                manifest.N_padded || source.blocks_per_row !=
                manifest.blocks_per_row || source.encoding !=
                manifest.cpu_encoding || source.data_stride !=
                static_cast<int>(manifest.cpu_data_stride) ||
            source.interleaved_block_stride !=
                manifest.cpu_block_stride || source.native_interleaved.empty())
        {
            throw std::invalid_argument(
                "Remote CPU-to-CPU projection source does not match canonical final storage");
        }
        manifest.packing = MoEOverlayRemoteProjectionPacking::
            CpuNativeVnniInterleaved;
        manifest.region_bytes[0] = source.native_interleaved.size();
        return seal(std::move(manifest));
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
        const NativeVnniSourceIdentity source_identity{
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
            ExpertTierWeightPacking::GpuSeparatedNativeVnni;
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
            CpuNativeVnniInterleaved;
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
        manifest.region_bytes[0] = stream.total_stream_bytes;
        return seal(std::move(manifest));
    }

    MoEOverlayRemoteProjectionManifest
    makeMoEOverlayRemoteGpuProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const GpuExpertPackedDescriptor &source,
        NativeVnniSourceIdentity source_identity,
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
            GpuSeparatedNativeVnni;
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
                "Remote CPU/GPU projection chunks must contain complete NativeVNNI units");
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
            .cpu_data_stride = manifest.cpu_data_stride,
            .cpu_block_stride = manifest.cpu_block_stride,
            .unit_count = static_cast<std::uint32_t>(units),
            .maximum_units_per_chunk =
                static_cast<std::uint32_t>(maximum_units),
        };
        if (!layout.valid())
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
        writeLittleEndian(destination, offset, manifest.reserved_protocol);
        const auto &identity = manifest.identity;
        writeLittleEndian(destination, offset, identity.expected_epoch);
        writeLittleEndian(destination, offset, identity.candidate_epoch);
        writeLittleEndian(
            destination, offset, identity.transaction_fingerprint.low);
        writeLittleEndian(
            destination, offset, identity.transaction_fingerprint.high);
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
        for (const auto bytes : manifest.region_bytes)
            writeLittleEndian(destination, offset, bytes);
        writeLittleEndian(destination, offset, manifest.total_bytes);
        writeLittleEndian(destination, offset, manifest.maximum_chunk_bytes);
        writeLittleEndian(destination, offset, manifest.reserved_size);
        writeLittleEndian(destination, offset, manifest.manifest_hash);
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
        manifest->reserved_protocol =
            readLittleEndian<std::uint8_t>(packet, offset);
        auto &identity = manifest->identity;
        identity.expected_epoch =
            readLittleEndian<std::uint64_t>(packet, offset);
        identity.candidate_epoch =
            readLittleEndian<std::uint64_t>(packet, offset);
        identity.transaction_fingerprint.low =
            readLittleEndian<std::uint64_t>(packet, offset);
        identity.transaction_fingerprint.high =
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
                cpu::native_vnni::CPUNativeVNNIEncoding>(packet, offset);
        for (auto &byte : manifest->reserved_format)
            byte = readLittleEndian<std::uint8_t>(packet, offset);
        manifest->cpu_data_stride =
            readLittleEndian<std::uint32_t>(packet, offset);
        manifest->cpu_block_stride =
            readLittleEndian<std::uint32_t>(packet, offset);
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
            if (regions_[region].size() != manifest_.region_bytes[region])
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
    MoEOverlayRemoteProjectionChunkCursor::takeNext() noexcept
    {
        seekNextRegion();
        if (region_ == regions_.size())
            return std::nullopt;
        const std::uint64_t remaining =
            manifest_.region_bytes[region_] - region_offset_;
        const std::uint64_t bytes = std::min<std::uint64_t>(
            remaining, manifest_.maximum_chunk_bytes);
        const auto payload = regions_[region_].subspan(
            static_cast<std::size_t>(region_offset_),
            static_cast<std::size_t>(bytes));
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
