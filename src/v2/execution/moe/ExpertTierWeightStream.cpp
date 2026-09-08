/**
 * @file ExpertTierWeightStream.cpp
 * @brief Validation, receive protocol, and byte-exact host oracle for expert tiers.
 *
 * @details Cross-backend ExpertOverlay movement uses the CPU's final prepared
 * representation as its wire format. A GPU-to-CPU sender produces complete
 * 64-output-column by 32-K units; a CPU-to-GPU receiver consumes the same
 * units while reconstructing the common CUDA/ROCm separated representation.
 * This file owns the device-free contract and deliberately performs no device
 * I/O. Backend kernels are checked against the reference conversions below.
 *
 * The important invariants are:
 * - one manifest names one immutable residency epoch and projection;
 * - chunks arrive once, in increasing unit order, with no partial units;
 * - every layout field participates in the manifest hash;
 * - receiver state advances only after the complete chunk validates;
 * - padded CPU columns are deterministic zero bytes; and
 * - Q6_K high bits are transposed without changing quantized values.
 */

#include "ExpertTierWeightStream.h"

#include "../../utils/FNV1a.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace llaminar2
{
    namespace
    {
        using CPUEncoding = cpu::native_vnni::CPUNativeVNNIEncoding;
        using CPUWeights = cpu::native_vnni::CPUNativeVNNIPackedWeights;

        /**
         * @brief Record an optional validation diagnostic and return failure.
         * @param error Optional caller-owned diagnostic destination.
         * @param message Stable explanation of the rejected invariant.
         * @return Always `false`, allowing validation branches to return it.
         */
        bool reject(std::string *error, const char *message) noexcept
        {
            if (error)
                *error = message;
            return false;
        }

        /**
         * @brief Append one scalar to the canonical manifest hash.
         * @tparam T Trivially copyable scalar or scoped-enum storage type.
         * @param hash FNV-1a state produced by all preceding fields.
         * @param value Next scalar in the versioned field order.
         * @return Updated FNV-1a state.
         *
         * Hashing an explicit byte copy avoids aliasing violations. Manifest
         * versioning owns native-endian compatibility between participants.
         */
        template <typename T>
        uint64_t hashScalar(uint64_t hash, T value) noexcept
        {
            static_assert(std::is_trivially_copyable_v<T>);
            std::array<uint8_t, sizeof(T)> bytes{};
            std::memcpy(bytes.data(), &value, sizeof(T));
            return fnv1a64(bytes.data(), bytes.size(), hash);
        }

        /**
         * @brief Multiply two byte-count terms without unsigned overflow.
         * @param lhs Left factor.
         * @param rhs Right factor.
         * @param result Receives the product only when it is representable.
         * @return `true` when `result` is valid.
         */
        bool checkedMultiply(
            uint64_t lhs,
            uint64_t rhs,
            uint64_t &result) noexcept
        {
            if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
                return false;
            result = lhs * rhs;
            return true;
        }

        /**
         * @brief Identify GPU formats retained as compact nibble CPU payloads.
         * @param codebook NativeVNNI execution codebook.
         * @return Whether the CPU uses its nibble-LUT prepared encoding.
         */
        bool isNibbleCodebook(uint8_t codebook) noexcept
        {
            return codebook == 0 || codebook == 4 || codebook == 5;
        }

        /**
         * @brief Derive the only lossless GPU execution representation for a
         * prepared CPU projection.
         * @param cpu Valid or partially constructed CPU representation.
         * @param codebook Receives the GPU execution codebook.
         * @param payload_bytes Receives bytes in one separated GPU block.
         * @param is_asymmetric Receives whether a per-block minimum is present.
         * @param has_emins Receives whether packed effective minima are present.
         * @param error Optional rejection diagnostic.
         * @return `true` only when CPU metadata names a supported exact mapping.
         *
         * Expanded CPU data cannot recover its original compact codebook, so it
         * has explicit normalized GPU identities: codebook 19 when symmetric
         * and codebook 23 when a minimum must also be retained.
         */
        bool gpuFormatForCpu(
            const CPUWeights &cpu,
            uint8_t &codebook,
            uint8_t &payload_bytes,
            bool &is_asymmetric,
            bool &has_emins,
            std::string *error) noexcept
        {
            // None of the currently normalized stream formats uses Q2_K emins.
            has_emins = false;
            switch (cpu.encoding)
            {
            case CPUEncoding::NibbleLUT:
                if (!isNibbleCodebook(cpu.codebook_id) ||
                    cpu.payload_bytes != 16)
                {
                    return reject(
                        error,
                        "Nibble CPU stream requires codebook 0, 4, or 5 and a 16-byte payload");
                }
                codebook = cpu.codebook_id;
                payload_bytes = 16;
                is_asymmetric = cpu.is_asymmetric;
                return true;
            case CPUEncoding::Q6KNativeDualScale:
                if (cpu.codebook_id != 8 || cpu.payload_bytes != 24 ||
                    !cpu.is_asymmetric)
                {
                    return reject(
                        error,
                        "Native Q6 CPU stream requires codebook 8, 24-byte payload, and dual-scale metadata");
                }
                codebook = 8;
                payload_bytes = 24;
                is_asymmetric = true;
                return true;
            case CPUEncoding::ExpandedInt8:
                // Promotion from CPU retains already-decoded signed values.
                codebook = cpu.is_asymmetric
                               ? kNativeVnniExpandedInt8MinCodebook
                               : static_cast<uint8_t>(19);
                payload_bytes = 32;
                is_asymmetric = cpu.is_asymmetric;
                return true;
            }
            return reject(error, "Unknown CPU prepared encoding");
        }

        /**
         * @brief Locate the complete CPU unit containing column `n`, K block `kb`.
         * @param cpu Prepared CPU projection with validated geometry.
         * @param n Logical output column.
         * @param kb Logical 32-value K-block index.
         * @return Byte offset from `native_interleaved.data()` to the unit.
         */
        size_t cpuUnitOffset(
            const CPUWeights &cpu,
            int n,
            int kb) noexcept
        {
            // CPU units are ordered by 64-column chunk, then by K block.
            const size_t chunk = static_cast<size_t>(n / 64);
            return (chunk * static_cast<size_t>(cpu.blocks_per_row) +
                    static_cast<size_t>(kb)) *
                   static_cast<size_t>(cpu.interleaved_block_stride);
        }

        /**
         * @brief Locate one payload byte in the CPU VNNI register layout.
         * @param cpu Prepared CPU projection with validated geometry.
         * @param n Logical output column.
         * @param kb Logical 32-value K-block index.
         * @param value_index Byte index within the column's prepared payload.
         * @return Absolute byte offset in `native_interleaved`.
         *
         * Four adjacent bytes for one column occupy a lane in a 64-byte ZMM
         * image. Sixteen columns form one ZMM and four ZMMs cover the unit's
         * 64 columns. `group` advances to the next four values.
         */
        size_t interleavedValueOffset(
            const CPUWeights &cpu,
            int n,
            int kb,
            int value_index) noexcept
        {
            const int local_column = n % 64;
            const int group = value_index / 4;
            const int index = value_index % 4;
            const int zmm = local_column / 16;
            const int lane = local_column % 16;
            return cpuUnitOffset(cpu, n, kb) +
                   static_cast<size_t>(group) * 256u +
                   static_cast<size_t>(zmm) * 64u +
                   static_cast<size_t>(lane) * 4u +
                   static_cast<size_t>(index);
        }

        /**
         * @brief Load an unaligned 64-bit Q6 bit plane without aliasing UB.
         * @param source Address of eight serialized bytes.
         * @return Their native-endian integer bit representation.
         */
        uint64_t loadU64(const uint8_t *source) noexcept
        {
            uint64_t value = 0;
            std::memcpy(&value, source, sizeof(value));
            return value;
        }

        /**
         * @brief Store an unaligned 64-bit Q6 bit plane without aliasing UB.
         * @param destination Address of eight serialized bytes.
         * @param value Native-endian plane bits to write.
         */
        void storeU64(uint8_t *destination, uint64_t value) noexcept
        {
            std::memcpy(destination, &value, sizeof(value));
        }

        /**
         * @brief Validate a materialized CPU projection before streaming it.
         * @param cpu Candidate prepared representation.
         * @param error Optional exact rejection diagnostic.
         * @return `true` when geometry, strides, bytes, and GPU mapping agree.
         */
        bool validateCpuWeights(
            const CPUWeights &cpu,
            std::string *error) noexcept
        {
            // K must be a whole number of quant blocks; N is padded by units.
            if (cpu.N <= 0 || cpu.K <= 0 || cpu.N_padded < cpu.N ||
                (cpu.N_padded % 64) != 0 || (cpu.K % 32) != 0 ||
                cpu.blocks_per_row != cpu.K / 32)
            {
                return reject(error, "CPU packed geometry is invalid for tier streaming");
            }
            if (!cpu::native_vnni::isKnownPreparedEncoding(cpu.encoding))
                return reject(error, "CPU packed encoding is unknown");
            // Source provenance must identify exactly one of the 21 cataloged
            // formats even when its final CPU bytes have been normalized.
            if (native_vnni_formats::forSourceIdentity(
                    cpu.codebook_id, cpu.is_superblock) == nullptr)
            {
                return reject(
                    error,
                    "CPU packed source codebook/superblock identity is not cataloged");
            }

            // Never trust serialized stride fields independently of encoding.
            const uint32_t expected_data_stride =
                cpu::native_vnni::preparedDataStride(cpu.encoding);
            const int expected_block_stride =
                cpu::native_vnni::preparedInterleavedBlockStride(
                    cpu.encoding, cpu.is_asymmetric);
            if (cpu.data_stride != static_cast<int>(expected_data_stride) ||
                cpu.interleaved_block_stride != expected_block_stride)
            {
                return reject(error, "CPU packed stride does not match its prepared encoding");
            }

            // Overflow-safe size reconstruction prevents truncated allocations.
            uint64_t units = 0;
            uint64_t bytes = 0;
            if (!checkedMultiply(
                    static_cast<uint64_t>(cpu.N_padded / 64),
                    static_cast<uint64_t>(cpu.blocks_per_row),
                    units) ||
                !checkedMultiply(
                    units,
                    static_cast<uint64_t>(cpu.interleaved_block_stride),
                    bytes) ||
                bytes != cpu.native_interleaved.size())
            {
                return reject(error, "CPU interleaved byte count does not match geometry");
            }

            // A valid CPU byte layout must also have one exact GPU counterpart.
            uint8_t codebook = 0;
            uint8_t payload_bytes = 0;
            bool asym = false;
            bool emins = false;
            return gpuFormatForCpu(
                cpu, codebook, payload_bytes, asym, emins, error);
        }

        /**
         * @brief Initialize CPU layout metadata from a validated GPU projection.
         * @param gpu Source common separated representation.
         * @param cpu Receives metadata only; payload allocation happens later.
         *
         * This function selects physical CPU execution encoding, not the source
         * GGUF format. Expanded INT8 therefore remains expanded after migration.
         */
        void configureCpuFromGpu(
            const HostGpuExpertPackedProjection &gpu,
            CPUWeights &cpu)
        {
            // Start from a clean object so no stale optional buffer survives.
            cpu = CPUWeights{};
            cpu.N = gpu.N;
            cpu.K = gpu.K;
            cpu.N_padded = (gpu.N + 63) / 64 * 64;
            cpu.blocks_per_row = static_cast<int>(gpu.blocks_per_row);
            cpu.codebook_id = gpu.source_codebook_id;
            cpu.is_superblock = gpu.is_superblock;
            cpu.is_asymmetric = gpu.is_asymmetric;

            if (isNibbleCodebook(gpu.codebook_id))
            {
                cpu.encoding = CPUEncoding::NibbleLUT;
                cpu.payload_bytes = 16;
            }
            else if (gpu.codebook_id == 8)
            {
                cpu.encoding = CPUEncoding::Q6KNativeDualScale;
                cpu.payload_bytes = 24;
                cpu.is_asymmetric = true;
                cpu.is_superblock = true;
            }
            else
            {
                cpu.encoding = CPUEncoding::ExpandedInt8;
                cpu.payload_bytes = 32;
                /*
                 * ExpandedInt8 describes the CPU execution bytes, while
                 * `is_superblock` is part of the original GGUF provenance.
                 * Keep the latter intact so a later CPU-to-GPU promotion can
                 * distinguish, for example, Q8_0 from Q8_K even though both
                 * execute from normalized signed-byte payloads.
                 */
                cpu.is_superblock = gpu.is_superblock;
            }
            // Strides come from the CPU encoding's single source of truth.
            cpu.data_stride = static_cast<int>(
                cpu::native_vnni::preparedDataStride(cpu.encoding));
            cpu.interleaved_block_stride =
                cpu::native_vnni::preparedInterleavedBlockStride(
                    cpu.encoding, cpu.is_asymmetric);
        }

        /**
         * @brief Validate logical transfer identity shared by both builders.
         * @param residency_epoch Immutable placement epoch.
         * @param layer_idx Transformer layer identity.
         * @param expert_id Routed-expert identity.
         * @param projection Projection within the expert MLP.
         * @throws std::invalid_argument if any field is incomplete.
         */
        void validateStreamIdentity(
            uint64_t residency_epoch,
            int layer_idx,
            int expert_id,
            ExpertTierWeightProjection projection)
        {
            if (residency_epoch == 0 || layer_idx < 0 || expert_id < 0 ||
                static_cast<uint8_t>(projection) >
                    static_cast<uint8_t>(ExpertTierWeightProjection::Down))
            {
                throw std::invalid_argument("Tier stream identity is incomplete");
            }
        }

        /**
         * @brief Compare a caller descriptor with its canonical catalog entry.
         * @param lhs Caller-supplied source descriptor.
         * @param rhs Catalog-owned descriptor with the same source identity.
         * @return Whether every physical execution field is identical.
         */
        bool sameFormat(
            const NativeVnniFormatInfo &lhs,
            const NativeVnniFormatInfo &rhs) noexcept
        {
            return lhs.codebook_id == rhs.codebook_id &&
                   lhs.payload_bytes == rhs.payload_bytes &&
                   lhs.is_asymmetric == rhs.is_asymmetric &&
                   lhs.is_superblock == rhs.is_superblock &&
                   lhs.has_emins == rhs.has_emins &&
                   lhs.max_abs_factor == rhs.max_abs_factor;
        }

        /**
         * @brief Derive chunk totals, authenticate, and validate a manifest.
         * @param manifest Partially populated endpoint/layout contract.
         * @param maximum_units_per_chunk Persistent staging capacity.
         * @return Sealed manifest suitable for transport and kernel launch.
         * @throws std::invalid_argument when counts overflow or the complete
         *         direction-aware contract is inconsistent.
         */
        ExpertTierWeightStreamManifest sealManifest(
            ExpertTierWeightStreamManifest manifest,
            uint32_t maximum_units_per_chunk)
        {
            const uint64_t units =
                static_cast<uint64_t>(manifest.N_padded / 64) *
                static_cast<uint64_t>(manifest.blocks_per_row);
            if (units == 0 ||
                units > std::numeric_limits<uint32_t>::max() ||
                maximum_units_per_chunk == 0 ||
                maximum_units_per_chunk > units)
            {
                throw std::invalid_argument(
                    "Tier stream chunk capacity must be within the complete unit count");
            }
            manifest.unit_count = static_cast<uint32_t>(units);
            manifest.maximum_units_per_chunk = maximum_units_per_chunk;
            if (!checkedMultiply(
                    units,
                    static_cast<uint64_t>(manifest.cpu_block_stride),
                    manifest.total_stream_bytes))
            {
                throw std::invalid_argument("Tier stream byte count overflows uint64");
            }
            // Hash only after every semantic field has its final value.
            manifest.manifest_hash = manifest.computedHash();
            std::string error;
            if (!manifest.valid(&error))
                throw std::invalid_argument(error);
            return manifest;
        }
    } // namespace

    /**
     * @brief Fingerprint one complete chunk payload for corruption detection.
     * @param bytes Contiguous CPU-format bytes carried by the chunk.
     * @return Stable 64-bit FNV-1a payload hash.
     */
    uint64_t expertTierWeightBytesHash(
        std::span<const uint8_t> bytes) noexcept
    {
        return fnv1a64(bytes.data(), bytes.size());
    }

    /**
     * @brief Recompute the manifest identity from every semantic field.
     * @return Stable 64-bit FNV-1a hash, excluding `manifest_hash` itself.
     *
     * Field order is part of ABI version 2. New fields must be appended with a
     * version bump; silently reordering fields would invalidate peers' hashes.
     */
    uint64_t ExpertTierWeightStreamManifest::computedHash() const noexcept
    {
        uint64_t hash = kFNV1a64OffsetBasis;
        // Protocol and direction prevent one stream type being replayed as another.
        hash = hashScalar(hash, magic);
        hash = hashScalar(hash, version);
        hash = hashScalar(hash, static_cast<uint8_t>(source_packing));
        hash = hashScalar(hash, static_cast<uint8_t>(destination_packing));
        // Residency identity prevents a late chunk entering a newer placement.
        hash = hashScalar(hash, residency_epoch);
        hash = hashScalar(hash, layer_idx);
        hash = hashScalar(hash, expert_id);
        hash = hashScalar(hash, static_cast<uint8_t>(projection));
        // Geometry and both physical layouts determine every byte address.
        hash = hashScalar(hash, N);
        hash = hashScalar(hash, K);
        hash = hashScalar(hash, N_padded);
        hash = hashScalar(hash, blocks_per_row);
        hash = hashScalar(hash, cpu_codebook_id);
        hash = hashScalar(hash, gpu_source_codebook_id);
        hash = hashScalar(hash, gpu_source_is_superblock);
        hash = hashScalar(hash, gpu_codebook_id);
        hash = hashScalar(hash, gpu_payload_bytes_per_block);
        hash = hashScalar(hash, cpu_is_asymmetric);
        hash = hashScalar(hash, cpu_is_superblock);
        hash = hashScalar(hash, gpu_is_asymmetric);
        hash = hashScalar(hash, gpu_has_emins);
        hash = hashScalar(hash, static_cast<uint8_t>(cpu_encoding));
        hash = hashScalar(hash, cpu_data_stride);
        hash = hashScalar(hash, cpu_block_stride);
        // Chunk capacity belongs to the allocation/capture identity as well.
        hash = hashScalar(hash, unit_count);
        hash = hashScalar(hash, maximum_units_per_chunk);
        hash = hashScalar(hash, total_stream_bytes);
        return hash;
    }

    /**
     * @brief Validate a manifest as an internally complete wire contract.
     * @param error Optional exact rejection diagnostic.
     * @return `true` only when identity, direction, geometry, formats, sizes,
     *         and the authenticated hash are mutually consistent.
     */
    bool ExpertTierWeightStreamManifest::valid(
        std::string *error) const noexcept
    {
        // Fail closed before interpreting any version-dependent field.
        if (magic != kMagic || version != kVersion)
            return reject(error, "Tier stream manifest magic/version mismatch");
        // The streaming ABI exists only for one CPU/GPU format boundary.
        const bool cpu_to_gpu =
            source_packing == ExpertTierWeightPacking::CpuNativeVnniInterleaved &&
            destination_packing == ExpertTierWeightPacking::GpuSeparatedNativeVnni;
        const bool gpu_to_cpu =
            source_packing == ExpertTierWeightPacking::GpuSeparatedNativeVnni &&
            destination_packing == ExpertTierWeightPacking::CpuNativeVnniInterleaved;
        if (!cpu_to_gpu && !gpu_to_cpu)
            return reject(error, "Tier stream must cross exactly one CPU/GPU packing boundary");
        // Projection identity is needed because gate/up/down may share geometry.
        if (layer_idx < 0 || expert_id < 0 ||
            static_cast<uint8_t>(projection) >
                static_cast<uint8_t>(ExpertTierWeightProjection::Down))
        {
            return reject(error, "Tier stream identity is invalid");
        }
        // A unit is indivisible: exactly 64 N columns by one 32-value K block.
        if (N <= 0 || K <= 0 || N_padded < N || (N_padded % 64) != 0 ||
            (K % 32) != 0 || blocks_per_row != K / 32)
        {
            return reject(error, "Tier stream geometry is invalid");
        }
        if (!cpu::native_vnni::isKnownPreparedEncoding(cpu_encoding))
            return reject(error, "Tier stream CPU encoding is invalid");
        // Re-derive strides rather than accepting redundant wire fields on trust.
        const uint32_t expected_data_stride =
            cpu::native_vnni::preparedDataStride(cpu_encoding);
        const uint32_t expected_block_stride = static_cast<uint32_t>(
            cpu::native_vnni::preparedInterleavedBlockStride(
                cpu_encoding, cpu_is_asymmetric != 0));
        if (cpu_data_stride != expected_data_stride ||
            cpu_block_stride != expected_block_stride)
        {
            return reject(error, "Tier stream CPU stride is inconsistent with its encoding");
        }
        // Exact totals let receivers allocate once before accepting any bytes.
        const uint64_t expected_units =
            static_cast<uint64_t>(N_padded / 64) *
            static_cast<uint64_t>(blocks_per_row);
        const uint64_t expected_bytes =
            expected_units * static_cast<uint64_t>(cpu_block_stride);
        if (expected_units == 0 ||
            expected_units > std::numeric_limits<uint32_t>::max() ||
            unit_count != expected_units || total_stream_bytes != expected_bytes ||
            maximum_units_per_chunk == 0 ||
            maximum_units_per_chunk > unit_count)
        {
            return reject(error, "Tier stream unit/chunk byte totals are invalid");
        }

        // The allocation-free kernel contract is the single format-relation
        // authority for both directions and every cataloged source format.
        if (!deviceLayout().valid())
            return reject(error, "Tier stream endpoint formats are inconsistent");
        // Hash validation is last so diagnostics identify structural faults first.
        if (manifest_hash == 0 || manifest_hash != computedHash())
            return reject(error, "Tier stream manifest hash mismatch");
        return true;
    }

    /**
     * @brief Project a validated wire manifest into allocation-free kernel data.
     * @return Scalar layout consumed by CUDA and HIP conversion launchers.
     *
     * Transport identity and hashes intentionally stay on the host; kernels
     * receive only fields that participate in address or format arithmetic.
     */
    ExpertTierWeightDeviceLayout
    ExpertTierWeightStreamManifest::deviceLayout() const noexcept
    {
        return ExpertTierWeightDeviceLayout{
            .direction =
                source_packing ==
                        ExpertTierWeightPacking::GpuSeparatedNativeVnni
                    ? ExpertTierWeightConversionDirection::GpuToCpu
                    : ExpertTierWeightConversionDirection::CpuToGpu,
            .N = N,
            .K = K,
            .N_padded = N_padded,
            .blocks_per_row = blocks_per_row,
            .cpu_codebook_id = cpu_codebook_id,
            .gpu_source_codebook_id = gpu_source_codebook_id,
            .gpu_source_is_superblock = gpu_source_is_superblock,
            .gpu_codebook_id = gpu_codebook_id,
            .gpu_payload_bytes_per_block = gpu_payload_bytes_per_block,
            .cpu_is_asymmetric = cpu_is_asymmetric,
            .cpu_is_superblock = cpu_is_superblock,
            .gpu_is_asymmetric = gpu_is_asymmetric,
            .gpu_has_emins = gpu_has_emins,
            .cpu_encoding = cpu_encoding,
            .cpu_data_stride = cpu_data_stride,
            .cpu_block_stride = cpu_block_stride,
            .unit_count = unit_count,
            .maximum_units_per_chunk = maximum_units_per_chunk,
        };
    }

    /**
     * @brief Bind a strict receive state machine to its final CPU allocation.
     * @param manifest Immutable stream contract.
     * @param final_bytes Exact-size destination in its final NUMA allocation.
     *
     * Construction records invalid input rather than throwing so `accept()` can
     * reject all data through the same diagnostic path without partial writes.
     */
    ExpertTierWeightStreamReceiver::ExpertTierWeightStreamReceiver(
        ExpertTierWeightStreamManifest manifest,
        std::span<uint8_t> final_bytes)
        : manifest_(manifest), final_bytes_(final_bytes)
    {
        // An oversized span is rejected: offsets are defined by this manifest alone.
        construction_valid_ =
            manifest_.valid(nullptr) &&
            final_bytes_.size() == manifest_.total_stream_bytes;
    }

    /**
     * @brief Validate and atomically commit the next ordered stream chunk.
     * @param header Authenticated sequence/range metadata.
     * @param payload Complete CPU-format units for the declared range.
     * @param error Optional exact rejection diagnostic.
     * @return `true` only after bytes are copied and receiver state advances.
     *
     * Every rejection occurs before `memcpy`, so retry is deliberately absent
     * and callers can prove that corrupt or reordered input never mutates the
     * final expert allocation.
     */
    bool ExpertTierWeightStreamReceiver::accept(
        const ExpertTierWeightStreamChunkHeader &header,
        std::span<const uint8_t> payload,
        std::string *error) noexcept
    {
        // Lifecycle checks make complete and aborted states terminal.
        if (!construction_valid_)
            return reject(error, "Tier stream receiver was constructed with an invalid final buffer");
        if (aborted_)
            return reject(error, "Tier stream receiver is aborted");
        if (complete_)
            return reject(error, "Tier stream receiver already consumed its final chunk");
        // Epoch/layout identity must match before sequence arithmetic is trusted.
        if (header.manifest_hash != manifest_.manifest_hash)
            return reject(error, "Tier stream chunk references a different manifest");
        // Requiring both sequence and first unit detects gaps and duplicates.
        if (header.sequence != next_sequence_ ||
            header.first_unit != received_units_)
        {
            return reject(error, "Tier stream chunk is missing, duplicate, or out of order");
        }
        if (header.unit_count == 0 ||
            header.unit_count > manifest_.maximum_units_per_chunk ||
            header.unit_count > manifest_.unit_count - received_units_)
        {
            return reject(error, "Tier stream chunk unit range exceeds the manifest");
        }
        // Chunks cannot contain a partial CPU unit.
        const uint64_t expected_payload_bytes =
            static_cast<uint64_t>(header.unit_count) *
            static_cast<uint64_t>(manifest_.cpu_block_stride);
        if (expected_payload_bytes > std::numeric_limits<uint32_t>::max() ||
            header.payload_bytes != expected_payload_bytes ||
            payload.size() != expected_payload_bytes)
        {
            return reject(error, "Tier stream chunk byte count is invalid");
        }
        // Exactly one final marker is legal, on the chunk that fills the stream.
        const bool reaches_end =
            header.first_unit + header.unit_count == manifest_.unit_count;
        if ((header.final_chunk != 0) != reaches_end)
            return reject(error, "Tier stream final marker does not match the final unit range");
        if (header.payload_hash == 0 ||
            header.payload_hash != expertTierWeightBytesHash(payload))
        {
            return reject(error, "Tier stream chunk payload hash mismatch");
        }

        // All checks passed. Copy directly into final ownership, then publish state.
        const size_t destination_offset =
            static_cast<size_t>(header.first_unit) *
            static_cast<size_t>(manifest_.cpu_block_stride);
        std::memcpy(
            final_bytes_.data() + destination_offset,
            payload.data(),
            payload.size());
        received_units_ += header.unit_count;
        ++next_sequence_;
        complete_ = reaches_end;
        return true;
    }

    /**
     * @brief Validate host oracle storage against common GPU packed geometry.
     * @param error Optional exact rejection diagnostic.
     * @return `true` when every separated region has its exact required size.
     */
    bool HostGpuExpertPackedProjection::valid(
        std::string *error) const noexcept
    {
        // GPU matrices are block-major: [K block][logical N column].
        if (N <= 0 || K <= 0 || (K % 32) != 0 ||
            blocks_per_row != static_cast<uint32_t>(K / 32) ||
            payload_bytes_per_block == 0)
        {
            return reject(error, "Host GPU packed geometry is invalid");
        }
        const size_t blocks =
            static_cast<size_t>(N) * static_cast<size_t>(blocks_per_row);
        if (payload.size() != blocks * payload_bytes_per_block ||
            scales.size() != blocks ||
            mins.size() != (is_asymmetric ? blocks : 0) ||
            emins.size() != (has_emins ? blocks : 0))
        {
            return reject(error, "Host GPU packed region sizes do not match the descriptor");
        }
        // Source codebook plus superblock bit names exactly one of the 21
        // source tensor formats.  This catches ambiguous shared codebooks.
        const NativeVnniFormatInfo *source_format =
            native_vnni_formats::forSourceIdentity(
                source_codebook_id, is_superblock);
        if (source_format == nullptr)
            return reject(error, "Host GPU source format identity is not cataloged");

        const bool exact_source_representation =
            codebook_id == canonicalDeviceVnniCodebookId(
                               source_format->codebook_id) &&
            payload_bytes_per_block == source_format->payload_bytes &&
            is_asymmetric == source_format->is_asymmetric &&
            has_emins == source_format->has_emins;
        if (exact_source_representation)
            return true;

        // A CPU promotion may only produce the two explicit normalized INT8
        // destinations; it may never pretend to reconstruct a compact source.
        const bool normalized_symmetric =
            codebook_id == 19 && payload_bytes_per_block == 32 &&
            !is_asymmetric && !has_emins;
        const bool normalized_asymmetric =
            codebook_id == kNativeVnniExpandedInt8MinCodebook &&
            payload_bytes_per_block == 32 && is_asymmetric && !has_emins;
        if (normalized_symmetric || normalized_asymmetric)
            return true;
        return reject(error, "Host GPU endpoint format is inconsistent with source provenance");
    }

    /**
     * @brief Construct a direction-complete GPU-to-CPU projection manifest.
     * @param source_format Cataloged tensor format represented by GPU arrays.
     * @param N Logical output-column count.
     * @param K Logical reduction dimension.
     * @param residency_epoch Placement epoch protected by the transfer.
     * @param layer_idx Logical transformer layer.
     * @param expert_id Logical routed-expert identity.
     * @param projection Gate, up, or down projection identity.
     * @param maximum_units_per_chunk Persistent staging capacity in units.
     * @return Fully validated version-2 manifest covering compact, dual-scale,
     *         embedded-minimum, and normalized source formats.
     */
    ExpertTierWeightStreamManifest
    makeGpuToCpuExpertTierWeightStreamManifest(
        const NativeVnniFormatInfo &source_format,
        int N,
        int K,
        uint64_t residency_epoch,
        int layer_idx,
        int expert_id,
        ExpertTierWeightProjection projection,
        uint32_t maximum_units_per_chunk)
    {
        validateStreamIdentity(
            residency_epoch, layer_idx, expert_id, projection);
        if (N <= 0 || K <= 0 || (K % 32) != 0)
            throw std::invalid_argument("Tier stream GPU source geometry is invalid");

        const NativeVnniFormatInfo *catalog_format =
            native_vnni_formats::forSourceIdentity(
                source_format.codebook_id, source_format.is_superblock);
        if (catalog_format == nullptr ||
            !sameFormat(source_format, *catalog_format))
        {
            throw std::invalid_argument(
                "Tier stream GPU source descriptor is not a canonical catalog format");
        }

        const CPUEncoding cpu_encoding =
            cpu::native_vnni::preparedEncodingForCodebook(
                source_format.codebook_id);
        ExpertTierWeightStreamManifest manifest;
        manifest.source_packing =
            ExpertTierWeightPacking::GpuSeparatedNativeVnni;
        manifest.destination_packing =
            ExpertTierWeightPacking::CpuNativeVnniInterleaved;
        manifest.residency_epoch = residency_epoch;
        manifest.layer_idx = layer_idx;
        manifest.expert_id = expert_id;
        manifest.projection = projection;
        manifest.N = N;
        manifest.K = K;
        manifest.N_padded = (N + 63) / 64 * 64;
        manifest.blocks_per_row = K / 32;
        // Preserve original source identity even when Q8_1/Q8_K execute as 19.
        manifest.cpu_codebook_id = source_format.codebook_id;
        manifest.gpu_source_codebook_id = source_format.codebook_id;
        manifest.gpu_source_is_superblock =
            source_format.is_superblock ? 1u : 0u;
        manifest.gpu_codebook_id =
            canonicalDeviceVnniCodebookId(source_format.codebook_id);
        manifest.gpu_payload_bytes_per_block =
            static_cast<uint8_t>(source_format.payload_bytes);
        manifest.cpu_is_asymmetric =
            source_format.is_asymmetric ? 1u : 0u;
        manifest.cpu_is_superblock =
            source_format.is_superblock ? 1u : 0u;
        manifest.gpu_is_asymmetric =
            source_format.is_asymmetric ? 1u : 0u;
        manifest.gpu_has_emins = source_format.has_emins ? 1u : 0u;
        manifest.cpu_encoding = cpu_encoding;
        manifest.cpu_data_stride =
            cpu::native_vnni::preparedDataStride(cpu_encoding);
        manifest.cpu_block_stride = static_cast<uint32_t>(
            cpu::native_vnni::preparedInterleavedBlockStride(
                cpu_encoding, source_format.is_asymmetric));
        return sealManifest(manifest, maximum_units_per_chunk);
    }

    ExpertTierWeightStreamManifest
    makeGpuToCpuExpertTierWeightStreamManifest(
        const NativeVnniFormatInfo &source_format,
        const GpuExpertPackedDescriptor &gpu_source,
        uint64_t residency_epoch,
        int layer_idx,
        int expert_id,
        ExpertTierWeightProjection projection,
        uint32_t maximum_units_per_chunk)
    {
        if (!gpu_source.valid())
            throw std::invalid_argument(
                "Tier stream live GPU source descriptor is invalid");

        auto manifest = makeGpuToCpuExpertTierWeightStreamManifest(
            source_format,
            gpu_source.n,
            gpu_source.k,
            residency_epoch,
            layer_idx,
            expert_id,
            projection,
            maximum_units_per_chunk);

        const bool canonical_source =
            gpu_source.codebook_id ==
                canonicalDeviceVnniCodebookId(source_format.codebook_id) &&
            gpu_source.payload_bytes_per_block ==
                source_format.payload_bytes &&
            gpu_source.is_asymmetric == source_format.is_asymmetric &&
            gpu_source.has_emins == source_format.has_emins;
        const bool normalized_source =
            manifest.cpu_encoding == CPUEncoding::ExpandedInt8 &&
            gpu_source.payload_bytes_per_block == 32 &&
            !gpu_source.has_emins &&
            gpu_source.is_asymmetric == source_format.is_asymmetric &&
            gpu_source.codebook_id ==
                (source_format.is_asymmetric
                     ? kNativeVnniExpandedInt8MinCodebook
                     : static_cast<std::uint8_t>(19));
        if (!canonical_source && !normalized_source)
        {
            throw std::invalid_argument(
                "Tier stream live GPU source is neither canonical nor its exact CPU-promotion normalization");
        }

        /*
         * CPU fields remain derived from immutable source provenance. Only the
         * accelerator-facing fields describe the current physical bytes. This
         * is what makes CPU -> GPU -> CPU cycles preserve the same final CPU
         * representation even after compact source bits have been normalized.
         */
        manifest.gpu_codebook_id = gpu_source.codebook_id;
        manifest.gpu_payload_bytes_per_block =
            gpu_source.payload_bytes_per_block;
        manifest.gpu_is_asymmetric =
            static_cast<std::uint8_t>(gpu_source.is_asymmetric);
        manifest.gpu_has_emins =
            static_cast<std::uint8_t>(gpu_source.has_emins);
        return sealManifest(manifest, maximum_units_per_chunk);
    }

    /**
     * @brief Construct a direction-complete CPU-to-GPU projection manifest.
     * @param cpu_weights Final CPU execution bytes to promote.
     * @param residency_epoch Placement epoch protected by the transfer.
     * @param layer_idx Logical transformer layer.
     * @param expert_id Logical routed-expert identity.
     * @param projection Gate, up, or down projection identity.
     * @param maximum_units_per_chunk Persistent staging capacity in units.
     * @return Fully validated version-2 manifest with an explicit normalized
     *         GPU destination when compact source bits are no longer present.
     */
    ExpertTierWeightStreamManifest
    makeCpuToGpuExpertTierWeightStreamManifest(
        const CPUWeights &cpu_weights,
        uint64_t residency_epoch,
        int layer_idx,
        int expert_id,
        ExpertTierWeightProjection projection,
        uint32_t maximum_units_per_chunk)
    {
        validateStreamIdentity(
            residency_epoch, layer_idx, expert_id, projection);
        std::string error;
        if (!validateCpuWeights(cpu_weights, &error))
            throw std::invalid_argument(error);

        uint8_t gpu_codebook = 0;
        uint8_t payload_bytes = 0;
        bool gpu_asym = false;
        bool gpu_emins = false;
        if (!gpuFormatForCpu(
                cpu_weights,
                gpu_codebook,
                payload_bytes,
                gpu_asym,
                gpu_emins,
                &error))
        {
            throw std::invalid_argument(error);
        }

        ExpertTierWeightStreamManifest manifest;
        manifest.source_packing =
            ExpertTierWeightPacking::CpuNativeVnniInterleaved;
        manifest.destination_packing =
            ExpertTierWeightPacking::GpuSeparatedNativeVnni;
        manifest.residency_epoch = residency_epoch;
        manifest.layer_idx = layer_idx;
        manifest.expert_id = expert_id;
        manifest.projection = projection;
        manifest.N = cpu_weights.N;
        manifest.K = cpu_weights.K;
        manifest.N_padded = cpu_weights.N_padded;
        manifest.blocks_per_row = cpu_weights.blocks_per_row;
        manifest.cpu_codebook_id = cpu_weights.codebook_id;
        manifest.gpu_source_codebook_id = cpu_weights.codebook_id;
        manifest.gpu_source_is_superblock =
            cpu_weights.is_superblock ? 1u : 0u;
        manifest.gpu_codebook_id = gpu_codebook;
        manifest.gpu_payload_bytes_per_block = payload_bytes;
        manifest.cpu_is_asymmetric = cpu_weights.is_asymmetric ? 1u : 0u;
        manifest.cpu_is_superblock = cpu_weights.is_superblock ? 1u : 0u;
        manifest.gpu_is_asymmetric = gpu_asym ? 1u : 0u;
        manifest.gpu_has_emins = gpu_emins ? 1u : 0u;
        manifest.cpu_encoding = cpu_weights.encoding;
        manifest.cpu_data_stride =
            static_cast<uint32_t>(cpu_weights.data_stride);
        manifest.cpu_block_stride =
            static_cast<uint32_t>(cpu_weights.interleaved_block_stride);
        return sealManifest(manifest, maximum_units_per_chunk);
    }

    /**
     * @brief Deinterleave final CPU bytes into the common GPU representation.
     * @param cpu Valid prepared CPU projection.
     * @param gpu Receives a newly materialized separated host oracle.
     * @param error Optional exact rejection diagnostic.
     * @return `true` when all logical GPU blocks were reconstructed exactly.
     *
     * This is deliberately a clarity-first oracle. Backend kernels must match
     * every payload and FP16 metadata bit; its speed is irrelevant to runtime.
     */
    bool cpuToGpuExpertPackedReference(
        const CPUWeights &cpu,
        HostGpuExpertPackedProjection &gpu,
        std::string *error) noexcept
    {
        // Reject inconsistent strides or a non-streamable CPU format up front.
        if (!validateCpuWeights(cpu, error))
            return false;

        uint8_t gpu_codebook = 0;
        uint8_t payload_bytes = 0;
        bool gpu_asym = false;
        bool gpu_emins = false;
        if (!gpuFormatForCpu(
                cpu,
                gpu_codebook,
                payload_bytes,
                gpu_asym,
                gpu_emins,
                error))
        {
            return false;
        }

        // GPU arrays omit CPU's padded N columns and store K blocks major.
        HostGpuExpertPackedProjection output;
        output.N = cpu.N;
        output.K = cpu.K;
        output.blocks_per_row = static_cast<uint32_t>(cpu.blocks_per_row);
        output.source_codebook_id = cpu.codebook_id;
        output.codebook_id = gpu_codebook;
        output.payload_bytes_per_block = payload_bytes;
        output.is_asymmetric = gpu_asym;
        output.is_superblock = cpu.is_superblock;
        output.has_emins = gpu_emins;
        const size_t block_count =
            static_cast<size_t>(output.N) * output.blocks_per_row;
        output.payload.resize(block_count * payload_bytes);
        output.scales.resize(block_count);
        if (gpu_asym)
            output.mins.resize(block_count);

        // Visit in GPU block-major order so `linear` matches production kernels.
        for (int kb = 0; kb < cpu.blocks_per_row; ++kb)
        {
            for (int n = 0; n < cpu.N; ++n)
            {
                const size_t linear =
                    static_cast<size_t>(kb) * cpu.N + n;
                uint8_t *destination =
                    output.payload.data() + linear * payload_bytes;
                const size_t unit = cpuUnitOffset(cpu, n, kb);

                if (cpu.encoding == CPUEncoding::NibbleLUT)
                {
                    // Compact nibbles are only transposed; quantized bits do not change.
                    for (int value = 0; value < 16; ++value)
                    {
                        destination[value] =
                            cpu.native_interleaved[
                                interleavedValueOffset(cpu, n, kb, value)];
                    }
                    // Compensation is CPU-only derived metadata and is not returned.
                    std::memcpy(
                        &output.scales[linear],
                        cpu.native_interleaved.data() +
                            unit + cpu.data_stride + 128u +
                            static_cast<size_t>(n % 64) * sizeof(uint16_t),
                        sizeof(uint16_t));
                    if (cpu.is_asymmetric)
                    {
                        std::memcpy(
                            &output.mins[linear],
                            cpu.native_interleaved.data() +
                                unit + cpu.data_stride + 256u +
                                static_cast<size_t>(n % 64) * sizeof(uint16_t),
                            sizeof(uint16_t));
                    }
                    continue;
                }

                if (cpu.encoding == CPUEncoding::ExpandedInt8)
                {
                    // Expanded signed bytes use the same VNNI lane transpose.
                    for (int value = 0; value < 32; ++value)
                    {
                        destination[value] =
                            cpu.native_interleaved[
                                interleavedValueOffset(cpu, n, kb, value)];
                    }
                    std::memcpy(
                        &output.scales[linear],
                        cpu.native_interleaved.data() +
                            unit + cpu.data_stride + 128u +
                            static_cast<size_t>(n % 64) * sizeof(uint16_t),
                        sizeof(uint16_t));
                    if (cpu.is_asymmetric)
                    {
                        std::memcpy(
                            &output.mins[linear],
                            cpu.native_interleaved.data() +
                                unit + cpu.data_stride + 256u +
                                static_cast<size_t>(n % 64) * sizeof(uint16_t),
                            sizeof(uint16_t));
                    }
                    continue;
                }

                // Q6 low nibbles occupy the ordinary first four VNNI groups.
                for (int value = 0; value < 16; ++value)
                {
                    destination[value] =
                        cpu.native_interleaved[
                            interleavedValueOffset(cpu, n, kb, value)];
                }
                const int local_column = n % 64;
                const int zmm = local_column / 16;
                const int lane = local_column % 16;
                // CPU stores the two high bits as two 64-bit planes per
                // (four-value group, 16-column ZMM). Rebuild one byte per group.
                for (int group = 0; group < 8; ++group)
                {
                    const uint8_t *planes =
                        cpu.native_interleaved.data() + unit + 1024u +
                        static_cast<size_t>(group) * 64u +
                        static_cast<size_t>(zmm) * 16u;
                    const uint64_t low = loadU64(planes);
                    const uint64_t high = loadU64(planes + sizeof(uint64_t));
                    uint8_t packed_high = 0;
                    for (int index = 0; index < 4; ++index)
                    {
                        // Each lane owns four consecutive plane bits.
                        const uint64_t bit =
                            static_cast<uint64_t>(lane * 4 + index);
                        const uint8_t high_part = static_cast<uint8_t>(
                            ((low >> bit) & 1u) |
                            (((high >> bit) & 1u) << 1u));
                        packed_high |= static_cast<uint8_t>(
                            high_part << (index * 2));
                    }
                    destination[16 + group] = packed_high;
                }
                // Q6's `mins` region carries the high-half scale, not a minimum.
                std::memcpy(
                    &output.scales[linear],
                    cpu.native_interleaved.data() + unit + 1536u +
                        static_cast<size_t>(local_column) * sizeof(uint16_t),
                    sizeof(uint16_t));
                std::memcpy(
                    &output.mins[linear],
                    cpu.native_interleaved.data() + unit + 1664u +
                        static_cast<size_t>(local_column) * sizeof(uint16_t),
                    sizeof(uint16_t));
            }
        }

        // Validate the assembled oracle before publishing it to the caller.
        if (!output.valid(error))
            return false;
        gpu = std::move(output);
        return true;
    }

    /**
     * @brief Pack common GPU arrays into final CPU NativeVNNI stream bytes.
     * @param gpu Valid separated host representation.
     * @param cpu Receives complete padded CPU execution bytes and metadata.
     * @param error Optional exact rejection diagnostic.
     * @return `true` when the GPU format has an exact CPU execution mapping.
     *
     * GPU-to-CPU device kernels use this function as their byte oracle. It
     * includes CPU compensation vectors and deterministic zeros for padded N.
     */
    bool gpuToCpuExpertPackedReference(
        const HostGpuExpertPackedProjection &gpu,
        CPUWeights &cpu,
        std::string *error) noexcept
    {
        // Region and codebook validation prevents out-of-bounds source reads.
        if (!gpu.valid(error))
            return false;
        if (!isNibbleCodebook(gpu.codebook_id) && gpu.codebook_id != 8 &&
            gpu.codebook_id != 19 &&
            gpu.codebook_id != kNativeVnniExpandedInt8MinCodebook)
        {
            return reject(
                error,
                "Host reference requires production source-format transcoding for this GPU codebook");
        }

        CPUWeights output;
        configureCpuFromGpu(gpu, output);
        const size_t unit_count =
            static_cast<size_t>(output.N_padded / 64) *
            static_cast<size_t>(output.blocks_per_row);
        // Value-initialization fixes padding and reserved metadata bytes at zero.
        output.native_interleaved.resize(
            unit_count * static_cast<size_t>(output.interleaved_block_stride),
            0);

        // Traverse the GPU's [K block][N column] order.
        for (int kb = 0; kb < output.blocks_per_row; ++kb)
        {
            for (int n = 0; n < output.N; ++n)
            {
                const size_t linear =
                    static_cast<size_t>(kb) * output.N + n;
                const uint8_t *source =
                    gpu.payload.data() +
                    linear * gpu.payload_bytes_per_block;
                const size_t unit = cpuUnitOffset(output, n, kb);
                const int local_column = n % 64;

                if (output.encoding == CPUEncoding::NibbleLUT)
                {
                    // Scatter compact bytes into the four-value VNNI lane groups.
                    for (int value = 0; value < 16; ++value)
                    {
                        output.native_interleaved[
                            interleavedValueOffset(output, n, kb, value)] =
                            source[value];
                    }
                    // CPU dot-product correction consumes the sum of decoded weights.
                    int8_t decoded[32]{};
                    cpu::native_vnni::decode_native_block(
                        output.codebook_id, source, decoded);
                    int32_t sum = 0;
                    for (int value : decoded)
                        sum += value;
                    const int16_t compensation = static_cast<int16_t>(sum);
                    std::memcpy(
                        output.native_interleaved.data() +
                            unit + output.data_stride +
                            static_cast<size_t>(local_column) * sizeof(int16_t),
                        &compensation,
                        sizeof(compensation));
                    std::memcpy(
                        output.native_interleaved.data() +
                            unit + output.data_stride + 128u +
                            static_cast<size_t>(local_column) * sizeof(uint16_t),
                        &gpu.scales[linear],
                        sizeof(uint16_t));
                    if (output.is_asymmetric)
                    {
                        std::memcpy(
                            output.native_interleaved.data() +
                                unit + output.data_stride + 256u +
                                static_cast<size_t>(local_column) * sizeof(uint16_t),
                            &gpu.mins[linear],
                            sizeof(uint16_t));
                    }
                    continue;
                }

                if (output.encoding == CPUEncoding::ExpandedInt8)
                {
                    // Scatter signed values and derive the same correction inline.
                    int32_t sum = 0;
                    for (int value = 0; value < 32; ++value)
                    {
                        output.native_interleaved[
                            interleavedValueOffset(output, n, kb, value)] =
                            source[value];
                        sum += static_cast<int8_t>(source[value]);
                    }
                    const int16_t compensation = static_cast<int16_t>(sum);
                    std::memcpy(
                        output.native_interleaved.data() +
                            unit + output.data_stride +
                            static_cast<size_t>(local_column) * sizeof(int16_t),
                        &compensation,
                        sizeof(compensation));
                    std::memcpy(
                        output.native_interleaved.data() +
                            unit + output.data_stride + 128u +
                            static_cast<size_t>(local_column) * sizeof(uint16_t),
                        &gpu.scales[linear],
                        sizeof(uint16_t));
                    if (output.is_asymmetric)
                    {
                        std::memcpy(
                            output.native_interleaved.data() +
                                unit + output.data_stride + 256u +
                                static_cast<size_t>(local_column) * sizeof(uint16_t),
                            &gpu.mins[linear],
                            sizeof(uint16_t));
                    }
                    continue;
                }

                // Q6 low nibbles follow the ordinary VNNI byte transpose.
                for (int value = 0; value < 16; ++value)
                {
                    output.native_interleaved[
                        interleavedValueOffset(output, n, kb, value)] =
                        source[value];
                }
                const int zmm = local_column / 16;
                const int lane = local_column % 16;
                // Split each byte's eight two-bit high fields into the CPU's
                // low/high bit planes. A host read/modify/write is safe because
                // each logical column contributes disjoint bits.
                for (int group = 0; group < 8; ++group)
                {
                    const uint8_t packed_high = source[16 + group];
                    uint8_t *planes =
                        output.native_interleaved.data() + unit + 1024u +
                        static_cast<size_t>(group) * 64u +
                        static_cast<size_t>(zmm) * 16u;
                    uint64_t low = loadU64(planes);
                    uint64_t high = loadU64(planes + sizeof(uint64_t));
                    for (int index = 0; index < 4; ++index)
                    {
                        // Four values occupy four consecutive bits for this lane.
                        const uint64_t bit =
                            static_cast<uint64_t>(lane * 4 + index);
                        const uint8_t high_part = static_cast<uint8_t>(
                            (packed_high >> (index * 2)) & 0x03u);
                        low |= static_cast<uint64_t>(high_part & 1u) << bit;
                        high |= static_cast<uint64_t>(high_part >> 1u) << bit;
                    }
                    storeU64(planes, low);
                    storeU64(planes + sizeof(uint64_t), high);
                }
                // Preserve the low- and high-half Q6 scales bit-for-bit.
                std::memcpy(
                    output.native_interleaved.data() + unit + 1536u +
                        static_cast<size_t>(local_column) * sizeof(uint16_t),
                    &gpu.scales[linear],
                    sizeof(uint16_t));
                std::memcpy(
                    output.native_interleaved.data() + unit + 1664u +
                        static_cast<size_t>(local_column) * sizeof(uint16_t),
                    &gpu.mins[linear],
                    sizeof(uint16_t));
            }
        }

        // Publish only after every logical block and padded unit is complete.
        cpu = std::move(output);
        return true;
    }

} // namespace llaminar2
