/**
 * @file MoEOverlayRemoteProjectionProtocol.h
 * @brief Authenticated chunk protocol for cross-rank ExpertOverlay weights.
 *
 * A remote migration cannot infer the sender's physical representation from
 * GGUF provenance alone.  In particular, a GPU expert promoted through a CPU
 * tier may execute from normalized INT8 arrays while retaining its original
 * source-codebook identity.  This protocol therefore names both identities:
 * the immutable mathematical source and the exact CPU-interleaved or
 * GPU-separated bytes carried over the network.
 *
 * The protocol is device-free.  MPI, CUDA, ROCm, and CPU endpoint lanes use the
 * same canonical manifest and ordered chunk validator.  This lets unit tests
 * destroy the wire state machine without allocating a device, while production
 * lanes remain responsible for exact streams, events, and final storage.
 */

#pragma once

#include "ExpertWeightFormat.h"
#include "ExpertTierWeightStream.h"
#include "GPUExpertTransfer.h"
#include "MoEOverlayDistributedResidencyProtocol.h"
#include "backends/DeviceId.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <type_traits>

namespace llaminar2
{
    /** Physical representation transmitted between two MPI ranks. */
    enum class MoEOverlayRemoteProjectionPacking : std::uint8_t
    {
        CpuNativeVnniInterleaved = 1, ///< Live native CPU column planes; resident padding is excluded.
        GpuSeparatedNativeVnni = 2,   ///< Payload/scales/mins/emins blobs.
        ContiguousFloating = 3,       ///< One raw FP16/BF16/FP32 matrix blob.
    };

    /**
     * @brief Complete topology and transaction identity for one projection.
     *
     * `migration_index` is the canonical index in the globally fingerprinted
     * transaction.  It prevents two same-shaped experts in one wave from
     * sharing a network lane/tag identity accidentally.
     */
    struct MoEOverlayRemoteProjectionIdentity
    {
        std::uint64_t expected_epoch = 0;
        std::uint64_t candidate_epoch = 0;
        /** Executable plan identity; root-only economy evidence is excluded. */
        MoEOverlayResidencyExecutionFingerprint execution_fingerprint;
        std::uint64_t migration_index = 0;
        std::int32_t layer_idx = -1;
        std::int32_t expert_id = -1;
        ExpertTierWeightProjection projection =
            ExpertTierWeightProjection::Gate;
        std::uint8_t reserved_identity[3]{};
        std::int32_t source_participant = -1;
        std::int32_t destination_participant = -1;
        std::int32_t source_world_rank = -1;
        std::int32_t destination_world_rank = -1;
        DeviceId source_device = DeviceId::invalid();
        DeviceId destination_device = DeviceId::invalid();

        /** @return Whether epochs, ranks, devices, and projection are remote. */
        [[nodiscard]] bool valid() const noexcept;

        /** @brief Compare every semantic identity field. */
        bool operator==(
            const MoEOverlayRemoteProjectionIdentity &) const = default;
    };

    /** Number of independently addressed GPU separated-array regions. */
    inline constexpr std::size_t kMoEOverlayRemoteProjectionRegionCount = 4;

    /**
     * @brief Canonical format and byte contract sent before a projection.
     *
     * CPU wire payloads use region zero only.  GPU wire payloads use payload,
     * scales, optional minima, and optional effective-minima regions in that
     * order.  `manifest_hash` covers every semantic field and is repeated in
     * every chunk header.
     */
    struct MoEOverlayRemoteProjectionManifest
    {
        static constexpr std::uint32_t kMagic = 0x50524F4Du; // "MORP"
        // ABI 5: codebook-25 (MXFP4) CPU units carry BF16 power-of-two scale
        // words and GPU planes native E8M0 bytes (#26).
        static constexpr std::uint16_t kABIVersion = 5u;
        static constexpr std::size_t kWireBytes = 244u;

        std::uint32_t magic = kMagic;
        std::uint16_t abi_version = kABIVersion;
        MoEOverlayRemoteProjectionPacking packing =
            MoEOverlayRemoteProjectionPacking::CpuNativeVnniInterleaved;
        /** Mathematical representation family carried by this manifest. */
        ExpertWeightFormatKind format_kind =
            ExpertWeightFormatKind::Invalid;
        MoEOverlayRemoteProjectionIdentity identity;

        // Geometry is shared by CPU and GPU representations.
        std::int32_t N = 0;
        std::int32_t K = 0;
        std::int32_t N_padded = 0;
        std::int32_t blocks_per_row = 0;

        // Provenance and execution formats are intentionally separate.
        std::uint8_t source_codebook_id = 0;
        std::uint8_t source_is_superblock = 0;
        std::uint8_t cpu_codebook_id = 0;
        std::uint8_t gpu_codebook_id = 0;
        std::uint8_t gpu_payload_bytes_per_block = 0;
        std::uint8_t cpu_is_asymmetric = 0;
        std::uint8_t cpu_is_superblock = 0;
        std::uint8_t gpu_is_asymmetric = 0;
        std::uint8_t gpu_has_emins = 0;
        cpu::native_vnni::CPUNativeVNNIEncoding cpu_encoding =
            cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8;
        std::uint8_t reserved_format[2]{};

        std::uint32_t cpu_data_stride = 0;
        std::uint32_t cpu_block_stride = 0;
        /** Native columns per encoded wire unit; resident CPU tiles remain 64 columns. */
        std::uint32_t cpu_wire_columns_per_unit = 0;
        std::array<std::uint64_t, kMoEOverlayRemoteProjectionRegionCount>
            region_bytes{};
        std::uint64_t total_bytes = 0;
        std::uint32_t maximum_chunk_bytes = 0;
        std::uint32_t reserved_size = 0;
        std::uint64_t manifest_hash = 0;
        /** Original coordinate ownership retained across remote transfer envelopes. */
        MoEExpertProjectionOwnership::Projection source_projection{};
        MoEExpertProjectionOwnership::Projection destination_projection{};

        /**
         * @brief Bind native CPU endpoints to their checked original-coordinate overlap.
         * @param intersection Original source/destination allocations and nonempty overlap.
         * @return Resealed compact native wire contract using the existing send capacity.
         * @throws std::invalid_argument For a foreign source or unsupported native alignment.
         */
        [[nodiscard]] MoEOverlayRemoteProjectionManifest forCPUIntersection(
            const ExpertTierWeightProjectionIntersection& intersection) const;

        /**
         * @brief Bind floating source bytes to an original-coordinate overlap without conversion.
         * @param intersection Original admitted source/destination rectangles.
         * @return Resealed existing floating manifest, preserving source identity and byte precision.
         * @throws std::invalid_argument For a nonfloating source, empty overlap or source extent mismatch.
         */
        [[nodiscard]] MoEOverlayRemoteProjectionManifest forFloatingIntersection(
            const ExpertTierWeightProjectionIntersection& intersection) const;

        /** @return Canonical non-zero digest over every field except the digest. */
        [[nodiscard]] std::uint64_t computedHash() const noexcept;

        /**
         * @brief Validate remote identity, provenance, formats, and exact bytes.
         * @param error Optional exact rejection diagnostic.
         * @return True only for one direction-complete physical contract.
         */
        [[nodiscard]] bool valid(std::string *error = nullptr) const noexcept;

        /** @brief Compare every semantic, reserved, size, and digest field. */
        bool operator==(
            const MoEOverlayRemoteProjectionManifest &) const = default;

        /** @return Whether the wire carries final CPU execution units. */
        [[nodiscard]] bool carriesCpuBytes() const noexcept
        {
            return packing == MoEOverlayRemoteProjectionPacking::
                                  CpuNativeVnniInterleaved;
        }

        /** @return Whether the wire carries common separated GPU arrays. */
        [[nodiscard]] bool carriesGpuBytes() const noexcept
        {
            return packing == MoEOverlayRemoteProjectionPacking::
                                  GpuSeparatedNativeVnni;
        }

        /** @return Whether region zero is one contiguous floating matrix. */
        [[nodiscard]] bool carriesFloatingBytes() const noexcept
        {
            return packing == MoEOverlayRemoteProjectionPacking::
                                  ContiguousFloating;
        }
    };

    /**
     * @brief Construct a CPU-wire manifest for CPU-to-CPU movement.
     * @param identity Exact cross-rank transaction/projection identity.
     * @param source Immutable final CPU execution representation.
     * @param maximum_chunk_bytes Persistent network staging capacity.
     * @return Authenticated manifest with one final CPU byte region.
     */
    [[nodiscard]] MoEOverlayRemoteProjectionManifest
    makeMoEOverlayRemoteCpuProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const cpu::native_vnni::CPUNativeVNNIPackedWeights &source,
        std::uint32_t maximum_chunk_bytes);

    /**
     * @brief Author a receiver expectation from its genuine native destination root.
     * @param identity Original exact physical source/destination protocol identity.
     * @param destination Actual final CPU destination native storage metadata.
     * @param intersection Original checked source/destination rectangles.
     * @param maximum_chunk_bytes Existing admitted transport capacity.
     * @return Expected manifest only; grants no source payload or readiness authority.
     * @throws std::invalid_argument For foreign destination geometry/format or capacity.
     */
    [[nodiscard]] MoEOverlayRemoteProjectionManifest makeMoEOverlayRemoteCpuDestinationProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const cpu::native_vnni::CPUNativeVNNIPackedWeights &destination,
        const ExpertTierWeightProjectionIntersection &intersection, std::uint32_t maximum_chunk_bytes);

    /**
     * @brief Wrap a CPU/GPU conversion stream for cross-rank transport.
     * @param identity Exact cross-rank transaction/projection identity.
     * @param stream Valid GPU-to-CPU or CPU-to-GPU conversion manifest.
     * @param maximum_chunk_bytes Persistent network staging capacity in bytes.
     * @return Authenticated CPU-wire manifest retaining both endpoint layouts.
     */
    [[nodiscard]] MoEOverlayRemoteProjectionManifest
    makeMoEOverlayRemoteCpuProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const ExpertTierWeightStreamManifest &stream,
        std::uint32_t maximum_chunk_bytes);

    /**
     * @brief Construct a byte-preserving GPU-to-GPU remote blob manifest.
     * @param identity Exact cross-rank transaction/projection identity.
     * @param source Current separated GPU execution descriptor.
     * @param source_identity Original mathematical GGUF provenance.
     * @param maximum_chunk_bytes Persistent network staging capacity.
     * @return Authenticated four-region GPU blob contract.
     */
    [[nodiscard]] MoEOverlayRemoteProjectionManifest
    makeMoEOverlayRemoteGpuProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const GpuExpertPackedDescriptor &source,
        NativeVnniSourceIdentity source_identity,
        std::uint32_t maximum_chunk_bytes);

    /**
     * @brief Construct a byte-preserving contiguous floating manifest.
     * @param identity Exact cross-rank transaction/projection identity.
     * @param source Live FP16, BF16, or FP32 matrix retained by its GEMM.
     * @param maximum_chunk_bytes Persistent network staging capacity.
     * @return Authenticated one-region raw floating blob contract.
     *
     * CPU and GPU floating GEMMs share the same scalar representation. The
     * endpoint-specific source and destination classes determine whether each
     * chunk is read directly by MPI or crosses an exact auxiliary GPU stream.
     */
    [[nodiscard]] MoEOverlayRemoteProjectionManifest
    makeMoEOverlayRemoteFloatingProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const ContiguousFloatingPointWeightDescriptor &source,
        std::uint32_t maximum_chunk_bytes);

    /**
     * @brief Author only the expected floating wire geometry from a genuine final destination.
     * @param identity Original exact physical protocol identity.
     * @param destination Actual retained destination descriptor, CPU or GPU.
     * @param intersection Same checked original source/destination rectangles.
     * @param maximum_chunk_bytes Existing admitted transport capacity.
     * @return Receiver expectation, never a source-storage or native-readiness permit.
     */
    [[nodiscard]] MoEOverlayRemoteProjectionManifest makeMoEOverlayRemoteFloatingDestinationProjectionManifest(
        const MoEOverlayRemoteProjectionIdentity &identity,
        const ContiguousFloatingPointWeightDescriptor &destination,
        const ExpertTierWeightProjectionIntersection &intersection, std::uint32_t maximum_chunk_bytes);

    /** Select an actual CPU endpoint; no GPU geometry or execution authority is synthesized. */
    enum class MoEOverlayRemoteCPUStorageEndpoint { Source, Destination };
    /**
     * @brief Resolve native CPU root strides and compact wire groups from an authenticated manifest.
     * @param manifest Original projection and native format contract.
     * @param endpoint Actual CPU source or destination whose root is addressed.
     * @return Checked CPU-native root geometry, independent of any GPU representation.
     * @throws std::invalid_argument If the manifest/endpoint cannot describe a native CPU root.
     */
    [[nodiscard]] ExpertTierWeightCPUStorageLayout remoteCpuProjectionStorageLayout(
        const MoEOverlayRemoteProjectionManifest& manifest,MoEOverlayRemoteCPUStorageEndpoint endpoint);

    /**
     * @brief Recover the allocation-free repack layout from a CPU wire manifest.
     *
     * Cross-rank CPU/GPU transfers carry compact live native column planes.
     * Retained CPU allocation padding never enters the network. This preserves
     * every authenticated scalar while
     * selecting direction from the topology identity.  It rejects partial-unit
     * tile capacities so a destination GPU can repack each received live chunk
     * directly without buffering a complete projection.
     *
     * @param manifest Valid CPU-format remote projection with exactly one GPU
     *        endpoint.
     * @return Complete device-kernel layout whose chunk capacity is expressed
     *         in indivisible CPU NativeVNNI units.
     * @throws std::invalid_argument When topology, format, or chunk alignment
     *         cannot drive the production streaming repack kernel.
     */
    [[nodiscard]] ExpertTierWeightDeviceLayout remoteCpuProjectionDeviceLayout(
        const MoEOverlayRemoteProjectionManifest &manifest);

    /**
     * @brief Encode a manifest in canonical little-endian wire order.
     * @param manifest Valid authenticated manifest.
     * @param destination Exact @ref kWireBytes destination.
     * @param error Optional validation diagnostic.
     * @return True only when every destination byte was written.
     */
    bool encodeMoEOverlayRemoteProjectionManifest(
        const MoEOverlayRemoteProjectionManifest &manifest,
        std::span<std::uint8_t> destination,
        std::string *error = nullptr) noexcept;

    /**
     * @brief Decode and authenticate a canonical manifest packet.
     * @param packet Exact @ref kWireBytes packet.
     * @param manifest Receives a validated host representation.
     * @param error Optional malformed-packet diagnostic.
     * @return True only when size, reserved bytes, formats, and hash agree.
     */
    bool decodeMoEOverlayRemoteProjectionManifest(
        std::span<const std::uint8_t> packet,
        MoEOverlayRemoteProjectionManifest *manifest,
        std::string *error = nullptr) noexcept;

    /** Header authenticating one monotonically ordered bounded payload chunk. */
    struct MoEOverlayRemoteProjectionChunkHeader
    {
        static constexpr std::size_t kWireBytes = 48u;

        std::uint64_t manifest_hash = 0;
        std::uint64_t payload_hash = 0;
        std::uint64_t sequence = 0;
        std::uint8_t region = 0;
        std::uint8_t reserved_region[7]{};
        std::uint64_t region_offset = 0;
        std::uint32_t payload_bytes = 0;
        std::uint8_t final_chunk = 0;
        std::uint8_t reserved_final[3]{};

        /** @return Whether reserved bytes and scalar ranges are locally sane. */
        [[nodiscard]] bool structurallyValid() const noexcept;
    };

    static_assert(
        std::is_trivially_copyable_v<
            MoEOverlayRemoteProjectionChunkHeader>);

    /**
     * @brief Encode one chunk header in canonical little-endian order.
     * @param header Structurally valid header.
     * @param destination Exact @ref kWireBytes destination.
     * @param error Optional validation diagnostic.
     * @return True only when all bytes were written.
     */
    bool encodeMoEOverlayRemoteProjectionChunkHeader(
        const MoEOverlayRemoteProjectionChunkHeader &header,
        std::span<std::uint8_t> destination,
        std::string *error = nullptr) noexcept;

    /**
     * @brief Decode one canonical chunk header and reject reserved bits.
     * @param packet Exact @ref kWireBytes packet.
     * @param header Receives the structurally valid result.
     * @param error Optional malformed-packet diagnostic.
     * @return True only when the complete packet is accepted.
     */
    bool decodeMoEOverlayRemoteProjectionChunkHeader(
        std::span<const std::uint8_t> packet,
        MoEOverlayRemoteProjectionChunkHeader *header,
        std::string *error = nullptr) noexcept;

    /** Immutable header plus caller-owned bytes emitted by a chunk cursor. */
    struct MoEOverlayRemoteProjectionChunkView
    {
        MoEOverlayRemoteProjectionChunkHeader header;
        std::span<const std::uint8_t> payload;
    };

    /**
     * @brief Allocation-free sender cursor over final host-visible regions.
     *
     * GPU source lanes bind persistent pinned chunks after their exact D2H event.
     * CPU native roots gather into the caller's existing exclusive send staging;
     * floating roots expose only the next contiguous original row segment.
     * This convenience cursor is primarily for CPU paths and protocol tests.
     */
    class MoEOverlayRemoteProjectionChunkCursor final
    {
    public:
        /**
         * @brief Bind one manifest to exact immutable source regions.
         * @param manifest Valid authenticated projection contract.
         * @param regions Region spans whose sizes exactly match the manifest.
         * @throws std::invalid_argument For invalid or mismatched input.
         */
        MoEOverlayRemoteProjectionChunkCursor(
            MoEOverlayRemoteProjectionManifest manifest,
            std::array<
                std::span<const std::uint8_t>,
                kMoEOverlayRemoteProjectionRegionCount> regions);

        /**
         * @brief Take the next region-bounded authenticated chunk.
         * @param staging Existing exclusive admitted send-lane bytes for CPU native gathering.
         * @return Next view, or no value after every byte was emitted or invalid staging.
         */
        [[nodiscard]] std::optional<
            MoEOverlayRemoteProjectionChunkView> takeNext(std::span<std::uint8_t> staging = {}) noexcept;

        /** @return Whether the original CPU root needs an exclusive encoded send chunk. */
        [[nodiscard]] bool requiresCPUStaging() const noexcept {
            return manifest_.carriesCpuBytes() && manifest_.identity.source_device.is_cpu();
        }
        /** @return Whether every declared region byte has been emitted. */
        [[nodiscard]] bool complete() const noexcept;

    private:
        /** @brief Advance past empty or completely emitted regions. */
        void seekNextRegion() noexcept;

        MoEOverlayRemoteProjectionManifest manifest_;
        std::array<
            std::span<const std::uint8_t>,
            kMoEOverlayRemoteProjectionRegionCount> regions_;
        std::size_t region_ = 0;
        std::uint64_t region_offset_ = 0;
        std::uint64_t sequence_ = 0;
        std::uint64_t emitted_bytes_ = 0;
    };

    /**
     * @brief Strict ordered validator shared by CPU and device destinations.
     *
     * Validation advances only after the payload hash and exact next range are
     * accepted.  A destination GPU can therefore validate a pinned receive
     * chunk before enqueueing H2D/repack, without maintaining a full host mirror.
     */
    class MoEOverlayRemoteProjectionChunkValidator final
    {
    public:
        /**
         * @brief Start at region zero of one valid manifest.
         * @throws std::invalid_argument For an invalid manifest.
         */
        explicit MoEOverlayRemoteProjectionChunkValidator(
            MoEOverlayRemoteProjectionManifest manifest);

        /**
         * @brief Authenticate and consume exactly the next chunk identity.
         * @param header Sender-provided ordered range and payload digest.
         * @param payload Complete received bytes named by @p header.
         * @param error Optional exact rejection diagnostic.
         * @return True only when state advanced to the following range.
         */
        bool accept(
            const MoEOverlayRemoteProjectionChunkHeader &header,
            std::span<const std::uint8_t> payload,
            std::string *error = nullptr) noexcept;

        /** @brief Make the validator terminal after an external lane failure. */
        void abort() noexcept { aborted_ = true; }

        /** @return Whether the unique final chunk has been accepted. */
        [[nodiscard]] bool complete() const noexcept { return complete_; }

        /** @return Whether an external failure made further input illegal. */
        [[nodiscard]] bool aborted() const noexcept { return aborted_; }

    private:
        /** @brief Advance the expected cursor past empty regions. */
        void seekNextRegion() noexcept;

        MoEOverlayRemoteProjectionManifest manifest_;
        std::size_t region_ = 0;
        std::uint64_t region_offset_ = 0;
        std::uint64_t next_sequence_ = 0;
        std::uint64_t received_bytes_ = 0;
        bool complete_ = false;
        bool aborted_ = false;
    };
} // namespace llaminar2
