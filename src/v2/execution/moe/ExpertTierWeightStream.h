/**
 * @file ExpertTierWeightStream.h
 * @brief Typed streaming ABI for lossless ExpertOverlay CPU/GPU migration.
 *
 * Cross-backend expert movement streams the CPU execution representation in
 * complete 64-output-column by 32-K units. A source GPU produces those units
 * directly from its separated packed arrays; a destination GPU consumes the
 * same units into its separated arrays. GPU-to-GPU movement deliberately does
 * not use this ABI because accelerator packed arrays are already compatible.
 */

#pragma once

#include "ExpertTierWeightDeviceLayout.h"
#include "kernels/common/NativeQuantisedScaleLoad.h"
#include "GPUExpertTransfer.h"
#include "../../kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <mutex>
#include <string>
#include <vector>

namespace llaminar2
{
    /** Packed representation named by one stream endpoint. */
    enum class ExpertTierWeightPacking : uint8_t
    {
        GpuSeparatedNativeQuantised = 1,
        CpuNativeQuantisedInterleaved = 2,
    };

    /** Projection identity carried independently for gate, up, and down. */
    enum class ExpertTierWeightProjection : uint8_t
    {
        Gate = 0,
        Up = 1,
        Down = 2,
    };

    /**
     * @brief Immutable description of one cross-backend projection stream.
     *
     * `unit_count` units are transmitted in monotonically increasing order.
     * Each unit indexes one64-column tile/32-K block, but its payload contains
     * only live encoded columns and required metadata. Aligned full units keep
     * direct copies; origins/tails scatter into independent final local strides.
     * `manifest_hash` authenticates all semantic fields, including original ownership.
     */
    struct ExpertTierWeightStreamManifest
    {
        // Wire discriminator: receivers reject unknown layouts before parsing.
        static constexpr uint32_t kMagic = 0x45545753u; // "ETWS"
        // Version 5: coded units carry live payload/code planes and one
        // header at each source-superblock boundary (including intersections).
        static constexpr uint16_t kVersion = 5;

        // Direction is explicit so CPU bytes cannot be replayed as GPU bytes.
        uint32_t magic = kMagic;
        uint16_t version = kVersion;
        ExpertTierWeightPacking source_packing =
            ExpertTierWeightPacking::CpuNativeQuantisedInterleaved;
        ExpertTierWeightPacking destination_packing =
            ExpertTierWeightPacking::GpuSeparatedNativeQuantised;

        // Logical identity binds late chunks to one immutable placement epoch.
        uint64_t residency_epoch = 0;
        int32_t layer_idx = -1;
        int32_t expert_id = -1;
        ExpertTierWeightProjection projection =
            ExpertTierWeightProjection::Gate;
        uint8_t reserved_identity[3]{};

        // CPU pads N to 64; K must contain complete 32-value quant blocks.
        int32_t N = 0;
        int32_t K = 0;
        int32_t N_padded = 0;
        int32_t blocks_per_row = 0;

        // Source provenance and both endpoint formats are recorded because
        // CPU→GPU promotion may normalize a compact source representation.
        uint8_t cpu_codebook_id = 0;
        uint8_t gpu_source_codebook_id = 0;
        uint8_t gpu_source_is_superblock = 0;
        uint8_t gpu_codebook_id = 0;
        uint8_t gpu_payload_bytes_per_block = 0;
        uint8_t cpu_is_asymmetric = 0;
        uint8_t cpu_is_superblock = 0;
        uint8_t gpu_is_asymmetric = 0;
        uint8_t gpu_has_emins = 0;
        cpu::native_quantised::CPUNativeQuantisedEncoding cpu_encoding =
            cpu::native_quantised::CPUNativeQuantisedEncoding::SignedInt8SingleScale;

        // Allocation and ordered-chunk geometry derived from endpoint formats.
        uint32_t cpu_data_stride = 0;
        uint32_t cpu_block_stride = 0;
        uint32_t unit_count = 0;
        uint32_t maximum_units_per_chunk = 0;
        uint64_t total_stream_bytes = 0;
        uint64_t manifest_hash = 0;
        /** Original source and destination ownership; physical shapes alone are insufficient. */
        MoEExpertProjectionOwnership::Projection source_projection{};
        MoEExpertProjectionOwnership::Projection destination_projection{};

        /**
         * @brief Bind this format/provenance stream to an actual ownership intersection.
         * @param intersection Checked original source/destination geometry.
         * @return Resealed stream with local strides and compact live-column bytes.
         * @throws std::invalid_argument If source allocation geometry differs or overlap is empty.
         */
        [[nodiscard]] ExpertTierWeightStreamManifest forIntersection(
            const ExpertTierWeightProjectionIntersection& intersection) const;

        /**
         * @brief Recompute the canonical hash over all semantic fields.
         * @return Versioned FNV-1a manifest identity.
         */
        [[nodiscard]] uint64_t computedHash() const noexcept;

        /**
         * @brief Validate format, geometry, byte totals, direction, and hash.
         * @param error Optional exact rejection reason.
         * @return `true` only when every field forms one complete stream.
         */
        [[nodiscard]] bool valid(std::string *error = nullptr) const noexcept;

        /**
         * @brief Remove transport-only identity from a validated manifest.
         * @return Allocation-free scalar layout consumed by device launchers.
         */
        [[nodiscard]] ExpertTierWeightDeviceLayout deviceLayout() const noexcept;
    };

    /**
     * @brief Bind canonical native physical traits to one original CPU root/intersection.
     * @param native_format Existing CPU layout's native format fields; dimensions are derived afresh.
     * @param owned Exact original source or destination local rectangle.
     * @param intersection Same checked source/destination overlap used by every consumer.
     * @param maximum_chunk_bytes Existing admitted bounded lane capacity.
     * @return Checked CPU-only storage/wire geometry, without a GPU or remote identity.
     * @throws std::invalid_argument For foreign rectangles, native traits or insufficient capacity.
     */
    [[nodiscard]] ExpertTierWeightCPUStorageLayout cpuExpertTierStorageLayoutForIntersection(
        ExpertTierWeightCPUStorageLayout native_format,
        MoEExpertProjectionOwnership::Projection owned,
        const ExpertTierWeightProjectionIntersection &intersection,
        std::uint32_t maximum_chunk_bytes);

    /**
     * @brief Copy exact live native columns directly between existing CPU roots.
     * @param source_layout Checked original source/root geometry.
     * @param source Immutable whole original local source allocation.
     * @param destination_layout Same wire intersection and native encoding, independent root stride.
     * @param destination Existing final destination root, never a temporary parent.
     * @param first_unit First actual intersection-relative unit.
     * @param units Actual bounded units, excluding all resident padding.
     * @param encoded_write_mutex Shared destination-role authority for Q6 nibble RMW; null only for an exclusive writer.
     * @return False before mutation for incompatible geometry, ranges or aliases.
     * @note Scalar encoded fields are copied exactly; no decode/requantization or payload scratch exists.
     */
    bool copyExpertTierCPUWeightChunk(
        const ExpertTierWeightCPUStorageLayout &source_layout,
        std::span<const std::uint8_t> source,
        const ExpertTierWeightCPUStorageLayout &destination_layout,
        std::span<std::uint8_t> destination,
        std::uint32_t first_unit, std::uint32_t units,
        std::mutex *encoded_write_mutex = nullptr) noexcept;

    /** @brief Scatter live planes under the SAME destination-role Q6 word-write authority. */
    bool scatterExpertTierCPUWeightChunk(const ExpertTierWeightDeviceLayout &layout,
        std::span<const std::uint8_t> compact_chunk, std::uint32_t first_unit, std::uint32_t units,
        std::span<std::uint8_t> cpu_root, std::mutex *encoded_write_mutex) noexcept;
    /** @brief CPU-only form of the same shared encoded-word writer. */
    bool scatterExpertTierCPUWeightChunk(const ExpertTierWeightCPUStorageLayout &layout,
        std::span<const std::uint8_t> compact_chunk, std::uint32_t first_unit, std::uint32_t units,
        std::span<std::uint8_t> cpu_root, std::mutex *encoded_write_mutex) noexcept;

    /** Header for one monotonically ordered stream chunk. */
    struct ExpertTierWeightStreamChunkHeader
    {
        // Manifest identity rejects chunks from another projection or epoch.
        uint64_t manifest_hash = 0;
        // Payload identity detects corruption before final-memory mutation.
        uint64_t payload_hash = 0;
        // Sequence and first_unit jointly reject gaps, reordering, and replay.
        uint32_t sequence = 0;
        uint32_t first_unit = 0;
        uint32_t unit_count = 0;
        uint32_t payload_bytes = 0;
        // Exactly the chunk reaching unit_count must carry this marker.
        uint8_t final_chunk = 0;
        uint8_t reserved[7]{};
    };

    /**
     * @brief Strict receiver that writes chunks directly into final CPU bytes.
     *
     * The caller owns the destination allocation and is responsible for its
     * NUMA placement. This state machine never allocates, reorders, retries, or
     * accepts duplicate data.
     */
    class ExpertTierWeightStreamReceiver final
    {
    public:
        /**
         * @brief Construct one receiver over an already allocated final buffer.
         * @param manifest Immutable stream identity and layout contract.
         * @param final_bytes Exact-size final CPU execution allocation.
         */
        ExpertTierWeightStreamReceiver(
            ExpertTierWeightStreamManifest manifest,
            std::span<uint8_t> final_bytes);

        /**
         * @brief Validate and commit one next chunk.
         * @param header Ordered range and payload identity.
         * @param payload Complete CPU-format unit bytes.
         * @param error Optional exact rejection reason.
         * @return `true` only when the exact expected chunk was committed.
         */
        bool accept(
            const ExpertTierWeightStreamChunkHeader &header,
            std::span<const uint8_t> payload,
            std::string *error = nullptr) noexcept;

        /** @brief Abort the receiver; all subsequent chunks are rejected. */
        void abort() noexcept { aborted_ = true; }

        /**
         * @brief Query successful terminal state.
         * @return Whether every unit and the unique final marker arrived.
         */
        [[nodiscard]] bool complete() const noexcept { return complete_; }

        /**
         * @brief Query failed terminal state.
         * @return Whether the stream has been explicitly aborted.
         */
        [[nodiscard]] bool aborted() const noexcept { return aborted_; }

        /**
         * @brief Query committed progress.
         * @return Number of final units committed so far.
         */
        [[nodiscard]] uint32_t receivedUnits() const noexcept
        {
            return received_units_;
        }

        /**
         * @brief Query ordering state.
         * @return Next required chunk sequence.
         */
        [[nodiscard]] uint32_t nextSequence() const noexcept
        {
            return next_sequence_;
        }

    private:
        // The receiver borrows final storage; its owner controls NUMA lifetime.
        ExpertTierWeightStreamManifest manifest_;
        std::span<uint8_t> final_bytes_;
        // Progress advances only after a validated memcpy has completed.
        uint32_t next_sequence_ = 0;
        uint32_t received_units_ = 0;
        bool complete_ = false;
        bool aborted_ = false;
        bool construction_valid_ = false;
    };

    /** Host-owned representation of the common CUDA/ROCm separated layout. */
    struct HostGpuExpertPackedProjection
    {
        // Common CUDA/ROCm block-major matrix geometry and format metadata.
        int N = 0;
        int K = 0;
        uint32_t blocks_per_row = 0;
        uint8_t source_codebook_id = 0;
        uint8_t codebook_id = 0;
        uint8_t payload_bytes_per_block = 0;
        bool is_asymmetric = false;
        bool is_superblock = false;
        bool has_emins = false;
        // Separate arrays mirror the production device descriptor byte-for-byte.
        // `scales` holds nativeQuantisedScalePlaneStorageWords(codebook_id, blocks)
        // storage words: the exact primary plane (one byte per block for a
        // native E8M0 plane). Use scaleWord/setScaleWord for per-block access.
        std::vector<uint8_t> payload;
        std::vector<uint16_t> scales;
        std::vector<uint16_t> mins;
        std::vector<uint32_t> emins;
        // Coded endpoints use these exact planes in place of per-block metadata.
        uint8_t metadata_layout = 0;
        std::vector<uint8_t> codes;
        std::vector<uint8_t> superblocks;

        /** @return Staged/CPU-unit scale word of block @p linear (FP16, or BF16 power of two). */
        [[nodiscard]] uint16_t scaleWord(size_t linear) const noexcept
        {
            return native_quantised_scale::loadStagedPrimary(codebook_id, scales.data(), linear);
        }

        /** @return Exact live primary-plane bytes (device-view and transfer length). */
        [[nodiscard]] size_t scaleLiveBytes() const noexcept
        {
            return nativeQuantisedScalePlaneLiveBytes(
                codebook_id, static_cast<size_t>(N) * blocks_per_row);
        }

        /**
         * @brief Store a CPU-unit scale word into the exact plane.
         * @return False for a word that the native E8M0 plane cannot represent.
         */
        [[nodiscard]] bool setScaleWord(size_t linear, uint16_t word) noexcept
        {
            return native_quantised_scale::storePrimaryFromCpuWord(codebook_id, scales.data(), linear, word);
        }

        /**
         * @brief Validate separated host storage against declared geometry.
         * @param error Optional exact rejection reason.
         * @return Whether all regions exactly match the declared geometry.
         */
        [[nodiscard]] bool valid(std::string *error = nullptr) const noexcept;
    };

    /**
     * @brief Compute the stable payload fingerprint used by chunk validation.
     * @param bytes Complete contiguous bytes in one chunk.
     * @return FNV-1a hash carried by the chunk header.
     */
    [[nodiscard]] uint64_t expertTierWeightBytesHash(
        std::span<const uint8_t> bytes) noexcept;

    /**
     * @brief Build a GPU-to-CPU stream manifest for any cataloged source format.
     * @param source_format Original tensor format represented by the GPU arrays.
     * @param N Logical output-column count.
     * @param K Logical reduction dimension, divisible by 32.
     * @param residency_epoch Exact immutable residency epoch.
     * @param layer_idx Logical transformer layer.
     * @param expert_id Logical routed expert.
     * @param projection Gate/up/down identity.
     * @param maximum_units_per_chunk Persistent staging capacity in units.
     * @return Complete authenticated stream manifest.
     * @throws std::invalid_argument when the source identity or geometry is
     *         not part of the exhaustive NativeQuantised catalog.
     */
    [[nodiscard]] ExpertTierWeightStreamManifest
    makeGpuToCpuExpertTierWeightStreamManifest(
        const NativeQuantisedFormatInfo &source_format,
        int N,
        int K,
        uint64_t residency_epoch,
        int layer_idx,
        int expert_id,
        ExpertTierWeightProjection projection,
        uint32_t maximum_units_per_chunk);

    /**
     * @brief Build a GPU-to-CPU stream from the live physical GPU descriptor.
     *
     * Authenticate the live canonical GPU layout and retain the original GGUF
     * identity so demotion produces the same CPU encoding as direct loading.
     *
     * @param source_format Original catalogued GGUF format identity.
     * @param gpu_source Exact live separated GPU descriptor.
     * @param residency_epoch Immutable source residency epoch.
     * @param layer_idx Logical transformer layer.
     * @param expert_id Logical routed expert.
     * @param projection Gate/up/down identity.
     * @param maximum_units_per_chunk Persistent staging capacity in units.
     * @return Complete authenticated stream using the live GPU execution format.
     * @throws std::invalid_argument when provenance, geometry, or physical
     *         representation is not an exact supported migration state.
     */
    [[nodiscard]] ExpertTierWeightStreamManifest
    makeGpuToCpuExpertTierWeightStreamManifest(
        const NativeQuantisedFormatInfo &source_format,
        const GpuExpertPackedDescriptor &gpu_source,
        uint64_t residency_epoch,
        int layer_idx,
        int expert_id,
        ExpertTierWeightProjection projection,
        uint32_t maximum_units_per_chunk);

    /**
     * @brief Build a CPU-to-GPU stream manifest from final prepared CPU bytes.
     * @param cpu_weights Source CPU execution representation.
     * @param residency_epoch Exact immutable residency epoch.
     * @param layer_idx Logical transformer layer.
     * @param expert_id Logical routed expert.
     * @param projection Gate/up/down identity.
     * @param maximum_units_per_chunk Persistent staging capacity in units.
     * @return Complete authenticated stream manifest for the source's canonical GPU format.
     * @throws std::invalid_argument when the CPU representation cannot form a
     *         complete, allocation-stable cross-tier stream.
     */
    [[nodiscard]] ExpertTierWeightStreamManifest
    makeCpuToGpuExpertTierWeightStreamManifest(
        const cpu::native_quantised::CPUNativeQuantisedPackedWeights &cpu_weights,
        uint64_t residency_epoch,
        int layer_idx,
        int expert_id,
        ExpertTierWeightProjection projection,
        uint32_t maximum_units_per_chunk);

    /**
     * @brief Reference deinterleave from final CPU bytes to GPU separated data.
     *
     * Canonical CPU storage preserves enough source information to reproduce
     * the original GPU payload and metadata planes without requantization.
     * @param cpu_weights Valid final CPU execution representation.
     * @param gpu_weights Receives common separated GPU-format bytes.
     * @param error Optional exact rejection reason.
     * @return `true` when the conversion completed byte-exactly.
     */
    [[nodiscard]] bool cpuToGpuExpertPackedReference(
        const cpu::native_quantised::CPUNativeQuantisedPackedWeights &cpu_weights,
        HostGpuExpertPackedProjection &gpu_weights,
        std::string *error = nullptr) noexcept;

    /**
     * @brief Reference pack from a canonical GPU projection to CPU bytes.
     *
     * The authenticated source tuple determines the CPU encoding, including
     * code-plane geometry for formats with retained superblock metadata.
     * @param gpu_weights Valid common separated GPU-format bytes.
     * @param cpu_weights Receives final padded CPU execution bytes.
     * @param error Optional exact rejection reason.
     * @return `true` when the conversion completed byte-exactly.
     */
    [[nodiscard]] bool gpuToCpuExpertPackedReference(
        const HostGpuExpertPackedProjection &gpu_weights,
        cpu::native_quantised::CPUNativeQuantisedPackedWeights &cpu_weights,
        std::string *error = nullptr) noexcept;

} // namespace llaminar2
