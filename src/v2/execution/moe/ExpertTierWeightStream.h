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
#include "GPUExpertTransfer.h"
#include "../../kernels/cpu/gemm/CPUNativeVNNIWeightPacker.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{
    /** Packed representation named by one stream endpoint. */
    enum class ExpertTierWeightPacking : uint8_t
    {
        GpuSeparatedNativeVnni = 1,
        CpuNativeVnniInterleaved = 2,
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
     * Every unit is exactly one CPU interleaved block and can therefore be
     * copied directly to its final CPU address or converted independently by a
     * destination GPU. `manifest_hash` authenticates every field preceding it.
     */
    struct ExpertTierWeightStreamManifest
    {
        // Wire discriminator: receivers reject unknown layouts before parsing.
        static constexpr uint32_t kMagic = 0x45545753u; // "ETWS"
        static constexpr uint16_t kVersion = 2;

        // Direction is explicit so CPU bytes cannot be replayed as GPU bytes.
        uint32_t magic = kMagic;
        uint16_t version = kVersion;
        ExpertTierWeightPacking source_packing =
            ExpertTierWeightPacking::CpuNativeVnniInterleaved;
        ExpertTierWeightPacking destination_packing =
            ExpertTierWeightPacking::GpuSeparatedNativeVnni;

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
        cpu::native_vnni::CPUNativeVNNIEncoding cpu_encoding =
            cpu::native_vnni::CPUNativeVNNIEncoding::ExpandedInt8;

        // Allocation and ordered-chunk geometry derived from endpoint formats.
        uint32_t cpu_data_stride = 0;
        uint32_t cpu_block_stride = 0;
        uint32_t unit_count = 0;
        uint32_t maximum_units_per_chunk = 0;
        uint64_t total_stream_bytes = 0;
        uint64_t manifest_hash = 0;

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
        std::vector<uint8_t> payload;
        std::vector<uint16_t> scales;
        std::vector<uint16_t> mins;
        std::vector<uint32_t> emins;

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
     *         not part of the exhaustive NativeVNNI catalog.
     */
    [[nodiscard]] ExpertTierWeightStreamManifest
    makeGpuToCpuExpertTierWeightStreamManifest(
        const NativeVnniFormatInfo &source_format,
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
     * A prior CPU promotion may have irreversibly normalized an expanded CPU
     * representation into physical codebook 19 or 23 while retaining its
     * original GGUF provenance. This overload authenticates that live physical
     * layout and preserves the original CPU format in the outgoing stream, so
     * arbitrary promotion/demotion cycles remain lossless.
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
        const NativeVnniFormatInfo &source_format,
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
     * @return Complete authenticated stream manifest whose GPU destination is
     *         compact when reversible and normalized INT8 otherwise.
     * @throws std::invalid_argument when the CPU representation cannot form a
     *         complete, allocation-stable cross-tier stream.
     */
    [[nodiscard]] ExpertTierWeightStreamManifest
    makeCpuToGpuExpertTierWeightStreamManifest(
        const cpu::native_vnni::CPUNativeVNNIPackedWeights &cpu_weights,
        uint64_t residency_epoch,
        int layer_idx,
        int expert_id,
        ExpertTierWeightProjection projection,
        uint32_t maximum_units_per_chunk);

    /**
     * @brief Reference deinterleave from final CPU bytes to GPU separated data.
     *
     * Expanded signed INT8 becomes codebook 19 when symmetric or the explicit
     * ExpertOverlay INT8+minimum codebook when asymmetric. Nibble and Q6
     * encodings preserve their compact device payload exactly.
     * @param cpu_weights Valid final CPU execution representation.
     * @param gpu_weights Receives common separated GPU-format bytes.
     * @param error Optional exact rejection reason.
     * @return `true` when the conversion completed byte-exactly.
     */
    [[nodiscard]] bool cpuToGpuExpertPackedReference(
        const cpu::native_vnni::CPUNativeVNNIPackedWeights &cpu_weights,
        HostGpuExpertPackedProjection &gpu_weights,
        std::string *error = nullptr) noexcept;

    /**
     * @brief Reference pack from a normalized GPU projection to CPU bytes.
     *
     * Accepted sources are the reversible nibble encodings, native Q6_K, raw
     * signed INT8, and ExpertOverlay signed INT8 with a per-block minimum.
     * Source-codebook transcoding for other formats is owned by the backend
     * GPU conversion kernels and is deliberately not guessed here.
     * @param gpu_weights Valid common separated GPU-format bytes.
     * @param cpu_weights Receives final padded CPU execution bytes.
     * @param error Optional exact rejection reason.
     * @return `true` when the conversion completed byte-exactly.
     */
    [[nodiscard]] bool gpuToCpuExpertPackedReference(
        const HostGpuExpertPackedProjection &gpu_weights,
        cpu::native_vnni::CPUNativeVNNIPackedWeights &cpu_weights,
        std::string *error = nullptr) noexcept;

} // namespace llaminar2
