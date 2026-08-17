/**
 * @file GPUExpertTransfer.h
 * @brief Same-backend GPU expert weight transfer for MoE rebalancing.
 *
 * Provides explicit-stream device-to-device transfer of packed MoE expert
 * weights between same-backend GPU devices. This is ~50x faster
 * than the serialize → MPI → deserialize → repack path for intra-node
 * GPU↔GPU transfers since both devices use identical packed weight format.
 *
 * ── Format Conversion Note (CPU ↔ GPU) ──────────────────────────────
 *
 * CPU (CPUNativeVNNIPackedWeights) and GPU (MoEBatchPackedWeightsROCm/CUDA)
 * use DIFFERENT memory layouts for the same underlying quantized bytes:
 *
 *   CPU interleaved:
 *     Per K-block across 64 rows (N-chunk): [1024B payload (4 groups × 4 ZMMs,
 *     transposed)] [128B comp_int16] [128B scales_fp16] [128B mins_fp16]
 *     - Payload bytes are transposed into VNNI register layout for AVX-512
 *     - stride = 1280B (symmetric) or 1408B (asymmetric)
 *
 *   GPU separated:
 *     Flat arrays: all_native_vnni[linear_block * payload_bytes], one scale per block,
 *     one min per block. Payload bytes are in per-row linear order.
 *
 * Metadata granularity is IDENTICAL (1 FP16 scale, 1 FP16 min per 32-element block).
 * The payload bytes are the SAME raw quantized bytes, but stored in different
 * transposition order (VNNI group/ZMM/lane layout vs. linear per-row). Conversion
 * is a pure byte-level scatter/gather (no mathematical transformation), but the
 * transposition across 64-row N-chunks adds non-trivial complexity.
 *
 * GPU↔GPU transfer does NOT require format conversion since both ROCm (and CUDA)
 * devices use the identical separated layout. CPU↔GPU transfer requires the
 * existing serialize → deserialize → repack path until deinterleave kernels are
 * implemented.
 */

#pragma once

#include "../../backends/DeviceId.h"
#include "../../tensors/TensorKernels.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaminar2 {

/// Pointers to one expert's packed weight arrays on a GPU device.
struct GPUExpertPointers {
    uint8_t* d_vnni = nullptr;
    void* d_scales = nullptr;    // uint16_t* (FP16)
    void* d_mins = nullptr;      // uint16_t* (FP16), nullptr if symmetric
    void* d_emins = nullptr;     // uint32_t*, nullptr if not present
};

/// Backend-neutral descriptor for one GPU-resident expert projection in the
/// separated NativeVNNI layout consumed by both CUDA and ROCm kernels.
struct GpuExpertPackedDescriptor {
    GPUExpertPointers ptrs;
    int n = 0;
    int k = 0;
    uint32_t blocks_per_row = 0;
    uint8_t codebook_id = 0;
    uint8_t payload_bytes_per_block = 0;
    bool is_asymmetric = false;
    bool has_emins = false;

    /**
     * Physical capacity of the persistent allocation backing this live view.
     *
     * These fields are deliberately distinct from the execution fields above:
     * an initially compact expert can occupy a slot large enough for a later
     * CPU-normalized promotion without changing the bytes consumed by GEMM.
     * Zero identifies an ordinary immutable allocation with no recycling
     * contract.
     */
    uint8_t allocation_payload_bytes_per_block = 0;
    bool allocation_has_mins = false;
    bool allocation_has_emins = false;

    size_t vnni_bytes = 0;
    size_t scales_bytes = 0;
    size_t mins_bytes = 0;
    size_t emins_bytes = 0;

    bool valid() const
    {
        return ptrs.d_vnni != nullptr &&
               ptrs.d_scales != nullptr &&
               n > 0 &&
               k > 0 &&
               blocks_per_row > 0 &&
               payload_bytes_per_block > 0 &&
               vnni_bytes > 0 &&
               scales_bytes > 0;
    }

    size_t totalBytes() const
    {
        return vnni_bytes + scales_bytes + mins_bytes + emins_bytes;
    }


    /** @return Number of logical 32-value blocks in this projection. */
    [[nodiscard]] size_t blockCount() const
    {
        return static_cast<size_t>(blocks_per_row) *
               static_cast<size_t>(n);
    }

    /** @return Bytes physically reserved for payload, or zero if unspecified. */
    [[nodiscard]] size_t allocationPayloadBytes() const
    {
        return blockCount() * allocation_payload_bytes_per_block;
    }

    /** @return Bytes physically reserved for minima, or zero when absent. */
    [[nodiscard]] size_t allocationMinsBytes() const
    {
        return allocation_has_mins
                   ? blockCount() * sizeof(uint16_t)
                   : 0;
    }

    /** @return Bytes physically reserved for extended minima, or zero when absent. */
    [[nodiscard]] size_t allocationEminsBytes() const
    {
        return allocation_has_emins
                   ? blockCount() * sizeof(uint32_t)
                   : 0;
    }
};

struct GpuExpertStagedActivation
{
    GpuExpertPackedDescriptor staged;
    GpuExpertPackedDescriptor active;
};

inline GpuExpertPackedDescriptor makeGpuExpertPackedDescriptor(
    const DeviceNativeVNNIMatrixDesc& desc,
    uint8_t payload_bytes_per_block,
    bool is_asymmetric,
    bool has_emins)
{
    const size_t block_count =
        static_cast<size_t>(desc.blocks_per_row) * static_cast<size_t>(desc.n);

    GpuExpertPackedDescriptor out;
    out.ptrs.d_vnni = const_cast<uint8_t*>(desc.payload);
    out.ptrs.d_scales = const_cast<void*>(desc.scales);
    out.ptrs.d_mins = const_cast<void*>(desc.mins);
    out.ptrs.d_emins = const_cast<void*>(desc.emins);
    out.n = desc.n;
    out.k = desc.k;
    out.blocks_per_row = desc.blocks_per_row;
    out.codebook_id = desc.codebook_id;
    out.payload_bytes_per_block = payload_bytes_per_block;
    out.is_asymmetric = is_asymmetric;
    out.has_emins = has_emins;
    out.allocation_payload_bytes_per_block =
        desc.allocation_payload_bytes_per_block;
    out.allocation_has_mins = desc.allocation_has_mins != 0;
    out.allocation_has_emins = desc.allocation_has_emins != 0;
    out.vnni_bytes = block_count * payload_bytes_per_block;
    out.scales_bytes = block_count * sizeof(uint16_t);
    out.mins_bytes = is_asymmetric ? out.scales_bytes : 0;
    out.emins_bytes = has_emins ? block_count * sizeof(uint32_t) : 0;
    return out;
}

inline bool gpuExpertPackedDescriptorsCompatible(
    const GpuExpertPackedDescriptor& src,
    const GpuExpertPackedDescriptor& dst)
{
    return src.valid() &&
           dst.valid() &&
           src.n == dst.n &&
           src.k == dst.k &&
           src.blocks_per_row == dst.blocks_per_row &&
           src.codebook_id == dst.codebook_id &&
           src.payload_bytes_per_block == dst.payload_bytes_per_block &&
           src.is_asymmetric == dst.is_asymmetric &&
           src.has_emins == dst.has_emins &&
           src.vnni_bytes == dst.vnni_bytes &&
           src.scales_bytes == dst.scales_bytes &&
           src.mins_bytes == dst.mins_bytes &&
           src.emins_bytes == dst.emins_bytes;
}

/// Transfer expert weights between same-backend GPU devices on an explicit stream.
///
/// Both source and destination must use the same packed NativeVNNI layout.
class GPUExpertTransfer {
public:
    /// Transfer one expert's packed weights from src to dst device.
    /// Both src and dst must be ROCm devices (or both CUDA).
    /// @param src_ptrs Source device pointers (on src_device)
    /// @param dst_ptrs Destination device pointers (pre-allocated on dst_device)
    /// @param src_device Source device
    /// @param dst_device Destination device
    /// @param vnni_bytes Size of vnni array for this expert
    /// @param scales_bytes Size of scales array for this expert (in bytes)
    /// @param mins_bytes Size of mins array in bytes (0 if symmetric)
    /// @param emins_bytes Size of emins array in bytes (0 if not present)
    /// @param stream Explicit stream on dst_device for async transfer. nullptr
    /// is rejected; callers must never use the CUDA/HIP legacy stream here.
    /// @return true on success
    static bool transferExpert(
        const GPUExpertPointers& src_ptrs,
        const GPUExpertPointers& dst_ptrs,
        const DeviceId& src_device,
        const DeviceId& dst_device,
        size_t vnni_bytes,
        size_t scales_bytes,
        size_t mins_bytes,
        size_t emins_bytes,
        void* stream);

    /// Transfer one expert projection described by backend-neutral descriptors.
    static bool transferExpert(
        const GpuExpertPackedDescriptor& src,
        const GpuExpertPackedDescriptor& dst,
        const DeviceId& src_device,
        const DeviceId& dst_device,
        void* stream);

    /// Activate a staged same-device expert arrival by copying the transfer-slot
    /// descriptor into its compute-facing active slot on an explicit stream.
    /// This is intentionally named separately from cross-device transfer because
    /// the caller may enqueue it later, after a staging event has completed.
    static bool activateStagedExpert(
        const GpuExpertPackedDescriptor& staged,
        const GpuExpertPackedDescriptor& active,
        const DeviceId& device,
        void* stream);

    /// Activate a batch of staged same-device expert projections on one explicit
    /// stream. The call enqueues all copies in order and does not synchronize.
    static bool activateStagedExperts(
        const std::vector<GpuExpertStagedActivation>& activations,
        const DeviceId& device,
        void* stream);

};

} // namespace llaminar2
