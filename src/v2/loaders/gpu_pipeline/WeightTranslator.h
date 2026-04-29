#pragma once

/**
 * @file WeightTranslator.h
 * @brief Host-side API for translating weights between GPU packed format and raw GGUF blocks.
 *
 * Provides four translation paths:
 *
 * 1. **packGpuWeightsForTransfer()** — GPU separated arrays → contiguous host buffer
 *    with GpuPackedWeightsHeader. Used for GPU→GPU MPI transfer and serialization.
 *
 * 2. **uploadGpuPackedWeights()** — Contiguous GpuPacked buffer → pre-allocated GPU
 *    separated arrays. Direct H2D memcpy (no repack kernel needed).
 *
 * 3. **reverseRepackOnDevice()** — GPU separated → raw GGUF blocks on device.
 *    Only supports reversible per-block formats (Q4_0, IQ4_NL, Q4_1, Q5_0, Q5_1, Q8_0).
 *    Used when converting GPU weights back for CPU packing.
 *
 * 4. **forwardRepackOnDevice()** — Raw GGUF blocks on device → GPU separated arrays.
 *    Delegates to existing CUDAVnniRepackKernels / VnniRepackKernels.
 *    Used when loading MoE expert weights from raw blocks.
 *
 * All operations are stream-ordered: the caller must ensure the stream is synchronized
 * before reading output buffers.
 */

#include "backends/DeviceType.h"
#include "backends/IBackend.h"
#include "loaders/gpu_pipeline/GpuPackedWeightsFormat.h"
#include "loaders/gpu_pipeline/RepackFormat.h"

#include <cstdint>
#include <cstring>
#include <vector>

// Forward repack kernels (CUDA and ROCm)
#ifdef HAVE_CUDA
#include "kernels/cuda/repack/CUDAVnniRepackKernels.h"
#include "kernels/cuda/repack/CUDAVnniUnpackKernels.h"
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/repack/VnniRepackKernels.h"
#include "kernels/rocm/repack/VnniUnpackKernels.h"
#endif

namespace llaminar2 {

/**
 * @brief Raw GGUF block sizes for reversible formats.
 *
 * Returns sizeof(block struct) for per-block formats, 0 for unsupported.
 */
inline size_t rawBlockSizeBytes(RepackFormat format)
{
    switch (format) {
    case RepackFormat::Q4_0:    return 18;  // Q4_0Block
    case RepackFormat::IQ4_NL:  return 18;  // IQ4_NLBlock (same layout as Q4_0)
    case RepackFormat::Q4_1:    return 20;  // Q4_1Block
    case RepackFormat::Q5_0:    return 22;  // Q5_0Block
    case RepackFormat::Q5_1:    return 24;  // Q5_1Block
    case RepackFormat::Q8_0:    return 34;  // Q8_0Block
    default:                    return 0;   // Not supported for reverse repack
    }
}

/**
 * @brief Compute total raw GGUF block buffer size for a weight matrix.
 *
 * @param format     Quantization format (must be reversible)
 * @param N          Output features (rows)
 * @param K          Input features (columns), must be multiple of 32
 * @return Total bytes needed for N * (K/32) raw blocks, or 0 if not reversible
 */
inline size_t rawBlockBufferSize(RepackFormat format, int N, int K)
{
    size_t block_bytes = rawBlockSizeBytes(format);
    if (block_bytes == 0) return 0;
    return static_cast<size_t>(N) * (K / 32) * block_bytes;
}

class WeightTranslator {
public:
    // ========================================================================
    // Path 1: GPU separated → contiguous host buffer (for MPI/serialization)
    // ========================================================================

    /**
     * @brief Download GPU separated VNNI arrays into a contiguous GpuPacked buffer.
     *
     * Performs D2H copies of payload, scales, mins (if asymmetric), and emins
     * (if Q2_K), then assembles them behind a GpuPackedWeightsHeader.
     *
     * @param backend    Device backend for D2H transfers
     * @param device_id  GPU device ordinal
     * @param d_payload  Device payload pointer
     * @param d_scales   Device scales pointer (FP16 as uint16_t*)
     * @param d_mins     Device mins pointer (nullptr if symmetric)
     * @param d_emins    Device emins pointer (nullptr unless Q2_K)
     * @param header     Pre-built header describing the weight layout
     * @param stream     GPU stream for async D2H (nullptr = default)
     * @return Contiguous host buffer: [GpuPackedWeightsHeader][payload][scales][mins?][emins?]
     */
    static std::vector<uint8_t> packGpuWeightsForTransfer(
        IBackend& backend, int device_id,
        const uint8_t* d_payload,
        const uint16_t* d_scales,
        const uint16_t* d_mins,
        const uint32_t* d_emins,
        const GpuPackedWeightsHeader& header,
        void* stream = nullptr)
    {
        const size_t total = gpuPackedTotalSize(header);
        std::vector<uint8_t> buf(total);

        // Write header
        std::memcpy(buf.data(), &header, sizeof(header));
        size_t offset = sizeof(GpuPackedWeightsHeader);

        // D2H payload
        backend.deviceToHost(buf.data() + offset, d_payload, header.payload_size, device_id, stream);
        offset += header.payload_size;

        // D2H scales
        backend.deviceToHost(buf.data() + offset, d_scales, header.scales_size, device_id, stream);
        offset += header.scales_size;

        // D2H mins (if asymmetric)
        if (header.is_asymmetric && header.mins_size > 0 && d_mins) {
            backend.deviceToHost(buf.data() + offset, d_mins, header.mins_size, device_id, stream);
            offset += header.mins_size;
        }

        // D2H emins (if Q2_K)
        if (header.has_emins && header.emins_size > 0 && d_emins) {
            backend.deviceToHost(buf.data() + offset, d_emins, header.emins_size, device_id, stream);
        }

        return buf;
    }

    // ========================================================================
    // Path 2: Contiguous GpuPacked buffer → GPU separated arrays (direct H2D)
    // ========================================================================

    /**
     * @brief Upload a GpuPacked buffer directly into pre-allocated GPU arrays.
     *
     * No repack kernel needed — the buffer sections are already in GPU separated layout.
     * The destination arrays must be pre-allocated (e.g., via WeightVRAMPool).
     *
     * @param backend    Device backend for H2D transfers
     * @param device_id  GPU device ordinal
     * @param view       Parsed GpuPackedWeightsView (from parseGpuPackedBuffer)
     * @param d_payload  Device destination for payload
     * @param d_scales   Device destination for scales
     * @param d_mins     Device destination for mins (ignored if symmetric)
     * @param d_emins    Device destination for emins (ignored unless Q2_K)
     * @param stream     GPU stream for async H2D (nullptr = default)
     * @return true on success
     */
    static bool uploadGpuPackedWeights(
        IBackend& backend, int device_id,
        const GpuPackedWeightsView& view,
        uint8_t* d_payload,
        uint16_t* d_scales,
        uint16_t* d_mins,
        uint32_t* d_emins,
        void* stream = nullptr)
    {
        const auto& h = view.header;

        // H2D payload
        if (!backend.hostToDevice(d_payload, view.payload, h.payload_size, device_id, stream))
            return false;

        // H2D scales
        if (!backend.hostToDevice(d_scales, view.scales, h.scales_size, device_id, stream))
            return false;

        // H2D mins
        if (h.is_asymmetric && h.mins_size > 0 && view.mins && d_mins) {
            if (!backend.hostToDevice(d_mins, view.mins, h.mins_size, device_id, stream))
                return false;
        }

        // H2D emins
        if (h.has_emins && h.emins_size > 0 && view.emins && d_emins) {
            if (!backend.hostToDevice(d_emins, view.emins, h.emins_size, device_id, stream))
                return false;
        }

        return true;
    }

    // ========================================================================
    // Path 3: GPU separated → raw GGUF blocks on device (reverse repack)
    // ========================================================================

    /**
     * @brief Reverse-repack GPU separated arrays into raw GGUF blocks on device.
     *
     * Only works for reversible per-block formats. Caller must pre-allocate
     * d_raw_blocks with rawBlockBufferSize() bytes on the target device.
     *
     * @param format      Quantization format (must pass isReversibleFormat())
     * @param d_payload   Device payload pointer (GPU separated)
     * @param d_scales    Device scales pointer (GPU separated, FP16)
     * @param d_mins      Device mins pointer (nullptr if symmetric)
     * @param d_raw_blocks Device output buffer for raw GGUF blocks
     * @param N           Output features (rows)
     * @param K           Input features (columns)
     * @param device_type CUDA or ROCm
     * @param stream      GPU stream (nullptr = default)
     * @return true on success, false if format not reversible or kernel launch fails
     */
    static bool reverseRepackOnDevice(
        RepackFormat format,
        const uint8_t* d_payload,
        const uint16_t* d_scales,
        const uint16_t* d_mins,
        void* d_raw_blocks,
        int N, int K,
        DeviceType device_type,
        void* stream = nullptr)
    {
        if (!isReversibleFormat(format))
            return false;

        switch (device_type) {
#ifdef HAVE_CUDA
        case DeviceType::CUDA:
            return launchVnniUnpackCUDA(format, d_payload, d_scales, d_mins,
                                        d_raw_blocks, N, K, stream);
#endif
#ifdef HAVE_ROCM
        case DeviceType::ROCm:
            return launchVnniUnpack(format, d_payload, d_scales, d_mins,
                                   d_raw_blocks, N, K, stream);
#endif
        default:
            return false;
        }
    }

    // ========================================================================
    // Path 4: Raw GGUF blocks on device → GPU separated arrays (forward repack)
    // ========================================================================

    /**
     * @brief Forward-repack raw GGUF blocks into GPU separated VNNI arrays on device.
     *
     * Delegates to existing CUDAVnniRepackKernels / VnniRepackKernels.
     * Supports all 19 RepackFormat values.
     *
     * @param format      Quantization format
     * @param d_raw_blocks Device pointer to raw GGUF blocks
     * @param d_payload   Device output for payload
     * @param d_scales    Device output for scales
     * @param d_mins      Device output for mins (nullptr if symmetric)
     * @param d_emins     Device output for emins (nullptr unless Q2_K)
     * @param N           Output features (rows)
     * @param K           Input features (columns)
     * @param device_type CUDA or ROCm
     * @param stream      GPU stream (nullptr = default)
     * @return true on success, false on error
     */
    static bool forwardRepackOnDevice(
        RepackFormat format,
        const void* d_raw_blocks,
        uint8_t* d_payload,
        uint16_t* d_scales,
        uint16_t* d_mins,
        uint32_t* d_emins,
        int N, int K,
        DeviceType device_type,
        void* stream = nullptr)
    {
        switch (device_type) {
#ifdef HAVE_CUDA
        case DeviceType::CUDA:
            return launchVnniRepackCUDA(format, d_raw_blocks, d_payload,
                                        d_scales, d_mins, d_emins, N, K, stream);
#endif
#ifdef HAVE_ROCM
        case DeviceType::ROCm:
            return launchVnniRepack(format, d_raw_blocks, d_payload,
                                   d_scales, d_mins, d_emins, N, K, stream);
#endif
        default:
            return false;
        }
    }
};

} // namespace llaminar2
