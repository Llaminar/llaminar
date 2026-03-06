/**
 * @file ROCmWeightPacker.cpp
 * @brief Weight packing utilities: native-VNNI and INT8 conversion for ROCm GEMM
 *
 * ## Architecture
 *
 * Weight packing uses polymorphic dispatch on IINT8Unpackable:
 *   1. vnniFormatInfo()       → per-format metadata (payload bytes, codebook, etc.)
 *   2. packVnniBlock()        → per-format block extraction (payload, scale, min)
 *   3. requantizeRowToInt8()  → per-format INT8 requantization (no FP32 round-trip)
 *   4. packNativeVNNI()       → generic loop calling the polymorphic methods
 *
 * @see ROCmWeightPacker.h for public API
 * @see ROCmQuantisedGemmKernel.h for ROCmPackedWeights struct
 * @see tensors/VnniPackContext.h for packing context struct
 */

#include "ROCmWeightPacker.h"
#include "gemm/ROCmQuantisedGemmKernel.h"
#include "tensors/TensorClasses.h"   // IINT8Unpackable (for packVnniBlock, requantizeRowToInt8)
#include "tensors/VnniPackContext.h" // VnniPackContext, vnniLinearIdx, etc.
#include "tensors/IQQuantTables.h"   // iq3s_grid, ksigns_iq2xs, etc. (for IQ grid init)
#include "tensors/TensorType.h"      // isNativeVnniFormat, isInt8VnniFormat
#include "utils/Logger.h"
#include "utils/DebugEnv.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <mutex>
#include <set>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

// IQ grid table initialization (implemented in ROCmGemvKernel_native_VNNI.hip)
extern "C" bool rocmInitIQGridTables(
    int device_id,
    const void *h_iq3s_grid, const void *h_iq3xxs_grid,
    const void *h_iq2s_grid, const void *h_iq2xs_grid,
    const void *h_iq2xxs_grid, const void *h_iq1s_grid);

// IQ grid table initialization for GEMM TU (implemented in ROCmGemmKernel_native_VNNI.hip)
extern "C" bool rocmInitIQGridTables_gemm(
    int device_id,
    const void *h_iq3s_grid, const void *h_iq3xxs_grid,
    const void *h_iq2s_grid, const void *h_iq2xs_grid,
    const void *h_iq2xxs_grid, const void *h_iq1s_grid);

namespace llaminar2
{
    namespace rocm
    {

        // =================================================================
        // packNativeVNNI: polymorphic metadata + polymorphic block packing
        // =================================================================

        bool packNativeVNNI(const TensorBase *tensor, ROCmPackedWeights &out)
        {
            if (!tensor)
                return false;

            // Get IINT8Unpackable interface (needed for vnniFormatInfo, get_block_scale, get_block_min)
            auto *quant_accessor = dynamic_cast<const IINT8Unpackable *>(tensor);
            if (!quant_accessor)
                return false;

            // Format metadata now lives on the tensor class itself
            const auto *info = quant_accessor->vnniFormatInfo();
            if (!info)
                return false;

            // Lazy-initialize IQ grid lookup tables in GPU __constant__ memory
            if (info->codebook_id >= 11 && info->codebook_id <= 17)
            {
                static std::mutex iq_grid_mutex;
                static std::set<int> iq_grids_initialized_devices;

                int current_device = 0;
#ifdef HAVE_ROCM
                hipGetDevice(&current_device);
#elif defined(HAVE_CUDA)
                cudaGetDevice(&current_device);
#endif

                bool needs_init = false;
                {
                    std::lock_guard<std::mutex> lock(iq_grid_mutex);
                    needs_init = (iq_grids_initialized_devices.find(current_device) ==
                                  iq_grids_initialized_devices.end());
                }

                if (needs_init)
                {
                    LOG_INFO("[packNativeVNNI] Initializing IQ grid LUT tables on device "
                             << current_device);
                    if (!rocmInitIQGridTables(
                            current_device,
                            llaminar2::iq3s_grid,
                            llaminar2::iq3xxs_grid,
                            llaminar2::iq2s_grid,
                            llaminar2::iq2xs_grid,
                            llaminar2::iq2xxs_grid,
                            llaminar2::iq1s_grid))
                    {
                        LOG_ERROR("[packNativeVNNI] IQ grid GEMV init failed on device " << current_device);
                        return false;
                    }
                    if (!rocmInitIQGridTables_gemm(
                            current_device,
                            llaminar2::iq3s_grid,
                            llaminar2::iq3xxs_grid,
                            llaminar2::iq2s_grid,
                            llaminar2::iq2xs_grid,
                            llaminar2::iq2xxs_grid,
                            llaminar2::iq1s_grid))
                    {
                        LOG_ERROR("[packNativeVNNI] IQ grid GEMM init failed on device " << current_device);
                        return false;
                    }
                    std::lock_guard<std::mutex> lock(iq_grid_mutex);
                    iq_grids_initialized_devices.insert(current_device);
                }
            }

            const int N = static_cast<int>(tensor->rows());
            const int K = static_cast<int>(tensor->cols());
            if ((K % 32) != 0)
                return false;

            const int blocks_per_row = K / 32;
            const int payload_bytes = info->payload_bytes;
            const bool is_asymmetric = info->is_asymmetric;

            // Allocate output buffers
            out.native_vnni_payload.resize(static_cast<size_t>(blocks_per_row) * N * payload_bytes);
            out.native_vnni_scales.resize(static_cast<size_t>(blocks_per_row) * N);
            if (is_asymmetric)
                out.native_vnni_mins.resize(static_cast<size_t>(blocks_per_row) * N);
            if (info->has_emins)
                out.native_vnni_emins.resize(static_cast<size_t>(blocks_per_row) * N);
            out.native_vnni_codebook_id = info->codebook_id;
            out.native_vnni_blocks_per_row = static_cast<uint32_t>(blocks_per_row);

            // Per-row max-abs for CK prefill INT8 requantization compatibility
            out.scales.resize(N);
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                float max_abs = 0.0f;
                for (int b = 0; b < blocks_per_row; ++b)
                {
                    const float scale_b = std::abs(quant_accessor->get_block_scale(
                        static_cast<size_t>(n), static_cast<size_t>(b)));
                    float block_max = scale_b * info->max_abs_factor;
                    if (is_asymmetric)
                    {
                        const float min_b = std::abs(quant_accessor->get_block_min(
                            static_cast<size_t>(n), static_cast<size_t>(b)));
                        block_max += min_b;
                    }
                    max_abs = std::max(max_abs, block_max);
                }
                out.scales[n] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            }

            // Interleave payload, scales (and mins) by N for coalesced GPU access
            // (Block data is accessed via tensor->typed_data() in packVnniBlock)

            // Build the packing context
            VnniPackContext ctx{};
            ctx.raw_bytes = nullptr; // unused — tensor classes use typed_data()
            ctx.N = N;
            ctx.K = K;
            ctx.blocks_per_row = blocks_per_row;
            ctx.payload_bytes = payload_bytes;
            ctx.payload_array = out.native_vnni_payload.data();
            ctx.scales_array = out.native_vnni_scales.data();
            ctx.mins_array = is_asymmetric ? out.native_vnni_mins.data() : nullptr;
            ctx.emins_array = info->has_emins ? out.native_vnni_emins.data() : nullptr;

            // Main packing loop — dispatches to per-format virtual method
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                for (int b = 0; b < blocks_per_row; ++b)
                {
                    quant_accessor->packVnniBlock(ctx, n, b);
                }
            }

            LOG_DEBUG("[packNativeVNNI] Built native-VNNI container for " << N << "x" << K
                                                                          << " (codebook=" << static_cast<int>(out.native_vnni_codebook_id) << ")"
                                                                          << " payload=" << (out.native_vnni_payload.size() / 1024) << " KB"
                                                                          << " scales=" << (out.native_vnni_scales.size() * 2 / 1024) << " KB"
                                                                          << (is_asymmetric ? (" mins=" + std::to_string(out.native_vnni_mins.size() * 2 / 1024) + " KB") : ""));
            return true;
        }

        // =================================================================
        // packWeightsToROCm: Convert any quantized tensor to INT8 + scales
        // =================================================================

        bool packWeightsToROCm(const TensorBase *tensor, ROCmPackedWeights &out)
        {
            if (!tensor)
            {
                LOG_ERROR("[packWeightsToROCm] Null tensor");
                return false;
            }

            const int N = static_cast<int>(tensor->rows());
            const int K = static_cast<int>(tensor->cols());

            out.K = K;
            out.N = N;

            const TensorType wt = tensor->native_type();

            // ---- Native-VNNI path (≤6-bit formats) ----
            if (isNativeVnniFormat(wt))
            {
                if (!packNativeVNNI(tensor, out))
                {
                    LOG_ERROR("[packWeightsToROCm] Native-VNNI packing failed for "
                              << tensorTypeName(wt) << " " << N << "x" << K);
                    return false;
                }
                LOG_DEBUG("[packWeightsToROCm] Packed " << N << "x" << K << " "
                                                        << tensorTypeName(wt) << " to native-VNNI only");
                return true;
            }

            // ---- INT8-VNNI path (8-bit formats: Q8_0, Q8_1, Q8_K) ----
            // Uses IINT8Unpackable::requantizeRowToInt8() for zero-FP32-copy
            // requantization. Each tensor type has an efficient override that
            // reads its native blocks directly without virtual dispatch per block.
            if (isInt8VnniFormat(wt))
            {
                const auto *quant_accessor = dynamic_cast<const IINT8Unpackable *>(tensor);
                if (!quant_accessor)
                {
                    LOG_ERROR("[packWeightsToROCm] INT8 format " << tensorTypeName(wt)
                                                                 << " does not implement IINT8Unpackable");
                    return false;
                }

                if (K % 32 != 0)
                {
                    LOG_ERROR("[packWeightsToROCm] K=" << K << " not divisible by 32 for "
                                                       << tensorTypeName(wt));
                    return false;
                }

                out.scales.resize(N);
                out.int8_data.resize(static_cast<size_t>(N) * K);

                const bool build_vnni = (K % 4) == 0;
                const size_t k_groups = build_vnni ? (static_cast<size_t>(K) / 4) : 0;
                if (build_vnni)
                    out.int8_data_vnni.resize(k_groups * static_cast<size_t>(N) * 4);
                else
                    out.int8_data_vnni.clear();

#pragma omp parallel for schedule(static)
                for (int n = 0; n < N; ++n)
                {
                    int8_t *row_dst = out.int8_data.data() + static_cast<size_t>(n) * K;

                    // Polymorphic per-row requantization (no FP32 materialization)
                    out.scales[n] = quant_accessor->requantizeRowToInt8(
                        static_cast<size_t>(n), static_cast<size_t>(K), row_dst);

                    // Build VNNI interleaved layout: [K/4][N][4]
                    if (build_vnni)
                    {
                        for (size_t kg = 0; kg < k_groups; ++kg)
                        {
                            const size_t src_offset = kg * 4;
                            const size_t dst_offset = (kg * static_cast<size_t>(N) + static_cast<size_t>(n)) * 4;
                            out.int8_data_vnni[dst_offset + 0] = row_dst[src_offset + 0];
                            out.int8_data_vnni[dst_offset + 1] = row_dst[src_offset + 1];
                            out.int8_data_vnni[dst_offset + 2] = row_dst[src_offset + 2];
                            out.int8_data_vnni[dst_offset + 3] = row_dst[src_offset + 3];
                        }
                    }
                }

                if (debugEnv().rocm.pack_vnni_only_host)
                {
                    if (!out.int8_data_vnni.empty())
                    {
                        out.int8_data.clear();
                        out.int8_data.shrink_to_fit();
                        LOG_DEBUG("[packWeightsToROCm] VNNI-only host pack enabled; released row-major host copy for "
                                  << N << "x" << K << " weights");
                    }
                    else
                    {
                        LOG_WARN("[packWeightsToROCm] LLAMINAR_ROCM_PACK_VNNI_ONLY=1 requested but VNNI layout unavailable "
                                 << "(K=" << K << " not divisible by 4). Falling back to row-major host pack.");
                    }
                }

                LOG_DEBUG("[packWeightsToROCm] Packed " << N << "x" << K << " "
                                                        << tensorTypeName(wt) << " to INT8 (no FP32 round-trip)"
                                                        << (out.int8_data_vnni.empty() ? "" : " + VNNI"));
                return true;
            }

            LOG_ERROR("[packWeightsToROCm] Unsupported tensor type for weight packing: "
                      << tensorTypeName(wt));
            return false;
        }

    } // namespace rocm
} // namespace llaminar2
