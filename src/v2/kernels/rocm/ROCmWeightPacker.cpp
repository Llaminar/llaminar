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

// IQ grid table initialization for GEMM TU (implemented in ROCmQuantisedGemmKernel_native_VNNI.hip)
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
                    LOG_DEBUG("[packNativeVNNI] Initializing IQ grid LUT tables on device "
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

    // =========================================================================
    // MoE Batch Packing — pack all experts into one contiguous slab for ROCm
    // =========================================================================

    extern "C"
    {
        void rocmQuantGemm_freeDevice(void *d_ptr, int rocm_device_id);
    }

    MoEBatchPackedWeightsROCm::~MoEBatchPackedWeightsROCm()
    {
#ifdef HAVE_ROCM
        for (auto &[device_id, upload] : device_uploads)
        {
            if (upload.d_native_vnni)
                rocmQuantGemm_freeDevice(upload.d_native_vnni, device_id);
            if (upload.d_native_scales)
                rocmQuantGemm_freeDevice(upload.d_native_scales, device_id);
            if (upload.d_native_mins)
                rocmQuantGemm_freeDevice(upload.d_native_mins, device_id);
            if (upload.d_native_emins)
                rocmQuantGemm_freeDevice(upload.d_native_emins, device_id);
        }
#endif
    }

    bool MoEBatchPackedWeightsROCm::uploadToDevice(int rocm_device_id)
    {
#ifdef HAVE_ROCM
        std::lock_guard<std::mutex> lock(upload_mutex);

        auto it = device_uploads.find(rocm_device_id);
        if (it != device_uploads.end())
            return true;

        DeviceUpload upload;

        auto uploadBuffer = [&](const void *host_data, size_t bytes, void **d_ptr) -> bool
        {
            *d_ptr = nullptr;
            if (!host_data || bytes == 0)
                return true;
            hipError_t err = hipSetDevice(rocm_device_id);
            if (err != hipSuccess)
                return false;
            err = hipMalloc(d_ptr, bytes);
            if (err != hipSuccess || !*d_ptr)
                return false;
            err = hipMemcpy(*d_ptr, host_data, bytes, hipMemcpyHostToDevice);
            if (err != hipSuccess)
            {
                hipFree(*d_ptr);
                *d_ptr = nullptr;
                return false;
            }
            return true;
        };

        void *d_vnni = nullptr, *d_scales = nullptr, *d_mins = nullptr, *d_emins = nullptr;

        if (!uploadBuffer(all_native_vnni.data(),
                          all_native_vnni.size() * sizeof(uint8_t), &d_vnni) ||
            !uploadBuffer(all_native_scales.data(),
                          all_native_scales.size() * sizeof(uint16_t), &d_scales) ||
            !uploadBuffer(all_native_mins.empty() ? nullptr : all_native_mins.data(),
                          all_native_mins.size() * sizeof(uint16_t), &d_mins) ||
            !uploadBuffer(all_native_emins.empty() ? nullptr : all_native_emins.data(),
                          all_native_emins.size() * sizeof(uint32_t), &d_emins))
        {
            if (d_vnni) hipFree(d_vnni);
            if (d_scales) hipFree(d_scales);
            if (d_mins) hipFree(d_mins);
            if (d_emins) hipFree(d_emins);
            return false;
        }

        upload.d_native_vnni = reinterpret_cast<uint8_t *>(d_vnni);
        upload.d_native_scales = d_scales;
        upload.d_native_mins = d_mins;
        upload.d_native_emins = d_emins;
        device_uploads.emplace(rocm_device_id, upload);
        return true;
#else
        (void)rocm_device_id;
        return false;
#endif
    }

    MoEBatchPackedWeightsROCm::DeviceUpload
    MoEBatchPackedWeightsROCm::getExpertDevicePointers(int rocm_device_id, int expert_id) const
    {
        auto it = device_uploads.find(rocm_device_id);
        if (it == device_uploads.end())
            return {};

        const auto &base = it->second;
        DeviceUpload expert;
        expert.d_native_vnni = base.d_native_vnni
                                   ? base.d_native_vnni + expert_id * vnni_bytes_per_expert
                                   : nullptr;
        expert.d_native_scales = base.d_native_scales
                                     ? reinterpret_cast<uint16_t *>(base.d_native_scales) + expert_id * scales_per_expert
                                     : nullptr;
        expert.d_native_mins = base.d_native_mins
                                   ? reinterpret_cast<uint16_t *>(base.d_native_mins) + expert_id * mins_per_expert
                                   : nullptr;
        expert.d_native_emins = base.d_native_emins
                                    ? reinterpret_cast<uint32_t *>(base.d_native_emins) + expert_id * emins_per_expert
                                    : nullptr;
        return expert;
    }

    void MoEBatchPackedWeightsROCm::freeHostBuffers()
    {
        const size_t freed = all_native_vnni.capacity() +
                             all_native_scales.capacity() * sizeof(uint16_t) +
                             all_native_mins.capacity() * sizeof(uint16_t) +
                             all_native_emins.capacity() * sizeof(uint32_t);
        all_native_vnni.clear();
        all_native_vnni.shrink_to_fit();
        all_native_scales.clear();
        all_native_scales.shrink_to_fit();
        all_native_mins.clear();
        all_native_mins.shrink_to_fit();
        all_native_emins.clear();
        all_native_emins.shrink_to_fit();
        if (freed > 0)
        {
            LOG_DEBUG("[MoEBatchPackedWeightsROCm] Released host buffers: "
                      << (freed / (1024 * 1024)) << " MB");
        }
    }

    std::shared_ptr<MoEBatchPackedWeightsROCm> packMoEExpertsROCm(
        const std::vector<std::shared_ptr<TensorBase>> &expert_views,
        int num_experts, int rows_per_expert)
    {
        if (expert_views.empty() || num_experts <= 0 || rows_per_expert <= 0)
        {
            LOG_ERROR("[packMoEExpertsROCm] Invalid arguments");
            return nullptr;
        }

        const auto *quant = dynamic_cast<const IINT8Unpackable *>(expert_views[0].get());
        if (!quant)
        {
            LOG_ERROR("[packMoEExpertsROCm] Expert view does not implement IINT8Unpackable");
            return nullptr;
        }
        const auto *info = quant->vnniFormatInfo();
        if (!info)
        {
            LOG_ERROR("[packMoEExpertsROCm] Expert view has no VNNI format info");
            return nullptr;
        }

        const int K = static_cast<int>(expert_views[0]->cols());
        if ((K % 32) != 0)
        {
            LOG_ERROR("[packMoEExpertsROCm] K=" << K << " not divisible by 32");
            return nullptr;
        }

        const int blocks_per_row = K / 32;

        auto batch = std::make_shared<MoEBatchPackedWeightsROCm>();
        batch->num_experts = num_experts;
        batch->rows_per_expert = rows_per_expert;
        batch->K = K;
        batch->blocks_per_row = blocks_per_row;
        batch->codebook_id = info->codebook_id;

        batch->vnni_bytes_per_expert = static_cast<size_t>(blocks_per_row) * rows_per_expert * info->payload_bytes;
        batch->scales_per_expert = static_cast<size_t>(blocks_per_row) * rows_per_expert;
        batch->mins_per_expert = info->is_asymmetric ? batch->scales_per_expert : 0;
        batch->emins_per_expert = info->has_emins ? batch->scales_per_expert : 0;

        batch->all_native_vnni.assign(static_cast<size_t>(num_experts) * batch->vnni_bytes_per_expert, uint8_t{0});
        batch->all_native_scales.assign(static_cast<size_t>(num_experts) * batch->scales_per_expert, uint16_t{0});
        if (info->is_asymmetric)
            batch->all_native_mins.assign(static_cast<size_t>(num_experts) * batch->mins_per_expert, uint16_t{0});
        if (info->has_emins)
            batch->all_native_emins.assign(static_cast<size_t>(num_experts) * batch->emins_per_expert, uint32_t{0});

        for (int e = 0; e < num_experts; ++e)
        {
            const auto *eq = dynamic_cast<const IINT8Unpackable *>(expert_views[e].get());
            if (!eq)
            {
                LOG_ERROR("[packMoEExpertsROCm] Expert " << e << " not IINT8Unpackable");
                return nullptr;
            }

            VnniPackContext ctx{};
            ctx.raw_bytes = nullptr;
            ctx.N = rows_per_expert;
            ctx.K = K;
            ctx.blocks_per_row = blocks_per_row;
            ctx.payload_bytes = info->payload_bytes;
            ctx.payload_array = batch->all_native_vnni.data() + e * batch->vnni_bytes_per_expert;
            ctx.scales_array = batch->all_native_scales.data() + e * batch->scales_per_expert;
            ctx.mins_array = info->is_asymmetric
                                 ? batch->all_native_mins.data() + e * batch->mins_per_expert
                                 : nullptr;
            ctx.emins_array = info->has_emins
                                  ? batch->all_native_emins.data() + e * batch->emins_per_expert
                                  : nullptr;

#pragma omp parallel for schedule(static)
            for (int n = 0; n < rows_per_expert; ++n)
            {
                for (int b = 0; b < blocks_per_row; ++b)
                {
                    eq->packVnniBlock(ctx, n, b);
                }
            }
        }

        LOG_DEBUG("[packMoEExpertsROCm] Packed " << num_experts << " experts ("
                 << rows_per_expert << "x" << K << " each), total slab "
                 << (batch->all_native_vnni.size() / (1024 * 1024)) << " MB vnni + "
                 << (batch->all_native_scales.size() * 2 / (1024 * 1024)) << " MB scales");

        return batch;
    }

    } // namespace rocm
} // namespace llaminar2
