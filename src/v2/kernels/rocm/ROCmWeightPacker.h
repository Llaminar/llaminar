#ifndef LLAMINAR2_KERNELS_ROCM_ROCMWEIGHTPACKER_H
#define LLAMINAR2_KERNELS_ROCM_ROCMWEIGHTPACKER_H

/**
 * @file ROCmWeightPacker.h
 * @brief Weight packing utilities for ROCm native-VNNI and INT8 GEMM kernels
 *
 * Extracted from ROCmQuantisedGemmKernel.cpp to isolate the weight conversion
 * pipeline (one-time at model load) from the GEMM execution hot path.
 *
 * ## Format Classification
 *
 * | Category         | Formats                                          | Packing Path         |
 * |------------------|--------------------------------------------------|----------------------|
 * | Native-VNNI      | Q4_0, IQ4_NL, Q4_1, Q5_0, Q5_1, IQ4_XS, Q4_K,  | packNativeVNNI()     |
 * | (≤6-bit)         | Q5_K, Q6_K, Q3_K, Q2_K, IQ3_S, IQ3_XXS, IQ2_S,  |                      |
 * |                  | IQ2_XS, IQ2_XXS, IQ1_S, IQ1_M                   |                      |
 * | INT8-VNNI (8-bit)| Q8_0, Q8_1, Q8_K                                | INT8 requantization  |
 *
 * @see ROCmQuantisedGemmKernel.h for the GEMM kernel and ROCmPackedWeights struct
 */

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    // Forward declarations
    class TensorBase;

    namespace rocm
    {
        // Forward declaration (defined in ROCmQuantisedGemmKernel.h)
        struct ROCmPackedWeights;

        // NativeVnniFormatInfo metadata and block packing are now provided
        // by each tensor class via IINT8Unpackable::vnniFormatInfo() and
        // IINT8Unpackable::packVnniBlock().  See:
        //   - tensors/NativeVnniFormatInfo.h   (format metadata struct)
        //   - tensors/VnniPackContext.h         (packing context struct)
        //   - tensors/TensorClasses.h           (per-class overrides)

        // =================================================================
        // Public API (packWeightsToROCm is declared in ROCmQuantisedGemmKernel.h)
        // =================================================================

        /**
         * @brief Pack tensor to native-VNNI layout (internal, called by packWeightsToROCm)
         *
         * Converts any supported ≤6-bit quantized tensor to native-VNNI format:
         * - Interleaved payload bytes for coalesced GPU access
         * - Separate FP16 scale/min arrays
         * - Pre-computed sub-block scales for super-block formats
         *
         * @param tensor Source quantized tensor
         * @param out Output packed weights structure
         * @return true on success
         */
        bool packNativeVNNI(const TensorBase *tensor, ROCmPackedWeights &out);

        /**
         * @brief Batch-packed MoE expert weights for ROCm (single GPU allocation for all experts).
         *
         * Same design as MoEBatchPackedWeightsCUDA but for ROCm/HIP devices.
         */
        struct MoEBatchPackedWeightsROCm
        {
            std::vector<uint8_t> all_native_vnni;
            std::vector<uint16_t> all_native_scales;
            std::vector<uint16_t> all_native_mins;
            std::vector<uint32_t> all_native_emins;

            int num_experts = 0;
            int rows_per_expert = 0;
            int K = 0;
            int blocks_per_row = 0;
            uint8_t codebook_id = 0;

            size_t vnni_bytes_per_expert = 0;
            size_t scales_per_expert = 0;
            size_t mins_per_expert = 0;
            size_t emins_per_expert = 0;

            struct DeviceUpload
            {
                uint8_t *d_native_vnni = nullptr;
                void *d_native_scales = nullptr;  // __half* on device
                void *d_native_mins = nullptr;
                void *d_native_emins = nullptr;
            };

            std::mutex upload_mutex;
            std::unordered_map<int, DeviceUpload> device_uploads;

            MoEBatchPackedWeightsROCm() = default;
            MoEBatchPackedWeightsROCm(const MoEBatchPackedWeightsROCm &) = delete;
            MoEBatchPackedWeightsROCm &operator=(const MoEBatchPackedWeightsROCm &) = delete;
            ~MoEBatchPackedWeightsROCm();

            bool uploadToDevice(int rocm_device_id);
            DeviceUpload getExpertDevicePointers(int rocm_device_id, int expert_id) const;
            void freeHostBuffers();
        };

        std::shared_ptr<MoEBatchPackedWeightsROCm> packMoEExpertsROCm(
            const std::vector<std::shared_ptr<TensorBase>> &expert_views,
            int num_experts, int rows_per_expert);

    } // namespace rocm
} // namespace llaminar2

#endif // LLAMINAR2_KERNELS_ROCM_ROCMWEIGHTPACKER_H
