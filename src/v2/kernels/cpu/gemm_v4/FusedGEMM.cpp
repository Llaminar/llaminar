/**
 * @file FusedGEMM.cpp
 * @brief Implementation of generic fused multi-GEMM
 * @author David Sanftenberg
 * @date 2025-11-26
 */

#include "FusedGEMM.h"
#include "../../KernelFactory.h"
#include "../../../utils/Logger.h"
#include "../../../utils/KernelProfiler.h"

namespace llaminar2
{
    FusedGEMM::FusedGEMM(const std::vector<const TensorBase *> &weights,
                         const std::vector<std::string> &projection_names)
        : projection_names_(projection_names)
    {
        if (weights.empty())
        {
            throw std::invalid_argument("[FusedGEMM] Must provide at least one weight tensor");
        }

        // Validate and extract common k dimension
        k_dim_ = -1;
        for (size_t i = 0; i < weights.size(); ++i)
        {
            if (!weights[i])
            {
                throw std::invalid_argument("[FusedGEMM] Weight tensor " + std::to_string(i) + " is null");
            }

            const auto &shape = weights[i]->shape();
            if (shape.size() != 2)
            {
                throw std::invalid_argument("[FusedGEMM] Weight tensor " + std::to_string(i) + " must be 2D");
            }

            int weight_k = static_cast<int>(shape[1]);
            if (k_dim_ < 0)
            {
                k_dim_ = weight_k;
            }
            else if (weight_k != k_dim_)
            {
                throw std::invalid_argument("[FusedGEMM] All weights must have matching input dimension (k). "
                                            "Weight 0 has k=" +
                                            std::to_string(k_dim_) +
                                            ", weight " + std::to_string(i) + " has k=" + std::to_string(weight_k));
            }
        }

        // Ensure we have names for all projections (generate defaults if needed)
        projection_names_.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i)
        {
            if (projection_names_[i].empty())
            {
                projection_names_[i] = "projection_" + std::to_string(i);
            }
        }

        // Get GEMM kernels from KernelFactory (don't create new ones!)
        // This ensures we use the pre-packed weights and don't re-access raw data.
        gemm_kernels_.reserve(weights.size());
        for (size_t i = 0; i < weights.size(); ++i)
        {
            const auto *weight = weights[i];
            // KernelFactory::getOrCreateGemm() returns cached ITensorGemm*
            auto *gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(weight);
            auto *quantised_gemm = dynamic_cast<gemm_v4::QuantisedGemmKernel *>(gemm);
            if (!quantised_gemm)
            {
                throw std::runtime_error("[FusedGEMM] Weight tensor returned non-QuantisedGemmKernel from KernelFactory::getOrCreateGemm()");
            }
            LOG_DEBUG("[FusedGEMM] Got kernel " << i << " ptr=" << static_cast<const void *>(gemm)
                                                << " weight ptr=" << static_cast<const void *>(weight));
            gemm_kernels_.push_back(quantised_gemm);
        }
    }

    FusedGEMM::FusedGEMM(const TensorBase *weight1, const TensorBase *weight2)
        : FusedGEMM(std::vector<const TensorBase *>{weight1, weight2},
                    {"gate", "up"})
    {
    }

    FusedGEMM::FusedGEMM(const TensorBase *weight1, const TensorBase *weight2, const TensorBase *weight3)
        : FusedGEMM(std::vector<const TensorBase *>{weight1, weight2, weight3},
                    {"Q", "K", "V"})
    {
    }

    bool FusedGEMM::execute(
        const float *input,
        const std::vector<GEMMProjection> &projections,
        int m, int k,
        const MPIContext *ctx,
        int device_idx)
    {
        // Validate inputs
        if (!input)
        {
            LOG_ERROR("[FusedGEMM] Input pointer is null");
            return false;
        }

        // DEBUG: Log input pointer and first values for parity testing
        LOG_INFO("[FusedGEMM] input ptr=" << static_cast<const void *>(input)
                                          << " input[0:4]=" << std::setprecision(10)
                                          << input[0] << "," << input[1] << ","
                                          << input[2] << "," << input[3]);

        if (projections.size() != gemm_kernels_.size())
        {
            LOG_ERROR("[FusedGEMM] Number of projections (" << projections.size()
                                                            << ") doesn't match number of weights (" << gemm_kernels_.size() << ")");
            return false;
        }

        if (m <= 0 || k <= 0)
        {
            LOG_ERROR("[FusedGEMM] Invalid dimensions: m=" << m << " k=" << k);
            return false;
        }

        for (size_t i = 0; i < projections.size(); ++i)
        {
            if (!projections[i].output)
            {
                LOG_ERROR("[FusedGEMM] Output pointer for " << projection_names_[i] << " is null");
                return false;
            }
            if (projections[i].n <= 0)
            {
                LOG_ERROR("[FusedGEMM] Invalid n dimension for " << projection_names_[i] << ": " << projections[i].n);
                return false;
            }
        }

        // =====================================================================
        // FUSED PATH: Quantize activations once, reuse for all GEMMs
        // =====================================================================

        // Step 1: Allocate shared Q8_1 buffer for quantized activations
        size_t buffer_size = gemm_kernels_[0]->get_quantized_activation_buffer_size(m, k);
        std::vector<uint8_t> q8_1_buffer(buffer_size);

        // Step 2: Quantize activations once (profile separately)
        bool success;
        {
            KERNEL_PROFILE_SCOPE(KernelType::QUANTIZE_Q8);
            success = gemm_kernels_[0]->quantize_activations(input, q8_1_buffer.data(), m, k);
        }
        if (!success)
        {
            LOG_ERROR("[FusedGEMM] Failed to quantize activations");
            return false;
        }

        // Debug: dump first Q8_1 block for comparison
        const Q8_1Block *blocks = reinterpret_cast<const Q8_1Block *>(q8_1_buffer.data());
        LOG_DEBUG("[FusedGEMM] First Q8_1 block: d=" << blocks[0].d
                                                     << " sum_qs=" << blocks[0].sum_qs
                                                     << " qs[0:4]=" << (int)blocks[0].qs[0] << ","
                                                     << (int)blocks[0].qs[1] << ","
                                                     << (int)blocks[0].qs[2] << ","
                                                     << (int)blocks[0].qs[3]);

        // Step 3: Execute all projections with shared quantized activations
        for (size_t i = 0; i < gemm_kernels_.size(); ++i)
        {
            KERNEL_PROFILE_SCOPE(KernelType::GEMM_Q8);

            const auto &proj = projections[i];
            const auto &name = projection_names_[i];

            // Build fused ops configuration
            GemmFusedOps fused_ops = proj.do_swiglu
                                         ? GemmFusedOps::swiglu(proj.gate_input)
                                         : GemmFusedOps::none();

            // Debug: Log kernel info before GEMM call for layer 0 Q projection
            if (i == 0 && proj.n == 896 && k == 896)
            {
                LOG_INFO("[FusedGEMM] Calling multiply_with_precomputed_q8_1 for Q: "
                         << "kernel=" << static_cast<const void *>(gemm_kernels_[i])
                         << " m=" << m << " n=" << proj.n << " k=" << k);
            }

            success = gemm_kernels_[i]->multiply_with_precomputed_q8_1(
                q8_1_buffer.data(),
                proj.output,
                m, proj.n, k,
                proj.bias,  // Fused bias
                false,      // No accumulation
                1.0f, 0.0f, // alpha=1, beta=0
                ctx, device_idx,
                fused_ops);

            if (!success)
            {
                LOG_ERROR("[FusedGEMM] " << name << " GEMM failed");
                return false;
            }

            // Debug: dump first 4 output values for Q projection (n=896, k=896)
            if (i == 0 && proj.n == 896 && k == 896)
            {
                LOG_INFO("[FusedGEMM Q OUTPUT] n=" << proj.n << " output[0:4]="
                                                   << std::setprecision(10) << proj.output[0] << "," << proj.output[1] << "," << proj.output[2] << "," << proj.output[3]);
            }
        }

        return true;
    }

    bool FusedGEMM::execute(
        const float *input,
        float *output1, float *output2,
        const TensorBase *bias1, const TensorBase *bias2,
        int m, int n, int k,
        const MPIContext *ctx,
        int device_idx)
    {
        if (gemm_kernels_.size() != 2)
        {
            LOG_ERROR("[FusedGEMM] 2-projection execute() called but kernel has "
                      << gemm_kernels_.size() << " projections");
            return false;
        }

        return execute(
            input,
            {{output1, bias1, n, projection_names_[0]},
             {output2, bias2, n, projection_names_[1]}},
            m, k, ctx, device_idx);
    }

    bool FusedGEMM::execute(
        const float *input,
        float *output1, float *output2, float *output3,
        const TensorBase *bias1, const TensorBase *bias2, const TensorBase *bias3,
        int m, int n1, int n2, int n3, int k,
        const MPIContext *ctx,
        int device_idx)
    {
        if (gemm_kernels_.size() != 3)
        {
            LOG_ERROR("[FusedGEMM] 3-projection execute() called but kernel has "
                      << gemm_kernels_.size() << " projections");
            return false;
        }

        return execute(
            input,
            {{output1, bias1, n1, projection_names_[0]},
             {output2, bias2, n2, projection_names_[1]},
             {output3, bias3, n3, projection_names_[2]}},
            m, k, ctx, device_idx);
    }

    // =========================================================================
    // Q8_1 Output Implementation
    // =========================================================================

    bool FusedGEMM::execute_to_q8_1(
        const float *input,
        const std::vector<GEMMProjectionQ8_1> &projections,
        int m, int k,
        const MPIContext *ctx,
        int device_idx)
    {
        // Validate inputs
        if (!input)
        {
            LOG_ERROR("[FusedGEMM::execute_to_q8_1] Input pointer is null");
            return false;
        }

        if (projections.size() != gemm_kernels_.size())
        {
            LOG_ERROR("[FusedGEMM::execute_to_q8_1] Number of projections (" << projections.size()
                                                                             << ") doesn't match number of weights (" << gemm_kernels_.size() << ")");
            return false;
        }

        if (m <= 0 || k <= 0)
        {
            LOG_ERROR("[FusedGEMM::execute_to_q8_1] Invalid dimensions: m=" << m << " k=" << k);
            return false;
        }

        for (size_t i = 0; i < projections.size(); ++i)
        {
            if (!projections[i].output_q8_1)
            {
                LOG_ERROR("[FusedGEMM::execute_to_q8_1] Output pointer for " << projection_names_[i] << " is null");
                return false;
            }
            if (projections[i].n <= 0)
            {
                LOG_ERROR("[FusedGEMM::execute_to_q8_1] Invalid n dimension for " << projection_names_[i] << ": " << projections[i].n);
                return false;
            }
        }

        // =====================================================================
        // FUSED PATH: Quantize activations once, reuse for all GEMMs with Q8_1 output
        // =====================================================================

        // Step 1: Allocate shared Q8_1 buffer for quantized activations
        size_t buffer_size = gemm_kernels_[0]->get_quantized_activation_buffer_size(m, k);
        std::vector<uint8_t> q8_1_buffer(buffer_size);

        // Step 2: Quantize activations once (profile separately)
        bool success;
        {
            KERNEL_PROFILE_SCOPE(KernelType::QUANTIZE_Q8);
            success = gemm_kernels_[0]->quantize_activations(input, q8_1_buffer.data(), m, k);
        }
        if (!success)
        {
            LOG_ERROR("[FusedGEMM::execute_to_q8_1] Failed to quantize activations");
            return false;
        }

        // Step 3: Execute all projections with Q8_1 output
        for (size_t i = 0; i < gemm_kernels_.size(); ++i)
        {
            KERNEL_PROFILE_SCOPE(KernelType::GEMM_Q8);

            const auto &proj = projections[i];
            const auto &name = projection_names_[i];

            success = gemm_kernels_[i]->multiply_with_precomputed_q8_1_to_q8_1(
                q8_1_buffer.data(),
                proj.output_q8_1,
                m, proj.n, k,
                proj.bias, // Fused bias (added before Q8_1 requantization)
                false,     // No accumulation for Q8_1 output
                ctx, device_idx);

            if (!success)
            {
                LOG_ERROR("[FusedGEMM::execute_to_q8_1] " << name << " GEMM failed");
                return false;
            }
        }

        return true;
    }

    bool FusedGEMM::execute_to_q8_1(
        const float *input,
        void *output_q, void *output_k, void *output_v,
        const TensorBase *bias_q, const TensorBase *bias_k, const TensorBase *bias_v,
        int m, int n_q, int n_kv, int k,
        const MPIContext *ctx,
        int device_idx)
    {
        if (gemm_kernels_.size() != 3)
        {
            LOG_ERROR("[FusedGEMM::execute_to_q8_1] 3-projection execute_to_q8_1() called but kernel has "
                      << gemm_kernels_.size() << " projections");
            return false;
        }

        return execute_to_q8_1(
            input,
            {{output_q, bias_q, n_q, projection_names_[0]},
             {output_k, bias_k, n_kv, projection_names_[1]},
             {output_v, bias_v, n_kv, projection_names_[2]}},
            m, k, ctx, device_idx);
    }

    // =========================================================================
    // Q8_1-to-Q8_1 Implementation (avoids double quantization)
    // =========================================================================

    bool FusedGEMM::execute_q8_1_to_q8_1(
        const void *input_q8_1,
        const std::vector<GEMMProjectionQ8_1> &projections,
        int m, int k,
        const MPIContext *ctx,
        int device_idx)
    {
        // Validate inputs
        if (!input_q8_1)
        {
            LOG_ERROR("[FusedGEMM::execute_q8_1_to_q8_1] Input Q8_1 pointer is null");
            return false;
        }

        if (projections.size() != gemm_kernels_.size())
        {
            LOG_ERROR("[FusedGEMM::execute_q8_1_to_q8_1] Number of projections (" << projections.size()
                                                                                  << ") doesn't match number of weights (" << gemm_kernels_.size() << ")");
            return false;
        }

        if (m <= 0 || k <= 0)
        {
            LOG_ERROR("[FusedGEMM::execute_q8_1_to_q8_1] Invalid dimensions: m=" << m << " k=" << k);
            return false;
        }

        for (size_t i = 0; i < projections.size(); ++i)
        {
            if (!projections[i].output_q8_1)
            {
                LOG_ERROR("[FusedGEMM::execute_q8_1_to_q8_1] Output pointer for " << projection_names_[i] << " is null");
                return false;
            }
            if (projections[i].n <= 0)
            {
                LOG_ERROR("[FusedGEMM::execute_q8_1_to_q8_1] Invalid n dimension for " << projection_names_[i] << ": " << projections[i].n);
                return false;
            }
        }

        // =====================================================================
        // FUSED PATH: Use pre-quantized Q8_1 input directly (no requantization)
        // =====================================================================

        // Execute all projections with Q8_1→Q8_1 path
        for (size_t i = 0; i < gemm_kernels_.size(); ++i)
        {
            KERNEL_PROFILE_SCOPE(KernelType::GEMM_Q8);

            const auto &proj = projections[i];
            const auto &name = projection_names_[i];

            bool success = gemm_kernels_[i]->multiply_with_precomputed_q8_1_to_q8_1(
                input_q8_1,
                proj.output_q8_1,
                m, proj.n, k,
                proj.bias, // Fused bias (added before Q8_1 requantization)
                false,     // No accumulation for Q8_1 output
                ctx, device_idx);

            if (!success)
            {
                LOG_ERROR("[FusedGEMM::execute_q8_1_to_q8_1] " << name << " GEMM failed");
                return false;
            }
        }

        return true;
    }

    bool FusedGEMM::execute_q8_1_to_q8_1(
        const void *input_q8_1,
        void *output_q, void *output_k, void *output_v,
        const TensorBase *bias_q, const TensorBase *bias_k, const TensorBase *bias_v,
        int m, int n_q, int n_kv, int k,
        const MPIContext *ctx,
        int device_idx)
    {
        if (gemm_kernels_.size() != 3)
        {
            LOG_ERROR("[FusedGEMM::execute_q8_1_to_q8_1] 3-projection execute_q8_1_to_q8_1() called but kernel has "
                      << gemm_kernels_.size() << " projections");
            return false;
        }

        return execute_q8_1_to_q8_1(
            input_q8_1,
            {{output_q, bias_q, n_q, projection_names_[0]},
             {output_k, bias_k, n_kv, projection_names_[1]},
             {output_v, bias_v, n_kv, projection_names_[2]}},
            m, k, ctx, device_idx);
    }

    bool FusedGEMM::execute_q8_1_mixed_qkv(
        const void *input_q8_1,
        void *output_q, // Q8_1 output
        void *output_k, // Q16_1 output (high precision!)
        void *output_v, // Q8_1 output
        const TensorBase *bias_q, const TensorBase *bias_k, const TensorBase *bias_v,
        int m, int n_q, int n_kv, int k,
        int k_block_size,
        const MPIContext *ctx,
        int device_idx)
    {
        if (gemm_kernels_.size() != 3)
        {
            LOG_ERROR("[FusedGEMM::execute_q8_1_mixed_qkv] Called with "
                      << gemm_kernels_.size() << " kernels, expected 3 for QKV");
            return false;
        }

        if (!input_q8_1)
        {
            LOG_ERROR("[FusedGEMM::execute_q8_1_mixed_qkv] Input is null");
            return false;
        }

        if (!output_q || !output_k || !output_v)
        {
            LOG_ERROR("[FusedGEMM::execute_q8_1_mixed_qkv] One or more output buffers is null");
            return false;
        }

        // Validate K block size
        if (k_block_size != 32 && k_block_size != 64 && k_block_size != 128)
        {
            LOG_ERROR("[FusedGEMM::execute_q8_1_mixed_qkv] Invalid k_block_size "
                      << k_block_size << ", must be 32, 64, or 128");
            return false;
        }

        // Validate K dimension is divisible by block size
        if (n_kv % k_block_size != 0)
        {
            LOG_ERROR("[FusedGEMM::execute_q8_1_mixed_qkv] n_kv=" << n_kv
                                                                  << " must be divisible by k_block_size=" << k_block_size);
            return false;
        }

        LOG_DEBUG("[FusedGEMM::execute_q8_1_mixed_qkv] Mixed QKV: m=" << m << " n_q=" << n_q
                                                                      << " n_kv=" << n_kv << " k=" << k << " k_block_size=" << k_block_size);

        // Execute Q projection: Q8_1 → Q8_1
        bool success_q = gemm_kernels_[0]->multiply_with_precomputed_q8_1_to_q8_1(
            input_q8_1,
            output_q,
            m, n_q, k,
            bias_q,
            false, // No accumulation
            ctx, device_idx);

        if (!success_q)
        {
            LOG_ERROR("[FusedGEMM::execute_q8_1_mixed_qkv] Q projection failed");
            return false;
        }

        // Execute K projection: Q8_1 → Q16_1 (the key precision fix!)
        // This uses multiply_with_precomputed_q8_1_to_q16_1 which provides 256× better precision
        bool success_k = gemm_kernels_[1]->multiply_with_precomputed_q8_1_to_q16_1(
            input_q8_1,
            output_k,
            m, n_kv, k,
            k_block_size,
            bias_k,
            ctx, device_idx);

        if (!success_k)
        {
            LOG_ERROR("[FusedGEMM::execute_q8_1_mixed_qkv] K projection failed");
            return false;
        }

        // Execute V projection: Q8_1 → Q8_1
        bool success_v = gemm_kernels_[2]->multiply_with_precomputed_q8_1_to_q8_1(
            input_q8_1,
            output_v,
            m, n_kv, k,
            bias_v,
            false, // No accumulation
            ctx, device_idx);

        if (!success_v)
        {
            LOG_ERROR("[FusedGEMM::execute_q8_1_mixed_qkv] V projection failed");
            return false;
        }

        LOG_DEBUG("[FusedGEMM::execute_q8_1_mixed_qkv] All projections complete");
        return true;
    }

} // namespace llaminar2
