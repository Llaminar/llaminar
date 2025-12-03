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
        for (const auto *weight : weights)
        {
            // KernelFactory::getOrCreateGemm() returns cached ITensorGemm*
            auto *gemm = llaminar::v2::kernels::KernelFactory::getOrCreateGemm(weight);
            auto *quantised_gemm = dynamic_cast<gemm_v4::QuantisedGemmKernel *>(gemm);
            if (!quantised_gemm)
            {
                throw std::runtime_error("[FusedGEMM] Weight tensor returned non-QuantisedGemmKernel from KernelFactory::getOrCreateGemm()");
            }
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
        }

        return true;
    }

    bool FusedGEMM::execute(
        const float *input,
        float *output1, float *output2,
        const float *bias1, const float *bias2,
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
        const float *bias1, const float *bias2, const float *bias3,
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

} // namespace llaminar2
