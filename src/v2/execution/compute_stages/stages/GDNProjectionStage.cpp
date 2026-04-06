/**
 * @file GDNProjectionStage.cpp
 * @brief Implementation of GDN 4-projection stage
 */

#include "GDNProjectionStage.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    GDNProjectionStage::GDNProjectionStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    ITensorGemm *GDNProjectionStage::resolveGemm(
        const ITensor *weight, ITensorGemm *&cached, const char *name)
    {
        if (cached)
            return cached;

        auto *B_base = requireTensorBase(weight, name);
        if (!B_base)
            return nullptr;

        auto *prepared = KernelFactory::getOrCreatePreparedGemmWeights(
            B_base, params_.device_id);
        auto *gemm = KernelFactory::getOrCreateGemmEngine(prepared);
        if (!gemm)
        {
            LOG_ERROR("[GDNProjectionStage] Failed to resolve GEMM kernel for " << name);
            return nullptr;
        }
        cached = gemm;
        return gemm;
    }

    bool GDNProjectionStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "GDNProjectionStage"))
            return false;

        if (!ensureRequiredPointers("GDNProjectionStage",
                                    {{"input", params_.input},
                                     {"w_qkv", params_.w_qkv},
                                     {"output_qkv", params_.output_qkv},
                                     {"w_z", params_.w_z},
                                     {"output_z", params_.output_z},
                                     {"w_a", params_.w_a},
                                     {"output_a", params_.output_a},
                                     {"w_b", params_.w_b},
                                     {"output_b", params_.output_b}}))
            return false;

        const int M = params_.m;
        const int K = params_.k;
        auto *A_base = requireTensorBasePtr(params_.input, "input");
        if (!A_base)
            return false;

        // Execute each projection via its GEMM kernel (lazy resolution from weight tensors)
        // QKV projection
        {
            auto *gemm = resolveGemm(params_.w_qkv, params_.gemm_qkv, "w_qkv");
            if (!gemm)
                return false;
            auto *C_base = asTensorBase(params_.output_qkv, "output_qkv");
            if (!gemm->multiply_tensor(A_base, C_base, M, params_.n_qkv, K))
            {
                LOG_ERROR("[GDNProjectionStage] QKV GEMM failed");
                return false;
            }
        }

        // Z projection
        {
            auto *gemm = resolveGemm(params_.w_z, params_.gemm_z, "w_z");
            if (!gemm)
                return false;
            auto *C_base = asTensorBase(params_.output_z, "output_z");
            if (!gemm->multiply_tensor(A_base, C_base, M, params_.n_z, K))
            {
                LOG_ERROR("[GDNProjectionStage] Z GEMM failed");
                return false;
            }
        }

        // A projection
        {
            auto *gemm = resolveGemm(params_.w_a, params_.gemm_a, "w_a");
            if (!gemm)
                return false;
            auto *C_base = asTensorBase(params_.output_a, "output_a");
            if (!gemm->multiply_tensor(A_base, C_base, M, params_.n_a, K))
            {
                LOG_ERROR("[GDNProjectionStage] A GEMM failed");
                return false;
            }
        }

        // B projection
        {
            auto *gemm = resolveGemm(params_.w_b, params_.gemm_b, "w_b");
            if (!gemm)
                return false;
            auto *C_base = asTensorBase(params_.output_b, "output_b");
            if (!gemm->multiply_tensor(A_base, C_base, M, params_.n_b, K))
            {
                LOG_ERROR("[GDNProjectionStage] B GEMM failed");
                return false;
            }
        }

        LOG_DEBUG("[GDNProjectionStage] Executed: M=" << M << " K=" << K
                                                      << " n_qkv=" << params_.n_qkv
                                                      << " n_z=" << params_.n_z
                                                      << " n_a=" << params_.n_a
                                                      << " n_b=" << params_.n_b);

        return true;
    }

    size_t GDNProjectionStage::estimatedFlops() const
    {
        // Each GEMM: 2*M*N*K flops
        const size_t M = static_cast<size_t>(params_.m);
        const size_t K = static_cast<size_t>(params_.k);
        return 2 * M * K * (params_.n_qkv + params_.n_z + params_.n_a + params_.n_b);
    }

    size_t GDNProjectionStage::estimatedMemoryBytes() const
    {
        const size_t M = static_cast<size_t>(params_.m);
        // Read input once, read 4 weight matrices, write 4 outputs
        const size_t input_bytes = M * params_.k * sizeof(float);
        const size_t output_bytes = M * (params_.n_qkv + params_.n_z + params_.n_a + params_.n_b) * sizeof(float);
        return input_bytes + output_bytes;
    }

    bool GDNProjectionStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageDumpInfo GDNProjectionStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Use actual dimensions (params_.m = total_tokens), not buffer capacity
        const size_t rows = static_cast<size_t>(params_.m);
        const size_t k = static_cast<size_t>(params_.k);

        // Inputs: normalized hidden state + 4 weight matrices
        if (params_.input)
            info.addInput("input", params_.input, rows, k);
        if (params_.w_qkv)
            info.addInput("w_qkv", params_.w_qkv,
                          params_.w_qkv->shape()[0], params_.w_qkv->shape()[1]);
        if (params_.w_z)
            info.addInput("w_z", params_.w_z,
                          params_.w_z->shape()[0], params_.w_z->shape()[1]);
        if (params_.w_a)
            info.addInput("w_a", params_.w_a,
                          params_.w_a->shape()[0], params_.w_a->shape()[1]);
        if (params_.w_b)
            info.addInput("w_b", params_.w_b,
                          params_.w_b->shape()[0], params_.w_b->shape()[1]);

        // Outputs: 4 projection results
        if (params_.output_qkv)
            info.addOutput("output_qkv", params_.output_qkv, rows,
                           static_cast<size_t>(params_.n_qkv));
        if (params_.output_z)
            info.addOutput("output_z", params_.output_z, rows,
                           static_cast<size_t>(params_.n_z));
        if (params_.output_a)
            info.addOutput("output_a", params_.output_a, rows,
                           static_cast<size_t>(params_.n_a));
        if (params_.output_b)
            info.addOutput("output_b", params_.output_b, rows,
                           static_cast<size_t>(params_.n_b));

        return info;
    }

    StageBufferRequirements GDNProjectionStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract GDNProjectionStage::bufferContract() const
    {
        StageBufferContract contract;
        if (params_.input_buffer_id)
            contract.addInput(*params_.input_buffer_id);
        if (params_.output_qkv_buffer_id)
            contract.addOutput(*params_.output_qkv_buffer_id);
        if (params_.output_z_buffer_id)
            contract.addOutput(*params_.output_z_buffer_id);
        if (params_.output_a_buffer_id)
            contract.addOutput(*params_.output_a_buffer_id);
        if (params_.output_b_buffer_id)
            contract.addOutput(*params_.output_b_buffer_id);
        return contract;
    }

} // namespace llaminar2
