/**
 * @file GDNProjectionStage.cpp
 * @brief Implementation of GDN 4-projection stage
 */

#include "GDNProjectionStage.h"
#include "../../../loaders/PreparedWeightStore.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/Logger.h"

#include <typeinfo>

namespace llaminar2
{
    namespace
    {
        bool sameKernelType(const ITensorGemm *lhs, const ITensorGemm *rhs)
        {
            return lhs && rhs && typeid(*lhs) == typeid(*rhs);
        }

        bool multiplyProjectionFallback(
            const TensorBase *input,
            const std::vector<ITensorGemm::TensorProjectionDesc> &projections,
            int m,
            int k,
            DeviceWorkspaceManager *workspace)
        {
            for (const auto &projection : projections)
            {
                if (!projection.kernel || !projection.output)
                    return false;

                const bool ok = projection.kernel->multiply_tensor(
                    input,
                    projection.output,
                    m,
                    projection.n,
                    k,
                    true,
                    1.0f,
                    0.0f,
                    projection.bias,
                    nullptr,
                    -1,
                    workspace);
                if (!ok)
                {
                    LOG_ERROR("[GDNProjectionStage] Projection fallback failed for "
                              << (projection.name ? projection.name : "unnamed"));
                    return false;
                }
            }
            return true;
        }
    } // namespace

    GDNProjectionStage::GDNProjectionStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool GDNProjectionStage::validatePreparedWeights(std::string *error) const
    {
        auto fail = [error](const std::string &message)
        {
            if (error)
                *error = message;
            return false;
        };

        if (!params_.w_qkv && !params_.w_z && !params_.w_a && !params_.w_b)
        {
            if (error)
                error->clear();
            return true;
        }

        if (!params_.prepared_store)
            return fail("PreparedWeightStore is required for GDNProjectionStage weights");

        auto check = [&](const char *name, const TensorBase *weight, const std::optional<PreparedWeightRef> &ref)
        {
            if (!weight)
                return true;
            if (!ref.has_value())
                return fail(std::string("missing PreparedWeightRef for ") + name);
            if (!params_.prepared_store->contains(ref.value()))
                return fail(std::string("PreparedWeightStore does not contain ref for ") + name);
            return true;
        };

        if (!check("w_qkv", requireTensorBase(params_.w_qkv, "w_qkv"), params_.prepared_ref_qkv))
            return false;
        if (!check("w_z", requireTensorBase(params_.w_z, "w_z"), params_.prepared_ref_z))
            return false;
        if (!check("w_a", requireTensorBase(params_.w_a, "w_a"), params_.prepared_ref_a))
            return false;
        if (!check("w_b", requireTensorBase(params_.w_b, "w_b"), params_.prepared_ref_b))
            return false;

        if (error)
            error->clear();
        return true;
    }

    ITensorGemm *GDNProjectionStage::resolveGemm(
        const ITensor *weight, ITensorGemm *&cached, const char *name)
    {
        if (cached)
            return cached;

        auto *B_base = requireTensorBase(weight, name);
        if (!B_base)
            return nullptr;

        if (!params_.prepared_store)
        {
            LOG_ERROR("[GDNProjectionStage] PreparedWeightStore is required for " << name);
            return nullptr;
        }

        const std::optional<PreparedWeightRef> *ref = nullptr;
        if (weight == params_.w_qkv)
            ref = &params_.prepared_ref_qkv;
        else if (weight == params_.w_z)
            ref = &params_.prepared_ref_z;
        else if (weight == params_.w_a)
            ref = &params_.prepared_ref_a;
        else if (weight == params_.w_b)
            ref = &params_.prepared_ref_b;

        if (!ref || !ref->has_value())
        {
            LOG_ERROR("[GDNProjectionStage] PreparedWeightRef is required for " << name);
            return nullptr;
        }

        auto *gemm = params_.prepared_store->gemmKernel(ref->value());
        if (!gemm)
        {
            LOG_ERROR("[GDNProjectionStage] PreparedWeightRef was provided but no GEMM kernel was found in PreparedWeightStore for " << name);
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

        // Resolve all 4 GEMM engines (lazy, cached after first call)
        auto *gemm_qkv = resolveGemm(params_.w_qkv, params_.gemm_qkv, "w_qkv");
        auto *gemm_z = resolveGemm(params_.w_z, params_.gemm_z, "w_z");
        auto *gemm_a = resolveGemm(params_.w_a, params_.gemm_a, "w_a");
        auto *gemm_b = resolveGemm(params_.w_b, params_.gemm_b, "w_b");
        if (!gemm_qkv || !gemm_z || !gemm_a || !gemm_b)
            return false;

        auto *C_qkv = asTensorBase(params_.output_qkv, "output_qkv");
        auto *C_z = asTensorBase(params_.output_z, "output_z");
        auto *C_a = asTensorBase(params_.output_a, "output_a");
        auto *C_b = asTensorBase(params_.output_b, "output_b");
        if (!C_qkv || !C_z || !C_a || !C_b)
            return false;

        // Set GPU stream on all engines (no-op for CPU)
        gemm_qkv->setGPUStream(gpuStream());
        gemm_z->setGPUStream(gpuStream());
        gemm_a->setGPUStream(gpuStream());
        gemm_b->setGPUStream(gpuStream());

        // Fused 4-projection GEMM: quantizes input once, single OMP region
        // for decode (M=1). For prefill (M>1), falls back to sequential GEMMs
        // inside the kernel but still avoids 4 separate stage-level dispatches.
        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {gemm_qkv, C_qkv, params_.n_qkv, nullptr, "qkv"},
            {gemm_z, C_z, params_.n_z, nullptr, "z"},
            {gemm_a, C_a, params_.n_a, nullptr, "alpha"},
            {gemm_b, C_b, params_.n_b, nullptr, "beta"}};

        const bool homogeneous_projection_kernels =
            sameKernelType(gemm_qkv, gemm_z) &&
            sameKernelType(gemm_qkv, gemm_a) &&
            sameKernelType(gemm_qkv, gemm_b);

        bool success = false;
        if (homogeneous_projection_kernels)
        {
            success = gemm_qkv->multiply_fused_tensor(A_base, projections, M, K, nullptr, bound_workspace_);
            if (!success)
                LOG_WARN("[GDNProjectionStage] Fused 4-projection GEMM failed; retrying per-projection fallback");
        }
        else
        {
            LOG_DEBUG("[GDNProjectionStage] Mixed projection GEMM kernels; using per-projection fallback");
        }

        if (!success)
            success = multiplyProjectionFallback(A_base, projections, M, K, bound_workspace_);

        if (!success)
        {
            LOG_ERROR("[GDNProjectionStage] 4-projection GEMM failed");
            return false;
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
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#ifdef HAVE_CUDA
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#ifdef HAVE_ROCM
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
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
        // Model weights are not arena-managed
        if (params_.w_qkv)
            contract.addWeight(const_cast<ITensor *>(params_.w_qkv));
        if (params_.w_z)
            contract.addWeight(const_cast<ITensor *>(params_.w_z));
        if (params_.w_a)
            contract.addWeight(const_cast<ITensor *>(params_.w_a));
        if (params_.w_b)
            contract.addWeight(const_cast<ITensor *>(params_.w_b));
        return contract;
    }

    // =========================================================================
    // IWorkspaceConsumerStage — Multi-kernel workspace binding (4 GEMM kernels)
    // =========================================================================

    IWorkspaceConsumer *GDNProjectionStage::getKernelAsWorkspaceConsumer()
    {
        // Return QKV kernel (largest) for workspace requirements sizing
        auto *gemm = resolveGemm(params_.w_qkv, params_.gemm_qkv, "w_qkv");
        return dynamic_cast<IWorkspaceConsumer *>(gemm);
    }

    WorkspaceRequirements GDNProjectionStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        // Must merge requirements from ALL 4 GEMM kernels, not just QKV.
        // Each ROCm kernel has a unique slice_id_ generating per-instance buffer
        // names (e.g., gemm_temp_a_fp32_<id>). Reporting only QKV's requirements
        // leaves the Z, A, B kernels' buffers unallocated.
        auto *self = const_cast<GDNProjectionStage *>(this);

        WorkspaceRequirements combined;
        auto mergeFrom = [&](const ITensor *weight, ITensorGemm *&cached, const char *name)
        {
            auto *gemm = self->resolveGemm(weight, cached, name);
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm))
            {
                combined.merge(consumer->getWorkspaceRequirements(m, n, k));
            }
        };

        mergeFrom(self->params_.w_qkv, self->params_.gemm_qkv, "w_qkv");
        mergeFrom(self->params_.w_z, self->params_.gemm_z, "w_z");
        mergeFrom(self->params_.w_a, self->params_.gemm_a, "w_a");
        mergeFrom(self->params_.w_b, self->params_.gemm_b, "w_b");
        return combined;
    }

    void GDNProjectionStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        // Resolve all 4 GEMM kernels and bind workspace to each
        auto bindOne = [&](const ITensor *weight, ITensorGemm *&cached, const char *name)
        {
            auto *gemm = resolveGemm(weight, cached, name);
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm))
            {
                consumer->bindWorkspace(workspace);
                LOG_DEBUG("[GDNProjectionStage] Bound workspace to " << name << " kernel");
            }
        };

        bindOne(params_.w_qkv, params_.gemm_qkv, "w_qkv");
        bindOne(params_.w_z, params_.gemm_z, "w_z");
        bindOne(params_.w_a, params_.gemm_a, "w_a");
        bindOne(params_.w_b, params_.gemm_b, "w_b");

        bound_workspace_ = workspace;
    }

    void GDNProjectionStage::unbindWorkspace()
    {
        auto unbindOne = [](ITensorGemm *gemm)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm))
                consumer->unbindWorkspace();
        };

        unbindOne(params_.gemm_qkv);
        unbindOne(params_.gemm_z);
        unbindOne(params_.gemm_a);
        unbindOne(params_.gemm_b);

        bound_workspace_ = nullptr;
    }

} // namespace llaminar2
