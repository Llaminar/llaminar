/**
 * @file GDNProjectionStage.cpp
 * @brief Implementation of GDN 4-projection stage
 */

#include "GDNProjectionStage.h"
#include "../ComputeStageUtils.h"
#include "../../../loaders/PreparedWeightStore.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <array>
#include <optional>
#include <sstream>
#include <typeinfo>

namespace llaminar2
{
    namespace
    {
        std::optional<uint8_t> nativeVNNICodebook(ITensorGemm *kernel)
        {
            if (!kernel)
                return std::nullopt;

            DeviceNativeVNNIMatrixDesc desc;
            if (!kernel->exportNativeVNNIMatrixDesc(desc) || !desc.valid())
                return std::nullopt;
            return desc.codebook_id;
        }

        bool sameKernelType(const ITensorGemm *lhs, const ITensorGemm *rhs)
        {
            return lhs && rhs && typeid(*lhs) == typeid(*rhs);
        }

        bool fusedProjectionCompatible(
            ITensorGemm *lhs,
            ITensorGemm *rhs,
            const std::optional<uint8_t> &lhs_codebook,
            const std::optional<uint8_t> &rhs_codebook)
        {
            if (!sameKernelType(lhs, rhs))
                return false;
            if (lhs_codebook.has_value() != rhs_codebook.has_value())
                return false;
            if (lhs_codebook.has_value())
                return lhs_codebook.value() == rhs_codebook.value();
            return true;
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

        std::vector<ITensorGemm::TensorProjectionDesc> selectProjections(
            const std::vector<ITensorGemm::TensorProjectionDesc> &projections,
            const std::vector<size_t> &indices)
        {
            std::vector<ITensorGemm::TensorProjectionDesc> selected;
            selected.reserve(indices.size());
            for (size_t index : indices)
                selected.push_back(projections[index]);
            return selected;
        }

        void recordGDNProjectionRoute(
            const char *route,
            int m,
            int k,
            const std::vector<ITensorGemm::TensorProjectionDesc> &projections,
            const std::vector<size_t> &indices,
            const std::array<std::optional<uint8_t>, 4> &native_codebooks)
        {
            if (!PerfStatsCollector::isEnabled())
                return;

            std::ostringstream names;
            std::ostringstream ns;
            std::ostringstream codebooks;
            for (size_t i = 0; i < indices.size(); ++i)
            {
                const size_t index = indices[i];
                if (i > 0)
                {
                    names << '+';
                    ns << '+';
                    codebooks << '+';
                }
                names << (projections[index].name ? projections[index].name : "unnamed");
                ns << projections[index].n;
                if (index < native_codebooks.size() && native_codebooks[index].has_value())
                    codebooks << static_cast<unsigned>(native_codebooks[index].value());
                else
                    codebooks << "none";
            }

            PerfStatsCollector::addCounter(
                "kernel",
                "gdn_projection_route",
                1.0,
                {},
                {},
                PerfStatsCollector::Tags{
                    {"route", route ? route : "unknown"},
                    {"m", std::to_string(m)},
                    {"k", std::to_string(k)},
                    {"projections", std::to_string(indices.size())},
                    {"names", names.str()},
                    {"n", ns.str()},
                    {"codebooks", codebooks.str()},
                });
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

        std::array<std::optional<uint8_t>, 4> native_codebooks = {
            nativeVNNICodebook(gemm_qkv),
            nativeVNNICodebook(gemm_z),
            nativeVNNICodebook(gemm_a),
            nativeVNNICodebook(gemm_b)};

        const bool homogeneous_projection_kernels =
            fusedProjectionCompatible(gemm_qkv, gemm_z, native_codebooks[0], native_codebooks[1]) &&
            fusedProjectionCompatible(gemm_qkv, gemm_a, native_codebooks[0], native_codebooks[2]) &&
            fusedProjectionCompatible(gemm_qkv, gemm_b, native_codebooks[0], native_codebooks[3]);

        bool success = false;
        if (homogeneous_projection_kernels)
        {
            recordGDNProjectionRoute(
                "homogeneous_full",
                M,
                K,
                projections,
                {0, 1, 2, 3},
                native_codebooks);
            success = gemm_qkv->multiply_fused_tensor(A_base, projections, M, K, nullptr, bound_workspace_);
            if (!success)
            {
                LOG_ERROR("[GDNProjectionStage] Fused 4-projection GEMM failed");
                return false;
            }
        }
        else
        {
            LOG_DEBUG("[GDNProjectionStage] Mixed projection GEMM kernels; trying supported fused subgroups");

            std::vector<bool> completed(projections.size(), false);
            auto runFusedSubgroups = [&](bool require_native_compatibility) -> bool
            {
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    if (completed[i] || !projections[i].kernel ||
                        !projections[i].kernel->supports_fused_projection())
                    {
                        continue;
                    }

                    std::vector<size_t> group_indices;
                    group_indices.push_back(i);
                    for (size_t j = i + 1; j < projections.size(); ++j)
                    {
                        if (!completed[j] && projections[j].kernel &&
                            projections[j].kernel->supports_fused_projection())
                        {
                            const bool compatible =
                                require_native_compatibility
                                    ? (native_codebooks[i].has_value() &&
                                       native_codebooks[j].has_value() &&
                                       fusedProjectionCompatible(
                                          projections[i].kernel,
                                          projections[j].kernel,
                                          native_codebooks[i],
                                          native_codebooks[j]))
                                    : sameKernelType(projections[i].kernel, projections[j].kernel);
                            if (compatible)
                                group_indices.push_back(j);
                        }
                    }

                    if (group_indices.size() < 2)
                        continue;

                    auto group = selectProjections(projections, group_indices);
                    recordGDNProjectionRoute(
                        require_native_compatibility ? "native_subgroup" : "same_kernel_mixed_codebook_subgroup",
                        M,
                        K,
                        projections,
                        group_indices,
                        native_codebooks);
                    if (!group.front().kernel->multiply_fused_tensor(
                            A_base, group, M, K, nullptr, bound_workspace_))
                    {
                        LOG_ERROR("[GDNProjectionStage] Fused projection subgroup failed for "
                                  << (group.front().name ? group.front().name : "unnamed")
                                  << " group_size=" << group.size());
                        return false;
                    }

                    for (size_t index : group_indices)
                        completed[index] = true;
                }
                return true;
            };

            if (!runFusedSubgroups(/*require_native_compatibility=*/true))
                return false;
            if (!runFusedSubgroups(/*require_native_compatibility=*/false))
                return false;

            std::vector<ITensorGemm::TensorProjectionDesc> remaining;
            remaining.reserve(projections.size());
            for (size_t i = 0; i < projections.size(); ++i)
            {
                if (!completed[i])
                {
                    recordGDNProjectionRoute(
                        "fallback_single",
                        M,
                        K,
                        projections,
                        {i},
                        native_codebooks);
                    remaining.push_back(projections[i]);
                }
            }

            success = remaining.empty() ||
                      multiplyProjectionFallback(A_base, remaining, M, K, bound_workspace_);
        }

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
        StageBufferRequirements reqs;

        if (!params_.input ||
            !params_.w_qkv || !params_.w_z || !params_.w_a || !params_.w_b)
        {
            return reqs;
        }

        const BufferTensorType input_type = toBufferTensorType(params_.input->native_type());
        reqs.addInput(
            "input",
            {static_cast<size_t>(params_.m), static_cast<size_t>(params_.k)},
            input_type);

        reqs.addWeight(
            "w_qkv",
            {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_qkv)},
            toBufferTensorType(params_.w_qkv->native_type()));
        reqs.addWeight(
            "w_z",
            {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_z)},
            toBufferTensorType(params_.w_z->native_type()));
        reqs.addWeight(
            "w_a",
            {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_a)},
            toBufferTensorType(params_.w_a->native_type()));
        reqs.addWeight(
            "w_b",
            {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n_b)},
            toBufferTensorType(params_.w_b->native_type()));

        const auto addOutput = [&](const char *name, const ITensor *tensor, int n)
        {
            if (!tensor || n <= 0)
            {
                return;
            }
            reqs.addOutput(
                name,
                {static_cast<size_t>(params_.m), static_cast<size_t>(n)},
                toBufferTensorType(tensor->native_type()));
        };

        addOutput("output_qkv", params_.output_qkv, params_.n_qkv);
        addOutput("output_z", params_.output_z, params_.n_z);
        addOutput("output_a", params_.output_a, params_.n_a);
        addOutput("output_b", params_.output_b, params_.n_b);

        return reqs;
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

        const int workspace_m = (m > 0) ? m : params_.m;
        const int workspace_k = (k > 0) ? k : params_.k;

        WorkspaceRequirements combined;
        auto mergeFrom = [&](const ITensor *weight, ITensorGemm *&cached, const char *name, int projection_n)
        {
            auto *gemm = self->resolveGemm(weight, cached, name);
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm))
            {
                const int workspace_n = (projection_n > 0) ? projection_n : n;
                combined.merge(consumer->getWorkspaceRequirements(workspace_m, workspace_n, workspace_k));
            }
        };

        mergeFrom(self->params_.w_qkv, self->params_.gemm_qkv, "w_qkv", self->params_.n_qkv);
        mergeFrom(self->params_.w_z, self->params_.gemm_z, "w_z", self->params_.n_z);
        mergeFrom(self->params_.w_a, self->params_.gemm_a, "w_a", self->params_.n_a);
        mergeFrom(self->params_.w_b, self->params_.gemm_b, "w_b", self->params_.n_b);
        addCudaConcurrentDecodeGemvSideStreamWorkspace(
            combined, self->params_.device_id, workspace_m, /*projection_count=*/4);
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
