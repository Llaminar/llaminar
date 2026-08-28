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

#include <algorithm>
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

            /*
             * Native-VNNI engines are compatible only when the decoder
             * codebook agrees.  A single grouped launch is driven by the seed
             * engine, so accepting a different codebook here would make that
             * engine reinterpret the following projection's packed bytes.
             */
            if (lhs_codebook.has_value() != rhs_codebook.has_value())
                return false;
            if (lhs_codebook.has_value())
                return lhs_codebook.value() == rhs_codebook.value();

            /*
             * Floating GEMM implementations use one C++ class for FP16,
             * BF16, and FP32 weights.  Type identity alone therefore is not a
             * complete fusion key.  Query the public immutable weight view so
             * a BF16 QKV/Z pair and an FP32 alpha/beta pair become two valid
             * fused subgroups instead of asking the BF16 seed kernel to decode
             * FP32 bytes.  Geometry intentionally is not part of this key:
             * grouped implementations split different N widths internally
             * while retaining one physical weight format per subgroup.
             */
            ContiguousFloatingPointWeightDescriptor lhs_floating;
            ContiguousFloatingPointWeightDescriptor rhs_floating;
            const bool lhs_exports_floating =
                lhs->exportContiguousFloatingPointWeights(lhs_floating) &&
                lhs_floating.valid();
            const bool rhs_exports_floating =
                rhs->exportContiguousFloatingPointWeights(rhs_floating) &&
                rhs_floating.valid();
            if (lhs_exports_floating != rhs_exports_floating)
                return false;
            if (lhs_exports_floating)
                return lhs_floating.type == rhs_floating.type;

            return true;
        }

        bool validateProjectionTensorCapacity(
            const TensorBase *tensor,
            const char *name,
            int rows_required,
            int cols_required,
            const DeviceId &device_id)
        {
            if (!tensor)
            {
                LOG_ERROR("[GDNProjectionStage] Missing tensor for " << name
                                                                     << " on " << device_id.toString());
                return false;
            }
            if (rows_required <= 0 || cols_required <= 0)
            {
                LOG_ERROR("[GDNProjectionStage] Invalid projection extent for " << name
                                                                                << " on " << device_id.toString()
                                                                                << ": rows=" << rows_required
                                                                                << " cols=" << cols_required);
                return false;
            }

            const size_t required_rows = static_cast<size_t>(rows_required);
            const size_t required_cols = static_cast<size_t>(cols_required);
            const size_t required_elements = required_rows * required_cols;
            const bool shape_ok = tensor->rows() >= required_rows && tensor->cols() >= required_cols;
            const bool numel_ok = tensor->numel() >= required_elements;
            if (shape_ok && numel_ok)
                return true;

            LOG_ERROR("[GDNProjectionStage] Tensor capacity is too small for " << name
                                                                               << " on " << device_id.toString()
                                                                               << ": shape=" << tensor->rows()
                                                                               << "x" << tensor->cols()
                                                                               << " numel=" << tensor->numel()
                                                                               << " bytes=" << tensor->size_bytes()
                                                                               << " dtype=" << tensor->dtype_name()
                                                                               << " gpu_ptr=" << tensor->gpu_data_ptr()
                                                                               << " required_shape=" << required_rows
                                                                               << "x" << required_cols
                                                                               << " required_elements=" << required_elements);
            return false;
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
            if (!PerfStatsCollector::isDomainEnabled("kernel"))
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

    void GDNProjectionStage::clearCachedGemmStreams()
    {
        if (params_.gemm_qkv)
            params_.gemm_qkv->clearGPUStreamBinding();
        if (params_.gemm_z)
            params_.gemm_z->clearGPUStreamBinding();
        if (params_.gemm_a)
            params_.gemm_a->clearGPUStreamBinding();
        if (params_.gemm_b)
            params_.gemm_b->clearGPUStreamBinding();
    }

    void GDNProjectionStage::resetSessionState()
    {
        IComputeStage::resetSessionState();
        clearCachedGemmStreams();
    }

    void GDNProjectionStage::resetSessionStatePreservingCapturedReplay()
    {
        IComputeStage::resetSessionStatePreservingCapturedReplay();
        clearCachedGemmStreams();
    }

    void GDNProjectionStage::resetSessionStatePreservingLazyInitialization()
    {
        resetSessionStatePreservingCapturedReplay();
    }

    bool GDNProjectionStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        (void)ctx;
        if (!stream)
        {
            LOG_ERROR("[GDNProjectionStage] Graph launch preparation requires "
                      "the exact non-null producer stream");
            return false;
        }
        setGPUStream(stream);

        auto *gemm_qkv = resolveGemm(
            params_.w_qkv, params_.gemm_qkv, "w_qkv");
        auto *gemm_z = resolveGemm(params_.w_z, params_.gemm_z, "w_z");
        auto *gemm_a = resolveGemm(params_.w_a, params_.gemm_a, "w_a");
        auto *gemm_b = resolveGemm(params_.w_b, params_.gemm_b, "w_b");
        if (!gemm_qkv || !gemm_z || !gemm_a || !gemm_b)
            return false;

        constexpr size_t projection_count = 4;
        const std::array<ITensorGemm *, projection_count> kernels = {
            gemm_qkv, gemm_z, gemm_a, gemm_b};
        for (ITensorGemm *kernel : kernels)
        {
            bindStageStream(kernel);
            if (!kernel->prepareFusedProjectionGraphCapture(
                    projection_count))
            {
                LOG_ERROR("[GDNProjectionStage] Failed to provision persistent "
                          "four-projection resources before graph capture"
                          << " device=" << params_.device_id.toString());
                return false;
            }
        }
        return true;
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

        if (!validateProjectionTensorCapacity(A_base, "input", M, K, params_.device_id) ||
            !validateProjectionTensorCapacity(C_qkv, "output_qkv", M, params_.n_qkv, params_.device_id) ||
            !validateProjectionTensorCapacity(C_z, "output_z", M, params_.n_z, params_.device_id) ||
            !validateProjectionTensorCapacity(C_a, "output_a", M, params_.n_a, params_.device_id) ||
            !validateProjectionTensorCapacity(C_b, "output_b", M, params_.n_b, params_.device_id))
        {
            return false;
        }

        // Set GPU stream on all engines (no-op for CPU)
        bindStageStream(gemm_qkv);
        bindStageStream(gemm_z);
        bindStageStream(gemm_a);
        bindStageStream(gemm_b);

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

        if (params_.force_decode_equivalent_verifier_prefill && M > 1)
        {
            /**
             * Group verifier projections only when the prepared GEMM engines
             * can legally share one fused decode path.  The verifier rows must
             * be bitwise-equivalent to running the decode stage once per row;
             * mixing native VNNI projections with different codebooks under the
             * first projection's kernel can decode the later projections with
             * the wrong format even though the C++ kernel type is identical.
             */
            auto try_grouped_verifier_projections =
                [&](const std::vector<ITensorGemm::TensorProjectionDesc> &all_projections) -> bool
            {
                std::vector<bool> completed(all_projections.size(), false);

                for (size_t i = 0; i < all_projections.size(); ++i)
                {
                    if (completed[i] || !all_projections[i].kernel)
                        continue;

                    std::vector<size_t> group_indices;
                    group_indices.push_back(i);
                    for (size_t j = i + 1; j < all_projections.size(); ++j)
                    {
                        if (!completed[j] && all_projections[j].kernel &&
                            fusedProjectionCompatible(
                                all_projections[i].kernel,
                                all_projections[j].kernel,
                                native_codebooks[i],
                                native_codebooks[j]))
                        {
                            group_indices.push_back(j);
                        }
                    }

                    auto group = selectProjections(all_projections, group_indices);
                    if (!group.front().kernel->multiply_fused_verifier_rows_decode_equivalent(
                            A_base,
                            group,
                            M,
                            K,
                            nullptr,
                            bound_workspace_))
                    {
                        LOG_ERROR("[GDNProjectionStage] Grouped decode-equivalent verifier projection "
                                  "subgroup failed for "
                                  << (group.front().name ? group.front().name : "unnamed")
                                  << " group_size=" << group.size());
                        return false;
                    }

                    recordGDNProjectionRoute(
                        "grouped_decode_equivalent_verifier_subgroup",
                        M,
                        K,
                        all_projections,
                        group_indices,
                        native_codebooks);

                    for (size_t index : group_indices)
                        completed[index] = true;
                }

                return std::all_of(completed.begin(), completed.end(), [](bool done) { return done; });
            };

            const bool homogeneous_projection_group =
                fusedProjectionCompatible(gemm_qkv, gemm_z, native_codebooks[0], native_codebooks[1]) &&
                fusedProjectionCompatible(gemm_qkv, gemm_a, native_codebooks[0], native_codebooks[2]) &&
                fusedProjectionCompatible(gemm_qkv, gemm_b, native_codebooks[0], native_codebooks[3]);

            if ((homogeneous_projection_group &&
                 gemm_qkv->multiply_fused_verifier_rows_decode_equivalent(
                     A_base,
                     projections,
                     M,
                     K,
                     nullptr,
                     bound_workspace_)) ||
                try_grouped_verifier_projections(projections))
            {
                recordGDNProjectionRoute(
                    homogeneous_projection_group
                        ? "grouped_decode_equivalent_verifier"
                        : "grouped_decode_equivalent_verifier_mixed",
                    M,
                    K,
                    projections,
                    {0, 1, 2, 3},
                    native_codebooks);
                return true;
            }

            LOG_ERROR("[GDNProjectionStage] Verifier projection requires grouped "
                      "decode-equivalent kernels for every backend; serial row replay "
                      "is no longer an accepted Phase 9.8 path on "
                      << params_.device_id.toString());
            return false;
        }

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
            LOG_TRACE("[GDNProjectionStage] Mixed projection GEMM kernels; trying supported fused subgroups");

            std::vector<bool> completed(projections.size(), false);
            auto runFusedSubgroups = [&](bool require_native_compatibility) -> bool
            {
                for (size_t i = 0; i < projections.size(); ++i)
                {
                    if (completed[i])
                        continue;
                    if (!projections[i].kernel)
                        return false;
                    if (require_native_compatibility &&
                        !native_codebooks[i].has_value())
                        continue;
                    if (!projections[i].kernel->supports_fused_projection())
                    {
                        LOG_ERROR("[GDNProjectionStage] Projection bundle member "
                                  << (projections[i].name
                                          ? projections[i].name
                                          : "unnamed")
                                  << " has no first-class fused implementation");
                        return false;
                    }

                    std::vector<size_t> group_indices;
                    group_indices.push_back(i);
                    for (size_t j = i + 1; j < projections.size(); ++j)
                    {
                        if (!completed[j] && projections[j].kernel &&
                            projections[j].kernel->supports_fused_projection() &&
                            (!require_native_compatibility ||
                             native_codebooks[j].has_value()))
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
                                    : fusedProjectionCompatible(
                                          projections[i].kernel,
                                          projections[j].kernel,
                                          native_codebooks[i],
                                          native_codebooks[j]);
                            if (compatible)
                                group_indices.push_back(j);
                        }
                    }

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

            success = std::all_of(
                completed.begin(),
                completed.end(),
                [](bool projection_complete)
                {
                    return projection_complete;
                });
            if (!success)
            {
                LOG_ERROR("[GDNProjectionStage] Projection bundle left an "
                          "unexecuted member; per-projection replay is forbidden");
            }
        }

        if (!success)
        {
            LOG_ERROR("[GDNProjectionStage] 4-projection GEMM failed");
            return false;
        }

        LOG_TRACE("[GDNProjectionStage] Executed: M=" << M << " K=" << K
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

        // The normalized hidden state is the only activation input. Projection
        // weights are declared through the dedicated weight channel because
        // production GPU execution consumes immutable PreparedWeightStore
        // handles; their source tensors may intentionally release raw host data
        // after preparation and therefore are not valid activation buffers.
        if (params_.input)
            info.addInput("input", params_.input, rows, k);
        if (params_.w_qkv)
            info.addWeight("w_qkv", params_.w_qkv);
        if (params_.w_z)
            info.addWeight("w_z", params_.w_z);
        if (params_.w_a)
            info.addWeight("w_a", params_.w_a);
        if (params_.w_b)
            info.addWeight("w_b", params_.w_b);

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
        // Every GDN projection consumes a store-owned prepared representation.
        if (params_.w_qkv)
            contract.addPreparedWeight(
                const_cast<ITensor *>(params_.w_qkv),
                params_.prepared_store,
                params_.prepared_ref_qkv.value_or(PreparedWeightRef{}));
        if (params_.w_z)
            contract.addPreparedWeight(
                const_cast<ITensor *>(params_.w_z),
                params_.prepared_store,
                params_.prepared_ref_z.value_or(PreparedWeightRef{}));
        if (params_.w_a)
            contract.addPreparedWeight(
                const_cast<ITensor *>(params_.w_a),
                params_.prepared_store,
                params_.prepared_ref_a.value_or(PreparedWeightRef{}));
        if (params_.w_b)
            contract.addPreparedWeight(
                const_cast<ITensor *>(params_.w_b),
                params_.prepared_store,
                params_.prepared_ref_b.value_or(PreparedWeightRef{}));
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

        const std::array<int, 4> fused_columns = {
            self->params_.n_qkv > 0 ? self->params_.n_qkv : n,
            self->params_.n_z > 0 ? self->params_.n_z : n,
            self->params_.n_a > 0 ? self->params_.n_a : n,
            self->params_.n_b > 0 ? self->params_.n_b : n};
        if (auto *anchor = dynamic_cast<IWorkspaceConsumer *>(self->params_.gemm_qkv))
        {
            anchor->appendFusedProjectionWorkspaceRequirements(
                combined, workspace_m, fused_columns, workspace_k);
        }
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
                LOG_TRACE("[GDNProjectionStage] Bound workspace to " << name << " kernel");
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
