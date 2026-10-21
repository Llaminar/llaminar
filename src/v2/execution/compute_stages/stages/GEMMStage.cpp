/**
 * @file GEMMStage.cpp
 * @brief Prepared participant-local projections with ordered output publication.
 *
 * Verifier scopes borrow device counts; adapters retain physical scratch and
 * exact stream ordering. Fused SwiGLU consumes two [M,K] operands before the
 * projection produces [M,N]. Diagnostic descriptors preserve those distinct
 * dimensions so a stage dump cannot truncate or overread the gate operand.
 */

#include "GEMMStage.h"
#include "kernels/common/FloatingProjectionRows.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../loaders/PreparedWeightStore.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace llaminar2
{
    WorkspaceRequirements GEMMStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        const int rows = std::max({1, m, params_.m});
        if(params_.quantized_operand_precision==QuantizedActivationOperandPrecision::TwoTermQ8) {
            auto *kernel=const_cast<GEMMStage*>(this)->getKernelAsWorkspaceConsumer();
            auto *gemm=dynamic_cast<ITensorGemm*>(kernel);
            if(!gemm)throw std::logic_error("Two-term Q8 requires an actual prepared tensor GEMM");
            const int columns=params_.output_range.empty()?params_.n:static_cast<int>(params_.output_range.size());
            return gemm->q8ResidualCorrectionWorkspaceRequirements(rows,columns,params_.k);
        }
        auto requirements = IWorkspaceConsumerStage::getWorkspaceRequirements(rows, n, k);
        // Execution selects the transform from the gate operand. The legacy
        // diagnostic flag is not an independent source of operation intent.
        if (params_.gate_input)
            if (auto *consumer = const_cast<GEMMStage *>(this)->getKernelAsWorkspaceConsumer())
                consumer->appendSwiGLUWorkspaceRequirements(
                    requirements, rows, params_.k > 0 ? params_.k : k);
        if (params_.beta != 0.0f)
            if (auto *consumer = const_cast<GEMMStage *>(this)->getKernelAsWorkspaceConsumer())
                consumer->appendOutputAccumulationWorkspaceRequirements(
                    requirements, rows, params_.n > 0 ? params_.n : n);
        return requirements;
    }

    namespace
    {
        bool validateMatrixExtent(
            const TensorBase *tensor,
            const char *tensor_name,
            int required_rows,
            int required_cols)
        {
            if (!tensor)
            {
                LOG_ERROR("[GEMMStage] " << tensor_name << " is not a TensorBase");
                return false;
            }

            const auto &shape = tensor->shape();
            const size_t rows = shape.empty() ? 0 : shape[0];
            const size_t cols = shape.size() < 2 ? 1 : shape[1];
            if (rows < static_cast<size_t>(required_rows) ||
                cols < static_cast<size_t>(required_cols))
            {
                LOG_ERROR("[GEMMStage] Tensor extent mismatch for " << tensor_name
                                                                    << ": required >= ["
                                                                    << required_rows << ", " << required_cols
                                                                    << "], actual=[" << rows << ", " << cols << "]");
                return false;
            }
            return true;
        }

        ITensorGemm *resolvePreparedGemmForStage(
            const char *caller,
            const GEMMStage::Params &params,
            bool is_sliced)
        {
            if (!params.prepared_store || !params.prepared_ref.has_value())
            {
                LOG_ERROR("[" << caller << "] PreparedWeightStore and PreparedWeightRef are required for model GEMM resolution");
                return nullptr;
            }

            ITensorGemm *gemm = nullptr;
            if (is_sliced)
            {
                gemm = params.prepared_store->slicedGemmKernel(
                    params.prepared_ref.value(),
                    params.output_range.start,
                    params.output_range.end);
            }
            else
            {
                gemm = params.prepared_store->gemmKernel(params.prepared_ref.value());
            }

            if (!gemm)
            {
                LOG_ERROR("[" << caller << "] PreparedWeightRef was provided but no GEMM kernel was found in PreparedWeightStore");
            }
            return gemm;
        }

        /**
         * @brief Infer the transpose flag expected by floating-point GEMM adapters.
         *
         * Quantized kernels in Llaminar own the `[N,K]` weight layout internally,
         * so graph builders pass `transpose_B=false` for ordinary model
         * projections.  The cuBLAS/hipBLAS floating-point adapters expose the
         * lower-level BLAS convention: a row-major model weight stored as
         * `[N,K]` must be consumed with `transpose_B=true` to compute
         * `C[M,N] = A[M,K] * W[N,K]^T`.
         *
         * MTP grouped verifier publication does not carry an explicit transpose
         * flag; the floating adapters' grouped hooks already use the `[N,K]`
         * model-weight convention.  Serial M=1 decode must therefore make the
         * same inference or byte-equivalence tests compare different matrix
         * products.  If a caller truly provides a floating weight stored as
         * `[K,N]`, its shape will not match `[n,k]` and the explicit
         * `params.transpose_B=false` setting is preserved.
         */
        bool effectiveTransposeBForWeight(
            const TensorBase *weight,
            int n,
            int k,
            bool requested_transpose)
        {
            if (requested_transpose || !weight)
                return requested_transpose;

            const TensorType type = weight->native_type();
            const bool floating_weight =
                type == TensorType::FP32 ||
                type == TensorType::FP16 ||
                type == TensorType::BF16;
            if (!floating_weight)
                return requested_transpose;

            const auto &shape = weight->shape();
            if (shape.size() >= 2 &&
                shape[0] == static_cast<size_t>(n) &&
                shape[1] == static_cast<size_t>(k))
            {
                return true;
            }
            return requested_transpose;
        }

    }

    // =============================================================================
    // GEMMStage::Params Implementation
    // =============================================================================

    const float *GEMMStage::Params::getBiasData() const
    {
        // Prefer bias_tensor over raw bias pointer for TensorSlice compatibility
        if (bias_tensor)
        {
            // Cast to TensorBase to access is_fp32_backed()
            auto *bias_base = dynamic_cast<const TensorBase *>(bias_tensor);
            if (!bias_base)
            {
                LOG_WARN("[GEMMStage::Params] bias_tensor is not TensorBase, ignoring.");
                return nullptr;
            }
            // Use unified data access interface - works for FP32Tensor and TensorSlice
            if (bias_base->is_fp32_backed())
            {
                return bias_base->data();
            }
            else
            {
                // Non-FP32-backed bias tensor (e.g., quantized) - not supported
                LOG_WARN("[GEMMStage::Params] bias_tensor is not FP32-backed, ignoring. "
                         "Type: "
                         << bias_base->dtype_name());
                return nullptr;
            }
        }
        // Fall back to raw bias pointer (legacy interface)
        return bias;
    }

    void GEMMStage::Params::validate(const std::string &stage_name) const
    {
        std::ostringstream errors;

        // Check required tensors
        if (!A)
        {
            errors << "Input tensor A is null. ";
        }
        if (!B)
        {
            errors << "Weight tensor B is null. ";
        }
        if (!C)
        {
            errors << "Output tensor C is null. ";
        }

        // Check dimensions
        if (m <= 0 || n <= 0 || k <= 0)
        {
            errors << "Invalid dimensions (m=" << m << ", n=" << n << ", k=" << k << "). ";
        }

        if(request_rows && (!force_decode_equivalent_verifier_prefill || verifier_row_range || request_rows->physicalRows()!=m ||
            gate_input || do_swiglu || bias || bias_tensor || bias_required || !output_range.empty() || alpha!=1.0f || beta!=0.0f))
            errors << "Counted floating requests require explicit decode-equivalent arithmetic, exact rows, no prefix/bias/fusion, alpha1 beta0. ";

        // Check bias requirement
        if (bias_required)
        {
            const float *bias_data = getBiasData();
            if (!bias_data)
            {
                errors << "Bias is required but not provided (neither bias nor bias_tensor set). ";
            }
        }

        // Throw if any errors
        std::string error_str = errors.str();
        if (!error_str.empty())
        {
            throw std::runtime_error("[" + stage_name + "] Configuration error: " + error_str);
        }
    }

    // =============================================================================
    // GEMMStage Implementation
    // =============================================================================

    GEMMStage::GEMMStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if(params_.quantized_operand_precision!=QuantizedActivationOperandPrecision::SingleQ8&&
            params_.quantized_operand_precision!=QuantizedActivationOperandPrecision::TwoTermQ8)
            throw std::invalid_argument("GEMM activation operand precision is invalid");
        if(params_.request_rows) {
            params_.validate();
            if(params_.quantized_operand_precision!=QuantizedActivationOperandPrecision::SingleQ8)
                throw std::invalid_argument("Counted floating request rows do not support TwoTermQ8 activation arithmetic");
            if(params_.A->native_type()!=TensorType::FP32 || params_.C->native_type()!=TensorType::FP32 ||
                (params_.B->native_type()!=TensorType::FP32 && params_.B->native_type()!=TensorType::FP16 &&
                 params_.B->native_type()!=TensorType::BF16))
                throw std::invalid_argument("Counted projection requires native floating weights and FP32 rows/output");
            if(!validateMatrixExtent(dynamic_cast<const TensorBase *>(params_.A),"counted input A",params_.m,params_.k) ||
                !validateMatrixExtent(dynamic_cast<const TensorBase *>(params_.C),"counted output C",params_.m,params_.n))
                throw std::invalid_argument("Counted floating projection exceeds admitted source/output matrix");
        }
        if(params_.quantized_operand_precision==QuantizedActivationOperandPrecision::TwoTermQ8) {
            validateQ8ResidualCorrectionGeometry(params_.m,params_.n,params_.k);
            if(!params_.A||!params_.B||!params_.C||params_.A==params_.C||params_.gate_input||params_.bias||params_.bias_tensor||
                params_.alpha!=1.f||params_.beta!=0.f||params_.A->native_type()!=TensorType::FP32||
                params_.C->native_type()!=TensorType::FP32||params_.B->native_type()!=TensorType::Q8_0)
                throw std::invalid_argument("Two-term Q8 requires a plain prepared native Q8_0 projection with distinct FP32 operands");
            validateQ8ResidualTensorCapacities(params_.m,params_.output_range.empty()?params_.n:static_cast<int>(params_.output_range.size()),
                params_.k,params_.A->numel(),params_.C->numel());
            const auto*input=dynamic_cast<const TensorBase*>(params_.A);
            const auto*output=dynamic_cast<const TensorBase*>(params_.C);
            // Construction can precede arena materialization. Validate any
            // already prepared addresses now; the operation repeats this check
            // against the eventual exact pointers before its first launch.
            const bool hostPrepared=params_.device_id.is_cpu()&&input&&output&&input->isHostValid()&&output->isHostValid();
            const void*inputPointer=hostPrepared?input->data():params_.device_id.is_gpu()?params_.A->gpu_data_ptr():nullptr;
            const void*outputPointer=hostPrepared?output->data():params_.device_id.is_gpu()?params_.C->gpu_data_ptr():nullptr;
            if(inputPointer&&outputPointer) {
                const auto outputWidth=params_.output_range.empty()?params_.n:static_cast<int>(params_.output_range.size());
                const std::array writes{Q8ResidualStorageSpan{outputPointer,static_cast<std::size_t>(params_.m)*outputWidth*sizeof(float),alignof(float)}};
                const std::array reads{Q8ResidualStorageSpan{inputPointer,static_cast<std::size_t>(params_.m)*params_.k*sizeof(float),alignof(float)}};
                validateQ8ResidualStorage(writes,reads);
            }
        }
    }

    uint64_t GEMMStage::graphCaptureVariantSignature() const
    {
        if(params_.quantized_operand_precision==QuantizedActivationOperandPrecision::SingleQ8 &&
           !params_.verifier_row_range && !params_.request_rows)return 0;
        // Preserve the qualified HC quantizer/scalars, then identify one sealed
        // row alternative. Count pointer/geometry are immutable recording data;
        // execution-owned count values never enter graph identity.
        uint64_t signature=0x5138524553320002ULL;
        const auto mix=[&](uint64_t value){signature^=value+0x9e3779b97f4a7c15ULL+(signature<<6)+(signature>>2);};
        const auto backend=params_.device_id.type;
        const auto quantizer=backend==DeviceType::CPU?Q8ResidualQuantizerPolicy::CPUHalfAfterQuantization:
            backend==DeviceType::CUDA?Q8ResidualQuantizerPolicy::CUDAFloatScaleNativeReciprocal:Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal;
        mix(static_cast<uint64_t>(params_.quantized_operand_precision));
        mix(static_cast<uint64_t>(quantizer));mix(static_cast<uint64_t>(backend));mix(params_.device_id.ordinal);
        mix(params_.m);mix(params_.n);mix(params_.k);mix(params_.output_range.start);mix(params_.output_range.end);
        mix(reinterpret_cast<std::uintptr_t>(params_.A));mix(reinterpret_cast<std::uintptr_t>(params_.B));
        mix(reinterpret_cast<std::uintptr_t>(params_.C));mix(params_.force_decode_equivalent_verifier_prefill);
        if(params_.request_rows) {
            mix(2);mix(FloatingProjectionRows::requestRecordingIdentity(*params_.request_rows));
        } else {
            mix(params_.verifier_row_range ? 1 : 0);
            const auto rows=params_.verifier_row_range.value_or(DeviceRowRange::fullyActive(params_.m));
            const auto row_identity = rows.captureIdentity();
            mix(reinterpret_cast<std::uintptr_t>(row_identity.count));
            mix(row_identity.count_capacity);
            mix(row_identity.first_physical_row);
            mix(row_identity.physical_rows);
            mix(row_identity.repeats_per_row);
            mix(row_identity.schema_version);
        }
        return signature?signature:1;
    }

    void GEMMStage::clearCachedGemmStream()
    {
        if (cached_gemm_)
            cached_gemm_->clearGPUStreamBinding();
    }

    void GEMMStage::resetSessionState()
    {
        IComputeStage::resetSessionState();
        clearCachedGemmStream();
    }

    void GEMMStage::resetSessionStatePreservingCapturedReplay()
    {
        IComputeStage::resetSessionStatePreservingCapturedReplay();
        clearCachedGemmStream();
    }

    void GEMMStage::resetSessionStatePreservingLazyInitialization()
    {
        resetSessionStatePreservingCapturedReplay();
    }

    bool GEMMStage::validatePreparedWeights(std::string *error) const
    {
        if (!params_.B)
            return true;
        if (!params_.prepared_store || !params_.prepared_ref.has_value())
        {
            if (error) *error = "GEMMStage requires PreparedWeightStore and PreparedWeightRef";
            return false;
        }
        if (!params_.prepared_store->contains(params_.prepared_ref.value()))
        {
            if (error) *error = "GEMMStage PreparedWeightRef is not present in PreparedWeightStore";
            return false;
        }
        if(params_.request_rows) {
            try {
                auto *engine=resolvePreparedGemmForStage("counted floating validation",params_,!params_.output_range.empty());
                if(!engine)throw std::invalid_argument("Counted floating projection has no prepared engine");
                auto mode=engine->beginFloatingRequestRowsScope(*params_.request_rows);
            } catch(const std::exception &exception) {
                if(error)*error=exception.what();
                return false;
            }
        }
        return true;
    }

    bool GEMMStage::execute(IDeviceContext *ctx)
    {
        ScopedGemmContext gemm_ctx(params_.gemm_context);

        if (!ctx)
        {
            LOG_ERROR("[GEMMStage] Null device context");
            return false;
        }

        // Validate inputs
        if (!params_.A)
        {
            LOG_ERROR("[GEMMStage] Null input A");
            return false;
        }

        if (!params_.B || !params_.C)
        {
            LOG_ERROR("[GEMMStage] Invalid parameters: B=" << params_.B << " C=" << params_.C);
            return false;
        }

        if (params_.m <= 0 || params_.n <= 0 || params_.k <= 0)
        {
            LOG_ERROR("[GEMMStage] Invalid dimensions: m=" << params_.m
                                                           << " n=" << params_.n << " k=" << params_.k);
            return false;
        }

        // === Stage Tracing (Task 3) ===
        traceInput("A", params_.A);
        traceInput("B", params_.B);

        // Check if this is a sliced (tensor-parallel) GEMM
        const bool is_sliced = !params_.output_range.empty();
        const int effective_n = is_sliced ? static_cast<int>(params_.output_range.size()) : params_.n;

        const auto *A_extent = requireTensorBase(params_.A, "input A");
        auto *C_extent = asTensorBase(params_.C, "output C");
        if (!validateMatrixExtent(A_extent, "A", params_.m, params_.k) ||
            !validateMatrixExtent(C_extent, "C", params_.m, effective_n))
        {
            return false;
        }
        if (params_.gate_input)
        {
            const auto *gate_extent = requireTensorBase(params_.gate_input, "gate input");
            if (!validateMatrixExtent(gate_extent, "gate_input", params_.m, params_.k))
                return false;
        }

        LOG_TRACE("[GEMMStage] Execute GEMM: " << params_.m << "x" << effective_n << "x" << params_.k
                                               << (is_sliced ? " (SLICED)" : "")
                                               << " weight ptr=" << static_cast<const void *>(params_.B)
                                               << " weight shape=[" << (params_.B ? params_.B->shape()[0] : 0) << ","
                                               << (params_.B ? params_.B->shape()[1] : 0) << "]"
                                               << " input ptr=" << static_cast<const void *>(params_.A)
                                               << " input_type=" << params_.A->dtype_name()
                                               << " output_type=" << params_.C->dtype_name());

        // Debug: Log input values for Wo projection (weight shape [896, 448] or [896, 896])
        // Only log if input is FP32 (Q8_1 data() throws to prevent accidental dequantization)
        // NOTE: ONLY call data() when TRACE logging is actually enabled - it triggers D2H transfer!
#if 0 // DISABLED: calling data() triggers expensive D2H transfer even if LOG_TRACE doesn't print
        if (params_.B && params_.B->shape()[0] == 896 && (params_.B->shape()[1] == 448 || params_.B->shape()[1] == 896))
        {
            if (std::string(params_.A->dtype_name()) == "FP32")
            {
                const float *input_data = params_.A->data();
                if (input_data)
                {
                    LOG_TRACE("[GEMMStage] Wo input[0:8]=" << std::setprecision(10)
                                                           << input_data[0] << "," << input_data[1] << "," << input_data[2] << "," << input_data[3] << ","
                                                           << input_data[4] << "," << input_data[5] << "," << input_data[6] << "," << input_data[7]
                                                           << " weight_k=" << params_.B->shape()[1]);
                }
            }
        }
#endif

        // Cast weights to TensorBase for diagnostics and tensor-aware kernel calls.
        auto *B_base = requireTensorBase(params_.B, "weight B");
        const bool effective_transpose_B =
            effectiveTransposeBForWeight(B_base, effective_n, params_.k, params_.transpose_B);

        // Get kernel — use stage-level cache to avoid store lookup per call.
        llaminar2::ITensorGemm *gemm = nullptr;

        if (cache_resolved_)
        {
            // Fast path: reuse previously-resolved kernel (no mutex)
            gemm = cached_gemm_;
        }
        else
        {
            gemm = resolvePreparedGemmForStage("GEMMStage", params_, is_sliced);
            if (!gemm)
                return false;
            cached_gemm_ = gemm;
            cache_resolved_ = true;
            if (is_sliced)
            {
                LOG_TRACE("[GEMMStage] Using prepared sliced kernel for rows [" << params_.output_range.start
                                                                                << ", " << params_.output_range.end << ")");
            }
        }

        if (!gemm)
        {
            LOG_ERROR("[GEMMStage] Failed to get GEMM kernel for weight tensor");
            return false;
        }

        if(params_.quantized_operand_precision==QuantizedActivationOperandPrecision::TwoTermQ8) {
            if(params_.gate_input||params_.bias_tensor||params_.bias||params_.alpha!=1.f||params_.beta!=0.f)
                throw std::logic_error("Two-term Q8 projection requires plain alpha1,beta0 without gate/bias epilogues");
            bindStageStream(gemm);
            gemm->multiply_q8_residual_corrected(A_extent,C_extent,
                params_.verifier_row_range.value_or(DeviceRowRange::fullyActive(params_.m)),effective_n,params_.k,getWorkspace());
            publishStageOutput(C_extent);
            traceOutput("C",params_.C);
            return true;
        }

        // Thread GPU stream for graph capture
        bindStageStream(gemm);

        LOG_TRACE("[GEMMStage] Got kernel ptr=" << static_cast<const void *>(gemm)
                                                << " for weight ITensor*=" << static_cast<const void *>(params_.B)
                                                << " TensorBase*=" << static_cast<const void *>(B_base));

        if (params_.request_rows || (params_.force_decode_equivalent_verifier_prefill &&
            (params_.m > 1 || params_.verifier_row_range.has_value())))
        {
            auto *A_base = requireTensorBase(params_.A, "input A");
            auto *C_base = asTensorBase(params_.C, "output C");
            return executeDecodeEquivalentVerifierPrefill(
                A_base, C_base, gemm, effective_n);
        }

        // Fused SwiGLU + GEMM: output = W @ (silu(gate) * up)
        // Both CPU and GPU kernels implement multiply_tensor_with_fused_swiglu().
        if (params_.gate_input)
        {
            auto *gate_base = requireTensorBase(params_.gate_input, "gate input");
            auto *A_base_up = requireTensorBase(params_.A, "input A (up)");
            auto *C_base = asTensorBase(params_.C, "output C");

            if (!gemm->multiply_tensor_with_fused_swiglu(
                    gate_base, A_base_up, C_base,
                    params_.m, effective_n, params_.k,
                    params_.alpha, params_.beta,
                    getWorkspace()))
            {
                LOG_ERROR("[GEMMStage] Required fused SwiGLU+GEMM contract failed; "
                          "separate activation/GEMM replay is forbidden");
                return false;
            }

            publishStageOutput(C_base);
            LOG_TRACE("[GEMMStage] Fused SwiGLU+GEMM completed via ITensorGemm");
            traceOutput("C", params_.C);
            return true;
        }

        // Primary path: use tensor-aware multiply_tensor for type-aware dispatch.
        // Works for CPU (NativeVNNI, FloatingPointGemm) and GPU (CUDA/ROCm) kernels.
        {
            auto *A_base = requireTensorBase(params_.A, "input A");
            auto *C_base = asTensorBase(params_.C, "output C");

            LOG_TRACE("[GEMMStage] Using multiply_tensor for type-aware dispatch: "
                      << "input_type=" << params_.A->dtype_name()
                      << " output_type=" << params_.C->dtype_name());
            bool success = gemm->multiply_tensor(
                A_base, C_base,
                params_.m, effective_n, params_.k,
                effective_transpose_B,
                params_.alpha, params_.beta,
                nullptr, // bias
                params_.mpi_ctx, params_.device_id.toKernelDeviceIndex(),
                getWorkspace());

            if (success)
            {
                publishStageOutput(C_base);
                traceOutput("C", params_.C);
            }
            return success;
        }
    }

    bool GEMMStage::executeDecodeEquivalentVerifierPrefill(
        const TensorBase *A_base,
        TensorBase *C_base,
        ITensorGemm *gemm,
        int effective_n)
    {
        if (!A_base || !C_base || !gemm)
            return false;
        if (params_.alpha != 1.0f || params_.beta != 0.0f)
        {
            LOG_ERROR("[GEMMStage] Grouped verifier GEMM supports alpha=1,beta=0 only; got alpha="
                      << params_.alpha << " beta=" << params_.beta);
            return false;
        }

        const bool is_gpu = params_.device_id.is_gpu();
        void *stream = gpuStream();
        if (is_gpu && !stream)
        {
            LOG_ERROR("[GEMMStage] Grouped verifier GPU GEMM requires an explicit stream");
            return false;
        }

        const auto *gate_base = params_.gate_input
                                    ? requireTensorBase(params_.gate_input, "gate input")
                                    : nullptr;
        if (params_.gate_input && !gate_base)
            return false;

        // Nested tensor adapters inherit this geometry while retaining their
        // backend-specific serial arithmetic policy. No device count is read here.
        auto verifier_rows = params_.request_rows
            ? gemm->beginFloatingRequestRowsScope(*params_.request_rows)
            : gemm->beginVerifierDecodeEquivalentScope(params_.verifier_row_range);
        bool success = false;
        if (params_.gate_input)
        {
            success = gemm->multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                gate_base,
                A_base,
                C_base,
                params_.m,
                effective_n,
                params_.k,
                params_.alpha,
                params_.beta,
                getWorkspace());
        }
        else
        {
            std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                {gemm, C_base, effective_n, nullptr, "gemm"}};
            success = gemm->multiply_fused_verifier_rows_decode_equivalent(
                A_base,
                projections,
                params_.m,
                params_.k,
                params_.mpi_ctx,
                getWorkspace());
        }

        if (!success)
        {
            LOG_ERROR("[GEMMStage] Grouped decode-equivalent verifier GEMM failed"
                      << " device=" << params_.device_id.to_string()
                      << " swiglu=" << (params_.gate_input ? "1" : "0")
                      << " m=" << params_.m
                      << " n=" << effective_n
                      << " k=" << params_.k);
            return false;
        }

        if (is_gpu)
            gpuExecution().publish(C_base);
        PerfStatsCollector::addCounter(
            "mtp",
            "gemm_grouped_decode_equivalent_verifier_prefill_rows",
            static_cast<double>(params_.m),
            {},
            params_.device_id.to_string(),
            {{"stage", "GEMM"},
             {"swiglu", params_.gate_input ? "1" : "0"},
             {"route", "grouped"}});
        traceOutput("C", params_.C);
        return true;
    }

    size_t GEMMStage::estimatedFlops() const
    {
        // GEMM: 2 * M * N * K (multiply + add)
        return static_cast<size_t>(2) * params_.m * params_.n * params_.k;
    }

    size_t GEMMStage::estimatedMemoryBytes() const
    {
        // A: m * k reads, B: k * n reads, C: m * n writes (+ reads if beta != 0)
        size_t a_bytes = static_cast<size_t>(params_.m) * params_.k * sizeof(float);
        size_t c_bytes = static_cast<size_t>(params_.m) * params_.n * sizeof(float);

        // B may be quantized, so we estimate based on tensor
        // For now, assume FP32 - tensor introspection would be better
        size_t b_bytes = static_cast<size_t>(params_.k) * params_.n * sizeof(float);

        return a_bytes + b_bytes + c_bytes;
    }

    bool GEMMStage::supportsBackend(ComputeBackendType backend) const
    {
        // Unified GEMMStage supports all backends via KernelFactory dispatch
        // KernelFactory will select the appropriate kernel based on tensor device affinity
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
        case ComputeBackendType::GPU_CUDA:
#ifdef HAVE_CUDA
            return true; // KernelFactory can create CUDA kernels
#else
            return false;
#endif
        case ComputeBackendType::GPU_ROCM:
#ifdef HAVE_ROCM
            return true; // KernelFactory can create ROCm kernels
#else
            return false;
#endif
        case ComputeBackendType::GPU_VULKAN:
        case ComputeBackendType::GPU_METAL:
            return false; // Not yet implemented
        default:
            return false;
        }
    }

    StageDumpInfo GEMMStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Input A: activation tensor
        if (params_.A)
        {
            info.addInput("A", params_.A, params_.m, params_.k);
        }

        // Weight tensor B
        info.addWeight("B", params_.B);

        // Output C
        if (params_.C)
        {
            info.addOutput("C", params_.C, params_.m, params_.n);
        }

        // Bias is immutable model state, not a per-execution activation. Prefer
        // the typed tensor so device-only prepared ownership remains visible.
        if (params_.bias_tensor)
        {
            info.addWeight("bias", params_.bias_tensor);
        }
        else if (params_.bias)
        {
            info.addRawWeight(
                "bias",
                params_.bias,
                static_cast<size_t>(params_.n) * sizeof(float),
                /*rows=*/1,
                static_cast<size_t>(params_.n),
                "FP32");
        }
        if (params_.gate_input)
        {
            // Gate and up are multiplied before projection: both reduce over
            // K. N is only the output width and may be smaller or larger.
            info.addInput("gate_input", params_.gate_input, params_.m, params_.k);
        }

        // Scalar params
        info.addScalarInt("m", params_.m);
        info.addScalarInt("n", params_.n);
        info.addScalarInt("k", params_.k);
        info.addScalar("alpha", params_.alpha);
        info.addScalar("beta", params_.beta);
        info.addScalarBool("transpose_B", params_.transpose_B);
        info.addScalarBool("do_swiglu", params_.do_swiglu);
        info.addScalarInt("quantized_operand_precision",static_cast<int>(params_.quantized_operand_precision));
        info.addScalarInt("device_id", params_.device_id.toKernelDeviceIndex());

        return info;
    }

    StageBufferRequirements GEMMStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.A || !params_.B || !params_.C)
            return reqs; // Empty if tensors not set

        // Convert tensor type to buffer tensor type
        BufferTensorType a_type = toBufferTensorType(params_.A->native_type());
        BufferTensorType b_type = toBufferTensorType(params_.B->native_type());
        BufferTensorType c_type = toBufferTensorType(params_.C->native_type());

        // INPUT buffer (activations)
        reqs.addInput("A", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.k)}, a_type);

        // WEIGHT buffer (read-only, may be quantized)
        reqs.addWeight("B", {static_cast<size_t>(params_.k), static_cast<size_t>(params_.n)}, b_type);

        // OUTPUT buffer
        reqs.addOutput("C", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.n)}, c_type);

        // Optional bias (check both raw pointer and tensor)
        const float *bias_data = params_.getBiasData();
        if (bias_data)
        {
            reqs.addWeight("bias", {static_cast<size_t>(params_.n)}, BufferTensorType::FP32);
        }

        // Optional gate_input for SwiGLU fusion
        if (params_.gate_input)
        {
            BufferTensorType gate_type = toBufferTensorType(params_.gate_input->native_type());
            reqs.addInput("gate_input", {static_cast<size_t>(params_.m), static_cast<size_t>(params_.k)}, gate_type);
        }

        return reqs;
    }

    // =============================================================================
    // IWorkspaceConsumerStage Implementation
    // =============================================================================

    IWorkspaceConsumer *GEMMStage::getKernelAsWorkspaceConsumer()
    {
        if (!params_.B)
        {
            LOG_WARN("[GEMMStage::getKernelAsWorkspaceConsumer] Weight tensor B not set");
            return nullptr;
        }

        auto *B_base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(params_.B));
        if (!B_base)
        {
            LOG_WARN("[GEMMStage::getKernelAsWorkspaceConsumer] Weight tensor B is not TensorBase");
            return nullptr;
        }

        const bool is_sliced = !params_.output_range.empty();
        ITensorGemm *gemm = resolvePreparedGemmForStage(
            "GEMMStage::getKernelAsWorkspaceConsumer", params_, is_sliced);

        return dynamic_cast<IWorkspaceConsumer *>(gemm);
    }

    StageBufferContract GEMMStage::bufferContract() const
    {
        if (!params_.a_buffer_id || !params_.c_buffer_id)
            return {};

        auto contract = StageBufferContract::build()
            .addInput(*params_.a_buffer_id);
        if (params_.gate_input && params_.gate_buffer_id)
            contract.addInput(*params_.gate_buffer_id);
        contract.addOutput(*params_.c_buffer_id);
        // GEMM consumes the PreparedWeightStore-owned representation, never the
        // source tensor's raw device pointer.
        if (params_.B)
        {
            contract.addPreparedWeight(
                const_cast<ITensor *>(params_.B),
                params_.prepared_store,
                params_.prepared_ref.value_or(PreparedWeightRef{}));
        }
        if (params_.bias_tensor)
            contract.addWeight(const_cast<ITensor *>(static_cast<const ITensor *>(params_.bias_tensor)));
        return contract;
    }

} // namespace llaminar2
