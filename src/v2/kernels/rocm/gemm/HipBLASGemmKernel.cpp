/**
 * @file HipBLASGemmKernel.cpp
 * @brief Context-borrowing hipBLAS projection views with atomic host submission.
 *
 * Device contexts retain the library through expert movement and graph replay.
 * Projection destruction releases no device library. The context's scoped
 * handle lock covers exact stream/workspace selection through asynchronous
 * enqueue; it never waits for GPU completion.
 *
 * This file is compiled with hipcc to use HIP runtime APIs.
 *
 * NOTE: This file cannot include utils/Logger.h or <iostream> because hipcc has issues
 * with libstdc++ locale/ostream headers when inside a namespace.
 * Errors are reported via return codes and exceptions only.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

// HIP headers MUST be included BEFORE any other headers
// to avoid std:: namespace conflicts in HIP template metaprogramming
#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h> // For __half and __float2half
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h> // For fused GEMM+bias
#endif

#include "HipBLASGemmKernel.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/GPUDeviceContextPool.h"
#include "kernels/common/FloatingPointGemmWorkspaceABI.h"
#include "../../../backends/rocm/HipDeviceGuard.h"
#include <stdexcept>
#include <string>

// Logging disabled in HIP files due to hipcc/libstdc++ conflicts
// All errors are reported via exceptions or return codes
#define HIP_LOG_DEBUG(msg) ((void)0)
#define HIP_LOG_ERROR(msg) ((void)0)

namespace llaminar2
{
    namespace rocm
    {

#ifdef HAVE_ROCM

        namespace
        {
            constexpr auto kBiasMatmulWorkspace = floating_gemm_abi::kROCmBlasMatmulWorkspace;
            constexpr auto kBiasMatmulWorkspaceBytes = floating_gemm_abi::kBlasMatmulWorkspaceBytes;

            /**
             * @brief Validate the backend before creating or borrowing any context.
             * @param device_id Explicit ROCm device requested by the projection.
             * @return Its persistent, initialized library authority.
             * @throws std::invalid_argument If the device is not a ROCm device.
             */
            IWorkerGPUContext *projectionContext(const DeviceId &device_id)
            {
                if (!device_id.is_rocm())
                    throw std::invalid_argument("hipBLAS GEMM requires a ROCm device");
                return &GPUDeviceContextPool::instance().getContext(device_id);
            }

            /** @brief Bind the mutually exclusive BLAS/Lt region from this stage's arena. */
            bool bindMatmulWorkspace(void *handle, DeviceWorkspaceManager *workspace)
            {
                if (!workspace ||
                    workspace->getBufferSize(kBiasMatmulWorkspace) < kBiasMatmulWorkspaceBytes)
                    throw std::logic_error("hipBLAS submission requires its declared arena workspace");
                void *buffer = workspace->getBuffer(kBiasMatmulWorkspace);
                if (!buffer)
                    throw std::logic_error("hipBLAS arena workspace has not been materialized");
                return hipblasSetWorkspace(static_cast<hipblasHandle_t>(handle),
                                           buffer, kBiasMatmulWorkspaceBytes) == HIPBLAS_STATUS_SUCCESS;
            }
        } // namespace

        // =====================================================================
        // Helper macros for error checking
        // =====================================================================

#define HIPBLAS_CHECK(call)                                                           \
    do                                                                                \
    {                                                                                 \
        hipblasStatus_t status = call;                                                \
        if (status != HIPBLAS_STATUS_SUCCESS)                                         \
        {                                                                             \
            HIP_LOG_ERROR("[HipBLASGemmKernel] hipBLAS error: " << status             \
                                                                << " at " << __FILE__ \
                                                                << ":" << __LINE__);  \
            return false;                                                             \
        }                                                                             \
    } while (0)

#define HIPBLASLT_CHECK(call)                                                           \
    do                                                                                  \
    {                                                                                   \
        hipblasStatus_t status = call;                                                  \
        if (status != HIPBLAS_STATUS_SUCCESS)                                           \
        {                                                                               \
            HIP_LOG_ERROR("[HipBLASGemmKernel] hipBLASLt error: " << status             \
                                                                  << " at " << __FILE__ \
                                                                  << ":" << __LINE__);  \
            return false;                                                               \
        }                                                                               \
    } while (0)

#define HIP_CHECK(call)                                                               \
    do                                                                                \
    {                                                                                 \
        hipError_t err = call;                                                        \
        if (err != hipSuccess)                                                        \
        {                                                                             \
            HIP_LOG_ERROR("[HipBLASGemmKernel] HIP error: " << hipGetErrorString(err) \
                                                            << " at " << __FILE__     \
                                                            << ":" << __LINE__);      \
            return false;                                                             \
        }                                                                             \
    } while (0)

        // =====================================================================
        // Constructor / Destructor
        // =====================================================================

        HipBLASGemmKernel::HipBLASGemmKernel(const DeviceId &device_id, Precision precision)
            : HipBLASGemmKernel(projectionContext(device_id), precision)
        {
        }

        HipBLASGemmKernel::HipBLASGemmKernel(IWorkerGPUContext *ctx, Precision precision)
            : precision_(precision)
        {
            if (!ctx || !ctx->isInitialized())
                throw std::invalid_argument("hipBLAS GEMM requires an initialized worker context");
            setDeviceContext(ctx);
            device_id_ = DeviceId::rocm(ctx->deviceOrdinal());
            const auto submission = ctx->acquireBlasSubmission();
        }

        HipBLASGemmKernel::~HipBLASGemmKernel() = default;
        HipBLASGemmKernel::HipBLASGemmKernel(HipBLASGemmKernel &&) noexcept = default;
        HipBLASGemmKernel &HipBLASGemmKernel::operator=(HipBLASGemmKernel &&) noexcept = default;

        bool HipBLASGemmKernel::execute(
            const float *d_A, const float *d_B, float *d_C,
            int M, int N, int K,
            bool transA, bool transB,
            float alpha, float beta)
        {
            try
            {
                return executeOnStream(
                    ExplicitGPUStream{
                        requireStream("HipBLASGemmKernel::execute")},
                    d_A, d_B, d_C,
                    M, N, K,
                    transA, transB,
                    alpha, beta);
            }
            catch (const std::exception &exception)
            {
                HIP_LOG_ERROR(exception.what());
                return false;
            }
        }

        bool HipBLASGemmKernel::executeOnStream(
            ExplicitGPUStream stream,
            const float *d_A, const float *d_B, float *d_C,
            int M, int N, int K,
            bool transA, bool transB,
            float alpha, float beta)
        {
            const auto submission = device_ctx_->acquireBlasSubmission();
            void *const handle_ = submission.handle();
            if (!handle_)
            {
                HIP_LOG_ERROR("[HipBLASGemmKernel::execute] hipBLAS handle is null");
                return false;
            }

            // Ensure we're on the correct device
            hipError_t hip_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(device_id_.ordinal));
            if (hip_err != hipSuccess)
            {
                HIP_LOG_ERROR("[HipBLASGemmKernel::execute] Failed to set device: " << hipGetErrorString(hip_err));
                return false;
            }

            const hipblasStatus_t stream_status = hipblasSetStream(
                static_cast<hipblasHandle_t>(handle_),
                static_cast<hipStream_t>(stream.get()));
            if (stream_status != HIPBLAS_STATUS_SUCCESS)
            {
                HIP_LOG_ERROR(
                    "[HipBLASGemmKernel::execute] hipblasSetStream failed: "
                    << static_cast<int>(stream_status));
                return false;
            }

            if (!bindMatmulWorkspace(handle_, workspace_))
                return false;

            // =========================================================================
            // Row-major to column-major conversion for hipBLAS
            // =========================================================================
            //
            // Our data is stored in row-major format:
            //   A[M×K] row-major: element (i,j) at offset i*K + j, stride = K
            //   B[K×N] row-major: element (i,j) at offset i*N + j, stride = N
            //   (Or if transB=true, B is stored as N×K row-major with stride K)
            //
            // hipBLAS interprets memory as column-major:
            //   A memory with stride K seen as K×M column-major matrix = A^T
            //   B memory with stride N seen as N×K column-major matrix = B^T
            //
            // We want: C = A @ B (row-major)
            // Which is: C^T = (A @ B)^T = B^T @ A^T (column-major)
            //
            // hipBLAS already sees A^T and B^T in our memory, so:
            //   - For no user transpose (transA=false, transB=false):
            //     Call hipBLAS with HIPBLAS_OP_N to use A^T and B^T as-is
            //   - For user transpose (transA=true):
            //     We want A^T in the product, but hipBLAS sees A^T already,
            //     so we need HIPBLAS_OP_T to transpose it back to A
            //
            // Summary:
            //   transA=false → opA=HIPBLAS_OP_N (use hipBLAS's A^T view)
            //   transA=true  → opA=HIPBLAS_OP_T (transpose hipBLAS's A^T back to A)

            hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
            hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

            // Leading dimensions are the memory strides:
            //   lda = K (stride of A memory)
            //   ldb = N if transB=false (B is K×N row-major, stride N)
            //       = K if transB=true  (B is N×K row-major, stride K)
            //   ldc = N (stride of C memory, C is M×N row-major)
            int lda = K;
            int ldb = transB ? K : N;
            int ldc = N;

            // hipBLAS call: gemm(opB, opA, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc)
            // Computes: op(B_cm) @ op(A_cm) where B_cm, A_cm are hipBLAS's col-major views

            HIPBLAS_CHECK(hipblasSgemm(
                static_cast<hipblasHandle_t>(handle_),
                opB, opA,
                N, M, K,
                &alpha,
                d_B, ldb,
                d_A, lda,
                &beta,
                d_C, ldc));

            return true;
        }

        bool HipBLASGemmKernel::execute_batched(
            const float *const *d_A_array,
            const float *const *d_B_array,
            float *const *d_C_array,
            int M, int N, int K,
            int batch_count,
            bool transA, bool transB,
            float alpha, float beta)
        {
            try
            {
                return executeBatchedOnStream(
                    ExplicitGPUStream{
                        requireStream("HipBLASGemmKernel::execute_batched")},
                    d_A_array, d_B_array, d_C_array,
                    M, N, K, batch_count,
                    transA, transB,
                    alpha, beta);
            }
            catch (const std::exception &exception)
            {
                HIP_LOG_ERROR(exception.what());
                return false;
            }
        }

        bool HipBLASGemmKernel::executeBatchedOnStream(
            ExplicitGPUStream stream,
            const float *const *d_A_array,
            const float *const *d_B_array,
            float *const *d_C_array,
            int M, int N, int K,
            int batch_count,
            bool transA, bool transB,
            float alpha, float beta)
        {
            const auto submission = device_ctx_->acquireBlasSubmission();
            void *const handle_ = submission.handle();
            if (!handle_)
            {
                HIP_LOG_ERROR("[HipBLASGemmKernel::execute_batched] hipBLAS handle is null");
                return false;
            }

            if (!d_A_array || !d_B_array || !d_C_array || M <= 0 || N <= 0 || K <= 0 || batch_count <= 0)
            {
                HIP_LOG_ERROR("[HipBLASGemmKernel::execute_batched] Invalid arguments");
                return false;
            }

            const hipblasStatus_t stream_status = hipblasSetStream(
                static_cast<hipblasHandle_t>(handle_),
                static_cast<hipStream_t>(stream.get()));
            if (stream_status != HIPBLAS_STATUS_SUCCESS)
            {
                HIP_LOG_ERROR(
                    "[HipBLASGemmKernel::execute_batched] hipblasSetStream failed: "
                    << static_cast<int>(stream_status));
                return false;
            }

            hipError_t hip_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(device_id_.ordinal));
            if (hip_err != hipSuccess)
            {
                HIP_LOG_ERROR("[HipBLASGemmKernel::execute_batched] Failed to set device: " << hipGetErrorString(hip_err));
                return false;
            }

            hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
            hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
            int lda = K;
            if (!bindMatmulWorkspace(handle_, workspace_))
                return false;
            int ldb = transB ? K : N;
            int ldc = N;

            HIPBLAS_CHECK(hipblasSgemmBatched(
                static_cast<hipblasHandle_t>(handle_),
                opB, opA,
                N, M, K,
                &alpha,
                d_B_array, ldb,
                d_A_array, lda,
                &beta,
                d_C_array, ldc,
                batch_count));

            return true;
        }

        // =====================================================================
        // GEMM with Fused Bias (using hipBLASLt)
        // =====================================================================

        bool HipBLASGemmKernel::execute_with_bias(
            const float *d_A, const float *d_B, float *d_C,
            const float *d_bias,
            int M, int N, int K,
            bool transA, bool transB,
            float alpha, float beta)
        {
            try
            {
                return executeWithBiasOnStream(
                    ExplicitGPUStream{
                        requireStream(
                            "HipBLASGemmKernel::execute_with_bias")},
                    d_A, d_B, d_C, d_bias,
                    M, N, K,
                    transA, transB,
                    alpha, beta);
            }
            catch (const std::exception &exception)
            {
                HIP_LOG_ERROR(exception.what());
                return false;
            }
        }

        bool HipBLASGemmKernel::executeWithBiasOnStream(
            ExplicitGPUStream stream,
            const float *d_A, const float *d_B, float *d_C,
            const float *d_bias,
            int M, int N, int K,
            bool transA, bool transB,
            float alpha, float beta)
        {
            const auto submission = device_ctx_->acquireBlasSubmission();
            void *const lt_handle_ = submission.ltHandle();
            if (!lt_handle_)
            {
                HIP_LOG_ERROR("[HipBLASGemmKernel::execute_with_bias] hipBLASLt handle is null");
                return false;
            }

            // Ensure we're on the correct device
            hipError_t hip_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(device_id_.ordinal));
            if (hip_err != hipSuccess)
            {
                HIP_LOG_ERROR("[HipBLASGemmKernel::execute_with_bias] Failed to set device: " << hipGetErrorString(hip_err));
                return false;
            }

            hipblasLtHandle_t ltHandle = static_cast<hipblasLtHandle_t>(lt_handle_);

            // Create operation descriptors
            hipblasLtMatmulDesc_t operationDesc = nullptr;
            hipblasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
            hipblasLtMatmulPreference_t preference = nullptr;

            // =================================================================
            // Create operation descriptor with epilogue
            // =================================================================

            HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&operationDesc,
                                                      HIPBLAS_COMPUTE_32F,
                                                      HIP_R_32F));

            // Set transpose operations - same logic as for cuBLASLt
            hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
            hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

            HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                            HIPBLASLT_MATMUL_DESC_TRANSA,
                                                            &opB, sizeof(opB)));
            HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                            HIPBLASLT_MATMUL_DESC_TRANSB,
                                                            &opA, sizeof(opA)));

            // Set epilogue with bias
            hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
            HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                            HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                            &epilogue, sizeof(epilogue)));

            // Set bias pointer
            HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                            HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                            &d_bias, sizeof(d_bias)));

            // =================================================================
            // Create matrix layouts
            // =================================================================

            int lda = K;
            int ldb = transB ? K : N;
            int ldc = N;

            // Similar to cuBLASLt: we swap A and B for row-major handling
            if (transB)
            {
                // B[N×K] row-major → hipBLASLt sees [K×N] col-major
                HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_32F, K, N, ldb));
            }
            else
            {
                // B[K×N] row-major → hipBLASLt sees [N×K] col-major
                HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_32F, N, K, ldb));
            }

            // Create layout for "B" in hipBLASLt call (which is our A[M×K])
            HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&Bdesc, HIP_R_32F, K, M, lda));

            // Create layout for C (and D): C[M×N] row-major → [N×M] col-major
            HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&Cdesc, HIP_R_32F, N, M, ldc));

            // =================================================================
            // Set up algorithm heuristics
            // =================================================================

            HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));

            /*
             * hipBLASLt may use this workspace during warmup, capture, and
             * replay. Its address is therefore part of the graph contract and
             * must be bound before execution rather than lazily allocated here.
             */
            const size_t workspaceSize = kBiasMatmulWorkspaceBytes;
            if (!workspace_)
            {
                HIP_LOG_ERROR(
                    "[HipBLASGemmKernel::execute_with_bias] Required graph workspace is not bound");
                hipblasLtMatmulPreferenceDestroy(preference);
                hipblasLtMatrixLayoutDestroy(Cdesc);
                hipblasLtMatrixLayoutDestroy(Bdesc);
                hipblasLtMatrixLayoutDestroy(Adesc);
                hipblasLtMatmulDescDestroy(operationDesc);
                return false;
            }
            void *workspace = workspace_->getBuffer(kBiasMatmulWorkspace);
            if (!workspace ||
                workspace_->getBufferSize(kBiasMatmulWorkspace) < workspaceSize)
            {
                HIP_LOG_ERROR(
                    "[HipBLASGemmKernel::execute_with_bias] Missing or undersized graph workspace");
                hipblasLtMatmulPreferenceDestroy(preference);
                hipblasLtMatrixLayoutDestroy(Cdesc);
                hipblasLtMatrixLayoutDestroy(Bdesc);
                hipblasLtMatrixLayoutDestroy(Adesc);
                hipblasLtMatmulDescDestroy(operationDesc);
                return false;
            }

            HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
                                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                                  &workspaceSize, sizeof(workspaceSize)));

            // Get best algorithm
            int returnedResults = 0;
            hipblasLtMatmulHeuristicResult_t heuristicResult = {};
            HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(ltHandle,
                                                            operationDesc,
                                                            Adesc, Bdesc, Cdesc, Cdesc,
                                                            preference,
                                                            1, &heuristicResult, &returnedResults));

            if (returnedResults == 0)
            {
                HIP_LOG_ERROR("[HipBLASGemmKernel::execute_with_bias] No suitable algorithm found");
                hipblasLtMatmulPreferenceDestroy(preference);
                hipblasLtMatrixLayoutDestroy(Cdesc);
                hipblasLtMatrixLayoutDestroy(Bdesc);
                hipblasLtMatrixLayoutDestroy(Adesc);
                hipblasLtMatmulDescDestroy(operationDesc);
                return false;
            }

            // =================================================================
            // Execute the matmul with fused bias
            // =================================================================

            hipblasStatus_t status = hipblasLtMatmul(ltHandle,
                                                     operationDesc,
                                                     &alpha,
                                                     d_B, Adesc, // A in hipBLASLt = our B
                                                     d_A, Bdesc, // B in hipBLASLt = our A
                                                     &beta,
                                                     d_C, Cdesc, // C
                                                     d_C, Cdesc, // D (output, same as C)
                                                     &heuristicResult.algo,
                                                     workspace, workspaceSize,
                                                     static_cast<hipStream_t>(stream.get()));

            // Cleanup descriptor state; the graph workspace remains setup-owned.
            hipblasLtMatmulPreferenceDestroy(preference);
            hipblasLtMatrixLayoutDestroy(Cdesc);
            hipblasLtMatrixLayoutDestroy(Bdesc);
            hipblasLtMatrixLayoutDestroy(Adesc);
            hipblasLtMatmulDescDestroy(operationDesc);

            if (status != HIPBLAS_STATUS_SUCCESS)
            {
                HIP_LOG_ERROR("[HipBLASGemmKernel::execute_with_bias] hipblasLtMatmul failed: " << status);
                return false;
            }

            return true;
        }

        WorkspaceRequirements HipBLASGemmKernel::getWorkspaceRequirements(
            int, int, int) const
        {
            WorkspaceRequirements requirements;
            requirements.buffers.push_back(
                {kBiasMatmulWorkspace, kBiasMatmulWorkspaceBytes, 256, true});
            return requirements;
        }

        // =====================================================================
        // FP16 GEMM Implementation
        // =====================================================================

        bool HipBLASGemmKernel::execute_fp16(
            const void *d_A, const void *d_B, void *d_C,
            int M, int N, int K,
            bool transA, bool transB,
            float alpha, float beta)
        {
            try
            {
                return executeFP16OnStream(
                    ExplicitGPUStream{
                        requireStream("HipBLASGemmKernel::execute_fp16")},
                    d_A, d_B, d_C,
                    M, N, K,
                    transA, transB,
                    alpha, beta);
            }
            catch (const std::exception &exception)
            {
                HIP_LOG_ERROR(exception.what());
                return false;
            }
        }

        bool HipBLASGemmKernel::executeFP16OnStream(
            ExplicitGPUStream stream,
            const void *d_A, const void *d_B, void *d_C,
            int M, int N, int K,
            bool transA, bool transB,
            float alpha, float beta)
        {
            const auto submission = device_ctx_->acquireBlasSubmission();
            void *const handle_ = submission.handle();
            if (!handle_)
            {
                HIP_LOG_ERROR("[HipBLASGemmKernel::execute_fp16] hipBLAS handle is null");
                return false;
            }

            // Ensure we're on the correct device
            hipError_t hip_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(device_id_.ordinal));
            if (hip_err != hipSuccess)
            {
                HIP_LOG_ERROR("[HipBLASGemmKernel::execute_fp16] Failed to set device: " << hipGetErrorString(hip_err));
                return false;
            }

            const hipblasStatus_t stream_status = hipblasSetStream(
                static_cast<hipblasHandle_t>(handle_),
                static_cast<hipStream_t>(stream.get()));
            if (stream_status != HIPBLAS_STATUS_SUCCESS)
            {
                HIP_LOG_ERROR(
                    "[HipBLASGemmKernel::execute_fp16] hipblasSetStream failed: "
                    << static_cast<int>(stream_status));
                return false;
            }

            hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
            hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

            int lda = K;
            int ldb = transB ? K : N;
            int ldc = N;

            if (!bindMatmulWorkspace(handle_, workspace_))
                return false;

            // Convert alpha/beta to half precision
            hipblasHalf alpha_h = __float2half(alpha);
            hipblasHalf beta_h = __float2half(beta);

            // hipblasHgemm for native FP16 computation
            HIPBLAS_CHECK(hipblasHgemm(
                static_cast<hipblasHandle_t>(handle_),
                opB, opA,
                N, M, K,
                &alpha_h,
                static_cast<const hipblasHalf *>(d_B), ldb,
                static_cast<const hipblasHalf *>(d_A), lda,
                &beta_h,
                static_cast<hipblasHalf *>(d_C), ldc));

            return true;
        }

        // =====================================================================
        // Factory function
        // =====================================================================

        std::unique_ptr<HipBLASGemmKernel> createHipBLASGemm(
            const DeviceId &device_id,
            HipBLASGemmKernel::Precision precision)
        {
            return std::make_unique<HipBLASGemmKernel>(device_id, precision);
        }

        void HipBLASGemmKernel::bindStream(ExplicitGPUStream stream)
        {
            ROCmKernelBase::bindGPUStream(stream);
            /*
             * The cached handle is shared. Mutating it here would create a
             * bind/submit race with another floating projection. Every launch
             * instead binds inside its dispatch lock via an OnStream method.
             */
        }

        void HipBLASGemmKernel::clearStreamBinding() noexcept
        {
            ROCmKernelBase::clearGPUStreamBinding();
        }


#else // !HAVE_ROCM

        // Stub implementations when ROCm is not available

        HipBLASGemmKernel::HipBLASGemmKernel(const DeviceId &device_id, Precision precision)
            : device_id_(device_id), precision_(precision)
        {
            throw std::runtime_error("[HipBLASGemmKernel] ROCm support not compiled");
        }

        HipBLASGemmKernel::HipBLASGemmKernel(IWorkerGPUContext * /*ctx*/, Precision precision)
            : precision_(precision)
        {
            throw std::runtime_error("[HipBLASGemmKernel] ROCm support not compiled");
        }

        HipBLASGemmKernel::~HipBLASGemmKernel() {}

        HipBLASGemmKernel::HipBLASGemmKernel(HipBLASGemmKernel &&) noexcept = default;
        HipBLASGemmKernel &HipBLASGemmKernel::operator=(HipBLASGemmKernel &&) noexcept = default;

        bool HipBLASGemmKernel::execute(
            const float *, const float *, float *,
            int, int, int,
            bool, bool, float, float)
        {
            return false;
        }

        bool HipBLASGemmKernel::executeOnStream(
            ExplicitGPUStream,
            const float *, const float *, float *,
            int, int, int,
            bool, bool, float, float)
        {
            return false;
        }

        bool HipBLASGemmKernel::execute_batched(
            const float *const *, const float *const *, float *const *,
            int, int, int, int,
            bool, bool, float, float)
        {
            return false;
        }

        bool HipBLASGemmKernel::executeBatchedOnStream(
            ExplicitGPUStream,
            const float *const *, const float *const *, float *const *,
            int, int, int, int,
            bool, bool, float, float)
        {
            return false;
        }

        WorkspaceRequirements HipBLASGemmKernel::getWorkspaceRequirements(
            int, int, int) const
        {
            return WorkspaceRequirements{};
        }

        bool HipBLASGemmKernel::execute_with_bias(
            const float *, const float *, float *,
            const float *,
            int, int, int,
            bool, bool, float, float)
        {
            return false;
        }

        bool HipBLASGemmKernel::executeWithBiasOnStream(
            ExplicitGPUStream,
            const float *, const float *, float *, const float *,
            int, int, int,
            bool, bool, float, float)
        {
            return false;
        }

        bool HipBLASGemmKernel::execute_fp16(
            const void *, const void *, void *,
            int, int, int,
            bool, bool, float, float)
        {
            return false;
        }

        bool HipBLASGemmKernel::executeFP16OnStream(
            ExplicitGPUStream,
            const void *, const void *, void *,
            int, int, int,
            bool, bool, float, float)
        {
            return false;
        }

        std::unique_ptr<HipBLASGemmKernel> createHipBLASGemm(const DeviceId &, HipBLASGemmKernel::Precision)
        {
            return nullptr;
        }

        std::unique_ptr<HipBLASGemmKernel> createHipBLASGemm(IWorkerGPUContext *, HipBLASGemmKernel::Precision)
        {
            return nullptr;
        }

        void HipBLASGemmKernel::bindStream(ExplicitGPUStream) {}
        void HipBLASGemmKernel::clearStreamBinding() noexcept {}

#endif // HAVE_ROCM

    } // namespace rocm
} // namespace llaminar2
