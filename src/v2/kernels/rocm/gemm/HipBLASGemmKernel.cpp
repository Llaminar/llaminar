/**
 * @file HipBLASGemmKernel.cpp
 * @brief hipBLAS GEMM kernel implementation for AMD GPUs
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
            : device_id_(device_id), precision_(precision), owns_handle_(true)
        {
            if (device_id.type != DeviceType::ROCm)
            {
                throw std::runtime_error(
                    "[HipBLASGemmKernel] Requires ROCm device, got: " + device_id.to_string());
            }

            // Set device
            hipError_t hip_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(device_id_.ordinal));
            if (hip_err != hipSuccess)
            {
                throw std::runtime_error(
                    std::string("[HipBLASGemmKernel] Failed to set HIP device ") +
                    std::to_string(device_id_.ordinal) + ": " + hipGetErrorString(hip_err));
            }

            // Create hipBLAS handle
            hipblasHandle_t temp_handle = nullptr;
            hipblasStatus_t hipblas_err = hipblasCreate(&temp_handle);
            if (hipblas_err != HIPBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    std::string("[HipBLASGemmKernel] Failed to create hipBLAS handle: ") +
                    std::to_string(static_cast<int>(hipblas_err)));
            }
            handle_ = static_cast<void *>(temp_handle);

            // Create hipBLASLt handle for fused operations (e.g., GEMM + bias)
            hipblasLtHandle_t temp_lt_handle = nullptr;
            hipblasStatus_t lt_err = hipblasLtCreate(&temp_lt_handle);
            if (lt_err != HIPBLAS_STATUS_SUCCESS)
            {
                hipblasDestroy(temp_handle);
                throw std::runtime_error(
                    std::string("[HipBLASGemmKernel] Failed to create hipBLASLt handle: ") +
                    std::to_string(static_cast<int>(lt_err)));
            }
            lt_handle_ = static_cast<void *>(temp_lt_handle);

            // Note: hipBLAS doesn't have explicit Tensor Core mode like cuBLAS
            // It will automatically use the best available math mode

            HIP_LOG_DEBUG("[HipBLASGemmKernel] Created on device " << device_id_.to_string()
                                                                   << " with precision "
                                                                   << static_cast<int>(precision_)
                                                                   << " (owns handle)");
        }

        HipBLASGemmKernel::HipBLASGemmKernel(IWorkerGPUContext *ctx, Precision precision)
            : precision_(precision), owns_handle_(false)
        {
            if (!ctx)
            {
                throw std::runtime_error(
                    "[HipBLASGemmKernel] Device context is null");
            }

            if (!ctx->isInitialized())
            {
                throw std::runtime_error(
                    "[HipBLASGemmKernel] Device context is not initialized");
            }

            // Store the device context (sets device_ctx_ in base class)
            setDeviceContext(ctx);
            device_id_ = DeviceId::rocm(ctx->deviceOrdinal());

            // Get hipBLAS handle from context via submitAndWait
            // blasHandle() must be called from the worker thread per thread-safety model
            void *blas_handle = nullptr;
            std::exception_ptr eptr = nullptr;
            ctx->submitAndWait([&]()
                               {
                try {
                    blas_handle = ctx->blasHandle();
                } catch (...) {
                    eptr = std::current_exception();
                } });
            if (eptr)
            {
                std::rethrow_exception(eptr);
            }
            if (!blas_handle)
            {
                throw std::runtime_error(
                    "[HipBLASGemmKernel] Device context has no hipBLAS handle");
            }
            handle_ = blas_handle;

            // Get hipBLASLt handle from context (required for fused operations)
            void *lt_handle_ptr = nullptr;
            ctx->submitAndWait([&]()
                               { lt_handle_ptr = ctx->blasLtHandle(); });
            if (!lt_handle_ptr)
            {
                throw std::runtime_error(
                    "[HipBLASGemmKernel] Device context has no hipBLASLt handle");
            }
            lt_handle_ = lt_handle_ptr;
            owns_lt_handle_ = false;

            HIP_LOG_DEBUG("[HipBLASGemmKernel] Created on device " << device_id_.to_string()
                                                                   << " with precision "
                                                                   << static_cast<int>(precision_)
                                                                   << " (using context handle)");
        }

        HipBLASGemmKernel::~HipBLASGemmKernel()
        {
            // Free cached workspace
            if (lt_workspace_)
            {
                hipFree(lt_workspace_);
                lt_workspace_ = nullptr;
                lt_workspace_size_ = 0;
            }
            // Only destroy lt_handle_ if we own it
            if (owns_lt_handle_ && lt_handle_)
            {
                hipblasLtDestroy(static_cast<hipblasLtHandle_t>(lt_handle_));
                lt_handle_ = nullptr;
            }
            // Only destroy hipBLAS handle if we own it
            if (owns_handle_ && handle_)
            {
                hipblasDestroy(static_cast<hipblasHandle_t>(handle_));
                handle_ = nullptr;
            }
        }

        // Move constructor
        HipBLASGemmKernel::HipBLASGemmKernel(HipBLASGemmKernel &&other) noexcept
            : ROCmKernelBase(std::move(other)),
              handle_(other.handle_),
              lt_handle_(other.lt_handle_),
              device_id_(other.device_id_),
              precision_(other.precision_),
              owns_handle_(other.owns_handle_),
              owns_lt_handle_(other.owns_lt_handle_),
              lt_workspace_(other.lt_workspace_),
              lt_workspace_size_(other.lt_workspace_size_)
        {
            other.handle_ = nullptr;
            other.lt_handle_ = nullptr;
            other.owns_handle_ = false;    // Moved-from object shouldn't destroy anything
            other.owns_lt_handle_ = false; // Moved-from object shouldn't destroy anything
            other.lt_workspace_ = nullptr;
            other.lt_workspace_size_ = 0;
        }

        // Move assignment
        HipBLASGemmKernel &HipBLASGemmKernel::operator=(HipBLASGemmKernel &&other) noexcept
        {
            if (this != &other)
            {
                // Destroy our resources if we own them
                if (lt_workspace_)
                {
                    hipFree(lt_workspace_);
                }
                if (owns_lt_handle_ && lt_handle_)
                {
                    hipblasLtDestroy(static_cast<hipblasLtHandle_t>(lt_handle_));
                }
                if (owns_handle_ && handle_)
                {
                    hipblasDestroy(static_cast<hipblasHandle_t>(handle_));
                }

                // Move base class
                ROCmKernelBase::operator=(std::move(other));

                // Take ownership of other's resources
                handle_ = other.handle_;
                lt_handle_ = other.lt_handle_;
                device_id_ = other.device_id_;
                precision_ = other.precision_;
                owns_handle_ = other.owns_handle_;
                owns_lt_handle_ = other.owns_lt_handle_;
                lt_workspace_ = other.lt_workspace_;
                lt_workspace_size_ = other.lt_workspace_size_;

                other.handle_ = nullptr;
                other.lt_handle_ = nullptr;
                other.owns_handle_ = false;
                other.owns_lt_handle_ = false;
                other.lt_workspace_ = nullptr;
                other.lt_workspace_size_ = 0;
            }
            return *this;
        }

        // =====================================================================
        // FP32 GEMM Implementation
        // =====================================================================

        bool HipBLASGemmKernel::execute(
            const float *d_A, const float *d_B, float *d_C,
            int M, int N, int K,
            bool transA, bool transB,
            float alpha, float beta)
        {
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

            // Request workspace (cached to avoid per-call hipMalloc/hipFree)
            size_t workspaceSize = 4 * 1024 * 1024; // 4MB workspace
            if (!lt_workspace_ || lt_workspace_size_ < workspaceSize)
            {
                if (lt_workspace_)
                    hipFree(lt_workspace_);
                HIP_CHECK(hipMalloc(&lt_workspace_, workspaceSize));
                lt_workspace_size_ = workspaceSize;
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
                                                     lt_workspace_, lt_workspace_size_,
                                                     0); // stream = 0 (default)

            // Cleanup (workspace is cached, not freed here)
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

        // =====================================================================
        // FP16 GEMM Implementation
        // =====================================================================

        bool HipBLASGemmKernel::execute_fp16(
            const void *d_A, const void *d_B, void *d_C,
            int M, int N, int K,
            bool transA, bool transB,
            float alpha, float beta)
        {
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

            hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
            hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

            int lda = K;
            int ldb = transB ? K : N;
            int ldc = N;

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

        void HipBLASGemmKernel::setStream(void *stream)
        {
            if (handle_)
            {
                hipblasSetStream(static_cast<hipblasHandle_t>(handle_),
                                 static_cast<hipStream_t>(stream));
            }
        }

        void registerHipBLASGemmKernelFactory()
        {
            DeviceKernelCache::registerFactory(
                DeviceType::ROCm,
                KernelType::BLAS_GEMM,
                [](const DeviceId &device) -> std::unique_ptr<IDeviceKernel>
                {
                    return std::make_unique<HipBLASGemmKernel>(device);
                });
            HIP_LOG_DEBUG("[HipBLASGemmKernel] Registered factory for ROCm BLAS_GEMM");
        }

#else // !HAVE_ROCM

        // Stub implementations when ROCm is not available

        HipBLASGemmKernel::HipBLASGemmKernel(const DeviceId &device_id, Precision precision)
            : device_id_(device_id), precision_(precision), owns_handle_(true)
        {
            throw std::runtime_error("[HipBLASGemmKernel] ROCm support not compiled");
        }

        HipBLASGemmKernel::HipBLASGemmKernel(IWorkerGPUContext * /*ctx*/, Precision precision)
            : precision_(precision), owns_handle_(false)
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

        bool HipBLASGemmKernel::execute_with_bias(
            const float *, const float *, float *,
            const float *,
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

        std::unique_ptr<HipBLASGemmKernel> createHipBLASGemm(const DeviceId &, HipBLASGemmKernel::Precision)
        {
            return nullptr;
        }

        std::unique_ptr<HipBLASGemmKernel> createHipBLASGemm(IWorkerGPUContext *, HipBLASGemmKernel::Precision)
        {
            return nullptr;
        }

        void registerHipBLASGemmKernelFactory() {}

        void HipBLASGemmKernel::setStream(void *) {}

#endif // HAVE_ROCM

    } // namespace rocm
} // namespace llaminar2
