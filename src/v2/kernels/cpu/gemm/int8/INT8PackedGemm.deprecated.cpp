/**
 * @file INT8PackedGemm.cpp
 * @brief INT8×INT8→INT32 GEMM with AVX512 VNNI auto-tuned micro-kernels
 *
 * Architecture:
 * - INT8Tensor implements ITensorGemmTileDataProvider interface for INT8 data
 * - INT8MicroKernelAdapter uses INT8-specific micro-kernels with VNNI
 * - Auto-tuner selects optimal tile/unroll/prefetch for each (m,n,k) triple
 * - Packing functions prepare INT8 data for VNNI efficiency (groups of 4)
 * - Accumulation: INT32 results (NO dequantization in kernel)
 * - Caller must dequantize INT32→FP32 separately if needed
 *
 * This design allows:
 * - Accumulating multiple INT32 matmuls before dequantization
 * - Fused operations (e.g., matmul + bias) at INT32 precision
 * - Delayed dequantization for better numerical stability
 *
 * @author David Sanftenberg
 */

#include "INT8PackedGemm.h"
#include "../GemmAutoTuner.h"
#include "../GemmMicroKernelAdapter.h"
#include "../GemmMicroKernelRegistry.h"
#include "../GemmMicroKernelMacros.h"
#include "GemmMicroKernelTemplateINT8.h"
#include "../../../../tensors/Tensors.h" // For INT8Tensor
#include "../../../../utils/Logger.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include <mutex>
#include <omp.h>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {
            /**
             * @brief Forward declaration of INT8Tensor (will be implemented later)
             *
             * For now, we'll work with raw int8_t pointers and assume INT8Tensor
             * will be similar to BF16Tensor structure.
             */
            class INT8Tensor; // Forward declaration

            /**
             * @brief Adapter that wraps INT8 VNNI micro-kernel with packing
             *
             * Handles INT8×INT8→INT32 computation using AVX512 VNNI, then dequantizes
             * int32 results to float output.
             *
             * Key differences from BF16 adapter:
             * - Uses INT8-specific micro-kernel (dpbusd instruction)
             * - Requires int32 accumulators instead of float
             * - Needs dequantization parameters (scale, zero_point)
             * - VNNI-friendly packing (groups of 4)
             */
            class INT8MicroKernelAdapter : public llaminar::v2::kernels::IQuantizedGemmVariant
            {
            public:
                INT8MicroKernelAdapter(
                    int mr, int nr, int unroll_k, int prefetch_dist)
                    : mr_(mr),
                      nr_(nr),
                      unroll_k_(unroll_k),
                      prefetch_dist_(prefetch_dist)
                {
                    // Build name
                    std::ostringstream oss;
                    oss << "INT8_AVX512VNNI_"
                        << mr << "x" << nr << "_u" << unroll_k << "_p" << prefetch_dist;
                    name_ = oss.str();

                    // Select cache tile sizes
                    if (mr * nr >= 32)
                    {
                        mc_ = 256;
                        nc_ = 256;
                        kc_ = 512;
                    }
                    else if (mr * nr >= 16)
                    {
                        mc_ = 128;
                        nc_ = 128;
                        kc_ = 256;
                    }
                    else
                    {
                        mc_ = 64;
                        nc_ = 64;
                        kc_ = 128;
                    }

                    // Ensure KC is multiple of 4 for VNNI
                    kc_ = (kc_ / 4) * 4;
                }

                const char *name() const override
                {
                    return name_.c_str();
                }

                llaminar::v2::kernels::GemmKernelConfig config() const override
                {
                    llaminar::v2::kernels::GemmKernelConfig cfg;
                    cfg.unroll_factor = unroll_k_;
                    cfg.prefetch_blocks = prefetch_dist_;
                    cfg.tile_m = mr_;
                    cfg.tile_n = nr_;
                    return cfg;
                }

                bool multiply(
                    const float *A_int8_as_float, // INT8 data cast to float* (reinterpret)
                    float *C_float,               // Will be reinterpreted as int32_t*
                    int m, int n, int k,
                    const ITensorGemmTileDataProvider *decoder, // Points to INT8Tensor
                    bool transpose_B,
                    float alpha = 1.0f,
                    float beta = 0.0f) override
                {
                    // Extract INT8Tensor from decoder
                    const auto *B_tensor = dynamic_cast<const llaminar2::INT8Tensor *>(decoder);
                    if (!B_tensor)
                    {
                        LOG_ERROR("INT8MicroKernelAdapter: decoder must be INT8Tensor");
                        return false;
                    }

                    // Get INT8 data pointers
                    const int8_t *A_int8 = reinterpret_cast<const int8_t *>(A_int8_as_float);
                    const int8_t *B_int8 = B_tensor->int8_data();

                    // Reinterpret C as int32_t* for direct INT32 accumulation
                    int32_t *C = reinterpret_cast<int32_t *>(C_float);

                    // Convert alpha/beta to int32 for integer scaling
                    // Note: Caller should ensure alpha/beta are appropriate for int32 operations
                    int32_t alpha_i32 = static_cast<int32_t>(alpha);
                    int32_t beta_i32 = static_cast<int32_t>(beta);

                    // Parallel loop over N dimension with cache blocking
#pragma omp parallel
                    {
                        // Thread-local buffers with alignment for VNNI
                        constexpr size_t VNNI_ALIGN = 64; // 64-byte alignment for AVX512

                        std::vector<int8_t> A_packed_local(mc_ * kc_ + VNNI_ALIGN);
                        std::vector<int8_t> B_packed_local(kc_ * nc_ + VNNI_ALIGN);

#pragma omp for schedule(dynamic)
                        for (int jc = 0; jc < n; jc += nc_)
                        {
                            int nc = std::min(nc_, n - jc);

                            for (int pc = 0; pc < k; pc += kc_)
                            {
                                int kc = std::min(kc_, k - pc);

                                // For K-panel accumulation:
                                // - First panel (pc=0): Use caller's beta (typically 0 to initialize)
                                // - Subsequent panels: Use beta=1 to accumulate
                                int32_t beta_panel = (pc == 0) ? beta_i32 : 1;

                                // Step 1: Pack B panel [kc × nc]
                                packBPanel_INT8(
                                    B_int8, B_packed_local.data(),
                                    kc, nc, jc, n, pc, transpose_B);

                                // Step 2: Loop over M dimension with cache blocking
                                for (int ic = 0; ic < m; ic += mc_)
                                {
                                    int mc = std::min(mc_, m - ic);

                                    // Step 3: Pack A panel [mc × kc]
                                    packAPanel_INT8(
                                        A_int8 + ic * k + pc, A_packed_local.data(),
                                        mc, kc, k);

                                    // Step 4: Compute C[ic:ic+mc, jc:jc+nc] using INT8 VNNI micro-kernel
                                    for (int ir = 0; ir < mc; ir += mr_)
                                    {
                                        int mr = std::min(mr_, mc - ir);

                                        for (int jr = 0; jr < nc; jr += nr_)
                                        {
                                            int nr = std::min(nr_, nc - jr);

                                            int32_t *C_tile = C + (ic + ir) * n + (jc + jr);

                                            // Call INT8 VNNI micro-kernel (now returns int32)
                                            using ISA = simd::AVX512VNNITag;
                                            MicroKernelTemplateINT8<ISA, 8, 8, 4, 2, 256, 512, 128>::micro_kernel(
                                                A_packed_local.data() + ir * kc,
                                                B_packed_local.data() + jr * kc,
                                                C_tile,
                                                n,          // ldc
                                                kc,         // k_panel
                                                alpha_i32,  // INT32 alpha
                                                beta_panel, // INT32 beta (0 for first panel, 1 for accumulation)
                                                mr, nr);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    return true;
                }

            private:
                std::string name_;
                int mr_, nr_, unroll_k_, prefetch_dist_;
                int mc_, nc_, kc_;

                /**
                 * @brief Pack A panel [m_panel × k] for VNNI efficiency
                 *
                 * Input: int8_t A[m][k] (row-major)
                 * Output: int8_t A_packed[m][k] (contiguous, VNNI-friendly)
                 *
                 * VNNI works best with data in groups of 4, but simple row-major
                 * packing is sufficient for the A matrix.
                 */
                void packAPanel_INT8(
                    const int8_t *A_int8,
                    int8_t *A_packed,
                    int m_panel, int k_panel, int lda)
                {
                    // Simple row-major packing (same as BF16 but with int8)
                    for (int i = 0; i < m_panel; ++i)
                    {
                        const int8_t *A_row = A_int8 + i * lda;
                        int8_t *A_packed_row = A_packed + i * k_panel;

                        // Prefetch next row
                        if (i + 1 < m_panel)
                        {
                            __builtin_prefetch(A_int8 + (i + 1) * lda, 0, 3);
                        }

                        // Copy with unrolling (groups of 64 for cache line efficiency)
                        std::memcpy(A_packed_row, A_row, k_panel);
                    }
                }

                /**
                 * @brief Pack B panel [k × n_panel] for VNNI efficiency
                 *
                 * Handles both transposed and non-transposed B matrices.
                 * Data is arranged for optimal VNNI access patterns.
                 */
                void packBPanel_INT8(
                    const int8_t *B_int8,
                    int8_t *B_packed,
                    int k_panel, int n_panel, int j_offset, int ldb, int k_offset,
                    bool transpose_B)
                {
                    if (transpose_B)
                    {
                        // B is [n, k], we need B^T = [k, n]
                        // B_int8[j, k_idx] → B_packed[j * k_panel + k_idx]
                        for (int j = 0; j < n_panel; ++j)
                        {
                            const int8_t *B_row = B_int8 + (j_offset + j) * ldb + k_offset;
                            int8_t *B_packed_col = B_packed + j * k_panel;
                            std::memcpy(B_packed_col, B_row, k_panel);
                        }
                    }
                    else
                    {
                        // B is [k, n], pack column-major: B_packed[j * k_panel + k_idx] = B[k_idx, j]
                        for (int j = 0; j < n_panel; ++j)
                        {
                            for (int k_idx = 0; k_idx < k_panel; ++k_idx)
                            {
                                const int8_t *B_row = B_int8 + (k_offset + k_idx) * ldb;
                                B_packed[j * k_panel + k_idx] = B_row[j_offset + j];
                            }
                        }
                    }
                }
            };

            /**
             * @brief Register all INT8 VNNI micro-kernel variants
             */
            std::vector<std::unique_ptr<llaminar::v2::kernels::IQuantizedGemmVariant>>
            registerINT8MicroKernelVariants()
            {
                std::vector<std::unique_ptr<llaminar::v2::kernels::IQuantizedGemmVariant>> variants;

                // INT8 VNNI only available on AVX512VNNI
                const std::vector<int> mr_values = {1, 2, 4, 8, 16, 32};
                const std::vector<int> nr_values = {1, 2, 4, 6, 8, 16, 32};
                const std::vector<int> unroll_values = {1, 2, 4, 8};
                const std::vector<int> prefetch_values = {0, 1, 2, 3};

                LOG_DEBUG("Registering INT8 VNNI micro-kernel variants");

                // Enumerate all combinations
                for (int mr : mr_values)
                {
                    for (int nr : nr_values)
                    {
                        // Register constraints (similar to AVX512 FP32)
                        if (mr * nr > 48)
                            continue;

                        for (int unroll_k : unroll_values)
                        {
                            for (int prefetch : prefetch_values)
                            {
                                // Create INT8 adapter
                                variants.push_back(
                                    std::make_unique<INT8MicroKernelAdapter>(
                                        mr, nr, unroll_k, prefetch));
                            }
                        }
                    }
                }

                LOG_DEBUG("Registered " << variants.size() << " INT8 VNNI micro-kernel variants");

                return variants;
            }

            /**
             * @brief Auto-tuned INT8×INT8→INT32 GEMM kernel
             */
            class AutoTunedINT8PackedGemm : public ITensorGemm
            {
            public:
                explicit AutoTunedINT8PackedGemm(const llaminar2::INT8Tensor *weight_tensor)
                    : weight_tensor_(weight_tensor)
                {
                    ensureVariantsRegistered();
                }

                /**
                 * @brief INT8×INT8→INT32 GEMM: C = A_int8 @ B_int8 (output in INT32)
                 *
                 * @param A Pointer to INT8 activations (cast from float* for interface compatibility)
                 * @param C Pointer to INT32 output (cast from float* for interface compatibility)
                 * @param m Number of rows in A and C
                 * @param n Number of columns in output C
                 * @param k Number of columns in A / rows in B
                 * @param transpose_B Whether B (weight) is transposed
                 * @param alpha Scaling factor (passed to micro-kernel, typically 1.0 for INT32)
                 * @param beta Scaling factor (passed to micro-kernel, typically 0.0 for INT32)
                 * @param mpi_ctx MPI context (unused)
                 * @param device_idx Device index (must be -1 for CPU)
                 *
                 * @return true on success
                 *
                 * @note A is actually int8_t*, C is actually int32_t* - cast via reinterpret_cast in caller
                 * @note Caller must dequantize INT32→FP32 using per-channel scales separately
                 * @note variant->multiply() internally reinterprets pointers, so we pass them as-is
                 */
                bool multiply(
                    const float *A, float *C,
                    int m, int n, int k,
                    bool transpose_B,
                    float alpha, float beta,
                    const MPIContext *mpi_ctx,
                    int device_idx) override
                {
                    (void)mpi_ctx;

                    if (device_idx != -1)
                    {
                        LOG_ERROR("AutoTunedINT8PackedGemm: Only CPU supported (device_idx must be -1)");
                        return false;
                    }

                    // Get or select optimal variant for this problem size
                    auto variant = selectVariant(m, n, k);
                    if (!variant)
                    {
                        LOG_ERROR("AutoTunedINT8PackedGemm: No suitable variant found");
                        return false;
                    }

                    // Execute INT8 GEMM with weight tensor as decoder
                    // variant->multiply expects (float *A, float *C, ...) interface
                    // but internally reinterprets them as int8_t* and int32_t*
                    // Pass alpha/beta through (micro-kernel converts to int32_t internally)
                    bool debug = (std::getenv("LLAMINAR_INT8_VNNI_DEBUG") != nullptr);
                    if (debug)
                    {
                        // Compute scalar reference directly on original A (row-major m×k) and B (k×n)
                        const int8_t *A_int8 = reinterpret_cast<const int8_t *>(A);
                        const int8_t *B_int8 = weight_tensor_->int8_data();
                        // First element only to limit overhead
                        long long ref = 0;
                        for (int kk = 0; kk < k; ++kk)
                        {
                            ref += static_cast<long long>(A_int8[0 * k + kk]) * static_cast<long long>(B_int8[kk * n + 0]);
                        }
                        std::cerr << "[INT8_GEMM_DEBUG] Pre-pack scalar C(0,0) ref=" << ref << std::endl;
                    }
                    bool ok = variant->multiply(A, C, m, n, k, weight_tensor_, transpose_B, alpha, beta);
                    if (debug)
                    {
                        const int32_t *C_i32 = reinterpret_cast<const int32_t *>(C);
                        std::cerr << "[INT8_GEMM_DEBUG] Post-kernel C(0,0)=" << C_i32[0] << std::endl;
                    }
                    return ok;
                }

            private:
                const llaminar2::INT8Tensor *weight_tensor_;

                bool multiply_activations(
                    const float *A, const float *B, float *C,
                    int m, int n, int k,
                    bool transpose_B,
                    float alpha, float beta,
                    const MPIContext *mpi_ctx,
                    int device_idx) override
                {
                    (void)A;
                    (void)B;
                    (void)C;
                    (void)m;
                    (void)n;
                    (void)k;
                    (void)transpose_B;
                    (void)alpha;
                    (void)beta;
                    (void)mpi_ctx;
                    (void)device_idx;

                    LOG_ERROR("AutoTunedINT8PackedGemm: multiply_activations not yet implemented");
                    return false;
                }

                bool multiply_activations_strided(
                    const float *A, const float *B, float *C,
                    int m, int n, int k,
                    int lda, int ldb, int ldc,
                    bool transpose_B,
                    float alpha, float beta,
                    const MPIContext *mpi_ctx,
                    int device_idx) override
                {
                    (void)A;
                    (void)B;
                    (void)C;
                    (void)m;
                    (void)n;
                    (void)k;
                    (void)lda;
                    (void)ldb;
                    (void)ldc;
                    (void)transpose_B;
                    (void)alpha;
                    (void)beta;
                    (void)mpi_ctx;
                    (void)device_idx;

                    LOG_ERROR("AutoTunedINT8PackedGemm: multiply_activations_strided not yet implemented");
                    return false;
                }

                bool supports_device(int device_idx) const override
                {
                    // INT8 VNNI kernels only support CPU (device_idx = -1)
                    return (device_idx == -1);
                }

            private:
                void ensureVariantsRegistered()
                {
                    static std::once_flag flag;
                    std::call_once(flag, []()
                                   {
                        auto variants = registerINT8MicroKernelVariants();
                        auto &tuner = llaminar::v2::kernels::GemmAutoTuner::instance();
                        for (auto &variant : variants)
                        {
                            tuner.registerVariant(std::move(variant));
                        } });
                }

                llaminar::v2::kernels::IQuantizedGemmVariant *selectVariant(int m, int n, int k)
                {
                    auto &tuner = llaminar::v2::kernels::GemmAutoTuner::instance();
                    return tuner.getOptimalKernel(m, n, k);
                }
            };

            // ========== PUBLIC API ==========

            bool isINT8GemmSupported()
            {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
                return true;
#else
                return false;
#endif
            }

            std::unique_ptr<ITensorGemm> createINT8PackedGemm(
                const TensorBase *A,
                const TensorBase *B)
            {
                (void)A;

                if (!isINT8GemmSupported())
                {
                    LOG_ERROR("INT8 GEMM not supported (requires AVX512F + AVX512VNNI)");
                    return nullptr;
                }

                // Cast B to INT8Tensor
                const auto *weight_tensor = dynamic_cast<const llaminar2::INT8Tensor *>(B);
                if (!weight_tensor)
                {
                    LOG_ERROR("createINT8PackedGemm: B tensor must be INT8Tensor");
                    return nullptr;
                }

                LOG_INFO("Creating AutoTunedINT8PackedGemm kernel");
                return std::make_unique<AutoTunedINT8PackedGemm>(weight_tensor);
            }

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
