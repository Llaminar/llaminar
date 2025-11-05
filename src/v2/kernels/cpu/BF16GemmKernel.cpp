/**
 * @file BF16GemmKernel.cpp
 * @brief CPU BF16 GEMM kernel implementation
 *
 * Supports:
 * - OneDNN bf16bf16f32 matmul (preferred, hardware-accelerated on AVX-512 BF16)
 * - Fallback: BF16→FP32 expansion + cblas_sgemm (OpenBLAS, portable)
 *
 * @author David Sanftenberg
 */

#include "BF16GemmKernel.h"
#include "../../utils/BFloat16.h"
#include "../../tensors/SIMDHelpers.h"

#ifdef HAVE_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#endif

#include <cblas.h>
#include <vector>

namespace llaminar2
{

    // Helper: Batch convert FP32 to BF16
    static void fp32_to_bf16(const float *src, uint16_t *dst, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = simd::fp32_to_bf16(src[i]);
        }
    }

    // Helper: Batch convert BF16 to FP32
    static void bf16_to_fp32(const uint16_t *src, float *dst, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = simd::bf16_to_fp32(src[i]);
        }
    }

    bool BF16GemmKernel::multiply(
        const float *A, float *C,
        int m, int n, int k,
        bool transpose_B,
        float alpha, float beta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

        if (!weight_tensor_)
        {
            return false;
        }

        const uint16_t *B_bf16 = weight_tensor_->bf16_data();
        auto shape = weight_tensor_->shape();

        // Validate dimensions
        if (shape.size() != 2)
        {
            return false;
        }

        int B_rows = shape[0];
        int B_cols = shape[1];

        // Check dimension compatibility
        if (transpose_B)
        {
            // B is [n, k], transposed to [k, n] for multiplication
            if (B_rows != n || B_cols != k)
            {
                return false;
            }
        }
        else
        {
            // B is [k, n], no transpose
            if (B_rows != k || B_cols != n)
            {
                return false;
            }
        }

        // Try OneDNN first if available
#ifdef HAVE_ONEDNN
        using namespace dnnl;

        try
        {
            // Create OneDNN engine (CPU)
            engine eng(engine::kind::cpu, 0);
            stream s(eng);

            // Convert A from FP32 to BF16
            std::vector<uint16_t> A_bf16(m * k);
            fp32_to_bf16(A, A_bf16.data(), m * k);

            // Define memory descriptors
            memory::dims a_dims = {m, k};
            memory::dims b_dims = transpose_B ? memory::dims{n, k} : memory::dims{k, n};
            memory::dims c_dims = {m, n};

            auto a_md = memory::desc(a_dims, memory::data_type::bf16, memory::format_tag::ab);
            auto b_md = transpose_B
                            ? memory::desc(b_dims, memory::data_type::bf16, memory::format_tag::ba)
                            : memory::desc(b_dims, memory::data_type::bf16, memory::format_tag::ab);
            auto c_md = memory::desc(c_dims, memory::data_type::f32, memory::format_tag::ab);

            // Create memory objects
            auto a_mem = memory(a_md, eng, A_bf16.data());
            auto b_mem = memory(b_md, eng, const_cast<uint16_t *>(B_bf16));
            auto c_mem = memory(c_md, eng, C);

            // Create matmul primitive
            matmul::primitive_desc matmul_pd(eng, a_md, b_md, c_md);
            auto matmul_prim = matmul(matmul_pd);

            // Execute (note: OneDNN doesn't support alpha/beta directly in bf16 matmul)
            // We'll handle scaling manually if needed
            if (alpha != 1.0f || beta != 0.0f)
            {
                // Compute into temp buffer, then scale
                std::vector<float> C_temp(m * n);
                auto c_temp_mem = memory(c_md, eng, C_temp.data());

                matmul_prim.execute(s, {{DNNL_ARG_SRC, a_mem},
                                        {DNNL_ARG_WEIGHTS, b_mem},
                                        {DNNL_ARG_DST, c_temp_mem}});
                s.wait();

                // Scale: C = alpha * C_temp + beta * C
                for (int i = 0; i < m * n; ++i)
                {
                    C[i] = alpha * C_temp[i] + beta * C[i];
                }
            }
            else
            {
                matmul_prim.execute(s, {{DNNL_ARG_SRC, a_mem},
                                        {DNNL_ARG_WEIGHTS, b_mem},
                                        {DNNL_ARG_DST, c_mem}});
                s.wait();
            }

            return true; // OneDNN succeeded
        }
        catch (const dnnl::error &e)
        {
            // OneDNN failed, fall through to OpenBLAS
        }
#endif

        // OpenBLAS fallback (used when OneDNN unavailable or fails)
        size_t B_size = static_cast<size_t>(B_rows) * B_cols;
        std::vector<float> B_fp32(B_size);
        bf16_to_fp32(B_bf16, B_fp32.data(), B_size);

        // Standard FP32 GEMM (A is already FP32)
        if (transpose_B)
        {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasTrans,
                m, n, k,
                alpha,
                A, k,             // A is [m, k]
                B_fp32.data(), k, // B is [n, k]
                beta,
                C, n // C is [m, n]
            );
        }
        else
        {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                m, n, k,
                alpha,
                A, k,             // A is [m, k]
                B_fp32.data(), n, // B is [k, n]
                beta,
                C, n // C is [m, n]
            );
        }

        return true;
    }

    bool BF16GemmKernel::multiply_activations(
        const float *A, const float *B, float *C,
        int m, int n, int k,
        bool transpose_B,
        float alpha, float beta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

        // Activation-activation GEMM with BF16 intermediate precision
        // Convert both A and B from FP32 to BF16, compute in BF16, output FP32
        // A: [m, k] (FP32)
        // B: [n, k] if transpose_B, else [k, n] (FP32)
        // C: [m, n] (FP32)

        // Convert A to BF16
        std::vector<uint16_t> A_bf16(m * k);
        fp32_to_bf16(A, A_bf16.data(), m * k);

        // Convert B to BF16
        int B_rows = transpose_B ? n : k;
        int B_cols = transpose_B ? k : n;
        size_t B_size = static_cast<size_t>(B_rows) * B_cols;
        std::vector<uint16_t> B_bf16(B_size);
        fp32_to_bf16(B, B_bf16.data(), B_size);

        // Try OneDNN first if available
#ifdef HAVE_ONEDNN
        using namespace dnnl;

        try
        {
            engine eng(engine::kind::cpu, 0);
            stream s(eng);

            // Define memory descriptors
            memory::dims a_dims = {m, k};
            memory::dims b_dims = transpose_B ? memory::dims{n, k} : memory::dims{k, n};
            memory::dims c_dims = {m, n};

            auto a_md = memory::desc(a_dims, memory::data_type::bf16, memory::format_tag::ab);
            auto b_md = transpose_B
                            ? memory::desc(b_dims, memory::data_type::bf16, memory::format_tag::ba)
                            : memory::desc(b_dims, memory::data_type::bf16, memory::format_tag::ab);
            auto c_md = memory::desc(c_dims, memory::data_type::f32, memory::format_tag::ab);

            // Create memory objects
            auto a_mem = memory(a_md, eng, A_bf16.data());
            auto b_mem = memory(b_md, eng, B_bf16.data());
            auto c_mem = memory(c_md, eng, C);

            // Create matmul primitive
            matmul::primitive_desc matmul_pd(eng, a_md, b_md, c_md);
            auto matmul_prim = matmul(matmul_pd);

            // Execute with scaling
            if (alpha != 1.0f || beta != 0.0f)
            {
                std::vector<float> C_temp(m * n);
                auto c_temp_mem = memory(c_md, eng, C_temp.data());

                matmul_prim.execute(s, {{DNNL_ARG_SRC, a_mem},
                                        {DNNL_ARG_WEIGHTS, b_mem},
                                        {DNNL_ARG_DST, c_temp_mem}});
                s.wait();

                for (int i = 0; i < m * n; ++i)
                {
                    C[i] = alpha * C_temp[i] + beta * C[i];
                }
            }
            else
            {
                matmul_prim.execute(s, {{DNNL_ARG_SRC, a_mem},
                                        {DNNL_ARG_WEIGHTS, b_mem},
                                        {DNNL_ARG_DST, c_mem}});
                s.wait();
            }

            return true; // OneDNN succeeded
        }
        catch (const dnnl::error &e)
        {
            // OneDNN failed, fall through to OpenBLAS
        }
#endif

        // OpenBLAS fallback: Expand to FP32 and use SGEMM
        std::vector<float> A_fp32(m * k);
        std::vector<float> B_fp32(B_size);

        bf16_to_fp32(A_bf16.data(), A_fp32.data(), m * k);
        bf16_to_fp32(B_bf16.data(), B_fp32.data(), B_size);

        if (transpose_B)
        {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasTrans,
                m, n, k,
                alpha,
                A_fp32.data(), k,
                B_fp32.data(), k,
                beta,
                C, n);
        }
        else
        {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                m, n, k,
                alpha,
                A_fp32.data(), k,
                B_fp32.data(), n,
                beta,
                C, n);
        }

        return true;
    }

    bool BF16GemmKernel::multiply_activations_strided(
        const float *A, const float *B, float *C,
        int m, int n, int k,
        int lda, int ldb, int ldc,
        bool transpose_B,
        float alpha, float beta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

        // Strided activation-activation GEMM with BF16 intermediate precision
        // Convert both A and B from FP32 to BF16, compute with strides

        // Note: We need to handle strided conversions
        // For simplicity, convert to contiguous BF16 buffers first
        // Future optimization: Support strided BF16 conversion

        std::vector<uint16_t> A_bf16(m * k);
        std::vector<uint16_t> B_bf16(transpose_B ? n * k : k * n);

        // Convert A with stride
        for (int i = 0; i < m; ++i)
        {
            fp32_to_bf16(A + i * lda, A_bf16.data() + i * k, k);
        }

        // Convert B with stride
        if (transpose_B)
        {
            for (int i = 0; i < n; ++i)
            {
                fp32_to_bf16(B + i * ldb, B_bf16.data() + i * k, k);
            }
        }
        else
        {
            for (int i = 0; i < k; ++i)
            {
                fp32_to_bf16(B + i * ldb, B_bf16.data() + i * n, n);
            }
        }

        // Try OneDNN first if available
#ifdef HAVE_ONEDNN
        using namespace dnnl;

        try
        {
            engine eng(engine::kind::cpu, 0);
            stream s(eng);

            // Define memory descriptors
            memory::dims a_dims = {m, k};
            memory::dims b_dims = transpose_B ? memory::dims{n, k} : memory::dims{k, n};
            memory::dims c_dims = {m, n};

            auto a_md = memory::desc(a_dims, memory::data_type::bf16, memory::format_tag::ab);
            auto b_md = transpose_B
                            ? memory::desc(b_dims, memory::data_type::bf16, memory::format_tag::ba)
                            : memory::desc(b_dims, memory::data_type::bf16, memory::format_tag::ab);
            auto c_md = memory::desc(c_dims, memory::data_type::f32, memory::format_tag::ab);

            // Create memory objects
            auto a_mem = memory(a_md, eng, A_bf16.data());
            auto b_mem = memory(b_md, eng, B_bf16.data());

            // Allocate temporary contiguous output buffer
            std::vector<float> C_temp(m * n);
            auto c_temp_mem = memory(c_md, eng, C_temp.data());

            // Create matmul primitive
            matmul::primitive_desc matmul_pd(eng, a_md, b_md, c_md);
            auto matmul_prim = matmul(matmul_pd);

            // Execute
            matmul_prim.execute(s, {{DNNL_ARG_SRC, a_mem},
                                    {DNNL_ARG_WEIGHTS, b_mem},
                                    {DNNL_ARG_DST, c_temp_mem}});
            s.wait();

            // Copy result to strided output with beta scaling
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    int dst_idx = i * ldc + j;
                    C[dst_idx] = alpha * C_temp[i * n + j] + beta * C[dst_idx];
                }
            }

            return true; // OneDNN succeeded
        }
        catch (const dnnl::error &e)
        {
            // OneDNN failed, fall through to OpenBLAS
        }
#endif

        // OpenBLAS fallback: Expand to FP32 and use strided SGEMM
        std::vector<float> A_fp32(m * k);
        std::vector<float> B_fp32(transpose_B ? n * k : k * n);

        bf16_to_fp32(A_bf16.data(), A_fp32.data(), m * k);
        bf16_to_fp32(B_bf16.data(), B_fp32.data(), transpose_B ? n * k : k * n);

        // Allocate temporary contiguous output
        std::vector<float> C_temp(m * n);

        if (transpose_B)
        {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasTrans,
                m, n, k,
                alpha,
                A_fp32.data(), k,
                B_fp32.data(), k,
                0.0f,
                C_temp.data(), n);
        }
        else
        {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                m, n, k,
                alpha,
                A_fp32.data(), k,
                B_fp32.data(), n,
                0.0f,
                C_temp.data(), n);
        }

        // Copy to strided output with beta scaling
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                int dst_idx = i * ldc + j;
                C[dst_idx] = C_temp[i * n + j] + beta * C[dst_idx];
            }
        }

        return true;
    }

} // namespace llaminar2
