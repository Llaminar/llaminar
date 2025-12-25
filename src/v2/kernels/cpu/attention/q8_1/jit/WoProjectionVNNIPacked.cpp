#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "JitFusedAttentionWo.h"       // For JitPackedWoParams
#include "tensors/QuantizationUtils.h" // For dequantize_vnni_packed_row_to_fp32
#include "utils/Logger.h"
#include "utils/OpenMPUtils.h"
#include <unordered_map>
#include <mutex>
#include <memory>
#include <vector>
#include <immintrin.h>

namespace
{
    // Cache of QuantisedGemmKernel instances keyed by original_packed pointer.
    // Since packed weights are already cached in TensorCache and have stable addresses,
    // we can use the pointer as a key to avoid recreating kernels on every call.
    struct KernelCache
    {
        std::mutex mutex;
        std::unordered_map<const void *, std::unique_ptr<llaminar2::gemm_v4::QuantisedGemmKernel>> kernels;

        llaminar2::gemm_v4::QuantisedGemmKernel *getOrCreate(
            const llaminar2::gemm_v4::QuantisedPackedWeights *packed)
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = kernels.find(packed);
            if (it != kernels.end())
            {
                return it->second.get();
            }
            auto kernel = std::make_unique<llaminar2::gemm_v4::QuantisedGemmKernel>(packed);
            auto *ptr = kernel.get();
            kernels[packed] = std::move(kernel);
            return ptr;
        }
    };

    KernelCache &getKernelCache()
    {
        static KernelCache cache;
        return cache;
    }
}

// C ABI thunk called from JIT code.
//
// wo_packed: pointer to JitPackedWoParams (contains raw pointers + original QuantisedPackedWeights*)
// A:         FP32 activations [m, k]
// C:         FP32 output      [m, n]
// m/n/k:     GEMM sizes
extern "C" void llaminar2_wo_q8_1_vnni_packed_gemm(
    const void *wo_packed,
    const float *A,
    float *C,
    int m,
    int n,
    int k)
{
    using llaminar::v2::kernels::jit::JitPackedWoParams;
    using llaminar2::gemm_v4::QuantisedGemmKernel;
    using llaminar2::gemm_v4::QuantisedPackedWeights;

    // Debug: dump raw bytes at the struct location
    const uint8_t *raw = reinterpret_cast<const uint8_t *>(wo_packed);
    LOG_DEBUG("WoProjectionVNNIPacked raw bytes: "
              << std::hex << (int)raw[0] << " " << (int)raw[1] << " " << (int)raw[2] << " " << (int)raw[3] << " ... "
              << "at offset 48: " << (int)raw[48] << " " << (int)raw[49] << " " << (int)raw[50] << " " << (int)raw[51]
              << std::dec);

    // wo_packed is actually a JitPackedWoParams*, which contains original_packed
    // pointing to the real QuantisedPackedWeights struct
    const auto *params = reinterpret_cast<const JitPackedWoParams *>(wo_packed);

    // Debug: check that original_packed is valid
    if (!params->original_packed)
    {
        LOG_ERROR("WoProjectionVNNIPacked: original_packed is null! params="
                  << wo_packed << " N=" << params->N << " K=" << params->K);
        return;
    }

    const auto *packed = reinterpret_cast<const QuantisedPackedWeights *>(params->original_packed);

    // Debug: log dimensions
    LOG_DEBUG("WoProjectionVNNIPacked: params->original_packed=" << params->original_packed
                                                                 << " params->N=" << params->N << " params->K=" << params->K
                                                                 << " packed->N=" << packed->N << " packed->K=" << packed->K);

    // Get cached kernel or create new one (avoids repeated construction overhead)
    QuantisedGemmKernel *kernel = getKernelCache().getOrCreate(packed);

    // We want: C[m,n] = A[m,k] @ W[n,k]
    // gemm_v4 packs weights in a pre-transposed internal layout; API still matches A@W.
    kernel->multiply(A, C, m, n, k, false, 1.0f, 0.0f, nullptr, -1);
}

// ============================================================================
// FP32 STREAMING DEQUANTIZATION GEMM (Hybrid mode)
// ============================================================================
//
// This function provides highest numerical precision by:
// 1. Dequantizing VNNI-packed weights to FP32 row-by-row (streaming)
// 2. Computing FP32×FP32 → FP32 dot products
//
// This avoids the precision loss from quantizing FP32 activations to Q8_1.
//
// For decode (m=1), the flow is:
//   for each output row n:
//     dequant_row[k] = dequantize(packed_weights[n, :])
//     output[n] = dot(context[k], dequant_row[k])
//
// Memory: Uses a temporary buffer of size K floats (aligned to 64 bytes)
// Performance: Slower than VNNI path due to FP32 computation, but ~same
//              throughput as pure FP32 weights path
// ============================================================================
extern "C" void llaminar2_wo_fp32_streaming_dequant_gemm(
    const void *wo_packed,
    const float *A, // FP32 context [m, k]
    float *C,       // FP32 output [m, n]
    int m,
    int n,
    int k)
{
    using llaminar::v2::kernels::jit::JitPackedWoParams;
    using llaminar2::gemm_v4::QuantisedPackedWeights;

    // Extract packed weight pointers from JitPackedWoParams
    const auto *params = reinterpret_cast<const JitPackedWoParams *>(wo_packed);

    if (!params->original_packed)
    {
        LOG_ERROR("FP32StreamingDequantGEMM: original_packed is null!");
        return;
    }

    const auto *packed = reinterpret_cast<const QuantisedPackedWeights *>(params->original_packed);
    const int8_t *packed_data = packed->packed_data.data();
    const float *scales = packed->scales.data();
    const float *mins = packed->has_mins ? packed->mins.data() : nullptr;
    const int K = packed->K;
    const int N = packed->N;

    LOG_DEBUG("FP32StreamingDequantGEMM: m=" << m << " n=" << n << " k=" << k
                                             << " N=" << N << " K=" << K << " has_mins=" << packed->has_mins);

    // Validate dimensions
    if (k != K || n != N)
    {
        LOG_ERROR("FP32StreamingDequantGEMM: dimension mismatch! k=" << k << " K=" << K
                                                                     << " n=" << n << " N=" << N);
        return;
    }

    // Allocate aligned temporary buffer for dequantized row
    // Size: K floats, aligned to 64 bytes for AVX-512
    const size_t dequant_buf_size = llaminar2::quantization::dequantized_row_buffer_size(K);

    // For m=1 (typical decode), process sequentially
    // For m>1 (prefill), we could parallelize over output rows

    if (m == 1)
    {
        // Single row - allocate one temp buffer
        alignas(64) std::vector<float> dequant_row((K + 15) / 16 * 16); // Padded for SIMD

        // Process each output column (Wo row)
        for (int out_idx = 0; out_idx < N; ++out_idx)
        {
            // Dequantize this Wo row to FP32
            llaminar2::quantization::dequantize_vnni_packed_row_to_fp32(
                packed_data, scales, mins,
                dequant_row.data(),
                out_idx, K, N);

            // Compute dot product: C[out_idx] = A[0:K] · dequant_row[0:K]
            float sum = 0.0f;

#ifdef __AVX512F__
            // AVX-512 vectorized dot product
            __m512 acc = _mm512_setzero_ps();
            int i = 0;
            for (; i + 15 < K; i += 16)
            {
                __m512 a_vec = _mm512_loadu_ps(A + i);
                __m512 b_vec = _mm512_loadu_ps(dequant_row.data() + i);
                acc = _mm512_fmadd_ps(a_vec, b_vec, acc);
            }
            sum = _mm512_reduce_add_ps(acc);
            // Handle remainder
            for (; i < K; ++i)
            {
                sum += A[i] * dequant_row[i];
            }
#elif defined(__AVX2__)
            // AVX2 vectorized dot product
            __m256 acc = _mm256_setzero_ps();
            int i = 0;
            for (; i + 7 < K; i += 8)
            {
                __m256 a_vec = _mm256_loadu_ps(A + i);
                __m256 b_vec = _mm256_loadu_ps(dequant_row.data() + i);
                acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
            }
            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 sum128 = _mm_add_ps(hi, lo);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum = _mm_cvtss_f32(sum128);
            // Handle remainder
            for (; i < K; ++i)
            {
                sum += A[i] * dequant_row[i];
            }
#else
            // Scalar fallback
            for (int i = 0; i < K; ++i)
            {
                sum += A[i] * dequant_row[i];
            }
#endif

            C[out_idx] = sum;
        }
    }
    else
    {
        // Multi-row case (prefill) - parallelize over output rows
        auto do_prefill_work = [&]()
        {
            // Thread-local dequant buffer
            alignas(64) std::vector<float> dequant_row((K + 15) / 16 * 16);

#pragma omp for schedule(static)
            for (int row = 0; row < m; ++row)
            {
                const float *A_row = A + row * K;
                float *C_row = C + row * N;

                for (int out_idx = 0; out_idx < N; ++out_idx)
                {
                    // Dequantize Wo row
                    llaminar2::quantization::dequantize_vnni_packed_row_to_fp32(
                        packed_data, scales, mins,
                        dequant_row.data(),
                        out_idx, K, N);

                    // Compute dot product
                    float sum = 0.0f;
#ifdef __AVX512F__
                    __m512 acc = _mm512_setzero_ps();
                    int i = 0;
                    for (; i + 15 < K; i += 16)
                    {
                        __m512 a_vec = _mm512_loadu_ps(A_row + i);
                        __m512 b_vec = _mm512_loadu_ps(dequant_row.data() + i);
                        acc = _mm512_fmadd_ps(a_vec, b_vec, acc);
                    }
                    sum = _mm512_reduce_add_ps(acc);
                    for (; i < K; ++i)
                    {
                        sum += A_row[i] * dequant_row[i];
                    }
#else
                    for (int i = 0; i < K; ++i)
                    {
                        sum += A_row[i] * dequant_row[i];
                    }
#endif
                    C_row[out_idx] = sum;
                }
            }
        };

        OMP_WORKSHARE_REGION(do_prefill_work);
    }

    LOG_DEBUG("FP32StreamingDequantGEMM: completed");
}
