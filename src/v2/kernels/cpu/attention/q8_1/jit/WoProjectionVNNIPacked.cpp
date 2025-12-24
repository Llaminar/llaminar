#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"

// C ABI thunk called from JIT code.
//
// wo_packed: pointer to gemm_v4::QuantisedPackedWeights (already packed for VNNI)
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
    using llaminar2::gemm_v4::QuantisedGemmKernel;
    using llaminar2::gemm_v4::QuantisedPackedWeights;

    const auto *packed = reinterpret_cast<const QuantisedPackedWeights *>(wo_packed);
    QuantisedGemmKernel kernel(packed);

    // We want: C[m,n] = A[m,k] @ W[n,k]
    // gemm_v4 packs weights in a pre-transposed internal layout; API still matches A@W.
    kernel.multiply(A, C, m, n, k, false, 1.0f, 0.0f, nullptr, -1);
}
