#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "utils/Logger.h"
#include <unordered_map>
#include <mutex>
#include <memory>

namespace
{
    // Cache of QuantisedGemmKernel instances keyed by packed weights pointer.
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

    // Get cached kernel or create new one (avoids repeated construction overhead)
    QuantisedGemmKernel *kernel = getKernelCache().getOrCreate(packed);

    // We want: C[m,n] = A[m,k] @ W[n,k]
    // gemm_v4 packs weights in a pre-transposed internal layout; API still matches A@W.
    kernel->multiply(A, C, m, n, k, false, 1.0f, 0.0f, nullptr, -1);
}
