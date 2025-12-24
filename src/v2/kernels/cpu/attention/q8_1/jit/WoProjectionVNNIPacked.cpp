#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "JitFusedAttentionWo.h" // For JitPackedWoParams
#include "utils/Logger.h"
#include <unordered_map>
#include <mutex>
#include <memory>

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
