// Prefill backend abstraction layer
// Lightweight interface to allow future GPU insertion (rocBLAS / cuBLAS) while
// current implementation remains CPU-only (OpenBLAS / COSMA path via adaptive_matmul).
// Author: David Sanftenberg

#pragma once
#include <string>
#include <memory>
#include <cstdint>
#include "device_kind.h"

namespace llaminar
{

    // High-level intent of operation (guides backend choice)
    enum class PrefillOpKind
    {
        MatMul,
        AttentionProj,
        MLPFused
    };

    // Logical device family (future extension)

    struct PrefillOpDesc
    {
        PrefillOpKind kind = PrefillOpKind::MatMul;
        int64_t M = 0, N = 0, K = 0; // GEMM dims (M x K) * (K x N) = (M x N)
        bool is_quantized = false;   // future: adjust epilog / dequant
        bool is_prefill = true;      // prefill vs decode nuance if reused
    };

    struct PrefillLaunchContext
    {
        const float *A = nullptr;
        const float *B = nullptr;
        float *C = nullptr;
        // Future: add bias pointer, activation enum, stream handle, etc.
    };

    // Result / status enumeration kept minimal; extend as GPU integration matures.
    enum class PrefillStatus
    {
        Success,
        Fallback,
        Unsupported,
        Error
    };

    // Strategy decision describing which concrete backend executed.
    struct PrefillBackendDecision
    {
        DeviceKind device = DeviceKind::CPU;
        std::string backend; // "openblas", "cosma", "rocblas", "cublas", "gpu_fused" (future)
        PrefillStatus status = PrefillStatus::Success;
        std::string reason; // human-readable selection rationale
    };

    class PrefillBackendInterface
    {
    public:
        virtual ~PrefillBackendInterface() = default;
        virtual PrefillBackendDecision launch(const PrefillOpDesc &desc,
                                              const PrefillLaunchContext &ctx) = 0;
    };

    // CPU implementation adapter (delegates to existing adaptive_matmul)
    class CpuPrefillBackend : public PrefillBackendInterface
    {
    public:
        PrefillBackendDecision launch(const PrefillOpDesc &desc,
                                      const PrefillLaunchContext &ctx) override;
    };

    // Stub GPU backend (not yet implemented). Returns Unsupported.
    class GpuPrefillBackendStub : public PrefillBackendInterface
    {
    public:
        PrefillBackendDecision launch(const PrefillOpDesc &desc,
                                      const PrefillLaunchContext &ctx) override;
    };

    // Factory (currently always CPU, optional GPU stub if env forces)
    class PrefillBackendFactory
    {
    public:
        static std::unique_ptr<PrefillBackendInterface> create();
    };

} // namespace llaminar
