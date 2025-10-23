// Inference (decode / small batch) backend abstraction layer
// Author: David Sanftenberg

#pragma once
#include <string>
#include <memory>
#include <cstdint>
#include "DeviceKind.h"

namespace llaminar
{

    enum class InferenceOpKind
    {
        MatMul,
        Embedding,
        LayerNorm
    };

    struct InferenceOpDesc
    {
        InferenceOpKind kind = InferenceOpKind::MatMul;
        int64_t M = 0, N = 0, K = 0;  // for matmul
        bool latency_critical = true; // decode path vs bulk
        bool transpose_A = false;     // transpose first operand
        bool transpose_B = false;     // transpose second operand (weight)
    };

    struct InferenceLaunchContext
    {
        const float *A = nullptr;
        const float *B = nullptr;
        float *C = nullptr;
    };

    enum class InferenceStatus
    {
        Success,
        Fallback,
        Unsupported,
        Error
    };

    struct InferenceBackendDecision
    {
        DeviceKind device = DeviceKind::CPU;
        std::string backend; // openblas | cosma | rocblas | cublas | gpu_fused
        InferenceStatus status = InferenceStatus::Success;
        std::string reason;
    };

    class InferenceBackendInterface
    {
    public:
        virtual ~InferenceBackendInterface() = default;
        virtual InferenceBackendDecision launch(const InferenceOpDesc &desc,
                                                const InferenceLaunchContext &ctx) = 0;
    };

    class CpuInferenceBackend : public InferenceBackendInterface
    {
    public:
        InferenceBackendDecision launch(const InferenceOpDesc &desc,
                                        const InferenceLaunchContext &ctx) override;
    };

    class GpuInferenceBackendStub : public InferenceBackendInterface
    {
    public:
        InferenceBackendDecision launch(const InferenceOpDesc &desc,
                                        const InferenceLaunchContext &ctx) override;
    };

    class InferenceBackendFactory
    {
    public:
        static std::unique_ptr<InferenceBackendInterface> create();
    };

} // namespace llaminar
