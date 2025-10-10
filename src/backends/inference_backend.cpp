// Implementation of inference backend abstraction (CPU only now)
// Author: David Sanftenberg

#include "inference_backend.h"
#include "prefill_backend.h" // reuse DeviceKind enum include
#include "../adaptive_matmul.h"
#include "../logger.h"

namespace llaminar
{

    InferenceBackendDecision CpuInferenceBackend::launch(const InferenceOpDesc &d,
                                                         const InferenceLaunchContext &ctx)
    {
        InferenceBackendDecision decision;
        decision.device = DeviceKind::CPU;
        decision.backend = "openblas";
        if (d.kind != InferenceOpKind::MatMul)
        {
            decision.status = InferenceStatus::Unsupported;
            decision.reason = "non-matmul op not yet wired";
            return decision;
        }
        if (!ctx.A || !ctx.B || !ctx.C)
        {
            decision.status = InferenceStatus::Error;
            decision.reason = "null pointer";
            return decision;
        }
        AdaptiveMatMulManager mgr;
        auto backend = mgr.selectBackend((int)d.M, (int)d.N, (int)d.K, /*is_prefill=*/false);
        decision.backend = (backend == MatMulBackend::COSMA) ? "cosma" : "openblas";
        decision.reason = d.latency_critical ? "latency-critical decode" : "generic decode";
        if (!adaptiveMatMul(ctx.A, ctx.B, ctx.C, (int)d.M, (int)d.N, (int)d.K, false, false, d.transpose_A, d.transpose_B))
        {
            decision.status = InferenceStatus::Error;
            decision.reason = "adaptive_matmul failure";
        }
        return decision;
    }

    InferenceBackendDecision GpuInferenceBackendStub::launch(const InferenceOpDesc &d,
                                                             const InferenceLaunchContext &ctx)
    {
        (void)d;
        (void)ctx;
        InferenceBackendDecision decision;
        decision.device = DeviceKind::GPU;
        decision.backend = "gpu_stub";
        decision.status = InferenceStatus::Unsupported;
        decision.reason = "GPU backend not compiled (stub)";
        return decision;
    }

    std::unique_ptr<InferenceBackendInterface> InferenceBackendFactory::create()
    {
        return std::unique_ptr<InferenceBackendInterface>(new CpuInferenceBackend());
    }

} // namespace llaminar
