// Implementation of prefill backend abstraction (CPU-only functional now)
// Author: David Sanftenberg

#include "PrefillBackend.h"
#include "../AdaptiveMatmul.h"
#include "../Logger.h"
#include "../utils/DebugEnv.h"

namespace llaminar
{

    PrefillBackendDecision CpuPrefillBackend::launch(const PrefillOpDesc &d,
                                                     const PrefillLaunchContext &ctx)
    {
        PrefillBackendDecision decision;
        decision.device = DeviceKind::CPU;
        decision.backend = "openblas"; // may be overwritten if COSMA selected
        decision.status = PrefillStatus::Success;

        if (!ctx.A || !ctx.B || !ctx.C)
        {
            decision.status = PrefillStatus::Error;
            decision.reason = "null pointer in launch context";
            return decision;
        }

        // Reuse existing adaptive manager (stack instance is cheap)
        AdaptiveMatMulManager mgr;
        bool is_prefill = d.is_prefill;
        auto backend = mgr.selectBackend((int)d.M, (int)d.N, (int)d.K, is_prefill);
        decision.backend = (backend == MatMulBackend::COSMA) ? "cosma" : "openblas";
        decision.reason = is_prefill ? "prefill path" : "generic path";

        if (!adaptiveMatMul(ctx.A, ctx.B, ctx.C, (int)d.M, (int)d.N, (int)d.K, is_prefill, false, d.transpose_A, d.transpose_B))
        {
            decision.status = PrefillStatus::Error;
            decision.reason = "adaptive_matmul failure";
        }
        return decision;
    }

    PrefillBackendDecision GpuPrefillBackendStub::launch(const PrefillOpDesc &d,
                                                         const PrefillLaunchContext &ctx)
    {
        (void)d;
        (void)ctx;
        PrefillBackendDecision decision;
        decision.device = DeviceKind::GPU;
        decision.backend = "gpu_stub";
        decision.status = PrefillStatus::Unsupported;
        decision.reason = "GPU backend not compiled (stub)";
        return decision;
    }

    std::unique_ptr<PrefillBackendInterface> PrefillBackendFactory::create()
    {
        // Simple heuristic: if user explicitly forces GPU distribution mode but GPU not available, still return stub.
        const auto &env = debugEnv();
        bool pretend_gpu = (env.distribution.distribution_mode == "sharded" && env.distribution.force_sharded);
        if (pretend_gpu)
        {
            return std::unique_ptr<PrefillBackendInterface>(new GpuPrefillBackendStub());
        }
        return std::unique_ptr<PrefillBackendInterface>(new CpuPrefillBackend());
    }

} // namespace llaminar
