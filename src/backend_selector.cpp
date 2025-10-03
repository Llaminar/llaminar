/**
 * @file backend_selector.cpp
 */
#include "backend_selector.h"
#include "utils/debug_env.h"
#include "logger.h"
#include <sstream>

namespace llaminar
{

    BackendDecision selectAttentionBackend(const BackendContext &ctx)
    {
        const auto &env = debugEnv();
        BackendDecision dec{BackendKind::OpenBLAS_MultiThread, "default-mt"};

        // If COSMA globally disabled (adaptive path) -> stay OpenBLAS
        if (env.adaptive.disable_cosma || env.cosma.disable || env.cosma.force_fallback)
        {
            dec.kind = BackendKind::OpenBLAS_MultiThread;
            dec.reason = "cosma-disabled";
            return dec;
        }
        // Only consider COSMA during prefill
        if (!ctx.is_prefill)
        {
            dec.kind = BackendKind::OpenBLAS_MultiThread;
            dec.reason = "decode-no-cosma";
            return dec;
        }
        // Sequence length threshold (env override)
        int thr = env.cosma.prefill_threshold;
        if (ctx.seq_len < thr)
        {
            // small ops: maybe single-thread if very small
            size_t elems = static_cast<size_t>(ctx.seq_len) * ctx.d_model;
            if (elems < 8192)
            {
                dec.kind = BackendKind::OpenBLAS_SingleThread;
                dec.reason = "small-single-thread";
            }
            else
            {
                dec.kind = BackendKind::OpenBLAS_MultiThread;
                dec.reason = "below-threshold";
            }
            return dec;
        }
        // Forced replicated or diagnostics may block COSMA path
        if (env.cosma.force_replicated || env.cosma.force_replicated_diag)
        {
            dec.kind = BackendKind::OpenBLAS_MultiThread;
            dec.reason = "force-replicated";
            return dec;
        }
        // If user explicitly forces COSMA
        if (env.cosma.force || env.cosma.force_direct)
        {
            dec.kind = BackendKind::COSMA_Prefill;
            dec.reason = env.cosma.force_direct ? "force-direct" : "force-cosma";
            return dec;
        }
        // Basic heuristic: large seq_len * d_model and world >1
        size_t volume = static_cast<size_t>(ctx.seq_len) * ctx.d_model;
        if (ctx.world > 1 && volume >= static_cast<size_t>(thr) * ctx.d_model)
        {
            dec.kind = BackendKind::COSMA_Prefill;
            std::ostringstream oss;
            oss << "auto-cosma(seq_len=" << ctx.seq_len << ",thr=" << thr << ")";
            dec.reason = oss.str();
            return dec;
        }
        // Fallback
        dec.kind = BackendKind::OpenBLAS_MultiThread;
        dec.reason = "heuristic-fallback";
        return dec;
    }

} // namespace llaminar
