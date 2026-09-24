/**
 * @file KVCacheWorkspaceBuffers.h
 * @brief Shared names for graph-planned KV cache conversion buffers.
 * @author David Sanftenberg
 *
 * All GPU KV cache implementations publish converted K/V rows through these
 * two planner-owned buffers. Keeping the names in one backend-neutral header
 * prevents compressed and floating-point caches from silently developing
 * different workspace contracts.
 */

#pragma once

namespace llaminar2::KVCacheWorkspaceBuffers
{
    /// K conversion scratch used by append/read precision adaptation.
    inline constexpr const char *CONV_SCRATCH_K = "kvcache_conv_scratch_k";

    /// V conversion scratch used by append/read precision adaptation.
    inline constexpr const char *CONV_SCRATCH_V = "kvcache_conv_scratch_v";
} // namespace llaminar2::KVCacheWorkspaceBuffers
