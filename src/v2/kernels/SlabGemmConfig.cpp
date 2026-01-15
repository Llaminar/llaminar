/**
 * @file SlabGemmConfig.cpp
 * @brief Implementation of slab-based GEMM configuration
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "SlabGemmConfig.h"
#include "execution/WorkspaceDescriptor.h"
#include "utils/Logger.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace llaminar2
{

    // ============================================================================
    // Memory budget constants
    // ============================================================================

    // Default workspace budget if not specified (64MB)
    constexpr size_t DEFAULT_WORKSPACE_BUDGET = 64 * 1024 * 1024;

    // Minimum slab dimensions for efficient GPU execution
    constexpr int MIN_SLAB_M = 32;
    constexpr int MIN_SLAB_N = 64; // hipBLAS hgemm prefers N≥64
    constexpr int MIN_SLAB_K = 32;

    // Maximum slab dimensions (beyond this, full GEMM is better)
    constexpr int MAX_SLAB_M = 1024;
    constexpr int MAX_SLAB_N = 8192;
    constexpr int MAX_SLAB_K = 2048;

    // Alignment for GPU memory (256 bytes for AMD GPUs)
    constexpr size_t GPU_ALIGN = 256;

    // ============================================================================
    // WorkspaceRequirements factory
    // ============================================================================

    WorkspaceRequirements SlabGemmConfig::workspaceRequirements(SlabDataType dtype) const
    {
        WorkspaceRequirements reqs;

        const char *suffix = (dtype == SlabDataType::FP16) ? "_fp16" : (dtype == SlabDataType::FP32) ? "_fp32"
                                                                                                     : "_bf16";

        // Slab A buffer: [slab_m × slab_k]
        {
            std::string name = std::string("slab_a") + suffix;
            size_t size = slabABytes(dtype);
            reqs.buffers.emplace_back(name, size, GPU_ALIGN, true);
        }

        // Slab B buffer: [slab_k × slab_n] (or 2× for double-buffering)
        {
            std::string name = std::string("slab_b") + suffix;
            size_t size = slabBBytes(dtype);
            if (prefetch_b_slabs)
            {
                size *= 2; // Double-buffer for prefetch
            }
            reqs.buffers.emplace_back(name, size, GPU_ALIGN, true);
        }

        // Slab C buffer: [slab_m × slab_n]
        {
            std::string name = std::string("slab_c") + suffix;
            size_t size = slabCBytes(dtype);
            reqs.buffers.emplace_back(name, size, GPU_ALIGN, true);
        }

        return reqs;
    }

    // ============================================================================
    // Budget-based configuration
    // ============================================================================

    SlabGemmConfig SlabGemmConfig::fromBudget(
        size_t budget_bytes,
        int m, int n, int k,
        SlabDataType dtype)
    {
        SlabGemmConfig config;

        if (budget_bytes == 0)
        {
            budget_bytes = DEFAULT_WORKSPACE_BUDGET;
        }

        const size_t elem_size = bytesPerElement(dtype);

        // =========================================================================
        // Strategy: Maximize slab_n first (B matrix reused across M iterations)
        //
        // Memory: slab_m * slab_k * elem_size  (A)
        //       + slab_k * slab_n * elem_size  (B)
        //       + slab_m * slab_n * elem_size  (C)
        //
        // For decode (small M), we want large slab_n and slab_k.
        // For prefill (large M), we balance slab_m with slab_n.
        // =========================================================================

        // Start with reasonable defaults
        int slab_m = std::min(256, m);
        int slab_n = std::min(1024, n);
        int slab_k = std::min(512, k);

        // Apply minimums
        slab_m = std::max(slab_m, MIN_SLAB_M);
        slab_n = std::max(slab_n, MIN_SLAB_N);
        slab_k = std::max(slab_k, MIN_SLAB_K);

        // Calculate memory for current config
        auto calc_memory = [elem_size](int sm, int sn, int sk) -> size_t
        {
            size_t a_mem = static_cast<size_t>(sm) * sk * elem_size;
            size_t b_mem = static_cast<size_t>(sk) * sn * elem_size;
            size_t c_mem = static_cast<size_t>(sm) * sn * elem_size;

            // Add alignment overhead
            a_mem = (a_mem + GPU_ALIGN - 1) & ~(GPU_ALIGN - 1);
            b_mem = (b_mem + GPU_ALIGN - 1) & ~(GPU_ALIGN - 1);
            c_mem = (c_mem + GPU_ALIGN - 1) & ~(GPU_ALIGN - 1);

            return a_mem + b_mem + c_mem;
        };

        // =========================================================================
        // Phase 1: Try to fit entire N dimension (maximize B reuse)
        // =========================================================================

        if (static_cast<size_t>(n) <= MAX_SLAB_N)
        {
            int test_n = n;
            size_t mem = calc_memory(slab_m, test_n, slab_k);
            if (mem <= budget_bytes)
            {
                slab_n = test_n;
            }
        }

        // If that doesn't fit, try progressively smaller slab_n
        while (calc_memory(slab_m, slab_n, slab_k) > budget_bytes && slab_n > MIN_SLAB_N)
        {
            slab_n = slab_n * 3 / 4; // Reduce by 25%
            slab_n = std::max(slab_n, MIN_SLAB_N);
        }

        // =========================================================================
        // Phase 2: Optimize slab_k for remaining budget
        // =========================================================================

        // Try to increase slab_k if we have budget headroom
        while (slab_k < k && slab_k < MAX_SLAB_K)
        {
            int test_k = std::min(slab_k * 2, std::min(k, MAX_SLAB_K));
            if (calc_memory(slab_m, slab_n, test_k) <= budget_bytes)
            {
                slab_k = test_k;
            }
            else
            {
                break;
            }
        }

        // =========================================================================
        // Phase 3: Adjust slab_m based on actual M and remaining budget
        // =========================================================================

        // For small M (decode), don't waste memory on large slab_m
        if (m <= 32)
        {
            slab_m = std::max(MIN_SLAB_M, m);
        }
        else
        {
            // Try to increase slab_m if we have budget
            while (slab_m < m && slab_m < MAX_SLAB_M)
            {
                int test_m = std::min(slab_m * 2, std::min(m, MAX_SLAB_M));
                if (calc_memory(test_m, slab_n, slab_k) <= budget_bytes)
                {
                    slab_m = test_m;
                }
                else
                {
                    break;
                }
            }
        }

        // =========================================================================
        // Final: Reduce dimensions if still over budget
        // =========================================================================

        while (calc_memory(slab_m, slab_n, slab_k) > budget_bytes)
        {
            // Find which dimension to reduce (prefer reducing M over N for B reuse)
            if (slab_m > MIN_SLAB_M && slab_m >= slab_n)
            {
                slab_m = slab_m * 3 / 4;
                slab_m = std::max(slab_m, MIN_SLAB_M);
            }
            else if (slab_k > MIN_SLAB_K)
            {
                slab_k = slab_k * 3 / 4;
                slab_k = std::max(slab_k, MIN_SLAB_K);
            }
            else if (slab_n > MIN_SLAB_N)
            {
                slab_n = slab_n * 3 / 4;
                slab_n = std::max(slab_n, MIN_SLAB_N);
            }
            else
            {
                // Can't reduce further, budget is too small
                LOG_WARN("[SlabGemmConfig] Budget too small for minimum slab config");
                break;
            }
        }

        config.slab_m = slab_m;
        config.slab_n = slab_n;
        config.slab_k = slab_k;

        LOG_DEBUG("[SlabGemmConfig] fromBudget(" << (budget_bytes / (1024 * 1024)) << "MB): "
                                                 << "slab_m=" << slab_m << ", slab_n=" << slab_n << ", slab_k=" << slab_k
                                                 << " (total=" << (calc_memory(slab_m, slab_n, slab_k) / (1024 * 1024)) << "MB)");

        return config;
    }

    SlabGemmConfig SlabGemmConfig::forDecode(int n, int k, size_t budget_bytes)
    {
        if (budget_bytes == 0)
        {
            budget_bytes = DEFAULT_WORKSPACE_BUDGET;
        }

        // For decode, M is typically 1-8
        // Optimize for maximum N and K coverage
        SlabGemmConfig config = fromBudget(budget_bytes, 8, n, k, SlabDataType::FP16);

        // Force small slab_m for decode
        config.slab_m = std::min(config.slab_m, 32);

        return config;
    }

    SlabGemmConfig SlabGemmConfig::forPrefill(int m, int n, int k, size_t budget_bytes)
    {
        if (budget_bytes == 0)
        {
            budget_bytes = DEFAULT_WORKSPACE_BUDGET;
        }

        // For prefill, M is typically 128-2048
        // Balance slab_m with slab_n for good parallelism
        return fromBudget(budget_bytes, m, n, k, SlabDataType::FP16);
    }

    // ============================================================================
    // Debug / Logging
    // ============================================================================

    // Thread-local buffer for describe()
    static thread_local char g_describe_buffer[256];

    const char *SlabGemmConfig::describe() const
    {
        std::snprintf(g_describe_buffer, sizeof(g_describe_buffer),
                      "SlabGemmConfig{m=%d, n=%d, k=%d, workspace=%.2fMB}",
                      slab_m, slab_n, slab_k,
                      static_cast<double>(totalWorkspaceBytes()) / (1024 * 1024));
        return g_describe_buffer;
    }

} // namespace llaminar2
