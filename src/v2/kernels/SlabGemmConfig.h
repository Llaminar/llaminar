/**
 * @file SlabGemmConfig.h
 * @brief Configuration for slab-based chunked GEMM execution
 *
 * This header defines the configuration structure for slab-based GEMM operations,
 * which convert INT8 data to FP16 in fixed-size "slabs" to bound memory usage.
 *
 * ============================================================================
 * MOTIVATION
 * ============================================================================
 *
 * For large GEMMs (e.g., 7B model FFN with K=3584, N=18944), the full FP16
 * conversion requires:
 *   - A_fp16: M × K × 2 bytes
 *   - B_fp16: K × N × 2 bytes  ← 136MB for K=3584, N=18944!
 *   - C_fp16: M × N × 2 bytes
 *
 * On memory-constrained GPUs (e.g., MI50 16GB shared with weights + KV cache),
 * we can instead convert and compute in "slabs":
 *   - slab_a_fp16: slab_m × slab_k × 2 bytes
 *   - slab_b_fp16: slab_k × slab_n × 2 bytes
 *   - slab_c_fp16: slab_m × slab_n × 2 bytes
 *
 * This trades compute (more kernel launches) for memory (fixed workspace).
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * For C[M×N] = A[M×K] × B[K×N]:
 *
 *   Zero-initialize output accumulator
 *   For k_start = 0 to K step slab_k:
 *     For n_start = 0 to N step slab_n:
 *       Convert B[k_start:k_end, n_start:n_end] → FP16 with scale
 *       For m_start = 0 to M step slab_m:
 *         Convert A[m_start:m_end, k_start:k_end] → FP16 with scale
 *         slab_c = hipblasHgemm(slab_a, slab_b)
 *         Accumulate slab_c → output[m_start:m_end, n_start:n_end]
 *
 * Total iterations = ceil(M/slab_m) × ceil(N/slab_n) × ceil(K/slab_k)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <algorithm>

namespace llaminar2
{

    // Forward declaration
    struct WorkspaceRequirements;

    /**
     * @brief Data type for slab buffers
     */
    enum class SlabDataType : uint8_t
    {
        FP16 = 0, ///< 2 bytes per element
        FP32 = 1, ///< 4 bytes per element
        BF16 = 2, ///< 2 bytes per element
    };

    /**
     * @brief Get bytes per element for a data type
     */
    inline size_t bytesPerElement(SlabDataType dtype)
    {
        switch (dtype)
        {
        case SlabDataType::FP16:
        case SlabDataType::BF16:
            return 2;
        case SlabDataType::FP32:
            return 4;
        default:
            return 2;
        }
    }

    /**
     * @brief Configuration for slab-based GEMM execution
     *
     * Specifies the dimensions of each "slab" (tile) used for chunked GEMM.
     * Larger slabs = fewer iterations but more memory.
     * Smaller slabs = more iterations but less memory.
     */
    struct SlabGemmConfig
    {
        // =========================================================================
        // Slab dimensions
        // =========================================================================

        int slab_m = 256; ///< Rows per slab (activation batches)
        int slab_n = 256; ///< Columns per slab (output features)
        int slab_k = 512; ///< Inner dimension per slab (reduction)

        // =========================================================================
        // Tuning parameters
        // =========================================================================

        bool use_streams = true;       ///< Use async streams for overlap
        bool prefetch_b_slabs = false; ///< Prefetch next B slab during compute
        int stream_count = 2;          ///< Number of streams for pipelining

        // =========================================================================
        // Memory calculations
        // =========================================================================

        /**
         * @brief Calculate bytes needed for slab A buffer
         * @param dtype Data type (FP16/FP32)
         * @return Bytes for slab_m × slab_k buffer
         */
        size_t slabABytes(SlabDataType dtype = SlabDataType::FP16) const
        {
            return static_cast<size_t>(slab_m) * slab_k * bytesPerElement(dtype);
        }

        /**
         * @brief Calculate bytes needed for slab B buffer
         * @param dtype Data type (FP16/FP32)
         * @return Bytes for slab_k × slab_n buffer
         */
        size_t slabBBytes(SlabDataType dtype = SlabDataType::FP16) const
        {
            return static_cast<size_t>(slab_k) * slab_n * bytesPerElement(dtype);
        }

        /**
         * @brief Calculate bytes needed for slab C buffer
         * @param dtype Data type (FP16/FP32)
         * @return Bytes for slab_m × slab_n buffer
         */
        size_t slabCBytes(SlabDataType dtype = SlabDataType::FP16) const
        {
            return static_cast<size_t>(slab_m) * slab_n * bytesPerElement(dtype);
        }

        /**
         * @brief Calculate total workspace bytes for all slab buffers
         *
         * If prefetch_b_slabs is enabled, allocates 2× B slab for double-buffering.
         *
         * @param dtype Data type (FP16/FP32)
         * @return Total workspace bytes needed
         */
        size_t totalWorkspaceBytes(SlabDataType dtype = SlabDataType::FP16) const
        {
            size_t a_bytes = slabABytes(dtype);
            size_t b_bytes = slabBBytes(dtype);
            size_t c_bytes = slabCBytes(dtype);

            // Double-buffer B if prefetching enabled
            if (prefetch_b_slabs)
            {
                b_bytes *= 2;
            }

            // Add alignment padding (256-byte alignment)
            constexpr size_t ALIGN = 256;
            a_bytes = (a_bytes + ALIGN - 1) & ~(ALIGN - 1);
            b_bytes = (b_bytes + ALIGN - 1) & ~(ALIGN - 1);
            c_bytes = (c_bytes + ALIGN - 1) & ~(ALIGN - 1);

            return a_bytes + b_bytes + c_bytes;
        }

        /**
         * @brief Get workspace requirements for GPU workspace manager
         *
         * Returns WorkspaceRequirements with named buffers:
         *   - "slab_a_fp16": slab_m × slab_k × 2 bytes
         *   - "slab_b_fp16": slab_k × slab_n × 2 bytes (×2 if prefetch)
         *   - "slab_c_fp16": slab_m × slab_n × 2 bytes
         *
         * @param dtype Data type (FP16/FP32)
         * @return Workspace requirements
         */
        WorkspaceRequirements workspaceRequirements(SlabDataType dtype = SlabDataType::FP16) const;

        // =========================================================================
        // Iteration estimation
        // =========================================================================

        /**
         * @brief Estimate number of slab GEMM iterations
         *
         * Total iterations = ceil(M/slab_m) × ceil(N/slab_n) × ceil(K/slab_k)
         *
         * @param m Full GEMM M dimension
         * @param n Full GEMM N dimension
         * @param k Full GEMM K dimension
         * @return Number of slab GEMM calls
         */
        int estimateIterations(int m, int n, int k) const
        {
            if (slab_m <= 0 || slab_n <= 0 || slab_k <= 0)
                return 0;

            int m_iters = (m + slab_m - 1) / slab_m;
            int n_iters = (n + slab_n - 1) / slab_n;
            int k_iters = (k + slab_k - 1) / slab_k;

            return m_iters * n_iters * k_iters;
        }

        /**
         * @brief Estimate number of B slab conversions
         *
         * B conversion happens once per (k_iter, n_iter) pair.
         * If M is large, we reuse the same B slab for multiple A slabs.
         *
         * @param n Full GEMM N dimension
         * @param k Full GEMM K dimension
         * @return Number of B slab conversions
         */
        int estimateBConversions(int n, int k) const
        {
            if (slab_n <= 0 || slab_k <= 0)
                return 0;

            int n_iters = (n + slab_n - 1) / slab_n;
            int k_iters = (k + slab_k - 1) / slab_k;

            return n_iters * k_iters;
        }

        /**
         * @brief Estimate number of A slab conversions
         *
         * A conversion happens for each (k_iter, n_iter, m_iter) triple.
         *
         * @param m Full GEMM M dimension
         * @param n Full GEMM N dimension
         * @param k Full GEMM K dimension
         * @return Number of A slab conversions
         */
        int estimateAConversions(int m, int n, int k) const
        {
            // A is converted for each (m_iter, k_iter) pair, and we loop over all n_iters
            // But we can optimize: A conversion doesn't depend on n, so we only need
            // m_iters × k_iters conversions if we're smart about loop order.
            // Current algorithm converts A inside the n loop, so:
            return estimateIterations(m, n, k);
        }

        // =========================================================================
        // Factory methods
        // =========================================================================

        /**
         * @brief Create config from memory budget
         *
         * Calculates largest slab dimensions that fit within budget.
         * Prioritizes larger slab_n (output dimension) for better reuse.
         *
         * Strategy:
         *   1. Start with balanced slabs (m=n=k roughly equal)
         *   2. Scale slab_n up first (B matrix reused across M iterations)
         *   3. Scale slab_k to balance memory vs iterations
         *   4. Set slab_m based on typical M values (prefill vs decode)
         *
         * @param budget_bytes Available workspace memory
         * @param m Full GEMM M dimension (hint for optimization)
         * @param n Full GEMM N dimension
         * @param k Full GEMM K dimension
         * @param dtype Data type (FP16/FP32)
         * @return Optimized slab configuration
         */
        static SlabGemmConfig fromBudget(
            size_t budget_bytes,
            int m, int n, int k,
            SlabDataType dtype = SlabDataType::FP16);

        /**
         * @brief Create default config for decode (M=1)
         *
         * For decode, M is small (1-8), so we optimize for:
         *   - Large slab_n (process all output features)
         *   - Large slab_k (minimize inner loop iterations)
         *   - Small slab_m (just enough for the batch)
         *
         * @param n Full GEMM N dimension
         * @param k Full GEMM K dimension
         * @param budget_bytes Memory budget (0 = default 64MB)
         * @return Config optimized for decode
         */
        static SlabGemmConfig forDecode(int n, int k, size_t budget_bytes = 0);

        /**
         * @brief Create default config for prefill (M=128-2048)
         *
         * For prefill, M is larger, so we optimize for:
         *   - Balanced slab_m and slab_n
         *   - Moderate slab_k for good parallelism
         *
         * @param m Full GEMM M dimension
         * @param n Full GEMM N dimension
         * @param k Full GEMM K dimension
         * @param budget_bytes Memory budget (0 = default 64MB)
         * @return Config optimized for prefill
         */
        static SlabGemmConfig forPrefill(int m, int n, int k, size_t budget_bytes = 0);

        /**
         * @brief Check if slab dimensions are valid
         *
         * @return true if all dimensions are positive
         */
        bool isValid() const
        {
            return slab_m > 0 && slab_n > 0 && slab_k > 0;
        }

        /**
         * @brief Check if slabs cover entire GEMM without chunking
         *
         * @param m Full GEMM M dimension
         * @param n Full GEMM N dimension
         * @param k Full GEMM K dimension
         * @return true if M ≤ slab_m && N ≤ slab_n && K ≤ slab_k
         */
        bool coversEntireGemm(int m, int n, int k) const
        {
            return m <= slab_m && n <= slab_n && k <= slab_k;
        }

        /**
         * @brief Get actual slab dimensions for a specific position
         *
         * At the edges of the matrix, slabs may be smaller than configured.
         *
         * @param m_start Row offset in full matrix
         * @param n_start Column offset in full matrix
         * @param k_start Inner dimension offset
         * @param full_m Full M dimension
         * @param full_n Full N dimension
         * @param full_k Full K dimension
         * @param out_m Actual slab M (output)
         * @param out_n Actual slab N (output)
         * @param out_k Actual slab K (output)
         */
        void actualSlabDims(
            int m_start, int n_start, int k_start,
            int full_m, int full_n, int full_k,
            int &out_m, int &out_n, int &out_k) const
        {
            out_m = std::min(slab_m, full_m - m_start);
            out_n = std::min(slab_n, full_n - n_start);
            out_k = std::min(slab_k, full_k - k_start);
        }

        // =========================================================================
        // Debug / Logging
        // =========================================================================

        /**
         * @brief Get human-readable description
         */
        const char *describe() const;
    };

} // namespace llaminar2
