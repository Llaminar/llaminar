/**
 * @file tp_generic_matmul_executor.h
 * @brief Generic tensor-parallel matmul executor abstraction (row or column split) with reconstruction helpers.
 *
 * This generalizes the specialized TPOutputProjectionExecutor so FFN / projection kernels can share
 * a common interface. It wraps existing MatmulSplitter variants (row/col) and provides:
 *  - Config-driven splitter selection (row vs col)
 *  - Local execution producing a slice buffer and partition specs
 *  - Static reconstruction (concatenate rows or columns) for parity tests / host aggregation
 *
 * Design Notes:
 *  - Keeps packing inside splitter for column splits (minimizes duplication)
 *  - Row split assumes caller advances A pointer; we expose specA so caller can adjust
 *  - Future: can extend with batched/fused GEMM or strided backend support
 */
#pragma once

#include "tp_partition.h"
#include "../logger.h" // for LOG_DEBUG
#include <chrono>
#include <functional>
#include <vector>
#include <stdexcept>
#include <cstring>

namespace llaminar
{

    struct TPGemmExecConfig
    {
        enum class Mode
        {
            Column,
            Row
        } mode = Mode::Column; // Which output axis is partitioned
        int tp_size = 1;
        int tp_rank = 0;
    };

    struct TPGemmLocalResult
    {
        TPPartitionSpec partA;     // Row partition (if active)
        TPPartitionSpec partB;     // Column partition (if active)
        std::vector<float> buffer; // Local C slice (shape: M_local x N_local)
        std::size_t M_local = 0;   // Rows in local output
        std::size_t N_local = 0;   // Cols in local output
    };

    /**
     * @brief Generic TP matmul executor: C = A * B (A:[M,K], B:[K,N]).
     * Column mode partitions N, Row mode partitions M.
     */
    class TPGemmExecutor
    {
    public:
        using MatmulFn = std::function<bool(const float *, const float *, float *, std::size_t, std::size_t, std::size_t)>; // (A,B,C,M,N,K)

        TPGemmExecutor(MatmulFn fn, TPGemmExecConfig cfg, std::size_t M, std::size_t N, std::size_t K)
            : fn_(std::move(fn)), cfg_(cfg), M_(M), N_(N), K_(K)
        {
            if (cfg_.mode == TPGemmExecConfig::Mode::Row)
            {
                specA_ = compute_tp_partition(M_, cfg_.tp_size, cfg_.tp_rank, TPPartitionSpec::Axis::Row);
            }
            else
            {
                specB_ = compute_tp_partition(N_, cfg_.tp_size, cfg_.tp_rank, TPPartitionSpec::Axis::Col);
            }
        }

        TPGemmLocalResult run(const float *A, const float *B) const
        {
            TPGemmLocalResult r;
            r.partA = specA_;
            r.partB = specB_;
            auto t_start = std::chrono::high_resolution_clock::now();
            if (cfg_.tp_size <= 1)
            {
                r.M_local = M_;
                r.N_local = N_;
                r.buffer.resize(M_ * N_);
                if (!fn_(A, B, r.buffer.data(), M_, N_, K_))
                    throw std::runtime_error("TPGemmExecutor single-partition matmul failed");
                auto t_end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
                LOG_DEBUG("TP_GEMM_PART time_ms=" << ms << " mode=" << (cfg_.mode == TPGemmExecConfig::Mode::Row ? "row" : "col")
                                                  << " tp_size=" << cfg_.tp_size << " rank=" << cfg_.tp_rank
                                                  << " M=" << M_ << " N=" << N_ << " K=" << K_ << " local_M=" << r.M_local << " local_N=" << r.N_local);
                return r;
            }
            if (cfg_.mode == TPGemmExecConfig::Mode::Row)
            {
                r.M_local = specA_.local_dim;
                r.N_local = N_;
                const float *A_local = A + specA_.local_offset * K_; // row-major advance
                r.buffer.resize(r.M_local * r.N_local);
                if (!fn_(A_local, B, r.buffer.data(), r.M_local, r.N_local, K_))
                    throw std::runtime_error("Row-split local gemm failed");
            }
            else
            { // Column mode
                r.M_local = M_;
                r.N_local = specB_.local_dim;
                std::vector<float> B_packed;
                B_packed.resize(K_ * r.N_local);
                for (std::size_t k = 0; k < K_; ++k)
                {
                    const float *src = B + k * N_ + specB_.local_offset;
                    float *dst = B_packed.data() + k * r.N_local;
                    std::memcpy(dst, src, sizeof(float) * r.N_local);
                }
                r.buffer.resize(r.M_local * r.N_local);
                if (!fn_(A, B_packed.data(), r.buffer.data(), r.M_local, r.N_local, K_))
                    throw std::runtime_error("Column-split local gemm failed");
            }
            auto t_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
            LOG_DEBUG("TP_GEMM_PART time_ms=" << ms << " mode=" << (cfg_.mode == TPGemmExecConfig::Mode::Row ? "row" : "col")
                                              << " tp_size=" << cfg_.tp_size << " rank=" << cfg_.tp_rank
                                              << " M=" << M_ << " N=" << N_ << " K=" << K_ << " local_M=" << r.M_local << " local_N=" << r.N_local
                                              << " offset_M=" << specA_.local_offset << " offset_N=" << specB_.local_offset);
            return r;
        }

        // Reconstruction helpers
        static void reconstruct_columns(const std::vector<TPGemmLocalResult> &parts, float *C_full, std::size_t M, std::size_t N)
        {
            for (const auto &p : parts)
            {
                if (p.M_local != M)
                    throw std::runtime_error("Column reconstruct: M mismatch");
                for (std::size_t m = 0; m < M; ++m)
                {
                    std::memcpy(C_full + m * N + p.partB.local_offset, p.buffer.data() + m * p.N_local, sizeof(float) * p.N_local);
                }
            }
        }
        static void reconstruct_rows(const std::vector<TPGemmLocalResult> &parts, float *C_full, std::size_t M, std::size_t N)
        {
            for (const auto &p : parts)
            {
                if (p.N_local != N)
                    throw std::runtime_error("Row reconstruct: N mismatch");
                for (std::size_t r = 0; r < p.M_local; ++r)
                {
                    std::memcpy(C_full + (p.partA.local_offset + r) * N, p.buffer.data() + r * N, sizeof(float) * N);
                }
            }
        }

        TPPartitionSpec specA() const { return specA_; }
        TPPartitionSpec specB() const { return specB_; }

    private:
        MatmulFn fn_;
        TPGemmExecConfig cfg_;
        std::size_t M_, N_, K_;
        TPPartitionSpec specA_{}; // row partition
        TPPartitionSpec specB_{}; // col partition
    };

} // namespace llaminar
