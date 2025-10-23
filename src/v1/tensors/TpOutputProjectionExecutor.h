/**
 * @file TpOutputProjectionExecutor.h
 * @brief Tensor-parallel (intra-socket) output projection executors (column & row) with reconstruction helpers.
 */
#pragma once

#include "TpPartition.h"
#include <vector>
#include <functional>
#include <cassert>
#include <stdexcept>

namespace llaminar
{

    struct TPOutputExecConfig
    {
        int tp_size = 1;        ///< number of partitions
        int tp_rank = 0;        ///< this partition index
        bool row_split = false; ///< false = column split (split N), true = row split (split M)
    };

    /**
     * @brief Result descriptor for local execution.
     */
    struct TPOutputLocalResult
    {
        TPPartitionSpec partA;     ///< row split spec (if active)
        TPPartitionSpec partB;     ///< col split spec (if active)
        std::vector<float> buffer; ///< local output slice (shape: M_local x N_local)
        std::size_t M_local = 0;
        std::size_t N_local = 0;
    };

    /**
     * @brief Execute output projection C = A * B (A:[M,K], B:[K,N]) under a TP partition.
     * Column split: each rank owns subset of columns (N partition). Row split: subset of rows (M partition).
     * Uses a provided matmul functor (adaptive backend) for the local slice.
     */
    class TPOutputProjectionExecutor
    {
    public:
        using MatmulFn = std::function<bool(const float *, const float *, float *, std::size_t, std::size_t, std::size_t)>; // (A,B,C, M,N,K)

        TPOutputProjectionExecutor(MatmulFn fn, TPOutputExecConfig cfg, std::size_t M, std::size_t N, std::size_t K)
            : fn_(std::move(fn)), cfg_(cfg), M_(M), N_(N), K_(K)
        {
            if (cfg_.row_split)
            {
                specA_ = compute_tp_partition(M_, cfg_.tp_size, cfg_.tp_rank, TPPartitionSpec::Axis::Row);
            }
            else
            {
                specB_ = compute_tp_partition(N_, cfg_.tp_size, cfg_.tp_rank, TPPartitionSpec::Axis::Col);
            }
        }

        TPOutputLocalResult run(const float *A, const float *B) const
        {
            TPOutputLocalResult r;
            r.partA = specA_;
            r.partB = specB_;
            if (cfg_.tp_size <= 1)
            {
                r.M_local = M_;
                r.N_local = N_;
                r.buffer.resize(M_ * N_);
                bool ok = fn_(A, B, r.buffer.data(), M_, N_, K_);
                if (!ok)
                    throw std::runtime_error("TPOutputProjectionExecutor single-partition matmul failed");
                return r;
            }
            if (cfg_.row_split)
            {
                // Row split: take rows [offset, offset+local_dim) from A
                r.M_local = specA_.local_dim;
                r.N_local = N_;
                const float *A_local = A + specA_.local_offset * K_; // row-major A:[M,K]
                r.buffer.resize(r.M_local * r.N_local);
                bool ok = fn_(A_local, B, r.buffer.data(), r.M_local, r.N_local, K_);
                if (!ok)
                    throw std::runtime_error("Row-split local matmul failed");
            }
            else
            {
                // Column split: need packed subset of columns from B
                r.M_local = M_;
                r.N_local = specB_.local_dim;
                std::vector<float> B_packed;
                B_packed.resize(K_ * r.N_local);
                for (std::size_t k = 0; k < K_; ++k)
                {
                    const float *row_src = B + k * N_ + specB_.local_offset;
                    float *row_dst = B_packed.data() + k * r.N_local;
                    std::memcpy(row_dst, row_src, sizeof(float) * r.N_local);
                }
                r.buffer.resize(r.M_local * r.N_local);
                bool ok = fn_(A, B_packed.data(), r.buffer.data(), r.M_local, r.N_local, K_);
                if (!ok)
                    throw std::runtime_error("Column-split local matmul failed");
            }
            return r;
        }

        // Reconstruction helpers -------------------------------------------------

        static void reconstruct_columns(const std::vector<TPOutputLocalResult> &parts, float *C_full, std::size_t M, std::size_t N)
        {
            // Expect each part has M_local==M and disjoint column ranges summing to N
            std::size_t col_accum = 0;
            for (const auto &p : parts)
            {
                if (p.M_local != M)
                    throw std::runtime_error("Column reconstruct: mismatched M_local");
                for (std::size_t m = 0; m < M; ++m)
                {
                    std::memcpy(C_full + m * N + p.partB.local_offset,
                                p.buffer.data() + m * p.N_local,
                                sizeof(float) * p.N_local);
                }
                col_accum += p.N_local;
            }
            if (col_accum < N)
                throw std::runtime_error("Column reconstruct: insufficient columns");
        }

        static void reconstruct_rows(const std::vector<TPOutputLocalResult> &parts, float *C_full, std::size_t M, std::size_t N)
        {
            // Expect disjoint row ranges
            std::size_t row_accum = 0;
            for (const auto &p : parts)
            {
                if (p.N_local != N)
                    throw std::runtime_error("Row reconstruct: mismatched N_local");
                for (std::size_t r_idx = 0; r_idx < p.M_local; ++r_idx)
                {
                    std::memcpy(C_full + (p.partA.local_offset + r_idx) * N,
                                p.buffer.data() + r_idx * N,
                                sizeof(float) * N);
                }
                row_accum += p.M_local;
            }
            if (row_accum < M)
                throw std::runtime_error("Row reconstruct: insufficient rows");
        }

        TPPartitionSpec specA() const { return specA_; }
        TPPartitionSpec specB() const { return specB_; }

    private:
        MatmulFn fn_;
        TPOutputExecConfig cfg_;
        std::size_t M_, N_, K_;
        TPPartitionSpec specA_{}; // row partition (if row_split)
        TPPartitionSpec specB_{}; // col partition (if !row_split)
    };

} // namespace llaminar
