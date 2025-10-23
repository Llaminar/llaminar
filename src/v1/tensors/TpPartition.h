/**
 * @file TpPartition.h
 * @brief Tensor Parallel (intra-socket) partition specification and splitter interfaces.
 */
#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cstring>

namespace llaminar
{

    /**
     * @brief Describes an intra-socket tensor parallel partition of a single logical matrix.
     *
     * Supports row or column partitioning. A single-partition (tp_size=1) acts as a transparent
     * passthrough and should impose no overhead.
     */
    struct TPPartitionSpec
    {
        enum class Axis
        {
            Row,
            Col
        };
        Axis axis = Axis::Row;        ///< Split along which logical axis of the left operand (A) for matmul A*B
        int tp_size = 1;              ///< Number of intra-socket partitions
        int tp_rank = 0;              ///< Local partition index [0, tp_size)
        std::size_t global_dim = 0;   ///< Global dimension along split axis
        std::size_t local_offset = 0; ///< Offset into the global dimension for this partition
        std::size_t local_dim = 0;    ///< Local slice size along axis

        bool active() const { return tp_size > 1; }
        bool is_row_split() const { return axis == Axis::Row; }
        bool is_col_split() const { return axis == Axis::Col; }

        std::string to_string() const
        {
            char buf[256];
            snprintf(buf, sizeof(buf), "tp axis=%s rank=%d/%d offset=%zu size=%zu global=%zu",
                     (axis == Axis::Row ? "row" : "col"), tp_rank, tp_size, local_offset, local_dim, global_dim);
            return std::string(buf);
        }
    };

    /**
     * @brief Compute a block partition (ceil distribution) for a TP dimension.
     */
    inline TPPartitionSpec compute_tp_partition(std::size_t global_dim, int tp_size, int tp_rank, TPPartitionSpec::Axis axis)
    {
        TPPartitionSpec spec;
        spec.axis = axis;
        spec.tp_size = tp_size;
        spec.tp_rank = tp_rank;
        spec.global_dim = global_dim;
        if (tp_size <= 1)
        {
            spec.local_dim = global_dim;
            spec.local_offset = 0;
            return spec;
        }
        std::size_t base = global_dim / tp_size;
        std::size_t rem = global_dim % tp_size;
        spec.local_dim = base + (tp_rank < (int)rem ? 1 : 0);
        spec.local_offset = base * tp_rank + (tp_rank < (int)rem ? tp_rank : rem);
        return spec;
    }

    /**
     * @brief Abstract splitter interface for matmul. Provides local views & a merge hook.
     *
     * For now we expose a minimal functional wrapper so we can slot in specialized kernels later.
     */
    class MatmulSplitter
    {
    public:
        virtual ~MatmulSplitter() = default;

        struct Args
        {
            const float *A; // [M,K] or shard
            const float *B; // [K,N] or shard
            float *C;       // [M,N] local output buffer (row/col shard or full)
            std::size_t M;
            std::size_t N;
            std::size_t K;
        };

        virtual bool run(const Args &a) = 0; // returns true on success
        virtual TPPartitionSpec specA() const = 0;
        virtual TPPartitionSpec specB() const = 0;
    };

    /**
     * @brief Trivial splitter: single partition fallback, directly invokes provided matmul functor.
     */
    class TrivialMatmulSplitter : public MatmulSplitter
    {
    public:
        using MatmulFn = std::function<bool(const float *, const float *, float *, std::size_t, std::size_t, std::size_t)>;
        explicit TrivialMatmulSplitter(MatmulFn fn) : fn_(std::move(fn)) {}
        bool run(const Args &a) override { return fn_(a.A, a.B, a.C, a.M, a.N, a.K); }
        TPPartitionSpec specA() const override { return TPPartitionSpec{}; }
        TPPartitionSpec specB() const override
        {
            TPPartitionSpec s;
            s.axis = TPPartitionSpec::Axis::Col;
            return s;
        }

    private:
        MatmulFn fn_;
    };

    /**
     * @brief Column-splitting matmul splitter: partitions N dimension of B (and C) across tp_size.
     * Assumes caller provides full A and full B; this splitter will compute only its column slice into C_local.
     * Reconstruction (concatenating columns) is left to higher-level orchestration.
     */
    class ColumnSplitMatmulSplitter : public MatmulSplitter
    {
    public:
        using MatmulFn = std::function<bool(const float *, const float *, float *, std::size_t, std::size_t, std::size_t)>;
        ColumnSplitMatmulSplitter(std::size_t global_N, int tp_size, int tp_rank, MatmulFn fn)
            : specB_(compute_tp_partition(global_N, tp_size, tp_rank, TPPartitionSpec::Axis::Col)), fn_(std::move(fn)) {}

        bool run(const Args &a) override
        {
            if (!specB_.active())
            {
                return fn_(a.A, a.B, a.C, a.M, a.N, a.K);
            }
            // Need to pack the column slice [offset, offset+local_dim) into contiguous K*local_dim buffer
            // Original B layout: row-major [K,N_full]; element (k,j) at B[k*N_full + j]
            std::vector<float> packed;
            packed.resize(a.K * specB_.local_dim);
            for (std::size_t k = 0; k < a.K; ++k)
            {
                const float *row_src = a.B + k * specB_.global_dim + specB_.local_offset; // stride by global N
                float *row_dst = packed.data() + k * specB_.local_dim;
                std::memcpy(row_dst, row_src, sizeof(float) * specB_.local_dim);
            }
            return fn_(a.A, packed.data(), a.C, a.M, specB_.local_dim, a.K);
        }

        TPPartitionSpec specA() const override { return TPPartitionSpec{}; }
        TPPartitionSpec specB() const override { return specB_; }

    private:
        TPPartitionSpec specB_;
        MatmulFn fn_;
    };

    /**
     * @brief Row-splitting matmul splitter: partitions M dimension of A (and C) across tp_size.
     * Caller must provide full B; A_local and C_local correspond to local M slice.
     */
    class RowSplitMatmulSplitter : public MatmulSplitter
    {
    public:
        using MatmulFn = std::function<bool(const float *, const float *, float *, std::size_t, std::size_t, std::size_t)>;
        RowSplitMatmulSplitter(std::size_t global_M, int tp_size, int tp_rank, MatmulFn fn)
            : specA_(compute_tp_partition(global_M, tp_size, tp_rank, TPPartitionSpec::Axis::Row)), fn_(std::move(fn)) {}

        bool run(const Args &a) override
        {
            // a.M is expected to be specA_.local_dim (caller supplies local A pointer already).
            return fn_(a.A, a.B, a.C, specA_.local_dim, a.N, a.K);
        }

        TPPartitionSpec specA() const override { return specA_; }
        TPPartitionSpec specB() const override { return TPPartitionSpec{}; }

    private:
        TPPartitionSpec specA_;
        MatmulFn fn_;
    };

} // namespace llaminar
