/**
 * @file softmax_core.h
 * @brief Unified row-major and (future) distributed softmax primitives.
 * @author David Sanftenberg
 */
#pragma once

#include <cstddef>
#include <mpi.h>

namespace llaminar::kernels
{

    struct SoftmaxRowArgs
    {
        float *scores{nullptr}; // in-place scores (rows x cols)
        int rows{0};
        int cols{0};
        bool causal{false}; // if true: mask positions j>i to 0 probability
        float scale{1.0f};  // optional multiplicative scale before exp
    };

    struct DistributedSoftmaxCtx
    {
        MPI_Comm comm{MPI_COMM_NULL};
        int world_size{1};
        int rank{0};
        bool use_barrier{false}; // allow inserting MPI_Barrier around collectives
        // Future: scratch buffers / fused reduction workspace
    };

    // Pure row-major stable softmax with optional causal masking + scaling.
    // Scores replaced in-place with probabilities; masked positions become 0.
    void softmax_row_major(const SoftmaxRowArgs &args);

    // True distributed softmax supporting column partitioning across ranks.
    // Each rank holds a contiguous column slice [local_col_offset, local_col_offset + local_args.cols).
    // Parameters:
    //  local_args.rows : number of local rows owned (row sharding) – if rows are fully replicated set row_offset=0 & global_rows=local_args.rows
    //  global_rows     : total global row count
    //  row_offset      : global index of first local row (0 if replicated rows)
    //  global_cols     : total number of columns across all ranks
    //  local_col_offset: starting global column index for this rank's slice
    // Algorithm:
    //  1. Local pass computes per-row local max (respecting causal masking w.r.t global column index)
    //  2. Allreduce (MAX) to obtain global row maxima
    //  3. Local exp + partial sum using global maxima
    //  4. Allreduce (SUM) to obtain global denominators
    //  5. Local normalization in-place
    // Causal masking: probability for any column j where j>row_index is zero.
    void softmax_distributed(const SoftmaxRowArgs &local_args,
                             int global_rows,
                             int row_offset,
                             int global_cols,
                             int local_col_offset,
                             const DistributedSoftmaxCtx &ctx);

} // namespace llaminar::kernels
