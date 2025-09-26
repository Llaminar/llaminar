#pragma once

#include <mpi.h>

namespace llaminar::kernels
{

    struct SoftmaxArgs
    {
        float *scores{nullptr};
        int rows{0};
        int cols{0};
        bool apply_causal_mask{false};
        float scale{1.0f};
    };

    struct DistributedSoftmaxContext
    {
        int world_size{1};
        int rank{0};
        MPI_Comm comm{MPI_COMM_NULL};
        bool use_barrier{false};
    };

    void softmax_row_major(const SoftmaxArgs &args);

    void softmax_distributed(const SoftmaxArgs &local_args,
                             int global_rows,
                             int row_offset,
                             const DistributedSoftmaxContext &ctx);

} // namespace llaminar::kernels
