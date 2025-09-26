#pragma once

#include <cstddef>
#include <mpi.h>

namespace llaminar::kernels
{

    struct RMSNormArgs
    {
        const float *input{nullptr};
        const float *weight{nullptr};
        float *output{nullptr};
        int rows{0};
        int cols{0};
        float epsilon{1e-6f};
    };

    struct DistributedRMSNormContext
    {
        int world_size{1};
        int rank{0};
        MPI_Comm comm{MPI_COMM_NULL};
        bool use_barrier{false};
    };

    // Row-major RMSNorm executed entirely on the calling rank.
    void rmsnorm_row_major(const RMSNormArgs &args);

    // Distributed RMSNorm for layouts where each rank owns a contiguous subset of rows.
    // `row_offset` is the global row index of the first local row owned by this rank.
    void rmsnorm_distributed(const RMSNormArgs &local_args,
                             int global_rows,
                             int row_offset,
                             const DistributedRMSNormContext &ctx);

    // Helper to compute inverse scale given accumulated sum of squares and population size.
    float rmsnorm_inverse_scale(double sum_sq, int population, float epsilon);

} // namespace llaminar::kernels
