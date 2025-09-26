#include "normalization.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <mpi.h>

namespace llaminar::kernels
{
    namespace
    {
        inline bool validate_args(const RMSNormArgs &args)
        {
            return args.input != nullptr && args.output != nullptr && args.rows > 0 && args.cols > 0;
        }

        inline void apply_row(const float *src_row, float *dst_row, const float *weight, int cols, float inv_scale)
        {
            for (int c = 0; c < cols; ++c)
            {
                float value = src_row[c] * inv_scale;
                if (weight)
                    value *= weight[c];
                dst_row[c] = value;
            }
        }
    } // namespace

    void rmsnorm_row_major(const RMSNormArgs &args)
    {
        if (!validate_args(args))
            return;

        const float *input = args.input;
        float *output = args.output;
        const float *weight = args.weight;
        int rows = args.rows;
        int cols = args.cols;
        float eps = args.epsilon;

        for (int r = 0; r < rows; ++r)
        {
            const float *src_row = input + static_cast<std::size_t>(r) * cols;
            float *dst_row = output + static_cast<std::size_t>(r) * cols;

            double sum_sq = 0.0;
            for (int c = 0; c < cols; ++c)
            {
                double v = src_row[c];
                sum_sq += v * v;
            }

            float inv_scale = rmsnorm_inverse_scale(sum_sq, cols, eps);
            apply_row(src_row, dst_row, weight, cols, inv_scale);
        }
    }

    float rmsnorm_inverse_scale(double sum_sq, int population, float epsilon)
    {
        if (population <= 0)
            return 1.0f;
        double mean_sq = sum_sq / static_cast<double>(population);
        double denom = std::sqrt(mean_sq + static_cast<double>(epsilon));
        if (denom <= 0.0)
            return 1.0f;
        return static_cast<float>(1.0 / denom);
    }

    void rmsnorm_distributed(const RMSNormArgs &local_args,
                             int global_rows,
                             int row_offset,
                             const DistributedRMSNormContext &ctx)
    {
        if (!validate_args(local_args) || global_rows <= 0)
            return;

        const float *input = local_args.input;
        float *output = local_args.output;
        const float *weight = local_args.weight;
        int local_rows = local_args.rows;
        int cols = local_args.cols;
        float eps = local_args.epsilon;

        if (ctx.use_barrier && ctx.comm != MPI_COMM_NULL)
            MPI_Barrier(ctx.comm);

        // Treat rows as fully local (current distribution strategy).
        rmsnorm_row_major(local_args);
    }

} // namespace llaminar::kernels
