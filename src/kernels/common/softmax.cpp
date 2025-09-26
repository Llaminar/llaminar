#include "softmax.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace llaminar::kernels
{
    namespace
    {
        inline float masked_value(float raw, bool mask_active)
        {
            return mask_active ? -std::numeric_limits<float>::infinity() : raw;
        }
    } // namespace

    void softmax_row_major(const SoftmaxArgs &args)
    {
        if (!args.scores || args.rows <= 0 || args.cols <= 0)
            return;

        float *scores = args.scores;
        int rows = args.rows;
        int cols = args.cols;
        bool causal = args.apply_causal_mask;
        float scale = args.scale;

        for (int r = 0; r < rows; ++r)
        {
            float *row = scores + static_cast<std::size_t>(r) * cols;
            float row_max = -std::numeric_limits<float>::infinity();

            for (int c = 0; c < cols; ++c)
            {
                bool masked = causal && c > r;
                float val = row[c];
                if (scale != 1.0f)
                    val *= scale;
                val = masked_value(val, masked);
                row_max = std::max(row_max, val);
            }

            if (!std::isfinite(row_max))
                row_max = 0.0f;

            double denom = 0.0;
            for (int c = 0; c < cols; ++c)
            {
                bool masked = causal && c > r;
                float val = row[c];
                if (scale != 1.0f)
                    val *= scale;
                val = masked_value(val, masked);
                float exp_val = masked ? 0.0f : std::exp(val - row_max);
                row[c] = exp_val;
                denom += exp_val;
            }

            if (denom <= 0.0)
                denom = 1.0;
            float inv_denom = static_cast<float>(1.0 / denom);

            for (int c = 0; c < cols; ++c)
            {
                bool masked = causal && c > r;
                row[c] = masked ? 0.0f : row[c] * inv_denom;
            }
        }
    }

    void softmax_distributed(const SoftmaxArgs &local_args,
                             int global_rows,
                             int row_offset,
                             const DistributedSoftmaxContext &ctx)
    {
        if (!local_args.scores || local_args.rows < 0 || local_args.cols <= 0 || global_rows <= 0)
            return;

        if (ctx.use_barrier && ctx.comm != MPI_COMM_NULL)
            MPI_Barrier(ctx.comm);

        // Current implementation assumes each rank owns full rows, so the computation is local.
        softmax_row_major(local_args);
    }

} // namespace llaminar::kernels
