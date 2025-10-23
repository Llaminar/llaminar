#pragma once
#include <mpi.h>
#include "PerfCounters.h"
#include <memory>
#include <cstring>
#include "../tensors/TensorBase.h"
#include "../tensors/sharded_SimpleTensor.h"
#include "../tensors/ShardSpec.h"

namespace llaminar
{
    /**
     * @brief Perform an in-place MPI_Allreduce (SUM) for a tensor that represents an additive shard.
     *
     * This helper is intended for row/feature sharded outputs where each rank holds a partial
     * contribution that must be summed across ranks to obtain a replicated activation.
     *
     * Semantics:
     * - For row/feature (Hidden) or head-reduced (after projecting concatenated heads through a
     *   row-sharded matrix) outputs, each rank's buffer is the same logical shape and must be summed.
     * - The function detects a ShardedSimpleTensor and validates the axis when `validate_axis` is true.
     * - If the tensor is not sharded (no ShardedSimpleTensor), it is treated as already replicated
     *   and the function returns immediately.
     *
     * @param tensor Shared pointer to TensorBase (may or may not be sharded)
     * @param validate_axis If true, ensures shard axis is Hidden; otherwise skips axis check.
     */
    inline void reduce_row_sharded_inplace(const std::shared_ptr<TensorBase> &tensor,
                                           bool validate_axis = true,
                                           bool treat_nonsharded_as_additive = false)
    {
        auto *sharded = dynamic_cast<ShardedSimpleTensor *>(tensor.get());
        if (!sharded)
        {
            if (treat_nonsharded_as_additive)
            {
                PerfAllreduce(MPI_IN_PLACE, tensor->data(), (int)tensor->size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            }
            return; // replicated or other tensor type (unless forced additive)
        }
        const ShardSpec &spec = sharded->shard_spec();
        if (validate_axis && !(spec.axis == ShardSpec::Axis::Hidden || spec.axis == ShardSpec::Axis::None))
        {
            if (spec.axis != ShardSpec::Axis::None)
            {
                return; // unexpected axis
            }
            return; // replicated
        }
        PerfAllreduce(MPI_IN_PLACE, tensor->data(), (int)tensor->size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }

} // namespace llaminar
