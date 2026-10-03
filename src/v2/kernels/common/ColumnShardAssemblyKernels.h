/**
 * @file ColumnShardAssemblyKernels.h
 * @brief Shared exact-byte column-shard assembly for native graph collectives.
 *
 * Both KV state gathering and projection-distributed MoE use the same mapping:
 * [participant,row,local-column] to [row,participant,local-column]. Assembly
 * neither adds nor converts FP32 values. The caller owns admitted disjoint
 * buffers and the collective-to-consumer event/graph dependency.
 */
#pragma once
#include "backends/DeviceId.h"

namespace llaminar2
{
    /**
     * @brief Enqueue the existing native column deinterleave kernel on one exact stream.
     * @param device Participant-local physical GPU.
     * @param rank_major Native allgather result, covering rows*participants*columns FP32 values.
     * @param row_major Disjoint destination with the same physical extent.
     * @param rows Captured row count.
     * @param participants Positive number of equal-sized output shards in communicator order.
     * @param columns Complete local output width.
     * @param stream Exact non-null producer/consumer stream.
     * @return Whether geometry/storage were valid and the kernel was enqueued.
     * @note Swapping rows and participants performs the inverse packing. The
     *       one-row case remains a disjoint copy, never an aliasing shortcut.
     */
    bool assembleColumnShardsFP32(DeviceId device, const float *rank_major, float *row_major,
        int rows, int participants, int columns, void *stream);
}
