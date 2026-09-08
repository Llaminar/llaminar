/**
 * @file RoPEDeviceParams.h
 * @brief Device-side parameter buffer for graph-captured RoPE stages
 *
 * Like AttentionDeviceParams, this struct lives in a workspace-owned device
 * buffer. The buffer is updated before graph capture/replay, then captured
 * RoPE execution reads the device pointer instead of frozen scalar arguments.
 *
 * Flow:
 * 1. During graph capture: kernel launch with device_params pointer is recorded
 * 2. Before capture/replay: prepareGraphLaunch() enqueues a launch-by-value
 *    device publication on the explicit graph stream
 * 3. During capture/replay: the graph records and launches RoPE kernels only
 *
 * A device publication kernel is intentional here. Reusing mutable pinned host
 * staging for asynchronous H2D copies permits a later request to overwrite an
 * earlier request's source before DMA consumes it.
 */

#pragma once

namespace llaminar2
{
    namespace rope
    {
        struct RoPEDeviceParams
        {
            int pos_offset = 0; ///< Position offset for decode (seq_len=1) or contiguous positions
        };
    } // namespace rope
} // namespace llaminar2
