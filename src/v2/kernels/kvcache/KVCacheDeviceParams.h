/**
 * @file KVCacheDeviceParams.h
 * @brief Device-side parameter buffer for graph-captured KV cache append stages
 *
 * Like RoPEDeviceParams and AttentionDeviceParams, this struct lives in a
 * device buffer and is updated via H2D memcpy before each graph replay.
 * The ring_append_kernel reads the head position from the device buffer
 * instead of a frozen scalar argument, allowing the write position to
 * change between graph replays.
 *
 * Flow:
 * 1. During graph capture: H2D memcpy (pinned host → device) + kernel launch
 *    with device_params pointer are recorded
 * 2. Between replays: setDynamicHead() updates pinned host copy +
 *    issues explicit H2D for stream-only mode
 * 3. During replay: captured H2D re-reads updated pinned host value →
 *    kernel sees new head position
 */

#pragma once

namespace llaminar2
{
    namespace kvcache
    {
        /**
         * @brief Device-side parameters for ring buffer KV cache append
         *
         * Contains the ring buffer head (write position) that changes
         * every decode step. Stored in a device buffer and read by the
         * ring_append_kernel_dynamic variant.
         */
        struct KVCacheDeviceParams
        {
            int head = 0; ///< Ring buffer write position
        };
    } // namespace kvcache
} // namespace llaminar2
