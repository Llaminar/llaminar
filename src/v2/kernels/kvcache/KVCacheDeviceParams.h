/**
 * @file KVCacheDeviceParams.h
 * @brief Device-side parameter buffer for graph-captured KV cache append stages
 *
 * Like RoPEDeviceParams and AttentionDeviceParams, this struct lives in a
 * workspace-owned device buffer. It is updated before graph capture/replay,
 * and the ring_append_kernel reads the head position from the device buffer
 * instead of a frozen scalar argument.
 *
 * Flow:
 * 1. Before capture/replay: setDynamicHead() uploads the latest value on the
 *    explicit graph stream
 * 2. During capture/replay: only the dynamic append kernel is recorded/launched
 * 3. Replay callbacks advance host-side ring metadata after graph launch
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
