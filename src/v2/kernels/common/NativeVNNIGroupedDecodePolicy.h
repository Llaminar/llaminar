/**
 * @file NativeVNNIGroupedDecodePolicy.h
 * @brief Shared capacity contract for decode-equivalent NativeVNNI split-K work.
 *
 * Serial M=1 NativeVNNI dispatch chooses a generated K-partition count. Grouped
 * verifier and routed-MoE kernels inherit that count because changing it changes
 * the FP32 expression tree. This header gives graph workspace planners and GPU
 * launchers one common upper bound, preventing a generated serial policy from
 * selecting arithmetic that a captured grouped graph cannot represent.
 */

#pragma once

namespace llaminar2
{
    /**
     * @brief Graph-safe limits shared by serial and grouped NativeVNNI kernels.
     *
     * The value is a capacity bound, not a tuning choice. Runtime launchers use
     * the exact generated serial-M1 partition count for their geometry and only
     * reserve enough persistent workspace to make every valid count representable.
     */
    struct NativeVNNIGroupedDecodePolicy final
    {
        /** Maximum generated K-partition count accepted by captured GPU graphs. */
        static constexpr int maximum_k_partitions = 64;
    };
}
