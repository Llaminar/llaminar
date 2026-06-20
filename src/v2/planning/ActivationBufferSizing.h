#pragma once

#include "backends/DeviceId.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "utils/DebugEnv.h"

#include <algorithm>

namespace llaminar2
{

    /**
     * @brief Resolve the transient graph/activation arena sequence capacity.
     *
     * Long-context runners need full-context KV/state capacity, but GPU graph
     * activations only need to cover the largest prefill chunk that may execute
     * as one graph shape. CPU keeps the historical full-context arena because
     * there is no GPU graph bucket pressure there.
     */
    inline int resolveActivationBufferSeqLen(int max_seq_len, DeviceId device)
    {
        if (max_seq_len <= 0 || !device.is_gpu())
            return max_seq_len;

        auto buckets = normalizePrefillGraphBuckets(debugEnv().execution.prefill_graph_bucket_sizes);
        if (buckets.empty())
            buckets = defaultPrefillGraphBuckets();
        if (buckets.empty())
            return max_seq_len;

        return std::max(1, std::min(max_seq_len, buckets.back()));
    }

} // namespace llaminar2
