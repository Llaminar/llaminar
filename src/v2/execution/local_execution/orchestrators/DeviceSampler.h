/**
 * @file DeviceSampler.h
 * @brief GPU-side sampling across tensor-parallel device runners
 *
 * Extracted from RankOrchestrator to isolate GPU sampling logic:
 * per-device argmax (greedy) and top-k/top-p (non-greedy) with cross-device
 * result merging on host.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include <memory>
#include <vector>

namespace llaminar2
{

    class IInferenceRunner;
    struct LogitsLocalInfo;
    struct SamplingParams;

    /**
     * @brief GPU-side sampling across tensor-parallel device runners.
     *
     * For greedy sampling: runs argmax on each device's local logits shard via
     * IBackend::argmaxF32(), then picks the global winner on host. Only 8 bytes
     * (float value + int index) are transferred per device instead of the full
     * vocab shard (~600 KB per device for 152K vocab).
     *
     * For non-greedy sampling: runs top-k on each device via IBackend::topKF32(),
     * merges candidates on host, applies temperature + softmax + top-p nucleus
     * filtering, then multinomial samples.
     *
     * All methods return -1 if GPU-side sampling is unsupported (single device,
     * CPU-only, no column-parallel LM head, backend unavailable). The caller
     * then falls back to CPU logits gathering + host-side sampling.
     */
    class DeviceSampler
    {
    public:
        /**
         * @brief GPU-side greedy argmax across TP device runners.
         *
         * @param runners Device runners with logits_local on GPU
         * @return Token ID (>= 0) if succeeded, -1 if not supported
         */
        static int sampleGreedy(const std::vector<std::unique_ptr<IInferenceRunner>> &runners);

        /**
         * @brief Greedy argmax across already-collected local logits views.
         *
         * This is used for main logits, MTP sidecar logits, and all-position
         * verifier logits. GPU views use backend argmax when scratch is
         * available; CPU views sample directly from the local tensor data.
         */
        static int sampleGreedyFromLocalInfos(const std::vector<LogitsLocalInfo> &infos,
                                              int row = 0);

        /**
         * @brief GPU-side sampling with top-k/top-p support.
         *
         * For greedy params, delegates to sampleGreedy(). For non-greedy,
         * runs per-device top-k on GPU, then host-side merge + softmax +
         * nucleus filtering + multinomial sampling.
         *
         * @param runners Device runners with logits_local on GPU
         * @param params Sampling parameters (temperature, top_k, top_p, seed)
         * @return Token ID (>= 0) if succeeded, -1 if not supported
         */
        static int sample(const std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                          const SamplingParams &params);
    };

} // namespace llaminar2
