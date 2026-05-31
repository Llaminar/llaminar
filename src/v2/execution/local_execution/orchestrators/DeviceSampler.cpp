/**
 * @file DeviceSampler.cpp
 * @brief Implementation of GPU-side sampling across tensor-parallel device runners
 * @author David Sanftenberg
 * @date April 2026
 */

#include "DeviceSampler.h"
#include "IInferenceRunner.h"
#include "../../../backends/BackendManager.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/Sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace llaminar2
{

    int DeviceSampler::sampleGreedy(
        const std::vector<std::unique_ptr<IInferenceRunner>> &runners)
    {
        // Need at least 2 TP participants for cross-shard sampling.
        if (runners.size() < 2)
            return -1;

        // Collect per-device logits info via IInferenceRunner interface
        std::vector<LogitsLocalInfo> infos;
        infos.reserve(runners.size());

        for (const auto &runner : runners)
        {
            if (!runner || !runner->hasLogitsLocal())
                return -1;

            auto info = runner->getLogitsLocalInfo();
            if (!info || info.vocab_local == 0)
                return -1;
            LOG_TRACE("[DeviceSampler::sampleGreedy] Device "
                      << runner->primaryDeviceId().toString()
                      << " gpu_ptr=" << info.gpu_ptr << " vocab_local=" << info.vocab_local);

            infos.push_back(info);
        }

        return sampleGreedyFromLocalInfos(infos, 0);
    }

    int DeviceSampler::sampleGreedyFromLocalInfos(
        const std::vector<LogitsLocalInfo> &infos,
        int row)
    {
        if (infos.size() < 2 || row < 0)
            return -1;

        static bool logged_once = false;

        // Per-device argmax, then cross-device max on host.
        struct DeviceResult
        {
            float value;
            int local_index;
            size_t col_offset;
        };

        std::vector<DeviceResult> results;
        results.reserve(infos.size());
        size_t col_offset = 0;

        for (const auto &info : infos)
        {
            if (!info || info.vocab_local == 0)
                return -1;

            float max_val = -std::numeric_limits<float>::infinity();
            int max_idx = 0;

            if (info.device.has_value() && info.device->is_gpu())
            {
                if (!info.gpu_ptr)
                {
                    LOG_TRACE("[DeviceSampler::sampleGreedyFromLocalInfos] gpu_data_ptr() null for device "
                              << info.device->toString());
                    return -1;
                }

                IBackend *backend = getBackendFor(*info.device);
                if (!backend)
                    return -1;

                // Drive the multi-block argmax with the runner's arena-owned scratch
                // (null/zero capacity -> argmaxF32 fails, and we degrade to host-side
                // sampling below). No allocation happens on this hot path.
                if (!backend->argmaxF32(info.gpu_ptr,
                                        static_cast<int>(info.vocab_local),
                                        info.device->gpu_ordinal(),
                                        &max_val, &max_idx, info.stream,
                                        info.argmax_partial_vals,
                                        info.argmax_partial_idxs,
                                        info.argmax_partial_capacity))
                {
                    LOG_TRACE("[DeviceSampler::sampleGreedyFromLocalInfos] argmaxF32 failed for device "
                              << info.device->toString());
                    return -1;
                }
            }
            else
            {
                const auto &shape = info.tensor->shape();
                const size_t rows = shape.size() >= 2 ? shape[0] : 1;
                const size_t cols = info.vocab_local;
                if (cols == 0 || static_cast<size_t>(row) >= rows)
                    return -1;

                const float *data = info.tensor->fp32_data();
                if (!data)
                    return -1;

                const float *row_data = data + static_cast<size_t>(row) * cols;
                max_idx = 0;
                max_val = row_data[0];
                for (size_t i = 1; i < cols; ++i)
                {
                    if (row_data[i] > max_val)
                    {
                        max_val = row_data[i];
                        max_idx = static_cast<int>(i);
                    }
                }
            }

            LOG_TRACE("[DeviceSampler::sampleGreedyFromLocalInfos] local_argmax="
                      << max_idx << " val=" << max_val << " offset=" << col_offset);

            results.push_back({max_val, max_idx, col_offset});
            col_offset += info.vocab_local;
        }

        // Pick global winner across devices
        int best_token = -1;
        float best_value = -std::numeric_limits<float>::infinity();
        for (const auto &r : results)
        {
            const int token = static_cast<int>(r.col_offset) + r.local_index;
            if (r.value > best_value ||
                (r.value == best_value && (best_token < 0 || token < best_token)))
            {
                best_value = r.value;
                best_token = token;
            }
        }

        LOG_TRACE("[DeviceSampler::sampleGreedyFromLocalInfos] Winner: token=" << best_token
                                                                               << " val=" << best_value);

        if (!logged_once)
        {
            LOG_DEBUG("[DeviceSampler::sampleGreedyFromLocalInfos] cross-shard greedy argmax active ("
                      << infos.size() << " participants, vocab_local="
                      << infos[0].vocab_local << " each)");
            logged_once = true;
        }

        return best_token;
    }

    int DeviceSampler::sample(
        const std::vector<std::unique_ptr<IInferenceRunner>> &runners,
        const SamplingParams &params)
    {
        // Greedy: delegate to argmax path
        if (params.is_greedy())
            return sampleGreedy(runners);

        // Need at least 2 GPU devices
        if (runners.size() < 2)
            return -1;

        // Determine effective top-k
        int effective_k = params.top_k;
        if (effective_k <= 0)
            effective_k = 40; // Sensible default for top-p only mode
        if (effective_k > 256)
            effective_k = 256; // Kernel limit

        static bool logged_once = false;

        // Collect per-device logits info
        std::vector<LogitsLocalInfo> infos;
        infos.reserve(runners.size());

        for (const auto &runner : runners)
        {
            if (!runner || !runner->hasLogitsLocal())
                return -1;

            auto info = runner->getLogitsLocalInfo();
            if (!info || info.vocab_local == 0)
                return -1;

            if (!info.gpu_ptr)
                return -1;

            infos.push_back(info);
        }

        // Per-device GPU top-k → host merge
        struct DeviceCandidate
        {
            float value;
            int global_index;
        };

        std::vector<DeviceCandidate> all_candidates;
        all_candidates.reserve(runners.size() * static_cast<size_t>(effective_k));

        // Host buffers for top-k results (thread_local for reuse)
        thread_local std::vector<float> topk_values(256);
        thread_local std::vector<int> topk_indices(256);
        if (static_cast<int>(topk_values.size()) < effective_k)
        {
            topk_values.resize(static_cast<size_t>(effective_k));
            topk_indices.resize(static_cast<size_t>(effective_k));
        }

        size_t col_offset = 0;
        for (const auto &info : infos)
        {
            if (!info.device.has_value())
                return -1;

            IBackend *backend = getBackendFor(*info.device);
            if (!backend)
                return -1;

            if (!backend->topKF32(info.gpu_ptr,
                                  static_cast<int>(info.vocab_local),
                                  effective_k,
                                  info.device->gpu_ordinal(),
                                  topk_values.data(),
                                  topk_indices.data(), info.stream))
            {
                LOG_TRACE("[DeviceSampler::sample] topKF32 failed for device "
                          << info.device->toString());
                return -1;
            }

            for (int i = 0; i < effective_k; ++i)
            {
                if (topk_indices[i] >= 0)
                {
                    all_candidates.push_back(
                        {topk_values[i],
                         static_cast<int>(col_offset) + topk_indices[i]});
                }
            }
            col_offset += info.vocab_local;
        }

        if (all_candidates.empty())
            return -1;

        // Sort all candidates by value descending
        std::sort(all_candidates.begin(), all_candidates.end(),
                  [](const DeviceCandidate &a, const DeviceCandidate &b)
                  { return a.value > b.value; });

        // Keep only global top-k
        if (static_cast<int>(all_candidates.size()) > effective_k)
            all_candidates.resize(static_cast<size_t>(effective_k));

        // Apply temperature scaling + softmax
        float temperature = params.temperature;
        if (temperature <= 0.0f)
            temperature = 1.0f;

        float max_logit = all_candidates[0].value;
        std::vector<float> probs(all_candidates.size());
        float sum = 0.0f;
        for (size_t i = 0; i < all_candidates.size(); ++i)
        {
            probs[i] = std::exp((all_candidates[i].value - max_logit) / temperature);
            sum += probs[i];
        }
        for (auto &p : probs)
            p /= sum;

        // Top-p (nucleus) filtering
        float top_p = params.top_p;
        int nucleus_size = static_cast<int>(probs.size());
        if (top_p < 1.0f && top_p > 0.0f)
        {
            float cumulative = 0.0f;
            for (size_t i = 0; i < probs.size(); ++i)
            {
                cumulative += probs[i];
                if (cumulative >= top_p)
                {
                    nucleus_size = static_cast<int>(i) + 1;
                    break;
                }
            }
            // Renormalize
            float renorm_sum = 0.0f;
            for (int i = 0; i < nucleus_size; ++i)
                renorm_sum += probs[i];
            for (int i = 0; i < nucleus_size; ++i)
                probs[i] /= renorm_sum;
        }

        // Multinomial sampling
        thread_local std::mt19937 rng{std::random_device{}()};
        if (params.seed != 0)
        {
            static unsigned int last_seed = 0;
            if (params.seed != last_seed)
            {
                rng.seed(params.seed);
                last_seed = params.seed;
            }
        }

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        float cumulative = 0.0f;
        int selected = all_candidates[0].global_index;
        for (int i = 0; i < nucleus_size; ++i)
        {
            cumulative += probs[i];
            if (r <= cumulative)
            {
                selected = all_candidates[i].global_index;
                break;
            }
        }

        LOG_TRACE("[DeviceSampler::sample] top-k/p selected token=" << selected
                                                                    << " (k=" << effective_k << ", p=" << top_p
                                                                    << ", T=" << temperature << ", nucleus=" << nucleus_size << ")");

        if (!logged_once)
        {
            LOG_DEBUG("[DeviceSampler::sample] GPU-side top-k/top-p active ("
                      << runners.size() << " devices, k=" << effective_k
                      << ", vocab_local=" << infos[0].vocab_local << " each)");
            logged_once = true;
        }

        return selected;
    }

} // namespace llaminar2
