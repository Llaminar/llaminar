/**
 * @file DeviceSampler.cpp
 * @brief Implementation of GPU-side sampling across tensor-parallel device runners
 * @author David Sanftenberg
 * @date April 2026
 */

#include "DeviceSampler.h"
#include "IInferenceRunner.h"
#include "../../../backends/BackendManager.h"
#include "../../../kernels/common/SamplingMath.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/Sampler.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <vector>

namespace llaminar2
{
    namespace
    {
        size_t logitsRowStride(const LogitsLocalInfo &info)
        {
            return info.row_stride > 0 ? info.row_stride : info.vocab_local;
        }
    } // namespace

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

            auto info = runner->consumeLogitsLocalInfoForSampling();
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
        const bool use_explicit_vocab_offsets =
            std::any_of(
                infos.begin(),
                infos.end(),
                [](const LogitsLocalInfo &info)
                {
                    return info.vocab_offset != 0;
                });

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
                if (!info.stream)
                {
                    LOG_ERROR("[DeviceSampler::sampleGreedyFromLocalInfos] explicit GPU stream required for device "
                              << info.device->toString());
                    return -1;
                }

                const auto &shape = info.tensor->shape();
                const size_t rows = shape.size() >= 2 ? shape[0] : 1;
                const size_t cols = info.vocab_local;
                const size_t row_stride = logitsRowStride(info);
                if (cols == 0 || row_stride < cols || static_cast<size_t>(row) >= rows)
                    return -1;

                const auto *row_ptr =
                    static_cast<const float *>(info.gpu_ptr) + static_cast<size_t>(row) * row_stride;

                // Drive the multi-block argmax with the runner's arena-owned scratch
                // (null/zero capacity -> argmaxF32 fails, and we degrade to host-side
                // sampling below). No allocation happens on this hot path.
                if (!backend->argmaxF32(row_ptr,
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
                const size_t row_stride = logitsRowStride(info);
                if (cols == 0 || row_stride < cols || static_cast<size_t>(row) >= rows)
                    return -1;

                const float *data = info.tensor->fp32_data();
                if (!data)
                    return -1;

                const float *row_data = data + static_cast<size_t>(row) * row_stride;
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

            const size_t global_offset =
                use_explicit_vocab_offsets ? info.vocab_offset : col_offset;
            results.push_back({max_val, max_idx, global_offset});
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

    bool DeviceSampler::sampleGreedyRowsFromLocalInfos(
        const std::vector<LogitsLocalInfo> &infos,
        int start_row,
        int row_count,
        int32_t *out_tokens)
    {
        if (infos.size() < 2 || start_row < 0 || row_count <= 0 || !out_tokens)
            return false;

        struct ShardRows
        {
            size_t col_offset = 0;
            std::vector<float> values;
            std::vector<int> indices;
        };

        std::vector<ShardRows> shard_rows;
        shard_rows.reserve(infos.size());
        size_t col_offset = 0;
        const bool use_explicit_vocab_offsets =
            std::any_of(
                infos.begin(),
                infos.end(),
                [](const LogitsLocalInfo &info)
                {
                    return info.vocab_offset != 0;
                });

        for (const auto &info : infos)
        {
            if (!info || info.vocab_local == 0 || !info.tensor)
                return false;

            const auto &shape = info.tensor->shape();
            const size_t rows = shape.size() >= 2 ? shape[0] : 1;
            const size_t cols = info.vocab_local;
            const size_t row_stride = logitsRowStride(info);
            if (cols == 0 ||
                row_stride < cols ||
                static_cast<size_t>(start_row) >= rows ||
                static_cast<size_t>(start_row + row_count) > rows ||
                cols > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                return false;
            }

            ShardRows rows_out;
            rows_out.col_offset =
                use_explicit_vocab_offsets ? info.vocab_offset : col_offset;
            rows_out.values.assign(static_cast<size_t>(row_count), 0.0f);
            rows_out.indices.assign(static_cast<size_t>(row_count), -1);

            if (info.device.has_value() && info.device->is_gpu())
            {
                /*
                 * The backend batched argmax API assumes contiguous rows. If a
                 * tensor has padded local-logit rows, keep the exact old
                 * per-row sampler path instead of silently reading padding.
                 */
                if (!info.gpu_ptr || row_stride != cols || !info.stream)
                    return false;

                IBackend *backend = getBackendFor(*info.device);
                if (!backend ||
                    !info.argmax_partial_vals ||
                    !info.argmax_partial_idxs ||
                    info.argmax_partial_capacity < row_count)
                {
                    return false;
                }

                const auto *base = static_cast<const float *>(info.gpu_ptr);
                const void *first_row =
                    base + static_cast<size_t>(start_row) * row_stride;
                if (!backend->argmaxF32BatchedRows(
                        first_row,
                        row_count,
                        static_cast<int>(cols),
                        info.device->gpu_ordinal(),
                        rows_out.values.data(),
                        rows_out.indices.data(),
                        info.stream,
                        info.argmax_partial_vals,
                        info.argmax_partial_idxs,
                        info.argmax_partial_capacity))
                {
                    return false;
                }
            }
            else
            {
                const float *data = info.tensor->fp32_data();
                if (!data)
                    return false;

                for (int i = 0; i < row_count; ++i)
                {
                    const float *row_data =
                        data +
                        static_cast<size_t>(start_row + i) * row_stride;
                    int max_idx = 0;
                    float max_val = row_data[0];
                    for (size_t col = 1; col < cols; ++col)
                    {
                        if (row_data[col] > max_val)
                        {
                            max_val = row_data[col];
                            max_idx = static_cast<int>(col);
                        }
                    }
                    rows_out.values[static_cast<size_t>(i)] = max_val;
                    rows_out.indices[static_cast<size_t>(i)] = max_idx;
                }
            }

            shard_rows.push_back(std::move(rows_out));
            col_offset += cols;
        }

        for (int row = 0; row < row_count; ++row)
        {
            int best_token = -1;
            float best_value = -std::numeric_limits<float>::infinity();
            for (const auto &shard : shard_rows)
            {
                const int local_index = shard.indices[static_cast<size_t>(row)];
                if (local_index < 0)
                    return false;
                const int token =
                    static_cast<int>(shard.col_offset) + local_index;
                const float value = shard.values[static_cast<size_t>(row)];
                if (value > best_value ||
                    (value == best_value &&
                     (best_token < 0 || token < best_token)))
                {
                    best_value = value;
                    best_token = token;
                }
            }
            if (best_token < 0)
                return false;
            out_tokens[row] = static_cast<int32_t>(best_token);
        }

        return true;
    }

    int DeviceSampler::sample(
        const std::vector<std::unique_ptr<IInferenceRunner>> &runners,
        const SamplingParams &params,
        float threshold)
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

            auto info = runner->consumeLogitsLocalInfoForSampling();
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

        // Sort all candidates by value descending. Equal logits use the lower
        // token id, matching greedy tie-breaking and keeping top-k/top-p rows
        // independent of backend-local top-k emission order.
        std::sort(all_candidates.begin(), all_candidates.end(),
                  [](const DeviceCandidate &a, const DeviceCandidate &b)
                  {
                      if (a.value != b.value)
                          return a.value > b.value;
                      return a.global_index < b.global_index;
                  });

        // Keep only global top-k
        if (static_cast<int>(all_candidates.size()) > effective_k)
            all_candidates.resize(static_cast<size_t>(effective_k));

        std::vector<float> sorted_logits(all_candidates.size(), 0.0f);
        std::vector<int> sorted_token_ids(all_candidates.size(), -1);
        std::vector<float> scratch(all_candidates.size(), 0.0f);
        for (size_t i = 0; i < all_candidates.size(); ++i)
        {
            sorted_logits[i] = all_candidates[i].value;
            sorted_token_ids[i] = all_candidates[i].global_index;
        }
        const int selected =
            sampling_math::sample_topk_topp_from_sorted_with_threshold(
                sorted_logits.data(),
                sorted_token_ids.data(),
                static_cast<int>(sorted_logits.size()),
                params.top_p,
                params.temperature,
                threshold,
                scratch.data());

        LOG_TRACE("[DeviceSampler::sample] top-k/p selected token=" << selected
                                                                    << " (k=" << effective_k << ", p=" << params.top_p
                                                                    << ", T=" << params.temperature << ")");

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
