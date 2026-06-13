/**
 * @file DeviceGraphOrchestrator.cpp
 * @brief Implementation of Qwen2 compute graph orchestrator
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file implements the execution layer for Qwen2 models, managing
 * graph execution, device contexts, and caching.
 */

#include "DeviceGraphOrchestrator.h"
#include "MTPSidecarStreamBinding.h"
#include "../../config/HybridPrecisionConfig.h"
#include "../../config/InferenceMode.h"
#include "../../../loaders/WeightManager.h"
#include "../../../loaders/WeightPlacementMap.h"
#include "../../../loaders/IWeightManager.h"
#include "../../../loaders/IWeightPlacementMap.h"
#include "../../../config/TensorParallelConfig.h"
#include "../../../config/PipelineConfig.h"
#include "../../../collective/ILocalTPContext.h" // createLocalTPContext()
#include "../../../collective/ILocalPPContext.h" // createLocalPPContext(), HierarchicalPPConfig
#include "../../../collective/PPStage.h"         // PPStage variant type
#include "../../../collective/BackendRouter.h"   // GlobalBackendRouter for PP copy
#include "../../../backends/GPUDeviceContextPool.h"
#include "../graph/GraphCaptureGuard.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/MPIContext.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../tensors/TensorFactory.h"
#include "../../../tensors/TensorClasses.h" // For FP32Tensor::createMapped()
#include "../../../kernels/cpu/CPUKVCache.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/HybridKVCacheConfig.h"
#include "../../../kernels/IHybridKVCache.h"
#include "../../compute_stages/stages/MoEExpertComputeStage.h"
#include "../../mtp/MTPSpecKVPublisher.h"
#include "../../mtp/MTPSpecStatePublisher.h"
#include "../../moe/ExpertWeightTransfer.h"
#include "../../moe/ExpertWeightPayloadProvider.h"
#include "../../../loaders/PreparedWeightStore.h"
#include "../../../loaders/WeightPlan.h"
#include "../../../backends/BackendManager.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../kernels/common/SamplingMath.h"
#include "../../moe/MoERebalanceController.h"
#include "../../../utils/Sampler.h" // LogitPenalty
#include "execution/prefix_cache/BlockHash.h"
#include "execution/prefix_cache/DeviceHotPrefixStorageBackend.h"
#include "execution/prefix_cache/DiskPrefixStorageBackend.h"
#include "execution/prefix_cache/PrefixCacheFingerprint.h"
#include "execution/prefix_cache/PrefixStateCache.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "transfer/TransferEngine.h"
#include <chrono>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        constexpr size_t kStochasticDistributionMaxK = 256;
        constexpr size_t kStochasticTopKSmallKCap = 64;
        constexpr size_t kStochasticTopKPartialBlocks = 128;
        constexpr size_t kStochasticTopKSmallKThreads = 64;
        constexpr size_t kStochasticTargetRows = 4; // verifier M=2..4 includes terminal row
        constexpr size_t kStochasticDraftRows = 3;  // --mtp-draft-tokens max
        constexpr size_t kStochasticTopKPartialRows = kStochasticTargetRows;
        constexpr size_t kStochasticTopKPartialCapacity =
            kStochasticTopKPartialRows *
            kStochasticTopKPartialBlocks *
            kStochasticTopKSmallKCap;
        /**
         * @brief Logical perf/capture lane for publishing accepted shifted MTP KV rows.
         *
         * One-row sequential commits, batched hidden-row catch-up, and
         * device-token target commits use separate graph-cache shapes
         * internally, but they all implement the same accepted-state
         * publication contract.  Keeping the diagnostics on one context makes
         * graph capture/replay ownership easy to audit.
         */
        constexpr const char *kMTPDecodeCatchupContext = "mtp_decode_catchup";
        /**
         * @brief Fixed token-input slots for graph-captured MTP sidecar caches.
         *
         * GPU graph replay records the embedding stage's token pointer.  Each
         * sidecar cache therefore receives a stable slice of the arena-owned
         * condition-token buffer so full-draft, chained-draft, catch-up, and
         * shifted-prefill graphs cannot overwrite each other's mutable input
         * while another capture stream is still consuming it.
         */
        constexpr int kMTPSidecarConditionTokenSlotCount = 10;

        /**
         * @brief Complete a short helper stream owned by the orchestrator.
         *
         * MTP maintenance helpers such as terminal-hidden row selection read
         * from the live main hidden-state buffer after an ordinary forward.
         * Arena output events make the selected row visible to later MTP
         * readers, but they do not protect the input buffer from the next main
         * forward overwriting it.  When the helper creates its own stream, it
         * therefore also owns completing that read before returning.  Callers
         * that pass a stream keep responsibility for ordering and readiness.
         */
        bool synchronizeOwnedGPUHelperStream(
            DeviceId device,
            void *stream,
            const char *operation)
        {
            if (!device.is_gpu())
                return true;
            if (!stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] "
                          << (operation ? operation : "gpu_helper")
                          << " requires an explicit non-null stream on "
                          << device.toString());
                return false;
            }

            auto &gpu_ctx = GPUDeviceContextPool::instance().getContext(device);
            if (!gpu_ctx.synchronizeStreamChecked(stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] "
                          << (operation ? operation : "gpu_helper")
                          << " stream synchronization failed on "
                          << device.toString());
                return false;
            }
            return true;
        }

        class ScopedStringOverride
        {
        public:
            ScopedStringOverride(std::string &target, std::string value)
                : target_(target), previous_(target)
            {
                target_ = std::move(value);
            }

            ~ScopedStringOverride()
            {
                target_ = std::move(previous_);
            }

            ScopedStringOverride(const ScopedStringOverride &) = delete;
            ScopedStringOverride &operator=(const ScopedStringOverride &) = delete;

        private:
            std::string &target_;
            std::string previous_;
        };

        /// @brief Emit a coarse VRAM checkpoint for orchestrator allocation phases.
        void logOrchestratorVramTrace(DeviceId device, const char *label)
        {
            if (!debugEnv().vram_trace || !device.is_gpu())
                return;

            IBackend *backend = getBackendFor(device);
            if (!backend)
                return;

            const int ordinal = device.gpu_ordinal();
            const size_t free_bytes = backend->deviceMemoryFree(ordinal);
            const size_t total_bytes = backend->deviceMemoryTotal(ordinal);
            const size_t used_bytes = total_bytes > free_bytes ? total_bytes - free_bytes : 0;
            LOG_INFO("[VRAM_TRACE] " << label
                                     << " device=" << device.toString()
                                     << " used_mib=" << (used_bytes / (1024 * 1024))
                                     << " free_mib=" << (free_bytes / (1024 * 1024))
                                     << " total_mib=" << (total_bytes / (1024 * 1024)));
        }

        bool applyPenaltiesToTensorRowOnDevice(
            TensorBase *tensor,
            DeviceId device,
            const std::vector<LogitPenalty> &penalties,
            int vocab_size,
            int row,
            int token_offset,
            void *stream,
            const char *operation)
        {
            if (penalties.empty())
                return true;
            if (!tensor || !device.is_gpu() || !stream || vocab_size <= 0 || row < 0)
                return false;
            if (!tensor->deviceValid())
                return false;

            IBackend *backend = getBackendFor(device);
            if (!backend)
                return false;

            void *gpu_ptr = tensor->gpu_data_ptr();
            if (!gpu_ptr)
                return false;

            const auto &shape = tensor->shape();
            if (shape.empty())
                return false;
            const size_t rows = shape.size() >= 2 ? shape[0] : 1;
            const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
            if (cols == 0 || static_cast<size_t>(row) >= rows ||
                cols > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                return false;
            }

            std::vector<int> local_token_ids;
            std::vector<float> local_penalties;
            local_token_ids.reserve(penalties.size());
            local_penalties.reserve(penalties.size());
            const int local_vocab = static_cast<int>(cols);
            const int local_begin = std::max(0, token_offset);
            const int local_end = local_begin + local_vocab;
            for (const auto &penalty : penalties)
            {
                if (penalty.token_id < local_begin || penalty.token_id >= local_end)
                    continue;
                local_token_ids.push_back(penalty.token_id - local_begin);
                local_penalties.push_back(penalty.penalty);
            }

            if (local_token_ids.empty())
                return true;

            float *row_ptr = static_cast<float *>(gpu_ptr) +
                             static_cast<size_t>(row) * cols;
            const bool ok = backend->applyLogitPenaltiesF32(
                row_ptr,
                local_token_ids.data(),
                local_penalties.data(),
                static_cast<int>(local_token_ids.size()),
                local_vocab,
                device.gpu_ordinal(),
                stream);
            if (!ok)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] "
                          << (operation ? operation : "applyPenaltiesToTensorRowOnDevice")
                          << " failed on " << device.toString());
            }
            return ok;
        }

        bool prefixCacheTraceEnabled()
        {
            const char *value = DebugEnv::envValue("LLAMINAR_PREFIX_CACHE_TRACE");
            return value && value[0] != '\0' && value[0] != '0';
        }

        std::string lowerASCII(std::string value)
        {
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch)
                           { return static_cast<char>(std::tolower(ch)); });
            return value;
        }

        std::string boolTag(bool value)
        {
            return value ? "true" : "false";
        }

        size_t fp32LogitsRowBytes(const TensorBase *tensor)
        {
            if (!tensor)
                return 0;
            const auto &shape = tensor->shape();
            if (shape.empty())
                return 0;
            const size_t cols = shape.size() >= 2 ? shape.back() : shape.front();
            return cols * sizeof(float);
        }

        bool containsAny(const std::string &haystack, std::initializer_list<const char *> needles)
        {
            for (const char *needle : needles)
            {
                if (haystack.find(needle) != std::string::npos)
                {
                    return true;
                }
            }
            return false;
        }

        void validateMoERebalanceDomain(
            const DeviceGraphOrchestrator &orchestrator,
            const std::string &domain_id,
            const char *operation)
        {
            if (domain_id.empty())
                return;

            const auto *controller = orchestrator.moeRebalanceControllerForDomain(domain_id);
            if (!controller)
            {
                throw std::runtime_error(
                    std::string(operation) + " rejected for MoE rebalance domain '" +
                    domain_id + "': runner has no domain controller");
            }
        }

        struct GreedyLogitCandidate
        {
            float value = 0.0f;
            int32_t token = -1;
            int32_t valid = 0;
            int32_t reserved = 0;
        };

        int vocabOffsetForTPConfig(const GraphConfig &config)
        {
            if (!config.lm_head_column_parallel || !config.tp_config)
                return 0;

            const int rank = config.local_rank >= 0 ? config.local_rank : config.tp_device_idx;
            if (rank < 0 || rank >= config.tp_config->worldSize())
                return 0;

            return config.tp_config->forRank(rank).vocab_start;
        }

        bool isBetterGreedyCandidate(const GreedyLogitCandidate &candidate,
                                     const GreedyLogitCandidate &best)
        {
            if (!candidate.valid)
                return false;
            if (!best.valid)
                return true;
            if (candidate.value > best.value)
                return true;
            return candidate.value == best.value && candidate.token >= 0 &&
                   (best.token < 0 || candidate.token < best.token);
        }

        bool synchronizeGpuBackendsBeforeMmapRelease()
        {
            bool ok = true;
            auto sync_backend = [&ok](const char *name, IBackend *backend)
            {
                if (!backend)
                    return;

                const int device_count = backend->deviceCount();
                for (int device_idx = 0; device_idx < device_count; ++device_idx)
                {
                    if (debugEnv().vram_trace)
                    {
                        LOG_INFO("[VRAM_TRACE] mmap_release.before_sync backend=" << name
                                                                                  << " device=" << device_idx);
                    }
                    else
                    {
                        LOG_DEBUG("[DeviceGraphOrchestrator] Synchronizing " << name << ":" << device_idx
                                                                             << " before mmap DONTNEED");
                    }
                    if (!backend->synchronize(device_idx))
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Failed to synchronize " << name << ":"
                                                                                     << device_idx
                                                                                     << " before mmap DONTNEED");
                        ok = false;
                    }
                    else if (debugEnv().vram_trace)
                    {
                        LOG_INFO("[VRAM_TRACE] mmap_release.after_sync backend=" << name
                                                                                 << " device=" << device_idx);
                    }
                }
            };

            sync_backend("CUDA", getCUDABackend());
            sync_backend("ROCm", getROCmBackend());
            return ok;
        }

        const char *perfPhaseName()
        {
            switch (GraphExecutorStats::currentPhase())
            {
            case ExecutionPhase::PREFILL:
                return "prefill";
            case ExecutionPhase::DECODE:
                return "decode";
            case ExecutionPhase::COMBINED:
            default:
                return "combined";
            }
        }

        bool isTruthyProfilingEnv(const char *name)
        {
            const char *value = DebugEnv::envValue(name);
            if (!value)
                return false;
            if (std::strcmp(value, "0") == 0 ||
                std::strcmp(value, "false") == 0 ||
                std::strcmp(value, "FALSE") == 0 ||
                std::strcmp(value, "off") == 0 ||
                std::strcmp(value, "OFF") == 0)
            {
                return false;
            }
            return true;
        }

        bool greedyMarginStatsEnabled()
        {
            return PerfStatsCollector::isEnabled() &&
                   isTruthyProfilingEnv("LLAMINAR_GREEDY_MARGIN_STATS");
        }

        void recordGreedyMarginStats(
            const char *source,
            const DeviceId &device,
            int row,
            float top1,
            float top2)
        {
            if (!greedyMarginStatsEnabled())
                return;

            const float margin = top1 - top2;
            const std::string phase = perfPhaseName();
            const std::string device_name = device.toString();
            PerfStatsCollector::Tags tags{
                {"source", source ? source : "unknown"},
                {"row", std::to_string(row)}};

            PerfStatsCollector::addCounter(
                "sampling", "greedy_samples", 1.0, phase, device_name, tags);
            PerfStatsCollector::addCounter(
                "sampling", "greedy_top2_margin", margin, phase, device_name, tags);

            auto near_tie = [&](const char *name, float threshold)
            {
                if (margin <= threshold)
                {
                    PerfStatsCollector::addCounter(
                        "sampling", name, 1.0, phase, device_name, tags);
                }
            };
            near_tie("greedy_near_tie_le_1e-6", 1.0e-6f);
            near_tie("greedy_near_tie_le_1e-5", 1.0e-5f);
            near_tie("greedy_near_tie_le_1e-4", 1.0e-4f);
            near_tie("greedy_near_tie_le_1e-3", 1.0e-3f);
            near_tie("greedy_near_tie_le_1e-2", 1.0e-2f);
        }

        void recordGreedyMarginUnavailable(
            const char *source,
            const DeviceId &device,
            int row)
        {
            if (!greedyMarginStatsEnabled())
                return;

            PerfStatsCollector::addCounter(
                "sampling",
                "greedy_top2_margin_unavailable",
                1.0,
                perfPhaseName(),
                device.toString(),
                PerfStatsCollector::Tags{
                    {"source", source ? source : "unknown"},
                    {"row", std::to_string(row)}});
        }

        GreedyLogitCandidate sampleGreedyCandidateFromTensor(
            TensorBase *tensor,
            int row,
            int token_offset,
            void *argmax_partial_vals = nullptr,
            void *argmax_partial_idxs = nullptr,
            int argmax_partial_capacity = 0,
            void *stream = nullptr,
            const char *source = "unknown")
        {
            GreedyLogitCandidate candidate;
            if (!tensor || row < 0)
                return candidate;

            const auto &shape = tensor->shape();
            if (shape.empty())
                return candidate;

            const size_t rows = shape.size() >= 2 ? shape[0] : 1;
            const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
            if (cols == 0 || static_cast<size_t>(row) >= rows ||
                cols > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                return candidate;
            }

            const size_t row_offset = static_cast<size_t>(row) * cols;
            float max_val = 0.0f;
            int max_idx = -1;

            auto device_opt = tensor->current_device();
            if (device_opt.has_value() && device_opt->is_gpu() && tensor->deviceValid())
            {
                if (!stream)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] GPU greedy sampling requires an explicit non-null stream");
                    return candidate;
                }
                IBackend *backend = getBackendFor(*device_opt);
                const void *gpu_ptr = tensor->gpu_data_ptr();
                if (backend && gpu_ptr)
                {
                    const auto *base = static_cast<const float *>(gpu_ptr);
                    const void *target_row = base + row_offset;
                    if (backend->argmaxF32(target_row,
                                           static_cast<int>(cols),
                                           device_opt->gpu_ordinal(),
                                           &max_val,
                                           &max_idx,
                                           stream,
                                           argmax_partial_vals,
                                           argmax_partial_idxs,
                                           argmax_partial_capacity))
                    {
                        candidate.value = max_val;
                        candidate.token = token_offset + max_idx;
                        candidate.valid = 1;

                        if (greedyMarginStatsEnabled())
                        {
                            float top_values[2] = {};
                            int top_indices[2] = {};
                            if (backend->topKF32(target_row,
                                                 static_cast<int>(cols),
                                                 2,
                                                 device_opt->gpu_ordinal(),
                                                 top_values,
                                                 top_indices,
                                                 stream))
                            {
                                recordGreedyMarginStats(
                                    source,
                                    *device_opt,
                                    row,
                                    top_values[0],
                                    top_values[1]);
                            }
                            else
                            {
                                recordGreedyMarginUnavailable(source, *device_opt, row);
                            }
                        }
                        return candidate;
                    }
                }

                if (stream)
                {
                    auto &pool = GPUDeviceContextPool::instance();
                    pool.getContext(*device_opt).synchronizeStream(stream);
                }
            }

            if (!tensor->hostValid())
            {
                auto download = TransferEngine::instance().download(tensor);
                if (!download.success)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to download logits for greedy sampling: "
                              << download.error);
                    return candidate;
                }
            }

            const float *data = tensor->fp32_data();
            if (!data)
                return candidate;

            const float *row_data = data + row_offset;
            max_idx = 0;
            max_val = row_data[0];
            float second_val = -std::numeric_limits<float>::infinity();
            for (size_t i = 1; i < cols; ++i)
            {
                if (row_data[i] > max_val)
                {
                    second_val = max_val;
                    max_val = row_data[i];
                    max_idx = static_cast<int>(i);
                }
                else if (row_data[i] > second_val)
                {
                    second_val = row_data[i];
                }
            }

            candidate.value = max_val;
            candidate.token = token_offset + max_idx;
            candidate.valid = 1;
            if (cols >= 2)
            {
                auto device_for_stats = device_opt.value_or(DeviceId::cpu());
                recordGreedyMarginStats(source, device_for_stats, row, max_val, second_val);
            }
            return candidate;
        }

        int coordinateGreedyCandidate(
            const GreedyLogitCandidate &local_candidate,
            IGlobalTPContext *ctx)
        {
            if (!ctx || ctx->degree() <= 1)
                return local_candidate.valid ? local_candidate.token : -1;

            std::vector<GreedyLogitCandidate> candidates(static_cast<size_t>(ctx->degree()));
            if (!ctx->allgatherBytes(&local_candidate,
                                     candidates.data(),
                                     sizeof(GreedyLogitCandidate)))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to allgather GlobalTP greedy candidates");
                return -1;
            }

            GreedyLogitCandidate best;
            for (const auto &candidate : candidates)
            {
                if (isBetterGreedyCandidate(candidate, best))
                    best = candidate;
            }
            return best.valid ? best.token : -1;
        }

        int prefixFALayerForIndex(const IKVCache &cache, int fa_index)
        {
            if (fa_index < 0)
            {
                return -1;
            }

            const auto *hybrid = dynamic_cast<const IHybridKVCache *>(&cache);
            if (!hybrid)
            {
                return cache.first_layer_index() + fa_index;
            }

            int seen = 0;
            for (int layer = 0; layer < cache.n_layers(); ++layer)
            {
                if (!hybrid->isFullAttentionLayer(layer))
                {
                    continue;
                }
                if (seen == fa_index)
                {
                    return cache.first_layer_index() + layer;
                }
                ++seen;
            }
            return -1;
        }

        bool attachMTPPayloadLayout(PrefixPayloadLayout &layout, const IKVCache &mtp_cache)
        {
            if (layout.block_size <= 0)
            {
                return false;
            }

            PrefixPayloadLayout mtp_layout = buildDensePrefixPayloadLayout(
                mtp_cache,
                layout.device,
                layout.block_size);
            if (mtp_layout.fa_layers <= 0 || mtp_layout.faKVBytes() == 0)
            {
                return false;
            }

            layout.mtp_layers = mtp_layout.fa_layers;
            layout.mtp_local_kv_heads = mtp_layout.local_kv_heads;
            layout.mtp_kv_head_start = mtp_layout.kv_head_start;
            layout.mtp_head_dim = mtp_layout.head_dim;
            layout.mtp_k_precision = mtp_layout.k_precision;
            layout.mtp_v_precision = mtp_layout.v_precision;
            layout.mtp_kv_layout = mtp_layout.kv_layout;
            layout.bytes_per_mtp_layer_k = mtp_layout.bytes_per_fa_layer_k;
            layout.bytes_per_mtp_layer_v = mtp_layout.bytes_per_fa_layer_v;
            layout.mtp_kv_bytes = mtp_layout.faKVBytes();
            layout.includes_mtp_state = layout.mtp_kv_bytes > 0;
            return layout.includes_mtp_state;
        }

        int mtpTokenStartForPrefixBlock(const PrefixCacheKey &key)
        {
            return key.token_start == 0 ? 0 : key.token_start - 1;
        }

        int mtpTokenCountForPrefixBlock(const PrefixCacheKey &key)
        {
            if (key.token_count <= 0)
            {
                return 0;
            }
            return key.token_start == 0 ? std::max(0, key.token_count - 1) : key.token_count;
        }

        size_t tensorElementCountForBytes(size_t bytes)
        {
            return (bytes + sizeof(float) - 1) / sizeof(float);
        }

        std::shared_ptr<TensorBase> allocateDeviceByteStorage(size_t bytes,
                                                              DeviceId device)
        {
            if (bytes == 0 || !device.is_gpu())
            {
                return nullptr;
            }

            auto tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{tensorElementCountForBytes(bytes)},
                DeviceId::cpu());
            if (!tensor->allocateOnDevice(device))
            {
                return nullptr;
            }
            return tensor;
        }

        bool exportMTPPrefixPayload(const IKVCache &mtp_cache,
                                    int seq_idx,
                                    const PrefixCacheKey &key,
                                    PrefixBlockHandle *handle,
                                    void *stream = nullptr)
        {
            if (!handle || !handle->layout.includes_mtp_state)
            {
                return true;
            }

            const int token_count = mtpTokenCountForPrefixBlock(key);
            if (token_count == 0)
            {
                return true;
            }
            if (token_count < 0 || !handle->mtpKData() || !handle->mtpVData())
            {
                return false;
            }

            for (int local_layer = 0; local_layer < handle->layout.mtp_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(mtp_cache, local_layer);
                if (global_layer < 0)
                {
                    return false;
                }

                uint8_t *k_dst = handle->mtpKData() +
                                 static_cast<size_t>(local_layer) * handle->layout.bytes_per_mtp_layer_k;
                uint8_t *v_dst = handle->mtpVData() +
                                 static_cast<size_t>(local_layer) * handle->layout.bytes_per_mtp_layer_v;

                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = mtpTokenStartForPrefixBlock(key);
                desc.token_count = token_count;
                desc.stream = stream;
                if (!mtp_cache.exportLogicalBlock(desc, k_dst, v_dst))
                {
                    return false;
                }
            }
            return true;
        }

        bool importMTPPrefixPayload(IKVCache &mtp_cache,
                                    int seq_idx,
                                    const PrefixBlockHandle &handle,
                                    void *stream = nullptr)
        {
            if (!handle.layout.includes_mtp_state)
            {
                return true;
            }

            const int token_count = mtpTokenCountForPrefixBlock(handle.key);
            if (token_count == 0)
            {
                return true;
            }
            if (token_count < 0 || !handle.mtpKData() || !handle.mtpVData())
            {
                return false;
            }

            for (int local_layer = 0; local_layer < handle.layout.mtp_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(mtp_cache, local_layer);
                if (global_layer < 0)
                {
                    return false;
                }

                const uint8_t *k_src = handle.mtpKData() +
                                       static_cast<size_t>(local_layer) * handle.layout.bytes_per_mtp_layer_k;
                const uint8_t *v_src = handle.mtpVData() +
                                       static_cast<size_t>(local_layer) * handle.layout.bytes_per_mtp_layer_v;

                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = mtpTokenStartForPrefixBlock(handle.key);
                desc.token_count = token_count;
                desc.stream = stream;
                if (!mtp_cache.importLogicalBlock(desc, k_src, v_src))
                {
                    return false;
                }
            }
            return true;
        }

        void *hybridPayloadHostPtr(PrefixBlockHandle &handle)
        {
            return handle.layout.hybrid_host_state_bytes > 0 ? handle.hybrid_payload : nullptr;
        }

        const void *hybridPayloadHostPtr(const PrefixBlockHandle &handle)
        {
            return handle.layout.hybrid_host_state_bytes > 0 ? handle.hybrid_payload : nullptr;
        }

        void *hybridPayloadDevicePtr(PrefixBlockHandle &handle)
        {
            if (handle.device_hybrid_storage)
            {
                return handle.device_hybrid_storage->gpu_data_ptr();
            }
            return nullptr;
        }

        const void *hybridPayloadDevicePtr(const PrefixBlockHandle &handle)
        {
            if (handle.device_hybrid_storage)
            {
                return handle.device_hybrid_storage->gpu_data_ptr();
            }
            return nullptr;
        }

        bool exportHybridPrefixPayload(
            const IKVCache &cache,
            int seq_idx,
            int logical_token_count,
            PrefixBlockHandle *handle,
            bool synchronize = true,
            void *stream = nullptr)
        {
            if (!handle || !handle->layout.includes_hybrid_state)
            {
                return true;
            }
            const auto *hybrid = dynamic_cast<const IHybridKVCache *>(&cache);
            void *host_payload = hybridPayloadHostPtr(*handle);
            void *device_payload = hybridPayloadDevicePtr(*handle);
            if (!host_payload && handle->layout.hybrid_device_state_bytes > 0 && handle->hybrid_payload)
                host_payload = handle->hybrid_payload;
            const bool host_staged_device_state =
                handle->layout.hybrid_device_state_bytes > 0 &&
                !device_payload &&
                host_payload != nullptr;
            if (!hybrid ||
                (handle->layout.hybrid_host_state_bytes > 0 && !host_payload) ||
                (handle->layout.hybrid_device_state_bytes > 0 &&
                 !device_payload &&
                 !host_staged_device_state))
            {
                return false;
            }

            HybridPrefixStateDescriptor desc;
            desc.seq_idx = seq_idx;
            desc.logical_token_count = logical_token_count;
            desc.stream = stream;
            desc.synchronize = synchronize;
            desc.include_host_state = handle->layout.hybrid_host_state_bytes > 0;
            desc.include_device_state = handle->layout.hybrid_device_state_bytes > 0;
            if (!hybrid->exportHybridPrefixState(
                    desc,
                    host_payload,
                    device_payload))
            {
                return false;
            }
            handle->has_hybrid_state = true;
            return true;
        }

        bool importHybridPrefixPayload(
            IKVCache &cache,
            const PrefixBlockHandle &handle,
            int seq_idx,
            bool synchronize = true,
            void *stream = nullptr)
        {
            if (!handle.layout.includes_hybrid_state)
            {
                return true;
            }
            auto *hybrid = dynamic_cast<IHybridKVCache *>(&cache);
            const void *host_payload = hybridPayloadHostPtr(handle);
            const void *device_payload = hybridPayloadDevicePtr(handle);
            if (!host_payload && handle.layout.hybrid_device_state_bytes > 0 && handle.hybrid_payload)
                host_payload = handle.hybrid_payload;
            const bool host_staged_device_state =
                handle.layout.hybrid_device_state_bytes > 0 &&
                !device_payload &&
                host_payload != nullptr;
            if (!hybrid || !handle.has_hybrid_state ||
                (handle.layout.hybrid_host_state_bytes > 0 && !host_payload) ||
                (handle.layout.hybrid_device_state_bytes > 0 &&
                 !device_payload &&
                 !host_staged_device_state))
            {
                return false;
            }

            HybridPrefixStateDescriptor desc;
            desc.seq_idx = seq_idx;
            desc.logical_token_count = handle.key.token_start + handle.key.token_count;
            desc.stream = stream;
            desc.synchronize = synchronize;
            desc.include_host_state = handle.layout.hybrid_host_state_bytes > 0;
            desc.include_device_state = handle.layout.hybrid_device_state_bytes > 0;
            return hybrid->importHybridPrefixState(
                desc,
                host_payload,
                device_payload);
        }

        void resetHybridPrefixPayloadState(IKVCache &cache)
        {
            auto *hybrid = dynamic_cast<IHybridKVCache *>(&cache);
            if (!hybrid)
            {
                return;
            }
            for (int layer = 0; layer < cache.n_layers(); ++layer)
            {
                if (hybrid->isGDNLayer(layer))
                {
                    cache.clear_layer(layer);
                }
            }
        }

        bool liveCheckpointHasHybridState(const IKVCache &cache,
                                          DeviceId device,
                                          int cached_tokens)
        {
            if (cached_tokens <= 0)
            {
                return false;
            }

            const PrefixPayloadLayout layout =
                buildDensePrefixPayloadLayout(cache, device, cached_tokens);
            return layout.includes_hybrid_state && layout.hybrid_state_bytes > 0;
        }

        bool liveCheckpointLacksHeadroom(const IKVCache &cache,
                                         int cached_tokens,
                                         int speculative_append_headroom)
        {
            if (cached_tokens < 0 || speculative_append_headroom < 0)
            {
                return true;
            }

            return cached_tokens + speculative_append_headroom > cache.max_seq_len();
        }

        PrefixPayloadLayout hybridOnlyCheckpointLayout(const PrefixPayloadLayout &layout)
        {
            PrefixPayloadLayout hybrid_layout = layout;
            hybrid_layout.fa_layers = 0;
            hybrid_layout.local_kv_heads = 0;
            hybrid_layout.kv_head_start = 0;
            hybrid_layout.head_dim = 0;
            hybrid_layout.bytes_per_fa_layer_k = 0;
            hybrid_layout.bytes_per_fa_layer_v = 0;
            hybrid_layout.terminal_hidden_bytes = 0;
            hybrid_layout.terminal_logits_bytes = 0;
            hybrid_layout.includes_terminal_hidden = false;
            hybrid_layout.includes_terminal_logits = false;
            hybrid_layout.mtp_layers = 0;
            hybrid_layout.mtp_local_kv_heads = 0;
            hybrid_layout.mtp_kv_head_start = 0;
            hybrid_layout.mtp_head_dim = 0;
            hybrid_layout.bytes_per_mtp_layer_k = 0;
            hybrid_layout.bytes_per_mtp_layer_v = 0;
            hybrid_layout.mtp_kv_bytes = 0;
            hybrid_layout.includes_mtp_state = false;
            return hybrid_layout;
        }

        PrefixPayloadLayout liveHybridCheckpointLayout(const PrefixPayloadLayout &layout,
                                                       bool device_only)
        {
            PrefixPayloadLayout hybrid_layout = hybridOnlyCheckpointLayout(layout);
            if (device_only && hybrid_layout.hybrid_device_state_bytes > 0)
            {
                hybrid_layout.hybrid_host_state_bytes = 0;
                hybrid_layout.hybrid_state_bytes = hybrid_layout.hybrid_device_state_bytes;
            }
            hybrid_layout.includes_hybrid_state = hybrid_layout.hybrid_state_bytes > 0;
            return hybrid_layout;
        }
    }

    // =========================================================================
    // Shared Executor Configuration
    // =========================================================================

    void DeviceGraphOrchestrator::configureExecutor()
    {
        GraphExecutorConfig exec_config = graph_builder_->config().executor_config;
        exec_config.default_device = graph_builder_->config().default_device;

        const auto &env = debugEnv();
        exec_config.enable_profiling = exec_config.enable_profiling || graph_builder_->config().enable_profiling || env.execution.executor_profiling;
        exec_config.enable_validation = exec_config.enable_validation || graph_builder_->config().enable_validation;

        if (env.execution.execution_mode == "parallel")
        {
            exec_config.mode = ExecutionMode::PARALLEL;
        }
        else if (env.execution.execution_mode == "pipelined")
        {
            exec_config.mode = ExecutionMode::PIPELINED;
        }
        else
        {
            exec_config.mode = ExecutionMode::SEQUENTIAL;
        }

        executor_ = DeviceGraphExecutor(exec_config);
    }

    bool DeviceGraphOrchestrator::validateConfigurationForForward() const
    {
        bool valid = true;

        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] graph_builder_ is null");
            valid = false;
        }

        if (!arena_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] BufferArena not initialized "
                      "(call initializeInferenceStateFromArena() first)");
            valid = false;
        }

        if (!managed_buffers_.current_hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] No current_hidden buffer "
                      "(call initializeInferenceStateFromArena() first)");
            valid = false;
        }

        // Weight check: either graph builder has weights or weight_manager is set
        bool has_weights = (graph_builder_ && graph_builder_->isInitialized()) ||
                           weight_manager_ != nullptr;
        if (!has_weights)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] No weights loaded "
                      "(call setWeights() or setWeightManager())");
            valid = false;
        }

        return valid;
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    DeviceGraphOrchestrator::DeviceGraphOrchestrator(
        Dependencies deps)
        : graph_builder_(std::move(deps.graph_builder)),
          mpi_ctx_(nullptr), // No direct MPI context - use injected topology
          cache_config_(deps.cache_config),
          injected_model_ctx_(std::move(deps.model_ctx)),
          injected_topology_(std::move(deps.topology)),
          injected_collective_ctx_(std::move(deps.collective_ctx)),
          turboquant_ctx_(std::move(deps.turboquant_ctx)),
          kv_rotation_(std::move(deps.kv_rotation)),
          pp_stage_config_(std::move(deps.pp_stage_config)),
          pipeline_config_(std::move(deps.pipeline_config)),
          domain_tp_contexts_(std::move(deps.domain_tp_contexts)),
          weight_streamer_(std::move(deps.weight_streamer)),
          weight_manager_(std::move(deps.weight_manager)),
          weight_placement_map_(std::move(deps.weight_placement_map)),
          tp_config_(std::move(deps.tp_config)),
          domain_config_(std::move(deps.domain_config))
    {
        if (!injected_model_ctx_)
        {
            throw std::invalid_argument("DeviceGraphOrchestrator Dependencies requires a valid model_ctx");
        }

        if (!graph_builder_)
        {
            throw std::invalid_argument("DeviceGraphOrchestrator Dependencies requires a valid graph_builder");
        }

        configureExecutor();

        for (auto &[name, ctx] : domain_tp_contexts_)
        {
            if (ctx)
                graph_builder_->setTPContext(name, ctx.get());
        }
        if (!domain_tp_contexts_.empty())
            tp_contexts_initialized_ = true;

        // Propagate MPI rank to executor for stage dumping (from injected topology)
        if (injected_topology_)
        {
            executor_.setMPIRank(injected_topology_->rank());
        }

        // Wire CollectiveContext to executor for GPU-native collectives (RCCL/NCCL/HOST)
        if (injected_collective_ctx_)
        {
            executor_.setCollectiveContext(injected_collective_ctx_.get());
            LOG_DEBUG("[DeviceGraphOrchestrator] Wired CollectiveContext to DeviceGraphExecutor");
        }

        // Validate PP stage config if provided
        if (pp_stage_config_.has_value() && !pp_stage_config_->isValid())
        {
            throw std::invalid_argument("Invalid FactoryPPStageConfig in Dependencies: "
                                        "first_layer=" +
                                        std::to_string(pp_stage_config_->first_layer) +
                                        ", last_layer=" + std::to_string(pp_stage_config_->last_layer));
        }

        LOG_DEBUG("[DeviceGraphOrchestrator] Initialized with dependencies, caching="
                  << (cache_config_.enabled ? "enabled" : "disabled")
                  << ", topology=" << (injected_topology_ ? "provided" : "none")
                  << ", collective=" << (injected_collective_ctx_ ? "provided" : "none")
                  << ", turboquant=" << (turboquant_ctx_ ? "provided" : "none")
                  << ", pp_stage=" << (pp_stage_config_.has_value() ? "configured" : "none")
                  << ", pipeline=" << (pipeline_config_ ? "provided" : "none"));
    }

    DeviceGraphOrchestrator::DeviceGraphOrchestrator(
        std::shared_ptr<IGraphBuilder> graph_builder,
        std::shared_ptr<IMPIContext> mpi_ctx,
        const GraphCacheConfig &cache_config)
        : graph_builder_(std::move(graph_builder)),
          mpi_ctx_(std::move(mpi_ctx)),
          cache_config_(cache_config)
    {
        if (!graph_builder_)
        {
            throw std::invalid_argument("DeviceGraphOrchestrator requires a valid graph builder");
        }

        configureExecutor();

        // Propagate MPI rank to executor for stage dumping
        if (mpi_ctx_)
        {
            executor_.setMPIRank(mpi_ctx_->rank());
        }

        LOG_DEBUG("[DeviceGraphOrchestrator] Initialized with graph builder, caching="
                  << (cache_config_.enabled ? "enabled" : "disabled"));
    }

    DeviceGraphOrchestrator::~DeviceGraphOrchestrator() = default;
    DeviceGraphOrchestrator::DeviceGraphOrchestrator(DeviceGraphOrchestrator &&) noexcept = default;
    DeviceGraphOrchestrator &DeviceGraphOrchestrator::operator=(DeviceGraphOrchestrator &&) noexcept = default;

    // =========================================================================
    // Device Context Management
    // =========================================================================

    IDeviceContext *DeviceGraphOrchestrator::getDeviceContext(DeviceId device)
    {
        auto it = device_contexts_.find(device);
        if (it != device_contexts_.end())
        {
            return it->second.get();
        }

        // Create new context using DeviceId
        auto ctx = IDeviceContext::create(device);
        if (!ctx)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to create device context for device " << device.toString());
            return nullptr;
        }

        IDeviceContext *raw_ptr = ctx.get();
        device_contexts_[device] = std::move(ctx);

        LOG_DEBUG("[DeviceGraphOrchestrator] Created device context for device " << device.to_string());
        return raw_ptr;
    }

    bool DeviceGraphOrchestrator::isMoeRebalancingActive() const
    {
        if (!moe_rebalance_controller_)
            return false;
        return moe_rebalance_controller_->mode() == MoERebalanceMode::DYNAMIC;
    }

    uint64_t DeviceGraphOrchestrator::moePlacementEpoch() const
    {
        if (!moe_rebalance_controller_)
            return 0;
        return moe_rebalance_controller_->placementEpoch();
    }

    std::string DeviceGraphOrchestrator::prefillGraphDomainId() const
    {
        if (!moe_rebalance_controller_)
            return "single";
        return moe_rebalance_controller_->domainId();
    }

    int DeviceGraphOrchestrator::prefillGraphParticipantId() const
    {
        return moeRebalanceParticipantId();
    }

    PrefillChunkMaintenanceState DeviceGraphOrchestrator::prefillChunkMaintenanceState(
        const PrefillChunkPlan &chunk) const
    {
        PrefillChunkMaintenanceState state;
        state.chunk_index = chunk.chunk_index;
        state.histograms_merged = true;
        state.manual_boundaries_complete = true;
        state.graph_capture_active = isGraphCaptureActive();
        state.graph_replay_active = false;
        state.participants_at_same_boundary = true;

        if (!moe_rebalance_controller_)
            return state;

        if (auto *histogram = moe_rebalance_controller_->histogram())
        {
            state.histograms_merged = histogram->syncRuntimeHistograms();
        }

        state.rebalance_requested =
            state.histograms_merged && moe_rebalance_controller_->shouldRebalance();
        return state;
    }

    bool DeviceGraphOrchestrator::onPrefillChunkMaintenance(
        const PrefillChunkPlan &chunk,
        const PrefillChunkMaintenanceDecision &decision)
    {
        if (!decision.ok)
            return false;
        if (!decision.can_run || !moe_rebalance_controller_)
            return true;
        if (!moe_rebalance_controller_->shouldRebalance())
            return true;

        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            LOG_WARN("[DGO] Prefill chunk maintenance reached a multi-rank MoE rebalance "
                     "boundary at chunk "
                     << chunk.chunk_index
                     << "; deferring to rank/runner-level coordination");
            return !decision.required;
        }

        if (moe_rebalance_controller_->maxReplicasPerSocket() > 0 ||
            debugEnv().moe_rebalance.gpu_cache_experts_per_layer > 0)
        {
            LOG_WARN("[DGO] Prefill chunk maintenance reached a MoE replica/cache rebalance "
                     "boundary at chunk "
                     << chunk.chunk_index
                     << "; replica and GPU-cache placement require runner-level coordination");
            return !decision.required;
        }

        try
        {
            const auto old_placement = moe_rebalance_controller_->currentPlacement();
            const auto new_placement = moe_rebalance_controller_->rebalance();
            moe_rebalance_controller_->syncReplicaPlacement();

            if (new_placement.empty())
                return true;

            ReceivedWeightsMap received;
            auto manifest = ExpertWeightTransfer::buildManifest(old_placement, new_placement);
            if (!manifest.empty())
                received = transferExpertWeights(manifest, moe_rebalance_controller_->numLayers());

            const int participant_id = mpi_ctx_ ? mpi_ctx_->local_rank() : 0;
            auto masks = moe_rebalance_controller_->computeExpertMasksForParticipant(participant_id);
            applyExpertMasks(masks, received);

            if (forward_engine_)
                forward_engine_->discardAllCachedGraphs();
            mtp_sidecar_depth0_cache_.invalidate();
            mtp_sidecar_depth0_device_token_cache_.invalidate();
            mtp_sidecar_depth0_chained_cache_.invalidate();
            mtp_sidecar_depth0_chained_device_token_cache_.invalidate();
            mtp_sidecar_depth0_kv_only_cache_.invalidate();
            mtp_sidecar_depth0_kv_only_device_token_cache_.invalidate();
            for (auto &cache : mtp_sidecar_depth0_kv_only_batch_caches_)
                cache.invalidate();
            mtp_terminal_hidden_row_select_cache_.invalidate();
            mtp_terminal_hidden_rows_select_cache_.invalidate();
            for (auto &cache : layer_graph_cache_)
                cache.invalidate();
            resetKernelDynamicState();
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DGO] Prefill chunk MoE maintenance failed at chunk "
                      << chunk.chunk_index << ": " << e.what());
            return false;
        }

        return true;
    }

    // =========================================================================
    // Weight and Buffer Configuration
    // =========================================================================

    void DeviceGraphOrchestrator::setWeights(const ModelWeights &weights)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot set weights: graph builder not initialized");
            return;
        }

        // Phase 6: Pre-resolve all layer weights to freeze weight resolution.
        // After this, graph construction reads pre-resolved pointers rather than
        // calling getWeightForDevice() lazily during graph build.
        const auto &cfg = graph_builder_->config();
        const int n_layers = cfg.n_layers;
        if (n_layers > 0 && weights.get_layer_weights)
        {
            // For PP stages, graph builders use global layer indices
            // (pp_layer_offset .. pp_layer_offset + n_layers - 1).
            // For single-device, pp_layer_offset=0 and indices are 0..n_layers-1.
            const int first_layer = cfg.pp_layer_offset;
            const int last_layer = first_layer + n_layers;

            ModelWeights frozen_weights = weights;
            auto resolved = std::make_shared<std::unordered_map<int, LayerWeights>>();
            resolved->reserve(n_layers);
            for (int i = first_layer; i < last_layer; ++i)
                (*resolved)[i] = weights.get_layer_weights(i);

            frozen_weights.get_layer_weights = [resolved](int layer_idx) -> LayerWeights
            {
                auto it = resolved->find(layer_idx);
                if (it != resolved->end())
                    return it->second;
                return {};
            };

            // Phase 6 (continued): Build FrozenModelWeightSet for audit, validation,
            // and Phase 7 PreparedWeightStore integration.
            buildFrozenWeightSet(weights, *resolved, first_layer, last_layer);

            if (frozen_weight_set_)
            {
                auto weight_bindings = makeModelWeightBindings(*frozen_weight_set_);
                graph_builder_->setWeightBindings(weight_bindings);
                graph_builder_->setWeights(toLegacyModelWeights(weight_bindings));
            }
            else
            {
                graph_builder_->setWeights(frozen_weights);
            }

            LOG_DEBUG("[DeviceGraphOrchestrator] Model weights frozen for " << n_layers
                                                                            << " layers [" << first_layer << ", " << last_layer << ")");
        }
        else
        {
            graph_builder_->setWeights(weights);
            LOG_DEBUG("[DeviceGraphOrchestrator] Model weights configured for full forward pass");
        }
    }

    void DeviceGraphOrchestrator::setFrozenWeightSet(std::unique_ptr<FrozenModelWeightSet> weight_set)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot set frozen weights: graph builder not initialized");
            return;
        }
        if (!weight_set)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot set frozen weights: null weight set");
            return;
        }

        weight_set->validateForGraph();
        frozen_weight_set_ = std::move(weight_set);

        auto weight_bindings = makeModelWeightBindings(*frozen_weight_set_);
        graph_builder_->setWeightBindings(weight_bindings);
        graph_builder_->setWeights(toLegacyModelWeights(weight_bindings));

        LOG_DEBUG("[DeviceGraphOrchestrator] FrozenModelWeightSet configured directly with "
                  << frozen_weight_set_->bindings().size() << " bindings");
    }

    void DeviceGraphOrchestrator::buildFrozenWeightSet(
        const ModelWeights &weights,
        const std::unordered_map<int, LayerWeights> &resolved_layers,
        int first_layer, int last_layer)
    {
        // Determine strategy from current orchestrator state
        InferenceStrategy strategy;
        strategy.mode = WeightInferenceMode::SingleDevice;
        strategy.pp_stages = 1;
        strategy.tp_degree = 1;
        if (graph_builder_)
        {
            const auto &cfg = graph_builder_->config();
            strategy.devices.push_back(cfg.default_device);
        }

        // Build bindings from global weights + resolved layer weights
        ModelWeightSetBuilder builder(strategy);

        // Global weights
        auto addGlobal = [&](TensorBase *tensor, const std::string &name, WeightRole role)
        {
            if (!tensor)
                return;
            WeightIdentity id;
            id.canonical_name = name;
            id.role = role;
            id.logical_id = stableWeightLogicalId(name);
            WeightBinding binding;
            binding.identity = id;
            binding.tensor = tensor;
            binding.immutable = true;
            builder.addBinding(std::move(binding));
        };

        addGlobal(weights.embedding_table, "token_embd.weight", WeightRole::Embedding);
        addGlobal(weights.final_norm, "output_norm.weight", WeightRole::OutputNorm);
        addGlobal(weights.lm_head, "output.weight", WeightRole::LMHead);

        // Per-layer weights
        for (int layer_idx = first_layer; layer_idx < last_layer; ++layer_idx)
        {
            auto it = resolved_layers.find(layer_idx);
            if (it == resolved_layers.end())
                continue;
            const auto &lw = it->second;

            auto addLayer = [&](TensorBase *tensor, const std::string &suffix, WeightRole role)
            {
                if (!tensor)
                    return;
                std::string canonical = "blk." + std::to_string(layer_idx) + "." + suffix;
                WeightIdentity id;
                id.canonical_name = canonical;
                id.role = role;
                id.layer = layer_idx;
                id.logical_id = stableWeightLogicalId(canonical);
                WeightBinding binding;
                binding.identity = id;
                binding.tensor = tensor;
                binding.immutable = true;
                builder.addBinding(std::move(binding));
            };

            addLayer(lw.wq, "attn_q.weight", WeightRole::AttentionQ);
            addLayer(lw.wk, "attn_k.weight", WeightRole::AttentionK);
            addLayer(lw.wv, "attn_v.weight", WeightRole::AttentionV);
            addLayer(lw.wo, "attn_output.weight", WeightRole::AttentionWO);
            addLayer(lw.attn_norm, "attn_norm.weight", WeightRole::Norm);
            addLayer(lw.q_bias, "attn_q.bias", WeightRole::Bias);
            addLayer(lw.k_bias, "attn_k.bias", WeightRole::Bias);
            addLayer(lw.v_bias, "attn_v.bias", WeightRole::Bias);
            addLayer(lw.q_norm, "attn_q_norm.weight", WeightRole::Norm);
            addLayer(lw.k_norm, "attn_k_norm.weight", WeightRole::Norm);
            addLayer(lw.attn_qkv, "attn_qkv.weight", WeightRole::FusedQKV);
            addLayer(lw.attn_gate, "attn_gate.weight", WeightRole::GDNProjection);
            addLayer(lw.ssm_alpha, "ssm_alpha.weight", WeightRole::GDNSsmParam);
            addLayer(lw.ssm_beta, "ssm_beta.weight", WeightRole::GDNSsmParam);
            addLayer(lw.ssm_conv1d, "ssm_conv1d.weight", WeightRole::GDNSsmParam);
            addLayer(lw.ssm_dt_bias, "ssm_dt.bias", WeightRole::Bias);
            addLayer(lw.ssm_a, "ssm_a", WeightRole::GDNSsmParam);
            addLayer(lw.ssm_norm, "ssm_norm.weight", WeightRole::Norm);
            addLayer(lw.ssm_out, "ssm_out.weight", WeightRole::GDNProjection);
            addLayer(lw.gate_proj, "ffn_gate.weight", WeightRole::FFNGate);
            addLayer(lw.up_proj, "ffn_up.weight", WeightRole::FFNUp);
            addLayer(lw.down_proj, "ffn_down.weight", WeightRole::FFNDown);
            addLayer(lw.ffn_norm, "ffn_norm.weight", WeightRole::Norm);
            addLayer(lw.moe_gate, "ffn_gate_inp.weight", WeightRole::MoERouter);
            addLayer(lw.moe_gate_exps, "ffn_gate_exps.weight", WeightRole::MoEExpertGate);
            addLayer(lw.moe_up_exps, "ffn_up_exps.weight", WeightRole::MoEExpertUp);
            addLayer(lw.moe_down_exps, "ffn_down_exps.weight", WeightRole::MoEExpertDown);
            addLayer(lw.shared_expert_gate, "ffn_gate_shexp.weight", WeightRole::SharedExpertGate);
            addLayer(lw.shared_expert_up, "ffn_up_shexp.weight", WeightRole::SharedExpertUp);
            addLayer(lw.shared_expert_down, "ffn_down_shexp.weight", WeightRole::SharedExpertDown);
            addLayer(lw.shared_expert_gate_inp, "ffn_gate_inp_shexp.weight", WeightRole::SharedExpertGate);
        }

        frozen_weight_set_ = std::make_unique<FrozenModelWeightSet>(
            strategy, builder.freezeBindings());
        frozen_weight_set_->validateForGraph();
        LOG_DEBUG("[DeviceGraphOrchestrator] FrozenModelWeightSet built with "
                  << frozen_weight_set_->bindings().size() << " bindings");
    }

    void DeviceGraphOrchestrator::setBuffers(const ModelBuffers &buffers)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot set buffers: graph builder not initialized");
            return;
        }
        graph_builder_->setBuffers(buffers);
        managed_buffers_ = buffers;
        LOG_DEBUG("[DeviceGraphOrchestrator] Model buffers configured for full forward pass");
    }

    bool DeviceGraphOrchestrator::hasGlobalWeights() const
    {
        if (!graph_builder_)
        {
            return false;
        }
        // Check if the builder's isInitialized returns true AND we have global weights
        // isInitialized() checks get_layer_weights, but we also need embedding_table etc.
        // For now, rely on the graph builder's internal state
        return graph_builder_->isInitialized();
    }

    // =========================================================================
    // Graph Buffer Management (Phase 3 - moved from QwenStandardGraph)
    // =========================================================================

    bool DeviceGraphOrchestrator::initializeBuffers(int seq_len)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] initializeBuffers called but graph_builder not set");
            return false;
        }

        const auto &config = graph_builder_->config();
        if (!config.use_graph_buffer_management)
        {
            LOG_WARN("[DeviceGraphOrchestrator] initializeBuffers called but use_graph_buffer_management=false");
            return false;
        }

        LOG_DEBUG("[DeviceGraphOrchestrator] Initializing buffers with graph management, seq_len=" << seq_len);

        // Get schema and resolver config from graph builder
        GraphSchema schema = graph_builder_->getSchema();
        GraphResolverConfig resolver_config = graph_builder_->getResolverConfig(seq_len);

        // Verify TensorFactory is set
        if (!tensor_factory_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] TensorFactory not set. Call setTensorFactory() before initializeBuffers()");
            return false;
        }

        // =====================================================================
        // Configure ArenaConfig for BufferArena allocation
        // =====================================================================
        ArenaConfig arena_config;
        arena_config.factory = tensor_factory_;

        // Configure mapped memory for GPU + snapshot scenarios
        bool use_mapped = state_.device_id.is_gpu() && snapshot_enabled_;
        if (use_mapped)
        {
            arena_config.use_mapped_memory = true;
            LOG_DEBUG("[DeviceGraphOrchestrator] Enabling mapped memory for GPU + snapshot mode (zero-copy host access)");
        }

        // Create BufferArena with config
        arena_ = std::make_unique<BufferArena>(arena_config);

        // =====================================================================
        // Register layer buffers from schema
        // =====================================================================
        auto layer_reqs = BufferAllocator::resolveLayerBuffers(schema, resolver_config);
        for (const auto &desc : layer_reqs.buffers)
        {
            BufferId id = BufferArena::bufferNameToId(desc.name);
            if (id == BufferId::_COUNT)
            {
                // Check model-provided buffer mappings
                auto it = resolver_config.buffer_name_to_id.find(desc.name);
                if (it != resolver_config.buffer_name_to_id.end())
                {
                    id = it->second;
                }
                else
                {
                    LOG_WARN("[DeviceGraphOrchestrator] No BufferId mapping for layer buffer '" << desc.name << "', skipping");
                    continue;
                }
            }

            // Skip unused O(S²) attention workspace buffers.
            // All flash attention kernels (CPU, CUDA, ROCm) use tiled online
            // softmax and never read these buffers — they accept them as
            // optional parameters and (void)-cast them.  Skipping avoids
            // allocating seq_len² × n_heads × sizeof(float) bytes that would
            // otherwise dominate memory at long context lengths.
            if (id == BufferId::ATTN_SCORES_WORKSPACE || id == BufferId::ATTN_CONTEXT_WORKSPACE)
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] Skipping unused O(S²) buffer: " << bufferIdName(id));
                continue;
            }

            size_t rows = desc.shape.size() >= 1 ? desc.shape[0] : 0;
            size_t cols = desc.shape.size() >= 2 ? desc.shape[1] : 0;
            const char *dtype = BufferArena::bufferTensorTypeToStr(desc.tensor_type);
            LOG_DEBUG("[DeviceGraphOrchestrator] Registering layer buffer: '" << desc.name
                                                                              << "' → " << bufferIdName(id) << " [" << rows << "x" << cols << "] dtype=" << dtype);
            if (!arena_->registerBuffer(id, rows, cols, dtype, desc.device))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to register layer buffer: " << desc.name);
                return false;
            }
        }

        // =====================================================================
        // Register model-level buffers (current_hidden, logits)
        // =====================================================================
        auto model_reqs = BufferAllocator::resolveModelBuffers(schema, resolver_config);
        for (const auto &desc : model_reqs.buffers)
        {
            BufferId id = BufferArena::bufferNameToId(desc.name);
            if (id == BufferId::_COUNT)
            {
                // Check model-provided buffer mappings
                auto it = resolver_config.buffer_name_to_id.find(desc.name);
                if (it != resolver_config.buffer_name_to_id.end())
                {
                    id = it->second;
                }
                else
                {
                    LOG_WARN("[DeviceGraphOrchestrator] No BufferId mapping for model buffer '" << desc.name << "', skipping");
                    continue;
                }
            }
            size_t rows = desc.shape.size() >= 1 ? desc.shape[0] : 0;
            size_t cols = desc.shape.size() >= 2 ? desc.shape[1] : 0;
            const char *dtype = BufferArena::bufferTensorTypeToStr(desc.tensor_type);
            if (!arena_->registerBuffer(id, rows, cols, dtype, desc.device))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to register model buffer: " << desc.name);
                return false;
            }
        }

        // =====================================================================
        // Register argmax partial-reduction scratch (GPU greedy sampling)
        // =====================================================================
        // The two-pass GPU argmax (sampleGreedyOnDevice) needs a small device
        // scratch holding one (value, index) partial per pass-1 block. We allocate
        // it through the arena (the V2 central buffer manager) instead of doing a
        // lazy cudaMalloc inside the backend. Capacity bounds the pass-1 grid;
        // 1024 partials far exceeds the ~74 blocks a 152K vocab needs.
        if (state_.device_id.is_gpu())
        {
            if (config.vocab_size <= 0)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] GPU stochastic buffers require a positive vocab size");
                return false;
            }
            const size_t stochastic_vocab_cols =
                static_cast<size_t>(config.vocab_size);
            constexpr size_t kArgmaxPartialCapacity = 1024;
            if (!arena_->registerBuffer(BufferId::ARGMAX_PARTIAL_VALS, 1, kArgmaxPartialCapacity,
                                        "FP32", state_.device_id))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to register ARGMAX_PARTIAL_VALS buffer");
                return false;
            }
            if (!arena_->registerBuffer(BufferId::ARGMAX_PARTIAL_IDXS, 1, kArgmaxPartialCapacity,
                                        "INT32", state_.device_id))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to register ARGMAX_PARTIAL_IDXS buffer");
                return false;
            }
            if (!arena_->registerBuffer(BufferId::STOCHASTIC_TARGET_TOKEN_IDS,
                                        kStochasticTargetRows,
                                        kStochasticDistributionMaxK,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_TARGET_PROBS,
                                        kStochasticTargetRows,
                                        kStochasticDistributionMaxK,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_DRAFT_TOKEN_IDS,
                                        kStochasticDraftRows,
                                        kStochasticDistributionMaxK,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_DRAFT_PROBS,
                                        kStochasticDraftRows,
                                        kStochasticDistributionMaxK,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_PROCESSED_LOGITS,
                                        kStochasticTargetRows,
                                        stochastic_vocab_cols,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_INVERSE_REJECTION_SAMPLES,
                                        kStochasticDraftRows,
                                        stochastic_vocab_cols,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS,
                                        1,
                                        sampling_math::kSpeculativeBatchMaxOutputTokens,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS,
                                        1,
                                        sampling_math::kSpeculativeBatchMaxRows,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_DRAFT_SAMPLE_PROBS,
                                        1,
                                        sampling_math::kSpeculativeBatchMaxRows,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::MTP_CONDITION_TOKEN,
                                        1,
                                        sampling_math::kSpeculativeBatchMaxRows *
                                            kMTPSidecarConditionTokenSlotCount,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::MTP_VERIFIER_INPUT_TOKENS,
                                        1,
                                        sampling_math::kSpeculativeBatchMaxRows + 1,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_TOPK_PARTIAL_VALS,
                                        1,
                                        kStochasticTopKPartialCapacity,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_TOPK_PARTIAL_IDXS,
                                        1,
                                        kStochasticTopKPartialCapacity,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_VERIFY_TOKENS,
                                        1,
                                        kStochasticTargetRows,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_VERIFY_ACCEPTED,
                                        1,
                                        kStochasticTargetRows,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_VERIFY_ACCEPT_PROBS,
                                        1,
                                        kStochasticTargetRows,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_VERIFY_THRESHOLDS,
                                        1,
                                        kStochasticTargetRows,
                                        "FP32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_BATCH_OUTPUT_TOKENS,
                                        1,
                                        sampling_math::kSpeculativeBatchMaxOutputTokens,
                                        "INT32",
                                        state_.device_id) ||
                !arena_->registerBuffer(BufferId::STOCHASTIC_BATCH_OUTPUT_META,
                                        1,
                                        sampling_math::kSpeculativeBatchMetaCount,
                                        "INT32",
                                        state_.device_id))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to register stochastic MTP sampling buffers");
                return false;
            }
        }

        // =====================================================================
        // Allocate all registered buffers
        // =====================================================================
        logOrchestratorVramTrace(state_.device_id, "arena.before_allocate");
        if (!arena_->allocate())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] BufferArena allocation failed");
            return false;
        }
        logOrchestratorVramTrace(state_.device_id, "arena.after_allocate");

        // Resolve the argmax partial scratch device pointers once, up front, so
        // the per-decode-step greedy-sampling hot path never touches the arena.
        // prepareForWrite forces device-side allocation; the buffers are pure
        // scratch (overwritten every call) so no coherence tracking is needed.
        if (state_.device_id.is_gpu() &&
            arena_->isRegistered(BufferId::ARGMAX_PARTIAL_VALS) &&
            arena_->isRegistered(BufferId::ARGMAX_PARTIAL_IDXS))
        {
            arena_->prepareForWrite(BufferId::ARGMAX_PARTIAL_VALS, state_.device_id);
            arena_->prepareForWrite(BufferId::ARGMAX_PARTIAL_IDXS, state_.device_id);
            argmax_partial_vals_dev_ = arena_->getDevicePtr(BufferId::ARGMAX_PARTIAL_VALS, state_.device_id);
            argmax_partial_idxs_dev_ = arena_->getDevicePtr(BufferId::ARGMAX_PARTIAL_IDXS, state_.device_id);
            if (argmax_partial_vals_dev_ && argmax_partial_idxs_dev_)
                argmax_partial_capacity_ = static_cast<int>(arena_->getCols(BufferId::ARGMAX_PARTIAL_VALS));
        }

        if (state_.device_id.is_gpu() &&
            arena_->isRegistered(BufferId::STOCHASTIC_TARGET_TOKEN_IDS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_TARGET_PROBS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_DRAFT_TOKEN_IDS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_DRAFT_PROBS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_PROCESSED_LOGITS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_INVERSE_REJECTION_SAMPLES) &&
            arena_->isRegistered(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_DRAFT_SAMPLE_PROBS) &&
            arena_->isRegistered(BufferId::MTP_CONDITION_TOKEN) &&
            arena_->isRegistered(BufferId::MTP_VERIFIER_INPUT_TOKENS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_TOPK_PARTIAL_VALS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_TOPK_PARTIAL_IDXS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_VERIFY_TOKENS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_VERIFY_ACCEPTED) &&
            arena_->isRegistered(BufferId::STOCHASTIC_VERIFY_ACCEPT_PROBS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_VERIFY_THRESHOLDS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_BATCH_OUTPUT_TOKENS) &&
            arena_->isRegistered(BufferId::STOCHASTIC_BATCH_OUTPUT_META))
        {
            arena_->prepareForWrite(BufferId::STOCHASTIC_TARGET_TOKEN_IDS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_TARGET_PROBS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_DRAFT_TOKEN_IDS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_DRAFT_PROBS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_PROCESSED_LOGITS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_INVERSE_REJECTION_SAMPLES, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_DRAFT_SAMPLE_PROBS, state_.device_id);
            arena_->prepareForWrite(BufferId::MTP_CONDITION_TOKEN, state_.device_id);
            arena_->prepareForWrite(BufferId::MTP_VERIFIER_INPUT_TOKENS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_TOPK_PARTIAL_VALS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_TOPK_PARTIAL_IDXS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_VERIFY_TOKENS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_VERIFY_ACCEPTED, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_VERIFY_ACCEPT_PROBS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_VERIFY_THRESHOLDS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_BATCH_OUTPUT_TOKENS, state_.device_id);
            arena_->prepareForWrite(BufferId::STOCHASTIC_BATCH_OUTPUT_META, state_.device_id);

            stochastic_target_token_ids_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_TARGET_TOKEN_IDS, state_.device_id);
            stochastic_target_probs_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_TARGET_PROBS, state_.device_id);
            stochastic_draft_token_ids_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_DRAFT_TOKEN_IDS, state_.device_id);
            stochastic_draft_probs_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_DRAFT_PROBS, state_.device_id);
            stochastic_processed_logits_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_PROCESSED_LOGITS, state_.device_id);
            stochastic_inverse_rejection_samples_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_INVERSE_REJECTION_SAMPLES, state_.device_id);
            stochastic_target_sample_tokens_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_TARGET_SAMPLE_TOKENS, state_.device_id);
            stochastic_draft_sample_tokens_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_DRAFT_SAMPLE_TOKENS, state_.device_id);
            stochastic_draft_sample_probs_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_DRAFT_SAMPLE_PROBS, state_.device_id);
            mtp_sidecar_condition_token_dev_ =
                arena_->getDevicePtr(BufferId::MTP_CONDITION_TOKEN, state_.device_id);
            mtp_sidecar_condition_token_capacity_ =
                static_cast<int>(arena_->getCols(BufferId::MTP_CONDITION_TOKEN));
            mtp_verifier_input_tokens_dev_ =
                arena_->getDevicePtr(BufferId::MTP_VERIFIER_INPUT_TOKENS, state_.device_id);
            stochastic_topk_partial_vals_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_TOPK_PARTIAL_VALS, state_.device_id);
            stochastic_topk_partial_idxs_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_TOPK_PARTIAL_IDXS, state_.device_id);
            if (stochastic_topk_partial_vals_dev_ && stochastic_topk_partial_idxs_dev_)
                stochastic_topk_partial_capacity_ =
                    static_cast<int>(arena_->getCols(BufferId::STOCHASTIC_TOPK_PARTIAL_VALS));
            stochastic_verify_tokens_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_VERIFY_TOKENS, state_.device_id);
            stochastic_verify_accepted_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_VERIFY_ACCEPTED, state_.device_id);
            stochastic_verify_accept_probs_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_VERIFY_ACCEPT_PROBS, state_.device_id);
            stochastic_verify_thresholds_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_VERIFY_THRESHOLDS, state_.device_id);
            stochastic_batch_output_tokens_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_BATCH_OUTPUT_TOKENS, state_.device_id);
            stochastic_batch_output_meta_dev_ =
                arena_->getDevicePtr(BufferId::STOCHASTIC_BATCH_OUTPUT_META, state_.device_id);
        }

        // Wire arena directly to graph builder (replaces bindArenaToManagedBuffers + setBuffers shim)
        graph_builder_->setArena(arena_.get());

        // Wire arena to executor for contract-based coherence
        executor_.setArena(arena_.get());

        // Log per-buffer allocation details
        arena_->logAllocationSummary();

        // Log theoretical aliasing savings
        auto [original, optimized] = BufferAllocator::estimateMemorySavings(schema, resolver_config);
        double savings = (original > 0) ? 100.0 * (original - optimized) / original : 0.0;
        LOG_DEBUG("[DeviceGraphOrchestrator] Theoretical aliasing savings: "
                  << (original / 1024.0) << " KB -> " << (optimized / 1024.0) << " KB"
                  << " (" << savings << "% reduction)");

        return true;
    }

    bool DeviceGraphOrchestrator::initializeMTPKVCaches(
        int batch_size,
        int max_seq_len,
        ActivationPrecision kv_cache_prec,
        KVCacheLayoutMode kv_layout_mode,
        DeviceId device,
        const std::shared_ptr<IMPIContext> &local_mpi_ctx,
        bool use_sharded_cache,
        bool has_tp,
        bool is_global_tp)
    {
        const auto &config = graph_builder_->config();
        state_.mtp_kv_caches.clear();

        // Qwen3.5/Qwen3.6 support is currently D=1. Additional depths need
        // distinct sidecar weights before they can have independent caches.
        constexpr int kSupportedMTPDepth = 1;
        state_.mtp_kv_caches.reserve(kSupportedMTPDepth);

        for (int depth = 0; depth < kSupportedMTPDepth; ++depth)
        {
            llaminar::v2::kernels::KVCacheConfig kv_config;
            kv_config.precision = kv_cache_prec;
            kv_config.device = device;
            kv_config.num_layers = 1;
            kv_config.first_layer_index = 0;
            kv_config.batch_size = batch_size;
            kv_config.max_seq_len = max_seq_len;
            kv_config.n_kv_heads = config.n_kv_heads;
            kv_config.head_dim = config.head_dim;
            kv_config.layout_mode = kv_layout_mode;
            kv_config.mpi_ctx = local_mpi_ctx.get();
            kv_config.turboquant_ctx = config.turboquant_ctx;

            if (use_sharded_cache && (has_tp || is_global_tp))
            {
                kv_config.local_n_kv_heads = config.local_n_kv_heads;
                kv_config.kv_head_start = has_tp
                                               ? config.tp_device_idx * config.local_n_kv_heads
                                               : mpi_ctx_->rank() * config.local_n_kv_heads;
            }

            auto cache = llaminar::v2::kernels::KernelFactory::createKVCache(kv_config);
            if (!cache)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create MTP KV cache depth " << depth);
                return false;
            }
            state_.mtp_kv_caches.push_back(std::move(cache));
        }

        LOG_DEBUG("[DeviceGraphOrchestrator] Created " << state_.mtp_kv_caches.size()
                                                       << " request-local MTP KV cache(s)");
        return true;
    }

    void DeviceGraphOrchestrator::releaseBuffers()
    {
        if (arena_)
        {
            arena_.reset();
            LOG_DEBUG("[DeviceGraphOrchestrator] BufferArena released");
        }

        owned_buffers_.clear();

        // Clear buffer pointers
        managed_buffers_ = ModelBuffers{};
    }

    ActivationBuffers &DeviceGraphOrchestrator::getInternalBuffers()
    {
        return managed_buffers_.layer_buffers;
    }

    const ActivationBuffers &DeviceGraphOrchestrator::getInternalBuffers() const
    {
        return managed_buffers_.layer_buffers;
    }

    const ModelBuffers &DeviceGraphOrchestrator::getModelBuffers() const
    {
        return managed_buffers_;
    }

    const ArenaAllocationStats *DeviceGraphOrchestrator::bufferStats() const
    {
        if (!arena_)
        {
            return nullptr;
        }
        return &arena_->stats();
    }

    // NOTE: Legacy initializeArena() was removed. The arena is now exclusively
    // created and populated by the schema-driven initializeBuffers() path.
    // The legacy path only registered ~15 standard buffers and missed
    // model-specific ones (GDN_QKV, FA_GATE, etc.), causing crashes for
    // architectures like Qwen3.5.

    // =========================================================================
    // Execution Methods
    // =========================================================================

    bool DeviceGraphOrchestrator::executeForward(
        const ForwardInput &input,
        ForwardOutput &output)
    {
        // Enable device-scoped logging if not already set by caller (e.g., from forward())
        ScopedDeviceLog device_log(input.device);

        // Validate that all required configuration has been set
        if (!validateConfigurationForForward())
        {
            return false;
        }

        // Token input OR activation input is required
        bool has_token_input = input.token_ids || input.token_ids_device || input.batches;
        bool has_activation_input = external_hidden_state_input_ != nullptr;

        // For PP stages without embedding, activation input is required instead of tokens
        bool is_pp_middle_stage = pp_stage_config_.has_value() &&
                                  !pp_stage_config_.value().has_embedding;

        if (!has_token_input && !has_activation_input)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] No token or activation input provided");
            return false;
        }

        if (is_pp_middle_stage && !has_activation_input)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] PP middle stage requires activation input "
                      "via setHiddenState()");
            return false;
        }

        if (input.seq_len <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid sequence length: " << input.seq_len);
            return false;
        }

        LOG_TRACE("[DeviceGraphOrchestrator] executeForward: batch_size=" << input.batch_size
                                                                          << ", seq_len=" << input.seq_len
                                                                          << ", device=" << input.device);

        // Build position IDs if not provided externally
        std::vector<int> position_ids_storage;
        ForwardInput effective_input = input;

        if (!input.position_ids)
        {
            position_ids_storage = IGraphBuilder::buildPositionIds(
                input.seq_len, input.batch_size, input.position_offset);
            effective_input.position_ids = position_ids_storage.data();
        }

        // Pass external hidden state to graph builder input for PP middle/final stages
        if (external_hidden_state_input_)
        {
            effective_input.external_hidden_state = external_hidden_state_input_;
            LOG_DEBUG("[DeviceGraphOrchestrator] Using external hidden state input: "
                      << external_hidden_state_input_->numel() << " elements");
            external_hidden_state_input_ = nullptr; // single-use semantics
        }

        // Ensure engine is initialized with current config
        ensureForwardEngine();

        // Delegate to ForwardExecutionEngine
        return forward_engine_->execute(effective_input, output, *this);
    }

    // =====================================================================
    // IForwardExecutionHost interface implementations
    // =====================================================================

    void DeviceGraphOrchestrator::ensureForwardEngine()
    {
        if (forward_engine_)
            return;

        ForwardExecutionEngine::Config engine_config;
        engine_config.cache_config = cache_config_;
        engine_config.pp_stage_config = pp_stage_config_;
        engine_config.has_unified_pp =
            pipeline_config_ && pipeline_config_->hasPP();

        forward_engine_ = std::make_unique<ForwardExecutionEngine>(
            std::move(engine_config), executor_);

        // Forward current timeline flags
        forward_engine_->setSuppressTimeline(suppress_timeline_);
        forward_engine_->setAccumulatePrefill(accumulate_prefill_);
    }

    GraphBuildResult DeviceGraphOrchestrator::buildForwardGraph(
        const ForwardInput &input)
    {
        auto session = buildGraph().forInput(input);

        auto finalize = [&](GraphBuildResult result) -> GraphBuildResult
        {
            if (!result || raw_expert_weights_released_after_graph_build_ || !graph_builder_)
                return result;

            const auto &cfg = graph_builder_->config();
            const bool release_enabled = cfg.moe.rebalance_config.release_raw_expert_weights ||
                                         debugEnv().moe_rebalance.release_raw_weights;
            if (!release_enabled || cfg.default_device.is_gpu())
                return result;

            size_t total_freed = 0;
            int stage_count = 0;
            PreparedWeightStore *summary_store = prepared_weight_store_.get();
            for (const auto &node_name : result.graph().getExecutionOrder())
            {
                ComputeNode *node = result.graph().getNode(node_name);
                if (!node || !node->stage || node->stage->type() != ComputeStageType::MOE_EXPERT_FFN)
                    continue;
                auto *moe = dynamic_cast<MoEExpertComputeStage *>(node->stage.get());
                if (!moe)
                    continue;
                if (!summary_store)
                    summary_store = moe->buildWeightContext().prepared_store;
                total_freed += moe->releaseRawExpertWeights();
                ++stage_count;
            }

            raw_expert_weights_released_after_graph_build_ = true;

            size_t cached_freed = 0;
            if (auto concrete_weight_manager = std::dynamic_pointer_cast<WeightManager>(weight_manager_))
            {
                cached_freed = concrete_weight_manager->releaseMoEExpertHostWeightData();
                concrete_weight_manager->logHostMemorySummary("after eager graph build raw release");
            }

            if (summary_store)
            {
                LOG_DEBUG("[DGO] Released raw expert weights after eager graph build across "
                          << stage_count << " MoE stages: " << (total_freed >> 20)
                          << " MB stage-owned + " << (cached_freed >> 20)
                          << " MB WeightManager-cached freed");
            }
            return result;
        };

        if (pipeline_config_ && pipeline_config_->hasPP())
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Building UNIFIED PIPELINE graph: "
                      << pipeline_config_->numStages() << " PP stages, "
                      << pipeline_config_->total_layers << " layers");

            if (!pp_contexts_initialized_ && !initializePPContexts())
                return GraphBuildResult("Failed to initialize PP contexts");

            if (!tp_contexts_initialized_ && !initializeTPContexts())
                return GraphBuildResult("Failed to initialize TP contexts");

            for (const auto &[key, ctx] : pp_contexts_)
                session.withPPContext(key.first, key.second, ctx.get());

            for (const auto &[name, ctx] : domain_tp_contexts_)
                session.withTPContext(name, ctx.get());

            return finalize(session
                                .withPipelineConfig(pipeline_config_)
                                .buildUnified());
        }
        else if (pp_stage_config_.has_value())
        {
            const auto &pp = pp_stage_config_.value();
            LOG_DEBUG("[DeviceGraphOrchestrator] Building PARTIAL forward graph: "
                      << "layers=[" << pp.first_layer << ", " << pp.last_layer << ") "
                      << "has_embedding=" << pp.has_embedding
                      << " has_lm_head=" << pp.has_lm_head);

            return finalize(session
                                .forPPStage(pp.first_layer, pp.last_layer,
                                            pp.has_embedding, pp.has_lm_head)
                                .buildPartial());
        }
        else
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Building FULL forward graph...");
            return finalize(session.buildForward());
        }
    }

    std::unordered_map<DeviceId, IDeviceContext *>
    DeviceGraphOrchestrator::getPipelineDeviceContexts()
    {
        std::unordered_map<DeviceId, IDeviceContext *> contexts;
        if (!pipeline_config_)
            return contexts;

        for (const auto &device : pipeline_config_->getAllDevices())
        {
            IDeviceContext *ctx = getDeviceContext(device);
            if (!ctx)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to get device context for "
                          << device);
                return {};
            }
            contexts[device] = ctx;
        }
        return contexts;
    }

    TensorBase *DeviceGraphOrchestrator::logitsTensor()
    {
        return state_.logits.get();
    }

    IForwardExecutionHost::PPCopyInfo
    DeviceGraphOrchestrator::resolvePPCopyInfo(
        const ForwardInput &input) const
    {
        PPCopyInfo info;
        if (!input.external_hidden_state || !graph_builder_)
            return info;

        const auto &cfg = graph_builder_->config();
        const auto &bufs = graph_builder_->buffers();
        InferenceMode mode(cfg.activation_precision);

        TensorBase *working_buffer =
            bufs.layer_buffers.residual && mode.isHybridQ16()
                ? bufs.layer_buffers.residual
                : bufs.current_hidden;

        if (!working_buffer ||
            input.external_hidden_state == working_buffer)
            return info;

        size_t copy_elems = static_cast<size_t>(
            input.batch_size * input.seq_len * cfg.d_model);

        if (mode.isHybridQ16())
        {
            size_t num_blocks = (copy_elems + 31) / 32;
            info.copy_bytes = num_blocks * sizeof(Q16_1Block);
        }
        else
        {
            info.copy_bytes = copy_elems * sizeof(float);
        }

        info.external_hidden = input.external_hidden_state;
        info.working_buffer = working_buffer;
        info.device = cfg.default_device;
        info.needs_copy = true;

        LOG_DEBUG("[DeviceGraphOrchestrator] Resolved PP copy info: "
                  << info.copy_bytes << " bytes on "
                  << cfg.default_device.toString());
        return info;
    }

    void DeviceGraphOrchestrator::syncLogitsAtBoundary(IDeviceContext *ctx)
    {
        if (!ctx || !ctx->isGPU())
        {
            return; // CPU execution is synchronous — nothing to sync
        }

        // Single stream sync ensures ALL GPU work (all stages) is complete.
        // This replaces the lazy per-tensor hipEventSynchronize that would
        // otherwise fire inside ensureOnHost() when logits->data() is called.
        ctx->synchronize();

        // Clear the mapped sync flag so data()/fp32_data() return the mapped
        // pointer immediately without any further synchronization.
        if (state_.logits && state_.logits->isMapped())
        {
            state_.logits->markMappedSynced();
        }
    }

    DeviceGraphExecutor::DecodeCapturePolicy DeviceGraphOrchestrator::buildDecodeCapturePolicy(
        bool has_collective_nodes,
        IDeviceContext *ctx,
        int segment_consecutive_failures) const
    {
        DeviceGraphExecutor::DecodeCapturePolicy policy;

        const auto &env = debugEnv();
        policy.allow_fast_decode =
            env.execution.fast_decode &&
            !executor_.config().snapshot_callback;

        if (!policy.allow_fast_decode)
        {
            return policy;
        }

        const bool allow_collective_segmented = env.execution.gpu_graph_collective_segmented;
        bool collective_segmented_backend_supported = true;
        if (has_collective_nodes && allow_collective_segmented)
        {
            collective_segmented_backend_supported = collectivesSupportSegmentedReplay();
        }

        policy.collective_segmented_enabled =
            has_collective_nodes &&
            allow_collective_segmented &&
            collective_segmented_backend_supported;

        // Capturing TP collectives directly into HIP/CUDA graphs is still an
        // experimental Tier-2 collective-capture project. By default, graphs
        // that contain collectives may use segmented replay only when the
        // explicit segmented-collective switch is enabled; the collective
        // stages themselves stay manual synchronization points.
        policy.collectives_graph_capturable = false;

        const bool can_use_segmented_graph =
            !has_collective_nodes ||
            policy.collective_segmented_enabled;

        // When profiling is enabled (LLAMINAR_PROFILING=1), disable GPU graph
        // capture/replay so decode runs through executeFastDecode(). This ensures
        // StageTimeline GPU events are recorded for every stage on every iteration,
        // giving accurate per-stage-type GPU timing. Without this, segmented replay
        // runs hipGraphLaunch() which bypasses per-stage event recording, causing
        // the accumulated timeline to report ~0 GPU time for Phase 3 iterations.
        policy.allow_segmented_capture =
            env.execution.gpu_graphs &&
            !env.execution.executor_profiling &&
            ctx && ctx->isGPU() &&
            can_use_segmented_graph &&
            segment_consecutive_failures < DeviceGraphExecutor::GraphSegmentCache::kMaxFailures;

        policy.max_segment_failures = DeviceGraphExecutor::GraphSegmentCache::kMaxFailures;
        return policy;
    }

    bool DeviceGraphOrchestrator::collectivesSupportSegmentedReplay() const
    {
        const auto &graph_cfg = graph_builder_->config();
        const bool has_local_tp = graph_cfg.tp_ctx && graph_cfg.tp_ctx->isLocal() && graph_cfg.tp_ctx->degree() > 1;

        // For Local TP (multi-GPU, single MPI rank), the per-device
        // DeviceGraphOrchestrator may not have an injected_collective_ctx_
        // because the RankOrchestrator owns the LocalTPContext.
        // The collective stages (TPAllreduceStage) execute as manual segments
        // between graph-captured compute segments, so segmented replay is safe
        // as long as the backend supports stream-ordered collectives.
        const bool single_rank_collectives =
            (injected_collective_ctx_ && injected_collective_ctx_->worldSize() == 1) ||
            has_local_tp; // Local TP implies single-rank collectives

        if (!(has_local_tp && single_rank_collectives))
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Disabling collective segmented GPU-graph replay for cross-rank or non-local-TP collectives");
            return false;
        }

        const auto backend = graph_cfg.tp_ctx->backend();
        const bool supported =
            (backend == CollectiveBackendType::NCCL ||
             backend == CollectiveBackendType::RCCL);

        if (!supported)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Disabling collective segmented GPU-graph replay for non-stream backend");
        }

        return supported;
    }

    bool DeviceGraphOrchestrator::execute(ComputeGraph &graph, IDeviceContext *ctx)
    {
        // Ensure declared CPU/GPU workspace is allocated for kernels and stages.
        if (!ensureDeviceWorkspaceAllocated(graph))
            return false;
        return executor_.execute(graph, ctx);
    }

    bool DeviceGraphOrchestrator::ensureDeviceWorkspaceAllocated(
        const ComputeGraph &graph,
        int workspace_seq_len)
    {
        const auto &config = graph_builder_->config();
        if (!workspace_allocator_)
        {
            workspace_allocator_ = std::make_unique<WorkspaceAllocator>();
        }

        WorkspaceSizingHints hints;
        const int default_workspace_seq_len = config.max_seq_len > 0 ? config.max_seq_len : 4096;

        // Bucketed prefill asks for the active bucket length so workspace can
        // grow with observed shapes instead of reserving the full configured
        // context up front. ForwardExecutionEngine tracks the allocator
        // generation and invalidates captured replay state after any grow.
        hints.max_seq_len = workspace_seq_len > 0 ? workspace_seq_len : default_workspace_seq_len;
        hints.n_heads = config.n_heads > 0 ? config.n_heads : 128;
        hints.head_dim = config.d_model > 0 && config.n_heads > 0
                             ? config.d_model / config.n_heads
                             : 128;
        hints.d_model = config.d_model > 0 ? config.d_model : 896;
        hints.batch_size = state_.batch_size > 0 ? state_.batch_size : 1;
        hints.vocab_size = config.vocab_size > 0 ? config.vocab_size : 151936;

        std::vector<WorkspaceConsumerRequest> extras;
        if (state_.kv_cache)
        {
            auto *kv_consumer = dynamic_cast<IWorkspaceConsumer *>(state_.kv_cache.get());
            if (kv_consumer && state_.device_id.is_gpu())
            {
                extras.push_back(WorkspaceConsumerRequest{
                    kv_consumer,
                    state_.device_id,
                    std::max(1, hints.max_seq_len),
                    std::max(1, hints.batch_size),
                    0,
                });
            }
        }
        for (auto &mtp_kv_cache : state_.mtp_kv_caches)
        {
            if (!mtp_kv_cache)
                continue;

            auto *kv_consumer = dynamic_cast<IWorkspaceConsumer *>(mtp_kv_cache.get());
            if (kv_consumer && state_.device_id.is_gpu())
            {
                extras.push_back(WorkspaceConsumerRequest{
                    kv_consumer,
                    state_.device_id,
                    std::max(1, hints.max_seq_len),
                    std::max(1, hints.batch_size),
                    0,
                });
            }
        }
        const bool mtp_metadata_workspace_active =
            compute_row_indexed_all_position_logits_ ||
            pending_mtp_spec_verifier_input_plan_.has_value() ||
            mtp_spec_decode_metadata_binding_.getWorkspace() != nullptr;
        if (state_.device_id.is_gpu() && mtp_metadata_workspace_active)
        {
            // This is runner metadata, not stage scratch. The row-select stage
            // reads these stable device pointers during graph replay, and
            // OrchestrationRunner uploads new values for each verifier step.
            //
            // Keep the binding in the consumer set after it has ever been
            // bound. Otherwise a later graph rebuild without a pending verifier
            // plan can destroy the DeviceWorkspaceManager without unbinding
            // this persistent runner-owned consumer, leaving a dangling manager
            // pointer that the next setShape() would refresh through.
            const auto metadata_shape = mtp_spec_decode_metadata_binding_.shape();
            extras.push_back(WorkspaceConsumerRequest{
                &mtp_spec_decode_metadata_binding_,
                state_.device_id,
                std::max(1, metadata_shape.max_requests),
                std::max(3, metadata_shape.max_draft_tokens),
                /*k=*/0,
            });
        }

        WorkspaceBudgetConfig workspace_budget;
        return workspace_allocator_->allocateForGraph(graph, hints, extras, workspace_budget);
    }

    uint64_t DeviceGraphOrchestrator::workspaceGeneration(DeviceId device) const
    {
        return workspace_allocator_ ? workspace_allocator_->deviceGeneration(device) : 0;
    }

    void DeviceGraphOrchestrator::onFirstGraphReady()
    {
        // Intentionally no-op for mmap reclaim. This callback fires when this
        // participant-local graph has a workspace, which is too early for TP/MoE
        // domains whose sibling participants may still be resolving captured
        // transfers. Reclaim happens after the first successful prefill instead.
    }

    void DeviceGraphOrchestrator::adviseMmapDontneedAfterFirstPrefill()
    {
        if (mmap_dontneed_advised_ || !weight_manager_)
            return;

        mmap_dontneed_advised_ = true;
        if (!synchronizeGpuBackendsBeforeMmapRelease())
        {
            LOG_WARN("[DeviceGraphOrchestrator] Skipping mmap DONTNEED after prefill because GPU synchronization failed");
            return;
        }
        if (debugEnv().vram_trace)
            LOG_INFO("[VRAM_TRACE] mmap_release.before_advise phase=after_first_prefill");
        const size_t advised_bytes = weight_manager_->adviseMmapDontneed();
        if (debugEnv().vram_trace)
            LOG_INFO("[VRAM_TRACE] mmap_release.after_advise phase=after_first_prefill bytes=" << advised_bytes);
    }

    bool DeviceGraphOrchestrator::executeAttention(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device)
    {
        // Debug: dump input to attention (for layer 0 only)
        if (layer_idx == 0)
        {
            const float *input = buffers.current_hidden->fp32_data();
            LOG_TRACE("[ORCH_ATTN_INPUT] layer=" << layer_idx << " seq_len=" << seq_len
                                                 << " input[0:4]=" << input[0] << "," << input[1]
                                                 << "," << input[2] << "," << input[3]);
        }

        // Get device context
        IDeviceContext *ctx = getDeviceContext(device);
        if (!ctx)
        {
            return false;
        }

        int pos_offset = position_ids ? position_ids[0] : 0;

        // =============================================================================
        // Graph Caching for Decode Mode (Phase 10)
        // =============================================================================
        if (cache_config_.enabled && cache_config_.cache_attention &&
            seq_len <= std::max(1, cache_config_.decode_seq_len) &&
            layer_idx >= 0 && static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            auto &cache = layer_graph_cache_[layer_idx];

            // Check if we have a valid cached graph
            if (cache.attention_decode &&
                cache.attention_cached_seq_len == seq_len &&
                cache.attention_cached_all_position_logits == compute_all_position_logits_ &&
                cache.valid)
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] Reusing cached attention graph for layer "
                          << layer_idx << " (pos_offset=" << pos_offset << ")");

                // Update dynamic parameters (position offset)
                updateCachedGraphParams(*cache.attention_decode, pos_offset, seq_len);

                // Execute cached graph
                bool success = executor_.execute(*cache.attention_decode, ctx);
                if (!success)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Cached attention graph failed at layer " << layer_idx);
                }

                cache.attention_decode->reset();
                cache_stats_.attention_cache_hits++;
                return success;
            }

            // Build and cache the graph using fluent API
            LOG_DEBUG("[DeviceGraphOrchestrator] Building and caching attention graph for layer "
                      << layer_idx << " (decode mode)");

            auto result = buildAttentionGraph()
                              .forLayer(layer, layer_idx)
                              .withBuffers(buffers)
                              .withSequence(seq_len, 1)
                              .onDevice(device)
                              .withKVCache(kv_cache)
                              .withPositionIds(position_ids)
                              .build();

            if (!result)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Attention graph build failed: " << result.error());
                return false;
            }

            cache.attention_decode = std::make_unique<ComputeGraph>(result.takeGraph());
            cache.cached_seq_len = seq_len;
            cache.attention_cached_seq_len = seq_len;
            cache.attention_cached_all_position_logits = compute_all_position_logits_;
            cache.valid = true;
            cache_stats_.attention_cache_misses++;

            // Execute the newly built graph
            ensureDeviceWorkspaceAllocated(*cache.attention_decode);
            bool success = executor_.execute(*cache.attention_decode, ctx);
            if (!success)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Attention block failed at layer " << layer_idx);
            }

            cache.attention_decode->reset();
            return success;
        }

        // =============================================================================
        // Non-cached path (prefill or caching disabled) using fluent API
        // =============================================================================
        cache_stats_.attention_cache_misses++;

        auto result = buildAttentionGraph()
                          .forLayer(layer, layer_idx)
                          .withBuffers(buffers)
                          .withSequence(seq_len, 1)
                          .onDevice(device)
                          .withKVCache(kv_cache)
                          .withPositionIds(position_ids)
                          .build();

        if (!result)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Attention graph build failed: " << result.error());
            return false;
        }

        ComputeGraph graph = result.takeGraph();

        // Debug: log graph structure
        if (layer_idx == 0)
        {
            auto order = graph.getExecutionOrder();
            LOG_TRACE("[ORCH_ATTN] Graph has " << graph.size() << " nodes, execution order:");
            for (const auto &name : order)
            {
                LOG_TRACE("[ORCH_ATTN]   - " << name);
            }
        }

        ensureDeviceWorkspaceAllocated(graph);
        bool success = executor_.execute(graph, ctx);

        // Debug: dump intermediate buffers (for layer 0 only)
        if (layer_idx == 0 && buffers.normalized && buffers.Q &&
            buffers.attn_output && buffers.attn_proj)
        {
            const float *output = buffers.current_hidden->fp32_data();
            LOG_TRACE("[ORCH_ATTN_OUTPUT] layer=" << layer_idx << " seq_len=" << seq_len
                                                  << " output[0:4]=" << output[0] << "," << output[1]
                                                  << "," << output[2] << "," << output[3]);
        }

        if (!success)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Attention block failed at layer " << layer_idx);
        }

        return success;
    }

    bool DeviceGraphOrchestrator::executeFFN(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        DeviceId device)
    {
        // Get device context
        IDeviceContext *ctx = getDeviceContext(device);
        if (!ctx)
        {
            return false;
        }

        // =============================================================================
        // Graph Caching for Decode Mode (Phase 10)
        // =============================================================================
        if (cache_config_.enabled && cache_config_.cache_ffn &&
            seq_len <= std::max(1, cache_config_.decode_seq_len) &&
            layer_idx >= 0 && static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            auto &cache = layer_graph_cache_[layer_idx];

            // Check if we have a valid cached FFN graph
            if (cache.ffn_decode &&
                cache.ffn_cached_seq_len == seq_len &&
                cache.ffn_cached_all_position_logits == compute_all_position_logits_ &&
                cache.valid)
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] Reusing cached FFN graph for layer " << layer_idx);

                // Execute cached graph (no params to update for FFN)
                bool success = executor_.execute(*cache.ffn_decode, ctx);
                if (!success)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Cached FFN graph failed at layer " << layer_idx);
                }

                cache.ffn_decode->reset();
                cache_stats_.ffn_cache_hits++;
                return success;
            }

            // Build and cache the graph using fluent API
            LOG_DEBUG("[DeviceGraphOrchestrator] Building and caching FFN graph for layer "
                      << layer_idx << " (decode mode)");

            auto result = buildFFNGraph()
                              .forLayer(layer, layer_idx)
                              .withBuffers(buffers)
                              .withSequence(seq_len, 1)
                              .onDevice(device)
                              .build();

            if (!result)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] FFN graph build failed: " << result.error());
                return false;
            }

            cache.ffn_decode = std::make_unique<ComputeGraph>(result.takeGraph());
            cache.cached_seq_len = seq_len;
            cache.ffn_cached_seq_len = seq_len;
            cache.ffn_cached_all_position_logits = compute_all_position_logits_;
            cache.valid = true;
            cache_stats_.ffn_cache_misses++;

            // Execute the newly built graph
            ensureDeviceWorkspaceAllocated(*cache.ffn_decode);
            bool success = executor_.execute(*cache.ffn_decode, ctx);
            if (!success)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] FFN block failed at layer " << layer_idx);
            }

            cache.ffn_decode->reset();
            return success;
        }

        // =============================================================================
        // Non-cached path (prefill or caching disabled) using fluent API
        // =============================================================================
        cache_stats_.ffn_cache_misses++;

        auto result = buildFFNGraph()
                          .forLayer(layer, layer_idx)
                          .withBuffers(buffers)
                          .withSequence(seq_len, 1)
                          .onDevice(device)
                          .build();

        if (!result)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] FFN graph build failed: " << result.error());
            return false;
        }

        ComputeGraph graph = result.takeGraph();

        ensureDeviceWorkspaceAllocated(graph);
        bool success = executor_.execute(graph, ctx);

        if (!success)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] FFN block failed at layer " << layer_idx);
        }

        return success;
    }

    bool DeviceGraphOrchestrator::executeLayer(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device)
    {
        LOG_TRACE("[DeviceGraphOrchestrator::executeLayer] LAYER_EXEC_ENTERED layer_idx="
                  << layer_idx << " seq_len=" << seq_len);

        // =====================================================================
        // Weight Streaming Hooks (Option B)
        // =====================================================================
        // Before layer execution: Ensure weights are on device
        // After layer execution: Release layer and prefetch next
        // =====================================================================
        if (weight_streamer_)
        {
            // Ensure this layer's weights are on the target device
            if (!weight_streamer_->ensureLayerOnDevice(layer_idx, device))
            {
                LOG_ERROR("[DeviceGraphOrchestrator::executeLayer] Failed to stream layer "
                          << layer_idx << " to device " << device.toString());
                return false;
            }

            // Prefetch next layer(s) asynchronously to overlap with compute
            int n_layers = graph_builder_ ? graph_builder_->config().n_layers : 0;
            if (layer_idx + 1 < n_layers)
            {
                weight_streamer_->prefetchLayer(layer_idx + 1, device);
                LOG_TRACE("[DeviceGraphOrchestrator::executeLayer] Prefetching layer " << (layer_idx + 1));
            }
        }

        // Execute attention block
        if (!executeAttention(layer, buffers, layer_idx, seq_len, kv_cache, position_ids, device))
        {
            // Release layer on failure if streaming
            if (weight_streamer_)
            {
                weight_streamer_->releaseLayer(layer_idx);
            }
            return false;
        }

        // Execute FFN block
        if (!executeFFN(layer, buffers, layer_idx, seq_len, device))
        {
            // Release layer on failure if streaming
            if (weight_streamer_)
            {
                weight_streamer_->releaseLayer(layer_idx);
            }
            return false;
        }

        // After layer execution: Release layer (marks as eligible for eviction)
        if (weight_streamer_)
        {
            weight_streamer_->releaseLayer(layer_idx);
            LOG_TRACE("[DeviceGraphOrchestrator::executeLayer] Released layer " << layer_idx);
        }

        return true;
    }

    // =========================================================================
    // Cache Management
    // =========================================================================

    void DeviceGraphOrchestrator::invalidateExecutionCaches()
    {
        // Clear graph caches
        for (auto &cache : layer_graph_cache_)
        {
            cache.invalidate();
        }

        // Clear forward graph caches
        if (forward_engine_)
            forward_engine_->discardAllCachedGraphs();
        mtp_sidecar_depth0_cache_.invalidate();
        mtp_sidecar_depth0_device_token_cache_.invalidate();
        mtp_sidecar_depth0_chained_cache_.invalidate();
        mtp_sidecar_depth0_chained_device_token_cache_.invalidate();
        mtp_sidecar_depth0_kv_only_cache_.invalidate();
        mtp_sidecar_depth0_kv_only_device_token_cache_.invalidate();
        for (auto &cache : mtp_sidecar_depth0_kv_only_batch_caches_)
            cache.invalidate();
        mtp_terminal_hidden_row_select_cache_.invalidate();
        mtp_terminal_hidden_rows_select_cache_.invalidate();

        // Clear device contexts
        device_contexts_.clear();

        // Reset state
        last_pos_offset_ = -1;

        // Reset stats
        cache_stats_ = CacheStats{};

        // Reset input-dependent cached state on all kernels
        resetKernelDynamicState();
        ++session_epoch_;

        LOG_DEBUG("[DeviceGraphOrchestrator] Execution caches invalidated");
    }

    void DeviceGraphOrchestrator::invalidateGraphCache(int layer_idx)
    {
        if (layer_idx < 0)
        {
            // Invalidate all layers
            for (auto &cache : layer_graph_cache_)
            {
                cache.invalidate();
            }
            cache_stats_.cached_layers = 0;
            LOG_DEBUG("[DeviceGraphOrchestrator] All layer graph caches invalidated");
        }
        else if (static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            layer_graph_cache_[layer_idx].invalidate();
            LOG_DEBUG("[DeviceGraphOrchestrator] Layer " << layer_idx << " graph cache invalidated");
        }
    }

    void DeviceGraphOrchestrator::resetKernelDynamicState()
    {
        llaminar::v2::kernels::KernelFactory::resetAllDynamicState();
        if (prepared_weight_store_)
        {
            prepared_weight_store_->resetDynamicState();
        }
    }

    bool DeviceGraphOrchestrator::hasValidCachedGraph(int layer_idx, bool is_attention) const
    {
        if (!cache_config_.enabled)
            return false;
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_graph_cache_.size())
            return false;

        const auto &cache = layer_graph_cache_[layer_idx];
        if (!cache.valid)
            return false;

        return is_attention ? (cache.attention_decode != nullptr) : (cache.ffn_decode != nullptr);
    }

    void DeviceGraphOrchestrator::setGraphCachingEnabled(bool enabled)
    {
        if (cache_config_.enabled != enabled)
        {
            cache_config_.enabled = enabled;
            if (!enabled)
            {
                invalidateGraphCache(-1);
            }
            LOG_DEBUG("[DeviceGraphOrchestrator] Graph caching "
                      << (enabled ? "enabled" : "disabled"));
        }
    }

    void DeviceGraphOrchestrator::initializeGraphCache(int n_layers)
    {
        layer_graph_cache_.resize(n_layers);
        cache_stats_.cached_layers = n_layers;
        LOG_DEBUG("[DeviceGraphOrchestrator] Graph cache initialized for " << n_layers << " layers");
    }

    // =========================================================================
    // Inference State Management
    // =========================================================================

    bool DeviceGraphOrchestrator::initializeInferenceStateFromArena(
        int batch_size,
        int max_seq_len,
        DeviceId device,
        const InferenceStateInitConfig &init_config)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot initialize state: no graph builder");
            return false;
        }

        const auto &config = graph_builder_->config();
        const int activation_seq_len =
            init_config.activation_seq_len > 0
                ? std::min(max_seq_len, init_config.activation_seq_len)
                : max_seq_len;
        if (batch_size <= 0 || max_seq_len <= 0 || activation_seq_len <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid inference state shape: batch_size="
                      << batch_size << " max_seq_len=" << max_seq_len
                      << " activation_seq_len=" << activation_seq_len);
            return false;
        }

        // =====================================================================
        // Ensure TensorFactory exists (needed for arena allocation + snapshots)
        // =====================================================================
        if (!tensor_factory_)
        {
            std::shared_ptr<IMPIContext> local_mpi_ctx = mpi_ctx_;
            if (!local_mpi_ctx)
            {
                local_mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
            }
            owned_tensor_factory_ = std::make_unique<TensorFactory>(*local_mpi_ctx);
            tensor_factory_ = owned_tensor_factory_.get();

            // Enable mapped memory for GPU + zero-copy scenarios
            if (init_config.use_mapped_memory && device.is_gpu())
            {
                owned_tensor_factory_->setUseMappedMemoryForGPU(true);
                LOG_DEBUG("[DeviceGraphOrchestrator] Arena path: enabling mapped memory for GPU tensors");
            }
        }

        // =====================================================================
        // Create arena if not already set up via initializeBuffers()
        // =====================================================================
        if (!arena_)
        {
            // Set device_id early (initializeBuffers reads it for mapped memory)
            state_.device_id = device;

            // Temporarily set snapshot_enabled_ if mapped memory requested
            // (initializeBuffers checks snapshot_enabled_ for mapped memory decision)
            bool prev_snapshot = snapshot_enabled_;
            if (init_config.use_mapped_memory)
            {
                snapshot_enabled_ = true;
            }

            if (!initializeBuffers(activation_seq_len))
            {
                snapshot_enabled_ = prev_snapshot;
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to initialize buffers via arena");
                return false;
            }
            snapshot_enabled_ = prev_snapshot;
        }

        // =====================================================================
        // Pull activation buffers from arena (schema-driven allocation)
        // =====================================================================
        state_.hidden = arena_->getSharedTensor(BufferId::HIDDEN_STATE);
        state_.logits = arena_->getSharedTensor(BufferId::LOGITS);
        state_.logits_local = arena_->getSharedTensor(BufferId::LOGITS_LOCAL); // nullptr if not TP
        state_.normalized = arena_->getSharedTensor(BufferId::NORMALIZED);
        state_.residual = arena_->getSharedTensor(BufferId::RESIDUAL);
        state_.Q = arena_->getSharedTensor(BufferId::Q_PROJ);
        state_.K = arena_->getSharedTensor(BufferId::K_PROJ);
        state_.V = arena_->getSharedTensor(BufferId::V_PROJ);
        state_.attn_output = arena_->getSharedTensor(BufferId::ATTN_OUTPUT);
        state_.attn_proj = arena_->getSharedTensor(BufferId::ATTN_PROJ);
        state_.gate = arena_->getSharedTensor(BufferId::GATE_PROJ);
        state_.up = arena_->getSharedTensor(BufferId::UP_PROJ);
        state_.ffn_output = arena_->getSharedTensor(BufferId::FFN_OUTPUT);
        state_.workspace_scores = arena_->getSharedTensor(BufferId::ATTN_SCORES_WORKSPACE);
        state_.workspace_context = arena_->getSharedTensor(BufferId::ATTN_CONTEXT_WORKSPACE);
        state_.workspace_mask = arena_->getSharedTensor(BufferId::GEMM_WORKSPACE);

        // Conditional buffers (Hybrid/HybridQ16 mode only — nullptr if not in schema)
        state_.Q_rope = arena_->getSharedTensor(BufferId::Q_ROPE);
        state_.K_rope = arena_->getSharedTensor(BufferId::K_ROPE);
        state_.V_dequant = arena_->getSharedTensor(BufferId::V_DEQUANT);

        // Auto-discover extension buffers (model-specific BufferIds registered
        // by the schema, e.g. GDN, MoE). Any BufferId that doesn't map to a
        // named InferenceState field is stored in extension_buffers and flows
        // through toModelBuffers() → ActivationBuffers::extensions automatically.
        static const std::unordered_set<BufferId> core_ids = {
            BufferId::HIDDEN_STATE,
            BufferId::LOGITS,
            BufferId::LOGITS_LOCAL,
            BufferId::NORMALIZED,
            BufferId::RESIDUAL,
            BufferId::Q_PROJ,
            BufferId::K_PROJ,
            BufferId::V_PROJ,
            BufferId::ATTN_OUTPUT,
            BufferId::ATTN_PROJ,
            BufferId::GATE_PROJ,
            BufferId::UP_PROJ,
            BufferId::FFN_OUTPUT,
            BufferId::ATTN_SCORES_WORKSPACE,
            BufferId::ATTN_CONTEXT_WORKSPACE,
            BufferId::GEMM_WORKSPACE,
            BufferId::Q_ROPE,
            BufferId::K_ROPE,
            BufferId::V_DEQUANT,
        };
        state_.extension_buffers.clear();
        arena_->forEachRegistered([&](BufferId id)
                                  {
            if (core_ids.count(id) == 0)
            {
                auto tensor = arena_->getSharedTensor(id);
                if (tensor)
                {
                    state_.extension_buffers[id] = std::move(tensor);
                }
            } });

        // Validate required buffers
        if (!state_.hidden || !state_.logits || !state_.normalized ||
            !state_.Q || !state_.K || !state_.V ||
            !state_.attn_output || !state_.attn_proj ||
            !state_.gate || !state_.up || !state_.ffn_output)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Missing required buffers from arena. "
                      "Ensure Qwen2Schema provides all layer_buffers and model_buffers.");
            return false;
        }

        // =====================================================================
        // Non-arena state: K_head_scales (HybridQ16 only)
        // =====================================================================
        ActivationPrecision act_prec = config.activation_precision;
        if (act_prec == ActivationPrecision::HybridQ16)
        {
            int buffer_n_kv_heads = config.qkv_column_parallel ? config.local_n_kv_heads : config.n_kv_heads;
            const size_t k_head_scales_size = static_cast<size_t>(batch_size * activation_seq_len * buffer_n_kv_heads);
            state_.K_head_scales.resize(k_head_scales_size, 1.0f);
            LOG_DEBUG("[DeviceGraphOrchestrator] HybridQ16 K precision fix: allocated K_head_scales ("
                      << k_head_scales_size << " floats)");
        }

        // =====================================================================
        // Snapshot buffers (allocated directly, not in schema yet — Phase 2)
        // =====================================================================
#ifdef ENABLE_PIPELINE_SNAPSHOTS
        if (tensor_factory_)
        {
            int buffer_n_heads = config.qkv_column_parallel ? config.local_n_heads : config.n_heads;
            int head_dim = config.head_dim;
            int d_model = config.d_model;

            state_.context_snapshot = tensor_factory_->createFP32(
                {static_cast<size_t>(batch_size * activation_seq_len), static_cast<size_t>(buffer_n_heads * head_dim)},
                device);
            state_.attention_output_snapshot = tensor_factory_->createFP32(
                {static_cast<size_t>(batch_size * activation_seq_len), static_cast<size_t>(d_model)},
                device);
            state_.attention_residual_snapshot = tensor_factory_->createFP32(
                {static_cast<size_t>(batch_size * activation_seq_len), static_cast<size_t>(d_model)},
                device);
            LOG_DEBUG("[DeviceGraphOrchestrator] Allocated snapshot buffers from arena path");
        }
#endif

        // =====================================================================
        // KV cache creation (not arena-managed)
        // =====================================================================
        int n_layers = pp_stage_config_.has_value()
                           ? pp_stage_config_.value().layerCount()
                           : config.n_layers;

        std::shared_ptr<IMPIContext> local_mpi_ctx = mpi_ctx_;
        if (!local_mpi_ctx)
        {
            local_mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
        }

        if (!initializeKVCaches(batch_size, max_seq_len, n_layers, device, local_mpi_ctx))
        {
            return false;
        }

        // =====================================================================
        // Initialize position tracking and config
        // =====================================================================
        state_.positions.assign(batch_size, 0);
        state_.sequence_lengths.assign(batch_size, 0);
        state_.batch_size = batch_size;
        state_.max_seq_len = max_seq_len;
        state_.activation_seq_len = activation_seq_len;
        state_.d_model = config.d_model;
        state_.vocab_size = config.vocab_size;
        state_.device_id = device;

        LOG_DEBUG("[DeviceGraphOrchestrator] Inference state initialized from arena: "
                  << "batch_size=" << batch_size
                  << " max_seq_len=" << max_seq_len
                  << " activation_seq_len=" << activation_seq_len
                  << " device=" << device.toString());
        return true;
    }

    bool DeviceGraphOrchestrator::initializeKVCaches(
        int batch_size, int max_seq_len, int n_layers,
        DeviceId device, const std::shared_ptr<IMPIContext> &local_mpi_ctx)
    {
        const auto &config = graph_builder_->config();
        const int n_kv_heads = config.n_kv_heads;
        const int head_dim = config.head_dim;

        // Resolve activation precision from config
        ActivationPrecision act_prec = config.activation_precision;

        // Resolve KV cache precision and layout
        ActivationPrecision kv_cache_prec = resolveBufferPrecision(
            act_prec, HybridBufferType::KV_Cache, nullptr);
        kv_cache_prec = resolveKVCacheStoragePrecision(config.kv_cache_precision, device.is_cpu());
        LOG_DEBUG("[DeviceGraphOrchestrator] KV cache precision: " << activationPrecisionToString(kv_cache_prec));
        LOG_DEBUG("[DeviceGraphOrchestrator] KV cache precision mode: "
                  << kvCachePrecisionToString(config.kv_cache_precision));

        // Determine KV cache layout mode:
        // - Q16_1 precision requires HEAD_MAJOR layout for Q16IntegerAttention kernel
        // - Other precisions use POSITION_MAJOR (legacy layout)
        KVCacheLayoutMode kv_layout_mode = (kv_cache_prec == ActivationPrecision::Q16_1)
                                               ? KVCacheLayoutMode::HEAD_MAJOR
                                               : KVCacheLayoutMode::POSITION_MAJOR;
        LOG_DEBUG("[DeviceGraphOrchestrator] KV cache layout mode: "
                  << (kv_layout_mode == KVCacheLayoutMode::HEAD_MAJOR ? "HEAD_MAJOR" : "POSITION_MAJOR"));

        // Set sharding parameters if needed (tensor parallelism)
        // Sharding is needed when local_n_kv_heads < n_kv_heads, AND tensor parallelism is active.
        // TP can be:
        // - GLOBAL TP: Multiple MPI ranks (mpi_ctx_->world_size() > 1)
        // - LOCAL TP: Multiple devices within single rank (tp_ctx->isLocal() && degree() > 1)
        // - NODE_LOCAL TP: Cross-rank same node (tp_ctx->isNodeLocal())
        // - GLOBAL TP: Cross-rank (tp_ctx->isGlobal()) or MPI world_size > 1
        bool use_sharded_cache = (config.local_n_kv_heads > 0 && config.local_n_kv_heads < n_kv_heads);
        bool has_tp = config.tp_ctx && config.tp_ctx->degree() > 1;
        bool is_global_tp = !has_tp && mpi_ctx_ && mpi_ctx_->world_size() > 1;

        // =====================================================================
        // KV Cache Creation: Per-stage for PP, single for non-PP
        // =====================================================================
        if (pipeline_config_ && pipeline_config_->hasPP())
        {
            // Pipeline Parallelism: Create a KV cache for each PP stage's device
            // Each cache only stores the layers processed by that stage
            LOG_DEBUG("[DeviceGraphOrchestrator] Creating per-stage KV caches for PP ("
                      << pipeline_config_->numStages() << " stages)");

            for (const auto &pp_stage : pipeline_config_->pp_stages)
            {
                // Get device for this stage
                const TPDomainConfig *domain = pipeline_config_->getDomainForStage(pp_stage.stage_id);
                if (!domain)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] No domain for PP stage " << pp_stage.stage_id);
                    return false;
                }
                DeviceId stage_device = domain->primaryDevice();

                // Skip if we already have a cache for this device
                // (multiple stages on same device share one cache)
                if (state_.pp_kv_caches.find(stage_device) != state_.pp_kv_caches.end())
                {
                    LOG_DEBUG("[DeviceGraphOrchestrator] Reusing existing KV cache for device "
                              << stage_device.to_string());
                    continue;
                }

                // Count layers on this device
                int layers_on_device = 0;
                int first_layer_on_device = -1;
                for (const auto &stage : pipeline_config_->pp_stages)
                {
                    const TPDomainConfig *stage_domain = pipeline_config_->getDomainForStage(stage.stage_id);
                    if (stage_domain && stage_domain->primaryDevice() == stage_device)
                    {
                        int stage_layers = stage.last_layer - stage.first_layer;
                        if (first_layer_on_device < 0)
                            first_layer_on_device = stage.first_layer;
                        layers_on_device += stage_layers;
                    }
                }

                // Build KVCacheConfig for this stage
                llaminar::v2::kernels::KVCacheConfig kv_config;
                kv_config.precision = kv_cache_prec;
                kv_config.device = stage_device;
                kv_config.num_layers = layers_on_device;
                kv_config.first_layer_index = first_layer_on_device; // Layer index offset
                kv_config.batch_size = batch_size;
                kv_config.max_seq_len = max_seq_len;
                kv_config.n_kv_heads = n_kv_heads;
                kv_config.head_dim = head_dim;
                kv_config.layout_mode = kv_layout_mode;
                kv_config.mpi_ctx = local_mpi_ctx.get();
                kv_config.turboquant_ctx = config.turboquant_ctx;

                if (use_sharded_cache && (has_tp || is_global_tp))
                {
                    kv_config.local_n_kv_heads = config.local_n_kv_heads;
                    if (has_tp)
                    {
                        kv_config.kv_head_start = config.tp_device_idx * config.local_n_kv_heads;
                    }
                    else
                    {
                        kv_config.kv_head_start = mpi_ctx_->rank() * config.local_n_kv_heads;
                    }
                }

                LOG_DEBUG("[DeviceGraphOrchestrator] Creating KV cache for PP stage device "
                          << stage_device.to_string() << ": layers [" << first_layer_on_device
                          << ", " << (first_layer_on_device + layers_on_device) << "), "
                          << layers_on_device << " layers, precision="
                          << activationPrecisionToString(kv_cache_prec));

                state_.pp_kv_caches[stage_device] =
                    llaminar::v2::kernels::KernelFactory::createKVCache(kv_config);

                if (!state_.pp_kv_caches[stage_device])
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to create KV cache for device "
                              << stage_device.to_string());
                    return false;
                }
            }

            LOG_DEBUG("[DeviceGraphOrchestrator] Created " << state_.pp_kv_caches.size()
                                                           << " per-device KV caches for PP");
        }
        else
        {
            // Non-PP: Single KV cache for all layers
            // (also used per-stage in PP mode; each stage has its own orchestrator)
            llaminar::v2::kernels::KVCacheConfig kv_config;
            kv_config.precision = kv_cache_prec;
            kv_config.device = device;
            kv_config.num_layers = n_layers;
            kv_config.batch_size = batch_size;
            kv_config.max_seq_len = max_seq_len;
            kv_config.n_kv_heads = n_kv_heads;
            kv_config.head_dim = head_dim;
            kv_config.layout_mode = kv_layout_mode;
            kv_config.mpi_ctx = local_mpi_ctx.get();

            // PP layer offset for hybrid KV cache: ensures GDN kernel init
            // loop uses global layer indices (e.g., 12..23 for PP stage 2).
            if (pp_stage_config_.has_value())
            {
                kv_config.first_layer_index = pp_stage_config_.value().first_layer;
            }
            kv_config.turboquant_ctx = config.turboquant_ctx;

            // For hybrid models (e.g., Qwen 3.5 with GDN + FA layers),
            // configure the hybrid KV cache with layer type mapping and GDN sizing
            std::unique_ptr<HybridKVCacheConfig> hybrid_config_storage;
            if (config.hasGDN() && !config.layer_types.empty())
            {
                hybrid_config_storage = std::make_unique<HybridKVCacheConfig>();
                hybrid_config_storage->first_layer_index = kv_config.first_layer_index;
                const int first_layer = std::max(0, kv_config.first_layer_index);
                const int layer_count = std::max(0, kv_config.num_layers);
                if (first_layer < static_cast<int>(config.layer_types.size()) &&
                    layer_count > 0 &&
                    first_layer + layer_count <= static_cast<int>(config.layer_types.size()))
                {
                    hybrid_config_storage->layer_types.assign(
                        config.layer_types.begin() + first_layer,
                        config.layer_types.begin() + first_layer + layer_count);
                }
                else
                {
                    hybrid_config_storage->first_layer_index = 0;
                    hybrid_config_storage->layer_types = config.layer_types;
                }
                hybrid_config_storage->gdn_conv_kernel_size = config.gdn.conv_kernel_size;
                hybrid_config_storage->gdn_state_size = config.gdn.state_size;
                hybrid_config_storage->gdn_inner_size = config.gdn.inner_size;
                hybrid_config_storage->gdn_group_count = config.gdn.group_count;
                hybrid_config_storage->gdn_time_step_rank = config.gdn.time_step_rank;
                hybrid_config_storage->n_heads = config.n_heads;
                hybrid_config_storage->local_n_heads = config.local_n_heads;
                kv_config.hybrid_config = hybrid_config_storage.get();

                LOG_DEBUG("[DeviceGraphOrchestrator] Hybrid KV cache config: "
                          << "layers [" << hybrid_config_storage->first_layer_index
                          << ", " << (hybrid_config_storage->first_layer_index +
                                      static_cast<int>(hybrid_config_storage->layer_types.size()))
                          << "), "
                          << hybrid_config_storage->countKVLayers() << " KV layers, "
                          << (static_cast<int>(hybrid_config_storage->layer_types.size()) -
                              hybrid_config_storage->countKVLayers())
                          << " GDN layers");
            }
            kv_config.max_seq_len = max_seq_len;
            kv_config.n_kv_heads = n_kv_heads;
            kv_config.head_dim = head_dim;
            kv_config.layout_mode = kv_layout_mode;
            kv_config.mpi_ctx = local_mpi_ctx.get();
            kv_config.turboquant_ctx = config.turboquant_ctx;

            if (use_sharded_cache && (has_tp || is_global_tp))
            {
                kv_config.local_n_kv_heads = config.local_n_kv_heads;

                // Calculate kv_head_start based on TP mode:
                // - Any TP context: Use tp_device_idx (works for LOCAL, NODE_LOCAL, GLOBAL)
                // - Legacy GLOBAL TP (no tp_ctx): Use MPI rank
                if (has_tp)
                {
                    kv_config.kv_head_start = config.tp_device_idx * config.local_n_kv_heads;
                    LOG_DEBUG("[DeviceGraphOrchestrator] Creating sharded KV cache (TP scope="
                              << static_cast<int>(config.tp_ctx->scope()) << "): "
                              << n_kv_heads << " total KV heads, "
                              << config.local_n_kv_heads << " local KV heads (tp_idx="
                              << config.tp_device_idx << ", start=" << kv_config.kv_head_start << ")"
                              << " precision=" << activationPrecisionToString(kv_cache_prec));
                }
                else
                {
                    kv_config.kv_head_start = mpi_ctx_->rank() * config.local_n_kv_heads;
                    LOG_DEBUG("[DeviceGraphOrchestrator] Creating sharded KV cache (GLOBAL TP): "
                              << n_kv_heads << " total KV heads, "
                              << config.local_n_kv_heads << " local KV heads (rank="
                              << mpi_ctx_->rank() << ", start=" << kv_config.kv_head_start << ")"
                              << " precision=" << activationPrecisionToString(kv_cache_prec));
                }
            }

            // Create cache via factory (handles sharded vs non-sharded automatically)
            state_.kv_cache = llaminar::v2::kernels::KernelFactory::createKVCache(kv_config);
        }

        if (config.mtp.enabled)
        {
            if (pipeline_config_ && pipeline_config_->hasPP())
            {
                state_.mtp_kv_caches.clear();
                LOG_WARN("[DeviceGraphOrchestrator] MTP request-local KV caches are single-device only in Phase 5");
            }
            else if (!initializeMTPKVCaches(
                         batch_size,
                         max_seq_len,
                         kv_cache_prec,
                         kv_layout_mode,
                         device,
                         local_mpi_ctx,
                         use_sharded_cache,
                         has_tp,
                         is_global_tp))
            {
                return false;
            }
        }
        else
        {
            state_.mtp_kv_caches.clear();
        }

        return true;
    }

    const float *DeviceGraphOrchestrator::forward(
        const int *tokens,
        int seq_len,
        int batch_size)
    {
        return forwardImpl(tokens, /*token_ids_device=*/nullptr, seq_len, batch_size);
    }

    bool DeviceGraphOrchestrator::forwardWithDeviceTokenIds(
        const int *token_shadow,
        const void *token_ids_device,
        int seq_len)
    {
        if (!token_shadow || !token_ids_device || seq_len <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardWithDeviceTokenIds requires a host token shadow, "
                      "a device token pointer, and a positive sequence length");
            return false;
        }
        if (!state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardWithDeviceTokenIds is only valid for GPU runners");
            return false;
        }
        return forwardImpl(token_shadow, token_ids_device, seq_len, /*batch_size=*/1) != nullptr;
    }

    const void *DeviceGraphOrchestrator::prepareMTPVerifierInputTokensOnDevice(
        int32_t first_token,
        int first_draft_slot,
        int draft_token_count,
        int total_verifier_input_tokens)
    {
        if (!state_.device_id.is_gpu() ||
            !mtp_verifier_input_tokens_dev_ ||
            !stochastic_draft_sample_tokens_dev_)
        {
            return nullptr;
        }
        if (total_verifier_input_tokens <= 0 ||
            total_verifier_input_tokens >
                static_cast<int>(sampling_math::kSpeculativeBatchMaxRows + 1) ||
            draft_token_count < 0 ||
            draft_token_count > static_cast<int>(sampling_math::kSpeculativeBatchMaxRows) ||
            draft_token_count + 1 != total_verifier_input_tokens ||
            first_draft_slot < 0 ||
            first_draft_slot + draft_token_count >
                static_cast<int>(sampling_math::kSpeculativeBatchMaxRows))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP verifier device-token plan: total="
                      << total_verifier_input_tokens
                      << " drafts=" << draft_token_count
                      << " first_draft_slot=" << first_draft_slot);
            return nullptr;
        }

        /*
         * Do not copy immediately. The forward engine chooses the graph replay
         * stream later, so eager copies could race captured replay on another
         * stream.  The metadata hook will materialize this plan on the exact
         * stream that consumes the token row.
         */
        pending_mtp_verifier_device_token_plan_ =
            PendingMTPVerifierDeviceTokenPlan{
                first_token,
                -1,
                false,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens};
        return mtp_verifier_input_tokens_dev_;
    }

    const void *DeviceGraphOrchestrator::prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
        int first_target_sample_slot,
        int first_draft_slot,
        int draft_token_count,
        int total_verifier_input_tokens)
    {
        if (!state_.device_id.is_gpu() ||
            !mtp_verifier_input_tokens_dev_ ||
            !stochastic_target_sample_tokens_dev_ ||
            !stochastic_draft_sample_tokens_dev_)
        {
            return nullptr;
        }
        if (total_verifier_input_tokens <= 0 ||
            total_verifier_input_tokens >
                static_cast<int>(sampling_math::kSpeculativeBatchMaxRows + 1) ||
            draft_token_count < 0 ||
            draft_token_count > static_cast<int>(sampling_math::kSpeculativeBatchMaxRows) ||
            draft_token_count + 1 != total_verifier_input_tokens ||
            first_target_sample_slot < 0 ||
            first_target_sample_slot >=
                static_cast<int>(sampling_math::kSpeculativeBatchMaxOutputTokens) ||
            first_draft_slot < 0 ||
            first_draft_slot + draft_token_count >
                static_cast<int>(sampling_math::kSpeculativeBatchMaxRows))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP verifier device-first-token plan: total="
                      << total_verifier_input_tokens
                      << " drafts=" << draft_token_count
                      << " first_target_slot=" << first_target_sample_slot
                      << " first_draft_slot=" << first_draft_slot);
            return nullptr;
        }

        /*
         * The sampled target token is still on the sampler stream.  Store a
         * logical copy plan here and let materializePendingMTPVerifierInputTokensOnDevice()
         * wait/copy on the verifier graph stream once that stream is known.
         */
        pending_mtp_verifier_device_token_plan_ =
            PendingMTPVerifierDeviceTokenPlan{
                -1,
                first_target_sample_slot,
                true,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens};
        return mtp_verifier_input_tokens_dev_;
    }

    const float *DeviceGraphOrchestrator::forwardImpl(
        const int *tokens,
        const void *token_ids_device,
        int seq_len,
        int batch_size)
    {
        // Enable device-scoped logging for this execution
        // All LOG_* calls from this thread will include the device ID
        ScopedDeviceLog device_log(state_.device_id);

        if (!state_.isInitialized())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forward() called without initialized state");
            return nullptr;
        }

        if (!hasGlobalWeights())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forward() called without global weights set");
            return nullptr;
        }

        if (batch_size > state_.batch_size)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batch size " << batch_size
                                                              << " exceeds initialized batch size " << state_.batch_size);
            return nullptr;
        }

        int total_tokens = batch_size * seq_len;
        if (total_tokens > state_.batch_size * state_.max_seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Total tokens " << total_tokens
                                                                << " exceeds buffer capacity "
                                                                << state_.batch_size * state_.max_seq_len);
            return nullptr;
        }
        const int activation_seq_len =
            state_.activation_seq_len > 0 ? state_.activation_seq_len : state_.max_seq_len;
        if (total_tokens > state_.batch_size * activation_seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Total tokens " << total_tokens
                                                                << " exceeds activation graph buffer capacity "
                                                                << state_.batch_size * activation_seq_len
                                                                << " (context capacity is "
                                                                << state_.batch_size * state_.max_seq_len
                                                                << "); run long prompts through prefill chunk scheduling "
                                                                   "or increase the activation arena length");
            return nullptr;
        }

        // =====================================================================
        // Gap 4: Automatic phase transition based on live request position.
        // =====================================================================
        // Short multi-token continuations are used by greedy MTP verification:
        // they extend an existing KV/GDN history and must use decode semantics
        // even though seq_len > 1. Treat only position-zero multi-token input as
        // prompt prefill.
        // =====================================================================
        const int decode_max_seq_len = std::max(1, cache_config_.decode_seq_len);
        const bool is_single_token_decode = (seq_len == 1 && batch_size <= 1);
        const bool is_short_continuation_decode =
            batch_size <= 1 &&
            seq_len > 1 &&
            seq_len <= decode_max_seq_len &&
            state_.positions[0] > 0;
        const InferencePhase new_phase =
            (is_single_token_decode || is_short_continuation_decode)
                ? InferencePhase::DECODE
                : InferencePhase::PREFILL;
        transitionToPhase(new_phase);

        // Build position IDs (per-batch offsets for variable-length sequences)
        std::vector<int> position_ids;
        position_ids.reserve(total_tokens);
        for (int b = 0; b < batch_size; ++b)
        {
            int pos_offset = state_.positions[b];
            for (int s = 0; s < seq_len; ++s)
            {
                position_ids.push_back(pos_offset + s);
            }
        }

        TensorBase *logits_output = state_.logits.get();
        TensorBase *logits_local_output = state_.logits_local.get();
        if (compute_all_position_logits_)
        {
            const auto &config = graph_builder_->config();
            size_t rows = static_cast<size_t>(total_tokens);
            if (compute_row_indexed_all_position_logits_)
            {
                if (batch_size > 1 && !pending_mtp_spec_verifier_input_plan_)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Batched row-indexed all-position logits require an explicit MTP verifier row plan");
                    return nullptr;
                }
                if (row_indexed_all_position_logits_row_count_ <= 0 ||
                    row_indexed_all_position_logits_row_count_ > total_tokens)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Invalid row-indexed all-position logit row count: rows="
                              << row_indexed_all_position_logits_row_count_
                              << " total_tokens=" << total_tokens);
                    return nullptr;
                }
                rows = static_cast<size_t>(row_indexed_all_position_logits_row_count_);
            }
            const int rows_key = static_cast<int>(rows);
            if (config.lm_head_column_parallel)
            {
                if (!state_.logits_local)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] All-position local logits require logits_local");
                    return nullptr;
                }
                if (!tensor_factory_)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] All-position logits require an initialized TensorFactory");
                    return nullptr;
                }

                const auto &local_shape = state_.logits_local->shape();
                const size_t local_vocab =
                    local_shape.size() >= 2 ? local_shape[1] : static_cast<size_t>(std::max(0, config.vocab_local));
                if (local_vocab == 0)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] All-position local logits require a non-zero local vocab");
                    return nullptr;
                }

                auto &local_logits_owner = state_.all_position_logits_local_by_rows[rows_key];
                bool needs_allocate = !local_logits_owner;
                if (local_logits_owner)
                {
                    const auto &shape = local_logits_owner->shape();
                    needs_allocate = shape.size() != 2 || shape[0] != rows || shape[1] != local_vocab;
                }
                if (needs_allocate)
                {
                    auto tensor = tensor_factory_->createFP32({rows, local_vocab}, state_.device_id);
                    local_logits_owner = std::shared_ptr<TensorBase>(tensor.release());
                }
                state_.all_position_logits_local = local_logits_owner;
                logits_local_output = state_.all_position_logits_local.get();
                if (!arena_ ||
                    !arena_->bindExternalBuffer(BufferId::ALL_POSITION_LOGITS_LOCAL,
                                                logits_local_output))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to bind all-position local logits buffer");
                    return nullptr;
                }

                const size_t vocab = static_cast<size_t>(state_.vocab_size);
                auto &logits_owner = state_.all_position_logits_by_rows[rows_key];
                needs_allocate = !logits_owner;
                if (logits_owner)
                {
                    const auto &shape = logits_owner->shape();
                    needs_allocate = shape.size() != 2 || shape[0] != rows || shape[1] != vocab;
                }
                if (needs_allocate)
                {
                    auto tensor = tensor_factory_->createFP32({rows, vocab}, state_.device_id);
                    logits_owner = std::shared_ptr<TensorBase>(tensor.release());
                }
                state_.all_position_logits = logits_owner;
                logits_output = state_.all_position_logits.get();
                if (!arena_ ||
                    !arena_->bindExternalBuffer(BufferId::ALL_POSITION_LOGITS,
                                                logits_output))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to bind all-position logits buffer");
                    return nullptr;
                }
            }
            else
            {
                if (!tensor_factory_)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] All-position logits require an initialized TensorFactory");
                    return nullptr;
                }

                const size_t vocab = static_cast<size_t>(state_.vocab_size);
                auto &logits_owner = state_.all_position_logits_by_rows[rows_key];
                bool needs_allocate = !logits_owner;
                if (logits_owner)
                {
                    const auto &shape = logits_owner->shape();
                    needs_allocate = shape.size() != 2 || shape[0] != rows || shape[1] != vocab;
                }
                if (needs_allocate)
                {
                    auto tensor = tensor_factory_->createFP32({rows, vocab}, state_.device_id);
                    logits_owner = std::shared_ptr<TensorBase>(tensor.release());
                }
                state_.all_position_logits = logits_owner;
                logits_output = state_.all_position_logits.get();
                if (!arena_ ||
                    !arena_->bindExternalBuffer(BufferId::ALL_POSITION_LOGITS,
                                                logits_output))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to bind all-position logits buffer");
                    return nullptr;
                }
            }
        }

        // Prepare model buffers from state
        ModelBuffers model_buffers = state_.toModelBuffers();
        model_buffers.logits = logits_output;
        model_buffers.logits_local = logits_local_output;

        setBuffers(model_buffers);

        // Build forward input
        ForwardInput input;
        input.token_ids = tokens;
        input.token_ids_device = token_ids_device;
        input.position_ids = position_ids.data();
        input.batch_size = batch_size;
        input.seq_len = seq_len;
        input.position_offset = state_.positions[0]; // Legacy compat
        input.device = state_.device_id;
        input.kv_cache = state_.kv_cache.get();

        // For PP mode: build raw pointer map from per-device KV caches
        std::unordered_map<DeviceId, IKVCache *> pp_kv_cache_ptrs;
        if (!state_.pp_kv_caches.empty())
        {
            for (const auto &[device, cache] : state_.pp_kv_caches)
            {
                pp_kv_cache_ptrs[device] = cache.get();
            }
            input.pp_kv_caches = &pp_kv_cache_ptrs;
            LOG_DEBUG("[DeviceGraphOrchestrator] Set " << pp_kv_cache_ptrs.size()
                                                       << " per-device KV caches for PP forward");
        }

        // Pass sequence_lengths for batch-aware attention masking
        // This enables proper separation of sequences in batched execution
        input.sequence_lengths = (batch_size > 1 && !state_.sequence_lengths.empty())
                                     ? &state_.sequence_lengths
                                     : nullptr;

        // Build forward output
        ForwardOutput output;
        output.logits = logits_output;
        output.hidden = state_.hidden.get();

        // Execute forward pass
        bool success = executeForward(input, output);

        if (!success)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forward() execution failed");
            return nullptr;
        }

        // Update positions
        for (int b = 0; b < batch_size; ++b)
        {
            state_.positions[b] += seq_len;
            state_.sequence_lengths[b] += seq_len;
        }

        if (new_phase == InferencePhase::PREFILL &&
            !populateMTPShiftedCacheFromPrefill(tokens, seq_len, batch_size, input.position_offset))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to populate MTP shifted prefill cache");
            return nullptr;
        }

        noteMainForwardHiddenProducedForMTP(seq_len, batch_size);

        // After first prefill, release host-resident weight data. Keep the host
        // bytes alive until MTP shifted-prefill has materialized its sidecar
        // weights through normal stage-contract coherence.
        if (release_host_resident_after_forward_ && !host_resident_released_ && seq_len > 1 && weight_manager_)
        {
            host_resident_released_ = true;
            weight_manager_->releaseHostResidentWeightData();
            adviseMmapDontneedAfterFirstPrefill();
        }

        LOG_TRACE("[FORWARD_TRACE] seq_len=" << seq_len
                                             << " pos_offset=" << input.position_offset
                                             << " token_ids[0]=" << (tokens ? tokens[0] : -1)
                                             << " positions_after=" << state_.positions[0]);

        // Return logits pointer
        // GPU: logits remain on device (DEVICE_AUTHORITATIVE) — avoid massive D2H transfer.
        // Callers that need host data should call logits() explicitly.
        // CPU: logits are already on host, fp32_data() is essentially free.
        if (state_.device_id.is_gpu())
            return reinterpret_cast<const float *>(logits_output);
        return logits_output ? logits_output->fp32_data() : nullptr;
    }

    bool DeviceGraphOrchestrator::supportsPrefillChunkSchedule(int seq_len) const
    {
        const auto &env = debugEnv().execution;
        if (seq_len <= 1 ||
            !state_.isInitialized() ||
            !state_.device_id.is_gpu() ||
            !cache_config_.enabled ||
            !env.gpu_graphs ||
            !env.prefill_graph_buckets ||
            seq_len < env.prefill_graph_min_seq ||
            pp_stage_config_.has_value() ||
            (pipeline_config_ && pipeline_config_->hasPP()) ||
            compute_all_position_logits_)
        {
            return false;
        }

        const auto buckets = normalizePrefillGraphBuckets(env.prefill_graph_bucket_sizes);
        return !buckets.empty();
    }

    bool DeviceGraphOrchestrator::forwardPrefillChunkSchedule(
        const int *tokens,
        int seq_len,
        const PrefillChunkSchedulerPolicy &policy,
        int pad_token_id,
        bool allow_padded_execution)
    {
        ScopedDeviceLog device_log(state_.device_id);

        if (!tokens || seq_len <= 1)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid prefill chunk schedule input");
            return false;
        }
        if (!supportsPrefillChunkSchedule(seq_len))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill chunk schedule is not supported for current runner state");
            return false;
        }
        if (!hasGlobalWeights())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] prefill chunk schedule called without global weights set");
            return false;
        }
        if (seq_len > state_.max_seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill chunk schedule length " << seq_len
                                                                                 << " exceeds buffer capacity "
                                                                                 << state_.max_seq_len);
            return false;
        }
        const int activation_seq_len =
            state_.activation_seq_len > 0 ? state_.activation_seq_len : state_.max_seq_len;
        const auto normalized_buckets = normalizePrefillGraphBuckets(policy.bucket_sizes);
        if (!normalized_buckets.empty() && normalized_buckets.back() > activation_seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill chunk bucket " << normalized_buckets.back()
                                                                        << " exceeds activation graph buffer capacity "
                                                                        << activation_seq_len);
            return false;
        }

        transitionToPhase(InferencePhase::PREFILL);

        TensorBase *logits_output = state_.logits.get();
        TensorBase *logits_local_output = state_.logits_local.get();
        ModelBuffers model_buffers = state_.toModelBuffers();
        model_buffers.logits = logits_output;
        model_buffers.logits_local = logits_local_output;
        setBuffers(model_buffers);

        ForwardInput input;
        input.token_ids = tokens;
        input.batch_size = 1;
        input.seq_len = seq_len;
        input.real_seq_len = seq_len;
        input.position_offset = state_.positions[0];
        input.token_offset = state_.positions[0];
        input.device = state_.device_id;
        input.kv_cache = state_.kv_cache.get();

        ensureForwardEngine();
        auto schedule = ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
            input,
            policy,
            pad_token_id,
            allow_padded_execution);
        if (!schedule)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to prepare prefill chunk schedule: "
                      << schedule.error);
            return false;
        }

        ForwardOutput output;
        output.logits = logits_output;
        output.hidden = state_.hidden.get();

        if (!forward_engine_->runPrefillChunkSchedule(input, schedule, output, *this))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefill chunk schedule execution failed");
            return false;
        }

        state_.positions[0] += seq_len;
        state_.sequence_lengths[0] += seq_len;

        if (!populateMTPShiftedCacheFromPrefill(tokens, seq_len, 1, input.position_offset))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to populate MTP shifted prefill cache after chunk schedule");
            return false;
        }

        const int terminal_seq_len = schedule.chunks.empty()
                                         ? seq_len
                                         : schedule.chunks.back().chunk.real_count;
        noteMainForwardHiddenProducedForMTP(terminal_seq_len, 1);

        // Chunked prefill has the same sidecar ordering requirement as the
        // monolithic path: shifted MTP cache population must see unreleased
        // sidecar weights so coherence can upload them normally.
        if (release_host_resident_after_forward_ && !host_resident_released_ && weight_manager_)
        {
            host_resident_released_ = true;
            weight_manager_->releaseHostResidentWeightData();
            adviseMmapDontneedAfterFirstPrefill();
        }

        LOG_TRACE("[FORWARD_TRACE] chunk_schedule seq_len=" << seq_len
                                                            << " pos_offset=" << input.position_offset
                                                            << " chunks=" << schedule.chunks.size()
                                                            << " positions_after=" << state_.positions[0]);
        return true;
    }

    bool DeviceGraphOrchestrator::ensureMTPTerminalHiddenBuffer(int min_rows)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (min_rows <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP terminal hidden row capacity request: "
                      << min_rows);
            return false;
        }
        const int required_rows = std::max(1, min_rows);

        auto register_with_arena = [&]() -> bool
        {
            if (!arena_)
                return true;
            if (arena_->isRegistered(BufferId::PREFIX_TERMINAL_HIDDEN))
            {
                if (arena_->getTensor(BufferId::PREFIX_TERMINAL_HIDDEN) == state_.prefix_terminal_hidden.get())
                {
                    return true;
                }
                if (!arena_->bindExternalBuffer(BufferId::PREFIX_TERMINAL_HIDDEN,
                                                state_.prefix_terminal_hidden.get()))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to rebind MTP terminal hidden with BufferArena");
                    return false;
                }
                return true;
            }
            if (!arena_->registerExternalBuffer(BufferId::PREFIX_TERMINAL_HIDDEN,
                                                state_.prefix_terminal_hidden.get()))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to register MTP terminal hidden with BufferArena");
                return false;
            }
            return true;
        };

        if (state_.prefix_terminal_hidden)
        {
            const auto &shape = state_.prefix_terminal_hidden->shape();
            if (shape.size() == 2 &&
                shape[0] >= static_cast<size_t>(required_rows) &&
                shape[1] >= static_cast<size_t>(state_.d_model))
            {
                return register_with_arena();
            }
            state_.prefix_terminal_hidden.reset();
            state_.mtp_terminal_hidden_current = false;
        }

        if (!tensor_factory_ || state_.d_model <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot allocate MTP terminal hidden buffer without tensor factory/d_model");
            return false;
        }

        auto tensor = tensor_factory_->createFP32(
            {static_cast<size_t>(required_rows), static_cast<size_t>(state_.d_model)},
            state_.device_id);
        if (!tensor)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to allocate MTP terminal hidden buffer");
            return false;
        }

        state_.prefix_terminal_hidden = std::shared_ptr<TensorBase>(tensor.release());
        mtp_terminal_hidden_row_select_cache_.invalidate();
        mtp_terminal_hidden_rows_select_cache_.invalidate();
        return register_with_arena();
    }

    bool DeviceGraphOrchestrator::executeMTPHiddenRowSelect(
        TensorBase *input,
        BufferId input_buffer_id,
        TensorBase *output,
        BufferId output_buffer_id,
            MTPTerminalHiddenRowSelectGraphCache &cache,
            const char *node_name,
            int row_idx,
            int seq_len,
            void *stream)
    {
        if (row_idx < 0 || row_idx >= seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP hidden row selection: row="
                      << row_idx << " seq_len=" << seq_len);
            return false;
        }
        if (!input || !output)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP hidden row without input/output buffer");
            return false;
        }

        if (state_.d_model <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP hidden row with invalid d_model="
                      << state_.d_model);
            return false;
        }

        const size_t hidden_rows = input->rows();
        if (hidden_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hidden-state row capacity exceeds int range: rows="
                      << hidden_rows);
            return false;
        }
        const int seq_capacity = static_cast<int>(hidden_rows);
        if (seq_capacity < seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hidden-state row-select capacity too small: capacity="
                      << seq_capacity << " seq_len=" << seq_len);
            return false;
        }

        const bool rebuild =
            !cache.valid ||
            !cache.graph ||
            !cache.stage ||
            cache.input != input ||
            cache.output != output ||
            cache.device != state_.device_id ||
            cache.seq_capacity != seq_capacity ||
            cache.d_model != state_.d_model;

        if (rebuild)
        {
            HiddenStateRowSelectStage::Params params;
            params.device_id = state_.device_id;
            params.input = input;
            params.output = output;
            params.seq_len = seq_capacity;
            params.d_model = state_.d_model;
            params.selected_row_idx = row_idx;
            params.input_buffer_id = input_buffer_id;
            params.output_buffer_id = output_buffer_id;
            params.workspace_buffer_name =
                std::string(HiddenStateRowSelectStage::WS_SELECTED_ROW_SCALAR) +
                "_" + (node_name && node_name[0] != '\0' ? node_name : "mtp_hidden_row_select");

            auto stage = ComputeStageFactory::createHiddenStateRowSelect(params);
            auto *row_select_stage = dynamic_cast<HiddenStateRowSelectStage *>(stage.get());
            if (!row_select_stage)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] HiddenStateRowSelect factory returned an incompatible stage");
                return false;
            }

            auto graph = std::make_unique<ComputeGraph>();
            graph->addNode(node_name && node_name[0] != '\0' ? node_name : "mtp_hidden_row_select",
                           std::move(stage),
                           state_.device_id);

            cache.graph = std::move(graph);
            cache.stage = row_select_stage;
            cache.input = input;
            cache.output = output;
            cache.device = state_.device_id;
            cache.seq_capacity = seq_capacity;
            cache.d_model = state_.d_model;
            cache.valid = true;
        }

        cache.stage->setSelectedRowForReplay(row_idx);
        void *owned_row_select_stream = nullptr;
        if (state_.device_id.is_gpu())
        {
            void *row_select_stream = stream
                                          ? stream
                                          : explicitGPUStreamForOperation("mtp_hidden_row_select");
            if (!row_select_stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP hidden row-select requires an explicit GPU stream");
                return false;
            }
            cache.stage->setGPUStream(row_select_stream);
            if (!stream)
                owned_row_select_stream = row_select_stream;
        }

        IDeviceContext *ctx = getDeviceContext(state_.device_id);
        if (!ctx)
            return false;

        if (!execute(*cache.graph, ctx))
            return false;

        if (owned_row_select_stream &&
            !synchronizeOwnedGPUHelperStream(
                state_.device_id,
                owned_row_select_stream,
                "mtp_hidden_row_select"))
        {
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::executeMTPTerminalHiddenRowSelect(int row_idx, int seq_len, void *stream)
    {
        if (!state_.hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP hidden row without hidden-state buffer");
            return false;
        }
        if (!ensureMTPTerminalHiddenBuffer())
            return false;
        return executeMTPHiddenRowSelect(
            state_.hidden.get(),
            BufferId::HIDDEN_STATE,
            state_.prefix_terminal_hidden.get(),
            BufferId::PREFIX_TERMINAL_HIDDEN,
            mtp_terminal_hidden_row_select_cache_,
            "mtp_terminal_hidden_row_select",
            row_idx,
            seq_len,
            stream);
    }

    bool DeviceGraphOrchestrator::executeMTPHiddenRowsSelect(
        TensorBase *input,
        BufferId input_buffer_id,
        TensorBase *output,
        BufferId output_buffer_id,
        MTPTerminalHiddenRowsSelectGraphCache &cache,
        const char *node_name,
        int row_start,
        int row_count,
        int seq_len,
        void *stream)
    {
        if (row_start < 0 || row_count <= 0 || row_start + row_count > seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP hidden rows selection: start="
                      << row_start << " count=" << row_count << " seq_len=" << seq_len);
            return false;
        }
        if (row_count > 4)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP hidden rows selection supports at most four rows, got "
                      << row_count);
            return false;
        }
        if (!input || !output)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP hidden rows without input/output buffer");
            return false;
        }
        if (state_.d_model <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP hidden rows with invalid d_model="
                      << state_.d_model);
            return false;
        }

        const size_t hidden_rows = input->rows();
        if (hidden_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hidden-state rows-select capacity exceeds int range: rows="
                      << hidden_rows);
            return false;
        }
        const int seq_capacity = static_cast<int>(hidden_rows);
        if (seq_capacity < seq_len)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hidden-state rows-select capacity too small: capacity="
                      << seq_capacity << " seq_len=" << seq_len);
            return false;
        }

        std::vector<int> selected_rows;
        selected_rows.reserve(static_cast<size_t>(row_count));
        for (int i = 0; i < row_count; ++i)
            selected_rows.push_back(row_start + i);

        const bool rebuild =
            !cache.valid ||
            !cache.graph ||
            !cache.stage ||
            cache.input != input ||
            cache.output != output ||
            cache.device != state_.device_id ||
            cache.seq_capacity != seq_capacity ||
            cache.d_model != state_.d_model ||
            cache.selected_row_count != row_count;

        if (rebuild)
        {
            HiddenStateRowsSelectStage::Params params;
            params.device_id = state_.device_id;
            params.input = input;
            params.output = output;
            params.seq_len = seq_capacity;
            params.d_model = state_.d_model;
            params.selected_row_count = row_count;
            params.selected_row_indices = selected_rows;
            params.input_buffer_id = input_buffer_id;
            params.output_buffer_id = output_buffer_id;
            params.workspace_buffer_name =
                std::string(HiddenStateRowsSelectStage::WS_SELECTED_ROWS_ARRAY) +
                "_" + (node_name && node_name[0] != '\0' ? node_name : "mtp_hidden_rows_select");

            auto stage = ComputeStageFactory::createHiddenStateRowsSelect(params);
            auto *rows_select_stage = dynamic_cast<HiddenStateRowsSelectStage *>(stage.get());
            if (!rows_select_stage)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] HiddenStateRowsSelect factory returned an incompatible stage");
                return false;
            }

            auto graph = std::make_unique<ComputeGraph>();
            graph->addNode(node_name && node_name[0] != '\0' ? node_name : "mtp_hidden_rows_select",
                           std::move(stage),
                           state_.device_id);

            cache.graph = std::move(graph);
            cache.stage = rows_select_stage;
            cache.input = input;
            cache.output = output;
            cache.device = state_.device_id;
            cache.seq_capacity = seq_capacity;
            cache.d_model = state_.d_model;
            cache.selected_row_count = row_count;
            cache.valid = true;
        }

        if (!cache.stage->setSelectedRowsForReplay(selected_rows))
            return false;

        void *owned_rows_select_stream = nullptr;
        if (state_.device_id.is_gpu())
        {
            void *rows_select_stream = stream
                                           ? stream
                                           : explicitGPUStreamForOperation("mtp_hidden_rows_select");
            if (!rows_select_stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP hidden rows-select requires an explicit GPU stream");
                return false;
            }
            cache.stage->setGPUStream(rows_select_stream);
            if (!stream)
                owned_rows_select_stream = rows_select_stream;
        }

        IDeviceContext *ctx = getDeviceContext(state_.device_id);
        if (!ctx)
            return false;

        if (!execute(*cache.graph, ctx))
            return false;

        if (owned_rows_select_stream &&
            !synchronizeOwnedGPUHelperStream(
                state_.device_id,
                owned_rows_select_stream,
                "mtp_hidden_rows_select"))
        {
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRows(
        int row_start,
        int row_count,
        int seq_len,
        void *stream)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "terminal_hidden_rows_select",
            perfPhaseName(),
            state_.device_id.toString(),
            {{"rows", std::to_string(row_count)}});
        if (seq_len <= 0)
            return false;
        state_.mtp_terminal_hidden_current = false;
        if (row_count == 1)
            return executeMTPTerminalHiddenRowSelect(row_start, seq_len);
        if (!state_.hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Cannot select MTP terminal hidden rows without hidden-state buffer");
            return false;
        }
        if (!ensureMTPTerminalHiddenBuffer(row_count))
            return false;
        return executeMTPHiddenRowsSelect(
            state_.hidden.get(),
            BufferId::HIDDEN_STATE,
            state_.prefix_terminal_hidden.get(),
            BufferId::PREFIX_TERMINAL_HIDDEN,
            mtp_terminal_hidden_rows_select_cache_,
            "mtp_terminal_hidden_rows_select",
            row_start,
            row_count,
            seq_len,
            stream);
    }

    void DeviceGraphOrchestrator::noteMainForwardHiddenProducedForMTP(int seq_len, int batch_size)
    {
        state_.last_forward_seq_len = seq_len;
        state_.last_forward_batch_size = batch_size;
        state_.mtp_terminal_hidden_current = false;
    }

    bool DeviceGraphOrchestrator::resolveMTPTerminalHiddenInput(
        TensorBase **terminal_hidden,
        BufferId *terminal_hidden_buffer_id,
        const char *operation)
    {
        if (!terminal_hidden || !terminal_hidden_buffer_id)
            return false;

        *terminal_hidden = nullptr;
        *terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;

        if (state_.prefix_terminal_hidden && state_.mtp_terminal_hidden_current)
        {
            *terminal_hidden = state_.prefix_terminal_hidden.get();
            *terminal_hidden_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN;
            return true;
        }

        if (state_.last_forward_batch_size == 1 && state_.last_forward_seq_len > 1)
        {
            if (!refreshMTPTerminalHiddenState(
                    state_.last_forward_seq_len,
                    state_.last_forward_batch_size))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to refresh terminal hidden for "
                          << (operation ? operation : "MTP sidecar"));
                return false;
            }
            if (!state_.prefix_terminal_hidden || !state_.mtp_terminal_hidden_current)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP terminal hidden refresh did not publish a current buffer for "
                          << (operation ? operation : "MTP sidecar"));
                return false;
            }
            *terminal_hidden = state_.prefix_terminal_hidden.get();
            *terminal_hidden_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN;
            return true;
        }

        if (state_.last_forward_batch_size > 1)
        {
            if (state_.last_forward_seq_len == 1)
            {
                if (!refreshMTPTerminalHiddenState(
                        state_.last_forward_seq_len,
                        state_.last_forward_batch_size))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to refresh batched terminal hidden for "
                              << (operation ? operation : "MTP sidecar"));
                    return false;
                }
                if (!state_.prefix_terminal_hidden || !state_.mtp_terminal_hidden_current)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP terminal hidden refresh did not publish"
                              << " a current prefix buffer for "
                              << (operation ? operation : "MTP sidecar"));
                    return false;
                }
                *terminal_hidden = state_.prefix_terminal_hidden.get();
                *terminal_hidden_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN;
                return true;
            }
            LOG_ERROR("[DeviceGraphOrchestrator] "
                      << (operation ? operation : "MTP sidecar")
                      << " requires one terminal hidden row per request; last batch_size="
                      << state_.last_forward_batch_size
                      << " seq_len=" << state_.last_forward_seq_len);
            return false;
        }

        if (state_.last_forward_seq_len == 1 && state_.last_forward_batch_size == 1 && state_.hidden)
        {
            /*
             * GPU MTP sidecar graphs are captured against concrete tensor
             * addresses.  Feeding HIDDEN_STATE after ordinary decode but
             * PREFIX_TERMINAL_HIDDEN after accepted-state publication makes
             * the first-sidecar graph alternate signatures and rebuild in the
             * hot loop.  Copy the single live row into the stable prefix
             * terminal-hidden buffer instead; this matches the vLLM-style
             * persistent-input-buffer contract and keeps graph replay reusable.
             */
            if (state_.device_id.is_gpu())
            {
                if (!refreshMTPTerminalHiddenState(
                        state_.last_forward_seq_len,
                        state_.last_forward_batch_size))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to refresh single-row terminal hidden for "
                              << (operation ? operation : "MTP sidecar"));
                    return false;
                }
                if (!state_.prefix_terminal_hidden || !state_.mtp_terminal_hidden_current)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Single-row MTP terminal hidden refresh did not publish"
                              << " a current prefix buffer for "
                              << (operation ? operation : "MTP sidecar"));
                    return false;
                }
                *terminal_hidden = state_.prefix_terminal_hidden.get();
                *terminal_hidden_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN;
                return true;
            }
            *terminal_hidden = state_.hidden.get();
            *terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
            return true;
        }

        LOG_ERROR("[DeviceGraphOrchestrator] "
                  << (operation ? operation : "MTP sidecar")
                  << " requires a current terminal hidden row, but no prefix restore or main forward is current");
        return false;
    }

    bool DeviceGraphOrchestrator::refreshMTPTerminalHiddenState(int seq_len, int batch_size)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        state_.mtp_terminal_hidden_current = false;
        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "terminal_hidden_refresh",
            perfPhaseName(),
            state_.device_id.toString());
        if (seq_len <= 0 || batch_size <= 0)
            return false;
        const int total_rows = seq_len * batch_size;
        if (batch_size > 1 && seq_len != 1)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP terminal hidden capture supports only seq_len=1, got batch_size="
                      << batch_size << " seq_len=" << seq_len);
            return false;
        }
        if (batch_size > 4)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP terminal hidden capture supports at most four rows, got "
                      << batch_size);
            return false;
        }

        void *producer_stream = nullptr;
        if (state_.device_id.is_gpu())
            producer_stream = peekPendingLogitsStream(PendingLogitsStreamRole::MainDecode);

        if (batch_size == 1)
        {
            if (!executeMTPTerminalHiddenRowSelect(seq_len - 1, seq_len, producer_stream))
                return false;
        }
        else
        {
            // A one-token request batch is flattened as rows [0, batch_size).
            // Copy every row into the stable sidecar input buffer so a batched
            // MTP graph sees one terminal hidden state per request.
            if (!selectMTPTerminalHiddenRows(
                    /*row_start=*/0,
                    /*row_count=*/batch_size,
                    /*seq_len=*/total_rows,
                    producer_stream))
            {
                return false;
            }
        }

        if (producer_stream &&
            !synchronizeOwnedGPUHelperStream(
                state_.device_id,
                producer_stream,
                "mtp_terminal_hidden_refresh"))
        {
            return false;
        }

        state_.mtp_terminal_hidden_current = true;
        return true;
    }

    bool DeviceGraphOrchestrator::selectMTPTerminalHiddenRow(int row_idx, int seq_len, void *stream)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        state_.mtp_terminal_hidden_current = false;
        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "terminal_hidden_row_select",
            perfPhaseName(),
            state_.device_id.toString());
        if (seq_len <= 0)
            return false;

        if (!executeMTPTerminalHiddenRowSelect(row_idx, seq_len, stream))
            return false;
        state_.mtp_terminal_hidden_current = true;
        return true;
    }

    bool DeviceGraphOrchestrator::executeMTPDepth0(
        int32_t draft_condition_token,
        TensorBase *terminal_hidden,
        int position_id,
        const char *sidecar_perf_context,
        bool kv_cache_only,
        BufferId terminal_hidden_buffer_id,
        bool defer_final_sync)
    {
        return executeMTPDepth0Batched(
            &draft_condition_token,
            1,
            terminal_hidden,
            position_id,
            sidecar_perf_context,
            kv_cache_only,
            terminal_hidden_buffer_id,
            defer_final_sync);
    }

    bool DeviceGraphOrchestrator::executeMTPDepth0Batched(
        const int32_t *draft_condition_tokens,
        int token_count,
        TensorBase *terminal_hidden,
        int position_id,
        const char *sidecar_perf_context,
        bool kv_cache_only,
        BufferId terminal_hidden_buffer_id,
        bool defer_final_sync,
        const void *draft_condition_tokens_device,
        int draft_condition_ready_slot,
        bool draft_condition_ready_is_target)
    {
        const std::string device_key = state_.device_id.toString();
        const std::string phase = perfPhaseName();
        const std::string sidecar_context =
            (sidecar_perf_context && sidecar_perf_context[0] != '\0')
                ? sidecar_perf_context
                : ((phase == "prefill") ? "mtp_shifted_prefill" : "mtp_decode_sidecar");
        const bool external_device_condition_tokens = draft_condition_tokens_device != nullptr;
        /*
         * GPU graph replay must not depend on host token storage.  The
         * embedding kernels can read token IDs from a persistent device
         * pointer, so even ordinary greedy host tokens are first staged into
         * the arena-owned MTP_CONDITION_TOKEN buffer.  That keeps the captured
         * graph shape stable while allowing each replay to observe new token
         * values.
         */
        const bool stage_host_condition_tokens_on_device =
            state_.device_id.is_gpu() && !external_device_condition_tokens;
        const bool use_device_condition_tokens =
            external_device_condition_tokens || stage_host_condition_tokens_on_device;
        if ((!draft_condition_tokens && !external_device_condition_tokens) ||
            token_count <= 0 || token_count > 4)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar received invalid token_count=" << token_count);
            return false;
        }
        if (external_device_condition_tokens && token_count != 1)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] External device-token MTP sidecar currently supports one token per replay");
            return false;
        }
        if (!kv_cache_only && token_count != 1)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batched MTP sidecar is only supported for kv_cache_only catchup");
            return false;
        }
        if (draft_condition_tokens)
        {
            const int vocab_size = graph_builder_
                                       ? graph_builder_->config().vocab_size
                                       : 0;
            for (int i = 0; i < token_count; ++i)
            {
                const int32_t token = draft_condition_tokens[i];
                if (token < 0 || (vocab_size > 0 && token >= vocab_size))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar received invalid host token="
                              << token << " at row=" << i
                              << " seq_len=" << token_count
                              << " context=" << sidecar_context
                              << " vocab=" << vocab_size);
                    return false;
                }
            }
        }
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "sidecar_depth0_total",
            phase,
            device_key,
            {{"context", sidecar_context},
             {"depth", "0"},
             {"device_tokens", boolTag(use_device_condition_tokens)},
             {"kv_cache_only", boolTag(kv_cache_only)},
             {"seq_len", std::to_string(token_count)}});
        PerfStatsCollector::addCounter(
            "mtp",
            "sidecar_depth0_calls",
            1.0,
            phase,
            device_key,
            {{"context", sidecar_context},
             {"depth", "0"},
             {"device_tokens", boolTag(use_device_condition_tokens)},
             {"kv_cache_only", boolTag(kv_cache_only)},
             {"seq_len", std::to_string(token_count)}});
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return false;
        if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar requires an initialized MTP KV cache");
            return false;
        }
        if (!frozen_weight_set_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar requires frozen MTP weight bindings");
            return false;
        }

        ModelWeightBindings bindings;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sidecar_resolve_weight_bindings",
                phase,
                device_key,
                {{"depth", "0"}});
            bindings = makeModelWeightBindings(*frozen_weight_set_);
        }
        if (bindings.mtp.empty() || bindings.mtp.depths.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar requested without MTP weight bindings");
            return false;
        }
        const auto &depth0_bindings = bindings.mtp.depths[0];
        const bool mtp_moe_sidecar =
            depth0_bindings.fa_block.moe_gate ||
            depth0_bindings.fa_block.moe_gate_exps ||
            depth0_bindings.fa_block.moe_up_exps ||
            depth0_bindings.fa_block.moe_down_exps;

        auto get_extension = [&](BufferId id) -> TensorBase *
        {
            auto it = state_.extension_buffers.find(id);
            return it == state_.extension_buffers.end() ? nullptr : it->second.get();
        };

        TensorBase *mtp_logits = get_extension(BufferId::MTP_LOGITS);
        TensorBase *mtp_hidden = get_extension(BufferId::MTP_HIDDEN);
        TensorBase *mtp_embedding = get_extension(BufferId::MTP_EMBEDDING);
        TensorBase *mtp_norm_hidden = get_extension(BufferId::MTP_NORM_HIDDEN);
        TensorBase *mtp_norm_embedding = get_extension(BufferId::MTP_NORM_EMBEDDING);
        TensorBase *mtp_concat = get_extension(BufferId::MTP_CONCAT);
        TensorBase *mtp_projected = get_extension(BufferId::MTP_PROJECTED);
        TensorBase *mtp_q = get_extension(BufferId::MTP_Q_PROJ);
        TensorBase *mtp_k = get_extension(BufferId::MTP_K_PROJ);
        TensorBase *mtp_v = get_extension(BufferId::MTP_V_PROJ);
        TensorBase *mtp_q_raw = get_extension(BufferId::MTP_FA_Q_RAW);
        TensorBase *mtp_q_gate = get_extension(BufferId::MTP_FA_GATE);
        TensorBase *mtp_attn_output = get_extension(BufferId::MTP_ATTN_OUTPUT);
        TensorBase *mtp_attn_proj = get_extension(BufferId::MTP_ATTN_PROJ);
        TensorBase *mtp_gate = get_extension(BufferId::MTP_GATE_PROJ);
        TensorBase *mtp_up = get_extension(BufferId::MTP_UP_PROJ);
        TensorBase *mtp_ffn_output = get_extension(BufferId::MTP_FFN_OUTPUT);
        TensorBase *mtp_moe_expert_indices = get_extension(BufferId::MOE_EXPERT_INDICES);
        TensorBase *mtp_moe_expert_weights = get_extension(BufferId::MOE_EXPERT_WEIGHTS);
        TensorBase *mtp_moe_combined_output = get_extension(BufferId::MOE_COMBINED_OUTPUT);
        TensorBase *mtp_moe_shared_expert_output = get_extension(BufferId::MOE_SHARED_EXPERT_OUTPUT);
        TensorBase *mtp_moe_gate_scratch = get_extension(BufferId::MOE_GATE_SCRATCH);
        TensorBase *mtp_moe_up_scratch = get_extension(BufferId::MOE_UP_SCRATCH);

        const bool missing_common_buffers =
            !terminal_hidden ||
            !mtp_embedding || !mtp_norm_hidden || !mtp_norm_embedding ||
            !mtp_concat || !mtp_projected ||
            !mtp_q || !mtp_k || !mtp_v || !mtp_q_raw || !mtp_q_gate;
        const bool missing_full_sidecar_buffers =
            !mtp_logits || !mtp_hidden ||
            !mtp_attn_output || !mtp_attn_proj || !mtp_gate || !mtp_up ||
            !mtp_ffn_output;
        if (missing_common_buffers || (!kv_cache_only && missing_full_sidecar_buffers))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar missing required buffers");
            return false;
        }
        if (!kv_cache_only && mtp_moe_sidecar &&
            (!mtp_moe_expert_indices || !mtp_moe_expert_weights ||
             !mtp_moe_combined_output || !mtp_moe_shared_expert_output ||
             !mtp_moe_gate_scratch || !mtp_moe_up_scratch))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MoE MTP sidecar missing required MoE scratch buffers");
            return false;
        }
        auto require_rows = [&](const char *name, const TensorBase *tensor) -> bool
        {
            if (!tensor)
                return true;
            if (tensor->rows() >= static_cast<size_t>(token_count))
                return true;
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar buffer " << name
                                                                      << " has only " << tensor->rows()
                                                                      << " rows for seq_len=" << token_count);
            return false;
        };
        if (!require_rows("terminal_hidden", terminal_hidden) ||
            !require_rows("mtp_embedding", mtp_embedding) ||
            !require_rows("mtp_norm_hidden", mtp_norm_hidden) ||
            !require_rows("mtp_norm_embedding", mtp_norm_embedding) ||
            !require_rows("mtp_concat", mtp_concat) ||
            !require_rows("mtp_projected", mtp_projected) ||
            !require_rows("mtp_q", mtp_q) ||
            !require_rows("mtp_k", mtp_k) ||
            !require_rows("mtp_v", mtp_v) ||
            !require_rows("mtp_q_raw", mtp_q_raw) ||
            !require_rows("mtp_q_gate", mtp_q_gate) ||
            (!kv_cache_only &&
             (!require_rows("mtp_logits", mtp_logits) ||
              !require_rows("mtp_hidden", mtp_hidden) ||
              !require_rows("mtp_attn_output", mtp_attn_output) ||
              !require_rows("mtp_attn_proj", mtp_attn_proj) ||
              !require_rows("mtp_gate", mtp_gate) ||
              !require_rows("mtp_up", mtp_up) ||
              !require_rows("mtp_ffn_output", mtp_ffn_output))))
        {
            return false;
        }

        MTPForwardOutput output;
        output.logits = mtp_logits;
        output.hidden = mtp_hidden;
        output.embedding = mtp_embedding;
        output.norm_hidden = mtp_norm_hidden;
        output.norm_embedding = mtp_norm_embedding;
        output.concat = mtp_concat;
        output.projected = mtp_projected;
        output.q = mtp_q;
        output.k = mtp_k;
        output.v = mtp_v;
        output.q_raw = mtp_q_raw;
        output.q_gate = mtp_q_gate;
        output.attn_output = mtp_attn_output;
        output.attn_proj = mtp_attn_proj;
        output.gate = mtp_gate;
        output.up = mtp_up;
        output.ffn_output = mtp_ffn_output;
        output.moe_expert_indices = mtp_moe_expert_indices;
        output.moe_expert_weights = mtp_moe_expert_weights;
        output.moe_combined_output = mtp_moe_combined_output;
        output.moe_shared_expert_output = mtp_moe_shared_expert_output;
        output.moe_gate_scratch = mtp_moe_gate_scratch;
        output.moe_up_scratch = mtp_moe_up_scratch;

        auto &sidecar_cache = [&]() -> MTPSidecarGraphCache &
        {
            if (kv_cache_only)
            {
                if (token_count == 1)
                {
                    return use_device_condition_tokens
                               ? mtp_sidecar_depth0_kv_only_device_token_cache_
                               : mtp_sidecar_depth0_kv_only_cache_;
                }
                return mtp_sidecar_depth0_kv_only_batch_caches_[static_cast<size_t>(token_count)];
            }

            const bool chained_hidden = terminal_hidden_buffer_id == BufferId::MTP_HIDDEN;
            if (chained_hidden)
            {
                return use_device_condition_tokens
                           ? mtp_sidecar_depth0_chained_device_token_cache_
                           : mtp_sidecar_depth0_chained_cache_;
            }
            return use_device_condition_tokens
                       ? mtp_sidecar_depth0_device_token_cache_
                       : mtp_sidecar_depth0_cache_;
        }();

        auto sidecar_condition_token_slot = [&](const MTPSidecarGraphCache &cache) -> int
        {
            if (&cache == &mtp_sidecar_depth0_device_token_cache_)
                return 0;
            if (&cache == &mtp_sidecar_depth0_chained_device_token_cache_)
                return 1;
            if (&cache == &mtp_sidecar_depth0_kv_only_device_token_cache_)
                return 2;
            for (size_t i = 0; i < mtp_sidecar_depth0_kv_only_batch_caches_.size(); ++i)
            {
                if (&cache == &mtp_sidecar_depth0_kv_only_batch_caches_[i])
                    return 3 + static_cast<int>(i);
            }
            return -1;
        };

        const int condition_token_slot =
            use_device_condition_tokens ? sidecar_condition_token_slot(sidecar_cache) : -1;
        int32_t *condition_token_device = nullptr;
        if (use_device_condition_tokens)
        {
            constexpr int kConditionTokenSlotWidth =
                static_cast<int>(sampling_math::kSpeculativeBatchMaxRows);
            const int required_capacity =
                (condition_token_slot + 1) * kConditionTokenSlotWidth;
            if (condition_token_slot < 0 ||
                !mtp_sidecar_condition_token_dev_ ||
                mtp_sidecar_condition_token_capacity_ < required_capacity ||
                token_count > kConditionTokenSlotWidth)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-token MTP sidecar requires a role-owned"
                          << " MTP_CONDITION_TOKEN slot: slot=" << condition_token_slot
                          << " capacity=" << mtp_sidecar_condition_token_capacity_
                          << " required=" << required_capacity
                          << " requested_rows=" << token_count
                          << " context=" << sidecar_context);
                return false;
            }
            condition_token_device =
                static_cast<int32_t *>(mtp_sidecar_condition_token_dev_) +
                condition_token_slot * kConditionTokenSlotWidth;
        }

        MTPForwardInput input;
        input.draft_token_ids = draft_condition_tokens;
        input.draft_token_ids_device =
            use_device_condition_tokens ? condition_token_device : nullptr;
        input.terminal_hidden = terminal_hidden;
        input.kv_cache = state_.mtp_kv_caches[0].get();
        input.position_ids = &position_id;
        input.sequence_lengths = nullptr;
        input.batch_size = 1;
        input.seq_len = token_count;
        input.device = state_.device_id;
        input.terminal_hidden_buffer_id = terminal_hidden_buffer_id;
        input.kv_cache_only = kv_cache_only;

        const uint64_t current_moe_placement_epoch = moePlacementEpoch();
        const bool sidecar_moe_epoch_sensitive = mtp_moe_sidecar && !kv_cache_only;
        const uint64_t sidecar_moe_epoch_key =
            sidecar_moe_epoch_sensitive ? current_moe_placement_epoch : 0;
        const bool needs_graph_rebuild =
            !sidecar_cache.valid ||
            !sidecar_cache.graph ||
            sidecar_cache.terminal_hidden != terminal_hidden ||
            sidecar_cache.seq_len != token_count ||
            sidecar_cache.uses_device_token_ids != use_device_condition_tokens ||
            sidecar_cache.condition_token_slot != condition_token_slot ||
            sidecar_cache.condition_token_device != condition_token_device ||
            sidecar_cache.moe_epoch_sensitive != sidecar_moe_epoch_sensitive ||
            sidecar_cache.moe_placement_epoch != sidecar_moe_epoch_key;

        const bool rebuilt_graph = needs_graph_rebuild;
        if (needs_graph_rebuild)
        {
            sidecar_cache.invalidate();
            sidecar_cache.token_id =
                use_device_condition_tokens ? -1 : draft_condition_tokens[0];
            sidecar_cache.token_ids.assign(static_cast<size_t>(token_count), 0);
            sidecar_cache.position_ids.assign(static_cast<size_t>(token_count), 0);
            sidecar_cache.position_id = position_id;
            sidecar_cache.seq_len = token_count;
            sidecar_cache.uses_device_token_ids = use_device_condition_tokens;
            sidecar_cache.condition_token_slot = condition_token_slot;
            sidecar_cache.condition_token_device = condition_token_device;
            sidecar_cache.terminal_hidden = terminal_hidden;
            sidecar_cache.moe_placement_epoch = sidecar_moe_epoch_key;
            sidecar_cache.moe_epoch_sensitive = sidecar_moe_epoch_sensitive;

            MTPForwardInput cached_input = input;
            cached_input.draft_token_ids = sidecar_cache.token_ids.data();
            cached_input.draft_token_ids_device =
                use_device_condition_tokens ? condition_token_device : nullptr;
            cached_input.position_ids = sidecar_cache.position_ids.data();

            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sidecar_build_graph",
                phase,
                device_key,
                {{"depth", "0"}, {"seq_len", std::to_string(token_count)}});
            ComputeGraph graph = graph_builder_->buildMTPGraph(
                0,
                bindings.mtp.depths[0],
                cached_input,
                output);
            if (graph.size() == 0)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to build MTP sidecar graph");
                return false;
            }

            sidecar_cache.graph = std::make_unique<ComputeGraph>(std::move(graph));
            sidecar_cache.dynamic_param_stages.clear();
            sidecar_cache.collective_nodes.clear();
            for (const auto &node_name : sidecar_cache.graph->getExecutionOrder())
            {
                ComputeNode *node = sidecar_cache.graph->getNode(node_name);
                if (!node || !node->stage)
                    continue;
                if (node->stage->hasDynamicParams())
                    sidecar_cache.dynamic_param_stages.push_back(node->stage.get());

                const ComputeStageType type = node->stage->type();
                if (type == ComputeStageType::ALLREDUCE ||
                    type == ComputeStageType::ALLGATHER ||
                    type == ComputeStageType::ALLGATHER_V)
                {
                    sidecar_cache.collective_nodes.insert(node_name);
                }
            }
            sidecar_cache.valid = true;
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_collective_node_scans",
                1.0,
                phase,
                device_key,
                {{"depth", "0"},
                 {"seq_len", std::to_string(token_count)},
                 {"has_collectives", boolTag(!sidecar_cache.collective_nodes.empty())},
                 {"node_count", std::to_string(sidecar_cache.collective_nodes.size())}});
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_graph_cache_misses",
                1.0,
                phase,
                device_key,
                {{"context", sidecar_context},
                 {"depth", "0"},
                 {"device_tokens", boolTag(use_device_condition_tokens)},
                 {"kv_cache_only", boolTag(kv_cache_only)},
                 {"seq_len", std::to_string(token_count)},
                 {"moe_placement_epoch", std::to_string(sidecar_moe_epoch_key)}});
        }
        else
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_graph_cache_hits",
                1.0,
                phase,
                device_key,
                {{"context", sidecar_context},
                 {"depth", "0"},
                 {"device_tokens", boolTag(use_device_condition_tokens)},
                 {"kv_cache_only", boolTag(kv_cache_only)},
                 {"seq_len", std::to_string(token_count)},
                 {"moe_placement_epoch", std::to_string(sidecar_moe_epoch_key)}});
        }

        IDeviceContext *ctx = getDeviceContext(state_.device_id);
        if (!ctx)
            return false;

        IWorkerGPUContext *sidecar_gpu_ctx = nullptr;
        void *sidecar_dynamic_stream = nullptr;
        const bool try_gpu_graph_capture =
            state_.device_id.is_gpu() &&
            debugEnv().execution.gpu_graphs;
        if (state_.device_id.is_gpu())
        {
            auto &pool = GPUDeviceContextPool::instance();
            sidecar_gpu_ctx = &pool.getContext(state_.device_id);
            sidecar_dynamic_stream = sidecar_gpu_ctx->defaultStream();
            if (!sidecar_dynamic_stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar requires an explicit non-null GPU stream for "
                          << state_.device_id.toString());
                return false;
            }
        }
        if (try_gpu_graph_capture && !rebuilt_graph)
        {
            if (sidecar_cache.segment_cache.ensureCaptureStream(
                    sidecar_gpu_ctx,
                    state_.device_id))
            {
                void *capture_stream = sidecar_cache.segment_cache.capture_stream;
                sidecar_dynamic_stream = capture_stream;
            }
            else
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create MTP sidecar graph capture stream");
                return false;
            }
        }
        if (state_.device_id.is_gpu() &&
            !waitForPendingShiftedMTPKVReady(
                sidecar_dynamic_stream,
                "mtp_sidecar_before_execute"))
        {
            return false;
        }
        /*
         * Accepted-state publication is queued by the verifier graph, but the
         * next speculative step usually begins with a sidecar forward rather
         * than a main-model forward.  Make that first live-state reader wait on
         * the publication event so shifted KV, main KV, GDN, and short-conv
         * state are observed as one atomic commit.
         */
        if (state_.device_id.is_gpu() &&
            !waitForPendingAcceptedSpecPublicationReady(
                sidecar_dynamic_stream,
                "mtp_sidecar_before_execute"))
        {
            return false;
        }
        if (stage_host_condition_tokens_on_device)
        {
            if (sidecar_cache.token_ids.size() != static_cast<size_t>(token_count))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar cache has unstable staged token storage for seq_len="
                          << token_count);
                return false;
            }
            std::copy(draft_condition_tokens,
                      draft_condition_tokens + token_count,
                      sidecar_cache.token_ids.begin());
        }
        if (use_device_condition_tokens)
        {
            if (!sidecar_dynamic_stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-token MTP sidecar requires an explicit non-null stream");
                return false;
            }
            if (external_device_condition_tokens &&
                draft_condition_ready_slot >= 0 &&
                !(draft_condition_ready_is_target
                      ? waitForRequiredStochasticTargetSampleReady(
                            draft_condition_ready_slot,
                            sidecar_dynamic_stream,
                            "mtp_sidecar_device_target_token")
                      : waitForRequiredStochasticDraftSampleReadyRange(
                            draft_condition_ready_slot,
                            1,
                            sidecar_dynamic_stream,
                            "mtp_sidecar_device_token")))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to order MTP sidecar token copy after deferred sample");
                return false;
            }
            IBackend *backend = getBackendFor(state_.device_id);
            if (!backend)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to resolve backend for MTP sidecar token staging");
                return false;
            }

            const size_t staged_token_bytes =
                sizeof(int32_t) * static_cast<size_t>(token_count);
            bool staged_tokens_ok = false;
            if (external_device_condition_tokens)
            {
                /*
                 * Keep the sidecar graph pointer stable by copying the
                 * previous sampler's token slot into an arena-owned condition
                 * buffer.  The D2D copy is ordered on the same explicit stream
                 * as the following embedding kernel.
                 */
                staged_tokens_ok = backend->deviceCopyAsync(
                    condition_token_device,
                    draft_condition_tokens_device,
                    staged_token_bytes,
                    state_.device_id.gpu_ordinal(),
                    sidecar_dynamic_stream);
            }
            else
            {
                /*
                 * Greedy and catch-up paths still start with host scalar
                 * tokens, but captured graphs must never read host token
                 * memory.  Stage those scalars into the same persistent device
                 * buffer before dynamic params are updated.
                 */
                staged_tokens_ok = backend->hostToDeviceOnStream(
                    condition_token_device,
                    sidecar_cache.token_ids.data(),
                    staged_token_bytes,
                    state_.device_id.gpu_ordinal(),
                    sidecar_dynamic_stream);
            }

            if (!staged_tokens_ok)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to stage MTP sidecar token rows"
                          << " source=" << (external_device_condition_tokens ? "device" : "host")
                          << " rows=" << token_count);
                return false;
            }

            if (debugEnv().validation.validate_buffers)
            {
                std::array<int32_t, 4> staged_tokens{};
                if (!backend->deviceToHostFast(
                        staged_tokens.data(),
                        condition_token_device,
                        staged_token_bytes,
                        state_.device_id.gpu_ordinal(),
                        sidecar_dynamic_stream))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to validate staged device MTP sidecar tokens");
                    return false;
                }
                const int vocab_size = graph_builder_
                                           ? graph_builder_->config().vocab_size
                                           : 0;
                for (int i = 0; i < token_count; ++i)
                {
                    const int32_t staged_token = staged_tokens[static_cast<size_t>(i)];
                    if (staged_token < 0 ||
                        (vocab_size > 0 && staged_token >= vocab_size))
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Staged device MTP sidecar token is out of range: token="
                                  << staged_token << " row=" << i
                                  << " vocab=" << vocab_size);
                        return false;
                    }
                    if (staged_token == 0)
                    {
                        LOG_WARN("[DeviceGraphOrchestrator] Staged device MTP sidecar token is zero for context="
                                 << sidecar_context << " row=" << i);
                    }
                }
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "mtp_sidecar_device_token_inputs",
                1.0,
                phase,
                device_key,
                {{"seq_len", std::to_string(token_count)},
                 {"slot", std::to_string(condition_token_slot)},
                 {"source", external_device_condition_tokens ? "device" : "host"}});
        }

        const bool sidecar_uses_gpu_workspace =
            state_.device_id.is_gpu() && sidecar_cache.graph;
        const uint64_t current_workspace_generation =
            sidecar_uses_gpu_workspace ? workspaceGeneration(state_.device_id) : 0;
        const bool sidecar_workspace_validated =
            sidecar_uses_gpu_workspace &&
            sidecar_cache.workspace_generation != 0 &&
            current_workspace_generation == sidecar_cache.workspace_generation;

        if (!sidecar_workspace_validated &&
            !ensureDeviceWorkspaceAllocated(*sidecar_cache.graph))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to allocate MTP sidecar workspace before dynamic param update");
            return false;
        }
        if (sidecar_uses_gpu_workspace && !sidecar_workspace_validated)
        {
            const uint64_t new_workspace_generation = workspaceGeneration(state_.device_id);
            if (new_workspace_generation != sidecar_cache.workspace_generation)
            {
                if (sidecar_cache.workspace_generation != 0)
                {
                    LOG_DEBUG("[DeviceGraphOrchestrator] MTP sidecar workspace generation changed on "
                              << state_.device_id.toString()
                              << " from " << sidecar_cache.workspace_generation
                              << " to " << new_workspace_generation
                              << "; dropping captured sidecar replay state");
                    sidecar_cache.resetReplayStateAfterWorkspaceRebind();
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "sidecar_workspace_rebinds",
                        1.0,
                        phase,
                        device_key,
                        {{"depth", "0"}, {"seq_len", std::to_string(token_count)}});
                }
                sidecar_cache.workspace_generation = new_workspace_generation;
            }
        }

        sidecar_cache.token_id =
            use_device_condition_tokens ? -1 : draft_condition_tokens[0];
        if (sidecar_cache.token_ids.size() != static_cast<size_t>(token_count) ||
            sidecar_cache.position_ids.size() != static_cast<size_t>(token_count))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar cache has unstable token/position storage for seq_len="
                      << token_count);
            return false;
        }
        if (external_device_condition_tokens)
        {
            std::fill(sidecar_cache.token_ids.begin(), sidecar_cache.token_ids.end(), 0);
        }
        else if (stage_host_condition_tokens_on_device)
        {
            std::copy(draft_condition_tokens,
                      draft_condition_tokens + token_count,
                      sidecar_cache.token_ids.begin());
        }
        else
        {
            std::copy(draft_condition_tokens,
                      draft_condition_tokens + token_count,
                      sidecar_cache.token_ids.begin());
        }
        sidecar_cache.position_id = position_id;
        for (int i = 0; i < token_count; ++i)
            sidecar_cache.position_ids[static_cast<size_t>(i)] = position_id + i;
        if (sidecar_dynamic_stream)
        {
            std::string stream_binding_error;
            if (!mtp_sidecar::bindStagesToCaptureStream(*sidecar_cache.graph,
                                                         sidecar_dynamic_stream,
                                                         &stream_binding_error))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to bind MTP sidecar stages to explicit stream: "
                          << stream_binding_error);
                return false;
            }
        }
        for (auto *stage : sidecar_cache.dynamic_param_stages)
        {
            if (stage)
                stage->updateDynamicParams(position_id, token_count);
        }
        sidecar_cache.graph->reset();

        const bool has_sidecar_collectives = !sidecar_cache.collective_nodes.empty();

        bool ok = false;
        bool used_capture_policy = false;
        bool deferred_kv_completion = false;
        {
            ScopedStringOverride snapshot_scope(snapshot_context_, sidecar_context);
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sidecar_execute_graph",
                phase,
                device_key,
                {{"context", sidecar_context},
                 {"depth", "0"},
                 {"device_tokens", boolTag(use_device_condition_tokens)},
                 {"kv_cache_only", boolTag(kv_cache_only)},
                 {"seq_len", std::to_string(token_count)}});

            bool used_segmented_capture = false;
            const bool can_defer_sidecar_sync =
                defer_final_sync &&
                try_gpu_graph_capture &&
                !rebuilt_graph &&
                !has_sidecar_collectives &&
                !debugEnv().gpu_stage_timing &&
                !debugEnv().gpu_stage_timing_detail;
            clearPendingLogitsStream(
                PendingLogitsStreamRole::MTPSidecar,
                "executeMTPDepth0_begin");
            if (try_gpu_graph_capture && !rebuilt_graph)
            {
                used_capture_policy = true;
                if (!sidecar_gpu_ctx)
                {
                    auto &pool = GPUDeviceContextPool::instance();
                    sidecar_gpu_ctx = &pool.getContext(state_.device_id);
                }
                sidecar_cache.segment_cache.perf_context = sidecar_context;
                auto capture_policy = buildDecodeCapturePolicy(
                    has_sidecar_collectives,
                    ctx,
                    sidecar_cache.segment_cache.consecutive_failures);
                capture_policy.defer_final_sync = can_defer_sidecar_sync;
                PerfStatsCollector::addCounter(
                    "mtp",
                    "sidecar_decode_capture_policy",
                    1.0,
                    phase,
                    device_key,
                    {{"context", sidecar_cache.segment_cache.perf_context},
                     {"seq_len", std::to_string(token_count)},
                     {"allow_segmented", boolTag(capture_policy.allow_segmented_capture)},
                     {"force_recapture", boolTag(capture_policy.force_recapture)},
                     {"defer_final_sync", boolTag(capture_policy.defer_final_sync)},
                     {"has_collectives", boolTag(has_sidecar_collectives)},
                     {"collective_segmented", boolTag(capture_policy.collective_segmented_enabled)},
                     {"collectives_graph_capturable", boolTag(capture_policy.collectives_graph_capturable)}});
                ok = executor_.executeDecodeWithCapturePolicy(
                    *sidecar_cache.graph,
                    ctx,
                    &sidecar_cache.segment_cache,
                    sidecar_gpu_ctx->defaultStream(),
                    sidecar_gpu_ctx,
                    has_sidecar_collectives ? &sidecar_cache.collective_nodes : nullptr,
                    capture_policy,
                    &used_segmented_capture);
                deferred_kv_completion =
                    ok && used_segmented_capture &&
                    kv_cache_only && capture_policy.defer_final_sync;
                if (!kv_cache_only)
                {
                    void *sample_stream = nullptr;
                    std::string deferred_stream_error;
                    if (!mtp_sidecar::deferredSamplingStream(
                            ok,
                            capture_policy.defer_final_sync,
                            sidecar_cache.segment_cache.capture_stream,
                            &sample_stream,
                            &deferred_stream_error))
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] " << deferred_stream_error);
                        return false;
                    }
                    if (sample_stream)
                    {
                        // Full sidecar replay produces logits for the sampler.
                        // KV-only replay produces shifted MTP KV instead, so it
                        // must use the shifted-KV readiness event below rather
                        // than publishing an unrelated logits handoff.
                        publishPendingLogitsStream(
                            PendingLogitsStreamRole::MTPSidecar,
                            sample_stream,
                            "executeMTPDepth0_deferred_sampling_stream");
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "sidecar_forward_sample_sync_fusions",
                            1.0,
                            phase,
                            device_key,
                            {{"context", sidecar_context},
                             {"seq_len", std::to_string(token_count)}});
                    }
                }
            }
            else
            {
                ok = execute(*sidecar_cache.graph, ctx);
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_graph_capture_path",
                1.0,
                phase,
                device_key,
                {{"context", sidecar_context},
                 {"depth", "0"},
                 {"device_tokens", boolTag(use_device_condition_tokens)},
                 {"kv_cache_only", boolTag(kv_cache_only)},
                 {"seq_len", std::to_string(token_count)},
                 {"path", used_segmented_capture ? "segmented" : (rebuilt_graph ? "plain_after_build" : "plain")}});
        }
        const bool plain_sidecar_execution = !used_capture_policy || rebuilt_graph;
        if (ok &&
            state_.device_id.is_gpu() &&
            sidecar_dynamic_stream &&
            !kv_cache_only &&
            defer_final_sync &&
            plain_sidecar_execution &&
            !peekPendingLogitsStream(PendingLogitsStreamRole::MTPSidecar))
        {
            /*
             * The first use of a sidecar graph executes as an ordinary graph
             * before segmented replay exists.  It is still stream ordered: every
             * stage was bound to sidecar_dynamic_stream above.  Publish that
             * stream instead of synchronizing here, so the immediate sampler or
             * distribution-builder becomes the synchronization point.
             */
            publishPendingLogitsStream(
                PendingLogitsStreamRole::MTPSidecar,
                sidecar_dynamic_stream,
                "executeMTPDepth0_plain_deferred_sampling_stream");
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_plain_stream_handoffs",
                1.0,
                phase,
                device_key,
                {{"context", sidecar_context},
                 {"seq_len", std::to_string(token_count)}});
        }
        if (ok && state_.device_id.is_gpu() && sidecar_dynamic_stream &&
            !peekPendingLogitsStream(PendingLogitsStreamRole::MTPSidecar))
        {
            const bool needs_sidecar_completion_before_return =
                (kv_cache_only && !deferred_kv_completion) ||
                plain_sidecar_execution;
            if (needs_sidecar_completion_before_return)
            {
                if (!sidecar_gpu_ctx)
                {
                    auto &pool = GPUDeviceContextPool::instance();
                    sidecar_gpu_ctx = &pool.getContext(state_.device_id);
                }
                if (!sidecar_gpu_ctx->synchronizeStreamChecked(sidecar_dynamic_stream))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] MTP sidecar stream synchronization failed after "
                              << sidecar_context << " on " << state_.device_id.toString());
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "sidecar_explicit_stream_completions",
                    1.0,
                    phase,
                    device_key,
                    {{"context", sidecar_context},
                     {"seq_len", std::to_string(token_count)},
                     {"kv_cache_only", boolTag(kv_cache_only)},
                     {"path", plain_sidecar_execution ? "plain" : "capture"}});
            }
            else if (kv_cache_only)
            {
                if (!recordShiftedMTPKVReady(
                        sidecar_dynamic_stream,
                        sidecar_context.c_str()))
                {
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "shifted_mtp_kv_stream_syncs_deferred",
                    1.0,
                    phase,
                    device_key,
                    {{"context", sidecar_context},
                     {"seq_len", std::to_string(token_count)}});
            }
        }
        if (ok && state_.device_id.is_gpu() &&
            PerfStatsCollector::gpuStageEventTimingEnabled())
        {
            auto &timeline = executor_.stageTimeline();
            if (timeline.isInitialized())
            {
                // MTP sidecar graph capture/replay can leave a timeline initialized
                // without fresh events. Clear it here so stale warmup events are never queried.
                if (suppress_timeline_ || !timeline.hasValidRecords())
                {
                    timeline.resetTimings();
                    return ok;
                }

                auto &pool = GPUDeviceContextPool::instance();
                auto &gpu_ctx = pool.getContext(state_.device_id);
                timeline.collect(&gpu_ctx);
                const std::string mtp_phase = "mtp_" + phase;
                timeline.recordPerfStats(
                    mtp_phase.c_str(),
                    device_key.c_str(),
                    "mtp_stage_gpu",
                    {{"context", sidecar_context},
                     {"depth", "0"}});
                timeline.resetTimings();
            }
        }
        return ok;
    }

    bool DeviceGraphOrchestrator::populateMTPShiftedCacheFromPrefill(
        const int *tokens,
        int seq_len,
        int batch_size,
        int position_offset)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled || state_.mtp_kv_caches.empty())
            return true;
        if (!tokens || seq_len <= 0)
            return false;

        if (batch_size != 1)
        {
            updateMTPShiftedCacheMetadata(batch_size);
            return true;
        }

        if (!frozen_weight_set_)
        {
            updateMTPShiftedCacheMetadata(batch_size);
            return true;
        }

        auto bindings = makeModelWeightBindings(*frozen_weight_set_);
        if (bindings.mtp.empty() || bindings.mtp.depths.empty())
        {
            updateMTPShiftedCacheMetadata(batch_size);
            return true;
        }

        const int previous_tokens = std::max(0, position_offset);
        auto &cache = state_.mtp_kv_caches[0];
        if (!cache)
            return true;

        const int current_mtp_tokens = cache->get_cached_tokens(cache->first_layer_index(), 0);
        const int expected_previous_mtp_tokens = std::max(0, previous_tokens - 1);
        if (current_mtp_tokens != expected_previous_mtp_tokens)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] MTP shifted prefill payload replay skipped: "
                      "cache_count=" << current_mtp_tokens
                                     << " expected_previous=" << expected_previous_mtp_tokens);
            updateMTPShiftedCacheMetadata(batch_size);
            return true;
        }

        if (previous_tokens > 0 &&
            state_.prefix_terminal_hidden &&
            state_.mtp_terminal_hidden_current)
        {
            if (!executeMTPDepth0(tokens[0],
                                  state_.prefix_terminal_hidden.get(),
                                  previous_tokens,
                                  "mtp_shifted_prefill",
                                  true))
            {
                return false;
            }
        }

        std::array<int32_t, 4> token_batch{};
        for (int row = 0; row + 1 < seq_len;)
        {
            const int remaining_rows = (seq_len - 1) - row;
            const int batch_rows = std::min<int>(
                static_cast<int>(token_batch.size()),
                remaining_rows);
            if (!selectMTPTerminalHiddenRows(row, batch_rows, seq_len))
                return false;
            for (int i = 0; i < batch_rows; ++i)
                token_batch[static_cast<size_t>(i)] = static_cast<int32_t>(tokens[row + 1 + i]);
            if (!executeMTPDepth0Batched(token_batch.data(),
                                         batch_rows,
                                         state_.prefix_terminal_hidden.get(),
                                         position_offset + row + 1,
                                         "mtp_shifted_prefill",
                                         true,
                                         BufferId::PREFIX_TERMINAL_HIDDEN))
            {
                return false;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "shifted_prefill_sidecar_batches",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"rows", std::to_string(batch_rows)}});
            row += batch_rows;
        }

        return true;
    }

    void DeviceGraphOrchestrator::updateMTPShiftedCacheMetadata(int active_batch_size)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled || state_.mtp_kv_caches.empty())
            return;

        void *stream = explicitGPUStreamForOperation("updateMTPShiftedCacheMetadata");
        if (state_.device_id.is_gpu() && !stream)
        {
            return;
        }
        if (state_.device_id.is_gpu() &&
            !waitForPendingShiftedMTPKVReady(
                stream,
                "update_mtp_shifted_cache_metadata"))
        {
            LOG_WARN("[DeviceGraphOrchestrator] Failed to order shifted-cache metadata update after deferred MTP KV append");
            return;
        }

        const int seq_count = std::min(active_batch_size, state_.batch_size);
        for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
        {
            auto &cache = state_.mtp_kv_caches[depth];
            if (!cache)
                continue;

            for (int seq = 0; seq < seq_count; ++seq)
            {
                if (!cache->truncateSequence(seq, 0, stream))
                {
                    LOG_WARN("[DeviceGraphOrchestrator] Failed to reset MTP shifted cache metadata for depth="
                             << depth << " seq=" << seq);
                    continue;
                }
            }

            for (int layer = 0; layer < cache->n_layers(); ++layer)
            {
                for (int seq = 0; seq < seq_count; ++seq)
                {
                    const int shifted_count = std::max(
                        0,
                        state_.positions[seq] - static_cast<int>(depth) - 1);
                    const int bounded_count = std::min(shifted_count, cache->max_seq_len());

                    // Phase 5 wires the request-local shifted state contract before
                    // decode consumes it. The sidecar execution slice will replace
                    // this metadata-only update with real MTP K/V appends.
                    if (bounded_count > 0)
                        cache->advanceHead(layer, seq, bounded_count);
                }
            }
        }
    }

    const float *DeviceGraphOrchestrator::logits() const
    {
        if (!state_.logits)
        {
            return nullptr;
        }
        return state_.logits->fp32_data();
    }

    bool DeviceGraphOrchestrator::forwardMTP(int32_t draft_condition_token)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTP requires an initialized MTP KV cache");
            return false;
        }
        if (!frozen_weight_set_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTP requires frozen MTP weight bindings");
            return false;
        }

        auto bindings = makeModelWeightBindings(*frozen_weight_set_);
        if (bindings.mtp.empty() || bindings.mtp.depths.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTP requested without MTP weight bindings");
            return false;
        }

        int position_id = state_.positions.empty() ? 0 : state_.positions[0];
        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "forwardMTP"))
            return false;
        return executeMTPDepth0(
            draft_condition_token,
            terminal_hidden,
            position_id,
            "mtp_decode_sidecar",
            false,
            terminal_hidden_buffer_id);
    }

    bool DeviceGraphOrchestrator::forwardMTPForDeviceSampling(int32_t draft_condition_token)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTPForDeviceSampling requires an initialized MTP KV cache");
            return false;
        }
        if (!frozen_weight_set_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTPForDeviceSampling requires frozen MTP weight bindings");
            return false;
        }

        auto bindings = makeModelWeightBindings(*frozen_weight_set_);
        if (bindings.mtp.empty() || bindings.mtp.depths.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTPForDeviceSampling requested without MTP weight bindings");
            return false;
        }

        const int position_id = state_.positions.empty() ? 0 : state_.positions[0];
        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "forwardMTPForDeviceSampling"))
            return false;

        /*
         * Request a deferred final sync. If graph capture, stage timing, or
         * collectives make that impossible, executeMTPDepth0 synchronizes before
         * returning and leaves no pending stream, preserving correctness.
         */
        return executeMTPDepth0(
            draft_condition_token,
            terminal_hidden,
            position_id,
            "mtp_decode_sidecar",
            false,
            terminal_hidden_buffer_id,
            /*defer_final_sync=*/true);
    }

    bool DeviceGraphOrchestrator::forwardMTPFromLastDraft(int32_t draft_condition_token, int position_id)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (position_id < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar requires a non-negative position_id");
            return false;
        }

        auto it = state_.extension_buffers.find(BufferId::MTP_HIDDEN);
        TensorBase *mtp_hidden =
            (it == state_.extension_buffers.end() || !it->second) ? nullptr : it->second.get();
        if (!mtp_hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar requires previous MTP hidden state");
            return false;
        }

        return executeMTPDepth0(
            draft_condition_token,
            mtp_hidden,
            position_id,
            "mtp_decode_sidecar_chain",
            false,
            BufferId::MTP_HIDDEN);
    }

    bool DeviceGraphOrchestrator::forwardMTPFromLastDraftForDeviceSampling(
        int32_t draft_condition_token,
        int position_id)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (position_id < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar device-sampling path requires a non-negative position_id");
            return false;
        }

        auto it = state_.extension_buffers.find(BufferId::MTP_HIDDEN);
        TensorBase *mtp_hidden =
            (it == state_.extension_buffers.end() || !it->second) ? nullptr : it->second.get();
        if (!mtp_hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar device-sampling path requires previous MTP hidden state");
            return false;
        }

        return executeMTPDepth0(
            draft_condition_token,
            mtp_hidden,
            position_id,
            "mtp_decode_sidecar_chain",
            false,
            BufferId::MTP_HIDDEN,
            /*defer_final_sync=*/true);
    }

    bool DeviceGraphOrchestrator::forwardMTPFromDeviceDraftForDeviceSampling(
        int draft_sample_slot,
        int position_id)
    {
        if (!supportsMTPDeviceDraftTokenInput())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token chained MTP sidecar is not supported by this runner");
            return false;
        }
        if (draft_sample_slot < 0 ||
            draft_sample_slot >= sampling_math::kSpeculativeBatchMaxRows)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token chained MTP sidecar received invalid draft slot="
                      << draft_sample_slot);
            return false;
        }
        if (position_id < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token chained MTP sidecar requires a non-negative position_id");
            return false;
        }

        auto it = state_.extension_buffers.find(BufferId::MTP_HIDDEN);
        TensorBase *mtp_hidden =
            (it == state_.extension_buffers.end() || !it->second) ? nullptr : it->second.get();
        if (!mtp_hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token chained MTP sidecar requires previous MTP hidden state");
            return false;
        }

        const int *draft_token_device =
            static_cast<const int *>(stochastic_draft_sample_tokens_dev_) + draft_sample_slot;
        return executeMTPDepth0Batched(
            /*draft_condition_tokens=*/nullptr,
            /*token_count=*/1,
            mtp_hidden,
            position_id,
            "mtp_decode_sidecar_chain_device_token",
            /*kv_cache_only=*/false,
            BufferId::MTP_HIDDEN,
            /*defer_final_sync=*/true,
            draft_token_device,
            draft_sample_slot);
    }

    bool DeviceGraphOrchestrator::forwardMTPFromDeviceTargetForDeviceSampling(
        int target_sample_slot,
        int position_id)
    {
        if (!supportsMTPDeviceDraftTokenInput())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token first MTP sidecar is not supported by this runner");
            return false;
        }
        if (target_sample_slot < 0 ||
            target_sample_slot >= sampling_math::kSpeculativeBatchMaxOutputTokens)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token first MTP sidecar received invalid target slot="
                      << target_sample_slot);
            return false;
        }
        if (position_id < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token first MTP sidecar requires a non-negative position_id");
            return false;
        }
        if (!stochastic_target_sample_tokens_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-token first MTP sidecar requires STOCHASTIC_TARGET_SAMPLE_TOKENS");
            return false;
        }

        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "forwardMTPFromDeviceTargetForDeviceSampling"))
            return false;

        const int *target_token_device =
            static_cast<const int *>(stochastic_target_sample_tokens_dev_) +
            target_sample_slot;
        return executeMTPDepth0Batched(
            /*draft_condition_tokens=*/nullptr,
            /*token_count=*/1,
            terminal_hidden,
            position_id,
            "mtp_decode_sidecar_device_target_token",
            /*kv_cache_only=*/false,
            terminal_hidden_buffer_id,
            /*defer_final_sync=*/true,
            target_token_device,
            target_sample_slot,
            /*draft_condition_ready_is_target=*/true);
    }

    bool DeviceGraphOrchestrator::forwardMTPAndSampleGreedy(
        int32_t draft_condition_token,
        int32_t *out_token)
    {
        if (!out_token)
            return false;
        *out_token = -1;
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTPAndSampleGreedy requires an initialized MTP KV cache");
            return false;
        }
        if (!frozen_weight_set_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forwardMTPAndSampleGreedy requires frozen MTP weight bindings");
            return false;
        }

        int position_id = state_.positions.empty() ? 0 : state_.positions[0];
        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "forwardMTPAndSampleGreedy"))
            return false;

        if (!executeMTPDepth0(
                draft_condition_token,
                terminal_hidden,
                position_id,
                "mtp_decode_sidecar",
                false,
                terminal_hidden_buffer_id,
                /*defer_final_sync=*/true))
        {
            return false;
        }

        const int token = sampleGreedyFromMTPLogitsOnDevice();
        if (token < 0)
            return false;
        *out_token = token;
        return true;
    }

    bool DeviceGraphOrchestrator::forwardMTPFromLastDraftAndSampleGreedy(
        int32_t draft_condition_token,
        int position_id,
        int32_t *out_token)
    {
        if (!out_token)
            return false;
        *out_token = -1;
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return false;
        }
        if (position_id < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar+sample requires a non-negative position_id");
            return false;
        }

        auto it = state_.extension_buffers.find(BufferId::MTP_HIDDEN);
        TensorBase *mtp_hidden =
            (it == state_.extension_buffers.end() || !it->second) ? nullptr : it->second.get();
        if (!mtp_hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Chained MTP sidecar+sample requires previous MTP hidden state");
            return false;
        }

        if (!executeMTPDepth0(
                draft_condition_token,
                mtp_hidden,
                position_id,
                "mtp_decode_sidecar_chain",
                false,
                BufferId::MTP_HIDDEN,
                /*defer_final_sync=*/true))
        {
            return false;
        }

        const int token = sampleGreedyFromMTPLogitsOnDevice();
        if (token < 0)
            return false;
        *out_token = token;
        return true;
    }

    bool DeviceGraphOrchestrator::flushPendingMTPWork()
    {
        void *stream = consumePendingLogitsStream(
            PendingLogitsStreamRole::MTPSidecar,
            "flushPendingMTPWork");
        if (!stream)
        {
            return true;
        }

        if (!state_.device_id.is_gpu())
        {
            return true;
        }

        try
        {
            /*
             * MTP sidecar replay can intentionally hand a live stream to the
             * runner so sampling/verifier setup becomes the synchronization
             * boundary. Time that handoff explicitly; otherwise backend-specific
             * waits disappear into decode-step wall time.
             */
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sidecar_deferred_stream_flush",
                "decode",
                state_.device_id.toString(),
                {{"role", "mtp_sidecar"}});
            GPUDeviceContextPool::instance()
                .getContext(state_.device_id)
                .synchronizeStream(stream);
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_deferred_stream_flushes",
                1.0,
                "decode",
                state_.device_id.toString());
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to flush pending MTP sidecar stream on "
                      << state_.device_id.toString() << ": " << e.what());
            return false;
        }
    }

    const char *DeviceGraphOrchestrator::pendingLogitsStreamRoleName(
        PendingLogitsStreamRole role)
    {
        switch (role)
        {
        case PendingLogitsStreamRole::MTPSidecar:
            return "mtp_sidecar";
        case PendingLogitsStreamRole::MainDecode:
            return "main_decode";
        case PendingLogitsStreamRole::AllPositionVerifier:
            return "all_position_verifier";
        }
        return "unknown";
    }

    DeviceGraphOrchestrator::PendingLogitsStreamHandoff &
    DeviceGraphOrchestrator::pendingLogitsStreamHandoff(
        PendingLogitsStreamRole role)
    {
        switch (role)
        {
        case PendingLogitsStreamRole::MTPSidecar:
            return pending_logits_streams_[0];
        case PendingLogitsStreamRole::MainDecode:
            return pending_logits_streams_[1];
        case PendingLogitsStreamRole::AllPositionVerifier:
            return pending_logits_streams_[2];
        }
        throw std::logic_error("Unknown pending logits stream role");
    }

    const DeviceGraphOrchestrator::PendingLogitsStreamHandoff &
    DeviceGraphOrchestrator::pendingLogitsStreamHandoff(
        PendingLogitsStreamRole role) const
    {
        switch (role)
        {
        case PendingLogitsStreamRole::MTPSidecar:
            return pending_logits_streams_[0];
        case PendingLogitsStreamRole::MainDecode:
            return pending_logits_streams_[1];
        case PendingLogitsStreamRole::AllPositionVerifier:
            return pending_logits_streams_[2];
        }
        throw std::logic_error("Unknown pending logits stream role");
    }

    void *DeviceGraphOrchestrator::pendingLogitsStreamValue(
        PendingLogitsStreamRole role) const
    {
        return pendingLogitsStreamHandoff(role).peek();
    }

    void DeviceGraphOrchestrator::publishPendingLogitsStream(
        PendingLogitsStreamRole role,
        void *stream,
        const char *producer)
    {
        auto &handoff = pendingLogitsStreamHandoff(role);
        if (!stream)
        {
            clearPendingLogitsStream(
                role,
                producer ? producer : "publishPendingLogitsStream(nullptr)");
            return;
        }
        if (!handoff.canPublish(stream))
        {
            std::ostringstream message;
            message << "Cannot replace unconsumed pending logits stream for role "
                    << pendingLogitsStreamRoleName(role)
                    << " from producer "
                    << (producer ? producer : "unknown")
                    << ". Consume or explicitly clear the previous handoff first.";
            throw std::logic_error(message.str());
        }

        handoff.publish(stream);

        PerfStatsCollector::addCounter(
            "mtp",
            "pending_logits_stream_publishes",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"role", pendingLogitsStreamRoleName(role)},
             {"producer", producer ? producer : "unknown"}});
    }

    void *DeviceGraphOrchestrator::consumePendingLogitsStream(
        PendingLogitsStreamRole role,
        const char *consumer)
    {
        auto &handoff = pendingLogitsStreamHandoff(role);
        void *stream = handoff.consume();
        if (stream)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "pending_logits_stream_consumes",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"role", pendingLogitsStreamRoleName(role)},
                 {"consumer", consumer ? consumer : "unknown"}});
        }
        return stream;
    }

    void *DeviceGraphOrchestrator::peekPendingLogitsStream(
        PendingLogitsStreamRole role) const
    {
        return pendingLogitsStreamValue(role);
    }

    void DeviceGraphOrchestrator::clearPendingLogitsStream(
        PendingLogitsStreamRole role,
        const char *reason)
    {
        auto &handoff = pendingLogitsStreamHandoff(role);
        if (handoff.hasStream())
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "pending_logits_stream_clears",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"role", pendingLogitsStreamRoleName(role)},
                 {"reason", reason ? reason : "unknown"}});
        }
        handoff.clear();
    }

    void DeviceGraphOrchestrator::clearAllPendingLogitsStreams(const char *reason)
    {
        clearPendingLogitsStream(PendingLogitsStreamRole::MTPSidecar, reason);
        clearPendingLogitsStream(PendingLogitsStreamRole::MainDecode, reason);
        clearPendingLogitsStream(PendingLogitsStreamRole::AllPositionVerifier, reason);
    }

    void DeviceGraphOrchestrator::setMTPAllPositionVerifierSyncDeferralEnabled(bool enabled)
    {
        defer_all_position_verifier_sync_ = enabled;
        if (!enabled)
        {
            clearPendingLogitsStream(
                PendingLogitsStreamRole::AllPositionVerifier,
                "setMTPAllPositionVerifierSyncDeferralEnabled(false)");
            stochastic_target_distribution_streams_.fill(nullptr);
            stochastic_draft_distribution_streams_.fill(nullptr);
            clearStochasticTargetSampleReadySlots();
            clearStochasticDraftSampleReadySlots();
        }
    }

    void DeviceGraphOrchestrator::setMTPMainDecodeSyncDeferralEnabled(bool enabled)
    {
        defer_next_mtp_main_decode_sync_ = enabled;
        if (!enabled)
        {
            clearPendingLogitsStream(
                PendingLogitsStreamRole::MainDecode,
                "setMTPMainDecodeSyncDeferralEnabled(false)");
        }
    }

    bool DeviceGraphOrchestrator::shouldDeferAllPositionVerifierFinalSync() const
    {
        return defer_all_position_verifier_sync_ &&
               compute_all_position_logits_ &&
               state_.device_id.is_gpu() &&
               debugEnv().execution.gpu_graphs &&
               !debugEnv().gpu_stage_timing &&
               !debugEnv().gpu_stage_timing_detail;
    }

    void DeviceGraphOrchestrator::setPendingAllPositionVerifierStream(void *stream)
    {
        publishPendingLogitsStream(
            PendingLogitsStreamRole::AllPositionVerifier,
            stream,
            "setPendingAllPositionVerifierStream");
        if (stream)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "all_position_verifier_deferred_stream_handoffs",
                1.0,
                "decode",
                state_.device_id.toString());
        }
    }

    bool DeviceGraphOrchestrator::shouldDeferMainDecodeFinalSync() const
    {
        return defer_next_mtp_main_decode_sync_ &&
               !compute_all_position_logits_ &&
               state_.device_id.is_gpu() &&
               debugEnv().execution.gpu_graphs &&
               !debugEnv().gpu_stage_timing &&
               !debugEnv().gpu_stage_timing_detail;
    }

    void DeviceGraphOrchestrator::setPendingMainDecodeStream(void *stream)
    {
        /*
         * The main-decode stream handoff is one-shot by design.  MTP condition
         * forward is the producer, and the next GPU sampler/distribution build
         * is the only valid consumer.  Clearing the arm flag here prevents a
         * later unrelated decode from inheriting stale replay ordering.
        */
        defer_next_mtp_main_decode_sync_ = false;
        publishPendingLogitsStream(
            PendingLogitsStreamRole::MainDecode,
            stream,
            "setPendingMainDecodeStream");
        if (stream)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "main_decode_deferred_stream_handoffs",
                1.0,
                "decode",
                state_.device_id.toString());
        }
    }

    bool DeviceGraphOrchestrator::supportsMTPSpecStatePublication() const
    {
        return graph_builder_ && graph_builder_->config().mtp.enabled &&
               state_.kv_cache != nullptr && !state_.mtp_kv_caches.empty();
    }

    bool DeviceGraphOrchestrator::publishAcceptedMTPSpecState(
        const MTPSpecStepPlan &plan,
        std::string *error)
    {
        auto fail = [&](std::string reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] " << reason);
            return false;
        };

        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
        {
            return fail("MTP spec-state publication requires an MTP-enabled graph builder");
        }
        if (!forward_engine_)
        {
            return fail("MTP spec-state publication requires an initialized forward engine");
        }

        auto verifier_graph = forward_engine_->lastExecutedForwardGraph();
        if (!verifier_graph || !*verifier_graph || !verifier_graph->graph)
        {
            return fail("MTP spec-state publication has no cached verifier graph from the last forward");
        }
        if (!verifier_graph->is_decode || !verifier_graph->all_position_logits)
        {
            return fail("MTP spec-state publication requires the last forward to be an all-position verifier decode graph");
        }
        if (verifier_graph->signature.seq_len != plan.draft_count)
        {
            std::ostringstream msg;
            msg << "MTP spec-state publication verifier-row mismatch: plan="
                << plan.draft_count << " graph=" << verifier_graph->signature.seq_len
                << " metadata_target_rows=" << plan.target_rows;
            return fail(msg.str());
        }

        void *stream = verifier_graph->stream;
        if (state_.device_id.is_gpu() && !stream)
        {
            stream = explicitGPUStreamForOperation("mtp_spec_state_publication");
        }
        if (state_.device_id.is_gpu() && !stream)
        {
            return fail("MTP spec-state publication could not resolve an explicit GPU stream");
        }
        if (plan.publish_mtp_shifted_kv &&
            state_.device_id.is_gpu() &&
            !waitForPendingShiftedMTPKVReady(
                stream,
                "mtp_spec_state_publication"))
        {
            return fail("MTP spec-state publication could not order after deferred shifted MTP KV append");
        }

        if (!state_.kv_cache)
        {
            return fail("MTP spec-state publication requires an initialized main KV cache");
        }
        std::vector<IKVCache *> mtp_caches;
        if (plan.publish_mtp_shifted_kv)
        {
            mtp_caches.reserve(state_.mtp_kv_caches.size());
            for (const auto &cache : state_.mtp_kv_caches)
            {
                if (!cache)
                {
                    return fail("MTP spec-state publication requires initialized MTP KV caches");
                }
                mtp_caches.push_back(cache.get());
            }
        }

        MTPSpecKVPublicationResult kv_result =
            publishAcceptedMTPSpecKVState(
                plan,
                *state_.kv_cache,
                mtp_caches,
                /*seq_idx=*/0,
                stream);
        if (!kv_result.ok)
        {
            return fail(kv_result.error);
        }

        MTPSpecStatePublicationResult result =
            llaminar2::publishAcceptedMTPSpecState(
                plan,
                *verifier_graph->graph,
                state_.device_id,
                stream,
                /*require_captured_stage=*/mtpSpecStatePublicationRequiresCapturedStage());
        if (!result.ok)
        {
            return fail(result.error);
        }

        /*
         * The next MTP sidecar consumes PREFIX_TERMINAL_HIDDEN as its main-model
         * hidden input.  GDN/short-conv verifier-state publication restores
         * recurrent state, and KV publication truncates/appends cache state, but
         * neither one updates this terminal-hidden buffer.  Select the accepted
         * verifier row explicitly and enqueue the copy on the same stream used
         * for publication so the readiness event below covers every live-state
         * component the next sidecar will read.
         */
        if (plan.publish_mtp_shifted_kv && result.accepted_count > 0)
        {
            const int accepted_hidden_row = result.accepted_count - 1;
            if (!selectMTPTerminalHiddenRow(
                    accepted_hidden_row,
                    verifier_graph->signature.seq_len,
                    stream))
            {
                return fail("MTP spec-state publication could not restore accepted terminal hidden row");
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "spec_state_terminal_hidden_publications",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"accepted_row", std::to_string(accepted_hidden_row)},
                 {"target_rows", std::to_string(verifier_graph->signature.seq_len)}});
        }

        if (!state_.positions.empty())
            state_.positions[0] = plan.target_cached_tokens;
        if (!state_.sequence_lengths.empty())
            state_.sequence_lengths[0] = plan.target_cached_tokens;
        /*
         * Publication advances live state to a verifier-captured accepted row.
         * That is a hard mutation boundary for every captured decode/sidecar
         * graph, even when the speculative step accepted all draft rows.  The
         * verifier graph was captured against the old live-state epoch and may
         * contain backend-owned dynamic pointers or row metadata that are not
         * part of PrefixStateSnapshot.  Resetting replay state here keeps the
         * published state atomic: future graph launches must be captured from
         * the newly published positions, KV/GDN state, and shifted MTP cache.
         */
        const LivePrefixMutationReason mutation_reason =
            plan.requiresCorrectionReplay()
                ? LivePrefixMutationReason::RejectedCorrection
                : LivePrefixMutationReason::AcceptedSpecPublication;
        handleLivePrefixReplayStateAfterMutation(
            mutation_reason,
            "mtp_spec_state_publication",
            /*preserve_gpu_replay_state=*/false);

        if (state_.device_id.is_gpu() &&
            !recordAcceptedSpecPublicationReady(
                stream,
                "mtp_spec_state_publication"))
        {
            return fail("MTP spec-state publication could not record live-state readiness");
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "spec_state_publications",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"accepted_count", std::to_string(result.accepted_count)},
             {"main_kv_tokens", std::to_string(kv_result.main_truncated_tokens)},
             {"publishes_mtp_shifted_kv", plan.publish_mtp_shifted_kv ? "true" : "false"},
             {"restored_stages", std::to_string(result.restored_stage_count)},
             {"skipped_stages", std::to_string(result.skipped_stage_count)},
             {"target_rows", std::to_string(plan.target_rows)}});
        return true;
    }

    std::vector<ForwardExecutionEngine::ReplayCacheObservation>
    DeviceGraphOrchestrator::forwardReplayCacheObservations() const
    {
        if (!forward_engine_)
        {
            return {};
        }
        return forward_engine_->replayCacheObservations(live_replay_state_epoch_);
    }

    bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromLastForward(
        const int32_t *tokens,
        int token_count,
        int already_appended_tokens)
    {
        return commitMTPShiftedRowsFromPartialForward(
            tokens,
            token_count,
            already_appended_tokens,
            token_count);
    }

    bool DeviceGraphOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden(
        int32_t token,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (already_appended_tokens < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Sequential MTP shifted-row commit received negative already_appended_tokens");
            return false;
        }
        if (state_.positions.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Sequential MTP shifted-row commit requires position state");
            return false;
        }

        const int position_offset =
            position_offset_override >= 0
                ? position_offset_override
                : state_.positions[0] - already_appended_tokens;
        if (position_offset < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid sequential MTP shifted-row commit position_offset="
                      << position_offset << " already_appended=" << already_appended_tokens
                      << " override=" << position_offset_override);
            return false;
        }

        IKVCache *cache = state_.mtp_kv_caches.empty() ? nullptr : state_.mtp_kv_caches[0].get();
        if (!cache)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Sequential MTP shifted-row commit requires an initialized MTP KV cache");
            return false;
        }

        const int expected_cached_tokens =
            std::max(0, position_offset - 1 + already_appended_tokens);
        int current_cached_tokens =
            cache->get_cached_tokens(cache->first_layer_index(), 0);
        if (current_cached_tokens > expected_cached_tokens)
        {
            if (!allow_speculative_discard)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Sequential MTP shifted-row commit cache has unexpected extra rows: current="
                          << current_cached_tokens << " expected=" << expected_cached_tokens
                          << " position_offset=" << position_offset
                          << " already_appended=" << already_appended_tokens);
                return false;
            }
            void *stream = explicitGPUStreamForOperation("commitMTPShiftedRowFromCurrentTerminalHidden");
            if (state_.device_id.is_gpu() && !stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Sequential MTP shifted-row commit requires an explicit GPU stream");
                return false;
            }
            if (state_.device_id.is_gpu() &&
                !waitForPendingShiftedMTPKVReady(
                    stream,
                    "shifted_row_sequential_truncate"))
            {
                return false;
            }
            if (!cache->truncateSequence(0, expected_cached_tokens, stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Sequential MTP shifted-row commit failed to discard speculative rows: current="
                          << current_cached_tokens << " expected=" << expected_cached_tokens);
                return false;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "speculative_shifted_rows_discarded",
                static_cast<double>(current_cached_tokens - expected_cached_tokens),
                perfPhaseName(),
                state_.device_id.toString());
            current_cached_tokens = expected_cached_tokens;
        }
        if (current_cached_tokens < expected_cached_tokens)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Sequential MTP shifted-row commit cache mismatch: current="
                      << current_cached_tokens << " expected=" << expected_cached_tokens
                      << " position_offset=" << position_offset
                      << " already_appended=" << already_appended_tokens);
            return false;
        }

        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "commitMTPShiftedRowFromCurrentTerminalHidden"))
            return false;

        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "shifted_row_sequential_commit",
            perfPhaseName(),
            state_.device_id.toString());
        const uint64_t workspace_generation_before_commit =
            state_.device_id.is_gpu() ? workspaceGeneration(state_.device_id) : 0;

        if (!executeMTPDepth0(token,
                              terminal_hidden,
                              position_offset + already_appended_tokens,
                              kMTPDecodeCatchupContext,
                              true,
                              terminal_hidden_buffer_id,
                              /*defer_final_sync=*/state_.device_id.is_gpu()))
        {
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_rows_committed",
            1.0,
            perfPhaseName(),
            state_.device_id.toString());
        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_row_sequential_commits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"already_appended", std::to_string(already_appended_tokens)},
             {"rebuilt_first_row", already_appended_tokens == 0 ? "true" : "false"}});

        if (state_.device_id.is_gpu())
        {
            const uint64_t workspace_generation_after_commit =
                workspaceGeneration(state_.device_id);
            if (workspace_generation_after_commit != workspace_generation_before_commit)
            {
                handleLivePrefixReplayStateAfterMutation(
                    LivePrefixMutationReason::Unknown,
                    "mtp_shifted_row_sequential_commit_workspace_rebind");
            }
        }
        return true;
    }

    bool DeviceGraphOrchestrator::commitMTPShiftedRowFromDeviceTargetSample(
        int target_sample_slot,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (!state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit requires a GPU runner");
            return false;
        }
        if (!supportsMTPDeviceDraftTokenInput() || !stochastic_target_sample_tokens_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit requires device sample-token support");
            return false;
        }
        if (target_sample_slot < 0 ||
            target_sample_slot >= sampling_math::kSpeculativeBatchMaxOutputTokens)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit received invalid target slot="
                      << target_sample_slot);
            return false;
        }
        if (already_appended_tokens < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit received negative already_appended_tokens");
            return false;
        }
        if (state_.positions.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit requires position state");
            return false;
        }

        const int position_offset =
            position_offset_override >= 0
                ? position_offset_override
                : state_.positions[0] - already_appended_tokens;
        if (position_offset < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid device-target MTP shifted-row commit position_offset="
                      << position_offset << " already_appended=" << already_appended_tokens
                      << " override=" << position_offset_override);
            return false;
        }

        IKVCache *cache = state_.mtp_kv_caches.empty() ? nullptr : state_.mtp_kv_caches[0].get();
        if (!cache)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit requires an initialized MTP KV cache");
            return false;
        }

        const int expected_cached_tokens =
            std::max(0, position_offset - 1 + already_appended_tokens);
        int current_cached_tokens =
            cache->get_cached_tokens(cache->first_layer_index(), 0);
        if (current_cached_tokens > expected_cached_tokens)
        {
            if (!allow_speculative_discard)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit cache has unexpected extra rows: current="
                          << current_cached_tokens << " expected=" << expected_cached_tokens
                          << " position_offset=" << position_offset
                          << " already_appended=" << already_appended_tokens);
                return false;
            }
            void *stream = explicitGPUStreamForOperation("commitMTPShiftedRowFromDeviceTargetSample");
            if (!stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit requires an explicit GPU stream");
                return false;
            }
            if (!waitForPendingShiftedMTPKVReady(
                    stream,
                    "shifted_row_device_target_truncate"))
            {
                return false;
            }
            if (!cache->truncateSequence(0, expected_cached_tokens, stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit failed to discard speculative rows: current="
                          << current_cached_tokens << " expected=" << expected_cached_tokens);
                return false;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "speculative_shifted_rows_discarded",
                static_cast<double>(current_cached_tokens - expected_cached_tokens),
                perfPhaseName(),
                state_.device_id.toString());
            current_cached_tokens = expected_cached_tokens;
        }
        if (current_cached_tokens < expected_cached_tokens)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-target MTP shifted-row commit cache mismatch: current="
                      << current_cached_tokens << " expected=" << expected_cached_tokens
                      << " position_offset=" << position_offset
                      << " already_appended=" << already_appended_tokens);
            return false;
        }

        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_buffer_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(
                &terminal_hidden,
                &terminal_hidden_buffer_id,
                "commitMTPShiftedRowFromDeviceTargetSample"))
            return false;

        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "shifted_row_device_target_commit",
            perfPhaseName(),
            state_.device_id.toString());
        const uint64_t workspace_generation_before_commit =
            workspaceGeneration(state_.device_id);

        const int *target_token_device =
            static_cast<const int *>(stochastic_target_sample_tokens_dev_) +
            target_sample_slot;
        if (!executeMTPDepth0Batched(
                /*draft_condition_tokens=*/nullptr,
                /*token_count=*/1,
                terminal_hidden,
                position_offset + already_appended_tokens,
                kMTPDecodeCatchupContext,
                /*kv_cache_only=*/true,
                terminal_hidden_buffer_id,
                /*defer_final_sync=*/true,
                target_token_device,
                target_sample_slot,
                /*draft_condition_ready_is_target=*/true))
        {
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_rows_committed",
            1.0,
            perfPhaseName(),
            state_.device_id.toString());
        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_row_device_target_commits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"target_slot", std::to_string(target_sample_slot)},
             {"already_appended", std::to_string(already_appended_tokens)},
             {"rebuilt_first_row", already_appended_tokens == 0 ? "true" : "false"}});

        const uint64_t workspace_generation_after_commit =
            workspaceGeneration(state_.device_id);
        if (workspace_generation_after_commit != workspace_generation_before_commit)
        {
            handleLivePrefixReplayStateAfterMutation(
                LivePrefixMutationReason::Unknown,
                "mtp_shifted_row_device_target_commit_workspace_rebind");
        }
        return true;
    }

    bool DeviceGraphOrchestrator::commitMTPShiftedRowsFromPartialForward(
        const int32_t *tokens,
        int token_count,
        int already_appended_tokens,
        int main_forward_token_count,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled)
            return true;
        if (token_count <= already_appended_tokens)
            return true;
        if (!tokens || token_count <= 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit requires accepted tokens");
            return false;
        }
        if (already_appended_tokens < 1)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit expects the sidecar to own the first accepted token");
            return false;
        }
        if (!state_.hidden)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit requires hidden rows from the last main forward");
            return false;
        }
        if (state_.positions.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit requires position state");
            return false;
        }
        const int catchup_token_count = token_count - already_appended_tokens;
        const int hidden_source_row_start = already_appended_tokens - 1;
        const int hidden_source_row_end = hidden_source_row_start + catchup_token_count;
        if (main_forward_token_count <= 0 ||
            hidden_source_row_start < 0 ||
            hidden_source_row_end > main_forward_token_count)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit received invalid main_forward_token_count="
                      << main_forward_token_count << " token_count=" << token_count
                      << " already_appended_tokens=" << already_appended_tokens
                      << " catchup_token_count=" << catchup_token_count
                      << " hidden_source_row_start=" << hidden_source_row_start
                      << " hidden_source_row_end=" << hidden_source_row_end);
            return false;
        }

        const int position_offset =
            position_offset_override >= 0
                ? position_offset_override
                : state_.positions[0] - main_forward_token_count;
        if (position_offset < 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid MTP shifted-row commit position_offset="
                      << position_offset << " token_count=" << token_count
                      << " main_forward_token_count=" << main_forward_token_count
                      << " override=" << position_offset_override);
            return false;
        }

        IKVCache *cache = state_.mtp_kv_caches.empty() ? nullptr : state_.mtp_kv_caches[0].get();
        if (!cache)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit requires an initialized MTP KV cache");
            return false;
        }

        const int expected_cached_tokens =
            std::max(0, position_offset - 1 + already_appended_tokens);
        int current_cached_tokens =
            cache->get_cached_tokens(cache->first_layer_index(), 0);
        if (current_cached_tokens > expected_cached_tokens)
        {
            const int configured_draft_tokens =
                graph_builder_ ? std::max(1, graph_builder_->config().mtp.draft_tokens) : 1;
            if (configured_draft_tokens <= 1 && !allow_speculative_discard)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit cache has unexpected extra rows: current="
                          << current_cached_tokens << " expected=" << expected_cached_tokens
                          << " position_offset=" << position_offset
                          << " already_appended=" << already_appended_tokens);
                return false;
            }
            void *stream = explicitGPUStreamForOperation("commitMTPShiftedRowsFromPartialForward");
            if (state_.device_id.is_gpu() && !stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit requires an explicit GPU stream");
                return false;
            }
            if (state_.device_id.is_gpu() &&
                !waitForPendingShiftedMTPKVReady(
                    stream,
                    "shifted_row_partial_truncate"))
            {
                return false;
            }
            if (!cache->truncateSequence(0, expected_cached_tokens, stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit failed to discard speculative rows: current="
                          << current_cached_tokens << " expected=" << expected_cached_tokens);
                return false;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "speculative_shifted_rows_discarded",
                static_cast<double>(current_cached_tokens - expected_cached_tokens),
                perfPhaseName(),
                state_.device_id.toString());
            current_cached_tokens = expected_cached_tokens;
        }
        if (current_cached_tokens < expected_cached_tokens)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row commit cache mismatch: current="
                      << current_cached_tokens << " expected=" << expected_cached_tokens
                      << " position_offset=" << position_offset
                      << " already_appended=" << already_appended_tokens);
            return false;
        }

        PerfStatsCollector::ScopedTimer timer(
            "mtp",
            "shifted_row_commit",
            perfPhaseName(),
            state_.device_id.toString());
        const uint64_t workspace_generation_before_commit =
            state_.device_id.is_gpu() ? workspaceGeneration(state_.device_id) : 0;

        if (catchup_token_count > 0)
        {
            if (catchup_token_count > 4)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row batched catchup exceeds graph capacity: "
                          << catchup_token_count);
                return false;
            }
            if (catchup_token_count > main_forward_token_count)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row batched catchup exceeds available verifier hidden rows: "
                          << catchup_token_count << " > " << main_forward_token_count);
                return false;
            }
            std::array<int32_t, 4> token_batch{};
            for (int row = 0; row < catchup_token_count; ++row)
                token_batch[static_cast<size_t>(row)] = tokens[already_appended_tokens + row];
            if (!selectMTPTerminalHiddenRows(
                    hidden_source_row_start,
                    catchup_token_count,
                    main_forward_token_count))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] MTP shifted-row catchup failed to select verifier hidden rows "
                          << hidden_source_row_start << ".."
                          << (hidden_source_row_start + catchup_token_count - 1)
                          << " of " << main_forward_token_count);
                return false;
            }
            if (!executeMTPDepth0Batched(token_batch.data(),
                                         catchup_token_count,
                                         state_.prefix_terminal_hidden.get(),
                                         position_offset + already_appended_tokens,
                                         kMTPDecodeCatchupContext,
                                         true,
                                         BufferId::PREFIX_TERMINAL_HIDDEN,
                                         /*defer_final_sync=*/state_.device_id.is_gpu()))
            {
                return false;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "shifted_row_catchup_hidden_row_selects",
                static_cast<double>(catchup_token_count),
                perfPhaseName(),
                state_.device_id.toString(),
                {{"source_row_start", std::to_string(hidden_source_row_start)},
                 {"main_forward_token_count", std::to_string(main_forward_token_count)}});
            PerfStatsCollector::addCounter(
                "mtp",
                "shifted_row_catchup_sidecar_batches",
                1.0,
                perfPhaseName(),
                state_.device_id.toString(),
                {{"rows", std::to_string(catchup_token_count)}});
        }
        if (!refreshMTPTerminalHiddenState(main_forward_token_count, 1))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to restore terminal hidden after MTP shifted-row commit");
            return false;
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_rows_committed",
            static_cast<double>(token_count - already_appended_tokens),
            perfPhaseName(),
            state_.device_id.toString());
        if (state_.device_id.is_gpu())
        {
            const uint64_t workspace_generation_after_commit =
                workspaceGeneration(state_.device_id);
            if (workspace_generation_after_commit != workspace_generation_before_commit)
            {
                handleLivePrefixReplayStateAfterMutation(
                    LivePrefixMutationReason::Unknown,
                    "mtp_shifted_row_commit_workspace_rebind");
            }
        }
        return true;
    }

    const float *DeviceGraphOrchestrator::mtpLogits() const
    {
        auto it = state_.extension_buffers.find(BufferId::MTP_LOGITS);
        if (it == state_.extension_buffers.end() || !it->second)
        {
            return nullptr;
        }
        return it->second->fp32_data();
    }

    bool DeviceGraphOrchestrator::setComputeAllPositionLogits(bool enabled)
    {
        if (!graph_builder_)
        {
            return false;
        }
        if (!graph_builder_->setComputeAllPositionLogits(enabled))
        {
            return false;
        }
        if (!enabled &&
            !graph_builder_->setComputeRowIndexedAllPositionLogits(false, 0))
        {
            return false;
        }
        compute_all_position_logits_ = enabled;
        if (!enabled)
        {
            // Row-indexed logits are meaningful only while the verifier asks
            // for all-position logits. Clearing both knobs together prevents a
            // stale compact row count from shaping the next verifier graph.
            compute_row_indexed_all_position_logits_ = false;
            row_indexed_all_position_logits_row_count_ = 0;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::setComputeRowIndexedAllPositionLogits(bool enabled, int row_count)
    {
        if (!graph_builder_)
        {
            return false;
        }
        if (!graph_builder_->setComputeRowIndexedAllPositionLogits(enabled, row_count))
        {
            return false;
        }
        compute_row_indexed_all_position_logits_ = enabled;
        row_indexed_all_position_logits_row_count_ = enabled ? row_count : 0;
        return true;
    }

    bool DeviceGraphOrchestrator::setMTPSpecVerifierInputPlan(
        const MTPSpecDecodeVerifierInputPlan &plan)
    {
        if (!plan.ok)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Refusing invalid MTP verifier input plan: "
                      << plan.error);
            return false;
        }
        const int max_compact_rows =
            plan.shape.maxTargetQueryLen() * plan.shape.max_requests;
        if (plan.compact_logit_row_count <= 0 ||
            plan.compact_logit_row_count > max_compact_rows ||
            plan.compact_logit_row_count >
                static_cast<int>(plan.verifier_logit_rows.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Refusing malformed MTP verifier row plan: rows="
                      << plan.compact_logit_row_count
                      << " row_values=" << plan.verifier_logit_rows.size()
                      << " max_rows=" << max_compact_rows);
            return false;
        }
        const MTPSpecDecodeVerifierGraphForwardPlan graph_forward_plan =
            buildMTPSpecDecodeVerifierGraphForwardPlan(plan);
        if (!graph_forward_plan.ok)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Refusing MTP verifier row plan that cannot be materialized for graph execution: "
                      << graph_forward_plan.error);
            return false;
        }
        std::vector<int> selected_rows(
            graph_forward_plan.verifier_logit_rows.begin(),
            graph_forward_plan.verifier_logit_rows.begin() + plan.compact_logit_row_count);
        if (!graph_builder_->setRowIndexedAllPositionLogitRows(selected_rows))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Graph builder rejected MTP verifier row plan");
            return false;
        }

        mtp_spec_decode_metadata_binding_.setShape(plan.shape);
        pending_mtp_spec_verifier_input_plan_ = plan;
        return true;
    }

    void DeviceGraphOrchestrator::clearMTPSpecVerifierInputPlan()
    {
        pending_mtp_spec_verifier_input_plan_.reset();
        pending_mtp_verifier_device_token_plan_.reset();
        if (graph_builder_)
            graph_builder_->setRowIndexedAllPositionLogitRows({});
    }

    bool DeviceGraphOrchestrator::materializePendingMTPVerifierInputTokensOnDevice(
        void *execution_stream,
        DeviceId execution_device)
    {
        if (!pending_mtp_verifier_device_token_plan_)
            return true;
        if (!state_.device_id.is_gpu())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Pending MTP verifier device-token plan on non-GPU runner");
            return false;
        }
        if (!execution_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP verifier device-token upload requires an explicit stream");
            return false;
        }
        if (!mtp_verifier_input_tokens_dev_ || !stochastic_draft_sample_tokens_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP verifier device-token buffers are not allocated");
            return false;
        }
        if (pending_mtp_verifier_device_token_plan_->first_token_from_device &&
            !stochastic_target_sample_tokens_dev_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] MTP verifier device-token plan requires STOCHASTIC_TARGET_SAMPLE_TOKENS");
            return false;
        }

        const DeviceId token_device =
            execution_device.is_gpu() ? execution_device : state_.device_id;
        IBackend *backend = getBackendFor(token_device);
        if (!backend)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] No backend for MTP verifier device-token upload on "
                      << token_device.toString());
            return false;
        }

        const auto &plan = *pending_mtp_verifier_device_token_plan_;
        auto *verifier_tokens =
            static_cast<int32_t *>(mtp_verifier_input_tokens_dev_);
        const auto *draft_tokens =
            static_cast<const int32_t *>(stochastic_draft_sample_tokens_dev_) +
            plan.first_draft_slot;

        /*
         * The first verifier input token may still be a host scalar on legacy
         * paths, or it may already be device-resident in the vLLM-style fast
         * lane.  Later entries are sampled draft tokens.  Every copy is queued
         * on the forward engine's stream so graph replay observes a coherent
         * token row without a null-stream dependency.
         */
        if (plan.first_token_from_device)
        {
            const auto *target_token =
                static_cast<const int32_t *>(stochastic_target_sample_tokens_dev_) +
                plan.first_target_sample_slot;
            if (!waitForRequiredStochasticTargetSampleReady(
                    plan.first_target_sample_slot,
                    execution_stream,
                    "mtp_verifier_first_token") ||
                !backend->deviceCopyAsync(
                    verifier_tokens,
                    target_token,
                    sizeof(int32_t),
                    token_device.gpu_ordinal(),
                    execution_stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to copy device-resident first MTP verifier token");
                return false;
            }
        }
        else
        {
            if (!backend->hostToDeviceOnStream(
                    verifier_tokens,
                    &pending_mtp_verifier_device_token_plan_->first_token,
                    sizeof(int32_t),
                    token_device.gpu_ordinal(),
                    execution_stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload first MTP verifier token");
                return false;
            }
        }
        if (plan.draft_token_count > 0 &&
            !waitForRequiredStochasticDraftSampleReadyRange(
                plan.first_draft_slot,
                plan.draft_token_count,
                execution_stream,
                "mtp_verifier_input_tokens"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to order verifier token copy after deferred draft samples");
            return false;
        }
        if (plan.draft_token_count > 0 &&
            !backend->deviceCopyAsync(
                verifier_tokens + 1,
                draft_tokens,
                sizeof(int32_t) * static_cast<size_t>(plan.draft_token_count),
                token_device.gpu_ordinal(),
                execution_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to copy MTP draft verifier tokens");
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_device_token_input_prepares",
            1.0,
            "decode",
            token_device.toString(),
            {{"total_tokens", std::to_string(plan.total_verifier_input_tokens)},
             {"draft_tokens", std::to_string(plan.draft_token_count)}});
        return true;
    }

    void DeviceGraphOrchestrator::clearStochasticDraftSampleReadySlot(int slot)
    {
        if (slot < 0 ||
            slot >= static_cast<int>(stochastic_draft_sample_ready_.size()))
        {
            return;
        }

        auto &ready = stochastic_draft_sample_ready_[static_cast<size_t>(slot)];
        ready.valid = false;
        ready.producer_stream = nullptr;
    }

    void DeviceGraphOrchestrator::clearStochasticDraftSampleReadySlots()
    {
        for (int slot = 0;
             slot < static_cast<int>(stochastic_draft_sample_ready_.size());
             ++slot)
        {
            clearStochasticDraftSampleReadySlot(slot);
        }
    }

    void DeviceGraphOrchestrator::clearStochasticTargetSampleReadySlot(int slot)
    {
        if (slot < 0 ||
            slot >= static_cast<int>(stochastic_target_sample_ready_.size()))
        {
            return;
        }

        auto &ready = stochastic_target_sample_ready_[static_cast<size_t>(slot)];
        ready.valid = false;
        ready.producer_stream = nullptr;
    }

    void DeviceGraphOrchestrator::clearStochasticTargetSampleReadySlots()
    {
        for (int slot = 0;
             slot < static_cast<int>(stochastic_target_sample_ready_.size());
             ++slot)
        {
            clearStochasticTargetSampleReadySlot(slot);
        }
    }

    bool DeviceGraphOrchestrator::recordStochasticDraftSampleReady(
        int slot,
        void *producer_stream)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (slot < 0 ||
            slot >= static_cast<int>(stochastic_draft_sample_ready_.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid deferred draft sample slot=" << slot);
            return false;
        }
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred draft sample readiness requires an explicit producer stream");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        auto &ready = stochastic_draft_sample_ready_[static_cast<size_t>(slot)];
        if (!ready.event)
        {
            void *raw_event = backend->createEvent(state_.device_id.gpu_ordinal());
            if (!raw_event)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create deferred draft sample event");
                return false;
            }
            const int device_ordinal = state_.device_id.gpu_ordinal();
            ready.event = std::shared_ptr<void>(
                raw_event,
                [backend, device_ordinal](void *event)
                {
                    if (event)
                        backend->destroyEvent(event, device_ordinal);
                });
        }

        if (!backend->recordEvent(
                ready.event.get(),
                state_.device_id.gpu_ordinal(),
                producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record deferred draft sample event");
            ready.valid = false;
            ready.producer_stream = nullptr;
            return false;
        }

        ready.valid = true;
        ready.producer_stream = producer_stream;
        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_draft_sample_ready_events",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"slot", std::to_string(slot)}});
        return true;
    }

    bool DeviceGraphOrchestrator::recordStochasticTargetSampleReady(
        int slot,
        void *producer_stream)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (slot < 0 ||
            slot >= static_cast<int>(stochastic_target_sample_ready_.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid deferred target sample slot=" << slot);
            return false;
        }
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred target sample readiness requires an explicit producer stream");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        auto &ready = stochastic_target_sample_ready_[static_cast<size_t>(slot)];
        if (!ready.event)
        {
            void *raw_event = backend->createEvent(state_.device_id.gpu_ordinal());
            if (!raw_event)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create deferred target sample event");
                return false;
            }
            const int device_ordinal = state_.device_id.gpu_ordinal();
            ready.event = std::shared_ptr<void>(
                raw_event,
                [backend, device_ordinal](void *event)
                {
                    if (event)
                        backend->destroyEvent(event, device_ordinal);
                });
        }

        if (!backend->recordEvent(
                ready.event.get(),
                state_.device_id.gpu_ordinal(),
                producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record deferred target sample event");
            ready.valid = false;
            ready.producer_stream = nullptr;
            return false;
        }

        ready.valid = true;
        ready.producer_stream = producer_stream;
        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_target_sample_ready_events",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"slot", std::to_string(slot)}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForStochasticDraftSampleReadyRange(
        int first_slot,
        int slot_count,
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu() || slot_count <= 0)
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred draft sample consumer requires an explicit stream");
            return false;
        }
        if (first_slot < 0 ||
            first_slot + slot_count >
                static_cast<int>(stochastic_draft_sample_ready_.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid deferred draft sample wait range first_slot="
                      << first_slot << " slot_count=" << slot_count);
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const char *consumer = consumer_name && consumer_name[0] != '\0'
                                   ? consumer_name
                                   : "unknown";
        for (int i = 0; i < slot_count; ++i)
        {
            const int slot = first_slot + i;
            const auto &ready =
                stochastic_draft_sample_ready_[static_cast<size_t>(slot)];
            if (!ready.valid)
                continue;
            if (!ready.event)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Deferred draft sample slot="
                          << slot << " is marked valid without an event");
                return false;
            }
            if (ready.producer_stream != consumer_stream &&
                !backend->streamWaitEvent(
                    consumer_stream,
                    ready.event.get(),
                    state_.device_id.gpu_ordinal()))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue deferred draft sample wait for slot="
                          << slot << " consumer=" << consumer);
                return false;
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_draft_sample_ready_waits",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"slot", std::to_string(slot)},
                 {"consumer", consumer},
                 {"same_stream", boolTag(ready.producer_stream == consumer_stream)}});
        }
        return true;
    }

    bool DeviceGraphOrchestrator::waitForRequiredStochasticDraftSampleReadyRange(
        int first_slot,
        int slot_count,
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu() || slot_count <= 0)
            return true;
        if (first_slot < 0 ||
            first_slot + slot_count >
                static_cast<int>(stochastic_draft_sample_ready_.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid required deferred draft sample wait range first_slot="
                      << first_slot << " slot_count=" << slot_count);
            return false;
        }

        const char *consumer = consumer_name && consumer_name[0] != '\0'
                                   ? consumer_name
                                   : "unknown";
        for (int i = 0; i < slot_count; ++i)
        {
            const int slot = first_slot + i;
            const auto &ready =
                stochastic_draft_sample_ready_[static_cast<size_t>(slot)];
            if (!ready.valid || !ready.event)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Required deferred draft sample slot="
                          << slot << " is not ready for consumer=" << consumer);
                return false;
            }
        }

        return waitForStochasticDraftSampleReadyRange(
            first_slot,
            slot_count,
            consumer_stream,
            consumer_name);
    }

    bool DeviceGraphOrchestrator::waitForStochasticTargetSampleReady(
        int slot,
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred target sample consumer requires an explicit stream");
            return false;
        }
        if (slot < 0 ||
            slot >= static_cast<int>(stochastic_target_sample_ready_.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid deferred target sample wait slot="
                      << slot);
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const char *consumer = consumer_name && consumer_name[0] != '\0'
                                   ? consumer_name
                                   : "unknown";
        const auto &ready =
            stochastic_target_sample_ready_[static_cast<size_t>(slot)];
        if (!ready.valid)
            return true;
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred target sample slot="
                      << slot << " is marked valid without an event");
            return false;
        }
        if (ready.producer_stream != consumer_stream &&
            !backend->streamWaitEvent(
                consumer_stream,
                ready.event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue deferred target sample wait for slot="
                      << slot << " consumer=" << consumer);
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_target_sample_ready_waits",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"slot", std::to_string(slot)},
             {"consumer", consumer},
             {"same_stream", boolTag(ready.producer_stream == consumer_stream)}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForRequiredStochasticTargetSampleReady(
        int slot,
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (slot < 0 ||
            slot >= static_cast<int>(stochastic_target_sample_ready_.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid required deferred target sample wait slot="
                      << slot);
            return false;
        }

        const char *consumer = consumer_name && consumer_name[0] != '\0'
                                   ? consumer_name
                                   : "unknown";
        const auto &ready =
            stochastic_target_sample_ready_[static_cast<size_t>(slot)];
        if (!ready.valid || !ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Required deferred target sample slot="
                      << slot << " is not ready for consumer=" << consumer);
            return false;
        }

        return waitForStochasticTargetSampleReady(
            slot,
            consumer_stream,
            consumer_name);
    }

    bool DeviceGraphOrchestrator::recordShiftedMTPKVReady(
        void *producer_stream,
        const char *producer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred shifted MTP KV readiness requires an explicit producer stream");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        auto &ready = shifted_mtp_kv_ready_;
        if (ready.valid && ready.producer_stream != producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred shifted MTP KV producer attempted to overwrite an unconsumed event from a different stream");
            return false;
        }
        if (!ready.event)
        {
            void *raw_event = backend->createEvent(state_.device_id.gpu_ordinal());
            if (!raw_event)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create shifted MTP KV readiness event");
                return false;
            }
            const int device_ordinal = state_.device_id.gpu_ordinal();
            ready.event = std::shared_ptr<void>(
                raw_event,
                [backend, device_ordinal](void *event)
                {
                    if (event)
                        backend->destroyEvent(event, device_ordinal);
                });
        }

        if (!backend->recordEvent(
                ready.event.get(),
                state_.device_id.gpu_ordinal(),
                producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record shifted MTP KV readiness event");
            ready.valid = false;
            ready.producer_stream = nullptr;
            return false;
        }

        ready.valid = true;
        ready.producer_stream = producer_stream;
        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_mtp_kv_ready_events",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"producer", producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown"}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForPendingShiftedMTPKVReady(
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        auto &ready = shifted_mtp_kv_ready_;
        if (!ready.valid)
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred shifted MTP KV consumer requires an explicit stream");
            return false;
        }
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Deferred shifted MTP KV readiness is marked valid without an event");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const bool same_stream = ready.producer_stream == consumer_stream;
        if (!same_stream &&
            !backend->streamWaitEvent(
                consumer_stream,
                ready.event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue shifted MTP KV wait for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "shifted_mtp_kv_ready_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"},
             {"same_stream", boolTag(same_stream)}});
        ready.valid = false;
        ready.producer_stream = nullptr;
        return true;
    }

    bool DeviceGraphOrchestrator::recordAcceptedSpecPublicationReady(
        void *producer_stream,
        const char *producer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        if (!producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Accepted spec-state publication readiness requires an explicit producer stream");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        auto &ready = accepted_spec_publication_ready_;
        if (ready.valid && ready.producer_stream != producer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Accepted spec-state publication attempted to overwrite an unconsumed event from a different stream");
            return false;
        }
        if (!ready.event)
        {
            void *raw_event = backend->createEvent(state_.device_id.gpu_ordinal());
            if (!raw_event)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create accepted spec-state publication readiness event");
                return false;
            }
            const int device_ordinal = state_.device_id.gpu_ordinal();
            ready.event = std::shared_ptr<void>(
                raw_event,
                [backend, device_ordinal](void *event)
                {
                    if (event)
                        backend->destroyEvent(event, device_ordinal);
                });
        }

        if (!backend->recordEvent(
                ready.event.get(),
                state_.device_id.gpu_ordinal(),
                producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to record accepted spec-state publication readiness event");
            ready.valid = false;
            ready.producer_stream = nullptr;
            ready.live_state_epoch = 0;
            return false;
        }

        ready.valid = true;
        ready.producer_stream = producer_stream;
        ready.live_state_epoch = live_replay_state_epoch_;
        PerfStatsCollector::addCounter(
            "mtp",
            "accepted_spec_publication_ready_events",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"producer", producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown"},
             {"live_state_epoch", std::to_string(ready.live_state_epoch)}});
        return true;
    }

    bool DeviceGraphOrchestrator::waitForPendingAcceptedSpecPublicationReady(
        void *consumer_stream,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;
        auto &ready = accepted_spec_publication_ready_;
        if (!ready.valid)
            return true;
        if (!consumer_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Accepted spec-state publication consumer requires an explicit stream");
            return false;
        }
        if (!ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Accepted spec-state publication readiness is marked valid without an event");
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const bool same_stream = ready.producer_stream == consumer_stream;
        if (!same_stream &&
            !backend->streamWaitEvent(
                consumer_stream,
                ready.event.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to queue accepted spec-state publication wait for consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "accepted_spec_publication_ready_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer", consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"},
             {"same_stream", boolTag(same_stream)},
             {"live_state_epoch", std::to_string(ready.live_state_epoch)}});
        ready.valid = false;
        ready.producer_stream = nullptr;
        ready.live_state_epoch = 0;
        return true;
    }

    void DeviceGraphOrchestrator::clearPendingAcceptedSpecPublicationReady()
    {
        accepted_spec_publication_ready_.valid = false;
        accepted_spec_publication_ready_.producer_stream = nullptr;
        accepted_spec_publication_ready_.live_state_epoch = 0;
    }

    bool DeviceGraphOrchestrator::prepareAllPositionVerifierGraphMetadata(
        const ForwardInput &input,
        void *execution_stream,
        DeviceId execution_device)
    {
        (void)input;
        if (!compute_all_position_logits_ ||
            !compute_row_indexed_all_position_logits_)
        {
            return true;
        }

        // CPU row selection consumes the row plan captured in the graph builder.
        // The metadata workspace is needed only when GPU cached graphs read row
        // indices from persistent device buffers during replay.
        if (!state_.device_id.is_gpu())
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "verifier_row_metadata_path",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"path", "cpu_builder_row_plan"},
                 {"rows", std::to_string(row_indexed_all_position_logits_row_count_)}});
            return true;
        }

        if (!pending_mtp_spec_verifier_input_plan_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Row-indexed GPU verifier graph has no pending MTP row plan");
            return false;
        }
        if (!execution_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Row-indexed GPU verifier metadata upload requires an explicit stream");
            return false;
        }

        const DeviceId metadata_device =
            execution_device.is_gpu() ? execution_device : state_.device_id;
        IBackend *backend = getBackendFor(metadata_device);
        if (!backend)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] No backend for verifier metadata upload on "
                      << metadata_device.toString());
            return false;
        }

        const int expected_rows = row_indexed_all_position_logits_row_count_;
        const auto &plan = *pending_mtp_spec_verifier_input_plan_;
        if (plan.compact_logit_row_count != expected_rows)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Verifier row plan count "
                      << plan.compact_logit_row_count
                      << " does not match graph row count " << expected_rows);
            return false;
        }

        const auto upload = uploadMTPSpecDecodeVerifierInputPlan(
            plan,
            mtp_spec_decode_metadata_binding_,
            metadata_device,
            backend,
            execution_stream);
        if (!upload.ok)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload verifier row metadata: "
                      << upload.error);
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_row_metadata_upload_bytes",
            static_cast<double>(upload.bytes_uploaded),
            "decode",
            metadata_device.toString());
        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_row_metadata_path",
            1.0,
            "decode",
            metadata_device.toString(),
            {{"path", "persistent_workspace"},
             {"rows", std::to_string(expected_rows)}});
        if (!materializePendingMTPVerifierInputTokensOnDevice(
                execution_stream,
                metadata_device))
        {
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::prepareLiveStateForForwardGraphExecution(
        const ForwardInput &input,
        void *execution_stream,
        DeviceId execution_device)
    {
        (void)input;
        if (!state_.device_id.is_gpu())
            return true;
        if (!accepted_spec_publication_ready_.valid)
            return true;
        if (!execution_stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Pending accepted spec-state publication requires an explicit forward stream");
            return false;
        }

        const DeviceId consumer_device =
            execution_device.is_gpu() ? execution_device : state_.device_id;
        if (consumer_device != state_.device_id)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Accepted spec-state publication wait requested on "
                      << consumer_device.toString()
                      << " but live state belongs to " << state_.device_id.toString());
            return false;
        }
        return waitForPendingAcceptedSpecPublicationReady(
            execution_stream,
            "forward_graph_live_state_consumer");
    }

    const float *DeviceGraphOrchestrator::getAllPositionLogits() const
    {
        if (!state_.all_position_logits)
        {
            return nullptr;
        }
        return state_.all_position_logits->fp32_data();
    }

    std::string DeviceGraphOrchestrator::mtpDecodeUnsupportedReason() const
    {
        return {};
    }

    bool DeviceGraphOrchestrator::supportsMTPTokenCoordination() const
    {
        const IGlobalTPContext *ctx = globalTPContextForMTPCoordination();
        return ctx && ctx->degree() > 1;
    }

    bool DeviceGraphOrchestrator::supportsMTPSidecarSampleFusion() const
    {
        return state_.device_id.is_gpu();
    }

    bool DeviceGraphOrchestrator::supportsMTPSidecarLogitsStreamHandoff() const
    {
        return graph_builder_ &&
               graph_builder_->config().mtp.enabled &&
               state_.device_id.is_gpu() &&
               debugEnv().execution.gpu_graphs &&
               !debugEnv().gpu_stage_timing &&
               !debugEnv().gpu_stage_timing_detail;
    }

    bool DeviceGraphOrchestrator::supportsMTPDeviceDraftTokenInput() const
    {
        return supportsMTPSidecarLogitsStreamHandoff() &&
               supportsDeviceStochasticMTPVerification() &&
               mtp_sidecar_condition_token_dev_;
    }

    bool DeviceGraphOrchestrator::supportsMTPSidecarPreservesMainState() const
    {
        if (!graph_builder_ ||
            !graph_builder_->config().mtp.enabled ||
            !state_.kv_cache ||
            state_.mtp_kv_caches.empty())
        {
            return false;
        }

        const auto &config = graph_builder_->config();
        if (!config.isMoE())
        {
            /*
             * Dense Qwen MTP sidecars are graph-native fragments that write
             * only to MTP-prefixed scratch buffers plus the request-local
             * shifted MTP KV cache. They do not advance main KV, GDN recurrence,
             * terminal hidden, positions, or sampler bookkeeping.
             */
            return true;
        }

        /*
         * SingleDevice GPU MoE with replicated experts now has the same
         * preservation proof as dense: CUDA/ROCm real-model parity runs compare
         * verifier rows from the post-sidecar state against rows from the
         * restored verifier-base state. Advertising this lets the verifier keep
         * its segmented graph replay state warm instead of restoring the base
         * checkpoint before every draft.
         *
         * CPU and sparse-overlay MoE stay conservative here. CPU gets no GPU
         * graph replay benefit from this shortcut, and sparse collectives have
         * separate scratch/participant ownership contracts that need their own
         * equivalence guard before they can skip restores.
         */
        const bool owns_all_routed_experts =
            !config.moe.expert_parallel_plan &&
            config.moe.local_expert_start == 0 &&
            config.moe.local_expert_count < 0;
        return state_.device_id.is_gpu() &&
               owns_all_routed_experts;
    }

    IGlobalTPContext *DeviceGraphOrchestrator::globalTPContextForMTPCoordination() const
    {
        if (global_tp_ctx_)
        {
            return global_tp_ctx_.get();
        }
        if (!graph_builder_)
        {
            return nullptr;
        }

        ITPContext *tp_ctx = graph_builder_->config().tp_ctx;
        if (!tp_ctx || tp_ctx->isLocal())
        {
            return nullptr;
        }

        auto *global_ctx = dynamic_cast<IGlobalTPContext *>(tp_ctx);
        return global_ctx && global_ctx->degree() > 1 ? global_ctx : nullptr;
    }

    int DeviceGraphOrchestrator::sampleGreedyFromMTPLogitsOnDevice()
    {
        auto it = state_.extension_buffers.find(BufferId::MTP_LOGITS);
        if (it == state_.extension_buffers.end() || !it->second)
        {
            return -1;
        }

        const int token_offset = graph_builder_
                                     ? vocabOffsetForTPConfig(graph_builder_->config())
                                     : 0;
        void *pending_stream = consumePendingLogitsStream(
            PendingLogitsStreamRole::MTPSidecar,
            "sampleGreedyFromMTPLogitsOnDevice");
        TensorBase *mtp_logits = it->second.get();
        if (!pending_stream && mtp_logits)
        {
            auto device_opt = mtp_logits->current_device();
            if (device_opt.has_value() && device_opt->is_gpu())
            {
                pending_stream = explicitGPUStreamForOperation("sampleGreedyOnMTPLogits");
                if (!pending_stream)
                {
                    return -1;
                }
            }
        }
        return coordinateGreedyCandidate(
            sampleGreedyCandidateFromTensor(mtp_logits, 0, token_offset,
                                            argmax_partial_vals_dev_,
                                            argmax_partial_idxs_dev_,
                                            argmax_partial_capacity_,
                                            pending_stream,
                                            "mtp_logits"),
            globalTPContextForMTPCoordination());
    }

    int DeviceGraphOrchestrator::sampleGreedyFromAllPositionLogitsOnDevice(int row)
    {
        if (row < 0)
        {
            return -1;
        }

        if (state_.all_position_logits)
        {
            void *stream = nullptr;
            auto device_opt = state_.all_position_logits->current_device();
            if (device_opt.has_value() && device_opt->is_gpu())
            {
                stream = consumePendingLogitsStream(
                    PendingLogitsStreamRole::AllPositionVerifier,
                    "sampleGreedyFromAllPositionLogitsOnDevice");
                if (!stream)
                    stream = explicitGPUStreamForOperation("sampleGreedyFromAllPositionLogitsOnDevice");
                if (!stream)
                {
                    return -1;
                }
            }
            return coordinateGreedyCandidate(
                sampleGreedyCandidateFromTensor(state_.all_position_logits.get(), row, 0,
                                                argmax_partial_vals_dev_,
                                                argmax_partial_idxs_dev_,
                                                argmax_partial_capacity_,
                                                stream,
                                                "all_position_logits"),
                globalTPContextForMTPCoordination());
        }

        if (state_.all_position_logits_local)
        {
            const int token_offset = graph_builder_
                                         ? vocabOffsetForTPConfig(graph_builder_->config())
                                         : 0;
            void *stream = nullptr;
            auto device_opt = state_.all_position_logits_local->current_device();
            if (device_opt.has_value() && device_opt->is_gpu())
            {
                stream = consumePendingLogitsStream(
                    PendingLogitsStreamRole::AllPositionVerifier,
                    "sampleGreedyFromAllPositionLogitsLocalOnDevice");
                if (!stream)
                    stream = explicitGPUStreamForOperation("sampleGreedyFromAllPositionLogitsLocalOnDevice");
                if (!stream)
                {
                    return -1;
                }
            }
            return coordinateGreedyCandidate(
                sampleGreedyCandidateFromTensor(state_.all_position_logits_local.get(), row, token_offset,
                                                argmax_partial_vals_dev_,
                                                argmax_partial_idxs_dev_,
                                                argmax_partial_capacity_,
                                                stream,
                                                "all_position_logits_local"),
                globalTPContextForMTPCoordination());
        }

        return -1;
    }

    bool DeviceGraphOrchestrator::sampleGreedyFromAllPositionLogitsOnDeviceRows(
        int start_row,
        int row_count,
        int32_t *out_tokens)
    {
        if (start_row < 0 || row_count <= 0 || !out_tokens)
            return false;

        const IGlobalTPContext *global_ctx = globalTPContextForMTPCoordination();
        if (global_ctx && global_ctx->degree() > 1)
        {
            return IInferenceRunner::sampleGreedyFromAllPositionLogitsOnDeviceRows(
                start_row, row_count, out_tokens);
        }

        TensorBase *tensor = nullptr;
        int token_offset = 0;
        const char *sample_source = "all_position_logits_rows";
        if (state_.all_position_logits)
        {
            tensor = state_.all_position_logits.get();
        }
        else if (state_.all_position_logits_local)
        {
            tensor = state_.all_position_logits_local.get();
            sample_source = "all_position_logits_local_rows";
            token_offset = graph_builder_
                               ? vocabOffsetForTPConfig(graph_builder_->config())
                               : 0;
        }
        if (!tensor)
            return false;

        constexpr int kMaxStackVerifierRows = 16;
        if (row_count > kMaxStackVerifierRows)
        {
            if (peekPendingLogitsStream(PendingLogitsStreamRole::AllPositionVerifier))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Deferred all-position verifier sampling exceeds stack row capacity");
                clearPendingLogitsStream(
                    PendingLogitsStreamRole::AllPositionVerifier,
                    "sampleGreedyFromAllPositionLogitsOnDeviceRows_over_capacity");
                return false;
            }
            return IInferenceRunner::sampleGreedyFromAllPositionLogitsOnDeviceRows(
                start_row, row_count, out_tokens);
        }

        const auto &shape = tensor->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            static_cast<size_t>(start_row) >= rows ||
            static_cast<size_t>(start_row + row_count) > rows ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return false;
        }

        auto device_opt = tensor->current_device();
        if (device_opt.has_value() && device_opt->is_gpu() && tensor->deviceValid())
        {
            IBackend *backend = getBackendFor(*device_opt);
            const void *gpu_ptr = tensor->gpu_data_ptr();
            if (!backend || !gpu_ptr)
                return false;

            void *stream = consumePendingLogitsStream(
                PendingLogitsStreamRole::AllPositionVerifier,
                "sampleGreedyFromAllPositionLogitsOnDeviceRows");
            const bool consumed_deferred_stream = stream != nullptr;
            if (!stream)
                stream = explicitGPUStreamForOperation("sampleGreedyFromAllPositionLogitsOnDeviceRows");
            if (!stream)
            {
                return false;
            }

            const auto *base = static_cast<const float *>(gpu_ptr);
            const void *first_row =
                base + static_cast<size_t>(start_row) * static_cast<size_t>(cols);
            std::array<float, kMaxStackVerifierRows> values{};
            std::array<int, kMaxStackVerifierRows> indices{};
            const bool ok = backend->argmaxF32BatchedRows(
                first_row,
                row_count,
                static_cast<int>(cols),
                device_opt->gpu_ordinal(),
                values.data(),
                indices.data(),
                stream,
                argmax_partial_vals_dev_,
                argmax_partial_idxs_dev_,
                argmax_partial_capacity_);
            if (!ok)
            {
                if (consumed_deferred_stream)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Batched verifier argmax failed after deferred verifier replay");
                    return false;
                }
                if (backend->backendDeviceType() == DeviceType::ROCm)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] ROCm batched verifier argmax failed");
                    return false;
                }
                return IInferenceRunner::sampleGreedyFromAllPositionLogitsOnDeviceRows(
                    start_row, row_count, out_tokens);
            }

            for (int i = 0; i < row_count; ++i)
            {
                if (indices[static_cast<size_t>(i)] < 0)
                    return false;
                out_tokens[i] = static_cast<int32_t>(
                    token_offset + indices[static_cast<size_t>(i)]);
            }
            if (greedyMarginStatsEnabled())
            {
                for (int i = 0; i < row_count; ++i)
                {
                    const void *row_ptr =
                        base + static_cast<size_t>(start_row + i) * static_cast<size_t>(cols);
                    float top_values[2] = {};
                    int top_indices[2] = {};
                    if (backend->topKF32(row_ptr,
                                         static_cast<int>(cols),
                                         2,
                                         device_opt->gpu_ordinal(),
                                         top_values,
                                         top_indices,
                                         stream))
                    {
                        recordGreedyMarginStats(
                            sample_source,
                            *device_opt,
                            start_row + i,
                            top_values[0],
                            top_values[1]);
                    }
                    else
                    {
                        recordGreedyMarginUnavailable(sample_source, *device_opt, start_row + i);
                    }
                }
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "verifier_token_device_batch_samples",
                static_cast<double>(row_count),
                "decode",
                state_.device_id.toString(),
                {{"rows", std::to_string(row_count)}});
            return true;
        }

        return IInferenceRunner::sampleGreedyFromAllPositionLogitsOnDeviceRows(
            start_row, row_count, out_tokens);
    }

    bool DeviceGraphOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDevice(
        const int32_t *draft_tokens,
        int draft_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        DeviceSpeculativeVerifyBatchOutcome *out)
    {
        using namespace sampling_math;
        if (!out)
            return false;
        *out = DeviceSpeculativeVerifyBatchOutcome{};

        const int compare_rows = draft_token_count - 1;
        if (!draft_tokens ||
            draft_token_count <= 0 ||
            draft_token_count > static_cast<int>(kStochasticTargetRows) ||
            compare_rows < 0 ||
            compare_rows > static_cast<int>(kStochasticDraftRows) ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens) ||
            !state_.device_id.is_gpu() ||
            !state_.all_position_logits ||
            !state_.all_position_logits->deviceValid() ||
            !mtp_verifier_input_tokens_dev_ ||
            !stochastic_verify_tokens_dev_ ||
            !stochastic_verify_accept_probs_dev_ ||
            !stochastic_batch_output_tokens_dev_ ||
            !stochastic_batch_output_meta_dev_)
        {
            return false;
        }

        /*
         * This first implementation is deliberately full-vocabulary only.
         * Column-parallel all-position logits need a domain-wide argmax and
         * summary contract; treating a local shard as global would accept or
         * reject the wrong draft token.
         */
        if (graph_builder_ && graph_builder_->config().lm_head_column_parallel)
            return false;

        const IGlobalTPContext *global_ctx = globalTPContextForMTPCoordination();
        if (global_ctx && global_ctx->degree() > 1)
            return false;

        TensorBase *tensor = state_.all_position_logits.get();
        const auto &shape = tensor->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            static_cast<size_t>(draft_token_count) > rows ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return false;
        }

        auto device_opt = tensor->current_device();
        if (!device_opt.has_value() || !device_opt->is_gpu())
            return false;

        IBackend *backend = getBackendFor(*device_opt);
        const void *gpu_ptr = tensor->gpu_data_ptr();
        if (!backend || !gpu_ptr)
            return false;

        /*
         * The all-position verifier graph may have replayed launch-only and
         * handed us its stream. Consume that stream here so row argmax and the
         * reducer are ordered after verifier logits but before the compact D2H
         * summary. Falling back to a fresh stream before validation would race.
         */
        void *stream = consumePendingLogitsStream(
            PendingLogitsStreamRole::AllPositionVerifier,
            "verifyGreedyAllPositionBatchOutcomeOnDevice");
        if (!stream)
        {
            stream = explicitGPUStreamForOperation(
                "verifyGreedyAllPositionBatchOutcomeOnDevice");
        }
        if (!stream)
            return false;

        /*
         * Greedy sidecar sampling still returns host token ids today. Copy the
         * compact verifier input row to the persistent workspace on the same
         * stream that produced verifier logits, then the reducer can compare
         * device tokens without a verifier-row D2H handoff. A later pure
         * device-token greedy sampler can replace this tiny H2D without
         * changing the summary contract.
         */
        if (!backend->hostToDeviceOnStream(
                mtp_verifier_input_tokens_dev_,
                draft_tokens,
                sizeof(int32_t) * static_cast<size_t>(draft_token_count),
                device_opt->gpu_ordinal(),
                stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload greedy MTP verifier token row on "
                      << device_opt->toString());
            return false;
        }

        const auto *base = static_cast<const float *>(gpu_ptr);
        const bool argmax_ok = backend->enqueueArgmaxF32BatchedRowsDevice(
            base,
            draft_token_count,
            static_cast<int>(cols),
            device_opt->gpu_ordinal(),
            stream,
            stochastic_verify_accept_probs_dev_,
            stochastic_verify_tokens_dev_,
            argmax_partial_vals_dev_,
            argmax_partial_idxs_dev_,
            argmax_partial_capacity_);
        if (!argmax_ok)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Greedy all-position verifier device argmax failed on "
                      << device_opt->toString());
            return false;
        }

        std::array<int, kSpeculativeBatchMaxStopTokens> packed_stop_tokens =
            {-1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < stop_token_count; ++i)
            packed_stop_tokens[static_cast<size_t>(i)] = stop_tokens[i];

        if (!backend->enqueueSummarizeGreedySpeculativeVerifyBatch(
                stochastic_verify_tokens_dev_,
                mtp_verifier_input_tokens_dev_,
                compare_rows,
                draft_tokens[0],
                packed_stop_tokens.data(),
                stop_token_count,
                device_opt->gpu_ordinal(),
                stream,
                stochastic_batch_output_tokens_dev_,
                stochastic_batch_output_meta_dev_))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Greedy all-position verifier device summary failed on "
                      << device_opt->toString());
            return false;
        }

        std::array<int32_t, kSpeculativeBatchMaxOutputTokens> output_tokens{};
        std::array<int, kSpeculativeBatchMetaCount> meta{};
        if (!backend->deviceToHostFast(output_tokens.data(),
                                       stochastic_batch_output_tokens_dev_,
                                       sizeof(int32_t) * output_tokens.size(),
                                       device_opt->gpu_ordinal(),
                                       stream) ||
            !backend->deviceToHostFast(meta.data(),
                                       stochastic_batch_output_meta_dev_,
                                       sizeof(int) * meta.size(),
                                       device_opt->gpu_ordinal(),
                                       stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Greedy all-position verifier compact summary D2H failed on "
                      << device_opt->toString());
            return false;
        }

        if (meta[kSpecBatchMetaOk] == 0 ||
            meta[kSpecBatchMetaOutputCount] < 0 ||
            meta[kSpecBatchMetaOutputCount] > kSpeculativeBatchMaxOutputTokens)
        {
            return false;
        }
        for (int i = 0; i < meta[kSpecBatchMetaOutputCount]; ++i)
        {
            if (output_tokens[static_cast<size_t>(i)] < 0)
                return false;
        }

        out->ok = true;
        out->output_tokens = output_tokens;
        out->output_token_count = meta[kSpecBatchMetaOutputCount];
        out->accepted_speculative_prefix =
            meta[kSpecBatchMetaAcceptedSpeculativePrefix];
        out->target_verifier_state_commit_count =
            meta[kSpecBatchMetaTargetVerifierStateCommitCount];
        out->ready_token = meta[kSpecBatchMetaReadyToken];
        out->rejected_verified_token = meta[kSpecBatchMetaRejectedVerifiedToken];
        out->stopped_on_output = meta[kSpecBatchMetaStoppedOnOutput] != 0;
        out->all_speculative_accepted =
            meta[kSpecBatchMetaAllSpeculativeAccepted] != 0;
        out->consumed_verifier_rows = meta[kSpecBatchMetaConsumedVerifierRows];
        out->sampled_terminal = meta[kSpecBatchMetaSampledTerminal] != 0;

        PerfStatsCollector::addCounter(
            "mtp",
            "all_position_greedy_device_batched_rows",
            static_cast<double>(draft_token_count),
            "decode",
            state_.device_id.toString(),
            {{"compare_rows", std::to_string(compare_rows)},
             {"implementation", "device_batch_outcome"}});
        return true;
    }

    bool DeviceGraphOrchestrator::supportsGreedyAllPositionBatchOutcomeOnDevice() const
    {
        /*
         * The compact reducer consumes one device's all-position logits tensor.
         * CPU runners and rank-level sharded TP need different reduction
         * contracts, so only a concrete GPU device runner advertises this path.
         */
        return state_.device_id.is_gpu();
    }

    PrefixRuntimeStateSnapshot DeviceGraphOrchestrator::prefixStateProbe() const
    {
        PrefixRuntimeStateSnapshot snapshot;
        snapshot.initialized = state_.isInitialized();
        snapshot.has_hidden = static_cast<bool>(state_.hidden);
        snapshot.has_logits = static_cast<bool>(state_.logits);
        snapshot.architecture = architecture();
        snapshot.execution_path = "graph";
        snapshot.primary_device = state_.device_id;
        snapshot.current_position = getPosition(0);
        snapshot.session_epoch = session_epoch_;
        snapshot.live_state_epoch = live_replay_state_epoch_;
        snapshot.live_state_mutations = live_state_mutation_count_;
        snapshot.last_live_state_mutation_reason =
            livePrefixMutationReasonName(last_live_state_mutation_reason_);
        snapshot.last_live_state_mutation_operation =
            last_live_state_mutation_operation_;
        snapshot.live_state_accepted_publications =
            live_state_accepted_publications_;
        snapshot.live_state_rejected_corrections =
            live_state_rejected_corrections_;
        snapshot.live_state_prefix_restores = live_state_prefix_restores_;
        snapshot.live_state_prefix_truncates = live_state_prefix_truncates_;
        snapshot.live_state_session_resets = live_state_session_resets_;
        snapshot.prefix_cache_config_enabled =
            graph_builder_ && graph_builder_->config().prefix_cache.enabled;
        snapshot.prefix_cache_bypassed =
            snapshot.prefix_cache_config_enabled && prefix_cache_bypassed_;
        snapshot.prefix_cache_bypass_reason =
            snapshot.prefix_cache_bypassed ? prefix_cache_bypass_reason_ : "";
        snapshot.prefix_cache_ready =
            static_cast<bool>(prefix_cache_) && !snapshot.prefix_cache_bypassed;
        if (prefix_cache_)
        {
            const auto &stats = prefix_cache_->stats();
            snapshot.prefix_cache_lookups = stats.lookups;
            snapshot.prefix_cache_hits = stats.hits;
            snapshot.prefix_cache_partial_hits = stats.partial_hits;
            snapshot.prefix_cache_misses = stats.misses;
            snapshot.prefix_cache_matched_blocks = stats.matched_blocks;
            snapshot.prefix_cache_matched_tokens = stats.matched_tokens;
            snapshot.prefix_cache_stores = stats.stores;
            snapshot.prefix_cache_inserts = stats.inserts;
            snapshot.prefix_cache_evictions = stats.evictions;
            snapshot.prefix_cache_promotions = stats.promotions;
            snapshot.prefix_cache_disk_hydrations = stats.disk_hydrations;
            snapshot.prefix_cache_terminal_state_hits = stats.terminal_state_hits;
            snapshot.prefix_cache_ram_bytes = stats.ram_bytes;
            snapshot.prefix_cache_device_bytes = stats.device_bytes;
            snapshot.prefix_cache_disk_bytes = stats.disk_bytes;
            snapshot.prefix_cache_hybrid_state_bytes = stats.hybrid_state_bytes;
            snapshot.prefix_cache_mtp_state_bytes = stats.mtp_state_bytes;
        }
        snapshot.prefix_cache_bypasses = prefix_cache_stats_.bypasses;
        snapshot.prefix_cache_unsupported_backend_bypasses =
            prefix_cache_stats_.unsupported_backend_bypasses;
        snapshot.prefix_cache_fingerprint_bypasses =
            prefix_cache_stats_.fingerprint_bypasses;
        snapshot.prefix_cache_terminal_state_bypasses =
            prefix_cache_stats_.terminal_state_bypasses;
        snapshot.mtp_config_enabled =
            graph_builder_ && graph_builder_->config().mtp.enabled;
        snapshot.positions = state_.positions;
        snapshot.sequence_lengths = state_.sequence_lengths;

        const int sequence_count = state_.batch_size > 0 ? state_.batch_size : 1;
        if (state_.kv_cache)
        {
            auto cache_probe = inspectKVCacheForPrefixProbe(
                *state_.kv_cache, "primary", state_.device_id, sequence_count);
            auto gdn_probes = inspectHybridGDNForPrefixProbe(*state_.kv_cache);
            snapshot.gdn_layers.insert(snapshot.gdn_layers.end(),
                                       gdn_probes.begin(), gdn_probes.end());
            snapshot.kv_caches.push_back(std::move(cache_probe));
        }

        for (const auto &[device, cache] : state_.pp_kv_caches)
        {
            if (!cache)
            {
                continue;
            }
            auto cache_probe = inspectKVCacheForPrefixProbe(
                *cache, "pp:" + device.to_string(), device, sequence_count);
            auto gdn_probes = inspectHybridGDNForPrefixProbe(*cache);
            snapshot.gdn_layers.insert(snapshot.gdn_layers.end(),
                                       gdn_probes.begin(), gdn_probes.end());
            snapshot.kv_caches.push_back(std::move(cache_probe));
        }

        for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
        {
            const auto &cache = state_.mtp_kv_caches[depth];
            if (!cache)
            {
                continue;
            }
            snapshot.mtp_kv_caches.push_back(inspectKVCacheForPrefixProbe(
                *cache,
                "mtp:" + std::to_string(depth),
                state_.device_id,
                sequence_count));
        }

        return snapshot;
    }

    void DeviceGraphOrchestrator::disablePrefixCacheForRunner(const std::string &reason)
    {
        const bool should_record =
            !prefix_cache_bypassed_ &&
            graph_builder_ &&
            graph_builder_->config().prefix_cache.enabled &&
            reason != "feature disabled";

        if (should_record)
        {
            ++prefix_cache_stats_.bypasses;
            const std::string normalized_reason = lowerASCII(reason);
            if (containsAny(normalized_reason, {"fingerprint", "moe placement policy"}))
            {
                ++prefix_cache_stats_.fingerprint_bypasses;
            }
            else if (containsAny(normalized_reason, {"terminal"}))
            {
                ++prefix_cache_stats_.terminal_state_bypasses;
            }
            else if (containsAny(normalized_reason,
                                 {"unavailable",
                                  "not implemented",
                                  "does not expose",
                                  "without an initialized",
                                  "graph builder unavailable"}))
            {
                ++prefix_cache_stats_.unsupported_backend_bypasses;
            }
        }
        prefix_cache_bypassed_ = true;
        prefix_cache_bypass_reason_ = reason;
        if (!reason.empty())
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Prefix cache bypassed: " << reason);
        }
    }

    bool DeviceGraphOrchestrator::isPrefixCacheMoEModel() const
    {
        if (!graph_builder_)
            return false;
        const std::string architecture = graph_builder_->architectureName();
        return architecture.find("moe") != std::string::npos ||
               architecture.find("MoE") != std::string::npos;
    }

    void *DeviceGraphOrchestrator::explicitGPUStreamForOperation(const char *operation) const
    {
        if (!state_.device_id.is_gpu())
        {
            return nullptr;
        }

        try
        {
            void *stream = GPUDeviceContextPool::instance()
                               .getContext(state_.device_id)
                               .defaultStream();
            if (!stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] "
                          << (operation ? operation : "gpu_operation")
                          << " requires an explicit non-null GPU stream on "
                          << state_.device_id.toString());
            }
            return stream;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] "
                      << (operation ? operation : "gpu_operation")
                      << " could not acquire an explicit GPU stream on "
                      << state_.device_id.toString() << ": " << e.what());
            return nullptr;
        }
    }

    const char *DeviceGraphOrchestrator::livePrefixMutationReasonName(
        LivePrefixMutationReason reason)
    {
        switch (reason)
        {
        case LivePrefixMutationReason::AcceptedSpecPublication:
            return "accepted_publication";
        case LivePrefixMutationReason::RejectedCorrection:
            return "rejected_correction";
        case LivePrefixMutationReason::PrefixRestore:
            return "prefix_restore";
        case LivePrefixMutationReason::PrefixTruncate:
            return "prefix_truncate";
        case LivePrefixMutationReason::SessionReset:
            return "session_reset";
        case LivePrefixMutationReason::Unknown:
        default:
            return "unknown";
        }
    }

    DeviceGraphOrchestrator::LivePrefixMutationRecord
    DeviceGraphOrchestrator::recordLivePrefixMutation(
        LivePrefixMutationReason reason,
        const char *operation)
    {
        const uint64_t previous_epoch = live_replay_state_epoch_;
        ++live_replay_state_epoch_;
        ++live_state_mutation_count_;
        last_live_state_mutation_reason_ = reason;
        last_live_state_mutation_operation_ =
            operation ? operation : "unknown";

        switch (reason)
        {
        case LivePrefixMutationReason::AcceptedSpecPublication:
            ++live_state_accepted_publications_;
            break;
        case LivePrefixMutationReason::RejectedCorrection:
            ++live_state_rejected_corrections_;
            break;
        case LivePrefixMutationReason::PrefixRestore:
            ++live_state_prefix_restores_;
            break;
        case LivePrefixMutationReason::PrefixTruncate:
            ++live_state_prefix_truncates_;
            break;
        case LivePrefixMutationReason::SessionReset:
            ++live_state_session_resets_;
            break;
        case LivePrefixMutationReason::Unknown:
        default:
            break;
        }

        return LivePrefixMutationRecord{
            previous_epoch,
            live_replay_state_epoch_,
            livePrefixMutationReasonName(reason)};
    }

    void DeviceGraphOrchestrator::recordLivePrefixSessionReset(const char *operation)
    {
        clearPendingAcceptedSpecPublicationReady();
        PerfStatsCollector::Tags tags{{"operation", operation ? operation : "unknown"}};
        const LivePrefixMutationRecord mutation =
            recordLivePrefixMutation(
                LivePrefixMutationReason::SessionReset,
                operation);
        tags["mutation_reason"] = mutation.reason_name;
        tags["previous_live_state_epoch"] =
            std::to_string(mutation.previous_epoch);
        tags["live_state_epoch"] = std::to_string(mutation.live_state_epoch);
        tags["live_state_mutation_count"] =
            std::to_string(live_state_mutation_count_);
        if (isPrefixCacheMoEModel())
        {
            tags["model"] = "moe";
            tags["moe_placement_epoch"] = std::to_string(moePlacementEpoch());
        }
        tags["forward_replay_reset_scope"] = "session";
        tags["replay_state"] = "reset";
        tags["sidecar_replay_state"] = "reset";
        tags["kernel_dynamic_state"] = "reset";
        PerfStatsCollector::addCounter("mtp",
                                       "live_prefix_replay_state_after_mutation",
                                       1.0,
                                       "decode",
                                       state_.device_id.toString(),
                                       std::move(tags));
    }

    void DeviceGraphOrchestrator::handleLivePrefixReplayStateAfterMutation(
        LivePrefixMutationReason reason,
        const char *operation,
        bool preserve_gpu_replay_state)
    {
        if (reason != LivePrefixMutationReason::AcceptedSpecPublication &&
            reason != LivePrefixMutationReason::RejectedCorrection)
        {
            clearPendingAcceptedSpecPublicationReady();
        }

        PerfStatsCollector::Tags tags{{"operation", operation ? operation : "unknown"}};
        const LivePrefixMutationRecord mutation =
            recordLivePrefixMutation(reason, operation);
        tags["mutation_reason"] = mutation.reason_name;
        tags["previous_live_state_epoch"] =
            std::to_string(mutation.previous_epoch);
        tags["live_state_epoch"] = std::to_string(mutation.live_state_epoch);
        tags["live_state_mutation_count"] =
            std::to_string(live_state_mutation_count_);
        if (isPrefixCacheMoEModel())
        {
            tags["model"] = "moe";
            tags["moe_placement_epoch"] = std::to_string(moePlacementEpoch());
        }
        bool preserves_correction_graph_replay = false;
        if (state_.device_id.is_gpu() && forward_engine_ && !preserve_gpu_replay_state)
        {
            const bool correction_replay_boundary =
                reason == LivePrefixMutationReason::AcceptedSpecPublication ||
                reason == LivePrefixMutationReason::RejectedCorrection;
            if (correction_replay_boundary)
            {
                const ForwardExecutionEngine::ReplayStateResetSummary summary =
                    forward_engine_->resetCapturedReplayStateForCorrectionReplay(
                        live_replay_state_epoch_);
                preserves_correction_graph_replay =
                    summary.preserved_for_stream_rebind > 0 || state_.device_id.is_gpu();
                tags["forward_replay_reset_scope"] = "correction_replay_decode_only";
                tags["forward_replay_reset_cache_count"] =
                    std::to_string(summary.reset_replay_state);
                tags["forward_replay_stream_rebind_cache_count"] =
                    std::to_string(summary.preserved_for_stream_rebind);
                tags["forward_replay_ordinary_decode_reset_count"] =
                    std::to_string(summary.ordinary_decode_reset);
                tags["forward_replay_all_position_verifier_rebind_count"] =
                    std::to_string(summary.all_position_verifier_preserved);
                tags["forward_replay_other_rebind_count"] =
                    std::to_string(summary.other_preserved);
            }
            else
            {
                forward_engine_->resetCapturedReplayState();
                tags["forward_replay_reset_scope"] = "all";
            }

            /*
             * Accepted-state publication mutates live KV/position state, but
             * depth-0 sidecar graphs are shaped exactly like ordinary decode:
             * they update position/token dynamic params before each launch and
             * read from stable arena buffers.  Resetting their segmented replay
             * state here forced every speculative step back through warmup and
             * erased GPU graph speedups.  Hard restores/truncates still reset
             * sidecar replay below through the non-publication path.
             */
            if (!correction_replay_boundary)
            {
                mtp_sidecar_depth0_cache_.resetReplayState();
                mtp_sidecar_depth0_device_token_cache_.resetReplayState();
                mtp_sidecar_depth0_chained_cache_.resetReplayState();
                mtp_sidecar_depth0_chained_device_token_cache_.resetReplayState();
                mtp_sidecar_depth0_kv_only_cache_.resetReplayState();
                mtp_sidecar_depth0_kv_only_device_token_cache_.resetReplayState();
                for (auto &cache : mtp_sidecar_depth0_kv_only_batch_caches_)
                {
                    cache.resetReplayState();
                }
                tags["sidecar_replay_state"] = "reset";
            }
            else
            {
                tags["sidecar_replay_state"] = "preserved_for_spec_publication";
                preserves_correction_graph_replay = true;
            }
            tags["replay_state"] = "reset";
        }
        else
        {
            tags["replay_state"] = "preserved";
            tags["sidecar_replay_state"] = "preserved";
            if (state_.device_id.is_gpu() && preserve_gpu_replay_state)
            {
                tags["gpu_replay_preserve_reason"] =
                    operation ? operation : "live_prefix_mutation";
            }
        }
        /*
         * KernelFactory::resetAllDynamicState() clears per-kernel dynamic
         * device-param validity and stream bindings.  That is correct for a
         * hard graph reset, but unsafe when this boundary deliberately keeps
         * graph executables alive: CUDA/HIP graphs capture kernel argument
         * pointer identity, while updateDynamicParams() only restamps the
         * already-captured dynamic buffers before replay.  Preserve kernel
         * dynamic objects for accepted/rejected MTP publication boundaries and
         * rely on the explicit stream rebind plus per-stage dynamic updates.
         */
        if (!preserve_gpu_replay_state && preserves_correction_graph_replay)
        {
            tags["kernel_dynamic_state"] =
                "preserved_for_correction_graph_replay";
        }
        else if (!preserve_gpu_replay_state)
        {
            resetKernelDynamicState();
            tags["kernel_dynamic_state"] = "reset";
        }
        else
        {
            tags["kernel_dynamic_state"] = "preserved";
        }
        PerfStatsCollector::addCounter("mtp",
                                       "live_prefix_replay_state_after_mutation",
                                       1.0,
                                       "decode",
                                       state_.device_id.toString(),
                                       std::move(tags));
    }

    PrefixCacheFingerprintResult DeviceGraphOrchestrator::buildCurrentPrefixFingerprint(
        const PrefixCacheRuntimeConfig &prefix_config) const
    {
        const auto &config = graph_builder_->config();
        PrefixFingerprintMaterial material;
        material.model = {
            {"architecture", graph_builder_->architectureName()},
            {"n_layers", std::to_string(config.n_layers)},
            {"d_model", std::to_string(config.d_model)},
            {"n_heads", std::to_string(config.n_heads)},
            {"n_kv_heads", std::to_string(config.n_kv_heads)},
            {"vocab_size", std::to_string(config.vocab_size)},
        };
        material.runtime = {
            {"activation_precision", activationPrecisionToString(config.activation_precision)},
            {"kv_cache_precision", kvCachePrecisionToString(config.kv_cache_precision)},
            {"k_precision", activationPrecisionToString(prefix_layout_.k_precision)},
            {"v_precision", activationPrecisionToString(prefix_layout_.v_precision)},
            {"kv_layout", std::to_string(static_cast<int>(prefix_layout_.kv_layout))},
            {"block_size", std::to_string(prefix_layout_.block_size)},
            {"terminal_state", prefixCacheTerminalStateModeToString(prefix_config.terminal_state)},
        };
        material.topology = {
            {"first_layer_index", std::to_string(prefix_layout_.first_layer_index)},
            {"total_layers", std::to_string(prefix_layout_.total_layers)},
            {"head_dim", std::to_string(prefix_layout_.head_dim)},
        };
        if (config.tp_config && config.tp_config->worldSize() > 1)
        {
            material.topology.push_back({"scope", "tp-domain"});
            material.topology.push_back({"tp_world_size", std::to_string(config.tp_config->worldSize())});
            material.topology.push_back({"tp_total_heads", std::to_string(config.tp_config->totalHeads())});
            material.topology.push_back({"tp_total_kv_heads", std::to_string(config.tp_config->totalKVHeads())});
            material.topology.push_back({"tp_total_d_ff", std::to_string(config.tp_config->totalDFF())});
            material.topology.push_back({"tp_total_vocab", std::to_string(config.tp_config->totalVocab())});
            material.topology.push_back({"tp_proportional", config.tp_config->isProportional() ? "true" : "false"});
            material.topology.push_back({"qkv_column_parallel", config.qkv_column_parallel ? "true" : "false"});
            material.topology.push_back({"lm_head_column_parallel", config.lm_head_column_parallel ? "true" : "false"});
        }
        else
        {
            material.topology.push_back({"scope", "single-participant"});
            material.topology.push_back({"device", state_.device_id.toString()});
            material.topology.push_back({"local_kv_heads", std::to_string(prefix_layout_.local_kv_heads)});
            material.topology.push_back({"kv_head_start", std::to_string(prefix_layout_.kv_head_start)});
        }
        material.hybrid = prefix_layout_.includes_hybrid_state
                              ? std::vector<PrefixFingerprintField>{
                                    {"enabled", "true"},
                                    {"gdn_layers", std::to_string(prefix_layout_.gdn_layers)},
                                    {"hybrid_host_state_bytes", std::to_string(prefix_layout_.hybrid_host_state_bytes)},
                                    {"hybrid_device_state_bytes", std::to_string(prefix_layout_.hybrid_device_state_bytes)},
                                    {"hybrid_state_bytes", std::to_string(prefix_layout_.hybrid_state_bytes)},
                                }
                              : std::vector<PrefixFingerprintField>{{"enabled", "false"}};
        material.mtp = config.mtp.enabled
                           ? std::vector<PrefixFingerprintField>{
                                 {"enabled", "true"},
                                 {"draft_tokens", std::to_string(config.mtp.draft_tokens)},
                                 {"verify_mode", mtpVerifyModeToString(config.mtp.verify_mode)},
                                 {"require_terminal_hidden_for_full_hit",
                                  config.mtp.require_terminal_hidden_for_full_hit ? "true" : "false"},
                                 {"terminal_hidden_bytes", std::to_string(prefix_layout_.terminal_hidden_bytes)},
                             }
                           : std::vector<PrefixFingerprintField>{{"enabled", "false"}};

        const bool model_is_moe = isPrefixCacheMoEModel();
        if (model_is_moe)
        {
            material.moe.push_back({"policy", prefixCacheMoEPolicyToString(prefix_config.moe_policy)});
            material.moe.push_back({"model_is_moe", "true"});
            if (moe_rebalance_controller_)
            {
                const auto *controller = moe_rebalance_controller_.get();
                material.moe.push_back({"controller.num_layers", std::to_string(controller->numLayers())});
                material.moe.push_back({"controller.num_experts", std::to_string(controller->numExperts())});
                material.moe.push_back({"controller.top_k", std::to_string(controller->topK())});
                material.moe.push_back({"controller.placement_epoch", std::to_string(controller->placementEpoch())});
                material.moe.push_back({"controller.total_rebalances", std::to_string(controller->totalRebalances())});

                const auto &placement = controller->currentPlacement();
                for (size_t expert = 0; expert < placement.size(); ++expert)
                {
                    material.moe.push_back({"controller.expert." + std::to_string(expert) + ".owner_participant",
                                            std::to_string(placement[expert])});
                }

                const auto &replicas = controller->currentReplicas();
                material.moe.push_back({"controller.replicas.count", std::to_string(replicas.num_replicated)});
                for (size_t expert = 0; expert < replicas.is_replicated.size(); ++expert)
                {
                    const std::string prefix = "controller.replica." + std::to_string(expert);
                    material.moe.push_back({prefix + ".enabled", replicas.is_replicated[expert] ? "1" : "0"});
                    if (expert < replicas.owner_socket.size())
                    {
                        material.moe.push_back({prefix + ".owner_participant",
                                                std::to_string(replicas.owner_socket[expert])});
                    }
                }
            }
        }

        if (graph_builder_)
            graph_builder_->appendPrefixCacheFingerprintMaterial(material);

        return buildPrefixCacheFingerprint(
            material,
            model_is_moe,
            prefix_config.moe_policy);
    }

    bool DeviceGraphOrchestrator::ensurePrefixCacheReady()
    {
        if (!graph_builder_)
        {
            disablePrefixCacheForRunner("graph builder unavailable");
            return false;
        }

        const auto &config = graph_builder_->config();
        const auto &prefix_config = config.prefix_cache;
        if (prefix_cache_bypassed_)
        {
            return false;
        }
        if (prefix_cache_)
        {
            auto fingerprint = buildCurrentPrefixFingerprint(prefix_config);
            if (fingerprint.bypass || fingerprint.key == 0)
            {
                disablePrefixCacheForRunner(
                    fingerprint.bypass_reason.empty() ? "fingerprint refresh requested bypass"
                                                      : fingerprint.bypass_reason);
                return false;
            }
            if (fingerprint.key != prefix_fingerprint_)
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] Prefix cache fingerprint refreshed for "
                          << graph_builder_->architectureName());
                prefix_fingerprint_ = fingerprint.key;
            }
            return true;
        }
        if (!prefix_config.enabled ||
            prefix_config.storage_mode == PrefixCacheStorageMode::Disabled)
        {
            disablePrefixCacheForRunner("feature disabled");
            return false;
        }
        if (prefix_config.storage_mode == PrefixCacheStorageMode::Device)
        {
            disablePrefixCacheForRunner("device-only prefix cache tier is not implemented yet");
            return false;
        }
        if (!state_.isInitialized() || !state_.kv_cache)
        {
            disablePrefixCacheForRunner("inference state or KV cache unavailable");
            return false;
        }
        if (!state_.pp_kv_caches.empty())
        {
            disablePrefixCacheForRunner("pipeline-parallel KV cache restore is not implemented yet");
            return false;
        }
        const int block_size = prefix_config.block_size > 0 ? prefix_config.block_size : 64;
        const bool terminal_state_enabled =
            prefix_config.terminal_state != PrefixCacheTerminalStateMode::Off;
        const bool owns_terminal_state =
            !pp_stage_config_.has_value() || pp_stage_config_->has_lm_head;
        const bool use_local_terminal_logits =
            config.lm_head_column_parallel && static_cast<bool>(state_.logits_local);
        const auto *terminal_logits_tensor =
            use_local_terminal_logits ? state_.logits_local.get() : state_.logits.get();
        const size_t terminal_hidden_bytes =
            owns_terminal_state && config.mtp.enabled && terminal_state_enabled && state_.d_model > 0
                ? static_cast<size_t>(state_.d_model) * sizeof(float)
                : 0;
        const size_t terminal_logits_bytes =
            !terminal_state_enabled || !owns_terminal_state
                ? 0
                : fp32LogitsRowBytes(terminal_logits_tensor);

        prefix_layout_ = buildDensePrefixPayloadLayout(
            *state_.kv_cache,
            state_.device_id,
            block_size,
            terminal_hidden_bytes,
            terminal_logits_bytes);

        if (config.mtp.enabled)
        {
            if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
            {
                disablePrefixCacheForRunner("MTP prefix cache requested without an initialized MTP KV cache");
                return false;
            }
            if (!attachMTPPayloadLayout(prefix_layout_, *state_.mtp_kv_caches[0]))
            {
                disablePrefixCacheForRunner("MTP KV cache does not expose a logical payload layout");
                return false;
            }
        }

        if (prefix_layout_.fa_layers <= 0 || prefix_layout_.faKVBytes() == 0)
        {
            disablePrefixCacheForRunner("KV cache does not expose a dense logical payload layout");
            return false;
        }
        const size_t block_bytes = prefix_layout_.totalBytes();
        if (block_bytes == 0 || prefix_config.ram_budget_bytes < block_bytes)
        {
            disablePrefixCacheForRunner("RAM budget cannot hold one complete prefix block");
            return false;
        }

        auto fingerprint = buildCurrentPrefixFingerprint(prefix_config);
        if (fingerprint.bypass || fingerprint.key == 0)
        {
            disablePrefixCacheForRunner(
                fingerprint.bypass_reason.empty() ? "fingerprint build requested bypass"
                                                  : fingerprint.bypass_reason);
            return false;
        }

        prefix_fingerprint_ = fingerprint.key;
        prefix_ram_backend_ = std::make_shared<RamPrefixStorageBackend>(prefix_config.ram_budget_bytes);
        prefix_device_hot_backend_.reset();
        if (prefix_config.storage_mode == PrefixCacheStorageMode::Tiered &&
            state_.device_id.is_gpu() &&
            prefix_config.device_budget_bytes >= block_bytes)
        {
            prefix_device_hot_backend_ = std::make_shared<DeviceHotPrefixStorageBackend>(
                state_.device_id,
                prefix_config.device_budget_bytes);
        }
        prefix_disk_backend_.reset();
        if (prefix_config.storage_mode == PrefixCacheStorageMode::Tiered &&
            prefix_config.disk_budget_bytes > 0 &&
            !prefix_config.disk_dir.empty())
        {
            prefix_disk_backend_ = std::make_shared<DiskPrefixStorageBackend>(
                prefix_config.disk_dir,
                prefix_config.disk_budget_bytes);
        }
        prefix_cache_ = std::make_shared<PrefixStateCache>(
            prefix_config.ram_budget_bytes,
            prefix_ram_backend_,
            prefix_disk_backend_,
            prefix_device_hot_backend_);
        prefix_cache_bypassed_ = false;
        prefix_cache_bypass_reason_.clear();
        LOG_INFO("[DeviceGraphOrchestrator] Prefix cache enabled: block_size="
                 << prefix_layout_.block_size
                 << " block_bytes=" << block_bytes
                 << " ram_budget_bytes=" << prefix_config.ram_budget_bytes
                 << " device_hot_budget_bytes=" << (prefix_device_hot_backend_ ? prefix_config.device_budget_bytes : 0)
                 << " disk_budget_bytes=" << (prefix_disk_backend_ ? prefix_config.disk_budget_bytes : 0)
                 << " disk_dir=" << (prefix_disk_backend_ ? prefix_config.disk_dir : ""));
        return true;
    }

    PrefixCacheKey DeviceGraphOrchestrator::makePrefixKeyForBlock(
        const std::vector<int32_t> &tokens,
        int block_index,
        uint64_t parent_hash) const
    {
        const int start = block_index * prefix_layout_.block_size;
        const int end = std::min<int>(start + prefix_layout_.block_size, tokens.size());
        std::vector<int32_t> block_tokens(tokens.begin() + start, tokens.begin() + end);
        return makePrefixCacheKey(prefix_fingerprint_, parent_hash, block_index, start, block_tokens);
    }

    PrefixLookupResult DeviceGraphOrchestrator::lookupPrefix(const std::vector<int32_t> &tokens)
    {
        PrefixLookupResult result;
        result.cache_enabled = graph_builder_ && graph_builder_->config().prefix_cache.enabled;

        if (!ensurePrefixCacheReady())
        {
            result.bypass_reason = prefix_cache_bypass_reason_;
            return result;
        }

        result.supported = true;
        result.block_size = prefix_layout_.block_size;
        result.fingerprint_key = prefix_fingerprint_;
        result.placement_epoch = moePlacementEpoch();
        result.requires_terminal_logits = prefix_layout_.includes_terminal_logits;
        result.requires_terminal_hidden = prefix_layout_.includes_terminal_hidden;
        if (tokens.empty() || prefix_layout_.block_size <= 0)
        {
            return result;
        }

        const int complete_blocks =
            (static_cast<int>(tokens.size()) + prefix_layout_.block_size - 1) /
            prefix_layout_.block_size;
        uint64_t parent_hash = 0;
        for (int block = 0; block < complete_blocks; ++block)
        {
            PrefixCacheKey key = makePrefixKeyForBlock(tokens, block, parent_hash);
            auto handle = prefix_cache_->find(key);
            if (!handle || !handle->layout.compatiblePayloadShape(prefix_layout_))
            {
                if (prefixCacheTraceEnabled())
                {
                    LOG_INFO("[PREFIX_TRACE] lookup miss device=" << state_.device_id.toString()
                                                                  << " block=" << block
                                                                  << " key=" << key.toHex()
                                                                  << " fingerprint="
                                                                  << prefixHashHex(prefix_fingerprint_)
                                                                  << " handle="
                                                                  << (handle ? "present" : "absent")
                                                                  << " shape_compatible="
                                                                  << (handle && handle->layout.compatiblePayloadShape(prefix_layout_) ? "yes" : "no"));
                }
                break;
            }

            if (prefixCacheTraceEnabled())
            {
                LOG_INFO("[PREFIX_TRACE] lookup hit device=" << state_.device_id.toString()
                                                             << " block=" << block
                                                             << " key=" << key.toHex()
                                                             << " token_count=" << handle->key.token_count
                                                             << " terminal_logits="
                                                             << (handle->has_terminal_logits ? "yes" : "no")
                                                             << " terminal_hidden="
                                                             << (handle->has_terminal_hidden ? "yes" : "no")
                                                             << " mtp_state="
                                                             << (handle->layout.includes_mtp_state ? "yes" : "no"));
            }
            result.blocks.push_back(*handle);
            result.cached_tokens += handle->key.token_count;
            result.has_terminal_hidden = handle->has_terminal_hidden;
            result.has_terminal_logits = handle->has_terminal_logits;
            parent_hash = key.stableHash();
        }

        while (!result.blocks.empty() &&
               prefix_layout_.includes_hybrid_state &&
               !result.blocks.back().has_hybrid_state)
        {
            result.blocks.pop_back();
            result.cached_tokens = result.blocks.empty()
                                       ? 0
                                       : result.blocks.back().key.token_start +
                                             result.blocks.back().key.token_count;
            result.has_terminal_hidden =
                !result.blocks.empty() && result.blocks.back().has_terminal_hidden;
            result.has_terminal_logits =
                !result.blocks.empty() && result.blocks.back().has_terminal_logits;
        }

        prefix_cache_->recordRequestLookup(
            static_cast<int>(tokens.size()),
            result.cached_tokens,
            static_cast<int>(result.blocks.size()));

        if (prefixCacheTraceEnabled())
        {
            LOG_INFO("[PREFIX_TRACE] lookup result device=" << state_.device_id.toString()
                                                            << " cached_tokens=" << result.cached_tokens
                                                            << " blocks=" << result.blocks.size()
                                                            << " terminal_logits="
                                                            << (result.has_terminal_logits ? "yes" : "no")
                                                            << " terminal_hidden="
                                                            << (result.has_terminal_hidden ? "yes" : "no")
                                                            << " fingerprint="
                                                            << prefixHashHex(prefix_fingerprint_));
        }

        return result;
    }

    bool DeviceGraphOrchestrator::populatePrefix(const PrefixLookupResult &hit, int seq_idx)
    {
        if (hit.cached_tokens <= 0)
        {
            return true;
        }
        if (!ensurePrefixCacheReady() || !state_.kv_cache)
        {
            return false;
        }
        if (seq_idx < 0 || seq_idx >= state_.batch_size)
        {
            return false;
        }
        if (hit.blocks.empty())
        {
            return false;
        }

        void *stream = explicitGPUStreamForOperation("populatePrefix");
        if (state_.device_id.is_gpu() && !stream)
        {
            return false;
        }

        for (const auto &handle : hit.blocks)
        {
            if (!handle.valid() || !handle.layout.compatiblePayloadShape(prefix_layout_) ||
                !handle.kvKData() || !handle.kvVData())
            {
                return false;
            }

            for (int local_layer = 0; local_layer < prefix_layout_.fa_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(*state_.kv_cache, local_layer);
                if (global_layer < 0)
                {
                    return false;
                }
                const uint8_t *k_src = handle.kvKData() +
                                       static_cast<size_t>(local_layer) * prefix_layout_.bytes_per_fa_layer_k;
                const uint8_t *v_src = handle.kvVData() +
                                       static_cast<size_t>(local_layer) * prefix_layout_.bytes_per_fa_layer_v;
                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = handle.key.token_start;
                desc.token_count = handle.key.token_count;
                desc.stream = stream;
                if (!state_.kv_cache->importLogicalBlock(desc, k_src, v_src))
                {
                    return false;
                }
            }

            if (prefix_layout_.includes_mtp_state)
            {
                if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
                {
                    return false;
                }
                if (!importMTPPrefixPayload(*state_.mtp_kv_caches[0], seq_idx, handle, stream))
                {
                    return false;
                }
            }
        }

        if (prefix_layout_.includes_hybrid_state &&
            !importHybridPrefixPayload(*state_.kv_cache, hit.blocks.back(), seq_idx, /*synchronize=*/true, stream))
        {
            return false;
        }

        if (graph_builder_ && graph_builder_->config().mtp.enabled &&
            hit.has_terminal_hidden &&
            hit.blocks.back().has_terminal_hidden &&
            hit.blocks.back().terminal_hidden)
        {
            if (!ensureMTPTerminalHiddenBuffer())
                return false;
            void *hidden_dst = state_.prefix_terminal_hidden
                                   ? state_.prefix_terminal_hidden->raw_mutable_data()
                                   : nullptr;
            if (!hidden_dst || prefix_layout_.terminal_hidden_bytes == 0)
                return false;
            std::memcpy(hidden_dst,
                        hit.blocks.back().terminal_hidden,
                        prefix_layout_.terminal_hidden_bytes);
            state_.prefix_terminal_hidden->mark_host_dirty();
            state_.mtp_terminal_hidden_current = true;
            if (state_.device_id.is_gpu())
            {
                auto upload = TransferEngine::instance().uploadFull(
                    state_.prefix_terminal_hidden.get(),
                    state_.device_id,
                    stream);
                if (!upload.success)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload restored prefix terminal hidden: "
                              << upload.error);
                    return false;
                }
            }
            if (arena_ && arena_->isRegistered(BufferId::PREFIX_TERMINAL_HIDDEN))
            {
                arena_->markWrittenFlagsOnly(
                    BufferId::PREFIX_TERMINAL_HIDDEN,
                    state_.device_id.is_gpu() ? state_.device_id : DeviceId::cpu());
            }
        }

        state_.positions[seq_idx] = hit.cached_tokens;
        state_.sequence_lengths[seq_idx] = hit.cached_tokens;
        return true;
    }

    bool DeviceGraphOrchestrator::restorePrefixTerminalState(const PrefixLookupResult &hit)
    {
        TensorBase *terminal_logits_tensor =
            graph_builder_ && graph_builder_->config().lm_head_column_parallel && state_.logits_local
                ? state_.logits_local.get()
                : state_.logits.get();
        const BufferId terminal_logits_buffer_id =
            terminal_logits_tensor == state_.logits_local.get() ? BufferId::LOGITS_LOCAL : BufferId::LOGITS;

        if (!prefix_layout_.includes_terminal_logits ||
            prefix_layout_.terminal_logits_bytes == 0)
        {
            return true;
        }
        if (!hit.has_terminal_logits || hit.blocks.empty() || !terminal_logits_tensor)
        {
            return false;
        }

        const PrefixBlockHandle &terminal = hit.blocks.back();
        if (!terminal.has_terminal_logits || !terminal.terminal_logits)
        {
            return false;
        }

        if (fp32LogitsRowBytes(terminal_logits_tensor) < prefix_layout_.terminal_logits_bytes)
        {
            return false;
        }

        void *dst = terminal_logits_tensor->raw_mutable_data();
        if (!dst)
        {
            return false;
        }
        std::memcpy(dst, terminal.terminal_logits, prefix_layout_.terminal_logits_bytes);
        terminal_logits_tensor->mark_host_dirty();
        if (state_.device_id.is_gpu())
        {
            void *stream = explicitGPUStreamForOperation("restorePrefixTerminalState");
            if (!stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload restored prefix terminal logits: "
                          "missing explicit GPU stream");
                return false;
            }
            auto upload = TransferEngine::instance().uploadFull(
                terminal_logits_tensor,
                state_.device_id,
                stream);
            if (!upload.success)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload restored prefix terminal logits: "
                          << upload.error);
                return false;
            }
        }
        if (arena_ && arena_->isRegistered(terminal_logits_buffer_id))
        {
            arena_->markWrittenFlagsOnly(
                terminal_logits_buffer_id,
                state_.device_id.is_gpu() ? state_.device_id : DeviceId::cpu());
        }
        if (prefix_cache_)
        {
            prefix_cache_->recordTerminalStateHit();
        }

        if (graph_builder_ && graph_builder_->config().mtp.enabled &&
            terminal.has_terminal_hidden && terminal.terminal_hidden)
        {
            if (!ensureMTPTerminalHiddenBuffer())
                return false;
            void *hidden_dst = state_.prefix_terminal_hidden
                                   ? state_.prefix_terminal_hidden->raw_mutable_data()
                                   : nullptr;
            if (!hidden_dst || prefix_layout_.terminal_hidden_bytes == 0)
                return false;
            std::memcpy(hidden_dst, terminal.terminal_hidden, prefix_layout_.terminal_hidden_bytes);
            state_.prefix_terminal_hidden->mark_host_dirty();
            state_.mtp_terminal_hidden_current = true;
            if (state_.device_id.is_gpu())
            {
                void *stream = explicitGPUStreamForOperation("restorePrefixTerminalState");
                if (!stream)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload restored prefix terminal hidden: "
                              "missing explicit GPU stream");
                    return false;
                }
                auto upload = TransferEngine::instance().uploadFull(
                    state_.prefix_terminal_hidden.get(),
                    state_.device_id,
                    stream);
                if (!upload.success)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload restored prefix terminal hidden: "
                              << upload.error);
                    return false;
                }
            }
            if (arena_ && arena_->isRegistered(BufferId::PREFIX_TERMINAL_HIDDEN))
            {
                arena_->markWrittenFlagsOnly(
                    BufferId::PREFIX_TERMINAL_HIDDEN,
                    state_.device_id.is_gpu() ? state_.device_id : DeviceId::cpu());
            }
        }
        return true;
    }

    bool DeviceGraphOrchestrator::harvestPrefix(
        const std::vector<int32_t> &tokens,
        int prompt_token_count)
    {
        if (prompt_token_count <= 0 || prompt_token_count > static_cast<int>(tokens.size()))
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Prefix harvest skipped: invalid prompt_token_count="
                      << prompt_token_count << " token_count=" << tokens.size());
            return false;
        }
        if (!ensurePrefixCacheReady() || !state_.kv_cache || !prefix_ram_backend_)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Prefix harvest skipped: cache_ready="
                      << (prefix_cache_ ? "yes" : "no")
                      << " kv_cache=" << (state_.kv_cache ? "yes" : "no")
                      << " ram_backend=" << (prefix_ram_backend_ ? "yes" : "no")
                      << " bypass_reason=" << prefix_cache_bypass_reason_);
            return false;
        }

        void *stream = explicitGPUStreamForOperation("harvestPrefix");
        if (state_.device_id.is_gpu() && !stream)
        {
            return false;
        }

        const int complete_blocks =
            (prompt_token_count + prefix_layout_.block_size - 1) /
            prefix_layout_.block_size;

        uint64_t parent_hash = 0;
        for (int block = 0; block < complete_blocks; ++block)
        {
            PrefixCacheKey key = makePrefixKeyForBlock(tokens, block, parent_hash);
            parent_hash = key.stableHash();
            if (prefix_cache_->contains(key))
            {
                if (prefixCacheTraceEnabled())
                {
                    LOG_INFO("[PREFIX_TRACE] harvest skip-existing device=" << state_.device_id.toString()
                                                                            << " block=" << block
                                                                            << " key=" << key.toHex());
                }
                continue;
            }

            const bool terminal_block =
                (key.token_start + key.token_count) == prompt_token_count;
            PrefixPayloadLayout block_layout = prefix_layout_;
            if (!terminal_block)
            {
                block_layout.includes_hybrid_state = false;
                block_layout.includes_terminal_hidden = false;
                block_layout.includes_terminal_logits = false;
            }

            if (!prefix_cache_->reserveRam(block_layout.totalBytes()))
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] Prefix harvest failed: RAM reserve rejected block_bytes="
                          << block_layout.totalBytes());
                return false;
            }

            PrefixBlockHandle handle = prefix_ram_backend_->allocate(key, block_layout);
            if (!handle.valid())
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] Prefix harvest failed: RAM allocate rejected key="
                          << key.toHex()
                          << " block_bytes=" << block_layout.totalBytes());
                return false;
            }

            bool ok = true;
            for (int local_layer = 0; local_layer < prefix_layout_.fa_layers && ok; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(*state_.kv_cache, local_layer);
                if (global_layer < 0)
                {
                    LOG_DEBUG("[DeviceGraphOrchestrator] Prefix harvest failed: no FA layer for local index "
                              << local_layer);
                    ok = false;
                    break;
                }
                uint8_t *k_dst = handle.kvKData() +
                                 static_cast<size_t>(local_layer) * prefix_layout_.bytes_per_fa_layer_k;
                uint8_t *v_dst = handle.kvVData() +
                                 static_cast<size_t>(local_layer) * prefix_layout_.bytes_per_fa_layer_v;
                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = 0;
                desc.logical_token_start = key.token_start;
                desc.token_count = key.token_count;
                desc.stream = stream;
                ok = state_.kv_cache->exportLogicalBlock(desc, k_dst, v_dst);
                if (!ok)
                {
                    const auto seq_state = state_.kv_cache->sequenceState(global_layer, desc.seq_idx);
                    LOG_DEBUG("[DeviceGraphOrchestrator] Prefix harvest failed: exportLogicalBlock layer="
                              << global_layer
                              << " token_start=" << desc.logical_token_start
                              << " token_count=" << desc.token_count
                              << " cached_tokens=" << seq_state.cached_tokens
                              << " first_layer_index=" << state_.kv_cache->first_layer_index()
                              << " total_layers=" << state_.kv_cache->n_layers());
                }
            }

            if (ok && prefix_layout_.includes_mtp_state)
            {
                if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
                {
                    LOG_DEBUG("[DeviceGraphOrchestrator] Prefix harvest failed: missing MTP KV cache");
                    ok = false;
                }
                else
                {
                    ok = exportMTPPrefixPayload(*state_.mtp_kv_caches[0], 0, key, &handle, stream);
                    if (!ok)
                    {
                        LOG_DEBUG("[DeviceGraphOrchestrator] Prefix harvest failed: MTP payload export failed");
                    }
                }
            }

            if (ok && terminal_block && prefix_layout_.includes_hybrid_state)
            {
                ok = exportHybridPrefixPayload(
                    *state_.kv_cache,
                    0,
                    key.token_start + key.token_count,
                    &handle,
                    /*synchronize=*/true,
                    stream);
                if (!ok)
                {
                    LOG_DEBUG("[DeviceGraphOrchestrator] Prefix harvest failed: hybrid payload export failed");
                }
            }

            if (ok && terminal_block && prefix_layout_.includes_terminal_hidden &&
                !state_.mtp_terminal_hidden_current)
            {
                /*
                 * Prefix full hits with MTP need the terminal hidden row from
                 * the prompt prefill before the first draft sidecar can run.
                 * MTP sidecar execution lazily refreshes this row too, but
                 * prefix harvest happens immediately after prefill and before
                 * decode.  Refresh it here so a stored terminal block never
                 * advertises MTP state while silently lacking terminal hidden.
                 */
                ok = refreshMTPTerminalHiddenState(prompt_token_count, 1);
                if (!ok)
                {
                    LOG_DEBUG("[DeviceGraphOrchestrator] Prefix harvest failed: terminal hidden refresh failed");
                }
            }

            TensorBase *terminal_logits_tensor =
                graph_builder_ && graph_builder_->config().lm_head_column_parallel && state_.logits_local
                    ? state_.logits_local.get()
                    : state_.logits.get();
            if (ok && terminal_block && prefix_layout_.includes_terminal_logits &&
                terminal_logits_tensor && handle.terminal_logits)
            {
                if (fp32LogitsRowBytes(terminal_logits_tensor) < prefix_layout_.terminal_logits_bytes)
                {
                    ok = false;
                }
                if (state_.device_id.is_gpu())
                {
                    auto download = TransferEngine::instance().download(terminal_logits_tensor);
                    if (!download.success)
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Failed to download prefix terminal logits: "
                                  << download.error);
                        ok = false;
                    }
                }
                const float *logits = ok ? terminal_logits_tensor->fp32_data() : nullptr;
                if (ok && logits)
                {
                    std::memcpy(handle.terminal_logits, logits, prefix_layout_.terminal_logits_bytes);
                    handle.has_terminal_logits = true;
                }
                else if (ok)
                {
                    LOG_DEBUG("[DeviceGraphOrchestrator] Prefix harvest failed: terminal logits host pointer unavailable");
                    ok = false;
                }
            }

            if (ok && terminal_block && prefix_layout_.includes_terminal_hidden &&
                state_.prefix_terminal_hidden &&
                state_.mtp_terminal_hidden_current &&
                handle.terminal_hidden)
            {
                if (state_.device_id.is_gpu())
                {
                    auto download = TransferEngine::instance().download(state_.prefix_terminal_hidden.get());
                    if (!download.success)
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Failed to download prefix terminal hidden: "
                                  << download.error);
                        ok = false;
                    }
                }
                const void *hidden = ok ? state_.prefix_terminal_hidden->raw_data() : nullptr;
                if (ok && hidden)
                {
                    std::memcpy(handle.terminal_hidden, hidden, prefix_layout_.terminal_hidden_bytes);
                    handle.has_terminal_hidden = true;
                }
                else if (ok)
                {
                    LOG_DEBUG("[DeviceGraphOrchestrator] Prefix harvest failed: terminal hidden host pointer unavailable");
                    ok = false;
                }
            }

            if (!ok || !prefix_cache_->insert(handle))
            {
                if (ok)
                {
                    LOG_DEBUG("[DeviceGraphOrchestrator] Prefix harvest failed: cache insert rejected key="
                              << key.toHex());
                }
                if (prefixCacheTraceEnabled())
                {
                    LOG_INFO("[PREFIX_TRACE] harvest failed device=" << state_.device_id.toString()
                                                                     << " block=" << block
                                                                     << " key=" << key.toHex()
                                                                     << " ok_before_insert="
                                                                     << (ok ? "yes" : "no")
                                                                     << " terminal_block="
                                                                     << (terminal_block ? "yes" : "no")
                                                                     << " terminal_logits="
                                                                     << (handle.has_terminal_logits ? "yes" : "no")
                                                                     << " terminal_hidden="
                                                                     << (handle.has_terminal_hidden ? "yes" : "no")
                                                                     << " mtp_state="
                                                                     << (handle.layout.includes_mtp_state ? "yes" : "no")
                                                                     << " total_bytes=" << handle.total_bytes);
                }
                prefix_ram_backend_->release(handle);
                return false;
            }

            if (prefixCacheTraceEnabled())
            {
                LOG_INFO("[PREFIX_TRACE] harvest insert device=" << state_.device_id.toString()
                                                                 << " block=" << block
                                                                 << " key=" << key.toHex()
                                                                 << " token_count=" << key.token_count
                                                                 << " terminal_logits="
                                                                 << (handle.has_terminal_logits ? "yes" : "no")
                                                                 << " terminal_hidden="
                                                                 << (handle.has_terminal_hidden ? "yes" : "no")
                                                                 << " mtp_state="
                                                                 << (handle.layout.includes_mtp_state ? "yes" : "no")
                                                                 << " total_bytes=" << handle.total_bytes);
            }
        }

        return true;
    }

    PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixState(int seq_idx) const
    {
        PrefixStateSnapshot snapshot;
        if (!state_.kv_cache || seq_idx < 0 || seq_idx >= state_.batch_size)
        {
            return snapshot;
        }

        const int cached_tokens = restorablePrefixCachedTokens(*state_.kv_cache, seq_idx);
        if (cached_tokens < 0 || cached_tokens > state_.kv_cache->max_seq_len())
        {
            return snapshot;
        }

        void *stream = explicitGPUStreamForOperation("captureLivePrefixState");
        if (state_.device_id.is_gpu() && !stream)
        {
            return {};
        }

        snapshot.valid = true;
        snapshot.provenance = PrefixStateProvenance::PayloadCheckpoint;
        snapshot.cached_tokens = cached_tokens;
        if (cached_tokens > 0)
        {
            size_t live_terminal_hidden_bytes = 0;
            if (graph_builder_ &&
                graph_builder_->config().mtp.enabled &&
                state_.prefix_terminal_hidden &&
                state_.mtp_terminal_hidden_current &&
                state_.d_model > 0)
            {
                live_terminal_hidden_bytes =
                    static_cast<size_t>(state_.d_model) * sizeof(float);
            }

            PrefixPayloadLayout layout = buildDensePrefixPayloadLayout(
                *state_.kv_cache,
                state_.device_id,
                cached_tokens,
                live_terminal_hidden_bytes);

            PrefixBlockHandle handle;
            handle.key.fingerprint = prefix_fingerprint_ != 0 ? prefix_fingerprint_ : 1;
            handle.key.block_index = 0;
            handle.key.token_start = 0;
            handle.key.token_count = cached_tokens;
            handle.layout = layout;
            handle.total_bytes = layout.totalBytes();

            const size_t kv_bytes = layout.faKVBytes();
            if (kv_bytes == 0)
            {
                return {};
            }
            handle.kv_storage = std::make_shared<std::vector<uint8_t>>(kv_bytes, 0);
            handle.kv_payload = handle.kv_storage->data();
            if (layout.includes_hybrid_state && layout.hybrid_state_bytes > 0)
            {
                handle.hybrid_storage = std::make_shared<std::vector<uint8_t>>(layout.hybrid_state_bytes, 0);
                handle.hybrid_payload = handle.hybrid_storage->data();
            }

            for (int local_layer = 0; local_layer < layout.fa_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(*state_.kv_cache, local_layer);
                if (global_layer < 0)
                {
                    return {};
                }
                uint8_t *k_dst = handle.kvKData() +
                                 static_cast<size_t>(local_layer) * layout.bytes_per_fa_layer_k;
                uint8_t *v_dst = handle.kvVData() +
                                 static_cast<size_t>(local_layer) * layout.bytes_per_fa_layer_v;

                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = 0;
                desc.token_count = cached_tokens;
                desc.stream = stream;
                if (!state_.kv_cache->exportLogicalBlock(desc, k_dst, v_dst))
                {
                    return {};
                }
            }

            if (!exportHybridPrefixPayload(
                    *state_.kv_cache,
                    seq_idx,
                    cached_tokens,
                    &handle,
                    /*synchronize=*/true,
                    stream))
            {
                return {};
            }

            if (live_terminal_hidden_bytes > 0)
            {
                handle.terminal_hidden_storage =
                    std::make_shared<std::vector<uint8_t>>(live_terminal_hidden_bytes, 0);
                handle.terminal_hidden = handle.terminal_hidden_storage->data();
                if (!handle.terminal_hidden)
                {
                    return {};
                }
                if (state_.device_id.is_gpu())
                {
                    auto download = TransferEngine::instance().downloadFull(
                        state_.prefix_terminal_hidden.get(),
                        stream);
                    if (!download.success)
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Failed to download payload checkpoint terminal hidden: "
                                  << download.error);
                        return {};
                    }
                }
                const void *hidden = state_.prefix_terminal_hidden->raw_data();
                if (!hidden)
                {
                    return {};
                }
                std::memcpy(
                    handle.terminal_hidden,
                    hidden,
                    live_terminal_hidden_bytes);
                handle.has_terminal_hidden = true;
            }

            snapshot.blocks.push_back(std::move(handle));
        }

        for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
        {
            const auto &cache = state_.mtp_kv_caches[depth];
            if (!cache)
            {
                continue;
            }

            const int mtp_cached_tokens = restorablePrefixCachedTokens(*cache, seq_idx);
            if (mtp_cached_tokens < 0 || mtp_cached_tokens > cache->max_seq_len())
            {
                return {};
            }
            if (mtp_cached_tokens == 0)
            {
                continue;
            }

            PrefixPayloadLayout layout = buildDensePrefixPayloadLayout(
                *cache,
                state_.device_id,
                mtp_cached_tokens);

            PrefixBlockHandle handle;
            handle.key.fingerprint = prefix_fingerprint_ != 0 ? prefix_fingerprint_ : 1;
            handle.key.block_index = static_cast<int>(depth);
            handle.key.token_start = 0;
            handle.key.token_count = mtp_cached_tokens;
            handle.layout = layout;
            handle.total_bytes = layout.totalBytes();

            const size_t kv_bytes = layout.faKVBytes();
            if (kv_bytes == 0)
            {
                return {};
            }
            handle.kv_storage = std::make_shared<std::vector<uint8_t>>(kv_bytes, 0);
            handle.kv_payload = handle.kv_storage->data();

            for (int local_layer = 0; local_layer < layout.fa_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(*cache, local_layer);
                if (global_layer < 0)
                {
                    return {};
                }
                uint8_t *k_dst = handle.kvKData() +
                                 static_cast<size_t>(local_layer) * layout.bytes_per_fa_layer_k;
                uint8_t *v_dst = handle.kvVData() +
                                 static_cast<size_t>(local_layer) * layout.bytes_per_fa_layer_v;

                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = 0;
                desc.token_count = mtp_cached_tokens;
                desc.stream = stream;
                if (!cache->exportLogicalBlock(desc, k_dst, v_dst))
                {
                    return {};
                }
            }

            snapshot.mtp_blocks.push_back(std::move(handle));
        }

        return snapshot;
    }

    bool DeviceGraphOrchestrator::ensureLiveHybridCheckpointStorage(PrefixBlockHandle &handle) const
    {
        if (!handle.layout.includes_hybrid_state)
        {
            return true;
        }

        return acquireLiveHybridCheckpointStorage(handle);
    }

    bool DeviceGraphOrchestrator::acquireLiveHybridCheckpointStorage(PrefixBlockHandle &handle) const
    {
        if (!handle.layout.includes_hybrid_state)
        {
            return true;
        }

        const size_t host_bytes = handle.layout.hybrid_host_state_bytes;
        const size_t device_bytes = handle.layout.hybrid_device_state_bytes;
        const bool needs_host = host_bytes > 0;
        const bool needs_device = device_bytes > 0;
        auto assign_slot = [&](LiveHybridCheckpointStorageSlot &slot)
        {
            if (needs_host)
            {
                slot.host_storage->resize(host_bytes);
                handle.hybrid_storage = slot.host_storage;
                handle.hybrid_payload = slot.host_storage->data();
            }
            else
            {
                handle.hybrid_storage.reset();
                handle.hybrid_payload = nullptr;
            }

            if (needs_device)
            {
                handle.device_hybrid_storage = slot.device_storage;
            }
            else
            {
                handle.device_hybrid_storage.reset();
            }
        };

        for (auto &slot : live_hybrid_checkpoint_storage_pool_)
        {
            const bool slot_free =
                (!slot.host_storage || slot.host_storage.use_count() == 1) &&
                (!slot.device_storage || slot.device_storage.use_count() == 1);
            const bool host_ok =
                !needs_host ||
                (slot.host_storage && slot.host_capacity_bytes >= host_bytes);
            const bool device_ok =
                !needs_device ||
                (slot.device_storage &&
                 slot.device_capacity_bytes >= device_bytes &&
                 slot.device == state_.device_id);
            if (!slot_free || !host_ok || !device_ok)
            {
                continue;
            }

            assign_slot(slot);
            PerfStatsCollector::addCounter(
                "mtp",
                "live_prefix_checkpoint_hybrid_storage_pool_hits",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"host_bytes", std::to_string(host_bytes)},
                 {"device_bytes", std::to_string(device_bytes)}});
            return true;
        }

        LiveHybridCheckpointStorageSlot slot;
        slot.device = state_.device_id;

        if (needs_host)
        {
            slot.host_storage =
                std::make_shared<std::vector<uint8_t>>(host_bytes);
            slot.host_capacity_bytes = slot.host_storage->capacity();
        }

        if (needs_device)
        {
            slot.device_storage =
                allocateDeviceByteStorage(device_bytes, state_.device_id);
            if (!slot.device_storage)
            {
                return false;
            }
            slot.device_capacity_bytes = device_bytes;
        }

        live_hybrid_checkpoint_storage_pool_.push_back(std::move(slot));
        assign_slot(live_hybrid_checkpoint_storage_pool_.back());
        PerfStatsCollector::addCounter(
            "mtp",
            "live_prefix_checkpoint_hybrid_storage_pool_misses",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"host_bytes", std::to_string(host_bytes)},
             {"device_bytes", std::to_string(device_bytes)},
             {"pool_size", std::to_string(live_hybrid_checkpoint_storage_pool_.size())}});
        return true;
    }

    PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixCheckpoint(int seq_idx) const
    {
        PrefixStateSnapshot snapshot;
        if (!state_.kv_cache || seq_idx < 0 || seq_idx >= state_.batch_size)
        {
            return snapshot;
        }

        const int cached_tokens = restorablePrefixCachedTokens(*state_.kv_cache, seq_idx);
        if (cached_tokens < 0 || cached_tokens > state_.kv_cache->max_seq_len())
        {
            return snapshot;
        }

        void *stream = explicitGPUStreamForOperation("captureLivePrefixCheckpoint");
        if (state_.device_id.is_gpu() && !stream)
        {
            return {};
        }

        const int draft_tokens =
            graph_builder_ ? std::max(1, graph_builder_->config().mtp.draft_tokens) : 1;
        const int main_headroom = draft_tokens + 2;
        if (liveCheckpointLacksHeadroom(*state_.kv_cache, cached_tokens, main_headroom))
        {
            PerfStatsCollector::addCounter("mtp", "live_prefix_checkpoint_payload_required", 1.0, "decode",
                                           state_.device_id.toString(),
                                           {{"reason", "main_cache_headroom"}});
            return captureLivePrefixState(seq_idx);
        }

        snapshot.valid = true;
        snapshot.logical_checkpoint = true;
        snapshot.provenance = PrefixStateProvenance::LogicalCheckpoint;
        snapshot.cached_tokens = cached_tokens;
        snapshot.mtp_cached_tokens.assign(state_.mtp_kv_caches.size(), -1);

        PrefixPayloadLayout main_layout;
        if (cached_tokens > 0)
        {
            PerfStatsCollector::ScopedTimer timer("mtp",
                                                  "live_prefix_checkpoint_layout",
                                                  "decode",
                                                  state_.device_id.toString(),
                                                  {{"cache", "main"}});
            main_layout = buildDensePrefixPayloadLayout(
                *state_.kv_cache,
                state_.device_id,
                cached_tokens);
        }

        size_t live_terminal_hidden_bytes = 0;
        const float *live_terminal_hidden_source = nullptr;
        const bool mtp_terminal_hidden_checkpoint_enabled =
            graph_builder_ &&
            graph_builder_->config().mtp.enabled &&
            cached_tokens > 0 &&
            state_.d_model > 0;
        if (graph_builder_ &&
            graph_builder_->config().mtp.enabled &&
            state_.prefix_terminal_hidden &&
            state_.mtp_terminal_hidden_current &&
            cached_tokens > 0 &&
            state_.d_model > 0)
        {
            live_terminal_hidden_bytes =
                static_cast<size_t>(state_.d_model) * sizeof(float);
        }
        else if (mtp_terminal_hidden_checkpoint_enabled &&
                 !state_.device_id.is_gpu() &&
                 state_.last_forward_batch_size == 1 &&
                 state_.last_forward_seq_len > 0 &&
                 state_.hidden)
        {
            /*
             * CPU logical checkpoints do not need to materialize the compact
             * PREFIX_TERMINAL_HIDDEN tensor just to publish accepted MTP state.
             * If the main forward produced a multi-row hidden tensor, copy the
             * terminal row directly from HIDDEN_STATE.  GPU keeps using the
             * explicit row-select path so stream ownership remains visible.
             */
            const auto &hidden_shape = state_.hidden->shape();
            const size_t hidden_rows = hidden_shape.empty() ? 0 : hidden_shape[0];
            const size_t terminal_row =
                static_cast<size_t>(std::max(0, state_.last_forward_seq_len - 1));
            const float *hidden_rows_fp32 = state_.hidden->fp32_data();
            if (hidden_rows_fp32 &&
                hidden_rows > terminal_row &&
                hidden_shape.size() >= 2 &&
                hidden_shape[1] >= static_cast<size_t>(state_.d_model))
            {
                live_terminal_hidden_source =
                    hidden_rows_fp32 + terminal_row * static_cast<size_t>(state_.d_model);
                live_terminal_hidden_bytes =
                    static_cast<size_t>(state_.d_model) * sizeof(float);
            }
        }

        if ((main_layout.includes_hybrid_state && main_layout.hybrid_state_bytes > 0) ||
            live_terminal_hidden_bytes > 0)
        {
            PrefixBlockHandle handle;
            handle.key.fingerprint = prefix_fingerprint_ != 0 ? prefix_fingerprint_ : 1;
            handle.key.block_index = 0;
            handle.key.token_start = 0;
            handle.key.token_count = cached_tokens;
            const bool device_only_checkpoint =
                state_.device_id.is_gpu() && main_layout.hybrid_device_state_bytes > 0;
            handle.layout = liveHybridCheckpointLayout(main_layout, device_only_checkpoint);
            if (live_terminal_hidden_bytes > 0)
            {
                handle.layout.terminal_hidden_bytes = live_terminal_hidden_bytes;
                handle.layout.includes_terminal_hidden = true;
            }
            handle.total_bytes = handle.layout.totalBytes();
            PerfStatsCollector::addCounter("mtp",
                                           "live_prefix_checkpoint_hybrid_host_bytes",
                                           static_cast<double>(handle.layout.hybrid_host_state_bytes),
                                           "decode",
                                           state_.device_id.toString());
            PerfStatsCollector::addCounter("mtp",
                                           "live_prefix_checkpoint_hybrid_device_bytes",
                                           static_cast<double>(handle.layout.hybrid_device_state_bytes),
                                           "decode",
                                           state_.device_id.toString());
            if (device_only_checkpoint)
            {
                PerfStatsCollector::addCounter("mtp",
                                               "live_prefix_checkpoint_hybrid_device_only_captures",
                                               1.0,
                                               "decode",
                                               state_.device_id.toString());
            }
            {
                PerfStatsCollector::ScopedTimer timer("mtp",
                                                      "live_prefix_checkpoint_hybrid_storage",
                                                      "decode",
                                                      state_.device_id.toString());
                if (!ensureLiveHybridCheckpointStorage(handle))
                    return {};
            }
            if (handle.layout.includes_hybrid_state && handle.layout.hybrid_state_bytes > 0)
            {
                PerfStatsCollector::ScopedTimer timer("mtp",
                                                      "live_prefix_checkpoint_hybrid_export",
                                                      "decode",
                                                      state_.device_id.toString());
                if (!exportHybridPrefixPayload(
                        *state_.kv_cache,
                        seq_idx,
                        cached_tokens,
                        &handle,
                        /*synchronize=*/false,
                        stream))
                {
                    return {};
                }
                if (!handle.has_hybrid_state)
                {
                    return {};
                }
            }
            if (live_terminal_hidden_bytes > 0)
            {
                handle.terminal_hidden_storage =
                    std::make_shared<std::vector<uint8_t>>(live_terminal_hidden_bytes, 0);
                handle.terminal_hidden = handle.terminal_hidden_storage->data();
                if (!handle.terminal_hidden)
                {
                    return {};
                }
                if (state_.device_id.is_gpu())
                {
                    auto download = TransferEngine::instance().downloadFull(
                        state_.prefix_terminal_hidden.get(),
                        stream);
                    if (!download.success)
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Failed to download live checkpoint terminal hidden: "
                                  << download.error);
                        return {};
                    }
                }
                const void *hidden = live_terminal_hidden_source
                                         ? static_cast<const void *>(live_terminal_hidden_source)
                                         : state_.prefix_terminal_hidden->raw_data();
                if (!hidden)
                {
                    return {};
                }
                std::memcpy(
                    handle.terminal_hidden,
                    hidden,
                    live_terminal_hidden_bytes);
                handle.has_terminal_hidden = true;
                PerfStatsCollector::addCounter(
                    "mtp",
                    "live_prefix_checkpoint_terminal_hidden_captures",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"bytes", std::to_string(live_terminal_hidden_bytes)}});
            }
            snapshot.blocks.push_back(std::move(handle));
            if (main_layout.includes_hybrid_state && main_layout.hybrid_state_bytes > 0)
            {
                PerfStatsCollector::addCounter("mtp", "live_prefix_checkpoint_hybrid_state_captures", 1.0, "decode",
                                               state_.device_id.toString());
            }
        }

        for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
        {
            const auto &cache = state_.mtp_kv_caches[depth];
            if (!cache)
            {
                continue;
            }

            const int mtp_cached_tokens = restorablePrefixCachedTokens(*cache, seq_idx);
            if (mtp_cached_tokens < 0 || mtp_cached_tokens > cache->max_seq_len())
            {
                return {};
            }
            bool mtp_has_hybrid_state = false;
            {
                PerfStatsCollector::ScopedTimer timer("mtp",
                                                      "live_prefix_checkpoint_layout",
                                                      "decode",
                                                      state_.device_id.toString(),
                                                      {{"cache", "mtp"}});
                mtp_has_hybrid_state = liveCheckpointHasHybridState(
                    *cache,
                    state_.device_id,
                    mtp_cached_tokens);
            }
            if (mtp_has_hybrid_state)
            {
                PerfStatsCollector::addCounter("mtp", "live_prefix_checkpoint_payload_required", 1.0, "decode",
                                               state_.device_id.toString(),
                                               {{"reason", "mtp_hybrid_state"}});
                return captureLivePrefixState(seq_idx);
            }
            if (liveCheckpointLacksHeadroom(*cache, mtp_cached_tokens, draft_tokens))
            {
                PerfStatsCollector::addCounter("mtp", "live_prefix_checkpoint_payload_required", 1.0, "decode",
                                               state_.device_id.toString(),
                                               {{"reason", "mtp_cache_headroom"}});
                return captureLivePrefixState(seq_idx);
            }
            snapshot.mtp_cached_tokens[depth] = mtp_cached_tokens;
        }

        PerfStatsCollector::addCounter("mtp", "live_prefix_checkpoint_logical_captures", 1.0, "decode",
                                       state_.device_id.toString());
        return snapshot;
    }

    bool DeviceGraphOrchestrator::restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx)
    {
        auto fail = [](const std::string &reason) -> bool
        {
            LOG_ERROR("[DeviceGraphOrchestrator] restoreLivePrefixState failed: " << reason);
            return false;
        };

        if (!snapshot.valid || !state_.kv_cache || seq_idx < 0 || seq_idx >= state_.batch_size)
        {
            return fail("invalid snapshot, missing KV cache, or sequence index out of range");
        }
        if (snapshot.cached_tokens < 0 || snapshot.cached_tokens > state_.kv_cache->max_seq_len())
        {
            return fail("cached token count out of range: cached_tokens=" +
                        std::to_string(snapshot.cached_tokens) +
                        " max_seq_len=" + std::to_string(state_.kv_cache->max_seq_len()));
        }

        void *stream = explicitGPUStreamForOperation("restoreLivePrefixState");
        if (state_.device_id.is_gpu() && !stream)
        {
            return fail("missing explicit GPU stream");
        }

        if (snapshot.logical_checkpoint)
        {
            if (!snapshot.mtp_blocks.empty())
            {
                return fail("logical checkpoint unexpectedly contains MTP payload blocks");
            }
            if (snapshot.blocks.size() > 1)
            {
                return fail("logical checkpoint contains more than one hybrid payload block");
            }
            const PrefixBlockHandle *hybrid_handle = nullptr;
            if (!snapshot.blocks.empty())
            {
                hybrid_handle = &snapshot.blocks.front();
                const bool includes_hybrid_state =
                    hybrid_handle->layout.includes_hybrid_state &&
                    hybrid_handle->layout.hybrid_state_bytes > 0;
                const bool includes_terminal_hidden =
                    hybrid_handle->layout.includes_terminal_hidden &&
                    hybrid_handle->layout.terminal_hidden_bytes > 0;
                if ((!includes_hybrid_state && !includes_terminal_hidden) ||
                    (includes_hybrid_state &&
                     (!hybrid_handle->has_hybrid_state ||
                      (hybrid_handle->layout.hybrid_host_state_bytes > 0 && !hybrid_handle->hybrid_payload) ||
                      (hybrid_handle->layout.hybrid_device_state_bytes > 0 && !hybridPayloadDevicePtr(*hybrid_handle)))) ||
                    (includes_terminal_hidden &&
                     (!hybrid_handle->has_terminal_hidden || !hybrid_handle->terminal_hidden)) ||
                    hybrid_handle->key.token_start != 0 ||
                    hybrid_handle->key.token_count != snapshot.cached_tokens)
                {
                    return fail("logical checkpoint payload block is invalid");
                }
            }
            if (snapshot.mtp_cached_tokens.size() != state_.mtp_kv_caches.size())
            {
                return fail("logical checkpoint MTP cache count does not match runner state");
            }
            if (!state_.kv_cache->truncateSequence(seq_idx, snapshot.cached_tokens, stream))
            {
                return fail("main KV logical checkpoint truncate failed for tokens=" +
                            std::to_string(snapshot.cached_tokens));
            }
            if (snapshot.cached_tokens == 0)
            {
                if (state_.device_id.is_gpu() &&
                    dynamic_cast<IHybridKVCache *>(state_.kv_cache.get()))
                {
                    return fail("zero-token hybrid logical checkpoint restore requires streamful GDN reset");
                }
                resetHybridPrefixPayloadState(*state_.kv_cache);
            }
            else if (hybrid_handle)
            {
                if (hybrid_handle->layout.includes_hybrid_state &&
                    hybrid_handle->layout.hybrid_state_bytes > 0)
                {
                    PerfStatsCollector::ScopedTimer timer("mtp",
                                                          "restore_live_prefix_checkpoint_hybrid_import",
                                                          "decode",
                                                          state_.device_id.toString());
                    if (!importHybridPrefixPayload(
                            *state_.kv_cache,
                            *hybrid_handle,
                            seq_idx,
                            /*synchronize=*/false,
                            stream))
                    {
                        return fail("logical checkpoint hybrid payload import failed");
                    }
                }
                if (hybrid_handle->layout.includes_terminal_hidden &&
                    hybrid_handle->layout.terminal_hidden_bytes > 0)
                {
                    if (!ensureMTPTerminalHiddenBuffer())
                    {
                        return fail("logical checkpoint terminal hidden buffer allocation failed");
                    }
                    void *hidden_dst = state_.prefix_terminal_hidden
                                           ? state_.prefix_terminal_hidden->raw_mutable_data()
                                           : nullptr;
                    if (!hidden_dst || !hybrid_handle->terminal_hidden)
                    {
                        return fail("logical checkpoint terminal hidden payload is unavailable");
                    }
                    std::memcpy(
                        hidden_dst,
                        hybrid_handle->terminal_hidden,
                        hybrid_handle->layout.terminal_hidden_bytes);
                    state_.prefix_terminal_hidden->mark_host_dirty();
                    state_.mtp_terminal_hidden_current = true;
                    if (state_.device_id.is_gpu())
                    {
                        auto upload = TransferEngine::instance().uploadFull(
                            state_.prefix_terminal_hidden.get(),
                            state_.device_id,
                            stream);
                        if (!upload.success)
                        {
                            return fail("logical checkpoint terminal hidden upload failed: " +
                                        upload.error);
                        }
                    }
                    if (arena_ && arena_->isRegistered(BufferId::PREFIX_TERMINAL_HIDDEN))
                    {
                        arena_->markWrittenFlagsOnly(
                            BufferId::PREFIX_TERMINAL_HIDDEN,
                            state_.device_id.is_gpu() ? state_.device_id : DeviceId::cpu());
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "restore_live_prefix_checkpoint_terminal_hidden",
                        1.0,
                        "decode",
                        state_.device_id.toString(),
                        {{"bytes", std::to_string(hybrid_handle->layout.terminal_hidden_bytes)}});
                }
            }

            for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
            {
                auto &cache = state_.mtp_kv_caches[depth];
                const int mtp_cached_tokens = snapshot.mtp_cached_tokens[depth];
                if (!cache)
                {
                    if (mtp_cached_tokens >= 0)
                    {
                        return fail("logical checkpoint contains MTP tokens for a missing cache");
                    }
                    continue;
                }
                if (mtp_cached_tokens < 0 || mtp_cached_tokens > cache->max_seq_len())
                {
                    return fail("logical checkpoint MTP token count out of range for depth=" +
                                std::to_string(depth) +
                                " tokens=" + std::to_string(mtp_cached_tokens));
                }
                if (!cache->truncateSequence(seq_idx, mtp_cached_tokens, stream))
                {
                    return fail("MTP KV logical checkpoint truncate failed for depth=" +
                                std::to_string(depth) +
                                " tokens=" + std::to_string(mtp_cached_tokens));
                }
            }

            state_.positions[seq_idx] = snapshot.cached_tokens;
            state_.sequence_lengths[seq_idx] = snapshot.cached_tokens;
            handleLivePrefixReplayStateAfterMutation(
                LivePrefixMutationReason::PrefixRestore,
                "restore_logical_checkpoint");
            return true;
        }

        if (!state_.kv_cache->truncateSequence(seq_idx, 0, stream))
        {
            return fail("main KV payload checkpoint clear failed");
        }
        for (auto &cache : state_.mtp_kv_caches)
        {
            if (cache)
            {
                if (!cache->truncateSequence(seq_idx, 0, stream))
                {
                    return fail("MTP KV payload checkpoint clear failed");
                }
            }
        }

        if (snapshot.cached_tokens == 0)
        {
            if (!snapshot.mtp_blocks.empty())
            {
                return fail("zero-token snapshot unexpectedly contains MTP blocks");
            }
            if (state_.device_id.is_gpu() &&
                dynamic_cast<IHybridKVCache *>(state_.kv_cache.get()))
            {
                return fail("zero-token hybrid payload checkpoint restore requires streamful GDN reset");
            }
            resetHybridPrefixPayloadState(*state_.kv_cache);
            state_.positions[seq_idx] = 0;
            state_.sequence_lengths[seq_idx] = 0;
            handleLivePrefixReplayStateAfterMutation(
                LivePrefixMutationReason::PrefixRestore,
                "restore_payload_checkpoint_zero");
            return true;
        }

        for (const auto &handle : snapshot.blocks)
        {
            if (!handle.valid() || !handle.kvKData() || !handle.kvVData())
            {
                return fail("main KV block handle is invalid");
            }

            PrefixPayloadLayout expected = buildDensePrefixPayloadLayout(
                *state_.kv_cache,
                state_.device_id,
                handle.key.token_count,
                handle.layout.terminal_hidden_bytes,
                handle.layout.terminal_logits_bytes);
            if (!handle.layout.compatiblePayloadShape(expected))
            {
                return fail("main KV block layout is incompatible");
            }

            for (int local_layer = 0; local_layer < handle.layout.fa_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(*state_.kv_cache, local_layer);
                if (global_layer < 0)
                {
                    return fail("main KV block layer index is invalid");
                }
                const uint8_t *k_src = handle.kvKData() +
                                       static_cast<size_t>(local_layer) * handle.layout.bytes_per_fa_layer_k;
                const uint8_t *v_src = handle.kvVData() +
                                       static_cast<size_t>(local_layer) * handle.layout.bytes_per_fa_layer_v;

                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = handle.key.token_start;
                desc.token_count = handle.key.token_count;
                desc.stream = stream;
                if (!state_.kv_cache->importLogicalBlock(desc, k_src, v_src))
                {
                    return fail("main KV logical block import failed for layer=" +
                                std::to_string(global_layer) +
                                " tokens=" + std::to_string(handle.key.token_count));
                }
            }
        }

        if (!snapshot.blocks.empty() &&
            !importHybridPrefixPayload(
                *state_.kv_cache,
                snapshot.blocks.back(),
                seq_idx,
                /*synchronize=*/true,
                stream))
        {
            return fail("hybrid payload import failed");
        }

        if (!snapshot.blocks.empty())
        {
            const PrefixBlockHandle &terminal_handle = snapshot.blocks.back();
            if (terminal_handle.layout.includes_terminal_hidden &&
                terminal_handle.layout.terminal_hidden_bytes > 0)
            {
                if (!terminal_handle.has_terminal_hidden ||
                    !terminal_handle.terminal_hidden)
                {
                    return fail("payload checkpoint terminal hidden payload is unavailable");
                }
                if (!ensureMTPTerminalHiddenBuffer())
                {
                    return fail("payload checkpoint terminal hidden buffer allocation failed");
                }
                void *hidden_dst = state_.prefix_terminal_hidden
                                       ? state_.prefix_terminal_hidden->raw_mutable_data()
                                       : nullptr;
                if (!hidden_dst)
                {
                    return fail("payload checkpoint terminal hidden destination is unavailable");
                }
                std::memcpy(
                    hidden_dst,
                    terminal_handle.terminal_hidden,
                    terminal_handle.layout.terminal_hidden_bytes);
                state_.prefix_terminal_hidden->mark_host_dirty();
                state_.mtp_terminal_hidden_current = true;
                if (state_.device_id.is_gpu())
                {
                    auto upload = TransferEngine::instance().uploadFull(
                        state_.prefix_terminal_hidden.get(),
                        state_.device_id,
                        stream);
                    if (!upload.success)
                    {
                        return fail("payload checkpoint terminal hidden upload failed: " +
                                    upload.error);
                    }
                }
                if (arena_ && arena_->isRegistered(BufferId::PREFIX_TERMINAL_HIDDEN))
                {
                    arena_->markWrittenFlagsOnly(
                        BufferId::PREFIX_TERMINAL_HIDDEN,
                        state_.device_id.is_gpu() ? state_.device_id : DeviceId::cpu());
                }
            }
        }

        for (const auto &handle : snapshot.mtp_blocks)
        {
            if (!handle.valid() || !handle.kvKData() || !handle.kvVData())
            {
                return fail("MTP KV block handle is invalid");
            }

            const int depth = handle.key.block_index;
            if (depth < 0 || depth >= static_cast<int>(state_.mtp_kv_caches.size()))
            {
                return fail("MTP KV block depth out of range: depth=" +
                            std::to_string(depth) +
                            " caches=" + std::to_string(state_.mtp_kv_caches.size()));
            }
            auto &cache = state_.mtp_kv_caches[static_cast<size_t>(depth)];
            if (!cache || handle.key.token_count < 0 || handle.key.token_count > cache->max_seq_len())
            {
                return fail("MTP KV block token count out of range: tokens=" +
                            std::to_string(handle.key.token_count));
            }

            PrefixPayloadLayout expected = buildDensePrefixPayloadLayout(
                *cache,
                state_.device_id,
                handle.key.token_count);
            if (!handle.layout.compatiblePayloadShape(expected))
            {
                return fail("MTP KV block layout is incompatible");
            }

            for (int local_layer = 0; local_layer < handle.layout.fa_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(*cache, local_layer);
                if (global_layer < 0)
                {
                    return fail("MTP KV block layer index is invalid");
                }
                const uint8_t *k_src = handle.kvKData() +
                                       static_cast<size_t>(local_layer) * handle.layout.bytes_per_fa_layer_k;
                const uint8_t *v_src = handle.kvVData() +
                                       static_cast<size_t>(local_layer) * handle.layout.bytes_per_fa_layer_v;

                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = handle.key.token_start;
                desc.token_count = handle.key.token_count;
                desc.stream = stream;
                if (!cache->importLogicalBlock(desc, k_src, v_src))
                {
                    return fail("MTP KV logical block import failed for layer=" +
                                std::to_string(global_layer) +
                                " tokens=" + std::to_string(handle.key.token_count));
                }
            }
        }

        state_.positions[seq_idx] = snapshot.cached_tokens;
        state_.sequence_lengths[seq_idx] = snapshot.cached_tokens;
        handleLivePrefixReplayStateAfterMutation(
            LivePrefixMutationReason::PrefixRestore,
            "restore_payload_checkpoint");
        return true;
    }

    bool DeviceGraphOrchestrator::truncateLivePrefixState(int cached_tokens, int seq_idx)
    {
        if (!state_.kv_cache || seq_idx < 0 || seq_idx >= state_.batch_size)
        {
            return false;
        }
        void *stream = explicitGPUStreamForOperation("truncateLivePrefixState");
        if (state_.device_id.is_gpu() && !stream)
        {
            return false;
        }
        if (!state_.kv_cache->truncateSequence(seq_idx, cached_tokens, stream))
        {
            return false;
        }
        for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
        {
            auto &cache = state_.mtp_kv_caches[depth];
            if (!cache)
            {
                continue;
            }
            const int shifted_count = std::max(
                0,
                cached_tokens - static_cast<int>(depth) - 1);
            const int bounded_count = std::min(shifted_count, cache->max_seq_len());
            if (!cache->truncateSequence(seq_idx, bounded_count, stream))
            {
                return false;
            }
        }
        state_.positions[seq_idx] = cached_tokens;
        state_.sequence_lengths[seq_idx] = cached_tokens;
        handleLivePrefixReplayStateAfterMutation(
            LivePrefixMutationReason::PrefixTruncate,
            "truncate_live_prefix");
        return true;
    }

    bool DeviceGraphOrchestrator::mtpSpecStatePublicationRequiresCapturedStage() const
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled || !state_.kv_cache)
        {
            return false;
        }

        auto *hybrid = dynamic_cast<IHybridKVCache *>(state_.kv_cache.get());
        if (!hybrid)
        {
            return false;
        }

        for (int layer = 0; layer < state_.kv_cache->n_layers(); ++layer)
        {
            if (hybrid->isGDNLayer(layer))
            {
                return true;
            }
        }
        return false;
    }

    bool DeviceGraphOrchestrator::requiresMTPDecodeEquivalentVerifierReplay() const
    {
        return mtpSpecStatePublicationRequiresCapturedStage() &&
               !supportsMTPSpecStatePublication();
    }

    int DeviceGraphOrchestrator::sampleGreedyOnDevice()
    {
        // LmHeadStage always writes the last-token logits to row 0 for both
        // prefill and decode.  In GlobalTP/NodeLocalTP, the terminal restore path
        // repopulates logits_local, so greedy sampling must use the shard-local
        // tensor and coordinate the winning candidate across ranks.
        if (graph_builder_ && graph_builder_->config().lm_head_column_parallel &&
            state_.logits_local)
        {
            const int token_offset = vocabOffsetForTPConfig(graph_builder_->config());
            void *stream = nullptr;
            auto device_opt = state_.logits_local->current_device();
            if (device_opt.has_value() && device_opt->is_gpu())
            {
                stream = consumePendingLogitsStream(
                    PendingLogitsStreamRole::MainDecode,
                    "sampleGreedyOnDeviceLocal");
                if (!stream)
                    stream = explicitGPUStreamForOperation("sampleGreedyOnDeviceLocal");
                if (!stream)
                {
                    return -1;
                }
            }
            return coordinateGreedyCandidate(
                sampleGreedyCandidateFromTensor(state_.logits_local.get(),
                                                0,
                                                token_offset,
                                                argmax_partial_vals_dev_,
                                                argmax_partial_idxs_dev_,
                                                argmax_partial_capacity_,
                                                stream,
                                                "logits_local"),
                globalTPContextForMTPCoordination());
        }

        if (!state_.logits)
            return -1;

        void *stream = nullptr;
        auto device_opt = state_.logits->current_device();
        if (device_opt.has_value() && device_opt->is_gpu())
        {
            stream = consumePendingLogitsStream(
                PendingLogitsStreamRole::MainDecode,
                "sampleGreedyOnDevice");
            if (!stream)
                stream = explicitGPUStreamForOperation("sampleGreedyOnDevice");
            if (!stream)
            {
                return -1;
            }
        }
        return coordinateGreedyCandidate(
            sampleGreedyCandidateFromTensor(state_.logits.get(),
                                            0,
                                            0,
                                            argmax_partial_vals_dev_,
                                            argmax_partial_idxs_dev_,
                                            argmax_partial_capacity_,
                                            stream,
                                            "logits"),
            globalTPContextForMTPCoordination());
    }

    int DeviceGraphOrchestrator::sampleOnDevice(const SamplingParams &params)
    {
        if (params.is_greedy())
        {
            return sampleGreedyOnDevice();
        }
        if (!state_.device_id.is_gpu() || !state_.logits)
        {
            return -1;
        }
        if (graph_builder_ && graph_builder_->config().lm_head_column_parallel)
        {
            return -1;
        }
        if (params.top_k <= 0 || params.top_k > 256)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] GPU stochastic sampling requires 1 <= top_k <= 256; got top_k="
                      << params.top_k);
            return -1;
        }
        if (!state_.logits->deviceValid())
        {
            return -1;
        }

        void *gpu_ptr = state_.logits->gpu_data_ptr();
        if (!gpu_ptr)
        {
            return -1;
        }

        const auto &shape = state_.logits->shape();
        if (shape.empty())
        {
            return -1;
        }
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (rows < 1 || cols == 0 || cols > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return -1;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            return -1;
        }
        void *stream = consumePendingLogitsStream(
            PendingLogitsStreamRole::MainDecode,
            "sampleOnDevice");
        if (!stream)
            stream = explicitGPUStreamForOperation("sampleOnDevice");
        if (!stream)
        {
            return -1;
        }

        int token = -1;
        const uint64_t seed =
            params.seed != 0
                ? static_cast<uint64_t>(params.seed)
                : (0xD1B54A32D192ED03ull ^
                   (session_epoch_ * 0x9E3779B97F4A7C15ull));
        const uint64_t offset = device_sampling_counter_++;
        const bool ok = backend->sampleTopKTopPF32(
            static_cast<const float *>(gpu_ptr),
            static_cast<int>(cols),
            params.top_k,
            params.top_p,
            params.temperature,
            seed,
            offset,
            state_.device_id.gpu_ordinal(),
            &token,
            stream);
        if (!ok)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] GPU stochastic sampling failed on "
                      << state_.device_id.toString());
            return -1;
        }
        return token;
    }

    /**
     * @brief Apply sparse penalties to the current main logits on the logits stream.
     *
     * GPU graph replay can leave logits pending on a capture stream. Penalty
     * application is an in-place mutation of those logits, so it is also a
     * logits producer. Keep the same stream in pending_main_decode_logits_stream_
     * for the next sampler/distribution consumer; otherwise a penalty kernel on
     * one stream can race a top-k kernel on another stream.
     */
    bool DeviceGraphOrchestrator::applyPenaltiesOnDevice(const std::vector<LogitPenalty> &penalties,
                                                         int vocab_size)
    {
        if (penalties.empty())
            return true; // Nothing to apply, success

        if (!state_.device_id.is_gpu())
            return false;
        void *stream = peekPendingLogitsStream(PendingLogitsStreamRole::MainDecode);
        if (!stream)
            stream = explicitGPUStreamForOperation("applyPenaltiesOnDevice");
        if (!stream)
            return false;

        bool ok = false;
        if (graph_builder_ && graph_builder_->config().lm_head_column_parallel &&
            state_.logits_local)
        {
            const int token_offset = vocabOffsetForTPConfig(graph_builder_->config());
            ok = applyPenaltiesToTensorRowOnDevice(
                state_.logits_local.get(),
                state_.device_id,
                penalties,
                vocab_size,
                0,
                token_offset,
                stream,
                "applyPenaltiesOnDeviceLocal");
        }
        else
        {
            ok = applyPenaltiesToTensorRowOnDevice(
                state_.logits.get(),
                state_.device_id,
                penalties,
                vocab_size,
                0,
                0,
                stream,
                "applyPenaltiesOnDevice");
        }

        if (ok)
        {
            /*
             * The sampler owns clearing this pending stream. We intentionally
             * leave it set even if it came from explicitGPUStreamForOperation()
             * because the penalty kernel is now the newest logits writer.
             */
            publishPendingLogitsStream(
                PendingLogitsStreamRole::MainDecode,
                stream,
                "applyPenaltiesOnDevice");
        }
        return ok;
    }

    /**
     * @brief Apply sparse penalties to MTP sidecar logits on the logits stream.
     *
     * Sidecar logits often come from a captured MTP graph. The penalty kernel
     * must stay ordered with that graph replay and with the following draft
     * sampler, so this method republishes the stream for the next MTP logits
     * consumer instead of switching to an unrelated operation stream.
     */
    bool DeviceGraphOrchestrator::applyPenaltiesToMTPLogitsOnDevice(
        const std::vector<LogitPenalty> &penalties,
        int vocab_size)
    {
        if (penalties.empty())
            return true;
        if (!state_.device_id.is_gpu())
            return false;

        auto it = state_.extension_buffers.find(BufferId::MTP_LOGITS);
        if (it == state_.extension_buffers.end() || !it->second)
            return false;

        void *stream = peekPendingLogitsStream(PendingLogitsStreamRole::MTPSidecar);
        if (!stream)
        {
            stream = explicitGPUStreamForOperation("applyPenaltiesToMTPLogitsOnDevice");
        }
        if (!stream)
            return false;

        const int token_offset = graph_builder_
                                     ? vocabOffsetForTPConfig(graph_builder_->config())
                                     : 0;
        const bool ok = applyPenaltiesToTensorRowOnDevice(
            it->second.get(),
            state_.device_id,
            penalties,
            vocab_size,
            0,
            token_offset,
            stream,
            "applyPenaltiesToMTPLogitsOnDevice");
        if (ok)
        {
            publishPendingLogitsStream(
                PendingLogitsStreamRole::MTPSidecar,
                stream,
                "applyPenaltiesToMTPLogitsOnDevice");
        }
        return ok;
    }

    /**
     * @brief Apply sparse penalties to one all-position verifier logits row.
     *
     * All-position verifier replay may build several target distributions from
     * one captured logits buffer. Penalties are row-local mutations of that
     * buffer, so they must be enqueued on the all-position producer stream and
     * leave that stream available for the next distribution build.
     */
    bool DeviceGraphOrchestrator::applyPenaltiesToAllPositionLogitsOnDeviceRow(
        int row,
        const std::vector<LogitPenalty> &penalties,
        int vocab_size)
    {
        if (penalties.empty())
            return true;
        if (!state_.device_id.is_gpu() || row < 0)
            return false;

        TensorBase *tensor = nullptr;
        int token_offset = 0;
        if (state_.all_position_logits)
        {
            tensor = state_.all_position_logits.get();
        }
        else if (state_.all_position_logits_local)
        {
            tensor = state_.all_position_logits_local.get();
            token_offset = graph_builder_
                               ? vocabOffsetForTPConfig(graph_builder_->config())
                               : 0;
        }
        if (!tensor)
            return false;

        void *stream = peekPendingLogitsStream(PendingLogitsStreamRole::AllPositionVerifier);
        if (!stream)
        {
            stream = explicitGPUStreamForOperation(
                "applyPenaltiesToAllPositionLogitsOnDeviceRow");
        }
        if (!stream)
            return false;

        const bool ok = applyPenaltiesToTensorRowOnDevice(
            tensor,
            state_.device_id,
            penalties,
            vocab_size,
            row,
            token_offset,
            stream,
            "applyPenaltiesToAllPositionLogitsOnDeviceRow");
        if (ok)
        {
            publishPendingLogitsStream(
                PendingLogitsStreamRole::AllPositionVerifier,
                stream,
                "applyPenaltiesToAllPositionLogitsOnDeviceRow");
        }
        return ok;
    }

    bool DeviceGraphOrchestrator::supportsDeviceStochasticMTPVerification() const
    {
        if (!state_.device_id.is_gpu())
            return false;
        if (graph_builder_ && graph_builder_->config().lm_head_column_parallel)
            return false;
        return stochastic_target_token_ids_dev_ &&
               stochastic_target_probs_dev_ &&
               stochastic_draft_token_ids_dev_ &&
               stochastic_draft_probs_dev_ &&
               stochastic_processed_logits_dev_ &&
               stochastic_inverse_rejection_samples_dev_ &&
               stochastic_target_sample_tokens_dev_ &&
               stochastic_draft_sample_tokens_dev_ &&
               stochastic_draft_sample_probs_dev_ &&
               argmax_partial_vals_dev_ &&
               argmax_partial_idxs_dev_ &&
               argmax_partial_capacity_ > 0 &&
               mtp_verifier_input_tokens_dev_ &&
               stochastic_verify_tokens_dev_ &&
               stochastic_verify_accepted_dev_ &&
               stochastic_verify_accept_probs_dev_ &&
               stochastic_verify_thresholds_dev_ &&
               stochastic_batch_output_tokens_dev_ &&
               stochastic_batch_output_meta_dev_;
    }

    bool DeviceGraphOrchestrator::buildStochasticProcessedLogitRowsOnDevice(
        DeviceLogitsSource source,
        int first_row,
        DeviceDistributionBuffer buffer,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size,
        void *stream,
        const char *operation_name)
    {
        if (!supportsDeviceStochasticMTPVerification() ||
            !stochastic_processed_logits_dev_ ||
            first_row < 0 || first_slot < 0 || row_count <= 0 ||
            vocab_size <= 0 || params.top_k <= 0 ||
            stream == nullptr)
        {
            return false;
        }

        const int max_slots =
            buffer == DeviceDistributionBuffer::Target
                ? static_cast<int>(kStochasticTargetRows)
                : static_cast<int>(kStochasticDraftRows);
        if (first_slot + row_count > max_slots ||
            row_count > static_cast<int>(kStochasticTargetRows))
        {
            return false;
        }

        TensorBase *tensor = nullptr;
        switch (source)
        {
        case DeviceLogitsSource::Main:
            tensor = state_.logits.get();
            break;
        case DeviceLogitsSource::MTP:
        {
            auto it = state_.extension_buffers.find(BufferId::MTP_LOGITS);
            tensor = it == state_.extension_buffers.end() ? nullptr : it->second.get();
            break;
        }
        case DeviceLogitsSource::AllPosition:
            tensor = state_.all_position_logits.get();
            break;
        }

        if (!tensor || !tensor->deviceValid() || !tensor->gpu_data_ptr())
            return false;

        const auto &shape = tensor->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            static_cast<size_t>(first_row + row_count) > rows ||
            static_cast<int>(cols) != vocab_size)
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const float *row_ptr =
            static_cast<const float *>(tensor->gpu_data_ptr()) +
            static_cast<size_t>(first_row) * cols;
        auto *processed_rows =
            static_cast<float *>(stochastic_processed_logits_dev_) +
            static_cast<size_t>(first_slot) * cols;

        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_processed_rows_build_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"operation", operation_name ? operation_name : "unknown"},
                 {"rows", std::to_string(row_count)},
                 {"top_k", std::to_string(params.top_k)}});
            if (!backend->enqueueBuildTopKTopPProcessedLogitsF32Device(
                    row_ptr,
                    row_count,
                    static_cast<int>(cols),
                    static_cast<int>(cols),
                    params.top_k,
                    params.top_p,
                    params.temperature,
                    state_.device_id.gpu_ordinal(),
                    stream,
                    processed_rows,
                    static_cast<int>(cols),
                    stochastic_topk_partial_vals_dev_,
                    stochastic_topk_partial_idxs_dev_,
                    stochastic_topk_partial_capacity_))
            {
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_processed_rows",
            static_cast<double>(row_count),
            "decode",
            state_.device_id.toString(),
            {{"buffer", buffer == DeviceDistributionBuffer::Target ? "target" : "draft"}});
        for (int row = 0; row < row_count; ++row)
        {
            const int slot = first_slot + row;
            if (buffer == DeviceDistributionBuffer::Target)
            {
                stochastic_target_top_k_[static_cast<size_t>(slot)] = params.top_k;
                stochastic_target_distribution_streams_[static_cast<size_t>(slot)] = stream;
            }
            else
            {
                stochastic_draft_top_k_[static_cast<size_t>(slot)] = params.top_k;
                stochastic_draft_distribution_streams_[static_cast<size_t>(slot)] = stream;
            }
        }
        return true;
    }

    bool DeviceGraphOrchestrator::buildStochasticProcessedLogitRowsOnDevice(
        DeviceLogitsSource source,
        int first_row,
        DeviceDistributionBuffer buffer,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size)
    {
        void *stream = nullptr;
        switch (source)
        {
        case DeviceLogitsSource::Main:
            stream = peekPendingLogitsStream(PendingLogitsStreamRole::MainDecode);
            break;
        case DeviceLogitsSource::MTP:
            stream = peekPendingLogitsStream(PendingLogitsStreamRole::MTPSidecar);
            break;
        case DeviceLogitsSource::AllPosition:
            stream = peekPendingLogitsStream(PendingLogitsStreamRole::AllPositionVerifier);
            break;
        }
        if (!stream)
        {
            stream = explicitGPUStreamForOperation("buildStochasticProcessedLogitRowsOnDevice");
        }
        if (!stream)
            return false;

        return buildStochasticProcessedLogitRowsOnDevice(
            source,
            first_row,
            buffer,
            first_slot,
            row_count,
            params,
            vocab_size,
            stream,
            "buildStochasticProcessedLogitRowsOnDevice");
    }

    bool DeviceGraphOrchestrator::buildStochasticDistributionOnDevice(
        DeviceLogitsSource source,
        int row,
        DeviceDistributionBuffer buffer,
        int slot,
        const SamplingParams &params,
        int vocab_size)
    {
        if (!supportsDeviceStochasticMTPVerification() ||
            row < 0 || slot < 0 || vocab_size <= 0 ||
            params.top_k <= 0 ||
            params.top_k > static_cast<int>(kStochasticDistributionMaxK))
        {
            return false;
        }

        TensorBase *tensor = nullptr;
        switch (source)
        {
        case DeviceLogitsSource::Main:
            tensor = state_.logits.get();
            break;
        case DeviceLogitsSource::MTP:
        {
            auto it = state_.extension_buffers.find(BufferId::MTP_LOGITS);
            tensor = it == state_.extension_buffers.end() ? nullptr : it->second.get();
            break;
        }
        case DeviceLogitsSource::AllPosition:
            tensor = state_.all_position_logits.get();
            break;
        }

        if (!tensor || !tensor->deviceValid())
            return false;

        const auto &shape = tensor->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            static_cast<size_t>(row) >= rows ||
            static_cast<int>(cols) != vocab_size)
        {
            return false;
        }

        int *token_ids = nullptr;
        float *probs = nullptr;
        int max_slots = 0;
        if (buffer == DeviceDistributionBuffer::Target)
        {
            token_ids = static_cast<int *>(stochastic_target_token_ids_dev_);
            probs = static_cast<float *>(stochastic_target_probs_dev_);
            max_slots = static_cast<int>(kStochasticTargetRows);
        }
        else
        {
            token_ids = static_cast<int *>(stochastic_draft_token_ids_dev_);
            probs = static_cast<float *>(stochastic_draft_probs_dev_);
            max_slots = static_cast<int>(kStochasticDraftRows);
        }
        if (!token_ids || !probs || slot >= max_slots)
            return false;

        if (buffer == DeviceDistributionBuffer::Target)
        {
            stochastic_target_top_k_[static_cast<size_t>(slot)] = 0;
            stochastic_target_distribution_streams_[static_cast<size_t>(slot)] = nullptr;
        }
        else
        {
            stochastic_draft_top_k_[static_cast<size_t>(slot)] = 0;
            stochastic_draft_distribution_streams_[static_cast<size_t>(slot)] = nullptr;
            clearStochasticDraftSampleReadySlot(slot);
        }

        void *stream = nullptr;
        bool used_pending_main_stream = false;
        bool used_pending_mtp_stream = false;
        if (source == DeviceLogitsSource::Main)
        {
            /*
             * MTP condition-forward replay may leave main logits pending on
             * the capture stream. Build the stochastic distribution on that
             * same stream so the CPU does not force a replay sync before the
             * sampler kernel has been queued.
             */
            stream = consumePendingLogitsStream(
                PendingLogitsStreamRole::MainDecode,
                "buildStochasticDistributionOnDevice(Main)");
            used_pending_main_stream = stream != nullptr;
        }
        else if (source == DeviceLogitsSource::MTP)
        {
            stream = consumePendingLogitsStream(
                PendingLogitsStreamRole::MTPSidecar,
                "buildStochasticDistributionOnDevice(MTP)");
            used_pending_mtp_stream = stream != nullptr;
        }
        else if (source == DeviceLogitsSource::AllPosition)
        {
            /*
             * A deferred all-position verifier replay leaves its capture stream
             * here for the next device-side consumer. Stochastic verification
             * may build several target/bonus distributions before the summary
             * kernel finally copies compact metadata to host, so the stream is
             * intentionally not cleared by individual distribution builds.
             */
            stream = peekPendingLogitsStream(PendingLogitsStreamRole::AllPositionVerifier);
        }
        if (!stream)
            stream = explicitGPUStreamForOperation("buildStochasticDistributionOnDevice");
        if (!stream)
            return false;

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        const void *gpu_ptr = tensor->gpu_data_ptr();
        if (!gpu_ptr)
            return false;
        const float *row_ptr =
            static_cast<const float *>(gpu_ptr) +
            static_cast<size_t>(row) * cols;

        const size_t slot_offset =
            static_cast<size_t>(slot) * kStochasticDistributionMaxK;
        bool ok = false;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_distribution_build_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"slot", std::to_string(slot)},
                 {"source", source == DeviceLogitsSource::Main
                                ? "main"
                                : (source == DeviceLogitsSource::MTP
                                       ? "mtp"
                                       : "all_position")},
                 {"top_k", std::to_string(params.top_k)}});
            ok = backend->enqueueBuildTopKTopPDistributionF32Device(
                row_ptr,
                static_cast<int>(cols),
                params.top_k,
                params.top_p,
                params.temperature,
                state_.device_id.gpu_ordinal(),
                stream,
                token_ids + slot_offset,
                probs + slot_offset,
                stochastic_topk_partial_vals_dev_,
                stochastic_topk_partial_idxs_dev_,
                stochastic_topk_partial_capacity_);
        }
        if (ok)
        {
            if (used_pending_main_stream)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_main_distribution_stream_handoffs",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"slot", std::to_string(slot)}});
            }
            if (used_pending_mtp_stream)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_mtp_distribution_stream_handoffs",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"slot", std::to_string(slot)}});
            }
            const int effective_top_k =
                std::min(params.top_k, static_cast<int>(cols));
            size_t partial_blocks =
                (cols + kStochasticTopKSmallKThreads - 1) /
                kStochasticTopKSmallKThreads;
            partial_blocks = std::max<size_t>(partial_blocks, 1);
            partial_blocks = std::min(partial_blocks, kStochasticTopKPartialBlocks);
            const bool used_gpu_smallk_scratch =
                state_.device_id.is_gpu() &&
                effective_top_k > 0 &&
                effective_top_k <= static_cast<int>(kStochasticTopKSmallKCap) &&
                stochastic_topk_partial_vals_dev_ &&
                stochastic_topk_partial_idxs_dev_ &&
                stochastic_topk_partial_capacity_ >=
                    static_cast<int>(partial_blocks * static_cast<size_t>(effective_top_k));
            if (used_gpu_smallk_scratch)
            {
                const char *source_name = "unknown";
                switch (source)
                {
                case DeviceLogitsSource::Main:
                    source_name = "main";
                    break;
                case DeviceLogitsSource::MTP:
                    source_name = "mtp";
                    break;
                case DeviceLogitsSource::AllPosition:
                    source_name = "all_position";
                    break;
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "stochastic_topk_smallk_scratch_distribution_builds",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {
                        {"source", source_name},
                        {"top_k", std::to_string(effective_top_k)},
                    });
            }
            if (buffer == DeviceDistributionBuffer::Target)
            {
                stochastic_target_top_k_[static_cast<size_t>(slot)] = params.top_k;
                stochastic_target_distribution_streams_[static_cast<size_t>(slot)] = stream;
                /*
                 * Target distribution slots are reused for all-position
                 * verifier rows after the first target token has already been
                 * sampled.  Do not clear target sample readiness here: that
                 * readiness belongs to STOCHASTIC_TARGET_SAMPLE_TOKENS and is
                 * the event edge that lets the verifier summary read the
                 * deferred first token safely from another stream.
                 */
            }
            else
            {
                stochastic_draft_top_k_[static_cast<size_t>(slot)] = params.top_k;
                stochastic_draft_distribution_streams_[static_cast<size_t>(slot)] = stream;
                clearStochasticDraftSampleReadySlot(slot);
            }
        }
        return ok;
    }

    bool DeviceGraphOrchestrator::buildStochasticDistributionsOnDevice(
        DeviceLogitsSource source,
        int first_row,
        DeviceDistributionBuffer buffer,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size)
    {
        if (!supportsDeviceStochasticMTPVerification() ||
            source != DeviceLogitsSource::AllPosition ||
            buffer != DeviceDistributionBuffer::Target ||
            first_row < 0 ||
            first_slot < 0 ||
            row_count <= 0 ||
            params.top_k <= 0 ||
            params.top_k > static_cast<int>(kStochasticDistributionMaxK))
        {
            return false;
        }

        if (first_slot + row_count > static_cast<int>(kStochasticTargetRows))
            return false;

        auto *token_ids = static_cast<int *>(stochastic_target_token_ids_dev_);
        auto *probs = static_cast<float *>(stochastic_target_probs_dev_);
        if (!token_ids || !probs)
            return false;

        for (int row = 0; row < row_count; ++row)
        {
            const int slot = first_slot + row;
            stochastic_target_top_k_[static_cast<size_t>(slot)] = 0;
            stochastic_target_distribution_streams_[static_cast<size_t>(slot)] = nullptr;
        }

        /*
         * Batched all-position verifier distributions must stay ordered behind
         * the launch-only verifier replay. Individual rows do not consume the
         * pending stream; the final compact summary copy closes this handoff.
         */
        void *stream = peekPendingLogitsStream(PendingLogitsStreamRole::AllPositionVerifier);
        if (!stream)
            stream = explicitGPUStreamForOperation("buildStochasticDistributionsOnDevice");
        if (!stream)
            return false;

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        TensorBase *tensor = state_.all_position_logits.get();
        if (!tensor || !tensor->gpu_data_ptr())
            return false;
        const auto &shape = tensor->shape();
        if (shape.size() < 2)
            return false;
        const size_t rows = shape[0];
        const size_t cols = shape[1];
        if (first_row + row_count > static_cast<int>(rows) ||
            (vocab_size > 0 && static_cast<size_t>(vocab_size) != cols))
        {
            return false;
        }

        const float *row_ptr =
            static_cast<const float *>(tensor->gpu_data_ptr()) +
            static_cast<size_t>(first_row) * cols;
        const size_t slot_offset =
            static_cast<size_t>(first_slot) * kStochasticDistributionMaxK;

        bool ok = false;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_distribution_batch_build_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"first_slot", std::to_string(first_slot)},
                 {"rows", std::to_string(row_count)},
                 {"source", "all_position"},
                 {"top_k", std::to_string(params.top_k)}});
            ok = backend->enqueueBuildTopKTopPDistributionsF32Device(
                row_ptr,
                row_count,
                static_cast<int>(cols),
                static_cast<int>(cols),
                params.top_k,
                params.top_p,
                params.temperature,
                state_.device_id.gpu_ordinal(),
                stream,
                token_ids + slot_offset,
                static_cast<int>(kStochasticDistributionMaxK),
                probs + slot_offset,
                stochastic_topk_partial_vals_dev_,
                stochastic_topk_partial_idxs_dev_,
                stochastic_topk_partial_capacity_);
        }

        if (!ok)
            return false;

        for (int row = 0; row < row_count; ++row)
        {
            const int slot = first_slot + row;
            stochastic_target_top_k_[static_cast<size_t>(slot)] = params.top_k;
            stochastic_target_distribution_streams_[static_cast<size_t>(slot)] = stream;
        }

        const int effective_top_k =
            std::min(params.top_k, static_cast<int>(cols));
        size_t partial_blocks =
            (cols + kStochasticTopKSmallKThreads - 1) /
            kStochasticTopKSmallKThreads;
        partial_blocks = std::max<size_t>(partial_blocks, 1);
        partial_blocks = std::min(partial_blocks, kStochasticTopKPartialBlocks);
        if (effective_top_k > 0 &&
            effective_top_k <= static_cast<int>(kStochasticTopKSmallKCap))
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_topk_smallk_scratch_distribution_builds",
                static_cast<double>(row_count),
                "decode",
                state_.device_id.toString(),
                {
                    {"source", "all_position_batched"},
                    {"top_k", std::to_string(effective_top_k)},
                    {"partial_blocks", std::to_string(partial_blocks)},
                });
        }

        return true;
    }

    bool DeviceGraphOrchestrator::sampleStochasticDistributionOnDeviceImpl(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold,
        int32_t *out_token_host)
    {
        if (!supportsDeviceStochasticMTPVerification() || slot < 0)
            return false;

        int *token_ids = nullptr;
        float *probs = nullptr;
        int max_slots = 0;
        if (buffer == DeviceDistributionBuffer::Target)
        {
            token_ids = static_cast<int *>(stochastic_target_token_ids_dev_);
            probs = static_cast<float *>(stochastic_target_probs_dev_);
            max_slots = static_cast<int>(kStochasticTargetRows);
        }
        else
        {
            token_ids = static_cast<int *>(stochastic_draft_token_ids_dev_);
            probs = static_cast<float *>(stochastic_draft_probs_dev_);
            max_slots = static_cast<int>(kStochasticDraftRows);
        }
        if (!token_ids || !probs || slot >= max_slots)
            return false;
        const int active_top_k =
            buffer == DeviceDistributionBuffer::Target
                ? stochastic_target_top_k_[static_cast<size_t>(slot)]
                : stochastic_draft_top_k_[static_cast<size_t>(slot)];
        if (active_top_k <= 0 ||
            active_top_k > static_cast<int>(kStochasticDistributionMaxK))
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;
        void *stream = nullptr;
        if (buffer == DeviceDistributionBuffer::Target)
        {
            stream = stochastic_target_distribution_streams_[static_cast<size_t>(slot)];
            stochastic_target_distribution_streams_[static_cast<size_t>(slot)] = nullptr;
        }
        else
        {
            stream = stochastic_draft_distribution_streams_[static_cast<size_t>(slot)];
            stochastic_draft_distribution_streams_[static_cast<size_t>(slot)] = nullptr;
        }
        if (!stream)
        {
            // The build either ran on the context stream or completed through a
            // synchronized fallback. We still require a real backend stream; the
            // device-default/null stream is never acceptable here.
            stream = explicitGPUStreamForOperation("sampleStochasticDistributionOnDevice");
        }
        if (!stream)
            return false;

        const char *buffer_name =
            buffer == DeviceDistributionBuffer::Draft ? "draft" : "target";

        /*
         * Sampled target and draft tokens both outlive the sampler kernel.
         * Keep them out of STOCHASTIC_VERIFY_TOKENS because verifier rows reuse
         * that buffer for row-local accept/reject outputs later in the same
         * speculative step.
         */
        int *out_token_dev =
            (buffer == DeviceDistributionBuffer::Draft)
                ? static_cast<int *>(stochastic_draft_sample_tokens_dev_) + slot
                : static_cast<int *>(stochastic_target_sample_tokens_dev_) + slot;
        if (!out_token_dev)
            return false;

        /*
         * The selected draft probability is small but valuable metadata for the
         * vLLM-style fused verifier: the verifier can read q(draft_token)
         * directly instead of rebuilding or rescanning the draft distribution.
         * Target samples do not currently need the same metadata, so only draft
         * slots write this companion buffer.
         */
        float *out_probability_dev =
            (buffer == DeviceDistributionBuffer::Draft && stochastic_draft_sample_probs_dev_)
                ? static_cast<float *>(stochastic_draft_sample_probs_dev_) + slot
                : nullptr;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sample_stochastic_distribution_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"buffer", buffer_name},
                 {"slot", std::to_string(slot)},
                 {"top_k", std::to_string(active_top_k)}});
            if (!backend->enqueueSampleDistributionF32Device(
                    token_ids + static_cast<size_t>(slot) * kStochasticDistributionMaxK,
                    probs + static_cast<size_t>(slot) * kStochasticDistributionMaxK,
                    active_top_k,
                    threshold,
                    state_.device_id.gpu_ordinal(),
                    stream,
                    out_token_dev,
                    out_probability_dev))
            {
                return false;
            }
        }

        if (buffer == DeviceDistributionBuffer::Draft)
        {
            if (!recordStochasticDraftSampleReady(slot, stream))
                return false;
        }
        else
        {
            if (!recordStochasticTargetSampleReady(slot, stream))
                return false;
        }

        if (!out_token_host)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "sample_stochastic_distribution_deferred_host_reads",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"buffer", buffer_name},
                 {"slot", std::to_string(slot)},
                 {"top_k", std::to_string(active_top_k)}});
            return true;
        }

        int32_t token = -1;
        /*
         * This read is the semantic point where the runner needs the sampled
         * draft token on the host. It drains all earlier distribution work on
         * `stream`, but the source is trusted arena scratch, so use the fast
         * D2H path and avoid per-token ROCm pointer-attribute validation.
         */
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sample_stochastic_distribution_d2h_sync",
                "decode",
                state_.device_id.toString(),
                {{"buffer", buffer_name},
                 {"slot", std::to_string(slot)},
                 {"top_k", std::to_string(active_top_k)}});
            if (!backend->deviceToHostFast(&token, out_token_dev, sizeof(int),
                                           state_.device_id.gpu_ordinal(), stream))
            {
                return false;
            }
        }
        *out_token_host = token;
        return true;
    }

    bool DeviceGraphOrchestrator::sampleStochasticDraftProposalOnDeviceImpl(
        DeviceLogitsSource source,
        int row,
        int slot,
        const SamplingParams &params,
        int vocab_size,
        float threshold,
        int32_t *out_token_host)
    {
        (void)params;
        (void)threshold;

        if (!supportsDeviceStochasticMTPVerification() ||
            source != DeviceLogitsSource::MTP ||
            row < 0 || slot < 0 ||
            slot >= static_cast<int>(kStochasticDraftRows) ||
            vocab_size <= 0 ||
            !stochastic_draft_sample_tokens_dev_ ||
            !stochastic_draft_sample_probs_dev_ ||
            !argmax_partial_vals_dev_ ||
            !argmax_partial_idxs_dev_ ||
            argmax_partial_capacity_ <= 0)
        {
            return false;
        }

        auto it = state_.extension_buffers.find(BufferId::MTP_LOGITS);
        TensorBase *tensor =
            it == state_.extension_buffers.end() ? nullptr : it->second.get();
        if (!tensor || !tensor->deviceValid() || !tensor->gpu_data_ptr())
            return false;

        const auto &shape = tensor->shape();
        if (shape.empty())
            return false;
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (cols == 0 ||
            cols > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            static_cast<size_t>(row) >= rows ||
            static_cast<int>(cols) != vocab_size)
        {
            return false;
        }

        void *stream = consumePendingLogitsStream(
            PendingLogitsStreamRole::MTPSidecar,
            "sampleStochasticDraftProposalOnDevice(MTP)");
        if (!stream)
            stream = explicitGPUStreamForOperation("sampleStochasticDraftProposalOnDevice");
        if (!stream)
            return false;

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;

        stochastic_draft_top_k_[static_cast<size_t>(slot)] = 0;
        stochastic_draft_distribution_streams_[static_cast<size_t>(slot)] = nullptr;
        clearStochasticDraftSampleReadySlot(slot);

        const float *row_ptr =
            static_cast<const float *>(tensor->gpu_data_ptr()) +
            static_cast<size_t>(row) * cols;
        auto *out_token_dev =
            static_cast<int *>(stochastic_draft_sample_tokens_dev_) + slot;
        auto *out_value_dev =
            static_cast<float *>(stochastic_draft_sample_probs_dev_) + slot;

        /*
         * vLLM's default MTP proposal is greedy. The verifier receives the
         * sampled token and treats the draft distribution q as one-hot at that
         * token, avoiding full draft probability/logit rows in runner scratch.
         */
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_draft_greedy_proposal_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"slot", std::to_string(slot)},
                 {"vocab", std::to_string(vocab_size)}});
            if (!backend->enqueueArgmaxF32BatchedRowsDevice(
                    row_ptr,
                    /*rows=*/1,
                    static_cast<int>(cols),
                    state_.device_id.gpu_ordinal(),
                    stream,
                    out_value_dev,
                    out_token_dev,
                    argmax_partial_vals_dev_,
                    argmax_partial_idxs_dev_,
                    argmax_partial_capacity_))
            {
                return false;
            }
        }

        if (!recordStochasticDraftSampleReady(slot, stream))
            return false;

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_draft_greedy_proposals",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"slot", std::to_string(slot)}});

        if (!out_token_host)
            return true;

        int32_t token = -1;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_draft_greedy_proposal_d2h_sync",
                "decode",
                state_.device_id.toString(),
                {{"slot", std::to_string(slot)}});
            if (!backend->deviceToHostFast(&token,
                                           out_token_dev,
                                           sizeof(int),
                                           state_.device_id.gpu_ordinal(),
                                           stream))
            {
                return false;
            }
        }
        *out_token_host = token;
        return token >= 0;
    }

    int DeviceGraphOrchestrator::sampleStochasticDraftProposalOnDevice(
        DeviceLogitsSource source,
        int row,
        int slot,
        const SamplingParams &params,
        int vocab_size,
        float threshold)
    {
        int32_t token = -1;
        if (!sampleStochasticDraftProposalOnDeviceImpl(
                source,
                row,
                slot,
                params,
                vocab_size,
                threshold,
                &token))
        {
            return -1;
        }
        return token;
    }

    bool DeviceGraphOrchestrator::sampleStochasticDraftProposalOnDeviceDeferred(
        DeviceLogitsSource source,
        int row,
        int slot,
        const SamplingParams &params,
        int vocab_size,
        float threshold)
    {
        return sampleStochasticDraftProposalOnDeviceImpl(
            source,
            row,
            slot,
            params,
            vocab_size,
            threshold,
            /*out_token_host=*/nullptr);
    }

    int DeviceGraphOrchestrator::sampleStochasticDistributionOnDevice(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold)
    {
        int32_t token = -1;
        if (!sampleStochasticDistributionOnDeviceImpl(
                buffer,
                slot,
                threshold,
                &token))
        {
            return -1;
        }
        return token;
    }

    bool DeviceGraphOrchestrator::sampleStochasticDistributionOnDeviceDeferred(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold)
    {
        return sampleStochasticDistributionOnDeviceImpl(
            buffer,
            slot,
            threshold,
            /*out_token_host=*/nullptr);
    }

    bool DeviceGraphOrchestrator::verifyStochasticDistributionsOnDevice(
        int target_slot,
        int draft_slot,
        int draft_token,
        float accept_threshold,
        float residual_threshold,
        DeviceSpeculativeVerifyResult *out)
    {
        if (!supportsDeviceStochasticMTPVerification() || !out ||
            target_slot < 0 || draft_slot < 0 ||
            target_slot >= static_cast<int>(kStochasticTargetRows) ||
            draft_slot >= static_cast<int>(kStochasticDraftRows))
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;
        void *stream = explicitGPUStreamForOperation("verifyStochasticDistributionsOnDevice");
        if (!stream)
            return false;

        int *target_ids = static_cast<int *>(stochastic_target_token_ids_dev_) +
                          static_cast<size_t>(target_slot) * kStochasticDistributionMaxK;
        float *target_probs = static_cast<float *>(stochastic_target_probs_dev_) +
                              static_cast<size_t>(target_slot) * kStochasticDistributionMaxK;
        int *draft_ids = static_cast<int *>(stochastic_draft_token_ids_dev_) +
                         static_cast<size_t>(draft_slot) * kStochasticDistributionMaxK;
        float *draft_probs = static_cast<float *>(stochastic_draft_probs_dev_) +
                             static_cast<size_t>(draft_slot) * kStochasticDistributionMaxK;
        const int target_top_k = stochastic_target_top_k_[static_cast<size_t>(target_slot)];
        const int draft_top_k = stochastic_draft_top_k_[static_cast<size_t>(draft_slot)];
        if (target_top_k <= 0 ||
            draft_top_k <= 0 ||
            target_top_k != draft_top_k ||
            target_top_k > static_cast<int>(kStochasticDistributionMaxK))
        {
            return false;
        }

        int *out_token_dev = static_cast<int *>(stochastic_verify_tokens_dev_) + target_slot;
        int *out_accepted_dev = static_cast<int *>(stochastic_verify_accepted_dev_) + target_slot;
        float *out_accept_prob_dev =
            static_cast<float *>(stochastic_verify_accept_probs_dev_) + target_slot;
        float *out_threshold_dev =
            static_cast<float *>(stochastic_verify_thresholds_dev_) + target_slot;

        if (!backend->enqueueSpeculativeVerifyDistributionsF32DeviceThresholds(
                target_ids,
                target_probs,
                draft_ids,
                draft_probs,
                target_top_k,
                draft_token,
                accept_threshold,
                residual_threshold,
                state_.device_id.gpu_ordinal(),
                stream,
                out_token_dev,
                out_accepted_dev,
                out_accept_prob_dev,
                out_threshold_dev))
        {
            return false;
        }

        int token = -1;
        int accepted = 0;
        float accept_probability = 0.0f;
        float threshold = 0.0f;
        /*
         * These compact result buffers are written by the verifier kernel on
         * the same stream. Fast D2H preserves the synchronization point while
         * skipping backend pointer validation that dominates tiny ROCm copies.
         */
        if (!backend->deviceToHostFast(&token, out_token_dev, sizeof(int),
                                       state_.device_id.gpu_ordinal(), stream) ||
            !backend->deviceToHostFast(&accepted, out_accepted_dev, sizeof(int),
                                       state_.device_id.gpu_ordinal(), stream) ||
            !backend->deviceToHostFast(&accept_probability, out_accept_prob_dev, sizeof(float),
                                       state_.device_id.gpu_ordinal(), stream) ||
            !backend->deviceToHostFast(&threshold, out_threshold_dev, sizeof(float),
                                       state_.device_id.gpu_ordinal(), stream))
        {
            return false;
        }

        out->token = token;
        out->accepted = accepted != 0;
        out->accept_probability = accept_probability;
        out->accept_threshold = threshold;
        return token >= 0;
    }

    bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOnDevice(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        DeviceSpeculativeVerifyResult *out)
    {
        if (!supportsDeviceStochasticMTPVerification() ||
            first_target_slot < 0 || first_draft_slot < 0 ||
            row_count <= 0 ||
            first_target_slot + row_count > static_cast<int>(kStochasticTargetRows) ||
            first_draft_slot + row_count > static_cast<int>(kStochasticDraftRows) ||
            !draft_tokens || !accept_thresholds || !residual_thresholds || !out)
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;
        void *stream = explicitGPUStreamForOperation("verifyStochasticDistributionsBatchOnDevice");
        if (!stream)
            return false;

        const int target_top_k =
            stochastic_target_top_k_[static_cast<size_t>(first_target_slot)];
        if (target_top_k <= 0 ||
            target_top_k > static_cast<int>(kStochasticDistributionMaxK))
        {
            return false;
        }
        for (int row = 0; row < row_count; ++row)
        {
            const int target_slot = first_target_slot + row;
            const int draft_slot = first_draft_slot + row;
            const int draft_top_k =
                stochastic_draft_top_k_[static_cast<size_t>(draft_slot)];
            if (stochastic_target_top_k_[static_cast<size_t>(target_slot)] != target_top_k ||
                draft_top_k != target_top_k)
            {
                return false;
            }
        }

        auto *target_ids =
            static_cast<int *>(stochastic_target_token_ids_dev_) +
            static_cast<size_t>(first_target_slot) * kStochasticDistributionMaxK;
        auto *target_probs =
            static_cast<float *>(stochastic_target_probs_dev_) +
            static_cast<size_t>(first_target_slot) * kStochasticDistributionMaxK;
        auto *draft_ids =
            static_cast<int *>(stochastic_draft_token_ids_dev_) +
            static_cast<size_t>(first_draft_slot) * kStochasticDistributionMaxK;
        auto *draft_probs =
            static_cast<float *>(stochastic_draft_probs_dev_) +
            static_cast<size_t>(first_draft_slot) * kStochasticDistributionMaxK;
        auto *sampled_draft_tokens =
            stochastic_draft_sample_tokens_dev_
                ? static_cast<int *>(stochastic_draft_sample_tokens_dev_) +
                      first_draft_slot
                : nullptr;
        auto *out_token_dev =
            static_cast<int *>(stochastic_verify_tokens_dev_) + first_target_slot;
        auto *out_accepted_dev =
            static_cast<int *>(stochastic_verify_accepted_dev_) + first_target_slot;
        auto *out_accept_prob_dev =
            static_cast<float *>(stochastic_verify_accept_probs_dev_) + first_target_slot;
        auto *out_threshold_dev =
            static_cast<float *>(stochastic_verify_thresholds_dev_) + first_target_slot;

        if (!backend->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(
                target_ids,
                target_probs,
                draft_ids,
                draft_probs,
                target_top_k,
                static_cast<int>(kStochasticDistributionMaxK),
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                state_.device_id.gpu_ordinal(),
                stream,
                out_token_dev,
                out_accepted_dev,
                out_accept_prob_dev,
                out_threshold_dev))
        {
            return false;
        }

        std::array<int, 4> tokens{};
        std::array<int, 4> accepted{};
        std::array<float, 4> accept_probabilities{};
        std::array<float, 4> thresholds{};
        if (!backend->deviceToHostFast(tokens.data(), out_token_dev,
                                       sizeof(int) * static_cast<size_t>(row_count),
                                       state_.device_id.gpu_ordinal(), stream) ||
            !backend->deviceToHostFast(accepted.data(), out_accepted_dev,
                                       sizeof(int) * static_cast<size_t>(row_count),
                                       state_.device_id.gpu_ordinal(), stream) ||
            !backend->deviceToHostFast(accept_probabilities.data(), out_accept_prob_dev,
                                       sizeof(float) * static_cast<size_t>(row_count),
                                       state_.device_id.gpu_ordinal(), stream) ||
            !backend->deviceToHostFast(thresholds.data(), out_threshold_dev,
                                       sizeof(float) * static_cast<size_t>(row_count),
                                       state_.device_id.gpu_ordinal(), stream))
        {
            return false;
        }

        for (int row = 0; row < row_count; ++row)
        {
            out[row].accepted = accepted[static_cast<size_t>(row)] != 0;
            out[row].token = tokens[static_cast<size_t>(row)];
            out[row].accept_probability =
                accept_probabilities[static_cast<size_t>(row)];
            out[row].accept_threshold = thresholds[static_cast<size_t>(row)];
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_verify_batch_rows",
            static_cast<double>(row_count),
            "decode",
            state_.device_id.toString(),
            {{"top_k", std::to_string(target_top_k)}});
        return true;
    }

    bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDevice(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        int32_t first_token,
        const int32_t *stop_tokens,
        int stop_token_count,
        int bonus_target_slot,
        float bonus_threshold,
        DeviceSpeculativeVerifyBatchOutcome *out,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        bool use_vllm_probability_rejection)
    {
        return verifyStochasticDistributionsBatchOutcomeOnDeviceCommon(
            first_target_slot,
            first_draft_slot,
            draft_tokens,
            accept_thresholds,
            residual_thresholds,
            row_count,
            first_token,
            /*first_target_sample_slot=*/-1,
            /*first_token_from_device=*/false,
            stop_tokens,
            stop_token_count,
            bonus_target_slot,
            bonus_threshold,
            out,
            inverse_sample_seed,
            inverse_sample_first_logical_position,
            use_vllm_probability_rejection);
    }

    bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        int first_target_sample_slot,
        const int32_t *stop_tokens,
        int stop_token_count,
        int bonus_target_slot,
        float bonus_threshold,
        DeviceSpeculativeVerifyBatchOutcome *out,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        bool use_vllm_probability_rejection)
    {
        return verifyStochasticDistributionsBatchOutcomeOnDeviceCommon(
            first_target_slot,
            first_draft_slot,
            draft_tokens,
            accept_thresholds,
            residual_thresholds,
            row_count,
            /*first_token=*/-1,
            first_target_sample_slot,
            /*first_token_from_device=*/true,
            stop_tokens,
            stop_token_count,
            bonus_target_slot,
            bonus_threshold,
            out,
            inverse_sample_seed,
            inverse_sample_first_logical_position,
            use_vllm_probability_rejection);
    }

    bool DeviceGraphOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceCommon(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        int32_t first_token,
        int first_target_sample_slot,
        bool first_token_from_device,
        const int32_t *stop_tokens,
        int stop_token_count,
        int bonus_target_slot,
        float bonus_threshold,
        DeviceSpeculativeVerifyBatchOutcome *out,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        bool use_vllm_probability_rejection)
    {
        using namespace sampling_math;
        (void)draft_tokens;
        if (!out)
            return false;
        *out = DeviceSpeculativeVerifyBatchOutcome{};

        const bool has_bonus = bonus_target_slot >= 0;
        if (!supportsDeviceStochasticMTPVerification() ||
            first_target_slot < 0 || first_draft_slot < 0 ||
            row_count <= 0 ||
            row_count > kSpeculativeBatchMaxRows ||
            first_target_slot + row_count > static_cast<int>(kStochasticTargetRows) ||
            first_draft_slot + row_count > static_cast<int>(kStochasticDraftRows) ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens) ||
            (first_token_from_device &&
             (first_target_sample_slot < 0 ||
              first_target_sample_slot >= kSpeculativeBatchMaxOutputTokens ||
              !stochastic_target_sample_tokens_dev_)) ||
            (has_bonus &&
             bonus_target_slot >= static_cast<int>(kStochasticTargetRows)) ||
            !accept_thresholds || !residual_thresholds)
        {
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
            return false;
        /*
         * If the verifier replay was launch-only deferred, all target
         * distribution builds have already been enqueued on this stream. The
         * batch reducer and compact D2H copy consume the same stream and close
         * the handoff. Falling back to a fresh explicit stream would race the
         * verifier rows, so consume-and-clear the pending stream here.
         */
        void *stream = consumePendingLogitsStream(
            PendingLogitsStreamRole::AllPositionVerifier,
            "verifyStochasticDistributionsBatchOutcomeOnDevice");
        if (!stream)
        {
            stream = explicitGPUStreamForOperation(
                "verifyStochasticDistributionsBatchOutcomeOnDevice");
        }
        if (!stream)
            return false;

        const int target_top_k =
            stochastic_target_top_k_[static_cast<size_t>(first_target_slot)];
        if (target_top_k <= 0 ||
            target_top_k > static_cast<int>(kStochasticDistributionMaxK))
        {
            return false;
        }

        for (int row = 0; row < row_count; ++row)
        {
            const int target_slot = first_target_slot + row;
            const int draft_slot = first_draft_slot + row;
            const int draft_top_k =
                stochastic_draft_top_k_[static_cast<size_t>(draft_slot)];
            if (stochastic_target_top_k_[static_cast<size_t>(target_slot)] != target_top_k ||
                (!use_vllm_probability_rejection &&
                 draft_top_k != target_top_k))
            {
                return false;
            }
        }
        if (has_bonus &&
            stochastic_target_top_k_[static_cast<size_t>(bonus_target_slot)] != target_top_k)
        {
            return false;
        }

        auto *target_ids =
            static_cast<int *>(stochastic_target_token_ids_dev_) +
            static_cast<size_t>(first_target_slot) * kStochasticDistributionMaxK;
        auto *target_probs =
            static_cast<float *>(stochastic_target_probs_dev_) +
            static_cast<size_t>(first_target_slot) * kStochasticDistributionMaxK;
        auto *draft_ids =
            static_cast<int *>(stochastic_draft_token_ids_dev_) +
            static_cast<size_t>(first_draft_slot) * kStochasticDistributionMaxK;
        auto *draft_probs =
            static_cast<float *>(stochastic_draft_probs_dev_) +
            static_cast<size_t>(first_draft_slot) * kStochasticDistributionMaxK;
        auto *sampled_draft_tokens =
            static_cast<int *>(stochastic_draft_sample_tokens_dev_) + first_draft_slot;
        auto *sampled_draft_probs =
            (!use_vllm_probability_rejection && stochastic_draft_sample_probs_dev_)
                ? static_cast<float *>(stochastic_draft_sample_probs_dev_) + first_draft_slot
                : nullptr;
        const int full_vocab_size =
            graph_builder_ ? graph_builder_->config().vocab_size : state_.vocab_size;
        auto *target_processed_logits =
            static_cast<float *>(stochastic_processed_logits_dev_);
        auto *out_token_dev =
            static_cast<int *>(stochastic_verify_tokens_dev_) + first_target_slot;
        auto *out_accepted_dev =
            static_cast<int *>(stochastic_verify_accepted_dev_) + first_target_slot;
        auto *out_accept_prob_dev =
            static_cast<float *>(stochastic_verify_accept_probs_dev_) + first_target_slot;
        auto *out_threshold_dev =
            static_cast<float *>(stochastic_verify_thresholds_dev_) + first_target_slot;

        if (!waitForRequiredStochasticDraftSampleReadyRange(
                first_draft_slot,
                row_count,
                stream,
                "stochastic_batch_verifier"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to order stochastic batch verifier after deferred draft samples");
            return false;
        }

        if (debugEnv().validation.validate_buffers)
        {
            std::array<int32_t, kSpeculativeBatchMaxRows> sampled_drafts{};
            if (!backend->deviceToHostFast(
                    sampled_drafts.data(),
                    sampled_draft_tokens,
                    sizeof(int32_t) * static_cast<size_t>(row_count),
                    state_.device_id.gpu_ordinal(),
                    stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to inspect deferred stochastic draft tokens");
                return false;
            }
            for (int row = 0; row < row_count; ++row)
            {
                const int32_t token = sampled_drafts[static_cast<size_t>(row)];
                if (token < 0)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Deferred stochastic draft sample slot="
                              << (first_draft_slot + row)
                              << " is invalid before batch verifier: token="
                              << token);
                    return false;
                }
            }
        }

        /*
         * First launch: row-local stochastic accept/reject decisions. The
         * sampled MTP draft tokens come from STOCHASTIC_DRAFT_SAMPLE_TOKENS,
         * which is written by sampleStochasticDistributionOnDevice(). The
         * compact outcome path never reads a host draft-token shadow.
         */
        bool verify_enqueued = false;
        if (sampled_draft_tokens)
        {
            if (use_vllm_probability_rejection)
            {
                if (!target_processed_logits ||
                    full_vocab_size <= 0 ||
                    first_target_slot + row_count > static_cast<int>(kStochasticTargetRows) ||
                    first_draft_slot + row_count > static_cast<int>(kStochasticDraftRows))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] vLLM stochastic verifier requested without processed target rows");
                    return false;
                }

                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "stochastic_vllm_processed_batch_verify_enqueue",
                    "decode",
                    state_.device_id.toString(),
                    {{"rows", std::to_string(row_count)},
                     {"vocab", std::to_string(full_vocab_size)}});
                verify_enqueued =
                    backend->enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
                        target_processed_logits + static_cast<size_t>(first_target_slot) * full_vocab_size,
                        /*draft_probabilities_device=*/nullptr,
                        row_count,
                        full_vocab_size,
                        full_vocab_size,
                        /*draft_row_stride=*/0,
                        sampled_draft_tokens,
                        accept_thresholds,
                        inverse_sample_seed,
                        inverse_sample_first_logical_position,
                        state_.device_id.gpu_ordinal(),
                        stream,
                        out_token_dev,
                        out_accepted_dev,
                        out_accept_prob_dev,
                        out_threshold_dev,
                        /*no_draft_probabilities=*/true);
            }
            else
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "stochastic_batch_verify_enqueue",
                    "decode",
                    state_.device_id.toString(),
                    {{"rows", std::to_string(row_count)},
                     {"top_k", std::to_string(target_top_k)}});
                verify_enqueued =
                    backend->enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
                        target_ids,
                        target_probs,
                        draft_ids,
                        draft_probs,
                        target_top_k,
                        static_cast<int>(kStochasticDistributionMaxK),
                        sampled_draft_tokens,
                        accept_thresholds,
                        residual_thresholds,
                        row_count,
                        state_.device_id.gpu_ordinal(),
                        stream,
                        out_token_dev,
                        out_accepted_dev,
                        out_accept_prob_dev,
                        out_threshold_dev,
                        sampled_draft_probs);
            }
        }
        if (!verify_enqueued)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU stochastic batch verifier requires device-resident draft tokens");
            return false;
        }
        PerfStatsCollector::addCounter(
            "mtp",
            use_vllm_probability_rejection
                ? "stochastic_vllm_processed_verify_batch_device_token_rows"
                : "stochastic_verify_batch_device_token_rows",
            static_cast<double>(row_count),
            "decode",
            state_.device_id.toString(),
            {{"top_k", std::to_string(target_top_k)}});

        if (debugEnv().validation.validate_buffers)
        {
            std::array<int32_t, kSpeculativeBatchMaxRows> verifier_tokens{};
            std::array<int32_t, kSpeculativeBatchMaxRows> verifier_accepted{};
            if (!backend->deviceToHostFast(
                    verifier_tokens.data(),
                    out_token_dev,
                    sizeof(int32_t) * static_cast<size_t>(row_count),
                    state_.device_id.gpu_ordinal(),
                    stream) ||
                !backend->deviceToHostFast(
                    verifier_accepted.data(),
                    out_accepted_dev,
                    sizeof(int32_t) * static_cast<size_t>(row_count),
                    state_.device_id.gpu_ordinal(),
                    stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to inspect stochastic verifier row outputs");
                return false;
            }
            for (int row = 0; row < row_count; ++row)
            {
                const int32_t token = verifier_tokens[static_cast<size_t>(row)];
                if (token < 0)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Stochastic verifier row="
                              << row << " produced invalid token=" << token
                              << " accepted="
                              << verifier_accepted[static_cast<size_t>(row)]);
                    return false;
                }
            }
        }

        const int *bonus_token_dev = nullptr;
        if (has_bonus)
        {
            // Bonus-ready sampling is unconditional in the batched lane. The
            // summary kernel exposes it only when every row accepts.
            auto *bonus_out =
                static_cast<int *>(stochastic_verify_tokens_dev_) + bonus_target_slot;
            bool bonus_enqueued = false;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "stochastic_batch_bonus_sample_enqueue",
                    "decode",
                    state_.device_id.toString(),
                    {{"top_k", std::to_string(target_top_k)}});
                if (use_vllm_probability_rejection)
                {
                    auto *bonus_logits =
                        static_cast<float *>(stochastic_processed_logits_dev_) +
                        static_cast<size_t>(bonus_target_slot) * full_vocab_size;
                    bonus_enqueued = backend->enqueueSampleProcessedLogitsF32Device(
                        bonus_logits,
                        full_vocab_size,
                        full_vocab_size,
                        bonus_threshold,
                        state_.device_id.gpu_ordinal(),
                        stream,
                        bonus_out);
                }
                else
                {
                    auto *bonus_ids =
                        static_cast<int *>(stochastic_target_token_ids_dev_) +
                        static_cast<size_t>(bonus_target_slot) * kStochasticDistributionMaxK;
                    auto *bonus_probs =
                        static_cast<float *>(stochastic_target_probs_dev_) +
                        static_cast<size_t>(bonus_target_slot) * kStochasticDistributionMaxK;
                    bonus_enqueued = backend->enqueueSampleDistributionF32Device(
                        bonus_ids,
                        bonus_probs,
                        target_top_k,
                        bonus_threshold,
                        state_.device_id.gpu_ordinal(),
                        stream,
                        bonus_out);
                }
            }
            if (!bonus_enqueued)
            {
                return false;
            }
            bonus_token_dev = bonus_out;
        }

        std::array<int, kSpeculativeBatchMaxStopTokens> packed_stop_tokens =
            {-1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < stop_token_count; ++i)
            packed_stop_tokens[static_cast<size_t>(i)] = stop_tokens[i];

        const int *first_token_dev = nullptr;
        if (first_token_from_device)
        {
            first_token_dev =
                static_cast<const int *>(stochastic_target_sample_tokens_dev_) +
                first_target_sample_slot;
            if (!waitForRequiredStochasticTargetSampleReady(
                    first_target_sample_slot,
                    stream,
                    "stochastic_batch_summary_first_token"))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to order stochastic batch summary after deferred target sample");
                return false;
            }
        }

        bool summary_enqueued = false;
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_batch_summary_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"rows", std::to_string(row_count)},
                 {"top_k", std::to_string(target_top_k)},
                 {"first_token_source", first_token_from_device ? "device" : "host"},
                 {"has_bonus", has_bonus ? "true" : "false"}});
            summary_enqueued =
                first_token_from_device
                    ? backend->enqueueSummarizeSpeculativeVerifyBatchDeviceFirstToken(
                          out_token_dev,
                          out_accepted_dev,
                          row_count,
                          first_token_dev,
                          packed_stop_tokens.data(),
                          stop_token_count,
                          bonus_token_dev,
                          has_bonus,
                          state_.device_id.gpu_ordinal(),
                          stream,
                          stochastic_batch_output_tokens_dev_,
                          stochastic_batch_output_meta_dev_)
                    : backend->enqueueSummarizeSpeculativeVerifyBatch(
                          out_token_dev,
                          out_accepted_dev,
                          row_count,
                          first_token,
                          packed_stop_tokens.data(),
                          stop_token_count,
                          bonus_token_dev,
                          has_bonus,
                          state_.device_id.gpu_ordinal(),
                          stream,
                          stochastic_batch_output_tokens_dev_,
                          stochastic_batch_output_meta_dev_);
        }
        if (!summary_enqueued)
        {
            return false;
        }
        if (first_token_from_device)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_batch_summary_device_first_tokens",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"slot", std::to_string(first_target_sample_slot)}});
        }

        std::array<int32_t, kSpeculativeBatchMaxOutputTokens> output_tokens{};
        std::array<int, kSpeculativeBatchMetaCount> meta{};
        const bool capture_row_debug =
            debugEnv().validation.validate_buffers ||
            Logger::getInstance().shouldLog(LogLevel::VERBOSITY_DEBUG);
        std::array<float, kSpeculativeBatchMaxRows> debug_accept_probs{};
        std::array<float, kSpeculativeBatchMaxRows> debug_accept_thresholds{};
        std::array<int32_t, kSpeculativeBatchMaxRows> debug_draft_tokens{};
        std::array<float, kSpeculativeBatchMaxRows> debug_draft_probs{};
        /*
         * The batch summary is tiny, fixed-shape metadata emitted by our own
         * kernel. Fast D2H is safe here and keeps ROCm from spending more time
         * validating the pointer than copying the result.
         */
        bool copied_summary = false;
        {
            /*
             * This is the intentional host-visible boundary for the compact
             * stochastic outcome. Its time includes any previously enqueued
             * distribution or verifier kernels that have not yet completed.
             */
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "stochastic_batch_summary_d2h_sync",
                "decode",
                state_.device_id.toString(),
                {{"rows", std::to_string(row_count)},
                 {"top_k", std::to_string(target_top_k)}});
            copied_summary =
                backend->deviceToHostFast(output_tokens.data(),
                                          stochastic_batch_output_tokens_dev_,
                                          sizeof(int32_t) * output_tokens.size(),
                                          state_.device_id.gpu_ordinal(),
                                          stream) &&
                backend->deviceToHostFast(meta.data(),
                                          stochastic_batch_output_meta_dev_,
                                          sizeof(int) * meta.size(),
                                          state_.device_id.gpu_ordinal(),
                                          stream);
            /*
             * The row-detail arrays exist only for debug logging and buffer
             * validation.  Keep production verification to the two semantic
             * transfers above: output tokens and compact outcome metadata.
             * On ROCm each fast D2H copy synchronizes the stream, so copying
             * debug-only rows here would turn one intentional host boundary
             * into several extra synchronization points.
             */
            if (copied_summary && capture_row_debug)
            {
                copied_summary =
                    backend->deviceToHostFast(debug_accept_probs.data(),
                                              out_accept_prob_dev,
                                              sizeof(float) * static_cast<size_t>(row_count),
                                              state_.device_id.gpu_ordinal(),
                                              stream) &&
                    backend->deviceToHostFast(debug_accept_thresholds.data(),
                                              out_threshold_dev,
                                              sizeof(float) * static_cast<size_t>(row_count),
                                              state_.device_id.gpu_ordinal(),
                                              stream) &&
                    backend->deviceToHostFast(debug_draft_tokens.data(),
                                              sampled_draft_tokens,
                                              sizeof(int32_t) * static_cast<size_t>(row_count),
                                              state_.device_id.gpu_ordinal(),
                                              stream) &&
                    (!sampled_draft_probs ||
                     backend->deviceToHostFast(debug_draft_probs.data(),
                                               sampled_draft_probs,
                                               sizeof(float) * static_cast<size_t>(row_count),
                                               state_.device_id.gpu_ordinal(),
                                               stream));
            }
        }
        if (!copied_summary)
        {
            return false;
        }
        if (capture_row_debug && row_count > 0)
        {
            std::ostringstream row_debug;
            row_debug << "[DeviceGraphOrchestrator] stochastic batch outcome rows="
                      << row_count
                      << " output_count=" << meta[kSpecBatchMetaOutputCount]
                      << " accepted_prefix=" << meta[kSpecBatchMetaAcceptedSpeculativePrefix]
                      << " all_accepted=" << meta[kSpecBatchMetaAllSpeculativeAccepted]
                      << " sampled_terminal=" << meta[kSpecBatchMetaSampledTerminal]
                      << " ready_token=" << meta[kSpecBatchMetaReadyToken]
                      << " rejected_token=" << meta[kSpecBatchMetaRejectedVerifiedToken]
                      << " per_row=";
            for (int row = 0; row < row_count; ++row)
            {
                if (row > 0)
                    row_debug << ",";
                row_debug << "{p=" << debug_accept_probs[static_cast<size_t>(row)]
                          << ",u=" << debug_accept_thresholds[static_cast<size_t>(row)]
                          << ",draft=" << debug_draft_tokens[static_cast<size_t>(row)]
                          << ",q=";
                if (sampled_draft_probs)
                    row_debug << debug_draft_probs[static_cast<size_t>(row)];
                else
                    row_debug << "one_hot";
                row_debug
                          << "}";
            }
            LOG_DEBUG(row_debug.str());
        }

        if (meta[kSpecBatchMetaOk] == 0 ||
            meta[kSpecBatchMetaOutputCount] < 0 ||
            meta[kSpecBatchMetaOutputCount] > kSpeculativeBatchMaxOutputTokens)
        {
            return false;
        }
        for (int i = 0; i < meta[kSpecBatchMetaOutputCount]; ++i)
        {
            if (output_tokens[static_cast<size_t>(i)] < 0)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Stochastic batch summary produced invalid output token="
                          << output_tokens[static_cast<size_t>(i)]
                          << " at index=" << i
                          << " output_count=" << meta[kSpecBatchMetaOutputCount]
                          << " accepted_prefix="
                          << meta[kSpecBatchMetaAcceptedSpeculativePrefix]
                          << " consumed_rows="
                          << meta[kSpecBatchMetaConsumedVerifierRows]);
                return false;
            }
        }

        for (int row = 0; row < row_count; ++row)
        {
            stochastic_target_distribution_streams_[
                static_cast<size_t>(first_target_slot + row)] = nullptr;
            stochastic_draft_distribution_streams_[
                static_cast<size_t>(first_draft_slot + row)] = nullptr;
            stochastic_draft_top_k_[static_cast<size_t>(first_draft_slot + row)] = 0;
            clearStochasticDraftSampleReadySlot(first_draft_slot + row);
        }
        if (has_bonus)
        {
            stochastic_target_distribution_streams_[
                static_cast<size_t>(bonus_target_slot)] = nullptr;
        }
        if (first_token_from_device)
        {
            clearStochasticTargetSampleReadySlot(first_target_sample_slot);
        }

        out->ok = true;
        out->output_tokens = output_tokens;
        out->output_token_count = meta[kSpecBatchMetaOutputCount];
        out->accepted_speculative_prefix =
            meta[kSpecBatchMetaAcceptedSpeculativePrefix];
        out->target_verifier_state_commit_count =
            meta[kSpecBatchMetaTargetVerifierStateCommitCount];
        out->ready_token = meta[kSpecBatchMetaReadyToken];
        out->rejected_verified_token = meta[kSpecBatchMetaRejectedVerifiedToken];
        out->stopped_on_output = meta[kSpecBatchMetaStoppedOnOutput] != 0;
        out->all_speculative_accepted =
            meta[kSpecBatchMetaAllSpeculativeAccepted] != 0;
        out->consumed_verifier_rows = meta[kSpecBatchMetaConsumedVerifierRows];
        out->sampled_terminal = meta[kSpecBatchMetaSampledTerminal] != 0;

        PerfStatsCollector::addCounter(
            "mtp",
            "stochastic_verify_batch_outcome_rows",
            static_cast<double>(row_count),
            "decode",
            state_.device_id.toString(),
            {{"top_k", std::to_string(target_top_k)}});
        return true;
    }

    // =========================================================================
    // Batch Interface Implementation
    // =========================================================================

    bool DeviceGraphOrchestrator::forward_batch(const std::vector<std::vector<int>> &token_batches)
    {
        // Enable device-scoped logging for this execution
        ScopedDeviceLog device_log(state_.device_id);

        if (token_batches.empty())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] forward_batch() called with empty batch");
            return false;
        }

        int batch_size = static_cast<int>(token_batches.size());
        if (batch_size > state_.batch_size)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Batch size " << batch_size
                                                              << " exceeds initialized batch size " << state_.batch_size);
            return false;
        }

        // Find max sequence length (for padding)
        int max_len = 0;
        for (const auto &seq : token_batches)
        {
            max_len = std::max(max_len, static_cast<int>(seq.size()));
        }
        padded_seq_len_ = max_len;

        // Store actual lengths BEFORE calling forward and publish them through
        // state_.sequence_lengths so attention masks see request-local lengths
        // during the padded graph execution. forward() increments positions and
        // sequence lengths by padded_seq_len; we restore old+actual below.
        std::vector<int> actual_lengths(batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            actual_lengths[i] = static_cast<int>(token_batches[i].size());
        }
        std::vector<int> old_positions = state_.positions;
        std::vector<int> old_sequence_lengths = state_.sequence_lengths;
        state_.sequence_lengths.resize(batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            state_.sequence_lengths[i] = actual_lengths[i];
        }

        // Create flattened, padded token array [batch_size * padded_seq_len]
        std::vector<int> flat_tokens(batch_size * padded_seq_len_, 0); // pad with 0
        for (int b = 0; b < batch_size; ++b)
        {
            const auto &seq = token_batches[b];
            for (size_t s = 0; s < seq.size(); ++s)
            {
                flat_tokens[b * padded_seq_len_ + s] = seq[s];
            }
        }

        // Call the 3-parameter forward() with padded tokens
        // Note: forward() will set sequence_lengths[b] = padded_seq_len for all b
        const float *result = forward(flat_tokens.data(), padded_seq_len_, batch_size);

        // Restore actual request progress after the padded graph execution.
        // This is important for:
        // 1. Proper logits extraction (only extract non-padded logits)
        // 2. Snapshot comparison (shapes should match actual token count)
        // 3. KV cache position tracking (only actual tokens contribute to cache)
        state_.positions.resize(batch_size);
        state_.sequence_lengths.resize(batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            const int old_pos =
                i < static_cast<int>(old_positions.size()) ? old_positions[i] : 0;
            const int old_len =
                i < static_cast<int>(old_sequence_lengths.size()) ? old_sequence_lengths[i] : old_pos;
            state_.positions[i] = old_pos + actual_lengths[i];
            state_.sequence_lengths[i] = old_len + actual_lengths[i];
        }

        return result != nullptr;
    }

    const float *DeviceGraphOrchestrator::getLogits(int seq_idx) const
    {
        if (!state_.logits)
        {
            return nullptr;
        }

        if (seq_idx < 0 || seq_idx >= state_.batch_size)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid sequence index " << seq_idx
                                                                          << " (batch_size=" << state_.batch_size << ")");
            return nullptr;
        }

        // Return pointer to logits for requested sequence
        // Layout: [batch_size, vocab_size] (LM head always computes M=1 per batch entry)
        // For sequence seq_idx, logits start at row seq_idx
        const float *base = state_.logits->fp32_data();
        if (!base)
        {
            return nullptr;
        }

        return base + (seq_idx * state_.vocab_size);
    }

    int DeviceGraphOrchestrator::getPosition(int seq_idx) const
    {
        if (seq_idx < 0 || static_cast<size_t>(seq_idx) >= state_.positions.size())
        {
            return 0;
        }
        return state_.positions[seq_idx];
    }

    void DeviceGraphOrchestrator::clearInferenceState()
    {
        state_.clear();

        if (forward_engine_)
            forward_engine_->resetSessionReplayState();
        mtp_sidecar_depth0_cache_.resetSessionState();
        mtp_sidecar_depth0_device_token_cache_.resetSessionState();
        mtp_sidecar_depth0_chained_cache_.resetSessionState();
        mtp_sidecar_depth0_chained_device_token_cache_.resetSessionState();
        mtp_sidecar_depth0_kv_only_cache_.resetSessionState();
        mtp_sidecar_depth0_kv_only_device_token_cache_.resetSessionState();
        for (auto &cache : mtp_sidecar_depth0_kv_only_batch_caches_)
            cache.resetSessionState();
        mtp_terminal_hidden_row_select_cache_.resetSessionState();
        mtp_terminal_hidden_rows_select_cache_.resetSessionState();
        defer_next_mtp_main_decode_sync_ = false;
        defer_all_position_verifier_sync_ = false;
        clearAllPendingLogitsStreams("clearInferenceState");
        stochastic_target_distribution_streams_.fill(nullptr);
        stochastic_draft_distribution_streams_.fill(nullptr);
        stochastic_target_top_k_.fill(0);
        stochastic_draft_top_k_.fill(0);
        clearStochasticTargetSampleReadySlots();
        clearStochasticDraftSampleReadySlots();

        for (auto &cache : layer_graph_cache_)
        {
            cache.resetSessionState();
        }

        resetKernelDynamicState();
        recordLivePrefixSessionReset("clearInferenceState");

        LOG_DEBUG("[DeviceGraphOrchestrator] Inference state cleared (cached graph topology preserved)");
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    void DeviceGraphOrchestrator::updateCachedGraphParams(ComputeGraph &graph, int pos_offset, int seq_len)
    {
        // Update all stages in the graph that have dynamic parameters
        const auto &order = graph.getExecutionOrder();

        for (const auto &node_name : order)
        {
            ComputeNode *node = graph.getNode(node_name);
            if (!node || !node->stage)
                continue;

            // Update dynamic params (pos_offset, seq_len)
            // Only stages that override updateDynamicParams will actually do anything
            node->stage->updateDynamicParams(pos_offset, seq_len);
        }

        LOG_TRACE("[DeviceGraphOrchestrator] Updated cached graph params: pos_offset="
                  << pos_offset << " seq_len=" << seq_len);
    }

    bool DeviceGraphOrchestrator::canUseCachedGraph(int layer_idx, int seq_len) const
    {
        if (!cache_config_.enabled)
            return false;
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_graph_cache_.size())
            return false;

        const auto &cache = layer_graph_cache_[layer_idx];
        return cache.valid && cache.cached_seq_len == seq_len;
    }

    // =========================================================================
    // MoE Expert Rebalance Controller
    // =========================================================================

    void DeviceGraphOrchestrator::setMoERebalanceController(
        std::unique_ptr<MoERebalanceController> controller)
    {
        moe_rebalance_extra_controllers_.clear();
        moe_rebalance_controller_ = std::move(controller);
    }

    void DeviceGraphOrchestrator::addMoERebalanceController(
        std::unique_ptr<MoERebalanceController> controller)
    {
        if (!controller)
            return;
        if (!moe_rebalance_controller_)
        {
            moe_rebalance_controller_ = std::move(controller);
            return;
        }
        moe_rebalance_extra_controllers_.push_back(std::move(controller));
    }

    std::vector<MoERebalanceController *> DeviceGraphOrchestrator::moeRebalanceControllers() const
    {
        std::vector<MoERebalanceController *> controllers;
        if (moe_rebalance_controller_)
            controllers.push_back(moe_rebalance_controller_.get());
        for (const auto &controller : moe_rebalance_extra_controllers_)
        {
            if (controller)
                controllers.push_back(controller.get());
        }
        return controllers;
    }

    MoERebalanceController *DeviceGraphOrchestrator::moeRebalanceControllerForDomain(
        const std::string &domain_id) const
    {
        if (moe_rebalance_controller_ && moe_rebalance_controller_->domainId() == domain_id)
            return moe_rebalance_controller_.get();
        for (const auto &controller : moe_rebalance_extra_controllers_)
        {
            if (controller && controller->domainId() == domain_id)
                return controller.get();
        }
        return nullptr;
    }

    int DeviceGraphOrchestrator::moeRebalanceParticipantId() const
    {
        if (const auto *global_ctx = globalTPContextForMTPCoordination())
            return global_ctx->myIndex();
        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
            return mpi_ctx_->rank();
        return 0;
    }

    void DeviceGraphOrchestrator::initializeExpertPayloadProvider()
    {
        // Count MoE stages to decide if a provider is needed
        int moe_stage_count = 0;
        if (forward_engine_)
        {
            forward_engine_->forEachCachedStage(
                ComputeStageType::MOE_EXPERT_FFN,
                [&](IComputeStage *)
                { ++moe_stage_count; });
        }

        if (moe_stage_count == 0)
        {
            LOG_DEBUG("[DGO] No MoE stages found — skipping payload provider initialization");
            return;
        }

        // Create provider
        expert_payload_provider_ = std::make_unique<ExpertWeightPayloadProvider>();

        // Wire to all cached MoE stages
        int wired = 0;
        forward_engine_->forEachCachedStage(
            ComputeStageType::MOE_EXPERT_FFN,
            [&](IComputeStage *s)
            {
                auto *moe = dynamic_cast<MoEExpertComputeStage *>(s);
                if (moe)
                {
                    moe->setPayloadProvider(expert_payload_provider_.get());
                    ++wired;
                }
            });

        // Wire to WeightManager for host retention decisions
        if (weight_manager_)
        {
            weight_manager_->setExpertPayloadProvider(expert_payload_provider_.get());
        }

        LOG_DEBUG("[DGO] Expert payload provider initialized and wired to "
                  << wired << " MoE stages");
    }

    void DeviceGraphOrchestrator::initializePreparedWeightStore(DeviceId device)
    {
        if (!weight_manager_)
        {
            LOG_DEBUG("[DGO] No weight manager — skipping prepared weight store");
            return;
        }

        ModelContextId requested_model_id{};
        requested_model_id.value = reinterpret_cast<uint64_t>(this); // Fallback only when no model-owned store exists
        if (frozen_weight_set_ && frozen_weight_set_->strategy().model_id.value != 0)
            requested_model_id = frozen_weight_set_->strategy().model_id;

        if (!prepared_weight_store_)
        {
            if (auto concrete_weight_manager = std::dynamic_pointer_cast<WeightManager>(weight_manager_))
            {
                prepared_weight_store_ = concrete_weight_manager->preparedWeightStoreIfInitialized();
            }
            if (!prepared_weight_store_)
            {
                prepared_weight_store_ = std::make_shared<PreparedWeightStore>(requested_model_id);
            }
            if (auto concrete_weight_manager = std::dynamic_pointer_cast<WeightManager>(weight_manager_))
            {
                concrete_weight_manager->setPreparedWeightStore(prepared_weight_store_);
            }
        }

        if (!prepared_weight_store_->bindModelIdIfUnset(requested_model_id))
        {
            throw std::runtime_error(
                "[DGO] Prepared weight store model id mismatch: store=" +
                std::to_string(prepared_weight_store_->modelId().value) +
                " requested=" + std::to_string(requested_model_id.value));
        }

        ModelContextId model_id = prepared_weight_store_->modelId();
        if (model_id.value == 0)
            model_id = requested_model_id;

        int registered = 0;
        uint64_t next_binding_id = 1;
        if (frozen_weight_set_)
        {
            for (const auto &binding : frozen_weight_set_->bindings())
                next_binding_id = std::max(next_binding_id, binding.binding_id + 1);
        }

        auto register_if_prepared = [&](const WeightBinding &source_binding)
        {
            const std::string &name = source_binding.identity.canonical_name;
            TensorBase *tensor = source_binding.tensor;
            if (!tensor)
                return;

            WeightBinding binding = source_binding;

            if (!frozen_weight_set_)
                binding.identity.model_id = model_id;
            else if (binding.identity.model_id.value == 0)
                binding.identity.model_id = model_id;
            binding.residency.home_device = device;
            binding.residency.resident_device = device;

            try
            {
                if (binding.identity.role == WeightRole::Embedding)
                {
                    const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor);
                    if (device.is_gpu() && unpackable && unpackable->vnniFormatInfo())
                    {
                        if (prepared_weight_store_->preparedRefForBinding(binding.binding_id, device).has_value())
                            return;
                        const auto &cfg = graph_builder_->config();
                        size_t vocab_offset = 0;
                        size_t total_vocab = static_cast<size_t>(cfg.vocab_size);
                        if (cfg.tp_config)
                        {
                            try
                            {
                                const auto &assignment = cfg.tp_config->forDevice(device);
                                vocab_offset = static_cast<size_t>(assignment.vocab_start);
                                total_vocab = static_cast<size_t>(cfg.tp_config->totalVocab());
                            }
                            catch (const std::exception &)
                            {
                                // Fall back to unsharded metadata below.
                            }
                        }

                        prepared_weight_store_->prepareEmbedding(
                            binding,
                            cfg.d_model,
                            vocab_offset,
                            total_vocab);
                        ++registered;
                    }
                    return;
                }

                if (binding.identity.role == WeightRole::Embedding ||
                    binding.identity.role == WeightRole::Norm ||
                    binding.identity.role == WeightRole::Bias ||
                    tensor->shape().size() != 2)
                {
                    return;
                }

                if (prepared_weight_store_->preparedRefForBinding(binding.binding_id, device).has_value())
                {
                    return;
                }

                if (device.is_cpu())
                {
                    prepared_weight_store_->prepareGemm(binding);
                    ++registered;
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN("[DGO] Failed to register prepared weight '"
                         << name << "' in store: " << e.what());
            }
        };

        if (frozen_weight_set_)
        {
            // Iterate graph-frozen bindings rather than WeightManager names so
            // aliases such as tied output.weight -> token_embd.weight keep their
            // own binding ids and roles.
            for (const auto &binding : frozen_weight_set_->bindings())
                register_if_prepared(binding);
        }
        else
        {
            // Fallback for non-frozen legacy graph setup.
            weight_manager_->forEachPreparedWeight([&](const std::string &name, TensorBase *tensor)
                                                   {
                WeightBinding binding;
                binding.binding_id = next_binding_id++;
                binding.identity.canonical_name = name;
                binding.identity.role = inferWeightRole(name);
                binding.identity.layer = inferWeightLayer(name);
                binding.tensor = tensor;
                register_if_prepared(binding); });
        }

        LOG_DEBUG("[DGO] Prepared weight store initialized with "
                  << prepared_weight_store_->size() << " entries for device " << device.toString()
                  << " (new=" << registered << ")");
        if (auto concrete_weight_manager = std::dynamic_pointer_cast<WeightManager>(weight_manager_))
            concrete_weight_manager->logHostMemorySummary("after base preparation");

        // Phase 10: Wire prepared weight store to graph builder so stages
        // can resolve kernels through the store instead of KernelFactory fallbacks.
        if (graph_builder_)
        {
            graph_builder_->setPreparedWeightStore(prepared_weight_store_.get());
        }
    }

    void DeviceGraphOrchestrator::applyExpertMasks(
        const std::vector<std::vector<bool>> &masks,
        const ReceivedWeightsMap &received_weights)
    {
        applyExpertMasksForDomain({}, masks, received_weights);
    }

    void DeviceGraphOrchestrator::applyExpertMasksForDomain(
        const std::string &domain_id,
        const std::vector<std::vector<bool>> &masks,
        const ReceivedWeightsMap &received_weights)
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        validateMoERebalanceDomain(*this, domain_id, "applyExpertMasks");

        // Lazily initialize payload provider on first mask application
        if (!expert_payload_provider_)
        {
            initializeExpertPayloadProvider();
        }

        // ── Step 1: Collect all MoE stages ──────────────────────────────
        struct StageInfo
        {
            MoEExpertComputeStage *stage;
            int layer;
        };
        std::vector<StageInfo> moe_stages;

        if (forward_engine_)
            if (prepared_weight_store_)
            {
                forward_engine_->forEachCachedStage(
                    ComputeStageType::MOE_EXPERT_FFN,
                    [&](IComputeStage *s)
                    {
                        auto *moe = dynamic_cast<MoEExpertComputeStage *>(s);
                        if (!moe)
                            return;
                        int layer = moe->layerIndex();
                        if (layer >= 0 && static_cast<size_t>(layer) < masks.size())
                            moe_stages.push_back({moe, layer});
                    });
            }

        // Fallback: legacy layer_graph_cache_ path
        if (moe_stages.empty())
        {
            for (size_t layer = 0; layer < layer_graph_cache_.size() && layer < masks.size(); ++layer)
            {
                auto &cache = layer_graph_cache_[layer];
                if (!cache.valid || !cache.ffn_decode)
                    continue;
                for (const auto &node_name : cache.ffn_decode->getExecutionOrder())
                {
                    auto *node = cache.ffn_decode->getNode(node_name);
                    if (!node || !node->stage)
                        continue;
                    if (node->stage->type() == ComputeStageType::MOE_EXPERT_FFN)
                    {
                        auto *moe = dynamic_cast<MoEExpertComputeStage *>(node->stage.get());
                        if (moe)
                            moe_stages.push_back({moe, static_cast<int>(layer)});
                    }
                }
            }
        }

        // ── Step 2: Phase 1 — release departed experts ─────────────────
        for (auto &[stage, layer] : moe_stages)
        {
            (void)stage->releaseDepartedExperts(masks[layer]);
        }

        // ── Step 3: Phase 2 — register + prepare (parallel across stages)
        std::atomic<int> applied{0};
        const int n = static_cast<int>(moe_stages.size());

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i)
        {
            auto &[stage, layer] = moe_stages[i];
            const std::unordered_map<int, ExpertWeightBlobs> *layer_received = nullptr;
            auto layer_it = received_weights.find(layer);
            if (layer_it != received_weights.end())
                layer_received = &layer_it->second;

            if (stage->registerAndPrepareNewExperts(masks[layer], layer_received))
                applied.fetch_add(1, std::memory_order_relaxed);
        }

        if (applied.load(std::memory_order_relaxed) != n)
        {
            LOG_ERROR("[DGO] Failed to prepare " << (n - applied.load(std::memory_order_relaxed))
                                                 << " MoEExpertComputeStages during expert mask application");
            throw std::runtime_error("MoE expert mask application failed: missing prepared expert weights");
        }

        // ── Step 5: Phase 3 — apply masks (fast, no heavy ops) ─────────
        for (auto &[stage, layer] : moe_stages)
            stage->applyExpertMask(masks[layer]);

        LOG_DEBUG("[DGO] Applied expert masks to " << applied.load()
                                                   << " MoEExpertComputeStages across " << masks.size() << " layers");

        auto t_end = std::chrono::high_resolution_clock::now();
        double prep_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        if (moe_rebalance_controller_)
            moe_rebalance_controller_->recordPrepDuration(prep_ms);
        LOG_DEBUG("[DGO] Expert mask application + engine prep took "
                  << std::fixed << std::setprecision(1) << prep_ms << " ms");
    }

    ReceivedWeightsMap DeviceGraphOrchestrator::collectExpertWeightsForMasks(
        const std::vector<std::vector<bool>> &masks) const
    {
        ReceivedWeightsMap result;

        std::unordered_map<int, const MoEExpertComputeStage *> moe_by_layer;
        if (forward_engine_)
        {
            forward_engine_->forEachCachedStage(
                ComputeStageType::MOE_EXPERT_FFN,
                [&](IComputeStage *stage)
                {
                    const auto *moe_stage = dynamic_cast<const MoEExpertComputeStage *>(stage);
                    if (moe_stage && moe_stage->layerIndex() >= 0)
                        moe_by_layer[moe_stage->layerIndex()] = moe_stage;
                });
        }

        for (size_t layer_idx = 0; layer_idx < masks.size(); ++layer_idx)
        {
            auto stage_it = moe_by_layer.find(static_cast<int>(layer_idx));
            if (stage_it == moe_by_layer.end())
                continue;

            const auto &mask = masks[layer_idx];
            for (size_t expert_idx = 0; expert_idx < mask.size(); ++expert_idx)
            {
                if (!mask[expert_idx])
                    continue;
                auto blobs = stage_it->second->serializeExpert(static_cast<int>(expert_idx));
                if (!blobs.empty())
                    result[static_cast<int>(layer_idx)][static_cast<int>(expert_idx)] = std::move(blobs);
            }
        }

        return result;
    }

    void DeviceGraphOrchestrator::setExpertReplicaSetForParticipant(
        const ExpertReplicaSet &replicas, int participant_id)
    {
        validateMoERebalanceDomain(*this, replicas.domain_id, "setExpertReplicaSetForParticipant");

        int count = 0;
        if (forward_engine_)
        {
            forward_engine_->forEachCachedStage(
                ComputeStageType::MOE_EXPERT_FFN,
                [&](IComputeStage *s)
                {
                    auto *moe = dynamic_cast<MoEExpertComputeStage *>(s);
                    if (moe)
                    {
                        moe->setReplicaSet(replicas, participant_id);
                        count++;
                    }
                });
        }

        LOG_DEBUG("[DGO] Set expert replica info (" << replicas.num_replicated
                                                    << " replicas) on " << count
                                                    << " MoE stages (participant " << participant_id << ")");
    }

    void DeviceGraphOrchestrator::setExpertReplicaSet(
        const ExpertReplicaSet &replicas, int socket_id)
    {
        setExpertReplicaSetForParticipant(replicas, socket_id);
    }

    size_t DeviceGraphOrchestrator::releaseRawExpertWeights()
    {
        size_t total_freed = 0;
        int stage_count = 0;
        PreparedWeightStore *summary_store = prepared_weight_store_.get();

        if (forward_engine_)
        {
            forward_engine_->forEachCachedStage(
                ComputeStageType::MOE_EXPERT_FFN,
                [&](IComputeStage *s)
                {
                    auto *moe = dynamic_cast<MoEExpertComputeStage *>(s);
                    if (moe)
                    {
                        if (!summary_store)
                            summary_store = moe->buildWeightContext().prepared_store;
                        total_freed += moe->releaseRawExpertWeights();
                        ++stage_count;
                    }
                });
        }

        LOG_DEBUG("[DGO] Released raw expert weights across " << stage_count
                                                              << " MoE stages: " << (total_freed >> 20) << " MB freed");
        if (summary_store)
            if (auto concrete_weight_manager = std::dynamic_pointer_cast<WeightManager>(weight_manager_))
                concrete_weight_manager->logHostMemorySummary("after rebalance raw release");
        return total_freed;
    }

    bool DeviceGraphOrchestrator::materializeForwardGraphForShape(int seq_len, int batch_size)
    {
        if (!state_.isInitialized())
        {
            LOG_ERROR("[DGO] Cannot materialize forward graph before inference state initialization");
            return false;
        }
        if (!hasGlobalWeights())
        {
            LOG_ERROR("[DGO] Cannot materialize forward graph before weights are configured");
            return false;
        }
        if (seq_len <= 0 || batch_size <= 0)
        {
            LOG_ERROR("[DGO] Invalid materialization shape: seq_len=" << seq_len
                                                                      << " batch_size=" << batch_size);
            return false;
        }
        if (batch_size > state_.batch_size)
        {
            LOG_ERROR("[DGO] Materialization batch size " << batch_size
                                                          << " exceeds initialized batch size " << state_.batch_size);
            return false;
        }

        const int total_tokens = batch_size * seq_len;
        if (total_tokens > state_.batch_size * state_.max_seq_len)
        {
            LOG_ERROR("[DGO] Materialization token count " << total_tokens
                                                           << " exceeds buffer capacity "
                                                           << (state_.batch_size * state_.max_seq_len));
            return false;
        }
        const int activation_seq_len =
            state_.activation_seq_len > 0 ? state_.activation_seq_len : state_.max_seq_len;
        if (total_tokens > state_.batch_size * activation_seq_len)
        {
            LOG_ERROR("[DGO] Materialization token count " << total_tokens
                                                           << " exceeds activation graph buffer capacity "
                                                           << (state_.batch_size * activation_seq_len)
                                                           << " (context capacity is "
                                                           << (state_.batch_size * state_.max_seq_len)
                                                           << ")");
            return false;
        }

        if (pp_stage_config_.has_value() && !pp_stage_config_->has_embedding)
        {
            LOG_DEBUG("[DGO] Skipping eager graph materialization for non-embedding PP stage");
            return true;
        }

        std::vector<int> token_ids(static_cast<size_t>(total_tokens), 0);
        std::vector<int> position_ids(static_cast<size_t>(total_tokens));
        for (int i = 0; i < total_tokens; ++i)
            position_ids[static_cast<size_t>(i)] = i % seq_len;

        ModelBuffers model_buffers = state_.toModelBuffers();
        setBuffers(model_buffers);

        ForwardInput input;
        input.token_ids = token_ids.data();
        input.position_ids = position_ids.data();
        input.batch_size = batch_size;
        input.seq_len = seq_len;
        input.position_offset = 0;
        input.device = state_.device_id;
        input.kv_cache = state_.kv_cache.get();

        std::unordered_map<DeviceId, IKVCache *> pp_kv_cache_ptrs;
        if (!state_.pp_kv_caches.empty())
        {
            for (const auto &[device, cache] : state_.pp_kv_caches)
                pp_kv_cache_ptrs[device] = cache.get();
            input.pp_kv_caches = &pp_kv_cache_ptrs;
        }

        auto result = buildForwardGraph(input);
        if (!result)
        {
            LOG_ERROR("[DGO] Eager forward graph materialization failed: " << result.error());
            return false;
        }

        const size_t stage_count = result.graph().size();
        LOG_DEBUG("[DGO] Eagerly materialized forward graph for "
                  << (graph_builder_ ? graph_builder_->architectureName() : std::string("unknown"))
                  << " shape=[batch=" << batch_size << ", seq=" << seq_len << "]"
                  << " stages=" << stage_count);
        return true;
    }

    ReceivedWeightsMap DeviceGraphOrchestrator::transferExpertWeights(
        const std::vector<ExpertMigration> &manifest,
        int num_layers)
    {
        if (manifest.empty() || !mpi_ctx_)
            return {};

        int my_rank = mpi_ctx_->rank();
        MPI_Comm comm = mpi_ctx_->communicator();

        // Collect MoE stages by layer from forward engine's graph cache.
        std::unordered_map<int, MoEExpertComputeStage *> moe_by_layer;
        if (forward_engine_)
        {
            forward_engine_->forEachCachedStage(
                ComputeStageType::MOE_EXPERT_FFN,
                [&](IComputeStage *stage)
                {
                    auto *moe_stage = dynamic_cast<MoEExpertComputeStage *>(stage);
                    if (moe_stage && moe_stage->layerIndex() >= 0)
                        moe_by_layer[moe_stage->layerIndex()] = moe_stage;
                });
        }

        // Fallback: legacy layer_graph_cache_
        if (moe_by_layer.empty())
        {
            for (size_t layer_idx = 0; layer_idx < layer_graph_cache_.size(); ++layer_idx)
            {
                auto &cache = layer_graph_cache_[layer_idx];
                if (!cache.valid || !cache.ffn_decode)
                    continue;
                for (const auto &node_name : cache.ffn_decode->getExecutionOrder())
                {
                    auto *node = cache.ffn_decode->getNode(node_name);
                    if (!node || !node->stage)
                        continue;
                    if (node->stage->type() == ComputeStageType::MOE_EXPERT_FFN)
                    {
                        auto *moe_stage = dynamic_cast<MoEExpertComputeStage *>(node->stage.get());
                        if (moe_stage)
                            moe_by_layer[static_cast<int>(layer_idx)] = moe_stage;
                    }
                }
            }
        }

        LOG_DEBUG("[DGO] Found " << moe_by_layer.size() << " MoE stages for weight transfer"
                                 << " (forward_engine=" << (forward_engine_ ? "yes" : "no") << ")");

        auto get_blobs = [&](int layer_idx, int expert_id) -> ExpertWeightBlobs
        {
            auto it = moe_by_layer.find(layer_idx);
            if (it == moe_by_layer.end())
                return {};
            return it->second->detachAndSerializeExpert(expert_id);
        };

        return ExpertWeightTransfer::transferAllLayers(manifest, num_layers, get_blobs, my_rank, comm);
    }

    ReceivedWeightsMap DeviceGraphOrchestrator::transferReplicaWeights(
        const ExpertReplicaSet &replicas,
        int num_layers)
    {
        if (replicas.num_replicated == 0 || !mpi_ctx_)
            return {};

        int my_rank = mpi_ctx_->rank();
        int world_size = mpi_ctx_->world_size();
        MPI_Comm comm = mpi_ctx_->communicator();

        // Build manifest: for each replicated expert, the owner sends to all
        // non-owner ranks. With 2 sockets this is a simple bidirectional exchange.
        std::vector<ExpertMigration> manifest;
        for (int e = 0; e < static_cast<int>(replicas.is_replicated.size()); ++e)
        {
            if (!replicas.is_replicated[e])
                continue;
            int owner = replicas.owner_socket[e];
            // Owner sends to every other rank
            for (int r = 0; r < world_size; ++r)
            {
                if (r != owner)
                    manifest.push_back({e, owner, r});
            }
        }

        if (manifest.empty())
            return {};

        LOG_DEBUG("[DGO] Transferring " << replicas.num_replicated
                                        << " replicated experts × " << num_layers << " layers via MPI");

        // Collect MoE stages by layer
        std::unordered_map<int, MoEExpertComputeStage *> moe_by_layer;
        if (forward_engine_)
        {
            forward_engine_->forEachCachedStage(
                ComputeStageType::MOE_EXPERT_FFN,
                [&](IComputeStage *stage)
                {
                    auto *moe_stage = dynamic_cast<MoEExpertComputeStage *>(stage);
                    if (moe_stage && moe_stage->layerIndex() >= 0)
                        moe_by_layer[moe_stage->layerIndex()] = moe_stage;
                });
        }

        // Non-destructive serialize callback — owner keeps its weights
        auto get_blobs = [&](int layer_idx, int expert_id) -> ExpertWeightBlobs
        {
            auto it = moe_by_layer.find(layer_idx);
            if (it == moe_by_layer.end())
                return {};
            return it->second->serializeExpert(expert_id);
        };

        return ExpertWeightTransfer::transferAllLayers(manifest, num_layers, get_blobs, my_rank, comm);
    }

    // =========================================================================
    // Phase-Aware Weight Access (Gap 3 - CPU Decode Participation)
    // =========================================================================

    void DeviceGraphOrchestrator::setWeightManager(std::shared_ptr<IWeightManager> weight_manager)
    {
        weight_manager_ = std::move(weight_manager);
        LOG_DEBUG("[DeviceGraphOrchestrator] WeightManager set");
    }

    void DeviceGraphOrchestrator::setWeightPlacementMap(std::shared_ptr<IWeightPlacementMap> placement_map)
    {
        weight_placement_map_ = std::move(placement_map);
        LOG_DEBUG("[DeviceGraphOrchestrator] WeightPlacementMap set");
    }

    // =========================================================================
    // Tensor Parallel Configuration (Phase 1c: Proportional TP)
    // =========================================================================

    void DeviceGraphOrchestrator::setTensorParallelConfig(std::shared_ptr<TensorParallelConfig> config)
    {
        tp_config_ = std::move(config);

        if (tp_config_)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] TensorParallelConfig set: "
                      << "world_size=" << tp_config_->worldSize()
                      << ", proportional=" << (tp_config_->isProportional() ? "yes" : "no"));

            // If we have a graph builder, propagate the config to it
            if (graph_builder_)
            {
                // Note: The graph builder's config is read-only after construction,
                // but we store the tp_config for use in buffer allocation and KV cache creation
                LOG_DEBUG("[DeviceGraphOrchestrator] TensorParallelConfig will be used for buffer sizing");
            }
        }
        else
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] TensorParallelConfig cleared");
        }
    }

    // =========================================================================
    // Multi-Domain Tensor Parallel Configuration (Phase 6.3: Heterogeneous TP)
    // =========================================================================

    void DeviceGraphOrchestrator::setDomainConfig(std::shared_ptr<MultiDomainTPConfig> config)
    {
        domain_config_ = std::move(config);

        if (domain_config_)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] MultiDomainTPConfig set: "
                      << "domains=" << domain_config_->domains().size()
                      << ", has_gpu=" << (domain_config_->gpuDomain() ? "yes" : "no")
                      << ", has_cpu=" << (domain_config_->cpuDomain() ? "yes" : "no")
                      << ", cross_rank=" << (domain_config_->hasCrossRankTP() ? "yes" : "no"));

            // Propagate domain config to graph builder if present
            if (graph_builder_)
            {
                // Graph builder can access domain config through config_.multi_domain_tp_config
                // Note: The graph builder uses getDomainForLayer() which delegates to this config
                LOG_DEBUG("[DeviceGraphOrchestrator] Domain config available for AllreduceStage routing");
            }
        }
        else
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] MultiDomainTPConfig cleared (legacy MPI path)");
        }
    }

    // =========================================================================
    // Pipeline Parallelism Configuration
    // =========================================================================

    void DeviceGraphOrchestrator::setPPStageConfig(const FactoryPPStageConfig &config)
    {
        if (!config.isValid())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid FactoryPPStageConfig: "
                      << "first_layer=" << config.first_layer
                      << ", last_layer=" << config.last_layer);
            throw std::invalid_argument("Invalid FactoryPPStageConfig");
        }

        pp_stage_config_ = config;

        LOG_DEBUG("[DeviceGraphOrchestrator] PP stage configured: "
                  << "layers=[" << config.first_layer << ", " << config.last_layer << ") "
                  << "has_embedding=" << (config.has_embedding ? "yes" : "no")
                  << " has_lm_head=" << (config.has_lm_head ? "yes" : "no"));
    }

    // =========================================================================
    // Hidden State API (for Pipeline Parallelism)
    // =========================================================================

    TensorBase *DeviceGraphOrchestrator::getHiddenState()
    {
        if (!state_.hidden)
        {
            LOG_WARN("[DeviceGraphOrchestrator] getHiddenState: no hidden state available");
            return nullptr;
        }
        return state_.hidden.get();
    }

    const TensorBase *DeviceGraphOrchestrator::getHiddenState() const
    {
        return state_.hidden.get();
    }

    void DeviceGraphOrchestrator::setHiddenState(TensorBase *hidden_state)
    {
        external_hidden_state_input_ = hidden_state;
        LOG_DEBUG("[DeviceGraphOrchestrator] setHiddenState: "
                  << (hidden_state ? "set" : "cleared")
                  << " external hidden state input");
    }

    bool DeviceGraphOrchestrator::hasHiddenStateInput() const
    {
        return external_hidden_state_input_ != nullptr;
    }

    void DeviceGraphOrchestrator::clearHiddenStateInput()
    {
        external_hidden_state_input_ = nullptr;
    }

    const TPDomain *DeviceGraphOrchestrator::getDomainForLayer(int layer_idx, bool is_attention) const
    {
        if (!domain_config_)
        {
            return nullptr; // No domain config - use legacy MPI path
        }

        const TPDomain *domain = domain_config_->domainForLayer(layer_idx, is_attention);

        LOG_DEBUG("[DeviceGraphOrchestrator] getDomainForLayer: layer=" << layer_idx
                                                                        << ", is_attention=" << (is_attention ? "true" : "false")
                                                                        << " -> domain=" << (domain ? domain->name : "nullptr"));

        return domain;
    }

    void DeviceGraphOrchestrator::transitionToPhase(InferencePhase phase)
    {
        if (current_phase_ != phase)
        {
            InferencePhase old_phase = current_phase_;
            current_phase_ = phase;

            LOG_DEBUG("[DeviceGraphOrchestrator] Phase transition: " << toString(old_phase)
                                                                     << " -> " << toString(phase));

            // Notify weight streamer of phase transition (if streaming is enabled)
            if (weight_streamer_)
            {
                weight_streamer_->onPhaseTransition(old_phase, phase);
                LOG_DEBUG("[DeviceGraphOrchestrator] Weight streamer notified of phase transition");
            }
        }
    }

    // =========================================================================
    // Weight Streaming (Option B)
    // =========================================================================

    void DeviceGraphOrchestrator::setWeightStreamer(std::shared_ptr<IWeightStreamer> streamer)
    {
        weight_streamer_ = std::move(streamer);
        if (weight_streamer_)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Weight streaming enabled");
        }
        else
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Weight streaming disabled");
        }
    }

    bool DeviceGraphOrchestrator::isWeightStreamingEnabled() const
    {
        return weight_streamer_ != nullptr;
    }

    void DeviceGraphOrchestrator::setCollectiveContext(std::shared_ptr<ICollectiveContext> collective_ctx)
    {
        injected_collective_ctx_ = std::move(collective_ctx);
        if (injected_collective_ctx_)
        {
            // Wire to executor for GPU-native collective interception
            executor_.setCollectiveContext(injected_collective_ctx_.get());
            LOG_DEBUG("[DeviceGraphOrchestrator] GPU-native collectives enabled via CollectiveContext");
        }
        else
        {
            executor_.setCollectiveContext(nullptr);
            LOG_DEBUG("[DeviceGraphOrchestrator] CollectiveContext cleared - using CPU MPI fallback");
        }
    }

    std::shared_ptr<TensorBase> DeviceGraphOrchestrator::getPhaseAwareWeight(
        const std::string &name,
        int layer_idx,
        InferencePhase phase) const
    {
        if (!weight_manager_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator::getPhaseAwareWeight] WeightManager not set");
            return nullptr;
        }

        // CRITICAL: Use getWeightForDevice() instead of getWeight() to get
        // device-isolated tensor instances. In multi-device (LOCAL TP) scenarios,
        // getWeight() returns the SAME shared tensor to all devices, which causes
        // a race condition: Device 0's ensureOnDevice(rocm:0) allocates GPU memory,
        // then Device 1's ensureOnDevice(rocm:1) frees Device 0's allocation and
        // reallocates on Device 1 — while Device 0's kernels are still using it.
        // getWeightForDevice() returns clones for non-primary devices.
        const DeviceId device = state_.device_id;

        // PREFILL phase: Always use full weight (GPU is primary, compute-bound)
        if (phase == InferencePhase::PREFILL)
        {
            LOG_TRACE("[DeviceGraphOrchestrator::getPhaseAwareWeight] PREFILL phase - returning full weight for " << name
                                                                                                                  << " on " << device.to_string());
            auto weight = weight_manager_->getWeightForDevice(name, device, layer_idx);
            if (!weight)
            {
                LOG_ERROR("[DeviceGraphOrchestrator::getPhaseAwareWeight] Failed to load weight: " << name);
            }
            return weight;
        }

        // DECODE phase: Check if CPU should participate
        if (!shouldUseCPUDecodeWeight(name, layer_idx))
        {
            // No CPU participation - use full weight (GPU handles it)
            LOG_TRACE("[DeviceGraphOrchestrator::getPhaseAwareWeight] DECODE phase, no CPU participation - returning full weight for " << name
                                                                                                                                       << " on " << device.to_string());
            return weight_manager_->getWeightForDevice(name, device, layer_idx);
        }

        // CPU decode participation enabled - get decode shard
        if (!weight_placement_map_)
        {
            // No placement map - fall back to full weight
            LOG_WARN("[DeviceGraphOrchestrator::getPhaseAwareWeight] CPU decode participation but no placement map - using full weight for " << name);
            return weight_manager_->getWeightForDevice(name, device, layer_idx);
        }

        // Get device info from placement map
        WeightDeviceInfo device_info = weight_placement_map_->getDeviceInfoForWeight(name, layer_idx);

        if (!device_info.cpu_decode_participation)
        {
            // This weight doesn't have CPU decode participation
            LOG_TRACE("[DeviceGraphOrchestrator::getPhaseAwareWeight] DECODE phase, weight " << name << " has no CPU participation - returning full weight");
            return weight_manager_->getWeightForDevice(name, device, layer_idx);
        }

        // Find the CPU device in decode_devices and get its fraction
        for (size_t i = 0; i < device_info.decode_devices.size(); ++i)
        {
            if (device_info.decode_devices[i].is_cpu())
            {
                float fraction = device_info.decode_fractions[i];
                LOG_DEBUG("[DeviceGraphOrchestrator::getPhaseAwareWeight] DECODE phase - returning CPU decode shard for "
                          << name << " (fraction=" << fraction << ")");
                return weight_manager_->getDecodeWeight(name, DeviceId::cpu(), fraction, layer_idx);
            }
        }

        // No CPU in decode devices - use full weight
        LOG_TRACE("[DeviceGraphOrchestrator::getPhaseAwareWeight] DECODE phase, CPU not in decode devices - returning full weight for " << name);
        return weight_manager_->getWeightForDevice(name, device, layer_idx);
    }

    bool DeviceGraphOrchestrator::shouldUseCPUDecodeWeight(const std::string &name, int layer_idx) const
    {
        // Check phase constraint:
        // - Default (cpu_prefill_participate=false): CPU only participates in DECODE phase
        // - With cpu_prefill_participate=true: CPU participates in BOTH phases (Option C fallback)
        if (current_phase_ == InferencePhase::PREFILL)
        {
            // Check if CPU prefill participation is enabled (Option C: memory-constrained systems)
            if (!debugEnv().execution.cpu_prefill_participate)
            {
                return false;
            }
            // If cpu_prefill_participate is true, continue with the rest of the checks
        }
        // DECODE phase always allows CPU participation if configured

        // Must have placement map
        if (!weight_placement_map_)
        {
            return false;
        }

        // Get device info
        WeightDeviceInfo device_info = weight_placement_map_->getDeviceInfoForWeight(name, layer_idx);

        // Check if CPU decode participation is enabled for this weight
        if (!device_info.cpu_decode_participation)
        {
            return false;
        }

        // Check if this MPI rank should handle CPU decode
        // For now, rank 0 is the designated CPU decode participant
        // This could be made configurable in the future
        int my_rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;

        // The CPU decode participant is typically the first rank (rank 0)
        // In future, could look at device_info.decode_devices to find which
        // ranks have CPUs and distribute work accordingly
        bool is_cpu_decode_rank = (my_rank == 0);

        if (is_cpu_decode_rank)
        {
            // Verify that CPU is actually in the decode devices
            for (const auto &dev : device_info.decode_devices)
            {
                if (dev.is_cpu())
                {
                    return true;
                }
            }
        }

        return false;
    }

    // =========================================================================
    // Unified Pipeline Configuration (Phase 6)
    // =========================================================================

    void DeviceGraphOrchestrator::setPipelineConfig(std::shared_ptr<PipelineConfig> config)
    {
        if (config)
        {
            std::string validation_error;
            if (!config->validate(&validation_error))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Invalid PipelineConfig: " << validation_error);
                throw std::invalid_argument("Invalid PipelineConfig: " + validation_error);
            }
        }

        pipeline_config_ = std::move(config);

        // Reset initialization flags - contexts need to be recreated
        pp_contexts_initialized_ = false;
        tp_contexts_initialized_ = false;
        pp_contexts_.clear();
        domain_tp_contexts_.clear();

        if (pipeline_config_)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Unified pipeline configured: "
                      << pipeline_config_->numStages() << " PP stages, "
                      << pipeline_config_->tp_domains.size() << " TP domains, "
                      << pipeline_config_->total_layers << " total layers");

            // Propagate to graph builder via setter
            graph_builder_->setPipelineConfig(pipeline_config_);
        }
        else
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] Pipeline configuration cleared (single-device mode)");
            graph_builder_->setPipelineConfig(nullptr);
        }
    }

    void DeviceGraphOrchestrator::setDomainTPContexts(std::map<std::string, std::shared_ptr<ITPContext>> contexts)
    {
        domain_tp_contexts_ = std::move(contexts);
        for (auto &[name, ctx] : domain_tp_contexts_)
        {
            if (ctx)
                graph_builder_->setTPContext(name, ctx.get());
        }
        if (!domain_tp_contexts_.empty())
            tp_contexts_initialized_ = true;
    }

    bool DeviceGraphOrchestrator::initializePPContexts()
    {
        if (pp_contexts_initialized_)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] PP contexts already initialized");
            return true;
        }

        if (!pipeline_config_ || !pipeline_config_->hasPP())
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] No PP configuration - skipping PP context initialization");
            pp_contexts_initialized_ = true;
            return true;
        }

        // Ensure TP contexts are initialized first (needed for PPStage::fromTPContext)
        if (!initializeTPContexts())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to initialize TP contexts - cannot create PP contexts");
            return false;
        }

        // Check if any domain has internal TP (degree > 1)
        // If so, we need to use HierarchicalPPConfig to properly handle TP domains
        const bool has_internal_tp = pipeline_config_->hasTP();

        LOG_DEBUG("[DeviceGraphOrchestrator] Initializing PP contexts for "
                  << (pipeline_config_->numStages() - 1) << " inter-stage transfers"
                  << (has_internal_tp ? " (with TP domains)" : "") << "...");

        // Create PP context for each adjacent pair of stages
        for (int stage = 0; stage < pipeline_config_->numStages() - 1; ++stage)
        {
            int next_stage = stage + 1;
            auto key = std::make_pair(stage, next_stage);

            // Get domains for the two stages
            const auto *domain_from = pipeline_config_->getDomainForStage(stage);
            const auto *domain_to = pipeline_config_->getDomainForStage(next_stage);

            if (!domain_from || !domain_to)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to get domains for PP transfer "
                          << stage << " -> " << next_stage);
                return false;
            }

            // Build layer boundaries for the PP context
            // We need to include all stages up to and including next_stage
            std::vector<int> layer_boundaries;
            for (int s = 0; s <= next_stage; ++s)
            {
                const auto &pp_stage = pipeline_config_->pp_stages[s];
                if (s == 0)
                {
                    layer_boundaries.push_back(pp_stage.first_layer);
                }
                layer_boundaries.push_back(pp_stage.last_layer);
            }

            std::unique_ptr<ILocalPPContext> pp_ctx;

            if (has_internal_tp)
            {
                // Use HierarchicalPPConfig with PPStage variant type
                // This allows PP transfers to understand TP domains
                HierarchicalPPConfig pp_config;
                pp_config.layer_boundaries = layer_boundaries;

                // Build PPStage for each stage 0..next_stage
                for (int s = 0; s <= next_stage; ++s)
                {
                    const auto *domain = pipeline_config_->getDomainForStage(s);
                    if (!domain)
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Missing domain for stage " << s);
                        return false;
                    }

                    if (domain->degree() > 1)
                    {
                        // This stage has internal TP - look up the TP context
                        auto it = domain_tp_contexts_.find(domain->name);
                        if (it == domain_tp_contexts_.end())
                        {
                            LOG_ERROR("[DeviceGraphOrchestrator] TP context not found for domain '"
                                      << domain->name << "' (stage " << s << ")");
                            return false;
                        }
                        const auto &tp_context = it->second;
                        if (!tp_context)
                        {
                            LOG_ERROR("[DeviceGraphOrchestrator] Null TP context for domain '"
                                      << domain->name << "' (stage " << s << ")");
                            return false;
                        }

                        if (tp_context->isLocal())
                        {
                            auto local_context = std::dynamic_pointer_cast<ILocalTPContext>(tp_context);
                            if (!local_context)
                            {
                                LOG_ERROR("[DeviceGraphOrchestrator] TP context for domain '"
                                          << domain->name << "' reports LOCAL but is not an ILocalTPContext");
                                return false;
                            }
                            pp_config.stages.push_back(PPStage::fromTPContext(local_context));
                            LOG_DEBUG("[DeviceGraphOrchestrator] Stage " << s << " → LOCAL TP domain '"
                                                                         << domain->name << "' (" << domain->degree() << " devices)");
                        }
                        else
                        {
                            auto global_context = std::dynamic_pointer_cast<IGlobalTPContext>(tp_context);
                            if (!global_context)
                            {
                                LOG_ERROR("[DeviceGraphOrchestrator] TP context for domain '"
                                          << domain->name << "' is not LOCAL and cannot be used as an IGlobalTPContext");
                                return false;
                            }
                            pp_config.stages.push_back(PPStage::fromGlobalTPContext(global_context));
                            LOG_DEBUG("[DeviceGraphOrchestrator] Stage " << s << " → "
                                                                         << (tp_context->isNodeLocal() ? "NODE_LOCAL" : "GLOBAL")
                                                                         << " TP domain '" << domain->name << "'");
                        }
                    }
                    else
                    {
                        // Single device stage
                        auto device = GlobalDeviceAddress::fromLocalDeviceId(domain->primaryDevice());
                        pp_config.stages.push_back(PPStage::fromDevice(device));
                        LOG_DEBUG("[DeviceGraphOrchestrator] Stage " << s << " → single device "
                                                                     << device.toString());
                    }
                }

                if (!pp_config.isValid())
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Invalid HierarchicalPPConfig for stages "
                              << stage << " -> " << next_stage);
                    return false;
                }

                pp_ctx = createLocalPPContext(pp_config);
            }
            else
            {
                // Use flat LocalPPConfig (simpler, no TP domain awareness needed)
                std::vector<GlobalDeviceAddress> stage_devices;
                for (int s = 0; s <= next_stage; ++s)
                {
                    const auto *domain = pipeline_config_->getDomainForStage(s);
                    if (domain && !domain->devices.empty())
                    {
                        stage_devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(domain->primaryDevice()));
                    }
                }

                LocalPPConfig pp_ctx_config{
                    .stage_devices = stage_devices,
                    .layer_boundaries = layer_boundaries};

                pp_ctx = createLocalPPContext(pp_ctx_config);
            }

            if (!pp_ctx)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create PP context for transfer "
                          << stage << " -> " << next_stage);
                return false;
            }

            pp_contexts_[key] = std::move(pp_ctx);

            LOG_DEBUG("[DeviceGraphOrchestrator] Created PP context: stage " << stage
                                                                             << " (" << domain_from->name << ") -> stage " << next_stage
                                                                             << " (" << domain_to->name << ")");
        }

        // Wire PP contexts to graph builder via setters
        for (auto &[key, ctx] : pp_contexts_)
        {
            graph_builder_->setPPContext(key.first, key.second, ctx.get());
        }

        pp_contexts_initialized_ = true;
        LOG_DEBUG("[DeviceGraphOrchestrator] Initialized " << pp_contexts_.size() << " PP contexts"
                                                           << (has_internal_tp ? " (hierarchical)" : " (flat)"));
        return true;
    }

    bool DeviceGraphOrchestrator::initializeTPContexts()
    {
        if (tp_contexts_initialized_)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] TP contexts already initialized");
            return true;
        }

        if (!pipeline_config_ || pipeline_config_->tp_domains.empty())
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] No TP domains - skipping TP context initialization");
            tp_contexts_initialized_ = true;
            return true;
        }

        LOG_DEBUG("[DeviceGraphOrchestrator] Initializing TP contexts for "
                  << pipeline_config_->tp_domains.size() << " domains...");

        // Create TP context for each domain that has degree > 1
        for (const auto &domain : pipeline_config_->tp_domains)
        {
            if (domain.devices.size() <= 1)
            {
                LOG_DEBUG("[DeviceGraphOrchestrator] Domain '" << domain.name
                                                               << "' has degree " << domain.devices.size() << " - no TP context needed");
                continue;
            }

            // Convert DeviceId to GlobalDeviceAddress
            std::vector<GlobalDeviceAddress> addresses;
            for (const auto &dev : domain.devices)
            {
                addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(dev));
            }

            // Equal weights for now (TODO: support proportional TP)
            std::vector<float> weights(domain.devices.size(), 1.0f / domain.devices.size());

            // Create TP context (convert unique_ptr to shared_ptr so PPStage can reference it)
            auto tp_ctx = createLocalTPContext(addresses, weights, domain.tp_backend);
            if (!tp_ctx)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to create TP context for domain '"
                          << domain.name << "'");
                return false;
            }

            domain_tp_contexts_[domain.name] = std::shared_ptr<ITPContext>(std::move(tp_ctx));

            LOG_DEBUG("[DeviceGraphOrchestrator] Created TP context for domain '" << domain.name
                                                                                  << "': " << domain.devices.size() << " devices, backend="
                                                                                  << static_cast<int>(domain.tp_backend));
        }

        // Wire TP contexts to graph builder via setters
        for (auto &[name, ctx] : domain_tp_contexts_)
        {
            graph_builder_->setTPContext(name, ctx.get());
        }

        tp_contexts_initialized_ = true;
        LOG_DEBUG("[DeviceGraphOrchestrator] Initialized " << domain_tp_contexts_.size() << " TP contexts");
        return true;
    }

    // =========================================================================
    // GraphBuildSession Implementation (nested class)
    // =========================================================================

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::forInput(const ForwardInput &input)
    {
        input_ = input;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withPositionIds(const int *position_ids)
    {
        explicit_position_ids_ = position_ids;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withExternalHiddenState(TensorBase *hidden_state)
    {
        external_hidden_state_ = hidden_state;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withPipelineConfig(std::shared_ptr<PipelineConfig> config)
    {
        pipeline_config_ = std::move(config);
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::forPPStage(int first_layer, int last_layer,
                                                           bool has_embedding, bool has_lm_head)
    {
        pp_stage_ = PPStageSpec{first_layer, last_layer, has_embedding, has_lm_head};
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withPPContext(int from_stage, int to_stage, ILocalPPContext *context)
    {
        pp_contexts_[{from_stage, to_stage}] = context;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withTPContext(const std::string &domain_name, ITPContext *context)
    {
        tp_contexts_[domain_name] = context;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withWeights(const ModelWeights &weights)
    {
        weights_ = weights;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withBuffers(const ModelBuffers &buffers)
    {
        buffers_ = buffers;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildSession &
    DeviceGraphOrchestrator::GraphBuildSession::withKVCache(IKVCache *kv_cache)
    {
        kv_cache_ = kv_cache;
        return *this;
    }

    DeviceGraphOrchestrator::GraphBuildResult
    DeviceGraphOrchestrator::GraphBuildSession::buildForward()
    {
        if (!isValid())
        {
            return GraphBuildResult(validationError());
        }

        applyConfiguration();
        auto prepared_input = prepareInput();

        auto *graph_builder = orchestrator_.graphBuilder();
        if (!graph_builder)
        {
            return GraphBuildResult("No graph builder available");
        }

        ForwardOutput output;
        ComputeGraph graph = graph_builder->buildFullForwardGraph(prepared_input, output);

        if (graph.size() == 0)
        {
            return GraphBuildResult("buildFullForwardGraph returned empty graph");
        }

        LOG_DEBUG("[GraphBuildSession] Built full forward graph with " << graph.size() << " nodes");
        return GraphBuildResult(std::move(graph), output);
    }

    DeviceGraphOrchestrator::GraphBuildResult
    DeviceGraphOrchestrator::GraphBuildSession::buildPartial()
    {
        if (!isValid())
        {
            return GraphBuildResult(validationError());
        }

        if (!pp_stage_.has_value())
        {
            return GraphBuildResult("buildPartial() requires forPPStage() configuration");
        }

        applyConfiguration();
        auto prepared_input = prepareInput();

        auto *graph_builder = orchestrator_.graphBuilder();
        if (!graph_builder)
        {
            return GraphBuildResult("No graph builder available");
        }

        const auto &stage = pp_stage_.value();
        ForwardOutput output;
        ComputeGraph graph = graph_builder->buildPartialForwardGraph(
            prepared_input, output,
            stage.first_layer, stage.last_layer,
            stage.has_embedding, stage.has_lm_head);

        if (graph.size() == 0)
        {
            return GraphBuildResult("buildPartialForwardGraph returned empty graph");
        }

        LOG_DEBUG("[GraphBuildSession] Built partial forward graph: layers=["
                  << stage.first_layer << ", " << stage.last_layer << ") with "
                  << graph.size() << " nodes");
        return GraphBuildResult(std::move(graph), output);
    }

    DeviceGraphOrchestrator::GraphBuildResult
    DeviceGraphOrchestrator::GraphBuildSession::buildUnified()
    {
        if (!isValid())
        {
            return GraphBuildResult(validationError());
        }

        if (!pipeline_config_ || !pipeline_config_->hasPP())
        {
            return GraphBuildResult("buildUnified() requires withPipelineConfig() with hasPP()");
        }

        applyConfiguration();
        auto prepared_input = prepareInput();

        auto *graph_builder = orchestrator_.graphBuilder();
        if (!graph_builder)
        {
            return GraphBuildResult("No graph builder available");
        }

        // Set pipeline config on graph builder
        graph_builder->setPipelineConfig(pipeline_config_);

        // Wire PP contexts
        for (const auto &[key, ctx] : pp_contexts_)
        {
            graph_builder->setPPContext(key.first, key.second, ctx);
        }

        // Wire TP contexts
        for (const auto &[name, ctx] : tp_contexts_)
        {
            graph_builder->setTPContext(name, ctx);
        }

        ForwardOutput output;
        ComputeGraph graph = graph_builder->buildUnifiedPipelineGraph(prepared_input, output);

        if (graph.size() == 0)
        {
            return GraphBuildResult("buildUnifiedPipelineGraph returned empty graph");
        }

        LOG_DEBUG("[GraphBuildSession] Built unified PP graph: "
                  << pipeline_config_->numStages() << " stages, "
                  << graph.size() << " nodes");
        return GraphBuildResult(std::move(graph), output);
    }

    DeviceGraphOrchestrator::GraphBuildResult
    DeviceGraphOrchestrator::GraphBuildSession::build()
    {
        // Auto-select based on configuration
        if (pipeline_config_ && pipeline_config_->hasPP())
        {
            return buildUnified();
        }
        else if (pp_stage_.has_value())
        {
            return buildPartial();
        }
        else
        {
            return buildForward();
        }
    }

    bool DeviceGraphOrchestrator::GraphBuildSession::isValid() const
    {
        return validationError().empty();
    }

    std::string DeviceGraphOrchestrator::GraphBuildSession::validationError() const
    {
        if (!input_.has_value())
        {
            return "No input configured (call forInput())";
        }

        const auto &input = input_.value();

        // Check if this is a PP middle/final stage (no embedding, uses external hidden state)
        bool is_pp_non_embedding_stage = pp_stage_.has_value() && !pp_stage_->has_embedding;
        // Check both session-level and input-level external hidden state
        bool has_external_hidden = external_hidden_state_ != nullptr ||
                                   input.external_hidden_state != nullptr;

        if (input.token_ids == nullptr && input.token_ids_device == nullptr)
        {
            // PP middle/final stages don't need token_ids if they have external hidden state
            if (is_pp_non_embedding_stage && has_external_hidden)
            {
                // Valid: PP stage with external hidden state input
            }
            else if (is_pp_non_embedding_stage)
            {
                return "PP stage without embedding requires external hidden state (call withExternalHiddenState())";
            }
            else
            {
                return "Input token_ids are null";
            }
        }

        if (input.seq_len <= 0)
        {
            return "Invalid sequence length: " + std::to_string(input.seq_len);
        }

        if (input.batch_size <= 0)
        {
            return "Invalid batch size: " + std::to_string(input.batch_size);
        }

        // Unified PP requires pipeline config
        if (pipeline_config_ && pipeline_config_->hasPP())
        {
            // PP contexts should be registered for all stage pairs
            int num_stages = pipeline_config_->numStages();
            for (int s = 0; s < num_stages - 1; ++s)
            {
                auto key = std::make_pair(s, s + 1);
                if (pp_contexts_.find(key) == pp_contexts_.end())
                {
                    return "Missing PP context for stage transfer " + std::to_string(s) +
                           " -> " + std::to_string(s + 1);
                }
            }
        }

        return "";
    }

    ForwardInput DeviceGraphOrchestrator::GraphBuildSession::prepareInput() const
    {
        ForwardInput prepared = input_.value();

        // Override position IDs if explicitly set
        if (explicit_position_ids_)
        {
            prepared.position_ids = explicit_position_ids_;
        }

        // Set external hidden state for PP middle/final stages
        if (external_hidden_state_)
        {
            prepared.external_hidden_state = external_hidden_state_;
        }

        // Set KV cache
        if (kv_cache_)
        {
            prepared.kv_cache = kv_cache_;
        }

        return prepared;
    }

    void DeviceGraphOrchestrator::GraphBuildSession::applyConfiguration()
    {
        auto *graph_builder = orchestrator_.graphBuilder();
        if (!graph_builder)
        {
            return;
        }

        // Apply weights if provided
        if (weights_.has_value())
        {
            graph_builder->setWeights(weights_.value());
        }

        // Apply buffers if provided
        if (buffers_.has_value())
        {
            graph_builder->setBuffers(buffers_.value());
        }
    }

    // =========================================================================
    // AttentionGraphSession Implementation (nested class)
    // =========================================================================

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::forLayer(const LayerWeights &layer, int layer_idx)
    {
        layer_ = &layer;
        layer_idx_ = layer_idx;
        return *this;
    }

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::withBuffers(ActivationBuffers &buffers)
    {
        buffers_ = &buffers;
        return *this;
    }

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::withSequence(int seq_len, int batch_size)
    {
        seq_len_ = seq_len;
        batch_size_ = batch_size;
        return *this;
    }

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::onDevice(DeviceId device)
    {
        device_ = device;
        return *this;
    }

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::withKVCache(IKVCache *kv_cache)
    {
        kv_cache_ = kv_cache;
        return *this;
    }

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::withPositionIds(const int *position_ids)
    {
        position_ids_ = position_ids;
        return *this;
    }

    DeviceGraphOrchestrator::AttentionGraphSession &
    DeviceGraphOrchestrator::AttentionGraphSession::withSequenceLengths(const std::vector<int> *lengths)
    {
        sequence_lengths_ = lengths;
        return *this;
    }

    bool DeviceGraphOrchestrator::AttentionGraphSession::isValid() const
    {
        return validationError().empty();
    }

    std::string DeviceGraphOrchestrator::AttentionGraphSession::validationError() const
    {
        if (!layer_)
        {
            return "No layer weights configured (call forLayer())";
        }
        if (!buffers_)
        {
            return "No buffers configured (call withBuffers())";
        }
        if (layer_idx_ < 0)
        {
            return "Invalid layer index: " + std::to_string(layer_idx_);
        }
        if (seq_len_ <= 0)
        {
            return "Invalid sequence length (call withSequence())";
        }
        if (batch_size_ <= 0)
        {
            return "Invalid batch size: " + std::to_string(batch_size_);
        }
        if (!device_.has_value())
        {
            return "No device configured (call onDevice())";
        }
        return "";
    }

    DeviceGraphOrchestrator::SubGraphBuildResult
    DeviceGraphOrchestrator::AttentionGraphSession::build()
    {
        if (!isValid())
        {
            return SubGraphBuildResult(validationError());
        }

        auto *graph_builder = orchestrator_.graphBuilder();
        if (!graph_builder)
        {
            return SubGraphBuildResult("No graph builder available");
        }

        ComputeGraph graph = graph_builder->buildAttentionGraph(
            *layer_, *buffers_, layer_idx_, seq_len_, batch_size_,
            kv_cache_, position_ids_, device_.value(), sequence_lengths_);

        if (graph.size() == 0)
        {
            return SubGraphBuildResult("buildAttentionGraph returned empty graph");
        }

        LOG_DEBUG("[AttentionGraphSession] Built attention graph for layer "
                  << layer_idx_ << " with " << graph.size() << " nodes");

        return SubGraphBuildResult(std::move(graph));
    }

    // =========================================================================
    // FFNGraphSession Implementation (nested class)
    // =========================================================================

    DeviceGraphOrchestrator::FFNGraphSession &
    DeviceGraphOrchestrator::FFNGraphSession::forLayer(const LayerWeights &layer, int layer_idx)
    {
        layer_ = &layer;
        layer_idx_ = layer_idx;
        return *this;
    }

    DeviceGraphOrchestrator::FFNGraphSession &
    DeviceGraphOrchestrator::FFNGraphSession::withBuffers(ActivationBuffers &buffers)
    {
        buffers_ = &buffers;
        return *this;
    }

    DeviceGraphOrchestrator::FFNGraphSession &
    DeviceGraphOrchestrator::FFNGraphSession::withSequence(int seq_len, int batch_size)
    {
        seq_len_ = seq_len;
        batch_size_ = batch_size;
        return *this;
    }

    DeviceGraphOrchestrator::FFNGraphSession &
    DeviceGraphOrchestrator::FFNGraphSession::onDevice(DeviceId device)
    {
        device_ = device;
        return *this;
    }

    bool DeviceGraphOrchestrator::FFNGraphSession::isValid() const
    {
        return validationError().empty();
    }

    std::string DeviceGraphOrchestrator::FFNGraphSession::validationError() const
    {
        if (!layer_)
        {
            return "No layer weights configured (call forLayer())";
        }
        if (!buffers_)
        {
            return "No buffers configured (call withBuffers())";
        }
        if (layer_idx_ < 0)
        {
            return "Invalid layer index: " + std::to_string(layer_idx_);
        }
        if (seq_len_ <= 0)
        {
            return "Invalid sequence length (call withSequence())";
        }
        if (batch_size_ <= 0)
        {
            return "Invalid batch size: " + std::to_string(batch_size_);
        }
        if (!device_.has_value())
        {
            return "No device configured (call onDevice())";
        }
        return "";
    }

    DeviceGraphOrchestrator::SubGraphBuildResult
    DeviceGraphOrchestrator::FFNGraphSession::build()
    {
        if (!isValid())
        {
            return SubGraphBuildResult(validationError());
        }

        auto *graph_builder = orchestrator_.graphBuilder();
        if (!graph_builder)
        {
            return SubGraphBuildResult("No graph builder available");
        }

        ComputeGraph graph = graph_builder->buildFFNGraph(
            *layer_, *buffers_, layer_idx_, seq_len_, batch_size_, device_.value());

        if (graph.size() == 0)
        {
            return SubGraphBuildResult("buildFFNGraph returned empty graph");
        }

        LOG_DEBUG("[FFNGraphSession] Built FFN graph for layer "
                  << layer_idx_ << " with " << graph.size() << " nodes");

        return SubGraphBuildResult(std::move(graph));
    }

} // namespace llaminar2
