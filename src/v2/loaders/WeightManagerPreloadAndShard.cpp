/**
 * @file WeightManagerPreloadAndShard.cpp
 * @brief WeightManager weight packing, preloading, LOCAL TP sharding, and PP layer filtering.
 */
#include "WeightManagerInternal.h"

namespace llaminar2
{
    // =========================================================================
    // Weight Packing and Preloading (folded from WeightPreloader)
    // =========================================================================

    bool WeightManager::packGemmWeights(
        DeviceId target_device,
        PreloadProgressCallback progress_cb,
        bool release_raw_data,
        std::function<bool(const std::string &)> layer_filter)
    {
        using namespace llaminar::v2::kernels;
        using Clock = std::chrono::high_resolution_clock;

        const bool detail_enabled = WeightLoadingProfiler::isEnabled();
        auto classify_weight = [](const std::string &name) -> std::string
        {
            if (name.find("attn_q.weight") != std::string::npos)
                return "attn_q";
            if (name.find("attn_k.weight") != std::string::npos)
                return "attn_k";
            if (name.find("attn_v.weight") != std::string::npos)
                return "attn_v";
            if (name.find("attn_output.weight") != std::string::npos)
                return "attn_output";
            if (name.find("ffn_gate.weight") != std::string::npos)
                return "ffn_gate";
            if (name.find("ffn_up.weight") != std::string::npos)
                return "ffn_up";
            if (name.find("ffn_down.weight") != std::string::npos)
                return "ffn_down";
            if (name == "output.weight")
                return "lm_head";
            if (name == "token_embd.weight")
                return "embedding";
            return "other";
        };

        std::unordered_map<std::string, double> bucket_ms;
        std::vector<std::pair<std::string, double>> per_weight_ms;
        std::mutex detail_mutex;
        auto pack_loop_start = Clock::now();

        // Collect GEMM tensors under lock, then pack outside lock.
        // This allows overlap with other preload operations (e.g., non-GEMM uploads).
        //
        // For multi-device LOCAL TP: check per_device_cache_ first for device-specific
        // sharded tensors. Fall back to cache_ for single-device scenarios.
        std::vector<std::pair<std::string, std::shared_ptr<TensorBase>>> gemm_weights;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            bool has_per_device_entries = false;

            // Check if we have device-specific entries in per_device_cache_
            for (const auto &[key, tensor] : per_device_cache_)
            {
                if (key.device == target_device)
                {
                    has_per_device_entries = true;
                    break;
                }
            }

            if (has_per_device_entries)
            {
                // LOCAL TP: iterate per_device_cache_ entries for this device
                for (const auto &[key, tensor] : per_device_cache_)
                {
                    if (key.device != target_device)
                        continue;
                    // The typed key retains the canonical source name directly.
                    const std::string &name = key.name;
                    if (isGemmWeight(name) && tensor)
                    {
                        if (layer_filter && !layer_filter(name))
                            continue;
                        gemm_weights.emplace_back(name, tensor);
                    }
                }
            }

            // Fall back to main cache if per_device_cache had no GEMM weights
            // (e.g. only non-GEMM entries exist in per_device_cache for this device)
            if (gemm_weights.empty())
            {
                gemm_weights.reserve(cache_.size());
                for (const auto &[name, tensor] : cache_)
                {
                    if (isGemmWeight(name) && tensor)
                    {
                        if (layer_filter && !layer_filter(name))
                            continue;
                        gemm_weights.emplace_back(name, tensor);
                    }
                }
            }
        }

        if (gemm_weights.empty())
        {
            LOG_DEBUG("[WeightManager] No GEMM weights to pack");
            return true;
        }

        // Apply weight preprocessor (e.g., activation rotation) before packing.
        // This runs once per weight and replaces the cache entry, so that
        // subsequent getWeightForDevice() returns the preprocessed version.
        if (weight_preprocessor_)
        {
            const auto preproc_start = Clock::now();
            size_t preprocessed_count = 0;

            for (auto &[name, tensor] : gemm_weights)
            {
                auto result = weight_preprocessor_(name, tensor);
                if (result && result != tensor)
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    cache_[name] = result;
                    tensor = result;
                    ++preprocessed_count;
                }
            }

            const auto preproc_end = Clock::now();
            const auto preproc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        preproc_end - preproc_start)
                                        .count();

            if (preprocessed_count > 0)
            {
                LOG_DEBUG("[WeightManager] Preprocessed " << preprocessed_count << "/" << gemm_weights.size()
                                                          << " weights in " << preproc_ms << " ms");
            }
        }

        LOG_DEBUG("[WeightManager] Packing " << gemm_weights.size() << " GEMM weights for "
                                             << target_device.to_string());

        const size_t total = gemm_weights.size();
        // This cache-owned public operation prepares the primary model. Other
        // persistent owners enter through the frozen binding overload, which
        // preserves their exact source-plan owner rather than this cache API.
        auto store = preparedWeightStore();

        // CPU-only path: single producer, single consumer.
        // GPU devices use packGemmWeightsViaPipeline() instead.
        constexpr unsigned prepare_workers = 1;
        constexpr unsigned upload_workers = 1;
        constexpr size_t queue_capacity = 4;

        struct PreparedJob
        {
            std::string name;
            std::shared_ptr<TensorBase> tensor;
            std::optional<PreparedWeightRef> prepared;
            bool preparation_ok = false;
        };

        std::deque<PreparedJob> job_queue;
        std::mutex queue_mutex;
        std::condition_variable queue_not_empty;
        std::condition_variable queue_not_full;

        std::atomic<size_t> next_prepare_index{0};
        std::atomic<size_t> completed_count{0};
        std::atomic<bool> producer_done{false};
        std::atomic<bool> all_success{true};
        std::atomic<size_t> local_cpu_packed{0};

        auto prepare_job = [&](const std::pair<std::string, std::shared_ptr<TensorBase>> &entry) -> PreparedJob
        {
            PreparedJob job;
            job.name = entry.first;
            job.tensor = entry.second;

            try
            {
                WeightBinding binding;
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    binding.binding_id = next_pipeline_prepared_binding_id_++;
                    binding.identity = makeSourceWeightIdentity(
                        job.name, store->modelId(), binding.binding_id);
                    if (weight_metadata_)
                    {
                        if (const auto metadata = weight_metadata_->metadata(job.tensor.get()))
                        {
                            binding.identity.derivation = metadata->identity.derivation;
                            binding.identity.role = metadata->identity.role == WeightRole::Other
                                ? inferWeightRole(job.name) : metadata->identity.role;
                            binding.identity.layer = metadata->identity.layer;
                            binding.identity.expert = metadata->identity.expert;
                            binding.slice = metadata->slice;
                            binding.residency = metadata->residency;
                        }
                    }
                }
                binding.tensor = job.tensor.get();
                binding.tensor_owner = job.tensor;
                if (binding.slice.source_rows == 0 && binding.slice.source_cols == 0 &&
                    binding.slice.row_count == 0 && binding.slice.col_count == 0)
                    binding.slice = fullSliceSpec(*job.tensor);
                binding.residency.home_device = target_device;
                binding.residency.resident_device = target_device;
                binding.immutable = true;
                binding.prepared = PreparedWeightRef{
                    store->modelId(), binding.binding_id,
                    PreparedWeightKind::CpuPackedGemm, target_device};
                if (store->adoptPreparedGemmForBinding(binding, target_device))
                    job.prepared = store->preparedRefForBinding(
                        binding.binding_id, target_device, PreparedWeightKind::CpuPackedGemm);
                else
                    job.prepared = store->prepareGemm(
                        binding, PhysicalMemoryOwner::PrimaryModelWeights);
                // The model store retains the same final pack before raw source
                // release. Temporary preparation jobs never own its lifetime.
                job.preparation_ok = job.prepared.has_value();
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[WeightManager] Prepared GEMM creation failed for " << job.name << ": " << e.what());
                job.preparation_ok = false;
            }

            return job;
        };

        auto upload_job = [&](PreparedJob &job) -> bool
        {
            if (!job.preparation_ok || !job.prepared)
            {
                return false;
            }

            try
            {
                auto *kernel = store->gemmKernel(*job.prepared);
                if (!kernel)
                {
                    return false;
                }
                kernel->prepareWeights();
                if (!kernel->weights_converted())
                {
                    LOG_ERROR("[WeightManager] prepareWeights() did not complete upload for: "
                              << job.name << " on " << target_device.to_string());
                    return false;
                }
                return true;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[WeightManager] GEMM upload failed for " << job.name << ": " << e.what());
                return false;
            }
        };

        auto producer_worker = [&]()
        {
            while (true)
            {
                const size_t index = next_prepare_index.fetch_add(1, std::memory_order_relaxed);
                if (index >= total)
                {
                    return;
                }

                const auto prep_start = Clock::now();
                PreparedJob job = prepare_job(gemm_weights[index]);
                const auto prep_end = Clock::now();

                if (detail_enabled)
                {
                    const double prep_ms = std::chrono::duration<double, std::milli>(prep_end - prep_start).count();
                    WeightLoadingProfiler::addDetail("weights.gemm_pack.stage_prepare", prep_ms);
                }

                if (!job.preparation_ok)
                {
                    markPrepState(job.name, target_device, WeightPrepState::FAILED, true, "prepared GEMM creation failed");
                    all_success.store(false, std::memory_order_relaxed);
                    continue;
                }

                std::unique_lock<std::mutex> queue_lock(queue_mutex);
                const auto slot_wait_start = Clock::now();
                queue_not_full.wait(queue_lock, [&]()
                                    { return job_queue.size() < queue_capacity; });
                if (detail_enabled)
                {
                    const auto slot_wait_end = Clock::now();
                    const double slot_wait_ms = std::chrono::duration<double, std::milli>(slot_wait_end - slot_wait_start).count();
                    WeightLoadingProfiler::addDetail("weights.gemm_pack.slot_wait_time", slot_wait_ms);
                }
                job_queue.push_back(std::move(job));
                queue_lock.unlock();
                queue_not_empty.notify_one();
            }
        };

        auto consumer_worker = [&]()
        {
            while (true)
            {
                PreparedJob job;
                {
                    std::unique_lock<std::mutex> queue_lock(queue_mutex);
                    const auto bubble_wait_start = Clock::now();
                    queue_not_empty.wait(queue_lock, [&]()
                                         { return !job_queue.empty() || producer_done.load(std::memory_order_relaxed); });
                    if (detail_enabled)
                    {
                        const auto bubble_wait_end = Clock::now();
                        const double bubble_wait_ms = std::chrono::duration<double, std::milli>(bubble_wait_end - bubble_wait_start).count();
                        WeightLoadingProfiler::addDetail("weights.gemm_pack.pipeline_bubble_time", bubble_wait_ms);
                    }

                    if (job_queue.empty())
                    {
                        return;
                    }

                    job = std::move(job_queue.front());
                    job_queue.pop_front();
                }

                queue_not_full.notify_one();

                const auto upload_start = Clock::now();
                const bool upload_ok = upload_job(job);
                const auto upload_end = Clock::now();

                if (detail_enabled)
                {
                    const double upload_ms = std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
                    {
                        std::lock_guard<std::mutex> detail_lock(detail_mutex);
                        bucket_ms[classify_weight(job.name)] += upload_ms;
                        per_weight_ms.emplace_back(job.name, upload_ms);
                    }
                    WeightLoadingProfiler::addDetail("weights.gemm_pack.stage_upload", upload_ms);
                }

                if (!upload_ok)
                {
                    markPrepState(job.name, target_device, WeightPrepState::FAILED, true, "prepareWeights failed");
                    LOG_ERROR("[WeightManager] Failed to pack weight: " << job.name);
                    all_success.store(false, std::memory_order_relaxed);
                }
                else
                {
                    markPrepState(job.name, target_device, WeightPrepState::PACKED_HOST, true, "GEMM packed on CPU");
                    local_cpu_packed.fetch_add(1, std::memory_order_relaxed);
                    // Only release host data for quantized tensors on CPU.
                    // Quantized GEMM kernels (VNNI) prepack data into their own buffers,
                    // so the original host data is no longer needed.
                    // Floating-point GEMM (oneDNN) reads weight_tensor_->data() live
                    // at every inference call — releasing would cause a null dereference.
                    if (release_raw_data && hasVnniPackedQuantizedPayload(job.tensor.get()))
                    {
                        job.tensor->release_host_weight_data();
                    }

                    markPrepState(job.name, target_device, WeightPrepState::READY, true, "GEMM weight ready");
                    evaluateReclaimEligibility(job.name, true);

                    // Release mmap pages for this weight now that the GEMM engine
                    // has its own copy (VNNI-packed on CPU).
                    // is_mmap_data() checks the tensor's data is actually in an
                    // mmap'd region. is_view() is NOT safe here — TensorSlice
                    // (from TP sharding) returns is_view()=true but has heap data.
                    // MADV_DONTNEED on heap pages zeroes them, causing corruption.
                    if (job.tensor && job.tensor->is_mmap_data())
                    {
                        MmapRegion::adviseDontneedRange(
                            job.tensor->raw_data(), job.tensor->size_bytes());
                    }
                }

                const size_t finished = completed_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if (progress_cb)
                {
                    progress_cb(finished, total, job.name);
                }
            }
        };

        std::vector<std::future<void>> producer_futures;
        producer_futures.reserve(prepare_workers);
        for (unsigned i = 0; i < prepare_workers; ++i)
        {
            producer_futures.emplace_back(std::async(std::launch::async, producer_worker));
        }

        std::vector<std::future<void>> consumer_futures;
        consumer_futures.reserve(upload_workers);
        for (unsigned i = 0; i < upload_workers; ++i)
        {
            consumer_futures.emplace_back(std::async(std::launch::async, consumer_worker));
        }

        for (auto &future : producer_futures)
        {
            future.get();
        }

        producer_done.store(true, std::memory_order_relaxed);
        queue_not_empty.notify_all();

        for (auto &future : consumer_futures)
        {
            future.get();
        }

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            num_cpu_packed_ += local_cpu_packed.load(std::memory_order_relaxed);
        }

        if (detail_enabled)
        {
            const auto pack_loop_end = Clock::now();
            const double total_pack_ms = std::chrono::duration<double, std::milli>(pack_loop_end - pack_loop_start).count();
            WeightLoadingProfiler::addDetail("weights.gemm_pack.loop_total", total_pack_ms);

            {
                std::lock_guard<std::mutex> detail_lock(detail_mutex);
                for (const auto &[bucket, ms] : bucket_ms)
                {
                    WeightLoadingProfiler::addDetail(std::string("weights.gemm_pack.bucket.") + bucket, ms);
                }

                std::sort(per_weight_ms.begin(), per_weight_ms.end(), [](const auto &a, const auto &b)
                          { return a.second > b.second; });

                const size_t top_n = std::min<size_t>(8, per_weight_ms.size());
                for (size_t i = 0; i < top_n; ++i)
                {
                    WeightLoadingProfiler::addDetail(std::string("weights.gemm_pack.weight_top.") + per_weight_ms[i].first,
                                                     per_weight_ms[i].second);
                }
            }
        }

        return all_success.load(std::memory_order_relaxed);
    }

    // -------------------------------------------------------------------------
    // Helper: compute raw byte size for a 2D quantized view whose size_bytes()
    // may report 0 (views alias parent storage without tracking their own size).
    // Duplicated from MoEExpertWeightService.cpp — kept as a static helper here
    // to avoid cross-module dependencies.
    // -------------------------------------------------------------------------
    static size_t quantizedViewRawBytes(const TensorBase &tensor)
    {
        const size_t reported = tensor.size_bytes();
        if (reported > 0)
            return reported;

        const auto &shape = tensor.shape();
        if (shape.size() != 2)
            return 0;

        const size_t rows = shape[0];
        const size_t cols = shape[1];
        auto bytes_for = [rows, cols](size_t block_size, size_t block_bytes) -> size_t
        {
            const size_t blocks_per_row = (cols + block_size - 1) / block_size;
            return rows * blocks_per_row * block_bytes;
        };

        switch (tensor.native_type())
        {
        case TensorType::IQ4_NL:
            return bytes_for(IQ4_NLBlock::BLOCK_SIZE, sizeof(IQ4_NLBlock));
        case TensorType::IQ4_XS:
            return bytes_for(IQ4_XSBlock::BLOCK_SIZE, sizeof(IQ4_XSBlock));
        case TensorType::Q8_0:
            return bytes_for(Q8_0Block::BLOCK_SIZE, sizeof(Q8_0Block));
        case TensorType::Q4_0:
            return bytes_for(Q4_0Block::BLOCK_SIZE, sizeof(Q4_0Block));
        case TensorType::Q4_1:
            return bytes_for(Q4_1Block::BLOCK_SIZE, sizeof(Q4_1Block));
        case TensorType::Q5_0:
            return bytes_for(Q5_0Block::BLOCK_SIZE, sizeof(Q5_0Block));
        case TensorType::Q5_1:
            return bytes_for(Q5_1Block::BLOCK_SIZE, sizeof(Q5_1Block));
        case TensorType::Q2_K:
            return bytes_for(Q2_KBlock::BLOCK_SIZE, sizeof(Q2_KBlock));
        case TensorType::Q3_K:
            return bytes_for(Q3_KBlock::BLOCK_SIZE, sizeof(Q3_KBlock));
        case TensorType::Q4_K:
            return bytes_for(Q4_KBlock::BLOCK_SIZE, sizeof(Q4_KBlock));
        case TensorType::Q5_K:
            return bytes_for(Q5_KBlock::BLOCK_SIZE, sizeof(Q5_KBlock));
        case TensorType::Q6_K:
            return bytes_for(Q6_KBlock::BLOCK_SIZE, sizeof(Q6_KBlock));
        case TensorType::Q8_K:
            return bytes_for(Q8_KBlock::BLOCK_SIZE, sizeof(Q8_KBlock));
        case TensorType::IQ2_XXS:
            return bytes_for(IQ2_XXSBlock::BLOCK_SIZE, sizeof(IQ2_XXSBlock));
        case TensorType::IQ2_XS:
            return bytes_for(IQ2_XSBlock::BLOCK_SIZE, sizeof(IQ2_XSBlock));
        case TensorType::IQ2_S:
            return bytes_for(IQ2_SBlock::BLOCK_SIZE, sizeof(IQ2_SBlock));
        case TensorType::IQ3_XXS:
            return bytes_for(IQ3_XXSBlock::BLOCK_SIZE, sizeof(IQ3_XXSBlock));
        case TensorType::IQ3_S:
            return bytes_for(IQ3_SBlock::BLOCK_SIZE, sizeof(IQ3_SBlock));
        case TensorType::IQ1_S:
            return bytes_for(IQ1_SBlock::BLOCK_SIZE, sizeof(IQ1_SBlock));
        case TensorType::IQ1_M:
            return bytes_for(IQ1_MBlock::BLOCK_SIZE, sizeof(IQ1_MBlock));
        default:
            return 0;
        }
    }

    // -------------------------------------------------------------------------
    // Helper: parse layer index from "blk.N.ffn_gate_exps.weight" etc.
    // Returns -1 on parse failure.
    // -------------------------------------------------------------------------
    static int parseMoELayerIndex(const std::string &name)
    {
        // Expect "blk.N.<suffix>"
        if (name.compare(0, 4, "blk.") != 0)
            return -1;
        auto dot2 = name.find('.', 4);
        if (dot2 == std::string::npos)
            return -1;
        try
        {
            return std::stoi(name.substr(4, dot2 - 4));
        }
        catch (...)
        {
            return -1;
        }
    }

    static const char *moeWeightRoleName(ExpertGemmRegistry::WeightRole role)
    {
        switch (role)
        {
        case ExpertGemmRegistry::WeightRole::GATE:
            return "gate";
        case ExpertGemmRegistry::WeightRole::UP:
            return "up";
        case ExpertGemmRegistry::WeightRole::DOWN:
            return "down";
        default:
            return "unknown";
        }
    }

    static std::string registrySlotComponent(const std::string &value)
    {
        std::string result;
        result.reserve(value.size());
        for (unsigned char ch : value)
            result.push_back(std::isalnum(ch) ? static_cast<char>(std::tolower(ch)) : '_');
        return result.empty() ? std::string("default") : result;
    }

    static bool parseMoEExpertParentName(const std::string &name,
                                         int &layer_idx,
                                         ExpertGemmRegistry::WeightRole &role)
    {
        layer_idx = parseMoELayerIndex(name);
        if (layer_idx < 0)
            return false;

        if (name.find("ffn_gate_exps.weight") != std::string::npos)
        {
            role = ExpertGemmRegistry::WeightRole::GATE;
            return true;
        }
        if (name.find("ffn_up_exps.weight") != std::string::npos)
        {
            role = ExpertGemmRegistry::WeightRole::UP;
            return true;
        }
        if (name.find("ffn_down_exps.weight") != std::string::npos)
        {
            role = ExpertGemmRegistry::WeightRole::DOWN;
            return true;
        }

        return false;
    }

    static std::string moeParentNameForRole(int layer_idx, ExpertGemmRegistry::WeightRole role)
    {
        switch (role)
        {
        case ExpertGemmRegistry::WeightRole::GATE:
            return "blk." + std::to_string(layer_idx) + ".ffn_gate_exps.weight";
        case ExpertGemmRegistry::WeightRole::UP:
            return "blk." + std::to_string(layer_idx) + ".ffn_up_exps.weight";
        case ExpertGemmRegistry::WeightRole::DOWN:
            return "blk." + std::to_string(layer_idx) + ".ffn_down_exps.weight";
        }
        return {};
    }

    static int moeExpertCountFromParentTensor(const TensorBase &tensor)
    {
        const auto &shape = tensor.shape();
        if (shape.size() != 3)
            return 0;
        return static_cast<int>(shape[2]);
    }

    static size_t estimateMoEOverlayRoutedExpertBytesPerExpert(
        const std::unordered_map<std::string, std::shared_ptr<TensorBase>> &cache)
    {
        struct LayerEstimate
        {
            size_t bytes = 0;
            int role_count = 0;
        };

        std::unordered_map<int, LayerEstimate> by_layer;
        for (const auto &[name, tensor] : cache)
        {
            if (!tensor)
                continue;

            int layer_idx = -1;
            ExpertGemmRegistry::WeightRole role = ExpertGemmRegistry::WeightRole::GATE;
            if (!parseMoEExpertParentName(name, layer_idx, role))
                continue;

            const int num_experts = moeExpertCountFromParentTensor(*tensor);
            if (num_experts <= 0)
                continue;

            auto &estimate = by_layer[layer_idx];
            estimate.bytes += tensor->size_bytes() / static_cast<size_t>(num_experts);
            ++estimate.role_count;
        }

        size_t best_partial = 0;
        for (const auto &[_, estimate] : by_layer)
        {
            if (estimate.role_count >= 3)
                return estimate.bytes;
            best_partial = std::max(best_partial, estimate.bytes);
        }
        return best_partial;
    }

    static std::string formatMiB(size_t bytes)
    {
        return std::to_string(bytes / (1024 * 1024)) + " MiB";
    }

    /** @brief Setup-only classification of an original scoped whole-expert bank. */
    enum class MoEOverlayCpuPreparationState : uint8_t { Empty, Complete, Invalid };

    /** @brief Exact original participant/layer request, classified before any preparation mutation. */
    struct MoEOverlayCpuPreparationGroup
    {
        std::string domain_name;
        DeviceId device = DeviceId::cpu();
        int participant_world_rank = -1;
        int participant_index = -1;
        int layer = -1;
        std::vector<int> experts;
        MoEOverlayCpuPreparationState existing = MoEOverlayCpuPreparationState::Empty;
    };

    /**
     * @brief Match the original participant/layer scope before collecting expert IDs.
     * @param group Original domain, allocator, rank-known identity, participant and layer.
     * @param request Original typed preparation request, never inferred from a device ordinal.
     * @return Whether both declarations refer to the same physical participant/layer.
     */
    static bool sameMoEOverlayCpuPreparationGroup(
        const MoEOverlayCpuPreparationGroup &group,
        const MoEExpertOverlayPreparationRequest &request)
    {
        const int request_world_rank = request.participant_world_rank_known
                                           ? request.participant_world_rank
                                           : -1;
        return group.domain_name == request.domain_name &&
               group.device == request.device &&
               group.participant_world_rank == request_world_rank &&
               group.participant_index == request.participant_index &&
               group.layer == request.layer;
    }

    /**
     * @brief Authenticate complete ordinary CPU slots using their current scoped registry owner.
     * @param registry Original model registry, quiescent during preparation or terminal sealed reuse.
     * @param group Exact declared participant/layer and complete requested expert set.
     * @return Empty only when every alias is absent; Complete only for owned matching triplets.
     * @note Terminal residency replacement atomically rebinds both aliases to
     * the current physical slots. Cold Store source ordinals cannot supersede
     * those slots after migration. Fixed joint fragments use their separate
     * original source/cohort contract and never enter this classifier.
     */
    static MoEOverlayCpuPreparationState classifyMoEOverlayCpuPreparationGroup(
        const ExpertGemmRegistry &registry, const MoEOverlayCpuPreparationGroup &group)
    {
        size_t present_roles = 0;
        for (const int expert : group.experts)
            for (const auto role : {ExpertGemmRegistry::WeightRole::GATE,
                                   ExpertGemmRegistry::WeightRole::UP,
                                   ExpertGemmRegistry::WeightRole::DOWN})
            {
                const auto *participant = registry.getEngineForParticipant(group.domain_name, group.device,
                    group.participant_world_rank, group.participant_index, group.layer, expert, role);
                const auto *domain = registry.getEngineForDomain(group.domain_name, group.device, group.layer, expert, role);
                if ((participant == nullptr) != (domain == nullptr) || (participant && participant != domain))
                    return MoEOverlayCpuPreparationState::Invalid;
                if (!participant) continue;
                const auto participant_owner = registry.getEngineLifetimeForParticipant(group.domain_name, group.device,
                    group.participant_world_rank, group.participant_index, group.layer, expert, role);
                const auto domain_owner = registry.getEngineLifetimeForDomain(group.domain_name, group.device,
                    group.layer, expert, role);
                if (!participant_owner || !domain_owner || participant_owner.get() != participant ||
                    domain_owner.get() != domain || participant_owner.owner_before(domain_owner) ||
                    domain_owner.owner_before(participant_owner))
                    return MoEOverlayCpuPreparationState::Invalid;
                ++present_roles;
            }
        if (!present_roles) return MoEOverlayCpuPreparationState::Empty;
        return present_roles == group.experts.size() * 3u ? MoEOverlayCpuPreparationState::Complete :
            MoEOverlayCpuPreparationState::Invalid;
    }

    static std::shared_ptr<ITensorGemm> ownerForExpertEngine(
        const std::vector<std::shared_ptr<ITensorGemm>> &owners,
        ITensorGemm *engine)
    {
        auto it = std::find_if(owners.begin(), owners.end(),
                               [&](const auto &owner)
                               { return owner.get() == engine; });
        return it == owners.end() ? nullptr : *it;
    }

    bool WeightManager::prepareMoEExpertOverlayWeights(
        const MoEExpertOverlayRuntimePlan &runtime_plan,
        DeviceId target_device,
        const FrozenModelWeightSet *frozen_weights,
        const MoEExpertOverlayExecutionPlan *execution_plan,
        PreparedWeightAdmission admission,
        std::shared_ptr<const MoEExpertOverlayPreparationPlan> original_preparation)
    {
        return prepareMoEExpertOverlayWeightsWithScope(runtime_plan, target_device,
            frozen_weights, execution_plan, admission, std::nullopt, std::move(original_preparation));
    }

    bool WeightManager::prepareMoEExpertOverlayWeightsForGraph(
        const MoEExpertOverlayRuntimePlan &runtime_plan,
        const MoEExpertOverlayGraphScope &graph_scope,
        const FrozenModelWeightSet *frozen_weights,
        const MoEExpertOverlayExecutionPlan *execution_plan,
        PreparedWeightAdmission admission,
        std::shared_ptr<const MoEExpertOverlayPreparationPlan> original_preparation)
    {
        graph_scope.validate(runtime_plan, graph_scope.device());
        return prepareMoEExpertOverlayWeightsWithScope(runtime_plan, graph_scope.device(),
            frozen_weights, execution_plan, admission, graph_scope, std::move(original_preparation));
    }

    bool WeightManager::prepareMoEExpertOverlayWeightsWithScope(
        const MoEExpertOverlayRuntimePlan &runtime_plan,
        DeviceId target_device,
        const FrozenModelWeightSet *frozen_weights,
        const MoEExpertOverlayExecutionPlan *execution_plan,
        PreparedWeightAdmission admission,
        const std::optional<MoEExpertOverlayGraphScope> &graph_scope,
        std::shared_ptr<const MoEExpertOverlayPreparationPlan> original_preparation)
    {
        if (!runtime_plan.sourcePlan().usesExpertOverlayAuthority())
            return true;
        if (!target_device.is_valid())
        {
            LOG_ERROR("[WeightManager] MoE overlay preparation requires an explicit participant device");
            return false;
        }

        size_t routed_expert_bytes_per_expert = 0;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            routed_expert_bytes_per_expert = estimateMoEOverlayRoutedExpertBytesPerExpert(cache_);
        }

        auto preparation_plan = MoEExpertOverlayPreparationPlan::forOriginalSource(
            runtime_plan, loader_, std::move(original_preparation), routed_expert_bytes_per_expert);
        return prepareMoEExpertOverlayWeightsFromPlan(runtime_plan, std::move(preparation_plan),
            target_device, frozen_weights, execution_plan, admission, graph_scope);
    }

    bool WeightManager::prepareMoEExpertOverlayWeightsFromPlan(
        const MoEExpertOverlayRuntimePlan &runtime_plan,
        MoEExpertOverlayPreparationPlan preparation_plan,
        DeviceId target_device,
        const FrozenModelWeightSet *frozen_weights,
        const MoEExpertOverlayExecutionPlan *execution_plan,
        PreparedWeightAdmission admission,
        const std::optional<MoEExpertOverlayGraphScope> &graph_scope)
    {
        if (execution_plan)
            preparation_plan = preparation_plan.filteredForRank(execution_plan->currentRankPlan());
        const bool owns_colocated_cpu = execution_plan && target_device.is_gpu() &&
            execution_plan->currentRankPlan().hasRole(OverlayRankRole::ContinuationRoot) &&
            target_device == runtime_plan.continuationDevice();
        if (graph_scope)
        {
            graph_scope->validate(runtime_plan, target_device);
            preparation_plan = preparation_plan.filteredForGraphScope(runtime_plan, *graph_scope);
        }
        else
        {
            // The existing device-wide preparation API intentionally owns every
            // matching bank in its complete frozen authority. Graph factories
            // always supply exact logical scope before entering this method.
            std::vector<DeviceId> graph_execution_devices{target_device};
            if (owns_colocated_cpu)
                graph_execution_devices.push_back(DeviceId::cpu());
            preparation_plan = preparation_plan.filteredForDevices(graph_execution_devices);
        }

        if (admission ==
            PreparedWeightAdmission::ReuseCertifiedCompleteSet)
        {
            /*
             * A reusable ModelContext has already crossed the typed
             * Sealing -> Reusable -> RunnerExclusive boundary.  Its terminal
             * physical seal atomically rebound both participant and domain
             * registry keys to the canonical prepared slots.  The 3-D GGUF
             * selections are no longer an authority here: a migrated logical
             * expert may occupy a different loader-era slot, and rebuilding a
             * tensor selection would both copy gigabytes and reinterpret that
             * physical identity incorrectly.
             */
            if (!lifecycle_gates_.device_preparation_complete ||
                !lifecycle_gates_.graph_materialization_complete)
            {
                LOG_ERROR(
                    "[WeightManager] Certified ExpertOverlay reuse reached "
                    "weight preparation before the model-owned preparation "
                    "and graph lifecycle gates completed");
                return false;
            }

            if (frozen_weights)
            {
                const bool contains_routed_source = std::any_of(
                    frozen_weights->bindings().begin(),
                    frozen_weights->bindings().end(),
                    [](const WeightBinding &binding)
                    { return isRoutedExpertRole(binding.identity.role); });
                if (contains_routed_source)
                {
                    LOG_ERROR(
                        "[WeightManager] Certified ExpertOverlay reuse received "
                        "a frozen routed-expert source binding; retained engines "
                        "must be the sole weight authority");
                    return false;
                }
            }

            size_t adopted_requests = 0u;
            for (const auto &request : preparation_plan.requests())
            {
                const int participant_world_rank =
                    request.participant_world_rank_known
                        ? request.participant_world_rank
                        : -1;
                const auto participant_engine =
                    expert_gemm_registry_.getEngineLifetimeForParticipant(
                        request.domain_name,
                        request.device,
                        participant_world_rank,
                        request.participant_index,
                        request.layer,
                        request.expert_id,
                        request.role, request.projection_ownership);
                const auto domain_engine =
                    expert_gemm_registry_.getEngineLifetimeForDomain(
                        request.domain_name,
                        request.device,
                        request.layer,
                        request.expert_id,
                        request.role, request.projection_ownership);
                const bool same_owner =
                    participant_engine && domain_engine &&
                    !participant_engine.owner_before(domain_engine) &&
                    !domain_engine.owner_before(participant_engine);
                if (!same_owner ||
                    participant_engine.get() != domain_engine.get())
                {
                    LOG_ERROR(
                        "[WeightManager] Certified ExpertOverlay reuse has a "
                        "missing or torn prepared registry entry for domain="
                        << request.domain_name
                        << " device=" << request.device.to_string()
                        << " participant_world_rank="
                        << participant_world_rank
                        << " participant=" << request.participant_index
                        << " layer=" << request.layer
                        << " expert=" << request.expert_id
                        << " role=" << moeWeightRoleName(request.role));
                    return false;
                }
                ++adopted_requests;
            }

            PerfStatsCollector::addCounter(
                "weight_loading",
                "overlay_certified_prepared_registry_reuses",
                1.0,
                "load",
                target_device.to_string(),
                {{"requests", std::to_string(adopted_requests)},
                 {"source_materialization", "false"}});
            LOG_DEBUG(
                "[WeightManager] Adopted " << adopted_requests
                                             << " certified ExpertOverlay "
                                                "prepared registry entries for "
                                             << target_device.to_string()
                                             << " without source materialization");
            return true;
        }

        /**
         * Accelerator expert preparation must consume the same immutable bindings
         * that the graph will later resolve. Falling back to WeightManager's mutable
         * legacy caches here is incorrect for LocalTP replication: those caches may
         * contain a participant slice while the graph owns a complete replicated
         * expert tensor. That mismatch previously let preparation report success and
         * left the graph without any engines under its binding/device identity.
         *
         * CPU tiers materialize their original source plan through the common
         * bounded CPU consumer below. CUDA/ROCm retain the runner-owned frozen
         * source set consumed by their existing asynchronous upload pipeline.
         */
        if (preparation_plan.hasAcceleratorRequests() && !frozen_weights)
        {
            LOG_ERROR("[WeightManager] MoE overlay accelerator preparation requires "
                      "the graph-frozen weight bindings; cache-backed preparation is forbidden");
            return false;
        }

        LOG_DEBUG("[WeightManager] participant-local overlay preparation device="
                  << target_device.to_string() << " " << preparation_plan.diagnostics().render());

        std::vector<MoEOverlayCpuPreparationGroup> groups;
        for (const auto &request : preparation_plan.requests())
        {
            if (!request.device.is_cpu() || request.joint_source ||
                request.role != ExpertGemmRegistry::WeightRole::GATE)
            {
                continue;
            }

            auto it = std::find_if(groups.begin(), groups.end(),
                                   [&](const auto &group)
                                   { return sameMoEOverlayCpuPreparationGroup(group, request); });
            if (it == groups.end())
            {
                MoEOverlayCpuPreparationGroup group;
                group.domain_name = request.domain_name;
                group.device = request.device;
                group.participant_world_rank = request.participant_world_rank_known
                                                   ? request.participant_world_rank
                                                   : -1;
                group.participant_index = request.participant_index;
                group.layer = request.layer;
                group.experts.push_back(request.expert_id);
                groups.push_back(std::move(group));
            }
            else if (std::find(it->experts.begin(), it->experts.end(), request.expert_id) == it->experts.end())
            {
                it->experts.push_back(request.expert_id);
            }
        }

        // Classify every included ordinary group before reading sources or
        // mutating any bank. A torn sibling cannot be repaired as a side effect
        // of preparing another empty participant on the same CPU allocator.
        for (auto &group : groups) {
            std::sort(group.experts.begin(), group.experts.end());
            group.existing = classifyMoEOverlayCpuPreparationGroup(expert_gemm_registry_, group);
            if (group.existing == MoEOverlayCpuPreparationState::Invalid) {
                LOG_ERROR("[WeightManager] MoE overlay CPU expert registry is partial, unowned or disagrees between participant and domain scopes for domain "
                          << group.domain_name << " layer " << group.layer << " participant=" << group.participant_index);
                return false;
            }
        }
        const bool has_joint_cpu = std::any_of(preparation_plan.requests().begin(), preparation_plan.requests().end(),
            [](const auto &request) { return request.device.is_cpu() && bool(request.joint_source); });
        const bool has_empty_cpu = std::any_of(groups.begin(), groups.end(), [](const auto &group) {
            return group.existing == MoEOverlayCpuPreparationState::Empty;
        });

        bool ok = true;

        // Preserve the existing accelerator pipeline ordering for mixed
        // preparation. The preceding CPU classification performs no source I/O.
        if (!preparation_plan.empty() && preparation_plan.hasAcceleratorRequests())
        {
            if (!target_device.is_gpu())
            {
                LOG_ERROR("[WeightManager] participant-local overlay plan retained accelerator requests for non-GPU target "
                          << target_device.to_string());
                return false;
            }

            auto layer_role_filter = [&preparation_plan, target_device](const std::string &name) -> bool
            {
                int layer_idx = -1;
                ExpertGemmRegistry::WeightRole role = ExpertGemmRegistry::WeightRole::GATE;
                if (!parseMoEExpertParentName(name, layer_idx, role))
                    return false;
                return preparation_plan.hasAnyRequestForDeviceLayerRole(
                    target_device, layer_idx, role);
            };

            LOG_DEBUG("[WeightManager] Preparing participant-local MoE overlay experts for "
                      << target_device.to_string());
            ok = packGemmWeightsViaPipeline(
                target_device,
                layer_role_filter,
                frozen_weights,
                true,
                &preparation_plan);
        }

        std::optional<FrozenModelWeightSet> original_cpu_sources;
        if (has_joint_cpu || has_empty_cpu) {
            const auto model_id = frozen_weights ? frozen_weights->strategy().model_id : preparedWeightStore()->modelId();
            if (!model_id.value)
                throw std::invalid_argument("CPU source preparation requires its original model identity");
            const auto declared = preparation_plan.sourceWeightPlan(loader_,model_id);
            WeightPlan cpu_sources(declared.strategy(),declared.physicalMemoryOwner());
            std::vector<unsigned> selected_roles(groups.size(), 0u);
            for (const auto &requirement : declared.requirements()) {
                if (!requirement.target_device.is_cpu()) continue;
                if (requirement.joint_source) {
                    cpu_sources.add(requirement);
                    continue;
                }
                const auto group = std::find_if(groups.begin(), groups.end(), [&](const auto &candidate) {
                    return candidate.domain_name == requirement.overlay_domain && candidate.device == requirement.target_device &&
                        candidate.participant_world_rank == requirement.overlay_participant_world_rank &&
                        candidate.participant_index == requirement.overlay_participant_index && candidate.layer == requirement.layer;
                });
                if (group == groups.end())
                    throw std::invalid_argument("CPU source requirement has no original participant/layer group");
                if (group->existing == MoEOverlayCpuPreparationState::Complete) continue;
                if (requirement.slice.expert_ids != group->experts)
                    throw std::invalid_argument("CPU source requirement changed its original empty-group expert selection");
                const unsigned role = requirement.role == WeightRole::MoEExpertGate ? 1u :
                    requirement.role == WeightRole::MoEExpertUp ? 2u : requirement.role == WeightRole::MoEExpertDown ? 4u : 0u;
                auto &mask = selected_roles[size_t(group - groups.begin())];
                if (!role || (mask & role))
                    throw std::invalid_argument("CPU source requirement has a missing or duplicate original role");
                mask |= role;
                cpu_sources.add(requirement);
            }
            for (size_t index = 0; index < groups.size(); ++index)
                if (groups[index].existing == MoEOverlayCpuPreparationState::Empty && selected_roles[index] != 7u)
                    throw std::invalid_argument("CPU source plan omitted an original empty-group projection role");
            // Only cold empty groups enter the common bounded source consumer.
            // Complete ordinary banks retain their current physical registry;
            // joint fragments still validate the same original source/cohort.
            original_cpu_sources.emplace(materialize(cpu_sources));
        }


        if (preparation_plan.hasCpuRoutedAssignments())
        {
            const bool has_joint=std::any_of(preparation_plan.requests().begin(),preparation_plan.requests().end(),
                [](const auto& request){return request.device.is_cpu() && bool(request.joint_source);});
            if(has_joint) {
                if(!original_cpu_sources)throw std::invalid_argument("CPU projection preparation requires original frozen source bindings");
                const auto matrices=original_cpu_sources->projectedExpertMatrices();
                std::vector<WeightBinding> joint_sources;
                for(const auto& source:matrices.bindings()) {
                    if(!source.joint_source)continue;
                    if(*preparation_plan.shardSourceFor(source.identity.layer,source.joint_source->participant())!=*source.joint_source)
                        throw std::invalid_argument("CPU joint preparation changed its frozen source selection");
                    joint_sources.push_back(source);
                }
                // A runner's aggregate frozen set also owns primary model
                // bindings. Only these exact source-bound overlay fragments
                // consume the routed-weight admission, just as the accelerator
                // overlay pipeline does. Never relabel unrelated dense weights.
                const FrozenModelWeightSet routed_sources(matrices.strategy(),
                    std::move(joint_sources), PhysicalMemoryOwner::RoutedExpertWeights);
                if(!prepareWeightsForDevice(routed_sources,DeviceId::cpu(),true))return false;
            }
            for (const auto &group : groups) {
                if (group.existing == MoEOverlayCpuPreparationState::Complete) {
                    LOG_DEBUG("[WeightManager] Reusing sealed MoE overlay CPU expert bank for domain "
                              << group.domain_name << " layer " << group.layer << " participant=" << group.participant_index
                              << " experts=" << group.experts.size());
                    continue;
                }
                if (classifyMoEOverlayCpuPreparationGroup(expert_gemm_registry_, group) !=
                    MoEOverlayCpuPreparationState::Complete)
                    throw std::logic_error("CPU empty-group preparation did not publish its complete original scoped bank");
            }
        }

        if (ok && frozen_weights)
        {
            // Frozen expert selections bypass cache_, so the legacy host-release
            // sweep cannot see their owned storage. Finalized participant engines
            // now own all execution/migration bytes; retire the GPU-only sources
            // here, before graph/prefix allocation increases setup's host peak.
            const size_t retired = retirePreparedOverlaySources(
                *frozen_weights, preparation_plan, expert_gemm_registry_,
                *weight_metadata_);
            LOG_DEBUG("[WeightManager] Retired " << retired
                      << " frozen accelerator expert source bytes after preparation on "
                      << target_device.to_string());
        }
        return ok;
    }

    bool WeightManager::packGemmWeightsViaPipeline(
        DeviceId target_device,
        std::function<bool(const std::string &)> layer_filter,
        const FrozenModelWeightSet *frozen_weights,
        bool include_expert_jobs,
        const MoEExpertOverlayPreparationPlan *overlay_preparation_plan,
        std::optional<size_t> staging_budget_bytes_override,
        PhysicalMemoryOwner persistent_owner)
    {
        if (frozen_weights)
            validateStaticProjectionPreparation(*frozen_weights, target_device, include_expert_jobs);
        using namespace llaminar::v2::kernels;
        using Clock = std::chrono::high_resolution_clock;
        const auto start = Clock::now();

        if (!target_device.is_gpu())
        {
            LOG_ERROR("[WeightManager] GPU pipeline only supports GPU devices");
            return false;
        }

        // ------------------------------------------------------------------
        // Step 1: Collect GEMM weights (same logic as packGemmWeights)
        // ------------------------------------------------------------------
        struct DenseGemmJob
        {
            std::string name;
            TensorBase *tensor = nullptr;
            std::shared_ptr<TensorBase> owner;
            std::optional<WeightBinding> binding;
            FloatingActivationOperandPrecision operand_precision = FloatingActivationOperandPrecision::Native;
        };

        std::vector<DenseGemmJob> gemm_weights;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            if (frozen_weights)
            {
                for (const auto &binding : frozen_weights->bindings())
                {
                    if (binding.projection_ownership)
                        continue; // Static matrices enter only participant-scoped expert jobs below.
                    if (!shouldPrepareFrozenGemmBinding(*this, binding, target_device, include_expert_jobs))
                    {
                        if (bindingTargetsDevice(binding, target_device) &&
                            !include_expert_jobs &&
                            isRoutedExpertBinding(binding))
                        {
                            LOG_TRACE("[WeightManager] GPU pipeline: deferring routed expert binding "
                                      << binding.identity.canonical_name
                                      << " to explicit MoE expert preparation");
                        }
                        continue;
                    }
                    if (layer_filter && !layer_filter(binding.identity.canonical_name))
                        continue;

                    gemm_weights.push_back(DenseGemmJob{
                        binding.identity.canonical_name,
                        binding.tensor,
                        nullptr,
                        binding});
                }
            }
            else
            {
                bool has_per_device_entries = false;
                std::unordered_set<std::string> collected_names;

                for (const auto &[key, tensor] : per_device_cache_)
                {
                    if (key.device == target_device)
                    {
                        has_per_device_entries = true;
                        break;
                    }
                }

                if (has_per_device_entries)
                {
                    for (const auto &[key, tensor] : per_device_cache_)
                    {
                        if (key.device != target_device)
                            continue;
                        const std::string &name = key.name;
                        if (isGemmWeight(name) && tensor)
                        {
                            if (layer_filter && !layer_filter(name))
                                continue;
                            gemm_weights.push_back(DenseGemmJob{name, tensor.get(), tensor, std::nullopt});
                            collected_names.insert(name);
                        }
                    }
                }

                // Supplement from the main cache for weights that do not have a
                // device-specific clone. Hybrid PP/TP can populate per_device_cache_
                // partially; treating any per-device entry as complete skips dense
                // PP-stage weights that still live in cache_.
                {
                    for (const auto &[name, tensor] : cache_)
                    {
                        if (collected_names.count(name) != 0)
                            continue;
                        if (isGemmWeight(name) && tensor)
                        {
                            if (layer_filter && !layer_filter(name))
                                continue;
                            gemm_weights.push_back(DenseGemmJob{name, tensor.get(), tensor, std::nullopt});
                            collected_names.insert(name);
                        }
                    }
                }

                // Tied LM head: if this stage owns lm_head (layer_filter passes
                // output.weight) but output.weight wasn't collected, the model
                // ties token_embd.weight as the LM projection weight. Include
                // it in the pipeline so it gets GPU-repacked with the rest of the
                // resident GEMM weights.
                if (!layer_filter || layer_filter("output.weight"))
                {
                    bool has_output_weight = false;
                    for (const auto &job : gemm_weights)
                    {
                        if (job.name == "output.weight")
                        {
                            has_output_weight = true;
                            break;
                        }
                    }
                    if (!has_output_weight)
                    {
                        std::shared_ptr<TensorBase> embed_tensor;
                        const auto device_key = DeviceWeightCacheKey::forDevice(target_device, "token_embd.weight");
                        auto pdit = per_device_cache_.find(device_key);
                        if (pdit != per_device_cache_.end() && pdit->second)
                            embed_tensor = pdit->second;
                        else
                        {
                            auto it = cache_.find("token_embd.weight");
                            if (it != cache_.end() && it->second)
                                embed_tensor = it->second;
                        }
                        if (embed_tensor)
                        {
                            // The tensor data comes from token_embd.weight, but the
                            // graph owns and resolves the LM projection through the
                            // canonical output.weight binding. Registering the
                            // prepared handle under output.weight keeps tied-embedding
                            // models on the same binding-first path as untied models.
                            gemm_weights.push_back(DenseGemmJob{"output.weight", embed_tensor.get(), embed_tensor, std::nullopt});
                            collected_names.insert("output.weight");
                            LOG_DEBUG("[WeightManager] GPU pipeline: including tied token_embd.weight as canonical output.weight LM head GEMM");
                        }
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Step 1b: Collect MoE expert weights (3D _exps.weight tensors)
        //
        // These are filtered OUT by isGemmWeight() because isNonGemmWeight()
        // returns true for _exps.weight. We scan for them separately, create
        // 2D views for each expert, and include them in the same orchestrator.
        // ------------------------------------------------------------------
        struct MoEExpertJob
        {
            int layer_idx;
            int expert_idx;
            ExpertGemmRegistry::WeightRole role;
            std::string slot_name; // e.g., "moe_L0_gate_e5"
            std::string domain_name;
            std::string tier_name;
            int tier_index = -1;
            int participant_world_rank = -1;
            int participant_index = -1;
            /** Stable 3D tensor identity used to coalesce adjacent expert views. */
            const TensorBase *parent_identity = nullptr;
            /** Optional explicit owner retained by graph-frozen bindings. */
            std::shared_ptr<TensorBase> parent_owner;
            /** 2D expert view; every tensor implementation retains its root parent. */
            std::shared_ptr<TensorBase> view;
            /** Complete source/partition identity, never inferred from the smaller physical N. */
            std::optional<MoEExpertProjectionOwnership> projection_ownership;
            std::optional<WeightBinding> static_source = std::nullopt; ///< Exact marked view; no domain/device alias is published.
            GPUPreparedWeightPoolLayout pool_layout = GPUPreparedWeightPoolLayout::SourceNative; ///< Same physical stride policy for planning, repacking and engine binding.
        };
        std::vector<MoEExpertJob> moe_jobs;
        size_t moe_jobs_already_satisfied = 0;

        // Track 3D parent tensors by name so release readiness is tensor-specific.
        struct MoEParentTensorRecord
        {
            std::string name;
            std::shared_ptr<TensorBase> parent;
            std::vector<int> expected_experts;
        };
        std::vector<MoEParentTensorRecord> moe_parent_tensors;

        // Static F shards are already exact original 2D views. The frozen
        // declarations supply scope and complete partition; no overlay request
        // or quotient over E/P is manufactured. Validate all 3E before planning
        // any physical storage or starting an upload.
        struct StaticLayerSources
        {
            ExpertGemmRegistry::ParticipantLayerScope scope;
            MoEExpertProjectionOwnership projection;
            std::vector<WeightBinding> sources;
            GPUPreparedWeightPoolLayout pool_layout;
        };
        std::vector<StaticLayerSources> static_layers;
        // Index the immutable declarations once. A linear request scan for
        // every matrix would make full-model fragment preparation quadratic.
        using StaticRequestKey = std::tuple<std::string, DeviceId, int, int, int, int,
            ExpertGemmRegistry::WeightRole>;
        std::map<StaticRequestKey, const MoEExpertOverlayPreparationRequest *> unmatched_static_requests;
        bool ordinary_overlay_requests = false;
        if (include_expert_jobs && overlay_preparation_plan)
            for (const auto &request : overlay_preparation_plan->requests())
                if (request.device == target_device)
                {
                    if (request.projection_ownership && !request.projection_ownership->hasMovableProjectionSet())
                    {
                        const StaticRequestKey key{request.domain_name, request.device, request.participant_world_rank,
                            request.participant_index, request.layer, request.expert_id, request.role};
                        if (!unmatched_static_requests.emplace(key, &request).second)
                            throw std::invalid_argument("Static preparation repeats an original fragment request");
                    }
                    else
                        ordinary_overlay_requests = true;
                }
        if (frozen_weights && include_expert_jobs)
        {
            for (const auto &binding : frozen_weights->bindings())
            {
                if (!binding.projection_ownership || !bindingTargetsDevice(binding, target_device)) continue;
                const auto scope = ExpertGemmRegistry::staticProjectionScope(binding);
                auto pool_layout = GPUPreparedWeightPoolLayout::SourceNative;
                if (overlay_preparation_plan)
                {
                    const auto role = binding.identity.role == WeightRole::MoEExpertGate
                        ? ExpertGemmRegistry::WeightRole::GATE
                        : binding.identity.role == WeightRole::MoEExpertUp
                            ? ExpertGemmRegistry::WeightRole::UP : ExpertGemmRegistry::WeightRole::DOWN;
                    if (binding.identity.role != WeightRole::MoEExpertGate &&
                        binding.identity.role != WeightRole::MoEExpertUp &&
                        binding.identity.role != WeightRole::MoEExpertDown)
                        throw std::invalid_argument("Static source has no routed expert projection role");
                    const StaticRequestKey key{scope.domain_name, scope.device, scope.participant_world_rank,
                        scope.participant_index, scope.layer, binding.identity.expert, role};
                    const auto found = unmatched_static_requests.find(key);
                    const auto *request = found == unmatched_static_requests.end() ? nullptr : found->second;
                    if (!request || request->projection_ownership != binding.projection_ownership ||
                        bool(request->joint_source) != bool(binding.joint_source) ||
                        (binding.joint_source &&
                         *request->joint_source != *binding.joint_source))
                        throw std::invalid_argument("Static source lacks its exact original fragment preparation request");
                    pool_layout = request->gpuPoolLayout();
                    unmatched_static_requests.erase(found);
                }
                if (layer_filter && !layer_filter(binding.identity.canonical_name))
                    throw std::invalid_argument("Static source declarations must be filtered as complete participant layers");
                auto found = std::find_if(static_layers.begin(), static_layers.end(), [&](const auto &layer) {
                    return layer.scope == scope;
                });
                if (found == static_layers.end())
                {
                    static_layers.push_back({scope, *binding.projection_ownership, {}, pool_layout});
                    found = static_layers.end() - 1;
                }
                if (found->pool_layout != pool_layout)
                    throw std::invalid_argument("One original expert fragment bank declares conflicting pool layouts");
                found->sources.push_back(binding);
            }
            if (!unmatched_static_requests.empty())
                throw std::invalid_argument("Static preparation omitted an original fragment request");
            for (const auto &layer : static_layers)
            {
                const auto existing = expert_gemm_registry_.acquireStaticProjectionLayer(
                    layer.scope, layer.projection, layer.sources);
                if (existing)
                {
                    moe_jobs_already_satisfied += layer.sources.size();
                    continue;
                }
                for (const auto &source : layer.sources)
                {
                    const auto role = source.identity.role == WeightRole::MoEExpertGate ? ExpertGemmRegistry::WeightRole::GATE
                        : source.identity.role == WeightRole::MoEExpertUp ? ExpertGemmRegistry::WeightRole::UP : ExpertGemmRegistry::WeightRole::DOWN;
                    const auto slot = "static_" + registrySlotComponent(layer.scope.domain_name) + "_rank" +
                        std::to_string(layer.scope.participant_world_rank) + "_p" + std::to_string(layer.scope.participant_index) +
                        "_binding" + std::to_string(source.binding_id);
                    moe_jobs.push_back({layer.scope.layer, source.identity.expert, role, slot,
                        layer.scope.domain_name, {}, -1, layer.scope.participant_world_rank, layer.scope.participant_index,
                        source.projection_source_owner.get(), source.projection_source_owner,
                        source.tensor_owner, source.projection_ownership, source, layer.pool_layout});
                }
            }
        }

        if (include_expert_jobs)
        {
            // Group 3D expert tensors by layer: layer_idx -> {gate, up, down}
            struct MoERoleTensor
            {
                std::shared_ptr<TensorBase> owner;
                TensorBase *tensor = nullptr;
                std::string name;
                std::optional<MoEExpertSourceView> selection;
            };
            struct MoELayerTensors
            {
                MoERoleTensor gate;
                MoERoleTensor up;
                MoERoleTensor down;
            };
            std::unordered_map<int, MoELayerTensors> moe_layers;

            auto add_moe_parent = [&](const std::string &name,
                                      std::shared_ptr<TensorBase> owner,
                                      TensorBase *tensor,
                                      const WeightSliceSpec &slice)
            {
                if (!tensor || name.find("_exps.weight") == std::string::npos)
                    return;
                if (layer_filter && !layer_filter(name))
                    return;

                int layer_idx = parseMoELayerIndex(name);
                if (layer_idx < 0)
                    return;

                const auto &shape = tensor->shape();
                if (shape.size() != 3 || shape[2] == 0)
                    throw std::invalid_argument(
                        "GPU MoE preparation requires a nonempty GGUF [K,N,E] source parent: " + name);

                MoERoleTensor source;
                source.owner = std::move(owner);
                source.tensor = tensor;
                source.name = name;
                // Validate the two coordinate systems once: global router IDs
                // versus physical slots in this exact frozen source parent.
                source.selection.emplace(shape, slice);

                if (name.find("ffn_gate_exps.weight") != std::string::npos)
                {
                    moe_layers[layer_idx].gate = std::move(source);
                }
                else if (name.find("ffn_up_exps.weight") != std::string::npos)
                {
                    moe_layers[layer_idx].up = std::move(source);
                }
                else if (name.find("ffn_down_exps.weight") != std::string::npos)
                {
                    moe_layers[layer_idx].down = std::move(source);
                }
            };

            bool collected_from_frozen = false;
            if (frozen_weights)
            {
                for (const auto &binding : frozen_weights->bindings())
                {
                    if (!binding.tensor || binding.projection_ownership)
                        continue; // Static matrices were consumed above, never reinterpreted as 3D parents.
                    const bool resident_on_target =
                        binding.prepared.has_value()
                            ? binding.prepared->device == target_device
                            : (binding.residency.resident_device.has_value()
                                   ? *binding.residency.resident_device == target_device
                                   : binding.residency.home_device == target_device);
                    if (!resident_on_target)
                        continue;
                    if (isRoutedExpertRole(binding.identity.role) && std::any_of(static_layers.begin(), static_layers.end(), [&](const auto &layer) {
                            return layer.scope.layer == binding.identity.layer;
                        }))
                        throw std::invalid_argument("Static participant source cannot be mixed with ordinary/overlay expert parents in one layer");

                    add_moe_parent(binding.identity.canonical_name,
                                   binding.tensor_owner,
                                   binding.tensor,
                                   binding.slice);
                    if (binding.identity.canonical_name.find("_exps.weight") != std::string::npos)
                        collected_from_frozen = true;
                }
            }

            if (!collected_from_frozen && !frozen_weights)
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                for (const auto &[name, tensor] : cache_)
                {
                    WeightSliceSpec full_slice;
                    add_moe_parent(name, tensor, tensor.get(), full_slice);
                }
            }
            else if (!collected_from_frozen && ordinary_overlay_requests)
            {
                LOG_ERROR("[WeightManager] GPU overlay preparation for "
                          << target_device.to_string()
                          << " has routed-expert requests but its graph-frozen set owns no matching expert parents; mutable-cache substitution is forbidden");
                return false;
            }

            // The preparation plan, not the set of available source tensors,
            // owns completeness. Include absent layers so a partial frozen set
            // cannot silently certify a model with missing prepared experts.
            if (overlay_preparation_plan)
                for (const auto &request : overlay_preparation_plan->requests())
                    if (request.device == target_device &&
                        (!request.projection_ownership || request.projection_ownership->hasMovableProjectionSet()))
                        moe_layers.try_emplace(request.layer);

            // Compile each projection independently. A fixed down bank covers
            // every expert; its gate/up parents may cover only a sparse subset.
            for (auto &[layer_idx, tensors] : moe_layers)
            {
                if (!overlay_preparation_plan &&
                    (!tensors.gate.tensor || !tensors.up.tensor || !tensors.down.tensor))
                {
                    LOG_WARN("[WeightManager] GPU pipeline: incomplete MoE expert tensors for layer "
                             << layer_idx << " — skipping");
                    continue;
                }

                if (!overlay_preparation_plan &&
                    (tensors.gate.selection->experts() != tensors.up.selection->experts() ||
                     tensors.gate.selection->experts() != tensors.down.selection->experts()))
                {
                    throw std::runtime_error(
                        "[WeightManager] GPU pipeline found inconsistent MoE expert slices for layer " +
                        std::to_string(layer_idx));
                }

                struct RoleTensor
                {
                    ExpertGemmRegistry::WeightRole role;
                    WeightRole semantic_role;
                    const char *tag;
                    const MoERoleTensor *source;
                };
                RoleTensor roles[] = {
                    {ExpertGemmRegistry::WeightRole::GATE, WeightRole::MoEExpertGate, "gate", &tensors.gate},
                    {ExpertGemmRegistry::WeightRole::UP, WeightRole::MoEExpertUp, "up", &tensors.up},
                    {ExpertGemmRegistry::WeightRole::DOWN, WeightRole::MoEExpertDown, "down", &tensors.down},
                };

                for (const auto &rt : roles)
                {
                    const std::vector<int> empty_source;
                    const auto &source_experts = rt.source->selection
                        ? rt.source->selection->experts() : empty_source;
                    const auto expected_experts = overlay_preparation_plan
                        ? overlay_preparation_plan->expertsForDeviceLayerRole(target_device, layer_idx, rt.role)
                        : source_experts;
                    if (rt.source->selection)
                        moe_parent_tensors.push_back({rt.source->name, rt.source->owner, expected_experts});

                    for (const int global_expert : expected_experts)
                    {
                        const auto *overlay_request = overlay_preparation_plan
                                                          ? overlay_preparation_plan->requestFor(target_device, layer_idx, global_expert, rt.role)
                                                          : nullptr;
                        if (overlay_preparation_plan && !overlay_request)
                        {
                            throw std::logic_error("GPU MoE preparation lost a planned projection request");
                        }
                        const auto projection = overlay_request ? overlay_request->projection_ownership : std::nullopt;

                        std::string slot_name = "moe_L" + std::to_string(layer_idx) + "_" + rt.tag + "_e" + std::to_string(global_expert);
                        std::string domain_name;
                        std::string tier_name;
                        int tier_index = -1;
                        int participant_world_rank = -1;
                        int participant_index = -1;
                        if (overlay_request)
                        {
                            domain_name = overlay_request->domain_name;
                            tier_name = overlay_request->tier_name;
                            tier_index = overlay_request->tier_index;
                            participant_world_rank = overlay_request->participant_world_rank;
                            participant_index = overlay_request->participant_index;
                            slot_name = "moe_" + registrySlotComponent(domain_name) + "_tier" +
                                        std::to_string(tier_index) + "_L" + std::to_string(layer_idx) +
                                        "_" + rt.tag + "_e" + std::to_string(global_expert);

                            auto *domain_engine =
                                expert_gemm_registry_.getEngineForDomain(
                                    domain_name, target_device, layer_idx,
                                    global_expert, rt.role, projection);
                            auto *participant_engine =
                                expert_gemm_registry_.getEngineForParticipant(
                                    domain_name, target_device,
                                    participant_world_rank, participant_index,
                                    layer_idx, global_expert, rt.role, projection);
                            if ((domain_engine == nullptr) !=
                                    (participant_engine == nullptr) ||
                                (domain_engine != nullptr &&
                                 domain_engine != participant_engine))
                            {
                                throw std::runtime_error(
                                    "[WeightManager] GPU overlay registry is partial or disagrees between participant and domain scopes for " +
                                    domain_name + " layer=" +
                                    std::to_string(layer_idx) + " expert=" +
                                    std::to_string(global_expert));
                            }
                            if (domain_engine != nullptr)
                            {
                                ++moe_jobs_already_satisfied;
                                continue;
                            }

                            if (projection && expert_gemm_registry_.getEngine(
                                    target_device, layer_idx, global_expert, rt.role))
                                throw std::logic_error(
                                    "GPU projected expert preparation cannot duplicate a complete device-scoped expert allocation");

                            /* A fresh overlay may adopt a legacy device-scoped
                             * engine exactly once. Both scoped aliases must be
                             * published from that same owner; accepting only
                             * one would create a torn bank that later context
                             * sealing cannot certify. */
                            const bool domain_aliased =
                                !projection && expert_gemm_registry_.aliasEngineForDomainFromDevice(
                                    domain_name, target_device, layer_idx,
                                    global_expert, rt.role);
                            const bool participant_aliased =
                                !projection && expert_gemm_registry_.aliasEngineForParticipantFromDevice(
                                    domain_name, target_device,
                                    participant_world_rank, participant_index,
                                    layer_idx, global_expert, rt.role);
                            if (domain_aliased || participant_aliased)
                            {
                                domain_engine =
                                    expert_gemm_registry_.getEngineForDomain(
                                        domain_name, target_device, layer_idx,
                                        global_expert, rt.role);
                                participant_engine =
                                    expert_gemm_registry_.getEngineForParticipant(
                                        domain_name, target_device,
                                        participant_world_rank,
                                        participant_index, layer_idx,
                                        global_expert, rt.role);
                                if (!domain_aliased || !participant_aliased ||
                                    domain_engine == nullptr ||
                                    domain_engine != participant_engine)
                                {
                                    throw std::runtime_error(
                                        "[WeightManager] GPU overlay could not atomically adopt one device-scoped expert into both registry scopes for " +
                                        domain_name + " layer=" +
                                        std::to_string(layer_idx) +
                                        " expert=" +
                                        std::to_string(global_expert));
                                }
                                ++moe_jobs_already_satisfied;
                                continue;
                            }
                        }

                        // Slice original contiguous GGUF rows before native packing.
                        // No complete GPU matrix or conversion round trip is needed.
                        // A previously prepared expert needs no surviving source;
                        // otherwise the exact logical resident must be present.
                        if (!rt.source->selection)
                            throw std::runtime_error("GPU preparation is missing a requested MoE source projection");
                        const auto matrix = rt.source->selection->matrix(global_expert, rt.semantic_role,
                            projection ? &*projection : nullptr);
                        const auto &role_shape = rt.source->tensor->shape();
                        const size_t element_offset = matrix.element_offset;
                        std::vector<size_t> view_shape = {matrix.rows, matrix.columns};
                        std::shared_ptr<TensorBase> view;
                        try
                        {
                            view = rt.source->tensor->create_view(view_shape, element_offset);
                        }
                        catch (const std::exception &e)
                        {
                            std::ostringstream oss;
                            oss << "[WeightManager] GPU pipeline: failed to create expert view for layer "
                                << layer_idx << " " << rt.tag
                                << " global_expert=" << global_expert
                                << " expert_count=" << source_experts.size()
                                << " tensor_shape=[";
                            for (size_t i = 0; i < role_shape.size(); ++i)
                            {
                                if (i != 0)
                                    oss << ",";
                                oss << role_shape[i];
                            }
                            oss << "] view_shape=[" << matrix.rows << "," << matrix.columns
                                << "] offset_elements=" << element_offset
                                << ": " << e.what();
                            throw std::runtime_error(oss.str());
                        }
                        if (!view)
                        {
                            throw std::runtime_error(
                                "[WeightManager] GPU pipeline: failed to create expert view for layer " + std::to_string(layer_idx) + " " + rt.tag + " expert " + std::to_string(global_expert) + " (shape=[" + std::to_string(matrix.rows) + "," + std::to_string(matrix.columns) + "], offset=" + std::to_string(element_offset) + ")");
                        }

                        moe_jobs.push_back({layer_idx, global_expert, rt.role, std::move(slot_name),
                                            std::move(domain_name), std::move(tier_name), tier_index,
                                            participant_world_rank, participant_index,
                                            rt.source->tensor,
                                            rt.source->owner,
                                            std::move(view), projection, std::nullopt,
                                            overlay_preparation_plan ? GPUPreparedWeightPoolLayout::MigrationReusable
                                                                     : GPUPreparedWeightPoolLayout::SourceNative});
                    }
                }

                LOG_DEBUG("[WeightManager] GPU pipeline: collected exact per-role MoE source views for layer " << layer_idx);
            }
        }

        for (auto &dense_job : gemm_weights)
        {
            // A frozen binding already owns its final scalar and operand policy.
            // Reinterpreting a Q8 control's decoded FP32 bank as a new original
            // source would round it a second time during GPU preparation.
            if (dense_job.binding)
            {
                dense_job.operand_precision = dense_job.binding->floating_operand_precision;
                continue;
            }
            const auto role = inferWeightRole(dense_job.name);
            dense_job.operand_precision = PreparedWeightRepresentationContract::operandPrecision(
                role, dense_job.tensor->native_type(), dense_job.name, loader_.architecture(),
                debugEnv().gemm.gdn_projection_arithmetic, debugEnv().gemm.moe_router_arithmetic);
            WeightSliceSpec logical_slice;
            WeightDerivationKind logical_derivation = WeightDerivationKind::Source;
            if (weight_metadata_)
            {
                if (const auto metadata = weight_metadata_->metadata(dense_job.tensor))
                {
                    logical_slice = metadata->slice;
                    logical_derivation = metadata->identity.derivation;
                }
            }
            if (!hasExplicitSlice(logical_slice))
                logical_slice = fullSliceSpec(*dense_job.tensor);

            auto prepared_override = createModelPreparedFloatingOverride(
                dense_job.name,
                role,
                dense_job.tensor,
                target_device,
                logical_slice,
                logical_derivation);
            if (!prepared_override)
                continue;

            dense_job.owner = prepared_override;
            dense_job.tensor = prepared_override.get();
        }

        if (frozen_weights)
        {
            std::vector<DenseGemmJob> pending_gemm_weights;
            pending_gemm_weights.reserve(gemm_weights.size());
            size_t adopted_dense = 0;

            for (auto &dense_job : gemm_weights)
            {
                if (!dense_job.binding.has_value())
                {
                    pending_gemm_weights.push_back(std::move(dense_job));
                    continue;
                }

                auto &binding = *dense_job.binding;
                if (preparedWeightStore()->adoptPreparedGemmForBinding(binding, target_device))
                {
                    markPrepState(dense_job.name, target_device, WeightPrepState::READY, true,
                                  "GPU pipeline: adopted already-loaded prepared GEMM handle");
                    ++adopted_dense;
                    continue;
                }

                pending_gemm_weights.push_back(std::move(dense_job));
            }

            if (adopted_dense > 0)
            {
                LOG_DEBUG("[WeightManager] GPU pipeline: adopted " << adopted_dense
                                                                    << " already-prepared GEMM bindings for "
                                                                    << target_device.to_string());
            }

            gemm_weights = std::move(pending_gemm_weights);
        }

        // Prepared adoption above is the only valid post-retirement route.
        // A pending job still owns an upload obligation, so missing immutable
        // bytes prove an invalid source lifecycle. Reloading would conceal the
        // early release and replace the frozen graph's physical source identity.
        for (const auto &dense_job : gemm_weights)
        {
            if (!dense_job.tensor || (!boundedGPUWeightSource(*dense_job.tensor) &&
                (dense_job.tensor->is_raw_data_released() || dense_job.tensor->size_bytes() == 0 || dense_job.tensor->raw_data() == nullptr)))
            {
                throw std::runtime_error(
                    "[WeightManager] GPU preparation requires a live immutable source for '" +
                    dense_job.name + "' on " + target_device.toString() +
                    "; source retired before preparation completed");
            }
        }

        if (gemm_weights.empty() && moe_jobs.empty())
        {
            if (moe_jobs_already_satisfied > 0)
            {
                LOG_DEBUG("[WeightManager] GPU pipeline: all " << moe_jobs_already_satisfied
                                                               << " requested MoE overlay expert slots already have registry entries for "
                                                               << target_device.to_string());
            }
            else
            {
                LOG_DEBUG("[WeightManager] GPU pipeline: no weights to load");
            }
            return true;
        }

        LOG_DEBUG("[WeightManager] GPU pipeline: loading " << gemm_weights.size()
                                                           << " GEMM weights + " << moe_jobs.size()
                                                           << " MoE expert slots for " << target_device.to_string());

        // ------------------------------------------------------------------
        // Step 2: Get backend
        // ------------------------------------------------------------------
        IBackend *backend = target_device.is_rocm()   ? getROCmBackend()
                            : target_device.is_cuda() ? getCUDABackend()
                                                      : nullptr;
        if (!backend)
        {
            LOG_ERROR("[WeightManager] GPU pipeline: no backend for " << target_device.to_string());
            return false;
        }

        // ------------------------------------------------------------------
        // Step 3: Create orchestrator and plan weights
        // ------------------------------------------------------------------
        const auto memory_authority = physicalMemoryAuthority();
        if (!memory_authority)
        {
            throw std::logic_error(
                "WeightManager GPU preparation reached allocation before topology-wide physical-memory admission");
        }
        const PhysicalMemoryOwner effective_persistent_owner =
            overlay_preparation_plan
                ? PhysicalMemoryOwner::RoutedExpertWeights
                : persistent_owner;
        auto orchestrator = std::make_shared<LoadOrchestrator>(
            backend,
            memory_authority,
            effective_persistent_owner);
        orchestrator->addDevice(target_device.ordinal);

        size_t max_raw_bytes = 0;
        size_t planned_storage_count = 0;
        std::vector<std::pair<std::string, const NativeQuantisedFormatInfo *>> weight_formats;
        weight_formats.reserve(gemm_weights.size());

        for (const auto &job : gemm_weights)
        {
            const std::string &name = job.name;
            TensorBase *tensor = job.tensor;
            if (!tensor)
                continue;

            const auto *bounded = boundedGPUWeightSource(*tensor);
            const NativeQuantisedFormatInfo *vnni = gpuWeightSourceFormat(*tensor);
            if (!vnni)
            {
                // Floating-point types (FP32, FP16, BF16) don't use VNNI repack —
                // they go through the pipeline as raw H2D copies (no GPU repack kernel).
                const auto type = tensor->native_type();
                if (type == TensorType::FP32 || type == TensorType::FP16 || type == TensorType::BF16)
                {
                    const int N = static_cast<int>(tensor->rows());
                    const int K = static_cast<int>(tensor->cols());
                    const size_t raw_bytes = tensor->size_bytes();

                    orchestrator->planRawWeight(target_device.ordinal, name, N, K, raw_bytes);
                    max_raw_bytes = std::max(max_raw_bytes, raw_bytes);
                    ++planned_storage_count;
                    weight_formats.emplace_back(name, nullptr);
                    continue;
                }
                // Any quantized type that doesn't implement IINT8Unpackable is a bug.
                throw std::runtime_error(
                    "[WeightManager] GPU pipeline: weight does not implement "
                    "IINT8Unpackable (unsupported quantized type for GPU repack): " +
                    name);
            }

            const int N = static_cast<int>(tensor->rows());
            const int K = static_cast<int>(tensor->cols());
            const size_t raw_bytes = bounded ? bounded->sourceBytes() : tensor->size_bytes();

            orchestrator->planWeight(target_device.ordinal, name, N, K,
                                     vnni->payload_bytes, canonicalDeviceVnniFormat(*vnni).is_asymmetric,
                                     vnni->has_emins, raw_bytes);
            max_raw_bytes = std::max(max_raw_bytes, raw_bytes);
            ++planned_storage_count;
            weight_formats.emplace_back(name, vnni);
        }

        // Plan MoE expert weights into contiguous parent-backed slabs. Graphs
        // retain one logical GEMM engine per expert, but GGUF stores every expert
        // for one layer/role parent in adjacent bytes. Keeping those two
        // granularities separate avoids tens of thousands of tiny pread, H2D,
        // event, and repack transactions without changing graph-visible expert
        // identity.
        struct MoEPackedStorageRef
        {
            std::string slot_name;
            int row_offset = 0;
        };

        std::vector<const NativeQuantisedFormatInfo *> moe_vnni_infos;
        moe_vnni_infos.reserve(moe_jobs.size());
        std::vector<MoEPackedStorageRef> moe_storage_refs(moe_jobs.size());
        std::vector<WeightJob> moe_logical_jobs;
        std::vector<const void *> moe_source_identities;
        std::vector<size_t> moe_logical_to_semantic;
        moe_logical_jobs.reserve(moe_jobs.size());
        moe_source_identities.reserve(moe_jobs.size());
        moe_logical_to_semantic.reserve(moe_jobs.size());
        size_t moe_jobs_without_raw_source = 0;

        for (size_t semantic_index = 0; semantic_index < moe_jobs.size(); ++semantic_index)
        {
            const auto &mj = moe_jobs[semantic_index];
            const NativeQuantisedFormatInfo *vnni = gpuWeightSourceFormat(*mj.view);
            const auto *bounded = dynamic_cast<const BoundedGPUExpertSource *>(mj.view.get());
            WeightJob logical_job;
            logical_job.name = mj.slot_name;
            logical_job.host_raw_data = mj.view ? mj.view->raw_data() : nullptr;

            if (!vnni)
            {
                // Floating-point types (FP32, FP16, BF16) don't use VNNI repack —
                // contiguous expert rows go through the same slab plan as a raw
                // H2D copy.
                const auto type = mj.view->native_type();
                if (type == TensorType::FP32 || type == TensorType::FP16 || type == TensorType::BF16)
                {
                    logical_job.raw_bytes = bounded ? bounded->sourceBytes() : mj.view->size_bytes();
                    logical_job.format = RepackFormat::RAW_FP;
                    logical_job.N = static_cast<int>(mj.view->rows());
                    logical_job.K = static_cast<int>(mj.view->cols());
                    logical_job.is_asymmetric = false;
                    moe_vnni_infos.push_back(nullptr);
                }
                else
                {
                    // Any quantized type that doesn't implement IINT8Unpackable is a bug.
                    throw std::runtime_error(
                        "[WeightManager] GPU pipeline: MoE expert view does not implement "
                        "IINT8Unpackable (unsupported quantized type for GPU repack): " +
                        mj.slot_name);
                }
            }
            else
            {
                const auto repack_format =
                    codebookIdToRepackFormat(vnni->codebook_id, vnni->is_superblock);
                if (!repack_format)
                {
                    throw std::runtime_error(
                        "[WeightManager] GPU pipeline has no MoE repack implementation for " +
                        mj.slot_name + " (codebook=" +
                        std::to_string(static_cast<int>(vnni->codebook_id)) +
                        ", superblock=" + std::to_string(vnni->is_superblock) + ")");
                }

#ifdef HAVE_ROCM
                if (target_device.is_rocm() &&
                    vnni->codebook_id >= 11 && vnni->codebook_id <= 17 &&
                    !rocm::ensureIQGridTablesInitialized(target_device.ordinal))
                {
                    throw std::runtime_error(
                        "[WeightManager] GPU pipeline failed to initialize ROCm IQ grid tables for " +
                        target_device.to_string());
                }
#endif

                logical_job.raw_bytes = bounded ? bounded->sourceBytes() : quantizedViewRawBytes(*mj.view);
                logical_job.format = *repack_format;
                logical_job.N = static_cast<int>(mj.view->rows());
                logical_job.K = static_cast<int>(mj.view->cols());
                logical_job.is_asymmetric = canonicalDeviceVnniFormat(*vnni).is_asymmetric;
                const auto allocation = gpuPreparedWeightPoolAllocation(*vnni, mj.pool_layout);
                logical_job.packed_payload_capacity_bytes_per_block =
                    allocation.payload_bytes_per_block;
                moe_vnni_infos.push_back(vnni);
            }

            if (bounded)
            {
                logical_job.bounded_source = bounded->source();
                logical_job.source_offset_bytes = bounded->sourceOffset();
            }
            if ((!logical_job.host_raw_data && !logical_job.bounded_source) || logical_job.raw_bytes == 0)
            {
                ++moe_jobs_without_raw_source;
                continue;
            }

            moe_logical_to_semantic.push_back(semantic_index);
            /*
             * Coalescing needs the immutable parent tensor's identity, not a
             * second owning handle. Graph-frozen overlay bindings may retain
             * ownership transitively through their expert views, so using the
             * optional parent_owner here incorrectly turns a valid lifetime
             * into a null grouping key. The view remains alive in `moe_jobs`
             * for the complete asynchronous staging operation.
             */
            moe_source_identities.push_back(mj.parent_identity);
            moe_logical_jobs.push_back(std::move(logical_job));
        }

        if (moe_jobs_without_raw_source != 0 &&
            moe_jobs_without_raw_source != moe_jobs.size())
        {
            throw std::runtime_error(
                "[WeightManager] GPU pipeline observed a partially released MoE parent set; "
                "all requested expert sources must share one explicit preparation lifetime");
        }
        if (!moe_jobs.empty() &&
            moe_jobs_without_raw_source == moe_jobs.size() &&
            max_raw_bytes != 0)
        {
            throw std::runtime_error(
                "[WeightManager] GPU pipeline cannot mix host-backed dense preparation with "
                "MoE expert requests whose immutable parent sources were already released");
        }

        std::vector<CoalescedWeightJobRun> moe_storage_runs;
        if (!moe_logical_jobs.empty())
        {
            moe_storage_runs = coalesceContiguousWeightJobs(
                moe_logical_jobs, moe_source_identities);

            for (size_t run_index = 0; run_index < moe_storage_runs.size(); ++run_index)
            {
                auto &run = moe_storage_runs[run_index];
                const size_t first_semantic_index =
                    moe_logical_to_semantic.at(run.members.front().source_job_index);
                run.job.name = moe_jobs[first_semantic_index].slot_name +
                               "__packed_run_" + std::to_string(run_index);

                for (const auto &member : run.members)
                {
                    const size_t semantic_index =
                        moe_logical_to_semantic.at(member.source_job_index);
                    moe_storage_refs[semantic_index] = {
                        .slot_name = run.job.name,
                        .row_offset = member.row_offset,
                    };
                }

                const auto *vnni = moe_vnni_infos[first_semantic_index];
                if (vnni)
                {
                    const auto layout = moe_jobs[first_semantic_index].pool_layout;
                    for (const auto &member : run.members)
                        if (moe_jobs[moe_logical_to_semantic.at(member.source_job_index)].pool_layout != layout)
                            throw std::logic_error("Coalesced MoE sources changed their physical pool layout");
                    const auto allocation = gpuPreparedWeightPoolAllocation(*vnni, layout);
                    if (run.job
                            .packed_payload_capacity_bytes_per_block !=
                        allocation.payload_bytes_per_block)
                    {
                        throw std::logic_error(
                            "[WeightManager] Coalesced MoE run lost its payload-allocation contract");
                    }
                    orchestrator->planWeight(
                        target_device.ordinal,
                        run.job.name,
                        run.job.N,
                        run.job.K,
                        allocation.payload_bytes_per_block,
                        allocation.has_mins,
                        allocation.has_emins,
                        run.job.raw_bytes);
                }
                else
                {
                    orchestrator->planRawWeight(
                        target_device.ordinal,
                        run.job.name,
                        run.job.N,
                        run.job.K,
                        run.job.raw_bytes);
                }

                max_raw_bytes = std::max(max_raw_bytes, run.job.raw_bytes);
                ++planned_storage_count;
            }

            LOG_DEBUG("[WeightManager] GPU pipeline: coalesced "
                      << moe_jobs.size() << " logical MoE expert matrices into "
                      << moe_storage_runs.size() << " contiguous packed source runs for "
                      << target_device.to_string());
            if (PerfStatsCollector::isDomainEnabled("weight_loading"))
            {
                const std::string device = target_device.to_string();
                PerfStatsCollector::addCounter(
                    "weight_loading", "gpu_pipeline_moe_logical_jobs",
                    static_cast<double>(moe_jobs.size()), "load", device);
                PerfStatsCollector::addCounter(
                    "weight_loading", "gpu_pipeline_moe_packed_source_runs",
                    static_cast<double>(moe_storage_runs.size()), "load", device);
            }
        }

        if (max_raw_bytes == 0)
        {
            // Dense jobs either adopted their exact prepared owner before
            // planning or retain a nonempty source. Reaching this boundary
            // with pending dense work means its storage plan is incomplete.
            if (!gemm_weights.empty())
                throw std::logic_error(
                    "[WeightManager] GPU dense source jobs have no planned upload storage on " +
                    target_device.toString());

            size_t aliased_experts = 0;
            size_t missing_experts = 0;
            for (const auto &moe_job : moe_jobs)
            {
                if (moe_job.static_source)
                {
                    // Complete static layers were already acquired before jobs
                    // were emitted. A new job with retired source is a fatal
                    // missing preparation, never an unscoped adoption.
                    ++missing_experts;
                    continue;
                }
                if (moe_job.domain_name.empty())
                    continue;

                auto *domain_engine =
                    expert_gemm_registry_.getEngineForDomain(
                        moe_job.domain_name,
                        target_device,
                        moe_job.layer_idx,
                        moe_job.expert_idx,
                        moe_job.role, moe_job.projection_ownership);
                auto *participant_engine =
                    expert_gemm_registry_.getEngineForParticipant(
                        moe_job.domain_name,
                        target_device,
                        moe_job.participant_world_rank,
                        moe_job.participant_index,
                        moe_job.layer_idx,
                        moe_job.expert_idx,
                        moe_job.role, moe_job.projection_ownership);
                if ((domain_engine == nullptr) !=
                        (participant_engine == nullptr) ||
                    (domain_engine != nullptr &&
                     domain_engine != participant_engine))
                {
                    throw std::runtime_error(
                        "[WeightManager] Released-source GPU overlay registry is partial or disagrees between participant and domain scopes for " +
                        moe_job.domain_name + " layer=" +
                        std::to_string(moe_job.layer_idx) + " expert=" +
                        std::to_string(moe_job.expert_idx));
                }
                if (domain_engine == nullptr && !moe_job.projection_ownership)
                {
                    const bool domain_aliased =
                        expert_gemm_registry_.aliasEngineForDomainFromDevice(
                            moe_job.domain_name,
                            target_device,
                            moe_job.layer_idx,
                            moe_job.expert_idx,
                            moe_job.role);
                    const bool participant_aliased =
                        expert_gemm_registry_.aliasEngineForParticipantFromDevice(
                            moe_job.domain_name,
                            target_device,
                            moe_job.participant_world_rank,
                            moe_job.participant_index,
                            moe_job.layer_idx,
                            moe_job.expert_idx,
                            moe_job.role);
                    domain_engine =
                        expert_gemm_registry_.getEngineForDomain(
                            moe_job.domain_name,
                            target_device,
                            moe_job.layer_idx,
                            moe_job.expert_idx,
                            moe_job.role);
                    participant_engine =
                        expert_gemm_registry_.getEngineForParticipant(
                            moe_job.domain_name,
                            target_device,
                            moe_job.participant_world_rank,
                            moe_job.participant_index,
                            moe_job.layer_idx,
                            moe_job.expert_idx,
                            moe_job.role);
                    if (domain_aliased != participant_aliased ||
                        (domain_aliased &&
                         (domain_engine == nullptr ||
                          domain_engine != participant_engine)))
                    {
                        throw std::runtime_error(
                            "[WeightManager] Released-source GPU overlay could not publish one coherent scoped alias for " +
                            moe_job.domain_name + " layer=" +
                            std::to_string(moe_job.layer_idx) +
                            " expert=" +
                            std::to_string(moe_job.expert_idx));
                    }
                }

                if (domain_engine != nullptr &&
                    domain_engine == participant_engine)
                {
                    ++aliased_experts;
                }
                else
                {
                    ++missing_experts;
                    LOG_WARN("[WeightManager] GPU pipeline: no already-loaded expert GEMM engine for "
                             << moe_job.domain_name << " layer=" << moe_job.layer_idx
                             << " expert=" << moe_job.expert_idx
                             << " role=" << moeWeightRoleName(moe_job.role)
                             << " on " << target_device.to_string());
                }
            }

            LOG_DEBUG("[WeightManager] GPU pipeline: resolved "
                      << (gemm_weights.size() + moe_jobs.size())
                      << " logical weights from existing prepared state for "
                      << target_device.to_string()
                      << " with no host-backed raw bytes; aliased_experts=" << aliased_experts
                      << " missing_experts=" << missing_experts);

            return missing_experts == 0;
        }

        // ------------------------------------------------------------------
        // Step 4: Allocate VRAM pool + pinned ring buffer
        // ------------------------------------------------------------------
        /**
         * The upload pipeline is a ring of pinned host slots paired with H2D
         * streams. Resolve that policy through the same typed authority used
         * by automatic ExpertOverlay capacity admission; this call site must
         * not reproduce the slot, reserve, or fit equations.
         */
        const auto *planned_pool = orchestrator->getPool(target_device.ordinal);
        const size_t planned_weight_bytes = planned_pool ? planned_pool->totalPlannedBytes() : 0;
        (void)planned_weight_bytes;
        const auto load_geometry = resolveGPUWeightLoadMemoryGeometry(
            max_raw_bytes,
            configuredGPUWeightLoadMemoryPolicy(
                staging_budget_bytes_override));
        const int repack_streams = load_geometry.staging_stream_count;
        const size_t staging_slot_bytes =
            load_geometry.staging_slot_bytes;
        const size_t staging_bytes = load_geometry.staging_bytes;
        LOG_DEBUG("[WeightManager] GPU load staging bounded to " << formatMiB(staging_bytes)
                                                                  << " total (" << repack_streams
                                                                  << " slots x " << formatMiB(staging_slot_bytes)
                                                                  << ", largest weight=" << formatMiB(max_raw_bytes) << ")");
        /*
         * LoadOrchestrator owns the only live free-memory observation and
         * performs it immediately before allocation. A second check here was
         * both redundant and racy with allocations by other TP participants.
         * This BOM call resolves only the shared staging policy; automatic
         * ExpertOverlay capacity and the allocator-adjacent preflight consume
         * the same arithmetic authority.
         */
        orchestrator->allocate(staging_slot_bytes, repack_streams);

        // ------------------------------------------------------------------
        // Step 5: Create weight jobs
        // ------------------------------------------------------------------
        for (size_t i = 0; i < gemm_weights.size(); ++i)
        {
            const auto &dense_job = gemm_weights[i];
            const std::string &name = dense_job.name;
            TensorBase *tensor = dense_job.tensor;
            const auto *vnni = weight_formats[i].second;
            if (!tensor)
                continue;
            if (!vnni)
            {
                // Floating-point weight: submit as RAW_FP passthrough job
                const auto type = tensor->native_type();
                if (type == TensorType::FP32 || type == TensorType::FP16 || type == TensorType::BF16)
                {
                    WeightJob job;
                    job.name = name;
                    job.host_raw_data = tensor->raw_data();
                    job.raw_bytes = tensor->size_bytes();
                    job.format = RepackFormat::RAW_FP;
                    job.N = static_cast<int>(tensor->rows());
                    job.K = static_cast<int>(tensor->cols());
                    job.is_asymmetric = false;
                    orchestrator->addWeightJob(target_device.ordinal, job);
                }
                continue;
            }

            auto repack_fmt = codebookIdToRepackFormat(vnni->codebook_id, vnni->is_superblock);
            if (!repack_fmt)
            {
                throw std::runtime_error(
                    "[WeightManager] GPU pipeline has no repack implementation for " +
                    name + " (codebook=" +
                    std::to_string(static_cast<int>(vnni->codebook_id)) +
                    ", superblock=" + std::to_string(vnni->is_superblock) + ")");
            }

#ifdef HAVE_ROCM
            if (target_device.is_rocm() && vnni->codebook_id >= 11 && vnni->codebook_id <= 17)
            {
                if (!rocm::ensureIQGridTablesInitialized(target_device.ordinal))
                {
                    LOG_ERROR("[WeightManager] GPU pipeline: failed to initialize ROCm IQ grid tables for "
                              << target_device.to_string());
                    return false;
                }
            }
#endif

            WeightJob job;
            job.name = name;
            if (const auto *bounded = boundedGPUWeightSource(*tensor))
            {
                job.bounded_source = bounded->source();
                job.source_offset_bytes = bounded->sourceOffset();
                job.columns = bounded->columns();
                job.raw_bytes = bounded->sourceBytes();
            }
            else
            {
                job.host_raw_data = tensor->raw_data();
                job.raw_bytes = tensor->size_bytes();
            }
            job.format = *repack_fmt;
            job.N = static_cast<int>(tensor->rows());
            job.K = static_cast<int>(tensor->cols());
            job.is_asymmetric = canonicalDeviceVnniFormat(*vnni).is_asymmetric;

            orchestrator->addWeightJob(target_device.ordinal, job);
        }

        // Add one source job per contiguous MoE parent run. The orchestrator may
        // split a run further at row boundaries to honor the fixed staging
        // budget, but it never reintroduces one transaction per logical expert.
        for (const auto &run : moe_storage_runs)
        {
            orchestrator->addWeightJob(target_device.ordinal, run.job);
        }

        // ------------------------------------------------------------------
        // Step 6: Execute pipeline (H2D + GPU repack)
        // ------------------------------------------------------------------
        DeviceLoadPipeline::ProgressCallback progress_cb;
        int progress_device_idx = -1;
        if (weight_load_progress_)
        {
            const size_t total_raw = orchestrator->totalPendingBytes(target_device.ordinal);
            progress_device_idx = weight_load_progress_->registerDevice(
                weight_load_progress_->makeDeviceLabel(target_device.to_string()), total_raw);
            progress_cb = weight_load_progress_->makeCallback(progress_device_idx);
        }
        orchestrator->load(progress_cb);
        if (weight_load_progress_ && progress_device_idx >= 0)
        {
            weight_load_progress_->finish(progress_device_idx);
        }

        // ------------------------------------------------------------------
        // Step 7: Create GEMM kernels from pool slots and register in KernelFactory
        // ------------------------------------------------------------------
        auto *pool = orchestrator->getPool(target_device.ordinal);
        if (!pool)
        {
            LOG_ERROR("[WeightManager] GPU pipeline: pool not found after load");
            return false;
        }

        size_t registered = 0;
        auto prepared_kind_for_device = [](DeviceId device)
        {
            if (device.is_cuda())
                return PreparedWeightKind::CudaInt8PackedGemm;
            if (device.is_rocm())
                return PreparedWeightKind::RocmInt8PackedGemm;
            return PreparedWeightKind::CpuPackedGemm;
        };

        auto register_pipeline_gemm = [&](const DenseGemmJob &dense_job,
                                          std::shared_ptr<ITensorGemm> kernel) -> bool
        {
            const std::string &name = dense_job.name;
            TensorBase *tensor = dense_job.tensor;
            if (!tensor || !kernel)
                return false;

            const bool quantized = gpuWeightSourceFormat(*tensor) != nullptr;
            KernelFactory::GemmPreparationKind resolved_kind = quantized
                                                                   ? (target_device.is_cuda()   ? KernelFactory::GemmPreparationKind::CUDA_INT8_PACKED
                                                                      : target_device.is_rocm() ? KernelFactory::GemmPreparationKind::ROCM_INT8_PACKED
                                                                                                : KernelFactory::GemmPreparationKind::CPU_PACKED)
                                                                   : KernelFactory::GemmPreparationKind::FLOATING_POINT;

            auto owned_kernel = std::move(kernel);
            auto prepared_weights = std::make_shared<KernelFactory::PreparedGemmWeights>();
            prepared_weights->kind = resolved_kind;
            prepared_weights->owned_kernel = owned_kernel;
            prepared_weights->kernel = owned_kernel.get();

            auto handle = std::make_shared<KernelFactory::PreparedGemmHandle>();
            handle->tensor = tensor;
            handle->device_id = target_device;
            handle->kind = resolved_kind;
            handle->variant = static_cast<int>(tensor->native_type());
            handle->prepared_weights = std::move(prepared_weights);

            WeightBinding binding;
            if (dense_job.binding.has_value())
            {
                binding = *dense_job.binding;
            }
            else
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                binding.binding_id = next_pipeline_prepared_binding_id_++;
                binding.identity = makeSourceWeightIdentity(name, {}, binding.binding_id);
                binding.floating_operand_precision = dense_job.operand_precision;
                if (weight_metadata_)
                {
                    if (auto metadata = weight_metadata_->metadata(tensor))
                    {
                        binding.identity.derivation = metadata->identity.derivation;
                        binding.identity.role = metadata->identity.role == WeightRole::Other
                                                    ? inferWeightRole(name)
                                                    : metadata->identity.role;
                        binding.identity.layer = metadata->identity.layer;
                        binding.identity.expert = metadata->identity.expert;
                        binding.slice = metadata->slice;
                        binding.residency = metadata->residency;
                    }
                }
            }
            binding.tensor = tensor;
            if (dense_job.owner)
                binding.tensor_owner = dense_job.owner;
            if (binding.slice.source_rows == 0 && binding.slice.source_cols == 0 &&
                binding.slice.row_count == 0 && binding.slice.col_count == 0)
            {
                binding.slice = fullSliceSpec(*tensor);
            }
            binding.residency.home_device = target_device;
            binding.residency.resident_device = target_device;
            // Preparing this device cannot weaken the frozen source lifetime
            // promised to another participant sharing the same TensorBase.
            binding.immutable = true;

            try
            {
                preparedWeightStore()->registerPreparedGemmHandle(
                    binding,
                    prepared_kind_for_device(target_device),
                    target_device,
                    std::move(handle));
                return true;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[WeightManager] GPU pipeline: failed to register prepared store entry for "
                          << name << ": " << e.what());
                return false;
            }
        };

        for (size_t i = 0; i < gemm_weights.size(); ++i)
        {
            const auto &dense_job = gemm_weights[i];
            const std::string &name = dense_job.name;
            TensorBase *tensor = dense_job.tensor;
            if (!tensor)
                continue;

            // One backend/format binding path is shared with bounded startup samples.
            auto kernel = llaminar::v2::kernels::KernelFactory::createGemmFromGPUWeightPool(
                *tensor, target_device, orchestrator, name, 0,
                GPUPreparedWeightPoolLayout::SourceNative, dense_job.operand_precision);

            if (kernel)
            {
                if (register_pipeline_gemm(dense_job, std::shared_ptr<ITensorGemm>(std::move(kernel))))
                {
                    markPrepState(name, target_device, WeightPrepState::UPLOADED_DEVICE, true,
                                  "GPU pipeline: loaded + registered");
                    markPrepState(name, target_device, WeightPrepState::READY, true,
                                  "GPU pipeline: GEMM weight ready");
                    ++registered;

                    // Preparation never retires shared source bytes. The
                    // complete model graph gate and physical source's CPU
                    // retention policy own reclamation together, regardless
                    // of this individual binding's shorter lifetime.
                }
                else
                {
                    LOG_ERROR("[WeightManager] GPU pipeline: failed to register kernel for " << name);
                }
            }
        }

        // ------------------------------------------------------------------
        // Step 7b: Create MoE expert GEMM kernels and register in ExpertGemmRegistry
        // ------------------------------------------------------------------
        size_t moe_registered = 0;
        for (size_t i = 0; i < moe_jobs.size(); ++i)
        {
            const auto &mj = moe_jobs[i];
            const auto &storage = moe_storage_refs[i];

            if (storage.slot_name.empty() || storage.row_offset < 0)
            {
                throw std::runtime_error(
                    "[WeightManager] GPU pipeline has no packed storage mapping for MoE expert " +
                    mj.slot_name);
            }

            // Coalesced offsets use the same checked pool-stride contract as
            // dense weights. Reusable storage is not a different source codebook.
            std::shared_ptr<ITensorGemm> kernel(
                llaminar::v2::kernels::KernelFactory::createGemmFromGPUWeightPool(
                    *mj.view, target_device, orchestrator, storage.slot_name,
                    static_cast<size_t>(storage.row_offset), mj.pool_layout));

            if (kernel)
            {
                ITensorGemm *raw_ptr = kernel.get();
                if (mj.static_source)
                {
                    const DenseGemmJob declared{mj.slot_name, mj.view.get(), mj.view, mj.static_source};
                    if (!register_pipeline_gemm(declared, kernel))
                        throw std::runtime_error("Static expert prepared-store publication failed");
                    expert_gemm_registry_.registerStaticProjection(*mj.static_source, raw_ptr, std::move(kernel));
                }
                else if (!mj.domain_name.empty())
                {
                    expert_gemm_registry_.registerEngineForParticipant(
                        mj.domain_name,
                        target_device, mj.participant_world_rank, mj.participant_index,
                        mj.layer_idx, mj.expert_idx, mj.role,
                        raw_ptr, kernel, mj.projection_ownership);
                    expert_gemm_registry_.registerEngineForDomain(
                        mj.domain_name,
                        target_device, mj.layer_idx, mj.expert_idx, mj.role,
                        raw_ptr, std::move(kernel), mj.projection_ownership);
                }
                else
                {
                    expert_gemm_registry_.registerEngine(
                        target_device, mj.layer_idx, mj.expert_idx, mj.role,
                        raw_ptr, std::move(kernel));
                }
                ++moe_registered;
            }
        }

        for (const auto &layer : static_layers)
        {
            const auto prepared = expert_gemm_registry_.acquireStaticProjectionLayer(
                layer.scope, layer.projection, layer.sources);
            if (!prepared) throw std::runtime_error("Static participant preparation omitted an original expert role");
            for (const auto &source : layer.sources)
                markPrepState(source.identity.canonical_name, target_device, WeightPrepState::READY, true,
                              "GPU static original projection source prepared");
        }

        // Mark 3D MoE parent tensors ready only after their exact layer/role has
        // all expert engines registered for this device. Host release happens in
        // releaseAllHostWeightData(), after all expected device tickets are ready.
        for (const auto &parent_record : moe_parent_tensors)
        {
            if (!parent_record.parent || parent_record.expected_experts.empty())
                continue;

            int layer_idx = -1;
            ExpertGemmRegistry::WeightRole role = ExpertGemmRegistry::WeightRole::GATE;
            if (!parseMoEExpertParentName(parent_record.name, layer_idx, role))
            {
                markPrepState(parent_record.name, target_device, WeightPrepState::FAILED, true,
                              "GPU pipeline: invalid MoE parent tensor metadata");
                continue;
            }

            if (expert_gemm_registry_.hasCompleteRoleForExperts(
                    target_device, layer_idx, parent_record.expected_experts, role))
            {
                markPrepState(parent_record.name, target_device, WeightPrepState::READY, true,
                              std::string("GPU pipeline: MoE ") + moeWeightRoleName(role) +
                                  " expert parent ready");
            }
            else if (overlay_preparation_plan)
            {
                const auto expected_domains = overlay_preparation_plan->domainsForDeviceLayerRole(
                    target_device, layer_idx, role);
                bool overlay_ready = !expected_domains.empty();
                std::string missing_domain;
                for (const auto &domain_name : expected_domains)
                {
                    const auto expected_experts = overlay_preparation_plan->expertsForDomainDeviceLayerRole(
                        domain_name, target_device, layer_idx, role);
                    if (expected_experts.empty())
                        continue;
                    const bool complete = std::all_of(expected_experts.begin(), expected_experts.end(), [&](int expert)
                    {
                        const auto *request = overlay_preparation_plan->requestFor(target_device, layer_idx, expert, role);
                        return request && request->domain_name == domain_name &&
                            expert_gemm_registry_.getEngineForDomain(domain_name, target_device, layer_idx,
                                expert, role, request->projection_ownership) != nullptr;
                    });
                    if (!complete)
                    {
                        overlay_ready = false;
                        missing_domain = domain_name;
                        break;
                    }
                }

                if (overlay_ready)
                {
                    markPrepState(parent_record.name, target_device, WeightPrepState::READY, true,
                                  std::string("GPU pipeline: MoE overlay ") + moeWeightRoleName(role) +
                                      " expert subset ready");
                }
                else
                {
                    markPrepState(parent_record.name, target_device, WeightPrepState::FAILED, true,
                                  std::string("GPU pipeline: incomplete MoE overlay ") + moeWeightRoleName(role) +
                                      " expert registry entries" +
                                      (missing_domain.empty() ? std::string() : " for domain " + missing_domain));
                }
            }
            else
            {
                markPrepState(parent_record.name, target_device, WeightPrepState::FAILED, true,
                              std::string("GPU pipeline: incomplete MoE ") + moeWeightRoleName(role) +
                                  " expert registry entries");
            }
        }

        if (moe_registered > 0)
        {
            LOG_DEBUG("[WeightManager] GPU pipeline: registered " << moe_registered
                                                                  << " MoE expert GEMM kernels in ExpertGemmRegistry");
        }

        const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        LOG_DEBUG("[WeightManager] GPU pipeline: loaded " << registered << "/" << gemm_weights.size()
                                                          << " dense + " << moe_registered << "/" << moe_jobs.size()
                                                          << " MoE expert weights in " << std::fixed << std::setprecision(1) << elapsed << " ms");

        // Release orchestrator staging resources now that all weights are loaded.
        // The VRAM pool is kept alive by the GEMM kernels' lifetime_owner_ shared_ptrs,
        // but the staging buffers (pinned ring) are no longer needed.
        orchestrator->finalize();
        {
            std::lock_guard<std::mutex> lock(physical_memory_authority_mutex_);
            const auto released=orchestrator->releasedWeightLoadStaging(target_device.ordinal);
            const auto prior=released_weight_load_staging_.find(target_device);
            // Several ordinary preparation families may finish in any order.
            // Retain their largest actual resolved slot, never a late smaller
            // family's cap or a new size derived from the complete table.
            if (prior == released_weight_load_staging_.end() || prior->second.slotBytes() < released.slotBytes())
                released_weight_load_staging_.insert_or_assign(target_device,released);
        }

        /*
         * Publish only the finalized persistent region. The registry retains a
         * weak owner, so this record cannot keep an abandoned or Dynamic-only
         * pool alive. At the reusable-model seal, live kernel ownership is the
         * exact predicate deciding whether these bytes remain in the retirement
         * BOM; loader staging has already been released and is never counted.
         */
        const auto *finalized_pool =
            orchestrator->getPool(target_device.ordinal);
        if (!finalized_pool || !finalized_pool->isAllocated() ||
            finalized_pool->totalPlannedBytes() == 0u)
        {
            throw std::runtime_error(
                "WeightManager GPU pipeline finalized without a persistent prepared-weight pool");
        }
        const auto allocation_lifetime =
            finalized_pool->persistentAllocationLifetime();
        if (!allocation_lifetime)
        {
            throw std::logic_error(
                "WeightManager GPU pipeline finalized without a persistent allocation lifetime token");
        }
        prepared_device_allocations_.registerAllocation(
            target_device,
            allocation_lifetime,
            finalized_pool->totalPlannedBytes());

        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            num_gpu_packed_ += registered;
        }

        const bool complete =
            registered == gemm_weights.size() &&
            moe_registered == moe_jobs.size();
        if (!complete)
        {
            LOG_ERROR("[WeightManager] GPU pipeline registration incomplete for "
                      << target_device.to_string()
                      << ": dense=" << registered << "/" << gemm_weights.size()
                      << " moe=" << moe_registered << "/" << moe_jobs.size()
                      << " packed_storage_runs=" << planned_storage_count);
        }
        return complete;
    }

    bool WeightManager::uploadFrozenNonGemmWeights(
        DeviceId target_device,
        const FrozenModelWeightSet &frozen_weights,
        const std::function<bool(const std::string &)> &layer_filter,
        bool include_expert_jobs,
        PhysicalMemoryOwner persistent_owner)
    {
        (void)include_expert_jobs;

        if (!target_device.is_gpu())
        {
            throw std::invalid_argument(
                "WeightManager::uploadFrozenNonGemmWeights requires a GPU target");
        }

        size_t uploaded_count = 0;
        std::unordered_set<TensorBase *> visited;
        visited.reserve(frozen_weights.bindings().size());

        for (const auto &binding : frozen_weights.bindings())
        {
            if (!bindingTargetsDevice(binding, target_device))
                continue;

            const std::string &name = binding.identity.canonical_name;
            if (layer_filter && !layer_filter(name))
                continue;

            if (!binding.tensor)
            {
                throw std::runtime_error(
                    "[WeightManager] Frozen binding targeting " +
                    target_device.to_string() +
                    " has no tensor: " + name);
            }

            /*
             * Dense GEMM bindings are prepared into backend-native packed
             * storage by packGemmWeightsViaPipeline(). Routed experts are
             * prepared into expert slabs by the explicit expert pipeline.
             * Neither representation consumes the binding's raw tensor pointer.
             */
            if (shouldPrepareFrozenGemmBinding(
                    *this,
                    binding,
                    target_device,
                    /*include_expert_jobs=*/true) ||
                isRoutedExpertBinding(binding) ||
                name.find("_exps.weight") != std::string::npos)
            {
                continue;
            }

            /*
             * Quantized embeddings have one production representation on a
             * GPU: source-native PreparedEmbeddingWeights. Preparing that view
             * here keeps its exact native block bytes inside the device lifecycle
             * without a second raw upload or another quantization operation.
             *
             * Do this before pointer de-duplication. A tied embedding/LM-head
             * tensor may legitimately need both an embedding representation
             * and a GEMM representation, and those representations have
             * independent binding identities.
             */
            if (binding.identity.role == WeightRole::Embedding &&
                IINT8Unpackable::fromTensor(binding.tensor))
            {
                WeightBinding prepared_binding = binding;
                prepared_binding.residency.home_device = target_device;
                prepared_binding.residency.resident_device = target_device;

                auto store = preparedWeightStore();
                if (!store->adoptPreparedEmbeddingForBinding(
                        prepared_binding,
                        target_device))
                {
                    const size_t vocab_offset =
                        prepared_binding.slice.row_start;
                    const size_t total_vocab =
                        prepared_binding.slice.source_rows > 0
                            ? prepared_binding.slice.source_rows
                            : prepared_binding.tensor->rows();
                    store->prepareEmbedding(
                        prepared_binding,
                        static_cast<int>(prepared_binding.tensor->cols()),
                        vocab_offset,
                        total_vocab,
                        persistent_owner);
                }

                markPrepState(
                    name,
                    target_device,
                    WeightPrepState::READY,
                    false,
                    "native quantized embedding prepared without duplicate raw upload");
                ++uploaded_count;
                continue;
            }

            TensorBase *tensor = binding.tensor;
            if (!visited.insert(tensor).second)
                continue;

            if (tensor->debugName().empty())
                tensor->setDebugName(name);

            const auto upload =
                TransferEngine::instance().upload(tensor, target_device);
            if (!upload.success)
            {
                markPrepState(
                    name,
                    target_device,
                    WeightPrepState::FAILED,
                    false,
                    "frozen non-GEMM binding upload failed");
                throw std::runtime_error(
                    "[WeightManager] Exact frozen non-GEMM binding '" + name +
                    "' failed upload to " + target_device.to_string() +
                    ": " + upload.error);
            }

            if (!tensor->gpu_data_ptr() ||
                !tensor->is_on_device(target_device))
            {
                // Prepared representations were handled by their typed roles
                // above. A HOST_RESIDENT flag alone cannot excuse a missing raw
                // graph operand, especially for native floating embeddings.
                markPrepState(
                    name, target_device, WeightPrepState::FAILED, false,
                    "frozen non-GEMM binding lacks exact device residency");
                throw std::runtime_error(
                    "[WeightManager] Exact frozen non-GEMM binding '" + name +
                    "' was not resident on " + target_device.to_string() +
                    " after preparation; a raw graph weight cannot be host-only");
            }

            markPrepState(
                name,
                target_device,
                WeightPrepState::UPLOADED_DEVICE,
                false,
                "exact frozen non-GEMM binding uploaded");
            markPrepState(
                name,
                target_device,
                WeightPrepState::READY,
                false,
                "exact frozen non-GEMM binding ready");
            ++uploaded_count;
        }

        LOG_DEBUG("[WeightManager] Uploaded " << uploaded_count
                                               << " exact frozen non-GEMM bindings to "
                                               << target_device.to_string());
        return true;
    }

    bool WeightManager::uploadNonGemmWeights(
        DeviceId target_device,
        std::function<bool(const std::string &)> layer_filter)
    {
        if (!target_device.is_gpu())
        {
            LOG_DEBUG("[WeightManager] uploadNonGemmWeights skipped for CPU target");
            return true;
        }

        size_t uploaded_count = 0;

        LOG_DEBUG("[WeightManager] Uploading non-GEMM weights to " << target_device.to_string() << "...");

        // Snapshot non-GEMM names under lock, then process each tensor with
        // lock held only for cache access/clone bookkeeping.
        std::vector<std::string> non_gemm_names;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            non_gemm_names.reserve(cache_.size());
            for (const auto &[name, tensor] : cache_)
            {
                if (!isGemmWeight(name) && tensor)
                {
                    // MoE 3D expert parent tensors are prepared through per-expert
                    // views in the unified GPU pipeline. Uploading the raw parent
                    // as a non-GEMM tensor duplicates several GB of data and can
                    // exhaust VRAM before graph execution.
                    if (name.find("_exps.weight") != std::string::npos)
                        continue;
                    if (layer_filter && !layer_filter(name))
                        continue;
                    non_gemm_names.push_back(name);
                }
            }
        }

        for (const auto &name : non_gemm_names)
        {
            std::shared_ptr<TensorBase> tensor;
            std::shared_ptr<TensorBase> device_tensor_holder;

            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto base_it = cache_.find(name);
                if (base_it == cache_.end() || !base_it->second)
                {
                    throw std::runtime_error(
                        "[WeightManager] Non-GEMM weight missing from cache: " + name);
                }
                tensor = base_it->second;

                // Get the device-specific tensor if multi-GPU, otherwise use original
                if (first_device_.has_value() && first_device_.value() == target_device)
                {
                    // First device uses original tensor
                    device_tensor_holder = tensor;
                }
                else if (first_device_.has_value())
                {
                    // Subsequent devices need clones - check per-device cache
                    const auto cache_key = DeviceWeightCacheKey::forDevice(target_device, name);
                    auto it = per_device_cache_.find(cache_key);
                    if (it != per_device_cache_.end())
                    {
                        device_tensor_holder = it->second;
                    }
                    else
                    {
                        // Create clone for this device
                        auto clone = cloneTensorForDevice(name, tensor, target_device);
                        if (clone)
                        {
                            device_tensor_holder = clone;
                            per_device_cache_[cache_key] = clone;
                        }
                    }
                }
                else
                {
                    // First device not set yet, set it now
                    first_device_ = target_device;
                    device_tensor_holder = tensor;
                }
            }

            if (!device_tensor_holder)
            {
                throw std::runtime_error(
                    "[WeightManager] Failed to get device tensor for: " + name);
            }

            TensorBase *device_tensor = device_tensor_holder.get();

            // Set debug name so transfers can be traced
            device_tensor->setDebugName(name);

            // Non-GEMM weights still use the sole transfer authority; they do
            // not need the GEMM packing pipeline, but they must obey the same
            // explicit placement and failure contract.
            const auto upload =
                TransferEngine::instance().upload(device_tensor, target_device);
            if (upload.success)
            {
                size_t rows = device_tensor->shape()[0];
                size_t cols = device_tensor->shape().size() > 1 ? device_tensor->shape()[1] : 1;
                size_t bytes = rows * cols * sizeof(float);
                markPrepState(name, target_device, WeightPrepState::UPLOADED_DEVICE, false, "non-GEMM TransferEngine upload completed");
                markPrepState(name, target_device, WeightPrepState::READY, false, "non-GEMM ready");
                LOG_TRACE("[WeightManager] Uploaded non-GEMM weight: " << name
                                                                       << " [" << rows << "x" << cols << "] = " << bytes << " bytes");
                uploaded_count++;
            }
            else
            {
                markPrepState(name, target_device, WeightPrepState::FAILED, false, "non-GEMM TransferEngine upload failed");
                throw std::runtime_error(
                    "[WeightManager] Required non-GEMM weight '" + name +
                    "' failed upload to " + target_device.to_string() +
                    ": " + upload.error);
            }
        }

        LOG_DEBUG("[WeightManager] Uploaded " << uploaded_count << " non-GEMM weights to "
                                              << target_device.to_string());
        return true;
    }

    size_t WeightManager::releaseAllHostWeightData()
    {
        // Phase 9: Hard gate — do NOT release host data until all lifecycle gates
        // are complete. This replaces the old ad-hoc deferred_host_release_pending_
        // flag in RankOrchestrator. The caller must mark graph_materialization_complete
        // after all graphs have resolved their prepared weight bindings.
        if (!lifecycle_gates_.canReleaseHostData())
        {
            LOG_DEBUG("[WeightManager] releaseAllHostWeightData() blocked by lifecycle gate "
                      "(state="
                      << toString(lifecycle_gates_.currentState())
                      << "). Host data retained until all gates complete.");
            return 0;
        }

        // Mark host_release_allowed now that we're proceeding
        lifecycle_gates_.host_release_allowed = true;

        std::lock_guard<std::mutex> lock(cache_mutex_);

        size_t released_count = 0;
        size_t skipped_count = 0;
        size_t error_count = 0;
        size_t released_bytes = 0;
        size_t retained_bytes = 0;
        size_t retained_count = 0;
        std::unordered_set<TensorBase *> visited_ptrs;

        auto moe_parent_ready_for_release = [&](const TensorBase &tensor, const std::string &key) -> bool
        {
            int layer_idx = -1;
            ExpertGemmRegistry::WeightRole role = ExpertGemmRegistry::WeightRole::GATE;
            if (!parseMoEExpertParentName(key, layer_idx, role))
                return false;

            const int num_experts = moeExpertCountFromParentTensor(tensor);
            if (num_experts <= 0)
                return false;

            std::lock_guard<std::mutex> prep_lock(prep_ticket_mutex_);
            auto expected_it = expected_devices_by_weight_.find(key);
            auto tickets_it = prep_tickets_.find(key);
            if (expected_it == expected_devices_by_weight_.end() ||
                tickets_it == prep_tickets_.end() ||
                expected_it->second.empty())
            {
                return false;
            }

            const auto &tickets = tickets_it->second;
            return std::all_of(expected_it->second.begin(), expected_it->second.end(),
                               [&](const std::string &device_key)
                               {
                                   auto ticket_it = tickets.find(device_key);
                                   return ticket_it != tickets.end() &&
                                          ticket_it->second.state == WeightPrepState::READY;
                               });
        };

        auto try_release = [&](TensorBase *ptr, const std::string &key)
        {
            if (!ptr || !visited_ptrs.insert(ptr).second)
                return; // null or already visited

            const size_t tensor_bytes = ptr->is_raw_data_released() ? 0 : ptr->size_bytes();

            // Skip tensors that are already released
            if (ptr->is_raw_data_released())
            {
                skipped_count++;
                return;
            }

            // Phase 9: hostDataRequired() handles policy-based retention. The
            // lifecycle gate ensures graph construction has consumed prepared
            // bindings before raw host data can be reclaimed.
            if (hostDataRequired(ptr, key))
            {
                skipped_count++;
                retained_bytes += tensor_bytes;
                retained_count++;
                LOG_DEBUG("[WeightManager] RETAINED host data for " << key
                                                                    << " (" << (tensor_bytes / 1024) << " KB)"
                                                                    << " (host policy requires retention)");
                return;
            }

            // Release host data for tensors that have valid GPU data OR
            // kernel-managed device data (GEMM packed weights, prepared embedding).
            // GEMM kernels upload pre-packed representations to their own device
            // buffers and never read the raw TensorBase data, so the host copy can
            // be freed. Similarly, PreparedEmbeddingWeights hold their own GPU copy.
            if (!ptr->deviceValid())
            {
                bool has_kernel_device_data = ptr->hasPreparedDeviceState();

                // 3D MoE expert tensors (_exps.weight) are safe to release only
                // after this exact parent layer/role has complete expert engines
                // for every expected device. A non-empty global registry is not
                // sufficient proof for an unrelated parent tensor.
                if (!has_kernel_device_data && key.find("_exps.weight") != std::string::npos)
                {
                    has_kernel_device_data = moe_parent_ready_for_release(*ptr, key);
                }

                if (!has_kernel_device_data)
                {
                    skipped_count++;
                    retained_bytes += tensor_bytes;
                    retained_count++;
                    LOG_TRACE("[WeightManager] RETAINED host data for " << key
                                                                        << " (" << (tensor_bytes / 1024) << " KB)"
                                                                        << " deviceValid=" << ptr->deviceValid()
                                                                        << " hasPreparedDeviceState=" << ptr->hasPreparedDeviceState()
                                                                        << " hostResident=" << ptr->isHostResident()
                                                                        << " type=" << static_cast<int>(ptr->native_type()));
                    return;
                }
            }

            try
            {
                released_bytes += tensor_bytes;
                ptr->release_host_weight_data();
                released_count++;
            }
            catch (const std::exception &e)
            {
                LOG_WARN("[WeightManager] Failed to release host data for tensor "
                         << ptr->debugName() << " (type=" << static_cast<int>(ptr->native_type())
                         << "): " << e.what());
                error_count++;
            }
        };

        // Sweep main cache
        for (auto &[name, tensor] : cache_)
        {
            try_release(tensor.get(), name);
        }

        // Sweep per-device cache (clones for multi-GPU)
        for (auto &[key, tensor] : per_device_cache_)
        {
            try_release(tensor.get(), key.name);
        }

        LOG_DEBUG("[WeightManager] Released host weight data: " << released_count
                                                                << " tensors (" << (released_bytes / (1024 * 1024)) << " MB) released, "
                                                                << retained_count << " tensors (" << (retained_bytes / (1024 * 1024)) << " MB) retained, "
                                                                << skipped_count << " already released, " << error_count << " errors"
                                                                << " | cache=" << cache_.size()
                                                                << " per_device=" << per_device_cache_.size()
                                                                << " decode=" << decode_cache_.size());
        return released_count;
    }

    size_t WeightManager::releaseHostResidentWeightData()
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        size_t released_count = 0;
        size_t released_bytes = 0;
        size_t skipped_views = 0;
        std::unordered_set<TensorBase *> visited_ptrs;

        auto try_release = [&](TensorBase *ptr, const std::string &key)
        {
            if (!ptr || !visited_ptrs.insert(ptr).second)
                return;
            if (ptr->is_view())
            {
                ++skipped_views;
                return;
            }
            if (ptr->is_raw_data_released())
                return;
            if (!ptr->isHostResident())
                return;
            if (hostDataRequired(ptr, key))
            {
                LOG_DEBUG("[WeightManager] RETAINED host-resident weight for " << key
                                                                               << " (host policy requires retention)");
                return;
            }

            size_t tensor_bytes = ptr->size_bytes();
            ptr->release_host_weight_data();
            released_bytes += tensor_bytes;
            released_count++;
            LOG_DEBUG("[WeightManager] Released host-resident weight: " << key
                                                                        << " (" << (tensor_bytes / (1024 * 1024)) << " MB)");
        };

        for (auto &[name, tensor] : cache_)
        {
            try_release(tensor.get(), name);
        }

        for (auto &[key, tensor] : per_device_cache_)
        {
            try_release(tensor.get(), key.name);
        }

        if (released_count > 0)
        {
            LOG_DEBUG("[WeightManager] Post-upload host-resident release: "
                      << released_count << " tensors (" << (released_bytes / (1024 * 1024)) << " MB) freed");
        }
        if (skipped_views > 0)
        {
            LOG_DEBUG("[WeightManager] Post-upload host-resident release skipped "
                      << skipped_views << " borrowed tensor views");
        }

        /*
         * Do not call loader_.adviseMmapDontneed() here.  This broad release
         * intentionally skips borrowed TensorSlice views, and those views can
         * still have CUDA/HIP host registrations retained for their completed
         * upload lifecycle. MADV_DONTNEED before unregistering those views can
         * leave the runtime with a stale DMA mapping and corrupt an unrelated
         * host consumer such as the parity snapshot bank.
         *
         * The exactly-once scheduleMmapReclaim() phase owns that second half of
         * the protocol on its background worker: it first releases every
         * remaining mmap host registration, then applies backing-aware advice.
         * Callers publish that phase only after the completed first-prefill
         * producer; the typed lifecycle makes duplicate or inline work
         * unrepresentable in normal runner code.
         */

        return released_count;
    }

    MmapReclaimLifecycle::Submission WeightManager::scheduleMmapReclaim()
    {
        return mmap_reclaim_lifecycle_.schedule();
    }

    MmapReclaimLifecycle::Completion
    WeightManager::awaitMmapReclaimBeforeHostAllocation()
    {
        return mmap_reclaim_lifecycle_.awaitBeforeHostAllocation();
    }

    size_t WeightManager::performMmapReclaim()
    {
        /*
         * First-prefill reclamation is one complete worker-owned operation.
         * Releasing residual embedding/sidecar host buffers here, rather than
         * immediately before scheduleMmapReclaim(), keeps both destructor work
         * and glibc arena scans off the inference authority thread.  The
         * completed load pipeline and first successful prefill are the exact
         * producer boundary that makes these bytes unreachable by execution.
         */
        const size_t released_host_tensors =
            releaseHostResidentWeightData();

        std::vector<std::shared_ptr<TensorBase>> mapped_tensors;
        size_t mmap_tensors_unregistered = 0;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            std::unordered_set<TensorBase *> visited;

            auto retain_mapped_tensor = [&](const std::shared_ptr<TensorBase> &tensor)
            {
                TensorBase *ptr = tensor.get();
                if (!ptr || !visited.insert(ptr).second || !ptr->is_mmap_data())
                    return;
                mapped_tensors.push_back(tensor);
            };

            for (const auto &[_, tensor] : cache_)
                retain_mapped_tensor(tensor);
            for (const auto &[_, tensor] : per_device_cache_)
                retain_mapped_tensor(tensor);
            for (const auto &[_, tensor] : decode_cache_)
                retain_mapped_tensor(tensor);
        }

        // Host registration retirement can enter CUDA/HIP runtime locks and
        // durable-file advice can walk very large page tables. Both operations
        // stay on this background worker; the shared_ptr snapshot preserves
        // tensor lifetimes without holding cache_mutex_ against inference.
        for (const auto &tensor : mapped_tensors)
        {
            tensor->releaseMmapHostRegistration();
            ++mmap_tensors_unregistered;
        }

        if (mmap_tensors_unregistered > 0)
        {
            LOG_DEBUG("[WeightManager] Released host registrations for "
                      << mmap_tensors_unregistered
                      << " mmap-backed tensors before backing-aware reclaim");
        }

        const size_t advised_bytes = loader_.adviseMmapDontneed();

#if defined(__GLIBC__)
        /*
         * malloc_trim() can walk every process arena and issue anonymous
         * MADV_DONTNEED operations.  It is therefore maintenance work even
         * when the individual ranges are small.  Run it only on this worker,
         * after all model-owned frees, and let later model/JIT admission cross
         * awaitMmapReclaimBeforeHostAllocation() before it needs the capacity.
         */
        if (released_host_tensors > 0u)
            ::malloc_trim(0);
#else
        (void)released_host_tensors;
#endif

        return advised_bytes;
    }

    size_t WeightManager::releaseMoEExpertHostWeightData()
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        size_t released_count = 0;
        size_t released_bytes = 0;
        size_t skipped_views = 0;
        size_t already_released = 0;
        size_t error_count = 0;
        std::unordered_set<TensorBase *> visited;

        auto try_release = [&](const std::string &key, const std::shared_ptr<TensorBase> &tensor)
        {
            if (key.find("_exps.weight") == std::string::npos || !tensor)
                return;

            TensorBase *ptr = tensor.get();
            if (!visited.insert(ptr).second)
                return;

            if (ptr->is_view())
            {
                ++skipped_views;
                return;
            }

            if (ptr->is_raw_data_released())
            {
                ++already_released;
                return;
            }

            const size_t bytes = ptr->size_bytes();
            try
            {
                ptr->release_host_weight_data();
                released_bytes += bytes;
                ++released_count;
            }
            catch (const std::exception &e)
            {
                ++error_count;
                LOG_WARN("[WeightManager] Failed to release cached MoE expert raw data for "
                         << key << " (" << (bytes >> 20) << " MB): " << e.what());
            }
        };

        for (const auto &[name, tensor] : cache_)
            try_release(name, tensor);
        for (const auto &[key, tensor] : per_device_cache_)
            try_release(key.name, tensor);

        LOG_DEBUG("[WeightManager] Released cached MoE expert raw data: "
                  << released_count << " tensors (" << (released_bytes >> 20) << " MB) released, "
                  << already_released << " already released, "
                  << skipped_views << " borrowed views skipped, "
                  << error_count << " errors");

#if defined(__GLIBC__)
        if (released_bytes > 0)
            ::malloc_trim(0);
#endif

        return released_bytes;
    }

    void WeightManager::logHostMemorySummary(const char *context) const
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        struct Bucket
        {
            size_t entries = 0;
            size_t unique_tensors = 0;
            size_t live_heap_bytes = 0;
            size_t live_mmap_bytes = 0;
            size_t released = 0;
            size_t borrowed_views = 0;
        };

        Bucket cache_bucket;
        Bucket per_device_bucket;
        Bucket decode_bucket;
        std::unordered_set<TensorBase *> visited;

        auto visit = [&](Bucket &bucket, const std::shared_ptr<TensorBase> &tensor)
        {
            ++bucket.entries;
            TensorBase *ptr = tensor.get();
            if (!ptr || !visited.insert(ptr).second)
                return;
            ++bucket.unique_tensors;
            if (ptr->is_view())
            {
                ++bucket.borrowed_views;
                return;
            }
            if (ptr->is_raw_data_released())
            {
                ++bucket.released;
                return;
            }
            const size_t bytes = ptr->size_bytes();
            if (ptr->is_mmap_data())
                bucket.live_mmap_bytes += bytes;
            else
                bucket.live_heap_bytes += bytes;
        };

        for (const auto &[_, tensor] : cache_)
            visit(cache_bucket, tensor);
        for (const auto &[_, tensor] : per_device_cache_)
            visit(per_device_bucket, tensor);
        for (const auto &[_, tensor] : decode_cache_)
            visit(decode_bucket, tensor);

        auto total_heap = cache_bucket.live_heap_bytes + per_device_bucket.live_heap_bytes + decode_bucket.live_heap_bytes;
        auto total_mmap = cache_bucket.live_mmap_bytes + per_device_bucket.live_mmap_bytes + decode_bucket.live_mmap_bytes;
        auto total_released = cache_bucket.released + per_device_bucket.released + decode_bucket.released;
        auto total_views = cache_bucket.borrowed_views + per_device_bucket.borrowed_views + decode_bucket.borrowed_views;

        LOG_DEBUG("[WeightManager] Host memory summary"
                  << (context ? std::string(" (") + context + ")" : std::string{})
                  << ": heap=" << (total_heap >> 20) << " MB"
                  << " mmap=" << (total_mmap >> 20) << " MB"
                  << " released=" << total_released
                  << " borrowed_views=" << total_views
                  << " | cache=" << (cache_bucket.live_heap_bytes >> 20) << "/" << (cache_bucket.live_mmap_bytes >> 20)
                  << " MB heap/mmap entries=" << cache_bucket.entries
                  << " unique=" << cache_bucket.unique_tensors
                  << " | per_device=" << (per_device_bucket.live_heap_bytes >> 20) << "/" << (per_device_bucket.live_mmap_bytes >> 20)
                  << " MB heap/mmap entries=" << per_device_bucket.entries
                  << " unique=" << per_device_bucket.unique_tensors
                  << " | decode=" << (decode_bucket.live_heap_bytes >> 20) << "/" << (decode_bucket.live_mmap_bytes >> 20)
                  << " MB heap/mmap entries=" << decode_bucket.entries
                  << " unique=" << decode_bucket.unique_tensors);
    }

    // =============================================================================
    // Device-aware weight slicing for LOCAL TP
    // =============================================================================

    std::shared_ptr<TensorBase> WeightManager::sliceRowRange(
        const std::shared_ptr<TensorBase> &tensor,
        size_t row_start,
        size_t row_count)
    {
        if (!tensor)
        {
            throw std::runtime_error("[WeightManager] sliceRowRange: null tensor");
        }

        const auto &shape = tensor->shape();
        if (shape.size() != 2)
        {
            throw std::runtime_error("[WeightManager] sliceRowRange requires 2D tensor, got " +
                                     std::to_string(shape.size()) + "D");
        }

        size_t out_dim = shape[0];
        size_t in_dim = shape[1];

        if (row_start + row_count > out_dim)
        {
            throw std::runtime_error("[WeightManager] sliceRowRange: row_start=" + std::to_string(row_start) +
                                     " + row_count=" + std::to_string(row_count) +
                                     " > out_dim=" + std::to_string(out_dim));
        }

        // Currently only FP32 slicing is supported
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(tensor.get());
        if (!fp32_tensor)
        {
            throw std::runtime_error("[WeightManager] sliceRowRange currently requires FP32 tensor. "
                                     "For quantized weights, use GGUF loadTensorRowSlice instead.");
        }

        // Create sliced tensor
        std::vector<size_t> slice_shape = {row_count, in_dim};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);

        // Copy the specified row range
        const float *src = fp32_tensor->data() + row_start * in_dim;
        float *dst = sliced->mutable_data();
        std::memcpy(dst, src, row_count * in_dim * sizeof(float));

        LOG_TRACE("[WeightManager] sliceRowRange: [" << out_dim << ", " << in_dim
                                                     << "] -> rows [" << row_start << ", " << (row_start + row_count)
                                                     << ") = [" << row_count << ", " << in_dim << "]");

        return sliced;
    }

    std::shared_ptr<TensorBase> WeightManager::sliceColumnRange(
        const std::shared_ptr<TensorBase> &tensor,
        size_t col_start,
        size_t col_count)
    {
        if (!tensor)
        {
            throw std::runtime_error("[WeightManager] sliceColumnRange: null tensor");
        }

        const auto &shape = tensor->shape();
        if (shape.size() != 2)
        {
            throw std::runtime_error("[WeightManager] sliceColumnRange requires 2D tensor, got " +
                                     std::to_string(shape.size()) + "D");
        }

        size_t rows = shape[0];
        size_t cols = shape[1];

        if (col_start + col_count > cols)
        {
            throw std::runtime_error("[WeightManager] sliceColumnRange: col_start=" + std::to_string(col_start) +
                                     " + col_count=" + std::to_string(col_count) +
                                     " > cols=" + std::to_string(cols));
        }

        if (col_count == 0)
        {
            throw std::runtime_error("[WeightManager] sliceColumnRange: col_count cannot be 0");
        }

        // Currently only FP32 slicing is supported
        auto *fp32_tensor = dynamic_cast<FP32Tensor *>(tensor.get());
        if (!fp32_tensor)
        {
            throw std::runtime_error("[WeightManager] sliceColumnRange currently requires FP32 tensor. "
                                     "For quantized weights, use GGUF loadTensorColumnSlice instead.");
        }

        // Create sliced tensor
        std::vector<size_t> slice_shape = {rows, col_count};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);

        // Copy column range row by row
        const float *src = fp32_tensor->data();
        float *dst = sliced->mutable_data();
        for (size_t row = 0; row < rows; ++row)
        {
            std::memcpy(dst + row * col_count, src + row * cols + col_start, col_count * sizeof(float));
        }

        LOG_TRACE("[WeightManager] sliceColumnRange: [" << rows << ", " << cols
                                                        << "] -> cols [" << col_start << ", " << (col_start + col_count)
                                                        << ") = [" << rows << ", " << col_count << "]");

        return sliced;
    }

    GDNHeadAssignment WeightManager::gdnHeadAssignmentFor(
        const DeviceShardingAssignment &assignment, const TensorParallelConfig &tp_config) const
    {
        if (!has_gdn_dimensions_ || gdn_n_k_heads_ <= 0 ||
            gdn_n_v_heads_ <= 0 || gdn_d_state_ <= 0)
        {
            throw std::runtime_error("[WeightManager] GDN linked sharding requires complete GDN dimensions");
        }

        return GDNHeadAssignment::fromPartition(
            gdn_n_k_heads_,
            gdn_n_v_heads_,
            assignment.head_start,
            assignment.head_count,
            tp_config.totalHeads());
    }

    GDNHeadAssignment WeightManager::gdnHeadAssignmentForEqualRank(
        int rank,
        int world_size) const
    {
        if (!has_gdn_dimensions_ || gdn_n_k_heads_ <= 0 ||
            gdn_n_v_heads_ <= 0 || gdn_d_state_ <= 0)
        {
            throw std::runtime_error("[WeightManager] GDN linked sharding requires complete GDN dimensions");
        }
        return GDNHeadAssignment::forEqualRank(
            gdn_n_k_heads_, gdn_n_v_heads_, rank, world_size);
    }

    int WeightManager::gdnValueElementsPerHead(
        const std::string &name,
        size_t total_size) const
    {
        if (!has_gdn_dimensions_ || gdn_n_v_heads_ <= 0 || gdn_d_state_ <= 0)
            return 0;

        int elements_per_head = 0;
        if (endsWith(name, "attn_gate.weight") ||
            endsWith(name, "ssm_out.weight"))
        {
            elements_per_head = gdn_d_state_;
        }
        else if (endsWith(name, "ssm_alpha.weight") ||
                 endsWith(name, "ssm_beta.weight") ||
                 endsWith(name, "ssm_dt.bias") ||
                 endsWith(name, "ssm_a"))
        {
            elements_per_head = 1;
        }
        else
        {
            return 0;
        }

        const size_t expected =
            static_cast<size_t>(gdn_n_v_heads_) *
            static_cast<size_t>(elements_per_head);
        return total_size == expected ? elements_per_head : 0;
    }

    std::shared_ptr<TensorBase> WeightManager::loadNativeRowSpanConcat(
        const std::string &name,
        DeviceId device,
        const std::vector<GDNHeadSpan> &spans,
        size_t source_cols, int original_tp_rank)
    {
        if (spans.empty() || source_cols == 0)
            throw std::invalid_argument("[WeightManager] Row-span concat requires non-empty geometry for: " + name);

        if (device.is_cpu()) {
            const auto source = loader_.nativeWeightSourceGeometry(name);
            if (source.original_shape.size() != 2 || source.original_shape[0] != source_cols)
                throw std::invalid_argument("CPU ordered rows changed original native source geometry");
            auto assembly = orderedCPUNativeAssembly(source, NativeWeightSourceAssembly::Axis::Rows, spans);
            return loadCPUWeightRange(name, device, WeightDerivationKind::FusedSubblockConcat,
                {}, std::move(assembly), original_tp_rank);
        }

        std::vector<std::shared_ptr<TensorBase>> slices;
        slices.reserve(spans.size());
        size_t output_rows = 0;
        size_t output_bytes = 0;
        std::optional<TensorType> native_type;

        for (const GDNHeadSpan span : spans)
        {
            if (span.start < 0 || span.count <= 0)
                throw std::invalid_argument("[WeightManager] Invalid GDN row span for: " + name);

            auto slice = loader_.loadTensorRowSlice(
                name,
                static_cast<size_t>(span.start),
                static_cast<size_t>(span.end()),
                device,
                WeightPrecision::NATIVE);
            if (!slice)
                throw std::runtime_error("[WeightManager] Failed to load native GDN row span for: " + name);
            if (slice->shape().size() != 2 ||
                slice->shape()[0] != static_cast<size_t>(span.count) ||
                slice->shape()[1] != source_cols)
            {
                throw std::runtime_error("[WeightManager] GDN row span returned an incompatible shape for: " + name);
            }
            if (!slice->raw_data())
                throw std::runtime_error("[WeightManager] GDN row span has no native bytes for: " + name);
            if (native_type && *native_type != slice->native_type())
                throw std::runtime_error("[WeightManager] GDN row spans disagree on tensor type for: " + name);

            native_type = slice->native_type();
            output_rows += static_cast<size_t>(span.count);
            output_bytes += slice->size_bytes();
            slices.push_back(std::move(slice));
        }

        std::vector<uint8_t> packed(output_bytes);
        size_t byte_offset = 0;
        for (const auto &slice : slices)
        {
            std::memcpy(
                packed.data() + byte_offset,
                slice->raw_data(),
                slice->size_bytes());
            byte_offset += slice->size_bytes();
        }

        auto packed_tensor = createTensorFromRawData(
            *native_type,
            {output_rows, source_cols},
            std::move(packed));
        if (!packed_tensor)
            throw std::runtime_error("[WeightManager] Failed to create packed GDN row tensor for: " + name);
        return std::shared_ptr<TensorBase>(std::move(packed_tensor));
    }

    std::shared_ptr<TensorBase> WeightManager::loadNativeColumnSpanConcat(
        const std::string &name,
        DeviceId device,
        const std::vector<GDNHeadSpan> &spans,
        size_t source_rows, int original_tp_rank)
    {
        if (spans.empty() || source_rows == 0)
            throw std::invalid_argument("[WeightManager] Column-span concat requires non-empty geometry for: " + name);

        if (device.is_cpu()) {
            const auto source = loader_.nativeWeightSourceGeometry(name);
            if (source.original_shape.size() != 2 || source.original_shape[1] != source_rows)
                throw std::invalid_argument("CPU ordered columns changed original native source geometry");
            auto assembly = orderedCPUNativeAssembly(source, NativeWeightSourceAssembly::Axis::Columns, spans);
            return loadCPUWeightRange(name, device, WeightDerivationKind::FusedSubblockConcat,
                {}, std::move(assembly), original_tp_rank);
        }

        std::vector<std::shared_ptr<TensorBase>> slices;
        std::vector<size_t> row_bytes;
        slices.reserve(spans.size());
        row_bytes.reserve(spans.size());
        size_t output_cols = 0;
        size_t output_row_bytes = 0;
        std::optional<TensorType> native_type;

        for (const GDNHeadSpan span : spans)
        {
            if (span.start < 0 || span.count <= 0)
                throw std::invalid_argument("[WeightManager] Invalid GDN column span for: " + name);

            auto slice = loader_.loadTensorColumnSlice(
                name,
                static_cast<size_t>(span.start),
                static_cast<size_t>(span.end()),
                device,
                WeightPrecision::NATIVE);
            if (!slice)
            {
                throw std::runtime_error(
                    "[WeightManager] Native GDN column span is not representable for '" +
                    name + "'; TP head boundaries must align to the tensor codebook block");
            }
            if (slice->shape().size() != 2 ||
                slice->shape()[0] != source_rows ||
                slice->shape()[1] != static_cast<size_t>(span.count) ||
                slice->size_bytes() % source_rows != 0)
            {
                throw std::runtime_error("[WeightManager] GDN column span returned an incompatible shape for: " + name);
            }
            if (!slice->raw_data())
                throw std::runtime_error("[WeightManager] GDN column span has no native bytes for: " + name);
            if (native_type && *native_type != slice->native_type())
                throw std::runtime_error("[WeightManager] GDN column spans disagree on tensor type for: " + name);

            native_type = slice->native_type();
            const size_t bytes = slice->size_bytes() / source_rows;
            row_bytes.push_back(bytes);
            output_row_bytes += bytes;
            output_cols += static_cast<size_t>(span.count);
            slices.push_back(std::move(slice));
        }

        std::vector<uint8_t> packed(source_rows * output_row_bytes);
        for (size_t row = 0; row < source_rows; ++row)
        {
            size_t destination_offset = row * output_row_bytes;
            for (size_t slice_idx = 0; slice_idx < slices.size(); ++slice_idx)
            {
                const auto &slice = slices[slice_idx];
                const size_t bytes = row_bytes[slice_idx];
                const auto *source =
                    static_cast<const uint8_t *>(slice->raw_data()) + row * bytes;
                std::memcpy(packed.data() + destination_offset, source, bytes);
                destination_offset += bytes;
            }
        }

        auto packed_tensor = createTensorFromRawData(
            *native_type,
            {source_rows, output_cols},
            std::move(packed));
        if (!packed_tensor)
            throw std::runtime_error("[WeightManager] Failed to create packed GDN column tensor for: " + name);
        return std::shared_ptr<TensorBase>(std::move(packed_tensor));
    }

    std::shared_ptr<TensorBase> WeightManager::loadGDNFusedQKVColumnParallel(
        const std::string &name,
        DeviceId device,
        const GDNHeadAssignment &head_assignment,
        int rank,
        int world_size,
        const std::vector<size_t> &dimensions)
    {
        if (!device.is_valid() || dimensions.size() != 2)
            throw std::invalid_argument("[WeightManager] GDN fused loader received an invalid target or shape");

        const size_t key_rows =
            static_cast<size_t>(gdn_n_k_heads_) * gdn_d_state_;
        const size_t value_rows =
            static_cast<size_t>(gdn_n_v_heads_) * gdn_d_state_;
        const size_t expected_rows = 2 * key_rows + value_rows;
        if (dimensions[0] != expected_rows)
            throw std::invalid_argument("[WeightManager] GDN fused loader received a non-GDN tensor: " + name);

        const GDNHeadSpan key_span =
            head_assignment.keyElementSpan(gdn_d_state_);
        std::vector<GDNHeadSpan> source_spans;
        source_spans.reserve(
            2 + head_assignment.valueHeadSpans().size());
        source_spans.push_back(key_span);
        source_spans.push_back(
            {.start = static_cast<int>(key_rows) + key_span.start,
             .count = key_span.count});
        for (const GDNHeadSpan value_span :
             head_assignment.valueElementSpans(gdn_d_state_))
        {
            source_spans.push_back(
                {.start = static_cast<int>(2 * key_rows) + value_span.start,
                 .count = value_span.count});
        }

        auto packed = loadNativeRowSpanConcat(
            name, device, source_spans, dimensions[1], rank);
        const size_t expected_local_rows =
            head_assignment.localFusedRows(gdn_d_state_);
        if (packed->shape()[0] != expected_local_rows)
            throw std::runtime_error("[WeightManager] GDN fused packed row count is inconsistent");

        auto result = std::make_shared<TensorSlice>(
            std::move(packed),
            SliceMetadata::forColumnParallel(
                dimensions[0], dimensions[1], rank, world_size, true));

        WeightSliceSpec slice;
        slice.source_rows = dimensions[0];
        slice.source_cols = dimensions[1];
        slice.row_count = expected_local_rows;
        slice.col_count = dimensions[1];
        slice.inner_is_presliced = true;
        registerDerivedMetadata(
            name, result, WeightDerivationKind::FusedSubblockConcat, slice, device);
        return result;
    }

    std::shared_ptr<TensorBase> WeightManager::loadGDNValueRows(
        const std::string &name,
        DeviceId device,
        const GDNHeadAssignment &head_assignment,
        int rank,
        int world_size,
        const std::vector<size_t> &dimensions)
    {
        if (!device.is_valid() || dimensions.empty() || dimensions.size() > 2)
            throw std::invalid_argument("[WeightManager] GDN value-row loader received an invalid target or shape");

        const int elements_per_head =
            gdnValueElementsPerHead(name, dimensions[0]);
        if (elements_per_head <= 0)
            throw std::invalid_argument("[WeightManager] GDN value-row loader received a non-GDN tensor: " + name);

        const std::vector<GDNHeadSpan> spans =
            head_assignment.valueElementSpans(elements_per_head);
        const size_t local_elements =
            static_cast<size_t>(head_assignment.localValueHeads()) *
            static_cast<size_t>(elements_per_head);

        std::shared_ptr<TensorBase> packed;
        if (dimensions.size() == 1 && device.is_cpu()) {
            const auto source = loader_.nativeWeightSourceGeometry(name);
            if (source.original_shape != dimensions)
                throw std::invalid_argument("CPU ordered scalar source changed original vector geometry");
            auto assembly = orderedCPUNativeAssembly(source, NativeWeightSourceAssembly::Axis::Columns, spans);
            packed = loadCPUWeightRange(name, device, WeightDerivationKind::FusedSubblockConcat,
                {}, std::move(assembly), rank);
        }
        else if (dimensions.size() == 1)
        {
            auto full = loader_.loadTensor(name, device,
                    isImmutableFP32WeightRole(inferWeightRole(name)) ? WeightPrecision::CONVERT_TO_FP32 : WeightPrecision::NATIVE);
            if (!full)
                throw std::runtime_error("[WeightManager] Failed to load GDN scalar tensor: " + name);
            auto *source = dynamic_cast<FP32Tensor *>(full.get());
            if (!source)
                throw std::runtime_error("[WeightManager] GDN scalar tensor must be FP32: " + name);

            auto destination = std::make_shared<FP32Tensor>(
                std::vector<size_t>{local_elements});
            size_t destination_offset = 0;
            for (const GDNHeadSpan span : spans)
            {
                std::memcpy(
                    destination->mutable_data() + destination_offset,
                    source->data() + span.start,
                    static_cast<size_t>(span.count) * sizeof(float));
                destination_offset += static_cast<size_t>(span.count);
            }
            packed = std::move(destination);
        }
        else
        {
            packed = loadNativeRowSpanConcat(
                name, device, spans, dimensions[1], rank);
        }

        /*
         * The packed payload is already the final participant-local byte range,
         * but it must still carry the same typed ownership wrapper as every
         * other TP-derived weight.  Returning a bare FP32Tensor for scalar GDN
         * weights made a second cache lookup indistinguishable from an
         * accidentally replicated clone.  Keep 1D and 2D value weights under
         * one cache identity contract.
         */
        auto result = std::make_shared<TensorSlice>(
            std::move(packed),
            SliceMetadata::forColumnParallel(
                dimensions[0],
                dimensions.size() == 2 ? dimensions[1] : 1,
                rank,
                world_size,
                true));

        WeightSliceSpec slice;
        slice.source_rows = dimensions[0];
        slice.source_cols = dimensions.size() == 2 ? dimensions[1] : 1;
        slice.row_count = local_elements;
        slice.col_count = slice.source_cols;
        slice.inner_is_presliced = true;
        registerDerivedMetadata(
            name, result, WeightDerivationKind::FusedSubblockConcat, slice, device);
        return result;
    }

    std::shared_ptr<TensorBase> WeightManager::loadGDNValueColumns(
        const std::string &name,
        DeviceId device,
        const GDNHeadAssignment &head_assignment,
        int rank,
        int world_size,
        const std::vector<size_t> &dimensions)
    {
        if (!device.is_valid() || dimensions.size() != 2)
            throw std::invalid_argument("[WeightManager] GDN value-column loader received an invalid target or shape");

        const int elements_per_head =
            gdnValueElementsPerHead(name, dimensions[1]);
        if (elements_per_head <= 0)
            throw std::invalid_argument("[WeightManager] GDN value-column loader received a non-GDN tensor: " + name);

        const std::vector<GDNHeadSpan> spans =
            head_assignment.valueElementSpans(elements_per_head);
        auto packed = loadNativeColumnSpanConcat(
            name, device, spans, dimensions[0], rank);

        auto result = std::make_shared<TensorSlice>(
            std::move(packed),
            SliceMetadata::forRowParallel(
                dimensions[0], dimensions[1], rank, world_size, true));

        WeightSliceSpec slice;
        slice.source_rows = dimensions[0];
        slice.source_cols = dimensions[1];
        slice.row_count = dimensions[0];
        slice.col_count = result->shape()[1];
        slice.inner_is_presliced = true;
        registerDerivedMetadata(
            name, result, WeightDerivationKind::FusedSubblockConcat, slice, device);
        return result;
    }

    // ========================================================================
    // Sharding helper methods
    // ========================================================================

    // ========================================================================
    // Helper: Compute slice boundaries based on dimension type from config
    // ========================================================================

    bool WeightManager::computeSliceBoundaries(
        const std::string &name,
        size_t total_size,
        const DeviceShardingAssignment &assignment,
        const TensorParallelConfig &tp_config,
        size_t &out_start,
        size_t &out_count) const
    {
        if (!has_sharding_config_)
        {
            throw std::runtime_error("[WeightManager] No sharding config set - cannot compute slice boundaries");
        }

        WeightDimensionType dim_type = sharding_config_.getDimensionType(name);

        switch (dim_type)
        {
        case WeightDimensionType::Heads:
        {
            const int total_heads = tp_config.totalHeads();
            if (total_heads <= 0)
            {
                throw std::runtime_error("[WeightManager] Invalid total_heads in TensorParallelConfig");
            }
            const size_t head_dim = total_size / static_cast<size_t>(total_heads);
            out_start = assignment.head_start * head_dim;
            out_count = assignment.head_count * head_dim;
            LOG_TRACE("[WeightManager] " << name << " (Heads): total_heads=" << total_heads
                                         << " head_dim=" << head_dim
                                         << " -> [" << out_start << ", " << (out_start + out_count) << ")");
            return true;
        }

        case WeightDimensionType::KVHeads:
        {
            const int total_kv_heads = tp_config.totalKVHeads();
            if (total_kv_heads <= 0)
            {
                throw std::runtime_error("[WeightManager] Invalid total_kv_heads in TensorParallelConfig");
            }
            const size_t head_dim = total_size / static_cast<size_t>(total_kv_heads);
            out_start = assignment.kv_head_start * head_dim;
            out_count = assignment.kv_head_count * head_dim;
            LOG_TRACE("[WeightManager] " << name << " (KVHeads): total_kv_heads=" << total_kv_heads
                                         << " head_dim=" << head_dim
                                         << " -> [" << out_start << ", " << (out_start + out_count) << ")");
            return true;
        }

        case WeightDimensionType::FFNHidden:
        {
            out_start = assignment.d_ff_start;
            out_count = assignment.d_ff_count;
            LOG_TRACE("[WeightManager] " << name << " (FFNHidden): d_ff=["
                                         << out_start << ", " << (out_start + out_count) << ")");
            return true;
        }

        case WeightDimensionType::Vocab:
        {
            out_start = assignment.vocab_start;
            out_count = assignment.vocab_count;
            LOG_TRACE("[WeightManager] " << name << " (Vocab): vocab=["
                                         << out_start << ", " << (out_start + out_count) << ")");
            return true;
        }

        case WeightDimensionType::ProportionalHeads:
        {
            const int total_heads = tp_config.totalHeads();
            if (total_heads <= 0)
            {
                throw std::runtime_error("[WeightManager] Invalid total_heads for ProportionalHeads slicing");
            }
            out_start = total_size * static_cast<size_t>(assignment.head_start) / static_cast<size_t>(total_heads);
            const size_t end = total_size * static_cast<size_t>(assignment.head_start + assignment.head_count) / static_cast<size_t>(total_heads);
            out_count = end - out_start;
            LOG_TRACE("[WeightManager] " << name << " (ProportionalHeads): total_heads=" << total_heads
                                         << " weight_size=" << total_size
                                         << " -> [" << out_start << ", " << (out_start + out_count) << ")");
            return true;
        }

        case WeightDimensionType::Bias1D:
        case WeightDimensionType::None:
        default:
            throw std::runtime_error("[WeightManager] Cannot compute slice boundaries for dimension type " +
                                     std::to_string(static_cast<int>(dim_type)) + " on weight: " + name);
        } // switch
    }

    std::shared_ptr<TensorBase> WeightManager::loadColumnParallel1DBias(
        const std::string &name,
        DeviceId device,
        const DeviceShardingAssignment &assignment,
        const std::vector<size_t> &dimensions)
    {
        const auto tp_config = tensorParallelConfig();
        // 1D bias tensor - slice along single dimension
        size_t total_size = dimensions[0];
        if (tp_config && tp_config->worldSize() > 1 &&
            gdnValueElementsPerHead(name, total_size) > 0)
        {
            const GDNHeadAssignment head_assignment =
                gdnHeadAssignmentFor(assignment, *tp_config);
            return loadGDNValueRows(
                name, device, head_assignment, assignment.local_rank,
                tp_config->worldSize(), dimensions);
        }

        size_t slice_start = 0;
        size_t slice_count = 0;

        // Use config to determine dimension type for slicing
        if (!computeSliceBoundaries(name, total_size, assignment, *tp_config, slice_start, slice_count))
        {
            throw std::runtime_error("[WeightManager] Failed to compute slice boundaries for 1D tensor: " + name);
        }

        if (device.is_cpu())
            return loadCPUWeightRange(name,device,WeightDerivationKind::RowSlice,
                WeightSliceSpec{.row_start=slice_start,.row_count=slice_count,.inner_is_presliced=true},{},assignment.local_rank);
        // Load full tensor and slice in memory (biases are small)
        auto full_tensor = loader_.loadTensor(name, device,
                    isImmutableFP32WeightRole(inferWeightRole(name)) ? WeightPrecision::CONVERT_TO_FP32 : WeightPrecision::NATIVE);
        if (!full_tensor)
        {
            throw std::runtime_error("[WeightManager] Failed to load 1D tensor: " + name);
        }
        registerSourceMetadata(name, full_tensor, device);

        // Create sliced tensor
        auto *fp32_full = dynamic_cast<FP32Tensor *>(full_tensor.get());
        if (!fp32_full)
        {
            throw std::runtime_error("[WeightManager] 1D bias must be FP32: " + name);
        }

        std::vector<size_t> slice_shape = {slice_count};
        auto sliced = std::make_shared<FP32Tensor>(slice_shape);
        std::memcpy(sliced->mutable_data(),
                    fp32_full->data() + slice_start,
                    slice_count * sizeof(float));

        LOG_TRACE("[WeightManager] Device " << device.to_string()
                                            << " (rank " << assignment.local_rank << "/" << tp_config->worldSize() << ")"
                                            << " column-parallel 1D " << name
                                            << " [" << total_size << "]"
                                            << " -> [" << slice_start << ", " << (slice_start + slice_count) << ")"
                                            << " = " << slice_count << " elements");

        auto result = std::make_shared<TensorSlice>(
            std::move(sliced),
            SliceMetadata::forColumnParallel(
                total_size,
                1,
                assignment.local_rank,
                tp_config->worldSize(),
                true));

        WeightSliceSpec bias_slice;
        bias_slice.source_rows = total_size;
        bias_slice.source_cols = 1;
        bias_slice.row_start = slice_start;
        bias_slice.row_count = slice_count;
        bias_slice.col_start = 0;
        bias_slice.col_count = 1;
        bias_slice.inner_is_presliced = true;
        registerDerivedMetadata(name, result, WeightDerivationKind::RowSlice, bias_slice, device);

        return result;
    }

    std::shared_ptr<TensorBase> WeightManager::loadColumnParallel2DWeight(
        const std::string &name,
        DeviceId device,
        const DeviceShardingAssignment &assignment,
        const std::vector<size_t> &dimensions)
    {
        const auto tp_config = tensorParallelConfig();
        size_t total_rows = dimensions[0];
        size_t cols = dimensions[1];

        const WeightDimensionType dimension =
            has_sharding_config_
                ? sharding_config_.getDimensionType(name)
                : WeightDimensionType::None;
        const size_t expected_gdn_fused_rows =
            has_gdn_dimensions_
                ? (2 * static_cast<size_t>(gdn_n_k_heads_) +
                   static_cast<size_t>(gdn_n_v_heads_)) *
                      static_cast<size_t>(gdn_d_state_)
                : 0;
        if (tp_config && tp_config->worldSize() > 1)
        {
            if (dimension == WeightDimensionType::FusedQKVHeads &&
                total_rows == expected_gdn_fused_rows)
            {
                const GDNHeadAssignment head_assignment =
                    gdnHeadAssignmentFor(assignment, *tp_config);
                return loadGDNFusedQKVColumnParallel(
                    name, device, head_assignment, assignment.local_rank,
                    tp_config->worldSize(), dimensions);
            }
            if (gdnValueElementsPerHead(name, total_rows) > 0)
            {
                const GDNHeadAssignment head_assignment =
                    gdnHeadAssignmentFor(assignment, *tp_config);
                return loadGDNValueRows(
                    name, device, head_assignment, assignment.local_rank,
                    tp_config->worldSize(), dimensions);
            }
        }

        // Fused QKV weights need special handling: 3 concatenated sub-blocks
        // each split independently by heads
        if (has_sharding_config_ &&
            sharding_config_.getDimensionType(name) == WeightDimensionType::FusedQKVHeads)
        {
            return loadFusedQKVColumnParallel(name, device, assignment, dimensions);
        }

        size_t row_start = 0;
        size_t row_count = 0;

        // Use config-based dimension type to determine slicing
        if (!computeSliceBoundaries(name, total_rows, assignment, *tp_config, row_start, row_count))
        {
            throw std::runtime_error("[WeightManager] Failed to compute slice boundaries for 2D column-parallel: " + name);
        }

        // Load only the slice from GGUF file (memory efficient, preserves quantization)
        auto slice_tensor = (device.is_cpu() ? loadCPUWeightRange(name,device,WeightDerivationKind::RowSlice,
                WeightSliceSpec{.row_start=row_start,.row_count=(row_start + row_count)-(row_start),.inner_is_presliced=true},{},assignment.local_rank) : loader_.loadTensorRowSlice(
            name, row_start, row_start + row_count, device, WeightPrecision::NATIVE));

        if (!slice_tensor)
        {
            throw std::runtime_error("[WeightManager] Failed to load row slice for: " + name);
        }

        // Wrap in TensorSlice with metadata
        auto meta = SliceMetadata::forColumnParallel(
            total_rows, cols, assignment.local_rank, tp_config->worldSize(),
            true /* inner_is_presliced */);

        auto result = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);

        WeightSliceSpec column_slice;
        column_slice.source_rows = total_rows;
        column_slice.source_cols = cols;
        column_slice.row_start = row_start;
        column_slice.row_count = row_count;
        column_slice.col_start = 0;
        column_slice.col_count = cols;
        column_slice.inner_is_presliced = true;
        registerDerivedMetadata(name, result, WeightDerivationKind::RowSlice, column_slice, device);

        LOG_TRACE("[WeightManager] Device " << device.to_string()
                                            << " (rank " << assignment.local_rank << "/" << tp_config->worldSize() << ")"
                                            << " column-parallel " << name
                                            << " [" << total_rows << ", " << cols << "]"
                                            << " -> rows [" << row_start << ", " << (row_start + row_count) << ")"
                                            << " = " << row_count << " rows");

        return result;
    }

    std::vector<GDNHeadSpan> WeightManager::fusedQKVSourceSpans(
        const std::string &name, size_t total_rows, const DeviceShardingAssignment &assignment,
        const std::shared_ptr<const TensorParallelConfig> &tp_config) const
    {
        const ModelDimensions model_dimensions{
            .n_heads = model_n_heads_,
            .n_kv_heads = model_n_kv_heads_,
            .head_dim = model_head_dim_,
            .gdn_n_k_heads = gdn_n_k_heads_,
            .gdn_n_v_heads = gdn_n_v_heads_,
            .gdn_d_state = gdn_d_state_,
        };
        WeightSlicer slicer(
            model_dimensions, sharding_config_, tp_config);
        auto slice = slicer.computeFusedQKVSliceForAssignment(
            name, total_rows, assignment);
        if (!slice)
        {
            throw std::runtime_error(
                "[WeightManager] FusedQKV weight has no authenticated sub-block geometry: " +
                name);
        }

        std::vector<GDNHeadSpan> source_spans;
        source_spans.reserve(2u + slice->v.size());
        const auto append_source_span =
            [&source_spans, &name](size_t base, SliceSpec local)
        {
            if (local.empty() || local.start >
                                     static_cast<size_t>(
                                         std::numeric_limits<int>::max()) ||
                local.count > static_cast<size_t>(
                                  std::numeric_limits<int>::max()) ||
                base > static_cast<size_t>(
                           std::numeric_limits<int>::max()) -
                           local.start)
            {
                throw std::runtime_error(
                    "[WeightManager] FusedQKV source span exceeds the native row-slice contract: " +
                    name);
            }
            source_spans.push_back({
                .start = static_cast<int>(base + local.start),
                .count = static_cast<int>(local.count),
            });
        };
        append_source_span(0u, slice->q);
        append_source_span(slice->q_total, slice->k);
        for (const SliceSpec value_span : slice->v)
        {
            append_source_span(
                slice->q_total + slice->k_total, value_span);
        }
        return source_spans;
    }

    std::shared_ptr<TensorBase> WeightManager::loadFusedQKVColumnParallel(
        const std::string &name,
        DeviceId device,
        const DeviceShardingAssignment &assignment,
        const std::vector<size_t> &dimensions)
    {
        const auto tp_config = tensorParallelConfig();
        const size_t total_rows = dimensions[0];
        const size_t cols = dimensions[1];
        if (!tp_config)
            throw std::runtime_error(
                "[WeightManager] FusedQKV assignment loading requires TensorParallelConfig");
        const int world_size = tp_config->worldSize();
        const int rank = assignment.local_rank;

        const auto source_spans = fusedQKVSourceSpans(name, total_rows, assignment, tp_config);
        auto result_tensor = loadNativeRowSpanConcat(
            name, device, source_spans, cols, assignment.local_rank);
        const size_t total_out_rows = result_tensor->shape()[0];
        const TensorType native_type = result_tensor->native_type();

        // Wrap in TensorSlice with column-parallel metadata so downstream TP
        // logic (allreduce detection, etc.) works correctly
        auto meta = SliceMetadata::forColumnParallel(
            total_rows, cols, assignment.local_rank, tp_config->worldSize(),
            true /* inner_is_presliced */);

        auto result = std::make_shared<TensorSlice>(
            std::move(result_tensor), meta);

        WeightSliceSpec fused_slice;
        fused_slice.source_rows = total_rows;
        fused_slice.source_cols = cols;
        fused_slice.row_start = 0;
        fused_slice.row_count = total_out_rows;
        fused_slice.col_start = 0;
        fused_slice.col_count = cols;
        fused_slice.inner_is_presliced = true;
        registerDerivedMetadata(name, result, WeightDerivationKind::FusedSubblockConcat, fused_slice, device);

        LOG_TRACE("[WeightManager] Device " << device.to_string()
                                            << " (rank " << rank << "/" << world_size << ")"
                                            << " fused-QKV column-parallel " << name
                                            << " [" << total_rows << ", " << cols << "]"
                                            << " -> [" << total_out_rows << ", " << cols << "]"
                                            << " (" << static_cast<int>(native_type) << " native)"
                                            << " (typed assignment layout)");

        return result;
    }

    std::shared_ptr<TensorBase> WeightManager::loadRowParallelWeight(
        const std::string &name,
        DeviceId device,
        const DeviceShardingAssignment &assignment,
        const std::vector<size_t> &dimensions)
    {
        const auto tp_config = tensorParallelConfig();
        size_t total_rows = dimensions[0];
        size_t cols = dimensions[1];
        size_t row_start = 0;
        size_t row_count = 0;

        // Use config-based dimension type to determine row slicing
        if (!computeSliceBoundaries(name, total_rows, assignment, *tp_config, row_start, row_count))
        {
            throw std::runtime_error("[WeightManager] Failed to compute slice boundaries for row-parallel: " + name);
        }

        auto slice_tensor = (device.is_cpu() ? loadCPUWeightRange(name,device,WeightDerivationKind::RowSlice,
                WeightSliceSpec{.row_start=row_start,.row_count=(row_start + row_count)-(row_start),.inner_is_presliced=true},{},assignment.local_rank) : loader_.loadTensorRowSlice(
            name, row_start, row_start + row_count, device, WeightPrecision::NATIVE));

        if (!slice_tensor)
        {
            throw std::runtime_error("[WeightManager] Failed to load row slice for row-parallel: " + name);
        }

        auto meta = SliceMetadata::forRowParallel(
            total_rows, cols, assignment.local_rank, tp_config->worldSize(),
            true /* inner_is_presliced */);

        auto result = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);

        WeightSliceSpec row_slice;
        row_slice.source_rows = total_rows;
        row_slice.source_cols = cols;
        row_slice.row_start = row_start;
        row_slice.row_count = row_count;
        row_slice.col_start = 0;
        row_slice.col_count = cols;
        row_slice.inner_is_presliced = true;
        registerDerivedMetadata(name, result, WeightDerivationKind::RowSlice, row_slice, device);

        LOG_TRACE("[WeightManager] Device " << device.to_string()
                                            << " (rank " << assignment.local_rank << "/" << tp_config->worldSize() << ")"
                                            << " row-parallel " << name
                                            << " [" << total_rows << ", " << cols << "]"
                                            << " -> rows [" << row_start << ", " << (row_start + row_count) << ")"
                                            << " = " << row_count << " rows (needs allreduce)");

        return result;
    }

    std::shared_ptr<TensorBase> WeightManager::loadInputParallelWeight(
        const std::string &name,
        DeviceId device,
        const DeviceShardingAssignment &assignment,
        const std::vector<size_t> &dimensions)
    {
        const auto tp_config = tensorParallelConfig();
        size_t rows = dimensions[0];
        size_t total_cols = dimensions[1];

        if (tp_config && tp_config->worldSize() > 1 &&
            gdnValueElementsPerHead(name, total_cols) > 0)
        {
            const GDNHeadAssignment head_assignment =
                gdnHeadAssignmentFor(assignment, *tp_config);
            return loadGDNValueColumns(
                name, device, head_assignment, assignment.local_rank,
                tp_config->worldSize(), dimensions);
        }

        // Use config-based dimension type to determine column slicing
        size_t col_start = 0;
        size_t col_count = 0;

        if (!computeSliceBoundaries(name, total_cols, assignment, *tp_config, col_start, col_count))
        {
            throw std::runtime_error("[WeightManager] Failed to compute slice boundaries for input-parallel: " + name);
        }

        auto slice_tensor = (device.is_cpu() ? loadCPUWeightRange(name,device,WeightDerivationKind::ColumnSlice,
                WeightSliceSpec{.col_start=col_start,.col_count=(col_start + col_count)-(col_start),.inner_is_presliced=true},{},assignment.local_rank) : loader_.loadTensorColumnSlice(
            name, col_start, col_start + col_count, device, WeightPrecision::NATIVE));

        if (!slice_tensor)
            throw std::runtime_error("[WeightManager] Native input-parallel source cannot represent logical columns [" +
                std::to_string(col_start) + "," + std::to_string(col_start + col_count) + ") for " + name);

        // Input-parallel uses row-parallel metadata (mathematically similar but with column slicing)
        auto meta = SliceMetadata::forRowParallel(
            rows, total_cols, assignment.local_rank, tp_config->worldSize(),
            true /* inner_is_presliced */);

        auto result = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);

        WeightSliceSpec input_slice;
        input_slice.source_rows = rows;
        input_slice.source_cols = total_cols;
        input_slice.row_start = 0;
        input_slice.row_count = rows;
        input_slice.col_start = col_start;
        input_slice.col_count = col_count;
        input_slice.inner_is_presliced = true;
        registerDerivedMetadata(name, result, WeightDerivationKind::ColumnSlice, input_slice, device);

        LOG_TRACE("[WeightManager] Device " << device.to_string()
                                            << " (rank " << assignment.local_rank << "/" << tp_config->worldSize() << ")"
                                            << " input-parallel " << name
                                            << " [" << rows << ", " << total_cols << "]"
                                            << " -> cols [" << col_start << ", " << (col_start + col_count) << ")"
                                            << " = " << col_count << " cols (needs allreduce)");

        return result;
    }

    // ========================================================================
    // Main sharding entry point
    // ========================================================================

    std::shared_ptr<TensorBase> WeightManager::getShardedWeightForAssignment(
        const std::string &name,
        DeviceId device,
        const DeviceShardingAssignment &assignment,
        int layer_idx)
    {
        const auto tp_config = tensorParallelConfig();
        if(device.is_cpu()) {
            if(!tp_config)
                throw std::invalid_argument("CPU assignment source requires its original retained TP declaration");
            const auto &original=tp_config->forRank(assignment.local_rank);
            if(original.device!=assignment.device || original.local_rank!=assignment.local_rank ||
                original.head_start!=assignment.head_start || original.head_count!=assignment.head_count ||
                original.kv_head_start!=assignment.kv_head_start || original.kv_head_count!=assignment.kv_head_count ||
                original.d_ff_start!=assignment.d_ff_start || original.d_ff_count!=assignment.d_ff_count ||
                original.vocab_start!=assignment.vocab_start || original.vocab_count!=assignment.vocab_count ||
                original.work_fraction!=assignment.work_fraction)
                throw std::invalid_argument("CPU assignment source changed its original retained TP definition");
        }
        // Get sharding mode for this weight before consulting the per-device
        // cache. Older/preload paths may have populated the cache with a plain
        // device clone; for TP-sharded weights graph builders must see the
        // TensorSlice wrapper so row/column-parallel metadata is preserved.
        ShardingMode mode = getShardingMode(name);
        if(device.is_cpu() && isRoutedExpertRole(inferWeightRole(name)))
            throw std::logic_error("CPU expert source access requires its complete original prepared registry bank, not a raw 3D parent");

        std::optional<std::vector<int>> expected_expert_ids;
        if (mode == ShardingMode::EXPERT_ID_APPORTIONED)
        {
            if (!tp_config)
            {
                throw std::runtime_error(
                    "[WeightManager] Local routed-expert apportionment requires a tensor-parallel configuration for " +
                    name);
            }
            expected_expert_ids = expectedRoutedExpertIds(
                name,
                assignment.local_rank,
                tp_config->worldSize(),
                layer_idx);
        }

        // The retained assignment owns host-staged shard identity. CPU NUMA
        // participants and GPU sources explicitly staged onto CPU share a host
        // address space; actual GPU storage remains device-keyed.
        const auto cache_key = DeviceWeightCacheKey::forAssignment(device, name, assignment, tp_config);
        if(device.is_cpu()) {
            std::shared_ptr<TensorBase> cached;
            std::shared_ptr<PreparedWeightStore> original_store;
            {
                std::lock_guard snapshot(cache_mutex_);
                original_store=prepared_weight_store_;
                const auto entry=per_device_cache_.find(cache_key);
                if(entry!=per_device_cache_.end())cached=entry->second;
            }
            if(cached) {
                if(!hasOriginalCPUFinalOwner(cached,device,original_store))
                    throw std::logic_error("CPU assignment cache lost its original completed Store owner");
                std::lock_guard verify(cache_mutex_);
                if(prepared_weight_store_!=original_store)
                    throw std::logic_error("CPU assignment cache changed its original Store while authenticating");
                return cached;
            }
        }
        if(!device.is_cpu())
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto it = per_device_cache_.find(cache_key);
            if (it != per_device_cache_.end() && !device.is_cpu())
            {
                if (expected_expert_ids)
                {
                    if (isCachedFullRoutedExpertSource(name, it->second))
                    {
                        /*
                         * A pre-topology full source and a TP ExpertSlice used
                         * the historical device/name cache key. The metadata
                         * proves this is the complete immutable source, so
                         * replace that binding with the authoritative slice.
                         */
                        per_device_cache_.erase(it);
                    }
                    else
                    {
                        validateCachedRoutedExpertSlice(
                            name, it->second, *expected_expert_ids, cache_key.describe());
                        return it->second;
                    }
                }
                else
                {
                    TensorSlice *slice = dynamic_cast<TensorSlice *>(it->second.get());
                    const bool slice_matches_assignment =
                        slice &&
                        slice->metadata().rank == assignment.local_rank &&
                        tp_config &&
                        slice->metadata().world_size == tp_config->worldSize();
                    const bool cache_matches_mode =
                        (mode == ShardingMode::REPLICATE) ||
                        (mode == ShardingMode::ROW_PARALLEL &&
                         slice_matches_assignment && slice->is_row_parallel()) ||
                        (mode == ShardingMode::INPUT_PARALLEL &&
                         slice_matches_assignment && slice->is_row_parallel()) ||
                        (mode == ShardingMode::COLUMN_PARALLEL &&
                         slice_matches_assignment && slice->is_column_parallel());

                    if (cache_matches_mode)
                    {
                        return it->second;
                    }

                    throw std::runtime_error(
                        "[WeightManager] Fatal per-device weight cache mode mismatch for " +
                        cache_key.describe() + ": requested_mode=" +
                        std::to_string(static_cast<int>(mode)) +
                        " requested_rank=" + std::to_string(assignment.local_rank) +
                        " requested_world_size=" +
                        std::to_string(tp_config ? tp_config->worldSize() : 0) +
                        " cached_rank=" +
                        std::to_string(slice ? slice->metadata().rank : -1) +
                        " cached_world_size=" +
                        std::to_string(slice ? slice->metadata().world_size : 0));
                }
            }
        }

        LOG_TRACE("[WeightManager] getShardedWeightForAssignment: " << name
                                                                    << " device=" << device.to_string()
                                                                    << " local_rank=" << assignment.local_rank
                                                                    << " mode=" << static_cast<int>(mode));

        std::shared_ptr<TensorBase> result;

        switch (mode)
        {
        case ShardingMode::REPLICATE:
        {
            if (device.is_cpu()) { result=getReplicatedWeight(name,device,assignment.local_rank); break; }
            /*
             * A replicated weight has one of two ownership contracts:
             *
             * 1. Norms, biases, and other raw runtime operands receive a
             *    distinct TensorBase/device allocation per participant.
             * 2. Preparation-only sources remain immutable on the host while
             *    the GEMM or expert subsystem creates independent device-owned
             *    packed storage.
             *
             * Decide that contract before cloning. Marking a tensor
             * HOST_RESIDENT later in preloadForDevices() is too late: by then
             * the full payload has already been copied once per GPU. Qwen3.6
             * expert tensors make that ordering bug tens of gigabytes.
             */
            std::shared_ptr<TensorBase> cached;
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto cached_it = cache_.find(name);
                if (cached_it != cache_.end())
                    cached = cached_it->second;
            }

            if (!cached)
            {
                auto original = getReplicatedWeight(name, device);
                if (original)
                {
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    auto [cached_it, inserted] = cache_.emplace(name, original);
                    if (!inserted && !cached_it->second)
                        cached_it->second = original;
                    cached = cached_it->second;
                }
            }

            const bool share_preparation_source =
                cached &&
                device.is_gpu() &&
                isGpuPreparationSource(name, *cached);
            if (share_preparation_source)
            {
                cached->setHostResident();
            }

            if (cached && cached->isHostResident())
            {
                result = cached;
                if (share_preparation_source &&
                    PerfStatsCollector::isDomainEnabled("weight_loading"))
                {
                    PerfStatsCollector::addCounter(
                        "weight_loading",
                        "gpu_avoided_replicated_host_clone_bytes",
                        static_cast<double>(cached->size_bytes()),
                        "preload",
                        device.to_string(),
                        {{"weight_role", toString(inferWeightRole(name))}});
                }
                LOG_TRACE("[WeightManager] Sharing host-resident REPLICATE weight: " << name
                                                                                     << " for " << device.to_string()
                                                                                     << " (" << (result->size_bytes() / (1024 * 1024)) << " MB)");
            }
            else if (cached)
            {
                result = cloneTensorForDevice(name, cached, device);
            }
            if (result)
            {
                LOG_TRACE("[WeightManager] Device " << device.to_string()
                                                    << " gets REPLICATED weight: " << name);
            }
            break;
        }

        case ShardingMode::COLUMN_PARALLEL:
        {
            // For tied embeddings: if output.weight is missing, substitute token_embd.weight
            auto dims_opt = loader_.getTensorShape(name);
            if ((!dims_opt || dims_opt->empty()) && name == "output.weight")
            {
                auto embd_dims = loader_.getTensorShape("token_embd.weight");
                if (embd_dims && embd_dims->size() == 2)
                {
                    LOG_TRACE("[WeightManager] Device " << device.to_string()
                                                        << " output.weight not in GGUF — using tied embedding "
                                                        << "token_embd.weight as column-parallel LM head");

                    // Vocab-dimension slicing: use assignment's vocab_start/vocab_count
                    size_t total_rows = (*embd_dims)[0];
                    size_t cols = (*embd_dims)[1];
                    size_t row_start = assignment.vocab_start;
                    size_t row_count = assignment.vocab_count;

                    auto slice_tensor = (device.is_cpu() ? loadCPUWeightRange("token_embd.weight",device,WeightDerivationKind::RowSlice,
                WeightSliceSpec{.row_start=row_start,.row_count=(row_start + row_count)-(row_start),.inner_is_presliced=true},{},assignment.local_rank) : loader_.loadTensorRowSlice(
                        "token_embd.weight", row_start, row_start + row_count, device, WeightPrecision::NATIVE));
                    if (!slice_tensor)
                    {
                        throw std::runtime_error("[WeightManager] Failed to load tied embedding row slice");
                    }

                    auto meta = SliceMetadata::forColumnParallel(
                        total_rows, cols, assignment.local_rank, tp_config->worldSize(),
                        true /* inner_is_presliced */);
                    result = std::make_shared<TensorSlice>(std::move(slice_tensor), meta);

                    WeightSliceSpec tied_slice;
                    tied_slice.source_rows = total_rows;
                    tied_slice.source_cols = cols;
                    tied_slice.row_start = row_start;
                    tied_slice.row_count = row_count;
                    tied_slice.col_start = 0;
                    tied_slice.col_count = cols;
                    tied_slice.inner_is_presliced = true;
                    registerDerivedMetadata(name, result, WeightDerivationKind::TiedAlias, tied_slice, device);

                    LOG_TRACE("[WeightManager] Device " << device.to_string()
                                                        << " tied embedding LM head"
                                                        << " [" << total_rows << ", " << cols << "]"
                                                        << " -> rows [" << row_start << ", " << (row_start + row_count) << ")"
                                                        << " = " << row_count << " rows");
                    break;
                }
            }
            if (!dims_opt || dims_opt->empty())
            {
                // Tensor doesn't exist in GGUF — return nullptr so the caller
                // (graph config builder) can probe for layer-type-specific weights.
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for column-parallel: " << name);
                return nullptr;
            }
            const auto &dims = *dims_opt;

            if (dims.size() == 1)
            {
                result = loadColumnParallel1DBias(name, device, assignment, dims);
            }
            else if (dims.size() == 2)
            {
                result = loadColumnParallel2DWeight(name, device, assignment, dims);
            }
            else
            {
                throw std::runtime_error("[WeightManager] Invalid tensor for column-parallel (expected 1D or 2D): " + name);
            }
            break;
        }

        case ShardingMode::ROW_PARALLEL:
        {
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->empty())
            {
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for row-parallel: " << name);
                return nullptr;
            }
            if (dims_opt->size() != 2)
            {
                throw std::runtime_error("[WeightManager] Invalid tensor for row-parallel (expected 2D): " + name);
            }
            result = loadRowParallelWeight(name, device, assignment, *dims_opt);
            break;
        }

        case ShardingMode::INPUT_PARALLEL:
        {
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->empty())
            {
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for input-parallel: " << name);
                return nullptr;
            }
            if (dims_opt->size() != 2)
            {
                throw std::runtime_error("[WeightManager] Invalid tensor for input-parallel (expected 2D): " + name);
            }
            result = loadInputParallelWeight(name, device, assignment, *dims_opt);
            break;
        }

        case ShardingMode::EXPERT_ID_APPORTIONED:
        {
            // For LocalTP expert-ID apportionment, split the 3D tensor's expert axis.
            auto dims_opt = loader_.getTensorShape(name);
            if (!dims_opt || dims_opt->empty())
            {
                LOG_DEBUG("[WeightManager] Tensor not in GGUF for expert-id-apportioned loading: " << name);
                return nullptr;
            }
            if (dims_opt->size() != 3)
            {
                throw std::runtime_error("[WeightManager] Expert-ID apportionment requires a 3D tensor, got " +
                                         std::to_string(dims_opt->size()) + "D for: " + name);
            }
            const auto &dims = *dims_opt;
            const size_t ne2 = dims[2]; // num_experts
            const int local_ws = tp_config->worldSize();
            const int resolved_layer_idx =
                layer_idx >= 0 ? layer_idx : inferWeightLayer(name);
            if (resolved_layer_idx < 0)
                throw std::runtime_error(
                    "[WeightManager] Cannot derive a layer index for routed-expert weight " + name);
            const std::vector<int> selected_experts =
                routed_expert_ownership::expertIdsForParticipant(
                    static_cast<int>(ne2),
                    local_ws,
                    assignment.local_rank,
                    resolved_layer_idx,
                    routed_expert_assignment_.owner_order);
            recordRoutedExpertWeightSelection(
                selected_experts,
                routed_expert_assignment_.owner_order,
                assignment.local_rank,
                resolved_layer_idx,
                device);
            std::vector<size_t> selected_expert_indices;
            selected_expert_indices.reserve(selected_experts.size());
            for (const int expert_id : selected_experts)
                selected_expert_indices.push_back(static_cast<size_t>(expert_id));

            result = loader_.loadTensorExpertSelection(
                name,
                selected_expert_indices,
                device,
                WeightPrecision::NATIVE);

            if (!result)
            {
                throw std::runtime_error("[WeightManager] Failed to load expert slice for: " + name);
            }

            WeightSliceSpec expert_slice;
            expert_slice.source_rows = dims[0];
            expert_slice.source_cols = dims[1];
            expert_slice.row_start = 0;
            expert_slice.row_count = dims[0];
            expert_slice.col_start = 0;
            expert_slice.col_count = dims[1];
            expert_slice.expert_start = selected_experts.empty()
                                              ? 0u
                                              : static_cast<size_t>(selected_experts.front());
            expert_slice.expert_count = selected_experts.size();
            expert_slice.expert_ids = selected_experts;
            expert_slice.inner_is_presliced = true;
            registerDerivedMetadata(name, result, WeightDerivationKind::ExpertSlice, expert_slice, device);

            LOG_TRACE("[WeightManager] Device " << device.to_string()
                                                << " expert-id-apportioned " << name
                                                << " [" << dims[0] << ", " << dims[1] << ", " << ne2
                                                << "] -> " << selected_experts.size()
                                                << "/" << ne2 << " explicit experts"
                                                << " owner_order="
                                                << routedExpertOwnerOrderToString(
                                                       routed_expert_assignment_.owner_order));
            break;
        }

        default:
            throw std::runtime_error("[WeightManager] Unknown sharding mode for: " + name);
        }

        // Cache indexes borrow final handles; source/Store validation never
        // runs under the manager mutex (Store lookup uses independent locks).
        if(result && device.is_cpu()) {
            const auto original_store=preparedWeightStoreIfInitialized();
            if(!hasOriginalCPUFinalOwner(result,device,original_store))
                throw std::logic_error("CPU assignment publication has no original completed Store owner");
            std::shared_ptr<TensorBase> prior;
            {
                std::lock_guard publish(cache_mutex_);
                if(prepared_weight_store_!=original_store)
                    throw std::logic_error("CPU assignment publication changed its original Store");
                const auto [entry,inserted]=per_device_cache_.emplace(cache_key,result);
                if(!inserted)prior=entry->second;
            }
            if(prior) {
                if(!hasOriginalCPUFinalOwner(prior,device,original_store))
                    throw std::logic_error("CPU assignment publication replaced its original Store owner");
                const auto unwrap=[](const TensorBase *value) {
                    while(const auto *view=dynamic_cast<const TensorSlice *>(value))value=view->inner();
                    return value;
                };
                if(unwrap(prior.get())!=unwrap(result.get()))
                    throw std::logic_error("CPU assignment publication replaced its original final root");
                return prior;
            }
            result->setDebugName(name);
        } else if(result) {
            std::lock_guard publish(cache_mutex_);
            per_device_cache_[cache_key]=result;
            result->setDebugName(name);
        }

        return result;
    }

    // =========================================================================
    // Layer Range Filtering for Pipeline Parallelism
    // =========================================================================

    void WeightManager::setHasEmbedding(bool has_embedding)
    {
        has_embedding_ = has_embedding;
        LOG_DEBUG("[WeightManager] has_embedding = " << (has_embedding ? "true" : "false"));
    }

    void WeightManager::setHasLmHead(bool has_lm_head)
    {
        has_lm_head_ = has_lm_head;
        LOG_DEBUG("[WeightManager] has_lm_head = " << (has_lm_head ? "true" : "false"));
    }

    bool WeightManager::isWeightInLayerRange(const std::string &name) const
    {
        // If no layer range is set, include all weights
        if (!has_layer_range_)
        {
            return true;
        }
        return isWeightInLayerRange(name, layer_first_, layer_last_, has_embedding_, has_lm_head_);
    }

    bool WeightManager::isWeightInLayerRange(const std::string &name,
                                             int first_layer, int last_layer,
                                             bool has_embedding, bool has_lm_head) const
    {
        // Handle special weights (embedding, output norm, LM head)
        if (name == "token_embd.weight")
        {
            if (has_embedding)
            {
                return true;
            }
            // Allow embedding through for the LM head stage ONLY if the model
            // uses tied embeddings (no separate output.weight in GGUF).
            if (has_lm_head)
            {
                auto output_shape = loader_.getTensorShape("output.weight");
                bool tied = !output_shape || output_shape->empty();
                return tied;
            }
            return false;
        }
        if (name == "output_norm.weight" || name == "output.weight")
        {
            return has_lm_head;
        }

        // Extract layer index from "blk.N.xxx" pattern
        // Pattern: blk.{layer_idx}.{component}.weight
        static const std::regex layer_pattern(R"(blk\.(\d+)\.)");
        std::smatch match;
        if (std::regex_search(name, match, layer_pattern))
        {
            int layer_idx = std::stoi(match[1].str());
            const std::string sidecar_prefix = "blk." + std::to_string(layer_idx) + ".";
            if (loader_.hasTensor(sidecar_prefix + "nextn.eh_proj.weight"))
            {
                // Qwen3.6 encodes MTP as a trailing blk.N sidecar. The main graph
                // layer range intentionally excludes that block. Pipeline stages
                // treat the sidecar like final norm and LM head: only the stage
                // that can produce terminal hidden/logits should load and bind it.
                return has_lm_head;
            }
            // Layer range is [first, last) - first inclusive, last exclusive
            return layer_idx >= first_layer && layer_idx < last_layer;
        }

        // Unknown weight pattern - include by default (e.g., custom weights)
        LOG_DEBUG("[WeightManager] Unknown weight pattern, including: " << name);
        return true;
    }

} // namespace llaminar2
