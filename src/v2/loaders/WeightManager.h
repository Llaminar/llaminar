/**
 * @file WeightManager.h
 * @brief Weight distribution and caching for MPI-aware pipelines
 *
 * Manages model weight tensors with support for different distribution strategies:
 * - REPLICATED: Full copy per rank (default, simple)
 * - SHARDED: Partition across ranks (memory efficient, requires Allreduce)
 * - INTERLEAVED: NUMA-aware global allocation (shared memory optimization)
 *
 * For SHARDED strategy, weights are partitioned based on model-specific
 * sharding configuration defined in the model's schema (e.g., Qwen2Schema).
 * This keeps WeightManager generic and model-agnostic.
 *
 * @see GraphSchema.h for WeightShardingConfig structure
 * @see models/qwen/Qwen2Schema.h for Qwen2-specific sharding patterns
 *
 * @author David Sanftenberg
 */

#pragma once

#include "ExpertGemmRegistry.h"
#include "IModelLoader.h"
#include "WeightLifecycleTrace.h"
#include "WeightMetadataRegistry.h"
#include "WeightPlan.h"
#include "WeightPlacementMap.h"
#include "WeightManagerConfig.h"
#include "PreparedWeightAdmission.h"
#include "PreparedDeviceAllocationLedger.h"
#include "MmapReclaimLifecycle.h"
#include "../execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "../backends/DeviceId.h"
#include "../config/TensorParallelConfig.h"
#include "../config/GDNHeadAssignment.h"
#include "../execution/local_execution/graph/GraphSchema.h"
#include "IWeightManager.h"
#include "../utils/MPIContext.h"
#include "../tensors/Tensors.h"
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <vector>

namespace llaminar2
{
    class ExpertWeightPayloadProvider;
    class PhysicalMemoryAuthority;

    // WeightDistributionStrategy and ShardingMode are now in WeightTypes.h
    // (included transitively via WeightManagerConfig.h → WeightTypes.h)

    class PreparedWeightStore;
    class MoEExpertOverlayRuntimePlan;
    struct MoEExpertOverlayExecutionPlan;
    class WeightLoadProgress;

    /**
     * @brief Weight manager with distribution strategy and caching
     *
     * Sits between ModelContext and pipelines:
     * - Loads weights from ModelLoader
     * - Applies distribution strategy (replicated/sharded/interleaved)
     * - Caches loaded tensors for reuse
     * - Coordinates across MPI ranks
     *
     * Usage:
     *   auto mgr = std::make_shared<WeightManager>(loader, mpi_ctx);
     *   auto wq = mgr->getWeightForDevice("blk.0.attn_q.weight", device_idx);
     *   auto wk = mgr->getWeightForDevice("blk.0.attn_k.weight", device_idx);
     */
    class WeightManager : public IWeightManager
    {
    public:
        /**
         * @brief Construct weight manager
         *
         * @param loader Model loader interface for tensor loading (GGUF, mock, etc.)
         * @param mpi_ctx MPI context for rank coordination (nullptr = single rank)
         * @param placement_map Fine-grained weight→device mapping (nullptr = default to device 0)
         * @param strategy Distribution strategy (default: REPLICATED)
         * @param weight_precision How weights are loaded (NATIVE, CONVERT_TO_FP32, etc.)
         */
        WeightManager(IModelLoader &loader,
                      std::shared_ptr<IMPIContext> mpi_ctx = nullptr,
                      std::shared_ptr<WeightPlacementMap> placement_map = nullptr,
                      WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED,
                      WeightPrecision weight_precision = WeightPrecision::NATIVE);

        /**
         * @brief Join any submitted mapping reclaim before loader state is destroyed.
         *
         * The owned worker borrows this manager and its loader, so teardown owns
         * an explicit completion edge rather than relying on member destruction
         * order alone.
         */
        ~WeightManager() override;

        /**
         * @brief Get weight tensor for a specific device (device-isolated instance)
         *
         * Loads from GGUF if not cached, applies distribution strategy.
         * For multi-device scenarios (LOCAL TP), each device needs its own tensor
         * instance to track coherence state independently. This method:
         * - Returns the original tensor for the first device that requests it
         * - Creates and caches a clone for subsequent devices
         * - Clones are uploaded to GPU independently, avoiding race conditions
         *
         * @param name GGUF tensor name (e.g., "token_embd.weight", "blk.0.attn_q.weight")
         * @param device Target device for this tensor instance (default: CPU)
         * @param layer_idx Optional layer index for placement map lookup
         * @return Device-specific tensor instance, or nullptr on error
         */
        std::shared_ptr<TensorBase> getWeightForDevice(const std::string &name, DeviceId device = DeviceId::cpu(), int layer_idx = -1) override;

        /**
         * @brief Pre-load and upload weights for multiple devices
         *
         * For LOCAL tensor parallelism scenarios, this method:
         * 1. Creates device-specific clones of all cached weights
         * 2. Uploads each clone to its target device
         * 3. Caches the device-specific tensors for getWeightForDevice()
         *
         * This MUST be called BEFORE creating device runners.
         *
         * @param devices List of devices that will use the weights
         * @return true if all weights were pre-loaded successfully
         */
        bool preloadForDevices(const std::vector<DeviceId> &devices) override;

        // =========================================================================
        // Weight Lifecycle (single entry points)
        // =========================================================================

        // =========================================================================
        // Weight Preparation (repack + upload, NO host release)
        // =========================================================================

        /**
         * @brief Prepare weights for a single device (pack + upload, no release)
         *
         * Runs GEMM packing, non-GEMM upload, and embedding preparation for
         * all weights currently in cache. Does NOT release host copies.
         * Call releaseAllHostWeightData() separately after ALL devices are prepared.
         *
         * @param device Target device
         * @return true on success
         */
        bool prepareWeightsForDevice(DeviceId device) override;

        /**
         * @brief Prepare weights for a device using graph-frozen bindings.
         *
         * This is the preferred graph-build path: prepared GEMM handles are
         * registered under the exact WeightBinding ids that graph stages will
         * reference, including PP/TP slices and tied aliases. Set
         * include_expert_jobs=false when an explicit expert-cache preparation
         * pass has already populated ExpertGemmRegistry for this device.
         */
        bool prepareWeightsForDevice(
            const FrozenModelWeightSet &frozen_weights,
            DeviceId device,
            bool include_expert_jobs = true);

        /**
         * @brief Prepare the overlay experts owned by one graph participant.
         *
         * Accelerator preparation consumes only the exact frozen bindings owned
         * by @p target_device.  It never scans another participant's requests and
         * never substitutes mutable process-wide cache tensors for a missing
         * frozen binding.  CPU participants likewise prepare only their explicit
         * device-scoped requests.  This keeps LocalTP graph construction
         * participant-local and prevents duplicate cross-device repack work.
         *
         * @param runtime_plan Declarative rank/domain overlay policy.
         * @param target_device Device owned by the calling graph participant.
         * @param frozen_weights Immutable graph bindings for that participant;
         *        mandatory when the scoped plan contains accelerator requests.
         * @param execution_plan Optional rank filter applied before device scope.
         * @param admission Whether to create exact prepared engines or adopt a
         *        complete set certified by the model-context reuse lifecycle.
         *        Certified adoption never falls back to source materialization.
         * @return true after all scoped requests have been prepared.
         */
        bool prepareMoEExpertOverlayWeights(
            const MoEExpertOverlayRuntimePlan &runtime_plan,
            DeviceId target_device,
            const FrozenModelWeightSet *frozen_weights = nullptr,
            const MoEExpertOverlayExecutionPlan *execution_plan = nullptr,
            PreparedWeightAdmission admission =
                PreparedWeightAdmission::AllocateCompleteSet);

        /**
         * @brief Prepare weights for a single device, filtered to a layer range
         *
         * Same as prepareWeightsForDevice(device) but only processes weights
         * belonging to the specified layer range. Used by Pipeline Parallelism
         * stages to avoid packing weights that belong to other stages.
         *
         * @param device Target device
         * @param first_layer First layer index (inclusive)
         * @param last_layer Last layer index (exclusive)
         * @param has_embedding Whether this stage owns the embedding table
         * @param has_lm_head Whether this stage owns the LM head
         * @return true on success
         */
        bool prepareWeightsForDevice(
            DeviceId device,
            int first_layer, int last_layer,
            bool has_embedding, bool has_lm_head) override;

        // =========================================================================
        // Weight Lifecycle (convenience entry points)
        // =========================================================================

        /**
         * @brief Complete weight lifecycle for a single device
         *
         * Convenience method: prepareWeightsForDevice() + releaseAllHostWeightData().
         * Use when there is only ONE device and no PP sharing.
         *
         * @param device Target device
         * @return true on success
         */
        bool finalizeForDevice(DeviceId device) override;

        /**
         * @brief Complete weight lifecycle for multiple LOCAL TP devices
         *
         * Runs the full multi-device sequence:
         * 1. Clone and upload weights to all devices (preloadForDevices)
         * 2. Pack GEMM weights per device
         * 3. Release all host weight data (cache_ + per_device_cache_)
         *
         * @param devices List of devices
         * @param release_host_data If true, release host weight copies after
         *                          packing and upload. Set false for nested
         *                          TP-in-PP setups where a later PP stage on
         *                          a different device still needs host copies.
         * @param include_expert_jobs If false, routed MoE expert jobs are not
         *                            packed by this broad finalize pass.
         * @return true on success
         */
        bool finalizeForDevices(const std::vector<DeviceId> &devices,
                                bool release_host_data = true,
                                bool include_expert_jobs = true) override;

        // =========================================================================
        // Weight Packing and Preloading (folded from WeightPreloader)
        // =========================================================================

        /**
         * @brief Pack all GEMM weights for a target device
         *
         * Creates GEMM kernels for all GEMM weights and calls prepareWeights()
         * on each kernel. For GPU kernels, this uploads weights to device memory.
         * For CPU kernels, this is typically a no-op (packing is lazy).
         *
         * @param target_device Target device for weight packing
         * @param progress_cb Optional callback for progress reporting
         * @param release_raw_data If true, release raw tensor data after packing (CPU only)
         * @return true if all GEMM weights were packed successfully
         */
        bool packGemmWeights(
            DeviceId target_device,
            PreloadProgressCallback progress_cb = nullptr,
            bool release_raw_data = false,
            std::function<bool(const std::string &)> layer_filter = nullptr) override;

        /**
         * @brief Pack GEMM weights via GPU pipeline (LoadOrchestrator)
         *
         * Primary GPU weight loading path: single VRAM allocation, pipelined
         * H2D transfers, and GPU-side VNNI repack kernels. Used unconditionally
         * for all GPU devices. CPU devices use packGemmWeights() instead.
         *
         * @param target_device Target GPU device (ROCm or CUDA)
         * @param layer_filter Optional filter for specific layers
         * @return true if all GEMM weights were loaded successfully
         */
        bool packGemmWeightsViaPipeline(
            DeviceId target_device,
            std::function<bool(const std::string &)> layer_filter = nullptr,
            const FrozenModelWeightSet *frozen_weights = nullptr,
            bool include_expert_jobs = true,
            const MoEExpertOverlayPreparationPlan *overlay_preparation_plan = nullptr,
            std::optional<size_t> staging_budget_bytes_override = std::nullopt,
            PhysicalMemoryOwner persistent_owner =
                PhysicalMemoryOwner::PrimaryModelWeights);

        /**
         * @brief Upload all non-GEMM weights to GPU
         *
         * Non-GEMM weights (norms, embeddings, biases) don't need GEMM packing
         * but still need to be uploaded to GPU for kernel access.
         * This eliminates lazy upload overhead during inference.
         *
         * @param target_device Target GPU device
         * @return true if all non-GEMM weights were uploaded successfully
         */
        bool uploadNonGemmWeights(
            DeviceId target_device,
            std::function<bool(const std::string &)> layer_filter = nullptr) override;

        /**
         * @brief Release ALL host-side weight data after all GPU uploads are complete
         * @return Number of tensors whose host data was released
         */
        size_t releaseAllHostWeightData() override;

        /**
         * @brief Release host data for host-resident tensors after GPU kernels
         * have uploaded their own device copies (e.g., embedding repack+upload).
         *
         * Called after the first forward pass completes. Unlike releaseAllHostWeightData()
         * which retains host-resident tensors because they haven't been uploaded yet,
         * this method releases them because the GPU kernels have now created their own copies.
         * This method deliberately does not reclaim mmap pages: borrowed views
         * are retired by the asynchronous scheduleMmapReclaim() operation before
         * the shared mapping may be advised.
         *
         * @return Number of tensors whose host data was released
         */
        size_t releaseHostResidentWeightData() override;

        /**
         * @brief Release cached MoE expert parent tensors after eager expert packing.
         *
         * CPU MoE execution uses PreparedWeightStore-owned VNNI engines after graph
         * materialization. Dynamic transfer serializes those packed engines, so the
         * raw 3D *_exps.weight cache entries must not remain as a fallback source.
         *
         * @return Bytes of raw expert tensor data released
         */
        size_t releaseMoEExpertHostWeightData();

        /**
         * @brief Submit exactly-once mmap reclaim to the prestarted worker.
         *
         * Submission is non-blocking and performs no allocator, device-runtime,
         * or madvise work on the inference authority thread. The worker retires
         * remaining mmap host registrations before applying the loader's typed
         * backing-storage policy.
         *
         * @return Typed submission outcome.
         */
        MmapReclaimLifecycle::Submission scheduleMmapReclaim() override;

        /**
         * @brief Wait for reclaim before an allocation that depends on its RAM.
         *
         * This is an admission/teardown boundary, never an inference hot-path
         * operation.
         *
         * @return Terminal state and total bytes actually advised.
         */
        MmapReclaimLifecycle::Completion
        awaitMmapReclaimBeforeHostAllocation() override;

        /**
         * @brief Get statistics about preloaded weights
         *
         * @return Pair of (num_cpu_packed, num_gpu_packed)
         */
        std::pair<size_t, size_t> preloadStats() const override { return {num_cpu_packed_, num_gpu_packed_}; }

        /**
         * @brief Get current distribution strategy
         */
        WeightDistributionStrategy strategy() const override { return strategy_; }

        /**
         * @brief Get number of cached weights
         */
        size_t cacheSize() const;

        /**
         * @brief Log live host tensor bytes held by WeightManager caches.
         *
         * This is diagnostic accounting only. Borrowed views are counted as
         * objects but not as owned bytes.
         */
        void logHostMemorySummary(const char *context) const;

        /**
         * @brief Clear weight cache (frees memory)
         */
        void clearCache();

        /**
         * @brief Configure weight manager with unified config struct
         *
         * Replaces multi-step setter chain with a single call. Sets all members
         * directly and performs a single cache invalidation.
         *
         * @param config Configuration struct with all sharding/dimension settings
         */
        void configure(const WeightManagerConfig &config) override
        {
            std::lock_guard<std::mutex> lock(sharding_mode_cache_mutex_);

            // Sharding config
            if (config.hasShardingConfig())
            {
                sharding_config_ = config.sharding;
                has_sharding_config_ = true;
            }

            // Model dimensions
            if (config.hasModelDimensions())
            {
                model_n_heads_ = config.dimensions.n_heads;
                model_n_kv_heads_ = config.dimensions.n_kv_heads;
                model_head_dim_ = config.dimensions.head_dim;
                has_model_dimensions_ = true;
            }

            // GDN dimensions
            if (config.hasGDNDimensions())
            {
                gdn_n_k_heads_ = config.dimensions.gdn_n_k_heads;
                gdn_n_v_heads_ = config.dimensions.gdn_n_v_heads;
                gdn_d_state_ = config.dimensions.gdn_d_state;
                has_gdn_dimensions_ = true;
            }

            // Tensor parallel config
            if (config.hasTensorParallelConfig())
            {
                tp_config_ = config.tp_config;
            }

            if (!config.routed_expert_assignment.valid())
            {
                throw std::invalid_argument(
                    "WeightManager routed-expert assignment has invalid participant coordinates");
            }
            routed_expert_assignment_ = config.routed_expert_assignment;

            // Preprocessor
            if (config.preprocessor)
            {
                weight_preprocessor_ = config.preprocessor;
            }

            // Layer range (Pipeline Parallelism)
            if (config.hasLayerRange())
            {
                layer_first_ = config.layer_range->first;
                layer_last_ = config.layer_range->last;
                has_embedding_ = config.layer_range->has_embedding;
                has_lm_head_ = config.layer_range->has_lm_head;
                has_layer_range_ = true;
            }

            // Invalidate sharding mode cache (sharding config may have changed).
            // NOTE: We intentionally do NOT clear cache_ here. Sharding config
            // changes affect how weights are SLICED, not the raw loaded data.
            // Clearing cache_ would destroy weights loaded by previous PP stages
            // when the shared WeightManager is configured multiple times.
            sharding_mode_cache_.clear();
        }

        /**
         * @brief Set model-specific weight sharding configuration
         *
         * This should be called after construction with the config from
         * the model's schema factory (e.g., Qwen2SchemaFactory).
         *
         * @param config Weight sharding configuration from model schema
         */
        void setWeightShardingConfig(const WeightShardingConfig &config)
        {
            std::lock_guard<std::mutex> lock(sharding_mode_cache_mutex_);
            sharding_config_ = config;
            has_sharding_config_ = true;
            sharding_mode_cache_.clear(); // Invalidate cache
        }

        /**
         * @brief Set model head dimensions for FusedQKV sub-block computation
         *
         * Required for correct FusedQKVHeads sharding under GQA (n_kv_heads < n_heads).
         * Without this, FusedQKV weights fall back to simple equal row splitting.
         *
         * @param n_heads Number of query attention heads
         * @param n_kv_heads Number of key/value attention heads (GQA)
         * @param head_dim Dimension per attention head
         */
        void setModelDimensions(int n_heads, int n_kv_heads, int head_dim)
        {
            model_n_heads_ = n_heads;
            model_n_kv_heads_ = n_kv_heads;
            model_head_dim_ = head_dim;
            has_model_dimensions_ = true;
        }

        /**
         * @brief Set GDN (Gated Delta Net) dimensions for FusedQKV sub-block slicing
         *
         * GDN layers have asymmetric QKV: [Q(n_k*d) | K(n_k*d) | V(n_v*d)]
         * where n_k_heads != n_v_heads. The standard FA-based slicing doesn't
         * handle this layout. This provides the dimensions needed for correct
         * sub-block aware TP sharding.
         *
         * @param n_k_heads Number of key heads (ssm.group_count)
         * @param n_v_heads Number of value heads (ssm.time_step_rank)
         * @param d_state State dimension per head (ssm.state_size, used as d_k = d_v)
         */
        void setGDNDimensions(int n_k_heads, int n_v_heads, int d_state)
        {
            gdn_n_k_heads_ = n_k_heads;
            gdn_n_v_heads_ = n_v_heads;
            gdn_d_state_ = d_state;
            has_gdn_dimensions_ = true;
        }

        void setWeightPreprocessor(WeightPreprocessor preprocessor) override
        {
            weight_preprocessor_ = std::move(preprocessor);
        }

        WeightMetadataRegistry *weightMetadataRegistry() const { return weight_metadata_.get(); }

        /**
         * @brief Set the expert weight payload provider.
         *
         * Used by MoE stages to track which experts have been prepared/transferred.
         * After Phase 9 final, the provider is informational only — host release
         * no longer depends on per-expert preparation state because all experts
         * are prepared upfront at graph-build time.
         *
         * @param provider Model-context owned payload provider (may be nullptr)
         */
        void setExpertPayloadProvider(ExpertWeightPayloadProvider *provider)
        {
            expert_payload_provider_ = provider;
        }

        /**
         * @brief Check if raw host data is still required for a tensor.
         *
         * Uses WeightMetadataRegistry host policy to determine whether the tensor's
         * raw host data should be retained. Only RequiredForCPUExecution policy
         * causes retention. All other policies (including expert weights) allow
         * release after device preparation completes.
         *
         * @param tensor The tensor to check
         * @param key The cache key (canonical name) of the tensor
         * @return true if raw host data must be retained
         */
        bool hostDataRequired(const TensorBase *tensor, const std::string &key) const;

        FrozenModelWeightSet materialize(const WeightPlan &plan);

        /**
         * @brief Iterate over all cached weights (source cache).
         *
         * Calls the visitor for each (name, tensor) pair in the primary weight cache.
         * Thread-safe: acquires cache_mutex_ during iteration.
         *
         * @param visitor Callback receiving (canonical_name, raw_tensor_ptr)
         */
        void forEachWeight(std::function<void(const std::string &, TensorBase *)> visitor) const;

        /**
         * @brief Iterate ALL prepared tensors (cache_ + per_device_cache_)
         *
         * For PreparedWeightStore population in TP mode: iterates both the primary
         * cache and per-device cache (which holds TP-sliced tensors). This ensures
         * the store knows about every tensor pointer that stages will receive.
         *
         * @param visitor Callback receiving (canonical_name, raw_tensor_ptr)
         */
        void forEachPreparedWeight(std::function<void(const std::string &, TensorBase *)> visitor) const;

        /**
         * @brief Set layer range for LAYER_PARTITIONED strategy
         *
         * For Pipeline Parallelism, restricts which layer weights are loaded.
         * Weights outside this range will return nullptr from getWeightForDevice().
         *
         * Layer range is [first, last) - first is inclusive, last is exclusive.
         *
         * Special weights:
         * - has_embedding: if true, token embedding weight is loaded
         * - has_lm_head: if true, output norm and LM head weights are loaded
         *
         * @param first_layer First layer index (inclusive)
         * @param last_layer Last layer index (exclusive)
         * @param has_embedding True if this stage should load embedding
         * @param has_lm_head True if this stage should load output norm and LM head
         */
        void setLayerRange(int first_layer, int last_layer, bool has_embedding, bool has_lm_head)
        {
            layer_first_ = first_layer;
            layer_last_ = last_layer;
            has_embedding_ = has_embedding;
            has_lm_head_ = has_lm_head;
            has_layer_range_ = true;
            LOG_DEBUG("[WeightManager] Layer range set: layers [" << first_layer << ", " << last_layer
                                                                  << "), embedding=" << (has_embedding ? "yes" : "no")
                                                                  << ", lm_head=" << (has_lm_head ? "yes" : "no"));
        }

        /**
         * @brief Set layer range without changing global weight flags
         *
         * Use setHasEmbedding() and setHasLmHead() separately if needed.
         *
         * @param first_layer First layer index (inclusive)
         * @param last_layer Last layer index (exclusive, -1 = all remaining)
         */
        void setLayerRange(int first_layer, int last_layer)
        {
            layer_first_ = first_layer;
            layer_last_ = last_layer;
            has_layer_range_ = true;
            LOG_DEBUG("[WeightManager] Layer range set: layers [" << first_layer << ", " << last_layer << ")");
        }

        /**
         * @brief Check if a layer range is configured
         */
        bool hasLayerRange() const { return has_layer_range_; }

        /**
         * @brief Get configured layer range
         * @return Pair of (first_layer, last_layer_exclusive)
         */
        std::pair<int, int> layerRange() const { return {layer_first_, layer_last_}; }

        /**
         * @brief Check if this stage has embedding
         */
        bool hasEmbedding() const override { return has_embedding_; }

        /**
         * @brief Check if this stage has LM head
         */
        bool hasLMHead() const override { return has_lm_head_; }

        /**
         * @brief Set whether this weight manager should provide embedding weights
         * @param has_embedding True if embedding should be loaded
         */
        void setHasEmbedding(bool has_embedding);

        /**
         * @brief Set whether this weight manager should provide lm_head weights
         * @param has_lm_head True if output_norm and lm_head should be loaded
         */
        void setHasLmHead(bool has_lm_head);

        /**
         * @brief Set tensor parallelism configuration for proportional slicing
         *
         * When set, weight slicing uses the assignment from TensorParallelConfig
         * instead of the default 1/world_size calculation. This enables
         * heterogeneous tensor parallelism where devices get proportional work.
         *
         * @param config Tensor parallelism configuration with device assignments
         */
        void setTensorParallelConfig(std::shared_ptr<TensorParallelConfig> config)
        {
            tp_config_ = std::move(config);
            cache_.clear(); // Invalidate cache since slices may change
        }

        /**
         * @brief Get tensor parallelism configuration
         * @return Pointer to config, or nullptr if not set
         */
        const TensorParallelConfig *tensorParallelConfig() const { return tp_config_.get(); }

        /**
         * @brief Check if a weight is sharded (only valid for SHARDED strategy)
         * @param name Weight tensor name
         * @return true if weight is column or row parallel sharded
         */
        bool isWeightSharded(const std::string &name) const override;

        /**
         * @brief Get sharding mode for a weight
         * @param name Weight tensor name
         * @return ShardingMode indicating how weight is partitioned
         */
        ShardingMode getShardingMode(const std::string &name) const override;

        /**
         * @brief Check if a weight is used for GEMM operations
         *
         * Uses the model's sharding config if set, otherwise uses default patterns.
         *
         * @param name Weight tensor name
         * @return true if weight is a GEMM matrix (should release raw data after packing)
         */
        bool isGemmWeight(const std::string &name) const override;

        /**
         * @brief Pre-populate cache with shared tensors from a WeightViewSet
        // =========================================================================
        // Static utility methods (public for tensor slicing)
        // =========================================================================

        /**
         * @brief Slice tensor columns for column-parallel sharding
         *
         * For weight matrix [out_dim, in_dim], extracts [out_local, in_dim]
         * where out_local = out_dim / world_size for this rank.
         *
         * @param full_tensor Full weight tensor
         * @param rank MPI rank
         * @param world_size Total MPI ranks
         * @return Sliced tensor containing only this rank's columns
         */
        static std::shared_ptr<TensorBase> sliceColumns(
            const std::shared_ptr<TensorBase> &full_tensor,
            int rank, int world_size);

        /**
         * @brief Slice tensor rows for row-parallel sharding
         *
         * For weight matrix [out_dim, in_dim], extracts [out_dim, in_local]
         * where in_local = in_dim / world_size for this rank.
         *
         * @param full_tensor Full weight tensor
         * @param rank MPI rank
         * @param world_size Total MPI ranks
         * @return Sliced tensor containing only this rank's rows
         */
        static std::shared_ptr<TensorBase> sliceRows(
            const std::shared_ptr<TensorBase> &full_tensor,
            int rank, int world_size);

        /**
         * @brief Get the placement map (Phase 6: Multi-GPU support)
         *
         * @return Shared pointer to placement map (may be nullptr)
         */
        std::shared_ptr<WeightPlacementMap> placementMap() const { return placement_map_; }

        /**
         * @brief Get or create decode weight shard for a weight tensor
         *
         * For Option A (Selective Duplication) in CPU decode participation:
         * - Returns a SLICED copy of the weight for decode phase
         * - The shard is the "tail" portion based on fraction
         * - Cached separately from the full prefill weight
         *
         * Slicing behavior by ShardingMode:
         * - COLUMN_PARALLEL (Q, K, V, Gate, Up): slice tail rows (output dimension)
         * - ROW_PARALLEL (Wo): slice tail columns (input dimension)
         * - INPUT_PARALLEL (Down): slice tail columns (input dimension)
         * - REPLICATE (norms): return full copy (no slicing needed)
         *
         * @param name Weight tensor name (e.g., "blk.0.attn_q.weight")
         * @param decode_device Device for the decode shard (typically CPU)
         * @param fraction Fraction of weight for this shard (e.g., 0.20 = tail 20%)
         * @param layer_idx Layer index for sharding mode lookup
         * @return Shared pointer to sliced weight tensor, or nullptr on error
         */
        std::shared_ptr<TensorBase> getDecodeWeight(
            const std::string &name,
            DeviceId decode_device,
            float fraction,
            int layer_idx = -1) override;

        /**
         * @brief Get number of cached decode weight shards
         */
        size_t decodeCacheSize() const;

        /**
         * @brief Clear decode weight cache (frees decode shard memory)
         */
        void clearDecodeCache();

        // =========================================================================
        // Static utility methods for decode shard slicing
        // =========================================================================

        /**
         * @brief Slice tail rows from a tensor (for column-parallel decode shards)
         *
         * For weight matrix [out_dim, in_dim], extracts [out_local, in_dim]
         * where out_local = out_dim * fraction, starting from out_dim - out_local.
         *
         * @param full_tensor Full weight tensor
         * @param fraction Fraction of rows to extract (from tail)
         * @return Sliced tensor containing tail rows
         */
        static std::shared_ptr<TensorBase> sliceTailRows(
            const std::shared_ptr<TensorBase> &full_tensor,
            float fraction);

        /**
         * @brief Slice tail columns from a tensor (for row/input-parallel decode shards)
         *
         * For weight matrix [out_dim, in_dim], extracts [out_dim, in_local]
         * where in_local = in_dim * fraction, starting from in_dim - in_local.
         *
         * @param full_tensor Full weight tensor
         * @return Sliced tensor containing tail columns
         */
        static std::shared_ptr<TensorBase> sliceTailColumns(
            const std::shared_ptr<TensorBase> &full_tensor,
            float fraction);

        // =========================================================================
        // Progress Tracking
        // =========================================================================

        /// Set a shared progress tracker for weight loading visualization.
        /// Must be set before calling prepareWeightsForDevice().
        void setWeightLoadProgress(std::shared_ptr<WeightLoadProgress> progress)
        {
            weight_load_progress_ = std::move(progress);
        }

        /// Get the current progress tracker (may be null).
        std::shared_ptr<WeightLoadProgress> weightLoadProgress() const
        {
            return weight_load_progress_;
        }

        // =========================================================================
        // Phase 9: Weight Lifecycle Gates
        // =========================================================================

        /**
         * @brief Get the current lifecycle gates (read-only)
         */
        const WeightLifecycleGates &lifecycleGates() const { return lifecycle_gates_; }

        /// Access the expert GEMM registry (populated by GPU pipeline, queried by graph builders)
        ExpertGemmRegistry &expertGemmRegistry() { return expert_gemm_registry_; }
        const ExpertGemmRegistry &expertGemmRegistry() const { return expert_gemm_registry_; }

        std::shared_ptr<PreparedWeightStore> preparedWeightStore();
        std::shared_ptr<PreparedWeightStore> preparedWeightStoreIfInitialized() const;
        void setPreparedWeightStore(std::shared_ptr<PreparedWeightStore> store);

        /**
         * @brief Install the sole admitted CPU/GPU allocation authority.
         *
         * Orchestration performs this transition after topology-wide memory
         * admission and before any graph or prepared-weight allocation. The
         * authority is immutable by identity for the remaining model-context
         * lifetime; replacing it would split live allocation accounting.
         *
         * @param authority Rank-bound authority shared by every setup worker.
         * @throws std::invalid_argument for null input.
         * @throws std::logic_error when a different authority is already live.
         */
        void installPhysicalMemoryAuthority(
            std::shared_ptr<PhysicalMemoryAuthority> authority);

        /** @return The installed allocation authority, or null before admission. */
        [[nodiscard]] std::shared_ptr<PhysicalMemoryAuthority>
        physicalMemoryAuthority() const;

        /**
         * @brief Count model-owned prepared records for one exact device.
         *
         * Dense GEMMs, embeddings, and explicit slabs live in
         * PreparedWeightStore; routed ExpertOverlay GEMMs live in the
         * model-owned ExpertGemmRegistry. This is the single existence query
         * used by reuse certification and memory admission. It is not a byte
         * estimate because scoped expert aliases may share one allocation.
         */
        [[nodiscard]] size_t preparedRecordCountForDevice(
            DeviceId device) const;

        /**
         * @brief Return exact model-owned GPU weight bytes still live at seal.
         * @param device Exact GPU backend and ordinal.
         * @return Checked sum of live persistent weight pools and embeddings.
         *
         * Pool entries retain only weak ownership, so runner-only Dynamic
         * shadow allocations disappear before this query. Model-owned kernels
         * keep their original pool owner alive; prepared embeddings expose
         * their independent allocation through PreparedWeightStore. This is an
         * ownership query, never an allocator free-memory estimate.
         */
        [[nodiscard]] size_t retainedPreparedDeviceBytes(
            DeviceId device) const;

        /**
         * @brief Get the current lifecycle state (derived from gates)
         */
        WeightLifecycleState lifecycleState() const { return lifecycle_gates_.currentState(); }

        /**
         * @brief Mark source materialization complete
         *
         * Call after all source tensors and derived tensors (slices, clones,
         * TP shards, expert views) are loaded and created. After this gate,
         * no new tensors will be created from the model file.
         */
        void markMaterializationComplete()
        {
            lifecycle_gates_.materialization_complete = true;
            LOG_DEBUG("[WeightManager] Lifecycle gate: materialization_complete");
        }

        /**
         * @brief Mark device preparation complete
         *
         * Call after all GEMM weights are packed, all embeddings are prepared,
         * and all GPU uploads are done. After this gate, no new prepared handles
         * will be created for model weights.
         */
        void markDevicePreparationComplete()
        {
            lifecycle_gates_.device_preparation_complete = true;
            LOG_DEBUG("[WeightManager] Lifecycle gate: device_preparation_complete");
        }

        /**
         * @brief Mark graph materialization complete
         *
         * Call after all compute graphs have resolved their weight bindings.
         * After this gate, graph replay will not attempt to resolve new weights.
         */
        void markGraphMaterializationComplete() override
        {
            lifecycle_gates_.graph_materialization_complete = true;
            lifecycle_gates_.host_release_allowed = lifecycle_gates_.canReleaseHostData();
            LOG_DEBUG("[WeightManager] Lifecycle gate: graph_materialization_complete"
                      << " (host_release_allowed=" << lifecycle_gates_.host_release_allowed << ")");
        }

    private:
        /**
         * @brief Load weight with replicated strategy
         *
         * Each rank loads full tensor independently.
         */
        std::shared_ptr<TensorBase> getReplicatedWeight(const std::string &name, DeviceId device);

        /**
         * @brief Load weight with sharded strategy
         *
         * Tensor partitioned across ranks based on weight type:
         * - Column-parallel: QKV, Gate/Up projections (split output dim)
         * - Row-parallel: Wo, Down projections (split input dim)
         * - Replicated: Norms, biases, embeddings (full copy)
         */
        std::shared_ptr<TensorBase> getShardedWeight(const std::string &name, DeviceId device);

        /**
         * @brief Load weight with interleaved strategy (not yet implemented)
         *
         * NUMA-aware allocation with page interleaving.
         */
        std::shared_ptr<TensorBase> getInterleavedWeight(const std::string &name, DeviceId device);

        /**
         * @brief Determine sharding mode using config or legacy patterns
         *
         * Uses sharding_config_ if set, otherwise falls back to legacy hardcoded patterns.
         */
        ShardingMode determineShardingMode(const std::string &name) const;

        /**
         * @brief Convert WeightShardingMode (from schema) to ShardingMode (internal)
         */
        static ShardingMode toShardingMode(WeightShardingMode mode);

        /**
         * @brief Identity of one immutable model-prepared FP32 override.
         *
         * A canonical source can be bound under different semantic roles and
         * to different devices and logical slices.  All three facts affect the
         * representation and its eventual device residency, so none may be
         * inferred from tensor shape or omitted from cache identity.
         */
        struct ModelPreparedFp32OverrideKey
        {
            std::string canonical_name; ///< Stable model weight identity.
            WeightRole role = WeightRole::Other; ///< Graph semantic role.
            DeviceId target_device = DeviceId::invalid(); ///< Exact consumer.
            WeightSliceSpec slice; ///< Exact logical source interval represented.

            /** @return true when all representation-defining fields match. */
            bool operator==(const ModelPreparedFp32OverrideKey &other) const
            {
                return canonical_name == other.canonical_name &&
                       role == other.role &&
                       target_device == other.target_device &&
                       slice.source_rows == other.slice.source_rows &&
                       slice.source_cols == other.slice.source_cols &&
                       slice.row_start == other.slice.row_start &&
                       slice.row_count == other.slice.row_count &&
                       slice.col_start == other.slice.col_start &&
                       slice.col_count == other.slice.col_count &&
                       slice.expert_start == other.slice.expert_start &&
                       slice.expert_count == other.slice.expert_count &&
                       slice.expert_ids == other.slice.expert_ids &&
                       slice.inner_is_presliced == other.slice.inner_is_presliced;
            }
        };

        /** @brief Hashes the complete prepared-override identity. */
        struct ModelPreparedFp32OverrideKeyHash
        {
            /**
             * @param key Complete override identity.
             * @return Stable process-local hash suitable for unordered lookup.
             */
            size_t operator()(const ModelPreparedFp32OverrideKey &key) const noexcept
            {
                size_t hash = std::hash<std::string>{}(key.canonical_name);
                hash ^= std::hash<int>{}(static_cast<int>(key.role)) +
                        0x9e3779b9u + (hash << 6u) + (hash >> 2u);
                hash ^= std::hash<DeviceId>{}(key.target_device) +
                        0x9e3779b9u + (hash << 6u) + (hash >> 2u);
                const auto mix = [&hash](size_t value)
                {
                    hash ^= std::hash<size_t>{}(value) +
                            0x9e3779b9u + (hash << 6u) + (hash >> 2u);
                };
                mix(key.slice.source_rows);
                mix(key.slice.source_cols);
                mix(key.slice.row_start);
                mix(key.slice.row_count);
                mix(key.slice.col_start);
                mix(key.slice.col_count);
                mix(key.slice.expert_start);
                mix(key.slice.expert_count);
                mix(key.slice.expert_ids.size());
                for (const int expert_id : key.slice.expert_ids)
                    mix(static_cast<size_t>(expert_id));
                mix(key.slice.inner_is_presliced ? 1u : 0u);
                return hash;
            }
        };

        /**
         * @brief Acquire a model-owned FP32 representation for special graph roles.
         *
         * The returned object is cached for the lifetime of this WeightManager,
         * not the lifetime of a FrozenModelWeightSet or graph runner.  This is
         * required because raw GGUF bytes may be released after the first runner
         * is prepared while a later campaign cell reuses the same ModelContext.
         *
         * @param name Canonical model weight name.
         * @param role Semantic graph role selected by the weight plan.
         * @param source Loaded source tensor.
         * @param target_device Exact consuming device.
         * @param logical_slice Exact source interval represented by @p source.
         * @param logical_derivation Sharding/alias derivation that produced the
         *        logical value; scalar conversion does not replace this fact.
         * @return Shared immutable FP32 tensor, or nullptr when no override applies.
         * @throws std::runtime_error when an uncached override is requested after
         *         its source bytes have already been released.
         */
        std::shared_ptr<TensorBase> createModelPreparedFp32Override(
            const std::string &name,
            WeightRole role,
            const TensorBase *source,
            DeviceId target_device,
            const WeightSliceSpec &logical_slice,
            WeightDerivationKind logical_derivation);

        IModelLoader &loader_;                                                      ///< Model loader (GGUF, mock, etc.)
        std::shared_ptr<IMPIContext> mpi_ctx_;                                      ///< MPI context (nullptr = single rank)
        std::shared_ptr<WeightPlacementMap> placement_map_;                         ///< Fine-grained placement decisions
        std::shared_ptr<TensorParallelConfig> tp_config_;                           ///< Tensor parallelism configuration (optional)
        WeightDistributionStrategy strategy_;                                       ///< Distribution strategy
        WeightPrecision weight_precision_;                                          ///< How weights are loaded (NATIVE, CONVERT_TO_FP32, etc.)
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> cache_;        ///< Weight cache
        /**
         * Immutable derived values whose lifetime is the complete model context.
         *
         * These deliberately do not participate in host-weight release sweeps:
         * they are small canonical graph inputs needed to rematerialize runners
         * after the corresponding raw source bytes have been reclaimed.
         */
        std::unordered_map<
            ModelPreparedFp32OverrideKey,
            std::shared_ptr<TensorBase>,
            ModelPreparedFp32OverrideKeyHash>
            model_prepared_fp32_overrides_;
        mutable std::mutex cache_mutex_;                                            ///< Protects cache_ and decode_cache_ access
        mutable std::unordered_map<std::string, ShardingMode> sharding_mode_cache_; ///< Cached sharding modes
        mutable std::mutex sharding_mode_cache_mutex_;                              ///< Protects sharding_mode_cache_ (separate from cache_mutex_ to avoid deadlock — getShardingMode may be called while cache_mutex_ is held)
        WeightShardingConfig sharding_config_;                                      ///< Model-specific sharding patterns
        bool has_sharding_config_ = false;                                          ///< True if config was set explicitly
        WeightPreprocessor weight_preprocessor_;                                    ///< Optional per-weight transform before packing
        RoutedExpertWeightAssignment routed_expert_assignment_;                     ///< Static owner policy shared with graph construction

        // =========================================================================
        // Model head dimensions for FusedQKV sub-block computation
        // =========================================================================
        int model_n_heads_ = 0;             ///< Number of query attention heads
        int model_n_kv_heads_ = 0;          ///< Number of KV attention heads (GQA)
        int model_head_dim_ = 0;            ///< Dimension per attention head
        bool has_model_dimensions_ = false; ///< True if setModelDimensions() was called

        // GDN (Gated Delta Net) head dimensions for FusedQKV sharding
        int gdn_n_k_heads_ = 0;           ///< Number of GDN key heads (ssm.group_count)
        int gdn_n_v_heads_ = 0;           ///< Number of GDN value heads (ssm.time_step_rank)
        int gdn_d_state_ = 0;             ///< GDN state dimension per head (ssm.state_size)
        bool has_gdn_dimensions_ = false; ///< True if setGDNDimensions() was called

        // Decode weight shard cache (separate from prefill cache)
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> decode_cache_; ///< Decode shard cache

        std::shared_ptr<WeightMetadataRegistry> weight_metadata_;
        ExpertWeightPayloadProvider *expert_payload_provider_ = nullptr; ///< Optional, model-context owned

        void registerSourceMetadata(
            const std::string &name,
            const std::shared_ptr<TensorBase> &tensor,
            DeviceId device);
        void registerDerivedMetadata(
            const std::string &name,
            const std::shared_ptr<TensorBase> &tensor,
            WeightDerivationKind derivation,
            WeightSliceSpec slice,
            DeviceId device);
        void registerCloneMetadata(
            const std::string &name,
            const std::shared_ptr<TensorBase> &source,
            const std::shared_ptr<TensorBase> &clone,
            DeviceId device);
        WeightSliceSpec fullSliceSpec(const TensorBase &tensor) const;

        /**
         * @brief Resolve the exact ordered routed-expert IDs for one owner.
         *
         * Expert slices with different owner policies can have identical tensor
         * shapes. Cache identity therefore cannot be inferred from dimensions;
         * it must be derived from the same layer-aware ownership policy used by
         * graph construction and physical GGUF loading.
         *
         * @param name Canonical three-dimensional routed-expert weight name.
         * @param participant_index Static owner index requesting the weight.
         * @param participant_count Number of static whole-expert owners.
         * @param layer_idx Explicit layer index, or a negative value to derive it
         *        from @p name.
         * @return Sorted global expert IDs in packed source-tensor order.
         * @throws std::runtime_error when the tensor geometry or layer identity
         *         cannot establish an unambiguous ownership set.
         */
        std::vector<int> expectedRoutedExpertIds(
            const std::string &name,
            int participant_index,
            int participant_count,
            int layer_idx) const;

        /**
         * @brief Identify an exact full source that may precede TP slicing.
         *
         * Model-context construction can cache a complete immutable routed
         * tensor before the final TP topology is installed. Such a source is
         * not a stale expert slice: its typed source/clone derivation, canonical
         * name, and complete GGUF geometry make it an unambiguous input from
         * which the configured expert selection must subsequently be loaded.
         *
         * @param name Canonical routed-expert weight name.
         * @param tensor Candidate cached tensor.
         * @return true only for an unsliced full source or full device clone.
         */
        bool isCachedFullRoutedExpertSource(
            const std::string &name,
            const std::shared_ptr<TensorBase> &tensor) const;

        /**
         * @brief Prove that a cached expert tensor has the requested identity.
         *
         * A cache hit is accepted only when registry metadata identifies an
         * explicit expert slice with the exact global expert-ID sequence and a
         * packed tensor shape consistent with that sequence. Missing or stale
         * metadata is fatal; callers must never erase and silently reconstruct
         * an ambiguously identified expert tensor. A separately typed complete
         * source recognized by isCachedFullRoutedExpertSource() is a lifecycle
         * input rather than a candidate slice and is handled before this check.
         *
         * @param name Canonical routed-expert weight name.
         * @param tensor Candidate cached tensor.
         * @param expected_expert_ids Exact IDs requested by the active policy.
         * @param cache_key Human-readable cache key for diagnostics.
         * @throws std::runtime_error when any cache identity invariant fails.
         */
        void validateCachedRoutedExpertSlice(
            const std::string &name,
            const std::shared_ptr<TensorBase> &tensor,
            const std::vector<int> &expected_expert_ids,
            const std::string &cache_key) const;

        // =========================================================================
        // Layer range for Pipeline Parallelism (LAYER_PARTITIONED strategy)
        // =========================================================================

        int layer_first_ = 0;          ///< First layer index (inclusive)
        int layer_last_ = 0;           ///< Last layer index (exclusive)
        bool has_embedding_ = true;    ///< True if this stage loads embedding
        bool has_lm_head_ = true;      ///< True if this stage loads LM head
        bool has_layer_range_ = false; ///< True if setLayerRange() was called

        /**
         * @brief Check if a weight should be loaded based on layer range
         *
         * When LAYER_PARTITIONED strategy is active, this filters weights:
         * - "token_embd.weight" only if has_embedding_
         * - "output_norm.weight", "output.weight" only if has_lm_head_
         * - "blk.N.*" only if N is in [layer_first_, layer_last_)
         *
         * @param name Weight tensor name
         * @return true if weight should be loaded, false if filtered out
         */
        bool isWeightInLayerRange(const std::string &name) const;

        /**
         * @brief Check if a weight belongs to a given layer range (explicit parameters)
         *
         * Same logic as isWeightInLayerRange(name) but uses explicit parameters
         * instead of instance state. Used by prepareWeightsForDevice(layer_range)
         * for layer-filtered preparation.
         */
        bool isWeightInLayerRange(const std::string &name,
                                  int first_layer, int last_layer,
                                  bool has_embedding, bool has_lm_head) const;

        /**
         * @brief Internal implementation of prepareWeightsForDevice
         *
         * Runs GEMM packing, non-GEMM upload, and embedding preparation.
         * Optional layer_filter restricts which weights are processed.
         */
        bool prepareWeightsForDeviceImpl(
            DeviceId device,
            std::function<bool(const std::string &)> layer_filter,
            const FrozenModelWeightSet *frozen_weights = nullptr,
            bool include_expert_jobs = true,
            PhysicalMemoryOwner persistent_owner =
                PhysicalMemoryOwner::PrimaryModelWeights);

        /**
         * @brief Upload exact non-GEMM tensors owned by a frozen binding set.
         *
         * Frozen graph bindings may own freshly materialized replicated slices
         * or derived tensors that are intentionally absent from the broad
         * loader cache. This binding-driven pass uploads those exact objects
         * before execution so graph stages never repair missing residency.
         *
         * @param target_device GPU targeted by the frozen bindings.
         * @param frozen_weights Immutable bindings used by the graph.
         * @param layer_filter Optional canonical-name filter.
         * @param include_expert_jobs Retained for preparation-policy symmetry;
         *        routed expert tensors remain owned by the expert pipeline.
         */
        bool uploadFrozenNonGemmWeights(
            DeviceId target_device,
            const FrozenModelWeightSet &frozen_weights,
            const std::function<bool(const std::string &)> &layer_filter,
            bool include_expert_jobs,
            PhysicalMemoryOwner persistent_owner);

        // =========================================================================
        // Per-device tensor cache for multi-device scenarios (LOCAL TP)
        // =========================================================================

        /// Progress tracker for weight loading (shared across devices)
        std::shared_ptr<WeightLoadProgress> weight_load_progress_;

        /// Key: "device_type:ordinal:weight_name" e.g. "cuda:0:token_embd.weight"
        /// Value: Device-specific tensor clone, uploaded to that device
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> per_device_cache_;

        /// First device that requested weights - original tensors are used for this device
        std::optional<DeviceId> first_device_;

        /// Helper to create a clone of a tensor for a different device
        std::shared_ptr<TensorBase> cloneTensorForDevice(
            const std::string &name,
            const std::shared_ptr<TensorBase> &original,
            DeviceId target_device);

        /**
         * @brief Return whether a replicated GPU weight is only a host-side
         *        source for a later device-owned preparation step.
         *
         * Quantized GEMM weights, mirrored floating-point GEMM weights, token
         * embeddings, and complete MoE expert tensors are not consumed through
         * a raw per-device TensorBase allocation. Their owning GPU subsystem
         * repacks or slices the immutable model bytes into its own device
         * storage. Replicating the host tensor per GPU therefore adds no useful
         * residency; for large 3D expert tensors it can copy the complete model
         * once per participant before loading even begins.
         *
         * @param name Canonical GGUF tensor name.
         * @return true when every GPU participant may share the immutable host
         *         tensor while constructing independent device-owned state.
         */
        [[nodiscard]] bool isReplicatedGpuPreparationSource(
            const std::string &name) const;

        // =========================================================================
        // Proportional slicing helpers (used when tp_config_ is set)
        // =========================================================================

        /**
         * @brief Calculate proportional column slice for this rank
         *
         * Uses TensorParallelConfig to determine slice bounds instead of 1/world_size.
         * Returns {start_row, num_rows} for column-parallel weights.
         *
         * @param name Weight tensor name (to determine weight type)
         * @param total_rows Total number of rows in the weight
         * @return {start_row, num_rows} pair for this rank's slice
         */
        std::pair<size_t, size_t> calculateProportionalColumnSlice(
            const std::string &name, size_t total_rows) const;

        /**
         * @brief Calculate proportional row slice for this rank
         *
         * Uses TensorParallelConfig to determine slice bounds instead of 1/world_size.
         * Returns {start_col, num_cols} for row-parallel/input-parallel weights.
         *
         * @param name Weight tensor name (to determine weight type)
         * @param total_cols Total number of columns in the weight
         * @return {start_col, num_cols} pair for this rank's slice
         */
        std::pair<size_t, size_t> calculateProportionalRowSlice(
            const std::string &name, size_t total_cols) const;

    public:
        // =========================================================================
        // Device-aware weight slicing for LOCAL TP (Phase 1)
        // =========================================================================

        /**
         * @brief Load weight with device-specific sharding from TensorParallelConfig
         *
         * Uses the DeviceShardingAssignment to determine which slice of the weight
         * this device should receive. Supports proportional slicing for heterogeneous
         * GPU configurations.
         *
         * @param name Weight tensor name
         * @param device Target device
         * @param assignment Device's sharding assignment from TensorParallelConfig
         * @param layer_idx Layer index for logging
         * @return Shared pointer to sliced weight tensor, or nullptr on error
         */
        std::shared_ptr<TensorBase> getShardedWeightForAssignment(
            const std::string &name,
            DeviceId device,
            const DeviceShardingAssignment &assignment,
            int layer_idx);

        /**
         * @brief Slice a specific row range from tensor
         *
         * Creates a new tensor containing only the specified rows.
         * Supports FP32 tensors (quantized tensors should use GGUF row slice loading).
         *
         * @param tensor Source tensor to slice
         * @param row_start First row index (0-based)
         * @param row_count Number of rows to extract
         * @return New tensor with the specified row range, or nullptr on error
         */
        static std::shared_ptr<TensorBase> sliceRowRange(
            const std::shared_ptr<TensorBase> &tensor,
            size_t row_start,
            size_t row_count);

    private:
        // ========================================================================
        // Sharding helper methods (used by getShardedWeightForAssignment)
        // ========================================================================

        /**
         * @brief Compute slice boundaries based on dimension type from WeightShardingConfig
         *
         * Uses the sharding config to determine which dimension (Heads, KVHeads, FFNHidden, Vocab)
         * should be used for slicing, then computes the appropriate start/count values.
         *
         * @param name Weight tensor name
         * @param total_size Total size of the dimension to slice
         * @param assignment Device's sharding assignment
         * @param out_start Output: start index for slicing
         * @param out_count Output: count of elements to include
         * @return true if boundaries computed successfully, false on error
         */
        bool computeSliceBoundaries(
            const std::string &name,
            size_t total_size,
            const DeviceShardingAssignment &assignment,
            size_t &out_start,
            size_t &out_count) const;

        /**
         * @brief Load a column-parallel 1D bias tensor (e.g., Q/K/V biases)
         *
         * Slices a 1D bias tensor based on head assignment.
         *
         * @param name Tensor name
         * @param device Target device
         * @param assignment Device's sharding assignment
         * @param dimensions Tensor dimensions
         * @return Sliced bias tensor, or nullptr on error
         */
        std::shared_ptr<TensorBase> loadColumnParallel1DBias(
            const std::string &name,
            DeviceId device,
            const DeviceShardingAssignment &assignment,
            const std::vector<size_t> &dimensions);

        /**
         * @brief Load a column-parallel 2D weight tensor (Q/K/V, Gate/Up, LM Head)
         *
         * Slices rows based on the weight type (heads, d_ff, or vocab).
         *
         * @param name Tensor name
         * @param device Target device
         * @param assignment Device's sharding assignment
         * @param dimensions Tensor dimensions
         * @return Sliced weight tensor wrapped in TensorSlice, or nullptr on error
         */
        std::shared_ptr<TensorBase> loadColumnParallel2DWeight(
            const std::string &name,
            DeviceId device,
            const DeviceShardingAssignment &assignment,
            const std::vector<size_t> &dimensions);

        /**
         * @brief Load a fused QKV weight with per-sub-block head slicing
         *
         * Handles weights stored as [Q_all | K_all | V_all] concatenated rows.
         * Splits each sub-block independently by heads, then reassembles as
         * [Q_local | K_local | V_local] so downstream code sees the correct
         * per-head Q/K/V ordering.
         *
         * @param name Tensor name (e.g., "blk.0.attn_qkv.weight")
         * @param device Target device
         * @param assignment Device's sharding assignment
         * @param dimensions Tensor dimensions [total_rows, cols]
         * @return FP32 tensor with correctly ordered local Q/K/V rows
         */
        std::shared_ptr<TensorBase> loadFusedQKVColumnParallel(
            const std::string &name,
            DeviceId device,
            const DeviceShardingAssignment &assignment,
            const std::vector<size_t> &dimensions);

        /**
         * @brief Load a row-parallel weight tensor (unused - Wo uses INPUT_PARALLEL)
         *
         * Slices rows based on head assignment for weights where output is reduced.
         *
         * @param name Tensor name
         * @param device Target device
         * @param assignment Device's sharding assignment
         * @param dimensions Tensor dimensions
         * @return Sliced weight tensor wrapped in TensorSlice, or nullptr on error
         */
        std::shared_ptr<TensorBase> loadRowParallelWeight(
            const std::string &name,
            DeviceId device,
            const DeviceShardingAssignment &assignment,
            const std::vector<size_t> &dimensions);

        /**
         * @brief Load an input-parallel weight tensor (Wo, FFN Down)
         *
         * Slices columns based on weight type:
         * - Wo: slices by head assignment (matches Q/K/V output)
         * - FFN Down: slices by d_ff assignment (matches Gate/Up output)
         *
         * @param name Tensor name
         * @param device Target device
         * @param assignment Device's sharding assignment
         * @param dimensions Tensor dimensions
         * @return Sliced weight tensor wrapped in TensorSlice, or nullptr on error
         */
        std::shared_ptr<TensorBase> loadInputParallelWeight(
            const std::string &name,
            DeviceId device,
            const DeviceShardingAssignment &assignment,
            const std::vector<size_t> &dimensions);

        /**
         * @brief Resolve economical modulo-linked GDN ownership for one TP participant.
         *
         * The attention-head partition is used only as a backend-independent
         * proportional partition space. The returned assignment owns a true
         * Q/K shard and every V head whose @c global_v % global_k falls inside
         * that shard.
         */
        GDNHeadAssignment gdnHeadAssignmentFor(
            const DeviceShardingAssignment &assignment) const;

        /**
         * @brief Resolve economical modulo-linked ownership for equal-rank TP.
         *
         * This is the global-MPI companion to @ref gdnHeadAssignmentFor. Both
         * entry points produce the same typed ownership object, so rank-local
         * and LocalTP loading cannot drift into different GDN layouts.
         */
        GDNHeadAssignment gdnHeadAssignmentForEqualRank(
            int rank,
            int world_size) const;

        /**
         * @brief Return the logical element width of a GDN value head.
         *
         * @return One for per-head scalar projections, @c gdn_d_state for
         *         value-channel projections, or zero when @p name/@p total_size
         *         is not an exact GDN value-head tensor.
         */
        int gdnValueElementsPerHead(
            const std::string &name,
            size_t total_size) const;

        /**
         * @brief Load and concatenate non-contiguous source row intervals.
         *
         * Every interval is read directly in native format and concatenated in
         * the supplied order. This preserves source quantization bytes while
         * constructing a participant-local semantic order (including linked
         * GDN heads and independently sliced fused Q/K/V blocks) once during
         * model loading.
         */
        std::shared_ptr<TensorBase> loadNativeRowSpanConcat(
            const std::string &name,
            DeviceId device,
            const std::vector<GDNHeadSpan> &spans,
            size_t source_cols);

        /**
         * @brief Load and row-wise concatenate non-contiguous source columns.
         *
         * This is the input-parallel companion to
         * @ref loadNativeRowSpanConcat. Quantized intervals must be naturally
         * block aligned; an unrepresentable geometry fails explicitly.
         */
        std::shared_ptr<TensorBase> loadNativeColumnSpanConcat(
            const std::string &name,
            DeviceId device,
            const std::vector<GDNHeadSpan> &spans,
            size_t source_rows);

        /**
         * @brief Load a TP GDN fused [Q|K|V] tensor in linked local order.
         *
         * The returned native payload is backend-neutral preparation input.
         * CUDA, ROCm, and CPU therefore consume exactly the same semantic head
         * assignment before their respective prepared-weight stores take
         * ownership of the bytes.
         */
        std::shared_ptr<TensorBase> loadGDNFusedQKVColumnParallel(
            const std::string &name,
            DeviceId device,
            const GDNHeadAssignment &head_assignment,
            int rank,
            int world_size,
            const std::vector<size_t> &dimensions);

        /** @brief Load a value-associated row-sharded GDN tensor. */
        std::shared_ptr<TensorBase> loadGDNValueRows(
            const std::string &name,
            DeviceId device,
            const GDNHeadAssignment &head_assignment,
            int rank,
            int world_size,
            const std::vector<size_t> &dimensions);

        /** @brief Load a value-associated input-sharded GDN tensor. */
        std::shared_ptr<TensorBase> loadGDNValueColumns(
            const std::string &name,
            DeviceId device,
            const GDNHeadAssignment &head_assignment,
            int rank,
            int world_size,
            const std::vector<size_t> &dimensions);

    public:
        /**
         * @brief Slice a specific column range from tensor
         *
         * Creates a new tensor containing only the specified columns.
         * Supports FP32 tensors (quantized tensors should use GGUF column slice loading).
         *
         * This is used for INPUT_PARALLEL weight slicing where the input dimension
         * (columns) is split across devices. Different from row slicing because
         * columns are non-contiguous in row-major memory layout.
         *
         * @param tensor Source tensor to slice
         * @param col_start First column index (0-based)
         * @param col_count Number of columns to extract
         * @return New tensor with the specified column range, or nullptr on error
         */
        static std::shared_ptr<TensorBase> sliceColumnRange(
            const std::shared_ptr<TensorBase> &tensor,
            size_t col_start,
            size_t col_count);

        /**
         * @brief Determine weight category from name
         *
         * Categories:
         * - ATTENTION_QKV: Q, K, V projections (column-parallel by heads)
         * - ATTENTION_WO: Output projection (row-parallel, matches Q output)
         * - FFN_GATE_UP: Gate and Up projections (column-parallel by d_ff)
         * - FFN_DOWN: Down projection (input-parallel, matches gate/up output)
         * - LM_HEAD: Language model head (column-parallel by vocab)
         * - REPLICATE: Everything else (norms, biases, embeddings)
         */
        enum class WeightCategory
        {
            ATTENTION_QKV,
            ATTENTION_WO,
            FFN_GATE_UP,
            FFN_DOWN,
            LM_HEAD,
            REPLICATE
        };

        WeightCategory categorizeWeight(const std::string &name) const;

        // =========================================================================
        // Preload statistics (folded from WeightPreloader)
        // =========================================================================
        size_t num_cpu_packed_ = 0;

        size_t num_gpu_packed_ = 0;

    private:
        // =========================================================================
        // Phase 9: Lifecycle gates (model-level state machine)
        // =========================================================================

        WeightLifecycleGates lifecycle_gates_;

        ExpertGemmRegistry expert_gemm_registry_;
        std::shared_ptr<PreparedWeightStore> prepared_weight_store_;
        /** Serializes the one-time admission-to-materialization publication. */
        mutable std::mutex physical_memory_authority_mutex_;
        /** Sole CPU/GPU allocation authority shared by all preparation lanes. */
        std::shared_ptr<PhysicalMemoryAuthority> physical_memory_authority_;
        uint64_t next_pipeline_prepared_binding_id_ = (1ULL << 48);

        /** Exact weak ownership of every finalized GPU weight pool. */
        PreparedDeviceAllocationLedger prepared_device_allocations_;

        // =========================================================================
        // Phase 2: Per-weight/device readiness tickets and TP-safe reclaim eligibility
        // =========================================================================

        enum class WeightPrepState
        {
            UNKNOWN = 0,
            LOADED_HOST,
            PACKED_HOST,
            UPLOADED_DEVICE,
            READY,
            FAILED
        };
        struct WeightPrepTicket
        {
            WeightPrepState state = WeightPrepState::UNKNOWN;
            bool is_gemm = false;
            std::string detail;
        };

        mutable std::mutex prep_ticket_mutex_;
        std::unordered_map<std::string, std::unordered_map<std::string, WeightPrepTicket>> prep_tickets_;
        std::unordered_map<std::string, std::unordered_set<std::string>> expected_devices_by_weight_;
        std::unordered_set<std::string> reclaim_ready_weights_;
        std::unordered_set<std::string> reclaim_applied_weights_;

        void registerExpectedDeviceForWeight(const std::string &name, DeviceId device);
        void markPrepState(const std::string &name,
                           DeviceId device,
                           WeightPrepState state,
                           bool is_gemm,
                           const std::string &detail = "");
        void evaluateReclaimEligibility(const std::string &name, bool is_gemm);
        bool tryReleaseReclaimHostRawData(const std::string &name);
        static const char *weightPrepStateName(WeightPrepState state);

        /**
         * @brief Execute the complete reclaim sequence on the lifecycle worker.
         *
         * Snapshots all surviving mapped tensors under the cache mutex, then
         * retires their accelerator host registrations before asking the loader
         * to advise its durable mappings. The snapshot owns tensor lifetimes
         * after the mutex is released, so inference never waits on the expensive
         * device-runtime or page-table operations.
         *
         * @return Number of durable mapping bytes advised by the loader.
         */
        size_t performMmapReclaim();

        /** Sole exactly-once authority for asynchronous model-mapping reclaim. */
        MmapReclaimLifecycle mmap_reclaim_lifecycle_;
    };

} // namespace llaminar2
