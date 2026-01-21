/**
 * @file PlacementStrategy.cpp
 * @brief Implementation of placement strategies
 * @author David Sanftenberg
 * @date December 2025
 */

#include "PlacementStrategy.h"
#include "placement/HeterogeneousMultiDomainStrategy.h"
#include "../utils/Logger.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // Memory Estimation Helpers
    // =========================================================================

    /**
     * @brief Get native bytes per weight element for a quantization type (GGUF native format)
     *
     * These are the sizes of weights as stored in GGUF files, before repacking.
     * Block-based formats store block_bytes / block_size bytes per element.
     */
    static float getNativeBytesPerWeight(const std::string &quant_type)
    {
        // =================================================================
        // Full precision formats
        // =================================================================
        if (quant_type == "F32")
            return 4.0f;
        if (quant_type == "F16" || quant_type == "FP16")
            return 2.0f;
        if (quant_type == "BF16")
            return 2.0f;
        if (quant_type == "INT8" || quant_type == "I8")
            return 1.0f;

        // =================================================================
        // Simple block formats (32-element blocks)
        // =================================================================
        // Q8_0: 34 bytes per 32 elements (2B FP16 scale + 32B int8)
        if (quant_type == "Q8_0")
            return 34.0f / 32.0f; // 1.0625

        // Q8_1: 36 bytes per 32 elements (2B FP16 scale + 2B INT16 sum + 32B int8)
        if (quant_type == "Q8_1")
            return 36.0f / 32.0f; // 1.125

        // Q4_0: 18 bytes per 32 elements (2B FP16 scale + 16B packed 4-bit)
        if (quant_type == "Q4_0")
            return 18.0f / 32.0f; // 0.5625

        // Q4_1: 20 bytes per 32 elements (2B scale + 2B min + 16B packed)
        if (quant_type == "Q4_1")
            return 20.0f / 32.0f; // 0.625

        // Q5_0: 22 bytes per 32 elements (2B scale + 4B high bits + 16B low)
        if (quant_type == "Q5_0")
            return 22.0f / 32.0f; // 0.6875

        // Q5_1: 24 bytes per 32 elements (2B scale + 2B min + 4B high + 16B low)
        if (quant_type == "Q5_1")
            return 24.0f / 32.0f; // 0.75

        // IQ4_NL: 18 bytes per 32 elements (non-linear lookup 4-bit)
        if (quant_type == "IQ4_NL")
            return 18.0f / 32.0f; // 0.5625

        // =================================================================
        // Q16_1 variable block sizes (for integer attention)
        // =================================================================
        // Q16_1 (32): 72 bytes per 32 elements (4B FP32 scale + 4B int32 sum + 64B int16)
        if (quant_type == "Q16_1" || quant_type == "Q16_1_32")
            return 72.0f / 32.0f; // 2.25

        // Q16_1_64: 136 bytes per 64 elements
        if (quant_type == "Q16_1_64")
            return 136.0f / 64.0f; // 2.125

        // Q16_1_128: 264 bytes per 128 elements
        if (quant_type == "Q16_1_128")
            return 264.0f / 128.0f; // 2.0625

        // =================================================================
        // K-quant formats (256-element super-blocks)
        // =================================================================
        // Q2_K: 84 bytes per 256 elements
        if (quant_type == "Q2_K")
            return 84.0f / 256.0f; // 0.328

        // Q3_K: 110 bytes per 256 elements
        if (quant_type == "Q3_K" || quant_type == "Q3_K_S" || quant_type == "Q3_K_M" ||
            quant_type == "Q3_K_L")
            return 110.0f / 256.0f; // 0.430

        // Q4_K: 144 bytes per 256 elements
        if (quant_type == "Q4_K" || quant_type == "Q4_K_S" || quant_type == "Q4_K_M")
            return 144.0f / 256.0f; // 0.5625

        // Q5_K: 176 bytes per 256 elements
        if (quant_type == "Q5_K" || quant_type == "Q5_K_S" || quant_type == "Q5_K_M")
            return 176.0f / 256.0f; // 0.6875

        // Q6_K: 210 bytes per 256 elements
        if (quant_type == "Q6_K")
            return 210.0f / 256.0f; // 0.820

        // Q8_K: 288 bytes per 256 elements
        if (quant_type == "Q8_K")
            return 288.0f / 256.0f; // 1.125

        // =================================================================
        // IQ (importance quantization) formats (256-element super-blocks)
        // =================================================================
        // IQ1_S: 50 bytes per 256 elements
        if (quant_type == "IQ1_S")
            return 50.0f / 256.0f; // 0.195

        // IQ1_M: 56 bytes per 256 elements
        if (quant_type == "IQ1_M")
            return 56.0f / 256.0f; // 0.219

        // IQ2_XXS: 66 bytes per 256 elements
        if (quant_type == "IQ2_XXS")
            return 66.0f / 256.0f; // 0.258

        // IQ2_XS: 74 bytes per 256 elements
        if (quant_type == "IQ2_XS")
            return 74.0f / 256.0f; // 0.289

        // IQ2_S: 82 bytes per 256 elements
        if (quant_type == "IQ2_S")
            return 82.0f / 256.0f; // 0.320

        // IQ3_XXS: 98 bytes per 256 elements
        if (quant_type == "IQ3_XXS")
            return 98.0f / 256.0f; // 0.383

        // IQ3_S: 110 bytes per 256 elements
        if (quant_type == "IQ3_S")
            return 110.0f / 256.0f; // 0.430

        // IQ4_XS: 136 bytes per 256 elements
        if (quant_type == "IQ4_XS")
            return 136.0f / 256.0f; // 0.531

        // Default to Q4_0 estimate if unknown
        LOG_WARN("[getNativeBytesPerWeight] Unknown quant type '" << quant_type
                                                                  << "', defaulting to Q4_0 estimate");
        return 18.0f / 32.0f; // Q4_0 default
    }

    /**
     * @brief Get CPU packed weight bytes per element (after VNNI repacking)
     *
     * CPU execution repacks weights to VNNI-friendly INT8 format with:
     * - packed_data: [K/4][N][4] = K*N bytes (INT8)
     * - scales: [K/32][N] = (K/32)*N*4 bytes (FP32)
     * - compensation: [K/32][N] = (K/32)*N*4 bytes (INT32)
     * - mins: [K/32][N] = (K/32)*N*4 bytes (FP32, optional for asymmetric)
     *
     * Per element: 1 + 4/(32) + 4/(32) + 4/(32) = 1 + 0.375 = 1.375 bytes (with mins)
     *              or 1 + 0.25 = 1.25 bytes (without mins, symmetric quantization)
     */
    static float getCPUPackedBytesPerWeight(const std::string &quant_type)
    {
        // All quantized weights are repacked to INT8 + metadata for VNNI
        // Symmetric quantization (Q4_0, Q8_0, etc.): ~1.25 bytes/element
        // Asymmetric quantization (IQ4_NL, Q4_1): ~1.375 bytes/element

        // Check for asymmetric types that need mins
        bool has_mins = (quant_type.find("IQ") != std::string::npos) ||
                        (quant_type == "Q4_1") ||
                        (quant_type == "Q5_1");

        if (has_mins)
        {
            // packed_data (1) + scales (4/32) + compensation (4/32) + mins (4/32)
            return 1.0f + 3.0f * (4.0f / 32.0f); // 1.375
        }
        else
        {
            // packed_data (1) + scales (4/32) + compensation (4/32)
            return 1.0f + 2.0f * (4.0f / 32.0f); // 1.25
        }
    }

    /**
     * @brief Get CUDA packed weight bytes per element (after INT8 column-major repacking)
     *
     * CUDA execution repacks weights to column-major INT8 with per-column scales:
     * - int8_data: [K × N] = K*N bytes (INT8, column-major)
     * - scales: [N] = N*4 bytes (FP32, per-column)
     *
     * Per element: 1 + 4/K bytes (scales amortized across K)
     * For typical K (4096), this is ~1.001 bytes/element
     */
    static float getCUDAPackedBytesPerWeight(size_t K)
    {
        // INT8 data + amortized scale overhead
        return 1.0f + 4.0f / static_cast<float>(K);
    }

    /**
     * @brief Estimate activation buffer memory per layer (excluding KV cache)
     *
     * Activation buffers per layer include:
     * - hidden/normalized: B × S × d_model × 4 (FP32)
     * - Q/K/V projections: depends on precision (Q8_1 or FP32)
     * - attention output: B × S × d_model × 4 (FP32)
     * - FFN gate/up: B × S × d_ff × 4 (FP32)
     * - FFN output: B × S × d_model × 4 (FP32)
     * - attention workspace: n_heads × S × S × 4 (FP32 scores)
     */
    struct ActivationBufferEstimate
    {
        size_t per_layer_bytes;     ///< Memory per transformer layer
        size_t attention_workspace; ///< Attention score workspace (sequence-length dependent)
        size_t global_buffers;      ///< Embedding output, logits, etc.
    };

    static ActivationBufferEstimate estimateActivationBuffers(
        size_t batch_size,
        size_t seq_len,
        size_t d_model,
        size_t d_ff,
        size_t n_heads,
        size_t n_kv_heads)
    {
        ActivationBufferEstimate est{};

        // Per-layer buffers (FP32 unless noted)
        // hidden + normalized + residual = 3 × B × S × d_model × 4
        size_t hidden_buffers = 3 * batch_size * seq_len * d_model * sizeof(float);

        // QKV projections: For Q16 attention pipeline, these are Q8_1 or Q16_1
        // Q: B × S × d_model (n_heads × head_dim)
        // K/V: B × S × d_kv (n_kv_heads × head_dim)
        size_t head_dim = d_model / n_heads;
        size_t d_kv = n_kv_heads * head_dim;
        // Using Q8_1 estimate (~1.125 bytes per element)
        size_t qkv_buffers = static_cast<size_t>(
            (batch_size * seq_len * d_model + // Q
             2 * batch_size * seq_len * d_kv) // K + V
            * 1.125f);

        // Attention output: B × S × d_model × 4 (FP32)
        size_t attn_output = batch_size * seq_len * d_model * sizeof(float);

        // FFN buffers: gate + up (each B × S × d_ff) + output (B × S × d_model)
        // Can be Q8_1 or FP32 depending on pipeline
        size_t ffn_buffers = 2 * batch_size * seq_len * d_ff * sizeof(float) +
                             batch_size * seq_len * d_model * sizeof(float);

        est.per_layer_bytes = hidden_buffers + qkv_buffers + attn_output + ffn_buffers;

        // Attention workspace: scores matrix for softmax
        // For each head: [S × S] scores
        // Total: n_heads × S × S × 4 (FP32)
        est.attention_workspace = n_heads * seq_len * seq_len * sizeof(float);

        // Global buffers: embedding output + logits
        // These are shared across layers, not per-layer
        est.global_buffers = 0; // Estimated separately

        return est;
    }

    // Legacy wrapper for backward compatibility
    static float getBytesPerWeight(const std::string &quant_type)
    {
        return getNativeBytesPerWeight(quant_type);
    }

    // =========================================================================
    // CPUOnlyLayerPlacementStrategy Implementation
    // =========================================================================

    bool CPUOnlyLayerPlacementStrategy::isApplicable(const PlacementInput &input) const
    {
        // CPU-only is ALWAYS applicable (every system has a CPU)
        // User can explicitly choose CPU even when GPU is available
        (void)input; // unused
        return true;
    }

    PlacementPlan CPUOnlyLayerPlacementStrategy::compute(const PlacementInput &input) const
    {
        PlacementPlan plan;

        // Copy input parameters
        plan.n_layers = input.n_layers;
        plan.model_memory_bytes = input.estimated_memory_bytes;
        plan.architecture = input.architecture;
        plan.world_size = input.world_size;
        plan.ranks_per_node = input.ranks_per_node;
        plan.node_count = input.node_count;
        plan.has_gpu = false; // CPU-only strategy never uses GPU
        plan.total_gpu_memory = 0;
        plan.strategy_name = name();

        // Global tensors: All on CPU, shard if multi-rank
        plan.global.embedding_device = PlacementDevice::cpu();
        plan.global.lm_head_device = PlacementDevice::cpu();
        plan.global.final_norm_device = PlacementDevice::cpu();
        plan.global.shard_embedding = (input.world_size > 1 && input.vocab_size > 100000);
        plan.global.shard_lm_head = (input.world_size > 1 && input.vocab_size > 100000);

        // Allocate layer placements
        plan.layers.resize(input.n_layers);

        // Simple round-robin distribution of layers across ranks
        // For CPU-only TP, all ranks participate in all layers (row-parallel)
        // The "owner_rank" here is less meaningful - all ranks have all weights
        // but we track it for future pipeline parallelism support
        for (int layer = 0; layer < input.n_layers; ++layer)
        {
            LayerPlacement &lp = plan.layers[layer];
            lp.layer_idx = layer;

            // For tensor parallelism, all ranks own all layers (row-sharded)
            // owner_rank = 0 means "all ranks" for TP
            // For pipeline parallelism (future), this would be layer / layers_per_rank
            lp.owner_rank = 0;

            lp.device = PlacementDevice::cpu();
            lp.attention_device = PlacementDevice::cpu();
            lp.ffn_device = PlacementDevice::cpu();
            lp.split_attention_ffn = false;
        }

        LOG_DEBUG("[CPUOnlyLayerPlacementStrategy] Generated plan: " << input.n_layers << " layers, "
                                                                     << input.world_size << " ranks, all CPU");

        return plan;
    }

    // =========================================================================
    // GPUFirstLayerPlacementStrategy Implementation
    // =========================================================================

    bool GPUFirstLayerPlacementStrategy::isApplicable(const PlacementInput &input) const
    {
        // GPU-first requires at least one GPU and not forced CPU-only
        return !input.force_cpu_only && input.any_rank_has_gpu;
    }

    size_t GPUFirstLayerPlacementStrategy::estimateLayerMemory(const PlacementInput &input,
                                                               PlacementDevice device) const
    {
        // Estimate memory per transformer layer on specified device:
        // 1. Attention weights: Q, K, V, O projections
        // 2. FFN weights: Gate, Up, Down projections
        // 3. Norm weights: attention_norm, ffn_norm (always FP32, negligible)
        // 4. Activation buffers (device-specific)
        // 5. KV cache per-layer allocation

        size_t d_model = input.d_model;
        size_t d_ff = input.d_ff > 0 ? input.d_ff : d_model * 4;
        size_t d_kv = (input.n_kv_heads * d_model) / input.n_heads;
        size_t head_dim = d_model / input.n_heads;

        // Weight element counts per layer
        size_t attn_weight_elements = d_model * d_model + // Q
                                      d_model * d_kv +    // K
                                      d_model * d_kv +    // V
                                      d_model * d_model;  // O

        size_t ffn_weight_elements = d_ff * d_model + // Gate
                                     d_ff * d_model + // Up
                                     d_model * d_ff;  // Down

        size_t total_weight_elements = attn_weight_elements + ffn_weight_elements;

        // Calculate packed weight size based on target device
        float bytes_per_weight;
        if (device.isCPU())
        {
            // CPU: VNNI-packed INT8 with scales/compensation
            bytes_per_weight = getCPUPackedBytesPerWeight(input.quant_type);
        }
        else
        {
            // GPU: Column-major INT8 with per-column scales
            bytes_per_weight = getCUDAPackedBytesPerWeight(d_model);
        }

        size_t weight_memory = static_cast<size_t>(total_weight_elements * bytes_per_weight);

        // Norm weights (always FP32, negligible but include for completeness)
        size_t norm_memory = 2 * d_model * sizeof(float);

        // Activation buffers for this layer
        // For placement estimation, use batch=1, seq_len=512 (typical inference)
        auto act_est = estimateActivationBuffers(1, 512, d_model, d_ff, input.n_heads, input.n_kv_heads);
        size_t activation_memory = act_est.per_layer_bytes;

        // KV cache per layer (estimate for max_seq_len = 2048)
        // K and V each: n_kv_heads × max_seq_len × head_dim × sizeof(float)
        size_t kv_cache_per_layer = 2 * input.n_kv_heads * 2048 * head_dim * sizeof(float);

        return weight_memory + norm_memory + activation_memory + kv_cache_per_layer;
    }

    size_t GPUFirstLayerPlacementStrategy::estimateGlobalMemory(const PlacementInput &input,
                                                                PlacementDevice device) const
    {
        (void)device; // Currently same for all devices
        size_t d_model = input.d_model;
        size_t vocab_size = input.vocab_size;

        // Embedding and LM head are typically NOT repacked (they're FP16/BF16 or native quant)
        // Use native bytes per weight for these
        float bytes_per_weight = getNativeBytesPerWeight(input.quant_type);

        // Embedding: [vocab_size, d_model]
        size_t embedding = static_cast<size_t>(vocab_size * d_model * bytes_per_weight);

        // LM head: [vocab_size, d_model] (often tied to embedding weights)
        size_t lm_head = static_cast<size_t>(vocab_size * d_model * bytes_per_weight);

        // Final norm: d_model (FP32)
        size_t final_norm = d_model * sizeof(float);

        // Logits buffer: [batch × vocab_size × sizeof(float)]
        // For placement estimation, batch=1
        size_t logits_buffer = vocab_size * sizeof(float);

        return embedding + lm_head + final_norm + logits_buffer;
    }

    PlacementPlan GPUFirstLayerPlacementStrategy::compute(const PlacementInput &input) const
    {
        PlacementPlan plan;

        // Copy input parameters
        plan.n_layers = input.n_layers;
        plan.model_memory_bytes = input.estimated_memory_bytes;
        plan.architecture = input.architecture;
        plan.world_size = input.world_size;
        plan.ranks_per_node = input.ranks_per_node;
        plan.node_count = input.node_count;
        plan.has_gpu = true;
        plan.total_gpu_memory = input.total_gpu_memory;
        plan.strategy_name = name();

        // Calculate memory requirements for GPU placement
        size_t layer_memory_gpu = estimateLayerMemory(input, PlacementDevice::gpu(0));
        size_t global_memory_gpu = estimateGlobalMemory(input, PlacementDevice::gpu(0));

        // =====================================================================
        // Build a unified view of all GPUs across all ranks
        // Each GPU is identified by (rank, local_gpu_index)
        // =====================================================================
        struct GPUSlot
        {
            int rank;
            int local_gpu_idx;
            size_t memory;
            size_t remaining;
            std::vector<int> assigned_layers;
        };
        std::vector<GPUSlot> all_gpus;

        // Collect GPUs from cluster inventory if available
        if (!input.cluster_inventory.ranks.empty())
        {
            for (const auto &rank_inv : input.cluster_inventory.ranks)
            {
                for (int g = 0; g < static_cast<int>(rank_inv.gpus.size()); ++g)
                {
                    GPUSlot slot;
                    slot.rank = rank_inv.rank;
                    slot.local_gpu_idx = g;
                    slot.memory = rank_inv.gpus[g].memory_bytes;
                    slot.remaining = static_cast<size_t>(slot.memory * 0.90); // 10% overhead
                    all_gpus.push_back(slot);
                }
            }
        }
        // Fall back to legacy gpu_memory_per_device (assumes single rank)
        else if (!input.gpu_memory_per_device.empty())
        {
            for (int g = 0; g < static_cast<int>(input.gpu_memory_per_device.size()); ++g)
            {
                GPUSlot slot;
                slot.rank = 0; // Assume rank 0
                slot.local_gpu_idx = g;
                slot.memory = input.gpu_memory_per_device[g];
                slot.remaining = static_cast<size_t>(slot.memory * 0.90);
                all_gpus.push_back(slot);
            }
        }
        else if (input.total_gpu_memory > 0)
        {
            // Single GPU fallback
            GPUSlot slot;
            slot.rank = 0;
            slot.local_gpu_idx = 0;
            slot.memory = input.total_gpu_memory;
            slot.remaining = static_cast<size_t>(slot.memory * 0.90);
            all_gpus.push_back(slot);
        }

        int total_gpus = static_cast<int>(all_gpus.size());
        if (total_gpus == 0)
        {
            LOG_WARN("[GPUFirstLayerPlacementStrategy] No GPU memory info available, falling back to CPU");
        }

        LOG_DEBUG("[GPUFirstLayerPlacementStrategy] Layer memory estimate (GPU): "
                  << (layer_memory_gpu / (1024 * 1024)) << " MB");
        LOG_DEBUG("[GPUFirstLayerPlacementStrategy] Global memory estimate: "
                  << (global_memory_gpu / (1024 * 1024)) << " MB");
        LOG_DEBUG("[GPUFirstLayerPlacementStrategy] Total GPUs across cluster: " << total_gpus);
        for (const auto &gpu : all_gpus)
        {
            LOG_DEBUG("[GPUFirstLayerPlacementStrategy] Rank " << gpu.rank << " GPU " << gpu.local_gpu_idx
                                                               << " usable memory: "
                                                               << (gpu.remaining / (1024 * 1024)) << " MB");
        }

        // Apply max_gpu_layers constraint if specified
        int max_gpu_layers = input.n_layers;
        if (input.max_gpu_layers >= 0)
        {
            max_gpu_layers = std::min(max_gpu_layers, input.max_gpu_layers);
        }

        // Reserve memory for global tensors on first GPU that has space
        int global_gpu_idx = -1; // Index into all_gpus, -1 means CPU
        if (total_gpus > 0 && global_memory_gpu < all_gpus[0].remaining * 0.3)
        {
            global_gpu_idx = 0;
            all_gpus[0].remaining -= global_memory_gpu;
        }

        // =====================================================================
        // Distribute layers across GPUs cluster-wide
        // Strategy: Fill each GPU before moving to next (greedy)
        // =====================================================================
        int layers_assigned_to_gpu = 0;
        for (int layer = 0; layer < input.n_layers && layers_assigned_to_gpu < max_gpu_layers; ++layer)
        {
            // Find a GPU with enough memory
            int target_gpu_idx = -1;
            for (int g = 0; g < total_gpus; ++g)
            {
                if (all_gpus[g].remaining >= layer_memory_gpu)
                {
                    target_gpu_idx = g;
                    break;
                }
            }

            if (target_gpu_idx >= 0)
            {
                all_gpus[target_gpu_idx].assigned_layers.push_back(layer);
                all_gpus[target_gpu_idx].remaining -= layer_memory_gpu;
                layers_assigned_to_gpu++;
            }
            // else: layer will stay on CPU
        }

        // Log placement summary
        LOG_INFO("[GPUFirstLayerPlacementStrategy] Placing " << layers_assigned_to_gpu << "/"
                                                             << input.n_layers << " layers on GPU(s)");
        for (int g = 0; g < total_gpus; ++g)
        {
            if (!all_gpus[g].assigned_layers.empty())
            {
                LOG_INFO("[GPUFirstLayerPlacementStrategy] Rank " << all_gpus[g].rank
                                                                  << " GPU_" << all_gpus[g].local_gpu_idx << ": "
                                                                  << all_gpus[g].assigned_layers.size()
                                                                  << " layers, "
                                                                  << (all_gpus[g].remaining / (1024 * 1024))
                                                                  << " MB remaining");
            }
        }

        // =====================================================================
        // Build placement plan
        // =====================================================================

        // Set global placement
        if (global_gpu_idx >= 0)
        {
            auto &gpu = all_gpus[global_gpu_idx];
            // Note: For now, global tensors go on the GPU of the first rank that has space
            // In tensor parallel mode, they'll be sharded anyway
            plan.global.embedding_device = PlacementDevice::gpu(gpu.local_gpu_idx);
            plan.global.lm_head_device = PlacementDevice::gpu(gpu.local_gpu_idx);
            plan.global.final_norm_device = PlacementDevice::gpu(gpu.local_gpu_idx);
        }
        else
        {
            plan.global.embedding_device = PlacementDevice::cpu();
            plan.global.lm_head_device = PlacementDevice::cpu();
            plan.global.final_norm_device = PlacementDevice::cpu();
        }

        // Sharding for large vocab (even with GPU, multi-rank needs sharding)
        plan.global.shard_embedding = (input.world_size > 1 && input.vocab_size > 100000);
        plan.global.shard_lm_head = (input.world_size > 1 && input.vocab_size > 100000);

        // Allocate layer placements (default to CPU)
        plan.layers.resize(input.n_layers);
        for (int layer = 0; layer < input.n_layers; ++layer)
        {
            LayerPlacement &lp = plan.layers[layer];
            lp.layer_idx = layer;
            lp.owner_rank = 0; // For tensor parallelism, all ranks participate
            lp.device = PlacementDevice::cpu();
            lp.attention_device = PlacementDevice::cpu();
            lp.ffn_device = PlacementDevice::cpu();
            lp.split_attention_ffn = false;
        }

        // Assign GPU layers based on distribution
        for (int g = 0; g < total_gpus; ++g)
        {
            const auto &gpu = all_gpus[g];
            PlacementDevice dev = PlacementDevice::gpu(gpu.local_gpu_idx);

            for (int layer : gpu.assigned_layers)
            {
                LayerPlacement &lp = plan.layers[layer];
                lp.owner_rank = gpu.rank; // Owner is the rank with this GPU
                lp.device = dev;
                lp.attention_device = dev;
                lp.ffn_device = dev;
            }
        }

        return plan;
    }

    // =========================================================================
    // HybridOptimalLayerPlacementStrategy Implementation
    // =========================================================================

    bool HybridOptimalLayerPlacementStrategy::isApplicable(const PlacementInput &input) const
    {
        // Hybrid requires:
        // 1. At least one GPU
        // 2. Not forced to CPU-only or GPU-only
        // 3. Have bandwidth info (optional but helps)
        return !input.force_cpu_only &&
               !input.force_gpu_only &&
               input.any_rank_has_gpu;
    }

    size_t HybridOptimalLayerPlacementStrategy::estimateBytesPerToken(const PlacementInput &input,
                                                                      PlacementDevice device) const
    {
        // For decode (seq_len=1), memory bandwidth dominates
        // Bytes read = all weights for one layer (after packing)
        size_t d_model = input.d_model;
        size_t d_ff = input.d_ff > 0 ? input.d_ff : d_model * 4;
        size_t d_kv = (input.n_kv_heads * d_model) / input.n_heads;

        // Total weight elements per layer
        size_t attn_elements = d_model * d_model + // Q
                               d_model * d_kv +    // K
                               d_model * d_kv +    // V
                               d_model * d_model;  // O

        size_t ffn_elements = d_ff * d_model + // Gate
                              d_ff * d_model + // Up
                              d_model * d_ff;  // Down

        size_t total_elements = attn_elements + ffn_elements;

        // Bytes read depends on device's packed format
        float bytes_per_weight;
        if (device.isCPU())
        {
            bytes_per_weight = getCPUPackedBytesPerWeight(input.quant_type);
        }
        else
        {
            bytes_per_weight = getCUDAPackedBytesPerWeight(d_model);
        }

        return static_cast<size_t>(total_elements * bytes_per_weight);
    }

    float HybridOptimalLayerPlacementStrategy::estimateCPUDecodeTokensPerSec(const PlacementInput &input) const
    {
        // Memory-bound estimate for CPU decode
        // tokens/sec = bandwidth / (bytes_per_token × n_layers)
        float bandwidth_bps = input.cpu_memory_bandwidth_gbps * 1e9f;
        if (bandwidth_bps <= 0)
        {
            // Default estimate for DDR5-4800 quad-channel: ~150 GB/s
            bandwidth_bps = 150e9f;
        }

        size_t bytes_per_token = estimateBytesPerToken(input, PlacementDevice::cpu());
        if (bytes_per_token == 0)
            return 0.0f;

        // Full model estimate
        return bandwidth_bps / static_cast<float>(bytes_per_token * input.n_layers);
    }

    float HybridOptimalLayerPlacementStrategy::estimateGPUDecodeTokensPerSec(const PlacementInput &input) const
    {
        // Memory-bound estimate for GPU decode (single token doesn't saturate compute)
        // tokens/sec = bandwidth / (bytes_per_token × n_layers)
        float bandwidth_bps = input.gpu_memory_bandwidth_gbps * 1e9f;
        if (bandwidth_bps <= 0)
        {
            // Default estimate for RTX 4090: ~1000 GB/s
            bandwidth_bps = 1000e9f;
        }

        size_t bytes_per_token = estimateBytesPerToken(input, PlacementDevice::gpu(0));
        if (bytes_per_token == 0)
            return 0.0f;

        // Full model estimate
        return bandwidth_bps / static_cast<float>(bytes_per_token * input.n_layers);
    }

    int HybridOptimalLayerPlacementStrategy::computeOptimalGPULayers(const PlacementInput &input,
                                                                     const std::vector<size_t> &gpu_memories) const
    {
        // Calculate total usable GPU memory across all GPUs
        size_t total_gpu_memory = 0;
        for (auto mem : gpu_memories)
        {
            total_gpu_memory += static_cast<size_t>(mem * 0.85); // 15% overhead
        }

        // Estimate bytes per layer on GPU
        size_t d_model = input.d_model;
        size_t d_ff = input.d_ff > 0 ? input.d_ff : d_model * 4;
        size_t d_kv = (input.n_kv_heads * d_model) / input.n_heads;
        size_t head_dim = d_model / input.n_heads;

        // Weight elements per layer
        size_t weight_elements = 2 * d_model * d_model + // Q + O
                                 2 * d_model * d_kv +    // K + V
                                 3 * d_ff * d_model;     // Gate + Up + Down

        // GPU packed bytes
        float bytes_per_weight = getCUDAPackedBytesPerWeight(d_model);
        size_t layer_weight_bytes = static_cast<size_t>(weight_elements * bytes_per_weight);

        // Activation buffers per layer
        auto act_est = estimateActivationBuffers(1, 512, d_model, d_ff, input.n_heads, input.n_kv_heads);

        // KV cache per layer
        size_t kv_cache = 2 * input.n_kv_heads * 2048 * head_dim * sizeof(float);

        size_t bytes_per_layer_gpu = layer_weight_bytes + act_est.per_layer_bytes + kv_cache;

        // How many layers fit?
        int max_gpu_layers = bytes_per_layer_gpu > 0
                                 ? static_cast<int>(total_gpu_memory / bytes_per_layer_gpu)
                                 : input.n_layers;
        max_gpu_layers = std::min(max_gpu_layers, input.n_layers);

        // Now consider bandwidth efficiency
        float cpu_tps = estimateCPUDecodeTokensPerSec(input);
        float gpu_tps = estimateGPUDecodeTokensPerSec(input);

        LOG_DEBUG("[HybridOptimalLayerPlacementStrategy] CPU decode estimate: " << cpu_tps << " tok/s");
        LOG_DEBUG("[HybridOptimalLayerPlacementStrategy] GPU decode estimate: " << gpu_tps << " tok/s");

        // If CPU is competitive (>30% of GPU), consider hybrid split
        float cpu_efficiency = cpu_tps / (gpu_tps + 1e-6f);

        if (cpu_efficiency > 0.5f && input.total_cpu_memory > 0)
        {
            // CPU is highly competitive - use hybrid split
            // Put compute-heavy attention on GPU, memory-heavy FFN can spill to CPU
            // Heuristic: use GPU for up to 75% of capacity
            int optimal_gpu = static_cast<int>(max_gpu_layers * 0.75);
            LOG_INFO("[HybridOptimalLayerPlacementStrategy] CPU is " << static_cast<int>(cpu_efficiency * 100)
                                                                     << "% as efficient as GPU, using hybrid split");
            return std::max(optimal_gpu, 1);
        }

        // GPU dominant: use as much GPU as possible
        return max_gpu_layers;
    }

    PlacementPlan HybridOptimalLayerPlacementStrategy::compute(const PlacementInput &input) const
    {
        PlacementPlan plan;

        // Copy input parameters
        plan.n_layers = input.n_layers;
        plan.model_memory_bytes = input.estimated_memory_bytes;
        plan.architecture = input.architecture;
        plan.world_size = input.world_size;
        plan.ranks_per_node = input.ranks_per_node;
        plan.node_count = input.node_count;
        plan.has_gpu = true;
        plan.total_gpu_memory = input.total_gpu_memory;
        plan.strategy_name = name();

        // =====================================================================
        // Build a unified view of all GPUs across all ranks
        // =====================================================================
        struct GPUSlot
        {
            int rank;
            int local_gpu_idx;
            size_t memory;
            size_t remaining;
            std::vector<int> assigned_layers;
        };
        std::vector<GPUSlot> all_gpus;

        // Collect GPUs from cluster inventory if available
        if (!input.cluster_inventory.ranks.empty())
        {
            for (const auto &rank_inv : input.cluster_inventory.ranks)
            {
                for (int g = 0; g < static_cast<int>(rank_inv.gpus.size()); ++g)
                {
                    GPUSlot slot;
                    slot.rank = rank_inv.rank;
                    slot.local_gpu_idx = g;
                    slot.memory = rank_inv.gpus[g].memory_bytes;
                    slot.remaining = static_cast<size_t>(slot.memory * 0.85); // 15% overhead for hybrid
                    all_gpus.push_back(slot);
                }
            }
        }
        // Fall back to legacy gpu_memory_per_device
        else if (!input.gpu_memory_per_device.empty())
        {
            for (int g = 0; g < static_cast<int>(input.gpu_memory_per_device.size()); ++g)
            {
                GPUSlot slot;
                slot.rank = 0;
                slot.local_gpu_idx = g;
                slot.memory = input.gpu_memory_per_device[g];
                slot.remaining = static_cast<size_t>(slot.memory * 0.85);
                all_gpus.push_back(slot);
            }
        }
        else if (input.total_gpu_memory > 0)
        {
            GPUSlot slot;
            slot.rank = 0;
            slot.local_gpu_idx = 0;
            slot.memory = input.total_gpu_memory;
            slot.remaining = static_cast<size_t>(slot.memory * 0.85);
            all_gpus.push_back(slot);
        }

        int total_gpus = static_cast<int>(all_gpus.size());

        // Calculate total usable GPU memory
        size_t total_usable_gpu_memory = 0;
        for (const auto &gpu : all_gpus)
        {
            total_usable_gpu_memory += gpu.remaining;
        }

        // Compute optimal split using bandwidth analysis
        std::vector<size_t> gpu_memories_for_split;
        for (const auto &gpu : all_gpus)
        {
            gpu_memories_for_split.push_back(gpu.memory);
        }
        int gpu_layers = computeOptimalGPULayers(input, gpu_memories_for_split);

        // Apply max_gpu_layers constraint if specified
        if (input.max_gpu_layers >= 0)
        {
            gpu_layers = std::min(gpu_layers, input.max_gpu_layers);
        }

        LOG_INFO("[HybridOptimalLayerPlacementStrategy] Placing " << gpu_layers << "/" << input.n_layers
                                                                  << " layers on " << total_gpus << " GPU(s) (hybrid optimal)");

        // Estimate bytes per layer for distribution
        size_t d_model = input.d_model;
        size_t d_ff = input.d_ff > 0 ? input.d_ff : d_model * 4;
        size_t head_dim = d_model / input.n_heads;

        size_t weight_elements = 2 * d_model * d_model +
                                 2 * (input.n_kv_heads * head_dim) * d_model +
                                 3 * d_ff * d_model;
        float bytes_per_weight = getCUDAPackedBytesPerWeight(d_model);
        auto act_est = estimateActivationBuffers(1, 512, d_model, d_ff, input.n_heads, input.n_kv_heads);
        size_t kv_cache = 2 * input.n_kv_heads * 2048 * head_dim * sizeof(float);
        size_t bytes_per_layer = static_cast<size_t>(weight_elements * bytes_per_weight) +
                                 act_est.per_layer_bytes + kv_cache;

        // =====================================================================
        // Distribute layers across GPUs cluster-wide
        // =====================================================================
        int layers_on_gpu = 0;
        for (int layer = 0; layer < input.n_layers && layer < gpu_layers; ++layer)
        {
            // Find a GPU with space
            int target_gpu_idx = -1;
            for (int g = 0; g < total_gpus; ++g)
            {
                if (all_gpus[g].remaining >= bytes_per_layer)
                {
                    target_gpu_idx = g;
                    break;
                }
            }

            if (target_gpu_idx >= 0)
            {
                all_gpus[target_gpu_idx].assigned_layers.push_back(layer);
                all_gpus[target_gpu_idx].remaining -= bytes_per_layer;
                layers_on_gpu++;
            }
        }

        // =====================================================================
        // Build placement plan
        // =====================================================================
        plan.layers.resize(input.n_layers);

        // Calculate phase-aware device weights
        // PREFILL: GPU only (compute-bound) - handled by primary device assignment
        // DECODE: Bandwidth-proportional (CPU + GPU)
        auto [decode_gpu_weight, decode_cpu_weight] = input.getPhaseDeviceWeights(InferencePhase::DECODE);
        bool cpu_should_decode = input.cpuShouldParticipate(InferencePhase::DECODE);

        LOG_INFO("[HybridOptimalLayerPlacementStrategy] Phase-aware decode: CPU_weight="
                 << decode_cpu_weight << ", GPU_weight=" << decode_gpu_weight
                 << ", CPU_participates=" << (cpu_should_decode ? "yes" : "no"));

        // Default all to CPU
        for (int layer = 0; layer < input.n_layers; ++layer)
        {
            LayerPlacement &lp = plan.layers[layer];
            lp.layer_idx = layer;
            lp.owner_rank = 0;
            lp.split_attention_ffn = false;
            lp.device = PlacementDevice::cpu();
            lp.attention_device = PlacementDevice::cpu();
            lp.ffn_device = PlacementDevice::cpu();

            // Phase-aware decode: CPU participates for ALL layers if bandwidth is meaningful
            lp.cpu_participates_in_decode = cpu_should_decode && total_gpus > 0;
        }

        // Assign GPU layers (for PREFILL - compute-bound phase)
        for (int g = 0; g < total_gpus; ++g)
        {
            const auto &gpu = all_gpus[g];
            PlacementDevice dev = PlacementDevice::gpu(gpu.local_gpu_idx);

            for (int layer : gpu.assigned_layers)
            {
                LayerPlacement &lp = plan.layers[layer];
                lp.owner_rank = gpu.rank;
                lp.device = dev;
                lp.attention_device = dev;
                lp.ffn_device = dev;

                // DECODE: Set up bandwidth-proportional multi-device execution
                // CPU + GPU both participate in decode, weighted by memory bandwidth
                if (cpu_should_decode)
                {
                    lp.decode_devices.clear();
                    lp.decode_weight_fractions.clear();

                    // Add GPU as primary decode device
                    lp.decode_devices.push_back(dev);
                    lp.decode_weight_fractions.push_back(decode_gpu_weight);

                    // Add CPU as secondary decode device
                    lp.decode_devices.push_back(PlacementDevice::cpu());
                    lp.decode_weight_fractions.push_back(decode_cpu_weight);

                    lp.cpu_participates_in_decode = true;

                    LOG_DEBUG("[HybridOptimalLayerPlacementStrategy] Layer " << layer
                                                                             << ": PREFILL on GPU_" << gpu.local_gpu_idx
                                                                             << ", DECODE on GPU(" << decode_gpu_weight
                                                                             << ") + CPU(" << decode_cpu_weight << ")");
                }
            }
        }

        // Global tensors placement based on where most layers are
        if (layers_on_gpu > input.n_layers / 2 && total_gpus > 0)
        {
            // Mostly GPU: put globals on first GPU
            plan.global.embedding_device = PlacementDevice::gpu(0);
            plan.global.lm_head_device = PlacementDevice::gpu(0);
            plan.global.final_norm_device = PlacementDevice::gpu(0);
        }
        else
        {
            // Mostly CPU or no GPUs: keep globals on CPU
            plan.global.embedding_device = PlacementDevice::cpu();
            plan.global.lm_head_device = PlacementDevice::cpu();
            plan.global.final_norm_device = PlacementDevice::cpu();
        }

        plan.global.shard_embedding = (input.world_size > 1 && input.vocab_size > 100000);
        plan.global.shard_lm_head = (input.world_size > 1 && input.vocab_size > 100000);

        return plan;
    }

    // =========================================================================
    // LayerPlacementStrategyFactory Implementation
    // =========================================================================

    std::unique_ptr<LayerPlacementStrategy> LayerPlacementStrategyFactory::create(const std::string &name)
    {
        if (name == "CPUOnly" || name == "cpu" || name == "cpu_only")
        {
            return std::make_unique<CPUOnlyLayerPlacementStrategy>();
        }
        if (name == "GPUFirst" || name == "gpu" || name == "gpu_first")
        {
            return std::make_unique<GPUFirstLayerPlacementStrategy>();
        }
        if (name == "HybridOptimal" || name == "hybrid" || name == "hybrid_optimal")
        {
            return std::make_unique<HybridOptimalLayerPlacementStrategy>();
        }
        if (name == "HeterogeneousMultiDomain" || name == "heterogeneous" || name == "multi_domain")
        {
            return std::make_unique<HeterogeneousMultiDomainStrategy>();
        }

        LOG_WARN("[LayerPlacementStrategyFactory] Unknown strategy: " << name);
        return nullptr;
    }

    std::unique_ptr<LayerPlacementStrategy> LayerPlacementStrategyFactory::autoSelect(const PlacementInput &input)
    {
        // Priority 1: User-specified strategy
        if (!input.preferred_strategy.empty())
        {
            auto strategy = create(input.preferred_strategy);
            if (!strategy)
            {
                throw std::runtime_error(
                    "[LayerPlacementStrategyFactory] Unknown strategy: '" + input.preferred_strategy +
                    "'. Valid strategies: CPUOnly, GPUFirst, HybridOptimal.");
            }
            if (!strategy->isApplicable(input))
            {
                throw std::runtime_error(
                    "[LayerPlacementStrategyFactory] Strategy '" + input.preferred_strategy +
                    "' is not applicable for current configuration. "
                    "Check GPU availability and force_cpu_only flag.");
            }
            LOG_DEBUG("[LayerPlacementStrategyFactory] Using user-specified strategy: "
                      << input.preferred_strategy);
            return strategy;
        }

        // Priority 2: Force flags
        if (input.force_cpu_only)
        {
            LOG_DEBUG("[LayerPlacementStrategyFactory] CPU-only mode forced");
            return std::make_unique<CPUOnlyLayerPlacementStrategy>();
        }

        if (input.force_gpu_only && input.any_rank_has_gpu)
        {
            LOG_DEBUG("[LayerPlacementStrategyFactory] GPU-only mode forced");
            return std::make_unique<GPUFirstLayerPlacementStrategy>();
        }

        // Priority 3: Auto-select based on device availability and info
        if (input.any_rank_has_gpu)
        {
            // Have GPU: choose between GPUFirst and HybridOptimal
            // Use HybridOptimal if we have bandwidth info (suggests optimization intent)
            if (input.cpu_memory_bandwidth_gbps > 0 || input.gpu_memory_bandwidth_gbps > 0)
            {
                LOG_DEBUG("[LayerPlacementStrategyFactory] GPU available with bandwidth info, using HybridOptimal");
                return std::make_unique<HybridOptimalLayerPlacementStrategy>();
            }
            else
            {
                LOG_DEBUG("[LayerPlacementStrategyFactory] GPU available, using GPUFirst");
                return std::make_unique<GPUFirstLayerPlacementStrategy>();
            }
        }

        // No GPU available: CPU-only
        LOG_DEBUG("[LayerPlacementStrategyFactory] No GPU available, using CPUOnly");
        return std::make_unique<CPUOnlyLayerPlacementStrategy>();
    }

    std::vector<std::string> LayerPlacementStrategyFactory::availableStrategies()
    {
        return {"CPUOnly", "GPUFirst", "HybridOptimal", "HeterogeneousMultiDomain"};
    }

} // namespace llaminar2
