/**
 * @file GraphResolver.cpp
 * @brief Implementation of declarative graph schema resolver
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file implements GraphResolver, the "brain" that transforms declarative
 * schemas into executable graph specifications by evaluating all runtime
 * conditionals in one place.
 *
 * KEY PRINCIPLE: All imperative logic (MPI checks, debugEnv toggles, attention
 * mode detection) is consolidated HERE, not scattered across model-specific
 * graph builders.
 */

#include "GraphResolver.h"
#include "GraphExecutor.h" // For ComputeGraph
#include "compute_stages/ComputeStages.h"
#include "../utils/DebugEnv.h"
#include "../utils/Logger.h"
#include "../tensors/Tensors.h"
#include "../tensors/UnifiedKVCache.h"
#include <stdexcept>
#include <algorithm>

namespace llaminar2
{

    // =========================================================================
    // ExecutionPolicyFlags Implementation
    // =========================================================================

    bool ExecutionPolicyFlags::shouldExecute(const std::string &policy_key) const
    {
        if (policy_key.empty())
            return true;

        // Map policy keys to flags
        if (policy_key == "exec_embedding")
            return exec_embedding;
        if (policy_key == "exec_rmsnorm")
            return exec_rmsnorm;
        if (policy_key == "exec_gemm")
            return exec_gemm;
        if (policy_key == "exec_rope")
            return exec_rope;
        if (policy_key == "exec_attention")
            return exec_attention;
        if (policy_key == "exec_swiglu")
            return exec_swiglu;
        if (policy_key == "exec_residual")
            return exec_residual;
        if (policy_key == "exec_lm_head")
            return exec_lm_head;

        // Unknown key - default to enabled
        LOG_WARN("[ExecutionPolicyFlags] Unknown policy key: " << policy_key);
        return true;
    }

    ExecutionPolicyFlags ExecutionPolicyFlags::fromDebugEnv()
    {
        const auto &env = debugEnv();
        ExecutionPolicyFlags flags;

        flags.exec_embedding = env.execution.exec_embedding;
        flags.exec_rmsnorm = env.execution.exec_rmsnorm;
        flags.exec_gemm = env.execution.exec_gemm;
        flags.exec_rope = env.execution.exec_rope;
        flags.exec_attention = env.execution.exec_attention;
        flags.exec_swiglu = env.execution.exec_swiglu;
        flags.exec_residual = env.execution.exec_residual;
        flags.exec_lm_head = env.execution.exec_lm_head;

        return flags;
    }

    // =========================================================================
    // TensorContext Implementation
    // =========================================================================

    TensorBase *TensorContext::resolve(const TensorRef &ref, int layer_idx) const
    {
        const std::string &name = ref.name;

        // Check for "weights." prefix
        if (name.rfind("weights.", 0) == 0)
        {
            std::string weight_name = name.substr(8); // Remove "weights." prefix

            // Check model-level weights first
            auto it = model_weights.find(weight_name);
            if (it != model_weights.end())
            {
                return it->second;
            }

            // Try layer-level weights if layer_idx is valid
            if (layer_idx >= 0 && get_layer_weight)
            {
                TensorBase *tensor = get_layer_weight(layer_idx, weight_name);
                if (tensor)
                {
                    return tensor;
                }
            }

            LOG_DEBUG("[TensorContext::resolve] Weight not found: " << name
                                                                    << " (layer_idx=" << layer_idx << ")");
            return nullptr;
        }

        // Check special references
        if (name == "kv_cache")
        {
            // KV cache is not a TensorBase - return nullptr
            // Caller should check kv_cache member directly
            return nullptr;
        }

        // Check buffers
        auto it = buffers.find(name);
        if (it != buffers.end())
        {
            return it->second;
        }

        LOG_DEBUG("[TensorContext::resolve] Buffer not found: " << name);
        return nullptr;
    }

    // =========================================================================
    // GraphResolver Implementation
    // =========================================================================

    ResolvedGraphSpec GraphResolver::resolve(
        const GraphSchema &schema,
        const GraphResolverConfig &runtime,
        const TensorContext &tensors) const
    {
        ResolvedGraphSpec result;
        result.name = schema.name;

        LOG_DEBUG("[GraphResolver] Resolving schema: " << schema.name
                                                       << " (n_layers=" << runtime.n_layers
                                                       << ", world_size=" << runtime.world_size
                                                       << ", batch_size=" << runtime.batch_size
                                                       << ", seq_len=" << runtime.seq_len << ")");

        // ==============================================================
        // Stage 1: Embedding
        // ==============================================================
        auto embedding_stage = resolveStage(schema.embedding, -1, runtime, tensors);
        if (embedding_stage)
        {
            result.stages.push_back(std::move(*embedding_stage));
            result.stats.stages_emitted++;
        }
        else
        {
            result.stats.stages_skipped++;
        }

        // ==============================================================
        // Stage 2: Transformer Layers
        // ==============================================================
        std::string prev_layer_final_stage = "embedding";

        for (int layer = 0; layer < runtime.n_layers; ++layer)
        {
            auto layer_stages = resolveLayer(schema.layer_template, layer, runtime, tensors);

            // Update dependencies: first stage of layer depends on previous layer's final
            if (!layer_stages.empty() && layer > 0)
            {
                // The first stage of attention should depend on previous layer's FFN residual
                // This is handled by dependency resolution below
            }

            for (auto &stage : layer_stages)
            {
                result.stages.push_back(std::move(stage));
            }
        }

        // ==============================================================
        // Stage 3: LM Head
        // ==============================================================
        std::string prev_node = "layer" + std::to_string(runtime.n_layers - 1) + "_ffn_residual";

        for (const auto &spec : schema.lm_head_stages)
        {
            auto resolved = resolveStage(spec, -1, runtime, tensors);
            if (resolved)
            {
                // Update dependency to connect to last layer
                if (resolved->dependencies.empty() && spec.name == "final_norm")
                {
                    resolved->dependencies.push_back(prev_node);
                }

                result.stages.push_back(std::move(*resolved));
                result.stats.stages_emitted++;

                // Check for TP collective insertion
                auto collective = resolveTPCollective(spec, result.stages.back(), runtime);
                if (collective)
                {
                    result.stages.push_back(std::move(*collective));
                    result.stats.allgather_inserted++;
                }
            }
            else
            {
                result.stats.stages_skipped++;
            }
        }

        LOG_INFO("[GraphResolver] Resolved " << result.stats.stages_emitted << " stages"
                                             << " (skipped=" << result.stats.stages_skipped
                                             << ", allreduce=" << result.stats.allreduce_inserted
                                             << ", allgather=" << result.stats.allgather_inserted << ")");

        return result;
    }

    std::vector<ResolvedStage> GraphResolver::resolveLayer(
        const LayerTemplate &layer_template,
        int layer_idx,
        const GraphResolverConfig &runtime,
        const TensorContext &tensors) const
    {
        std::vector<ResolvedStage> stages;
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";

        // ==============================================================
        // Attention Block
        // ==============================================================
        for (const auto &spec : layer_template.attention_stages)
        {
            auto resolved = resolveStage(spec, layer_idx, runtime, tensors);
            if (resolved)
            {
                stages.push_back(std::move(*resolved));

                // Check for TP collective insertion after this stage
                auto collective = resolveTPCollective(spec, stages.back(), runtime);
                if (collective)
                {
                    stages.push_back(std::move(*collective));
                }
            }
        }

        // ==============================================================
        // FFN Block
        // ==============================================================
        for (const auto &spec : layer_template.ffn_stages)
        {
            auto resolved = resolveStage(spec, layer_idx, runtime, tensors);
            if (resolved)
            {
                stages.push_back(std::move(*resolved));

                // Check for TP collective insertion after this stage
                auto collective = resolveTPCollective(spec, stages.back(), runtime);
                if (collective)
                {
                    stages.push_back(std::move(*collective));
                }
            }
        }

        return stages;
    }

    std::optional<ResolvedStage> GraphResolver::resolveStage(
        const StageSpec &spec,
        int layer_idx,
        const GraphResolverConfig &runtime,
        const TensorContext &tensors) const
    {
        // ==============================================================
        // Check if stage should be emitted
        // ==============================================================
        if (!shouldEmitStage(spec, runtime))
        {
            LOG_TRACE("[GraphResolver] Skipping stage: " << spec.name
                                                         << " (policy_key=" << spec.exec_policy_key << ")");
            return std::nullopt;
        }

        // ==============================================================
        // Create resolved stage
        // ==============================================================
        ResolvedStage resolved;
        resolved.name = expandStageName(spec.name, layer_idx);
        resolved.type = spec.type;
        resolved.device_idx = (spec.device_idx >= 0) ? spec.device_idx : runtime.default_device;

        // ==============================================================
        // Resolve tensor references
        // ==============================================================
        for (const auto &ref : spec.inputs)
        {
            TensorBase *tensor = resolveTensorRef(ref, layer_idx, tensors);
            resolved.inputs.push_back(tensor);
        }

        for (const auto &ref : spec.outputs)
        {
            TensorBase *tensor = resolveTensorRef(ref, layer_idx, tensors);
            resolved.outputs.push_back(tensor);
        }

        // ==============================================================
        // Resolve dependencies (expand layer prefix)
        // ==============================================================
        for (const auto &dep : spec.dependencies)
        {
            resolved.dependencies.push_back(expandStageName(dep, layer_idx));
        }

        // ==============================================================
        // Populate stage-specific parameters
        // ==============================================================
        populateStageParams(resolved, spec, runtime, tensors, layer_idx);

        return resolved;
    }

    // =========================================================================
    // Resolution Helpers
    // =========================================================================

    bool GraphResolver::shouldEmitStage(
        const StageSpec &spec,
        const GraphResolverConfig &runtime) const
    {
        // Skip empty/invalid stages
        if (spec.name.empty())
        {
            return false;
        }

        // Check execution policy (debugEnv toggles)
        if (spec.is_optional)
        {
            if (!runtime.exec_policy.shouldExecute(spec.exec_policy_key))
            {
                return false;
            }
        }

        // Check KV cache requirement
        if (spec.requires_kv_cache && !runtime.has_kv_cache)
        {
            return false;
        }

        return true;
    }

    std::optional<ResolvedStage> GraphResolver::resolveTPCollective(
        const StageSpec &spec,
        const ResolvedStage &resolved,
        const GraphResolverConfig &runtime) const
    {
        // No TP collective needed for single rank
        if (runtime.world_size <= 1)
        {
            return std::nullopt;
        }

        // Check if this stage has TP annotation
        if (spec.tp_mode == TPMode::None)
        {
            return std::nullopt;
        }

        // Check if the weight is actually sharded
        // For row-parallel: input weight is sharded, need allreduce after
        // For column-parallel: output weight is sharded, need allgather after

        switch (spec.tp_mode)
        {
        case TPMode::RowParallel:
        {
            // Row-parallel projection: allreduce the output
            ResolvedStage allreduce;
            allreduce.name = resolved.name + "_allreduce";
            allreduce.type = StageType::Allreduce;
            allreduce.device_idx = resolved.device_idx;
            allreduce.dependencies.push_back(resolved.name);

            // The output buffer is the first output of the resolved stage
            if (!resolved.outputs.empty())
            {
                allreduce.inputs.push_back(resolved.outputs[0]);
                allreduce.outputs.push_back(resolved.outputs[0]); // In-place
            }

            // MPI parameters - store MPI_Comm directly as void*
            allreduce.opaque_params["mpi_comm"] = static_cast<void *>(runtime.mpi_comm);

            // Calculate allreduce count
            // For Wo/down_proj: [seq_len, d_model] - full d_model dimension
            size_t count = static_cast<size_t>(runtime.batch_size) *
                           static_cast<size_t>(runtime.seq_len) *
                           static_cast<size_t>(runtime.d_model);
            allreduce.int_params["count"] = static_cast<int>(count);

            LOG_TRACE("[GraphResolver] Inserting allreduce after " << resolved.name);
            return allreduce;
        }

        case TPMode::ColumnParallel:
        {
            // Column-parallel projection on LM head: allgather the output
            // Note: For QKV and gate/up, we don't need allgather - the local
            // computation continues with local dimensions. Only LM head needs
            // allgather to collect full vocab logits.

            // Check if this is the LM head stage
            if (spec.type != StageType::LMHead)
            {
                // For other column-parallel stages (QKV, gate/up), no collective needed
                // The subsequent stages work with local dimensions
                return std::nullopt;
            }

            // LM head needs allgather to collect vocab slices
            ResolvedStage allgather;
            allgather.name = resolved.name + "_allgather";
            allgather.type = StageType::Allgather;
            allgather.device_idx = resolved.device_idx;
            allgather.dependencies.push_back(resolved.name);

            // Input is local logits, output is full logits
            // These should be different buffers
            // Input: logits_local [seq_len, vocab_local]
            // Output: logits [seq_len, vocab_size]
            if (!resolved.outputs.empty())
            {
                allgather.inputs.push_back(resolved.outputs[0]); // logits_local
            }
            // Full logits buffer should be in tensors.buffers["logits"]
            // but we need to resolve it - for now store the info
            allgather.int_params["world_size"] = runtime.world_size;
            allgather.int_params["seq_len"] = runtime.batch_size * runtime.seq_len;

            // MPI parameters - store MPI_Comm directly as void*
            allgather.opaque_params["mpi_comm"] = static_cast<void *>(runtime.mpi_comm);

            LOG_TRACE("[GraphResolver] Inserting allgather after " << resolved.name);
            return allgather;
        }

        case TPMode::ExpertParallel:
            // MoE expert parallelism - TODO for Phase 4e
            LOG_WARN("[GraphResolver] ExpertParallel not yet implemented");
            return std::nullopt;

        default:
            return std::nullopt;
        }
    }

    int GraphResolver::detectAttentionMode(const GraphResolverConfig &runtime) const
    {
        // Attention mode detection based on batch/seq/kv dimensions
        // Returns: 0 = prefill, 1 = decode_single, 2 = decode_batched

        int batch_size = runtime.batch_size;
        int seq_len = runtime.seq_len;
        int kv_len = runtime.cached_tokens;

        if (kv_len == 0)
        {
            // No cached tokens - this is prefill
            return 0; // PREFILL
        }
        else if (batch_size == 1 && seq_len == 1)
        {
            // Single-sequence decode
            return 1; // DECODE_SINGLE
        }
        else if (seq_len == 1)
        {
            // Batched decode
            return 2; // DECODE_BATCHED
        }
        else
        {
            // Batched prefill or variable-length
            return 0; // PREFILL (conservative)
        }
    }

    std::string GraphResolver::expandStageName(
        const std::string &base_name,
        int layer_idx) const
    {
        if (layer_idx < 0)
        {
            // Non-layer stage (embedding, final_norm, lm_head)
            return base_name;
        }

        return "layer" + std::to_string(layer_idx) + "_" + base_name;
    }

    TensorBase *GraphResolver::resolveTensorRef(
        const TensorRef &ref,
        int layer_idx,
        const TensorContext &tensors) const
    {
        return tensors.resolve(ref, layer_idx);
    }

    void GraphResolver::populateStageParams(
        ResolvedStage &resolved,
        const StageSpec &spec,
        const GraphResolverConfig &runtime,
        const TensorContext &tensors,
        int layer_idx) const
    {
        // Common parameters
        resolved.int_params["seq_len"] = runtime.batch_size * runtime.seq_len;
        resolved.int_params["batch_size"] = runtime.batch_size;
        resolved.int_params["d_model"] = runtime.d_model;

        // Stage-specific parameters
        switch (spec.type)
        {
        case StageType::RMSNorm:
            resolved.float_params["eps"] = spec.rms_norm_eps.value_or(runtime.rms_norm_eps);
            break;

        case StageType::RoPE:
            resolved.float_params["theta_base"] = spec.rope_theta.value_or(runtime.rope_theta);
            resolved.int_params["n_heads"] = runtime.local_n_heads > 0 ? runtime.local_n_heads : runtime.n_heads;
            resolved.int_params["n_kv_heads"] = runtime.local_n_kv_heads > 0 ? runtime.local_n_kv_heads : runtime.n_kv_heads;
            resolved.int_params["head_dim"] = runtime.head_dim;
            break;

        case StageType::AttentionCompute:
            resolved.bool_params["causal"] = spec.causal.value_or(true);
            resolved.int_params["window_size"] = spec.window_size.value_or(-1);
            resolved.int_params["n_heads"] = runtime.local_n_heads > 0 ? runtime.local_n_heads : runtime.n_heads;
            resolved.int_params["n_kv_heads"] = runtime.local_n_kv_heads > 0 ? runtime.local_n_kv_heads : runtime.n_kv_heads;
            resolved.int_params["head_dim"] = runtime.head_dim;
            resolved.int_params["kv_len"] = runtime.cached_tokens > 0 ? runtime.cached_tokens : runtime.seq_len;
            resolved.int_params["attention_mode"] = detectAttentionMode(runtime);
            resolved.tensor_params["kv_cache"] = nullptr; // Placeholder - actual cache passed separately
            break;

        case StageType::FusedQKVGEMM:
            resolved.int_params["m"] = runtime.batch_size * runtime.seq_len;
            resolved.int_params["k"] = runtime.d_model;
            // n dimensions come from weight shapes
            break;

        case StageType::FusedGateUpGEMM:
            resolved.int_params["m"] = runtime.batch_size * runtime.seq_len;
            resolved.int_params["k"] = runtime.d_model;
            break;

        case StageType::GEMM:
            resolved.int_params["m"] = runtime.batch_size * runtime.seq_len;
            resolved.float_params["alpha"] = 1.0f;
            resolved.float_params["beta"] = 0.0f;
            break;

        case StageType::Embedding:
            resolved.int_params["vocab_size"] = runtime.vocab_size;
            resolved.int_params["num_tokens"] = runtime.batch_size * runtime.seq_len;
            break;

        case StageType::LMHead:
            resolved.int_params["vocab_size"] = runtime.local_vocab > 0 ? runtime.local_vocab : runtime.vocab_size;
            break;

        case StageType::KVCacheAppend:
            resolved.int_params["layer_idx"] = layer_idx;
            resolved.int_params["num_tokens"] = runtime.batch_size * runtime.seq_len;
            break;

        case StageType::ResidualAdd:
            resolved.int_params["num_elements"] = runtime.batch_size * runtime.seq_len * runtime.d_model;
            break;

        case StageType::SwiGLU:
            // seq_len already set above
            break;

        default:
            // Other stage types - no additional params needed
            break;
        }
    }

    // =========================================================================
    // GraphBuilder Implementation
    // =========================================================================

    std::unique_ptr<IComputeStage> GraphBuilder::createStage(const ResolvedStage &stage)
    {
        // Extract common params
        int seq_len = stage.int_params.count("seq_len") ? stage.int_params.at("seq_len") : 0;
        int batch_size = stage.int_params.count("batch_size") ? stage.int_params.at("batch_size") : 1;
        int d_model = stage.int_params.count("d_model") ? stage.int_params.at("d_model") : 0;

        switch (stage.type)
        {
        case StageType::Embedding:
        {
            EmbeddingStage::Params params;
            params.embed_table = stage.inputs.size() > 1 ? stage.inputs[1] : nullptr;
            params.token_ids = stage.inputs.size() > 0 ? reinterpret_cast<const int *>(stage.inputs[0]) : nullptr;
            params.output = stage.outputs.size() > 0 ? stage.outputs[0] : nullptr;
            params.num_tokens = stage.int_params.count("num_tokens") ? stage.int_params.at("num_tokens") : seq_len;
            params.d_model = d_model;
            params.vocab_size = stage.int_params.count("vocab_size") ? stage.int_params.at("vocab_size") : 0;
            params.device_idx = stage.device_idx;
            return ComputeStageFactory::createEmbedding(params);
        }

        case StageType::RMSNorm:
        {
            RMSNormStage::Params params;
            params.input = stage.inputs.size() > 0 ? stage.inputs[0] : nullptr;
            params.gamma = stage.inputs.size() > 1 ? stage.inputs[1] : nullptr;
            params.output = stage.outputs.size() > 0 ? stage.outputs[0] : nullptr;
            params.eps = stage.float_params.count("eps") ? stage.float_params.at("eps") : 1e-6f;
            params.seq_len = seq_len;
            params.device_idx = stage.device_idx;
            return ComputeStageFactory::createRMSNorm(params);
        }

        case StageType::FusedQKVGEMM:
        {
            FusedQKVGEMMStage::Params params;
            params.input = stage.inputs.size() > 0 ? stage.inputs[0] : nullptr;
            params.wq = stage.inputs.size() > 1 ? stage.inputs[1] : nullptr;
            params.wk = stage.inputs.size() > 2 ? stage.inputs[2] : nullptr;
            params.wv = stage.inputs.size() > 3 ? stage.inputs[3] : nullptr;
            params.output_q = stage.outputs.size() > 0 ? stage.outputs[0] : nullptr;
            params.output_k = stage.outputs.size() > 1 ? stage.outputs[1] : nullptr;
            params.output_v = stage.outputs.size() > 2 ? stage.outputs[2] : nullptr;
            params.m = stage.int_params.count("m") ? stage.int_params.at("m") : seq_len;
            params.k = stage.int_params.count("k") ? stage.int_params.at("k") : d_model;
            // n dimensions come from weight shapes
            if (params.wq)
                params.n_q = static_cast<int>(params.wq->shape()[0]);
            if (params.wk)
                params.n_k = static_cast<int>(params.wk->shape()[0]);
            if (params.wv)
                params.n_v = static_cast<int>(params.wv->shape()[0]);
            // Biases (optional - indices 4,5,6 in inputs)
            params.bias_q = stage.inputs.size() > 4 && stage.inputs[4] ? stage.inputs[4]->data() : nullptr;
            params.bias_k = stage.inputs.size() > 5 && stage.inputs[5] ? stage.inputs[5]->data() : nullptr;
            params.bias_v = stage.inputs.size() > 6 && stage.inputs[6] ? stage.inputs[6]->data() : nullptr;
            return ComputeStageFactory::createFusedQKVGEMM(params);
        }

        case StageType::RoPE:
        {
            RoPEStage::Params params;
            params.Q = stage.inputs.size() > 0 ? stage.inputs[0] : nullptr;
            params.K = stage.inputs.size() > 1 ? stage.inputs[1] : nullptr;
            params.n_heads = stage.int_params.count("n_heads") ? stage.int_params.at("n_heads") : 0;
            params.n_kv_heads = stage.int_params.count("n_kv_heads") ? stage.int_params.at("n_kv_heads") : 0;
            params.head_dim = stage.int_params.count("head_dim") ? stage.int_params.at("head_dim") : 0;
            params.pos_offset = stage.int_params.count("pos_offset") ? stage.int_params.at("pos_offset") : 0;
            params.theta_base = stage.float_params.count("theta_base") ? stage.float_params.at("theta_base") : 10000.0f;
            params.seq_len = seq_len;
            return ComputeStageFactory::createRoPE(params);
        }

        case StageType::KVCacheAppend:
        {
            KVCacheAppendStage::Params params;
            params.K = stage.inputs.size() > 0 ? stage.inputs[0] : nullptr;
            params.V = stage.inputs.size() > 1 ? stage.inputs[1] : nullptr;
            // KV cache is passed via opaque_params
            if (stage.opaque_params.count("kv_cache"))
            {
                params.kv_cache = static_cast<IUnifiedKVCache *>(stage.opaque_params.at("kv_cache"));
            }
            params.layer_idx = stage.int_params.count("layer_idx") ? stage.int_params.at("layer_idx") : 0;
            params.seq_idx = 0;
            params.num_tokens = stage.int_params.count("num_tokens") ? stage.int_params.at("num_tokens") : seq_len;
            params.batch_size = batch_size;
            params.seq_len = seq_len / batch_size;
            return ComputeStageFactory::createKVCacheAppend(params);
        }

        case StageType::AttentionCompute:
        {
            AttentionComputeStage::Params params;
            params.Q = stage.inputs.size() > 0 ? stage.inputs[0] : nullptr;
            params.K = stage.inputs.size() > 1 ? stage.inputs[1] : nullptr;
            params.V = stage.inputs.size() > 2 ? stage.inputs[2] : nullptr;
            params.output = stage.outputs.size() > 0 ? stage.outputs[0] : nullptr;
            params.batch_size = batch_size;
            params.seq_len = seq_len / batch_size;
            params.kv_len = stage.int_params.count("kv_len") ? stage.int_params.at("kv_len") : seq_len;
            params.n_heads = stage.int_params.count("n_heads") ? stage.int_params.at("n_heads") : 0;
            params.n_kv_heads = stage.int_params.count("n_kv_heads") ? stage.int_params.at("n_kv_heads") : 0;
            params.head_dim = stage.int_params.count("head_dim") ? stage.int_params.at("head_dim") : 0;
            params.causal = stage.bool_params.count("causal") ? stage.bool_params.at("causal") : true;
            params.window_size = stage.int_params.count("window_size") ? stage.int_params.at("window_size") : -1;
            params.attention_mode = static_cast<AttentionMode>(
                stage.int_params.count("attention_mode") ? stage.int_params.at("attention_mode") : 0);
            params.auto_detect_mode = true;
            params.device_idx = stage.device_idx;
            return ComputeStageFactory::createAttentionCompute(params);
        }

        case StageType::GEMM:
        {
            GEMMStage::Params params;
            params.A = stage.inputs.size() > 0 ? stage.inputs[0] : nullptr;
            params.B = stage.inputs.size() > 1 ? stage.inputs[1] : nullptr;
            params.C = stage.outputs.size() > 0 ? stage.outputs[0] : nullptr;
            params.m = stage.int_params.count("m") ? stage.int_params.at("m") : seq_len;
            if (params.B)
            {
                params.n = static_cast<int>(params.B->shape()[0]);
                params.k = static_cast<int>(params.B->shape()[1]);
            }
            params.alpha = stage.float_params.count("alpha") ? stage.float_params.at("alpha") : 1.0f;
            params.beta = stage.float_params.count("beta") ? stage.float_params.at("beta") : 0.0f;
            params.transpose_B = false;
            return ComputeStageFactory::createGEMM(params);
        }

        case StageType::FusedGateUpGEMM:
        {
            FusedGateUpGEMMStage::Params params;
            params.input = stage.inputs.size() > 0 ? stage.inputs[0] : nullptr;
            params.w_gate = stage.inputs.size() > 1 ? stage.inputs[1] : nullptr;
            params.w_up = stage.inputs.size() > 2 ? stage.inputs[2] : nullptr;
            params.output_gate = stage.outputs.size() > 0 ? stage.outputs[0] : nullptr;
            params.output_up = stage.outputs.size() > 1 ? stage.outputs[1] : nullptr;
            params.m = stage.int_params.count("m") ? stage.int_params.at("m") : seq_len;
            params.k = stage.int_params.count("k") ? stage.int_params.at("k") : d_model;
            if (params.w_gate)
                params.n_gate = static_cast<int>(params.w_gate->shape()[0]);
            if (params.w_up)
                params.n_up = static_cast<int>(params.w_up->shape()[0]);
            params.device_idx = stage.device_idx;
            return ComputeStageFactory::createFusedGateUpGEMM(params);
        }

        case StageType::SwiGLU:
        {
            SwiGLUStage::Params params;
            params.gate = stage.inputs.size() > 0 ? stage.inputs[0] : nullptr;
            params.up = stage.inputs.size() > 1 ? stage.inputs[1] : nullptr;
            params.output = stage.outputs.size() > 0 ? stage.outputs[0] : nullptr;
            params.seq_len = seq_len;
            return ComputeStageFactory::createSwiGLU(params);
        }

        case StageType::ResidualAdd:
        {
            ResidualAddStage::Params params;
            params.input = stage.inputs.size() > 0 ? stage.inputs[0] : nullptr;
            params.residual = stage.inputs.size() > 1 ? stage.inputs[1] : nullptr;
            params.output = stage.outputs.size() > 0 ? stage.outputs[0] : nullptr;
            params.num_elements = stage.int_params.count("num_elements")
                                      ? static_cast<size_t>(stage.int_params.at("num_elements"))
                                      : static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
            return ComputeStageFactory::createResidualAdd(params);
        }

        case StageType::LMHead:
        {
            LMHeadStage::Params params;
            params.hidden_states = stage.inputs.size() > 0 ? stage.inputs[0] : nullptr;
            params.lm_head_weight = stage.inputs.size() > 1 ? stage.inputs[1] : nullptr;
            params.logits = stage.outputs.size() > 0 ? stage.outputs[0] : nullptr;
            params.seq_len = seq_len;
            params.d_model = d_model;
            params.vocab_size = stage.int_params.count("vocab_size") ? stage.int_params.at("vocab_size") : 0;
            params.bias = nullptr;
            params.device_idx = stage.device_idx;
            return ComputeStageFactory::createLMHead(params);
        }

        case StageType::Allreduce:
        {
            AllreduceStage::Params params;
            params.buffer = stage.inputs.size() > 0 ? stage.inputs[0] : nullptr;
            if (stage.opaque_params.count("mpi_comm"))
            {
                params.mpi_comm = static_cast<MPI_Comm>(stage.opaque_params.at("mpi_comm"));
            }
            params.count = stage.int_params.count("count") ? static_cast<size_t>(stage.int_params.at("count")) : 0;
            return ComputeStageFactory::createAllreduce(params);
        }

        case StageType::Allgather:
        {
            AllGatherStage::Params params;
            params.local_input = stage.inputs.size() > 0 ? stage.inputs[0] : nullptr;
            params.full_output = stage.outputs.size() > 0 ? stage.outputs[0] : nullptr;
            if (stage.opaque_params.count("mpi_comm"))
            {
                params.mpi_comm = static_cast<MPI_Comm>(stage.opaque_params.at("mpi_comm"));
            }
            params.world_size = stage.int_params.count("world_size") ? stage.int_params.at("world_size") : 1;
            params.actual_seq_len = stage.int_params.count("actual_seq_len")
                                        ? static_cast<size_t>(stage.int_params.at("actual_seq_len"))
                                        : static_cast<size_t>(seq_len);
            return ComputeStageFactory::createAllGather(params);
        }

        default:
            LOG_WARN("[GraphBuilder] Unknown stage type: " << static_cast<int>(stage.type)
                                                           << " for stage " << stage.name);
            return nullptr;
        }
    }

    ComputeGraph GraphBuilder::build(const ResolvedGraphSpec &spec)
    {
        ComputeGraph graph;

        LOG_DEBUG("[GraphBuilder] Building graph from " << spec.stages.size() << " resolved stages");

        for (const auto &stage : spec.stages)
        {
            // Create the compute stage using the factory method
            std::unique_ptr<IComputeStage> compute_stage = createStage(stage);

            // Add node to graph
            graph.addNode(stage.name, std::move(compute_stage), stage.device_idx);

            // Add dependencies
            for (const auto &dep : stage.dependencies)
            {
                graph.addDependency(stage.name, dep);
            }
        }

        LOG_DEBUG("[GraphBuilder] Built graph with " << graph.size() << " nodes");

        return graph;
    }

    // =========================================================================
    // BufferAllocator Implementation
    // =========================================================================

    size_t ResolvedBufferSpec::totalBytes() const
    {
        if (shape.empty())
            return 0;

        size_t total = 1;
        for (size_t dim : shape)
        {
            total *= dim;
        }

        // Calculate bytes per element based on dtype
        size_t bytes_per_element = 4; // default to fp32
        if (dtype == "fp32" || dtype == "float32")
            bytes_per_element = 4;
        else if (dtype == "bf16" || dtype == "bfloat16" || dtype == "fp16" || dtype == "float16")
            bytes_per_element = 2;
        else if (dtype == "q8_0" || dtype == "int8")
            bytes_per_element = 1;
        else if (dtype == "q4_0" || dtype == "q4_1" || dtype == "iq4_nl")
            bytes_per_element = 1; // 4-bit, but we round up

        return total * bytes_per_element;
    }

    size_t BufferAllocator::evaluateFormula(
        const std::string &formula,
        const GraphResolverConfig &config)
    {
        // Direct dimension lookups
        if (formula == "d_model" || formula == "hidden_size")
            return static_cast<size_t>(config.d_model);
        if (formula == "seq_len" || formula == "sequence_length")
            return static_cast<size_t>(config.seq_len);
        if (formula == "n_heads" || formula == "num_attention_heads")
            return static_cast<size_t>(config.n_heads);
        if (formula == "n_kv_heads" || formula == "num_key_value_heads")
            return static_cast<size_t>(config.n_kv_heads);
        if (formula == "head_dim")
            return static_cast<size_t>(config.head_dim);
        if (formula == "n_layers" || formula == "num_layers")
            return static_cast<size_t>(config.n_layers);
        if (formula == "d_ff" || formula == "intermediate_size")
            return static_cast<size_t>(config.d_ff);
        if (formula == "vocab_size")
            return static_cast<size_t>(config.vocab_size);

        // Tensor-parallel (local) dimensions
        if (formula == "local_n_heads")
            return static_cast<size_t>(config.local_n_heads);
        if (formula == "local_n_kv_heads")
            return static_cast<size_t>(config.local_n_kv_heads);
        if (formula == "local_d_ff")
            return static_cast<size_t>(config.local_d_ff);
        if (formula == "local_vocab")
            return static_cast<size_t>(config.local_vocab);

        // Computed dimensions
        if (formula == "qkv_dim" || formula == "n_heads*head_dim")
            return static_cast<size_t>(config.n_heads * config.head_dim);
        if (formula == "kv_dim" || formula == "n_kv_heads*head_dim")
            return static_cast<size_t>(config.n_kv_heads * config.head_dim);
        if (formula == "local_qkv_dim" || formula == "local_n_heads*head_dim")
            return static_cast<size_t>(config.local_n_heads * config.head_dim);
        if (formula == "local_kv_dim" || formula == "local_n_kv_heads*head_dim")
            return static_cast<size_t>(config.local_n_kv_heads * config.head_dim);

        // Try to parse as a literal number
        try
        {
            return static_cast<size_t>(std::stol(formula));
        }
        catch (...)
        {
            LOG_WARN("[BufferAllocator] Unknown formula: " << formula << ", returning 0");
            return 0;
        }
    }

    ResolvedBufferSpec BufferAllocator::resolve(
        const BufferSpec &spec,
        const GraphResolverConfig &config)
    {
        ResolvedBufferSpec resolved;
        resolved.name = spec.name;
        resolved.dtype = spec.dtype;
        resolved.semantic = spec.semantic;
        resolved.alias_group = spec.alias_group;
        resolved.alias_priority = spec.alias_priority;
        resolved.description = spec.description;
        resolved.device_idx = 0; // Default device

        // Resolve each shape dimension
        resolved.shape.reserve(spec.shape.size());
        for (const auto &formula : spec.shape)
        {
            resolved.shape.push_back(evaluateFormula(formula, config));
        }

        return resolved;
    }

    std::vector<ResolvedBufferSpec> BufferAllocator::resolveAll(
        const GraphSchema &schema,
        const GraphResolverConfig &config)
    {
        std::vector<ResolvedBufferSpec> resolved;
        resolved.reserve(schema.layer_buffers.size() + schema.model_buffers.size());

        // Resolve layer buffers
        for (const auto &spec : schema.layer_buffers)
        {
            resolved.push_back(resolve(spec, config));
        }

        // Resolve model buffers
        for (const auto &spec : schema.model_buffers)
        {
            resolved.push_back(resolve(spec, config));
        }

        LOG_DEBUG("[BufferAllocator] Resolved " << resolved.size()
                                                << " buffers from schema '" << schema.name << "'");

        return resolved;
    }

    std::vector<AliasGroupSpec> BufferAllocator::resolveAliasGroups(
        const GraphSchema &schema,
        const GraphResolverConfig &config)
    {
        // Return alias groups from schema - they're already declarative
        // We could enhance this to compute buffer sizes per group
        return schema.alias_groups;
    }

    std::pair<size_t, size_t> BufferAllocator::estimateMemorySavings(
        const GraphSchema &schema,
        const GraphResolverConfig &config)
    {
        // Resolve all buffers
        auto resolved = resolveAll(schema, config);

        // Calculate original bytes (all buffers separately)
        size_t original_bytes = 0;
        for (const auto &buf : resolved)
        {
            original_bytes += buf.totalBytes();
        }

        // Calculate optimized bytes (max per alias group)
        std::unordered_map<std::string, size_t> group_max_bytes;
        size_t non_aliased_bytes = 0;

        for (const auto &buf : resolved)
        {
            size_t bytes = buf.totalBytes();
            if (buf.alias_group.empty())
            {
                non_aliased_bytes += bytes;
            }
            else
            {
                group_max_bytes[buf.alias_group] =
                    std::max(group_max_bytes[buf.alias_group], bytes);
            }
        }

        size_t optimized_bytes = non_aliased_bytes;
        for (const auto &[group, max_bytes] : group_max_bytes)
        {
            optimized_bytes += max_bytes;
        }

        return {original_bytes, optimized_bytes};
    }

    StageBufferRequirements BufferAllocator::toBufferRequirements(
        const std::vector<ResolvedBufferSpec> &resolved)
    {
        StageBufferRequirements reqs;
        reqs.buffers.reserve(resolved.size());

        for (const auto &buf : resolved)
        {
            // Convert dtype string to BufferTensorType
            BufferTensorType tensor_type = BufferTensorType::FP32;
            if (buf.dtype == "bf16" || buf.dtype == "bfloat16")
                tensor_type = BufferTensorType::BF16;
            else if (buf.dtype == "fp16" || buf.dtype == "float16")
                tensor_type = BufferTensorType::FP16;
            else if (buf.dtype == "q8_0")
                tensor_type = BufferTensorType::Q8_0;

            // Convert semantic to role
            BufferRole role = BufferRole::SCRATCH;
            switch (buf.semantic)
            {
            case BufferSemantic::Input:
                role = BufferRole::INPUT;
                break;
            case BufferSemantic::Output:
                role = BufferRole::OUTPUT;
                break;
            case BufferSemantic::InOut:
                role = BufferRole::INOUT;
                break;
            case BufferSemantic::Scratch:
                role = BufferRole::SCRATCH;
                break;
            default:
                role = BufferRole::SCRATCH;
                break;
            }

            BufferDescriptor desc;
            desc.name = buf.name;
            desc.shape = buf.shape;
            desc.tensor_type = tensor_type;
            desc.role = role;
            desc.device_idx = buf.device_idx;

            reqs.buffers.push_back(std::move(desc));
        }

        return reqs;
    }

    StageBufferRequirements BufferAllocator::resolveLayerBuffers(
        const GraphSchema &schema,
        const GraphResolverConfig &config)
    {
        std::vector<ResolvedBufferSpec> resolved;
        resolved.reserve(schema.layer_buffers.size());

        for (const auto &spec : schema.layer_buffers)
        {
            resolved.push_back(resolve(spec, config));
        }

        LOG_DEBUG("[BufferAllocator] Resolved " << resolved.size()
                                                << " layer buffers from schema '" << schema.name << "'");

        return toBufferRequirements(resolved);
    }

    StageBufferRequirements BufferAllocator::resolveModelBuffers(
        const GraphSchema &schema,
        const GraphResolverConfig &config)
    {
        std::vector<ResolvedBufferSpec> resolved;
        resolved.reserve(schema.model_buffers.size());

        for (const auto &spec : schema.model_buffers)
        {
            resolved.push_back(resolve(spec, config));
        }

        LOG_DEBUG("[BufferAllocator] Resolved " << resolved.size()
                                                << " model buffers from schema '" << schema.name << "'");

        return toBufferRequirements(resolved);
    }

} // namespace llaminar2
