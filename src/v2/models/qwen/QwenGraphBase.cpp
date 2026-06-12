/**
 * @file QwenGraphBase.cpp
 * @brief Implementation of shared Qwen-family graph builder infrastructure
 * @author David Sanftenberg
 * @date January 2026
 *
 * This file implements QwenGraphBase, the common base class for all
 * Qwen-family graph builders (Qwen2, Qwen3, Qwen3.5).
 * Contains shared infrastructure: forward graph construction, FFN,
 * embedding, LM head, TP/PP support, arena management.
 */

#include "QwenGraphBase.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"
#include "../../tensors/TensorSlice.h"
#include "../../tensors/Tensors.h"
#include "../../utils/MPIContext.h"
#include "../../utils/MPIStrategy.h"
#include "../../execution/local_execution/graph/GraphBuildUtils.h"
#include "../../execution/config/RuntimeConfig.h"
#include "../../loaders/PreparedWeightStore.h"
#include "../../collective/ILocalTPContext.h"
#include "../../collective/ITPContext.h"
#include "../../collective/ILocalPPContext.h"
#include "../../collective/ITPContext.h"
#include "../../collective/BackendRouter.h"
#include "../../execution/compute_stages/stages/TPAllreduceStage.h"
#include "../../execution/compute_stages/stages/LocalPPTransferStage.h"
#include "../../execution/compute_stages/stages/FusedResidualNormStage.h"
#include "../../execution/compute_stages/stages/QKNormStage.h"
#include "../../execution/mtp/MTPSpecDecodeMetadata.h"
#include "../../config/PipelineConfig.h"
#include "../../memory/BufferId.h" // Phase 2: contract BufferIds
#include <algorithm>
#include <chrono>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        int embeddingVocabOffsetForDevice(const GraphConfig &config, DeviceId device)
        {
            if (!config.tp_config)
                return 0;

            // NodeLocalTP uses one MPI rank per CPU socket, but DeviceId::cpu()
            // is a singleton.  Matching by device would always find rank 0's
            // assignment, so prefer the current rank's assignment when it owns
            // the requested device.
            if (const auto *assignment = config.getAssignment())
            {
                if (assignment->device == device)
                    return assignment->vocab_start;
            }

            for (const auto &assignment : config.tp_config->assignments())
            {
                if (assignment.device == device)
                    return assignment.vocab_start;
            }
            return 0;
        }

        BufferId logitsBufferId(bool column_parallel, bool all_positions)
        {
            if (all_positions)
            {
                return column_parallel ? BufferId::ALL_POSITION_LOGITS_LOCAL
                                       : BufferId::ALL_POSITION_LOGITS;
            }
            return column_parallel ? BufferId::LOGITS_LOCAL : BufferId::LOGITS;
        }

        BufferId gatheredLogitsBufferId(bool all_positions)
        {
            return all_positions ? BufferId::ALL_POSITION_LOGITS : BufferId::LOGITS;
        }

        /**
         * @brief Describes the LM-head input shape and buffer contract.
         *
         * Normal decode/prefill passes the whole normalized activation tensor to
         * LMHeadStage and lets that stage read only the final row. Full
         * all-position verification projects every normalized row. The vLLM-style
         * compact verifier path is the middle ground: row-select first packs the
         * verifier rows into LM_HEAD_INPUT_ROWS, then LMHeadStage projects every
         * row of that compact scratch tensor.
         */
        struct LMHeadInputLayout
        {
            int seq_len = 1;
            BufferId input_buffer_id = BufferId::NORMALIZED;
            bool compute_all_positions = false;
            bool use_prefill_replay_row_offset = true;
        };

        /**
         * @brief Resolve LMHeadStage shape/contract fields from the selected input tensor.
         *
         * Keeping this logic in one helper prevents the single-device, PP, and
         * schema-built graph paths from drifting as we add verifier modes.
         */
        LMHeadInputLayout describeLMHeadInputLayout(
            const GraphConfig &config,
            const ActivationBuffers &buffers,
            TensorBase *lm_head_input,
            int total_tokens)
        {
            const TensorBase *normalized = buffers.normalized;
            const TensorBase *compact_rows = buffers.get(BufferId::LM_HEAD_INPUT_ROWS);
            const bool input_is_normalized = lm_head_input == normalized;
            const bool input_is_compact_rows =
                config.compute_all_position_logits &&
                config.compute_row_indexed_logits &&
                lm_head_input == compact_rows;

            LMHeadInputLayout layout;
            if (input_is_compact_rows)
            {
                layout.seq_len = config.row_indexed_logits_row_count;
                layout.input_buffer_id = BufferId::LM_HEAD_INPUT_ROWS;
                layout.compute_all_positions = true;
                // Compact rows already start at row 0, so replay row offsets
                // would be wrong and are intentionally disabled here.
                layout.use_prefill_replay_row_offset = false;
                return layout;
            }

            if (input_is_normalized)
            {
                layout.seq_len = total_tokens;
                layout.input_buffer_id = BufferId::NORMALIZED;
                layout.compute_all_positions = config.compute_all_position_logits;
                layout.use_prefill_replay_row_offset = true;
                return layout;
            }

            layout.seq_len = 1;
            layout.input_buffer_id = BufferId::LM_HEAD_INPUT_ROW;
            layout.compute_all_positions = false;
            layout.use_prefill_replay_row_offset = false;
            return layout;
        }
    }

    // Import graph_utils for cleaner code
    using namespace graph_utils;

    // =============================================================================
    // Constructors
    // =============================================================================

    QwenGraphBase::QwenGraphBase(std::shared_ptr<ModelContext> model_ctx,
                                 std::shared_ptr<IMPIContext> mpi_ctx,
                                 const GraphConfig &config)
        : config_(config),
          model_ctx_(std::move(model_ctx)),
          mpi_ctx_(std::move(mpi_ctx))
    {
        LOG_DEBUG("[QwenGraphBase] Initializing (full): n_layers=" << config_.n_layers
                                                                   << " d_model=" << config_.d_model
                                                                   << " vocab_size=" << config_.vocab_size
                                                                   << " d_ff=" << config_.d_ff
                                                                   << " ffn_column_parallel=" << config_.ffn_column_parallel
                                                                   << " n_heads=" << config_.n_heads
                                                                   << " n_kv_heads=" << config_.n_kv_heads);

        LOG_DEBUG("[QwenGraphBase] Initialized (full) with " << config_.n_layers
                                                             << " layers, precision="
                                                             << activationPrecisionToString(config_.activation_precision));
    }

    QwenGraphBase::QwenGraphBase(const GraphConfig &config,
                                 std::shared_ptr<IMPIContext> mpi_ctx)
        : config_(config),
          model_ctx_(nullptr),
          mpi_ctx_(std::move(mpi_ctx))
    {
        LOG_DEBUG("[QwenGraphBase] Initializing (layer-only): d_model=" << config_.d_model
                                                                        << " d_ff=" << config_.d_ff
                                                                        << " ffn_column_parallel=" << config_.ffn_column_parallel
                                                                        << " n_heads=" << config_.n_heads
                                                                        << " n_kv_heads=" << config_.n_kv_heads
                                                                        << " mpi_ctx=" << (mpi_ctx_ ? "valid" : "nullptr")
                                                                        << " world_size=" << (mpi_ctx_ ? mpi_ctx_->world_size() : -1)
                                                                        << " default_device=" << config_.default_device.to_string());

        LOG_DEBUG("[QwenGraphBase] Initialized (layer-only)");
    }

    void QwenGraphBase::setModelContext(std::shared_ptr<IModelContext> model_ctx)
    {
        model_ctx_ = std::dynamic_pointer_cast<ModelContext>(std::move(model_ctx));
        if (!model_ctx_)
        {
            LOG_DEBUG("[QwenGraphBase] setModelContext ignored non-ModelContext instance");
        }
    }

    bool QwenGraphBase::hasLayerWeightSource() const
    {
        return weight_bindings_.get_layer_weights != nullptr || weights_.get_layer_weights != nullptr;
    }

    LayerWeightBindings QwenGraphBase::layerWeightBindingsForGraph(int layer_idx) const
    {
        if (weight_bindings_.get_layer_weights)
            return weight_bindings_.get_layer_weights(layer_idx);
        return {};
    }

    LayerWeights QwenGraphBase::layerWeightsForGraph(int layer_idx) const
    {
        if (weight_bindings_.get_layer_weights)
            return toLegacyLayerWeights(weight_bindings_.get_layer_weights(layer_idx));
        if (weights_.get_layer_weights)
            return weights_.get_layer_weights(layer_idx);
        return {};
    }

    TensorBase *QwenGraphBase::modelEmbeddingTable() const
    {
        TensorBase *bound = legacyTensor(weight_bindings_.embedding_table);
        return bound ? bound : weights_.embedding_table;
    }

    TensorBase *QwenGraphBase::modelFinalNorm() const
    {
        TensorBase *bound = legacyTensor(weight_bindings_.final_norm);
        return bound ? bound : weights_.final_norm;
    }

    TensorBase *QwenGraphBase::modelLMHead() const
    {
        TensorBase *bound = legacyTensor(weight_bindings_.lm_head);
        return bound ? bound : weights_.lm_head;
    }

    const WeightBinding *QwenGraphBase::modelEmbeddingBinding() const
    {
        return weight_bindings_.embedding_table;
    }

    const WeightBinding *QwenGraphBase::modelFinalNormBinding() const
    {
        return weight_bindings_.final_norm;
    }

    const WeightBinding *QwenGraphBase::modelLMHeadBinding() const
    {
        return weight_bindings_.lm_head;
    }

    std::optional<PreparedWeightRef> QwenGraphBase::preparedRefForGraphWeight(
        const WeightBinding *binding,
        DeviceId device) const
    {
        // Model GEMM stages must be wired from frozen WeightBinding ids. The
        // store lookup is intentionally binding-first so cloned or sliced tensor
        // pointers cannot accidentally select a prepared handle for a different
        // graph binding, device, PP stage, or TP shard.
        if (binding && prepared_weight_store_)
        {
            auto ref = prepared_weight_store_->preparedRefForBinding(binding->binding_id, device);
            if (ref.has_value())
                return ref;
        }
        if (binding && binding->prepared.has_value() && binding->prepared->device == device)
            return binding->prepared;
        return std::nullopt;
    }

    bool QwenGraphBase::hasActiveExpertMask(const std::vector<bool> &expert_mask) const
    {
        return std::any_of(expert_mask.begin(), expert_mask.end(), [](bool active)
                           { return active; });
    }

    std::string QwenGraphBase::describeMissingExpertGemmEngine(
        int num_experts,
        const std::vector<bool> &expert_mask,
        const std::vector<ITensorGemm *> &gate_gemm,
        const std::vector<ITensorGemm *> &up_gemm,
        const std::vector<ITensorGemm *> &down_gemm) const
    {
        for (int expert = 0; expert < num_experts; ++expert)
        {
            if (!expert_mask.empty())
            {
                if (expert >= static_cast<int>(expert_mask.size()))
                    return "expert=" + std::to_string(expert) + " role=mask";
                if (!expert_mask[expert])
                    continue;
            }

            if (expert >= static_cast<int>(gate_gemm.size()) || gate_gemm[expert] == nullptr)
                return "expert=" + std::to_string(expert) + " role=gate";
            if (expert >= static_cast<int>(up_gemm.size()) || up_gemm[expert] == nullptr)
                return "expert=" + std::to_string(expert) + " role=up";
            if (expert >= static_cast<int>(down_gemm.size()) || down_gemm[expert] == nullptr)
                return "expert=" + std::to_string(expert) + " role=down";
        }

        return "unknown missing expert engine";
    }

    void QwenGraphBase::failMissingGpuExpertGemmEngines(
        DeviceId device,
        int layer_idx,
        const std::string &reason) const
    {
        const std::string message = "[" + architectureName() + "] Missing GPU expert GEMM engines for layer " +
                                    std::to_string(layer_idx) + " on " + device.to_string() + ": " + reason;
        LOG_ERROR(message);
        throw std::runtime_error(message);
    }

    // =============================================================================
    // Arena-Aware Buffer Resolution
    // =============================================================================

    void QwenGraphBase::setArena(BufferArena *arena)
    {
        arena_ = arena;

        if (!arena_)
            return;

        // Populate buffers_ from arena so all graph-building code can continue
        // using buffers_.* paths without change.  This replaces the orchestrator's
        // bindArenaToManagedBuffers() shim.
        auto toBase = [](ITensor *t) -> TensorBase *
        {
            return t ? dynamic_cast<TensorBase *>(t) : nullptr;
        };

        // Layer activation buffers
        auto &lb = buffers_.layer_buffers;
        lb.residual = toBase(arena_->getTensor(BufferId::RESIDUAL));
        lb.normalized = toBase(arena_->getTensor(BufferId::NORMALIZED));
        lb.Q = toBase(arena_->getTensor(BufferId::Q_PROJ));
        lb.K = toBase(arena_->getTensor(BufferId::K_PROJ));
        lb.V = toBase(arena_->getTensor(BufferId::V_PROJ));
        lb.attn_output = toBase(arena_->getTensor(BufferId::ATTN_OUTPUT));
        lb.attn_proj = toBase(arena_->getTensor(BufferId::ATTN_PROJ));
        lb.workspace_scores = toBase(arena_->getTensor(BufferId::ATTN_SCORES_WORKSPACE));
        lb.workspace_context = toBase(arena_->getTensor(BufferId::ATTN_CONTEXT_WORKSPACE));
        lb.workspace_mask = toBase(arena_->getTensor(BufferId::GEMM_WORKSPACE));
        lb.gate = toBase(arena_->getTensor(BufferId::GATE_PROJ));
        lb.up = toBase(arena_->getTensor(BufferId::UP_PROJ));
        lb.ffn_output = toBase(arena_->getTensor(BufferId::FFN_OUTPUT));

        // Hybrid mode buffers
        lb.Q_rope = toBase(arena_->getTensor(BufferId::Q_ROPE));
        lb.K_rope = toBase(arena_->getTensor(BufferId::K_ROPE));
        lb.V_dequant = toBase(arena_->getTensor(BufferId::V_DEQUANT));

        // Model-level buffers
        buffers_.current_hidden = toBase(arena_->getTensor(BufferId::HIDDEN_STATE));
        buffers_.logits = toBase(arena_->getTensor(BufferId::LOGITS));
        buffers_.logits_local = toBase(arena_->getTensor(BufferId::LOGITS_LOCAL));

        // Ensure current_hidden alias in layer_buffers (expected by some stages)
        lb.current_hidden = buffers_.current_hidden;
        if (auto *scratch_row = toBase(arena_->getTensor(BufferId::LM_HEAD_INPUT_ROW)))
            lb.extensions[BufferId::LM_HEAD_INPUT_ROW] = scratch_row;
        if (auto *scratch_rows = toBase(arena_->getTensor(BufferId::LM_HEAD_INPUT_ROWS)))
            lb.extensions[BufferId::LM_HEAD_INPUT_ROWS] = scratch_rows;

        LOG_DEBUG("[QwenGraphBase] Arena bound: "
                  << "residual=" << lb.residual
                  << " Q=" << lb.Q
                  << " gate=" << lb.gate
                  << " current_hidden=" << buffers_.current_hidden
                  << " logits=" << buffers_.logits);
    }

    const ModelBuffers &QwenGraphBase::buffers() const
    {
        return buffers_;
    }

    TensorBase *QwenGraphBase::maybeAddLMHeadRowSelect(
        ComputeGraph &graph,
        const std::string &dependency_node,
        TensorBase *final_norm_output,
        int total_tokens,
        int real_seq_len,
        int bucket_seq_len,
        DeviceId device,
        std::string &dependency_out,
        BufferId input_buffer_id) const
    {
        dependency_out = dependency_node;

        if (config_.compute_all_position_logits)
        {
            if (!config_.compute_row_indexed_logits)
                return final_norm_output;

            const int row_count = config_.row_indexed_logits_row_count;
            if (row_count <= 0 || row_count > total_tokens || row_count > 4)
            {
                LOG_ERROR("[QwenGraphBase] Row-indexed all-position logits require 1..min(4,total_tokens) rows, got "
                          << row_count << " for total_tokens=" << total_tokens);
                throw std::runtime_error("invalid row-indexed all-position logits row count");
            }

            TensorBase *scratch_rows = buffers_.layer_buffers.get(BufferId::LM_HEAD_INPUT_ROWS);
            if (!scratch_rows)
            {
                LOG_ERROR("[QwenGraphBase] Row-indexed all-position logits require lm_head_input_rows scratch buffer");
                throw std::runtime_error("row-indexed all-position logits scratch buffer missing");
            }

            // The verifier sequence starts at the current live position. Row i
            // contains the main-model logits after consuming verifier token i,
            // exactly the rows consumed by the greedy/stochastic verifier logic.
            std::vector<int> selected_rows;
            selected_rows.reserve(static_cast<size_t>(row_count));
            for (int i = 0; i < row_count; ++i)
                selected_rows.push_back(i);

            HiddenStateRowsSelectStage::Params row_params;
            row_params.input = final_norm_output;
            row_params.output = scratch_rows;
            row_params.seq_len = total_tokens;
            row_params.d_model = config_.d_model;
            row_params.selected_row_count = row_count;
            row_params.selected_row_indices = std::move(selected_rows);
            row_params.device_id = device;
            row_params.input_buffer_id = input_buffer_id;
            row_params.output_buffer_id = BufferId::LM_HEAD_INPUT_ROWS;
            if (device.is_gpu())
            {
                row_params.workspace_buffer_name =
                    MTPSpecDecodeWorkspaceBuffers::VERIFIER_LOGIT_ROWS;
                row_params.declare_selected_rows_workspace = false;
                row_params.upload_selected_rows_to_workspace = false;
            }

            graph.addNode("lm_head_rows_select",
                          ComputeStageFactory::createHiddenStateRowsSelect(row_params),
                          device);
            graph.addDependency("lm_head_rows_select", dependency_node);
            dependency_out = "lm_head_rows_select";
            return scratch_rows;
        }

        // Only bucketed prefill needs a dynamic row-select. Normal exact-shape
        // prefill and decode preserve the existing LMHeadStage offset behavior.
        const bool bucketed_prefill = bucket_seq_len > 0 && bucket_seq_len == total_tokens && total_tokens > 1;
        if (!bucketed_prefill)
            return final_norm_output;

        TensorBase *scratch_row = buffers_.layer_buffers.get(BufferId::LM_HEAD_INPUT_ROW);
        if (!scratch_row)
        {
            LOG_ERROR("[QwenGraphBase] Bucketed prefill LM head requires lm_head_input_row scratch buffer");
            throw std::runtime_error("Bucketed prefill LM head row-select scratch buffer missing");
        }

        // Initial execution uses the real count available at graph build. Later
        // cache hits call updatePrefillReplayParams() on the stage, which mutates
        // the captured pinned scalar before graph replay.
        const int initial_real_seq_len = real_seq_len > 0 ? real_seq_len : total_tokens;
        HiddenStateRowSelectStage::Params row_params;
        row_params.input = final_norm_output;
        row_params.output = scratch_row;
        row_params.seq_len = total_tokens;
        row_params.d_model = config_.d_model;
        row_params.selected_row_idx = initial_real_seq_len - 1;
        row_params.device_id = device;
        row_params.input_buffer_id = input_buffer_id;
        row_params.output_buffer_id = BufferId::LM_HEAD_INPUT_ROW;

        graph.addNode("lm_head_row_select",
                      ComputeStageFactory::createHiddenStateRowSelect(row_params),
                      device);
        graph.addDependency("lm_head_row_select", dependency_node);
        dependency_out = "lm_head_row_select";
        return scratch_row;
    }

    // =============================================================================
    // GraphConfig Helper Methods
    // =============================================================================

    bool GraphConfig::hasUnifiedPP() const
    {
        return pipeline_config != nullptr && pipeline_config->hasPP();
    }

    DeviceId GraphConfig::getDeviceForLayer(int layer_idx) const
    {
        if (pipeline_config)
        {
            return pipeline_config->getDeviceForLayer(layer_idx);
        }
        return default_device;
    }

    // =============================================================================
    // IGraphBuilder Interface Implementation
    // =============================================================================

    ComputeGraph QwenGraphBase::buildForwardGraph(
        const ForwardInput &input,
        ForwardOutput &output)
    {
        // Adapt generic ForwardInput to ForwardInput
        ForwardInput qwen_input;
        qwen_input.token_ids = input.token_ids;
        qwen_input.token_ids_device = input.token_ids_device;
        qwen_input.position_ids = input.position_ids;
        qwen_input.batch_size = input.batch_size;
        qwen_input.seq_len = input.seq_len;
        qwen_input.real_seq_len = input.real_seq_len;
        qwen_input.bucket_seq_len = input.bucket_seq_len;
        qwen_input.token_offset = input.token_offset;
        qwen_input.position_offset = input.position_offset;
        qwen_input.device = input.device;
        qwen_input.kv_cache = input.kv_cache;
        qwen_input.sequence_lengths = input.sequence_lengths;

        // Adapt generic ForwardOutput to ForwardOutput
        ForwardOutput qwen_output;
        qwen_output.logits = output.logits;
        qwen_output.hidden = output.hidden;

        // Build the graph
        ComputeGraph graph = buildFullForwardGraph(qwen_input, qwen_output);

        // Copy back output pointers (in case they were set by the builder)
        output.logits = qwen_output.logits;
        output.hidden = qwen_output.hidden;

        return graph;
    }

    ComputeGraph QwenGraphBase::buildLayerGraph(const LayerContext &ctx)
    {
        // Validate layer index against total model layers (supports absolute PP indices)
        int max_layers = config_.total_n_layers > 0 ? config_.total_n_layers : config_.n_layers;
        if (ctx.layer_idx < 0 || ctx.layer_idx >= max_layers)
        {
            LOG_ERROR("[QwenGraphBase::buildLayerGraph] Invalid layer index: " << ctx.layer_idx);
            return ComputeGraph{};
        }

        // Get layer weights
        if (!hasLayerWeightSource())
        {
            LOG_ERROR("[QwenGraphBase::buildLayerGraph] Layer weight accessor not set");
            return ComputeGraph{};
        }

        LayerWeights layer_weights = layerWeightsForGraph(ctx.layer_idx);

        // Build attention graph
        ComputeGraph attn_graph = buildAttentionGraph(
            layer_weights, buffers_.layer_buffers, ctx.layer_idx, ctx.seq_len,
            ctx.batch_size, ctx.kv_cache, ctx.position_ids, ctx.device,
            ctx.sequence_lengths);

        // Build FFN graph
        ComputeGraph ffn_graph = buildFFNGraph(
            layer_weights, buffers_.layer_buffers, ctx.layer_idx, ctx.seq_len,
            ctx.batch_size, ctx.device);

        // Merge: attention -> FFN
        std::string attn_last = attn_graph.terminalNode();
        attn_graph.merge(std::move(ffn_graph), attn_last);

        return attn_graph;
    }

    // =============================================================================
    // Model-Level Graph Building
    // =============================================================================

    ComputeGraph QwenGraphBase::buildFullForwardGraph(
        const ForwardInput &input,
        ForwardOutput &output)
    {
        LOG_DEBUG("[QwenGraphBase] Building full forward graph: "
                  << "batch_size=" << input.batch_size
                  << ", seq_len=" << input.seq_len);

        if (!modelEmbeddingTable() || !modelFinalNorm() || !modelLMHead())
        {
            LOG_ERROR("[QwenGraphBase] Weights not set! Call setWeights() first.");
            throw std::runtime_error("QwenStandardGraph weights not initialized");
        }

        if (!buffers_.current_hidden || !buffers_.logits)
        {
            LOG_ERROR("[QwenGraphBase] Buffers not set! Call setBuffers() first.");
            throw std::runtime_error("QwenStandardGraph buffers not initialized");
        }

        DeviceId device = config_.default_device;
        int total_tokens = input.batch_size * input.seq_len;

        ComputeGraph graph;

        // -------------------------------------------------------------------------
        // Stage 1: Embedding Lookup
        // -------------------------------------------------------------------------
        // For HybridQ16 mode: output to Q16_1 residual buffer (the residual stream)
        // For other modes: output to FP32 current_hidden
        TensorBase *embed_output = (buffers_.layer_buffers.residual && config_.isHybridQ16())
                                       ? buffers_.layer_buffers.residual
                                       : buffers_.current_hidden;

        EmbeddingStage::Params embed_params;
        embed_params.embed_table = modelEmbeddingTable();
        embed_params.token_ids = input.token_ids;
        embed_params.token_ids_device = input.token_ids_device;
        embed_params.output = embed_output;
        embed_params.num_tokens = total_tokens;
        embed_params.d_model = config_.d_model;
        embed_params.vocab_size = config_.vocab_size;
        embed_params.vocab_offset = embeddingVocabOffsetForDevice(config_, config_.default_device);
        embed_params.local_vocab_size = modelEmbeddingTable() ? static_cast<int>(modelEmbeddingTable()->rows()) : 0;
        embed_params.device_id = config_.default_device;
        embed_params.output_buffer_id = BufferId::HIDDEN_STATE;
        embed_params.mpi_ctx = mpi_ctx_.get();
        embed_params.prepared_ref = preparedRefForGraphWeight(modelEmbeddingBinding(), config_.default_device);
        embed_params.prepared_store = prepared_weight_store_;

        graph.addNode("embedding",
                      ComputeStageFactory::createEmbedding(embed_params),
                      device);

        // -------------------------------------------------------------------------
        // Stage 1b: Embedding AllReduce (vocab-parallel embedding sharding)
        // -------------------------------------------------------------------------
        // When embedding is column-parallel sharded, each device holds
        // vocab_size/tp_degree rows. Tokens outside the local range produce zeros.
        // AllReduce(sum) combines the partial results.
        const bool embedding_is_sharded =
            modelEmbeddingTable() &&
            static_cast<int>(modelEmbeddingTable()->rows()) < config_.vocab_size;
        if (embedding_is_sharded && needsTPAllreduce())
        {
            size_t allreduce_count = static_cast<size_t>(total_tokens) * config_.d_model;
            auto allreduce_stage = createTPAllreduceStage(
                embed_output, allreduce_count, device, -1,
                /*is_attention=*/false, "embedding_allreduce",
                BufferId::HIDDEN_STATE);
            if (allreduce_stage)
            {
                graph.addNode("embedding_allreduce", std::move(allreduce_stage), device);
                graph.addDependency("embedding_allreduce", "embedding");
            }
        }

        // -------------------------------------------------------------------------
        // Stage 2: Transformer Layers (complete graphs, not placeholders)
        // -------------------------------------------------------------------------
        std::string prev_node = embedding_is_sharded && needsTPAllreduce()
                                    ? "embedding_allreduce"
                                    : "embedding";

        // Position IDs must be provided externally (or use fallback for backward compat)
        const int *position_ids = input.position_ids;
        std::vector<int> local_position_ids;
        if (!position_ids)
        {
            // Fallback: build position IDs internally (deprecated path)
            local_position_ids = buildPositionIds(input.seq_len, input.batch_size, input.position_offset);
            position_ids = local_position_ids.data();
            LOG_DEBUG("[QwenGraphBase] Position IDs built internally (deprecated - prefer external input)");
        }

        // Check if we have layer weight accessor
        if (!hasLayerWeightSource())
        {
            LOG_ERROR("[QwenGraphBase] Layer weight accessor not set!");
            throw std::runtime_error("QwenStandardGraph layer weight accessor not initialized");
        }

        // Build complete graphs for each layer
        for (int layer = 0; layer < config_.n_layers; ++layer)
        {
            // Get layer weights
            LayerWeights layer_weights = layerWeightsForGraph(layer);

            // Build attention graph for this layer
            // Pass sequence_lengths for proper batch-aware attention masking
            ComputeGraph attn_graph = buildAttentionGraph(
                layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                input.batch_size, input.kv_cache, position_ids, device,
                input.sequence_lengths);

            // Get the terminal node of attention sub-graph
            std::string attn_last = attn_graph.terminalNode();
            if (attn_last.empty())
                attn_last = prev_node;

            // Merge attention graph, connecting to previous node
            graph.merge(std::move(attn_graph), prev_node);

            // Build FFN graph for this layer
            ComputeGraph ffn_graph = buildFFNGraph(
                layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                input.batch_size, device);

            // Get the terminal node of FFN sub-graph
            std::string ffn_last = ffn_graph.terminalNode();
            if (ffn_last.empty())
                ffn_last = attn_last;

            // Merge FFN graph, connecting to attention terminal
            graph.merge(std::move(ffn_graph), attn_last);

            // Use the determined FFN terminal as the prev node for next layer
            prev_node = ffn_last;
        }

        // -------------------------------------------------------------------------
        // Stage 3: Final RMSNorm
        // -------------------------------------------------------------------------
        // IMPORTANT:
        // - In HybridQ16, the live activation stream is Q16_1 in buffers_.layer_buffers.residual.
        // - Our Q16_1 RMSNorm kernel is Q16_1 input -> FP32 output.
        // Therefore final_norm must:
        //   1) read from the correct activation stream (residual for HybridQ16)
        //   2) write out-of-place into the FP32 normalized buffer
        TensorBase *final_norm_input = (config_.isHybridQ16() && buffers_.layer_buffers.residual)
                                           ? buffers_.layer_buffers.residual
                                           : buffers_.current_hidden;

        addFinalNormToGraph(graph, final_norm_input, buffers_.layer_buffers.normalized,
                            prev_node, total_tokens, device);
        prev_node = "final_norm";

        std::string lm_head_dependency = prev_node;
        TensorBase *lm_head_input = maybeAddLMHeadRowSelect(
            graph,
            prev_node,
            buffers_.layer_buffers.normalized,
            total_tokens,
            input.real_seq_len,
            input.bucket_seq_len,
            device,
            lm_head_dependency);

        // -------------------------------------------------------------------------
        // Stage 4: LM Head (with optional Column-Parallel + AllGather)
        // -------------------------------------------------------------------------
        // When lm_head_column_parallel is enabled:
        // - LM head weight is sharded by vocab: [vocab_local, d_model]
        // - LM head outputs to buffers_.logits_local: [seq_len, vocab_local]
        // - AllGather collects to buffers_.logits: [seq_len, vocab_size]
        // -------------------------------------------------------------------------
        bool use_column_parallel = config_.lm_head_column_parallel && buffers_.logits_local != nullptr;

        // Determine output buffer and vocab size for LM head stage
        TensorBase *lm_head_output = use_column_parallel ? buffers_.logits_local : buffers_.logits;
        int lm_head_vocab_size = use_column_parallel ? config_.vocab_local : config_.vocab_size;

        LOG_DEBUG("[QwenGraphBase] LM head in buildFullForwardGraph: use_column_parallel="
                  << use_column_parallel << " lm_head_vocab_size=" << lm_head_vocab_size);

        const LMHeadInputLayout lm_layout =
            describeLMHeadInputLayout(config_, buffers_.layer_buffers, lm_head_input, total_tokens);

        LMHeadStage::Params lm_params;
        // Feed LM head from final RMSNorm, bucket one-row scratch, or compact
        // verifier rows. The layout helper keeps the row count and BufferId
        // contract synchronized.
        lm_params.hidden_states = lm_head_input;
        lm_params.lm_head_weight = modelLMHead();
        lm_params.prepared_ref = preparedRefForGraphWeight(modelLMHeadBinding(), device);
        lm_params.logits = lm_head_output;
        lm_params.seq_len = lm_layout.seq_len;
        lm_params.d_model = config_.d_model;
        lm_params.vocab_size = lm_head_vocab_size;
        lm_params.bias_tensor = nullptr; // Qwen2 has no LM head bias
        lm_params.device_id = config_.default_device;
        lm_params.prepared_store = prepared_weight_store_;
        lm_params.input_buffer_id = lm_layout.input_buffer_id;
        lm_params.output_buffer_id = logitsBufferId(
            use_column_parallel,
            lm_layout.compute_all_positions);
        lm_params.use_prefill_replay_row_offset = lm_layout.use_prefill_replay_row_offset;
        lm_params.compute_all_positions = lm_layout.compute_all_positions;

        graph.addNode("lm_head",
                      ComputeStageFactory::createLMHead(lm_params),
                      device);
        graph.addDependency("lm_head", lm_head_dependency);
        prev_node = "lm_head";

        // Phase 5: AllGather stage for column-parallel LM head
        if (use_column_parallel && mpi_ctx_)
        {
            LOG_DEBUG("[QwenGraphBase] Adding lm_head_allgather in buildFullForwardGraph: world_size="
                      << mpi_ctx_->world_size() << " total_tokens=" << total_tokens);

            AllGatherStage::Params allgather_params;
            allgather_params.local_input = buffers_.logits_local;
            allgather_params.full_output = buffers_.logits;
            allgather_params.mpi_ctx = mpi_ctx_.get();
            allgather_params.actual_seq_len = lm_layout.compute_all_positions ? lm_layout.seq_len : 1;
            // LM head is not layer-specific; use nullptr for domain (legacy MPI path)
            // Multi-domain TP typically doesn't route LM head to a specific domain
            allgather_params.domain = nullptr;
            allgather_params.input_buffer_id = logitsBufferId(
                /*column_parallel=*/true,
                lm_layout.compute_all_positions);
            allgather_params.output_buffer_id = gatheredLogitsBufferId(
                lm_layout.compute_all_positions);

            graph.addNode("lm_head_allgather",
                          ComputeStageFactory::createAllGather(allgather_params),
                          device);
            graph.addDependency("lm_head_allgather", prev_node);
        }

        // Set output
        output.logits = buffers_.logits;

        LOG_DEBUG("[QwenGraphBase] Built full forward graph with "
                  << graph.size() << " nodes");

        return graph;
    }

    // =========================================================================
    // Partial Forward Graph Building (Pipeline Parallelism)
    // =========================================================================

    ComputeGraph QwenGraphBase::buildPartialForwardGraph(
        const ForwardInput &input,
        ForwardOutput &output,
        int first_layer,
        int last_layer,
        bool has_embedding,
        bool has_lm_head)
    {
        LOG_DEBUG("[QwenGraphBase] Building partial forward graph: "
                  << "batch_size=" << input.batch_size
                  << ", seq_len=" << input.seq_len
                  << ", layers=[" << first_layer << ", " << last_layer << ")"
                  << ", has_embedding=" << has_embedding
                  << ", has_lm_head=" << has_lm_head);

        // Validate layer range against total model layers (not local PP stage count)
        int max_layers = config_.total_n_layers > 0 ? config_.total_n_layers : config_.n_layers;
        if (first_layer < 0 || last_layer > max_layers || first_layer >= last_layer)
        {
            LOG_ERROR("[QwenGraphBase] Invalid layer range: [" << first_layer << ", " << last_layer
                                                               << ") for model with " << max_layers << " total layers");
            throw std::invalid_argument("Invalid layer range for partial forward graph");
        }

        // Validate required weights based on configuration
        if (has_embedding && !modelEmbeddingTable())
        {
            LOG_ERROR("[QwenGraphBase] Embedding weights required but not set");
            throw std::runtime_error("QwenStandardGraph embedding weights not initialized");
        }
        if (has_lm_head && (!modelFinalNorm() || !modelLMHead()))
        {
            LOG_ERROR("[QwenGraphBase] LM head weights required but not set");
            throw std::runtime_error("QwenStandardGraph LM head weights not initialized");
        }
        if (!hasLayerWeightSource())
        {
            LOG_ERROR("[QwenGraphBase] Layer weight accessor not set");
            throw std::runtime_error("QwenStandardGraph layer weight accessor not initialized");
        }

        // Validate buffers
        if (has_embedding && !buffers_.current_hidden && !buffers_.layer_buffers.residual)
        {
            LOG_ERROR("[QwenGraphBase] Hidden buffer required for embedding output");
            throw std::runtime_error("QwenStandardGraph hidden buffer not initialized");
        }
        if (has_lm_head && !buffers_.logits)
        {
            LOG_ERROR("[QwenGraphBase] Logits buffer required for LM head output");
            throw std::runtime_error("QwenStandardGraph logits buffer not initialized");
        }

        DeviceId device = config_.default_device;
        int total_tokens = input.batch_size * input.seq_len;

        ComputeGraph graph;
        std::string prev_node;

        // -------------------------------------------------------------------------
        // Stage 1: Embedding Lookup (optional - only for first PP stage)
        // -------------------------------------------------------------------------
        if (has_embedding)
        {
            // For HybridQ16 mode: output to Q16_1 residual buffer (the residual stream)
            // For other modes: output to FP32 current_hidden
            TensorBase *embed_output = (buffers_.layer_buffers.residual && config_.isHybridQ16())
                                           ? buffers_.layer_buffers.residual
                                           : buffers_.current_hidden;

            EmbeddingStage::Params embed_params;
            embed_params.embed_table = modelEmbeddingTable();
            embed_params.token_ids = input.token_ids;
            embed_params.token_ids_device = input.token_ids_device;
            embed_params.output = embed_output;
            embed_params.num_tokens = total_tokens;
            embed_params.d_model = config_.d_model;
            embed_params.vocab_size = config_.vocab_size;
            embed_params.vocab_offset = embeddingVocabOffsetForDevice(config_, config_.default_device);
            embed_params.local_vocab_size = modelEmbeddingTable() ? static_cast<int>(modelEmbeddingTable()->rows()) : 0;
            embed_params.device_id = config_.default_device;
            embed_params.output_buffer_id = BufferId::HIDDEN_STATE;
            embed_params.mpi_ctx = mpi_ctx_.get();
            embed_params.prepared_ref = preparedRefForGraphWeight(modelEmbeddingBinding(), config_.default_device);
            embed_params.prepared_store = prepared_weight_store_;

            graph.addNode("embedding",
                          ComputeStageFactory::createEmbedding(embed_params),
                          device);
            prev_node = "embedding";

            // Stage 1b: Embedding AllReduce (vocab-parallel embedding sharding)
            // When embedding is column-parallel sharded, each device holds
            // vocab_size/tp_degree rows. Tokens outside the local range produce zeros.
            // AllReduce(sum) combines the partial results.
            const bool embedding_is_sharded =
                modelEmbeddingTable() &&
                static_cast<int>(modelEmbeddingTable()->rows()) < config_.vocab_size;
            if (embedding_is_sharded && needsTPAllreduce())
            {
                size_t allreduce_count = static_cast<size_t>(total_tokens) * config_.d_model;
                auto allreduce_stage = createTPAllreduceStage(
                    embed_output, allreduce_count, device, -1,
                    /*is_attention=*/false, "embedding_allreduce",
                    BufferId::HIDDEN_STATE);
                if (allreduce_stage)
                {
                    graph.addNode("embedding_allreduce", std::move(allreduce_stage), device);
                    graph.addDependency("embedding_allreduce", "embedding");
                    prev_node = "embedding_allreduce";
                }
            }
        }
        else if (input.external_hidden_state)
        {
            // -----------------------------------------------------------------
            // PP middle/final stage: Use external hidden state as starting point
            // The external_hidden_state tensor contains activations from the
            // previous PP stage. We need to copy it to our working buffer if
            // they're different.
            // -----------------------------------------------------------------
            TensorBase *working_buffer = (buffers_.layer_buffers.residual && config_.isHybridQ16())
                                             ? buffers_.layer_buffers.residual
                                             : buffers_.current_hidden;

            // Check if external buffer differs from working buffer
            if (input.external_hidden_state != working_buffer)
            {
                LOG_DEBUG("[QwenGraphBase] PP middle stage: copying external hidden state to working buffer");

                // NOTE: TensorCopyStage is not yet implemented. For now, we perform
                // the copy inline. In Phase 3, we should add TensorCopyStage for
                // proper graph-based memory transfer with device awareness.

                size_t copy_bytes = static_cast<size_t>(total_tokens * config_.d_model);
                if (config_.isHybridQ16())
                {
                    // Q16_1: copy the raw Q16_1 blocks
                    // Block size = 32 elements, so num_blocks = copy_elements / 32
                    size_t num_blocks = (copy_bytes + 31) / 32;
                    copy_bytes = num_blocks * sizeof(Q16_1Block);
                }
                else
                {
                    // FP32: copy floats
                    copy_bytes *= sizeof(float);
                }

                // Unified PP copy: data() handles all device coherence sync
                // automatically (including D2H via staging buffer).
                // This eliminates the previous 3-way D2D/CPU branch.
                const void *src = input.external_hidden_state->data();
                void *dst = working_buffer->mutable_data();
                std::memcpy(dst, src, copy_bytes);

                if (config_.default_device.is_gpu())
                {
                    DeviceId target_device = config_.default_device;
                    if (!working_buffer->ensureOnDevice(target_device))
                    {
                        LOG_ERROR("[QwenGraphBase] Failed to upload working buffer to "
                                  << target_device.toString());
                        throw std::runtime_error("Failed to upload working buffer");
                    }
                    working_buffer->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, target_device);
                }

                LOG_DEBUG("[QwenGraphBase] PP copy: " << copy_bytes << " bytes to "
                                                      << config_.default_device.toString());
            }
            else
            {
                LOG_DEBUG("[QwenGraphBase] PP middle stage: external hidden state IS working buffer, no copy needed");
            }
            // No prev_node set - first layer will have no dependencies
        }
        else if (!has_embedding)
        {
            LOG_ERROR("[QwenGraphBase] PP stage without embedding requires external_hidden_state input");
            throw std::runtime_error("PP stage without embedding requires external_hidden_state");
        }

        // -------------------------------------------------------------------------
        // Stage 2: Transformer Layers (subset for this PP stage)
        // -------------------------------------------------------------------------
        // Position IDs must be provided externally (or use fallback for backward compat)
        const int *position_ids = input.position_ids;
        std::vector<int> local_position_ids;
        if (!position_ids)
        {
            // Fallback: build position IDs internally (deprecated path)
            local_position_ids = buildPositionIds(input.seq_len, input.batch_size, input.position_offset);
            position_ids = local_position_ids.data();
            LOG_DEBUG("[QwenGraphBase] Position IDs built internally (deprecated - prefer external input)");
        }

        // Build graphs for assigned layer range [first_layer, last_layer)
        for (int layer = first_layer; layer < last_layer; ++layer)
        {
            // Get layer weights
            LayerWeights layer_weights = layerWeightsForGraph(layer);

            // Build attention graph for this layer
            ComputeGraph attn_graph = buildAttentionGraph(
                layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                input.batch_size, input.kv_cache, position_ids, device,
                input.sequence_lengths);

            // Get the terminal node of attention sub-graph
            std::string attn_last = attn_graph.terminalNode();
            if (attn_last.empty())
                attn_last = prev_node;

            // Merge attention graph, connecting to previous node
            if (!prev_node.empty())
            {
                graph.merge(std::move(attn_graph), prev_node);
            }
            else
            {
                // First layer without embedding: merge directly
                // The attention graph root nodes have no dependencies initially
                for (const auto &root : attn_graph.getRootNodes())
                {
                    // Move all nodes from attn_graph to main graph
                }
                graph.merge(std::move(attn_graph), "");
            }

            // Build FFN graph for this layer
            ComputeGraph ffn_graph = buildFFNGraph(
                layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                input.batch_size, device);

            // Get the terminal node of FFN sub-graph
            std::string ffn_last = ffn_graph.terminalNode();
            if (ffn_last.empty())
                ffn_last = attn_last;

            // Merge FFN graph, connecting to attention terminal
            graph.merge(std::move(ffn_graph), attn_last);

            // Use the determined FFN terminal as the prev node for next layer
            prev_node = ffn_last;
        }

        // -------------------------------------------------------------------------
        // Stage 3: Final RMSNorm + LM Head (optional - only for final PP stage)
        // -------------------------------------------------------------------------
        if (has_lm_head)
        {
            // Final RMSNorm
            TensorBase *final_norm_input = (config_.isHybridQ16() && buffers_.layer_buffers.residual)
                                               ? buffers_.layer_buffers.residual
                                               : buffers_.current_hidden;

            addFinalNormToGraph(graph, final_norm_input, buffers_.layer_buffers.normalized,
                                prev_node, total_tokens, device);
            prev_node = "final_norm";

            std::string lm_head_dependency = prev_node;
            TensorBase *lm_head_input = maybeAddLMHeadRowSelect(
                graph,
                prev_node,
                buffers_.layer_buffers.normalized,
                total_tokens,
                input.real_seq_len,
                input.bucket_seq_len,
                device,
                lm_head_dependency);

            // LM Head (with optional Column-Parallel + AllGather)
            bool use_column_parallel = config_.lm_head_column_parallel && buffers_.logits_local != nullptr;

            TensorBase *lm_head_output = use_column_parallel ? buffers_.logits_local : buffers_.logits;
            int lm_head_vocab_size = use_column_parallel ? config_.vocab_local : config_.vocab_size;

            LOG_DEBUG("[QwenGraphBase] LM head in buildPartialForwardGraph: use_column_parallel="
                      << use_column_parallel << " lm_head_vocab_size=" << lm_head_vocab_size);

            const LMHeadInputLayout lm_layout =
                describeLMHeadInputLayout(config_, buffers_.layer_buffers, lm_head_input, total_tokens);

            LMHeadStage::Params lm_params;
            lm_params.hidden_states = lm_head_input;
            lm_params.lm_head_weight = modelLMHead();
            lm_params.prepared_ref = preparedRefForGraphWeight(modelLMHeadBinding(), device);
            lm_params.logits = lm_head_output;
            lm_params.seq_len = lm_layout.seq_len;
            lm_params.d_model = config_.d_model;
            lm_params.vocab_size = lm_head_vocab_size;
            lm_params.bias_tensor = nullptr;
            lm_params.device_id = config_.default_device;
            lm_params.prepared_store = prepared_weight_store_;
            lm_params.input_buffer_id = lm_layout.input_buffer_id;
            lm_params.output_buffer_id = logitsBufferId(
                use_column_parallel,
                lm_layout.compute_all_positions);
            lm_params.use_prefill_replay_row_offset = lm_layout.use_prefill_replay_row_offset;
            lm_params.compute_all_positions = lm_layout.compute_all_positions;

            graph.addNode("lm_head",
                          ComputeStageFactory::createLMHead(lm_params),
                          device);
            graph.addDependency("lm_head", lm_head_dependency);
            prev_node = "lm_head";

            // AllGather stage for column-parallel LM head
            if (use_column_parallel && mpi_ctx_)
            {
                LOG_DEBUG("[QwenGraphBase] Adding lm_head_allgather in buildPartialForwardGraph: world_size="
                          << mpi_ctx_->world_size() << " total_tokens=" << total_tokens);

                AllGatherStage::Params allgather_params;
                allgather_params.local_input = buffers_.logits_local;
                allgather_params.full_output = buffers_.logits;
                allgather_params.mpi_ctx = mpi_ctx_.get();
                allgather_params.actual_seq_len = lm_layout.compute_all_positions ? lm_layout.seq_len : 1;
                allgather_params.domain = nullptr;
                allgather_params.input_buffer_id = logitsBufferId(
                    /*column_parallel=*/true,
                    lm_layout.compute_all_positions);
                allgather_params.output_buffer_id = gatheredLogitsBufferId(
                    lm_layout.compute_all_positions);

                graph.addNode("lm_head_allgather",
                              ComputeStageFactory::createAllGather(allgather_params),
                              device);
                graph.addDependency("lm_head_allgather", prev_node);
            }

            // Set output logits
            output.logits = buffers_.logits;
        }
        else
        {
            // Non-final PP stage: output hidden states directly
            // The residual buffer contains the hidden states after the last layer
            output.hidden = (config_.isHybridQ16() && buffers_.layer_buffers.residual)
                                ? buffers_.layer_buffers.residual
                                : buffers_.current_hidden;
            output.logits = nullptr;
        }

        LOG_DEBUG("[QwenGraphBase] Built partial forward graph with "
                  << graph.size() << " nodes for layers [" << first_layer << ", " << last_layer << ")");

        return graph;
    }

    // =========================================================================
    // Unified Pipeline Graph Building (PP + TP Composition)
    // =========================================================================

    ComputeGraph QwenGraphBase::buildUnifiedPipelineGraph(
        const ForwardInput &input,
        ForwardOutput &output)
    {
        // =====================================================================
        // 1. Validate prerequisites
        // =====================================================================
        if (!config_.pipeline_config)
        {
            LOG_ERROR("[QwenGraphBase] buildUnifiedPipelineGraph called without pipeline_config");
            throw std::runtime_error("pipeline_config required for unified PP graph");
        }

        std::string validation_error;
        if (!config_.pipeline_config->validate(&validation_error))
        {
            LOG_ERROR("[QwenGraphBase] Invalid pipeline_config: " << validation_error);
            throw std::runtime_error("Invalid pipeline_config: " + validation_error);
        }

        LOG_DEBUG("[QwenGraphBase] Building unified pipeline graph: "
                  << config_.pipeline_config->numStages() << " PP stages, "
                  << "batch_size=" << input.batch_size
                  << ", seq_len=" << input.seq_len);

        // Validate weights
        if (!hasLayerWeightSource())
        {
            LOG_ERROR("[QwenGraphBase] Layer weight accessor not set");
            throw std::runtime_error("QwenStandardGraph layer weight accessor not initialized");
        }

        // =====================================================================
        // 2. Prepare execution context
        // =====================================================================
        ComputeGraph graph;
        std::string prev_node;
        int total_tokens = input.batch_size * input.seq_len;

        // Position IDs
        const int *position_ids = input.position_ids;
        std::vector<int> local_position_ids;
        if (!position_ids)
        {
            local_position_ids = buildPositionIds(input.seq_len, input.batch_size, input.position_offset);
            position_ids = local_position_ids.data();
            LOG_DEBUG("[QwenGraphBase] Position IDs built internally for unified PP graph");
        }

        // =====================================================================
        // 3. Iterate over PP stages
        // =====================================================================
        for (const auto &pp_stage : config_.pipeline_config->pp_stages)
        {
            LOG_DEBUG("[QwenGraphBase] Building PP stage " << pp_stage.stage_id
                                                           << " (" << pp_stage.domain_name << "): layers ["
                                                           << pp_stage.first_layer << ", " << pp_stage.last_layer << ")"
                                                           << " has_embedding=" << pp_stage.has_embedding
                                                           << " has_lm_head=" << pp_stage.has_lm_head);

            // -----------------------------------------------------------------
            // 3a. Get domain config for this stage
            // -----------------------------------------------------------------
            const TPDomainConfig *domain = config_.pipeline_config->getDomainForStage(pp_stage.stage_id);
            if (!domain)
            {
                LOG_ERROR("[QwenGraphBase] Domain not found for stage " << pp_stage.stage_id
                                                                        << " (domain_name=" << pp_stage.domain_name << ")");
                throw std::runtime_error("Domain not found for PP stage " +
                                         std::to_string(pp_stage.stage_id));
            }

            // Get device and TP context for this stage
            DeviceId stage_device = domain->primaryDevice();
            ITPContext *stage_tp_ctx = nullptr;
            auto tp_it = config_.domain_tp_contexts.find(pp_stage.domain_name);
            if (tp_it != config_.domain_tp_contexts.end())
            {
                stage_tp_ctx = tp_it->second;
            }

            LOG_DEBUG("[QwenGraphBase] Stage " << pp_stage.stage_id
                                               << " device=" << stage_device.to_string()
                                               << " has_tp_ctx=" << (stage_tp_ctx != nullptr));

            // -----------------------------------------------------------------
            // 3b. Build embedding if this is the first stage
            // -----------------------------------------------------------------
            if (pp_stage.has_embedding)
            {
                if (!modelEmbeddingTable())
                {
                    LOG_ERROR("[QwenGraphBase] Embedding weights required but not set");
                    throw std::runtime_error("Embedding weights not initialized for unified PP graph");
                }

                // For HybridQ16 mode: output to Q16_1 residual buffer
                // For other modes: output to FP32 current_hidden
                TensorBase *embed_output = buffers_.layer_buffers.residual &&
                                                   config_.isHybridQ16()
                                               ? buffers_.layer_buffers.residual
                                               : buffers_.current_hidden;

                EmbeddingStage::Params embed_params;
                embed_params.embed_table = modelEmbeddingTable();
                embed_params.token_ids = input.token_ids;
                embed_params.token_ids_device = input.token_ids_device;
                embed_params.output = embed_output;
                embed_params.num_tokens = total_tokens;
                embed_params.d_model = config_.d_model;
                embed_params.vocab_size = config_.vocab_size;
                embed_params.vocab_offset = embeddingVocabOffsetForDevice(config_, stage_device);
                embed_params.local_vocab_size = modelEmbeddingTable() ? static_cast<int>(modelEmbeddingTable()->rows()) : 0;
                embed_params.device_id = stage_device;
                embed_params.mpi_ctx = mpi_ctx_.get();
                embed_params.prepared_ref = preparedRefForGraphWeight(modelEmbeddingBinding(), stage_device);
                embed_params.prepared_store = prepared_weight_store_;

                graph.addNode("embedding",
                              ComputeStageFactory::createEmbedding(embed_params),
                              stage_device);
                prev_node = "embedding";

                LOG_DEBUG("[QwenGraphBase] Added embedding stage on device " << stage_device.to_string());
            }

            // -----------------------------------------------------------------
            // 3c. Build layers for this PP stage
            // -----------------------------------------------------------------
            // RAII guard for config overrides — ensures exception-safe restoration
            struct ConfigGuard
            {
                GraphConfig &cfg;
                DeviceId original_device;
                ITPContext *original_tp_ctx;
                ConfigGuard(GraphConfig &c, DeviceId dev, ITPContext *tp)
                    : cfg(c), original_device(c.default_device), original_tp_ctx(c.tp_ctx)
                {
                    cfg.default_device = dev;
                    if (tp)
                        cfg.tp_ctx = tp;
                }
                ~ConfigGuard()
                {
                    cfg.default_device = original_device;
                    cfg.tp_ctx = original_tp_ctx;
                }
            } config_guard(config_, stage_device, stage_tp_ctx);

            for (int layer = pp_stage.first_layer; layer < pp_stage.last_layer; ++layer)
            {
                // Get layer weights
                LayerWeights layer_weights = layerWeightsForGraph(layer);

                // Get KV cache for this stage's device (PP-aware)
                IKVCache *layer_kv_cache = input.getKVCacheForDevice(stage_device);

                // Build attention graph with stage_device
                ComputeGraph attn_graph = buildAttentionGraph(
                    layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                    input.batch_size, layer_kv_cache, position_ids, stage_device,
                    input.sequence_lengths);

                // Get the terminal node of attention sub-graph
                std::string attn_last = attn_graph.terminalNode();
                if (attn_last.empty())
                    attn_last = prev_node;

                // Merge attention graph
                if (!prev_node.empty())
                {
                    graph.merge(std::move(attn_graph), prev_node);
                }
                else
                {
                    graph.merge(std::move(attn_graph), "");
                }

                // Build FFN graph
                ComputeGraph ffn_graph = buildFFNGraph(
                    layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                    input.batch_size, stage_device);

                // Get the terminal node of FFN sub-graph
                std::string ffn_last = ffn_graph.terminalNode();
                if (ffn_last.empty())
                    ffn_last = attn_last;

                // Merge FFN graph
                graph.merge(std::move(ffn_graph), attn_last);

                prev_node = ffn_last;
            }

            // ConfigGuard restores config_ automatically at scope exit

            // -----------------------------------------------------------------
            // 3d. Insert PP transfer stage if not the last PP stage
            // -----------------------------------------------------------------
            int next_stage_id = pp_stage.stage_id + 1;
            if (next_stage_id < config_.pipeline_config->numStages())
            {
                // Get PP context for this transfer
                auto pp_key = std::make_pair(pp_stage.stage_id, next_stage_id);
                ILocalPPContext *pp_ctx = nullptr;
                auto pp_it = config_.pp_contexts.find(pp_key);
                if (pp_it != config_.pp_contexts.end())
                {
                    pp_ctx = pp_it->second;
                }

                if (!pp_ctx)
                {
                    LOG_ERROR("[QwenGraphBase] PP context not found for transfer "
                              << pp_stage.stage_id << " -> " << next_stage_id);
                    throw std::runtime_error("PP context not found for stage transfer " +
                                             std::to_string(pp_stage.stage_id) + " -> " +
                                             std::to_string(next_stage_id));
                }

                // Get the hidden state buffer for transfer
                TensorBase *hidden_state = (config_.isHybridQ16() && buffers_.layer_buffers.residual)
                                               ? buffers_.layer_buffers.residual
                                               : buffers_.current_hidden;

                // Create LocalPPTransferStage
                LocalPPTransferStage::Params transfer_params;
                transfer_params.pp_ctx = pp_ctx;
                transfer_params.tensor = hidden_state;
                transfer_params.stage_from = pp_stage.stage_id;
                transfer_params.stage_to = next_stage_id;
                transfer_params.stage_name = "pp_transfer_" +
                                             std::to_string(pp_stage.stage_id) + "_to_" +
                                             std::to_string(next_stage_id);
                transfer_params.device_id = stage_device;
                transfer_params.mpi_ctx = mpi_ctx_.get();

                std::string transfer_name = transfer_params.stage_name;

                graph.addNode(transfer_name,
                              std::make_unique<LocalPPTransferStage>(transfer_params),
                              stage_device); // Transfer stage starts on source device
                graph.addDependency(transfer_name, prev_node);

                prev_node = transfer_name;

                LOG_DEBUG("[QwenGraphBase] Added PP transfer: " << transfer_name
                                                                << " (source device=" << stage_device.to_string() << ")");
            }

            // -----------------------------------------------------------------
            // 3e. Build LM head if this is the last stage
            // -----------------------------------------------------------------
            if (pp_stage.has_lm_head)
            {
                if (!modelFinalNorm() || !modelLMHead())
                {
                    LOG_ERROR("[QwenGraphBase] LM head weights required but not set");
                    throw std::runtime_error("LM head weights not initialized for unified PP graph");
                }

                // Final RMSNorm
                TensorBase *final_norm_input = (config_.isHybridQ16() && buffers_.layer_buffers.residual)
                                                   ? buffers_.layer_buffers.residual
                                                   : buffers_.current_hidden;

                addFinalNormToGraph(graph, final_norm_input, buffers_.layer_buffers.normalized,
                                    prev_node, total_tokens, stage_device);
                prev_node = "final_norm";

                std::string lm_head_dependency = prev_node;
                TensorBase *lm_head_input = maybeAddLMHeadRowSelect(
                    graph,
                    prev_node,
                    buffers_.layer_buffers.normalized,
                    total_tokens,
                    input.real_seq_len,
                    input.bucket_seq_len,
                    stage_device,
                    lm_head_dependency);

                // LM Head
                bool use_column_parallel = config_.lm_head_column_parallel && buffers_.logits_local != nullptr;
                TensorBase *lm_head_output = use_column_parallel ? buffers_.logits_local : buffers_.logits;
                int lm_head_vocab_size = use_column_parallel ? config_.vocab_local : config_.vocab_size;

                LOG_DEBUG("[QwenGraphBase] LM head in unified PP: use_column_parallel="
                          << use_column_parallel << " vocab_size=" << lm_head_vocab_size);

                const LMHeadInputLayout lm_layout =
                    describeLMHeadInputLayout(config_, buffers_.layer_buffers, lm_head_input, total_tokens);

                LMHeadStage::Params lm_params;
                lm_params.hidden_states = lm_head_input;
                lm_params.lm_head_weight = modelLMHead();
                lm_params.prepared_ref = preparedRefForGraphWeight(modelLMHeadBinding(), stage_device);
                lm_params.logits = lm_head_output;
                lm_params.seq_len = lm_layout.seq_len;
                lm_params.d_model = config_.d_model;
                lm_params.vocab_size = lm_head_vocab_size;
                lm_params.bias_tensor = nullptr;
                lm_params.device_id = stage_device;
                lm_params.prepared_store = prepared_weight_store_;
                lm_params.input_buffer_id = lm_layout.input_buffer_id;
                lm_params.output_buffer_id = logitsBufferId(
                    use_column_parallel,
                    lm_layout.compute_all_positions);
                lm_params.use_prefill_replay_row_offset = lm_layout.use_prefill_replay_row_offset;
                lm_params.compute_all_positions = lm_layout.compute_all_positions;

                graph.addNode("lm_head",
                              ComputeStageFactory::createLMHead(lm_params),
                              stage_device);
                graph.addDependency("lm_head", lm_head_dependency);
                prev_node = "lm_head";

                // AllGather stage for column-parallel LM head
                if (use_column_parallel && mpi_ctx_)
                {
                    LOG_DEBUG("[QwenGraphBase] Adding lm_head_allgather in unified PP: world_size="
                              << mpi_ctx_->world_size());

                    AllGatherStage::Params allgather_params;
                    allgather_params.local_input = buffers_.logits_local;
                    allgather_params.full_output = buffers_.logits;
                    allgather_params.mpi_ctx = mpi_ctx_.get();
                    allgather_params.actual_seq_len = lm_layout.compute_all_positions ? lm_layout.seq_len : 1;
                    allgather_params.domain = nullptr;
                    allgather_params.input_buffer_id = logitsBufferId(
                        /*column_parallel=*/true,
                        lm_layout.compute_all_positions);
                    allgather_params.output_buffer_id = gatheredLogitsBufferId(
                        lm_layout.compute_all_positions);

                    graph.addNode("lm_head_allgather",
                                  ComputeStageFactory::createAllGather(allgather_params),
                                  stage_device);
                    graph.addDependency("lm_head_allgather", prev_node);
                }

                output.logits = buffers_.logits;

                LOG_DEBUG("[QwenGraphBase] Added LM head stage on device " << stage_device.to_string());
            }
        }

        LOG_DEBUG("[QwenGraphBase] Built unified pipeline graph with " << graph.size()
                                                                       << " nodes across " << config_.pipeline_config->numStages() << " PP stages");

        return graph;
    }

    // =========================================================================
    // Schema-Based Graph Building (Phase 4d)
    // =========================================================================

    ComputeGraph QwenGraphBase::buildForwardGraphFromSchema(
        const ForwardInput &input,
        ForwardOutput &output)
    {
        LOG_DEBUG("[QwenGraphBase] Building forward graph from schema: "
                  << "batch_size=" << input.batch_size
                  << ", seq_len=" << input.seq_len);

        // =====================================================================
        // Step 1: Create the declarative schema (via virtual dispatch)
        // =====================================================================
        GraphSchema schema = getSchema();

        // =====================================================================
        // Step 2: Build runtime configuration
        // =====================================================================
        GraphResolverConfig config = getResolverConfig(input.seq_len);
        config.batch_size = input.batch_size;

        // KV cache state
        config.has_kv_cache = (input.kv_cache != nullptr);
        if (config.has_kv_cache)
        {
            config.cached_tokens = input.kv_cache->get_cached_tokens(0, 0);
        }

        // Execution policy from debugEnv
        config.exec_policy = ExecutionPolicyFlags::fromDebugEnv();

        // =====================================================================
        // Step 3: Build TensorContext for name → tensor resolution
        // =====================================================================
        TensorContext tensors = buildTensorContext();

        // =====================================================================
        // Step 4: Resolve schema to concrete graph spec
        // =====================================================================
        GraphResolver resolver;
        ResolvedGraphSpec resolved = resolver.resolve(schema, config, tensors);

        LOG_DEBUG("[QwenGraphBase] Schema resolved: " << resolved.stages.size() << " stages"
                                                      << " (emitted=" << resolved.stats.stages_emitted
                                                      << ", skipped=" << resolved.stats.stages_skipped
                                                      << ", allreduce=" << resolved.stats.allreduce_inserted
                                                      << ", allgather=" << resolved.stats.allgather_inserted << ")");

        // =====================================================================
        // Step 5: Build the ComputeGraph
        // =====================================================================
        ComputeGraph graph = GraphBuilder::build(resolved);

        // Set output
        output.logits = buffers_.logits;

        LOG_DEBUG("[QwenGraphBase] Built schema-based forward graph with "
                  << graph.size() << " nodes");

        return graph;
    }

    ComputeGraph QwenGraphBase::buildEmbeddingGraph(
        const ForwardInput &input,
        TensorBase *output_hidden)
    {
        LOG_DEBUG("[QwenGraphBase] Building embedding graph for "
                  << (input.batch_size * input.seq_len) << " tokens");

        ComputeGraph graph;

        EmbeddingStage::Params params;
        params.embed_table = modelEmbeddingTable();
        params.token_ids = input.token_ids;
        params.token_ids_device = input.token_ids_device;
        params.output = output_hidden;
        params.num_tokens = input.batch_size * input.seq_len;
        params.d_model = config_.d_model;
        params.vocab_size = config_.vocab_size;
        params.vocab_offset = embeddingVocabOffsetForDevice(config_, config_.default_device);
        params.local_vocab_size = modelEmbeddingTable() ? static_cast<int>(modelEmbeddingTable()->rows()) : 0;
        params.device_id = config_.default_device;
        params.prepared_ref = preparedRefForGraphWeight(modelEmbeddingBinding(), config_.default_device);
        params.prepared_store = prepared_weight_store_;

        graph.addNode("embedding",
                      ComputeStageFactory::createEmbedding(params),
                      config_.default_device);

        return graph;
    }

    ComputeGraph QwenGraphBase::buildTransformerLayersGraph(
        TensorBase *input_hidden,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device)
    {
        LOG_DEBUG("[QwenGraphBase] Building transformer layers graph: "
                  << config_.n_layers << " layers");

        ComputeGraph graph;
        std::string prev_node;

        // Build placeholder nodes for layer sequencing
        for (int layer = 0; layer < config_.n_layers; ++layer)
        {
            std::string layer_name = "layer_" + std::to_string(layer);

            // Add node with nullptr stage - placeholder for sequencing
            graph.addNode(layer_name, nullptr, device);

            if (!prev_node.empty())
            {
                graph.addDependency(layer_name, prev_node);
            }

            prev_node = layer_name;
        }

        return graph;
    }

    ComputeGraph QwenGraphBase::buildLayerGraph(
        int layer_idx,
        TensorBase *input_hidden,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device)
    {
        LOG_DEBUG("[QwenGraphBase] Building layer " << layer_idx << " graph");

        ComputeGraph graph;
        graph.addNode("layer_" + std::to_string(layer_idx), nullptr, device);

        return graph;
    }

    ComputeGraph QwenGraphBase::buildLMHeadGraph(
        TensorBase *hidden_states,
        TensorBase *output_logits,
        int total_tokens,
        DeviceId device,
        TensorBase *logits_local)
    {
        LOG_DEBUG("[QwenGraphBase] Building LM head graph for " << total_tokens << " tokens"
                                                                << " lm_head_column_parallel=" << config_.lm_head_column_parallel
                                                                << " vocab_local=" << config_.vocab_local);

        ComputeGraph graph;

        // Final RMSNorm
        RMSNormStage::Params norm_params;
        norm_params.input = hidden_states;
        norm_params.output = hidden_states; // In-place norm
        norm_params.gamma = modelFinalNorm();
        norm_params.eps = config_.rms_norm_eps;
        norm_params.seq_len = total_tokens;
        norm_params.device_id = device;

        graph.addNode("final_norm",
                      ComputeStageFactory::createRMSNorm(norm_params),
                      device);

        // =================================================================
        // LM Head Projection - Column-Parallel or Full
        // =================================================================
        // When lm_head_column_parallel is enabled:
        // - LM head weight is sharded by vocab: [vocab_local, d_model]
        // - LM head outputs to logits_local: [seq_len, vocab_local]
        // - AllGather collects to output_logits: [seq_len, vocab_size]
        // =================================================================

        bool use_column_parallel = config_.lm_head_column_parallel && logits_local != nullptr;

        // Determine output buffer and vocab size for LM head stage
        TensorBase *lm_head_output = use_column_parallel ? logits_local : output_logits;
        int lm_head_vocab_size = use_column_parallel ? config_.vocab_local : config_.vocab_size;

        LOG_DEBUG("[QwenGraphBase] LM head: use_column_parallel=" << use_column_parallel
                                                                  << " lm_head_vocab_size=" << lm_head_vocab_size
                                                                  << " lm_head_output=" << lm_head_output);

        TensorBase *lm_head_input = hidden_states;
        std::string lm_head_dependency = "final_norm";
        int lm_head_seq_len = total_tokens;
        BufferId lm_head_input_buffer_id = BufferId::HIDDEN_STATE;
        bool lm_head_compute_all_positions = config_.compute_all_position_logits;
        bool lm_head_use_prefill_row_offset = true;

        if (config_.compute_all_position_logits && config_.compute_row_indexed_logits)
        {
            const int row_count = config_.row_indexed_logits_row_count;
            if (row_count <= 0 || row_count > total_tokens || row_count > 4)
            {
                LOG_ERROR("[QwenGraphBase] Standalone LM-head graph row-indexed verifier requires 1..min(4,total_tokens) rows, got "
                          << row_count << " for total_tokens=" << total_tokens);
                throw std::runtime_error("invalid standalone row-indexed all-position logits row count");
            }

            TensorBase *scratch_rows = buffers_.layer_buffers.get(BufferId::LM_HEAD_INPUT_ROWS);
            if (!scratch_rows)
            {
                LOG_ERROR("[QwenGraphBase] Standalone row-indexed verifier requires lm_head_input_rows scratch buffer");
                throw std::runtime_error("standalone row-indexed verifier scratch buffer missing");
            }

            std::vector<int> selected_rows;
            selected_rows.reserve(static_cast<size_t>(row_count));
            for (int i = 0; i < row_count; ++i)
                selected_rows.push_back(i);

            HiddenStateRowsSelectStage::Params row_params;
            row_params.input = hidden_states;
            row_params.output = scratch_rows;
            row_params.seq_len = total_tokens;
            row_params.d_model = config_.d_model;
            row_params.selected_row_count = row_count;
            row_params.selected_row_indices = std::move(selected_rows);
            row_params.device_id = device;
            row_params.input_buffer_id = BufferId::HIDDEN_STATE;
            row_params.output_buffer_id = BufferId::LM_HEAD_INPUT_ROWS;
            if (device.is_gpu())
            {
                row_params.workspace_buffer_name =
                    MTPSpecDecodeWorkspaceBuffers::VERIFIER_LOGIT_ROWS;
                row_params.declare_selected_rows_workspace = false;
                row_params.upload_selected_rows_to_workspace = false;
            }

            graph.addNode("lm_head_rows_select",
                          ComputeStageFactory::createHiddenStateRowsSelect(row_params),
                          device);
            graph.addDependency("lm_head_rows_select", lm_head_dependency);

            // Compact verifier rows are a new dense matrix. The LM head should
            // project every compact row starting at row zero.
            lm_head_input = scratch_rows;
            lm_head_dependency = "lm_head_rows_select";
            lm_head_seq_len = row_count;
            lm_head_input_buffer_id = BufferId::LM_HEAD_INPUT_ROWS;
            lm_head_compute_all_positions = true;
            lm_head_use_prefill_row_offset = false;
        }

        // LM Head projection
        LMHeadStage::Params lm_params;
        lm_params.hidden_states = lm_head_input;
        lm_params.lm_head_weight = modelLMHead();
        lm_params.prepared_ref = preparedRefForGraphWeight(modelLMHeadBinding(), device);
        lm_params.logits = lm_head_output;
        lm_params.seq_len = lm_head_seq_len;
        lm_params.d_model = config_.d_model;
        lm_params.vocab_size = lm_head_vocab_size;
        lm_params.bias_tensor = nullptr;
        lm_params.device_id = device;
        lm_params.prepared_store = prepared_weight_store_;
        lm_params.input_buffer_id = lm_head_input_buffer_id;
        lm_params.output_buffer_id = logitsBufferId(
            use_column_parallel,
            lm_head_compute_all_positions);
        lm_params.compute_all_positions = lm_head_compute_all_positions;
        lm_params.use_prefill_replay_row_offset = lm_head_use_prefill_row_offset;

        graph.addNode("lm_head",
                      ComputeStageFactory::createLMHead(lm_params),
                      device);
        graph.addDependency("lm_head", lm_head_dependency);

        // =================================================================
        // AllGather stage for column-parallel LM head
        // =================================================================
        if (use_column_parallel && mpi_ctx_)
        {
            LOG_DEBUG("[QwenGraphBase] Adding lm_head_allgather: world_size=" << mpi_ctx_->world_size()
                                                                              << " total_tokens=" << total_tokens);

            AllGatherStage::Params allgather_params;
            allgather_params.local_input = logits_local;
            allgather_params.full_output = output_logits;
            allgather_params.mpi_ctx = mpi_ctx_.get();
            allgather_params.actual_seq_len = lm_head_compute_all_positions ? lm_head_seq_len : 1;
            // LM head is not layer-specific; use nullptr for domain (legacy MPI path)
            allgather_params.domain = nullptr;
            allgather_params.input_buffer_id = logitsBufferId(
                /*column_parallel=*/true,
                lm_head_compute_all_positions);
            allgather_params.output_buffer_id = gatheredLogitsBufferId(
                lm_head_compute_all_positions);

            graph.addNode("lm_head_allgather",
                          ComputeStageFactory::createAllGather(allgather_params),
                          device);
            graph.addDependency("lm_head_allgather", "lm_head");
        }

        return graph;
    }

    // =============================================================================
    // buildAttentionGraph is pure virtual - implemented by QwenStandardGraph, Qwen35Graph
    // =============================================================================

    ComputeGraph QwenGraphBase::buildFFNGraph(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int batch_size,
        DeviceId device)
    {
        ComputeGraph graph;
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";
        std::string ffn_terminal; // Track the last node for terminalNode()

        // Compute total tokens for GEMM m parameter
        int total_tokens = batch_size * seq_len;
        LayerWeightBindings layer_bindings = layerWeightBindingsForGraph(layer_idx);
        const WeightBinding *gate_proj_binding =
            layer.gate_proj_binding ? layer.gate_proj_binding : layer_bindings.gate_proj;
        const WeightBinding *up_proj_binding =
            layer.up_proj_binding ? layer.up_proj_binding : layer_bindings.up_proj;
        const WeightBinding *down_proj_binding =
            layer.down_proj_binding ? layer.down_proj_binding : layer_bindings.down_proj;

        // Stage 1: Pre-FFN RMSNorm (fused with attention residual add)
        // Combines the attention output residual add with FFN normalization.
        // On GPU: single fused kernel. On CPU: sequential ResidualAdd + RMSNorm via KernelFactory.
        {
            FusedResidualNormStage::Params fused_params;
            fused_params.device_id = device;
            fused_params.input = buffers.attn_proj;         // Wo output (to be added)
            fused_params.residual = buffers.current_hidden; // Hidden state (in-place update)
            fused_params.gamma = layer.ffn_norm;            // FFN norm gamma
            fused_params.norm_output = buffers.normalized;
            fused_params.eps = config_.rms_norm_eps;
            fused_params.seq_len = total_tokens;
            fused_params.hidden_dim = config_.d_model;
            fused_params.input_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
            fused_params.residual_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);
            fused_params.norm_output_buffer_id = buffers.idFor(BufferId::NORMALIZED);

            graph.addNode(prefix + "ffn_norm",
                          ComputeStageFactory::createFusedResidualNorm(fused_params),
                          device);
            ffn_terminal = prefix + "ffn_norm";
        }

        // Stage 2: Gate and Up projections using FusedGateUpGEMMStage
        const bool has_gate_up = (layer.gate_proj && layer.up_proj);
        if (has_gate_up)
        {
            LOG_DEBUG("[QwenGraphBase] FFN using FusedGateUpGEMMStage");

            int k = config_.d_model;
            int gate_n = static_cast<int>(layer.gate_proj->shape()[0]);
            int up_n = static_cast<int>(layer.up_proj->shape()[0]);

            FusedGateUpGEMMStage::Params gate_up_params;
            gate_up_params.input = buffers.normalized;
            gate_up_params.m = total_tokens; // Use total_tokens = batch_size * seq_len
            gate_up_params.k = k;
            gate_up_params.w_gate = layer.gate_proj;
            gate_up_params.prepared_ref_gate = preparedRefForGraphWeight(gate_proj_binding, device);
            gate_up_params.output_gate = buffers.gate;
            gate_up_params.n_gate = gate_n;
            gate_up_params.w_up = layer.up_proj;
            gate_up_params.prepared_ref_up = preparedRefForGraphWeight(up_proj_binding, device);
            gate_up_params.output_up = buffers.up;
            gate_up_params.n_up = up_n;
            gate_up_params.mpi_ctx = mpi_ctx_.get();
            gate_up_params.device_id = device;
            gate_up_params.prepared_store = prepared_weight_store_;
            gate_up_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
            gate_up_params.output_gate_buffer_id = buffers.idFor(BufferId::GATE_PROJ);
            gate_up_params.output_up_buffer_id = buffers.idFor(BufferId::UP_PROJ);

            graph.addNode(prefix + "gate_up_proj",
                          ComputeStageFactory::createFusedGateUpGEMM(gate_up_params),
                          device);

            graph.addDependency(prefix + "gate_up_proj", prefix + "ffn_norm");
        }

        // Stage 3: SwiGLU activation
        // Always fused into down_proj GEMM: silu(gate)*up + GEMM in single dispatch.
        // On GPU: kernel-level fusion (quantize + SwiGLU + GEMM).
        // On CPU: multiply_tensor_with_fused_swiglu() applies SwiGLU then GEMM.
        const bool swiglu_fusion = true;

        // Stage 4: Down projection
        const bool has_down_proj = (layer.down_proj != nullptr);
        if (has_down_proj)
        {
            int down_n = static_cast<int>(layer.down_proj->shape()[0]);
            int down_k = static_cast<int>(layer.down_proj->shape()[1]);

            GEMMStage::Params down_params{
                .device_id = device,
                .A = buffers.up,
                .B = layer.down_proj,
                .C = buffers.attn_proj,
                .m = total_tokens,
                .n = down_n,
                .k = down_k,
                .alpha = 1.0f,
                .beta = 0.0f,
                .transpose_B = false,
                .gemm_context = GemmContext::FFN,
                .a_buffer_id = buffers.idFor(BufferId::UP_PROJ),
                .gate_buffer_id = buffers.idFor(BufferId::GATE_PROJ),
                .c_buffer_id = buffers.idFor(BufferId::ATTN_PROJ),
                .prepared_ref = preparedRefForGraphWeight(down_proj_binding, device),
                .prepared_store = prepared_weight_store_};

            // SwiGLU fusion: pass gate buffer to GEMM for fused silu(gate)*up + GEMM
            if (swiglu_fusion)
            {
                down_params.gate_input = buffers.gate;
                down_params.do_swiglu = true;
            }

            graph.addNode(prefix + "down_proj",
                          ComputeStageFactory::createGEMM(down_params),
                          device);
            ffn_terminal = prefix + "down_proj";

            // SwiGLU is always fused into GEMM: down_proj depends directly on gate_up_proj
            if (has_gate_up)
            {
                graph.addDependency(prefix + "down_proj", prefix + "gate_up_proj");
            }

            bool down_is_row_sharded = isRowParallelSharded(layer.down_proj);
            bool needs_allreduce = (down_is_row_sharded || config_.ffn_column_parallel);

            if (needs_allreduce && needsTPAllreduce())
            {
                size_t allreduce_count = static_cast<size_t>(total_tokens) * down_n;
                LOG_DEBUG("[buildFFNGraph] Adding down_allreduce: ffn_column_parallel="
                          << config_.ffn_column_parallel << " down_is_row_sharded=" << down_is_row_sharded
                          << " count=" << allreduce_count);

                std::string stage_name = prefix + "down_allreduce";
                auto allreduce_stage = createTPAllreduceStage(
                    buffers.attn_proj, allreduce_count, device, layer_idx, /*is_attention=*/false, stage_name,
                    buffers.idFor(BufferId::ATTN_PROJ));

                if (allreduce_stage)
                {
                    graph.addNode(stage_name, std::move(allreduce_stage), device);
                    graph.addDependency(stage_name, prefix + "down_proj");
                    ffn_terminal = stage_name;
                }
            }
        }

        // Stage 5: Residual connection
        // Non-last layers: Skip - fused into next layer's FusedResidualNormStage attn_norm
        // Last layer must keep ffn_residual since final_norm doesn't include residual add
        const bool skip_ffn_residual = (layer_idx < config_.pp_layer_offset + config_.n_layers - 1);
        if (!skip_ffn_residual)
        {
            ResidualAddStage::Params res_params;
            res_params.device_id = device;
            res_params.input = buffers.attn_proj;
            res_params.residual = buffers.current_hidden;
            res_params.output = buffers.current_hidden;
            res_params.num_elements = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
            res_params.input_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
            res_params.residual_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);
            res_params.output_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE); // In-place with residual

            graph.addNode(prefix + "ffn_residual",
                          ComputeStageFactory::createResidualAdd(res_params),
                          device);
            ffn_terminal = prefix + "ffn_residual";

            if (has_down_proj)
            {
                bool down_is_row_sharded = isRowParallelSharded(layer.down_proj);
                bool needs_allreduce = (down_is_row_sharded || config_.ffn_column_parallel);

                if (needs_allreduce && needsTPAllreduce())
                {
                    graph.addDependency(prefix + "ffn_residual", prefix + "down_allreduce");
                }
                else
                {
                    graph.addDependency(prefix + "ffn_residual", prefix + "down_proj");
                }
            }
        }

        graph.setTerminalNode(ffn_terminal);
        return graph;
    }

    // =============================================================================
    // Helper Methods
    // =============================================================================

    std::vector<int> QwenGraphBase::buildPositionIds(int seq_len, int batch_size, int offset)
    {
        std::vector<int> pos_ids(batch_size * seq_len);

        for (int b = 0; b < batch_size; ++b)
        {
            for (int s = 0; s < seq_len; ++s)
            {
                pos_ids[b * seq_len + s] = offset + s;
            }
        }

        return pos_ids;
    }

    // getSchema() is pure virtual - implemented by QwenStandardGraph, Qwen35Graph

    TensorContext QwenGraphBase::buildTensorContext() const
    {
        TensorContext tensors;

        // Activation buffers
        tensors.buffers["hidden"] = buffers_.current_hidden;
        tensors.buffers["normalized"] = buffers_.layer_buffers.normalized;
        tensors.buffers["Q"] = buffers_.layer_buffers.Q;
        tensors.buffers["K"] = buffers_.layer_buffers.K;
        tensors.buffers["V"] = buffers_.layer_buffers.V;
        tensors.buffers["attn_output"] = buffers_.layer_buffers.attn_output;
        tensors.buffers["attn_proj"] = buffers_.layer_buffers.attn_proj;
        tensors.buffers["gate"] = buffers_.layer_buffers.gate;
        tensors.buffers["up"] = buffers_.layer_buffers.up;
        tensors.buffers["logits"] = buffers_.logits;
        tensors.buffers["logits_local"] = buffers_.logits_local;

        // Model-level weights
        tensors.model_weights["embedding_table"] = modelEmbeddingTable();
        tensors.model_weights["final_norm"] = modelFinalNorm();
        tensors.model_weights["lm_head"] = modelLMHead();
        tensors.model_weight_bindings["embedding_table"] = modelEmbeddingBinding();
        tensors.model_weight_bindings["final_norm"] = modelFinalNormBinding();
        tensors.model_weight_bindings["lm_head"] = modelLMHeadBinding();

        // Layer weight accessor
        tensors.get_layer_weight = [this](int layer_idx, const std::string &name) -> TensorBase *
        {
            if (!hasLayerWeightSource())
                return nullptr;

            LayerWeights layer = layerWeightsForGraph(layer_idx);

            if (name == "wq")
                return layer.wq;
            if (name == "wk")
                return layer.wk;
            if (name == "wv")
                return layer.wv;
            if (name == "wo")
                return layer.wo;
            if (name == "attn_norm")
                return layer.attn_norm;
            if (name == "q_bias")
                return layer.q_bias;
            if (name == "k_bias")
                return layer.k_bias;
            if (name == "v_bias")
                return layer.v_bias;
            if (name == "q_norm")
                return layer.q_norm;
            if (name == "k_norm")
                return layer.k_norm;
            if (name == "attn_qkv")
                return layer.attn_qkv;
            if (name == "attn_gate")
                return layer.attn_gate;
            if (name == "ssm_alpha")
                return layer.ssm_alpha;
            if (name == "ssm_beta")
                return layer.ssm_beta;
            if (name == "ssm_conv1d")
                return layer.ssm_conv1d;
            if (name == "ssm_dt_bias")
                return layer.ssm_dt_bias;
            if (name == "ssm_a")
                return layer.ssm_a;
            if (name == "ssm_norm")
                return layer.ssm_norm;
            if (name == "ssm_out")
                return layer.ssm_out;
            if (name == "gate_proj")
                return layer.gate_proj;
            if (name == "up_proj")
                return layer.up_proj;
            if (name == "down_proj")
                return layer.down_proj;
            if (name == "ffn_norm")
                return layer.ffn_norm;
            if (name == "moe_gate")
                return layer.moe_gate;
            if (name == "moe_gate_exps")
                return layer.moe_gate_exps;
            if (name == "moe_up_exps")
                return layer.moe_up_exps;
            if (name == "moe_down_exps")
                return layer.moe_down_exps;
            if (name == "shared_expert_gate")
                return layer.shared_expert_gate;
            if (name == "shared_expert_up")
                return layer.shared_expert_up;
            if (name == "shared_expert_down")
                return layer.shared_expert_down;
            if (name == "shared_expert_gate_inp")
                return layer.shared_expert_gate_inp;

            LOG_WARN("[TensorContext] Unknown layer weight: " << name);
            return nullptr;
        };
        tensors.get_layer_weight_binding = [this](int layer_idx, const std::string &name) -> const WeightBinding *
        {
            LayerWeightBindings layer = layerWeightBindingsForGraph(layer_idx);

            if (name == "wq")
                return layer.wq;
            if (name == "wk")
                return layer.wk;
            if (name == "wv")
                return layer.wv;
            if (name == "wo")
                return layer.wo;
            if (name == "attn_norm")
                return layer.attn_norm;
            if (name == "q_bias")
                return layer.q_bias;
            if (name == "k_bias")
                return layer.k_bias;
            if (name == "v_bias")
                return layer.v_bias;
            if (name == "q_norm")
                return layer.q_norm;
            if (name == "k_norm")
                return layer.k_norm;
            if (name == "attn_qkv")
                return layer.attn_qkv;
            if (name == "attn_gate")
                return layer.attn_gate;
            if (name == "ssm_alpha")
                return layer.ssm_alpha;
            if (name == "ssm_beta")
                return layer.ssm_beta;
            if (name == "ssm_conv1d")
                return layer.ssm_conv1d;
            if (name == "ssm_dt_bias")
                return layer.ssm_dt_bias;
            if (name == "ssm_a")
                return layer.ssm_a;
            if (name == "ssm_norm")
                return layer.ssm_norm;
            if (name == "ssm_out")
                return layer.ssm_out;
            if (name == "gate_proj")
                return layer.gate_proj;
            if (name == "up_proj")
                return layer.up_proj;
            if (name == "down_proj")
                return layer.down_proj;
            if (name == "ffn_norm")
                return layer.ffn_norm;
            if (name == "moe_gate")
                return layer.moe_gate;
            if (name == "moe_gate_exps")
                return layer.moe_gate_exps;
            if (name == "moe_up_exps")
                return layer.moe_up_exps;
            if (name == "moe_down_exps")
                return layer.moe_down_exps;
            if (name == "shared_expert_gate")
                return layer.shared_expert_gate;
            if (name == "shared_expert_up")
                return layer.shared_expert_up;
            if (name == "shared_expert_down")
                return layer.shared_expert_down;
            if (name == "shared_expert_gate_inp")
                return layer.shared_expert_gate_inp;
            return nullptr;
        };

        return tensors;
    }

    GraphResolverConfig QwenGraphBase::getResolverConfig(int seq_len) const
    {
        GraphResolverConfig config;

        // MPI context
        if (mpi_ctx_)
        {
            config.world_size = mpi_ctx_->world_size();
            config.rank = mpi_ctx_->rank();
            config.mpi_ctx = mpi_ctx_.get();
        }

        // Sequence configuration
        config.seq_len = seq_len;
        config.batch_size = 1;

        // Device configuration
        config.default_device = config_.default_device;

        // Model architecture
        config.n_layers = config_.n_layers;
        config.d_model = config_.d_model;
        config.n_heads = config_.n_heads;
        config.n_kv_heads = config_.n_kv_heads;
        config.head_dim = config_.head_dim;
        config.d_ff = config_.d_ff;
        config.vocab_size = config_.vocab_size;
        config.rms_norm_eps = config_.rms_norm_eps;
        config.rope_theta = config_.rope_theta;

        // Phase 5.4: VNNI-safe Q16 KV cache scale
        config.kv_cache_scale_k = config_.kv_cache_scale_k;
        config.kv_cache_scale_v = config_.kv_cache_scale_v;

        // TP-adjusted local dimensions
        // Use local head counts when QKV is column-parallel
        config.local_n_heads = config_.qkv_column_parallel
                                   ? config_.local_n_heads
                                   : config_.n_heads;
        config.local_n_kv_heads = config_.qkv_column_parallel
                                      ? config_.local_n_kv_heads
                                      : config_.n_kv_heads;

        // Validate (safety check for uninitialized config)
        if (config.local_n_heads <= 0)
            config.local_n_heads = config_.n_heads;
        if (config.local_n_kv_heads <= 0)
            config.local_n_kv_heads = config_.n_kv_heads;

        // Use local FFN dimension when FFN is column-parallel
        config.local_d_ff = config_.ffn_column_parallel
                                ? config_.d_ff_local
                                : config_.d_ff;
        if (config.local_d_ff <= 0)
            config.local_d_ff = config_.d_ff;

        // Use local vocab when LM head is column-parallel
        config.local_vocab = config_.lm_head_column_parallel
                                 ? config_.vocab_local
                                 : config_.vocab_size;
        if (config.local_vocab <= 0)
            config.local_vocab = config_.vocab_size;

        LOG_DEBUG("[QwenGraphBase::getResolverConfig] Created config: "
                  << "seq_len=" << config.seq_len << ", "
                  << "local_n_heads=" << config.local_n_heads << " (n_heads=" << config.n_heads << "), "
                  << "local_n_kv_heads=" << config.local_n_kv_heads << " (n_kv_heads=" << config.n_kv_heads << "), "
                  << "local_d_ff=" << config.local_d_ff << " (d_ff=" << config.d_ff << "), "
                  << "local_vocab=" << config.local_vocab << " (vocab_size=" << config.vocab_size << ")");

        return config;
    }

    void QwenGraphBase::addFinalNormToGraph(
        ComputeGraph &graph,
        TensorBase *hidden,
        TensorBase *normalized_out,
        const std::string &prev_node,
        int n_tokens,
        DeviceId device)
    {
        RMSNormStage::Params norm_params;
        norm_params.input = hidden;
        norm_params.output = normalized_out;
        norm_params.gamma = modelFinalNorm();
        norm_params.eps = config_.rms_norm_eps;
        norm_params.seq_len = n_tokens;
        norm_params.device_id = device;
        norm_params.input_buffer_id = BufferId::HIDDEN_STATE;
        norm_params.output_buffer_id = BufferId::NORMALIZED;

        graph.addNode("final_norm",
                      ComputeStageFactory::createRMSNorm(norm_params),
                      device);

        if (!prev_node.empty())
        {
            graph.addDependency("final_norm", prev_node);
        }
    }

    const TPDomain *QwenGraphBase::getDomainForLayer(int layer_idx, bool is_attention) const
    {
        if (!config_.multi_domain_tp_config)
        {
            return nullptr; // No domain config - use legacy MPI path
        }
        return config_.multi_domain_tp_config->domainForLayer(layer_idx, is_attention);
    }

    // =========================================================================
    // TP Allreduce Helpers
    // =========================================================================

    bool QwenGraphBase::needsTPAllreduce() const
    {
        // Unified check: any TP context with degree > 1 (LOCAL or GLOBAL)
        if (config_.tp_ctx && config_.tp_ctx->degree() > 1)
        {
            return true;
        }
        // Legacy fallback: GLOBAL TP without tp_ctx (direct MPI path)
        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            return true;
        }
        return false;
    }

    std::unique_ptr<IComputeStage> QwenGraphBase::createTPAllreduceStage(
        TensorBase *buffer,
        size_t count,
        DeviceId device,
        int layer_idx,
        bool is_attention,
        const std::string &stage_name,
        std::optional<BufferId> tensor_buffer_id) const
    {
        // Unified path: use polymorphic ITPContext for both LOCAL and GLOBAL TP
        if (config_.tp_ctx && config_.tp_ctx->degree() > 1)
        {
            LOG_DEBUG("[QwenGraphBase] Creating TPAllreduceStage: degree="
                      << config_.tp_ctx->degree()
                      << " device_idx=" << config_.tp_device_idx
                      << " count=" << count
                      << " backend=" << static_cast<int>(config_.tp_ctx->backend())
                      << " local=" << config_.tp_ctx->isLocal()
                      << " stage_name=" << stage_name);

            TPAllreduceStage::Params params;
            params.device_id = device;
            params.tp_ctx = config_.tp_ctx; // ITPContext* — polymorphic for LOCAL and GLOBAL
            params.tensor = buffer;
            params.count = count;
            params.stage_name = stage_name;
            params.precision = config_.getAllreducePrecisionForLayer(layer_idx);
            params.tensor_buffer_id = tensor_buffer_id;

            return std::make_unique<TPAllreduceStage>(params);
        }

        // Legacy fallback: GLOBAL TP without tp_ctx wired (direct MPI path)
        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            LOG_DEBUG("[QwenGraphBase] Creating AllreduceStage (legacy MPI): world_size="
                      << mpi_ctx_->world_size()
                      << " rank=" << mpi_ctx_->rank()
                      << " count=" << count);

            AllreduceStage::Params params;
            params.device_id = device;
            params.mpi_ctx = mpi_ctx_.get();
            params.buffer = buffer;
            params.count = count;
            params.collective_ctx = nullptr;
            params.domain = getDomainForLayer(layer_idx, is_attention);

            return ComputeStageFactory::createAllreduce(params);
        }

        // No TP active - should not reach here (caller should check needsTPAllreduce())
        LOG_WARN("[QwenGraphBase] createTPAllreduceStage called but no TP active");
        return nullptr;
    }

    // =========================================================================
    // Shared Attention Building Blocks
    // =========================================================================

    std::pair<int, int> QwenGraphBase::resolveLocalHeadCounts() const
    {
        int local_n_heads = config_.qkv_column_parallel
                                ? config_.local_n_heads
                                : config_.n_heads;
        int local_n_kv_heads = config_.qkv_column_parallel
                                   ? config_.local_n_kv_heads
                                   : config_.n_kv_heads;
        if (local_n_heads <= 0)
            local_n_heads = config_.n_heads;
        if (local_n_kv_heads <= 0)
            local_n_kv_heads = config_.n_kv_heads;
        return {local_n_heads, local_n_kv_heads};
    }

    std::string QwenGraphBase::addPreAttentionNorm(
        ComputeGraph &graph,
        const std::string &prefix,
        ActivationBuffers &buffers,
        TensorBase *norm_gamma,
        int total_tokens,
        int layer_idx,
        DeviceId device,
        bool check_hybrid_q16)
    {
        const std::string node_name = prefix + "attn_norm";
        const bool use_fused = (!check_hybrid_q16 || !config_.isHybridQ16()) &&
                               layer_idx > config_.pp_layer_offset;

        if (use_fused)
        {
            FusedResidualNormStage::Params fused_params;
            fused_params.device_id = device;
            fused_params.input = buffers.attn_proj;
            fused_params.residual = buffers.current_hidden;
            fused_params.gamma = norm_gamma;
            fused_params.norm_output = buffers.normalized;
            fused_params.eps = config_.rms_norm_eps;
            fused_params.seq_len = total_tokens;
            fused_params.hidden_dim = config_.d_model;
            fused_params.input_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
            fused_params.residual_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);
            fused_params.norm_output_buffer_id = buffers.idFor(BufferId::NORMALIZED);

            graph.addNode(node_name,
                          ComputeStageFactory::createFusedResidualNorm(fused_params),
                          device);
        }
        else
        {
            RMSNormStage::Params norm_params;
            norm_params.input = buffers.current_hidden;
            norm_params.output = buffers.normalized;
            norm_params.gamma = norm_gamma;
            norm_params.eps = config_.rms_norm_eps;
            norm_params.seq_len = total_tokens;
            norm_params.device_id = device;
            norm_params.input_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);
            norm_params.output_buffer_id = buffers.idFor(BufferId::NORMALIZED);

            graph.addNode(node_name,
                          ComputeStageFactory::createRMSNorm(norm_params),
                          device);
        }

        return node_name;
    }

    bool QwenGraphBase::addQKNorms(
        ComputeGraph &graph,
        const std::string &prefix,
        ActivationBuffers &buffers,
        const LayerWeights &layer,
        int local_n_heads,
        int local_n_kv_heads,
        int total_tokens,
        DeviceId device,
        const std::string &q_dependency,
        const std::string &k_dependency)
    {
        if (!layer.q_norm || !layer.k_norm)
            return false;

        LOG_DEBUG("[QwenGraphBase] Layer using QK norm");

        graph.addNode(prefix + "q_norm",
                      ComputeStageFactory::createQKNorm({
                          .device_id = device,
                          .input = buffers.Q,
                          .output = buffers.Q,
                          .gamma = layer.q_norm,
                          .n_heads = local_n_heads,
                          .head_dim = config_.head_dim,
                          .eps = config_.rms_norm_eps,
                          .seq_len = total_tokens,
                          .input_buffer_id = buffers.idFor(BufferId::Q_PROJ),
                          .output_buffer_id = buffers.idFor(BufferId::Q_PROJ),
                      }),
                      device);
        graph.addDependency(prefix + "q_norm", q_dependency);

        graph.addNode(prefix + "k_norm",
                      ComputeStageFactory::createQKNorm({
                          .device_id = device,
                          .input = buffers.K,
                          .output = buffers.K,
                          .gamma = layer.k_norm,
                          .n_heads = local_n_kv_heads,
                          .head_dim = config_.head_dim,
                          .eps = config_.rms_norm_eps,
                          .seq_len = total_tokens,
                          .input_buffer_id = buffers.idFor(BufferId::K_PROJ),
                          .output_buffer_id = buffers.idFor(BufferId::K_PROJ),
                      }),
                      device);
        graph.addDependency(prefix + "k_norm", k_dependency);

        return true;
    }

    std::string QwenGraphBase::addRoPE(
        ComputeGraph &graph,
        const std::string &prefix,
        ActivationBuffers &buffers,
        int local_n_heads,
        int local_n_kv_heads,
        int total_tokens,
        const int *position_ids,
        DeviceId device)
    {
        const std::string node_name = prefix + "rope";
        int pos_offset = position_ids ? position_ids[0] : 0;

        graph.addNode(node_name,
                      ComputeStageFactory::createRoPE({
                          .device_id = device,
                          .Q = buffers.Q,
                          .K = buffers.K,
                          .n_heads = local_n_heads,
                          .n_kv_heads = local_n_kv_heads,
                          .head_dim = config_.head_dim,
                          .pos_offset = pos_offset,
                          .theta_base = config_.rope_theta,
                          .seq_len = total_tokens,
                          .partial_rotary_factor = config_.partial_rotary_factor,
                          .position_ids = position_ids,
                          .skip_k = config_.rope_on_read,
                          .q_buffer_id = buffers.idFor(BufferId::Q_PROJ),
                          .k_buffer_id = buffers.idFor(BufferId::K_PROJ),
                      }),
                      device);

        return node_name;
    }

    std::string QwenGraphBase::addKVCacheAppend(
        ComputeGraph &graph,
        const std::string &prefix,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int batch_size,
        IKVCache *kv_cache,
        DeviceId device,
        const std::string &rope_dependency,
        bool layer_idx_is_cache_local)
    {
        int total_tokens = batch_size * seq_len;
        int kv_local_layer = layer_idx_is_cache_local
                                 ? layer_idx
                                 : layer_idx - config_.pp_layer_offset;

        if (kv_cache)
        {
            graph.addNode(prefix + "kv_append",
                          ComputeStageFactory::createKVCacheAppend({
                              .device_id = device,
                              .K = buffers.K,
                              .V = buffers.V,
                              .kv_cache = kv_cache,
                              .layer_idx = kv_local_layer,
                              .seq_idx = 0,
                              .num_tokens = total_tokens,
                              .batch_size = batch_size,
                              .seq_len = seq_len,
                              .kv_cache_scale_k = config_.kv_cache_scale_k,
                              .kv_cache_scale_v = config_.kv_cache_scale_v,
                              .head_dim = config_.head_dim,
                              .turboquant_ctx = config_.turboquant_ctx,
                              .kv_rotation = config_.kv_rotation,
                              .k_buffer_id = buffers.idFor(BufferId::K_PROJ),
                              .v_buffer_id = buffers.idFor(BufferId::V_PROJ),
                          }),
                          device);

            // In rope-on-read mode the RoPE stage skips K mutation, but it still
            // carries the Q/K norm dependencies. Appending directly after QKV GEMM
            // can cache pre-normalized K on Qwen3.5 FA layers.
            graph.addDependency(prefix + "kv_append", rope_dependency);
            return prefix + "kv_append";
        }

        return rope_dependency;
    }

    std::string QwenGraphBase::addKVCacheAndAttention(
        ComputeGraph &graph,
        const std::string &prefix,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int batch_size,
        int local_n_heads,
        int local_n_kv_heads,
        IKVCache *kv_cache,
        const int *position_ids,
        DeviceId device,
        bool has_qkv_proj,
        const std::string &rope_dependency,
        bool layer_idx_is_cache_local)
    {
        int total_tokens = batch_size * seq_len;
        int kv_local_layer = layer_idx_is_cache_local
                                 ? layer_idx
                                 : layer_idx - config_.pp_layer_offset;

        const std::string kv_append_dependency =
            addKVCacheAppend(
                graph,
                prefix,
                buffers,
                layer_idx,
                seq_len,
                batch_size,
                kv_cache,
                device,
                rope_dependency,
                layer_idx_is_cache_local);

        // --- Determine K/V source for attention ---
        ITensor *K_for_attn = buffers.K;
        ITensor *V_for_attn = buffers.V;
        int kv_len = total_tokens;
        bool use_gather_stage = false;

        if (kv_cache)
        {
            int cached_tokens = kv_cache->get_cached_tokens(kv_local_layer, 0);
            if (cached_tokens > 0 && batch_size == 1)
            {
                K_for_attn = kv_cache->get_k(kv_local_layer, 0);
                V_for_attn = kv_cache->get_v(kv_local_layer, 0);
                kv_len = cached_tokens;
                LOG_TRACE("[QwenGraphBase] Layer " << layer_idx << " using cached K/V (decode mode)");
            }
            else if (cached_tokens > 0 && batch_size > 1)
            {
                if (buffers.gathered_K && buffers.gathered_V)
                {
                    use_gather_stage = true;
                    K_for_attn = buffers.gathered_K;
                    V_for_attn = buffers.gathered_V;
                    kv_len = cached_tokens;
                    LOG_TRACE("[QwenGraphBase] Layer " << layer_idx << " using gathered K/V (batched decode)");
                }
                else
                {
                    LOG_WARN("[QwenGraphBase] Layer " << layer_idx
                                                      << " batched decode but no gather buffers");
                }
            }
        }

        // --- KV Cache Gather (batched decode) ---
        if (use_gather_stage)
        {
            graph.addNode(prefix + "kv_gather",
                          ComputeStageFactory::createKVCacheGather({
                              .kv_cache = kv_cache,
                              .layer_idx = kv_local_layer,
                              .batch_size = batch_size,
                              .out_K = buffers.gathered_K,
                              .out_V = buffers.gathered_V,
                          }),
                          device);
            graph.addDependency(prefix + "kv_gather", prefix + "kv_append");
        }

        // --- Attention Compute ---
        {
            AttentionMode mode = detect_attention_mode(batch_size, seq_len, kv_len);
            LOG_TRACE("[QwenGraphBase] Layer " << layer_idx
                                               << " attention mode: " << attention_mode_name(mode)
                                               << " (batch=" << batch_size << ", seq=" << seq_len << ", kv=" << kv_len << ")");

            AttentionComputeStage::Params attn_params;
            attn_params.device_id = device;
            attn_params.Q = buffers.Q;
            attn_params.K = K_for_attn;
            attn_params.V = V_for_attn;
            attn_params.output = buffers.attn_output;
            attn_params.batch_size = batch_size;
            attn_params.seq_len = seq_len;
            attn_params.kv_len = kv_len;
            attn_params.n_heads = local_n_heads;
            attn_params.n_kv_heads = local_n_kv_heads;
            attn_params.head_dim = config_.head_dim;
            attn_params.head_start = config_.head_start;
            // GQA rep: only when KV heads are replicated (not column-parallel)
            if (local_n_kv_heads == config_.n_kv_heads && config_.n_kv_heads > 0 && local_n_heads != config_.n_heads)
                attn_params.gqa_n_rep = config_.n_heads / config_.n_kv_heads;
            attn_params.causal = true;
            attn_params.window_size = -1;
            attn_params.attention_mode = mode;
            attn_params.auto_detect_mode = true;
            attn_params.workspace_scores = buffers.workspace_scores;
            attn_params.workspace_context = buffers.workspace_context;
            attn_params.workspace_mask = buffers.workspace_mask;
            attn_params.kv_cache = kv_cache;
            attn_params.layer_idx = kv_local_layer;
            attn_params.read_kv_from_cache = device.is_gpu() &&
                                             (!kv_cache || kv_cache->precision() != ActivationPrecision::Q8_1) &&
                                             (!kv_cache || (kv_cache->precision() != ActivationPrecision::TQ8 &&
                                                            kv_cache->precision() != ActivationPrecision::TQ4));
            attn_params.position_offset = position_ids ? position_ids[0] : 0;
            attn_params.mpi_ctx = mpi_ctx_.get();
            attn_params.q_buffer_id = buffers.idFor(BufferId::Q_PROJ);
            attn_params.output_buffer_id = buffers.idFor(BufferId::ATTN_OUTPUT);
            // NOTE: workspace_scores/workspace_context are no longer registered in
            // the arena (CPUFlashAttentionKernelT doesn't use them — O(S²) dead
            // buffers removed). Don't set buffer_ids for the contract.
            attn_params.turboquant_ctx = config_.turboquant_ctx;
            attn_params.kv_rotation = config_.kv_rotation;

            if (config_.rope_on_read)
            {
                attn_params.apply_rope_to_k = true;
                attn_params.rope_theta = config_.rope_theta;
                attn_params.partial_rotary_factor = config_.partial_rotary_factor;
            }

            graph.addNode(prefix + "attention",
                          ComputeStageFactory::createAttentionCompute(attn_params),
                          device);

            if (use_gather_stage)
                graph.addDependency(prefix + "attention", prefix + "kv_gather");
            else if (kv_cache)
                graph.addDependency(prefix + "attention", kv_append_dependency);
            else
                graph.addDependency(prefix + "attention", rope_dependency);
        }

        return prefix + "attention";
    }

    std::string QwenGraphBase::addWoProjectionAndAllreduce(
        ComputeGraph &graph,
        const std::string &prefix,
        ActivationBuffers &buffers,
        TensorBase *wo_weight,
        const WeightBinding *wo_binding,
        int total_tokens,
        int layer_idx,
        DeviceId device,
        const std::string &dependency,
        const std::string &wo_node_suffix,
        const std::string &allreduce_node_suffix)
    {
        if (!wo_weight)
            return dependency;

        int wo_n = static_cast<int>(wo_weight->shape()[0]);
        int wo_k = static_cast<int>(wo_weight->shape()[1]);

        std::string wo_node = prefix + wo_node_suffix;
        graph.addNode(wo_node,
                      ComputeStageFactory::createGEMM({
                          .device_id = device,
                          .A = buffers.attn_output,
                          .B = wo_weight,
                          .C = buffers.attn_proj,
                          .m = total_tokens,
                          .n = wo_n,
                          .k = wo_k,
                          .alpha = 1.0f,
                          .beta = 0.0f,
                          .transpose_B = false,
                          .gemm_context = GemmContext::ATTN,
                          .a_buffer_id = buffers.idFor(BufferId::ATTN_OUTPUT),
                          .c_buffer_id = buffers.idFor(BufferId::ATTN_PROJ),
                          .prepared_ref = preparedRefForGraphWeight(wo_binding, device),
                          .prepared_store = prepared_weight_store_,
                      }),
                      device);
        graph.addDependency(wo_node, dependency);

        std::string terminal = wo_node;

        // TP allreduce if row-parallel sharded
        if (isRowParallelSharded(wo_weight) && needsTPAllreduce())
        {
            size_t allreduce_count = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
            std::string ar_node = prefix + allreduce_node_suffix;

            auto allreduce_stage = createTPAllreduceStage(
                buffers.attn_proj, allreduce_count, device, layer_idx,
                /*is_attention=*/true, ar_node, buffers.idFor(BufferId::ATTN_PROJ));

            if (allreduce_stage)
            {
                graph.addNode(ar_node, std::move(allreduce_stage), device);
                graph.addDependency(ar_node, wo_node);
                terminal = ar_node;
            }
        }

        return terminal;
    }

} // namespace llaminar2
