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
#include "../../execution/moe/MoERebalanceController.h"
#include "../../execution/mtp/MTPSpecDecodeMetadata.h"
#include "../../kernels/IHybridKVCache.h"
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
         * @brief Validate position ownership and select optional host rows.
         *
         * Graph builders must preserve the position representation declared by
         * ForwardInput. They may not manufacture host rows for a contiguous GPU
         * range: doing so records an asynchronous H2D copy whose mutable source
         * can race the next chunk. ExplicitRows accepts either host rows or
         * device-resident rows; ContiguousOffset accepts only a single request
         * represented by the scalar position_offset.
         *
         * @param input Forward invocation whose position contract is validated.
         * @param graph_name Graph path used to make fatal diagnostics actionable.
         * @return Host position rows, or null for device rows/contiguous offsets.
         * @throws std::runtime_error when the declared contract is inconsistent.
         */
        const int *validatedHostPositionRows(
            const ForwardInput &input,
            const char *graph_name)
        {
            if (input.position_policy == ForwardPositionPolicy::ContiguousOffset)
            {
                if (input.batch_size != 1)
                {
                    throw std::runtime_error(
                        std::string(graph_name) +
                        ": contiguous position offsets require batch_size=1");
                }
                if (input.position_ids || input.position_ids_device)
                {
                    throw std::runtime_error(
                        std::string(graph_name) +
                        ": contiguous position offsets must not also provide position rows");
                }
                return nullptr;
            }

            if (!input.position_ids && !input.position_ids_device)
            {
                throw std::runtime_error(
                    std::string(graph_name) +
                    ": explicit position policy requires host or device rows");
            }
            if (input.position_ids_device && !input.device.is_gpu())
            {
                throw std::runtime_error(
                    std::string(graph_name) +
                    ": device position rows require a GPU graph");
            }
            return input.position_ids;
        }

        /**
         * @brief Resolve the layer id passed to a KV-cache stage.
         *
         * Plain PP-local ring caches are sized only for the current stage, so
         * they still need compact local ids. Hybrid caches own the FA/GDN layer
         * map internally and need the absolute model layer id; otherwise a
         * second PP-stage offset subtraction can alias different FA layers onto
         * the same compressed cache slot.
         *
         * MTP sidecar caches opt into already-local ids with
         * layer_idx_is_cache_local and intentionally bypass both mappings.
         */
        int kvCacheLayerForGraphStage(const IKVCache *kv_cache,
                                      int layer_idx,
                                      int pp_layer_offset,
                                      bool layer_idx_is_cache_local)
        {
            if (layer_idx_is_cache_local)
            {
                return layer_idx;
            }
            if (dynamic_cast<const IHybridKVCache *>(kv_cache))
            {
                return layer_idx;
            }
            return layer_idx - pp_layer_offset;
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
                layout.compute_all_positions =
                    config.compute_all_position_logits ||
                    config.live_mtp_request_batch_condition;
                /*
                 * A full-identity verifier bypasses the compact row-copy stage,
                 * but its normalized activation still begins at physical row
                 * zero. It must therefore retain the verifier's no-offset
                 * semantics rather than inheriting bucketed-prefill replay
                 * offsets merely because both use NORMALIZED.
                 */
                layout.use_prefill_replay_row_offset =
                    !config.compute_row_indexed_logits;
                return layout;
            }

            layout.seq_len = 1;
            layout.input_buffer_id = BufferId::LM_HEAD_INPUT_ROW;
            layout.compute_all_positions = false;
            layout.use_prefill_replay_row_offset = false;
            return layout;
        }

        /**
         * @brief Decide whether compact verifier LM-head rows need decode-equivalent grouping.
         *
         * MTP verifier rows are publishable live-model state, so their LM-head
         * projection must be byte-identical to running each row through the
         * backend's normal M=1 decode projection.  Plain small-M GEMM may choose
         * a different reduction order and drift by a few ULPs, which is enough to
         * break the verifier contract.  The LMHeadStage decode-equivalent route
         * keeps the grouped API while selecting backend policies proven against
         * serial decode for the compact verifier window.
         */
        bool shouldForceDecodeEquivalentLMHeadVerifierPrefill(
            const GraphConfig &config,
            DeviceId device,
            const LMHeadInputLayout &layout)
        {
            return (device.is_cpu() || device.is_cuda() || device.is_rocm()) &&
                   layout.compute_all_positions &&
                   layout.seq_len > 1 &&
                   config.usesMTPGroupedDecodeEquivalentRows();
        }

        /**
         * @brief Resolve and validate compact verifier source rows.
         *
         * The request-batched MTP verifier publishes a logical row plan that
         * may skip over non-target rows in a flattened verifier sequence. Older
         * single-request paths did not need to publish a plan because compact
         * rows were always the leading rows. Keeping that default here lets
         * existing tests and exact-shape graphs continue to behave the same,
         * while non-empty plans fail loudly if they do not match the graph
         * shape or the verifier activation tensor.
         */
        std::vector<int> resolveRowIndexedLogitRows(
            const GraphConfig &config,
            int row_count,
            int total_tokens,
            const char *context)
        {
            std::vector<int> selected_rows;
            if (config.row_indexed_logits_selected_rows.empty())
            {
                selected_rows.reserve(static_cast<size_t>(row_count));
                for (int row = 0; row < row_count; ++row)
                    selected_rows.push_back(row);
            }
            else
            {
                selected_rows = config.row_indexed_logits_selected_rows;
                if (static_cast<int>(selected_rows.size()) != row_count)
                {
                    LOG_ERROR("[QwenGraphBase] " << context
                                                 << " row plan length "
                                                 << selected_rows.size()
                                                 << " does not match row_count="
                                                 << row_count);
                    throw std::runtime_error("row-indexed all-position logits row plan length mismatch");
                }
            }

            for (int row : selected_rows)
            {
                if (row < 0 || row >= total_tokens)
                {
                    LOG_ERROR("[QwenGraphBase] " << context
                                                 << " row plan selected row "
                                                 << row
                                                 << " outside verifier token range 0.."
                                                 << (total_tokens - 1));
                    throw std::runtime_error("row-indexed all-position logits row plan out of range");
                }
            }
            return selected_rows;
        }

        /**
         * @brief Test whether row-indexed logits already form the dense input.
         *
         * The empty row list has the established meaning `[0, row_count)`.
         * An explicit list is a dense identity cover only when it has one entry
         * for every physical activation row and each entry names that same row.
         * Malformed explicit lists deliberately return false so the ordinary
         * resolver emits its detailed fatal validation error.
         *
         * @param config Graph policy carrying an optional explicit row list.
         * @param row_count Number of compact logits requested by the verifier.
         * @param total_tokens Number of physical rows in the activation tensor.
         * @return True when LM head can consume the activation without copying.
         */
        bool rowIndexedLogitsCoverFullPhysicalInput(
            const GraphConfig &config,
            int row_count,
            int total_tokens)
        {
            if (row_count <= 0 || row_count != total_tokens)
                return false;

            if (config.row_indexed_logits_selected_rows.empty())
                return true;
            if (static_cast<int>(
                    config.row_indexed_logits_selected_rows.size()) !=
                total_tokens)
            {
                return false;
            }

            for (int row = 0; row < total_tokens; ++row)
            {
                if (config.row_indexed_logits_selected_rows[
                        static_cast<size_t>(row)] != row)
                {
                    return false;
                }
            }
            return true;
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

    bool QwenGraphBase::hasDecodeReplicatedDenseWeightSource() const
    {
        return decode_replicated_dense_weight_bindings_.get_layer_weights != nullptr;
    }

    std::string QwenGraphBase::describeDecodeReplicatedDenseBindingState() const
    {
        auto state = [](const void *ptr)
        {
            return ptr ? "present" : "missing";
        };
        return std::string("layers=") +
               (decode_replicated_dense_weight_bindings_.get_layer_weights ? "present" : "missing") +
               " embedding=" + state(decode_replicated_dense_weight_bindings_.embedding_table) +
               " final_norm=" + state(decode_replicated_dense_weight_bindings_.final_norm) +
               " lm_head=" + state(decode_replicated_dense_weight_bindings_.lm_head);
    }

    bool QwenGraphBase::useDecodeReplicatedDenseWeights() const
    {
        return decode_replicated_dense_graph_active_ &&
               hasDecodeReplicatedDenseWeightSource();
    }

    bool QwenGraphBase::hasDecodeMirroredEmbeddingWeightSource() const
    {
        return decode_replicated_dense_weight_bindings_.embedding_table != nullptr;
    }

    bool QwenGraphBase::useDecodeMirroredEmbeddingWeights() const
    {
        return decode_mirrored_embedding_graph_active_ &&
               hasDecodeMirroredEmbeddingWeightSource();
    }

    bool QwenGraphBase::hasMirroredMTPHeadWeightSource() const
    {
        return decode_replicated_dense_weight_bindings_.final_norm != nullptr &&
               decode_replicated_dense_weight_bindings_.lm_head != nullptr;
    }

    bool QwenGraphBase::localTPMirroredMTPHeadConfigured() const
    {
        /*
         * Terminal-head ownership is a graph-layout policy, not a switch that
         * may change when speculative execution is enabled.  Serial decode is
         * the arithmetic oracle for grouped MTP, so both lanes must bind the
         * same full-vocabulary weight and sampling surface.  Tying mirroring to
         * mtp.enabled made the oracle column-sharded while the verifier was
         * mirrored, leaving stochastic parity dependent on two different
         * reduction and sampling implementations.
         */
        return mtpTerminalHeadIsMirrored(config_.mtp.terminal_head_policy) &&
               config_.lm_head_column_parallel &&
               config_.tp_ctx != nullptr &&
               config_.tp_ctx->isLocal();
    }

    bool QwenGraphBase::mirroredMTPHeadActiveForProjectedRows(int total_tokens) const
    {
        /*
         * Ordinary prefill can contain thousands of transformer rows while its
         * LM head projects only the final row. Request-batched prefill projects
         * one compact terminal row per request, and an all-position verifier
         * projects its explicit target rows. The mirrored-head decision follows
         * that terminal projection geometry, never the transformer activation
         * size. Otherwise a long prefill leaves the scalar MTP first-target
         * sample on a sharded head even though every participant already owns
         * the replicated weight.
         */
        const int projected_rows =
            config_.compute_row_indexed_logits
                ? config_.row_indexed_logits_row_count
                : (config_.compute_all_position_logits ||
                   config_.live_mtp_request_batch_condition)
                      ? total_tokens
                      : 1;
        return localTPMirroredMTPHeadConfigured() &&
               projected_rows > 0;
    }

    bool QwenGraphBase::useMirroredMTPHeadWeights() const
    {
        return mtp_mirrored_lm_head_graph_active_;
    }

    bool QwenGraphBase::useFullVocabEmbeddingForCurrentGraph() const
    {
        return useDecodeReplicatedDenseWeights() ||
               useDecodeMirroredEmbeddingWeights();
    }

    bool QwenGraphBase::usesVocabParallelEmbeddingForCurrentGraph() const
    {
        TensorBase *embedding = modelEmbeddingTable();
        return embedding && static_cast<int>(embedding->rows()) < config_.vocab_size;
    }

    int QwenGraphBase::embeddingVocabOffsetForCurrentGraph(DeviceId device) const
    {
        return usesVocabParallelEmbeddingForCurrentGraph()
                   ? embeddingVocabOffsetForDevice(config_, device)
                   : 0;
    }

    bool QwenGraphBase::useReplicatedAttentionStateWeights() const
    {
        return replicated_attention_state_graph_active_ &&
               hasDecodeReplicatedDenseWeightSource();
    }

    QwenGraphBase::FinalProjectionPolicy QwenGraphBase::resolveFinalProjectionPolicy(
        const FinalProjectionPolicyRequest &request) const
    {
        FinalProjectionPolicy policy{
            .norm_source = request.norm_source,
            .head_policy = FinalHeadPolicy::PrimaryFullVocabulary,
            .norm_gamma = nullptr,
            .lm_head_weight = nullptr,
            .lm_head_binding = nullptr,
            .lm_head_output = nullptr,
            .lm_head_vocab_size = 0,
            .serial_equivalent_partition_width = 0,
            .column_parallel = false,
            .needs_allgather = false,
        };

        switch (request.norm_source)
        {
        case FinalNormSource::ModelOutputNorm:
            policy.norm_gamma = modelFinalNorm();
            break;
        case FinalNormSource::MTPSidecarNorm:
            policy.norm_gamma = request.mtp_norm;
            break;
        }

        /*
         * Keep the layout decision declarative and local to the policy resolver.
         * Callers state whether a full-vocabulary head is required; this method
         * chooses the tensor, binding, vocab width, and follow-up allgather
         * contract as one coherent package.
         */
        policy.column_parallel =
            !request.force_full_vocabulary_head &&
            useColumnParallelLMHeadForGraph(request.column_parallel_output);
        policy.lm_head_output =
            policy.column_parallel
                ? request.column_parallel_output
                : request.full_vocab_output;
        policy.lm_head_vocab_size =
            policy.column_parallel ? config_.vocab_local : config_.vocab_size;
        policy.needs_allgather =
            needsDistributedLMHeadAllGather(policy.column_parallel);
        policy.serial_equivalent_partition_width =
            serialEquivalentLMHeadPartitionWidth(policy.column_parallel);

        if (policy.column_parallel)
        {
            policy.head_policy = FinalHeadPolicy::PrimaryColumnParallel;
        }
        else if (useMirroredMTPHeadWeights())
        {
            policy.head_policy = FinalHeadPolicy::MirroredLocalTPMTPFullVocabulary;
        }
        else if (useDecodeReplicatedDenseWeights())
        {
            policy.head_policy = FinalHeadPolicy::DecodeReplicatedFullVocabulary;
        }
        else
        {
            policy.head_policy = FinalHeadPolicy::PrimaryFullVocabulary;
        }

        policy.lm_head_weight = modelLMHeadForGraph(policy.column_parallel);
        policy.lm_head_binding = modelLMHeadBindingForGraph(policy.column_parallel);
        return policy;
    }

    int QwenGraphBase::serialEquivalentLMHeadPartitionWidth(
        bool column_parallel) const
    {
        if (column_parallel || !useMirroredMTPHeadWeights())
            return 0;

        if (config_.vocab_local <= 0 || config_.vocab_size <= 0 ||
            config_.vocab_local > config_.vocab_size ||
            (config_.vocab_size % config_.vocab_local) != 0)
        {
            throw std::logic_error(
                "[QwenGraphBase] Mirrored LocalTP LM head requires an equal, positive serial vocabulary partition");
        }

        return config_.vocab_local == config_.vocab_size
                   ? 0
                   : config_.vocab_local;
    }

    bool QwenGraphBase::denseDecodeReplicatedActiveForTokens(int total_tokens) const
    {
        const int max_decode_like_rows = config_.mtp.enabled
            ? std::max(1, resolveMTPMaxTargetQueryRows(config_.mtp))
            : 1;
        const bool decode_like =
            config_.dense_tp_enabled &&
            config_.dense_tp_decode_replicated &&
            forwardPhaseAllowsDecodeTopology() &&
            total_tokens > 0 &&
            total_tokens <= max_decode_like_rows;
        if (!decode_like)
            return false;
        if (!hasDecodeReplicatedDenseWeightSource())
        {
            throw std::runtime_error(
                "[QwenGraphBase] Replicated dense decode is configured but its layer weight bindings are missing: " +
                describeDecodeReplicatedDenseBindingState());
        }
        return true;
    }

    bool QwenGraphBase::denseDecodeMirroredEmbeddingActiveForTokens(int total_tokens) const
    {
        const int max_decode_like_rows = config_.mtp.enabled
            ? std::max(1, resolveMTPMaxTargetQueryRows(config_.mtp))
            : 1;
        return config_.dense_tp_enabled &&
               config_.dense_tp_decode_mirrored_embedding &&
               forwardPhaseAllowsDecodeTopology() &&
               total_tokens > 0 &&
               total_tokens <= max_decode_like_rows &&
               hasDecodeMirroredEmbeddingWeightSource();
    }

    bool QwenGraphBase::replicatedAttentionStateActiveForTokens(int total_tokens) const
    {
        const int max_decode_like_rows = config_.mtp.enabled
            ? std::max(1, resolveMTPMaxTargetQueryRows(config_.mtp))
            : 1;
        const bool decode_like =
            config_.dense_tp_enabled &&
            config_.dense_tp_decode_replicated &&
            forwardPhaseAllowsDecodeTopology() &&
            total_tokens > 0 &&
            total_tokens <= max_decode_like_rows;
        if (!decode_like)
            return false;
        if (!hasDecodeReplicatedDenseWeightSource())
        {
            throw std::runtime_error(
                "[QwenGraphBase] Replicated attention-state decode is configured but dense decode layer bindings are missing: " +
                describeDecodeReplicatedDenseBindingState());
        }
        return true;
    }

    bool QwenGraphBase::forwardPhaseAllowsDecodeTopology() const noexcept
    {
        return !forward_execution_phase_.has_value() ||
               *forward_execution_phase_ ==
                   ForwardExecutionPhase::Decode;
    }

    bool QwenGraphBase::denseTPAllreduceEnabledForCurrentGraph() const
    {
        return !useDecodeReplicatedDenseWeights();
    }

    bool QwenGraphBase::attentionTPAllreduceEnabledForCurrentGraph() const
    {
        return !useReplicatedAttentionStateWeights();
    }

    bool QwenGraphBase::useColumnParallelLMHeadForGraph(
        TensorBase *logits_local) const
    {
        if (!config_.lm_head_column_parallel || logits_local == nullptr)
            return false;

        /*
         * LocalTP MTP can explicitly mirror the terminal LM head on every
         * participant.  In that mode MTP verifier/sidecar graphs produce the
         * same full-vocab rows as serial decode, and advertising a local shard
         * would reintroduce the tiny verifier logits collective this mode is
         * designed to remove.
         */
        if (useMirroredMTPHeadWeights())
            return false;

        /*
         * Dense decode-replicated graphs, including compact all-position
         * verifier rows, must project through the same full-vocab LM head that
         * ordinary serial decode uses.  Running the verifier through the primary
         * sharded LM head changes both the WeightBinding and the prepared GEMM
         * descriptor, so the hidden states can be identical while logits drift.
         */
        return denseTPAllreduceEnabledForCurrentGraph();
    }

    bool QwenGraphBase::needsDistributedLMHeadAllGather(
        bool column_parallel) const
    {
        return column_parallel &&
               mpi_ctx_ &&
               mpi_ctx_->world_size() > 1;
    }

    TensorBase *QwenGraphBase::graphLMHeadOutput(
        bool column_parallel) const
    {
        if (needsDistributedLMHeadAllGather(column_parallel))
            return buffers_.logits;
        return column_parallel ? buffers_.logits_local : buffers_.logits;
    }

    bool QwenGraphBase::needsPhaseSplitPrefillKVCacheHandoff(
        int total_tokens,
        IKVCache *kv_cache,
        DeviceId device) const
    {
        if (!kv_cache || !device.is_gpu())
            return false;
        if (!config_.dense_tp_enabled ||
            !config_.dense_tp_decode_replicated ||
            !hasDecodeReplicatedDenseWeightSource() ||
            !config_.qkv_column_parallel)
        {
            return false;
        }
        if (!mtp_kv_cache_only_graph_active_ &&
            (useReplicatedAttentionStateWeights() || useDecodeReplicatedDenseWeights()))
            return false;
        if (total_tokens <= 1)
            return false;
        if (config_.n_kv_heads <= 0 ||
            config_.head_dim <= 0 ||
            config_.local_n_kv_heads <= 0 ||
            config_.local_n_kv_heads >= config_.n_kv_heads)
        {
            return false;
        }
        return !kv_cache->is_sharded();
    }

    QwenGraphBase::DecodeReplicatedDenseScope::DecodeReplicatedDenseScope(
        QwenGraphBase &owner,
        int total_tokens)
        : owner_(owner),
          previous_(owner.decode_replicated_dense_graph_active_),
          previous_attention_(owner.replicated_attention_state_graph_active_),
          previous_embedding_(owner.decode_mirrored_embedding_graph_active_),
          previous_mtp_head_(owner.mtp_mirrored_lm_head_graph_active_)
    {
        owner_.decode_replicated_dense_graph_active_ =
            owner_.denseDecodeReplicatedActiveForTokens(total_tokens);
        owner_.decode_mirrored_embedding_graph_active_ =
            owner_.denseDecodeMirroredEmbeddingActiveForTokens(total_tokens);
        owner_.replicated_attention_state_graph_active_ =
            owner_.replicatedAttentionStateActiveForTokens(total_tokens);
        owner_.mtp_mirrored_lm_head_graph_active_ =
            owner_.mirroredMTPHeadActiveForProjectedRows(total_tokens);
    }

    QwenGraphBase::DecodeReplicatedDenseScope::~DecodeReplicatedDenseScope()
    {
        owner_.decode_replicated_dense_graph_active_ = previous_;
        owner_.replicated_attention_state_graph_active_ = previous_attention_;
        owner_.decode_mirrored_embedding_graph_active_ = previous_embedding_;
        owner_.mtp_mirrored_lm_head_graph_active_ = previous_mtp_head_;
    }

    QwenGraphBase::ForwardExecutionPhaseScope::ForwardExecutionPhaseScope(
        QwenGraphBase &owner,
        ForwardExecutionPhase phase)
        : owner_(owner),
          previous_(owner.forward_execution_phase_)
    {
        owner_.forward_execution_phase_ = phase;
    }

    QwenGraphBase::ForwardExecutionPhaseScope::~ForwardExecutionPhaseScope()
    {
        owner_.forward_execution_phase_ = previous_;
    }

    QwenGraphBase::MirroredMTPHeadScope::MirroredMTPHeadScope(
        QwenGraphBase &owner,
        bool active)
        : owner_(owner),
          previous_(owner.mtp_mirrored_lm_head_graph_active_)
    {
        owner_.mtp_mirrored_lm_head_graph_active_ = active;
    }

    QwenGraphBase::MirroredMTPHeadScope::~MirroredMTPHeadScope()
    {
        owner_.mtp_mirrored_lm_head_graph_active_ = previous_;
    }

    QwenGraphBase::MTPKVCacheOnlyScope::MTPKVCacheOnlyScope(
        QwenGraphBase &owner,
        bool active)
        : owner_(owner),
          previous_(owner.mtp_kv_cache_only_graph_active_)
    {
        owner_.mtp_kv_cache_only_graph_active_ = active;
    }

    QwenGraphBase::MTPKVCacheOnlyScope::~MTPKVCacheOnlyScope()
    {
        owner_.mtp_kv_cache_only_graph_active_ = previous_;
    }

    namespace
    {
        LayerWeightBindings mergeReplicatedAttentionBindings(
            LayerWeightBindings base,
            const LayerWeightBindings &replicated)
        {
            auto pick = [](const WeightBinding *override_binding,
                           const WeightBinding *base_binding)
            {
                return override_binding ? override_binding : base_binding;
            };

            base.wq = pick(replicated.wq, base.wq);
            base.wk = pick(replicated.wk, base.wk);
            base.wv = pick(replicated.wv, base.wv);
            base.wo = pick(replicated.wo, base.wo);
            base.q_bias = pick(replicated.q_bias, base.q_bias);
            base.k_bias = pick(replicated.k_bias, base.k_bias);
            base.v_bias = pick(replicated.v_bias, base.v_bias);
            base.attn_qkv = pick(replicated.attn_qkv, base.attn_qkv);
            base.attn_gate = pick(replicated.attn_gate, base.attn_gate);
            base.ssm_alpha = pick(replicated.ssm_alpha, base.ssm_alpha);
            base.ssm_beta = pick(replicated.ssm_beta, base.ssm_beta);
            base.ssm_conv1d = pick(replicated.ssm_conv1d, base.ssm_conv1d);
            base.ssm_dt_bias = pick(replicated.ssm_dt_bias, base.ssm_dt_bias);
            base.ssm_a = pick(replicated.ssm_a, base.ssm_a);
            base.ssm_norm = pick(replicated.ssm_norm, base.ssm_norm);
            base.ssm_out = pick(replicated.ssm_out, base.ssm_out);
            return base;
        }

        LayerWeightBindings mergeDenseDecodeBindings(
            LayerWeightBindings base,
            const LayerWeightBindings &decode_dense)
        {
            auto pick = [](const WeightBinding *override_binding,
                           const WeightBinding *base_binding)
            {
                return override_binding ? override_binding : base_binding;
            };

            base.wq = pick(decode_dense.wq, base.wq);
            base.wk = pick(decode_dense.wk, base.wk);
            base.wv = pick(decode_dense.wv, base.wv);
            base.wo = pick(decode_dense.wo, base.wo);
            base.q_bias = pick(decode_dense.q_bias, base.q_bias);
            base.k_bias = pick(decode_dense.k_bias, base.k_bias);
            base.v_bias = pick(decode_dense.v_bias, base.v_bias);
            base.attn_qkv = pick(decode_dense.attn_qkv, base.attn_qkv);
            base.attn_gate = pick(decode_dense.attn_gate, base.attn_gate);
            base.ssm_alpha = pick(decode_dense.ssm_alpha, base.ssm_alpha);
            base.ssm_beta = pick(decode_dense.ssm_beta, base.ssm_beta);
            base.ssm_conv1d = pick(decode_dense.ssm_conv1d, base.ssm_conv1d);
            base.ssm_dt_bias = pick(decode_dense.ssm_dt_bias, base.ssm_dt_bias);
            base.ssm_a = pick(decode_dense.ssm_a, base.ssm_a);
            base.ssm_norm = pick(decode_dense.ssm_norm, base.ssm_norm);
            base.ssm_out = pick(decode_dense.ssm_out, base.ssm_out);
            base.gate_proj = pick(decode_dense.gate_proj, base.gate_proj);
            base.up_proj = pick(decode_dense.up_proj, base.up_proj);
            base.down_proj = pick(decode_dense.down_proj, base.down_proj);
            base.moe_gate = pick(decode_dense.moe_gate, base.moe_gate);
            base.moe_gate_exps = pick(decode_dense.moe_gate_exps, base.moe_gate_exps);
            base.moe_up_exps = pick(decode_dense.moe_up_exps, base.moe_up_exps);
            base.moe_down_exps = pick(decode_dense.moe_down_exps, base.moe_down_exps);
            base.shared_expert_gate = pick(decode_dense.shared_expert_gate, base.shared_expert_gate);
            base.shared_expert_up = pick(decode_dense.shared_expert_up, base.shared_expert_up);
            base.shared_expert_down = pick(decode_dense.shared_expert_down, base.shared_expert_down);
            base.shared_expert_gate_inp = pick(decode_dense.shared_expert_gate_inp, base.shared_expert_gate_inp);

            return base;
        }
    }

    LayerWeightBindings QwenGraphBase::layerWeightBindingsForGraph(int layer_idx) const
    {
        LayerWeightBindings bindings;
        if (weight_bindings_.get_layer_weights)
            bindings = weight_bindings_.get_layer_weights(layer_idx);

        if (useReplicatedAttentionStateWeights() &&
            decode_replicated_dense_weight_bindings_.get_layer_weights)
        {
            bindings = mergeReplicatedAttentionBindings(
                bindings,
                decode_replicated_dense_weight_bindings_.get_layer_weights(layer_idx));
        }

        if (useDecodeReplicatedDenseWeights() &&
            decode_replicated_dense_weight_bindings_.get_layer_weights)
        {
            bindings = mergeDenseDecodeBindings(
                bindings,
                decode_replicated_dense_weight_bindings_.get_layer_weights(layer_idx));
        }

        return bindings;
    }

    LayerWeights QwenGraphBase::layerWeightsForGraph(int layer_idx) const
    {
        if (weight_bindings_.get_layer_weights)
            return toLegacyLayerWeights(layerWeightBindingsForGraph(layer_idx));
        if (weights_.get_layer_weights)
            return weights_.get_layer_weights(layer_idx);
        return {};
    }

    TensorBase *QwenGraphBase::modelEmbeddingTable() const
    {
        if (useFullVocabEmbeddingForCurrentGraph())
        {
            TensorBase *decode_bound = legacyTensor(decode_replicated_dense_weight_bindings_.embedding_table);
            if (decode_bound)
                return decode_bound;
            throw std::runtime_error(
                "[QwenGraphBase] Full-vocab decode embedding was requested but no replicated embedding binding is available");
        }
        TensorBase *bound = legacyTensor(weight_bindings_.embedding_table);
        return bound ? bound : weights_.embedding_table;
    }

    TensorBase *QwenGraphBase::modelFinalNorm() const
    {
        if (useDecodeReplicatedDenseWeights() || useMirroredMTPHeadWeights())
        {
            /*
             * A mirrored LocalTP MTP verifier head must consume the same final
             * normalization vector on every participant before the replicated
             * full-vocabulary projection.  Pairing a replicated LM head with a
             * TP-local final_norm lets otherwise identical mirrored children
             * produce different draft distributions, which breaks stochastic
             * MTP's single-token decode equivalence contract.
             */
            if (!decode_replicated_dense_weight_bindings_.final_norm)
            {
                throw std::runtime_error(
                    useMirroredMTPHeadWeights()
                        ? "[QwenGraphBase] Mirrored LocalTP MTP head was requested but no replicated final_norm binding is available"
                        : "[QwenGraphBase] Replicated dense decode was requested but no replicated final_norm binding is available");
            }
            TensorBase *decode_bound =
                legacyTensor(decode_replicated_dense_weight_bindings_.final_norm);
            if (decode_bound)
                return decode_bound;
            throw std::runtime_error(
                useMirroredMTPHeadWeights()
                    ? "[QwenGraphBase] Mirrored LocalTP MTP head final_norm binding has no tensor"
                    : "[QwenGraphBase] Replicated dense decode final_norm binding has no tensor");
        }
        TensorBase *bound = legacyTensor(weight_bindings_.final_norm);
        return bound ? bound : weights_.final_norm;
    }

    TensorBase *QwenGraphBase::modelLMHead() const
    {
        if (useDecodeReplicatedDenseWeights() || useMirroredMTPHeadWeights())
        {
            if (useMirroredMTPHeadWeights() && !hasMirroredMTPHeadWeightSource())
            {
                throw std::runtime_error(
                    "[QwenGraphBase] Mirrored LocalTP MTP head was requested but replicated final_norm/lm_head bindings are missing: " +
                    describeDecodeReplicatedDenseBindingState());
            }
            TensorBase *decode_bound = legacyTensor(decode_replicated_dense_weight_bindings_.lm_head);
            if (decode_bound)
                return decode_bound;
            throw std::runtime_error(
                "[QwenGraphBase] Replicated dense decode was requested but no replicated LM head binding is available");
        }
        TensorBase *bound = legacyTensor(weight_bindings_.lm_head);
        return bound ? bound : weights_.lm_head;
    }

    TensorBase *QwenGraphBase::modelLMHeadForGraph(bool column_parallel) const
    {
        if (column_parallel)
        {
            TensorBase *bound = legacyTensor(weight_bindings_.lm_head);
            if (bound)
                return bound;
            return weights_.lm_head;
        }
        return modelLMHead();
    }

    const WeightBinding *QwenGraphBase::modelEmbeddingBinding() const
    {
        if (useFullVocabEmbeddingForCurrentGraph())
        {
            if (!decode_replicated_dense_weight_bindings_.embedding_table)
                throw std::runtime_error(
                    "[QwenGraphBase] Full-vocab decode embedding was requested but no replicated embedding binding is available");
            return decode_replicated_dense_weight_bindings_.embedding_table;
        }
        return weight_bindings_.embedding_table;
    }

    const WeightBinding *QwenGraphBase::modelFinalNormBinding() const
    {
        if (useDecodeReplicatedDenseWeights() || useMirroredMTPHeadWeights())
        {
            if (!decode_replicated_dense_weight_bindings_.final_norm)
                throw std::runtime_error(
                    useMirroredMTPHeadWeights()
                        ? "[QwenGraphBase] Mirrored LocalTP MTP head was requested but no replicated final_norm binding is available"
                        : "[QwenGraphBase] Replicated dense decode was requested but no replicated final_norm binding is available");
            return decode_replicated_dense_weight_bindings_.final_norm;
        }
        return weight_bindings_.final_norm;
    }

    const WeightBinding *QwenGraphBase::modelLMHeadBinding() const
    {
        if (useDecodeReplicatedDenseWeights() || useMirroredMTPHeadWeights())
        {
            if (!decode_replicated_dense_weight_bindings_.lm_head)
                throw std::runtime_error(
                    useMirroredMTPHeadWeights()
                        ? "[QwenGraphBase] Mirrored LocalTP MTP head was requested but no replicated LM head binding is available"
                        : "[QwenGraphBase] Replicated dense decode was requested but no replicated LM head binding is available");
            return decode_replicated_dense_weight_bindings_.lm_head;
        }
        return weight_bindings_.lm_head;
    }

    const WeightBinding *QwenGraphBase::modelLMHeadBindingForGraph(
        bool column_parallel) const
    {
        if (column_parallel)
            return weight_bindings_.lm_head;
        return modelLMHeadBinding();
    }

    std::optional<PreparedWeightRef> QwenGraphBase::preparedRefForGraphWeight(
        const WeightBinding *binding,
        DeviceId device) const
    {
        // Model GEMM stages must be wired from store-owned frozen WeightBinding
        // ids.  A binding may still carry an old PreparedWeightRef from an
        // earlier loader/materialization pass, but graph stages resolve refs
        // through PreparedWeightStore at execution time. Returning a ref that
        // the store cannot resolve creates a graph that validates cleanly at
        // construction and then fails on the first replay. Treat the store as
        // the sole source of truth for graph-executable prepared weights.
        if (binding && prepared_weight_store_)
        {
            auto ref = prepared_weight_store_->preparedRefForBinding(binding->binding_id, device);
            if (ref.has_value())
                return ref;
        }
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
        lb.K_full_prefill = toBase(arena_->getTensor(BufferId::K_FULL_PREFILL));
        lb.V_full_prefill = toBase(arena_->getTensor(BufferId::V_FULL_PREFILL));
        lb.attn_output = toBase(arena_->getTensor(BufferId::ATTN_OUTPUT));
        lb.attn_proj = toBase(arena_->getTensor(BufferId::ATTN_PROJ));
        lb.workspace_scores = toBase(arena_->getTensor(BufferId::ATTN_SCORES_WORKSPACE));
        lb.workspace_context = toBase(arena_->getTensor(BufferId::ATTN_CONTEXT_WORKSPACE));
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

    HiddenStateRowsSelectStage::DeviceRowIndexSource
    QwenGraphBase::resolveLMHeadDeviceRowIndexSource(
        bool has_request_sequence_lengths) const
    {
        return has_request_sequence_lengths &&
                       config_.row_indexed_logits_selected_rows.empty()
                   ? HiddenStateRowsSelectStage::DeviceRowIndexSource::
                         RequestTerminalLengths
                   : HiddenStateRowsSelectStage::DeviceRowIndexSource::
                         WorkspaceBoundDeviceIndices;
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
        const int32_t *request_sequence_lengths_device,
        int request_row_stride,
        BufferId input_buffer_id) const
    {
        dependency_out = dependency_node;

        if (config_.compute_all_position_logits)
        {
            if (!config_.compute_row_indexed_logits)
                return final_norm_output;

            const int row_count = config_.row_indexed_logits_row_count;
            if (rowIndexedLogitsCoverFullPhysicalInput(
                    config_,
                    row_count,
                    total_tokens))
            {
                /*
                 * The verifier requests the complete physical activation in
                 * native order. Feed it directly to LM head: a row-copy stage
                 * and its device index publication would be pure overhead.
                 */
                return final_norm_output;
            }
            TensorBase *scratch_rows = buffers_.layer_buffers.get(BufferId::LM_HEAD_INPUT_ROWS);
            if (!scratch_rows)
            {
                LOG_ERROR("[QwenGraphBase] Row-indexed all-position logits require lm_head_input_rows scratch buffer");
                throw std::runtime_error("row-indexed all-position logits scratch buffer missing");
            }
            const int scratch_row_capacity =
                scratch_rows ? static_cast<int>(scratch_rows->rows()) : 0;
            if (row_count <= 0 ||
                row_count > total_tokens ||
                row_count > scratch_row_capacity)
            {
                LOG_ERROR("[QwenGraphBase] Row-indexed all-position logits require "
                          << "1..min(scratch_rows,total_tokens) rows, got "
                          << row_count << " for total_tokens=" << total_tokens
                          << " scratch_rows=" << scratch_row_capacity);
                throw std::runtime_error("invalid row-indexed all-position logits row count");
            }

            std::vector<int> selected_rows = resolveRowIndexedLogitRows(
                config_,
                row_count,
                total_tokens,
                "LM-head row-select");

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
                const bool has_explicit_verifier_rows =
                    !config_.row_indexed_logits_selected_rows.empty();
                row_params.device_row_index_source =
                    resolveLMHeadDeviceRowIndexSource(
                        request_sequence_lengths_device != nullptr);
                if (!has_explicit_verifier_rows &&
                    row_params.device_row_index_source ==
                        HiddenStateRowsSelectStage::DeviceRowIndexSource::
                            RequestTerminalLengths)
                {
                    /*
                     * Request-batched prefill owns real lengths in the arena.
                     * The row-select kernel derives terminal rows from that
                     * device array, so compact logits cannot accidentally read
                     * stale verifier metadata from an earlier transaction.
                     */
                    row_params.request_sequence_lengths_device =
                        request_sequence_lengths_device;
                    row_params.request_row_stride = request_row_stride;
                }
                else
                {
                    /*
                     * A grouped verifier carries one explicit source index per
                     * compact logit row. The request-length arena remains live
                     * during this forward, but it describes request terminals,
                     * not verifier query rows. Explicit verifier metadata must
                     * therefore take precedence whenever a row plan is present.
                     */
                    row_params.workspace_buffer_name =
                        MTPSpecDecodeWorkspaceBuffers::VERIFIER_LOGIT_ROWS;
                }
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

        // CPU replay owns its selected row directly. GPU replay derives the
        // row from the request-admission length array already consumed by the
        // rest of the captured prefill graph; no pinned host scalar may become
        // a second, independently mutable owner of the same logical length.
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
        if (device.is_gpu())
        {
            if (!request_sequence_lengths_device ||
                request_row_stride != total_tokens)
            {
                throw std::runtime_error(
                    "Bucketed GPU LM-head row selection requires the exact "
                    "device-resident request length and padded row stride");
            }
            row_params.selection_policy =
                HiddenStateRowSelectStage::SelectionPolicy::
                    DeviceResidentRequestLength;
            row_params.request_sequence_length_device =
                request_sequence_lengths_device;
        }
        else
        {
            row_params.selection_policy =
                HiddenStateRowSelectStage::SelectionPolicy::
                    DynamicDeviceScalar;
        }

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

    bool GraphConfig::requiresEagerGPUWorkspaceFamilyManifest() const noexcept
    {
        const bool phase_split_dense_graphs =
            dense_tp_enabled &&
            (dense_tp_decode_replicated ||
             dense_tp_decode_mirrored_embedding);
        const bool dynamic_moe_graphs =
            moe.rebalance_mode == MoERebalanceMode::DYNAMIC;
        return mtp.enabled || phase_split_dense_graphs || dynamic_moe_graphs;
    }

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
        const int total_tokens = input.batch_size * input.seq_len;
        ForwardExecutionPhaseScope execution_phase_scope(
            *this,
            input.execution_phase);
        DecodeReplicatedDenseScope decode_dense_scope(*this, total_tokens);

        /*
         * Preserve the complete declarative input contract while entering the
         * Qwen builder.  ForwardInput is already the graph system's shared
         * value type, so a field-by-field adapter is both unnecessary and
         * unsafe: adding a field such as position_policy can otherwise leave
         * Qwen on that field's default while the execution engine requested a
         * different policy.  Whole-value copying makes future input-policy
         * additions structurally impossible to omit at this boundary.
         */
        ForwardInput qwen_input = input;

        // Adapt generic ForwardOutput to ForwardOutput
        ForwardOutput qwen_output;
        qwen_output.logits = output.logits;
        qwen_output.hidden = output.hidden;

        /*
         * Prefix runtime rehydration changes graph topology: every MoE layer
         * prepends one captured payload-transfer transaction. Keep that policy
         * in a scoped graph-build value so derived declarative FFN builders can
         * attach the transaction without consulting mutable host runtime state.
         */
        const bool previous_rehydration_scope =
            prefix_runtime_rehydration_graph_active_;
        prefix_runtime_rehydration_graph_active_ =
            qwen_input.rehydrate_prefix_runtime_on_device;
        ComputeGraph graph;
        try
        {
            graph = buildFullForwardGraph(qwen_input, qwen_output);
            prefix_runtime_rehydration_graph_active_ =
                previous_rehydration_scope;
        }
        catch (...)
        {
            prefix_runtime_rehydration_graph_active_ =
                previous_rehydration_scope;
            throw;
        }

        // Copy back output pointers (in case they were set by the builder)
        output.logits = qwen_output.logits;
        output.hidden = qwen_output.hidden;

        return graph;
    }

    ComputeGraph QwenGraphBase::buildLayerGraph(const LayerContext &ctx)
    {
        DecodeReplicatedDenseScope decode_dense_scope(*this, ctx.seq_len * ctx.batch_size);

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
            ctx.sequence_lengths, ctx.position_ids_device,
            ctx.sequence_lengths_device);

        // Build FFN graph
        ComputeGraph ffn_graph = buildFFNGraph(
            layer_weights, buffers_.layer_buffers, ctx.layer_idx, ctx.seq_len,
            ctx.batch_size, ctx.device,
            ctx.device_state_publication_stream,
            ctx.sequence_lengths_device,
            static_cast<const int32_t *>(ctx.position_ids_device));

        // Merge: attention -> FFN
        std::string attn_last = attn_graph.terminalNode();
        attn_graph.merge(std::move(ffn_graph), attn_last);

        return attn_graph;
    }

    // =============================================================================
    // Model-Level Graph Building
    // =============================================================================

    std::string QwenGraphBase::addMTPVerifierOutcomeToGraph(
        ComputeGraph &graph,
        const std::string &dependency_node,
        TensorBase *logits,
        int verifier_row_count,
        DeviceId device) const
    {
        const MTPVerifierOutcomeGraphMode mode =
            config_.mtp_verifier_outcome_graph_mode;
        if (mode == MTPVerifierOutcomeGraphMode::Disabled)
            return dependency_node;

        if (!config_.grouped_mtp_verifier ||
            !config_.compute_all_position_logits ||
            !config_.compute_row_indexed_logits)
        {
            throw std::runtime_error(
                "MTP verifier outcome graph mode requires the explicit grouped, "
                "all-position, row-indexed verifier policy");
        }
        if (!device.is_gpu())
        {
            throw std::runtime_error(
                "Graph-owned MTP verifier outcomes are GPU-only");
        }
        if (mode != MTPVerifierOutcomeGraphMode::Greedy)
        {
            throw std::runtime_error(
                "Requested MTP verifier outcome graph mode is not implemented");
        }
        if (!logits || verifier_row_count <= 0 ||
            verifier_row_count != config_.row_indexed_logits_row_count ||
            !config_.mtp_verifier_outcome_graph_binding.validForGreedy())
        {
            throw std::runtime_error(
                "Greedy MTP verifier outcome graph has malformed geometry or "
                "persistent device bindings");
        }

        if (config_.mtp_verifier_outcome_ownership !=
            MTPVerifierOutcomeOwnershipPolicy::ParticipantLocal)
        {
            throw std::runtime_error(
                "Graph-owned MTP verifier outcomes require an explicit "
                "participant-local ownership policy");
        }

        const bool mirrored_local_tp =
            config_.tp_ctx &&
            config_.tp_ctx->isLocal() &&
            config_.tp_ctx->degree() > 1;
        if (mirrored_local_tp &&
            !mtpTerminalHeadIsMirrored(config_.mtp.terminal_head_policy))
        {
            throw std::runtime_error(
                "Participant-local LocalTP MTP outcomes require a mirrored "
                "full verifier head on every participant");
        }
        if (config_.tp_ctx && !config_.tp_ctx->isLocal() &&
            config_.tp_ctx->degree() > 1)
        {
            throw std::runtime_error(
                "Participant-local compact MTP outcomes require a complete "
                "local verifier head and cannot be lowered onto sharded GlobalTP");
        }

        MTPVerifierOutcomeStage::Params params{
            .device_id = device,
            .logits = logits,
            .mode = mode,
            .binding = config_.mtp_verifier_outcome_graph_binding,
            .verifier_row_count = verifier_row_count,
            .vocab_size = config_.vocab_size,
            .ownership_policy = config_.mtp_verifier_outcome_ownership,
            .mirrored_local_tp = mirrored_local_tp,
            .stage_name = "mtp_verifier_outcome",
        };
        graph.addNode(
            params.stage_name,
            ComputeStageFactory::createMTPVerifierOutcome(params),
            device);
        graph.addDependency(params.stage_name, dependency_node);
        return params.stage_name;
    }

    std::string QwenGraphBase::maybeAddShiftedMTPPrefillTransaction(
        ComputeGraph &graph,
        const ForwardInput &input,
        const std::string &dependency_node,
        DeviceId device)
    {
        if (!input.shifted_mtp_prefill.has_value())
            return dependency_node;

        const auto &binding = *input.shifted_mtp_prefill;
        const int total_tokens = input.batch_size * input.seq_len;
        if (!device.is_gpu() ||
            input.execution_role != ForwardExecutionRole::MainInference ||
            input.execution_phase != ForwardExecutionPhase::Prefill ||
            total_tokens <= 0 ||
            !binding.validForRequestCount(input.batch_size) ||
            !input.sequence_lengths_device ||
            config_.mtp_shifted_prefill_hidden_publication !=
                MTPShiftedPrefillHiddenPublicationPolicy::
                    GraphIntegratedKVTransaction)
        {
            throw std::runtime_error(
                "Qwen GPU shifted MTP prefill requires a complete graph-integrated main-prefill binding");
        }
        if (weight_bindings_.mtp.empty() ||
            weight_bindings_.mtp.depths.empty())
        {
            throw std::runtime_error(
                "Qwen GPU shifted MTP prefill requires depth-zero MTP weight bindings");
        }
        if (!buffers_.current_hidden || !buffers_.layer_buffers.normalized ||
            buffers_.current_hidden->rows() <
                static_cast<size_t>(total_tokens) ||
            buffers_.layer_buffers.normalized->rows() <
                static_cast<size_t>(total_tokens))
        {
            throw std::runtime_error(
                "Qwen GPU shifted MTP prefill exceeds its persistent hidden-state arena capacity");
        }

        const std::string prepare_node =
            "mtp_shifted_prefill_prepare_and_archive";
        HiddenStateRowsSelectStage::Params prepare_params{
            .device_id = device,
            .input = buffers_.current_hidden,
            .output = buffers_.layer_buffers.normalized,
            .seq_len = total_tokens,
            .d_model = config_.d_model,
            .selected_row_count = total_tokens,
            .input_buffer_id = BufferId::HIDDEN_STATE,
            .output_buffer_id = BufferId::NORMALIZED,
            .device_row_index_source =
                HiddenStateRowsSelectStage::DeviceRowIndexSource::
                    ShiftedPrefillTransaction,
            .request_sequence_lengths_device =
                input.sequence_lengths_device,
            .request_row_stride_source =
                HiddenStateRowsSelectStage::RequestRowStrideSource::
                    ExternalDeviceScalar,
            .request_row_stride_device =
                binding.request_row_stride_device,
            .input_token_ids_device =
                binding.request_token_ids_device,
            .input_position_ids_device =
                binding.request_position_ids_device,
            .shifted_token_ids_output_device =
                binding.shifted_token_ids_device,
            .shifted_position_ids_output_device =
                binding.shifted_position_ids_device,
            .shifted_append_lengths_output_device =
                binding.append_lengths_device,
            .terminal_hidden_archive =
                binding.terminal_hidden_archive,
            .main_cached_tokens_by_request =
                binding.main_cached_tokens_device,
            .shifted_cached_tokens_by_request =
                binding.shifted_cached_tokens_device,
            .request_count = input.batch_size,
            .terminal_hidden_archive_buffer_id =
                BufferId::PREFIX_TERMINAL_HIDDEN,
        };
        graph.addNode(
            prepare_node,
            ComputeStageFactory::createHiddenStateRowsSelect(
                std::move(prepare_params)),
            device);
        graph.addDependency(prepare_node, dependency_node);

        MTPForwardInput mtp_input{
            .draft_token_ids = nullptr,
            .draft_token_ids_device =
                binding.shifted_token_ids_device,
            .terminal_hidden = buffers_.layer_buffers.normalized,
            .kv_cache = binding.kv_cache,
            .position_ids = nullptr,
            .position_ids_device =
                binding.shifted_position_ids_device,
            .sequence_lengths = nullptr,
            .sequence_lengths_device = binding.append_lengths_device,
            .batch_size = input.batch_size,
            .seq_len = input.seq_len,
            .first_sequence_index = 0,
            .device_state_publication_stream =
                input.device_state_publication_stream,
            .device = device,
            .terminal_hidden_buffer_id = BufferId::NORMALIZED,
            .kv_cache_only = true,
        };
        MTPForwardOutput mtp_output = binding.output;
        ComputeGraph mtp_kv_graph = buildMTPGraph(
            /*depth_idx=*/0,
            weight_bindings_.mtp.depths.front(),
            mtp_input,
            mtp_output);
        if (mtp_kv_graph.size() == 0 ||
            mtp_kv_graph.terminalNode().empty())
        {
            throw std::runtime_error(
                "Qwen GPU shifted MTP prefill produced an empty depth-zero KV graph");
        }
        const std::string transaction_terminal =
            mtp_kv_graph.terminalNode();
        graph.merge(std::move(mtp_kv_graph), prepare_node);
        graph.setTerminalNode(transaction_terminal);
        return transaction_terminal;
    }

    ComputeGraph QwenGraphBase::buildFullForwardGraph(
        const ForwardInput &input,
        ForwardOutput &output)
    {
        const int total_tokens = input.batch_size * input.seq_len;
        ForwardExecutionPhaseScope execution_phase_scope(
            *this,
            input.execution_phase);
        DecodeReplicatedDenseScope decode_dense_scope(*this, total_tokens);

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
        embed_params.vocab_offset = embeddingVocabOffsetForCurrentGraph(config_.default_device);
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
        const bool embedding_is_sharded = usesVocabParallelEmbeddingForCurrentGraph();
        bool embedding_allreduce_added = false;
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
                embedding_allreduce_added = true;
            }
        }

        // -------------------------------------------------------------------------
        // Stage 2: Transformer Layers (complete graphs, not placeholders)
        // -------------------------------------------------------------------------
        std::string prev_node = embedding_allreduce_added
                                    ? "embedding_allreduce"
                                    : "embedding";
        prev_node = maybeAddEmbeddingDiagnosticCheckpoints(
            graph,
            embed_output,
            prev_node,
            total_tokens,
            device);

        const int *position_ids =
            validatedHostPositionRows(input, "Qwen standard forward graph");

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
                input.sequence_lengths, input.position_ids_device,
                input.sequence_lengths_device);

            // Get the terminal node of attention sub-graph
            std::string attn_last = attn_graph.terminalNode();
            if (attn_last.empty())
                attn_last = prev_node;

            // Merge attention graph, connecting to previous node
            graph.merge(std::move(attn_graph), prev_node);

            // Build FFN graph for this layer
            ComputeGraph ffn_graph = buildFFNGraph(
                layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                input.batch_size, device,
                input.device_state_publication_stream,
                input.sequence_lengths_device,
                static_cast<const int32_t *>(input.position_ids_device));

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
        const BufferId final_norm_input_id =
            (final_norm_input == buffers_.layer_buffers.residual)
                ? BufferId::RESIDUAL
                : BufferId::HIDDEN_STATE;

        prev_node = addFinalNormToGraph(
            graph,
            final_norm_input,
            buffers_.layer_buffers.normalized,
            prev_node,
            total_tokens,
            device,
            final_norm_input_id,
            input.sequence_lengths_device);

        std::string lm_head_dependency = prev_node;
        TensorBase *lm_head_input = maybeAddLMHeadRowSelect(
            graph,
            prev_node,
            buffers_.layer_buffers.normalized,
            total_tokens,
            input.real_seq_len,
            input.bucket_seq_len,
            device,
            lm_head_dependency,
            input.sequence_lengths_device,
            input.seq_len);

        // -------------------------------------------------------------------------
        // Stage 4: LM Head (with optional Column-Parallel + AllGather)
        // -------------------------------------------------------------------------
        // When lm_head_column_parallel is enabled:
        // - LM head weight is sharded by vocab: [vocab_local, d_model]
        // - LM head outputs to buffers_.logits_local: [seq_len, vocab_local]
        // - AllGather collects to buffers_.logits: [seq_len, vocab_size]
        // -------------------------------------------------------------------------
        bool use_column_parallel =
            useColumnParallelLMHeadForGraph(buffers_.logits_local);

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
        lm_params.lm_head_weight = modelLMHeadForGraph(use_column_parallel);
        lm_params.prepared_ref = preparedRefForGraphWeight(
            modelLMHeadBindingForGraph(use_column_parallel), device);
        lm_params.logits = lm_head_output;
        lm_params.seq_len = lm_layout.seq_len;
        lm_params.d_model = config_.d_model;
        lm_params.vocab_size = lm_head_vocab_size;
        lm_params.serial_equivalent_partition_width =
            serialEquivalentLMHeadPartitionWidth(use_column_parallel);
        lm_params.bias_tensor = nullptr; // Qwen2 has no LM head bias
        lm_params.device_id = config_.default_device;
        lm_params.prepared_store = prepared_weight_store_;
        lm_params.input_buffer_id = lm_layout.input_buffer_id;
        lm_params.output_buffer_id = logitsBufferId(
            use_column_parallel,
            lm_layout.compute_all_positions);
        lm_params.use_prefill_replay_row_offset = lm_layout.use_prefill_replay_row_offset;
        lm_params.compute_all_positions = lm_layout.compute_all_positions;
        lm_params.force_decode_equivalent_verifier_prefill =
            shouldForceDecodeEquivalentLMHeadVerifierPrefill(
                config_,
                device,
                lm_layout);

        graph.addNode("lm_head",
                      ComputeStageFactory::createLMHead(lm_params),
                      device);
        graph.addDependency("lm_head", lm_head_dependency);
        prev_node = "lm_head";

        // Phase 5: AllGather stage for column-parallel LM head
        if (needsDistributedLMHeadAllGather(use_column_parallel))
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
            prev_node = "lm_head_allgather";
        }

        prev_node = addMTPVerifierOutcomeToGraph(
            graph,
            prev_node,
            graphLMHeadOutput(use_column_parallel),
            lm_layout.seq_len,
            device);

        prev_node = maybeAddShiftedMTPPrefillTransaction(
            graph,
            input,
            prev_node,
            device);
        graph.setTerminalNode(prev_node);

        // Set output
        output.logits = graphLMHeadOutput(use_column_parallel);

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
        const int total_tokens = input.batch_size * input.seq_len;
        DecodeReplicatedDenseScope decode_dense_scope(*this, total_tokens);

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
            embed_params.vocab_offset = embeddingVocabOffsetForCurrentGraph(config_.default_device);
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
            const bool embedding_is_sharded = usesVocabParallelEmbeddingForCurrentGraph();
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
            // PP middle/final stage: the execution engine has already handed the
            // external activation to this graph's working buffer through
            // TransferEngine. Graph construction only declares the layer DAG;
            // it must never materialize or move tensor contents itself.
            // -----------------------------------------------------------------
            TensorBase *working_buffer = (buffers_.layer_buffers.residual && config_.isHybridQ16())
                                             ? buffers_.layer_buffers.residual
                                             : buffers_.current_hidden;

            if (input.external_hidden_state != working_buffer)
            {
                LOG_TRACE(
                    "[QwenGraphBase] PP external activation was prepared in the "
                    "graph working buffer before declarative graph construction");
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
        const int *position_ids =
            validatedHostPositionRows(input, "Qwen partial pipeline forward graph");

        // Build graphs for assigned layer range [first_layer, last_layer)
        for (int layer = first_layer; layer < last_layer; ++layer)
        {
            // Get layer weights
            LayerWeights layer_weights = layerWeightsForGraph(layer);

            // Build attention graph for this layer
            ComputeGraph attn_graph = buildAttentionGraph(
                layer_weights, buffers_.layer_buffers, layer, input.seq_len,
                input.batch_size, input.kv_cache, position_ids, device,
                input.sequence_lengths, input.position_ids_device,
                input.sequence_lengths_device);

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
                input.batch_size, device,
                input.device_state_publication_stream,
                input.sequence_lengths_device,
                static_cast<const int32_t *>(input.position_ids_device));

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
            const BufferId final_norm_input_id =
                (final_norm_input == buffers_.layer_buffers.residual)
                    ? BufferId::RESIDUAL
                    : BufferId::HIDDEN_STATE;

            prev_node = addFinalNormToGraph(
                graph,
                final_norm_input,
                buffers_.layer_buffers.normalized,
                prev_node,
                total_tokens,
                device,
                final_norm_input_id,
                input.sequence_lengths_device);

            std::string lm_head_dependency = prev_node;
            TensorBase *lm_head_input = maybeAddLMHeadRowSelect(
                graph,
                prev_node,
                buffers_.layer_buffers.normalized,
                total_tokens,
                input.real_seq_len,
                input.bucket_seq_len,
                device,
                lm_head_dependency,
                input.sequence_lengths_device,
                input.seq_len);

            // LM Head (with optional Column-Parallel + AllGather)
            bool use_column_parallel =
                useColumnParallelLMHeadForGraph(buffers_.logits_local);

            TensorBase *lm_head_output = use_column_parallel ? buffers_.logits_local : buffers_.logits;
            int lm_head_vocab_size = use_column_parallel ? config_.vocab_local : config_.vocab_size;

            LOG_DEBUG("[QwenGraphBase] LM head in buildPartialForwardGraph: use_column_parallel="
                      << use_column_parallel << " lm_head_vocab_size=" << lm_head_vocab_size);

            const LMHeadInputLayout lm_layout =
                describeLMHeadInputLayout(config_, buffers_.layer_buffers, lm_head_input, total_tokens);

            LMHeadStage::Params lm_params;
            lm_params.hidden_states = lm_head_input;
            lm_params.lm_head_weight = modelLMHeadForGraph(use_column_parallel);
            lm_params.prepared_ref = preparedRefForGraphWeight(
                modelLMHeadBindingForGraph(use_column_parallel), device);
            lm_params.logits = lm_head_output;
            lm_params.seq_len = lm_layout.seq_len;
            lm_params.d_model = config_.d_model;
            lm_params.vocab_size = lm_head_vocab_size;
            lm_params.serial_equivalent_partition_width =
                serialEquivalentLMHeadPartitionWidth(use_column_parallel);
            lm_params.bias_tensor = nullptr;
            lm_params.device_id = config_.default_device;
            lm_params.prepared_store = prepared_weight_store_;
            lm_params.input_buffer_id = lm_layout.input_buffer_id;
            lm_params.output_buffer_id = logitsBufferId(
                use_column_parallel,
                lm_layout.compute_all_positions);
            lm_params.use_prefill_replay_row_offset = lm_layout.use_prefill_replay_row_offset;
            lm_params.compute_all_positions = lm_layout.compute_all_positions;
            lm_params.force_decode_equivalent_verifier_prefill =
                shouldForceDecodeEquivalentLMHeadVerifierPrefill(
                    config_,
                    device,
                    lm_layout);

            graph.addNode("lm_head",
                          ComputeStageFactory::createLMHead(lm_params),
                          device);
            graph.addDependency("lm_head", lm_head_dependency);
            prev_node = "lm_head";

            // AllGather stage for column-parallel LM head
            if (needsDistributedLMHeadAllGather(use_column_parallel))
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
                prev_node = "lm_head_allgather";
            }

            prev_node = addMTPVerifierOutcomeToGraph(
                graph,
                prev_node,
                graphLMHeadOutput(use_column_parallel),
                lm_layout.seq_len,
                device);

            // Set output logits
            output.logits = graphLMHeadOutput(use_column_parallel);
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

        const int *position_ids =
            validatedHostPositionRows(input, "Qwen unified pipeline forward graph");

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
                embed_params.vocab_offset = embeddingVocabOffsetForCurrentGraph(stage_device);
                embed_params.local_vocab_size = modelEmbeddingTable() ? static_cast<int>(modelEmbeddingTable()->rows()) : 0;
                embed_params.device_id = stage_device;
                embed_params.mpi_ctx = mpi_ctx_.get();
                embed_params.prepared_ref = preparedRefForGraphWeight(modelEmbeddingBinding(), stage_device);
                embed_params.prepared_store = prepared_weight_store_;

                graph.addNode("embedding",
                              ComputeStageFactory::createEmbedding(embed_params),
                              stage_device);
                prev_node = "embedding";

                if (usesVocabParallelEmbeddingForCurrentGraph() && needsTPAllreduce())
                {
                    const size_t allreduce_count = static_cast<size_t>(total_tokens) * config_.d_model;
                    auto allreduce_stage = createTPAllreduceStage(
                        embed_output,
                        allreduce_count,
                        stage_device,
                        -1,
                        /*is_attention=*/false,
                        "embedding_allreduce",
                        BufferId::HIDDEN_STATE);
                    if (allreduce_stage)
                    {
                        graph.addNode("embedding_allreduce", std::move(allreduce_stage), stage_device);
                        graph.addDependency("embedding_allreduce", "embedding");
                        prev_node = "embedding_allreduce";
                    }
                }

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
                    input.sequence_lengths, input.position_ids_device,
                    input.sequence_lengths_device);

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
                    input.batch_size,
                    stage_device,
                    input.device_state_publication_stream,
                    input.sequence_lengths_device,
                    static_cast<const int32_t *>(
                        input.position_ids_device));

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
                const BufferId final_norm_input_id =
                    (final_norm_input == buffers_.layer_buffers.residual)
                        ? BufferId::RESIDUAL
                        : BufferId::HIDDEN_STATE;

                prev_node = addFinalNormToGraph(
                    graph,
                    final_norm_input,
                    buffers_.layer_buffers.normalized,
                    prev_node,
                    total_tokens,
                    stage_device,
                    final_norm_input_id,
                    input.sequence_lengths_device);

                std::string lm_head_dependency = prev_node;
                TensorBase *lm_head_input = maybeAddLMHeadRowSelect(
                    graph,
                    prev_node,
                    buffers_.layer_buffers.normalized,
                    total_tokens,
                    input.real_seq_len,
                    input.bucket_seq_len,
                    stage_device,
                    lm_head_dependency,
                    input.sequence_lengths_device,
                    input.seq_len);

                // LM Head
                bool use_column_parallel = useColumnParallelLMHeadForGraph(buffers_.logits_local);
                TensorBase *lm_head_output = use_column_parallel ? buffers_.logits_local : buffers_.logits;
                int lm_head_vocab_size = use_column_parallel ? config_.vocab_local : config_.vocab_size;

                LOG_DEBUG("[QwenGraphBase] LM head in unified PP: use_column_parallel="
                          << use_column_parallel << " vocab_size=" << lm_head_vocab_size);

                const LMHeadInputLayout lm_layout =
                    describeLMHeadInputLayout(config_, buffers_.layer_buffers, lm_head_input, total_tokens);

                LMHeadStage::Params lm_params;
                lm_params.hidden_states = lm_head_input;
                lm_params.lm_head_weight = modelLMHeadForGraph(use_column_parallel);
                lm_params.prepared_ref = preparedRefForGraphWeight(
                    modelLMHeadBindingForGraph(use_column_parallel), stage_device);
                lm_params.logits = lm_head_output;
                lm_params.seq_len = lm_layout.seq_len;
                lm_params.d_model = config_.d_model;
                lm_params.vocab_size = lm_head_vocab_size;
                lm_params.serial_equivalent_partition_width =
                    serialEquivalentLMHeadPartitionWidth(use_column_parallel);
                lm_params.bias_tensor = nullptr;
                lm_params.device_id = stage_device;
                lm_params.prepared_store = prepared_weight_store_;
                lm_params.input_buffer_id = lm_layout.input_buffer_id;
                lm_params.output_buffer_id = logitsBufferId(
                    use_column_parallel,
                    lm_layout.compute_all_positions);
                lm_params.use_prefill_replay_row_offset = lm_layout.use_prefill_replay_row_offset;
                lm_params.compute_all_positions = lm_layout.compute_all_positions;
                lm_params.force_decode_equivalent_verifier_prefill =
                    shouldForceDecodeEquivalentLMHeadVerifierPrefill(
                        config_,
                        stage_device,
                        lm_layout);

                graph.addNode("lm_head",
                              ComputeStageFactory::createLMHead(lm_params),
                              stage_device);
                graph.addDependency("lm_head", lm_head_dependency);
                prev_node = "lm_head";

                // AllGather stage for column-parallel LM head
                if (needsDistributedLMHeadAllGather(use_column_parallel))
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

                output.logits = graphLMHeadOutput(use_column_parallel);

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
        const int total_tokens = input.batch_size * input.seq_len;
        DecodeReplicatedDenseScope decode_dense_scope(*this, total_tokens);

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

        // KV cache topology is declarative. The admitted request range owns
        // the graph-planning cursor; GPU cache contents remain device-owned and
        // must never be observed while constructing a graph.
        config.has_kv_cache = (input.kv_cache != nullptr);
        if (config.has_kv_cache)
        {
            config.cached_tokens = std::max(
                0,
                input.token_offset != 0
                    ? input.token_offset
                    : input.position_offset);
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
        output.logits = graphLMHeadOutput(
            useColumnParallelLMHeadForGraph(buffers_.logits_local));

        LOG_DEBUG("[QwenGraphBase] Built schema-based forward graph with "
                  << graph.size() << " nodes");

        return graph;
    }

    ComputeGraph QwenGraphBase::buildEmbeddingGraph(
        const ForwardInput &input,
        TensorBase *output_hidden)
    {
        const int total_tokens = input.batch_size * input.seq_len;
        DecodeReplicatedDenseScope decode_dense_scope(*this, total_tokens);

        LOG_DEBUG("[QwenGraphBase] Building embedding graph for "
                  << total_tokens << " tokens");

        ComputeGraph graph;

        EmbeddingStage::Params params;
        params.embed_table = modelEmbeddingTable();
        params.token_ids = input.token_ids;
        params.token_ids_device = input.token_ids_device;
        params.output = output_hidden;
        params.num_tokens = total_tokens;
        params.d_model = config_.d_model;
        params.vocab_size = config_.vocab_size;
        params.vocab_offset = embeddingVocabOffsetForCurrentGraph(config_.default_device);
        params.local_vocab_size = modelEmbeddingTable() ? static_cast<int>(modelEmbeddingTable()->rows()) : 0;
        params.device_id = config_.default_device;
        params.prepared_ref = preparedRefForGraphWeight(modelEmbeddingBinding(), config_.default_device);
        params.prepared_store = prepared_weight_store_;

        graph.addNode("embedding",
                      ComputeStageFactory::createEmbedding(params),
                      config_.default_device);

        if (usesVocabParallelEmbeddingForCurrentGraph() && needsTPAllreduce())
        {
            const size_t allreduce_count = static_cast<size_t>(total_tokens) * config_.d_model;
            auto allreduce_stage = createTPAllreduceStage(
                output_hidden,
                allreduce_count,
                config_.default_device,
                -1,
                /*is_attention=*/false,
                "embedding_allreduce",
                BufferId::HIDDEN_STATE);
            if (allreduce_stage)
            {
                graph.addNode("embedding_allreduce", std::move(allreduce_stage), config_.default_device);
                graph.addDependency("embedding_allreduce", "embedding");
            }
        }

        return graph;
    }

    ComputeGraph QwenGraphBase::buildTransformerLayersGraph(
        TensorBase *input_hidden,
        IKVCache *kv_cache,
        const int *position_ids,
        const void *position_ids_device,
        DeviceId device)
    {
        (void)position_ids_device;
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
        const void *position_ids_device,
        DeviceId device)
    {
        (void)position_ids_device;
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
        DecodeReplicatedDenseScope decode_dense_scope(*this, total_tokens);

        LOG_DEBUG("[QwenGraphBase] Building LM head graph for " << total_tokens << " tokens"
                                                                << " lm_head_column_parallel=" << config_.lm_head_column_parallel
                                                                << " vocab_local=" << config_.vocab_local);

        ComputeGraph graph;
        const FinalProjectionPolicy final_projection =
            resolveFinalProjectionPolicy({
                .norm_source = FinalNormSource::ModelOutputNorm,
                .mtp_norm = nullptr,
                .full_vocab_output = output_logits,
                .column_parallel_output = logits_local,
                .total_tokens = total_tokens,
                .force_full_vocabulary_head = false,
                .compute_all_positions =
                    config_.compute_all_position_logits ||
                    config_.live_mtp_request_batch_condition,
            });

        // Final RMSNorm
        RMSNormStage::Params norm_params;
        norm_params.input = hidden_states;
        norm_params.output = hidden_states; // In-place norm
        norm_params.gamma = final_projection.norm_gamma;
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

        // Determine output buffer and vocab size for LM head stage
        const bool use_column_parallel = final_projection.column_parallel;
        TensorBase *lm_head_output = final_projection.lm_head_output;
        int lm_head_vocab_size = final_projection.lm_head_vocab_size;

        LOG_DEBUG("[QwenGraphBase] LM head: use_column_parallel=" << use_column_parallel
                                                                  << " lm_head_vocab_size=" << lm_head_vocab_size
                                                                  << " lm_head_output=" << lm_head_output);

        TensorBase *lm_head_input = hidden_states;
        std::string lm_head_dependency = "final_norm";
        int lm_head_seq_len = total_tokens;
        BufferId lm_head_input_buffer_id = BufferId::HIDDEN_STATE;
        bool lm_head_compute_all_positions =
            config_.compute_all_position_logits ||
            config_.live_mtp_request_batch_condition;
        bool lm_head_use_prefill_row_offset = true;

        if (config_.compute_all_position_logits && config_.compute_row_indexed_logits)
        {
            const int row_count = config_.row_indexed_logits_row_count;
            if (rowIndexedLogitsCoverFullPhysicalInput(
                    config_,
                    row_count,
                    total_tokens))
            {
                /*
                 * Standalone verifier LM-head graphs obey the same layout
                 * contract as full model graphs. The existing dense defaults
                 * below already project every row from HIDDEN_STATE.
                 */
                lm_head_seq_len = total_tokens;
                lm_head_compute_all_positions = true;
                lm_head_use_prefill_row_offset = false;
            }
            else
            {
                TensorBase *scratch_rows =
                    buffers_.layer_buffers.get(BufferId::LM_HEAD_INPUT_ROWS);
                if (!scratch_rows)
                {
                    LOG_ERROR("[QwenGraphBase] Standalone row-indexed verifier requires lm_head_input_rows scratch buffer");
                    throw std::runtime_error("standalone row-indexed verifier scratch buffer missing");
                }
                const int scratch_row_capacity =
                    scratch_rows ? static_cast<int>(scratch_rows->rows()) : 0;
                if (row_count <= 0 ||
                    row_count > total_tokens ||
                    row_count > scratch_row_capacity)
                {
                    LOG_ERROR("[QwenGraphBase] Standalone LM-head graph row-indexed verifier requires "
                              << "1..min(scratch_rows,total_tokens) rows, got "
                              << row_count << " for total_tokens=" << total_tokens
                              << " scratch_rows=" << scratch_row_capacity);
                    throw std::runtime_error("invalid standalone row-indexed all-position logits row count");
                }

                std::vector<int> selected_rows = resolveRowIndexedLogitRows(
                    config_,
                    row_count,
                    total_tokens,
                    "standalone LM-head row-select");

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
                    row_params.device_row_index_source =
                        HiddenStateRowsSelectStage::DeviceRowIndexSource::
                            WorkspaceBoundDeviceIndices;
                    row_params.workspace_buffer_name =
                        MTPSpecDecodeWorkspaceBuffers::VERIFIER_LOGIT_ROWS;
                }

                graph.addNode(
                    "lm_head_rows_select",
                    ComputeStageFactory::createHiddenStateRowsSelect(row_params),
                    device);
                graph.addDependency(
                    "lm_head_rows_select",
                    lm_head_dependency);

                /*
                 * Compact verifier rows are a new dense matrix. The LM head
                 * projects every compact row starting at row zero.
                 */
                lm_head_input = scratch_rows;
                lm_head_dependency = "lm_head_rows_select";
                lm_head_seq_len = row_count;
                lm_head_input_buffer_id = BufferId::LM_HEAD_INPUT_ROWS;
                lm_head_compute_all_positions = true;
                lm_head_use_prefill_row_offset = false;
            }
        }

        // LM Head projection
        LMHeadStage::Params lm_params;
        lm_params.hidden_states = lm_head_input;
        lm_params.lm_head_weight = final_projection.lm_head_weight;
        lm_params.prepared_ref = preparedRefForGraphWeight(
            final_projection.lm_head_binding, device);
        lm_params.logits = lm_head_output;
        lm_params.seq_len = lm_head_seq_len;
        lm_params.d_model = config_.d_model;
        lm_params.vocab_size = lm_head_vocab_size;
        lm_params.serial_equivalent_partition_width =
            final_projection.serial_equivalent_partition_width;
        lm_params.bias_tensor = nullptr;
        lm_params.device_id = device;
        lm_params.prepared_store = prepared_weight_store_;
        lm_params.input_buffer_id = lm_head_input_buffer_id;
        lm_params.output_buffer_id = logitsBufferId(
            use_column_parallel,
            lm_head_compute_all_positions);
        lm_params.compute_all_positions = lm_head_compute_all_positions;
        lm_params.use_prefill_replay_row_offset = lm_head_use_prefill_row_offset;
        const LMHeadInputLayout lm_layout{
            lm_head_seq_len,
            lm_head_input_buffer_id,
            lm_head_compute_all_positions,
            lm_head_use_prefill_row_offset};
        lm_params.force_decode_equivalent_verifier_prefill =
            shouldForceDecodeEquivalentLMHeadVerifierPrefill(
                config_,
                device,
                lm_layout);

        graph.addNode("lm_head",
                      ComputeStageFactory::createLMHead(lm_params),
                      device);
        graph.addDependency("lm_head", lm_head_dependency);

        // =================================================================
        // AllGather stage for column-parallel LM head
        // =================================================================
        if (final_projection.needs_allgather)
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
        DeviceId device,
        void *device_state_publication_stream,
        const int32_t *sequence_lengths_device,
        const int32_t *absolute_position_ids_device)
    {
        if (device.is_gpu() && !device_state_publication_stream)
        {
            throw std::invalid_argument(
                "[QwenGraphBase::buildFFNGraph] GPU graph construction "
                "requires an explicit non-null device-state publication stream");
        }
        (void)sequence_lengths_device;
        (void)absolute_position_ids_device;
        ComputeGraph graph;
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";
        std::string ffn_terminal; // Track the last node for terminalNode()

        // Compute total tokens for GEMM m parameter
        int total_tokens = batch_size * seq_len;
        const bool force_decode_equivalent_ffn_verifier_prefill =
            (device.is_cpu() || device.is_cuda() || device.is_rocm()) &&
            total_tokens > 1 &&
            config_.usesMTPGroupedDecodeEquivalentRows();
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
            gate_up_params.force_decode_equivalent_verifier_prefill =
                force_decode_equivalent_ffn_verifier_prefill;

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
                .force_decode_equivalent_verifier_prefill =
                    force_decode_equivalent_ffn_verifier_prefill,
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

            if (needs_allreduce && needsTPAllreduce() && denseTPAllreduceEnabledForCurrentGraph())
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

                if (needs_allreduce && needsTPAllreduce() && denseTPAllreduceEnabledForCurrentGraph())
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
        const bool reserve_full_dense_decode_buffers = config_.dense_tp_decode_replicated;
        const bool reserve_full_mtp_head_buffers =
            localTPMirroredMTPHeadConfigured();

        config.local_n_heads = config_.qkv_column_parallel && !reserve_full_dense_decode_buffers
                                   ? config_.local_n_heads
                                   : config_.n_heads;
        config.local_n_kv_heads = config_.qkv_column_parallel && !reserve_full_dense_decode_buffers
                                      ? config_.local_n_kv_heads
                                      : config_.n_kv_heads;

        // Validate (safety check for uninitialized config)
        if (config.local_n_heads <= 0)
            config.local_n_heads = config_.n_heads;
        if (config.local_n_kv_heads <= 0)
            config.local_n_kv_heads = config_.n_kv_heads;

        // Use local FFN dimension when FFN is column-parallel
        config.local_d_ff = config_.ffn_column_parallel && !reserve_full_dense_decode_buffers
                                ? config_.d_ff_local
                                : config_.d_ff;
        if (config.local_d_ff <= 0)
            config.local_d_ff = config_.d_ff;

        // Use local vocab when LM head is column-parallel
        config.local_vocab = config_.lm_head_column_parallel && !reserve_full_dense_decode_buffers
                                 ? config_.vocab_local
                                 : config_.vocab_size;
        if (config.local_vocab <= 0)
            config.local_vocab = config_.vocab_size;

        config.custom_formulas["mtp_target_query_rows"] =
            static_cast<size_t>(resolveMTPMaxTargetQueryRows(config_.mtp));
        /*
         * Grouped verification remains bounded by configured speculative depth,
         * while graph-integrated shifted prefill performs one KV-only MTP pass
         * across the complete captured prompt bucket. The shared MTP projection
         * scratch must cover the larger of those two graph families.
         */
        config.custom_formulas["mtp_kv_prefill_rows"] =
            static_cast<size_t>(std::max(
                std::max(1, seq_len),
                resolveMTPMaxTargetQueryRows(config_.mtp)));
        config.custom_formulas["mtp_vocab"] =
            static_cast<size_t>(reserve_full_mtp_head_buffers
                                    ? config_.vocab_size
                                    : config.local_vocab);
        /*
         * CPU GlobalTP computes the MTP LM head column-sharded, then allgathers
         * only its compact sidecar rows for stochastic sampling.  Other
         * topologies use either a single full head or mirrored LocalTP heads and
         * reserve a 1x1 placeholder instead of wasting a full-vocabulary arena.
         */
        const bool reserve_global_mtp_gather =
            config_.mtp.enabled &&
            config_.lm_head_column_parallel &&
            ((config_.tp_ctx != nullptr &&
              !config_.tp_ctx->isLocal() &&
              config_.tp_ctx->degree() > 1) ||
             (config_.tp_ctx == nullptr &&
              mpi_ctx_ != nullptr &&
              mpi_ctx_->world_size() > 1));
        config.custom_formulas["mtp_global_gather_rows"] =
            reserve_global_mtp_gather
                ? static_cast<size_t>(resolveMTPMaxTargetQueryRows(config_.mtp))
                : 1ULL;
        config.custom_formulas["mtp_global_gather_vocab"] =
            reserve_global_mtp_gather
                ? static_cast<size_t>(config_.vocab_size)
                : 1ULL;

        LOG_DEBUG("[QwenGraphBase::getResolverConfig] Created config: "
                  << "seq_len=" << config.seq_len << ", "
                  << "local_n_heads=" << config.local_n_heads << " (n_heads=" << config.n_heads << "), "
                  << "local_n_kv_heads=" << config.local_n_kv_heads << " (n_kv_heads=" << config.n_kv_heads << "), "
                  << "local_d_ff=" << config.local_d_ff << " (d_ff=" << config.d_ff << "), "
                  << "local_vocab=" << config.local_vocab << " (vocab_size=" << config.vocab_size << ")");

        return config;
    }

    std::string QwenGraphBase::addFinalNormToGraph(
        ComputeGraph &graph,
        TensorBase *hidden,
        TensorBase *normalized_out,
        const std::string &prev_node,
        int n_tokens,
        DeviceId device,
        BufferId input_buffer_id,
        const int32_t *sequence_lengths_device)
    {
        RMSNormStage::Params norm_params;
        norm_params.input = hidden;
        norm_params.output = normalized_out;
        norm_params.gamma = modelFinalNorm();
        norm_params.eps = config_.rms_norm_eps;
        norm_params.seq_len = n_tokens;
        norm_params.device_id = device;
        /*
         * The buffer contract must describe the arena slot that backs the
         * tensor pointer above. Hybrid Qwen3.6 routes its live residual stream
         * through RESIDUAL for final norm; advertising HIDDEN_STATE here lets
         * the executor cohere the wrong buffer and can leave GPU final-norm
         * inputs stale after decode/prefix state handoff.
         */
        norm_params.input_buffer_id = input_buffer_id;
        norm_params.output_buffer_id = BufferId::NORMALIZED;

        const std::string norm_dependency =
            maybeAddFinalNormDiagnosticCheckpoint(
                graph,
                "final_norm_input",
                hidden,
                prev_node,
                n_tokens,
                device,
                sequence_lengths_device);
        graph.addNode("final_norm",
                      ComputeStageFactory::createRMSNorm(norm_params),
                      device);

        if (!norm_dependency.empty())
        {
            graph.addDependency("final_norm", norm_dependency);
        }
        return maybeAddFinalNormDiagnosticCheckpoint(
            graph,
            "final_norm_output",
            normalized_out,
            "final_norm",
            n_tokens,
            device,
            sequence_lengths_device);
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
        if (!config_.dense_tp_enabled)
        {
            return false;
        }

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
        std::optional<BufferId> tensor_buffer_id,
        std::vector<TPAllreduceSidebandWorkspaceBinding> sideband_workspace_bindings,
        std::optional<std::string> precision_override) const
    {
        // Unified path: use polymorphic ITPContext for both LOCAL and GLOBAL TP
        if (config_.tp_ctx && config_.tp_ctx->degree() > 1)
        {
            LOG_TRACE("[QwenGraphBase] Creating TPAllreduceStage: degree="
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
            params.precision = precision_override.value_or(
                config_.getAllreducePrecisionForLayer(layer_idx));
            params.tensor_buffer_id = tensor_buffer_id;
            params.sideband_device_index = config_.tp_device_idx;
            params.sideband_workspace_bindings = std::move(sideband_workspace_bindings);

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
        if (useReplicatedAttentionStateWeights())
            return {config_.n_heads, config_.n_kv_heads};

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

        LOG_TRACE("[QwenGraphBase] Layer using QK norm");

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

    attention::AttentionExecutionPolicy
    QwenGraphBase::resolveAttentionExecutionPolicy(
        DeviceId device,
        bool cache_backed) const
    {
        if (!device.is_cpu() && !device.is_gpu())
        {
            throw std::invalid_argument(
                "attention execution policy requires a valid CPU, CUDA, or ROCm device");
        }

        const bool device_transforms_cached_keys =
            cache_backed && device.is_gpu() && config_.rope_on_read;
        return {
            .prefill_parallel_axis =
                attention::AttentionPrefillParallelAxis::GeometrySelected,
            .key_cache = {
                .encoding =
                    device_transforms_cached_keys
                        ? attention::AttentionKeyCacheEncoding::
                              PreRotaryDeviceTransform
                        : attention::AttentionKeyCacheEncoding::PostRotary,
                .rope_theta = config_.rope_theta,
                .partial_rotary_factor = config_.partial_rotary_factor,
            },
        };
    }

    std::string QwenGraphBase::addKeyCachePublicationTransforms(
        ComputeGraph &graph,
        const std::string &prefix,
        ActivationBuffers &buffers,
        const LayerWeights &layer,
        int local_n_kv_heads,
        int total_tokens,
        const int *position_ids,
        const void *position_ids_device,
        DeviceId device,
        const attention::AttentionExecutionPolicy &execution_policy,
        const std::string &projection_dependency)
    {
        if (!buffers.K || local_n_kv_heads <= 0 || total_tokens <= 0 ||
            projection_dependency.empty())
        {
            throw std::invalid_argument(
                "K cache-publication transforms require K, positive geometry, and a producer node");
        }

        std::string key_terminal = projection_dependency;
        if (layer.k_norm)
        {
            const std::string norm_node = prefix + "k_norm";
            graph.addNode(norm_node,
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
            graph.addDependency(norm_node, key_terminal);
            key_terminal = norm_node;
        }

        if (!execution_policy.key_cache.transformsOnRead())
        {
            const std::string rope_node = prefix + "k_rope";
            const int pos_offset = position_ids ? position_ids[0] : 0;
            const bool force_decode_equivalent_rope_verifier_prefill =
                device.is_cpu() &&
                config_.usesMTPGroupedDecodeEquivalentRows() &&
                total_tokens > 1;
            graph.addNode(rope_node,
                          ComputeStageFactory::createKeyOnlyRoPE({
                              .device_id = device,
                              .K = buffers.K,
                              .n_kv_heads = local_n_kv_heads,
                              .head_dim = config_.head_dim,
                              .pos_offset = pos_offset,
                              .theta_base = config_.rope_theta,
                              .seq_len = total_tokens,
                              .partial_rotary_factor = config_.partial_rotary_factor,
                              .position_ids = position_ids,
                              .position_ids_device = position_ids_device,
                              .force_decode_equivalent_verifier_prefill =
                                  force_decode_equivalent_rope_verifier_prefill,
                              .k_buffer_id = buffers.idFor(BufferId::K_PROJ),
                          }),
                          device);
            graph.addDependency(rope_node, key_terminal);
            key_terminal = rope_node;
        }

        return key_terminal;
    }

    std::string QwenGraphBase::addRoPE(
        ComputeGraph &graph,
        const std::string &prefix,
        ActivationBuffers &buffers,
        int local_n_heads,
        int local_n_kv_heads,
        int total_tokens,
        const int *position_ids,
        const void *position_ids_device,
        DeviceId device,
        const attention::AttentionExecutionPolicy &execution_policy)
    {
        const std::string node_name = prefix + "rope";
        int pos_offset = position_ids ? position_ids[0] : 0;
        const bool force_decode_equivalent_rope_verifier_prefill =
            device.is_cpu() &&
            config_.usesMTPGroupedDecodeEquivalentRows() &&
            total_tokens > 1;

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
                          .position_ids_device = position_ids_device,
                          .skip_k = execution_policy.key_cache.transformsOnRead(),
                          .force_decode_equivalent_verifier_prefill =
                              force_decode_equivalent_rope_verifier_prefill,
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
        const int32_t *request_sequence_lengths_device,
        DeviceId device,
        const attention::AttentionExecutionPolicy &execution_policy,
        const std::string &rope_dependency,
        const std::vector<std::string> &cache_source_dependencies,
        bool layer_idx_is_cache_local,
        int first_seq_idx)
    {
        int total_tokens = batch_size * seq_len;
        const int kv_stage_layer = kvCacheLayerForGraphStage(
            kv_cache, layer_idx, config_.pp_layer_offset, layer_idx_is_cache_local);

        if (kv_cache)
        {
            /*
             * Request-prefill lengths and request-condition append widths are
             * different policies even though both are represented by one
             * device INT32 row at the orchestration boundary. During padded
             * prefill, each value is the number of real rows to copy from a
             * fixed bucket and must govern KV append. During a live MTP main
             * condition, the values are absolute post-append logical lengths
             * used by positions, recurrent state, and terminal-row selection;
             * the KV graph itself appends exactly its captured `seq_len` rows.
             * Binding those absolute lengths as append counts makes the
             * capture guard reject every value greater than `seq_len`, turning
             * the condition append into a silent zero-row operation.
             *
             * Grouped verifier ownership is equally explicit. Compact
             * request-batch prefill also enables all-position logits so it can
             * select one terminal row per request; that output policy does not
             * make its cache mutation speculative. Inferring append semantics
             * from `compute_all_position_logits` would append every padded row
             * and corrupt every request shorter than the physical graph width.
             */
            const int32_t *kv_append_lengths_device =
                config_.live_mtp_request_batch_condition
                    ? nullptr
                    : request_sequence_lengths_device;
            const bool phase_split_handoff =
                needsPhaseSplitPrefillKVCacheHandoff(total_tokens, kv_cache, device);
            ITensor *append_K = buffers.K;
            ITensor *append_V = buffers.V;
            std::optional<BufferId> append_k_buffer_id = buffers.idFor(BufferId::K_PROJ);
            std::optional<BufferId> append_v_buffer_id = buffers.idFor(BufferId::V_PROJ);
            std::string append_dependency = rope_dependency;

            if (phase_split_handoff)
            {
                if (!buffers.K_full_prefill || !buffers.V_full_prefill)
                {
                    LOG_ERROR("[QwenGraphBase] Phase-split prefill KV handoff requires K_full_prefill/V_full_prefill buffers");
                    throw std::runtime_error("phase-split prefill full K/V buffers missing");
                }
                auto *local_tp = dynamic_cast<ILocalTPContext *>(config_.tp_ctx);
                if (!local_tp)
                {
                    LOG_ERROR("[QwenGraphBase] Phase-split prefill KV handoff requires a LocalTP context");
                    throw std::runtime_error("phase-split prefill KV handoff requires LocalTP context");
                }
                if (!local_tp->supportsRawAllgatherOnStreamGraphCapture())
                {
                    LOG_ERROR("[QwenGraphBase] Phase-split prefill KV handoff requires graph-capturable raw allgather");
                    throw std::runtime_error("phase-split prefill KV handoff requires graph-capturable raw allgather");
                }

                const int local_kv_dim = config_.local_n_kv_heads * config_.head_dim;
                const int full_kv_dim = config_.n_kv_heads * config_.head_dim;
                if (!buffers.K || !buffers.V ||
                    local_kv_dim <= 0 ||
                    static_cast<int>(buffers.K->cols()) < local_kv_dim ||
                    static_cast<int>(buffers.V->cols()) < local_kv_dim ||
                    static_cast<int>(buffers.K_full_prefill->cols()) < full_kv_dim ||
                    static_cast<int>(buffers.V_full_prefill->cols()) < full_kv_dim)
                {
                    LOG_ERROR("[QwenGraphBase] Invalid phase-split prefill K/V handoff dimensions"
                              << " local_kv_dim=" << local_kv_dim
                              << " full_kv_dim=" << full_kv_dim);
                    throw std::runtime_error("invalid phase-split prefill K/V handoff dimensions");
                }

                const std::string handoff_node = prefix + "tp_kv_state_allgather";
                graph.addNode(handoff_node,
                              ComputeStageFactory::createTPKVCacheStateAllGather({
                                  .device_id = device,
                                  .tp_ctx = local_tp,
                                  .local_K = buffers.K,
                                  .local_V = buffers.V,
                                  .full_K = buffers.K_full_prefill,
                                  .full_V = buffers.V_full_prefill,
                                  .layer_idx = kv_stage_layer,
                                  .tp_device_idx = config_.tp_device_idx,
                                  .tokens = total_tokens,
                                  .local_kv_dim = local_kv_dim,
                                  .full_kv_dim = full_kv_dim,
                                  .local_k_stride = local_kv_dim,
                                  .local_v_stride = local_kv_dim,
                                  .stage_name = handoff_node,
                                  .local_k_buffer_id = buffers.idFor(BufferId::K_PROJ),
                                  .local_v_buffer_id = buffers.idFor(BufferId::V_PROJ),
                                  .full_k_buffer_id = buffers.idFor(BufferId::K_FULL_PREFILL),
                                  .full_v_buffer_id = buffers.idFor(BufferId::V_FULL_PREFILL),
                              }),
                              device);

                if (execution_policy.key_cache.transformsOnRead() &&
                    !cache_source_dependencies.empty())
                {
                    for (const auto &dep : cache_source_dependencies)
                    {
                        if (!dep.empty())
                            graph.addDependency(handoff_node, dep);
                    }
                }
                else
                {
                    /*
                     * Without RoPE-on-read the replicated cache must receive
                     * post-RoPE K. Make the full-row allgather consume the RoPE
                     * output directly. With RoPE-on-read, the branch above
                     * gathers pre-RoPE K while Q RoPE proceeds independently.
                     */
                    graph.addDependency(handoff_node, rope_dependency);
                }

                append_K = buffers.K_full_prefill;
                append_V = buffers.V_full_prefill;
                append_k_buffer_id = buffers.idFor(BufferId::K_FULL_PREFILL);
                append_v_buffer_id = buffers.idFor(BufferId::V_FULL_PREFILL);
                append_dependency = handoff_node;
            }

            graph.addNode(prefix + "kv_append",
                          ComputeStageFactory::createKVCacheAppend({
                              .device_id = device,
                              .K = append_K,
                              .V = append_V,
                              .kv_cache = kv_cache,
                              .layer_idx = kv_stage_layer,
                              .seq_idx = first_seq_idx,
                              .num_tokens = total_tokens,
                              .batch_size = batch_size,
                              .seq_len = seq_len,
                              .append_semantics =
                                  config_.grouped_mtp_verifier
                                      ? KVCacheAppendSemantics::DecodeEquivalentVerifier
                                      : KVCacheAppendSemantics::Standard,
                              .request_sequence_lengths_device =
                                  kv_append_lengths_device,
                              .kv_cache_scale_k = config_.kv_cache_scale_k,
                              .kv_cache_scale_v = config_.kv_cache_scale_v,
                              .head_dim = config_.head_dim,
                              .turboquant_ctx = config_.turboquant_ctx,
                              .kv_rotation = config_.kv_rotation,
                              .k_buffer_id = append_k_buffer_id,
                              .v_buffer_id = append_v_buffer_id,
                          }),
                          device);

            // In rope-on-read mode the RoPE stage skips K mutation, but it still
            // carries the Q/K norm dependencies. Appending directly after QKV GEMM
            // can cache pre-normalized K on Qwen3.5 FA layers.
            graph.addDependency(prefix + "kv_append", append_dependency);
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
        const void *position_ids_device,
        const int32_t *request_sequence_lengths_device,
        DeviceId device,
        bool has_qkv_proj,
        const attention::AttentionExecutionPolicy &execution_policy,
        const std::string &rope_dependency,
        const std::vector<std::string> &cache_source_dependencies,
        bool layer_idx_is_cache_local)
    {
        (void)position_ids_device;
        int total_tokens = batch_size * seq_len;
        const int kv_stage_layer = kvCacheLayerForGraphStage(
            kv_cache, layer_idx, config_.pp_layer_offset, layer_idx_is_cache_local);
        const bool phase_split_handoff =
            needsPhaseSplitPrefillKVCacheHandoff(total_tokens, kv_cache, device);

        const std::string kv_append_dependency =
            addKVCacheAppend(
                graph,
                prefix,
                buffers,
                layer_idx,
                seq_len,
                batch_size,
                kv_cache,
                request_sequence_lengths_device,
                device,
                execution_policy,
                rope_dependency,
                cache_source_dependencies,
                layer_idx_is_cache_local);

        // --- Determine K/V source for attention ---
        ITensor *K_for_attn = buffers.K;
        ITensor *V_for_attn = buffers.V;
        int kv_len = total_tokens;
        bool use_gather_stage = false;

        if (kv_cache && !device.is_gpu())
        {
            int cached_tokens = kv_cache->get_cached_tokens(kv_stage_layer, 0);
            if (cached_tokens > 0 && batch_size == 1)
            {
                K_for_attn = kv_cache->get_k(kv_stage_layer, 0);
                V_for_attn = kv_cache->get_v(kv_stage_layer, 0);
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
        else if (kv_cache && device.is_gpu())
        {
            /*
             * GPU attention resolves the live cache through its captured
             * device-state read. Keep only a host-known launch horizon here;
             * neither cache metadata nor cache-sized tensor wrappers belong in
             * graph construction. The first position ID is request geometry,
             * not a mirror of the cache's canonical head/count allocation.
             */
            const int request_offset =
                position_ids ? std::max(0, position_ids[0]) : 0;
            kv_len = std::clamp(
                request_offset + seq_len,
                std::max(1, seq_len),
                std::max(1, kv_cache->max_seq_len()));
        }

        // --- KV Cache Gather (batched decode) ---
        if (use_gather_stage)
        {
            graph.addNode(prefix + "kv_gather",
                          ComputeStageFactory::createKVCacheGather({
                              .kv_cache = kv_cache,
                              .layer_idx = kv_stage_layer,
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
            /*
             * Phase-split prefill seeds an unsharded cache so replicated decode
             * can run without a KV collective. Prefill must consume that same
             * post-append cache to remain numerically identical across full and
             * chunked requests. The attention kernel therefore sees global KV
             * geometry while its Q heads remain TP-local; head_start plus
             * gqa_n_rep maps each local Q head to the correct replicated KV head.
             */
            const int attention_n_kv_heads =
                phase_split_handoff ? config_.n_kv_heads : local_n_kv_heads;
            attn_params.n_kv_heads = attention_n_kv_heads;
            attn_params.head_dim = config_.head_dim;
            attn_params.head_start = config_.head_start;
            // GQA rep: only when KV heads are replicated (not column-parallel)
            if (attention_n_kv_heads == config_.n_kv_heads &&
                config_.n_kv_heads > 0 &&
                local_n_heads != config_.n_heads)
                attn_params.gqa_n_rep = config_.n_heads / config_.n_kv_heads;
            attn_params.causal = true;
            attn_params.window_size = -1;
            attn_params.attention_mode = mode;
            attn_params.auto_detect_mode = true;
            /*
             * The graph resolves physical parallelism and cached-K encoding
             * once. The exact same value already governed RoPE and cache append,
             * so the attention consumer cannot silently choose different bytes.
             */
            attn_params.execution_policy = execution_policy;
            attn_params.workspace_scores = buffers.workspace_scores;
            attn_params.workspace_context = buffers.workspace_context;
            /*
             * Qwen attention has no external additive mask tensor. CUDA and
             * ROCm FA2 derive causal, sliding-window, and continuation bounds
             * from their device-owned attention geometry. Historically this
             * field aliased GEMM_WORKSPACE even though no stage populated it
             * as a mask; graph replay could consequently interpret recycled
             * GEMM bytes with a new live-KV stride. Leave the optional mask
             * null here. Explicit-mask kernel and stage callers retain their
             * separate API contract.
             */
            attn_params.kv_cache = kv_cache;
            attn_params.layer_idx = kv_stage_layer;
            attn_params.active_query_rows_device =
                device.is_gpu() && batch_size == 1
                    ? request_sequence_lengths_device
                    : nullptr;
            /*
             * GPU attention has one production source: the post-append,
             * device-owned KV cache. This is true for prefill, continuation
             * prefill, serial decode, and grouped verifier rows. Selecting the
             * transient projection buffer for one phase changes both precision
             * and RoPE timing, breaking batch invariance. Backends missing a
             * native or converted grouped cache view fail at that contract
             * boundary; there is no projection-buffer fallback.
             */
            attn_params.read_kv_from_cache = kv_cache && device.is_gpu();
            attn_params.position_offset = position_ids ? position_ids[0] : 0;
            attn_params.mpi_ctx = mpi_ctx_.get();
            attn_params.q_buffer_id = buffers.idFor(BufferId::Q_PROJ);
            attn_params.output_buffer_id = buffers.idFor(BufferId::ATTN_OUTPUT);
            // NOTE: workspace_scores/workspace_context are no longer registered in
            // the arena (CPUFlashAttentionKernelT doesn't use them — O(S²) dead
            // buffers removed). Don't set buffer_ids for the contract.
            attn_params.turboquant_ctx = config_.turboquant_ctx;
            attn_params.kv_rotation = config_.kv_rotation;

            graph.addNode(prefix + "attention",
                          ComputeStageFactory::createAttentionCompute(attn_params),
                          device);

            if (use_gather_stage)
                graph.addDependency(prefix + "attention", prefix + "kv_gather");
            else if (kv_cache)
            {
                graph.addDependency(prefix + "attention", kv_append_dependency);
                if (phase_split_handoff)
                    graph.addDependency(prefix + "attention", rope_dependency);
            }
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
        /*
         * Grouped verifier rows are promoted through the attention output
         * projection as a normal small-M GEMM.  The serial M=1 verifier loop is
         * kept available inside GEMMStage for explicit diagnostics, but the
         * production graph relies on the Phase 9.8 dense verifier proof of
         * byte-identical runtime-M grouped output instead of paying a row-copy
         * loop in every attention block.
         */
        const bool force_decode_equivalent_wo_verifier_prefill =
            (device.is_cpu() || device.is_cuda() || device.is_rocm()) &&
            total_tokens > 1 &&
            config_.usesMTPGroupedDecodeEquivalentRows();

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
                          .force_decode_equivalent_verifier_prefill = force_decode_equivalent_wo_verifier_prefill,
                          .prepared_ref = preparedRefForGraphWeight(wo_binding, device),
                          .prepared_store = prepared_weight_store_,
                      }),
                      device);
        graph.addDependency(wo_node, dependency);

        std::string terminal = wo_node;

        // TP allreduce if row-parallel sharded
        if (isRowParallelSharded(wo_weight) && needsTPAllreduce() && attentionTPAllreduceEnabledForCurrentGraph())
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
