/**
 * @file ForwardGraphTypes.h
 * @brief Data types for forward graph caching (extracted from DeviceGraphOrchestrator)
 *
 * Contains ForwardGraphSignature, ForwardGraphSignatureHash, and ForwardGraphCache —
 * the key data structures for caching compiled forward graphs between decode steps.
 */

#pragma once

#include "../../../backends/DeviceId.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../graph/DeviceGraphExecutor.h"
#include "../../compute_stages/IComputeStage.h"
#include "../graph/IGraphBuilder.h"             // For ForwardOutput

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Result of a graph build operation
     *
     * Holds either a successfully built ComputeGraph + ForwardOutput, or
     * an error string. Extracted from DeviceGraphOrchestrator for use by the
     * ForwardExecutionEngine.
     */
    class GraphBuildResult
    {
    public:
        GraphBuildResult() = default;
        GraphBuildResult(ComputeGraph graph, ForwardOutput output)
            : graph_(std::move(graph)), output_(output), success_(true) {}
        explicit GraphBuildResult(std::string error)
            : error_(std::move(error)), success_(false) {}

        [[nodiscard]] bool success() const { return success_; }
        [[nodiscard]] bool failed() const { return !success_; }
        [[nodiscard]] const std::string &error() const { return error_; }
        [[nodiscard]] ComputeGraph &graph() { return graph_; }
        [[nodiscard]] const ComputeGraph &graph() const { return graph_; }
        [[nodiscard]] const ForwardOutput &output() const { return output_; }
        [[nodiscard]] ComputeGraph takeGraph() { return std::move(graph_); }
        explicit operator bool() const { return success_; }

    private:
        ComputeGraph graph_;
        ForwardOutput output_{};
        std::string error_;
        bool success_ = false;
    };

    /**
     * @brief Configuration for graph caching behaviour
     */
    struct GraphCacheConfig
    {
        bool enabled = true;         ///< Enable graph caching (Phase 10)
        int decode_seq_len = 1;      ///< Sequence length that triggers decode caching
        bool cache_attention = true; ///< Cache attention graphs
        bool cache_ffn = true;       ///< Cache FFN graphs
    };

    /**
     * @brief Signature for caching full forward graphs.
     *
     * Captures the execution shape so that graphs built for identical shapes
     * can be reused across decode steps without rebuilding stages/kernels.
     */
    struct ForwardGraphSignature
    {
        int seq_len = 0;
        int batch_size = 0;
        DeviceId device = DeviceId::cpu();
        bool decode = false;
        bool standard_path = true;
        bool pp_stage_enabled = false;
        int pp_first_layer = -1;
        int pp_last_layer = -1;
        bool pp_has_embedding = false;
        bool pp_has_lm_head = false;

        bool operator==(const ForwardGraphSignature &other) const
        {
            return seq_len == other.seq_len &&
                   batch_size == other.batch_size &&
                   device == other.device &&
                   decode == other.decode &&
                   standard_path == other.standard_path &&
                   pp_stage_enabled == other.pp_stage_enabled &&
                   pp_first_layer == other.pp_first_layer &&
                   pp_last_layer == other.pp_last_layer &&
                   pp_has_embedding == other.pp_has_embedding &&
                   pp_has_lm_head == other.pp_has_lm_head;
        }
    };

    struct ForwardGraphSignatureHash
    {
        size_t operator()(const ForwardGraphSignature &sig) const
        {
            size_t h = std::hash<int>{}(sig.seq_len);
            h ^= (std::hash<int>{}(sig.batch_size) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<DeviceId>{}(sig.device) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.decode) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.standard_path) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.pp_stage_enabled) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<int>{}(sig.pp_first_layer) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<int>{}(sig.pp_last_layer) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.pp_has_embedding) + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= (std::hash<bool>{}(sig.pp_has_lm_head) + 0x9e3779b9 + (h << 6) + (h >> 2));
            return h;
        }
    };

    /**
     * @brief Cached full forward graph for decode mode
     *
     * During decode (seq_len=1), the graph structure is identical between
     * steps — only token_ids, position_ids, and position_offset change.
     * Instead of rebuilding hundreds of stage objects every forward() call,
     * we cache the graph and its stages after the first decode step.
     *
     * Stable buffers (token_ids, position_ids) are owned here so that
     * cached stages' pointers remain valid across calls.
     */
    struct ForwardGraphCache
    {
        std::unique_ptr<ComputeGraph> graph; ///< Cached compute graph
        ForwardOutput output;           ///< Cached output (logits pointer)
        bool valid = false;                  ///< Whether cache is usable

        // Stable buffers — stages point to these, contents updated each step
        std::vector<int> token_ids;    ///< Persistent decode token storage
        std::vector<int> position_ids; ///< Persistent decode position IDs

        // PP hidden state copy — for non-embedding PP stages, the external
        // hidden state must be copied to the working buffer on every forward.
        // During graph build (cache MISS) this copy happens inline in
        // QwenStandardGraph::buildPartialForwardGraph(). On cache HIT we must redo
        // the copy here because the graph build code is not re-executed.
        TensorBase *pp_external_hidden_state = nullptr; ///< Source (stage N-1 output)
        TensorBase *pp_working_buffer = nullptr;        ///< Destination (local residual/hidden)
        size_t pp_copy_bytes = 0;
        DeviceId pp_device;
        bool pp_needs_copy = false;

        // Pre-computed collective stage names for fast decode intercept
        std::unordered_set<std::string> collective_nodes;

        // Pre-cached pointers to stages that override updateDynamicParams().
        // Only ~4 stages (RoPE, Attention, FusedAttention, KVCacheAppend) need
        // updating — avoids iterating all ~339 stages with hash lookups each step.
        std::vector<IComputeStage *> dynamic_param_stages;
        bool dynamic_param_stages_cached = false;

        // Tracks whether setGPUStream has been applied to all stages.
        // The capture_stream never changes once set, so we skip the
        // 339-stage loop on subsequent decode steps.
        bool gpu_stream_applied = false;

        // The stream pointer that was last applied to all stages. If
        // segment_cache.reset() destroys the capture_stream and a new one is
        // created, we must re-apply it to avoid stages holding a dangling
        // stream pointer.
        void *applied_stream = nullptr;

        // Tracks whether Phase 3 graph replay is active (no markCompleted calls),
        // allowing us to skip the 339-node graph.reset() since flags are already clear.
        bool phase3_active = false;

        /// GPU graph capture/replay for eliminating per-kernel launch overhead
        std::unique_ptr<IGPUGraphCapture> gpu_graph;

        /// Segmented GPU graph cache — excludes non-capturable stages (attention, KV cache)
        /// and captures contiguous runs of capturable stages into separate graphs
        DeviceGraphExecutor::GraphSegmentCache segment_cache;

        /// GPU stream (from IWorkerGPUContext::defaultStream()) for kernel dispatch
        /// Set when gpu_graph is created; used by stages to dispatch on correct stream
        void *gpu_stream = nullptr;

        /// GPU context for creating new graph captures (not owned)
        IWorkerGPUContext *gpu_ctx = nullptr;

        /// Number of consecutive graph update failures (fallback heuristic)
        int gpu_graph_update_failures = 0;

        /// Maximum consecutive update failures before disabling graph capture
        static constexpr int kMaxGraphUpdateFailures = 4;

        void invalidate()
        {
            if (gpu_graph)
            {
                gpu_graph->reset();
                gpu_graph.reset();
            }
            segment_cache.reset();
            gpu_stream = nullptr;
            gpu_ctx = nullptr;
            gpu_graph_update_failures = 0;
            graph.reset();
            valid = false;
            token_ids.clear();
            position_ids.clear();
            collective_nodes.clear();
            dynamic_param_stages.clear();
            dynamic_param_stages_cached = false;
            gpu_stream_applied = false;
            applied_stream = nullptr;
            phase3_active = false;
            pp_external_hidden_state = nullptr;
            pp_working_buffer = nullptr;
            pp_copy_bytes = 0;
            pp_needs_copy = false;
        }
    };

} // namespace llaminar2
