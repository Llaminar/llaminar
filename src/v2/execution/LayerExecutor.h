/**
 * @file LayerExecutor.h
 * @brief Layer-level execution orchestration for transformer blocks
 * @author David Sanftenberg
 * @date December 2025
 *
 * LayerExecutor coordinates the execution of ComputeStages within a transformer
 * layer, managing:
 * - Compute graph construction for attention and FFN blocks
 * - Device-aware stage scheduling
 * - Asynchronous execution with dependency tracking
 * - Multi-device work distribution via WorkDistributor
 *
 * Architecture:
 *   Pipeline
 *      |
 *      v
 *   LayerExecutor  <-- This component
 *      |
 *      v
 *   ComputeStage[] (device-specific implementations)
 *      |
 *      v
 *   DeviceContext (execution environment)
 */

#pragma once

#include "ComputeStage.h"
#include "DeviceContext.h"
#include "WorkDistributor.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;

    /**
     * @brief Execution mode for LayerExecutor
     */
    enum class ExecutionMode
    {
        SEQUENTIAL, ///< Execute stages one at a time (debugging)
        PARALLEL,   ///< Execute independent stages in parallel
        PIPELINED   ///< Overlap stages where possible (future)
    };

    /**
     * @brief Represents a node in the compute graph
     */
    struct ComputeNode
    {
        std::string name;                      ///< Node identifier
        std::unique_ptr<IComputeStage> stage;  ///< The compute stage
        std::vector<std::string> dependencies; ///< Names of nodes this depends on
        int device_idx;                        ///< Target device for execution
        bool completed;                        ///< Execution complete flag

        ComputeNode() : device_idx(-1), completed(false) {}
        ComputeNode(std::string n, std::unique_ptr<IComputeStage> s, int dev = -1)
            : name(std::move(n)), stage(std::move(s)), device_idx(dev), completed(false) {}
    };

    /**
     * @brief Compute graph for layer execution
     *
     * A directed acyclic graph (DAG) of ComputeNodes with dependency tracking.
     * Enables parallel execution of independent nodes and proper ordering of
     * dependent operations.
     */
    class ComputeGraph
    {
    public:
        ComputeGraph() = default;
        ~ComputeGraph() = default;

        // Non-copyable
        ComputeGraph(const ComputeGraph &) = delete;
        ComputeGraph &operator=(const ComputeGraph &) = delete;

        // Movable
        ComputeGraph(ComputeGraph &&) = default;
        ComputeGraph &operator=(ComputeGraph &&) = default;

        /**
         * @brief Add a node to the graph
         * @param name Unique node identifier
         * @param stage The compute stage to execute
         * @param device_idx Target device (-1 for auto)
         * @return Reference to this graph for chaining
         */
        ComputeGraph &addNode(const std::string &name,
                              std::unique_ptr<IComputeStage> stage,
                              int device_idx = -1);

        /**
         * @brief Add a dependency between nodes
         * @param node_name The dependent node
         * @param depends_on The node that must complete first
         * @return Reference to this graph for chaining
         */
        ComputeGraph &addDependency(const std::string &node_name,
                                    const std::string &depends_on);

        /**
         * @brief Get execution order respecting dependencies
         * @return Vector of node names in valid execution order
         */
        std::vector<std::string> getExecutionOrder() const;

        /**
         * @brief Get nodes that can execute in parallel (no unmet dependencies)
         * @return Vector of names of ready nodes
         */
        std::vector<std::string> getReadyNodes() const;

        /**
         * @brief Get a node by name
         * @param name Node identifier
         * @return Pointer to node (nullptr if not found)
         */
        ComputeNode *getNode(const std::string &name);
        const ComputeNode *getNode(const std::string &name) const;

        /**
         * @brief Mark a node as completed
         * @param name Node identifier
         */
        void markCompleted(const std::string &name);

        /**
         * @brief Reset all completion flags
         */
        void reset();

        /**
         * @brief Check if all nodes are completed
         */
        bool allCompleted() const;

        /**
         * @brief Get number of nodes
         */
        size_t size() const { return nodes_.size(); }

        /**
         * @brief Get total estimated FLOPs for all stages
         */
        size_t totalEstimatedFlops() const;

        /**
         * @brief Clear the graph
         */
        void clear();

    private:
        std::vector<std::unique_ptr<ComputeNode>> nodes_;
        std::unordered_map<std::string, size_t> node_index_;
    };

    /**
     * @brief Configuration for LayerExecutor
     */
    struct LayerExecutorConfig
    {
        ExecutionMode mode = ExecutionMode::SEQUENTIAL;
        bool enable_profiling = false;  ///< Track per-stage timing
        bool enable_validation = false; ///< Validate outputs after each stage
        int default_device = 0;         ///< Default device for stages
    };

    /**
     * @brief Layer execution statistics
     */
    struct LayerExecutorStats
    {
        size_t total_stages_executed = 0;
        size_t total_flops = 0;
        double total_time_ms = 0.0;
        std::unordered_map<std::string, double> stage_times_ms;

        void reset()
        {
            total_stages_executed = 0;
            total_flops = 0;
            total_time_ms = 0.0;
            stage_times_ms.clear();
        }
    };

    /**
     * @brief Orchestrates execution of compute stages within a layer
     *
     * LayerExecutor manages the execution of transformer blocks (attention, FFN)
     * by building compute graphs and executing them across devices.
     *
     * Usage:
     * @code
     * LayerExecutor executor(config);
     *
     * // Build attention block graph
     * auto attn_graph = executor.buildAttentionGraph(layer_weights, buffers);
     * executor.execute(attn_graph, ctx);
     *
     * // Build FFN block graph
     * auto ffn_graph = executor.buildFFNGraph(layer_weights, buffers);
     * executor.execute(ffn_graph, ctx);
     * @endcode
     */
    class LayerExecutor
    {
    public:
        /**
         * @brief Construct with configuration
         * @param config Executor configuration
         */
        explicit LayerExecutor(const LayerExecutorConfig &config = LayerExecutorConfig());
        ~LayerExecutor();

        // Non-copyable, movable
        LayerExecutor(const LayerExecutor &) = delete;
        LayerExecutor &operator=(const LayerExecutor &) = delete;
        LayerExecutor(LayerExecutor &&) = default;
        LayerExecutor &operator=(LayerExecutor &&) = default;

        // =========================================================================
        // Configuration
        // =========================================================================

        /**
         * @brief Get current configuration
         */
        const LayerExecutorConfig &config() const { return config_; }

        /**
         * @brief Set execution mode
         */
        void setExecutionMode(ExecutionMode mode) { config_.mode = mode; }

        /**
         * @brief Get current execution mode
         */
        ExecutionMode executionMode() const { return config_.mode; }

        /**
         * @brief Enable/disable profiling
         */
        void setProfilingEnabled(bool enabled) { config_.enable_profiling = enabled; }

        /**
         * @brief Enable/disable output validation
         */
        void setValidationEnabled(bool enabled) { config_.enable_validation = enabled; }

        // =========================================================================
        // Graph Building - Attention Block
        // =========================================================================

        /**
         * @brief Parameters for attention block
         */
        struct AttentionParams
        {
            // Input/output tensors
            float *input = nullptr;    ///< Input activation [seq_len, d_model]
            float *output = nullptr;   ///< Output [seq_len, d_model]
            float *residual = nullptr; ///< Residual connection

            // Weight tensors
            const float *wq = nullptr;        ///< Query weights [d_model, d_model]
            const float *wk = nullptr;        ///< Key weights [d_model, kv_dim]
            const float *wv = nullptr;        ///< Value weights [d_model, kv_dim]
            const float *wo = nullptr;        ///< Output weights [d_model, d_model]
            const float *attn_norm = nullptr; ///< Attention norm weights

            // KV cache
            float *k_cache = nullptr; ///< Key cache [max_seq, n_kv_heads, head_dim]
            float *v_cache = nullptr; ///< Value cache [max_seq, n_kv_heads, head_dim]

            // Dimensions
            int seq_len = 0;
            int d_model = 0;
            int n_heads = 0;
            int n_kv_heads = 0;
            int head_dim = 0;
            int pos = 0; ///< Current position for RoPE
            float rope_theta = 10000.0f;
            float rms_norm_eps = 1e-5f;
        };

        /**
         * @brief Build compute graph for attention block
         * @param params Attention parameters
         * @return Compute graph ready for execution
         */
        ComputeGraph buildAttentionGraph(const AttentionParams &params);

        // =========================================================================
        // Graph Building - FFN Block
        // =========================================================================

        /**
         * @brief Parameters for FFN block
         */
        struct FFNParams
        {
            // Input/output tensors
            float *input = nullptr;    ///< Input [seq_len, d_model]
            float *output = nullptr;   ///< Output [seq_len, d_model]
            float *residual = nullptr; ///< Residual connection

            // Weight tensors
            const float *gate_weight = nullptr; ///< Gate weights [d_ff, d_model]
            const float *up_weight = nullptr;   ///< Up weights [d_ff, d_model]
            const float *down_weight = nullptr; ///< Down weights [d_model, d_ff]
            const float *ffn_norm = nullptr;    ///< FFN norm weights

            // Dimensions
            int seq_len = 0;
            int d_model = 0;
            int d_ff = 0;
            float rms_norm_eps = 1e-5f;
        };

        /**
         * @brief Build compute graph for FFN block
         * @param params FFN parameters
         * @return Compute graph ready for execution
         */
        ComputeGraph buildFFNGraph(const FFNParams &params);

        // =========================================================================
        // Graph Building - MoE Block
        // =========================================================================

        /**
         * @brief Parameters for MoE block
         */
        struct MoEParams
        {
            // Input/output tensors
            float *input = nullptr;    ///< Input [seq_len, d_model]
            float *output = nullptr;   ///< Output [seq_len, d_model]
            float *residual = nullptr; ///< Residual connection

            // Router weights
            const float *router_weight = nullptr; ///< [n_experts, d_model]

            // Expert weights (array of pointers, one per expert)
            const float *const *gate_weights = nullptr; ///< [n_experts][d_ff, d_model]
            const float *const *up_weights = nullptr;   ///< [n_experts][d_ff, d_model]
            const float *const *down_weights = nullptr; ///< [n_experts][d_model, d_ff]

            // Norm weights
            const float *ffn_norm = nullptr;

            // Dimensions
            int seq_len = 0;
            int d_model = 0;
            int d_ff = 0;
            int n_experts = 0;
            int top_k = 2; ///< Top-k experts per token
            float rms_norm_eps = 1e-5f;

            // Expert parallelism
            bool enable_expert_parallel = false;    ///< Distribute experts across devices
            const int *expert_device_map = nullptr; ///< Expert -> device mapping
        };

        /**
         * @brief Build compute graph for MoE block
         * @param params MoE parameters
         * @return Compute graph ready for execution
         */
        ComputeGraph buildMoEGraph(const MoEParams &params);

        // =========================================================================
        // Execution
        // =========================================================================

        /**
         * @brief Execute a compute graph
         * @param graph The compute graph to execute
         * @param ctx Device context for execution
         * @return true on success
         */
        bool execute(ComputeGraph &graph, IDeviceContext *ctx);

        /**
         * @brief Execute a compute graph with multi-device support
         * @param graph The compute graph to execute
         * @param contexts Map of device_idx -> DeviceContext
         * @return true on success
         */
        bool executeMultiDevice(
            ComputeGraph &graph,
            const std::unordered_map<int, IDeviceContext *> &contexts);

        // =========================================================================
        // Statistics
        // =========================================================================

        /**
         * @brief Get execution statistics
         */
        const LayerExecutorStats &stats() const { return stats_; }

        /**
         * @brief Reset statistics
         */
        void resetStats() { stats_.reset(); }

    private:
        LayerExecutorConfig config_;
        LayerExecutorStats stats_;

        // Internal execution helpers
        bool executeSequential(ComputeGraph &graph, IDeviceContext *ctx);
        bool executeParallel(ComputeGraph &graph, IDeviceContext *ctx);
        bool executeNode(ComputeNode &node, IDeviceContext *ctx);

        // Workspace management
        std::vector<float> temp_buffer_;
        size_t temp_buffer_size_ = 0;

        float *getTemporaryBuffer(size_t elements);
    };

    /**
     * @brief Get human-readable name for execution mode
     */
    const char *executionModeName(ExecutionMode mode);

} // namespace llaminar2
