/**
 * @file LayerExecutor.cpp
 * @brief Layer-level execution orchestration implementation
 * @author David Sanftenberg
 * @date December 2025
 */

#include "LayerExecutor.h"
#include "../utils/Logger.h"
#include <algorithm>
#include <chrono>
#include <queue>
#include <stdexcept>

namespace llaminar2
{

    // =============================================================================
    // ExecutionMode Helpers
    // =============================================================================

    const char *executionModeName(ExecutionMode mode)
    {
        switch (mode)
        {
        case ExecutionMode::SEQUENTIAL:
            return "SEQUENTIAL";
        case ExecutionMode::PARALLEL:
            return "PARALLEL";
        case ExecutionMode::PIPELINED:
            return "PIPELINED";
        default:
            return "UNKNOWN";
        }
    }

    // =============================================================================
    // ComputeGraph Implementation
    // =============================================================================

    ComputeGraph &ComputeGraph::addNode(const std::string &name,
                                        std::unique_ptr<IComputeStage> stage,
                                        int device_idx)
    {
        if (node_index_.find(name) != node_index_.end())
        {
            LOG_WARN("[ComputeGraph] Node '" << name << "' already exists, replacing");
            size_t idx = node_index_[name];
            nodes_[idx]->stage = std::move(stage);
            nodes_[idx]->device_idx = device_idx;
            nodes_[idx]->completed = false;
            return *this;
        }

        auto node = std::make_unique<ComputeNode>(name, std::move(stage), device_idx);
        node_index_[name] = nodes_.size();
        nodes_.push_back(std::move(node));
        return *this;
    }

    ComputeGraph &ComputeGraph::addDependency(const std::string &node_name,
                                              const std::string &depends_on)
    {
        auto it = node_index_.find(node_name);
        if (it == node_index_.end())
        {
            LOG_ERROR("[ComputeGraph] Node '" << node_name << "' not found");
            return *this;
        }

        if (node_index_.find(depends_on) == node_index_.end())
        {
            LOG_ERROR("[ComputeGraph] Dependency '" << depends_on << "' not found");
            return *this;
        }

        nodes_[it->second]->dependencies.push_back(depends_on);
        return *this;
    }

    std::vector<std::string> ComputeGraph::getExecutionOrder() const
    {
        // Kahn's algorithm for topological sort
        std::unordered_map<std::string, int> in_degree;
        std::unordered_map<std::string, std::vector<std::string>> adjacency;

        // Initialize
        for (const auto &node : nodes_)
        {
            in_degree[node->name] = 0;
            adjacency[node->name] = {};
        }

        // Build adjacency list and compute in-degrees
        for (const auto &node : nodes_)
        {
            for (const auto &dep : node->dependencies)
            {
                adjacency[dep].push_back(node->name);
                in_degree[node->name]++;
            }
        }

        // Find all nodes with no dependencies
        std::queue<std::string> ready;
        for (const auto &[name, degree] : in_degree)
        {
            if (degree == 0)
            {
                ready.push(name);
            }
        }

        // Process in topological order
        std::vector<std::string> order;
        order.reserve(nodes_.size());

        while (!ready.empty())
        {
            std::string current = ready.front();
            ready.pop();
            order.push_back(current);

            for (const auto &neighbor : adjacency[current])
            {
                in_degree[neighbor]--;
                if (in_degree[neighbor] == 0)
                {
                    ready.push(neighbor);
                }
            }
        }

        if (order.size() != nodes_.size())
        {
            LOG_ERROR("[ComputeGraph] Cycle detected in graph!");
        }

        return order;
    }

    std::vector<std::string> ComputeGraph::getReadyNodes() const
    {
        std::vector<std::string> ready;

        for (const auto &node : nodes_)
        {
            if (node->completed)
                continue;

            bool all_deps_complete = true;
            for (const auto &dep : node->dependencies)
            {
                auto dep_node = getNode(dep);
                if (dep_node && !dep_node->completed)
                {
                    all_deps_complete = false;
                    break;
                }
            }

            if (all_deps_complete)
            {
                ready.push_back(node->name);
            }
        }

        return ready;
    }

    ComputeNode *ComputeGraph::getNode(const std::string &name)
    {
        auto it = node_index_.find(name);
        if (it == node_index_.end())
            return nullptr;
        return nodes_[it->second].get();
    }

    const ComputeNode *ComputeGraph::getNode(const std::string &name) const
    {
        auto it = node_index_.find(name);
        if (it == node_index_.end())
            return nullptr;
        return nodes_[it->second].get();
    }

    void ComputeGraph::markCompleted(const std::string &name)
    {
        auto *node = getNode(name);
        if (node)
        {
            node->completed = true;
        }
    }

    void ComputeGraph::reset()
    {
        for (auto &node : nodes_)
        {
            node->completed = false;
        }
    }

    bool ComputeGraph::allCompleted() const
    {
        for (const auto &node : nodes_)
        {
            if (!node->completed)
                return false;
        }
        return true;
    }

    size_t ComputeGraph::totalEstimatedFlops() const
    {
        size_t total = 0;
        for (const auto &node : nodes_)
        {
            if (node->stage)
            {
                total += node->stage->estimatedFlops();
            }
        }
        return total;
    }

    void ComputeGraph::clear()
    {
        nodes_.clear();
        node_index_.clear();
    }

    // =============================================================================
    // LayerExecutor Implementation
    // =============================================================================

    LayerExecutor::LayerExecutor(const LayerExecutorConfig &config)
        : config_(config) {}

    LayerExecutor::~LayerExecutor() = default;

    // =============================================================================
    // Attention Graph Building
    // =============================================================================

    ComputeGraph LayerExecutor::buildAttentionGraph(const AttentionParams &params)
    {
        ComputeGraph graph;

        // Validate required parameters
        if (!params.input || !params.output)
        {
            LOG_ERROR("[LayerExecutor] Invalid attention params: null input/output");
            return graph;
        }

        const int device = config_.default_device;
        const auto backend = ComputeBackendType::CPU_OPENBLAS; // Default backend

        // Pre-attention norm
        if (params.attn_norm)
        {
            RMSNormStage::Params norm_params;
            norm_params.input = params.input;
            norm_params.gamma = params.attn_norm;
            norm_params.seq_len = params.seq_len;
            norm_params.hidden_dim = params.d_model;
            norm_params.eps = params.rms_norm_eps;

            graph.addNode("attn_norm",
                          ComputeStageFactory::createRMSNorm(norm_params, backend),
                          device);
        }

        // QKV projections could be parallel
        // For now, we use placeholders - actual GEMM integration needs KernelFactory

        // Note: GEMM stages are placeholders in current implementation
        // The full implementation would use:
        //   - ComputeStageFactory::createGEMM() with weight tensors
        //   - Parallel execution of Q, K, V projections
        //   - RoPE on Q and K
        //   - Attention computation
        //   - Output projection
        //   - Residual add

        // For demonstration, create a simplified graph structure:
        // attn_norm -> [Q_proj, K_proj, V_proj] -> [Q_rope, K_rope] -> attention -> out_proj -> residual

        // This shows the DAG structure without actual GEMM implementation
        LOG_DEBUG("[LayerExecutor] Building attention graph with " << params.seq_len
                                                                   << " tokens, " << params.n_heads << " heads");

        // RoPE stage (operates on Q and K after projection)
        // In practice, this would be split into Q_rope and K_rope running in parallel

        // Residual add at the end
        if (params.residual)
        {
            ResidualAddStage::Params res_params;
            res_params.input = params.output; // Attention output
            res_params.residual = params.residual;
            res_params.output = params.output; // In-place
            res_params.num_elements = static_cast<size_t>(params.seq_len) * params.d_model;

            auto residual_stage = ComputeStageFactory::createResidualAdd(res_params, backend);
            graph.addNode("attn_residual", std::move(residual_stage), device);

            if (params.attn_norm)
            {
                graph.addDependency("attn_residual", "attn_norm");
            }
        }

        return graph;
    }

    // =============================================================================
    // FFN Graph Building
    // =============================================================================

    ComputeGraph LayerExecutor::buildFFNGraph(const FFNParams &params)
    {
        ComputeGraph graph;

        if (!params.input || !params.output)
        {
            LOG_ERROR("[LayerExecutor] Invalid FFN params: null input/output");
            return graph;
        }

        const int device = config_.default_device;
        const auto backend = ComputeBackendType::CPU_OPENBLAS;
        const int seq_len = params.seq_len;
        const int d_model = params.d_model;
        const int d_ff = params.d_ff;

        // Pre-FFN norm
        if (params.ffn_norm)
        {
            RMSNormStage::Params norm_params;
            norm_params.input = params.input;
            norm_params.gamma = params.ffn_norm;
            norm_params.seq_len = seq_len;
            norm_params.hidden_dim = d_model;
            norm_params.eps = params.rms_norm_eps;

            graph.addNode("ffn_norm",
                          ComputeStageFactory::createRMSNorm(norm_params, backend),
                          device);
        }

        // Gate and Up projections are independent (can run in parallel)
        // [seq_len, d_model] @ [d_model, d_ff].T = [seq_len, d_ff]
        // These are GEMM placeholders

        // For actual implementation, you would:
        // graph.addNode("ffn_gate", createGEMM(...), device);
        // graph.addNode("ffn_up", createGEMM(...), device);
        // graph.addDependency("ffn_gate", "ffn_norm");
        // graph.addDependency("ffn_up", "ffn_norm");

        // SwiGLU: silu(gate) * up
        // Get temporary buffer for intermediate results
        float *gate_buffer = getTemporaryBuffer(static_cast<size_t>(seq_len) * d_ff);
        float *up_buffer = gate_buffer + seq_len * d_ff; // Adjacent in buffer

        if (gate_buffer && up_buffer)
        {
            SwiGLUStage::Params swiglu_params;
            swiglu_params.gate = gate_buffer; // Would come from GEMM in real impl
            swiglu_params.up = up_buffer;
            swiglu_params.output = gate_buffer; // Reuse gate buffer
            swiglu_params.seq_len = seq_len;
            swiglu_params.intermediate_dim = d_ff;

            graph.addNode("ffn_swiglu",
                          ComputeStageFactory::createSwiGLU(swiglu_params, backend),
                          device);

            if (params.ffn_norm)
            {
                graph.addDependency("ffn_swiglu", "ffn_norm");
            }
        }

        // Down projection: [seq_len, d_ff] @ [d_ff, d_model].T = [seq_len, d_model]
        // Another GEMM placeholder

        // Residual add
        if (params.residual)
        {
            ResidualAddStage::Params res_params;
            res_params.input = params.output;
            res_params.residual = params.residual;
            res_params.output = params.output;
            res_params.num_elements = static_cast<size_t>(seq_len) * d_model;

            auto residual_stage = ComputeStageFactory::createResidualAdd(res_params, backend);
            graph.addNode("ffn_residual", std::move(residual_stage), device);
            graph.addDependency("ffn_residual", "ffn_swiglu");
        }

        LOG_DEBUG("[LayerExecutor] Built FFN graph: " << graph.size() << " stages");

        return graph;
    }

    // =============================================================================
    // MoE Graph Building
    // =============================================================================

    ComputeGraph LayerExecutor::buildMoEGraph(const MoEParams &params)
    {
        ComputeGraph graph;

        if (!params.input || !params.output)
        {
            LOG_ERROR("[LayerExecutor] Invalid MoE params: null input/output");
            return graph;
        }

        const int default_device = config_.default_device;
        const auto backend = ComputeBackendType::CPU_OPENBLAS;
        const int seq_len = params.seq_len;
        const int d_model = params.d_model;
        const int n_experts = params.n_experts;
        const int top_k = params.top_k;

        // Pre-MoE norm
        if (params.ffn_norm)
        {
            RMSNormStage::Params norm_params;
            norm_params.input = params.input;
            norm_params.gamma = params.ffn_norm;
            norm_params.seq_len = seq_len;
            norm_params.hidden_dim = d_model;
            norm_params.eps = params.rms_norm_eps;

            graph.addNode("moe_norm",
                          ComputeStageFactory::createRMSNorm(norm_params, backend),
                          default_device);
        }

        // Router: compute expert scores for each token
        // Output: [seq_len, n_experts] scores
        {
            MoERouterStage::Params router_params;
            router_params.hidden = params.input;
            router_params.gate_weights = params.router_weight;
            router_params.router_logits = nullptr; // Would be allocated in real impl
            router_params.seq_len = seq_len;
            router_params.d_model = d_model;
            router_params.num_experts = n_experts;

            graph.addNode("moe_router",
                          ComputeStageFactory::createMoERouter(router_params, backend),
                          default_device);

            if (params.ffn_norm)
            {
                graph.addDependency("moe_router", "moe_norm");
            }
        }

        // Expert FFNs: Each expert processes its assigned tokens
        // With expert parallelism, different experts can run on different devices
        for (int e = 0; e < n_experts; ++e)
        {
            int expert_device = default_device;
            if (params.enable_expert_parallel && params.expert_device_map)
            {
                expert_device = params.expert_device_map[e];
            }

            MoEExpertStage::Params expert_params;
            expert_params.expert_id = e;
            expert_params.input = params.input;
            expert_params.output = nullptr; // Placeholder - would be allocated
            // Note: MoEExpertStage expects TensorBase*, but we have float* const*
            // In a real implementation, we'd have proper tensor wrappers
            expert_params.gate_w = nullptr;        // Placeholder
            expert_params.up_w = nullptr;          // Placeholder
            expert_params.down_w = nullptr;        // Placeholder
            expert_params.token_indices = nullptr; // Would come from router
            expert_params.d_model = d_model;
            expert_params.intermediate_dim = params.d_ff;

            std::string node_name = "expert_" + std::to_string(e);
            graph.addNode(node_name,
                          ComputeStageFactory::createMoEExpert(expert_params, backend),
                          expert_device);
            graph.addDependency(node_name, "moe_router");
        }

        // Combine: Weighted sum of expert outputs per token
        {
            MoECombineStage::Params combine_params;
            combine_params.expert_outputs = nullptr;   // Placeholder
            combine_params.expert_weights = nullptr;   // From router
            combine_params.token_expert_map = nullptr; // From router
            combine_params.output = params.output;
            combine_params.seq_len = seq_len;
            combine_params.d_model = d_model;
            combine_params.top_k = top_k;

            graph.addNode("moe_combine",
                          ComputeStageFactory::createMoECombine(combine_params, backend),
                          default_device);

            // Combine depends on all experts completing
            for (int e = 0; e < n_experts; ++e)
            {
                graph.addDependency("moe_combine", "expert_" + std::to_string(e));
            }
        }

        // Residual add
        if (params.residual)
        {
            ResidualAddStage::Params res_params;
            res_params.input = params.output;
            res_params.residual = params.residual;
            res_params.output = params.output;
            res_params.num_elements = static_cast<size_t>(seq_len) * d_model;

            graph.addNode("moe_residual",
                          ComputeStageFactory::createResidualAdd(res_params, backend),
                          default_device);
            graph.addDependency("moe_residual", "moe_combine");
        }

        LOG_DEBUG("[LayerExecutor] Built MoE graph: " << graph.size()
                                                      << " stages, " << n_experts << " experts");

        return graph;
    }

    // =============================================================================
    // Execution
    // =============================================================================

    bool LayerExecutor::execute(ComputeGraph &graph, IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[LayerExecutor] Null device context");
            return false;
        }

        if (graph.size() == 0)
        {
            return true; // Empty graph is success
        }

        graph.reset();

        switch (config_.mode)
        {
        case ExecutionMode::SEQUENTIAL:
            return executeSequential(graph, ctx);
        case ExecutionMode::PARALLEL:
            return executeParallel(graph, ctx);
        case ExecutionMode::PIPELINED:
            LOG_WARN("[LayerExecutor] Pipelined mode not yet implemented, using sequential");
            return executeSequential(graph, ctx);
        default:
            LOG_ERROR("[LayerExecutor] Unknown execution mode");
            return false;
        }
    }

    bool LayerExecutor::executeSequential(ComputeGraph &graph, IDeviceContext *ctx)
    {
        auto order = graph.getExecutionOrder();

        auto total_start = std::chrono::high_resolution_clock::now();

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[LayerExecutor] Invalid node: " << name);
                return false;
            }

            if (!executeNode(*node, ctx))
            {
                LOG_ERROR("[LayerExecutor] Stage failed: " << name);
                return false;
            }

            graph.markCompleted(name);
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        stats_.total_time_ms += total_ms;
        stats_.total_stages_executed += order.size();
        stats_.total_flops += graph.totalEstimatedFlops();

        return true;
    }

    bool LayerExecutor::executeParallel(ComputeGraph &graph, IDeviceContext *ctx)
    {
        auto total_start = std::chrono::high_resolution_clock::now();

        while (!graph.allCompleted())
        {
            auto ready = graph.getReadyNodes();

            if (ready.empty() && !graph.allCompleted())
            {
                LOG_ERROR("[LayerExecutor] Deadlock detected in graph");
                return false;
            }

            // Execute all ready nodes
            // In a true parallel implementation, this would dispatch to different threads
            // For now, execute sequentially (parallel execution requires more infrastructure)
            for (const auto &name : ready)
            {
                auto *node = graph.getNode(name);
                if (!node || !node->stage)
                    continue;

                if (!executeNode(*node, ctx))
                {
                    LOG_ERROR("[LayerExecutor] Stage failed: " << name);
                    return false;
                }

                graph.markCompleted(name);
            }
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        stats_.total_time_ms += total_ms;
        stats_.total_flops += graph.totalEstimatedFlops();

        return true;
    }

    bool LayerExecutor::executeMultiDevice(
        ComputeGraph &graph,
        const std::unordered_map<int, IDeviceContext *> &contexts)
    {

        if (contexts.empty())
        {
            LOG_ERROR("[LayerExecutor] No device contexts provided");
            return false;
        }

        // Default context for nodes without explicit device assignment
        IDeviceContext *default_ctx = nullptr;
        for (const auto &[idx, ctx] : contexts)
        {
            default_ctx = ctx;
            break;
        }

        graph.reset();
        auto order = graph.getExecutionOrder();

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage)
                continue;

            // Find appropriate context for this node's device
            IDeviceContext *ctx = default_ctx;
            if (node->device_idx >= 0)
            {
                auto it = contexts.find(node->device_idx);
                if (it != contexts.end())
                {
                    ctx = it->second;
                }
            }

            if (!executeNode(*node, ctx))
            {
                LOG_ERROR("[LayerExecutor] Stage failed: " << name << " on device " << node->device_idx);
                return false;
            }

            graph.markCompleted(name);
        }

        return true;
    }

    bool LayerExecutor::executeNode(ComputeNode &node, IDeviceContext *ctx)
    {
        if (!node.stage)
        {
            LOG_ERROR("[LayerExecutor] Node '" << node.name << "' has no stage");
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        bool success = node.stage->execute(ctx);

        if (config_.enable_profiling)
        {
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            stats_.stage_times_ms[node.name] = ms;

            LOG_DEBUG("[LayerExecutor] Stage '" << node.name << "' took " << ms << " ms");
        }

        return success;
    }

    // =============================================================================
    // Workspace Management
    // =============================================================================

    float *LayerExecutor::getTemporaryBuffer(size_t elements)
    {
        size_t needed = elements * 2; // Double for gate+up buffers

        if (needed > temp_buffer_size_)
        {
            temp_buffer_.resize(needed);
            temp_buffer_size_ = needed;
        }

        return temp_buffer_.data();
    }

} // namespace llaminar2
