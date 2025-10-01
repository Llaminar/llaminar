#include "graph_compute.h"
#include "tensors/tensor_factory.h"
#include "logger.h"
#include "utils/perf_counters.h" // For PerfMatmulPhaseScope layer index propagation
#include <iostream>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <stack>
#include <cmath>
#include <cblas.h> // For BLAS operations

// ComputeNode implementation
ComputeNode::ComputeNode(const std::string &name, const std::string &operation_type)
    : name_(name), operation_type_(operation_type), executed_(false) {}

void ComputeNode::addInput(std::shared_ptr<ComputeNode> input)
{
    inputs_.push_back(input);
    input->outputs_.push_back(shared_from_this());
}

void ComputeNode::addOutput(std::shared_ptr<ComputeNode> output)
{
    outputs_.push_back(output);
    output->inputs_.push_back(shared_from_this());
}

// ComputeGraph implementation
ComputeGraph::ComputeGraph(const std::string &name) : name_(name) {}

void ComputeGraph::addNode(std::shared_ptr<ComputeNode> node)
{
    if (!node)
        return;

    // Check if node already exists
    if (node_map_.find(node->getName()) != node_map_.end())
    {
        std::cerr << "Warning: Node '" << node->getName() << "' already exists in graph" << std::endl;
        return;
    }

    nodes_.push_back(node);
    node_map_[node->getName()] = node;
}

std::shared_ptr<ComputeNode> ComputeGraph::getNode(const std::string &name) const
{
    auto it = node_map_.find(name);
    return (it != node_map_.end()) ? it->second : nullptr;
}

bool ComputeGraph::execute()
{
    if (!validate())
    {
        std::cerr << "Graph validation failed before execution" << std::endl;
        return false;
    }

    // Get execution order through topological sort
    auto execution_order = getExecutionOrder();
    if (execution_order.empty() && !nodes_.empty())
    {
        std::cerr << "Failed to determine execution order (possible cycle detected)" << std::endl;
        return false;
    }

    // Execute nodes in order
    for (auto &node : execution_order)
    {
        std::cout << "Executing node: " << node->getName() << " (" << node->getOperationType() << ")" << std::endl;

        if (!node->execute())
        {
            std::cerr << "Failed to execute node: " << node->getName() << std::endl;
            return false;
        }

        node->markExecuted();
    }

    std::cout << "Graph execution completed successfully" << std::endl;
    return true;
}

bool ComputeGraph::validate() const
{
    // Check for cycles
    if (hasCycles())
    {
        std::cerr << "Graph validation failed: Cycle detected" << std::endl;
        return false;
    }

    // Validate each node
    for (const auto &node : nodes_)
    {
        if (!node->validate())
        {
            std::cerr << "Graph validation failed: Node '" << node->getName() << "' validation failed" << std::endl;
            return false;
        }
    }

    return true;
}

void ComputeGraph::reset()
{
    for (auto &node : nodes_)
    {
        node->reset();
    }
}

void ComputeGraph::clear()
{
    nodes_.clear();
    node_map_.clear();
}

ComputeGraph &ComputeGraph::operator+(std::shared_ptr<ComputeNode> node)
{
    addNode(node);
    return *this;
}

ComputeGraph &ComputeGraph::operator+=(std::shared_ptr<ComputeNode> node)
{
    addNode(node);
    return *this;
}

std::vector<std::shared_ptr<ComputeNode>> ComputeGraph::getExecutionOrder() const
{
    return topologicalSort();
}

void ComputeGraph::printGraph() const
{
    std::cout << "\n=== Compute Graph: " << name_ << " ===" << std::endl;
    std::cout << "Nodes: " << nodes_.size() << std::endl;

    for (const auto &node : nodes_)
    {
        std::cout << "  " << node->getName() << " (" << node->getOperationType() << ")";
        std::cout << " - Inputs: " << node->getInputs().size();
        std::cout << ", Outputs: " << node->getOutputs().size() << std::endl;
    }
    std::cout << "=========================" << std::endl;
}

bool ComputeGraph::hasCycles() const
{
    std::unordered_map<std::shared_ptr<ComputeNode>, int> state; // 0=unvisited, 1=visiting, 2=visited

    for (const auto &node : nodes_)
    {
        if (state[node] == 0)
        {
            std::vector<std::shared_ptr<ComputeNode>> dummy;
            try
            {
                dfsVisit(node, state, dummy);
            }
            catch (const std::runtime_error &)
            {
                return true; // Cycle detected
            }
        }
    }
    return false;
}

std::vector<std::shared_ptr<ComputeNode>> ComputeGraph::topologicalSort() const
{
    std::unordered_map<std::shared_ptr<ComputeNode>, int> state; // 0=unvisited, 1=visiting, 2=visited
    std::vector<std::shared_ptr<ComputeNode>> result;

    for (const auto &node : nodes_)
    {
        if (state[node] == 0)
        {
            try
            {
                dfsVisit(node, state, result);
            }
            catch (const std::runtime_error &)
            {
                return {}; // Cycle detected, return empty vector
            }
        }
    }

    std::reverse(result.begin(), result.end());
    return result;
}

void ComputeGraph::dfsVisit(std::shared_ptr<ComputeNode> node,
                            std::unordered_map<std::shared_ptr<ComputeNode>, int> &state,
                            std::vector<std::shared_ptr<ComputeNode>> &result) const
{
    state[node] = 1; // Mark as visiting

    for (const auto &output : node->getOutputs())
    {
        if (state[output] == 1)
        {
            throw std::runtime_error("Cycle detected");
        }
        if (state[output] == 0)
        {
            dfsVisit(output, state, result);
        }
    }

    state[node] = 2; // Mark as visited
    result.push_back(node);
}

// MatMulNode implementation
MatMulNode::MatMulNode(const std::string &name, int m, int n, int k)
    : ComputeNode(name, "MatMul"), m_(m), n_(n), k_(k) {}

bool MatMulNode::execute()
{
    std::cout << "  MatMul: " << m_ << "x" << n_ << "x" << k_ << std::endl;
    // TODO: Call COSMA kernel through KernelManager
    return true;
}

bool MatMulNode::validate() const
{
    return m_ > 0 && n_ > 0 && k_ > 0;
}

// DataNode implementation
DataNode::DataNode(const std::string &name, const std::vector<double> &data)
    : ComputeNode(name, "Data"), data_(data) {}

bool DataNode::execute()
{
    std::cout << "  Data node: " << data_.size() << " elements" << std::endl;
    return true;
}

bool DataNode::validate() const
{
    return !data_.empty();
}

// OutputNode implementation
OutputNode::OutputNode(const std::string &name)
    : ComputeNode(name, "Output") {}

bool OutputNode::execute()
{
    std::cout << "  Output node: Collecting results" << std::endl;
    // TODO: Collect results from input nodes
    return true;
}

bool OutputNode::validate() const
{
    return !inputs_.empty(); // Must have at least one input
}

// Transformer node implementations

TransformerNode::TransformerNode(const std::string &name, const std::string &operation_type)
    : ComputeNode(name, operation_type) {}

// RMSNormNode implementation
RMSNormNode::RMSNormNode(const std::string &name, std::shared_ptr<llaminar::TensorBase> weight, float eps)
    : TransformerNode(name, "RMSNorm"), weight_(weight), eps_(eps) {}

bool RMSNormNode::execute()
{
    if (!input_tensor_ || !output_tensor_ || !weight_)
    {
        LOG_ERROR("RMSNorm: Missing tensors");
        return false;
    }

    // Create and use RMSNormKernel directly
    // llaminar::RMSNormKernel kernel;
    // kernel.setEpsilon(eps_);

    // TODO: Replace with MPI-based RMS normalization
    LOG_WARN("RMSNormNode::execute() - RMSNormKernel removed, using placeholder");
    return true;

    /*

    */

    return false; // Placeholder
}

bool RMSNormNode::validate() const
{
    return weight_ && weight_->shape().size() == 1;
}

// AttentionNode implementation
AttentionNode::AttentionNode(const std::string &name,
                             std::shared_ptr<llaminar::TensorBase> q_weight,
                             std::shared_ptr<llaminar::TensorBase> k_weight,
                             std::shared_ptr<llaminar::TensorBase> v_weight,
                             std::shared_ptr<llaminar::TensorBase> out_weight,
                             std::shared_ptr<llaminar::TensorBase> k_cache,
                             std::shared_ptr<llaminar::TensorBase> v_cache,
                             int n_head, int n_head_kv, int n_past)
    : TransformerNode(name, "Attention"),
      q_weight_(q_weight), k_weight_(k_weight), v_weight_(v_weight),
      out_weight_(out_weight), k_cache_(k_cache), v_cache_(v_cache),
      n_head_(n_head), n_head_kv_(n_head_kv), n_past_(n_past)
{
    // Calculate head_dim from q_weight dimensions
    // Assuming q_weight shape is [d_model, n_head * head_dim]
    if (q_weight_ && q_weight_->shape().size() >= 2)
    {
        head_dim_ = q_weight_->shape()[1] / n_head_;
    }
    else
    {
        head_dim_ = 64; // Default fallback
    }
}

bool AttentionNode::execute()
{
    if (!input_tensor_ || !output_tensor_)
    {
        LOG_ERROR("Attention: Missing input/output tensors");
        return false;
    }

    if (!q_weight_ || !k_weight_ || !v_weight_ || !out_weight_ || !k_cache_ || !v_cache_)
    {
        LOG_ERROR("Attention: Missing weight or cache tensors");
        return false;
    }

    // Create and use AttentionKernel
    // llaminar::AttentionKernel kernel(n_head_, n_head_kv_, head_dim_);

    // TODO: Replace with MPI-based attention
    LOG_WARN("AttentionNode::execute() - AttentionKernel removed, using placeholder");
    return true;

    /*

    // Configure kernel parameters
    kernel.setSequencePosition(n_past_);

    std::vector<std::shared_ptr<llaminar::TensorBase>> inputs = {
        input_tensor_,
        q_weight_,
        k_weight_,
        v_weight_,
        out_weight_,
        k_cache_,
        v_cache_};
    std::vector<std::shared_ptr<llaminar::TensorBase>> outputs = {output_tensor_};

    bool success = kernel.execute(inputs, outputs);

    if (success)
    {
        */
    return false; // Placeholder
}

bool AttentionNode::validate() const
{
    return q_weight_ && k_weight_ && v_weight_ && out_weight_ &&
           k_cache_ && v_cache_ &&
           n_head_ > 0 && n_head_kv_ > 0;
}

// MLPNode implementation
MLPNode::MLPNode(const std::string &name,
                 std::shared_ptr<llaminar::TensorBase> gate_weight,
                 std::shared_ptr<llaminar::TensorBase> up_weight,
                 std::shared_ptr<llaminar::TensorBase> down_weight)
    : TransformerNode(name, "MLP"),
      gate_weight_(gate_weight), up_weight_(up_weight), down_weight_(down_weight) {}

bool MLPNode::execute()
{
    if (!input_tensor_ || !output_tensor_)
    {
        LOG_ERROR("MLP: Missing input/output tensors");
        return false;
    }

    if (!gate_weight_ || !up_weight_ || !down_weight_)
    {
        LOG_ERROR("MLP: Missing weight tensors");
        return false;
    }

    // Create and use MLPKernel
    // llaminar::MLPKernel kernel;

    // TODO: Replace with MPI-based MLP
    LOG_WARN("MLPNode::execute() - MLPKernel removed, using placeholder");
    return true;

    /*
    // std::vector<std::shared_ptr<llaminar::TensorBase>> inputs = {input_tensor_, gate_weight_, up_weight_, down_weight_};
    // std::vector<std::shared_ptr<llaminar::TensorBase>> outputs = {output_tensor_};

    // bool success = kernel.execute(inputs, outputs);
    bool success = false; // Placeholder

    if (success)
    {
        LOG_DEBUG("MLP executed successfully");
    }
    else
    {
        LOG_ERROR("MLP kernel execution failed");
    }

    return success;
    */
}

bool MLPNode::validate() const
{
    return gate_weight_ && up_weight_ && down_weight_;
}

// TransformerBlockNode implementation
TransformerBlockNode::TransformerBlockNode(const std::string &name, int layer_idx,
                                           std::shared_ptr<RMSNormNode> attn_norm,
                                           std::shared_ptr<AttentionNode> attention,
                                           std::shared_ptr<RMSNormNode> ffn_norm,
                                           std::shared_ptr<MLPNode> mlp)
    : TransformerNode(name, "TransformerBlock"), layer_idx_(layer_idx),
      attn_norm_(attn_norm), attention_(attention), ffn_norm_(ffn_norm), mlp_(mlp) {}

bool TransformerBlockNode::execute()
{
    if (!input_tensor_ || !output_tensor_)
    {
        LOG_ERROR("TransformerBlock: Missing input/output tensors");
        return false;
    }

    const auto &input_shape = input_tensor_->shape();
    int seq_len = input_shape[0];
    int n_embd = input_shape[1];

    // Create temporary tensors for intermediate results
    auto attn_input = llaminar::TensorFactory::create_simple(input_shape);
    auto attn_output = llaminar::TensorFactory::create_simple(input_shape);
    auto ffn_input = llaminar::TensorFactory::create_simple(input_shape);
    auto ffn_output = llaminar::TensorFactory::create_simple(input_shape);

    // Attention path: norm -> attention -> residual
    attn_norm_->setInput(input_tensor_);
    attn_norm_->setOutput(attn_input);
    attn_norm_->execute();

    attention_->setInput(attn_input);
    attention_->setOutput(attn_output);
    // Establish layer index context for downstream matmuls inside attention
    {
        llaminar::PerfMatmulPhaseScope layer_ctx(2, 0, layer_idx_); // layer_type=2 (ATTN), phase=0 (block-level)
        attention_->execute();
    }

    // Residual connection: input + attention_output
    const float *input_data = input_tensor_->data();
    const float *attn_data = attn_output->data();
    float *ffn_input_data = ffn_input->data();

    for (int i = 0; i < seq_len * n_embd; ++i)
    {
        ffn_input_data[i] = input_data[i] + attn_data[i];
    }

    // MLP path: norm -> mlp -> residual
    ffn_norm_->setInput(ffn_input);
    ffn_norm_->setOutput(attn_input); // Reuse attn_input tensor
    ffn_norm_->execute();

    mlp_->setInput(attn_input);
    mlp_->setOutput(ffn_output);
    {
        llaminar::PerfMatmulPhaseScope layer_ctx(1, 0, layer_idx_); // layer_type=1 (MLP), phase=0
        mlp_->execute();
    }

    // Final residual connection: ffn_input + mlp_output
    const float *mlp_data = ffn_output->data();
    float *output_data = output_tensor_->data();

    for (int i = 0; i < seq_len * n_embd; ++i)
    {
        output_data[i] = ffn_input_data[i] + mlp_data[i];
    }

    LOG_DEBUG("TransformerBlock " << layer_idx_ << " executed: " << seq_len << "x" << n_embd);
    return true;
}

bool TransformerBlockNode::validate() const
{
    return attn_norm_ && attention_ && ffn_norm_ && mlp_;
}

// EmbeddingNode implementation
EmbeddingNode::EmbeddingNode(const std::string &name, std::shared_ptr<llaminar::TensorBase> embedding_weights)
    : TransformerNode(name, "Embedding"), embedding_weights_(embedding_weights) {}

bool EmbeddingNode::execute()
{
    if (!output_tensor_ || !embedding_weights_ || token_ids_.empty())
    {
        LOG_ERROR("Embedding: Missing tensors or token IDs");
        return false;
    }

    const auto &weight_shape = embedding_weights_->shape();
    if (weight_shape.size() != 2)
    {
        LOG_ERROR("Embedding: Invalid weight shape");
        return false;
    }

    int vocab_size = weight_shape[0];
    int n_embd = weight_shape[1];
    int seq_len = token_ids_.size();

    const float *weight_data = embedding_weights_->data();
    float *output_data = output_tensor_->data();

    // Lookup embeddings for each token
    for (int s = 0; s < seq_len; ++s)
    {
        int token_id = token_ids_[s];
        if (token_id < 0 || token_id >= vocab_size)
        {
            LOG_ERROR("Embedding: Invalid token ID " << token_id);
            return false;
        }

        // Copy embedding for this token
        const float *token_emb = weight_data + token_id * n_embd;
        float *output_pos = output_data + s * n_embd;
        std::copy(token_emb, token_emb + n_embd, output_pos);
    }

    LOG_DEBUG("Embedding executed: " << seq_len << " tokens -> " << seq_len << "x" << n_embd);
    return true;
}

bool EmbeddingNode::validate() const
{
    return embedding_weights_ && embedding_weights_->shape().size() == 2;
}

// LinearNode implementation
LinearNode::LinearNode(const std::string &name,
                       std::shared_ptr<llaminar::TensorBase> weight,
                       std::shared_ptr<llaminar::TensorBase> bias)
    : TransformerNode(name, "Linear"), weight_(weight), bias_(bias) {}

bool LinearNode::execute()
{
    if (!input_tensor_ || !output_tensor_ || !weight_)
    {
        LOG_ERROR("Linear: Missing tensors");
        return false;
    }

    const auto &input_shape = input_tensor_->shape();
    const auto &weight_shape = weight_->shape();

    if (input_shape.size() != 2 || weight_shape.size() != 2)
    {
        LOG_ERROR("Linear: Invalid tensor shapes");
        return false;
    }

    int seq_len = input_shape[0];
    int in_features = input_shape[1];
    int out_features = weight_shape[1];

    if (weight_shape[0] != in_features)
    {
        LOG_ERROR("Linear: Weight dimension mismatch");
        return false;
    }

    const float *input_data = input_tensor_->data();
    const float *weight_data = weight_->data();
    float *output_data = output_tensor_->data();

    // Matrix multiplication: output = input * weight^T
    // Using CBLAS for efficient computation
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_features, in_features,
                1.0f, input_data, in_features,
                weight_data, in_features,
                0.0f, output_data, out_features);

    // Add bias if present
    if (bias_)
    {
        const float *bias_data = bias_->data();
        for (int s = 0; s < seq_len; ++s)
        {
            for (int o = 0; o < out_features; ++o)
            {
                output_data[s * out_features + o] += bias_data[o];
            }
        }
    }

    LOG_DEBUG("Linear executed: " << seq_len << "x" << in_features << " -> " << seq_len << "x" << out_features);
    return true;
}

bool LinearNode::validate() const
{
    return weight_ && weight_->shape().size() == 2;
}

// KernelNode implementation
KernelNode::KernelNode(const std::string &name, std::unique_ptr<llaminar::KernelBase> kernel)
    : ComputeNode(name, kernel ? kernel->getKernelType() : "Unknown"), kernel_(std::move(kernel))
{
}

bool KernelNode::execute()
{
    if (!kernel_)
    {
        LOG_ERROR("KernelNode " << name_ << ": No kernel implementation");
        return false;
    }

    if (input_tensors_.size() != kernel_->getExpectedInputCount())
    {
        LOG_ERROR("KernelNode " << name_ << ": Expected " << kernel_->getExpectedInputCount()
                                << " inputs, got " << input_tensors_.size());
        return false;
    }

    if (output_tensors_.size() != kernel_->getExpectedOutputCount())
    {
        LOG_ERROR("KernelNode " << name_ << ": Expected " << kernel_->getExpectedOutputCount()
                                << " outputs, got " << output_tensors_.size());
        return false;
    }

    // Validate tensors before execution
    if (!kernel_->validate(input_tensors_, output_tensors_))
    {
        LOG_ERROR("KernelNode " << name_ << ": Tensor validation failed");
        return false;
    }

    // Execute the kernel
    bool success = kernel_->execute(input_tensors_, output_tensors_);

    if (success)
    {
        LOG_DEBUG("KernelNode " << name_ << " (" << kernel_->getKernelType() << ") executed successfully");
    }
    else
    {
        LOG_ERROR("KernelNode " << name_ << " (" << kernel_->getKernelType() << ") execution failed");
    }

    return success;
}

bool KernelNode::validate() const
{
    if (!kernel_)
    {
        return false;
    }

    if (input_tensors_.size() != kernel_->getExpectedInputCount() ||
        output_tensors_.size() != kernel_->getExpectedOutputCount())
    {
        return false;
    }

    return kernel_->validate(input_tensors_, output_tensors_);
}

void KernelNode::setInputTensors(const std::vector<std::shared_ptr<llaminar::TensorBase>> &inputs)
{
    input_tensors_ = inputs;
}

void KernelNode::setOutputTensors(const std::vector<std::shared_ptr<llaminar::TensorBase>> &outputs)
{
    output_tensors_ = outputs;
}

void KernelNode::addInputTensor(std::shared_ptr<llaminar::TensorBase> input)
{
    input_tensors_.push_back(input);
}

void KernelNode::addOutputTensor(std::shared_ptr<llaminar::TensorBase> output)
{
    output_tensors_.push_back(output);
}

const std::string &KernelNode::getKernelType() const
{
    static const std::string unknown = "Unknown";
    if (kernel_)
    {
        static std::string type = kernel_->getKernelType();
        return type;
    }
    return unknown;
}