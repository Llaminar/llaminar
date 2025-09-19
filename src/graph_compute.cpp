#include "graph_compute.h"
#include "kernels/RMSNormKernel.h"
#include "kernels/LinearKernel.h"
#include "kernels/EmbeddingKernel.h"
#include "kernels/MLPKernel.h"
#include "kernels/AttentionKernel.h"
#include "logger.h"
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
RMSNormNode::RMSNormNode(const std::string &name, std::shared_ptr<Tensor> weight, float eps)
    : TransformerNode(name, "RMSNorm"), weight_(weight), eps_(eps) {}

bool RMSNormNode::execute()
{
    if (!input_tensor_ || !output_tensor_ || !weight_)
    {
        LOG_ERROR("RMSNorm: Missing tensors");
        return false;
    }

    // Create and use RMSNormKernel directly
    llaminar::RMSNormKernel kernel;
    kernel.setEpsilon(eps_);

    std::vector<std::shared_ptr<Tensor>> inputs = {input_tensor_, weight_};
    std::vector<std::shared_ptr<Tensor>> outputs = {output_tensor_};

    bool success = kernel.execute(inputs, outputs);

    if (success)
    {
        LOG_DEBUG("RMSNorm executed successfully");
    }
    else
    {
        LOG_ERROR("RMSNorm kernel execution failed");
    }

    return success;
}

bool RMSNormNode::validate() const
{
    return weight_ && weight_->shape.size() == 1;
}

// AttentionNode implementation
AttentionNode::AttentionNode(const std::string &name,
                             std::shared_ptr<Tensor> q_weight,
                             std::shared_ptr<Tensor> k_weight,
                             std::shared_ptr<Tensor> v_weight,
                             std::shared_ptr<Tensor> out_weight,
                             std::shared_ptr<Tensor> k_cache,
                             std::shared_ptr<Tensor> v_cache,
                             int n_head, int n_head_kv, int n_past)
    : TransformerNode(name, "Attention"),
      q_weight_(q_weight), k_weight_(k_weight), v_weight_(v_weight),
      out_weight_(out_weight), k_cache_(k_cache), v_cache_(v_cache),
      n_head_(n_head), n_head_kv_(n_head_kv), n_past_(n_past) {}

bool AttentionNode::execute()
{
    if (!input_tensor_ || !output_tensor_)
    {
        LOG_ERROR("Attention: Missing input/output tensors");
        return false;
    }

    const auto &input_shape = input_tensor_->shape;
    if (input_shape.size() != 2)
    {
        LOG_ERROR("Attention: Invalid input shape");
        return false;
    }

    int seq_len = input_shape[0];
    int n_embd = input_shape[1];

    const float *input_data = input_tensor_->ptr();
    float *output_data = output_tensor_->ptr();

    // For now, implement a simplified attention (placeholder)
    // In a full implementation, this would include:
    // 1. Q, K, V projections with weights and biases
    // 2. Reshape to multi-head format
    // 3. Scaled dot-product attention with KV cache
    // 4. Output projection

    LOG_INFO("Attention: Simplified implementation - copying input to output");
    std::copy(input_data, input_data + seq_len * n_embd, output_data);

    LOG_DEBUG("Attention executed: " << seq_len << "x" << n_embd << " with " << n_head_ << " heads");
    return true;
}

bool AttentionNode::validate() const
{
    return q_weight_ && k_weight_ && v_weight_ && out_weight_ &&
           k_cache_ && v_cache_ &&
           n_head_ > 0 && n_head_kv_ > 0;
}

// MLPNode implementation
MLPNode::MLPNode(const std::string &name,
                 std::shared_ptr<Tensor> gate_weight,
                 std::shared_ptr<Tensor> up_weight,
                 std::shared_ptr<Tensor> down_weight)
    : TransformerNode(name, "MLP"),
      gate_weight_(gate_weight), up_weight_(up_weight), down_weight_(down_weight) {}

bool MLPNode::execute()
{
    if (!input_tensor_ || !output_tensor_)
    {
        LOG_ERROR("MLP: Missing input/output tensors");
        return false;
    }

    const auto &input_shape = input_tensor_->shape;
    if (input_shape.size() != 2)
    {
        LOG_ERROR("MLP: Invalid input shape");
        return false;
    }

    int seq_len = input_shape[0];
    int n_embd = input_shape[1];

    const float *input_data = input_tensor_->ptr();
    float *output_data = output_tensor_->ptr();

    // Simplified MLP implementation (placeholder)
    // Full implementation would include:
    // 1. Gate projection: gate = input * gate_weight
    // 2. Up projection: up = input * up_weight
    // 3. SiLU activation: gate = gate * sigmoid(gate)
    // 4. Element-wise multiply: intermediate = gate * up
    // 5. Down projection: output = intermediate * down_weight

    LOG_INFO("MLP: Simplified implementation - copying input to output");
    std::copy(input_data, input_data + seq_len * n_embd, output_data);

    LOG_DEBUG("MLP executed: " << seq_len << "x" << n_embd);
    return true;
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

    const auto &input_shape = input_tensor_->shape;
    int seq_len = input_shape[0];
    int n_embd = input_shape[1];

    // Create temporary tensors for intermediate results
    auto attn_input = std::make_shared<Tensor>(input_shape);
    auto attn_output = std::make_shared<Tensor>(input_shape);
    auto ffn_input = std::make_shared<Tensor>(input_shape);
    auto ffn_output = std::make_shared<Tensor>(input_shape);

    // Attention path: norm -> attention -> residual
    attn_norm_->setInput(input_tensor_);
    attn_norm_->setOutput(attn_input);
    attn_norm_->execute();

    attention_->setInput(attn_input);
    attention_->setOutput(attn_output);
    attention_->execute();

    // Residual connection: input + attention_output
    const float *input_data = input_tensor_->ptr();
    const float *attn_data = attn_output->ptr();
    float *ffn_input_data = ffn_input->ptr();

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
    mlp_->execute();

    // Final residual connection: ffn_input + mlp_output
    const float *mlp_data = ffn_output->ptr();
    float *output_data = output_tensor_->ptr();

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
EmbeddingNode::EmbeddingNode(const std::string &name, std::shared_ptr<Tensor> embedding_weights)
    : TransformerNode(name, "Embedding"), embedding_weights_(embedding_weights) {}

bool EmbeddingNode::execute()
{
    if (!output_tensor_ || !embedding_weights_ || token_ids_.empty())
    {
        LOG_ERROR("Embedding: Missing tensors or token IDs");
        return false;
    }

    const auto &weight_shape = embedding_weights_->shape;
    if (weight_shape.size() != 2)
    {
        LOG_ERROR("Embedding: Invalid weight shape");
        return false;
    }

    int vocab_size = weight_shape[0];
    int n_embd = weight_shape[1];
    int seq_len = token_ids_.size();

    const float *weight_data = embedding_weights_->ptr();
    float *output_data = output_tensor_->ptr();

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
    return embedding_weights_ && embedding_weights_->shape.size() == 2;
}

// LinearNode implementation
LinearNode::LinearNode(const std::string &name,
                       std::shared_ptr<Tensor> weight,
                       std::shared_ptr<Tensor> bias)
    : TransformerNode(name, "Linear"), weight_(weight), bias_(bias) {}

bool LinearNode::execute()
{
    if (!input_tensor_ || !output_tensor_ || !weight_)
    {
        LOG_ERROR("Linear: Missing tensors");
        return false;
    }

    const auto &input_shape = input_tensor_->shape;
    const auto &weight_shape = weight_->shape;

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

    const float *input_data = input_tensor_->ptr();
    const float *weight_data = weight_->ptr();
    float *output_data = output_tensor_->ptr();

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
        const float *bias_data = bias_->ptr();
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
    return weight_ && weight_->shape.size() == 2;
}