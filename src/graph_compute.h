#pragma once

#include "common.h"
#include "tensor.h" // For legacy Tensor struct
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>

// Bring Tensor from llaminar namespace to this scope for backward compatibility
using llaminar::Tensor;

// Forward declarations
class ComputeNode;

// Base class for all compute nodes in the graph
class ComputeNode : public std::enable_shared_from_this<ComputeNode>
{
public:
    ComputeNode(const std::string &name, const std::string &operation_type);
    virtual ~ComputeNode() = default;

    // Core functionality
    virtual bool execute() = 0;
    virtual bool validate() const = 0;

    // Graph connectivity
    void addInput(std::shared_ptr<ComputeNode> input);
    void addOutput(std::shared_ptr<ComputeNode> output);

    // Getters
    const std::string &getName() const { return name_; }
    const std::string &getOperationType() const { return operation_type_; }
    const std::vector<std::shared_ptr<ComputeNode>> &getInputs() const { return inputs_; }
    const std::vector<std::shared_ptr<ComputeNode>> &getOutputs() const { return outputs_; }

    // Execution state
    bool isExecuted() const { return executed_; }
    void markExecuted() { executed_ = true; }
    void reset() { executed_ = false; }

protected:
    std::string name_;
    std::string operation_type_;
    std::vector<std::shared_ptr<ComputeNode>> inputs_;
    std::vector<std::shared_ptr<ComputeNode>> outputs_;
    bool executed_;
};

// Compute graph class for managing execution flow
class ComputeGraph
{
public:
    ComputeGraph(const std::string &name = "DefaultGraph");
    ~ComputeGraph() = default;

    // Node management
    void addNode(std::shared_ptr<ComputeNode> node);
    std::shared_ptr<ComputeNode> getNode(const std::string &name) const;

    // Graph operations
    bool execute();
    bool validate() const;
    void reset();
    void clear();

    // Operator overloading for easy graph construction
    ComputeGraph &operator+(std::shared_ptr<ComputeNode> node);
    ComputeGraph &operator+=(std::shared_ptr<ComputeNode> node);

    // Graph analysis
    std::vector<std::shared_ptr<ComputeNode>> getExecutionOrder() const;
    void printGraph() const;
    size_t getNodeCount() const { return nodes_.size(); }

    // Getters
    const std::string &getName() const { return name_; }

private:
    std::string name_;
    std::vector<std::shared_ptr<ComputeNode>> nodes_;
    std::unordered_map<std::string, std::shared_ptr<ComputeNode>> node_map_;

    // Helper functions
    bool hasCycles() const;
    std::vector<std::shared_ptr<ComputeNode>> topologicalSort() const;
    void dfsVisit(std::shared_ptr<ComputeNode> node,
                  std::unordered_map<std::shared_ptr<ComputeNode>, int> &state,
                  std::vector<std::shared_ptr<ComputeNode>> &result) const;
};

// Specific node types

// Matrix multiplication node
class MatMulNode : public ComputeNode
{
public:
    MatMulNode(const std::string &name, int m, int n, int k);

    bool execute() override;
    bool validate() const override;

    // Matrix dimensions
    int getM() const { return m_; }
    int getN() const { return n_; }
    int getK() const { return k_; }

private:
    int m_, n_, k_;
};

// Input/Data node for feeding data into the graph
class DataNode : public ComputeNode
{
public:
    DataNode(const std::string &name, const std::vector<double> &data);

    bool execute() override;
    bool validate() const override;

    const std::vector<double> &getData() const { return data_; }
    void setData(const std::vector<double> &data) { data_ = data; }

private:
    std::vector<double> data_;
};

// Output node for extracting results from the graph
class OutputNode : public ComputeNode
{
public:
    OutputNode(const std::string &name);

    bool execute() override;
    bool validate() const override;

    const std::vector<double> &getResult() const { return result_; }

private:
    std::vector<double> result_;
};

// Transformer-specific compute nodes for LLM inference

// RMS normalization node for neural network layers

// Base class for transformer operations
class TransformerNode : public ComputeNode
{
public:
    TransformerNode(const std::string &name, const std::string &operation_type);

    void setInput(std::shared_ptr<Tensor> input) { input_tensor_ = input; }
    void setOutput(std::shared_ptr<Tensor> output) { output_tensor_ = output; }
    std::shared_ptr<Tensor> getInputTensor() const { return input_tensor_; }
    std::shared_ptr<Tensor> getOutputTensor() const { return output_tensor_; }

protected:
    std::shared_ptr<Tensor> input_tensor_;
    std::shared_ptr<Tensor> output_tensor_;
};

// RMS Normalization node
class RMSNormNode : public TransformerNode
{
public:
    RMSNormNode(const std::string &name, std::shared_ptr<Tensor> weight, float eps = 1e-6f);

    bool execute() override;
    bool validate() const override;

private:
    std::shared_ptr<Tensor> weight_;
    float eps_;
};

// Multi-head attention node
class AttentionNode : public TransformerNode
{
public:
    AttentionNode(const std::string &name,
                  std::shared_ptr<Tensor> q_weight,
                  std::shared_ptr<Tensor> k_weight,
                  std::shared_ptr<Tensor> v_weight,
                  std::shared_ptr<Tensor> out_weight,
                  std::shared_ptr<Tensor> k_cache,
                  std::shared_ptr<Tensor> v_cache,
                  int n_head, int n_head_kv, int n_past);

    bool execute() override;
    bool validate() const override;

    void setKVCache(std::shared_ptr<Tensor> k_cache, std::shared_ptr<Tensor> v_cache)
    {
        k_cache_ = k_cache;
        v_cache_ = v_cache;
    }

private:
    std::shared_ptr<Tensor> q_weight_;
    std::shared_ptr<Tensor> k_weight_;
    std::shared_ptr<Tensor> v_weight_;
    std::shared_ptr<Tensor> out_weight_;
    std::shared_ptr<Tensor> k_cache_, v_cache_;

    int n_head_, n_head_kv_, n_past_;
    int kv_seq_len_ = 0; // Current KV cache length
};

// MLP (Feed Forward) node
class MLPNode : public TransformerNode
{
public:
    MLPNode(const std::string &name,
            std::shared_ptr<Tensor> gate_weight,
            std::shared_ptr<Tensor> up_weight,
            std::shared_ptr<Tensor> down_weight);

    bool execute() override;
    bool validate() const override;

private:
    std::shared_ptr<Tensor> gate_weight_;
    std::shared_ptr<Tensor> up_weight_;
    std::shared_ptr<Tensor> down_weight_;
};

// Complete transformer block node
class TransformerBlockNode : public TransformerNode
{
public:
    TransformerBlockNode(const std::string &name, int layer_idx,
                         std::shared_ptr<RMSNormNode> attn_norm,
                         std::shared_ptr<AttentionNode> attention,
                         std::shared_ptr<RMSNormNode> ffn_norm,
                         std::shared_ptr<MLPNode> mlp);

    bool execute() override;
    bool validate() const override;

private:
    int layer_idx_;
    std::shared_ptr<RMSNormNode> attn_norm_;
    std::shared_ptr<AttentionNode> attention_;
    std::shared_ptr<RMSNormNode> ffn_norm_;
    std::shared_ptr<MLPNode> mlp_;

    // Temporary tensors for residual connections
    std::shared_ptr<Tensor> attn_residual_;
    std::shared_ptr<Tensor> ffn_residual_;
};

// Embedding lookup node
class EmbeddingNode : public TransformerNode
{
public:
    EmbeddingNode(const std::string &name, std::shared_ptr<Tensor> embedding_weights);

    bool execute() override;
    bool validate() const override;

    void setTokenIds(const std::vector<int> &token_ids) { token_ids_ = token_ids; }

private:
    std::shared_ptr<Tensor> embedding_weights_;
    std::vector<int> token_ids_;
};

// Linear projection node (matrix multiplication + optional bias)
class LinearNode : public TransformerNode
{
public:
    LinearNode(const std::string &name,
               std::shared_ptr<Tensor> weight,
               std::shared_ptr<Tensor> bias = nullptr);

    bool execute() override;
    bool validate() const override;

private:
    std::shared_ptr<Tensor> weight_;
    std::shared_ptr<Tensor> bias_;
};