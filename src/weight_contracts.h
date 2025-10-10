/**
 * @file weight_contracts.h
 * @brief Weight dimension and orientation contracts for model loading.
 *
 * This system validates that model weights loaded from GGUF (or other formats)
 * match the expected dimensions and orientations that kernels and pipelines expect.
 * By validating at load time, we eliminate runtime shape detection and provide
 * clear error messages when format mismatches occur.
 *
 * @author David Sanftenberg
 */
#pragma once

#include "tensors/tensor_base.h"
#include "transformer_config.h"
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>

namespace llaminar
{
    /**
     * @brief Describes expected shape for a model weight tensor.
     *
     * Dimensions can be symbolic (e.g., "d_model", "n_head*head_dim") or literal integers.
     * The contract is evaluated against actual ModelConfig values at validation time.
     *
     * CRITICAL: dim_expressions specify GGUF format (what's in the file).
     * If requires_transpose=true, loader MUST transpose before use.
     */
    struct WeightShapeContract
    {
        std::string weight_name;                  ///< Name of the weight (e.g., "attn_q.weight")
        std::vector<std::string> dim_expressions; ///< Dimension expressions (e.g., ["n_head*head_dim", "d_model"])
        std::string description;                  ///< Human-readable description
        bool requires_transpose;                  ///< If true, tensor must be transposed after loading

        WeightShapeContract(const std::string &name,
                            const std::vector<std::string> &dims,
                            const std::string &desc = "",
                            bool transpose = false)
            : weight_name(name), dim_expressions(dims), description(desc), requires_transpose(transpose) {}

        /**
         * @brief Evaluate dimension expressions against model config.
         *
         * @param cfg Model configuration with architectural parameters
         * @return Expected shape as concrete integer dimensions
         */
        std::vector<int> evaluate(const TransformerLayerConfig &cfg) const
        {
            std::vector<int> shape;
            shape.reserve(dim_expressions.size());

            for (const auto &expr : dim_expressions)
            {
                shape.push_back(evaluate_expression(expr, cfg));
            }

            return shape;
        }

        /**
         * @brief Validate that actual tensor matches this contract.
         *
         * @param tensor The loaded weight tensor
         * @param cfg Model configuration
         * @param layer_index Optional layer index for per-layer weights
         * @throws std::runtime_error if validation fails
         */
        void validate(const std::shared_ptr<TensorBase> &tensor,
                      const TransformerLayerConfig &cfg,
                      int layer_index = -1) const
        {
            if (!tensor)
            {
                throw std::runtime_error("Weight contract validation failed: " + weight_name + " is null");
            }

            auto expected = evaluate(cfg);
            auto actual = tensor->shape();

            if (expected.size() != actual.size())
            {
                throw std::runtime_error(format_error(expected, actual, layer_index,
                                                      "Rank mismatch"));
            }

            for (size_t i = 0; i < expected.size(); ++i)
            {
                if (expected[i] != actual[i])
                {
                    throw std::runtime_error(format_error(expected, actual, layer_index,
                                                          "Dimension " + std::to_string(i) + " mismatch"));
                }
            }
        }

        /**
         * @brief Validate that tensor has been transposed correctly (for C++ matmul).
         *
         * If requires_transpose=true, checks that tensor dimensions are SWAPPED
         * from the GGUF format. This enforces the contract that C++ kernels
         * receive weights in [in_features, out_features] format.
         *
         * @param tensor The weight tensor AFTER transpose should have been applied
         * @param cfg Model configuration
         * @param layer_index Optional layer index for per-layer weights
         * @throws std::runtime_error if tensor not transposed when required
         */
        void validate_post_transpose(const std::shared_ptr<TensorBase> &tensor,
                                     const TransformerLayerConfig &cfg,
                                     int layer_index = -1) const
        {
            if (!requires_transpose)
            {
                // No transpose required, validate as normal
                validate(tensor, cfg, layer_index);
                return;
            }

            if (!tensor)
            {
                throw std::runtime_error("Weight contract validation failed: " + weight_name + " is null");
            }

            // For transpose-required weights, expect dimensions SWAPPED
            auto gguf_format = evaluate(cfg);
            auto actual = tensor->shape();

            // After transpose: [out, in] → [in, out]
            std::vector<int> expected_after_transpose(gguf_format.rbegin(), gguf_format.rend());

            if (expected_after_transpose.size() != actual.size())
            {
                throw std::runtime_error(format_error(expected_after_transpose, actual, layer_index,
                                                      "Rank mismatch after transpose"));
            }

            for (size_t i = 0; i < expected_after_transpose.size(); ++i)
            {
                if (expected_after_transpose[i] != actual[i])
                {
                    std::ostringstream oss;
                    oss << "Weight '" << weight_name << "' NOT TRANSPOSED! "
                        << "GGUF format is " << format_shape(gguf_format) << ", "
                        << "C++ needs " << format_shape(expected_after_transpose) << " (transposed), "
                        << "but got " << format_shape(actual);
                    if (layer_index >= 0)
                    {
                        oss << " (layer " << layer_index << ")";
                    }
                    throw std::runtime_error(oss.str());
                }
            }
        }

    private:
        /**
         * @brief Evaluate a single dimension expression.
         *
         * Supports:
         * - Literals: "896", "128"
         * - Variables: "d_model", "n_head", "head_dim", "n_head_kv", "d_ff", "vocab_size"
         * - Multiplication: "n_head*head_dim"
         * - Division: "n_head/2" (integer division)
         */
        int evaluate_expression(const std::string &expr, const TransformerLayerConfig &cfg) const
        {
            // Handle multiplication
            size_t mul_pos = expr.find('*');
            if (mul_pos != std::string::npos)
            {
                std::string left = expr.substr(0, mul_pos);
                std::string right = expr.substr(mul_pos + 1);
                return evaluate_expression(left, cfg) * evaluate_expression(right, cfg);
            }

            // Handle division
            size_t div_pos = expr.find('/');
            if (div_pos != std::string::npos)
            {
                std::string left = expr.substr(0, div_pos);
                std::string right = expr.substr(div_pos + 1);
                int divisor = evaluate_expression(right, cfg);
                if (divisor == 0)
                {
                    throw std::runtime_error("Division by zero in dimension expression: " + expr);
                }
                return evaluate_expression(left, cfg) / divisor;
            }

            // Try parsing as literal integer
            try
            {
                return std::stoi(expr);
            }
            catch (...)
            {
                // Not a literal, try variables
            }

            // Variable lookup
            if (expr == "d_model")
                return cfg.d_model;
            if (expr == "n_head")
                return cfg.n_head;
            if (expr == "head_dim")
                return cfg.head_dim;
            if (expr == "n_head_kv")
                return cfg.n_head_kv;
            if (expr == "d_ff")
                return cfg.d_ff;
            if (expr == "vocab_size")
                return cfg.vocab_size;
            if (expr == "max_seq_len")
                return cfg.max_seq_len;

            throw std::runtime_error("Unknown dimension expression: " + expr);
        }

        std::string format_error(const std::vector<int> &expected,
                                 const std::vector<int> &actual,
                                 int layer_index,
                                 const std::string &reason) const
        {
            std::ostringstream oss;
            oss << "Weight contract validation failed for '" << weight_name << "'";
            if (layer_index >= 0)
            {
                oss << " (layer " << layer_index << ")";
            }
            oss << ":\n";
            if (!description.empty())
            {
                oss << "  Description: " << description << "\n";
            }
            oss << "  Reason: " << reason << "\n";
            oss << "  Expected shape: [";
            for (size_t i = 0; i < expected.size(); ++i)
            {
                if (i > 0)
                    oss << ", ";
                oss << expected[i];
            }
            oss << "] (from [";
            for (size_t i = 0; i < dim_expressions.size(); ++i)
            {
                if (i > 0)
                    oss << ", ";
                oss << dim_expressions[i];
            }
            oss << "])\n";
            oss << "  Actual shape:   [";
            for (size_t i = 0; i < actual.size(); ++i)
            {
                if (i > 0)
                    oss << ", ";
                oss << actual[i];
            }
            oss << "]";
            return oss.str();
        }

        std::string format_shape(const std::vector<int> &shape) const
        {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (i > 0)
                    oss << ", ";
                oss << shape[i];
            }
            oss << "]";
            return oss.str();
        }
    };

    /**
     * @brief Collection of weight contracts for a complete model architecture.
     *
     * Defines the canonical weight format that ModelLoader must provide and
     * that all kernels can rely upon.
     */
    struct ModelWeightContracts
    {
        std::vector<WeightShapeContract> global_weights; ///< Non-layer weights (embedding, lm_head, etc.)
        std::vector<WeightShapeContract> layer_weights;  ///< Per-layer weights (repeated for each layer)

        /**
         * @brief Validate all global (non-layer) weights.
         */
        void validate_global(const std::shared_ptr<TensorBase> &token_embedding,
                             const std::shared_ptr<TensorBase> &output_norm,
                             const std::shared_ptr<TensorBase> &lm_head,
                             const TransformerLayerConfig &cfg) const
        {
            for (const auto &contract : global_weights)
            {
                std::shared_ptr<TensorBase> tensor;

                if (contract.weight_name == "token_embedding" || contract.weight_name == "embed_tokens")
                {
                    tensor = token_embedding;
                }
                else if (contract.weight_name == "output_norm" || contract.weight_name == "norm")
                {
                    tensor = output_norm;
                }
                else if (contract.weight_name == "lm_head" || contract.weight_name == "output")
                {
                    tensor = lm_head;
                }

                if (tensor)
                {
                    contract.validate(tensor, cfg);
                }
            }
        }

        /**
         * @brief Validate per-layer weights for a single layer.
         */
        void validate_layer(int layer_idx,
                            const std::shared_ptr<TensorBase> &attn_norm,
                            const std::shared_ptr<TensorBase> &wq,
                            const std::shared_ptr<TensorBase> &wk,
                            const std::shared_ptr<TensorBase> &wv,
                            const std::shared_ptr<TensorBase> &wo,
                            const std::shared_ptr<TensorBase> &ffn_norm,
                            const std::shared_ptr<TensorBase> &w_gate,
                            const std::shared_ptr<TensorBase> &w_up,
                            const std::shared_ptr<TensorBase> &w_down,
                            const TransformerLayerConfig &cfg) const
        {
            for (const auto &contract : layer_weights)
            {
                std::shared_ptr<TensorBase> tensor;

                if (contract.weight_name.find("attn_q") != std::string::npos ||
                    contract.weight_name.find("q_proj") != std::string::npos)
                {
                    tensor = wq;
                }
                else if (contract.weight_name.find("attn_k") != std::string::npos ||
                         contract.weight_name.find("k_proj") != std::string::npos)
                {
                    tensor = wk;
                }
                else if (contract.weight_name.find("attn_v") != std::string::npos ||
                         contract.weight_name.find("v_proj") != std::string::npos)
                {
                    tensor = wv;
                }
                else if (contract.weight_name.find("attn_output") != std::string::npos ||
                         contract.weight_name.find("o_proj") != std::string::npos)
                {
                    tensor = wo;
                }
                else if (contract.weight_name.find("attn_norm") != std::string::npos)
                {
                    tensor = attn_norm;
                }
                else if (contract.weight_name.find("ffn_norm") != std::string::npos)
                {
                    tensor = ffn_norm;
                }
                else if (contract.weight_name.find("gate") != std::string::npos)
                {
                    tensor = w_gate;
                }
                else if (contract.weight_name.find("up") != std::string::npos)
                {
                    tensor = w_up;
                }
                else if (contract.weight_name.find("down") != std::string::npos)
                {
                    tensor = w_down;
                }

                if (tensor)
                {
                    contract.validate(tensor, cfg, layer_idx);
                }
            }
        }
    };

    /**
     * @brief Qwen/Qwen2 model weight contracts.
     *
     * CRITICAL: ALL KERNELS USE GGUF FORMAT DIRECTLY
     *
     * Both MPIAttentionKernel and MPILinearKernel expect weights in GGUF format:
     * - Format: [output_features, input_features] (PyTorch convention)
     * - Both kernels use GEMM with transpose_B flags internally
     * - NO post-load transposes are needed!
     *
     * This is the correct design because:
     * 1. GEMM operations can transpose on-the-fly more efficiently
     * 2. Avoids expensive O(n²) transpose operations during model loading
     * 3. Maintains consistency with GGUF storage format
     *
     * Weight dimensions in GGUF format (requires_transpose=false for all):
     * - Q/K/V: [n_head*head_dim, d_model]
     * - O proj: [d_model, n_head*head_dim]
     * - FFN gate/up: [d_ff, d_model]
     * - FFN down: [d_model, d_ff]
     */
    inline ModelWeightContracts getQwenWeightContracts()
    {
        ModelWeightContracts contracts;

        // Global weights
        contracts.global_weights = {
            WeightShapeContract("token_embedding", {"vocab_size", "d_model"},
                                "Token embedding lookup table",
                                false), // No transpose (lookup table)
            WeightShapeContract("output_norm", {"d_model"},
                                "Final RMS norm before lm_head",
                                false), // No transpose (1D)
            WeightShapeContract("lm_head", {"vocab_size", "d_model"},
                                "Output projection to vocabulary (GGUF: [vocab, d_model])",
                                false)}; // No transpose

        // Per-layer weights - ALL use GGUF format directly (no transposes!)
        contracts.layer_weights = {
            WeightShapeContract("attn_norm", {"d_model"},
                                "Attention input RMS norm",
                                false), // No transpose (1D)
            WeightShapeContract("attn_q.weight", {"n_head*head_dim", "d_model"},
                                "Query projection (GGUF: [out,in], kernel uses GEMM transpose_B)",
                                false), // NO TRANSPOSE
            WeightShapeContract("attn_k.weight", {"n_head_kv*head_dim", "d_model"},
                                "Key projection (GGUF: [out,in], kernel uses GEMM transpose_B)",
                                false), // NO TRANSPOSE
            WeightShapeContract("attn_v.weight", {"n_head_kv*head_dim", "d_model"},
                                "Value projection (GGUF: [out,in], kernel uses GEMM transpose_B)",
                                false), // NO TRANSPOSE
            WeightShapeContract("attn_output.weight", {"d_model", "n_head*head_dim"},
                                "Output projection (GGUF: [out,in], kernel uses GEMM transpose_B)",
                                false), // NO TRANSPOSE
            WeightShapeContract("ffn_norm", {"d_model"},
                                "FFN input RMS norm",
                                false), // No transpose (1D)
            WeightShapeContract("w_gate", {"d_ff", "d_model"},
                                "FFN gate projection (GGUF: [out,in], kernel uses GEMM transpose_B)",
                                false), // NO TRANSPOSE
            WeightShapeContract("w_up", {"d_ff", "d_model"},
                                "FFN up projection (GGUF: [out,in], kernel uses GEMM transpose_B)",
                                false), // NO TRANSPOSE
            WeightShapeContract("w_down", {"d_model", "d_ff"},
                                "FFN down projection (GGUF: [out,in], kernel uses GEMM transpose_B)",
                                false)}; // NO TRANSPOSE

        return contracts;
    }

} // namespace llaminar
