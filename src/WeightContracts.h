/**
 * @file WeightContracts.h
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

#include "tensors/TensorBase.h"
#include "TransformerConfig.h"
#include "Logger.h"
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>

// Forward declaration to avoid circular dependency
class ModelLoader;

namespace llaminar
{
    /**
     * @brief MPI slicing strategy for weight distribution across ranks.
     */
    enum class WeightSliceType
    {
        REPLICATED, ///< Full weight on every rank (no slicing)
        ROW_SLICED, ///< Row-wise slicing (first dimension sliced by heads)
        COL_SLICED  ///< Column-wise slicing (second dimension sliced by heads)
    };

    /**
     * @brief Describes expected shape for a model weight tensor with MPI slicing support.
     *
     * Dimensions can be symbolic (e.g., "d_model", "n_head*head_dim") or literal integers.
     * The contract is evaluated against actual ModelConfig values at validation time.
     *
     * CRITICAL: dim_expressions specify GGUF format (what's in the file).
     * If requires_transpose=true, loader MUST transpose before use.
     *
     * MPI SLICING: When mpi_size > 1, weights may be sliced across ranks:
     * - REPLICATED: Full weight on every rank (embedding, norms, etc.)
     * - ROW_SLICED: First dimension sliced (Q/K/V by attention heads)
     * - COL_SLICED: Second dimension sliced (O by attention heads)
     */
    struct WeightShapeContract
    {
        std::string weight_name;                  ///< Name of the weight (e.g., "attn_q.weight")
        std::vector<std::string> dim_expressions; ///< Dimension expressions (e.g., ["n_head*head_dim", "d_model"])
        std::string description;                  ///< Human-readable description
        bool requires_transpose;                  ///< If true, tensor must be transposed after loading

        // MPI slicing metadata
        WeightSliceType slice_type;  ///< How this weight is distributed across MPI ranks
        std::string slice_parameter; ///< Config parameter to slice by ("n_head", "n_head_kv", "d_ff")

        // GGUF storage format metadata (NEW - for contract-driven loading)
        std::vector<std::string> gguf_dim_expressions; ///< Dimensions as stored in GGUF file (before transpose)
        bool needs_transpose_data;                     ///< Whether to transpose actual data during loading
        std::string name_pattern;                      ///< Pattern with {layer} placeholder (e.g., "blk.{layer}.attn_q.weight")
        std::string role_description;                  ///< Role description for logging

        WeightShapeContract(const std::string &name,
                            const std::vector<std::string> &dims,
                            const std::string &desc = "",
                            bool transpose = false,
                            WeightSliceType slice = WeightSliceType::REPLICATED,
                            const std::string &slice_param = "",
                            const std::vector<std::string> &gguf_dims = {},
                            bool transpose_data = false)
            : weight_name(name), dim_expressions(dims), description(desc),
              requires_transpose(transpose), slice_type(slice), slice_parameter(slice_param),
              gguf_dim_expressions(gguf_dims.empty() ? dims : gguf_dims),
              needs_transpose_data(transpose_data),
              name_pattern(name),
              role_description(desc) {}

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
#ifdef LLAMINAR_ENABLE_VALIDATION
            // Default to single-rank validation (MPI size = 1)
            validate_with_mpi(tensor, cfg, 0, 1, layer_index);
#else
            (void)tensor;
            (void)cfg;
            (void)layer_index;
#endif
        }

        /**
         * @brief Validate tensor with MPI context (slicing-aware).
         *
         * @param tensor The loaded weight tensor
         * @param cfg Model configuration
         * @param mpi_rank Current MPI rank
         * @param mpi_size Total MPI world size
         * @param layer_index Optional layer index for per-layer weights
         * @throws std::runtime_error if validation fails
         */
        void validate_with_mpi(const std::shared_ptr<TensorBase> &tensor,
                               const TransformerLayerConfig &cfg,
                               int mpi_rank,
                               int mpi_size,
                               int layer_index = -1) const
        {
#ifdef LLAMINAR_ENABLE_VALIDATION
            if (!tensor)
            {
                throw std::runtime_error("Weight contract validation failed: " + weight_name + " is null");
            }

            std::vector<int> expected;

            if (mpi_size == 1)
            {
                // Single rank: validate full dimensions
                expected = evaluate(cfg);
            }
            else
            {
                // Multi-rank: validate sliced dimensions
                expected = evaluate_sliced(cfg, mpi_rank, mpi_size);
            }

            auto actual = tensor->shape();

            if (expected.size() != actual.size())
            {
                throw std::runtime_error(format_error_mpi(expected, actual, mpi_rank, mpi_size, layer_index,
                                                          "Rank mismatch"));
            }

            for (size_t i = 0; i < expected.size(); ++i)
            {
                if (expected[i] != actual[i])
                {
                    throw std::runtime_error(format_error_mpi(expected, actual, mpi_rank, mpi_size, layer_index,
                                                              "Dimension " + std::to_string(i) + " mismatch"));
                }
            }

            // Log successful validation with details
            std::stringstream log_msg;
            log_msg << "[WeightContract] ✓ " << weight_name;
            if (layer_index >= 0)
            {
                log_msg << " (layer " << layer_index << ")";
            }
            log_msg << ": shape=[";
            for (size_t i = 0; i < actual.size(); ++i)
            {
                if (i > 0)
                    log_msg << ", ";
                log_msg << actual[i];
            }
            log_msg << "]";
            if (mpi_size > 1)
            {
                log_msg << " (rank " << mpi_rank << "/" << mpi_size;
                if (slice_type != WeightSliceType::REPLICATED)
                {
                    log_msg << ", sliced ";
                    if (slice_type == WeightSliceType::ROW_SLICED)
                    {
                        log_msg << "rows";
                    }
                    else if (slice_type == WeightSliceType::COL_SLICED)
                    {
                        log_msg << "cols";
                    }
                }
                log_msg << ")";
            }
            LOG_DEBUG(log_msg.str());
#else
            (void)tensor;
            (void)cfg;
            (void)mpi_rank;
            (void)mpi_size;
            (void)layer_index;
#endif
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

        /**
         * @brief Load tensor with automatic MPI slicing based on this contract.
         *
         * This is the PRIMARY METHOD for loading weights in an MPI context.
         * Delegates to mpi_slicing::load_with_contract() for actual loading.
         *
         * @param loader ModelLoader instance for accessing GGUF file
         * @param cfg Transformer layer configuration
         * @param mpi_rank Current MPI rank
         * @param mpi_size Total number of MPI ranks
         * @param layer_index Layer index for layer-specific tensors (-1 for global)
         * @return Correctly-sliced tensor in expected PyTorch layout
         *
         * @note This method is declared here but implemented in mpi_slicing_helper.cpp
         *       to avoid circular dependencies.
         */
        std::shared_ptr<TensorBase> load(ModelLoader &loader,
                                         const TransformerLayerConfig &cfg,
                                         int mpi_rank,
                                         int mpi_size,
                                         int layer_index = -1) const;

    private:
        /**
         * @brief Evaluate dimension expressions for sliced weights.
         *
         * @param cfg Model configuration
         * @param mpi_rank Current MPI rank
         * @param mpi_size Total MPI world size
         * @return Expected shape for this rank's slice
         */
        std::vector<int> evaluate_sliced(const TransformerLayerConfig &cfg,
                                         int mpi_rank,
                                         int mpi_size) const
        {
            if (slice_type == WeightSliceType::REPLICATED)
            {
                // Replicated weights: full dimensions on every rank
                return evaluate(cfg);
            }

            // Get full dimensions first
            auto full_shape = evaluate(cfg);

            // Get the parameter value to slice by
            int slice_param_value = get_slice_parameter_value(slice_parameter, cfg);

            // Calculate local slice size
            if (slice_param_value % mpi_size != 0)
            {
                std::ostringstream oss;
                oss << "Weight '" << weight_name << "' cannot be evenly sliced: "
                    << slice_parameter << "=" << slice_param_value
                    << " is not divisible by mpi_size=" << mpi_size;
                throw std::runtime_error(oss.str());
            }

            int local_count = slice_param_value / mpi_size;

            // Apply slicing to appropriate dimension
            if (slice_type == WeightSliceType::ROW_SLICED)
            {
                // Row slicing: first dimension is sliced
                // e.g., Q: [n_head*head_dim, d_model] → [local_heads*head_dim, d_model]
                // The first dimension expression is "n_head*head_dim" or "n_head_kv*head_dim"
                // We need to replace the parameter value with the local count
                full_shape[0] = local_count * cfg.head_dim;
            }
            else if (slice_type == WeightSliceType::COL_SLICED)
            {
                // Column slicing: second dimension is sliced
                // e.g., O: [d_model, n_head*head_dim] → [d_model, local_heads*head_dim]
                full_shape[1] = local_count * cfg.head_dim;
            }

            return full_shape;
        }

        /**
         * @brief Get the value of a slice parameter from config.
         *
         * @param param_name Parameter name (e.g., "n_head", "n_head_kv", "d_ff")
         * @param cfg Model configuration
         * @return Parameter value
         */
        int get_slice_parameter_value(const std::string &param_name, const TransformerLayerConfig &cfg) const
        {
            if (param_name.empty())
            {
                return 1; // Replicated weights have no slice parameter
            }

            if (param_name == "n_head")
                return cfg.n_head;
            if (param_name == "n_head_kv")
                return cfg.n_head_kv;
            if (param_name == "d_ff")
                return cfg.d_ff;

            throw std::runtime_error("Unknown slice parameter: " + param_name);
        }

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
            if (expr == "d_ff" || expr == "intermediate_size") // intermediate_size is alias for d_ff
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
            return format_error_mpi(expected, actual, 0, 1, layer_index, reason);
        }

        std::string format_error_mpi(const std::vector<int> &expected,
                                     const std::vector<int> &actual,
                                     int mpi_rank,
                                     int mpi_size,
                                     int layer_index,
                                     const std::string &reason) const
        {
            std::ostringstream oss;
            oss << "Weight contract validation failed for '" << weight_name << "'";
            if (layer_index >= 0)
            {
                oss << " (layer " << layer_index << ")";
            }
            if (mpi_size > 1)
            {
                oss << " on rank " << mpi_rank << "/" << mpi_size;
            }
            oss << ":\n";
            if (!description.empty())
            {
                oss << "  Description: " << description << "\n";
            }

            // Explain slicing context
            if (mpi_size > 1)
            {
                oss << "  MPI Slicing: ";
                if (slice_type == WeightSliceType::REPLICATED)
                {
                    oss << "REPLICATED (full weight on every rank)";
                }
                else if (slice_type == WeightSliceType::ROW_SLICED)
                {
                    oss << "ROW_SLICED by " << slice_parameter;
                }
                else if (slice_type == WeightSliceType::COL_SLICED)
                {
                    oss << "COL_SLICED by " << slice_parameter;
                }
                oss << "\n";
            }

            oss << "  Reason: " << reason << "\n";
            oss << "  Expected shape: [";
            for (size_t i = 0; i < expected.size(); ++i)
            {
                if (i > 0)
                    oss << ", ";
                oss << expected[i];
            }
            oss << "]";

            if (mpi_size > 1)
            {
                oss << " (sliced for rank " << mpi_rank << ")";
            }
            else
            {
                oss << " (from [";
                for (size_t i = 0; i < dim_expressions.size(); ++i)
                {
                    if (i > 0)
                        oss << ", ";
                    oss << dim_expressions[i];
                }
                oss << "])";
            }

            oss << "\n  Actual shape:   [";
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
            validate_global_with_mpi(token_embedding, output_norm, lm_head, cfg, 0, 1);
        }

        /**
         * @brief Validate all global weights with MPI context.
         */
        void validate_global_with_mpi(const std::shared_ptr<TensorBase> &token_embedding,
                                      const std::shared_ptr<TensorBase> &output_norm,
                                      const std::shared_ptr<TensorBase> &lm_head,
                                      const TransformerLayerConfig &cfg,
                                      int mpi_rank,
                                      int mpi_size) const
        {
#ifdef LLAMINAR_ENABLE_VALIDATION
            LOG_INFO("[WeightContract] Validating " << global_weights.size() << " global weights (rank "
                                                    << mpi_rank << "/" << mpi_size << ")");

            int validated_count = 0;
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
                    contract.validate_with_mpi(tensor, cfg, mpi_rank, mpi_size);
                    validated_count++;
                }
            }

            LOG_INFO("[WeightContract] ✓ Global weights validated: " << validated_count << "/" << global_weights.size());
#else
            (void)token_embedding;
            (void)output_norm;
            (void)lm_head;
            (void)cfg;
            (void)mpi_rank;
            (void)mpi_size;
#endif
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
            validate_layer_with_mpi(layer_idx, attn_norm, wq, wk, wv, wo, ffn_norm,
                                    w_gate, w_up, w_down, cfg, 0, 1);
        }

        /**
         * @brief Validate per-layer weights with MPI context.
         */
        void validate_layer_with_mpi(int layer_idx,
                                     const std::shared_ptr<TensorBase> &attn_norm,
                                     const std::shared_ptr<TensorBase> &wq,
                                     const std::shared_ptr<TensorBase> &wk,
                                     const std::shared_ptr<TensorBase> &wv,
                                     const std::shared_ptr<TensorBase> &wo,
                                     const std::shared_ptr<TensorBase> &ffn_norm,
                                     const std::shared_ptr<TensorBase> &w_gate,
                                     const std::shared_ptr<TensorBase> &w_up,
                                     const std::shared_ptr<TensorBase> &w_down,
                                     const TransformerLayerConfig &cfg,
                                     int mpi_rank,
                                     int mpi_size) const
        {
#ifdef LLAMINAR_ENABLE_VALIDATION
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
                    contract.validate_with_mpi(tensor, cfg, mpi_rank, mpi_size, layer_idx);
                }
            }
#else
            (void)layer_idx;
            (void)attn_norm;
            (void)wq;
            (void)wk;
            (void)wv;
            (void)wo;
            (void)ffn_norm;
            (void)w_gate;
            (void)w_up;
            (void)w_down;
            (void)cfg;
            (void)mpi_rank;
            (void)mpi_size;
#endif
        }
    };

    /**
     * @brief Qwen/Qwen2 model weight contracts with MPI slicing support.
     *
     * CRITICAL: ModelLoader Automatic Dimension AND Data Transpose
     * =============================================================
     * The ModelLoader performs TWO operations for all 2D tensors:
     *
     * 1. **Dimension swap during parseTensorInfo()** (line ~1403):
     *    - Swaps metadata from GGUF [in, out] → Llaminar [out, in]
     *
     * 2. **Data transpose during loadTensor()** (line ~1090):
     *    - Transposes actual data to match swapped metadata
     *    - Result: Metadata and data are ALWAYS consistent
     *
     * Example for any weight matrix:
     *   - GGUF file stores: [896, 4864] (in × out layout)
     *   - ModelLoader returns: shape=[4864, 896], data layout=[4864, 896] ✓
     *   - Metadata and data are NOW CONSISTENT!
     *
     * Implication for weight contracts:
     *   - needs_transpose_data should be FALSE for all weights
     *   - ModelLoader has already done the necessary transpose
     *   - Contract system just validates shapes, no additional transpose needed
     *
     * CRITICAL: ALL KERNELS USE GGUF FORMAT DIRECTLY
     *
     * Both MPIAttentionOperator and MPILinearOperator expect weights in GGUF format:
     * - Format: [output_features, input_features] (PyTorch convention)
     * - Both kernels use GEMM with transpose_B flags internally
     * - NO post-load transposes are needed!
     *
     * This is the correct design because:
     * 1. GEMM operations can transpose on-the-fly more efficiently
     * 2. Avoids expensive O(n²) transpose operations during model loading
     * 3. Maintains consistency with GGUF storage format
     *
     * MPI SLICING STRATEGY (mpi_size > 1):
     * - Q weights: ROW_SLICED by n_head (each rank gets local_heads × head_dim rows)
     * - K/V weights: ROW_SLICED by n_head_kv (GQA: fewer KV heads)
     * - O weights: COL_SLICED by n_head (each rank gets local_heads × head_dim columns)
     * - FFN weights: COL_SLICED by d_ff (gate/up) or ROW_SLICED (down)
     * - Embeddings/Norms: REPLICATED (full copy on every rank)
     *
     * Weight dimensions in GGUF format (requires_transpose=false for all):
     * - Q/K/V: [n_head*head_dim, d_model] → [local_heads*head_dim, d_model] when sliced
     * - O proj: [d_model, n_head*head_dim] → [d_model, local_heads*head_dim] when sliced
     * - FFN gate/up: [d_ff, d_model] → [local_d_ff, d_model] when sliced
     * - FFN down: [d_model, d_ff] → [d_model, local_d_ff] when sliced
     */
    /**
     * @brief Get weight contracts for batch-mode Qwen pipeline
     *
     * Batch operators (MPIAttentionBatchOperator, etc.) handle weight distribution
     * internally at runtime. They expect FULL (REPLICATED) weights, not pre-sliced.
     *
     * This differs from getQwenWeightContracts() which returns ROW_SLICED/COL_SLICED
     * contracts for the sequential pipeline where operators expect pre-sliced weights.
     */
    inline ModelWeightContracts getBatchQwenWeightContracts()
    {
        using ST = WeightSliceType;
        ModelWeightContracts contracts;

        // Global weights - same as sequential mode
        contracts.global_weights = {
            WeightShapeContract("token_embedding", {"vocab_size", "d_model"},
                                "Token embedding lookup table",
                                false, ST::REPLICATED, "",
                                {"vocab_size", "d_model"}, false),
            WeightShapeContract("output_norm", {"d_model"},
                                "Final RMS norm before lm_head",
                                false, ST::REPLICATED, "",
                                {"d_model"}, false),
            WeightShapeContract("lm_head", {"vocab_size", "d_model"},
                                "Output projection to vocabulary (GGUF: [vocab, d_model])",
                                false, ST::REPLICATED, "",
                                {"vocab_size", "d_model"}, false)};

        // Per-layer weights - ALL REPLICATED for batch mode!
        // Batch operators handle runtime distribution themselves
        contracts.layer_weights = {
            WeightShapeContract("blk.{layer}.attn_norm.weight", {"d_model"},
                                "Attention input RMS norm",
                                false, ST::REPLICATED, "",
                                {"d_model"}, false),

            // Attention Q/K/V/O: REPLICATED (batch operators do runtime slicing)
            WeightShapeContract("blk.{layer}.attn_q.weight", {"n_head*head_dim", "d_model"},
                                "Query projection (REPLICATED for batch mode)",
                                false, ST::REPLICATED, "",
                                {"d_model", "n_head*head_dim"}, false),
            WeightShapeContract("blk.{layer}.attn_k.weight", {"n_head_kv*head_dim", "d_model"},
                                "Key projection (REPLICATED for batch mode)",
                                false, ST::REPLICATED, "",
                                {"d_model", "n_head_kv*head_dim"}, false),
            WeightShapeContract("blk.{layer}.attn_v.weight", {"n_head_kv*head_dim", "d_model"},
                                "Value projection (REPLICATED for batch mode)",
                                false, ST::REPLICATED, "",
                                {"d_model", "n_head_kv*head_dim"}, false),
            WeightShapeContract("blk.{layer}.attn_output.weight", {"d_model", "n_head*head_dim"},
                                "Output projection (REPLICATED for batch mode)",
                                false, ST::REPLICATED, "",
                                {"n_head*head_dim", "d_model"}, false),

            WeightShapeContract("blk.{layer}.ffn_norm.weight", {"d_model"},
                                "FFN input RMS norm",
                                false, ST::REPLICATED, "",
                                {"d_model"}, false),

            // FFN weights: REPLICATED (batch operators handle distribution)
            WeightShapeContract("blk.{layer}.ffn_gate.weight", {"intermediate_size", "d_model"},
                                "FFN gate projection (REPLICATED for batch mode)",
                                false, ST::REPLICATED, "",
                                {"d_model", "intermediate_size"}, false),
            WeightShapeContract("blk.{layer}.ffn_up.weight", {"intermediate_size", "d_model"},
                                "FFN up projection (REPLICATED for batch mode)",
                                false, ST::REPLICATED, "",
                                {"d_model", "intermediate_size"}, false),
            WeightShapeContract("blk.{layer}.ffn_down.weight", {"d_model", "intermediate_size"},
                                "FFN down projection (REPLICATED for batch mode)",
                                false, ST::REPLICATED, "",
                                {"intermediate_size", "d_model"}, false)};

        return contracts;
    }

    inline ModelWeightContracts getQwenWeightContracts()
    {
        using ST = WeightSliceType;
        ModelWeightContracts contracts;

        // Global weights (all replicated - full copy on every rank)
        // Token embedding and lm_head are stored as [vocab_size, d_model] in both GGUF and PyTorch
        // (No transpose needed for embeddings!)
        contracts.global_weights = {
            WeightShapeContract("token_embedding", {"vocab_size", "d_model"},
                                "Token embedding lookup table",
                                false, ST::REPLICATED, "",
                                {"vocab_size", "d_model"}, false),
            WeightShapeContract("output_norm", {"d_model"},
                                "Final RMS norm before lm_head",
                                false, ST::REPLICATED, "",
                                {"d_model"}, false),
            WeightShapeContract("lm_head", {"vocab_size", "d_model"},
                                "Output projection to vocabulary (GGUF: [vocab, d_model])",
                                false, ST::REPLICATED, "",
                                {"vocab_size", "d_model"}, false)};

        // Per-layer weights with MPI slicing metadata
        contracts.layer_weights = {
            WeightShapeContract("blk.{layer}.attn_norm.weight", {"d_model"},
                                "Attention input RMS norm",
                                false, ST::REPLICATED, "",
                                {"d_model"}, false),

            // Attention Q/K/V: Row-sliced by attention heads
            // CRITICAL FIX: needs_transpose_data = FALSE after ModelLoader dimension correction!
            // ====================================================================================
            // GGUF metadata dimensions are backwards from actual data layout, but ModelLoader
            // now corrects this by swapping metadata to match reality. The data layout in memory
            // already matches PyTorch convention, so NO transpose is needed!
            //
            // PyTorch Linear: weight is [out_features, in_features], forward does input @ weight.T
            // GGUF after ModelLoader: dimensions corrected to match PyTorch [out_features, in_features]
            // Data layout matches - ready to use directly!
            //
            // Q/K/V weights: [n_head*head_dim, d_model] - row-sliced by heads for MPI distribution
            // O weight: [d_model, n_head*head_dim] - column-sliced by heads for MPI distribution
            WeightShapeContract("blk.{layer}.attn_q.weight", {"n_head*head_dim", "d_model"},
                                "Query projection (row-sliced by Q heads in MPI mode)",
                                false, ST::ROW_SLICED, "n_head",
                                {"d_model", "n_head*head_dim"}, false),
            WeightShapeContract("blk.{layer}.attn_k.weight", {"n_head_kv*head_dim", "d_model"},
                                "Key projection (row-sliced by KV heads in MPI mode, GQA)",
                                false, ST::ROW_SLICED, "n_head_kv",
                                {"d_model", "n_head_kv*head_dim"}, false),
            WeightShapeContract("blk.{layer}.attn_v.weight", {"n_head_kv*head_dim", "d_model"},
                                "Value projection (row-sliced by KV heads in MPI mode, GQA)",
                                false, ST::ROW_SLICED, "n_head_kv",
                                {"d_model", "n_head_kv*head_dim"}, false),

            // Attention O: Column-sliced by Q heads (matches partial head outputs)
            WeightShapeContract("blk.{layer}.attn_output.weight", {"d_model", "n_head*head_dim"},
                                "Output projection (column-sliced by Q heads in MPI mode)",
                                false, ST::COL_SLICED, "n_head",
                                {"n_head*head_dim", "d_model"}, false),

            // NOTE: Attention biases (attn_q.bias, attn_k.bias, attn_v.bias) are NOT in contracts
            // They are loaded and pre-sliced manually in QwenPipeline to avoid complexity in the
            // contract system (biases are 1D, stored REPLICATED, but consumed as sliced).
            // This is architecturally cleaner than adding 1D slicing support to contracts.

            WeightShapeContract("blk.{layer}.ffn_norm.weight", {"d_model"},
                                "FFN input RMS norm",
                                false, ST::REPLICATED, "",
                                {"d_model"}, false),

            // FFN weights: Currently NOT sliced (TODO: implement FFN slicing in future)
            //
            // CRITICAL: needs_transpose_data = FALSE for REPLICATED weights!
            // ================================================================
            // Replicated (non-sliced) weights are loaded via ModelLoader.loadTensor()
            // which automatically transposes the data to match swapped dimensions.
            //
            // Sliced weights (Q/K/V/O above) have needs_transpose_data = TRUE because
            // they're loaded via loadTensorRowShard/ColumnShard which bypasses ModelLoader's
            // automatic transpose, so mpi_slicing_helper must transpose them instead.
            //
            WeightShapeContract("blk.{layer}.ffn_gate.weight", {"d_ff", "d_model"},
                                "FFN gate projection (currently replicated, slicing TODO)",
                                false, ST::REPLICATED, "",
                                {"d_model", "d_ff"}, false),
            WeightShapeContract("blk.{layer}.ffn_up.weight", {"d_ff", "d_model"},
                                "FFN up projection (currently replicated, slicing TODO)",
                                false, ST::REPLICATED, "",
                                {"d_model", "d_ff"}, false),
            WeightShapeContract("blk.{layer}.ffn_down.weight", {"d_model", "d_ff"},
                                "FFN down projection (currently replicated, slicing TODO)",
                                false, ST::REPLICATED, "",
                                {"d_ff", "d_model"}, false)};

        return contracts;
    }

} // namespace llaminar
