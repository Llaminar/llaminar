/**
 * @file AttentionStageContracts.h
 * @brief Stage contracts for MPIAttentionOperator data flow validation
 * @author David Sanftenberg
 *
 * Defines explicit contracts between attention kernel transformation stages:
 * - Shape expectations (dimensions and compatibility)
 * - Memory layout specifications (row-major, head-interleaved, etc.)
 * - Semantic guarantees (what tensors represent)
 * - Runtime validation with clear error messages
 *
 * Purpose: Eliminate dimension confusion and transpose bugs by making
 * data flow contracts explicit and verifiable at runtime.
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <memory>
#include "../../tensors/TensorBase.h"
#include "../../Logger.h"

namespace llaminar
{

    /**
     * @enum TensorLayout
     * @brief Memory layout specification for tensors
     */
    enum class TensorLayout
    {
        RowMajor,        ///< Standard C/C++ row-major layout
        HeadInterleaved, ///< [seq, heads*head_dim] with heads interleaved: [h0_d0..h0_d63, h1_d0..h1_d63]
        HeadSequential,  ///< [seq, heads*head_dim] with heads sequential: [h0_d0..h0_d63...hN_d0..hN_d63]
        Transposed       ///< Explicitly transposed (requires dimension swap)
    };

    /**
     * @enum TensorSemantic
     * @brief Semantic meaning of tensor in computation graph
     */
    enum class TensorSemantic
    {
        Activation,      ///< Activation tensor (sequence, features)
        Weight,          ///< Weight matrix (PyTorch nn.Linear format: [out_features, in_features])
        Bias,            ///< Bias vector (broadcast-compatible)
        AttentionScores, ///< QK^T score matrix [seq_len_q, seq_len_k]
        AttentionProbs,  ///< Softmax probability matrix [seq_len_q, seq_len_k]
        KVCache          ///< Key or Value cache tensor
    };

    /**
     * @struct ShapeSpec
     * @brief Shape specification with dynamic dimension support
     *
     * Uses -1 to denote dynamic dimensions (e.g., seq_len can vary).
     */
    struct ShapeSpec
    {
        std::vector<int> dims;           ///< Expected dimensions (-1 for dynamic)
        std::vector<std::string> labels; ///< Dimension labels (e.g., ["seq_len", "d_model"])

        ShapeSpec(std::vector<int> d, std::vector<std::string> l = {})
            : dims(std::move(d)), labels(std::move(l)) {}

        /**
         * @brief Check if actual shape matches specification
         * @param actual Actual tensor shape
         * @param allow_dynamic If true, -1 in spec matches any value in actual
         * @return True if shapes compatible
         */
        bool matches(const std::vector<int> &actual, bool allow_dynamic = true) const
        {
            if (dims.size() != actual.size())
                return false;
            for (size_t i = 0; i < dims.size(); ++i)
            {
                if (dims[i] == -1 && allow_dynamic)
                    continue; // Dynamic dimension
                if (dims[i] != actual[i])
                    return false;
            }
            return true;
        }

        std::string to_string() const
        {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < dims.size(); ++i)
            {
                if (i > 0)
                    oss << ", ";
                if (dims[i] == -1)
                    oss << "?";
                else
                    oss << dims[i];
                if (i < labels.size() && !labels[i].empty())
                    oss << " /*" << labels[i] << "*/";
            }
            oss << "]";
            return oss.str();
        }
    };

    /**
     * @struct TensorContract
     * @brief Contract specification for a single tensor
     *
     * Encodes expectations about shape, layout, and semantics.
     * Used for runtime validation at stage boundaries.
     */
    struct TensorContract
    {
        std::string name;        ///< Tensor name (for error messages)
        ShapeSpec shape;         ///< Expected shape
        TensorLayout layout;     ///< Memory layout
        TensorSemantic semantic; ///< Semantic meaning
        bool optional;           ///< Whether tensor can be nullptr
        bool allow_broadcast;    ///< For bias terms (1D matching last dim)

        TensorContract(std::string n, ShapeSpec s, TensorLayout l, TensorSemantic sem,
                       bool opt = false, bool bcast = false)
            : name(std::move(n)), shape(std::move(s)), layout(l), semantic(sem),
              optional(opt), allow_broadcast(bcast) {}

        /**
         * @brief Validate tensor against contract
         * @param tensor Tensor to validate (can be nullptr if optional)
         * @throws std::runtime_error if contract violated
         */
        void validate(const TensorBase *tensor) const
        {
            if (!tensor)
            {
                if (!optional)
                {
                    throw std::runtime_error("Contract violation: " + name + " is nullptr but required");
                }
                return; // Null is OK for optional tensors
            }

            const auto &actual_shape = tensor->shape();

            // Handle broadcast-compatible bias tensors (1D)
            if (allow_broadcast && actual_shape.size() == 1)
            {
                // For bias: should match last dimension of expected shape
                if (shape.dims.empty())
                {
                    throw std::runtime_error("Contract violation: " + name + " - cannot broadcast to empty shape");
                }
                int expected_last_dim = shape.dims.back();
                if (expected_last_dim != -1 && actual_shape[0] != expected_last_dim)
                {
                    throw std::runtime_error("Contract violation: " + name + " - bias dim " +
                                             std::to_string(actual_shape[0]) + " doesn't match expected " +
                                             std::to_string(expected_last_dim));
                }
                return; // Broadcast-compatible bias is valid
            }

            // Normal shape validation
            if (!shape.matches(actual_shape))
            {
                std::ostringstream oss;
                oss << "Contract violation: " << name << " shape mismatch!\n"
                    << "  Expected: " << shape.to_string() << "\n"
                    << "  Actual:   [";
                for (size_t i = 0; i < actual_shape.size(); ++i)
                {
                    if (i > 0)
                        oss << ", ";
                    oss << actual_shape[i];
                }
                oss << "]";
                throw std::runtime_error(oss.str());
            }
        }

        /**
         * @brief Validate tensor against contract (shared_ptr overload)
         */
        void validate(const std::shared_ptr<TensorBase> &tensor) const
        {
            validate(tensor.get());
        }
    };

    /**
     * @struct StageContract
     * @brief Contract for a complete transformation stage
     *
     * Defines inputs, outputs, and optional custom validation logic.
     */
    struct StageContract
    {
        std::string stage_name;                     ///< Stage identifier
        std::vector<TensorContract> inputs;         ///< Input tensor contracts
        std::vector<TensorContract> outputs;        ///< Output tensor contracts
        std::function<void(void)> custom_validator; ///< Optional additional checks
        bool enabled;                               ///< Whether validation is active

        StageContract() : stage_name("uninitialized"), enabled(false) {}

        StageContract(std::string name, bool enable = true)
            : stage_name(std::move(name)), enabled(enable) {}

        /**
         * @brief Validate all input tensors
         * @param tensors Input tensors to validate
         * @throws std::runtime_error if any contract violated
         */
        void validate_inputs(const std::vector<std::shared_ptr<TensorBase>> &tensors) const
        {
            if (!enabled)
                return;

            if (tensors.size() != inputs.size())
            {
                std::ostringstream oss;
                oss << "Stage '" << stage_name << "' input count mismatch: "
                    << "expected " << inputs.size() << ", got " << tensors.size();
                throw std::runtime_error(oss.str());
            }

            for (size_t i = 0; i < inputs.size(); ++i)
            {
                try
                {
                    inputs[i].validate(tensors[i]);
                }
                catch (const std::exception &e)
                {
                    std::ostringstream oss;
                    oss << "Stage '" << stage_name << "' input[" << i << "] (" << inputs[i].name << "): " << e.what();
                    throw std::runtime_error(oss.str());
                }
            }
        }

        /**
         * @brief Validate all output tensors
         * @param tensors Output tensors to validate
         * @throws std::runtime_error if any contract violated
         */
        void validate_outputs(const std::vector<std::shared_ptr<TensorBase>> &tensors) const
        {
            if (!enabled)
                return;

            if (tensors.size() != outputs.size())
            {
                std::ostringstream oss;
                oss << "Stage '" << stage_name << "' output count mismatch: "
                    << "expected " << outputs.size() << ", got " << tensors.size();
                throw std::runtime_error(oss.str());
            }

            for (size_t i = 0; i < outputs.size(); ++i)
            {
                try
                {
                    outputs[i].validate(tensors[i]);
                }
                catch (const std::exception &e)
                {
                    std::ostringstream oss;
                    oss << "Stage '" << stage_name << "' output[" << i << "] (" << outputs[i].name << "): " << e.what();
                    throw std::runtime_error(oss.str());
                }
            }
        }

        /**
         * @brief Run custom validation logic (if defined)
         */
        void validate_custom() const
        {
            if (!enabled || !custom_validator)
                return;
            custom_validator();
        }
    };

    /**
     * @brief Helper to create a simple activation tensor contract
     */
    inline TensorContract makeActivationContract(const std::string &name, int seq_len, int features)
    {
        return TensorContract(
            name,
            ShapeSpec{{seq_len, features}, {"seq_len", "features"}},
            TensorLayout::RowMajor,
            TensorSemantic::Activation);
    }

    /**
     * @brief Helper to create a weight matrix contract (PyTorch format)
     */
    inline TensorContract makeWeightContract(const std::string &name, int out_features, int in_features)
    {
        return TensorContract(
            name,
            ShapeSpec{{out_features, in_features}, {"out_features", "in_features"}},
            TensorLayout::RowMajor,
            TensorSemantic::Weight);
    }

    /**
     * @brief Helper to create a bias vector contract
     */
    inline TensorContract makeBiasContract(const std::string &name, int features, bool optional = true)
    {
        return TensorContract(
            name,
            ShapeSpec{{features}, {"features"}},
            TensorLayout::RowMajor,
            TensorSemantic::Bias,
            optional,
            true); // allow_broadcast
    }

    /**
     * @brief Validate batch attention Q/K/V projection dimensions to prevent double-distribution bug
     *
     * CRITICAL CONTRACT: MPIAttentionBatchOperator already distributes heads across MPI ranks.
     * The Q/K/V projections MUST use local-only matmul (distributed_partition=false) to avoid
     * double-distributing the output dimensions.
     *
     * @param batch_size Batch size (B)
     * @param seq_len Sequence length (T)
     * @param d_model Model dimension (input features)
     * @param n_heads_local Number of attention heads assigned to this MPI rank
     * @param head_dim Dimension per head
     * @param input_tensor Input activation tensor [B, T, d_model]
     * @param weight_local Local weight tensor [n_heads_local * head_dim, d_model]
     * @param output_local Output tensor [B, T, n_heads_local * head_dim]
     * @param projection_name Name of projection ("Q", "K", or "V") for error messages
     *
     * @throws std::runtime_error if dimensions are incorrect or suggest double-distribution
     */
    inline void validateBatchAttentionProjection(
        int batch_size,
        int seq_len,
        int d_model,
        int n_heads_local,
        int head_dim,
        const std::shared_ptr<TensorBase> &input_tensor,
        const std::shared_ptr<TensorBase> &weight_local,
        const std::shared_ptr<TensorBase> &output_local,
        const std::string &projection_name)
    {
        // Expected dimensions
        int expected_local_out_dim = n_heads_local * head_dim;

        // Validate input: [B, T, d_model]
        TensorContract input_contract(
            projection_name + "_input",
            ShapeSpec{{batch_size, seq_len, d_model}, {"B", "T", "d_model"}},
            TensorLayout::RowMajor,
            TensorSemantic::Activation);
        input_contract.validate(input_tensor);

        // Validate local weight: [n_heads_local * head_dim, d_model]
        // CRITICAL: This must be the FULL local head dimension, not further partitioned!
        TensorContract weight_contract(
            projection_name + "_weight_local",
            ShapeSpec{{expected_local_out_dim, d_model}, {"n_heads_local*head_dim", "d_model"}},
            TensorLayout::RowMajor,
            TensorSemantic::Weight);
        weight_contract.validate(weight_local);

        // Validate output: [B, T, n_heads_local * head_dim]
        // CRITICAL: Output must have FULL local head dimensions!
        // If output_dim < expected_local_out_dim, we have a double-distribution bug!
        TensorContract output_contract(
            projection_name + "_output_local",
            ShapeSpec{{batch_size, seq_len, expected_local_out_dim}, {"B", "T", "n_heads_local*head_dim"}},
            TensorLayout::RowMajor,
            TensorSemantic::Activation);
        output_contract.validate(output_local);

        // Additional runtime check for double-distribution bug
        const auto &output_shape = output_local->shape();
        int actual_out_dim = output_shape[2];

        if (actual_out_dim != expected_local_out_dim)
        {
            std::ostringstream oss;
            oss << "DOUBLE-DISTRIBUTION BUG DETECTED in " << projection_name << " projection!\n"
                << "  Expected output dim: " << expected_local_out_dim << " (n_heads_local=" << n_heads_local
                << " * head_dim=" << head_dim << ")\n"
                << "  Actual output dim: " << actual_out_dim << "\n"
                << "  This suggests the projection was further MPI-distributed!\n"
                << "  FIX: Use adaptiveMatMul with distributed_partition=false, NOT MPILinearBatchOperator!";
            throw std::runtime_error(oss.str());
        }

        LOG_TRACE("Batch attention " << projection_name << " projection contract validated: "
                                     << "input=[" << batch_size << "," << seq_len << "," << d_model << "] "
                                     << "weight=[" << expected_local_out_dim << "," << d_model << "] "
                                     << "output=[" << batch_size << "," << seq_len << "," << actual_out_dim << "]");
    }

} // namespace llaminar
