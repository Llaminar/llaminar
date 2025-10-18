/**
 * @file MpiSlicingHelper.cpp
 * @brief Implementation of MPI-aware tensor loading with automatic dimension handling
 * @author David Sanftenberg
 * @date 2025-01-12
 */

#include "MpiSlicingHelper.h"
#include "Logger.h"
#include <stdexcept>
#include <sstream>

namespace mpi_slicing
{

    /**
     * @brief Evaluate dimension expression in context of layer config
     *
     * Supports expressions like:
     * - "d_model" → cfg.d_model
     * - "n_head*head_dim" → cfg.n_head * cfg.head_dim
     * - "n_head_kv*head_dim" → cfg.n_head_kv * cfg.head_dim
     * - "d_ff" → cfg.d_ff
     * - Integer literals like "151669"
     */
    static int evaluate_dim_expression(const std::string &expr, const TransformerLayerConfig &cfg)
    {
        // Check for multiplication
        size_t star_pos = expr.find('*');
        if (star_pos != std::string::npos)
        {
            std::string left = expr.substr(0, star_pos);
            std::string right = expr.substr(star_pos + 1);
            return evaluate_dim_expression(left, cfg) * evaluate_dim_expression(right, cfg);
        }

        // Check for named dimensions
        if (expr == "d_model")
            return cfg.d_model;
        if (expr == "n_head")
            return cfg.n_head;
        if (expr == "n_head_kv")
            return cfg.n_head_kv;
        if (expr == "head_dim")
            return cfg.head_dim;
        if (expr == "d_ff" || expr == "intermediate_size") // intermediate_size is alias for d_ff
            return cfg.d_ff;
        if (expr == "vocab_size")
            return cfg.vocab_size;

        // Try to parse as integer
        try
        {
            return std::stoi(expr);
        }
        catch (...)
        {
            LOG_ERROR("Unknown dimension expression: " << expr);
            throw std::runtime_error("Unknown dimension expression: " + expr);
        }
    }

    SliceParams calculate_slice(const llaminar::WeightShapeContract &contract,
                                const TransformerLayerConfig &cfg,
                                int mpi_rank,
                                int mpi_size)
    {
        SliceParams params;

        // Handle REPLICATED weights - no slicing
        if (contract.slice_type == llaminar::WeightSliceType::REPLICATED)
        {
            params.offset = 0;
            params.count = -1; // Load full tensor
            params.is_column_slice = false;
            params.needs_transpose = contract.needs_transpose_data;

            // Expected shape from contract (PyTorch convention)
            for (const auto &expr : contract.dim_expressions)
            {
                params.expected_shape.push_back(evaluate_dim_expression(expr, cfg));
            }
            return params;
        }

        // Evaluate GGUF dimensions (as stored on disk)
        std::vector<int> gguf_dims;
        for (const auto &expr : contract.gguf_dim_expressions)
        {
            gguf_dims.push_back(evaluate_dim_expression(expr, cfg));
        }

        // Evaluate PyTorch dimensions (expected after loading)
        for (const auto &expr : contract.dim_expressions)
        {
            params.expected_shape.push_back(evaluate_dim_expression(expr, cfg));
        }

        // Determine slice parameter value
        int slice_dim_size = evaluate_dim_expression(contract.slice_parameter, cfg);
        int slice_dim_per_rank = slice_dim_size / mpi_size;

        // Calculate offset and count based on slice type
        //
        // CRITICAL FIX: After ModelLoader dimension correction, data layout ALWAYS matches
        // PyTorch convention (expected_shape). GGUF metadata is backwards, but ModelLoader
        // swaps it to match reality. Therefore, we can ignore gguf_dims and just slice
        // directly according to the slice type!
        //
        // ROW_SLICED → slice rows (dim 0)
        // COL_SLICED → slice columns (dim 1)
        // No transpose needed!
        //
        if (contract.slice_type == llaminar::WeightSliceType::ROW_SLICED)
        {
            // ROW_SLICED in PyTorch convention
            // Data is already in PyTorch layout, so slice rows directly
            params.is_column_slice = false; // Slice rows (dim 0)
            params.offset = mpi_rank * slice_dim_per_rank * cfg.head_dim;
            params.count = slice_dim_per_rank * cfg.head_dim;
            params.needs_transpose = false; // No transpose - data already correct!
        }
        else if (contract.slice_type == llaminar::WeightSliceType::COL_SLICED)
        {
            // COL_SLICED in PyTorch convention
            // Data is already in PyTorch layout, so slice columns directly
            params.is_column_slice = true; // Slice columns (dim 1)
            params.offset = mpi_rank * slice_dim_per_rank * cfg.head_dim;
            params.count = slice_dim_per_rank * cfg.head_dim;
            params.needs_transpose = false; // No transpose - data already correct!
        }

        LOG_DEBUG("Calculated slice params for " << contract.role_description
                                                 << " rank " << mpi_rank << "/" << mpi_size
                                                 << ": offset=" << params.offset << ", count=" << params.count
                                                 << ", is_column=" << params.is_column_slice
                                                 << ", transpose=" << params.needs_transpose);

        return params;
    }

    std::shared_ptr<llaminar::TensorBase> load_with_contract(
        ModelLoader &loader,
        const llaminar::WeightShapeContract &contract,
        const TransformerLayerConfig &cfg,
        int mpi_rank,
        int mpi_size,
        int layer_index)
    {

        // Build tensor name (replace "{layer}" if present)
        std::string tensor_name = contract.name_pattern;
        if (layer_index >= 0)
        {
            size_t pos = tensor_name.find("{layer}");
            if (pos != std::string::npos)
            {
                tensor_name.replace(pos, 7, std::to_string(layer_index));
            }
        }

        LOG_DEBUG("Loading tensor: " << tensor_name
                                     << " with contract: " << contract.role_description);

        // Calculate slice parameters
        SliceParams params = calculate_slice(contract, cfg, mpi_rank, mpi_size);

        // Create tensor for loading
        std::shared_ptr<llaminar::TensorBase> tensor;
        if (params.count == -1)
        {
            // REPLICATED - load full tensor
            tensor = loader.loadTensor(tensor_name);

            if (!tensor)
            {
                LOG_ERROR("Failed to load replicated tensor: " << tensor_name);
                throw std::runtime_error("Failed to load tensor: " + tensor_name);
            }

            // Log the shape as loaded from GGUF
            auto loaded_shape = tensor->shape();
            LOG_DEBUG("Loaded REPLICATED tensor " << tensor_name << " from GGUF with shape ["
                                                  << loaded_shape[0] << ", " << loaded_shape[1] << "], needs_transpose=" << params.needs_transpose);

            // Verify dimensions match contract expectations
            LOG_DEBUG("[CONTRACT_VERIFY] Tensor " << tensor_name
                                                  << ": loaded_shape=[" << loaded_shape[0] << "," << loaded_shape[1] << "]"
                                                  << " expected_shape=[" << params.expected_shape[0] << "," << params.expected_shape[1] << "]"
                                                  << " dims_match=" << (loaded_shape[0] == params.expected_shape[0] && loaded_shape[1] == params.expected_shape[1]));

            if (params.needs_transpose)
            {
                // Get original shape (should be 2D for transpose)
                auto orig_shape = tensor->shape();
                if (orig_shape.size() != 2)
                {
                    LOG_ERROR("Transpose requires 2D tensor, got " << orig_shape.size() << "D for " << tensor_name);
                    throw std::runtime_error("Cannot transpose non-2D tensor");
                }

                int orig_rows = orig_shape[0];
                int orig_cols = orig_shape[1];

                // Create transposed tensor with swapped dimensions
                auto transposed = llaminar::TensorFactory::create_simple({orig_cols, orig_rows});
                float *trans_data = const_cast<float *>(transposed->data());
                const float *orig_data = tensor->data();

                // Transpose: trans[j,i] = orig[i,j]
                for (int i = 0; i < orig_rows; ++i)
                {
                    for (int j = 0; j < orig_cols; ++j)
                    {
                        trans_data[j * orig_rows + i] = orig_data[i * orig_cols + j];
                    }
                }

                tensor = transposed;
                LOG_DEBUG("Transposed replicated tensor " << tensor_name
                                                          << " from [" << orig_rows << ", " << orig_cols << "]"
                                                          << " to [" << orig_cols << ", " << orig_rows << "]");
            }
        }
        else
        {
            // SLICED - load partial tensor

            // Determine shape for loaded slice (before transpose)
            int load_rows, load_cols;
            if (params.is_column_slice)
            {
                // Loading columns - shape is [full_rows, slice_cols]
                load_rows = params.expected_shape[params.needs_transpose ? 1 : 0];
                load_cols = params.count;
            }
            else
            {
                // Loading rows - shape is [slice_rows, full_cols]
                load_rows = params.count;
                load_cols = params.expected_shape[params.needs_transpose ? 0 : 1];
            }

            tensor = llaminar::TensorFactory::create_simple({load_rows, load_cols});

            // Log slice operation details
            LOG_DEBUG("[CONTRACT_VERIFY] SLICED tensor " << tensor_name
                                                         << ": slice_type=" << (params.is_column_slice ? "column" : "row")
                                                         << " offset=" << params.offset << " count=" << params.count
                                                         << " loaded_shape=[" << load_rows << "," << load_cols << "]"
                                                         << " expected_final_shape=[" << params.expected_shape[0] << "," << params.expected_shape[1] << "]"
                                                         << " needs_transpose=" << params.needs_transpose);

            // Load the slice
            if (params.is_column_slice)
            {
                if (!loader.loadTensorColumnShard(tensor_name, params.offset, params.count,
                                                  const_cast<float *>(tensor->data())))
                {
                    LOG_ERROR("Failed to load column slice of tensor: " << tensor_name);
                    throw std::runtime_error("Failed to load column slice: " + tensor_name);
                }
            }
            else
            {
                if (!loader.loadTensorRowShard(tensor_name, params.offset, params.count,
                                               const_cast<float *>(tensor->data())))
                {
                    LOG_ERROR("Failed to load row slice of tensor: " << tensor_name);
                    throw std::runtime_error("Failed to load row slice: " + tensor_name);
                }
            }

            // Transpose if needed
            if (params.needs_transpose)
            {
                // Create transposed tensor
                auto transposed = llaminar::TensorFactory::create_simple({load_cols, load_rows});
                float *trans_data = const_cast<float *>(transposed->data());
                const float *orig_data = tensor->data();

                // Transpose: trans[j,i] = orig[i,j]
                for (int i = 0; i < load_rows; ++i)
                {
                    for (int j = 0; j < load_cols; ++j)
                    {
                        trans_data[j * load_rows + i] = orig_data[i * load_cols + j];
                    }
                }

                tensor = transposed;
            }
        }

        // Validate shape matches expected
        auto actual_shape = tensor->shape();
        if (actual_shape.size() != params.expected_shape.size())
        {
            LOG_ERROR("Shape dimension mismatch for " << tensor_name
                                                      << ": expected " << params.expected_shape.size() << "D, got "
                                                      << actual_shape.size() << "D");
            throw std::runtime_error("Shape dimension mismatch");
        }

        // For sliced weights, the shape should match the sliced dimensions
        std::vector<int> expected_sliced_shape = params.expected_shape;
        if (contract.slice_type != llaminar::WeightSliceType::REPLICATED)
        {
            // Adjust the sliced dimension
            int slice_dim_size = evaluate_dim_expression(contract.slice_parameter, cfg);
            int slice_dim_per_rank = slice_dim_size / mpi_size;

            if (contract.slice_type == llaminar::WeightSliceType::ROW_SLICED)
            {
                // Dimension 0 is sliced in PyTorch layout
                expected_sliced_shape[0] = slice_dim_per_rank * cfg.head_dim;
            }
            else if (contract.slice_type == llaminar::WeightSliceType::COL_SLICED)
            {
                // Dimension 1 is sliced in PyTorch layout
                expected_sliced_shape[1] = slice_dim_per_rank * cfg.head_dim;
            }
        }

        for (size_t i = 0; i < actual_shape.size(); ++i)
        {
            if (actual_shape[i] != expected_sliced_shape[i])
            {
                LOG_ERROR("Shape mismatch for " << tensor_name
                                                << " dim " << i << ": expected " << expected_sliced_shape[i]
                                                << ", got " << actual_shape[i]);
                throw std::runtime_error("Shape mismatch");
            }
        }

        // Log success with proper shape formatting based on dimensionality
        std::ostringstream shape_str;
        shape_str << "[";
        for (size_t i = 0; i < actual_shape.size(); ++i)
        {
            if (i > 0)
                shape_str << ", ";
            shape_str << actual_shape[i];
        }
        shape_str << "]";

        LOG_INFO("Successfully loaded " << tensor_name << " with shape "
                                        << shape_str.str() << " for rank " << mpi_rank << "/" << mpi_size);

        return tensor;
    }

} // namespace mpi_slicing

// Implementation of WeightShapeContract::load() method
namespace llaminar
{

    std::shared_ptr<TensorBase> WeightShapeContract::load(
        ModelLoader &loader,
        const TransformerLayerConfig &cfg,
        int mpi_rank,
        int mpi_size,
        int layer_index) const
    {
        return mpi_slicing::load_with_contract(loader, *this, cfg, mpi_rank, mpi_size, layer_index);
    }

} // namespace llaminar
