/**
 * @file MPIResidualKernel.cpp
 * @brief Performs elementwise residual addition: Y = X + R (optionally in-place) with basic validation.
 *
 * @section Contract
 * Inputs:
 *  - inputs[0]: Base activation tensor X [seq_len, hidden_dim].
 *  - inputs[1]: Residual tensor R [seq_len, hidden_dim].
 * Outputs:
 *  - outputs[0]: Result tensor Y [seq_len, hidden_dim] (may alias inputs[0] if allowed by graph planner).
 * Behavior:
 *  - Pure elementwise addition in float32.
 *  - Optionally supports accumulation into preallocated buffer (checked via pointer equality by planner, not here).
 * Error Modes:
 *  - Shape mismatch or null tensors -> LOG_ERROR + return false.
 *  - Silent overflow not guarded (IEEE float expected to handle typical ranges).
 * Distribution:
 *  - Replicated; each rank performs identical addition.
 * Threading:
 *  - Parallel for over total elements; no data races (distinct index writes).
 * Numerical Expectations:
 *  - Bit-identical to reference addition for same ordering (single-pass, no reduction).
 * Future Extensions:
 *  - Fused residual + RMSNorm variant to reduce memory bandwidth.
 *  - Optional scaling factor (alpha) for weighted residual merges.
 * @warning Ensure that upstream kernels have synchronized outputs across ranks before residual addition to avoid divergence amplification.
 * @author David Sanftenberg
 */
#include "MPIResidualKernel.h"
#include "../debug_utils.h"
#include "../performance_timer.h"
#include "../tensors/simple_tensor.h"
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <omp.h>

namespace llaminar
{

    MPIResidualKernel::MPIResidualKernel(DistributionStrategy strategy)
        : MPIKernelBase(), strategy_(strategy), num_threads_(omp_get_max_threads()), broadcasting_enabled_(false)
    {
        LOG_DEBUG("MPIResidualKernel initialized on rank " << getRank() << "/" << getSize()
                                                           << " with strategy: " << (strategy == DistributionStrategy::SEQUENCE_WISE ? "SEQUENCE_WISE" : "ELEMENT_WISE")
                                                           << ", OpenMP threads: " << num_threads_);
    }

    bool MPIResidualKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                    std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        PERF_SCOPED_TIMER("MPIResidualKernel::execute");

        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPIResidualKernel: Validation failed");
            return false;
        }

        auto input_tensor = inputs[0];
        auto residual_tensor = inputs[1];
        auto output_tensor = outputs[0];

        const auto &input_shape = input_tensor->shape();
        const auto &residual_shape = residual_tensor->shape();

        // Check if we need broadcasting
        if (!areShapesCompatible(input_shape, residual_shape))
        {
            if (broadcasting_enabled_)
            {
                LOG_DEBUG("MPIResidualKernel: Using broadcasting for shape mismatch");
                const float *input_data = input_tensor->data();
                const float *residual_data = residual_tensor->data();
                float *output_data = output_tensor->data();

                executeBroadcast(input_data, residual_data, output_data, input_shape, residual_shape);
                return true;
            }
            else
            {
                LOG_ERROR("MPIResidualKernel: Shape mismatch and broadcasting disabled");
                return false;
            }
        }

        // Standard residual connection with matching shapes
        int seq_len = input_shape[0];
        int hidden_size = input_shape[1];
        size_t total_elements = seq_len * hidden_size;

        // Configure OpenMP threading based on problem size
        configureOpenMPThreading(total_elements);

        const bool input_distributed = input_tensor->is_distributed();
        const bool residual_distributed = residual_tensor->is_distributed();
        const bool output_distributed = output_tensor->is_distributed();

        const bool replicated_inputs = !input_distributed && !residual_distributed && !output_distributed;

        LOG_DEBUG("MPIResidualKernel tensor types: input=" << input_tensor->type_name()
                                                           << " residual=" << residual_tensor->type_name()
                                                           << " output=" << output_tensor->type_name()
                                                           << " distributed_flags=(" << (input_distributed ? "D" : "R")
                                                           << "," << (residual_distributed ? "D" : "R")
                                                           << "," << (output_distributed ? "D" : "R")
                                                           << ") replicated=" << (replicated_inputs ? "true" : "false"));

        const float *input_data = input_tensor->data();
        const float *residual_data = residual_tensor->data();
        float *output_data = output_tensor->data();

        // === RESIDUAL INPUT VALIDATION ===
        ASSERT_TENSOR_VALID(input_tensor, "Residual input");
        ASSERT_TENSOR_VALID(residual_tensor, "Residual connection");
        ASSERT_TENSOR_NOT_NAN(input_tensor, "Residual input has NaN");
        ASSERT_TENSOR_NOT_NAN(residual_tensor, "Residual connection has NaN");
        TensorLogger::logTensorStats(input_tensor, "residual_input");
        TensorLogger::logTensorStats(residual_tensor, "residual_connection");

        auto start_time = std::chrono::high_resolution_clock::now();

        // Execute based on distribution strategy
        try
        {
            switch (strategy_)
            {
            case DistributionStrategy::SEQUENCE_WISE:
                executeSequenceWise(input_data, residual_data, output_data, seq_len, hidden_size, replicated_inputs);
                break;
            case DistributionStrategy::ELEMENT_WISE:
                executeElementWise(input_data, residual_data, output_data, total_elements, replicated_inputs);
                break;
            default:
                LOG_ERROR("MPIResidualKernel: Unknown distribution strategy");
                return false;
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("MPIResidualKernel: Execution failed: " << e.what());
            return false;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        // === RESIDUAL OUTPUT VALIDATION ===
        ASSERT_TENSOR_NOT_NAN(output_tensor, "Residual output has NaN");
        TensorLogger::logTensorStats(output_tensor, "residual_output");

        LOG_DEBUG("MPIResidualKernel executed: " << seq_len << "x" << hidden_size
                                                 << " in " << std::fixed << std::setprecision(2) << execution_time
                                                 << "ms on rank " << getRank() << " (threads: " << omp_get_max_threads() << ")");

        return true;
    }

    bool MPIResidualKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                     const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        // Check input count
        if (inputs.size() != 2)
        {
            LOG_ERROR("MPIResidualKernel: Expected 2 inputs (input, residual), got " << inputs.size());
            return false;
        }

        // Check output count
        if (outputs.size() != 1)
        {
            LOG_ERROR("MPIResidualKernel: Expected 1 output (result), got " << outputs.size());
            return false;
        }

        // Validate input tensors
        auto input_tensor = inputs[0];
        auto residual_tensor = inputs[1];

        if (!input_tensor || !residual_tensor)
        {
            LOG_ERROR("MPIResidualKernel: Input tensors are null");
            return false;
        }

        const auto &input_shape = input_tensor->shape();
        const auto &residual_shape = residual_tensor->shape();

        // Check tensor dimensions
        if (input_shape.size() != 2 || residual_shape.size() != 2)
        {
            LOG_ERROR("MPIResidualKernel: Input tensors must be 2D, got input: "
                      << input_shape.size() << "D, residual: " << residual_shape.size() << "D");
            return false;
        }

        // Check shape compatibility (exact match or broadcasting)
        if (!areShapesCompatible(input_shape, residual_shape) && !broadcasting_enabled_)
        {
            LOG_ERROR("MPIResidualKernel: Input shape mismatch - input: ["
                      << input_shape[0] << ", " << input_shape[1] << "], residual: ["
                      << residual_shape[0] << ", " << residual_shape[1]
                      << "] and broadcasting disabled");
            return false;
        }

        // Validate output tensor
        auto output_tensor = outputs[0];
        if (!output_tensor)
        {
            LOG_ERROR("MPIResidualKernel: Output tensor is null");
            return false;
        }

        const auto &output_shape = output_tensor->shape();
        if (output_shape.size() != 2)
        {
            LOG_ERROR("MPIResidualKernel: Output tensor must be 2D, got " << output_shape.size() << "D");
            return false;
        }

        // Check output shape matches input (should be larger of the two inputs)
        if (output_shape[0] != input_shape[0] || output_shape[1] != input_shape[1])
        {
            LOG_ERROR("MPIResidualKernel: Output shape mismatch - expected: ["
                      << input_shape[0] << ", " << input_shape[1] << "], got: ["
                      << output_shape[0] << ", " << output_shape[1] << "]");
            return false;
        }

        return true;
    }

    void MPIResidualKernel::setDistributionStrategy(DistributionStrategy strategy)
    {
        strategy_ = strategy;
        LOG_DEBUG("MPIResidualKernel: Distribution strategy changed to "
                  << (strategy == DistributionStrategy::SEQUENCE_WISE ? "SEQUENCE_WISE" : "ELEMENT_WISE"));
    }

    void MPIResidualKernel::configureOpenMPThreading(size_t tensor_size)
    {
        // Small operations: single-threaded to avoid overhead
        if (tensor_size < 8192)
        {
            omp_set_num_threads(1);
            LOG_DEBUG("MPIResidualKernel: Using single thread for small tensor (" << tensor_size << " elements)");
            return;
        }

        // Medium to large operations: use available threads
        int max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
        LOG_DEBUG("MPIResidualKernel: Using " << max_threads << " threads for tensor (" << tensor_size << " elements)");
    }

    void MPIResidualKernel::distributeMPIWork(size_t total_elements, size_t &start_idx, size_t &end_idx, bool replicated) const
    {
        if (replicated || getSize() <= 1)
        {
            start_idx = 0;
            end_idx = total_elements;
            if (replicated)
            {
                LOG_DEBUG("MPIResidualKernel: Replicated tensors detected; each rank processing full range [0, "
                          << total_elements << ")");
            }
            return;
        }

        int rank = getRank();
        int size = getSize();

        size_t elements_per_rank = total_elements / size;
        size_t remainder = total_elements % size;

        start_idx = rank * elements_per_rank;
        end_idx = start_idx + elements_per_rank;

        // Distribute remainder elements to first ranks
        if (rank < remainder)
        {
            start_idx += rank;
            end_idx += rank + 1;
        }
        else
        {
            start_idx += remainder;
            end_idx += remainder;
        }

        LOG_DEBUG("MPIResidualKernel: Rank " << rank << " processing elements ["
                                             << start_idx << ", " << end_idx << ") of " << total_elements);
    }

    void MPIResidualKernel::executeSequenceWise(const float *input_data, const float *residual_data, float *output_data,
                                                int seq_len, int hidden_size, bool replicated)
    {
        // Distribute sequence positions across MPI ranks
        size_t total_positions = seq_len;
        size_t start_pos, end_pos;
        distributeMPIWork(total_positions, start_pos, end_pos, replicated);

        // Process assigned sequence positions with OpenMP parallelization
        auto omp_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(static)
        for (size_t pos = start_pos; pos < end_pos; ++pos)
        {
            size_t pos_offset = pos * hidden_size;

#pragma omp simd aligned(input_data, residual_data, output_data : 32)
            for (int dim = 0; dim < hidden_size; ++dim)
            {
                size_t idx = pos_offset + dim;
                output_data[idx] = input_data[idx] + residual_data[idx];
            }
        }

        auto omp_end = std::chrono::high_resolution_clock::now();
        double omp_time = std::chrono::duration<double, std::milli>(omp_end - omp_start).count();
        LOG_DEBUG("MPIResidualKernel sequence-wise OpenMP: " << omp_time << "ms, threads: " << omp_get_max_threads());

        // Synchronize all ranks before proceeding when work was partitioned
        if (!replicated && getSize() > 1)
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    void MPIResidualKernel::executeElementWise(const float *input_data, const float *residual_data, float *output_data,
                                               size_t total_elements, bool replicated)
    {
        // Distribute all elements across MPI ranks
        size_t start_idx, end_idx;
        distributeMPIWork(total_elements, start_idx, end_idx, replicated);

        // Process assigned elements with OpenMP SIMD parallelization
        auto omp_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for simd aligned(input_data, residual_data, output_data : 32) schedule(static)
        for (size_t i = start_idx; i < end_idx; ++i)
        {
            output_data[i] = input_data[i] + residual_data[i];
        }

        auto omp_end = std::chrono::high_resolution_clock::now();
        double omp_time = std::chrono::duration<double, std::milli>(omp_end - omp_start).count();
        LOG_DEBUG("MPIResidualKernel element-wise OpenMP: " << omp_time << "ms, threads: " << omp_get_max_threads());

        // Synchronize all ranks before proceeding when work was partitioned
        if (!replicated && getSize() > 1)
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    bool MPIResidualKernel::areShapesCompatible(const std::vector<int> &input_shape,
                                                const std::vector<int> &residual_shape) const
    {
        // Check for exact match
        if (input_shape.size() != residual_shape.size())
        {
            return false;
        }

        for (size_t i = 0; i < input_shape.size(); ++i)
        {
            if (input_shape[i] != residual_shape[i])
            {
                return false;
            }
        }

        return true;
    }

    void MPIResidualKernel::executeBroadcast(const float *input_data, const float *residual_data, float *output_data,
                                             const std::vector<int> &input_shape,
                                             const std::vector<int> &residual_shape)
    {
        // Simple broadcasting implementation for common cases
        // This can be extended for more complex broadcasting rules as needed

        int input_seq_len = input_shape[0];
        int input_hidden_size = input_shape[1];
        int residual_seq_len = residual_shape[0];
        int residual_hidden_size = residual_shape[1];

        auto omp_start = std::chrono::high_resolution_clock::now();

        // Case 1: Residual is [1, hidden_size] - broadcast across sequence
        if (residual_seq_len == 1 && residual_hidden_size == input_hidden_size)
        {
#pragma omp parallel for schedule(static)
            for (int pos = 0; pos < input_seq_len; ++pos)
            {
#pragma omp simd aligned(input_data, residual_data, output_data : 32)
                for (int dim = 0; dim < input_hidden_size; ++dim)
                {
                    size_t input_idx = pos * input_hidden_size + dim;
                    size_t residual_idx = dim; // Broadcast from single row
                    output_data[input_idx] = input_data[input_idx] + residual_data[residual_idx];
                }
            }
        }
        // Case 2: Residual is [seq_len, 1] - broadcast across hidden dimensions
        else if (residual_seq_len == input_seq_len && residual_hidden_size == 1)
        {
#pragma omp parallel for schedule(static)
            for (int pos = 0; pos < input_seq_len; ++pos)
            {
                float residual_val = residual_data[pos]; // Single value per position
#pragma omp simd aligned(input_data, output_data : 32)
                for (int dim = 0; dim < input_hidden_size; ++dim)
                {
                    size_t idx = pos * input_hidden_size + dim;
                    output_data[idx] = input_data[idx] + residual_val;
                }
            }
        }
        // Case 3: Scalar residual [1, 1] - broadcast to all elements
        else if (residual_seq_len == 1 && residual_hidden_size == 1)
        {
            float residual_val = residual_data[0];
            size_t total_elements = input_seq_len * input_hidden_size;

#pragma omp parallel for simd aligned(input_data, output_data : 32) schedule(static)
            for (size_t i = 0; i < total_elements; ++i)
            {
                output_data[i] = input_data[i] + residual_val;
            }
        }
        else
        {
            LOG_ERROR("MPIResidualKernel: Unsupported broadcasting pattern - input: ["
                      << input_seq_len << ", " << input_hidden_size << "], residual: ["
                      << residual_seq_len << ", " << residual_hidden_size << "]");
            throw std::runtime_error("Unsupported broadcasting pattern");
        }

        auto omp_end = std::chrono::high_resolution_clock::now();
        double omp_time = std::chrono::duration<double, std::milli>(omp_end - omp_start).count();
        LOG_DEBUG("MPIResidualKernel broadcast OpenMP: " << omp_time << "ms, threads: " << omp_get_max_threads());

        // Synchronize all ranks after broadcasting
        MPI_Barrier(MPI_COMM_WORLD);
    }

} // namespace llaminar