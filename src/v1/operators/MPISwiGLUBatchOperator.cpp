/**
 * @file MPISwiGLUBatchOperator.cpp
 * @brief SwiGLU activation with native batch support: output = swish(gate) ⊗ up
 *
 * @section Contract
 * Inputs:
 *  - inputs[0]: Gate tensor [batch, seq_len, hidden_ff] (for swish activation)
 *  - inputs[1]: Up tensor [batch, seq_len, hidden_ff] (multiplication partner)
 * Outputs:
 *  - outputs[0]: Activated tensor [batch, seq_len, hidden_ff]
 *
 * Formula:
 *  - swish(x) = x * sigmoid(x) where sigmoid(x) = 1 / (1 + exp(-x))
 *  - output[i] = swish(gate[i]) * up[i]  (element-wise)
 *
 * Batch Processing:
 *  - Processes all batch*seq_len*hidden_ff elements in parallel
 *  - No reshape needed (native 3D support)
 *  - Element-wise operation (no MPI communication)
 *
 * Numerical Expectations:
 *  - Stable for |x| < 20; extreme values saturate sigmoid
 *  - Relative error vs reference < 1e-6 for float32
 *  - Parity guarantee: batch=1 produces identical results to MPISwiGLUOperator
 *
 * Error Modes:
 *  - Shape mismatch between gate and up tensors
 *  - Null tensors
 *  - Invalid tensor dimensions
 *
 * Distribution:
 *  - Local operation (each rank processes its data independently)
 *  - No MPI communication required
 *  - Replicated across ranks
 *
 * Threading:
 *  - OpenMP parallelization over all elements
 *  - Thread count auto-configured based on problem size
 *  - Vectorization opportunities via compiler
 *
 * Performance Notes:
 *  - Memory bandwidth bound for large tensors
 *  - Fast sigmoid approximation for better performance
 *  - OMP_NUM_THREADS environment variable respected
 *
 * @author David Sanftenberg
 */
#include "MPISwiGLUBatchOperator.h"
#include "../Logger.h"
#include "../PerformanceTracer.h"
#include "../DebugUtils.h"
#include "../utils/DebugEnv.h"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <omp.h>

namespace llaminar
{

    MPISwiGLUBatchOperator::MPISwiGLUBatchOperator(MPI_Comm comm)
        : MPIOperatorBase(comm, false), num_threads_(omp_get_max_threads())
    {
        LOG_DEBUG("MPISwiGLUBatchOperator initialized on rank " << getRank()
                                                                << " with " << num_threads_ << " OpenMP threads");
    }

    bool MPISwiGLUBatchOperator::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                         std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        PERF_TRACE_SCOPE_CAT("mpi_swiglu_batch_execute", "activation");

        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPISwiGLUBatchOperator validation failed on rank " << getRank());
            return false;
        }

        auto gate = inputs[0];
        auto up = inputs[1];
        auto output = outputs[0];

        // === COMPREHENSIVE TENSOR VALIDATION ===
        ASSERT_TENSOR_VALID(gate, "SwiGLUBatch gate");
        ASSERT_TENSOR_VALID(up, "SwiGLUBatch up");
        ASSERT_TENSOR_VALID(output, "SwiGLUBatch output");

        // Check for NaN in inputs before computation
        ASSERT_TENSOR_NOT_NAN(gate, "SwiGLUBatch gate");
        ASSERT_TENSOR_NOT_NAN(up, "SwiGLUBatch up");

        // Extract dimensions
        size_t batch_size = gate->shape()[0];
        size_t seq_len = gate->shape()[1];
        size_t hidden_ff = gate->shape()[2];
        size_t total_elements = batch_size * seq_len * hidden_ff;

        // Configure threading
        configureOpenMPThreading(total_elements);

        // Get data pointers
        const float *gate_data = gate->data();
        const float *up_data = up->data();
        float *output_data = output->data();

        // Perform SwiGLU activation: output = swish(gate) * up
        auto start = std::chrono::high_resolution_clock::now();
        {
            PERF_TRACE_SCOPE_CAT("swiglu_activation", "activation");

#pragma omp parallel for
            for (size_t i = 0; i < total_elements; ++i)
            {
                float gate_val = gate_data[i];
                float up_val = up_data[i];

                // swish(x) = x * sigmoid(x)
                float swish_val = swish(gate_val);

                // Element-wise product
                output_data[i] = swish_val * up_val;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_DEBUG("MPISwiGLUBatch: " << ms << "ms for " << total_elements << " elements"
                                     << " (batch=" << batch_size << ", seq=" << seq_len << ", hidden_ff=" << hidden_ff << ")");

        // === POST-COMPUTATION VALIDATION ===
        ASSERT_TENSOR_NOT_NAN(output, "SwiGLUBatch output after computation");

        // Log output tensor statistics
        TensorLogger::logTensorStats(output, "SwiGLUBatch final_output", "MPISwiGLUBatchOperator_COMPLETE");

        LOG_DEBUG("MPISwiGLUBatchOperator executed successfully on rank " << getRank());
        return true;
    }

    bool MPISwiGLUBatchOperator::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                          const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        // Basic validation
        if (inputs.size() != 2 || outputs.size() != 1)
        {
            LOG_ERROR("MPISwiGLUBatchOperator: Expected 2 inputs and 1 output, got "
                      << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        auto gate = inputs[0];
        auto up = inputs[1];
        auto output = outputs[0];

        if (!gate || !up || !output)
        {
            LOG_ERROR("MPISwiGLUBatchOperator: Null tensor provided - gate: " << (gate ? "valid" : "null")
                                                                              << ", up: " << (up ? "valid" : "null")
                                                                              << ", output: " << (output ? "valid" : "null"));
            return false;
        }

        // Check gate is 3D [batch, seq_len, hidden_ff]
        if (gate->shape().size() != 3)
        {
            LOG_ERROR("MPISwiGLUBatchOperator: Gate must be 3D [batch, seq_len, hidden_ff], got " << gate->shape().size() << " dimensions");
            return false;
        }

        // Check up is 3D [batch, seq_len, hidden_ff]
        if (up->shape().size() != 3)
        {
            LOG_ERROR("MPISwiGLUBatchOperator: Up must be 3D [batch, seq_len, hidden_ff], got " << up->shape().size() << " dimensions");
            return false;
        }

        // Check dimensions match
        if (gate->shape()[0] != up->shape()[0] ||
            gate->shape()[1] != up->shape()[1] ||
            gate->shape()[2] != up->shape()[2])
        {
            LOG_ERROR("MPISwiGLUBatchOperator: Gate and up shape mismatch - gate: ["
                      << gate->shape()[0] << ", " << gate->shape()[1] << ", " << gate->shape()[2]
                      << "], up: [" << up->shape()[0] << ", " << up->shape()[1] << ", " << up->shape()[2] << "]");
            return false;
        }

        // Check output shape matches input
        if (output->shape().size() != 3 ||
            output->shape()[0] != gate->shape()[0] ||
            output->shape()[1] != gate->shape()[1] ||
            output->shape()[2] != gate->shape()[2])
        {
            LOG_ERROR("MPISwiGLUBatchOperator: Output shape mismatch - expected ["
                      << gate->shape()[0] << ", " << gate->shape()[1] << ", " << gate->shape()[2]
                      << "], got [" << output->shape()[0] << ", " << output->shape()[1] << ", " << output->shape()[2] << "]");
            return false;
        }

        return true;
    }

    void MPISwiGLUBatchOperator::configureOpenMPThreading(size_t total_elements)
    {
        // Use all available threads for large problems
        // For small problems, reduce thread count to avoid overhead
        int optimal_threads = num_threads_;

        if (total_elements < 1024)
        {
            optimal_threads = 1; // Single thread for tiny problems
        }
        else if (total_elements < 16384)
        {
            optimal_threads = std::min(2, num_threads_); // Max 2 threads for small problems
        }

        omp_set_num_threads(optimal_threads);

        LOG_DEBUG("Configured OpenMP threads: " << optimal_threads << " for " << total_elements << " elements");
    }

    inline float MPISwiGLUBatchOperator::fastSigmoid(float x) const
    {
        // Fast sigmoid approximation for better performance
        // Accurate for typical ranges, saturates for extreme values
        if (x < -20.0f)
            return 0.0f;
        if (x > 20.0f)
            return 1.0f;

        // Standard sigmoid for normal range
        return 1.0f / (1.0f + std::exp(-x));
    }

    inline float MPISwiGLUBatchOperator::swish(float x) const
    {
        // swish(x) = x * sigmoid(x)
        return x * fastSigmoid(x);
    }

} // namespace llaminar
