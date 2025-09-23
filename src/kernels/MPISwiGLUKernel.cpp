#include "MPISwiGLUKernel.h"
#include "../debug_utils.h"
#include "../performance_timer.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <omp.h>

namespace llaminar
{

    MPISwiGLUKernel::MPISwiGLUKernel(DistributionStrategy strategy)
        : MPIKernelBase(), strategy_(strategy), num_threads_(omp_get_max_threads())
    {
        LOG_DEBUG("MPISwiGLUKernel initialized on rank " << getRank() << "/" << getSize()
                                                         << " with strategy: " << (strategy == DistributionStrategy::SEQUENCE_WISE ? "SEQUENCE_WISE" : "FEATURE_WISE")
                                                         << ", OpenMP threads: " << num_threads_);
    }

    bool MPISwiGLUKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                  std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        PERF_SCOPED_TIMER("MPISwiGLUKernel::execute");

        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPISwiGLUKernel: Validation failed");
            return false;
        }

        auto gate_tensor = inputs[0];
        auto up_tensor = inputs[1];
        auto output_tensor = outputs[0];

        const auto &gate_shape = gate_tensor->shape();
        int seq_len = gate_shape[0];
        int d_ff = gate_shape[1];

        // Configure OpenMP threading based on problem size
        size_t total_elements = seq_len * d_ff;
        configureOpenMPThreading(total_elements);

        const float *gate_data = gate_tensor->data();
        const float *up_data = up_tensor->data();
        float *output_data = output_tensor->data();

        // === SWIGLU INPUT VALIDATION ===
        ASSERT_TENSOR_VALID(gate_tensor, "SwiGLU gate input");
        ASSERT_TENSOR_VALID(up_tensor, "SwiGLU up input");
        ASSERT_TENSOR_NOT_NAN(gate_tensor, "SwiGLU gate input has NaN");
        ASSERT_TENSOR_NOT_NAN(up_tensor, "SwiGLU up input has NaN");
        TensorLogger::logTensorStats(gate_tensor, "swiglu_gate_input");
        TensorLogger::logTensorStats(up_tensor, "swiglu_up_input");

        auto start_time = std::chrono::high_resolution_clock::now();

        // Execute based on distribution strategy
        try
        {
            switch (strategy_)
            {
            case DistributionStrategy::SEQUENCE_WISE:
                executeSequenceWise(gate_data, up_data, output_data, seq_len, d_ff);
                break;
            case DistributionStrategy::FEATURE_WISE:
                executeFeatureWise(gate_data, up_data, output_data, seq_len, d_ff);
                break;
            default:
                LOG_ERROR("MPISwiGLUKernel: Unknown distribution strategy");
                return false;
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("MPISwiGLUKernel: Execution failed: " << e.what());
            return false;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        // === SWIGLU OUTPUT VALIDATION ===
        ASSERT_TENSOR_NOT_NAN(output_tensor, "SwiGLU output has NaN");
        TensorLogger::logTensorStats(output_tensor, "swiglu_output");

        LOG_DEBUG("MPISwiGLUKernel executed: " << seq_len << "x" << d_ff
                                               << " in " << std::fixed << std::setprecision(2) << execution_time
                                               << "ms on rank " << getRank() << " (threads: " << omp_get_max_threads() << ")");

        return true;
    }

    bool MPISwiGLUKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                   const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        // Check input count
        if (inputs.size() != 2)
        {
            LOG_ERROR("MPISwiGLUKernel: Expected 2 inputs (gate, up), got " << inputs.size());
            return false;
        }

        // Check output count
        if (outputs.size() != 1)
        {
            LOG_ERROR("MPISwiGLUKernel: Expected 1 output (result), got " << outputs.size());
            return false;
        }

        // Validate input tensors
        auto gate_tensor = inputs[0];
        auto up_tensor = inputs[1];

        if (!gate_tensor || !up_tensor)
        {
            LOG_ERROR("MPISwiGLUKernel: Input tensors are null");
            return false;
        }

        const auto &gate_shape = gate_tensor->shape();
        const auto &up_shape = up_tensor->shape();

        // Check tensor dimensions
        if (gate_shape.size() != 2 || up_shape.size() != 2)
        {
            LOG_ERROR("MPISwiGLUKernel: Input tensors must be 2D, got gate: "
                      << gate_shape.size() << "D, up: " << up_shape.size() << "D");
            return false;
        }

        // Check shape compatibility
        if (gate_shape[0] != up_shape[0] || gate_shape[1] != up_shape[1])
        {
            LOG_ERROR("MPISwiGLUKernel: Input shape mismatch - gate: ["
                      << gate_shape[0] << ", " << gate_shape[1] << "], up: ["
                      << up_shape[0] << ", " << up_shape[1] << "]");
            return false;
        }

        // Validate output tensor
        auto output_tensor = outputs[0];
        if (!output_tensor)
        {
            LOG_ERROR("MPISwiGLUKernel: Output tensor is null");
            return false;
        }

        const auto &output_shape = output_tensor->shape();
        if (output_shape.size() != 2)
        {
            LOG_ERROR("MPISwiGLUKernel: Output tensor must be 2D, got " << output_shape.size() << "D");
            return false;
        }

        // Check output shape matches input
        if (output_shape[0] != gate_shape[0] || output_shape[1] != gate_shape[1])
        {
            LOG_ERROR("MPISwiGLUKernel: Output shape mismatch - expected: ["
                      << gate_shape[0] << ", " << gate_shape[1] << "], got: ["
                      << output_shape[0] << ", " << output_shape[1] << "]");
            return false;
        }

        return true;
    }

    void MPISwiGLUKernel::setDistributionStrategy(DistributionStrategy strategy)
    {
        strategy_ = strategy;
        LOG_DEBUG("MPISwiGLUKernel: Distribution strategy changed to "
                  << (strategy == DistributionStrategy::SEQUENCE_WISE ? "SEQUENCE_WISE" : "FEATURE_WISE"));
    }

    void MPISwiGLUKernel::configureOpenMPThreading(size_t tensor_size)
    {
        // Small operations: single-threaded to avoid overhead
        if (tensor_size < 8192)
        {
            omp_set_num_threads(1);
            LOG_DEBUG("MPISwiGLUKernel: Using single thread for small tensor (" << tensor_size << " elements)");
            return;
        }

        // Medium to large operations: use available threads
        int max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
        LOG_DEBUG("MPISwiGLUKernel: Using " << max_threads << " threads for tensor (" << tensor_size << " elements)");
    }

    void MPISwiGLUKernel::distributeMPIWork(size_t total_elements, size_t &start_idx, size_t &end_idx) const
    {
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

        LOG_DEBUG("MPISwiGLUKernel: Rank " << rank << " processing elements ["
                                           << start_idx << ", " << end_idx << ") of " << total_elements);
    }

    void MPISwiGLUKernel::executeSequenceWise(const float *gate_data, const float *up_data, float *output_data,
                                              int seq_len, int d_ff)
    {
        // Distribute sequence positions across MPI ranks
        size_t total_positions = seq_len;
        size_t start_pos, end_pos;
        distributeMPIWork(total_positions, start_pos, end_pos);

        // Process assigned sequence positions with OpenMP parallelization
        auto omp_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2) schedule(static)
        for (size_t pos = start_pos; pos < end_pos; ++pos)
        {
            for (int dim = 0; dim < d_ff; dim += 8)
            { // Process in chunks for better vectorization
                int end_dim = std::min(dim + 8, d_ff);

#pragma omp simd aligned(gate_data, up_data, output_data : 32)
                for (int k = dim; k < end_dim; ++k)
                {
                    size_t idx = pos * d_ff + k;
                    float up_val = up_data[idx];
                    float silu_val = computeSiLU(up_val);
                    output_data[idx] = gate_data[idx] * silu_val;
                }
            }
        }

        auto omp_end = std::chrono::high_resolution_clock::now();
        double omp_time = std::chrono::duration<double, std::milli>(omp_end - omp_start).count();
        LOG_DEBUG("MPISwiGLUKernel sequence-wise OpenMP: " << omp_time << "ms, threads: " << omp_get_max_threads());

        // Synchronize all ranks before proceeding
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void MPISwiGLUKernel::executeFeatureWise(const float *gate_data, const float *up_data, float *output_data,
                                             int seq_len, int d_ff)
    {
        // Distribute feature dimensions across MPI ranks
        size_t total_features = d_ff;
        size_t start_feature, end_feature;
        distributeMPIWork(total_features, start_feature, end_feature);

        // Process assigned features across all sequence positions
        auto omp_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(static)
        for (int pos = 0; pos < seq_len; ++pos)
        {
            size_t pos_offset = pos * d_ff;

#pragma omp simd aligned(gate_data, up_data, output_data : 32)
            for (size_t dim = start_feature; dim < end_feature; ++dim)
            {
                size_t idx = pos_offset + dim;
                float up_val = up_data[idx];
                float silu_val = computeSiLU(up_val);
                output_data[idx] = gate_data[idx] * silu_val;
            }
        }

        auto omp_end = std::chrono::high_resolution_clock::now();
        double omp_time = std::chrono::duration<double, std::milli>(omp_end - omp_start).count();
        LOG_DEBUG("MPISwiGLUKernel feature-wise OpenMP: " << omp_time << "ms, threads: " << omp_get_max_threads());

        // Synchronize all ranks before proceeding
        MPI_Barrier(MPI_COMM_WORLD);
    }

    inline float MPISwiGLUKernel::computeSiLU(float x) const
    {
        // SiLU (Swish) activation: x / (1 + exp(-x))
        // Use stable computation to avoid overflow
        if (x > 20.0f)
        {
            return x; // For large x, silu(x) ≈ x
        }
        else if (x < -20.0f)
        {
            return 0.0f; // For very negative x, silu(x) ≈ 0
        }
        else
        {
            return x / (1.0f + std::exp(-x));
        }
    }

} // namespace llaminar