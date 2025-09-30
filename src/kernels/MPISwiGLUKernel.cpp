/**
 * @file MPISwiGLUKernel.cpp
 * @brief Applies SwiGLU feed-forward activation: (X * W1) ⊗ swish(X * W2) * W3 (variant) depending on architecture.
 *
 * @section Contract
 * Inputs (typical 3-projection form after prior linear kernels):
 *  - inputs[0]: Gate projection tensor G [seq_len, hidden_dim_ff] (pre-activation for swish).
 *  - inputs[1]: Up projection tensor U [seq_len, hidden_dim_ff] (multiplicative partner).
 *  - inputs[2] (optional): Down projection tensor D or may be handled by a subsequent linear kernel depending on config.
 * Outputs:
 *  - outputs[0]: Activated tensor A [seq_len, hidden_dim_ff] or reduced back to model_dim if fused-down path executed.
 * Formula (canonical):
 *  - swish(x) = x * sigmoid(x)
 *  - y = (swish(G) ⊗ U)  (Hadamard). Some model variants: y = swish(G) * U (elementwise) then linear down-projection outside this kernel.
 * Numerical Expectations:
 *  - Stable for |x| < 20; extreme values saturate sigmoid; relative error vs reference < 1e-6 float32.
 * Error Modes:
 *  - Shape mismatch between G and U.
 *  - Null tensors, inconsistent seq_len.
 * Distribution:
 *  - Currently replicated; future optimization could partition hidden_dim_ff across ranks with reduce-scatter + allgather.
 * Threading:
 *  - Parallel for over rows*columns; no shared mutable state beyond output buffer.
 * Performance Notes:
 *  - Memory bandwidth + elementwise compute; vectorization opportunities (FMA + fast sigmoid approx) considered later.
 * Future Extensions:
 *  - Quantized input handling, fused down-projection linear, activation checkpointing.
 * @warning Ensure that upstream linear projections have consistent floating layout to avoid silent divergence.
 * @author David Sanftenberg
 */
#include "MPISwiGLUKernel.h"
#include "../debug_utils.h"
#include "../performance_timer.h"
#include "../utils/debug_env.h"
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

        auto gate_tensor = inputs[0]; // Projection typically named "gate" in model weights
        auto up_tensor = inputs[1];   // Projection typically named "up" (a.k.a. value) in SwiGLU
        auto output_tensor = outputs[0];

        const auto &gate_shape = gate_tensor->shape();
        int seq_len = gate_shape[0];
        int d_ff = gate_shape[1];

        // Configure OpenMP threading based on problem size
        size_t total_elements = seq_len * d_ff;
        configureOpenMPThreading(total_elements);

        const bool gate_distributed = gate_tensor->is_distributed();
        const bool up_distributed = up_tensor->is_distributed();
        const bool output_distributed = output_tensor->is_distributed();
        const bool replicated_inputs = !gate_distributed && !up_distributed && !output_distributed;

        LOG_DEBUG("MPISwiGLUKernel tensor types: gate=" << gate_tensor->type_name()
                                                        << " up=" << up_tensor->type_name()
                                                        << " output=" << output_tensor->type_name()
                                                        << " distributed_flags=(" << (gate_distributed ? "D" : "R")
                                                        << "," << (up_distributed ? "D" : "R")
                                                        << "," << (output_distributed ? "D" : "R")
                                                        << ") replicated=" << (replicated_inputs ? "true" : "false"));

        const float *gate_data = gate_tensor->data();
        const float *up_data = up_tensor->data();
        float *output_data = output_tensor->data();

        // Algorithm selection (default = correct LLaMA style: silu(gate) * up)
        // Legacy (buggy) ordering previously used: gate * silu(up)
        //   Enable via: export LLAMINAR_SWIGLU_ALGO=legacy
        // Validation (compute both & report diff) via: export LLAMINAR_SWIGLU_VALIDATE=1
        static bool algo_initialized = false;
        static bool legacy_algo = false; // true => use legacy buggy ordering
        static bool validation_enabled = false;
        static int rank_cached = -1;
        if (!algo_initialized)
        {
            if (!debugEnv().swiglu.algo.empty())
            {
                std::string v = debugEnv().swiglu.algo;
                std::transform(v.begin(), v.end(), v.begin(), ::tolower);
                if (v == "legacy")
                    legacy_algo = true;
            }
            validation_enabled = debugEnv().swiglu.validate;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_cached);
            if (rank_cached == 0)
            {
                LOG_INFO("MPISwiGLUKernel SwiGLU algorithm: " << (legacy_algo ? "legacy(gate * silu(up))" : "standard(silu(gate) * up)")
                                                              << (validation_enabled ? " (validation enabled)" : ""));
            }
            algo_initialized = true;
        }

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
                executeSequenceWise(gate_data, up_data, output_data, seq_len, d_ff, replicated_inputs);
                break;
            case DistributionStrategy::FEATURE_WISE:
                executeFeatureWise(gate_data, up_data, output_data, seq_len, d_ff, replicated_inputs);
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
                                              int seq_len, int d_ff, bool replicated_inputs)
    {
        // Distribute sequence positions across MPI ranks
        size_t total_positions = seq_len;
        size_t start_pos = 0;
        size_t end_pos = total_positions;

        if (!replicated_inputs)
        {
            distributeMPIWork(total_positions, start_pos, end_pos);
        }
        else
        {
            LOG_DEBUG("MPISwiGLUKernel: Replicated tensors detected; each rank processing full range [0, "
                      << total_positions << ")");
        }

        // Process assigned sequence positions with OpenMP parallelization
        auto omp_start = std::chrono::high_resolution_clock::now();

        const bool legacy_algo = (!debugEnv().swiglu.algo.empty() && debugEnv().swiglu.algo == "legacy");
        const bool validate = debugEnv().swiglu.validate;
        double accum_abs_diff = 0.0;
        double accum_max_diff = 0.0;

#pragma omp parallel for collapse(2) schedule(static) reduction(+ : accum_abs_diff) reduction(max : accum_max_diff)
        for (size_t pos = start_pos; pos < end_pos; ++pos)
        {
            for (int dim = 0; dim < d_ff; dim += 8)
            { // Process in chunks for better vectorization
                int end_dim = std::min(dim + 8, d_ff);

#pragma omp simd aligned(gate_data, up_data, output_data : 32) reduction(+ : accum_abs_diff) reduction(max : accum_max_diff)
                for (int k = dim; k < end_dim; ++k)
                {
                    size_t idx = pos * d_ff + k;
                    float gate_v = gate_data[idx];
                    float up_v = up_data[idx];
                    float out_val;
                    if (legacy_algo)
                    {
                        // Legacy buggy ordering: gate * silu(up)
                        out_val = gate_v * computeSiLU(up_v);
                        if (validate)
                        {
                            float correct = computeSiLU(gate_v) * up_v;
                            float diff = std::fabs(static_cast<double>(out_val) - static_cast<double>(correct));
                            accum_abs_diff += diff;
                            if (diff > accum_max_diff)
                                accum_max_diff = diff;
                        }
                    }
                    else
                    {
                        // Correct LLaMA formulation: silu(gate) * up
                        out_val = computeSiLU(gate_v) * up_v;
                        if (validate)
                        {
                            float legacy = gate_v * computeSiLU(up_v);
                            float diff = std::fabs(static_cast<double>(out_val) - static_cast<double>(legacy));
                            accum_abs_diff += diff;
                            if (diff > accum_max_diff)
                                accum_max_diff = diff;
                        }
                    }
                    output_data[idx] = out_val;
                }
            }
        }

        if (validate)
        {
            int world_rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            double global_abs = 0.0;
            double global_max = 0.0;
            MPI_Allreduce(&accum_abs_diff, &global_abs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&accum_max_diff, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            size_t local_elems = (end_pos - start_pos) * static_cast<size_t>(d_ff);
            size_t global_elems = static_cast<size_t>(seq_len) * static_cast<size_t>(d_ff);
            double mean_abs = global_abs / static_cast<double>(global_elems);
            if (world_rank == 0)
            {
                LOG_INFO("[SwiGLUValidate] algo=" << (legacy_algo ? "legacy_vs_correct" : "correct_vs_legacy")
                                                  << " mean_abs_diff=" << mean_abs
                                                  << " max_diff=" << global_max
                                                  << " elems=" << global_elems);
            }
        }

        auto omp_end = std::chrono::high_resolution_clock::now();
        double omp_time = std::chrono::duration<double, std::milli>(omp_end - omp_start).count();
        LOG_DEBUG("MPISwiGLUKernel sequence-wise OpenMP: " << omp_time << "ms, threads: " << omp_get_max_threads());

        // Synchronize all ranks before proceeding
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void MPISwiGLUKernel::executeFeatureWise(const float *gate_data, const float *up_data, float *output_data,
                                             int seq_len, int d_ff, bool replicated_inputs)
    {
        // Distribute feature dimensions across MPI ranks
        size_t total_features = d_ff;
        size_t start_feature = 0;
        size_t end_feature = total_features;

        if (!replicated_inputs)
        {
            distributeMPIWork(total_features, start_feature, end_feature);
        }
        else
        {
            LOG_DEBUG("MPISwiGLUKernel: Replicated tensors detected; each rank processing full range [0, "
                      << total_features << ")");
        }

        // Process assigned features across all sequence positions
        auto omp_start = std::chrono::high_resolution_clock::now();

        const bool legacy_algo = (!debugEnv().swiglu.algo.empty() && debugEnv().swiglu.algo == "legacy");
        const bool validate = debugEnv().swiglu.validate;
        double accum_abs_diff = 0.0;
        double accum_max_diff = 0.0;

#pragma omp parallel for schedule(static) reduction(+ : accum_abs_diff) reduction(max : accum_max_diff)
        for (int pos = 0; pos < seq_len; ++pos)
        {
            size_t pos_offset = static_cast<size_t>(pos) * static_cast<size_t>(d_ff);

#pragma omp simd aligned(gate_data, up_data, output_data : 32) reduction(+ : accum_abs_diff) reduction(max : accum_max_diff)
            for (size_t dim = start_feature; dim < end_feature; ++dim)
            {
                size_t idx = pos_offset + dim;
                float gate_v = gate_data[idx];
                float up_v = up_data[idx];
                float out_val;
                if (legacy_algo)
                {
                    out_val = gate_v * computeSiLU(up_v);
                    if (validate)
                    {
                        float correct = computeSiLU(gate_v) * up_v;
                        float diff = std::fabs(static_cast<double>(out_val) - static_cast<double>(correct));
                        accum_abs_diff += diff;
                        if (diff > accum_max_diff)
                            accum_max_diff = diff;
                    }
                }
                else
                {
                    out_val = computeSiLU(gate_v) * up_v;
                    if (validate)
                    {
                        float legacy = gate_v * computeSiLU(up_v);
                        float diff = std::fabs(static_cast<double>(out_val) - static_cast<double>(legacy));
                        accum_abs_diff += diff;
                        if (diff > accum_max_diff)
                            accum_max_diff = diff;
                    }
                }
                output_data[idx] = out_val;
            }
        }

        if (validate)
        {
            int world_rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            double global_abs = 0.0;
            double global_max = 0.0;
            MPI_Allreduce(&accum_abs_diff, &global_abs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&accum_max_diff, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            size_t shard_elems = static_cast<size_t>(seq_len) * (end_feature - start_feature);
            size_t global_elems = static_cast<size_t>(seq_len) * static_cast<size_t>(d_ff);
            double mean_abs = global_abs / static_cast<double>(global_elems);
            if (world_rank == 0)
            {
                LOG_INFO("[SwiGLUValidate] algo=" << (legacy_algo ? "legacy_vs_correct" : "correct_vs_legacy")
                                                  << " mean_abs_diff=" << mean_abs
                                                  << " max_diff=" << global_max
                                                  << " elems=" << global_elems);
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