/**
 * @file MPILinearBatchOperator.cpp
 * @brief MPI-aware column-partitioned linear projection kernel with native batch support.
 *
 * @section Contract
 * Inputs:
 *  - inputs[0]: Activations tensor [batch, seq_len, in_dim] (row-major, replicated on all ranks).
 *  - inputs[1]: Global weight tensor [out_dim, in_dim] (row-major; will be column-partition distributed).
 *  - inputs[2] (optional): Global bias [out_dim].
 * Outputs:
 *  - outputs[0]: Global output tensor [batch, seq_len, out_dim] (row-major; assembled via Allgatherv).
 *
 * Weight Convention:
 *  - Weights are stored as [out_features, in_features] matching PyTorch nn.Linear and GGUF format.
 *  - Applied as output = input @ weight^T (transpose during matmul).
 *  - See docs/WEIGHT_MATRIX_CONVENTIONS.md for detailed rationale.
 *
 * Distribution Strategy:
 *  - Weight columns are block-distributed across ranks; activations are replicated.
 *  - For each batch: local GEMM: [batch*seq_len, in_dim] @ [in_dim, out_dim_local]^T -> [batch*seq_len, out_dim_local].
 *  - Optional local bias add then global gather.
 *
 * Batch Processing:
 *  - Flattens batch and sequence dimensions for matmul: [B, T, D] -> [B*T, D]
 *  - Processes all sequences in batch together (efficient BLAS usage)
 *  - Reshapes result back to [B, T, D_out] after computation
 *
 * Numerical Expectations:
 *  - Deterministic for identical OpenMP scheduling when OMP_NUM_THREADS=1.
 *  - Accumulation in float; differences vs single-process reference bounded by floating reduction order.
 *  - Parity guarantee: batch=1 produces identical results to MPILinearOperator (within float32 precision).
 *
 * Error Modes:
 *  - Shape mismatches, null tensors, NaN detection trigger logged error and return false.
 *  - MPI distribution inconsistencies (size mismatch) abort execution.
 *
 * Threading:
 *  - OpenMP parallelism inside adaptiveMatMul; environment may cap threads.
 *  - No concurrent mutation of shared state beyond local temporaries.
 *
 * MPI:
 *  - Uses rank/size from MPIOperatorBase; collective communications must remain ordered.
 *
 * @author David Sanftenberg
 */
#include "MPILinearBatchOperator.h"
#include "../Logger.h"
#include "../PerformanceTracer.h"
#include "../DebugUtils.h"
#include "../AdaptiveMatmul.h"
#include "../utils/DebugEnv.h"
#include "../tensors/SimpleTensor.h"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <omp.h>

namespace llaminar
{

    MPILinearBatchOperator::MPILinearBatchOperator(MPI_Comm comm) : MPIOperatorBase(comm, false)
    {
        LOG_DEBUG("MPILinearBatchOperator initialized on rank " << getRank() << " of " << getSize());
    }

    bool MPILinearBatchOperator::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                         std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        PERF_TRACE_SCOPE_CAT("mpi_linear_batch_execute", "linear");

        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPILinearBatchOperator validation failed on rank " << getRank());
            return false;
        }

        auto input = inputs[0];
        auto weight_input = inputs[1]; // May be pre-sliced or replicated
        auto global_output = outputs[0];

        // === COMPREHENSIVE TENSOR VALIDATION ===
        ASSERT_TENSOR_VALID(input, "LinearBatch input");
        ASSERT_TENSOR_VALID(weight_input, "LinearBatch weight");
        ASSERT_TENSOR_VALID(global_output, "LinearBatch output");

        // Check for NaN in inputs before computation
        ASSERT_TENSOR_NOT_NAN(input, "LinearBatch input");
        ASSERT_TENSOR_NOT_NAN(weight_input, "LinearBatch weight");

        // Log detailed tensor information
        TensorLogger::logMatMulOperation(input, weight_input, global_output, "MPILinearBatchOperator");

        // Extract dimensions
        size_t batch_size = input->shape()[0];
        size_t seq_len = input->shape()[1];
        size_t input_size = input->shape()[2];

        // Weight dimensions from input tensor
        size_t weight_out_dim = weight_input->shape()[0];
        size_t weight_in_dim = weight_input->shape()[1];

        // Validate dimension compatibility
        if (weight_in_dim != input_size)
        {
            LOG_ERROR("MPILinearBatch: Weight input dimension mismatch - weight[" << weight_input->shape()[0] << ", " << weight_input->shape()[1] << "] vs input[" << batch_size << ", " << seq_len << ", " << input_size << "]");
            return false;
        }

        // Determine if weight is pre-sliced or replicated by checking global output size
        // If weight is pre-sliced, weight_out_dim will match our local partition size
        // If weight is replicated, weight_out_dim will match the full global output size
        size_t global_output_size = global_output->shape()[2];
        auto [expected_local_size, output_offset] = getRowDistribution(global_output_size);

        bool is_pre_sliced = (weight_out_dim == expected_local_size);
        bool is_replicated = (weight_out_dim == global_output_size);

        if (!is_pre_sliced && !is_replicated)
        {
            LOG_ERROR("MPILinearBatch: Weight dimension " << weight_out_dim
                                                          << " doesn't match expected local (" << expected_local_size
                                                          << ") or global (" << global_output_size << ") size");
            return false;
        }

        std::shared_ptr<TensorBase> local_weight;

        if (is_pre_sliced)
        {
            // Weight is already sliced - use directly
            PERF_TRACE_SCOPE_CAT("weight_pre_sliced", "linear_kernel");
            local_weight = weight_input;

            if (getRank() == 0)
            {
                LOG_DEBUG("MPILinearBatch: Using PRE-SLICED weight [" << weight_out_dim << "," << weight_in_dim << "]");
            }
        }
        else // is_replicated
        {
            // Weight is replicated - need to distribute
            const float *weight_key = weight_input->data();

            auto weight_cache_it = weight_cache_.find(weight_key);
            if (weight_cache_it != weight_cache_.end())
            {
                // Cache hit: reuse previously distributed weight
                PERF_TRACE_SCOPE_CAT("weight_cache_hit", "linear_kernel");
                local_weight = weight_cache_it->second;
            }
            else
            {
                // Cache miss: distribute and cache the weight
                PERF_TRACE_SCOPE_CAT("weight_cache_miss", "linear_kernel");
                local_weight = createLocalTensor({static_cast<size_t>(expected_local_size), input_size});
                {
                    PERF_TRACE_SCOPE_CAT("distribute_weight", "linear_kernel");
                    distributeWeight(weight_input, local_weight, global_output_size);
                }
                weight_cache_[weight_key] = local_weight;
            }

            if (getRank() == 0)
            {
                LOG_DEBUG("MPILinearBatch: Distributed REPLICATED weight to local [" << expected_local_size << "," << weight_in_dim << "]");
            }
        }

        // Create local output tensor [batch, seq_len, local_out_dim]
        auto local_output = createLocalTensor({batch_size, seq_len, static_cast<size_t>(expected_local_size)});

        // Handle optional bias with caching
        std::shared_ptr<TensorBase> local_bias = nullptr;
        if (inputs.size() >= 3 && inputs[2])
        {
            const float *bias_key = inputs[2]->data();
            auto bias_cache_it = bias_cache_.find(bias_key);

            if (bias_cache_it != bias_cache_.end())
            {
                // Cache hit: reuse previously distributed bias
                PERF_TRACE_SCOPE_CAT("bias_cache_hit", "linear_kernel");
                local_bias = bias_cache_it->second;
            }
            else
            {
                // Cache miss: distribute and cache the bias
                PERF_TRACE_SCOPE_CAT("bias_cache_miss", "linear_kernel");
                local_bias = createLocalTensor({static_cast<size_t>(expected_local_size)});
                {
                    PERF_TRACE_SCOPE_CAT("distribute_bias", "linear_kernel");
                    distributeBias(inputs[2], local_bias, global_output_size);
                }
                bias_cache_[bias_key] = local_bias;
            }
        }

        // Perform local computation using adaptive matmul
        // Flatten batch dimension: [B, T, D] -> [B*T, D] for efficient BLAS
        const float *input_data = input->data();
        const float *weight_data = local_weight->data();
        float *output_data = local_output->data();

        int total_seq = static_cast<int>(batch_size * seq_len);
        int d_model = static_cast<int>(input_size);
        int d_out = static_cast<int>(expected_local_size);

        // Optional lightweight diagnostics
        bool linear_diag = debugEnv().linear.diag;
        if (linear_diag && getRank() == 0)
        {
            LOG_DEBUG("[LinearBatchDiag] rank=" << getRank() << " batch=" << batch_size
                                                << " seq_len=" << seq_len << " total_seq=" << total_seq
                                                << " d_model=" << d_model << " d_out_local=" << d_out
                                                << " weight_mode=" << (is_pre_sliced ? "PRE_SLICED" : "REPLICATED"));
        }

        // Use adaptive matrix multiplication
        auto start = std::chrono::high_resolution_clock::now();
        bool matmul_success;
        {
            PERF_TRACE_SCOPE_CAT("linear_batch_matmul", "linear_kernel");
            // Matrix multiplication: output = input @ weight^T
            // Weight is [local_out_dim, in_dim], so we transpose it during matmul
            // Input is effectively [batch*seq_len, in_dim]

            // DEBUG: Log ALL parameters to see what we're getting
            static int call_count = 0;
            if (getRank() == 0 && call_count < 10)
            {
                LOG_DEBUG("[MPILinearBatch_CALL_" << call_count << "] rank=0:");
                LOG_DEBUG("  batch=" << batch_size << " seq=" << seq_len << " total_seq=" << total_seq);
                LOG_DEBUG("  d_model(k)=" << d_model << " d_out(n)=" << d_out);
                LOG_DEBUG("  input_size=" << input_size << " global_output_size=" << global_output_size
                                          << " expected_local_size=" << expected_local_size << " output_offset=" << output_offset);
                call_count++;
            }

            // Use new TensorBase* overload for potential fused quantized GEMM
            matmul_success = adaptiveMatMul(input_data, local_weight.get(), output_data,
                                            total_seq, d_out, d_model,
                                            /*is_prefill*/ false,
                                            /*distributed_partition*/ true,
                                            /*transpose_A*/ false,
                                            /*transpose_B*/ true, // CRITICAL: Transpose weight per PyTorch/GGUF convention
                                            /*alpha*/ 1.0f,
                                            /*beta*/ 0.0f);
        }
        auto end = std::chrono::high_resolution_clock::now();

        if (!matmul_success)
        {
            LOG_ERROR("Adaptive matrix multiplication failed on rank " << getRank());
            return false;
        }

        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_DEBUG("MPILinearBatch matmul: " << ms << "ms for " << total_seq << "x" << d_model << " * " << d_model << "x" << d_out);

        // Add bias if provided (broadcast across batch and sequence)
        if (local_bias)
        {
            PERF_TRACE_SCOPE_CAT("add_bias_batch", "linear_kernel");
            addBiasLocal(local_output->data(), local_bias->data(), batch_size, seq_len, expected_local_size);
        }

        // Gather results from all processes
        {
            PERF_TRACE_SCOPE_CAT("gather_output_batch", "mpi_collective");
            gatherOutput(local_output, global_output, batch_size, seq_len, global_output_size);
        }

        if (linear_diag && getRank() == 0)
        {
            LOG_DEBUG("[LinearBatchDiag] rank=" << getRank() << " gathered_global_shape=[" << batch_size << "," << seq_len << "," << global_output_size << "]");
        }

        // === POST-COMPUTATION VALIDATION ===
        ASSERT_TENSOR_NOT_NAN(global_output, "LinearBatch output after computation");

        // Log output tensor statistics
        TensorLogger::logTensorStats(global_output, "LinearBatch final_output", "MPILinearBatchOperator_COMPLETE");

        LOG_DEBUG("MPILinearBatchOperator executed successfully on rank " << getRank());
        return true;
    }

    bool MPILinearBatchOperator::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                          const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        // Basic validation
        if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1)
        {
            LOG_ERROR("MPILinearBatchOperator: Expected 2-3 inputs and 1 output, got "
                      << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        auto input = inputs[0];
        auto weight = inputs[1];
        auto output = outputs[0];

        if (!input || !weight || !output)
        {
            LOG_ERROR("MPILinearBatchOperator: Null tensor provided - input: " << (input ? "valid" : "null")
                                                                               << ", weight: " << (weight ? "valid" : "null")
                                                                               << ", output: " << (output ? "valid" : "null"));
            return false;
        }

        // Check input is 3D [batch, seq_len, in_dim]
        if (input->shape().size() != 3)
        {
            LOG_ERROR("MPILinearBatchOperator: Input must be 3D [batch, seq_len, in_dim], got " << input->shape().size() << " dimensions");
            return false;
        }

        // Check weight is 2D [out_dim, in_dim] per convention
        if (weight->shape().size() != 2)
        {
            LOG_ERROR("MPILinearBatchOperator: Weight must be 2D [out_dim, in_dim], got " << weight->shape().size() << " dimensions");
            return false;
        }

        // Check dimensions match: input[2] (in_dim) should match weight[1] (in_dim)
        if (input->shape()[2] != weight->shape()[1])
        {
            LOG_ERROR("MPILinearBatchOperator: Input size " << input->shape()[2]
                                                            << " doesn't match weight input size " << weight->shape()[1]
                                                            << " (weight shape=[" << weight->shape()[0] << ", " << weight->shape()[1] << "])");
            return false;
        }

        // Check output is 3D [batch, seq_len, out_dim]
        // output[2] should match weight[0] (out_dim)
        if (output->shape().size() != 3 ||
            output->shape()[0] != input->shape()[0] ||
            output->shape()[1] != input->shape()[1] ||
            output->shape()[2] != weight->shape()[0])
        {
            LOG_ERROR("MPILinearBatchOperator: Output shape mismatch - expected [" << input->shape()[0]
                                                                                   << ", " << input->shape()[1]
                                                                                   << ", " << weight->shape()[0]
                                                                                   << "], got [" << output->shape()[0] << ", " << output->shape()[1] << ", " << output->shape()[2] << "]");
            return false;
        }

        // Check optional bias
        if (inputs.size() == 3 && inputs[2])
        {
            auto bias = inputs[2];
            if (bias->shape().size() != 1 || bias->shape()[0] != weight->shape()[0])
            {
                LOG_ERROR("MPILinearBatchOperator: Bias shape mismatch - expected [" << weight->shape()[0] << "], got [" << bias->shape()[0] << "]");
                return false;
            }
        }

        return true;
    }

    void MPILinearBatchOperator::distributeWeight(const std::shared_ptr<TensorBase> &global_weight,
                                                  std::shared_ptr<TensorBase> &local_weight,
                                                  size_t output_size)
    {
        PERF_TRACE_SCOPE_CAT("distribute_weight_batch", "mpi_collective");

        auto [local_output_size, output_offset] = getRowDistribution(output_size);
        size_t input_size = global_weight->shape()[1];

        // Extract local rows from global weight
        // Global weight is [out_dim, in_dim], we want rows [offset:offset+local_size]
        const float *global_data = global_weight->data();
        float *local_data = local_weight->data();

        // Copy local portion of weight matrix
        size_t src_offset = output_offset * input_size;
        size_t copy_size = local_output_size * input_size * sizeof(float);

        std::memcpy(local_data, global_data + src_offset, copy_size);

        LOG_DEBUG("Rank " << getRank() << " distributed weight: local_shape=[" << local_output_size << ", " << input_size << "] from global offset " << output_offset);
    }

    void MPILinearBatchOperator::distributeBias(const std::shared_ptr<TensorBase> &global_bias,
                                                std::shared_ptr<TensorBase> &local_bias,
                                                size_t output_size)
    {
        PERF_TRACE_SCOPE_CAT("distribute_bias_batch", "mpi_collective");

        auto [local_output_size, output_offset] = getRowDistribution(output_size);

        // Extract local portion of bias vector
        const float *global_data = global_bias->data();
        float *local_data = local_bias->data();

        std::memcpy(local_data, global_data + output_offset, local_output_size * sizeof(float));

        LOG_DEBUG("Rank " << getRank() << " distributed bias: local_size=" << local_output_size << " from global offset " << output_offset);
    }

    void MPILinearBatchOperator::gatherOutput(const std::shared_ptr<TensorBase> &local_output,
                                              std::shared_ptr<TensorBase> &global_output,
                                              size_t batch_size, size_t seq_len, size_t output_size)
    {
        PERF_TRACE_SCOPE_CAT("gather_output_batch", "mpi_collective");

        auto [local_output_size, output_offset] = getRowDistribution(output_size);

        // Calculate per-position receive counts and displacements
        // Each rank contributes a different chunk of the output dimension
        std::vector<int> recvcounts_per_pos(getSize());
        std::vector<int> displs_per_pos(getSize());

        for (int r = 0; r < getSize(); ++r)
        {
            auto [rank_local_size, rank_offset] = getRowDistribution(output_size, r);
            recvcounts_per_pos[r] = rank_local_size; // Elements per rank for this position
            displs_per_pos[r] = rank_offset;         // Offset in output dimension
        }

        // Tensor layout is [batch, seq_len, out_dim] in row-major order.
        // Each rank has local_output with layout [batch, seq_len, local_out_dim].
        // We need to gather the local_out_dim chunks from all ranks for each (batch, seq) position.
        //
        // Issue: MPI_Allgatherv with a single call would assume contiguous data blocks,
        // but our dimensions are interleaved in memory. Solution: gather each position separately.

        size_t total_positions = batch_size * seq_len;

        for (size_t pos = 0; pos < total_positions; ++pos)
        {
            // Pointers to this position's data
            const float *local_ptr = local_output->data() + pos * local_output_size;
            float *global_ptr = global_output->data() + pos * output_size;

            // Gather dimension chunks from all ranks into the correct position
            MPI_Allgatherv(local_ptr, local_output_size, MPI_FLOAT,
                           global_ptr, recvcounts_per_pos.data(), displs_per_pos.data(),
                           MPI_FLOAT, getComm());
        }

        LOG_DEBUG("Rank " << getRank() << " gathered output: " << total_positions
                          << " positions × " << local_output_size << " local dims = "
                          << (total_positions * local_output_size) << " total elements");
    }

    void MPILinearBatchOperator::addBiasLocal(float *output, const float *bias,
                                              size_t batch_size, size_t seq_len, size_t local_output_size)
    {
        PERF_TRACE_SCOPE_CAT("add_bias_batch", "linear_kernel");

        // Broadcast bias across batch and sequence dimensions
        // Output is [batch, seq_len, local_out_dim] in row-major order
        size_t total_positions = batch_size * seq_len;

#pragma omp parallel for
        for (size_t pos = 0; pos < total_positions; ++pos)
        {
            size_t offset = pos * local_output_size;
            for (size_t d = 0; d < local_output_size; ++d)
            {
                output[offset + d] += bias[d];
            }
        }

        LOG_DEBUG("Added bias to " << total_positions << " positions with " << local_output_size << " dims each");
    }

    std::shared_ptr<TensorBase> MPILinearBatchOperator::createLocalTensor(const std::vector<size_t> &shape)
    {
        // Convert size_t to int for SimpleTensor
        std::vector<int> int_shape(shape.begin(), shape.end());
        return std::make_shared<SimpleTensor>(int_shape);
    }

} // namespace llaminar
