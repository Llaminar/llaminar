/**
 * @file MPILinearKernel.cpp
 * @brief MPI-aware column-partitioned linear projection kernel.
 *
 * @section Contract
 * Inputs:
 *  - inputs[0]: Activations tensor [seq_len, in_dim] (row-major, replicated on all ranks).
 *  - inputs[1]: Global weight tensor [out_dim, in_dim] (row-major; will be column-partition distributed).
 *  - inputs[2] (optional): Global bias [out_dim].
 * Outputs:
 *  - outputs[0]: Global output tensor [seq_len, out_dim] (row-major; assembled via Allgatherv over column partitions).
 * Weight Convention:
 *  - Weights are stored as [out_features, in_features] matching PyTorch nn.Linear and GGUF format.
 *  - Applied as output = input @ weight^T (transpose during matmul).
 *  - See docs/WEIGHT_MATRIX_CONVENTIONS.md for detailed rationale.
 * Distribution Strategy:
 *  - Weight columns are block-distributed across ranks; activations are replicated.
 *  - Local GEMM: [seq_len, in_dim] @ [in_dim, out_dim_local]^T -> [seq_len, out_dim_local].
 *  - Optional local bias add then global gather.
 * Numerical Expectations:
 *  - Deterministic for identical OpenMP scheduling when OMP_NUM_THREADS=1.
 *  - Accumulation in float; differences vs single-process reference bounded by floating reduction order on final gather.
 * Error Modes:
 *  - Shape mismatches, null tensors, NaN detection trigger logged error and return false.
 *  - MPI distribution inconsistencies (size mismatch) abort execution.
 * Threading:
 *  - OpenMP parallelism inside adaptiveMatMul; environment may cap threads.
 *  - No concurrent mutation of shared state beyond local temporaries.
 * MPI:
 *  - Uses rank/size from MPIKernelBase; collective communications must remain ordered.
 * @author David Sanftenberg
 */
#include "MPILinearKernel.h"
#include "../logger.h"
#include "../debug_utils.h"
#include "../adaptive_matmul.h"
#include "../utils/debug_env.h"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <omp.h>

namespace llaminar
{

    MPILinearKernel::MPILinearKernel(MPI_Comm comm) : MPIKernelBase(comm, false)
    {
        LOG_DEBUG("MPILinearKernel initialized on rank " << getRank() << " of " << getSize());
    }

    bool MPILinearKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                  std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPILinearKernel validation failed on rank " << getRank());
            return false;
        }

        auto input = inputs[0];
        auto global_weight = inputs[1];
        auto global_output = outputs[0];

        // === COMPREHENSIVE TENSOR VALIDATION ===
        ASSERT_TENSOR_VALID(input, "Linear input");
        ASSERT_TENSOR_VALID(global_weight, "Linear weight");
        ASSERT_TENSOR_VALID(global_output, "Linear output");

        // Check for NaN in inputs before computation
        ASSERT_TENSOR_NOT_NAN(input, "Linear input");
        ASSERT_TENSOR_NOT_NAN(global_weight, "Linear weight");

        // Log detailed tensor information
        TensorLogger::logMatMulOperation(input, global_weight, global_output, "MPILinearKernel");

        // Extract dimensions
        size_t seq_len = input->shape()[0];
        size_t input_size = input->shape()[1];
        // Weight is [out_dim, in_dim] per new convention
        size_t output_size = global_weight->shape()[0];
        size_t weight_in_dim = global_weight->shape()[1];

        // Validate dimension compatibility
        if (weight_in_dim != input_size)
        {
            LOG_ERROR("MPILinear: Weight input dimension mismatch - weight[" << global_weight->shape()[0] << ", " << global_weight->shape()[1] << "] vs input[" << seq_len << ", " << input_size << "]");
            return false;
        }

        // Calculate local distribution
        auto [local_output_size, output_offset] = getRowDistribution(output_size);

        // Create local tensors - weight is [local_out_dim, in_dim]
        auto local_weight = createLocalTensor({static_cast<size_t>(local_output_size), input_size});
        auto local_output = createLocalTensor({seq_len, static_cast<size_t>(local_output_size)});

        // Distribute weight matrix
        distributeWeight(global_weight, local_weight, output_size);

        // Handle optional bias
        std::shared_ptr<TensorBase> local_bias = nullptr;
        if (inputs.size() >= 3 && inputs[2])
        {
            local_bias = createLocalTensor({static_cast<size_t>(local_output_size)});
            distributeBias(inputs[2], local_bias, output_size);
        }

        // Ensure input is available on all processes (broadcast if needed)
        // For simplicity, assuming input is already replicated across processes

        // Perform local computation using COSMA
        // Matrix multiplication: local_output = input * local_weight
        // Use adaptive matrix multiplication for optimal performance
        const float *input_data = input->data();
        const float *weight_data = local_weight->data();
        float *output_data = local_output->data();

        int seq_len_int = static_cast<int>(seq_len);
        int d_model = static_cast<int>(input_size);
        int d_out = static_cast<int>(local_output_size);

        // Optional lightweight diagnostics (now minimal): only if LLAMINAR_LINEAR_DIAG is set.
        bool linear_diag = debugEnv().linear.diag;
        if (linear_diag && getRank() == 0)
        {
            LOG_INFO("[LinearDiag] rank=" << getRank() << " seq_len=" << seq_len_int
                                          << " d_model=" << d_model << " d_out_local=" << d_out);
        }

        // Use adaptive matrix multiplication
        auto start = std::chrono::high_resolution_clock::now();
        // Matrix multiplication: output = input @ weight^T
        // Weight is [local_out_dim, in_dim], so we transpose it during matmul
        bool matmul_success = adaptiveMatMul(input_data, weight_data, output_data,
                                             seq_len_int, d_out, d_model,
                                             /*is_prefill*/ false,
                                             /*distributed_partition*/ true,
                                             /*transpose_A*/ false,
                                             /*transpose_B*/ true, // CRITICAL: Transpose weight per PyTorch/GGUF convention
                                             /*alpha*/ 1.0f,
                                             /*beta*/ 0.0f);
        auto end = std::chrono::high_resolution_clock::now();

        if (!matmul_success)
        {
            LOG_ERROR("Adaptive matrix multiplication failed on rank " << getRank());
            return false;
        }

        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_DEBUG("MPILinear matmul: " << ms << "ms for " << seq_len_int << "x" << d_model << " * " << d_model << "x" << d_out);
        // (Legacy deep diagnostic & auto parity overwrite removed.)

        // Add bias if provided
        if (local_bias)
        {
            addBiasLocal(local_output->data(), local_bias->data(), seq_len, local_output_size);
        }

        // Gather results from all processes
        gatherOutput(local_output, global_output, seq_len, output_size);

        if (linear_diag && getRank() == 0)
        {
            LOG_INFO("[LinearDiag] rank=" << getRank() << " gathered_global_shape=[" << seq_len << "," << output_size << "]");
        }

        // === POST-COMPUTATION VALIDATION ===
        ASSERT_TENSOR_NOT_NAN(global_output, "Linear output after computation");

        // Log output tensor statistics
        TensorLogger::logTensorStats(global_output, "Linear final_output", "MPILinearKernel_COMPLETE");

        LOG_DEBUG("MPILinearKernel executed successfully on rank " << getRank());
        return true;
    }

    bool MPILinearKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                   const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        // Basic validation similar to LinearKernel
        if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1)
        {
            LOG_ERROR("MPILinearKernel: Expected 2-3 inputs and 1 output, got "
                      << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        auto input = inputs[0];
        auto weight = inputs[1];
        auto output = outputs[0];

        if (!input || !weight || !output)
        {
            LOG_ERROR("MPILinearKernel: Null tensor provided - input: " << (input ? "valid" : "null")
                                                                        << ", weight: " << (weight ? "valid" : "null")
                                                                        << ", output: " << (output ? "valid" : "null"));
            return false;
        }

        // Check input is 2D [seq_len, input_size]
        if (input->shape().size() != 2)
        {
            LOG_ERROR("MPILinearKernel: Input must be 2D, got " << input->shape().size() << " dimensions");
            return false;
        }

        // Check weight is 2D [output_size, input_size] per new convention
        if (weight->shape().size() != 2)
        {
            LOG_ERROR("MPILinearKernel: Weight must be 2D, got " << weight->shape().size() << " dimensions");
            return false;
        }

        // Check dimensions match: input[1] (in_dim) should match weight[1] (in_dim)
        if (input->shape()[1] != weight->shape()[1])
        {
            LOG_ERROR("MPILinearKernel: Input size " << input->shape()[1]
                                                     << " doesn't match weight input size " << weight->shape()[1]
                                                     << " (weight shape=[" << weight->shape()[0] << ", " << weight->shape()[1] << "])");
            return false;
        }

        // Check output is 2D [seq_len, output_size]
        // output[1] should match weight[0] (out_dim)
        if (output->shape().size() != 2 ||
            output->shape()[0] != input->shape()[0] ||
            output->shape()[1] != weight->shape()[0]) // Changed from weight->shape()[1]
        {
            LOG_ERROR("MPILinearKernel: Output shape mismatch - expected [" << input->shape()[0]
                                                                            << ", " << weight->shape()[0] << "], got [" << output->shape()[0] << ", " << output->shape()[1] << "]");
            return false;
        }

        // Check optional bias
        if (inputs.size() == 3 && inputs[2])
        {
            auto bias = inputs[2];
            if (bias->shape().size() != 1 || bias->shape()[0] != weight->shape()[1])
            {
                LOG_ERROR("MPILinearKernel: Bias shape mismatch");
                return false;
            }
        }

        return true;
    }

    void MPILinearKernel::distributeWeight(const std::shared_ptr<TensorBase> &global_weight,
                                           std::shared_ptr<TensorBase> &local_weight,
                                           size_t output_size)
    {
        // New convention: weight is [output_size, input_size]
        size_t input_size = global_weight->shape()[1];
        auto [local_output_size, output_offset] = getRowDistribution(output_size);

        // Extract local portion of weight matrix
        // Global weight: [output_size, input_size] - row-major storage
        // Local weight: [local_output_size, input_size] - subset of rows
        const float *global_data = global_weight->data();
        float *local_data = local_weight->data();

        // With weight as [output_size, input_size], each row is one output feature.
        // We partition output features across ranks, so each rank gets a contiguous block of rows.
        // Row offset in global: output_offset
        // Number of rows to copy: local_output_size
        // Each row has input_size elements (contiguous in memory).
        const size_t elements_to_copy = local_output_size * input_size;
        const float *src_start = global_data + output_offset * input_size;

        // Single contiguous memcpy is optimal for row-major slicing
        std::memcpy(local_data, src_start, elements_to_copy * sizeof(float));

        LOG_DEBUG("Distributed weight matrix: local size [" << local_output_size << ", " << input_size
                                                            << "], offset " << output_offset << " on rank " << getRank());
    }

    void MPILinearKernel::distributeBias(const std::shared_ptr<TensorBase> &global_bias,
                                         std::shared_ptr<TensorBase> &local_bias,
                                         size_t output_size)
    {
        auto [local_output_size, output_offset] = getRowDistribution(output_size);

        // Extract local portion of bias vector
        const float *global_data = global_bias->data();
        float *local_data = local_bias->data();

        std::memcpy(local_data, global_data + output_offset, local_output_size * sizeof(float));

        LOG_DEBUG("Distributed bias vector: local size " << local_output_size
                                                         << ", offset " << output_offset << " on rank " << getRank());
    }

    void MPILinearKernel::gatherOutput(const std::shared_ptr<TensorBase> &local_output,
                                       std::shared_ptr<TensorBase> &global_output,
                                       size_t seq_len,
                                       size_t output_size)
    {
        // Gather all local outputs to form the complete global output
        // Use MPI_Allgatherv to handle variable local sizes per rank

        auto [local_output_size, output_offset] = getRowDistribution(output_size);
        // Precompute counts/offsets once; previous implementation recomputed per row.
        std::vector<int> recv_counts(getSize());
        std::vector<int> recv_offsets(getSize());
        for (int rank = 0; rank < getSize(); ++rank)
        {
            auto [rank_local_size, rank_offset] = getRowDistribution(output_size, rank);
            recv_counts[rank] = static_cast<int>(rank_local_size);
            recv_offsets[rank] = static_cast<int>(rank_offset);
        }

        // Row-wise Allgatherv (cannot trivially collapse into single collective without packing
        // because column blocks for each rank are strided in row-major layout). Future optimization:
        // pack local_output into an interleaved buffer (seq-major) with one Allgatherv using a
        // custom MPI datatype for the column block.
        for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx)
        {
            const float *local_row = local_output->data() + seq_idx * local_output_size;
            float *global_row = global_output->data() + seq_idx * output_size;
            checkMPIError(MPI_Allgatherv(local_row, static_cast<int>(local_output_size), MPI_FLOAT,
                                         global_row, recv_counts.data(), recv_offsets.data(), MPI_FLOAT,
                                         getComm()),
                          "MPI_Allgatherv in gatherOutput");
        }

        LOG_DEBUG("Gathered output: [" << seq_len << ", " << output_size << "] on rank " << getRank());
    }

    void MPILinearKernel::addBiasLocal(float *output, const float *bias,
                                       size_t seq_len, size_t local_output_size)
    {
        // Add bias to each sequence position: output[i, j] += bias[j]
        auto omp_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (size_t i = 0; i < seq_len; ++i)
        {
#pragma omp simd
            for (size_t j = 0; j < local_output_size; ++j)
            {
                output[i * local_output_size + j] += bias[j];
            }
        }
        auto omp_end = std::chrono::high_resolution_clock::now();
        double omp_time = std::chrono::duration<double, std::milli>(omp_end - omp_start).count();
        LOG_DEBUG("OpenMP bias addition: " << omp_time << "ms, threads: " << omp_get_max_threads() << ", rank: " << getRank());

        LOG_DEBUG("Local bias addition completed: [" << seq_len << ", " << local_output_size
                                                     << "] on rank " << getRank());
    }

    std::shared_ptr<TensorBase> MPILinearKernel::createLocalTensor(const std::vector<size_t> &shape)
    {
        // Convert size_t vector to int vector for TensorFactory
        std::vector<int> int_shape(shape.begin(), shape.end());
        // Use TensorFactory to create a modern tensor
        return TensorFactory::create_simple(int_shape);
    }

} // namespace llaminar