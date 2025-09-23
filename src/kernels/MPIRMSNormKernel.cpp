#include "MPIRMSNormKernel.h"
#include "../logger.h"
#include "../debug_utils.h"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <mpi.h>

namespace llaminar
{

    MPIRMSNormKernel::MPIRMSNormKernel(DistributionStrategy strategy)
        : strategy_(strategy), epsilon_(1e-6f)
    {
        LOG_DEBUG("MPIRMSNormKernel initialized with epsilon=" << epsilon_ << " on rank " << getRank());
    }

    bool MPIRMSNormKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                   std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPIRMSNormKernel validation failed on rank " << getRank());
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto global_input = inputs[0];
        auto weight = inputs[1];
        auto global_output = outputs[0];

        // === COMPREHENSIVE TENSOR VALIDATION ===
        ASSERT_TENSOR_VALID(global_input, "RMSNorm input");
        ASSERT_TENSOR_VALID(weight, "RMSNorm weight");
        ASSERT_TENSOR_VALID(global_output, "RMSNorm output");

        // Check for NaN in inputs before computation
        ASSERT_TENSOR_NOT_NAN(global_input, "RMSNorm input");
        ASSERT_TENSOR_NOT_NAN(weight, "RMSNorm weight");

        // Log detailed tensor information
        TensorLogger::logNormalizationOperation(global_input, weight, global_output, "MPIRMSNormKernel");

        // Additional epsilon logging
        LOG_INFO("[MPIRMSNormKernel] Using epsilon=" << epsilon_);

        // Extract dimensions
        size_t global_seq_len = static_cast<size_t>(global_input->shape()[0]);
        size_t hidden_size = static_cast<size_t>(global_input->shape()[1]);

        // Calculate local distribution
        auto [local_seq_len, seq_offset] = getRowDistribution(global_seq_len);

        // Create local tensors
        auto local_input = createLocalTensor({static_cast<size_t>(local_seq_len), hidden_size});
        auto local_output = createLocalTensor({static_cast<size_t>(local_seq_len), hidden_size});

        // Distribute input data
        distributeInput(global_input, local_input, global_seq_len, hidden_size);

        // Compute distributed RMS normalization
        computeDistributedRMSNorm(local_input->data(), weight->data(),
                                  local_output->data(), local_seq_len,
                                  hidden_size, global_seq_len);

        // Gather results back to global output
        gatherOutput(local_output, global_output, global_seq_len, hidden_size);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // === POST-COMPUTATION VALIDATION ===
        ASSERT_TENSOR_NOT_NAN(global_output, "RMSNorm output after computation");

        // Log output tensor statistics
        TensorLogger::logTensorStats(global_output, "RMSNorm final_output", "MPIRMSNormKernel_COMPLETE");

        LOG_DEBUG("MPIRMSNormKernel executed successfully on rank " << getRank()
                                                                    << " in " << duration.count() << " microseconds");

        return true;
    }

    bool MPIRMSNormKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                    const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        // Basic validation
        if (inputs.size() != 2 || outputs.size() != 1)
        {
            LOG_ERROR("MPIRMSNormKernel: Expected 2 inputs and 1 output, got "
                      << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        auto input = inputs[0];
        auto weight = inputs[1];
        auto output = outputs[0];

        if (!input || !weight || !output)
        {
            LOG_ERROR("MPIRMSNormKernel: Null tensor provided");
            return false;
        }

        // Check input is 2D [seq_len, hidden_size]
        if (input->shape().size() != 2)
        {
            LOG_ERROR("MPIRMSNormKernel: Input must be 2D [seq_len, hidden_size], got "
                      << input->shape().size() << "D");
            return false;
        }

        // Check weight is 1D [hidden_size]
        if (weight->shape().size() != 1)
        {
            LOG_ERROR("MPIRMSNormKernel: Weight must be 1D [hidden_size], got "
                      << weight->shape().size() << "D");
            return false;
        }

        // Check output is 2D [seq_len, hidden_size]
        if (output->shape().size() != 2)
        {
            LOG_ERROR("MPIRMSNormKernel: Output must be 2D [seq_len, hidden_size], got "
                      << output->shape().size() << "D");
            return false;
        }

        // Check dimension compatibility
        if (input->shape()[1] != weight->shape()[0])
        {
            LOG_ERROR("MPIRMSNormKernel: Dimension mismatch between input hidden_size ("
                      << input->shape()[1] << ") and weight (" << weight->shape()[0] << ")");
            return false;
        }

        // Check input and output have same shape
        if (input->shape()[0] != output->shape()[0] || input->shape()[1] != output->shape()[1])
        {
            LOG_ERROR("MPIRMSNormKernel: Input and output shape mismatch");
            return false;
        }

        return true;
    }

    void MPIRMSNormKernel::distributeInput(const std::shared_ptr<TensorBase> &global_input,
                                           std::shared_ptr<TensorBase> &local_input,
                                           size_t global_seq_len, size_t hidden_size)
    {
        auto [local_seq_len, seq_offset] = getRowDistribution(global_seq_len);

        const float *global_data = global_input->data();
        float *local_data = local_input->data();

        // Copy the local portion of the input
        for (size_t i = 0; i < local_seq_len; ++i)
        {
            size_t global_row = seq_offset + i;
            std::memcpy(local_data + i * hidden_size,
                        global_data + global_row * hidden_size,
                        hidden_size * sizeof(float));
        }

        LOG_DEBUG("Distributed input: local size [" << local_seq_len << ", " << hidden_size
                                                    << "], offset " << seq_offset << " on rank " << getRank());
    }

    void MPIRMSNormKernel::gatherOutput(const std::shared_ptr<TensorBase> &local_output,
                                        std::shared_ptr<TensorBase> &global_output,
                                        size_t global_seq_len, size_t hidden_size)
    {
        auto [local_seq_len, seq_offset] = getRowDistribution(global_seq_len);

        // Prepare MPI gather parameters
        std::vector<int> recv_counts(getSize());
        std::vector<int> recv_offsets(getSize());

        for (int rank = 0; rank < getSize(); ++rank)
        {
            auto [rank_local_seq_len, rank_seq_offset] = getRowDistribution(global_seq_len, rank);
            recv_counts[rank] = static_cast<int>(rank_local_seq_len * hidden_size);
            recv_offsets[rank] = static_cast<int>(rank_seq_offset * hidden_size);
        }

        checkMPIError(MPI_Allgatherv(local_output->data(),
                                     static_cast<int>(local_seq_len * hidden_size), MPI_FLOAT,
                                     global_output->data(),
                                     recv_counts.data(), recv_offsets.data(), MPI_FLOAT,
                                     getComm()),
                      "MPI_Allgatherv in gatherOutput");

        LOG_DEBUG("Gathered output: [" << global_seq_len << ", " << hidden_size << "] on rank " << getRank());
    }

    void MPIRMSNormKernel::computeDistributedRMSNorm(const float *local_input, const float *weight,
                                                     float *local_output, size_t local_seq_len,
                                                     size_t hidden_size, size_t global_seq_len)
    {
        // Instrumentation: compute basic stats of local input before normalization (parallelized)
        float local_min = std::numeric_limits<float>::infinity();
        float local_max = -std::numeric_limits<float>::infinity();
        double local_sum = 0.0;
        size_t local_count = local_seq_len * hidden_size;
#pragma omp parallel for reduction(min:local_min) reduction(max:local_max) reduction(+:local_sum) schedule(static)
        for (long long i = 0; i < (long long)local_count; ++i)
        {
            float v = local_input[i];
            if (v < local_min) local_min = v;
            if (v > local_max) local_max = v;
            local_sum += v;
        }
        double local_mean = local_count ? (local_sum / local_count) : 0.0;

        // Compute local contribution to global RMS (parallel)
        float local_sum_sq = 0.0f;
#pragma omp parallel for reduction(+:local_sum_sq) collapse(2) schedule(static)
        for (long long i = 0; i < (long long)local_seq_len; ++i)
            for (long long j = 0; j < (long long)hidden_size; ++j)
            {
                float val = local_input[i * (long long)hidden_size + j];
                local_sum_sq += val * val;
            }

        // Compute global RMS via MPI reduction
        float global_sum_sq = 0.0f;
        checkMPIError(MPI_Allreduce(&local_sum_sq, &global_sum_sq, 1, MPI_FLOAT, MPI_SUM, getComm()),
                      "MPI_Allreduce for RMS computation");

        float rms = std::sqrt(global_sum_sq / (global_seq_len * hidden_size) + epsilon_);

        // Gather global stats for debugging (min, max, mean)
        float global_min = 0.0f, global_max = 0.0f;
        double global_mean_sum = 0.0;
        double global_mean = 0.0;
        checkMPIError(MPI_Allreduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN, getComm()), "MPI_Allreduce min in RMSNorm");
        checkMPIError(MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, getComm()), "MPI_Allreduce max in RMSNorm");
        checkMPIError(MPI_Allreduce(&local_sum, &global_mean_sum, 1, MPI_DOUBLE, MPI_SUM, getComm()), "MPI_Allreduce mean sum in RMSNorm");
        global_mean = (global_seq_len * hidden_size) ? (global_mean_sum / (double)(global_seq_len * hidden_size)) : 0.0;

        // Weight stats (local - identical across ranks expected). Just compute once and broadcast rank 0's view.
        float w_min = std::numeric_limits<float>::infinity();
        float w_max = -std::numeric_limits<float>::infinity();
        double w_sum = 0.0;
        for (size_t j = 0; j < hidden_size; ++j)
        {
            float w = weight[j];
            if (w < w_min)
                w_min = w;
            if (w > w_max)
                w_max = w;
            w_sum += w;
        }
        double w_sum_global = 0.0;
        float w_min_global = 0.0f, w_max_global = 0.0f;
        checkMPIError(MPI_Allreduce(&w_min, &w_min_global, 1, MPI_FLOAT, MPI_MIN, getComm()), "MPI_Allreduce weight min");
        checkMPIError(MPI_Allreduce(&w_max, &w_max_global, 1, MPI_FLOAT, MPI_MAX, getComm()), "MPI_Allreduce weight max");
        checkMPIError(MPI_Allreduce(&w_sum, &w_sum_global, 1, MPI_DOUBLE, MPI_SUM, getComm()), "MPI_Allreduce weight sum");
        double w_mean_global = (double)w_sum_global / (double)(hidden_size * getSize());

        if (getRank() == 0)
        {
            LOG_INFO("[MPIRMSNormKernel] Pre-Norm stats: min=" << global_min << " max=" << global_max << " mean=" << global_mean
                                                               << " global_sum_sq=" << global_sum_sq << " rms=" << rms);
            LOG_INFO("[MPIRMSNormKernel] Weight stats: min=" << w_min_global << " max=" << w_max_global << " mean=" << w_mean_global);
        }

        // Apply normalization to local data (parallel)
#pragma omp parallel for collapse(2) schedule(static)
        for (long long i = 0; i < (long long)local_seq_len; ++i)
            for (long long j = 0; j < (long long)hidden_size; ++j)
            {
                size_t idx = (size_t)i * hidden_size + (size_t)j;
                local_output[idx] = (local_input[idx] / rms) * weight[j];
            }

        // Post-norm local stats to detect zeroing (parallel)
        float out_local_min = std::numeric_limits<float>::infinity();
        float out_local_max = -std::numeric_limits<float>::infinity();
        double out_local_sum = 0.0;
#pragma omp parallel for reduction(min:out_local_min) reduction(max:out_local_max) reduction(+:out_local_sum) schedule(static)
        for (long long i = 0; i < (long long)local_count; ++i)
        {
            float v = local_output[i];
            if (v < out_local_min) out_local_min = v;
            if (v > out_local_max) out_local_max = v;
            out_local_sum += v;
        }
        float out_global_min = 0.0f, out_global_max = 0.0f;
        double out_global_sum = 0.0;
        checkMPIError(MPI_Allreduce(&out_local_min, &out_global_min, 1, MPI_FLOAT, MPI_MIN, getComm()), "MPI_Allreduce out min");
        checkMPIError(MPI_Allreduce(&out_local_max, &out_global_max, 1, MPI_FLOAT, MPI_MAX, getComm()), "MPI_Allreduce out max");
        checkMPIError(MPI_Allreduce(&out_local_sum, &out_global_sum, 1, MPI_DOUBLE, MPI_SUM, getComm()), "MPI_Allreduce out sum");
        double out_global_mean = (global_seq_len * hidden_size) ? (out_global_sum / (double)(global_seq_len * hidden_size)) : 0.0;
        if (getRank() == 0)
        {
            LOG_INFO("[MPIRMSNormKernel] Post-Norm stats: min=" << out_global_min << " max=" << out_global_max << " mean=" << out_global_mean);
        }

        LOG_DEBUG("Computed distributed RMS normalization: rms=" << rms
                                                                 << ", local_seq_len=" << local_seq_len << " on rank " << getRank());
    }

    std::shared_ptr<TensorBase> MPIRMSNormKernel::createLocalTensor(const std::vector<size_t> &shape)
    {
        // Convert size_t vector to int vector for TensorFactory
        std::vector<int> int_shape(shape.begin(), shape.end());
        // Use TensorFactory to create a modern tensor
        return TensorFactory::create_simple(int_shape);
    }

} // namespace llaminar