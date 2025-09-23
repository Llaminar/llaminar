#include "MPIMLPKernel.h"
#include "../adaptive_matmul.h"
#include "../logger.h"
#include <algorithm>
#include <cmath>

namespace llaminar
{

    MPIMLPKernel::MPIMLPKernel()
    {
        initializeMPI();
        LOG_DEBUG("MPIMLPKernel initialized on rank " << rank_ << "/" << size_);
    }

    bool MPIMLPKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                               std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPIMLPKernel validation failed");
            return false;
        }

        auto input = inputs[0];
        auto w_gate = inputs[1];
        auto w_up = inputs[2];
        auto w_down = inputs[3];
        auto output = outputs[0];

        size_t seq_len = input->shape()[0];
        size_t d_model = input->shape()[1];
        size_t local_d_ff = w_gate->shape()[1]; // Local d_ff on this rank

        LOG_DEBUG("MPIMLPKernel processing: seq_len=" << seq_len
                                                      << ", d_model=" << d_model << ", local_d_ff=" << local_d_ff);

        // Create temporary buffers for intermediate results
        auto gate_proj = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(local_d_ff)});
        auto up_proj = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(local_d_ff)});
        auto activated = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(local_d_ff)});
        auto local_output = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(d_model)});

        // Step 1: Compute gate projection using COSMA
        computeGateProjection(input, w_gate, gate_proj, seq_len, d_model, local_d_ff);

        // Step 2: Compute up projection using COSMA
        computeUpProjection(input, w_up, up_proj, seq_len, d_model, local_d_ff);

        // Step 3: Apply SwiGLU activation locally (no MPI communication)
        applySwiGLU(gate_proj, up_proj, activated, seq_len, local_d_ff);

        // Step 4: Compute down projection using COSMA
        computeDownProjection(activated, w_down, local_output, seq_len, local_d_ff, d_model);

        // Step 5: Gather final output using MPI_Allreduce
        gatherFinalOutput(local_output, output, seq_len, d_model);

        return true;
    }

    bool MPIMLPKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.size() != 4 || outputs.size() != 1)
        {
            LOG_ERROR("MPIMLPKernel: Expected 4 inputs and 1 output, got "
                      << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        // Basic null checks
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            if (!inputs[i])
            {
                LOG_ERROR("MPIMLPKernel: Input " << i << " is null");
                return false;
            }
        }

        if (!outputs[0])
        {
            LOG_ERROR("MPIMLPKernel: Output is null");
            return false;
        }

        auto input = inputs[0];
        auto w_gate = inputs[1];
        auto w_up = inputs[2];
        auto w_down = inputs[3];
        auto output = outputs[0];

        // Check input is 2D [seq_len, d_model]
        if (input->shape().size() != 2)
        {
            LOG_ERROR("MPIMLPKernel: Input must be 2D, got " << input->shape().size() << " dimensions");
            return false;
        }

        size_t seq_len = input->shape()[0];
        size_t d_model = input->shape()[1];

        // Check weight dimensions for distributed setting
        if (w_gate->shape().size() != 2 || w_gate->shape()[0] != d_model)
        {
            LOG_ERROR("MPIMLPKernel: Gate weight shape mismatch. Expected ["
                      << d_model << ", local_d_ff], got ["
                      << w_gate->shape()[0] << ", " << w_gate->shape()[1] << "]");
            return false;
        }

        size_t local_d_ff = w_gate->shape()[1];
        if (w_up->shape().size() != 2 || w_up->shape()[0] != d_model || w_up->shape()[1] != local_d_ff)
        {
            LOG_ERROR("MPIMLPKernel: Up weight shape mismatch. Expected ["
                      << d_model << ", " << local_d_ff << "], got ["
                      << w_up->shape()[0] << ", " << w_up->shape()[1] << "]");
            return false;
        }

        if (w_down->shape().size() != 2 || w_down->shape()[0] != local_d_ff || w_down->shape()[1] != d_model)
        {
            LOG_ERROR("MPIMLPKernel: Down weight shape mismatch. Expected ["
                      << local_d_ff << ", " << d_model << "], got ["
                      << w_down->shape()[0] << ", " << w_down->shape()[1] << "]");
            return false;
        }

        // Check output shape
        if (output->shape().size() != 2 || output->shape()[0] != seq_len || output->shape()[1] != d_model)
        {
            LOG_ERROR("MPIMLPKernel: Output shape mismatch. Expected ["
                      << seq_len << ", " << d_model << "], got ["
                      << output->shape()[0] << ", " << output->shape()[1] << "]");
            return false;
        }

        return true;
    }

    void MPIMLPKernel::computeGateProjection(const std::shared_ptr<TensorBase> &input,
                                             const std::shared_ptr<TensorBase> &w_gate,
                                             std::shared_ptr<TensorBase> &gate_output,
                                             size_t seq_len, size_t d_model, size_t local_d_ff)
    {
        // Use adaptive matrix multiplication: gate_output = input * w_gate
        // input: [seq_len, d_model], w_gate: [d_model, local_d_ff] -> gate_output: [seq_len, local_d_ff]
        if (!adaptive_matmul(input->data(), w_gate->data(), gate_output->data(),
                             seq_len, local_d_ff, d_model, false))
        {
            LOG_ERROR("MPIMLPKernel: Gate projection matrix multiplication failed");
            throw std::runtime_error("Gate projection matrix multiplication failed");
        }

        LOG_DEBUG("Rank " << rank_ << " completed gate projection: ["
                          << seq_len << ", " << d_model << "] * [" << d_model << ", " << local_d_ff
                          << "] -> [" << seq_len << ", " << local_d_ff << "]");
    }

    void MPIMLPKernel::computeUpProjection(const std::shared_ptr<TensorBase> &input,
                                           const std::shared_ptr<TensorBase> &w_up,
                                           std::shared_ptr<TensorBase> &up_output,
                                           size_t seq_len, size_t d_model, size_t local_d_ff)
    {
        // Use adaptive matrix multiplication: up_output = input * w_up
        // input: [seq_len, d_model], w_up: [d_model, local_d_ff] -> up_output: [seq_len, local_d_ff]
        if (!adaptive_matmul(input->data(), w_up->data(), up_output->data(),
                             seq_len, local_d_ff, d_model, false))
        {
            LOG_ERROR("MPIMLPKernel: Up projection matrix multiplication failed");
            throw std::runtime_error("Up projection matrix multiplication failed");
        }

        LOG_DEBUG("Rank " << rank_ << " completed up projection: ["
                          << seq_len << ", " << d_model << "] * [" << d_model << ", " << local_d_ff
                          << "] -> [" << seq_len << ", " << local_d_ff << "]");
    }

    void MPIMLPKernel::applySwiGLU(const std::shared_ptr<TensorBase> &gate_output,
                                   const std::shared_ptr<TensorBase> &up_output,
                                   std::shared_ptr<TensorBase> &activated_output,
                                   size_t seq_len, size_t local_d_ff)
    {
        // Apply SwiGLU: activated_output = silu(gate_output) * up_output
        // This is element-wise operation, no MPI communication needed
        const float *gate_data = gate_output->data();
        const float *up_data = up_output->data();
        float *output_data = activated_output->data();

        size_t total_elements = seq_len * local_d_ff;
        // OpenMP parallelization: use single thread for very small ops to avoid overhead
        if (total_elements < 8192) {
            for (size_t i = 0; i < total_elements; ++i) {
                output_data[i] = silu(gate_data[i]) * up_data[i];
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < total_elements; ++i) {
                // vectorization hint
                output_data[i] = silu(gate_data[i]) * up_data[i];
            }
        }

        LOG_DEBUG("Rank " << rank_ << " completed SwiGLU activation for "
                          << total_elements << " elements");
    }

    void MPIMLPKernel::computeDownProjection(const std::shared_ptr<TensorBase> &activated_input,
                                             const std::shared_ptr<TensorBase> &w_down,
                                             std::shared_ptr<TensorBase> &local_output,
                                             size_t seq_len, size_t local_d_ff, size_t d_model)
    {
        // Use adaptive matrix multiplication: local_output = activated_input * w_down
        // activated_input: [seq_len, local_d_ff], w_down: [local_d_ff, d_model] -> local_output: [seq_len, d_model]
        if (!adaptive_matmul(activated_input->data(), w_down->data(), local_output->data(),
                             seq_len, d_model, local_d_ff, false))
        {
            LOG_ERROR("MPIMLPKernel: Down projection matrix multiplication failed");
            throw std::runtime_error("Down projection matrix multiplication failed");
        }

        LOG_DEBUG("Rank " << rank_ << " completed down projection: ["
                          << seq_len << ", " << local_d_ff << "] * [" << local_d_ff << ", " << d_model
                          << "] -> [" << seq_len << ", " << d_model << "]");
    }

    void MPIMLPKernel::gatherFinalOutput(const std::shared_ptr<TensorBase> &local_output,
                                         std::shared_ptr<TensorBase> &global_output,
                                         size_t seq_len, size_t d_model)
    {
        // Sum contributions from all ranks using MPI_Allreduce
        checkMPIError(MPI_Allreduce(local_output->data(), global_output->data(),
                                    static_cast<int>(seq_len * d_model), MPI_FLOAT, MPI_SUM, getComm()),
                      "MPI_Allreduce in gatherFinalOutput");

        LOG_DEBUG("Rank " << rank_ << " completed final output gather for "
                          << seq_len << " x " << d_model << " elements");
    }

    float MPIMLPKernel::silu(float x) const
    {
        // SiLU (Swish) activation: x * sigmoid(x) = x / (1 + exp(-x))
        return x / (1.0f + std::exp(-x));
    }

    size_t MPIMLPKernel::calculateLocalDff(size_t global_d_ff) const
    {
        // Distribute d_ff dimension across ranks
        size_t base_size = global_d_ff / size_;
        size_t remainder = global_d_ff % size_;

        // First 'remainder' ranks get an extra element
        return (rank_ < static_cast<int>(remainder)) ? base_size + 1 : base_size;
    }

} // namespace llaminar