#include "MLPKernel.h"
#include "graph_compute.h" // For Tensor definition
#include "logger.h"
#include <cblas.h>
#include <chrono>

namespace llaminar
{

    MLPKernel::MLPKernel()
    {
        LOG_DEBUG("MLPKernel initialized");
    }

    bool MLPKernel::execute(const std::vector<std::shared_ptr<Tensor>> &inputs,
                            std::vector<std::shared_ptr<Tensor>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto input = inputs[0];       // [seq_len, hidden_size]
        auto gate_weight = inputs[1]; // [hidden_size, ff_size]
        auto up_weight = inputs[2];   // [hidden_size, ff_size]
        auto down_weight = inputs[3]; // [ff_size, hidden_size]
        auto output = outputs[0];     // [seq_len, hidden_size]

        int seq_len = input->shape[0];
        int hidden_size = input->shape[1];
        int ff_size = gate_weight->shape[1];

        // Temporary buffers
        std::vector<float> gate_proj(seq_len * ff_size);
        std::vector<float> up_proj(seq_len * ff_size);
        std::vector<float> activated(seq_len * ff_size);

        // Compute gate and up projections
        computeGateUpProjections(input->data.data(),
                                 gate_weight->data.data(), up_weight->data.data(),
                                 gate_proj.data(), up_proj.data(),
                                 seq_len, hidden_size, ff_size);

        // Apply SiLU activation: gate_proj * silu(up_proj)
        applySiLUActivation(gate_proj.data(), up_proj.data(), activated.data(),
                            seq_len, ff_size);

        // Compute down projection
        computeDownProjection(activated.data(), down_weight->data.data(),
                              output->data.data(), seq_len, ff_size, hidden_size);

        auto end = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end - start).count();

        LOG_DEBUG("MLP executed in " + std::to_string(execution_time) + " ms");
        return true;
    }

    bool MLPKernel::validate(const std::vector<std::shared_ptr<Tensor>> &inputs,
                             const std::vector<std::shared_ptr<Tensor>> &outputs) const
    {
        if (inputs.size() != 4)
        {
            LOG_ERROR("MLP requires exactly 4 inputs (input, gate_weight, up_weight, down_weight)");
            return false;
        }

        if (outputs.size() != 1)
        {
            LOG_ERROR("MLP requires exactly 1 output");
            return false;
        }

        auto input = inputs[0];
        auto gate_weight = inputs[1];
        auto up_weight = inputs[2];
        auto down_weight = inputs[3];
        auto output = outputs[0];

        // Check tensor dimensions
        if (input->shape.size() != 2 || gate_weight->shape.size() != 2 ||
            up_weight->shape.size() != 2 || down_weight->shape.size() != 2 ||
            output->shape.size() != 2)
        {
            LOG_ERROR("MLP requires all tensors to be 2D");
            return false;
        }

        int hidden_size = input->shape[1];
        int ff_size = gate_weight->shape[1];

        // Validate weight matrix dimensions
        if (gate_weight->shape[0] != hidden_size || up_weight->shape[0] != hidden_size)
        {
            LOG_ERROR("MLP gate and up weight input dimensions must match hidden_size");
            return false;
        }

        if (gate_weight->shape[1] != ff_size || up_weight->shape[1] != ff_size)
        {
            LOG_ERROR("MLP gate and up weight output dimensions must match");
            return false;
        }

        if (down_weight->shape[0] != ff_size || down_weight->shape[1] != hidden_size)
        {
            LOG_ERROR("MLP down weight dimensions mismatch");
            return false;
        }

        // Validate output dimensions
        if (input->shape[0] != output->shape[0] || input->shape[1] != output->shape[1])
        {
            LOG_ERROR("MLP output dimensions must match input");
            return false;
        }

        return true;
    }

    void MLPKernel::computeGateUpProjections(const float *input,
                                             const float *gate_weight, const float *up_weight,
                                             float *gate_proj, float *up_proj,
                                             int seq_len, int hidden_size, int ff_size)
    {
        // Gate projection: input @ gate_weight
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, ff_size, hidden_size,
                    1.0f, input, hidden_size,
                    gate_weight, ff_size,
                    0.0f, gate_proj, ff_size);

        // Up projection: input @ up_weight
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, ff_size, hidden_size,
                    1.0f, input, hidden_size,
                    up_weight, ff_size,
                    0.0f, up_proj, ff_size);
    }

    void MLPKernel::applySiLUActivation(const float *gate_proj, const float *up_proj,
                                        float *activated, int seq_len, int ff_size)
    {
        for (int i = 0; i < seq_len * ff_size; ++i)
        {
            activated[i] = gate_proj[i] * silu(up_proj[i]);
        }
    }

    void MLPKernel::computeDownProjection(const float *activated, const float *down_weight,
                                          float *output, int seq_len, int ff_size, int hidden_size)
    {
        // Down projection: activated @ down_weight
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, hidden_size, ff_size,
                    1.0f, activated, ff_size,
                    down_weight, hidden_size,
                    0.0f, output, hidden_size);
    }

} // namespace llaminar