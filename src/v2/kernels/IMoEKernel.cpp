/**
 * @file IMoEKernel.cpp
 * @brief Default (CPU) implementations for tensor-aware MoE kernel methods
 *
 * These defaults use data()/mutable_data() which work for CPU tensors.
 * GPU kernels (ROCmMoEKernel, etc.) override with gpu_data_ptr()-based
 * implementations that avoid host round-trips.
 */

#include "IMoEKernel.h"
#include "../tensors/ITensor.h"
#include "../utils/Logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace llaminar2
{

    bool IMoEKernel::routeWithTensors(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        MoERoutingResult &host_result)
    {
        // CPU default: route with host pointers, write results via mutable_data()
        if (!route(hidden->data(), gate_weights->data(),
                   seq_len, d_model, num_experts, top_k,
                   normalize_weights, host_result))
            return false;

        const size_t n = static_cast<size_t>(seq_len) * top_k;
        float *idx = output_indices->mutable_data();
        float *wt = output_weights->mutable_data();
        for (size_t i = 0; i < n; ++i)
            idx[i] = static_cast<float>(host_result.expert_indices[i]);
        std::copy(host_result.expert_weights.begin(),
                  host_result.expert_weights.end(), wt);
        return true;
    }

    bool IMoEKernel::routeWithTensorsEffectiveSeqLen(
        ITensor *hidden, ITensor *gate_weights,
        int seq_len, int d_model, int num_experts, int top_k,
        bool normalize_weights,
        ITensor *output_indices, ITensor *output_weights,
        MoERoutingResult &host_result,
        const int *device_effective_seq_len)
    {
        if (device_effective_seq_len)
            return false;
        return routeWithTensors(hidden, gate_weights,
                                seq_len, d_model, num_experts, top_k,
                                normalize_weights,
                                output_indices, output_weights,
                                host_result);
    }

    void IMoEKernel::zeroBuffer(ITensor *tensor, size_t bytes)
    {
        std::memset(tensor->mutable_data(), 0, bytes);
    }

    void IMoEKernel::gatherTokenBatchFromTensors(
        ITensor *hidden, ITensor *batch_buffer,
        const int *host_token_indices, int num_tokens, int d_model)
    {
        gatherTokenBatch(hidden->data(), batch_buffer->mutable_data(),
                         host_token_indices, num_tokens, d_model);
    }

    bool IMoEKernel::copyTokenRowFromTensor(
        ITensor *source, ITensor *row_buffer,
        int row_index, int row_width)
    {
        if (!source || !row_buffer || row_index < 0 || row_width <= 0)
            return false;
        std::copy_n(source->data() + static_cast<size_t>(row_index) * row_width,
                    row_width,
                    row_buffer->mutable_data());
        return true;
    }

    void IMoEKernel::scatterAddWeightedFromTensors(
        ITensor *output, ITensor *expert_output,
        const int *host_token_indices, const float *host_weights,
        int num_tokens, int d_model)
    {
        scatterAddWeighted(output->mutable_data(), expert_output->data(),
                           host_token_indices, host_weights,
                           num_tokens, d_model);
    }

    bool IMoEKernel::writeTokenRowToTensor(
        ITensor *destination, ITensor *row_buffer,
        int row_index, int row_width)
    {
        if (!destination || !row_buffer || row_index < 0 || row_width <= 0)
            return false;
        std::copy_n(row_buffer->data(),
                    row_width,
                    destination->mutable_data() + static_cast<size_t>(row_index) * row_width);
        return true;
    }

    void IMoEKernel::sharedExpertGateFromTensors(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        int seq_len, int d_model)
    {
        sharedExpertGate(input->data(), gate_inp->data(),
                         shared_output->mutable_data(), seq_len, d_model);
    }

    bool IMoEKernel::sharedExpertGateFromTensorsEffectiveSeqLen(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        int seq_len, int d_model,
        const int *device_effective_seq_len)
    {
        if (device_effective_seq_len)
        {
            LOG_ERROR("[IMoEKernel] CPU/default sharedExpertGateFromTensorsEffectiveSeqLen "
                      "cannot consume a device effective-length scalar");
            return false;
        }
        sharedExpertGateFromTensors(input, gate_inp, shared_output, seq_len, d_model);
        return true;
    }

    void IMoEKernel::sharedExpertGateAddFromTensors(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        ITensor *routed_residual, ITensor *combined_output,
        int seq_len, int d_model)
    {
        const float *input_data = input->data();
        const float *gate_data = gate_inp->data();
        float *shared_data = shared_output->mutable_data();
        const float *residual_data = routed_residual->data();
        float *combined_data = combined_output->mutable_data();

        for (int t = 0; t < seq_len; ++t)
        {
            const float *row = input_data + static_cast<size_t>(t) * d_model;
            float dot = 0.0f;
            for (int j = 0; j < d_model; ++j)
                dot += gate_data[j] * row[j];

            const float gate = 1.0f / (1.0f + std::exp(-dot));
            const size_t row_offset = static_cast<size_t>(t) * d_model;
            for (int j = 0; j < d_model; ++j)
            {
                const float gated_shared = gate * shared_data[row_offset + j];
                shared_data[row_offset + j] = gated_shared;
                combined_data[row_offset + j] = residual_data[row_offset + j] + gated_shared;
            }
        }
    }

    bool IMoEKernel::sharedExpertGateAddFromTensorsEffectiveSeqLen(
        ITensor *input, ITensor *gate_inp, ITensor *shared_output,
        ITensor *routed_residual, ITensor *combined_output,
        int seq_len, int d_model,
        const int *device_effective_seq_len)
    {
        if (device_effective_seq_len)
        {
            LOG_ERROR("[IMoEKernel] CPU/default sharedExpertGateAddFromTensorsEffectiveSeqLen "
                      "cannot consume a device effective-length scalar");
            return false;
        }
        sharedExpertGateAddFromTensors(input, gate_inp, shared_output,
                                       routed_residual, combined_output,
                                       seq_len, d_model);
        return true;
    }

    void IMoEKernel::swiGLUFromTensors(ITensor *gate, ITensor *up, int count)
    {
        swiGLU(gate->mutable_data(), up->data(), count);
    }

    void IMoEKernel::weightedAddFromTensors(
        ITensor *output, ITensor *input, float weight, int count)
    {
        weightedAdd(output->mutable_data(), input->data(), weight, count);
    }

    // =================================================================
    // Phase 4: GPU-side expert dispatch — CPU defaults
    // =================================================================

    bool IMoEKernel::prepareExpertGroups(
        ITensor *routing_indices, ITensor *routing_weights,
        int seq_len, int num_experts, int top_k)
    {
        // CPU default: read routing data from host, build token lists
        const float *idx = routing_indices->data();
        const float *wt = routing_weights->data();
        const int total = seq_len * top_k;

        host_expert_counts_.assign(num_experts, 0);
        host_expert_offsets_.resize(num_experts);
        host_grouped_indices_.resize(total);
        host_grouped_weights_.resize(total);

        // Count per expert
        for (int i = 0; i < total; ++i)
            host_expert_counts_[static_cast<int>(idx[i])]++;

        // Exclusive scan
        int running = 0;
        for (int e = 0; e < num_experts; ++e)
        {
            host_expert_offsets_[e] = running;
            running += host_expert_counts_[e];
        }

        // Scatter into grouped arrays
        std::vector<int> write_pos(num_experts, 0);
        for (int i = 0; i < total; ++i)
        {
            int expert_id = static_cast<int>(idx[i]);
            int dest = host_expert_offsets_[expert_id] + write_pos[expert_id]++;
            host_grouped_indices_[dest] = i / top_k; // token index
            host_grouped_weights_[dest] = wt[i];
        }

        prepared_num_experts_ = num_experts;
        return true;
    }

    int IMoEKernel::getExpertTokenCount(int expert_id) const
    {
        if (expert_id < 0 || expert_id >= prepared_num_experts_)
            return 0;
        return host_expert_counts_[expert_id];
    }

    void IMoEKernel::gatherExpertBatch(
        ITensor *hidden, ITensor *batch_buffer,
        int expert_id, int d_model)
    {
        int count = getExpertTokenCount(expert_id);
        if (count <= 0) return;
        int offset = host_expert_offsets_[expert_id];
        gatherTokenBatchFromTensors(
            hidden, batch_buffer,
            host_grouped_indices_.data() + offset, count, d_model);
    }

    void IMoEKernel::scatterExpertResults(
        ITensor *output, ITensor *expert_results,
        int expert_id, int d_model)
    {
        int count = getExpertTokenCount(expert_id);
        if (count <= 0) return;
        int offset = host_expert_offsets_[expert_id];
        scatterAddWeightedFromTensors(
            output, expert_results,
            host_grouped_indices_.data() + offset,
            host_grouped_weights_.data() + offset,
            count, d_model);
    }

} // namespace llaminar2
