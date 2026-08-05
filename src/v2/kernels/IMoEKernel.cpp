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
#include <stdexcept>

namespace llaminar2
{

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

    bool IMoEKernel::publishSharedExpertRankBank(
        ITensor *shared_output,
        ITensor *canonical_publication,
        int seq_len,
        int top_k,
        int d_model,
        int participant_index,
        int participant_count,
        const int *device_effective_seq_len)
    {
        if (!shared_output || !canonical_publication ||
            seq_len <= 0 || top_k <= 0 || d_model <= 0 ||
            participant_index < 0 || participant_index >= participant_count)
        {
            return false;
        }
        if (device_effective_seq_len)
        {
            LOG_ERROR("[IMoEKernel] CPU canonical rank-bank publication cannot "
                      "consume a device effective-length scalar");
            return false;
        }

        const size_t row_elements =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t route_elements =
            row_elements * static_cast<size_t>(top_k);
        const size_t required_elements =
            route_elements +
            row_elements * static_cast<size_t>(participant_count);
        if (shared_output->numel() < row_elements ||
            canonical_publication->numel() < required_elements)
        {
            return false;
        }

        float *rank_banks =
            canonical_publication->mutable_data() + route_elements;
        std::fill_n(
            rank_banks,
            row_elements * static_cast<size_t>(participant_count),
            0.0f);
        std::copy_n(
            shared_output->data(),
            row_elements,
            rank_banks +
                static_cast<size_t>(participant_index) * row_elements);
        return true;
    }

    bool IMoEKernel::finalizeCanonicalMoEPublication(
        ITensor *input,
        ITensor *gate_inp,
        ITensor *canonical_publication,
        ITensor *routed_output,
        ITensor *shared_output,
        ITensor *combined_output,
        int seq_len,
        int top_k,
        int d_model,
        int participant_count,
        const int *device_effective_seq_len)
    {
        if (!input || !gate_inp || !canonical_publication ||
            !routed_output || !shared_output || !combined_output ||
            seq_len <= 0 || top_k <= 0 || d_model <= 0 ||
            participant_count <= 0)
        {
            return false;
        }
        if (device_effective_seq_len)
        {
            LOG_ERROR("[IMoEKernel] CPU canonical publication finalizer cannot "
                      "consume a device effective-length scalar");
            return false;
        }

        const size_t row_elements =
            static_cast<size_t>(seq_len) * static_cast<size_t>(d_model);
        const size_t route_elements =
            row_elements * static_cast<size_t>(top_k);
        const size_t required_elements =
            route_elements +
            row_elements * static_cast<size_t>(participant_count);
        if (input->numel() < row_elements ||
            gate_inp->numel() < static_cast<size_t>(d_model) ||
            canonical_publication->numel() < required_elements ||
            routed_output->numel() < row_elements ||
            shared_output->numel() < row_elements ||
            combined_output->numel() < row_elements)
        {
            return false;
        }

        const float *input_data = input->data();
        const float *gate_data = gate_inp->data();
        const float *publication = canonical_publication->data();
        const float *rank_banks = publication + route_elements;
        float *routed_data = routed_output->mutable_data();
        float *shared_data = shared_output->mutable_data();
        float *combined_data = combined_output->mutable_data();

        for (int token = 0; token < seq_len; ++token)
        {
            const size_t row_offset =
                static_cast<size_t>(token) * static_cast<size_t>(d_model);
            float gate_dot = 0.0f;
            for (int column = 0; column < d_model; ++column)
            {
                gate_dot += input_data[row_offset + column] * gate_data[column];
            }
            const float gate = 1.0f / (1.0f + std::exp(-gate_dot));

            for (int column = 0; column < d_model; ++column)
            {
                float routed = 0.0f;
                for (int route = 0; route < top_k; ++route)
                {
                    const size_t route_index =
                        (static_cast<size_t>(token) *
                             static_cast<size_t>(top_k) +
                         static_cast<size_t>(route)) *
                            static_cast<size_t>(d_model) +
                        static_cast<size_t>(column);
                    routed = std::fma(1.0f, publication[route_index], routed);
                }

                float shared = 0.0f;
                for (int participant = 0;
                     participant < participant_count;
                     ++participant)
                {
                    const size_t shared_index =
                        static_cast<size_t>(participant) * row_elements +
                        row_offset + static_cast<size_t>(column);
                    shared = std::fma(1.0f, rank_banks[shared_index], shared);
                }

                const size_t output_index =
                    row_offset + static_cast<size_t>(column);
                const float gated_shared = gate * shared;
                routed_data[output_index] = routed;
                shared_data[output_index] = gated_shared;
                combined_data[output_index] = routed + gated_shared;
            }
        }
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

    bool IMoEKernel::materializePrefillLeastLoadedTransferCommands(
        const MoEKernelLaunchContext &launch,
        const DeviceMoELayerRuntime *runtime_layer,
        DeviceMoERebalancePlanEntry *plan_entries,
        uint32_t *plan_count,
        uint32_t plan_capacity,
        DeviceMoERebalanceCommandBufferHeader *command_header,
        DeviceMoERebalanceStatus *status,
        const DeviceMoERebalanceConfig &config,
        uint32_t payload_slot_capacity,
        uint32_t layer_idx,
        uint32_t command_buffer_count)
    {
        (void)launch;
        (void)runtime_layer;
        (void)plan_entries;
        (void)plan_count;
        (void)plan_capacity;
        (void)command_header;
        (void)status;
        (void)config;
        (void)payload_slot_capacity;
        (void)layer_idx;
        (void)command_buffer_count;
        LOG_ERROR("[IMoEKernel] LLEP prefill transfer command materialization "
                  "was requested on a backend that does not implement it");
        throw std::logic_error(
            "LLEP prefill transfer command materialization is not implemented by this MoE kernel");
    }

    bool IMoEKernel::planPrefillRoutesLeastLoadedCurrentBatch(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens, int max_tokens,
        int num_experts, int top_k,
        const least_loaded_ep::LeastLoadedExpertAssignmentConfig &config)
    {
        (void)launch;
        (void)runtime_layer;
        (void)current_tokens;
        (void)max_tokens;
        (void)num_experts;
        (void)top_k;
        (void)config;
        LOG_ERROR("[IMoEKernel] LLEP prefill route planning was requested "
                  "on a backend that does not implement current-batch LLEP");
        throw std::logic_error(
            "LLEP current-batch prefill route planning is not implemented by this MoE kernel");
    }

    bool IMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens, int max_tokens,
        int num_experts, int top_k)
    {
        (void)launch;
        (void)runtime_layer;
        (void)current_tokens;
        (void)max_tokens;
        (void)num_experts;
        (void)top_k;
        LOG_ERROR("[IMoEKernel] LLEP resident-only prefill route assignment "
                  "was requested on a backend that does not implement current-batch LLEP");
        throw std::logic_error(
            "LLEP resident-only current-batch prefill assignment is not implemented by this MoE kernel");
    }

    bool IMoEKernel::assignPrefillRoutesFromLeastLoadedCurrentBatchPlanAfterTransfers(
        const MoEKernelLaunchContext &launch,
        DeviceMoELayerRuntime *runtime_layer,
        int current_tokens, int max_tokens,
        int num_experts, int top_k,
        const DeviceMoERebalanceStatus *transfer_status,
        const DeviceMoERebalanceApplyStatus *apply_status)
    {
        (void)launch;
        (void)runtime_layer;
        (void)current_tokens;
        (void)max_tokens;
        (void)num_experts;
        (void)top_k;
        (void)transfer_status;
        (void)apply_status;
        LOG_ERROR("[IMoEKernel] LLEP transfer-backed prefill route assignment "
                  "was requested on a backend that does not implement current-batch LLEP");
        throw std::logic_error(
            "LLEP transfer-backed current-batch prefill assignment is not implemented by this MoE kernel");
    }

} // namespace llaminar2
