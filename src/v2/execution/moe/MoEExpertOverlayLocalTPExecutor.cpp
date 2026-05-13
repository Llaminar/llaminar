/**
 * @file MoEExpertOverlayLocalTPExecutor.cpp
 * @brief Domain-scoped LocalTP TensorParallelExperts helper for MoE expert overlays.
 */

#include "MoEExpertOverlayLocalTPExecutor.h"

#include "backends/GPUDeviceContextPool.h"
#include "config/CollectiveBackendType.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/moe/MoEExpertOverlayProfiler.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "tensors/CoherenceState.h"
#include "tensors/TensorKernels.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <sstream>
#include <thread>

namespace llaminar2
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

namespace
{
    int shardStart(int elements, int degree, int index)
    {
        return static_cast<int>((static_cast<long long>(elements) * index) / degree);
    }

    int shardEnd(int elements, int degree, int index)
    {
        return static_cast<int>((static_cast<long long>(elements) * (index + 1)) / degree);
    }

    float silu(float value)
    {
        return value / (1.0f + std::exp(-value));
    }

    bool isFP32Tensor(const TensorBase *tensor)
    {
        return tensor && tensor->native_type() == TensorType::FP32;
    }

    double elapsedMs(std::chrono::steady_clock::time_point start)
    {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    bool validateFP32Tensor(const TensorBase *tensor, const char *name)
    {
        if (!tensor)
        {
            LOG_ERROR("[MoELocalTP] Null " << name << " tensor");
            return false;
        }
        if (!isFP32Tensor(tensor))
        {
            LOG_ERROR("[MoELocalTP] " << name
                      << " tensor must be FP32 for activations/routing/output, got "
                      << tensor->dtype_name());
            return false;
        }
        return true;
    }

    struct ExpertWeightHostData
    {
        const float *gate = nullptr;
        const float *up = nullptr;
        const float *down = nullptr;

        std::vector<float> gate_cache;
        std::vector<float> up_cache;
        std::vector<float> down_cache;
    };

    void logWeightBridgeFailure(
        const MoEOverlayRuntimeDomain &domain,
        const MoEExpertOverlayLocalTPRunParams &params,
        const TensorBase *tensor,
        const char *name,
        const std::string &detail)
    {
        LOG_ERROR("[MoELocalTP] layer=" << params.layer_idx
                  << " domain='" << domain.name << "' " << name
                  << " dtype=" << (tensor ? tensor->dtype_name() : "<null>")
                  << " is not available to the host-dequant correctness bridge: "
                  << detail
                  << ". This bridge requires resident, dequant-readable expert weights; "
                  << "if host weight data was released after preparation, use an optimized prepared-weight TensorParallelExperts path instead.");
    }

    bool readDequantReadableData(
        const MoEOverlayRuntimeDomain &domain,
        const MoEExpertOverlayLocalTPRunParams &params,
        const TensorBase *tensor,
        const char *name,
        const float *&data)
    {
        data = nullptr;
        if (!tensor)
        {
            logWeightBridgeFailure(domain, params, tensor, name, "tensor is null");
            return false;
        }

        try
        {
            data = tensor->data();
        }
        catch (const std::exception &e)
        {
            logWeightBridgeFailure(domain, params, tensor, name,
                                   std::string("data() failed: ") + e.what());
            return false;
        }

        if (!data)
        {
            logWeightBridgeFailure(domain, params, tensor, name,
                                   "data() returned null; host weight data may have been released after preparation");
            return false;
        }
        return true;
    }

    bool validateMatrixShape(
        const TensorBase *tensor,
        const char *name,
        int expected_rows,
        int expected_cols)
    {
        if (!validateFP32Tensor(tensor, name))
            return false;

        const auto &shape = tensor->shape();
        if (shape.size() != 2 ||
            shape[0] < static_cast<size_t>(expected_rows) ||
            shape[1] != static_cast<size_t>(expected_cols))
        {
            LOG_ERROR("[MoELocalTP] " << name << " shape mismatch: got rank " << shape.size()
                      << ", expected at least [" << expected_rows << ", " << expected_cols << "]");
            return false;
        }
        return true;
    }

    bool validateExpertTensorShape(
        const TensorBase *tensor,
        const char *name,
        int expected_cols,
        int expected_rows,
        int expected_experts)
    {
        if (!tensor)
        {
            LOG_ERROR("[MoELocalTP] Null " << name << " tensor");
            return false;
        }

        const auto &shape = tensor->shape();
        if (shape.size() != 3)
        {
            LOG_ERROR("[MoELocalTP] " << name
                      << " tensor must be 3D for TensorParallelExperts, got rank "
                      << shape.size() << " dtype=" << tensor->dtype_name());
            return false;
        }

        if (shape[0] != static_cast<size_t>(expected_cols) ||
            shape[1] != static_cast<size_t>(expected_rows) ||
            shape[2] != static_cast<size_t>(expected_experts))
        {
            LOG_ERROR("[MoELocalTP] " << name << " shape mismatch: got ["
                      << shape[0] << ", " << shape[1] << ", " << shape[2]
                      << "], expected [" << expected_cols << ", "
                      << expected_rows << ", " << expected_experts << "] dtype="
                      << tensor->dtype_name());
            return false;
        }
        return true;
    }

    bool materializeExpertWeightData(
        const MoEOverlayRuntimeDomain &domain,
        const MoEExpertOverlayLocalTPRunParams &params,
        TensorBase *tensor,
        const char *name,
        int expected_cols,
        int expected_rows,
        int expected_experts,
        std::vector<float> &cache,
        const float *&data)
    {
        data = nullptr;
        if (!tensor)
        {
            logWeightBridgeFailure(domain, params, tensor, name, "tensor is null");
            return false;
        }

        if (tensor->native_type() == TensorType::FP32)
            return readDequantReadableData(domain, params, tensor, name, data);

        if (tensor->is_raw_data_released())
        {
            logWeightBridgeFailure(domain, params, tensor, name,
                                   "raw host weight data was released after preparation");
            return false;
        }

        const size_t expert_elements = static_cast<size_t>(expected_rows) *
                                       static_cast<size_t>(expected_cols);
        cache.assign(static_cast<size_t>(expected_experts) * expert_elements, 0.0f);

        for (int expert = 0; expert < expected_experts; ++expert)
        {
            const size_t element_offset = static_cast<size_t>(expert) * expert_elements;
            std::shared_ptr<TensorBase> view;
            try
            {
                view = tensor->create_view(
                    {static_cast<size_t>(expected_rows), static_cast<size_t>(expected_cols)},
                    element_offset);
            }
            catch (const std::exception &e)
            {
                std::ostringstream detail;
                detail << "create_view([" << expected_rows << ", " << expected_cols
                       << "], offset=" << element_offset << ") failed: " << e.what();
                logWeightBridgeFailure(domain, params, tensor, name, detail.str());
                return false;
            }

            if (!view)
            {
                std::ostringstream detail;
                detail << "create_view([" << expected_rows << ", " << expected_cols
                       << "], offset=" << element_offset << ") returned null";
                logWeightBridgeFailure(domain, params, tensor, name, detail.str());
                return false;
            }

            const float *view_data = nullptr;
            if (!readDequantReadableData(domain, params, view.get(), name, view_data))
                return false;

            std::copy_n(view_data, expert_elements, cache.data() + element_offset);
        }

        data = cache.data();
        LOG_DEBUG("[MoELocalTP] layer=" << params.layer_idx
                  << " domain='" << domain.name << "' materialized " << name
                  << " dtype=" << tensor->dtype_name()
                  << " through host-dequant correctness bridge");
        return true;
    }

    bool prepareExpertWeightHostData(
        const MoEOverlayRuntimeDomain &domain,
        const MoEExpertOverlayLocalTPRunParams &params,
        ExpertWeightHostData &weights)
    {
        return materializeExpertWeightData(domain,
                                           params,
                                           params.gate_exps,
                                           "gate_exps",
                                           params.d_model,
                                           params.expert_intermediate,
                                           params.num_experts,
                                           weights.gate_cache,
                                           weights.gate) &&
               materializeExpertWeightData(domain,
                                           params,
                                           params.up_exps,
                                           "up_exps",
                                           params.d_model,
                                           params.expert_intermediate,
                                           params.num_experts,
                                           weights.up_cache,
                                           weights.up) &&
               materializeExpertWeightData(domain,
                                           params,
                                           params.down_exps,
                                           "down_exps",
                                           params.expert_intermediate,
                                           params.d_model,
                                           params.num_experts,
                                           weights.down_cache,
                                           weights.down);
    }

    bool validateRunParams(const MoEExpertOverlayLocalTPRunParams &params)
    {
        if (params.seq_len <= 0 || params.d_model <= 0 || params.num_experts <= 0 ||
            params.top_k <= 0 || params.expert_intermediate <= 0)
        {
            LOG_ERROR("[MoELocalTP] Invalid dimensions: seq_len=" << params.seq_len
                      << " d_model=" << params.d_model
                      << " num_experts=" << params.num_experts
                      << " top_k=" << params.top_k
                      << " expert_intermediate=" << params.expert_intermediate);
            return false;
        }

        if (!validateMatrixShape(params.input, "input", params.seq_len, params.d_model) ||
            !validateMatrixShape(params.routing_indices, "routing_indices", params.seq_len, params.top_k) ||
            !validateMatrixShape(params.routing_weights, "routing_weights", params.seq_len, params.top_k) ||
            !validateMatrixShape(params.output, "output", params.seq_len, params.d_model))
        {
            return false;
        }

        const bool has_prepared_participants = params.prepared_participants &&
                                               !params.prepared_participants->empty();
        const bool has_bridge_weights = params.gate_exps && params.up_exps && params.down_exps;
        if (!has_prepared_participants && has_bridge_weights &&
            (!validateExpertTensorShape(params.gate_exps,
                                        "gate_exps",
                                        params.d_model,
                                        params.expert_intermediate,
                                        params.num_experts) ||
             !validateExpertTensorShape(params.up_exps,
                                        "up_exps",
                                        params.d_model,
                                        params.expert_intermediate,
                                        params.num_experts) ||
             !validateExpertTensorShape(params.down_exps,
                                        "down_exps",
                                        params.expert_intermediate,
                                        params.d_model,
                                        params.num_experts)))
        {
            return false;
        }

        if (!params.expert_mask.empty() &&
            static_cast<int>(params.expert_mask.size()) != params.num_experts)
        {
            LOG_ERROR("[MoELocalTP] expert_mask size " << params.expert_mask.size()
                      << " does not match num_experts " << params.num_experts);
            return false;
        }
        return true;
    }

    std::vector<bool> fullExpertMask(const MoEExpertOverlayLocalTPRunParams &params)
    {
        if (!params.expert_mask.empty())
            return params.expert_mask;
        return std::vector<bool>(static_cast<size_t>(params.num_experts), true);
    }

    bool routedExpertId(float value, int token, int route_slot, int &expert_id)
    {
        if (!std::isfinite(value))
        {
            LOG_ERROR("[MoELocalTP] Non-finite routed expert id at token=" << token
                      << " topk=" << route_slot);
            return false;
        }
        const float rounded = std::round(value);
        if (std::fabs(value - rounded) > 1e-4f)
        {
            LOG_ERROR("[MoELocalTP] Non-integral routed expert id " << value
                      << " at token=" << token << " topk=" << route_slot);
            return false;
        }
        expert_id = static_cast<int>(rounded);
        return true;
    }

    bool buildExpertTokenLists(
        const MoEExpertOverlayLocalTPRunParams &params,
        std::vector<std::vector<std::pair<int, float>>> &expert_token_lists)
    {
        const auto mask = fullExpertMask(params);
        expert_token_lists.assign(static_cast<size_t>(params.num_experts), {});

        if (params.dispatch_entries)
        {
            auto compact_row_for = [&](int original_row) -> int {
                if (!params.dispatch_token_rows)
                    return original_row;
                const auto &rows = *params.dispatch_token_rows;
                auto it = std::find(rows.begin(), rows.end(), original_row);
                if (it == rows.end())
                    return -1;
                return static_cast<int>(std::distance(rows.begin(), it));
            };

            for (const auto &entry : *params.dispatch_entries)
            {
                const int expert_id = entry.expert_id;
                if (expert_id < 0 || expert_id >= params.num_experts)
                {
                    LOG_ERROR("[MoELocalTP] Dispatch descriptor has invalid expert id "
                              << expert_id << " at token=" << entry.token_row
                              << " topk=" << entry.route_slot);
                    return false;
                }
                if (!mask[static_cast<size_t>(expert_id)])
                    continue;

                const int token_row = compact_row_for(entry.token_row);
                if (token_row < 0 || token_row >= params.seq_len)
                {
                    LOG_ERROR("[MoELocalTP] Dispatch descriptor token row " << entry.token_row
                              << " cannot be mapped into active run seq_len=" << params.seq_len
                              << " for domain-local execution");
                    return false;
                }
                expert_token_lists[static_cast<size_t>(expert_id)].emplace_back(
                    token_row, entry.route_weight);
            }
            return true;
        }

        const float *routing_idx_data = params.routing_indices->data();
        const float *routing_wt_data = params.routing_weights->data();

        for (int token = 0; token < params.seq_len; ++token)
        {
            for (int topk = 0; topk < params.top_k; ++topk)
            {
                const int offset = token * params.top_k + topk;
                int expert_id = -1;
                if (!routedExpertId(routing_idx_data[offset], token, topk, expert_id))
                    return false;
                if (expert_id < 0 || expert_id >= params.num_experts)
                {
                    LOG_ERROR("[MoELocalTP] Invalid routed expert id " << expert_id
                              << " at token=" << token << " topk=" << topk);
                    return false;
                }
                if (mask[static_cast<size_t>(expert_id)])
                    expert_token_lists[static_cast<size_t>(expert_id)].emplace_back(
                        token, routing_wt_data[offset]);
            }
        }
        return true;
    }

    void computeTensorParallelExpertPartial(
        const MoEExpertOverlayLocalTPRunParams &params,
        const ExpertWeightHostData &weights,
        int expert_id,
        int intermediate_begin,
        int intermediate_end,
        const std::vector<std::pair<int, float>> &token_list,
        FP32Tensor &partial)
    {
        float *partial_data = partial.mutable_data();
        std::fill_n(partial_data, partial.numel(), 0.0f);

        const float *input = params.input->data();
        const float *gate = weights.gate;
        const float *up = weights.up;
        const float *down = weights.down;

        const size_t gate_expert_base = static_cast<size_t>(expert_id) *
                                        params.expert_intermediate * params.d_model;
        const size_t down_expert_base = static_cast<size_t>(expert_id) *
                                        params.d_model * params.expert_intermediate;

        for (const auto &[token, route_weight] : token_list)
        {
            (void)route_weight;
            const float *token_input = input + static_cast<size_t>(token) * params.d_model;
            float *token_partial = partial_data + static_cast<size_t>(token) * params.d_model;

            for (int intermediate = intermediate_begin; intermediate < intermediate_end; ++intermediate)
            {
                const float *gate_row = gate + gate_expert_base +
                                        static_cast<size_t>(intermediate) * params.d_model;
                const float *up_row = up + gate_expert_base +
                                      static_cast<size_t>(intermediate) * params.d_model;

                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (int dim = 0; dim < params.d_model; ++dim)
                {
                    gate_acc += gate_row[dim] * token_input[dim];
                    up_acc += up_row[dim] * token_input[dim];
                }

                const float activated = silu(gate_acc) * up_acc;
                for (int out_dim = 0; out_dim < params.d_model; ++out_dim)
                {
                    const size_t down_offset = down_expert_base +
                                               static_cast<size_t>(out_dim) * params.expert_intermediate +
                                               static_cast<size_t>(intermediate);
                    token_partial[out_dim] += activated * down[down_offset];
                }
            }
        }
    }

    void accumulateExpertOutput(
        const MoEExpertOverlayLocalTPRunParams &params,
        const FP32Tensor &expert_output,
        const std::vector<std::pair<int, float>> &token_list)
    {
        float *combined = params.output->mutable_data();
        const float *expert_data = expert_output.data();

        for (const auto &[token, route_weight] : token_list)
        {
            float *dst = combined + static_cast<size_t>(token) * params.d_model;
            const float *src = expert_data + static_cast<size_t>(token) * params.d_model;
            for (int dim = 0; dim < params.d_model; ++dim)
                dst[dim] += route_weight * src[dim];
        }
    }

    void setReason(std::string *reason, const std::string &message)
    {
        if (reason)
            *reason = message;
    }

    bool validateDomainAndContext(
        const MoEOverlayRuntimeDomain &domain,
        const ILocalTPContext &local_tp_context,
        std::string *reason)
    {
        if (domain.kind != ExpertDomainKind::LocalTP)
        {
            setReason(reason, "domain kind is not LocalTP");
            return false;
        }
        if (domain.compute_kind != ExpertDomainComputeKind::TensorParallelExperts)
        {
            setReason(reason, "domain compute kind is not TensorParallelExperts");
            return false;
        }
        if (domain.participants.size() < 2)
        {
            setReason(reason, "LocalTP TensorParallelExperts requires at least two participants");
            return false;
        }
        if (local_tp_context.degree() != static_cast<int>(domain.participants.size()))
        {
            std::ostringstream message;
            message << "LocalTPContext degree " << local_tp_context.degree()
                    << " does not match runtime domain participant count "
                    << domain.participants.size();
            setReason(reason, message.str());
            return false;
        }
        if (domain.backend != CollectiveBackendType::AUTO &&
            local_tp_context.backend() != domain.backend)
        {
            std::ostringstream message;
            message << "LocalTPContext backend "
                    << collectiveBackendTypeToString(local_tp_context.backend())
                    << " does not match runtime domain backend "
                    << collectiveBackendTypeToString(domain.backend);
            setReason(reason, message.str());
            return false;
        }

        for (size_t index = 0; index < domain.participants.size(); ++index)
        {
            const auto &participant = domain.participants[index];
            if (!participant.locally_addressable || !participant.local_device.is_valid())
            {
                std::ostringstream message;
                message << "participant " << index << " is not locally addressable";
                setReason(reason, message.str());
                return false;
            }

            const DeviceId context_device = local_tp_context.deviceAt(static_cast<int>(index)).toLocalDeviceId();
            if (context_device != participant.local_device)
            {
                std::ostringstream message;
                message << "participant " << index << " device mismatch: runtime="
                        << participant.local_device.to_string()
                        << " context=" << context_device.to_string();
                setReason(reason, message.str());
                return false;
            }
        }

        setReason(reason, "ok");
        return true;
    }

    std::string allreduceStageName(const MoEExpertOverlayLocalTPRunParams &params, int expert_id)
    {
        std::ostringstream stage;
        stage << params.stage_name_prefix;
        if (params.layer_idx >= 0)
            stage << "_layer" << params.layer_idx;
        stage << "_expert" << expert_id;
        return stage.str();
    }

    bool uploadPartialsToParticipantDevices(
        const MoEOverlayRuntimeDomain &domain,
        const std::vector<std::unique_ptr<FP32Tensor>> &partials)
    {
        for (size_t index = 0; index < partials.size(); ++index)
        {
            const DeviceId device = domain.participants[index].local_device;
            if (!device.is_gpu())
                continue;
            if (!partials[index]->ensureOnDevice(device))
            {
                LOG_ERROR("[MoELocalTP] Failed to upload participant " << index
                          << " partial to " << device.to_string());
                return false;
            }
        }
        return true;
    }

    bool allreduceExpertPartials(
        ILocalTPContext &local_tp_context,
        const std::vector<std::unique_ptr<FP32Tensor>> &partials,
        const std::string &stage_name)
    {
        std::atomic<bool> all_ok{true};
        std::vector<std::thread> threads;
        threads.reserve(partials.size());

        for (size_t index = 0; index < partials.size(); ++index)
        {
            threads.emplace_back([&, index]() {
                try
                {
                    const bool ok = local_tp_context.allreduce(
                        partials[index].get(), stage_name, partials[index]->numel());
                    if (!ok)
                        all_ok.store(false, std::memory_order_relaxed);
                }
                catch (const std::exception &e)
                {
                    LOG_ERROR("[MoELocalTP] allreduce failed for participant " << index
                              << " stage=" << stage_name << ": " << e.what());
                    all_ok.store(false, std::memory_order_relaxed);
                }
            });
        }

        for (auto &thread : threads)
            thread.join();

        local_tp_context.synchronize();
        return all_ok.load(std::memory_order_relaxed);
    }

    void initializeDiagnostics(
        const MoEOverlayRuntimeDomain &domain,
        const ILocalTPContext &local_tp_context,
        int expert_intermediate,
        MoEExpertOverlayLocalTPDiagnostics *diagnostics)
    {
        if (!diagnostics)
            return;

        diagnostics->clear();
        diagnostics->domain_name = domain.name;
        diagnostics->backend = local_tp_context.backend();
        diagnostics->degree = local_tp_context.degree();
        diagnostics->participants.reserve(domain.participants.size());

        for (size_t index = 0; index < domain.participants.size(); ++index)
        {
            MoEExpertOverlayLocalTPParticipantDiagnostics participant;
            participant.participant_index = static_cast<int>(index);
            participant.device = domain.participants[index].local_device;
            participant.intermediate_start = shardStart(expert_intermediate, local_tp_context.degree(), static_cast<int>(index));
            participant.intermediate_end = shardEnd(expert_intermediate, local_tp_context.degree(), static_cast<int>(index));
            diagnostics->participants.push_back(std::move(participant));
        }
    }

    void recordExecutedExpert(
        MoEExpertOverlayLocalTPDiagnostics *diagnostics,
        int expert_id,
        int routed_entry_count)
    {
        if (!diagnostics)
            return;

        diagnostics->total_routed_entries += routed_entry_count;
        for (auto &participant : diagnostics->participants)
        {
            participant.executed_expert_ids.push_back(expert_id);
            participant.routed_entry_count += routed_entry_count;
            ++participant.expert_allreduce_count;
        }
    }

    bool isAcceleratorProductionDomain(
        const MoEOverlayRuntimeDomain &domain,
        const ILocalTPContext &local_tp_context)
    {
        if (local_tp_context.backend() == CollectiveBackendType::HOST)
            return false;
        return std::all_of(domain.participants.begin(), domain.participants.end(), [](const auto &participant) {
            return participant.local_device.is_gpu();
        });
    }

    bool tensorHasShapeAtLeast(const FP32Tensor *tensor, int rows, int cols)
    {
        if (!tensor)
            return false;
        const auto &shape = tensor->shape();
        return shape.size() == 2 &&
               shape[0] >= static_cast<size_t>(rows) &&
               shape[1] == static_cast<size_t>(cols);
    }

    bool tensorHasDeviceAllocation(const FP32Tensor *tensor)
    {
        return tensor && tensor->gpu_data_ptr() != nullptr;
    }

    bool validatePreparedParticipants(
        const MoEOverlayRuntimeDomain &domain,
        const ILocalTPContext &local_tp_context,
        const MoEExpertOverlayLocalTPRunParams &params,
        const std::vector<std::vector<std::pair<int, float>>> &expert_token_lists)
    {
        if (!params.prepared_participants || params.prepared_participants->empty())
        {
            if (isAcceleratorProductionDomain(domain, local_tp_context))
            {
                LOG_ERROR("[MoELocalTP] Bridge Phase 5E production path required for layer="
                          << params.layer_idx << " domain='" << domain.name
                          << "' backend=" << collectiveBackendTypeToString(local_tp_context.backend())
                          << ": no prepared ExpertGemmRegistry participant engines were supplied. "
                          << "HOST/mock fallback may use the host-dequant bridge, but accelerator LocalTP parity cannot.");
                return false;
            }
            return true;
        }

        if (static_cast<int>(params.prepared_participants->size()) != local_tp_context.degree())
        {
            LOG_ERROR("[MoELocalTP] Prepared participant count "
                      << params.prepared_participants->size()
                      << " does not match LocalTP degree " << local_tp_context.degree()
                      << " for domain '" << domain.name << "'");
            return false;
        }

        for (int participant_index = 0; participant_index < local_tp_context.degree(); ++participant_index)
        {
            auto &participant = (*params.prepared_participants)[static_cast<size_t>(participant_index)];
            const DeviceId expected_device = domain.participants[static_cast<size_t>(participant_index)].local_device;
            if (participant.participant_index != participant_index || participant.device != expected_device)
            {
                LOG_ERROR("[MoELocalTP] Prepared participant metadata mismatch for domain '"
                          << domain.name << "' index=" << participant_index
                          << " expected_device=" << expected_device.to_string()
                          << " actual_index=" << participant.participant_index
                          << " actual_device=" << participant.device.to_string());
                return false;
            }
            if (participant.device.is_cuda() && local_tp_context.backend() != CollectiveBackendType::HOST)
            {
                LOG_ERROR("[MoELocalTP] CUDA LocalTP prepared expert execution for domain '"
                          << domain.name << "' is missing a device-resident MoE gather/scatter kernel. "
                          << "Refusing CPU-kernel fallback for production accelerator LocalTP.");
                return false;
            }
            if (participant.gate_gemm.size() < static_cast<size_t>(params.num_experts) ||
                participant.up_gemm.size() < static_cast<size_t>(params.num_experts) ||
                participant.down_gemm.size() < static_cast<size_t>(params.num_experts))
            {
                LOG_ERROR("[MoELocalTP] Prepared participant " << participant_index
                          << " in domain '" << domain.name
                          << "' has undersized GEMM vectors for " << params.num_experts
                          << " experts");
                return false;
            }
            if (!tensorHasShapeAtLeast(participant.input_scratch.get(), params.seq_len, params.d_model) ||
                !tensorHasShapeAtLeast(participant.batch_scratch.get(), params.seq_len, params.d_model) ||
                !tensorHasShapeAtLeast(participant.gate_scratch.get(), params.seq_len, params.expert_intermediate) ||
                !tensorHasShapeAtLeast(participant.up_scratch.get(), params.seq_len, params.expert_intermediate) ||
                !tensorHasShapeAtLeast(participant.partial_scratch.get(), params.seq_len, params.d_model))
            {
                LOG_ERROR("[MoELocalTP] Prepared participant " << participant_index
                          << " in domain '" << domain.name
                          << "' is missing stage-owned scratch with capacity seq_len="
                          << params.seq_len << " d_model=" << params.d_model
                          << " intermediate=" << params.expert_intermediate);
                return false;
            }
            if (participant.device.is_gpu() &&
                (!tensorHasDeviceAllocation(participant.input_scratch.get()) ||
                 !tensorHasDeviceAllocation(participant.batch_scratch.get()) ||
                 !tensorHasDeviceAllocation(participant.gate_scratch.get()) ||
                 !tensorHasDeviceAllocation(participant.up_scratch.get()) ||
                 !tensorHasDeviceAllocation(participant.partial_scratch.get())))
            {
                LOG_ERROR("[MoELocalTP] Prepared participant " << participant_index
                          << " in domain '" << domain.name
                          << "' requires preallocated GPU scratch on "
                          << participant.device.to_string()
                          << "; allocate scratch during stage construction or test setup, not inside execute()");
                return false;
            }

            for (int expert_id = 0; expert_id < params.num_experts; ++expert_id)
            {
                if (expert_token_lists[static_cast<size_t>(expert_id)].empty())
                    continue;
                if (participant.gate_gemm[static_cast<size_t>(expert_id)] == nullptr ||
                    participant.up_gemm[static_cast<size_t>(expert_id)] == nullptr ||
                    participant.down_gemm[static_cast<size_t>(expert_id)] == nullptr)
                {
                    LOG_ERROR("[MoELocalTP] Missing prepared expert engine for layer="
                              << params.layer_idx << " domain='" << domain.name
                              << "' participant=" << participant_index
                              << " device=" << participant.device.to_string()
                              << " expert=" << expert_id
                              << " gate=" << (participant.gate_gemm[static_cast<size_t>(expert_id)] != nullptr)
                              << " up=" << (participant.up_gemm[static_cast<size_t>(expert_id)] != nullptr)
                              << " down=" << (participant.down_gemm[static_cast<size_t>(expert_id)] != nullptr));
                    return false;
                }
            }
        }
        return true;
    }

    bool allreduceExpertPartials(
        ILocalTPContext &local_tp_context,
        const std::vector<FP32Tensor *> &partials,
        const std::string &stage_name,
        size_t count,
        const std::vector<void *> &participant_streams)
    {
        std::atomic<bool> all_ok{true};
        std::vector<std::thread> threads;
        threads.reserve(partials.size());

        const bool use_on_stream_allreduce =
            participant_streams.size() == partials.size() &&
            std::all_of(participant_streams.begin(), participant_streams.end(), [](void *stream) {
                return stream != nullptr;
            });

        for (size_t index = 0; index < partials.size(); ++index)
        {
            threads.emplace_back([&, index]() {
                try
                {
                    const bool ok = use_on_stream_allreduce
                                        ? local_tp_context.allreduceOnStream(
                                              partials[index], stage_name, count,
                                              participant_streams[index])
                                        : local_tp_context.allreduce(
                                              partials[index], stage_name, count);
                    if (!ok)
                        all_ok.store(false, std::memory_order_relaxed);
                }
                catch (const std::exception &e)
                {
                    LOG_ERROR("[MoELocalTP] allreduce failed for participant " << index
                              << " stage=" << stage_name << ": " << e.what());
                    all_ok.store(false, std::memory_order_relaxed);
                }
            });
        }

        for (auto &thread : threads)
            thread.join();

        if (!use_on_stream_allreduce)
            local_tp_context.synchronize();
        return all_ok.load(std::memory_order_relaxed);
    }

    void *participantComputeStream(
        DeviceId device,
        const MoEOverlayRuntimeDomain &domain,
        int layer_idx,
        int participant_index)
    {
        if (!device.is_gpu())
            return nullptr;

        try
        {
            void *stream = GPUDeviceContextPool::instance().getContext(device).defaultStream();
            if (!stream)
            {
                LOG_ERROR("[MoELocalTP] Missing explicit compute stream for layer="
                          << layer_idx << " domain='" << domain.name
                          << "' participant=" << participant_index
                          << " device=" << device.to_string());
            }
            return stream;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[MoELocalTP] Failed to resolve compute stream for layer="
                      << layer_idx << " domain='" << domain.name
                      << "' participant=" << participant_index
                      << " device=" << device.to_string() << ": " << e.what());
            return nullptr;
        }
    }

    bool resolveParticipantStreams(
        const MoEOverlayRuntimeDomain &domain,
        const MoEExpertOverlayLocalTPRunParams &params,
        const std::vector<MoEExpertOverlayLocalTPPreparedParticipant> &prepared,
        std::vector<void *> &participant_streams)
    {
        participant_streams.assign(prepared.size(), nullptr);
        bool has_gpu_participant = false;

        for (size_t index = 0; index < prepared.size(); ++index)
        {
            const auto &participant = prepared[index];
            if (!participant.device.is_gpu())
                continue;

            has_gpu_participant = true;
            participant_streams[index] = participantComputeStream(
                participant.device,
                domain,
                params.layer_idx,
                static_cast<int>(index));
            if (!participant_streams[index])
                return false;
        }

        (void)has_gpu_participant;
        return true;
    }

    void bindParticipantGemmStreams(
        MoEExpertOverlayLocalTPPreparedParticipant &participant,
        void *stream)
    {
        if (!stream)
            return;

        for (auto *kernel : participant.gate_gemm)
            if (kernel)
                kernel->setGPUStream(stream);
        for (auto *kernel : participant.up_gemm)
            if (kernel)
                kernel->setGPUStream(stream);
        for (auto *kernel : participant.down_gemm)
            if (kernel)
                kernel->setGPUStream(stream);
    }

    bool refreshParticipantInputScratch(
        const MoEOverlayRuntimeDomain &domain,
        const MoEExpertOverlayLocalTPRunParams &params,
        std::vector<MoEExpertOverlayLocalTPPreparedParticipant> &prepared,
        const std::vector<void *> &participant_streams)
    {
        const bool trace = debugEnv().moe_expert_overlay.trace;
        const size_t live_elements = static_cast<size_t>(params.seq_len) *
                                     static_cast<size_t>(params.d_model);
        if (trace)
        {
            LOG_INFO("[MoELocalTP][trace] layer=" << params.layer_idx
                     << " domain='" << domain.name
                     << "' refresh input scratch begin rows=" << params.seq_len
                     << " cols=" << params.d_model
                     << " participants=" << prepared.size());
        }
        const float *source_data = params.input->data();
        if (!source_data)
        {
            LOG_ERROR("[MoELocalTP] Failed to read host-visible input for layer="
                      << params.layer_idx << " domain='" << domain.name << "'");
            return false;
        }

        for (size_t index = 0; index < prepared.size(); ++index)
        {
            auto &participant = prepared[index];
            if (!participant.input_scratch)
            {
                LOG_ERROR("[MoELocalTP] Prepared participant " << index
                          << " in domain '" << domain.name
                          << "' is missing input scratch");
                return false;
            }

            auto *scratch_data = static_cast<float *>(participant.input_scratch->raw_mutable_data());
            if (!scratch_data)
            {
                LOG_ERROR("[MoELocalTP] Prepared participant " << index
                          << " in domain '" << domain.name
                          << "' input scratch has no host buffer");
                return false;
            }

            std::copy_n(source_data, live_elements, scratch_data);
            if (trace)
            {
                LOG_INFO("[MoELocalTP][trace] layer=" << params.layer_idx
                         << " domain='" << domain.name
                         << "' participant=" << index
                         << " upload input scratch device=" << participant.device.to_string());
            }

            if (participant.device.is_gpu() &&
                !participant.input_scratch->ensureOnDevice(
                    participant.device,
                    index < participant_streams.size() ? participant_streams[index] : nullptr))
            {
                LOG_ERROR("[MoELocalTP] Failed to upload input scratch for participant "
                          << index << " device=" << participant.device.to_string()
                          << " domain='" << domain.name << "'");
                return false;
            }
        }

        if (trace)
        {
            LOG_INFO("[MoELocalTP][trace] layer=" << params.layer_idx
                     << " domain='" << domain.name << "' refresh input scratch done");
        }
        return true;
    }

    bool fusedSwigluDownPrepared(
        FP32Tensor *gate_tensor,
        FP32Tensor *up_tensor,
        TensorBase *output,
        ITensorGemm *down_gemm,
        IMoEKernel *moe_kernel,
        int m,
        int n,
        int intermediate,
        float alpha)
    {
        if (down_gemm->multiply_tensor_with_fused_swiglu(
                gate_tensor, up_tensor, output,
                m, n, intermediate, alpha, 0.0f))
        {
            return true;
        }

        moe_kernel->swiGLUFromTensors(gate_tensor, up_tensor, m * intermediate);
        return down_gemm->multiply_tensor(
            gate_tensor, output,
            m, n, intermediate,
            true,
            alpha,
            0.0f);
    }

    bool runPreparedTensorParallelExperts(
        const MoEOverlayRuntimeDomain &domain,
        ILocalTPContext &local_tp_context,
        const MoEExpertOverlayLocalTPRunParams &params,
        const std::vector<std::vector<std::pair<int, float>>> &expert_token_lists,
        MoEExpertOverlayLocalTPDiagnostics *diagnostics)
    {
        if (!validatePreparedParticipants(domain, local_tp_context, params, expert_token_lists))
            return false;
        if (!params.prepared_participants || params.prepared_participants->empty())
            return false;
        if (!params.prepared_partial_views ||
            static_cast<int>(params.prepared_partial_views->size()) != local_tp_context.degree())
        {
            LOG_ERROR("[MoELocalTP] Prepared LocalTP path for domain '" << domain.name
                      << "' requires stage-owned participant partial views for degree "
                      << local_tp_context.degree());
            return false;
        }

        auto &prepared = *params.prepared_participants;
        auto &partials = *params.prepared_partial_views;
        const int degree = local_tp_context.degree();

        std::vector<void *> participant_streams;
        if (!resolveParticipantStreams(domain, params, prepared, participant_streams))
            return false;
        if (!participant_streams.empty() &&
            std::all_of(participant_streams.begin(), participant_streams.end(), [](void *stream) {
                return stream != nullptr;
            }))
        {
            local_tp_context.setComputeStreams(participant_streams);
        }
        for (int participant_index = 0; participant_index < degree; ++participant_index)
        {
            bindParticipantGemmStreams(
                prepared[static_cast<size_t>(participant_index)],
                participant_streams[static_cast<size_t>(participant_index)]);
        }
        const bool trace = debugEnv().moe_expert_overlay.trace;
        if (trace)
        {
            LOG_INFO("[MoELocalTP][trace] layer=" << params.layer_idx
                     << " domain='" << domain.name
                     << "' prepared LocalTP begin degree=" << degree
                     << " seq_len=" << params.seq_len
                     << " d_model=" << params.d_model);
        }
        if (!refreshParticipantInputScratch(domain, params, prepared, participant_streams))
            return false;

        const DeviceId primary_device = prepared.front().device;
        IMoEKernel *primary_kernel = KernelFactory::getOrCreateMoEKernel(primary_device);
        if (!primary_kernel)
        {
            LOG_ERROR("[MoELocalTP] Failed to resolve primary MoE kernel for "
                      << primary_device.to_string() << " domain='" << domain.name << "'");
            return false;
        }
        primary_kernel->setGPUStream(participant_streams.empty() ? nullptr : participant_streams.front());

        std::fill_n(params.output->mutable_data(), params.output->numel(), 0.0f);
        if (primary_device.is_gpu() &&
            !params.output->ensureOnDevice(
                primary_device,
                participant_streams.empty() ? nullptr : participant_streams.front()))
        {
            LOG_ERROR("[MoELocalTP] Failed to initialize output on " << primary_device.to_string()
                      << " for domain '" << domain.name << "'");
            return false;
        }

        const float replica_scale = 1.0f / static_cast<float>(degree);

        for (int expert_id = 0; expert_id < params.num_experts; ++expert_id)
        {
            const auto &token_list = expert_token_lists[static_cast<size_t>(expert_id)];
            if (token_list.empty())
                continue;

            const int num_tokens = static_cast<int>(token_list.size());
            if (trace)
            {
                LOG_INFO("[MoELocalTP][trace] layer=" << params.layer_idx
                         << " domain='" << domain.name
                         << "' expert=" << expert_id
                         << " tokens=" << num_tokens << " begin");
            }
            for (int participant_index = 0; participant_index < degree; ++participant_index)
            {
                auto &participant = prepared[static_cast<size_t>(participant_index)];
                participant.token_indices.clear();
                participant.token_weights.clear();
                for (const auto &[token, weight] : token_list)
                {
                    participant.token_indices.push_back(token);
                    participant.token_weights.push_back(weight);
                }

                IMoEKernel *moe_kernel = KernelFactory::getOrCreateMoEKernel(participant.device);
                if (!moe_kernel)
                {
                    LOG_ERROR("[MoELocalTP] Failed to resolve MoE kernel for participant "
                              << participant_index << " device=" << participant.device.to_string());
                    return false;
                }
                moe_kernel->setGPUStream(participant_streams[static_cast<size_t>(participant_index)]);

                LOG_DEBUG("[MoELocalTP] prepared expert=" << expert_id
                          << " participant=" << participant_index
                          << " device=" << participant.device.to_string()
                          << " tokens=" << num_tokens << " gather");
                if (trace)
                {
                    LOG_INFO("[MoELocalTP][trace] layer=" << params.layer_idx
                             << " expert=" << expert_id
                             << " participant=" << participant_index
                             << " gather begin");
                }
                moe_kernel->gatherTokenBatchFromTensors(
                    participant.input_scratch.get(),
                    participant.batch_scratch.get(),
                    participant.token_indices.data(),
                    num_tokens,
                    params.d_model);

                ITensorGemm *gate_gemm = participant.gate_gemm[static_cast<size_t>(expert_id)];
                ITensorGemm *up_gemm = participant.up_gemm[static_cast<size_t>(expert_id)];
                ITensorGemm *down_gemm = participant.down_gemm[static_cast<size_t>(expert_id)];

                std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                    {gate_gemm, participant.gate_scratch.get(), params.expert_intermediate, nullptr, "gate"},
                    {up_gemm, participant.up_scratch.get(), params.expert_intermediate, nullptr, "up"}};
                if (trace)
                {
                    LOG_INFO("[MoELocalTP][trace] layer=" << params.layer_idx
                             << " expert=" << expert_id
                             << " participant=" << participant_index
                             << " gate_up begin");
                }
                if (!gate_gemm->multiply_fused_tensor(
                        participant.batch_scratch.get(),
                        projections,
                        num_tokens,
                        params.d_model))
                {
                    LOG_ERROR("[MoELocalTP] Gate/up prepared GEMM failed for layer="
                              << params.layer_idx << " domain='" << domain.name
                              << "' participant=" << participant_index
                              << " expert=" << expert_id);
                    return false;
                }

                LOG_DEBUG("[MoELocalTP] prepared expert=" << expert_id
                          << " participant=" << participant_index
                          << " device=" << participant.device.to_string()
                          << " down");
                if (trace)
                {
                    LOG_INFO("[MoELocalTP][trace] layer=" << params.layer_idx
                             << " expert=" << expert_id
                             << " participant=" << participant_index
                             << " down begin");
                }
                if (!fusedSwigluDownPrepared(
                        participant.gate_scratch.get(),
                        participant.up_scratch.get(),
                        participant.partial_scratch.get(),
                        down_gemm,
                        moe_kernel,
                        num_tokens,
                        params.d_model,
                        params.expert_intermediate,
                        replica_scale))
                {
                    LOG_ERROR("[MoELocalTP] Down prepared GEMM failed for layer="
                              << params.layer_idx << " domain='" << domain.name
                              << "' participant=" << participant_index
                              << " expert=" << expert_id);
                    return false;
                }
                if (trace)
                {
                    LOG_INFO("[MoELocalTP][trace] layer=" << params.layer_idx
                             << " expert=" << expert_id
                             << " participant=" << participant_index
                             << " down queued");
                }
            }

            const std::string stage_name = allreduceStageName(params, expert_id);
            const size_t live_partial_count = static_cast<size_t>(num_tokens) * static_cast<size_t>(params.d_model);
            LOG_DEBUG("[MoELocalTP] prepared expert=" << expert_id
                      << " allreduce stage=" << stage_name
                      << " count=" << live_partial_count);
            if (trace)
            {
                LOG_INFO("[MoELocalTP][trace] layer=" << params.layer_idx
                         << " expert=" << expert_id
                         << " allreduce begin count=" << live_partial_count);
            }
            const bool reduce_profile_enabled = MoEExpertOverlayProfiler::isEnabled() && diagnostics;
            const auto reduce_start = reduce_profile_enabled ? std::chrono::steady_clock::now()
                                                             : std::chrono::steady_clock::time_point{};
            const bool reduced = allreduceExpertPartials(
                local_tp_context,
                partials,
                stage_name,
                live_partial_count,
                participant_streams);
            if (reduce_profile_enabled)
                diagnostics->domain_reduce_ms += elapsedMs(reduce_start);
            if (!reduced)
            {
                LOG_ERROR("[MoELocalTP] Domain '" << domain.name
                          << "' prepared allreduce failed for expert " << expert_id
                          << " stage=" << stage_name);
                return false;
            }

            LOG_DEBUG("[MoELocalTP] prepared expert=" << expert_id
                      << " scatter tokens=" << num_tokens);
            if (trace)
            {
                LOG_INFO("[MoELocalTP][trace] layer=" << params.layer_idx
                         << " expert=" << expert_id
                         << " scatter begin tokens=" << num_tokens);
            }
            primary_kernel->scatterAddWeightedFromTensors(
                params.output,
                prepared.front().partial_scratch.get(),
                prepared.front().token_indices.data(),
                prepared.front().token_weights.data(),
                num_tokens,
                params.d_model);
            if (primary_device.is_gpu())
                params.output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, primary_device);

            recordExecutedExpert(diagnostics, expert_id, static_cast<int>(token_list.size()));
            if (trace)
            {
                LOG_INFO("[MoELocalTP][trace] layer=" << params.layer_idx
                         << " domain='" << domain.name
                         << "' expert=" << expert_id << " done");
            }
        }

        LOG_DEBUG("[MoELocalTP] layer=" << params.layer_idx
                  << " domain='" << domain.name
                  << "' executed prepared LocalTP expert path degree=" << degree
                  << " backend=" << collectiveBackendTypeToString(local_tp_context.backend()));
        if (primary_device.is_gpu())
        {
            params.output->transitionToWithEvent(
                TensorCoherenceState::DEVICE_AUTHORITATIVE,
                primary_device,
                participant_streams.empty() ? nullptr : participant_streams.front());
        }
        return true;
    }
} // namespace

    bool MoEExpertOverlayLocalTPExecutor::canExecute(
        const MoEOverlayRuntimeDomain &domain,
        const ILocalTPContext &local_tp_context,
        std::string *reason)
    {
        return validateDomainAndContext(domain, local_tp_context, reason);
    }

    bool MoEExpertOverlayLocalTPExecutor::runTensorParallelExperts(
        const MoEOverlayRuntimeDomain &domain,
        ILocalTPContext &local_tp_context,
        const MoEExpertOverlayLocalTPRunParams &params,
        MoEExpertOverlayLocalTPDiagnostics *diagnostics)
    {
        std::string reason;
        if (!validateDomainAndContext(domain, local_tp_context, &reason))
        {
            LOG_ERROR("[MoELocalTP] Cannot execute domain '" << domain.name << "': " << reason);
            return false;
        }
        if (!validateRunParams(params))
            return false;

        initializeDiagnostics(domain, local_tp_context, params.expert_intermediate, diagnostics);
        const bool profile_enabled = MoEExpertOverlayProfiler::isEnabled() && diagnostics;
        const auto compute_start = profile_enabled ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};

        std::vector<std::vector<std::pair<int, float>>> expert_token_lists;
        if (!buildExpertTokenLists(params, expert_token_lists))
            return false;

        if (params.prepared_participants && !params.prepared_participants->empty())
        {
            const bool ok = runPreparedTensorParallelExperts(
                domain,
                local_tp_context,
                params,
                expert_token_lists,
                diagnostics);
            if (profile_enabled)
                diagnostics->compute_ms += elapsedMs(compute_start);
            return ok;
        }

        if (!validatePreparedParticipants(domain, local_tp_context, params, expert_token_lists))
            return false;

        ExpertWeightHostData weights;
        if (!prepareExpertWeightHostData(domain, params, weights))
            return false;

        float *output = params.output->mutable_data();
        std::fill_n(output, params.output->numel(), 0.0f);

        const int degree = local_tp_context.degree();
        for (int expert_id = 0; expert_id < params.num_experts; ++expert_id)
        {
            const auto &token_list = expert_token_lists[static_cast<size_t>(expert_id)];
            if (token_list.empty())
                continue;

            std::vector<std::unique_ptr<FP32Tensor>> partials;
            partials.reserve(static_cast<size_t>(degree));
            for (int participant_index = 0; participant_index < degree; ++participant_index)
            {
                auto partial = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(params.seq_len),
                                        static_cast<size_t>(params.d_model)});
                computeTensorParallelExpertPartial(params,
                                                   weights,
                                                   expert_id,
                                                   shardStart(params.expert_intermediate, degree, participant_index),
                                                   shardEnd(params.expert_intermediate, degree, participant_index),
                                                   token_list,
                                                   *partial);
                partials.push_back(std::move(partial));
            }

            if (params.upload_partials_to_participant_devices &&
                !uploadPartialsToParticipantDevices(domain, partials))
                return false;

            const std::string stage_name = allreduceStageName(params, expert_id);
            const bool reduce_profile_enabled = MoEExpertOverlayProfiler::isEnabled() && diagnostics;
            const auto reduce_start = reduce_profile_enabled ? std::chrono::steady_clock::now()
                                                             : std::chrono::steady_clock::time_point{};
            const bool reduced = allreduceExpertPartials(local_tp_context, partials, stage_name);
            if (reduce_profile_enabled)
                diagnostics->domain_reduce_ms += elapsedMs(reduce_start);
            if (!reduced)
            {
                LOG_ERROR("[MoELocalTP] Domain '" << domain.name
                          << "' allreduce failed for expert " << expert_id
                          << " stage=" << stage_name);
                return false;
            }

            accumulateExpertOutput(params, *partials.front(), token_list);
            recordExecutedExpert(diagnostics, expert_id, static_cast<int>(token_list.size()));
        }

        if (profile_enabled)
            diagnostics->compute_ms += elapsedMs(compute_start);
        return true;
    }

} // namespace llaminar2