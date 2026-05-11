/**
 * @file MoEExpertOverlayCPUFallback.cpp
 * @brief Domain-scoped CPU fallback execution helper for MoE expert overlays.
 */

#include "MoEExpertOverlayCPUFallback.h"

#include "collective/GlobalTPContext.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/moe/MoEExpertTokenRowTransfer.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
namespace
{
    int findIndex(const std::vector<int> &values, int value)
    {
        auto it = std::find(values.begin(), values.end(), value);
        if (it == values.end())
            return -1;
        return static_cast<int>(std::distance(values.begin(), it));
    }

    bool isFP32Tensor(const TensorBase *tensor)
    {
        return tensor && tensor->native_type() == TensorType::FP32;
    }

    bool validateDenseTensorForMPI(const TensorBase *tensor, const char *name)
    {
        if (!tensor)
        {
            LOG_ERROR("[MoECPUFallback] Null " << name << " tensor");
            return false;
        }
        if (!isFP32Tensor(tensor))
        {
            LOG_ERROR("[MoECPUFallback] " << name << " tensor must be FP32 for Phase 6 dense transfer");
            return false;
        }
        if (tensor->numel() > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            LOG_ERROR("[MoECPUFallback] " << name << " tensor is too large for Phase 6 MPI transfer");
            return false;
        }
        return true;
    }

    bool allRanksOk(MPI_Comm comm, bool local_ok)
    {
        int ok = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, comm);
        return ok == 1;
    }

    bool sendTensorWorld(const TensorBase *tensor, int dest_rank, int tag, MPI_Comm comm)
    {
        const float *data = tensor->data();
        const int count = static_cast<int>(tensor->numel());
        return MPI_Send(data, count, MPI_FLOAT, dest_rank, tag, comm) == MPI_SUCCESS;
    }

    bool recvTensorWorld(TensorBase *tensor, int source_rank, int tag, MPI_Comm comm)
    {
        float *data = tensor->mutable_data();
        const int count = static_cast<int>(tensor->numel());
        MPI_Status status;
        return MPI_Recv(data, count, MPI_FLOAT, source_rank, tag, comm, &status) == MPI_SUCCESS;
    }

    bool rankTouchesFallbackTransfer(const MoECPUFallbackDomainContext &domain)
    {
        return domain.participates() || domain.world_rank == domain.root_world_rank;
    }

    bool validateRunParams(const MoECPUFallbackRunParams &params, bool needs_expert_weights)
    {
        if (params.seq_len <= 0 || params.d_model <= 0 || params.num_experts <= 0 ||
            params.top_k <= 0 || params.expert_intermediate <= 0)
        {
            LOG_ERROR("[MoECPUFallback] Invalid run dimensions: seq_len=" << params.seq_len
                      << " d_model=" << params.d_model
                      << " num_experts=" << params.num_experts
                      << " top_k=" << params.top_k
                      << " expert_intermediate=" << params.expert_intermediate);
            return false;
        }

        if (!validateDenseTensorForMPI(params.input, "input") ||
            !validateDenseTensorForMPI(params.routing_indices, "routing_indices") ||
            !validateDenseTensorForMPI(params.routing_weights, "routing_weights") ||
            !validateDenseTensorForMPI(params.output, "output"))
        {
            return false;
        }

        if (needs_expert_weights && (!params.gate_exps || !params.up_exps || !params.down_exps))
        {
            LOG_ERROR("[MoECPUFallback] Null expert weight tensor");
            return false;
        }

        if (!params.fallback_expert_mask.empty() &&
            static_cast<int>(params.fallback_expert_mask.size()) != params.num_experts)
        {
            LOG_ERROR("[MoECPUFallback] fallback_expert_mask size " << params.fallback_expert_mask.size()
                      << " does not match num_experts " << params.num_experts);
            return false;
        }

        return true;
    }

    std::vector<bool> fullFallbackMask(const MoECPUFallbackRunParams &params)
    {
        if (!params.fallback_expert_mask.empty())
            return params.fallback_expert_mask;
        return std::vector<bool>(static_cast<size_t>(params.num_experts), true);
    }

    struct ExpertWeightHostData
    {
        const float *gate = nullptr;
        const float *up = nullptr;
        const float *down = nullptr;

        std::vector<const float *> gate_by_expert;
        std::vector<const float *> up_by_expert;
        std::vector<const float *> down_by_expert;

        std::vector<std::vector<float>> gate_cache;
        std::vector<std::vector<float>> up_cache;
        std::vector<std::vector<float>> down_cache;
    };

    void logWeightBridgeFailure(
        const MoECPUFallbackDomainContext &domain,
        const MoECPUFallbackRunParams &params,
        const TensorBase *tensor,
        const char *name,
        const std::string &detail)
    {
        LOG_ERROR("[MoECPUFallback] layer=" << params.layer_idx
                  << " domain='" << domain.domain_name << "' " << name
                  << " dtype=" << (tensor ? tensor->dtype_name() : "<null>")
                  << " is not available to the CPU TensorParallelExperts host-dequant bridge: "
                  << detail
                  << ". This bridge requires resident, dequant-readable expert weights; "
                  << "if host weight data was released after preparation, keep host weights resident "
                  << "or route this domain through an optimized prepared-weight TensorParallelExperts path.");
    }

    bool readDequantReadableData(
        const MoECPUFallbackDomainContext &domain,
        const MoECPUFallbackRunParams &params,
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

    bool validateExpertTensorShape(const TensorBase *tensor,
                                   const char *name,
                                   int expected_cols,
                                   int expected_rows,
                                   int expected_experts)
    {
        if (!tensor)
        {
            LOG_ERROR("[MoECPUFallback] Null " << name << " tensor");
            return false;
        }

        const auto &shape = tensor->shape();
        if (shape.size() != 3)
        {
            LOG_ERROR("[MoECPUFallback] " << name
                      << " tensor must be 3D for tensor-parallel CPU fallback, got rank "
                      << shape.size() << " dtype=" << tensor->dtype_name());
            return false;
        }

        if (shape[0] != static_cast<size_t>(expected_cols) ||
            shape[1] != static_cast<size_t>(expected_rows) ||
            shape[2] != static_cast<size_t>(expected_experts))
        {
            LOG_ERROR("[MoECPUFallback] " << name << " shape mismatch: got ["
                      << shape[0] << ", " << shape[1] << ", " << shape[2]
                      << "], expected [" << expected_cols << ", "
                      << expected_rows << ", " << expected_experts << "] dtype="
                      << tensor->dtype_name());
            return false;
        }

        return true;
    }

    bool materializeExpertWeightData(
        const MoECPUFallbackDomainContext &domain,
        const MoECPUFallbackRunParams &params,
        TensorBase *tensor,
        const char *name,
        int expected_cols,
        int expected_rows,
        int expected_experts,
        const std::vector<bool> &expert_needs,
        std::vector<std::vector<float>> &cache,
        std::vector<const float *> &expert_data,
        const float *&data)
    {
        data = nullptr;
        expert_data.clear();
        if (!tensor)
        {
            logWeightBridgeFailure(domain, params, tensor, name, "tensor is null");
            return false;
        }

        if (static_cast<int>(expert_needs.size()) != expected_experts)
        {
            std::ostringstream detail;
            detail << "expert need mask size " << expert_needs.size()
                   << " does not match expected expert count " << expected_experts;
            logWeightBridgeFailure(domain, params, tensor, name, detail.str());
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
        cache.clear();
        cache.resize(static_cast<size_t>(expected_experts));
        expert_data.assign(static_cast<size_t>(expected_experts), nullptr);

        int materialized_count = 0;

        for (int expert = 0; expert < expected_experts; ++expert)
        {
            if (!expert_needs[static_cast<size_t>(expert)])
                continue;

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

            auto &expert_cache = cache[static_cast<size_t>(expert)];
            expert_cache.assign(expert_elements, 0.0f);
            std::copy_n(view_data, expert_elements, expert_cache.data());
            expert_data[static_cast<size_t>(expert)] = expert_cache.data();
            ++materialized_count;
        }

        LOG_DEBUG("[MoECPUFallback] layer=" << params.layer_idx
                  << " domain='" << domain.domain_name << "' materialized "
                  << materialized_count << "/" << expected_experts << " routed " << name
                  << " dtype=" << tensor->dtype_name()
                  << " through CPU TensorParallelExperts host-dequant bridge");
        return true;
    }

    bool prepareExpertWeightHostData(
        const MoECPUFallbackDomainContext &domain,
        const MoECPUFallbackRunParams &params,
        const std::vector<std::vector<std::pair<int, float>>> &expert_token_lists,
        ExpertWeightHostData &weights)
    {
        std::vector<bool> expert_needs(static_cast<size_t>(params.num_experts), false);
        for (int expert = 0; expert < params.num_experts; ++expert)
            expert_needs[static_cast<size_t>(expert)] = !expert_token_lists[static_cast<size_t>(expert)].empty();

        return materializeExpertWeightData(domain,
                                           params,
                                           params.gate_exps,
                                           "gate_exps",
                                           params.d_model,
                                           params.expert_intermediate,
                                           params.num_experts,
                                           expert_needs,
                                           weights.gate_cache,
                                           weights.gate_by_expert,
                                           weights.gate) &&
               materializeExpertWeightData(domain,
                                           params,
                                           params.up_exps,
                                           "up_exps",
                                           params.d_model,
                                           params.expert_intermediate,
                                           params.num_experts,
                                           expert_needs,
                                           weights.up_cache,
                                           weights.up_by_expert,
                                           weights.up) &&
               materializeExpertWeightData(domain,
                                           params,
                                           params.down_exps,
                                           "down_exps",
                                           params.expert_intermediate,
                                           params.d_model,
                                           params.num_experts,
                                           expert_needs,
                                           weights.down_cache,
                                           weights.down_by_expert,
                                           weights.down);
    }

    bool validateTensorParallelRunParams(const MoECPUFallbackRunParams &params, bool needs_expert_weights)
    {
        if (!validateRunParams(params, needs_expert_weights))
            return false;

        if (!needs_expert_weights)
            return true;

        return validateExpertTensorShape(params.gate_exps,
                                         "gate_exps",
                                         params.d_model,
                                         params.expert_intermediate,
                                         params.num_experts) &&
               validateExpertTensorShape(params.up_exps,
                                         "up_exps",
                                         params.d_model,
                                         params.expert_intermediate,
                                         params.num_experts) &&
               validateExpertTensorShape(params.down_exps,
                                         "down_exps",
                                         params.expert_intermediate,
                                         params.d_model,
                                         params.num_experts);
    }

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

    bool buildFallbackTokenLists(
        const MoECPUFallbackRunParams &params,
        std::vector<std::vector<std::pair<int, float>>> &expert_token_lists)
    {
        const auto mask = fullFallbackMask(params);
        const float *routing_idx_data = params.routing_indices->data();
        const float *routing_wt_data = params.routing_weights->data();

        expert_token_lists.assign(static_cast<size_t>(params.num_experts), {});
        for (int token = 0; token < params.seq_len; ++token)
        {
            for (int k = 0; k < params.top_k; ++k)
            {
                const int offset = token * params.top_k + k;
                const int expert_id = static_cast<int>(routing_idx_data[offset]);
                if (expert_id < 0 || expert_id >= params.num_experts)
                {
                    LOG_ERROR("[MoECPUFallback] Invalid routed expert id " << expert_id
                              << " at token=" << token << " topk=" << k);
                    return false;
                }
                if (mask[static_cast<size_t>(expert_id)])
                    expert_token_lists[static_cast<size_t>(expert_id)].emplace_back(
                        token, routing_wt_data[offset]);
            }
        }
        return true;
    }

    const float *expertWeightDataFor(
        const float *base,
        const std::vector<const float *> &by_expert,
        int expert_id,
        size_t base_offset)
    {
        if (!by_expert.empty())
            return by_expert[static_cast<size_t>(expert_id)];
        return base ? base + base_offset : nullptr;
    }

    bool computeTensorParallelExpertPartial(
        const MoECPUFallbackRunParams &params,
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

        const size_t gate_expert_base = static_cast<size_t>(expert_id) *
                                        params.expert_intermediate * params.d_model;
        const size_t down_expert_base = static_cast<size_t>(expert_id) *
                                        params.d_model * params.expert_intermediate;
        const float *gate = expertWeightDataFor(weights.gate,
                                                weights.gate_by_expert,
                                                expert_id,
                                                gate_expert_base);
        const float *up = expertWeightDataFor(weights.up,
                                              weights.up_by_expert,
                                              expert_id,
                                              gate_expert_base);
        const float *down = expertWeightDataFor(weights.down,
                                                weights.down_by_expert,
                                                expert_id,
                                                down_expert_base);

        if (!gate || !up || !down)
        {
            LOG_ERROR("[MoECPUFallback] Missing prepared host expert weights for expert "
                      << expert_id << " layer=" << params.layer_idx);
            return false;
        }

        for (const auto &[token, route_weight] : token_list)
        {
            (void)route_weight;
            const float *token_input = input + static_cast<size_t>(token) * params.d_model;
            float *token_partial = partial_data + static_cast<size_t>(token) * params.d_model;

            for (int intermediate = intermediate_begin; intermediate < intermediate_end; ++intermediate)
            {
                const float *gate_row = gate + static_cast<size_t>(intermediate) * params.d_model;
                const float *up_row = up + static_cast<size_t>(intermediate) * params.d_model;

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
                    const size_t down_offset = static_cast<size_t>(out_dim) * params.expert_intermediate +
                                               static_cast<size_t>(intermediate);
                    token_partial[out_dim] += activated * down[down_offset];
                }
            }
        }
        return true;
    }

    void accumulateExpertOutput(
        const MoECPUFallbackRunParams &params,
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

    MoEExpertTransferMode resolveFallbackTransferMode(const MoECPUFallbackRunParams &params)
    {
        if (params.transfer_mode == MoEExpertTransferMode::DenseFullSequence)
            return MoEExpertTransferMode::DenseFullSequence;
        if (params.transfer_mode == MoEExpertTransferMode::SparseTokenRows ||
            params.transfer_mode == MoEExpertTransferMode::DecodeOneToken)
            return params.transfer_mode;

        if (params.transfer_token_rows.empty())
            return MoEExpertTransferMode::DenseFullSequence;
        if (params.seq_len == 1 && params.transfer_token_rows.size() == 1)
            return MoEExpertTransferMode::DecodeOneToken;
        return MoEExpertTransferMode::SparseTokenRows;
    }

    struct FallbackTransferPlan
    {
        MoEExpertTransferMode mode = MoEExpertTransferMode::DenseFullSequence;
        std::vector<int> token_rows;
    };

    bool decodeTransferMode(int value, MoEExpertTransferMode &mode)
    {
        const auto decoded = static_cast<MoEExpertTransferMode>(value);
        switch (decoded)
        {
        case MoEExpertTransferMode::DenseFullSequence:
        case MoEExpertTransferMode::SparseTokenRows:
        case MoEExpertTransferMode::DecodeOneToken:
            mode = decoded;
            return true;
        case MoEExpertTransferMode::Auto:
            break;
        }
        return false;
    }

    bool synchronizeFallbackTransferPlan(
        const MoECPUFallbackDomainContext &domain,
        MoEExpertTransferMode local_mode,
        const std::vector<int> &local_token_rows,
        FallbackTransferPlan &plan)
    {
        bool local_ok = true;
        int header[2] = {
            static_cast<int>(MoEExpertTransferMode::DenseFullSequence),
            0,
        };

        if (domain.world_rank == domain.root_world_rank)
        {
            if (local_token_rows.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                LOG_ERROR("[MoECPUFallback] Sparse fallback token row count exceeds MPI int range");
                local_ok = false;
            }
            else
            {
                header[0] = static_cast<int>(local_mode);
                header[1] = static_cast<int>(local_token_rows.size());
            }
        }

        if (!allRanksOk(domain.world_comm, local_ok))
            return false;

        local_ok = MPI_Bcast(header, 2, MPI_INT, domain.root_world_rank, domain.world_comm) == MPI_SUCCESS;

        MoEExpertTransferMode mode = MoEExpertTransferMode::DenseFullSequence;
        if (local_ok && !decodeTransferMode(header[0], mode))
        {
            LOG_ERROR("[MoECPUFallback] Unsupported synchronized fallback transfer mode value " << header[0]);
            local_ok = false;
        }
        if (local_ok && header[1] < 0)
        {
            LOG_ERROR("[MoECPUFallback] Invalid synchronized fallback token row count " << header[1]);
            local_ok = false;
        }

        if (!allRanksOk(domain.world_comm, local_ok))
            return false;

        plan.mode = mode;
        plan.token_rows.resize(static_cast<size_t>(header[1]));
        if (domain.world_rank == domain.root_world_rank && !plan.token_rows.empty())
            std::copy(local_token_rows.begin(), local_token_rows.end(), plan.token_rows.begin());

        if (!plan.token_rows.empty())
        {
            local_ok = MPI_Bcast(plan.token_rows.data(),
                                 static_cast<int>(plan.token_rows.size()),
                                 MPI_INT,
                                 domain.root_world_rank,
                                 domain.world_comm) == MPI_SUCCESS;
        }

        return allRanksOk(domain.world_comm, local_ok);
    }

    void recordTransferStats(
        const MoECPUFallbackRunParams &params,
        MoEExpertTransferMode mode,
        const std::vector<int> &token_rows)
    {
        if (!params.transfer_stats)
            return;

        params.transfer_stats->mode = mode;
        params.transfer_stats->token_rows = token_rows;
        params.transfer_stats->volume = MoEExpertTokenRowTransfer::estimateVolume(
            params.seq_len,
            params.top_k,
            params.d_model,
            token_rows.size(),
            mode);
    }

    bool validateSparseTransferRequest(
        const MoECPUFallbackRunParams &params,
        MoEExpertTransferMode mode)
    {
        if (mode == MoEExpertTransferMode::DenseFullSequence)
            return true;
        if (mode != MoEExpertTransferMode::SparseTokenRows &&
            mode != MoEExpertTransferMode::DecodeOneToken)
        {
            LOG_ERROR("[MoECPUFallback] Unsupported fallback transfer mode " << toString(mode));
            return false;
        }

        const auto row_errors = MoEExpertTokenRowTransfer::validateTokenRows(
            params.transfer_token_rows,
            params.seq_len);
        for (const auto &error : row_errors)
            LOG_ERROR("[MoECPUFallback] " << error);
        if (!row_errors.empty())
            return false;

        if (mode == MoEExpertTransferMode::DecodeOneToken &&
            !(params.seq_len == 1 && params.transfer_token_rows.size() == 1 && params.transfer_token_rows[0] == 0))
        {
            LOG_ERROR("[MoECPUFallback] DecodeOneToken fallback transfer requires token_rows={0} and seq_len=1");
            return false;
        }

        return true;
    }

    MoECPUFallbackRunParams compactRunParams(
        const MoECPUFallbackRunParams &params,
        MoEExpertSparseTransferBuffers &buffers,
        size_t row_count)
    {
        MoECPUFallbackRunParams compact = params;
        compact.input = buffers.hidden.get();
        compact.routing_indices = buffers.routing_indices.get();
        compact.routing_weights = buffers.routing_weights.get();
        compact.output = buffers.output.get();
        compact.seq_len = static_cast<int>(row_count);
        compact.transfer_mode = MoEExpertTransferMode::DenseFullSequence;
        compact.transfer_token_rows.clear();
        compact.transfer_stats = nullptr;
        return compact;
    }

    bool prepareSparseTransferBuffers(
        const MoECPUFallbackDomainContext &domain,
        const MoECPUFallbackRunParams &params,
        MoEExpertTransferMode mode,
        MoEExpertSparseTransferBuffers &buffers)
    {
        const size_t row_count = params.transfer_token_rows.size();
        if (row_count == 0)
            return true;

        MoEExpertTransferVolume volume;
        bool local_ok = true;
        if (domain.world_rank == domain.root_world_rank)
        {
            local_ok = MoEExpertTokenRowTransfer::gatherRows(
                params.input,
                params.routing_indices,
                params.routing_weights,
                params.transfer_token_rows,
                params.seq_len,
                params.top_k,
                params.d_model,
                &buffers,
                &volume,
                mode);
        }
        else
        {
            buffers = MoEExpertTokenRowTransfer::allocateBuffers(row_count, params.top_k, params.d_model);
            volume = MoEExpertTokenRowTransfer::estimateVolume(
                params.seq_len,
                params.top_k,
                params.d_model,
                row_count,
                mode);
        }

        if (!allRanksOk(domain.world_comm, local_ok))
            return false;
        if (params.transfer_stats)
        {
            params.transfer_stats->mode = mode;
            params.transfer_stats->token_rows = params.transfer_token_rows;
            params.transfer_stats->volume = volume;
        }
        return true;
    }

    bool scatterSparseOutputToRoot(
        const MoECPUFallbackDomainContext &domain,
        const MoECPUFallbackRunParams &params,
        const TensorBase *compact_output)
    {
        bool local_ok = true;
        if (domain.world_rank == domain.root_world_rank)
        {
            local_ok = MoEExpertTokenRowTransfer::scatterAddRows(
                compact_output,
                params.transfer_token_rows,
                params.output,
                params.seq_len,
                params.d_model);
        }
        return allRanksOk(domain.world_comm, local_ok);
    }

    bool returnFallbackOutputToRoot(
        const MoECPUFallbackDomainContext &domain,
        TensorBase *output,
        int tag = 43000)
    {
        bool local_ok = true;
        if (rankTouchesFallbackTransfer(domain))
            local_ok = validateDenseTensorForMPI(output, "tensor-parallel fallback output");
        if (!allRanksOk(domain.world_comm, local_ok))
            return false;

        MPI_Barrier(domain.world_comm);

        if (domain.root_domain_index < 0)
        {
            const int first_participant = domain.world_ranks.front();
            if (domain.world_rank == first_participant)
                local_ok = sendTensorWorld(output, domain.root_world_rank, tag, domain.world_comm) && local_ok;
            if (domain.world_rank == domain.root_world_rank)
                local_ok = recvTensorWorld(output, first_participant, tag, domain.world_comm) && local_ok;
        }

        MPI_Barrier(domain.world_comm);
        return allRanksOk(domain.world_comm, local_ok);
    }

    bool runSparseReplicatedExpertFallback(
        const MoECPUFallbackDomainContext &domain,
        const MoECPUFallbackRunParams &params,
        IDeviceContext *device_context,
        MoEExpertTransferMode mode)
    {
        recordTransferStats(params, mode, params.transfer_token_rows);
        bool local_ok = validateSparseTransferRequest(params, mode);
        if (!allRanksOk(domain.world_comm, local_ok))
            return false;
        if (params.transfer_token_rows.empty())
            return true;

        MoEExpertSparseTransferBuffers buffers;
        if (!prepareSparseTransferBuffers(domain, params, mode, buffers))
            return false;

        auto compact = compactRunParams(params, buffers, params.transfer_token_rows.size());
        if (!MoEExpertOverlayCPUFallback::transferInputsToFallbackDomain(
                domain,
                {compact.input, compact.routing_indices, compact.routing_weights},
                41100))
        {
            return false;
        }

        if (domain.participates())
        {
            auto local_mask = MoEExpertOverlayCPUFallback::participantExpertMask(
                compact.fallback_expert_mask,
                compact.num_experts,
                domain.tp_context->degree(),
                domain.tp_context->myIndex());

            MoEExpertComputeStage::Params expert_params;
            expert_params.device_id = DeviceId::cpu();
            expert_params.input = compact.input;
            expert_params.seq_len = compact.seq_len;
            expert_params.d_model = compact.d_model;
            expert_params.num_experts = compact.num_experts;
            expert_params.top_k = compact.top_k;
            expert_params.gate_exps = compact.gate_exps;
            expert_params.up_exps = compact.up_exps;
            expert_params.down_exps = compact.down_exps;
            expert_params.expert_intermediate = compact.expert_intermediate;
            expert_params.layer_idx = compact.layer_idx;
            expert_params.expert_mask = std::move(local_mask);
            expert_params.routing_indices = compact.routing_indices;
            expert_params.routing_weights = compact.routing_weights;
            expert_params.output = compact.output;
            expert_params.prepared_store = compact.prepared_store;

            if (!MoEExpertComputeStage::extractExpertViews(expert_params) ||
                !MoEExpertComputeStage::prepareExpertGemmEngines(expert_params))
            {
                local_ok = false;
            }
            else
            {
                MoEExpertComputeStage stage(std::move(expert_params));
                local_ok = stage.execute(device_context);
            }
        }

        if (!allRanksOk(domain.world_comm, local_ok))
            return false;
        if (!MoEExpertOverlayCPUFallback::reduceFallbackPartialToRoot(domain, compact.output, 42100))
            return false;
        return scatterSparseOutputToRoot(domain, params, compact.output);
    }

    bool runSparseTensorParallelExpertFallback(
        const MoECPUFallbackDomainContext &domain,
        const MoECPUFallbackRunParams &params,
        MoEExpertTransferMode mode)
    {
        recordTransferStats(params, mode, params.transfer_token_rows);
        bool local_ok = validateSparseTransferRequest(params, mode);
        if (domain.participates())
        {
            const bool has_multi_rank_domain = domain.tp_context->degree() > 1;
            if (!has_multi_rank_domain)
                LOG_ERROR("[MoECPUFallback] TensorParallelExperts requires a multi-rank fallback domain");
            local_ok = local_ok && has_multi_rank_domain;
        }
        if (!allRanksOk(domain.world_comm, local_ok))
            return false;
        if (params.transfer_token_rows.empty())
            return true;

        MoEExpertSparseTransferBuffers buffers;
        if (!prepareSparseTransferBuffers(domain, params, mode, buffers))
            return false;

        auto compact = compactRunParams(params, buffers, params.transfer_token_rows.size());
        if (!MoEExpertOverlayCPUFallback::transferInputsToFallbackDomain(
                domain,
                {compact.input, compact.routing_indices, compact.routing_weights},
                41150))
        {
            return false;
        }

        std::vector<std::vector<std::pair<int, float>>> expert_token_lists;
        ExpertWeightHostData weights;
        int intermediate_begin = 0;
        int intermediate_end = 0;

        if (domain.participates())
        {
            const int domain_index = domain.tp_context->myIndex();
            const int domain_degree = domain.tp_context->degree();
            intermediate_begin = shardStart(compact.expert_intermediate, domain_degree, domain_index);
            intermediate_end = shardEnd(compact.expert_intermediate, domain_degree, domain_index);

            if (compact.tensor_parallel_stats)
            {
                compact.tensor_parallel_stats->domain_index = domain_index;
                compact.tensor_parallel_stats->domain_degree = domain_degree;
                compact.tensor_parallel_stats->intermediate_start = intermediate_begin;
                compact.tensor_parallel_stats->intermediate_end = intermediate_end;
                compact.tensor_parallel_stats->expert_allreduce_count = 0;
                compact.tensor_parallel_stats->processed_expert_ids.clear();
            }

            local_ok = buildFallbackTokenLists(compact, expert_token_lists) &&
                       prepareExpertWeightHostData(domain, compact, expert_token_lists, weights);
        }

        if (!allRanksOk(domain.world_comm, local_ok))
            return false;

        if (domain.participates())
        {
            float *output = compact.output->mutable_data();
            std::fill_n(output, compact.output->numel(), 0.0f);

            FP32Tensor expert_partial({static_cast<size_t>(compact.seq_len),
                                       static_cast<size_t>(compact.d_model)});

            for (int expert_id = 0; expert_id < compact.num_experts; ++expert_id)
            {
                const auto &token_list = expert_token_lists[static_cast<size_t>(expert_id)];
                if (token_list.empty())
                    continue;

                if (!computeTensorParallelExpertPartial(compact,
                                                        weights,
                                                        expert_id,
                                                        intermediate_begin,
                                                        intermediate_end,
                                                        token_list,
                                                        expert_partial))
                {
                    local_ok = false;
                    break;
                }

                domain.tp_context->barrier();
                if (!domain.tp_context->allreduce(&expert_partial,
                                                  "moe_cpu_fallback_sparse_tensor_parallel_expert",
                                                  expert_partial.numel()))
                {
                    local_ok = false;
                }
                domain.tp_context->barrier();

                if (!local_ok)
                    break;

                accumulateExpertOutput(compact, expert_partial, token_list);
                if (compact.tensor_parallel_stats)
                {
                    compact.tensor_parallel_stats->processed_expert_ids.push_back(expert_id);
                    ++compact.tensor_parallel_stats->expert_allreduce_count;
                }
            }
        }

        if (!allRanksOk(domain.world_comm, local_ok))
            return false;
        if (!returnFallbackOutputToRoot(domain, compact.output, 43150))
            return false;
        return scatterSparseOutputToRoot(domain, params, compact.output);
    }

} // namespace

    int MoEExpertOverlayCPUFallback::stableDomainId(const std::string &domain_name)
    {
        uint32_t hash = 2166136261u;
        for (unsigned char ch : domain_name)
        {
            hash ^= ch;
            hash *= 16777619u;
        }
        return static_cast<int>(hash & 0x7fffffff);
    }

    std::vector<int> MoEExpertOverlayCPUFallback::domainWorldRanks(const ExpertComputeDomain &domain)
    {
        if (!domain.world_ranks.empty())
            return domain.world_ranks;

        std::vector<int> ranks(domain.participants.size());
        std::iota(ranks.begin(), ranks.end(), 0);
        return ranks;
    }

    MoECPUFallbackDomainContext MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(
        const MoECPUFallbackDomainConfig &config)
    {
        MoECPUFallbackDomainContext result;
        result.world_comm = config.world_comm;
        result.root_world_rank = config.root_world_rank;
        result.domain_name = config.domain.name;
        result.compute_kind = config.domain.compute_kind;
        result.domain_id = config.domain_id >= 0 ? config.domain_id : stableDomainId(config.domain.name);
        result.world_ranks = domainWorldRanks(config.domain);

        MPI_Comm_rank(config.world_comm, &result.world_rank);
        MPI_Comm_size(config.world_comm, &result.world_size);

        if (config.domain.kind != ExpertDomainKind::NodeLocalTP)
            throw std::invalid_argument("MoE CPU fallback domain must be NodeLocalTP");
        if (result.world_ranks.empty())
            throw std::invalid_argument("MoE CPU fallback domain must have at least one participant rank");

        for (int rank : result.world_ranks)
        {
            if (rank < 0 || rank >= result.world_size)
                throw std::invalid_argument("MoE CPU fallback domain rank is outside MPI_COMM_WORLD size");
        }

        result.root_domain_index = findIndex(result.world_ranks, config.root_world_rank);
        const int my_domain_index = findIndex(result.world_ranks, result.world_rank);
        const int color = my_domain_index >= 0 ? 0 : MPI_UNDEFINED;
        const int key = my_domain_index >= 0 ? my_domain_index : result.world_rank;

        auto created = GlobalTPContext::createWithSplit(
            config.world_comm,
            result.domain_id,
            color,
            key,
            config.hostfile_path);

        if (my_domain_index >= 0)
        {
            if (!created || !created->isValid())
                throw std::runtime_error("Failed to create MoE CPU fallback NodeLocalTP context");
            if (created->scope() != TPScope::NODE_LOCAL)
                throw std::runtime_error("MoE CPU fallback ranks are not node-local");
            result.tp_context = std::shared_ptr<IGlobalTPContext>(std::move(created));
        }

        MPI_Barrier(config.world_comm);
        return result;
    }

    std::vector<bool> MoEExpertOverlayCPUFallback::participantExpertMask(
        const std::vector<bool> &fallback_expert_mask,
        int num_experts,
        int domain_degree,
        int domain_index)
    {
        std::vector<bool> full_mask = fallback_expert_mask;
        if (full_mask.empty())
            full_mask.assign(static_cast<size_t>(num_experts), true);

        std::vector<bool> local_mask(static_cast<size_t>(num_experts), false);
        if (domain_degree <= 0 || domain_index < 0 || domain_index >= domain_degree)
            return local_mask;

        int active_ordinal = 0;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            if (!full_mask[static_cast<size_t>(expert)])
                continue;
            if ((active_ordinal % domain_degree) == domain_index)
                local_mask[static_cast<size_t>(expert)] = true;
            ++active_ordinal;
        }
        return local_mask;
    }

    bool MoEExpertOverlayCPUFallback::transferInputsToFallbackDomain(
        const MoECPUFallbackDomainContext &domain,
        const std::vector<TensorBase *> &tensors,
        int tag_base)
    {
        bool local_ok = !domain.world_ranks.empty();
        if (rankTouchesFallbackTransfer(domain))
        {
            for (TensorBase *tensor : tensors)
                local_ok = validateDenseTensorForMPI(tensor, "transfer input") && local_ok;
        }

        if (!allRanksOk(domain.world_comm, local_ok))
            return false;

        MPI_Barrier(domain.world_comm);

        const int first_participant = domain.world_ranks.front();
        if (domain.root_domain_index >= 0)
        {
            if (domain.participates())
            {
                for (TensorBase *tensor : tensors)
                {
                    if (!domain.tp_context->broadcast(tensor, domain.root_domain_index))
                        local_ok = false;
                }
            }
        }
        else
        {
            for (size_t i = 0; i < tensors.size(); ++i)
            {
                const int tag = tag_base + static_cast<int>(i);
                if (domain.world_rank == domain.root_world_rank)
                    local_ok = sendTensorWorld(tensors[i], first_participant, tag, domain.world_comm) && local_ok;
                if (domain.world_rank == first_participant)
                    local_ok = recvTensorWorld(tensors[i], domain.root_world_rank, tag, domain.world_comm) && local_ok;
            }

            if (domain.participates())
            {
                for (TensorBase *tensor : tensors)
                {
                    if (!domain.tp_context->broadcast(tensor, 0))
                        local_ok = false;
                }
            }
        }

        MPI_Barrier(domain.world_comm);
        return allRanksOk(domain.world_comm, local_ok);
    }

    bool MoEExpertOverlayCPUFallback::reduceFallbackPartialToRoot(
        const MoECPUFallbackDomainContext &domain,
        TensorBase *partial,
        int tag)
    {
        bool local_ok = true;
        if (rankTouchesFallbackTransfer(domain))
            local_ok = validateDenseTensorForMPI(partial, "fallback partial");
        if (!allRanksOk(domain.world_comm, local_ok))
            return false;

        MPI_Barrier(domain.world_comm);

        if (domain.participates())
        {
            domain.tp_context->barrier();
            if (!domain.tp_context->allreduce(partial, "moe_cpu_fallback", partial->numel()))
                local_ok = false;
            domain.tp_context->barrier();
        }

        if (domain.root_domain_index < 0)
        {
            const int first_participant = domain.world_ranks.front();
            if (domain.world_rank == first_participant)
                local_ok = sendTensorWorld(partial, domain.root_world_rank, tag, domain.world_comm) && local_ok;
            if (domain.world_rank == domain.root_world_rank)
                local_ok = recvTensorWorld(partial, first_participant, tag, domain.world_comm) && local_ok;
        }

        MPI_Barrier(domain.world_comm);
        return allRanksOk(domain.world_comm, local_ok);
    }

    bool MoEExpertOverlayCPUFallback::runReplicatedExpertFallback(
        const MoECPUFallbackDomainContext &domain,
        const MoECPUFallbackRunParams &params,
        IDeviceContext *device_context)
    {
        const bool relevant_rank = rankTouchesFallbackTransfer(domain);
        bool local_ok = !relevant_rank || validateRunParams(params, domain.participates());
        if (!allRanksOk(domain.world_comm, local_ok))
            return false;

        FallbackTransferPlan transfer_plan;
        if (!synchronizeFallbackTransferPlan(domain,
                                             resolveFallbackTransferMode(params),
                                             params.transfer_token_rows,
                                             transfer_plan))
        {
            return false;
        }

        MoECPUFallbackRunParams synced_params = params;
        synced_params.transfer_mode = transfer_plan.mode;
        synced_params.transfer_token_rows = std::move(transfer_plan.token_rows);

        if (synced_params.transfer_mode != MoEExpertTransferMode::DenseFullSequence)
            return runSparseReplicatedExpertFallback(domain, synced_params, device_context, synced_params.transfer_mode);
        recordTransferStats(synced_params, synced_params.transfer_mode, synced_params.transfer_token_rows);

        if (!transferInputsToFallbackDomain(domain,
                                            {synced_params.input, synced_params.routing_indices, synced_params.routing_weights}))
        {
            return false;
        }

        if (domain.participates())
        {
            auto local_mask = participantExpertMask(synced_params.fallback_expert_mask,
                                                    synced_params.num_experts,
                                                    domain.tp_context->degree(),
                                                    domain.tp_context->myIndex());

            MoEExpertComputeStage::Params expert_params;
            expert_params.device_id = DeviceId::cpu();
            expert_params.input = synced_params.input;
            expert_params.seq_len = synced_params.seq_len;
            expert_params.d_model = synced_params.d_model;
            expert_params.num_experts = synced_params.num_experts;
            expert_params.top_k = synced_params.top_k;
            expert_params.gate_exps = synced_params.gate_exps;
            expert_params.up_exps = synced_params.up_exps;
            expert_params.down_exps = synced_params.down_exps;
            expert_params.expert_intermediate = synced_params.expert_intermediate;
            expert_params.layer_idx = synced_params.layer_idx;
            expert_params.expert_mask = std::move(local_mask);
            expert_params.routing_indices = synced_params.routing_indices;
            expert_params.routing_weights = synced_params.routing_weights;
            expert_params.output = synced_params.output;
            expert_params.prepared_store = synced_params.prepared_store;

            if (!MoEExpertComputeStage::extractExpertViews(expert_params) ||
                !MoEExpertComputeStage::prepareExpertGemmEngines(expert_params))
            {
                local_ok = false;
            }
            else
            {
                MoEExpertComputeStage stage(std::move(expert_params));
                local_ok = stage.execute(device_context);
            }
        }

        if (!allRanksOk(domain.world_comm, local_ok))
            return false;

        return reduceFallbackPartialToRoot(domain, synced_params.output);
    }

    bool MoEExpertOverlayCPUFallback::runTensorParallelExpertFallback(
        const MoECPUFallbackDomainContext &domain,
        const MoECPUFallbackRunParams &params)
    {
        const bool relevant_rank = rankTouchesFallbackTransfer(domain);
        bool local_ok = !relevant_rank || validateTensorParallelRunParams(params, domain.participates());
        if (domain.participates())
        {
            const bool has_multi_rank_domain = domain.tp_context->degree() > 1;
            if (!has_multi_rank_domain)
                LOG_ERROR("[MoECPUFallback] TensorParallelExperts requires a multi-rank fallback domain");
            local_ok = local_ok && has_multi_rank_domain;
        }
        if (!allRanksOk(domain.world_comm, local_ok))
            return false;

        FallbackTransferPlan transfer_plan;
        if (!synchronizeFallbackTransferPlan(domain,
                                             resolveFallbackTransferMode(params),
                                             params.transfer_token_rows,
                                             transfer_plan))
        {
            return false;
        }

        MoECPUFallbackRunParams synced_params = params;
        synced_params.transfer_mode = transfer_plan.mode;
        synced_params.transfer_token_rows = std::move(transfer_plan.token_rows);

        if (synced_params.transfer_mode != MoEExpertTransferMode::DenseFullSequence)
            return runSparseTensorParallelExpertFallback(domain, synced_params, synced_params.transfer_mode);
        recordTransferStats(synced_params, synced_params.transfer_mode, synced_params.transfer_token_rows);

        if (!transferInputsToFallbackDomain(domain,
                                            {synced_params.input, synced_params.routing_indices, synced_params.routing_weights}))
        {
            return false;
        }

        std::vector<std::vector<std::pair<int, float>>> expert_token_lists;
        ExpertWeightHostData weights;
        int intermediate_begin = 0;
        int intermediate_end = 0;

        if (domain.participates())
        {
            const int domain_index = domain.tp_context->myIndex();
            const int domain_degree = domain.tp_context->degree();
            intermediate_begin = shardStart(synced_params.expert_intermediate, domain_degree, domain_index);
            intermediate_end = shardEnd(synced_params.expert_intermediate, domain_degree, domain_index);

            if (synced_params.tensor_parallel_stats)
            {
                synced_params.tensor_parallel_stats->domain_index = domain_index;
                synced_params.tensor_parallel_stats->domain_degree = domain_degree;
                synced_params.tensor_parallel_stats->intermediate_start = intermediate_begin;
                synced_params.tensor_parallel_stats->intermediate_end = intermediate_end;
                synced_params.tensor_parallel_stats->expert_allreduce_count = 0;
                synced_params.tensor_parallel_stats->processed_expert_ids.clear();
            }

            local_ok = buildFallbackTokenLists(synced_params, expert_token_lists) &&
                       prepareExpertWeightHostData(domain, synced_params, expert_token_lists, weights);
        }

        if (!allRanksOk(domain.world_comm, local_ok))
            return false;

        if (domain.participates())
        {
            float *output = synced_params.output->mutable_data();
            std::fill_n(output, synced_params.output->numel(), 0.0f);

            FP32Tensor expert_partial({static_cast<size_t>(synced_params.seq_len),
                                       static_cast<size_t>(synced_params.d_model)});

            for (int expert_id = 0; expert_id < synced_params.num_experts; ++expert_id)
            {
                const auto &token_list = expert_token_lists[static_cast<size_t>(expert_id)];
                if (token_list.empty())
                    continue;

                if (!computeTensorParallelExpertPartial(synced_params,
                                                        weights,
                                                        expert_id,
                                                        intermediate_begin,
                                                        intermediate_end,
                                                        token_list,
                                                        expert_partial))
                {
                    local_ok = false;
                    break;
                }

                domain.tp_context->barrier();
                if (!domain.tp_context->allreduce(&expert_partial,
                                                  "moe_cpu_fallback_tensor_parallel_expert",
                                                  expert_partial.numel()))
                {
                    local_ok = false;
                }
                domain.tp_context->barrier();

                if (!local_ok)
                    break;

                accumulateExpertOutput(synced_params, expert_partial, token_list);
                if (synced_params.tensor_parallel_stats)
                {
                    synced_params.tensor_parallel_stats->processed_expert_ids.push_back(expert_id);
                    ++synced_params.tensor_parallel_stats->expert_allreduce_count;
                }
            }
        }

        if (!allRanksOk(domain.world_comm, local_ok))
            return false;

        return returnFallbackOutputToRoot(domain, synced_params.output);
    }

    bool MoEExpertOverlayCPUFallback::runExpertFallback(
        const MoECPUFallbackDomainContext &domain,
        const MoECPUFallbackRunParams &params,
        IDeviceContext *device_context)
    {
        switch (domain.compute_kind)
        {
        case ExpertDomainComputeKind::TensorParallelExperts:
            return runTensorParallelExpertFallback(domain, params);
        case ExpertDomainComputeKind::ReplicatedExperts:
        case ExpertDomainComputeKind::ExpertIdSharded:
            return runReplicatedExpertFallback(domain, params, device_context);
        }

        LOG_ERROR("[MoECPUFallback] Unsupported CPU fallback compute kind");
        return false;
    }

} // namespace llaminar2