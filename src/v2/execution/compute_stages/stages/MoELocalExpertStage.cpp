/**
 * @file MoELocalExpertStage.cpp
 * @brief Implementation of participant-local graph-native sparse MoE expert compute stage.
 */

#include "MoELocalExpertStage.h"

#include "MoEExpertComputeStage.h"
#include "../../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../loaders/PreparedWeightStore.h"
#include "../../../tensors/CoherenceState.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <set>
#include <sstream>
#include <utility>

namespace llaminar2
{
    namespace
    {
        bool validateDeviceContext(IDeviceContext *ctx, DeviceId device, const char *stage_name)
        {
            if (!ctx)
            {
                LOG_ERROR("[" << stage_name << "] Null device context");
                return false;
            }
            (void)device;
            // Host-staged graph-native MoE stages can be scheduled inside a GPU graph
            // executor while executing participant-local CPU work, and GPU stages can
            // be driven by host orchestration while uploading compact payloads.
            return true;
        }

        bool validateSparseRows(const MoEOverlaySparseRows &rows, int top_k, int d_model)
        {
            if (rows.d_model != d_model || rows.top_k != top_k)
            {
                LOG_ERROR("[MoELocalExpertStage] Sparse row dimension mismatch: rows d_model="
                          << rows.d_model << " top_k=" << rows.top_k << ", stage d_model="
                          << d_model << " top_k=" << top_k);
                return false;
            }
            if (!rows.row_ids_host || !rows.entry_offsets_host || !rows.expert_ids_host ||
                !rows.route_weights_host || !rows.hidden_rows_fp32)
            {
                LOG_ERROR("[MoELocalExpertStage] Sparse input rows missing host buffers");
                return false;
            }
            if (rows.live_row_count > rows.row_capacity || rows.live_entry_count > rows.entry_capacity)
            {
                LOG_ERROR("[MoELocalExpertStage] Sparse live counts exceed capacity");
                return false;
            }
            return true;
        }

        /// True when prepared_gate_gemm (size == num_experts) or slab refs + store are set.
        bool hasPreparedExpertState(const MoELocalExpertStage::Params &p)
        {
            const auto expected = static_cast<size_t>(std::max(p.num_experts, 0));
            if (expected > 0 &&
                p.prepared_gate_gemm.size() == expected &&
                p.prepared_up_gemm.size() == expected &&
                p.prepared_down_gemm.size() == expected)
                return true;
            if (p.prepared_store && p.gate_slab_ref.has_value() &&
                p.up_slab_ref.has_value() && p.down_slab_ref.has_value())
                return true;
            return false;
        }
    } // namespace

    MoELocalExpertStage::MoELocalExpertStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (!params_.input_rows && params_.input_rows_lifetime)
            params_.input_rows = params_.input_rows_lifetime.get();
        if (!params_.output_rows && params_.output_rows_lifetime)
            params_.output_rows = params_.output_rows_lifetime.get();
    }

    bool MoELocalExpertStage::ensureCompactCapacity(size_t rows) const
    {
        const size_t capacity = std::max<size_t>(rows, 1u);
        if (compact_capacity_ >= capacity)
            return true;

        compact_hidden_ = std::make_shared<FP32Tensor>(std::vector<size_t>{capacity, static_cast<size_t>(params_.d_model)});
        compact_routing_indices_ = std::make_shared<FP32Tensor>(std::vector<size_t>{capacity, static_cast<size_t>(params_.top_k)});
        compact_routing_weights_ = std::make_shared<FP32Tensor>(std::vector<size_t>{capacity, static_cast<size_t>(params_.top_k)});
        compact_output_ = std::make_shared<FP32Tensor>(std::vector<size_t>{capacity, static_cast<size_t>(params_.d_model)});
        compact_capacity_ = capacity;
        return true;
    }

    bool MoELocalExpertStage::execute(IDeviceContext *ctx)
    {
        if (!validateDeviceContext(ctx, params_.device_id, "MoELocalExpertStage"))
            return false;
        if (!params_.input_rows || !params_.output_rows)
        {
            LOG_ERROR("[MoELocalExpertStage] Missing input or output row view");
            return false;
        }
        if (params_.num_experts <= 0 || params_.top_k <= 0 || params_.d_model <= 0 || params_.expert_intermediate <= 0)
        {
            LOG_ERROR("[MoELocalExpertStage] Invalid dimensions num_experts=" << params_.num_experts
                                                                              << " top_k=" << params_.top_k
                                                                              << " d_model=" << params_.d_model
                                                                              << " intermediate=" << params_.expert_intermediate);
            return false;
        }

        const bool has_prepared = hasPreparedExpertState(params_);
        if (!has_prepared && (!params_.gate_exps || !params_.up_exps || !params_.down_exps))
        {
            LOG_ERROR("[MoELocalExpertStage] Missing expert weight tensors and no prepared expert state");
            return false;
        }
        std::string prepared_error;
        if (!validatePreparedWeights(&prepared_error))
        {
            LOG_ERROR("[MoELocalExpertStage] Prepared weight validation failed: " << prepared_error);
            return false;
        }

        if (!params_.expert_mask.empty() && params_.expert_mask.size() != static_cast<size_t>(params_.num_experts))
        {
            LOG_ERROR("[MoELocalExpertStage] expert_mask size " << params_.expert_mask.size()
                                                                << " != num_experts " << params_.num_experts);
            return false;
        }
        const auto &input = *params_.input_rows;
        auto &output = *params_.output_rows;
        if (!validateSparseRows(input, params_.top_k, params_.d_model))
            return false;
        if (output.d_model != params_.d_model || !output.row_ids_host || !output.output_rows_fp32 ||
            output.row_capacity < input.live_row_count)
        {
            LOG_ERROR("[MoELocalExpertStage] Return row view cannot hold local expert output");
            return false;
        }

        output.key = input.key;
        output.source_participant = input.target_participant;
        output.target_participant = input.source_participant;
        output.live_row_count = 0;

        if (input.live_row_count == 0)
            return true;

        if (!ensureCompactCapacity(input.live_row_count))
            return false;

        float *hidden = compact_hidden_->mutable_data();
        float *routing_indices = compact_routing_indices_->mutable_data();
        float *routing_weights = compact_routing_weights_->mutable_data();
        float *compact_output = compact_output_->mutable_data();
        std::fill_n(routing_indices, input.live_row_count * static_cast<size_t>(params_.top_k), 0.0f);
        std::fill_n(routing_weights, input.live_row_count * static_cast<size_t>(params_.top_k), 0.0f);
        std::fill_n(compact_output, input.live_row_count * static_cast<size_t>(params_.d_model), 0.0f);

        for (size_t row = 0; row < input.live_row_count; ++row)
        {
            const int32_t entry_begin = input.entry_offsets_host[row];
            const int32_t entry_end = input.entry_offsets_host[row + 1u];
            if (entry_begin < 0 || entry_end < entry_begin ||
                entry_end > static_cast<int32_t>(input.live_entry_count) ||
                entry_end - entry_begin > params_.top_k)
            {
                LOG_ERROR("[MoELocalExpertStage] Invalid entry offsets for compact row " << row);
                return false;
            }

            std::memcpy(hidden + row * static_cast<size_t>(params_.d_model),
                        input.hidden_rows_fp32 + row * static_cast<size_t>(params_.d_model),
                        static_cast<size_t>(params_.d_model) * sizeof(float));

            for (int32_t entry = entry_begin; entry < entry_end; ++entry)
            {
                const int slot = static_cast<int>(entry - entry_begin);
                const int expert_id = input.expert_ids_host[entry];
                if (expert_id < 0 || expert_id >= params_.num_experts)
                {
                    LOG_ERROR("[MoELocalExpertStage] Expert id " << expert_id
                                                                 << " outside num_experts=" << params_.num_experts);
                    return false;
                }
                routing_indices[row * static_cast<size_t>(params_.top_k) + static_cast<size_t>(slot)] =
                    static_cast<float>(expert_id);
                routing_weights[row * static_cast<size_t>(params_.top_k) + static_cast<size_t>(slot)] =
                    input.route_weights_host[entry];
            }
        }

        MoEExpertComputeStage::Params compute_params;
        compute_params.device_id = params_.device_id;
        compute_params.input = compact_hidden_.get();
        compute_params.seq_len = static_cast<int>(input.live_row_count);
        compute_params.d_model = params_.d_model;
        compute_params.num_experts = params_.num_experts;
        compute_params.top_k = params_.top_k;
        compute_params.gate_exps = params_.gate_exps;
        compute_params.up_exps = params_.up_exps;
        compute_params.down_exps = params_.down_exps;
        compute_params.expert_intermediate = params_.expert_intermediate;
        compute_params.layer_idx = params_.layer_idx;
        compute_params.expert_mask = params_.expert_mask;
        compute_params.routing_indices = compact_routing_indices_.get();
        compute_params.routing_weights = compact_routing_weights_.get();
        compute_params.output = compact_output_.get();
        compute_params.output_registered_in_arena = false;
        // Propagate prepared expert state so execute() uses them directly.
        compute_params.prepared_gate_gemm = params_.prepared_gate_gemm;
        compute_params.prepared_up_gemm = params_.prepared_up_gemm;
        compute_params.prepared_down_gemm = params_.prepared_down_gemm;
        compute_params.prepared_store = params_.prepared_store;
        compute_params.expert_registry = params_.expert_registry;
        compute_params.gate_slab_ref = params_.gate_slab_ref;
        compute_params.up_slab_ref = params_.up_slab_ref;
        compute_params.down_slab_ref = params_.down_slab_ref;

        if (!has_prepared)
        {
            // Legacy / test fallback: extract expert views and prepare engines inline.
            if (!MoEExpertComputeStage::extractExpertViews(compute_params) ||
                !MoEExpertComputeStage::prepareExpertGemmEngines(compute_params))
            {
                LOG_ERROR("[MoELocalExpertStage] Failed to prepare compact expert compute stage");
                return false;
            }
        }

        // For GPU participants: upload compact input tensors to device before expert
        // kernel dispatch.  After execute(), reading compact_output_->data() triggers
        // an implicit D2H sync via the coherence state machine.
        const bool is_gpu = params_.device_id.is_gpu();
        if (is_gpu)
        {
            if (!compact_hidden_->ensureOnDevice(params_.device_id) ||
                !compact_routing_indices_->ensureOnDevice(params_.device_id) ||
                !compact_routing_weights_->ensureOnDevice(params_.device_id) ||
                !compact_output_->ensureOnDevice(params_.device_id))
            {
                LOG_ERROR("[MoELocalExpertStage] GPU coherence upload failed for compact tensors");
                return false;
            }
        }

        MoEExpertComputeStage compute_stage(std::move(compute_params));
        if (bound_workspace_)
            compute_stage.bindWorkspace(bound_workspace_);
        else if (is_gpu)
        {
            LOG_ERROR("[MoELocalExpertStage] GPU local expert compute requires a bound workspace");
            return false;
        }
        if (has_prepared)
        {
            // Signal to MoEExpertComputeStage that raw weight pointers may be null —
            // prepared engines are used instead.
            compute_stage.releaseRawExpertWeights();
        }

        std::chrono::steady_clock::time_point t_compute_start;
        if (MoEExpertOverlayProfiler::isEnabled())
            t_compute_start = std::chrono::steady_clock::now();

        if (!compute_stage.execute(ctx))
            return false;

        if (MoEExpertOverlayProfiler::isEnabled())
        {
            const double compute_ms = std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - t_compute_start)
                                          .count();
            std::set<int> unique_set;
            for (size_t e = 0; e < input.live_entry_count; ++e)
                unique_set.insert(input.expert_ids_host[e]);
            MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
                params_.layer_idx,
                params_.device_id.to_string(),
                params_.device_id.is_cpu(),
                input.live_row_count,
                input.live_row_count, // output rows == input rows for local expert
                std::vector<int>(unique_set.begin(), unique_set.end()),
                compute_ms);
        }

        if (is_gpu)
        {
            // Mark output device-authoritative so that data() below triggers D2H.
            compact_output_->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        }

        const float *compact_result = compact_output_->data();
        for (size_t row = 0; row < input.live_row_count; ++row)
        {
            output.row_ids_host[row] = input.row_ids_host[row];
            std::memcpy(output.output_rows_fp32 + row * static_cast<size_t>(params_.d_model),
                        compact_result + row * static_cast<size_t>(params_.d_model),
                        static_cast<size_t>(params_.d_model) * sizeof(float));
        }
        output.live_row_count = input.live_row_count;
        return true;
    }

    bool MoELocalExpertStage::validatePreparedWeights(std::string *error) const
    {
        if (error)
            error->clear();

        const bool has_prepared_vectors =
            !params_.prepared_gate_gemm.empty() &&
            params_.prepared_gate_gemm.size() == static_cast<size_t>(params_.num_experts);
        const bool has_slab_refs = params_.prepared_store &&
                                   params_.gate_slab_ref.has_value() &&
                                   params_.up_slab_ref.has_value() &&
                                   params_.down_slab_ref.has_value();

        if (!has_prepared_vectors && !has_slab_refs)
        {
            // No prepared state — stage will use inline view extraction (legacy path).
            // This is only valid when raw tensors are present.
            if (params_.gate_exps && params_.up_exps && params_.down_exps)
                return true;
            if (error)
                *error = "[MoELocalExpertStage] No prepared expert state and no raw expert tensors";
            return false;
        }

        if (has_prepared_vectors)
        {
            // Verify every active expert has a non-null engine.
            const auto &gate_v = params_.prepared_gate_gemm;
            const auto &up_v = params_.prepared_up_gemm;
            const auto &down_v = params_.prepared_down_gemm;

            if (up_v.size() != gate_v.size() || down_v.size() != gate_v.size())
            {
                if (error)
                    *error = "[MoELocalExpertStage] Prepared engine vector size mismatch";
                return false;
            }

            for (int expert = 0; expert < params_.num_experts; ++expert)
            {
                const bool active = params_.expert_mask.empty() ||
                                    (static_cast<size_t>(expert) < params_.expert_mask.size() &&
                                     params_.expert_mask[static_cast<size_t>(expert)]);
                if (!active)
                    continue;
                const bool ok = gate_v[expert] && up_v[expert] && down_v[expert];
                if (!ok)
                {
                    if (error)
                    {
                        std::ostringstream oss;
                        oss << "[MoELocalExpertStage] Layer " << params_.layer_idx
                            << " expert " << expert << " is active but prepared engine is null"
                            << " (gate=" << (bool)gate_v[expert]
                            << " up=" << (bool)up_v[expert]
                            << " down=" << (bool)down_v[expert] << ")";
                        *error = oss.str();
                    }
                    return false;
                }
            }
            return true;
        }

        // Slab-ref path: verify store contains every active expert in each slab.
        if (!has_slab_refs)
        {
            if (error)
                *error = "[MoELocalExpertStage] validatePreparedWeights reached slab path without all slab refs — internal logic error";
            return false;
        }

        auto validate_slab = [&](const char *name, const ExpertSlabRef &ref)
        {
            const auto availability = params_.prepared_store->expertAvailabilityMask(ref);
            if (availability.empty())
            {
                if (error)
                    *error = std::string("[MoELocalExpertStage] PreparedWeightStore missing expert slab for ") + name;
                return false;
            }
            if (availability.size() != static_cast<size_t>(params_.num_experts))
            {
                if (error)
                {
                    std::ostringstream oss;
                    oss << "[MoELocalExpertStage] PreparedWeightStore slab size mismatch for " << name
                        << ": got " << availability.size()
                        << ", expected " << params_.num_experts;
                    *error = oss.str();
                }
                return false;
            }

            for (int expert = 0; expert < params_.num_experts; ++expert)
            {
                const bool active = params_.expert_mask.empty() ||
                                    params_.expert_mask[static_cast<size_t>(expert)];
                if (!active)
                    continue;
                if (!availability[static_cast<size_t>(expert)])
                {
                    if (error)
                    {
                        std::ostringstream oss;
                        oss << "[MoELocalExpertStage] Layer " << params_.layer_idx
                            << " expert " << expert << " is active but missing in PreparedWeightStore slab "
                            << name;
                        *error = oss.str();
                    }
                    return false;
                }
            }
            return true;
        };

        if (!validate_slab("gate", *params_.gate_slab_ref) ||
            !validate_slab("up", *params_.up_slab_ref) ||
            !validate_slab("down", *params_.down_slab_ref))
        {
            return false;
        }
        return true;
    }

    size_t MoELocalExpertStage::estimatedFlops() const
    {
        const size_t rows = params_.input_rows ? params_.input_rows->live_row_count : 0;
        return rows * static_cast<size_t>(params_.top_k) *
               static_cast<size_t>(6) * static_cast<size_t>(params_.d_model) *
               static_cast<size_t>(params_.expert_intermediate);
    }

    bool MoELocalExpertStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU ||
               backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageBufferRequirements MoELocalExpertStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.gate_exps)
            reqs.addWeight("gate_exps", params_.gate_exps->shape(), toBufferTensorType(params_.gate_exps->native_type()));
        if (params_.up_exps)
            reqs.addWeight("up_exps", params_.up_exps->shape(), toBufferTensorType(params_.up_exps->native_type()));
        if (params_.down_exps)
            reqs.addWeight("down_exps", params_.down_exps->shape(), toBufferTensorType(params_.down_exps->native_type()));
        return reqs;
    }

    StageDumpInfo MoELocalExpertStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.gate_exps)
            info.addWeight("gate_exps", params_.gate_exps);
        if (params_.up_exps)
            info.addWeight("up_exps", params_.up_exps);
        if (params_.down_exps)
            info.addWeight("down_exps", params_.down_exps);
        info.addScalarInt("num_experts", params_.num_experts);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("expert_intermediate", params_.expert_intermediate);
        info.addScalarInt("layer_idx", params_.layer_idx);
        info.addScalarInt("live_rows", params_.input_rows ? static_cast<int>(params_.input_rows->live_row_count) : 0);
        return info;
    }

    WorkspaceRequirements MoELocalExpertStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        auto firstRequirements = [&](const std::vector<ITensorGemm *> &engines) -> WorkspaceRequirements
        {
            for (auto *gemm : engines)
            {
                if (!gemm)
                    continue;
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer)
                    return consumer->getWorkspaceRequirements(m, n, k);
            }
            return WorkspaceRequirements{};
        };

        WorkspaceRequirements reqs = firstRequirements(params_.prepared_gate_gemm);
        if (!reqs.buffers.empty())
            return reqs;
        reqs = firstRequirements(params_.prepared_up_gemm);
        if (!reqs.buffers.empty())
            return reqs;
        return firstRequirements(params_.prepared_down_gemm);
    }

    void MoELocalExpertStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        auto bindAll = [workspace](const std::vector<ITensorGemm *> &engines)
        {
            for (auto *gemm : engines)
            {
                if (!gemm)
                    continue;
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer)
                    consumer->bindWorkspace(workspace);
            }
        };

        bindAll(params_.prepared_gate_gemm);
        bindAll(params_.prepared_up_gemm);
        bindAll(params_.prepared_down_gemm);
        bound_workspace_ = workspace;
    }

    void MoELocalExpertStage::unbindWorkspace()
    {
        auto unbindAll = [](const std::vector<ITensorGemm *> &engines)
        {
            for (auto *gemm : engines)
            {
                if (!gemm)
                    continue;
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer)
                    consumer->unbindWorkspace();
            }
        };

        unbindAll(params_.prepared_gate_gemm);
        unbindAll(params_.prepared_up_gemm);
        unbindAll(params_.prepared_down_gemm);
        bound_workspace_ = nullptr;
    }

} // namespace llaminar2