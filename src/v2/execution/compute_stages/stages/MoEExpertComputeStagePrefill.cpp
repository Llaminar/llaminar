/**
 * @file MoEExpertComputeStagePrefill.cpp
 * @brief MoEExpertComputeStage prefill, decode-equivalent verifier, and LLEP routing execution.
 */
#include "MoEExpertComputeStageInternal.h"

namespace llaminar2
{

    bool MoEExpertComputeStage::ensureGroupedGateUpDescriptorTable(
        IMoEKernel *kernel, int d_model, int intermediate)
    {
        if (!kernel || params_.num_experts <= 0 || d_model <= 0 || intermediate <= 0)
            return false;

        if (grouped_gateup_desc_table_id_ >= 0 &&
            grouped_gateup_desc_table_num_experts_ == params_.num_experts &&
            grouped_gateup_desc_table_d_model_ == d_model &&
            grouped_gateup_desc_table_intermediate_ == intermediate &&
            !grouped_gateup_desc_table_dirty_)
        {
            return true;
        }

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] Grouped gate/up descriptor table is dirty during graph capture"
                      << " layer=" << params_.layer_idx);
            return false;
        }

        std::vector<DeviceNativeQuantisedMatrixDesc> gate_descs(
            static_cast<size_t>(params_.num_experts));
        std::vector<DeviceNativeQuantisedMatrixDesc> up_descs(
            static_cast<size_t>(params_.num_experts));
        std::vector<DeviceMoEFloatingMatrixDesc> floating_gate_descs(
            static_cast<size_t>(params_.num_experts));
        std::vector<DeviceMoEFloatingMatrixDesc> floating_up_descs(
            static_cast<size_t>(params_.num_experts));
        std::optional<DeviceMoEWeightFormat> table_format;
        int valid_descs = 0;
        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            if (cached_gate_gemm_.size() <= static_cast<size_t>(expert_id) ||
                cached_up_gemm_.size() <= static_cast<size_t>(expert_id) ||
                cached_down_gemm_.size() <= static_cast<size_t>(expert_id) ||
                !cached_gate_gemm_[static_cast<size_t>(expert_id)] ||
                !cached_up_gemm_[static_cast<size_t>(expert_id)] ||
                !cached_down_gemm_[static_cast<size_t>(expert_id)])
            {
                continue;
            }

            DeviceMoEExpertDescriptor exported;
            if (!exportDeviceMoEExpertWeightDescriptors(
                    cached_gate_gemm_[static_cast<size_t>(expert_id)],
                    cached_up_gemm_[static_cast<size_t>(expert_id)],
                    cached_down_gemm_[static_cast<size_t>(expert_id)],
                    d_model,
                    intermediate,
                    exported))
            {
                LOG_DEBUG("[MoEExpertComputeStage] Unable to export grouped gate/up descriptor for expert "
                          << expert_id << " layer " << params_.layer_idx);
                return false;
            }
            if (table_format && *table_format != exported.weight_format)
            {
                LOG_ERROR("[MoEExpertComputeStage] Mixed routed-expert weight formats in layer "
                          << params_.layer_idx << " expert=" << expert_id);
                return false;
            }
            table_format = exported.weight_format;

            if (exported.weight_format == DeviceMoEWeightFormat::NativeQuantised)
            {
                gate_descs[static_cast<size_t>(expert_id)] = exported.gate;
                up_descs[static_cast<size_t>(expert_id)] = exported.up;
            }
            else
            {
                floating_gate_descs[static_cast<size_t>(expert_id)] =
                    exported.floating_gate;
                floating_up_descs[static_cast<size_t>(expert_id)] =
                    exported.floating_up;
            }
            ++valid_descs;
        }

        if (valid_descs == 0 || !table_format)
            return false;

        int table_id = grouped_gateup_desc_table_id_;
        if (table_id >= 0 &&
            grouped_gateup_desc_table_num_experts_ == params_.num_experts &&
            grouped_gateup_desc_table_d_model_ == d_model &&
            grouped_gateup_desc_table_intermediate_ == intermediate &&
            grouped_gateup_desc_table_weight_format_ == *table_format)
        {
            const bool updated =
                *table_format == DeviceMoEWeightFormat::NativeQuantised
                    ? kernel->updateGroupedExpertGateUpDescriptorTables(
                          table_id,
                          gate_descs.data(),
                          up_descs.data(),
                          params_.num_experts,
                          d_model,
                          intermediate)
                    : kernel->updateGroupedExpertFloatingGateUpDescriptorTables(
                          table_id,
                          floating_gate_descs.data(),
                          floating_up_descs.data(),
                          *table_format,
                          params_.num_experts,
                          d_model,
                          intermediate);
            if (!updated)
            {
                return false;
            }
            grouped_gateup_desc_table_dirty_ = false;
            return true;
        }

        table_id = *table_format == DeviceMoEWeightFormat::NativeQuantised
                       ? kernel->uploadGroupedExpertGateUpDescriptorTables(
                             gate_descs.data(),
                             up_descs.data(),
                             params_.num_experts,
                             d_model,
                             intermediate,
                             params_.weight_descriptor_source)
                       : kernel->uploadGroupedExpertFloatingGateUpDescriptorTables(
                             floating_gate_descs.data(),
                             floating_up_descs.data(),
                             *table_format,
                             params_.num_experts,
                             d_model,
                             intermediate);
        if (table_id < 0)
            return false;

        grouped_gateup_desc_table_id_ = table_id;
        grouped_gateup_desc_table_num_experts_ = params_.num_experts;
        grouped_gateup_desc_table_d_model_ = d_model;
        grouped_gateup_desc_table_intermediate_ = intermediate;
        grouped_gateup_desc_table_weight_format_ = *table_format;
        grouped_gateup_desc_table_dirty_ = false;
        runtime_grouped_decode_launch_state_prepared_ = false;
        return true;
    }

    bool MoEExpertComputeStage::ensureGroupedDownDescriptorTable(
        IMoEKernel *kernel, int d_model, int intermediate)
    {
        if (!kernel || params_.num_experts <= 0 || d_model <= 0 || intermediate <= 0)
            return false;

        if (grouped_down_desc_table_id_ >= 0 &&
            grouped_down_desc_table_num_experts_ == params_.num_experts &&
            grouped_down_desc_table_d_model_ == d_model &&
            grouped_down_desc_table_intermediate_ == intermediate &&
            !grouped_down_desc_table_dirty_)
        {
            return true;
        }

        if (isGraphCaptureActive())
        {
            LOG_ERROR("[MoEExpertComputeStage] Grouped down descriptor table is dirty during graph capture"
                      << " layer=" << params_.layer_idx);
            return false;
        }

        std::vector<DeviceNativeQuantisedMatrixDesc> down_descs(
            static_cast<size_t>(params_.num_experts));
        std::vector<DeviceMoEFloatingMatrixDesc> floating_down_descs(
            static_cast<size_t>(params_.num_experts));
        std::optional<DeviceMoEWeightFormat> table_format;
        int valid_descs = 0;
        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            if (cached_gate_gemm_.size() <= static_cast<size_t>(expert_id) ||
                cached_up_gemm_.size() <= static_cast<size_t>(expert_id) ||
                cached_down_gemm_.size() <= static_cast<size_t>(expert_id) ||
                !cached_gate_gemm_[static_cast<size_t>(expert_id)] ||
                !cached_up_gemm_[static_cast<size_t>(expert_id)] ||
                !cached_down_gemm_[static_cast<size_t>(expert_id)])
            {
                continue;
            }

            DeviceMoEExpertDescriptor exported;
            if (!exportDeviceMoEExpertWeightDescriptors(
                    cached_gate_gemm_[static_cast<size_t>(expert_id)],
                    cached_up_gemm_[static_cast<size_t>(expert_id)],
                    cached_down_gemm_[static_cast<size_t>(expert_id)],
                    d_model,
                    intermediate,
                    exported))
            {
                LOG_DEBUG("[MoEExpertComputeStage] Unable to export grouped down descriptor for expert "
                          << expert_id << " layer " << params_.layer_idx);
                return false;
            }
            if (table_format && *table_format != exported.weight_format)
            {
                LOG_ERROR("[MoEExpertComputeStage] Mixed routed-expert weight formats in layer "
                          << params_.layer_idx << " expert=" << expert_id);
                return false;
            }
            table_format = exported.weight_format;

            if (exported.weight_format == DeviceMoEWeightFormat::NativeQuantised)
            {
                down_descs[static_cast<size_t>(expert_id)] = exported.down;
            }
            else
            {
                floating_down_descs[static_cast<size_t>(expert_id)] =
                    exported.floating_down;
            }
            ++valid_descs;
        }

        if (valid_descs == 0 || !table_format)
            return false;

        int table_id = grouped_down_desc_table_id_;
        if (table_id >= 0 &&
            grouped_down_desc_table_num_experts_ == params_.num_experts &&
            grouped_down_desc_table_d_model_ == d_model &&
            grouped_down_desc_table_intermediate_ == intermediate &&
            grouped_down_desc_table_weight_format_ == *table_format)
        {
            const bool updated =
                *table_format == DeviceMoEWeightFormat::NativeQuantised
                    ? kernel->updateGroupedExpertDownDescriptorTable(
                          table_id,
                          down_descs.data(),
                          params_.num_experts,
                          d_model,
                          intermediate)
                    : kernel->updateGroupedExpertFloatingDownDescriptorTable(
                          table_id,
                          floating_down_descs.data(),
                          *table_format,
                          params_.num_experts,
                          d_model,
                          intermediate);
            if (!updated)
            {
                return false;
            }
            grouped_down_desc_table_dirty_ = false;
            return true;
        }

        table_id = *table_format == DeviceMoEWeightFormat::NativeQuantised
                       ? kernel->uploadGroupedExpertDownDescriptorTable(
                             down_descs.data(),
                             params_.num_experts,
                             d_model,
                             intermediate,
                             params_.weight_descriptor_source)
                       : kernel->uploadGroupedExpertFloatingDownDescriptorTable(
                             floating_down_descs.data(),
                             *table_format,
                             params_.num_experts,
                             d_model,
                             intermediate);
        if (table_id < 0)
            return false;

        grouped_down_desc_table_id_ = table_id;
        grouped_down_desc_table_num_experts_ = params_.num_experts;
        grouped_down_desc_table_d_model_ = d_model;
        grouped_down_desc_table_intermediate_ = intermediate;
        grouped_down_desc_table_weight_format_ = *table_format;
        grouped_down_desc_table_dirty_ = false;
        runtime_grouped_decode_launch_state_prepared_ = false;
        return true;
    }

    bool MoEExpertComputeStage::ensureCombinedSharedVerifierResources(
        IMoEKernel *kernel, int d_model, int intermediate)
    {
        if (!kernel || !params_.combine_shared_expert_in_verifier ||
            !params_.prepared_store ||
            !params_.prepared_shared_ref_gate ||
            !params_.prepared_shared_ref_up ||
            !params_.prepared_shared_ref_down)
        {
            LOG_ERROR("[MoEExpertComputeStage] Combined shared verifier path is missing required prepared state"
                      << " layer=" << params_.layer_idx
                      << " kernel=" << (kernel ? "yes" : "no")
                      << " store=" << (params_.prepared_store ? "yes" : "no")
                      << " gate_ref=" << (params_.prepared_shared_ref_gate ? "yes" : "no")
                      << " up_ref=" << (params_.prepared_shared_ref_up ? "yes" : "no")
                      << " down_ref=" << (params_.prepared_shared_ref_down ? "yes" : "no"));
            return false;
        }

        if (combined_shared_gate_gemm_ &&
            combined_shared_up_gemm_ &&
            combined_shared_down_gemm_ &&
            combined_shared_desc_table_d_model_ == d_model &&
            combined_shared_desc_table_intermediate_ == intermediate &&
            combined_shared_gateup_desc_table_id_ >= 0 &&
            combined_shared_gateup_desc_table_d_model_ == d_model &&
            combined_shared_gateup_desc_table_intermediate_ == intermediate &&
            combined_shared_down_desc_table_id_ >= 0 &&
            combined_shared_down_desc_table_d_model_ == d_model &&
            combined_shared_down_desc_table_intermediate_ == intermediate)
        {
            return true;
        }

        ITensorGemm *shared_gate =
            params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_gate);
        ITensorGemm *shared_up =
            params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_up);
        ITensorGemm *shared_down =
            params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_down);
        if (!shared_gate || !shared_up || !shared_down)
        {
            LOG_ERROR("[MoEExpertComputeStage] Combined shared verifier path requires prepared shared expert GEMM engines"
                      << " layer=" << params_.layer_idx
                      << " gate=" << (shared_gate ? "yes" : "no")
                      << " up=" << (shared_up ? "yes" : "no")
                      << " down=" << (shared_down ? "yes" : "no"));
            return false;
        }

        bindStageStream(shared_gate);
        bindStageStream(shared_up);
        bindStageStream(shared_down);
        auto bind_if_needed = [&](ITensorGemm *gemm)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm))
                consumer->bindWorkspace(bound_workspace_);
        };
        bind_if_needed(shared_gate);
        bind_if_needed(shared_up);
        bind_if_needed(shared_down);

        DeviceNativeQuantisedMatrixDesc gate_desc;
        DeviceNativeQuantisedMatrixDesc up_desc;
        DeviceNativeQuantisedMatrixDesc down_desc;
        if (!shared_gate->exportNativeQuantisedMatrixDesc(gate_desc) ||
            !shared_up->exportNativeQuantisedMatrixDesc(up_desc) ||
            !shared_down->exportNativeQuantisedMatrixDesc(down_desc) ||
            gate_desc.n != intermediate || gate_desc.k != d_model ||
            up_desc.n != intermediate || up_desc.k != d_model ||
            down_desc.n != d_model || down_desc.k != intermediate)
        {
            LOG_ERROR("[MoEExpertComputeStage] Combined safe verifier requires shared expert "
                      "GEMM engines with decode-equivalent native descriptors"
                      << " layer=" << params_.layer_idx);
            return false;
        }

        const int gateup_table_id = kernel->uploadGroupedExpertGateUpDescriptorTables(
            &gate_desc, &up_desc, /*num_experts=*/1, d_model, intermediate);
        if (gateup_table_id < 0)
        {
            LOG_ERROR("[MoEExpertComputeStage] Combined safe verifier failed to upload "
                      "shared expert grouped gate/up descriptor table"
                      << " layer=" << params_.layer_idx);
            return false;
        }

        const int down_table_id = kernel->uploadGroupedExpertDownDescriptorTable(
            &down_desc, /*num_experts=*/1, d_model, intermediate);
        if (down_table_id < 0)
        {
            LOG_ERROR("[MoEExpertComputeStage] Combined safe verifier failed to upload "
                      "shared expert grouped down descriptor table"
                      << " layer=" << params_.layer_idx);
            return false;
        }

        combined_shared_gate_gemm_ = shared_gate;
        combined_shared_up_gemm_ = shared_up;
        combined_shared_down_gemm_ = shared_down;
        combined_shared_desc_table_d_model_ = d_model;
        combined_shared_desc_table_intermediate_ = intermediate;
        combined_shared_gateup_desc_table_id_ = gateup_table_id;
        combined_shared_gateup_desc_table_d_model_ = d_model;
        combined_shared_gateup_desc_table_intermediate_ = intermediate;
        combined_shared_down_desc_table_id_ = down_table_id;
        combined_shared_down_desc_table_d_model_ = d_model;
        combined_shared_down_desc_table_intermediate_ = intermediate;
        return true;
    }

    bool MoEExpertComputeStage::initializeMoERuntimeTableForGroupedDecode()
    {
        if (!params_.moe_runtime_table || params_.layer_idx < 0 ||
            params_.num_experts <= 0 || params_.top_k <= 0 ||
            !supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id))
        {
            return false;
        }

        try
        {
            if (!moe_runtime_layer_)
                moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
            if (runtimeTableHasActiveGroupedDecodeBank())
                return true;

            const bool publication_required =
                params_.moe_runtime_table->decodeRuntimePublicationRequired(params_.layer_idx);
            const auto &state = params_.moe_runtime_table->hostLayerState(params_.layer_idx);
            if (!publication_required &&
                state.active_epoch == std::numeric_limits<uint32_t>::max())
            {
                LOG_ERROR("[MoEExpertComputeStage] Cannot initialize MoE runtime decode bank for layer "
                          << params_.layer_idx << ": epoch counter exhausted");
                return false;
            }

            const bool has_replicas = params_.replica_set.num_replicated > 0;
            if (usesRuntimePlacementWeightDescriptors() ||
                params_.runtime_decode_has_explicit_owner_metadata)
            {
                LOG_ERROR("[MoEExpertComputeStage] Cannot synthesize MoE runtime decode bank for layer "
                          << params_.layer_idx
                          << ": explicit owner/resident metadata decode requires a "
                          << "graph-initialized placement table");
                return false;
            }
            if (has_replicas &&
                (!params_.replica_set.hasLayerReplicaPlacement() ||
                 params_.replica_set.base_ownership.expertCount() != params_.num_experts ||
                 params_.layer_idx < 0 ||
                 params_.layer_idx >= params_.replica_set.base_ownership.layerCount()))
            {
                LOG_ERROR("[MoEExpertComputeStage] Cannot initialize MoE runtime decode bank for layer "
                          << params_.layer_idx << ": replica metadata does not match num_experts="
                          << params_.num_experts);
                return false;
            }

            const int participant_count = expectedGroupedDecodeParticipantCount();
            if (params_.my_socket_id < 0 ||
                params_.my_socket_id >= participant_count ||
                participant_count > static_cast<int>(kDeviceMoEMaxParticipants))
            {
                LOG_ERROR("[MoEExpertComputeStage] Cannot initialize MoE runtime decode bank for layer "
                          << params_.layer_idx << ": invalid participant metadata id="
                          << params_.my_socket_id << " count=" << participant_count);
                return false;
            }
            if (!has_replicas &&
                !expertMaskAllEnabled() &&
                participant_count > 1)
            {
                LOG_ERROR("[MoEExpertComputeStage] Cannot synthesize masked multi-participant "
                          << "MoE runtime decode bank for layer " << params_.layer_idx
                          << ": explicit owner metadata is required");
                return false;
            }

            all_expert_ids_.clear();
            all_expert_ids_.reserve(static_cast<size_t>(params_.num_experts));
            for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
            {
                if (expertComputesLocally(expert_id))
                    all_expert_ids_.push_back(expert_id);
            }
            if (all_expert_ids_.empty())
                return false;
            if (!ensureGemmEnginesForExperts(all_expert_ids_))
                return false;

            MoEPlacementUpdate update;
            update.epoch = publication_required ? 1u : state.active_epoch + 1u;
            update.expert_count = static_cast<uint32_t>(params_.num_experts);
            update.participant_id = static_cast<uint32_t>(params_.my_socket_id);
            update.participant_count = static_cast<uint32_t>(participant_count);
            update.experts.resize(static_cast<size_t>(params_.num_experts));
            update.local_compute_mask.assign(static_cast<size_t>(params_.num_experts), 0u);
            update.replica_role.assign(static_cast<size_t>(params_.num_experts),
                                       static_cast<uint8_t>(DeviceMoEReplicaRole::None));
            update.resident_participant_mask.assign(static_cast<size_t>(params_.num_experts), 0u);

            for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
            {
                const bool computes_locally = expertComputesLocally(expert_id);
                const bool replicated =
                    has_replicas && params_.replica_set.isReplicatedForLayer(params_.layer_idx, expert_id);
                int owner_participant = params_.my_socket_id;
                if (has_replicas)
                    owner_participant = params_.replica_set.ownerParticipant(
                        params_.layer_idx, expert_id);

                uint32_t resident_mask = 0u;
                if (owner_participant >= 0 && owner_participant < participant_count)
                    resident_mask |= (1u << static_cast<uint32_t>(owner_participant));
                if (replicated)
                {
                    for (int participant = 0; participant < participant_count; ++participant)
                    {
                        if (params_.replica_set.hasReplicaOnParticipant(
                                params_.layer_idx, expert_id, participant))
                        {
                            resident_mask |= (1u << static_cast<uint32_t>(participant));
                        }
                    }
                }
                if (computes_locally)
                    resident_mask |= (1u << static_cast<uint32_t>(params_.my_socket_id));
                update.resident_participant_mask[static_cast<size_t>(expert_id)] = resident_mask;

                DeviceMoEExpertDescriptor desc;
                desc.logical_expert_id = expert_id;
                desc.owner_participant = owner_participant;
                desc.local_slot = computes_locally ? expert_id : -1;
                if (replicated)
                    desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Replicated);
                update.experts[static_cast<size_t>(expert_id)] = desc;

                if (!expertComputesLocally(expert_id))
                    continue;

                if (!exportDeviceMoEExpertWeightDescriptors(
                        cached_gate_gemm_[static_cast<size_t>(expert_id)],
                        cached_up_gemm_[static_cast<size_t>(expert_id)],
                        cached_down_gemm_[static_cast<size_t>(expert_id)],
                        params_.d_model,
                        params_.expert_intermediate,
                        desc))
                {
                    LOG_DEBUG("[MoEExpertComputeStage] Cannot initialize MoE runtime decode bank for layer "
                              << params_.layer_idx << " expert " << expert_id
                              << ": prepared GEMM engines did not export a uniform "
                                 "NativeQuantised/FP16/BF16/FP32 descriptor family");
                    return false;
                }

                DeviceMoEExpertFlags flags = DeviceMoEExpertFlags::Valid |
                                             DeviceMoEExpertFlags::Resident |
                                             DeviceMoEExpertFlags::LocalCompute;
                if (!has_replicas || owner_participant == params_.my_socket_id)
                    flags |= DeviceMoEExpertFlags::PreferredOwner;
                if (replicated)
                    flags |= DeviceMoEExpertFlags::Replicated;

                desc.local_slot = expert_id;
                desc.flags = toMoEExpertFlags(flags);
                update.experts[static_cast<size_t>(expert_id)] = desc;
                update.local_compute_mask[static_cast<size_t>(expert_id)] = 1u;
                if (replicated)
                {
                    update.replica_role[static_cast<size_t>(expert_id)] =
                        static_cast<uint8_t>(owner_participant == params_.my_socket_id
                                                 ? DeviceMoEReplicaRole::Primary
                                                 : DeviceMoEReplicaRole::Replica);
                }
                else
                {
                    update.replica_role[static_cast<size_t>(expert_id)] =
                        static_cast<uint8_t>(DeviceMoEReplicaRole::Primary);
                }
            }

            params_.moe_runtime_table->prepareInactiveBank(params_.layer_idx, update);
            params_.moe_runtime_table->flipActiveBank(params_.layer_idx, update.epoch, gpuStream());
            moe_runtime_layer_ = params_.moe_runtime_table->deviceLayerState(params_.layer_idx);
            return runtimeTableHasActiveGroupedDecodeBank();
        }
        catch (const std::exception &ex)
        {
            LOG_ERROR("[MoEExpertComputeStage] Failed to initialize MoE runtime decode bank for layer "
                      << params_.layer_idx << ": " << ex.what());
            return false;
        }
    }

    int MoEExpertComputeStage::expectedGroupedDecodeParticipantCount() const
    {
        int participant_count = params_.participant_count;
        const bool has_replicas = params_.replica_set.num_replicated > 0;
        if (participant_count <= 0 && has_replicas)
            participant_count = params_.replica_set.participantCount();
        if (participant_count <= 0)
            participant_count = std::max(1, params_.my_socket_id + 1);
        return participant_count;
    }

    bool MoEExpertComputeStage::initializeMoERuntimeTableForGroupedRows()
    {
        moe_runtime_row_grouping_available_ = false;
        const bool fully_replicated_local_runtime_grouping =
            params_.routed_row_execution_policy ==
                RoutedExpertRowExecutionPolicy::FullyReplicatedLocal &&
            hasFullLocalExpertOwnership() &&
            expertMaskAllEnabled();
        const bool static_owner_runtime_grouping =
            params_.routed_row_execution_policy ==
                RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
            params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::StaticOwner &&
            ((hasFullLocalExpertOwnership() && expertMaskAllEnabled()) ||
             hasFixedTopologyPrefillExpertMask());
        const bool least_loaded_runtime_grouping =
            params_.routed_row_execution_policy ==
                RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
            params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident &&
            hasFixedTopologyPrefillExpertMask();
        if (!params_.use_runtime_row_grouping ||
            !params_.moe_runtime_table ||
            params_.layer_idx < 0 ||
            params_.seq_len <= 1 ||
            params_.num_experts <= 0 ||
            params_.top_k <= 0 ||
            !supportsGroupedPrefillExecutionBackend(params_.device_id) ||
            (!fully_replicated_local_runtime_grouping &&
             !static_owner_runtime_grouping &&
             !least_loaded_runtime_grouping))
        {
            return false;
        }

        auto *runtime_table = dynamic_cast<DeviceMoERuntimeTable *>(params_.moe_runtime_table);
        if (!runtime_table)
        {
            LOG_ERROR("[MoEExpertComputeStage] Runtime prefill grouping requires DeviceMoERuntimeTable"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        if (!runtime_table->hasPrefillRouteScratchCapacity(params_.layer_idx, params_.seq_len))
        {
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[MoEExpertComputeStage] Runtime prefill scratch was not warmed before graph capture"
                          << " layer=" << params_.layer_idx
                          << " seq_len=" << params_.seq_len);
                return false;
            }
            runtime_table->ensurePrefillRouteScratchCapacity(params_.seq_len, gpuStream());
        }

        if (!runtime_table->hasPrefillRouteScratchCapacity(params_.layer_idx, params_.seq_len))
        {
            LOG_ERROR("[MoEExpertComputeStage] Runtime prefill scratch capacity check failed"
                      << " layer=" << params_.layer_idx
                      << " seq_len=" << params_.seq_len);
            return false;
        }

        moe_runtime_layer_ = runtime_table->deviceLayerState(params_.layer_idx);
        const auto &state = runtime_table->hostLayerState(params_.layer_idx);
        moe_runtime_row_grouping_available_ =
            moe_runtime_layer_ &&
            state.expert_count == static_cast<uint32_t>(params_.num_experts) &&
            state.top_k == static_cast<uint32_t>(params_.top_k) &&
            state.prefill_token_capacity >= static_cast<uint32_t>(params_.seq_len) &&
            state.prefill_route_capacity >= static_cast<uint32_t>(params_.seq_len * params_.top_k) &&
            state.route_expert_ids &&
            state.route_weights &&
            state.route_participant_ids &&
            state.expert_counts &&
            state.expert_offsets &&
            state.grouped_token_ids &&
            state.grouped_route_weights;
        return moe_runtime_row_grouping_available_;
    }

    bool MoEExpertComputeStage::initializeFixedTopologyGroupedPrefill()
    {
        return false;
    }

    bool MoEExpertComputeStage::runtimeTableHasActiveGroupedDecodeBank() const
    {
        if (!params_.moe_runtime_table ||
            params_.moe_runtime_table->decodeRuntimePublicationRequired(params_.layer_idx))
        {
            return false;
        }

        const DeviceMoEPlacementBank *bank = activeRuntimePlacementBank();
        if (!bank || !moe_runtime_layer_)
            return false;

        const bool mutable_runtime_descriptors =
            usesRuntimePlacementWeightDescriptors();
        const bool runtime_owner_metadata =
            mutable_runtime_descriptors ||
            params_.runtime_decode_has_explicit_owner_metadata;
        const bool has_replicas = params_.replica_set.num_replicated > 0;
        int participant_count = 0;
        try
        {
            const auto &state = params_.moe_runtime_table->hostLayerState(params_.layer_idx);
            if (state.participant_id != static_cast<uint32_t>(params_.my_socket_id))
                return false;
            if (state.expert_count != static_cast<uint32_t>(params_.num_experts) ||
                state.top_k != static_cast<uint32_t>(params_.top_k))
            {
                return false;
            }
            if (has_replicas &&
                (!params_.replica_set.hasLayerReplicaPlacement() ||
                 params_.replica_set.base_ownership.expertCount() != params_.num_experts ||
                 params_.layer_idx < 0 ||
                 params_.layer_idx >= params_.replica_set.base_ownership.layerCount()))
            {
                return false;
            }
            participant_count = expectedGroupedDecodeParticipantCount();
            if (params_.my_socket_id < 0 ||
                params_.my_socket_id >= participant_count ||
                participant_count <= 0 ||
                participant_count > static_cast<int>(kDeviceMoEMaxParticipants) ||
                state.participant_count != static_cast<uint32_t>(participant_count))
            {
                return false;
            }
        }
        catch (...)
        {
            return false;
        }

        const uint32_t valid_mask =
            participant_count >= static_cast<int>(kDeviceMoEMaxParticipants)
                ? ((1u << kDeviceMoEMaxParticipants) - 1u)
                : ((1u << static_cast<uint32_t>(participant_count)) - 1u);
        const uint32_t local_participant_bit =
            1u << static_cast<uint32_t>(params_.my_socket_id);

        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            const uint32_t raw_resident_mask =
                bank->resident_participant_mask[static_cast<size_t>(expert_id)];
            const auto &expert = bank->experts[static_cast<size_t>(expert_id)];
            int owner_participant = expert.owner_participant;
            bool replicated = false;
            uint32_t effective_resident_mask = raw_resident_mask & valid_mask;

            if (runtime_owner_metadata)
            {
                if (owner_participant >= 0 && owner_participant < participant_count)
                    effective_resident_mask |=
                        1u << static_cast<uint32_t>(owner_participant);
                const bool external_to_domain =
                    owner_participant == -1 &&
                    effective_resident_mask == 0u &&
                    bank->local_compute_mask[
                        static_cast<size_t>(expert_id)] == 0u;
                if ((raw_resident_mask & ~valid_mask) != 0u ||
                    (effective_resident_mask == 0u &&
                     !external_to_domain) ||
                    expert.logical_expert_id != expert_id ||
                    owner_participant < -1 ||
                    owner_participant >= participant_count)
                {
                    return false;
                }
                replicated =
                    (effective_resident_mask & (effective_resident_mask - 1u)) != 0u;
            }
            else
            {
                owner_participant = params_.my_socket_id;
                if (has_replicas)
                    owner_participant = params_.replica_set.ownerParticipant(
                        params_.layer_idx, expert_id);
                replicated =
                    has_replicas &&
                    params_.replica_set.isReplicatedForLayer(params_.layer_idx, expert_id);
            }

            const bool expected_local =
                runtime_owner_metadata
                    ? ((effective_resident_mask & local_participant_bit) != 0u)
                    : expertComputesLocally(expert_id);
            if (bank->local_compute_mask[static_cast<size_t>(expert_id)] !=
                (expected_local ? 1u : 0u))
            {
                return false;
            }

            uint32_t expected_resident_mask = 0u;
            if (owner_participant >= 0 && owner_participant < participant_count)
                expected_resident_mask |= (1u << static_cast<uint32_t>(owner_participant));
            // A graph-initialized runtime bank is the sole placement authority
            // when it carries explicit owner metadata.  In that mode the stage
            // may intentionally have no static ReplicaSet at all, so consulting
            // it here would both violate ownership and index an absent geometry.
            // Static banks still reconstruct the expected mask from ReplicaSet
            // and compare it below.
            if (!runtime_owner_metadata && replicated)
            {
                for (int participant = 0; participant < participant_count; ++participant)
                {
                    if (params_.replica_set.hasReplicaOnParticipant(
                            params_.layer_idx,
                            expert_id,
                            participant))
                    {
                        expected_resident_mask |= (1u << static_cast<uint32_t>(participant));
                    }
                }
            }
            if (expected_local)
                expected_resident_mask |= (1u << static_cast<uint32_t>(params_.my_socket_id));
            if (!runtime_owner_metadata && raw_resident_mask != expected_resident_mask)
            {
                return false;
            }

            if (expert.logical_expert_id != expert_id)
                return false;
            if (!runtime_owner_metadata &&
                expert.owner_participant != owner_participant)
            {
                return false;
            }

            if (!mutable_runtime_descriptors)
            {
                if (replicated != hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::Replicated))
                    return false;
                const auto expected_role =
                    replicated
                        ? (owner_participant == params_.my_socket_id
                               ? DeviceMoEReplicaRole::Primary
                               : DeviceMoEReplicaRole::Replica)
                        : (expected_local ? DeviceMoEReplicaRole::Primary : DeviceMoEReplicaRole::None);
                if (bank->replica_role[static_cast<size_t>(expert_id)] !=
                    static_cast<uint8_t>(expected_role))
                {
                    return false;
                }
            }

            if (!expected_local)
                continue;

            if (expert.local_slot < 0 ||
                !expert.weightsReady() ||
                !hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::Valid) ||
                !hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::Resident) ||
                !hasMoEExpertFlag(expert.flags, DeviceMoEExpertFlags::LocalCompute))
            {
                return false;
            }
        }
        return true;
    }

    bool MoEExpertComputeStage::supportsRequestedRoutedAssignmentPolicy() const
    {
        if (params_.routed_row_execution_policy ==
            RoutedExpertRowExecutionPolicy::FullyReplicatedLocal)
        {
            return hasFullLocalExpertOwnership() && expertMaskAllEnabled();
        }

        if (params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::StaticOwner)
            return true;

        if (params_.routed_assignment_policy != RoutedExpertAssignmentPolicy::LeastLoadedResident)
            return false;

        const bool supports_llep_prefill =
            params_.seq_len > 1 &&
            params_.use_runtime_row_grouping &&
            params_.moe_runtime_table &&
            params_.prefill_llep_tp_ctx &&
            supportsGroupedPrefillExecutionBackend(params_.device_id);
        if (supports_llep_prefill)
            return true;

        return params_.seq_len == 1 &&
               !params_.force_grouped_verifier_prefill_for_decode &&
               params_.moe_runtime_table &&
               params_.layer_idx >= 0 &&
               params_.num_experts > 0 &&
               params_.top_k > 0 &&
               supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id);
    }

    bool MoEExpertComputeStage::canUseRuntimeRowGrouping() const
    {
        if (!params_.use_runtime_row_grouping ||
            !moe_runtime_row_grouping_available_ ||
            !params_.moe_runtime_table ||
            !moe_runtime_layer_ ||
            params_.seq_len <= 1)
        {
            return false;
        }
        /*
         * Prefix restore and MTP transaction reset mutate the contents behind
         * this stable runtime-table pointer.  A cached stage-local warmup bit
         * therefore cannot prove that the current table still owns every
         * persistent grouping buffer.  Revalidate the authoritative table on
         * every execution boundary; this is a host metadata check only and
         * performs no allocation, copy, or synchronization in the hot path.
         */
        const auto *runtime_table =
            dynamic_cast<const DeviceMoERuntimeTable *>(params_.moe_runtime_table);
        if (!runtime_table ||
            !runtime_table->hasPrefillRouteScratchCapacity(
                params_.layer_idx,
                params_.seq_len))
        {
            return false;
        }
        if (params_.routed_row_execution_policy ==
            RoutedExpertRowExecutionPolicy::FullyReplicatedLocal)
        {
            return hasFullLocalExpertOwnership() && expertMaskAllEnabled();
        }
        if (params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::StaticOwner)
            return (hasFullLocalExpertOwnership() && expertMaskAllEnabled()) ||
                   hasFixedTopologyPrefillExpertMask();
        if (params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident)
            return hasFixedTopologyPrefillExpertMask();
        return false;
    }

    bool MoEExpertComputeStage::ownsRuntimePrefillHistogramPublication() const
        noexcept
    {
        return params_.device_id.is_gpu() &&
               params_.moe_runtime_table != nullptr &&
               params_.use_runtime_row_grouping &&
               params_.seq_len > 1 &&
               groupedVerifierHistogramRole() ==
                   MoEGroupedVerifierHistogramRole::NotOwner;
    }

    bool MoEExpertComputeStage::canUseFixedTopologyGroupedPrefill() const
    {
        const bool forced_decode_replay =
            params_.force_grouped_verifier_prefill_for_decode && params_.seq_len == 1;
        return supportsGroupedPrefillExecutionBackend(params_.device_id) &&
               (params_.seq_len > 1 || forced_decode_replay) &&
               (!forced_decode_replay || params_.replica_set.num_replicated == 0) &&
               ((hasFullLocalExpertOwnership() && expertMaskAllEnabled()) ||
                hasFixedTopologyPrefillExpertMask());
    }

    bool MoEExpertComputeStage::canUseSafeCombinedSharedVerifierComposite() const
    {
        return params_.combine_shared_expert_in_verifier &&
               (params_.device_id.is_cuda() || params_.device_id.is_rocm()) &&
               params_.seq_len > 1 &&
               params_.seq_len <= 4 &&
               params_.d_model > 0 &&
               params_.expert_intermediate > 0 &&
               params_.num_experts > 0 &&
               params_.top_k > 0 &&
               params_.routing_indices &&
               params_.routing_weights &&
               params_.shared_gate_inp &&
               params_.prepared_store &&
               params_.prepared_shared_ref_gate &&
               params_.prepared_shared_ref_up &&
               params_.prepared_shared_ref_down &&
               hasFullLocalExpertOwnership() &&
               expertMaskAllEnabled() &&
               params_.replica_set.num_replicated == 0;
    }

    TensorBase *MoEExpertComputeStage::effectiveSafeCompositeSharedGateInput() const
    {
        if (!params_.shared_gate_inp)
            return nullptr;
        return params_.shared_gate_inp->native_type() == TensorType::FP32
                   ? params_.shared_gate_inp
                   : nullptr;
    }

    bool MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite(IMoEKernel *kernel) const
    {
        if (!kernel || grouped_gateup_desc_table_id_ < 0 ||
            grouped_down_desc_table_id_ < 0 ||
            combined_shared_gateup_desc_table_id_ < 0 ||
            combined_shared_down_desc_table_id_ < 0 ||
            !combined_shared_gate_gemm_ ||
            !combined_shared_up_gemm_ ||
            !combined_shared_down_gemm_)
        {
            return false;
        }

        TensorBase *shared_gate_inp = effectiveSafeCompositeSharedGateInput();
        if (!shared_gate_inp)
        {
            LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] "
                      "failed to prepare FP32 shared gate input");
            return false;
        }

        const int seq_len = params_.seq_len;
        const int d_model = params_.d_model;
        const int intermediate = params_.expert_intermediate;

        auto ensure_scratch = [&](std::shared_ptr<FP32Tensor> &slot,
                                  size_t rows,
                                  size_t cols,
                                  const char *name) -> bool
        {
            const std::vector<size_t> expected{rows, cols};
            if (slot && slot->shape() == expected)
                return true;
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] "
                          << name << " scratch was not warmed before graph capture"
                          << " layer=" << params_.layer_idx
                          << " rows=" << rows
                          << " cols=" << cols);
                return false;
            }
            slot = makeScratchFP32(rows, cols, params_.device_id);
            return slot != nullptr;
        };

        if (!ensure_scratch(combined_routed_output_,
                            static_cast<size_t>(seq_len),
                            static_cast<size_t>(d_model),
                            "routed_output") ||
            !ensure_scratch(combined_shared_output_,
                            static_cast<size_t>(seq_len),
                            static_cast<size_t>(d_model),
                            "shared_output"))
        {
            return false;
        }

        if (!kernel->prepareExpertGroupsAsync(
                params_.routing_indices,
                params_.routing_weights,
                seq_len,
                params_.num_experts,
                params_.top_k))
        {
            LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] grouping failed");
            return false;
        }

        /*
         * This replaces the rejected "shared expert as one more routed expert"
         * table.  Keep routed and shared math as separate, already-proven branch
         * computations, but let one stage own the lifetime/order so the graph can
         * later overlap them without reintroducing shared backend scratch races.
         */
        if (!kernel->executeGroupedPrefillPipeline(
            params_.input,
            combined_routed_output_.get(),
            grouped_gateup_desc_table_id_,
            grouped_down_desc_table_id_,
            seq_len,
            d_model,
            intermediate,
            params_.num_experts,
            params_.top_k))
        {
            LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] routed grouped verifier failed");
            return false;
        }
        gpuExecution().publish(combined_routed_output_.get());

        /*
         * The standalone shared-expert verifier path targets serial decode's
         * grouped table-decode family by using the sibling grouped table-prefill
         * kernels for M=2..4.  The composite path must use that same production
         * route; otherwise it silently reintroduces the dense GEMM verifier hooks
         * that can differ by a single FP32 ulp in layer 0 and later flip MoE
         * routing decisions.  This is still a grouped implementation: one device
         * grouping setup and one grouped prefill pipeline cover all verifier rows.
         */
        if (!kernel->prepareSharedExpertPrefillGroup(
                params_.active_row_count_device
                    ? DeviceRowRange::deviceCounted(seq_len, params_.active_row_count_device)
                    : DeviceRowRange::fullyActive(seq_len)))
        {
            LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] "
                      "shared grouped verifier setup failed");
            return false;
        }
        if (!kernel->executeGroupedPrefillPipeline(
                params_.input,
                combined_shared_output_.get(),
                combined_shared_gateup_desc_table_id_,
                combined_shared_down_desc_table_id_,
                seq_len,
                d_model,
                intermediate,
                /*num_experts=*/1,
                /*top_k=*/1))
        {
            LOG_ERROR("[MoEExpertComputeStage::executeSafeCombinedSharedVerifierComposite] "
                      "shared grouped verifier failed");
            return false;
        }
        gpuExecution().publish(combined_shared_output_.get());

        kernel->sharedExpertGateAddFromTensors(
            params_.input,
            shared_gate_inp,
            combined_shared_output_.get(),
            combined_routed_output_.get(),
            params_.output,
            seq_len,
            d_model);
        gpuExecution().publish(combined_shared_output_.get());
            if (params_.device_id.is_gpu())
            {
                gpuExecution().publish(
                    params_.canonical_route_contributions
                        ? params_.canonical_route_contributions
                        : params_.output);
            }

        PerfStatsCollector::addCounter(
            "mtp",
            "moe_combined_decode_equivalent_verifier_prefill_rows",
            static_cast<double>(seq_len),
            "verifier",
            params_.device_id.toString(),
            {{"stage", "routed_plus_shared"},
             {"route", "safe_composite_grouped_table_prefill"},
             {"seq_len", std::to_string(seq_len)},
             {"routed_top_k", std::to_string(params_.top_k)},
             {"routed_experts", std::to_string(params_.num_experts)},
             {"layer", std::to_string(params_.layer_idx)}});
        return true;
    }

    bool MoEExpertComputeStage::
        usesGraphPhasedCurrentBatchPrefillLLEP() const noexcept
    {
        return params_.prefill_llep_assignment_mode ==
               PrefillLLEPAssignmentMode::GraphPhasedCurrentBatch;
    }

    bool MoEExpertComputeStage::hasValidCompactLLEPTransferBinding() const
    {
        if (params_.routed_assignment_policy != RoutedExpertAssignmentPolicy::LeastLoadedResident ||
            !params_.prefill_llep_tp_ctx ||
            !params_.moe_runtime_table ||
            !params_.prefill_llep_transfer_slots ||
            params_.prefill_llep_transfer_slot_count == 0 ||
            params_.prefill_llep_payload_slot_bytes == 0 ||
            params_.prefill_llep_payload_slot_capacity == 0 ||
            !params_.prefill_llep_transfer_state ||
            !params_.moe_runtime_table->containsModelLayer(params_.layer_idx))
        {
            return false;
        }

        if (params_.prefill_llep_transfer_mode !=
            DeviceMoERebalanceTransferMode::CompactTransferSlots)
        {
            return false;
        }
        if (!validateDeviceMoERebalanceConfig(params_.prefill_llep_rebalance_config) ||
            params_.prefill_llep_rebalance_config.num_experts !=
                static_cast<uint32_t>(params_.num_experts) ||
            params_.prefill_llep_rebalance_config.top_k !=
                static_cast<uint32_t>(params_.top_k) ||
            params_.moe_runtime_table->layerCount() !=
                static_cast<int>(params_.prefill_llep_rebalance_config.num_layers))
            return false;
        if (params_.prefill_llep_tp_ctx->degree() <= 1 ||
            params_.prefill_llep_tp_ctx->degree() !=
                static_cast<int>(params_.prefill_llep_rebalance_config.participant_count))
        {
            return false;
        }
        return params_.prefill_llep_rebalance_config.participant_id <
               params_.prefill_llep_rebalance_config.participant_count;
    }

    bool MoEExpertComputeStage::
        executePrefixRuntimeRehydrationPayloadMovement(
        IMoEKernel *kernel,
        DeviceMoERebalanceStatus **transfer_status_out,
        DeviceMoERebalanceApplyStatus **apply_status_out,
        DeviceMoERebalanceTransferState *transfer_state_override) const
    {
        if (!kernel)
            return false;
        if (transfer_status_out)
            *transfer_status_out = nullptr;
        if (apply_status_out)
            *apply_status_out = nullptr;
        if (!hasValidCompactLLEPTransferBinding())
        {
            throw std::logic_error(
                "MoE compact LLEP payload movement requires a valid graph-owned "
                "LocalTP transport binding");
        }
        if (!bound_workspace_)
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill requires bound workspace buffers");
            return false;
        }

        void *compute_stream = gpuStream();
        if (!compute_stream)
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill requires an explicit compute stream");
            return false;
        }

        auto *runtime_layers = params_.moe_runtime_table->deviceLayerState(
            params_.moe_runtime_table->firstModelLayer());
        if (!runtime_layers || !moe_runtime_layer_)
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill requires device runtime layers");
            return false;
        }

        /* Prefix rehydration publishes placement and histogram evidence that
         * was already restored from the portable snapshot. This transport must
         * preserve that evidence so the first maintenance decision after a
         * cache hit matches uninterrupted execution. Current-batch movement is
         * owned by MoEGPUCurrentBatchLLEPStage and never enters this method. */
        const DeviceMoERebalanceConfig config =
            prefillLLEPTransferConfig(
                params_.prefill_llep_rebalance_config,
                PrefillLLEPTransferPurpose::PrefixRuntimeRehydration);
        if (tracePrefillLLEPStatusEnabled())
        {
            LOG_INFO("[MoEExpertComputeStage] prefill LLEP transfer config"
                     << " device=" << params_.device_id.to_string()
                     << " layer=" << params_.layer_idx
                     << " participant_id=" << config.participant_id
                     << " participant_count=" << config.participant_count
                     << " root_participant=" << config.root_participant);
        }
        const uint32_t captured_payload_slots =
            std::min<uint32_t>(
                params_.prefill_llep_payload_slot_capacity,
                params_.prefill_llep_transfer_slot_count);
        const uint64_t configured_plan_capacity =
            deviceMoERebalanceCommandPlanCapacity(
                config,
                params_.prefill_llep_transfer_mode);
        const uint32_t plan_capacity =
            std::max<uint32_t>(
                1u,
                boundedU32(deviceMoEPrefillLLEPMergedPlanCapacity(
                    config,
                    configured_plan_capacity,
                    captured_payload_slots)));
        const uint32_t payload_slot_count =
            std::min<uint32_t>(captured_payload_slots, plan_capacity);
        if (payload_slot_count == 0)
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill has zero payload slots");
            return false;
        }

        const std::string &workspace_name = params_.prefill_llep_workspace_name;
        auto *plan_entries = static_cast<DeviceMoERebalancePlanEntry *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_TRANSFER_PLAN,
                workspace_name)));
        auto *plan_count = static_cast<uint32_t *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_TRANSFER_PLAN_COUNT,
                workspace_name)));
        auto *command_header = static_cast<DeviceMoERebalanceCommandBufferHeader *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_COMMAND_HEADER,
                workspace_name)));
        auto *gathered_plan_entries = static_cast<DeviceMoERebalancePlanEntry *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PLAN,
                workspace_name)));
        auto *gathered_command_headers = static_cast<DeviceMoERebalanceCommandBufferHeader *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_GATHERED_COMMAND_HEADER,
                workspace_name)));
        auto *transfer_slot_claim_index =
            static_cast<DeviceMoETransferSlotClaimIndex *>(
                bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                    MoEDeviceRebalanceStage::WS_TRANSFER_SLOT_CLAIM_INDEX,
                    workspace_name)));
        auto *status = static_cast<DeviceMoERebalanceStatus *>(
            bound_workspace_->getBuffer(prefillLLEPLayerPublicationBufferName(
                MoEDeviceRebalanceStage::WS_STATUS,
                workspace_name,
                params_.layer_idx)));
        auto *apply_status = static_cast<DeviceMoERebalanceApplyStatus *>(
            bound_workspace_->getBuffer(prefillLLEPLayerPublicationBufferName(
                MoEDeviceRebalanceStage::WS_APPLY_STATUS,
                workspace_name,
                params_.layer_idx)));
        auto *local_source_descriptors = static_cast<DeviceMoEExpertDirectoryEntry *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_LOCAL_SOURCE_DESCRIPTORS,
                workspace_name)));
        auto *local_payload = static_cast<uint8_t *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_LOCAL_TRANSFER_PAYLOAD,
                workspace_name)));
        auto *gathered_payload = static_cast<uint8_t *>(
            bound_workspace_->getBuffer(prefillLLEPWorkspaceBufferName(
                MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PAYLOAD,
                workspace_name)));

        if (!plan_entries || !plan_count || !command_header ||
            !gathered_plan_entries || !gathered_command_headers ||
            !transfer_slot_claim_index ||
            !status || !apply_status || !local_source_descriptors ||
            !local_payload || !gathered_payload)
        {
            LOG_ERROR("[MoEExpertComputeStage] Missing transfer-backed LLEP prefill workspace buffers"
                      << " plan_entries=" << static_cast<void *>(plan_entries)
                      << " plan_count=" << static_cast<void *>(plan_count)
                      << " command_header=" << static_cast<void *>(command_header)
                      << " gathered_plan_entries=" << static_cast<void *>(gathered_plan_entries)
                      << " gathered_command_headers=" << static_cast<void *>(gathered_command_headers)
                      << " transfer_slot_claim_index="
                      << static_cast<void *>(transfer_slot_claim_index)
                      << " status=" << static_cast<void *>(status)
                      << " apply_status=" << static_cast<void *>(apply_status)
                      << " local_source_descriptors=" << static_cast<void *>(local_source_descriptors)
                      << " local_payload=" << static_cast<void *>(local_payload)
                      << " gathered_payload=" << static_cast<void *>(gathered_payload));
            return false;
        }
        if (transfer_status_out)
            *transfer_status_out = status;
        if (apply_status_out)
            *apply_status_out = apply_status;

        auto *transfer_state =
            transfer_state_override
                ? transfer_state_override
                : params_.prefill_llep_transfer_state.get();
        if (!transfer_state ||
            !transfer_state->isMaterializedFor(
                params_.device_id,
                workspace_name) ||
            !transfer_state->transferStream() ||
            !transfer_state->computeReadyEvent() ||
            !transfer_state->transferDoneEvent())
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill could not initialize transfer stream state");
            return false;
        }

        IWorkerGPUContext *gpu_ctx = nullptr;
        try
        {
            gpu_ctx = &GPUDeviceContextPool::instance().getContext(params_.device_id);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill could not resolve GPU context for "
                      << params_.device_id.to_string() << ": " << e.what());
            return false;
        }
        if (!gpu_ctx)
            return false;

        const MoEKernelLaunchContext compute_launch{
            .stream = compute_stream,
            .workspace = bound_workspace_,
        };
        if (!kernel->materializePrefillLeastLoadedTransferCommands(
                compute_launch,
                moe_runtime_layer_,
                plan_entries,
                plan_count,
                plan_capacity,
                command_header,
                status,
                config,
                payload_slot_count,
                static_cast<uint32_t>(
                    params_.moe_runtime_table->storageIndexForModelLayer(params_.layer_idx)),
                1))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed prefix-runtime rehydration command materialization failed");
            return false;
        }
        IBackend *backend = getBackendFor(params_.device_id);
        if (!tracePrefillLLEPStatus(
                backend,
                params_.device_id,
                compute_stream,
                "after_materialize",
                status,
                apply_status))
        {
            return false;
        }

        void *transfer_stream = transfer_state->transferStream();
        const MoEKernelLaunchContext transfer_launch{
            .stream = transfer_stream,
            .workspace = bound_workspace_,
        };
        if (!gpu_ctx->recordEventChecked(transfer_state->computeReadyEvent(), compute_stream) ||
            !gpu_ctx->waitEventChecked(transfer_state->computeReadyEvent(), transfer_stream))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill failed to queue compute-to-transfer dependency");
            return false;
        }

        /* Prefix rehydration restores participant-owned request lists, so it
         * gathers those lists before destination projection. Ordinary
         * current-batch LLEP derives mirrored commands locally and uses no
         * metadata collective. */
        static_assert((sizeof(DeviceMoERebalancePlanEntry) % sizeof(int32_t)) == 0);
        static_assert((sizeof(DeviceMoERebalanceCommandBufferHeader) % sizeof(int32_t)) == 0);
        const size_t plan_int32_words =
            (static_cast<size_t>(plan_capacity) * sizeof(DeviceMoERebalancePlanEntry)) /
            sizeof(int32_t);
        const size_t header_int32_words =
            sizeof(DeviceMoERebalanceCommandBufferHeader) / sizeof(int32_t);
        if (!params_.prefill_llep_tp_ctx->allgatherRawOnStream(
                plan_entries,
                gathered_plan_entries,
                plan_int32_words,
                CollectiveDataType::INT32,
                static_cast<int>(config.participant_id),
                transfer_stream,
                workspace_name.empty()
                    ? std::string("moe_prefix_runtime_rehydration_transfer_plan")
                    : workspace_name + "_plan"))
        {
            LOG_ERROR("[MoEExpertComputeStage] Prefix-runtime rehydration plan allgather failed");
            return false;
        }
        if (!params_.prefill_llep_tp_ctx->allgatherRawOnStream(
                command_header,
                gathered_command_headers,
                header_int32_words,
                CollectiveDataType::INT32,
                static_cast<int>(config.participant_id),
                transfer_stream,
                workspace_name.empty()
                    ? std::string("moe_prefix_runtime_rehydration_transfer_header")
                    : workspace_name + "_header"))
        {
            LOG_ERROR("[MoEExpertComputeStage] Prefix-runtime rehydration command-header allgather failed");
            return false;
        }
        if (!kernel->projectPrefillLeastLoadedDomainCommands(
                transfer_launch,
                gathered_plan_entries,
                gathered_command_headers,
                plan_capacity,
                plan_entries,
                plan_count,
                command_header,
                config,
                status,
                payload_slot_count,
                runtime_layers,
                params_.prefill_llep_transfer_slots,
                params_.prefill_llep_transfer_slot_count,
                transfer_slot_claim_index,
                1))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill domain command projection failed");
            return false;
        }
        if (!kernel->packDeviceRebalanceSourceDescriptors(
                transfer_launch,
                runtime_layers,
                plan_entries,
                command_header,
                plan_capacity,
                local_source_descriptors,
                config,
                nullptr,
                1))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill source descriptor pack failed");
            return false;
        }
        if (!kernel->packDeviceRebalanceCompactPayloads(
                transfer_launch,
                plan_entries,
                command_header,
                plan_capacity,
                local_source_descriptors,
                local_payload,
                payload_slot_count,
                params_.prefill_llep_payload_slot_bytes,
                config,
                apply_status,
                nullptr,
                1))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill payload pack failed");
            return false;
        }

        if (!tracePrefillLLEPStatus(
                backend,
                params_.device_id,
                transfer_stream,
                "after_pack",
                status,
                apply_status))
        {
            return false;
        }

        const size_t local_payload_bytes =
            static_cast<size_t>(payload_slot_count) *
            static_cast<size_t>(params_.prefill_llep_payload_slot_bytes);
        if (!params_.prefill_llep_tp_ctx->allgatherRawOnStream(
                local_payload,
                gathered_payload,
                local_payload_bytes,
                CollectiveDataType::INT8,
                static_cast<int>(config.participant_id),
                transfer_stream,
                workspace_name.empty()
                    ? std::string("moe_prefill_llep_transfer_payload")
                    : workspace_name + "_payload"))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill payload allgather failed");
            return false;
        }

        if (!kernel->unpackDeviceRebalanceCollectivePayloads(
                transfer_launch,
                plan_entries,
                plan_count,
                plan_capacity,
                command_header,
                gathered_payload,
                payload_slot_count,
                params_.prefill_llep_payload_slot_bytes,
                params_.prefill_llep_transfer_slots,
                params_.prefill_llep_transfer_slot_count,
                config,
                apply_status,
                nullptr,
                1))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill payload unpack failed");
            return false;
        }

        if (!tracePrefillLLEPStatus(
                backend,
                params_.device_id,
                transfer_stream,
                "after_unpack",
                status,
                apply_status))
        {
            return false;
        }

        if (!gpu_ctx->recordEventChecked(transfer_state->transferDoneEvent(), transfer_stream) ||
            !gpu_ctx->waitEventChecked(transfer_state->transferDoneEvent(), compute_stream))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill failed to queue transfer-to-compute dependency");
            return false;
        }

        if (!kernel->applyDeviceRebalanceArrivals(
                compute_launch,
                runtime_layers,
                plan_entries,
                plan_count,
                plan_capacity,
                params_.prefill_llep_transfer_slots,
                params_.prefill_llep_transfer_slot_count,
                config,
                apply_status,
                command_header,
                params_.layer_idx))
        {
            LOG_ERROR("[MoEExpertComputeStage] Transfer-backed LLEP prefill arrival apply failed");
            return false;
        }

        if (!tracePrefillLLEPStatus(
                backend,
                params_.device_id,
                compute_stream,
                "after_apply",
                status,
                apply_status))
        {
            return false;
        }

        return true;
    }

    bool MoEExpertComputeStage::executeFixedTopologyGroupedPrefill(IMoEKernel *kernel, int max_tokens)
    {
        (void)max_tokens;
        if (!kernel)
            return false;

        const int seq_len = params_.seq_len;
        const int num_experts = params_.num_experts;
        const int top_k = params_.top_k;
        const int d_model = params_.d_model;
        const int intermediate = params_.expert_intermediate;
        const MoEKernelLaunchContext compute_launch{
            .stream = gpuStream(),
            .workspace = bound_workspace_,
        };
        if (!compute_launch.hasExplicitStream())
        {
            LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                      "an explicit stage-owned compute stream is required");
            return false;
        }
        if (params_.use_runtime_row_grouping && !canUseRuntimeRowGrouping())
        {
            auto *self = const_cast<MoEExpertComputeStage *>(this);
            if (!self->initializeMoERuntimeTableForGroupedRows())
            {
                LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                          "runtime prefill grouping requested but runtime scratch is unavailable");
                return false;
            }
        }
        const bool runtime_grouping = canUseRuntimeRowGrouping();
        if (ownsRuntimePrefillHistogramPublication())
        {
            if (!runtime_grouping)
            {
                LOG_ERROR(
                    "[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                    "ordinary prefill histogram publication lost its runtime "
                    "grouping authority"
                    << " layer=" << params_.layer_idx);
                return false;
            }
            try
            {
                /* Validation and identity publication only: producer events
                 * were allocated before capture by prepareGraphLaunch(). */
                params_.moe_runtime_table->recordDecodeHistogramProducerStream(
                    compute_launch.stream);
            }
            catch (const std::exception &error)
            {
                LOG_ERROR(
                    "[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                    "ordinary prefill histogram producer was not prepared"
                    << " layer=" << params_.layer_idx
                    << " reason=" << error.what());
                return false;
            }
        }
        uint64_t fixed_topology_trace_sequence = 0;
        IBackend *assignment_trace_backend = nullptr;
        void *assignment_trace_stream = nullptr;
        auto trace_runtime_assignment = [&](const char *tag) -> bool {
            if (!tracePrefillAssignmentEnabled())
                return true;
            if (!params_.device_id.is_gpu())
                return true;
            if (!assignment_trace_backend)
                assignment_trace_backend = getBackendFor(params_.device_id);
            if (!assignment_trace_stream)
                assignment_trace_stream = gpuStream();
            return tracePrefillAssignmentRuntime(
                assignment_trace_backend,
                params_.device_id,
                assignment_trace_stream,
                tag,
                moe_runtime_layer_,
                params_.layer_idx,
                seq_len,
                num_experts,
                top_k);
        };

        // Async grouping (no D2H, no sync). Masked LocalTP overlays exclude
        // non-local experts from this participant's grouping scratch while
        // preserving the original routing tensors for rebalance histograms.
        const bool masked_grouping = usesPublishedFixedTopologyMaskGrouping();
        if (tracePrefillAssignmentEnabled() &&
            tracePrefillAssignmentLayerMatches(params_.layer_idx) &&
            tracePrefillAssignmentCheckpointMatches("before_group"))
        {
            fixed_topology_trace_sequence = nextPrefillAssignmentTraceSequence();
            const std::vector<int> fixed_expert_ids = fixedTopologyPrefillExpertIds();
            const std::vector<uint8_t> fixed_mask = fixedTopologyPrefillExpertMaskBytes();
            uint64_t runtime_placement_hash = 0;
            uint32_t runtime_active_bank = 0;
            uint32_t runtime_active_epoch = 0;
            uint32_t runtime_participant_id = 0;
            uint32_t runtime_participant_count = 0;
            bool runtime_state_available = false;
            if (params_.moe_runtime_table && params_.layer_idx >= 0)
            {
                try
                {
                    const auto &runtime_state =
                        params_.moe_runtime_table->hostLayerState(params_.layer_idx);
                    runtime_placement_hash =
                        hashPlacementBankSemantics(runtime_state, static_cast<uint32_t>(num_experts));
                    runtime_active_bank = runtime_state.active_bank;
                    runtime_active_epoch = runtime_state.active_epoch;
                    runtime_participant_id = runtime_state.participant_id;
                    runtime_participant_count = runtime_state.participant_count;
                    runtime_state_available = true;
                }
                catch (const std::exception &ex)
                {
                    LOG_ERROR("[MoEExpertComputeStage] fixed topology prefill assignment trace "
                              "failed to observe host runtime table"
                              << " device=" << params_.device_id.to_string()
                              << " layer=" << params_.layer_idx
                              << " error=" << ex.what());
                    return false;
                }
            }

            const auto enabled_mask_entries =
                static_cast<uint32_t>(std::count(fixed_mask.begin(), fixed_mask.end(), uint8_t{1}));
            const uint64_t prepared_desc_surface_hash =
                hashPreparedExpertDescriptorSurface(
                    fixed_expert_ids,
                    cached_gate_gemm_,
                    cached_up_gemm_,
                    cached_down_gemm_);
            LOG_INFO("[MoEExpertComputeStage] fixed topology prefill assignment trace"
                     << " seq=" << fixed_topology_trace_sequence
                     << " tag=before_group"
                     << " device=" << params_.device_id.to_string()
                     << " layer=" << params_.layer_idx
                     << " seq_len=" << seq_len
                     << " top_k=" << top_k
                     << " assignment_policy="
                     << routedExpertAssignmentPolicyToString(params_.routed_assignment_policy)
                     << " row_execution_policy="
                     << routedExpertRowExecutionPolicyToString(
                            params_.routed_row_execution_policy)
                     << " runtime_grouping=" << perfBool(runtime_grouping)
                     << " masked_grouping=" << perfBool(masked_grouping)
                     << " requested_runtime_grouping=" << perfBool(params_.use_runtime_row_grouping)
                     << " has_full_local_ownership=" << perfBool(hasFullLocalExpertOwnership())
                     << " expert_mask_all_enabled=" << perfBool(expertMaskAllEnabled())
                     << " participant_id=" << params_.my_socket_id
                     << " participant_count=" << params_.participant_count
                     << " local_start=" << params_.local_expert_start
                     << " local_count=" << params_.local_expert_count
                     << " fixed_expert_count=" << fixed_expert_ids.size()
                     << " enabled_mask_entries=" << enabled_mask_entries
                     << " fixed_expert_ids_hash=" << hashTraceVector(fixed_expert_ids)
                     << " fixed_mask_hash=" << hashTraceVector(fixed_mask)
                     << " prepared_desc_surface_hash=" << prepared_desc_surface_hash
                     << " expert_mask_hash=" << hashTraceBoolVector(params_.expert_mask)
                     << " replica_prefill_mask_hash="
                     << hashTraceBoolVector(params_.replica_set.prefill_mask)
                     << " replica_count=" << params_.replica_set.num_replicated
                     << " runtime_state_available=" << perfBool(runtime_state_available)
                     << " runtime_active_bank=" << runtime_active_bank
                     << " runtime_active_epoch=" << runtime_active_epoch
                     << " runtime_participant_id=" << runtime_participant_id
                     << " runtime_participant_count=" << runtime_participant_count
                     << " runtime_placement_hash=" << runtime_placement_hash
                     << " gateup_desc_table=" << grouped_gateup_desc_table_id_
                     << " down_desc_table=" << grouped_down_desc_table_id_);
        }
        const bool retain_routes_for_deferred_commit =
            groupedVerifierHistogramRole() ==
            MoEGroupedVerifierHistogramRole::DeferredAcceptedRows;
        // Grouped kernels also execute predictor/static work and one-row
        // decode. Absence of a deferred ledger is not proof of prefill.
        const auto immediate_demand = seq_len <= 1 ||
                params_.service_phase == MoEOverlayServicePhaseHint::GroupedVerifier ||
                groupedVerifierHistogramRole() == MoEGroupedVerifierHistogramRole::StaticNoPublication
            ? MoEGroupedPlanDemand::None : MoEGroupedPlanDemand::OrdinaryPrefill;
        const auto grouped_plan_demand = retain_routes_for_deferred_commit
            ? MoEGroupedPlanDemand::DeferredAcceptedRows : immediate_demand;
        if (retain_routes_for_deferred_commit)
        {
            if (!runtime_grouping || !params_.moe_runtime_table ||
                !moe_runtime_layer_)
            {
                LOG_ERROR(
                    "[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                    "deferred verifier publication requires runtime grouping and "
                    "a per-layer device route ledger"
                    << " layer=" << params_.layer_idx);
                return false;
            }
            if (params_.routed_assignment_policy !=
                    RoutedExpertAssignmentPolicy::StaticOwner &&
                params_.routed_assignment_policy !=
                    RoutedExpertAssignmentPolicy::LeastLoadedResident)
            {
                LOG_ERROR(
                    "[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                    "deferred verifier publication has no final-route boundary "
                    "for assignment policy="
                    << routedExpertAssignmentPolicyToString(
                           params_.routed_assignment_policy)
                    << " layer=" << params_.layer_idx);
                return false;
            }
            const auto &runtime_state =
                params_.moe_runtime_table->hostLayerState(params_.layer_idx);
            const uint64_t required_routes =
                static_cast<uint64_t>(seq_len) *
                static_cast<uint64_t>(top_k);
            if (!runtime_state.deferred_verifier_route_expert_ids ||
                !runtime_state.deferred_verifier_route_participant_ids ||
                required_routes >
                    runtime_state.deferred_verifier_route_capacity)
            {
                LOG_ERROR(
                    "[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                    "per-layer deferred verifier route ledger is absent or undersized"
                    << " layer=" << params_.layer_idx
                    << " required_routes=" << required_routes
                    << " capacity="
                    << runtime_state.deferred_verifier_route_capacity);
                return false;
            }
        }

        bool groups_prepared = false;
        if (runtime_grouping)
        {
            const bool fully_replicated_local_rows =
                params_.routed_row_execution_policy ==
                RoutedExpertRowExecutionPolicy::FullyReplicatedLocal;
            const bool filter_runtime_grouping_to_local_experts =
                fully_replicated_local_rows ||
                params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::StaticOwner;
            const bool retain_routes_during_initial_grouping =
                retain_routes_for_deferred_commit &&
                (fully_replicated_local_rows ||
                 params_.routed_assignment_policy ==
                     RoutedExpertAssignmentPolicy::StaticOwner);
            const bool requires_participant_assignment =
                !fully_replicated_local_rows &&
                params_.routed_assignment_policy ==
                    RoutedExpertAssignmentPolicy::LeastLoadedResident;
            if (requires_participant_assignment)
            {
                if (usesGraphPhasedCurrentBatchPrefillLLEP())
                {
                    /* The explicit PlanAndPack and UnpackApplyAndAssign graph
                     * nodes already published the route-only ledger and final
                     * participant ids. Re-grouping here would overwrite that
                     * publication and make the payload collective's result
                     * causally irrelevant. This stage owns only the final
                     * descriptor-table publication below. */
                    groups_prepared = true;
                }
                else
                {
                    /* Resident-only LLEP consumes global route counts before it
                     * can assign participants. This first publication is
                     * deliberately route-only; the final assignment boundary
                     * below publishes the complete grouped execution plan. */
                    groups_prepared = kernel->groupPrefillRoutes(
                        moe_runtime_layer_,
                        params_.routing_indices,
                        params_.routing_weights,
                        seq_len,
                        seq_len,
                        num_experts,
                        top_k,
                        filter_runtime_grouping_to_local_experts,
                        /*retain_routes_for_deferred_commit=*/false);
                }
            }
            else
            {
                groups_prepared =
                    kernel->publishCompleteGroupedPrefillPlanFromRouter(
                        moe_runtime_layer_,
                        params_.routing_indices,
                        params_.routing_weights,
                        seq_len,
                        seq_len,
                        num_experts,
                        top_k,
                        grouped_gateup_desc_table_id_,
                        grouped_down_desc_table_id_,
                        filter_runtime_grouping_to_local_experts,
                        retain_routes_during_initial_grouping
                            ? MoEGroupedPlanDemand::DeferredAcceptedRows : immediate_demand);
            }
            if (groups_prepared && fully_replicated_local_rows)
            {
                /*
                 * A mirrored verifier owns every routed row locally on every
                 * participant.  It deliberately has no participant assignment,
                 * current-batch expert transport, or routed-result collective.
                 * Publish that complete policy as one typed observation so the
                 * serving gate can distinguish this economical first-class mode
                 * from a missing LLEP assignment launch.
                 */
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "fully_replicated_local_verifier_execution_calls",
                    1.0,
                    params_.force_grouped_verifier_prefill_for_decode
                        ? "verifier"
                        : "prefill",
                    params_.device_id.toString(),
                    {{"stage", "moe_expert_grouped_prefill"},
                     {"execution_policy", "fully_replicated_local"},
                     {"participant_assignment", "none"},
                     {"current_batch_transport", "none"},
                     {"routed_result_collective", "none"},
                     {"layer", std::to_string(params_.layer_idx)},
                     {"seq_len", std::to_string(seq_len)},
                     {"top_k", std::to_string(top_k)}});
            }
            if (groups_prepared &&
                params_.force_grouped_verifier_prefill_for_decode &&
                !fully_replicated_local_rows &&
                params_.routed_assignment_policy ==
                    RoutedExpertAssignmentPolicy::StaticOwner)
            {
                /*
                 * Current-batch LLEP is an ordinary-prefill policy. The
                 * economical verifier target keeps whole experts on their
                 * static owners and runs the grouped runtime-table kernel,
                 * with the graph's apportioned routed-expert result
                 * collective combining participant outputs afterward.
                 */
                PerfStatsCollector::addCounter(
                    "moe_routed_execution",
                    "static_owner_grouped_verifier_calls",
                    1.0,
                    "verifier",
                    params_.device_id.toString(),
                    {{"stage", "moe_expert_grouped_rows"},
                     {"execution_policy", "static_owner_grouped"},
                     {"assignment", "static_owner"},
                     {"runtime_grouping", "runtime_table"},
                     {"row_execution_policy", "participant_assigned"},
                     {"current_batch_transport", "none"},
                     {"layer", std::to_string(params_.layer_idx)},
                     {"seq_len", std::to_string(seq_len)},
                     {"top_k", std::to_string(top_k)}});
            }
            if (groups_prepared &&
                !trace_runtime_assignment("after_group"))
            {
                return false;
            }
            if (groups_prepared &&
                !fully_replicated_local_rows &&
                params_.routed_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident)
            {
                if (usesGraphPhasedCurrentBatchPrefillLLEP())
                {
                    /* The preceding explicit apply stage is the only owner of
                     * movement and route assignment. Retain the diagnostic
                     * boundary here, then publish this expert stage's prepared
                     * descriptor table over those already-final assignments. */
                    if (!trace_runtime_assignment("after_llep_transfer_apply"))
                        return false;
                    groups_prepared = true;
                }
                else
                {
                    /*
                     * Resident grouped assignment consumes the exact
                     * device-owned position row used by RoPE. Serial decode
                     * uses the same logical-position tie key, so speculative
                     * histogram publication remains useful maintenance
                     * evidence without changing the numerical partition.
                     *
                     * This is a first-class no-transfer LLEP policy. Prefix
                     * restore may bind transport resources beside it, but no
                     * host position, row replay, or transfer path participates
                     * in the assignment.
                     */
                    groups_prepared =
                        kernel->assignPrefillRoutesLeastLoadedResident(
                            compute_launch,
                            moe_runtime_layer_,
                            seq_len,
                            seq_len,
                            num_experts,
                            top_k,
                            params_.absolute_position_ids_device,
                            params_.active_row_count_device);
                    if (groups_prepared)
                    {
                        PerfStatsCollector::addCounter(
                            "moe_rebalance",
                            "device_rebalance_llep_resident_assignment_calls",
                            1.0,
                            params_.force_grouped_verifier_prefill_for_decode
                                ? "verifier"
                                : "prefill",
                            params_.device_id.toString(),
                            {{"stage", "moe_expert_grouped_prefill"},
                             {"assignment", "logical_position_resident"},
                             {"current_batch_transport", "none"},
                             {"layer", std::to_string(params_.layer_idx)},
                             {"seq_len", std::to_string(seq_len)},
                             {"top_k", std::to_string(top_k)}});
                    }
                }
                if (!groups_prepared)
                {
                    LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                              "least-loaded current-batch route assignment failed");
                    return false;
                }
                if (!trace_runtime_assignment("after_llep_assign"))
                    return false;
                groups_prepared =
                    kernel->publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
                        moe_runtime_layer_,
                        seq_len,
                        seq_len,
                        num_experts,
                        top_k,
                        grouped_gateup_desc_table_id_,
                        grouped_down_desc_table_id_,
                        grouped_plan_demand);
                if (groups_prepared &&
                    !trace_runtime_assignment("after_llep_regroup"))
                {
                    return false;
                }
            }
        }
        else if (masked_grouping)
        {
            if (!publishFixedTopologyMaskBeforeCapture(kernel))
                return false;
            groups_prepared = kernel->prepareExpertGroupsAsyncUsingPublishedMask(
                params_.routing_indices, params_.routing_weights,
                seq_len, num_experts, top_k);
        }
        // Static all-route grouping is submitted with its complete projection
        // below, so a backend can choose the directory from the original call.
        if ((runtime_grouping || masked_grouping) && !groups_prepared)
        {
            const char *grouping_name = runtime_grouping
                                            ? "groupPrefillRoutes"
                                            : (masked_grouping ? "prepareExpertGroupsAsyncUsingPublishedMask"
                                                               : "prepareExpertGroupsAsync");
            LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                      << grouping_name << " failed");
            return false;
        }

        if (retain_routes_for_deferred_commit)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "moe_grouped_verifier_route_ledger_fused_publications",
                1.0,
                "verifier",
                params_.device_id.toString(),
                {{"layer", std::to_string(params_.layer_idx)},
                 {"rows", std::to_string(seq_len)},
                 {"top_k", std::to_string(top_k)}});
        }

        // Submit the complete pipeline on the original stream without a host wait.
        bool pipeline_ok = false;
        if (runtime_grouping)
        {
            if (!trace_runtime_assignment("before_runtime_pipeline"))
                return false;
            const auto &runtime_state = params_.moe_runtime_table->hostLayerState(params_.layer_idx);
            pipeline_ok = kernel->executeGroupedPrefillPipelineFromPublishedRuntimePlan(
                moe_runtime_layer_,
                runtime_state,
                params_.input,
                params_.output,
                grouped_gateup_desc_table_id_,
                grouped_down_desc_table_id_,
                seq_len,
                d_model,
                intermediate,
                num_experts,
                top_k,
                params_.canonical_route_contributions);
        }
        else if (masked_grouping)
        {
            pipeline_ok = kernel->executeGroupedPrefillPipeline(
                params_.input, params_.output,
                grouped_gateup_desc_table_id_,
                grouped_down_desc_table_id_,
                seq_len, d_model, intermediate,
                num_experts, top_k,
                params_.canonical_route_contributions);
        }
        else
        {
            pipeline_ok = kernel->executeStaticRoutedPrefillTransaction(
                compute_launch, params_.routing_indices, params_.routing_weights,
                params_.input, params_.output, grouped_gateup_desc_table_id_, grouped_down_desc_table_id_,
                seq_len, d_model, intermediate, num_experts, top_k, params_.canonical_route_contributions);
        }

        if (!pipeline_ok)
        {
            LOG_ERROR("[MoEExpertComputeStage::executeFixedTopologyGroupedPrefill] "
                      "grouped prefill pipeline failed");
            return false;
        }

        /*
         * `after_pipeline` is the only checkpoint that can attribute an
         * asynchronous grouped GEMM fault to the layer that launched it.
         * The diagnostic is entirely absent unless assignment tracing is
         * enabled, and tracePrefillAssignmentRuntime() defers it during native
         * graph capture so production capture topology remains unchanged.
         */
        if (!trace_runtime_assignment("after_pipeline"))
            return false;

        if (fixed_topology_trace_sequence != 0 &&
            tracePrefillAssignmentCheckpointMatches("after_pipeline"))
        {
            LOG_INFO("[MoEExpertComputeStage] fixed topology prefill assignment trace"
                     << " seq=" << fixed_topology_trace_sequence
                     << " tag=after_pipeline"
                     << " device=" << params_.device_id.to_string()
                     << " layer=" << params_.layer_idx
                     << " seq_len=" << seq_len
                     << " runtime_grouping=" << perfBool(runtime_grouping)
                     << " masked_grouping=" << perfBool(masked_grouping)
                     << " gateup_desc_table=" << grouped_gateup_desc_table_id_
                     << " down_desc_table=" << grouped_down_desc_table_id_);
        }

        return true;
    }

    bool MoEExpertComputeStage::
        supportsDeviceRoutedOneRowGraphCapturePreflight() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        return supportsDeviceRoutedDecodeGraphCaptureBackend(params_.device_id) &&
               !params_.force_grouped_verifier_prefill_for_decode &&
               !params_.require_device_routing_tensor_decode &&
               params_.seq_len == 1 &&
               params_.d_model > 0 &&
               params_.expert_intermediate > 0 &&
               params_.num_experts > 0 &&
               params_.top_k > 0 &&
               params_.top_k <= 16 &&
               params_.top_k <= params_.num_experts &&
               params_.input &&
               params_.output &&
               params_.moe_runtime_table &&
               params_.layer_idx >= 0;
#endif
    }

    bool MoEExpertComputeStage::isDeviceRoutedDecodeGraphCapturable() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        return supportsDeviceRoutedOneRowGraphCapturePreflight() &&
               moe_kernel_ &&
               runtime_grouped_decode_launch_state_prepared_ &&
               moe_runtime_layer_ &&
               moe_runtime_table_initialized_ &&
               runtimeTableHasActiveGroupedDecodeBank();
#endif
    }

    bool MoEExpertComputeStage::supportsExplicitRoutingDecodeGraphCapturePreflight() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        if (params_.force_grouped_verifier_prefill_for_decode ||
            params_.seq_len != 1 ||
            !params_.require_device_routing_tensor_decode ||
            !supportsDeviceRoutingTensorDecodeExecutionBackend(params_.device_id) ||
            params_.d_model <= 0 ||
            params_.expert_intermediate <= 0 ||
            params_.num_experts <= 0 ||
            params_.top_k <= 0 ||
            params_.top_k > 16 ||
            params_.top_k > params_.num_experts ||
            !params_.input ||
            !params_.output ||
            !params_.routing_indices ||
            !params_.routing_weights ||
            params_.replica_set.num_replicated != 0 ||
            (usesRuntimePlacementWeightDescriptors() &&
             (!params_.moe_runtime_table || params_.layer_idx < 0)) ||
            (!params_.expert_mask.empty() &&
             params_.expert_mask.size() !=
                 static_cast<size_t>(params_.num_experts)))
        {
            return false;
        }

        return hasPreparedExpertGemmEnginesForExperts(
            fixedTopologyPrefillExpertIds());
#endif
    }

    bool MoEExpertComputeStage::isExplicitRoutingDecodeGraphCapturable() const
    {
        if (!supportsExplicitRoutingDecodeGraphCapturePreflight() ||
            !moe_kernel_ ||
            !runtime_grouped_decode_launch_state_prepared_)
        {
            return false;
        }

        if (usesPublishedFixedTopologyMaskGrouping() &&
            fixed_topology_mask_publication_state_ !=
                FixedTopologyMaskPublicationState::Published)
        {
            return false;
        }

        return grouped_gateup_desc_table_id_ >= 0 &&
               grouped_down_desc_table_id_ >= 0 &&
               !grouped_gateup_desc_table_dirty_ &&
               !grouped_down_desc_table_dirty_;
    }

    bool MoEExpertComputeStage::prepareExplicitRoutingDecodeLaunchState(
        IMoEKernel *kernel)
    {
        if (!kernel ||
            !supportsExplicitRoutingDecodeGraphCapturePreflight() ||
            isGraphCaptureActive())
        {
            return false;
        }

        const std::vector<int> local_expert_ids =
            fixedTopologyPrefillExpertIds();
        if (local_expert_ids.empty() ||
            !ensureGemmEnginesForExperts(local_expert_ids) ||
            !ensureGroupedGateUpDescriptorTable(
                kernel, params_.d_model, params_.expert_intermediate) ||
            !ensureGroupedDownDescriptorTable(
                kernel, params_.d_model, params_.expert_intermediate) ||
            !publishFixedTopologyMaskBeforeCapture(kernel) ||
            !kernel->prepareGroupedRuntimeDecodeLaunchState(
                grouped_gateup_desc_table_id_,
                grouped_down_desc_table_id_,
                params_.top_k,
                params_.d_model,
                params_.expert_intermediate,
                params_.weight_descriptor_source))
        {
            runtime_grouped_decode_launch_state_prepared_ = false;
            LOG_ERROR("[MoEExpertComputeStage] Failed to prepare fused "
                      "explicit-routing grouped-decode launch state"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.to_string());
            return false;
        }

        runtime_grouped_decode_launch_state_prepared_ = true;
        return true;
    }

    bool MoEExpertComputeStage::supportsFixedTopologyPrefillGraphCapturePreflight() const
    {
        // Cold preflight validates the fixed-topology grouped prefill contract
        // without requiring lazy warmup resources such as the MoE kernel or its
        // grouping scratch. Those are checked by isGraphCapturable() before the
        // actual capture pass begins.
        const bool forced_decode_replay =
            params_.force_grouped_verifier_prefill_for_decode && params_.seq_len == 1;
        if (!supportsGroupedPrefillGraphCaptureBackend(params_.device_id) ||
            (params_.seq_len <= 1 && !forced_decode_replay))
            return false;

        // Require either full local expert ownership or a fixed local expert
        // mask whose non-local routes can be dropped before grouping.
        if (!((hasFullLocalExpertOwnership() && expertMaskAllEnabled()) ||
              hasFixedTopologyPrefillExpertMask()))
            return false;
        if (forced_decode_replay && params_.replica_set.num_replicated != 0)
            return false;

        if (params_.d_model <= 0 ||
            params_.expert_intermediate <= 0 ||
            params_.num_experts <= 0 ||
            params_.top_k <= 0 ||
            params_.top_k > params_.num_experts ||
            !params_.input ||
            !params_.output ||
            !params_.routing_indices ||
            !params_.routing_weights)
            return false;

        // Must have prepared GEMM engines for every expert this participant
        // can actually compute. Descriptor tables tolerate holes for masked
        // experts because masked grouping never schedules those rows.
        return hasPreparedExpertGemmEnginesForExperts(fixedTopologyPrefillExpertIds());
    }

    bool MoEExpertComputeStage::isFixedTopologyPrefillGraphCapturable() const
    {
        if (!supportsFixedTopologyPrefillGraphCapturePreflight())
            return false;

        // MoE kernel must exist and have pre-allocated grouping + scratch
        if (!moe_kernel_)
            return false;
        if (params_.use_runtime_row_grouping && !canUseRuntimeRowGrouping())
            return false;
        if (usesPublishedFixedTopologyMaskGrouping() &&
            fixed_topology_mask_publication_state_ !=
                FixedTopologyMaskPublicationState::Published)
        {
            return false;
        }

        return true;
    }

    bool MoEExpertComputeStage::expertComputesLocally(int expert_id) const
    {
        if (expert_id < 0 || expert_id >= params_.num_experts)
            return false;

        if (!params_.expert_mask.empty())
        {
            return params_.expert_mask.size() == static_cast<size_t>(params_.num_experts) &&
                   params_.expert_mask[static_cast<size_t>(expert_id)];
        }

        const int local_count =
            params_.local_expert_count < 0 ? params_.num_experts : params_.local_expert_count;
        return expert_id >= params_.local_expert_start &&
               expert_id < params_.local_expert_start + local_count;
    }

    const std::vector<bool> *MoEExpertComputeStage::fixedTopologyPrefillMask() const
    {
        if (params_.num_experts <= 0)
            return nullptr;
        if (params_.replica_set.num_replicated > 0 &&
            params_.replica_set.prefill_mask.size() == static_cast<size_t>(params_.num_experts))
        {
            return &params_.replica_set.prefill_mask;
        }
        if (params_.expert_mask.size() == static_cast<size_t>(params_.num_experts))
            return &params_.expert_mask;
        return nullptr;
    }

    bool MoEExpertComputeStage::hasFixedTopologyPrefillExpertMask() const
    {
        const std::vector<bool> *mask = fixedTopologyPrefillMask();
        if (mask)
        {
            return std::any_of(
                mask->begin(),
                mask->end(),
                [](bool enabled)
                {
                    return enabled;
                });
        }

        /*
         * Static contiguous ownership is already a complete fixed topology.
         * Synthesize its device mask instead of requiring graph construction to
         * duplicate the same range into expert_mask.
         */
        const int local_count =
            params_.local_expert_count < 0
                ? params_.num_experts
                : params_.local_expert_count;
        return params_.local_expert_start >= 0 &&
               local_count > 0 &&
               params_.local_expert_start + local_count <= params_.num_experts &&
               !(params_.local_expert_start == 0 &&
                 local_count == params_.num_experts);
    }

    bool MoEExpertComputeStage::usesMaskedFixedTopologyPrefill() const
    {
        return hasFixedTopologyPrefillExpertMask() &&
               !(hasFullLocalExpertOwnership() && expertMaskAllEnabled());
    }

    bool MoEExpertComputeStage::usesPublishedFixedTopologyMaskGrouping() const
    {
        /*
         * `use_runtime_row_grouping` is a required route, not a preference:
         * executeFixedTopologyGroupedPrefill() hard-fails if its persistent
         * runtime table or scratch is unavailable.  Therefore it is both safe
         * and necessary to choose the placement source from the requested
         * policy rather than from transient warmup readiness.  Falling through
         * to the fixed mask when runtime initialization fails would hide a
         * broken device-resident LLEP contract.
         */
        return !params_.use_runtime_row_grouping &&
               usesMaskedFixedTopologyPrefill();
    }

    std::vector<int> MoEExpertComputeStage::fixedTopologyPrefillExpertIds() const
    {
        std::vector<int> expert_ids;
        if (const std::vector<bool> *mask = fixedTopologyPrefillMask())
        {
            expert_ids.reserve(static_cast<size_t>(params_.num_experts));
            for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
            {
                if ((*mask)[static_cast<size_t>(expert_id)])
                    expert_ids.push_back(expert_id);
            }
            return expert_ids;
        }

        const int local_count =
            params_.local_expert_count < 0
                ? params_.num_experts
                : params_.local_expert_count;
        const int first =
            local_count == params_.num_experts ? 0 : params_.local_expert_start;
        expert_ids.resize(static_cast<size_t>(local_count));
        std::iota(expert_ids.begin(), expert_ids.end(), first);
        return expert_ids;
    }

    std::vector<uint8_t> MoEExpertComputeStage::fixedTopologyPrefillExpertMaskBytes() const
    {
        std::vector<uint8_t> mask(static_cast<size_t>(params_.num_experts), 0u);
        const std::vector<bool> *prefill_mask = fixedTopologyPrefillMask();
        if (!prefill_mask)
        {
            const int local_count =
                params_.local_expert_count < 0
                    ? params_.num_experts
                    : params_.local_expert_count;
            const int first =
                local_count == params_.num_experts ? 0 : params_.local_expert_start;
            const int last = std::min(params_.num_experts, first + local_count);
            for (int expert_id = std::max(0, first);
                 expert_id < last;
                 ++expert_id)
            {
                mask[static_cast<size_t>(expert_id)] = 1u;
            }
            return mask;
        }

        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            mask[static_cast<size_t>(expert_id)] =
                (*prefill_mask)[static_cast<size_t>(expert_id)] ? 1u : 0u;
        }
        return mask;
    }

    bool MoEExpertComputeStage::hasFullLocalExpertOwnership() const
    {
        const int local_count = params_.local_expert_count < 0 ? params_.num_experts : params_.local_expert_count;
        return params_.local_expert_start == 0 && local_count == params_.num_experts;
    }

    bool MoEExpertComputeStage::expertMaskAllEnabled() const
    {
        return params_.expert_mask.empty() ||
               (params_.expert_mask.size() == static_cast<size_t>(params_.num_experts) &&
                std::all_of(params_.expert_mask.begin(), params_.expert_mask.end(),
                            [](bool enabled)
                            { return enabled; }));
    }

    bool MoEExpertComputeStage::hasAllPreparedExpertGemmEngines() const
    {
        if (params_.prepared_gate_gemm.size() != static_cast<size_t>(params_.num_experts) ||
            params_.prepared_up_gemm.size() != static_cast<size_t>(params_.num_experts) ||
            params_.prepared_down_gemm.size() != static_cast<size_t>(params_.num_experts))
        {
            return false;
        }
        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            if (!params_.prepared_gate_gemm[static_cast<size_t>(expert_id)] ||
                !params_.prepared_up_gemm[static_cast<size_t>(expert_id)] ||
                !params_.prepared_down_gemm[static_cast<size_t>(expert_id)])
            {
                return false;
            }
        }
        return true;
    }

    bool MoEExpertComputeStage::hasPreparedExpertGemmEnginesForLocalOwnership() const
    {
        if (cpu_joint_invocation_layer_)
            return cpu_joint_invocation_layer_has_local_;
        if (params_.prepared_gate_gemm.size() != static_cast<size_t>(params_.num_experts) ||
            params_.prepared_up_gemm.size() != static_cast<size_t>(params_.num_experts) ||
            params_.prepared_down_gemm.size() != static_cast<size_t>(params_.num_experts))
        {
            return false;
        }

        bool has_local_expert = false;
        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            if (!expertComputesLocally(expert_id))
                continue;

            has_local_expert = true;
            if (!params_.prepared_gate_gemm[static_cast<size_t>(expert_id)] ||
                !params_.prepared_up_gemm[static_cast<size_t>(expert_id)] ||
                !params_.prepared_down_gemm[static_cast<size_t>(expert_id)])
            {
                return false;
            }
        }

        /*
         * An empty ownership set is not a usable weight source for this stage.
         * Device participants that legitimately own no experts are represented
         * by a graph-level no-op stage rather than by pretending an empty
         * prepared table is complete.
         */
        return has_local_expert;
    }

    bool MoEExpertComputeStage::hasPreparedExpertGemmEnginesForExperts(
        const std::vector<int> &expert_ids) const
    {
        if (expert_ids.empty())
            return false;
        if (params_.prepared_gate_gemm.size() != static_cast<size_t>(params_.num_experts) ||
            params_.prepared_up_gemm.size() != static_cast<size_t>(params_.num_experts) ||
            params_.prepared_down_gemm.size() != static_cast<size_t>(params_.num_experts))
        {
            return false;
        }

        for (int expert_id : expert_ids)
        {
            if (expert_id < 0 || expert_id >= params_.num_experts)
                return false;
            if (!params_.prepared_gate_gemm[static_cast<size_t>(expert_id)] ||
                !params_.prepared_up_gemm[static_cast<size_t>(expert_id)] ||
                !params_.prepared_down_gemm[static_cast<size_t>(expert_id)])
            {
                return false;
            }
        }
        return true;
    }

    bool MoEExpertComputeStage::hasGroupedDecodeDescriptorExportSupport() const
    {
        if (!hasAllPreparedExpertGemmEngines())
            return false;

        std::optional<DeviceMoEWeightFormat> layer_format;
        for (int expert_id = 0; expert_id < params_.num_experts; ++expert_id)
        {
            DeviceMoEExpertDescriptor desc;
            if (!exportDeviceMoEExpertWeightDescriptors(
                    params_.prepared_gate_gemm[static_cast<size_t>(expert_id)],
                    params_.prepared_up_gemm[static_cast<size_t>(expert_id)],
                    params_.prepared_down_gemm[static_cast<size_t>(expert_id)],
                    params_.d_model,
                    params_.expert_intermediate,
                    desc))
            {
                return false;
            }
            if (layer_format && *layer_format != desc.weight_format)
                return false;
            layer_format = desc.weight_format;
        }
        return layer_format.has_value();
    }

    const DeviceMoEPlacementBank *MoEExpertComputeStage::activeRuntimePlacementBank() const
    {
        if (!params_.moe_runtime_table || params_.layer_idx < 0 || params_.num_experts <= 0)
            return nullptr;
        if (params_.moe_runtime_table->decodeRuntimePublicationRequired(params_.layer_idx))
            return nullptr;

        const auto &state = params_.moe_runtime_table->hostLayerState(params_.layer_idx);
        if (state.active_bank > 1 ||
            state.active_epoch == 0 ||
            state.expert_count != static_cast<uint32_t>(params_.num_experts) ||
            state.top_k != static_cast<uint32_t>(params_.top_k))
        {
            return nullptr;
        }

        const auto &bank = state.banks[state.active_bank];
        if (bank.epoch != state.active_epoch ||
            bank.expert_count != static_cast<uint32_t>(params_.num_experts))
        {
            return nullptr;
        }
        return &bank;
    }

    bool MoEExpertComputeStage::runtimeLocalComputeEnabled(const DeviceMoEPlacementBank *bank, int expert_id) const
    {
        if (!bank || expert_id < 0 || expert_id >= params_.num_experts)
            return false;
        return bank->local_compute_mask[static_cast<size_t>(expert_id)] != 0u;
    }

    // =========================================================================
    // MoEExpertComputeStage::extractExpertViews — Delegates to MoEExpertWeightService
    // =========================================================================

    bool MoEExpertComputeStage::extractExpertViews(Params &params)
    {
        if (params.device_id.is_cpu() && params.prepared_store)
        {
            // Model CPU population has already committed exact final engines.
            // Graph wiring borrows that bank; a raw parent (including a cached
            // GGUF view) can never trigger another allocation/preparation path.
            if (!params.prepared_store->physicalMemoryAuthority() || !params.expert_registry ||
                params.num_experts <= 0 || params.d_model <= 0 || params.expert_intermediate <= 0 || params.layer_idx < 0 ||
                (!params.expert_mask.empty() && params.expert_mask.size() != size_t(params.num_experts)))
                throw std::invalid_argument("CPU model experts require their original admitted complete registry and geometry");
            const int first = params.local_expert_start;
            const int count = params.local_expert_count < 0 ? params.num_experts : params.local_expert_count;
            if (first < 0 || first > params.num_experts || count < 0 || count > params.num_experts-first)
                throw std::invalid_argument("CPU model expert bank has a foreign original ownership interval");
            std::vector<ITensorGemm *> gate(size_t(params.num_experts),nullptr), up(gate), down(gate);
            std::vector<std::shared_ptr<ITensorGemm>> owners;
            for (int expert = 0; expert < params.num_experts; ++expert)
            {
                const bool active = !params.expert_mask.empty() ? params.expert_mask[size_t(expert)] :
                    expert >= first && expert < first+count;
                if (!active) continue;
                for (const auto role : {ExpertGemmRegistry::WeightRole::GATE,ExpertGemmRegistry::WeightRole::UP,
                                       ExpertGemmRegistry::WeightRole::DOWN})
                {
                    auto owner = params.expert_registry->getEngineLifetime(params.device_id,params.layer_idx,expert,role);
                    const bool is_down = role == ExpertGemmRegistry::WeightRole::DOWN;
                    if (!owner || !owner->weights_converted() ||
                        owner->get_n() != (is_down ? params.d_model : params.expert_intermediate) ||
                        owner->get_k() != (is_down ? params.expert_intermediate : params.d_model))
                        throw std::invalid_argument("CPU model expert registry is missing or replaced an original prepared projection");
                    auto &target = role == ExpertGemmRegistry::WeightRole::GATE ? gate :
                        role == ExpertGemmRegistry::WeightRole::UP ? up : down;
                    target[size_t(expert)] = owner.get();
                    owners.push_back(std::move(owner));
                }
            }
            if ((!params.prepared_gate_gemm.empty() && params.prepared_gate_gemm != gate) ||
                (!params.prepared_up_gemm.empty() && params.prepared_up_gemm != up) ||
                (!params.prepared_down_gemm.empty() && params.prepared_down_gemm != down))
                throw std::invalid_argument("CPU model expert parameters replaced the original prepared registry bank");
            if (!params.moe_owned_kernels.empty()) {
                if (params.moe_owned_kernels.size() != owners.size())
                    throw std::invalid_argument("CPU model expert parameters changed original bank ownership");
                for (size_t index = 0; index < owners.size(); ++index)
                    if (params.moe_owned_kernels[index].get() != owners[index].get() ||
                        params.moe_owned_kernels[index].owner_before(owners[index]) ||
                        owners[index].owner_before(params.moe_owned_kernels[index]))
                        throw std::invalid_argument("CPU model expert parameters borrowed another prepared owner");
            }
            params.prepared_gate_gemm = std::move(gate);
            params.prepared_up_gemm = std::move(up);
            params.prepared_down_gemm = std::move(down);
            params.moe_owned_kernels = std::move(owners);
            params.expert_gate_views.clear(); params.expert_up_views.clear(); params.expert_down_views.clear();
            params.gate_exps = nullptr; params.up_exps = nullptr; params.down_exps = nullptr;
            params.expert_weight_resolution_policy = MoEExpertWeightResolutionPolicy::PreparedRegistryOnly;
            return true;
        }
        MoEWeightContext ctx{
            params.device_id,
            params.num_experts,
            params.expert_intermediate,
            params.d_model,
            params.local_expert_start,
            params.local_expert_count,
            params.layer_idx,
            params.expert_mask,
            params.gate_exps,
            params.up_exps,
            params.down_exps,
            params.expert_gate_views,
            params.expert_up_views,
            params.expert_down_views,
            params.prepared_gate_gemm,
            params.prepared_up_gemm,
            params.prepared_down_gemm,
            params.moe_owned_kernels,
            params.moe_packed_gate_lifetime,
            params.moe_packed_up_lifetime,
            params.moe_packed_down_lifetime,
            nullptr,
            nullptr,
            nullptr,
            {},
            {},
            {},
            true,
            nullptr,
            CPUExpertNUMAPlacement::forDevice(params.device_id),
            params.expert_weight_resolution_policy,
            params.persistent_weight_owner};
        return MoEExpertWeightService::extractExpertViews(ctx);
    }

    bool MoEExpertComputeStage::prepareExpertGemmEngines(Params &params)
    {
        if (params.device_id.is_cpu() && params.expert_weight_resolution_policy == MoEExpertWeightResolutionPolicy::PreparedRegistryOnly)
        {
            if (!params.prepared_store)
                throw std::invalid_argument("CPU prepared-registry preparation lost its original model store");
            return extractExpertViews(params); // Authenticate only; never extract/read/pack a source parent.
        }
        if (params.device_id.is_cpu() &&
            params.expert_weight_resolution_policy == MoEExpertWeightResolutionPolicy::RawOrPrepared &&
            (!params.prepared_store || !params.prepared_store->physicalMemoryAuthority()))
            throw std::logic_error("CPU MoE preparation requires physical-memory admission");
        MoEWeightContext ctx{
            params.device_id,
            params.num_experts,
            params.expert_intermediate,
            params.d_model,
            params.local_expert_start,
            params.local_expert_count,
            params.layer_idx,
            params.expert_mask,
            params.gate_exps,
            params.up_exps,
            params.down_exps,
            params.expert_gate_views,
            params.expert_up_views,
            params.expert_down_views,
            params.prepared_gate_gemm,
            params.prepared_up_gemm,
            params.prepared_down_gemm,
            params.moe_owned_kernels,
            params.moe_packed_gate_lifetime,
            params.moe_packed_up_lifetime,
            params.moe_packed_down_lifetime,
            nullptr,
            params.prepared_store,
            params.expert_registry,
            params.gate_slab_ref,
            params.up_slab_ref,
            params.down_slab_ref,
            true,
            nullptr,
            CPUExpertNUMAPlacement::forDevice(params.device_id),
            params.expert_weight_resolution_policy,
            params.persistent_weight_owner};
        bool ok = MoEExpertWeightService::prepareGemmEngines(ctx);
        // Phase C: Copy slab refs back to params for rebalance reuse
        params.gate_slab_ref = ctx.gate_slab_ref;
        params.up_slab_ref = ctx.up_slab_ref;
        params.down_slab_ref = ctx.down_slab_ref;
        return ok;
    }

    size_t MoEExpertComputeStage::estimatedFlops() const
    {
        // Per token: top_k experts × (gate + up + down projections)
        // gate/up: d_model × intermediate
        // down: intermediate × d_model
        size_t per_expert = static_cast<size_t>(6) * params_.d_model * params_.expert_intermediate;
        return static_cast<size_t>(params_.seq_len) * params_.top_k * per_expert;
    }

    bool MoEExpertComputeStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return !params_.expert_gate_views.empty() ||
                   (params_.expert_weight_resolution_policy ==
                        MoEExpertWeightResolutionPolicy::PreparedRegistryOnly &&
                    hasPreparedExpertGemmEnginesForLocalOwnership());
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return !params_.expert_gate_views.empty() ||
                   (params_.expert_weight_resolution_policy ==
                        MoEExpertWeightResolutionPolicy::PreparedRegistryOnly &&
                    hasPreparedExpertGemmEnginesForLocalOwnership());
#endif
        default:
            return false;
        }
    }

    bool MoEExpertComputeStage::isCollectiveStage() const
    {
        /*
         * Report the graph-build policy, not whether every pointer in the
         * binding happens to validate. An invalid transport request must remain
         * a collective-stage construction error; classifying it as ordinary
         * compute would permit segmented or unordered execution before the
         * invariant is diagnosed.
         */
        return params_.prefix_runtime_device_rehydration;
    }

    bool MoEExpertComputeStage::isGraphCapturable() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        if (params_.force_grouped_verifier_prefill_for_decode)
            return isFixedTopologyPrefillGraphCapturable();

        // Device-routed grouped decode path: after warmup, all descriptor tables
        // are built and execution is pure kernel launches reading routing info
        // from the device-resident MoE runtime table.
        if (isDeviceRoutedDecodeGraphCapturable())
            return true;

        // Heterogeneous followers consume explicit participant-local routing
        // tensors. Dynamic followers additionally filter those tensors through
        // their live epoch-pinned runtime placement table.
        if (isExplicitRoutingDecodeGraphCapturable())
            return true;

        // Fixed-topology grouped prefill path
        return isFixedTopologyPrefillGraphCapturable();
#endif
    }

    std::string MoEExpertComputeStage::graphCaptureReadinessDebugString() const
    {
        const char *mask_publication = "not_required";
        switch (fixed_topology_mask_publication_state_)
        {
        case FixedTopologyMaskPublicationState::NotRequired:
            break;
        case FixedTopologyMaskPublicationState::NeedsPublication:
            mask_publication = "needs_publication";
            break;
        case FixedTopologyMaskPublicationState::Published:
            mask_publication = "published";
            break;
        }

        const bool runtime_row_grouping_requested =
            params_.use_runtime_row_grouping;
        const bool runtime_row_grouping_ready =
            canUseRuntimeRowGrouping();
        const bool fixed_preflight =
            supportsFixedTopologyPrefillGraphCapturePreflight();
        const bool runtime_decode_ready =
            isDeviceRoutedDecodeGraphCapturable();
        const bool explicit_decode_preflight =
            supportsExplicitRoutingDecodeGraphCapturePreflight();
        const bool explicit_decode_ready =
            isExplicitRoutingDecodeGraphCapturable();
        const std::vector<int> fixed_experts =
            fixedTopologyPrefillExpertIds();
        const bool fixed_engines_ready =
            hasPreparedExpertGemmEnginesForExperts(fixed_experts);

        std::ostringstream out;
        out << "device=" << params_.device_id.to_string()
            << " layer=" << params_.layer_idx
            << " seq_len=" << params_.seq_len
            << " forced_grouped_verifier="
            << perfBool(params_.force_grouped_verifier_prefill_for_decode)
            << " route="
            << (runtime_row_grouping_requested
                    ? "runtime_table_grouped_rows"
                    : (explicit_decode_preflight
                           ? "explicit_routing_decode"
                           : (params_.seq_len == 1 &&
                                      !params_.force_grouped_verifier_prefill_for_decode
                                  ? "runtime_table_decode"
                                  : "fixed_topology_prefill")))
            << " backend_grouped="
            << perfBool(supportsGroupedPrefillGraphCaptureBackend(params_.device_id))
            << " kernel=" << perfBool(moe_kernel_ != nullptr)
            << " fixed_preflight=" << perfBool(fixed_preflight)
            << " fixed_engines=" << perfBool(fixed_engines_ready)
            << " fixed_expert_count=" << fixed_experts.size()
            << " active_rows="
            << perfBool(params_.active_row_count_device != nullptr)
            << " require_device_routes="
            << perfBool(params_.require_device_routing_tensor_decode)
            << " routing_indices=" << perfBool(params_.routing_indices != nullptr)
            << " routing_weights=" << perfBool(params_.routing_weights != nullptr)
            << " replica_count=" << params_.replica_set.num_replicated
            << " expert_mask_size=" << params_.expert_mask.size()
            << " runtime_descriptors="
            << perfBool(usesRuntimePlacementWeightDescriptors())
            << " fixed_mask_consumed="
            << perfBool(usesPublishedFixedTopologyMaskGrouping())
            << " fixed_mask_publication=" << mask_publication
            << " runtime_row_grouping_requested="
            << perfBool(runtime_row_grouping_requested)
            << " runtime_row_grouping_ready=" << perfBool(runtime_row_grouping_ready)
            << " runtime_layer=" << perfBool(moe_runtime_layer_ != nullptr)
            << " runtime_table=" << perfBool(params_.moe_runtime_table != nullptr)
            << " runtime_decode_initialized="
            << perfBool(moe_runtime_table_initialized_)
            << " runtime_decode_warmed="
            << perfBool(runtime_grouped_decode_launch_state_prepared_)
            << " runtime_decode_ready=" << perfBool(runtime_decode_ready)
            << " explicit_decode_preflight="
            << perfBool(explicit_decode_preflight)
            << " explicit_decode_ready=" << perfBool(explicit_decode_ready)
            << " gateup_desc=" << grouped_gateup_desc_table_id_
            << " down_desc=" << grouped_down_desc_table_id_;
        return out.str();
    }

    bool MoEExpertComputeStage::supportsGraphCaptureAfterLaunchPreparation() const
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        return false;
#else
        const bool one_row_device_routed_supported =
            supportsDeviceRoutedOneRowGraphCapturePreflight();

        return one_row_device_routed_supported ||
               supportsExplicitRoutingDecodeGraphCapturePreflight() ||
               supportsFixedTopologyPrefillGraphCapturePreflight();
#endif
    }

    bool MoEExpertComputeStage::supportsLazyPrefillGraphCapturePreflight() const
    {
        /*
         * A restored prefix may leave exactly one uncached token. Its routed
         * expert arithmetic is one of the ordinary production one-row paths:
         * a locally routed runtime-table graph, or an explicit-routing graph
         * used by a heterogeneous continuation participant.  The stable
         * device row-count pointer proves that the enclosing transaction is
         * prefill rather than an inferred decode.  Keep both route authorities
         * behind that same proof so a one-row bucket cannot be rejected merely
         * because its routes cross an ExpertOverlay domain boundary.
         */
        return supportsFixedTopologyPrefillGraphCapturePreflight() ||
               (params_.active_row_count_device != nullptr &&
                (supportsDeviceRoutedOneRowGraphCapturePreflight() ||
                 supportsExplicitRoutingDecodeGraphCapturePreflight()));
    }

    bool MoEExpertComputeStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        return supportsLazyPrefillGraphCapturePreflight() &&
               params_.active_row_count_device != nullptr;
    }

    bool MoEExpertComputeStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        (void)ctx;
        if (!stream)
        {
            LOG_ERROR("[MoEExpertComputeStage] Graph launch preparation requires an explicit stream"
                      << " device=" << params_.device_id.to_string()
                      << " layer=" << params_.layer_idx);
            return false;
        }
        setGPUStream(stream);

        if (ownsRuntimePrefillHistogramPublication())
        {
            try
            {
                /* The complete grouped-prefill planner publishes ordinary
                 * demand from this stage, not from MoERoutingStage. Admit its
                 * exact capture/replay stream before any graph can retain the
                 * runtime-table pointer. */
                params_.moe_runtime_table->prepareDecodeHistogramProducerStream(
                    stream);
            }
            catch (const std::exception &error)
            {
                LOG_ERROR(
                    "[MoEExpertComputeStage] Failed to prepare the ordinary "
                    "prefill histogram producer"
                    << " device=" << params_.device_id.to_string()
                    << " layer=" << params_.layer_idx
                    << " reason=" << error.what());
                return false;
            }
        }

        if (supportsGraphCaptureAfterLaunchPreparation())
        {
            if (!refreshGraphStablePlacement(
                    /*preserve_capture_ready=*/false))
            {
                return false;
            }

            const bool one_row_device_routed =
                supportsDeviceRoutedOneRowGraphCapturePreflight();
            if (one_row_device_routed)
            {
                IMoEKernel *kernel = ensureMoEKernel();
                const MoEDecodeDescriptorSource descriptor_source =
                    params_.weight_descriptor_source;
                if (!kernel ||
                    !kernel->prepareGroupedRuntimeDecodeLaunchState(
                        grouped_gateup_desc_table_id_,
                        grouped_down_desc_table_id_,
                        params_.top_k,
                        params_.d_model,
                        params_.expert_intermediate,
                        descriptor_source))
                {
                    LOG_ERROR("[MoEExpertComputeStage] Failed to prepare fused "
                              "runtime grouped-decode launch state"
                              << " layer=" << params_.layer_idx
                              << " device=" << params_.device_id.to_string());
                    runtime_grouped_decode_launch_state_prepared_ = false;
                    return false;
                }
                runtime_grouped_decode_launch_state_prepared_ = true;
            }
            else if (supportsExplicitRoutingDecodeGraphCapturePreflight())
            {
                IMoEKernel *kernel = ensureMoEKernel();
                if (!prepareExplicitRoutingDecodeLaunchState(kernel))
                    return false;
            }
        }

        const bool prepare_prefix_rehydration =
            params_.prefix_runtime_device_rehydration;
        if (!prepare_prefix_rehydration)
        {
            if (sparse_overlay_invocation_bound_)
                sparse_overlay_launch_metadata_dirty_ = false;
            return true;
        }
        if (!hasValidCompactLLEPTransferBinding())
        {
            throw std::logic_error(
                "MoE graph launch requested compact LLEP transport without a "
                "valid graph-owned LocalTP binding");
        }
        if (prepare_prefix_rehydration &&
            (!params_.prefix_runtime_rehydration_transfer_state ||
             !params_.prefix_runtime_rehydration_transfer_state->
                 isMaterializedFor(
                     params_.device_id,
                     params_.prefill_llep_workspace_name)))
        {
            throw std::runtime_error(
                "Prefix-runtime rehydration graph preparation could not "
                "materialize distinct transfer-event ownership");
        }

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "device_rebalance_llep_prefill_capture_resources_validated",
            1.0,
            "prefill",
            params_.device_id.to_string(),
            {{"stage", "moe_expert_grouped_prefill"},
             {"layer", std::to_string(params_.layer_idx)},
             {"current_batch", "explicit_graph_phases"},
             {"prefix_rehydration",
              prepare_prefix_rehydration ? "enabled" : "disabled"},
             {"workspace", params_.prefill_llep_workspace_name}});
        if (sparse_overlay_invocation_bound_)
            sparse_overlay_launch_metadata_dirty_ = false;
        return true;
    }

    StageBufferRequirements MoEExpertComputeStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.canonical_route_contributions)
        {
            reqs.addOutput(
                "canonical_route_contributions",
                params_.canonical_route_contributions->shape(),
                toBufferTensorType(
                    params_.canonical_route_contributions->native_type()));
        }
        else if (params_.output)
        {
            reqs.addOutput(
                "output",
                params_.output->shape(),
                toBufferTensorType(params_.output->native_type()));
        }
        return reqs;
    }

    StageBufferContract MoEExpertComputeStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();
        contract.inputs.reserve(3);
        contract.outputs.reserve(1);

        contract.addInput(params_.input_buffer_id);
        contract.addInput(params_.routing_indices_buffer_id);
        contract.addInput(params_.routing_weights_buffer_id);
        if (params_.canonical_route_contributions)
        {
            contract.addOutput(
                params_.canonical_route_contributions_buffer_id);
        }
        else if (params_.output_registered_in_arena)
            contract.addOutput(params_.output_buffer_id);

        if (params_.combine_shared_expert_in_verifier)
        {
            if (params_.shared_gate_w)
            {
                contract.addPreparedWeight(
                    params_.shared_gate_w,
                    params_.prepared_store,
                    params_.prepared_shared_ref_gate.value_or(PreparedWeightRef{}));
            }
            if (params_.shared_up_w)
            {
                contract.addPreparedWeight(
                    params_.shared_up_w,
                    params_.prepared_store,
                    params_.prepared_shared_ref_up.value_or(PreparedWeightRef{}));
            }
            if (params_.shared_down_w)
            {
                contract.addPreparedWeight(
                    params_.shared_down_w,
                    params_.prepared_store,
                    params_.prepared_shared_ref_down.value_or(PreparedWeightRef{}));
            }
            if (params_.shared_gate_inp)
                contract.addWeight(params_.shared_gate_inp);
        }

        return contract;
    }

    StageDumpInfo MoEExpertComputeStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.input)
            info.addInput("input", params_.input, params_.seq_len, params_.d_model);
        if (params_.routing_indices)
            info.addInput("routing_indices", params_.routing_indices,
                          static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.top_k));
        if (params_.routing_weights)
            info.addInput("routing_weights", params_.routing_weights,
                          static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.top_k));
        if (params_.gate_exps)
            info.addWeight("gate_exps", params_.gate_exps);
        if (params_.up_exps)
            info.addWeight("up_exps", params_.up_exps);
        if (params_.down_exps)
            info.addWeight("down_exps", params_.down_exps);
        if (params_.canonical_route_contributions)
        {
            /*
             * Canonical LocalTP publication defers routed-row reduction and the
             * shared-expert add to the rooted finalizer. This stage therefore
             * owns route slots, not the final MoE output tensor. Keep diagnostics
             * identical to the arena contract so graph snapshots cannot capture
             * a stale alias from a previous request.
             */
            info.addOutput(
                params_.canonical_route_diagnostic_publication ==
                        MoECanonicalRouteDiagnosticPublication::ContinuationLocalBank
                    ? "local_canonical_route_contributions" : "canonical_route_contributions",
                params_.canonical_route_contributions,
                static_cast<size_t>(params_.seq_len) *
                    static_cast<size_t>(params_.top_k),
                params_.d_model);
        }
        else if (params_.output)
        {
            /*
             * The combined shared-verifier path writes the final routed+shared
             * row into this stage's output buffer.  Give snapshots the semantic
             * name so parity CSVs do not report a combined row as routed-only
             * expert output.
             */
            info.addOutput(params_.combine_shared_expert_in_verifier
                               ? "combined_output"
                               : "output",
                           params_.output,
                           params_.seq_len,
                           params_.d_model);
        }

        info.addScalarInt("num_experts", params_.num_experts);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("expert_intermediate", params_.expert_intermediate);
        info.addScalarInt("local_expert_start", params_.local_expert_start);
        info.addScalarInt("local_expert_count", params_.local_expert_count);
        info.addScalarInt("participant_id", params_.my_socket_id);
        info.addScalarInt("participant_count", params_.participant_count);
        return info;
    }

    // =========================================================================
    // MoEExpertComputeStage — IWorkspaceConsumer Implementation
    // =========================================================================

    WorkspaceRequirements MoEExpertComputeStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        WorkspaceRequirements combined;
        /*
         * `params_.seq_len` is the concrete graph bucket and `m` is the
         * allocator's complete serial-family envelope.  Every routed scratch
         * buffer and nested GEMM must use the same maximum row contract.  If
         * one component retains the first request's row count, a later larger
         * request either aliases past its allocation or is rejected after a
         * graph has already captured the published address.
         */
        const int workspace_rows =
            std::max({1, m, cpu_workspace_row_capacity_});
        const int workspace_experts = params_.num_experts;
        const int workspace_top_k = params_.top_k;
        if (params_.device_id.is_cpu() && !params_.cpu_grouped_serial_workspace)
        {
            combined.cpu_grouped_moe_storage =
                std::make_shared<const CPUGroupedMoEStorageLayout>(
                    CPUGroupedMoEStorageLayout::declare(
                        workspace_rows, params_.d_model, params_.expert_intermediate,
                        params_.num_experts, sparse_overlay_route_width_capacity_, params_.device_id));
        }
        if (params_.device_id.is_cuda())
        {
            combined.merge(MoEWorkspaceBuffers::cudaMoE(
                workspace_rows,
                params_.d_model,
                params_.expert_intermediate,
                workspace_experts,
                workspace_top_k));
        }
        else if (params_.device_id.is_rocm())
        {
            combined.merge(MoEWorkspaceBuffers::rocmMoE(
                workspace_rows,
                params_.d_model,
                params_.expert_intermediate,
                workspace_experts,
                workspace_top_k));
        }

        // Names can be shared; format-derived CPU arithmetic geometry cannot.
        // Each prepared CPU member contributes its own slab envelope. GPUs keep
        // their existing format-independent simultaneous-width declaration.
        const auto &engines = params_.prepared_gate_gemm.empty()
                                  ? cached_gate_gemm_
                                  : params_.prepared_gate_gemm;
        for (auto *gemm : engines)
        {
            if (!gemm)
                continue;
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
            if (consumer)
            {
                combined.merge(consumer->getWorkspaceRequirements(
                    workspace_rows,
                    n,
                    k));
                if (params_.device_id.is_cpu())
                    continue;
                const int projection_width =
                    params_.expert_intermediate > 0
                        ? params_.expert_intermediate
                        : n;
                const size_t projection_count = static_cast<size_t>(
                    std::max(1, workspace_top_k)) * 2u;
                const std::vector<int> fused_columns(
                    projection_count, projection_width);
                consumer->appendFusedProjectionWorkspaceRequirements(
                    combined,
                    workspace_rows,
                    fused_columns,
                    params_.d_model > 0 ? params_.d_model : k);
                addCudaConcurrentDecodeGemvSideStreamWorkspace(
                    combined,
                    params_.device_id,
                    workspace_rows,
                    static_cast<size_t>(std::max(0, workspace_top_k)) * 2u);
                break;
            }
        }

        if (params_.device_id.is_cpu())
        {
            const auto &up_engines = params_.prepared_up_gemm.empty()
                ? cached_up_gemm_ : params_.prepared_up_gemm;
            for (auto *gemm : up_engines)
                if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm))
                    combined.merge(consumer->getWorkspaceRequirements(
                        workspace_rows, gemm->get_n(), gemm->get_k()));
        }

        // Down's input width is distinct from gate/up's. Declare its transform
        // from the actual prepared engine, without mutating shared CPU bindings.
        const auto &down_engines = params_.prepared_down_gemm.empty()
            ? cached_down_gemm_ : params_.prepared_down_gemm;
        for (auto *gemm : down_engines)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm))
            {
                // Serial gate/up and down reuse one activation bank. Decode's
                // fused expert-down bundle keeps top-k distinct input rows live.
                combined.merge(consumer->getWorkspaceRequirements(
                    std::max(workspace_rows, workspace_top_k), gemm->get_n(), gemm->get_k()));
                consumer->appendSwiGLUWorkspaceRequirements(
                    combined, workspace_rows, gemm->get_k());
                if (!params_.device_id.is_cpu()) break;
            }
        }

        if (params_.combine_shared_expert_in_verifier &&
            params_.prepared_store &&
            params_.prepared_shared_ref_gate &&
            params_.prepared_shared_ref_up &&
            params_.prepared_shared_ref_down)
        {
            ITensorGemm *shared_gate =
                params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_gate);
            ITensorGemm *shared_up =
                params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_up);
            ITensorGemm *shared_down =
                params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_down);
            const int rows = workspace_rows;
            const int d_model = params_.d_model > 0 ? params_.d_model : n;
            const int intermediate =
                params_.expert_intermediate > 0 ? params_.expert_intermediate : k;

            if (auto *c = dynamic_cast<IWorkspaceConsumer *>(shared_gate))
                combined.merge(c->getWorkspaceRequirements(rows, intermediate, d_model));
            if (auto *c = dynamic_cast<IWorkspaceConsumer *>(shared_up))
                combined.merge(c->getWorkspaceRequirements(rows, intermediate, d_model));
            if (auto *c = dynamic_cast<IWorkspaceConsumer *>(shared_down))
            {
                combined.merge(c->getWorkspaceRequirements(rows, d_model, intermediate));
                c->appendSwiGLUWorkspaceRequirements(combined, rows, shared_down->get_k());
            }
            if (auto *c = dynamic_cast<IWorkspaceConsumer *>(shared_gate))
            {
                const std::array<int, 2> fused_columns = {
                    intermediate, intermediate};
                c->appendFusedProjectionWorkspaceRequirements(
                    combined, rows, fused_columns, d_model);
            }
            addCudaConcurrentDecodeGemvSideStreamWorkspace(
                combined,
                params_.device_id,
                rows,
                /*projection_count=*/2u);
        }
        const bool needs_llep_transport_workspace =
            params_.prefix_runtime_device_rehydration;
        if (needs_llep_transport_workspace &&
            !hasValidCompactLLEPTransferBinding())
        {
            throw std::logic_error(
                "MoE compact LLEP transport workspace requested without a "
                "valid graph-owned LocalTP binding");
        }
        if (needs_llep_transport_workspace)
        {
            const auto &config = params_.prefill_llep_rebalance_config;
            const uint32_t captured_payload_slots =
                std::min<uint32_t>(
                    params_.prefill_llep_payload_slot_capacity,
                    params_.prefill_llep_transfer_slot_count);
            const uint32_t plan_capacity =
                std::max<uint32_t>(
                    1u,
                    boundedU32(deviceMoEPrefillLLEPMergedPlanCapacity(
                        config,
                        deviceMoERebalanceCommandPlanCapacity(
                            config,
                            params_.prefill_llep_transfer_mode),
                        captured_payload_slots)));
            const uint32_t payload_slot_count =
                std::min<uint32_t>(captured_payload_slots, plan_capacity);
            const size_t local_payload_bytes =
                static_cast<size_t>(payload_slot_count) *
                static_cast<size_t>(params_.prefill_llep_payload_slot_bytes);
            const size_t participant_count =
                static_cast<size_t>(std::max<uint32_t>(1u, config.participant_count));
            const std::string &workspace_name = params_.prefill_llep_workspace_name;

            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_TRANSFER_PLAN, workspace_name),
                static_cast<size_t>(plan_capacity) * sizeof(DeviceMoERebalancePlanEntry),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_TRANSFER_PLAN_COUNT, workspace_name),
                sizeof(uint32_t),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_COMMAND_HEADER, workspace_name),
                sizeof(DeviceMoERebalanceCommandBufferHeader),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PLAN, workspace_name),
                participant_count * static_cast<size_t>(plan_capacity) *
                    sizeof(DeviceMoERebalancePlanEntry),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_GATHERED_COMMAND_HEADER, workspace_name),
                participant_count * sizeof(DeviceMoERebalanceCommandBufferHeader),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(
                    MoEDeviceRebalanceStage::WS_TRANSFER_SLOT_CLAIM_INDEX,
                    workspace_name),
                deviceMoETransferSlotClaimIndexBytes(
                    params_.prefill_llep_transfer_slot_count),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPLayerPublicationBufferName(
                    MoEDeviceRebalanceStage::WS_STATUS,
                    workspace_name,
                    params_.layer_idx),
                sizeof(DeviceMoERebalanceStatus),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPLayerPublicationBufferName(
                    MoEDeviceRebalanceStage::WS_APPLY_STATUS,
                    workspace_name,
                    params_.layer_idx),
                sizeof(DeviceMoERebalanceApplyStatus),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_LOCAL_SOURCE_DESCRIPTORS, workspace_name),
                participant_count * static_cast<size_t>(plan_capacity) *
                    sizeof(DeviceMoEExpertDirectoryEntry),
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_LOCAL_TRANSFER_PAYLOAD, workspace_name),
                local_payload_bytes,
                256,
                true});
            combined.buffers.push_back({
                prefillLLEPWorkspaceBufferName(MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PAYLOAD, workspace_name),
                participant_count * local_payload_bytes,
                256,
                true});
        }
        return combined;
    }

    void MoEExpertComputeStage::bindCPUGroupedStorage(
        std::shared_ptr<CPUGroupedMoEStorageBacking> backing)
    {
        // Logical invocation extents are independently checked against the
        // original backing. Its NUMA placement and initialization policy remain
        // authoritative: rebuilding an implicit heap layout would reject a
        // correctly admitted socket-local fragment without changing any bytes.
        if (!backing || !backing->layout().contains(
                CPUGroupedMoEStorageLayout::declare(
                    std::max(1, cpu_workspace_row_capacity_), params_.d_model,
                    params_.expert_intermediate, params_.num_experts,
                    sparse_overlay_route_width_capacity_, params_.device_id,
                    backing->layout().batch.placement(),
                    backing->layout().batch.initialization())))
            throw std::logic_error("CPU MoE stage requires compatible declared workspace roots");
        if (backing != cpu_grouped_backing_)
        {
            cpu_grouped_backing_ = std::move(backing);
            scratch_gate_batch_.clear();
            scratch_up_batch_.clear();
            scratch_down_batch_.clear();
            swiglu_scratch_batch_.clear();
            for (int route = 0; route < sparse_overlay_route_width_capacity_; ++route)
            {
                const size_t f = static_cast<size_t>(params_.expert_intermediate);
                const size_t d = static_cast<size_t>(params_.d_model);
                scratch_gate_batch_.push_back(std::dynamic_pointer_cast<FP32Tensor>(
                    cpu_grouped_backing_->gate->create_view({1, f}, route * f)));
                scratch_up_batch_.push_back(std::dynamic_pointer_cast<FP32Tensor>(
                    cpu_grouped_backing_->up->create_view({1, f}, route * f)));
                scratch_down_batch_.push_back(std::dynamic_pointer_cast<FP32Tensor>(
                    cpu_grouped_backing_->output->create_view({1, d}, route * d)));
                swiglu_scratch_batch_.emplace_back(
                    cpu_grouped_backing_->route_outputs.data() + route * f, f);
            }
            scratch_out_ = scratch_down_batch_.front();
        }
        auto *kernel = dynamic_cast<CPUMoEKernel *>(acquireMoEKernelObject());
        if (!kernel)
            throw std::logic_error("CPU MoE stage has no CPU kernel");
        kernel->bindRouterQ8HiddenStorage(cpu_grouped_backing_->kernel_router_q8);
    }

    void MoEExpertComputeStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        clearCompletedCPUActiveExpertView();
        // Allocator rebuilds revoke borrows through the public null binding.
        // The graph-local kernel and its backing retain their own lifetimes.
        if (!workspace)
        {
            unbindWorkspace();
            return;
        }
        if (workspace != bound_workspace_)
        {
            runtime_grouped_decode_launch_state_prepared_ = false;
            invalidateFixedTopologyMaskPublication();
        }

        if (params_.device_id.is_cpu())
            bindCPUGroupedStorage(params_.cpu_grouped_serial_workspace
                ? params_.cpu_grouped_serial_workspace->backing_
                : (workspace ? workspace->cpuGroupedMoEStorage() : nullptr));

        // Bind workspace to ALL expert GEMM engines (gate, up, down for each expert)
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

        const auto &gate = params_.prepared_gate_gemm.empty() ? cached_gate_gemm_ : params_.prepared_gate_gemm;
        const auto &up = params_.prepared_up_gemm.empty() ? cached_up_gemm_ : params_.prepared_up_gemm;
        const auto &down = params_.prepared_down_gemm.empty() ? cached_down_gemm_ : params_.prepared_down_gemm;

        bindAll(gate);
        bindAll(up);
        bindAll(down);

        auto bindShared = [workspace](ITensorGemm *gemm)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm))
                consumer->bindWorkspace(workspace);
        };
        if (params_.combine_shared_expert_in_verifier &&
            params_.prepared_store &&
            params_.prepared_shared_ref_gate &&
            params_.prepared_shared_ref_up &&
            params_.prepared_shared_ref_down)
        {
            bindShared(params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_gate));
            bindShared(params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_up));
            bindShared(params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_down));
        }

        if (moe_kernel_)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(moe_kernel_))
                consumer->bindWorkspace(workspace);
        }

        bound_workspace_ = workspace;
        const bool bind_prefix_rehydration =
            params_.prefix_runtime_device_rehydration;
        if (workspace && bind_prefix_rehydration)
        {
            if (!hasValidCompactLLEPTransferBinding())
            {
                throw std::logic_error(
                    "MoE LLEP workspace binding requires a complete compact "
                    "transfer topology");
            }
            if (bind_prefix_rehydration &&
                (!params_.prefix_runtime_rehydration_transfer_state ||
                 !params_.prefix_runtime_rehydration_transfer_state->
                      materializePersistentResources(
                          params_.device_id,
                          params_.prefill_llep_workspace_name)))
            {
                throw std::runtime_error(
                    "MoE LLEP workspace binding could not materialize its "
                    "distinct prefix-rehydration transfer resources");
            }
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_rebalance_llep_prefill_persistent_resources_bound",
                1.0,
                "setup",
                params_.device_id.to_string(),
                {{"stage", "moe_expert_grouped_prefill"},
                 {"layer", std::to_string(params_.layer_idx)},
                 {"current_batch", "explicit_graph_phases"},
                 {"prefix_rehydration",
                  bind_prefix_rehydration ? "enabled" : "disabled"},
                 {"workspace", params_.prefill_llep_workspace_name}});
        }
        LOG_TRACE("[MoEExpertComputeStage] Bound workspace to "
                  << gate.size() + up.size() + down.size() << " expert GEMM engines");
    }

    void MoEExpertComputeStage::unbindWorkspace()
    {
        clearCompletedCPUActiveExpertView();
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

        const auto &gate = params_.prepared_gate_gemm.empty() ? cached_gate_gemm_ : params_.prepared_gate_gemm;
        const auto &up = params_.prepared_up_gemm.empty() ? cached_up_gemm_ : params_.prepared_up_gemm;
        const auto &down = params_.prepared_down_gemm.empty() ? cached_down_gemm_ : params_.prepared_down_gemm;

        unbindAll(gate);
        unbindAll(up);
        unbindAll(down);

        auto unbindShared = [](ITensorGemm *gemm)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm))
                consumer->unbindWorkspace();
        };
        if (params_.combine_shared_expert_in_verifier &&
            params_.prepared_store &&
            params_.prepared_shared_ref_gate &&
            params_.prepared_shared_ref_up &&
            params_.prepared_shared_ref_down)
        {
            unbindShared(params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_gate));
            unbindShared(params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_up));
            unbindShared(params_.prepared_store->gemmKernel(*params_.prepared_shared_ref_down));
        }

        if (moe_kernel_)
        {
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(moe_kernel_))
                consumer->unbindWorkspace();
        }

        bound_workspace_ = nullptr;
        runtime_grouped_decode_launch_state_prepared_ = false;
        invalidateFixedTopologyMaskPublication();
        sparse_overlay_launch_metadata_dirty_ =
            sparse_overlay_invocation_bound_;
    }

    bool MoEExpertComputeStage::bindSparseOverlayInvocation(
        const SparseOverlayInvocation &invocation)
    {
        // A rejected packet cannot lend a prior observation or execution lease.
        clearCompletedCPUActiveExpertView();
        releaseCPUJointInvocation();
        MoEOverlayParticipantBankLease joint_bank;
        const size_t expert_count =
            static_cast<size_t>(std::max(params_.num_experts, 0));
        bool verified_immutable_joint_reuse = false;
        if(params_.cpu_joint_layer) {
            if(invocation.binding_kind==SparseOverlayBindingKind::SetupPriming) {
                if(invocation.cpu_joint_epoch)return false;
            } else {
                const auto* epoch=invocation.cpu_joint_epoch;
                if(!epoch || !epoch->bankLease() || epoch->family()!=params_.cpu_joint_layer->family() ||
                   epoch->family().owner_before(params_.cpu_joint_layer->family()) ||
                   params_.cpu_joint_layer->family().owner_before(epoch->family()) ||
                   epoch->epoch()!=invocation.engine_binding_generation || epoch->layer()!=params_.layer_idx ||
                   epoch->participantId()!=params_.cpu_joint_layer->participantId()) {
                    LOG_ERROR("CPU fragment invocation has a foreign or absent original epoch family");return false;
                }
                joint_bank=epoch->bankLease();
                verified_immutable_joint_reuse =
                    invocation.immutable_bank_binding_reused &&
                    sparse_overlay_invocation_bound_ &&
                    sparse_overlay_engine_binding_generation_ == invocation.engine_binding_generation &&
                    joint_bank.get() == last_verified_cpu_joint_bank_ &&
                    invocation.expert_mask == last_verified_mask_ptr_ &&
                    invocation.gate_engines.data() == last_verified_gate_ptr_ &&
                    invocation.up_engines.data() == last_verified_up_ptr_ &&
                    invocation.down_engines.data() == last_verified_down_ptr_ &&
                    invocation.expert_mask &&
                    invocation.expert_mask->size() == expert_count &&
                    invocation.gate_engines.size() == expert_count &&
                    invocation.up_engines.size() == expert_count &&
                    invocation.down_engines.size() == expert_count &&
                    cached_gate_gemm_.size() == expert_count &&
                    cached_up_gemm_.size() == expert_count &&
                    cached_down_gemm_.size() == expert_count;
                if (!verified_immutable_joint_reuse) {
                    const auto& layer=joint_bank->layerForModelLayer(params_.layer_idx);
                    if(!invocation.expert_mask || *invocation.expert_mask!=layer.resident_mask ||
                       invocation.gate_engines.size()!=layer.fragments.size() ||
                       invocation.up_engines.size()!=layer.fragments.size() || invocation.down_engines.size()!=layer.fragments.size())return false;
                    for(size_t expert=0;expert<layer.fragments.size();++expert) {
                        const auto& actual=layer.fragments[expert];
                        if(invocation.gate_engines[expert]!=actual.gate().get() || invocation.up_engines[expert]!=actual.up().get() ||
                           invocation.down_engines[expert]!=actual.down().get()) {
                            LOG_ERROR("CPU fragment invocation replaced a candidate bank engine");return false;
                        }
                    }
                }
            }
        } else if(invocation.cpu_joint_epoch)return false;
        const bool host_packet_binding =
            invocation.binding_kind ==
            SparseOverlayBindingKind::HostPacketPublication;
        const bool production_service_phase =
            invocation.service_phase ==
                MoEOverlayServicePhaseHint::Decode ||
            invocation.service_phase ==
                MoEOverlayServicePhaseHint::Prefill ||
            invocation.service_phase ==
                MoEOverlayServicePhaseHint::GroupedVerifier;
        const bool valid_service_phase = host_packet_binding
            ? production_service_phase
            : invocation.service_phase ==
                      MoEOverlayServicePhaseHint::Auto ||
                  production_service_phase;
        bool valid_cpu_hidden_binding =
            invocation.cpu_transported_hidden.empty();
        if (host_packet_binding)
        {
            const auto &hidden = invocation.cpu_transported_hidden;
            valid_cpu_hidden_binding =
                params_.device_id.is_cpu() &&
                params_.cpu_router_q8_input_publication ==
                    CPURouterQ8InputPublicationPolicy::
                        PublishTransportedRows &&
                hidden.source != nullptr &&
                hidden.source_row_capacity > 0 &&
                hidden.compact_to_source_row.size() ==
                    static_cast<size_t>(invocation.live_rows);
            if (valid_cpu_hidden_binding)
            {
                valid_cpu_hidden_binding = std::all_of(
                    hidden.compact_to_source_row.begin(),
                    hidden.compact_to_source_row.end(),
                    [&](int source_row)
                    {
                        return source_row >= 0 &&
                               source_row < hidden.source_row_capacity;
                    });
            }
        }
        bool valid_binding_identity = false;
        switch (invocation.binding_kind)
        {
        case SparseOverlayBindingKind::SetupPriming:
            valid_binding_identity =
                invocation.engine_binding_generation == 0 &&
                !sparse_overlay_invocation_bound_;
            break;
        case SparseOverlayBindingKind::ResidencyPublication:
            valid_binding_identity =
                invocation.engine_binding_generation != 0;
            break;
        case SparseOverlayBindingKind::HostPacketPublication:
            valid_binding_identity =
                params_.device_id.is_cpu() &&
                invocation.engine_binding_generation != 0;
            break;
        }
        const int route_width =
            invocation.route_width > 0
                ? invocation.route_width
                : params_.top_k;
        const auto tensorHolds = [](const TensorBase *tensor,
                                    size_t required_elements)
        {
            return tensor && tensor->native_type() == TensorType::FP32 &&
                   tensor->numel() >= required_elements;
        };

        if (isGraphCaptureActive() || !params_.device_id.is_valid() ||
            sparse_overlay_route_width_capacity_ <= 0 ||
            sparse_overlay_route_width_capacity_ > 16 ||
            sparse_overlay_route_width_capacity_ > params_.num_experts ||
            route_width <= 0 ||
            route_width > sparse_overlay_route_width_capacity_ ||
            (!host_packet_binding && invocation.route_width > 0 &&
             route_width != params_.top_k) ||
            params_.num_experts <= 0 ||
            params_.d_model <= 0 || invocation.live_rows <= 0 ||
            !valid_binding_identity ||
            !valid_service_phase ||
            !valid_cpu_hidden_binding ||
            !invocation.expert_mask ||
            params_.expert_mask.size() != expert_count ||
            invocation.expert_mask->size() != expert_count ||
            invocation.gate_engines.size() != expert_count ||
            invocation.up_engines.size() != expert_count ||
            invocation.down_engines.size() != expert_count)
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Invalid sparse-overlay invocation geometry or lifecycle");
            return false;
        }

        const size_t rows = static_cast<size_t>(invocation.live_rows);
        if (rows > std::numeric_limits<size_t>::max() /
                       static_cast<size_t>(params_.d_model) ||
            rows > std::numeric_limits<size_t>::max() /
                       static_cast<size_t>(route_width))
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Sparse-overlay invocation element count overflows size_t");
            return false;
        }
        const size_t hidden_elements = rows * static_cast<size_t>(params_.d_model);
        const size_t routing_elements =
            rows * static_cast<size_t>(route_width);
        if (!tensorHolds(invocation.input, hidden_elements) ||
            !tensorHolds(invocation.routing_indices, routing_elements) ||
            !tensorHolds(invocation.routing_weights, routing_elements) ||
            !tensorHolds(invocation.output, hidden_elements))
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Sparse-overlay invocation tensors do not hold the live FP32 prefix");
            return false;
        }

        const bool first_binding = !sparse_overlay_invocation_bound_;
        if (!first_binding && !host_packet_binding &&
            (params_.input != invocation.input ||
             params_.routing_indices != invocation.routing_indices ||
             params_.routing_weights != invocation.routing_weights ||
             params_.output != invocation.output ||
             params_.seq_len != invocation.live_rows ||
             params_.top_k != route_width))
        {
            LOG_ERROR(
                "[MoEExpertComputeStage] Sparse-overlay replay attempted to change a captured tensor-family identity");
            return false;
        }

        const bool binding_generation_changed =
            first_binding ||
            sparse_overlay_engine_binding_generation_ !=
                invocation.engine_binding_generation;
        bool mask_changed = false;
        bool engines_changed = false;
        const bool direct_rcu_layer_view =
            host_packet_binding &&
            params_.cpu_joint_layer &&
            params_.prepared_engine_binding_lifecycle ==
                MoEPreparedEngineBindingLifecycle::SparseOverlayInvocationPublished &&
            static_cast<bool>(joint_bank);
        bool direct_rcu_has_local = false;
        if (direct_rcu_layer_view)
        {
            if (verified_immutable_joint_reuse)
            {
                direct_rcu_has_local = last_verified_cpu_joint_has_local_;
            }
            else
            {
                const auto &layer = joint_bank->layerForModelLayer(params_.layer_idx);
                for (size_t expert = 0; expert < expert_count; ++expert)
                {
                    if (!layer.resident_mask[expert])
                        continue;
                    direct_rcu_has_local = true;
                    if (!invocation.gate_engines[expert] ||
                        !invocation.up_engines[expert] ||
                        !invocation.down_engines[expert])
                    {
                        LOG_ERROR(
                            "[MoEExpertComputeStage] Sparse-overlay invocation has an active expert without complete prepared engines");
                        return false;
                    }
                }
            }
        }
        else if (!verified_immutable_joint_reuse && (binding_generation_changed || host_packet_binding))
        {
            mask_changed =
                !std::equal(
                    invocation.expert_mask->begin(),
                    invocation.expert_mask->end(),
                    params_.expert_mask.begin(),
                    params_.expert_mask.end());
            for (size_t expert = 0; expert < expert_count; ++expert)
            {
                const bool active = (*invocation.expert_mask)[expert];
                if (active &&
                    (!invocation.gate_engines[expert] ||
                     !invocation.up_engines[expert] ||
                     !invocation.down_engines[expert]))
                {
                    LOG_ERROR(
                        "[MoEExpertComputeStage] Sparse-overlay invocation has an active expert without complete prepared engines");
                    return false;
                }
                engines_changed = engines_changed ||
                                  params_.prepared_gate_gemm[expert] !=
                                      invocation.gate_engines[expert] ||
                                  params_.prepared_up_gemm[expert] !=
                                      invocation.up_engines[expert] ||
                                  params_.prepared_down_gemm[expert] !=
                                      invocation.down_engines[expert];
            }
        }

        params_.input = invocation.input;
        params_.routing_indices = invocation.routing_indices;
        params_.routing_weights = invocation.routing_weights;
        params_.output = invocation.output;
        params_.seq_len = invocation.live_rows;
        params_.top_k = route_width;
        if (host_packet_binding)
        {
            if (sparse_overlay_cpu_compact_to_source_row_.size() < rows)
            {
                LOG_ERROR(
                    "[MoEExpertComputeStage] CPU transported hidden-row map exceeds setup capacity");
                return false;
            }
            sparse_overlay_cpu_hidden_source_ =
                invocation.cpu_transported_hidden.source;
            sparse_overlay_cpu_hidden_source_row_capacity_ =
                invocation.cpu_transported_hidden.source_row_capacity;
            std::copy(
                invocation.cpu_transported_hidden.compact_to_source_row.begin(),
                invocation.cpu_transported_hidden.compact_to_source_row.end(),
                sparse_overlay_cpu_compact_to_source_row_.begin());
        }
        if (direct_rcu_layer_view)
        {
            sparse_overlay_invocation_bound_ = true;
            sparse_overlay_binding_kind_ = invocation.binding_kind;
            sparse_overlay_service_phase_ = invocation.service_phase;
            sparse_overlay_engine_binding_generation_ =
                invocation.engine_binding_generation;
            cpu_joint_invocation_bank_ = std::move(joint_bank);
            cpu_joint_invocation_layer_ =
                &cpu_joint_invocation_bank_->layerForModelLayer(params_.layer_idx);
            cpu_joint_invocation_layer_has_local_ = direct_rcu_has_local;
            last_verified_cpu_joint_bank_ = cpu_joint_invocation_bank_.get();
            last_verified_cpu_joint_has_local_ = direct_rcu_has_local;
            last_verified_mask_ptr_ = invocation.expert_mask;
            last_verified_gate_ptr_ = invocation.gate_engines.data();
            last_verified_up_ptr_ = invocation.up_engines.data();
            last_verified_down_ptr_ = invocation.down_engines.data();
            return true;
        }
        if (binding_generation_changed || mask_changed || engines_changed ||
            verified_immutable_joint_reuse)
        {
            std::copy(
                invocation.expert_mask->begin(),
                invocation.expert_mask->end(),
                params_.expert_mask.begin());
            std::copy(
                invocation.gate_engines.begin(),
                invocation.gate_engines.end(),
                params_.prepared_gate_gemm.begin());
            std::copy(
                invocation.up_engines.begin(),
                invocation.up_engines.end(),
                params_.prepared_up_gemm.begin());
            std::copy(
                invocation.down_engines.begin(),
                invocation.down_engines.end(),
                params_.prepared_down_gemm.begin());
            cpu_joint_tables_dirty_ = true;
        }

        /*
         * Weight descriptor tables contain engine-owned device pointers. A
         * residency epoch that changes only the live route count can reuse the
         * tables; an actual engine replacement must rebuild them before the
         * next launch. The vectors were construction-sized above, so these
         * copies do not allocate in the inference path.
         */
        const bool caches_uninitialized =
            cached_gate_gemm_.size() != expert_count ||
            cached_up_gemm_.size() != expert_count ||
            cached_down_gemm_.size() != expert_count;
        sparse_overlay_invocation_bound_ = true;
        sparse_overlay_binding_kind_ = invocation.binding_kind;
        sparse_overlay_service_phase_ = invocation.service_phase;
        sparse_overlay_engine_binding_generation_ =
            invocation.engine_binding_generation;
        if (first_binding || engines_changed || mask_changed ||
            caches_uninitialized || verified_immutable_joint_reuse)
        {
            if (caches_uninitialized)
            {
                cached_gate_gemm_ = params_.prepared_gate_gemm;
                cached_up_gemm_ = params_.prepared_up_gemm;
                cached_down_gemm_ = params_.prepared_down_gemm;
            }
            else
            {
                std::copy(
                    params_.prepared_gate_gemm.begin(),
                    params_.prepared_gate_gemm.end(),
                    cached_gate_gemm_.begin());
                std::copy(
                    params_.prepared_up_gemm.begin(),
                    params_.prepared_up_gemm.end(),
                    cached_up_gemm_.begin());
                std::copy(
                    params_.prepared_down_gemm.begin(),
                    params_.prepared_down_gemm.end(),
                    cached_down_gemm_.begin());
            }
            cpu_joint_tables_dirty_ = true;
            grouped_gateup_desc_table_dirty_ = true;
            grouped_down_desc_table_dirty_ = true;
            runtime_grouped_decode_launch_state_prepared_ = false;
            invalidateFixedTopologyMaskPublication();
            sparse_overlay_launch_metadata_dirty_ =
                !host_packet_binding;
        }
        cpu_joint_invocation_bank_=std::move(joint_bank);
        if (cpu_joint_invocation_bank_)
        {
            last_verified_cpu_joint_bank_ = cpu_joint_invocation_bank_.get();
            last_verified_cpu_joint_has_local_ = false;
            last_verified_mask_ptr_ = invocation.expert_mask;
            last_verified_gate_ptr_ = invocation.gate_engines.data();
            last_verified_up_ptr_ = invocation.up_engines.data();
            last_verified_down_ptr_ = invocation.down_engines.data();
        }
        else
        {
            last_verified_cpu_joint_bank_ = nullptr;
            last_verified_cpu_joint_has_local_ = false;
            last_verified_mask_ptr_ = nullptr;
            last_verified_gate_ptr_ = nullptr;
            last_verified_up_ptr_ = nullptr;
            last_verified_down_ptr_ = nullptr;
        }
        return true;
    }

    bool MoEExpertComputeStage::hasWorkspace() const
    {
        return bound_workspace_ != nullptr;
    }
} // namespace llaminar2
