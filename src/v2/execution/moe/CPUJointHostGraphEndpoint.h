/**
 * @file CPUJointHostGraphEndpoint.h
 * @brief Bind one actual CPU fragment and its original native canonical return edge.
 *
 * The existing prepared registry, serial arena and global CPU context remain
 * sole owners. This helper creates no peer graph, communicator, transport or
 * tensor. CPU continuation consumes raw canonical routes and performs its
 * existing one ordered-FMA fold followed by original-domain broadcast.
 */
#pragma once
#include "MoEOverlayPreparedGraphBank.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoECPUExpertShardRouteStage.h"
#include <array>
#include "collective/GlobalTPContext.h"
namespace llaminar2 {
/** @brief Immutable cold binding inputs from the actual original graph participant. */
struct CPUJointHostGraphEndpoint {
    const MoEOverlayParticipantResidencyRegistry& residency;
    const ExpertGemmRegistry& registry;
    ModelContextId model;
    std::shared_ptr<IGlobalTPContext> original_context;
    MoELocalExpertStage::CPUExpertWorkspaceSource workspace;
    int query_rows;
    std::string stage_name;
    /**
     * @brief Prepare this participant and return its explicit original native collective node.
     * @param local Participant-local compute declaration with admitted input/output/serial arena.
     * @return Native rooted gather/combine/publication stage following this same compute node.
     * @throws std::invalid_argument for missing/foreign source, context, CPU arena or return contract.
     */
    std::unique_ptr<MoECPUExpertShardRouteStage> bind(MoELocalExpertStage::Params& local) const {
        const auto arena=local.serial_compact_buffer_arena;
        const auto input=local.input_rows_lifetime;
        const auto output=local.output_rows_lifetime;
        if(!local.device_id.is_cpu() || !arena || !arena->cpuJointLayout() || !input || !output ||
            !original_context || query_rows<=0 || local.return_layout!=MoEOverlayReturnLayout::CanonicalExpertRoutes ||
            output->layout!=MoEOverlayReturnLayout::CanonicalExpertRoutes || workspace.workers<=0 ||
            local.overlay_participant_residency || local.cpu_current_batch_llep_state)
            throw std::invalid_argument("CPU joint continuation requires its exact original host endpoint and native context");
        const auto* native=dynamic_cast<GlobalTPContext*>(original_context.get());
        if(!native || !native->authenticatesOriginalCPUExpertBorrow(original_context))
            throw std::invalid_argument("CPU joint continuation requires its original admitted native CPU program");
        const auto endpoint=residency.endpoint(local.runtime_participant_index);
        if(!endpoint || !endpoint->device().is_cpu())
            throw std::invalid_argument("CPU joint continuation lost its original prepared endpoint");
        const auto frame=adoptMoEOverlayCPUJointGraphDeclaration(residency,registry,model,
            local.runtime_participant_index,local.layer_idx,local.num_experts,local.d_model,local.expert_intermediate);
        const auto& geometry=*arena->cpuJointLayout();
        if(!frame || !frame->sourceSelection() ||
            frame->sourceSelection()->participantIndex()!=geometry.member())
            throw std::invalid_argument("CPU joint host graph changed its original fragment/arena coordinate");
        // Compute owns the packed fragment only. The next rooted node owns
        // the canonical return descriptor and publishes its actual live rows.
        local.cpu_joint_layer=frame;
        local.return_layout=MoEOverlayReturnLayout::ParticipantTokenPartials;
        local.gate_exps=nullptr;local.up_exps=nullptr;local.down_exps=nullptr;
        local.expert_intermediate=frame->sourceSelection()->ownership().projection(WeightRole::MoEExpertUp).rows;
        local.cpu_workspace_source=workspace;
        local.cpu_joint_route_producer=MoELocalExpertStage::CPUJointRouteProducerBinding{
            .packed_rows=arena->cpuJointPackedRecords(),.original_route_slots=input->original_route_slots_host,
            .compact_route_slots=input->compact_route_slots_host,.route_slot_capacity=input->entry_capacity};
        local.prepared_gate_gemm.assign(local.num_experts,nullptr);
        local.prepared_up_gemm.assign(local.num_experts,nullptr);
        local.prepared_down_gemm.assign(local.num_experts,nullptr);
        local.expert_mask.assign(local.num_experts,false);
        return std::make_unique<MoECPUExpertShardRouteStage>(MoECPUExpertShardRouteStage::Params{
            .original_source=frame->sourceSelection(),.tp=original_context.get(),
            .packed_records=arena->cpuJointPackedRecords().get(),
            .combined_routes=geometry.isRoot()?arena->cpuCanonicalRoutes().get():nullptr,
            .source_ranges=arena->cpuJointSourceRanges(),.participant_route_maps=arena->cpuJointRouteMaps(),
            .required_route_weights=arena->cpuJointRequiredRoutes(),.rows=query_rows,.top_k=local.top_k,
            .columns=local.d_model,.root_participant=static_cast<int>(geometry.rootMember()),.stage_name=stage_name,
            .original_return_rows=output.get(),.original_return_route_capacity=output->row_capacity,
            .original_sparse_routes=input.get(),.original_prepared_layer=frame,.original_tp_owner=original_context,
            .original_continuation_device=DeviceId::cpu(),
            .return_publication=MoECPUExpertShardRouteStage::OriginalReturnPublication::CanonicalHostRows});
    }
};
}
