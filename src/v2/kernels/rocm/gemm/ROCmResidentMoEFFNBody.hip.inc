/**
 * @file ROCmResidentMoEFFNBody.hip.inc
 * @brief Cooperative resident MoE FFN kernel body.
 */
#pragma once

/**
 * @brief Complete post-router local routed/shared FFN under native grid ordering.
 * @tparam TaskIndex Cold-selected uint32 or uint64 task scheduling, never tensor address arithmetic.
 * @tparam DecoderMask Complete immutable execution-envelope group union, independent of source arithmetic policy.
 * @param a Original source-authenticated descriptor/runtime and persistent arena views.
 * @details All CTAs reach every grid.sync. Each numerical task keeps its original
 * K partition and scalar contribution helper; changing physical resident workers
 * changes scheduling only. Four independent 64-column teams retain the original
 * producer wave layout inside each physical 256-thread CTA. Routed scratch dies
 * before the shared branch reuses it, matching the original nonconcurrent arena plan. Public intermediate
 * tensors and canonical route contributions retain their original values.
 */
// Shared descriptor addresses name original immutable arena tables. No helper
// takes an address into this by-value argument, avoiding a private kernarg copy.
template<class TaskIndex,std::uint32_t DecoderMask>
__global__ __launch_bounds__(256,1) void rocmResidentMoEFFNKernel(const llaminar2::rocm::ResidentMoEFFNArgs a)
{
    static_assert(DecoderMask!=0&&(DecoderMask&~llaminar2::rocm::ResidentMoEFFNDecoderProgram::all_decoders)==0);
    const auto grid=cooperative_groups::this_grid();
    if(!grid.is_valid())__builtin_trap();
    const unsigned lane=threadIdx.x;
    const unsigned team_lane=lane%64;
    // Shipping kernels compile with the original wave64 contract. The team is
    // uniform within that wave, allowing descriptor/task division in scalar
    // registers while four waves retain independent projection tasks.
    const auto team=static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(lane/64)));
    const TaskIndex first_task=TaskIndex(blockIdx.x)*4+team;
    const TaskIndex task_stride=TaskIndex(gridDim.x)*4;
    using llaminar2::rocm::residentMoEFFNNextTask;
    using llaminar2::rocm::residentMoEFFNTaskCoordinates;
    const auto* runtime=reinterpret_cast<const DeviceMoELayerRuntimeView*>(a.runtime);
    const int H=a.hidden_columns;
    const bool live=a.rows.activeRows()!=0;
    if(!live)
    {
        for(size_t i=size_t(blockIdx.x)*256+lane;i<size_t(a.top_k)*H;i+=size_t(gridDim.x)*256)
            a.route_contributions[i]=0.f;
        for(size_t n=size_t(blockIdx.x)*256+lane;n<size_t(H);n+=size_t(gridDim.x)*256)
        {a.routed_output[n]=0.f;}
    }
    if(live&&(!runtime||runtime->expert_count!=unsigned(a.experts)||runtime->top_k!=unsigned(a.top_k)||
        !runtime->intermediate_shard.sameIdentity(a.original)||
        moe_runtime_execution_bank(runtime)==kInvalidMoERuntimeExecutionBank))__builtin_trap();

    // The same original blockwise publisher serves both nonconcurrent branches.
    // Every Q8 group remains a complete original32-lane XOR maximum reduction.
    for(size_t block=size_t(blockIdx.x)*8+lane/32;block<size_t(H/32);block+=size_t(gridDim.x)*8)
    {
        const size_t n=block*32+lane%32;
        const float value=a.hidden[n];float maximum=fabsf(value);
        #pragma unroll
        for(int mask=16;mask>0;mask>>=1)maximum=fmaxf(maximum,__shfl_xor(maximum,mask));
        const float scale=llaminar2::rocm::decode_equivalent::q8Scale(maximum);
        a.hidden_q8[n]=static_cast<int8_t>(llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(value,scale));
        if((lane%32)==0)a.hidden_scales[block]=scale;
    }
    grid.sync();
    // Two source-declared branches execute sequentially because their original
    // arena planes are shared, never concurrent. This is not serial row replay.
    for(int shared=0;shared<2;++shared)
    {
        // Ordinary shared-table M1 computes its public FP32 gate/up planes
        // even when the final gate epilogue masks an inactive row. Retain that
        // publication while never dereferencing inactive routed slot state.
        if(!shared&&!live)continue;
        const int F=shared?a.shared_columns:a.routed_columns;
        const int routes=shared?1:a.top_k;
        const auto gate_policy=shared?a.shared_gate_policy:a.routed_gate_policy;
        const auto down_policy=shared?a.shared_down_policy:a.routed_down_policy;
        const auto gate_execution=shared?a.execution.shared_gate:a.execution.routed_gate;
        const auto down_execution=shared?a.execution.shared_down:a.execution.routed_down;
        const TaskIndex gate_tiles=(static_cast<unsigned>(F)+63u)/64u;
        const TaskIndex route_count=static_cast<unsigned>(routes);
        const TaskIndex gate_tasks=gate_tiles*route_count*gate_policy.maximum;
        for(TaskIndex task=first_task;task<gate_tasks;task=residentMoEFFNNextTask(task,gate_tasks,task_stride))
        {
            const auto coordinate=residentMoEFFNTaskCoordinates(task,gate_tiles,route_count);
            const int route=int(coordinate.route),part=int(coordinate.partition);
            const size_t n=size_t(coordinate.tile)*64+team_lane;
            const DeviceNativeQuantisedMatrixDesc *gate=nullptr,*up=nullptr;
            const bool ready=shared?(gate=a.shared_gate,up=a.shared_up,true):
                moe_native_quantised_load_gate_up_descriptors<true>(nullptr,nullptr,runtime,
                    runtime->topk_expert_ids[route],a.experts,gate,up);
            if(n<F)
            {
                const auto index=(size_t(part)*routes+route)*F+n;
                // Keep one all-format decoder callsite and retire each scalar
                // partial before the next projection. This phase alone writes
                // both original arena banks; consumers wait at the unchanged
                // following grid.sync. Only the accumulator live range changes.
                #pragma unroll 1
                for(int projection=0;projection<2;++projection)
                {
                    const auto* descriptor=projection?up:gate;
                    float value=0.f;
                    if(ready)
                    {
                        const int partitions=residentMoEPartitions(descriptor,gate_policy,gate_execution);
                        if(part<partitions)value=moe_native_quantised_block_partial<DecoderMask>(
                            descriptor,n,a.hidden_q8,a.hidden_scales,F,H,part,partitions);
                    }
                    (projection?a.up_partials:a.gate_partials)[index]=value;
                }
            }
        }
        grid.sync();
        const TaskIndex fold_tasks=gate_tiles*route_count;
        for(TaskIndex task=first_task;task<fold_tasks;task=residentMoEFFNNextTask(task,fold_tasks,task_stride))
        {
            const auto coordinate=residentMoEFFNTaskCoordinates(task,gate_tiles,route_count);
            const int route=int(coordinate.route);
            const size_t n=size_t(coordinate.tile)*64+team_lane;
            const DeviceNativeQuantisedMatrixDesc *gate=nullptr,*up=nullptr;
            const bool ready=shared?(gate=a.shared_gate,up=a.shared_up,true):
                moe_native_quantised_load_gate_up_descriptors<true>(nullptr,nullptr,runtime,
                    runtime->topk_expert_ids[route],a.experts,gate,up);
            float value=0.f;
            if(ready&&n<F)
            {
                const float g=residentMoEOrderedPartial(a.gate_partials,route,n,routes,F,residentMoEPartitions(gate,gate_policy,gate_execution));
                const float u=residentMoEOrderedPartial(a.up_partials,route,n,routes,F,residentMoEPartitions(up,gate_policy,gate_execution));
                if(shared){a.shared_gate_output[n]=g;a.shared_up_output[n]=u;}
                value=llaminar2::rocm::decode_equivalent::swigluValue(g,u);
            }
            float maximum=fabsf(value);
            #pragma unroll
            for(int mask=16;mask>0;mask>>=1)maximum=fmaxf(maximum,__shfl_xor(maximum,mask));
            if(ready&&n<F)
            {
                const float scale=llaminar2::rocm::decode_equivalent::q8Scale(maximum);
                const auto index=size_t(route)*F+n;
                a.swiglu_q8[index]=static_cast<int8_t>(llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(value,scale));
                if((lane%32)==0)a.swiglu_scales[size_t(route)*(F/32)+n/32]=scale;
            }
        }
        grid.sync();
        const TaskIndex down_tiles=(static_cast<unsigned>(H)+63u)/64u;
        const TaskIndex down_tasks=down_tiles*route_count*down_policy.maximum;
        for(TaskIndex task=first_task;task<down_tasks;task=residentMoEFFNNextTask(task,down_tasks,task_stride))
        {
            const auto coordinate=residentMoEFFNTaskCoordinates(task,down_tiles,route_count);
            const int route=int(coordinate.route),part=int(coordinate.partition);
            const size_t n=size_t(coordinate.tile)*64+team_lane;
            const DeviceNativeQuantisedMatrixDesc* down=nullptr;
            const bool ready=shared?(down=a.shared_down,true):moe_native_quantised_load_down_descriptor<true>(
                nullptr,runtime,runtime->topk_expert_ids[route],a.experts,down);
            if(n<H)
            {
                float value=0.f;
                if(ready)
                {
                    const int parts=residentMoEPartitions(down,down_policy,down_execution);
                    if(part<parts)value=moe_native_quantised_block_partial<DecoderMask>(down,n,a.swiglu_q8+size_t(route)*F,
                        a.swiglu_scales+size_t(route)*(F/32),H,F,part,parts);
                }
                a.down_partials[(size_t(part)*routes+route)*H+n]=value;
            }
        }
        grid.sync();
        for(TaskIndex task=first_task;task<down_tiles;task=residentMoEFFNNextTask(task,down_tiles,task_stride))
        {
            const size_t n=size_t(task)*64+team_lane;
            if(n<H)
            {
                float total=0.f;
                for(int route=0;route<routes;++route)
                {
                    const DeviceNativeQuantisedMatrixDesc* down=nullptr;
                    const bool ready=shared?(down=a.shared_down,true):moe_native_quantised_load_down_descriptor<true>(
                        nullptr,runtime,runtime->topk_expert_ids[route],a.experts,down);
                    float value=0.f;
                    if(ready)value=rocm_native_quantised_weight_route_rn(shared?1.f:runtime->topk_weights[route],
                        residentMoEOrderedPartial(a.down_partials,route,n,routes,H,residentMoEPartitions(down,down_policy,down_execution)));
                    if(!shared)a.route_contributions[size_t(route)*H+n]=value;
                    total=rocm_native_quantised_accumulate_rn(total,value);
                }
                if(shared)a.shared_output[n]=total;else a.routed_output[n]=total;
            }
        }
        grid.sync(); // Close the original scratch lifetime before reusing it.
    }
    if(blockIdx.x==0)llaminar2::rocm::moe_shared_gate::gateAdd<false>(a.hidden,a.shared_input_gate,a.shared_output,
        a.routed_output,a.combined_output,1,H,0,live?1:0);
}

#include "ROCmPersistentMoEFFNBody.hip.inc"
#include "ROCmOrdinaryMoEFFNBody.hip.inc"

