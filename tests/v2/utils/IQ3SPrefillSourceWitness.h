/**
 * @file IQ3SPrefillSourceWitness.h
 * @brief Original IQ3_S factors through captured runtime-planned prefill.
 *
 * Gate, up and down source-factor roles retain the original signed-grid payload.
 * A single admitted workspace retains the large and partial
 * row graphs; alternating replays poison all observed values. Literal source
 * equations and the real Q8 publications determine expected outputs, without
 * consulting prepared factors or substituting row-wise prefill execution.
 */
#pragma once
#include "IQ3SFusedMoEGpuSourceWitness.h"
#include "execution/moe/MoERuntimeTable.h"

namespace llaminar2::test::iq3s_source_witness
{
/** @brief Keep runtime placement and both captured geometries alive through joins. */
struct PrefillOwners : GpuOwners
{
    std::unique_ptr<DeviceMoERuntimeTable> runtime;
    std::array<std::unique_ptr<IGPUGraphCapture>,2> retained;
    /** @brief Bind the exact endpoint before creating stream users.
     * @param b Backend authority. @param s Non-null worker stream.
     * @param d Explicit endpoint.
     */
    PrefillOwners(IBackend *b,void *s,DeviceId d):GpuOwners{b,s,d}{}
    /** @brief Join before derived graph/runtime members or base buffers retire. */
    ~PrefillOwners()
    {
        if(stream&&backend)(void)backend->synchronizeStream(stream,device.ordinal);
    }
};

/**
 * @brief Prove all three source roles in actual M512 and partial-M3 prefill graphs.
 * @param device Explicit CUDA or ROCm endpoint from its registered regression.
 * @details All output rows are checked, as are the live hidden/middle Q8
 * publications. Runtime plans own live route counts; allocation capacity is
 * neither a launch count nor the extent of a terminal observation.
 */
inline void runPrefill(DeviceId device)
{
    auto &context=GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]{
        auto *backend=getBackendFor(device);ASSERT_NE(backend,nullptr);
        void *stream=context.getOrCreateAuxiliaryStream("iq3s.original_source.prefill");
        ASSERT_NE(stream,nullptr);
        constexpr int capacity=512;
        constexpr std::array<int,2> geometries{capacity,3};
        for(const Role role:ordinaryRoles)
        {
            SCOPED_TRACE(int(role));
            // Every source role must supply its own real construction evidence.
            PerfStatsCollector::reset();
            PrefillOwners o(backend,stream,device);
            o.weights=sources(role);o.prepared.reserve(3);
            std::array<DeviceNativeVNNIMatrixDesc,3> desc{};
            const std::array<TensorBase*,3> raw{o.weights.gate.get(),o.weights.up.get(),o.weights.down.get()};
            for(int p=0;p<3;++p)
            {
                o.prepared.push_back(makeGpuPreparedGemm(raw[p],device,
                    "iq3s.original_source.prefill."+std::to_string(int(role))+"."+std::to_string(p),
                    ModelContextId{uint64_t(6001000+int(role)*16+p+(device.is_cuda()?0:1000))}));
                ASSERT_NE(o.prepared.back().kernel,nullptr);
                o.prepared.back().kernel->setGPUStream(stream);
                ASSERT_TRUE(o.prepared.back().kernel->exportNativeVNNIMatrixDesc(desc[p]));
            }
            const int target=gateTarget(role)?0:upTarget(role)?1:2;
            ASSERT_EQ(desc[target].codebook_id,11);
            ASSERT_EQ(desc[target].source_identity_present,1);
            ASSERT_EQ(desc[target].source_codebook_id,11);
            ASSERT_EQ(desc[target].source_is_superblock,1);
            ASSERT_EQ(desc[0].codebook_id,desc[1].codebook_id);
            const auto requirements=device.is_cuda()?
                MoEWorkspaceBuffers::cudaMoE(capacity,width,width,1,1):
                MoEWorkspaceBuffers::rocmMoE(capacity,width,width,1,1);
            o.workspace=std::make_unique<DeviceWorkspaceManager>(device,requirements.total_bytes_with_alignment());
            ASSERT_TRUE(o.workspace->allocate(requirements));
            o.kernel=llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
            ASSERT_NE(o.kernel,nullptr);o.kernel->setGPUStream(stream);
            auto *consumer=dynamic_cast<IWorkspaceConsumer*>(o.kernel.get());
            ASSERT_NE(consumer,nullptr);consumer->bindWorkspace(o.workspace.get());
            const int gate_table=o.kernel->uploadGroupedExpertGateUpDescriptorTables(&desc[0],&desc[1],1,width,width);
            const int down_table=o.kernel->uploadGroupedExpertDownDescriptorTable(&desc[2],1,width,width);
            ASSERT_GE(gate_table,0);ASSERT_GE(down_table,0);

            DeviceMoERuntimeTable::Config cfg;
            cfg.device_id=device;cfg.num_layers=1;cfg.num_experts=1;cfg.top_k=1;
            cfg.mirror_to_device=true;cfg.prefill_token_capacity=capacity;
            o.runtime=std::make_unique<DeviceMoERuntimeTable>(cfg);
            MoEPlacementUpdate placement;
            placement.epoch=1;placement.expert_count=1;placement.experts.resize(1);
            placement.local_compute_mask.assign(1,1);
            placement.replica_role.assign(1,uint8_t(DeviceMoEReplicaRole::Primary));
            auto &expert=placement.experts[0];
            expert.gate=desc[0];expert.up=desc[1];expert.down=desc[2];
            expert.logical_expert_id=0;expert.local_slot=0;expert.owner_participant=0;
            expert.flags=toMoEExpertFlags(DeviceMoEExpertFlags::Valid|DeviceMoEExpertFlags::Resident|
                DeviceMoEExpertFlags::PreferredOwner|DeviceMoEExpertFlags::LocalCompute);
            ASSERT_TRUE(o.runtime->prepareInactiveBank(0,placement));
            ASSERT_TRUE(o.runtime->flipActiveBank(0,1,stream));
            o.input=std::make_unique<FP32Tensor>(std::vector<size_t>{capacity,width});
            o.ids=std::make_unique<FP32Tensor>(std::vector<size_t>{capacity,1});
            o.route_weights=std::make_unique<FP32Tensor>(std::vector<size_t>{capacity,1});
            o.output=std::make_unique<FP32Tensor>(std::vector<size_t>{capacity,width});
            const auto x=hidden();
            for(int row=0;row<capacity;++row)std::copy(x.begin(),x.end(),o.input->mutable_data()+row*width);
            std::fill_n(o.ids->mutable_data(),capacity,0.0f);
            std::fill_n(o.route_weights->mutable_data(),capacity,1.0f);
            std::fill_n(o.output->mutable_data(),capacity*width,0.0f);
            for(auto *tensor:{o.input.get(),o.ids.get(),o.route_weights.get(),o.output.get()})
                TransferEngine::prepareDeviceInput(tensor,device,stream);
            const auto body=[&](int rows){
                return o.kernel->publishCompleteGroupedPrefillPlanFromRouter(o.runtime->deviceLayerState(0),
                    o.ids.get(),o.route_weights.get(),rows,rows,1,1,gate_table,down_table,true,MoEGroupedPlanDemand::None)&&
                    o.kernel->executeGroupedPrefillPipelineFromPublishedRuntimePlan(o.runtime->deviceLayerState(0),
                        o.runtime->hostLayerState(0),o.input.get(),o.output.get(),gate_table,down_table,rows,width,width,1,1);
            };
            for(size_t index=0;index<geometries.size();++index)
            {
                const int rows=geometries[index];
                ASSERT_TRUE(body(rows));ASSERT_TRUE(backend->synchronizeStream(stream,device.ordinal));
                o.retained[index]=context.createGraphCapture(stream);ASSERT_NE(o.retained[index],nullptr);
                {
                    ScopedBackendGraphCapture recording(context,*o.retained[index],"IQ3 source runtime prefill");
                    ASSERT_TRUE(recording.begin());ASSERT_TRUE(body(rows));recording.finish();
                }
                ASSERT_TRUE(o.retained[index]->instantiate());
                std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
                ASSERT_TRUE(o.retained[index]->inspectKernelNodes(nodes,&error))<<error;
                ASSERT_FALSE(nodes.empty());
            }
            const auto expected_middle=expectedMiddle(role);
            const float expected=expectedDown(role,*o.weights.down,expected_middle.data());
            for(int replay=0;replay<20;++replay)
            {
                const size_t index=size_t(replay%2);const int rows=geometries[index];
                SCOPED_TRACE("rows="+std::to_string(rows)+" replay="+std::to_string(replay));
                const auto poison=[&](const char *name,int pattern,size_t bytes){
                    void *pointer=o.workspace->getBuffer(name);
                    return pointer&&backend->memset(pointer,pattern,bytes,device.ordinal,stream);
                };
                ASSERT_TRUE(backend->memset(o.output->gpu_data_ptr(),0xff,rows*width*sizeof(float),device.ordinal,stream));
                ASSERT_TRUE(poison(MoEWorkspaceBuffers::PREFILL_A_INT8,replay&1?0x5a:0xa5,rows*width));
                ASSERT_TRUE(poison(MoEWorkspaceBuffers::PREFILL_A_SCALES,0xff,rows*blocks*sizeof(float)));
                ASSERT_TRUE(poison(MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8,replay&1?0xa5:0x5a,rows*width));
                ASSERT_TRUE(poison(MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES,0xff,rows*blocks*sizeof(float)));
                ASSERT_TRUE(o.retained[index]->launch());
                const auto outputs=terminal<float>(o,o.output->gpu_data_ptr(),rows*width);
                const auto codes=terminal<int8_t>(o,o.workspace->getBuffer(MoEWorkspaceBuffers::PREFILL_A_INT8),rows*width);
                const auto hidden_scales=terminal<float>(o,o.workspace->getBuffer(MoEWorkspaceBuffers::PREFILL_A_SCALES),rows*blocks);
                const auto middle=terminal<int8_t>(o,o.workspace->getBuffer(MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8),rows*width);
                const auto scales=terminal<float>(o,o.workspace->getBuffer(MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES),rows*blocks);
                for(int row=0;row<rows;++row)
                {
                    for(int col=0;col<width;++col)
                    {
                        ASSERT_EQ(outputs[row*width+col],expected);
                        ASSERT_EQ(codes[row*width+col],int8_t(x[col]));
                        ASSERT_EQ(middle[row*width+col],expected_middle[col/32].qs[col%32]);
                    }
                    for(int block=0;block<blocks;++block)
                    {
                        ASSERT_EQ(hidden_scales[row*blocks+block],1.0f);
                        ASSERT_EQ(scales[row*blocks+block],fp16_to_fp32(expected_middle[block].d));
                    }
                }
            }
            const std::string name=device.is_cuda()?"cuda_moe_grouped_prefill_swiglu_path_calls":
                "rocm_moe_grouped_prefill_batch_invariant_calls";
            const auto records=PerfStatsCollector::snapshot({"kernel."+name});
            for(const int rows:geometries)
                EXPECT_TRUE(std::any_of(records.begin(),records.end(),[&](const PerfStatRecord &record){
                    const auto count=record.tags.find("seq_len"),topk=record.tags.find("top_k");
                    return record.name==name&&record.value>0&&record.count>0&&count!=record.tags.end()&&
                        count->second==std::to_string(rows)&&topk!=record.tags.end()&&topk->second=="1";
                }))<<"Actual runtime prefill construction counter absent for M="<<rows;
        }
    });
}
} // namespace llaminar2::test::iq3s_source_witness
