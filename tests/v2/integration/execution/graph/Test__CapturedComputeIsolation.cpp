/**
 * @file Test__CapturedComputeIsolation.cpp
 * @brief Real prepared stage recording restores resident inputs before twenty poisoned replays.
 *
 * This explicit synthetic fixture proves the concrete adapter/canonical runner,
 * not a real-model snapshot or TP scaling result. Exact source preparation and
 * stage workspace are ordinary production APIs. Native graphs retire before
 * their tensor/store/workspace owners; all host observations are test terminal
 * boundaries. No collectives, sampler or generative state progress is enabled.
 */
#include "execution/diagnostics/PreparedGemmIsolationAdapter.h"
#include "execution/compute_stages/stages/LMHeadStage.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/Tensors.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/TestTensorFactory.h"
#include <gtest/gtest.h>
#include <cstring>
#ifdef HAVE_ROCM
#include "kernels/rocm/gemm/ROCmNativeVNNISerialGeometry.h"
#endif

// Existing backend host policy observers; these expose no device state and
// add no production entrypoint. Scope tests never change them via a raw setter.
#ifdef HAVE_CUDA
extern "C" int cudaNativeVNNIGemvTuned_getSerialPartitionN();
#endif
#ifdef HAVE_ROCM
extern "C" int rocmGemv_native_vnni_get_serial_partition_n();
#endif
namespace
{
using namespace llaminar2;
using namespace llaminar2::diagnostics;
/** @brief Fail native operations immediately without substituting another execution path. */
void checked(bool value,const char *message){if(!value)throw std::runtime_error(message);}
/** @brief Terminal test fence placed after each setup owner's construction, before its first enqueue. */
struct Join final
{
    IBackend &backend;DeviceId device;void *stream;
    /** @brief Retire setup copies/work before the enclosing owners are destroyed on exceptions. */
    ~Join(){if(!backend.synchronizeStream(stream,device.ordinal))std::terminate();}
};
/** @brief Join and destroy both captures before admitted reset/compute owners can retire. */
struct Retire final
{
    IBackend &backend;DeviceId device;void *stream;
    std::unique_ptr<IGPUGraphCapture> &restore,&compute;
    /** @brief Terminal-only test synchronization; inability to retire is fatal. */
    ~Retire()
    {
        if(!backend.synchronizeStream(stream,device.ordinal))std::terminate();
        compute.reset();restore.reset();
    }
};
/** @brief Observe only existing host-side arithmetic scope state, not native/GPU execution state. */
int currentPartition(DeviceId device)
{
#ifdef HAVE_CUDA
    if(device.is_cuda())return cudaNativeVNNIGemvTuned_getSerialPartitionN();
#endif
#ifdef HAVE_ROCM
    if(device.is_rocm())return rocmGemv_native_vnni_get_serial_partition_n();
#endif
    throw std::invalid_argument("No compiled native partition observer");
}
/** @brief Exercise actual prepared NativeVNNI GEMM through the canonical single-node setup runner. */
void prove(DeviceId device,int serial_width=0)
{
    auto *backend=getBackendFor(device);
    ASSERT_NE(backend,nullptr);ASSERT_GT(backend->deviceCount(),device.ordinal);
    auto context=IDeviceContext::create(device,1);
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]
    {
        const int M=1,N=serial_width?129:64,K=serial_width?640:256;
        const int previous_partition=currentPartition(device);
        auto *stream=worker.defaultStream();
        auto input=std::make_shared<FP32Tensor>(std::vector<std::size_t>{M,K});
        auto output=std::make_shared<FP32Tensor>(std::vector<std::size_t>{M,N});
        auto baseline=std::make_shared<FP32Tensor>(std::vector<std::size_t>{M,K});
        auto weight=std::shared_ptr<Q8_0Tensor>(test::TestTensorFactory::createQ8_0Random({N,K},19351));
        for(int i=0;i<K;++i)input->mutable_data()[i]=float((i*7)%29-14)*.03125f;
        Join input_retirement{*backend,device,stream};
        for(auto *tensor:{static_cast<TensorBase*>(input.get()),static_cast<TensorBase*>(output.get()),
                          static_cast<TensorBase*>(baseline.get())})
            TransferEngine::prepareDeviceInput(tensor,device,stream);
        auto fixture=test::makeGpuPreparedGemm(weight.get(),device,"diagnostic.isolation.synthetic.weight");
        auto store=std::shared_ptr<PreparedWeightStore>(std::move(fixture.store));
        auto stage=std::make_unique<GEMMStage>(GEMMStage::Params{.device_id=device,
            .A=input.get(),.B=weight.get(),.C=output.get(),.m=M,.n=N,.k=K,.alpha=1,.beta=0,
            .transpose_B=true,.gemm_context=GemmContext::LM_HEAD,.a_buffer_id=BufferId::NORMALIZED,.c_buffer_id=BufferId::GATE_PROJ,
            .prepared_ref=fixture.ref,.prepared_store=store.get()});
        stage->setGPUStream(stream);
        const auto requirements=[&]
        {auto arithmetic=fixture.kernel->beginOutputPartitionEquivalenceScope(N,serial_width?serial_width:N);
         return stage->getWorkspaceRequirements(M,N,K);}();
        EXPECT_EQ(currentPartition(device),previous_partition);
        const auto bytes=requirements.total_bytes_with_alignment();
        checked(bytes>0,"Prepared NativeVNNI GEMM declared no scratch");
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0,.device=device,.total_bytes=bytes,.admission_available_bytes=bytes},
            PhysicalMemoryOwner::ExecutionWorkspace,bytes);
        auto memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        auto workspace=std::make_shared<DeviceWorkspaceManager>(device,bytes,memory);
        checked(workspace->allocate(requirements),"Isolation exact stage workspace admission failed");
        stage->bindWorkspace(workspace.get());
        Join workspace_retirement{*backend,device,stream};
        BufferArena arena;
        checked(arena.registerExternalBuffer(BufferId::NORMALIZED,input.get()) &&
            arena.registerExternalBuffer(BufferId::GATE_PROJ,output.get()),"Isolation actual arena binding failed");
        DeviceGraphExecutor executor;executor.setArena(&arena);
        // Positive source width uses the real LMHeadStage, with its own
        // original scoped workspace/capture behavior and a physical tailN129.
        // Replay occurs after the recording policy token has already retired.
        if(serial_width)
        {
            ComputeGraph head_graph;
            auto head=std::make_unique<LMHeadStage>(LMHeadStage::Params{.device_id=device,
                .hidden_states=input.get(),.lm_head_weight=weight.get(),.logits=output.get(),
                .seq_len=M,.d_model=K,.vocab_size=N,.serial_equivalent_partition_width=serial_width,
                .use_prefill_replay_row_offset=false,.input_buffer_id=BufferId::NORMALIZED,.output_buffer_id=BufferId::GATE_PROJ,
                .prepared_ref=fixture.ref,.prepared_store=store.get()});
            head->setGPUStream(stream);head->bindWorkspace(workspace.get());
            EXPECT_EQ(head->getWorkspaceRequirements(M,N,K).total_bytes_with_alignment(),bytes);
            head_graph.addNode("original_lm_head_partition_oracle",std::move(head),device);
            auto captured_head=worker.createGraphCapture(stream);
            ASSERT_NE(captured_head,nullptr);
            Join head_retirement{*backend,device,stream};
            checked(executor.prepareGraphStorageForCapture(head_graph,context.get(),stream,"original head policy"),"Original head frontier failed");
            {ScopedBackendGraphCapture capture(worker,*captured_head,"original LMHeadStage partition oracle");
             checked(capture.begin(),"Original head capture begin failed");
             checked(executor.recordSetupGraphMaterialization(head_graph,context.get()),"Original head recording failed");capture.finish();}
            EXPECT_EQ(currentPartition(device),previous_partition);
#ifdef HAVE_ROCM
            if(device.is_rocm())
            {const auto leaf=rocm::lastNativeQ8SerialLaunch();ASSERT_TRUE(leaf);EXPECT_EQ(leaf->physical_n,N);
             EXPECT_EQ(leaf->arithmetic_policy_n,serial_width);EXPECT_EQ(leaf->stream,stream);}
#endif
            checked(captured_head->instantiate() && captured_head->launch(),"Original captured head oracle failed");
        }
        else checked(stage->execute(context.get()),"Isolation original identity stage oracle failed");
        const std::vector<float> original_input(input->data(),input->data()+K);
        std::vector<std::uint32_t> expected(N),actual(N);
        Join oracle_observation_retirement{*backend,device,stream};
        checked(backend->deviceToHostOnStream(expected.data(),output->gpu_data_ptr(),output->size_bytes(),device.ordinal,stream),
            "Isolation original output observation failed");
        checked(backend->synchronizeStream(stream,device.ordinal),"Isolation original terminal observation failed");
        std::vector<IsolationWorkspaceBaseline> scratch;
        std::vector<std::shared_ptr<FP32Tensor>> snapshot_owners;
        std::vector<WorkspaceDescriptor> snapshot_descriptors;
        Join snapshot_retirement{*backend,device,stream};
        for(const auto &buffer:requirements.buffers)
        {
            if(!buffer.size_bytes)continue;
            auto tensor=std::make_shared<FP32Tensor>(std::vector<std::size_t>{(buffer.size_bytes+3)/4});
            Join snapshot_slot_retirement{*backend,device,stream};
            TransferEngine::prepareDeviceInput(tensor.get(),device,stream);
            scratch.push_back({buffer.name,{tensor,0,buffer.size_bytes}});snapshot_owners.push_back(tensor);snapshot_descriptors.push_back(buffer);
        }
        // A foreign outer diagnostic scope must neither replace the frozen
        // adapter policy nor leak after normal/exceptional RAII retirement.
        try
        {auto token=fixture.kernel->beginOutputPartitionEquivalenceScope(N,7);
         EXPECT_EQ(currentPartition(device),7);throw std::runtime_error("controlled scope unwind");}
        catch(const std::runtime_error &){}
        EXPECT_EQ(currentPartition(device),previous_partition);
        auto outer=fixture.kernel->beginOutputPartitionEquivalenceScope(N,7);
        std::shared_ptr<const void> observation=baseline;
        PreparedGemmIsolationAdapter adapter(std::move(stage),{input,weight,output},store,workspace,arena,
            {M,N,K,0,N,0,1},{IsolationInputOrigin::SyntheticFixture,"synthetic-stage19351","prepared_input_A",1,observation},
            {baseline,0,input->size_bytes()},std::move(scratch),IsolationOutputPartitionPolicy{.serial_columns=serial_width});
        EXPECT_EQ(adapter.contract().outputPartitionPolicy().serial_columns,serial_width);
        EXPECT_EQ(currentPartition(device),7);
        auto restore=worker.createGraphCapture(stream),compute=worker.createGraphCapture(stream);
        ASSERT_NE(restore,nullptr);ASSERT_NE(compute,nullptr);
        Retire retire{*backend,device,stream,restore,compute};
        EXPECT_THROW(adapter.recordRestore(*backend,*restore,stream),std::logic_error);
        adapter.freezeCurrentInputs(*backend,stream);
        EXPECT_THROW(adapter.freezeCurrentInputs(*backend,stream),std::logic_error);
        adapter.prepareRecording(executor,context.get(),stream);
        EXPECT_EQ(currentPartition(device),7);
        checked(backend->synchronizeStream(stream,device.ordinal),"Isolation frozen input publication failed");
        ASSERT_FALSE(restore->hasExecutable());ASSERT_EQ(restore->nodeCount(),0u);
        ASSERT_FALSE(compute->hasExecutable());ASSERT_EQ(compute->nodeCount(),0u);
        {
            ScopedBackendGraphCapture recording(worker,*restore,"isolated resident reset");
            checked(recording.begin(),"Isolation reset capture begin failed");adapter.recordRestore(*backend,*restore,stream);recording.finish();
        }
        {
            ScopedBackendGraphCapture recording(worker,*compute,"actual canonical isolated prepared GEMM");
            checked(recording.begin(),"Isolation compute capture begin failed");adapter.recordCompute(executor,context.get(),*compute,stream);recording.finish();
        }
        EXPECT_EQ(currentPartition(device),7);
#ifdef HAVE_ROCM
        if(device.is_rocm())
        {const auto leaf=rocm::lastNativeQ8SerialLaunch();ASSERT_TRUE(leaf);EXPECT_EQ(leaf->physical_n,N);
         EXPECT_EQ(leaf->arithmetic_policy_n,serial_width?serial_width:N);EXPECT_EQ(leaf->stream,stream);}
#endif
        checked(restore->instantiate() && compute->instantiate(),"Isolation native graph instantiation failed");
        ASSERT_GT(compute->nodeCount(),0u);
        std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
        ASSERT_TRUE(compute->inspectKernelNodes(nodes,&error))<<error;
        ASSERT_FALSE(nodes.empty());
        bool native_projection=false;
        std::string kernel_names;
        for(const auto &node:nodes)
        {
            EXPECT_TRUE(node.name_resolved);
            kernel_names+=node.name+"\n";
            // CUDA and HIP retain distinct production entrypoint families.
            // A successful byte comparison alone must not bless another path.
            native_projection |= device.is_cuda()
                ? node.name.find("nativeVnniGemv_")!=std::string::npos ||
                    node.name.find("nativeVnniGemvTensorCoreSmallM")!=std::string::npos
                : node.name.find("gemv_native_vnni_kernel")!=std::string::npos;
            EXPECT_EQ(node.name.find("nccl"),std::string::npos);
            EXPECT_EQ(node.name.find("capturedTransfer"),std::string::npos);
        }
        EXPECT_TRUE(native_projection)<<"Canonical isolated stage did not retain a native projection:\n"<<kernel_names;
        // Graph role substitution is rejected before either original executable can enqueue.
        EXPECT_THROW(adapter.launchRestored(*compute,*restore,stream),std::logic_error);
        // Rebinding the canonical contract after capture must fail before either replay is submitted.
        ASSERT_TRUE(arena.bindExternalBuffer(BufferId::NORMALIZED,output.get()));
        EXPECT_THROW(adapter.launchRestored(*restore,*compute,stream),std::logic_error);
        ASSERT_TRUE(arena.bindExternalBuffer(BufferId::NORMALIZED,input.get()));
        std::vector<float> poison(K),restored(K);
        for(int replay=0;replay<20;++replay)
        {
            std::vector<std::uint32_t> output_poison(N,0x7fc00000u+unsigned(replay));
            Join replay_copy_retirement{*backend,device,stream};
            std::fill(poison.begin(),poison.end(),float(replay+11));
            checked(backend->hostToDeviceOnStream(input->gpu_data_ptr(),poison.data(),input->size_bytes(),device.ordinal,stream),
                "Isolation input poison failed");
            checked(backend->hostToDeviceOnStream(output->gpu_data_ptr(),output_poison.data(),output->size_bytes(),device.ordinal,stream),
                "Isolation output poison failed");
            for(const auto &buffer:requirements.buffers)
            {
                if(!buffer.size_bytes)continue;
                std::vector<std::byte> scratch_poison(buffer.size_bytes,std::byte(0xa5));
                Join scratch_copy_retirement{*backend,device,stream};
                checked(backend->hostToDeviceOnStream(workspace->getBuffer(buffer.name),scratch_poison.data(),
                    scratch_poison.size(),device.ordinal,stream),"Isolation scratch poison failed");
                // Copy completion must precede temporary host poison retirement.
                checked(backend->synchronizeStream(stream,device.ordinal),"Isolation test poison upload retirement failed");
            }
            checked(restore->launch(),"Isolation reset byte witness launch failed");
            checked(backend->deviceToHostOnStream(restored.data(),input->gpu_data_ptr(),input->size_bytes(),device.ordinal,stream),
                "Isolation reset input observation failed");
            checked(backend->synchronizeStream(stream,device.ordinal),"Isolation reset observation retirement failed");
            EXPECT_EQ(std::memcmp(restored.data(),original_input.data(),input->size_bytes()),0);
            for(std::size_t index=0;index<snapshot_owners.size();++index)
            {
                const auto &buffer=snapshot_descriptors[index];
                std::vector<std::byte> expected_scratch(buffer.size_bytes),actual_scratch(buffer.size_bytes);
                Join scratch_observation_retirement{*backend,device,stream};
                checked(backend->deviceToHostOnStream(expected_scratch.data(),snapshot_owners[index]->gpu_data_ptr(),buffer.size_bytes,device.ordinal,stream) &&
                    backend->deviceToHostOnStream(actual_scratch.data(),workspace->getBuffer(buffer.name),buffer.size_bytes,device.ordinal,stream),
                    "Isolation scratch reset byte observation failed");
                checked(backend->synchronizeStream(stream,device.ordinal),"Isolation scratch byte observation retirement failed");
                EXPECT_EQ(actual_scratch,expected_scratch)<<buffer.name<<" replay="<<replay;
            }
            adapter.launchRestored(*restore,*compute,stream);
            checked(backend->deviceToHostOnStream(actual.data(),output->gpu_data_ptr(),output->size_bytes(),device.ordinal,stream) &&
                backend->deviceToHostOnStream(restored.data(),input->gpu_data_ptr(),input->size_bytes(),device.ordinal,stream),
                "Isolation terminal byte observation failed");
            checked(backend->synchronizeStream(stream,device.ordinal),"Isolation terminal replay observation failed");
            EXPECT_EQ(actual,expected)<<"replay="<<replay;
            EXPECT_EQ(std::memcmp(restored.data(),original_input.data(),input->size_bytes()),0)<<"A baseline replay="<<replay;
        }
        outer.reset();EXPECT_EQ(currentPartition(device),previous_partition);
        // Retire joins/destroys graphs before adapter then source/store/workspace lifetime ends.
    });
}
}
#ifdef HAVE_CUDA
TEST(CapturedComputeIsolationNative,CUDA){prove(DeviceId::cuda(0));}
TEST(CapturedComputeIsolationNative,CUDAOriginalPartitionPolicy){prove(DeviceId::cuda(0),64);}
#endif
#ifdef HAVE_ROCM
TEST(CapturedComputeIsolationNative,ROCm){prove(DeviceId::rocm(0));}
TEST(CapturedComputeIsolationNative,ROCmOriginalPartitionPolicy){prove(DeviceId::rocm(0),64);}
#endif
