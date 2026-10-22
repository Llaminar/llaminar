/**
 * @file Test__CapturedComputeIsolation.cpp
 * @brief Real prepared stage recording restores resident inputs before twenty poisoned replays.
 *
 * This explicit synthetic fixture proves the concrete adapter/canonical runner,
 * not a real-model snapshot or TP scaling result. The complete GDN cases retain
 * all four production engines, original modulo-linked local source rows and
 * a separately captured complete-stage oracle at M1 and M9. Exact source preparation and
 * stage workspace are ordinary production APIs. Native graphs retire before
 * their tensor/store/workspace owners; all host observations are test terminal
 * boundaries. No collectives, sampler or generative state progress is enabled.
 */
#include "execution/diagnostics/PreparedGemmIsolationAdapter.h"
#include "execution/diagnostics/PreparedGDNIsolationAdapter.h"
#include "loaders/PreparedHalfWeights.h"
#include "config/GDNHeadAssignment.h"
#include "../../../performance/kernels/recorded/RecordedProjectionRuntime.h"
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
#include <array>
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

namespace
{
/**
 * @brief Pack synthetic original source rows without decoding or requantizing them.
 * @param full Original native matrix shared by every local-shape case.
 * @param spans Canonical modulo-linked global row intervals in local packed order.
 * @return Exact Q8 source blocks or exact F32 control words in the selected rows.
 * @note This is fixture construction only; production recorded work uses WeightManager.
 */
std::shared_ptr<TensorBase> packGDNFixtureRows(const TensorBase &full,const std::vector<GDNHeadSpan> &spans)
{
    const auto columns=full.cols();std::size_t rows=0;
    for(const auto span:spans)
    {
        checked(span.start>=0 && span.count>0 && std::size_t(span.end())<=full.rows(),"Synthetic GDN row span is outside its original source");
        rows+=span.count;
    }
    checked(full.native_type()==TensorType::Q8_0 || full.native_type()==TensorType::FP32,"Synthetic GDN source has an unsupported native type");
    const auto row_bytes=full.native_type()==TensorType::Q8_0?(columns/32)*sizeof(Q8_0Block):columns*sizeof(float);
    checked(full.raw_data() && full.size_bytes()==full.rows()*row_bytes,"Synthetic GDN original source has an incomplete native row layout");
    std::vector<std::uint8_t> raw(rows*row_bytes);std::size_t destination=0;
    for(const auto span:spans)
    {
        const auto bytes=std::size_t(span.count)*row_bytes;
        std::memcpy(raw.data()+destination,static_cast<const std::uint8_t *>(full.raw_data())+std::size_t(span.start)*row_bytes,bytes);
        destination+=bytes;
    }
    if(full.native_type()==TensorType::Q8_0)return std::make_shared<Q8_0Tensor>(std::vector<std::size_t>{rows,columns},raw);
    auto packed=std::make_shared<FP32Tensor>(std::vector<std::size_t>{rows,columns});
    std::memcpy(packed->mutable_data(),raw.data(),raw.size());return packed;
}
/**
 * @brief Prove the concrete full-bundle adapter on one exact local production-shaped source slice.
 * @param device Actual single GPU owning every native stage/stream/prepared handle.
 * @param rows Original physical row count, either one or nine, with distinct row inputs.
 * @param degree Canonical original equal head-partition degree, one/two/four.
 * @param rank Canonical logical source participant; this is a local-shape correctness fixture.
 * @note The independent oracle is a separately captured original complete GDNProjectionStage
 *       with the same local source arithmetic, not a serial-row loop or a timing label.
 */
void proveGDNShape(DeviceId device,int rows,int degree,int rank)
{
    SCOPED_TRACE(::testing::Message()<<device.toString()<<" M="<<rows<<" degree="<<degree<<" rank="<<rank);
    auto *backend=getBackendFor(device);ASSERT_NE(backend,nullptr);ASSERT_GT(backend->deviceCount(),device.ordinal);
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]
    {
        constexpr int K=640,key_heads=4,value_heads=12,state=32;
        const auto ownership=GDNHeadAssignment::forEqualRank(key_heads,value_heads,rank,degree);
        const auto key=ownership.keyElementSpan(state);
        std::array<std::vector<GDNHeadSpan>,4> spans;
        spans[0]={key,{key_heads*state+key.start,key.count}};
        for(auto value:ownership.valueElementSpans(state)){value.start+=2*key_heads*state;spans[0].push_back(value);}
        spans[1]=ownership.valueElementSpans(state);spans[2]=ownership.valueElementSpans(1);spans[3]=spans[2];
        const std::array<int,4> global_widths={(2*key_heads+value_heads)*state,value_heads*state,value_heads,value_heads};
        std::array<std::shared_ptr<TensorBase>,4> full,source,prepared;
        std::array<int,4> widths{};
        for(std::size_t i=0;i<4;++i)
        {
            if(i<2)full[i]=std::shared_ptr<Q8_0Tensor>(test::TestTensorFactory::createQ8_0Random({std::size_t(global_widths[i]),K},unsigned(19351+i)));
            else
            {
                auto control=std::make_shared<FP32Tensor>(std::vector<std::size_t>{std::size_t(global_widths[i]),K});
                for(std::size_t j=0;j<control->numel();++j)control->mutable_data()[j]=float(int((j*11+i*7)%41)-20)*.0078125f;
                full[i]=std::move(control);
            }
            source[i]=packGDNFixtureRows(*full[i],spans[i]);
            widths[i]=int(source[i]->rows());prepared[i]=i<2?source[i]:materializePreparedHalfWeights(*source[i]);
        }
        auto context=IDeviceContext::create(device,1);auto *stream=worker.defaultStream();
        ASSERT_GT(reinterpret_cast<std::uintptr_t>(stream),2u);
        auto input=std::make_shared<FP32Tensor>(std::vector<std::size_t>{std::size_t(rows),K});
        auto baseline=std::make_shared<FP32Tensor>(input->shape());
        std::array<std::shared_ptr<FP32Tensor>,4> outputs;
        for(std::size_t i=0;i<4;++i)outputs[i]=std::make_shared<FP32Tensor>(std::vector<std::size_t>{std::size_t(rows),std::size_t(widths[i])});
        for(std::size_t i=0;i<input->numel();++i)input->mutable_data()[i]=float(int((i*7+i/K*13)%53)-26)*.015625f;
        std::memcpy(baseline->mutable_data(),input->data(),input->size_bytes());
        std::vector<std::uint32_t> input_words(input->numel());
        std::memcpy(input_words.data(),input->data(),input->size_bytes());
        Join input_retirement{*backend,device,stream};
        TransferEngine::prepareDeviceInput(input.get(),device,stream);TransferEngine::prepareDeviceInput(baseline.get(),device,stream);
        for(const auto &output:outputs)TransferEngine::prepareDeviceInput(output.get(),device,stream);
        std::vector<std::shared_ptr<LoadOrchestrator>> pool_lifetimes;
        auto store=std::make_shared<PreparedWeightStore>(ModelContextId{88196});
        std::array<PreparedWeightRef,4> refs;
        const std::array<const char *,4> names={"blk.0.attn_qkv.weight","blk.0.attn_gate.weight","blk.0.ssm_alpha.weight","blk.0.ssm_beta.weight"};
        for(std::size_t i=0;i<4;++i)
            refs[i]=i<2?test::registerGpuPreparedGemmInStore(prepared[i].get(),device,names[i],store->modelId(),store.get(),&pool_lifetimes):
                test::registerGpuPreparedFloatingPointGemmInStore(prepared[i].get(),device,names[i],*store,pool_lifetimes,FloatingActivationOperandPrecision::FP16RNE);
        const std::array<BufferId,4> ids={BufferId::GDN_QKV,BufferId::GDN_Z,BufferId::GDN_ALPHA,BufferId::GDN_BETA};
        const auto make_stage=[&]
        {
            auto stage=std::make_unique<GDNProjectionStage>(GDNProjectionStage::Params{.device_id=device,.input=input.get(),.m=rows,.k=K,
                .w_qkv=prepared[0].get(),.output_qkv=outputs[0].get(),.n_qkv=widths[0],
                .w_z=prepared[1].get(),.output_z=outputs[1].get(),.n_z=widths[1],
                .w_a=prepared[2].get(),.output_a=outputs[2].get(),.n_a=widths[2],
                .w_b=prepared[3].get(),.output_b=outputs[3].get(),.n_b=widths[3],
                .input_buffer_id=BufferId::NORMALIZED,.output_qkv_buffer_id=ids[0],.output_z_buffer_id=ids[1],.output_a_buffer_id=ids[2],.output_b_buffer_id=ids[3],
                .prepared_ref_qkv=refs[0],.prepared_ref_z=refs[1],.prepared_ref_a=refs[2],.prepared_ref_b=refs[3],.prepared_store=store.get()});
            stage->setGPUStream(stream);return stage;
        };
        auto original=make_stage();const auto requirements=original->getWorkspaceRequirements(rows,0,K);
        const auto bytes=requirements.total_bytes_with_alignment();ASSERT_GT(bytes,0u);
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0,.device=device,.total_bytes=bytes,.admission_available_bytes=bytes},PhysicalMemoryOwner::ExecutionWorkspace,bytes);
        auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        auto workspace=std::make_shared<DeviceWorkspaceManager>(device,bytes,memory);
        checked(workspace->allocate(requirements),"Synthetic GDN exact workspace admission failed");original->bindWorkspace(workspace.get());
        Join workspace_retirement{*backend,device,stream};
        BufferArena arena;checked(arena.registerExternalBuffer(BufferId::NORMALIZED,input.get()),"Synthetic GDN input arena binding failed");
        for(std::size_t i=0;i<4;++i)checked(arena.registerExternalBuffer(ids[i],outputs[i].get()),"Synthetic GDN output arena binding failed");
        DeviceGraphExecutor executor;executor.setArena(&arena);
        // Provision the canonical backend launch resources without model arithmetic.
        // This oracle must be capture-ready even in a fresh backend test process.
        checked(original->prepareGraphLaunch(context.get(),stream),"Synthetic GDN original launch preparation failed");
        ComputeGraph original_graph;original_graph.addNode("original_complete_synthetic_gdn",std::move(original),device);
        std::array<std::vector<std::uint32_t>,4> expected,actual;
        for(std::size_t i=0;i<4;++i){expected[i].resize(outputs[i]->numel());actual[i].resize(expected[i].size());}
        std::vector<std::uint32_t> restored(input_words.size()),input_poison(input_words.size());
        std::vector<std::shared_ptr<FP32Tensor>> scratch_owners;
        std::vector<IsolationWorkspaceBaseline> scratch;
        std::vector<WorkspaceDescriptor> scratch_descriptors;
        std::unique_ptr<PreparedGDNIsolationAdapter> adapter;
        auto oracle=worker.createGraphCapture(stream);
        std::unique_ptr<IGPUGraphCapture> restore,compute,parent;
        Retire adapter_retirement{*backend,device,stream,restore,compute};
        // Retire the composed parent before either retained child executable.
        Retire original_retirement{*backend,device,stream,oracle,parent};
        ASSERT_NE(oracle,nullptr);
        checked(executor.prepareGraphStorageForCapture(original_graph,context.get(),stream,"original complete GDN oracle"),"Synthetic GDN oracle frontier failed");
        {ScopedBackendGraphCapture capture(worker,*oracle,"original complete synthetic GDN");checked(capture.begin(),"Synthetic GDN original capture failed");
         checked(executor.recordSetupGraphMaterialization(original_graph,context.get()),"Synthetic GDN original stage recording failed");capture.finish();}
        checked(oracle->instantiate() && oracle->launch(),"Synthetic GDN original captured oracle failed");
        for(std::size_t i=0;i<4;++i)checked(backend->deviceToHostOnStream(expected[i].data(),outputs[i]->gpu_data_ptr(),outputs[i]->size_bytes(),device.ordinal,stream),"Synthetic GDN original output observation failed");
        checked(backend->synchronizeStream(stream,device.ordinal),"Synthetic GDN original oracle observation did not retire");
        auto stage=make_stage();stage->bindWorkspace(workspace.get());
        for(const auto &buffer:requirements.buffers)if(buffer.size_bytes)
        {
            auto tensor=std::make_shared<FP32Tensor>(std::vector<std::size_t>{(buffer.size_bytes+3)/4});
            Join slot_retirement{*backend,device,stream};TransferEngine::prepareDeviceInput(tensor.get(),device,stream);
            scratch.push_back({buffer.name,{tensor,0,buffer.size_bytes}});scratch_owners.push_back(tensor);scratch_descriptors.push_back(buffer);
        }
        const std::array<std::shared_ptr<ITensor>,4> output_owners={outputs[0],outputs[1],outputs[2],outputs[3]};
        const std::array<std::shared_ptr<ITensor>,4> weight_owners={prepared[0],prepared[1],prepared[2],prepared[3]};
        const auto reject_source=[&](std::array<std::shared_ptr<ITensor>,4> malformed)
        {
            auto rejected=make_stage();rejected->bindWorkspace(workspace.get());
            PreparedGDNIsolationAdapter invalid(std::move(rejected),input,output_owners,std::move(malformed),store,workspace,arena,
                IsolationInputProvenance{IsolationInputOrigin::SyntheticFixture,"synthetic-gdn19351","synthetic_original_normalized_rows",1,baseline},
                IsolationResidentSlot{baseline,0,baseline->size_bytes()},scratch);
        };
        auto malformed=weight_owners;malformed[3].reset();EXPECT_THROW(reject_source(malformed),std::invalid_argument);
        malformed=weight_owners;std::swap(malformed[0],malformed[1]);EXPECT_THROW(reject_source(malformed),std::invalid_argument);
        const auto reject_ref=[&](bool missing)
        {
            // Keep the actual already-bound cached engines while deliberately
            // corrupting the authoritative stage ref. A cached pointer must
            // never make an absent or foreign prepared source admissible.
            auto malformed_params=stage->getParams();
            if(missing)malformed_params.prepared_ref_b.reset();
            else malformed_params.prepared_ref_b=refs[2];
            auto rejected=std::make_unique<GDNProjectionStage>(malformed_params);rejected->bindWorkspace(workspace.get());
            PreparedGDNIsolationAdapter invalid(std::move(rejected),input,output_owners,weight_owners,store,workspace,arena,
                IsolationInputProvenance{IsolationInputOrigin::SyntheticFixture,"synthetic-gdn19351","synthetic_original_normalized_rows",1,baseline},
                IsolationResidentSlot{baseline,0,baseline->size_bytes()},scratch);
        };
        EXPECT_THROW(reject_ref(true),std::invalid_argument);
        EXPECT_THROW(reject_ref(false),std::invalid_argument);
        adapter=std::make_unique<PreparedGDNIsolationAdapter>(std::move(stage),input,
            std::array<std::shared_ptr<ITensor>,4>{outputs[0],outputs[1],outputs[2],outputs[3]},
            std::array<std::shared_ptr<ITensor>,4>{prepared[0],prepared[1],prepared[2],prepared[3]},store,workspace,arena,
            IsolationInputProvenance{IsolationInputOrigin::SyntheticFixture,"synthetic-gdn19351","synthetic_original_normalized_rows",1,baseline},
            IsolationResidentSlot{baseline,0,baseline->size_bytes()},std::move(scratch));
        restore=worker.createGraphCapture(stream);compute=worker.createGraphCapture(stream);ASSERT_NE(restore,nullptr);ASSERT_NE(compute,nullptr);
        EXPECT_THROW(adapter->recordRestore(*backend,*restore,stream),std::logic_error);
        adapter->freezeScratchBaselines(*backend,stream);EXPECT_THROW(adapter->freezeScratchBaselines(*backend,stream),std::logic_error);
        adapter->prepareRecording(executor,context.get(),stream);
        {ScopedBackendGraphCapture capture(worker,*restore,"synthetic GDN immutable reset");checked(capture.begin(),"Synthetic GDN reset capture failed");adapter->recordRestore(*backend,*restore,stream);capture.finish();}
        {ScopedBackendGraphCapture capture(worker,*compute,"synthetic GDN actual isolated stage");checked(capture.begin(),"Synthetic GDN compute capture failed");adapter->recordCompute(executor,context.get(),*compute,stream);capture.finish();}
        checked(restore->instantiate() && compute->instantiate(),"Synthetic GDN reset/compute instantiation failed");
        namespace runtime=llaminar2::test::recorded_runtime;
        const auto original_nodes=runtime::inventory(*oracle,device,runtime::InventoryPolicy::PreparedStageBundle);
        const auto isolated_nodes=runtime::inventory(*compute,device,runtime::InventoryPolicy::PreparedStageBundle);
        EXPECT_TRUE(runtime::sameKernels(original_nodes,isolated_nodes));
        parent=worker.createGraphCapture(stream);ASSERT_NE(parent,nullptr);
        const std::array<GPUOrderedTimelineStep,2> steps{{
            {.name="recorded_restore",.kind=GPUOrderedTimelineStepKind::CapturedFragment,.capture=restore.get()},
            {.name="recorded_projection",.kind=GPUOrderedTimelineStepKind::CapturedFragment,.capture=compute.get()}}};
        checked(parent->buildOrderedTimelineTransaction(steps,GPUOrderedTimelineInstrumentation::PerStepEvents) && parent->instantiate(),"Synthetic GDN composed native transaction failed");
        EXPECT_TRUE(runtime::sameKernels(isolated_nodes,runtime::inventory(*parent,device,runtime::InventoryPolicy::PreparedStageBundle)));
        EXPECT_THROW(adapter->launchRestored(*compute,*restore,stream),std::logic_error);
        EXPECT_THROW(adapter->validateRecordedReplay(*restore,*compute,nullptr),std::logic_error);
        for(std::size_t i=0;i<4;++i)
        {
            ASSERT_TRUE(arena.bindExternalBuffer(ids[i],input.get()));EXPECT_THROW(adapter->validateRecordedReplay(*restore,*compute,stream),std::logic_error);
            ASSERT_TRUE(arena.bindExternalBuffer(ids[i],outputs[i].get()));
        }
        for(int replay=0;replay<20;++replay)
        {
            Join replay_retirement{*backend,device,stream};
            std::fill(input_poison.begin(),input_poison.end(),0x3f800000u+unsigned(replay));
            checked(backend->hostToDeviceOnStream(input->gpu_data_ptr(),input_poison.data(),input->size_bytes(),device.ordinal,stream),"Synthetic GDN input poison failed");
            for(std::size_t i=0;i<4;++i)
            {std::fill(actual[i].begin(),actual[i].end(),0x7fc00000u+unsigned(replay));checked(backend->hostToDeviceOnStream(outputs[i]->gpu_data_ptr(),actual[i].data(),outputs[i]->size_bytes(),device.ordinal,stream),"Synthetic GDN output poison failed");}
            for(const auto &buffer:scratch_descriptors)
            {
                std::vector<std::byte> poison(buffer.size_bytes,std::byte(0xa5));Join poison_retirement{*backend,device,stream};
                checked(backend->hostToDeviceOnStream(workspace->getBuffer(buffer.name),poison.data(),poison.size(),device.ordinal,stream),"Synthetic GDN named scratch poison failed");
                checked(backend->synchronizeStream(stream,device.ordinal),"Synthetic GDN poison source did not retire");
            }
            checked(restore->launch(),"Synthetic GDN reset witness launch failed");
            checked(backend->deviceToHostOnStream(restored.data(),input->gpu_data_ptr(),input->size_bytes(),device.ordinal,stream),"Synthetic GDN restored input observation failed");
            checked(backend->synchronizeStream(stream,device.ordinal),"Synthetic GDN restored input did not retire");EXPECT_EQ(restored,input_words);
            for(std::size_t i=0;i<scratch_descriptors.size();++i)
            {
                const auto &buffer=scratch_descriptors[i];std::vector<std::byte> expected_scratch(buffer.size_bytes),restored_scratch(buffer.size_bytes);
                Join scratch_observation{*backend,device,stream};
                checked(backend->deviceToHostOnStream(expected_scratch.data(),scratch_owners[i]->gpu_data_ptr(),buffer.size_bytes,device.ordinal,stream) &&
                    backend->deviceToHostOnStream(restored_scratch.data(),workspace->getBuffer(buffer.name),buffer.size_bytes,device.ordinal,stream),"Synthetic GDN scratch restoration observation failed");
                checked(backend->synchronizeStream(stream,device.ordinal),"Synthetic GDN scratch observation did not retire");EXPECT_EQ(restored_scratch,expected_scratch)<<buffer.name;
            }
            adapter->validateRecordedReplay(*restore,*compute,stream);
            if(replay%2==0)adapter->launchRestored(*restore,*compute,stream);
            else
            {
                checked(parent->launch(),"Synthetic GDN composed replay failed");
                // Consume the mandatory event receipt to validate ownership and
                // freshness only. No synthetic timing is reported or retained.
                (void)runtime::consumeTiming(*parent);
            }
            for(std::size_t i=0;i<4;++i)checked(backend->deviceToHostOnStream(actual[i].data(),outputs[i]->gpu_data_ptr(),outputs[i]->size_bytes(),device.ordinal,stream),"Synthetic GDN replay output observation failed");
            checked(backend->deviceToHostOnStream(restored.data(),input->gpu_data_ptr(),input->size_bytes(),device.ordinal,stream),"Synthetic GDN replay input observation failed");
            checked(backend->synchronizeStream(stream,device.ordinal),"Synthetic GDN replay observation did not retire");
            for(std::size_t i=0;i<4;++i)EXPECT_EQ(actual[i],expected[i])<<"projection="<<i<<" replay="<<replay;
            EXPECT_EQ(restored,input_words)<<"replay="<<replay;
        }
    });
}
/** @brief Cover every canonical local participant and both original synthetic row geometries. */
void proveGDN(DeviceId device)
{for(int rows:{1,9})for(int degree:{1,2,4})for(int rank=0;rank<degree;++rank)proveGDNShape(device,rows,degree,rank);}
} // namespace
#ifdef HAVE_CUDA
TEST(CapturedGDNIsolationNative,CUDA){proveGDN(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(CapturedGDNIsolationNative,ROCm){proveGDN(DeviceId::rocm(0));}
#endif
