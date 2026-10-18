/**
 * @file IQ4XSFusedMoEGpuSourceWitness.h
 * @brief Captured fixed-route GPU MoE transactions versus original IQ4_XS factors.
 *
 * Only terminal test observations copy live data to host. The exact production
 * repacker, descriptor tables, FromRouting fused call, workspace and graph API
 * execute unchanged. No packed factor or backend projection produces the oracle.
 */
#pragma once
#include "IQ4XSFusedMoESourceWitness.h"
#include "IQ4XSOriginalEquationSource.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "GpuPreparedGemmHarness.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "transfer/TransferEngine.h"
#include "utils/PerfStatsCollector.h"
#include <gtest/gtest.h>
#include <stdexcept>
#ifdef HAVE_CUDA
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#endif
#include <bit>
#include <span>

namespace llaminar2::test::iq4xs_fused_source_witness
{
/** @brief Stable graph/kernel/prepared/tensor owners joined before member teardown. */
struct GpuOwners
{
    IBackend *backend;
    void *stream;
    DeviceId device;
    Sources weights;
    std::vector<GpuPreparedGemm> prepared;
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    std::unique_ptr<IMoEKernel> kernel;
    std::unique_ptr<FP32Tensor> input,ids,route_weights,output;
    std::unique_ptr<IGPUGraphCapture> graph;
    /** @brief Retire even failed assertions while all DMA/graph users still live. */
    ~GpuOwners()
    {
        if(stream&&backend) (void)backend->synchronizeStream(stream,device.ordinal);
        if(kernel) if(auto *c=dynamic_cast<IWorkspaceConsumer*>(kernel.get())) c->bindWorkspace(nullptr);
    }
};
/**
 * @brief Read one live terminal field without outliving its staging owner.
 * @tparam T Observed element type.
 * @tparam Owners Retained owner with exact backend, stream and device fields.
 * @param owners Exact backend/stream and live graph owners.
 * @param pointer Device field from the actual workspace or tensor.
 * @param count Exact live element count, never capacity.
 * @return Joined terminal bytes.
 */
template<class T,class Owners>
std::vector<T> terminal(Owners &owners,const void *pointer,size_t count)
{
    std::vector<T> values(count);
    struct Join
    {
        Owners &o; bool armed=true;
        ~Join(){if(armed)(void)o.backend->synchronizeStream(o.stream,o.device.ordinal);}
    } join{owners};
    if(!pointer||!owners.backend->deviceToHostOnStream(values.data(),pointer,
        count*sizeof(T),owners.device.ordinal,owners.stream)||
        !owners.backend->synchronizeStream(owners.stream,owners.device.ordinal))
        throw std::runtime_error("IQ4_XS fused witness terminal publication failed");
    join.armed=false;
    return values;
}
/**
 * @brief Execute source witnesses through the real captured GPU fused transaction.
 * @param device Explicit CUDA or ROCm endpoint selected by its registered test.
 * @param roles Original source roles to exercise, including explicit mixed pairs.
 */
inline void runGpu(DeviceId device,std::span<const Role> roles=ordinaryRoles)
{
    auto &context=GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]{
        auto *backend=getBackendFor(device);
        ASSERT_NE(backend,nullptr);
        void *stream=context.getOrCreateAuxiliaryStream("iq4xs.original_source.fused_moe");
        ASSERT_NE(stream,nullptr);
        for(const Role role:roles)
        {
            SCOPED_TRACE(int(role));
            // Every source role must supply its own real construction evidence.
            PerfStatsCollector::reset();
            GpuOwners o{backend,stream,device};
            o.weights=sources(role);
            o.prepared.reserve(3);
            std::array<DeviceNativeVNNIMatrixDesc,3> desc{};
            const std::array<TensorBase*,3> raw{o.weights.gate.get(),o.weights.up.get(),o.weights.down.get()};
            std::array<std::vector<uint8_t>,3> original_bytes;
            for(int p=0;p<3;++p)
            {
                const auto *start=static_cast<const uint8_t*>(raw[p]->raw_data());
                original_bytes[p].assign(start,start+raw[p]->size_bytes());
                const std::string name="iq4xs.original_source.fused."+std::to_string(int(role))+"."+std::to_string(p);
                o.prepared.push_back(makeGpuPreparedGemm(raw[p],device,name,
                    ModelContextId{uint64_t(5990000+int(role)*16+p+(device.is_cuda()?0:1000))}));
                ASSERT_NE(o.prepared.back().kernel,nullptr);
                if(auto *tensor_kernel=dynamic_cast<ITensorKernel*>(o.prepared.back().kernel))
                    tensor_kernel->setGPUStream(stream);
                ASSERT_TRUE(o.prepared.back().kernel->exportNativeVNNIMatrixDesc(desc[p]));
                ASSERT_EQ(desc[p].n,width); ASSERT_EQ(desc[p].k,width);
            }
            if(mixedPair(role))
            {
                ASSERT_NE(desc[0].codebook_id,desc[1].codebook_id);
                ASSERT_EQ(desc[gateTarget(role)?1:0].source_is_superblock,0);
            }
            else ASSERT_EQ(desc[0].codebook_id,desc[1].codebook_id);
            const int target=gateTarget(role)?0:upTarget(role)?1:2;
            ASSERT_EQ(desc[target].source_identity_present,1);
            ASSERT_EQ(desc[target].source_codebook_id,4);
            ASSERT_EQ(desc[target].source_is_superblock,role==Role::IQ4NLDown?0:1);
            const auto req=device.is_cuda()?MoEWorkspaceBuffers::cudaMoE(1,width,width,1,1):
                MoEWorkspaceBuffers::rocmMoE(1,width,width,1,1);
            o.workspace=std::make_unique<DeviceWorkspaceManager>(device,req.total_bytes_with_alignment());
            ASSERT_TRUE(o.workspace->allocate(req));
            o.kernel=llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
            ASSERT_NE(o.kernel,nullptr);
            o.kernel->setGPUStream(stream);
            auto *consumer=dynamic_cast<IWorkspaceConsumer*>(o.kernel.get());
            ASSERT_NE(consumer,nullptr); consumer->bindWorkspace(o.workspace.get());
            const int gate_table=o.kernel->uploadGroupedExpertGateUpDescriptorTables(
                &desc[0],&desc[1],1,width,width);
            const int down_table=o.kernel->uploadGroupedExpertDownDescriptorTable(&desc[2],1,width,width);
            ASSERT_GE(gate_table,0); ASSERT_GE(down_table,0);
            const auto x=hidden();
            o.input=std::make_unique<FP32Tensor>(std::vector<size_t>{1,width});
            o.ids=std::make_unique<FP32Tensor>(std::vector<size_t>{1,1});
            o.route_weights=std::make_unique<FP32Tensor>(std::vector<size_t>{1,1});
            o.output=std::make_unique<FP32Tensor>(std::vector<size_t>{1,width});
            std::copy(x.begin(),x.end(),o.input->mutable_data());
            o.ids->mutable_data()[0]=0.0f; o.route_weights->mutable_data()[0]=1.0f;
            std::fill_n(o.output->mutable_data(),width,0.0f);
            for(auto *t:{o.input.get(),o.ids.get(),o.route_weights.get(),o.output.get()})
                TransferEngine::prepareDeviceInput(t,device,stream);
            ASSERT_TRUE(o.kernel->prepareGroupedRuntimeDecodeLaunchState(gate_table,down_table,
                1,width,width,MoEDecodeDescriptorSource::StaticDescriptorTable));
            const auto body=[&]{return o.kernel->groupedExpertDecodeFromRouting(o.input.get(),
                o.ids.get(),o.route_weights.get(),gate_table,down_table,1,o.output.get(),
                width,width,nullptr,nullptr,nullptr,MoEDecodeDescriptorSource::StaticDescriptorTable);};
            // Pay immutable pointer/table setup outside capture; each scored replay
            // below restores every observed field to a distinct poison first.
            ASSERT_TRUE(body());
            ASSERT_TRUE(backend->synchronizeStream(stream,device.ordinal));
            for(auto *t:{o.input.get(),o.ids.get(),o.route_weights.get()})
                TransferEngine::requireDeviceInput(t,device,stream);
            o.graph=context.createGraphCapture(stream); ASSERT_NE(o.graph,nullptr);
            {
                ScopedBackendGraphCapture capture(context,*o.graph,"IQ4_XS original-source fused MoE");
                ASSERT_TRUE(capture.begin()); ASSERT_TRUE(body()); capture.finish();
            }
            ASSERT_TRUE(o.graph->instantiate());
            std::vector<GPUGraphKernelNodeInfo> nodes; std::string error;
            ASSERT_TRUE(o.graph->inspectKernelNodes(nodes,&error))<<error;
            ASSERT_FALSE(nodes.empty());
            std::array<float,width> expected_gate{},expected_up{};
            expectedGateUp(role,expected_gate,expected_up);
            const auto expected_middle=expectedMiddle(role);
            ASSERT_EQ(fp16_to_fp32(expected_middle[0].d),
                gateTarget(role)?3.12109375f:upTarget(role)?399.5f:128.0f);
            for(int replay=0;replay<20;++replay)
            {
                const auto poison=[&](void *p,int value,size_t bytes){
                    return p&&backend->memset(p,value,bytes,device.ordinal,stream);};
                ASSERT_TRUE(poison(o.output->gpu_data_ptr(),0xff,width*sizeof(float)));
                ASSERT_TRUE(poison(o.workspace->getBuffer(MoEWorkspaceBuffers::DECODE_HIDDEN_INT8),
                    replay&1?0x5a:0xa5,width));
                ASSERT_TRUE(poison(o.workspace->getBuffer(MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES),0xff,blocks*sizeof(float)));
                ASSERT_TRUE(poison(o.workspace->getBuffer(MoEWorkspaceBuffers::DECODE_SWIGLU_INT8),
                    replay&1?0xa5:0x5a,width));
                ASSERT_TRUE(poison(o.workspace->getBuffer(MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES),0xff,blocks*sizeof(float)));
                if(device.is_cuda())
                {
                    ASSERT_TRUE(poison(o.workspace->getBuffer(MoEWorkspaceBuffers::PREFILL_GATE),0xff,width*sizeof(float)));
                    ASSERT_TRUE(poison(o.workspace->getBuffer(MoEWorkspaceBuffers::PREFILL_UP),0xff,width*sizeof(float)));
                }
                ASSERT_TRUE(o.graph->launch());
                const auto actual_hidden=terminal<int8_t>(o,o.workspace->getBuffer(MoEWorkspaceBuffers::DECODE_HIDDEN_INT8),width);
                const auto hidden_scales=terminal<float>(o,o.workspace->getBuffer(MoEWorkspaceBuffers::DECODE_HIDDEN_SCALES),blocks);
                const auto actual_middle=terminal<int8_t>(o,o.workspace->getBuffer(MoEWorkspaceBuffers::DECODE_SWIGLU_INT8),width);
                const auto middle_scales=terminal<float>(o,o.workspace->getBuffer(MoEWorkspaceBuffers::DECODE_SWIGLU_SCALES),blocks);
                std::array<Q8_1Block,blocks> consumed{};
                for(int g=0;g<blocks;++g)
                {
                    EXPECT_EQ(hidden_scales[g],1.0f);
                    ASSERT_TRUE(std::isfinite(middle_scales[g])&&middle_scales[g]>0.0f);
                    consumed[g].d=fp32_to_fp16(middle_scales[g]);
                    EXPECT_EQ(fp16_to_fp32(consumed[g].d),middle_scales[g]);
                    EXPECT_EQ(middle_scales[g],fp16_to_fp32(expected_middle[g].d));
                    for(int i=0;i<32;++i)
                    {
                        EXPECT_EQ(actual_hidden[g*32+i],int(x[g*32+i]));
                        consumed[g].qs[i]=actual_middle[g*32+i];
                        consumed[g].sum_qs+=consumed[g].qs[i];
                        EXPECT_EQ(actual_middle[g*32+i],expected_middle[g].qs[i]);
                    }
                }
                if(device.is_cuda())
                {
                    const auto gate=terminal<float>(o,o.workspace->getBuffer(MoEWorkspaceBuffers::PREFILL_GATE),width);
                    const auto up=terminal<float>(o,o.workspace->getBuffer(MoEWorkspaceBuffers::PREFILL_UP),width);
                    EXPECT_EQ(std::memcmp(gate.data(),expected_gate.data(),width*sizeof(float)),0);
                    EXPECT_EQ(std::memcmp(up.data(),expected_up.data(),width*sizeof(float)),0);
                }
                // ROCm has no fused raw FP32 gate/up publication. Its actual
                // middle check above is the observable source-equation boundary.
                const auto output=terminal<float>(o,o.output->gpu_data_ptr(),width);
                const float expected=expectedDown(role,*o.weights.down,consumed.data());
                for(float value:output) EXPECT_EQ(value,expected);
                EXPECT_EQ(terminal<float>(o,o.ids->gpu_data_ptr(),1)[0],0.0f);
                EXPECT_EQ(terminal<float>(o,o.route_weights->gpu_data_ptr(),1)[0],1.0f);
            }
            for(int p=0;p<3;++p)
                EXPECT_EQ(std::memcmp(raw[p]->raw_data(),original_bytes[p].data(),original_bytes[p].size()),0);
            const std::string name=device.is_cuda()?"cuda_moe_grouped_decode_fused_calls":"rocm_moe_grouped_decode_fused_calls";
            const auto records=PerfStatsCollector::snapshot({"kernel."+name});
            EXPECT_TRUE(std::any_of(records.begin(),records.end(),[&](const PerfStatRecord &r){
                const auto source=r.tags.find("source"),slots=r.tags.find("active_slots"),
                    hidden=r.tags.find("d_model"),intermediate=r.tags.find("intermediate"),route=r.tags.find("route");
                return r.name==name&&r.value>0&&r.count>0&&source!=r.tags.end()&&source->second=="routing"&&
                    slots!=r.tags.end()&&slots->second=="1"&&
                    hidden!=r.tags.end()&&hidden->second=="256"&&
                    intermediate!=r.tags.end()&&intermediate->second=="256"&&
                    route!=r.tags.end()&&route->second==(device.is_cuda()?"fused_block_down":"ordered_route_parallel_down");
            }))<<"Actual optimized fused fixed-route host-construction proof absent";
        }
    });
}

/** @brief Join all dense replay/DMA users before member storage can retire. */
struct DenseOwners
{
    IBackend *backend;
    void *stream;
    DeviceId device;
    iq4xs_original_equation::Source weights;
    GpuPreparedGemm prepared;
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    std::unique_ptr<FP32Tensor> input,output;
    std::unique_ptr<IGPUGraphCapture> graph;
    std::vector<float> upload;
    /** @brief Join on failure as well as success before graph/host/device teardown. */
    ~DenseOwners()
    {
        if(stream&&backend)(void)backend->synchronizeStream(stream,device.ordinal);
        if(prepared.kernel)
        {
            if(auto *c=dynamic_cast<IWorkspaceConsumer*>(prepared.kernel))c->bindWorkspace(nullptr);
            prepared.kernel->clearGPUStreamBinding();
        }
    }
};
/** @brief Explicit test-only scope; ordinary default continues production dispatch. */
enum class DenseExecutionScope { Ordinary, VerifierDecodeEquivalent };
/** @brief Existing ordinary dense population; retained backing stays 512 rows. */
inline constexpr std::array<int,4> defaultDenseRows{1,3,8,512};
/**
 * @brief Real captured dense source equations with exact observed Q8 publications.
 * @param device Explicit CUDA or ROCm endpoint; ordinary generated dispatch stays active.
 * @param row_populations Exact positive rows per retained geometry, at most512.
 * @param execution_scope Explicit verifier mode for separately named bypass diagnostics.
 * @details The IQ4_XS all-code/domain sweep is intentionally representation-neutral.
 * IQ4_NL exercises the shared legacy execution alias independently. Every graph
 * replays20changed blocks/signs, including32-value K tail and source-row view.
 */
inline void runDense(DeviceId device,
    std::span<const int> row_populations = defaultDenseRows,
    DenseExecutionScope execution_scope = DenseExecutionScope::Ordinary)
{
    ASSERT_FALSE(row_populations.empty());
    for (int rows : row_populations) ASSERT_TRUE(rows > 0 && rows <= 512);
    namespace eq=iq4xs_original_equation;
    auto &context=GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]{
        auto *backend=getBackendFor(device);ASSERT_NE(backend,nullptr);
        void *stream=context.getOrCreateAuxiliaryStream("iq4xs.original_source.dense");
        ASSERT_NE(stream,nullptr);
        for(auto format:{eq::Format::IQ4XS,eq::Format::IQ4NL})
        {
            DenseOwners o{backend,stream,device,eq::source(format)};
            o.prepared=makeGpuPreparedGemm(o.weights.view.get(),device,
                "iq4xs.original_source.dense."+std::to_string(int(format)),
                ModelContextId{uint64_t(5992000+int(format)+(device.is_cuda()?0:1000))});
            ASSERT_NE(o.prepared.kernel,nullptr);o.prepared.kernel->setGPUStream(stream);
#ifdef HAVE_CUDA
            if (device.is_cuda() && execution_scope == DenseExecutionScope::VerifierDecodeEquivalent)
            {
                // Explicit bypass diagnostics own their immutable ROWPAR view
                // before capture; DIRECT may reuse it but never rewrites it.
                auto *cuda_kernel = dynamic_cast<cuda::CUDAQuantisedGemmKernel *>(o.prepared.kernel);
                ASSERT_NE(cuda_kernel, nullptr);
                cuda_kernel->prepareRowMajorWeights(stream);
            }
#endif

            auto *consumer=dynamic_cast<IWorkspaceConsumer*>(o.prepared.kernel);ASSERT_NE(consumer,nullptr);
            WorkspaceRequirements requirements;
            for(int m:row_populations)requirements.merge(consumer->getWorkspaceRequirements(m,o.weights.n,eq::k));
            o.workspace=std::make_unique<DeviceWorkspaceManager>(device,requirements.total_bytes_with_alignment());
            ASSERT_TRUE(o.workspace->allocate(requirements));consumer->bindWorkspace(o.workspace.get());
            o.input=std::make_unique<FP32Tensor>(std::vector<size_t>{512,eq::k});
            o.output=std::make_unique<FP32Tensor>(std::vector<size_t>{512,size_t(o.weights.n)});
            std::fill_n(o.input->mutable_data(),512*eq::k,0.0f);
            TransferEngine::prepareDeviceInput(o.input.get(),device,stream);
            ASSERT_TRUE(o.output->allocateOnDevice(device,stream));
            o.upload.resize(512*eq::k);
            DeviceNativeVNNIMatrixDesc desc{};
            ASSERT_TRUE(o.prepared.kernel->exportNativeVNNIMatrixDesc(desc));
            ASSERT_EQ(desc.n,o.weights.n);ASSERT_EQ(desc.k,eq::k);
            ASSERT_EQ(desc.blocks_per_row,eq::k/32);
            ASSERT_EQ(desc.source_identity_present,1);
            ASSERT_EQ(desc.source_codebook_id,4);
            ASSERT_EQ(desc.source_is_superblock,format==eq::Format::IQ4XS?1:0);
            for(int m:row_populations)
            {
                // The scope is constructed on the actual context worker and
                // outlives setup, capture and every replay of this geometry.
                auto verifier = execution_scope == DenseExecutionScope::VerifierDecodeEquivalent
                    ? o.prepared.kernel->beginVerifierDecodeEquivalentScope() : nullptr;
                SCOPED_TRACE("format="+std::to_string(int(format))+" M="+std::to_string(m));
                // Pay immutable pointer/launch setup before each retained geometry.
                ASSERT_TRUE(o.prepared.kernel->multiply_tensor(o.input.get(),o.output.get(),m,o.weights.n,eq::k));
                ASSERT_TRUE(backend->synchronizeStream(stream,device.ordinal));
                TransferEngine::requireDeviceInput(o.input.get(),device,stream);
                o.graph=context.createGraphCapture(stream);ASSERT_NE(o.graph,nullptr);
                {
                    ScopedBackendGraphCapture capture(context,*o.graph,"IQ4_XS original source dense");
                    ASSERT_TRUE(capture.begin());
                    ASSERT_TRUE(o.prepared.kernel->multiply_tensor(o.input.get(),o.output.get(),m,o.weights.n,eq::k));
                    capture.finish();
                }
                ASSERT_TRUE(o.graph->instantiate());
                std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
                ASSERT_TRUE(o.graph->inspectKernelNodes(nodes,&error))<<error;ASSERT_FALSE(nodes.empty());
                for(int replay=0;replay<20;++replay)
                {
                    constexpr std::array<int,5> active{0,1,7,8,0};const int block=active[replay%active.size()];
                    std::fill(o.upload.begin(),o.upload.end(),0.0f);
                    for(int row=0;row<m;++row)
                    {
                        const int sign=(row+replay)&1?-1:1;
                        o.upload[row*eq::k+block*32]=float(sign);
                        o.upload[row*eq::k+block*32+31]=float(sign*127);
                    }
                    ASSERT_TRUE(backend->hostToDeviceOnStream(o.input->gpu_data_ptr(),o.upload.data(),m*eq::k*sizeof(float),device.ordinal,stream));
                    TransferEngine::publishCurrentDeviceWrite(o.input.get(),stream);
                    TransferEngine::requireDeviceInput(o.input.get(),device,stream);
                    ASSERT_TRUE(backend->memset(o.output->gpu_data_ptr(),0xff,m*o.weights.n*sizeof(float),device.ordinal,stream));
                    ASSERT_TRUE(o.graph->launch());
                    const auto actual=terminal<float>(o,o.output->gpu_data_ptr(),m*o.weights.n);
                    const auto codes=terminal<int8_t>(o,o.workspace->getBuffer(GemmWorkspaceBuffers::QUANT_A),m*eq::k);
                    const auto scales=terminal<float>(o,o.workspace->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE),m*(eq::k/32));
                    for(int row=0;row<m;++row)
                    {
                        const int sign=(row+replay)&1?-1:1;
                        for(int col=0;col<o.weights.n;++col)
                            EXPECT_EQ(std::bit_cast<uint32_t>(actual[row*o.weights.n+col]),
                                std::bit_cast<uint32_t>(eq::expected(o.weights,col,block,sign)))
                                <<"row="<<row<<" col="<<col<<" block="<<block;
                        for(int j=0;j<eq::k;++j)EXPECT_EQ(codes[row*eq::k+j],int8_t(o.upload[row*eq::k+j]));
                        for(int j=0;j<eq::k/32;++j)EXPECT_EQ(scales[row*(eq::k/32)+j],1.0f);
                    }
                }
                o.graph.reset();
            }
            EXPECT_EQ(std::memcmp(o.weights.parent->raw_data(),o.weights.bytes.data(),o.weights.bytes.size()),0);
        }
    });
}
} // namespace llaminar2::test::iq4xs_fused_source_witness
