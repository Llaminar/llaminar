/**
 * @file Test__ROCmMoEGroupedWmmaPromotion.cpp
 * @brief Captured physical23 matrix execution after genuine CPU expert promotion.
 *
 * The unchanged compact Q5_K Dot4 transaction is the arithmetic authority.
 * Production CPU packing and tier transfer supply the expanded candidate bytes;
 * retained WMMA graphs must reproduce both canonical routes and final outputs
 * across dirty full/partial/empty replays and replacement activation data.
 */
#include "../moe/NativeQuantisedExpertTransferParityTest.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedWmma.h"
#include <hip/hip_runtime.h>

namespace llaminar2::test
{
/** @test Preserve compact-source arithmetic through both captured expanded WMMA tiles. */
TEST(ROCmMoEGroupedWmmaPromotion, CapturedExpandedReplay)
{
    using namespace native_quantised_transfer_parity_detail;
    initializeROCmHardwareMetadata();
    int ordinal = -1;
    for (int i = 0; i < static_cast<int>(rocmHardwareMetadata().size()); ++i)
    {
        const auto &isa = rocmHardwareMetadata().at(i).isa_target;
        if (isa == "gfx1100" || isa.starts_with("gfx1100:")) { ordinal = i; break; }
    }
    if (ordinal < 0) GTEST_SKIP() << "Native gfx1100 is not visible";
    const auto device = DeviceId::rocm(ordinal);
    auto *backend = getBackendFor(device);
    ASSERT_NE(backend,nullptr);
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&] {
        constexpr int rows = 65, hidden_width = 2048, intermediate = 512, experts = 2, top_k = 2;
        void *stream = context.defaultStream();
        ASSERT_NE(stream,nullptr);
        const auto &format = quantizedVerifierFormat("Q5_K");
        std::vector<std::unique_ptr<TensorBase>> sources;
        std::vector<GpuPreparedGemm> prepared;
        std::array<DeviceNativeQuantisedMatrixDesc,3> compact{};
        std::array<cpu::native_quantised::CPUNativeQuantisedPackedWeights,3> cpu_weights;
        std::array<PromotedCPUProjection,3> promoted;
        constexpr std::array roles{ExpertTierWeightProjection::Gate,ExpertTierWeightProjection::Up,
                                   ExpertTierWeightProjection::Down};
        for (int role = 0; role < 3; ++role)
        {
            compact[role] = prepareMatrixDescriptor(format,device,stream,
                role == 2 ? hidden_width : intermediate,role == 2 ? intermediate : hidden_width,
                99281u + role,"test.rocm.wmma.promoted." + std::to_string(role),
                ModelContextId{2930000u + static_cast<unsigned>(role)},sources,prepared);
            ASSERT_EQ(compact[role].codebook_id,7u);
            ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(sources[role].get(),cpu_weights[role]));
            ASSERT_TRUE(cpu_weights[role].usesExpandedInt8());
            promoted[role] = promoteCPUProjection(backend,device,cpu_weights[role],roles[role],2930100u + role);
            const auto &desc = promoted[role].descriptor;
            ASSERT_EQ(desc.codebook_id,kNativeQuantisedExpandedInt8MinCodebook);
            ASSERT_TRUE(desc.source_identity_present);
            ASSERT_EQ(desc.source_codebook_id,7u);
            ASSERT_NE(desc.mins,nullptr);
        }
        auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        ASSERT_NE(kernel,nullptr);
        kernel->setGPUStream(stream);
        const auto requirements = MoEWorkspaceBuffers::rocmMoE(rows,hidden_width,intermediate,experts,top_k);
        DeviceWorkspaceManager workspace(device,requirements.total_bytes_with_alignment() + 4u*1024u*1024u);
        ASSERT_TRUE(workspace.allocate(requirements));
        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
        ASSERT_NE(consumer,nullptr);
        consumer->bindWorkspace(&workspace);
        std::array<int,2> gate_tables{},down_tables{};
        for (int storage = 0; storage < 2; ++storage)
        {
            std::array<DeviceNativeQuantisedMatrixDesc,experts> gates{},ups{},downs{};
            gates.fill(storage ? promoted[0].descriptor : compact[0]);
            ups.fill(storage ? promoted[1].descriptor : compact[1]);
            downs.fill(storage ? promoted[2].descriptor : compact[2]);
            gate_tables[storage] = kernel->uploadGroupedExpertGateUpDescriptorTables(
                gates.data(),ups.data(),experts,hidden_width,intermediate);
            down_tables[storage] = kernel->uploadGroupedExpertDownDescriptorTable(
                downs.data(),experts,hidden_width,intermediate);
            ASSERT_GE(gate_tables[storage],0);
            ASSERT_GE(down_tables[storage],0);
        }
        auto hidden = makeHidden(rows,hidden_width,77u,ExpertInputPattern::Scrambled);
        auto ids = TestTensorFactory::createFP32({rows,top_k});
        auto weights = TestTensorFactory::createFP32({rows,top_k});
        auto output = TestTensorFactory::createFP32({rows,hidden_width});
        auto canonical = TestTensorFactory::createFP32({rows,top_k,hidden_width});
        std::vector<float> full_ids(rows*top_k),route_weights(rows*top_k);
        for (int row = 0; row < rows; ++row)
            for (int slot = 0; slot < top_k; ++slot)
            {
                full_ids[row*top_k+slot] = float((row+slot)%experts);
                route_weights[row*top_k+slot] = slot ? 0.375f : 0.625f;
            }
        std::copy(full_ids.begin(),full_ids.end(),ids->mutable_data());
        std::copy(route_weights.begin(),route_weights.end(),weights->mutable_data());
        for (auto *tensor : std::array<TensorBase *,5>{hidden.get(),ids.get(),weights.get(),output.get(),canonical.get()})
            ASSERT_TRUE(tensor->ensureOnDevice(device,stream));
        std::array graphs{context.createGraphCapture(stream),context.createGraphCapture(stream),context.createGraphCapture(stream)};
        // Retire work before graph, tensor, workspace and promoted-byte owners.
        auto drain = [&](void *) { (void)backend->synchronizeStream(stream,ordinal); };
        std::unique_ptr<void,decltype(drain)> completion(&context,drain);
        const auto enqueue = [&](int storage) {
            return kernel->prepareExpertGroupsAsync(ids.get(),weights.get(),rows,experts,top_k) &&
                kernel->executeGroupedPrefillPipeline(hidden.get(),output.get(),gate_tables[storage],down_tables[storage],
                    rows,hidden_width,intermediate,experts,top_k,canonical.get()) &&
                kernel->reduceCanonicalRouteContributions(canonical.get(),output.get(),rows,top_k,hidden_width);
        };
        constexpr std::array engines{rocm::MoEGroupedPrefillEngine::OrderedDot4,
            rocm::MoEGroupedPrefillEngine::WmmaM16N32,rocm::MoEGroupedPrefillEngine::WmmaM16N64};
        std::array<std::vector<hipGraphNode_t>,3> native_nodes;
        for (int path = 0; path < 3; ++path)
        {
            rocm::ScopedMoEGroupedPrefillEngine engine(engines[path]);
            ASSERT_TRUE(enqueue(path != 0));
            ScopedBackendGraphCapture recording(context,*graphs[path],"promoted physical23 grouped WMMA");
            ASSERT_TRUE(recording.begin());
            ASSERT_TRUE(enqueue(path != 0));
            recording.finish();
            std::vector<GPUGraphKernelNodeInfo> nodes;
            ASSERT_TRUE(graphs[path]->inspectKernelNodes(nodes));
            int native_count = 0, gate_up_count = 0, down_count = 0;
            for (const auto &node : nodes)
                if (node.name.find("grouped_wmma_detail") != std::string::npos)
                {
                    ++native_count;
                    ASSERT_NE(node.name.find("NativeQuantisedGemmFormat23E"),std::string::npos) << node.name;
                    ASSERT_EQ(node.block_x,path == 1 ? 128u : 256u);
                    ASSERT_EQ(node.local_memory_bytes_per_thread,0u);
                    if (node.grid_z == 2) ++gate_up_count;
                    else if (node.grid_z == 1) ++down_count;
                }
            ASSERT_EQ(native_count,path ? 2 : 0);
            ASSERT_EQ(gate_up_count,path ? 1 : 0);
            ASSERT_EQ(down_count,path ? 1 : 0);
            ASSERT_TRUE(graphs[path]->instantiate());
            auto *native = dynamic_cast<HIPGraphCapture *>(graphs[path].get());
            ASSERT_NE(native,nullptr);
            size_t count = 0;
            ASSERT_EQ(hipGraphGetNodes(native->graph(),nullptr,&count),hipSuccess);
            std::vector<hipGraphNode_t> handles(count);
            ASSERT_EQ(hipGraphGetNodes(native->graph(),handles.data(),&count),hipSuccess);
            for (const auto handle : handles)
            {
                hipGraphNodeType type;
                ASSERT_EQ(hipGraphNodeGetType(handle,&type),hipSuccess);
                if (type != hipGraphNodeTypeKernel) continue;
                hipKernelNodeParams params{};
                ASSERT_EQ(hipGraphKernelNodeGetParams(handle,&params),hipSuccess);
                const char *name = hipKernelNameRefByPtr(params.func,static_cast<hipStream_t>(stream));
                ASSERT_NE(name,nullptr);
                if (std::string_view(name).find("grouped_wmma_detail") != std::string_view::npos)
                    native_nodes[path].push_back(handle);
            }
            ASSERT_EQ(native_nodes[path].size(),static_cast<size_t>(native_count));
        }
        std::vector<float> expected(rows*hidden_width),expected_routes(rows*top_k*hidden_width);
        std::vector<float> actual(expected.size()),actual_routes(expected_routes.size());
        const auto observe = [&](std::vector<float> &result,std::vector<float> &routes) {
            ASSERT_TRUE(backend->deviceToHost(result.data(),output->gpu_data_ptr(),result.size()*sizeof(float),ordinal,stream));
            ASSERT_TRUE(backend->deviceToHost(routes.data(),canonical->gpu_data_ptr(),routes.size()*sizeof(float),ordinal,stream));
            ASSERT_TRUE(backend->synchronizeStream(stream,ordinal));
        };
        const auto poison = [&] {
            for (const char *name : {MoEWorkspaceBuffers::PREFILL_A_INT8,MoEWorkspaceBuffers::PREFILL_A_SCALES,
                    MoEWorkspaceBuffers::PREFILL_GATE,MoEWorkspaceBuffers::PREFILL_UP,
                    MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8,MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES,
                    MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY})
            {
                ASSERT_NE(workspace.getBuffer(name),nullptr) << name;
                ASSERT_GT(workspace.getBufferSize(name),0u) << name;
                ASSERT_EQ(hipMemsetAsync(workspace.getBuffer(name),0xa5,workspace.getBufferSize(name),
                    static_cast<hipStream_t>(stream)),hipSuccess);
            }
            ASSERT_EQ(hipMemsetAsync(output->gpu_data_ptr(),0xa5,actual.size()*sizeof(float),static_cast<hipStream_t>(stream)),hipSuccess);
            ASSERT_EQ(hipMemsetAsync(canonical->gpu_data_ptr(),0xa5,actual_routes.size()*sizeof(float),static_cast<hipStream_t>(stream)),hipSuccess);
        };
        for (int generation = 0; generation < 2; ++generation)
        {
            auto next_hidden = makeHidden(rows,hidden_width,77u+generation,ExpertInputPattern::Scrambled);
            ASSERT_TRUE(backend->hostToDevice(hidden->gpu_data_ptr(),next_hidden->data(),hidden->numel()*sizeof(float),ordinal,stream));
            ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(),full_ids.data(),full_ids.size()*sizeof(float),ordinal,stream));
            ASSERT_TRUE(graphs[0]->launch());
            ASSERT_NO_FATAL_FAILURE(observe(expected,expected_routes));
            ASSERT_TRUE(std::any_of(expected.begin(),expected.end(),[](float value) { return value != 0.0f; }));
            // A recorded symbol is not proof of work. Independently omit each
            // native role, require failure against Dot4, then restore the node.
            if (generation == 0)
                for (int path = 1; path < 3; ++path)
                    for (const auto node : native_nodes[path])
                    {
                        auto *native = static_cast<HIPGraphCapture *>(graphs[path].get());
                        ASSERT_EQ(hipGraphNodeSetEnabled(native->executable(),node,0),hipSuccess);
                        ASSERT_NO_FATAL_FAILURE(poison());
                        ASSERT_TRUE(graphs[path]->launch());
                        ASSERT_NO_FATAL_FAILURE(observe(actual,actual_routes));
                        ASSERT_EQ(hipGraphNodeSetEnabled(native->executable(),node,1),hipSuccess);
                        ASSERT_NE(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0);
                        ASSERT_NE(std::memcmp(actual_routes.data(),expected_routes.data(),actual_routes.size()*sizeof(float)),0);
                    }
            for (int live : {rows,16,1,0,rows})
            {
                auto replay_ids = full_ids,reference = expected,reference_routes = expected_routes;
                std::fill(replay_ids.begin()+live*top_k,replay_ids.end(),-1.0f);
                std::fill(reference.begin()+live*hidden_width,reference.end(),0.0f);
                std::fill(reference_routes.begin()+live*top_k*hidden_width,reference_routes.end(),0.0f);
                ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(),replay_ids.data(),replay_ids.size()*sizeof(float),ordinal,stream));
                for (int path = 0; path < 3; ++path)
                {
                    SCOPED_TRACE("generation="+std::to_string(generation)+" live="+std::to_string(live)+" path="+std::to_string(path));
                    ASSERT_NO_FATAL_FAILURE(poison());
                    ASSERT_NO_FATAL_FAILURE(observe(actual,actual_routes));
                    ASSERT_NE(std::memcmp(actual.data(),reference.data(),actual.size()*sizeof(float)),0);
                    ASSERT_TRUE(graphs[path]->launch());
                    ASSERT_NO_FATAL_FAILURE(observe(actual,actual_routes));
                    ASSERT_NO_FATAL_FAILURE(expectByteEqual("expanded WMMA output",actual,reference,hidden_width));
                    ASSERT_NO_FATAL_FAILURE(expectByteEqual("expanded WMMA canonical",actual_routes,reference_routes,hidden_width));
                }
            }
        }
    });
}
}
