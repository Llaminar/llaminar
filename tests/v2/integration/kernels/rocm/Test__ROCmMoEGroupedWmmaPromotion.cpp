/**
 * @file Test__ROCmMoEGroupedWmmaPromotion.cpp
 * @brief Captured physical23 matrix execution after genuine CPU expert promotion.
 *
 * The unchanged compact Q5_K Dot4 transaction is the arithmetic authority.
 * Production CPU packing and tier transfer supply the expanded candidate bytes;
 * retained WMMA graphs must reproduce both canonical routes and final outputs
 * across dirty full/partial/empty replays and replacement activation data.
 * Mixed3/31 expert counts exercise both adaptive branches for genuine physical23
 * storage; the wide candidate adds cold13/15 against hot31. Suppressing either
 * selected down branch must preserve the other.
 */
#include "../moe/NativeQuantisedExpertTransferParityTest.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedWmma.h"
#include "../../../utils/ROCmGroupedBranchMutation.h"
#include <hip/hip_runtime.h>

namespace llaminar2::test
{
/** @brief Independently registered microtile and wide-candidate promotion evidence. */
enum class GroupedPromotionMode { Micro, Wide };

/** @brief Preserve compact-source arithmetic through captured expanded WMMA tiles.
 * @param mode Exact candidate roster and threshold histograms to qualify. */
void runGroupedWmmaPromotion(GroupedPromotionMode mode)
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
        using Engine = rocm::MoEGroupedPrefillEngine;
        using Floor = rocm::MoEGroupedWmmaRowFloor;
        std::vector<rocm::MoEGroupedPrefillPolicy> policies{{Engine::OrderedDot4,Floor::EveryExpert}};
        if (mode == GroupedPromotionMode::Wide)
            policies.push_back({Engine::WideDot4Wmma,Floor::Sixteen});
        else
        {
            policies.push_back({Engine::WmmaM16N32,Floor::EveryExpert});
            policies.push_back({Engine::WmmaM16N64,Floor::EveryExpert});
            for (auto floor : {Floor::Four,Floor::Eight,Floor::Twelve,Floor::Sixteen})
                for (auto engine : {Engine::WmmaM16N32,Engine::WmmaM16N64}) policies.push_back({engine,floor});
        }
        std::vector<decltype(context.createGraphCapture(stream))> graphs;
        for (size_t path = 0; path < policies.size(); ++path) graphs.push_back(context.createGraphCapture(stream));
        // Retire work before graph, tensor, workspace and promoted-byte owners.
        auto drain = [&](void *) { if (!backend->synchronizeStream(stream,ordinal)) std::terminate(); };
        std::unique_ptr<void,decltype(drain)> completion(&context,drain);
        const auto enqueue = [&](int storage) {
            return kernel->prepareExpertGroupsAsync(ids.get(),weights.get(),rows,experts,top_k) &&
                kernel->executeGroupedPrefillPipeline(hidden.get(),output.get(),gate_tables[storage],down_tables[storage],
                    rows,hidden_width,intermediate,experts,top_k,canonical.get()) &&
                kernel->reduceCanonicalRouteContributions(canonical.get(),output.get(),rows,top_k,hidden_width);
        };
        std::vector<std::vector<hipGraphNode_t>> native_nodes(policies.size());
        for (size_t path = 0; path < policies.size(); ++path)
        {
            rocm::ScopedMoEGroupedPrefillEngine engine(policies[path].engine,policies[path].minimum_expert_rows);
            ASSERT_TRUE(enqueue(path != 0));
            ScopedBackendGraphCapture recording(context,*graphs[path],"promoted physical23 grouped WMMA");
            ASSERT_TRUE(recording.begin());
            ASSERT_TRUE(enqueue(path != 0));
            recording.finish();
            std::vector<GPUGraphKernelNodeInfo> nodes;
            ASSERT_TRUE(graphs[path]->inspectKernelNodes(nodes));
            int native_count = 0, gate_up_count = 0, down_count = 0;
            for (const auto &node : nodes)
            {
                if (path > 0)
                {
                    ASSERT_EQ(node.name.find("moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel"),std::string::npos);
                    ASSERT_EQ(node.name.find("moe_grouped_down_batch_invariant_tiled_prefill_kernel"),std::string::npos);
                }
                if (node.name.find("grouped_wmma_detail") != std::string::npos)
                {
                    ++native_count;
                    ASSERT_NE(node.name.find("NativeQuantisedGemmFormat23E"),std::string::npos) << node.name;
                    const bool wide = policies[path].engine == Engine::WideDot4Wmma;
                    ASSERT_EQ(node.block_x,wide ? (node.grid_z == 2 ? 256u : 128u) :
                        policies[path].engine == Engine::WmmaM16N32 ? 128u : 256u);
                    if (wide)
                    {
                        ASSERT_NE(node.name.find("wideProjection"),std::string::npos);
                        const unsigned columns = node.grid_z == 2 ? 256u : 128u;
                        const unsigned native_columns = columns / 4;
                        const unsigned n = node.grid_z == 2 ? intermediate : hidden_width;
                        ASSERT_EQ(node.grid_x,(n+native_columns-1)/native_columns);
                    }
                    ASSERT_NE(node.name.find(policies[path].minimum_expert_rows == Floor::EveryExpert
                        ? "ELb0EE" : "ELb1EE"),std::string::npos) << node.name;
                    ASSERT_EQ(node.local_memory_bytes_per_thread,0u);
                    if (node.grid_z == 2) ++gate_up_count;
                    else if (node.grid_z == 1) ++down_count;
                }
            }
            ASSERT_EQ(native_count,path ? 2 : 0);
            ASSERT_EQ(gate_up_count,path ? 1 : 0);
            ASSERT_EQ(down_count,path ? 1 : 0);
            ASSERT_TRUE(graphs[path]->instantiate());
            auto *native = dynamic_cast<HIPGraphCapture *>(graphs[path].get());
            ASSERT_NE(native,nullptr);
            if (policies[path].engine == Engine::WideDot4Wmma)
            {
                std::array<size_t,2> roles;
                ASSERT_NO_THROW(roles = inspectROCmGroupedWideCapture(*native,static_cast<hipStream_t>(stream),Floor::Sixteen));
                EXPECT_EQ(roles,(std::array<size_t,2>{1,1}));
            }
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
        std::unique_ptr<void,decltype(drain)> observation_completion(&context,drain);
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
            std::unique_ptr<void,decltype(drain)> generation_completion(&context,drain);
            ASSERT_TRUE(backend->hostToDevice(hidden->gpu_data_ptr(),next_hidden->data(),hidden->numel()*sizeof(float),ordinal,stream));
            ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(),full_ids.data(),full_ids.size()*sizeof(float),ordinal,stream));
            ASSERT_TRUE(graphs[0]->launch());
            ASSERT_NO_FATAL_FAILURE(observe(expected,expected_routes));
            ASSERT_TRUE(std::any_of(expected.begin(),expected.end(),[](float value) { return value != 0.0f; }));
            // A recorded symbol is not proof of work. Independently omit each
            // native role, require failure against Dot4, then restore the node.
            if (generation == 0)
                for (size_t path = 1; path < policies.size(); ++path)
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
                std::unique_ptr<void,decltype(drain)> replay_publication_completion(&context,drain);
                std::fill(replay_ids.begin()+live*top_k,replay_ids.end(),-1.0f);
                std::fill(reference.begin()+live*hidden_width,reference.end(),0.0f);
                std::fill(reference_routes.begin()+live*top_k*hidden_width,reference_routes.end(),0.0f);
                ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(),replay_ids.data(),replay_ids.size()*sizeof(float),ordinal,stream));
                for (size_t path = 0; path < policies.size(); ++path)
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
            const std::vector<int> cold_counts = mode == GroupedPromotionMode::Wide
                ? std::vector<int>{13,15} : std::vector<int>{3};
            for (const int cold : cold_counts)
            {
            const std::array counts{cold,31};
            auto mixed_ids = full_ids;
            for (size_t slot = 0; slot < mixed_ids.size(); ++slot)
                if (slot/top_k >= static_cast<size_t>(counts[static_cast<int>(mixed_ids[slot])])) mixed_ids[slot] = -1.0f;
            // Keep the tiled family active while the complete cold expert
            // selects Dot4, including two TM12 tiles for wide counts13/15.
            ASSERT_EQ(std::count_if(mixed_ids.begin(),mixed_ids.end(),[](float id) { return id >= 0; }),cold+31);
            std::unique_ptr<void,decltype(drain)> mixed_publication_completion(&context,drain);
            ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(),mixed_ids.data(),mixed_ids.size()*sizeof(float),ordinal,stream));
            ASSERT_TRUE(graphs[0]->launch());
            ASSERT_NO_FATAL_FAILURE(observe(expected,expected_routes));
            for (size_t path = 1; path < policies.size(); ++path)
            {
                ASSERT_NO_FATAL_FAILURE(poison());
                ASSERT_TRUE(graphs[path]->launch());
                ASSERT_NO_FATAL_FAILURE(observe(actual,actual_routes));
                ASSERT_NO_FATAL_FAILURE(expectByteEqual("mixed expanded output",actual,expected,hidden_width));
                ASSERT_NO_FATAL_FAILURE(expectByteEqual("mixed expanded routes",actual_routes,expected_routes,hidden_width));
                if (policies[path].minimum_expert_rows == Floor::EveryExpert) continue;
                auto *capture = dynamic_cast<HIPGraphCapture *>(graphs[path].get());
                ASSERT_NE(capture,nullptr);
                ROCmGroupedBranchMutation mutation(*capture,static_cast<hipStream_t>(stream));
                ASSERT_EQ(mutation.nodeCount(),1u);
                for (auto branch : {rocm::MoEGroupedBranchSuppression::Ordered,rocm::MoEGroupedBranchSuppression::Wmma})
                {
                    mutation.suppress(branch);
                    ASSERT_NO_FATAL_FAILURE(poison());
                    ASSERT_TRUE(graphs[path]->launch());
                    ASSERT_NO_FATAL_FAILURE(observe(actual,actual_routes));
                    mutation.restore();
                    const int omitted_expert = branch == rocm::MoEGroupedBranchSuppression::Ordered ? 0 : 1;
                    int changed = 0, surviving = 0;
                    for (size_t slot = 0; slot < mixed_ids.size(); ++slot)
                    {
                        const int comparison = std::memcmp(actual_routes.data()+slot*hidden_width,
                            expected_routes.data()+slot*hidden_width,hidden_width*sizeof(float));
                        if (mixed_ids[slot] == omitted_expert) changed += comparison != 0;
                        else
                        {
                            ASSERT_EQ(comparison,0) << "surviving expanded route=" << slot;
                            surviving += mixed_ids[slot] >= 0;
                        }
                    }
                    ASSERT_GT(changed,0); ASSERT_GT(surviving,0);
                    ASSERT_NO_FATAL_FAILURE(poison());
                    ASSERT_TRUE(graphs[path]->launch());
                    ASSERT_NO_FATAL_FAILURE(observe(actual,actual_routes));
                    ASSERT_NO_FATAL_FAILURE(expectByteEqual("restored expanded output",actual,expected,hidden_width));
                    ASSERT_NO_FATAL_FAILURE(expectByteEqual("restored expanded routes",actual_routes,expected_routes,hidden_width));
                }
            }
            }
        }
    });
}

/** @test Retain the original complete microtile promotion inventory. */
TEST(ROCmMoEGroupedWmmaPromotion, CapturedExpandedReplay)
{ runGroupedWmmaPromotion(GroupedPromotionMode::Micro); }

/** @test Qualify wide cold second tiles and hot WMMA against compact source bytes. */
TEST(ROCmMoEGroupedWmmaPromotion, CapturedWideExpandedReplay)
{ runGroupedWmmaPromotion(GroupedPromotionMode::Wide); }
}
