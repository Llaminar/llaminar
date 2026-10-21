/**
 * @file Test__GPUAdaptiveImmutableLookupOwners.cpp
 * @brief Actual source/context/native root ownership and retirement, without loading model weights.
 *
 * The metadata selector supplies one canonical checked placement. Runtime PMA
 * here admits ONLY source-owned ngram component BOMs against actual GPU/CPU
 * resources; no weight/KV allocation or full-model capacity claim is made.
 * Tests initialize actual CUDA/HIP metadata and source owners but never execute
 * a model, lookup arithmetic, generation graph or performance workload.
 */
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "loaders/ModelContext.h"
#include "planning/MemoryPlanner.h"
#include "planning/PlanningModelMetadata.h"
#include "transfer/TransferEngine.h"
#include "models/qwen4exp/Qwen4ExpGraph.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "memory/BufferArena.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <unistd.h>
using namespace llaminar2;
namespace
{
/** @brief Retire every diagnostic submit before test owners on normal and exceptional exits. */
struct SetupRetirement final
{
    IBackend *backend;DeviceId device;void *stream;
    /** @brief This is ordinary setup/terminal retirement; no captured hot path is synchronized. */
    ~SetupRetirement() { if(!backend->synchronizeStream(stream,device.ordinal))std::terminate(); }
};
/** @return Original source words, never a requantized table or fabricated full resident view. */
ImmutableRowTableSource nativeSource(const ModelMemoryProfile &profile)
{
    const auto &g=*profile.flash_next;
    return {.rows=g.ple_table_rows,.row_elements=g.ple_row_elements,.format=static_cast<GGUFTensorType>(g.ple_table_gguf_type)};
}
/** @return Full metadata request for the actual production selector; capacity here is metadata-only. */
DevicePlanConfig metadataConsumer(DeviceId device,std::size_t slots)
{
    DevicePlanConfig c;c.world_rank=0;c.device=device;c.device_compute_units=64;
    c.device_total_bytes=512ull<<30;c.device_free_bytes=500ull<<30;
    c.associated_host_memory=PhysicalMemoryResource{0,DeviceId::cpu(),512ull<<30,500ull<<30};
    c.max_seq_len=8;c.activation_seq_len=2;c.first_layer=0;c.last_layer=47;c.batch_size=1;
    ImmutableRowTablePolicy policy;policy.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD;
    policy.window_bytes=65536;policy.window_count=2;
    policy.gpu_adaptive={slots,1u<<20,{}, {4,1u<<20,{}}};c.ngram_execution_policy=policy;return c;
}
/** @return Exact observed host capacity without subtracting another subsystem's allocations. */
PhysicalMemoryResource hostResource()
{
    const auto pages=::sysconf(_SC_PHYS_PAGES),available=::sysconf(_SC_AVPHYS_PAGES),page=::sysconf(_SC_PAGESIZE);
    if(pages<=0 || available<=0 || page<=0)throw std::runtime_error("Native owner fixture has no actual CPU capacity observation");
    return {0,DeviceId::cpu(),static_cast<std::size_t>(pages)*static_cast<std::size_t>(page),
        static_cast<std::size_t>(available)*static_cast<std::size_t>(page)};
}
/** @brief Validate actual retained claims, source identity and C0/nonzero lifetime on the selected native backend. */
void prove(DeviceId device)
{
    auto *backend=getBackendFor(device);
    if(!backend || backend->deviceCount()<=device.ordinal)
    {
        GTEST_SKIP()<<"Requested native owner backend unavailable";
    }
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    const auto &profile=source.metadata().memoryProfile();
    auto &worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&] {
        void *stream=ExplicitGPUStream(worker.defaultStream()).get();
        for(const std::size_t slots:{0u,4u})
        {
            auto c=metadataConsumer(device,slots);const std::vector<std::vector<DevicePlanConfig>> groups{{c}};
            auto request=MemoryPlanner::immutableRowTableAdmissionRequest(
                profile,*c.ngram_execution_policy,groups,{},c.ngram_host_placement);
            ASSERT_TRUE(request);const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();
            const auto selected=ImmutableRowTablePlacementPlanner::select(*request,frozen);
            const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(selected.placement->gpuAdaptiveLayout(),
                nativeSource(profile),selected.placement->gpuAdaptiveRequest());
            const PhysicalMemoryResource gpu{0,device,backend->deviceMemoryTotal(device.ordinal),backend->deviceMemoryFree(device.ordinal)};
            const auto host=hostResource();PhysicalMemoryPlanBuilder admitted(geometry.mandatoryBOM(gpu,host));
            const auto optional=geometry.cacheBOM(gpu,host);for(const auto &bom:optional.resources())admitted.add(bom);
            const auto component=admitted.build();
            auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(component),0);
            auto context=ModelContext::create(file.path());ASSERT_TRUE(context);
            GraphConfig graphConfig;
            graphConfig.default_device = device;
            graphConfig.max_seq_len = c.max_seq_len;
            graphConfig.max_activation_rows = c.activation_seq_len;
            graphConfig.max_request_count = 1;
            Qwen4ExpGraphConfigBuilder::populateFromDirectory(context->model(), graphConfig);
            graphConfig.ngram_table_placement = selected.placement;
            auto graph = std::make_unique<Qwen4ExpGraph>(graphConfig);
            graph->setModelContext(context);
            graph->bindStorageMemoryAuthority(memory);
            BufferArena declarationArena;
            std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary;
            AdaptiveImmutableLookupRoots roots;
            // All owners declared above survive this guard's terminal wait on
            // exceptional assertion exits. Successful boundary teardown also
            // independently retains/proves its ordinary setup frontier.
            SetupRetirement retire{backend,device,stream};
            ASSERT_FALSE(context->resolvedImmutableRowTablePlacement());
            ASSERT_NO_THROW(graph->setArena(&declarationArena));
            EXPECT_FALSE(context->adaptiveImmutableLookupBoundary(device));
            for (const auto role : {BufferId::QFN_NGRAM_ADAPTIVE_CACHE,
                                   BufferId::QFN_NGRAM_ADAPTIVE_FRAME,
                                   BufferId::QFN_NGRAM_LOOKUP_STATE})
                EXPECT_FALSE(declarationArena.isRegistered(role));
            // Real binding remains strict: declaration cannot substitute a
            // prepared ModelContext source or consume any PMA claim.
            try
            {
                (void)graph->retainedStorageBoundary(device);
                FAIL() << "An unprepared actual ModelContext cannot bind a runtime source boundary";
            }
            catch (const std::logic_error &error)
            {
                EXPECT_NE(std::string(error.what()).find(
                    "Adaptive graph and original source have different admitted placements"), std::string::npos);
            }
            EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableLookupStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),0u);
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),0u);
            context->prepareImmutableRowTableForConsumer(*selected.placement,memory,device,std::nullopt,
                TransferEngine::instance(),AdaptiveImmutableLookupDevicePreparation{device,stream});
            context->requireImmutableRowTablePlacementComplete(*selected.placement,memory);
            boundary=context->adaptiveImmutableLookupBoundary(device);ASSERT_TRUE(boundary);
            roots=boundary->adaptiveRoots();ASSERT_TRUE(roots.frame);ASSERT_TRUE(roots.protocol);
            EXPECT_EQ(bool(roots.cache),slots!=0);
            ASSERT_EQ(graph->retainedStorageBoundary(device).get(),boundary.get());
            EXPECT_EQ(declarationArena.getTensor(BufferId::QFN_NGRAM_ADAPTIVE_CACHE),roots.cache.get());
            EXPECT_EQ(declarationArena.getTensor(BufferId::QFN_NGRAM_ADAPTIVE_FRAME),roots.frame.get());
            EXPECT_EQ(declarationArena.getTensor(BufferId::QFN_NGRAM_LOOKUP_STATE),roots.protocol.get());
            EXPECT_EQ(graph->retainedStorageBoundary(device).get(),boundary.get());
            // An independently admitted changed policy cannot bind the existing
            // source even when its payload/cache byte geometry happens to match.
            {
                auto different = c;
                ++different.ngram_execution_policy->gpu_adaptive.gpu_policy.aging_accesses;
                const std::vector<std::vector<DevicePlanConfig>> differentGroups{{different}};
                auto differentRequest=MemoryPlanner::immutableRowTableAdmissionRequest(profile,
                    *different.ngram_execution_policy,differentGroups,{},different.ngram_host_placement);
                ASSERT_TRUE(differentRequest);
                const auto differentPlacement=ImmutableRowTablePlacementPlanner::select(
                    *differentRequest,MemoryPlanner::plan(profile,{different}).physicalPlan()).placement;
                ASSERT_NE(*differentPlacement,*selected.placement);
                auto differentConfig=graphConfig;
                differentConfig.ngram_table_placement=differentPlacement;
                Qwen4ExpGraph differentGraph(differentConfig);
                differentGraph.setModelContext(context);
                differentGraph.bindStorageMemoryAuthority(memory);
                BufferArena differentArena;
                ASSERT_NO_THROW(differentGraph.setArena(&differentArena));
                try
                {
                    (void)differentGraph.retainedStorageBoundary(device);
                    FAIL() << "A changed admitted placement cannot borrow the prepared original source";
                }
                catch (const std::logic_error &error)
                {
                    EXPECT_NE(std::string(error.what()).find(
                        "Adaptive graph and original source have different admitted placements"), std::string::npos);
                }
                EXPECT_FALSE(differentArena.isRegistered(BufferId::QFN_NGRAM_ADAPTIVE_CACHE));
                EXPECT_FALSE(differentArena.isRegistered(BufferId::QFN_NGRAM_ADAPTIVE_FRAME));
                EXPECT_FALSE(differentArena.isRegistered(BufferId::QFN_NGRAM_LOOKUP_STATE));
                EXPECT_EQ(context->adaptiveImmutableLookupBoundary(device).get(),boundary.get());
            }
            EXPECT_TRUE(roots.frame->is_raw_data_released());EXPECT_TRUE(roots.protocol->is_raw_data_released());
            EXPECT_THROW((void)roots.frame->data(),std::logic_error);
            EXPECT_THROW((void)roots.protocol->data(),std::logic_error);
            if(roots.cache)
            {
                EXPECT_TRUE(roots.cache->is_raw_data_released());
                EXPECT_THROW((void)roots.cache->data(),std::logic_error);
            }
            EXPECT_FALSE(context->residentImmutableRowTable(device));EXPECT_EQ(context->residentImmutableRowTableBytes(device),0u);
            auto provider=context->hostImmutableRowProvider();ASSERT_TRUE(provider);
            EXPECT_EQ(provider->source().path,file.path());EXPECT_EQ(provider->source().tensor_name,"per_layer_token_embd.weight");
            EXPECT_EQ(provider->subordinateRequester(),(PhysicalMemoryAllocatorIdentity{0,device}));
            std::array<std::byte,90> response;response.fill(std::byte{0x55});
            const std::array<std::int64_t,1> ids{0};
            EXPECT_THROW((void)provider->gather(ids,response),std::logic_error);
            for(const auto value:response)EXPECT_EQ(value,std::byte{0x55});
            const auto actualGPU=boundary->rootStorage()->retainedOwnerObservation();
            const auto actualHost=boundary->retainedSourceOwnerObservation();
            EXPECT_EQ(actualGPU.authority().get(),memory.get());EXPECT_EQ(actualHost.authority().get(),memory.get());
            EXPECT_EQ(actualGPU.tableStorageBytes(),geometry.gpu().cache().bytes);
            EXPECT_EQ(actualGPU.lookupStagingBytes(),geometry.gpu().mandatoryDeviceBytes());
            const auto *hostLines=component.find({0,DeviceId::cpu()});ASSERT_NE(hostLines,nullptr);
            EXPECT_EQ(actualHost.tableStorageBytes(),hostLines->bytes(PhysicalMemoryOwner::ImmutableTableStorage));
            EXPECT_EQ(actualHost.lookupStagingBytes(),hostLines->bytes(PhysicalMemoryOwner::ImmutableLookupStaging));
            const auto identity=boundary->captureIdentity();const auto frame=roots.frame.get();const auto protocol=roots.protocol.get();
            context->prepareImmutableRowTableForConsumer(*selected.placement,memory,device,std::nullopt,TransferEngine::instance());
            EXPECT_EQ(context->adaptiveImmutableLookupBoundary(device).get(),boundary.get());
            EXPECT_EQ(boundary->captureIdentity(),identity);EXPECT_EQ(boundary->adaptiveRoots().frame.get(),frame);
            EXPECT_EQ(boundary->adaptiveRoots().protocol.get(),protocol);
            auto foreign=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(component),0);
            EXPECT_THROW(context->prepareImmutableRowTableForConsumer(*selected.placement,foreign,device,std::nullopt,TransferEngine::instance()),std::logic_error);
            EXPECT_THROW(boundary->rootStorage()->requireFor(geometry.gpu().nativePlan(),foreign),std::logic_error);
            auto changed=selected.placement->gpuAdaptiveRequest();++changed.gpu_policy.aging_accesses;
            const auto changedPlan=AdaptiveImmutableLookupGeometry::fromLayout(selected.placement->gpuAdaptiveLayout(),nativeSource(profile),changed).gpu().nativePlan();
            EXPECT_THROW(boundary->rootStorage()->requireFor(changedPlan,memory),std::logic_error);
            EXPECT_EQ(boundary->captureIdentity(),identity);
            EXPECT_TRUE(backend->synchronizeStream(stream,device.ordinal));
            graph.reset();
            context.reset();provider.reset();
            // Every component root/provider/channel above was issued through
            // claimNewAllocation. Observe that exact live PMA class, never
            // substitute the admitted BOM or an adopted-residency credit.
            EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableLookupStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),actualGPU.lookupStagingBytes());
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),actualHost.lookupStagingBytes());
            boundary.reset(); // Last source boundary retires its exact setup event before service/root release.
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableTableStorage,
                PhysicalMemoryMaterializationKind::NewAllocation),0u);
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ImmutableLookupStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),0u);
            EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableLookupStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),actualGPU.lookupStagingBytes());
            roots={};
            EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableLookupStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),0u);
            EXPECT_EQ(memory->claimedBytes(device,PhysicalMemoryOwner::ImmutableTableStorage,
                PhysicalMemoryMaterializationKind::NewAllocation),0u);
        }
    });
}
}
#ifdef HAVE_CUDA
TEST(GPUAdaptiveImmutableLookupOwners, CUDA){prove(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(GPUAdaptiveImmutableLookupOwners, ROCm){prove(DeviceId::rocm(0));}
#endif
