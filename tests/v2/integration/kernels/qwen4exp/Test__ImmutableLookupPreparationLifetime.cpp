/**
 * @file Test__ImmutableLookupPreparationLifetime.cpp
 * @brief Ordinary and pooled factory-only native submission-owner teardown.
 *
 * No graph, fixture synchronization, payload poison or controller upload may
 * retire initialization/admission before these boundaries release. Production
 * submission-retirement events must preserve original physical/source owners.
 */
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "kernels/qwen4exp/ImmutableLookupDecode.h"
#include "kernels/qwen4exp/PooledImmutableNativeKernels.h"
#include "tensors/BlockStructures.h"
#include <optional>
#include <span>
#include <vector>
#include "backends/BackendManager.h"
#include "backends/ExplicitGPUStream.h"
#include "backends/GPUDeviceContextPool.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <unistd.h>
namespace llaminar2::test {
namespace {
namespace q=qwen4exp;
using namespace qwen4exp;
/** @brief Fail infrastructure without continuing into an invalid native address.
 * @param ok Required infrastructure invariant. @param why Precise diagnostic. */
inline void require(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
/** @return Checked extent before file/buffer allocation.
 * @param a First nonnegative factor. @param b Second nonnegative factor.
 * @throws std::overflow_error Before any allocation for unrepresentable extents. */
inline std::size_t product(std::size_t a,std::size_t b){if(b&&a>std::numeric_limits<std::size_t>::max()/b)throw std::overflow_error("Adaptive component fixture extent");return a*b;}
/** @brief Own only the synthetic original-native file, never a downloaded/model payload. */
class Source final {
public:
    /** @brief Persist an exact complete original-native table without conversion.
     * @param bytes Original row-major codebook bytes authored by the caller.
     * @param rows Exact complete original source row count.
     * @param columns Original logical width.
     * @param type Exact GGUF source discriminator.
     */
    Source(std::span<const std::byte> bytes,std::size_t rows,int columns,GGUFTensorType type)
        :native(bytes.begin(),bytes.end()),rows(rows),columns(columns),type(type){
        char pattern[]="/tmp/llaminar-adaptive-native-XXXXXX";fd=::mkstemp(pattern);require(fd>=0,"Adaptive fixture source create failed");path=pattern;
        try{std::size_t done=0;while(done<bytes.size()){const auto n=::write(fd,bytes.data()+done,bytes.size()-done);require(n>0,"Adaptive fixture source write failed");done+=static_cast<std::size_t>(n);}require(descriptor().tensorBytes()==native.size(),"Adaptive fixture source byte extent differs");}
        catch(...){::close(fd);::unlink(path.c_str());throw;}
    }
    /** @brief Keep file alive through every provider/source-read lease. */
    ~Source(){::close(fd);::unlink(path.c_str());}
    Source(const Source&)=delete;Source&operator=(const Source&)=delete;
    /** @return Original exact source descriptor, used by the real loaders. */
    ImmutableRowTableSource descriptor()const{return {.path=path,.tensor_name="per_layer_token_embd.weight",.file_offset=0,.rows=rows,.row_elements=static_cast<std::uint64_t>(columns),.format=type};}
    /** @brief Deterministic source-failure injection only after native/component preparation. */
    void removePayload(){require(::ftruncate(fd,0)==0,"Adaptive source failure injection failed");}
    std::vector<std::byte> native;std::size_t rows;int columns;GGUFTensorType type;int fd=-1;std::string path;
};
/** @return Exact native IQ4_NL bytes; no floating-weight requantization.
 * @param rows Original source row count. @param columns Exact complete-block logical width. */
inline std::vector<std::byte> iq4Source(std::size_t rows,int columns){
    require(columns>0&&columns%32==0,"IQ4_NL component width must contain complete original blocks");
    const auto n=product(rows,static_cast<std::size_t>(columns)/32);std::vector<IQ4_NLBlock> blocks(n);
    for(std::size_t i=0;i<n;++i){blocks[i].d=0x3c00;for(unsigned j=0;j<16;++j)blocks[i].qs[j]=static_cast<std::uint8_t>((i+j)%16|(((i*3+j)%16)<<4));}
    std::vector<std::byte> result(product(n,sizeof(IQ4_NLBlock)));std::memcpy(result.data(),blocks.data(),result.size());return result;
}
/**
 * @brief Retire ordinary mapped protocol writes if admission is abandoned before a graph.
 * @param device Exact explicit integration backend.
 * @note This is the symmetric existing RAM/mmap lane. Every source/channel and
 * tiny host/device protocol byte has its actual canonical PMA owner. There is
 * no stream join between successful admission and boundary destruction.
 */
void abandonOrdinaryAdmission(DeviceId device)
{
    auto *backend = getBackendFor(device);
    require(backend && backend->deviceCount()>device.ordinal,"Explicit ordinary preparation GPU unavailable");
    if (!hasCPUBackend())
        initCPUBackend(-1);
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait(
        [&]
        {
            const auto resource = [](DeviceId id)
            {
                return PhysicalMemoryResource{.world_rank = 0,
                                              .device = id,
                                              .total_bytes = std::size_t{1} << 28,
                                              .admission_available_bytes = std::size_t{1} << 28};
            };
            for (const auto mode : {ImmutableRowStorage::RAM, ImmutableRowStorage::MemoryMapped})
                for (int iteration = 0; iteration < 20; ++iteration)
                {
                    Source source(iq4Source(97, 160), 97, 160, GGUFTensorType::IQ4_NL);
                    const auto policy = ImmutableRowTableStoragePolicy::forStorage(mode, 4096, 2);
                    const auto layout = ImmutableLookupLayout::create(16, 90);
                    PhysicalMemoryPlanBuilder plan;
                    plan.add(resource(DeviceId::cpu()), PhysicalMemoryOwner::ImmutableTableStorage,
                             ImmutableRowTable::physicalBytes(source.descriptor(), policy));
                    plan.add(resource(DeviceId::cpu()), PhysicalMemoryOwner::ImmutableLookupStaging,
                             ImmutableLookupService::hostBytes(layout));
                    plan.add(resource(DeviceId::cpu()), PhysicalMemoryOwner::ActivationArena, 64);
                    plan.add(resource(device), PhysicalMemoryOwner::ActivationArena, 64);
                    auto memory = std::make_shared<PhysicalMemoryAuthority>(
                        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
                    auto hostClaim =
                        memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena, 64);
                    auto deviceClaim =
                        memory->claimNewAllocation(device, PhysicalMemoryOwner::ActivationArena, 64);
                    auto protocol = std::make_shared<INT32Tensor>(std::vector<std::size_t>{16});
                    const auto stream = ExplicitGPUStream(worker.defaultStream()).get();
                    TransferEngine::prepareDeviceInput(protocol.get(), device, stream);
                    TransferEngine::requireDeviceInput(protocol.get(), device, stream);
                    auto table = HostImmutableRowProvider::fromWholeTable(std::make_unique<ImmutableRowTable>(
                        source.descriptor(), policy, DeviceId::cpu(), memory,
                        ImmutableRowTableHostPlacement::taskPolicy()));
                    auto boundary = q::ImmutableLookupExecutionBoundary::create(table, device,
                                                                                       protocol, 1, memory);
                    // Null/default admission fails before any producer changes.
                    EXPECT_THROW(boundary->beginAdmission(1, nullptr), std::logic_error);
                    EXPECT_FALSE(boundary->hasActiveInterval());
                    boundary->beginAdmission(1, stream);
                    EXPECT_TRUE(boundary->hasActiveInterval());
                    boundary.reset();
                    protocol.reset();
                    table.reset();
                }
        });
}
/**
 * @brief Abandon native initialization or request admission before any graph.
 * @param device Exact explicitly registered backend participant.
 * @note No fixture join occurs after owner/coordinator initialize, source
 * createPooled or beginAdmission. The boundary records and retires that exact
 * producer frontier; only then can its last original root leases disappear.
 */
void abandonPooledPreparation(DeviceId device)
{
    auto* backend=getBackendFor(device);
    require(backend && backend->deviceCount()>device.ordinal,"Explicit pooled preparation GPU unavailable");
    if(!hasCPUBackend())initCPUBackend(-1);
    auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]{
        for(const auto slots:{0u,4u})for(const bool admit:{false,true})for(int iteration=0;iteration<20;++iteration){
            std::shared_ptr<PhysicalMemoryAuthority> memory;
            {
                Source file(iq4Source(97,160),97,160,GGUFTensorType::IQ4_NL);
                const auto source=file.descriptor();
                const PhysicalMemoryAllocatorIdentity gpuId{0,device},hostId{0,DeviceId::cpu()};
                const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
                const auto layout=ImmutableRowTableLayoutGeometry::checked(
                    ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),
                    ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,
                    {{{gpuId,ImmutableRowStorage::VRAM},0,slots,{}},{{hostId,ImmutableRowStorage::RAM},0,slots?2u:0u,{}}},
                    {gpuId},{},{{hostId,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},
                    {},slots?std::optional(gpuId):std::nullopt,16);
                const auto physical=PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,std::array{gpuId},gpuId);
                const auto policy=PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},64);
                const auto hostGeometry=CpuAdaptiveImmutableRowCacheGeometry::forSubordinateGPU(layout,hostId,gpuId);
                const PhysicalMemoryResource gpu{0,device,1ull<<34,1ull<<34},host{0,DeviceId::cpu(),1ull<<34,1ull<<34};
                const auto channel=ImmutableLookupLayout::createSubordinate(16,source.rowBytes());
                PhysicalMemoryPlanBuilder plan;
                for(const auto& bom:{hostGeometry.bom(host),physical.cacheBOM(0,gpu),physical.mandatoryBOM(0,gpu),
                    policy.policyBOM(gpu),policy.mandatoryBOM(gpu),
                    PooledImmutableOwnerNativePlan::mandatoryBOM(physical,policy,0,gpu),
                    PooledImmutableCoordinatorNativePlan::mandatoryBOM(physical,policy,gpu),
                    PooledImmutableSourceGeometry::mandatoryBOM(physical,policy,0,source,gpu)})
                    for(const auto& resource:bom.resources())plan.add(resource);
                plan.add(host,PhysicalMemoryOwner::ImmutableLookupStaging,ImmutableLookupService::hostBytes(channel));
                memory=std::make_shared<PhysicalMemoryAuthority>(
                    std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
                auto provider=HostImmutableRowProvider::fromSubordinateAdaptiveCache(
                    CpuAdaptiveImmutableRowCache::create(source,hostGeometry,memory));
                auto service=ImmutableLookupService::create(provider,channel,std::array{device},memory,DeviceId::cpu());
                const auto binding=service->bindSubordinate(device);
                const auto ownerPlan=PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,0,
                    binding.deviceView().base.table,binding.storageGeneration());
                const auto coordinatorPlan=PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,
                    binding.deviceView().base.table,binding.storageGeneration());
                const auto geometry=PooledImmutableSourceGeometry::fromPlans(ownerPlan,coordinatorPlan,source);
                const auto stream=ExplicitGPUStream(worker.defaultStream()).get();
                auto owner=PooledImmutableOwnerPreparedStorage::create(ownerPlan,memory,stream);
                auto coordinator=PooledImmutableCoordinatorPreparedStorage::create(coordinatorPlan,memory,stream);
                std::shared_ptr<IPooledImmutableNativeKernels> code;
#ifdef HAVE_CUDA
                if(device.is_cuda())code=createCUDAPooledImmutableNativeKernels();
#endif
#ifdef HAVE_ROCM
                if(device.is_rocm())code=createROCmPooledImmutableNativeKernels();
#endif
                require(bool(code),"Exact pooled native preparation bundle unavailable");
                code->initialize(*owner,stream);code->initialize(*coordinator,stream);
                auto boundary=ImmutableLookupExecutionBoundary::createPooled(service,geometry,owner,coordinator,memory,stream);
                EXPECT_FALSE(boundary->hasActiveInterval());EXPECT_EQ(service->stats().requests,0u);
                if(admit){boundary->beginAdmission(1,stream);EXPECT_TRUE(boundary->hasActiveInterval());}
                // The original boundary alone retires the final producer. No
                // graph/controller/helper object can hide an eager teardown.
                boundary.reset();owner.reset();coordinator.reset();
            }
            for(const auto participant:{device,DeviceId::cpu()})
                for(unsigned index=0;index<static_cast<unsigned>(PhysicalMemoryOwner::Count);++index)
                    EXPECT_EQ(memory->claimedBytes(participant,static_cast<PhysicalMemoryOwner>(index),
                        PhysicalMemoryMaterializationKind::NewAllocation),0u);
        }
    });
}
} // namespace
#ifdef HAVE_CUDA
TEST(ImmutableLookupPreparationLifetime,CUDAOrdinaryAdmissionTeardownBeforeFirstGraph){abandonOrdinaryAdmission(DeviceId::cuda(0));}
TEST(PooledImmutableProviderSourceDevice,CUDAProductionSourcePreparationLifetime){abandonPooledPreparation(DeviceId::cuda(0));}
#endif
#ifdef HAVE_ROCM
TEST(ImmutableLookupPreparationLifetime,ROCmOrdinaryAdmissionTeardownBeforeFirstGraph){abandonOrdinaryAdmission(DeviceId::rocm(0));}
TEST(PooledImmutableProviderSourceDevice,ROCmProductionSourcePreparationLifetime){abandonPooledPreparation(DeviceId::rocm(0));}
#endif
} // namespace llaminar2::test
