/**
 * @file Test__PooledImmutableDirectedOwnerProgram.cpp
 * @brief Pure original-geometry replay of the actual source-authored owner program class.
 *
 * These declarations neither create a worker nor mint a source/host certificate.
 * Stage/kernel native equivalence and recurring model execution remain separate gates.
 */
#include "execution/qwen4exp/PooledImmutableDirectedOwnerProgram.h"
#include "execution/qwen4exp/PooledImmutableDirectedMessageGeometry.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include "loaders/ModelLoader.h"
#include <gtest/gtest.h>
#include <array>
#include <type_traits>
namespace llaminar2::test {
namespace {
namespace q=qwen4exp;
static_assert(!std::is_constructible_v<PooledRemoteGraphProgram,PooledRemoteGraphGeometry,std::array<std::uint8_t,32>,GPUFlatGraphLimits>);
static_assert(!std::is_constructible_v<PooledNativeOwnerGraphProgram,PooledRemoteGraphGeometry,std::array<std::uint8_t,32>>);
/** @brief Native General metadata cannot be projected into a mapped resource/flat-kernel certificate. */
template<class T> concept HasMappedFlatCertificate=requires(const T& program){program.limits();program.qualificationClass();};
static_assert(!HasMappedFlatCertificate<PooledNativeOwnerGraphProgram>);

/** @return Checked storage coordinates and request geometry with no compute-domain substitution. */
auto declare(DeviceId source,DeviceId remote,std::uint32_t capacity,std::uint64_t rows=4096,std::optional<PeerAccessCoverage> peerAccess=std::nullopt){
    const PhysicalMemoryAllocatorIdentity coordinator{0,source},owner{0,remote};
    const std::array order{coordinator,owner};
    std::vector<ImmutableRowTableLayoutOwner> owners{{{coordinator,ImmutableRowStorage::VRAM},0,0,{}},{{owner,ImmutableRowStorage::VRAM},0,capacity,{}}};
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,rows,9000),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,std::move(owners),{coordinator},{},
        {{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},{},
        capacity?std::optional(coordinator):std::nullopt,16,ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator,peerAccess);
    auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,coordinator);
    auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},32);
    return std::pair{std::move(physical),std::move(policy)};
}
}
TEST(PooledImmutableDirectedOwnerProgram,ResetPartialAndPlacementClassReplay){
    for(const auto source:{DeviceId::cuda(0),DeviceId::rocm(0)})for(unsigned epoch=0;epoch<20;++epoch)
        for(const auto capacity:{0u,1u,16u,2u,0u}){
            const auto remote=source.is_cuda()?DeviceId::rocm(0):DeviceId::cuda(0);
            const auto [physical,policy]=declare(source,remote,capacity);
            const auto original=q::PooledImmutableDirectedOwnerProgram::declare(physical,policy,1);
            const auto repeated=q::PooledImmutableDirectedOwnerProgram::declare(physical,policy,1);
            EXPECT_EQ(original,repeated);EXPECT_NE(original.captureIdentity(),0u);
            EXPECT_EQ(original.geometry().coordinator.device,source);EXPECT_EQ(original.geometry().owner.device,remote);
            EXPECT_EQ(original.geometry().capacity,capacity);EXPECT_EQ(original.geometry().coordinate,1u);
            EXPECT_EQ(original.geometry().nonemptyRootMask,capacity?7u:6u);
            EXPECT_EQ(original.limits().allowedKinds,GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::Kernel));
            EXPECT_EQ(original.limits().maximumEdges+1,original.limits().maximumNodes);
            // Changing physical ordinals changes capture identity, while the
            // source-owned portable dispatch recipe is unchanged.
            const auto [otherPhysical,otherPolicy]=declare(source.is_cuda()?DeviceId::cuda(3):DeviceId::rocm(3),
                remote.is_cuda()?DeviceId::cuda(4):DeviceId::rocm(4),capacity);
            const auto other=q::PooledImmutableDirectedOwnerProgram::declare(otherPhysical,otherPolicy,1);
            EXPECT_NE(original.captureIdentity(),other.captureIdentity());EXPECT_EQ(original.qualificationClass(),other.qualificationClass());
        }
}
// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/** @test Homogeneous owners retain exact native-General semantics and can never mint mapped flat scope. */
TEST(PooledImmutableDirectedOwnerProgram, NativeGeneralPreservesOriginalGeometryWithoutFlatCertificate)
{
    for(const auto source:{DeviceId::cuda(0),DeviceId::rocm(0)})
    for(const auto peer:{PeerAccessCoverage::None,PeerAccessCoverage::Complete})
    for(const auto capacity:{0u,1u,64u}){
        const DeviceId remote{source.type,1};
        const auto [physical,policy]=declare(source,remote,capacity,4096,peer);
        const auto authored=q::PooledImmutableDirectedOwnerProgram::declareStorage(physical,policy,1);
        ASSERT_TRUE(std::holds_alternative<PooledNativeOwnerGraphProgram>(authored));
        const auto coordinator=q::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,701,17);
        const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,1,701,17);
        EXPECT_EQ(authored,q::PooledImmutableDirectedOwnerProgram::fromOriginalStoragePlans(owner,coordinator));
        EXPECT_EQ(pooledStorageOwnerGeometry(authored).peerAccess,peer);
        EXPECT_EQ(pooledStorageOwnerGeometry(authored).owner.device,remote);
        EXPECT_NE(pooledStorageOwnerIdentity(authored),0u);
        const auto& native=std::get<PooledNativeOwnerGraphProgram>(authored);
        const PhysicalMemoryResource host{0,DeviceId::cpu(),1ULL<<34,1ULL<<34},gpu{0,remote,1ULL<<34,1ULL<<34};
        const auto storage=q::PooledImmutableDirectedOwnerProgram::fromOriginalStoragePlans(owner,coordinator);
        EXPECT_EQ(pooledStorageOwnerIdentity(storage),native.captureIdentity());
        const auto bill=RetainedParticipantGraphExecutable::nativeStorageBOM(native,host,gpu);
        const auto model=RetainedParticipantGraphExecutable::generalBOM(host,gpu);
        for(const auto& row:bill.resources()){
            const auto* ordinary=model.find({row.resource().world_rank,row.resource().device});ASSERT_NE(ordinary,nullptr);
            EXPECT_EQ(row.bytes(PhysicalMemoryOwner::NativeStorageGraphExecutable),ordinary->bytes(PhysicalMemoryOwner::NativeGraphExecutable));
            EXPECT_EQ(row.bytes(PhysicalMemoryOwner::NativeGraphExecutable),0u);
        }
        auto foreign=gpu;foreign.device=DeviceId(remote.type,7);
        EXPECT_THROW(RetainedParticipantGraphExecutable::nativeStorageBOM(native,host,foreign),std::invalid_argument);
        EXPECT_THROW(q::PooledImmutableDirectedOwnerProgram::declare(physical,policy,1),std::invalid_argument);
        EXPECT_THROW(q::PooledImmutableDirectedOwnerProgram::fromOriginalPlans(owner,coordinator),std::invalid_argument);
        const auto [unobserved,unobservedPolicy]=declare(source,remote,capacity);
        EXPECT_THROW(q::PooledImmutableDirectedOwnerProgram::declareStorage(unobserved,unobservedPolicy,1),std::invalid_argument);
    }
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)
TEST(PooledImmutableDirectedOwnerProgram,CompiledRowKeyABIChangesPortableScope){
    const auto [small,smallPolicy]=declare(DeviceId::cuda(0),DeviceId::rocm(0),16,4096);
    const auto [large,largePolicy]=declare(DeviceId::cuda(0),DeviceId::rocm(0),16,0x100000100ULL);
    const auto row32=q::PooledImmutableDirectedOwnerProgram::declare(small,smallPolicy,1);
    const auto row64=q::PooledImmutableDirectedOwnerProgram::declare(large,largePolicy,1);
    EXPECT_EQ(row32.geometry().rowKeyBits,32u);EXPECT_EQ(row64.geometry().rowKeyBits,64u);
    EXPECT_NE(row32.captureIdentity(),row64.captureIdentity());EXPECT_NE(row32.qualificationClass(),row64.qualificationClass());
}
TEST(PooledImmutableDirectedOwnerProgram,UnsupportedOwnerCannotDeclareNativeScope){
    const auto [physical,policy]=declare(DeviceId::cuda(0),DeviceId::rocm(0),1);
    EXPECT_THROW(q::PooledImmutableDirectedOwnerProgram::declare(physical,policy,0),std::invalid_argument);
    EXPECT_THROW(q::PooledImmutableDirectedOwnerProgram::declare(physical,policy,2),std::exception);
}
}
