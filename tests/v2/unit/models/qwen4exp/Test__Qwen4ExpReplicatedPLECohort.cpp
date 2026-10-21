/**
 * @file Test__Qwen4ExpReplicatedPLECohort.cpp
 * @brief Device-free original source/head/communicator declaration checks.
 *
 * The real Rank constructor calls this validator. It returns no cohort/query
 * capability and cannot certify device token/history replication; canonical
 * full-model Rank lifecycle qualification remains mandatory.
 */
#include "models/qwen4exp/Qwen4ExpReplicatedPLECohort.h"
#include "Qwen4ExpMetadataFixture.h"
#include "../../../mocks/MockLocalTPContext.h"
#include <gtest/gtest.h>
#include <algorithm>
using namespace llaminar2;
using namespace llaminar2::test;
namespace {
/** @return Published metadata geometry with unequal original head domains. */
GGUFModel sourceMetadata()
{
    auto model=qwen4MetadataFixture(false);
    std::vector<std::uint64_t> sizes(16),offsets(16);std::uint64_t next=5;
    for(std::size_t h=0;h<16;++h){offsets[h]=next;sizes[h]=h+3;next+=sizes[h]+(h==7?7:1);}
    model.metadata["qwen4exp.ple.head_offsets"]=qwen4Array(GGUFValueType::UINT64,offsets);
    model.metadata["qwen4exp.ple.head_vocab_sizes"]=qwen4Array(GGUFValueType::UINT64,sizes);
    return model;
}
/** @return Source-ordered owners, deliberately reversing physical ordinals. */
std::vector<PhysicalMemoryAllocatorIdentity> owners(int degree,DeviceType type)
{
    std::vector<PhysicalMemoryAllocatorIdentity> result;
    for(int p=0;p<degree;++p)result.push_back({0,DeviceId(type,degree-p-1)});
    return result;
}
/** @brief Configure immutable mock metadata without starting a backend. */
void configure(MockLocalTPContext &context,const std::vector<PhysicalMemoryAllocatorIdentity> &members)
{
    std::vector<GlobalDeviceAddress> addresses;
    for(const auto &member:members)addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(member.device));
    context.setDevices(std::move(addresses));
    context.setBackend(members.front().device.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL);
    context.setRawAllgatherGraphCaptureSupported(true);
}
/** @return Sole checked layout declaration; no owner or memory claim is minted. */
auto declaration(const Qwen4ExpConfig &source,const std::vector<PhysicalMemoryAllocatorIdentity> &members)
{
    const auto &n=source.ngrams();
    return Qwen4ExpFixedHeadShards::declare(
        ImmutableRowTableExtent::fromNative(n.table.type,n.table.dimensions[0],n.row_bytes),
        n.head_offsets,n.head_vocab_sizes,members);
}
}
/** @test Source-head coordinate follows actual collective order, independent of range sizes. */
TEST(Qwen4ExpReplicatedPLECohort, ExactSourceOrderAndNativeCollectiveCoordinateAreRequired)
{
    const Qwen4ExpConfig source(sourceMetadata());
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})for(const int degree:{2,4,8})
    {
        auto members=owners(degree,type);const auto placement=declaration(source,members);MockLocalTPContext context;
        configure(context,members);
        EXPECT_NO_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source));
        EXPECT_NE(placement->fixedHeadSourceRangeFor(members.front()).nativeBytes(),
            placement->fixedHeadSourceRangeFor(members.back()).nativeBytes());
        std::reverse(members.begin(),members.end());configure(context,members);
        EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
        const auto reversed=declaration(source,members);
        EXPECT_NO_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*reversed,source));
    }
}
/** @test Foreign membership, source and unsupported transport reject before graph mutation. */
TEST(Qwen4ExpReplicatedPLECohort, ForeignMembershipSourceAndCaptureTransportRejectBeforeMutation)
{
    const Qwen4ExpConfig source(sourceMetadata());const auto members=owners(2,DeviceType::CUDA);
    const auto placement=declaration(source,members);MockLocalTPContext context;configure(context,members);
    context.setRawAllgatherGraphCaptureSupported(false);
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
    configure(context,members);context.setBackend(CollectiveBackendType::RCCL);
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
    auto foreign=members;foreign[1].device=DeviceId::cuda(9);configure(context,foreign);
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
    foreign=members;foreign[1]=foreign[0];configure(context,foreign);
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
    foreign=members;foreign[1].device=DeviceId::rocm(0);configure(context,foreign);
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
    configure(context,owners(4,DeviceType::CUDA));
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,source),std::invalid_argument);
    configure(context,members);const Qwen4ExpConfig different(qwen4MetadataFixture(false));
    EXPECT_THROW(Qwen4ExpReplicatedPLECohort::requireDeclaration(context,*placement,different),std::invalid_argument);
    EXPECT_EQ(context.allreduceCallCount(),0);
}
