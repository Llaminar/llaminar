/**
 * @file Test__RequestInputPositionAdmission.cpp
 * @brief Device-free ownership gates for external tokens with captured decode positions.
 *
 * Native addresses are opaque identities: no backend or accelerator is entered.
 * The tests distinguish a real host-position ingress from the existing graph's
 * canonical KV snapshot, ensuring a missing host row cannot trigger fabricated
 * positions, a second producer, or unused activation-tail uploads during decode.
 */
#include "execution/local_execution/graph/RequestInputPositionAdmission.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
using namespace llaminar2;
namespace
{
/** @brief Distinct opaque backend identities; these are never dereferenced. */
IBackend *backendIdentity(std::uintptr_t value) { return reinterpret_cast<IBackend *>(value); }
}
TEST(RequestInputPositionAdmission, HostPrefillOwnsItsPositionUploadAndBucketPadding)
{
    std::array<int,6> host{0,1,2,0,1,2};int position=0;
    const auto policy=RequestInputPositionAdmission::bind(host.data(),std::nullopt,
        backendIdentity(0x1000),nullptr,&position,6,2,3);
    EXPECT_EQ(policy.source(),RequestInputPositionAdmission::Source::HostAbsoluteRows);
    EXPECT_TRUE(policy.uploadsPositions());EXPECT_TRUE(policy.allowsBucketPadding());
}
TEST(RequestInputPositionAdmission, CanonicalScalarSnapshotCopiesNeitherPositionsNorTokenPadding)
{
    int canonical=9,position=-123;
    const DeviceDecodePositionBinding binding{backendIdentity(0x1000),&canonical,&position};
    const auto policy=RequestInputPositionAdmission::bind(nullptr,binding,binding.backend,&canonical,&position,1,1,1);
    EXPECT_EQ(policy.source(),RequestInputPositionAdmission::Source::CapturedCanonicalKVSnapshot);
    EXPECT_FALSE(policy.uploadsPositions());EXPECT_FALSE(policy.allowsBucketPadding());
    EXPECT_EQ(canonical,9);EXPECT_EQ(position,-123); // Policy never reads or writes replay values.
}
TEST(RequestInputPositionAdmission, RejectsMissingCompetingOrForeignPositionAuthority)
{
    int canonical=9,position=-123,foreign=3,host=4;
    const DeviceDecodePositionBinding correct{backendIdentity(0x1000),&canonical,&position};
    EXPECT_THROW(RequestInputPositionAdmission::bind(nullptr,std::nullopt,correct.backend,&canonical,&position,1,1,1),std::invalid_argument);
    EXPECT_THROW(RequestInputPositionAdmission::bind(&host,correct,correct.backend,&canonical,&position,1,1,1),std::invalid_argument);
    for(const DeviceDecodePositionBinding bad:{DeviceDecodePositionBinding{backendIdentity(0x2000),&canonical,&position},
        DeviceDecodePositionBinding{correct.backend,&foreign,&position},DeviceDecodePositionBinding{correct.backend,&canonical,&foreign},
        DeviceDecodePositionBinding{correct.backend,&canonical,&canonical},DeviceDecodePositionBinding{nullptr,&canonical,&position}})
        EXPECT_THROW(RequestInputPositionAdmission::bind(nullptr,bad,correct.backend,&canonical,&position,1,1,1),std::invalid_argument);
    EXPECT_EQ(canonical,9);EXPECT_EQ(position,-123);
}
TEST(RequestInputPositionAdmission, RejectsNonscalarSnapshotAndMalformedPhysicalGeometry)
{
    int canonical=9,position=0,host=7;
    const DeviceDecodePositionBinding binding{backendIdentity(0x1000),&canonical,&position};
    for(const auto shape:std::array{std::array{2,1,2},std::array{2,2,1},std::array{0,1,1},std::array{1,1,2}})
        EXPECT_THROW(RequestInputPositionAdmission::bind(nullptr,binding,binding.backend,&canonical,&position,
            shape[0],shape[1],shape[2]),std::invalid_argument);
    EXPECT_THROW(RequestInputPositionAdmission::bind(&host,std::nullopt,binding.backend,&canonical,&position,
        1,std::numeric_limits<int>::max(),std::numeric_limits<int>::max()),std::invalid_argument);
}

/** @test An immutable retained requirement is revalidated on each original ingress, never retained as a content proof. */
TEST(RequestInputPositionAdmission, ContiguousPrefillRequirementChecksEveryIngress)
{
    using Layout=RequestInputPositionLayoutRequirement;
    for(unsigned replay=0;replay<20;++replay){
        std::array<int,4> host{127,128,129,130},destination{};
        const auto admit=[&]{return RequestInputPositionAdmission::bind(host.data(),{},backendIdentity(0x1000),nullptr,
            destination.data(),4,1,4,Layout::ContiguousSingleRequestPrefill);};
        EXPECT_NO_THROW((void)admit());
        // Reuse the SAME original destination and retained requirement with a
        // new request. A previous valid request cannot authorize these values.
        host[2]=128;EXPECT_THROW((void)admit(),std::invalid_argument);
        host={211,212,213,214};EXPECT_NO_THROW((void)admit());
        EXPECT_EQ(destination.front(),0);
    }
}
/** @test Duplicate/gap/negative/overflow rows remain valid legacy ingress but cannot meet the selected future-family requirement. */
TEST(RequestInputPositionAdmission, ContiguousPrefillRequirementRejectsMalformedWithoutChangingLegacy)
{
    using Layout=RequestInputPositionLayoutRequirement;
    for(unsigned replay=0;replay<20;++replay)for(const auto host:std::array{
        std::array{7,7,9},std::array{7,9,10},std::array{-1,0,1},
        std::array{std::numeric_limits<int>::max()-1,std::numeric_limits<int>::max(),std::numeric_limits<int>::min()}}){
        std::array<int,3> destination{};
        const auto original=RequestInputPositionAdmission::bind(host.data(),{},backendIdentity(0x1000),nullptr,destination.data(),3,1,3);
        EXPECT_TRUE(original.uploadsPositions());EXPECT_TRUE(original.allowsBucketPadding());
        EXPECT_THROW((void)RequestInputPositionAdmission::bind(host.data(),{},backendIdentity(0x1000),nullptr,
            destination.data(),3,1,3,Layout::ContiguousSingleRequestPrefill),std::invalid_argument);
    }
}
/** @test Empty/batched/unknown requirements and scalar canonical decode cannot authorize a prefill window. */
TEST(RequestInputPositionAdmission, PrefillRequirementRejectsEmptyBatchedAndDecodeAuthority)
{
    using Layout=RequestInputPositionLayoutRequirement;
    int canonical=17,position=0;const std::array<int,4> host{0,1,2,3};
    EXPECT_THROW((void)RequestInputPositionAdmission::bind(host.data(),{},backendIdentity(0x1000),nullptr,&position,0,1,1,Layout::ContiguousSingleRequestPrefill),std::invalid_argument);
    EXPECT_THROW((void)RequestInputPositionAdmission::bind(host.data(),{},backendIdentity(0x1000),nullptr,&position,4,2,2,Layout::ContiguousSingleRequestPrefill),std::invalid_argument);
    EXPECT_THROW((void)RequestInputPositionAdmission::bind(host.data(),{},backendIdentity(0x1000),nullptr,&position,4,1,4,static_cast<Layout>(99)),std::invalid_argument);
    const DeviceDecodePositionBinding snapshot{backendIdentity(0x1000),&canonical,&position};
    EXPECT_NO_THROW((void)RequestInputPositionAdmission::bind(nullptr,snapshot,snapshot.backend,&canonical,&position,1,1,1));
    EXPECT_THROW((void)RequestInputPositionAdmission::bind(nullptr,snapshot,snapshot.backend,&canonical,&position,1,1,1,Layout::ContiguousSingleRequestPrefill),std::invalid_argument);
}
