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
