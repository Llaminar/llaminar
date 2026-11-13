/**
 * @file Test__HostPipelineActivationRows.cpp
 * @brief Device-free source-bound pipeline copy admission and exact-live replay tests.
 *
 * Opaque CPU/CUDA/ROCm owner identities exercise the same production boundary
 * helper, with deferred transport represented by an exact byte-copy observer.
 * No GPU counter is read, no tensor vtable is borrowed from an old core, and
 * dirty suffix bytes expose any stale capacity or large-to-small copy.
 */
#include "execution/local_execution/graph/HostPipelineActivationRows.h"
#include "execution/prefix_cache/ResidualRowGeometry.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <optional>
#include <vector>
namespace llaminar2::test
{
namespace
{
/** @brief Transfer metadata only, matching the engine's callback contract. */
struct CopyInfo { bool needs_copy;std::size_t copy_bytes; };
/** @return Opaque stable source identity without constructing a backend or tensor. */
const TensorBase *sourceOwner() { return reinterpret_cast<const TensorBase *>(std::uintptr_t{0x1234}); }
/** @return Every production backend family, represented without driver discovery. */
std::array<DeviceId,3> participants() { return {DeviceId{DeviceType::CPU,1},DeviceId::cuda(0),DeviceId::rocm(0)}; }
}
/** @test Every replay resolves fresh logical rows and leaves inactive dirty capacity untouched. */
TEST(HostPipelineActivationRows, LargeSmallEmptyReplayCopiesOnlyCurrentLiveBytesAcrossBackends)
{
    for (auto device : participants())
    for (int width : {2560,10240})
    {
        const ResidualRowGeometry geometry(width);
        const std::array<std::size_t,2> shape{16,static_cast<std::size_t>(width)};
        const auto capacity = geometry.bytesForRows(shape,16);
        std::vector<std::uint8_t> source(capacity,0x37),destination(capacity);
        std::size_t resolutions=0,copies=0,actual_bytes=0;
        for (int rows : {8,3,0,1,0})
        {
            std::fill(destination.begin(),destination.end(),0xa5);
            const auto binding=HostPipelineActivationRows::admitted(sourceOwner(),device,1,rows);
            consumeHostPipelineActivationBoundary(
                HostPipelineBoundaryExecution::Execute,HostPipelineBoundarySubmission::Current,
                [&] {
                    ++resolutions;
                    const auto live=requireHostPipelineActivationRows(binding,sourceOwner(),device,1,16);
                    return CopyInfo{true,geometry.bytesForRows(shape,live)};
                },
                [&](const CopyInfo &copy) {
                    ++copies;actual_bytes+=copy.copy_bytes;
                    std::memcpy(destination.data(),source.data(),copy.copy_bytes);
                });
            const auto live_bytes=static_cast<std::size_t>(rows)*geometry.rowBytes();
            for (std::size_t i=0;i<capacity;++i)
                ASSERT_EQ(destination[i],i<live_bytes?0x37:0xa5)<<i<<" rows="<<rows;
        }
        EXPECT_EQ(resolutions,5u);EXPECT_EQ(copies,3u);EXPECT_EQ(actual_bytes,12u*geometry.rowBytes());
    }
}
/** @test Setup validates only metadata; first-use graph routing does not duplicate the host prelude. */
TEST(HostPipelineActivationRows, MaterializationAndAlreadySubmittedNeverCopyPayload)
{
    std::size_t resolutions=0,copies=0;
    auto resolve=[&]{++resolutions;return CopyInfo{true,40960};};
    auto copy=[&](const auto &){++copies;};
    consumeHostPipelineActivationBoundary(HostPipelineBoundaryExecution::MaterializeOnly,
        HostPipelineBoundarySubmission::Current,resolve,copy);
    EXPECT_EQ(resolutions,1u);EXPECT_EQ(copies,0u);
    consumeHostPipelineActivationBoundary(HostPipelineBoundaryExecution::Execute,
        HostPipelineBoundarySubmission::AlreadySubmitted,resolve,copy);
    EXPECT_EQ(resolutions,1u);EXPECT_EQ(copies,0u);
}
/** @test Address-only setup cannot mint runtime permission for an unadmitted distinct source. */
TEST(HostPipelineActivationRows, SetupOnlyUnadmittedSourceHasNoLiveBytesOrRuntimePermission)
{
    for (auto device : participants())
    {
        EXPECT_EQ(requireHostPipelineActivationRows(std::nullopt,sourceOwner(),device,1,16,
            HostPipelineBoundaryExecution::MaterializeOnly),0u);
        EXPECT_THROW(requireHostPipelineActivationRows(std::nullopt,sourceOwner(),device,1,16,
            HostPipelineBoundaryExecution::Execute),std::invalid_argument);
        const auto foreign=HostPipelineActivationRows::admitted(
            reinterpret_cast<const TensorBase *>(std::uintptr_t{0x4321}),device,1,3);
        EXPECT_THROW(requireHostPipelineActivationRows(foreign,sourceOwner(),device,1,16,
            HostPipelineBoundaryExecution::MaterializeOnly),std::invalid_argument);
    }
}
/** @test Neither a device count alone nor a marker for a foreign source/device can authorize a copy. */
TEST(HostPipelineActivationRows, RejectsUnadmittedOrForeignMarkersBeforeSubmission)
{
    for (auto device : participants())
    {
        std::size_t copies=0;
        EXPECT_THROW(consumeHostPipelineActivationBoundary(HostPipelineBoundaryExecution::Execute,
            HostPipelineBoundarySubmission::Current,
            [&] {return CopyInfo{true,requireHostPipelineActivationRows(std::nullopt,sourceOwner(),device,1,16)};},
            [&](const auto &){++copies;}),std::invalid_argument);
        const auto binding=HostPipelineActivationRows::admitted(sourceOwner(),device,1,3);
        const auto foreign=reinterpret_cast<const TensorBase *>(std::uintptr_t{0x4321});
        EXPECT_THROW(binding.rowsFor(foreign,device,1,16),std::invalid_argument);
        EXPECT_THROW(binding.rowsFor(sourceOwner(),DeviceId{DeviceType::CPU,2},1,16),std::invalid_argument);
        EXPECT_THROW(binding.rowsFor(sourceOwner(),device,2,16),std::invalid_argument);
        EXPECT_THROW(binding.rowsFor(sourceOwner(),device,1,2),std::invalid_argument);
        EXPECT_EQ(copies,0u);
    }
}
/** @test Width and physical row capacity remain mandatory even when the live extent is empty. */
TEST(HostPipelineActivationRows, ResidualWidthAndCapacityAreAuthenticatedForEmptyAndLiveCopies)
{
    const ResidualRowGeometry geometry(10240);
    const std::array<std::size_t,2> wide{8,10240},narrow{8,2560};
    EXPECT_EQ(geometry.bytesForRows(wide,0),0u);
    EXPECT_THROW(geometry.bytesForRows(narrow,0),std::invalid_argument);
    EXPECT_THROW(geometry.bytesForRows(narrow,3),std::invalid_argument);
    EXPECT_THROW(geometry.bytesForRows(wide,9),std::invalid_argument);
}
/** @test Contiguous admission cannot invent a flattened prefix for padded request-local holes. */
TEST(HostPipelineActivationRows, RejectsRequestHolesAndInvalidGeometry)
{
    const auto device=DeviceId::cpu();
    EXPECT_THROW(HostPipelineActivationRows::admitted(nullptr,device,1,1),std::invalid_argument);
    EXPECT_THROW(HostPipelineActivationRows::admitted(sourceOwner(),device,0,1),std::invalid_argument);
    EXPECT_THROW(HostPipelineActivationRows::admitted(sourceOwner(),device,1,-1),std::invalid_argument);
    const auto partial=HostPipelineActivationRows::admitted(sourceOwner(),device,2,3);
    EXPECT_THROW(partial.rowsFor(sourceOwner(),device,2,8),std::invalid_argument);
    EXPECT_EQ(HostPipelineActivationRows::admitted(sourceOwner(),device,2,0).rowsFor(sourceOwner(),device,2,8),0u);
    EXPECT_EQ(HostPipelineActivationRows::admitted(sourceOwner(),device,2,8).rowsFor(sourceOwner(),device,2,8),16u);
}
/** @test Native graph PP returns no-copy metadata for its canonical receive alias. */
TEST(HostPipelineActivationRows, CanonicalReceiveAliasNeverCopiesOrRequiresHostMarker)
{
    std::size_t copies=0;
    consumeHostPipelineActivationBoundary(HostPipelineBoundaryExecution::Execute,
        HostPipelineBoundarySubmission::Current,[]{return CopyInfo{false,40960};},[&](const auto &){++copies;});
    EXPECT_EQ(copies,0u);
}
/** @test A failed explicit transfer propagates immediately without another submission policy. */
TEST(HostPipelineActivationRows, TransferFailurePropagatesWithoutRetry)
{
    std::size_t copies=0;
    EXPECT_THROW(consumeHostPipelineActivationBoundary(HostPipelineBoundaryExecution::Execute,
        HostPipelineBoundarySubmission::Current,[]{return CopyInfo{true,40960};},[&](const auto &){++copies;throw std::runtime_error("copy failure");}),std::runtime_error);
    EXPECT_EQ(copies,1u);
}
} // namespace llaminar2::test
