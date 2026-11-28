/**
 * @file Test__ROCmNativeM1CaptureABI.cpp
 * @brief Device-free bounds, metadata-value and row-authority checks for native M1 inspection.
 */
#include <gtest/gtest.h>
#include "../../utils/ROCmNativeM1CaptureArguments.h"
#include <array>
#include <type_traits>

using namespace llaminar2;
namespace {
using Producer = rocm::NativeM1ProducerSignature;
using Reducer = rocm::NativeM1ReducerSignature;
static_assert(rocm::NativeM1CaptureABI<Producer>::arity == 11);
static_assert(std::is_same_v<rocm::NativeM1CaptureABI<Producer>::Argument<2>, rocm_native_metadata::MetadataPlanes>);
static_assert(std::is_same_v<rocm::NativeM1CaptureABI<Producer>::Argument<9>, int>);
static_assert(std::is_same_v<rocm::NativeM1CaptureABI<Producer>::Argument<10>, ProjectionRowWork>);
static_assert(rocm::NativeM1CaptureABI<Reducer>::arity == 5);
static_assert(std::is_same_v<rocm::NativeM1CaptureABI<Reducer>::Argument<4>, ProjectionRowWork>);
}

/** @test The producer's exact eleven-slot table preserves metadata and terminal row work. */
TEST(ROCmNativeM1CaptureABI, ProducerMetadataPartitionsAndRowAuthority)
{
    std::array<void*,rocm::NativeM1CaptureABI<Producer>::arity> arguments{};
    std::uint16_t primary=7;
    rocm_native_metadata::MetadataPlanes metadata{};
    metadata.primary=&primary;metadata.superblock_phase=3;metadata.code_bit_shift=5;
    int partitions=4;
    auto work=ProjectionRowWork::fromPrefix(DeviceRowRange::fullyActive(1));
    arguments[2]=&metadata;arguments[9]=&partitions;arguments[10]=&work;
    const auto observed=test::nativeM1CaptureArgument<Producer,2>(arguments.data());
    EXPECT_EQ(observed.primary,&primary);
    EXPECT_EQ(observed.superblock_phase,3u);
    EXPECT_EQ(observed.code_bit_shift,5u);
    EXPECT_EQ((test::nativeM1CaptureArgument<Producer,9>(arguments.data())),partitions);
    EXPECT_EQ((test::nativeM1CaptureArgument<Producer,10>(arguments.data())).captureIdentity(),work.captureIdentity());
    EXPECT_THROW((test::nativeM1CaptureArgument<Producer,0>(arguments.data())),std::runtime_error);
    EXPECT_THROW((test::nativeM1CaptureArgument<Producer,9>(nullptr)),std::runtime_error);
}

/** @test Ordered reducer inspection uses its own five-slot signature. */
TEST(ROCmNativeM1CaptureABI, ReducerPartitionsAndRowAuthority)
{
    std::array<void*,rocm::NativeM1CaptureABI<Reducer>::arity> arguments{};
    int partitions=3;
    auto work=ProjectionRowWork::fromPrefix(DeviceRowRange::fullyActive(1));
    arguments[3]=&partitions;arguments[4]=&work;
    EXPECT_EQ((test::nativeM1CaptureArgument<Reducer,3>(arguments.data())),partitions);
    EXPECT_EQ((test::nativeM1CaptureArgument<Reducer,4>(arguments.data())).captureIdentity(),work.captureIdentity());
}
