/**
 * @file Test__WorkspaceOverwriteContract.cpp
 * @brief Device-free admission and submission-order proof for dirty scratch.
 *
 * Poisoned storage distinguishes a complete live overwrite from prior zero
 * initialization. Failure/exception cases prove readers cannot observe a
 * partially submitted producer. Capacity and changing live extents are checked
 * independently of GPU kernels; captured backend tests prove actual writers.
 */
#include "kernels/common/WorkspaceOverwriteContract.h"
#include "kernels/common/MoEPrefillProjectionExecution.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <type_traits>

using namespace llaminar2;

/** @test Checked construction rejects absent storage, streams and bad extents. */
TEST(WorkspaceOverwriteContract, AdmissionRejectsInvalidBindings)
{
    std::array<float, 8> storage{};
    int stream_token = 0;
    const auto extent = WorkspaceOverwriteExtent::matrix(2, 4);
    static_assert(!std::is_default_constructible_v<WorkspaceOverwriteExtent>);
    static_assert(!std::is_default_constructible_v<WorkspaceOverwrite<float>>);
    static_assert(!std::is_default_constructible_v<EnqueuedWorkspaceRead<float>>);
    EXPECT_THROW((void)WorkspaceOverwriteExtent::matrix(1, 0), std::invalid_argument);
    EXPECT_THROW((void)WorkspaceOverwriteExtent::matrix(1, 1, 0), std::invalid_argument);
    EXPECT_THROW((void)WorkspaceOverwriteExtent::matrix(std::numeric_limits<size_t>::max(), 2), std::overflow_error);
    EXPECT_THROW((void)WorkspaceOverwrite<float>::bind(nullptr, sizeof(storage), extent, &stream_token), std::invalid_argument);
    EXPECT_THROW((void)WorkspaceOverwrite<float>::bind(storage.data(), sizeof(storage), extent, nullptr), std::invalid_argument);
    EXPECT_THROW((void)WorkspaceOverwrite<float>::bind(storage.data(), sizeof(storage) - 1, extent, &stream_token), std::length_error);
    EXPECT_THROW((void)WorkspaceOverwrite<float>::bind(reinterpret_cast<float *>(reinterpret_cast<char *>(storage.data()) + 1),
        sizeof(storage), extent, &stream_token), std::invalid_argument);
    EXPECT_THROW((void)WorkspaceOverwriteExtent::matrix(std::numeric_limits<size_t>::max(), 1)
        .requireCapacity<float>(std::numeric_limits<size_t>::max()), std::overflow_error);
}

/** @test Large-to-empty-to-small writers define live bytes and preserve capacity. */
TEST(WorkspaceOverwriteContract, PoisonedLiveExtentIsOverwrittenBeforeRead)
{
    std::array<int, 17> storage;
    storage.fill(-991);
    int stream_token = 0;
    for (const size_t live : {size_t{16}, size_t{0}, size_t{3}, size_t{1}})
    {
        const auto before = storage;
        bool produced = false;
        const auto binding = WorkspaceOverwrite<int>::bind(storage.data(), sizeof(storage),
            WorkspaceOverwriteExtent::matrix(live, 1), &stream_token);
        ASSERT_TRUE(binding.overwriteThenRead(
            [&](std::span<int> values, void *stream)
            {
                EXPECT_EQ(stream, &stream_token);
                EXPECT_EQ(values.size(), live);
                for (size_t i = 0; i < values.size(); ++i) values[i] = int(i + live);
                produced = true;
                return true;
            },
            [&](const EnqueuedWorkspaceRead<int> &read)
            {
                EXPECT_TRUE(produced);
                EXPECT_EQ(read.stream(), &stream_token);
                EXPECT_EQ(read.values().size(), live);
                static_assert(std::is_const_v<std::remove_reference_t<decltype(read.values()[0])>>);
                for (size_t i = 0; i < read.values().size(); ++i) EXPECT_EQ(read.values()[i], int(i + live));
                return true;
            }));
        for (size_t i = live; i < storage.size(); ++i) EXPECT_EQ(storage[i], before[i]);
    }
}

/** @test A failed or throwing writer cannot admit a reader or publication. */
TEST(WorkspaceOverwriteContract, FailedProducerNeverExposesReadView)
{
    std::array<int, 4> storage{};
    int stream_token = 0;
    const auto binding = WorkspaceOverwrite<int>::bind(storage.data(), sizeof(storage),
        WorkspaceOverwriteExtent::matrix(1, 4), &stream_token);
    int reads = 0;
    const auto consume = [&](const EnqueuedWorkspaceRead<int> &) { ++reads; return true; };
    EXPECT_FALSE(binding.overwriteThenRead([](auto, void *) { return false; }, consume));
    EXPECT_THROW((void)binding.overwriteThenRead([](auto, void *) -> bool {
        throw std::runtime_error("producer submission failed");
    }, consume), std::runtime_error);
    EXPECT_EQ(reads, 0);
    EXPECT_FALSE(binding.overwriteThenRead([](auto, void *) { return true; }, [](const auto &) { return false; }));
}

/** @test Dense/shared token outputs and routed contributions use checked live strides. */
TEST(WorkspaceOverwriteContract, MoEPublicationOwnsExactLiveOutput)
{
    const auto complete = MoEPrefillProjectionExecution::complete(8);
    const auto slice = MoEPrefillProjectionExecution::down(8, 3, 2);
    const auto gate_up = MoEPrefillProjectionExecution::gateUp(8);
    EXPECT_EQ(complete.outputOverwriteExtent(3, 4, MoEPrefillOutputLayout::TokenRows).elements(), 24u);
    EXPECT_EQ(slice.outputOverwriteExtent(3, 4, MoEPrefillOutputLayout::OriginalRouteRows).elements(), 24u);
    EXPECT_EQ(slice.outputOverwriteExtent(0, 4, MoEPrefillOutputLayout::OriginalRouteRows).elements(), 0u);
    EXPECT_THROW((void)gate_up.outputOverwriteExtent(3, 4, MoEPrefillOutputLayout::TokenRows), std::logic_error);
    EXPECT_THROW((void)slice.outputOverwriteExtent(-1, 4, MoEPrefillOutputLayout::TokenRows), std::invalid_argument);
    EXPECT_THROW((void)slice.outputOverwriteExtent(3, 0, MoEPrefillOutputLayout::TokenRows), std::invalid_argument);
    EXPECT_THROW((void)slice.outputOverwriteExtent(3, 4, static_cast<MoEPrefillOutputLayout>(999)), std::invalid_argument);
    std::array<float, 25> storage;
    storage.fill(-991.0f);
    int stream_token = 0, publications = 0;
    auto writer = [&](std::span<float> output, void *stream)
    {
        EXPECT_EQ(stream, &stream_token);
        for (size_t i = 0; i < output.size(); ++i)
            output[i] = i % 4 == 0 ? 0.0f : float(i);
        return true;
    };
    auto publish = [&](const EnqueuedWorkspaceRead<float> &read)
    {
        ++publications;
        EXPECT_EQ(read.stream(), &stream_token);
        EXPECT_EQ(read.values().size(), 24u);
        for (size_t i = 0; i < read.values().size(); ++i)
            EXPECT_EQ(read.values()[i], i % 4 == 0 ? 0.0f : float(i));
        return true;
    };
    ASSERT_TRUE(slice.overwriteOutputThenPublish(storage.data(), sizeof(storage), 3, 4,
        MoEPrefillOutputLayout::OriginalRouteRows, &stream_token, writer, publish));
    EXPECT_EQ(storage.back(), -991.0f);
    EXPECT_EQ(publications, 1);
    ASSERT_FALSE(slice.overwriteOutputThenPublish(storage.data(), sizeof(storage), 3, 4,
        MoEPrefillOutputLayout::OriginalRouteRows, &stream_token,
        [](auto, void *) { return false; }, publish));
    EXPECT_EQ(publications, 1);
    bool produced = false;
    ASSERT_TRUE(gate_up.overwriteOutputThenPublish(nullptr, 0, 3, 4,
        MoEPrefillOutputLayout::OriginalRouteRows, &stream_token,
        [&](std::span<float> output, void *) { EXPECT_TRUE(output.empty()); produced = true; return true; }, publish));
    EXPECT_TRUE(produced);
    EXPECT_EQ(publications, 1);
    EXPECT_THROW((void)slice.overwriteOutputThenPublish(storage.data(), 23 * sizeof(float), 3, 4,
        MoEPrefillOutputLayout::OriginalRouteRows, &stream_token, writer, publish), std::length_error);
}
