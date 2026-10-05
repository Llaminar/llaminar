/**
 * @file Test__NativeGraphStageAnnotation.cpp
 * @brief Device-free lifecycle proof for setup-only native stage attribution.
 *
 * Fake callbacks receive immutable identity only. The tests never initialize a
 * GPU: they prove exact stream/device publication, nested stage ordering,
 * exceptional abort, and rejection of incomplete or repeated completion.
 */
#include <gtest/gtest.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "execution/local_execution/graph/NativeGraphStageAnnotation.h"

using namespace llaminar2;

namespace
{
    /** @brief One observer receipt without any tensor or GPU state. */
    struct Receipt
    {
        unsigned phase, backend;
        int ordinal;
        void *stream;
        std::string name, type;
    };
    thread_local std::vector<Receipt> receipts;

    /** @brief Record the same versioned ABI without calling a backend. */
    void observe(std::uint32_t phase, std::uint32_t backend, int ordinal,
                 void *stream, const char *name, const char *type) noexcept
    {
        receipts.push_back({phase, backend, ordinal, stream, name, type});
    }

    /** @brief Isolate setup receipts across every test. */
    class NativeGraphStageAnnotationTest : public ::testing::Test
    {
    protected:
        /** @brief Reset the fake observer; no production callback is installed. */
        void SetUp() override { receipts.clear(); }
        int stream_storage_ = 0;
        void *stream_ = &stream_storage_;
    };
}

TEST_F(NativeGraphStageAnnotationTest, ServingClosureHasNoObserverDependency)
{
    EXPECT_EQ(nativeGraphStageAnnotationObserver(), nullptr);
}

TEST_F(NativeGraphStageAnnotationTest, CompletePreservesExactGPUAndStageIdentity)
{
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(3)})
    {
        receipts.clear();
        {
            NativeGraphStageAnnotationScope stage(
                observe, device, stream_, "layer.7.ffn.down", "MATMUL");
            stage.complete();
        }
        ASSERT_EQ(receipts.size(), 2u);
        EXPECT_EQ(receipts[0].phase, 0u);
        EXPECT_EQ(receipts[1].phase, 1u);
        for (const auto &receipt : receipts)
        {
            EXPECT_EQ(receipt.backend, device.is_cuda() ? 1u : 2u);
            EXPECT_EQ(receipt.ordinal, device.ordinal);
            EXPECT_EQ(receipt.stream, stream_);
            EXPECT_EQ(receipt.name, "layer.7.ffn.down");
            EXPECT_EQ(receipt.type, "MATMUL");
        }
    }
}

TEST_F(NativeGraphStageAnnotationTest, FailedAndExceptionalStagesAbort)
{
    { NativeGraphStageAnnotationScope stage(observe, DeviceId::cuda(0), stream_, "failed", "COPY"); }
    ASSERT_EQ(receipts.size(), 2u);
    EXPECT_EQ(receipts.back().phase, 2u);
    receipts.clear();
    EXPECT_THROW(([&] {
        NativeGraphStageAnnotationScope stage(observe, DeviceId::cuda(0), stream_, "exception", "COPY");
        throw std::runtime_error("stage failed");
    }()), std::runtime_error);
    ASSERT_EQ(receipts.size(), 2u);
    EXPECT_EQ(receipts.back().phase, 2u);
}

TEST_F(NativeGraphStageAnnotationTest, CompleteCannotBeRepeated)
{
    {
        NativeGraphStageAnnotationScope stage(observe, DeviceId::cuda(0), stream_, "sum", "TP_ALLREDUCE");
        stage.complete();
        EXPECT_THROW(stage.complete(), std::logic_error);
    }
    ASSERT_EQ(receipts.size(), 2u);
    EXPECT_EQ(receipts.back().phase, 1u);
}

TEST_F(NativeGraphStageAnnotationTest, NestedScopesRetireInsideOut)
{
    {
        NativeGraphStageAnnotationScope outer(observe, DeviceId::cuda(0), stream_, "block", "SUBGRAPH");
        {
            NativeGraphStageAnnotationScope inner(observe, DeviceId::cuda(0), stream_, "projection", "MATMUL");
            inner.complete();
        }
        outer.complete();
    }
    ASSERT_EQ(receipts.size(), 4u);
    EXPECT_EQ(receipts[0].name, "block");
    EXPECT_EQ(receipts[1].name, "projection");
    EXPECT_EQ(receipts[2].name, "projection");
    EXPECT_EQ(receipts[2].phase, 1u);
    EXPECT_EQ(receipts[3].name, "block");
    EXPECT_EQ(receipts[3].phase, 1u);
}

TEST_F(NativeGraphStageAnnotationTest, IncompleteAuthorityIsRejectedBeforePublication)
{
    for (const auto device : {DeviceId::cpu(), DeviceId::invalid(), DeviceId::cuda(-1), DeviceId::rocm(-1)})
        EXPECT_THROW(NativeGraphStageAnnotationScope(observe, device, stream_, "stage", "COPY"), std::invalid_argument);
    EXPECT_THROW(NativeGraphStageAnnotationScope(nullptr, DeviceId::cuda(0), stream_, "stage", "COPY"), std::invalid_argument);
    EXPECT_THROW(NativeGraphStageAnnotationScope(observe, DeviceId::cuda(0), nullptr, "stage", "COPY"), std::invalid_argument);
    for (const char *empty : {static_cast<const char *>(nullptr), ""})
    {
        EXPECT_THROW(NativeGraphStageAnnotationScope(observe, DeviceId::cuda(0), stream_, empty, "COPY"), std::invalid_argument);
        EXPECT_THROW(NativeGraphStageAnnotationScope(observe, DeviceId::cuda(0), stream_, "stage", empty), std::invalid_argument);
    }
    EXPECT_TRUE(receipts.empty());
}
