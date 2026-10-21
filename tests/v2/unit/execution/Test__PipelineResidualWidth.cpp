/**
 * @file Test__PipelineResidualWidth.cpp
 * @brief Device-free regression for widened PP residuals and exact producer geometry.
 *
 * No model or accelerator is needed: the authenticated tensor shape must keep
 * the 10240-column QFN trunk distinct from its 2560-column compute/embedding path.
 */
#include "execution/local_execution/orchestrators/PipelineResidualWidth.h"
#include "execution/prefix_cache/ResidualRowGeometry.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <cstring>
#include <vector>

using namespace llaminar2;

/** @test Producer columns, independent of request rows, define PP payload width. */
TEST(PipelineResidualWidth, PreservesWidenedTrunkAndOrdinaryModelWidths)
{
    for (const auto rows : {std::size_t{1}, std::size_t{7}, std::size_t{128}})
        for (const auto width : {std::size_t{2560}, std::size_t{10240}})
            EXPECT_EQ(pipelineResidualWidth(std::array{rows, width}, TensorType::FP32), width);
}

/** @test Missing, padded-rank and unsupported scalar contracts cannot guess an embedding width. */
TEST(PipelineResidualWidth, RejectsInvalidProducerGeometry)
{
    EXPECT_THROW((void)pipelineResidualWidth({}, TensorType::FP32), std::invalid_argument);
    EXPECT_THROW((void)pipelineResidualWidth(std::array<std::size_t, 1>{10240}, TensorType::FP32), std::invalid_argument);
    EXPECT_THROW((void)pipelineResidualWidth(std::array<std::size_t, 3>{1, 2, 10240}, TensorType::FP32), std::invalid_argument);
    EXPECT_THROW((void)pipelineResidualWidth(std::array<std::size_t, 2>{0, 10240}, TensorType::FP32), std::invalid_argument);
    EXPECT_THROW((void)pipelineResidualWidth(std::array<std::size_t, 2>{1, 0}, TensorType::FP32), std::invalid_argument);
    EXPECT_THROW((void)pipelineResidualWidth(std::array<std::size_t, 2>{1, std::numeric_limits<std::size_t>::max()}, TensorType::FP32), std::invalid_argument);
    EXPECT_THROW((void)pipelineResidualWidth(std::array<std::size_t, 2>{1, 10240}, TensorType::FP16), std::invalid_argument);
}

/** @test Partial, empty and large-to-small replay sizes never communicate unused bucket rows. */
TEST(PipelineResidualWidth, PrefillPayloadUsesOnlyLiveRowsAcrossRetainedBuckets)
{
    const PPStageTransferContract contract{
        .embedding_dim = 10240, .dtype = ActivationDType::FP32, .max_seq_len = 128};
    for (int replay = 0; replay < 20; ++replay)
        for (const int actual : {128, 7, 0, 63, 1})
        {
            const PrefillChunkPlan chunk{.real_count = actual, .bucket_seq_len = 128};
            EXPECT_EQ(pipelinePrefillLiveBytes(contract, chunk), static_cast<std::size_t>(actual) * 10240 * sizeof(float));
        }
    EXPECT_THROW((void)pipelinePrefillLiveBytes(contract,
        PrefillChunkPlan{.real_count = 129, .bucket_seq_len = 128}), std::invalid_argument);
    EXPECT_THROW((void)pipelinePrefillLiveBytes(contract,
        PrefillChunkPlan{.real_count = 7, .bucket_seq_len = 256}), std::invalid_argument);
}

/** @test Ordinary PP copies every branch while retaining unused bucket rows byte-for-byte. */
TEST(PipelineResidualWidth, DirectConsumerCopyKeepsAllResidualBranchesAndOnlyLiveRows)
{
    constexpr std::size_t capacity=15;
    for(const int width:{2560,10240})
    {
        const ResidualRowGeometry geometry(width);
        const std::array shape{capacity,static_cast<std::size_t>(width)};
        std::vector<float> source(capacity*width),destination(capacity*width,-99.0f);
        for(std::size_t i=0;i<source.size();++i)source[i]=static_cast<float>(i%32768)+.25f;
        for(const std::size_t live:{15,1,7,0,2})
        {
            std::fill(destination.begin(),destination.end(),-99.0f);
            const auto bytes=geometry.bytesForRows(shape,live);
            std::memcpy(destination.data(),source.data(),bytes);
            EXPECT_EQ(bytes,live*static_cast<std::size_t>(width)*sizeof(float));
            EXPECT_TRUE(std::equal(source.begin(),source.begin()+live*width,destination.begin()));
            EXPECT_TRUE(std::all_of(destination.begin()+live*width,destination.end(),
                [](float v){return v==-99.0f;}));
            if(width==10240 && live)
                EXPECT_EQ(destination[(live-1)*width+width-1],source[(live-1)*width+width-1]);
        }
    }
}

/** @test Shifted-prefill selects its admitted wide MTP role and preserves ordinary H-wide wiring. */
TEST(PipelineResidualWidth, ShiftedPrefillChoosesScratchByDeclaredResidualWidth)
{
    const std::array<std::size_t,2> compute{15,2560},wide{15,10240};
    const ResidualRowGeometry ordinary(2560),qfn(10240);
    EXPECT_EQ(ordinary.scratchSource(2560,compute,{},15),ResidualRowGeometry::ScratchSource::ComputeRows);
    EXPECT_EQ(qfn.scratchSource(2560,compute,wide,15),ResidualRowGeometry::ScratchSource::ResidualRows);
    EXPECT_EQ(qfn.columns(),10240);
    EXPECT_EQ(qfn.bytesForRows(wide,15),15u*10240u*sizeof(float));
    // A missing wide owner cannot silently select the much smaller main normalized scratch.
    EXPECT_THROW((void)qfn.scratchSource(2560,compute,{},15),std::invalid_argument);
    EXPECT_THROW((void)qfn.scratchSource(2560,compute,compute,15),std::invalid_argument);
    EXPECT_THROW((void)qfn.scratchSource(0,compute,wide,15),std::invalid_argument);
}

/** @test Narrow or undersized source/destination/archive operands fail before the copy is submitted. */
TEST(PipelineResidualWidth, RejectsNarrowedResidualBoundaryBeforeAnyCopy)
{
    const ResidualRowGeometry geometry(10240);
    const std::array<std::size_t,2> source{3,10240},narrow{3,2560},shortRows{1,10240};
    std::vector<float> output(3*10240,-123.0f);
    EXPECT_THROW((void)geometry.bytesForRows(narrow,3),std::invalid_argument);
    EXPECT_THROW((void)geometry.bytesForRows(shortRows,3),std::invalid_argument);
    EXPECT_THROW((void)geometry.scratchSource(2560,narrow,shortRows,3),std::invalid_argument);
    EXPECT_EQ(geometry.bytesForRows(source,0),0u);
    EXPECT_TRUE(std::all_of(output.begin(),output.end(),[](float v){return v==-123.0f;}));
    const std::array<std::size_t,2> huge{std::numeric_limits<std::size_t>::max(),10240};
    EXPECT_THROW((void)geometry.bytesForRows(huge,huge[0]),std::overflow_error);
}
