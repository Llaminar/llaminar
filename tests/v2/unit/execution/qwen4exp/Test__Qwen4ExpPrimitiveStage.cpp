/**
 * @file Test__Qwen4ExpPrimitiveStage.cpp
 * @brief Device-free graph-stage binding, arithmetic and immutable-history regressions.
 *
 * The tests enter the real stage/factory path, including mandatory buffer
 * contracts, native scalar types, byte-offset control subviews, distinct branch
 * gains and packed exact-64-bit lookup IDs. Invalid capacity, alignment and
 * aliases fail at construction, before any graph operation can write data.
 */
#include <gtest/gtest.h>
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPrimitiveStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"
#include "snapshots/SnapshotCapture.h"
#include "utils/OpenMPUtils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>
#include <memory>
#include <string_view>

using namespace llaminar2;
namespace
{
using Stage = Qwen4ExpPrimitiveStage;
using Float = Qwen4ExpFloatBinding;
using Int = Qwen4ExpIntBinding;
using Weight = Qwen4ExpImmutableWeightBinding;
/** @brief Exact fixture partitions; row IDs remain full 64-bit values. */
qwen4exp::NgramHashPolicy hashPolicy()
{
    qwen4exp::NgramHashPolicy policy;
    policy.multipliers = {9007199254740993ULL,9007199254740995ULL,9007199254740997ULL};
    for (int head = 0; head < 16; ++head)
    {
        policy.moduli[head] = 1009 + 2 * head;
        policy.offsets[head] = (std::uint64_t{1} << 40) + static_cast<std::uint64_t>(head)*2000;
    }
    return policy;
}
} // namespace

/** @test The complete closed variant set reaches real CPU kernels through the central factory. */
TEST(Qwen4ExpPrimitiveStage, EverySemanticOperationHasMandatoryContractsAndRuns)
{
    llaminar::v2::ThreadCountGuard team(1);
    CPUDeviceContext context(DeviceId::cpu(),1);
    constexpr int rows = 3, hidden = 7, channels = 4*hidden;
    FP32Tensor input({128,128},DeviceId::cpu()), gain({128,128},DeviceId::cpu()), output({128,128},DeviceId::cpu());
    INT32Tensor tokens({128,128}), history({128,128}), ids({128,128});
    std::fill_n(input.mutable_data(),input.numel(),0.25f);
    std::fill_n(gain.mutable_data(),gain.numel(),1.0f);
    std::fill_n(output.mutable_data(),output.numel(),-999.0f);
    std::fill_n(static_cast<std::int32_t *>(tokens.raw_mutable_data()),tokens.numel(),17);
    std::fill_n(static_cast<std::int32_t *>(history.raw_mutable_data()),history.numel(),248044);
    const Float in(input,BufferId::NORMALIZED), out(output,BufferId::FFN_OUTPUT);
    const Weight weights(gain);
    const Int tok(tokens,BufferId::REQUEST_TOKEN_IDS), hist(history,BufferId::PREFILL_CHUNK_TOKEN_IDS), id(ids,BufferId::MOE_EXPERT_INDICES);
    const std::vector<Stage::Operation> operations{
        Stage::BranchRMSNorm{hidden,1e-6f,in,weights,out},
        Stage::ResidualExpand{hidden,in,out}, Stage::ResidualDown{hidden,in,out},
        Stage::ResidualRead{hidden,in,in,out}, Stage::ResidualWrite{hidden,in,in,in,out},
        Stage::NgramHash{tok,hist,id,hashPolicy()}, Stage::PLEGate{hidden,in,in,in,out},
        Stage::PLEConv{channels,in,in,weights,in,in,out},
        Stage::CommitConvHistory{channels,{{},2},in,in,out},
        Stage::CommitTokenHistory{{{},2},hist,tok,id}};
    const std::vector<ComputeStageType> types{
        ComputeStageType::QWEN4EXP_BRANCH_RMS_NORM,ComputeStageType::QWEN4EXP_RESIDUAL_EXPAND,
        ComputeStageType::QWEN4EXP_RESIDUAL_DOWN,ComputeStageType::QWEN4EXP_RESIDUAL_READ,
        ComputeStageType::QWEN4EXP_RESIDUAL_WRITE,ComputeStageType::QWEN4EXP_NGRAM_HASH,
        ComputeStageType::QWEN4EXP_PLE_GATE,ComputeStageType::QWEN4EXP_PLE_CONV,
        ComputeStageType::QWEN4EXP_COMMIT_CONV_HISTORY,ComputeStageType::QWEN4EXP_COMMIT_TOKEN_HISTORY};
    for (std::size_t index = 0; index < operations.size(); ++index)
    {
        Stage stage({.rows={rows,{}},.operation=operations[index]});
        EXPECT_EQ(stage.type(),types[index]);
        EXPECT_STRNE(computeStageTypeName(stage.type()),"UNKNOWN");
        EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
        EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
        EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::GPU_ROCM));
        const auto contract = stage.bufferContract();
        EXPECT_FALSE(contract.empty());
        EXPECT_FALSE(contract.inputs.empty());
        EXPECT_EQ(contract.outputs.size(),1u);
        for(const auto& output:contract.outputs)EXPECT_TRUE(output.prepare_write_storage);
        EXPECT_TRUE(stage.getBufferRequirements().empty());
        const auto dump = stage.buildDumpInfoImpl();
        EXPECT_EQ(dump.outputs.size(),1u);
        EXPECT_GT(stage.estimatedMemoryBytes(),0u);
        EXPECT_TRUE(stage.execute(&context));
        EXPECT_THROW(stage.execute(nullptr),std::invalid_argument);
    }
}

/** @test Branch gains, live control subviews, output offsets and inactive bytes agree. */
TEST(Qwen4ExpPrimitiveStage, BranchNormCountSubviewsAndDiagnosticOffsets)
{
    llaminar::v2::ThreadCountGuard team(1);
    CPUDeviceContext context(DeviceId::cpu(),1);
    constexpr int rows=3,hidden=19,channels=4*hidden;
    FP32Tensor input({rows*channels+2},DeviceId::cpu()), gain({channels},DeviceId::cpu()), output({rows*channels+8},DeviceId::cpu());
    INT32Tensor control({4});
    auto *words=static_cast<std::int32_t *>(control.raw_mutable_data());
    words[0]=-99;words[1]=-99;words[2]=2;words[3]=1;
    std::fill_n(input.mutable_data(),input.numel(),2.0f);
    std::fill_n(output.mutable_data(),output.numel(),-12345.0f);
    for(int branch=0;branch<4;++branch)
        std::fill_n(gain.mutable_data()+branch*hidden,hidden,static_cast<float>(branch+1));
    Stage stage({.rows={rows,Int(control,BufferId::PREFILL_CHUNK_GEOMETRY,8)},
        .operation=Stage::BranchRMSNorm{hidden,1e-6f,
            Float(input,BufferId::NORMALIZED,8),Weight(gain),Float(output,BufferId::FFN_OUTPUT,16)}});
    ASSERT_TRUE(stage.execute(&context));
    const auto *actual=output.data();
    for(int index=0;index<4;++index) EXPECT_EQ(actual[index],-12345);
    for(int row=0;row<2;++row)
        for(int branch=0;branch<4;++branch)
            for(int channel=0;channel<hidden;++channel)
                EXPECT_NEAR(actual[4+row*channels+branch*hidden+channel],
                            2.0f/std::sqrt(4.0f+1e-6f)*(branch+1),1e-6);
    for(std::size_t index=4+2*channels;index<output.numel();++index) EXPECT_EQ(actual[index],-12345);
    auto dump=stage.buildDumpInfoImpl();
    ASSERT_EQ(dump.inputs.size(),2u);
    ASSERT_EQ(dump.weights.size(),1u);
    EXPECT_EQ(dump.inputs[0].tensor_byte_offset,8u);
    EXPECT_EQ(dump.inputs[0].data,words+2);
    EXPECT_EQ(dump.outputs[0].tensor_byte_offset,16u);
    EXPECT_EQ(dump.outputs[0].data,actual+4);
    dump.ensureOutputsOnHost();
    EXPECT_EQ(dump.outputs[0].data,actual+4);
    EXPECT_EQ(dump.outputs[0].byte_size,rows*channels*sizeof(float));
}

/** @test Packed row IDs retain integer bits above INT32 and honor eight-byte subviews. */
TEST(Qwen4ExpPrimitiveStage, HashPackedInt64WordsAndAcceptedTokenHistory)
{
    llaminar::v2::ThreadCountGuard team(1);
    CPUDeviceContext context(DeviceId::cpu(),1);
    INT32Tensor tokens({3}), prior({2}), ids({3*32+4}), output({2}), control({4});
    auto *rawTokens=static_cast<std::int32_t *>(tokens.raw_mutable_data());
    rawTokens[0]=17;rawTokens[1]=248044;rawTokens[2]=19;
    auto *rawPrior=static_cast<std::int32_t *>(prior.raw_mutable_data());rawPrior[0]=7;rawPrior[1]=11;
    auto *rawControl=static_cast<std::int32_t *>(control.raw_mutable_data());rawControl[2]=3;rawControl[3]=1;
    auto policy=hashPolicy();
    Stage hash({.rows={3,Int(control,BufferId::PREFILL_CHUNK_GEOMETRY,8)},
        .operation=Stage::NgramHash{Int(tokens,BufferId::REQUEST_TOKEN_IDS),Int(prior,BufferId::PREFILL_CHUNK_TOKEN_IDS),
            Int(ids,BufferId::MOE_EXPERT_INDICES,8),policy}});
    ASSERT_TRUE(hash.execute(&context));
    const auto descriptor = hash.buildDumpInfoImpl();
    ASSERT_EQ(descriptor.outputs.size(), 1u);
    EXPECT_STREQ(descriptor.outputs.front().dtype, "INT64");
    EXPECT_EQ(descriptor.outputs.front().rows, 3u);
    EXPECT_EQ(descriptor.outputs.front().cols, 16u);
    EXPECT_EQ(descriptor.outputs.front().element_size, 8u);
    EXPECT_EQ(descriptor.outputs.front().tensor_byte_offset, 8u);
    EXPECT_EQ(descriptor.outputs.front().byte_size, 3u * 16u * 8u);
    EXPECT_EQ(descriptor.outputs.front().data,
        static_cast<const std::byte *>(ids.raw_data()) + 8);
    const auto *actual=reinterpret_cast<const std::int64_t *>(static_cast<const std::int32_t *>(ids.raw_data())+2);
    for(int row=0;row<3;++row)
        for(int head=0;head<16;++head)
        {
            EXPECT_GE(static_cast<std::uint64_t>(actual[row*16+head]),policy.offsets[head]);
            EXPECT_LT(static_cast<std::uint64_t>(actual[row*16+head]),policy.offsets[head]+policy.moduli[head]);
        }
    Stage commit({.rows={3,Int(control,BufferId::PREFILL_CHUNK_GEOMETRY,8)},
        .operation=Stage::CommitTokenHistory{{Int(control,BufferId::PREFILL_CHUNK_GEOMETRY,12),0},
            Int(prior,BufferId::PREFILL_CHUNK_TOKEN_IDS),Int(tokens,BufferId::REQUEST_TOKEN_IDS),Int(output,BufferId::MOE_EXPERT_WEIGHTS)}});
    ASSERT_TRUE(commit.execute(&context));
    const auto *committed=static_cast<const std::int32_t *>(output.raw_data());
    EXPECT_EQ(committed[0],11);EXPECT_EQ(committed[1],17);
    EXPECT_EQ(rawPrior[0],7);EXPECT_EQ(rawPrior[1],11);
    EXPECT_EQ(commit.bufferContract().inputs.size(),3u); // live/accepted share one canonical control ID.
}

/** @test Type, capacity, alias and packed-ID alignment failures occur before execution. */
TEST(Qwen4ExpPrimitiveStage, RejectsInvalidArenaViewsAtConstruction)
{
    FP32Tensor small({8},DeviceId::cpu()), large({1024},DeviceId::cpu());
    INT32Tensor integer({1024});
    EXPECT_THROW((Stage({.rows={1,{}},.operation=Stage::ResidualRead{7,
        Float(small,BufferId::NORMALIZED),Float(large,BufferId::GATE_PROJ),Float(large,BufferId::FFN_OUTPUT)}})),std::invalid_argument);
    EXPECT_THROW((Stage({.rows={1,{}},.operation=Stage::ResidualDown{7,
        Float(integer,BufferId::NORMALIZED),Float(large,BufferId::FFN_OUTPUT)}})),std::invalid_argument);
    EXPECT_THROW((Stage({.rows={1,{}},.operation=Stage::ResidualDown{7,
        Float(large,BufferId::NORMALIZED),Float(large,BufferId::NORMALIZED,4)}})),std::invalid_argument);
    EXPECT_THROW((Stage({.rows={1,{}},.operation=Stage::CommitConvHistory{4,{{},1},
        Float(large,BufferId::NORMALIZED),Float(small,BufferId::GATE_PROJ),Float(large,BufferId::NORMALIZED)}})),std::invalid_argument);
    EXPECT_THROW((Stage({.rows={1,{}},.operation=Stage::NgramHash{
        Int(integer,BufferId::REQUEST_TOKEN_IDS),Int(integer,BufferId::PREFILL_CHUNK_TOKEN_IDS,8),
        Int(integer,BufferId::MOE_EXPERT_INDICES,4),hashPolicy()}})),std::invalid_argument);
    EXPECT_THROW((Stage({.rows={0,{}},.operation=Stage::ResidualDown{7,
        Float(large,BufferId::NORMALIZED),Float(small,BufferId::FFN_OUTPUT)}})),std::invalid_argument);
}

/** @test Session-owned tokens/counts stay native and cannot acquire another backend owner. */
TEST(Qwen4ExpPrimitiveStage, NativeExecutionInputsReplayWithoutCopyingOrMutatingTheirOwner)
{
    llaminar::v2::ThreadCountGuard team(1);
    CPUDeviceContext context(DeviceId::cpu(),1);
    std::array<std::int32_t,3> tokens{17,248044,23};
    std::int32_t live=3,accepted=1;
    INT32Tensor prior({2}),next({2}),ids({3*32});
    auto *history=static_cast<std::int32_t *>(prior.raw_mutable_data());
    history[0]=7;history[1]=11;
    const auto native=[&](const std::int32_t *data,std::size_t count) {
        return Qwen4ExpNativeIntInput{DeviceId::cpu(),data,count};
    };
    Stage hash({.rows={3,native(&live,1)},.operation=Stage::NgramHash{
        native(tokens.data(),tokens.size()),Int(prior,BufferId::QFN_PLE_TOKEN_HISTORY),
        Int(ids,BufferId::QFN_NGRAM_ROW_IDS),hashPolicy()}});
    Stage commit({.rows={3,native(&live,1)},.operation=Stage::CommitTokenHistory{
        {native(&accepted,1),0},Int(prior,BufferId::QFN_PLE_TOKEN_HISTORY),
        native(tokens.data(),tokens.size()),Int(next,BufferId::MTP_CONDITION_TOKEN)}});
    ASSERT_TRUE(hash.execute(&context));
    ASSERT_TRUE(commit.execute(&context));
    const auto *output=static_cast<const std::int32_t *>(next.raw_data());
    EXPECT_EQ(output[0],11);EXPECT_EQ(output[1],17);
    live=1;accepted=1;tokens[0]=31;
    ASSERT_TRUE(hash.execute(&context));ASSERT_TRUE(commit.execute(&context));
    EXPECT_EQ(output[0],11);EXPECT_EQ(output[1],31);
    EXPECT_EQ(history[0],7);EXPECT_EQ(history[1],11);
    EXPECT_EQ(tokens[1],248044);
    EXPECT_THROW((Stage({.rows={3,{}},.operation=Stage::NgramHash{
        Qwen4ExpNativeIntInput{DeviceId::cuda(0),tokens.data(),3},
        Int(prior,BufferId::QFN_PLE_TOKEN_HISTORY),Int(ids,BufferId::QFN_NGRAM_ROW_IDS),hashPolicy()}})),std::invalid_argument);
    EXPECT_THROW((Stage({.rows={3,{}},.operation=Stage::NgramHash{
        native(tokens.data(),2),Int(prior,BufferId::QFN_PLE_TOKEN_HISTORY),
        Int(ids,BufferId::QFN_NGRAM_ROW_IDS),hashPolicy()}})),std::invalid_argument);
}

/** @test No operation may assume a write-only arena result already has native storage. */
TEST(Qwen4ExpPrimitiveStage, WriteOnlyArenaResultsRequireColdPreparation) {
 FP32Tensor input({2,7}),output({2,28});
 Stage expand({.rows={2,{}},.operation=Stage::ResidualExpand{7,Float(input,BufferId::QFN_EMBEDDING_ROWS),Float(output,BufferId::HIDDEN_STATE)}});
 const auto c=expand.bufferContract();ASSERT_EQ(c.outputs.size(),1u);EXPECT_TRUE(c.outputs[0].prepare_write_storage);
 EXPECT_EQ(c.writesRequiringPrepare().size(),1u);
 Stage alias({.rows={2,{}},.operation=Stage::ResidualDown{7,Float(input,BufferId::NORMALIZED),Float(input,BufferId::NORMALIZED)}});
 const auto inout=alias.bufferContract();EXPECT_TRUE(inout.outputs.empty());ASSERT_EQ(inout.inouts.size(),1u);
 EXPECT_FALSE(inout.inouts[0].prepare_write_storage);EXPECT_TRUE(inout.writesRequiringPrepare().empty());
}

/** @test Actual QFN branch norm publishes the exact wide readonly transaction input. */
TEST(Qwen4ExpPrimitiveStage, MTPTerminalHiddenPublicationRetainsOriginalWideInput)
{
    llaminar::v2::ThreadCountGuard team(1);
    CPUDeviceContext context(DeviceId::cpu(), 1);
    constexpr int hidden = 2560, channels = 10240;
    FP32Tensor input({channels + 2}, DeviceId::cpu()), gain({channels}, DeviceId::cpu());
    FP32Tensor output({channels + 4}, DeviceId::cpu());
    for (int index = 0; index < channels; ++index)
        input.mutable_data()[index + 2] = 0.25f + float(index % 37) / 16.0f;
    std::fill_n(gain.mutable_data(), channels, 1.0f);
    const std::vector<float> original(input.data() + 2, input.data() + channels + 2);
    const auto make = [&](Stage::DiagnosticInputPublication policy)
    {
        return Stage::Params{.rows = {1, {}}, .operation = Stage::BranchRMSNorm{hidden, 1e-6f,
            Float(input, BufferId::PREFIX_TERMINAL_HIDDEN, 8), Weight(gain),
            Float(output, BufferId::MTP_NORM_HIDDEN, 16), policy}};
    };
    Stage ordinary(make(Stage::DiagnosticInputPublication::None));
    Stage published(make(Stage::DiagnosticInputPublication::MTPTerminalHidden));
    EXPECT_EQ(ordinary.buildDumpInfoImpl().outputs.size(), 1u);
    EXPECT_EQ(ordinary.graphCaptureVariantSignature(), 0u);
    EXPECT_NE(published.graphCaptureVariantSignature(), ordinary.graphCaptureVariantSignature());
    const auto contract = published.bufferContract();
    EXPECT_EQ(contract.outputs.size(), 1u); // The alias is never an execution write.
    EXPECT_EQ(contract.inputs.size(), 1u);
    EXPECT_EQ(contract.inputs.front().id, BufferId::PREFIX_TERMINAL_HIDDEN);
    ASSERT_TRUE(published.execute(&context));
    EXPECT_TRUE(std::equal(original.begin(), original.end(), input.data() + 2));
    const auto dump = published.buildDumpInfoImpl();
    const auto view = std::find_if(dump.outputs.begin(), dump.outputs.end(), [](const auto &value) {
        return std::string_view(value.name) == "mtp_terminal_hidden_input";
    });
    ASSERT_NE(view, dump.outputs.end());
    EXPECT_EQ(view->tensor, &input);
    EXPECT_EQ(view->tensor_byte_offset, 8u);
    EXPECT_EQ(view->data, input.data() + 2);
    EXPECT_EQ(view->rows, 1u);
    EXPECT_EQ(view->cols, static_cast<std::size_t>(channels));
    EXPECT_EQ(view->byte_size, channels * sizeof(float));
    SnapshotCapture capture;
    capture.captureStage("mtp_decode_sidecar_device_target_token_live_position::MTP0_norm_hidden", dump);
    const auto *consumed = capture.get(
        "MTP_DECODE_SIDECAR_DEVICE_TARGET_TOKEN_LIVE_POSITION_MTP_TERMINAL_HIDDEN_ROW_SELECT");
    ASSERT_NE(consumed, nullptr);
    EXPECT_EQ(consumed->data, original);
    const auto *normalized = capture.get(
        "MTP_DECODE_SIDECAR_DEVICE_TARGET_TOKEN_LIVE_POSITION_MTP0_NORM_HIDDEN");
    ASSERT_NE(normalized, nullptr);
    EXPECT_NE(normalized->data, original);
    EXPECT_EQ(input.data()[2], original.front());
}

/** @test Unknown policy and in-place mutation cannot masquerade as original readonly input. */
TEST(Qwen4ExpPrimitiveStage, MTPTerminalHiddenPublicationRejectsInvalidOrInPlaceViews)
{
    FP32Tensor input({28}, DeviceId::cpu()), gain({28}, DeviceId::cpu());
    const auto params = [&](Stage::DiagnosticInputPublication policy)
    {
        return Stage::Params{.rows = {1, {}}, .operation = Stage::BranchRMSNorm{7, 1e-6f,
            Float(input, BufferId::MTP_NORM_HIDDEN), Weight(gain),
            Float(input, BufferId::MTP_NORM_HIDDEN), policy}};
    };
    EXPECT_NO_THROW((void)Stage(params(Stage::DiagnosticInputPublication::None)));
    EXPECT_THROW((void)Stage(params(Stage::DiagnosticInputPublication::MTPTerminalHidden)), std::invalid_argument);
    EXPECT_THROW((void)Stage(params(static_cast<Stage::DiagnosticInputPublication>(255))), std::invalid_argument);
}

/** @test Distinct tensor views cannot hide physical overlap from execute-time publication checks. */
TEST(Qwen4ExpPrimitiveStage, MTPTerminalHiddenPublicationChecksDistinctPhysicalViewsBeforeWrites)
{
    llaminar::v2::ThreadCountGuard team(1);
    CPUDeviceContext context(DeviceId::cpu(), 1);
    constexpr int hidden = 7, channels = 28;
    for (const std::size_t output_start : {std::size_t{5}, std::size_t{28}})
    {
        SCOPED_TRACE(output_start);
        auto parent = std::make_shared<FP32Tensor>(std::vector<std::size_t>{70}, DeviceId::cpu());
        for (std::size_t index = 0; index < parent->numel(); ++index)
            parent->mutable_data()[index] = 0.25f + float(index % 13) / 16.0f;
        FP32Tensor gain({channels}, DeviceId::cpu());
        std::fill_n(gain.mutable_data(), channels, 1.0f);
        // Distinct TensorBase objects and distinct logical buffer roles pass
        // constructor alias checks. Their original parent determines physical
        // overlap; binding offsets are deliberately nonzero in both views.
        auto input = parent->create_view({channels + 2}, 2);
        auto output = parent->create_view({channels + 4}, output_start);
        ASSERT_NE(input, nullptr);
        ASSERT_NE(output, nullptr);
        ASSERT_NE(input.get(), output.get());
        const std::vector<float> original(parent->data(), parent->data() + parent->numel());
        Stage published({.rows = {1, {}}, .operation = Stage::BranchRMSNorm{hidden, 1e-6f,
            Float(*input, BufferId::PREFIX_TERMINAL_HIDDEN, 8), Weight(gain),
            Float(*output, BufferId::MTP_NORM_HIDDEN, 16),
            Stage::DiagnosticInputPublication::MTPTerminalHidden}});
        const auto dump = published.buildDumpInfoImpl();
        const auto view = std::find_if(dump.outputs.begin(), dump.outputs.end(), [](const auto &value) {
            return std::string_view(value.name) == "mtp_terminal_hidden_input";
        });
        ASSERT_NE(view, dump.outputs.end());
        EXPECT_EQ(view->tensor, input.get());
        EXPECT_EQ(view->tensor_byte_offset, 8u);
        EXPECT_EQ(view->data, parent->data() + 4);
        if (output_start == 5)
        {
            // [4,32) versus [9,37): only the physical guard can reject.
            EXPECT_THROW((void)published.execute(&context), std::invalid_argument);
            EXPECT_TRUE(std::equal(original.begin(), original.end(), parent->data()));
        }
        else
        {
            // [4,32) versus [32,60) is adjacent, not overlapping.
            ASSERT_TRUE(published.execute(&context));
            EXPECT_TRUE(std::equal(original.begin() + 4, original.begin() + 32, parent->data() + 4));
            EXPECT_FALSE(std::equal(original.begin() + 32, original.begin() + 60, parent->data() + 32));
        }
    }
}
