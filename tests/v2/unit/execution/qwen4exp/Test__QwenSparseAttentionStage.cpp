/**
 * @file Test__QwenSparseAttentionStage.cpp
 * @brief Device-free sparse stage contracts, native cache layouts and acceptance state.
 *
 * Tests enter the real stage/factory path and real cache native accessors. Native
 * storage is imported before execution; selected reads use its physical ring
 * bytes directly. Independent zero-query means expose codec/basis/layout errors
 * without sharing a softmax implementation with the production kernel.
 */
#include <gtest/gtest.h>
#include "execution/compute_stages/stages/qwen4exp/QwenSparseAttentionStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "execution/local_execution/graph/StageVerifier.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "kernels/qwen4exp/QwenSparseAttentionArithmetic.h"
#include <stdexcept>
#include "tensors/Tensors.h"
#include "tensors/TensorVerification.h"
#include "tensors/GPUTensorVerification.h"
#include "utils/DebugEnv.h"
#include "utils/MPIContext.h"
#include "utils/OpenMPUtils.h"
#include "../../../integration/kernels/qwen4exp/QwenSparseNativeCodecFixture.h"
#include "../../../integration/kernels/qwen4exp/QwenSparseSelectionCases.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>

using namespace llaminar2;
namespace
{
using Stage=QwenSparseAttentionStage;
using Float=Qwen4ExpFloatBinding;
using Int=Qwen4ExpIntBinding;
using Weight=Qwen4ExpImmutableWeightBinding;
/** @brief Canonical source fixture packed in the cache's declared native layout. */
std::vector<unsigned char> pack(const qwen4exp::test::NativeCodecFixture &fixture, int tokens, bool headMajor, bool anchor)
{
    const auto anchorBytes=anchor ? fixture.anchor.size()*sizeof(float) : 0;
    std::vector<unsigned char> result(anchorBytes+fixture.storage.size());
    if (anchor) std::memcpy(result.data(),fixture.anchor.data(),anchorBytes);
    for (int token=0;token<tokens;++token) for (int head=0;head<2;++head) {
        const auto source=(std::size_t(token)*2+head)*fixture.head_bytes;
        const auto target=(headMajor ? std::size_t(head)*tokens+token : std::size_t(token)*2+head)*fixture.head_bytes;
        std::memcpy(result.data()+anchorBytes+target,fixture.storage.data()+source,fixture.head_bytes);
    }
    return result;
}
}

#if LLAMINAR_ASSERTIONS_ACTIVE
namespace
{
/** @brief Restore the exact parsed validation policy after a strict CPU stage observation. */
class ScopedSparseTailValidation final
{
public:
    /** @brief Keep finite, zero and storage checks enabled, without writing failure dumps. */
    ScopedSparseTailValidation() : saved_(debugEnv().validation)
    {
        auto &config = mutableDebugEnv().validation;
        config.validate_buffers = true;
        config.fail_on_zero = true;
        config.fail_on_nan = true;
        config.dump_on_failure = false;
        config.sample_rows = 4;
    }
    /** @brief Restore the previous parsed policy without mutating process environment. */
    ~ScopedSparseTailValidation() { mutableDebugEnv().validation = saved_; }
private:
    ValidationConfig saved_;
};
}

/**
 * @test Complete pools leave a valid zero accepted-tail bank under strict verification.
 *
 * The real CPU stage must preserve unowned suffix bytes, publish each nonempty
 * tail, and still reject nonfinite authored state. The physical bank is not a
 * requirement to produce four nonzero rows on every publication.
 */
TEST(QwenSparseAttentionStage, CompletePoolZeroTailRetainsStrictFiniteValidation)
{
    ScopedSparseTailValidation validation;
    llaminar::v2::ThreadCountGuard threads(1);
    CPUDeviceContext context(DeviceId::cpu(), 1);
    constexpr int rows = 8, width = 128;
    FP32Tensor raw({rows, width}), partial({4, width});
    INT32Tensor replay({3}), accepted({1});
    auto *state = static_cast<int *>(replay.raw_mutable_data());
    auto *count = static_cast<int *>(accepted.raw_mutable_data());
    state[0] = 0; state[1] = rows; state[2] = 0;
    for (int row = 0; row < rows; ++row)
        std::fill_n(raw.mutable_data() + row * width, width, float(row + 1));
    ComputeNode node("accepted_sparse_state", std::make_unique<Stage>(Stage::Params{
        .rowCapacity = rows, .tokenCapacity = 16,
        .replay = Int(replay, BufferId::QFN_QSA_REPLAY),
        .operation = Stage::CommitPartial{
            Float(partial, BufferId::QFN_QSA_PARTIAL_KEYS),
            Float(raw, BufferId::QFN_QSA_RAW_KEY),
            Int(accepted, BufferId::QFN_QSA_SELECTED_COUNT),
            Float(partial, BufferId::QFN_QSA_PARTIAL_KEYS)}}), DeviceId::cpu());
    EXPECT_TRUE(node.stage->allowsZeroOutput());
    for (int liveAccepted : {8, 5, 6, 7, 4, 0})
    {
        SCOPED_TRACE(liveAccepted);
        *count = liveAccepted;
        std::fill_n(partial.mutable_data(), partial.numel(), 0.f);
        ASSERT_TRUE(node.stage->execute(&context));
        EXPECT_NO_THROW(verifyStageExit(node, -1));
        const int tail = liveAccepted % 4;
        for (int row = 0; row < 4; ++row)
            for (int column = 0; column < width; ++column)
                EXPECT_EQ(partial.data()[row * width + column],
                          row < tail ? float(liveAccepted - tail + row + 1) : 0.f);
    }
    *count = 5; // The fifth raw key is the one live row in this accepted tail.
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()})
    {
        std::fill_n(partial.mutable_data(), partial.numel(), 0.f);
        raw.mutable_data()[4 * width] = invalid;
        ASSERT_TRUE(node.stage->execute(&context));
        EXPECT_THROW(verifyStageExit(node, -1), verification::VerificationFailure);
    }
}
#endif

/** @test Indexer and accepted-tail stages bind exact count subviews and immutable weights. */
TEST(QwenSparseAttentionStage, IndexerAndAcceptanceUseExactArenaContracts)
{
    llaminar::v2::ThreadCountGuard threads(1);
    CPUDeviceContext context(DeviceId::cpu(),1);
    constexpr int rows=3,tokens=8,positions=16;
    FP32Tensor query({rows,512}),raw({rows,128}),partial({4,128}),pools({2,128}),
        prepared({rows,512}),scores({rows,2}),gain({128}),cosine({positions,32}),sine({positions,32});
    INT32Tensor replay({5}),selected({rows,2051}),counts({rows});
    auto *state=static_cast<int *>(replay.raw_mutable_data());
    state[0]=-777;state[1]=2;state[2]=rows;state[3]=0;state[4]=rows;
    std::fill_n(query.mutable_data(),query.numel(),1.0f);
    std::fill_n(partial.mutable_data(),partial.numel(),0.25f);
    for(int row=0;row<rows;++row) std::fill_n(raw.mutable_data()+row*128,128,float(row+1));
    std::fill_n(pools.mutable_data(),pools.numel(),-111.0f);
    std::fill_n(scores.mutable_data(),scores.numel(),std::numeric_limits<float>::quiet_NaN());
    std::fill_n(gain.mutable_data(),gain.numel(),1.0f);
    std::fill_n(cosine.mutable_data(),cosine.numel(),1.0f);
    std::fill_n(sine.mutable_data(),sine.numel(),0.0f);
    const Int control(replay,BufferId::QFN_QSA_REPLAY,4);
    Stage indexer({.rowCapacity=rows,.tokenCapacity=tokens,.replay=control,
        .operation=Stage::Indexer{Float(query,BufferId::QFN_QSA_QUERY),Float(raw,BufferId::QFN_QSA_RAW_KEY),
            Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS),Float(pools,BufferId::QFN_QSA_POOLED_KEYS),
            Weight(gain),Weight(gain),Weight(cosine),Weight(sine),Float(prepared,BufferId::QFN_QSA_PREPARED_QUERY),
            Float(scores,BufferId::QFN_QSA_SCORES),Int(selected,BufferId::QFN_QSA_SELECTED),Int(counts,BufferId::QFN_QSA_SELECTED_COUNT),2,positions}});
    EXPECT_EQ(indexer.type(),ComputeStageType::QWEN4EXP_QSA_INDEXER);
    EXPECT_EQ(indexer.bufferContract().inouts.size(),1u);
    EXPECT_EQ(indexer.bufferContract().weight_tensors.size(),4u);
    EXPECT_TRUE(indexer.execute(&context));
    // A short context selects every visible block without authoring ranking
    // scores. Poison proves diagnostics do not depend on incidental arena zeroes.
    for (std::size_t i=0;i<scores.numel();++i) EXPECT_TRUE(std::isnan(scores.data()[i]));
    const auto dump=indexer.getDumpInfoSnapshot();
    for (const auto &output:dump.outputs) EXPECT_NE(std::string_view(output.name),"scores");
    for (const auto name:{"pooled_keys","prepared_query","selected","selected_count"})
        EXPECT_TRUE(std::any_of(dump.outputs.begin(),dump.outputs.end(),[&](const auto &output) {
            return std::string_view(output.name)==name;
        })) << name;
    const auto contract=indexer.bufferContract();
    EXPECT_TRUE(std::any_of(contract.outputs.begin(),contract.outputs.end(),[](const auto &output) {
        return output.id==BufferId::QFN_QSA_SCORES;
    }));
    EXPECT_FALSE(indexer.allowsZeroOutput()); // Real numerical outputs retain validation.
    const auto *count=static_cast<const int *>(counts.raw_data());
    EXPECT_EQ(count[0],3);EXPECT_EQ(count[1],4);EXPECT_EQ(count[2],5);
    for(int row=0;row<rows;++row) for(int token=0;token<count[row];++token)
        EXPECT_EQ(static_cast<const int *>(selected.raw_data())[row*2051+token],token);
    Stage commit({.rowCapacity=rows,.tokenCapacity=tokens,.replay=control,
        .operation=Stage::CommitPartial{Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS),Float(raw,BufferId::QFN_QSA_RAW_KEY),
            Int(replay,BufferId::QFN_QSA_REPLAY,16),Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS)}});
    EXPECT_EQ(commit.type(),ComputeStageType::QWEN4EXP_QSA_COMMIT_PARTIAL);
    EXPECT_EQ(commit.bufferContract().inouts.size(),1u);
    EXPECT_TRUE(commit.execute(&context));
    for(int coordinate=0;coordinate<128;++coordinate) EXPECT_EQ(partial.data()[coordinate],3.0f);
    EXPECT_EQ(state[1],2);EXPECT_EQ(state[2],rows); // Only controller may advance replay.
    EXPECT_THROW(commit.execute(nullptr),std::invalid_argument);
    auto bad=std::get<Stage::Indexer>(indexer.getParams().operation);
    bad.preparedQuery=Float(query,BufferId::QFN_QSA_QUERY);
    EXPECT_THROW((Stage({.rowCapacity=rows,.tokenCapacity=tokens,.replay=control,.operation=bad})),std::invalid_argument);
    bad=std::get<Stage::Indexer>(indexer.getParams().operation);bad.selected.byteOffset=4;
    EXPECT_THROW((Stage({.rowCapacity=rows,.tokenCapacity=tokens,.replay=control,.operation=bad})),std::invalid_argument);
}

/** @test Every public CPU cache codec exposes exact native rings in both physical layouts. */
TEST(QwenSparseAttentionStage, EveryCPUCodecAndLayoutUsesNativeRingAuthority)
{
    llaminar::v2::ThreadCountGuard threads(1);
    CPUDeviceContext context(DeviceId::cpu(),1);
    MPIContext mpi(0,1,MPI_COMM_WORLD);
    TurboQuantContext tq(256,31);
    constexpr int tokens=7;
    for (auto precision : {ActivationPrecision::FP32,ActivationPrecision::FP16,ActivationPrecision::BF16,
                          ActivationPrecision::Q8_1,ActivationPrecision::Q16_1,ActivationPrecision::TQ4,ActivationPrecision::TQ8})
    for (auto layout : {KVCacheLayoutMode::POSITION_MAJOR,KVCacheLayoutMode::HEAD_MAJOR}) {
        SCOPED_TRACE(static_cast<int>(precision));
        SCOPED_TRACE(static_cast<int>(layout));
        auto cache=createCPURingKVCache(precision,mpi,1,1,tokens,2,256,DeviceId::cpu(),layout,&tq);
        ASSERT_TRUE(cache);
        auto view=cache->acquireSparseAttentionNativeView(0,0,nullptr,{},nullptr);
        const int q16=view.keys.q16_block_size;
        qwen4exp::test::NativeCodecFixture key(view.keys.codec,tokens,q16),value(view.values.codec,tokens,q16);
        const bool headMajor=layout==KVCacheLayoutMode::HEAD_MAJOR;
        auto keyBytes=pack(key,tokens,headMajor,view.keys.codec==TensorType::AQ8);
        auto valueBytes=pack(value,tokens,headMajor,false);
        ASSERT_TRUE(cache->importLogicalBlock({0,0,0,tokens,nullptr},keyBytes.data(),valueBytes.data()));
        cache->evict_oldest(1);
        view=cache->acquireSparseAttentionNativeView(0,0,nullptr,{},nullptr);
        EXPECT_EQ(view.logical_view.logical_row_origin,1);
        EXPECT_EQ(view.logical_view.physical_row_capacity,tokens);
        EXPECT_EQ(view.keys.layout,headMajor ? qwen4exp::QwenSparseNativeLayout::HeadMajor : qwen4exp::QwenSparseNativeLayout::TokenMajor);
        FP32Tensor query({1,6144}),output({1,6144});
        INT32Tensor replay({3}),selected({2051}),count({1}),observations({24*sizeof(qwen4exp::QwenSparseAttentionObservation)/4});
        std::fill_n(query.mutable_data(),query.numel(),0.0f);
        auto *state=static_cast<int *>(replay.raw_mutable_data());state[0]=tokens-2;state[1]=1;state[2]=0;
        *static_cast<int *>(count.raw_mutable_data())=tokens-1;
        auto *indices=static_cast<int *>(selected.raw_mutable_data());for(int token=0;token<tokens-1;++token)indices[token]=token;
        Stage stage({.rowCapacity=1,.tokenCapacity=tokens,.replay=Int(replay,BufferId::QFN_QSA_REPLAY),
            .operation=Stage::SelectedAttention{{*cache,0,0,{}},Float(query,BufferId::QFN_QSA_PREPARED_QUERY),
                Int(selected,BufferId::QFN_QSA_SELECTED),Int(count,BufferId::QFN_QSA_SELECTED_COUNT),Float(output,BufferId::ATTN_OUTPUT),
                {},{},{},Int(observations,BufferId::QFN_QSA_OBSERVATIONS)}});
        EXPECT_TRUE(stage.execute(&context));
        for(int head=0;head<24;++head) for(int coordinate=0;coordinate<256;++coordinate) {
            const int kvHead=head/12;
            double expected=0;
            for(int token=1;token<tokens;++token) {
                if(view.values.codec==TensorType::TQ4 || view.values.codec==TensorType::TQ8) {
                    for(int d=0;d<256;++d) {
                        const unsigned code=unsigned(token*19+kvHead*23+d*11)%(view.values.codec==TensorType::TQ4 ? 16u : 256u);
                        expected+=double(view.values.rotations[kvHead][d*256+coordinate])*view.values.centroids[code]*0.5/(tokens-1);
                    }
                } else expected+=double(value.decoded[(token*2+kvHead)*256+coordinate])/(tokens-1);
            }
            EXPECT_NEAR(output.data()[head*256+coordinate],expected,2e-6);
        }
        const auto *traffic=reinterpret_cast<const qwen4exp::QwenSparseAttentionObservation *>(observations.raw_data());
        for(int head=0;head<24;++head) {
            EXPECT_EQ(traffic[head].key_elements,std::uint64_t(tokens-1)*256);
            EXPECT_EQ(traffic[head].value_elements,std::uint64_t(tokens-1)*256);
        }
        EXPECT_THROW(cache->acquireSparseAttentionNativeView(1,0,nullptr,{},nullptr),std::invalid_argument);
    }
}

/** @brief Diagnostic-only scratch must receive its native storage in the ordinary cold arena allocation pass. */
TEST(QwenSparseAttentionStage, WriteOnlyScratchRequiresColdArenaMaterialization){
    constexpr int rows=3,context=8,positions=16;
    FP32Tensor query({rows,512}),raw({rows,128}),partial({4,128}),pools({2,128}),
        prepared({rows,512}),scores({rows,2}),gain({128}),cosine({positions,32}),sine({positions,32});
    INT32Tensor replay({3}),selected({rows,2051}),counts({rows});
    Stage indexer({.rowCapacity=rows,.tokenCapacity=context,.replay=Int(replay,BufferId::QFN_QSA_REPLAY),
        .operation=Stage::Indexer{Float(query,BufferId::QFN_QSA_QUERY),Float(raw,BufferId::QFN_QSA_RAW_KEY),
            Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS),Float(pools,BufferId::QFN_QSA_POOLED_KEYS),
            Weight(gain),Weight(gain),Weight(cosine),Weight(sine),Float(prepared,BufferId::QFN_QSA_PREPARED_QUERY),
            Float(scores,BufferId::QFN_QSA_SCORES),Int(selected,BufferId::QFN_QSA_SELECTED),Int(counts,BufferId::QFN_QSA_SELECTED_COUNT),2,positions}});
    const auto contract=indexer.bufferContract();ASSERT_EQ(contract.outputs.size(),4u);
    for(const auto& output:contract.outputs)EXPECT_TRUE(output.prepare_write_storage)<<bufferIdName(output.id);
    ASSERT_EQ(contract.inouts.size(),1u);EXPECT_EQ(contract.inouts[0].id,BufferId::QFN_QSA_POOLED_KEYS);
    EXPECT_FALSE(contract.inouts[0].prepare_write_storage);
    const auto dump=indexer.getDumpInfoSnapshot();
    for(const auto& output:dump.outputs)if(std::string_view(output.name)=="prepared_query" || std::string_view(output.name)=="selected_count")
        EXPECT_TRUE(std::holds_alternative<SnapshotTokenRows>(output.semantic_extent));
    EXPECT_EQ(prepared.gpu_data_ptr(),nullptr);EXPECT_EQ(counts.gpu_data_ptr(),nullptr);
}

#if LLAMINAR_ASSERTIONS_ACTIVE
/**
 * @test Actual short-prefix Indexer outputs use independent authored live extents.
 *
 * Prepared queries remain strict when nonempty. Pooled state is legitimately
 * empty before four source rows; unowned capacity and containing-subview guards
 * are never numerical evidence. No stage-wide zero-output exception is enabled.
 */
TEST(QwenSparseAttentionStage, IndexerObservedRowsRetainStrictNumericalValidation)
{
    ScopedSparseTailValidation validation;
    llaminar::v2::ThreadCountGuard threads(1);
    CPUDeviceContext context(DeviceId::cpu(), 1);
    constexpr int rows = 8, tokens = 16, blocks = 4, positions = 32;
    FP32Tensor query({rows,512}), raw({rows,128}), prior({4,128}),
        pools({blocks+1,128}), prepared({rows+1,512}), scores({rows,blocks}),
        gain({128}), cosine({positions,32}), sine({positions,32});
    INT32Tensor replay({4}), selected({rows,2051}), counts({rows});
    std::fill_n(query.mutable_data(), query.numel(), 1.f);
    std::fill_n(raw.mutable_data(), raw.numel(), 1.f);
    std::fill_n(prior.mutable_data(), prior.numel(), 0.25f);
    std::fill_n(gain.mutable_data(), gain.numel(), 1.f);
    std::fill_n(cosine.mutable_data(), cosine.numel(), 1.f);
    std::fill_n(sine.mutable_data(), sine.numel(), 0.f);
    auto *state = static_cast<int *>(replay.raw_mutable_data());
    state[0] = -777; // Neither the stage nor the validator owns this containing word.
    ComputeNode node("observed_sparse_indexer", std::make_unique<Stage>(Stage::Params{
        .rowCapacity=rows, .tokenCapacity=tokens,
        .replay=Int(replay,BufferId::QFN_QSA_REPLAY,4),
        .operation=Stage::Indexer{
            Float(query,BufferId::QFN_QSA_QUERY), Float(raw,BufferId::QFN_QSA_RAW_KEY),
            Float(prior,BufferId::QFN_QSA_PARTIAL_KEYS), Float(pools,BufferId::QFN_QSA_POOLED_KEYS,128*4),
            Weight(gain),Weight(gain),Weight(cosine),Weight(sine),
            Float(prepared,BufferId::QFN_QSA_PREPARED_QUERY,512*4), Float(scores,BufferId::QFN_QSA_SCORES),
            Int(selected,BufferId::QFN_QSA_SELECTED), Int(counts,BufferId::QFN_QSA_SELECTED_COUNT),blocks,positions}}),
        DeviceId::cpu());
    ASSERT_FALSE(node.stage->allowsZeroOutput());
    for (const auto [prefix, live] : std::array<std::array<int,2>,8>{{
        {0,1},{0,8},{0,2},{0,3},{3,1},{5,2},{7,1},{0,0}}})
    {
        SCOPED_TRACE(prefix);
        SCOPED_TRACE(live);
        state[1]=prefix; state[2]=live; state[3]=0;
        std::fill_n(prepared.mutable_data(), prepared.numel(), std::numeric_limits<float>::quiet_NaN());
        std::fill_n(pools.mutable_data(), pools.numel(), std::numeric_limits<float>::quiet_NaN());
        std::fill_n(pools.mutable_data()+128, (prefix/4)*128, 1.f);
        if (prefix==0 && live<4)
            std::fill_n(pools.mutable_data()+128, blocks*128, 0.f);
        ASSERT_TRUE(node.stage->execute(&context));
        EXPECT_NO_THROW(verifyStageExit(node, -1));
        EXPECT_TRUE(std::isnan(prepared.data()[0]));
        EXPECT_TRUE(std::isnan(pools.data()[0]));
        if (live)
        {
            const auto saved = prepared.data()[512];
            for (float poison : {std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()})
            {
                prepared.mutable_data()[512]=poison;
                EXPECT_THROW(verifyStageExit(node,-1),verification::VerificationFailure);
            }
            prepared.mutable_data()[512]=saved;
            std::fill_n(prepared.mutable_data()+512, live*512, 0.f);
            EXPECT_THROW(verifyStageExit(node,-1),verification::VerificationFailure);
            ASSERT_TRUE(node.stage->execute(&context));
        }
        if ((prefix+live)/4)
        {
            for (float poison : {0.f, std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::infinity()})
            {
                std::fill_n(pools.mutable_data()+128, ((prefix+live)/4)*128, poison);
                EXPECT_THROW(verifyStageExit(node,-1),verification::VerificationFailure);
            }
            std::fill_n(pools.mutable_data()+128, ((prefix+live)/4)*128, 1.f);
        }
        state[2]=-1;
        EXPECT_THROW(verifyStageExit(node,-1),verification::VerificationFailure);
        state[1]=tokens; state[2]=1;
        EXPECT_THROW(verifyStageExit(node,-1),verification::VerificationFailure);
        state[1]=0; state[2]=rows+1;
        EXPECT_THROW(verifyStageExit(node,-1),verification::VerificationFailure);
    }
}

/** @test Invalid metadata and valid-empty are separate in the shared CPU/device resolver. */
TEST(QwenSparseAttentionStage, ObservedRowResolverRejectsMalformedFrontiers)
{
    const SnapshotContiguousRowGeometry state{SnapshotContiguousRowKind::StatePrefixRows,4,16,4};
    for (int live : {0,1,2,3})
    {
        const auto actual=resolveSnapshotContiguousRows(state,0,live);
        EXPECT_TRUE(actual.valid);
        EXPECT_EQ(actual.rows,0u);
    }
    EXPECT_EQ(resolveSnapshotContiguousRows(state,3,1).rows,1u);
    EXPECT_EQ(resolveSnapshotContiguousRows(state,5,2).rows,1u);
    EXPECT_EQ(resolveSnapshotContiguousRows(state,7,1).rows,2u);
    for (auto [prior,live] : std::array<std::array<int,2>,4>{{{-1,0},{0,-1},{16,1},{INT32_MAX,INT32_MAX}}})
        EXPECT_FALSE(resolveSnapshotContiguousRows(state,prior,live).valid);
    EXPECT_FALSE(resolveSnapshotContiguousRows({SnapshotContiguousRowKind::StatePrefixRows,1,16,4},8,0).valid);
    EXPECT_FALSE(resolveSnapshotContiguousRows({SnapshotContiguousRowKind::StatePrefixRows,4,16,0},0,0).valid);
    EXPECT_FALSE(resolveSnapshotContiguousRows({SnapshotContiguousRowKind::TokenRows,8},0,9).valid);
    EXPECT_FALSE(resolveSnapshotContiguousRows({SnapshotContiguousRowKind::TokenRows,8},1,1).valid);
    EXPECT_FALSE(resolveSnapshotContiguousRows({static_cast<SnapshotContiguousRowKind>(255),8},0,0).valid);
}
#endif

#if LLAMINAR_ASSERTIONS_ACTIVE
namespace
{
/** @brief Exercise the verifier declaration boundary without invoking any kernel/backend. */
class DeclaredObservationStage final : public IComputeStage
{
public:
    /** @brief Retain a call-only descriptor whose actual tensor owners outlive this stage. */
    DeclaredObservationStage(DeviceId device,StageDumpInfo::OutputBuffer output)
        : IComputeStage(device),output_(std::move(output)) {}
    /** @brief No inference operation is needed to test malformed diagnostic declarations. */
    bool execute(IDeviceContext *) override { return true; }
    /** @brief This fixture exposes the same output contract as a sparse indexer. */
    ComputeStageType type() const override { return ComputeStageType::QWEN4EXP_QSA_INDEXER; }
    /** @brief The declaration fixture performs no device work on any identity. */
    bool supportsBackend(ComputeBackendType) const override { return false; }
protected:
    /** @brief Return the original descriptor rather than constructing a fake count proof. */
    StageDumpInfo buildDumpInfoImpl() const override
    {
        StageDumpInfo result;
        result.outputs.push_back(output_);
        return result;
    }
private:
    StageDumpInfo::OutputBuffer output_;
};
}

/** @test Empty observations still require original typed, bounded host-owned storage. */
TEST(QwenSparseAttentionStage, ObservedEmptyRowsDoNotBypassOwnerOrSubviewValidation)
{
    ScopedSparseTailValidation validation;
    FP32Tensor payload({3,4}),wrongType({4});
    INT32Tensor frontier({4});
    std::fill_n(payload.mutable_data(),payload.numel(),std::numeric_limits<float>::quiet_NaN());
    auto *words=static_cast<int *>(frontier.raw_mutable_data());
    words[0]=-777;words[1]=0;words[2]=0;words[3]=-999;
    StageDumpInfo::OutputBuffer output{};
    output.name="observed_state";
    output.data=payload.data()+4;
    output.rows=2;output.cols=4;output.dtype="FP32";
    output.element_size=4;output.byte_size=2*4*4;
    output.tensor=&payload;output.tensor_byte_offset=4*4;
    output.semantic_extent=SnapshotStatePrefixRows{{&frontier,4},8,4};
    auto observe=[&](StageDumpInfo::OutputBuffer declaration,DeviceId device=DeviceId::cpu()) {
        ComputeNode node("observed_declaration",std::make_unique<DeclaredObservationStage>(device,declaration),device);
        verifyStageExit(node,-1);
    };
    EXPECT_NO_THROW(observe(output));
    auto bad=output;
    std::get<SnapshotStatePrefixRows>(bad.semantic_extent).frontier.tensor=nullptr;
    EXPECT_THROW(observe(bad),verification::VerificationFailure);
    bad=output;std::get<SnapshotStatePrefixRows>(bad.semantic_extent).frontier.tensor=&wrongType;
    EXPECT_THROW(observe(bad),verification::VerificationFailure);
    bad=output;std::get<SnapshotStatePrefixRows>(bad.semantic_extent).frontier.byteOffset=12;
    EXPECT_THROW(observe(bad),verification::VerificationFailure);
    bad=output;std::get<SnapshotStatePrefixRows>(bad.semantic_extent).frontier.byteOffset=1;
    EXPECT_THROW(observe(bad),verification::VerificationFailure);
    bad=output;std::get<SnapshotStatePrefixRows>(bad.semantic_extent).sourceRowsPerStateRow=0;
    EXPECT_THROW(observe(bad),verification::VerificationFailure);
    bad=output;bad.tensor_byte_offset=payload.size_bytes();
    EXPECT_THROW(observe(bad),verification::VerificationFailure);
    bad=output;bad.tensor=nullptr;
    EXPECT_THROW(observe(bad),verification::VerificationFailure);
    bad=output;bad.byte_size=4;
    EXPECT_THROW(observe(bad),verification::VerificationFailure);
    // A GPU identity cannot reinterpret these real host-only original roots;
    // this rejection requires neither backend activation nor a GPU allocation.
    EXPECT_THROW(observe(output,DeviceId::cuda(0)),verification::VerificationFailure);
    EXPECT_THROW(observe(output,DeviceId::rocm(0)),verification::VerificationFailure);
    words[1]=8;words[2]=1;
    EXPECT_THROW(observe(output),verification::VerificationFailure);
}
#endif

/** @test Diagnostic counter width is checked without backend construction or storage allocation. */
TEST(QwenSparseAttentionStage, ObservedValidationCapacityRejectsCounterOverflowWithoutAllocation)
{
    TensorValidationExtent extent{{SnapshotContiguousRowKind::Physical,UINT32_MAX},1,nullptr};
    EXPECT_EQ(checkedTensorValidationElementCapacity(extent,TensorValidationDataType::FP32),UINT32_MAX);
    extent.geometry.capacity=static_cast<std::size_t>(UINT32_MAX)+1;
    EXPECT_THROW(checkedTensorValidationElementCapacity(extent,TensorValidationDataType::FP32),std::invalid_argument);
    extent.geometry.capacity=UINT32_MAX;extent.columns=2;
    EXPECT_THROW(checkedTensorValidationElementCapacity(extent,TensorValidationDataType::FP32),std::invalid_argument);
    extent.geometry.capacity=1;extent.columns=std::numeric_limits<std::size_t>::max();
    EXPECT_THROW(checkedTensorValidationElementCapacity(extent,TensorValidationDataType::FP32),std::invalid_argument);
    extent.columns=1;extent.geometry.capacity=0;
    EXPECT_THROW(checkedTensorValidationElementCapacity(extent,TensorValidationDataType::FP32),std::invalid_argument);
}


/**
 * @test Every explicit selector seals an ordinary/verifier indexer independently
 * of live state, and every compiled backend carries the same typed contract.
 * @details Construction uses host tensor metadata and empty kernel bundles only;
 * GPU stages are never executed, materialized or probed in this Unit regression.
 */
TEST(QwenSparseAttentionStage, SelectorPolicySealsEveryIndexerCapture)
{
    using Selection=qwen4exp::QwenSparsePoolSelection;
    constexpr int rows=3,tokens=8,positions=16;
    constexpr std::array selections{Selection::Installed,Selection::Bounded256_512,
        Selection::Bounded512_512,Selection::Bounded1024_1024};
    FP32Tensor query({rows+1,512}),otherQuery({rows+1,512}),raw({rows,128}),partial({4,128}),pools({3,128}),
        prepared({rows,512}),scores({rows,3}),gain({128}),otherGain({128}),cosine({positions,32}),sine({positions,32});
    INT32Tensor replay({6}),selected({rows,2051}),counts({rows}),
        prior({static_cast<int>(sizeof(qwen4exp::QwenSparseTransactionPrior)/sizeof(std::int32_t))});
    const Stage::Indexer input{Float(query,BufferId::QFN_QSA_QUERY),Float(raw,BufferId::QFN_QSA_RAW_KEY),
        Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS),Float(pools,BufferId::QFN_QSA_POOLED_KEYS),
        Weight(gain),Weight(gain),Weight(cosine),Weight(sine),Float(prepared,BufferId::QFN_QSA_PREPARED_QUERY),
        Float(scores,BufferId::QFN_QSA_SCORES),Int(selected,BufferId::QFN_QSA_SELECTED),
        Int(counts,BufferId::QFN_QSA_SELECTED_COUNT),2,positions};
    EXPECT_EQ(input.poolSelection,Selection::Installed);
    const Stage::Params base{.device_id=DeviceId::cpu(),.rowCapacity=rows,.tokenCapacity=tokens,
        .replay=Int(replay,BufferId::QFN_QSA_REPLAY),.operation=input};
    const auto signature=[](Stage::Params params) {
        Stage stage(std::move(params));
        return stage.graphCaptureVariantSignature();
    };
    const auto baseline=signature(base);
    ASSERT_NE(baseline,0u);
    EXPECT_EQ(signature(base),baseline);
    std::vector<DeviceId> devices{DeviceId::cpu()};
#ifdef HAVE_CUDA
    devices.push_back(DeviceId::cuda(0));devices.push_back(DeviceId::cuda(1));
#endif
#ifdef HAVE_ROCM
    devices.push_back(DeviceId::rocm(0));devices.push_back(DeviceId::rocm(1));
#endif
    std::vector<std::uint64_t> identities;
    for(const auto device:devices)for(const auto policy:selections)for(const bool retained:{false,true}) {
        SCOPED_TRACE(device.to_string());
        SCOPED_TRACE(static_cast<int>(policy));
        SCOPED_TRACE(retained);
        auto params=base;params.device_id=device;
        auto& operation=std::get<Stage::Indexer>(params.operation);
        operation.poolSelection=policy;
        if(retained) {
            operation.transactionPrior=QwenSparseTransactionPriorBinding{Int(prior,BufferId::QFN_QSA_TRANSACTION_PRIOR)};
            params.replay=Int(prior,BufferId::QFN_QSA_TRANSACTION_PRIOR);
        }
        Stage stage(params);
        EXPECT_EQ(std::get<Stage::Indexer>(stage.getParams().operation).poolSelection,policy);
        const auto actual=stage.graphCaptureVariantSignature();
        ASSERT_NE(actual,0u);
        EXPECT_EQ(actual,signature(params));
        EXPECT_EQ(std::find(identities.begin(),identities.end(),actual),identities.end());
        identities.push_back(actual);
    }
    // Mutation of execution-owned bytes never mutates topology or selector policy.
    auto* state=static_cast<std::int32_t*>(replay.raw_mutable_data());
    state[0]=5;state[1]=0;state[2]=77;
    std::fill_n(query.mutable_data(),query.numel(),std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(signature(base),baseline);
    const auto changed=[&](auto alter) {
        auto params=base;alter(params);
        EXPECT_NE(signature(params),baseline);
    };
    changed([](auto& p){p.rowCapacity=2;});
    changed([](auto& p){p.tokenCapacity=7;});
    changed([](auto& p){p.tokenCapacity=12;std::get<Stage::Indexer>(p.operation).blockCapacity=3;});
    changed([](auto& p){p.replay.byteOffset=sizeof(std::int32_t);});
    changed([&](auto& p){std::get<Stage::Indexer>(p.operation).query=Float(otherQuery,BufferId::QFN_QSA_QUERY);});
    changed([](auto& p){std::get<Stage::Indexer>(p.operation).query.byteOffset=sizeof(float);});
    changed([&](auto& p){std::get<Stage::Indexer>(p.operation).queryNorm=Weight(otherGain);});
    changed([](auto& p){std::get<Stage::Indexer>(p.operation).epsilon=2.0e-6f;});
    changed([](auto& p){std::get<Stage::Indexer>(p.operation).ropeTheta=10000000.0f;});
    changed([](auto& p){std::get<Stage::Indexer>(p.operation).positionCapacity=8;});
    changed([](auto& p){std::get<Stage::Indexer>(p.operation).rotaryDim=32;});
    changed([](auto& p){std::get<Stage::Indexer>(p.operation).representation=qwen4exp::QwenSparseScalar::FP16;});
    changed([](auto& p){std::get<Stage::Indexer>(p.operation).representation=qwen4exp::QwenSparseScalar::BF16;});
    changed([](auto& p){
        auto& operation=std::get<Stage::Indexer>(p.operation);
        operation.rotarySource=qwen4exp::QwenSparseRotarySource::AnalyticTheta;
        operation.cosine.reset();operation.sine.reset();
    });
    auto invalid=base;
    std::get<Stage::Indexer>(invalid.operation).poolSelection=static_cast<Selection>(255);
    EXPECT_THROW((Stage(invalid)),std::invalid_argument);
    EXPECT_THROW(qwen4exp::validateQwenSparsePoolSelection(static_cast<Selection>(255)),std::invalid_argument);

}

/**
 * @test The real CPU Indexer stage retains exact policy outputs over full/partial/empty reuse.
 * @details CPU has no native graph function inspector. GPU companion tests witness
 * the common stage's exact selector forwarding; this counterpart proves actual
 * CPU execution, independent produced-score membership and retained byte behavior.
 * All four policies include exact8195/8196 and mixed live-row pool boundaries.
 */
TEST(QwenSparseAttentionStage, EveryCPUSelectorPolicyExecutesRetainedRows){
    using namespace qwen4exp;llaminar::v2::ThreadCountGuard threads(1);CPUDeviceContext context(DeviceId::cpu(),1);
    constexpr int rows=3,tokens=8320,blocks=tokens/4,positions=tokens+17;
    FP32Tensor query({rows,512}),raw({rows,128}),prior({4,128}),pools({blocks,128}),prepared({rows,512}),scores({rows,blocks}),
        gain({128}),cosine({positions,32}),sine({positions,32});INT32Tensor replay({3}),selected({rows,2051}),counts({rows});
    for(auto* value:{&query,&raw,&prior,&pools})for(std::size_t i=0;i<value->numel();++i)value->mutable_data()[i]=float((i*13+7)%67)/41.f-0.8f;
    std::fill_n(gain.mutable_data(),gain.numel(),1.f);std::fill_n(cosine.mutable_data(),cosine.numel(),1.f);std::fill_n(sine.mutable_data(),sine.numel(),0.f);
    const std::vector<float> poolSeed(pools.data(),pools.data()+pools.numel());
    struct Result{std::vector<float> pools,prepared,scores;std::vector<int> selected,counts;};std::vector<Result> reference;
    const std::array policies{QwenSparsePoolSelection::Installed,QwenSparsePoolSelection::Bounded256_512,QwenSparsePoolSelection::Bounded512_512,QwenSparsePoolSelection::Bounded1024_1024};
    // Prefix8194 with three live rows spans visible8195,8196,8197.
    const std::array cases{std::pair{4096,rows},std::pair{4096,2},std::pair{2047,rows},std::pair{8194,1},std::pair{8195,1},
        std::pair{8194,rows},std::pair{8195,2},std::pair{0,0},std::pair{0,1},std::pair{8195,rows}};
    std::vector<std::uint64_t> signatures;
    for(auto policy:policies){SCOPED_TRACE(static_cast<int>(policy));
        Stage::Indexer input{Float(query,BufferId::QFN_QSA_QUERY),Float(raw,BufferId::QFN_QSA_RAW_KEY),Float(prior,BufferId::QFN_QSA_PARTIAL_KEYS),
            Float(pools,BufferId::QFN_QSA_POOLED_KEYS),Weight(gain),Weight(gain),Weight(cosine),Weight(sine),
            Float(prepared,BufferId::QFN_QSA_PREPARED_QUERY),Float(scores,BufferId::QFN_QSA_SCORES),Int(selected,BufferId::QFN_QSA_SELECTED),
            Int(counts,BufferId::QFN_QSA_SELECTED_COUNT),blocks,positions};input.poolSelection=policy;
        Stage stage({.device_id=DeviceId::cpu(),.rowCapacity=rows,.tokenCapacity=tokens,.replay=Int(replay,BufferId::QFN_QSA_REPLAY),.operation=input});
        const auto identity=stage.graphCaptureVariantSignature();ASSERT_NE(identity,0u);
        EXPECT_EQ(std::find(signatures.begin(),signatures.end(),identity),signatures.end());signatures.push_back(identity);
        for(std::size_t iteration=0;iteration<cases.size();++iteration){const auto[prefix,live]=cases[iteration];SCOPED_TRACE(iteration);
            const std::array<int,3> words{prefix,live,0};std::copy(words.begin(),words.end(),static_cast<int*>(replay.raw_mutable_data()));
            std::copy(poolSeed.begin(),poolSeed.end(),pools.mutable_data());std::fill_n(prepared.mutable_data(),prepared.numel(),-271.75f);
            std::fill_n(scores.mutable_data(),scores.numel(),-8191.25f);std::fill_n(static_cast<int*>(selected.raw_mutable_data()),selected.numel(),-77);
            std::fill_n(static_cast<int*>(counts.raw_mutable_data()),counts.numel(),-99);ASSERT_TRUE(stage.execute(&context));
            const auto* actualIds=static_cast<const int*>(selected.raw_data());const auto* actualCounts=static_cast<const int*>(counts.raw_data());
            EXPECT_TRUE(std::equal(words.begin(),words.end(),static_cast<const int*>(replay.raw_data())));
            for(int row=0;row<rows;++row){const int visible=prefix+row+1,complete=visible/4;std::vector<std::int32_t> expected;
                if(row<live){if(complete>512)for(int b=0;b<complete;++b)ASSERT_TRUE(std::isfinite(scores.data()[row*blocks+b])&&scores.data()[row*blocks+b]>=0.f);
                    expected=selection_test::oracle(complete>512?scores.data()+row*blocks:nullptr,visible);}
                ASSERT_EQ(actualCounts[row],int(expected.size()));for(int i=0;i<2051;++i)EXPECT_EQ(actualIds[row*2051+i],i<int(expected.size())?expected[std::size_t(i)]:-77);
                for(int b=0;b<blocks;++b)if(row>=live||complete<=512||b>=complete)EXPECT_EQ(scores.data()[row*blocks+b],-8191.25f);
                if(row>=live)for(int d=0;d<512;++d)EXPECT_EQ(prepared.data()[row*512+d],-271.75f);}
            Result result{{pools.data(),pools.data()+pools.numel()},{prepared.data(),prepared.data()+prepared.numel()},{scores.data(),scores.data()+scores.numel()},
                {actualIds,actualIds+selected.numel()},{actualCounts,actualCounts+counts.numel()}};
            if(policy==QwenSparsePoolSelection::Installed)reference.push_back(result);else{const auto& expected=reference[iteration];
                EXPECT_EQ(result.selected,expected.selected);EXPECT_EQ(result.counts,expected.counts);
                EXPECT_EQ(std::memcmp(result.pools.data(),expected.pools.data(),pools.size_bytes()),0);
                EXPECT_EQ(std::memcmp(result.prepared.data(),expected.prepared.data(),prepared.size_bytes()),0);
                EXPECT_EQ(std::memcmp(result.scores.data(),expected.scores.data(),scores.size_bytes()),0);}
            EXPECT_EQ(stage.graphCaptureVariantSignature(),identity);ASSERT_FALSE(::testing::Test::HasFailure());}
    }
    EXPECT_EQ(signatures.size(),policies.size());
}

namespace
{
using SparseArithmetic=qwen4exp::QwenSparseAttentionArithmetic;
/** @brief Immutable cache metadata only; construction and identity must never probe live/native state. */
class SparseArithmeticMetadataCache final : public CPURingKVCacheFP16
{
public:
    ActivationPrecision keys=ActivationPrecision::FP16,values=ActivationPrecision::FP16;
    /** @brief Reuse the real tiny host cache's geometry; no accelerator or native view is acquired. */
    explicit SparseArithmeticMetadataCache(const IMPIContext &mpi) : CPURingKVCacheFP16(mpi,2,2,8,2,256) {}
    /** @copydoc IKVCache::k_precision */
    ActivationPrecision k_precision() const override { return keys; }
    /** @copydoc IKVCache::v_precision */
    ActivationPrecision v_precision() const override { return values; }
    /** @copydoc IKVCache::get_cached_tokens */
    int get_cached_tokens(int,int=0) const override { throw std::logic_error("Identity read live cache count"); }
    /** @copydoc IKVCache::ring_head */
    int ring_head(int,int=0) const override { throw std::logic_error("Identity read live cache cursor"); }
    /** @copydoc IKVCache::get_kv */
    bool get_kv(int,int,ITensor **,ITensor **,int * =nullptr) override { throw std::logic_error("Identity read KV payload"); }
    /** @copydoc IKVCache::get_kv */
    bool get_kv(int,int,const ITensor **,const ITensor **,int * =nullptr) const override { throw std::logic_error("Identity read KV payload"); }
    /** @copydoc IKVCache::append */
    bool append(int,int,const ITensor *,const ITensor *,int) override { throw std::logic_error("Identity executed cache append"); }
    /** @copydoc IKVCache::acquireSparseAttentionNativeView */
    qwen4exp::QwenSparseKV acquireSparseAttentionNativeView(int,int,
        const attention::AttentionDeviceParams *,attention::AttentionKeyCachePolicy,void *) override
    { throw std::logic_error("Identity probed native cache storage"); }
};
/** @brief Host-owned declaration fixtures; even ROCm cases construct no native storage or execution context. */
struct SparseArithmeticCaptureFixture
{
    FP32Tensor query{{4,6144}},otherQuery{{4,6144}},output{{4,6144}},gate{{4,6144}},
        cosine{{16,128}},sine{{16,128}},otherCosine{{16,128}};
    INT32Tensor replay{{8+4*sizeof(attention::AttentionDeviceParams)/sizeof(std::int32_t)}},selected{{4,2051}},counts{{4}},
        partials{{4*24*8*sizeof(qwen4exp::QwenSparseAttentionPartial)/sizeof(std::int32_t)}},
        observations{{4*24*sizeof(qwen4exp::QwenSparseAttentionObservation)/sizeof(std::int32_t)}};
    /** @return A fresh exact declaration borrowing the requested immutable cache object and participant. */
    Stage::Params params(IKVCache &cache,DeviceId device=DeviceId::cpu())
    {
        Stage::SelectedAttention operation{{cache,0,0,{}},Float(query,BufferId::Q_PROJ),
            Int(selected,BufferId::QFN_QSA_SELECTED),Int(counts,BufferId::QFN_QSA_SELECTED_COUNT),
            Float(output,BufferId::ATTN_OUTPUT),{},{},{},{}};
        if(device.is_gpu()) {
            operation.rowParameters=Int(replay,BufferId::QFN_QSA_REPLAY,8*sizeof(std::int32_t));
            operation.gpuPartials=Int(partials,BufferId::QFN_QSA_PARTIALS);
        }
        return {.device_id=device,.rowCapacity=3,.tokenCapacity=8,
            .replay=Int(replay,BufferId::QFN_QSA_REPLAY),.operation=operation};
    }
};
/** @return The real stage's sealed signature; no execution or native cache acquisition is permitted. */
std::uint64_t sparseArithmeticSignature(Stage::Params params)
{
    Stage stage(std::move(params));
    const auto dump=stage.getDumpInfoSnapshot();
    EXPECT_EQ(std::count_if(dump.scalars.begin(),dump.scalars.end(),[](const auto &scalar) {
        return std::string_view(scalar.name)=="attention_arithmetic_policy";
    }),1);
    const auto policy=std::find_if(dump.scalars.begin(),dump.scalars.end(),[](const auto &scalar) {
        return std::string_view(scalar.name)=="attention_arithmetic_policy";
    });
    if(policy!=dump.scalars.end()) {
        EXPECT_EQ(policy->value,static_cast<int>(std::get<Stage::SelectedAttention>(stage.getParams().operation).arithmetic));
        EXPECT_EQ(std::string_view(policy->dtype),"int");
    }
    return stage.graphCaptureVariantSignature();
}
}

/** @test Typed names, parser, defaults and backend admission reject silent arithmetic substitution. */
TEST(QwenSparseAttentionStage, AttentionArithmeticNamesDefaultsAndBackendAdmission)
{
    using namespace qwen4exp;
    EXPECT_EQ(kDefaultQwenSparseAttentionArithmetic,SparseArithmetic::FixedFP32);
    EXPECT_EQ(QwenSparseAttention{}.arithmetic,SparseArithmetic::FixedFP32);
    EXPECT_EQ(parseQwenSparseAttentionArithmetic("fp32"),SparseArithmetic::FixedFP32);
    EXPECT_EQ(parseQwenSparseAttentionArithmetic("fp16_tile"),SparseArithmetic::TileFP16WithFP32Totals);
    EXPECT_EQ(std::string_view(qwenSparseAttentionArithmeticName(SparseArithmetic::FixedFP32)),"fp32");
    EXPECT_EQ(std::string_view(qwenSparseAttentionArithmeticName(SparseArithmetic::TileFP16WithFP32Totals)),"fp16_tile");
    for(const auto text:{"","FP32","fp16","fp16_tile ","automatic"})
        EXPECT_THROW(parseQwenSparseAttentionArithmetic(text),std::invalid_argument) << text;
    for(const auto backend:{DeviceType::CPU,DeviceType::CUDA,DeviceType::ROCm}) {
        EXPECT_NO_THROW(validateQwenSparseAttentionArithmetic(SparseArithmetic::FixedFP32,backend));
        EXPECT_THROW(validateQwenSparseAttentionArithmetic(static_cast<SparseArithmetic>(255),backend),std::invalid_argument);
    }
    EXPECT_NO_THROW(validateQwenSparseAttentionArithmetic(SparseArithmetic::TileFP16WithFP32Totals,DeviceType::ROCm));
    for(const auto backend:{DeviceType::CPU,DeviceType::CUDA,DeviceType::Vulkan,DeviceType::Metal,static_cast<DeviceType>(255)})
        EXPECT_THROW(validateQwenSparseAttentionArithmetic(SparseArithmetic::TileFP16WithFP32Totals,backend),std::invalid_argument);
    for(const auto backend:{DeviceType::Vulkan,DeviceType::Metal,static_cast<DeviceType>(255)})
        EXPECT_THROW(validateQwenSparseAttentionArithmetic(SparseArithmetic::FixedFP32,backend),std::invalid_argument);
}

/** @test CPU rejects the explicit ROCm arithmetic before dereferencing even an empty native argument record. */
TEST(QwenSparseAttentionStage, AttentionArithmeticCPUEntryRejectsBeforeBindings)
{
    qwen4exp::QwenSparseAttention empty;
    empty.arithmetic=SparseArithmetic::TileFP16WithFP32Totals;
    try {
        qwen4exp::cpuQwenSparseAttention(empty);
        FAIL()<<"CPU entry accepted the explicit ROCm attention experiment";
    } catch(const std::invalid_argument &error) {
        EXPECT_NE(std::string_view(error.what()).find("ROCm"),std::string_view::npos)
            <<"Backend admission must precede the incomplete native binding check";
    }
}

/** @test Selected attention seals immutable CPU declarations while mutable payloads never enter capture identity. */
TEST(QwenSparseAttentionStage, AttentionArithmeticSelectedCaptureSealsStaticIdentity)
{
    MPIContext mpi(0,1,MPI_COMM_WORLD);
    SparseArithmeticMetadataCache cache(mpi),otherCache(mpi);
    SparseArithmeticCaptureFixture f;
    const auto base=f.params(cache);
    EXPECT_EQ(std::get<Stage::SelectedAttention>(base.operation).arithmetic,SparseArithmetic::FixedFP32);
    const auto baseline=sparseArithmeticSignature(base);
    ASSERT_NE(baseline,0u);
    EXPECT_EQ(sparseArithmeticSignature(base),baseline);
    const auto changed=[&](auto alter) {
        auto params=base;alter(params);
        EXPECT_NE(sparseArithmeticSignature(params),baseline);
    };
    EXPECT_NE(sparseArithmeticSignature(f.params(otherCache)),baseline);
    cache.keys=ActivationPrecision::BF16;EXPECT_NE(sparseArithmeticSignature(base),baseline);cache.keys=ActivationPrecision::FP16;
    cache.values=ActivationPrecision::BF16;EXPECT_NE(sparseArithmeticSignature(base),baseline);cache.values=ActivationPrecision::FP16;
    changed([](auto &p){p.rowCapacity=2;});
    changed([](auto &p){p.tokenCapacity=7;});
    changed([](auto &p){p.replay.byteOffset=4;});
    changed([](auto &p){std::get<Stage::SelectedAttention>(p.operation).native.layer=1;});
    changed([](auto &p){std::get<Stage::SelectedAttention>(p.operation).native.sequence=1;});
    changed([](auto &p){std::get<Stage::SelectedAttention>(p.operation).native.keyPolicy.rope_theta=20000.f;});
    changed([](auto &p){std::get<Stage::SelectedAttention>(p.operation).native.keyPolicy.partial_rotary_factor=.25f;});
    changed([](auto &p){std::get<Stage::SelectedAttention>(p.operation).queryHeads=12;});
    changed([](auto &p){auto &a=std::get<Stage::SelectedAttention>(p.operation);a.queryHeads=12;a.kvHeads=1;});
    changed([](auto &p){auto &a=std::get<Stage::SelectedAttention>(p.operation);a.queryHeads=12;a.kvHeads=1;a.queryHeadStart=12;a.kvHeadStart=1;});
    changed([&](auto &p){std::get<Stage::SelectedAttention>(p.operation).query=Float(f.otherQuery,BufferId::Q_PROJ);});
    changed([](auto &p){std::get<Stage::SelectedAttention>(p.operation).query.byteOffset=4;});
    changed([](auto &p){std::get<Stage::SelectedAttention>(p.operation).selected.byteOffset=4;});
    changed([](auto &p){std::get<Stage::SelectedAttention>(p.operation).selectedCount.byteOffset=4;});
    changed([](auto &p){std::get<Stage::SelectedAttention>(p.operation).output.byteOffset=4;});
    changed([&](auto &p){std::get<Stage::SelectedAttention>(p.operation).gate=Float(f.gate,BufferId::GATE_PROJ);});
    changed([&](auto &p){std::get<Stage::SelectedAttention>(p.operation).observations=Int(f.observations,BufferId::QFN_QSA_OBSERVATIONS);});
    auto *state=static_cast<std::int32_t *>(f.replay.raw_mutable_data());state[0]=5;state[1]=0;state[2]=77;
    std::fill_n(f.query.mutable_data(),f.query.numel(),std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(sparseArithmeticSignature(base),baseline);
    for(const auto arithmetic:{SparseArithmetic::TileFP16WithFP32Totals,static_cast<SparseArithmetic>(255)}) {
        auto bad=base;std::get<Stage::SelectedAttention>(bad.operation).arithmetic=arithmetic;
        EXPECT_THROW((Stage(bad)),std::invalid_argument);
    }
    auto cuda=f.params(cache,DeviceId::cuda(0));
    std::get<Stage::SelectedAttention>(cuda.operation).arithmetic=SparseArithmetic::TileFP16WithFP32Totals;
    EXPECT_THROW((Stage(cuda)),std::invalid_argument); // Rejection precedes any backend factory or device access.
}

#ifdef HAVE_ROCM
/** @test The ROCm tile policy is sealed without execution and rejects unsupported codecs/transform geometry cold. */
TEST(QwenSparseAttentionStage, AttentionArithmeticROCmTileCaptureAndColdAdmission)
{
    MPIContext mpi(0,1,MPI_COMM_WORLD);
    SparseArithmeticMetadataCache cache(mpi);
    SparseArithmeticCaptureFixture f;
    const auto base=f.params(cache,DeviceId::rocm(0));
    const auto fp32=sparseArithmeticSignature(base);
    auto tile=base;std::get<Stage::SelectedAttention>(tile.operation).arithmetic=SparseArithmetic::TileFP16WithFP32Totals;
    const auto tileIdentity=sparseArithmeticSignature(tile);
    ASSERT_NE(fp32,0u);ASSERT_NE(tileIdentity,0u);EXPECT_NE(tileIdentity,fp32);
    EXPECT_EQ(sparseArithmeticSignature(tile),tileIdentity);
    auto otherDevice=tile;otherDevice.device_id=DeviceId::rocm(1);
    EXPECT_NE(sparseArithmeticSignature(otherDevice),tileIdentity);
    auto otherParams=tile;std::get<Stage::SelectedAttention>(otherParams.operation).rowParameters->byteOffset+=sizeof(attention::AttentionDeviceParams);
    EXPECT_NE(sparseArithmeticSignature(otherParams),tileIdentity);
    auto otherPartials=tile;std::get<Stage::SelectedAttention>(otherPartials.operation).gpuPartials->byteOffset+=24*8*sizeof(qwen4exp::QwenSparseAttentionPartial);
    EXPECT_NE(sparseArithmeticSignature(otherPartials),tileIdentity);
    for(const auto precision:{ActivationPrecision::FP32,ActivationPrecision::BF16,ActivationPrecision::Q8_1}) {
        cache.keys=precision;EXPECT_THROW((Stage(tile)),std::invalid_argument);cache.keys=ActivationPrecision::FP16;
        cache.values=precision;EXPECT_THROW((Stage(tile)),std::invalid_argument);cache.values=ActivationPrecision::FP16;
    }
    auto transform=tile;auto &a=std::get<Stage::SelectedAttention>(transform.operation);
    a.native.keyPolicy.encoding=attention::AttentionKeyCacheEncoding::PreRotaryDeviceTransform;
    a.native.keyPolicy.partial_rotary_factor=.25f;
    a.keyCosine=Weight(f.cosine);a.keySine=Weight(f.sine);a.keyPositionCapacity=16;
    const auto transformIdentity=sparseArithmeticSignature(transform);
    EXPECT_NE(transformIdentity,tileIdentity);
    auto differentWeight=transform;std::get<Stage::SelectedAttention>(differentWeight.operation).keyCosine=Weight(f.otherCosine);
    EXPECT_NE(sparseArithmeticSignature(differentWeight),transformIdentity);
    auto theta=transform;auto &analytic=std::get<Stage::SelectedAttention>(theta.operation);
    analytic.keyRotarySource=qwen4exp::QwenSparseRotarySource::CanonicalEngineTheta;
    analytic.keyCosine.reset();analytic.keySine.reset();
    EXPECT_NE(sparseArithmeticSignature(theta),transformIdentity);
    a.keyRotaryDim=32;a.native.keyPolicy.partial_rotary_factor=.125f;
    EXPECT_THROW((Stage(transform)),std::invalid_argument);
    auto unknown=tile;std::get<Stage::SelectedAttention>(unknown.operation).arithmetic=static_cast<SparseArithmetic>(255);
    EXPECT_THROW((Stage(unknown)),std::invalid_argument);
}
#endif

/** @test New setup schedule and resolved33 tag are distinct, closed and device-free. */
TEST(QwenSparseAttentionStage, Phase33NamesAndUnsupportedBackends)
{
    using namespace qwen4exp;using A=QwenSparseAttentionArithmetic;using S=QwenSparseAttentionPhaseSchedule;
    EXPECT_EQ(qwenSparseAttentionPartialCount(A::FixedFP32),8);
    EXPECT_EQ(qwenSparseAttentionPartialCount(A::TileFP16WithFP32Totals),8);
    EXPECT_EQ(qwenSparseAttentionPartialCount(A::TileFP16WithFP32Totals33),33);
    EXPECT_EQ(qwenSparseAttentionArithmeticName(A::TileFP16WithFP32Totals33),"fp16_tile_33");
    EXPECT_EQ(parseQwenSparseAttentionPhaseSchedule("prefill8_decode33"),S::Prefill8Decode33);
    EXPECT_EQ(parseQwenSparseAttentionPhaseSchedule("uniform"),S::Uniform);
    EXPECT_THROW(parseQwenSparseAttentionArithmetic("fp16_tile_33"),std::invalid_argument); // Resolved tag is not an unresolved setup spelling.
    for(const auto text:{"","auto","Prefill8Decode33","prefill8_decode33 "})EXPECT_THROW(parseQwenSparseAttentionPhaseSchedule(text),std::invalid_argument);
    EXPECT_THROW(qwenSparseAttentionPartialCount(static_cast<A>(255)),std::invalid_argument);
    EXPECT_THROW(qwenSparseAttentionPhaseScheduleName(static_cast<S>(255)),std::invalid_argument);
    for(auto backend:{DeviceType::CPU,DeviceType::CUDA,DeviceType::Vulkan,DeviceType::Metal,static_cast<DeviceType>(255)})
        EXPECT_THROW(validateQwenSparseAttentionArithmetic(A::TileFP16WithFP32Totals33,backend),std::invalid_argument);
    EXPECT_NO_THROW(validateQwenSparseAttentionArithmetic(A::TileFP16WithFP32Totals33,DeviceType::ROCm));
    QwenSparseAttention empty;empty.arithmetic=A::TileFP16WithFP32Totals33;
    EXPECT_THROW(cpuQwenSparseAttention(empty),std::invalid_argument); // No native binding is touched on CPU.
}
#ifdef HAVE_ROCM
/** @test Real33 stage admission seals exact phase/R/program/extents without executing a native function. */
TEST(QwenSparseAttentionStage, Phase33CaptureExtentAndColdAdmission)
{
    using A=qwen4exp::QwenSparseAttentionArithmetic;using P=ForwardExecutionPhase;
    MPIContext mpi(0,1,MPI_COMM_WORLD);SparseArithmeticMetadataCache cache(mpi);SparseArithmeticCaptureFixture f;
    INT32Tensor large{{4*24*33*sizeof(qwen4exp::QwenSparseAttentionPartial)/4}};
    auto p=f.params(cache,DeviceId::rocm(0));auto& a=std::get<Stage::SelectedAttention>(p.operation);
    a.arithmetic=A::TileFP16WithFP32Totals33;a.executionPhase=P::Decode;a.retainedMtpCandidateRows=3;
    a.gpuPartials=Int(large,BufferId::QFN_QSA_PARTIALS);
    const auto signature=sparseArithmeticSignature(p);ASSERT_NE(signature,0u);
    auto other=p;std::get<Stage::SelectedAttention>(other.operation).retainedMtpCandidateRows=4;
    EXPECT_NE(sparseArithmeticSignature(other),signature);
    auto old=p;std::get<Stage::SelectedAttention>(old.operation).arithmetic=A::TileFP16WithFP32Totals;
    EXPECT_NE(sparseArithmeticSignature(old),signature);
    auto prompt=old;std::get<Stage::SelectedAttention>(prompt.operation).executionPhase=P::Prefill;
    EXPECT_NE(sparseArithmeticSignature(prompt),sparseArithmeticSignature(old));
    Stage admitted(p);const auto dump=admitted.getDumpInfoSnapshot();
    for(const auto [name,value]:{std::pair{"attention_arithmetic_parts",33},std::pair{"attention_execution_phase",int(P::Decode)}}) {
        const auto field=std::find_if(dump.scalars.begin(),dump.scalars.end(),[&](const auto& s){return std::string_view(s.name)==name;});
        ASSERT_NE(field,dump.scalars.end());EXPECT_EQ(field->value,value);
    }
    {
        auto bad=p;std::get<Stage::SelectedAttention>(bad.operation).gpuPartials=Int(f.partials,BufferId::QFN_QSA_PARTIALS);
        EXPECT_THROW((Stage(bad)),std::invalid_argument); // Existing8 bank is insufficient for3*24*33 records.
    }
    {
        auto bad=p;std::get<Stage::SelectedAttention>(bad.operation).executionPhase.reset();
        EXPECT_THROW((Stage(bad)),std::invalid_argument);
    }
    {
        auto bad=p;std::get<Stage::SelectedAttention>(bad.operation).executionPhase=P::Prefill;
        EXPECT_THROW((Stage(bad)),std::invalid_argument);
    }
    {
        auto bad=p;std::get<Stage::SelectedAttention>(bad.operation).retainedMtpCandidateRows=2;
        EXPECT_THROW((Stage(bad)),std::invalid_argument);
    }
    {
        auto bad=p;std::get<Stage::SelectedAttention>(bad.operation).executionPhase=static_cast<P>(255);
        EXPECT_THROW((Stage(bad)),std::invalid_argument);
    }
    cache.keys=ActivationPrecision::BF16;EXPECT_THROW((Stage(p)),std::invalid_argument);cache.keys=ActivationPrecision::FP16;
    cache.values=ActivationPrecision::BF16;EXPECT_THROW((Stage(p)),std::invalid_argument);
}
#endif

/** @test Omitted intent resolves by actual backend/geometry, with explicit choices preserved. */
TEST(QwenSparseAttentionStage, InstalledDefaultSetupResolvesBackendAndExplicitChoices)
{
    using namespace qwen4exp;
    using A = QwenSparseAttentionArithmetic;
    using S = QwenSparseAttentionPhaseSchedule;
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (const auto storage :
             {TensorType::FP16, TensorType::FP32, TensorType::Q16_1, TensorType::TQ4})
            for (const int requests : {1, 2})
            {
                const QwenSparseAttentionSetupGeometry geometry{
                    backend, storage, storage, requests, 256, 64};
                const bool eligible =
                    backend == DeviceType::ROCm && storage == TensorType::FP16 && requests == 1;
                const auto actual = resolveQwenSparseAttentionSetup({}, geometry);
                EXPECT_EQ(actual.arithmetic, eligible ? A::TileFP16WithFP32Totals : A::FixedFP32);
                EXPECT_EQ(actual.schedule, eligible ? S::Prefill8Decode33 : S::Uniform);
                const auto control =
                    resolveQwenSparseAttentionSetup({A::FixedFP32, std::nullopt}, geometry);
                EXPECT_EQ(control.arithmetic, A::FixedFP32);
                EXPECT_EQ(control.schedule, S::Uniform);
            }
    const QwenSparseAttentionSetupGeometry native{
        DeviceType::ROCm, TensorType::FP16, TensorType::FP16, 1, 256, 64};
    EXPECT_EQ(
        resolveQwenSparseAttentionSetup({A::TileFP16WithFP32Totals, std::nullopt}, native).schedule,
        S::Prefill8Decode33);
    EXPECT_EQ(resolveQwenSparseAttentionSetup({std::nullopt, S::Uniform}, native).schedule,
              S::Uniform);
    EXPECT_EQ(resolveQwenSparseAttentionSetup({std::nullopt, S::Uniform}, native).arithmetic,
              A::TileFP16WithFP32Totals);
    EXPECT_EQ(
        resolveQwenSparseAttentionSetup({std::nullopt, S::Prefill8Decode33}, native).arithmetic,
        A::TileFP16WithFP32Totals);
    auto multi = native;
    multi.request_capacity = 2;
    EXPECT_EQ(
        resolveQwenSparseAttentionSetup({A::TileFP16WithFP32Totals, S::Uniform}, multi).schedule,
        S::Uniform);
    EXPECT_THROW(resolveQwenSparseAttentionSetup({A::FixedFP32, S::Prefill8Decode33}, native),
                 std::invalid_argument);
    EXPECT_THROW(resolveQwenSparseAttentionSetup({std::nullopt, S::Prefill8Decode33}, multi),
                 std::invalid_argument);
    for (const auto invalid : {A::TileFP16WithFP32Totals33, static_cast<A>(255)})
        EXPECT_THROW(resolveQwenSparseAttentionSetup({invalid, std::nullopt}, native),
                     std::invalid_argument);
    EXPECT_THROW(resolveQwenSparseAttentionSetup({std::nullopt, static_cast<S>(255)}, native),
                 std::invalid_argument);
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA})
    {
        auto other = native;
        other.backend = backend;
        EXPECT_THROW(
            resolveQwenSparseAttentionSetup({A::TileFP16WithFP32Totals, std::nullopt}, other),
            std::invalid_argument);
    }
    for (const int bad : {0, -1})
    {
        auto malformed = native;
        malformed.request_capacity = bad;
        EXPECT_THROW(resolveQwenSparseAttentionSetup({}, malformed), std::invalid_argument);
    }
    auto other = native;
    other.rotary_dimension = 128;
    EXPECT_EQ(resolveQwenSparseAttentionSetup({}, other).arithmetic, A::FixedFP32);
    EXPECT_THROW(resolveQwenSparseAttentionSetup({A::TileFP16WithFP32Totals, std::nullopt}, other),
                 std::invalid_argument);
}
/** @test Factories can only borrow a unique actual admitted participant, without default
 * resolution. */
TEST(QwenSparseAttentionStage, AdmittedDefaultProjectionRejectsMissingDuplicateAndConflictingOwners)
{
    using namespace qwen4exp;
    using A = QwenSparseAttentionArithmetic;
    using S = QwenSparseAttentionPhaseSchedule;
    const QwenSparseAttentionResolvedPolicy half{A::TileFP16WithFP32Totals, S::Prefill8Decode33};
    const auto projection = sealQwenSparseAttentionParticipants(
        {{DeviceId::cpu(), {}, 0, 0}, {DeviceId::rocm(3), half, 0, 0}});
    EXPECT_EQ(admittedQwenSparseAttentionPolicy(projection, DeviceId::rocm(3), 0, 0), half);
    EXPECT_EQ(admittedQwenSparseAttentionPolicy(projection, DeviceId::cpu(), 0, 0).arithmetic,
              A::FixedFP32);
    EXPECT_THROW(admittedQwenSparseAttentionPolicy({}, DeviceId::rocm(3), 0, 0),
                 std::invalid_argument);
    EXPECT_THROW(admittedQwenSparseAttentionPolicy(projection, DeviceId::rocm(0), 0, 0),
                 std::invalid_argument);
    EXPECT_THROW(sealQwenSparseAttentionParticipants({}), std::invalid_argument);
    EXPECT_THROW(sealQwenSparseAttentionParticipants(
                     {{DeviceId::rocm(3), half, 0, 0}, {DeviceId::rocm(3), half, 0, 0}}),
                 std::invalid_argument);
    EXPECT_THROW(sealQwenSparseAttentionParticipants({{DeviceId::cuda(0), half, 0, 0}}),
                 std::invalid_argument);
    EXPECT_THROW(sealQwenSparseAttentionParticipants(
                     {{DeviceId::rocm(0), {A::FixedFP32, S::Prefill8Decode33}, 0, 0}}),
                 std::invalid_argument);
    const auto disjoint = sealQwenSparseAttentionParticipants(
        {{DeviceId::rocm(3), half, 0, 3}, {DeviceId::rocm(3), half, 4, 7}});
    EXPECT_EQ(admittedQwenSparseAttentionPolicy(disjoint, DeviceId::rocm(3), 4, 7), half);
    EXPECT_THROW(admittedQwenSparseAttentionPolicy(disjoint, DeviceId::rocm(3), 0, 7),
                 std::invalid_argument);
    EXPECT_THROW(sealQwenSparseAttentionParticipants(
                     {{DeviceId::rocm(3), half, 0, 3}, {DeviceId::rocm(3), half, 3, 7}}),
                 std::invalid_argument);
}

/** @test No-ranking retains required contracts while never publishing or observing query-only banks. */
TEST(QwenSparseAttentionStage, ProvenNoRankingOmitsQueryContractsAndSnapshots)
{
    llaminar::v2::ThreadCountGuard threads(1);
    CPUDeviceContext context(DeviceId::cpu(),1);
    constexpr int rows=3,tokens=8,positions=16;
    FP32Tensor query({rows,512}),raw({rows,128}),partial({4,128}),pools({2,128}),
        prepared({rows,512}),scores({rows,2}),queryGain({128}),keyGain({128}),cosine({positions,32}),sine({positions,32});
    INT32Tensor replay({3}),selected({rows,2051}),counts({rows});
    auto *state=static_cast<int *>(replay.raw_mutable_data());
    state[0]=2;state[1]=rows;state[2]=0;
    for (auto *unused : {&query,&prepared,&scores,&queryGain})
        std::fill_n(unused->mutable_data(),unused->numel(),std::numeric_limits<float>::quiet_NaN());
    std::fill_n(raw.mutable_data(),raw.numel(),0.75f);
    std::fill_n(partial.mutable_data(),partial.numel(),0.25f);
    std::fill_n(pools.mutable_data(),pools.numel(),-111.0f);
    std::fill_n(keyGain.mutable_data(),keyGain.numel(),1.0f);
    std::fill_n(cosine.mutable_data(),cosine.numel(),1.0f);
    std::fill_n(sine.mutable_data(),sine.numel(),0.0f);
    Stage::Indexer operation{Float(query,BufferId::QFN_QSA_QUERY),Float(raw,BufferId::QFN_QSA_RAW_KEY),
        Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS),Float(pools,BufferId::QFN_QSA_POOLED_KEYS),
        Weight(queryGain),Weight(keyGain),Weight(cosine),Weight(sine),Float(prepared,BufferId::QFN_QSA_PREPARED_QUERY),
        Float(scores,BufferId::QFN_QSA_SCORES),Int(selected,BufferId::QFN_QSA_SELECTED),
        Int(counts,BufferId::QFN_QSA_SELECTED_COUNT),2,positions};
    operation.queryDemand=qwen4exp::QwenSparseQueryDemand::ProvenNoRanking;
    Stage omitted({.rowCapacity=rows,.tokenCapacity=tokens,.replay=Int(replay,BufferId::QFN_QSA_REPLAY),.operation=operation});
    const auto contract=omitted.bufferContract();
    for (const auto role : {BufferId::QFN_QSA_QUERY,BufferId::QFN_QSA_PREPARED_QUERY,BufferId::QFN_QSA_SCORES}) {
        EXPECT_TRUE(std::none_of(contract.inputs.begin(),contract.inputs.end(),[&](const auto &v){return v.id==role;}));
        EXPECT_TRUE(std::none_of(contract.outputs.begin(),contract.outputs.end(),[&](const auto &v){return v.id==role;}));
        EXPECT_TRUE(std::none_of(contract.inouts.begin(),contract.inouts.end(),[&](const auto &v){return v.id==role;}));
    }
    ASSERT_EQ(contract.weight_tensors.size(),3u);
    EXPECT_TRUE(std::none_of(contract.weight_tensors.begin(),contract.weight_tensors.end(),
        [&](const auto *weight){return weight==&queryGain;}));
    ASSERT_EQ(contract.inouts.size(),1u);EXPECT_EQ(contract.inouts[0].id,BufferId::QFN_QSA_POOLED_KEYS);
    EXPECT_EQ(contract.outputs.size(),2u);
    EXPECT_TRUE(omitted.execute(&context));
    const auto dump=omitted.getDumpInfoSnapshot();
    for (const auto &input : dump.inputs) EXPECT_NE(std::string_view(input.name),"query");
    ASSERT_EQ(dump.outputs.size(),3u);
    for (const auto &output : dump.outputs) {
        EXPECT_NE(std::string_view(output.name),"prepared_query");
        EXPECT_NE(std::string_view(output.name),"scores");
    }
    for (auto *unused : {&query,&prepared,&scores,&queryGain})
        for (std::size_t i=0;i<unused->numel();++i) EXPECT_TRUE(std::isnan(unused->data()[i]));
    EXPECT_EQ(static_cast<const int *>(counts.raw_data())[0],3);
    operation.queryDemand=qwen4exp::QwenSparseQueryDemand::RankingOnly;
    Stage full({.rowCapacity=rows,.tokenCapacity=tokens,.replay=Int(replay,BufferId::QFN_QSA_REPLAY),.operation=operation});
    EXPECT_NE(omitted.graphCaptureVariantSignature(),full.graphCaptureVariantSignature());
}
