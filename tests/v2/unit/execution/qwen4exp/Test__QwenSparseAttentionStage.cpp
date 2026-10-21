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
#include "execution/local_execution/graph/StageVerifier.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "tensors/Tensors.h"
#include "tensors/TensorVerification.h"
#include "tensors/GPUTensorVerification.h"
#include "utils/DebugEnv.h"
#include "utils/MPIContext.h"
#include "utils/OpenMPUtils.h"
#include "../../../integration/kernels/qwen4exp/QwenSparseNativeCodecFixture.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string_view>

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
