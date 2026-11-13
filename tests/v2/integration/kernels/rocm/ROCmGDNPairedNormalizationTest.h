/**
 * @file ROCmGDNPairedNormalizationTest.h
 * @brief Captured production/historical grouped bytes with dirty device lengths.
 *
 * Included by the existing ROCm GDN fixture after its Buffer/Graph utilities.
 * Both scale-only and L2-normalized policies compare with the independent
 * literal historical grouped and serial kernels across the same retained history.
 * Serial-row replay exists only as a literal historical test oracle. Production
 * always performs one grouped native launch. All buffers and host upload spans
 * outlive the retirement join, and no allocation occurs inside capture.
 */
#pragma once
#include <exception>
#include <utility>

/**
 * @brief Closed Q/K policy admitted by the historical byte-proof harness.
 * @details ScaleOnly retains the historical Q*rsqrt(DK) and raw K contract;
 * L2Normalized additionally applies the original Q/K squared-norm expressions.
 * These test selectors invoke the ordinary production policy argument and do
 * not replace device arithmetic or add a production execution mode.
 */
enum class GDNHistoricalQKPolicy
{
    ScaleOnly, ///< Keep raw K and the original fixed Q scale.
    L2Normalized, ///< Use the existing additive-epsilon L2 normalization.
};

/**
 * @brief Compare complete outputs and state publication to a frozen native oracle.
 * @param rows Captured physical rows per request; live counts change only on device.
 * @param dk Immutable key width selecting the actual64/128 production specialization.
 * @param dv Value width;65 exercises inactive lanes in the final column tile.
 * @param snapshots Whether row snapshots or the terminal state owns publication.
 * @param normalization Explicit Q/K policy shared by production and both frozen native oracles.
 * @details Two requests have unequal/empty live prefixes. Mixed finite and signed
 * zero inputs exercise scale-only publication as well as both normalized
 * four-wave folds; whole-zero normalized heads stress the additive epsilon.
 * Gate values include exact half-representable20-boundary inputs, but this test
 * does not claim to execute the separate FP16 projection kernels.
 */
static void provePairedNormalizationHistorical(int rows,int dk,int dv,bool snapshots,GDNHistoricalQKPolicy normalization)
{
    const bool normalize=normalization==GDNHistoricalQKPolicy::L2Normalized;
    constexpr int requests=2,heads=2;
    const int total=requests*rows,qwidth=heads*dk,vwidth=heads*dv;
    const int state_width=heads*dk*dv;
    std::vector<float> q(total*qwidth),k(q.size()),v(total*vwidth),alpha(total*heads),beta(alpha.size());
    const std::array<float,8> values{0.0f,-0.0f,0x1p-20f,-0x1p-15f,0.25f,-0.5f,0.75f,-1.0f};
    for(int row=0;row<total;++row)for(int h=0;h<heads;++h)for(int d=0;d<dk;++d){
        const auto i=std::size_t(row)*qwidth+h*dk+d;
        q[i]=(row+h)%7==0 ? values[d%2] : values[(d+row+h)%8];
        k[i]=(row+h)%5==0 ? values[(d+1)%2] : values[(d*3+row+h+1)%8];
    }
    for(std::size_t i=0;i<v.size();++i)v[i]=values[(i*5+3)%8];
    const std::array<float,6> alphas{-0.5f,0.0f,19.75f,19.875f,20.0f,20.5f};
    for(std::size_t i=0;i<alpha.size();++i){alpha[i]=alphas[i%6];beta[i]=float(int(i%5)-2)*4.0f;}
    std::vector<float> initial(requests*state_width),a_log(heads,-0.375f),bias(heads,0.125f);
    for(std::size_t i=0;i<initial.size();++i)initial[i]=float(int(i%17)-8)*0.00073f;
    HipStreamHandle stream;
    HipFloatBuffer dq(q),dkv(k),dvbuf(v),da(alpha),db(beta),dlog(a_log),dbias(bias),dinitial(initial),serialState(initial);
    HipFloatBuffer actual(total*vwidth,-77.f),oldGrouped(total*vwidth,-77.f),oldSerial(total*vwidth,-77.f);
    HipFloatBuffer actualState(requests*state_width,-91.f),oldState(requests*state_width,-91.f);
    HipFloatBuffer actualSnapshots(total*state_width,-83.f),oldSnapshots(total*state_width,-83.f);
    HipIntBuffer lengths({rows,rows});
    std::array<int,requests> hostLengths{rows,rows};
    std::unique_ptr<HipCapturedGraph> candidateGraph,historicalGraph,serialGraph;
    /** @brief Last owner retires asynchronous work before graphs, buffers or upload spans. */
    struct Retirement final{hipStream_t stream;~Retirement(){if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();}} retirement{stream.stream};
    const gdn_historical_native::Arguments args{
        .q=dq.ptr,.k=dkv.ptr,.v=dvbuf.ptr,.alpha=da.ptr,.beta=db.ptr,.a_log=dlog.ptr,.dt_bias=dbias.ptr,
        .output=oldGrouped.ptr,.initial=dinitial.ptr,.updated=oldState.ptr,
        .requests=requests,.rows=rows,.heads=heads,.key_width=dk,.value_width=dv,.normalize=normalize,
        .lengths=lengths.ptr,.row_offset=0,.snapshots=snapshots?oldSnapshots.ptr:nullptr,
        .snapshot_stride=state_width,.snapshot_rows=snapshots?total:0,.stream=stream.stream};
    historicalGraph=std::make_unique<HipCapturedGraph>(stream.stream,[&]{return gdn_historical_native::launch(args);});
    candidateGraph=std::make_unique<HipCapturedGraph>(stream.stream,[&]{
        return rocmGDN_chunk_forward_batched_effective(dq.ptr,dkv.ptr,dvbuf.ptr,da.ptr,db.ptr,dlog.ptr,dbias.ptr,
            actual.ptr,dinitial.ptr,actualState.ptr,total,requests,rows,heads,dk,dv,normalize,lengths.ptr,
            snapshots?actualSnapshots.ptr:nullptr,state_width,snapshots?total:0,0,stream.stream);
    });
    serialGraph=std::make_unique<HipCapturedGraph>(stream.stream,[&]{
        for(int request=0;request<requests;++request)for(int row=0;row<rows;++row){
            auto one=args;const int flat=request*rows+row;
            one.q=dq.ptr+flat*qwidth;one.k=dkv.ptr+flat*qwidth;one.v=dvbuf.ptr+flat*vwidth;
            one.alpha=da.ptr+flat*heads;one.beta=db.ptr+flat*heads;one.output=oldSerial.ptr+flat*vwidth;
            one.initial=serialState.ptr+request*state_width;one.updated=serialState.ptr+request*state_width;
            one.requests=1;one.rows=1;one.lengths=lengths.ptr+request;one.row_offset=row;
            one.snapshots=nullptr;one.snapshot_rows=0;
            if(!gdn_historical_native::launch(one))return false;
        }
        return true;
    });
    const int repeats=rows==16?20:4;
    for(int repeat=0;repeat<repeats;++repeat){
        switch(repeat%4){
        case 0:hostLengths={rows,rows};break;
        case 1:hostLengths={rows-1,rows/2};break;
        case 2:hostLengths={0,0};break;
        default:hostLengths={rows,rows>1?1:0};break;
        }
        checkHip(hipMemcpyAsync(lengths.ptr,hostLengths.data(),sizeof(hostLengths),hipMemcpyHostToDevice,stream.stream),"publish historical QK lengths");
        actual.fill(-77.f);oldGrouped.fill(-77.f);oldSerial.fill(-77.f);
        actualState.fill(-91.f);oldState.fill(-91.f);serialState.copyFrom(initial);
        actualSnapshots.fill(-83.f);oldSnapshots.fill(-83.f);
        candidateGraph->launch(stream.stream);historicalGraph->launch(stream.stream);serialGraph->launch(stream.stream);
        checkHip(hipStreamSynchronize(stream.stream),"historical QK oracle terminal join");
        const auto result=actual.toHost(),reference=oldGrouped.toHost(),serial=oldSerial.toHost();
        EXPECT_EQ(std::memcmp(result.data(),reference.data(),result.size()*sizeof(float)),0);
        EXPECT_EQ(std::memcmp(result.data(),serial.data(),result.size()*sizeof(float)),0);
        const auto currentState=actualState.toHost(),referenceState=oldState.toHost(),serialStateWords=serialState.toHost();
        EXPECT_EQ(std::memcmp(currentState.data(),referenceState.data(),currentState.size()*sizeof(float)),0);
        const auto currentSnapshots=actualSnapshots.toHost(),referenceSnapshots=oldSnapshots.toHost();
        EXPECT_EQ(std::memcmp(currentSnapshots.data(),referenceSnapshots.data(),currentSnapshots.size()*sizeof(float)),0);
        for(int request=0;request<requests;++request){
            const auto begin=std::size_t(request)*state_width;
            const float* terminal=snapshots && hostLengths[request]>0 ?
                currentSnapshots.data()+std::size_t(request*rows+hostLengths[request]-1)*state_width :
                snapshots ? initial.data()+begin : currentState.data()+begin;
            EXPECT_EQ(std::memcmp(terminal,serialStateWords.data()+begin,state_width*sizeof(float)),0);
        }
        ASSERT_FALSE(::testing::Test::HasFailure())<<"rows="<<rows<<" dk="<<dk<<" dv="<<dv<<" normalize="<<normalize<<" phase="<<repeat;
    }
}

/**
 * @test Both Q/K policies preserve historical grouped and serial output/state bytes.
 * @details Each mode retains all M1..16, width/tail and full/partial/empty/unequal
 * histories; M16 performs20 replays. The existing explicit preflight selector
 * covers this policy expansion without adding another native suite.
 */
TEST(Test__ROCmGDNPaddedRealLength, PairedQKNormalizationHistoricalGroupedBytes)
{
    if(!hasROCm())GTEST_SKIP()<<"No ROCm device available";
    checkHip(hipSetDevice(0),"historical QK device");
    for(const auto normalization:{GDNHistoricalQKPolicy::ScaleOnly,GDNHistoricalQKPolicy::L2Normalized})
        for(int rows=1;rows<=16;++rows)for(const auto geometry:std::array{std::pair{64,64},std::pair{128,128},std::pair{128,65}})
            provePairedNormalizationHistorical(rows,geometry.first,geometry.second,true,normalization);
}

/**
 * @test Non-snapshot M1 preserves both historical Q/K policies and full terminal state.
 * @details The same independent historical grouped and serial oracles, dirty
 * counts, signed-zero/mixed finite inputs and width/tail cases are retained.
 */
TEST(Test__ROCmGDNPaddedRealLength, PairedQKNormalizationHistoricalScalarBytes)
{
    if(!hasROCm())GTEST_SKIP()<<"No ROCm device available";
    checkHip(hipSetDevice(0),"historical scalar QK device");
    for(const auto normalization:{GDNHistoricalQKPolicy::ScaleOnly,GDNHistoricalQKPolicy::L2Normalized})
        for(const auto geometry:std::array{std::pair{64,64},std::pair{128,128},std::pair{128,65}})
            provePairedNormalizationHistorical(1,geometry.first,geometry.second,false,normalization);
}
