/**
 * @file QwenSparseFP16ROCmTest.h
 * @brief Explicit native key-encoding FP32/FP16 events and retained replay proof.
 *
 * Included after the shared observation fixture. Both programs borrow identical
 * FP32 queries, authored native FP16 KV, selected IDs and replay positions.
 * Existing FixedRun/Capture/Events own allocation, native identity and lifetime.
 * Timing explicitly selects post-RoPE cached keys or pre-RoPE transform-on-read.
 * No cache conversion is included for post-RoPE; pre-RoPE includes selected-key
 * rotation. FP32/FP16 output equality is not required; modes are separate workloads.
 */
#pragma once
#include <utility>
namespace llaminar2::qwen4exp
{
/**
 * @brief Launch the explicit FP16 arithmetic candidate over the existing eight-partial ABI.
 * @param args Immutable native bindings and device-owned replay/selection state.
 * @param stream Exact non-default producer/merge stream, retained by the fixture.
 * @details Implementation and dependency registration belong to the production backend.
 */
void rocmQwenSparseFP16Attention(const QwenSparseAttention& args,hipStream_t stream);
}
namespace llaminar2::qwen4exp::ROCmQwenSparseAttention
{
namespace fp16_attention
{
using namespace observation115;
/**
 * @brief Bind explicit selected-key transform-on-read to shared raw native KV inputs.
 * @param args Original compact input bindings; the underlying buffers stay borrowed.
 * @param theta Actual model theta for timing, or the explicit test theta for conformance.
 * @param origin Valid physical ring origin in the retained native capacity.
 * @param capacity Actual native token capacity, not a selected-token count.
 * @return Bindings using canonical engine factors and the declared 64-wide rotary prefix.
 * @throws std::runtime_error For invalid theta or ring geometry.
 */
inline QwenSparseAttention selectedKeyTransform(QwenSparseAttention args,float theta,int origin,int capacity)
{
    require(std::isfinite(theta)&&theta>0 && capacity>0 && origin>=0&&origin<capacity,"Invalid selected-key transform fixture policy");
    args.kv.key_policy={.encoding=attention::AttentionKeyCacheEncoding::PreRotaryDeviceTransform,
        .rope_theta=theta,.partial_rotary_factor=0.25f};
    args.kv.key_rotary_source=QwenSparseRotarySource::CanonicalEngineTheta;
    args.kv.key_negative_log_theta=-std::log(theta);args.kv.key_rotary_dim=64;
    args.kv.key_cosine=args.kv.key_sine=nullptr;args.kv.key_position_capacity=0;
    args.kv.addressing=QwenSparseKVAddressing::FixedLogicalView;
    args.kv.logical_view={.logical_row_origin=origin,.physical_row_capacity=capacity};
    args.kv.row_parameters=nullptr;return args;
}
/**
 * @brief Publish authored membership and the exact absolute-position origin before replay.
 * @param f Persistent source fixture and exact stream.
 * @param prefix Accepted logical tokens before the first live row.
 * @param live Admitted live query rows; zero exercises retained inactive storage.
 * @param positionOrigin Actual absolute origin, independent of physical ring origin.
 * @return Original per-row selected counts from the shared authored-membership helper.
 * @details The explicit setup join retires the state upload before timing/correctness collection.
 */
inline std::vector<std::int32_t> publishReplayPosition(Fixture& f,int prefix,int live,int positionOrigin)
{
    require(positionOrigin>=0,"Negative absolute position origin");
    auto counts=publish(f,prefix,live,SelectedDistribution::CausalSpread512Groups);
    f.replay.position_origin=positionOrigin;f.state.upload(&f.replay,f.stream);
    check(hipStreamSynchronize(f.stream));return counts;
}
/** @brief Same output/partial owners as FixedRun, with the explicit FP16 producer. */
struct FP16Run final:FixedRun
{
    /**
     * @brief Bind existing eight-partial storage to the candidate before capture.
     * @param args Shared native input bindings with an explicit admitted key encoding.
     * @param stream Exact fixture stream; its owner outlives this run.
     * @param repeats Complete retained producer/merge pairs, from1 through256.
     */
    FP16Run(QwenSparseAttention args,hipStream_t stream,int repeats=1):FixedRun(args,stream,true,repeats){
        this->args.arithmetic=QwenSparseAttentionArithmetic::TileFP16WithFP32Totals;}
    /**
     * @brief Record the actual canonical-G4 producer and existing ordered reducer.
     * @details Both kernels use256 threads. Producer grid is [M,ceil((start%4+Q)/4),8];
     * merge grid is [M,Q,1]. The inherited validator authenticates functions and all edges.
     * The native Itanium symbol must name the exact true/false transform
     * specialization selected by the immutable key-encoding policy.
     */
    void record()
    {
        const int groups=(args.query_head_start%4+args.query_heads+3)/4;
        const bool transform=args.kv.key_policy.encoding==attention::AttentionKeyCacheEncoding::PreRotaryDeviceTransform;
        const std::string label=std::string("candidate_fp16_")+(transform?"pre_rotary_":"post_rotary_")+(args.observations?"on":"off");
        capture.record([&]{rocmQwenSparseAttention(args,capture.stream);},
            transform?"selectedAttentionFP16TilesILb1EE":"selectedAttentionFP16TilesILb0EE","reduceAttentionPartitions",
            dim3(args.row_capacity,groups,8),dim3(args.row_capacity,args.query_heads,1),256,256,label.c_str());
    }
};
/**
 * @brief Record the FP32 control with exactly the same explicit native key encoding.
 * @param run Existing current-program storage; its input lifetime spans terminal retirement.
 */
inline void recordControl(FixedRun& run)
{
    const bool transform=run.args.kv.key_policy.encoding==attention::AttentionKeyCacheEncoding::PreRotaryDeviceTransform;
    const std::string label=std::string("control_fp32_")+(transform?"pre_rotary_on":"post_rotary_on");
    run.capture.record([&]{rocmQwenSparseAttention(run.args,run.capture.stream);},
        "selectedAttentionPartitions","reduceAttentionPartitions",
        dim3(run.args.row_capacity,run.args.query_heads,8),dim3(run.args.row_capacity,run.args.query_heads,1),256,256,label.c_str());
}
}
/**
 * @test Both explicit native key encodings retain serial bytes through dirty captured replays.
 * @details Each encoding and M1..16 owns a fresh fixture and one retained grouped
 * graph. Both bind the same physical ring and absolute replay origin. The
 * long/short/inactive/partial/long lifecycle preserves empty payload and inactive
 * outputs while checking observations and output guards. Per-row serial captures
 * borrow that same fixture and retire before its persistent storage is released.
 */
TEST(ROCmQwenSparseAttention,FP16NativeKeyEncodingsGroupedSerialDirtyReplay)
{
    using namespace fp16_attention;
    constexpr float theta=10000000.0f; // Explicit conformance parameter; timings require the model theta.
    for(bool transform:{false,true}){
        SCOPED_TRACE(transform?"pre_rotary":"post_rotary");
        for(int rows=1;rows<=16;++rows){
            SCOPED_TRACE(rows);
            Fixture f(rows,2050,4096,QwenSparseScalar::FP16);Inputs inputs(f,0,6,0,1);
            auto base=inputs.base();
            base.kv.addressing=QwenSparseKVAddressing::FixedLogicalView;
            base.kv.logical_view={.logical_row_origin=4095,.physical_row_capacity=f.capacity};
            base.kv.row_parameters=nullptr;
            if(transform)base=selectedKeyTransform(base,theta,4095,f.capacity);
            FP16Run grouped(base,f.stream);grouped.record();
            auto prior=grouped.initial_partials;auto previousOutput=grouped.poison;
            for(const auto [prefix,live]:{std::pair{2050,rows},std::pair{0,std::min(rows,3)},std::pair{0,0},std::pair{256,rows},std::pair{2050,rows}}){
                const auto counts=publishReplayPosition(f,prefix,live,65536);
                grouped.capture.launch();
                const auto result=grouped.output.download(f.stream);const auto parts=grouped.partials.download(f.stream);
                const auto observations=grouped.observations.download(f.stream);
                for(int row=0;row<rows;++row){
                    const auto first=std::size_t(row)*6*8;
                    if(row>=live){
                        ASSERT_EQ(std::memcmp(parts.data()+first,prior.data()+first,6*8*sizeof(QwenSparseAttentionPartial)),0);
                        ASSERT_EQ(std::memcmp(result.data()+std::size_t(row)*1536,previousOutput.data()+std::size_t(row)*1536,1536*sizeof(float)),0);
                        for(int head=0;head<6;++head){const auto& o=observations[row*6+head];ASSERT_EQ(o.key_elements,0u);ASSERT_EQ(o.value_elements,0u);ASSERT_EQ(o.selected_tokens,0u);}
                        continue;
                    }
                    for(int head=0;head<6;++head)for(int part=0;part<8;++part)if(part*64>=counts[row]){
                        const auto i=(std::size_t(row)*6+head)*8+part;
                        ASSERT_EQ(parts[i].maximum,-INFINITY);ASSERT_EQ(parts[i].denominator,0.0f);
                        ASSERT_EQ(parts[i].key_elements,0u);ASSERT_EQ(parts[i].value_elements,0u);
                        ASSERT_EQ(std::memcmp(parts[i].weighted,prior[i].weighted,sizeof(parts[i].weighted)),0);
                    }
                    auto serialArgs=base;serialArgs.row_capacity=1;
                    serialArgs.query=inputs.query.data+std::size_t(row)*1536;
                    serialArgs.selected=f.selected.data+std::size_t(row)*QwenSparseSelected;
                    serialArgs.selected_count=f.count.data+row;
                    FP16Run serial(serialArgs,f.stream);serial.record();
                    std::copy_n(prior.data()+first,6*8,serial.initial_partials.data());
                    serial.partials.upload(serial.initial_partials.data(),f.stream);
                    f.replay={prefix+row,1,65536};f.state.upload(&f.replay,f.stream);
                    serial.capture.launch();
                    const auto one=serial.output.download(f.stream);const auto oneParts=serial.partials.download(f.stream);
                    ASSERT_EQ(std::memcmp(one.data(),result.data()+std::size_t(row)*1536,1536*sizeof(float)),0);
                    ASSERT_EQ(std::memcmp(oneParts.data(),parts.data()+first,6*8*sizeof(QwenSparseAttentionPartial)),0);
                    for(int head=0;head<6;++head){const auto& o=observations[row*6+head];const bool owner=head==0||head==4;
                        ASSERT_EQ(o.key_elements,owner?std::uint64_t(counts[row])*256:0u);
                        ASSERT_EQ(o.value_elements,owner?std::uint64_t(counts[row])*256:0u);ASSERT_EQ(o.selected_tokens,counts[row]);}
                }
                ASSERT_EQ(std::memcmp(result.data()+std::size_t(rows)*1536,grouped.poison.data()+std::size_t(rows)*1536,16*sizeof(float)),0);
                prior=parts;previousOutput=result;
            }
        }
    }
}
namespace fp16_attention
{
/**
 * @brief Check source-level shared read ownership by enumerating global-head consumers.
 * @param run Completed candidate invocation and its original native bindings.
 * @param counts Authored positive selected-token counts for every admitted row.
 * @details The first consumer encountered in each canonical four-head group owns
 * its source reads. These are decoded coordinates, not observed HBM transactions.
 */
inline void checkSharedReadOwners(FP16Run& run,const std::vector<std::int32_t>& counts)
{
    const auto& a=run.args;ASSERT_EQ(counts.size(),std::size_t(a.row_capacity));
    const auto observed=run.observations.download(run.capture.stream);
    const auto parts=run.partials.download(run.capture.stream);
    for(int row=0;row<a.row_capacity;++row)for(int head=0;head<a.query_heads;++head){
        const int global=a.query_head_start+head;bool first=true;
        for(int previous=0;previous<head;++previous)
            if((a.query_head_start+previous)/4==global/4)first=false;
        std::array<std::uint32_t,8> consumed{};
        for(int token=0;token<counts[row];++token)++consumed[(token/64)%8];
        const std::uint64_t keyCoordinates=256; // Paired RoPE loads each native coordinate once.
        const auto& o=observed[row*a.query_heads+head];
        ASSERT_EQ(o.key_elements,first?std::uint64_t(counts[row])*keyCoordinates:0u);
        ASSERT_EQ(o.value_elements,first?std::uint64_t(counts[row])*256:0u);
        ASSERT_EQ(o.selected_tokens,counts[row]);
        for(int part=0;part<8;++part){const auto& p=parts[(std::size_t(row)*a.query_heads+head)*8+part];
            ASSERT_EQ(p.key_elements,first?consumed[part]*keyCoordinates:0u);
            ASSERT_EQ(p.value_elements,first?consumed[part]*256u:0u);
        }
    }
}
/**
 * @brief Independently fold uniform-score constant values with declared FP16 rounding.
 * @param count Positive selected tokens, including arbitrary64-token stream tails.
 * @param value Exactly representable native half value, including maximum finite half.
 * @return Expected constant output with half-RNE at every PV update and FP32 merge.
 * @throws std::runtime_error For invalid live count or a non-native/nonfinite value.
 * @details Q=0 makes every real exponential exactly1. Each parity chain has at
 * most32 terms scaled by1/128. Scalar host half conversion is independent of the
 * optimized vector dot/PV implementation; no GPU result is used as the oracle.
 */
inline float constantHalfOracle(int count,float value)
{
    require(count>0&&count<=QwenSparseSelected&&std::isfinite(value)&&detail::decodeHalf(detail::encodeHalf(value))==value,
        "Constant oracle requires a live extent and exactly representable finite half value");
    float numerator=0.0f,denominator=0.0f;
    for(int part=0;part<8;++part){float parityTotals[2]{},partDenominator=0.0f;
        for(int begin=part*64;begin<count;begin+=512){
            const int live=std::min(64,count-begin);float parity[2]{};
            for(int side=0;side<2;++side)for(int chunk=0;chunk<2;++chunk)for(int step=0;step<16;++step){
                const int token=chunk*32+2*step+side;
                if(token>=live)continue;
                parity[side]=detail::decodeHalf(detail::encodeHalf(std::fmaf(0x1p-7f,value,parity[side])));
            }
            for(int side=0;side<2;++side)parityTotals[side]=std::fmaf(parityTotals[side],1.0f,parity[side]);
            partDenominator=std::fmaf(partDenominator,1.0f,float(live)*0x1p-7f);
        }
        numerator=std::fmaf(parityTotals[0]+parityTotals[1],1.0f,numerator);denominator+=partDenominator;
    }
    return numerator/denominator;
}
}
/** @test Global compact heads, canonical groups, read-only operands and observation neutrality. */
TEST(ROCmQwenSparseAttention,FP16CompactHeadsSharedReadsAndReadOnlyInputs)
{
    using namespace fp16_attention;
    Fixture f(3,63,4096,QwenSparseScalar::FP16);
    const auto counts=publishReplayPosition(f,63,3,65536);
    const auto selectedBefore=f.selected.download(f.stream),countsBefore=f.count.download(f.stream);
    for(bool transform:{false,true}){
        Inputs fullInput(f,0,24,0,2);auto fullBase=fullInput.base();
        fullBase.kv.addressing=QwenSparseKVAddressing::FixedLogicalView;
        fullBase.kv.logical_view={.logical_row_origin=4095,.physical_row_capacity=f.capacity};
        if(transform)fullBase=selectedKeyTransform(fullBase,10000000.0f,4095,f.capacity);
        FP16Run full(fullBase,f.stream);full.record();full.capture.launch();
        const auto expected=full.output.download(f.stream);const auto expectedParts=full.partials.download(f.stream);
        checkSharedReadOwners(full,counts);ASSERT_FALSE(::testing::Test::HasFailure());
        const auto checkView=[&](int first,int heads){
            SCOPED_TRACE(::testing::Message()<<"pre_rope="<<transform<<" first="<<first<<" heads="<<heads);
            const int firstKV=first/12,kvCount=(first+heads-1)/12-firstKV+1;
            Inputs input(f,first,heads,firstKV,kvCount);auto a=input.base();
            a.kv.addressing=QwenSparseKVAddressing::FixedLogicalView;
            a.kv.logical_view={.logical_row_origin=4095,.physical_row_capacity=f.capacity};
            if(transform)a=selectedKeyTransform(a,10000000.0f,4095,f.capacity);
            FP16Run compact(a,f.stream);compact.record();compact.capture.launch();
            const auto actual=compact.output.download(f.stream);const auto parts=compact.partials.download(f.stream);
            for(int row=0;row<3;++row)for(int head=0;head<heads;++head){
                ASSERT_EQ(std::memcmp(actual.data()+(std::size_t(row)*heads+head)*256,expected.data()+(std::size_t(row)*24+first+head)*256,256*sizeof(float)),0);
                for(int part=0;part<8;++part){
                    const auto i=(std::size_t(row)*heads+head)*8+part,j=(std::size_t(row)*24+first+head)*8+part;
                    ASSERT_EQ(std::memcmp(&parts[i].maximum,&expectedParts[j].maximum,sizeof(float)),0);
                    ASSERT_EQ(std::memcmp(&parts[i].denominator,&expectedParts[j].denominator,sizeof(float)),0);
                    ASSERT_EQ(std::memcmp(parts[i].weighted,expectedParts[j].weighted,sizeof(parts[i].weighted)),0);
                    if(part*64>=counts[row])ASSERT_EQ(std::memcmp(parts[i].weighted,compact.initial_partials[i].weighted,sizeof(parts[i].weighted)),0);
                }
            }
            checkSharedReadOwners(compact,counts);ASSERT_FALSE(::testing::Test::HasFailure());
            ASSERT_EQ(std::memcmp(actual.data()+std::size_t(3)*heads*256,compact.poison.data()+std::size_t(3)*heads*256,16*sizeof(float)),0);
            const auto q=input.query.download(f.stream);const auto keys=input.keys.download(f.stream),values=input.values.download(f.stream);
            ASSERT_EQ(std::memcmp(q.data(),input.host_query.data(),q.size()*sizeof(float)),0);
            ASSERT_EQ(std::memcmp(keys.data(),input.host_keys.data(),keys.size()*sizeof(std::uint16_t)),0);
            ASSERT_EQ(std::memcmp(values.data(),input.host_values.data(),values.size()*sizeof(std::uint16_t)),0);
        };
        for(const auto& [first,heads]:{std::pair{1,1},std::pair{11,3},std::pair{6,6},std::pair{12,12},std::pair{0,24}}){checkView(first,heads);ASSERT_FALSE(::testing::Test::HasFailure());}
        for(int degree=1;degree<=8;++degree)for(int rank=0;rank<degree;++rank){
            const int first=rank*24/degree,end=(rank+1)*24/degree;
            checkView(first,end-first);ASSERT_FALSE(::testing::Test::HasFailure());
        }
        FP16Run quiet(fullBase,f.stream);quiet.args.observations=nullptr;quiet.record();quiet.capture.launch();
        const auto quietOutput=quiet.output.download(f.stream);const auto quietParts=quiet.partials.download(f.stream);
        ASSERT_EQ(std::memcmp(quietOutput.data(),expected.data(),expected.size()*sizeof(float)),0);
        ASSERT_EQ(std::memcmp(quietParts.data(),expectedParts.data(),expectedParts.size()*sizeof(QwenSparseAttentionPartial)),0);
        const auto keys=fullInput.keys.download(f.stream),values=fullInput.values.download(f.stream);
        ASSERT_EQ(std::memcmp(keys.data(),fullInput.host_keys.data(),keys.size()*sizeof(std::uint16_t)),0);
        ASSERT_EQ(std::memcmp(values.data(),fullInput.host_values.data(),values.size()*sizeof(std::uint16_t)),0);
    }
    ASSERT_EQ(f.selected.download(f.stream),selectedBefore);ASSERT_EQ(f.count.download(f.stream),countsBefore);
}
/** @test Uniform scores preserve the declared constant-value result without half overflow. */
TEST(ROCmQwenSparseAttention,FP16UniformScoresExtremeFiniteValues)
{
    using namespace fp16_attention;
    Fixture f(3,2050,4096,QwenSparseScalar::FP16);Inputs input(f,0,6,0,1);
    std::fill(input.host_query.begin(),input.host_query.end(),0.0f);input.query.upload(input.host_query.data(),f.stream);check(hipStreamSynchronize(f.stream));
    for(bool transform:{false,true}){
        auto a=input.base();if(transform)a=selectedKeyTransform(a,10000000.0f,4095,f.capacity);
        FP16Run run(a,f.stream);run.record();
        for(float value:{0.0f,1.0f,32768.0f,65504.0f,-65504.0f}){
            std::fill(input.host_values.begin(),input.host_values.end(),detail::encodeHalf(value));input.values.upload(input.host_values.data(),f.stream);check(hipStreamSynchronize(f.stream));
            for(int prefix:{63,2050}){
                const auto counts=publishReplayPosition(f,prefix,3,65536);run.capture.launch();
                const auto output=run.output.download(f.stream);const auto parts=run.partials.download(f.stream);
                for(int row=0;row<3;++row){const float expected=constantHalfOracle(counts[row],value);ASSERT_TRUE(std::isfinite(expected));
                    for(int head=0;head<6;++head)for(int d=0;d<256;++d){const float actual=output[(std::size_t(row)*6+head)*256+d];ASSERT_TRUE(std::isfinite(actual));ASSERT_NEAR(actual,expected,std::max(1.0e-6f,std::fabs(expected)*2.0e-6f));}
                }
                for(const auto& part:parts)if(part.denominator>0)for(float v:part.weighted)ASSERT_TRUE(std::isfinite(v));
                ASSERT_EQ(std::memcmp(output.data()+3*1536,run.poison.data()+3*1536,16*sizeof(float)),0);
            }
        }
    }
}

/**
 * @test Complete current/candidate event pairs with an explicitly selected native key encoding.
 * @details Caller must supply QFN_FP16_KEY_ENCODING=post_rotary|pre_rotary
 * before allocation, plus theta and absolute/physical origins. Post-rotary uses
 * the original native bank with the same explicit physical ring; theta and
 * position do not transform that cached bank. Pre-rotary uses canonical selected
 * transform-on-read with the declared 64-wide prefix and physical ring view.
 * Event/profile/export metadata records the actual arithmetic/addressing mode.
 * No FP32/candidate byte comparison is made; candidate numerical acceptance uses KL.
 * QFN_FP16_PROFILE_SIDE selects only control/candidate diagnostics with repeats1;
 * its absence keeps the original paired mode and exact reported repeats.
 */
TEST(ROCmQwenSparseAttention,DISABLED_FP16NativeKeyEncodingAttentionEvents)
{
    using namespace fp16_attention;
    const auto integer=[](const char* name){const char* s=std::getenv(name);require(s&&*s,"Missing exact FP16 comparison selector");std::size_t used=0;int value=std::stoi(s,&used);require(used==std::strlen(s),"Invalid integer FP16 comparison selector");return value;};
    const char* encodingText=std::getenv("QFN_FP16_KEY_ENCODING");
    require(encodingText&&*encodingText,"QFN_FP16_KEY_ENCODING must explicitly select post_rotary or pre_rotary");
    const std::string encoding=encodingText;
    require(encoding=="post_rotary"||encoding=="pre_rotary","Unknown explicit FP16 native key encoding");
    const bool transform=encoding=="pre_rotary";
    const int rows=integer("QFN_FP16_M"),context=integer("QFN_FP16_CONTEXT"),repeats=integer("QFN_FP16_REPEATS");
    const int position=integer("QFN_FP16_POSITION_ORIGIN"),ring=integer("QFN_FP16_RING_ORIGIN");
    const char* text=std::getenv("QFN_FP16_ROPE_THETA");require(text&&*text,"Actual model theta is required");std::size_t used=0;float theta=std::stof(text,&used);require(used==std::strlen(text),"Invalid actual model theta");
    const char* selectedSide=std::getenv("QFN_FP16_PROFILE_SIDE");
    const std::string side=selectedSide?selectedSide:"paired";
    require(side=="paired"||side=="control"||side=="candidate","Unknown exact FP16 profile side");
    const bool paired=side=="paired";
    require(rows>0&&rows<=2048&&context>=rows&&context<=65536&&
        (paired?(repeats>=2&&repeats<=256):repeats==1)&&std::size_t(rows)*repeats<=8192,
        "FP16 comparison requires bounded paired repeats2..256 or one isolated repeat");
    require(position>=0&&std::isfinite(theta)&&theta>0&&ring>=0&&ring<context,
        "Native key encoding requires valid origins/theta and physical ring geometry");
    const char* rotarySource=transform?"canonical_engine":"cached_post_rotary";
    const int rotaryDim=transform?64:0,ringCapacity=context;
    const auto thetaBits=transform?detail::floatBits(theta):std::uint32_t(0);
    const char* addressing="fixed_logical_view";
    Fixture f(rows,context-rows,context,QwenSparseScalar::FP16);Inputs inputs(f,0,6,0,1);
    auto base=inputs.base();
    base.kv.addressing=QwenSparseKVAddressing::FixedLogicalView;
    base.kv.logical_view={.logical_row_origin=ring,.physical_row_capacity=f.capacity};
    base.kv.row_parameters=nullptr;
    if(transform)base=selectedKeyTransform(base,theta,ring,f.capacity);
    FixedRun current(base,f.stream,true,repeats);FP16Run candidate(base,f.stream,repeats);
    recordControl(current);candidate.record();publishReplayPosition(f,context-rows,rows,position);
    Events events(f.stream);
    // Isolated diagnostic processes submit only the named function family. The
    // unchanged complete producer/merge pair retains five warmups and occurrence6.
    if(!paired){
        auto& chosen=side=="control"?current.capture:candidate.capture;
        for(int warm=0;warm<5;++warm)(void)events.sample(chosen);
        const float us=events.sample(chosen);
        std::cout<<"QFN_FP16_SINGLE,"<<side<<",M="<<rows<<",context="<<context<<",Q=6,KV=1,partitions=8,group_heads=4"
            <<",key_encoding="<<encoding<<",key_rotary_source="<<rotarySource<<",key_rotary_dim="<<rotaryDim<<",theta_bits="<<thetaBits
            <<",position_origin="<<position<<",addressing="<<addressing<<",ring_origin="<<ring<<",ring_capacity="<<ringCapacity
            <<",repeats=1,observations=on,occurrence=6,us="<<us<<'\n';
        const auto output=side=="control"?current.output.download(f.stream):candidate.output.download(f.stream);
        const auto& poison=side=="control"?current.poison:candidate.poison;
        for(std::size_t i=0;i<std::size_t(rows)*1536;++i)ASSERT_TRUE(std::isfinite(output[i]));
        ASSERT_EQ(std::memcmp(output.data()+std::size_t(rows)*1536,poison.data()+std::size_t(rows)*1536,16*sizeof(float)),0);
        return;
    }
    for(int warm=0;warm<5;++warm){(void)events.sample(current.capture);(void)events.sample(candidate.capture);}
    for(int pair=0;pair<12;++pair){float control=0,fp16=0;
        if(pair&1){fp16=events.sample(candidate.capture);control=events.sample(current.capture);}
        else{control=events.sample(current.capture);fp16=events.sample(candidate.capture);}
        std::cout<<"QFN_FP16_PAIR,"<<pair<<",M="<<rows<<",context="<<context<<",Q=6,KV=1,partitions=8,group_heads=4"
            <<",key_encoding="<<encoding<<",key_rotary_source="<<rotarySource<<",key_rotary_dim="<<rotaryDim<<",theta_bits="<<thetaBits
            <<",position_origin="<<position<<",addressing="<<addressing<<",ring_origin="<<ring<<",ring_capacity="<<ringCapacity
            <<",repeats="<<repeats<<",observations=on,control_us="<<control<<",candidate_us="<<fp16<<'\n';
    }
    const auto control=current.output.download(f.stream),actual=candidate.output.download(f.stream);
    for(std::size_t i=0;i<std::size_t(rows)*1536;++i){ASSERT_TRUE(std::isfinite(control[i]));ASSERT_TRUE(std::isfinite(actual[i]));}
    ASSERT_EQ(std::memcmp(actual.data()+std::size_t(rows)*1536,candidate.poison.data()+std::size_t(rows)*1536,16*sizeof(float)),0);
    ASSERT_EQ(std::memcmp(control.data()+std::size_t(rows)*1536,current.poison.data()+std::size_t(rows)*1536,16*sizeof(float)),0);
    if(const char* destination=std::getenv("QFN_FP16_EXPORT_DIR")){
        const auto partials=candidate.partials.download(f.stream);std::vector<float> numerical;numerical.reserve(partials.size()*258);
        for(const auto& part:partials){numerical.push_back(part.maximum);numerical.push_back(part.denominator);numerical.insert(numerical.end(),std::begin(part.weighted),std::end(part.weighted));}
        const auto outputCount=std::size_t(rows)*6*256;
        require((outputCount+numerical.size())*sizeof(float)<=128ULL*1024*1024,"FP16 export exceeds128 MiB");
        const std::filesystem::path directory(destination);require(*destination&&!std::filesystem::exists(directory),"FP16 export requires a fresh directory");
        require(std::filesystem::create_directories(directory),"FP16 export directory creation failed");
        const auto write=[&](const char* name,const float* values,std::size_t count){std::ofstream stream(directory/name,std::ios::binary);stream.exceptions(std::ios::failbit|std::ios::badbit);stream.write(reinterpret_cast<const char*>(values),static_cast<std::streamsize>(count*sizeof(float)));stream.close();};
        write("output-fp32.bin",actual.data(),outputCount);write("partials-fp32.bin",numerical.data(),numerical.size());
        std::ofstream shape(directory/"shape.json");shape.exceptions(std::ios::failbit|std::ios::badbit);
        shape<<"{\"M\":"<<rows<<",\"context\":"<<context<<",\"Q\":6,\"KV\":1,\"partitions\":8,\"partial_coordinates\":258,\"selection\":\"causal-spread-512groups\",\"key_encoding\":\""<<encoding<<"\",\"key_rotary_source\":\""<<rotarySource<<"\",\"key_rotary_dim\":"<<rotaryDim<<",\"theta_bits\":"<<thetaBits<<",\"position_origin\":"<<position<<",\"addressing\":\""<<addressing<<"\",\"ring_origin\":"<<ring<<",\"ring_capacity\":"<<ringCapacity<<"}\n";shape.close();
    }
}
}
