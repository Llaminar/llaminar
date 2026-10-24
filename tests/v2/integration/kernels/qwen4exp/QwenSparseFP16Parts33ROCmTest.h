/**
 * @file QwenSparseFP16Parts33ROCmTest.h
 * @brief Production33-tile arithmetic, retained replay and bounded component timing.
 *
 * Included after the original eight-stream fixture. Every candidate row/head owns
 * exactly33 virtual partials, independent of physical scheduling. The explicit
 * production dispatcher borrows their sole pointer after fixture capacity checks.
 * Default policy stays eight; the explicit phase schedule admits this33 program.
 */
#pragma once
namespace llaminar2::qwen4exp::ROCmQwenSparseAttention
{
namespace fp16_attention
{
/**
 * @brief Reuse original storage/capture lifetime with an explicit33-part allocation.
 * @details Capture stays the last base member, so its terminal join retires all
 * borrowed buffers before their destruction. No derived device allocation exists.
 */
struct FP16Run33 final:FixedRun
{
    /**
     * @brief Bind the diagnostic program's exact33-part storage before capture.
     * @param a Borrowed original native inputs with explicit post/pre key policy.
     * @param stream Exact non-default fixture stream, retained through retirement.
     * @param repeats Complete diagnostic producer/merge pairs,1..256.
     */
    FP16Run33(QwenSparseAttention a,hipStream_t stream,int repeats=1)
        :FixedRun(a,stream,true,repeats,FixturePartialStorage::Virtual33){
        args.arithmetic=QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33;
        args.gpu_partials=partials.data;
    }
    /**
     * @brief Authenticate the explicit33 producer/reducer pair and original inputs.
     * @details The checked fixture bank is the sole native scratch authority. Producer
     * grid[M,intersected canonicalG6,33] uses384threads; merge[M,Q,1] uses256.
     * G6/384 is this fixture's static expected candidate policy, never inferred
     * from a captured runtime symbol/resource record.
     */
    void record(){
        const bool transform=args.kv.key_policy.encoding==attention::AttentionKeyCacheEncoding::PreRotaryDeviceTransform;
        const int groups=(args.query_head_start%6+args.query_heads+5)/6;
        require(partials.size==std::size_t(args.row_capacity)*args.query_heads*33 && args.gpu_partials==partials.data,"Wrong diagnostic partial capacity or authority");
        capture.record([&]{rocmQwenSparseAttention(args,capture.stream);},
            transform?"selectedAttentionFP16TilesILb1ELi33EE":"selectedAttentionFP16TilesILb0ELi33EE",
            "reduceAttentionPartitionsILNS_10TensorTypeE2ELi33EE",dim3(args.row_capacity,groups,33),dim3(args.row_capacity,args.query_heads,1),384,256,
            transform?"virtual33_pre_rotary":"virtual33_post_rotary");
    }
};
/**
 * @brief Enumerate native logical coordinates by original head owner and virtual tile.
 * @param run Completed diagnostic capture retaining its original scratch bindings.
 * @param counts Authored live selected counts for every capacity row.
 * @details First owned original head perG6 owns reads once; selected_tokens stays
 * per head. These counters describe decoded coordinates, not physical HBM traffic.
 */
inline void checkSharedReadOwners33(FP16Run33& run,const std::vector<std::int32_t>& counts)
{
    const auto& a=run.args;ASSERT_EQ(counts.size(),std::size_t(a.row_capacity));
    const auto observed=run.observations.download(run.capture.stream);
    const auto parts=run.partials.download(run.capture.stream);
    for(int row=0;row<a.row_capacity;++row)for(int head=0;head<a.query_heads;++head){
        const int global=a.query_head_start+head;bool first=true;
        for(int previous=0;previous<head;++previous)
            if((a.query_head_start+previous)/6==global/6)first=false;
        std::array<std::uint32_t,33> consumed{};
        for(int token=0;token<counts[row];++token)++consumed[token/64];
        const std::uint64_t keyCoordinates=256; // Paired RoPE loads each native coordinate once.
        const auto& o=observed[row*a.query_heads+head];
        ASSERT_EQ(o.key_elements,first?std::uint64_t(counts[row])*keyCoordinates:0u);
        ASSERT_EQ(o.value_elements,first?std::uint64_t(counts[row])*256:0u);
        ASSERT_EQ(o.selected_tokens,counts[row]);
        for(int part=0;part<33;++part){const auto& p=parts[(std::size_t(row)*a.query_heads+head)*33+part];
            ASSERT_EQ(p.key_elements,first?consumed[part]*keyCoordinates:0u);
            ASSERT_EQ(p.value_elements,first?consumed[part]*256u:0u);
        }
    }
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
TEST(ROCmQwenSparseAttention,FP16Virtual33NativeKeyEncodingsGroupedSerialDirtyReplay)
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
            FP16Run33 grouped(base,f.stream);grouped.record();
            auto prior=grouped.initial_partials;auto previousOutput=grouped.poison;
            for(const auto [prefix,live]:{std::pair{2050,rows},std::pair{0,std::min(rows,3)},std::pair{0,0},std::pair{63,rows},std::pair{511,rows},std::pair{512,rows},std::pair{256,rows},std::pair{2050,rows}}){
                const auto counts=publishReplayPosition(f,prefix,live,65536);
                grouped.capture.launch();
                const auto result=grouped.output.download(f.stream);const auto parts=grouped.partials.download(f.stream);
                const auto observations=grouped.observations.download(f.stream);
                for(int row=0;row<rows;++row){
                    const auto first=std::size_t(row)*6*33;
                    if(row>=live){
                        ASSERT_EQ(std::memcmp(parts.data()+first,prior.data()+first,6*33*sizeof(QwenSparseAttentionPartial)),0);
                        ASSERT_EQ(std::memcmp(result.data()+std::size_t(row)*1536,previousOutput.data()+std::size_t(row)*1536,1536*sizeof(float)),0);
                        for(int head=0;head<6;++head){const auto& o=observations[row*6+head];ASSERT_EQ(o.key_elements,0u);ASSERT_EQ(o.value_elements,0u);ASSERT_EQ(o.selected_tokens,0u);}
                        continue;
                    }
                    for(int head=0;head<6;++head)for(int part=0;part<33;++part)if(part*64>=counts[row]){
                        const auto i=(std::size_t(row)*6+head)*33+part;
                        ASSERT_EQ(parts[i].maximum,-INFINITY);ASSERT_EQ(parts[i].denominator,0.0f);
                        ASSERT_EQ(parts[i].key_elements,0u);ASSERT_EQ(parts[i].value_elements,0u);
                        ASSERT_EQ(std::memcmp(parts[i].weighted,prior[i].weighted,sizeof(parts[i].weighted)),0);
                    }
                    auto serialArgs=base;serialArgs.row_capacity=1;
                    serialArgs.query=inputs.query.data+std::size_t(row)*1536;
                    serialArgs.selected=f.selected.data+std::size_t(row)*QwenSparseSelected;
                    serialArgs.selected_count=f.count.data+row;
                    FP16Run33 serial(serialArgs,f.stream);serial.record();
                    std::copy_n(prior.data()+first,6*33,serial.initial_partials.data());
                    serial.partials.upload(serial.initial_partials.data(),f.stream);
                    f.replay={prefix+row,1,65536};f.state.upload(&f.replay,f.stream);
                    serial.capture.launch();
                    const auto one=serial.output.download(f.stream);const auto oneParts=serial.partials.download(f.stream);
                    ASSERT_EQ(std::memcmp(one.data(),result.data()+std::size_t(row)*1536,1536*sizeof(float)),0);
                    ASSERT_EQ(std::memcmp(oneParts.data(),parts.data()+first,6*33*sizeof(QwenSparseAttentionPartial)),0);
                    for(int head=0;head<6;++head){const auto& o=observations[row*6+head];const bool owner=head==0; // Q6 starts at original head0, so exactly one G6 consumer.
                        ASSERT_EQ(o.key_elements,owner?std::uint64_t(counts[row])*256:0u);
                        ASSERT_EQ(o.value_elements,owner?std::uint64_t(counts[row])*256:0u);ASSERT_EQ(o.selected_tokens,counts[row]);}
                }
                ASSERT_EQ(std::memcmp(result.data()+std::size_t(rows)*1536,grouped.poison.data()+std::size_t(rows)*1536,16*sizeof(float)),0);
                prior=parts;previousOutput=result;
            }
        }
    }
}
/** @test Global compact heads, canonical groups, read-only operands and observation neutrality. */
TEST(ROCmQwenSparseAttention,FP16Virtual33CompactHeadsSharedReadsAndReadOnlyInputs)
{
    using namespace fp16_attention;
    Fixture f(3,2050,4096,QwenSparseScalar::FP16);
    const auto counts=publishReplayPosition(f,2050,3,65536);
    const auto selectedBefore=f.selected.download(f.stream),countsBefore=f.count.download(f.stream);
    for(bool transform:{false,true}){
        Inputs fullInput(f,0,24,0,2);auto fullBase=fullInput.base();
        fullBase.kv.addressing=QwenSparseKVAddressing::FixedLogicalView;
        fullBase.kv.logical_view={.logical_row_origin=4095,.physical_row_capacity=f.capacity};
        if(transform)fullBase=selectedKeyTransform(fullBase,10000000.0f,4095,f.capacity);
        FP16Run33 full(fullBase,f.stream);full.record();full.capture.launch();
        const auto expected=full.output.download(f.stream);const auto expectedParts=full.partials.download(f.stream);
        checkSharedReadOwners33(full,counts);ASSERT_FALSE(::testing::Test::HasFailure());
        const auto checkView=[&](int first,int heads){
            SCOPED_TRACE(::testing::Message()<<"pre_rope="<<transform<<" first="<<first<<" heads="<<heads);
            const int firstKV=first/12,kvCount=(first+heads-1)/12-firstKV+1;
            Inputs input(f,first,heads,firstKV,kvCount);auto a=input.base();
            a.kv.addressing=QwenSparseKVAddressing::FixedLogicalView;
            a.kv.logical_view={.logical_row_origin=4095,.physical_row_capacity=f.capacity};
            if(transform)a=selectedKeyTransform(a,10000000.0f,4095,f.capacity);
            FP16Run33 compact(a,f.stream);compact.record();compact.capture.launch();
            const auto actual=compact.output.download(f.stream);const auto parts=compact.partials.download(f.stream);
            for(int row=0;row<3;++row)for(int head=0;head<heads;++head){
                ASSERT_EQ(std::memcmp(actual.data()+(std::size_t(row)*heads+head)*256,expected.data()+(std::size_t(row)*24+first+head)*256,256*sizeof(float)),0);
                for(int part=0;part<33;++part){
                    const auto i=(std::size_t(row)*heads+head)*33+part,j=(std::size_t(row)*24+first+head)*33+part;
                    ASSERT_EQ(std::memcmp(&parts[i].maximum,&expectedParts[j].maximum,sizeof(float)),0);
                    ASSERT_EQ(std::memcmp(&parts[i].denominator,&expectedParts[j].denominator,sizeof(float)),0);
                    ASSERT_EQ(std::memcmp(parts[i].weighted,expectedParts[j].weighted,sizeof(parts[i].weighted)),0);
                    if(part*64>=counts[row])ASSERT_EQ(std::memcmp(parts[i].weighted,compact.initial_partials[i].weighted,sizeof(parts[i].weighted)),0);
                }
            }
            checkSharedReadOwners33(compact,counts);ASSERT_FALSE(::testing::Test::HasFailure());
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
        FP16Run33 quiet(fullBase,f.stream);quiet.args.observations=nullptr;quiet.record();quiet.capture.launch();
        const auto quietOutput=quiet.output.download(f.stream);const auto quietParts=quiet.partials.download(f.stream);
        ASSERT_EQ(std::memcmp(quietOutput.data(),expected.data(),expected.size()*sizeof(float)),0);
        ASSERT_EQ(std::memcmp(quietParts.data(),expectedParts.data(),expectedParts.size()*sizeof(QwenSparseAttentionPartial)),0);
        const auto keys=fullInput.keys.download(f.stream),values=fullInput.values.download(f.stream);
        ASSERT_EQ(std::memcmp(keys.data(),fullInput.host_keys.data(),keys.size()*sizeof(std::uint16_t)),0);
        ASSERT_EQ(std::memcmp(values.data(),fullInput.host_values.data(),values.size()*sizeof(std::uint16_t)),0);
    }
    ASSERT_EQ(f.selected.download(f.stream),selectedBefore);ASSERT_EQ(f.count.download(f.stream),countsBefore);
}
namespace fp16_attention
{
/**
 * @brief Scalar uniform-score oracle for one independent64-token virtual tile.
 * @param count Positive original selected-token count, at most2051.
 * @param value Exactly representable finite native half, including signed zero.
 * @return Original half-PV rounding followed by ascending33-part FP32 folding.
 * @details Q0 makes exponentials1. Host scalar half conversions independently
 * encode every parity update; first-tile zero-scale FMA and even-plus-odd order
 * stay explicit. Empty slots still perform the declared zero-scale merge FMA.
 */
inline float constantHalfOracle33(int count,float value){
    require(count>0&&count<=QwenSparseSelected&&std::isfinite(value)&&detail::decodeHalf(detail::encodeHalf(value))==value,"Invalid33 constant oracle input");
    float numerator=0.0f,denominator=0.0f;
    for(int part=0;part<33;++part){
        const int live=std::max(0,std::min(64,count-part*64));float weighted=0.0f,partDenominator=0.0f;
        if(live){float parity[2]{};
            for(int side=0;side<2;++side)for(int token=side;token<live;token+=2)
                parity[side]=detail::decodeHalf(detail::encodeHalf(std::fmaf(0x1p-7f,value,parity[side])));
            const float even=std::fmaf(0.0f,0.0f,parity[0]*1.0f),odd=std::fmaf(0.0f,0.0f,parity[1]*1.0f);
            weighted=even+odd;partDenominator=std::fmaf(0.0f,0.0f,float(live)*0x1p-7f);
        }
        const float scale=live?1.0f:0.0f;
        numerator=std::fmaf(weighted,scale,numerator);denominator=std::fmaf(partDenominator,scale,denominator);
    }
    return numerator/denominator;
}
}
/** @test Uniform/exreme native values stay finite and preserve zero sign in33 arithmetic. */
TEST(ROCmQwenSparseAttention,FP16Virtual33UniformExtremeFiniteAndSignedZero)
{
    using namespace fp16_attention;
    Fixture f(3,2050,4096,QwenSparseScalar::FP16);Inputs input(f,0,6,0,1);
    std::fill(input.host_query.begin(),input.host_query.end(),0.0f);input.query.upload(input.host_query.data(),f.stream);check(hipStreamSynchronize(f.stream));
    for(bool transform:{false,true}){
        SCOPED_TRACE(transform?"pre_rotary":"post_rotary");
        auto a=input.base();a.kv.addressing=QwenSparseKVAddressing::FixedLogicalView;
        a.kv.logical_view={.logical_row_origin=4095,.physical_row_capacity=f.capacity};
        if(transform)a=selectedKeyTransform(a,10000000.0f,4095,f.capacity);
        FP16Run33 run(a,f.stream);run.record();
        for(float value:{0.0f,-0.0f,0x1p-24f,-0x1p-24f,1.0f,32768.0f,65504.0f,-65504.0f}){
            SCOPED_TRACE(detail::floatBits(value));
            std::fill(input.host_values.begin(),input.host_values.end(),detail::encodeHalf(value));input.values.upload(input.host_values.data(),f.stream);check(hipStreamSynchronize(f.stream));
            for(int prefix:{0,30,31,32,62,63,64,510,511,512,2047,2048,2049,2050}){
                const auto counts=publishReplayPosition(f,prefix,3,65536);run.capture.launch();
                const auto result=run.output.download(f.stream);const auto parts=run.partials.download(f.stream);
                for(int row=0;row<3;++row){const float expected=constantHalfOracle33(counts[row],value);ASSERT_TRUE(std::isfinite(expected));
                    for(int head=0;head<6;++head)for(int d=0;d<256;++d){const float actual=result[(std::size_t(row)*6+head)*256+d];ASSERT_TRUE(std::isfinite(actual));
                        if(expected==0.0f)ASSERT_EQ(detail::floatBits(actual),detail::floatBits(expected));
                        else ASSERT_NEAR(actual,expected,std::max(1.0e-6f,std::fabs(expected)*2.0e-6f));
                    }
                }
                for(const auto& part:parts)if(part.denominator>0)for(float x:part.weighted){
                    ASSERT_TRUE(std::isfinite(x));
                    // Tiny signed native values round to half zero in every PV
                    // update; the declared initial +0 FMA makes tile totals +0.
                    if(std::fabs(value)<=0x1p-24f)ASSERT_EQ(detail::floatBits(x),detail::floatBits(0.0f));
                }
                ASSERT_EQ(std::memcmp(result.data()+3*1536,run.poison.data()+3*1536,16*sizeof(float)),0);
            }
        }
    }
}
/**
 * @test Every33-slot merge step preserves its declared signed-zero edge.
 * @details Post-RoPE Q1 and K=-6.4375 give the exact final-tile score-103;
 * all earlier tiles have score0 and positive-zero values. With IEEE FP32
 * subnormals, exp(-103) rounds to the minimum positive subnormal. The final
 * negative numerator therefore underflows to negative zero. Empty slots after
 * that tile must still fold +0 and canonicalize the numerator to positive zero.
 * The2051-token case ends at virtual tile32 and must retain negative zero.
 * A real eight-stream512-token control authenticates the sensitive construction
 * on the actual native math path; no mirrored device kernel supplies an oracle.
 * M16 repeats the same explicitly authored causal selection for each row, then
 * compares every row's output/partials to its independently captured serial call.
 */
TEST(ROCmQwenSparseAttention,FP16Virtual33EmptyTailPreservesSignedZero)
{
    using namespace fp16_attention;
    for(int rows:{1,16})for(int context:{512,1024,2048,2051}){
        SCOPED_TRACE(::testing::Message()<<"M="<<rows<<" selected="<<context);
        Fixture f(rows,context-1,context+rows-1,QwenSparseScalar::FP16);Inputs input(f,0,6,0,1);
        std::fill(input.host_query.begin(),input.host_query.end(),1.0f);
        std::fill(input.host_keys.begin(),input.host_keys.end(),detail::encodeHalf(0.0f));
        std::fill(input.host_values.begin(),input.host_values.end(),detail::encodeHalf(0.0f));
        const int lastPart=(context-1)/64,lastBegin=lastPart*64,lastCount=context-lastBegin;
        for(int token=lastBegin;token<context;++token)for(int d=0;d<256;++d){
            input.host_keys[std::size_t(token)*256+d]=detail::encodeHalf(-6.4375f);
            input.host_values[std::size_t(token)*256+d]=detail::encodeHalf(-1.0f);
        }
        // A fixed causal prefix subset keeps the witness identical for every
        // row while row0 tests identity addressing and later rows test sparse.
        std::vector<std::int32_t> selected(f.selected.size,-77),counts(rows,context);
        for(int row=0;row<rows;++row)for(int token=0;token<context;++token)
            selected[std::size_t(row)*QwenSparseSelected+token]=token;
        f.replay={context-1,rows,0};f.state.upload(&f.replay,f.stream);
        input.query.upload(input.host_query.data(),f.stream);input.keys.upload(input.host_keys.data(),f.stream);input.values.upload(input.host_values.data(),f.stream);
        f.selected.upload(selected.data(),f.stream);f.count.upload(counts.data(),f.stream);check(hipStreamSynchronize(f.stream));
        auto args=input.base();ASSERT_EQ(args.kv.key_policy.encoding,attention::AttentionKeyCacheEncoding::PostRotary);
        if(rows==1&&context==512){
            FP16Run sensitive(args,f.stream);sensitive.record();sensitive.capture.launch();
            const auto control=sensitive.output.download(f.stream);
            for(std::size_t i=0;i<1536;++i)ASSERT_EQ(detail::floatBits(control[i]),0x80000000u)
                <<"Native eight-slot witness must produce -0; otherwise the assumed exp/subnormal behavior is not admitted";
        }
        FP16Run33 grouped(args,f.stream);grouped.record();grouped.capture.launch();
        const auto result=grouped.output.download(f.stream);const auto partials=grouped.partials.download(f.stream);
        const std::uint32_t expected=context==2051?0x80000000u:0u;
        for(int row=0;row<rows;++row){
            for(int head=0;head<6;++head){
                for(int d=0;d<256;++d)ASSERT_EQ(detail::floatBits(result[(std::size_t(row)*6+head)*256+d]),expected);
                for(int part=0;part<=lastPart;++part){const auto& p=partials[(std::size_t(row)*6+head)*33+part];
                    ASSERT_EQ(p.maximum,part==lastPart?-103.0f:0.0f);
                    ASSERT_EQ(p.denominator,part==lastPart?float(lastCount)*0x1p-7f:0.5f);
                    for(float x:p.weighted){
                        if(part==lastPart)ASSERT_EQ(x,-float(lastCount)*0x1p-7f);
                        else ASSERT_EQ(detail::floatBits(x),0u);
                    }
                }
            }
            if(rows==16){
                auto serialArgs=args;serialArgs.row_capacity=1;
                serialArgs.query=input.query.data+std::size_t(row)*1536;
                serialArgs.selected=f.selected.data+std::size_t(row)*QwenSparseSelected;
                serialArgs.selected_count=f.count.data+row;
                FP16Run33 serial(serialArgs,f.stream);serial.record();
                f.replay={context-1+row,1,0};f.state.upload(&f.replay,f.stream);serial.capture.launch();
                const auto one=serial.output.download(f.stream);const auto oneParts=serial.partials.download(f.stream);
                ASSERT_EQ(std::memcmp(one.data(),result.data()+std::size_t(row)*1536,1536*sizeof(float)),0);
                ASSERT_EQ(std::memcmp(oneParts.data(),partials.data()+std::size_t(row)*6*33,6*33*sizeof(QwenSparseAttentionPartial)),0);
            }
        }
        ASSERT_EQ(std::memcmp(result.data()+std::size_t(rows)*1536,grouped.poison.data()+std::size_t(rows)*1536,16*sizeof(float)),0);
    }
}

/**
 * @test Nonzero raw K1 scores have an independent analytic oracle in both key encodings.
 * @details Q is one only in coordinates128..255, and each authored logical
 * token has those key coordinates all +1/8 or -1/8, so its score is exactly
 * +1 or -1 after the original1/16 query scaling. K0 is zero, including its
 * transformed64-coordinate prefix. Values are the corresponding +1 or -1.
 * Class counts alone determine maxima, FP32 denominators and ideal softmax;
 * no host QK/PV implementation reproduces the native loop or reduction tree.
 *
 * Native probability rounding and each32-term half-PV chain contribute at
 * most gamma33=33u/(1-33u), u=2^-11, relative to the positive denominator.
 * This alternating-class construction keeps nonzero half intermediates normal.
 * The0.02 absolute numerator/denominator bound exceeds gamma33 plus FP32
 * exp/fold/division error; it still separates an omitted K1 score from the
 * expected signed logistic response. Maxima remain exact and denominators
 * use a separate tight FP32 bound. Full partial/output bytes must additionally
 * match serial capture; dirty inactive/empty records and guards stay unchanged.
 * Original row0 uses identity selection, later rows use causal sparse subsets;
 * wrapped physical placement is authored independently of the kernel mapper.
 * Setup uploads join before capture/replay; terminal downloads retire work
 * before borrowed host storage or retained graph owners leave scope.
 */
TEST(ROCmQwenSparseAttention,FP16Phase33NonzeroSecondKeyPanelOracle)
{
    using namespace fp16_attention;
    constexpr int rows=3,capacity=4096,origin=capacity-1;
    const double low=std::exp(-2.0);
    for(bool transform:{false,true}){
        SCOPED_TRACE(transform?"pre_rotary":"post_rotary");
        Fixture f(rows,2050,capacity,QwenSparseScalar::FP16);Inputs input(f,0,6,0,1);
        std::fill(input.host_query.begin(),input.host_query.end(),0.0f);
        for(int row=0;row<rows;++row)for(int head=0;head<6;++head)for(int d=128;d<256;++d)
            input.host_query[(std::size_t(row)*6+head)*256+d]=1.0f;
        std::fill(input.host_keys.begin(),input.host_keys.end(),detail::encodeHalf(0.0f));
        for(int logical=0;logical<capacity;++logical){
            const int physical=logical==0?capacity-1:logical-1;
            const bool positive=logical%4<2;
            for(int d=128;d<256;++d)input.host_keys[std::size_t(physical)*256+d]=detail::encodeHalf(positive?0.125f:-0.125f);
            for(int d=0;d<256;++d)input.host_values[std::size_t(physical)*256+d]=detail::encodeHalf(positive?1.0f:-1.0f);
        }
        input.query.upload(input.host_query.data(),f.stream);input.keys.upload(input.host_keys.data(),f.stream);input.values.upload(input.host_values.data(),f.stream);
        check(hipStreamSynchronize(f.stream));
        auto args=input.base();args.kv.addressing=QwenSparseKVAddressing::FixedLogicalView;
        args.kv.logical_view={.logical_row_origin=origin,.physical_row_capacity=capacity};args.kv.row_parameters=nullptr;
        if(transform)args=selectedKeyTransform(args,10000000.0f,origin,capacity);
        FP16Run33 grouped(args,f.stream);grouped.record();
        auto prior=grouped.initial_partials;auto previousOutput=grouped.poison;
        for(const auto [selectedCount,live]:{std::pair{2051,rows},std::pair{31,rows},std::pair{32,rows},std::pair{33,rows},
            std::pair{63,rows},std::pair{64,rows},std::pair{65,rows},std::pair{1,1},std::pair{3,0},std::pair{1,rows},std::pair{3,rows},std::pair{2051,rows}}){
            SCOPED_TRACE(::testing::Message()<<"selected="<<selectedCount<<" live="<<live);
            std::vector<std::int32_t> selected(f.selected.size,-77),counts(rows,selectedCount);
            for(int row=0;row<rows;++row)for(int token=0;token<selectedCount;++token)
                selected[std::size_t(row)*QwenSparseSelected+token]=row+token;
            f.replay={selectedCount-1,live,65536};f.state.upload(&f.replay,f.stream);
            f.selected.upload(selected.data(),f.stream);f.count.upload(counts.data(),f.stream);check(hipStreamSynchronize(f.stream));
            grouped.capture.launch();const auto result=grouped.output.download(f.stream);
            const auto parts=grouped.partials.download(f.stream);const auto observations=grouped.observations.download(f.stream);
            for(int row=0;row<rows;++row){
                const auto first=std::size_t(row)*6*33;
                if(row>=live){
                    ASSERT_EQ(std::memcmp(parts.data()+first,prior.data()+first,6*33*sizeof(QwenSparseAttentionPartial)),0);
                    ASSERT_EQ(std::memcmp(result.data()+std::size_t(row)*1536,previousOutput.data()+std::size_t(row)*1536,1536*sizeof(float)),0);
                    for(int head=0;head<6;++head){const auto& o=observations[row*6+head];ASSERT_EQ(o.key_elements,0u);ASSERT_EQ(o.value_elements,0u);ASSERT_EQ(o.selected_tokens,0u);}
                    continue;
                }
                int totalPositive=0;
                for(int token=0;token<selectedCount;++token)totalPositive+=(row+token)%4<2;
                const int totalNegative=selectedCount-totalPositive;
                const double expected=totalPositive==0?-1.0:(totalPositive-totalNegative*low)/(totalPositive+totalNegative*low);
                for(int head=0;head<6;++head){
                    for(int d=0;d<256;++d)ASSERT_NEAR(result[(std::size_t(row)*6+head)*256+d],expected,0.02);
                    for(int part=0;part<33;++part){
                        const auto i=(std::size_t(row)*6+head)*33+part;const auto& p=parts[i];
                        const int begin=part*64,end=std::min(selectedCount,begin+64);
                        if(begin>=selectedCount){
                            ASSERT_EQ(p.maximum,-INFINITY);ASSERT_EQ(p.denominator,0.0f);ASSERT_EQ(p.key_elements,0u);ASSERT_EQ(p.value_elements,0u);
                            ASSERT_EQ(std::memcmp(p.weighted,prior[i].weighted,sizeof(p.weighted)),0);continue;
                        }
                        int positive=0;for(int token=begin;token<end;++token)positive+=(row+token)%4<2;
                        const int negative=end-begin-positive;
                        const double denominator=(positive?positive+negative*low:double(negative))/128.0;
                        const double numerator=(positive?positive-negative*low:-double(negative))/128.0;
                        ASSERT_EQ(p.maximum,positive?1.0f:-1.0f);
                        ASSERT_NEAR(p.denominator,denominator,2.0e-6*denominator+1.0e-9);
                        for(float value:p.weighted)ASSERT_NEAR(value,numerator,0.02*denominator);
                    }
                }
                auto serialArgs=args;serialArgs.row_capacity=1;serialArgs.query=input.query.data+std::size_t(row)*1536;
                serialArgs.selected=f.selected.data+std::size_t(row)*QwenSparseSelected;serialArgs.selected_count=f.count.data+row;
                FP16Run33 serial(serialArgs,f.stream);serial.record();
                std::copy_n(prior.data()+first,6*33,serial.initial_partials.data());serial.partials.upload(serial.initial_partials.data(),f.stream);
                f.replay={selectedCount-1+row,1,65536};f.state.upload(&f.replay,f.stream);serial.capture.launch();
                const auto one=serial.output.download(f.stream);const auto oneParts=serial.partials.download(f.stream);
                ASSERT_EQ(std::memcmp(one.data(),result.data()+std::size_t(row)*1536,1536*sizeof(float)),0);
                ASSERT_EQ(std::memcmp(oneParts.data(),parts.data()+first,6*33*sizeof(QwenSparseAttentionPartial)),0);
                ASSERT_EQ(std::memcmp(one.data()+1536,serial.poison.data()+1536,16*sizeof(float)),0);
            }
            ASSERT_EQ(std::memcmp(result.data()+std::size_t(rows)*1536,grouped.poison.data()+std::size_t(rows)*1536,16*sizeof(float)),0);
            prior=parts;previousOutput=result;
        }
    }
}

/**
 * @brief Exercise real merge counter carries without executing its producer.
 * @test Prove uint64 observation carries, all-zero nonowner observations, null
 * publication and inactive-row reset using the actual captured FP16 merge.
 * @details Clone only the already authenticated production merge node. Authored
 * uint32 records cross32-bit boundaries; independent uint64 sums are the oracle.
 * The original capture and all argument storage outlive the clone's terminal join.
 */
TEST(ROCmQwenSparseAttention,FP16Phase33MergeWideCounters)
{
    using namespace fp16_attention;
    /** @brief Test-owned merge-only graph borrowing one authenticated pair's args. */
    struct MergeOnly final {
        hipStream_t stream;hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;
        /** @brief Borrow the non-default stream until terminal graph retirement. */
        explicit MergeOnly(hipStream_t s):stream(s){require(stream!=nullptr,"Wide-counter merge requires an exact non-default stream");}
        /** @brief Join before graph and borrowed parameter storage can retire. */
        ~MergeOnly(){if(hipStreamSynchronize(stream)!=hipSuccess)std::terminate();
            if(exec && hipGraphExecDestroy(exec)!=hipSuccess)std::terminate();
            if(graph && hipGraphDestroy(graph)!=hipSuccess)std::terminate();}
        /**
         * @brief Copy exactly one native production merge kernel and its bindings.
         * @param run Retained complete production pair, recorded but never launched.
         * @details Native graph creation copies kernel arguments; the original run
         * still outlives this clone. No new production launcher or arithmetic exists.
         */
        void record(FP16Run33& run){
            std::size_t n=0;check(hipGraphGetNodes(run.capture.graph,nullptr,&n));require(n==2,"Wide-counter source must retain one complete pair");
            std::vector<hipGraphNode_t> nodes(n);check(hipGraphGetNodes(run.capture.graph,nodes.data(),&n));
            hipKernelNodeParams params{};int found=0;
            for(auto node:nodes){const auto metadata=inspectObservationKernel(node,stream);
                if(metadata.name.find("reduceAttentionPartitionsILNS_10TensorTypeE2ELi33EE")!=std::string::npos){
                    check(hipGraphKernelNodeGetParams(node,&params));++found;}}
            require(found==1 && params.gridDim.x==2 && params.gridDim.y==1 && params.gridDim.z==1 &&
                params.blockDim.x==256 && params.blockDim.y==1 && params.blockDim.z==1,"Wide-counter merge geometry differs");
            check(hipGraphCreate(&graph,0));hipGraphNode_t merge=nullptr;
            check(hipGraphAddKernelNode(&merge,graph,nullptr,0,&params));
            check(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        }
        /** @brief Replay only the cloned merge after ordered fixture uploads. */
        void launch(){check(hipGraphLaunch(exec,stream));}
    };
    Fixture f(2,2050,4096,QwenSparseScalar::FP16);Inputs inputs(f,0,1,0,1);
    auto args=inputs.base();
    for(bool observed:{true,false}){
        SCOPED_TRACE(observed);FP16Run33 run(args,f.stream);
        if(!observed)run.args.observations=nullptr;
        run.record();MergeOnly merge(f.stream);merge.record(run);
        for(int fold:{33,8,16,32,33})for(bool zeroCounters:{false,true}){
            SCOPED_TRACE(::testing::Message()<<"fold="<<fold<<" zero_counters="<<zeroCounters);
            const int selected=(fold-1)*64+1;
            std::vector<std::int32_t> counts(2,selected);f.count.upload(counts.data(),f.stream);
            f.replay={2050,1,0};f.state.upload(&f.replay,f.stream);
            auto authored=run.initial_partials;std::uint64_t keys=0,values=0;
            for(std::size_t i=0;i<authored.size();++i){auto& part=authored[i];
                part.maximum=0.0f;part.denominator=1.0f;
                std::fill(std::begin(part.weighted),std::end(part.weighted),0.0f);
                part.key_elements=(i%3==0)?UINT32_MAX:std::uint32_t(0x80000000u+i);
                part.value_elements=(i%2==0)?UINT32_MAX:std::uint32_t(i+17);
                if(zeroCounters)part.key_elements=part.value_elements=0;
                if(i<std::size_t(fold)){keys+=std::uint64_t(part.key_elements);values+=std::uint64_t(part.value_elements);}}
            if(zeroCounters){ASSERT_EQ(keys,0u);ASSERT_EQ(values,0u);}
            else{ASSERT_GT(keys,std::uint64_t(UINT32_MAX));ASSERT_GT(values,std::uint64_t(UINT32_MAX));}
            std::vector<QwenSparseAttentionObservation> dirty(2);
            for(auto& o:dirty){o.key_elements=0x123456789abcdef0ull;o.value_elements=0xfedcba9876543210ull;o.selected_tokens=999;}
            run.partials.upload(authored.data(),f.stream);run.output.upload(run.poison.data(),f.stream);
            run.observations.upload(dirty.data(),f.stream);merge.launch();
            const auto actual=run.observations.download(f.stream);const auto output=run.output.download(f.stream);
            const auto after=run.partials.download(f.stream);
            ASSERT_EQ(std::memcmp(after.data(),authored.data(),authored.size()*sizeof(authored[0])),0);
            for(int d=0;d<256;++d)ASSERT_EQ(output[d],0.0f);
            ASSERT_EQ(std::memcmp(output.data()+256,run.poison.data()+256,(output.size()-256)*sizeof(float)),0);
            if(observed){ASSERT_EQ(actual[0].key_elements,keys);ASSERT_EQ(actual[0].value_elements,values);
                ASSERT_EQ(actual[0].selected_tokens,std::uint32_t(selected));
                ASSERT_EQ(actual[1].key_elements,0u);ASSERT_EQ(actual[1].value_elements,0u);ASSERT_EQ(actual[1].selected_tokens,0u);}
            else for(int row=0;row<2;++row){ASSERT_EQ(actual[row].key_elements,dirty[row].key_elements);
                ASSERT_EQ(actual[row].value_elements,dirty[row].value_elements);ASSERT_EQ(actual[row].selected_tokens,dirty[row].selected_tokens);}
        }
    }
}

/**
 * @test Disabled native-event comparison of explicit8/33 arithmetic on identical inputs.
 * @details Uses existing bounded selectors and source-owned native captures only;
 * this component does not claim cross-program equality or full-model qualification.
 * Optional QFN_FP16_PROFILE_SIDE=control|candidate isolates one complete pair
 * with repeats1, five warmups and exact sixth occurrence. Paired-only exports
 * retain candidate33 numerical coordinates and explicit mapping/policy metadata.
 */
TEST(ROCmQwenSparseAttention,DISABLED_FP16Virtual33AttentionEvents)
{
    using namespace fp16_attention;
    const auto integer=[](const char* name){const char* text=std::getenv(name);require(text&&*text,"Missing33 event selector");std::size_t used=0;int value=std::stoi(text,&used);require(used==std::strlen(text),"Invalid33 event selector");return value;};
    const char* modeText=std::getenv("QFN_FP16_KEY_ENCODING");require(modeText&&*modeText,"Explicit33 key encoding required");const std::string mode=modeText;
    require(mode=="post_rotary"||mode=="pre_rotary","Unknown33 key encoding");
    const int rows=integer("QFN_FP16_M"),context=integer("QFN_FP16_CONTEXT"),repeats=integer("QFN_FP16_REPEATS");
    const int position=integer("QFN_FP16_POSITION_ORIGIN"),ring=integer("QFN_FP16_RING_ORIGIN");
    const char* text=std::getenv("QFN_FP16_ROPE_THETA");require(text&&*text,"Explicit33 theta required");std::size_t used=0;const float theta=std::stof(text,&used);
    const char* sideText=std::getenv("QFN_FP16_PROFILE_SIDE");const std::string side=sideText?sideText:"paired";
    require(side=="paired"||side=="control"||side=="candidate","Unknown33 profile side");const bool paired=side=="paired";
    require(paired||!std::getenv("QFN_FP16_EXPORT_DIR"),"33 numerical export requires paired mode");
    require(used==std::strlen(text)&&std::isfinite(theta)&&theta>0&&rows>0&&rows<=16&&context>=rows&&context<=65536&&position>=0&&ring>=0&&ring<context&&
        (paired?(repeats>=2&&repeats<=256):repeats==1),"Invalid33 event/profile geometry/policy");
    Fixture f(rows,context-rows,context,QwenSparseScalar::FP16);Inputs input(f,0,6,0,1);auto a=input.base();
    a.kv.addressing=QwenSparseKVAddressing::FixedLogicalView;a.kv.logical_view={.logical_row_origin=ring,.physical_row_capacity=f.capacity};a.kv.row_parameters=nullptr;
    if(mode=="pre_rotary")a=selectedKeyTransform(a,theta,ring,f.capacity);
    FP16Run baseline(a,f.stream,repeats);FP16Run33 candidate(a,f.stream,repeats);baseline.record();candidate.record();publishReplayPosition(f,context-rows,rows,position);Events events(f.stream);
    // Recording authenticates both graphs without executing either program.
    // Isolated mode submits only the chosen complete producer/merge family.
    if(!paired){
        auto& chosen=side=="control"?baseline.capture:candidate.capture;
        for(int warm=0;warm<5;++warm)(void)events.sample(chosen);
        const float us=events.sample(chosen);const int parts=side=="control"?8:33;
        std::cout<<"QFN_FP16_PARTS33_SINGLE,"<<side<<",M="<<rows<<",context="<<context<<",Q=6,KV=1,parts="<<parts<<",virtual_parts="<<parts<<",physical_slots="<<parts<<",group_heads="<<(side=="control"?4:6)<<",producer_threads="<<(side=="control"?256:384)<<",merge_threads=256,key_encoding="<<mode
            <<",key_rotary_source="<<(mode=="pre_rotary"?"canonical_engine":"cached_post_rotary")<<",key_rotary_dim="<<(mode=="pre_rotary"?64:0)
            <<",addressing=fixed_logical_view,ring_origin="<<ring<<",ring_capacity="<<context<<",position_origin="<<position<<",theta_bits="<<(mode=="pre_rotary"?detail::floatBits(theta):0u)
            <<",repeats=1,warmups=5,occurrence=6,us="<<us<<",profiled_timings_used_as_score=false\n";
        const auto actual=side=="control"?baseline.output.download(f.stream):candidate.output.download(f.stream);
        const auto& poison=side=="control"?baseline.poison:candidate.poison;
        for(std::size_t i=0;i<std::size_t(rows)*1536;++i)ASSERT_TRUE(std::isfinite(actual[i]));
        ASSERT_EQ(std::memcmp(actual.data()+std::size_t(rows)*1536,poison.data()+std::size_t(rows)*1536,16*sizeof(float)),0);
        return;
    }
    for(int warm=0;warm<5;++warm){(void)events.sample(baseline.capture);(void)events.sample(candidate.capture);}
    for(int pair=0;pair<12;++pair){float old=0,newer=0;if(pair&1){newer=events.sample(candidate.capture);old=events.sample(baseline.capture);}else{old=events.sample(baseline.capture);newer=events.sample(candidate.capture);}
        std::cout<<"QFN_FP16_PARTS33_PAIR,"<<pair<<",M="<<rows<<",context="<<context<<",Q=6,KV=1,virtual_parts=33,physical_slots=33,candidate_group_heads=6,candidate_producer_threads=384,control_group_heads=4,control_producer_threads=256,merge_threads=256,key_encoding="<<mode
            <<",addressing=fixed_logical_view,ring_origin="<<ring<<",ring_capacity="<<context<<",position_origin="<<position<<",theta_bits="<<(mode=="pre_rotary"?detail::floatBits(theta):0u)
            <<",repeats="<<repeats<<",baseline8_us="<<old<<",candidate33_us="<<newer<<'\n';
    }
    const auto old=baseline.output.download(f.stream),actual=candidate.output.download(f.stream);
    for(std::size_t i=0;i<std::size_t(rows)*1536;++i){ASSERT_TRUE(std::isfinite(old[i]));ASSERT_TRUE(std::isfinite(actual[i]));}
    // Focused ordinary-input witness only: <=512 uses one tile per old stream.
    // It does not promise universal signed-zero equivalence between programs.
    if(context<=512)ASSERT_EQ(std::memcmp(old.data(),actual.data(),std::size_t(rows)*1536*sizeof(float)),0);
    ASSERT_EQ(std::memcmp(old.data()+std::size_t(rows)*1536,baseline.poison.data()+std::size_t(rows)*1536,16*sizeof(float)),0);
    ASSERT_EQ(std::memcmp(actual.data()+std::size_t(rows)*1536,candidate.poison.data()+std::size_t(rows)*1536,16*sizeof(float)),0);
    if(const char* destination=std::getenv("QFN_FP16_EXPORT_DIR")){
        const auto partials=candidate.partials.download(f.stream);std::vector<float> numerical;numerical.reserve(partials.size()*258);
        for(const auto& part:partials){numerical.push_back(part.maximum);numerical.push_back(part.denominator);numerical.insert(numerical.end(),std::begin(part.weighted),std::end(part.weighted));}
        const auto outputCount=std::size_t(rows)*6*256;
        require((outputCount+numerical.size())*sizeof(float)<=128ULL*1024*1024,"FP16 parts33 export exceeds128 MiB");
        const std::filesystem::path directory(destination);require(*destination&&!std::filesystem::exists(directory),"FP16 parts33 export requires a fresh directory");
        require(std::filesystem::create_directories(directory),"FP16 parts33 export directory creation failed");
        const auto write=[&](const char* name,const float* values,std::size_t count){std::ofstream stream(directory/name,std::ios::binary);stream.exceptions(std::ios::failbit|std::ios::badbit);stream.write(reinterpret_cast<const char*>(values),static_cast<std::streamsize>(count*sizeof(float)));stream.close();};
        write("output-fp32.bin",actual.data(),outputCount);write("partials-fp32.bin",numerical.data(),numerical.size());
        std::ofstream shape(directory/"shape.json");shape.exceptions(std::ios::failbit|std::ios::badbit);
        shape<<"{\"M\":"<<rows<<",\"context\":"<<context<<",\"Q\":6,\"KV\":1,\"partitions\":33,\"virtual_parts\":33,\"physical_slots\":33,\"group_heads\":6,\"partial_coordinates\":258,\"selection\":\"causal-spread-512groups\",\"key_encoding\":\""<<mode<<"\",\"key_rotary_source\":\""<<(mode=="pre_rotary"?"canonical_engine":"cached_post_rotary")<<"\",\"key_rotary_dim\":"<<(mode=="pre_rotary"?64:0)<<",\"theta_bits\":"<<(mode=="pre_rotary"?detail::floatBits(theta):0u)<<",\"position_origin\":"<<position<<",\"addressing\":\""<<"fixed_logical_view"<<"\",\"ring_origin\":"<<ring<<",\"ring_capacity\":"<<context<<"}\n";shape.close();
    }
}

}
