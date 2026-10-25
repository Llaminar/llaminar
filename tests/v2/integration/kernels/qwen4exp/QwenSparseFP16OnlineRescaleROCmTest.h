/**
 * @file QwenSparseFP16OnlineRescaleROCmTest.h
 * @brief Runtime evaluator and retained8/33 online-max arithmetic witnesses.
 *
 * Reuses existing input, stream, capture, partial and serial-row fixture owners.
 * The test-only native probe compares the actual original evaluator to the
 * candidate; host libm is not its oracle. Complete attention tests retain the
 * production two-node graph and all original selected/read/publication contracts.
 */
#pragma once
#include "../rocm/QwenSparseFP16OnlineRescaleProbe.h"
#include <bit>
#include <limits>
#include <type_traits>
namespace llaminar2::qwen4exp::ROCmQwenSparseAttention
{
/** @test Device-loaded signed zeros and all finite/nonfinite scale branches match original HIP math. */
TEST(ROCmQwenSparseAttention,FP16OnlineRescaleCompiledEvaluatorAndNonfiniteBytes)
{
    using namespace observation115;
    namespace trial=online_rescale_trial;
    static_assert(sizeof(trial::Input)==7*sizeof(float) && sizeof(trial::Output)==10*sizeof(float));
    static_assert(sizeof(trial::SideResult)==4*sizeof(float));
    static_assert(std::is_trivially_copyable_v<trial::Input> && std::is_trivially_copyable_v<trial::Output>);
    Fixture f(1,0,64,QwenSparseScalar::FP16);
    const float inf=std::numeric_limits<float>::infinity();
    const float tiny=std::numeric_limits<float>::denorm_min();
    const float nan=std::bit_cast<float>(std::uint32_t{0x7fc12345});
    const float snan=std::bit_cast<float>(std::uint32_t{0x7fa12345});
    const float negativeNan=std::bit_cast<float>(std::uint32_t{0xffc54321});
    const std::vector<std::pair<float,float>> maxima{
        {-2.f,-1.f},{-1.f,-2.f},{3.f,3.f},{0.f,-0.f},{-0.f,0.f},
        {tiny,0.f},{0.f,tiny},{1.f,std::bit_cast<float>(std::uint32_t{0x3f800001})},
        {0.f,-160.f},{-160.f,0.f},{-1e30f,1e30f},{1e30f,-1e30f},
        {-inf,-inf},{-inf,0.f},{-inf,inf},{0.f,-inf},{0.f,inf},
        {inf,-inf},{inf,0.f},{inf,inf},{nan,0.f},{0.f,nan},
        {nan,nan},{snan,0.f},{0.f,snan},{negativeNan,nan}};
    std::vector<trial::Input> input;
    for(auto pair:maxima) for(float denominator:{0.f,-0.f,1.f,nan}) for(float zero:{0.f,-0.f})
        input.push_back({pair.first,pair.second,denominator,zero,
            input.size()%4==0?inf:input.size()%4==1?nan:input.size()%4==2?tiny:-0.f,
            input.size()%3==0?inf:input.size()%3==1?0.25f:-0.f,
            input.size()%5==0?nan:0.125f});
    const auto count=input.size();ASSERT_LE(count,256u);
    input.push_back({12345.f,12345.f,12345.f,12345.f,12345.f,12345.f,12345.f});
    std::vector<trial::Output> poison(count+1);
    for(auto& item:poison)item={12345.f,12345.f,{{12345.f,12345.f,12345.f,12345.f},{12345.f,12345.f,12345.f,12345.f}}};
    Buffer<trial::Input> deviceInput(input.size());Buffer<trial::Output> deviceOutput(poison.size());
    Capture captured(f.stream);
    FixtureCopyRetirement retirement{&f.stream};
    const auto function=trial::evaluatorFunction();hipFuncAttributes resource{};
    ASSERT_NE(function,nullptr);check(hipFuncGetAttributes(&resource,function));
    ASSERT_EQ(resource.localSizeBytes,0u);ASSERT_GE(resource.maxThreadsPerBlock,512);
    std::cout<<"QSA_ONLINE_RESCALE_PROBE_NATIVE,raw_symbol="<<hipKernelNameRefByPtr(function,f.stream)
        <<",registers="<<resource.numRegs<<",static_lds="<<resource.sharedSizeBytes
        <<",private="<<resource.localSizeBytes<<",cases="<<count<<'\n';
    // Capture once into the existing owner, declared after both buffers.
    // Failed recording closes exactly once before any destructor can join.
    bool recording=false;
    try
    {
        check(hipStreamBeginCapture(f.stream,hipStreamCaptureModeThreadLocal));recording=true;
        trial::evaluate(deviceInput.data,deviceOutput.data,static_cast<int>(count),f.stream);
        const auto status=hipStreamEndCapture(f.stream,&captured.graph);recording=false;check(status);
    }
    catch(...)
    {
        if(recording) (void)hipStreamEndCapture(f.stream,&captured.graph);
        throw;
    }
    std::size_t nodes=0;check(hipGraphGetNodes(captured.graph,nullptr,&nodes));ASSERT_EQ(nodes,1u);
    hipGraphNode_t node=nullptr;check(hipGraphGetNodes(captured.graph,&node,&nodes));
    hipGraphNodeType type{};check(hipGraphNodeGetType(node,&type));ASSERT_EQ(type,hipGraphNodeTypeKernel);
    hipKernelNodeParams parameters{};check(hipGraphKernelNodeGetParams(node,&parameters));
    ASSERT_EQ(parameters.func,function);ASSERT_EQ(parameters.gridDim.x,1u);ASSERT_EQ(parameters.gridDim.y,1u);ASSERT_EQ(parameters.gridDim.z,1u);
    ASSERT_EQ(parameters.blockDim.x,512u);ASSERT_EQ(parameters.blockDim.y,1u);ASSERT_EQ(parameters.blockDim.z,1u);ASSERT_EQ(parameters.sharedMemBytes,0u);
    const auto read=[&]<class T>(int index)
    {
        std::array<std::byte,sizeof(T)> bytes{};
        require(parameters.kernelParams && parameters.kernelParams[index],"Missing probe argument");
        std::memcpy(bytes.data(),parameters.kernelParams[index],sizeof(T));return std::bit_cast<T>(bytes);
    };
    ASSERT_EQ(read.template operator()<const trial::Input*>(0),deviceInput.data);
    ASSERT_EQ(read.template operator()<trial::Output*>(1),deviceOutput.data);
    ASSERT_EQ(read.template operator()<int>(2),static_cast<int>(count));
    check(hipGraphInstantiate(&captured.exec,captured.graph,nullptr,nullptr,0));
    for(int dirty=0;dirty<3;++dirty)
    {
        deviceInput.upload(input.data(),f.stream);deviceOutput.upload(poison.data(),f.stream);
        check(hipStreamSynchronize(f.stream));
        captured.launch();
        const auto result=deviceOutput.download(f.stream);
        for(std::size_t i=0;i<count;++i)
        {
            SCOPED_TRACE(i);
            SCOPED_TRACE(dirty);
            // zero is an independent global-memory input; this cannot be a
            // source-level constant-folded expf(+0.f) surrogate.
            ASSERT_EQ(std::bit_cast<std::uint32_t>(result[i].zero_exponential),0x3f800000u);
            int field = 0;
            for(auto values:{std::pair{result[i].sides[0].previous,result[i].sides[1].previous},
                std::pair{result[i].sides[0].tile,result[i].sides[1].tile},
                std::pair{result[i].sides[0].total,result[i].sides[1].total},
                std::pair{result[i].sides[0].denominator,result[i].sides[1].denominator}})
            {
                SCOPED_TRACE(field++);
                EXPECT_EQ(std::bit_cast<std::uint32_t>(values.first),std::bit_cast<std::uint32_t>(values.second));
            }
        }
        ASSERT_EQ(std::memcmp(&result[count],&poison[count],sizeof(trial::Output)),0);
        const auto unchanged=deviceInput.download(f.stream);
        ASSERT_EQ(std::memcmp(unchanged.data(),input.data(),input.size()*sizeof(trial::Input)),0);
        // Different retained poison proves all authored output words are fresh.
        for(auto& item:poison)item={-817.25f,-817.25f,{{-817.25f,-817.25f,-817.25f,-817.25f},{-817.25f,-817.25f,-817.25f,-817.25f}}};
    }
    // EXPECT_EQ intentionally collects every field/case mismatch. Never print
    // a successful proof marker after any accumulated nonfatal assertion.
    ASSERT_FALSE(::testing::Test::HasFailure());
    std::cout<<"QSA_ONLINE_RESCALE_PROBE_BYTES,cases="<<count<<",replays=3,zero_signs=2,nonfinite_bits=exact,passed=1\n";
}
namespace fp16_attention
{
/**
 * @brief Directed score histories retain grouped/serial full-output/partial bytes.
 * @tparam Parts Existing production8 or33 arithmetic; no new dispatch program.
 * @details First/equal/rising/falling finite score histories and nonfinite V
 * payloads use identical original
 * inputs for both row organizations. Serial captures are retained once and seed
 * only their own previous partial payload, matching original empty-part semantics.
 */
template<int Parts> void proveOnlineRescaleAttention()
{
    static_assert(Parts==8 || Parts==33);
    using Run=std::conditional_t<Parts==33,FP16Run33,FP16Run>;
    for(int rows:{1,16}) for(bool transform:{false,true}) for(int pattern=0;pattern<4;++pattern)
    {
        SCOPED_TRACE(Parts);
        SCOPED_TRACE(rows);
        SCOPED_TRACE(transform);
        SCOPED_TRACE(pattern);
        constexpr int heads=5,width=heads*256;
        Fixture f(rows,2050,4096,QwenSparseScalar::FP16);Inputs input(f,1,heads,0,1);
        std::fill(input.host_query.begin(),input.host_query.end(),pattern==0?0.f:1.f);
        for(int token=0;token<f.capacity;++token) for(int d=0;d<256;++d)
        {
            float key=pattern==0?0.f:float(token/512)*(pattern==2?-0.125f:0.125f);
            input.host_keys[std::size_t(token)*256+d]=detail::encodeHalf(key);
            float value=d%2?-0.f:0.25f;
            // Nonfinite QK would violate the merge's finite-positive denominator
            // contract. Nonfinite V keeps scores/denominator admitted and tests
            // that no numerator multiplication/FMA or NaN edge disappeared.
            if(pattern==3 && token>=512 && token<576) value=std::numeric_limits<float>::infinity();
            if(pattern==3 && token>=1024 && token<1088) value=std::bit_cast<float>(std::uint32_t{0x7fc12345});
            input.host_values[std::size_t(token)*256+d]=detail::encodeHalf(value);
        }
        input.query.upload(input.host_query.data(),f.stream);input.keys.upload(input.host_keys.data(),f.stream);
        input.values.upload(input.host_values.data(),f.stream);check(hipStreamSynchronize(f.stream));
        auto base=input.base();base.kv.addressing=QwenSparseKVAddressing::FixedLogicalView;
        base.kv.logical_view={.logical_row_origin=0,.physical_row_capacity=f.capacity};base.kv.row_parameters=nullptr;
        if(transform)base=selectedKeyTransform(base,10000000.f,0,f.capacity);
        Run grouped(base,f.stream);grouped.record();
        std::vector<std::unique_ptr<Run>> serial;
        for(int row=0;row<rows;++row)
        {
            auto a=base;a.row_capacity=1;a.query=input.query.data+std::size_t(row)*width;
            a.selected=f.selected.data+std::size_t(row)*QwenSparseSelected;a.selected_count=f.count.data+row;
            serial.push_back(std::make_unique<Run>(a,f.stream));serial.back()->record();
        }
        auto prior=grouped.initial_partials;auto previous=grouped.poison;
        for(auto [prefix,live]:{std::pair{2050,rows},std::pair{0,rows},std::pair{0,0},
            std::pair{768,std::min(rows,3)},std::pair{2050,rows}})
        {
            publishReplayPosition(f,prefix,live,65536);grouped.capture.launch();
            const auto output=grouped.output.download(f.stream);const auto parts=grouped.partials.download(f.stream);
            for(int row=0;row<rows;++row)
            {
                const auto first=std::size_t(row)*heads*Parts;
                if(row>=live)
                {
                    ASSERT_EQ(std::memcmp(parts.data()+first,prior.data()+first,heads*Parts*sizeof(QwenSparseAttentionPartial)),0);
                    ASSERT_EQ(std::memcmp(output.data()+std::size_t(row)*width,previous.data()+std::size_t(row)*width,width*sizeof(float)),0);
                    continue;
                }
                auto& one=*serial[row];std::copy_n(prior.data()+first,heads*Parts,one.initial_partials.data());
                one.partials.upload(one.initial_partials.data(),f.stream);
                f.replay={prefix+row,1,65536};f.state.upload(&f.replay,f.stream);one.capture.launch();
                const auto single=one.output.download(f.stream);const auto singleParts=one.partials.download(f.stream);
                ASSERT_EQ(std::memcmp(single.data(),output.data()+std::size_t(row)*width,width*sizeof(float)),0);
                ASSERT_EQ(std::memcmp(singleParts.data(),parts.data()+first,heads*Parts*sizeof(QwenSparseAttentionPartial)),0);
                ASSERT_EQ(std::memcmp(single.data()+width,one.poison.data()+width,16*sizeof(float)),0);
            }
            ASSERT_EQ(std::memcmp(output.data()+std::size_t(rows)*width,grouped.poison.data()+std::size_t(rows)*width,16*sizeof(float)),0);
            prior=parts;previous=output;
        }
        const auto query=input.query.download(f.stream);
        const auto keys=input.keys.download(f.stream);
        const auto values=input.values.download(f.stream);
        ASSERT_EQ(std::memcmp(query.data(),input.host_query.data(),query.size()*sizeof(float)),0);
        ASSERT_EQ(keys,input.host_keys);ASSERT_EQ(values,input.host_values);
    }
}
}
/** @test Both phase programs retain adversarial score-history bytes through dirty row replay. */
TEST(ROCmQwenSparseAttention,FP16OnlineRescaleAdversarialGroupedSerialRetainedBytes)
{
    fp16_attention::proveOnlineRescaleAttention<8>();
    fp16_attention::proveOnlineRescaleAttention<33>();
}
}
