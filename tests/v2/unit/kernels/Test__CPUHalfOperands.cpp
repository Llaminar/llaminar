/**
 * @file Test__CPUHalfOperands.cpp
 * @brief Prove CPU physical-half linear and native BF16 router contracts.
 *
 * FP32 buffers use the public shape allocation followed by typed host writes;
 * no unsupported std::vector constructor or storage ownership is assumed.
 * Integer dyadic oracles make every product and intermediate sum exactly FP32,
 * while non-half source inputs discriminate actual A rounding from FP32-A math.
 * Complete M sweeps retain inactive-capacity poison and compare ordinary,
 * verifier and fixed-column paths. OpenMP policy is scoped; no model/device work
 * or timing threshold belongs in these correctness regressions.
 */
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "kernels/common/ProjectionRowWork.h"
#include "kernels/cpu/moe/CPUMoEKernel.h"
#if defined(__aarch64__)
#include "utils/ARMCPUFeatures.h"
#endif
#include "../../utils/CPURouterKernelTestAdmission.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <memory>
#include <fstream>
#include <set>
#include <sched.h>
#include <vector>
#include <utility>

namespace llaminar2::test {
namespace {
using Engine=gemm::FloatingPointGemmKernel;
using Mode=FloatingActivationOperandPrecision;
/** @brief Whether the RESOLVED route is the FMLAL leaf: the compiled
 *        envelope must declare FP16FML and the process ISA resolution must
 *        have selected a native NEON level (an explicit Scalar route in an
 *        I8MM binary legitimately executes none). */
bool fmlalRouteResolved()
{
    if (!fmlalLeafCompiledInEnvelope()) return false;
#if defined(__aarch64__)
    const auto level = activeISALevel();
    return level == ISALevel::NEON || level == ISALevel::NEONDotProd;
#else
    return false;
#endif
}
/** @brief Enable the FMLAL route probe for one contract, then restore off.
 *        A no-op in envelopes without the leaf: the probe does not exist
 *        there and the resolved-route expectations are zero. */
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_FML)
class FmlalProbe final {
public:
    FmlalProbe() { fmlalLeafProbeEnabled().store(true, std::memory_order_relaxed); }
    ~FmlalProbe() { fmlalLeafProbeEnabled().store(false, std::memory_order_relaxed); }
};
#else
class FmlalProbe final { public: };
#endif
/** @brief Restore physical worker selection and dynamic-team policy. */
class Workers final {
    int count_=omp_get_max_threads(),dynamic_=omp_get_dynamic();
public:
    explicit Workers(int count){omp_set_dynamic(0);omp_set_num_threads(count);}
    ~Workers(){omp_set_num_threads(count_);omp_set_dynamic(dynamic_);}
};
/** @brief Count unique affinity-admitted package/core pairs, never SMT siblings. */
int physicalWorkerCount(){
    cpu_set_t allowed;CPU_ZERO(&allowed);
    if(sched_getaffinity(0,sizeof(allowed),&allowed)!=0)return 0;
    std::set<std::pair<int,int>> cores;
    for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&allowed)){
        const std::string path="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
        int package=-1,core=-1;
        std::ifstream package_file(path+"physical_package_id"),core_file(path+"core_id");
        if(!(package_file>>package)||!(core_file>>core))return 0;
        cores.emplace(package,core);
    }
    return int(cores.size());
}
/** @brief Independently round n/4096 in the [1,2) binade, ties to even. */
int roundedNumerator(int n){const int q=n/4,r=n%4;return q+(r>2||(r==2&&(q&1)));}
/** @brief Construct exact representable half weights; oracle uses original ints. */
std::vector<uint16_t> weightWords(int n,int k){
    std::vector<uint16_t> words(size_t(n)*k);
    for(int j=0;j<n;++j)for(int c=0;c<k;++c)
        words[size_t(j)*k+c]=fp32_to_fp16(float(((j+c)&1?-1:1)*(1+(j*3+c*5)%17))/64.f);
    return words;
}
/** @brief Exhaustively check one geometry without one assertion per element. */
void proveLinear(int m,int k,int workers){
    SCOPED_TRACE("M="+std::to_string(m)+" K="+std::to_string(k)+" workers="+std::to_string(workers));
    Workers team(workers);constexpr int N=9;
    auto words=weightWords(N,k);
    FP16Tensor weight({N,size_t(k)},words);
    Engine engine(&weight,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE);
    std::vector<float> source(size_t(m+2)*k),initial(size_t(m+3)*N,-777.f),expected(size_t(m)*N);
    std::vector<uint16_t> half(source.size());
    for(int row=0;row<m+2;++row)for(int c=0;c<k;++c){
        const int n=4096+3*((row*7+c*5)%32);
        source[size_t(row)*k+c]=float(n)/4096.f;
        half[size_t(row)*k+c]=uint16_t(0x3c00+roundedNumerator(n)-1024);
    }
    for(int row=0;row<m;++row)for(int j=0;j<N;++j){
        int64_t sum=0;
        for(int c=0;c<k;++c)sum+=int64_t(roundedNumerator(4096+3*((row*7+c*5)%32)))*(((j+c)&1?-1:1)*(1+(j*3+c*5)%17));
        expected[size_t(row)*N+j]=std::ldexp(float(sum),-16);
    }
    FP32Tensor input({size_t(m+2),size_t(k)}),output({size_t(m+3),N});
    std::copy(source.begin(),source.end(),input.mutable_data());
    std::copy(initial.begin(),initial.end(),output.mutable_data());
    FP16Tensor half_input({size_t(m+2),size_t(k)},half);
    const auto unchanged=source;
    for(int lane=0;lane<4;++lane){
        std::copy(initial.begin(),initial.end(),output.mutable_data());
        if(lane==0)ASSERT_TRUE(engine.multiply_tensor(&input,&output,m,N,k));
        if(lane==1){ITensorGemm::VerifierKernelModeScope scope;ASSERT_TRUE(engine.multiply_fused_verifier_rows_decode_equivalent(&input,{{&engine,&output,N,nullptr,"test"}},m,k));}
        if(lane==2){auto scope=engine.beginOutputPartitionEquivalenceScope(N,7);ASSERT_NE(scope,nullptr);ASSERT_TRUE(engine.multiply_fused_tensor(&input,{{&engine,&output,N,nullptr,"test"}},m,k));}
        if(lane==3)ASSERT_TRUE(engine.multiply_tensor(&half_input,&output,m,N,k));
        EXPECT_EQ(std::memcmp(output.data(),expected.data(),expected.size()*sizeof(float)),0)<<"lane="<<lane;
        EXPECT_EQ(std::memcmp(output.data()+size_t(m)*N,initial.data()+size_t(m)*N,3*N*sizeof(float)),0);
    }
    EXPECT_EQ(std::memcmp(input.data(),unchanged.data(),source.size()*sizeof(float)),0);
    EXPECT_EQ(engine.packedWeightBytes(),0u);
}
/** @brief Compare complete typed routing while proving original-hidden Q8 bytes. */
void proveRouter(int m,int workers){
    Workers team(workers);constexpr int K=65,E=17,Top=4;
    std::vector<float> source(size_t(m)*K),rounded(source.size()),weights(E*K);
    std::vector<uint16_t> words(weights.size());
    for(int row=0;row<m;++row)for(int k=0;k<K;++k){source[row*K+k]=float(4099+4*row)/4096.f;rounded[row*K+k]=float(4100+4*row)/4096.f;}
    for(int e=0;e<E;++e)for(int k=0;k<K;++k){weights[e*K+k]=float((e+1)*(k%7+1))/1024.f;words[e*K+k]=fp32_to_fp16(weights[e*K+k]);}
    FP32Tensor hidden({size_t(m),K}),ids({size_t(m),Top}),coeff({size_t(m),Top});
    std::copy(source.begin(),source.end(),hidden.mutable_data());
    FP16Tensor gate({E,K},words);CPUMoEKernel typed,reference,original,q8;MoERoutingResult actual,want,old;
    for(auto *kernel:{&typed,&reference,&original,&q8})
        admitCPURouterKernelForTest(*kernel,m,K);
    ASSERT_TRUE(typed.routeWithTensors(&hidden,&gate,m,K,E,Top,true,&ids,&coeff,actual));
    ASSERT_TRUE(reference.route(rounded.data(),weights.data(),m,K,E,Top,true,want));
    ASSERT_TRUE(original.route(source.data(),weights.data(),m,K,E,Top,true,old));
    EXPECT_EQ(actual.expert_indices,want.expert_indices);
    ASSERT_EQ(actual.router_logits.size(),want.router_logits.size());
    EXPECT_EQ(std::memcmp(actual.router_logits.data(),want.router_logits.data(),want.router_logits.size()*sizeof(float)),0);
    EXPECT_EQ(std::memcmp(actual.expert_weights.data(),want.expert_weights.data(),want.expert_weights.size()*sizeof(float)),0);
    EXPECT_NE(std::memcmp(actual.router_logits.data(),old.router_logits.data(),old.router_logits.size()*sizeof(float)),0);
    ASSERT_TRUE(q8.publishTransportedRouterQ8Hidden(hidden.data(),m,K));
    const auto *a=typed.publishedRouterQ8Hidden(hidden.data(),m,K),*b=q8.publishedRouterQ8Hidden(hidden.data(),m,K);
    ASSERT_NE(a,nullptr);ASSERT_NE(b,nullptr);
    EXPECT_EQ(std::memcmp(a,b,size_t(m)*((K+31)/32)*sizeof(Q8_1Block)),0);
    EXPECT_EQ(std::memcmp(hidden.data(),source.data(),source.size()*sizeof(float)),0);
    for(int row=0;row<m;++row)for(int slot=0;slot<Top;++slot)EXPECT_EQ(actual.expert_indices[row*Top+slot],E-1-slot);
}

/**
 * @brief Authenticate native BF16 gates, serial row bytes and live publication.
 * @param m Live rows within a retained bank with two poisoned inactive rows.
 * @param k Hidden width, including tail and production MTP dimensions.
 * @param workers Physical CPU worker budget for both serial and grouped calls.
 *
 * A double dot/softmax checks the numerical result independently. Real M=1
 * routing owns the byte oracle; non-half FP32 inputs distinguish this native
 * BF16 path from the explicitly rounded FP16 operand policy.
 */
void proveBF16Router(int m, int k, int workers) {
    SCOPED_TRACE("BF16 M="+std::to_string(m)+" K="+std::to_string(k)+" workers="+std::to_string(workers));
    Workers team(workers);
    constexpr int E=17, Top=4;
    std::vector<uint16_t> words(size_t(E)*k);
    std::vector<float> weights(words.size()), source(size_t(m+2)*k);
    for(int e=0;e<E;++e)for(int c=0;c<k;++c) {
        const float value=float((e+1)*(c%7+1))/2048.f;
        words[e*k+c]=uint16_t(std::bit_cast<uint32_t>(value)>>16);
        weights[e*k+c]=std::bit_cast<float>(uint32_t(words[e*k+c])<<16);
    }
    for(int row=0;row<m+2;++row)for(int c=0;c<k;++c)
        source[row*k+c]=float(4099+4*row+2*(c%3))/4096.f;
    BF16Tensor gate({E,size_t(k)},words);
    FP32Tensor hidden({size_t(m+2),size_t(k)}), ids({size_t(m+2),Top}), coeff({size_t(m+2),Top});
    std::copy(source.begin(),source.end(),hidden.mutable_data());
    CPUMoEKernel grouped, serial, q8;
    admitCPURouterKernelForTest(grouped,m,k);
    admitCPURouterKernelForTest(serial,1,k);
    admitCPURouterKernelForTest(q8,m,k);
    MoERoutingResult actual, expected;
    // Reuse the large bank after shrinking the live geometry to one row.
    for(int live : {m,1}) {
        std::fill_n(ids.mutable_data(),(m+2)*Top,-777.f);
        std::fill_n(coeff.mutable_data(),(m+2)*Top,-777.f);
        ASSERT_TRUE(grouped.routeWithTensors(&hidden,&gate,live,k,E,Top,true,&ids,&coeff,actual));
        for(int row=0;row<live;++row) {
            FP32Tensor one({1,size_t(k)}), one_ids({1,Top}), one_weights({1,Top});
            std::copy_n(source.data()+row*k,k,one.mutable_data());
            ASSERT_TRUE(serial.routeWithTensors(&one,&gate,1,k,E,Top,true,&one_ids,&one_weights,expected));
            EXPECT_EQ(std::memcmp(ids.data()+row*Top,one_ids.data(),Top*sizeof(float)),0);
            EXPECT_EQ(std::memcmp(coeff.data()+row*Top,one_weights.data(),Top*sizeof(float)),0);
            EXPECT_EQ(std::memcmp(actual.router_logits.data()+row*E,expected.router_logits.data(),E*sizeof(float)),0);
            std::array<double,E> logits{}, probabilities{};
            for(int e=0;e<E;++e)for(int c=0;c<k;++c)
                logits[e]+=double(source[row*k+c])*weights[e*k+c];
            double denominator=0;
            for(int e=0;e<E;++e)denominator+=(probabilities[e]=std::exp(logits[e]-logits[E-1]));
            double selected=0;
            for(int slot=0;slot<Top;++slot)selected+=probabilities[E-1-slot];
            for(int e=0;e<E;++e)EXPECT_NEAR(actual.router_logits[row*E+e],probabilities[e]/denominator,2e-6);
            for(int slot=0;slot<Top;++slot) {
                EXPECT_EQ(ids.data()[row*Top+slot],E-1-slot);
                EXPECT_NEAR(coeff.data()[row*Top+slot],probabilities[E-1-slot]/selected,2e-6);
            }
        }
        for(int i=live*Top;i<(m+2)*Top;++i) {
            EXPECT_EQ(ids.data()[i],-777.f);
            EXPECT_EQ(coeff.data()[i],-777.f);
        }
        ASSERT_TRUE(q8.publishTransportedRouterQ8Hidden(hidden.data(),live,k));
        const auto *a=grouped.publishedRouterQ8Hidden(hidden.data(),live,k),*b=q8.publishedRouterQ8Hidden(hidden.data(),live,k);
        ASSERT_NE(a,nullptr);ASSERT_NE(b,nullptr);
        EXPECT_EQ(std::memcmp(a,b,size_t(live)*((k+31)/32)*sizeof(Q8_1Block)),0);
    }
    EXPECT_EQ(std::memcmp(hidden.data(),source.data(),source.size()*sizeof(float)),0);
    EXPECT_EQ(std::memcmp(gate.typed_data(),words.data(),words.size()*sizeof(uint16_t)),0);
}
}

TEST(CPUHalfOperands, ExactDyadicAllRowsAndReductionBoundaries){
    for(int workers:{1,3})for(int m:{1,2,3,4,5,8,17,63,64,65,448,512})
        for(int k:{1,7,8,15,16,31,32,65,257})proveLinear(m,k,workers);
}
TEST(CPUHalfOperands, HalfRNEBoundaryAndFP32Output){
    constexpr std::array<std::pair<float,uint32_t>,8> cases{{
        {0.f,0u},{-0.f,0x80000000u},{0x1p-25f,0u},{0x1.8p-24f,0x34000000u},
        {0x1.4p-23f,0x34000000u},{1.f+0x1p-11f,0x3f800000u},
        {1.f+3*0x1p-11f,0x3f804000u},{65520.f,0x7f800000u}}};
    for(auto [value,bits]:cases)EXPECT_EQ(std::bit_cast<uint32_t>(gemm::roundFloatingOperandHalfRNE(value)),bits);
    std::vector<float> tiny(65,0.f);
    const std::array<float,6> inputs{{0x1p-25f,0x1.8p-24f,0x1.4p-23f,1.f+0x1p-11f,1.f+3*0x1p-11f,-0x1.8p-24f}};
    const std::array<float,6> rounded{{0.f,0x1p-23f,0x1p-23f,1.f,1.f+0x1p-9f,-0x1p-23f}};
    std::copy(inputs.begin(),inputs.end(),tiny.begin());
    std::vector<uint16_t> selectors(6*65,0);
    for(int i=0;i<6;++i)selectors[i*65+i]=0x3c00;
    FP16Tensor select({6,65},selectors);FP32Tensor tiny_a({1,65}),tiny_c({1,6});
    std::copy(tiny.begin(),tiny.end(),tiny_a.mutable_data());
    Engine tiny_engine(&select,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE);
    ASSERT_TRUE(tiny_engine.multiply_tensor(&tiny_a,&tiny_c,1,6,65));
    for(int i=0;i<6;++i)EXPECT_EQ(tiny_c.data()[i],rounded[i]);
    FP16Tensor w({1,65},std::vector<uint16_t>(65,0x6400));
    FP32Tensor a({1,65}),c({1,1});
    std::fill_n(a.mutable_data(),65,1024.f);
    Engine e(&w,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE);
    ASSERT_TRUE(e.multiply_tensor(&a,&c,1,1,65));EXPECT_EQ(c.data()[0],68157440.f);
}
/** @test Row offsets and epilogues operate only on declared FP32 output bytes. */
TEST(CPUHalfOperands, OffsetBiasAlphaBetaAndSerialRowsAgree){
    constexpr int M=3,N=2,K=65;
    FP16Tensor weight({N,K},std::vector<uint16_t>(N*K,0x3c00));
    Engine engine(&weight,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE);
    FP32Tensor a({M+2,K});
    std::fill_n(a.mutable_data(),(M+2)*K,1.f+3*0x1p-12f);
    FP32Tensor c({M+2,N});
    std::fill_n(c.mutable_data(),(M+2)*N,4.f);
    FP32Tensor bias({N});
    std::fill_n(bias.mutable_data(),N,0.5f);
    ASSERT_TRUE(engine.multiply_tensor(&a,&c,M,N,K,true,0.5f,0.25f,&bias,nullptr,-1,nullptr,1));
    const float expected=65.f*(1.f+0x1p-10f)*0.5f+1.5f;
    for(int i=0;i<M*N;++i)EXPECT_EQ(c.data()[i],expected);
    for(int i=M*N;i<(M+2)*N;++i)EXPECT_EQ(c.data()[i],4.f);
}
TEST(CPUHalfOperands, ImmutableIdentityAndMixedBundlesRejectBeforeWrite){
    auto w=std::make_shared<FP16Tensor>(std::vector<size_t>{9,65},weightWords(9,65));
    auto source=std::make_shared<Engine>(w,Engine::NumericalPolicy::BackendNative,CPUWeightStoragePlacement::local(),Mode::FP16RNE);
    Engine alias{std::shared_ptr<const Engine>(source)},native(w.get());
    EXPECT_EQ(alias.operandPrecision(),Mode::FP16RNE);EXPECT_EQ(alias.floatingActivationOperandPrecision(),Mode::FP16RNE);
    EXPECT_EQ(source->packedWeightBytes(),9u*65u*2u);EXPECT_EQ(alias.packedWeightBytes(),0u);
    FP32Tensor a({3,65}),c({3,9}),d({3,9});
    std::fill_n(a.mutable_data(),3*65,1.f);
    std::fill_n(c.mutable_data(),27,-77.f);std::fill_n(d.mutable_data(),27,-77.f);
    for(auto *seed:{&alias,&native}){
        EXPECT_FALSE(seed->multiply_fused_tensor(&a,{{&alias,&c,9,nullptr,"half"},{&native,&d,9,nullptr,"native"}},3,65));
        EXPECT_FALSE(seed->multiply_fused_verifier_rows_decode_equivalent(&a,{{&native,&d,9,nullptr,"native"},{&alias,&c,9,nullptr,"half"}},3,65));
        for(int i=0;i<27;++i){EXPECT_EQ(c.data()[i],-77.f);EXPECT_EQ(d.data()[i],-77.f);}
    }
    FP32Tensor wrong({9,65});
    EXPECT_THROW((Engine(&wrong,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE)),std::invalid_argument);
    EXPECT_THROW((Engine(w.get(),Engine::NumericalPolicy::BackendNative,Mode::FP32)),std::invalid_argument);
    EXPECT_FALSE(alias.multiply_tensor(&a,&c,3,9,65,false));
    EXPECT_FALSE(alias.multiply_tensor(&a,&c,4,9,65));
    EXPECT_EQ(native.operandPrecision(),Mode::Native);
}
TEST(CPUHalfOperands, TypedRouterRoundsBothOperandsPreservesOriginalQ8){
    for(int workers:{1,2,3,7})for(int m:{1,3,8,17})proveRouter(m,workers);
}
TEST(CPUHalfOperands, NativeBF16RouterSerialRowsAndRetainedBank){
    for(int workers:{1,3})for(int k:{1,7,65,128,257}) {
        for(int m=1;m<=16;++m)proveBF16Router(m,k,workers);
        for(int m:{31,63})proveBF16Router(m,k,workers);
    }
    proveBF16Router(3,2048,3);
}
/** @test K tails of 0..255 around the 256-lane stripe and non-multiple widths. */
TEST(CPUHalfOperands, KTailBoundariesAllRowsMatchSerial){
    FmlalProbe probe;
    const long long expect_per_prove=fmlalRouteResolved()?4:0;
    for(int workers:{1,3})for(int m:{1,2,3,4})
        for(int k:{255,256,257,300,511,512,513,767}){
            const long long before=fmlalLeafInvocations();
            proveLinear(m,k,workers);
            // Four execution lanes over nine columns: four admitted live
            // FMLAL dispatches per proveLinear on the resolved route, zero
            // otherwise (an explicit Scalar route in an I8MM binary runs the
            // scalar schedule and counts nothing).
            EXPECT_EQ(fmlalLeafInvocations()-before,expect_per_prove)<<"k="<<k<<" m="<<m;
        }
    // Repeated runs of one boundary geometry stay byte-identical.
    proveLinear(2,257,3);
    proveLinear(2,257,3);
}
/** @brief Test-owned exact binary16 widening and fixed-256-lane tree over chosen half words. */
float canonicalHalfTree(const std::vector<uint16_t> &a_words,const std::vector<uint16_t> &b_words){
    constexpr int lanes=256;
    auto widen=[](uint16_t word){
        const uint32_t sign=uint32_t(word & 0x8000u)<<16;
        const uint32_t exponent=(word >> 10) & 0x1fu, mantissa=word & 0x3ffu;
        uint32_t bits;
        if(exponent==0){
            if(mantissa==0)bits=sign;
            else{
                // Half subnormals normalize into binary32 normals:
                // value = mantissa * 2^-24 = normalized * 2^(length-25).
                const int leading=std::countl_zero(mantissa);
                const int length=32-leading;
                const uint32_t normalized=mantissa << (10-length);
                const int binary32_exponent=102+length;
                bits=sign | (uint32_t(binary32_exponent)<<23) | ((normalized & 0x1ffu)<<14);
            }
        }
        else if(exponent==31)bits=sign | 0x7f800000u | (mantissa<<13);
        else bits=sign | ((exponent-15u+127u)<<23) | (mantissa<<13);
        float out;std::memcpy(&out,&bits,sizeof(out));return out;
    };
    std::array<float,lanes> partials{};
    for(size_t i=0;i<a_words.size();++i){
        const size_t lane=i & size_t(lanes-1);
        partials[lane]=std::fma(widen(a_words[i]),widen(b_words[i]),partials[lane]);
    }
    for(int stride=lanes/2;stride>0;stride>>=1)
        for(int lane=0;lane<stride;++lane)
            partials[size_t(lane)]=partials[size_t(lane)]+partials[size_t(lane+stride)];
    return partials[0];
}
/** @test Adversarial half operands, RNE ties and overflow-to-Inf: leaf bytes equal the owned tree. */
TEST(CPUHalfOperands, FmlalSpecialOperandsMatchOwnedTree){
    // Isolated specials: K=256 gives every tree lane exactly one product, so
    // one special per run isolates its class (finite baseline elsewhere).
    // The oracle is the test-owned fixed-256-lane tree with test-owned exact
    // widening; bytes must match, NaN payloads included.
    const std::array<uint16_t,12> special_words{{
        0x0000,0x8000,0x0001,0x03FF,0x7C00,0xFC00,0x7E00,0x7C01,0x7BFF,0x0400,0x8001,0x3C00}};
    constexpr int K=256;
    FmlalProbe probe;
    for(int repetition:{1,2}){
        // Weight-side specials: A is exact 1.0 in both storage forms.
        for(uint16_t special:special_words){
            std::vector<uint16_t> words(size_t(K),0x3C00),half(size_t(K),0x3C00);
            words[size_t(repetition-1)]=special;
            std::vector<float> source(size_t(K),1.f);
            FP16Tensor weight({size_t(1),size_t(K)},words);
            Engine engine(&weight,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE);
            FP32Tensor input({size_t(1),size_t(K)}),output({size_t(1),size_t(1)});
            std::copy(source.begin(),source.end(),input.mutable_data());
            FP16Tensor half_input({size_t(1),size_t(K)},half);
            std::vector<uint16_t> a_words(size_t(K),0x3C00),w_words=words;
            const float expected=canonicalHalfTree(a_words,w_words);
            const long long before=fmlalLeafInvocations();
            std::fill_n(output.mutable_data(),1,-777.f);
            ASSERT_TRUE(engine.multiply_tensor(&input,&output,1,1,K))<<"W="<<special;
            EXPECT_EQ(fmlalLeafInvocations()-before,fmlalRouteResolved()?1:0)<<"W="<<special;
            EXPECT_EQ(std::memcmp(output.data(),&expected,sizeof(float)),0)<<"fp32A W="<<special;
            std::fill_n(output.mutable_data(),1,-777.f);
            ASSERT_TRUE(engine.multiply_tensor(&half_input,&output,1,1,K))<<"W="<<special;
            EXPECT_EQ(std::memcmp(output.data(),&expected,sizeof(float)),0)<<"halfA W="<<special;
        }
        // A-side specials through the HalfStorage path, and FP32-A RNE
        // ties/overflow through the conversion contract.
        for(uint16_t special:special_words){
            std::vector<uint16_t> words(size_t(K),0x3C00),half(size_t(K),0x3C00);
            half[size_t(repetition-1)]=special;
            std::vector<float> source(size_t(K),1.f);
            FP16Tensor weight({size_t(1),size_t(K)},words);
            Engine engine(&weight,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE);
            FP32Tensor input({size_t(1),size_t(K)}),output({size_t(1),size_t(1)});
            std::copy(source.begin(),source.end(),input.mutable_data());
            FP16Tensor half_input({size_t(1),size_t(K)},half);
            std::vector<uint16_t> a_words=half,w_words(size_t(K),0x3C00);
            const float expected=canonicalHalfTree(a_words,w_words);
            std::fill_n(output.mutable_data(),1,-777.f);
            ASSERT_TRUE(engine.multiply_tensor(&half_input,&output,1,1,K))<<"A="<<special;
            EXPECT_EQ(std::memcmp(output.data(),&expected,sizeof(float)),0)<<"halfA A="<<special;
        }
        // The smallest subnormal, certified independently of the finite
        // background: every product is exactly zero except one lane carrying
        // 1.0 * 2^-24, so the tree returns that contribution alone — a case
        // the 255-ones background cannot round away.
        {
            std::vector<uint16_t> words(size_t(K),0x0000),half(size_t(K),0x3C00);
            words[size_t(repetition-1)]=0x0001;
            std::vector<float> minsub_source(size_t(K),1.f);
            FP16Tensor weight({size_t(1),size_t(K)},words);
            Engine engine(&weight,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE);
            FP32Tensor input({size_t(1),size_t(K)}),output({size_t(1),size_t(1)});
            std::copy(minsub_source.begin(),minsub_source.end(),input.mutable_data());
            FP16Tensor half_input({size_t(1),size_t(K)},half);
            std::vector<uint16_t> a_words(size_t(K),0x3C00),w_words=words;
            const float expected=canonicalHalfTree(a_words,w_words);
            EXPECT_EQ(std::bit_cast<uint32_t>(expected),0x33800000u);
            std::fill_n(output.mutable_data(),1,-777.f);
            ASSERT_TRUE(engine.multiply_tensor(&input,&output,1,1,K));
            EXPECT_EQ(std::memcmp(output.data(),&expected,sizeof(float)),0)<<"minsub W";
            std::fill_n(output.mutable_data(),1,-777.f);
            ASSERT_TRUE(engine.multiply_tensor(&half_input,&output,1,1,K));
            EXPECT_EQ(std::memcmp(output.data(),&expected,sizeof(float)),0)<<"minsub halfA";
        }
        // FP32-A conversion discriminators: RNE ties and the overflow-to-Inf
        // boundary, isolated per run (the hand table is the one
        // HalfRNEBoundaryAndFP32Output pins, with its corrected ties).
        const std::array<std::pair<float,uint16_t>,6> a_cases{{
            {0x1.8p-24f,0x0002},{0x1.4p-23f,0x0002},{1.f+0x1p-11f,0x3C00},
            {1.f+3*0x1p-11f,0x3C02},{65520.f,0x7C00},{65519.f,0x7BFF}}};
        for(auto [value,word]:a_cases){
            std::vector<uint16_t> words(size_t(K),0x3C00);
            words[size_t(repetition-1)]=word;
            std::vector<float> source(size_t(K),0.f);
            source[size_t(repetition-1)]=value;
            FP16Tensor weight({size_t(1),size_t(K)},words);
            Engine engine(&weight,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE);
            FP32Tensor input({size_t(1),size_t(K)}),output({size_t(1),size_t(1)});
            std::copy(source.begin(),source.end(),input.mutable_data());
            std::vector<uint16_t> a_words(size_t(K),0x0000),w_words=words;
            a_words[size_t(repetition-1)]=word;
            const float expected=canonicalHalfTree(a_words,w_words);
            std::fill_n(output.mutable_data(),1,-777.f);
            ASSERT_TRUE(engine.multiply_tensor(&input,&output,1,1,K))<<"A="<<word;
            EXPECT_EQ(std::memcmp(output.data(),&expected,sizeof(float)),0)<<"fp32A A="<<word;
        }
    }
}
/** @test The mixed-policy boundary (F,F is not FMLAL) on the route the
 *        GPU-aligned witness named: GPUAlignedExpert with FP32 A and FP16
 *        weights keeps unconverted FP32 A through the K=16 stripe end to end.
 *        (FP16RNE engines are a different contract: their dispatch enters
 *        executeHalfOperands unconditionally and the HalfRNE schedule rounds
 *        A to binary16 at every k, including full blocks — that behavior is
 *        pre-existing at the base and is the reviewer's contract question,
 *        not this regression.) */
TEST(CPUHalfOperands, MixedPolicyKeepsUnroundedFP32AtK16){
    // The real mixed FP32-A/FP16-B contract: Mode::Native with the
    // GPUAlignedExpert policy, weights shaped [N,K] (get_n/get_k are
    // rows/cols) and transpose_B selecting the transposed read. Unconverted
    // FP32 A survives the K=16 stripe end to end.
    FP16Tensor weight({size_t(1),size_t(16)},std::vector<uint16_t>(16,0x3C00));
    Engine engine(&weight,Engine::NumericalPolicy::GPUAlignedExpert,Mode::Native);
    FP32Tensor input({size_t(1),size_t(16)}),output({size_t(1),size_t(1)});
    std::vector<float> source(16,0.f);
    source[0]=1.f+0x1p-12f;
    std::copy(source.begin(),source.end(),input.mutable_data());
    ASSERT_TRUE(engine.multiply_tensor(&input,&output,1,1,16,true));
    EXPECT_EQ(output.data()[0],1.f+0x1p-12f);
}
TEST(CPUHalfOperands, FmlalLeafSelectionMatchesEnvelope){
    // The leaf is compiled exactly in the envelope that declares FP16FML; an
    // envelope without it keeps the scalar schedule and never selects the
    // leaf (the dispatch carries no NEON case without the guard).
#if !defined(__aarch64__)
    // No AArch64 target: the FMLAL leaf cannot be compiled at all, so the
    // route probe stays zero and no FMLAL schedule exists on this target.
    EXPECT_FALSE(fmlalLeafCompiledInEnvelope());
    EXPECT_EQ(fmlalLeafInvocations(), 0ull);
#else
    EXPECT_EQ(fmlalLeafCompiledInEnvelope(),
        (llaminar2::cpu::ARMCPUCapabilities::compiledEnvelopeFeatures() &
         uint32_t(llaminar2::cpu::ARMCPUFeature::FP16FML))!=0);
    // A compiled leaf requires the executing CPU to advertise FEAT_FHM; the
    // dispatch admission fails fatally before any leaf instruction otherwise.
    if(fmlalLeafCompiledInEnvelope())
        EXPECT_TRUE(llaminar2::cpu::ARMCPUCapabilities::native().supports(
            llaminar2::cpu::ARMCPUFeature::FP16FML));
    // The selection probe proves the FMLAL path — never a silent scalar
    // fallback — actually dispatched: one route probe per multiply. The
    // route probe is evidence-only, so this contract enables it for the
    // multiply.
    {
        FmlalProbe probe;
        FP16Tensor weight({size_t(2),size_t(65)},weightWords(2,65));
        Engine probe_engine(&weight,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE);
        FP32Tensor probe_a({size_t(1),size_t(65)}),probe_c({size_t(1),size_t(2)});
        std::fill_n(probe_a.mutable_data(),65,1.f);
        const long long before=fmlalLeafInvocations();
        ASSERT_TRUE(probe_engine.multiply_tensor(&probe_a,&probe_c,1,2,65));
        EXPECT_EQ(fmlalLeafInvocations()-before,fmlalRouteResolved()?1:0);
    }
    // Device-free admission semantics: an envelope declaring FP16 without the
    // FML extension admits on an FHM-less machine; the FMLAL envelope does
    // not — FP16 arithmetic alone never authorizes the leaf.
    const uint32_t observed_fp16_no_fhm = uint32_t(llaminar2::cpu::ARMCPUFeature::ASIMD) |
        uint32_t(llaminar2::cpu::ARMCPUFeature::FP16Arithmetic);
    EXPECT_NO_THROW(llaminar2::cpu::ARMCPUCapabilities::requireEnvelope(
        observed_fp16_no_fhm, observed_fp16_no_fhm));
    EXPECT_THROW(llaminar2::cpu::ARMCPUCapabilities::requireEnvelope(
        observed_fp16_no_fhm, observed_fp16_no_fhm |
            uint32_t(llaminar2::cpu::ARMCPUFeature::FP16FML)),
        std::runtime_error);
    // FPCR profile contract (F3): focused negative controls over synthetic
    // control-words — only RNE with FZ, FZ16 and AHP off admits the leaf's
    // ambient conversion; each wrong bit (or rounding mode) is rejected.
    EXPECT_TRUE(fmlalFpcrProfileAdmitted(0x0ull));
    EXPECT_FALSE(fmlalFpcrProfileAdmitted(0x1ull << 19));  // FZ16
    EXPECT_FALSE(fmlalFpcrProfileAdmitted(0x1ull << 26));  // AHP
    EXPECT_FALSE(fmlalFpcrProfileAdmitted(0x1ull << 24));  // FZ
    EXPECT_FALSE(fmlalFpcrProfileAdmitted(0x1ull << 22));  // RMode=01
    EXPECT_FALSE(fmlalFpcrProfileAdmitted(0x2ull << 22));  // RMode=10
    EXPECT_FALSE(fmlalFpcrProfileAdmitted(0x3ull << 22));  // RMode=11
#endif
}
/** @test The route probe discriminates on/off, zero-N, invalid pointers,
 *        zero rows and positive live work — resolved-route semantics only. */
TEST(CPUHalfOperands, FmlalRouteProbeDiscrimination){
    // The route probe is evidence-only: enabled, it counts one admitted live
    // FMLAL dispatch on the resolved route; a scalar or leaf-less route
    // counts zero. The no-op probe and the zero-count interface make this a
    // real control on every build, never a skipped body. Public-rejection
    // checks and primitive-count controls are asserted separately.
    FP16Tensor weight({size_t(1),size_t(65)},weightWords(1,65));
    FP32Tensor a65({size_t(1),size_t(65)}),c1({size_t(1),size_t(1)});
    std::fill_n(a65.mutable_data(),65,1.f);
    // (1) Probe off (the default): a valid FP16RNE multiply counts nothing.
    {
        Engine engine(&weight,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE);
        const long long before=fmlalLeafInvocations();
        ASSERT_TRUE(engine.multiply_tensor(&a65,&c1,1,1,65));
        EXPECT_EQ(fmlalLeafInvocations()-before,0)<<"probe-off";
    }
    // (2-4) Probe enabled: the resolved route decides, and the public
    // rejections never reach the primitive.
    long long live_count=0;
    {
        FmlalProbe probe;
        Engine engine(&weight,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE);
        // (2) Positive live work: exactly one admitted dispatch on the
        // resolved FMLAL route; zero on any other resolved route.
        const long long before=fmlalLeafInvocations();
        ASSERT_TRUE(engine.multiply_tensor(&a65,&c1,1,1,65));
        live_count=fmlalLeafInvocations()-before;
        EXPECT_EQ(live_count,fmlalRouteResolved()?1:0)<<"positive-live";
        // (3) Zero N: the public admission rejects before the primitive;
        // nothing counts.
        const long long before_n=fmlalLeafInvocations();
        EXPECT_FALSE(engine.multiply_tensor(&a65,&c1,1,0,65));
        EXPECT_EQ(fmlalLeafInvocations()-before_n,0)<<"zero-N";
        // (4) Null activation: the public admission rejects; nothing counts.
        const long long before_null=fmlalLeafInvocations();
        EXPECT_FALSE(engine.multiply_tensor(nullptr,&c1,1,1,65));
        EXPECT_EQ(fmlalLeafInvocations()-before_null,0)<<"null-A";
    }
    // (5) Primitive-boundary negatives at the production dispatch
    // (llaminar2::gemm::run_gpu_aligned_expert_fp32_matmul<FP16,HalfRNE=true>):
    // the namespace-level boundary permits zero M/N and reaches
    // pre-validation with null pointers, so these controls pin the counter
    // where the historical false-count lived. The probe stays enabled for
    // the whole group.
    {
        // The primitive controls instantiate the INTENDED FMLAL domain
        // (<FP16, HalfRNE=true>): the default <FP16,false,false> domain is
        // correctly excluded from FMLAL by production, so its calls would
        // never reach the route the counter names. The enable scope covers
        // exactly this group.
        FmlalProbe probe;
        const std::vector<uint16_t> weight_words=weightWords(1,65);
        const auto *a_data=a65.data();
        const auto *b_data=weight_words.data();
        float *c_data=c1.mutable_data();
        // (P1) Zero M: zero row tiles -> no live span -> no count.
        const long long before_m=fmlalLeafInvocations();
        const bool zero_m=llaminar2::gemm::run_gpu_aligned_expert_fp32_matmul<
            llaminar2::TensorType::FP16, true>(a_data,b_data,c_data,0,1,65);
        ASSERT_TRUE(zero_m);
        EXPECT_EQ(fmlalLeafInvocations()-before_m,0)<<"prim zero-M";
        // (P2) Zero N: zero columns -> no FMLAL vector work -> no count.
        const long long before_n=fmlalLeafInvocations();
        const bool zero_n=llaminar2::gemm::run_gpu_aligned_expert_fp32_matmul<
            llaminar2::TensorType::FP16, true>(a_data,b_data,c_data,1,0,65);
        ASSERT_TRUE(zero_n);
        EXPECT_EQ(fmlalLeafInvocations()-before_n,0)<<"prim zero-N";
        // (P3) Null A: pre-validation returns false -> no count.
        const long long before_null=fmlalLeafInvocations();
        const bool null_a=llaminar2::gemm::run_gpu_aligned_expert_fp32_matmul<
            llaminar2::TensorType::FP16, true>(nullptr,b_data,c_data,1,1,65);
        EXPECT_FALSE(null_a);
        EXPECT_EQ(fmlalLeafInvocations()-before_null,0)<<"prim null-A";
        // (P4) Positive live work at the primitive: exactly one admitted
        // dispatch on the resolved FMLAL route.
        const long long before_live=fmlalLeafInvocations();
        const bool live_ok=llaminar2::gemm::run_gpu_aligned_expert_fp32_matmul<
            llaminar2::TensorType::FP16, true>(a_data,b_data,c_data,1,1,65);
        ASSERT_TRUE(live_ok);
        EXPECT_EQ(fmlalLeafInvocations()-before_live,fmlalRouteResolved()?1:0)<<"prim live";
        // (P5) Positive capacity, ALL-INACTIVE span, at the primitive: the
        // work scope publishes a zero-count range, so the isa schedule's
        // selection spans are empty — the live-span scan finds nothing, the
        // count stays put and the output is untouched. This reaches the
        // primitive's own scan, which the public work-scoped route returns
        // before.
        {
            std::int32_t live_rows=0;
            ProjectionRowWorkScope scope(ProjectionRowWork::fromPrefix(
                llaminar2::DeviceRowRange::deviceCounted(1,&live_rows)));
            FP32Tensor work_input({size_t(1),size_t(65)});
            std::fill_n(work_input.mutable_data(),65,1.f);
            std::fill_n(c_data,1,-777.f);
            const long long before_inactive=fmlalLeafInvocations();
            const bool inactive_ok=llaminar2::gemm::run_gpu_aligned_expert_fp32_matmul<
                llaminar2::TensorType::FP16>(a_data,b_data,c_data,1,1,65);
            ASSERT_TRUE(inactive_ok);
            EXPECT_EQ(fmlalLeafInvocations()-before_inactive,0)<<"prim all-inactive";
            EXPECT_EQ(c1.data()[0],-777.f)<<"prim all-inactive C";
        }
    }
    // (6) Probe restored off: the count holds still.
    {
        Engine engine(&weight,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE);
        const long long before=fmlalLeafInvocations();
        ASSERT_TRUE(engine.multiply_tensor(&a65,&c1,1,1,65));
        EXPECT_EQ(fmlalLeafInvocations()-before,0)<<"restored-off";
    }
    // (6) Positive capacity, all-inactive span: the work-scoped route
    // returns before any schedule runs; nothing counts even with the probe
    // enabled.
    {
        FmlalProbe probe;
        Engine engine(&weight,Engine::NumericalPolicy::BackendNative,Mode::FP16RNE);
        std::int32_t live_rows=0;
        ProjectionRowWorkScope scope(ProjectionRowWork::fromPrefix(
            DeviceRowRange::deviceCounted(1,&live_rows)));
        FP32Tensor input({size_t(1),size_t(65)});
        std::fill_n(input.mutable_data(),65,1.f);
        const long long before=fmlalLeafInvocations();
        ASSERT_TRUE(engine.multiply_tensor(&input,&c1,1,1,65));
        EXPECT_EQ(fmlalLeafInvocations()-before,0)<<"all-inactive";
    }
    // The positive-live control asserts only where the route resolved; a
    // non-native or leaf-less route legitimately counts zero and is never
    // claimed as FMLAL coverage.
    if(fmlalRouteResolved())
        EXPECT_EQ(live_count,1);
}
/** @test Explicit preflight scaling sweep; excluded from the fast Unit alias. */
TEST(CPUHalfOperands, EveryPositiveWorkerBudget){
    const int maximum=physicalWorkerCount();
    ASSERT_GT(maximum,0);
    for(int workers=1;workers<=maximum;++workers){proveLinear(3,65,workers);proveRouter(3,workers);proveBF16Router(3,65,workers);}
}
}
