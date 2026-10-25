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
#include "kernels/cpu/moe/CPUMoEKernel.h"
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
/** @test Explicit preflight scaling sweep; excluded from the fast Unit alias. */
TEST(CPUHalfOperands, EveryPositiveWorkerBudget){
    const int maximum=physicalWorkerCount();
    ASSERT_GT(maximum,0);
    for(int workers=1;workers<=maximum;++workers){proveLinear(3,65,workers);proveRouter(3,workers);proveBF16Router(3,65,workers);}
}
}
