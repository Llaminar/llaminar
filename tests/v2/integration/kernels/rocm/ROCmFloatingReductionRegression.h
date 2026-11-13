/**
 * @file ROCmFloatingReductionRegression.h
 * @brief Literal historical native arithmetic versus production captured floating reductions.
 *
 * Test-only old kernels retain original FMA expressions, lane ownership and full
 * LDS barriers. No production dispatch may call these diagnostic oracles.
 */
#pragma once
#include "ROCmFloatingRegressionSupport.h"
#include <bit>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <string>
#include <type_traits>
#include <cxxabi.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "kernels/common/FloatingPointVerifierLaunch.h"
#include "kernels/common/FloatingProjectionRows.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <span>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
namespace qfn_reduction_regression {
constexpr int SMALL_N_FP32_BLOCK=256;
constexpr int SMALL_PROJECTION_WEIGHT_BF16=1;
constexpr int SMALL_SWIGLU_DOWN_WEIGHT_FP32=0,SMALL_SWIGLU_DOWN_WEIGHT_BF16=2;

/** @brief Frozen pre-optimization native oracle body; preserve arithmetic expression order. */
    __device__ __forceinline__ float uint32_bits_to_float(uint32_t bits)
    {
        union
        {
            uint32_t u;
            float f;
        } value{bits};
        return value.f;
    }

/** @brief Frozen pre-optimization native oracle body; preserve arithmetic expression order. */
    __device__ __forceinline__ float fp16_bits_to_float(uint16_t h)
    {
        const uint32_t native = __float_as_uint(__half2float(__ushort_as_half(h)));
        const uint32_t original_nan = (uint32_t(h & 0x8000u) << 16) |
            0x7F800000u | (uint32_t(h & 0x03FFu) << 13);
        const uint32_t nan_mask = 0u - uint32_t((h & 0x7FFFu) > 0x7C00u);
        // A bit select retains every NaN payload without divergent EXEC masks.
        return uint32_bits_to_float(native ^ ((native ^ original_nan) & nan_mask));
    }

/** @brief Frozen pre-optimization native oracle body; preserve arithmetic expression order. */
    __device__ __forceinline__ float bf16_bits_to_float(uint16_t bf16)
    {
        return uint32_bits_to_float(static_cast<uint32_t>(bf16) << 16);
    }

/** @brief Frozen pre-optimization native oracle body; preserve arithmetic expression order. */
    __device__ __forceinline__ float load_swiglu_down_weight(
        const void *__restrict__ weights,
        size_t index,
        int weight_dtype)
    {
        if (weight_dtype == SMALL_SWIGLU_DOWN_WEIGHT_FP32)
            return reinterpret_cast<const float *>(weights)[index];

        const uint16_t bits = reinterpret_cast<const uint16_t *>(weights)[index];
        return (weight_dtype == SMALL_SWIGLU_DOWN_WEIGHT_BF16)
                   ? bf16_bits_to_float(bits)
                   : fp16_bits_to_float(bits);
    }

/** @brief Frozen pre-optimization native oracle body; preserve arithmetic expression order. */
    __global__ __launch_bounds__(SMALL_N_FP32_BLOCK)
        void legacy_fp32_small_n_batched_projection_kernel(
            const float *const *__restrict__ d_A_array,
            const float *const *__restrict__ d_B_array,
            float *const *__restrict__ d_C_array,
            int M,
            int N,
            int K,
            llaminar2::FloatingProjectionRows row_range)
    {
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        const int batch = static_cast<int>(blockIdx.z);
        if (!row_range.rowIsActive(m) || n >= N)
            return;

        const float *A = d_A_array[batch];
        const float *B = d_B_array[batch];
        float *C = d_C_array[batch];
        if (!A || !B || !C)
            return;

        float sum = 0.0f;
        for (int k = static_cast<int>(threadIdx.x); k < K; k += SMALL_N_FP32_BLOCK)
        {
            sum += A[static_cast<size_t>(m) * K + k] *
                   B[static_cast<size_t>(n) * K + k];
        }

        __shared__ float scratch[SMALL_N_FP32_BLOCK];
        scratch[threadIdx.x] = sum;
        __syncthreads();

        for (int stride = SMALL_N_FP32_BLOCK / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            C[static_cast<size_t>(m) * N + n] = scratch[0];
    }

/** @brief Frozen pre-optimization native oracle body; preserve arithmetic expression order. */
    __global__ __launch_bounds__(SMALL_N_FP32_BLOCK)
        void legacy_fp32x16_batched_projection_kernel(
            const float *const *__restrict__ d_A_array,
            const float *const *__restrict__ d_B_array,
            float *const *__restrict__ d_C_array,
            int M,
            int N,
            int K,
            int weight_dtype,
            llaminar2::FloatingProjectionRows row_range)
    {
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        const int batch = static_cast<int>(blockIdx.z);
        if (!row_range.rowIsActive(m) || n >= N)
            return;

        const float *A = d_A_array[batch];
        const uint16_t *B = reinterpret_cast<const uint16_t *>(d_B_array[batch]);
        float *C = d_C_array[batch];
        if (!A || !B || !C)
            return;

        float sum = 0.0f;
        for (int k = static_cast<int>(threadIdx.x); k < K; k += SMALL_N_FP32_BLOCK)
        {
            const uint16_t weight_bits =
                B[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)];
            const float weight =
                (weight_dtype == SMALL_PROJECTION_WEIGHT_BF16)
                    ? bf16_bits_to_float(weight_bits)
                    : fp16_bits_to_float(weight_bits);
            const size_t activation_index =
                static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k);
            // Preserve the original FP32 products, strided lane ownership
            // and fixed 256-lane reduction tree.
            const float activation = A[activation_index];
            sum += activation * weight;
        }

        __shared__ float scratch[SMALL_N_FP32_BLOCK];
        scratch[threadIdx.x] = sum;
        __syncthreads();

        for (int stride = SMALL_N_FP32_BLOCK / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            C[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = scratch[0];
    }

/** @brief Frozen pre-optimization native oracle body; preserve arithmetic expression order. */
    __global__ __launch_bounds__(SMALL_N_FP32_BLOCK)
        void legacy_floating_swiglu_down_projection_kernel(
            const float *__restrict__ gate,
            const float *__restrict__ up,
            const void *__restrict__ weights,
            float *__restrict__ output,
            int M,
            int N,
            int K,
            int weight_dtype,
            llaminar2::FloatingProjectionRows row_range)
    {
        const int n = static_cast<int>(blockIdx.x);
        const int m = static_cast<int>(blockIdx.y);
        if (!row_range.rowIsActive(m) || n >= N)
            return;

        float sum = 0.0f;
        const size_t row_base = static_cast<size_t>(m) * static_cast<size_t>(K);
        const size_t weight_base = static_cast<size_t>(n) * static_cast<size_t>(K);
        for (int k = static_cast<int>(threadIdx.x); k < K; k += SMALL_N_FP32_BLOCK)
        {
            const float gate_value = gate[row_base + static_cast<size_t>(k)];
            const float up_value = up[row_base + static_cast<size_t>(k)];
            const float swiglu_value = (gate_value / (1.0f + expf(-gate_value))) * up_value;
            const float weight_value = load_swiglu_down_weight(
                weights,
                weight_base + static_cast<size_t>(k),
                weight_dtype);
            sum += swiglu_value * weight_value;
        }

        __shared__ float scratch[SMALL_N_FP32_BLOCK];
        scratch[threadIdx.x] = sum;
        __syncthreads();

        for (int stride = SMALL_N_FP32_BLOCK / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            output[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = scratch[0];
    }
/** @brief One authored original native weight word; storage width follows source type. */
struct NativeSourceWord {
    size_t index; ///< Original flat source coordinate, never a byte offset.
    uint32_t bits; ///< Exact FP32 word or zero-extended FP16/BF16 word.
};
/** @brief Setup-only backgrounds independent of either native arithmetic implementation. */
enum class SourceBackground {
    MixedFinite, ///< Original random full-mantissa source data.
    OrderedDotOne ///< A=4096, up=1, B=0; authored words make the old serial dot exactly one.
};
/** @brief Borrow authored native weights and input background during synchronous setup. */
struct NativeSourceOverrides {
    std::span<const NativeSourceWord> words{}; ///< No owner retained by captured execution.
    SourceBackground background=SourceBackground::MixedFinite; ///< Never a captured runtime policy.
};
/** @brief Exercise original production symbols against literal historical native arithmetic. */
inline void runCase(int type,bool fused,int m,int n,int k,int pattern,bool timing=false,int rawWord=-1,
    NativeSourceOverrides overrides={}){
    const size_t elements=size_t(m)*k,weights=size_t(n)*k,outputs=size_t(m)*n;
    Allocation<float> input(elements),up(elements),weight(weights),actual(outputs),expected(outputs);
    Allocation<uint16_t> narrow(weights);Allocation<int32_t> count(1);
    Allocation<const float*> aa(1),bb(1);Allocation<float*> cc(1),oldCC(1);
    std::vector<float> a(elements),u(elements),w(weights),dirty(outputs,-987.25f),got(outputs),gold(outputs);
    // Stable integer mixing fills all mantissa bits and varies exponents;
    // opposing signs create rounding-sensitive cancellation within each tree.
    const auto sample=[](size_t index,uint32_t seed){
        uint32_t word=uint32_t(index)+seed;word^=word>>16;word*=0x7feb352du;
        word^=word>>15;word*=0x846ca68bu;word^=word>>16;
        const uint32_t bits=(word&0x807fffffu)|((123u+(word>>24)%9u)<<23);
        float value;std::memcpy(&value,&bits,4);return value;
    };
    for(size_t i=0;i<elements;++i){a[i]=sample(i,0x159a73du);u[i]=sample(i,0x792a45bu);}
    for(size_t i=0;i<weights;++i)w[i]=sample(i,0x24ba8d3u);
    if(overrides.background==SourceBackground::OrderedDotOne){
        // Exact powers of two construct products +2^24,-2^24,+1 without
        // implementing a second reduction or dot-product oracle on the host.
        std::fill(a.begin(),a.end(),0x1p12f);std::fill(u.begin(),u.end(),1.f);
        std::fill(w.begin(),w.end(),0.f);
    }
    // Cases separate cancellation, signed zero, finite extremes and canonical
    // nonfinites so a NaN case does not conceal the finite/subnormal witnesses.
    if(pattern==1){for(size_t i=0;i<elements;++i)a[i]=(i&1)?-0.f:0.f;}
    if(pattern==2){for(size_t i=0;i<elements;++i)a[i]=(i&1)?-0x1p-140f:0x1p-140f;}
    if(pattern==3){for(size_t i=0;i<elements;++i)a[i]=(i%3)?0x1p100f:-0x1p100f;}
    if(pattern==4){a[0]=std::numeric_limits<float>::infinity();if(elements>1)a[1]=-a[0];}
    if(pattern==5){a[0]=std::numeric_limits<float>::quiet_NaN();}
    if(pattern==6){w[0]=std::numeric_limits<float>::quiet_NaN();}
    std::vector<uint16_t> bits(weights);
    for(size_t i=0;i<weights;++i){
        if(type==1)bits[i]=__half_as_ushort(__float2half(w[i]));
        else {uint32_t word;std::memcpy(&word,&w[i],4);bits[i]=uint16_t(word>>16);}
    }
    // Author one native source word directly. Other columns retain finite,
    // rounding-sensitive data, and each special word gets its own replay case:
    // NaNs can never hide finite/subnormal checks in another invocation.
    if(rawWord>=0){
        if(type==0)throw std::invalid_argument("Raw16 source case requires FP16 or BF16");
        bits[0]=static_cast<uint16_t>(rawWord);
        // K1 witnesses keep tiny source values observable: power-of-two scaling
        // maps even the minimum BF16 subnormal into a normal FP32 product.
        if(k==1)std::fill(a.begin(),a.end(),0x1p24f);
    }
    // Consume borrowed authored words synchronously into fixture-owned host
    // banks before any DMA. Neither span nor caller storage enters graph identity.
    for(const auto& source:overrides.words){
        if(source.index>=weights || (type!=0 && source.bits>0xffffu))
            throw std::invalid_argument("Native source override exceeds original word geometry");
        if(type==0)std::memcpy(&w[source.index],&source.bits,sizeof(source.bits));
        else bits[source.index]=static_cast<uint16_t>(source.bits);
    }
    // Every borrowed async host source precedes stream owners, including
    // loop-reused poison, count and pointer publications. Exception teardown
    // drains both streams before any such source or device storage is retired.
    std::vector<float> poisoned=a;
    const float* ap=input.p;const float* bp=type?reinterpret_cast<const float*>(narrow.p):weight.p;
    float* cp=actual.p;float* oldOutput=expected.p;int live=m;
    Captured candidate,oracle;
    upload(input.p,a,candidate.stream);upload(up.p,u,candidate.stream);upload(weight.p,w,candidate.stream);
    upload(narrow.p,bits,candidate.stream);upload(actual.p,dirty,candidate.stream);upload(expected.p,dirty,candidate.stream);
    checked(hipMemcpyAsync(aa.p,&ap,sizeof(ap),hipMemcpyHostToDevice,candidate.stream));
    checked(hipMemcpyAsync(bb.p,&bp,sizeof(bp),hipMemcpyHostToDevice,candidate.stream));
    checked(hipMemcpyAsync(cc.p,&cp,sizeof(cp),hipMemcpyHostToDevice,candidate.stream));
    checked(hipMemcpyAsync(count.p,&live,4,hipMemcpyHostToDevice,candidate.stream));
    checked(hipStreamSynchronize(candidate.stream));
    const auto rows=llaminar2::DeviceRowRange::deviceCounted(m,count.p);
    const auto selected=llaminar2::FloatingProjectionRows::bind(m,&rows,nullptr);
    auto launchCandidate=[&](hipStream_t stream){
        bool ok=fused?rocmFloating_swiglu_down_projection(input.p,up.p,bp,actual.p,m,n,k,type,0,stream,&rows):
            (type?rocmFp32x16_batched_projection(
                llaminar2::FixedFP32x16ProjectionGroup::single(input.p,narrow.p,actual.p),m,n,k,type-1,0,stream,&rows):
                rocmFp32_small_n_batched_projection(
                    llaminar2::FixedFP32ProjectionGroup::single(input.p,weight.p,actual.p),m,n,k,0,stream,&rows));
        if(!ok)throw std::runtime_error("Production floating projection rejected regression launch");
    };
    // Literal old kernel bodies below are compiled in this HIP target with its
    // production FMA settings. Oracle storage is independent of candidate writes.

    checked(hipMemcpyAsync(oldCC.p,&oldOutput,sizeof(oldOutput),hipMemcpyHostToDevice,oracle.stream));
    checked(hipStreamSynchronize(oracle.stream));
    auto launchOracle=[&](hipStream_t stream){
        if(fused)legacy_floating_swiglu_down_projection_kernel<<<dim3(n,m),256,0,stream>>>(input.p,up.p,bp,expected.p,m,n,k,type,selected);
        else if(type)legacy_fp32x16_batched_projection_kernel<<<dim3(n,m),256,0,stream>>>(aa.p,bb.p,oldCC.p,m,n,k,type-1,selected);
        else legacy_fp32_small_n_batched_projection_kernel<<<dim3(n,m),256,0,stream>>>(aa.p,bb.p,oldCC.p,m,n,k,selected);
        checked(hipGetLastError());
    };
    const int copies=timing?256:1;
    candidate.capture(launchCandidate,copies);oracle.capture(launchOracle,copies);
    if(timing && std::getenv("QFN_REDUCTION_PROFILE_SIDE")){
        const std::string side=std::getenv("QFN_REDUCTION_PROFILE_SIDE");
        if(side!="candidate" && side!="oracle")throw std::invalid_argument("Profile side must be candidate or oracle");
        // Separate profiler invocation: only the selected arithmetic family
        // executes. No correctness oracle or unrelated timing family runs.
        Captured& chosen=side=="candidate"?candidate:oracle;Events event;
        for(int repeat=0;repeat<12;++repeat)
            std::cout<<"REDUCTION_SINGLE,"<<side<<','<<type<<','<<fused<<','<<m<<','<<n<<','<<k<<','<<event.sample(chosen,copies)<<'\n';
        return;
    }
    for(int iteration=0;iteration<20;++iteration){
        live=iteration%4==0?m:iteration%4==1?std::max(0,m-1):iteration%4==2?0:std::min(1,m);
        poisoned=a;for(int row=live;row<m;++row)std::fill_n(poisoned.data()+size_t(row)*k,k,std::numeric_limits<float>::quiet_NaN());
        upload(input.p,poisoned,candidate.stream);
        checked(hipMemcpyAsync(count.p,&live,4,hipMemcpyHostToDevice,candidate.stream));
        checked(hipStreamSynchronize(candidate.stream));
        // Both independent banks receive the same changing dirty sentinel. No
        // inactive output is accidentally certified by a previous live result.
        std::fill(dirty.begin(),dirty.end(),-987.25f-iteration);
        upload(actual.p,dirty,candidate.stream);upload(expected.p,dirty,candidate.stream);
        checked(hipStreamSynchronize(candidate.stream));
        checked(hipGraphLaunch(candidate.exec,candidate.stream));checked(hipStreamSynchronize(candidate.stream));
        checked(hipGraphLaunch(oracle.exec,oracle.stream));checked(hipStreamSynchronize(oracle.stream));
        checked(hipMemcpyAsync(got.data(),actual.p,outputs*4,hipMemcpyDeviceToHost,candidate.stream));
        checked(hipMemcpyAsync(gold.data(),expected.p,outputs*4,hipMemcpyDeviceToHost,oracle.stream));
        checked(hipStreamSynchronize(candidate.stream));checked(hipStreamSynchronize(oracle.stream));
        ASSERT_EQ(std::memcmp(got.data(),gold.data(),outputs*4),0)<<"type="<<type<<" fused="<<fused<<" M/N/K="<<m<<'/'<<n<<'/'<<k<<" pattern="<<pattern<<" raw16="<<rawWord<<" replay="<<iteration;
        // Large and opposite products cancel in the original lane's ordered
        // dot before +1 arrives. Two accidentally zero outputs cannot pass.
        if(live>0 && overrides.background==SourceBackground::OrderedDotOne)
            ASSERT_EQ(gold[0],1.f);
        // Tiny-source witnesses must remain observable in the native oracle;
        // matching zero outputs would not prove preservation of their bits.
        if(rawWord>=0 && k==1 && live>0)ASSERT_NE(got[0],0.0f);
        for(size_t i=size_t(live)*n;i<outputs;++i)ASSERT_EQ(std::memcmp(&got[i],&dirty[i],4),0);
    }
    if(timing){
        live=m;upload(input.p,a,candidate.stream);checked(hipMemcpyAsync(count.p,&live,4,hipMemcpyHostToDevice,candidate.stream));
        checked(hipStreamSynchronize(candidate.stream));Events events;
        for(int warmup=0;warmup<5;++warmup){(void)events.sample(candidate,copies);(void)events.sample(oracle,copies);}
        for(int pair=0;pair<12;++pair){
            float current,old;
            if(pair&1){old=events.sample(oracle,copies);current=events.sample(candidate,copies);}
            else {current=events.sample(candidate,copies);old=events.sample(oracle,copies);}
            std::cout<<"REDUCTION_PAIR,"<<type<<','<<fused<<','<<m<<','<<n<<','<<k<<','<<pair<<','<<current<<','<<old<<'\n';
        }
    }
}
/** @brief Deliberately wrong adjacent tree proves this witness detects reassociation. */
__global__ void associationWitness(float* output){
    const unsigned lane=threadIdx.x;__shared__ float part[256];
    part[lane]=lane==0?0x1p24f:lane==1?1.f:lane==128?-0x1p24f:0.f;
    __syncthreads();
    if(blockIdx.x==0){
        for(unsigned width=128;width;width>>=1){if(lane<width)part[lane]=__fadd_rn(part[lane],part[lane+width]);__syncthreads();}
    }else{
        for(unsigned width=1;width<256;width<<=1){if(lane%(2*width)==0)part[lane]=__fadd_rn(part[lane],part[lane+width]);__syncthreads();}
    }
    if(lane==0)output[blockIdx.x]=part[0];
}
/** @brief Assert old-tree one differs bitwise from the deliberately reassociated zero. */
inline void proveAssociationSensitivity(){
    Allocation<float> device(2);std::vector<float> output(2);Captured graph;
    graph.capture([&](hipStream_t stream){associationWitness<<<2,256,0,stream>>>(device.p);checked(hipGetLastError());});
    checked(hipGraphLaunch(graph.exec,graph.stream));
    checked(hipMemcpyAsync(output.data(),device.p,8,hipMemcpyDeviceToHost,graph.stream));
    checked(hipStreamSynchronize(graph.stream));
    ASSERT_EQ(output[0],1.f);ASSERT_EQ(output[1],0.f);
    ASSERT_NE(std::memcmp(&output[0],&output[1],4),0);
}
} // namespace qfn_reduction_regression
/** @brief Independent old-native byte proof including dirty counted empty/partial capture replay. */
TEST(ROCmFloatingReduction,HistoricalTreeCapturedByteExact){
    using namespace qfn_reduction_regression;checked(hipSetDevice(0));proveAssociationSensitivity();
    for(bool fused:{false,true})for(int type:{0,1,2})for(int m:{1,3,17})
        for(auto shape:{std::pair{4,1},std::pair{33,63},std::pair{4,64},std::pair{33,65},std::pair{4,127},
                       std::pair{33,128},std::pair{4,129},std::pair{33,255},std::pair{4,256},
                       std::pair{33,257},std::pair{4,10240},std::pair{513,257}})
            runCase(type,fused,m,shape.first,shape.second,0);
    for(bool fused:{false,true})for(int type:{0,1,2})for(int pattern=1;pattern<=6;++pattern)
        runCase(type,fused,3,4,257,pattern);
    // Two distinct quiet payloads exercise the original cross-wave operands.
    // Exact old-native comparison is required; payload canonicalization on a
    // backend is permitted and must not be reported as a sensitivity witness.
    for(bool fused:{false,true})for(int type:{0,1,2}){
        const uint32_t first=type==0?0x7fc12345u:type==1?0x7e11u:0x7fc1u;
        const uint32_t second=type==0?0x7fc54321u:type==1?0x7e22u:0x7fe2u;
        const uint32_t sign=type==0?0x80000000u:0x8000u;
        for(auto edge:{std::pair{0,128},std::pair{64,192},std::pair{0,64}})
            for(unsigned signs=0;signs<4;++signs)for(bool reverse:{false,true}){
                const uint32_t left=(reverse?second:first)|((signs&1u)?sign:0u);
                const uint32_t right=(reverse?first:second)|((signs&2u)?sign:0u);
                const std::array<NativeSourceWord,2> words{{
                    {static_cast<size_t>(edge.first),left},{static_cast<size_t>(edge.second),right}}};
                SCOPED_TRACE(::testing::Message()<<"native pair type="<<type<<" fused="<<fused
                    <<" edge="<<edge.first<<'/'<<edge.second<<" signs="<<signs<<" reverse="<<reverse);
                runCase(type,fused,1,1,256,0,false,-1,NativeSourceOverrides{words});
            }
    }
    // Independent native16-bit finite/zero/subnormal/infinity and NaN words.
    // Signaling/quiet payloads are not routed through a host float conversion.
    for(bool fused:{false,true})for(int type:{1,2}){
        const std::vector<uint16_t> words=type==1 ?
            std::vector<uint16_t>{0x0000,0x8000,0x0001,0x8001,0x03ff,0x83ff,
                0x0400,0x8400,0x3c01,0xbc01,0x7bff,0xfbff,0x7c00,0xfc00,
                0x7c01,0xfc01,0x7d55,0xfd55,0x7e00,0xfe00,0x7fff,0xffff} :
            std::vector<uint16_t>{0x0000,0x8000,0x0001,0x8001,0x007f,0x807f,
                0x0080,0x8080,0x3f81,0xbf81,0x7f7f,0xff7f,0x7f80,0xff80,
                0x7f81,0xff81,0x7fa5,0xffa5,0x7fc0,0xffc0,0x7fff,0xffff};
        for(uint16_t word:words)runCase(type,fused,3,4,257,0,false,word);
        // Ordinary K1 has no later lane-local product or unrelated nonzero
        // reduction partial to round away subnormal/min-normal source evidence.
        if(!fused)for(size_t word=2;word<8;++word)
            runCase(type,false,3,4,1,0,false,words[word]);
    }
}
/** @test FP32 load groups preserve the original serial accumulator across tiles and scalar tails. */
TEST(ROCmFloatingReduction,FP32LoadAheadOriginalChainCapturedByteExact){
    using namespace qfn_reduction_regression;checked(hipSetDevice(0));proveAssociationSensitivity();
    for(int k:{1023,1024,1025,2047,2048,2049,10240})for(int m:{1,3}){
        // Full-mantissa finite source remains independently byte checked too.
        runCase(0,false,m,4,k,0);
        for(bool negative:{false,true})for(bool last:{false,true}){
            const size_t unit=last?static_cast<size_t>(k-1):512u;
            const uint32_t positive=0x45800000u,opposite=0xc5800000u;
            const std::array<NativeSourceWord,3> words{{
                {unit-512,negative?opposite:positive},
                {unit-256,negative?positive:opposite},{unit,0x39800000u}}};
            SCOPED_TRACE(::testing::Message()<<"FP32 load ahead M="<<m<<" K="<<k
                <<" unit="<<unit<<" negative="<<negative);
            // A=4096 and weights +/-4096,2^-12 yield +/-2^24,+1.
            // All three authored coordinates belong to the SAME stride256
            // lane. The last-coordinate cases cross complete-tile/scalar-tail
            // boundaries at K1025/2049 and reach the final HC iteration.
            runCase(0,false,m,4,k,0,false,-1,
                NativeSourceOverrides{words,SourceBackground::OrderedDotOne});
        }
    }
    // Existing special-value fixtures must actually enter the new 1024 tile.
    for(int k:{1025,2049})for(int pattern=1;pattern<=6;++pattern)
        runCase(0,false,3,4,k,pattern);
    // Reuse the existing payload-pair input recipe at a complete tile and a
    // tile-plus-tail shape. The old K256 recipe never entered the new body.
    for(int k:{1024,1025,2048,2049})for(auto edge:{std::pair{0,128},std::pair{64,192},std::pair{0,64}})
        for(unsigned signs=0;signs<4;++signs)for(bool reverse:{false,true}){
            const uint32_t left=(reverse?0x7fc54321u:0x7fc12345u)|((signs&1u)?0x80000000u:0u);
            const uint32_t right=(reverse?0x7fc12345u:0x7fc54321u)|((signs&2u)?0x80000000u:0u);
            const std::array<NativeSourceWord,2> words{{
                {static_cast<size_t>(edge.first),left},{static_cast<size_t>(edge.second),right}}};
            SCOPED_TRACE(::testing::Message()<<"FP32 tiled payload K="<<k<<" edge="<<edge.first<<'/'<<edge.second
                <<" signs="<<signs<<" reverse="<<reverse);
            runCase(0,false,1,1,k,0,false,-1,NativeSourceOverrides{words});
        }
}
/** @brief Opt-in exact-shape paired native-event timings; excluded from functional gate. */
TEST(ROCmFloatingReduction,DISABLED_ExactShapePairedEvents){
    using namespace qfn_reduction_regression;checked(hipSetDevice(0));
    // Select exactly one family/type/shape per process/profiler invocation.
    const auto read=[](const char* key,int fallback){const char* value=std::getenv(key);return value?std::stoi(value):fallback;};
    const int type=read("QFN_REDUCTION_TYPE",0),fused=read("QFN_REDUCTION_FUSED",0);
    const int m=read("QFN_REDUCTION_M",1),n=read("QFN_REDUCTION_N",4),k=read("QFN_REDUCTION_K",10240);
    ASSERT_GE(type,0);ASSERT_LE(type,2);ASSERT_GE(fused,0);ASSERT_LE(fused,1);
    ASSERT_GT(m,0);ASSERT_GT(n,0);ASSERT_GT(k,0);runCase(type,fused,m,n,k,0,true);
}

/**
 * @brief Scoped native16 query projection screen over the public production bridge.
 *
 * Only test-owned geometry/state selectors are read here. The loaded serving
 * core supplies the one production program under test; the historical full-LDS
 * kernel is an independent byte oracle and never appears in a timed graph.
 */
namespace qfn_reduction_regression::guard_grid_screen {
constexpr int columns = 512;
constexpr int reduction = 2560;
constexpr int guard = 32;
constexpr int threshold = 2051;

/** @brief Reject infrastructure and cold identity failures before retained replay. */
inline void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

/** @brief Parse a test-only integer completely, with no serving-policy selection. */
inline int setting(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    if (!value)
        return fallback;
    std::size_t used = 0;
    const std::string text(value);
    const int parsed = std::stoi(text, &used);
    require(used == text.size(), "Incomplete guard-grid test integer");
    return parsed;
}

/** @brief Drain exact streams before any captured pointer or DMA staging can retire. */
struct Retirement final {
    Captured &production;
    Captured &historical;
    ~Retirement()
    {
        if (hipStreamSynchronize(production.stream) != hipSuccess ||
            hipStreamSynchronize(historical.stream) != hipSuccess)
            std::terminate();
    }
};

/** @brief Capture once, including exception-safe closure of a failed native capture. */
template<class Launch>
inline void capture(Captured &owner, Launch launch, int copies)
{
    checked(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeThreadLocal));
    try
    {
        for (int copy = 0; copy < copies; ++copy)
            launch(owner.stream);
    }
    catch (...)
    {
        hipGraph_t discarded = nullptr;
        const auto status = hipStreamEndCapture(owner.stream, &discarded);
        if (discarded)
            checked(hipGraphDestroy(discarded));
        if (status != hipSuccess && status != hipErrorStreamCaptureInvalidated)
            std::terminate();
        throw;
    }
    checked(hipStreamEndCapture(owner.stream, &owner.graph));
    checked(hipGraphInstantiate(&owner.exec, owner.graph, nullptr, nullptr, 0));
}

/** @brief Read a captured argument without aliasing its typed host storage. */
template<class T>
inline T argument(const hipKernelNodeParams &node, int index)
{
    static_assert(std::is_trivially_copyable_v<T>);
    require(node.kernelParams && node.kernelParams[index], "Missing captured projection argument");
    // The complete tuple is checked by callers; its descriptor is not default constructible.
    std::array<std::byte, sizeof(T)> bytes;
    std::memcpy(bytes.data(), node.kernelParams[index], sizeof(T));
    return std::bit_cast<T>(bytes);
}

/** @brief Authenticate exact source format, all five arguments and a single retained chain. */
inline void inspect(Captured &owner, int copies, int expectedColumns, int dtype,
                    int physical, const float *input, const uint16_t *weight, float *output,
                    const llaminar2::FloatingProjectionRows &rows,
                    const llaminar2::ProjectionRowWork &work)
{
    std::size_t count = 0;
    checked(hipGraphGetNodes(owner.graph, nullptr, &count));
    require(count == std::size_t(copies), "Projection capture has extra or missing nodes");
    std::vector<hipGraphNode_t> nodes(count);
    checked(hipGraphGetNodes(owner.graph, nodes.data(), &count));
    std::size_t edges = 0;
    checked(hipGraphGetEdges(owner.graph, nullptr, nullptr, &edges));
    require(edges == std::size_t(copies - 1), "Projection capture is not one complete chain");
    std::vector<hipGraphNode_t> from(edges), to(edges);
    if (edges)
        checked(hipGraphGetEdges(owner.graph, from.data(), to.data(), &edges));
    std::vector<int> incoming(count), outgoing(count);
    for (std::size_t edge = 0; edge < edges; ++edge)
    {
        const auto first = std::find(nodes.begin(), nodes.end(), from[edge]);
        const auto last = std::find(nodes.begin(), nodes.end(), to[edge]);
        require(first != nodes.end() && last != nodes.end() && first != last,
                "Projection edge references an alien/self node");
        require(++outgoing[first - nodes.begin()] == 1 && ++incoming[last - nodes.begin()] == 1,
                "Projection capture branches or merges");
    }
    require(std::count(incoming.begin(), incoming.end(), 0) == 1 &&
            std::count(outgoing.begin(), outgoing.end(), 0) == 1,
            "Projection capture has multiple roots or terminals");
    auto cursor = std::size_t(std::find(incoming.begin(), incoming.end(), 0) - incoming.begin());
    std::vector<bool> visited(count);
    void *function = nullptr;
    std::string raw;
    hipFuncAttributes attributes{};
    for (std::size_t ordinal = 0; ordinal < count; ++ordinal)
    {
        require(!visited[cursor], "Projection capture contains a cycle");
        visited[cursor] = true;
        hipGraphNodeType kind;
        checked(hipGraphNodeGetType(nodes[cursor], &kind));
        require(kind == hipGraphNodeTypeKernel, "Non-kernel work entered timed projection graph");
        hipKernelNodeParams node{};
        checked(hipGraphKernelNodeGetParams(nodes[cursor], &node));
        require(node.gridDim.x == unsigned(expectedColumns) && node.gridDim.y == unsigned(physical) &&
                node.gridDim.z == 1 && node.blockDim.x == 256 && node.blockDim.y == 1 &&
                node.blockDim.z == 1 && node.sharedMemBytes == 0 && node.extra == nullptr,
                "Captured projection grid/block/dynamic LDS differs from declared side");
        const auto group = argument<llaminar2::FixedFP32x16ProjectionGroup>(node, 0);
        require(group.size() == 1 && group.input(0) == input && group.weight(0) == weight &&
                group.output(0) == output && argument<int>(node, 1) == physical &&
                argument<int>(node, 2) == columns && argument<int>(node, 3) == reduction,
                "Captured production operand/shape tuple differs");
        const auto recorded = argument<llaminar2::FloatingProjectionRows>(node, 4);
        for (int row = 0; row < physical; ++row)
            for (int live : {0, work.logicalRowsPerRequest()})
                for (int frontier : {threshold - 1, threshold})
                    require(recorded.rowAdmissionFor(row, live, frontier) ==
                            rows.rowAdmissionFor(row, live, frontier), "Captured causal admission differs");
        require(recorded.captureIdentity() == work.captureIdentity(),
                "Captured complete row publication/geometry identity differs");
        require(node.func && (!function || function == node.func), "Projection copies change native function");
        function = node.func;
        if (raw.empty())
        {
            const char *name = hipKernelNameRefByPtr(function, owner.stream);
            require(name && *name, "Missing actual native projection name");
            raw = name;
            int status = -1;
            char *decoded = abi::__cxa_demangle(raw.c_str(), nullptr, nullptr, &status);
            const std::string readable = decoded ? decoded : "";
            std::free(decoded);
            const bool scopedColumns = expectedColumns != columns;
            const std::string expected = "void (anonymous namespace)::" +
                std::string(scopedColumns ? "fp32x16_scoped_projection_columns_kernel<" :
                                             "fp32x16_batched_projection_kernel<") +
                "((anonymous namespace)::SmallProjectionWeightDtype)" + std::to_string(dtype) +
                (scopedColumns ? ">(" : ", llaminar2::FloatingProjectionRows>(") +
                "llaminar2::FixedFloatingProjectionGroup<unsigned short>, "
                "int, int, int, llaminar2::FloatingProjectionRows)";
            require(status == 0 && readable == expected, "Foreign native projection/template/argument signature");
            checked(hipFuncGetAttributes(&attributes, function));
            require(attributes.localSizeBytes == 0 && attributes.sharedSizeBytes == 1024 &&
                    attributes.numRegs > 0 && attributes.numRegs <= 256 &&
                    attributes.maxThreadsPerBlock >= 256,
                    "Projection native resources or thread admission differ");
        }
        if (ordinal + 1 < count)
        {
            const auto edge = std::find(from.begin(), from.end(), nodes[cursor]);
            require(edge != from.end(), "Projection chain terminates early");
            cursor = std::size_t(std::find(nodes.begin(), nodes.end(), to[edge - from.begin()]) - nodes.begin());
        }
    }
    require(std::all_of(visited.begin(), visited.end(), [](bool value) { return value; }),
            "Projection capture has unreachable nodes");
    const auto id = work.captureIdentity();
    std::cout << "GUARD_GRID_CAPTURE dtype=" << dtype << " M=" << work.logicalRowsPerRequest()
              << " requests=" << work.requests() << " physical=" << physical
              << " N=" << columns << " K=" << reduction << " copies=" << copies
              << " nodes=" << count << " edges=" << edges << " grid=" << expectedColumns << 'x'
              << physical << "x1 block=256x1x1 function=" << function << " raw_symbol=" << raw
              << " registers=" << attributes.numRegs << " static_lds=" << attributes.sharedSizeBytes
              << " dynamic_lds=0 private=" << attributes.localSizeBytes
              << " input=" << static_cast<const void *>(input) << " weight=" << static_cast<const void *>(weight)
              << " output=" << static_cast<void *>(output) << " frontier=" << id.frontiers
              << " live=" << id.counts << " stride=" << id.publication_stride_words
              << " source_capacity=" << id.source_capacity << " threshold=" << id.first_required_logical_row
              << " args=5 full_chain=1 passed=1\n";
}

/**
 * @brief One production graph and one untimed historical oracle with original publication owners.
 * @param dtype Native FP16=0 or BF16=1 source words.
 * @param m Original rows per request,1 or16 in the initial bounded screen.
 * @param requests Original request count,1 or3; physical strides are never compacted.
 * @param copies One operation or32 identical out-of-place operations per timed graph.
 * @param timing Whether to collect native events after all four dirty states pass.
 * @param timedState Full=0,partial=1,empty=2,mixed=3; setup state is outside events.
 */
inline void run(int dtype, int m, int requests, int copies, bool timing, int timedState)
{
    require((dtype == 0 || dtype == 1) && (m == 1 || m == 16) &&
            (requests == 1 || requests == 3) && (copies == 1 || copies == 32) &&
            timedState >= 0 && timedState < 4, "Unsupported initial guard-grid screen geometry");
    const int expectedColumns = setting("QFN_GUARD_GRID_EXPECTED_COLUMNS", columns);
    require(expectedColumns == columns || expectedColumns == 64 || expectedColumns == 128 ||
            expectedColumns == 256, "Expected columns must declare old512 or an explicit candidate side");
    const int physical = m * requests;
    const std::size_t inputCount = std::size_t(physical) * reduction;
    const std::size_t weightCount = std::size_t(columns) * reduction;
    const std::size_t outputCount = std::size_t(physical) * columns;
    Allocation<float> input(inputCount + 2 * guard), output(outputCount + 2 * guard),
        reference(outputCount + 2 * guard);
    Allocation<uint16_t> weights(weightCount + 2 * guard);
    Allocation<int32_t> publication(2 * requests);
    Allocation<const float *> aa(1), bb(1);
    Allocation<float *> cc(1);
    std::vector<float> a(inputCount + 2 * guard, -93.25f), poison(outputCount + 2 * guard),
        observed(outputCount + 2 * guard), gold(outputCount + 2 * guard),
        sourceObserved(inputCount + 2 * guard);
    std::vector<uint16_t> b(weightCount + 2 * guard, 0xa55a), weightObserved(b.size());
    std::vector<int32_t> words(2 * requests), published(words.size());
    const float *ap = input.p + guard;
    const float *bp = reinterpret_cast<const float *>(weights.p + guard);
    float *cp = reference.p + guard;
    const auto finite = [](std::size_t index, uint32_t seed) {
        uint32_t word = uint32_t(index) + seed;
        word ^= word >> 16;
        word *= 0x7feb352du;
        word ^= word >> 15;
        word *= 0x846ca68bu;
        word ^= word >> 16;
        const uint32_t bits = (word & 0x807fffffu) | ((123u + (word >> 24) % 7u) << 23);
        return std::bit_cast<float>(bits);
    };
    for (std::size_t i = 0; i < inputCount; ++i)
        a[guard + i] = i % 257 == 0 ? -0.0f : finite(i, 0x159a73du);
    for (std::size_t i = 0; i < weightCount; ++i)
    {
        const float value = finite(i, 0x24ba8d3u);
        if (dtype == 0)
            b[guard + i] = __half_as_ushort(__float2half(value));
        else
        {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            b[guard + i] = uint16_t(bits >> 16);
        }
    }
    // Setup and each later publication are joined by the exact producer event;
    // neither graph embeds a host copy or reads a mutable host count.
    hipEvent_t ready = nullptr;
    checked(hipEventCreateWithFlags(&ready, hipEventDisableTiming));
    struct EventOwner {
        hipEvent_t value;
        ~EventOwner() { if (hipEventDestroy(value) != hipSuccess) std::terminate(); }
    } readyOwner{ready};
    Captured production, historical;
    Retirement retirement{production, historical};
    upload(input.p, a, production.stream);
    upload(weights.p, b, production.stream);
    upload(publication.p, words, production.stream);
    checked(hipMemcpyAsync(aa.p, &ap, sizeof(ap), hipMemcpyHostToDevice, production.stream));
    checked(hipMemcpyAsync(bb.p, &bp, sizeof(bp), hipMemcpyHostToDevice, production.stream));
    checked(hipMemcpyAsync(cc.p, &cp, sizeof(cp), hipMemcpyHostToDevice, production.stream));
    const auto work = llaminar2::ProjectionRowWork::causalStrided(
        requests, m, publication.p, publication.p + 1, 2, 4096, threshold);
    llaminar2::ProjectionRowWorkScope scope(work);
    const auto rows = llaminar2::FloatingProjectionRows::bind(physical, nullptr, nullptr);
    // Finish setup transfers before either cold capture begins. Later updates
    // retain the same owners/addresses and order through this exact ready event.
    checked(hipEventRecord(ready, production.stream));
    checked(hipEventSynchronize(ready));
    capture(production, [&](hipStream_t stream) {
        require(rocmFp32x16_batched_projection(
                    llaminar2::FixedFP32x16ProjectionGroup::single(input.p + guard, weights.p + guard,
                                                                  output.p + guard),
                    physical, columns, reduction, dtype, 0, stream),
                "Public production projection bridge rejected scoped screen");
    }, copies);
    capture(historical, [&](hipStream_t stream) {
        legacy_fp32x16_batched_projection_kernel<<<dim3(columns, physical), 256, 0, stream>>>(
            aa.p, bb.p, cc.p, physical, columns, reduction, dtype, rows);
        checked(hipGetLastError());
    }, 1);
    inspect(production, copies, expectedColumns, dtype, physical,
            input.p + guard, weights.p + guard, output.p + guard, rows, work);
    const auto publish = [&](int state, int ordinal) {
        for (int request = 0; request < requests; ++request)
        {
            int frontier = threshold;
            int live = m;
            if (state == 1) { frontier = threshold - std::max(1, m / 2); live = m; }
            if (state == 2) { frontier = 0; live = m; }
            if (state == 3)
            {
                if (request % 3 == 1) live = 0;
                if (request % 3 == 2) frontier = threshold - std::max(1, m / 2);
            }
            words[2 * request] = frontier;
            words[2 * request + 1] = live;
        }
        std::fill(poison.begin(), poison.end(), -701.25f - ordinal);
        upload(publication.p, words, production.stream);
        upload(output.p, poison, production.stream);
        upload(reference.p, poison, production.stream);
        checked(hipEventRecord(ready, production.stream));
        checked(hipStreamWaitEvent(historical.stream, ready, 0));
        checked(hipGraphLaunch(historical.exec, historical.stream));
        checked(hipMemcpyAsync(gold.data(), reference.p, gold.size() * sizeof(float),
                               hipMemcpyDeviceToHost, historical.stream));
        checked(hipStreamSynchronize(historical.stream));
        // Gold is retained before timing, never regenerated by the observed replay.
        require(std::memcmp(gold.data(), poison.data(), guard * sizeof(float)) == 0 &&
                std::memcmp(gold.data() + guard + outputCount,
                            poison.data() + guard + outputCount, guard * sizeof(float)) == 0,
                "Historical output guard was modified");
        for (int row = 0; row < physical; ++row)
        {
            const int request = row / m;
            const auto span = work.resolveFor(request, words[2 * request + 1], words[2 * request]);
            require(span.valid(), "Authored screen publication is invalid");
            if (!span.contains(row))
                require(std::memcmp(gold.data() + guard + std::size_t(row) * columns,
                                    poison.data() + guard + std::size_t(row) * columns,
                                    columns * sizeof(float)) == 0, "Historical inactive output was touched");
        }
    };
    const auto observe = [&] {
        // This observes the ACTUAL preceding replay, including a timed replay;
        // it never launches/restores/re-publishes anything before reading C.
        checked(hipMemcpyAsync(observed.data(), output.p, observed.size() * sizeof(float),
                               hipMemcpyDeviceToHost, production.stream));
        checked(hipMemcpyAsync(sourceObserved.data(), input.p, a.size() * sizeof(float),
                               hipMemcpyDeviceToHost, production.stream));
        checked(hipMemcpyAsync(weightObserved.data(), weights.p, b.size() * sizeof(uint16_t),
                               hipMemcpyDeviceToHost, production.stream));
        checked(hipMemcpyAsync(published.data(), publication.p, words.size() * sizeof(int32_t),
                               hipMemcpyDeviceToHost, production.stream));
        checked(hipStreamSynchronize(production.stream));
        require(std::memcmp(observed.data(), gold.data(), gold.size() * sizeof(float)) == 0,
                "Production output/guards differ from historical original arithmetic");
        require(std::memcmp(sourceObserved.data(), a.data(), a.size() * sizeof(float)) == 0 &&
                std::memcmp(weightObserved.data(), b.data(), b.size() * sizeof(uint16_t)) == 0 &&
                published == words, "Projection modified retained input/weight/publication bytes");
    };
    for (int dirty = 0; dirty < 5; ++dirty)
    {
        publish(dirty % 4, dirty);
        checked(hipGraphLaunch(production.exec, production.stream));
        observe();
    }
    std::cout << "GUARD_GRID_BYTES dtype=" << dtype << " M=" << m << " requests=" << requests
              << " physical=" << physical << " N=" << columns << " K=" << reduction
              << " copies=" << copies << " dirty_states=5 sequence=full_partial_empty_mixed_full"
              << " output=full guards=both source=immutable publication=original passed=1\n";
    if (!timing)
        return;
    Events events;
    publish(timedState, 10);
    for (int warm = 0; warm < 5; ++warm)
        (void)events.sample(production, copies);
    observe();
    for (int sample = 0; sample < 12; ++sample)
    {
        publish(timedState, 20 + sample);
        const float microseconds = events.sample(production, copies);
        observe();
        require(std::isfinite(microseconds) && microseconds > 0, "Invalid native event duration");
        std::cout << std::setprecision(10) << "GUARD_GRID_EVENT sample=" << sample << " dtype=" << dtype
                  << " M=" << m << " requests=" << requests << " physical=" << physical
                  << " N=" << columns << " K=" << reduction << " copies=" << copies
                  << " state=" << timedState << " warmups=5 us=" << microseconds
                  << " unit=us_per_complete_projection actual_timed_bytes=1 oracle_timed=0 passed=1\n";
    }
}
} // namespace qfn_reduction_regression::guard_grid_screen

/** @test Real causal publications preserve historical native16 bytes with immutable captures. */
TEST(ROCmFloatingReduction, ScopedNative16QueryWorkCapturedBytes)
{
    using namespace qfn_reduction_regression;
    checked(hipSetDevice(0));
    for (int dtype : {0, 1})
        for (const auto geometry : {std::pair{1, 1}, std::pair{16, 3}})
            guard_grid_screen::run(dtype, geometry.first, geometry.second, 1, false, 0);
}

/** @test One loaded production core supplies all timed operations; old/new are separate processes. */
TEST(ROCmFloatingReduction, DISABLED_ScopedNative16QueryWorkEvents)
{
    using namespace qfn_reduction_regression;
    checked(hipSetDevice(0));
    guard_grid_screen::run(
        guard_grid_screen::setting("QFN_GUARD_GRID_DTYPE", 0),
        guard_grid_screen::setting("QFN_GUARD_GRID_M", 1),
        guard_grid_screen::setting("QFN_GUARD_GRID_REQUESTS", 1),
        guard_grid_screen::setting("QFN_GUARD_GRID_COPIES", 32), true,
        guard_grid_screen::setting("QFN_GUARD_GRID_STATE", 2));
}
