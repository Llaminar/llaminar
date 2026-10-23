/**
 * @file ROCmFloatingReductionRegression.h
 * @brief Literal historical native arithmetic versus production captured floating reductions.
 *
 * Test-only old kernels retain original FMA expressions, lane ownership and full
 * LDS barriers. No production dispatch may call these diagnostic oracles.
 */
#pragma once
#include "ROCmFloatingRegressionSupport.h"
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
/** @brief Borrow a small immutable authored source list only during synchronous setup. */
struct NativeSourceOverrides {
    std::span<const NativeSourceWord> words{}; ///< No owner retained by captured execution.
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
