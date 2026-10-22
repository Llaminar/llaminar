/**
 * @file ROCmFloatingReductionRegression.h
 * @brief Literal historical native arithmetic versus production captured floating reductions.
 *
 * Test-only old kernels retain original FMA expressions, lane ownership and full
 * LDS barriers. No production dispatch may call these diagnostic oracles.
 */
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "kernels/common/FloatingPointVerifierLaunch.h"
#include "kernels/common/FloatingProjectionRows.h"
#include <gtest/gtest.h>
#include <algorithm>
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
/** @brief Turn every native infrastructure failure into a visible test failure. */
inline void checked(hipError_t status) {
    if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));
}
/** @brief Setup-owned device allocation; destroyed after graph and terminal observation. */
template<class T> struct Allocation {
    T* p=nullptr;
    explicit Allocation(size_t count){checked(hipMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)));}
    ~Allocation(){if(p)(void)hipFree(p);}
    Allocation(const Allocation&)=delete;
    Allocation& operator=(const Allocation&)=delete;
};
/** @brief Retain one captured executable and its exact explicit stream. */
struct Captured {
    hipStream_t stream=nullptr;
    hipGraph_t graph=nullptr;
    hipGraphExec_t exec=nullptr;
    Captured(){checked(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));}
    ~Captured(){if(stream)(void)hipStreamSynchronize(stream);if(exec)(void)hipGraphExecDestroy(exec);
        if(graph)(void)hipGraphDestroy(graph);if(stream)(void)hipStreamDestroy(stream);}
    template<class F> void capture(F launch,int copies=1){
        checked(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
        for(int i=0;i<copies;++i)launch(stream);
        checked(hipStreamEndCapture(stream,&graph));
        checked(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
};
/** @brief Native event pair isolates a retained batch of kernels without launch-loop timing. */
struct Events {
    hipEvent_t first=nullptr,last=nullptr;
    Events(){checked(hipEventCreate(&first));checked(hipEventCreate(&last));}
    ~Events(){if(last)(void)hipEventDestroy(last);if(first)(void)hipEventDestroy(first);}
    float sample(Captured& graph,int copies){
        checked(hipEventRecord(first,graph.stream));checked(hipGraphLaunch(graph.exec,graph.stream));
        checked(hipEventRecord(last,graph.stream));checked(hipEventSynchronize(last));
        float milliseconds=0;checked(hipEventElapsedTime(&milliseconds,first,last));
        return milliseconds*1000.f/copies;
    }
};
/** @brief Copy setup/test observation bytes only outside captured execution. */
template<class T> void upload(T* target,const std::vector<T>& source,hipStream_t stream){
    checked(hipMemcpyAsync(target,source.data(),source.size()*sizeof(T),hipMemcpyHostToDevice,stream));
}
/** @brief Exercise original production symbols against literal historical native arithmetic. */
inline void runCase(int type,bool fused,int m,int n,int k,int pattern,bool timing=false){
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
            (type?rocmFp32x16_batched_projection(aa.p,bb.p,cc.p,m,n,k,1,type-1,0,stream,&rows):
                rocmFp32_small_n_batched_projection(aa.p,bb.p,cc.p,m,n,k,1,0,stream,&rows));
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
        ASSERT_EQ(std::memcmp(got.data(),gold.data(),outputs*4),0)<<"type="<<type<<" fused="<<fused<<" M/N/K="<<m<<'/'<<n<<'/'<<k<<" pattern="<<pattern<<" replay="<<iteration;
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
